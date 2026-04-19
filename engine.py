"""
Dance Mashup Engine
-------------------
Core processing: pose estimation, beat detection, transition planning, rendering.
Uses MediaPipe for pose, librosa for beats, OpenCV for frames, ffmpeg for encoding.
"""

import cv2
import numpy as np
# DELOGO_V1 — lazy import guarded; missing module is non-fatal
try:
    import logo_detect as _logo_detect
except Exception as _e:
    _logo_detect = None
    print(f'[warn] logo_detect unavailable: {_e}')
import json
import os
import platform
import re
import subprocess
import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

_SUBPROCESS_EXTRA = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                     if platform.system() == "Windows" else {})

# MediaPipe landmark indices important for dance comparison
# shoulders, elbows, wrists, hips, knees, ankles
DANCE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Distinct colours for the timeline visualisation
VIDEO_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
    '#FF9FF3', '#54A0FF', '#5F27CD', '#01CBC6',
]


@dataclass
class VideoInfo:
    path: str
    filename: str
    fps: float
    width: int
    height: int
    duration: float
    total_frames: int
    poses: Optional[np.ndarray] = None      # (n_samples, 33, 3)
    pose_fps: float = 0.0
    thumbnail: Optional[np.ndarray] = None  # RGB uint8
    source_url: str = ""


@dataclass
class Segment:
    video_idx: int
    start_time: float
    end_time: float
    similarity: float = 0.0
    # Top-K (video_idx, score) alternatives at this time window — populated
    # by plan_transitions, consumed by UI segment-swap.
    candidates: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ────────────────────────────────────────────────────────────────

# PROGRESS_HOOK_V1 — module-level hooks wired by web.py
progress_hook = None   # callable(text: str, pct: Optional[float])  for ticks
phase_hook    = None   # callable(action: str, name: str)  action ∈ {"begin","end"}


def set_progress_hook(fn):
    global progress_hook
    progress_hook = fn
    try:
        import person_tracker as _pt
        _pt.set_progress_hook(fn)
    except Exception:
        pass


def set_phase_hook(fn):
    global phase_hook
    phase_hook = fn


def _pg_tick(text, pct=None):
    try:
        if progress_hook is not None:
            progress_hook(text, pct)
    except Exception:
        pass


def _pg_phase_begin(name):
    try:
        if phase_hook is not None:
            phase_hook("begin", name)
    except Exception:
        pass


def _pg_phase_end(name=""):
    try:
        if phase_hook is not None:
            phase_hook("end", name)
    except Exception:
        pass


class DanceMashupEngine:
    """End-to-end engine for creating dance mashup videos."""

    def __init__(self):
        self.videos: List[VideoInfo] = []
        self.beat_times: Optional[np.ndarray] = None
        self.segments: List[Segment] = []
        self._ffmpeg: Optional[str] = None
        self.external_audio: Optional[str] = None
        self.external_audio_title: str = ""
        self._face_app = None
        self.reference_embedding: Optional[np.ndarray] = None
        # IDENTITY_POOL_V1 — multi-reference + peer-negative pools
        self.reference_embeddings: Optional[np.ndarray] = None
        self.negative_embeddings: Optional[np.ndarray] = None
        self.delogo_enabled = True  # DELOGO_V1
        self._final_cache = {}
        self._interp_cache: dict = {}  # video_idx -> (path, fps)
        self.audio_offsets: dict = {}   # video_idx -> offset_seconds
        self.remote_rife: Optional[dict] = None  # {'host': ..., 'bin': ..., 'model': ..., 'gpu': ..., 'work_dir': ...}
        self.face_visibility: dict = {}  # video_idx -> list of (time, bool)

    # ── helpers ──────────────────────────────────────────────────

    def _get_ffmpeg(self) -> str:
        if self._ffmpeg is None:
            import imageio_ffmpeg
            self._ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        return self._ffmpeg

    def _get_ytdlp_binary(self) -> Optional[str]:
        for candidate in (
            os.path.expanduser('~/DanceMashup/yt-dlp_macos'),
        ):
            if os.path.isfile(candidate):
                return candidate
        return shutil.which('yt-dlp_macos')

    # ── Video management ────────────────────────────────────────

    def add_video(self, path: str) -> VideoInfo:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps if fps > 0 else 0.0

        ok, frame = cap.read()
        thumb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None
        cap.release()

        info = VideoInfo(path=path, filename=os.path.basename(path),
                         fps=fps, width=w, height=h,
                         duration=dur, total_frames=n, thumbnail=thumb)
        self.videos.append(info)
        return info

    def remove_video(self, idx: int):
        if 0 <= idx < len(self.videos):
            self.videos.pop(idx)
            self.segments.clear()

    def get_thumbnail(self, idx: int,
                      max_w: int = 160, max_h: int = 284) -> Optional[np.ndarray]:
        if idx >= len(self.videos) or self.videos[idx].thumbnail is None:
            return None
        t = self.videos[idx].thumbnail
        h, w = t.shape[:2]
        s = min(max_w / w, max_h / h, 1.0)
        return cv2.resize(t, (int(w * s), int(h * s)))

    # ── Video search & download ─────────────────────────────────

    def search_videos(self, query: str, platform: str = 'youtube',
                      max_results: int = 10) -> List[dict]:
        """Search for dance videos online via yt-dlp."""
        import yt_dlp

        if platform == 'bilibili':
            sq = f'bilisearch{max_results}:{query}'
        else:
            sq = f'ytsearch{max_results}:{query}'

        ytdlp_binary = self._get_ytdlp_binary()
        if ytdlp_binary:
            proc = subprocess.run(
                [ytdlp_binary, '--dump-json', '--flat-playlist', '--quiet', '--no-warnings', sq],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                **_SUBPROCESS_EXTRA,
            )
            results = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entry_id = entry.get('id', '')
                entry_url = entry.get('webpage_url') or entry.get('url') or ''
                if platform == 'youtube' and entry_id and not entry_url.startswith('http'):
                    entry_url = f'https://www.youtube.com/watch?v={entry_id}'
                results.append({
                    'title': entry.get('title', ''),
                    'duration': entry.get('duration', 0),
                    'url': entry_url,
                    'id': entry_id,
                })
            return results

        with yt_dlp.YoutubeDL({
            'quiet': True, 'no_warnings': True,
            'extract_flat': 'in_playlist', 'skip_download': True,
        }) as ydl:
            info = ydl.extract_info(sq, download=False)

        return [
            {'title': e.get('title', ''), 'duration': e.get('duration', 0),
             'url': e.get('url') or e.get('webpage_url', ''),
             'id': e.get('id', '')}
            for e in (info.get('entries') or []) if e
        ]

    def download_video(self, url: str, output_dir: str,
                       progress_cb: Optional[Callable] = None) -> str:
        """Download a single video and return its local path."""
        import yt_dlp

        ffmpeg_path = self._get_ffmpeg()
        out_tmpl = os.path.join(output_dir, '%(title).80s.%(ext)s')
        ytdlp_binary = self._get_ytdlp_binary()
        final_path = None

        if ytdlp_binary:
            if progress_cb:
                progress_cb(0.0)
            cmd = [
                ytdlp_binary,
                '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                '--merge-output-format', 'mp4',
                '--ffmpeg-location', ffmpeg_path,
                '--newline',
                '-o', out_tmpl,
                url,
            ]
            recent_output = []
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **_SUBPROCESS_EXTRA,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                recent_output.append(line.rstrip())
                recent_output = recent_output[-20:]
                # Capture final output path from yt-dlp
                m_dest = re.search(r'\[download\] Destination: (.+\.mp4)', line)
                m_merge = re.search(r'\[Merger\] Merging formats into "(.+?)"', line)
                m_already = re.search(r'\[download\] (.+\.mp4) has already been downloaded', line)
                if m_merge:
                    final_path = m_merge.group(1)
                elif m_dest:
                    final_path = m_dest.group(1)
                elif m_already:
                    final_path = m_already.group(1)
                if progress_cb:
                    match = re.search(r'(\d+(?:\.\d+)?)%', line)
                    if match:
                        progress_cb(min(float(match.group(1)) / 100.0, 0.99))
            ret = proc.wait()
            if ret != 0:
                err = '\n'.join(recent_output).strip() or 'yt-dlp download failed'
                raise subprocess.CalledProcessError(ret, cmd, output=err)
            if progress_cb:
                progress_cb(1.0)
        else:

            def hook(d):
                nonlocal final_path
                if d['status'] == 'downloading' and progress_cb:
                    total = d.get('total_bytes') or d.get('total_bytes_estimate', 1)
                    progress_cb(d.get('downloaded_bytes', 0) / max(total, 1))
                elif d['status'] == 'finished':
                    final_path = d.get('filename', '')

            with yt_dlp.YoutubeDL({
                'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
                'outtmpl': out_tmpl,
                'quiet': True, 'no_warnings': True,
                'ffmpeg_location': ffmpeg_path,
                'merge_output_format': 'mp4',
                'progress_hooks': [hook],
            }) as ydl:
                info = ydl.extract_info(url, download=True)
                if not final_path:
                    final_path = ydl.prepare_filename(info)
                    mp4 = os.path.splitext(final_path)[0] + '.mp4'
                    if os.path.exists(mp4):
                        final_path = mp4

        if final_path and os.path.exists(final_path):
            return final_path

        import glob
        found = sorted(glob.glob(os.path.join(output_dir, '*.mp4')),
                       key=os.path.getmtime, reverse=True)
        if found:
            return found[0]
        raise FileNotFoundError("下載失敗")

    # ── External audio ──────────────────────────────────────────

    def download_audio(self, query: str,
                       progress_cb: Optional[Callable] = None) -> Tuple[str, str]:
        """Search YouTube for *query*, download audio only. Returns (path, title)."""
        import yt_dlp

        dl_dir = tempfile.mkdtemp(prefix='dancemashup_audio_')
        tmp_base = os.path.join(dl_dir, 'audio')
        ytdlp_binary = self._get_ytdlp_binary()

        if ytdlp_binary:
            if progress_cb:
                progress_cb(0.05)
            title = query
            try:
                search_results = self.search_videos(query, platform='youtube', max_results=1)
            except Exception:
                search_results = []
            if search_results:
                title = search_results[0].get('title') or query
            subprocess.run(
                [
                    ytdlp_binary,
                    '-x',
                    '--audio-format', 'mp3',
                    '--ffmpeg-location', self._get_ffmpeg(),
                    '-o', tmp_base + '.%(ext)s',
                    f'ytsearch1:{query}',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                **_SUBPROCESS_EXTRA,
            )
            if progress_cb:
                progress_cb(0.6)
        else:

            def hook(d):
                if progress_cb and d['status'] == 'downloading':
                    total = d.get('total_bytes') or d.get('total_bytes_estimate', 1)
                    progress_cb(d.get('downloaded_bytes', 0) / max(total, 1) * 0.6)

            with yt_dlp.YoutubeDL({
                'format': 'bestaudio/best',
                'outtmpl': tmp_base + '.%(ext)s',
                'default_search': 'ytsearch1',
                'noplaylist': True, 'quiet': True, 'no_warnings': True,
                'progress_hooks': [hook],
            }) as ydl:
                info = ydl.extract_info(query, download=True)
                title = info.get('title', query)

        import glob
        dl_files = glob.glob(tmp_base + '.*')
        if not dl_files:
            raise FileNotFoundError("下載失敗")

        wav_path = os.path.join(dl_dir, 'source.wav')
        subprocess.run(
            [self._get_ffmpeg(), '-y', '-i', dl_files[0], '-vn',
             '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, **_SUBPROCESS_EXTRA)
        try:
            os.unlink(dl_files[0])
        except OSError:
            pass

        self.external_audio = wav_path
        self.external_audio_title = title
        if progress_cb:
            progress_cb(1.0)
        return wav_path, title

    def set_external_audio(self, path: str):
        """Use a local audio file."""
        self.external_audio = path
        self.external_audio_title = os.path.basename(path)

    def set_remote_rife(self, host: str, gpu: int = 1,
                        bin_path: str = r'C:\tools\rife-ncnn-vulkan\rife-ncnn-vulkan.exe',
                        model_path: str = r'C:\tools\rife-ncnn-vulkan\rife-v4.6',
                        work_dir: str = r'C:\DanceMashup\rife_work'):
        """Configure remote RIFE interpolation via SSH.

        Args:
            host: SSH host (e.g. 'user@remote-gpu.local')
            gpu: GPU index on remote machine (default 1 for dedicated GPU)
            bin_path: Path to rife-ncnn-vulkan.exe on remote machine
            model_path: Path to RIFE model dir on remote machine
            work_dir: Working directory on remote machine for temp files
        """
        self.remote_rife = {
            'host': host,
            'bin': bin_path,
            'model': model_path,
            'gpu': gpu,
            'work_dir': work_dir,
        }
        print(f"Remote RIFE configured: {host} (GPU {gpu})", flush=True)

    # ── Face recognition ────────────────────────────────────────

    def _get_face_app(self):
        """Lazy-load insightface model (downloads ~300 MB on first use)."""
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
            available = ort.get_available_providers()
            # Prefer CoreML (Apple GPU/Neural Engine) > CPU
            if 'CoreMLExecutionProvider' in available:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            self._face_app = FaceAnalysis(
                name='buffalo_l', providers=providers)
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    def search_reference_face(self, name: str, num_images: int = 12,
                              progress_cb: Optional[Callable] = None
                              ) -> np.ndarray:
        """Search the web for *name*, build a consensus face embedding.
        Returns an RGB face crop for GUI display."""
        import re
        import requests
        from urllib.parse import quote

        if progress_cb:
            progress_cb(0.0, "搜尋圖片…")

        query = f"{name} face portrait"
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
        }
        url = (f"https://www.bing.com/images/search?"
               f"q={quote(query)}&first=0&count={num_images * 3}"
               f"&qft=+filterui:face-face")
        resp = requests.get(url, headers=headers, timeout=15)
        image_urls = re.findall(
            r'murl&quot;:&quot;(https?://[^&]*?)&quot;', resp.text)

        if not image_urls:
            raise ValueError(f"找不到 {name} 的圖片")
        image_urls = image_urls[:num_images]

        if progress_cb:
            progress_cb(0.10, "載入人臉辨識模型…")
        app = self._get_face_app()

        embeddings: list = []
        crops: list = []

        for i, img_url in enumerate(image_urls):
            if progress_cb:
                progress_cb(0.10 + (i + 1) / len(image_urls) * 0.70,
                            f"分析人臉 {i+1}/{len(image_urls)}…")
            try:
                r = requests.get(img_url, timeout=8, headers=headers)
                if r.status_code != 200:
                    continue
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                faces = app.get(img)
                if not faces:
                    continue
                faces.sort(
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True)
                emb = faces[0].embedding
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
                b = faces[0].bbox.astype(int)
                crop = img[max(0, b[1]):b[3], max(0, b[0]):b[2]]
                crops.append(crop)
            except Exception:
                continue

        if len(embeddings) < 2:
            raise ValueError(
                f"僅偵測到 {len(embeddings)} 張人臉，不足以建立參考")

        # ── consensus: keep faces most similar to each other ──
        embs = np.array(embeddings)
        sim_matrix = embs @ embs.T
        avg_sims = sim_matrix.mean(axis=1)
        median_sim = np.median(avg_sims)
        mask = avg_sims >= median_sim
        consensus = embs[mask].mean(axis=0)
        consensus /= np.linalg.norm(consensus) + 1e-8
        self.reference_embedding = consensus

        best_idx = int(np.argmax(avg_sims))
        best_crop = cv2.cvtColor(crops[best_idx], cv2.COLOR_BGR2RGB)

        if progress_cb:
            progress_cb(1.0,
                        f"已建立 {name} 的人臉參考（{int(mask.sum())} 張共識）")
        return best_crop


    # IDENTITY_POOL_V1 — multi-reference + peer-negative helpers
    def _set_ref_pool(self, pool, negatives=None):
        """Install a multi-reference pool (N,512) and optional negatives (M,512).
        Also sets self.reference_embedding to the pool's normalised mean for
        back-compat with any old callsites."""
        def _norm2d(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size == 0:
                return None
            if arr.ndim == 1:
                arr = arr[None, :]
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            return arr / norms
        self.reference_embeddings = _norm2d(pool)
        self.negative_embeddings = _norm2d(negatives)
        if self.reference_embeddings is not None:
            mean = self.reference_embeddings.mean(axis=0)
            self.reference_embedding = mean / (np.linalg.norm(mean) + 1e-8)

    def _face_score(self, emb, penalty_thresh: float = 0.35,
                    penalty_weight: float = 0.6) -> float:
        """Cosine score against the target pool with a peer-negative penalty.
        Falls back to single reference_embedding when no pool is set."""
        e = np.asarray(emb, dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-8)
        if self.reference_embeddings is not None:
            target = float(np.max(self.reference_embeddings @ e))
        elif self.reference_embedding is not None:
            target = float(np.dot(self.reference_embedding, e))
        else:
            return 0.0
        if self.negative_embeddings is not None:
            neg = float(np.max(self.negative_embeddings @ e))
            if neg > penalty_thresh:
                target -= penalty_weight * (neg - penalty_thresh)
        return target

    def set_reference_face(self, image_path: str) -> np.ndarray:
        """Set reference face from a local image. Returns RGB face crop."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
        app = self._get_face_app()
        faces = app.get(img)
        if not faces:
            raise ValueError("圖片中未偵測到人臉")
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True)
        emb = faces[0].embedding
        self.reference_embedding = emb / (np.linalg.norm(emb) + 1e-8)
        b = faces[0].bbox.astype(int)
        crop = img[max(0, b[1]):b[3], max(0, b[0]):b[2]]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def clear_reference_face(self):
        self.reference_embedding = None
        # IDENTITY_POOL_V1
        self.reference_embeddings = None
        self.negative_embeddings = None

    def verify_face_by_thumbnail(self, video_id: str,
                                platform: str = 'youtube',
                                threshold: float = 0.4
                                ) -> Tuple[bool, float]:
        """Check reference face against a video's online thumbnail (fast).
        Returns ``(is_match, best_cosine_similarity)``."""
        import requests

        if self.reference_embedding is None:
            return True, 1.0

        if platform == 'youtube' and video_id:
            urls = [
                f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            ]
        else:
            return True, 1.0  # can't verify, accept

        app = self._get_face_app()
        headers = {'User-Agent': 'Mozilla/5.0'}
        best = 0.0

        for url in urls:
            try:
                resp = requests.get(url, timeout=8, headers=headers)
                if resp.status_code != 200:
                    continue
                arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                faces = app.get(img)
                for face in faces:
                    emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                    sim = self._face_score(emb)
                    if sim > best:
                        best = sim
                if best >= threshold:
                    return True, best
                if faces:          # got faces but no match, no need to try next URL
                    return False, best
            except Exception:
                continue

        # Couldn't get thumbnail — fall through to accept
        return True if best == 0.0 else (best >= threshold), best

    def verify_face_in_video(self, video_path: str,
                             sample_count: int = 12,
                             threshold: float = 0.42
                             ) -> Tuple[bool, float]:
        """Check whether the reference face appears in *video_path*.
        Upscales small frames and samples edges for close-ups.
        Returns ``(is_match, best_cosine_similarity)``."""
        if self.reference_embedding is None:
            return True, 1.0

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if total <= 0:
            cap.release()
            return False, 0.0

        # Sample: first 5 s + last 5 s (close-ups) + evenly spaced middle
        edge_secs = [0.5, 1, 2, 3, 4]
        dur = total / fps
        edge_frames = [int(t * fps) for t in edge_secs]
        edge_frames += [int((dur - t) * fps) for t in edge_secs]
        mid_frames = [int(total * (i + 1) / (sample_count + 1))
                      for i in range(sample_count)]
        all_frames = sorted(set(
            min(max(0, f), total - 1) for f in edge_frames + mid_frames))

        app = self._get_face_app()
        best = 0.0
        scale = max(1, 960 // max(w, 1))  # upscale small frames to ~960px wide

        for fi in all_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            if scale > 1:
                frame = cv2.resize(frame, (frame.shape[1] * scale,
                                           frame.shape[0] * scale))
            faces = app.get(frame)
            for face in faces:
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                sim = self._face_score(emb)
                if sim > best:
                    best = sim
                if best >= threshold:
                    cap.release()
                    return True, best

        cap.release()
        return best >= threshold, best

    def measure_face_visibility_ratio(self, video_path: str,
                                      sample_count: int = 20,
                                      threshold: float = 0.45
                                      ) -> Tuple[float, float]:
        """Sample *sample_count* frames evenly and compute the fraction
        where the reference face is visible at sim >= *threshold*.

        Returns ``(ratio, best_sim)``. Ratio mirrors the plan_transitions
        video-level face_visibility filter (default 15 %), so videos that
        fail here would also be filtered later — catching them at download
        time saves YOLO tracking work.
        """
        if self.reference_embedding is None:
            return 1.0, 1.0

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if total <= 0:
            cap.release()
            return 0.0, 0.0

        frames = [int(total * (i + 1) / (sample_count + 1))
                  for i in range(sample_count)]
        app = self._get_face_app()
        scale = max(1, 960 // max(w, 1))

        hits = 0
        best = 0.0
        seen = 0
        for fi in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            seen += 1
            if scale > 1:
                frame = cv2.resize(frame, (frame.shape[1] * scale,
                                           frame.shape[0] * scale))
            faces = app.get(frame)
            top_sim = 0.0
            for face in faces:
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                sim = self._face_score(emb)
                if sim > top_sim:
                    top_sim = sim
            if top_sim > best:
                best = top_sim
            if top_sim >= threshold:
                hits += 1

        cap.release()
        ratio = hits / seen if seen else 0.0
        return ratio, best

    def detect_members_in_video(self, video_path: str,
                                group_embeddings: dict,
                                sample_count: int = 16,
                                threshold: float = 0.40
                                ) -> dict:
        """Scan *video_path* and return {member_name: best_similarity}.

        ``group_embeddings`` is a dict of {member_name: np.ndarray embedding}.
        Returns the max cosine similarity observed for each member.
        Members with sim >= threshold are considered "present" in the video.
        """
        result = {name: 0.0 for name in group_embeddings}
        if not group_embeddings:
            return result

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if total <= 0:
            cap.release()
            return result

        edge_secs = [0.5, 1.5, 3.0]
        dur = total / fps
        edge_frames = [int(t * fps) for t in edge_secs]
        edge_frames += [int((dur - t) * fps) for t in edge_secs]
        mid_frames = [int(total * (i + 1) / (sample_count + 1))
                      for i in range(sample_count)]
        all_frames = sorted(set(
            min(max(0, f), total - 1) for f in edge_frames + mid_frames))

        app = self._get_face_app()
        scale = max(1, 960 // max(w, 1))

        for fi in all_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            if scale > 1:
                frame = cv2.resize(frame, (frame.shape[1] * scale,
                                           frame.shape[0] * scale))
            faces = app.get(frame)
            for face in faces:
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                for name, ref in group_embeddings.items():
                    sim = float(np.dot(ref, emb))
                    if sim > result[name]:
                        result[name] = sim

        cap.release()
        return result

    # ── Pose analysis ───────────────────────────────────────────

    def analyze_poses(self, video_idx: int, sample_fps: float = 10.0,
                      progress_cb: Optional[Callable] = None):
        """Extract body poses from *video_idx* at ~sample_fps Hz."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        video = self.videos[video_idx]
        cap = cv2.VideoCapture(video.path)
        interval = max(1, round(video.fps / sample_fps))
        actual_fps = video.fps / interval
        poses: list = []

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )

        with vision.PoseLandmarker.create_from_options(options) as det:
            fi = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if fi % interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts_ms = int(fi / video.fps * 1000)
                    mp_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = det.detect_for_video(mp_img, ts_ms)
                    if res.pose_landmarks and len(res.pose_landmarks) > 0:
                        lm = np.array([[l.x, l.y, l.visibility]
                                       for l in res.pose_landmarks[0]])
                    else:
                        lm = poses[-1].copy() if poses else np.zeros((33, 3))
                    poses.append(lm)
                fi += 1
                if progress_cb and fi % 30 == 0:
                    progress_cb(fi / max(video.total_frames, 1))

        cap.release()
        video.poses = np.array(poses) if poses else np.zeros((1, 33, 3))
        video.pose_fps = actual_fps
        if progress_cb:
            progress_cb(1.0)

    # ── Beat detection ──────────────────────────────────────────

    def detect_beats(self, audio_source_idx: int = 0,
                     use_external: bool = False,
                     progress_cb: Optional[Callable] = None) -> np.ndarray:
        """Detect musical beats from audio."""
        import librosa

        if use_external and self.external_audio:
            audio_input = self.external_audio
        else:
            audio_input = self.videos[audio_source_idx].path
        tmp = tempfile.mktemp(suffix='.wav')
        try:
            ffmpeg = self._get_ffmpeg()
            subprocess.run(
                [ffmpeg, '-y', '-i', audio_input, '-vn',
                 '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True, **_SUBPROCESS_EXTRA,
            )
            if progress_cb:
                progress_cb(0.3)

            y, sr = librosa.load(tmp, sr=22050)
            if progress_cb:
                progress_cb(0.6)

            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            self.beat_times = librosa.frames_to_time(beats, sr=sr)
            if progress_cb:
                progress_cb(1.0)
            return self.beat_times
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ── Pose similarity ─────────────────────────────────────────

    @staticmethod
    def _normalize_pose(lm: np.ndarray) -> np.ndarray:
        """Centre on hips, scale by torso → position/scale invariant."""
        out = lm.copy()
        hip = (lm[23, :2] + lm[24, :2]) / 2.0
        out[:, :2] -= hip
        sh = (lm[11, :2] + lm[12, :2]) / 2.0
        torso = np.linalg.norm(sh - hip)
        if torso > 1e-6:
            out[:, :2] /= torso
        return out

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = self._normalize_pose(a), self._normalize_pose(b)
        ka, kb = na[DANCE_INDICES, :2], nb[DANCE_INDICES, :2]
        vis = np.minimum(na[DANCE_INDICES, 2], nb[DANCE_INDICES, 2])
        if vis.sum() < 1e-6:
            return 0.0
        diff = np.linalg.norm(ka - kb, axis=1)
        return float(np.clip(np.exp(-np.sum(diff * vis) / vis.sum() * 3), 0, 1))


    def _motion_similarity(self, vidx_a, vidx_b, t, dt=0.15, offsets=None):
        """Score motion direction match between two videos at time t."""
        off_a = (offsets or {}).get(vidx_a, 0.0)
        off_b = (offsets or {}).get(vidx_b, 0.0)
        pa0 = self._pose_at(vidx_a, t + off_a)
        pa1 = self._pose_at(vidx_a, t + off_a + dt)
        pb0 = self._pose_at(vidx_b, t + off_b)
        pb1 = self._pose_at(vidx_b, t + off_b + dt)
        if any(p is None for p in [pa0, pa1, pb0, pb1]):
            return 0.5
        na0 = self._normalize_pose(pa0)[DANCE_INDICES, :2]
        na1 = self._normalize_pose(pa1)[DANCE_INDICES, :2]
        nb0 = self._normalize_pose(pb0)[DANCE_INDICES, :2]
        nb1 = self._normalize_pose(pb1)[DANCE_INDICES, :2]
        va = (na1 - na0).flatten()
        vb = (nb1 - nb0).flatten()
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.7
        cos_sim = float(np.dot(va, vb) / (norm_a * norm_b))
        return float(np.clip((cos_sim + 1) / 2, 0, 1))

    def _pose_at(self, vidx: int, t: float) -> Optional[np.ndarray]:
        v = self.videos[vidx]
        if v.poses is None or len(v.poses) == 0:
            return None
        i = min(int(t * v.pose_fps), len(v.poses) - 1)
        return v.poses[max(i, 0)]


    def compute_audio_offsets(self, reference_idx: int = 0,
                              use_external: bool = False,
                              progress_cb: Optional[Callable] = None):
        """Compute time offset of each video relative to reference audio.

        Uses audio cross-correlation (onset strength envelope) to find
        how many seconds each video's audio is ahead/behind the reference.
        The offsets are stored in self.audio_offsets and used during rendering
        to keep dance moves synced with the music across cuts.
        """
        import librosa

        ffmpeg = self._get_ffmpeg()
        tmp_dir = tempfile.mkdtemp(prefix='audio_sync_')

        try:
            # Extract reference audio
            if use_external and self.external_audio:
                ref_path = self.external_audio
            else:
                ref_path = self.videos[reference_idx].path
            ref_wav = os.path.join(tmp_dir, 'ref.wav')
            subprocess.run(
                [ffmpeg, '-y', '-i', ref_path, '-vn',
                 '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                 ref_wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True, **_SUBPROCESS_EXTRA)
            ref_y, sr = librosa.load(ref_wav, sr=22050)
            ref_onset = librosa.onset.onset_strength(y=ref_y, sr=sr)

            for vi, video in enumerate(self.videos):
                if not use_external and vi == reference_idx:
                    self.audio_offsets[vi] = 0.0
                    if progress_cb:
                        progress_cb((vi + 1) / len(self.videos))
                    continue

                # Extract this video's audio
                vid_wav = os.path.join(tmp_dir, f'v{vi}.wav')
                subprocess.run(
                    [ffmpeg, '-y', '-i', video.path, '-vn',
                     '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                     vid_wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=True, **_SUBPROCESS_EXTRA)
                vid_y, _ = librosa.load(vid_wav, sr=22050)
                vid_onset = librosa.onset.onset_strength(y=vid_y, sr=sr)

                # Cross-correlate onset envelopes
                min_len = min(len(ref_onset), len(vid_onset))
                if min_len < 100:
                    self.audio_offsets[vi] = 0.0
                    continue

                corr = np.correlate(
                    ref_onset[:min_len],
                    vid_onset[:min_len],
                    mode='full')
                # Peak of correlation = offset in onset frames
                hop = 512  # librosa default hop_length
                peak = np.argmax(corr) - (min_len - 1)
                offset_sec = -peak * hop / sr  # negated: positive = vid is behind ref
                # Clamp to reasonable range (-30s to +30s)
                offset_sec = float(np.clip(offset_sec, -30, 30))
                self.audio_offsets[vi] = offset_sec
                print(f"  V{vi+1} offset: {offset_sec:+.2f}s", flush=True)

                if progress_cb:
                    progress_cb((vi + 1) / len(self.videos))
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return self.audio_offsets

    def compute_face_visibility(self, video_idx: int,
                                sample_interval: float = 1.0,
                                sim_thresh: float = 0.42):
        """Pre-compute face similarity at each sample time.
        Stores (time, max_similarity) tuples for finer-grained scoring."""
        if self.reference_embedding is None:
            return

        video = self.videos[video_idx]
        cap = cv2.VideoCapture(video.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return

        app = self._get_face_app()
        step = max(1, int(fps * sample_interval))
        visibility = []

        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            t = fi / fps
            faces = app.get(frame)
            max_sim = 0.0
            if faces:
                for face in faces:
                    emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                    sim = self._face_score(emb)
                    if sim > max_sim:
                        max_sim = sim
            visibility.append((t, max_sim))

        cap.release()
        self.face_visibility[video_idx] = visibility
        n_vis = sum(1 for _, s in visibility if s >= sim_thresh)
        avg_sim = np.mean([s for _, s in visibility if s > 0]) if any(s > 0 for _, s in visibility) else 0
        print(f"  V{video_idx+1} face visibility: {n_vis}/{len(visibility)} samples"
              f" (avg_sim={avg_sim:.3f})", flush=True)

    def _is_face_visible(self, video_idx: int, t: float,
                         window: float = 0.8,
                         sim_thresh: float = 0.50) -> bool:
        """Check if reference face is visible near time t in video."""
        if video_idx not in self.face_visibility:
            return True  # assume visible if not computed
        vis = self.face_visibility[video_idx]
        if not vis:
            return True
        # Check samples within window around t
        for sample_t, sim in vis:
            if abs(sample_t - t) <= window:
                if sim >= sim_thresh:
                    return True
        return False

    def _is_face_visible_continuous(self, video_idx: int, t_start: float,
                                    lookahead: float = 4.0,
                                    sample_interval: float = 1.0,
                                    min_pass_ratio: float = 0.7,
                                    window: float = 0.8,
                                    sim_thresh: float = 0.50) -> bool:
        """Require target face to be visible throughout a forward window.

        Samples [t_start, t_start+lookahead] every sample_interval and returns
        True only if at least min_pass_ratio of those samples have the target
        visible (sim >= sim_thresh within +/-window). This prevents cutting to
        a segment where the target briefly appears then disappears.
        """
        if video_idx not in self.face_visibility:
            return True
        vis = self.face_visibility[video_idx]
        if not vis:
            return True
        n = max(1, int(lookahead / sample_interval))
        passes = 0
        for i in range(n):
            check_t = t_start + i * sample_interval
            for sample_t, sim in vis:
                if abs(sample_t - check_t) <= window:
                    if sim >= sim_thresh:
                        passes += 1
                        break
        return (passes / n) >= min_pass_ratio

    def _face_score_at(self, video_idx: int, t: float,
                       window: float = 2.0) -> float:
        """Get the max face similarity score near time t."""
        if video_idx not in self.face_visibility:
            return 0.5  # neutral if not computed
        vis = self.face_visibility[video_idx]
        if not vis:
            return 0.5
        best = 0.0
        for sample_t, sim in vis:
            if abs(sample_t - t) <= window:
                if sim > best:
                    best = sim
        return best

            # ── Transition planning ─────────────────────────────────────

    def plan_transitions(self, min_dur: float = 3.5, max_dur: float = 8.0,
                         sim_thresh: float = 0.3) -> List[Segment]:
        """Decide which video plays in each time segment."""
        if self.beat_times is None or len(self.videos) < 2:
            if self.videos:
                self.segments = [Segment(0, 0.0, self.videos[0].duration)]
            return self.segments

        end = min(v.duration for v in self.videos)
        beats = self.beat_times[self.beat_times < end]
        nv = len(self.videos)

        # FACE_VISIBILITY_VIDEO_FILTER
        # Exclude videos whose target-face presence is too sparse to be worth
        # cutting to (e.g. the target only appears in the intro). Uses the
        # already-computed self.face_visibility samples.
        gate_thresh = 0.45
        min_fraction = 0.15
        good_videos = set(range(nv))
        if self.reference_embedding is not None and self.face_visibility:
            fractions = {}
            for v in range(nv):
                vis = self.face_visibility.get(v)
                if not vis:
                    fractions[v] = 1.0  # no data -> don't exclude
                    continue
                n_hit = sum(1 for _, s in vis if s >= gate_thresh)
                fractions[v] = n_hit / len(vis)
            qualified = {v for v, f in fractions.items() if f >= min_fraction}
            if len(qualified) >= 1:
                good_videos = qualified
                excluded = sorted(set(range(nv)) - qualified)
                if excluded:
                    print(f"  plan_transitions: excluding videos "
                          f"{[f'V{v+1}({fractions[v]*100:.0f}%)' for v in excluded]}"
                          f" (target face < {int(min_fraction*100)}% of samples)",
                          flush=True)
                if len(qualified) == 1:
                    only = next(iter(qualified))
                    print(f"  plan_transitions: WARNING only V{only+1} has"
                          f" target face — output will be single-video",
                          flush=True)
            else:
                print(f"  plan_transitions: NO video passes visibility"
                      f" gate — disabling filter", flush=True)

        cur, prev, last_t = 0, -1, 0.0
        segs: List[Segment] = []

        for bt in beats:
            elapsed = bt - last_t
            if elapsed < min_dur:
                continue

            cp = self._pose_at(cur, bt + self.audio_offsets.get(cur, 0.0))
            if cp is None:
                continue

            # Score every candidate video; remember top-K for UI swap.
            scored = []  # (score, video_idx)
            for v in range(nv):
                if v == cur:
                    continue
                if v not in good_videos:
                    continue
                if v == prev and elapsed < max_dur and nv > 2:
                    continue
                # Skip video if reference face not continuously visible
                # across the forward lookahead window (prevents cuts where
                # target appears briefly then disappears).
                vid_t = bt + self.audio_offsets.get(v, 0.0)
                if not self._is_face_visible_continuous(v, vid_t):
                    continue
                tp = self._pose_at(v, vid_t)
                if tp is None:
                    continue
                pose_s = self._similarity(cp, tp)
                motion_s = self._motion_similarity(cur, v, bt, offsets=self.audio_offsets)
                face_s = self._face_score_at(v, vid_t)
                face_bonus = max(0, face_s - 0.3) * 1.2
                s = 0.6 * pose_s + 0.4 * motion_s + face_bonus
                scored.append((s, v))

            scored.sort(key=lambda x: -x[0])
            best_s, best_v = (scored[0] if scored else (-1.0, -1))

            if best_v >= 0 and (best_s >= sim_thresh or elapsed >= max_dur):
                # Also score the current video (cur) + prev so UI can offer
                # "stay on V{cur}" as an alternative.
                extra = []
                for v in (cur,) + ((prev,) if prev >= 0 else ()):
                    if v in good_videos:
                        vid_t2 = bt + self.audio_offsets.get(v, 0.0)
                        tp2 = self._pose_at(v, vid_t2)
                        if tp2 is not None:
                            pose_s2 = self._similarity(cp, tp2)
                            extra.append((pose_s2, v))
                # Dedup by video_idx, keep max score
                pool = {}
                for sc, v in scored + extra:
                    if v not in pool or sc > pool[v]:
                        pool[v] = sc
                cands = sorted(pool.items(), key=lambda kv: -kv[1])[:4]
                cands_list = [{"video_idx": v, "score": float(sc)}
                              for v, sc in cands]
                seg = Segment(cur, last_t, bt, best_s, candidates=cands_list)
                segs.append(seg)
                prev, cur, last_t = cur, best_v, bt

        if last_t < end:
            segs.append(Segment(cur, last_t, end))

        self.segments = segs
        return segs

    # ── Plan persistence / swap ──────────────────────────────────

    def export_plan_json(self, mashup_path: str) -> str:
        """Write a self-contained plan JSON next to *mashup_path*.
        Returns the JSON path.
        """
        import json as _json
        out_path = os.path.splitext(mashup_path)[0] + ".plan.json"
        duration = float(sum(s.duration for s in self.segments)) if self.segments else 0.0
        vids = []
        for i, v in enumerate(self.videos):
            vids.append({
                "idx": i,
                "filename": v.filename,
                "source_url": getattr(v, "source_url", ""),
                "duration": float(v.duration),
            })
        face_vis = {}
        for vi, vis in (self.face_visibility or {}).items():
            face_vis[str(vi)] = [[float(t), float(s)] for t, s in vis]
        segs_out = []
        for i, s in enumerate(self.segments):
            segs_out.append({
                "idx": i,
                "t_start": float(s.start_time),
                "t_end": float(s.end_time),
                "video_idx": int(s.video_idx),
                "similarity": float(s.similarity),
                "candidates": list(s.candidates) if s.candidates else [],
            })
        beat_list = []
        try:
            if self.beat_times is not None:
                beat_list = [float(t) for t in list(self.beat_times)]
        except Exception:
            pass
        payload = {
            "schema_version": 1,
            "mashup": os.path.basename(mashup_path),
            # AUTOARCHIVE_V2 — carry target identity with the plan
            "target_group": getattr(self, "target_group", None) or "",
            "target_member": getattr(self, "target_member", None) or "",
            # TIKTOK_DLNAME_V1 — so downloads can be named <group>_<member>_<song>
            "song": getattr(self, "external_audio_title", "") or "",
            "duration": duration,
            "videos": vids,
            "audio_offsets": {str(k): float(v)
                              for k, v in (self.audio_offsets or {}).items()},
            "beat_times": beat_list,
            "face_visibility": face_vis,
            "segments": segs_out,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2)
        return out_path

    def apply_plan_segments(self, plan: dict):
        """Overwrite self.segments from a loaded plan dict."""
        new_segs = []
        for s in plan.get("segments", []):
            new_segs.append(Segment(
                video_idx=int(s["video_idx"]),
                start_time=float(s["t_start"]),
                end_time=float(s["t_end"]),
                similarity=float(s.get("similarity", 0.0)),
                candidates=list(s.get("candidates", [])),
            ))
        self.segments = new_segs

    # ── Smart crop (16:9 → 9:16 follow person) ────────────────

    def _compute_crop_centers(self, video_idx: int) -> Optional[tuple]:
        """Return (smoothed_centers, pose_fps) for horizontal person tracking."""
        from scipy.ndimage import gaussian_filter1d

        video = self.videos[video_idx]
        if video.poses is None or len(video.poses) == 0:
            return None

        # CROP_TIGHT_V1 — prefer torso midpoint (hips 23/24 + shoulders
        # 11/12); ignore raised arms/legs so the crop doesn't get yanked.
        raw = []
        last = 0.5
        for pose in video.poses:
            vis = pose[:, 2] > 0.3
            torso_ids = [23, 24, 11, 12]  # hips first, shoulders second
            torso_xs = [pose[k, 0] for k in torso_ids if pose[k, 2] > 0.3]
            if torso_xs:
                cx = float(np.mean(torso_xs))
            elif pose[0, 2] > 0.3:   # nose fallback
                cx = float(pose[0, 0])
            elif vis.any():          # last resort: old min/max midpoint
                cx = float((pose[vis, 0].min() + pose[vis, 0].max()) / 2.0)
            else:
                cx = last
            raw.append(cx)
            last = cx

        raw = np.array(raw)
        sigma = max(1, int(video.pose_fps * 0.15))   # CROP_TIGHT_V2: ~0.15s
        smoothed = gaussian_filter1d(raw, sigma=sigma)
        return smoothed, video.pose_fps

    def _compute_target_track_centers(
            self, video_idx: int,
            progress_cb=None,
            track_sim_thresh: float = 0.48,
            ):
        """Person-tracker based crop: YOLO+ByteTrack follows bodies; face
        recognition only picks which track is the target. Works when the
        target's face is turned away or occluded.

        Returns (smoothed_centers, pose_fps) or None if unavailable.
        """
        if self.reference_embedding is None:
            return None
        try:
            import person_tracker as pt
        except ImportError:
            return None

        from scipy.ndimage import gaussian_filter1d
        video = self.videos[video_idx]
        cache_dir = os.path.join(os.path.dirname(video.path), '.cache_tracks')
        _tag = f"V{video_idx+1}/{len(self.videos)}"
        _pg_phase_begin(f"Track {_tag}")
        try:
            tracks = pt.compute_tracks(video.path, cache_dir,
                                       progress_cb=progress_cb,
                                       tag=_tag)
        except Exception as e:
            print(f"  V{video_idx+1} YOLO-track failed: {e}", flush=True)
            _pg_phase_end(f"Track {_tag}")
            return None
        _pg_phase_end(f"Track {_tag}")
        if not tracks:
            return None
        app = self._get_face_app()
        _pg_phase_begin(f"Identify {_tag}")
        try:
            targets = pt.identify_target_tracks(
                video.path, tracks, app, self.reference_embedding,
                sim_thresh=track_sim_thresh,
                ref_embeddings=self.reference_embeddings,
                neg_embeddings=self.negative_embeddings,
                tag=_tag)
        except Exception as e:
            print(f"  V{video_idx+1} target-id failed: {e}", flush=True)
            _pg_phase_end(f"Identify {_tag}")
            return None
        _pg_phase_end(f"Identify {_tag}")
        if not targets:
            print(f"  V{video_idx+1} person-track: no targets >= {track_sim_thresh}",
                  flush=True)
            return None
        pfps = video.pose_fps if video.pose_fps > 0 else 10.0
        n = max(1, int(video.duration * pfps))
        # CROP_ZOOMOUT_V1 — also receive per-sample bbox area ratios
        centers, areas = pt.target_crop_centers(video.path, tracks, targets, n, pfps)
        sigma = max(1, int(pfps * 0.20))   # CROP_TIGHT_V2: was 0.8
        smoothed_c = gaussian_filter1d(centers, sigma=sigma)
        smoothed_a = gaussian_filter1d(areas,   sigma=max(sigma, int(pfps * 0.5)))
        total_target_frames = sum(len(tracks[t]) for t in targets)
        print(f"  V{video_idx+1} person-track: {len(targets)} target tracks,"
              f" {total_target_frames} frames, best sim={max(targets.values()):.3f}",
              flush=True)
        return smoothed_c, pfps, smoothed_a

    def _compute_face_crop_centers(self, video_idx,
                                   sample_interval=0.5,
                                   progress_cb=None):
        """Smart-crop center-tracking. Prefers YOLO+ByteTrack person tracking
        when available; falls back to face-only sampling if YOLO missing or
        finds no target. Lost frames freeze on last known position.
        """
        if self.reference_embedding is None:
            return self._compute_crop_centers(video_idx)

        pt_result = self._compute_target_track_centers(video_idx,
                                                      progress_cb=progress_cb)
        if pt_result is not None:
            return pt_result

        from scipy.ndimage import gaussian_filter1d
        video = self.videos[video_idx]
        cap = cv2.VideoCapture(video.path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if total <= 0:
            cap.release()
            return self._compute_crop_centers(video_idx)
        app = self._get_face_app()
        step = max(1, int(fps * sample_interval))
        confirmed = []
        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            faces = app.get(frame)
            if not faces:
                continue
            best_sim, best_cx, second = -1.0, 0.5, -1.0
            for face in faces:
                cx = (face.bbox[0] + face.bbox[2]) / 2.0 / w
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                sim = self._face_score(emb)
                if sim > best_sim:
                    second = best_sim
                    best_sim, best_cx = sim, cx
                elif sim > second:
                    second = sim
            margin_ok = len(faces) <= 1 or (best_sim - max(0.0, second)) >= 0.08
            if best_sim >= 0.42 and margin_ok:
                confirmed.append((fi, best_cx))
            if progress_cb and fi % (step * 10) == 0:
                progress_cb(fi / total)
        cap.release()
        if len(confirmed) < 3:
            return self._compute_crop_centers(video_idx)
        pfps = video.pose_fps if video.pose_fps > 0 else 10.0
        n_samples = max(1, int(video.duration * pfps))
        kf = np.array([k[0] for k in confirmed], dtype=float) / fps * pfps
        kc = np.array([k[1] for k in confirmed], dtype=float)
        idx = np.arange(n_samples, dtype=float)
        centers = np.interp(idx, kf, kc)
        # Freeze positions beyond 2s gaps
        kf_times = np.array([k[0] for k in confirmed]) / fps
        sample_t = idx / pfps
        left = np.clip(np.searchsorted(kf_times, sample_t) - 1, 0, len(kf_times) - 1)
        right = np.clip(left + 1, 0, len(kf_times) - 1)
        dist = np.minimum(np.abs(sample_t - kf_times[left]),
                          np.abs(sample_t - kf_times[right]))
        # For big gaps: hold nearest confirmed value (no drift)
        mask_gap = dist > 2.0
        if mask_gap.any():
            nearest_idx = np.where(np.abs(sample_t - kf_times[left]) <=
                                   np.abs(sample_t - kf_times[right]), left, right)
            centers[mask_gap] = kc[nearest_idx[mask_gap]]
        sigma = max(1, int(pfps * 0.30))   # CROP_TIGHT_V2: was 1.0
        smoothed = gaussian_filter1d(centers, sigma=sigma)
        return smoothed, pfps


    # DELOGO_V1 — wrapper that post-processes RIFE output through ffmpeg
    # delogo to blank out static overlays (broadcaster bugs, captions, etc.).
    def _ensure_60fps(self, video_idx: int,
                      progress_cb: Optional[Callable] = None
                      ) -> Tuple[str, float]:
        if not hasattr(self, "_final_cache"):
            self._final_cache = {}
        if video_idx in self._final_cache:
            return self._final_cache[video_idx]
        path, fps = self._ensure_60fps_raw(video_idx, progress_cb)
        if getattr(self, "delogo_enabled", True) and _logo_detect is not None:
            try:
                path, fps = self._apply_delogo(video_idx, path, fps)
            except Exception as e:
                print(f"[delogo] video {video_idx} skipped: {e}")
        self._final_cache[video_idx] = (path, fps)
        return (path, fps)

    def _apply_delogo(self, video_idx: int, path: str,
                      fps: float) -> Tuple[str, float]:
        """Detect static overlay boxes on the SOURCE video and re-encode
        *path* through ffmpeg delogo. Returns cleaned path (or input if
        nothing detected / ffmpeg unavailable)."""
        import subprocess, shutil
        if _logo_detect is None:
            return (path, fps)
        src = self.videos[video_idx].path
        boxes = _logo_detect.detect_logo_boxes(src, verbose=True)
        if not boxes:
            return (path, fps)
        # Scale boxes from source resolution to current 60fps output
        src_cap = cv2.VideoCapture(src)
        sw = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        sh = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        src_cap.release()
        dst_cap = cv2.VideoCapture(path)
        dw = int(dst_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or sw)
        dh = int(dst_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or sh)
        dst_cap.release()
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            return (path, fps)
        sx_r = dw / sw
        sy_r = dh / sh
        scaled = []
        for b in boxes:
            x = max(1, int(round(b["x"] * sx_r)))
            y = max(1, int(round(b["y"] * sy_r)))
            w = max(2, int(round(b["w"] * sx_r)))
            h = max(2, int(round(b["h"] * sy_r)))
            # delogo requires x+w < frame_w and y+h < frame_h
            if x + w >= dw:
                w = max(2, dw - x - 1)
            if y + h >= dh:
                h = max(2, dh - y - 1)
            scaled.append({"x": x, "y": y, "w": w, "h": h})
        chain = _logo_detect.delogo_filter_chain(scaled)
        if not chain:
            return (path, fps)

        stem, ext = os.path.splitext(path)
        clean_path = f"{stem}_clean{ext}"
        if os.path.exists(clean_path) and os.path.getsize(clean_path) > 0:
            return (clean_path, fps)

        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [ffmpeg, "-y", "-i", path,
               "-vf", chain,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-c:a", "copy", "-movflags", "+faststart",
               clean_path]
        print(f"[delogo] v{video_idx}: {len(scaled)} box(es) -> {os.path.basename(clean_path)}")
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", "ignore")[-400:]
            print(f"[delogo] ffmpeg failed: {err}")
            try:
                if os.path.exists(clean_path):
                    os.remove(clean_path)
            except Exception:
                pass
            return (path, fps)
        return (clean_path, fps)

    def _ensure_60fps_raw(self, video_idx: int,
                          progress_cb: Optional[Callable] = None
                          ) -> Tuple[str, float]:
        """Return (path, fps) for a ~60fps version of the source.

        Uses RIFE AI frame interpolation when available (best quality),
        falls back to ffmpeg framerate filter otherwise.  Results are
        cached in ``.cache_60fps/`` so preprocessing only happens once.
        """
        if video_idx in self._interp_cache:
            return self._interp_cache[video_idx]

        video = self.videos[video_idx]
        if video.fps >= 50:
            result = (video.path, video.fps)
            self._interp_cache[video_idx] = result
            return result

        # Build cache path based on source file identity
        cache_dir = os.path.join(os.path.dirname(video.path), '.cache_60fps')
        os.makedirs(cache_dir, exist_ok=True)
        stat = os.stat(video.path)
        key = f"{video.path}:{stat.st_size}"
        name_hash = hashlib.md5(key.encode()).hexdigest()[:12]

        # Check for RIFE binary (local or remote)
        base = os.path.dirname(os.path.abspath(__file__))
        rife_bin = os.path.join(
            base, 'rife-ncnn-vulkan',
            'rife-ncnn-vulkan-20221029-macos', 'rife-ncnn-vulkan')
        rife_model = os.path.join(
            base, 'rife-ncnn-vulkan',
            'rife-ncnn-vulkan-20221029-macos', 'rife-v4.6')
        use_local_rife = os.path.exists(rife_bin)
        use_remote_rife = self.remote_rife is not None

        tag = 'rife' if (use_local_rife or use_remote_rife) else 'fr'
        cached_path = os.path.join(cache_dir, f"{name_hash}_{tag}_60fps.mp4")

        if os.path.exists(cached_path):
            cap = cv2.VideoCapture(cached_path)
            if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
                cap.release()
                result = (cached_path, fps)
                self._interp_cache[video_idx] = result
                return result
            cap.release()

        if use_remote_rife:
            return self._rife_interpolate_remote(
                video_idx, cached_path, progress_cb)
        elif use_local_rife:
            return self._rife_interpolate(
                video_idx, rife_bin, rife_model, cached_path, progress_cb)
        else:
            return self._framerate_interpolate(
                video_idx, cached_path, progress_cb)

    def _rife_interpolate(self, video_idx: int,
                          rife_bin: str, rife_model: str,
                          cached_path: str,
                          progress_cb: Optional[Callable] = None
                          ) -> Tuple[str, float]:
        """Use RIFE AI to interpolate a low-fps video to ~60fps."""
        video = self.videos[video_idx]
        target_fps = 60.0
        target_frames = int(video.total_frames * target_fps / video.fps)
        ffmpeg = self._get_ffmpeg()

        tmp_base = tempfile.mkdtemp(prefix='rife_')
        in_dir = os.path.join(tmp_base, 'input')
        out_dir = os.path.join(tmp_base, 'output')
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        try:
            # 1) Extract source frames as high-quality JPEG
            if progress_cb:
                progress_cb(0, f"RIFE: extracting frames from {video.filename}")
            print(f"RIFE: extracting {video.total_frames} frames...", flush=True)
            subprocess.run([
                ffmpeg, '-y', '-i', video.path,
                '-qscale:v', '2',
                os.path.join(in_dir, '%08d.jpg'),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
               check=True, **_SUBPROCESS_EXTRA)

            # 2) Run RIFE AI interpolation
            if progress_cb:
                progress_cb(0, f"RIFE: AI interpolating ({video.total_frames} -> {target_frames} frames)")
            print(f"RIFE: interpolating {video.total_frames} -> {target_frames} frames...", flush=True)
            proc = subprocess.run([
                rife_bin, '-i', in_dir, '-o', out_dir,
                '-m', rife_model,
                '-n', str(target_frames),
                '-j', '4:4:4',
                '-f', '%08d.jpg',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
               **_SUBPROCESS_EXTRA)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode(errors='replace')[:500])

            # 3) Reassemble to cached video
            if progress_cb:
                progress_cb(0, f"RIFE: encoding {video.filename}")
            print("RIFE: encoding cached video...", flush=True)
            if platform.system() == "Darwin":
                enc = ['-c:v', 'h264_videotoolbox', '-b:v', '12M']
            else:
                enc = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']
            subprocess.run([
                ffmpeg, '-y',
                '-framerate', str(target_fps),
                '-i', os.path.join(out_dir, '%08d.jpg'),
                *enc, '-pix_fmt', 'yuv420p', '-an',
                cached_path,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
               check=True, **_SUBPROCESS_EXTRA)
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

        cap = cv2.VideoCapture(cached_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        cap.release()
        result = (cached_path, fps)
        self._interp_cache[video_idx] = result
        if progress_cb:
            progress_cb(0, f"RIFE done: {video.filename}")
        print(f"RIFE: done -> {cached_path}", flush=True)
        return result

    def _rife_interpolate_remote(self, video_idx: int,
                                    cached_path: str,
                                    progress_cb: Optional[Callable] = None
                                    ) -> Tuple[str, float]:
        """Use RIFE on a remote GPU machine via SSH."""
        video = self.videos[video_idx]
        target_fps = 60.0
        target_frames = int(video.total_frames * target_fps / video.fps)
        rc = self.remote_rife
        host = rc['host']
        work = rc['work_dir']

        # Use a unique job name based on video hash
        stat = os.stat(video.path)
        key = f"{video.path}:{stat.st_size}"
        job_id = hashlib.md5(key.encode()).hexdigest()[:12]

        # Windows paths for remote commands
        r_job = f"{work}\\{job_id}"
        r_in = f"{r_job}\\input"
        r_out = f"{r_job}\\output"
        r_src = f"{r_job}\\source.mp4"
        r_result = f"{r_job}\\result_60fps.mp4"

        try:
            # 1) Create remote dirs
            print(f"Remote RIFE: setting up job {job_id} on {host}...", flush=True)
            if progress_cb:
                progress_cb(0, f"Remote RIFE: uploading {video.filename}")
            subprocess.run(
                ['ssh', host, f'mkdir {r_job} & mkdir {r_in} & mkdir {r_out}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False)

            # 2) SCP source video to remote
            print(f"Remote RIFE: uploading {video.filename}...", flush=True)
            subprocess.run(
                ['scp', '-q', video.path, f'{host}:{r_src.replace(chr(92), "/")}'],
                check=True)

            # 3) Extract frames on remote (using ffmpeg on remote)
            print(f"Remote RIFE: extracting frames on remote...", flush=True)
            if progress_cb:
                progress_cb(0, f"Remote RIFE: extracting frames on {host}")
            subprocess.run(
                ['ssh', host,
                 f'C:/tools/ffmpeg.exe -y -i {r_src} -qscale:v 2 {r_in}/%08d.jpg'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True)

            # 4) Run RIFE on remote GPU
            print(f"Remote RIFE: interpolating {video.total_frames} -> {target_frames} frames on GPU {rc['gpu']}...", flush=True)
            if progress_cb:
                progress_cb(0, f"Remote RIFE: AI interpolating on GPU")
            rife_cmd = (
                f'{rc["bin"]} -i {r_in} -o {r_out} '
                f'-m {rc["model"]} '
                f'-n {target_frames} '
                f'-g {rc["gpu"]} '
                f'-j 1:4:4 '
                f'-f %08d.jpg'
            )
            proc = subprocess.run(
                ['ssh', host, rife_cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise RuntimeError(f"Remote RIFE failed: {proc.stderr.decode(errors='replace')[:500]}")

            # 5) Encode on remote
            print(f"Remote RIFE: encoding on remote...", flush=True)
            if progress_cb:
                progress_cb(0, f"Remote RIFE: encoding result")
            enc_cmd = (
                f'C:/tools/ffmpeg.exe -y -framerate {target_fps} '
                f'-i {r_out}/%08d.jpg '
                f'-c:v libx264 -preset fast -crf 18 '
                f'-pix_fmt yuv420p -an {r_result}'
            )
            subprocess.run(
                ['ssh', host, enc_cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True)

            # 6) SCP result back
            print(f"Remote RIFE: downloading result...", flush=True)
            if progress_cb:
                progress_cb(0, f"Remote RIFE: downloading result")
            subprocess.run(
                ['scp', '-q', f'{host}:{r_result.replace(chr(92), "/")}', cached_path],
                check=True)

            # 7) Clean up remote
            subprocess.run(
                ['ssh', host, f'rmdir /s /q {r_job}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False)

        except Exception as e:
            print(f"Remote RIFE failed: {e}", flush=True)
            print("Falling back to local processing...", flush=True)
            # Fallback to local RIFE or framerate interpolation
            base = os.path.dirname(os.path.abspath(__file__))
            rife_bin = os.path.join(
                base, 'rife-ncnn-vulkan',
                'rife-ncnn-vulkan-20221029-macos', 'rife-ncnn-vulkan')
            rife_model = os.path.join(
                base, 'rife-ncnn-vulkan',
                'rife-ncnn-vulkan-20221029-macos', 'rife-v4.6')
            if os.path.exists(rife_bin):
                return self._rife_interpolate(
                    video_idx, rife_bin, rife_model, cached_path, progress_cb)
            else:
                return self._framerate_interpolate(
                    video_idx, cached_path, progress_cb)

        cap = cv2.VideoCapture(cached_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        cap.release()
        result = (cached_path, fps)
        self._interp_cache[video_idx] = result
        if progress_cb:
            progress_cb(0, f"Remote RIFE done: {video.filename}")
        print(f"Remote RIFE: done -> {cached_path}", flush=True)
        return result

    def _framerate_interpolate(self, video_idx: int,
                               cached_path: str,
                               progress_cb: Optional[Callable] = None
                               ) -> Tuple[str, float]:
        """Fallback: ffmpeg framerate filter (fast but lower quality)."""
        video = self.videos[video_idx]
        if progress_cb:
            progress_cb(0, f"Interpolating {video.filename} (->60fps)")

        ffmpeg = self._get_ffmpeg()
        if platform.system() == "Darwin":
            enc = ['-c:v', 'h264_videotoolbox', '-b:v', '12M']
        else:
            enc = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']

        cmd = [
            ffmpeg, '-y', '-i', video.path,
            '-vf', 'framerate=fps=60:interp_start=0:interp_end=255:scene=8',
            *enc, '-pix_fmt', 'yuv420p', '-an',
            cached_path,
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            **_SUBPROCESS_EXTRA)
        _, stderr = proc.communicate()

        if proc.returncode != 0:
            print(f"Warning: framerate failed, using original", flush=True)
            result = (video.path, video.fps)
            self._interp_cache[video_idx] = result
            return result

        cap = cv2.VideoCapture(cached_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        cap.release()
        result = (cached_path, fps)
        self._interp_cache[video_idx] = result
        if progress_cb:
            progress_cb(0, f"Interpolation done: {video.filename}")
        return result

    @staticmethod
    def _smart_crop_frame(frame: np.ndarray, center_x: float,
                          target_aspect: float,
                          zoom: float = 1.0) -> np.ndarray:
        """Crop *frame* to *target_aspect* (w/h), following person.

        CROP_ZOOMOUT_V1 — `zoom` ∈ (0, 1]. zoom=1.0 keeps the tight portrait
        crop (crop_w = h*aspect). zoom<1 widens horizontally (crop_w grows),
        clamped to the frame width — used during wide/group shots where the
        target bbox is small so we pull back and keep them in frame.
        """
        h, w = frame.shape[:2]
        if w / h <= target_aspect * 1.1:
            return frame                          # already portrait-ish

        zoom = max(0.3, min(1.0, float(zoom)))
        crop_w = min(int(h * target_aspect / zoom), w)
        cx_px = int(center_x * w)
        left = max(0, min(cx_px - crop_w // 2, w - crop_w))
        return frame[:, left:left + crop_w]

    # ── Rendering ───────────────────────────────────────────────

    @staticmethod
    def _resize_frame(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
        """Resize with black letterbox / pillarbox."""
        h, w = frame.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def render(self, output_path: str, audio_source_idx: int = 0,
               use_external_audio: bool = False,
               output_width: int = 1080, output_height: int = 1920,
               smart_crop: bool = False, crossfade_sec: float = 0.15,
               progress_cb: Optional[Callable] = None):
        """Render the final mashup video to *output_path*."""
        if not self.segments:
            raise ValueError("No segments — run analysis first")

        ffmpeg = self._get_ffmpeg()
        if use_external_audio and self.external_audio:
            audio_src = self.external_audio
        else:
            audio_src = self.videos[audio_source_idx].path
        out_fps = 60.0
        cf = max(0, int(out_fps * crossfade_sec))   # crossfade frame count

        # Pre-process low-fps sources → 60fps (cached, fast ~6s per video)
        render_sources: dict = {}  # video_idx -> (path, actual_fps)
        _seen_vi = set()
        _pending_vi = [s.video_idx for s in self.segments
                       if s.video_idx not in _seen_vi and not _seen_vi.add(s.video_idx)]
        for _i, vi in enumerate(_pending_vi):
            _pg_phase_begin(f"RIFE 60fps V{vi+1} ({_i+1}/{len(_pending_vi)})")
            render_sources[vi] = self._ensure_60fps(vi, progress_cb)
            _pg_phase_end(f"RIFE 60fps V{vi+1}")

        # pre-compute per-segment frame counts using interpolated fps
        seg_meta = []
        total_out = 0
        for seg in self.segments:
            sfps = render_sources[seg.video_idx][1]
            out_n = max(1, int(seg.duration * out_fps))
            seg_meta.append((seg, out_n, sfps))
            total_out += out_n

        # pre-compute crop tracking data when smart_crop is on
        crop_data: dict = {}
        if smart_crop:
            for seg, _, _ in seg_meta:
                vi = seg.video_idx
                if vi not in crop_data:
                    if self.reference_embedding is not None:
                        crop_data[vi] = self._compute_face_crop_centers(vi)
                    else:
                        crop_data[vi] = self._compute_crop_centers(vi)

        # helper: process a raw BGR frame → final output buffer
        def _process(frame, seg, t):
            if smart_crop and crop_data.get(seg.video_idx):
                _cd = crop_data[seg.video_idx]
                # CROP_ZOOMOUT_V1 — tuple may be (centers, pfps) or
                # (centers, pfps, areas)
                if len(_cd) >= 3:
                    centers, cpfps, areas = _cd[0], _cd[1], _cd[2]
                else:
                    centers, cpfps = _cd[0], _cd[1]
                    areas = None
                fidx = t * cpfps
                ci = max(0, min(int(fidx), len(centers) - 1))
                ci_next = min(ci + 1, len(centers) - 1)
                frac = fidx - int(fidx)
                smooth_cx = centers[ci] * (1.0 - frac) + centers[ci_next] * frac
                # CROP_ZOOMOUT_V1 — derive zoom from smoothed bbox area
                zoom = 1.0
                if areas is not None and len(areas):
                    a0 = float(areas[ci]) * (1.0 - frac) + float(areas[ci_next]) * frac
                    # area thresholds tuned for K-pop fancams:
                    #   >= 0.09   full portrait (single subject fills ~30% height)
                    #   0.04-0.09 blend 1.00 → 0.70
                    #   < 0.04    0.55 (very wide group shot)
                    if a0 >= 0.09:
                        zoom = 1.0
                    elif a0 <= 0.04:
                        zoom = 0.55
                    else:
                        t01 = (a0 - 0.04) / (0.09 - 0.04)
                        zoom = 0.70 + (1.00 - 0.70) * t01
                frame = self._smart_crop_frame(
                    frame, smooth_cx,
                    output_width / output_height,
                    zoom=zoom)
            return self._resize_frame(frame, output_width, output_height)

        # helper: pre-read first `count` processed frames of a segment
        def _read_head(seg, sfps, count):
            rpath = render_sources[seg.video_idx][0]
            cap2 = cv2.VideoCapture(rpath)
            vid_offset = self.audio_offsets.get(seg.video_idx, 0.0)
            seek_time = max(0, seg.start_time + vid_offset)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(seek_time * sfps))
            frames = []
            for fi in range(count):
                ok2, fr2 = cap2.read()
                if not ok2:
                    break
                frames.append(_process(fr2, seg, seg.start_time + fi / out_fps))
            cap2.release()
            return frames

        # launch ffmpeg: raw video from stdin + audio from file → mp4
        # Use VideoToolbox HW encoder on macOS, libx264 elsewhere
        if platform.system() == "Darwin":
            v_enc = ['-c:v', 'h264_videotoolbox', '-b:v', '8M',
                     '-pix_fmt', 'yuv420p']
        else:
            v_enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                     '-pix_fmt', 'yuv420p']
        cmd = [
            ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{output_width}x{output_height}',
            '-r', str(out_fps), '-i', '-',
            '-i', audio_src,
            '-map', '0:v', '-map', '1:a',
            *v_enc,
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest', '-movflags', '+faststart',
            output_path,
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            **_SUBPROCESS_EXTRA,
        )

        written = 0
        try:
            for seg_idx, (seg, out_n, sfps) in enumerate(seg_meta):
                rpath = render_sources[seg.video_idx][0]
                cap = cv2.VideoCapture(rpath)
                vid_offset = self.audio_offsets.get(seg.video_idx, 0.0)
                seek_time = max(0, seg.start_time + vid_offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_time * sfps))

                # pre-read first frames of NEXT segment for crossfade
                next_head = None
                if cf > 0 and seg_idx + 1 < len(seg_meta):
                    ns, ns_n, ns_fps = seg_meta[seg_idx + 1]
                    next_head = _read_head(ns, ns_fps, cf)

                # Nearest-frame resampling: duplicate source frames
                # when output fps > source fps (no blending = no ghosting)
                last_frame = None
                src_read = 0

                for of in range(out_n):
                    needed = int(of * sfps / out_fps)
                    while src_read <= needed:
                        ok, fr = cap.read()
                        if ok:
                            last_frame = fr
                        src_read += 1

                    raw = last_frame

                    if raw is not None:
                        vid_off = self.audio_offsets.get(seg.video_idx, 0.0)
                        t = max(0, seg.start_time + vid_off) + of / out_fps
                        buf = _process(raw, seg, t)

                        # crossfade: blend end of this segment with start of next
                        if next_head and of >= out_n - len(next_head):
                            bi = of - (out_n - len(next_head))
                            alpha = (bi + 1) / (len(next_head) + 1)
                            buf = cv2.addWeighted(
                                buf, 1.0 - alpha,
                                next_head[bi], alpha, 0.0)

                        proc.stdin.write(buf.tobytes())

                    written += 1
                    if progress_cb and written % 5 == 0:
                        progress_cb(written / total_out)

                cap.release()
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                stderr = proc.communicate()[1]
            except ValueError:
                # macOS: flush of closed file — just wait for process
                proc.wait()
                stderr = b""
            if proc.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg error:\n{stderr.decode(errors='replace')}")

        if progress_cb:
            progress_cb(1.0)

    # ── misc ────────────────────────────────────────────────────

    def get_total_duration(self) -> float:
        return min((v.duration for v in self.videos), default=0.0)
