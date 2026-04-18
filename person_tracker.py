"""Person-track–based target tracking for DanceMashup.

Flow:
1. YOLOv8 + ByteTrack → per-frame list of (track_id, bbox) for "person" class.
2. For each track, sample up to N frames where a face is detected inside the
   bbox; get face embedding, compare to reference. A track is "target" if its
   MAX face-similarity across samples meets `target_sim_thresh`.
3. At render time, per-frame crop center = center of target-track bbox
   (picking the highest-confidence target-track if several present).
   When no target track present, crop center is frozen at last known position.

Cache is keyed by video path + size + mtime so re-use is free.
"""
from __future__ import annotations
import os, hashlib, time
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import cv2


_YOLO_MODEL = None  # lazy global


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL


def _cache_key(video_path: str) -> str:
    st = os.stat(video_path)
    h = hashlib.md5(f"{video_path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
    return h[:16]


def compute_tracks(video_path: str,
                   cache_dir: str,
                   conf: float = 0.3,
                   iou: float = 0.5,
                   imgsz: int = 640,
                   device: str = "cpu",
                   progress_cb: Optional[Callable[[float], None]] = None,
                   ) -> Dict[int, np.ndarray]:
    """Return {track_id: array of shape (N, 5) = [frame_idx, x1, y1, x2, y2]}.

    Cached to `{cache_dir}/tracks_{key}.npz`.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(video_path)
    cache_path = os.path.join(cache_dir, f"tracks_{key}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {int(k): data[k] for k in data.files}

    print(f"[person_tracker] YOLO+ByteTrack on {os.path.basename(video_path)[:40]}", flush=True)
    model = _get_yolo()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    results = model.track(
        source=video_path,
        classes=[0], conf=conf, iou=iou,
        tracker="bytetrack.yaml", stream=True, verbose=False,
        device=device, imgsz=imgsz,
    )

    per_track: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    last_pct = -1
    for fi, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            pass
        else:
            ids = r.boxes.id.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                per_track.setdefault(int(tid), []).append((fi, x1, y1, x2, y2))
        if progress_cb and total > 0:
            pct = int(100 * fi / total)
            if pct != last_pct and pct % 5 == 0:
                progress_cb(fi / total)
                last_pct = pct

    out = {tid: np.asarray(rows, dtype=np.float32) for tid, rows in per_track.items()}
    np.savez_compressed(cache_path, **{str(k): v for k, v in out.items()})
    print(f"[person_tracker] tracking done: {len(out)} tracks", flush=True)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def identify_target_tracks(video_path: str,
                           tracks: Dict[int, np.ndarray],
                           face_app,
                           ref_emb: np.ndarray,
                           samples_per_track: int = 10,
                           sim_thresh: float = 0.42,
                           min_frames: int = 12,
                           ) -> Dict[int, float]:
    """For each long-enough track, sample frames and test face match.

    Returns {track_id: best_face_sim} only for tracks where best_sim >= sim_thresh.
    """
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scored: Dict[int, float] = {}
    ref = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)
    n_tracks_long = sum(1 for r in tracks.values() if len(r) >= min_frames)
    print(f"[person_tracker] identify: scoring {n_tracks_long} tracks...", flush=True)
    scanned = 0

    for tid, rows in tracks.items():
        if len(rows) < min_frames:
            continue
        # Pick frames spread across track: prefer largest bboxes
        areas = (rows[:, 3] - rows[:, 1]) * (rows[:, 4] - rows[:, 2])
        order = np.argsort(-areas)
        picks = order[:samples_per_track]
        best = -1.0
        for p in picks:
            fi, x1, y1, x2, y2 = rows[p]
            fi = int(fi)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            # Pad bbox 10% up top for head
            bx1 = int(max(0, x1 - (x2-x1)*0.1))
            by1 = int(max(0, y1 - (y2-y1)*0.15))
            bx2 = int(min(W, x2 + (x2-x1)*0.1))
            by2 = int(min(H, y2 + (y2-y1)*0.05))
            crop = frame[by1:by2, bx1:bx2]
            if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
                continue
            faces = face_app.get(crop)
            for f in faces:
                emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-8)
                sim = float(np.dot(ref, emb))
                if sim > best:
                    best = sim
        if best >= sim_thresh:
            scored[tid] = best
        scanned += 1
        if scanned % 10 == 0:
            print(f"[person_tracker] identify: {scanned}/{n_tracks_long}", flush=True)
    cap.release()
    print(f"[person_tracker] identify done: {len(scored)} target tracks", flush=True)
    return scored


def target_crop_centers(video_path: str,
                        tracks: Dict[int, np.ndarray],
                        target_scores: Dict[int, float],
                        n_samples: int,
                        pose_fps: float,
                        ) -> np.ndarray:
    """Build per-sample normalised crop_x values using target tracks.

    Policy: at each sample_t, among target tracks active, pick the one with
    highest face_score. If none active, FREEZE at last known x (no pose drift).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # For each target track, build a fast lookup: frame -> bbox
    track_tables: Dict[int, Dict[int, np.ndarray]] = {}
    for tid in target_scores:
        rows = tracks[tid]
        track_tables[tid] = {int(r[0]): r[1:] for r in rows}

    centers = np.full(n_samples, 0.5, dtype=np.float32)
    last_x: Optional[float] = None
    for s in range(n_samples):
        sample_t = s / pose_fps
        fi = int(sample_t * fps)
        best_cx = None
        best_score = -1.0
        for tid, score in target_scores.items():
            # Find closest track frame within ±3 frames
            tbl = track_tables[tid]
            for df in (0, -1, 1, -2, 2, -3, 3):
                if fi + df in tbl:
                    bbox = tbl[fi + df]
                    cx = (bbox[0] + bbox[2]) / 2.0 / W
                    # Combine track face-score with "active now" preference
                    if score > best_score:
                        best_score = score
                        best_cx = float(cx)
                    break
        if best_cx is not None:
            centers[s] = best_cx
            last_x = best_cx
        elif last_x is not None:
            centers[s] = last_x
        else:
            centers[s] = 0.5  # first frames before any target: center
    return centers
