# Dance Mashup Studio

Automatically create multi-source dance mashups from K-pop (or any dance) videos, with smart camera following that tracks a specific person across group-choreography footage.

Given a person's name and a song title, the pipeline searches YouTube, downloads candidate videos, identifies the target person via face recognition, detects musical beats, and cuts between videos on-beat while keeping a 9:16 vertical crop locked on the target — even when they turn their head, are partially occluded, or dance in the middle of a group formation.

## Why this exists

Most dance-mashup tools either:
- Follow the face only, which breaks every time the target turns around or is briefly obscured;
- Follow pose keypoints, which drift onto backup dancers in group choreography;
- Require you to manually mark segments.

This project uses **YOLOv8 + ByteTrack** to track every person's body across frames, then uses face-recognition on a handful of sampled frames per body track to pick out *which* track is the target. Because the crop follows the body (not the face), it survives face-away, head-down, or occlusion frames. When the target isn't in the shot at all, the crop freezes on the last known position instead of drifting onto someone else.

## Features

- **One-click generation** — enter a person name and song title, get a finished mashup.
- **Manual workflow** — upload or search videos, pick a reference face, analyze, render.
- **Beat-synced cutting** — cuts happen on musical beats detected by librosa, with pose-similarity scoring to pick the most visually coherent transition.
- **Face-aware planning** — skips over segments where the target isn't on screen, so you don't cut to a shot that has the wrong member in frame.
- **Smart 9:16 crop** — YOLO+ByteTrack person tracks identified by face-embedding cosine similarity to a reference.
- **Face library** — prebuild reference embeddings for members of a group so the one-click flow can look them up offline.
- **Optional RIFE interpolation** — upscale frame-rate via [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan), locally or on a remote GPU box over SSH.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  web.py   — Flask + SSE, one-click pipeline, REST API          │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  engine.py — DanceMashupEngine                                 │
│    · video search / download (yt-dlp)                          │
│    · face search / library (InsightFace buffalo_l)             │
│    · pose extraction (MediaPipe)                               │
│    · audio / beat detection (librosa)                          │
│    · transition planning (pose + motion + face visibility)     │
│    · smart crop center (face-only fallback path)               │
│    · render (ffmpeg) + optional RIFE interpolation             │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│  person_tracker.py — YOLOv8 + ByteTrack person tracking with   │
│  face-embedding-based target track identification, used as     │
│  the primary smart-crop source.                                │
└────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.9+
- `ffmpeg` on PATH (or let `imageio-ffmpeg` install a bundled binary)
- `yt-dlp` — either on PATH, or drop the binary next to `engine.py`
- A GPU is not required, but speeds up YOLO tracking and frame interpolation significantly.

First run will auto-download:
- YOLOv8n weights (~6 MB) from Ultralytics
- InsightFace `buffalo_l` models (~300 MB)

## Install

```bash
git clone https://github.com/<you>/DanceMashup.git
cd DanceMashup
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For YouTube downloads, put the `yt-dlp` binary next to `engine.py` (or install it via pip/package manager):
```bash
# macOS/Linux
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o yt-dlp
chmod +x yt-dlp
```

## Run

```bash
python web.py
```
Then open http://localhost:5000.

### Environment variables

| Variable | Purpose |
|---|---|
| `DANCEMASHUP_REMOTE_RIFE` | Optional SSH target for remote RIFE interpolation, e.g. `user@gpu-box.local`. Leave unset for local-only. |
| `DANCEMASHUP_RIFE_GPU` | GPU index on the remote machine (default `0`). |

If you enable remote RIFE, the remote machine needs `rife-ncnn-vulkan` installed — defaults assume Windows paths under `C:\tools\rife-ncnn-vulkan\`. Adjust `engine.set_remote_rife(bin_path=..., model_path=..., work_dir=...)` if your layout differs.

## One-click workflow

1. Enter a person name (e.g. `Wonyoung`) and a song title (e.g. `Bang Bang`).
2. The pipeline:
   - looks up (or downloads) reference face embeddings for the person;
   - searches YouTube with multiple query variants (dance practice, fancam, performance, choreography, 안무, 직캠);
   - downloads the top candidates and filters out those where the target face isn't detected;
   - runs pose extraction, beat detection, and person tracking;
   - plans beat-aligned transitions favoring segments where the target is clearly visible;
   - renders a 1080×1920 vertical mashup, with the crop following the target's body track per-frame.

## Manual workflow

The web UI also exposes every step individually: upload video files, search online, pick a reference face from thumbnails, tune thresholds, re-plan segments, and re-render.

## Tuning notes

Key knobs that affect output quality:

**`person_tracker.identify_target_tracks(...)`**
- `sim_thresh` (default 0.42) — minimum cosine similarity between a body track's best sampled face and the reference embedding for the track to count as "target". Lower = more broken tracks in group choreography qualify, but risks backup dancers slipping in.
- `min_frames` (default 12) — minimum track length. Lower helps group-choreography footage where re-identification breaks tracks into short pieces.
- `samples_per_track` (default 10) — how many frames per track to run face recognition on.

**`DanceMashupEngine.plan_transitions(...)`**
- A video-level visibility filter skips source videos where the target appears in <15 % of sampled frames (typical for 6-person group practice where the target is rarely center-frame).

**`DanceMashupEngine._is_face_visible(...)`**
- Per-time-point gate checking whether a ±2 s window around a candidate cut contains a frame where the target face matches the reference at sim ≥ 0.45.

## Project layout

```
DanceMashup/
├── engine.py              # End-to-end engine (search, download, analyze, render)
├── web.py                 # Flask server + SSE
├── person_tracker.py      # YOLOv8 + ByteTrack target identification
├── pose_landmarker.task   # MediaPipe pose model
├── requirements.txt
├── templates/
│   └── index.html         # Single-page UI
└── static/
    └── style.css
```

Runtime-only directories (created on first use, excluded from git):

- `downloads/` — downloaded source videos
- `output/` — rendered mashups
- `audio/` — downloaded audio tracks
- `face_library/` — prebuilt face reference embeddings
- `.cache_tracks/` — YOLO tracking cache keyed by video path + size + mtime

## License

MIT — see [LICENSE](LICENSE).

This project uses third-party models with their own licenses:
- YOLOv8 — [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- InsightFace — [MIT](https://github.com/deepinsight/insightface/blob/master/LICENSE)
- MediaPipe — [Apache 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
- RIFE — [MIT (rife-ncnn-vulkan)](https://github.com/nihui/rife-ncnn-vulkan/blob/master/LICENSE)

Make sure your use of downloaded videos complies with the source platform's terms of service and with applicable copyright law.
