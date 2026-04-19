"""LOGO_DETECT_V1 — static overlay / watermark detection via per-pixel
temporal variance.

Sample ~30 frames uniformly, compute per-pixel stdev across time. Pixels
whose stdev stays near zero are static — the video's logo / caption bar /
broadcaster bug. Run morphology + connected components to turn the mask
into a few rectangular boxes, then filter by size, position and aspect.

Results are cached per source file (path+size+mtime) in
<dir>/.cache_delogo/logo_<hash>.json so the expensive pass runs once.
"""

import os
import json
import hashlib
from typing import List, Dict

import cv2
import numpy as np

CACHE_DIR_NAME = ".cache_delogo"


def _cache_key(video_path: str) -> str:
    try:
        st = os.stat(video_path)
        raw = f"{video_path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        raw = video_path
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def detect_logo_boxes(video_path: str,
                      *,
                      n_samples: int = 30,
                      std_thresh: float = 4.0,
                      min_area_ratio: float = 0.0003,
                      max_area_ratio: float = 0.06,
                      max_boxes: int = 4,
                      use_cache: bool = True,
                      verbose: bool = False) -> List[Dict]:
    """Return list of {"x","y","w","h"} boxes in SOURCE pixel coords.

    An empty list means no confident static overlay was found."""
    parent = os.path.dirname(video_path) or "."
    cache_dir = os.path.join(parent, CACHE_DIR_NAME)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    key = _cache_key(video_path)
    cache_path = os.path.join(cache_dir, f"logo_{key}.json")

    if use_cache and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("boxes"), list):
                return data["boxes"]
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total < 10 or w == 0 or h == 0:
        cap.release()
        return []

    margin = max(1, int(total * 0.05))
    idx_start = margin
    idx_end = max(margin + 1, total - margin)
    n = min(n_samples, max(8, idx_end - idx_start))
    indices = np.linspace(idx_start, idx_end - 1, n).astype(int)

    scale = 480.0 / max(w, h) if max(w, h) > 480 else 1.0
    dw = max(1, int(round(w * scale)))
    dh = max(1, int(round(h * scale)))

    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        if scale != 1.0:
            fr = cv2.resize(fr, (dw, dh), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames) < 8:
        return []

    stack = np.stack(frames).astype(np.float32)
    std = stack.std(axis=0)
    mean = stack.mean(axis=0)

    mask = (std < std_thresh).astype(np.uint8) * 255
    # Reject near-uniform fill (solid black bars, blown-out whites)
    mask[(mean < 8) | (mean > 248)] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    nlab, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    frame_area = dw * dh
    candidates = []
    for i in range(1, nlab):
        x, y, cw, ch, area = stats[i]
        ar = area / frame_area
        if ar < min_area_ratio or ar > max_area_ratio:
            continue
        cx_rel = (x + cw / 2.0) / dw
        cy_rel = (y + ch / 2.0) / dh
        # Central subject region — reject
        if 0.30 < cx_rel < 0.70 and 0.25 < cy_rel < 0.75:
            continue
        aspect = cw / max(1, ch)
        if aspect < 0.15 or aspect > 8.0:
            continue
        candidates.append((ar, x, y, cw, ch))

    candidates.sort(reverse=True)
    candidates = candidates[:max_boxes]

    inv = 1.0 / scale if scale != 0 else 1.0
    pad = 3
    boxes: List[Dict] = []
    for _, x, y, cw, ch in candidates:
        sx = max(0, int(round(x * inv)) - pad)
        sy = max(0, int(round(y * inv)) - pad)
        sw = min(w - sx, int(round(cw * inv)) + 2 * pad)
        sh = min(h - sy, int(round(ch * inv)) + 2 * pad)
        if sw <= 1 or sh <= 1:
            continue
        boxes.append({"x": int(sx), "y": int(sy),
                      "w": int(sw), "h": int(sh)})

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"src": video_path,
                       "samples": len(frames),
                       "std_thresh": std_thresh,
                       "boxes": boxes}, f, indent=2)
    except Exception:
        pass

    if verbose:
        print(f"[logo_detect] {os.path.basename(video_path)}: "
              f"{len(boxes)} box(es) {boxes}")
    return boxes


def delogo_filter_chain(boxes: List[Dict]) -> str:
    """Build an ffmpeg filter chain string from boxes. Empty string if none."""
    if not boxes:
        return ""
    parts = []
    for b in boxes:
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        if w < 2 or h < 2:
            continue
        parts.append(f"delogo=x={x}:y={y}:w={w}:h={h}")
    return ",".join(parts)
