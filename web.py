"""
Dance Mashup Studio — Web UI
-----------------------------
Flask backend exposing the DanceMashupEngine via REST + SSE.
"""

import io
import json
import os
import queue
import threading
import time
import uuid

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, send_from_directory)

from engine import DanceMashupEngine, VIDEO_COLORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB upload

engine = DanceMashupEngine()
# DOWNLOAD_ORG_V1 — resolve the download sub-folder from engine target.
def _download_subdir() -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
    import re as _re
    def _safe(x):
        x = _re.sub(r"[\\/:*?\"<>|]+", "_", str(x)).strip().strip(".")
        return x or ""
    g = _safe(getattr(engine, "target_group", None))
    m = _safe(getattr(engine, "target_member", None))
    if g and m:
        d = os.path.join(base, g, m)
    elif g:
        d = os.path.join(base, g, "_group")
    else:
        d = base
    os.makedirs(d, exist_ok=True)
    return d

_remote_rife_host = os.environ.get('REMOTE_RIFE_HOST', '').strip()
if _remote_rife_host:
    engine.set_remote_rife(_remote_rife_host)
_lock = threading.Lock()
_face_candidates = []  # List of (embedding, crop_image) tuples for face picker

# ── Face Library ────────────────────────────────────────────────
_face_library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_library")
_face_library_index = {}

def _load_face_library():
    global _face_library_index
    idx_path = os.path.join(_face_library_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            _face_library_index = json.load(f)
        print(f"Face library loaded: {len(_face_library_index)} entries")
    else:
        print("Face library not found")

_load_face_library()

# ── SSE progress bus ────────────────────────────────────────────
_subscribers: dict[str, queue.Queue] = {}

# Persistent task state — survives page reload
_task_state = {
    "active": False,
    "progress": 0.0,
    "text": "",
    "events": [],      # recent events for replay (max 20)
    "result": None,     # final result (render_done, error, etc.)
    "cancel": False,    # TASK_CANCEL_V1 — flipped by /api/cancel
    "phases": [],       # PROGRESS_HOOK_V1 — list of {name, t0, t1}
    "started_at": 0.0,  # set at task start
}


# AUTOARCHIVE_V2 + ARCHIVE_RENAME_V1 — archive to <group>/<member>/ and
# rename to TikTok-caption style when plan metadata is available.
def _archive_older_outputs(keep_filename: str) -> None:
    """Move every .mp4 in output/ except *keep_filename* (and anything
    younger than 60s) into output/archive/<group>/<member>/, renaming to
    '{Member} mashup [Part{N}] #{member_low} #{Group} #{SONG}.mp4' when
    plan.json metadata is available. Best-effort; never raises."""
    import shutil, re as _re
    try:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.isdir(out_dir):
            return
        arc_root = os.path.join(out_dir, "archive")
        now = time.time()

        def _safe(name: str) -> str:
            s = _re.sub(r"[\\/:*?\"<>|]+", "_", name).strip().strip(".")
            return s or "_unclassified"

        def _alnum(x):
            return _re.sub(r"[^A-Za-z0-9]", "", str(x or ""))

        def _plan_for(mp4_path):
            base = os.path.splitext(os.path.basename(mp4_path))[0]
            parent_base = _re.sub(r"(_tiktok\d+s|_tt\d+_\d+s)$", "", base)
            plan_path = os.path.join(out_dir, parent_base + ".plan.json")
            tt = _re.search(r"_tt(\d+)_(\d+)s$", base)
            part_n = int(tt.group(1)) if tt else 0
            plan = {}
            if os.path.isfile(plan_path):
                try:
                    plan = json.load(open(plan_path, encoding="utf-8"))
                except Exception:
                    plan = {}
            return plan, part_n

        def _dest_for(mp4_path):
            """Return (dest_dir, new_filename)."""
            plan, part_n = _plan_for(mp4_path)
            group  = (plan.get("target_group") or "").strip()
            member = (plan.get("target_member") or "").strip()
            song   = (plan.get("song") or "").strip()
            member_nm  = _alnum(member)
            group_tag  = _alnum(group)
            song_tag   = _alnum(song).upper()
            orig_name  = os.path.basename(mp4_path)
            if not group or not member:
                return os.path.join(arc_root, "_unclassified"), orig_name
            dest_dir = os.path.join(arc_root, _safe(group), _safe(member))
            if member_nm and group_tag and song_tag:
                if part_n > 0:
                    new_name = (f"{member_nm} mashup Part{part_n} "
                                f"#{member_nm.lower()} #{group_tag} #{song_tag}.mp4")
                else:
                    new_name = (f"{member_nm} mashup "
                                f"#{member_nm.lower()} #{group_tag} #{song_tag}.mp4")
            else:
                new_name = orig_name
            return dest_dir, new_name

        def _unique(path):
            if not os.path.exists(path):
                return path
            stem, ext = os.path.splitext(path)
            i = 2
            while True:
                cand = f"{stem} ({i}){ext}"
                if not os.path.exists(cand):
                    return cand
                i += 1

        moved = 0
        for name in os.listdir(out_dir):
            if name == keep_filename or name == "archive":
                continue
            if not name.endswith(".mp4"):
                continue
            src_path = os.path.join(out_dir, name)
            try:
                if not os.path.isfile(src_path):
                    continue
                if now - os.path.getmtime(src_path) < 60:
                    continue
                dest_dir, new_name = _dest_for(src_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_mp4 = _unique(os.path.join(dest_dir, new_name))
                shutil.move(src_path, dest_mp4)
                moved += 1
                # Sidecars: same stem as the renamed mp4
                new_stem = os.path.splitext(os.path.basename(dest_mp4))[0]
                old_stem = os.path.splitext(name)[0]
                for ext in (".plan.json", ".json"):
                    side = os.path.join(out_dir, old_stem + ext)
                    if os.path.isfile(side):
                        try:
                            shutil.move(side, os.path.join(dest_dir, new_stem + ext))
                        except Exception:
                            pass
            except Exception as e:
                print(f"[archive] skip {name}: {e}")
        if moved:
            print(f"[archive] moved {moved} file(s) into {arc_root}/<group>/<member>/ with friendly names")
    except Exception as e:
        print(f"[archive] error: {e}")




# AUTOSLICE_UI_V1 — stream tiktok_slice.py stdout, parse per-segment %,
# and rebroadcast to the web UI via the same `progress` SSE event that
# drives the main progress bar.
def _auto_tiktok_slice(mp4_name: str) -> None:
    import re as _re2
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(app_dir, "output", mp4_name)
        if not os.path.isfile(src):
            return
        script = os.path.join(app_dir, "tiktok_slice.py")
        if not os.path.isfile(script):
            print("[tiktok] tiktok_slice.py not found; skip")
            return
        py = os.path.join(app_dir, "venv", "bin", "python3")
        if not os.path.isfile(py):
            py = "python3"
        import subprocess
        proc = subprocess.Popen(
            [py, "-u", script, src],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        pat_plan = _re2.compile(r"^\[info\] (\d+) segments planned")
        pat_enc  = _re2.compile(r"^\[info\] encoding -> \S+.*segment (\d+)/(\d+)")
        pat_pct  = _re2.compile(r"^\[(\d+)/(\d+)\] \S+\s+(\d+)%")
        pat_done = _re2.compile(r"^\[done\] (\d+) clips")

        total = 0
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print(f"[tiktok] {line}", flush=True)
                m = pat_plan.match(line)
                if m:
                    total = int(m.group(1))
                    _broadcast("progress", {"value": 0.0,
                        "text": f"TikTok 切片: 準備切成 {total} 段..."})
                    continue
                m = pat_pct.match(line)
                if m:
                    i   = int(m.group(1))
                    n   = int(m.group(2))
                    pct = int(m.group(3))
                    frac = ((i - 1) + pct / 100.0) / max(n, 1)
                    _broadcast("progress", {"value": round(frac, 4),
                        "text": f"TikTok 切片 {i}/{n}: {pct}%"})
                    continue
                m = pat_enc.match(line)
                if m:
                    i = int(m.group(1)); n = int(m.group(2))
                    frac = (i - 1) / max(n, 1)
                    _broadcast("progress", {"value": round(frac, 4),
                        "text": f"TikTok 切片 {i}/{n}: 0%"})
                    continue
                m = pat_done.match(line)
                if m:
                    n = int(m.group(1))
                    _broadcast("progress", {"value": 1.0,
                        "text": f"TikTok 切片完成: {n} 段"})
                    _broadcast("tiktok_done",
                               {"filename": mp4_name, "count": n})
                    continue
        finally:
            try:
                proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.kill()
                _broadcast("error_event", {"text": "TikTok 切片 timeout"})
        if proc.returncode not in (0, None):
            _broadcast("error_event",
                       {"text": f"TikTok 切片失敗 rc={proc.returncode}"})
    except Exception as e:
        print(f"[tiktok] error: {e}")
        try:
            _broadcast("error_event", {"text": f"TikTok 切片錯誤: {e}"})
        except Exception:
            pass


def _broadcast(event: str, data: dict):
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    # Store in task state for reconnection
    if event == "progress":
        _task_state["progress"] = data.get("value", 0)
        _task_state["text"] = data.get("text", "")
    elif event in ("render_done", "error_event"):
        _task_state["result"] = {"event": event, "data": data}
        if event == "render_done":
            _task_state["active"] = False
            # AUTOARCHIVE_V1 — keep latest, shuffle the rest to output/archive/<date>/
            try:
                _fn = data.get("filename") if isinstance(data, dict) else None
                if _fn:
                    threading.Thread(
                        target=_archive_older_outputs, args=(_fn,), daemon=True
                    ).start()
                    # AUTOSLICE_V1 — also slice into consecutive 30-60s clips
                    threading.Thread(
                        target=_auto_tiktok_slice, args=(_fn,), daemon=True
                    ).start()
            except Exception as _e:
                print(f"[archive] spawn failed: {_e}")
    else:
        evts = _task_state["events"]
        evts.append({"event": event, "data": data})
        if len(evts) > 20:
            _task_state["events"] = evts[-20:]
    dead = []
    for sid, q in list(_subscribers.items()):
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(sid)
    for sid in dead:
        _subscribers.pop(sid, None)


def _progress(value: float, text: str = ""):
    _broadcast("progress", {"value": round(value, 4), "text": text})


# PROGRESS_HOOK_V1 — phase tracking
def _phase_begin(name: str):
    ph = _task_state.get("phases") or []
    # finalize previous
    if ph and ph[-1].get("t1") is None:
        ph[-1]["t1"] = time.time()
    ph.append({"name": name, "t0": time.time(), "t1": None})
    _task_state["phases"] = ph
    _broadcast("phase", {"phases": ph})


def _phase_end(_name: str = ""):
    ph = _task_state.get("phases") or []
    if ph and ph[-1].get("t1") is None:
        ph[-1]["t1"] = time.time()
    _task_state["phases"] = ph
    _broadcast("phase", {"phases": ph})


# Wire hooks into engine module so inner stages emit progress
def _engine_progress_tick(text, pct=None):
    if pct is not None:
        try:
            cur = float(_task_state.get("progress") or 0.0)
        except Exception:
            cur = 0.0
        _broadcast("progress", {"value": cur, "text": text, "sub_pct": round(pct, 4)})
    else:
        cur = float(_task_state.get("progress") or 0.0)
        _broadcast("progress", {"value": cur, "text": text})


def _engine_phase(action, name):
    if action == "begin":
        _phase_begin(name)
    else:
        _phase_end(name)


try:
    import engine as _engine_mod
    _engine_mod.set_progress_hook(_engine_progress_tick)
    _engine_mod.set_phase_hook(_engine_phase)
except Exception as _e:
    print(f"[warn] engine hook wiring failed: {_e}", flush=True)


# ── Pages ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/task_status")
def task_status():
    """Return current task state for reconnection."""

    return jsonify(_task_state)


# TASK_CANCEL_V1 + TASK_CANCEL_SWEEP_V1 — stop job, kill children,
# conservatively remove half-written files (keep reusable downloads).
@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    was_active = bool(_task_state.get("active"))
    started_at = float(_task_state.get("started_at") or 0.0)
    _task_state["cancel"] = True
    _task_state["active"] = False
    import subprocess as _sub
    killed = 0
    try:
        r = _sub.run(["pkill", "-TERM", "-P", str(os.getpid())],
                     stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
        killed = 1 if r.returncode == 0 else 0
    except Exception:
        pass
    # Give children a beat to release file handles before we sweep.
    time.sleep(0.4)

    app_dir = os.path.dirname(os.path.abspath(__file__))
    dl_dir  = os.path.join(app_dir, "downloads")
    out_dir = os.path.join(app_dir, "output")
    removed = []

    def _safe_unlink(path):
        try:
            os.remove(path)
            removed.append(os.path.relpath(path, app_dir))
        except Exception:
            pass

    # Garbage extensions anywhere under downloads/
    GARBAGE = (".part", ".ytdl", ".tmp")
    try:
        for root, _, files in os.walk(dl_dir):
            for name in files:
                low = name.lower()
                if low.endswith(GARBAGE) or ".frag" in low:
                    _safe_unlink(os.path.join(root, name))
    except FileNotFoundError:
        pass

    # Half-written mashup outputs (< 1 MB, touched after task start)
    try:
        cutoff = started_at if started_at > 0 else 0
        for name in os.listdir(out_dir):
            if not name.startswith("mashup_") or not name.endswith(".mp4"):
                continue
            full = os.path.join(out_dir, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if st.st_size < 1_048_576 and st.st_mtime >= cutoff:
                _safe_unlink(full)
    except FileNotFoundError:
        pass

    _broadcast("progress", {"value": 0, "text": "已中止"})
    if was_active:
        _broadcast("error_event", {"text": "使用者中止工作"})
    return jsonify(cancelled=True, was_active=was_active,
                   killed=killed, removed=removed)


@app.route("/api/events")
def events():
    """SSE endpoint for real-time progress."""
    sid = str(uuid.uuid4())
    q: queue.Queue = queue.Queue(maxsize=200)
    _subscribers[sid] = q

    def stream():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.pop(sid, None)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Video management ────────────────────────────────────────────

@app.route("/api/videos")
def list_videos():
    vids = []
    for i, v in enumerate(engine.videos):
        vids.append({
            "idx": i,
            "filename": v.filename,
            "duration": round(v.duration, 1),
            "fps": round(v.fps, 1),
            "width": v.width,
            "height": v.height,
            "has_poses": v.poses is not None,
            "source_url": getattr(v, "source_url", ""),
        })
    return jsonify(vids)


@app.route("/api/videos/upload", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file"), 400
    dl_dir = _download_subdir()   # DOWNLOAD_ORG_V1
    path = os.path.join(dl_dir, f.filename)
    f.save(path)
    with _lock:
        v = engine.add_video(path)
    return jsonify(idx=len(engine.videos) - 1, filename=v.filename,
                   duration=round(v.duration, 1), fps=round(v.fps, 1),
                   width=v.width, height=v.height)


@app.route("/api/videos/<int:idx>", methods=["DELETE"])
def remove_video(idx):
    with _lock:
        engine.remove_video(idx)
    return jsonify(ok=True)


@app.route("/api/videos/clear", methods=["POST"])
def clear_videos():
    with _lock:
        engine.videos.clear()
        engine.segments.clear()
        engine.beat_times = None
    return jsonify(ok=True)


@app.route("/api/videos/thumbnail/<int:idx>")
def video_thumbnail(idx):
    t = engine.get_thumbnail(idx, max_w=240, max_h=420)
    if t is None:
        return "", 404
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ── Video search & download ────────────────────────────────────

@app.route("/api/videos/search", methods=["POST"])
def search_videos():
    d = request.json or {}
    q = d.get("query", "").strip()
    pf = d.get("platform", "youtube")
    if not q:
        return jsonify(error="empty query"), 400
    try:
        results = engine.search_videos(q, platform=pf, max_results=12)
        return jsonify(results)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/videos/download", methods=["POST"])
def download_videos():
    """Download selected videos (runs in background, progress via SSE)."""
    d = request.json or {}
    items = d.get("items", [])
    if not items:
        return jsonify(error="nothing selected"), 400

    def _work():
        dl_dir = _download_subdir()   # DOWNLOAD_ORG_V1
        total = len(items)
        ok_count, skipped = 0, 0
        for idx, item in enumerate(items):
            url = item.get("url", "")
            title = item.get("title", "video")
            vid_id = item.get("id", "")

            # thumbnail face check
            if engine.reference_embedding is not None and vid_id:
                _progress((idx) / total, f"Verifying face: {title}")
                pf = "bilibili" if "bilibili" in url else "youtube"
                match, score = engine.verify_face_by_thumbnail(vid_id, platform=pf)
                if not match:
                    _progress((idx + 1) / total,
                              f"Face mismatch ({score:.2f}), skipped: {title}")
                    skipped += 1
                    continue

            _progress(idx / total, f"Downloading {idx+1}/{total}: {title}")
            try:
                path = engine.download_video(
                    url, dl_dir,
                    progress_cb=lambda p, i=idx: _progress(
                        (i + p) / total,
                        f"Downloading {i+1}/{total}: {title}"))
                with _lock:
                    engine.add_video(path)
                    if engine.videos:
                        engine.videos[-1].source_url = url
                ok_count += 1
            except Exception as e:
                _broadcast("error", {"text": f"Download failed: {title} — {e}"})

        msg = f"Done! {ok_count}/{total} downloaded"
        if skipped:
            msg += f", {skipped} skipped (face mismatch)"
        _progress(1.0, msg)
        _broadcast("download_done", {"ok": ok_count, "skipped": skipped})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True)


# ── External audio ─────────────────────────────────────────────

@app.route("/api/audio/search", methods=["POST"])
def search_audio():
    d = request.json or {}
    q = d.get("query", "").strip()
    if not q:
        return jsonify(error="empty query"), 400

    def _work():
        try:
            _progress(0, f"Searching audio: {q}")
            _, title = engine.download_audio(
                q, progress_cb=lambda p: _progress(p, f"Downloading audio: {q}"))
            _progress(1.0, f"Audio ready: {title}")
            _broadcast("audio_done", {"title": title})
        except Exception as e:
            _broadcast("error", {"text": f"Audio failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True)


@app.route("/api/audio/upload", methods=["POST"])
def upload_audio():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file"), 400
    dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
    os.makedirs(dl_dir, exist_ok=True)
    path = os.path.join(dl_dir, f.filename)
    f.save(path)
    engine.set_external_audio(path)
    return jsonify(title=f.filename)


@app.route("/api/audio/info")
def audio_info():
    return jsonify(
        has_external=engine.external_audio is not None,
        title=engine.external_audio_title,
    )


# ── Reference face ─────────────────────────────────────────────

@app.route("/api/face/search", methods=["POST"])
def search_face():
    d = request.json or {}
    name = d.get("name", "").strip()
    if not name:
        return jsonify(error="empty name"), 400

    # Check face library first (instant)
    q = name.lower()
    for key, info in _face_library_index.items():
        full = f"{info['group']} {info['member']}".lower()
        if q in full or q in info["member"].lower():
            emb_path = os.path.join(_face_library_dir, info["path"], "embedding.npy")
            if os.path.exists(emb_path):
                engine.reference_embedding = np.load(emb_path)
                engine.target_group = info["group"]   # AUTOARCHIVE_V2
                engine.target_member = info["member"]
                _broadcast("face_done", {"ok": True, "source": "library",
                                          "member": info["member"], "group": info["group"]})
                return jsonify(started=False, found=True,
                               member=info["member"], group=info["group"],
                               source="library")

    # Fallback to online search
    def _work():
        try:
            engine.search_reference_face(
                name,
                progress_cb=lambda p, msg: _progress(p, msg))
            _progress(1.0, "Reference face ready")
            _broadcast("face_done", {"ok": True})
        except Exception as e:
            _broadcast("error", {"text": f"Face search failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True)


@app.route("/api/face/upload", methods=["POST"])
def upload_face():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file"), 400
    # Save to temp, then use engine's file-based method
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        f.save(tmp.name)
        tmp.close()
        engine.set_reference_face(tmp.name)
        engine.target_group = None   # AUTOARCHIVE_V2: custom upload
        engine.target_member = None
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(error=str(e)), 400
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/api/face/thumbnail")
def face_thumbnail():
    if engine.reference_embedding is None:
        return "", 404
    # Return a placeholder — we don't store the crop permanently.
    # Re-search would be needed for actual image, but let's return status only.
    return "", 204


@app.route("/api/face/clear", methods=["POST"])
def clear_face():
    engine.clear_reference_face()
    engine.target_group = None   # AUTOARCHIVE_V2
    engine.target_member = None
    return jsonify(ok=True)


@app.route("/api/face/status")
def face_status():
    return jsonify(has_face=engine.reference_embedding is not None)


# ── Analysis ───────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if len(engine.videos) < 2:
        return jsonify(error="Need at least 2 videos"), 400

    d = request.json or {}
    min_dur = d.get("min_dur", 3.0)
    max_dur = d.get("max_dur", 8.0)
    sim_thresh = d.get("sim_thresh", 0.3)
    audio_idx = d.get("audio_idx", 0)
    use_external = d.get("use_external", False)

    def _work():
        try:
            n = len(engine.videos)
            for i in range(n):
                nm = engine.videos[i].filename
                _progress(i / n * 0.6, f"Analyzing poses {i+1}/{n}: {nm}")
                engine.analyze_poses(
                    i, sample_fps=10.0,
                    progress_cb=lambda p, _i=i: _progress(
                        (_i + p) / n * 0.6,
                        f"Analyzing poses {_i+1}/{n}"))

            _progress(0.65, "Detecting beats...")
            engine.detect_beats(
                audio_idx if not use_external else 0,
                use_external=use_external,
                progress_cb=lambda p: _progress(0.65 + p * 0.15,
                                                "Detecting beats..."))

            _progress(0.85, "Planning transitions...")
            engine.plan_transitions(min_dur, max_dur, sim_thresh)

            ns = len(engine.segments)
            _progress(1.0, f"Analysis done! {ns} segments")
            _broadcast("analysis_done", {"segments": ns})
        except Exception as e:
            _broadcast("error", {"text": f"Analysis failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True)


# ── Replan ─────────────────────────────────────────────────────

@app.route("/api/replan", methods=["POST"])
def replan():
    d = request.json or {}
    segs = engine.plan_transitions(
        d.get("min_dur", 3.0), d.get("max_dur", 8.0), d.get("sim_thresh", 0.3))
    return jsonify(segments=len(segs))


# ── Segments / timeline ────────────────────────────────────────

@app.route("/api/segments")
def segments():
    total = engine.get_total_duration()
    segs = []
    for s in engine.segments:
        segs.append({
            "video_idx": s.video_idx,
            "start": round(s.start_time, 3),
            "end": round(s.end_time, 3),
            "similarity": round(s.similarity, 3),
        })
    return jsonify(segments=segs, total_duration=round(total, 2),
                   colors=VIDEO_COLORS[:len(engine.videos)])


# ── Render ─────────────────────────────────────────────────────

@app.route("/api/cleanup_rife", methods=["POST"])
def cleanup_rife():
    """Manually kill stray rife-ncnn-vulkan.exe on the remote host."""
    ok = engine.cleanup_remote_rife(quiet=False)
    return jsonify(ok=ok)


@app.route("/api/render", methods=["POST"])
def render():
    if not engine.segments:
        return jsonify(error="No segments — run analysis first"), 400

    d = request.json or {}
    w = d.get("width", 1080)
    h = d.get("height", 1920)
    smart_crop = d.get("smart_crop", True)
    crossfade = d.get("crossfade", 0.2)
    audio_idx = d.get("audio_idx", 0)
    use_external = d.get("use_external", False)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"mashup_{int(time.time())}.mp4"
    out_path = os.path.join(out_dir, out_name)

    def _work():
        try:
            _progress(0, "Rendering...")
            engine.render(
                out_path, audio_idx,
                use_external_audio=use_external,
                output_width=w, output_height=h,
                smart_crop=smart_crop,
                crossfade_sec=crossfade,
                progress_cb=lambda p, msg="": _progress(p, f"Rendering {p*100:.0f}%"))
            sz = os.path.getsize(out_path) / 1024 / 1024
            _progress(1.0, f"Done! {sz:.1f} MB")
            # Save plan JSON alongside the mashup so the UI can offer swaps.
            try:
                engine.export_plan_json(out_path)
            except Exception as pe:
                print(f"[plan] export failed: {pe}", flush=True)
            _task_state["active"] = False
            _broadcast("render_done", {"filename": out_name, "size_mb": round(sz, 1)})
        except Exception as e:
            _broadcast("error", {"text": f"Render failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True, filename=out_name)




# TIKTOK_CLIPS_UI_V1 + TIKTOK_DLNAME_V1 — list clips with a friendly
# download-as filename built from plan.json (group/member/song).
@app.route("/api/tiktok_clips/<path:filename>")
def api_tiktok_clips(filename):
    import re as _re3, json as _json3
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    base = os.path.splitext(os.path.basename(filename))[0]

    def _safe(x):
        x = _re3.sub(r"[\\/:*?\"<>|]+", "_", str(x))
        x = _re3.sub(r"\s+", " ", x).strip().strip(".")
        return x

    # Derive <group>_<member>_<song> from plan.json, with fallbacks
    dl_base = ""
    plan = {}
    plan_path = os.path.join(out_dir, base + ".plan.json")
    if os.path.isfile(plan_path):
        try:
            with open(plan_path, encoding="utf-8") as _f:
                plan = _json3.load(_f)
            g    = _safe(plan.get("target_group")  or "")
            m    = _safe(plan.get("target_member") or "")
            song = _safe(plan.get("song") or "")
            # Fallback: mine quoted title from first video filename
            if not song:
                vids = plan.get("videos") or []
                if vids:
                    raw = vids[0].get("filename") or ""
                    mm = _re3.search(r"[\u2018\u2019\'\"]([^\u2018\u2019\'\"]+)[\u2018\u2019\'\"]", raw)
                    if mm:
                        song = _safe(mm.group(1))
            parts = [p for p in (g, m, song) if p]
            if parts:
                dl_base = "_".join(parts)
        except Exception:
            pass

    pat = _re3.compile(r"^" + _re3.escape(base) + r"_tt(\d+)_(\d+)s\.mp4$")
    clips = []
    try:
        for name in os.listdir(out_dir):
            mm = pat.match(name)
            if not mm:
                continue
            full = os.path.join(out_dir, name)
            idx, secs = int(mm.group(1)), int(mm.group(2))
            # TIKTOK_DLNAME_V2 — TikTok caption-style filename
            def _alnum(x):
                return _re3.sub(r"[^A-Za-z0-9]", "", str(x or ""))
            member_nm  = _alnum(plan.get("target_member", ""))
            group_tag  = _alnum(plan.get("target_group", ""))
            song_tag   = _alnum(plan.get("song", "")).upper()
            # song fallback: derived quoted title from plan mining (already in dl_base when present)
            if not song_tag and dl_base:
                # dl_base was g_m_song; try last part
                try:
                    song_tag = _alnum(dl_base.split("_")[-1]).upper()
                except Exception:
                    pass
            if member_nm and group_tag and song_tag:
                dl_name = (f"{member_nm} mashup Part{idx} "
                           f"#{member_nm.lower()} #{group_tag} #{song_tag}.mp4")
            else:
                dl_name = (f"{dl_base}_tt{idx:02d}_{secs}s.mp4"
                           if dl_base else name)
            clips.append({
                "name": name,
                "idx":  idx,
                "seconds": secs,
                "size_mb": round(os.path.getsize(full) / 1048576, 1),
                "url": "/output/" + name,
                "download_as": dl_name,
            })
    except FileNotFoundError:
        pass
    clips.sort(key=lambda c: c["idx"])
    return jsonify(clips=clips)

@app.route("/output/<path:filename>")
def serve_output(filename):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    return send_from_directory(out_dir, filename)


@app.route("/api/songs/refresh", methods=["POST"])
def api_songs_refresh():
    """Run build_song_library.py and return the new summary."""
    import subprocess, json as _json
    app_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(app_dir, "build_song_library.py")
    if not os.path.exists(script):
        return jsonify(error="build_song_library.py not found"), 500
    try:
        r = subprocess.run(
            ["python3", script],
            cwd=app_dir, timeout=300, capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return jsonify(error="builder timed out after 300s"), 504
    if r.returncode != 0:
        return jsonify(error="builder failed", stderr=r.stderr[-2000:]), 500
    idx = os.path.join(app_dir, "song_library", "index.json")
    try:
        data = _json.load(open(idx, encoding="utf-8"))
    except Exception as e:
        return jsonify(error=f"post-build read failed: {e}"), 500
    summary = {g: {
        "albums":      len(v.get("albums", [])),
        "mini_albums": len(v.get("mini_albums", [])),
        "singles":     len(v.get("singles", [])),
    } for g, v in data.items()}
    return jsonify(ok=True, summary=summary, groups=data)


@app.route("/api/songs")
def api_songs():
    """Return the full song library, or filter by ?group=<group>."""
    import json as _json
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "song_library")
    idx = os.path.join(lib_dir, "index.json")
    if not os.path.exists(idx):
        return jsonify(groups={})
    try:
        data = _json.load(open(idx, encoding="utf-8"))
    except Exception as e:
        return jsonify(error=f"song_library read failed: {e}"), 500
    g = request.args.get("group")
    if g:
        return jsonify(group=g, data=data.get(g, {}))
    return jsonify(groups=data)


@app.route("/api/groups")
def api_groups():
    """Return {group: [members_sorted]} built from face_library/index.json."""
    import json as _json
    from collections import defaultdict as _dd
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_library")
    idx_path = os.path.join(lib_dir, "index.json")
    if not os.path.exists(idx_path):
        return jsonify(groups={})
    try:
        with open(idx_path, encoding="utf-8") as f:
            data = _json.load(f)
    except Exception as e:
        return jsonify(error=f"index.json read failed: {e}"), 500
    by_group = _dd(list)
    for v in data.values():
        g = v.get("group") or ""
        m = v.get("member") or ""
        if g and m:
            by_group[g].append(m)
    out = {g: sorted(set(ms), key=str.lower) for g, ms in by_group.items()}
    return jsonify(groups=out)


# ── Plan JSON + segment swap ──────────────────────────────────

@app.route("/api/plan/<path:filename>")
def api_plan(filename):
    """Return the plan JSON saved alongside a rendered mashup."""
    import json as _json
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    base, _ = os.path.splitext(filename)
    plan_path = os.path.join(out_dir, base + ".plan.json")
    if not os.path.exists(plan_path):
        return jsonify(error="No plan for this mashup"), 404
    try:
        with open(plan_path, encoding="utf-8") as f:
            return jsonify(_json.load(f))
    except Exception as e:
        return jsonify(error=f"Read plan failed: {e}"), 500


@app.route("/api/swap_segment", methods=["POST"])
def api_swap_segment():
    """Re-render a mashup with one or more segment swaps.

    Body: {"mashup": "mashup_xxx.mp4",
           "swaps": [{"seg_idx": 5, "video_idx": 2}, ...],
           "smart_crop": true, "use_external": true, "crossfade": 0.15}

    Requires the engine still has the analyzed videos from the source plan
    loaded in memory (same session). Otherwise returns 409.
    """
    import json as _json
    d = request.json or {}
    mashup = d.get("mashup")
    swaps = d.get("swaps") or []
    if not mashup or not swaps:
        return jsonify(error="mashup + swaps required"), 400

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    base, _ = os.path.splitext(mashup)
    plan_path = os.path.join(out_dir, base + ".plan.json")
    if not os.path.exists(plan_path):
        return jsonify(error="No plan JSON for this mashup — re-render first"), 404

    with open(plan_path, encoding="utf-8") as f:
        plan = _json.load(f)

    # Verify engine still holds the videos from this plan (match by filename).
    engine_names = {v.filename for v in engine.videos}
    plan_names = {v["filename"] for v in plan.get("videos", [])}
    if not plan_names.issubset(engine_names):
        missing = sorted(plan_names - engine_names)
        return jsonify(error="Engine state mismatch — "
                             "re-run analysis. Missing: " + ", ".join(missing)), 409

    # Map plan video_idx -> engine video_idx (by filename)
    eng_idx_by_name = {v.filename: i for i, v in enumerate(engine.videos)}
    plan_to_engine = {}
    for v in plan["videos"]:
        plan_to_engine[int(v["idx"])] = eng_idx_by_name[v["filename"]]

    # Apply swaps onto the plan copy
    segs = list(plan["segments"])
    applied = []
    for sw in swaps:
        si = int(sw["seg_idx"])
        nvi = int(sw["video_idx"])
        if si < 0 or si >= len(segs):
            return jsonify(error=f"seg_idx {si} out of range"), 400
        segs[si] = dict(segs[si])
        segs[si]["video_idx"] = nvi
        applied.append({"seg_idx": si, "video_idx": nvi})
    plan["segments"] = segs

    # Translate plan video_idx -> engine video_idx for segments
    for s in segs:
        s["video_idx"] = plan_to_engine[int(s["video_idx"])]

    # Launch re-render
    smart_crop = bool(d.get("smart_crop", True))
    crossfade = float(d.get("crossfade", 0.15))
    use_external = bool(d.get("use_external", True))
    out_name = f"mashup_{int(time.time())}.mp4"
    out_path = os.path.join(out_dir, out_name)

    if _task_state.get("active"):
        return jsonify(error="Another task is running"), 409
    _task_state["active"] = True
    _task_state["progress"] = 0.0
    _task_state["text"] = "Swap re-render..."
    _task_state["events"] = []
    _task_state["result"] = None
    _task_state["cancel"] = False
    _task_state["started_at"] = time.time()  # TASK_CANCEL_SWEEP_V1

    def _work():
        try:
            engine.apply_plan_segments(plan)
            _progress(0.05, f"Swap rerender ({len(applied)} swap(s))...")
            engine.render(
                out_path, 0,
                use_external_audio=use_external,
                output_width=1080, output_height=1920,
                smart_crop=smart_crop,
                crossfade_sec=crossfade,
                progress_cb=lambda p, msg="": _progress(
                    0.05 + p * 0.9, f"Rendering {p*100:.0f}%"))
            try:
                engine.export_plan_json(out_path)
            except Exception as pe:
                print(f"[plan] export failed: {pe}", flush=True)
            sz = os.path.getsize(out_path) / 1024 / 1024
            _progress(1.0, f"Done! {sz:.1f} MB")
            _task_state["active"] = False
            _broadcast("render_done",
                       {"filename": out_name, "size_mb": round(sz, 1),
                        "source": mashup, "swaps": applied})
        except Exception as e:
            import traceback
            traceback.print_exc()
            _task_state["active"] = False
            _broadcast("error_event", {"text": f"Swap render failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True, filename=out_name, swaps=applied)


# ── Serve downloaded videos for preview ────────────────────────

@app.route("/downloads/<path:filename>")
def serve_download(filename):
    # DOWNLOAD_ORG_V1 — if filename is a bare basename that has been
    # filed under a sub-folder, walk the tree to resolve it once.
    dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
    direct = os.path.join(dl_dir, filename)
    if not os.path.isfile(direct) and "/" not in filename and "\\" not in filename:
        for root, _, files in os.walk(dl_dir):
            if filename in files:
                return send_from_directory(root, filename)
    return send_from_directory(dl_dir, filename)




# ── Face Library API ────────────────────────────────────────────

@app.route("/api/face/library")
def face_library_list():
    """List all members in the face library."""
    _load_face_library()
    groups = {}
    for key, info in sorted(_face_library_index.items()):
        g = info["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append({
            "key": key,
            "member": info["member"],
            "group": g,
        })
    return jsonify(groups=groups, total=len(_face_library_index))


@app.route("/api/face/library/thumbnail/<path:key>")
def face_library_thumbnail(key):
    """Return the reference face image for a library member."""
    info = _face_library_index.get(key)
    if not info:
        return "", 404
    ref_path = os.path.join(_face_library_dir, info["path"], "reference.jpg")
    if not os.path.exists(ref_path):
        return "", 404
    return send_file(ref_path, mimetype="image/jpeg")


@app.route("/api/face/library/select", methods=["POST"])
def face_library_select():
    """Select a library member as the reference face."""
    d = request.json or {}
    key = d.get("key", "")
    info = _face_library_index.get(key)
    if not info:
        return jsonify(ok=False, error=f"Not found: {key}"), 404
    # IDENTITY_POOL_V1 — prefer multi-ref pool when it exists
    pool_path = os.path.join(_face_library_dir, info["path"], "embeddings.npy")
    emb_path  = os.path.join(_face_library_dir, info["path"], "embedding.npy")
    if os.path.exists(pool_path):
        pool = np.load(pool_path)
    elif os.path.exists(emb_path):
        pool = np.load(emb_path)
    else:
        return jsonify(ok=False, error="Embedding not found"), 404
    # Peer negatives: other members of the same group
    negs = []
    g = info["group"]
    for k, i in _face_library_index.items():
        if i["group"] == g and k != key:
            p = os.path.join(_face_library_dir, i["path"], "embedding.npy")
            if os.path.exists(p):
                v = np.load(p)
                if v.ndim == 1:
                    v = v[None, :]
                negs.append(v)
    neg_pool = np.concatenate(negs, axis=0) if negs else None
    engine._set_ref_pool(pool, neg_pool)
    engine.target_group = info["group"]   # AUTOARCHIVE_V2
    engine.target_member = info["member"]
    n_refs = 1 if pool.ndim == 1 else int(pool.shape[0])
    n_negs = 0 if neg_pool is None else int(neg_pool.shape[0])
    return jsonify(ok=True, group=info["group"], member=info["member"],
                   n_refs=n_refs, n_negatives=n_negs)


@app.route("/api/face/library/search")
def face_library_search():
    """Search face library by name (fuzzy match)."""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify(results=[])
    results = []
    for key, info in _face_library_index.items():
        name = f"{info['group']} {info['member']}".lower()
        if q in name or q in info["member"].lower() or q in info["group"].lower():
            results.append({
                "key": key,
                "group": info["group"],
                "member": info["member"],
            })
    return jsonify(results=results)


# ── Face detection from videos ──────────────────────────────────

@app.route("/api/face/detect_from_videos", methods=["POST"])
def detect_faces_from_videos():
    """Detect distinct faces from loaded videos for user to pick."""
    global _face_candidates
    if not engine.videos:
        return jsonify(error="No videos loaded", faces=[])

    try:
        face_app = engine._get_face_app()
    except Exception as e:
        return jsonify(error=f"Face detection init failed: {e}", faces=[])

    all_faces = []  # List of (embedding, crop_bgr)

    for vi, video in enumerate(engine.videos):
        cap = cv2.VideoCapture(video.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = video.duration

        # Sample 5 evenly spaced frames
        for t in [duration * p for p in [0.1, 0.25, 0.4, 0.6, 0.8]]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = face_app.get(frame)
            for face in faces:
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                b = face.bbox.astype(int)
                h, w = frame.shape[:2]
                pad = int((b[2] - b[0]) * 0.3)
                y1, y2 = max(0, b[1] - pad), min(h, b[3] + pad)
                x1, x2 = max(0, b[0] - pad), min(w, b[2] + pad)
                crop = frame[y1:y2, x1:x2].copy()
                all_faces.append((emb, crop))
        cap.release()

    if not all_faces:
        _face_candidates = []
        return jsonify(faces=[], error="No faces detected")

    # Cluster faces by embedding similarity
    embeddings = np.array([f[0] for f in all_faces])
    n = len(embeddings)
    used = [False] * n
    clusters = []

    for i in range(n):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > 0.4:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)

    # Build candidates: one per cluster, pick the largest face crop
    _face_candidates = []
    result = []
    for cluster in clusters:
        if len(cluster) < 1:
            continue
        # Pick the face with largest crop area
        best_idx = max(cluster, key=lambda i: all_faces[i][1].shape[0] * all_faces[i][1].shape[1])
        emb = embeddings[cluster].mean(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        crop = all_faces[best_idx][1]
        _face_candidates.append((emb, crop))

        # Compute avg internal similarity
        if len(cluster) > 1:
            sims = []
            for a in cluster:
                for b in cluster:
                    if a < b:
                        sims.append(float(np.dot(embeddings[a], embeddings[b])))
            avg_sim = np.mean(sims) if sims else 1.0
        else:
            avg_sim = 1.0

        result.append({
            "index": len(_face_candidates) - 1,
            "count": len(cluster),
            "avg_sim": round(avg_sim, 3),
        })

    # Sort by frequency (most seen first)
    result.sort(key=lambda x: x["count"], reverse=True)
    # Re-index after sort
    old_candidates = _face_candidates[:]
    _face_candidates = []
    for i, r in enumerate(result):
        _face_candidates.append(old_candidates[r["index"]])
        r["index"] = i

    return jsonify(faces=result)


@app.route("/api/face/candidate/<int:idx>")
def face_candidate_thumbnail(idx):
    """Return face candidate crop as JPEG."""
    global _face_candidates
    if idx < 0 or idx >= len(_face_candidates):
        return "", 404
    crop = _face_candidates[idx][1]
    # Resize to 192x192 square
    size = 192
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = crop
    canvas = cv2.resize(canvas, (size, size))
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/face/select_candidate", methods=["POST"])
def select_face_candidate():
    """Set a detected face candidate as the reference."""
    global _face_candidates
    d = request.json or {}
    idx = d.get("index", -1)
    if idx < 0 or idx >= len(_face_candidates):
        return jsonify(ok=False, error="Invalid index"), 400
    emb = _face_candidates[idx][0]
    engine.reference_embedding = emb
    # Save to face_ref dir
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_ref")
    os.makedirs(ref_dir, exist_ok=True)
    np.save(os.path.join(ref_dir, "selected_embedding.npy"), emb)
    crop = _face_candidates[idx][1]
    cv2.imwrite(os.path.join(ref_dir, "selected_ref.jpg"), crop)
    return jsonify(ok=True)


# ── Auto Generate ──────────────────────────────────────────────

@app.route("/api/auto_generate", methods=["POST"])
def auto_generate():
    """One-click: search person face + search/download dance videos + analyze + render."""
    d = request.json or {}
    person = d.get("person", "").strip()
    song = d.get("song", "").strip()
    if not person or not song:
        return jsonify(error="Need person and song"), 400

    min_dur = d.get("min_dur", 3.0)
    max_dur = d.get("max_dur", 8.0)
    sim_thresh = d.get("sim_thresh", 0.3)
    smart_crop = d.get("smart_crop", True)
    crossfade = d.get("crossfade", 0.2)
    resolution = d.get("resolution", "1080x1920")
    res_parts = resolution.split("x")
    out_w, out_h = int(res_parts[0]), int(res_parts[1])

    def _work():
        try:
            # Clear previous state
            engine.videos.clear()
            engine.segments.clear()

            # Step 1: Search reference face
            _phase_begin("Face reference")
            _progress(0.0, f"Step 1/6: Looking up face reference for {person}...")
            # Check face library first
            face_found = False
            q_lower = person.lower().strip()
            # Prefer exact matches over substring to avoid "yuna" matching "yunah".
            def _score_match(info):
                m = info["member"].lower()
                full = (info["group"] + " " + info["member"]).lower()
                if m == q_lower:            return 100  # exact member
                if full == q_lower:         return 99   # exact "group member"
                if q_lower in m.split():    return 80   # whole-word match inside member
                if (" " + q_lower + " ") in (" " + full + " "):
                                            return 70   # whole-word in group+member
                if m.startswith(q_lower):   return 60   # prefix
                if q_lower in m:            return 40   # member substring (weakest)
                if q_lower in full:         return 20   # group+member substring
                return 0
            ranked = sorted(_face_library_index.items(),
                            key=lambda kv: -_score_match(kv[1]))
            for key, info in ranked:
                if _score_match(info) <= 0:
                    break
                emb_path = os.path.join(_face_library_dir, info["path"], "embedding.npy")
                if os.path.exists(emb_path):
                    engine.reference_embedding = np.load(emb_path)
                    # AUTOGEN_PLAN_META_V1 — record target for plan.json / filenames
                    engine.target_group  = info["group"]
                    engine.target_member = info["member"]
                    # AUTOGEN_PLAN_META_V1 — load IDENTITY_POOL_V1 pool + peer negatives
                    try:
                        import engine as _engmod, person_tracker as _pt_unused  # noqa
                        pool_path = os.path.join(_face_library_dir, info["path"], "embeddings.npy")
                        if os.path.exists(pool_path):
                            engine.reference_embeddings = np.load(pool_path)
                        else:
                            engine.reference_embeddings = engine.reference_embedding[None, :]
                        # peer negatives: other members of same group
                        _g = info["group"]
                        _negs = []
                        for _k, _i in _face_library_index.items():
                            if _i["group"] == _g and _k != key:
                                _ep = os.path.join(_face_library_dir, _i["path"], "embedding.npy")
                                if os.path.exists(_ep):
                                    _v = np.load(_ep)
                                    if _v.ndim == 1:
                                        _v = _v[None, :]
                                    _negs.append(_v)
                        engine.negative_embeddings = (np.concatenate(_negs, axis=0)
                                                      if _negs else None)
                        _n_refs = len(engine.reference_embeddings) if engine.reference_embeddings is not None else 0
                        _n_neg  = len(engine.negative_embeddings) if engine.negative_embeddings is not None else 0
                        _progress(0.1, f"Face ref from library: {info['group']}/{info['member']} "
                                       f"(pool={_n_refs}, negs={_n_neg})")
                    except Exception as _pe:
                        print(f"[warn] identity pool load failed: {_pe}", flush=True)
                        _progress(0.1, f"Face ref from library: {info['group']}/{info['member']}")
                    face_found = True
                    break
            if not face_found:
                try:
                    engine.search_reference_face(
                        person,
                        progress_cb=lambda p, msg: _progress(p * 0.1, msg))
                    _progress(0.1, "Face reference set (online)")
                except Exception as e:
                    _broadcast("error_event", {"text": f"Face search warning: {e} — continuing without face filter"})

            _phase_end()
            _phase_begin("Search videos")
            # Step 2: Search dance videos
            _progress(0.1, f"Step 2/6: Searching dance videos for {person} {song}...")
            # FANCAM_FIRST_V3 — single-person sources rank first so the
            # face-gate keeps more candidates.
            queries = [
                f"{person} {song} 직캠",               # Korean: fancam
                f"{person} {song} fancam 4k",
                f"{person} {song} focus cam",
                f"{person} {song} 세로 직캠",          # vertical fancam
                f"{person} {song} stage",
                f"{person} {song} performance MV",
                f"{person} {song} dance practice",
                f"{person} {song} 안무",               # Korean: choreography
                f"{person} {song} choreography",
            ]
            all_results = []
            seen_ids = set()
            for q in queries:
                try:
                    results = engine.search_videos(q, platform="youtube", max_results=8)
                    for r in results:
                        vid_id = r.get("id", r.get("url", ""))
                        if vid_id not in seen_ids:
                            seen_ids.add(vid_id)
                            all_results.append(r)
                except Exception:
                    pass

            if not all_results:
                _broadcast("error_event", {"text": "No videos found"})
                return

            # Filter by duration: real dance practice / MV / stage are 60s-600s.
            # >600s usually means BEHIND/compilation/vlog that won't mashup well.
            MIN_DUR, MAX_DUR = 60, 600
            filtered = []
            dropped = []
            for r in all_results:
                dur = r.get("duration") or 0
                title = r.get("title", "")[:50]
                if dur and (dur < MIN_DUR or dur > MAX_DUR):
                    dropped.append((title, dur))
                    continue
                filtered.append(r)
            if dropped:
                _progress(0.13, f"Dropped {len(dropped)} videos outside {MIN_DUR}-{MAX_DUR}s")
                for title, dur in dropped[:6]:
                    _broadcast("error_event",
                               {"text": f"Skip {title} ({dur//60}m{dur%60}s)"})

            if not filtered:
                _broadcast("error_event",
                           {"text": f"No videos in {MIN_DUR}-{MAX_DUR}s range"})
                return

            # Larger candidate pool — per-video face-gate will drop
            # group-dance videos where target is rarely visible.
            to_download = filtered[:15]
            _progress(0.15, f"Found {len(all_results)} videos ({len(filtered)} in range), screening up to {len(to_download)}...")

            _phase_end()
            _phase_begin("Download + face-gate")
            # Step 3: Download + per-video face-gate
            # Each candidate is downloaded one at a time; a quick 20-frame
            # face scan rejects videos where target appears in < MIN_RATIO
            # of samples (matches plan_transitions video-level filter).
            dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
            os.makedirs(dl_dir, exist_ok=True)
            TARGET_QUALIFIED = 5
            MIN_RATIO = 0.15
            GATE_SIM = 0.45
            ok_count = 0
            rejected = 0
            screened = 0
            for idx, item in enumerate(to_download):
                if ok_count >= TARGET_QUALIFIED:
                    break
                url = item.get("url", "")
                title = item.get("title", "video")
                _progress(0.15 + idx / len(to_download) * 0.25,
                          f"Step 3/6: Screen {idx+1}/{len(to_download)} "
                          f"(kept {ok_count}/{TARGET_QUALIFIED}): {title[:40]}")
                try:
                    path = engine.download_video(
                        url, dl_dir,
                        progress_cb=lambda p, i=idx: _progress(
                            0.15 + (i + p) / len(to_download) * 0.25))
                except Exception as e:
                    _broadcast("error_event", {"text": f"Download failed: {title[:30]} — {e}"})
                    continue
                screened += 1

                # Face-gate: only if we have a reference embedding
                if engine.reference_embedding is not None:
                    try:
                        ratio, best = engine.measure_face_visibility_ratio(
                            path, sample_count=20, threshold=GATE_SIM)
                    except Exception as e:
                        _broadcast("error_event",
                                   {"text": f"Face-gate error on {title[:30]}: {e} — keeping"})
                        ratio, best = 1.0, 1.0
                    if ratio < MIN_RATIO:
                        rejected += 1
                        _broadcast("error_event",
                                   {"text": f"Rejected {title[:40]}: visibility {int(ratio*100)}% < {int(MIN_RATIO*100)}% (best_sim={best:.2f})"})
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        continue
                    _progress(0.15 + idx / len(to_download) * 0.25,
                              f"Kept {title[:40]} visibility={int(ratio*100)}% best_sim={best:.2f}")

                with _lock:
                    engine.add_video(path)
                    if engine.videos:
                        engine.videos[-1].source_url = url
                ok_count += 1

            _broadcast("error_event",
                       {"text": f"Download gate: {ok_count} kept, {rejected} rejected, "
                                f"{screened - ok_count - rejected} other, "
                                f"target was {TARGET_QUALIFIED}"})

            if ok_count < 2:
                _broadcast("error_event", {"text": f"Only {ok_count} videos qualified, need at least 2 — try a different song"})
                return

            _phase_end()
            _phase_begin("Verify face/group")
            # Verify face in downloaded videos — remove cover dances / wrong person
            if engine.reference_embedding is not None and len(engine.videos) >= 2:
                _progress(0.42, "Verifying face/group members in downloaded videos...")

                # Build group roster: load embeddings for all members of the
                # selected person's group (for multi-member verification).
                # Re-use the same scoring so group lookup matches face lookup.
                target_group = None
                target_member = None
                _ranked_g = sorted(_face_library_index.items(),
                                   key=lambda kv: -_score_match(kv[1]))
                for key, info in _ranked_g:
                    if _score_match(info) <= 0:
                        break
                    target_group = info['group']
                    target_member = info['member']
                    break

                group_embs = {}
                if target_group:
                    for key, info in _face_library_index.items():
                        if info['group'] == target_group:
                            ep = os.path.join(_face_library_dir, info['path'], 'embedding.npy')
                            if os.path.exists(ep):
                                try:
                                    group_embs[info['member']] = np.load(ep)
                                except Exception:
                                    pass
                group_size = len(group_embs)
                is_solo = group_size <= 1 or (target_group and target_group.lower() == 'solo')
                # For groups: require at least min(3, group_size-1) members
                # visible across the video. Solo: just verify the one person.
                if is_solo:
                    min_members = 1
                else:
                    min_members = min(3, max(2, group_size - 1))
                _progress(0.42, f"Group={target_group} size={group_size} requires {min_members}+ members in video")

                # Tiered thresholds:
                #   < TARGET_MIN     -> reject (wrong person / cover by lookalike)
                #   >= TARGET_STRONG -> accept unconditionally (solo perf OK)
                #   otherwise        -> need multi-member evidence for groups
                TARGET_MIN    = 0.42
                TARGET_STRONG = 0.55
                MEMBER_THRESH = 0.40

                to_remove = []
                for vi in range(len(engine.videos)):
                    vpath = engine.videos[vi].path
                    vname = engine.videos[vi].filename[:30]
                    # Single-person check (raises to 0.42)
                    is_match, best_sim = engine.verify_face_in_video(
                        vpath, sample_count=12, threshold=TARGET_MIN)
                    if not is_match:
                        to_remove.append(vi)
                        _broadcast("error_event",
                                   {"text": f"Removed {vname} (target not found, sim={best_sim:.2f})"})
                        continue

                    # Strong match? Accept even if other members absent (solo songs).
                    if best_sim >= TARGET_STRONG or is_solo or len(group_embs) < 2:
                        _progress(0.42, f"OK {vname} sim={best_sim:.2f} (strong/solo)")
                        continue

                    # Borderline (0.42 <= sim < 0.55): need group context
                    member_sims = engine.detect_members_in_video(
                        vpath, group_embs, sample_count=14, threshold=MEMBER_THRESH)
                    present = [m for m, s in member_sims.items() if s >= MEMBER_THRESH]
                    top = sorted(member_sims.items(), key=lambda x: -x[1])[:4]
                    top_str = ", ".join(f"{m[:8]}={s:.2f}" for m, s in top)
                    if len(present) < min_members:
                        to_remove.append(vi)
                        _broadcast("error_event",
                                   {"text": f"Removed {vname}: sim={best_sim:.2f} borderline AND only {len(present)}/{min_members} members [{top_str}]"})
                    else:
                        _progress(0.42, f"OK {vname} sim={best_sim:.2f} + {len(present)} members [{top_str}]")

                for vi in sorted(to_remove, reverse=True):
                    engine.videos.pop(vi)
                if len(engine.videos) < 2:
                    _broadcast("error_event", {"text": f"Too few valid videos ({len(engine.videos)}) after verification — try a different song"})
                    return

            _broadcast("download_done", {"ok": len(engine.videos)})

            _phase_end()
            _phase_begin("Audio download")
            # Step 4: Download audio
            _progress(0.45, f"Step 4/6: Downloading audio for {song}...")
            try:
                engine.download_audio(
                    f"{person} {song}",
                    progress_cb=lambda p: _progress(0.45 + p * 0.05))
                _broadcast("audio_done", {"title": f"{person} {song}"})
            except Exception as e:
                _broadcast("error_event", {"text": f"Audio download failed: {e} — will use video audio"})
            # AUTOGEN_PLAN_META_V1 — overwrite with clean song name for filenames/plan
            try:
                if song:
                    engine.external_audio_title = song
            except Exception:
                pass

            _phase_end()
            _phase_begin("Analyze poses")
            # Step 5: Analyze
            n = len(engine.videos)
            for i in range(n):
                nm = engine.videos[i].filename
                _progress(0.5 + i / n * 0.3, f"Step 5/6: Analyzing {i+1}/{n}: {nm[:40]}")
                engine.analyze_poses(i, sample_fps=10.0)

            _phase_end()
            _phase_begin("Beats + sync")
            _progress(0.82, "Detecting beats...")
            use_ext = engine.external_audio is not None
            engine.detect_beats(0, use_external=use_ext)

            # Compute audio offsets if external audio
            if use_ext:
                _progress(0.84, "Computing audio sync...")
                engine.compute_audio_offsets(use_external=True)

            # Compute face visibility if face ref set
            _phase_end()
            _phase_begin("Face visibility")
            if engine.reference_embedding is not None:
                _progress(0.86, "Scanning face visibility...")
                for i in range(n):
                    engine.compute_face_visibility(i)

            _phase_end()
            _phase_begin("Plan transitions")
            _progress(0.88, "Planning transitions...")
            engine.plan_transitions(min_dur, max_dur, sim_thresh)
            ns = len(engine.segments)
            _broadcast("analysis_done", {"segments": ns})

            # Step 6: Render
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"mashup_{int(time.time())}.mp4"
            out_path = os.path.join(out_dir, out_name)

            _phase_end()
            _phase_begin("Render")
            _progress(0.9, "Step 6/6: Rendering...")
            try:
                engine.render(
                    out_path, 0,
                    use_external_audio=use_ext,
                    output_width=out_w, output_height=out_h,
                    smart_crop=smart_crop,
                    crossfade_sec=crossfade,
                    progress_cb=lambda p, msg="": _progress(0.9 + p * 0.1,
                                                    f"Rendering {p*100:.0f}%"))
            finally:
                # Always reap any stray remote RIFE processes
                try:
                    engine.cleanup_remote_rife()
                except Exception:
                    pass
            sz = os.path.getsize(out_path) / 1024 / 1024
            _progress(1.0, f"Done! {sz:.1f} MB")
            # Save plan JSON alongside the mashup so the UI can offer swaps.
            try:
                engine.export_plan_json(out_path)
            except Exception as pe:
                print(f"[plan] export failed: {pe}", flush=True)
            _phase_end()
            _task_state["active"] = False
            _broadcast("render_done", {"filename": out_name, "size_mb": round(sz, 1)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            _phase_end()
            _broadcast("error_event", {"text": f"Auto generate failed: {e}"})
            _task_state["active"] = False
            try:
                engine.cleanup_remote_rife()
            except Exception:
                pass

    _task_state.update(active=True, progress=0.0, text="Starting...",
                       events=[], result=None, cancel=False,
                       phases=[],                   # PROGRESS_HOOK_V1
                       started_at=time.time())  # TASK_CANCEL_SWEEP_V1
    threading.Thread(target=_work, daemon=True).start()
    return jsonify(started=True)


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
