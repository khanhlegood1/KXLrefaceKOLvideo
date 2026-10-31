"""
KXL FaceSwap — FastAPI application.

Endpoints:
    GET  /                         -> frontend (app/static/index.html)
    GET  /api/health               -> liveness + models status
    POST /api/jobs                 -> create job (multipart: photos + url)
    GET  /api/jobs                 -> list jobs
    GET  /api/jobs/{job_id}        -> job state
    POST /api/jobs/{job_id}/cancel -> cancel
    DELETE /api/jobs/{job_id}      -> delete job + files
    GET  /api/jobs/{job_id}/video  -> stream result.mp4
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from threading import Lock, Thread

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import pipeline, worker

JOB_ROOT = os.environ.get("JOB_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jobs"))
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "10"))
MAX_PHOTO_MB = int(os.environ.get("MAX_PHOTO_MB", "25"))
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "1024"))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))
MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", "1"))

app = FastAPI(title="KXL FaceSwap", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = Lock()
_jobs: dict[str, dict] = {}  # job_id -> {"job": worker.Job, "thread": Thread, "active": bool}
_active = 0


def _jobs_dir(job_id: str) -> str:
    return os.path.join(JOB_ROOT, job_id)


def _read_state(job_id: str) -> dict | None:
    p = os.path.join(_jobs_dir(job_id), "state.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _cleanup_old_jobs() -> None:
    now = time.time()
    for entry in list(_jobs.values()):
        job = entry["job"]
        st = _read_state(job.job_id)
        age_h = (now - (st or {}).get("created_at", now)) / 3600
        if age_h > JOB_TTL_HOURS and not entry["active"]:
            shutil.rmtree(job.job_dir, ignore_errors=True)
            del _jobs[job.job_id]
            global _active
            entry.pop("thread", None)


def _start_job(job: worker.Job) -> None:
    global _active
    entry = {"job": job, "active": True, "thread": None}

    def run():
        global _active
        try:
            worker.run_job(job)
        finally:
            with _lock:
                _active -= 1
                entry["active"] = False
            job.set_state(finished_at=time.time())

    with _lock:
        _jobs[job.job_id] = entry
        _active += 1
        t = Thread(target=run, daemon=True)
        entry["thread"] = t
        t.start()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    models = pipeline.MODELS_DIR
    return {
        "ok": True,
        "models_dir": models,
        "models_ready": all(
            os.path.exists(os.path.join(models, n)) for n in pipeline.NEEDED
        ),
        "active_jobs": _active,
    }


@app.post("/api/jobs")
async def create_job(
    photos: list[UploadFile] = File(...),
    url: str = Form(""),
    video: UploadFile | None = File(None),
):
    """Create a job. Provide either a YouTube `url` or an uploaded `video` file
    (video file wins — useful when YouTube blocks the server IP)."""
    url = (url or "").strip()
    has_video = video is not None
    if has_video:
        ext = os.path.splitext(video.filename or "video.mp4")[1].lower()
        if ext not in (".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi"):
            raise HTTPException(400, "File video: chỉ chấp nhận mp4/webm/mov/mkv.")
        size = video.size if video.size is not None else 0
        if size > MAX_VIDEO_MB * 1024 * 1024:
            raise HTTPException(400, "File video quá %d MB." % MAX_VIDEO_MB)
    elif not url:
        raise HTTPException(400, "Cần link YouTube hoặc file video.")
    else:
        import re
        if not re.search(r"(youtube\.com|youtu\.be)/\S+", url):
            raise HTTPException(400, "Link không giống link YouTube/YouTube Shorts.")

    if not photos:
        raise HTTPException(400, "Cần ít nhất 1 ảnh của bạn.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(400, "Tối đa %d ảnh." % MAX_PHOTOS)

    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = _jobs_dir(job_id)
    os.makedirs(os.path.join(job_dir, "photos"), exist_ok=True)

    source_file = None
    if has_video:
        source_file = os.path.join(job_dir, "input_upload" + os.path.splitext(video.filename or "video.mp4")[1].lower())
        with open(source_file, "wb") as f:
            while chunk := await video.read(8 * 1024 * 1024):
                f.write(chunk)

    saved = []
    for i, ph in enumerate(photos):
        data = await ph.read()
        if len(data) > MAX_PHOTO_MB * 1024 * 1024:
            raise HTTPException(400, "Ảnh %s quá %d MB." % (ph.filename, MAX_PHOTO_MB))
        ext = os.path.splitext(ph.filename or "photo.jpg")[1].lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            raise HTTPException(400, "Ảnh %s: chỉ chấp nhận jpg/png/webp." % ph.filename)
        p = os.path.join(job_dir, "photos", "%02d%s" % (i + 1, ext))
        with open(p, "wb") as f:
            f.write(data)
        saved.append(p)

    job = worker.Job(job_id=job_id, job_dir=job_dir, url=url, photos=saved,
                     source_file=source_file)
    job.set_state(status="queued", pct=0, url=url or ("file: " + (video.filename or "video")),
                  n_photos=len(saved), source="upload" if source_file else "youtube",
                  created_at=time.time(), message="Đang chờ xử lý…")

    if _active >= MAX_ACTIVE_JOBS:
        raise HTTPException(503, "Máy đang xử lý job khác, vui lòng thử lại sau ít phút.")

    _cleanup_old_jobs()
    _start_job(job)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    out = []
    for job_id in sorted(_jobs.keys(), reverse=True):
        st = _read_state(job_id)
        if st:
            out.append(st)
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    st = _read_state(job_id)
    if not st:
        raise HTTPException(404, "Không tìm thấy job.")
    return st


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    entry = _jobs.get(job_id)
    if not entry:
        raise HTTPException(404, "Không tìm thấy job.")
    st = _read_state(job_id)
    if st and st.get("status") not in ("queued", "downloading", "probing", "embedding", "swapping", "muxing"):
        raise HTTPException(400, "Job đã hoàn thành hoặc đã dừng.")
    open(os.path.join(entry["job"].job_dir, "cancel"), "w").close()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    entry = _jobs.get(job_id)
    if not entry:
        raise HTTPException(404, "Không tìm thấy job.")
    if entry["active"]:
        raise HTTPException(400, "Đang xử lý — vui lòng hủy job trước.")
    shutil.rmtree(entry["job"].job_dir, ignore_errors=True)
    del _jobs[job_id]
    return {"ok": True}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    p = _jobs.get(job_id, {}).get("job", None)
    path = os.path.join(_jobs_dir(job_id), "result.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, "Chưa có video kết quả.")
    return FileResponse(path, media_type="video/mp4", filename="face_swap_result.mp4")


# frontend
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
