"""
KXL FaceSwap — job orchestrator.

Flow:
    download YouTube video -> probe -> normalize (if needed) ->
    build source embedding -> run parts (subprocess each) ->
    concat video -> mux audio -> final mp4

Progress is written to <job_dir>/state.json after every state change and can be
polled by the API (app/main.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

from app import pipeline


PART_SIZE = int(os.environ.get("PART_SIZE", "600"))
MAX_RETRIES = int(os.environ.get("MAX_PART_RETRIES", "1"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class JobError(Exception):
    pass


@dataclass
class Job:
    job_id: str
    job_dir: str
    url: str
    photos: list = field(default_factory=list)
    source_file: str | None = None  # uploaded video — skips YouTube download

    @property
    def state_path(self) -> str:
        return os.path.join(self.job_dir, "state.json")

    def set_state(self, **kw) -> None:
        try:
            with open(self.state_path) as f:
                st = json.load(f)
        except Exception:
            st = {"job_id": self.job_id, "created_at": time.time()}
        st.update(kw)
        st["updated_at"] = time.time()
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, self.state_path)

    @property
    def cancelled(self) -> bool:
        return os.path.exists(os.path.join(self.job_dir, "cancel"))

    def result_path(self) -> str:
        return os.path.join(self.job_dir, "result.mp4")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def probe_video(path: str) -> dict:
    r = subprocess.run(["ffmpeg", "-i", path], capture_output=True, text=True)
    out = {"width": 1080, "height": 1080, "fps": 60, "duration": 0.0}
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    if m:
        out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"(\d{2,5})x(\d{2,5})", r.stderr)
    if m:
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", r.stderr)
    if m:
        out["fps"] = max(1, int(round(float(m.group(1)))))
    return out


def download_youtube(url: str, dest: str) -> None:
    import yt_dlp

    opts = {
        "format": "best[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": dest,
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
    }
    cookie_file = os.environ.get("YTDLP_COOKIE_FILE")
    if cookie_file:
        opts["cookiefile"] = cookie_file
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise JobError("Không tải được video từ YouTube: %s" % str(e)[:300])
    # find the actual file (outtmpl + extension)
    ext = os.path.splitext(info.get("requested_downloads", [{}])[0].get("filepath", dest) or dest)[1]
    final = os.path.splitext(dest)[0] + ext
    if final != dest and os.path.exists(final):
        os.replace(final, dest)


def normalize_video(src: str, dst: str, fps: int) -> None:
    """Scale/pad to 1080x1080 so the validated 1080 pipeline applies."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2:black,fps=%d" % fps,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
         "-an", dst],
        check=True)


def run_part(job: Job, work: str, video: str, start: int, frames: int, slow: bool,
             models: str = None) -> dict:
    models = models or pipeline.MODELS_DIR
    os.makedirs(os.path.join(work, "parts"), exist_ok=True)
    out = os.path.join(work, "parts", "part_%05d.mp4" % start)
    cmd = [
        sys.executable, "-m", "app.part_runner",
        "--start", str(start), "--frames", str(frames),
        "--video", video, "--out", out,
        "--models", models, "--embed", os.path.join(work, "src_embedding.npy"),
        "--fps", str(FPS_OF[job.job_id]),
    ]
    if slow:
        cmd.append("--slow-seek")
    for attempt in range(MAX_RETRIES + 1):
        if job.cancelled:
            raise JobError("cancelled")
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        last_line = ""
        for line in reversed((p.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                last_line = line
                break
        if p.returncode == 0 and last_line:
            return json.loads(last_line)
        # retry with slow seek on failure
        if attempt < MAX_RETRIES:
            job.set_state(note="part %d: retry (slow-seek) — %s" % (start, (p.stderr or "")[-200:]))
            slow = True
            time.sleep(5)
    raise JobError("part %d thất bại: %s" % (start, (p.stderr or p.stdout or "")[-300:]))


FPS_OF: dict = {}  # job_id -> fps (set by run_job)


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def run_job(job: Job) -> None:
    work = job.job_dir
    FPS_OF[job.job_id] = 60
    try:
        # 1) source video: uploaded file, or download from YouTube
        src = os.path.join(work, "input_src.mp4")
        if job.source_file and os.path.exists(job.source_file):
            job.set_state(status="preparing", pct=0)
            os.replace(job.source_file, src)
        else:
            job.set_state(status="downloading", pct=0)
            download_youtube(job.url, src)
        if job.cancelled:
            raise JobError("cancelled")

        # 2) probe + normalize
        job.set_state(status="probing")
        info = probe_video(src)
        fps = info["fps"] or 60
        fps = max(1, min(fps, 60))
        FPS_OF[job.job_id] = fps
        total_frames = int(round(info["duration"] * fps))
        if (info["width"], info["height"]) == (1080, 1080):
            video = src
        else:
            video = os.path.join(work, "input_norm.mp4")
            normalize_video(src, video, fps)
        if total_frames <= 0:
            total_frames = PART_SIZE  # probe failed — assume at least one part
        parts = max(1, (total_frames + PART_SIZE - 1) // PART_SIZE)
        job.set_state(status="embedding", pct=1, parts=parts,
                      total_frames=total_frames, width=info["width"], height=info["height"], fps=fps)

        # 3) source embedding — subprocess so the API server never holds model weights
        p = subprocess.run(
            [sys.executable, "-m", "app.embed_runner",
             "--photos", *job.photos, "--models", pipeline.MODELS_DIR,
             "--out", os.path.join(work, "src_embedding.npy")],
            capture_output=True, text=True, cwd=ROOT)
        if p.returncode != 0:
            raise JobError("Không nhận diện được khuôn mặt trong ảnh (chụp rõ mặt, đủ sáng): %s"
                           % (p.stderr or p.stdout or "")[-200:])
        if job.cancelled:
            raise JobError("cancelled")

        # 4) parts
        done_frames = 0
        faces_total = 0
        t_start = time.time()
        job.set_state(status="swapping", pct=1, part=0, parts=parts,
                      frames_done=0, total_frames=total_frames,
                      note="bắt đầu thay mặt…")
        for i in range(parts):
            start = i * PART_SIZE
            want = min(PART_SIZE, total_frames - start)
            res = run_part(job, work, video, start, want, slow=(i == parts - 1))
            done_frames += res["frames"]
            faces_total += res["faces"]
            if res["frames"] < want:
                break  # tail part: video ended
            el = time.time() - t_start
            rate = done_frames / max(el, 1e-6)
            eta = (total_frames - done_frames) / max(rate, 1e-6)
            job.set_state(
                status="swapping",
                pct=round(100.0 * done_frames / total_frames, 1),
                part=i + 1, parts=parts,
                frames_done=done_frames, total_frames=total_frames,
                faces=faces_total, eta_s=int(eta),
                note="thay mặt frame %d/%d (%d khuôn mặt đã thay)" % (done_frames, total_frames, faces_total),
            )

        # 5) concat + audio mux
        job.set_state(status="muxing", pct=100, note="ghép các phần + trộn âm thanh…")
        with open(os.path.join(work, "concat.txt"), "w") as f:
            i = 0
            while True:
                p = os.path.join(work, "parts", "part_%05d.mp4" % (i * PART_SIZE))
                if not os.path.exists(p):
                    break
                f.write("file '%s'\n" % os.path.abspath(p))
                i += 1
        video_only = os.path.join(work, "video_only.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", os.path.join(work, "concat.txt"), "-c", "copy", video_only],
            check=True)
        final = job.result_path()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only, "-i", src,
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", final],
            check=True)

        job.set_state(status="done", pct=100, result="result.mp4",
                      faces=faces_total,
                      note="Hoàn tất! Đã thay %d khuôn mặt." % faces_total)
    except JobError as e:
        job.set_state(status="error", error=str(e))
    except Exception as e:  # noqa: BLE001
        job.set_state(status="error", error="Lỗi bất ngờ: %s" % str(e)[:300])
    finally:
        FPS_OF.pop(job.job_id, None)


# Re-export for the CLI
def ensure_models(models_dir: str) -> None:
    pipeline.ensure_models(models_dir)
