#!/usr/bin/env python3
"""
KXL FaceSwap — one-shot CLI (no web UI).

Swap your face into a YouTube video in a single command:

    python3 scripts/one_shot_swap.py \
        --photos uploads/me1.jpg uploads/me2.jpg \
        --url  "https://youtube.com/shorts/XXXXXXXX" \
        --out  result.mp4

Options:
    --models-dir DIR   where the ONNX models live (default: ./models)
    --work DIR         scratch directory (default: ./work_one_shot)
    --no-audio         skip audio mux
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline, worker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="result.mp4")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--work", default="./work_one_shot")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    photos = [p for p in a.photos if os.path.exists(p)]
    if not photos:
        sys.exit("Không tìm thấy file ảnh nào.")

    models = a.models_dir or pipeline.MODELS_DIR
    if not all(os.path.exists(os.path.join(models, n)) for n in pipeline.NEEDED):
        print("=> models chưa đủ, đang tải (một lần duy nhất)…")
        pipeline.ensure_models(models)

    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()

    # 1) download
    print("=> Đang tải video từ YouTube…")
    src = os.path.join(a.work, "input_src.mp4")
    worker.download_youtube(a.url, src)

    # 2) probe + normalize
    info = worker.probe_video(src)
    fps = max(1, min(info["fps"] or 60, 60))
    total_frames = int(round(info["duration"] * fps))
    video = src
    if (info["width"], info["height"]) != (1080, 1080):
        video = os.path.join(a.work, "input_norm.mp4")
        worker.normalize_video(src, video, fps)
    print("   video: %dx%d @ %dfps, ~%ds (%d frames)" % (
        info["width"], info["height"], fps, info["duration"], total_frames))

    # 3) embedding (subprocess — giữ bộ nhớ cho part runner)
    print("=> Đang nhận diện khuôn mặt từ %d ảnh…" % len(photos))
    import subprocess as _sp
    pe = _sp.run(
        [sys.executable, "-m", "app.embed_runner",
         "--photos", *photos, "--models", models,
         "--out", os.path.join(a.work, "src_embedding.npy")],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True)
    if pe.returncode != 0:
        sys.exit("Không nhận diện được mặt trong ảnh: %s" % (pe.stderr or pe.stdout)[-300:])
    print("   %s" % pe.stdout.strip())

    # 4) parts
    parts = (total_frames + worker.PART_SIZE - 1) // worker.PART_SIZE
    done = 0
    faces = 0
    for i in range(parts):
        start = i * worker.PART_SIZE
        want = min(worker.PART_SIZE, total_frames - start)
        print("=> Phần %d/%d (frame %d–%d)…" % (i + 1, parts, start, start + want - 1), flush=True)
        job = worker.Job(job_id="cli", job_dir=a.work, url=a.url)
        worker.FPS_OF["cli"] = fps
        res = worker.run_part(job, a.work, video, start, want, slow=(i == parts - 1))
        done += res["frames"]
        faces += res["faces"]
        print("   xong: %d frames, %d khuôn mặt (%.1fs)" % (res["frames"], res["faces"], res["total_s"]))
        if res["frames"] < want:
            break

    # 5) concat + mux
    print("=> Ghép các phần + trộn âm thanh…")
    with open(os.path.join(a.work, "concat.txt"), "w") as f:
        i = 0
        while True:
            p = os.path.join(a.work, "parts", "part_%05d.mp4" % (i * worker.PART_SIZE))
            if not os.path.exists(p):
                break
            f.write("file '%s'\n" % os.path.abspath(p))
            i += 1
    video_only = os.path.join(a.work, "video_only.mp4")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", os.path.join(a.work, "concat.txt"), "-c", "copy", video_only], check=True)
    if a.no_audio:
        os.replace(video_only, a.out)
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only, "-i", src,
             "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
             "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], check=True)

    print("\n✅ Xong! %s  (%.1fs, đã thay %d khuôn mặt)" % (a.out, time.time() - t0, faces))


if __name__ == "__main__":
    main()
