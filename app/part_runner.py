"""
KXL FaceSwap — per-part runner (one video segment = one OS process).

Each part is a fresh Python process so the OS returns ALL memory (weights,
arenas, page cache) between parts. This is what makes very long videos safe on
small machines: a slow leak in one part can never grow into the next one.

Usage (launched by app/worker.py):
    python -m app.part_runner --start 1200 --frames 600 \
        --video work/input.webm --out work/parts/khanh_part_01200.mp4 \
        --models models --embed work/src_embedding.npy [--slow-seek]

Writes a JSON line to stdout on completion:
    {"start": 1200, "frames": 600, "faces": 431, "det_s": 20.1, "swap_s": 211.4}
Exit code 0 = ok (even a short tail part), non-zero = hard failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from app.pipeline import drop_file_cache, load_detector, load_swapper, compute_latent, swap_face

W = H = 1080  # source resolution is fixed for the validated pipeline


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--slow-seek", action="store_true",
                    help="accurate (but slow) seek: decode from file start")
    return ap.parse_args()


def main():
    a = parse_args()
    t0 = time.time()
    fps = a.fps
    frame_bytes = W * H * 3

    det = load_detector(a.models)
    swp, swp_in0, swp_in1 = load_swapper(a.models)
    latent = compute_latent(a.models, a.embed)

    ss = "%.3f" % (a.start / fps)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (W, H),
         "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
         "-r", str(fps), "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)

    if a.slow_seek:
        dec = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-threads", "2", "-i", a.video,
             "-ss", ss, "-frames:v", str(a.frames),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE)
    else:
        dec = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-threads", "2", "-ss", ss, "-i", a.video,
             "-frames:v", str(a.frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE)

    buf = b""
    n = 0
    n_faces = 0
    t_det = 0.0
    t_swap = 0.0
    while True:
        c = dec.stdout.read(frame_bytes)
        if not c:
            break
        buf += c
        while len(buf) >= frame_bytes:
            img = np.frombuffer(buf[:frame_bytes], np.uint8).copy().reshape(H, W, 3)
            buf = buf[frame_bytes:]
            t1 = time.time()
            d, k = det.detect(img)
            t_det += time.time() - t1
            for i in range(min(len(d), 4)):
                t1 = time.time()
                img = swap_face(img, k[i], swp, swp_in0, swp_in1, latent)
                t_swap += time.time() - t1
                n_faces += 1
            enc.stdin.write(img.tobytes())
            n += 1
            if n % 200 == 0:
                drop_file_cache(a.out)
                drop_file_cache(a.video)
                drop_file_cache(os.path.join(a.models, "inswapper_128.onnx"))

    enc.stdin.close()
    dec.stdout.close()
    enc.wait()
    dec.wait()

    print(json.dumps({
        "start": a.start, "frames": n, "faces": n_faces,
        "det_s": round(t_det, 1), "swap_s": round(t_swap, 1),
        "total_s": round(time.time() - t0, 1),
        "encoder_rc": enc.returncode,
    }), flush=True)
    sys.exit(0 if enc.returncode == 0 and n > 0 else 1)


if __name__ == "__main__":
    main()
