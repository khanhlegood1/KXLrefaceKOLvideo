"""
KXL FaceSwap — core pipeline.

Re-usable, memory-bounded face-swapping logic. The proven design (validated on a
2 vCPU / 2 GB box) is:

  * One onnxruntime session for the face detector (SCRFD det_10g, 640x640).
  * One onnxruntime session for the swapper (inswapper_128, fp32).
  * A fixed, pre-computed "latent" vector for the source identity (the person's
    face averaged over several photos), so per-face work is a single UNet forward.
  * The exact insightface paste-back (align -> warp back -> feathered mask blend).

Memory notes (important on small boxes):
  * onnxruntime is configured with intra_op_num_threads=2, inter_op_num_threads=1.
  * The large ONNX file page-cache is dropped with posix_fadvise(DONTNEED) right
    after the session loads, so the weight bytes live in anon memory only and the
    OS page cache for the model file does not stay charged to the cgroup.
  * Heavy per-frame work is isolated into per-part subprocesses (see part_runner),
    so the OS memory is fully returned between parts and a slow leak can never
    accumulate across a long video.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

# NOTE: numpy / cv2 / onnxruntime are imported lazily inside the functions that
# need them. The FastAPI server imports this module for config + ensure_models
# only; keeping the heavy libs out of the server process saves ~200MB of RAM,
# which is decisive on a 2GB box (part runners peak at ~1.5GB).

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

BUFFALO_L_ZIP = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
INSWAPPER_FP32 = "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx"
NEEDED = ("det_10g.onnx", "w600k_r50.onnx", "inswapper_128.onnx", "emap_fp32.npy")


def ensure_models(models_dir: str = None, log=print) -> str:
    """Download (once) every model file the pipeline needs. Idempotent."""
    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    def have(name):
        p = os.path.join(models_dir, name)
        return os.path.exists(p) and os.path.getsize(p) > 1000

    # detector + recognition come from buffalo_l.zip
    if not (have("det_10g.onnx") and have("w600k_r50.onnx")):
        zip_path = os.path.join(models_dir, "buffalo_l.zip")
        log("downloading buffalo_l.zip ...")
        urllib.request.urlretrieve(BUFFALO_L_ZIP, zip_path)
        log("extracting detector + recognition ...")
        with zipfile.ZipFile(zip_path) as z:
            for member in ("det_10g.onnx", "w600k_r50.onnx"):
                try:
                    z.extract(member, models_dir)
                except KeyError:
                    pass
        os.remove(zip_path)

    # swapper (fp32 — measured fastest on CPU, no fp16->fp32 double materialization)
    if not have("inswapper_128.onnx"):
        log("downloading inswapper_128.onnx (fp32, ~554 MB) ...")
        urllib.request.urlretrieve(INSWAPPER_FP32, os.path.join(models_dir, "inswapper_128.onnx"))

    # emap matrix (needed to project the source embedding into the latent space)
    if not have("emap_fp32.npy"):
        log("extracting emap ...")
        import onnx
        from onnx import numpy_helper
        g = onnx.load(os.path.join(models_dir, "inswapper_128.onnx")).graph
        np.save(os.path.join(models_dir, "emap_fp32.npy"), numpy_helper.to_array(g.initializer[-1]))

    missing = [n for n in NEEDED if not have(n)]
    if missing:
        raise RuntimeError("missing model files: %s" % missing)
    return models_dir

# --------------------------------------------------------------------------- #
# Session / model loading
# --------------------------------------------------------------------------- #

def make_session(onnx_path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])


def drop_file_cache(path: str) -> None:
    """Ask the OS to drop the page cache for `path` (best effort, no root needed).

    This keeps large model/video files from staying charged against the cgroup's
    memory limit while the process runs.
    """
    try:
        f = open(path, "rb")
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            f.close()
    except Exception:
        pass


def load_detector(models_dir: str):
    """Load the SCRFD face detector."""
    from insightface.model_zoo import get_model

    det = get_model(os.path.join(models_dir, "det_10g.onnx"), providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, input_size=(640, 640))
    drop_file_cache(os.path.join(models_dir, "det_10g.onnx"))
    return det


def load_swapper(models_dir: str):
    """Load the inswapper (fp32) session. Returns (session, in0, in1)."""
    swp = make_session(os.path.join(models_dir, "inswapper_128.onnx"))
    drop_file_cache(os.path.join(models_dir, "inswapper_128.onnx"))
    return swp, swp.get_inputs()[0].name, swp.get_inputs()[1].name


def compute_latent(models_dir: str, embedding_path: str) -> np.ndarray:
    """Project the (averaged) source embedding through the model's emap.

    This is constant for a job, so it is computed once.
    """
    import numpy as np

    emap = np.load(os.path.join(models_dir, "emap_fp32.npy")).astype(np.float32)
    src = np.load(embedding_path).astype(np.float32)
    src = src / np.linalg.norm(src)
    latent = (src.reshape(1, -1) @ emap).astype(np.float32)
    latent /= np.linalg.norm(latent)
    return latent


# --------------------------------------------------------------------------- #
# Per-face swap (the exact insightface paste-back)
# --------------------------------------------------------------------------- #

def swap_face(img, kps, swp, swp_in0, swp_in1, latent):
    """Swap the single face (landmarks `kps`) in `img` with the source identity."""
    import numpy as np
    import cv2
    from insightface.utils import face_align

    aimg, M = face_align.norm_crop2(img, kps, 128)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (128, 128), (0.0, 0.0, 0.0), swapRB=True).astype(np.float32)
    pred = swp.run(None, {swp_in0: blob, swp_in1: latent})[0]
    img_fake = pred.transpose((0, 2, 3, 1))[0]
    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

    target_img = img
    fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white[img_white > 20] = 255
    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) == 0:
        return img
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
    k = max(mask_size // 20, 5)
    blur_size = (2 * k + 1, 2 * k + 1)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    blur_size = (11, 11)
    fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    img_mask /= 255
    fake_diff /= 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    return fake_merged.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Source identity from a set of photos
# --------------------------------------------------------------------------- #

def build_source_embedding(photos, models_dir: str, out_path: str) -> int:
    """Average the face embedding over `photos` (paths) and save to out_path.

    Returns the number of photos that contributed a detected face.
    """
    import numpy as np
    import cv2
    from insightface.model_zoo import get_model
    from insightface.app.common import Face

    det = get_model(os.path.join(models_dir, "det_10g.onnx"), providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, input_size=(640, 640))
    drop_file_cache(os.path.join(models_dir, "det_10g.onnx"))
    rec = get_model(os.path.join(models_dir, "w600k_r50.onnx"), providers=["CPUExecutionProvider"])

    def detect_faces(img):
        d, k = det.detect(img)
        return [Face(bbox=d[i, :4], kps=k[i], det_score=float(d[i, 4])) for i in range(len(d))]

    embs = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
        faces = detect_faces(img)
        if not faces:
            continue
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embs.append(rec.get(img, best))

    if not embs:
        raise ValueError("No face detected in any of the provided photos.")
    avg = np.mean(embs, axis=0)
    avg = avg / np.linalg.norm(avg)
    np.save(out_path, avg.astype(np.float32))
    return len(embs)
