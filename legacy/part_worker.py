import sys, time, subprocess, os
import numpy as np
import cv2
import onnxruntime as ort
from insightface.model_zoo import get_model
from insightface.utils import face_align

W, H = 1080, 1080
FPS = 60
FRAME = W * H * 3
TOTAL = 9053
PART = int(os.environ.get('PART_SIZE', '600'))
START = int(sys.argv[1])
END = min(START + PART, TOTAL)
OUT = 'khanh_part_%05d.mp4' % START

def drop_file_cache(path):
    try:
        f = open(path, 'rb')
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            f.close()
    except Exception:
        pass

def mk_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

t0 = time.time()
det = get_model('models/det_10g.onnx', providers=['CPUExecutionProvider'])
det.prepare(ctx_id=-1, input_size=(640, 640))
drop_file_cache('models/det_10g.onnx')
swp = mk_session('models/inswapper_128.onnx')
drop_file_cache('models/inswapper_128.onnx')
swp_in0 = swp.get_inputs()[0].name
swp_in1 = swp.get_inputs()[1].name

emap = np.load('emap_fp32.npy').astype(np.float32)
src_emb = np.load('src_embedding.npy').astype(np.float32)
src_emb = src_emb / np.linalg.norm(src_emb)
LATENT = (src_emb.reshape(1, -1) @ emap).astype(np.float32)
LATENT /= np.linalg.norm(LATENT)
print('[%s] part %d-%d ready' % (time.strftime('%H:%M:%S'), START, END), flush=True)

def swap_face(img, kps):
    aimg, M = face_align.norm_crop2(img, kps, 128)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (128, 128), (0.0, 0.0, 0.0), swapRB=True).astype(np.float32)
    pred = swp.run(None, {swp_in0: blob, swp_in1: LATENT})[0]
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

enc = subprocess.Popen(
    ['ffmpeg', '-y', '-loglevel', 'error',
     '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', 'pipe:0',
     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p',
     '-r', str(FPS), '-movflags', '+faststart', OUT],
    stdin=subprocess.PIPE)

if os.environ.get('SLOW_SEEK') == '1':
    dec = subprocess.Popen(
        ['ffmpeg', '-loglevel', 'error', '-threads', '2', '-i', 'short.webm',
         '-ss', '%.3f' % (START / FPS),
         '-frames:v', str(END - START), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
        stdout=subprocess.PIPE)
else:
    dec = subprocess.Popen(
        ['ffmpeg', '-loglevel', 'error', '-threads', '2', '-ss', '%.3f' % (START / FPS), '-i', 'short.webm',
         '-frames:v', str(END - START), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
        stdout=subprocess.PIPE)

buf = b''
n = 0
n_faces = 0
t_det = 0.0
t_swap = 0.0
while True:
    c = dec.stdout.read(FRAME)
    if not c:
        break
    buf += c
    while len(buf) >= FRAME:
        img = np.frombuffer(buf[:FRAME], np.uint8).copy().reshape(H, W, 3)
        buf = buf[FRAME:]
        t1 = time.time()
        d, k = det.detect(img)
        t_det += time.time() - t1
        for i in range(min(len(d), 4)):
            t1 = time.time()
            img = swap_face(img, k[i])
            t_swap += time.time() - t1
            n_faces += 1
        enc.stdin.write(img.tobytes())
        n += 1
        if n % 200 == 0:
            drop_file_cache(OUT)
            drop_file_cache('short.webm')
            drop_file_cache('models/inswapper_128.onnx')

enc.stdin.close()
dec.stdout.close()
enc.wait()
dec.wait()
el = time.time() - t0
print('[%s] PART %d-%d done: frames=%d faces=%d det=%.0fs swap=%.0fs total=%.0fs rc_enc=%s' % (
    time.strftime('%H:%M:%S'), START, END, n, n_faces, t_det, t_swap, el, enc.returncode), flush=True)
sys.exit(0 if (n == END - START and enc.returncode == 0) else 1)
