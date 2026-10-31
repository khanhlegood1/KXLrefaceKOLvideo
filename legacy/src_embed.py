import cv2, time
import numpy as np
from insightface.model_zoo import get_model
from insightface.app.common import Face

t0 = time.time()
det = get_model('models/det_10g.onnx', providers=['CPUExecutionProvider'])
det.prepare(ctx_id=-1, input_size=(640, 640))
rec = get_model('models/w600k_r50.onnx', providers=['CPUExecutionProvider'])
print('models loaded in %.1fs' % (time.time() - t0), flush=True)

photos = [
    '/home/user/uploads/IMG_2070.jpeg',
    '/home/user/uploads/IMG_5277.jpeg',
    '/home/user/uploads/IMG_9970.jpeg',
    '/home/user/uploads/beauty_1663410437875.jpeg',
]

def detect_faces(model, img):
    d, k = model.detect(img)
    faces = []
    for i in range(len(d)):
        faces.append(Face(bbox=d[i, :4], kps=k[i], det_score=float(d[i, 4])))
    return faces

embs = []
for p in photos:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    s = min(1.0, 1280 / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    faces = detect_faces(det, img)
    if not faces:
        print(p, 'NO FACE', flush=True)
        continue
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = rec.get(img, best)
    embs.append(emb)
    print(p, 'n_faces=%d score=%.3f bbox=%s' % (len(faces), best.det_score, np.round(best.bbox).astype(int).tolist()), flush=True)

avg = np.mean(embs, axis=0)
avg = avg / np.linalg.norm(avg)
np.save('src_embedding.npy', avg.astype(np.float32))
print('saved src_embedding.npy from %d faces, total %.1fs' % (len(embs), time.time() - t0), flush=True)
