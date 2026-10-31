# KXL FaceSwap (KXLrefaceKOLvideo)

Thay mặt của bạn vào bất kỳ video YouTube Shorts nào — giữ nguyên âm thanh,
khuôn hình và tốc độ (60fps) của video gốc.

- **Engine**: insightface — detector SCRFD `det_10g` (640×640) + swapper
  `inswapper_128` (fp32) chạy trên **onnxruntime CPU** thuần (không cần GPU,
  không cần PyTorch).
- **Kiến trúc chịu máy yếu**: video dài được cắt thành các *phần* ~600 frame;
  mỗi phần chạy trong **một tiến trình Python riêng** nên bộ nhớ được hệ điều
  hành thu hồi hoàn toàn giữa các phần → video 2:30 phút chạy ổn định trên
  máy 2 vCPU / 2 GB RAM.
- **Đã kiểm chứng**: thay mặt toàn bộ video 2:30 (9053 frame, 1080×1080@60fps)
  trên máy 2 vCPU / 2GB — thời gian thực tế ~40 phút, chất lượng thay mặt tốt
  ở cả cảnh đội mũ bảo hiểm.

## Thành phần

```
app/
  main.py          # FastAPI: /api/jobs + serve frontend
  worker.py        # Điều phối job: tải YT → nhận diện ảnh → chạy parts → ghép + âm thanh
  part_runner.py   # Runner từng phần (một tiến trình = một phần, cách ly bộ nhớ)
  pipeline.py      # Logic core: load model, latent, paste-back insightface, embedding ảnh
  static/index.html# Frontend (1 file, không build): upload ảnh + link YouTube + tiến độ
scripts/
  download_models.py  # Tải 1 lần duy nhất ~760MB model
  one_shot_swap.py    # CLI: 1 lệnh thay mặt mà không cần web UI
legacy/
  part_worker.py, run_parts.py, src_embed.py, dropcache.py
                   # Script gốc dùng để chạy thử nghiệm session đầu (đã kiểm chứng)
```

## Chạy nhanh (CPU)

```bash
# 1) cài dependencies (khuyến nghị Python 3.10+, venv)
pip install -r requirements.txt
ffmpeg -version   # cần ffmpeg trong PATH

# 2) tải model (1 lần, ~760MB)
python3 scripts/download_models.py

# 3) CLI — thay mặt vào 1 video
python3 scripts/one_shot_swap.py \
    --photos uploads/me1.jpg uploads/me2.jpg \
    --url "https://youtube.com/shorts/XXXXXXXX" \
    --out result.mp4

# 4) ...hoặc chạy web app
uvicorn app.main:app --host 0.0.0.0 --port 8000
# mở http://localhost:8000
```

## API

| Method | Đường dẫn                    | Mô tả                                   |
|--------|------------------------------|-----------------------------------------|
| POST   | `/api/jobs`                  | multipart: `url` + `photos[]` → `{job_id}` |
| GET    | `/api/jobs/{id}`             | trạng thái: `queued→downloading→embedding→swapping(part x/y, frame, eta)→muxing→done` |
| POST   | `/api/jobs/{id}/cancel`      | hủy job                                  |
| GET    | `/api/jobs/{id}/video`       | tải video kết quả (mp4)                  |
| DELETE | `/api/jobs/{id}`             | xoá job + file                           |
| GET    | `/api/health`                | liveness + model đã có chưa              |

Biến môi trường chính: `MODELS_DIR`, `JOB_ROOT`, `PART_SIZE` (mặc định 600),
`MAX_ACTIVE_JOBS`, `JOB_TTL_HOURS`, `CORS_ORIGINS`, `YTDLP_COOKIE_FILE` (cookies
YouTube — xem DEPLOYMENT.md).

## Triển khai

Xem **[DEPLOYMENT.md](DEPLOYMENT.md)** — có sẵn 3 cách:
1. **VPS + systemd** (khuyến nghị) — kèm file unit, nginx reverse proxy;
2. **Docker** (Dockerfile + docker-compose kèm theo);
3. **Frontend lên Vercel** + backend riêng (hướng dẫn kết nối).

## Lưu ý

- Chỉ thay mặt của **chính bạn** hoặc với **sự đồng ý** của người trong video.
- Kết quả video giữ nguyên bản quyền thuộc về chủ sở hữu video gốc.
- YouTube có thể chặn tải video từ một số IP (datacenter) — hướng dẫn xử lý
  cookies trong DEPLOYMENT.md.
