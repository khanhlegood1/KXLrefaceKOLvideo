# Tài liệu triển khai — KXL FaceSwap

Tài liệu này hướng dẫn đưa ứng dụng lên chạy production. Có 3 cách:
**VPS + systemd** (khuyến nghị), **Docker**, hoặc **front-end Vercel + backend riêng**.

---

## 1. Yêu cầu hệ thống

| Hạng | CPU | RAM | SSD | Thời gian 1 video 2:30 |
|------|-----|-----|-----|------------------------|
| Tối thiểu (đã kiểm chứng) | 2 vCPU | 2 GB | 5 GB trống | ~40 phút |
| Khuyên dùng | 4 vCPU | 4–8 GB | 20 GB | ~15–20 phút |
| GPU (tùy chọn) | — | 4 GB | 20 GB | ~5–8 phút |

- **Python 3.10 – 3.12** (cài 3.13 cũng chạy được).
- **ffmpeg** trong PATH.
- **JS runtime cho yt-dlp** (bắt buộc với YouTube): `deno` hoặc `node`
  — YouTube giờ yêu cầu chạy JavaScript khi extract, không có runtime sẽ lỗi.

### Cài đặt cơ bản (Debian/Ubuntu)

```bash
# ffmpeg
apt update && apt install -y ffmpeg

# JS runtime cho yt-dlp (chọn deno — nhỏ nhất)
curl -fsSL https://deno.land/install.sh | sh        # vào PATH ở ~/.deno/bin
# hoặc: apt install -y nodejs

# Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Tải model (1 lần duy nhất, ~760 MB)

```bash
python3 scripts/download_models.py
# mặc định về ./models — hoặc chỉ định: --models-dir /opt/kxl/models
# các file: det_10g.onnx, w600k_r50.onnx, inswapper_128.onnx, emap_fp32.npy
```

> Model **không nằm trong repo** (file lớn). Mỗi lần cài mới chạy lệnh trên.

---

## 2. Chạy thử cục bộ

```bash
cd KXLrefaceKOLvideo
export MODELS_DIR=$PWD/models
uvicorn app.main:app --host 0.0.0.0 --port 8000
# mở http://localhost:8000
```

Hoặc CLI không cần web:

```bash
python3 scripts/one_shot_swap.py --photos a.jpg b.jpg \
    --url "https://youtube.com/shorts/XXXX" --out result.mp4
```

---

## 3. Cách 1 — VPS + systemd + nginx (khuyến nghị)

### 3.1. Tạo user & thư mục

```bash
useradd -m -s /bin/bash kxl
mkdir -p /opt/kxl && cp -r KXLrefaceKOLvideo /opt/kxl/app
chown -R kxl:kxl /opt/kxl
```

### 3.2. Cài môi trường (chạy bằng user kxl)

```bash
sudo -u kxl -H bash -c 'cd /opt/kxl/app && \
  python3 -m venv .venv && . .venv/bin/activate && \
  pip install -r requirements.txt && \
  python3 scripts/download_models.py --models-dir /opt/kxl/models'
```

### 3.3. Unit systemd

`/etc/systemd/system/kxl-faceswap.service`:

```ini
[Unit]
Description=KXL FaceSwap API
After=network.target

[Service]
User=kxl
WorkingDirectory=/opt/kxl/app
Environment=MODELS_DIR=/opt/kxl/models
Environment=JOB_ROOT=/opt/kxl/jobs
Environment=PORT=8000
Environment=MAX_ACTIVE_JOBS=1
Environment=JOB_TTL_HOURS=12
ExecStart=/opt/kxl/app/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
# Giới hạn tài nguyên an toàn cho máy 2GB RAM
MemoryMax=3G
# Chặn job chạy quá 4 giờ
RuntimeMaxSec=14400

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kxl-faceswap
journalctl -u kxl-faceswap -f     # xem log
```

### 3.4. Nginx reverse proxy (https + upload lớn)

`/etc/nginx/sites-available/kxl`:

```nginx
server {
    listen 80;
    server_name kxl.example.com;

    client_max_body_size 150m;          # 10 ảnh x 25MB

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;        # poll job không timeout sớm
    }
}
```

```bash
ln -s /etc/nginx/sites-available/kxl /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
# https: certbot --nginx -d kxl.example.com
```

> **Lưu ý mạng:** IP VPS châu Âu/Mỹ thường bị YouTube chặn "sign in to confirm
> you're not a bot" khi tải video (đã gặp thực tế). Hai cách xử lý:
> - đặt `Environment=YTDLP_COOKIE_FILE=/opt/kxl/cookies.txt` (file cookies xuất
>   từ trình duyệt của bạn, xem mục 6);
> - hoặc chạy VPS ở region khác / dùng proxy cho yt-dlp.

---

## 4. Cách 2 — Docker

Repository có sẵn `Dockerfile` + `docker-compose.yml`. Model được tải **khi
container khởi động lần đầu** và lưu trong volume `kxl_models`.

```bash
cd KXLrefaceKOLvideo
docker compose up -d --build
# logs: docker compose logs -f
# mở http://localhost:8000
```

Giới hạn RAM cho container trong `docker-compose.yml`
(`mem_limit: 3g`) — khớp với kiến trúc per-part của pipeline.

### GPU (tùy chọn, NVIDIA)

```dockerfile
# thay FROM python:3.11-slim bằng:
FROM nvidia/cuda:12.2.2-base-ubuntu22.04
# và trong requirements dùng: onnxruntime-gpu>=1.17 thay vì onnxruntime
```

Rồi chạy: `docker compose run --gpus all ...`
Lưu ý: với CPU yếu, fp32 + 2 thread đã là điểm cân bằng tốt nhất; GPU chủ yếu
giảm thời gian cho người dùng cao cấp.

---

## 5. Cách 3 — Frontend trên Vercel + backend riêng

Frontend là **1 file HTML tĩnh** (`app/static/index.html`) — deploy Vercel cực
dễ, backend vẫn chạy ở VPS/Docker như mục 3/4.

1. Tạo project Vercel (Import repo, framework: **None**, output:
   `app/static` — hoặc copy `app/static/index.html` ra root và đặt tên
   `index.html`).
2. Mở file `index.html`, sửa hằng số ở đầu `<script>`:

   ```js
   const API_BASE = "https://kxl.example.com";   // address backend của bạn
   ```

3. Deploy. Backend phải cho phép CORS từ domain Vercel:

   ```
   CORS_ORIGINS=https://your-app.vercel.app
   ```

Mẹo: tách `API_BASE` sang file `config.js` nhỏ để không phải sửa lại mỗi lần
thay backend.

---

## 6. Xử lý YouTube chặn bot ("Sign in to confirm you're not a bot")

Nguyên nhân: YouTube chặn IP datacenter / truy cập anonymous quá nhiều.
Cách xử lý theo thứ tự ưu tiên:

1. **File cookies của bạn** (đơn giản nhất, dùng 1 tài khoản Google bình thường):
   - Cài extension "Get cookies.txt LOCALLY" → mở youtube.com (đã đăng nhập)
     → export ra `cookies.txt`;
   - đặt file đó lên server, rồi: `export YTDLP_COOKIE_FILE=/opt/kxl/cookies.txt`
     (systemd: dòng `Environment=` như mục 3.3).
   - Cookies có tuổi đời vài tháng; hết thì export lại.
2. **Thay client extractor** (thỉnh thoảng hiệu quả, không cần cookies):
   ```bash
   python3 -m yt_dlp --extractor-args "youtube:player_client=tv" <url>
   ```
   Muốn mặc định, sửa dict `opts` trong `app/worker.py:download_youtube`.
3. **Chạy backend ở IP khác** (residential proxy / VPS khác region).
4. Cho phép người dùng **tải file video sẵn** (chưa có trong UI — có thể thêm
   input file video vào frontend, backend nhận `video` thay vì `url`).

---

## 7. Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|------|----------|----------|
| `MODELS_DIR` | `<repo>/models` | Thư mục chứa ONNX |
| `JOB_ROOT` | `<repo>/jobs` | Thư mục scratch từng job |
| `PART_SIZE` | `600` | Số frame / tiến trình (giảm nếu RAM < 2GB: 300) |
| `MAX_PART_RETRIES` | `1` | Số lần retry 1 phần (seek chậm hơn) |
| `MAX_ACTIVE_JOBS` | `1` | Số job song song (giữ 1 cho máy 2GB) |
| `MAX_PHOTOS` | `10` | Số ảnh tối đa mỗi job |
| `JOB_TTL_HOURS` | `24` | Tự xoá job sau N giờ |
| `CORS_ORIGINS` | `*` | Dừng `*` khi production: liệt kê domain front-end |
| `YTDLP_COOKIE_FILE` | — | File cookies.txt cho YouTube |

---

## 8. Troubleshooting

| Triệu chứng | Nguyên nhân / xử lý |
|---|---|
| `Memory cgroup out of memory` | Giảm `PART_SIZE=300`, tăng RAM, hoặc giảm `MAX_ACTIVE_JOBS=1`. Không nên chạy 2 job song song trên 2GB. |
| `No supported JavaScript runtime found` khi tải YouTube | Chưa cài deno/node (mục 1). |
| `Sign in to confirm you're not a bot` | Xem mục 6. |
| Video kết quả méo/blur ở phần đầu | Phần đầu dùng fast-seek; pipeline đã có fallback slow-seek tự động. Nếu lỗi lặp lại tăng `MAX_PART_RETRIES=2`. |
| `ffmpeg: command not found` | `apt install ffmpeg` và chắc chắn ffmpeg trong PATH của user chạy uvicorn. |
| Import `insightface` lỗi `matplotlib`/`easyocr` | `pip install -U matplotlib easyocr` (insightface kéo theo, một số hệ thiếu). |
| Job mất sau khi restart server | Job chỉ lưu trong RAM + `JOB_ROOT` (không có DB). Với nhu cầu production nặng, thêm SQLite lưu `state.json`. |
| Upload 413 (Nginx) | Tăng `client_max_body_size` (mục 3.4). |

---

## 9. Bảo mật (tóm tắt cho production)

- Không chạy public *trần* (không auth): ai có URL máy cũng dùng được.
  Thêm 1 lớp đăng nhập đơn giản (Basic Auth ở nginx, hoặc 1 middleware FastAPI
  kiểm tra `Authorization: Bearer <TOKEN>` cho nhóm `/api/jobs*`).
- `CORS_ORIGINS` không để `*` khi có front-end Vercel.
- Giới hạn `client_max_body_size` + số file upload (đã có trong code).
- Cookies.txt chứa thông tin đăng nhập YouTube — quyền file 600, user kxl.
- Xoá job định kỳ (đã có `JOB_TTL_HOURS`) + cron hàng ngày dọn `jobs/`.
