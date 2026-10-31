FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl unzip libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yt-dlp needs a JS runtime for YouTube
RUN curl -fsSL https://github.com/denoland/deno/releases/download/v2.1.4/deno-x86_64-unknown-linux-gnu.zip \
    -o /tmp/deno.zip && unzip -q /tmp/deno.zip -d /usr/local/bin && rm /tmp/deno.zip

COPY . .

RUN useradd -m appuser \
    && mkdir -p /app/models /app/jobs \
    && chown -R appuser:appuser /app
USER appuser
ENV MODELS_DIR=/app/models JOB_ROOT=/app/jobs

EXPOSE 8000

# download models on first start (cached in the /app/models volume)
CMD ["sh", "-c", "python scripts/download_models.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
