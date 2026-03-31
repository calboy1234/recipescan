# ── RecipeScan Docker Image ───────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY init_db.py       .
COPY ocr_pipeline.py  .
COPY recipe_detector.py .
COPY app.py           .
COPY templates/       ./templates/
COPY static/          ./static/

# /photos → mount your image library here (read-only recommended).
#            Organise multiple image sources as subfolders so the web UI
#            can list and select them independently at runtime:
#
#              Host path              Container path
#              /mnt/user/photos/main  → /photos/main
#              /mnt/user/photos/archive → /photos/archive
#              /mnt/user/photos/test  → /photos/test
#
#            In Unraid, add each host folder as a separate path mapping
#            under the same container path parent (/photos).
#            The web UI auto-discovers subfolders and shows checkboxes.
#            Leave images directly in /photos root for single-source use.
#
# /data   → mount your persistent data directory here (read-write required).
VOLUME ["/photos", "/data"]

EXPOSE 5000

# MODE controls startup behaviour:
#   web  → init_db + start Flask web server (default)
#   ocr  → init_db + run OCR pipeline then exit
#   both → init_db + run OCR pipeline + start Flask
ENV MODE=web

CMD ["sh", "-c", "\
  python init_db.py && \
  if [ \"$MODE\" = 'ocr' ] || [ \"$MODE\" = 'both' ]; then \
    python ocr_pipeline.py; \
  fi && \
  if [ \"$MODE\" = 'web' ] || [ \"$MODE\" = 'both' ]; then \
    python app.py; \
  fi \
"]
