# ---------- 1) build the React UI ----------
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- 2) API + UI in one image ----------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
# tesseract-ocr: the OCR engine. fonts-dejavu-core: used by the synthetic sample page (/api/sample).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docvision ./docvision
COPY --from=ui /ui/dist ./frontend/dist
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'))"
# Hosts like Render/Railway/Cloud Run inject $PORT. WEB_CONCURRENCY = worker processes (each uses its own RAM).
CMD ["sh", "-c", "uvicorn docvision.api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
