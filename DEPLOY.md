# Deploying DocVision AI

DocVision ships as **one Docker image**: the API (FastAPI + Tesseract) also serves the web UI.
Deploy it once and you get one URL that works as a website **and** installs as an app (PWA).

## Option A (recommended): one service, one URL

1. Push this folder to GitHub:
   ```
   git init
   git add .
   git commit -m "DocVision AI"
   git branch -M main
   git remote add origin https://github.com/<you>/docvision-ai.git
   git push -u origin main
   ```
2. **Render**: New > Blueprint > select the repo (it reads `render.yaml`). Or New > Web Service > Runtime: Docker.
   Any Docker host works the same way (Railway, Fly.io, Google Cloud Run): point it at the `Dockerfile`.
3. Wait for the build, then open `https://<your-service>/api/health` -> expect `{"ok":true,...,"tesseract":true}`.
4. Open `https://<your-service>/` -> that is the app. Share this link.

No environment variables are required for Option A.

## Option B: UI on Vercel, API elsewhere

1. Deploy the API as in Option A, and set `ALLOWED_ORIGINS=https://<your-app>.vercel.app` on it.
2. Vercel: Import the repo, **Root Directory = `frontend`**, Framework = Vite,
   env var `VITE_API_URL=https://<your-api-url>` (no trailing slash). Deploy.
3. Update `ALLOWED_ORIGINS` once you know the final Vercel URL, then redeploy the API.

Why the API can't also live on Vercel/Supabase: it needs the Tesseract binary and heavy native libraries,
which serverless functions don't provide. Supabase is for auth/database/storage (a later phase).

## Run it locally with Docker (no Python/Node/Tesseract install)

```
docker compose up --build
```
Open http://localhost:8000

## Install it as an app (PWA)

Needs the site to be served over HTTPS (Render and Vercel do this automatically) or localhost.

- **Chrome / Edge (desktop):** click the **Install app** button in the header, or the install icon in the address bar.
- **Android (Chrome):** the **Install app** button, or menu > Install app.
- **iPhone / iPad (Safari):** Share > **Add to Home Screen** (iOS has no install button).

The installed app opens in its own window. The interface loads offline, but analysing a document
always needs the server.

## Notes

- Uploads are capped at 25 MB and a page limit (see `docvision/service.py`).
- The API has no login or rate limiting yet. Before sharing the link widely, add auth and limits
  (Supabase Auth is the planned next step), or anyone can spend your CPU.
- Free hosting tiers usually sleep when idle, so the first request can be slow, and 512 MB RAM is too little for big PDFs.
- Optional neural backends (torch, transformers) are not in the image. Add them to `requirements.txt` only if you need them; the image gets much larger.
