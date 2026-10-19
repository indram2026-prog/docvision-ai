# Setup guide: run DocVision AI on any machine

This guide gets DocVision AI running on a fresh Windows, macOS or Linux machine, and optionally gives it a public URL through a free tunnel. No cloud account and no card are needed.

The app is one server: FastAPI serves both the API and the built React UI on port **8000**.

> **Status:** the commands below have not all been run on every OS. If a step fails, open an issue or paste the error message; the first run on a new machine is the real test.

## 1. Pick a route

| Route | Best when | You install |
|---|---|---|
| **A. Docker** | the machine already has Docker | Docker only |
| **B. Native** | no Docker | Python 3.12, Node 20, Tesseract, git |

Route A is the simplest: the Dockerfile builds the UI and installs Tesseract for you.

## 2. Get the code

```bash
git clone https://github.com/indram2026-prog/docvision-ai
cd docvision-ai
```

## 3A. Route A: Docker

```bash
docker build -t docvision .
docker run -d --name docvision -p 8000:8000 --restart unless-stopped docvision
```

Stop or remove it later:

```bash
docker stop docvision
docker rm docvision
```

## 3B. Route B: native install

### Install the tools

**Windows (PowerShell):**

```powershell
winget install Python.Python.3.12
winget install OpenJS.NodeJS.LTS
winget install Git.Git
winget install UB-Mannheim.TesseractOCR
```

Tesseract must be on your `PATH` (the app has no setting for its location). Add `C:\Program Files\Tesseract-OCR` to PATH, then open a **new** terminal and check:

```powershell
tesseract --version
```

**macOS (Homebrew):**

```bash
brew install python@3.12 node tesseract git
```

**Debian / Ubuntu:**

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nodejs npm tesseract-ocr fonts-dejavu-core git
```

Check the versions: `python --version` should be 3.12 or newer (the Docker image uses 3.12) and `node --version` should be 20 or newer. On Ubuntu, the `nodejs` package can be older than 20. If so, install Node 20 from nodejs.org instead.

### Build and run

Run these from the repo folder (`docvision-ai`).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
uvicorn docvision.api:app --host 0.0.0.0 --port 8000
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` first.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
(cd frontend && npm install && npm run build)
uvicorn docvision.api:app --host 0.0.0.0 --port 8000
```

## 4. Check that it works

Open <http://localhost:8000/api/health>. You should see something like:

```json
{"ok": true, "tesseract": true}
```

- `"tesseract": false` means Tesseract is not installed or not on `PATH` (Route B), or the image build is wrong (Route A).

Then open <http://localhost:8000/>, click **Try a generated sample page**, and upload a PDF.

## 5. Give it a public URL (optional, free)

A Cloudflare quick tunnel exposes your local port with no account.

**Install `cloudflared`:**

```powershell
# Windows
winget install --id Cloudflare.cloudflared
```

```bash
# macOS
brew install cloudflared

# Debian / Ubuntu (x86-64)
curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

**Start the tunnel** (the app from step 3 must already be running):

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a link like `https://random-words.trycloudflare.com`. Open that link from any device.

### What to know about the tunnel

- The link works only while this machine is on, online, and running **both** the app and the tunnel.
- A quick-tunnel URL changes every time you restart `cloudflared`. A fixed URL needs a free Cloudflare account and a domain.
- **There is no login and no rate limit.** Anyone with the link can use your CPU and upload documents. Share it only with people you trust, and do not upload confidential documents.
- Cloudflare quick tunnels are meant for testing, not production, and Cloudflare may change or limit the service.

## 6. Settings

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port the Docker image listens on |
| `WEB_CONCURRENCY` | `1` | Worker processes. Each loads OCR and OpenCV into its own memory, so keep it low on small machines |
| `ALLOWED_ORIGINS` | unset | Only needed if the UI is hosted on a different domain from the API |

Docker example: `docker run -p 8000:8000 -e WEB_CONCURRENCY=2 docvision`.

## 7. Update to the latest code

```bash
git pull
```

Then rebuild the Docker image (`docker build -t docvision .` and re-run it) or, on Route B, re-run `npm run build` in `frontend` and restart `uvicorn`.

## 8. Troubleshooting

| Problem | Likely fix |
|---|---|
| `tesseract is not installed or it's not in your PATH` | Install Tesseract and open a new terminal; check `tesseract --version` |
| `localhost:8000` shows the API but no UI | `frontend/dist` is missing. Run `npm run build` in `frontend`, then restart |
| `port is already allocated` / address in use | Another program uses 8000. Use another port, e.g. `-p 8080:8000` (Docker) or `--port 8080` (uvicorn) |
| Out of memory on large PDFs | Keep `WEB_CONCURRENCY=1`, close other programs, or use a machine with more RAM |
| `pip install` fails building `opencv` or `scipy` | Use Python 3.12 and upgrade pip: `python -m pip install --upgrade pip` |
| The tunnel link opens but shows an error | The app is not running on port 8000, or the tunnel points at a different port |
