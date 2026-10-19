"""FastAPI app. Run: uvicorn docvision.api:app --reload   (needs: pip install fastapi uvicorn python-multipart)"""
from __future__ import annotations
import io, os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from . import __version__
from .service import analyze_file

app = FastAPI(title="DocVision AI", version=__version__)
# Comma-separated list of allowed browser origins. Only needed when the UI is hosted on a different
# domain than the API (e.g. UI on Vercel, API on Render). Single-container deploys don't need it.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    import shutil
    return {"ok": True, "version": __version__, "tesseract": bool(shutil.which("tesseract"))}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), mode: str = Query("both", pattern="^(structured|ocr_only|both)$")):
    data = await file.read()
    try:
        return await run_in_threadpool(analyze_file, data, file.filename or "upload.png", mode)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # corrupt files, OCR failures
        raise HTTPException(422, f"Could not process document: {type(e).__name__}: {e}")


@app.get("/api/sample")
async def sample(seed: int = 7):
    """Generate a synthetic demo page (no upload needed)."""
    from .synth import make_page
    img, _ = make_page(seed, first=True, force_table=True)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return await run_in_threadpool(analyze_file, buf.getvalue(), f"sample_{seed}.png", "both")


_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
