"""Transport-agnostic service layer used by the API (and testable without FastAPI)."""
from __future__ import annotations
import base64, io, os, tempfile, time
from PIL import Image
from .pipeline import process_image, baseline_ocr_only, process_document
from .ingest import load_pages
from .schema import Document
from .structure import link_pages
from .export import to_markdown

MAX_PAGES = 12
MAX_BYTES = 25 * 1024 * 1024


def _png_b64(img: Image.Image, max_side=1600) -> str:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze_file(data: bytes, filename: str, mode: str = "structured", dpi: int = 150):
    """mode: 'structured' (full pipeline) | 'ocr_only' (baseline) | 'both'."""
    if len(data) > MAX_BYTES:
        raise ValueError("File too large (max 25 MB).")
    ext = os.path.splitext(filename)[1].lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data); path = f.name
    try:
        t0 = time.time()
        structured, baseline, images = [], [], []
        for no, pil, native in load_pages(path, dpi, MAX_PAGES):
            images.append(_png_b64(pil))
            if mode in ("structured", "both"):
                structured.append(process_image(pil, no, native_words=native))
            if mode in ("ocr_only", "both"):
                baseline.append(baseline_ocr_only(pil, no))
        out = {"filename": filename, "pages": len(images), "images": images,
               "seconds": round(time.time() - t0, 2)}
        if structured:
            d = Document(pages=structured); link_pages(d)
            out["structured"] = d.to_dict(); out["markdown"] = to_markdown(d)
        if baseline:
            b = Document(pages=baseline)
            out["ocr_only"] = b.to_dict()
        return out
    finally:
        os.unlink(path)
