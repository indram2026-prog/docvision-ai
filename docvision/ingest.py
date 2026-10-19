"""Load PDFs and images into page images (+ native text layer when available)."""
from __future__ import annotations
from PIL import Image
from .ocr import native_pdf_words

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_pages(path: str, dpi: int = 150, max_pages=None, prefer_native=True):
    """Yield (page_no, PIL image, native_words | None)."""
    low = path.lower()
    if any(low.endswith(e) for e in IMG_EXT):
        yield 0, Image.open(path).convert("RGB"), None
        return
    if not low.endswith(".pdf"):
        raise ValueError(f"Unsupported file type: {path}")
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    n = len(pdf) if max_pages is None else min(len(pdf), max_pages)
    scale = dpi / 72.0
    for i in range(n):
        pil = pdf[i].render(scale=scale).to_pil().convert("RGB")
        native = None
        if prefer_native:
            w = native_pdf_words(path, i, scale)
            native = w if len(w) >= 5 else None   # scanned pages have no usable text layer -> OCR
        yield i, pil, native
