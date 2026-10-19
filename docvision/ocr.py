"""Text sources: Tesseract OCR (any image) and native PDF text (born-digital)."""
from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
import pytesseract
from pytesseract import Output
from .schema import Word, Line, union

_JUNK = re.compile(r"^[\W_]+$")


def _clean(words):
    return [w for w in words if w.text.strip() and not (_JUNK.match(w.text) and len(w.text) > 1)]


def make_line(words) -> Line:
    words = sorted(words, key=lambda w: w.bbox[0])
    bb = union([w.bbox for w in words])
    return Line(words=words, bbox=bb, text=" ".join(w.text for w in words),
                height=max(w.bbox[3] - w.bbox[1] for w in words))


def tesseract_lines(rgb: np.ndarray, psm: int = 3, min_conf: float = 25.0):
    d = pytesseract.image_to_data(rgb, output_type=Output.DICT, config=f"--psm {psm}")
    groups = defaultdict(list)
    for i, t in enumerate(d["text"]):
        if not t.strip() or float(d["conf"][i]) < min_conf:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        groups[(d["block_num"][i], d["par_num"][i], d["line_num"][i])].append(
            Word(t, (x, y, x + w, y + h), float(d["conf"][i]) / 100))
    lines = []
    for ws in groups.values():
        ws = _clean(ws)
        if ws:
            lines.append(make_line(ws))
    return sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))


def ocr_cell(gray_crop: np.ndarray):
    """OCR a single table cell; returns (text, mean_conf)."""
    if gray_crop.size == 0:
        return "", 0.0
    pad = np.pad(gray_crop, 12, constant_values=255)
    d = pytesseract.image_to_data(pad, output_type=Output.DICT, config="--psm 6")
    toks = [(d["text"][i], float(d["conf"][i])) for i in range(len(d["text"]))
            if d["text"][i].strip() and float(d["conf"][i]) >= 0]
    if not toks:
        return "", 0.0
    return " ".join(t for t, _ in toks), float(np.mean([c for _, c in toks])) / 100


def native_pdf_words(pdf_path: str, page_index: int, scale: float):
    """Words from the PDF text layer (exact for born-digital files). scale = render_dpi / 72."""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False, extra_attrs=["size", "fontname"])
    def fix(t):  # unmapped glyphs: (cid:127) is the common bullet; drop other unknown CIDs
        t = t.replace("(cid:127)", "•")
        return re.sub(r"\(cid:\d+\)", "", t)
    words = [dict(w, text=fix(w["text"])) for w in words]
    words = [w for w in words if w["text"].strip()]
    return [Word(w["text"], (int(w["x0"] * scale), int(w["top"] * scale),
                             int(w["x1"] * scale), int(w["bottom"] * scale)), 1.0,
                 any(k in str(w.get("fontname", "")).lower() for k in ("bold", "black", "heavy", "semibold")))
            for w in words]


def lines_from_words(words, y_tol: float = 0.5):
    """Cluster words into lines by vertical centre proximity (for native text)."""
    words = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0]))
    lines, cur = [], []
    for w in words:
        cy = (w.bbox[1] + w.bbox[3]) / 2
        if cur:
            ref = cur[-1]
            rc = (ref.bbox[1] + ref.bbox[3]) / 2
            if abs(cy - rc) > y_tol * max(ref.bbox[3] - ref.bbox[1], w.bbox[3] - w.bbox[1]):
                lines.append(cur); cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for ws in lines:
        # split on very large horizontal gaps (columns)
        ws = sorted(ws, key=lambda w: w.bbox[0])
        seg = [ws[0]]
        for a, b in zip(ws, ws[1:]):
            hgt = max(a.bbox[3] - a.bbox[1], 1)
            if b.bbox[0] - a.bbox[2] > 4.0 * hgt:
                out.append(make_line(seg)); seg = []
            seg.append(b)
        out.append(make_line(seg))
    return sorted(out, key=lambda l: (l.bbox[1], l.bbox[0]))
