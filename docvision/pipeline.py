"""End-to-end pipeline: image/PDF -> structured Document."""
from __future__ import annotations
import time
import cv2
import numpy as np
from PIL import Image
from . import layout as L
from .ocr import tesseract_lines, ocr_cell, native_pdf_words, lines_from_words, make_line
from . import borderless as BL
from .fieldtypes import type_value
from .preprocess import prepare, map_bbox_back
from .schema import Element, Page, Document, Line, Word
from .structure import link_pages


def _cells_from_grid(gray, xs, ys, words=None):
    cells, boxes, confs = [], [], []
    for r in range(len(ys) - 1):
        row, brow = [], []
        for c in range(len(xs) - 1):
            bb = (xs[c] + 3, ys[r] + 3, xs[c + 1] - 2, ys[r + 1] - 2)
            if words is not None:
                ws = [w for w in words if bb[0] <= (w.bbox[0] + w.bbox[2]) / 2 <= bb[2]
                      and bb[1] <= (w.bbox[1] + w.bbox[3]) / 2 <= bb[3]]
                txt = " ".join(w.text for w in sorted(ws, key=lambda w: (w.bbox[1] // 10, w.bbox[0])))
                confs.append(1.0 if txt else 0.0)
            else:
                txt, cf = ocr_cell(gray[bb[1]:bb[3], bb[0]:bb[2]])
                confs.append(cf)
            row.append(txt); brow.append(bb)
        cells.append(row); boxes.append(brow)
    return cells, boxes, float(np.mean(confs)) if confs else 0.0


def refine_large_line(gray, line, target_h=26):
    """Large type OCRs worse at 150 dpi: re-read the crop at body-text scale, keep it if more confident."""
    x0, y0, x1, y1 = line.bbox
    crop = gray[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
    if crop.size == 0:
        return line
    f = target_h / max(1, line.height)
    if f >= 0.9:
        return line
    small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    txt, cf = ocr_cell(small)
    if txt and cf > line.conf and abs(len(txt) - len(line.text)) <= 3:
        return Line(words=line.words, bbox=line.bbox, text=txt, height=line.height)
    return line


def process_image(img, page_no=0, native_words=None, use_tables=True, use_figures=True, use_forms=True,
                  use_borderless=True):
    """Structured analysis of one page image. Returns a Page whose boxes are in the ORIGINAL image frame."""
    rgb, orig, skew = prepare(img)
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hor, ver = L.rule_masks(gray)

    tables = L.detect_tables(hor, ver) if use_tables else []
    # figure-first arbitration: grid-like edges inside a pictorial region are not a table
    if use_figures and tables:
        pre = [f[0] for f in L.detect_figures(rgb, [])]
        tables = [t for t in tables if max([L.containment(t[0], f) for f in pre] + [0]) < 0.5]
    tboxes = [t[0] for t in tables]
    figs = L.detect_figures(rgb, tboxes) if use_figures else []
    fboxes = [f[0] for f in figs]
    uls, ul_mask = L.underlines(hor, tboxes, fboxes)

    # text source: native PDF words (if given) else OCR on an image with non-text regions blanked
    if native_words is not None:
        words = [w for w in native_words
                 if not any(b[0] <= (w.bbox[0] + w.bbox[2]) / 2 <= b[2] and b[1] <= (w.bbox[1] + w.bbox[3]) / 2 <= b[3]
                            for b in tboxes + fboxes)]
        lines = lines_from_words(words)
        src = "pdf-native"
    else:
        clean = rgb.copy()
        for (x0, y0, x1, y1) in tboxes + fboxes:
            clean[max(0, y0 - 4):y1 + 5, max(0, x0 - 4):x1 + 5] = 255
        clean[cv2.dilate(ul_mask, np.ones((3, 3), np.uint8)) > 0] = 255
        lines = L.merge_row_fragments(tesseract_lines(clean))
        L.mark_bold(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY), lines)
        src = "tesseract"

    # borderless tables: word-alignment rivers; their words are removed from the text flow
    bl_tables = []
    if use_tables and use_borderless and lines:
        bl_tables = BL.detect_borderless_tables([w for l in lines for w in l.words], exclude=tboxes + fboxes)
        if bl_tables:
            used = {id(w) for t in bl_tables for w in t["words"]}
            lines = [make_line(ws) for ws in ([w for w in l.words if id(w) not in used] for l in lines) if ws]

    classified = L.classify_lines(lines, uls if use_forms else [])
    if native_words is None:
        classified = [(refine_large_line(gray, l) if k in ("title", "heading") else l, k, e)
                      for l, k, e in classified]
    if not use_forms:
        classified = [(l, "text" if k == "form" else k, e) for l, k, e in classified]
    els = L.build_elements(classified, page=page_no, source=src)

    for (bb, ys, xs) in tables:
        cells, boxes, cf = _cells_from_grid(gray, xs, ys, native_words if native_words is not None else None)
        els.append(Element(type="table", bbox=bb, page=page_no, cells=cells, cell_boxes=boxes, conf=cf,
                           text="\n".join(" | ".join(r) for r in cells), source="grid-rules+" + src))
    els += BL.to_elements(bl_tables, page_no, "word-alignment+" + src)
    for (bb, cf) in figs:
        els.append(Element(type="figure", bbox=bb, page=page_no, conf=cf, source="density-saturation"))

    # first page: the topmost centred heading in the top quarter is the document title
    if page_no == 0 and not any(e.type == "title" for e in els):
        top = min((e for e in els if e.type in ("heading", "paragraph")), key=lambda e: e.bbox[1], default=None)
        if (top is not None and top.type == "heading" and top.bbox[3] < 0.25 * H
                and abs((top.bbox[0] + top.bbox[2]) / 2 - W / 2) < 0.08 * W):
            top.type, top.level = "title", 1

    for e in els:
        if e.type == "form_field":
            e.value_type = type_value(e.value or "", e.key or "")
    els = L.xy_cut(els)
    for i, e in enumerate(els):
        e.order = i
        e.bbox = map_bbox_back(e.bbox, rgb.shape, skew)
        if e.cell_boxes:
            e.cell_boxes = [[map_bbox_back(b, rgb.shape, skew) for b in r] for r in e.cell_boxes]
    return Page(number=page_no, width=orig.shape[1], height=orig.shape[0], elements=els,
                skew_deg=skew, ocr_source=src)


def baseline_ocr_only(img, page_no=0):
    """OCR-only baseline: plain text lines grouped into paragraphs. No layout, tables, figures or forms."""
    rgb = prepare(img)[0]
    lines = tesseract_lines(rgb)
    cls = [(l, "text", None) for l in lines]
    els = L.build_elements(cls, page=page_no, source="ocr-only")
    for i, e in enumerate(L.xy_cut(els)):
        e.order = i
    return Page(number=page_no, width=rgb.shape[1], height=rgb.shape[0], elements=els, ocr_source="tesseract")


def process_document(path: str, dpi: int = 150, max_pages: int | None = None, prefer_native=True) -> Document:
    from .ingest import load_pages
    pages = []
    for no, pil, native in load_pages(path, dpi, max_pages, prefer_native):
        pages.append(process_image(pil, no, native_words=native))
    doc = Document(pages=pages)
    link_pages(doc)
    return doc
