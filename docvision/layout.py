"""Classical layout analysis: geometry + typography rules. Deterministic and CPU-fast.
Neural detectors can replace any function here through pipeline backends."""
from __future__ import annotations
import re
import cv2
import numpy as np
from .schema import Element, Line, union

BULLET_RE = re.compile(r"^\s*(?:[•▪◦●■\-–—*]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+(\S.*)$")
BULLET_ONLY = re.compile(r"^\s*[•▪◦●■*]\s*$")
FORM_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .\/#&-]{0,28}?)\s*:\s*(.*)$")


def _runs(profile: np.ndarray, thr: float):
    idx = np.where(profile > thr)[0]
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    return [int(g.mean()) for g in groups]


def ink_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m


def merge_row_fragments(lines, gap=1.0):
    """Large type sometimes OCRs as two lines on one row (a dropped glyph splits the word). Re-join lines that sit
    on the same baseline row, have similar height and are closer than `gap` x their height."""
    from .ocr import make_line
    out = []
    for l in sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0])):
        for k, p in enumerate(out):
            h = max(l.height, p.height)
            cy_l, cy_p = (l.bbox[1] + l.bbox[3]) / 2, (p.bbox[1] + p.bbox[3]) / 2
            # a label ending in ':' owns the value on its row however wide the gap (form layouts)
            reach = (12.0 if p.text.rstrip().endswith(":") else gap) * h
            if abs(cy_l - cy_p) < 0.4 * h and 0.75 <= l.height / max(1, p.height) <= 1.33 \
                    and 0 <= l.bbox[0] - p.bbox[2] < reach:
                out[k] = make_line(p.words + l.words)
                break
        else:
            out.append(l)
    return sorted(out, key=lambda l: (l.bbox[1], l.bbox[0]))


def mark_bold(gray, lines, ratio=1.09):
    """Pixel path has no font flags: estimate stroke width per line (distance transform on the ink mask)
    and flag lines whose strokes are clearly thicker than body text as bold (run-in section labels)."""
    if len(lines) < 4:
        return lines
    ink = (ink_mask(gray) > 0)
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    sw = []
    for l in lines:
        x0, y0, x1, y1 = l.bbox
        reg, d = ink[y0:y1, x0:x1], dt[y0:y1, x0:x1]
        sw.append(2 * float(d[reg].mean()) if reg.any() else 0.0)
    body = [v for v, l in zip(sw, lines) if len(l.text) >= 25 and v > 0]
    if len(body) < 3:
        return lines
    ref = float(np.median(body))
    for v, l in zip(sw, lines):
        if v >= ratio * ref and len(l.text) > 2:
            for w in l.words:
                w.bold = True
    return lines


def rule_masks(gray):
    H, W = gray.shape
    ink = ink_mask(gray)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 30), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 60)))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk), cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)


def containment(a, b) -> float:
    """Fraction of box a that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return (ix * iy) / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def detect_tables(hor, ver):
    """Ruled tables: a connected line-grid with >=3 horizontal and >=3 vertical rules."""
    grid = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grid)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 150 or h < 60:
            continue
        ys = _runs(hor[y:y + h, x:x + w].sum(1) / 255.0, 0.6 * w)
        xs = _runs(ver[y:y + h, x:x + w].sum(0) / 255.0, 0.6 * h)
        if len(ys) >= 3 and len(xs) >= 3:
            out.append(((xs[0] + x, ys[0] + y, xs[-1] + x, ys[-1] + y),
                        [v + y for v in ys], [v + x for v in xs]))
    return sorted(out, key=lambda t: t[0][1])


def detect_figures(rgb, exclude, min_w=110, min_h=80):
    """Pictorial regions: large, dense (or colourful) areas that are not text strokes."""
    den = cv2.medianBlur(rgb, 5)            # suppress sensor noise before measuring colour/darkness
    gray = cv2.cvtColor(den, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(den, cv2.COLOR_RGB2HSV)[..., 1]
    ink = ((gray < 246) | (sat > 40)).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    for (x0, y0, x1, y1) in exclude:
        ink[max(0, y0 - 4):y1 + 5, max(0, x0 - 4):x1 + 5] = 0
    core = ((gray < 200) | (sat > 40)).astype(np.uint8)
    for (x0, y0, x1, y1) in exclude:
        core[max(0, y0 - 4):y1 + 5, max(0, x0 - 4):x1 + 5] = 0
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < min_w or h < min_h:
            continue
        dens = core[y:y + h, x:x + w].mean()   # dark-core density: blur halos don't inflate it
        satf = (sat[y:y + h, x:x + w] > 60).mean()
        if dens >= 0.35 or satf >= 0.08:
            out.append(((int(x), int(y), int(x + w), int(y + h)), float(min(1.0, max(dens, satf * 3)))))
    return sorted(out, key=lambda t: t[0][1])


def underlines(hor, table_boxes, figure_boxes):
    """Horizontal rules that are not table borders: candidate form-field underlines."""
    m = hor.copy()
    for (x0, y0, x1, y1) in list(table_boxes) + list(figure_boxes):
        m[max(0, y0 - 4):y1 + 5, max(0, x0 - 4):x1 + 5] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats[1:] if w >= 60 and h <= 8], m


# ---------------------------------------------------------------- text classification
def _char_w(l: Line) -> float:
    return (l.bbox[2] - l.bbox[0]) / max(1, len(l.text))


def merge_bullets(lines):
    """Some PDFs emit the bullet glyph as its own line above/before the item text: fold it into the text line."""
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if BULLET_ONLY.match(l.text) and i + 1 < len(lines):
            n = lines[i + 1]
            if n.bbox[1] - l.bbox[3] < 2.5 * max(1, l.height) and abs(n.bbox[0] - l.bbox[0]) < 120:
                bw = l.words[0].text
                out.append(Line(words=list(n.words), bbox=(min(n.bbox[0], l.bbox[0]), n.bbox[1], n.bbox[2], n.bbox[3]),
                                text=f"{bw} {n.text}", height=n.height))
                i += 2; continue
        out.append(l); i += 1
    return out


def classify_lines(lines, ul_boxes):
    """Return list of (line, kind, extra) with kind in title/heading/text/form/list."""
    if not lines:
        return []
    lines = merge_bullets(lines)
    body_h = float(np.median([l.bbox[3] - l.bbox[1] for l in lines]))
    long_lines = [l for l in lines if len(l.text) >= 25] or lines
    body_cw = float(np.median([_char_w(l) for l in long_lines]))
    bold_frac = sum(l.bold for l in lines) / len(lines)
    res = []
    for l in lines:
        ul = next((u for u in ul_boxes
                   if l.bbox[3] - 8 <= u[1] <= l.bbox[3] + 26 and u[0] < l.bbox[2] and u[2] > l.bbox[0]), None)
        lm = BULLET_RE.match(l.text)
        if lm and len(l.text) < 400 and not FORM_RE.match(l.text):
            res.append((l, "list", lm.group(1).strip()))
            continue
        m = FORM_RE.match(l.text)
        if m and (ul is not None or len(l.text) < 40):
            res.append((l, "form", (m.group(1).strip(), m.group(2).strip(), ul)))
            continue
        hr, cr = (l.bbox[3] - l.bbox[1]) / body_h, _char_w(l) / body_cw
        # height is the stable title signal; char-width alone is noisy on short words
        if len(l.text) <= 90 and (hr >= 1.75 or (cr >= 2.2 and hr >= 1.35)):
            res.append((l, "title", None))
        elif len(l.text) <= 90 and max(hr, cr) >= 1.3:
            res.append((l, "heading", None))
        elif len(l.text) <= 90 and l.bold and bold_frac < 0.75 and len(l.text) > 2:
            res.append((l, "heading", None))      # same-size bold line = run-in section label
        else:
            res.append((l, "text", None))
    return res


def build_elements(classified, page=0, source="classical"):
    els, cur, cur_kind = [], [], None
    body_lh = float(np.median([l.bbox[3] - l.bbox[1] for l, _, _ in classified])) if classified else 20

    def flush():
        nonlocal cur, cur_kind
        if not cur:
            return
        ls = cur
        t = {"title": "title", "heading": "heading", "text": "paragraph"}[cur_kind]
        els.append(Element(type=t, bbox=union([l.bbox for l in ls]), page=page,
                           text=" ".join(l.text for l in ls),
                           conf=float(np.mean([l.conf for l in ls])), source=source,
                           level={"title": 1, "heading": 2}.get(cur_kind)))
        cur, cur_kind = [], None

    for l, kind, extra in classified:
        if kind == "list":
            flush()
            els.append(Element(type="list_item", bbox=l.bbox, page=page, text=extra, conf=l.conf, source=source))
            continue
        if kind == "form":
            flush()
            key, val, ul = extra
            bb = union([l.bbox] + ([ul] if ul else []))
            els.append(Element(type="form_field", bbox=bb, page=page, text=l.text, key=key, value=val,
                               conf=l.conf * (1.0 if ul else 0.7), source=source))
            continue
        if cur and (kind != cur_kind or l.bbox[1] - cur[-1].bbox[3] > 0.9 * body_lh
                    or l.bbox[0] > cur[-1].bbox[2] or l.bbox[2] < cur[-1].bbox[0]):
            flush()
        cur.append(l); cur_kind = kind
    flush()
    return els


# ---------------------------------------------------------------- reading order (XY-cut)
def xy_cut(items, min_gap_y=8, min_gap_x=30):
    """items: list of Element. Returns elements in reading order (handles multi-column)."""
    if len(items) <= 1:
        return list(items)

    def split(its, axis, gap):
        lo, hi = (1, 3) if axis == "y" else (0, 2)
        its = sorted(its, key=lambda e: e.bbox[lo])
        groups, cur, reach = [], [its[0]], its[0].bbox[hi]
        for e in its[1:]:
            if e.bbox[lo] - reach > gap:
                groups.append(cur); cur = []
            cur.append(e); reach = max(reach, e.bbox[hi])
        groups.append(cur)
        return groups

    # a column gutter spanning every item wins over row gaps (read down each column first)
    g = split(items, "x", min_gap_x)
    if len(g) > 1:
        return [e for grp in g for e in xy_cut(grp, min_gap_y, min_gap_x)]
    g = split(items, "y", min_gap_y)
    if len(g) > 1:
        return [e for grp in g for e in xy_cut(grp, min_gap_y, min_gap_x)]
    return sorted(items, key=lambda e: (e.bbox[1], e.bbox[0]))
