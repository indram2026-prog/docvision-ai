"""Borderless table detection from word geometry (no ruling lines needed).

Idea: a table is a run of text rows whose cells line up in vertical "whitespace rivers".
  1. cluster words into rows by vertical centre;
  2. split each row into cell segments wherever the horizontal gap is much wider than a word space;
  3. rows with >=2 segments are tabular candidates; consecutive candidates form a group;
  4. in a group, project all segments onto the x-axis; x-ranges that (almost) no segment covers are rivers,
     and the covered runs between rivers are the columns;
  5. accept groups with >=3 rows and >=2 columns (>=3 columns when only 3 rows), then assign words to cells.
Rows that are single-segment but start at a column edge are folded into the previous row (wrapped cell text).
"Label: value" form rows are excluded so aligned forms are not mistaken for tables.
"""
from __future__ import annotations
import re
import numpy as np
from .schema import Word, Element, union

FORM_ROW = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 .\/#&-]{0,28}:\s*\S*")


def _rows(words, y_tol=0.55):
    ws = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0]))
    rows, cur = [], []
    for w in ws:
        cy = (w.bbox[1] + w.bbox[3]) / 2
        if cur:
            rc = np.mean([(a.bbox[1] + a.bbox[3]) / 2 for a in cur])
            h = max(max(a.bbox[3] - a.bbox[1] for a in cur), w.bbox[3] - w.bbox[1])
            if abs(cy - rc) > y_tol * h:
                rows.append(cur); cur = []
        cur.append(w)
    if cur:
        rows.append(cur)
    return [sorted(r, key=lambda w: w.bbox[0]) for r in rows]


def _segments(row, gap_factor):
    """Split a row into cell segments on wide horizontal gaps. Returns list of word lists."""
    h = float(np.median([w.bbox[3] - w.bbox[1] for w in row]))
    segs, cur = [], [row[0]]
    for a, b in zip(row, row[1:]):
        if b.bbox[0] - a.bbox[2] > gap_factor * h:
            segs.append(cur); cur = []
        cur.append(b)
    segs.append(cur)
    return segs


def _seg_box(s):
    return union([w.bbox for w in s])


def _rivers(group_segs, x0, x1, min_river, tol=0.12):
    """Column x-intervals for a group: covered runs separated by uncovered gaps >= min_river px."""
    n = len(group_segs)
    cov = np.zeros(x1 - x0 + 1)
    for segs in group_segs:
        for s in segs:
            b = _seg_box(s)
            cov[b[0] - x0:b[2] - x0 + 1] += 1
    free = cov <= tol * n
    cols, start, gap = [], None, 0
    for i, f in enumerate(free):
        if not f:
            if start is None:
                start = i
            gap = 0
            end = i
        elif start is not None:
            gap += 1
            if gap >= min_river:
                cols.append((start + x0, end + x0)); start = None
    if start is not None:
        cols.append((start + x0, end + x0))
    return cols


def detect_borderless_tables(words, exclude=(), gap_factor=1.6, min_rows=3, max_row_gap=2.4):
    """Returns list of dicts: bbox, cells (list rows x cols of str), cell_boxes, conf, words(set of ids used)."""
    words = [w for w in words if not any(
        b[0] <= (w.bbox[0] + w.bbox[2]) / 2 <= b[2] and b[1] <= (w.bbox[1] + w.bbox[3]) / 2 <= b[3] for b in exclude)]
    if len(words) < 6:
        return []
    rows = _rows(words)
    segs = [_segments(r, gap_factor) for r in rows]
    tops = [min(w.bbox[1] for w in r) for r in rows]
    bots = [max(w.bbox[3] for w in r) for r in rows]
    hts = [float(np.median([w.bbox[3] - w.bbox[1] for w in r])) for r in rows]

    def is_form(i):
        text = " ".join(w.text for w in rows[i])
        return bool(FORM_ROW.match(text)) and text.split(":")[0].count(" ") <= 3 and len(segs[i]) == 2 \
            and segs[i][0][-1].text.endswith(":")

    cand = [len(s) >= 2 and not is_form(i) for i, s in enumerate(segs)]
    out, i = [], 0
    while i < len(rows):
        if not cand[i]:
            i += 1; continue
        j = i
        while j + 1 < len(rows):
            nxt = j + 1
            gap = tops[nxt] - bots[j]
            if gap > max_row_gap * hts[j]:
                break
            if cand[nxt]:
                j = nxt; continue
            # tolerate one single-segment wrapped line if a candidate row follows right after
            if nxt + 1 < len(rows) and cand[nxt + 1] and len(segs[nxt]) == 1 \
                    and tops[nxt + 1] - bots[nxt] <= max_row_gap * hts[nxt]:
                j = nxt + 1; continue
            break
        grp = list(range(i, j + 1))
        i = j + 1
        crow = [k for k in grp if cand[k]]
        if len(crow) < min_rows:
            continue
        gx0 = min(w.bbox[0] for k in grp for w in rows[k]); gx1 = max(w.bbox[2] for k in grp for w in rows[k])
        h = float(np.median([hts[k] for k in grp]))
        cols = _rivers([segs[k] for k in crow], gx0, gx1, min_river=max(8, int(0.9 * h)))
        if len(cols) < 2 or (len(crow) == min_rows and len(cols) < 3):
            continue
        # consistency: most candidate rows must populate >= 60% of the columns, one segment per column
        fill = []
        for k in crow:
            hit = {ci for s in segs[k] for ci, (a, b) in enumerate(cols) if a - 2 <= (_seg_box(s)[0] + _seg_box(s)[2]) / 2 <= b + 2}
            fill.append(len(hit) / len(cols))
        if np.mean(fill) < 0.7:
            continue
        # prose guard: real paragraphs have long, wide segments; table cells are short
        seg_lens = [len(" ".join(w.text for w in s)) for k in crow for s in segs[k]]
        if np.median(seg_lens) > 45:
            continue
        # build cells; non-candidate rows (wrapped lines) are folded into the previous row's cells
        grid, boxes = [], []
        for k in grp:
            cells = [[] for _ in cols]
            for w in rows[k]:
                cx = (w.bbox[0] + w.bbox[2]) / 2
                ci = min(range(len(cols)), key=lambda c: 0 if cols[c][0] - 2 <= cx <= cols[c][1] + 2
                         else min(abs(cx - cols[c][0]), abs(cx - cols[c][1])))
                cells[ci].append(w)
            if not cand[k] and grid:
                for ci, ws in enumerate(cells):
                    if ws:
                        grid[-1][ci] = grid[-1][ci] + ws
                continue
            grid.append(cells)
        cell_txt = [[" ".join(w.text for w in c) for c in r] for r in grid]
        cell_box = []
        for r in grid:
            rb = union([w.bbox for c in r for w in c])
            cell_box.append([union([w.bbox for w in c]) if c else (cols[ci][0], rb[1], cols[ci][1], rb[3])
                             for ci, c in enumerate(r)])
        used = [w for r in grid for c in r for w in c]
        out.append({"bbox": union([w.bbox for w in used]), "cells": cell_txt, "cell_boxes": cell_box,
                    "conf": float(np.mean(fill)), "words": used})
    return out


def to_elements(tables, page=0, source="word-alignment"):
    return [Element(type="table", bbox=t["bbox"], page=page, cells=t["cells"], cell_boxes=t["cell_boxes"],
                    conf=t["conf"], text="\n".join(" | ".join(r) for r in t["cells"]), source=source)
            for t in tables]
