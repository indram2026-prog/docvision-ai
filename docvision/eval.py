"""Evaluation harness: structured pipeline vs OCR-only baseline on the synthetic benchmark.
Usage: python -m docvision.eval --n 12 --n-deg 6 --out results"""
from __future__ import annotations
import argparse, json, os, time
from collections import defaultdict
import numpy as np
from .synth import make_page, degrade
from .pipeline import process_image, baseline_ocr_only
from .metrics import match, prf, cer, norm, reading_order_score
from .schema import CLASSES
from . import viz

TEXT_TYPES = ("title", "heading", "paragraph", "form_field")


def pred_items(page):
    return [(e.type, e.bbox) for e in page.elements]


def page_text(page):
    parts = []
    for e in sorted(page.elements, key=lambda e: e.order):
        if e.type == "table" and e.cells:
            parts.append(" ".join(c for r in e.cells for c in r))
        elif e.type in TEXT_TYPES:
            parts.append(e.text)
    return " ".join(parts)


def gt_text(gt):
    parts, tables = [], list(gt["tables"])
    for e in gt["elements"]:
        if e["type"] == "table":
            if tables:                      # real datasets (DocLayNet) carry no cell text
                parts.append(" ".join(c for r in tables.pop(0)["cells"] for c in r))
        elif e["type"] in TEXT_TYPES:
            parts.append(e["text"])
    return " ".join(parts)


def rotate_gt(gt, angle, W=1240, H=1754):
    """Transform GT boxes the same way PIL.Image.rotate(angle) moved the pixels (axis-aligned hull)."""
    import math
    th = math.radians(angle); c, s_ = math.cos(th), math.sin(th); cx, cy = W / 2, H / 2
    def f(b):
        pts = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
        q = [(cx + (x - cx) * c + (y - cy) * s_, cy - (x - cx) * s_ + (y - cy) * c) for x, y in pts]
        return [int(min(p[0] for p in q)), int(min(p[1] for p in q)), int(max(p[0] for p in q)), int(max(p[1] for p in q))]
    out = dict(gt); out["elements"] = [{**e, "bbox": f(e["bbox"])} for e in gt["elements"]]
    return out


def score_page(page, gt):
    G = [(e["type"], tuple(e["bbox"])) for e in gt["elements"]]
    P = pred_items(page)
    pairs = match(P, G)
    per = {}
    for c in CLASSES:
        tp = sum(1 for pi, gi, _ in pairs if G[gi][0] == c)
        fp = sum(1 for p in P if p[0] == c) - tp
        fn = sum(1 for g in G if g[0] == c) - tp
        per[c] = (tp, fp, fn)
    ious = [v for _, _, v in pairs]
    # text CER on matched text elements
    cers = []
    for pi, gi, _ in pairs:
        if G[gi][0] in TEXT_TYPES:
            cers.append(cer(page.elements[pi].text, gt["elements"][gi]["text"]))
    # tables
    tstats = {"gt": len(gt["tables"]), "matched": 0, "shape_ok": 0, "cells": 0, "cells_ok": 0}
    gt_tables = [e for e in gt["elements"] if e["type"] == "table"]
    for ti, t in enumerate(gt["tables"]):
        tstats["cells"] += sum(len(r) for r in t["cells"])
        gi = [i for i, e in enumerate(gt["elements"]) if e["type"] == "table"][ti]
        m = [pi for pi, g, _ in pairs if g == gi]
        if not m:
            continue
        tstats["matched"] += 1
        pc = page.elements[m[0]].cells or []
        if len(pc) == len(t["cells"]) and all(len(a) == len(b) for a, b in zip(pc, t["cells"])):
            tstats["shape_ok"] += 1
            tstats["cells_ok"] += sum(norm(a) == norm(b) for ra, rb in zip(pc, t["cells"]) for a, b in zip(ra, rb))
    # forms
    gp = {(norm(f["key"]), norm(f["value"])) for f in gt["forms"]}
    pp = {(norm(e.key), norm(e.value)) for e in page.elements if e.type == "form_field" and e.key}
    ftp = len(gp & pp)
    order = reading_order_score(pairs, [e.order for e in page.elements],
                                list(range(len(G))))
    return {"per": per, "ious": ious, "cers": cers, "tables": tstats, "form": (ftp, len(pp - gp), len(gp - pp)),
            "order": order, "page_cer": cer(page_text(page), gt_text(gt)),
            "errors": [{"type": G[gi][0], "bbox": G[gi][1], "kind": "FN"} for gi in range(len(G))
                       if gi not in {g for _, g, _ in pairs}] +
                      [{"type": P[pi][0], "bbox": P[pi][1], "kind": "FP"} for pi in range(len(P))
                       if pi not in {p for p, _, _ in pairs}]}


def aggregate(scores):
    out = {"classes": {}}
    f1s = []
    for c in CLASSES:
        tp = sum(s["per"][c][0] for s in scores); fp = sum(s["per"][c][1] for s in scores)
        fn = sum(s["per"][c][2] for s in scores)
        p, r, f = prf(tp, fp, fn)
        if tp + fn:
            f1s.append(f)
        out["classes"][c] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": tp + fn}
    out["macro_f1"] = round(float(np.mean(f1s)), 3) if f1s else 0.0
    ious = [v for s in scores for v in s["ious"]]
    out["mean_iou"] = round(float(np.mean(ious)), 3) if ious else 0.0
    cers = [v for s in scores for v in s["cers"]]
    out["element_cer"] = round(float(np.mean(cers)), 4) if cers else None
    out["page_cer"] = round(float(np.mean([s["page_cer"] for s in scores])), 4)
    t = defaultdict(int)
    for s in scores:
        for k, v in s["tables"].items():
            t[k] += v
    out["table"] = {"detected_rate": round(t["matched"] / t["gt"], 3) if t["gt"] else None,
                    "shape_exact_rate": round(t["shape_ok"] / t["gt"], 3) if t["gt"] else None,
                    "cell_accuracy": round(t["cells_ok"] / t["cells"], 3) if t["cells"] else None}
    tp = sum(s["form"][0] for s in scores); fp = sum(s["form"][1] for s in scores); fn = sum(s["form"][2] for s in scores)
    p, r, f = prf(tp, fp, fn)
    out["form_kv"] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    out["reading_order"] = round(float(np.mean([s["order"] for s in scores])), 3)
    return out


def run(n, n_deg, out_dir, seed0=1000):
    os.makedirs(out_dir, exist_ok=True)
    conds = {"clean": n, "noise": n_deg, "blur": n_deg, "lowcontrast": n_deg, "skew": n_deg, "borderless": n_deg}
    res, failures, timing = {}, [], defaultdict(list)
    for cond, cnt in conds.items():
        S, B = [], []
        for i in range(cnt):
            bl = cond == "borderless"
            img, gt = make_page(seed0 + i, first=(i % 2 == 0), force_table=(i % 3 == 0 or bl), borderless=bl)
            im = degrade(img, "clean" if bl else cond, seed=i)
            t0 = time.time(); pg = process_image(im); timing["structured"].append(time.time() - t0)
            t0 = time.time(); bp = baseline_ocr_only(im); timing["ocr_only"].append(time.time() - t0)
            if cond == "skew":
                gt = rotate_gt(gt, 1.8)
            s, b = score_page(pg, gt), score_page(bp, gt)
            S.append(s); B.append(b)
            if i == 0:
                viz.draw(im, pg).save(os.path.join(out_dir, f"overlay_{cond}.png"))
            for er in s["errors"]:
                failures.append({"condition": cond, "seed": seed0 + i, **er})
        res[cond] = {"structured": aggregate(S), "ocr_only": aggregate(B), "pages": cnt}
        print(cond, "structured macroF1", res[cond]["structured"]["macro_f1"], "| ocr-only", res[cond]["ocr_only"]["macro_f1"], flush=True)
    res["timing_sec_per_page"] = {k: round(float(np.mean(v)), 2) for k, v in timing.items()}
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    json.dump(failures, open(os.path.join(out_dir, "failures.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-deg", type=int, default=6)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run(a.n, a.n_deg, a.out)
