"""Run DocVision on the real datasets named in the brief and write numbers + failure cases.

  FUNSD (key-value extraction; arms: rules, layoutlmv3):
      python -m docvision.benchmark_real funsd --root /data/FUNSD --split testing_data --n 50 --arms rules,layoutlmv3
  DocLayNet (layout detection; arms: classical, ocr_only, yolo):
      python -m docvision.benchmark_real doclaynet --ann /data/DocLayNet/COCO/test.json --img-dir /data/DocLayNet/PNG --n 100 --arms classical,ocr_only,yolo

Outputs results/real_<dataset>.json (aggregate metrics) and results/real_<dataset>_failures.json (per-document errors).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from PIL import Image
from .metrics import cer, norm, prf
from .schema import iou, Page, Element, CLASSES

# ----------------------------------------------------------------------------- key-value scoring
def kv_match(pred, gt, tol=0.25):
    """Greedy one-to-one match of (key, value) pairs; a pair matches if both key and value are within `tol` CER."""
    cands = []
    for pi, (pk, pv) in enumerate(pred):
        for gi, (gk, gv) in enumerate(gt):
            ck, cv = cer(pk, gk), cer(pv, gv)
            if ck <= tol and cv <= tol:
                cands.append((ck + cv, pi, gi))
    up, ug, out = set(), set(), []
    for s, pi, gi in sorted(cands):
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); out.append((pi, gi))
    return out


def score_kv(pred, gt, tol=0.25):
    m = kv_match(pred, gt, tol)
    tp = len(m)
    return tp, len(pred) - tp, len(gt) - tp, m


def kv_from_page(page):
    return [(norm(e.key).rstrip(": "), norm(e.value)) for e in page.elements
            if e.type == "form_field" and e.key and (e.value or "").strip()]


def run_funsd(root, split, n, arms, out, seed=0):
    from .datasets import load_funsd, funsd_pairs
    from .pipeline import process_image
    from .ocr import tesseract_lines
    lm = None
    if "layoutlmv3" in arms:
        from .neural_forms import LayoutLMv3KV
        lm = LayoutLMv3KV()
    tot = {a: [0, 0, 0] for a in arms}; time_ = {a: 0.0 for a in arms}
    fails, docs = [], 0
    for img_path, form in load_funsd(root, split, n, seed):
        pil = Image.open(img_path).convert("RGB")
        gt = [(k, v) for k, v in funsd_pairs(form)]
        docs += 1
        arr = np.asarray(pil)
        for a in arms:
            t0 = time.time()
            if a == "rules":
                pred = kv_from_page(process_image(pil))
            elif a == "layoutlmv3":
                words = [w for l in tesseract_lines(arr) for w in l.words]
                pairs, _ = lm.extract_pairs(pil, words)
                pred = [(norm(k).rstrip(": "), norm(v)) for k, v, *_ in pairs]
            else:
                raise SystemExit(f"unknown arm {a}")
            time_[a] += time.time() - t0
            tp, fp, fn, m = score_kv(pred, gt)
            for i, v in enumerate((tp, fp, fn)):
                tot[a][i] += v
            matched_p = {p for p, _ in m}; matched_g = {g for _, g in m}
            fails.append({"doc": os.path.basename(str(img_path)), "arm": a,
                          "missed": [list(gt[g]) for g in range(len(gt)) if g not in matched_g],
                          "spurious": [list(pred[p]) for p in range(len(pred)) if p not in matched_p]})
        print(docs, {a: round(prf(*tot[a])[2], 3) for a in arms}, flush=True)
    res = {"dataset": "FUNSD", "split": split, "docs": docs, "metric": "key-value pair F1 (CER<=0.25 on key and value)",
           "arms": {a: dict(zip(("precision", "recall", "f1"), [round(x, 3) for x in prf(*tot[a])]),
                            tp=tot[a][0], fp=tot[a][1], fn=tot[a][2], sec_per_doc=round(time_[a] / max(1, docs), 2))
                    for a in arms}}
    return _write(out, "funsd", res, fails)


# ----------------------------------------------------------------------------- layout scoring
def _to_page(items, shape):
    els = [Element(type=t, bbox=tuple(int(v) for v in b), conf=c) for t, b, c in items]
    return Page(number=0, width=shape[1], height=shape[0], elements=els)


def run_doclaynet(ann, img_dir, n, arms, out, seed=0, width=1240):
    from .datasets import load_doclaynet
    from .pipeline import process_image, baseline_ocr_only
    from .eval import score_page, aggregate
    det = None
    if "yolo" in arms:
        from .neural import yolo_detector
        det = yolo_detector()
    scores = {a: [] for a in arms}; time_ = {a: 0.0 for a in arms}; fails = []
    docs = 0
    for img_path, gts in load_doclaynet(ann, img_dir, n, seed):
        pil = Image.open(img_path).convert("RGB")
        f = width / pil.width                       # work at the ~150 dpi A4 scale the classical rules assume
        pil = pil.resize((width, int(pil.height * f)), Image.LANCZOS)
        G = [(t, tuple(v * f for v in b)) for t, b in gts if t]
        ignored = [tuple(v * f for v in b) for t, b in gts if t is None]
        gt = {"elements": [{"type": t, "bbox": [int(v) for v in b], "text": ""} for t, b in G], "tables": [], "forms": []}
        docs += 1
        arr = np.asarray(pil)
        for a in arms:
            t0 = time.time()
            if a == "classical":
                page = process_image(pil)
            elif a == "ocr_only":
                page = baseline_ocr_only(pil)
            elif a == "yolo":
                page = _to_page([(t, b, c) for t, b, c in det(arr)], arr.shape)
            else:
                raise SystemExit(f"unknown arm {a}")
            time_[a] += time.time() - t0
            for e in page.elements:                 # list_item is a paragraph in DocLayNet's taxonomy
                if e.type in ("list_item", "code"):
                    e.type = "paragraph"
            # regions the benchmark does not score (page header/footer, formula) must not count as false positives
            page.elements = [e for e in page.elements if not any(iou(e.bbox, g) >= 0.5 for g in ignored)]
            s = score_page(page, gt)
            scores[a].append(s)
            for er in s["errors"]:
                fails.append({"doc": os.path.basename(str(img_path)), "arm": a, **er})
        print(docs, {a: aggregate(scores[a])["macro_f1"] for a in arms}, flush=True)
    res = {"dataset": "DocLayNet", "docs": docs, "metric": "per-class P/R/F1 at IoU>=0.5; macro over classes present",
           "arms": {a: {**aggregate(scores[a]), "sec_per_doc": round(time_[a] / max(1, docs), 2)} for a in arms}}
    for a in res["arms"]:                           # page_cer / reading order are undefined without text GT
        for k in ("page_cer", "element_cer", "reading_order", "table", "form_kv"):
            res["arms"][a].pop(k, None)
    return _write(out, "doclaynet", res, fails)


def _write(out, name, res, fails):
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"real_{name}.json"), "w"), indent=2)
    json.dump(fails, open(os.path.join(out, f"real_{name}_failures.json"), "w"), indent=1)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("funsd"); f.add_argument("--root", required=True); f.add_argument("--split", default="testing_data")
    d = sub.add_parser("doclaynet"); d.add_argument("--ann", required=True); d.add_argument("--img-dir", required=True)
    for p in (f, d):
        p.add_argument("--n", type=int, default=50); p.add_argument("--out", default="results"); p.add_argument("--seed", type=int, default=0)
    f.add_argument("--arms", default="rules,layoutlmv3"); d.add_argument("--arms", default="classical,ocr_only,yolo")
    a = ap.parse_args()
    if a.cmd == "funsd":
        run_funsd(a.root, a.split, a.n, a.arms.split(","), a.out, a.seed)
    else:
        run_doclaynet(a.ann, a.img_dir, a.n, a.arms.split(","), a.out, a.seed)
