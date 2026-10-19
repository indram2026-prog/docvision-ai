"""Loaders for the real benchmark datasets named in the brief (FUNSD, DocLayNet). Pure Python, no network.

FUNSD  (https://guillaumejaume.github.io/FUNSD/):  <root>/{training_data,testing_data}/{images,annotations}
DocLayNet core (COCO json + PNGs):                 --ann <root>/COCO/test.json  --img-dir <root>/PNG
"""
from __future__ import annotations
import json, random
from pathlib import Path
from .metrics import norm

# DocLayNet category -> DocVision class; None = not a DocVision class (regions are ignored, not penalised)
DOCLAYNET_MAP = {"title": "title", "section-header": "heading", "text": "paragraph", "list-item": "paragraph",
                 "caption": "paragraph", "footnote": "paragraph", "table": "table", "picture": "figure",
                 "page-header": None, "page-footer": None, "formula": None}


def _slug(s):
    return str(s).lower().replace("_", "-").replace(" ", "-")


# ------------------------------------------------------------------ FUNSD
def funsd_dir(root, split):
    root = Path(root)
    for c in (root / split, root / "dataset" / split, root):
        if (c / "annotations").is_dir() and (c / "images").is_dir():
            return c
    raise FileNotFoundError(f"FUNSD split '{split}' not found under {root} (expected annotations/ and images/)")


def load_funsd(root, split="testing_data", n=None, seed=0):
    base = funsd_dir(root, split)
    files = sorted((base / "annotations").glob("*.json"))
    if n and n < len(files):
        files = random.Random(seed).sample(files, n)
    for ann in files:
        img = next((base / "images" / (ann.stem + e) for e in (".png", ".jpg", ".jpeg") if (base / "images" / (ann.stem + e)).exists()), None)
        if img is not None:
            yield img, json.load(open(ann))["form"]


def _key(s):
    return norm(s).rstrip(": ").strip()


def funsd_pairs(form):
    """Ground-truth key/value pairs = linked (question, answer) entities."""
    by_id = {e["id"]: e for e in form}
    out = set()
    for e in form:
        if e.get("label") != "question":
            continue
        for a, b in e.get("linking", []):
            o = by_id.get(b if a == e["id"] else a)
            if o is not None and o.get("label") == "answer" and o.get("text", "").strip():
                out.add((_key(e["text"]), norm(o["text"])))
    return sorted(out)


# ------------------------------------------------------------------ DocLayNet (COCO)
def load_doclaynet(ann_path, img_dir, n=None, seed=0):
    """Yields (image_path, [(docvision_class_or_None, (x0,y0,x1,y1)), ...])."""
    coco = json.load(open(ann_path))
    cats = {c["id"]: _slug(c["name"]) for c in coco["categories"]}
    per = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        per.setdefault(a["image_id"], []).append((DOCLAYNET_MAP.get(cats[a["category_id"]]), (x, y, x + w, y + h)))
    imgs = sorted(coco["images"], key=lambda i: i["id"])
    if n and n < len(imgs):
        imgs = random.Random(seed).sample(imgs, n)
    for im in imgs:
        p = Path(img_dir) / im["file_name"]
        if p.exists():
            yield p, per.get(im["id"], [])
