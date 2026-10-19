"""Synthetic benchmark: pages with exact ground truth for layout, tables, forms and OCR.
Documented synthetic data (permitted by the challenge brief); fully reproducible via seed."""
from __future__ import annotations
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

W, H, M = 1240, 1754, 110  # A4 @150dpi
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WORDS = ("system model data signal energy report quality research network design process method result "
         "analysis planning review budget service market growth policy health safety traffic climate "
         "student record future measure sample output input value level region survey public sector "
         "project update total annual income support request change status order common simple").split()
LABELS = ["Name", "Address", "Date", "Phone", "Email", "Company", "Position", "Department", "City", "Reference"]
HEAD = ["Overview", "Background", "Methodology", "Findings", "Discussion", "Summary", "Scope", "Results"]


def _f(path, size):
    return ImageFont.truetype(path, size)


def _words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _wrap(draw, text, font, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def make_page(seed: int, first: bool = True, force_table: bool = False, borderless: bool = False):
    """Return (PIL image, ground-truth dict)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    gt = {"elements": [], "tables": [], "forms": []}
    y = M
    body, hf, tf = _f(FONT, 24), _f(BOLD, 34), _f(BOLD, 54)

    def add(t, box, text=""):
        gt["elements"].append({"type": t, "bbox": [int(v) for v in box], "text": text})

    if first:
        t = _words(rng, 3).title()
        tw = d.textlength(t, font=tf)
        x = M
        d.text((x, y), t, font=tf, fill=(10, 10, 10))
        add("title", (x, y + 10, x + tw, y + 54 + 10), t)
        y += 120
    kinds = ["heading", "paragraph", "table", "figure", "form", "paragraph", "heading", "paragraph"]
    rng.shuffle(kinds)
    if force_table and "table" not in kinds[:5]:
        kinds.insert(1, "table")
    for k in kinds:
        if y > H - 380:
            break
        if k == "heading":
            t = rng.choice(HEAD)
            d.text((M, y), t, font=hf, fill=(10, 10, 10))
            tw = d.textlength(t, font=hf)
            bb = d.textbbox((M, y), t, font=hf)
            add("heading", bb, t); y += 85
        elif k == "paragraph":
            txt = _words(rng, rng.randint(28, 60))
            ls = _wrap(d, txt, body, W - 2 * M)
            top = y
            for ln in ls:
                d.text((M, y), ln, font=body, fill=(20, 20, 20)); y += 36
            ws = [d.textbbox((M, top + 36 * i), ln, font=body) for i, ln in enumerate(ls)]
            add("paragraph", (M, min(b[1] for b in ws), max(b[2] for b in ws), max(b[3] for b in ws)), " ".join(ls))
            y += 36
        elif k == "table":
            nr, nc = rng.randint(3, 6), rng.randint(3, 4)
            cw = (W - 2 * M) // nc; rh = 52
            x0, y0 = M, y
            cells = []
            for r in range(nr):
                row = []
                for c in range(nc):
                    s = rng.choice(WORDS).title() if r == 0 else (str(rng.randint(10, 999)) if c > 0 else rng.choice(WORDS))
                    row.append(s)
                    f = hf if False else (_f(BOLD, 24) if r == 0 else body)
                    d.text((x0 + c * cw + 14, y0 + r * rh + 12), s, font=f, fill=(15, 15, 15))
                cells.append(row)
            if borderless:      # no ruling lines: columns are only implied by alignment
                tb = [d.textbbox((x0 + c * cw + 14, y0 + r * rh + 12), cells[r][c],
                                 font=(_f(BOLD, 24) if r == 0 else body)) for r in range(nr) for c in range(nc)]
                bb = (min(b[0] for b in tb), min(b[1] for b in tb), max(b[2] for b in tb), max(b[3] for b in tb))
                add("table", bb)
                gt["tables"].append({"bbox": list(bb), "cells": cells})
            else:
                for r in range(nr + 1):
                    d.line((x0, y0 + r * rh, x0 + nc * cw, y0 + r * rh), fill=(0, 0, 0), width=2)
                for c in range(nc + 1):
                    d.line((x0 + c * cw, y0, x0 + c * cw, y0 + nr * rh), fill=(0, 0, 0), width=2)
                add("table", (x0, y0, x0 + nc * cw, y0 + nr * rh))
                gt["tables"].append({"bbox": [x0, y0, x0 + nc * cw, y0 + nr * rh], "cells": cells})
            y += nr * rh + 60
        elif k == "figure":
            fw, fh = rng.randint(420, 760), rng.randint(220, 340)
            fig = Image.new("RGB", (fw, fh), (245, 245, 245))
            fd = ImageDraw.Draw(fig)
            if rng.random() < 0.5:   # bar chart
                n = rng.randint(4, 7); bw = fw // (n * 2)
                cols = [(rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220)) for _ in range(n)]
                for i in range(n):
                    bh = rng.randint(40, fh - 30)
                    fd.rectangle((20 + i * 2 * bw, fh - 10 - bh, 20 + i * 2 * bw + bw, fh - 10), fill=cols[i])
            else:                    # photo-like blob field
                arr = np.random.RandomState(seed + y).rand(fh // 20 + 1, fw // 20 + 1, 3) * 255
                fig = Image.fromarray(arr.astype("uint8")).resize((fw, fh), Image.BICUBIC)
            x0 = M + rng.randint(0, 100)
            img.paste(fig, (x0, y))
            add("figure", (x0, y, x0 + fw, y + fh))
            y += fh + 60
        elif k == "form":
            n = rng.randint(3, 5)
            for lab in rng.sample(LABELS, n):
                val = rng.choice(WORDS).title() + " " + str(rng.randint(100, 9999))
                lt = lab + ":"
                d.text((M, y), lt, font=body, fill=(20, 20, 20))
                d.text((M + 270, y), val, font=body, fill=(20, 20, 20))
                x1 = M + 720
                d.line((M + 260, y + 38, x1, y + 38), fill=(0, 0, 0), width=2)
                bb = d.textbbox((M, y), lt, font=body)
                add("form_field", (M, bb[1], x1, y + 40), f"{lab}: {val}")
                gt["forms"].append({"key": lab, "value": val})
                y += 62
            y += 30
    return img, gt


def degrade(img: Image.Image, kind: str, seed: int = 0) -> Image.Image:
    """Controlled degradations for robustness testing."""
    rng = np.random.RandomState(seed)
    if kind == "clean":
        return img
    if kind == "noise":
        a = np.asarray(img).astype(np.float32) + rng.normal(0, 18, (img.height, img.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.3))
    if kind == "lowcontrast":
        a = np.asarray(img).astype(np.float32) * 0.45 + 120
        return Image.fromarray(a.clip(0, 255).astype("uint8"))
    if kind == "skew":
        return img.rotate(1.8, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    raise ValueError(kind)
