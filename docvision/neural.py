"""OPTIONAL neural layout backend (YOLO11 trained on DocLayNet).

NOT exercised end-to-end in the build sandbox (no model download there). The plumbing
(`regions_to_elements`) is unit-tested with a stub detector; verify real-model accuracy on your machine:

    pip install ultralytics huggingface_hub
    python -m docvision.neural path/to/page.png

Model: Armaggheddon/yolo11-document-layout (files: yolo11n/s/m_doc_layout.pt, DocLayNet, MIT).
"""
from __future__ import annotations
from typing import Callable, List, Tuple
from .schema import Element, Line, union
from . import layout as L

# DocLayNet label -> DocVision class (None = ignore: headers/footers/formulas are not in our class set)
LABEL_MAP = {"title": "title", "section-header": "heading", "text": "paragraph", "list-item": "paragraph",
             "caption": "paragraph", "footnote": "paragraph", "table": "table", "picture": "figure",
             "page-header": None, "page-footer": None, "formula": None}

Detector = Callable[[object], List[Tuple[str, Tuple[int, int, int, int], float]]]


def yolo_detector(size: str = "s", conf: float = 0.25) -> Detector:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    path = hf_hub_download("Armaggheddon/yolo11-document-layout", f"yolo11{size}_doc_layout.pt")
    model = YOLO(path)

    def detect(rgb):
        r = model.predict(rgb, conf=conf, verbose=False)[0]
        out = []
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            name = str(r.names[int(k)]).lower().replace("_", "-").replace(" ", "-")
            t = LABEL_MAP.get(name)
            if t:
                out.append((t, tuple(int(v) for v in b), float(c)))
        return out
    return detect


def regions_to_elements(regions, lines: List[Line], page=0) -> List[Element]:
    """Assign OCR lines to detected regions (text regions get their lines' text); keep table/figure boxes.
    Lines claimed by no region are kept as paragraphs so no text is silently dropped."""
    els, used = [], set()
    for t, bb, cf in sorted(regions, key=lambda r: (r[1][1], r[1][0])):
        if t in ("table", "figure"):
            els.append(Element(type=t, bbox=bb, page=page, conf=cf, source="neural"))
            for i, l in enumerate(lines):
                if L.containment(l.bbox, bb) > 0.6:
                    used.add(i)
            continue
        inside = [(i, l) for i, l in enumerate(lines) if L.containment(l.bbox, bb) > 0.6]
        if not inside:
            continue
        used.update(i for i, _ in inside)
        ls = [l for _, l in sorted(inside, key=lambda p: (p[1].bbox[1], p[1].bbox[0]))]
        els.append(Element(type=t, bbox=union([l.bbox for l in ls]), page=page, text=" ".join(l.text for l in ls),
                           conf=cf, source="neural", level={"title": 1, "heading": 2}.get(t)))
    leftovers = [(l, "text", None) for i, l in enumerate(lines) if i not in used]
    els += L.build_elements(leftovers, page=page, source="neural-fallback")
    return els


if __name__ == "__main__":
    import sys
    from PIL import Image
    import numpy as np
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    for t, bb, c in yolo_detector()(img):
        print(f"{t:10s} {c:.2f} {bb}")
