"""Typed output schema. Every element keeps geometry, text, confidence and provenance."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1 in page-pixel coordinates
CLASSES = ["title", "heading", "paragraph", "table", "figure", "form_field"]


def iou(a: BBox, b: BBox) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def union(boxes) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float = 0.0
    bold: bool = False   # only known for native PDF text


@dataclass
class Line:
    words: List[Word]
    bbox: BBox
    text: str
    height: int  # max word height: robust proxy for font size

    @property
    def conf(self) -> float:
        return sum(w.conf for w in self.words) / max(1, len(self.words))

    @property
    def bold(self) -> bool:
        return bool(self.words) and all(w.bold for w in self.words if w.text.strip("•"))


@dataclass
class Element:
    type: str
    bbox: BBox
    page: int = 0
    text: str = ""
    conf: float = 0.0
    source: str = "classical"
    cells: Optional[List[List[str]]] = None
    cell_boxes: Optional[List[List[BBox]]] = None
    key: Optional[str] = None
    value: Optional[str] = None
    level: Optional[int] = None
    order: int = 0
    continued: bool = False
    value_type: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Page:
    number: int
    width: int
    height: int
    elements: List[Element] = field(default_factory=list)
    skew_deg: float = 0.0
    ocr_source: str = "tesseract"


@dataclass
class Document:
    pages: List[Page]
    sections: list = field(default_factory=list)

    def to_dict(self):
        return {"pages": [{"number": p.number, "width": p.width, "height": p.height,
                           "skew_deg": round(p.skew_deg, 2), "ocr_source": p.ocr_source,
                           "elements": [e.to_dict() for e in p.elements]} for p in self.pages],
                "sections": self.sections}
