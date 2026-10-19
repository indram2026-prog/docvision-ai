"""Document-level reasoning: cross-page table continuation and heading hierarchy."""
from __future__ import annotations
from .schema import Document


def link_pages(doc: Document, bottom_frac=0.12, top_frac=0.14):
    """Merge a table that runs off the bottom of one page into the table at the top of the next."""
    for a, b in zip(doc.pages, doc.pages[1:]):
        ta = [e for e in a.elements if e.type == "table"]
        tb = [e for e in b.elements if e.type == "table"]
        if not ta or not tb:
            continue
        last, first = ta[-1], tb[0]
        near_bottom = last.bbox[3] >= a.height * (1 - bottom_frac)
        near_top = first.bbox[1] <= b.height * top_frac
        if near_bottom and near_top and last.cells and first.cells and len(last.cells[0]) == len(first.cells[0]):
            first.continued = True
            if first.cells[0] == last.cells[0] and len(first.cells) > 1:   # repeated header row
                first.cells = first.cells[1:]
                first.cell_boxes = first.cell_boxes[1:]
            first.text = "\n".join(" | ".join(r) for r in first.cells)
    build_sections(doc)


def build_sections(doc: Document):
    """Nested outline from title/heading elements; attaches content element refs to each section."""
    root = {"title": "(document)", "level": 0, "page": 0, "children": [], "content": []}
    stack = [root]
    for p in doc.pages:
        for e in sorted(p.elements, key=lambda e: e.order):
            if e.type in ("title", "heading"):
                lvl = e.level or 2
                node = {"title": e.text, "level": lvl, "page": p.number, "children": [], "content": []}
                while stack[-1]["level"] >= lvl:
                    stack.pop()
                stack[-1]["children"].append(node)
                stack.append(node)
            else:
                stack[-1]["content"].append({"page": p.number, "order": e.order, "type": e.type})
    doc.sections = root["children"]
