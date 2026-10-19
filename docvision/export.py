"""Serialise a Document to Markdown / JSON while preserving structure."""
from __future__ import annotations
from .schema import Document


def _md_table(cells):
    if not cells:
        return ""
    esc = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")
    head, *rest = cells
    out = ["| " + " | ".join(esc(c) for c in head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rest]
    return "\n".join(out)


def to_markdown(doc: Document) -> str:
    out = []
    for p in doc.pages:
        if len(doc.pages) > 1:
            out.append(f"\n<!-- page {p.number + 1} -->\n")
        form_buf = []
        for e in sorted(p.elements, key=lambda e: e.order):
            if e.type != "form_field" and form_buf:
                out.append("\n".join(form_buf)); form_buf = []
            if e.type == "title":
                out.append(f"# {e.text}")
            elif e.type == "heading":
                out.append(f"## {e.text}")
            elif e.type == "paragraph":
                out.append(e.text)
            elif e.type == "code":
                out.append("```\n" + e.text + "\n```")
            elif e.type == "list_item":
                out.append(f"- {e.text}")
            elif e.type == "table":
                out.append(("*(table continued)*\n\n" if e.continued else "") + _md_table(e.cells))
            elif e.type == "figure":
                x0, y0, x1, y1 = e.bbox
                out.append(f"![figure p{p.number + 1} @({x0},{y0},{x1},{y1})]()")
            elif e.type == "form_field":
                form_buf.append(f"- **{e.key}:** {e.value or '_(blank)_'}")
        if form_buf:
            out.append("\n".join(form_buf))
    return "\n\n".join(s for s in out if s.strip()) + "\n"
