"""Lightweight NLP: type extracted values and keys (date, amount, email, phone, percent, id, number, text).
Rule-based on purpose (CPU-only, deterministic, explainable). A learned NER can be dropped in via `type_value`."""
from __future__ import annotations
import re

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
PATTERNS = [
    ("email", re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")),
    ("url", re.compile(r"^(?:https?://|www\.)\S+$", re.I)),
    ("date", re.compile(rf"^(?:\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|"
                        rf"\d{{1,2}}\s+{_MONTH}\s+\d{{2,4}}|{_MONTH}\s+\d{{1,2}},?\s+\d{{2,4}})$", re.I)),
    ("amount", re.compile(r"^(?:[$€£₹]|rs\.?|inr|usd|eur)\s?\d[\d,]*(?:\.\d+)?$|^\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|inr)$", re.I)),
    ("percent", re.compile(r"^-?\d+(?:\.\d+)?\s?%$")),
    ("phone", re.compile(r"^\+?\d[\d\s().-]{7,17}\d$")),
    ("number", re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")),
    ("id", re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9/_-]{3,}$")),
]
KEY_HINTS = {"date": ("date", "dob", "born", "issued", "due"), "email": ("email", "e-mail"),
             "phone": ("phone", "mobile", "tel", "fax"), "amount": ("amount", "total", "price", "cost", "balance", "fee"),
             "person": ("name", "contact", "manager", "author", "signed"), "address": ("address", "street", "city", "zip", "postal"),
             "organization": ("company", "organisation", "organization", "department", "employer", "vendor")}


def type_value(value: str, key: str = "") -> str:
    v = (value or "").strip()
    if not v:
        return "empty"
    for name, rx in PATTERNS:
        if rx.match(v):
            if name == "number" and any(h in (key or "").lower() for h in KEY_HINTS["amount"]):
                return "amount"
            if name == "phone" and re.fullmatch(r"\d{4}", v):
                return "number"
            return name
    k = (key or "").lower()
    for t in ("person", "address", "organization"):
        if any(h in k for h in KEY_HINTS[t]):
            return t
    return "text"


def type_cells(cells):
    """Per-column dominant type of a table body (header row excluded). Useful for schema inference."""
    if len(cells) < 2:
        return []
    out = []
    for c in range(len(cells[0])):
        ts = [type_value(r[c]) for r in cells[1:] if c < len(r) and r[c].strip()]
        out.append(max(set(ts), key=ts.count) if ts else "empty")
    return out
