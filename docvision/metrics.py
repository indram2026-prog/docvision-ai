"""Evaluation metrics (dependency-free)."""
from __future__ import annotations
import re
from .schema import iou, CLASSES


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    pred, ref = norm(pred), norm(ref)
    return lev(pred, ref) / max(1, len(ref))


def match(pred, gt, thr=0.5):
    """Greedy one-to-one matching, same class, IoU >= thr. pred/gt: lists of (type, bbox). Returns pairs (pi, gi, iou)."""
    cand = sorted(((iou(p[1], g[1]), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt)
                   if p[0] == g[0]), reverse=True)
    up, ug, out = set(), set(), []
    for v, pi, gi in cand:
        if v < thr:
            break
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); out.append((pi, gi, v))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def reading_order_score(pairs, pred_order, gt_order):
    """Pairwise concordance of reading order among matched elements (1.0 = identical order)."""
    ok = tot = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            tot += 1
            ok += (pred_order[a[0]] < pred_order[b[0]]) == (gt_order[a[1]] < gt_order[b[1]])
    return ok / tot if tot else 1.0
