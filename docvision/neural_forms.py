"""Multimodal-transformer arm: LayoutLMv3 fine-tuned on FUNSD (text + 2-D layout + image patches).

NOT executed in the build sandbox (no Hugging Face access there). Everything model-free (entity grouping, linking)
is unit-tested; the model call follows the documented transformers API - check it on your machine:

    pip install torch transformers
    python -m docvision.neural_forms path/to/form.png

Model: Narsil/layoutlmv3-finetuned-funsd  (token classes O, B/I-HEADER, B/I-QUESTION, B/I-ANSWER).
Same Tesseract words feed every arm, so differences come from the model, not the OCR.
"""
from __future__ import annotations
from typing import List, Tuple
from .schema import Word, union
from .metrics import norm

MODEL_ID = "Narsil/layoutlmv3-finetuned-funsd"


# ------------------------------------------------------------ model-free logic (tested)
def words_to_entities(words: List[Word], labels: List[str], max_gap_h: float = 1.6):
    """Group consecutive same-type words (B-/I- tagging) into entities in reading order.
    labels[i] is e.g. 'B-QUESTION', 'I-ANSWER' or 'O'."""
    order = sorted(range(len(words)), key=lambda i: ((words[i].bbox[1] + words[i].bbox[3]) // 2 // 12, words[i].bbox[0]))
    ents, cur = [], None
    for i in order:
        lab = labels[i]
        if lab == "O":
            cur = None; continue
        pre, typ = (lab.split("-", 1) + [""])[:2] if "-" in lab else ("B", lab)
        typ = typ.lower()
        w = words[i]
        h = max(1, w.bbox[3] - w.bbox[1])
        if cur and pre == "I" and cur["type"] == typ and (w.bbox[0] - cur["words"][-1].bbox[2] < max_gap_h * h * 6
                                                           or w.bbox[1] > cur["words"][-1].bbox[1] + h // 2):
            cur["words"].append(w)
        else:
            cur = {"type": typ, "words": [w]}; ents.append(cur)
    for e in ents:
        e["text"] = " ".join(w.text for w in e["words"])
        e["bbox"] = union([w.bbox for w in e["words"]])
    return ents


def link_entities(ents, max_dist_h: float = 12.0):
    """Pair each QUESTION with its nearest ANSWER: same row to the right first, else directly below."""
    qs = [e for e in ents if e["type"] == "question"]
    ans = [e for e in ents if e["type"] == "answer"]
    cands = []
    for qi, q in enumerate(qs):
        qh = max(1, q["bbox"][3] - q["bbox"][1])
        for ai, a in enumerate(ans):
            ov = min(q["bbox"][3], a["bbox"][3]) - max(q["bbox"][1], a["bbox"][1])
            same_row = ov > 0.5 * qh and a["bbox"][0] >= q["bbox"][2] - 5
            below = a["bbox"][1] >= q["bbox"][3] - 5 and abs(a["bbox"][0] - q["bbox"][0]) < 4 * qh \
                and a["bbox"][1] - q["bbox"][3] < 3 * qh
            if same_row:
                d = a["bbox"][0] - q["bbox"][2]
            elif below:
                d = 2 * qh + (a["bbox"][1] - q["bbox"][3]) + abs(a["bbox"][0] - q["bbox"][0])
            else:
                continue
            if d <= max_dist_h * qh * 4:
                cands.append((d, qi, ai))
    used_q, used_a, pairs = set(), set(), []
    for d, qi, ai in sorted(cands):
        if qi in used_q or ai in used_a:
            continue
        used_q.add(qi); used_a.add(ai)
        pairs.append((qs[qi]["text"].strip().rstrip(":").strip(), ans[ai]["text"].strip(), qs[qi]["bbox"], ans[ai]["bbox"]))
    return pairs


# ------------------------------------------------------------ model call (verify locally)
class LayoutLMv3KV:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        import torch
        from transformers import AutoProcessor, AutoModelForTokenClassification
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict_words(self, pil, words: List[Word]) -> List[str]:
        """One BIO label per input word (label of the word's first sub-token; windows of 512 tokens, stride 128)."""
        torch = self.torch
        W, H = pil.size
        clip = lambda v: max(0, min(1000, int(v)))
        boxes = [[clip(1000 * w.bbox[0] / W), clip(1000 * w.bbox[1] / H), clip(1000 * w.bbox[2] / W), clip(1000 * w.bbox[3] / H)]
                 for w in words]
        enc = self.processor(pil.convert("RGB"), [w.text for w in words], boxes=boxes, truncation=True, max_length=512,
                             stride=128, padding="max_length", return_overflowing_tokens=True, return_tensors="pt")
        word_ids = [enc.word_ids(i) for i in range(enc["input_ids"].shape[0])]
        for k in ("overflow_to_sample_mapping", "offset_mapping"):
            enc.pop(k, None)
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        probs = logits.softmax(-1)
        best = {}
        for b in range(probs.shape[0]):
            seen = set()
            for t, wid in enumerate(word_ids[b]):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                p, c = probs[b, t].max(-1)
                if wid not in best or p.item() > best[wid][0]:
                    best[wid] = (p.item(), self.id2label[int(c)])
        return [best.get(i, (0, "O"))[1] for i in range(len(words))]

    def extract_pairs(self, pil, words: List[Word]):
        labels = self.predict_words(pil, words)
        ents = words_to_entities(words, labels)
        return link_entities(ents), ents


if __name__ == "__main__":
    import sys, numpy as np
    from PIL import Image
    from .ocr import tesseract_lines
    img = Image.open(sys.argv[1]).convert("RGB")
    words = [w for l in tesseract_lines(np.asarray(img)) for w in l.words]
    pairs, _ = LayoutLMv3KV().extract_pairs(img, words)
    for k, v, *_ in pairs:
        print(f"{k!r:30} -> {v!r}")
