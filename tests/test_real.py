"""Plumbing tests for the real-dataset path using fixtures written in the genuine FUNSD / DocLayNet(COCO) formats."""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from docvision.synth import make_page
from docvision.schema import Word
from docvision.datasets import load_funsd, funsd_pairs, load_doclaynet
from docvision.benchmark_real import kv_match, score_kv, run_funsd, run_doclaynet
from docvision.neural_forms import words_to_entities, link_entities


def write_funsd(root, seeds):
    for sub in ("images", "annotations"):
        os.makedirs(os.path.join(root, "testing_data", sub), exist_ok=True)
    for s in seeds:
        img, gt = make_page(s, first=False)
        img.save(os.path.join(root, "testing_data", "images", f"d{s}.png"))
        form, i = [], 0
        for f in gt["forms"]:
            q, a = i, i + 1
            form.append({"id": q, "label": "question", "text": f["key"] + ":", "box": [0, 0, 1, 1], "words": [], "linking": [[q, a]]})
            form.append({"id": a, "label": "answer", "text": f["value"], "box": [0, 0, 1, 1], "words": [], "linking": [[q, a]]})
            i += 2
        form.append({"id": i, "label": "header", "text": "FORM", "box": [0, 0, 1, 1], "words": [], "linking": []})
        json.dump({"form": form}, open(os.path.join(root, "testing_data", "annotations", f"d{s}.json"), "w"))


def write_coco(root, seeds):
    os.makedirs(os.path.join(root, "PNG"), exist_ok=True); os.makedirs(os.path.join(root, "COCO"), exist_ok=True)
    names = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
    cat = {n: i + 1 for i, n in enumerate(names)}
    m = {"title": "Title", "heading": "Section-header", "paragraph": "Text", "table": "Table", "figure": "Picture", "form_field": "Text"}
    coco = {"images": [], "annotations": [], "categories": [{"id": v, "name": k} for k, v in cat.items()]}
    k = 0
    for s in seeds:
        img, gt = make_page(s, first=(s % 2 == 0))
        img.save(os.path.join(root, "PNG", f"p{s}.png"))
        coco["images"].append({"id": s, "file_name": f"p{s}.png", "width": img.width, "height": img.height})
        for e in gt["elements"]:
            x0, y0, x1, y1 = e["bbox"]
            k += 1
            coco["annotations"].append({"id": k, "image_id": s, "category_id": cat[m[e["type"]]], "bbox": [x0, y0, x1 - x0, y1 - y0]})
        # a page header the benchmark must ignore rather than penalise
        k += 1
        coco["annotations"].append({"id": k, "image_id": s, "category_id": cat["Page-header"], "bbox": [100, 20, 300, 24]})
    json.dump(coco, open(os.path.join(root, "COCO", "test.json"), "w"))


class Loaders(unittest.TestCase):
    def test_funsd_pairs_and_loader(self):
        with tempfile.TemporaryDirectory() as d:
            write_funsd(d, [1])
            (img, form), = list(load_funsd(d, "testing_data"))
            pairs = funsd_pairs(form)
            self.assertTrue(len(pairs) >= 3 and all(k and v for k, v in pairs))
            self.assertFalse(any(k.endswith(":") for k, _ in pairs))   # colons normalised away

    def test_doclaynet_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            write_coco(d, [2])
            (p, gts), = list(load_doclaynet(os.path.join(d, "COCO", "test.json"), os.path.join(d, "PNG")))
            types = [t for t, _ in gts]
            self.assertIn(None, types)            # page-header ignored, not mapped
            self.assertTrue({"title", "paragraph"} & set(types))


class KV(unittest.TestCase):
    def test_fuzzy_match_tolerates_ocr_noise(self):
        tp, fp, fn, _ = score_kv([("name", "ann lee"), ("x", "y")], [("name", "ann iee"), ("city", "pune")])
        self.assertEqual((tp, fp, fn), (1, 1, 1))

    def test_one_to_one(self):
        self.assertEqual(len(kv_match([("a", "b"), ("a", "b")], [("a", "b")])), 1)


class LinkingLogic(unittest.TestCase):
    W = staticmethod(lambda t, x0, y0, x1, y1: Word(t, (x0, y0, x1, y1), 1.0))

    def test_entities_and_links(self):
        W = self.W
        words = [W("Name", 100, 100, 160, 120), W(":", 160, 100, 168, 120), W("Ann", 300, 100, 340, 120), W("Lee", 345, 100, 385, 120),
                 W("City", 100, 160, 150, 180), W("Pune", 300, 160, 350, 180)]
        labels = ["B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "B-QUESTION", "B-ANSWER"]
        ents = words_to_entities(words, labels)
        self.assertEqual([(e["type"], e["text"]) for e in ents],
                         [("question", "Name :"), ("answer", "Ann Lee"), ("question", "City"), ("answer", "Pune")])
        pairs = link_entities(ents)
        self.assertEqual([(k.strip(), v) for k, v, *_ in pairs], [("Name", "Ann Lee"), ("City", "Pune")])

    def test_answer_below_question(self):
        W = self.W
        words = [W("Address", 100, 100, 200, 120), W("12", 100, 130, 130, 150), W("Main", 135, 130, 190, 150)]
        ents = words_to_entities(words, ["B-QUESTION", "B-ANSWER", "I-ANSWER"])
        self.assertEqual([(k, v) for k, v, *_ in link_entities(ents)], [("Address", "12 Main")])


class EndToEnd(unittest.TestCase):
    def test_funsd_rules_arm_runs(self):
        with tempfile.TemporaryDirectory() as d:
            write_funsd(d, [1, 2])
            res = run_funsd(d, "testing_data", 2, ["rules"], os.path.join(d, "out"))
            self.assertGreater(res["arms"]["rules"]["f1"], 0.8)     # synthetic forms: sanity, not a claim
            self.assertTrue(os.path.exists(os.path.join(d, "out", "real_funsd_failures.json")))

    def test_doclaynet_arms_run(self):
        with tempfile.TemporaryDirectory() as d:
            write_coco(d, [1, 2])
            res = run_doclaynet(os.path.join(d, "COCO", "test.json"), os.path.join(d, "PNG"), 2,
                                ["classical", "ocr_only"], os.path.join(d, "out"))
            self.assertGreater(res["arms"]["classical"]["macro_f1"], res["arms"]["ocr_only"]["macro_f1"])


if __name__ == "__main__":
    unittest.main()
