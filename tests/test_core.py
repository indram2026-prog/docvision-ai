import io, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from docvision.schema import iou, Document
from docvision.metrics import cer, match, lev
from docvision.synth import make_page, degrade
from docvision.pipeline import process_image, baseline_ocr_only
from docvision.preprocess import estimate_skew, map_bbox_back
from docvision.structure import link_pages
from docvision.export import to_markdown
from docvision.service import analyze_file
from docvision.layout import xy_cut
from docvision.schema import Element
import cv2, numpy as np


class Metrics(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 1, 1), (5, 5, 6, 6)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=4)

    def test_cer(self):
        self.assertEqual(lev("kitten", "sitting"), 3)
        self.assertEqual(cer("Hello  World", "hello world"), 0.0)

    def test_match_requires_same_class(self):
        pairs = match([("table", (0, 0, 10, 10))], [("figure", (0, 0, 10, 10))])
        self.assertEqual(pairs, [])


class Geometry(unittest.TestCase):
    def test_skew_estimate(self):
        img, _ = make_page(3)
        rot = degrade(img, "skew")  # +1.8 deg CCW
        ang = estimate_skew(cv2.cvtColor(np.asarray(rot), cv2.COLOR_RGB2GRAY))
        self.assertLess(abs(abs(ang) - 1.8), 0.4)

    def test_bbox_roundtrip_identity(self):
        self.assertEqual(map_bbox_back((1, 2, 30, 40), (100, 100, 3), 0.0), (1, 2, 30, 40))

    def test_xy_cut_two_columns(self):
        mk = lambda x0, y0, x1, y1: Element(type="paragraph", bbox=(x0, y0, x1, y1))
        a, b, c, d = mk(0, 0, 100, 50), mk(0, 60, 100, 120), mk(200, 0, 300, 50), mk(200, 60, 300, 120)
        self.assertEqual(xy_cut([c, a, d, b]), [a, b, c, d])


class Pipeline(unittest.TestCase):
    def test_clean_page_structure(self):
        img, gt = make_page(1)
        pg = process_image(img)
        got = [e.type for e in pg.elements]
        self.assertEqual(got, [e["type"] for e in gt["elements"]])
        tbl = [e for e in pg.elements if e.type == "table"][0]
        self.assertEqual(len(tbl.cells), len(gt["tables"][0]["cells"]))

    def test_baseline_has_no_structure(self):
        img, _ = make_page(1)
        types = {e.type for e in baseline_ocr_only(img).elements}
        self.assertEqual(types, {"paragraph"})

    def test_blank_page_is_safe(self):
        from PIL import Image
        pg = process_image(Image.new("RGB", (800, 1000), "white"))
        self.assertEqual(pg.elements, [])

    def test_cross_page_table_merge(self):
        def pg(n, y0, y1, cells, h=1754):
            from docvision.schema import Page
            e = Element(type="table", bbox=(100, y0, 900, y1), cells=cells, cell_boxes=[[(0, 0, 1, 1)] * 2 for _ in cells], page=n)
            return Page(number=n, width=1240, height=h, elements=[e])
        a = pg(0, 1000, 1700, [["A", "B"], ["1", "2"]])
        b = pg(1, 100, 400, [["A", "B"], ["3", "4"]])
        doc = Document(pages=[a, b]); link_pages(doc)
        self.assertTrue(b.elements[0].continued)
        self.assertEqual(b.elements[0].cells, [["3", "4"]])  # repeated header dropped

    def test_markdown_export(self):
        img, _ = make_page(1)
        md = to_markdown(Document(pages=[process_image(img)]))
        self.assertTrue(md.startswith("# ")); self.assertIn("|---|", md)


class Service(unittest.TestCase):
    def test_analyze_png_bytes(self):
        img, _ = make_page(2); buf = io.BytesIO(); img.save(buf, "PNG")
        out = analyze_file(buf.getvalue(), "x.png", "both")
        self.assertIn("structured", out); self.assertIn("ocr_only", out)
        self.assertEqual(out["pages"], 1)

    def test_rejects_bad_type(self):
        with self.assertRaises(Exception):
            analyze_file(b"not a file", "x.docx", "structured")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Neural(unittest.TestCase):
    def test_regions_to_elements_with_stub_detector(self):
        from docvision.neural import regions_to_elements
        from docvision.schema import Line, Word
        mk = lambda t, x0, y0, x1, y1: Line([Word(t, (x0, y0, x1, y1), .9)], (x0, y0, x1, y1), t, y1 - y0)
        lines = [mk("Report", 100, 100, 400, 140), mk("body one", 100, 200, 500, 224), mk("body two", 100, 230, 500, 254),
                 mk("stray", 100, 900, 300, 924)]
        regions = [("title", (90, 90, 410, 150), .9), ("paragraph", (90, 190, 510, 260), .8), ("figure", (90, 400, 500, 800), .7)]
        els = regions_to_elements(regions, lines)
        self.assertEqual([e.type for e in els], ["title", "paragraph", "figure", "paragraph"])
        self.assertEqual(els[1].text, "body one body two")
        self.assertEqual(els[3].text, "stray")   # unclaimed text is retained


class Borderless(unittest.TestCase):
    @staticmethod
    def W(t, x0, y0, w=60, h=18):
        from docvision.schema import Word
        return Word(t, (x0, y0, x0 + w, y0 + h), 1.0)

    def grid(self, rows, xs, y0=100, dy=34):
        return [self.W(t, xs[c], y0 + r * dy, w=8 * len(t)) for r, row in enumerate(rows) for c, t in enumerate(row)]

    def test_detects_aligned_columns(self):
        from docvision.borderless import detect_borderless_tables
        rows = [["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Pear", "10", "0.80"], ["Plum", "7", "2.05"]]
        t = detect_borderless_tables(self.grid(rows, [100, 400, 700]))
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["cells"], rows)

    def test_wrapped_cell_is_folded_into_row(self):
        from docvision.borderless import detect_borderless_tables
        ws = self.grid([["Name", "Notes"], ["Ann", "first"], ["Bob", "second"], ["Cy", "third"]], [100, 500])
        ws.append(self.W("continued", 500, 100 + 2 * 34 + 20))   # wraps under row 'Bob', within the row gap
        t = detect_borderless_tables(ws)
        self.assertEqual(len(t), 1)
        self.assertIn("continued", t[0]["cells"][2][1])

    def test_prose_is_not_a_table(self):
        from docvision.borderless import detect_borderless_tables
        words = []
        for r in range(5):
            x = 100
            for i in range(14):
                words.append(self.W("word", x, 100 + r * 30, w=40)); x += 48
        self.assertEqual(detect_borderless_tables(words), [])

    def test_form_rows_are_not_a_table(self):
        from docvision.borderless import detect_borderless_tables
        rows = [["Name:", "Ann Lee"], ["City:", "Pune"], ["Email:", "a@b.co"], ["Phone:", "5551234"]]
        self.assertEqual(detect_borderless_tables(self.grid(rows, [100, 400])), [])


class FieldTypes(unittest.TestCase):
    def test_types(self):
        from docvision.fieldtypes import type_value, type_cells
        self.assertEqual(type_value("12/03/2024"), "date")
        self.assertEqual(type_value("March 3, 2025"), "date")
        self.assertEqual(type_value("$1,200.50"), "amount")
        self.assertEqual(type_value("a.b@mail.com"), "email")
        self.assertEqual(type_value("45%"), "percent")
        self.assertEqual(type_value("+91 98765 43210"), "phone")
        self.assertEqual(type_value("1500", "Total"), "amount")
        self.assertEqual(type_value("Ann Lee", "Name"), "person")
        self.assertEqual(type_value("", "Name"), "empty")
        self.assertEqual(type_cells([["a", "b"], ["x", "3"], ["y", "4"]]), ["text", "number"])


class Bullets(unittest.TestCase):
    def test_bullet_on_own_line_is_merged(self):
        from docvision.layout import classify_lines
        from docvision.schema import Line, Word
        mk = lambda t, x0, y0, x1, y1: Line([Word(t, (x0, y0, x1, y1), 1.0)], (x0, y0, x1, y1), t, y1 - y0)
        lines = [mk("•", 147, 956, 147, 973), mk("Working prototype", 143, 980, 400, 997), mk("Body text goes here", 143, 1100, 600, 1117)]
        kinds = [k for _, k, _ in classify_lines(lines, [])]
        self.assertEqual(kinds, ["list", "text"])


class PixelHeuristics(unittest.TestCase):
    def test_mark_bold_flags_thicker_strokes(self):
        from PIL import Image, ImageDraw, ImageFont
        from docvision.layout import mark_bold
        from docvision.schema import Line, Word
        from docvision.synth import FONT, BOLD
        img = Image.new("L", (900, 400), 255); d = ImageDraw.Draw(img)
        reg, bold = ImageFont.truetype(FONT, 24), ImageFont.truetype(BOLD, 24)
        lines = []
        for i in range(5):
            y = 20 + i * 60
            f, t = (bold, "SECTION LABEL") if i == 2 else (reg, "a regular body line of running text that is long")
            d.text((20, y), t, font=f, fill=0)
            bb = d.textbbox((20, y), t, font=f)
            lines.append(Line([Word(t, bb, 1.0)], bb, t, bb[3] - bb[1]))
        mark_bold(np.asarray(img), lines)
        self.assertEqual([l.bold for l in lines], [False, False, True, False, False])

    def test_merge_row_fragments(self):
        from docvision.layout import merge_row_fragments
        from docvision.schema import Line, Word
        mk = lambda t, x0, x1: Line([Word(t, (x0, 120, x1, 162), 1.0)], (x0, 120, x1, 162), t, 42)
        out = merge_row_fragments([mk("Si", 114, 163), mk("nple Future", 201, 700)])
        self.assertEqual(len(out), 1)
        far = merge_row_fragments([mk("left", 100, 300), mk("right", 700, 900)])   # a real column gutter stays split
        self.assertEqual(len(far), 2)

    def test_label_with_colon_joins_distant_value(self):
        from docvision.layout import merge_row_fragments
        from docvision.schema import Line, Word
        mk = lambda t, x0, x1: Line([Word(t, (x0, 752, x1, 770), 1.0)], (x0, 752, x1, 770), t, 18)
        out = merge_row_fragments([mk("Company:", 111, 227), mk("Status 7157", 382, 523)])
        self.assertEqual([l.text for l in out], ["Company: Status 7157"])
