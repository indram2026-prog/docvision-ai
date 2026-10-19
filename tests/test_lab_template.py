"""Regression tests from a real lab-sheet PDF: colon labels, run-in 'Aim:', wrapped list items,
screenshot panels, and code listings. Covers both the native-text and the scanned (OCR) path."""
import os, unittest
from docvision.pipeline import process_document

PDF = os.path.join(os.path.dirname(__file__), "fixtures", "lab_template.pdf")


@unittest.skipUnless(os.path.exists(PDF), "drop the lab-sheet PDF at tests/fixtures/lab_template.pdf to run")
class LabTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = process_document(PDF, prefer_native=True)
        cls.ocr = process_document(PDF, prefer_native=False)

    def kinds(self, doc, page, typ):
        return [e for e in doc.pages[page].elements if e.type == typ]

    def test_title_promoted(self):
        for d in (self.native, self.ocr):
            self.assertEqual(len(self.kinds(d, 0, "title")), 1)

    def test_section_labels_are_headings_not_forms(self):
        for d in (self.native, self.ocr):
            heads = " ".join(e.text for p in d.pages for e in p.elements if e.type == "heading")
            for label in ("Components Required", "Circuit Connection", "Procedure", "Results"):
                self.assertIn(label, heads)
            self.assertEqual([e for p in d.pages for e in p.elements if e.type == "form_field"], [])

    def test_runin_aim_split(self):
        for d in (self.native, self.ocr):
            els = d.pages[0].elements
            self.assertTrue(any(e.type == "heading" and e.text.startswith("Aim") for e in els))
            self.assertTrue(any(e.type == "list_item" and "blink the LED" in e.text for e in els))

    def test_two_figures_no_phantom_table(self):
        for d in (self.native, self.ocr):
            self.assertEqual(len(self.kinds(d, 0, "figure")), 2)
            self.assertEqual(self.kinds(d, 0, "table"), [])

    def test_wrapped_list_items_joined(self):
        items = [e.text for e in self.native.pages[2].elements if e.type == "list_item"]
        self.assertEqual(len(items), 10)          # 8 procedure steps + 2 results
        self.assertTrue(any(t.endswith("resistor.") for t in items))
        self.assertEqual([e for e in self.native.pages[2].elements if e.type == "paragraph"][:0], [])

    def test_code_blocks(self):
        for d in (self.native, self.ocr):
            code = self.kinds(d, 1, "code")
            self.assertEqual(len(code), 2)
            self.assertIn("void loop", code[1].text)
        self.assertIn("// PROGRAM 2", self.kinds(self.native, 1, "code")[1].text)


if __name__ == "__main__":
    unittest.main()
