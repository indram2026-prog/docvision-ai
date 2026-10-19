---
title: DocVision AI
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# DocVision AI — Intelligent Document Structure Understanding

Challenge 9, AI Innovation Hackathon 2026. Treats a document as a **structured visual object** (layout + text + tables + forms + hierarchy) instead of a bag of OCR text. CPU-only, no GPU, no model downloads required for the core path.

## What it does

Upload a PDF/image → detects **title, heading, paragraph, table, figure, form field** → extracts **table cell grids** and **form key/value pairs** → recovers **reading order** (XY-cut, multi-column safe), **heading hierarchy**, and **cross-page table continuation** → exports JSON / Markdown. The React UI overlays boxes on the page, and a **Compare** view shows the OCR-only baseline side by side.

## Method (all classical, deterministic, explainable)

| Stage | Technique |
|---|---|
| Conditioning | percentile contrast stretch; projection-profile deskew (boxes mapped back to the original frame) |
| Text | Tesseract on a page with table/figure/underline regions blanked; **native PDF text layer** (with bold flags) for born-digital PDFs |
| Tables (ruled) | morphological rule extraction → connected grid → ≥3×3 rules → cell grid → per-cell OCR / word assignment |
| Tables (borderless) | word rows → cell segments → whitespace *rivers* across ≥3 rows → columns; wrapped cell lines folded into their row; `Label:` form rows and long prose rejected (`borderless.py`) |
| Figures | dark-core density + colour saturation on a denoised image; figure-first arbitration vs. table candidates |
| Forms | `Label: value` pattern + underline geometry; values typed as date / amount / email / phone / percent / id / number / person / address / organization (`fieldtypes.py`) |
| Lists | bullet / numbered markers (incl. PDFs that put the bullet glyph on its own line) → `list_item` |
| Headings | height ratio (title), height/char-width ratio (heading), same-size bold run-in labels — bold from PDF font flags, or on scans from stroke width (distance transform) |
| OCR repair | re-OCR of large type at body scale; lines split by a dropped glyph are re-joined |
| Document level | XY-cut reading order, nested section tree, repeated-header-aware cross-page table merge |

## Evaluation (synthetic benchmark, `python -m docvision.eval`)

Ground truth comes from a documented, seeded synthetic generator (`docvision/synth.py`) — permitted by the brief. The `borderless` condition draws tables with no ruling lines at all. Both systems share identical preprocessing, so the comparison isolates **structure understanding**. Detection = same class and IoU ≥ 0.5.

| Condition | Pages | Structured macro-F1 | OCR-only macro-F1 | Mean IoU | Table cell acc. | Form KV F1 | Reading order | Page CER (struct) | Page CER (OCR-only) |
|---|---|---|---|---|---|---|---|---|---|
| clean | 12 | **1.0** | 0.046 | 0.951 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0004 |
| noise | 6 | **1.0** | 0.055 | 0.955 | 0.973 | 1.0 | 1.0 | 0.0008 | 0.0021 |
| blur | 6 | **0.989** | 0.078 | 0.927 | 1.0 | 1.0 | 1.0 | 0.0028 | 0.1198 |
| lowcontrast | 6 | **1.0** | 0.047 | 0.956 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0008 |
| skew | 6 | **1.0** | 0.056 | 0.968 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0001 |
| borderless | 6 | **1.0** | 0.056 | 0.953 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |

**Per-class F1 (structured pipeline)**

| Condition | title | heading | paragraph | table | figure | form_field |
|---|---|---|---|---|---|---|
| clean | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| noise | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| blur | 1.0 | 0.933 | 1.0 | 1.0 | 1.0 | 1.0 |
| lowcontrast | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| skew | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| borderless | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Runtime (1 CPU core, Tesseract): structured 6.93 s/page, OCR-only 4.75 s/page.

**Read these numbers carefully.**

- Several conditions score a perfect 1.0. That means the benchmark is **saturated and was tuned against**, not that the system is perfect: a fresh seed range (5000+) exposed a real bug (Tesseract splitting `Label:` from its value on borderless pages), which was then fixed — so those seeds are no longer held out either. Treat 1.0 as "no regressions on a simple generator".
- They are near-ceiling because the generator is simple: one font, mostly single-column, ruled tables, clean forms. The rules were also debugged against this generator, and a few seeds were inspected while debugging, so treat this as a **regression/ablation harness, not a claim about real documents**.
- The OCR-only baseline scores ~0.05 macro-F1 by construction: it cannot emit tables, figures, forms or headings. **Structure is the contribution; raw character accuracy is not** — page CER is essentially tied (character accuracy is not where the structured path wins). The one real gap is blur, where the structured path's region-blanking helps Tesseract.
- Debugging history worth knowing: the first benchmark run exposed three real bugs (blurred text classified as figures, colour noise read as saturation, and an evaluation-frame error for skew), now fixed and covered by the benchmark.

## Real-document check (no hand labels)

Run on the challenge brief PDF (`results/overlay_real_pdf_page2.png`): titles, headings (including same-size bold section labels), paragraphs, the outline tree, **the contents list on page 1 as a borderless table, and the deliverable bullets as list items** come out correctly; 3 pages in ~2 s via the native text layer.

Because there is no ground truth, `results/FAILURES.md` also scores the **pixel path (Tesseract on rendered pages) against the text-layer path** of the same PDF. That isolates OCR/pixel degradation on real fonts; it does *not* validate layout correctness (both paths share the rules). Adding stroke-width bold detection raised heading agreement on that check from F1 0.50 to 0.84 — note that the check was run on the same document used to develop the fix.

Known defects: `→` is extracted as `fi` by pdfplumber's glyph mapping; table header rows are not typed; no merged cells.

## Real datasets named in the brief (run on your machine)

The sandbox that built this had no access to Hugging Face or dataset hosts, so **no number below exists yet for FUNSD / DocLayNet — they are produced by one command each**. Loaders, scoring and linking logic are unit-tested on fixtures written in the genuine FUNSD and DocLayNet-COCO formats (`tests/test_real.py`); the two neural models were never run by the author.

```bash
pip install torch transformers ultralytics huggingface_hub
# FUNSD: key-value pair F1, rules vs LayoutLMv3 (multimodal transformer, Narsil/layoutlmv3-finetuned-funsd)
python -m docvision.benchmark_real funsd --root /data/FUNSD --split testing_data --n 50 --arms rules,layoutlmv3
# DocLayNet core: per-class P/R/F1 @IoU 0.5 for classical vs OCR-only vs YOLO11-DocLayNet
python -m docvision.benchmark_real doclaynet --ann /data/DocLayNet/COCO/test.json --img-dir /data/DocLayNet/PNG --n 100
python -m docvision.failure_analysis --results results --pdf your.pdf     # regenerates results/FAILURES.md
```

Fairness notes: all arms see the same Tesseract words; the classical pipeline is rescaled to ~150 dpi width (1240 px) because its thresholds assume that; DocLayNet regions with no DocVision class (page-header/footer, formula) are ignored rather than counted as false positives; `list-item` is scored as `paragraph`. The OCR-only arm is not run on FUNSD: it cannot emit key-value pairs, so its score is zero by construction. Expect the classical rules to score low on FUNSD (many forms have no `Label:` colons); that gap is exactly what the LayoutLMv3 arm is there to show.

## Failure analysis

See `results/FAILURES.md` (generated). Categories: confusion, localisation, missed, spurious, with per-condition counts and the real-PDF pixel-vs-text-layer breakdown. Honest summary of what the failures taught: the first synthetic errors were OCR fragmenting large type (fixed by re-joining row fragments) and the pixel path lacking the bold flag that the text layer has (fixed with stroke width); the remaining weaknesses are structural, not tuning: merged cells, handwriting, non-Latin scripts, multi-column forms, and anything the generator never draws.

## Limitations

Borderless tables need ≥3 aligned rows (≥3 columns for exactly 3 rows) and can merge two side-by-side tables; no merged-cell (rowspan/colspan) recovery; no handwriting; Latin-script OCR only (Tesseract `eng`); heading levels are 2-tier (title/heading); photos with near-white backgrounds can be missed by the figure detector; classical form detection assumes `Label: value` (FUNSD forms without colons need the LayoutLMv3 arm); no handwriting model. Synthetic benchmark ≠ real-world accuracy.

## Run it

```bash
sudo apt install tesseract-ocr          # or: brew install tesseract
pip install -r requirements.txt
python -m unittest discover -s tests -v          # 30 tests
python -m docvision.eval --n 12 --n-deg 6 --out results && python -m docvision.report results
python -m docvision.failure_analysis --results results --pdf path/to/any.pdf

uvicorn docvision.api:app --reload               # API on :8000
cd frontend && npm install && npm run dev        # UI on :5173 (proxies /api)
# production: npm run build  -> FastAPI serves frontend/dist automatically
```

## What was verified vs. not (honest status)

| Component | Status |
|---|---|
| Pipeline, borderless tables, field typing, lists, metrics, benchmark, export, structure logic | **Run and tested** (30 unit tests + benchmark) |
| FUNSD / DocLayNet loaders, KV and layout scoring, LayoutLMv3 entity grouping + linking | **Run and tested on format-faithful fixtures** (no real data available in the sandbox) |
| `neural_forms.py` LayoutLMv3 model call | **Never run** (needs `torch`/`transformers` + weights); written to the documented API, expect small fixes |
| Real-PDF behaviour | **Run**, inspected visually, no quantitative ground truth |
| `service.py` (what the API calls) | **Tested** (`analyze_file`) |
| `api.py` FastAPI app | **Compiles only** — FastAPI could not be installed in the build sandbox |
| React frontend | **Bundled with esbuild and driven in headless Chromium** against the real `service.py` (skeleton, loaded view, cache hit, tooltip); no console errors. The Vite build and the FastAPI wrapper were not run |
| `neural.py` YOLO11-DocLayNet backend | **Plumbing unit-tested with a stub detector; real weights never run.** Model ID verified to exist on Hugging Face |

## Frontend notes

- **Skeleton loaders** in the same arrangement as the loaded page and inspector, so nothing jumps when results arrive. No spinner.
- **Result cache** (IndexedDB, keyed by SHA-256 of the file bytes + mode, newest 12 kept): re-opening a file is instant (measured 0.04 s vs 15.5 s on the mock backend, also after a page reload) and is labelled "loaded from cache".
- **Optimistic rendering**: an uploaded image appears immediately while analysis runs; copy-to-clipboard confirms instantly and reverts with a message if it fails; errors restore the start screen.
- **Tooltips** on every icon-only control and on each class in the legend; they also show on keyboard focus.
- Design choices: one accent colour, flat panels with hairlines, IBM Plex Sans/Mono, an 8-point spacing scale, state changes on hover with no fades, high-contrast light and dark themes, colour-blind-safe class colours.

## Responsible AI

Use public or permitted documents only; do not upload confidential documents or personal records (the UI says so). Outputs are research-prototype quality: OCR and layout errors occur, per-element confidence is shown, and results must not be used for consequential decisions without human review. Synthetic data is documented and seeded; the real PDF used is the hackathon's own brief.

## Layout

```
docvision/  schema synth preprocess ocr layout borderless fieldtypes pipeline structure export viz metrics eval report
            datasets benchmark_real neural neural_forms failure_analysis service api
frontend/   Vite + React (PageViewer overlay, Inspector tabs, OCR-only compare)
tests/      unittest suite
results/    results.json, RESULTS.md, FAILURES.md, failures.json, overlays
```
