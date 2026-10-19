"""Render results/results.json -> results/RESULTS.md (numbers are never hand-typed)."""
import json, sys, os

def main(d="results"):
    r = json.load(open(os.path.join(d, "results.json")))
    conds = [c for c in ("clean", "noise", "blur", "lowcontrast", "skew", "borderless") if c in r]
    L = ["| Condition | Pages | Structured macro-F1 | OCR-only macro-F1 | Mean IoU | Table cell acc. | Form KV F1 | Reading order | Page CER (struct) | Page CER (OCR-only) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        s, b = r[c]["structured"], r[c]["ocr_only"]
        L.append(f"| {c} | {r[c]['pages']} | **{s['macro_f1']}** | {b['macro_f1']} | {s['mean_iou']} | {s['table']['cell_accuracy']} | "
                 f"{s['form_kv']['f1']} | {s['reading_order']} | {s['page_cer']} | {b['page_cer']} |")
    L += ["", "**Per-class F1 (structured pipeline)**", "", "| Condition | " + " | ".join(r[conds[0]]["structured"]["classes"]) + " |",
          "|---|" + "---|" * len(r[conds[0]]["structured"]["classes"])]
    for c in conds:
        L.append(f"| {c} | " + " | ".join(str(v["f1"]) for v in r[c]["structured"]["classes"].values()) + " |")
    L += ["", f"Runtime (1 CPU core, Tesseract): structured {r['timing_sec_per_page']['structured']} s/page, "
              f"OCR-only {r['timing_sec_per_page']['ocr_only']} s/page."]
    open(os.path.join(d, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main(*(sys.argv[1:] or ["results"]))
