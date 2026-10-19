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
