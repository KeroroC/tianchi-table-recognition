# Two-stage hybrid pipeline with cached table extraction

908 competition questions span only 92 source documents (~10 questions per document), and 255 questions (`thinking` and `structure`) depend on exact row/column structure and deterministic arithmetic. We decided to use a two-stage hybrid pipeline—extracting reusable structured table representations per document first, then routing question answering across structured-table reasoning, deterministic Python computation, and targeted visual re-recognition—rather than single-step end-to-end VLM inference per question or OCR-only extraction.

## Considered Options

- **Two-stage hybrid pipeline (chosen)**: Amortizes visual extraction cost across questions on the same document, decouples perception from deterministic math/structure counting, and provides clean seams for six-layer optimizations.
- **Direct end-to-end VLM per question**: Simpler initially, but repeats costly visual parsing ~10x per document and suffers from VLM arithmetic and row-counting hallucinations.
- **OCR-only + text LLM**: Loses visual layout, merged-cell boundaries, and degraded-image context needed for complex tables.
