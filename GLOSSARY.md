# Tianchi Table Recognition

Automated multimodal table recognition and question-answering system for Alibaba Cloud Tianchi Competition 532510, operating under strict Alibaba Cloud Qwen-only model compliance and anti-cheating constraints.

## Language

### Inputs & Execution

**Source Document**:
An input PDF or image file from the competition dataset that contains one or more tables referenced by competition questions.
_Avoid_: Input file, raw file, attachment

**DocumentKey**:
An opaque, filename-stripped internal identifier for a Source Document used across inference prompts and caches so original filenames never leak into model inputs.
_Avoid_: File name, doc filename, file path

**Question**:
A single evaluation item from the competition question sheet specifying a Source Document reference, question type (`extract`, `thinking`, or `structure`), question text, optional table hint, and expected answer format (`string`, `number`, `json`, or `json_array`).
_Avoid_: Query, prompt, task row

**Model Role**:
A logical inference capability (such as visual table extraction, visual verification, or text reasoning) that maps via configuration to a compliant Alibaba Cloud Qwen model.
_Avoid_: Model name, model string, hardcoded model

**Run**:
A single isolated execution of the pipeline over all Questions or a chosen subset, producing answers, audit logs, and health metrics without mutating shared document caches.
_Avoid_: Batch, job, session

**Submission**:
The final competition Excel deliverable matching the required `(id, answer)` schema, generated fully programmatically from a Run.
_Avoid_: Output file, export, result sheet

### Extraction & Verification

**Page Triage**:
The lightweight page-level indexing and candidate-page selection step for long multi-page documents that narrows high-resolution extraction to relevant pages and adjacent continuation pages.
_Avoid_: Pre-filter, page search

**PageArtifact**:
The cached stage-1 extraction output for a single page of a Source Document, containing page-level context text and zero or more extracted tables.
_Avoid_: Page JSON, page dump, page cache

**TableRegion**:
A detected table area within a page, defined by its spatial bounding box and cropped image prior to grid extraction.
_Avoid_: Table crop, box, ROI

**TableGrid**:
A structured grid representation of a single table capturing row count, column count, header row count, and cells with text, row/column spans, bounding boxes, and uncertainty markers.
_Avoid_: Parsed table, table JSON, cell array

**Answer Validator**:
The deterministic post-solving verification stage that enforces competition format rules, strict JSON syntax, bracket fidelity, and subtotal consistency before an answer is finalized.
_Avoid_: Post-processor, formatter, linter

**Health Report**:
The automated unsupervised quality and cost summary produced for a Run, recording completion rate, strict format and JSON validity, subtotal consistency, re-recognition rate, fallback rate, and token spend.
_Avoid_: Eval report, metrics dump, summary log
