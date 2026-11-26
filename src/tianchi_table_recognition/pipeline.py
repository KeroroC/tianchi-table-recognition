"""End-to-end Run orchestration for Stage 1 caching, Stage 2 solving, and Submission export."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from tianchi_table_recognition.config import AppConfig
from tianchi_table_recognition.dataset import load_dataset, load_submission_template
from tianchi_table_recognition.gateway import ComplianceError, ModelGateway, ModelTransport
from tianchi_table_recognition.io_utils import atomic_write_text
from tianchi_table_recognition.models import PageArtifact, QuestionResult
from tianchi_table_recognition.stage1 import (
    PageArtifactCache,
    VisionPageExtractor,
    extract_image_document,
)
from tianchi_table_recognition.stage2 import ExtractSolver
from tianchi_table_recognition.submission import SubmissionExportResult, export_submission

_RESULTS_ADAPTER = TypeAdapter(dict[str, QuestionResult])


class RunSummary(BaseModel):
    """Summary returned after executing a pipeline Run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_dir: Path
    total_questions: int
    succeeded_count: int
    failed_count: int
    warnings_count: int
    submission_result: SubmissionExportResult | None = None


def _export_run_submission(
    config: AppConfig,
    *,
    run_id: str,
    results_by_id: dict[str, QuestionResult],
    allow_partial: bool,
    output_path: Path | None,
) -> SubmissionExportResult:
    template_ids = load_submission_template(config.submission_template_path)
    target_out = output_path or (config.work_dir / "submissions" / f"{run_id}.xlsx")
    return export_submission(
        template_ids,
        results_by_id,
        target_out,
        allow_partial=allow_partial,
    )


def _generate_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


def run_pipeline(
    config: AppConfig,
    transport: ModelTransport,
    *,
    selected_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    export: bool = True,
    allow_partial: bool = False,
    output_path: Path | None = None,
) -> RunSummary:
    """Execute Stage 1 extraction/caching, Stage 2 solving, and Submission export."""
    active_run_id = run_id or _generate_run_id()
    run_dir = config.work_dir / "runs" / active_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_dataset(
        config.questions_path,
        config.documents_dir_path,
        selected_ids=selected_ids,
    )

    # Save local Run metadata mapping DocumentKey -> Source Document (never passed to models).
    run_meta = {
        "run_id": active_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "selected_ids": list(selected_ids) if selected_ids else None,
        "question_count": len(loaded.questions),
        "normalization_warnings": [
            w.model_dump(mode="json") for w in loaded.warnings
        ],
        "document_registry": {
            key: {
                "document_hash": ref.document_hash,
                "file_name": ref.file_name,
                "media_type": ref.media_type,
            }
            for key, ref in loaded.documents.items()
        },
    }
    atomic_write_text(
        run_dir / "run_meta.json",
        json.dumps(run_meta, ensure_ascii=False, indent=2),
    )

    gateway = ModelGateway(
        config=config,
        transport=transport,
        run_id=active_run_id,
        ledger_path=run_dir / "audit_ledger.jsonl",
    )
    cache = PageArtifactCache(config.work_dir / "cache")
    extractor = VisionPageExtractor(gateway=gateway)
    solver = ExtractSolver(gateway=gateway)

    artifacts_by_doc: dict[str, PageArtifact] = {}
    stage1_errors: dict[str, str] = {}
    for doc_key, doc_ref in loaded.documents.items():
        try:
            artifacts_by_doc[doc_key] = extract_image_document(
                doc_ref=doc_ref,
                gateway=gateway,
                cache=cache,
                extractor=extractor,
            )
        except ComplianceError:
            raise
        except Exception as exc:
            stage1_errors[doc_key] = str(exc)

    results_by_id: dict[str, QuestionResult] = {}
    for question in loaded.questions:
        artifact = artifacts_by_doc.get(question.document_key)
        if artifact is None:
            results_by_id[question.id] = QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=stage1_errors.get(question.document_key, "Stage 1 extraction missing"),
            )
            continue
        results_by_id[question.id] = solver.solve(question, artifact)

    serialized_results = _RESULTS_ADAPTER.dump_json(results_by_id, indent=2).decode("utf-8")
    atomic_write_text(run_dir / "results.json", serialized_results)

    succeeded_count = sum(1 for r in results_by_id.values() if r.status == "ok")
    failed_count = len(results_by_id) - succeeded_count

    submission_result: SubmissionExportResult | None = None
    if export:
        submission_result = _export_run_submission(
            config,
            run_id=active_run_id,
            results_by_id=results_by_id,
            allow_partial=allow_partial,
            output_path=output_path,
        )

    return RunSummary(
        run_id=active_run_id,
        run_dir=run_dir,
        total_questions=len(loaded.questions),
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        warnings_count=len(loaded.warnings),
        submission_result=submission_result,
    )


def export_existing_run(
    config: AppConfig,
    *,
    run_id: str,
    allow_partial: bool = False,
    output_path: Path | None = None,
) -> SubmissionExportResult:
    """Export a Submission workbook from an already-persisted Run's results.json."""
    results_path = config.work_dir / "runs" / run_id / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Run results not found: {results_path}")

    results_by_id = _RESULTS_ADAPTER.validate_json(results_path.read_text(encoding="utf-8"))
    return _export_run_submission(
        config,
        run_id=run_id,
        results_by_id=results_by_id,
        allow_partial=allow_partial,
        output_path=output_path,
    )
