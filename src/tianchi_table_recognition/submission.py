"""Strict Submission exporter with full-run gate, --allow-partial, and Excel @ protection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from openpyxl import Workbook
from pydantic import BaseModel, ConfigDict

from tianchi_table_recognition.io_utils import atomic_write_file
from tianchi_table_recognition.models import QuestionResult
from tianchi_table_recognition.validator import validate_submission_answer


class SubmissionGateError(ValueError):
    """Raised when strict full-run Submission export finds missing or failed Questions."""

    def __init__(self, *, missing_ids: list[str], failed_ids: list[str]) -> None:
        self.missing_ids = missing_ids
        self.failed_ids = failed_ids
        retry_ids = missing_ids + [qid for qid in failed_ids if qid not in missing_ids]
        joined = ", ".join(retry_ids)
        super().__init__(
            f"Submission gate blocked export: {len(missing_ids)} missing, "
            f"{len(failed_ids)} failed. Pending/retry IDs: [{joined}]"
        )


class SubmissionExportResult(BaseModel):
    """Summary of an exported Submission workbook."""

    model_config = ConfigDict(frozen=True)

    output_path: Path
    exported_count: int
    total_template_count: int
    is_partial: bool
    missing_ids: list[str]
    failed_ids: list[str]


def _with_partial_suffix(path: Path) -> Path:
    if "partial" in path.stem.lower():
        return path
    suffix = path.suffix or ".xlsx"
    return path.with_name(f"{path.stem}_partial{suffix}")


def _write_text_cell(cell: object, value: str) -> None:
    # openpyxl resets data_type when value is assigned; assign value first, then text format (@).
    cell.value = value  # type: ignore[attr-defined]
    cell.data_type = "s"  # type: ignore[attr-defined]
    cell.number_format = "@"  # type: ignore[attr-defined]


def export_submission(
    template_ids: Sequence[str],
    results_by_id: Mapping[str, QuestionResult],
    output_path: Path,
    *,
    allow_partial: bool = False,
) -> SubmissionExportResult:
    """Export Submission Excel workbook enforcing full-run gate and @ text formatting."""
    missing_ids: list[str] = []
    failed_ids: list[str] = []
    rows_to_write: list[tuple[str, str]] = []

    for qid in template_ids:
        result = results_by_id.get(qid)
        if result is None:
            missing_ids.append(qid)
            continue
        if result.status != "ok" or result.answer is None:
            failed_ids.append(qid)
            continue
        answer_str = str(result.answer)
        if result.validation_errors or validate_submission_answer(
            result.answer_format, answer_str
        ):
            failed_ids.append(qid)
            continue
        rows_to_write.append((qid, answer_str))

    if not allow_partial and (missing_ids or failed_ids):
        raise SubmissionGateError(missing_ids=missing_ids, failed_ids=failed_ids)

    target_path = _with_partial_suffix(output_path) if allow_partial else output_path

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "submission"
    _write_text_cell(ws.cell(row=1, column=1), "id")
    _write_text_cell(ws.cell(row=1, column=2), "answer")

    for row_idx, (qid, answer) in enumerate(rows_to_write, start=2):
        _write_text_cell(ws.cell(row=row_idx, column=1), qid)
        _write_text_cell(ws.cell(row=row_idx, column=2), answer)

    try:
        atomic_write_file(target_path, wb.save, suffix=".xlsx")
    finally:
        wb.close()

    return SubmissionExportResult(
        output_path=target_path,
        exported_count=len(rows_to_write),
        total_template_count=len(template_ids),
        is_partial=allow_partial,
        missing_ids=missing_ids,
        failed_ids=failed_ids,
    )
