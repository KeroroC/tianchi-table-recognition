"""Tests for strict Submission gate, --allow-partial, and Excel @ text protection."""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import load_workbook

from tianchi_table_recognition.models import AnswerFormat, QuestionResult, QuestionType
from tianchi_table_recognition.submission import (
    SubmissionGateError,
    export_submission,
)


def _ok_result(qid: str, answer: str) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        document_key="doc_test",
        question_type=QuestionType.EXTRACT,
        answer_format=AnswerFormat.STRING,
        status="ok",
        answer=answer,
    )


def _failed_result(qid: str) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        document_key="doc_test",
        question_type=QuestionType.EXTRACT,
        answer_format=AnswerFormat.STRING,
        status="failed",
        error="model error",
    )


def test_strict_gate_rejects_missing_or_failed_questions_without_writing_file(
    tmp_path: Path,
) -> None:
    template_ids = ["001", "002", "003"]
    results = {
        "001": _ok_result("001", "Alice"),
        "002": _failed_result("002"),
    }
    out_path = tmp_path / "submission.xlsx"

    with pytest.raises(SubmissionGateError) as exc_info:
        export_submission(template_ids, results, out_path, allow_partial=False)

    err = exc_info.value
    assert err.missing_ids == ["003"]
    assert err.failed_ids == ["002"]
    assert not out_path.exists()


def test_strict_gate_prints_all_pending_ids_even_beyond_twenty(tmp_path: Path) -> None:
    template_ids = [f"{i:03d}" for i in range(1, 26)]
    out_path = tmp_path / "submission.xlsx"

    with pytest.raises(SubmissionGateError) as exc_info:
        export_submission(template_ids, {}, out_path, allow_partial=False)

    msg = str(exc_info.value)
    for qid in template_ids:
        assert qid in msg


def test_allow_partial_writes_partial_tagged_file_in_template_order(tmp_path: Path) -> None:
    template_ids = ["001", "002", "003"]
    results = {
        "003": _ok_result("003", "C"),
        "001": _ok_result("001", "A"),
    }
    requested = tmp_path / "sub.xlsx"

    res = export_submission(template_ids, results, requested, allow_partial=True)

    assert "partial" in res.output_path.stem
    assert res.output_path.exists()
    assert not requested.exists()
    assert res.exported_count == 2
    assert res.missing_ids == ["002"]

    wb = load_workbook(res.output_path, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        rows = list(ws.iter_rows(values_only=True))
        assert rows == [("id", "answer"), ("001", "A"), ("003", "C")]
    finally:
        wb.close()


def test_excel_at_text_protection_preserves_leading_zeros_dates_long_numbers_and_formulas(
    tmp_path: Path,
) -> None:
    template_ids = ["001", "002", "003", "004", "005"]
    results = {
        "001": _ok_result("001", "00123"),
        "002": _ok_result("002", "2024-03-01"),
        "003": _ok_result("003", "2024/03/01"),
        "004": _ok_result("004", "12345678901234567890"),
        "005": _ok_result("005", "=A1+B1"),
    }
    out_path = tmp_path / "submission.xlsx"

    res = export_submission(template_ids, results, out_path, allow_partial=False)
    assert res.output_path == out_path
    assert res.exported_count == 5

    wb = load_workbook(out_path, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        assert (ws.cell(1, 1).value, ws.cell(1, 2).value) == ("id", "answer")

        expected = [
            ("001", "00123"),
            ("002", "2024-03-01"),
            ("003", "2024/03/01"),
            ("004", "12345678901234567890"),
            ("005", "=A1+B1"),
        ]
        for row_idx, (exp_id, exp_ans) in enumerate(expected, start=2):
            id_cell = ws.cell(row_idx, 1)
            ans_cell = ws.cell(row_idx, 2)
            assert id_cell.value == exp_id
            assert isinstance(id_cell.value, str)
            assert id_cell.number_format == "@"
            assert ans_cell.value == exp_ans
            assert isinstance(ans_cell.value, str)
            assert ans_cell.data_type == "s"
            assert ans_cell.number_format == "@"
    finally:
        wb.close()
