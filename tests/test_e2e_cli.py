"""End-to-end CLI tests for the T1 minimal vertical loop."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from openpyxl import load_workbook

from tests.fakes import FakeModelTransport
from tests.fixtures_factory import (
    SyntheticQuestionRow,
    create_synthetic_image,
    write_questions_workbook,
    write_submission_template,
)
from tianchi_table_recognition.cli import main
from tianchi_table_recognition.models import ModelRole


def _build_synthetic_dataset(tmp_path: Path) -> tuple[Path, list[str]]:
    dataset_dir = tmp_path / "synthetic_dataset"
    docs_dir = dataset_dir / "docs"
    create_synthetic_image(docs_dir / "secret_balance_sheet_2024.png", label="BAL-2024")
    create_synthetic_image(docs_dir / "confidential_orders_q1.png", label="ORD-Q1")

    rows = [
        SyntheticQuestionRow(
            id="001",
            file_name="secret_balance_sheet_2024.png",
            question_type="extract",
            question="公司证券代码是多少？",
            table_hint="基本信息表",
            answer_format="string",
        ),
        SyntheticQuestionRow(
            id="002",
            file_name="secret_balance_sheet_2024.png",
            question_type="extract",
            question="报告截止日期是什么？",
            table_hint="基本信息表",
            answer_format="string",
        ),
        SyntheticQuestionRow(
            id="003",
            file_name="confidential_orders_q1.png",
            question_type="订单编号是多少？",
            question="订单编号是多少？",
            table_hint="订单明细",
            answer_format="string",
        ),
        SyntheticQuestionRow(
            id="004",
            file_name="confidential_orders_q1.png",
            question_type="extract",
            question="各地区金额JSON",
            table_hint="订单明细",
            answer_format="json",
        ),
    ]
    write_questions_workbook(dataset_dir / "tests.xlsx", rows)
    template_ids = ["001", "002", "003", "004"]
    write_submission_template(dataset_dir / "submission_template.xlsx", template_ids)
    return dataset_dir, ["secret_balance_sheet_2024.png", "confidential_orders_q1.png"]


def _make_fake_transport() -> FakeModelTransport:
    stage1_payload = json.dumps(
        {
            "page_context_text": "合成表格页面上下文",
            "tables": [
                {
                    "table_id": "t1",
                    "title": "明细表",
                    "row_count": 3,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": [
                        {"row": 0, "col": 0, "text": "字段", "uncertain": False},
                        {"row": 0, "col": 1, "text": "值", "uncertain": False},
                        {"row": 1, "col": 0, "text": "代码", "uncertain": False},
                        {"row": 1, "col": 1, "text": "00123", "uncertain": False},
                        {"row": 2, "col": 0, "text": "日期", "uncertain": False},
                        {"row": 2, "col": 1, "text": "2024-03-01", "uncertain": False},
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    return FakeModelTransport(
        role_responses={ModelRole.VISION_EXTRACT: stage1_payload},
        question_responses={
            "001": " 00123 ",
            "002": "2024-03-01",
            "003": "12345678901234567890",
            "004": '{"north":100}',
        },
    )


def test_e2e_cli_run_caches_stage1_strips_filenames_logs_ledger_and_exports_strict_xlsx(
    tmp_path: Path,
) -> None:
    dataset_dir, forbidden_filenames = _build_synthetic_dataset(tmp_path)
    work_dir = tmp_path / "work"
    out_xlsx = work_dir / "submissions" / "final_submission.xlsx"
    transport = _make_fake_transport()

    exit_code = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--run-id",
            "e2e_run_1",
            "--output",
            str(out_xlsx),
        ],
        transport=transport,
    )

    assert exit_code == 0
    # 2 unique Source Documents -> 2 vision_extract calls; 4 questions -> 4 text_reason calls
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 2
    assert transport.count_calls(ModelRole.TEXT_REASON) == 4
    transport.assert_no_filename_leak(forbidden_filenames)

    # Check run metadata recorded dirty question_type normalization warning
    run_meta = json.loads(
        (work_dir / "runs" / "e2e_run_1" / "run_meta.json").read_text(encoding="utf-8")
    )
    assert len(run_meta["normalization_warnings"]) == 1
    assert run_meta["normalization_warnings"][0]["question_id"] == "003"
    assert run_meta["normalization_warnings"][0]["normalized_question_type"] == "extract"

    # Check audit ledger entries
    ledger_lines = [
        json.loads(line)
        for line in (work_dir / "runs" / "e2e_run_1" / "audit_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(ledger_lines) == 6
    assert {entry["model_role"] for entry in ledger_lines} == {"vision_extract", "text_reason"}

    # Read back Excel Submission and verify @ pure text cells
    wb = load_workbook(out_xlsx, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        rows = [
            (ws.cell(r, 1).value, ws.cell(r, 2).value, ws.cell(r, 2).number_format)
            for r in range(1, ws.max_row + 1)
        ]
        assert rows == [
            ("id", "answer", "@"),
            ("001", "00123", "@"),
            ("002", "2024-03-01", "@"),
            ("003", "12345678901234567890", "@"),
            ("004", '{"north":100}', "@"),
        ]
    finally:
        wb.close()

    # Second run with same parameters hits Stage 1 disk cache -> 0 additional vision_extract calls
    out_xlsx_2 = work_dir / "submissions" / "final_submission_2.xlsx"
    exit_code_2 = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--run-id",
            "e2e_run_2",
            "--output",
            str(out_xlsx_2),
        ],
        transport=transport,
    )
    assert exit_code_2 == 0
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 2
    assert transport.count_calls(ModelRole.TEXT_REASON) == 8


def test_e2e_cli_ids_subset_gate_and_allow_partial_and_env_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir, _ = _build_synthetic_dataset(tmp_path)
    work_dir = tmp_path / "work"
    monkeypatch.setenv("TTR_DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("TTR_DOCUMENTS_DIR", "docs")
    monkeypatch.setenv("TTR_WORK_DIR", str(work_dir))

    transport = _make_fake_transport()
    strict_out = work_dir / "submissions" / "subset.xlsx"

    # Subset --ids without --allow-partial must refuse export and print missing IDs
    rc_blocked = main(
        [
            "run",
            "--ids",
            "001,003",
            "--run-id",
            "subset_run",
            "--output",
            str(strict_out),
        ],
        transport=transport,
    )
    assert rc_blocked != 0
    assert not strict_out.exists()
    captured = capsys.readouterr()
    assert "002" in captured.err and "004" in captured.err

    # Export existing subset run with --allow-partial succeeds and tags filename with partial
    rc_partial = main(
        [
            "export",
            "--run-id",
            "subset_run",
            "--allow-partial",
            "--output",
            str(strict_out),
        ]
    )
    assert rc_partial == 0
    partial_path = work_dir / "submissions" / "subset_partial.xlsx"
    assert partial_path.exists()

    wb = load_workbook(partial_path, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        assert list(ws.iter_rows(values_only=True)) == [
            ("id", "answer"),
            ("001", "00123"),
            ("003", "12345678901234567890"),
        ]
    finally:
        wb.close()


def test_e2e_cli_fails_fast_on_non_compliant_text_reason_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir, _ = _build_synthetic_dataset(tmp_path)
    work_dir = tmp_path / "work"
    transport = _make_fake_transport()

    rc = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--text-reason-model",
            "gpt-4o",
            "--allow-partial",
        ],
        transport=transport,
    )

    assert rc != 0
    assert transport.count_calls() == 0
    captured = capsys.readouterr()
    assert "qwen" in captured.err


def test_gitignore_blocks_secrets_caches_runs_and_submissions() -> None:
    ignored_paths = [
        ".env",
        ".env.local",
        "work/runs/r1/results.json",
        "cache/page_artifacts/abc.json",
        "runs/run1/audit_ledger.jsonl",
        "submissions/final.xlsx",
        "data/tests.xlsx",
        "sample.xlsx",
    ]
    res = subprocess.run(
        ["git", "check-ignore", *ignored_paths],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    matched = set(res.stdout.splitlines())
    assert set(ignored_paths) == matched
