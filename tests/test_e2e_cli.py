"""End-to-end CLI tests for the T1 minimal vertical loop."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from openpyxl import load_workbook

from tests.fakes import FakeModelTransport
from tests.fixtures_factory import (
    SyntheticQuestionRow,
    create_synthetic_image,
    create_synthetic_image_with_exif,
    create_synthetic_scanned_pdf,
    create_synthetic_text_pdf,
    write_questions_workbook,
    write_submission_template,
)
from tianchi_table_recognition.cli import main
from tianchi_table_recognition.gateway import TransportRequest, TransportResponse
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


def test_e2e_cli_validator_normalizes_all_four_formats_and_repairs_broken_json(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ds_t2"
    docs_dir = dataset_dir / "docs"
    forbidden = ["private_financial_report_t2.png"]
    create_synthetic_image(docs_dir / forbidden[0], label="FIN-T2")

    rows = [
        SyntheticQuestionRow(
            id="001",
            file_name=forbidden[0],
            question_type="extract",
            question="公告标题是什么？",
            table_hint="公告",
            answer_format="string",
        ),
        SyntheticQuestionRow(
            id="002",
            file_name=forbidden[0],
            question_type="extract",
            question="营业总收入是多少？",
            table_hint="财务",
            answer_format="number",
        ),
        SyntheticQuestionRow(
            id="003",
            file_name=forbidden[0],
            question_type="extract",
            question="同比增长率是多少？",
            table_hint="财务",
            answer_format="number",
        ),
        SyntheticQuestionRow(
            id="004",
            file_name=forbidden[0],
            question_type="extract",
            question="城市信息JSON",
            table_hint="地区",
            answer_format="json",
        ),
        SyntheticQuestionRow(
            id="005",
            file_name=forbidden[0],
            question_type="extract",
            question="明细数组JSON",
            table_hint="明细",
            answer_format="json_array",
        ),
    ]
    write_questions_workbook(dataset_dir / "tests.xlsx", rows)
    write_submission_template(
        dataset_dir / "submission_template.xlsx",
        ["001", "002", "003", "004", "005"],
    )

    stage1_payload = json.dumps(
        {
            "page_context_text": "【重要提示】公告",
            "tables": [
                {
                    "table_id": "t1",
                    "title": "财务表",
                    "row_count": 2,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": [
                        {"row": 0, "col": 0, "text": "公司", "uncertain": False},
                        {"row": 0, "col": 1, "text": "收入", "uncertain": False},
                        {
                            "row": 1,
                            "col": 0,
                            "text": "深圳市腾讯计算机系统有限公司",
                            "uncertain": False,
                        },
                        {"row": 1, "col": 1, "text": "1,234.00", "uncertain": False},
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )

    def _handler(req: TransportRequest) -> TransportResponse:
        if req.model_role == ModelRole.VISION_EXTRACT:
            return TransportResponse(
                content=stage1_payload,
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
            )
        if req.stage == "stage2_json_repair":
            return TransportResponse(
                content='[{"项":"甲"}]',
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            )
        answers = {
            "001": "答案：【重要提示】公告",
            "002": "1,234.00",
            "003": "25.0%",
            "004": '根据表格：\n```json\n{“城市”：“北京”，}\n```',
            "005": "[{项: 未闭合",
        }
        assert req.question_id is not None
        return TransportResponse(
            content=answers[req.question_id],
            prompt_tokens=60,
            completion_tokens=20,
            total_tokens=80,
        )

    work_dir = tmp_path / "work"
    out_xlsx = work_dir / "submissions" / "t2_submission.xlsx"
    transport = FakeModelTransport(handler=_handler)

    rc = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--run-id",
            "t2_run",
            "--output",
            str(out_xlsx),
        ],
        transport=transport,
    )

    assert rc == 0
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 1
    # 5 extract solves + 1 targeted stage2_json_repair for question 005
    assert transport.count_calls(ModelRole.TEXT_REASON) == 6
    repair_reqs = [r for r in transport.requests if r.stage == "stage2_json_repair"]
    assert len(repair_reqs) == 1
    assert repair_reqs[0].question_id == "005"
    transport.assert_no_filename_leak(forbidden)

    results_data = json.loads(
        (work_dir / "runs" / "t2_run" / "results.json").read_text(encoding="utf-8")
    )
    assert results_data["004"]["json_repaired"] is False
    assert results_data["005"]["json_repaired"] is True

    wb = load_workbook(out_xlsx, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        rows_out = [
            (ws.cell(r, 1).value, ws.cell(r, 2).value, ws.cell(r, 2).number_format)
            for r in range(1, ws.max_row + 1)
        ]
        assert rows_out == [
            ("id", "answer", "@"),
            ("001", "【重要提示】公告", "@"),
            ("002", "1234", "@"),
            ("003", "25%", "@"),
            ("004", '{"城市":"北京"}', "@"),
            ("005", '[{"项":"甲"}]', "@"),
        ]
    finally:
        wb.close()


def test_e2e_cli_strict_gate_blocks_validation_failures_and_allows_partial_export(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "ds_t2_fail"
    docs_dir = dataset_dir / "docs"
    create_synthetic_image(docs_dir / "doc_fail.png", label="FAIL-T2")

    rows = [
        SyntheticQuestionRow(
            id="001",
            file_name="doc_fail.png",
            question_type="extract",
            question="代码是多少？",
            table_hint=None,
            answer_format="string",
        ),
        SyntheticQuestionRow(
            id="002",
            file_name="doc_fail.png",
            question_type="extract",
            question="明细JSON",
            table_hint=None,
            answer_format="json",
        ),
        SyntheticQuestionRow(
            id="003",
            file_name="doc_fail.png",
            question_type="extract",
            question="公司全称是什么？",
            table_hint=None,
            answer_format="string",
        ),
    ]
    write_questions_workbook(dataset_dir / "tests.xlsx", rows)
    write_submission_template(dataset_dir / "submission_template.xlsx", ["001", "002", "003"])

    stage1_payload = json.dumps(
        {
            "page_context_text": "",
            "tables": [
                {
                    "table_id": "t1",
                    "title": "信息",
                    "row_count": 2,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": [
                        {"row": 0, "col": 0, "text": "代码", "uncertain": False},
                        {"row": 0, "col": 1, "text": "公司", "uncertain": False},
                        {"row": 1, "col": 0, "text": "00123", "uncertain": False},
                        {
                            "row": 1,
                            "col": 1,
                            "text": "深圳市腾讯计算机系统有限公司",
                            "uncertain": False,
                        },
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )

    transport = FakeModelTransport(
        role_responses={ModelRole.VISION_EXTRACT: stage1_payload},
        question_responses={
            "001": "00123",
            "002": "{unfixable broken json",
            "003": "深圳市 腾讯计算机系统有限公司",
        },
    )
    work_dir = tmp_path / "work"
    strict_out = work_dir / "submissions" / "strict.xlsx"

    rc_blocked = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--run-id",
            "t2_fail_run",
            "--output",
            str(strict_out),
        ],
        transport=transport,
    )
    assert rc_blocked != 0
    assert not strict_out.exists()
    err_text = capsys.readouterr().err
    assert "002" in err_text and "003" in err_text
    # At most 1 repair call for question 002 (none for 001/003)
    assert len([r for r in transport.requests if r.stage == "stage2_json_repair"]) == 1

    rc_partial = main(
        [
            "export",
            "--dataset-dir",
            str(dataset_dir),
            "--work-dir",
            str(work_dir),
            "--run-id",
            "t2_fail_run",
            "--allow-partial",
            "--output",
            str(strict_out),
        ]
    )
    assert rc_partial == 0
    partial_out = work_dir / "submissions" / "strict_partial.xlsx"
    assert partial_out.exists()

    wb = load_workbook(partial_out, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        assert list(ws.iter_rows(values_only=True)) == [
            ("id", "answer"),
            ("001", "00123"),
        ]
    finally:
        wb.close()


def test_e2e_cli_multiformat_exif_short_pdf_and_long_native_scanned_pdfs(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ds_t3"
    docs_dir = dataset_dir / "docs"
    forbidden = [
        "secret_photo_t3.jpg",
        "secret_scan_t3.webp",
        "secret_short_3p.pdf",
        "secret_long_native_10p.pdf",
        "secret_long_scanned_10p.pdf",
    ]
    jpg_path = create_synthetic_image_with_exif(
        docs_dir / forbidden[0],
        width=160,
        height=80,
        orientation=6,
        image_format="JPEG",
        label="JPG-T3",
    )
    webp_path = create_synthetic_image_with_exif(
        docs_dir / forbidden[1],
        width=200,
        height=100,
        orientation=6,
        image_format="WEBP",
        label="WEBP-T3",
    )
    jpg_bytes_before = jpg_path.read_bytes()
    webp_bytes_before = webp_path.read_bytes()

    create_synthetic_text_pdf(
        docs_dir / forbidden[2],
        [
            "第一页 公司概览\n成立日期 2020-01-01",
            "第二页 订单统计表\n区域 数量\n华北 42",
            "第三页 联系方式\n电话 010-12345678",
        ],
        width_pt=288.0,
        height_pt=144.0,
    )

    native_long_pages = [f"第{i}页 无关说明文本内容" for i in range(10)]
    native_long_pages[4] = "第五页 研发费用明细表\n项目 金额\n人员人工 1200"
    native_long_pages[5] = "研发费用明细表（续）\n折旧摊销 350\n合计 1550"
    create_synthetic_text_pdf(
        docs_dir / forbidden[3],
        native_long_pages,
        width_pt=288.0,
        height_pt=144.0,
    )

    create_synthetic_scanned_pdf(
        docs_dir / forbidden[4],
        [f"SCAN-{i}" for i in range(10)],
        width_pt=288.0,
        height_pt=144.0,
    )

    rows = [
        SyntheticQuestionRow(
            "001", forbidden[0], "extract", "照片代码是多少？", "照片表", "string"
        ),
        SyntheticQuestionRow(
            "002", forbidden[1], "extract", "WebP编号是多少？", "扫描表", "string"
        ),
        SyntheticQuestionRow(
            "003", forbidden[2], "extract", "订单统计表华北数量是多少？", "订单统计表", "number"
        ),
        SyntheticQuestionRow(
            "004",
            forbidden[3],
            "extract",
            "研发费用明细表中折旧摊销金额是多少？",
            "研发费用明细表",
            "number",
        ),
        SyntheticQuestionRow(
            "005",
            forbidden[4],
            "extract",
            "资产负债表负债合计是多少？",
            "资产负债表",
            "number",
        ),
    ]
    write_questions_workbook(dataset_dir / "tests.xlsx", rows)
    write_submission_template(
        dataset_dir / "submission_template.xlsx",
        ["001", "002", "003", "004", "005"],
    )

    def _page_json(page_idx: int, context: str, tables: list[dict[str, object]]) -> str:
        return json.dumps({"page_context_text": context, "tables": tables}, ensure_ascii=False)

    def _grid_dict(
        table_id: str,
        title: str,
        rows_data: list[list[str]],
    ) -> dict[str, object]:
        return {
            "table_id": table_id,
            "title": title,
            "row_count": len(rows_data),
            "col_count": len(rows_data[0]),
            "header_row_count": 1,
            "cells": [
                {"row": r, "col": c, "text": txt, "uncertain": False}
                for r, row in enumerate(rows_data)
                for c, txt in enumerate(row)
            ],
        }

    # Pre-compute expected document_keys by inspection? Wait: handler can infer doc by page/image!
    # Even simpler: map via stage1 prompts or unique page_index / dimensions, or read from run_meta?
    # Wait: let's track calls per (document_key, page_index) cleanly inside _handler!
    native_extracted_pages: list[int] = []
    scanned_extracted_pages: list[int] = []

    def _handler(req: TransportRequest) -> TransportResponse:
        if req.model_role == ModelRole.VISION_TRIAGE:
            match = re.search(r"page_indices:\s*(\[[^\]]+\])", req.user_prompt)
            assert match is not None
            indices = json.loads(match.group(1))
            assert all((img.width, img.height) == (288, 144) for img in req.images)
            pages_out = []
            for idx in indices:
                if idx == 6:
                    pages_out.append(
                        {
                            "page_index": 6,
                            "summary": "资产负债表 资产部分",
                            "table_titles": ["资产负债表"],
                            "has_table": True,
                            "continues_to_next": True,
                        }
                    )
                elif idx == 7:
                    pages_out.append(
                        {
                            "page_index": 7,
                            "summary": "资产负债表（续） 负债合计",
                            "table_titles": ["资产负债表（续）"],
                            "has_table": True,
                            "continues_from_previous": True,
                        }
                    )
                else:
                    pages_out.append({"page_index": idx, "summary": "正文说明", "has_table": False})
            return TransportResponse(
                content=json.dumps({"pages": pages_out}, ensure_ascii=False),
                prompt_tokens=60,
                completion_tokens=30,
                total_tokens=90,
            )

        if req.model_role == ModelRole.VISION_EXTRACT:
            img = req.images[0]
            if (img.width, img.height) == (80, 160):
                content = _page_json(
                    0,
                    "照片",
                    [_grid_dict("t_jpg", "照片表", [["键", "值"], ["照片代码", "JPG001"]])],
                )
            elif (img.width, img.height) == (100, 200):
                content = _page_json(
                    0,
                    "WebP",
                    [_grid_dict("t_webp", "扫描表", [["键", "值"], ["WebP编号", "WP002"]])],
                )
            else:
                assert (img.width, img.height) == (1200, 600)
                m = re.search(r"page_index:\s*(\d+)", req.user_prompt)
                assert m is not None
                pidx = int(m.group(1))
                if pidx in (0, 1, 2):
                    short_tables = (
                        [
                            _grid_dict(
                                f"t_short_{pidx}",
                                "订单统计表",
                                [["区域", "数量"], ["华北", "42"]],
                            )
                        ]
                        if pidx == 1
                        else []
                    )
                    content = _page_json(pidx, f"短PDF第{pidx}页", short_tables)
                elif pidx == 4:
                    native_extracted_pages.append(pidx)
                    content = _page_json(
                        4,
                        "第五页 研发费用明细表",
                        [
                            _grid_dict(
                                "t_rd_4",
                                "研发费用明细表",
                                [["项目", "金额"], ["人员人工", "1200"]],
                            )
                        ],
                    )
                elif pidx == 5:
                    native_extracted_pages.append(pidx)
                    content = _page_json(
                        5,
                        "研发费用明细表（续）",
                        [
                            _grid_dict(
                                "t_rd_5",
                                "研发费用明细表（续）",
                                [["项目", "金额"], ["折旧摊销", "350"]],
                            )
                        ],
                    )
                elif pidx == 6:
                    scanned_extracted_pages.append(pidx)
                    content = _page_json(
                        6,
                        "资产负债表",
                        [
                            _grid_dict(
                                "t_bs_6",
                                "资产负债表",
                                [["科目", "金额"], ["资产合计", "9000"]],
                            )
                        ],
                    )
                elif pidx == 7:
                    scanned_extracted_pages.append(pidx)
                    content = _page_json(
                        7,
                        "资产负债表（续）",
                        [
                            _grid_dict(
                                "t_bs_7",
                                "资产负债表（续）",
                                [["科目", "金额"], ["负债合计", "4200"]],
                            )
                        ],
                    )
                else:
                    raise AssertionError(
                        f"Unexpected high-res extraction on irrelevant page {pidx}"
                    )
            return TransportResponse(
                content=content,
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
            )

        assert req.model_role == ModelRole.TEXT_REASON
        assert req.question_id is not None
        if req.question_id in ("004", "005"):
            assert "CrossPageContinuation: true" in req.user_prompt
        if req.question_id == "004":
            assert "t_rd_4" in req.user_prompt and "t_rd_5" in req.user_prompt
        if req.question_id == "005":
            assert "t_bs_6" in req.user_prompt and "t_bs_7" in req.user_prompt

        answers = {
            "001": "JPG001",
            "002": "WP002",
            "003": "42",
            "004": "350",
            "005": "4200",
        }
        return TransportResponse(
            content=answers[req.question_id],
            prompt_tokens=60,
            completion_tokens=20,
            total_tokens=80,
        )

    work_dir = tmp_path / "work"
    out_xlsx = work_dir / "submissions" / "t3_submission.xlsx"
    transport = FakeModelTransport(handler=_handler)

    rc = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--pdf-triage-page-threshold",
            "8",
            "--pdf-render-dpi",
            "300",
            "--run-id",
            "t3_run_1",
            "--output",
            str(out_xlsx),
        ],
        transport=transport,
    )

    assert rc == 0
    # 1 JPG + 1 WebP + 3 short PDF pages + 2 native long PDF pages + 2 scanned long PDF pages = 9
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 9
    # Only the 10-page scanned PDF triggers vision_triage (3 batches: 4 + 4 + 2)
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 3
    assert transport.count_calls(ModelRole.TEXT_REASON) == 5
    assert sorted(native_extracted_pages) == [4, 5]
    assert sorted(scanned_extracted_pages) == [6, 7]

    # Source images were not mutated
    assert jpg_path.read_bytes() == jpg_bytes_before
    assert webp_path.read_bytes() == webp_bytes_before

    # Zero filename leaks in outbound requests or cache files
    transport.assert_no_filename_leak(forbidden)
    for cache_file in (work_dir / "cache").rglob("*"):
        if cache_file.is_file():
            assert not any(name in str(cache_file) for name in forbidden)
            if cache_file.suffix == ".json":
                text = cache_file.read_text(encoding="utf-8")
                assert not any(name in text for name in forbidden)

    # Verify Excel output
    wb = load_workbook(out_xlsx, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        assert [
            (ws.cell(r, 1).value, ws.cell(r, 2).value, ws.cell(r, 2).number_format)
            for r in range(1, ws.max_row + 1)
        ] == [
            ("id", "answer", "@"),
            ("001", "JPG001", "@"),
            ("002", "WP002", "@"),
            ("003", "42", "@"),
            ("004", "350", "@"),
            ("005", "4200", "@"),
        ]
    finally:
        wb.close()

    # Second run reuses Stage 1 & Triage caches -> 0 new VISION_EXTRACT / VISION_TRIAGE calls
    rc_2 = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--pdf-triage-page-threshold",
            "8",
            "--pdf-render-dpi",
            "300",
            "--run-id",
            "t3_run_2",
            "--output",
            str(work_dir / "submissions" / "t3_submission_2.xlsx"),
        ],
        transport=transport,
    )
    assert rc_2 == 0
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 9
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 3
    assert transport.count_calls(ModelRole.TEXT_REASON) == 10


def test_e2e_cli_stage2_routing_decimal_sandbox_fallback_and_structure_solver(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ds_t4"
    docs_dir = dataset_dir / "docs"
    forbidden = [
        "secret_t4_clean.png",
        "secret_t4_uncertain.png",
        "secret_t4_cross_2p.pdf",
    ]
    create_synthetic_image(docs_dir / forbidden[0], label="T4-CLEAN")
    create_synthetic_image(docs_dir / forbidden[1], label="T4-UNC")
    create_synthetic_text_pdf(
        docs_dir / forbidden[2],
        ["第一页 资产负债表\n项目 金额\n货币资金 100", "资产负债表（续）\n项目 金额\n负债合计 50"],
        width_pt=288.0,
        height_pt=144.0,
    )

    rows = [
        SyntheticQuestionRow(
            "001", forbidden[0], "extract", "公司名称是什么？", "净值表", "string"
        ),
        SyntheticQuestionRow(
            "002", forbidden[0], "thinking", "一月和二月金额之和是多少？", "净值表", "number"
        ),
        SyntheticQuestionRow(
            "003",
            forbidden[0],
            "thinking",
            "三月金额占一月二月之和的百分比是多少？",
            "净值表",
            "number",
        ),
        SyntheticQuestionRow(
            "004", forbidden[0], "thinking", "三个月总金额是多少？", "净值表", "number"
        ),
        SyntheticQuestionRow(
            "005", forbidden[0], "structure", "净值表一共有多少行？", "净值表", "number"
        ),
        SyntheticQuestionRow(
            "006", forbidden[1], "structure", "模糊表一共有多少行？", "模糊表", "number"
        ),
        SyntheticQuestionRow(
            "007", forbidden[2], "structure", "资产负债表总共有多少行？", "资产负债表", "number"
        ),
    ]
    write_questions_workbook(dataset_dir / "tests.xlsx", rows)
    write_submission_template(
        dataset_dir / "submission_template.xlsx",
        ["001", "002", "003", "004", "005", "006", "007"],
    )

    def _cells(
        rows_data: list[list[str]],
        *,
        uncertain_at: tuple[int, int] | None = None,
    ) -> list[dict[str, object]]:
        return [
            {
                "row": r,
                "col": c,
                "text": txt,
                "uncertain": uncertain_at == (r, c),
            }
            for r, row in enumerate(rows_data)
            for c, txt in enumerate(row)
        ]

    clean_stage1 = json.dumps(
        {
            "page_context_text": "基金净值报告",
            "tables": [
                {
                    "table_id": "t_clean",
                    "title": "净值表",
                    "row_count": 5,
                    "col_count": 2,
                    "header_row_count": 2,
                    "cells": _cells(
                        [
                            ["公司", "上海示例基金"],
                            ["月份", "金额"],
                            ["一月", "0.1"],
                            ["二月", "0.2"],
                            ["三月", "0.6"],
                        ]
                    ),
                }
            ],
        },
        ensure_ascii=False,
    )
    uncertain_stage1 = json.dumps(
        {
            "page_context_text": "模糊扫描件",
            "tables": [
                {
                    "table_id": "t_unc",
                    "title": "模糊表",
                    "row_count": 3,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": _cells(
                        [["项目", "值"], ["A", "1"], ["B", "2"]],
                        uncertain_at=(2, 1),
                    ),
                }
            ],
        },
        ensure_ascii=False,
    )
    pdf_p0_stage1 = json.dumps(
        {
            "page_context_text": "资产负债表",
            "tables": [
                {
                    "table_id": "t_pdf_0",
                    "title": "资产负债表",
                    "row_count": 2,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": _cells([["项目", "金额"], ["货币资金", "100"]]),
                }
            ],
        },
        ensure_ascii=False,
    )
    pdf_p1_stage1 = json.dumps(
        {
            "page_context_text": "资产负债表（续）",
            "tables": [
                {
                    "table_id": "t_pdf_1",
                    "title": "资产负债表（续）",
                    "row_count": 2,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": _cells([["项目", "金额"], ["负债合计", "50"]]),
                }
            ],
        },
        ensure_ascii=False,
    )

    seen_first_png = False
    q3_attempts = 0
    q4_attempts = 0

    def _handler(req: TransportRequest) -> TransportResponse:
        nonlocal seen_first_png, q3_attempts, q4_attempts
        if req.model_role == ModelRole.VISION_EXTRACT:
            img = req.images[0]
            if (img.width, img.height) == (1200, 600):
                m = re.search(r"page_index:\s*(\d+)", req.user_prompt)
                assert m is not None
                pidx = int(m.group(1))
                return TransportResponse(content=pdf_p0_stage1 if pidx == 0 else pdf_p1_stage1)
            if not seen_first_png:
                seen_first_png = True
                return TransportResponse(content=clean_stage1)
            return TransportResponse(content=uncertain_stage1)

        if req.model_role == ModelRole.VISION_VERIFY:
            if req.question_id == "006":
                assert len(req.images) == 1
                return TransportResponse(content="4")
            if req.question_id == "007":
                assert len(req.images) == 2
                return TransportResponse(content="3")
            raise AssertionError(f"Unexpected VISION_VERIFY question: {req.question_id}")

        assert req.model_role == ModelRole.TEXT_REASON
        if req.question_id == "001":
            return TransportResponse(content="上海示例基金")
        if req.question_id == "002":
            return TransportResponse(
                content=(
                    "```python\n"
                    "from decimal import Decimal\n"
                    "RESULT = Decimal('0.1') + Decimal('0.2')\n"
                    "```"
                )
            )
        if req.question_id == "003":
            q3_attempts += 1
            if q3_attempts == 1:
                assert req.stage == "stage2_thinking_codegen"
                return TransportResponse(content="import os\nRESULT = 200")
            assert req.stage == "stage2_thinking_codegen_repair"
            assert "ast_rejected" in req.user_prompt
            return TransportResponse(
                content=(
                    "from decimal import Decimal\n"
                    "RESULT = (Decimal('0.6') / (Decimal('0.1') + Decimal('0.2'))) * Decimal('100')"
                )
            )
        if req.question_id == "004":
            q4_attempts += 1
            if req.stage == "stage2_thinking_codegen":
                return TransportResponse(content="import subprocess\nRESULT = 1")
            if req.stage == "stage2_thinking_codegen_repair":
                return TransportResponse(content="RESULT = eval('1 + 1')")
            assert req.stage == "stage2_thinking_fallback"
            return TransportResponse(content="0.9")
        raise AssertionError(f"Unexpected TEXT_REASON request: {req}")

    work_dir = tmp_path / "work"
    out_xlsx = work_dir / "submissions" / "t4_submission.xlsx"
    transport = FakeModelTransport(handler=_handler)

    rc = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--documents-dir",
            "docs",
            "--work-dir",
            str(work_dir),
            "--run-id",
            "t4_run_1",
            "--output",
            str(out_xlsx),
        ],
        transport=transport,
    )

    assert rc == 0
    transport.assert_no_filename_leak(forbidden)
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 2

    results_data = json.loads(
        (work_dir / "runs" / "t4_run_1" / "results.json").read_text(encoding="utf-8")
    )
    assert results_data["001"]["solver_path"] == "extract"
    assert results_data["002"]["solver_path"] == "thinking_python"
    assert results_data["002"]["python_retry_count"] == 0
    assert results_data["003"]["solver_path"] == "thinking_python"
    assert results_data["003"]["python_retry_count"] == 1
    assert results_data["004"]["solver_path"] == "fallback_no_python"
    assert results_data["004"]["fallback_no_python"] is True
    assert results_data["004"]["python_retry_count"] == 1
    assert results_data["005"]["solver_path"] == "structure_deterministic"
    assert results_data["005"]["vision_verified"] is False
    assert results_data["006"]["solver_path"] == "structure_vision_verify"
    assert results_data["006"]["vision_verified"] is True
    assert results_data["007"]["solver_path"] == "structure_vision_verify"
    assert results_data["007"]["vision_verified"] is True

    run_meta = json.loads(
        (work_dir / "runs" / "t4_run_1" / "run_meta.json").read_text(encoding="utf-8")
    )
    assert run_meta["stage2_summary"]["fallback_no_python_count"] == 1
    assert run_meta["stage2_summary"]["fallback_no_python_ids"] == ["004"]
    assert run_meta["stage2_summary"]["vision_verified_count"] == 2

    wb = load_workbook(out_xlsx, data_only=False)
    try:
        ws = wb.active
        assert ws is not None
        assert [
            (ws.cell(r, 1).value, ws.cell(r, 2).value, ws.cell(r, 2).number_format)
            for r in range(1, ws.max_row + 1)
        ] == [
            ("id", "answer", "@"),
            ("001", "上海示例基金", "@"),
            ("002", "0.3", "@"),
            ("003", "200", "@"),
            ("004", "0.9", "@"),
            ("005", "5", "@"),
            ("006", "4", "@"),
            ("007", "3", "@"),
        ]
    finally:
        wb.close()
