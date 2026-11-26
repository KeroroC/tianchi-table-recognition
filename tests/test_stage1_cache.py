"""Tests for Stage 1 single-image extraction and atomic PageArtifact caching."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from tests.fakes import FakeModelTransport
from tests.fixtures_factory import (
    SyntheticQuestionRow,
    create_synthetic_image,
    create_synthetic_image_with_exif,
    create_synthetic_scanned_pdf,
    create_synthetic_text_pdf,
    write_questions_workbook,
)
from tianchi_table_recognition.config import CLIOverrides, load_config
from tianchi_table_recognition.dataset import load_dataset
from tianchi_table_recognition.gateway import ModelGateway, TransportRequest, TransportResponse
from tianchi_table_recognition.models import ModelRole, Question
from tianchi_table_recognition.stage1 import (
    PageArtifactCache,
    VisionPageExtractor,
    extract_document,
    extract_image_document,
)


def _sample_extraction_json() -> str:
    return json.dumps(
        {
            "page_context_text": "2024年度财务摘要表",
            "tables": [
                {
                    "table_id": "t1",
                    "title": "收入明细",
                    "row_count": 2,
                    "col_count": 2,
                    "header_row_count": 1,
                    "cells": [
                        {"row": 0, "col": 0, "text": "项目", "uncertain": False},
                        {"row": 0, "col": 1, "text": "金额", "uncertain": False},
                        {"row": 1, "col": 0, "text": "主营收入", "uncertain": False},
                        {
                            "row": 1,
                            "col": 1,
                            "text": None,
                            "bbox": [100, 120, 220, 150],
                            "uncertain": True,
                        },
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )


def test_stage1_extracts_page_artifact_caches_atomically_and_strips_filename(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    img_path = docs_dir / "confidential_revenue_statement.png"
    create_synthetic_image(img_path, width=200, height=100, label="REV")
    # Add EXIF orientation 6 (rotate 90 CW) to verify orientation correction
    raw_img = Image.open(img_path)
    exif = raw_img.getexif()
    exif[274] = 6
    raw_img.save(img_path, exif=exif)

    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [
            SyntheticQuestionRow(
                id="1",
                file_name="confidential_revenue_statement.png",
                question_type="extract",
                question="主营收入金额是多少？",
                table_hint="收入明细",
                answer_format="number",
            )
        ],
    )
    loaded = load_dataset(q_path, docs_dir)
    doc_ref = next(iter(loaded.documents.values()))

    transport = FakeModelTransport(
        role_responses={ModelRole.VISION_EXTRACT: _sample_extraction_json()}
    )
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="run1",
        ledger_path=tmp_path / "work" / "runs" / "run1" / "audit_ledger.jsonl",
    )
    cache = PageArtifactCache(tmp_path / "work" / "cache")
    extractor = VisionPageExtractor(gateway=gateway)

    artifact = extract_image_document(
        doc_ref=doc_ref,
        gateway=gateway,
        cache=cache,
        extractor=extractor,
    )

    assert artifact.document_key == doc_ref.document_key
    assert artifact.page_context_text == "2024年度财务摘要表"
    assert len(artifact.tables) == 1
    assert artifact.tables[0].row_count == 2
    assert artifact.tables[0].cells[3].uncertain is True
    assert artifact.tables[0].cells[3].text is None
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 1
    # Transposed EXIF orientation 6 swaps width=200,height=100 -> width=100,height=200
    sent_img = transport.requests[0].images[0]
    assert sent_img.width == 100
    assert sent_img.height == 200
    transport.assert_no_filename_leak(["confidential_revenue_statement.png"])

    # Cached JSON on disk must not contain raw filename
    cache_files = list((tmp_path / "work" / "cache").rglob("*.json"))
    assert len(cache_files) == 1
    assert "confidential_revenue_statement" not in cache_files[0].read_text(encoding="utf-8")

    # Second run with identical model & prompt hits disk cache with zero new vision calls
    artifact_cached = extract_image_document(
        doc_ref=doc_ref,
        gateway=gateway,
        cache=cache,
        extractor=extractor,
    )
    assert artifact_cached == artifact
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 1


def test_stage1_cache_invalidates_on_model_or_prompt_change(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    create_synthetic_image(docs_dir / "doc.png", label="D1")
    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [SyntheticQuestionRow("1", "doc.png", "extract", "Q?", None, "string")],
    )
    doc_ref = next(iter(load_dataset(q_path, docs_dir).documents.values()))
    cache = PageArtifactCache(tmp_path / "work" / "cache")
    transport = FakeModelTransport(
        role_responses={ModelRole.VISION_EXTRACT: _sample_extraction_json()}
    )

    cfg1 = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gw1 = ModelGateway(
        config=cfg1,
        transport=transport,
        run_id="r1",
        ledger_path=tmp_path / "work" / "runs" / "r1" / "audit_ledger.jsonl",
    )
    ext1 = VisionPageExtractor(gateway=gw1, prompt_template="Extract tables v1.")
    extract_image_document(doc_ref=doc_ref, gateway=gw1, cache=cache, extractor=ext1)
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 1

    # Changing model_id invalidates cache
    cfg2 = load_config(
        env={"TTR_WORK_DIR": str(tmp_path / "work")},
        cli_overrides=CLIOverrides(vision_extract_model="qwen-vl-plus"),
    )
    gw2 = ModelGateway(
        config=cfg2,
        transport=transport,
        run_id="r2",
        ledger_path=tmp_path / "work" / "runs" / "r2" / "audit_ledger.jsonl",
    )
    ext2 = VisionPageExtractor(gateway=gw2, prompt_template="Extract tables v1.")
    extract_image_document(doc_ref=doc_ref, gateway=gw2, cache=cache, extractor=ext2)
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 2

    # Changing prompt_template invalidates cache
    ext3 = VisionPageExtractor(gateway=gw2, prompt_template="Extract tables v2 with details.")
    extract_image_document(doc_ref=doc_ref, gateway=gw2, cache=cache, extractor=ext3)
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 3

    cache_files = list((tmp_path / "work" / "cache").rglob("*.json"))
    assert len(cache_files) == 3


def test_stage1_extract_document_handles_jpg_webp_and_short_pdf_atomically(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    create_synthetic_image_with_exif(
        docs_dir / "photo.jpg", width=180, height=90, orientation=6, image_format="JPEG"
    )
    create_synthetic_image_with_exif(
        docs_dir / "scan.webp", width=220, height=110, orientation=6, image_format="WEBP"
    )
    create_synthetic_text_pdf(
        docs_dir / "short_three_pages.pdf",
        ["第一页 摘要", "第二页 营收表\n营收 100", "第三页 利润表\n利润 50"],
        width_pt=288.0,
        height_pt=144.0,
    )
    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [
            SyntheticQuestionRow("1", "photo.jpg", "extract", "Q1?", None, "string"),
            SyntheticQuestionRow("2", "scan.webp", "extract", "Q2?", None, "string"),
            SyntheticQuestionRow(
                "3", "short_three_pages.pdf", "extract", "营收?", "营收表", "number"
            ),
        ],
    )
    loaded = load_dataset(q_path, docs_dir)
    transport = FakeModelTransport(
        role_responses={ModelRole.VISION_EXTRACT: _sample_extraction_json()}
    )
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="r_short",
        ledger_path=tmp_path / "work" / "runs" / "r_short" / "audit.jsonl",
    )
    cache = PageArtifactCache(tmp_path / "work" / "cache")

    by_name = {ref.file_name: ref for ref in loaded.documents.values()}
    jpg_res = extract_document(doc_ref=by_name["photo.jpg"], gateway=gateway, cache=cache)
    webp_res = extract_document(doc_ref=by_name["scan.webp"], gateway=gateway, cache=cache)
    pdf_res = extract_document(
        doc_ref=by_name["short_three_pages.pdf"],
        gateway=gateway,
        cache=cache,
        pdf_triage_page_threshold=8,
    )

    assert jpg_res.triage_mode == "single_image"
    assert (jpg_res.preserved_pages[0].width, jpg_res.preserved_pages[0].height) == (90, 180)
    assert webp_res.triage_mode == "single_image"
    assert (webp_res.preserved_pages[0].width, webp_res.preserved_pages[0].height) == (110, 220)

    assert pdf_res.triage_mode == "short_document"
    assert pdf_res.page_count == 3
    assert pdf_res.candidate_pages == (0, 1, 2)
    assert [a.page_index for a in pdf_res.artifacts] == [0, 1, 2]
    assert all((p.width, p.height) == (1200, 600) for p in pdf_res.preserved_pages)
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 0
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 5

    # Re-running short PDF hits page-level cache with zero new calls
    pdf_cached = extract_document(
        doc_ref=by_name["short_three_pages.pdf"],
        gateway=gateway,
        cache=cache,
        pdf_triage_page_threshold=8,
    )
    assert pdf_cached.artifacts == pdf_res.artifacts
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 5


def test_stage1_extract_document_long_native_and_scanned_pdfs_only_extract_candidates(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    native_pages = [f"第{i}页 无关正文内容说明" for i in range(10)]
    native_pages[3] = "研发费用明细表\n项目 金额\n人员人工 500"
    native_pages[4] = "研发费用明细表（续）\n折旧费 200"
    create_synthetic_text_pdf(docs_dir / "long_native.pdf", native_pages)
    create_synthetic_scanned_pdf(
        docs_dir / "long_scanned.pdf", [f"SCAN-{i}" for i in range(10)]
    )

    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [
            SyntheticQuestionRow(
                "1",
                "long_native.pdf",
                "extract",
                "研发费用人员人工是多少？",
                "研发费用明细表",
                "number",
            ),
            SyntheticQuestionRow(
                "2",
                "long_scanned.pdf",
                "extract",
                "资产负债表货币资金是多少？",
                "资产负债表",
                "number",
            ),
        ],
    )
    loaded = load_dataset(q_path, docs_dir)
    q_by_doc: dict[str, list[Question]] = {}
    for q in loaded.questions:
        q_by_doc.setdefault(q.document_key, []).append(q)

    def _handler(req: TransportRequest) -> TransportResponse:
        if req.model_role == ModelRole.VISION_TRIAGE:
            match = re.search(r"page_indices:\s*(\[[^\]]+\])", req.user_prompt)
            assert match is not None
            indices = json.loads(match.group(1))
            pages_json = []
            for idx in indices:
                if idx == 6:
                    pages_json.append(
                        {
                            "page_index": 6,
                            "summary": "资产负债表 货币资金",
                            "table_titles": ["资产负债表"],
                            "has_table": True,
                            "continues_to_next": True,
                        }
                    )
                elif idx == 7:
                    pages_json.append(
                        {
                            "page_index": 7,
                            "summary": "资产负债表（续） 流动负债",
                            "table_titles": ["资产负债表（续）"],
                            "has_table": True,
                            "continues_from_previous": True,
                        }
                    )
                else:
                    pages_json.append({"page_index": idx, "summary": "正文", "has_table": False})
            return TransportResponse(
                content=json.dumps({"pages": pages_json}, ensure_ascii=False),
                prompt_tokens=60,
                completion_tokens=30,
                total_tokens=90,
            )
        return TransportResponse(
            content=_sample_extraction_json(),
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
        )

    transport = FakeModelTransport(handler=_handler)
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="r_long",
        ledger_path=tmp_path / "work" / "runs" / "r_long" / "audit.jsonl",
    )
    cache = PageArtifactCache(tmp_path / "work" / "cache")
    by_name = {ref.file_name: ref for ref in loaded.documents.values()}

    native_ref = by_name["long_native.pdf"]
    native_res = extract_document(
        doc_ref=native_ref,
        gateway=gateway,
        cache=cache,
        questions=q_by_doc[native_ref.document_key],
        pdf_triage_page_threshold=8,
    )
    assert native_res.triage_mode == "native_text"
    assert native_res.candidate_pages == (3, 4)
    assert [a.page_index for a in native_res.artifacts] == [3, 4]
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 0
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 2

    scanned_ref = by_name["long_scanned.pdf"]
    scanned_res = extract_document(
        doc_ref=scanned_ref,
        gateway=gateway,
        cache=cache,
        questions=q_by_doc[scanned_ref.document_key],
        pdf_triage_page_threshold=8,
    )
    assert scanned_res.triage_mode == "vision_triage"
    assert scanned_res.candidate_pages == (6, 7)
    assert [a.page_index for a in scanned_res.artifacts] == [6, 7]
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 3
    assert transport.count_calls(ModelRole.VISION_EXTRACT) == 4
    extract_reqs = [r for r in transport.requests if r.model_role == ModelRole.VISION_EXTRACT]
    assert all((r.images[0].width, r.images[0].height) == (1200, 600) for r in extract_reqs)
    transport.assert_no_filename_leak(["long_native.pdf", "long_scanned.pdf"])
