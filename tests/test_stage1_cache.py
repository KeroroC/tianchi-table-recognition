"""Tests for Stage 1 single-image extraction and atomic PageArtifact caching."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tests.fakes import FakeModelTransport
from tests.fixtures_factory import (
    SyntheticQuestionRow,
    create_synthetic_image,
    write_questions_workbook,
)
from tianchi_table_recognition.config import CLIOverrides, load_config
from tianchi_table_recognition.dataset import load_dataset
from tianchi_table_recognition.gateway import ModelGateway
from tianchi_table_recognition.models import ModelRole
from tianchi_table_recognition.stage1 import (
    PageArtifactCache,
    VisionPageExtractor,
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
