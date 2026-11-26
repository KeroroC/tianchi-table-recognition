"""Tests for long PDF Page Triage (native text layer vs. low-res vision_triage fallback)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tests.fakes import FakeModelTransport
from tests.fixtures_factory import create_synthetic_scanned_pdf, create_synthetic_text_pdf
from tianchi_table_recognition.config import load_config
from tianchi_table_recognition.gateway import ModelGateway, TransportRequest, TransportResponse
from tianchi_table_recognition.models import (
    AnswerFormat,
    ModelRole,
    Question,
    QuestionType,
    SourceDocumentRef,
)
from tianchi_table_recognition.preprocessing import OriginalPageStore, inspect_pdf
from tianchi_table_recognition.triage import (
    PageIndexEntry,
    select_candidate_pages,
    triage_long_pdf,
)


def _pdf_ref(path: Path) -> SourceDocumentRef:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceDocumentRef(
        document_key=f"doc_{digest[:16]}",
        document_hash=digest,
        source_path=path,
        file_name=path.name,
        media_type="application/pdf",
    )


def test_triage_long_native_text_pdf_needs_zero_vision_calls_and_adds_continuation(
    tmp_path: Path,
) -> None:
    pages = [
        "第一章 公司基本情况介绍\n注册地址 北京",
        "第二章 董事会工作报告\n年度经营回顾",
        "第三章 重要风险提示\n市场竞争风险",
        "第四章 财务会计政策\n会计估计说明",
        "第五章 研发费用明细表\n项目 本期金额\n人员人工 1200",
        "研发费用明细表（续）\n折旧摊销 300\n合计 1500",
        "第六章 关联交易情况\n无重大关联交易",
        "第七章 重大诉讼事项\n无重大未决诉讼",
        "第八章 备查文件目录\n审计报告原件",
        "第九章 签署页\n法定代表人签字",
    ]
    pdf_path = create_synthetic_text_pdf(tmp_path / "docs" / "secret_annual_report.pdf", pages)
    doc_ref = _pdf_ref(pdf_path)
    inspection = inspect_pdf(pdf_path)

    transport = FakeModelTransport()
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="triage_native",
        ledger_path=tmp_path / "work" / "runs" / "triage_native" / "audit.jsonl",
    )
    page_store = OriginalPageStore(tmp_path / "work" / "cache")

    questions = [
        Question(
            id="q1",
            document_key=doc_ref.document_key,
            question_type=QuestionType.EXTRACT,
            question="研发费用明细表中人员人工金额是多少？",
            table_hint="研发费用明细表",
            answer_format=AnswerFormat.NUMBER,
        )
    ]

    result = triage_long_pdf(
        doc_ref=doc_ref,
        inspection=inspection,
        questions=questions,
        gateway=gateway,
        page_store=page_store,
    )

    assert result.mode == "native_text"
    assert result.page_count == 10
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 0
    assert result.candidate_pages == (4, 5)


def test_triage_long_scanned_pdf_uses_low_res_vision_triage_and_caches_index(
    tmp_path: Path,
) -> None:
    pdf_path = create_synthetic_scanned_pdf(
        tmp_path / "docs" / "scanned_secret_doc.pdf",
        [f"SCAN-{i}" for i in range(10)],
        width_pt=288.0,
        height_pt=144.0,
    )
    doc_ref = _pdf_ref(pdf_path)
    inspection = inspect_pdf(pdf_path)
    assert inspection.has_native_text_layer is False

    def _triage_handler(req: TransportRequest) -> TransportResponse:
        assert req.model_role == ModelRole.VISION_TRIAGE
        match = re.search(r"page_indices:\s*(\[[^\]]+\])", req.user_prompt)
        assert match is not None
        indices = json.loads(match.group(1))
        entries = []
        for idx in indices:
            if idx == 6:
                entries.append(
                    {
                        "page_index": 6,
                        "summary": "华东区域销售明细表 第一部分",
                        "table_titles": ["销售明细表"],
                        "has_table": True,
                        "continues_from_previous": False,
                        "continues_to_next": True,
                    }
                )
            elif idx == 7:
                entries.append(
                    {
                        "page_index": 7,
                        "summary": "销售明细表（续） 华东区域合计",
                        "table_titles": ["销售明细表（续）"],
                        "has_table": True,
                        "continues_from_previous": True,
                        "continues_to_next": False,
                    }
                )
            else:
                entries.append(
                    {
                        "page_index": idx,
                        "summary": "普通文字说明页",
                        "table_titles": [],
                        "has_table": False,
                        "continues_from_previous": False,
                        "continues_to_next": False,
                    }
                )
        return TransportResponse(
            content=json.dumps({"pages": entries}, ensure_ascii=False),
            prompt_tokens=80,
            completion_tokens=40,
            total_tokens=120,
        )

    transport = FakeModelTransport(handler=_triage_handler)
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="triage_scanned",
        ledger_path=tmp_path / "work" / "runs" / "triage_scanned" / "audit.jsonl",
    )
    cache_dir = tmp_path / "work" / "cache"
    page_store = OriginalPageStore(cache_dir)

    questions = [
        Question(
            id="q_scan",
            document_key=doc_ref.document_key,
            question_type=QuestionType.EXTRACT,
            question="销售明细表中华东区域合计是多少？",
            table_hint="销售明细表",
            answer_format=AnswerFormat.NUMBER,
        )
    ]

    result = triage_long_pdf(
        doc_ref=doc_ref,
        inspection=inspection,
        questions=questions,
        gateway=gateway,
        page_store=page_store,
        thumbnail_dpi=72,
        cache_dir=cache_dir,
        batch_size=4,
    )

    assert result.mode == "vision_triage"
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 3
    for req in transport.requests:
        for img in req.images:
            assert (img.width, img.height) == (288, 144)
    transport.assert_no_filename_leak(["scanned_secret_doc.pdf"])
    assert result.candidate_pages == (6, 7)

    # Second run reuses persisted triage index without additional VISION_TRIAGE calls
    result_cached = triage_long_pdf(
        doc_ref=doc_ref,
        inspection=inspection,
        questions=questions,
        gateway=gateway,
        page_store=page_store,
        thumbnail_dpi=72,
        cache_dir=cache_dir,
        batch_size=4,
    )
    assert result_cached == result
    assert transport.count_calls(ModelRole.VISION_TRIAGE) == 3


def test_triage_backward_continuation_pulls_preceding_table_start_page(tmp_path: Path) -> None:
    pages = [
        "目录",
        "正文概述",
        "利润分配表\n项目 金额\n2023年 600",
        "续表\n2024年 800\n合计 1400",
        "附注一",
        "附注二",
        "附注三",
        "附注四",
        "附注五",
        "结尾",
    ]
    pdf_path = create_synthetic_text_pdf(tmp_path / "docs" / "backward.pdf", pages)
    doc_ref = _pdf_ref(pdf_path)
    inspection = inspect_pdf(pdf_path)

    transport = FakeModelTransport()
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="triage_back",
        ledger_path=tmp_path / "work" / "runs" / "triage_back" / "audit.jsonl",
    )
    page_store = OriginalPageStore(tmp_path / "work" / "cache")

    questions = [
        Question(
            id="q_back",
            document_key=doc_ref.document_key,
            question_type=QuestionType.EXTRACT,
            question="2024年金额是多少？",
            table_hint=None,
            answer_format=AnswerFormat.NUMBER,
        )
    ]

    result = triage_long_pdf(
        doc_ref=doc_ref,
        inspection=inspection,
        questions=questions,
        gateway=gateway,
        page_store=page_store,
    )

    assert result.mode == "native_text"
    assert result.candidate_pages == (2, 3)


def test_triage_narrative_pages_do_not_hijack_table_candidate_selection() -> None:
    entries = [
        PageIndexEntry(page_index=0, summary="2024年度公司经营总结", has_table=False),
        PageIndexEntry(page_index=1, summary="普通说明页", has_table=False),
        PageIndexEntry(
            page_index=2,
            summary="财务数据表",
            table_titles=("财务摘要",),
            has_table=True,
        ),
    ]
    question = Question(
        id="q_nar",
        document_key="doc_0123456789abcdef",
        question_type=QuestionType.EXTRACT,
        question="2024年公司营收是多少？",
        table_hint=None,
        answer_format=AnswerFormat.NUMBER,
    )

    candidate_pages, _ = select_candidate_pages(entries, [question])
    assert candidate_pages == (2,)
