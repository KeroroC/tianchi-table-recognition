"""Tests for Stage 2 TableLocator and cross-page continuation context assembly."""

from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeModelTransport
from tianchi_table_recognition.config import load_config
from tianchi_table_recognition.gateway import ModelGateway
from tianchi_table_recognition.locator import TableLocator
from tianchi_table_recognition.models import (
    AnswerFormat,
    PageArtifact,
    Question,
    QuestionType,
    TableCell,
    TableGrid,
)
from tianchi_table_recognition.stage2 import ExtractSolver


def _make_page(
    page_index: int,
    tables: list[TableGrid],
    *,
    context: str = "",
    doc_key: str = "doc_1234567890abcdef",
) -> PageArtifact:
    return PageArtifact(
        document_key=doc_key,
        document_hash="a" * 64,
        page_index=page_index,
        page_context_text=context,
        tables=tables,
        stage="stage1_page_extract",
        stage_version="1.0",
        model_id="qwen-vl-max",
        prompt_hash="p" * 16,
    )


def _grid(
    table_id: str,
    title: str | None,
    rows: list[list[str]],
    *,
    header_row_count: int = 1,
) -> TableGrid:
    cells: list[TableCell] = []
    col_count = max((len(r) for r in rows), default=0)
    for r_idx, row in enumerate(rows):
        for c_idx, text in enumerate(row):
            cells.append(TableCell(row=r_idx, col=c_idx, text=text, uncertain=False))
    return TableGrid(
        table_id=table_id,
        title=title,
        row_count=len(rows),
        col_count=col_count,
        header_row_count=header_row_count,
        cells=cells,
    )


def test_locator_filters_multi_table_page_by_hint_and_question() -> None:
    t1 = _grid("t1", "基本信息表", [["字段", "值"], ["代码", "000001"]])
    t2 = _grid("t2", "研发费用明细表", [["项目", "金额"], ["人员人工", "1200"]])
    page = _make_page(0, [t1, t2], context="年度财务报告")

    q = Question(
        id="q1",
        document_key=page.document_key,
        question_type=QuestionType.EXTRACT,
        question="研发费用人员人工金额是多少？",
        table_hint="研发费用明细表",
        answer_format=AnswerFormat.NUMBER,
    )
    located = TableLocator().locate(q, [page])

    assert located.has_cross_page_continuation is False
    assert [lt.table.table_id for lt in located.tables] == ["t2"]
    assert [p.page_index for p in located.pages] == [0]


def test_locator_links_cross_page_continuation_without_merging_stage1_tables() -> None:
    p0 = _make_page(0, [_grid("t0", "股东情况表", [["股东", "比例"], ["甲", "51%"]])])
    t_start = _grid("t1", "营业收入明细表", [["区域", "金额"], ["华东", "500"]])
    t_cont = _grid("t2", "营业收入明细表（续）", [["区域", "金额"], ["华南", "300"]])
    p1 = _make_page(1, [t_start], context="分区域收入")
    p2 = _make_page(2, [t_cont], context="营业收入明细表（续）")

    q = Question(
        id="q2",
        document_key=p1.document_key,
        question_type=QuestionType.EXTRACT,
        question="华南区域金额是多少？",
        table_hint="营业收入明细表",
        answer_format=AnswerFormat.NUMBER,
    )
    located = TableLocator().locate(q, [p2, p0, p1])

    assert located.has_cross_page_continuation is True
    assert [p.page_index for p in located.pages] == [1, 2]
    assert [(lt.page_index, lt.table.table_id, lt.is_continuation) for lt in located.tables] == [
        (1, "t1", False),
        (2, "t2", True),
    ]
    # ADR-0004: Stage 1 PageArtifacts remain strictly page-atomic (unmerged)
    assert len(p1.tables) == 1 and p1.tables[0].row_count == 2
    assert len(p2.tables) == 1 and p2.tables[0].row_count == 2


def test_locator_links_three_page_headless_and_marked_chain_from_tail_hit() -> None:
    p0 = _make_page(
        0,
        [_grid("t0", "现金流量表", [["项目", "2024", "2023"], ["销售收现", "900", "800"]])],
    )
    p1 = _make_page(
        1,
        [_grid("t1", None, [["经营活动现金流净额", "300", "250"]], header_row_count=0)],
    )
    p2 = _make_page(
        2,
        [
            _grid(
                "t2",
                "续表",
                [["项目", "2024", "2023"], ["投资活动现金流净额", "-120", "-80"]],
                header_row_count=1,
            )
        ],
    )

    q = Question(
        id="q3",
        document_key=p0.document_key,
        question_type=QuestionType.EXTRACT,
        question="2024年投资活动现金流净额是多少？",
        table_hint=None,
        answer_format=AnswerFormat.NUMBER,
    )
    located = TableLocator().locate(q, [p0, p1, p2])

    assert located.has_cross_page_continuation is True
    assert [p.page_index for p in located.pages] == [0, 1, 2]
    assert [lt.table.table_id for lt in located.tables] == ["t0", "t1", "t2"]


def test_locator_links_repeated_headers_and_stops_before_distinct_table() -> None:
    p0 = _make_page(
        0,
        [_grid("t0", "主营业务构成表", [["业务", "收入", "毛利率"], ["软件", "1000", "60%"]])],
    )
    p1 = _make_page(
        1,
        [_grid("t1", None, [["业务", "收入", "毛利率"], ["硬件", "500", "20%"]])],
    )
    p2 = _make_page(
        2,
        [_grid("t2", "员工人数统计表", [["类别", "人数", "占比"], ["研发", "120", "40%"]])],
    )

    q = Question(
        id="q4",
        document_key=p0.document_key,
        question_type=QuestionType.EXTRACT,
        question="硬件毛利率是多少？",
        table_hint="主营业务构成表",
        answer_format=AnswerFormat.NUMBER,
    )
    located = TableLocator().locate(q, [p0, p1, p2])

    assert located.has_cross_page_continuation is True
    assert [p.page_index for p in located.pages] == [0, 1]
    assert [lt.table.table_id for lt in located.tables] == ["t0", "t1"]


def test_extract_solver_passes_joint_continuation_context_and_validates_string(
    tmp_path: Path,
) -> None:
    p0 = _make_page(
        0,
        [_grid("t0", "分支机构表", [["序号", "名称"], ["1", "【华东总部】"]])],
    )
    p1 = _make_page(
        1,
        [_grid("t1", "分支机构表（续）", [["序号", "名称"], ["2", "【华南特区】分公司"]])],
    )
    q = Question(
        id="q_str",
        document_key=p0.document_key,
        question_type=QuestionType.EXTRACT,
        question="序号2的分公司名称是什么？",
        table_hint="分支机构表",
        answer_format=AnswerFormat.STRING,
    )

    transport = FakeModelTransport(question_responses={"q_str": "【华南特区】分公司"})
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="solver_loc",
        ledger_path=tmp_path / "work" / "runs" / "solver_loc" / "audit.jsonl",
    )
    solver = ExtractSolver(gateway=gateway)

    res = solver.solve(q, [p0, p1])

    assert res.status == "ok"
    assert res.answer == "【华南特区】分公司"
    prompt = transport.requests[0].user_prompt
    assert "t0" in prompt and "t1" in prompt
    assert "CrossPageContinuation: true" in prompt


def test_locator_links_numbered_continuation_title_and_rejects_mismatched_number(
    tmp_path: Path,
) -> None:
    p0 = _make_page(
        0,
        [_grid("t0", "表1 销售明细表", [["区域", "金额"], ["华东", "100"]])],
    )
    p1 = _make_page(
        1,
        [_grid("t1", "续表1 销售明细表", [["区域", "金额"], ["华南", "200"]])],
    )
    p2 = _make_page(
        2,
        [_grid("t2", "续表2 员工统计表", [["部门", "人数"], ["研发", "50"]])],
    )
    q = Question(
        id="q_num",
        document_key=p0.document_key,
        question_type=QuestionType.EXTRACT,
        question="华南金额是多少？",
        table_hint="表1 销售明细表",
        answer_format=AnswerFormat.NUMBER,
    )

    located = TableLocator().locate(q, [p0, p1, p2])
    assert located.has_cross_page_continuation is True
    assert [lt.table.table_id for lt in located.tables] == ["t0", "t1"]
