"""Tests for Stage 2 StructureSolver deterministic counting and vision_verify fallback."""

from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeModelTransport
from tests.fixtures_factory import create_synthetic_image
from tianchi_table_recognition.config import load_config
from tianchi_table_recognition.gateway import ModelGateway
from tianchi_table_recognition.models import (
    AnswerFormat,
    ModelRole,
    PageArtifact,
    Question,
    QuestionType,
    TableCell,
    TableGrid,
)
from tianchi_table_recognition.preprocessing import PreparedPage
from tianchi_table_recognition.structure import StructureSolver


def _make_gateway(tmp_path: Path, transport: FakeModelTransport) -> ModelGateway:
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    return ModelGateway(
        config=cfg,
        transport=transport,
        run_id="structure_test",
        ledger_path=tmp_path / "work" / "runs" / "structure_test" / "audit.jsonl",
    )


def _make_page_artifact(
    page_index: int,
    tables: list[TableGrid],
    *,
    context: str = "",
    doc_key: str = "doc_struct_001",
) -> PageArtifact:
    return PageArtifact(
        document_key=doc_key,
        document_hash="b" * 64,
        page_index=page_index,
        page_context_text=context,
        tables=tables,
        stage="stage1_page_extract",
        stage_version="1.0",
        model_id="qwen-vl-max",
        prompt_hash="p" * 16,
    )


def _make_grid(
    table_id: str,
    title: str | None,
    row_count: int,
    col_count: int,
    *,
    header_row_count: int = 1,
    uncertain_cell: tuple[int, int] | None = None,
    merged_span: tuple[int, int, int, int] | None = None,
) -> TableGrid:
    cells: list[TableCell] = []
    covered: set[tuple[int, int]] = set()
    if merged_span is not None:
        r0, c0, rs, cs = merged_span
        cells.append(
            TableCell(
                row=r0,
                col=c0,
                rowspan=rs,
                colspan=cs,
                text="合并表头",
                uncertain=False,
            )
        )
        for rr in range(r0, r0 + rs):
            for cc in range(c0, c0 + cs):
                covered.add((rr, cc))

    for r in range(row_count):
        for c in range(col_count):
            if (r, c) in covered:
                continue
            cells.append(
                TableCell(
                    row=r,
                    col=c,
                    text=f"r{r}c{c}",
                    uncertain=(uncertain_cell == (r, c)),
                )
            )
    return TableGrid(
        table_id=table_id,
        title=title,
        row_count=row_count,
        col_count=col_count,
        header_row_count=header_row_count,
        cells=cells,
    )


def test_structure_deterministic_counts_include_header_rows_with_zero_model_calls(
    tmp_path: Path,
) -> None:
    grid = _make_grid(
        "t1",
        "利润表",
        row_count=5,
        col_count=4,
        header_row_count=2,
        merged_span=(0, 0, 1, 2),
    )
    page = _make_page_artifact(0, [grid])
    transport = FakeModelTransport()
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))

    cases = [
        ("q_rows", "该利润表一共有多少行？", "5"),
        ("q_rows_incl", "利润表总行数（包含表头）是多少？", "5"),
        ("q_cols", "该利润表共有多少列？", "4"),
        ("q_hdr", "该利润表的表头共有几行？", "2"),
        ("q_data", "利润表不含表头有多少行？", "3"),
        ("q_merged", "该表中有多少个合并单元格？", "1"),
    ]
    for qid, q_text, expected in cases:
        res = solver.solve(
            Question(
                id=qid,
                document_key=page.document_key,
                question_type=QuestionType.STRUCTURE,
                question=q_text,
                table_hint="利润表",
                answer_format=AnswerFormat.NUMBER,
            ),
            [page],
        )
        assert res.status == "ok"
        assert res.answer == expected
        assert res.solver_path == "structure_deterministic"
        assert res.vision_verified is False

    assert transport.count_calls() == 0


def test_structure_falls_back_to_vision_verify_on_uncertain_cell(tmp_path: Path) -> None:
    img_path = create_synthetic_image(tmp_path / "p0.png", label="STRUCT-UNCERTAIN")
    prepared = PreparedPage(
        document_key="doc_struct_001",
        document_hash="b" * 64,
        page_index=0,
        image_path=img_path,
        width=64,
        height=64,
    )
    grid = _make_grid("t1", "资产表", row_count=4, col_count=3, uncertain_cell=(3, 1))
    page = _make_page_artifact(0, [grid])

    transport = FakeModelTransport(question_responses={"q_unc": "6"})
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))
    res = solver.solve(
        Question(
            id="q_unc",
            document_key=page.document_key,
            question_type=QuestionType.STRUCTURE,
            question="该表共有多少行？",
            table_hint="资产表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [page],
        preserved_pages=[prepared],
    )

    assert res.status == "ok"
    assert res.answer == "6"
    assert res.solver_path == "structure_vision_verify"
    assert res.vision_verified is True
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 1
    req = transport.requests[0]
    assert len(req.images) == 1
    assert "uncertain_cell_or_boundary" in req.user_prompt


def test_structure_falls_back_to_vision_verify_on_inconsistent_bounds(tmp_path: Path) -> None:
    img_path = create_synthetic_image(tmp_path / "p0.png", label="STRUCT-BOUNDS")
    prepared = PreparedPage(
        document_key="doc_struct_001",
        document_hash="b" * 64,
        page_index=0,
        image_path=img_path,
        width=64,
        height=64,
    )
    # row_count claims 5, but cells only span up to row 3 (rows 0..2)
    grid = TableGrid(
        table_id="t1",
        title="明细表",
        row_count=5,
        col_count=2,
        header_row_count=1,
        cells=[
            TableCell(row=0, col=0, text="项", uncertain=False),
            TableCell(row=0, col=1, text="值", uncertain=False),
            TableCell(row=2, col=0, text="A", uncertain=False),
            TableCell(row=2, col=1, text="1", uncertain=False),
        ],
    )
    page = _make_page_artifact(0, [grid])
    transport = FakeModelTransport(question_responses={"q_bnd": "5"})
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))

    res = solver.solve(
        Question(
            id="q_bnd",
            document_key=page.document_key,
            question_type=QuestionType.STRUCTURE,
            question="该表一共有多少行？",
            table_hint="明细表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [page],
        preserved_pages=[prepared],
    )

    assert res.status == "ok"
    assert res.answer == "5"
    assert res.vision_verified is True
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 1
    assert "inconsistent_cell_bounds" in transport.requests[0].user_prompt


def test_structure_falls_back_to_vision_verify_on_cross_page_continuation(
    tmp_path: Path,
) -> None:
    p0_img = create_synthetic_image(tmp_path / "p0.png", label="CONT-0")
    p1_img = create_synthetic_image(tmp_path / "p1.png", label="CONT-1")
    prepared_pages = [
        PreparedPage(
            document_key="doc_struct_001",
            document_hash="b" * 64,
            page_index=0,
            image_path=p0_img,
            width=64,
            height=64,
        ),
        PreparedPage(
            document_key="doc_struct_001",
            document_hash="b" * 64,
            page_index=1,
            image_path=p1_img,
            width=64,
            height=64,
        ),
    ]
    p0 = _make_page_artifact(0, [_make_grid("t0", "人员名单表", row_count=4, col_count=2)])
    p1 = _make_page_artifact(1, [_make_grid("t1", "人员名单表（续）", row_count=3, col_count=2)])

    transport = FakeModelTransport(question_responses={"q_cross": "6"})
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))
    res = solver.solve(
        Question(
            id="q_cross",
            document_key=p0.document_key,
            question_type=QuestionType.STRUCTURE,
            question="人员名单表一共有多少行？",
            table_hint="人员名单表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [p0, p1],
        preserved_pages=prepared_pages,
    )

    assert res.status == "ok"
    assert res.answer == "6"
    assert res.vision_verified is True
    assert res.solver_path == "structure_vision_verify"
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 1
    assert len(transport.requests[0].images) == 2
    assert "cross_page_continuation" in transport.requests[0].user_prompt


def test_structure_falls_back_to_vision_verify_for_json_or_unrecognized_question(
    tmp_path: Path,
) -> None:
    img_path = create_synthetic_image(tmp_path / "p0.png", label="STRUCT-JSON")
    prepared = PreparedPage(
        document_key="doc_struct_001",
        document_hash="b" * 64,
        page_index=0,
        image_path=img_path,
        width=64,
        height=64,
    )
    grid = _make_grid("t1", "结构表", row_count=3, col_count=3)
    page = _make_page_artifact(0, [grid])

    transport = FakeModelTransport(question_responses={"q_json": '{"rows":3,"cols":3}'})
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))
    res = solver.solve(
        Question(
            id="q_json",
            document_key=page.document_key,
            question_type=QuestionType.STRUCTURE,
            question="以JSON输出该表的行列结构",
            table_hint="结构表",
            answer_format=AnswerFormat.JSON,
        ),
        [page],
        preserved_pages=[prepared],
    )

    assert res.status == "ok"
    assert res.answer == '{"rows":3,"cols":3}'
    assert res.vision_verified is True
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 1


def test_structure_fallback_fails_cleanly_without_calling_vision_when_images_missing(
    tmp_path: Path,
) -> None:
    grid = _make_grid("t1", "资产表", row_count=4, col_count=3, uncertain_cell=(1, 1))
    page = _make_page_artifact(0, [grid])

    transport = FakeModelTransport()
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))
    res = solver.solve(
        Question(
            id="q_no_img",
            document_key=page.document_key,
            question_type=QuestionType.STRUCTURE,
            question="该表共有多少行？",
            table_hint="资产表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [page],
        preserved_pages=(),
    )

    assert res.status == "failed"
    assert res.vision_verified is False
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 0


def test_structure_falls_back_when_middle_rows_are_missing_in_cells(tmp_path: Path) -> None:
    img_path = create_synthetic_image(tmp_path / "p0.png", label="STRUCT-GAPS")
    prepared = PreparedPage(
        document_key="doc_struct_001",
        document_hash="b" * 64,
        page_index=0,
        image_path=img_path,
        width=64,
        height=64,
    )
    # Row 1 is missing even though max row reaches 3 (row_count=3)
    grid = TableGrid(
        table_id="t1",
        title="明细表",
        row_count=3,
        col_count=2,
        header_row_count=1,
        cells=[
            TableCell(row=0, col=0, text="项", uncertain=False),
            TableCell(row=0, col=1, text="值", uncertain=False),
            TableCell(row=2, col=0, text="B", uncertain=False),
            TableCell(row=2, col=1, text="2", uncertain=False),
        ],
    )
    page = _make_page_artifact(0, [grid])
    transport = FakeModelTransport(question_responses={"q_gap": "3"})
    solver = StructureSolver(gateway=_make_gateway(tmp_path, transport))

    res = solver.solve(
        Question(
            id="q_gap",
            document_key=page.document_key,
            question_type=QuestionType.STRUCTURE,
            question="该表一共有多少行？",
            table_hint="明细表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [page],
        preserved_pages=[prepared],
    )

    assert res.status == "ok"
    assert res.answer == "3"
    assert res.vision_verified is True
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 1
    assert "inconsistent_cell_bounds" in transport.requests[0].user_prompt
