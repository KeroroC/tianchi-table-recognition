"""Tests for dataset loading, dirty question_type normalization, and DocumentKey."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.fixtures_factory import (
    SyntheticQuestionRow,
    create_synthetic_image,
    write_questions_workbook,
    write_submission_template,
)
from tianchi_table_recognition.dataset import (
    DatasetSchemaError,
    load_dataset,
    load_submission_template,
)
from tianchi_table_recognition.models import AnswerFormat, QuestionType


def test_load_dataset_parses_columns_and_strips_filenames_from_questions(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    create_synthetic_image(docs_dir / "secret_financial_2024.png", label="FIN-2024")
    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [
            SyntheticQuestionRow(
                id="001",
                file_name="secret_financial_2024.png",
                question_type="extract",
                question="2024年营业收入是多少？",
                table_hint="表1 利润表",
                answer_format="number",
            ),
            SyntheticQuestionRow(
                id="002",
                file_name="secret_financial_2024.png",
                question_type="extract",
                question="报告主体名称是什么？",
                table_hint=None,
                answer_format="string",
            ),
        ],
    )

    loaded = load_dataset(q_path, docs_dir)

    assert len(loaded.questions) == 2
    assert loaded.warnings == []
    q1, q2 = loaded.questions
    assert q1.id == "001"
    assert q1.question_type == QuestionType.EXTRACT
    assert q1.table_hint == "表1 利润表"
    assert q1.answer_format == AnswerFormat.NUMBER
    assert q2.table_hint is None
    # Same Source Document yields identical DocumentKey and zero filename leakage
    assert q1.document_key == q2.document_key
    assert "secret_financial_2024" not in q1.document_key
    assert "secret_financial_2024" not in q1.model_dump_json()
    assert not hasattr(q1, "file_name")


def test_deterministic_document_key_depends_on_content_not_filename(tmp_path: Path) -> None:
    docs_a = tmp_path / "docs_a"
    docs_b = tmp_path / "docs_b"
    img_a = create_synthetic_image(docs_a / "revenue_confidential_q4.png", label="SAME")
    docs_b.mkdir(parents=True, exist_ok=True)
    img_b = docs_b / "completely_different_name.png"
    shutil.copyfile(img_a, img_b)

    qa = write_questions_workbook(
        tmp_path / "qa.xlsx",
        [
            SyntheticQuestionRow("1", img_a.name, "extract", "Q?", None, "string"),
        ],
    )
    qb = write_questions_workbook(
        tmp_path / "qb.xlsx",
        [
            SyntheticQuestionRow("1", img_b.name, "extract", "Q?", None, "string"),
        ],
    )

    loaded_a = load_dataset(qa, docs_a)
    loaded_b = load_dataset(qb, docs_b)

    assert loaded_a.questions[0].document_key == loaded_b.questions[0].document_key
    assert (
        loaded_a.documents[loaded_a.questions[0].document_key].document_hash
        == loaded_b.documents[loaded_b.questions[0].document_key].document_hash
    )


def test_normalizes_dirty_question_type_via_general_rules_without_hardcoding(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    create_synthetic_image(docs_dir / "doc1.png", label="D1")
    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [
            SyntheticQuestionRow(
                id="arbitrary_id_99",
                file_name="doc1.png",
                question_type="该表格2023年的营业收入是多少？",
                question="该表格2023年的营业收入是多少？",
                table_hint=None,
                answer_format="number",
            ),
            SyntheticQuestionRow(
                id="arbitrary_id_100",
                file_name="doc1.png",
                question_type="请统计该表格一共有多少行多少列？",
                question="请统计该表格一共有多少行多少列？",
                table_hint=None,
                answer_format="json",
            ),
            SyntheticQuestionRow(
                id="arbitrary_id_101",
                file_name="doc1.png",
                question_type="请计算三年销售额的总和与同比增速",
                question="请计算三年销售额的总和与同比增速",
                table_hint=None,
                answer_format="number",
            ),
        ],
    )

    loaded = load_dataset(q_path, docs_dir)

    assert [q.question_type for q in loaded.questions] == [
        QuestionType.EXTRACT,
        QuestionType.STRUCTURE,
        QuestionType.THINKING,
    ]
    assert len(loaded.warnings) == 3
    assert loaded.warnings[0].question_id == "arbitrary_id_99"
    assert loaded.warnings[0].normalized_question_type == QuestionType.EXTRACT


def test_fails_fast_on_missing_required_column(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    create_synthetic_image(docs_dir / "doc1.png")
    q_path = write_questions_workbook(
        tmp_path / "bad.xlsx",
        [SyntheticQuestionRow("1", "doc1.png", "extract", "Q?", None, "string")],
        headers=["id", "file_name", "question_type", "question", "table_hint"],
    )

    with pytest.raises(DatasetSchemaError, match="answer_format"):
        load_dataset(q_path, docs_dir)


def test_selected_ids_filters_preserving_order_and_rejects_unknown_id(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    create_synthetic_image(docs_dir / "doc1.png")
    q_path = write_questions_workbook(
        tmp_path / "tests.xlsx",
        [
            SyntheticQuestionRow("10", "doc1.png", "extract", "Q10", None, "string"),
            SyntheticQuestionRow("20", "doc1.png", "extract", "Q20", None, "string"),
            SyntheticQuestionRow("30", "doc1.png", "extract", "Q30", None, "string"),
        ],
    )

    subset = load_dataset(q_path, docs_dir, selected_ids=["30", "10"])
    assert [q.id for q in subset.questions] == ["10", "30"]

    with pytest.raises(DatasetSchemaError, match="999"):
        load_dataset(q_path, docs_dir, selected_ids=["999"])


def test_load_submission_template_validates_schema_and_order(tmp_path: Path) -> None:
    tpl_path = write_submission_template(tmp_path / "tpl.xlsx", ["001", "002", "010"])
    assert load_submission_template(tpl_path) == ["001", "002", "010"]
