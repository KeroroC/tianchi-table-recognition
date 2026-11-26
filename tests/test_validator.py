"""Zero-mock and ModelTransport-seam tests for Answer Validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import FakeModelTransport
from tianchi_table_recognition.config import CLIOverrides, load_config
from tianchi_table_recognition.gateway import ComplianceError, ModelGateway, TransportResponse
from tianchi_table_recognition.models import (
    AnswerFormat,
    ModelRole,
    PageArtifact,
    Question,
    QuestionType,
    TableCell,
    TableGrid,
)
from tianchi_table_recognition.validator import AnswerValidator, validate_submission_answer


def _question(
    qid: str,
    answer_format: AnswerFormat,
    text: str = "请提取结果",
    question_type: QuestionType = QuestionType.EXTRACT,
) -> Question:
    return Question(
        id=qid,
        document_key="doc_test123",
        question_type=question_type,
        question=text,
        answer_format=answer_format,
    )


def test_json_and_json_array_deterministic_cleaning_and_type_checks() -> None:
    validator = AnswerValidator()

    # Valid compact Chinese JSON preserved unescaped
    res_ok = validator.validate(
        _question("j1", AnswerFormat.JSON),
        '{\n  "城市": "北京",\n  "数值": 100\n}',
    )
    assert res_ok.is_valid is True
    assert res_ok.answer == '{"城市":"北京","数值":100}'
    assert res_ok.repaired is False
    assert res_ok.repair_attempts == 0

    # Valid JSON with inner Chinese quotes must NOT be corrupted
    res_inner_quotes = validator.validate(
        _question("j2", AnswerFormat.JSON),
        '{"说明": "含“引号”文本", "a": 1,}',
    )
    assert res_inner_quotes.is_valid is True
    assert res_inner_quotes.answer == '{"说明":"含“引号”文本","a":1}'

    # Markdown fence + prose + Chinese structural quotes + fullwidth colon/comma + trailing comma
    res_dirty = validator.validate(
        _question("j3", AnswerFormat.JSON),
        '根据表格，结果为：\n```json\n{“名称”：“示例”， “数值”：12.5，}\n```\n以上供参考。',
    )
    assert res_dirty.is_valid is True
    assert res_dirty.answer == '{"名称":"示例","数值":12.5}'

    # Valid ASCII strings with inner Chinese quotes alongside outside-string fullwidth colon
    res_mixed_quotes = validator.validate(
        _question("j4", AnswerFormat.JSON),
        '{"说明": "含“引号”文本", "a"：1}',
    )
    assert res_mixed_quotes.is_valid is True
    assert res_mixed_quotes.answer == '{"说明":"含“引号”文本","a":1}'

    # Valid json_array with fence + trailing comma
    res_arr = validator.validate(
        _question("a1", AnswerFormat.JSON_ARRAY),
        '```json\n[\n  {"项": "甲"},\n  {"项": "乙"},\n]\n```',
    )
    assert res_arr.is_valid is True
    assert res_arr.answer == '[{"项":"甲"},{"项":"乙"}]'

    # Object vs Array type mismatch must NOT extract inner object or array
    res_obj_for_arr = validator.validate(
        _question("m1", AnswerFormat.JSON_ARRAY),
        '{"items":[1,2]}',
    )
    assert res_obj_for_arr.is_valid is False
    assert res_obj_for_arr.answer is None
    assert any("array" in err.lower() for err in res_obj_for_arr.errors)

    res_arr_for_obj = validator.validate(
        _question("m2", AnswerFormat.JSON),
        '[{"a":1}]',
    )
    assert res_arr_for_obj.is_valid is False
    assert res_arr_for_obj.answer is None
    assert any("object" in err.lower() for err in res_arr_for_obj.errors)

    # Non-standard JSON constants NaN / Infinity must be rejected
    res_nan = validator.validate(
        _question("n1", AnswerFormat.JSON),
        '{"v": NaN}',
    )
    assert res_nan.is_valid is False
    assert res_nan.answer is None


def test_json_repair_bounded_to_one_call_and_propagates_compliance_error(
    tmp_path: Path,
) -> None:
    config = load_config(cli_overrides=CLIOverrides(work_dir=tmp_path))
    transport = FakeModelTransport(
        question_responses={
            "r1": '```json\n{"结果": "成功", "值": 42,}\n```',
            "r2": "still not valid json",
            "r3": '{"wrong": "object"}',
        }
    )
    gateway = ModelGateway(
        config=config,
        transport=transport,
        run_id="validator_repair_test",
        ledger_path=tmp_path / "audit_ledger.jsonl",
    )
    validator = AnswerValidator(gateway=gateway)

    # r1: deterministic cleaning fails, 1 repair call succeeds (and cleans fence/trailing comma)
    res_fixed = validator.validate(
        _question("r1", AnswerFormat.JSON),
        "{结果: 未闭合",
    )
    assert res_fixed.is_valid is True
    assert res_fixed.answer == '{"结果":"成功","值":42}'
    assert res_fixed.repaired is True
    assert res_fixed.repair_attempts == 1
    assert transport.count_calls(ModelRole.TEXT_REASON) == 1
    assert transport.requests[0].stage == "stage2_json_repair"
    assert transport.requests[0].document_key == "doc_test123"
    assert transport.requests[0].question_id == "r1"

    # r2: 1 repair call returns invalid JSON -> fails without a 2nd attempt
    res_still_bad = validator.validate(
        _question("r2", AnswerFormat.JSON),
        "{broken",
    )
    assert res_still_bad.is_valid is False
    assert res_still_bad.answer is None
    assert res_still_bad.repaired is False
    assert res_still_bad.repair_attempts == 1
    assert transport.count_calls(ModelRole.TEXT_REASON) == 2

    # r3: 1 repair call returns dict for JSON_ARRAY -> fails type check without a 2nd attempt
    res_wrong_type = validator.validate(
        _question("r3", AnswerFormat.JSON_ARRAY),
        "[broken",
    )
    assert res_wrong_type.is_valid is False
    assert res_wrong_type.repair_attempts == 1
    assert any("array" in err.lower() for err in res_wrong_type.errors)
    assert transport.count_calls(ModelRole.TEXT_REASON) == 3

    # max_json_repair_attempts=0 disables repair calls
    no_repair_validator = AnswerValidator(gateway=gateway, max_json_repair_attempts=0)
    res_disabled = no_repair_validator.validate(
        _question("r1", AnswerFormat.JSON),
        "{broken",
    )
    assert res_disabled.is_valid is False
    assert res_disabled.repair_attempts == 0
    assert transport.count_calls(ModelRole.TEXT_REASON) == 3

    with pytest.raises(ValueError):
        AnswerValidator(max_json_repair_attempts=-1)

    # ComplianceError during repair must propagate immediately
    def _raise_compliance(_req: object) -> TransportResponse:
        raise ComplianceError("blocked non-compliant endpoint")

    bad_gateway = ModelGateway(
        config=config,
        transport=FakeModelTransport(handler=_raise_compliance),
        run_id="compliance_prop",
        ledger_path=tmp_path / "audit_ledger2.jsonl",
    )
    bad_validator = AnswerValidator(gateway=bad_gateway)
    with pytest.raises(ComplianceError):
        bad_validator.validate(_question("r4", AnswerFormat.JSON), "{broken")


def test_number_normalization_precision_commas_percent_and_unit_rules() -> None:
    validator = AnswerValidator()

    # Strip prefix, commas, trailing .0 / .00, and trailing sentence period
    assert (
        validator.validate(
            _question("num1", AnswerFormat.NUMBER, "总营收是多少？"),
            "答案：1,234,567.00。",
        ).answer
        == "1234567"
    )
    assert (
        validator.validate(
            _question("num2", AnswerFormat.NUMBER, "增长量是多少？"),
            "`-0.0`",
        ).answer
        == "0"
    )
    assert (
        validator.validate(
            _question("num3", AnswerFormat.NUMBER, "单价是多少？"),
            "12.50",
        ).answer
        == "12.5"
    )
    assert (
        validator.validate(
            _question("num4", AnswerFormat.NUMBER, "金额是多少？"),
            "￥1,000.0",
        ).answer
        == "1000"
    )

    # Preserve commas when question explicitly asks for thousands separator,
    # but strip commas when negated
    assert (
        validator.validate(
            _question("num_comma", AnswerFormat.NUMBER, "请计算总额（保留千分位逗号分隔）"),
            "1234567.0",
        ).answer
        == "1,234,567"
    )
    assert (
        validator.validate(
            _question("num_no_comma", AnswerFormat.NUMBER, "请计算总额（不要千分位逗号）"),
            "1,234,567.0",
        ).answer
        == "1234567"
    )

    # Decimal precision directives (Chinese digits, Arabic digits, 0.01, integer)
    assert (
        validator.validate(
            _question("num_p1", AnswerFormat.NUMBER, "净利率是多少（保留两位小数）？"),
            "12.3",
        ).answer
        == "12.30"
    )
    assert (
        validator.validate(
            _question("num_p2", AnswerFormat.NUMBER, "均值是多少（精确到0.01）？"),
            "12.345",
        ).answer
        == "12.35"
    )
    assert (
        validator.validate(
            _question("num_p3", AnswerFormat.NUMBER, "人数是多少（保留整数）？"),
            "12.6",
        ).answer
        == "13"
    )

    # Percent vs decimal: never silently convert between % and decimal
    assert (
        validator.validate(
            _question("pct1", AnswerFormat.NUMBER, "同比增长率是多少？"),
            "25.0%",
        ).answer
        == "25%"
    )
    assert (
        validator.validate(
            _question("pct2", AnswerFormat.NUMBER, "占比是多少？"),
            "0.25",
        ).answer
        == "0.25"
    )
    pct_mismatch = validator.validate(
        _question("pct_bad1", AnswerFormat.NUMBER, "占比是多少（以%表示）？"),
        "0.25",
    )
    assert pct_mismatch.is_valid is False
    dec_mismatch = validator.validate(
        _question("pct_bad2", AnswerFormat.NUMBER, "占比是多少（用小数表示）？"),
        "25%",
    )
    assert dec_mismatch.is_valid is False

    # Unit handling: strip redundant matching unit by default, keep if requested, fail on mismatch
    assert (
        validator.validate(
            _question("u1", AnswerFormat.NUMBER, "净利润是多少（单位：万元）？"),
            "120.0万元",
        ).answer
        == "120"
    )
    assert (
        validator.validate(
            _question("u2", AnswerFormat.NUMBER, "净利润是多少（请带单位，单位：万元）？"),
            "120.0",
        ).answer
        == "120万元"
    )
    unit_conflict = validator.validate(
        _question("u3", AnswerFormat.NUMBER, "净利润是多少（单位：万元）？"),
        "1.2亿元",
    )
    assert unit_conflict.is_valid is False
    paren_unit_conflict = validator.validate(
        _question("u3b", AnswerFormat.NUMBER, "营业收入（万元）是多少？"),
        "1.2亿元",
    )
    assert paren_unit_conflict.is_valid is False
    currency_conflict = validator.validate(
        _question("u4", AnswerFormat.NUMBER, "净利润是多少（单位：元）？"),
        "$100",
    )
    assert currency_conflict.is_valid is False

    # Reject non-numeric, NaN/Inf, malformed commas
    for bad in ("", "NaN", "Infinity", "无", "12,34"):
        res_bad = validator.validate(_question("bad", AnswerFormat.NUMBER, "数量是多少？"), bad)
        assert res_bad.is_valid is False


def _sample_artifact() -> PageArtifact:
    return PageArtifact(
        document_key="doc_test123",
        document_hash="h123",
        page_index=0,
        page_context_text="【重要提示】公告：第一季度 营业收入说明",
        tables=[
            TableGrid(
                table_id="t1",
                row_count=2,
                col_count=2,
                header_row_count=1,
                cells=[
                    TableCell(row=0, col=0, text="公司名称"),
                    TableCell(row=0, col=1, text="英文名"),
                    TableCell(row=1, col=0, text="深圳市腾讯计算机系统有限公司"),
                    TableCell(row=1, col=1, text="Apple  Inc."),
                ],
            )
        ],
        stage="stage1_page_extract",
        stage_version="1.0",
        model_id="qwen-vl-max",
        prompt_hash="p123",
    )


def test_string_prefix_stripping_bracket_fidelity_and_spurious_spaces() -> None:
    validator = AnswerValidator()
    artifact = _sample_artifact()

    # Strip outer whitespace, zero-width chars, and explanatory prefixes
    res1 = validator.validate(
        _question("s1", AnswerFormat.STRING),
        "\u3000根据表格，答案为：  【重要提示】公告 \u200b",
        artifact=artifact,
    )
    assert res1.is_valid is True
    assert res1.answer == "【重要提示】公告"

    # Distinct brackets preserved verbatim without artifact too
    assert (
        validator.validate(
            _question("s2", AnswerFormat.STRING),
            "Answer: [标准括号]与【方头括号】",
        ).answer
        == "[标准括号]与【方头括号】"
    )

    # Mismatched and unclosed bracket pairs must fail
    for bad_brackets in ("【错配]", "[标题", "【标题"):
        assert (
            validator.validate(_question("s_m1", AnswerFormat.STRING), bad_brackets).is_valid
            is False
        )

    # Bracket mutation relative to PageArtifact source text must fail
    res_bracket_drift = validator.validate(
        _question("s_b1", AnswerFormat.STRING),
        "[重要提示]公告",
        artifact=artifact,
    )
    assert res_bracket_drift.is_valid is False
    assert any("bracket" in err.lower() for err in res_bracket_drift.errors)

    # Authentic spaces present in source text pass
    assert (
        validator.validate(
            _question("s_ok_space1", AnswerFormat.STRING),
            "第一季度 营业收入",
            artifact=artifact,
        ).is_valid
        is True
    )
    assert (
        validator.validate(
            _question("s_ok_space2", AnswerFormat.STRING),
            "Apple Inc.",
            artifact=artifact,
        ).is_valid
        is True
    )

    # Spurious space inserted into spaceless source text fails
    res_spurious_art = validator.validate(
        _question("s_bad_space1", AnswerFormat.STRING),
        "深圳市 腾讯计算机系统有限公司",
        artifact=artifact,
    )
    assert res_spurious_art.is_valid is False

    # Structural spurious space checks without artifact
    for bad_str in ("北京 市", "第 1 名", "【 公告】", "12 345", "   ", "答案："):
        res_bad = validator.validate(_question("s_bad", AnswerFormat.STRING), bad_str)
        assert res_bad.is_valid is False

    # Valid English multi-word, datetime, codes, and words starting with 结果/Result
    for ok_str in (
        "New York",
        "Phase 2",
        "2024-03-01 12:00:00",
        "00123",
        "=A1+B1",
        "结果分析报告",
        "ResultSet",
    ):
        res_ok = validator.validate(_question("s_ok", AnswerFormat.STRING), ok_str)
        assert res_ok.is_valid is True
        assert res_ok.answer == ok_str


def test_validate_submission_answer_defense_in_depth() -> None:
    assert validate_submission_answer(AnswerFormat.JSON, '{"north":100}') == ()
    assert validate_submission_answer(AnswerFormat.JSON_ARRAY, '[{"a":"甲"}]') == ()
    assert validate_submission_answer(AnswerFormat.NUMBER, "123.45") == ()
    assert validate_submission_answer(AnswerFormat.NUMBER, "25%") == ()
    assert validate_submission_answer(AnswerFormat.STRING, "00123") == ()
    assert validate_submission_answer(AnswerFormat.STRING, "Apple Inc.") == ()

    assert validate_submission_answer(AnswerFormat.JSON, '{\n"a":1\n}') != ()
    assert validate_submission_answer(AnswerFormat.JSON, "[1,2]") != ()
    assert validate_submission_answer(AnswerFormat.JSON_ARRAY, '{"a":1}') != ()
    assert validate_submission_answer(AnswerFormat.JSON, '{"v":NaN}') != ()
    assert validate_submission_answer(AnswerFormat.NUMBER, "not-a-number") != ()
    assert validate_submission_answer(AnswerFormat.STRING, "  untrimmed ") != ()
    assert validate_submission_answer(AnswerFormat.STRING, "【错配]") != ()
    assert validate_submission_answer(AnswerFormat.STRING, "") != ()
    assert validate_submission_answer(AnswerFormat.STRING, "第一季度 营业收入") == ()



