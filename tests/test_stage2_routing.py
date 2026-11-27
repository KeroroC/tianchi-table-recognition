"""Tests for Stage2Router routing and ThinkingSolver Decimal sandbox + retry + fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes import FakeModelTransport
from tianchi_table_recognition.config import load_config
from tianchi_table_recognition.gateway import (
    ComplianceError,
    ModelGateway,
    TransportRequest,
    TransportResponse,
)
from tianchi_table_recognition.models import (
    AnswerFormat,
    ModelRole,
    PageArtifact,
    Question,
    QuestionType,
    TableCell,
    TableGrid,
)
from tianchi_table_recognition.sandbox import PythonSandbox
from tianchi_table_recognition.stage2 import Stage2Router, ThinkingSolver


def _make_gateway(
    tmp_path: Path,
    transport: FakeModelTransport,
    *,
    text_model: str = "qwen-max",
) -> ModelGateway:
    cfg = load_config(
        env={
            "TTR_WORK_DIR": str(tmp_path / "work"),
            "TTR_MODEL_TEXT_REASON": text_model,
        }
    )
    return ModelGateway(
        config=cfg,
        transport=transport,
        run_id="stage2_routing_test",
        ledger_path=tmp_path / "work" / "runs" / "stage2_routing_test" / "audit.jsonl",
    )


def _sample_artifact() -> PageArtifact:
    grid = TableGrid(
        table_id="t1",
        title="月度费用表",
        row_count=4,
        col_count=2,
        header_row_count=1,
        cells=[
            TableCell(row=0, col=0, text="项目", uncertain=False),
            TableCell(row=0, col=1, text="金额", uncertain=False),
            TableCell(row=1, col=0, text="一月", uncertain=False),
            TableCell(row=1, col=1, text="0.1", uncertain=False),
            TableCell(row=2, col=0, text="二月", uncertain=False),
            TableCell(row=2, col=1, text="0.2", uncertain=False),
            TableCell(row=3, col=0, text="三月", uncertain=False),
            TableCell(row=3, col=1, text="1.25", uncertain=False),
        ],
    )
    return PageArtifact(
        document_key="doc_stage2_001",
        document_hash="c" * 64,
        page_index=0,
        page_context_text="费用明细",
        tables=[grid],
        stage="stage1_page_extract",
        stage_version="1.0",
        model_id="qwen-vl-max",
        prompt_hash="p" * 16,
    )


def test_router_dispatches_extract_thinking_and_structure(tmp_path: Path) -> None:
    artifact = _sample_artifact()

    def _handler(req: TransportRequest) -> TransportResponse:
        if req.question_id == "q_ext":
            return TransportResponse(content="0.1")
        if req.question_id == "q_thk":
            script = (
                "```python\n"
                "from decimal import Decimal\n"
                "vals = [Decimal(c['text']) for c in TABLES[0]['cells'] "
                "if c['row'] >= 1 and c['col'] == 1]\n"
                "RESULT = sum(vals, Decimal('0'))\n"
                "```"
            )
            return TransportResponse(content=script)
        raise AssertionError(f"Unexpected model call: {req}")

    transport = FakeModelTransport(handler=_handler)
    router = Stage2Router(gateway=_make_gateway(tmp_path, transport))

    res_ext = router.solve(
        Question(
            id="q_ext",
            document_key=artifact.document_key,
            question_type=QuestionType.EXTRACT,
            question="一月金额是多少？",
            table_hint="月度费用表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [artifact],
    )
    res_thk = router.solve(
        Question(
            id="q_thk",
            document_key=artifact.document_key,
            question_type=QuestionType.THINKING,
            question="三个月金额合计是多少？",
            table_hint="月度费用表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [artifact],
    )
    res_str = router.solve(
        Question(
            id="q_str",
            document_key=artifact.document_key,
            question_type=QuestionType.STRUCTURE,
            question="该表一共有多少行？",
            table_hint="月度费用表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [artifact],
    )

    assert res_ext.status == "ok" and res_ext.answer == "0.1" and res_ext.solver_path == "extract"
    assert (
        res_thk.status == "ok"
        and res_thk.answer == "1.55"
        and res_thk.solver_path == "thinking_python"
        and res_thk.fallback_no_python is False
        and res_thk.python_retry_count == 0
    )
    assert (
        res_str.status == "ok"
        and res_str.answer == "4"
        and res_str.solver_path == "structure_deterministic"
    )
    assert transport.count_calls(ModelRole.TEXT_REASON) == 2
    assert transport.count_calls(ModelRole.VISION_VERIFY) == 0


def test_thinking_retries_once_on_ast_rejection_and_succeeds(tmp_path: Path) -> None:
    artifact = _sample_artifact()
    stages: list[str] = []

    def _handler(req: TransportRequest) -> TransportResponse:
        stages.append(req.stage)
        if req.stage == ThinkingSolver.CODEGEN_STAGE:
            return TransportResponse(content="import os\nRESULT = 1")
        assert req.stage == ThinkingSolver.REPAIR_STAGE
        assert "ast_rejected" in req.user_prompt
        assert "Disallowed module import" in req.user_prompt
        return TransportResponse(
            content="from decimal import Decimal\nRESULT = Decimal('0.1') + Decimal('0.2')"
        )

    transport = FakeModelTransport(handler=_handler)
    solver = ThinkingSolver(gateway=_make_gateway(tmp_path, transport))
    res = solver.solve(
        Question(
            id="q_ast_retry",
            document_key=artifact.document_key,
            question_type=QuestionType.THINKING,
            question="一月和二月金额之和是多少？",
            table_hint="月度费用表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [artifact],
    )

    assert res.status == "ok"
    assert res.answer == "0.3"
    assert res.solver_path == "thinking_python"
    assert res.fallback_no_python is False
    assert res.python_retry_count == 1
    assert stages == [ThinkingSolver.CODEGEN_STAGE, ThinkingSolver.REPAIR_STAGE]


def test_thinking_retries_once_on_runtime_error_and_succeeds(tmp_path: Path) -> None:
    artifact = _sample_artifact()

    def _handler(req: TransportRequest) -> TransportResponse:
        if req.stage == ThinkingSolver.CODEGEN_STAGE:
            return TransportResponse(content="RESULT = 1 / 0")
        assert req.stage == ThinkingSolver.REPAIR_STAGE
        assert "ZeroDivisionError" in req.user_prompt
        return TransportResponse(
            content="from decimal import Decimal\nRESULT = Decimal('1.25') * Decimal('2')"
        )

    transport = FakeModelTransport(handler=_handler)
    solver = ThinkingSolver(gateway=_make_gateway(tmp_path, transport))
    res = solver.solve(
        Question(
            id="q_rt_retry",
            document_key=artifact.document_key,
            question_type=QuestionType.THINKING,
            question="三月金额的两倍是多少？",
            table_hint="月度费用表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [artifact],
    )

    assert res.status == "ok"
    assert res.answer == "2.5"
    assert res.python_retry_count == 1
    assert res.fallback_no_python is False


def test_thinking_falls_back_to_text_reasoning_after_retry_exhausted(tmp_path: Path) -> None:
    artifact = _sample_artifact()
    stages: list[str] = []

    def _handler(req: TransportRequest) -> TransportResponse:
        stages.append(req.stage)
        if req.stage == ThinkingSolver.CODEGEN_STAGE:
            return TransportResponse(content="while True:\n    pass")
        if req.stage == ThinkingSolver.REPAIR_STAGE:
            return TransportResponse(content="import sys\nRESULT = 1")
        assert req.stage == ThinkingSolver.FALLBACK_STAGE
        payload = json.loads(req.user_prompt.split("TablesJSON: ", 1)[1])
        assert payload[0]["table_id"] == "t1"
        return TransportResponse(content="1.55")

    transport = FakeModelTransport(handler=_handler)
    solver = ThinkingSolver(
        gateway=_make_gateway(tmp_path, transport),
        sandbox=PythonSandbox(timeout_seconds=0.3),
    )
    res = solver.solve(
        Question(
            id="q_fallback",
            document_key=artifact.document_key,
            question_type=QuestionType.THINKING,
            question="三个月合计金额是多少？",
            table_hint="月度费用表",
            answer_format=AnswerFormat.NUMBER,
        ),
        [artifact],
    )

    assert res.status == "ok"
    assert res.answer == "1.55"
    assert res.solver_path == "fallback_no_python"
    assert res.fallback_no_python is True
    assert res.python_retry_count == 1
    assert stages == [
        ThinkingSolver.CODEGEN_STAGE,
        ThinkingSolver.REPAIR_STAGE,
        ThinkingSolver.FALLBACK_STAGE,
    ]


def test_thinking_propagates_compliance_error(tmp_path: Path) -> None:
    artifact = _sample_artifact()

    class _BadTransport(FakeModelTransport):
        pass

    transport = _BadTransport()
    cfg = load_config(env={"TTR_WORK_DIR": str(tmp_path / "work")})
    gateway = ModelGateway(
        config=cfg,
        transport=transport,
        run_id="compliance_check",
        ledger_path=tmp_path / "work" / "runs" / "compliance_check" / "audit.jsonl",
    )
    # Mutate config in memory to trigger ComplianceError at gateway.invoke
    object.__setattr__(gateway._config, "bailian_base_url", "https://evil.example.com/v1")
    solver = ThinkingSolver(gateway=gateway)
    with pytest.raises(ComplianceError):
        solver.solve(
            Question(
                id="q_comp",
                document_key=artifact.document_key,
                question_type=QuestionType.THINKING,
                question="求和",
                answer_format=AnswerFormat.NUMBER,
            ),
            [artifact],
        )
