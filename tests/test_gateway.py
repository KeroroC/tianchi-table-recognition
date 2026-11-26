"""Tests for ModelGateway compliance allowlists, ModelTransport seam, and audit ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes import FakeModelTransport
from tianchi_table_recognition.config import CLIOverrides, load_config
from tianchi_table_recognition.gateway import (
    ComplianceError,
    ModelGateway,
    TransportRequest,
    TransportResponse,
)
from tianchi_table_recognition.models import ModelRole


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://api.openai.com/v1",
        "http://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://evil-dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com.evil.com/v1",
        "http://localhost:11434/v1",
        "https://user:pass@dashscope.aliyuncs.com/compatible-mode/v1",
    ],
)
def test_gateway_rejects_non_bailian_endpoint_before_calling_transport(
    tmp_path: Path, bad_url: str
) -> None:
    cfg = load_config(env={"TTR_BAILIAN_BASE_URL": bad_url})
    transport = FakeModelTransport()

    with pytest.raises(ComplianceError, match="Bailian"):
        ModelGateway(
            config=cfg,
            transport=transport,
            run_id="run-bad-url",
            ledger_path=tmp_path / "run-bad-url" / "audit_ledger.jsonl",
        )

    assert transport.count_calls() == 0


@pytest.mark.parametrize(
    "bad_model",
    [
        "gpt-4o",
        "claude-3-5-sonnet",
        "llama-3.1-70b",
        "deepseek-r1",
        "local/qwen-max",
        "openai/qwen-max",
        "   ",
    ],
)
def test_gateway_rejects_non_qwen_model_immediately(tmp_path: Path, bad_model: str) -> None:
    cfg = load_config(env={"TTR_MODEL_TEXT_REASON": bad_model})
    transport = FakeModelTransport()

    with pytest.raises(ComplianceError, match="qwen"):
        ModelGateway(
            config=cfg,
            transport=transport,
            run_id="run-bad-model",
            ledger_path=tmp_path / "run-bad-model" / "audit_ledger.jsonl",
        )

    assert transport.count_calls() == 0


def test_gateway_rejects_empty_cli_model_immediately(tmp_path: Path) -> None:
    cfg = load_config(env={}, cli_overrides=CLIOverrides(text_reason_model=""))
    transport = FakeModelTransport()

    with pytest.raises(ComplianceError, match="qwen"):
        ModelGateway(
            config=cfg,
            transport=transport,
            run_id="run-empty-model",
            ledger_path=tmp_path / "run-empty-model" / "audit_ledger.jsonl",
        )

    assert transport.count_calls() == 0


def test_gateway_allows_compliant_qwen_models_and_writes_run_isolated_ledger(
    tmp_path: Path,
) -> None:
    cfg = load_config(
        env={
            "TTR_BAILIAN_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "TTR_MODEL_TEXT_REASON": "qwen2.5-72b-instruct",
        }
    )
    transport = FakeModelTransport(
        handler=lambda req: TransportResponse(
            content="42",
            prompt_tokens=500,
            completion_tokens=100,
            total_tokens=600,
        )
    )
    ledger_a = tmp_path / "runs" / "run_a" / "audit_ledger.jsonl"
    ledger_b = tmp_path / "runs" / "run_b" / "audit_ledger.jsonl"

    gw_a = ModelGateway(config=cfg, transport=transport, run_id="run_a", ledger_path=ledger_a)
    gw_b = ModelGateway(config=cfg, transport=transport, run_id="run_b", ledger_path=ledger_b)

    resp = gw_a.invoke(
        model_role=ModelRole.TEXT_REASON,
        system_prompt="sys",
        user_prompt="doc_abc question",
        stage="stage2_extract",
        document_key="doc_abc",
        question_id="001",
    )
    gw_b.invoke(
        model_role=ModelRole.VISION_EXTRACT,
        system_prompt="sys",
        user_prompt="doc_xyz",
        stage="stage1_page_extract",
        document_key="doc_xyz",
    )

    assert resp.content == "42"
    lines_a = [json.loads(line) for line in ledger_a.read_text(encoding="utf-8").splitlines()]
    lines_b = [json.loads(line) for line in ledger_b.read_text(encoding="utf-8").splitlines()]
    assert len(lines_a) == 1
    assert len(lines_b) == 1

    entry_a = lines_a[0]
    assert entry_a["run_id"] == "run_a"
    assert entry_a["model_role"] == "text_reason"
    assert entry_a["model_id"] == "qwen2.5-72b-instruct"
    assert entry_a["endpoint_host"] == "dashscope-intl.aliyuncs.com"
    assert entry_a["prompt_tokens"] == 500
    assert entry_a["completion_tokens"] == 100
    assert entry_a["total_tokens"] == 600
    assert entry_a["estimated_cost_cny"] > 0
    assert entry_a["status"] == "ok"


def test_fake_transport_detects_filename_leak_and_passes_when_clean() -> None:
    transport = FakeModelTransport()
    transport.send(
        TransportRequest(
            model_role=ModelRole.TEXT_REASON,
            model_id="qwen-max",
            system_prompt="sys",
            user_prompt="Answer using doc_a1b2c3d4",
            stage="stage2_extract",
            document_key="doc_a1b2c3d4",
            question_id="1",
        )
    )
    transport.assert_no_filename_leak(["secret_report.png"])

    transport.send(
        TransportRequest(
            model_role=ModelRole.TEXT_REASON,
            model_id="qwen-max",
            system_prompt="sys",
            user_prompt="Look at secret_report.png",
            stage="stage2_extract",
            document_key="doc_a1b2c3d4",
            question_id="2",
        )
    )
    with pytest.raises(AssertionError, match="secret_report.png"):
        transport.assert_no_filename_leak(["secret_report.png"])
