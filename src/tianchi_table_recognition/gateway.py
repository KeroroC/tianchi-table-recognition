"""Compliant ModelGateway, ModelTransport seam, Bailian adapter, and JSONL audit ledger."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from tianchi_table_recognition.config import AppConfig
from tianchi_table_recognition.models import ModelRole

ALLOWED_BAILIAN_HOSTS: frozenset[str] = frozenset(
    {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
    }
)

_QWEN_MODEL_PATTERN = re.compile(r"^qwen[a-z0-9._-]*$", re.IGNORECASE)


class ComplianceError(ValueError):
    """Raised when an endpoint or model violates Alibaba Cloud Bailian / Qwen guardrails."""


class EncodedImage(BaseModel):
    """Filename-stripped, base64-encoded image payload for multimodal requests."""

    model_config = ConfigDict(frozen=True)

    mime_type: str
    data_base64: str
    width: int
    height: int


class TransportRequest(BaseModel):
    """Outbound model request crossing the ModelTransport seam."""

    model_config = ConfigDict(frozen=True)

    model_role: ModelRole
    model_id: str
    system_prompt: str
    user_prompt: str
    images: tuple[EncodedImage, ...] = ()
    stage: str
    document_key: str | None = None
    question_id: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096


class TransportResponse(BaseModel):
    """Normalized model response returned by ModelTransport."""

    model_config = ConfigDict(frozen=True)

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelTransport(Protocol):
    """Single external seam below ModelGateway for production Bailian client or test fake."""

    def send(self, request: TransportRequest) -> TransportResponse:
        """Execute a single model completion call."""
        ...


def validate_bailian_url(base_url: str) -> str:
    """Validate that base_url points exclusively to an official HTTPS Bailian host."""
    parsed = urlparse(base_url.strip())
    if parsed.scheme != "https":
        raise ComplianceError(
            f"Non-compliant Bailian endpoint scheme {parsed.scheme!r}; HTTPS is required."
        )
    if parsed.username is not None or parsed.password is not None:
        raise ComplianceError("Bailian endpoint URL must not embed user credentials.")
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_BAILIAN_HOSTS:
        raise ComplianceError(
            f"Non-compliant Bailian endpoint host {hostname!r}; "
            f"allowed hosts: {sorted(ALLOWED_BAILIAN_HOSTS)}."
        )
    return hostname


def validate_compliant_qwen_model(model_id: str) -> str:
    """Validate that model_id matches the compliant qwen* allowlist."""
    candidate = model_id.strip()
    if not candidate or not _QWEN_MODEL_PATTERN.fullmatch(candidate):
        raise ComplianceError(
            f"Non-compliant model_id {model_id!r}; only Alibaba Cloud qwen* models are allowed."
        )
    return candidate


class BailianOpenAITransport:
    """Production ModelTransport backed by Alibaba Cloud Bailian OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str | None) -> None:
        validate_bailian_url(base_url)
        self._base_url = base_url
        self._api_key = api_key
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self._api_key:
                raise ComplianceError(
                    "DASHSCOPE_API_KEY is required to invoke Alibaba Cloud Bailian."
                )
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def send(self, request: TransportRequest) -> TransportResponse:
        """Send a chat completion request to Bailian's OpenAI-compatible endpoint."""
        client = self._get_client()
        if request.images:
            user_content: str | list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{img.data_base64}"},
                }
                for img in request.images
            ] + [{"type": "text", "text": request.user_prompt}]
        else:
            user_content = request.user_prompt

        messages: list[Any] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_content},
        ]
        completion = client.chat.completions.create(
            model=request.model_id,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        message = completion.choices[0].message
        content = message.content or ""
        usage = completion.usage
        prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0
        completion_tokens = (
            int(usage.completion_tokens) if usage and usage.completion_tokens else 0
        )
        total_tokens = (
            int(usage.total_tokens)
            if usage and usage.total_tokens
            else prompt_tokens + completion_tokens
        )
        return TransportResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class ModelGateway:
    """Single compliance gatekeeper and Run-isolated audit logger for all model calls."""

    def __init__(
        self,
        config: AppConfig,
        transport: ModelTransport,
        run_id: str,
        ledger_path: Path,
    ) -> None:
        self._config = config
        self._transport = transport
        self._run_id = run_id
        self._ledger_path = ledger_path
        self._endpoint_host = validate_bailian_url(config.bailian_base_url)
        for model_id in config.model_roles.values():
            validate_compliant_qwen_model(model_id)
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        """Identifier of the active Run."""
        return self._run_id

    @property
    def ledger_path(self) -> Path:
        """Path to the Run-isolated JSONL audit ledger."""
        return self._ledger_path

    def model_for_role(self, model_role: ModelRole) -> str:
        """Resolve and compliance-validate the model ID for a logical Model Role."""
        return validate_compliant_qwen_model(self._config.model_for_role(model_role))

    def invoke(
        self,
        *,
        model_role: ModelRole,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        images: Sequence[EncodedImage] = (),
        document_key: str | None = None,
        question_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> TransportResponse:
        """Validate compliance guardrails, execute via ModelTransport, and record audit ledger."""
        validate_bailian_url(self._config.bailian_base_url)
        model_id = self.model_for_role(model_role)

        request = TransportRequest(
            model_role=model_role,
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=tuple(images),
            stage=stage,
            document_key=document_key,
            question_id=question_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        started = time.perf_counter()
        try:
            response = self._transport.send(request)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            self._append_ledger_entry(
                request=request,
                response=None,
                latency_ms=elapsed_ms,
                status="error",
                error=str(exc),
            )
            raise

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self._append_ledger_entry(
            request=request,
            response=response,
            latency_ms=elapsed_ms,
            status="ok",
            error=None,
        )
        return response

    def _estimate_cost_cny(
        self, model_id: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        rate = self._config.pricing_for_model(model_id)
        cost = (prompt_tokens / 1000.0) * rate.input + (completion_tokens / 1000.0) * rate.output
        return round(cost, 8)

    def _append_ledger_entry(
        self,
        *,
        request: TransportRequest,
        response: TransportResponse | None,
        latency_ms: float,
        status: str,
        error: str | None,
    ) -> None:
        prompt_tokens = response.prompt_tokens if response else 0
        completion_tokens = response.completion_tokens if response else 0
        total_tokens = response.total_tokens if response else 0
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self._run_id,
            "model_role": request.model_role.value,
            "model_id": request.model_id,
            "endpoint_host": self._endpoint_host,
            "stage": request.stage,
            "document_key": request.document_key,
            "question_id": request.question_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_cny": self._estimate_cost_cny(
                request.model_id, prompt_tokens, completion_tokens
            ),
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
        }
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
