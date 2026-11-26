"""Deterministic in-memory FakeModelTransport test double at the ModelTransport seam."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from tianchi_table_recognition.gateway import TransportRequest, TransportResponse
from tianchi_table_recognition.models import ModelRole


@dataclass
class FakeModelTransport:
    """In-memory ModelTransport recording all outbound requests for verification."""

    handler: Callable[[TransportRequest], TransportResponse] | None = None
    role_responses: dict[ModelRole, str] = field(default_factory=dict)
    question_responses: dict[str, str] = field(default_factory=dict)
    document_responses: dict[str, str] = field(default_factory=dict)
    requests: list[TransportRequest] = field(default_factory=list)

    def send(self, request: TransportRequest) -> TransportResponse:
        """Record outbound request and return a deterministic response."""
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)

        if request.question_id and request.question_id in self.question_responses:
            content = self.question_responses[request.question_id]
        elif (
            request.model_role == ModelRole.VISION_EXTRACT
            and request.document_key
            and request.document_key in self.document_responses
        ):
            content = self.document_responses[request.document_key]
        elif request.model_role in self.role_responses:
            content = self.role_responses[request.model_role]
        else:
            content = "{}"

        return TransportResponse(
            content=content,
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
        )

    def count_calls(self, role: ModelRole | None = None) -> int:
        """Count recorded outbound calls, optionally filtered by ModelRole."""
        if role is None:
            return len(self.requests)
        return sum(1 for req in self.requests if req.model_role == role)

    def assert_no_filename_leak(self, forbidden_filenames: Iterable[str]) -> None:
        """Assert that no outbound request payload contains any raw Source Document filename."""
        forbidden = [name for name in forbidden_filenames if name]
        for req in self.requests:
            haystack = "\n".join(
                [
                    req.system_prompt,
                    req.user_prompt,
                    req.document_key or "",
                    req.question_id or "",
                    req.stage,
                ]
            )
            for name in forbidden:
                assert name not in haystack, (
                    f"Filename {name!r} leaked into outbound TransportRequest "
                    f"(stage={req.stage}, role={req.model_role.value})"
                )
