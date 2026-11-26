"""Stage 2 question answering over cached PageArtifact / TableGrid representations."""

from __future__ import annotations

import json

from tianchi_table_recognition.gateway import ComplianceError, ModelGateway
from tianchi_table_recognition.models import (
    ModelRole,
    PageArtifact,
    Question,
    QuestionResult,
    QuestionType,
)
from tianchi_table_recognition.validator import AnswerValidator

DEFAULT_EXTRACT_SYSTEM_PROMPT = (
    "You are an accurate table QA assistant. Answer strictly from the provided "
    "table grid and page context. Return only the final answer value with no "
    "explanation or markdown code fences."
)


class ExtractSolver:
    """Baseline Stage 2 solver for extract Questions using ModelRole.TEXT_REASON."""

    STAGE_NAME = "stage2_extract"

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        system_prompt: str = DEFAULT_EXTRACT_SYSTEM_PROMPT,
        validator: AnswerValidator | None = None,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._validator = validator or AnswerValidator(gateway=gateway)

    def solve(self, question: Question, artifact: PageArtifact) -> QuestionResult:
        """Solve a single extract Question using its anonymized PageArtifact."""
        if question.question_type != QuestionType.EXTRACT:
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=(
                    f"Unsupported question_type={question.question_type.value!r} "
                    "in T1 extract-only loop."
                ),
            )

        tables_payload = [table.model_dump(mode="json") for table in artifact.tables]
        user_prompt = "\n".join(
            [
                f"DocumentKey: {question.document_key}",
                f"Question: {question.question}",
                f"TableHint: {question.table_hint or ''}",
                f"AnswerFormat: {question.answer_format.value}",
                f"PageContext: {artifact.page_context_text}",
                f"TablesJSON: {json.dumps(tables_payload, ensure_ascii=False)}",
            ]
        )

        try:
            response = self._gateway.invoke(
                model_role=ModelRole.TEXT_REASON,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                stage=self.STAGE_NAME,
                document_key=question.document_key,
                question_id=question.id,
            )
            validation = self._validator.validate(
                question,
                response.content,
                artifact=artifact,
            )
            is_valid = validation.is_valid and validation.answer is not None
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="ok" if is_valid else "failed",
                answer=validation.answer if is_valid else None,
                error=None
                if is_valid
                else ("; ".join(validation.errors) or "Answer validation failed"),
                raw_answer=response.content,
                validation_errors=() if is_valid else validation.errors,
                json_repaired=validation.repaired,
            )
        except ComplianceError:
            raise
        except Exception as exc:
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=str(exc),
            )
