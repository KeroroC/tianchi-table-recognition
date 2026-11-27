"""Stage 2 question-type routing, extract, thinking (Decimal Python sandbox), and structure."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tianchi_table_recognition.gateway import ComplianceError, ModelGateway
from tianchi_table_recognition.locator import TableLocator, build_located_payload
from tianchi_table_recognition.models import (
    ModelRole,
    PageArtifact,
    Question,
    QuestionResult,
    QuestionType,
    SolverPath,
)
from tianchi_table_recognition.preprocessing import PreparedPage
from tianchi_table_recognition.sandbox import PythonSandbox, SandboxResult, extract_python_code
from tianchi_table_recognition.structure import StructureSolver
from tianchi_table_recognition.validator import AnswerValidator, validate_and_build_result

DEFAULT_EXTRACT_SYSTEM_PROMPT = (
    "You are an accurate table QA assistant. Answer strictly from the provided "
    "table grid and page context. Return only the final answer value with no "
    "explanation or markdown code fences."
)

DEFAULT_THINKING_CODEGEN_SYSTEM_PROMPT = (
    "You are an accurate financial and tabular calculation assistant. Write a concise, "
    "pure Python calculation script using `decimal.Decimal` (never float arithmetic for "
    "financial sums/ratios) over the pre-populated variables `DATA`, `TABLES`, "
    "`PAGE_CONTEXT`, and `QUESTION`. Allowed imports are strictly `decimal`, `math`, "
    "`statistics`, `re`, and `json`. Assign the final answer to `RESULT` or define "
    "`solve(data)`. Return Python code only."
)

DEFAULT_THINKING_FALLBACK_SYSTEM_PROMPT = (
    "You are an accurate table reasoning assistant. Answer strictly from the provided "
    "table grid and page context. Return only the final answer value with no "
    "explanation or markdown code fences."
)


class ExtractSolver:
    """Stage 2 solver for extract Questions using ModelRole.TEXT_REASON."""

    STAGE_NAME = "stage2_extract"

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        system_prompt: str = DEFAULT_EXTRACT_SYSTEM_PROMPT,
        validator: AnswerValidator | None = None,
        locator: TableLocator | None = None,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._validator = validator or AnswerValidator(gateway=gateway)
        self._locator = locator or TableLocator()

    def solve(
        self,
        question: Question,
        artifacts: PageArtifact | Sequence[PageArtifact],
    ) -> QuestionResult:
        """Solve a single extract Question using located page-atomic PageArtifacts."""
        if question.question_type != QuestionType.EXTRACT:
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=(
                    f"Unsupported question_type={question.question_type.value!r} "
                    "for ExtractSolver."
                ),
                solver_path=SolverPath.EXTRACT,
            )

        artifact_list = [artifacts] if isinstance(artifacts, PageArtifact) else list(artifacts)
        located = self._locator.locate(question, artifact_list)
        prompt_context = build_located_payload(located)
        user_prompt = prompt_context.format_user_prompt(question)

        try:
            response = self._gateway.invoke(
                model_role=ModelRole.TEXT_REASON,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                stage=self.STAGE_NAME,
                document_key=question.document_key,
                question_id=question.id,
            )
            return validate_and_build_result(
                validator=self._validator,
                question=question,
                raw_answer=response.content,
                artifact=located.pages,
                solver_path=SolverPath.EXTRACT,
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
                solver_path=SolverPath.EXTRACT,
            )


class ThinkingSolver:
    """Stage 2 solver for thinking Questions using AST-sandboxed Decimal Python + fallback."""

    CODEGEN_STAGE = "stage2_thinking_codegen"
    REPAIR_STAGE = "stage2_thinking_codegen_repair"
    FALLBACK_STAGE = "stage2_thinking_fallback"

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        codegen_system_prompt: str = DEFAULT_THINKING_CODEGEN_SYSTEM_PROMPT,
        fallback_system_prompt: str = DEFAULT_THINKING_FALLBACK_SYSTEM_PROMPT,
        validator: AnswerValidator | None = None,
        locator: TableLocator | None = None,
        sandbox: PythonSandbox | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self._gateway = gateway
        self._codegen_system_prompt = codegen_system_prompt
        self._fallback_system_prompt = fallback_system_prompt
        self._validator = validator or AnswerValidator(gateway=gateway)
        self._locator = locator or TableLocator()
        self._sandbox = sandbox or PythonSandbox()
        self._max_repair_attempts = max_repair_attempts

    def solve(
        self,
        question: Question,
        artifacts: PageArtifact | Sequence[PageArtifact],
    ) -> QuestionResult:
        """Generate Decimal Python script, run in sandbox with 1 repair retry, or fall back."""
        if question.question_type != QuestionType.THINKING:
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=(
                    f"Unsupported question_type={question.question_type.value!r} "
                    "for ThinkingSolver."
                ),
                solver_path=SolverPath.THINKING_PYTHON,
            )

        artifact_list = [artifacts] if isinstance(artifacts, PageArtifact) else list(artifacts)
        located = self._locator.locate(question, artifact_list)
        prompt_context = build_located_payload(located)
        base_prompt = prompt_context.format_user_prompt(question)
        sandbox_context: dict[str, Any] = {
            "document_key": question.document_key,
            "question": question.question,
            "table_hint": question.table_hint or "",
            "answer_format": question.answer_format.value,
            "page_context": prompt_context.page_context,
            "tables": prompt_context.tables_payload,
        }

        retry_count = 0
        sandbox_result: SandboxResult | None = None
        last_code = ""

        try:
            codegen_response = self._gateway.invoke(
                model_role=ModelRole.TEXT_REASON,
                system_prompt=self._codegen_system_prompt,
                user_prompt=base_prompt,
                stage=self.CODEGEN_STAGE,
                document_key=question.document_key,
                question_id=question.id,
            )
            last_code = extract_python_code(codegen_response.content)
            sandbox_result = self._sandbox.execute(last_code, context=sandbox_context)

            while not sandbox_result.ok and retry_count < self._max_repair_attempts:
                retry_count += 1
                repair_prompt = "\n".join(
                    [
                        base_prompt,
                        f"PreviousCode:\n```python\n{last_code}\n```",
                        f"SandboxErrorType: {sandbox_result.error_type or 'unknown'}",
                        f"SandboxError: {sandbox_result.error or ''}",
                        f"SandboxStderr: {sandbox_result.stderr}",
                    ]
                )
                repair_response = self._gateway.invoke(
                    model_role=ModelRole.TEXT_REASON,
                    system_prompt=self._codegen_system_prompt,
                    user_prompt=repair_prompt,
                    stage=self.REPAIR_STAGE,
                    document_key=question.document_key,
                    question_id=question.id,
                )
                last_code = extract_python_code(repair_response.content)
                sandbox_result = self._sandbox.execute(last_code, context=sandbox_context)
        except ComplianceError:
            raise
        except Exception as exc:
            sandbox_result = SandboxResult(
                ok=False,
                error_type="runtime_error",
                error=str(exc),
            )

        if sandbox_result is not None and sandbox_result.ok and sandbox_result.answer is not None:
            return validate_and_build_result(
                validator=self._validator,
                question=question,
                raw_answer=sandbox_result.answer,
                artifact=located.pages,
                solver_path=SolverPath.THINKING_PYTHON,
                fallback_no_python=False,
                python_retry_count=retry_count,
            )

        # Smoothly degrade to direct text reasoning after repair attempts are exhausted.
        try:
            fallback_response = self._gateway.invoke(
                model_role=ModelRole.TEXT_REASON,
                system_prompt=self._fallback_system_prompt,
                user_prompt=base_prompt,
                stage=self.FALLBACK_STAGE,
                document_key=question.document_key,
                question_id=question.id,
            )
            return validate_and_build_result(
                validator=self._validator,
                question=question,
                raw_answer=fallback_response.content,
                artifact=located.pages,
                solver_path=SolverPath.FALLBACK_NO_PYTHON,
                fallback_no_python=True,
                python_retry_count=retry_count,
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
                solver_path=SolverPath.FALLBACK_NO_PYTHON,
                fallback_no_python=True,
                python_retry_count=retry_count,
            )


class Stage2Router:
    """Dispatches Stage 2 Questions by normalized QuestionType to specialized solvers."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        validator: AnswerValidator | None = None,
        locator: TableLocator | None = None,
        sandbox: PythonSandbox | None = None,
    ) -> None:
        shared_validator = validator or AnswerValidator(gateway=gateway)
        shared_locator = locator or TableLocator()
        self._extract_solver = ExtractSolver(
            gateway=gateway,
            validator=shared_validator,
            locator=shared_locator,
        )
        self._thinking_solver = ThinkingSolver(
            gateway=gateway,
            validator=shared_validator,
            locator=shared_locator,
            sandbox=sandbox,
        )
        self._structure_solver = StructureSolver(
            gateway=gateway,
            validator=shared_validator,
            locator=shared_locator,
        )

    def solve(
        self,
        question: Question,
        artifacts: PageArtifact | Sequence[PageArtifact],
        *,
        preserved_pages: Sequence[PreparedPage] = (),
    ) -> QuestionResult:
        """Route a Question to extract, thinking, or structure Stage 2 solver."""
        if question.question_type == QuestionType.EXTRACT:
            return self._extract_solver.solve(question, artifacts)
        if question.question_type == QuestionType.THINKING:
            return self._thinking_solver.solve(question, artifacts)
        if question.question_type == QuestionType.STRUCTURE:
            return self._structure_solver.solve(
                question,
                artifacts,
                preserved_pages=preserved_pages,
            )
        return QuestionResult(
            question_id=question.id,
            document_key=question.document_key,
            question_type=question.question_type,
            answer_format=question.answer_format,
            status="failed",
            error=f"Unsupported question_type={question.question_type!r}",
        )
