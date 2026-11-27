"""Stage 2 structure solver: deterministic TableGrid counting with vision_verify fallback."""

from __future__ import annotations

import re
from collections.abc import Sequence

from tianchi_table_recognition.gateway import ComplianceError, EncodedImage, ModelGateway
from tianchi_table_recognition.locator import LocatedContext, TableLocator, build_located_payload
from tianchi_table_recognition.models import (
    AnswerFormat,
    ModelRole,
    PageArtifact,
    Question,
    QuestionResult,
    QuestionType,
    SolverPath,
    TableGrid,
)
from tianchi_table_recognition.preprocessing import PreparedPage, encode_prepared_page
from tianchi_table_recognition.text_matching import (
    has_continuation_marker,
    has_forward_continuation_marker,
)
from tianchi_table_recognition.validator import AnswerValidator, validate_and_build_result

DEFAULT_STRUCTURE_VISION_SYSTEM_PROMPT = (
    "You are an accurate table structure verification assistant. Inspect the original "
    "page image(s) together with the extracted TableGrid metadata. Unless the question "
    "explicitly excludes headers, header rows MUST be counted in the total row count "
    "(表头行必须计入总行数). Account carefully for merged cells and cross-page "
    "continuations. Return only the final answer value with no explanation or markdown fences."
)

_ROW_QUESTION_RE = re.compile(
    r"(?:总行数|行数|多少行|几行|共有.*行|row\s*count|how\s+many\s+rows|number\s+of\s+rows)",
    re.IGNORECASE,
)
_COL_QUESTION_RE = re.compile(
    r"(?:总列数|列数|多少列|几列|共有.*列|col(?:umn)?\s*count|"
    r"how\s+many\s+col(?:umn)?s|number\s+of\s+col(?:umn)?s)",
    re.IGNORECASE,
)
_EXCLUDE_HEADER_RE = re.compile(
    r"(?:不含表头|不包括表头|除去?表头|除表头外|不计表头|without\s+header)",
    re.IGNORECASE,
)
_DATA_ROW_RE = re.compile(r"(?:数据行数|正文行数|内容行数|data\s+rows?)", re.IGNORECASE)
_HEADER_ROW_RE = re.compile(
    r"(?:表头(?:共有|有|占|包含)?(?:多少|几)行|表头行数|header\s+rows?)",
    re.IGNORECASE,
)
_MERGED_CELL_RE = re.compile(
    r"(?:合并单元格.*(?:多少|几|数量|个数)|(?:多少|几)个?合并单元格|merged\s+cells?)",
    re.IGNORECASE,
)


def _cells_cover_grid(table: TableGrid) -> bool:
    if not table.cells or table.row_count <= 0 or table.col_count <= 0:
        return False
    occupied: set[tuple[int, int]] = set()
    for cell in table.cells:
        if cell.row + cell.rowspan > table.row_count or cell.col + cell.colspan > table.col_count:
            return False
        for r_idx in range(cell.row, cell.row + cell.rowspan):
            for c_idx in range(cell.col, cell.col + cell.colspan):
                if (r_idx, c_idx) in occupied:
                    return False
                occupied.add((r_idx, c_idx))
    return len(occupied) == table.row_count * table.col_count


def _detect_fallback_reasons(located: LocatedContext) -> tuple[str, ...]:
    if not located.tables:
        return ("no_table_located",)

    reasons: list[str] = []
    has_marker = any(
        has_continuation_marker(lt.table.title or "")
        or has_forward_continuation_marker(lt.table.title or "")
        for lt in located.tables
    ) or any(
        has_continuation_marker(page.page_context_text)
        or has_forward_continuation_marker(page.page_context_text)
        for page in located.pages
    )
    if located.has_cross_page_continuation or has_marker:
        reasons.append("cross_page_continuation")
    elif len(located.tables) > 1:
        reasons.append("multiple_candidate_tables")

    tbl = located.tables[0].table
    if tbl.row_count <= 0 or tbl.col_count <= 0:
        reasons.append("empty_table_dimensions")
    if tbl.header_row_count > tbl.row_count:
        reasons.append("invalid_header_row_count")
    if any(cell.uncertain for cell in tbl.cells):
        reasons.append("uncertain_cell_or_boundary")
    if not tbl.cells:
        reasons.append("missing_table_cells")
    elif not _cells_cover_grid(tbl):
        reasons.append("inconsistent_cell_bounds")

    return tuple(reasons)


def _compute_deterministic_structure_answer(
    question: Question,
    table: TableGrid,
) -> str | None:
    if question.answer_format not in (AnswerFormat.NUMBER, AnswerFormat.STRING):
        return None

    q_text = question.question
    if _MERGED_CELL_RE.search(q_text):
        merged_count = sum(1 for cell in table.cells if cell.rowspan > 1 or cell.colspan > 1)
        return str(merged_count)

    asks_rows = bool(_ROW_QUESTION_RE.search(q_text))
    asks_cols = bool(_COL_QUESTION_RE.search(q_text))
    if asks_rows and asks_cols:
        return None

    if _DATA_ROW_RE.search(q_text) or (_EXCLUDE_HEADER_RE.search(q_text) and asks_rows):
        return str(table.row_count - table.header_row_count)

    if _HEADER_ROW_RE.search(q_text):
        return str(table.header_row_count)

    if asks_rows:
        return str(table.row_count)

    if asks_cols:
        return str(table.col_count)

    return None


def _select_vision_images(
    located: LocatedContext,
    preserved_pages: Sequence[PreparedPage],
) -> list[EncodedImage]:
    if not preserved_pages:
        return []
    by_page_idx = {page.page_index: page for page in preserved_pages}
    target_indices = [page.page_index for page in located.pages]
    if target_indices:
        matched = [by_page_idx[idx] for idx in target_indices if idx in by_page_idx]
    else:
        matched = sorted(preserved_pages, key=lambda page: page.page_index)
    return [encode_prepared_page(page) for page in matched]


class StructureSolver:
    """Stage 2 solver for structure Questions with deterministic counting & vision fallback."""

    STAGE_NAME = "stage2_structure_vision_verify"

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        system_prompt: str = DEFAULT_STRUCTURE_VISION_SYSTEM_PROMPT,
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
        *,
        preserved_pages: Sequence[PreparedPage] = (),
    ) -> QuestionResult:
        """Solve a structure Question deterministically from TableGrid or via VISION_VERIFY."""
        if question.question_type != QuestionType.STRUCTURE:
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=(
                    f"Unsupported question_type={question.question_type.value!r} "
                    "for StructureSolver."
                ),
                solver_path=SolverPath.STRUCTURE_DETERMINISTIC,
            )

        artifact_list = [artifacts] if isinstance(artifacts, PageArtifact) else list(artifacts)
        located = self._locator.locate(question, artifact_list)
        fallback_reasons = _detect_fallback_reasons(located)

        if not fallback_reasons:
            det_answer = _compute_deterministic_structure_answer(
                question, located.tables[0].table
            )
            if det_answer is not None:
                return validate_and_build_result(
                    validator=self._validator,
                    question=question,
                    raw_answer=det_answer,
                    artifact=located.pages,
                    solver_path=SolverPath.STRUCTURE_DETERMINISTIC,
                    vision_verified=False,
                )
            fallback_reasons = ("unrecognized_or_complex_structure_question",)

        encoded_images = _select_vision_images(located, preserved_pages)
        if not encoded_images:
            return QuestionResult(
                question_id=question.id,
                document_key=question.document_key,
                question_type=question.question_type,
                answer_format=question.answer_format,
                status="failed",
                error=(
                    "Missing preserved original page images for structure vision_verify fallback "
                    f"({','.join(fallback_reasons)})"
                ),
                solver_path=SolverPath.STRUCTURE_VISION_VERIFY,
                vision_verified=False,
            )

        prompt_context = build_located_payload(located)
        user_prompt = prompt_context.format_user_prompt(
            question,
            extra_lines=(f"FallbackReasons: {','.join(fallback_reasons)}",),
        )

        try:
            response = self._gateway.invoke(
                model_role=ModelRole.VISION_VERIFY,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                images=encoded_images,
                stage=self.STAGE_NAME,
                document_key=question.document_key,
                question_id=question.id,
            )
            return validate_and_build_result(
                validator=self._validator,
                question=question,
                raw_answer=response.content,
                artifact=located.pages,
                solver_path=SolverPath.STRUCTURE_VISION_VERIFY,
                vision_verified=True,
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
                solver_path=SolverPath.STRUCTURE_VISION_VERIFY,
                vision_verified=False,
            )
