"""Core domain models aligned with GLOSSARY.md."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QuestionType(StrEnum):
    """Supported evaluation question categories."""

    EXTRACT = "extract"
    THINKING = "thinking"
    STRUCTURE = "structure"


class AnswerFormat(StrEnum):
    """Expected output contract for a Question."""

    STRING = "string"
    NUMBER = "number"
    JSON = "json"
    JSON_ARRAY = "json_array"


class ModelRole(StrEnum):
    """Logical inference capabilities mapped to compliant Qwen models."""

    VISION_EXTRACT = "vision_extract"
    VISION_VERIFY = "vision_verify"
    VISION_TRIAGE = "vision_triage"
    TEXT_REASON = "text_reason"


class TriageMode(StrEnum):
    """Stage 1 document triage/extraction mode."""

    SINGLE_IMAGE = "single_image"
    SHORT_DOCUMENT = "short_document"
    NATIVE_TEXT = "native_text"
    VISION_TRIAGE = "vision_triage"


class Question(BaseModel):
    """Single evaluation item stripped of original Source Document filenames."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_key: str
    question_type: QuestionType
    question: str
    table_hint: str | None = None
    answer_format: AnswerFormat


class NormalizationWarning(BaseModel):
    """Structured warning emitted when an out-of-enum question_type is normalized."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    raw_question_type: str
    normalized_question_type: QuestionType
    reason: str


class SourceDocumentRef(BaseModel):
    """Local mapping from anonymous DocumentKey back to Source Document on disk.

    Kept strictly out of model prompts, transport payloads, and PageArtifact caches.
    """

    model_config = ConfigDict(frozen=True)

    document_key: str
    document_hash: str
    source_path: Path
    file_name: str
    media_type: str


class TableCell(BaseModel):
    """Single cell within a TableGrid."""

    model_config = ConfigDict(frozen=True)

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    text: str | None = None
    bbox: list[float] | None = None
    uncertain: bool = False


class TableGrid(BaseModel):
    """Structured grid representation of a single table on a page."""

    model_config = ConfigDict(frozen=True)

    table_id: str
    title: str | None = None
    row_count: int = Field(ge=0)
    col_count: int = Field(ge=0)
    header_row_count: int = Field(default=1, ge=0)
    cells: list[TableCell] = Field(default_factory=list)


class PageArtifact(BaseModel):
    """Cached stage-1 extraction output for a single page of a Source Document."""

    model_config = ConfigDict(frozen=True)

    document_key: str
    document_hash: str
    page_index: int = Field(default=0, ge=0)
    page_context_text: str = ""
    tables: list[TableGrid] = Field(default_factory=list)
    stage: str
    stage_version: str
    model_id: str
    prompt_hash: str
    render_dpi: int | None = None


class QuestionResult(BaseModel):
    """Per-Question execution outcome within a Run."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    document_key: str
    question_type: QuestionType
    answer_format: AnswerFormat
    status: Literal["ok", "failed"]
    answer: str | None = None
    error: str | None = None
    raw_answer: str | None = None
    validation_errors: tuple[str, ...] = ()
    json_repaired: bool = False
