"""Dataset loader, defensive question_type normalization, and DocumentKey registry."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import re
from collections.abc import Sequence
from pathlib import Path

from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict

from tianchi_table_recognition.models import (
    AnswerFormat,
    NormalizationWarning,
    Question,
    QuestionType,
    SourceDocumentRef,
)

logger = logging.getLogger(__name__)

REQUIRED_QUESTION_COLUMNS: tuple[str, ...] = (
    "id",
    "file_name",
    "question_type",
    "question",
    "table_hint",
    "answer_format",
)

_STRUCTURE_PATTERNS = re.compile(
    r"(多少行|多少列|几行|几列|行数|列数|总行数|总列数|表头|合并单元格|表格结构|row_count|col_count)",
    re.IGNORECASE,
)
_THINKING_PATTERNS = re.compile(
    r"(计算|总和|合计|求和|汇总|平均|均值|占比|比例|增长率|同比|环比|差值|相差|多少倍|"
    r"净增|增幅|降幅|最大|最小|最高|最低|排名|前几|sum|average|ratio|percent)",
    re.IGNORECASE,
)
_COUNTING_PATTERNS = re.compile(
    r"(多少个|几个|多少家|几家|共有多少|一共有|数量是多少)",
    re.IGNORECASE,
)

_MEDIA_TYPE_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


class DatasetSchemaError(ValueError):
    """Raised when competition spreadsheets or referenced Source Documents are invalid."""


class LoadedDataset(BaseModel):
    """Loaded competition Questions, isolated DocumentKey registry, and normalization warnings."""

    model_config = ConfigDict(frozen=True)

    questions: list[Question]
    documents: dict[str, SourceDocumentRef]
    warnings: list[NormalizationWarning]


def _cell_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_question_type(
    question_id: str,
    raw_question_type: str,
    question_text: str,
    answer_format: AnswerFormat,
) -> tuple[QuestionType, NormalizationWarning | None]:
    candidate = raw_question_type.strip().lower()
    for qt in QuestionType:
        if candidate == qt.value:
            return qt, None

    if _STRUCTURE_PATTERNS.search(question_text):
        normalized = QuestionType.STRUCTURE
        reason = (
            f"Unrecognized question_type {raw_question_type!r}; "
            "matched structure keywords in question text."
        )
    elif _THINKING_PATTERNS.search(question_text):
        normalized = QuestionType.THINKING
        reason = (
            f"Unrecognized question_type {raw_question_type!r}; "
            "matched arithmetic/aggregation keywords in question text."
        )
    elif answer_format == AnswerFormat.NUMBER and _COUNTING_PATTERNS.search(question_text):
        normalized = QuestionType.THINKING
        reason = (
            f"Unrecognized question_type {raw_question_type!r}; "
            "matched counting keywords with answer_format='number'."
        )
    else:
        normalized = QuestionType.EXTRACT
        reason = (
            f"Unrecognized question_type {raw_question_type!r}; "
            f"defaulted to 'extract' for answer_format={answer_format.value!r}."
        )

    warning = NormalizationWarning(
        question_id=question_id,
        raw_question_type=raw_question_type,
        normalized_question_type=normalized,
        reason=reason,
    )
    logger.warning(
        "Normalized out-of-enum question_type for question_id=%s: %s -> %s (%s)",
        question_id,
        raw_question_type,
        normalized.value,
        reason,
    )
    return normalized, warning


def _parse_answer_format(raw_format: str, row_idx: int) -> AnswerFormat:
    candidate = raw_format.strip().lower()
    for fmt in AnswerFormat:
        if candidate == fmt.value:
            return fmt
    raise DatasetSchemaError(
        f"Invalid answer_format {raw_format!r} at row {row_idx}; "
        f"expected one of {[fmt.value for fmt in AnswerFormat]}."
    )


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MEDIA_TYPE_BY_SUFFIX:
        return _MEDIA_TYPE_BY_SUFFIX[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _register_source_document(
    documents_dir: Path,
    file_name: str,
    cache_by_path: dict[Path, SourceDocumentRef],
) -> SourceDocumentRef:
    source_path = (documents_dir / file_name).resolve()
    if source_path in cache_by_path:
        return cache_by_path[source_path]
    if not source_path.is_file():
        raise DatasetSchemaError(
            f"Referenced Source Document not found: {file_name!r} (resolved to {source_path})"
        )
    document_hash = _hash_file_sha256(source_path)
    document_key = f"doc_{document_hash[:16]}"
    ref = SourceDocumentRef(
        document_key=document_key,
        document_hash=document_hash,
        source_path=source_path,
        file_name=file_name,
        media_type=_infer_media_type(source_path),
    )
    cache_by_path[source_path] = ref
    return ref


def load_dataset(
    questions_path: Path,
    documents_dir: Path,
    selected_ids: Sequence[str] | None = None,
) -> LoadedDataset:
    """Load competition Questions and build an anonymized DocumentKey registry."""
    if not questions_path.is_file():
        raise DatasetSchemaError(f"Questions spreadsheet not found: {questions_path}")

    wb = load_workbook(questions_path, read_only=True, data_only=True)
    try:
        ws = wb.active
        if ws is None:
            raise DatasetSchemaError(f"Questions workbook has no active sheet: {questions_path}")
        rows_iter = ws.iter_rows(values_only=True)
        try:
            header_row = next(rows_iter)
        except StopIteration as exc:
            raise DatasetSchemaError(f"Questions workbook is empty: {questions_path}") from exc

        headers = [_cell_str(col) for col in header_row]
        col_index = {name: idx for idx, name in enumerate(headers) if name}
        missing_cols = [col for col in REQUIRED_QUESTION_COLUMNS if col not in col_index]
        if missing_cols:
            raise DatasetSchemaError(
                f"Questions workbook missing required columns {missing_cols}; found {headers}"
            )

        selected_set = {str(qid).strip() for qid in selected_ids} if selected_ids else None
        questions: list[Question] = []
        documents: dict[str, SourceDocumentRef] = {}
        path_cache: dict[Path, SourceDocumentRef] = {}
        warnings: list[NormalizationWarning] = []
        seen_ids: set[str] = set()

        for row_num, raw_row in enumerate(rows_iter, start=2):
            if raw_row is None or all(cell is None or _cell_str(cell) == "" for cell in raw_row):
                continue

            qid = _cell_str(raw_row[col_index["id"]])
            if not qid:
                raise DatasetSchemaError(f"Empty question id at row {row_num}")
            if qid in seen_ids:
                raise DatasetSchemaError(f"Duplicate question id {qid!r} at row {row_num}")
            seen_ids.add(qid)

            if selected_set is not None and qid not in selected_set:
                continue

            file_name = _cell_str(raw_row[col_index["file_name"]])
            if not file_name:
                raise DatasetSchemaError(
                    f"Empty file_name for question id={qid!r} at row {row_num}"
                )

            question_text = _cell_str(raw_row[col_index["question"]])
            if not question_text:
                raise DatasetSchemaError(f"Empty question text for id={qid!r} at row {row_num}")

            raw_qtype = _cell_str(raw_row[col_index["question_type"]])
            raw_hint = _cell_str(raw_row[col_index["table_hint"]])
            table_hint = raw_hint if raw_hint else None
            answer_format = _parse_answer_format(
                _cell_str(raw_row[col_index["answer_format"]]), row_num
            )
            question_type, warning = _normalize_question_type(
                qid, raw_qtype, question_text, answer_format
            )
            if warning is not None:
                warnings.append(warning)

            doc_ref = _register_source_document(documents_dir, file_name, path_cache)
            documents[doc_ref.document_key] = doc_ref

            questions.append(
                Question(
                    id=qid,
                    document_key=doc_ref.document_key,
                    question_type=question_type,
                    question=question_text,
                    table_hint=table_hint,
                    answer_format=answer_format,
                )
            )

        if selected_set is not None:
            found_ids = {q.id for q in questions}
            missing_selected = sorted(selected_set - found_ids)
            if missing_selected:
                raise DatasetSchemaError(
                    f"Requested --ids not found in questions sheet: {missing_selected}"
                )

        return LoadedDataset(questions=questions, documents=documents, warnings=warnings)
    finally:
        wb.close()


def load_submission_template(template_path: Path) -> list[str]:
    """Load ordered Question IDs from the competition Submission template."""
    if not template_path.is_file():
        raise DatasetSchemaError(f"Submission template not found: {template_path}")

    wb = load_workbook(template_path, read_only=True, data_only=True)
    try:
        ws = wb.active
        if ws is None:
            raise DatasetSchemaError(f"Submission template has no active sheet: {template_path}")
        rows_iter = ws.iter_rows(values_only=True)
        try:
            header_row = next(rows_iter)
        except StopIteration as exc:
            raise DatasetSchemaError(f"Submission template is empty: {template_path}") from exc

        headers = tuple(_cell_str(col) for col in header_row[:2])
        if headers != ("id", "answer"):
            raise DatasetSchemaError(
                f"Submission template header must start with ('id', 'answer'), got {headers}"
            )

        template_ids: list[str] = []
        for row_num, raw_row in enumerate(rows_iter, start=2):
            if raw_row is None or all(cell is None or _cell_str(cell) == "" for cell in raw_row):
                continue
            qid = _cell_str(raw_row[0])
            if not qid:
                raise DatasetSchemaError(f"Empty id in submission template at row {row_num}")
            template_ids.append(qid)
        return template_ids
    finally:
        wb.close()
