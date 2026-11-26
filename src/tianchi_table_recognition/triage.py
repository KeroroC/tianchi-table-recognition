"""Lightweight Page Triage for long multi-page PDFs (native text index + vision_triage fallback)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tianchi_table_recognition.gateway import ComplianceError, ModelGateway
from tianchi_table_recognition.io_utils import atomic_write_text
from tianchi_table_recognition.models import ModelRole, Question, SourceDocumentRef, TriageMode
from tianchi_table_recognition.preprocessing import OriginalPageStore, PdfInspection
from tianchi_table_recognition.text_matching import (
    extract_search_tokens,
    has_continuation_marker,
    has_forward_continuation_marker,
    normalize_base_title,
    normalize_text_for_match,
    strip_json_fences,
    token_overlap_score,
)

DEFAULT_TRIAGE_SYSTEM_PROMPT = (
    "You are a fast document page indexing assistant. "
    "For each page thumbnail provided, identify whether it contains a table, "
    "any table titles, whether it continues a table from the previous page, "
    "or continues to the next page. Output strict valid JSON only."
)

DEFAULT_TRIAGE_USER_PROMPT = (
    "Return JSON with key 'pages' as a list of objects, one per page_index, each having: "
    "'page_index' (int), 'summary' (short string), 'table_titles' (list of strings), "
    "'has_table' (bool), 'continues_from_previous' (bool), 'continues_to_next' (bool)."
)

_TITLE_CUE_RE = re.compile(r"(表|续|table|schedule|statement)", re.IGNORECASE)
_TABLE_TEXT_CUE_RE = re.compile(r"(\d|合计|总计|小计|项目|金额|\||\t)")
_WHITESPACE_RUN_RE = re.compile(r"\s+")

_TRIAGE_STAGE_NAME = "stage1_page_triage"
_TRIAGE_STAGE_VERSION = "1.0"
_MAX_CONTINUATION_STEPS = 4


class TriageError(RuntimeError):
    """Raised when vision_triage output cannot be parsed into valid page index entries."""


class PageIndexEntry(BaseModel):
    """Lightweight page-level index record used during Page Triage."""

    model_config = ConfigDict(frozen=True)

    page_index: int = Field(ge=0)
    summary: str = ""
    table_titles: tuple[str, ...] = ()
    has_table: bool = False
    continues_from_previous: bool = False
    continues_to_next: bool = False


class TriageResult(BaseModel):
    """Candidate page selection output from Page Triage."""

    model_config = ConfigDict(frozen=True)

    document_key: str
    page_count: int = Field(ge=1)
    mode: TriageMode
    entries: tuple[PageIndexEntry, ...] = ()
    candidate_pages: tuple[int, ...]
    per_question_pages: dict[str, tuple[int, ...]] = Field(default_factory=dict)


@dataclass(frozen=True)
class TriageCacheKey:
    """Deterministic cache key isolating vision_triage index caches."""

    document_hash: str
    stage: str
    stage_version: str
    model_id: str
    prompt_hash: str
    thumbnail_dpi: int

    def digest(self) -> str:
        """Compute deterministic SHA-256 hex digest of all triage cache key components."""
        material = json.dumps(
            {
                "document_hash": self.document_hash,
                "stage": self.stage,
                "stage_version": self.stage_version,
                "model_id": self.model_id,
                "prompt_hash": self.prompt_hash,
                "thumbnail_dpi": self.thumbnail_dpi,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def matches(self, payload: _TriageIndexCachePayload, page_count: int) -> bool:
        """Return True if cached triage payload matches every key component and page count."""
        return (
            payload.document_hash == self.document_hash
            and payload.stage == self.stage
            and payload.stage_version == self.stage_version
            and payload.model_id == self.model_id
            and payload.prompt_hash == self.prompt_hash
            and payload.thumbnail_dpi == self.thumbnail_dpi
            and len(payload.entries) == page_count
        )


class _TriageIndexCachePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_key: str
    document_hash: str
    stage: str
    stage_version: str
    model_id: str
    prompt_hash: str
    thumbnail_dpi: int
    entries: tuple[PageIndexEntry, ...]


def _build_native_text_entries(page_texts: Sequence[str]) -> tuple[PageIndexEntry, ...]:
    entries: list[PageIndexEntry] = []
    for page_index, raw_text in enumerate(page_texts):
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        titles: list[str] = []
        for line in lines:
            if len(line) <= 80 and (
                _TITLE_CUE_RE.search(line) or has_continuation_marker(line)
            ):
                if line not in titles:
                    titles.append(line)
        summary = _WHITESPACE_RUN_RE.sub(" ", raw_text).strip()[:400]
        has_table = bool(titles) or (
            len(lines) >= 2 and bool(_TABLE_TEXT_CUE_RE.search(raw_text))
        )
        entries.append(
            PageIndexEntry(
                page_index=page_index,
                summary=summary,
                table_titles=tuple(titles),
                has_table=has_table,
                continues_from_previous=has_continuation_marker(raw_text),
                continues_to_next=has_forward_continuation_marker(raw_text),
            )
        )
    return tuple(entries)


def _parse_triage_batch_response(
    raw_content: str,
    expected_indices: Sequence[int],
) -> list[PageIndexEntry]:
    cleaned = strip_json_fences(raw_content)
    try:
        parsed: Any = json.loads(cleaned)
    except Exception as exc:
        raise TriageError(f"Failed to parse vision_triage JSON response: {exc}") from exc

    raw_items: Any = parsed.get("pages") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_items, list):
        raise TriageError("vision_triage JSON must be a list or an object containing 'pages' list")

    by_index: dict[int, PageIndexEntry] = {}
    for item in raw_items:
        try:
            entry = PageIndexEntry.model_validate(item)
        except Exception as exc:
            raise TriageError(f"Invalid PageIndexEntry in vision_triage response: {exc}") from exc
        if entry.page_index in expected_indices:
            by_index[entry.page_index] = entry

    return [
        by_index.get(idx, PageIndexEntry(page_index=idx))
        for idx in expected_indices
    ]


def _triage_cache_path(cache_dir: Path, key: TriageCacheKey) -> Path:
    return cache_dir / "triage_index" / f"{key.document_hash[:16]}_{key.digest()}.json"


def _build_vision_triage_entries(
    *,
    doc_ref: SourceDocumentRef,
    page_count: int,
    gateway: ModelGateway,
    page_store: OriginalPageStore,
    thumbnail_dpi: int,
    cache_dir: Path | None,
    batch_size: int,
) -> tuple[PageIndexEntry, ...]:
    model_id = gateway.model_for_role(ModelRole.VISION_TRIAGE)
    prompt_material = f"{DEFAULT_TRIAGE_SYSTEM_PROMPT}\n---\n{DEFAULT_TRIAGE_USER_PROMPT}"
    prompt_hash = hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()[:16]
    cache_key = TriageCacheKey(
        document_hash=doc_ref.document_hash,
        stage=_TRIAGE_STAGE_NAME,
        stage_version=_TRIAGE_STAGE_VERSION,
        model_id=model_id,
        prompt_hash=prompt_hash,
        thumbnail_dpi=thumbnail_dpi,
    )

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _triage_cache_path(cache_dir, cache_key)
        if cache_path.is_file():
            cached = _TriageIndexCachePayload.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
            if cache_key.matches(cached, page_count):
                return cached.entries

    all_entries: list[PageIndexEntry] = []
    step = max(1, batch_size)
    for start in range(0, page_count, step):
        batch_indices = list(range(start, min(start + step, page_count)))
        thumbnails = [
            page_store.render_pdf_thumbnail(doc_ref, idx, dpi=thumbnail_dpi)
            for idx in batch_indices
        ]
        user_prompt = (
            f"DocumentKey: {doc_ref.document_key}\n"
            f"page_indices: {json.dumps(batch_indices)}\n"
            f"{DEFAULT_TRIAGE_USER_PROMPT}"
        )
        try:
            response = gateway.invoke(
                model_role=ModelRole.VISION_TRIAGE,
                system_prompt=DEFAULT_TRIAGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=thumbnails,
                stage=_TRIAGE_STAGE_NAME,
                document_key=doc_ref.document_key,
            )
        except ComplianceError:
            raise
        batch_entries = _parse_triage_batch_response(response.content, batch_indices)
        all_entries.extend(batch_entries)

    ordered = tuple(sorted(all_entries, key=lambda e: e.page_index))
    if cache_path is not None:
        payload = _TriageIndexCachePayload(
            document_key=doc_ref.document_key,
            document_hash=doc_ref.document_hash,
            stage=_TRIAGE_STAGE_NAME,
            stage_version=_TRIAGE_STAGE_VERSION,
            model_id=model_id,
            prompt_hash=prompt_hash,
            thumbnail_dpi=thumbnail_dpi,
            entries=ordered,
        )
        atomic_write_text(cache_path, payload.model_dump_json(indent=2))
    return ordered


def _score_entry_for_question(entry: PageIndexEntry, question: Question) -> float:
    titles_text = " ".join(entry.table_titles)
    title_tokens = extract_search_tokens(titles_text)
    summary_tokens = extract_search_tokens(entry.summary)

    score = 0.0
    if question.table_hint:
        hint_norm = normalize_text_for_match(question.table_hint)
        hint_base = normalize_base_title(question.table_hint)
        hint_tokens = extract_search_tokens(question.table_hint)

        for raw_title in entry.table_titles:
            t_norm = normalize_text_for_match(raw_title)
            t_base = normalize_base_title(raw_title)
            if hint_norm and (hint_norm in t_norm or t_norm in hint_norm):
                score += 8.0
                break
            if hint_base and t_base and (hint_base == t_base or hint_base in t_base):
                score += 7.0
                break
        if hint_norm and hint_norm in normalize_text_for_match(entry.summary):
            score += 4.0

        score += token_overlap_score(hint_tokens, title_tokens) * 3.0
        score += token_overlap_score(hint_tokens, summary_tokens) * 1.5

    q_tokens = extract_search_tokens(question.question)
    score += token_overlap_score(q_tokens, title_tokens) * 2.0
    score += token_overlap_score(q_tokens, summary_tokens) * 1.0

    if score > 0.0 and entry.has_table:
        score += 0.1
    return score


def _shared_base_title(prev_entry: PageIndexEntry, next_entry: PageIndexEntry) -> bool:
    prev_bases = {normalize_base_title(t) for t in prev_entry.table_titles} - {""}
    next_bases = {normalize_base_title(t) for t in next_entry.table_titles} - {""}
    return bool(prev_bases & next_bases)


def _entry_has_continuation_cue(entry: PageIndexEntry) -> bool:
    return (
        entry.continues_from_previous
        or has_continuation_marker(entry.summary)
        or any(has_continuation_marker(t) for t in entry.table_titles)
    )


def _can_continue_forward(prev_entry: PageIndexEntry, next_entry: PageIndexEntry) -> bool:
    if prev_entry.continues_to_next:
        return True
    if _entry_has_continuation_cue(next_entry):
        return True
    if _shared_base_title(prev_entry, next_entry):
        return True
    if prev_entry.has_table and next_entry.has_table and not next_entry.table_titles:
        return True
    return False


def _can_continue_backward(prev_entry: PageIndexEntry, curr_entry: PageIndexEntry) -> bool:
    if _entry_has_continuation_cue(curr_entry):
        return True
    if prev_entry.continues_to_next:
        return True
    if _shared_base_title(prev_entry, curr_entry):
        return True
    return False


def select_candidate_pages(
    entries: Sequence[PageIndexEntry],
    questions: Sequence[Question],
) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]]]:
    """Select candidate seed pages per Question and expand adjacent continuation pages."""
    if not entries:
        return (0,), {q.id: (0,) for q in questions}

    by_idx = {e.page_index: e for e in entries}
    min_idx = min(by_idx)
    max_idx = max(by_idx)

    if not questions:
        table_pages = tuple(e.page_index for e in entries if e.has_table)
        if table_pages:
            return table_pages, {}
        return tuple(e.page_index for e in entries), {}

    table_entries = [e for e in entries if e.has_table]
    candidate_pool = table_entries if table_entries else list(entries)

    per_question: dict[str, tuple[int, ...]] = {}
    union_pages: set[int] = set()

    for question in questions:
        scored = sorted(
            ((_score_entry_for_question(e, question), e.page_index) for e in candidate_pool),
            key=lambda item: (-item[0], item[1]),
        )
        max_score = scored[0][0]
        if max_score > 0.0:
            seeds = [
                idx for score, idx in scored if score >= max_score * 0.6 and score > 0.0
            ][:2]
        elif table_entries:
            seeds = [e.page_index for e in table_entries[:3]]
        else:
            seeds = [entries[0].page_index]

        selected: set[int] = set(seeds)
        for seed in seeds:
            # Expand forward
            curr = seed
            for _ in range(_MAX_CONTINUATION_STEPS):
                nxt = curr + 1
                if nxt > max_idx or curr not in by_idx or nxt not in by_idx:
                    break
                if not _can_continue_forward(by_idx[curr], by_idx[nxt]):
                    break
                selected.add(nxt)
                curr = nxt

            # Expand backward
            curr = seed
            for _ in range(_MAX_CONTINUATION_STEPS):
                prv = curr - 1
                if prv < min_idx or curr not in by_idx or prv not in by_idx:
                    break
                if not _can_continue_backward(by_idx[prv], by_idx[curr]):
                    break
                selected.add(prv)
                curr = prv

        ordered_q = tuple(sorted(selected))
        per_question[question.id] = ordered_q
        union_pages.update(ordered_q)

    return tuple(sorted(union_pages)), per_question


def triage_long_pdf(
    *,
    doc_ref: SourceDocumentRef,
    inspection: PdfInspection,
    questions: Sequence[Question],
    gateway: ModelGateway,
    page_store: OriginalPageStore,
    thumbnail_dpi: int = 72,
    cache_dir: Path | None = None,
    batch_size: int = 4,
) -> TriageResult:
    """Run Page Triage on a long PDF via native text index or vision_triage fallback."""
    if inspection.has_native_text_layer:
        mode = TriageMode.NATIVE_TEXT
        entries = _build_native_text_entries(inspection.page_texts)
    else:
        mode = TriageMode.VISION_TRIAGE
        entries = _build_vision_triage_entries(
            doc_ref=doc_ref,
            page_count=inspection.page_count,
            gateway=gateway,
            page_store=page_store,
            thumbnail_dpi=thumbnail_dpi,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

    candidate_pages, per_question = select_candidate_pages(entries, questions)
    return TriageResult(
        document_key=doc_ref.document_key,
        page_count=inspection.page_count,
        mode=mode,
        entries=entries,
        candidate_pages=candidate_pages,
        per_question_pages=per_question,
    )
