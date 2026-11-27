"""Stage 2 target TableGrid locator and cross-page continuation assembly (ADR-0004)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from tianchi_table_recognition.models import PageArtifact, Question, TableGrid
from tianchi_table_recognition.text_matching import (
    extract_search_tokens,
    extract_table_number,
    has_continuation_marker,
    has_forward_continuation_marker,
    normalize_base_title,
    normalize_text_for_match,
    strip_table_number_prefix,
    token_overlap_score,
)

_MAX_CONTINUATION_STEPS = 4


class LocatedTable(BaseModel):
    """Single TableGrid selected by TableLocator with its page index and score."""

    model_config = ConfigDict(frozen=True)

    page_index: int
    table: TableGrid
    score: float
    is_continuation: bool = False


class LocatedContext(BaseModel):
    """Page-ordered Stage 2 context assembled without physically merging Stage 1 grids."""

    model_config = ConfigDict(frozen=True)

    pages: tuple[PageArtifact, ...]
    tables: tuple[LocatedTable, ...]
    has_cross_page_continuation: bool = False


def _header_cell_texts(table: TableGrid) -> frozenset[str]:
    header_rows = max(1, table.header_row_count)
    texts: set[str] = set()
    for cell in table.cells:
        if cell.row < header_rows and cell.text:
            norm = normalize_text_for_match(cell.text)
            if norm:
                texts.add(norm)
    return frozenset(texts)


def _headers_match(prev_tbl: TableGrid, next_tbl: TableGrid) -> bool:
    prev_headers = _header_cell_texts(prev_tbl)
    next_headers = _header_cell_texts(next_tbl)
    if not prev_headers or not next_headers:
        return False
    union_size = len(prev_headers | next_headers)
    if union_size == 0:
        return False
    return (len(prev_headers & next_headers) / union_size) >= 0.7


def _can_link_tables(
    prev_art: PageArtifact,
    prev_tbl: TableGrid,
    next_art: PageArtifact,
    next_tbl: TableGrid,
) -> bool:
    if next_art.page_index != prev_art.page_index + 1:
        return False
    if not prev_art.tables or not next_art.tables:
        return False
    if prev_art.tables[-1].table_id != prev_tbl.table_id:
        return False
    if next_art.tables[0].table_id != next_tbl.table_id:
        return False

    explicit_next = has_continuation_marker(next_tbl.title or "") or has_continuation_marker(
        next_art.page_context_text
    )
    explicit_prev = has_forward_continuation_marker(
        prev_tbl.title or ""
    ) or has_forward_continuation_marker(prev_art.page_context_text)
    if explicit_next or explicit_prev:
        prev_num = extract_table_number(prev_tbl.title)
        next_num = extract_table_number(next_tbl.title)
        if prev_num and next_num and prev_num != next_num:
            return False
        prev_desc = strip_table_number_prefix(prev_tbl.title)
        next_desc = strip_table_number_prefix(next_tbl.title)
        if prev_desc and next_desc and prev_desc != next_desc:
            return False
        return True

    prev_base = normalize_base_title(prev_tbl.title)
    next_base = normalize_base_title(next_tbl.title)
    if prev_base and next_base and prev_base != next_base:
        return False

    if prev_base and prev_base == next_base:
        return True

    if prev_tbl.col_count >= 2 and prev_tbl.col_count == next_tbl.col_count:
        if next_tbl.header_row_count == 0 or _headers_match(prev_tbl, next_tbl):
            return True

    return False


def _score_table(
    question: Question,
    artifact: PageArtifact,
    table: TableGrid,
) -> float:
    title_text = table.title or ""
    header_rows = max(1, table.header_row_count)
    header_text = " ".join(
        cell.text for cell in table.cells if cell.row < header_rows and cell.text
    )
    body_text = " ".join(
        cell.text for cell in table.cells if cell.row >= header_rows and cell.text
    )
    context_text = artifact.page_context_text

    title_tokens = extract_search_tokens(title_text)
    header_tokens = extract_search_tokens(header_text)
    body_tokens = extract_search_tokens(body_text)
    context_tokens = extract_search_tokens(context_text)

    score = 0.0
    if question.table_hint:
        hint_norm = normalize_text_for_match(question.table_hint)
        hint_base = normalize_base_title(question.table_hint)
        hint_tokens = extract_search_tokens(question.table_hint)
        title_norm = normalize_text_for_match(title_text)
        title_base = normalize_base_title(title_text)

        if hint_norm and title_norm and (hint_norm in title_norm or title_norm in hint_norm):
            score += 10.0
        elif hint_base and title_base and (hint_base == title_base or hint_base in title_base):
            score += 9.0
        elif hint_norm and hint_norm in normalize_text_for_match(context_text):
            score += 5.0

        score += token_overlap_score(hint_tokens, title_tokens) * 3.0
        score += token_overlap_score(hint_tokens, header_tokens) * 2.0
        score += token_overlap_score(hint_tokens, context_tokens) * 1.5

    q_tokens = extract_search_tokens(question.question)
    score += token_overlap_score(q_tokens, title_tokens) * 2.5
    score += token_overlap_score(q_tokens, header_tokens) * 2.0
    score += token_overlap_score(q_tokens, body_tokens) * 1.5
    score += token_overlap_score(q_tokens, context_tokens) * 0.8
    return score


class TableLocator:
    """Selects relevant TableGrids for a Question and links cross-page continuation pages."""

    def locate(
        self,
        question: Question,
        artifacts: Sequence[PageArtifact],
    ) -> LocatedContext:
        """Rank PageArtifacts/TableGrids by relevance and link adjacent continuations."""
        ordered_pages = tuple(sorted(artifacts, key=lambda art: art.page_index))
        if not ordered_pages:
            return LocatedContext(pages=(), tables=(), has_cross_page_continuation=False)

        by_page = {art.page_index: art for art in ordered_pages}
        scored: list[tuple[float, int, int, PageArtifact, TableGrid]] = []
        for art in ordered_pages:
            for tbl_pos, table in enumerate(art.tables):
                scored.append(
                    (_score_table(question, art, table), art.page_index, tbl_pos, art, table)
                )

        if not scored:
            return LocatedContext(
                pages=ordered_pages,
                tables=(),
                has_cross_page_continuation=False,
            )

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        top_score = scored[0][0]
        if top_score > 0.0:
            primaries = [
                (art, tbl, score)
                for score, _p, _pos, art, tbl in scored
                if score >= top_score * 0.65 and score > 0.0
            ][:2]
        else:
            first_score, _p, _pos, first_art, first_tbl = scored[0]
            primaries = [(first_art, first_tbl, first_score)]

        selected_by_key: dict[tuple[int, str], tuple[PageArtifact, TableGrid, float]] = {}
        linked_continuations: set[tuple[int, str]] = set()

        for art, tbl, score in primaries:
            selected_by_key.setdefault((art.page_index, tbl.table_id), (art, tbl, score))

            # Expand forward across strictly consecutive pages
            curr_art, curr_tbl = art, tbl
            for _ in range(_MAX_CONTINUATION_STEPS):
                next_art = by_page.get(curr_art.page_index + 1)
                if next_art is None or not next_art.tables:
                    break
                next_tbl = next_art.tables[0]
                if not _can_link_tables(curr_art, curr_tbl, next_art, next_tbl):
                    break
                next_key = (next_art.page_index, next_tbl.table_id)
                selected_by_key.setdefault(next_key, (next_art, next_tbl, score))
                linked_continuations.add(next_key)
                curr_art, curr_tbl = next_art, next_tbl

            # Expand backward across strictly consecutive pages
            curr_art, curr_tbl = art, tbl
            for _ in range(_MAX_CONTINUATION_STEPS):
                prev_art = by_page.get(curr_art.page_index - 1)
                if prev_art is None or not prev_art.tables:
                    break
                prev_tbl = prev_art.tables[-1]
                if not _can_link_tables(prev_art, prev_tbl, curr_art, curr_tbl):
                    break
                prev_key = (prev_art.page_index, prev_tbl.table_id)
                selected_by_key.setdefault(prev_key, (prev_art, prev_tbl, score))
                linked_continuations.add((curr_art.page_index, curr_tbl.table_id))
                curr_art, curr_tbl = prev_art, prev_tbl

        ordered_items = sorted(
            selected_by_key.items(),
            key=lambda item: (item[0][0], next(
                (i for i, t in enumerate(item[1][0].tables) if t.table_id == item[0][1]),
                0,
            )),
        )
        located_tables = tuple(
            LocatedTable(
                page_index=art.page_index,
                table=tbl,
                score=score,
                is_continuation=key in linked_continuations,
            )
            for key, (art, tbl, score) in ordered_items
        )
        selected_page_indices = {lt.page_index for lt in located_tables}
        selected_pages = tuple(
            art for art in ordered_pages if art.page_index in selected_page_indices
        )
        return LocatedContext(
            pages=selected_pages,
            tables=located_tables,
            has_cross_page_continuation=any(lt.is_continuation for lt in located_tables),
        )


class LocatedPromptContext(BaseModel):
    """Serialized Stage 2 prompt/context bundle built from a LocatedContext."""

    model_config = ConfigDict(frozen=True)

    page_context: str
    tables_payload: list[dict[str, Any]]
    has_cross_page_continuation: bool

    def format_user_prompt(
        self,
        question: Question,
        *,
        extra_lines: Sequence[str] = (),
    ) -> str:
        """Render a compliant DocumentKey-only Stage 2 user prompt."""
        cross_flag = "true" if self.has_cross_page_continuation else "false"
        lines = [
            f"DocumentKey: {question.document_key}",
            f"Question: {question.question}",
            f"TableHint: {question.table_hint or ''}",
            f"AnswerFormat: {question.answer_format.value}",
            *extra_lines,
            f"CrossPageContinuation: {cross_flag}",
            f"PageContext: {self.page_context}",
            f"TablesJSON: {json.dumps(self.tables_payload, ensure_ascii=False)}",
        ]
        return "\n".join(lines)


def build_located_payload(located: LocatedContext) -> LocatedPromptContext:
    """Serialize located PageArtifacts and TableGrids into Stage 2 prompt/context payloads."""
    page_context = "\n".join(
        f"[page {page.page_index}] {page.page_context_text}"
        for page in located.pages
        if page.page_context_text
    )
    tables_payload = [
        {
            "page_index": item.page_index,
            "is_continuation": item.is_continuation,
            **item.table.model_dump(mode="json"),
        }
        for item in located.tables
    ]
    return LocatedPromptContext(
        page_context=page_context,
        tables_payload=tables_payload,
        has_cross_page_continuation=located.has_cross_page_continuation,
    )
