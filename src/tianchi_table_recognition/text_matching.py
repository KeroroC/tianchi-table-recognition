"""Shared text normalization, search tokenization, and cross-page continuation helpers."""

from __future__ import annotations

import re

_HAN_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]{2,}", re.IGNORECASE)
_MATCH_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)

_SEARCH_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "多少",
        "什么",
        "哪些",
        "请问",
        "是否",
        "根据",
        "怎么",
        "如何",
        "what",
        "which",
        "how",
        "the",
        "and",
        "for",
        "from",
    }
)

_BACKWARD_CONTINUATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"续表"),
    re.compile(r"[（(]\s*续\s*[)）]"),
    re.compile(r"续前页|续上页|接上页|承前页"),
    re.compile(r"\bcontinued\b|\bcont['’]?d\b|\(\s*cont\.?\s*\)", re.IGNORECASE),
)

_FORWARD_CONTINUATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"转下页|接下页|未完待续"),
    re.compile(r"continued\s+on\s+next\s+page", re.IGNORECASE),
)

_TITLE_STRIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"续表\s*\d*\s*[:：\-—]*"),
    re.compile(r"[（(]\s*续表?\s*\d*\s*[)）]"),
    re.compile(r"续前页|续上页|接上页|承前页|转下页|接下页|未完待续"),
    re.compile(
        r"\(\s*continued\s*\)|\(\s*cont\.?\s*\)|\bcontinued\b|\bcont['’]?d\b",
        re.IGNORECASE,
    ),
)

_TITLE_PUNCT_EDGE_RE = re.compile(r"^[\s:：\-—·•.,，。;；]+|[\s:：\-—·•.,，。;；]+$")
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_TABLE_NUMBER_RE = re.compile(r"(?:续表|表|table)\s*(\d+)", re.IGNORECASE)
_TABLE_NUMBER_PREFIX_RE = re.compile(r"^(?:续表|表|table)\s*\d+\s*[:：\-—.\s]*", re.IGNORECASE)


def strip_json_fences(raw_text: str) -> str:
    """Strip surrounding markdown JSON code fences from model output."""
    stripped = raw_text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def extract_table_number(title: str | None) -> str | None:
    """Extract numeric table identifier from titles like '表1 ...', '续表1', or 'Table 2'."""
    if not title:
        return None
    match = _TABLE_NUMBER_RE.search(title)
    return match.group(1) if match else None


def strip_table_number_prefix(title: str | None) -> str:
    """Strip continuation cues and leading table numbers to compare descriptive titles."""
    if not title:
        return ""
    cleaned = title
    for pattern in _TITLE_STRIP_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = _TITLE_PUNCT_EDGE_RE.sub("", cleaned)
    cleaned = _TABLE_NUMBER_PREFIX_RE.sub("", cleaned)
    cleaned = _TITLE_PUNCT_EDGE_RE.sub("", cleaned)
    return normalize_text_for_match(cleaned)


def extract_search_tokens(text: str) -> frozenset[str]:
    """Extract ASCII alphanumeric tokens (len>=2) and CJK bigrams, dropping stop words."""
    if not text:
        return frozenset()
    tokens: set[str] = set()
    for match in _ASCII_TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if tok not in _SEARCH_STOP_TOKENS:
            tokens.add(tok)
    for run_match in _HAN_RUN_RE.finditer(text):
        run = run_match.group(0)
        for idx in range(len(run) - 1):
            bigram = run[idx : idx + 2]
            if bigram not in _SEARCH_STOP_TOKENS:
                tokens.add(bigram)
    return frozenset(tokens)


def normalize_text_for_match(text: str) -> str:
    """Lowercase and strip whitespace and punctuation for substring comparisons."""
    return _MATCH_STRIP_RE.sub("", text).lower()


def has_continuation_marker(text: str) -> bool:
    """Return True when text contains backward/in-place table continuation cues."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _BACKWARD_CONTINUATION_PATTERNS)


def has_forward_continuation_marker(text: str) -> bool:
    """Return True when text contains forward table continuation cues."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _FORWARD_CONTINUATION_PATTERNS)


def normalize_base_title(title: str | None) -> str:
    """Strip continuation markers and edge punctuation so continued tables share a base title."""
    if not title:
        return ""
    cleaned = title
    for pattern in _TITLE_STRIP_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = _TITLE_PUNCT_EDGE_RE.sub("", cleaned)
    return normalize_text_for_match(cleaned)


def token_overlap_score(
    query_tokens: frozenset[str],
    target_tokens: frozenset[str],
) -> float:
    """Compute weighted token overlap score between query and target token sets."""
    if not query_tokens or not target_tokens:
        return 0.0
    overlap = len(query_tokens & target_tokens)
    if overlap == 0:
        return 0.0
    return float(overlap) + (overlap / len(query_tokens))
