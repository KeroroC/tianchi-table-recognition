"""Deterministic Answer Validator and bounded JSON repair for Stage 2 outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from tianchi_table_recognition.gateway import ComplianceError, ModelGateway
from tianchi_table_recognition.models import AnswerFormat, ModelRole, PageArtifact, Question

_ZERO_WIDTH_CHARS = "\ufeff\u200b\u200c\u200d"
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_ANY_FENCED_WHOLE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)
_EXPLANATORY_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:根据(?:上述|提供|该)?(?:表格|图表|材料|上下文|数据|内容)?(?:信息)?[，,]?\s*)?"
    r"(?:"
    r"(?:最终答案|答案|结果|提取结果)\s*(?:(?:是|为)\s*[:：]?|[:：])\s*"
    r"|(?:Final\s+Answer|Answer|Result)\s*[:：]\s*"
    r")"
    r")",
    re.IGNORECASE,
)
_FULLWIDTH_NUM_MAP = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "－": "-",
        "−": "-",
        "＋": "+",
        "．": ".",
        "％": "%",
    }
)
_CN_DIGIT_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_KNOWN_UNITS = (
    "万美元",
    "亿美元",
    "平方米",
    "万元",
    "亿元",
    "千元",
    "美元",
    "港元",
    "欧元",
    "万股",
    "亿股",
    "元",
    "人",
    "家",
    "个",
    "件",
    "台",
    "吨",
    "次",
    "项",
    "股",
    "㎡",
)
_UNIT_ALT = "|".join(re.escape(u) for u in _KNOWN_UNITS)
_NUMBER_CANDIDATE_RE = re.compile(
    rf"^(?P<currency>¥|￥|\$|US\$|USD|RMB|CNY)?\s*"
    rf"(?P<sign>[+-])?\s*"
    rf"(?P<digits>(?:\d{{1,3}}(?:,\d{{3}})+|\d+(?:,\d+)*)(?:\.\d+)?|\.\d+)"
    rf"(?P<percent>%)?\s*"
    rf"(?P<unit>{_UNIT_ALT})?$",
    re.IGNORECASE,
)
_VALID_COMMA_INT_RE = re.compile(r"^\d{1,3}(?:,\d{3})+$")
_RMB_UNITS = frozenset({"元", "千元", "万元", "亿元"})
_USD_UNITS = frozenset({"美元", "万美元", "亿美元"})
_BRACKET_PAIRS: dict[str, str] = {
    "【": "】",
    "[": "]",
    "〔": "〕",
    "「": "」",
    "『": "』",
    "《": "》",
}
_BRACKET_CLOSERS: frozenset[str] = frozenset(_BRACKET_PAIRS.values())
_DISTINCT_BRACKET_FOLD = str.maketrans(
    {
        "【": "<L>",
        "[": "<L>",
        "〔": "<L>",
        "「": "<L>",
        "『": "<L>",
        "】": "<R>",
        "]": "<R>",
        "〕": "<R>",
        "」": "<R>",
        "』": "<R>",
    }
)
_TRACKED_BRACKETS: frozenset[str] = frozenset("【】[]〔〕「」『』")
_WHITESPACE_RE = re.compile(r"\s+")
_CJK_CJK_SPACE_RE = re.compile(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]")
_CJK_ASCII_SPACE_RE = re.compile(
    r"(?:[\u4e00-\u9fff]\s+[A-Za-z0-9]|[A-Za-z0-9]\s+[\u4e00-\u9fff])"
)
_BRACKET_SPACE_RE = re.compile(
    r"(?:[【〔「『《\[]\s+|\s+[】〕」』》\]]|"
    r"[】〕」』》]\s+[\u4e00-\u9fffA-Za-z0-9]|"
    r"[\u4e00-\u9fff]\s+[【〔「『《\[])"
)
_DIGIT_DIGIT_SPACE_RE = re.compile(r"\d\s+\d")
_DATETIME_SPACE_RE = re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?\b")


class ValidationResult(BaseModel):
    """Outcome of AnswerValidator.validate."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    answer: str | None = None
    raw_answer: str
    errors: tuple[str, ...] = ()
    repaired: bool = False
    repair_attempts: int = 0


def _reject_nonstandard_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {token!r} is not allowed")


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_nonstandard_json_constant)


def _dump_compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _extract_outermost_json_spans(text: str) -> list[str]:
    """Scan text left-to-right and return outermost balanced {...} or [...] spans."""
    spans: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        start = i
        stack: list[str] = ["]" if ch == "[" else "}"]
        in_string = False
        escaped = False
        j = i + 1
        while j < n and stack:
            cur = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
            else:
                if cur == '"':
                    in_string = True
                elif cur == "{":
                    stack.append("}")
                elif cur == "[":
                    stack.append("]")
                elif cur in ("}", "]"):
                    if cur == stack[-1]:
                        stack.pop()
                    else:
                        break
            j += 1
        if not stack:
            spans.append(text[start:j])
            i = j
        else:
            i += 1
    return spans


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] outside JSON string literals."""
    out: list[str] = []
    n = len(text)
    in_string = False
    escaped = False
    i = 0
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            k = i + 1
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] in "}]":
                i += 1
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_smart_quotes_and_punctuation(text: str) -> str:
    """Replace outside-string Chinese/smart double quotes and fullwidth : / ,."""
    out: list[str] = []
    in_ascii_string = False
    escaped = False
    for ch in text:
        if in_ascii_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_ascii_string = False
            continue

        if ch == '"':
            in_ascii_string = True
            out.append(ch)
        elif ch in ("\u201c", "\u201d", "\u201e", "\u201f"):
            out.append('"')
        elif ch == "：":
            out.append(":")
        elif ch == "，":
            out.append(",")
        else:
            out.append(ch)
    return "".join(out)


def _candidate_json_texts(raw_text: str) -> list[str]:
    stripped = raw_text.strip().strip(_ZERO_WIDTH_CHARS).strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    for match in _FENCED_BLOCK_RE.finditer(stripped):
        fenced_body = match.group(1).strip()
        if fenced_body and fenced_body not in candidates:
            candidates.append(fenced_body)
    return candidates


def _check_json_type(parsed: Any, answer_format: AnswerFormat) -> str | None:
    if answer_format == AnswerFormat.JSON and not isinstance(parsed, dict):
        return f"Expected JSON object (dict), got {type(parsed).__name__}"
    if answer_format == AnswerFormat.JSON_ARRAY and not isinstance(parsed, list):
        return f"Expected JSON array (list), got {type(parsed).__name__}"
    return None


def _try_parse_json_candidate(
    text: str, answer_format: AnswerFormat
) -> tuple[str | None, str | None]:
    """Attempt strict JSON parse + type check; returns (compact_json, error_msg)."""
    try:
        parsed = _strict_json_loads(text)
    except Exception as exc:
        return None, f"Invalid JSON syntax: {exc}"
    type_err = _check_json_type(parsed, answer_format)
    if type_err is not None:
        return None, type_err
    return _dump_compact_json(parsed), None


def _spans_with_outermost(text: str) -> list[str]:
    spans = [text]
    for span in _extract_outermost_json_spans(text):
        if span not in spans:
            spans.append(span)
    return spans


def _try_parse_json_candidates(
    texts: list[str],
    answer_format: AnswerFormat,
    last_error: str,
) -> tuple[str | None, str]:
    for text in texts:
        compact, err = _try_parse_json_candidate(text, answer_format)
        if compact is not None:
            return compact, ""
        if err is not None:
            last_error = err
    return None, last_error


def _deterministic_clean_json(
    raw_text: str, answer_format: AnswerFormat
) -> tuple[str | None, str]:
    """Run incremental deterministic JSON cleaning passes without corrupting valid strings."""
    base_candidates = _candidate_json_texts(raw_text)
    if not base_candidates:
        return None, "Empty JSON answer"

    last_error = "Invalid JSON syntax"
    for candidate in base_candidates:
        spans = _spans_with_outermost(candidate)

        # Pass 1: as-is
        compact, last_error = _try_parse_json_candidates(spans, answer_format, last_error)
        if compact is not None:
            return compact, ""

        # Pass 2: trailing commas removed
        no_trailing_spans = [
            cleaned for text in spans if (cleaned := _strip_trailing_commas(text)) != text
        ]
        compact, last_error = _try_parse_json_candidates(
            no_trailing_spans, answer_format, last_error
        )
        if compact is not None:
            return compact, ""

        # Pass 3: outside-string smart quotes + fullwidth punctuation + trailing commas
        normalized_base = _normalize_smart_quotes_and_punctuation(candidate)
        norm_spans = [_strip_trailing_commas(t) for t in _spans_with_outermost(normalized_base)]
        compact, last_error = _try_parse_json_candidates(norm_spans, answer_format, last_error)
        if compact is not None:
            return compact, ""

    return None, last_error


@dataclass(frozen=True)
class _NumberDirectives:
    keep_commas: bool
    precision: int | None
    percent_mode: Literal["percent", "decimal"] | None
    keep_unit: bool
    expected_unit: str | None


@dataclass(frozen=True)
class _ParsedNumber:
    currency: str
    value: Decimal
    has_percent: bool
    unit: str | None


def _strip_explanatory_prefix_and_fences(raw_text: str) -> str:
    text = raw_text.strip().strip(_ZERO_WIDTH_CHARS).strip()
    fence_match = _ANY_FENCED_WHOLE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    elif len(text) >= 2 and text.startswith("`") and text.endswith("`") and "`" not in text[1:-1]:
        text = text[1:-1].strip()
    # Iteratively strip leading explanatory labels such as "根据表格，答案为："
    for _ in range(2):
        cleaned = _EXPLANATORY_PREFIX_RE.sub("", text, count=1).strip()
        if cleaned == text:
            break
        if not cleaned:
            if text.endswith((":", "：")):
                return ""
            break
        text = cleaned
    return text.strip().strip(_ZERO_WIDTH_CHARS).strip()


def _parse_cn_or_ascii_int(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _CN_DIGIT_MAP.get(token)


def _extract_number_directives(question_text: str) -> _NumberDirectives:
    forbid_commas = bool(
        re.search(r"(?:不要|无需|不保留|去掉|去除|不带)千?分?位?(?:分隔|逗号)", question_text)
    )
    keep_commas = (not forbid_commas) and bool(
        re.search(r"千分位|千位分隔|逗号分隔|保留逗号", question_text)
    )

    precision: int | None = None
    if re.search(r"保留整数|精确到(?:整数|个位)|取整", question_text):
        precision = 0
    else:
        for pat in (
            r"保留\s*([0-9]+|[零一二两三四五六七八九十])\s*位小数",
            r"精确到小数点后\s*([0-9]+|[零一二两三四五六七八九十])\s*位",
            r"小数点后保留\s*([0-9]+|[零一二两三四五六七八九十])\s*位",
            r"(\d+)\s+decimal\s+places?",
        ):
            m = re.search(pat, question_text, re.IGNORECASE)
            if m:
                precision = _parse_cn_or_ascii_int(m.group(1))
                break
        if precision is None:
            m_step = re.search(r"精确到\s*0\.(0*)1", question_text)
            if m_step:
                precision = len(m_step.group(1)) + 1

    percent_mode: Literal["percent", "decimal"] | None = None
    if re.search(r"以\s*%\s*表示|用百分比表示|以百分比表示|百分比表示|百分数表示", question_text):
        percent_mode = "percent"
    elif re.search(r"用小数表示|以小数表示|小数形式表示", question_text):
        percent_mode = "decimal"

    forbid_unit = bool(re.search(r"不带单位|无需单位|不要单位|不含单位", question_text))
    keep_unit = (not forbid_unit) and bool(
        re.search(r"请带单位|保留单位|带单位|附带单位|包含单位", question_text)
    )

    expected_unit: str | None = None
    for unit_pat in (
        rf"单位\s*[：:]\s*({_UNIT_ALT})",
        rf"以\s*({_UNIT_ALT})\s*为单位",
        rf"[（(]\s*(?:单位\s*[：:]\s*)?({_UNIT_ALT})\s*[）)]",
        rf"是多少\s*({_UNIT_ALT})",
    ):
        m_unit = re.search(unit_pat, question_text)
        if m_unit:
            expected_unit = m_unit.group(1)
            break

    return _NumberDirectives(
        keep_commas=keep_commas,
        precision=precision,
        percent_mode=percent_mode,
        keep_unit=keep_unit,
        expected_unit=expected_unit,
    )


def _format_decimal_number(
    value: Decimal,
    *,
    precision: int | None,
    keep_commas: bool,
) -> str:
    if value == 0:
        value = Decimal(0)
    if precision is not None:
        quant = Decimal(10) ** -precision
        value = value.quantize(quant, rounding=ROUND_HALF_UP)
        if value == 0:
            value = Decimal(0)
        rendered = f"{value:f}"
    else:
        rendered = f"{value:f}"
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")

    if not keep_commas:
        return rendered

    sign = ""
    body = rendered
    if body.startswith("-"):
        sign = "-"
        body = body[1:]
    int_part, dot, frac_part = body.partition(".")
    grouped_int = f"{int(int_part):,}"
    return f"{sign}{grouped_int}{dot}{frac_part}"


def _fail_result(raw_answer: str, message: str) -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        answer=None,
        raw_answer=raw_answer,
        errors=(message,),
    )


def _parse_number_components(
    text: str, *, allow_currency: bool
) -> tuple[_ParsedNumber | None, str | None]:
    match = _NUMBER_CANDIDATE_RE.fullmatch(text)
    if match is None:
        return None, f"Invalid numeric answer {text!r}"

    currency = (match.group("currency") or "").upper()
    if currency and not allow_currency:
        return None, f"Invalid normalized number answer {text!r}"

    sign = match.group("sign") or ""
    digits_raw = match.group("digits")
    has_percent = match.group("percent") is not None
    unit = match.group("unit")

    int_part = digits_raw.partition(".")[0]
    if "," in int_part and not _VALID_COMMA_INT_RE.fullmatch(int_part):
        return None, f"Malformed thousands comma grouping in {digits_raw!r}"

    if has_percent and unit is not None:
        return None, f"Ambiguous number with both '%' and unit {unit!r}"

    numeric_str = f"{sign}{digits_raw.replace(',', '')}"
    try:
        dec_val = Decimal(numeric_str)
    except InvalidOperation as exc:
        return None, f"Invalid Decimal value {numeric_str!r}: {exc}"
    if not dec_val.is_finite():
        return None, f"Non-finite number {numeric_str!r} is not allowed"

    return (
        _ParsedNumber(
            currency=currency,
            value=dec_val,
            has_percent=has_percent,
            unit=unit,
        ),
        None,
    )


def _validate_number(question: Question, raw_answer: str) -> ValidationResult:
    cleaned = _strip_explanatory_prefix_and_fences(raw_answer)
    cleaned = cleaned.translate(_FULLWIDTH_NUM_MAP)
    if cleaned.endswith("。"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return _fail_result(raw_answer, "Empty number answer")

    parsed, parse_err = _parse_number_components(cleaned, allow_currency=True)
    if parsed is None:
        return _fail_result(raw_answer, parse_err or "Invalid numeric answer")

    directives = _extract_number_directives(question.question)

    if directives.percent_mode == "percent" and not parsed.has_percent:
        return _fail_result(
            raw_answer,
            "Question requires '%' percentage representation; refusing silent conversion",
        )
    if directives.percent_mode == "decimal" and parsed.has_percent:
        return _fail_result(
            raw_answer,
            "Question requires decimal representation without '%'; refusing silent conversion",
        )

    if directives.expected_unit is not None:
        if parsed.unit is not None and parsed.unit != directives.expected_unit:
            return _fail_result(
                raw_answer,
                f"Number unit mismatch: got {parsed.unit!r}, expected {directives.expected_unit!r}",
            )
        if parsed.currency in ("$", "US$", "USD") and directives.expected_unit in _RMB_UNITS:
            return _fail_result(
                raw_answer,
                f"Currency mismatch: got {parsed.currency!r} for unit {directives.expected_unit!r}",
            )
        if parsed.currency in ("¥", "￥", "RMB", "CNY") and directives.expected_unit in _USD_UNITS:
            return _fail_result(
                raw_answer,
                f"Currency mismatch: got {parsed.currency!r} for unit {directives.expected_unit!r}",
            )

    formatted = _format_decimal_number(
        parsed.value,
        precision=directives.precision,
        keep_commas=directives.keep_commas,
    )
    if parsed.has_percent:
        formatted = f"{formatted}%"
    elif directives.keep_unit:
        chosen_unit = parsed.unit or directives.expected_unit
        if chosen_unit:
            formatted = f"{formatted}{chosen_unit}"

    return ValidationResult(
        is_valid=True,
        answer=formatted,
        raw_answer=raw_answer,
    )


def _artifact_source_texts(artifact: PageArtifact | Sequence[PageArtifact]) -> list[str]:
    artifacts = [artifact] if isinstance(artifact, PageArtifact) else list(artifact)
    texts: list[str] = []
    for page in artifacts:
        if page.page_context_text:
            texts.append(page.page_context_text)
        for table in page.tables:
            for cell in table.cells:
                if cell.text:
                    texts.append(cell.text)
    return texts


def _check_distinct_bracket_pairs(text: str) -> str | None:
    stack: list[str] = []
    for ch in text:
        if ch in _BRACKET_PAIRS:
            stack.append(ch)
        elif ch in _BRACKET_CLOSERS:
            if not stack:
                return f"Unmatched closing bracket {ch!r} in string answer"
            expected_closer = _BRACKET_PAIRS[stack[-1]]
            if ch != expected_closer:
                return f"Mismatched bracket pair {stack[-1]!r} and {ch!r} in string answer"
            stack.pop()
    if stack:
        return f"Unclosed bracket {stack[-1]!r} in string answer"
    return None


def _has_structural_spurious_space(text: str) -> str | None:
    if _CJK_CJK_SPACE_RE.search(text):
        return "Spurious whitespace inserted between Chinese characters"
    if _CJK_ASCII_SPACE_RE.search(text):
        return "Spurious whitespace inserted between Chinese and ASCII characters"
    if _BRACKET_SPACE_RE.search(text):
        return "Spurious whitespace adjacent to brackets"
    without_dt = _DATETIME_SPACE_RE.sub("", text)
    if _DIGIT_DIGIT_SPACE_RE.search(without_dt):
        return "Spurious whitespace splitting numeric digits"
    return None


def _validate_string(
    raw_answer: str,
    *,
    artifact: PageArtifact | Sequence[PageArtifact] | None = None,
) -> ValidationResult:
    cleaned = _strip_explanatory_prefix_and_fences(raw_answer)
    if not cleaned:
        return _fail_result(raw_answer, "Empty string answer")

    pair_err = _check_distinct_bracket_pairs(cleaned)
    if pair_err is not None:
        return _fail_result(raw_answer, pair_err)

    source_texts = _artifact_source_texts(artifact) if artifact is not None else []
    verbatim_in_source = any(cleaned in src for src in source_texts)

    if source_texts and not verbatim_in_source and any(ch in _TRACKED_BRACKETS for ch in cleaned):
        folded_candidate = cleaned.translate(_DISTINCT_BRACKET_FOLD)
        if any(folded_candidate in src.translate(_DISTINCT_BRACKET_FOLD) for src in source_texts):
            return _fail_result(
                raw_answer,
                "Bracket fidelity mismatch: distinct brackets (e.g. 【 vs [) "
                "differ from source text",
            )

    if _WHITESPACE_RE.search(cleaned) and not verbatim_in_source:
        if source_texts:
            spaceless = _WHITESPACE_RE.sub("", cleaned)
            if any(spaceless in src for src in source_texts):
                return _fail_result(
                    raw_answer,
                    "Spurious whitespace inserted into spaceless source text",
                )
        space_err = _has_structural_spurious_space(cleaned)
        if space_err is not None:
            return _fail_result(raw_answer, space_err)

    return ValidationResult(
        is_valid=True,
        answer=cleaned,
        raw_answer=raw_answer,
    )


def validate_submission_answer(answer_format: AnswerFormat, answer: str) -> tuple[str, ...]:
    """Defense-in-depth strict validation for already-normalized Submission answers."""
    if not answer:
        return ("Answer must not be empty",)

    if answer_format in (AnswerFormat.JSON, AnswerFormat.JSON_ARRAY):
        if "\n" in answer or "\r" in answer:
            return ("JSON answer must be compact single-line JSON",)
        _, err = _try_parse_json_candidate(answer, answer_format)
        return (err,) if err is not None else ()

    if answer_format == AnswerFormat.NUMBER:
        _, err = _parse_number_components(answer, allow_currency=False)
        return (err,) if err is not None else ()

    if answer_format == AnswerFormat.STRING:
        if answer != answer.strip() or any(ch in answer for ch in _ZERO_WIDTH_CHARS):
            return ("String answer contains untrimmed whitespace or zero-width characters",)
        pair_err = _check_distinct_bracket_pairs(answer)
        if pair_err is not None:
            return (pair_err,)
        return ()

    return (f"Unsupported answer_format={answer_format}",)


class AnswerValidator:
    """Deterministic post-solving validator with optional bounded JSON repair."""

    STAGE_NAME = "stage2_json_repair"
    _REPAIR_SYSTEM_PROMPT = (
        "You are a strict JSON syntax repair assistant. "
        "Fix only syntax/quoting/structure errors while preserving all original keys, "
        "values, numbers, and Chinese text. Output strict valid JSON only, with no "
        "markdown code fences or explanation."
    )

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        *,
        max_json_repair_attempts: int = 1,
    ) -> None:
        if max_json_repair_attempts < 0:
            raise ValueError("max_json_repair_attempts must be >= 0")
        self._gateway = gateway
        self._max_json_repair_attempts = max_json_repair_attempts

    def _validate_json(self, question: Question, raw_answer: str) -> ValidationResult:
        compact, err = _deterministic_clean_json(raw_answer, question.answer_format)
        if compact is not None:
            return ValidationResult(
                is_valid=True,
                answer=compact,
                raw_answer=raw_answer,
            )

        if self._gateway is None or self._max_json_repair_attempts == 0:
            return ValidationResult(
                is_valid=False,
                answer=None,
                raw_answer=raw_answer,
                errors=(err,),
            )

        expected_kind = (
            "JSON object (dict)"
            if question.answer_format == AnswerFormat.JSON
            else "JSON array (list)"
        )
        last_error = err
        candidate = raw_answer
        attempts = 0

        for _ in range(self._max_json_repair_attempts):
            attempts += 1
            user_prompt = "\n".join(
                [
                    f"DocumentKey: {question.document_key}",
                    f"ExpectedTopLevelType: {expected_kind}",
                    f"ValidationError: {last_error}",
                    f"BrokenOutput: {candidate}",
                ]
            )
            try:
                response = self._gateway.invoke(
                    model_role=ModelRole.TEXT_REASON,
                    system_prompt=self._REPAIR_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    stage=self.STAGE_NAME,
                    document_key=question.document_key,
                    question_id=question.id,
                )
            except ComplianceError:
                raise
            except Exception as exc:
                last_error = f"JSON repair call failed: {exc}"
                continue

            repaired_compact, repaired_err = _deterministic_clean_json(
                response.content, question.answer_format
            )
            if repaired_compact is not None:
                return ValidationResult(
                    is_valid=True,
                    answer=repaired_compact,
                    raw_answer=raw_answer,
                    repaired=True,
                    repair_attempts=attempts,
                )
            last_error = repaired_err
            candidate = response.content

        return ValidationResult(
            is_valid=False,
            answer=None,
            raw_answer=raw_answer,
            errors=(last_error,),
            repaired=False,
            repair_attempts=attempts,
        )

    def validate(
        self,
        question: Question,
        raw_answer: str,
        *,
        artifact: PageArtifact | Sequence[PageArtifact] | None = None,
    ) -> ValidationResult:
        """Validate and normalize a candidate answer according to question.answer_format."""
        if question.answer_format in (AnswerFormat.JSON, AnswerFormat.JSON_ARRAY):
            return self._validate_json(question, raw_answer)
        if question.answer_format == AnswerFormat.NUMBER:
            return _validate_number(question, raw_answer)
        if question.answer_format == AnswerFormat.STRING:
            return _validate_string(raw_answer, artifact=artifact)
        raise NotImplementedError(f"Unsupported answer_format={question.answer_format}")
