"""Stage 1 single-image extraction and atomic PageArtifact disk cache."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from tianchi_table_recognition.gateway import EncodedImage, ModelGateway
from tianchi_table_recognition.io_utils import atomic_write_text
from tianchi_table_recognition.models import (
    ModelRole,
    PageArtifact,
    Question,
    SourceDocumentRef,
    TableGrid,
)
from tianchi_table_recognition.preprocessing import (
    PDF_MEDIA_TYPE,
    SUPPORTED_IMAGE_MEDIA_TYPES,
    OriginalPageStore,
    PreparedPage,
    encode_prepared_page,
    inspect_pdf,
)
from tianchi_table_recognition.triage import triage_long_pdf

DEFAULT_STAGE1_SYSTEM_PROMPT = (
    "You are an expert multimodal table structure and content extractor. "
    "Output strict valid JSON only, with no commentary."
)

DEFAULT_STAGE1_USER_PROMPT = (
    "Extract all page context text and every table on this page as JSON with keys "
    "'page_context_text' and 'tables'. Each table must include 'table_id', 'title', "
    "'row_count', 'col_count', 'header_row_count', and 'cells' (each cell having "
    "'row', 'col', 'rowspan', 'colspan', 'text', 'bbox', 'uncertain'). "
    "If any cell text is blurry or illegible, set 'text' to null and 'uncertain' to true; "
    "never guess."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


class Stage1ExtractionError(RuntimeError):
    """Raised when Stage 1 extraction output cannot be parsed into a PageArtifact."""


class _RawPagePayload(BaseModel):
    page_context_text: str = ""
    tables: list[TableGrid] = Field(default_factory=list)


@dataclass(frozen=True)
class CacheKey:
    """Deterministic cache key isolating Stage 1 artifacts by content, model, and prompt."""

    document_hash: str
    stage: str
    stage_version: str
    model_id: str
    prompt_hash: str
    page_index: int = 0

    def digest(self) -> str:
        """Compute deterministic SHA-256 hex digest of all key components."""
        payload = json.dumps(
            {
                "document_hash": self.document_hash,
                "stage": self.stage,
                "stage_version": self.stage_version,
                "model_id": self.model_id,
                "prompt_hash": self.prompt_hash,
                "page_index": self.page_index,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


    def matches(self, artifact: PageArtifact) -> bool:
        """Return True if artifact metadata matches every component of this key."""
        return (
            artifact.document_hash == self.document_hash
            and artifact.stage == self.stage
            and artifact.stage_version == self.stage_version
            and artifact.model_id == self.model_id
            and artifact.prompt_hash == self.prompt_hash
            and artifact.page_index == self.page_index
        )


class PageArtifactCache:
    """Atomic disk cache for page-level Stage 1 PageArtifact objects."""

    def __init__(self, cache_dir: Path) -> None:
        self._root = cache_dir / "page_artifacts"

    @property
    def cache_dir(self) -> Path:
        """Root cache directory containing page_artifacts and related caches."""
        return self._root.parent

    def path_for(self, key: CacheKey) -> Path:
        """Return deterministic file path for a CacheKey."""
        return self._root / f"{key.document_hash[:16]}_p{key.page_index}_{key.digest()[:16]}.json"

    def get(self, key: CacheKey) -> PageArtifact | None:
        """Load cached PageArtifact if present and key metadata matches."""
        path = self.path_for(key)
        if not path.is_file():
            return None
        artifact = PageArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        return artifact if key.matches(artifact) else None

    def put(self, key: CacheKey, artifact: PageArtifact) -> Path:
        """Atomically persist PageArtifact to disk via temp file + fsync + replace."""
        target = self.path_for(key)
        return atomic_write_text(target, artifact.model_dump_json(indent=2))


class PageExtractor(Protocol):
    """Pluggable Stage 1 page extraction strategy interface."""

    @property
    def stage(self) -> str: ...

    @property
    def stage_version(self) -> str: ...

    @property
    def prompt_hash(self) -> str: ...

    def extract_page(
        self,
        *,
        document_key: str,
        document_hash: str,
        image: EncodedImage,
        page_index: int = 0,
    ) -> PageArtifact: ...


def _strip_json_fences(raw_text: str) -> str:
    stripped = raw_text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


class VisionPageExtractor:
    """Default Stage 1 extractor invoking ModelRole.VISION_EXTRACT via ModelGateway."""

    STAGE_NAME = "stage1_page_extract"
    STAGE_VERSION = "1.0"

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        system_prompt: str = DEFAULT_STAGE1_SYSTEM_PROMPT,
        prompt_template: str = DEFAULT_STAGE1_USER_PROMPT,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        prompt_material = f"{system_prompt}\n---\n{prompt_template}"
        self._prompt_hash = hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()[:16]

    @property
    def stage(self) -> str:
        """Stage identifier included in cache keys."""
        return self.STAGE_NAME

    @property
    def stage_version(self) -> str:
        """Stage schema/logic version included in cache keys."""
        return self.STAGE_VERSION

    @property
    def prompt_hash(self) -> str:
        """Hash of the extraction prompt included in cache keys."""
        return self._prompt_hash

    def extract_page(
        self,
        *,
        document_key: str,
        document_hash: str,
        image: EncodedImage,
        page_index: int = 0,
    ) -> PageArtifact:
        """Invoke VISION_EXTRACT for a single page image and parse a PageArtifact."""
        model_id = self._gateway.model_for_role(ModelRole.VISION_EXTRACT)
        user_prompt = (
            f"DocumentKey: {document_key}, page_index: {page_index}.\n"
            f"{self._prompt_template}"
        )
        response = self._gateway.invoke(
            model_role=ModelRole.VISION_EXTRACT,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            images=[image],
            stage=self.STAGE_NAME,
            document_key=document_key,
        )
        cleaned = _strip_json_fences(response.content)
        try:
            payload = _RawPagePayload.model_validate_json(cleaned)
        except Exception as exc:
            raise Stage1ExtractionError(
                f"Failed to parse Stage 1 PageArtifact JSON for {document_key} p{page_index}: {exc}"
            ) from exc

        return PageArtifact(
            document_key=document_key,
            document_hash=document_hash,
            page_index=page_index,
            page_context_text=payload.page_context_text,
            tables=payload.tables,
            stage=self.STAGE_NAME,
            stage_version=self.STAGE_VERSION,
            model_id=model_id,
            prompt_hash=self._prompt_hash,
        )


class DocumentExtractionResult(BaseModel):
    """Page-atomic Stage 1 extraction outputs for a single Source Document."""

    model_config = ConfigDict(frozen=True)

    document_key: str
    page_count: int = Field(ge=1)
    triage_mode: Literal["single_image", "short_document", "native_text", "vision_triage"]
    artifacts: tuple[PageArtifact, ...]
    preserved_pages: tuple[PreparedPage, ...]
    candidate_pages: tuple[int, ...]


def _extract_or_load_page(
    *,
    doc_ref: SourceDocumentRef,
    prepared: PreparedPage,
    cache: PageArtifactCache,
    extractor: PageExtractor,
    model_id: str,
) -> PageArtifact:
    key = CacheKey(
        document_hash=doc_ref.document_hash,
        stage=extractor.stage,
        stage_version=extractor.stage_version,
        model_id=model_id,
        prompt_hash=extractor.prompt_hash,
        page_index=prepared.page_index,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    encoded = encode_prepared_page(prepared)
    artifact = extractor.extract_page(
        document_key=doc_ref.document_key,
        document_hash=doc_ref.document_hash,
        image=encoded,
        page_index=prepared.page_index,
    )
    cache.put(key, artifact)
    return artifact


def extract_document(
    *,
    doc_ref: SourceDocumentRef,
    gateway: ModelGateway,
    cache: PageArtifactCache,
    questions: Sequence[Question] = (),
    page_store: OriginalPageStore | None = None,
    extractor: PageExtractor | None = None,
    pdf_render_dpi: int = 300,
    pdf_triage_page_threshold: int = 8,
    pdf_triage_thumbnail_dpi: int = 72,
) -> DocumentExtractionResult:
    """Extract and atomically cache page-level PageArtifacts for image or PDF Source Documents."""
    active_store = page_store or OriginalPageStore(cache.cache_dir)
    active_extractor = extractor or VisionPageExtractor(gateway=gateway)
    model_id = gateway.model_for_role(ModelRole.VISION_EXTRACT)

    if doc_ref.media_type in SUPPORTED_IMAGE_MEDIA_TYPES:
        prepared = active_store.preserve_image_document(doc_ref)
        artifact = _extract_or_load_page(
            doc_ref=doc_ref,
            prepared=prepared,
            cache=cache,
            extractor=active_extractor,
            model_id=model_id,
        )
        return DocumentExtractionResult(
            document_key=doc_ref.document_key,
            page_count=1,
            triage_mode="single_image",
            artifacts=(artifact,),
            preserved_pages=(prepared,),
            candidate_pages=(0,),
        )

    if doc_ref.media_type != PDF_MEDIA_TYPE:
        raise ValueError(
            f"Unsupported Source Document media_type {doc_ref.media_type!r} "
            f"for {doc_ref.document_key}"
        )

    inspection = inspect_pdf(doc_ref.source_path)
    if inspection.page_count <= pdf_triage_page_threshold:
        triage_mode: Literal["short_document", "native_text", "vision_triage"] = "short_document"
        selected_pages = tuple(range(inspection.page_count))
    else:
        triage = triage_long_pdf(
            doc_ref=doc_ref,
            inspection=inspection,
            questions=questions,
            gateway=gateway,
            page_store=active_store,
            thumbnail_dpi=pdf_triage_thumbnail_dpi,
            cache_dir=cache.cache_dir,
        )
        triage_mode = triage.mode
        selected_pages = tuple(sorted(set(triage.candidate_pages)))

    artifacts: list[PageArtifact] = []
    preserved_pages: list[PreparedPage] = []
    for page_index in selected_pages:
        prepared = active_store.render_pdf_page(doc_ref, page_index, dpi=pdf_render_dpi)
        preserved_pages.append(prepared)
        artifacts.append(
            _extract_or_load_page(
                doc_ref=doc_ref,
                prepared=prepared,
                cache=cache,
                extractor=active_extractor,
                model_id=model_id,
            )
        )

    return DocumentExtractionResult(
        document_key=doc_ref.document_key,
        page_count=inspection.page_count,
        triage_mode=triage_mode,
        artifacts=tuple(artifacts),
        preserved_pages=tuple(preserved_pages),
        candidate_pages=selected_pages,
    )


def extract_image_document(
    *,
    doc_ref: SourceDocumentRef,
    gateway: ModelGateway,
    cache: PageArtifactCache,
    extractor: PageExtractor | None = None,
) -> PageArtifact:
    """Extract and atomically cache PageArtifact for a single-image Source Document."""
    if doc_ref.media_type == PDF_MEDIA_TYPE:
        raise NotImplementedError(
            f"PDF Source Document {doc_ref.document_key} requires extract_document."
        )
    result = extract_document(
        doc_ref=doc_ref,
        gateway=gateway,
        cache=cache,
        extractor=extractor,
    )
    return result.artifacts[0]
