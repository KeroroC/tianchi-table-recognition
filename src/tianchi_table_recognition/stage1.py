"""Stage 1 single-image extraction and atomic PageArtifact disk cache."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from tianchi_table_recognition.gateway import EncodedImage, ModelGateway
from tianchi_table_recognition.io_utils import atomic_write_text
from tianchi_table_recognition.models import (
    ModelRole,
    PageArtifact,
    SourceDocumentRef,
    TableGrid,
)

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


def encode_image_file(source_path: Path) -> EncodedImage:
    """Load an image, apply EXIF orientation, and re-encode as clean PNG base64."""
    with Image.open(source_path) as raw_img:
        transposed = ImageOps.exif_transpose(raw_img)
        rgb = transposed.convert("RGB")
        width, height = rgb.size
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return EncodedImage(mime_type="image/png", data_base64=encoded, width=width, height=height)


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


def extract_image_document(
    *,
    doc_ref: SourceDocumentRef,
    gateway: ModelGateway,
    cache: PageArtifactCache,
    extractor: PageExtractor | None = None,
) -> PageArtifact:
    """Extract and atomically cache PageArtifact for a single-image Source Document."""
    if doc_ref.media_type == "application/pdf":
        raise NotImplementedError(
            f"PDF Source Document {doc_ref.document_key} requires T3 PDF rendering support."
        )

    active_extractor = extractor or VisionPageExtractor(gateway=gateway)
    model_id = gateway.model_for_role(ModelRole.VISION_EXTRACT)
    key = CacheKey(
        document_hash=doc_ref.document_hash,
        stage=active_extractor.stage,
        stage_version=active_extractor.stage_version,
        model_id=model_id,
        prompt_hash=active_extractor.prompt_hash,
        page_index=0,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    encoded = encode_image_file(doc_ref.source_path)
    artifact = active_extractor.extract_page(
        document_key=doc_ref.document_key,
        document_hash=doc_ref.document_hash,
        image=encoded,
        page_index=0,
    )
    cache.put(key, artifact)
    return artifact
