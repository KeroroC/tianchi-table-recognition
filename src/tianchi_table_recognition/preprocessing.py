"""Document preprocessing, EXIF correction, PDF inspection, and high-res original storage."""

from __future__ import annotations

import base64
import io
import math
import os
import re
from pathlib import Path
from typing import Any, cast

import pymupdf
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from tianchi_table_recognition.gateway import EncodedImage
from tianchi_table_recognition.io_utils import atomic_write_file
from tianchi_table_recognition.models import SourceDocumentRef

SUPPORTED_IMAGE_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp"}
)
PDF_MEDIA_TYPE = "application/pdf"

_BARE_PAGE_NUMBER_LINE_RE = re.compile(
    r"^\s*[-—–]?\s*\d+(?:\s*/\s*\d+)?\s*[-—–]?\s*$"
)
_WHITESPACE_RE = re.compile(r"\s+")
_MIN_NATIVE_TEXT_CHARS = 2
_NATIVE_TEXT_MIN_PAGE_RATIO = 0.8


class PreparedPage(BaseModel):
    """Preserved high-resolution page PNG on disk for a Source Document."""

    model_config = ConfigDict(frozen=True)

    document_key: str
    document_hash: str
    page_index: int = Field(ge=0)
    image_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int | None = None


class PdfInspection(BaseModel):
    """Lightweight structural inspection result for a PDF Source Document."""

    model_config = ConfigDict(frozen=True)

    page_count: int = Field(ge=1)
    page_texts: tuple[str, ...]
    has_native_text_layer: bool


def _has_meaningful_native_text(raw_text: str) -> bool:
    kept_lines = [
        line
        for line in raw_text.splitlines()
        if line.strip() and not _BARE_PAGE_NUMBER_LINE_RE.match(line)
    ]
    compact = _WHITESPACE_RE.sub("", "".join(kept_lines))
    return len(compact) >= _MIN_NATIVE_TEXT_CHARS


def _open_pdf(source_path: Path) -> Any:
    return cast(Any, pymupdf).open(source_path)


def inspect_pdf(source_path: Path) -> PdfInspection:
    """Inspect a PDF Source Document for page count and embedded native text layer."""
    doc = _open_pdf(source_path)
    try:
        page_count = int(doc.page_count)
        if page_count <= 0:
            raise ValueError(f"PDF has no pages: {source_path}")
        page_texts = tuple(
            str(doc.load_page(idx).get_text("text")).strip() for idx in range(page_count)
        )
    finally:
        doc.close()

    pages_with_text = sum(1 for txt in page_texts if _has_meaningful_native_text(txt))
    required_pages = max(1, math.ceil(page_count * _NATIVE_TEXT_MIN_PAGE_RATIO))
    return PdfInspection(
        page_count=page_count,
        page_texts=page_texts,
        has_native_text_layer=pages_with_text >= required_pages,
    )


def _write_png_bytes_atomically(target: Path, png_bytes: bytes) -> Path:
    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as fh:
            fh.write(png_bytes)
            fh.flush()
            os.fsync(fh.fileno())

    return atomic_write_file(target, _writer, suffix=".png")


def encode_prepared_page(page: PreparedPage) -> EncodedImage:
    """Base64-encode a preserved PNG page image into an EncodedImage."""
    encoded = base64.b64encode(page.image_path.read_bytes()).decode("ascii")
    return EncodedImage(
        mime_type="image/png",
        data_base64=encoded,
        width=page.width,
        height=page.height,
    )


class OriginalPageStore:
    """Content-hashed local store of EXIF-corrected originals and 300 DPI PDF page renders."""

    def __init__(self, cache_dir: Path) -> None:
        self._root = cache_dir / "original_pages"

    def _load_existing_prepared_page(
        self,
        doc_ref: SourceDocumentRef,
        target: Path,
        *,
        page_index: int,
        dpi: int | None,
    ) -> PreparedPage | None:
        if not target.is_file():
            return None
        with Image.open(target) as existing:
            width, height = existing.size
        return PreparedPage(
            document_key=doc_ref.document_key,
            document_hash=doc_ref.document_hash,
            page_index=page_index,
            image_path=target,
            width=width,
            height=height,
            dpi=dpi,
        )

    def _render_pdf_pixmap(
        self,
        doc_ref: SourceDocumentRef,
        page_index: int,
        *,
        dpi: int,
    ) -> tuple[int, int, bytes, bytes]:
        if doc_ref.media_type != PDF_MEDIA_TYPE:
            raise ValueError(
                f"Expected PDF media_type for {doc_ref.document_key}, got {doc_ref.media_type!r}"
            )
        if page_index < 0:
            raise ValueError(f"page_index must be >= 0, got {page_index}")

        doc = _open_pdf(doc_ref.source_path)
        try:
            if page_index >= int(doc.page_count):
                raise IndexError(
                    f"page_index {page_index} out of range for PDF with {doc.page_count} pages"
                )
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            return int(pix.width), int(pix.height), bytes(pix.samples), bytes(pix.tobytes("png"))
        finally:
            doc.close()

    def preserve_image_document(self, doc_ref: SourceDocumentRef) -> PreparedPage:
        """Correct EXIF orientation and persist full-resolution PNG without mutating source."""
        if doc_ref.media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"Unsupported image media_type {doc_ref.media_type!r} for {doc_ref.document_key}"
            )
        target = self._root / doc_ref.document_hash[:16] / "page_0.png"
        existing = self._load_existing_prepared_page(doc_ref, target, page_index=0, dpi=None)
        if existing is not None:
            return existing

        with Image.open(doc_ref.source_path) as raw_img:
            transposed = ImageOps.exif_transpose(raw_img)
            rgb = transposed.convert("RGB")
            width, height = rgb.size
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
        _write_png_bytes_atomically(target, buf.getvalue())
        return PreparedPage(
            document_key=doc_ref.document_key,
            document_hash=doc_ref.document_hash,
            page_index=0,
            image_path=target,
            width=width,
            height=height,
            dpi=None,
        )

    def render_pdf_page(
        self,
        doc_ref: SourceDocumentRef,
        page_index: int,
        *,
        dpi: int = 300,
    ) -> PreparedPage:
        """Render a single PDF page at high DPI and preserve as lossless PNG on disk."""
        target = self._root / doc_ref.document_hash[:16] / f"page_{page_index}_dpi{dpi}.png"
        existing = self._load_existing_prepared_page(
            doc_ref, target, page_index=page_index, dpi=dpi
        )
        if existing is not None:
            return existing

        width, height, _samples, png_bytes = self._render_pdf_pixmap(
            doc_ref, page_index, dpi=dpi
        )
        _write_png_bytes_atomically(target, png_bytes)
        return PreparedPage(
            document_key=doc_ref.document_key,
            document_hash=doc_ref.document_hash,
            page_index=page_index,
            image_path=target,
            width=width,
            height=height,
            dpi=dpi,
        )

    def render_pdf_thumbnail(
        self,
        doc_ref: SourceDocumentRef,
        page_index: int,
        *,
        dpi: int = 72,
        max_edge: int = 1024,
    ) -> EncodedImage:
        """Render a low-resolution PDF page thumbnail in memory for Page Triage."""
        width, height, samples, png_bytes = self._render_pdf_pixmap(
            doc_ref, page_index, dpi=dpi
        )
        if max(width, height) > max_edge:
            img = Image.frombytes("RGB", (width, height), samples)
            img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

        return EncodedImage(
            mime_type="image/png",
            data_base64=base64.b64encode(png_bytes).decode("ascii"),
            width=width,
            height=height,
        )
