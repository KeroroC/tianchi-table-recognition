"""Tests for document preprocessing, EXIF orientation correction, and PDF inspection/rendering."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, cast

import pymupdf
import pytest
from PIL import Image

from tests.fixtures_factory import (
    create_synthetic_image_with_exif,
    create_synthetic_scanned_pdf,
    create_synthetic_text_pdf,
)
from tianchi_table_recognition.models import SourceDocumentRef
from tianchi_table_recognition.preprocessing import (
    OriginalPageStore,
    encode_prepared_page,
    inspect_pdf,
)


def _doc_ref_for(path: Path, media_type: str) -> SourceDocumentRef:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceDocumentRef(
        document_key=f"doc_{digest[:16]}",
        document_hash=digest,
        source_path=path,
        file_name=path.name,
        media_type=media_type,
    )


@pytest.mark.parametrize(
    ("suffix", "image_format", "media_type"),
    [
        (".png", "PNG", "image/png"),
        (".jpg", "JPEG", "image/jpeg"),
        (".webp", "WEBP", "image/webp"),
    ],
)
def test_preserve_image_document_corrects_exif_and_keeps_original_untouched(
    tmp_path: Path,
    suffix: str,
    image_format: str,
    media_type: str,
) -> None:
    src = create_synthetic_image_with_exif(
        tmp_path / "docs" / f"secret_table{suffix}",
        width=240,
        height=120,
        orientation=6,
        image_format=image_format,
        label="EXIF-6",
    )
    original_bytes_before = src.read_bytes()
    doc_ref = _doc_ref_for(src, media_type)
    store = OriginalPageStore(tmp_path / "work" / "cache")

    prepared = store.preserve_image_document(doc_ref)

    assert prepared.page_index == 0
    assert prepared.width == 120
    assert prepared.height == 240
    assert prepared.image_path.is_file()
    assert "secret_table" not in str(prepared.image_path)
    assert src.read_bytes() == original_bytes_before

    with Image.open(prepared.image_path) as saved:
        assert saved.size == (120, 240)

    encoded = encode_prepared_page(prepared)
    assert encoded.mime_type == "image/png"
    assert encoded.width == 120
    assert encoded.height == 240
    assert len(encoded.data_base64) > 0


def test_render_pdf_page_300_dpi_and_thumbnail_72_dpi(tmp_path: Path) -> None:
    pdf_path = create_synthetic_text_pdf(
        tmp_path / "docs" / "secret_report.pdf",
        ["第一页：摘要", "第二页：研发费用明细表\n项目 金额"],
        width_pt=288.0,
        height_pt=144.0,
    )
    doc_ref = _doc_ref_for(pdf_path, "application/pdf")
    store = OriginalPageStore(tmp_path / "work" / "cache")

    # 288pt x 144pt = 4in x 2in -> 1200x600 px at 300 DPI
    page_300 = store.render_pdf_page(doc_ref, 1, dpi=300)
    assert page_300.page_index == 1
    assert page_300.dpi == 300
    assert page_300.width == 1200
    assert page_300.height == 600
    assert page_300.image_path.is_file()
    assert "secret_report" not in str(page_300.image_path)

    # Second call reuses preserved PNG on disk
    page_300_cached = store.render_pdf_page(doc_ref, 1, dpi=300)
    assert page_300_cached == page_300

    # Low-res thumbnail at 72 DPI: 288x144 px
    thumb = store.render_pdf_thumbnail(doc_ref, 1, dpi=72)
    assert thumb.mime_type == "image/png"
    assert thumb.width == 288
    assert thumb.height == 144
    assert thumb.width < page_300.width


def test_inspect_pdf_distinguishes_native_text_and_scanned_pdfs(tmp_path: Path) -> None:
    text_pdf = create_synthetic_text_pdf(
        tmp_path / "native.pdf",
        ["资产负债表\n货币资金 1000", "利润表\n营业收入 5000"],
    )
    text_info = inspect_pdf(text_pdf)
    assert text_info.page_count == 2
    assert text_info.has_native_text_layer is True
    assert "资产负债表" in text_info.page_texts[0]

    scanned_pdf = create_synthetic_scanned_pdf(
        tmp_path / "scanned.pdf",
        ["PAGE-0", "PAGE-1"],
    )
    scanned_info = inspect_pdf(scanned_pdf)
    assert scanned_info.page_count == 2
    assert scanned_info.has_native_text_layer is False
    assert scanned_info.page_texts == ("", "")

    # Bare page numbers alone must not count as a native text layer
    page_num_only_pdf = create_synthetic_text_pdf(
        tmp_path / "page_nums_only.pdf",
        ["- 1 -", "2 / 3", "3"],
    )
    page_num_info = inspect_pdf(page_num_only_pdf)
    assert page_num_info.page_count == 3
    assert page_num_info.has_native_text_layer is False


def test_inspect_pdf_classifies_hybrid_pdf_with_mostly_scanned_pages_as_scanned(
    tmp_path: Path,
) -> None:
    hybrid_path = tmp_path / "hybrid.pdf"
    doc: Any = cast(Any, pymupdf).open()
    try:
        for i in range(3):
            page = doc.new_page(width=288.0, height=144.0)
            page.insert_text((16.0, 24.0), f"第{i}页 目录说明", fontname="china-s", fontsize=10)
        for _ in range(7):
            img = Image.new("RGB", (200, 100), color=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            page = doc.new_page(width=288.0, height=144.0)
            page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(hybrid_path)
    finally:
        doc.close()

    info = inspect_pdf(hybrid_path)
    assert info.page_count == 10
    assert info.has_native_text_layer is False
