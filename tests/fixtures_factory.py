"""Synthetic fixture factory for creating temporary datasets and images."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pymupdf
from openpyxl import Workbook
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SyntheticQuestionRow:
    """Single row to write into a synthetic competition questions workbook."""

    id: str
    file_name: str
    question_type: str
    question: str
    table_hint: str | None
    answer_format: str


def create_synthetic_image(
    path: Path,
    *,
    width: int = 320,
    height: int = 180,
    color: tuple[int, int, int] = (255, 255, 255),
    label: str = "TABLE",
    image_format: str = "PNG",
) -> Path:
    """Create a real synthetic image file on disk for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, width - 10, height - 10], outline=(0, 0, 0), width=2)
    draw.line([10, 50, width - 10, 50], fill=(0, 0, 0), width=2)
    draw.text((20, 20), label, fill=(0, 0, 0))
    img.save(path, format=image_format)
    return path


def create_synthetic_image_with_exif(
    path: Path,
    *,
    width: int = 200,
    height: int = 100,
    orientation: int = 6,
    image_format: str = "JPEG",
    label: str = "IMG",
) -> Path:
    """Create a PNG, JPEG, or WebP synthetic image with an EXIF Orientation tag."""
    create_synthetic_image(
        path,
        width=width,
        height=height,
        label=label,
        image_format=image_format,
    )
    with Image.open(path) as img:
        exif = img.getexif()
        exif[274] = orientation
        img.save(path, format=image_format, exif=exif)
    return path


def create_synthetic_text_pdf(
    path: Path,
    page_texts: Sequence[str],
    *,
    width_pt: float = 288.0,
    height_pt: float = 144.0,
) -> Path:
    """Create a real multi-page PDF with an embedded native text layer per page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: Any = cast(Any, pymupdf).open()
    try:
        for text in page_texts:
            page = doc.new_page(width=width_pt, height=height_pt)
            y = 24.0
            for line in text.splitlines() or [""]:
                page.insert_text((16.0, y), line, fontname="china-s", fontsize=10)
                y += 14.0
        doc.save(path)
    finally:
        doc.close()
    return path


def create_synthetic_scanned_pdf(
    path: Path,
    page_labels: Sequence[str],
    *,
    width_pt: float = 288.0,
    height_pt: float = 144.0,
) -> Path:
    """Create a real multi-page scanned PDF containing only rasterized images (no text layer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: Any = cast(Any, pymupdf).open()
    try:
        img_w = max(64, int(round(width_pt * 2)))
        img_h = max(64, int(round(height_pt * 2)))
        for label in page_labels:
            img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([16, 16, img_w - 16, img_h - 16], outline=(0, 0, 0), width=3)
            draw.line([16, 64, img_w - 16, 64], fill=(0, 0, 0), width=2)
            draw.text((28, 30), label, fill=(0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            page = doc.new_page(width=width_pt, height=height_pt)
            page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(path)
    finally:
        doc.close()
    return path


def write_questions_workbook(
    path: Path,
    rows: list[SyntheticQuestionRow],
    *,
    headers: list[str] | None = None,
) -> Path:
    """Write a synthetic questions Excel workbook with text-formatted IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    cols = headers or [
        "id",
        "file_name",
        "question_type",
        "question",
        "table_hint",
        "answer_format",
    ]
    ws.append(cols)
    for row in rows:
        mapping: dict[str, Any] = {
            "id": row.id,
            "file_name": row.file_name,
            "question_type": row.question_type,
            "question": row.question,
            "table_hint": row.table_hint,
            "answer_format": row.answer_format,
        }
        ws.append([mapping.get(col) for col in cols])
        id_cell = ws.cell(row=ws.max_row, column=1)
        id_cell.number_format = "@"
    wb.save(path)
    wb.close()
    return path


def write_submission_template(path: Path, ids: list[str]) -> Path:
    """Write a synthetic submission template Excel workbook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["id", "answer"])
    for qid in ids:
        ws.append([qid, ""])
        cell = ws.cell(row=ws.max_row, column=1)
        cell.number_format = "@"
    wb.save(path)
    wb.close()
    return path
