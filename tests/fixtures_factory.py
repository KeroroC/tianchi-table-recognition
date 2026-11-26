"""Synthetic fixture factory for creating temporary datasets and images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
