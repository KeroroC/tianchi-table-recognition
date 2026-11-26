"""Shared filesystem utilities for atomic persistence."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write_file(
    target: Path,
    writer: Callable[[Path], None],
    *,
    suffix: str = ".tmp",
) -> Path:
    """Atomically create or replace target by writing a temp file then calling os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(target: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write UTF-8 text via temp file + fsync + os.replace."""

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())

    return atomic_write_file(target, _write)
