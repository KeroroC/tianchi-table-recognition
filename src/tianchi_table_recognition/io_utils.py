"""Shared filesystem utilities for atomic persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(target: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write UTF-8 text via temp file + fsync + os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target
