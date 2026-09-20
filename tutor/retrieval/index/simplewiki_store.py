"""Flat-file storage helpers for the dense sidecar (WP-B7).

Ours: no donor code. Two append-only files hold a dense index's payload:

``vectors.fp16``
    A flat file of IEEE-754 half-precision floats, ``count * dim`` of them,
    row-major (row ``i`` is passage ``i``'s embedding). Packed with
    ``struct`` (format code ``e``), not ``array`` -- the stdlib ``array``
    module has no half-float typecode.

``ids.txt``
    One passage id per line, UTF-8, row ``i`` names the passage embedded at
    row ``i`` of ``vectors.fp16``. A newline always terminates the last
    line that was actually appended, so ``ids_bytes`` (see
    ``manifest.BuildCheckpoint``) is meaningful as an exact truncation
    point.

Both files are opened in append/binary mode and grow in lockstep, one row
at a time, so ``vectors.fp16`` size and ``ids.txt`` size are always
``count * dim * 2`` bytes and (sum of id-line lengths + newlines)
respectively for the same ``count``. A crash mid-append can leave one file
longer than the checkpointed truth for the other; ``truncate_to`` restores
both to an exact prior state before a build resumes.
"""

from __future__ import annotations

import struct
from pathlib import Path

VECTORS_FILENAME = "vectors.fp16"
IDS_FILENAME = "ids.txt"
PATHS_FILENAME = "paths.txt"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "checkpoint.json"


def pack_vectors(vectors: list[list[float]], dim: int) -> bytes:
    """Pack rows of ``dim`` floats each as contiguous fp16 bytes."""
    out = bytearray()
    fmt = f"<{dim}e"
    for row in vectors:
        if len(row) != dim:
            raise ValueError(f"expected {dim}-dim vector, got {len(row)}")
        out += struct.pack(fmt, *row)
    return bytes(out)


def append_bytes(path: Path, data: bytes) -> None:
    with Path(path).open("ab") as fh:
        fh.write(data)


def append_ids(path: Path, ids: list[str]) -> bytes:
    """Append ``ids``, one per line, and return the bytes written."""
    data = ("".join(f"{i}\n" for i in ids)).encode("utf-8")
    with Path(path).open("ab") as fh:
        fh.write(data)
    return data


def read_ids(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def file_size(path: Path) -> int:
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def truncate_to(path: Path, size: int) -> None:
    """Truncate ``path`` to exactly ``size`` bytes (no-op if already <=)."""
    path = Path(path)
    if not path.exists():
        if size != 0:
            raise ValueError(f"cannot truncate missing file {path} to {size} bytes")
        return
    current = path.stat().st_size
    if current < size:
        raise ValueError(f"{path} is only {current} bytes, cannot truncate to {size}")
    if current > size:
        with path.open("r+b") as fh:
            fh.truncate(size)


def read_vectors_fp16(path: Path, count: int, dim: int) -> list[list[float]]:
    """Read ``count`` rows of ``dim`` fp16 floats back as Python floats."""
    path = Path(path)
    fmt = f"<{dim}e"
    row_bytes = dim * 2
    rows: list[list[float]] = []
    with path.open("rb") as fh:
        for _ in range(count):
            chunk = fh.read(row_bytes)
            if len(chunk) != row_bytes:
                raise ValueError(
                    f"truncated vectors file: expected {row_bytes} bytes, got {len(chunk)}"
                )
            rows.append(list(struct.unpack(fmt, chunk)))
    return rows
