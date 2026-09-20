"""Archive header/signature helpers, plus tutor validation/fingerprint/open API.

Adapted from openzim-mcp
Upstream path: openzim_mcp/zim/archive.py
Tag: v3.3.4
Commit: 9358db06f205bb0b95cc938c68405851a0e205a8
MIT License
Copyright (c) 2025-2026 Cameron Rye

What was changed: only the pure, dependency-free header/signature functions
(``ZIM_MAGIC``, ``_declared_zim_size``, ``has_zim_signature``,
``is_truncated_zim``) were kept, copied close to verbatim. All config/cache/
process-pool/MCP coupling was stripped. Everything below the
"tutor code below" marker is new code written for this project.
"""

from __future__ import annotations

import struct
from pathlib import Path

ZIM_HEADER_SIZE = 80
_ZIM_CHECKSUM_POS_OFFSET = 72
_ZIM_CHECKSUM_LENGTH = 16

# Every ZIM file starts with the little-endian magic number 72173914
# (0x044D495A). Checking these four bytes needs no libzim call, but it is only
# half a readability probe: a truncated download keeps them (see
# ``is_truncated_zim``).
ZIM_MAGIC = b"ZIM\x04"


def _declared_zim_size(header: bytes) -> int | None:
    """Total file size ``header`` claims for itself, or None if it makes none.

    Returns None when the header is too short to carry ``checksumPos`` or the
    field is zero (an archive written without a checksum): there is then no
    claim to compare a file size against, and inventing one would condemn a
    good archive.
    """
    if len(header) < ZIM_HEADER_SIZE:
        return None
    (checksum_pos,) = struct.unpack_from("<Q", header, _ZIM_CHECKSUM_POS_OFFSET)
    if not checksum_pos:
        return None
    return int(checksum_pos) + _ZIM_CHECKSUM_LENGTH


def has_zim_signature(path: Path) -> bool:
    """Return whether ``path`` looks like an archive libzim could open.

    A ``False`` result means the file is not an openable archive (plain
    text, a truncated download, a stray file renamed ``.zim``). Two cheap
    probes back that claim: the ZIM magic bytes, and — since a half-finished
    download keeps those — the header's own declared total size against the
    size on disk (see ``is_truncated_zim``). A ``True`` result remains only a
    plausibility check: it does not verify the archive's integrity.
    """
    try:
        with Path(path).open("rb") as fh:
            header = fh.read(ZIM_HEADER_SIZE)
    except OSError:
        return False
    if header[: len(ZIM_MAGIC)] != ZIM_MAGIC:
        return False
    declared = _declared_zim_size(header)
    if declared is None:
        return True
    try:
        return Path(path).stat().st_size >= declared
    except OSError:
        return True


def is_truncated_zim(path: Path) -> bool:
    """Return whether ``path`` stops short of the size its header declares.

    This is the shape an interrupted multi-GB Kiwix download leaves behind.
    The magic bytes survive a truncation, so a probe that reads only those
    calls the file readable and every later query then fails; comparing the
    on-disk size against the header's own ``checksumPos`` costs one ``stat``
    and catches it. A file *longer* than its declared size is not truncated.
    """
    try:
        with Path(path).open("rb") as fh:
            header = fh.read(ZIM_HEADER_SIZE)
        # v3.3.1 field report (donor): must re-check the magic here too, or a
        # stray text file renamed ".zim" is misreported as truncated.
        if header[: len(ZIM_MAGIC)] != ZIM_MAGIC:
            return False
        if len(header) < ZIM_HEADER_SIZE:
            return True
        declared = _declared_zim_size(header)
        if declared is None:
            return False
        return Path(path).stat().st_size < declared
    except OSError:
        return False


# --- tutor code below (not from openzim-mcp) ---

import enum
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from libzim.reader import Archive


class ArchiveState(enum.Enum):
    """Coarse-grained health of an on-disk ZIM archive."""

    VALID = "valid"
    NO_FULLTEXT_INDEX = "no_fulltext_index"
    TRUNCATED = "truncated"
    NOT_ZIM = "not_zim"
    MISSING = "missing"


@dataclass(frozen=True)
class ArchiveStatus:
    """Result of :func:`validate_archive`."""

    path: Path
    state: ArchiveState
    has_fulltext_index: bool | None = None
    article_count: int | None = None
    error: str | None = None


def validate_archive(path: Path) -> ArchiveStatus:
    """Classify ``path`` without ever raising.

    Cheap signature/truncation checks run first so a missing or obviously
    broken file never reaches libzim's ``Archive`` constructor. Only a file
    that passes those is actually opened, to check for a full-text index and
    read its article count.
    """
    path = Path(path)
    if not path.exists():
        return ArchiveStatus(path=path, state=ArchiveState.MISSING)
    if not has_zim_signature(path):
        if is_truncated_zim(path):
            return ArchiveStatus(path=path, state=ArchiveState.TRUNCATED)
        return ArchiveStatus(path=path, state=ArchiveState.NOT_ZIM)

    try:
        archive = Archive(str(path))
        has_fulltext_index = bool(archive.has_fulltext_index)
        article_count = int(archive.article_count)
    except Exception as exc:  # noqa: BLE001 - validate_archive must never raise
        return ArchiveStatus(path=path, state=ArchiveState.NOT_ZIM, error=str(exc))

    if not has_fulltext_index:
        return ArchiveStatus(
            path=path,
            state=ArchiveState.NO_FULLTEXT_INDEX,
            has_fulltext_index=False,
            article_count=article_count,
        )
    return ArchiveStatus(
        path=path,
        state=ArchiveState.VALID,
        has_fulltext_index=True,
        article_count=article_count,
    )


@dataclass(frozen=True)
class ArchiveFingerprint:
    """Identity of an archive's on-disk bytes plus its own edition metadata.

    ``digest`` deliberately never incorporates the path string: two files
    with identical bytes/mtime/size found at different locations must
    fingerprint identically, so caches keyed on this survive a rename/move.
    """

    uuid: str
    digest: str
    language: str | None
    name: str | None
    title: str | None
    date: str | None


def _read_metadata(archive: Archive, key: str) -> str | None:
    try:
        if key not in archive.metadata_keys:
            return None
        return archive.get_metadata(key).decode("utf-8")
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return None


def fingerprint(path: Path) -> ArchiveFingerprint:
    """Compute a stable identity for the archive at ``path``.

    The digest mixes the archive's UUID with its file size and mtime (never
    the path string), so a byte-identical copy shares the UUID but gets a
    different digest once its mtime differs, while the same file opened via
    two different paths (same inode/mtime) fingerprints identically.
    """
    path = Path(path)
    archive = Archive(str(path))
    uuid = str(archive.uuid)
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(uuid.encode("utf-8"))
    hasher.update(str(stat.st_size).encode("utf-8"))
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return ArchiveFingerprint(
        uuid=uuid,
        digest=hasher.hexdigest(),
        language=_read_metadata(archive, "Language"),
        name=_read_metadata(archive, "Name"),
        title=_read_metadata(archive, "Title"),
        date=_read_metadata(archive, "Date"),
    )


class ArchiveError(Exception):
    """Raised by :func:`open_archive` when ``path`` cannot be opened as a ZIM."""

    def __init__(self, status: ArchiveStatus) -> None:
        self.status = status
        super().__init__(f"Cannot open archive at {status.path}: {status.state.value}")


@contextmanager
def open_archive(path: Path) -> Iterator[Archive]:
    """Validate then open ``path``, raising :class:`ArchiveError` on failure.

    ``NO_FULLTEXT_INDEX`` archives are usable (just flagged elsewhere) and
    open normally here.
    """
    status = validate_archive(path)
    if status.state in (ArchiveState.MISSING, ArchiveState.NOT_ZIM, ArchiveState.TRUNCATED):
        raise ArchiveError(status)
    try:
        archive = Archive(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ArchiveError(status) from exc
    yield archive
