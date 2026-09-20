"""Manifest and checkpoint file formats for the dense sidecar build (WP-B7).

Ours: no donor code. Two small JSON files live alongside the flat vector
file and the ids file in a sidecar directory:

``manifest.json``
    Written only once the build is *complete*. Its presence is what
    :meth:`tutor.retrieval.hybrid.dense.DenseIndex.open` requires; a
    directory holding only a ``checkpoint.json`` is an unfinished build.

``checkpoint.json``
    Written periodically during a build (see ``simplewiki_build.py``) so an
    interrupted build can resume without re-embedding already-embedded
    passages and without duplicating rows. It records the archive/model
    identity the in-progress files were built against (so a config change
    is never silently resumed), the next archive entry id to process, and
    the exact byte/line lengths the vector and id files should have as of
    that checkpoint -- letting the builder truncate away any partially
    written tail from an interrupted append before resuming.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST_VERSION = 1


class ManifestError(Exception):
    """Raised on a malformed or version-mismatched manifest/checkpoint file."""


@dataclass(frozen=True)
class DenseManifest:
    """Identity and shape of a completed dense sidecar."""

    version: int
    archive_digest: str
    extractor_version: str
    embedding_model_name: str
    embedding_model_sha256: str
    dim: int
    count: int
    normalisation: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> DenseManifest:
        try:
            return DenseManifest(
                version=int(data["version"]),
                archive_digest=str(data["archive_digest"]),
                extractor_version=str(data["extractor_version"]),
                embedding_model_name=str(data["embedding_model_name"]),
                embedding_model_sha256=str(data["embedding_model_sha256"]),
                dim=int(data["dim"]),
                count=int(data["count"]),
                normalisation=str(data["normalisation"]),
                created_at=str(data["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"malformed manifest: {exc}") from exc


@dataclass(frozen=True)
class BuildCheckpoint:
    """In-progress build state, enough to resume without duplicating rows."""

    archive_digest: str
    extractor_version: str
    embedding_model_name: str
    embedding_model_sha256: str
    dim: int
    normalisation: str
    next_entry_id: int
    articles_processed: int
    count: int
    vectors_bytes: int
    ids_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> BuildCheckpoint:
        try:
            return BuildCheckpoint(
                archive_digest=str(data["archive_digest"]),
                extractor_version=str(data["extractor_version"]),
                embedding_model_name=str(data["embedding_model_name"]),
                embedding_model_sha256=str(data["embedding_model_sha256"]),
                dim=int(data["dim"]),
                normalisation=str(data["normalisation"]),
                next_entry_id=int(data["next_entry_id"]),
                articles_processed=int(data["articles_processed"]),
                count=int(data["count"]),
                vectors_bytes=int(data["vectors_bytes"]),
                ids_bytes=int(data["ids_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"malformed checkpoint: {exc}") from exc


def write_json_atomic(path: Path, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` via temp-file + atomic replace."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_manifest(path: Path) -> DenseManifest | None:
    data = read_json(path)
    if data is None:
        return None
    return DenseManifest.from_dict(data)


def read_checkpoint(path: Path) -> BuildCheckpoint | None:
    data = read_json(path)
    if data is None:
        return None
    return BuildCheckpoint.from_dict(data)
