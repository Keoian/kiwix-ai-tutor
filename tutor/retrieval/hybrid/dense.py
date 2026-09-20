"""Dense (embedding) retrieval sidecar: brute-force cosine top-k (WP-B7).

Ours: no donor code. See docs/dense_sidecar.md for the file formats and the
rationale for reaching for numpy here specifically (a pure-Python brute
force over ~100k x 384 fp16 vectors measured at ~2.4s per query in this
project's own benchmark -- roughly 50x over the < 50ms CPU query budget in
docs/plan/offline_tutor_implementation_plan.md's WP-B7 acceptance criterion
-- while numpy's BLAS-backed matrix-vector product clears it comfortably;
see the measured numbers in docs/dense_sidecar.md). numpy is the only
non-stdlib dependency this module adds, used solely for the search-time
dot product and top-k selection; the on-disk format is plain struct-packed
fp16, not a numpy-specific format.

A dense index lives in a directory (a "sidecar") containing
``vectors.fp16``, ``ids.txt``, and ``manifest.json`` (see
``tutor.retrieval.index.manifest`` and ``.simplewiki_store`` for the exact
formats). ``DenseIndex.open`` refuses a sidecar whose manifest was built
against a different archive fingerprint digest than the one the caller
passes in -- the archive on disk has since been replaced or re-downloaded,
so any passage id the sidecar could return is no longer guaranteed to
resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tutor.retrieval.index.manifest import DenseManifest, ManifestError, read_manifest
from tutor.retrieval.index.simplewiki_store import (
    IDS_FILENAME,
    MANIFEST_FILENAME,
    VECTORS_FILENAME,
    read_ids,
)


class DenseIndexError(Exception):
    """Raised when a dense sidecar cannot be opened or queried."""


@dataclass(frozen=True)
class DenseIndex:
    """An opened dense sidecar: ids, an in-memory float32 matrix, manifest."""

    ids: tuple[str, ...]
    vectors: np.ndarray  # shape (count, dim), float32, L2-normalised rows
    manifest: DenseManifest

    @classmethod
    def open(cls, directory: Path, *, archive_digest: str) -> DenseIndex:
        """Open the dense sidecar at ``directory``.

        Raises :class:`DenseIndexError` if no completed manifest is present,
        if it is malformed, or if its recorded archive fingerprint digest
        does not match ``archive_digest`` (a stale sidecar: the archive it
        was built from has since changed).
        """
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILENAME
        try:
            manifest = read_manifest(manifest_path)
        except ManifestError as exc:
            raise DenseIndexError(str(exc)) from exc
        if manifest is None:
            raise DenseIndexError(
                f"no completed dense sidecar at {directory} (missing {MANIFEST_FILENAME})"
            )
        if manifest.archive_digest != archive_digest:
            raise DenseIndexError(
                "stale dense sidecar: manifest was built for archive digest "
                f"{manifest.archive_digest!r}, current archive digest is "
                f"{archive_digest!r}. Rebuild the sidecar."
            )

        ids = read_ids(directory / IDS_FILENAME)
        if len(ids) != manifest.count:
            raise DenseIndexError(
                f"corrupt dense sidecar: manifest says {manifest.count} rows, "
                f"{IDS_FILENAME} has {len(ids)} ids"
            )

        vectors_path = directory / VECTORS_FILENAME
        expected_bytes = manifest.count * manifest.dim * 2
        actual_bytes = vectors_path.stat().st_size if vectors_path.exists() else 0
        if actual_bytes != expected_bytes:
            raise DenseIndexError(
                f"corrupt dense sidecar: expected {expected_bytes} bytes in "
                f"{VECTORS_FILENAME}, found {actual_bytes}"
            )

        raw = np.fromfile(vectors_path, dtype="<f2", count=manifest.count * manifest.dim)
        vectors = raw.astype(np.float32).reshape(manifest.count, manifest.dim)
        return cls(ids=tuple(ids), vectors=vectors, manifest=manifest)

    def search(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        """Brute-force cosine top-``k`` over the whole index.

        ``query_vec`` is L2-normalised here regardless of the manifest's
        normalisation, and index rows are assumed already unit-length (the
        builder normalises before storing, manifest.normalisation ==
        "l2"), so the score is a plain dot product == cosine similarity.
        """
        if len(query_vec) != self.manifest.dim:
            raise DenseIndexError(
                f"query vector has dim {len(query_vec)}, index dim is {self.manifest.dim}"
            )
        q = np.asarray(query_vec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        if len(self.ids) == 0 or k <= 0:
            return []

        scores = self.vectors @ q
        k = min(k, len(self.ids))
        # argpartition for the top-k unordered, then sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.ids[i], float(scores[i])) for i in top_idx]
