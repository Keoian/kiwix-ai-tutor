"""Unit tests for tutor.retrieval.hybrid.dense.DenseIndex (WP-B7)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tutor.retrieval.hybrid.dense import DenseIndex, DenseIndexError
from tutor.retrieval.index.manifest import DenseManifest, write_json_atomic
from tutor.retrieval.index.simplewiki_store import (
    IDS_FILENAME,
    MANIFEST_FILENAME,
    VECTORS_FILENAME,
    append_bytes,
    append_ids,
    pack_vectors,
)

DIM = 4


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _write_index(
    tmp_path: Path,
    *,
    archive_digest: str = "digest-a",
    ids: list[str] | None = None,
    vectors: list[list[float]] | None = None,
    dim: int = DIM,
    complete: bool = True,
) -> Path:
    out_dir = tmp_path / "sidecar"
    out_dir.mkdir()
    if ids is None:
        ids = ["id0", "id1", "id2"]
    if vectors is None:
        vectors = [_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([1, 1, 0, 0])]
    append_bytes(out_dir / VECTORS_FILENAME, pack_vectors(vectors, dim))
    append_ids(out_dir / IDS_FILENAME, ids)
    if complete:
        manifest = DenseManifest(
            version=1,
            archive_digest=archive_digest,
            extractor_version="zim-bundle-v1",
            embedding_model_name="test-model",
            embedding_model_sha256="a" * 64,
            dim=dim,
            count=len(ids),
            normalisation="l2",
            created_at="2026-01-01T00:00:00+00:00",
        )
        write_json_atomic(out_dir / MANIFEST_FILENAME, manifest.to_dict())
    return out_dir


def test_open_and_search_returns_best_match_first(tmp_path):
    out_dir = _write_index(tmp_path)
    index = DenseIndex.open(out_dir, archive_digest="digest-a")

    results = index.search(_unit([1, 0, 0, 0]), k=2)

    assert len(results) == 2
    assert results[0][0] == "id0"
    assert results[0][1] == pytest.approx(1.0, abs=1e-2)
    ids = {r[0] for r in results}
    assert "id0" in ids


def test_search_k_larger_than_count_returns_all(tmp_path):
    out_dir = _write_index(tmp_path)
    index = DenseIndex.open(out_dir, archive_digest="digest-a")
    results = index.search(_unit([1, 0, 0, 0]), k=100)
    assert len(results) == 3


def test_open_missing_manifest_raises(tmp_path):
    out_dir = _write_index(tmp_path, complete=False)
    with pytest.raises(DenseIndexError):
        DenseIndex.open(out_dir, archive_digest="digest-a")


def test_open_stale_fingerprint_raises(tmp_path):
    out_dir = _write_index(tmp_path, archive_digest="digest-a")
    with pytest.raises(DenseIndexError, match="stale"):
        DenseIndex.open(out_dir, archive_digest="digest-b")


def test_open_corrupt_ids_count_mismatch_raises(tmp_path):
    out_dir = _write_index(tmp_path)
    # Corrupt: append an extra id line without a matching vector row.
    append_ids(out_dir / IDS_FILENAME, ["extra"])
    with pytest.raises(DenseIndexError):
        DenseIndex.open(out_dir, archive_digest="digest-a")


def test_open_corrupt_vector_bytes_mismatch_raises(tmp_path):
    out_dir = _write_index(tmp_path)
    with (out_dir / VECTORS_FILENAME).open("ab") as fh:
        fh.write(b"\x00\x01")  # one stray byte, not a full row
    with pytest.raises(DenseIndexError):
        DenseIndex.open(out_dir, archive_digest="digest-a")


def test_search_wrong_dim_query_raises(tmp_path):
    out_dir = _write_index(tmp_path)
    index = DenseIndex.open(out_dir, archive_digest="digest-a")
    with pytest.raises(DenseIndexError):
        index.search([1.0, 0.0], k=1)


def test_search_empty_index_returns_empty(tmp_path):
    out_dir = _write_index(tmp_path, ids=[], vectors=[])
    index = DenseIndex.open(out_dir, archive_digest="digest-a")
    assert index.search(_unit([1, 0, 0, 0]), k=5) == []


def test_open_with_paths_search_returns_triples(tmp_path):
    from tutor.retrieval.index.simplewiki_store import PATHS_FILENAME
    from tutor.retrieval.index.simplewiki_store import append_ids as _append

    out_dir = _write_index(tmp_path)
    _append(out_dir / PATHS_FILENAME, ["Article/A", "Article/B", "Article/C"])
    index = DenseIndex.open(out_dir, archive_digest="digest-a")
    results = index.search(_unit([1, 0, 0, 0]), k=2)
    assert len(results) == 2
    assert len(results[0]) == 3
    assert results[0][0] == "id0"
    assert results[0][1] == "Article/A"


def test_open_without_paths_search_returns_pairs(tmp_path):
    out_dir = _write_index(tmp_path)
    index = DenseIndex.open(out_dir, archive_digest="digest-a")
    assert index.paths is None
    results = index.search(_unit([1, 0, 0, 0]), k=2)
    assert len(results[0]) == 2


def test_open_with_mismatched_paths_count_raises(tmp_path):
    from tutor.retrieval.index.simplewiki_store import PATHS_FILENAME
    from tutor.retrieval.index.simplewiki_store import append_ids as _append

    out_dir = _write_index(tmp_path)
    _append(out_dir / PATHS_FILENAME, ["Article/A"])  # too few
    with pytest.raises(DenseIndexError):
        DenseIndex.open(out_dir, archive_digest="digest-a")
