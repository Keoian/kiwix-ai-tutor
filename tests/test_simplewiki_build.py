"""Resumable dense sidecar builder tests (WP-B7).

Uses the shared ``fixture_zim`` (tests/conftest.py) and a deterministic
fake embedder. The core acceptance criterion: interrupting a build partway
through (simulated by raising from the embed callable) and resuming it
yields an index byte-identical to an uninterrupted build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tutor.retrieval.index.manifest import read_manifest
from tutor.retrieval.index.simplewiki_build import BuildConfigError, build_index
from tutor.retrieval.index.simplewiki_store import (
    IDS_FILENAME,
    VECTORS_FILENAME,
    file_size,
)

DIM = 8
MODEL_NAME = "fake-embedder-v1"
MODEL_SHA = "0" * 64


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic, content-derived fake embedding."""
    out = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:DIM]])
    return out


def _build_kwargs(out_dir: Path, **overrides) -> dict:
    kwargs = dict(
        out_dir=out_dir,
        embed=fake_embed,
        archive_id="fixture",
        embedding_model_name=MODEL_NAME,
        embedding_model_sha256=MODEL_SHA,
        dim=DIM,
        batch_size=3,
        checkpoint_every=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_uninterrupted_build_produces_manifest(fixture_zim, tmp_path):
    out_dir = tmp_path / "sidecar"
    manifest = build_index(fixture_zim, **_build_kwargs(out_dir))
    assert manifest.count > 0
    assert manifest.dim == DIM
    assert manifest.embedding_model_name == MODEL_NAME
    loaded = read_manifest(out_dir / "manifest.json")
    assert loaded == manifest


def test_limit_articles_caps_article_count(fixture_zim, tmp_path):
    out_dir_full = tmp_path / "full"
    out_dir_limited = tmp_path / "limited"
    build_index(fixture_zim, **_build_kwargs(out_dir_full))
    limited_manifest = build_index(
        fixture_zim, **_build_kwargs(out_dir_limited, limit_articles=3)
    )
    full_manifest = read_manifest(out_dir_full / "manifest.json")
    assert limited_manifest.count <= full_manifest.count


def test_resume_after_interruption_is_byte_identical(fixture_zim, tmp_path):
    out_dir_uninterrupted = tmp_path / "uninterrupted"
    build_index(fixture_zim, **_build_kwargs(out_dir_uninterrupted))

    out_dir_resumed = tmp_path / "resumed"
    calls = {"n": 0}

    def flaky_embed(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash mid-build")
        return fake_embed(texts)

    with pytest.raises(RuntimeError):
        build_index(fixture_zim, **_build_kwargs(out_dir_resumed, embed=flaky_embed))

    # Resume: same out_dir, config unchanged, no more simulated failures.
    build_index(fixture_zim, **_build_kwargs(out_dir_resumed))

    a_vectors = (out_dir_uninterrupted / VECTORS_FILENAME).read_bytes()
    b_vectors = (out_dir_resumed / VECTORS_FILENAME).read_bytes()
    assert a_vectors == b_vectors

    a_ids = (out_dir_uninterrupted / IDS_FILENAME).read_text(encoding="utf-8")
    b_ids = (out_dir_resumed / IDS_FILENAME).read_text(encoding="utf-8")
    assert a_ids == b_ids

    manifest_a = read_manifest(out_dir_uninterrupted / "manifest.json")
    manifest_b = read_manifest(out_dir_resumed / "manifest.json")
    assert manifest_a.count == manifest_b.count
    assert manifest_a.archive_digest == manifest_b.archive_digest


def test_resume_detects_and_truncates_torn_tail(fixture_zim, tmp_path):
    """A crash between "bytes appended" and "checkpoint written" leaves a
    tail longer than the last checkpoint recorded. Resuming must truncate
    that tail away rather than double-count it."""
    out_dir = tmp_path / "sidecar"
    build_index(fixture_zim, **_build_kwargs(out_dir, checkpoint_every=1000))
    # A checkpoint_every this high with a small fixture ZIM means the only
    # checkpoint written is the final one (post-flush), so simulate a torn
    # tail directly: append garbage bytes past the recorded checkpoint size.
    checkpoint_path = out_dir / "checkpoint.json"
    from tutor.retrieval.index.manifest import read_checkpoint

    checkpoint = read_checkpoint(checkpoint_path)
    vectors_path = out_dir / VECTORS_FILENAME
    ids_path = out_dir / IDS_FILENAME

    with vectors_path.open("ab") as fh:
        fh.write(b"\x00" * (DIM * 2))  # one garbage row, no matching id
    with ids_path.open("ab") as fh:
        fh.write(b"garbage-id-not-flushed\n")

    assert file_size(vectors_path) == checkpoint.vectors_bytes + DIM * 2
    assert file_size(ids_path) == checkpoint.ids_bytes + len(b"garbage-id-not-flushed\n")

    # Rebuild fresh for comparison.
    out_dir_clean = tmp_path / "clean"
    build_index(fixture_zim, **_build_kwargs(out_dir_clean, checkpoint_every=1000))

    # Resuming the torn directory (with the *same* config so it is accepted
    # as a resume, not a fresh build) must reproduce the clean build exactly
    # -- proving the torn tail was truncated away before re-processing.
    build_index(fixture_zim, **_build_kwargs(out_dir, checkpoint_every=1000))

    assert vectors_path.read_bytes() == (out_dir_clean / VECTORS_FILENAME).read_bytes()
    assert ids_path.read_text(encoding="utf-8") == (
        out_dir_clean / IDS_FILENAME
    ).read_text(encoding="utf-8")


def test_resume_with_different_model_config_raises(fixture_zim, tmp_path):
    out_dir = tmp_path / "sidecar"
    calls = {"n": 0}

    def flaky_embed(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return fake_embed(texts)

    with pytest.raises(RuntimeError):
        build_index(fixture_zim, **_build_kwargs(out_dir, embed=flaky_embed))

    with pytest.raises(BuildConfigError):
        build_index(fixture_zim, **_build_kwargs(out_dir, embedding_model_name="other-model"))
