"""RED tests for ``ResearchResponse.unknown_terms`` (owner request
2026-09-21): the query's content terms that had ~zero corpus-wide matches
AND that neither the spelling nor the compound-split fallback managed to
correct. This lets the app later catch a phonetic misspelling the
edit-distance fallback cannot reach ("ardweeno" for Arduino, edit distance
3) and ask "Did you mean X?" instead of answering "not found".

Reuses the same fake-worker / fixture pattern as the spelling-fallback
tests in ``tests/test_research.py`` (``_EntityFakeWorker``,
``_MisspellFakeWorker``) so nothing here reads gitignored ``data/`` or a
real ZIM.
"""

from __future__ import annotations

import pytest

from tests.test_research import (
    _EntityFakeWorker,
    _MisspellFakeWorker,
    _write_registry_toml,
)
from tutor.retrieval.research import ResearchResponse
from tutor.retrieval.snapshots import SnapshotStore
from tutor.retrieval.zim.worker import WorkerResult


@pytest.fixture
def registry_toml(tmp_path, fixture_zim):
    return _write_registry_toml(
        tmp_path,
        fixture_zim,
        [{"id": "tier1", "tier": 1, "subjects": []}],
    )


@pytest.fixture
def snapshot_store(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    yield store
    store.close()


def _engine(registry_toml, snapshot_store, tmp_path, **kwargs):
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    cache_dir = kwargs.pop("cache_dir", None) or (tmp_path / "cache")
    from tutor.retrieval.research import ResearchEngine

    return ResearchEngine(
        registry,
        snapshot_store=snapshot_store,
        cache_dir=cache_dir,
        **kwargs,
    )


class _NoCandidateWorker(_EntityFakeWorker):
    """A term with zero corpus-wide matches and no fuzzy title-suggestion
    candidate at all -- a genuinely absent/nonsense word, not a
    misspelling of anything in the archive."""

    def request(self, op: str, *, deadline_s: float, **kwargs) -> WorkerResult:
        if op == "search_titles" and kwargs.get("query", "").strip() == "xylophonium":
            return WorkerResult(status="ok", value=[], error=None, elapsed_s=0.0)
        return super().request(op, deadline_s=deadline_s, **kwargs)


def test_unresolved_zero_match_term_is_unknown(registry_toml, snapshot_store, tmp_path):
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        worker_factory=lambda path: _NoCandidateWorker(path),
    )
    entry = registry.for_subject(None)[0]
    unknown_terms_out: list[list[str]] = []
    engine._process_archive(
        entry,
        "What is the boiling point of xylophonium?",
        lambda: 5.0,
        unknown_terms_out=unknown_terms_out,
    )
    assert unknown_terms_out
    assert "xylophonium" in unknown_terms_out[0]


def test_unresolved_zero_match_term_surfaces_on_response(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        worker_factory=lambda path: _NoCandidateWorker(path),
    )
    resp = engine.research("What is the boiling point of xylophonium?")
    assert "xylophonium" in resp.unknown_terms
    assert "xylophonium" in resp.to_dict()["unknown_terms"]


def test_spelling_corrected_term_is_not_unknown(registry_toml, snapshot_store, tmp_path):
    """A term the spelling fallback DID correct (e.g. "heluim" ->
    "helium") must not also be reported as unknown."""
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        worker_factory=lambda path: _MisspellFakeWorker(path),
    )
    resp = engine.research("What is the boiling point of heluim?")
    assert "heluim" not in resp.unknown_terms
    assert resp.corrected_terms == {"heluim": "helium"}


def test_well_known_term_is_not_unknown(registry_toml, snapshot_store, tmp_path):
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        worker_factory=lambda path: _EntityFakeWorker(path),
    )
    resp = engine.research("Output the boiling point of helium in celsius and farenheit.")
    assert "helium" not in resp.unknown_terms
    assert "celsius" not in resp.unknown_terms


def test_response_built_without_field_defaults_to_empty_list():
    resp = ResearchResponse(
        version=1,
        status="empty",
        passages=[],
        coverage={},
        archives_consulted=[],
    )
    assert resp.unknown_terms == []
    assert resp.to_dict()["unknown_terms"] == []
