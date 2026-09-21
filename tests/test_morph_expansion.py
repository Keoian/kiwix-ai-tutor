"""Rule 2: morphological-variant fallback (tutor/retrieval/morph.py) wired
into ResearchEngine._process_archive -- see research.py's
``_expand_morph_variants`` and its call site (searched for
"Rule 2 (morphological-variant fallback"). Uses the same fake-worker
pattern as the spelling/compound fallback tests in tests/test_research.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.zim.worker import WorkerResult


def _write_registry_toml(tmp_path: Path, fixture_zim: Path) -> Path:
    lines = [
        "[[archive]]",
        'id = "tier1"',
        f'path = "{fixture_zim.as_posix()}"',
        "tier = 1",
        'kind = "encyclopedia"',
        "subjects = []",
        'storage = "ssd"',
        "",
    ]
    toml_path = tmp_path / "registry.toml"
    toml_path.write_text("\n".join(lines), encoding="utf-8")
    return toml_path


@pytest.fixture
def registry_toml(tmp_path, fixture_zim):
    return _write_registry_toml(tmp_path, fixture_zim)


@pytest.fixture
def snapshot_store(tmp_path):
    from tutor.retrieval.snapshots import SnapshotStore

    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    yield store
    store.close()


def _engine(registry_toml, snapshot_store, tmp_path, **kwargs):
    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.research import ResearchEngine

    registry = load_registry(registry_toml)
    cache_dir = kwargs.pop("cache_dir", None) or (tmp_path / "cache")
    return ResearchEngine(registry, snapshot_store=snapshot_store, cache_dir=cache_dir, **kwargs)


def _hits(pairs, source: str):
    from tutor.retrieval.zim.search import SearchHit

    return [
        SearchHit(path=p, title=t, rank=i, snippet="", source=source)
        for i, (p, t) in enumerate(pairs)
    ]


class _GardenFakeWorker:
    """The question "square foot garden" AND-queries for the literal word
    "garden", which never occurs in the fake corpus -- the real article is
    titled "Square foot gardening" and contains only the word "gardening".
    "garden" and the joined AND query both have zero estimated matches;
    "gardening" (a morph_variants("garden") candidate) has real, plentiful
    matches -- the signal the expansion needs to try it.
    """

    _MATCHES = {"square": 8109, "foot": 2515, "garden": 0, "gardening": 450}
    _HTML = (
        "<html><head><title>Square foot gardening</title></head>"
        "<body><h1>Square foot gardening</h1>"
        "<p>Square foot gardening is a method of gardening in small "
        "spaces.</p></body></html>"
    )

    def __init__(self, archive_path: Path) -> None:
        self.ops_count = 0

    def request(self, op: str, *, deadline_s: float, **kwargs) -> WorkerResult:
        if op == "multi":
            results = []
            for sub_op, sub_kwargs in kwargs.get("ops", []):
                sub_res = self.request(sub_op, deadline_s=deadline_s, **sub_kwargs)
                results.append(
                    {"status": sub_res.status, "value": sub_res.value, "error": sub_res.error}
                )
            return WorkerResult(status="ok", value=results, error=None, elapsed_s=0.0)
        self.ops_count += 1
        if op == "estimated_matches":
            term = kwargs["term"].strip('"')
            return WorkerResult(
                status="ok", value=self._MATCHES.get(term, 0), error=None, elapsed_s=0.0
            )
        if op == "search_fulltext":
            terms = set(kwargs["query"].split())
            if "gardening" in terms:
                return WorkerResult(
                    status="ok",
                    value=_hits(
                        [("Square_foot_gardening", "Square foot gardening")], "fulltext"
                    ),
                    error=None,
                    elapsed_s=0.0,
                )
            return WorkerResult(status="ok", value=[], error=None, elapsed_s=0.0)
        if op == "search_titles":
            terms = set(kwargs["query"].split())
            if "gardening" in terms:
                return WorkerResult(
                    status="ok",
                    value=_hits(
                        [("Square_foot_gardening", "Square foot gardening")], "title"
                    ),
                    error=None,
                    elapsed_s=0.0,
                )
            return WorkerResult(status="ok", value=[], error=None, elapsed_s=0.0)
        if op == "fetch_entry":
            from tutor.retrieval.zim.search import FetchedEntry

            return WorkerResult(
                status="ok",
                value=FetchedEntry(
                    path="Square_foot_gardening",
                    title="Square foot gardening",
                    mimetype="text/html",
                    html=self._HTML,
                    hops=(),
                ),
                error=None,
                elapsed_s=0.0,
            )
        return WorkerResult(status="error", value=None, error=f"unknown op {op}", elapsed_s=0.0)

    def close(self) -> None:
        pass


class _StrongFakeWorker:
    """A first pass that already succeeds outright -- the joined AND query
    directly returns the right article with real matches. Morph expansion
    must never run: no extra `estimated_matches`/search ops beyond the
    ones the normal (pre-morph) pipeline already issues.
    """

    _HTML = (
        "<html><head><title>Helium</title></head>"
        "<body><h1>Helium</h1><p>Helium is a chemical element.</p></body></html>"
    )

    def __init__(self, archive_path: Path) -> None:
        self.morph_ops: list[str] = []

    def request(self, op: str, *, deadline_s: float, **kwargs) -> WorkerResult:
        if op == "multi":
            results = []
            for sub_op, sub_kwargs in kwargs.get("ops", []):
                sub_res = self.request(sub_op, deadline_s=deadline_s, **sub_kwargs)
                results.append(
                    {"status": sub_res.status, "value": sub_res.value, "error": sub_res.error}
                )
            return WorkerResult(status="ok", value=results, error=None, elapsed_s=0.0)
        if op == "estimated_matches":
            # Any lookup at all beyond the plain query terms (i.e. any
            # morph-variant candidate string) signals the expansion ran.
            term = kwargs["term"]
            if term not in ("helium", "what is helium"):
                self.morph_ops.append(term)
            return WorkerResult(status="ok", value=500, error=None, elapsed_s=0.0)
        if op == "search_fulltext":
            return WorkerResult(
                status="ok", value=_hits([("Helium", "Helium")], "fulltext"), error=None,
                elapsed_s=0.0,
            )
        if op == "search_titles":
            return WorkerResult(
                status="ok", value=_hits([("Helium", "Helium")], "title"), error=None,
                elapsed_s=0.0,
            )
        if op == "fetch_entry":
            from tutor.retrieval.zim.search import FetchedEntry

            return WorkerResult(
                status="ok",
                value=FetchedEntry(
                    path="Helium", title="Helium", mimetype="text/html", html=self._HTML, hops=()
                ),
                error=None,
                elapsed_s=0.0,
            )
        return WorkerResult(status="error", value=None, error=f"unknown op {op}", elapsed_s=0.0)

    def close(self) -> None:
        pass


def test_morph_expansion_finds_gardening_article(registry_toml, snapshot_store, tmp_path):
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    engine = _engine(
        registry_toml, snapshot_store, tmp_path, worker_factory=lambda p: _GardenFakeWorker(p)
    )
    entry = registry.for_subject(None)[0]
    expanded_terms_out: dict[str, list[str]] = {}
    candidates, _timed_out, _note, _key_facts = engine._process_archive(
        entry,
        "square foot garden",
        lambda: 5.0,
        expanded_terms_out=expanded_terms_out,
    )
    paths = {c["path"] for c in candidates}
    assert "Square_foot_gardening" in paths, candidates
    assert "garden" in expanded_terms_out
    assert "gardening" in expanded_terms_out["garden"]


def test_morph_expansion_skipped_on_strong_result(registry_toml, snapshot_store, tmp_path):
    engine = _engine(
        registry_toml, snapshot_store, tmp_path, worker_factory=lambda p: _StrongFakeWorker(p)
    )
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    entry = registry.for_subject(None)[0]
    worker = engine._get_worker(entry)
    expanded_terms_out: dict[str, list[str]] = {}
    engine._process_archive(
        entry, "what is helium", lambda: 5.0, expanded_terms_out=expanded_terms_out
    )
    assert worker.morph_ops == [], worker.morph_ops
    assert expanded_terms_out == {}


def test_morph_candidates_absent_from_archive_not_used(registry_toml, snapshot_store, tmp_path):
    """A morph_variants() candidate ("gardened", "gardener", ...) that the
    fake archive reports zero matches for must never be used -- only
    "gardening" (>= the match floor) qualifies."""
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    engine = _engine(
        registry_toml, snapshot_store, tmp_path, worker_factory=lambda p: _GardenFakeWorker(p)
    )
    entry = registry.for_subject(None)[0]
    expanded_terms_out: dict[str, list[str]] = {}
    engine._process_archive(
        entry, "square foot garden", lambda: 5.0, expanded_terms_out=expanded_terms_out
    )
    assert expanded_terms_out["garden"] == ["gardening"]


def test_morph_expansion_skipped_when_deadline_exhausted(registry_toml, snapshot_store, tmp_path):
    engine = _engine(
        registry_toml, snapshot_store, tmp_path, worker_factory=lambda p: _GardenFakeWorker(p)
    )
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    entry = registry.for_subject(None)[0]
    expanded_terms_out: dict[str, list[str]] = {}
    # remaining() already exhausted -> expansion must not run, but a result
    # (possibly empty) is still returned without raising.
    candidates, _timed_out, _note, _key_facts = engine._process_archive(
        entry, "square foot garden", lambda: 0.0, expanded_terms_out=expanded_terms_out
    )
    assert expanded_terms_out == {}
    assert isinstance(candidates, list)


def test_morph_toggle_off_matches_pre_morph_behavior(
    registry_toml, snapshot_store, tmp_path, monkeypatch
):
    monkeypatch.setenv("TUTOR_RETRIEVAL_MORPH_VARIANTS", "0")
    import importlib

    from tutor.retrieval import research as research_module

    importlib.reload(research_module)
    try:
        registry = research_module.Registry if False else None  # noqa: F841
        from tutor.retrieval.registry import load_registry

        registry = load_registry(registry_toml)
        engine = research_module.ResearchEngine(
            registry,
            snapshot_store=snapshot_store,
            cache_dir=tmp_path / "cache",
            worker_factory=lambda p: _GardenFakeWorker(p),
        )
        entry = registry.for_subject(None)[0]
        expanded_terms_out: dict[str, list[str]] = {}
        candidates, _timed_out, _note, _key_facts = engine._process_archive(
            entry, "square foot garden", lambda: 5.0, expanded_terms_out=expanded_terms_out
        )
        assert expanded_terms_out == {}
        assert candidates == []
    finally:
        monkeypatch.delenv("TUTOR_RETRIEVAL_MORPH_VARIANTS", raising=False)
        importlib.reload(research_module)
