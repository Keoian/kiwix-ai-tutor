"""RED tests for tutor.retrieval.research (the single research() entry point).

Module does not exist yet; every test here should fail with
ModuleNotFoundError until implemented. Authoritative sources:

- offline_tutor_spec_v0.3.md §7 (contract, pipeline, response, deadlines)
  and §7.4 ("Soft deadline 3 s, hard 8 s ... kills the worker at the hard
  deadline, returns completed stages as `partial`").
- offline_tutor_kiwix_reuse_plan.md §4.3 ("partial success" fields:
  archives searched/failed, limits hit, timeout, missing index, fallback
  route) and the RRF/BM25/diversity/packer sections.
- offline_tutor_implementation_plan.md §0.4 (search order tier1->2->3,
  subject routing, `[Q&A]` marker, storage class in log lines) and
  §10.8-9 (ranking flags default OFF; ONE entry point `research()`).

Contract decisions made here (spec/reuse-plan disagreed or were silent):
- `ResearchResponse.version` is a plain int starting at 1 (neither doc
  pins a format; chosen for trivial JSON round-tripping).
- `status` values are exactly {"ok", "partial", "empty", "error"} per
  reuse-plan §4.3 "partial success" concept plus spec's "empty" case for
  a nonsense query (spec never enumerates the closed set explicitly).
- Deadline test uses "deadline + 1.5 s" wall-clock slack (test-harness
  allowance, not a spec number) because the spec's own hard-deadline
  worker-kill machinery (already built in tutor/retrieval/zim/worker.py)
  needs a moment to actually terminate the child process.
- Tier-3 consultation trigger: spec §7.2 says tier2 is used only when
  tier-1 coverage is weak, and implementation_plan §0.4 adds tier-3 by
  subject hint; this test file additionally requires tier-3 for
  procedural/pedagogical phrasing ("how do I", "how would you explain")
  per implementation_plan §0.4, which the spec text does not mention.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tutor.retrieval.research import RankingFlags, ResearchEngine, compute_coverage
from tutor.retrieval.snapshots import SnapshotStore
from tutor.retrieval.zim.worker import WorkerResult


def _write_registry_toml(tmp_path: Path, fixture_zim: Path, entries: list[dict]) -> Path:
    lines = []
    for e in entries:
        lines.append("[[archive]]")
        lines.append(f'id = "{e["id"]}"')
        lines.append(f'path = "{fixture_zim.as_posix()}"')
        lines.append(f'tier = {e["tier"]}')
        lines.append(f'kind = "{e.get("kind", "encyclopedia")}"')
        subj = ", ".join(f'"{s}"' for s in e.get("subjects", []))
        lines.append(f"subjects = [{subj}]")
        lines.append(f'storage = "{e.get("storage", "ssd")}"')
        lines.append("")
    toml_path = tmp_path / "registry.toml"
    toml_path.write_text("\n".join(lines), encoding="utf-8")
    return toml_path


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
    return ResearchEngine(
        registry,
        snapshot_store=snapshot_store,
        cache_dir=cache_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# RankingFlags
# ---------------------------------------------------------------------------


def test_ranking_flags_all_default_false():
    flags = RankingFlags()
    assert flags.title_boost is False
    assert flags.mention_penalty is False
    assert flags.heading_affinity is False
    assert flags.lead_augmentation is False


# ---------------------------------------------------------------------------
# Basic research() behavior against the fixture ZIM
# ---------------------------------------------------------------------------


def test_research_returns_relevant_article_first(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("What is the Pythagorean theorem?", budget_tokens=2000)
    assert resp.status in ("ok", "partial")
    assert resp.passages
    assert resp.passages[0].path == "pythagorean_theorem"
    assert resp.passages[0].label == "S1"


def test_response_to_dict_is_json_serializable(registry_toml, snapshot_store, tmp_path):
    import json

    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem")
    d = resp.to_dict()
    assert isinstance(d["version"], int)
    assert d["status"] in ("ok", "partial", "empty", "error")
    json.dumps(d)  # must not raise


def test_passages_never_exceed_token_budget(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem", budget_tokens=200)
    total = sum(getattr(p, "estimated_tokens", len(p.text.split())) for p in resp.passages)
    assert total <= 200


def test_passage_fields_present(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem")
    p = resp.passages[0]
    for attr in (
        "label", "passage_id", "archive_id", "title", "path",
        "heading_path", "text", "start", "end", "score", "kind",
    ):
        assert hasattr(p, attr)


def test_every_returned_passage_was_snapshotted(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem")
    for p in resp.passages:
        snap = snapshot_store.get(p.passage_id)
        assert snap is not None
        assert snap.text == p.text


# ---------------------------------------------------------------------------
# Stability across instances / after cache deletion
# ---------------------------------------------------------------------------


def test_passage_ids_and_offsets_stable_across_engine_instances(
    registry_toml, snapshot_store, tmp_path
):
    engine1 = _engine(registry_toml, snapshot_store, tmp_path, cache_dir=tmp_path / "cache")
    resp1 = engine1.research("Pythagorean theorem")

    engine2 = _engine(registry_toml, snapshot_store, tmp_path, cache_dir=tmp_path / "cache")
    resp2 = engine2.research("Pythagorean theorem")

    ids1 = [(p.passage_id, p.start, p.end) for p in resp1.passages]
    ids2 = [(p.passage_id, p.start, p.end) for p in resp2.passages]
    assert ids1 == ids2


def test_passage_ids_stable_after_deleting_cache_dir(registry_toml, snapshot_store, tmp_path):
    import shutil

    cache_dir = tmp_path / "cache"
    engine1 = _engine(registry_toml, snapshot_store, tmp_path, cache_dir=cache_dir)
    resp1 = engine1.research("Pythagorean theorem")
    ids1 = [(p.passage_id, p.start, p.end) for p in resp1.passages]

    shutil.rmtree(cache_dir, ignore_errors=True)

    engine2 = _engine(registry_toml, snapshot_store, tmp_path, cache_dir=cache_dir)
    resp2 = engine2.research("Pythagorean theorem")
    ids2 = [(p.passage_id, p.start, p.end) for p in resp2.passages]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# Caching: identical query twice
# ---------------------------------------------------------------------------


def test_identical_query_twice_identical_except_timings(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp1 = engine.research("Pythagorean theorem")
    resp2 = engine.research("Pythagorean theorem")

    d1 = resp1.to_dict()
    d2 = resp2.to_dict()
    d1.pop("timings", None)
    d2.pop("timings", None)
    assert d1 == d2
    assert resp2.timings.get("cache_hit") is True


# ---------------------------------------------------------------------------
# Deadline / worker injection
# ---------------------------------------------------------------------------


class _SleepyWorker:
    """Fake worker whose every request sleeps past any reasonable deadline."""

    def __init__(self, archive_path: Path) -> None:
        self._archive_path = archive_path

    def request(self, op: str, *, deadline_s: float, **kwargs) -> WorkerResult:
        time.sleep(deadline_s + 5.0)
        return WorkerResult(
            status="timeout", value=None, error="forced timeout", elapsed_s=deadline_s
        )

    def close(self) -> None:
        pass


def test_deadline_never_raises_and_returns_partial_or_empty(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        worker_factory=lambda path: _SleepyWorker(path),
    )
    deadline_s = 1.0
    started = time.monotonic()
    resp = engine.research("Pythagorean theorem", deadline_s=deadline_s)
    elapsed = time.monotonic() - started
    assert elapsed <= deadline_s + 1.5
    assert resp.status in ("partial", "empty")


# ---------------------------------------------------------------------------
# Nonsense query
# ---------------------------------------------------------------------------


def test_nonsense_query_returns_empty_with_weak_coverage_flag(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("zzqxw fnorble asdkjqwe blorptastic")
    assert resp.status == "empty"
    assert resp.coverage.get("weak") is True


# ---------------------------------------------------------------------------
# Subject routing / tier gating
# ---------------------------------------------------------------------------


@pytest.fixture
def two_tier_registry_toml(tmp_path, fixture_zim):
    return _write_registry_toml(
        tmp_path,
        fixture_zim,
        [
            {"id": "tier1", "tier": 1, "subjects": []},
            {"id": "tier3-math", "tier": 3, "subjects": ["math"]},
        ],
    )


def test_tier3_not_consulted_for_unrelated_topic_hint(
    two_tier_registry_toml, snapshot_store, tmp_path
):
    engine = _engine(two_tier_registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem", topic_hint="history")
    consulted_ids = {a["id"] for a in resp.archives_consulted}
    assert "tier3-math" not in consulted_ids


def test_tier3_consulted_for_procedural_query(two_tier_registry_toml, snapshot_store, tmp_path):
    engine = _engine(two_tier_registry_toml, snapshot_store, tmp_path)
    resp = engine.research(
        "How do I prove the Pythagorean theorem?", topic_hint="math"
    )
    consulted_ids = {a["id"] for a in resp.archives_consulted}
    assert "tier3-math" in consulted_ids


def test_archives_consulted_carries_storage_class(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem")
    assert resp.archives_consulted
    for a in resp.archives_consulted:
        assert a["storage"] in ("ssd", "hdd")


# ---------------------------------------------------------------------------
# Redirect dedupe
# ---------------------------------------------------------------------------


def test_redirects_deduped_in_results(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    # redirect_b -> redirect_a -> pythagorean_theorem in the fixture ZIM.
    resp = engine.research("redirect b pythagorean theorem")
    seen_paths = [p.path for p in resp.passages]
    assert seen_paths.count("pythagorean_theorem") <= 2  # diversity cap, not per-redirect dup
    assert "redirect_a" not in seen_paths
    assert "redirect_b" not in seen_paths


# ---------------------------------------------------------------------------
# Coverage flags (offline_tutor_spec_v0.3.md §7.3: status "empty" when
# evidence is genuinely absent; coverage signals are explainable).
# ---------------------------------------------------------------------------


def test_compute_coverage_title_match_is_not_weak():
    coverage = compute_coverage(
        query_terms={"pythagorean", "theorem"},
        title="Pythagorean theorem",
        text="Some unrelated filler text about nothing in particular here.",
    )
    assert coverage["title_match"] is True
    assert coverage["weak"] is False


def test_compute_coverage_high_term_overlap_is_not_weak():
    coverage = compute_coverage(
        query_terms={"pythagoras", "triangle", "hypotenuse", "right", "angle"},
        title="Geometry intro",
        text="Pythagoras discovered a rule about the right triangle and its hypotenuse.",
    )
    assert coverage["weak"] is False


def test_compute_coverage_no_overlap_and_no_title_match_is_weak():
    coverage = compute_coverage(
        query_terms={"flibbertigibbetopolis", "quantum"},
        title="Bread baking",
        text="Bread is made from flour, water, yeast, and salt.",
    )
    assert coverage["term_coverage"] == 0.0
    assert coverage["title_match"] is False
    assert coverage["weak"] is True


def test_compute_coverage_no_candidate_is_weak():
    coverage = compute_coverage(query_terms={"anything"}, title=None, text=None)
    assert coverage["weak"] is True


def test_compute_coverage_empty_query_terms_is_weak():
    coverage = compute_coverage(query_terms=set(), title="Anything", text="Anything")
    assert coverage["weak"] is True


def test_compute_coverage_topic_hint_title_match_is_not_weak():
    # An elliptical follow-up ("what made it explode?") whose bare query
    # has little to go on, but whose candidate's title matches the
    # host-supplied topic_hint entity once merged into query_terms (see
    # `_query_terms`), is trustworthy.
    coverage = compute_coverage(
        query_terms={"made", "explode", "volcano"},
        title="Volcano",
        text="A volcano is an opening that lets hot magma escape.",
    )
    assert coverage["weak"] is False
    assert coverage["title_match"] is True


# ---------------------------------------------------------------------------
# Weak-coverage query returns status "empty" with no passages even when
# some low-relevance candidates were fetched (not merely "zero hits").
# ---------------------------------------------------------------------------


def test_weak_coverage_query_returns_empty_and_no_passages(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    # "capital of the moon"-style false-premise/absent query against a
    # fixture corpus with no matching article and no shared vocabulary.
    resp = engine.research("What is the flibbertigibbetopolis effect?")
    assert resp.status == "empty"
    assert resp.passages == []
    assert resp.coverage["weak"] is True


# ---------------------------------------------------------------------------
# topic_hint used as conversational context for elliptical follow-ups
# (spec §5 step 3: research(query=message, topic_hint=current_subject)).
# ---------------------------------------------------------------------------


def test_topic_hint_disambiguates_elliptical_query(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Can you tell me more about it?", topic_hint="Pythagorean theorem")
    assert resp.status in ("ok", "partial")
    assert resp.passages
    assert resp.passages[0].path == "pythagorean_theorem"


# ---------------------------------------------------------------------------
# Tier-2 gating: spec §6 "Default search order: tier 1 -> tier 2 only if
# coverage flags are weak after tier 1."
# ---------------------------------------------------------------------------


@pytest.fixture
def two_storage_tier_registry_toml(tmp_path, fixture_zim):
    return _write_registry_toml(
        tmp_path,
        fixture_zim,
        [
            {"id": "tier1", "tier": 1, "subjects": []},
            {"id": "tier2", "tier": 2, "subjects": []},
        ],
    )


def test_tier2_not_consulted_when_tier1_coverage_strong(
    two_storage_tier_registry_toml, snapshot_store, tmp_path
):
    engine = _engine(two_storage_tier_registry_toml, snapshot_store, tmp_path)
    resp = engine.research("What is the Pythagorean theorem?")
    consulted_ids = {a["id"] for a in resp.archives_consulted}
    assert consulted_ids == {"tier1"}


def test_tier2_consulted_when_tier1_coverage_weak(
    two_storage_tier_registry_toml, snapshot_store, tmp_path
):
    engine = _engine(two_storage_tier_registry_toml, snapshot_store, tmp_path)
    resp = engine.research("What is the flibbertigibbetopolis effect?")
    consulted_ids = {a["id"] for a in resp.archives_consulted}
    assert "tier2" in consulted_ids


# ---------------------------------------------------------------------------
# Dense retrieval wiring (WP-B7 sidecar consumed by research()).
# ---------------------------------------------------------------------------


def _make_dense_index(fixture_zim, *, paths, archive_digest=None, dim=2):
    import numpy as np

    from tutor.retrieval.hybrid.dense import DenseIndex
    from tutor.retrieval.index.manifest import DenseManifest
    from tutor.retrieval.zim.archive import fingerprint as fingerprint_archive

    if archive_digest is None:
        archive_digest = fingerprint_archive(fixture_zim).digest
    n = len(paths)
    vectors = np.eye(n, dim, dtype=np.float32) if dim >= n else np.eye(n, dtype=np.float32)[:, :dim]
    manifest = DenseManifest(
        version=1,
        archive_digest=archive_digest,
        extractor_version="zim-bundle-v1",
        embedding_model_name="fake",
        embedding_model_sha256="a" * 64,
        dim=dim,
        count=n,
        normalisation="l2",
        created_at="2026-01-01T00:00:00+00:00",
    )
    ids = tuple(f"fakeid{i}" for i in range(n))
    return DenseIndex(ids=ids, vectors=vectors, manifest=manifest, paths=tuple(paths))


def test_dense_candidates_fused_and_sourced(registry_toml, snapshot_store, tmp_path):
    import tomllib

    zim_path = Path(tomllib.loads(registry_toml.read_text())["archive"][0]["path"])
    index = _make_dense_index(zim_path, paths=["pythagorean_theorem", "redirect_a"])

    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        dense_indexes={"tier1": index},
        embed_query=lambda q: [1.0, 0.0],
    )
    resp = engine.research("Pythagorean theorem", budget_tokens=2000)
    assert resp.dense_used is True
    matching = [p for p in resp.passages if p.path == "pythagorean_theorem"]
    assert matching
    assert "dense" in matching[0].sources


def test_dense_hits_do_not_defeat_abstention(registry_toml, snapshot_store, tmp_path):
    import tomllib

    zim_path = Path(tomllib.loads(registry_toml.read_text())["archive"][0]["path"])
    index = _make_dense_index(zim_path, paths=["pythagorean_theorem", "redirect_a"])

    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        dense_indexes={"tier1": index},
        embed_query=lambda q: [1.0, 0.0],
    )
    resp = engine.research("zzqxw fnorble asdkjqwe blorptastic")
    assert resp.status == "empty"
    assert resp.passages == []
    assert resp.coverage["weak"] is True


def test_stale_dense_sidecar_ignored_with_note(registry_toml, snapshot_store, tmp_path):
    index = _make_dense_index(
        Path("does-not-matter"),
        paths=["pythagorean_theorem"],
        archive_digest="deliberately-wrong-digest",
    )
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        dense_indexes={"tier1": index},
        embed_query=lambda q: [1.0, 0.0],
    )
    resp = engine.research("Pythagorean theorem")
    assert resp.dense_used is False
    assert resp.dense_note is not None
    assert "stale" in resp.dense_note


def test_embed_query_failure_degrades_to_lexical_only(registry_toml, snapshot_store, tmp_path):
    import tomllib

    zim_path = Path(tomllib.loads(registry_toml.read_text())["archive"][0]["path"])
    index = _make_dense_index(zim_path, paths=["pythagorean_theorem"])

    def _broken_embed(q):
        raise RuntimeError("embedder is down")

    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        dense_indexes={"tier1": index},
        embed_query=_broken_embed,
    )
    resp = engine.research("Pythagorean theorem")
    assert resp.dense_used is False
    assert resp.dense_note is not None
    assert resp.status in ("ok", "partial")
    assert resp.passages


def test_passage_sources_default_to_lexical_without_dense(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem")
    assert resp.passages
    for p in resp.passages:
        assert p.sources == ("lexical",)


def test_response_to_dict_includes_dense_fields(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Pythagorean theorem")
    d = resp.to_dict()
    assert "dense_used" in d
    assert "dense_note" in d
    assert d["dense_used"] is False


# ---------------------------------------------------------------------------
# WP-B8 ranking flags applied in the pipeline, default-off identity.
# ---------------------------------------------------------------------------


def test_ranking_flags_none_vs_default_object_identical_response(
    registry_toml, snapshot_store, tmp_path
):
    from tutor.retrieval.research import RankingFlags as RF

    engine_none = _engine(registry_toml, snapshot_store, tmp_path, ranking_flags=None)
    resp_none = engine_none.research("Pythagorean theorem")

    engine_default = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        cache_dir=tmp_path / "cache2",
        ranking_flags=RF(),
    )
    resp_default = engine_default.research("Pythagorean theorem")

    d1 = resp_none.to_dict()
    d2 = resp_default.to_dict()
    d1.pop("timings", None)
    d2.pop("timings", None)
    assert d1 == d2


# ---------------------------------------------------------------------------
# Coverage floor: a title match on one generic own term must not be enough
# when the question has >= 3 own content terms and almost none of them show
# up in the candidate's text (the real "Output the boiling point of helium
# in celsius and fahrenheit" failure -- title-matching "Output" while
# "helium"/"boiling"/"celsius"/"fahrenheit" are all absent).
# ---------------------------------------------------------------------------


def test_compute_coverage_generic_title_match_below_floor_is_weak():
    coverage = compute_coverage(
        query_terms={"boiling", "point", "helium", "celsius", "fahrenheit"},
        title="Output",
        text="Output is data sent from a computer, such as to a printer or screen.",
        own_term_count=5,
    )
    assert coverage["title_match"] is False
    assert coverage["weak"] is True


def test_compute_coverage_title_match_above_floor_is_not_weak():
    coverage = compute_coverage(
        query_terms={"boiling", "point", "helium", "celsius", "fahrenheit"},
        title="Helium",
        text="Helium boils at a very low temperature, about -269 celsius or -452 fahrenheit.",
        own_term_count=5,
    )
    assert coverage["title_match"] is True
    assert coverage["weak"] is False


def test_compute_coverage_floor_not_applied_below_min_own_term_count():
    # Only 2 own content terms -- the floor gate (>= 3) never kicks in, so
    # the existing single-shared-term title match behavior is unchanged.
    coverage = compute_coverage(
        query_terms={"pythagorean", "theorem"},
        title="Pythagorean theorem",
        text="Some unrelated filler text about nothing in particular here.",
        own_term_count=2,
    )
    assert coverage["title_match"] is True
    assert coverage["weak"] is False


# ---------------------------------------------------------------------------
# Candidate-generation fallback (item 3): when the all-terms query returns
# nothing, candidate articles are ranked by distinct-term coordination and
# term rarity, not first-seen order -- a rare term (Erdős, appearing in only
# 2 of the fixture's ~55 articles) must win over a common one ("school",
# appearing in ~50 of them), and a misspelt/OOV term must be dropped
# entirely rather than diluting the merge.
# ---------------------------------------------------------------------------


def test_fallback_prefers_rare_term_article_over_common_term_articles(
    registry_toml, snapshot_store, tmp_path
):
    # "hypotenuse" only appears in the fixture's Pythagorean theorem
    # article; "school" appears in ~50 of the ~55 fixture articles (every
    # generated simple-topic page says "X is a school topic"). No article
    # has both, so the joined all-terms query returns nothing and the
    # fallback must win on "hypotenuse"'s rarity, not "school"'s ubiquity.
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    entry = registry.for_subject(None)[0]
    candidates, _timed_out, _note, _key_facts = engine._process_archive(
        entry, "hypotenuse school", lambda: 5.0
    )
    assert candidates, "expected fallback candidates"
    candidates.sort(key=lambda c: -c["score"])
    assert candidates[0]["used_fallback"] is True
    assert candidates[0]["title"] == "Pythagorean Theorem"


# ---------------------------------------------------------------------------
# Baseline v4: entity candidates via real corpus-IDF (getEstimatedMatches).
#
# Reproduces the real-archive failure reported against simplewiki: an
# all-terms AND query returns hits (so the zero-hits-only fallback above
# never runs), but every hit is a generic wrong article ("Boiling point",
# "Boiling", ...) because the true entity article ("Helium") lacks one
# word the question used ("celsius"/"farenheit"). A FakeWorker below stands
# in for the real ZIM worker so the scenario is exact and deterministic
# (the real archive is exercised separately, out of process, per the task
# brief -- not from the unit suite).
# ---------------------------------------------------------------------------


class _EntityFakeWorker:
    """Fake worker reproducing the "AND succeeds but wrong" bug: the
    all-terms fulltext query returns several generic "Boiling_*" articles
    (none of them the true entity), while a direct title search on either
    of the two rarest terms ("celsius", "helium") finds the right article
    outright, and `estimated_matches` gives each term a distinct,
    realistic corpus-wide rarity ranking.
    """

    _MATCHES = {"boiling": 1042, "point": 17517, "helium": 332, "celsius": 222}
    _GENERIC_HITS = [
        ("Boiling_point", "Boiling point"),
        ("Boiling", "Boiling"),
        ("Boiling_tube", "Boiling tube"),
    ]
    _TITLE_HITS = {
        "celsius": [("Celsius", "Celsius")],
        "helium": [("Helium", "Helium")],
    }
    _HTML_BY_PATH = {
        "Boiling_point": "<html><head><title>Boiling point</title></head>"
        "<body><h1>Boiling point</h1><p>Boiling point is a physical "
        "property.</p></body></html>",
        "Boiling": "<html><head><title>Boiling</title></head>"
        "<body><h1>Boiling</h1><p>Boiling is a process.</p></body></html>",
        "Boiling_tube": "<html><head><title>Boiling tube</title></head>"
        "<body><h1>Boiling tube</h1><p>A boiling tube is lab glassware.</p>"
        "</body></html>",
        "Celsius": "<html><head><title>Celsius</title></head>"
        "<body><h1>Celsius</h1><p>Celsius is a temperature scale.</p>"
        "</body></html>",
        "Helium": "<html><head><title>Helium</title></head>"
        "<body><h1>Helium</h1><p>Helium has the lowest boiling point of "
        "all the elements, about -269 degrees celsius.</p></body></html>",
    }

    def __init__(self, archive_path: Path) -> None:
        self._archive_path = archive_path

    def _lead(self, path: str) -> str:
        # Real Xapian snippets are drawn from the article's own indexed
        # text -- stand in with each fake article's lead paragraph so the
        # scorer's lead-text coordination term has something realistic to
        # work with (see ``_score_articles``).
        html = self._HTML_BY_PATH.get(path, "")
        if "<p>" in html:
            return html.split("<p>", 1)[1].split("</p>", 1)[0]
        return ""

    def _hits(self, pairs, source: str):
        from tutor.retrieval.zim.search import SearchHit

        return [
            SearchHit(path=p, title=t, rank=i, snippet=self._lead(p), source=source)
            for i, (p, t) in enumerate(pairs)
        ]

    def request(self, op: str, *, deadline_s: float, **kwargs) -> WorkerResult:
        if op == "estimated_matches":
            term = kwargs["term"]
            return WorkerResult(
                status="ok", value=self._MATCHES.get(term, 0), error=None, elapsed_s=0.0
            )
        if op == "search_fulltext":
            query = kwargs["query"]
            terms = set(query.split())
            if terms >= {"boiling", "point", "helium", "celsius"}:
                return WorkerResult(
                    status="ok",
                    value=self._hits(self._GENERIC_HITS, "fulltext"),
                    error=None,
                    elapsed_s=0.0,
                )
            # Relaxed-AND queries of just the rarest terms: no article in
            # this fake corpus contains both "celsius" and "helium" as
            # literal words (Helium's own text spells out "celsius" only
            # lowercase within a sentence -- treated here as not a fulltext
            # AND match, matching the real bug: the Helium article uses a
            # degree symbol, not the literal word, in the real archive).
            return WorkerResult(status="ok", value=[], error=None, elapsed_s=0.0)
        if op == "search_titles":
            query = kwargs["query"].strip()
            pairs = self._TITLE_HITS.get(query)
            if pairs:
                return WorkerResult(
                    status="ok", value=self._hits(pairs, "title"), error=None, elapsed_s=0.0
                )
            return WorkerResult(status="ok", value=[], error=None, elapsed_s=0.0)
        if op == "fetch_entry":
            from tutor.retrieval.zim.search import FetchedEntry

            path = kwargs["path"]
            html = self._HTML_BY_PATH.get(path)
            if html is None:
                return WorkerResult(status="error", value=None, error="unknown path", elapsed_s=0.0)
            title = html.split("<title>")[1].split("</title>")[0]
            return WorkerResult(
                status="ok",
                value=FetchedEntry(
                    path=path, title=title, mimetype="text/html", html=html, hops=()
                ),
                error=None,
                elapsed_s=0.0,
            )
        return WorkerResult(status="error", value=None, error=f"unknown op {op}", elapsed_s=0.0)

    def close(self) -> None:
        pass


def _titles_by_rank(candidates: list[dict]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for c in sorted(candidates, key=lambda c: -c["score"]):
        if c["title"] not in seen:
            seen.add(c["title"])
            titles.append(c["title"])
    return titles


# Baseline v5: article SCORING (not slot-forcing) must put Helium first --
# not merely top-2 -- for each of the owner's three phrasings of the same
# question (cases A/B/C). Case A is the exact real-archive failure quoted
# in docs/retrieval_baseline.md ("boiling point of helium" ranked "Boiling"
# above "Helium", and "Celsius" above "Helium").
@pytest.mark.parametrize(
    "query",
    [
        # Case A: instruction-word wrapped, both units named.
        "Output the boiling point of helium in celsius and farenheit.",
        # Case B: plain question form.
        "What is the boiling point of helium in celsius?",
        # Case C: unit-first phrasing.
        "In celsius, what is helium's boiling point?",
    ],
    ids=["case_a", "case_b", "case_c"],
)
def test_helium_article_scoring_ranks_helium_first(
    registry_toml, snapshot_store, tmp_path, query
):
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    engine = _engine(
        registry_toml,
        snapshot_store,
        tmp_path,
        worker_factory=lambda path: _EntityFakeWorker(path),
    )
    entry = registry.for_subject(None)[0]
    candidates, _timed_out, _note, _key_facts = engine._process_archive(entry, query, lambda: 5.0)
    titles = _titles_by_rank(candidates)
    assert titles[0] == "Helium", titles


def test_direct_style_question_keeps_correct_article_first(
    registry_toml, snapshot_store, tmp_path
):
    """Article scoring must not perturb the common case: a "direct" style
    question whose correct article is already unambiguous stays first."""
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("What is the Pythagorean theorem?")
    assert resp.passages
    assert resp.passages[0].path == "pythagorean_theorem"


def test_idf_lookup_is_cached_per_archive_and_term(registry_toml, snapshot_store, tmp_path):
    """The real `getEstimatedMatches` call is made at most once per
    (archive, term) across a whole request -- repeat lookups (e.g. the same
    term reused across the fulltext/title fallback and the entity-candidate
    pass) hit the engine's own bounded cache instead of the worker."""

    class _CountingWorker(_EntityFakeWorker):
        calls: dict[str, int] = {}

        def request(self, op: str, *, deadline_s: float, **kwargs):
            if op == "estimated_matches":
                term = kwargs["term"]
                self.calls[term] = self.calls.get(term, 0) + 1
            return super().request(op, deadline_s=deadline_s, **kwargs)

    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    worker = _CountingWorker(Path("unused"))
    engine = _engine(
        registry_toml, snapshot_store, tmp_path, worker_factory=lambda path: worker
    )
    entry = registry.for_subject(None)[0]
    engine._process_archive(entry, "boiling point helium celsius", lambda: 5.0)
    calls_after_first = dict(worker.calls)
    engine._process_archive(entry, "boiling point helium celsius", lambda: 5.0)
    assert worker.calls == calls_after_first, "second call must hit the IDF cache, not the worker"
