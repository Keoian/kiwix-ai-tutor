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
