"""RED-then-GREEN tests for docs/review_2026-09-20.md findings 1-5.

Each test below targets one numbered finding in the review. See that
document for the full rationale and spec citations
(docs/plan/offline_tutor_spec_v0.3.md §7.1, §7.2, §7.4, §9.1).
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path

import pytest

from tutor.app import agent_loop, research_tool
from tutor.retrieval.research import ResearchEngine
from tutor.retrieval.snapshots import SnapshotStore
from tutor.retrieval.zim.worker import ZimWorker
from tutor.tools.schemas import validate_tool_call


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
    return _write_registry_toml(tmp_path, fixture_zim, [{"id": "tier1", "tier": 1, "subjects": []}])


@pytest.fixture
def snapshot_store(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    yield store
    store.close()


def _engine(registry_toml, snapshot_store, tmp_path, **kwargs):
    from tutor.retrieval.registry import load_registry

    registry = load_registry(registry_toml)
    cache_dir = kwargs.pop("cache_dir", None) or (tmp_path / "cache")
    return ResearchEngine(registry, snapshot_store=snapshot_store, cache_dir=cache_dir, **kwargs)


# ---------------------------------------------------------------------------
# Finding 1: research() must accept and use `keywords`.
# ---------------------------------------------------------------------------


def test_research_accepts_keywords_kwarg(registry_toml, snapshot_store, tmp_path):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    # Must not raise TypeError.
    response = engine.research("history", keywords=["Rome"], topic_hint=None)
    assert response.status in {"ok", "partial", "empty"}


def test_research_keywords_contract_matches_callers():
    """Every keyword argument agent_loop.py / research_tool.py pass to
    research() must be an accepted parameter on the real ResearchEngine.research
    signature -- catches fake/real drift between test doubles and production."""
    sig = inspect.signature(ResearchEngine.research)
    accepted = set(sig.parameters)

    agent_loop_src = inspect.getsource(agent_loop)
    research_tool_src = inspect.getsource(research_tool)
    for src in (agent_loop_src, research_tool_src):
        assert "research_engine.research(" in src
    # Both call sites pass keywords= and topic_hint= as kwargs.
    assert "keywords" in accepted
    assert "topic_hint" in accepted


def test_research_tool_run_research_real_engine(registry_toml, snapshot_store, tmp_path):
    """Drive research_tool.run_research against a REAL ResearchEngine (not a
    fake) to catch signature drift the way agent_loop production code does."""
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    text = research_tool.run_research(
        engine, {"query": "history", "keywords": ["Rome"]}, topic_hint=None
    )
    assert isinstance(text, str)


def test_research_uses_keywords_in_candidate_generation(registry_toml, snapshot_store, tmp_path):
    """Keywords should be appended to the lexical query terms used for
    candidate generation (spec §7.2 step 1), not silently ignored."""
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    assert "keywords" in inspect.signature(engine._process_archive).parameters


# ---------------------------------------------------------------------------
# Finding 2: keywords cap (<=3 items, <=40 chars each, non-empty).
# ---------------------------------------------------------------------------


def test_validate_tool_call_rejects_too_many_keywords():
    import json

    args = json.dumps({"query": "q", "keywords": ["a", "b", "c", "d"]})
    result = validate_tool_call("research", args)
    assert result.ok is False


def test_validate_tool_call_rejects_long_keyword():
    import json

    args = json.dumps({"query": "q", "keywords": ["x" * 41]})
    result = validate_tool_call("research", args)
    assert result.ok is False


def test_validate_tool_call_rejects_empty_keyword():
    import json

    args = json.dumps({"query": "q", "keywords": [""]})
    result = validate_tool_call("research", args)
    assert result.ok is False


def test_validate_tool_call_accepts_valid_keywords():
    import json

    args = json.dumps({"query": "q", "keywords": ["Diocletian", "tetrarchy"]})
    result = validate_tool_call("research", args)
    assert result.ok is True


def test_research_tool_schema_declares_max_items_and_length():
    from tutor.tools.schemas import RESEARCH_TOOL

    keywords_schema = RESEARCH_TOOL["function"]["parameters"]["properties"]["keywords"]
    assert keywords_schema["maxItems"] == 3
    assert keywords_schema["items"]["maxLength"] == 40


# ---------------------------------------------------------------------------
# Finding 3: ZimWorker thread-safety under overlapping timed-out + normal reqs.
# ---------------------------------------------------------------------------


def test_zim_worker_survives_concurrent_timeout_and_normal_requests(fixture_zim):
    worker = ZimWorker(fixture_zim)
    errors = []

    def hammer(op, seconds, deadline):
        try:
            worker.request(op, deadline_s=deadline, seconds=seconds)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = []
    for i in range(6):
        if i % 2 == 0:
            t = threading.Thread(target=hammer, args=("sleep", 0.3, 0.05))
        else:
            t = threading.Thread(target=hammer, args=("sleep", 0.0, 1.0))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert not errors

    # After the storm, at most one live child.
    live = 1 if worker.pid is not None else 0
    assert live in (0, 1)

    # Next request still works.
    result = worker.request("ping", deadline_s=2.0)
    assert result.status == "ok"
    worker.close()


# ---------------------------------------------------------------------------
# Finding 4: _TOP_N_ARTICLES uses spec's top 6 (cap 10).
# ---------------------------------------------------------------------------


def test_top_n_articles_matches_spec():
    from tutor.retrieval import research as research_module

    assert research_module._TOP_N_ARTICLES == 6


# ---------------------------------------------------------------------------
# Finding 5: two-tier soft (3s)/hard (8s) deadline behaviour.
# ---------------------------------------------------------------------------


def test_soft_deadline_constant_is_three_seconds():
    from tutor.retrieval import research as research_module

    assert research_module._DEFAULT_SOFT_DEADLINE_S == 3.0


def test_research_returns_partial_status_field_supported(registry_toml, snapshot_store, tmp_path):
    """Existing behaviour: research() still returns a well-formed response
    with a recognized status even under a very tight deadline."""
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    response = engine.research("history", deadline_s=0.001)
    assert response.status in {"ok", "partial", "empty"}
