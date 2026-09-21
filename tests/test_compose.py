"""RED tests for tutor.app.compose.build_deps -- wiring the real llama-server
client, ResearchEngine, SnapshotStore, session store, calc, Budget, and the
turn_runner/status_provider adapters into an AppDeps, all exercisable
without a live llama-server or a real ZIM archive.
"""

from __future__ import annotations

import dataclasses
import hashlib
import socket
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from tutor.app.compose import build_deps
from tutor.app.llm_client import StreamEvent
from tutor.app.main import create_app
from tutor.retrieval.index.simplewiki_build import build_index
from tutor.settings import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TOML = REPO_ROOT / "config" / "dev.toml"


def _closed_port() -> int:
    """A local TCP port nothing is listening on (bind, then close)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _write_registry(tmp_path: Path) -> Path:
    registry_path = tmp_path / "archives.toml"
    registry_path.write_text(
        """
[[archive]]
id = "demo"
path = "missing.zim"
tier = 1
kind = "encyclopedia"
subjects = ["general"]
storage = "ssd"
""",
        encoding="utf-8",
    )
    return registry_path


def _cfg_with_unreachable_llm_and_missing_archives(tmp_path: Path):
    cfg = load_config(DEV_TOML)
    closed_port = _closed_port()
    server = dataclasses.replace(cfg.server, host="127.0.0.1", port=closed_port)
    app_cfg = dataclasses.replace(
        cfg.app,
        data_dir=(tmp_path / "data").resolve(),
        registry_path=_write_registry(tmp_path).resolve(),
    )
    return dataclasses.replace(cfg, server=server, app=app_cfg)


def test_build_deps_succeeds_with_unreachable_llm_and_missing_archives(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    deps = build_deps(cfg)
    assert deps.turn_runner is not None
    assert deps.status_provider is not None
    assert deps.sessions is not None


def test_status_endpoint_honestly_reports_unhealthy_llm_and_missing_archive(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    deps = build_deps(cfg)
    app = create_app(deps)
    with TestClient(app) as client:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["llm"]["healthy"] is False
        assert len(body["archives"]) == 1
        assert body["archives"][0]["id"] == "demo"
        assert body["archives"][0]["state"] == "MISSING"
        assert body["model_path"] == cfg.runtime.model_path.name
        assert isinstance(body["context_ceiling"], int)


def test_turn_against_unreachable_llm_ends_with_student_safe_error_event(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    deps = build_deps(cfg)
    app = create_app(deps)
    with TestClient(app) as client:
        sid = client.post("/api/session").json()["session_id"]
        with client.stream(
            "POST", f"/api/session/{sid}/turn", json={"text": "what is water?"}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
    assert "event: error" in body
    assert "Traceback" not in body
    assert "traceback" not in body.lower()
    assert "Errno" not in body


# ---------------------------------------------------------------------------
# turn_runner event mapping, exercised directly against a fake llm.
# ---------------------------------------------------------------------------


class _FakeLlm:
    def __init__(self, events):
        self._events = events

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def health(self) -> bool:
        return True

    def stream_chat(self, messages, *, max_tokens=None, tools=None, cancel=None, temperature=None):
        yield from self._events


class _FakeResearchEngine:
    def research(self, query, *, topic_hint=None, keywords=None):
        return {"passages": []}


def test_turn_runner_maps_token_tool_citations_done_events(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    events = [
        StreamEvent(kind="token", text="Water "),
        StreamEvent(kind="token", text="boils."),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())

    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(session_id, _Input(kind="text", text="Does water boil?"), emit, cancel)

    names = [name for name, _ in frames]
    assert names.count("token") == 2
    assert "citations" in names
    assert names[-1] == "done"
    done_data = frames[-1][1]
    assert done_data["status"] == "ok"
    assert "Water" in done_data["answer"]
    # 2026-09-20 evidence-dump follow-up: no citations at all -> "uncited",
    # never "unsupported" (there is nothing to fail to support).
    assert done_data["citation_quality"] == "uncited"
    assert done_data["evidence_dump"] is False
    citations_data = dict(frames)["citations"]
    assert citations_data["unsupported_labels"] == []
    # Bounded-generation follow-up (2026-09-20): a normal, un-truncated
    # turn reports truncated: None on the done event.
    assert done_data["truncated"] is None


def test_turn_runner_reports_repetition_truncation_on_done_event(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    sentence = "Water boils at exactly one hundred degrees Celsius. "
    events = [StreamEvent(kind="token", text=sentence) for _ in range(5)]
    events.append(StreamEvent(kind="done", finish_reason="stop", usage={}))
    fake_llm = _FakeLlm(events)
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())

    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(session_id, _Input(kind="text", text="Does water boil?"), emit, cancel)

    done_data = dict(frames)["done"]
    assert done_data["status"] == "ok"
    assert done_data["truncated"] == "repetition"
    assert done_data["answer"].count(sentence.strip()) == 1


def test_turn_runner_emits_attributions_event_before_done(tmp_path):
    """2026-09-20 attribution follow-up: the host attributes sentences to
    passages as a separate SSE event, sent before `done`, never editing the
    model's text."""
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    answer = "The mitochondria is the powerhouse of the cell. [S1]"
    events = [
        StreamEvent(kind="token", text=answer),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)

    class _PassagesResearchEngine:
        def research(self, query, *, topic_hint=None, keywords=None):
            return {
                "passages": [
                    {
                        "label": "S1",
                        "id": "p1",
                        "title": "Cell biology",
                        "path": "Biology/Cell",
                        "text": "The mitochondria is the powerhouse of the cell.",
                        "kind": "article",
                    }
                ]
            }

    deps = build_deps(cfg, llm=fake_llm, research_engine=_PassagesResearchEngine())
    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(
        session_id, _Input(kind="text", text="What is the mitochondria?"), emit, cancel
    )

    names = [name for name, _ in frames]
    assert "attributions" in names
    assert names.index("attributions") < names.index("done")

    attributions_data = dict(frames)["attributions"]
    assert "attributions" in attributions_data
    assert "unbacked" in attributions_data
    assert attributions_data["attributions"][0]["passage_id"] == "p1"
    assert attributions_data["attributions"][0]["label"] == "S1"


def test_turn_runner_survives_attribution_exception(tmp_path, monkeypatch):
    """A broken attribution layer must never fail the turn -- it just emits
    no `attributions` event."""
    import tutor.app.compose as compose_module

    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    events = [
        StreamEvent(kind="token", text="Water boils at 100C."),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(compose_module, "attribute_sentences", _boom)

    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(session_id, _Input(kind="text", text="Does water boil?"), emit, cancel)

    names = [name for name, _ in frames]
    assert "attributions" not in names
    assert names[-1] == "done"
    assert dict(frames)["done"]["status"] == "ok"


def test_turn_runner_flags_unsupported_citations_and_evidence_dump(tmp_path):
    """The owner's real failure: 11 off-topic passages, all labels resolve
    but none support the answer's claim, and the answer dumps the evidence
    back out label-stacked on one trailer sentence."""
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    topics = [f"Output device number {i} converts signals for a computer." for i in range(1, 12)]
    bullets = [f"[S{i}]: {topics[i - 1]}" for i in range(1, 12)]
    stacked = "".join(f"[S{i}]" for i in range(1, 12))
    answer = (
        "The boiling point of helium is -268.92C.\n\n"
        "Here's the reasoning:\n" + "\n".join(bullets) + "\n" + stacked
    )
    events = [
        StreamEvent(kind="token", text=answer),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)

    class _PassagesResearchEngine:
        def research(self, query, *, topic_hint=None, keywords=None):
            passages = [
                {
                    "label": f"S{i}",
                    "id": f"p{i}",
                    "title": f"Output topic {i}",
                    "path": f"Computing/Output{i}",
                    "text": topics[i - 1],
                    "kind": "article",
                }
                for i in range(1, 12)
            ]
            return {"passages": passages}

    deps = build_deps(cfg, llm=fake_llm, research_engine=_PassagesResearchEngine())
    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(
        session_id,
        _Input(kind="text", text="Output the boiling point of helium."),
        emit,
        cancel,
    )

    citations_data = dict(frames)["citations"]
    assert len(citations_data["unsupported_labels"]) == 11
    done_data = dict(frames)["done"]
    assert done_data["citation_quality"] == "unsupported"
    assert done_data["evidence_dump"] is True


def test_turn_runner_reports_error_status_without_crashing(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    fake_llm = _FakeLlm([StreamEvent(kind="error", error="connection refused")])
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())

    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    deps.turn_runner(session_id, _Input(kind="text", text="hi"), emit, threading.Event())

    assert frames
    assert frames[-1][0] == "error"
    assert "connection refused" not in frames[-1][1]["message"]


@dataclasses.dataclass
class _Input:
    kind: str
    text: str | None = None
    action: str | None = None


# ---------------------------------------------------------------------------
# 2026-09-20 computed-statement follow-up (docs/calc_investigation.md fix
# #1): the host verifies stated arithmetic against tutor.tools.calc_tool's
# sandboxed evaluator and reports it as a "computed" list on the same
# "attributions" SSE event, without ever editing the model's own text.
# ---------------------------------------------------------------------------


def test_turn_runner_includes_computed_mismatch_in_attributions_event(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    answer = "12.5% of 640 is 108.8."
    events = [
        StreamEvent(kind="token", text=answer),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())
    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(
        session_id, _Input(kind="text", text="What is 12.5% of 640?"), emit, cancel
    )

    attributions_data = dict(frames)["attributions"]
    assert "computed" in attributions_data
    mismatches = [c for c in attributions_data["computed"] if c["status"] == "mismatch"]
    assert mismatches
    assert mismatches[0]["computed"] == 80.0
    assert mismatches[0]["stated"] == 108.8
    # The host never touches the model's own text.
    assert dict(frames)["done"]["answer"] == answer


def test_turn_runner_survives_computed_check_exception(tmp_path, monkeypatch):
    import tutor.app.compose as compose_module

    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    events = [
        StreamEvent(kind="token", text="12.5% of 640 is 108.8."),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(compose_module, "check_computed_statements", _boom)

    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(session_id, _Input(kind="text", text="What is 12.5% of 640?"), emit, cancel)

    names = [name for name, _ in frames]
    assert names[-1] == "done"
    assert dict(frames)["done"]["status"] == "ok"
    # attributions may still fire (attribution layer is independent), but
    # never with a "computed" list when the check itself blew up.
    if "attributions" in names:
        assert dict(frames)["attributions"]["computed"] == []


def test_verified_computed_number_suppresses_unbacked_number_flag(tmp_path):
    """docs/attribution_design.md: a figure that is the stated/computed
    value of a VERIFIED computed item must not also be flagged
    'unbacked_number' just because no evidence passage contains it."""
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    answer = "2 x 3 x 4 = 24, which is a fun fact about multiplication."
    events = [
        StreamEvent(kind="token", text=answer),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    fake_llm = _FakeLlm(events)
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())
    session_id = deps.sessions.create()
    frames = []

    def emit(event_name, data):
        frames.append((event_name, data))

    cancel = threading.Event()
    deps.turn_runner(session_id, _Input(kind="text", text="What is 2x3x4?"), emit, cancel)

    attributions_data = dict(frames)["attributions"]
    assert any(c["status"] == "verified" for c in attributions_data["computed"])
    unbacked_number_reasons = [
        u for u in attributions_data["unbacked"] if u["reason"] == "unbacked_number"
    ]
    assert unbacked_number_reasons == []


# ---------------------------------------------------------------------------
# Dense sidecar wiring (WP-B8 follow-up): build_deps opens a DenseIndex and
# passes dense_indexes/embed_query into ResearchEngine only when the
# sidecar is present, complete, and fingerprint-fresh; otherwise it starts
# normally, lexical only, and /api/status reports why.
# ---------------------------------------------------------------------------

_DENSE_DIM = 8


def _fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:_DENSE_DIM]])
    return out


def _write_registry_for(tmp_path: Path, zim_path: Path, archive_id: str = "fixture") -> Path:
    registry_path = tmp_path / "archives.toml"
    registry_path.write_text(
        f"""
[[archive]]
id = "{archive_id}"
path = "{zim_path.as_posix()}"
tier = 1
kind = "encyclopedia"
subjects = ["general"]
storage = "ssd"
""",
        encoding="utf-8",
    )
    return registry_path


def _cfg_with_dense(
    tmp_path: Path, fixture_zim: Path, *, sidecar_dir: Path | None, archive_id="fixture"
):
    cfg = load_config(DEV_TOML)
    closed_port = _closed_port()
    server = dataclasses.replace(cfg.server, host="127.0.0.1", port=closed_port)
    app_cfg = dataclasses.replace(
        cfg.app,
        data_dir=(tmp_path / "data").resolve(),
        registry_path=_write_registry_for(tmp_path, fixture_zim, archive_id).resolve(),
    )
    embedding = dataclasses.replace(
        cfg.embedding,
        sidecar_dir=sidecar_dir if sidecar_dir is not None else (tmp_path / "no_such_sidecar"),
        archive_id=archive_id,
        # dev.toml ships with dense disabled by default (M4 gate FAILED on
        # held-out; see docs/M4_report.md). These tests exercise the sidecar
        # wiring itself, so opt in explicitly.
        enabled=True,
    )
    return dataclasses.replace(cfg, server=server, app=app_cfg, embedding=embedding)


def _build_sidecar(fixture_zim: Path, out_dir: Path, *, archive_id: str = "fixture") -> None:
    build_index(
        fixture_zim,
        out_dir,
        _fake_embed,
        archive_id=archive_id,
        embedding_model_name="fake-embedder-v1",
        embedding_model_sha256="0" * 64,
        dim=_DENSE_DIM,
        batch_size=3,
        checkpoint_every=1,
    )


def test_build_deps_wires_dense_index_when_sidecar_is_fresh(tmp_path, fixture_zim):
    sidecar_dir = tmp_path / "sidecar"
    _build_sidecar(fixture_zim, sidecar_dir)
    cfg = _cfg_with_dense(tmp_path, fixture_zim, sidecar_dir=sidecar_dir)

    deps = build_deps(cfg)
    status = deps.status_provider()

    assert status["dense"]["available"] is True
    assert status["dense"]["reason"] is None
    assert status["dense"]["rows"] > 0
    assert status["dense"]["model"] == "fake-embedder-v1"


def test_build_deps_stays_lexical_only_when_sidecar_missing(tmp_path, fixture_zim):
    cfg = _cfg_with_dense(tmp_path, fixture_zim, sidecar_dir=None)

    deps = build_deps(cfg)
    status = deps.status_provider()

    assert status["dense"]["available"] is False
    assert "sidecar" in status["dense"]["reason"] or "manifest" in status["dense"]["reason"]
    assert status["dense"]["rows"] is None


def test_build_deps_stays_lexical_only_when_sidecar_is_stale(tmp_path, fixture_zim, noindex_zim):
    sidecar_dir = tmp_path / "sidecar"
    # Build the sidecar against a DIFFERENT (but validly opened) archive, so
    # its manifest's archive_digest cannot match the fixture_zim registered
    # under this archive id.
    _build_sidecar(noindex_zim, sidecar_dir)
    cfg = _cfg_with_dense(tmp_path, fixture_zim, sidecar_dir=sidecar_dir)

    deps = build_deps(cfg)
    status = deps.status_provider()

    assert status["dense"]["available"] is False
    assert "stale" in status["dense"]["reason"]


def test_build_deps_stays_lexical_only_when_sidecar_has_no_paths_file(tmp_path, fixture_zim):
    sidecar_dir = tmp_path / "sidecar"
    _build_sidecar(fixture_zim, sidecar_dir)
    (sidecar_dir / "paths.txt").unlink()
    cfg = _cfg_with_dense(tmp_path, fixture_zim, sidecar_dir=sidecar_dir)

    deps = build_deps(cfg)
    status = deps.status_provider()

    assert status["dense"]["available"] is False


def test_build_deps_stays_lexical_only_when_dense_disabled_in_config(tmp_path, fixture_zim):
    # Fresh, valid sidecar -- but [embedding].enabled is false (the M4 gate
    # FAILED on held-out; docs/M4_report.md). Dense must stay off.
    sidecar_dir = tmp_path / "sidecar"
    _build_sidecar(fixture_zim, sidecar_dir)
    cfg = _cfg_with_dense(tmp_path, fixture_zim, sidecar_dir=sidecar_dir)
    cfg = dataclasses.replace(cfg, embedding=dataclasses.replace(cfg.embedding, enabled=False))

    deps = build_deps(cfg)
    status = deps.status_provider()

    assert status["dense"]["available"] is False
    assert status["dense"]["reason"] == "disabled in config"
    assert status["dense"]["rows"] is None


def test_build_deps_reports_dense_unavailable_when_not_configured(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    cfg = dataclasses.replace(cfg, embedding=None)

    deps = build_deps(cfg)
    status = deps.status_provider()

    assert status["dense"] == {
        "available": False,
        "reason": "not configured",
        "rows": None,
        "model": None,
    }


def test_status_endpoint_includes_dense_object(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    cfg = dataclasses.replace(cfg, embedding=None)
    deps = build_deps(cfg)
    app = create_app(deps)
    with TestClient(app) as client:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert "dense" in resp.json()


# ---------------------------------------------------------------------------
# GAP 1 (real M5 soak, docs/M5_report.md "known gaps"): the app itself
# persists lesson turns, so an app restart can resume a lesson.
# ---------------------------------------------------------------------------


def test_full_app_persists_turns_and_resumes_byte_identical(tmp_path):
    from tutor.app.prompt import serialize_messages

    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    events = [
        StreamEvent(kind="token", text="Answer text."),
        StreamEvent(kind="done", finish_reason="stop", usage={"cached_tokens": 3}),
    ]
    fake_llm = _FakeLlm(events)

    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())
    app = create_app(deps)
    with TestClient(app) as client:
        created = client.post(
            "/api/lesson", json={"profile_id": "p1", "subject": "math"}
        ).json()
        lesson_id = created["lesson_id"]
        session_id = created["session_id"]

        for text in ["What is 2+2?", "And 3+3?"]:
            with client.stream(
                "POST", f"/api/session/{session_id}/turn", json={"text": text}
            ) as resp:
                assert resp.status_code == 200
                body = "".join(resp.iter_text())
            assert "event: done" in body

    pre_restart_session = deps.sessions.get(session_id)
    pre_restart_bytes = serialize_messages(pre_restart_session.log.render())
    pre_restart_session.log.validate()

    # A fresh AppDeps/app built the same (production) way, over the same
    # data dir -- simulating a process restart.
    deps2 = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())
    app2 = create_app(deps2)
    with TestClient(app2) as client2:
        resumed = client2.post(f"/api/lesson/{lesson_id}/resume").json()
        new_session_id = resumed["session_id"]

        resumed_session = deps2.sessions.get(new_session_id)
        resumed_session.log.validate()
        assert serialize_messages(resumed_session.log.render()) == pre_restart_bytes

        with client2.stream(
            "POST", f"/api/session/{new_session_id}/turn", json={"text": "One more thing?"}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "event: done" in body

    turns = deps2.lessons.list_turns(lesson_id)
    assert len(turns) == 3
    assert all(t.route for t in turns)


def test_no_lesson_attached_session_still_works_and_is_never_persisted(tmp_path):
    cfg = _cfg_with_unreachable_llm_and_missing_archives(tmp_path)
    events = [
        StreamEvent(kind="token", text="Hi."),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    deps = build_deps(cfg, llm=_FakeLlm(events), research_engine=_FakeResearchEngine())
    app = create_app(deps)
    with TestClient(app) as client:
        session_id = client.post("/api/session").json()["session_id"]
        with client.stream(
            "POST", f"/api/session/{session_id}/turn", json={"text": "hi"}
        ) as resp:
            body = "".join(resp.iter_text())
        assert "event: done" in body
    assert deps.sessions.lesson_for(session_id) is None


# ---------------------------------------------------------------------------
# GAP 2: eviction_reprefill is forwarded as an SSE "eviction" event and
# surfaced in /api/status's last_eviction.
# ---------------------------------------------------------------------------


def test_eviction_event_forwarded_over_sse_and_status(tmp_path):
    from tutor.app.compose import _CalcTool, _make_status_provider, _make_turn_runner, _SessionStore
    from tutor.app.prompt import Budget
    from tutor.retrieval.registry import Registry

    def count_tokens(text: str) -> int:
        return len(text.split())

    sessions = _SessionStore(count_tokens)
    session_id = sessions.create()
    session = sessions.get(session_id)
    session.log.append_system("sys")
    for i in range(3):
        session.log.append_user(f"question {i} " * 5)
        session.log.append_assistant(f"answer {i} " * 5, cited_labels=[])

    # A tiny operating ceiling (system + history) forces eviction on the
    # very next turn against this already-long pre-seeded history.
    budget = Budget(ceiling=1000, system=1, history=2, newest=1000, generation=100, margin=1000)

    events = [
        StreamEvent(kind="token", text="ok"),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]
    last_eviction: dict = {}
    turn_runner = _make_turn_runner(
        sessions=sessions,
        llm=_FakeLlm(events),
        research_engine=_FakeResearchEngine(),
        calc=_CalcTool(),
        budget=budget,
        last_eviction=last_eviction,
    )

    frames = []

    def emit(name, data):
        frames.append((name, data))

    turn_runner(session_id, _Input(kind="text", text="one more?"), emit, threading.Event())

    names = [name for name, _ in frames]
    assert "eviction" in names
    eviction_data = dict(frames[names.index("eviction")][1])
    assert eviction_data["evicted_turns"] >= 1
    assert last_eviction[session_id]["evicted_turns"] >= 1
    # Forwarded before the model's answer, per spec.
    assert names.index("eviction") < names.index("done")

    status_provider = _make_status_provider(
        llm=_FakeLlm(events),
        registry=Registry(archives=()),
        sessions=sessions,
        budget=budget,
        model_name="m",
        last_eviction=last_eviction,
    )
    status = status_provider(session_id=session_id)
    assert status["last_eviction"]["evicted_turns"] >= 1

    # A session with no eviction yet reports None.
    other_session_id = sessions.create()
    other_status = status_provider(session_id=other_session_id)
    assert other_status["last_eviction"] is None


def test_no_file_in_tutor_app_imports_retrieval_zim_directly():
    app_dir = REPO_ROOT / "tutor" / "app"
    offenders = []
    for path in app_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "retrieval.zim" in text:
            offenders.append(str(path))
    assert offenders == []
