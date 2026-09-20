"""RED tests for the tutor app's HTTP surface: tutor.app.main.create_app and
tutor.app.routes.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §12 (UI: chat
pane, SSE, [S#] chips, action buttons simpler/deeper/hint/research_this,
stop button, subject selector, source viewer, status panel; loopback-only
serving) and WP-C3 in docs/plan/offline_tutor_implementation_plan.md
(static HTML/JS UI, SSE, stop, subject selector, action buttons, source
viewer, status panel).

Contract decisions made here (not spelled out verbatim by the spec/plan,
adopted for this RED step so a GREEN implementer has something concrete to
satisfy):

- ``tutor.app.main.create_app(deps: AppDeps) -> FastAPI`` where
  ``tutor.app.main.AppDeps`` is a dataclass with fields: ``turn_runner``,
  ``snapshot_store``, ``status_provider``, ``sessions``, ``subjects``.
- ``turn_runner`` is a callable
  ``turn_runner(session_id, user_input, emit, cancel) -> TurnResult-like``
  where ``user_input`` is a simple object with ``kind``/``text``/``action``
  attributes (mirroring tests/test_agent_loop.py's local ``_UserInput``),
  ``emit`` is called with ``(event_name, data_dict)`` for each SSE frame the
  route should forward, and ``cancel`` is a ``threading.Event`` the route
  wires to ``POST .../stop``. The fake runner used below busy-waits briefly
  on ``cancel`` so the stop endpoint can be exercised deterministically.
- ``sessions`` is a minimal in-memory store with ``create() -> str``,
  ``get(session_id) -> object | None`` and ``set_subject(session_id, subject)
  -> bool`` (False/raises for an unknown id -- the route maps that to 404).
- ``status_provider`` is a zero-arg-or-``session_id``-kwarg callable
  returning a plain dict (already JSON-shaped) for ``GET /api/status``.
- All fakes here are in-process, no network, no model, no real archive.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tutor.app.main import AppDeps, build_uvicorn_config, create_app
from tutor.retrieval.snapshots import SnapshotStore

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeUserInput:
    kind: str
    text: str | None = None
    action: str | None = None


class _FakeSessions:
    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._counter = 0

    def create(self) -> str:
        self._counter += 1
        sid = f"sess-{self._counter}"
        self._sessions[sid] = {"subject": None}
        return sid

    def get(self, session_id: str):
        return self._sessions.get(session_id)

    def set_subject(self, session_id: str, subject: str) -> bool:
        if session_id not in self._sessions:
            return False
        self._sessions[session_id]["subject"] = subject
        return True


@dataclass
class _ScriptedRunner:
    """Fake turn_runner: emits a scripted sequence of (event, data) frames.

    Records the cancel Event it was given so tests can assert stop() set it,
    and tracks concurrently-running session ids to emulate the "second
    concurrent turn -> 409" rule at the route layer via a lock map the route
    itself is expected to hold (see test_turn_conflict_returns_409, which
    drives this through the real route, not by calling the fake directly).
    """

    frames: list[tuple[str, dict]] = field(default_factory=lambda: [("done", {"status": "ok"})])
    wait_for_cancel: bool = False
    seen_cancel: threading.Event | None = None
    hold_seconds: float = 0.0

    def __call__(self, session_id, user_input, emit, cancel):
        self.seen_cancel = cancel
        if self.hold_seconds:
            time.sleep(self.hold_seconds)
        if self.wait_for_cancel:
            deadline = time.monotonic() + 2.0
            while not cancel.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            emit("done", {"status": "cancelled" if cancel.is_set() else "ok"})
            return
        for name, data in self.frames:
            emit(name, data)


def _fake_status_provider(session_id: str | None = None) -> dict:
    return {
        "llm": {"healthy": True},
        "archives": [{"id": "wiki_demo", "storage": "ssd"}],
        "model_path": "demo-model.gguf",
        "context_ceiling": 32000,
        "tokens_used": 0 if session_id else None,
        "headroom": 32000 if session_id else None,
        "last_eviction": None,
    }


def _make_app(tmp_path: Path, runner=None):
    store = SnapshotStore(tmp_path / "snaps.sqlite3")
    deps = AppDeps(
        turn_runner=runner or _ScriptedRunner(),
        snapshot_store=store,
        status_provider=_fake_status_provider,
        sessions=_FakeSessions(),
        subjects=["general", "math", "history"],
    )
    app = create_app(deps)
    return app, deps


@pytest.fixture
def client(tmp_path):
    app, deps = _make_app(tmp_path)
    with TestClient(app) as c:
        yield c, deps


# ---------------------------------------------------------------------------
# Static / index
# ---------------------------------------------------------------------------


def test_get_root_serves_index_html(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_static_app_js_served(client):
    c, _ = client
    resp = c.get("/static/app.js")
    assert resp.status_code == 200


def test_static_app_css_served(client):
    c, _ = client
    resp = c.get("/static/app.css")
    assert resp.status_code == 200


def test_responses_carry_content_security_policy(client):
    c, _ = client
    resp = c.get("/")
    assert "content-security-policy" in {k.lower() for k in resp.headers.keys()}
    csp = resp.headers.get("content-security-policy", "").lower()
    assert "'self'" in csp


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_session_returns_session_id(client):
    c, _ = client
    resp = c.post("/api/session")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["session_id"], str) and body["session_id"]


def test_set_subject_on_known_session(client):
    c, _ = client
    sid = c.post("/api/session").json()["session_id"]
    resp = c.post(f"/api/session/{sid}/subject", json={"subject": "math"})
    assert resp.status_code == 200


def test_set_subject_on_unknown_session_is_404(client):
    c, _ = client
    resp = c.post("/api/session/does-not-exist/subject", json={"subject": "math"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Turn: SSE streaming, validation, conflict
# ---------------------------------------------------------------------------


def _sid(c) -> str:
    return c.post("/api/session").json()["session_id"]


def test_turn_text_returns_event_stream_with_named_events(client):
    c, deps = client
    deps.turn_runner.frames = [
        ("token", {"text": "Hello"}),
        ("citations", {"citations": []}),
        ("done", {"status": "ok"}),
    ]
    sid = _sid(c)
    with c.stream("POST", f"/api/session/{sid}/turn", json={"text": "hi"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body


def test_turn_unknown_action_is_422(client):
    c, _ = client
    sid = _sid(c)
    resp = c.post(f"/api/session/{sid}/turn", json={"action": "not_a_real_action"})
    assert resp.status_code == 422


def test_turn_with_both_text_and_action_is_422(client):
    c, _ = client
    sid = _sid(c)
    resp = c.post(f"/api/session/{sid}/turn", json={"text": "hi", "action": "hint"})
    assert resp.status_code == 422


def test_turn_with_neither_text_nor_action_is_422(client):
    c, _ = client
    sid = _sid(c)
    resp = c.post(f"/api/session/{sid}/turn", json={})
    assert resp.status_code == 422


def test_turn_on_unknown_session_is_404(client):
    c, _ = client
    resp = c.post("/api/session/does-not-exist/turn", json={"text": "hi"})
    assert resp.status_code == 404


def test_turn_valid_action_is_accepted(client):
    c, deps = client
    sid = _sid(c)
    resp = c.post(f"/api/session/{sid}/turn", json={"action": "simpler"})
    assert resp.status_code == 200


def test_second_concurrent_turn_on_same_session_is_409(tmp_path):
    runner = _ScriptedRunner(wait_for_cancel=False, hold_seconds=0.3)
    app, deps = _make_app(tmp_path, runner=runner)
    with TestClient(app) as c:
        sid = c.post("/api/session").json()["session_id"]

        results = {}

        def _first():
            with c.stream("POST", f"/api/session/{sid}/turn", json={"text": "long one"}) as resp:
                results["first_status"] = resp.status_code
                list(resp.iter_text())

        t = threading.Thread(target=_first)
        t.start()
        time.sleep(0.1)
        second = c.post(f"/api/session/{sid}/turn", json={"text": "second"})
        t.join(timeout=5)

        assert results.get("first_status") == 200
        assert second.status_code == 409


def test_stop_sets_cancel_and_stream_ends_cancelled(tmp_path):
    runner = _ScriptedRunner(wait_for_cancel=True)
    app, deps = _make_app(tmp_path, runner=runner)
    with TestClient(app) as c:
        sid = c.post("/api/session").json()["session_id"]

        body_holder = {}

        def _run():
            with c.stream("POST", f"/api/session/{sid}/turn", json={"text": "hi"}) as resp:
                body_holder["body"] = "".join(resp.iter_text())

        t = threading.Thread(target=_run)
        t.start()
        time.sleep(0.1)
        stop_resp = c.post(f"/api/session/{sid}/stop")
        t.join(timeout=5)

        assert stop_resp.status_code == 200
        assert runner.seen_cancel is not None
        assert runner.seen_cancel.is_set()
        assert '"status": "cancelled"' in body_holder["body"].replace(" ", "") or (
            '"status":"cancelled"' in body_holder["body"].replace(" ", "")
        )


def test_error_event_is_forwarded(tmp_path):
    def _erroring_runner(session_id, user_input, emit, cancel):
        emit("error", {"message": "boom"})

    app, deps = _make_app(tmp_path, runner=_erroring_runner)
    with TestClient(app) as c:
        sid = c.post("/api/session").json()["session_id"]
        with c.stream("POST", f"/api/session/{sid}/turn", json={"text": "hi"}) as resp:
            body = "".join(resp.iter_text())
    assert "event: error" in body


# ---------------------------------------------------------------------------
# Source viewer endpoint
# ---------------------------------------------------------------------------


def _fake_snapshot(passage_id="p1"):
    @dataclass
    class _Passage:
        passage_id: str
        archive_id: str = "wiki_demo"
        path: str = "A/Water"
        title: str = "Water"
        heading_path: tuple = ("Water", "Properties")
        start: int = 10
        end: int = 20
        text: str = "boils at 100"

    return _Passage(passage_id=passage_id)


def test_source_endpoint_returns_passage_details(client):
    c, deps = client
    deps.snapshot_store.put(_fake_snapshot("p1"), fingerprint_digest="deadbeef")
    resp = c.get("/api/source/p1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Water"
    assert body["path"] == "A/Water"
    assert body["archive_id"] == "wiki_demo"
    assert body["heading_path"] == ["Water", "Properties"]
    assert body["highlight"]["start"] >= 0
    assert body["highlight"]["end"] > body["highlight"]["start"]
    assert "context_before" in body
    assert "context_after" in body
    assert body["kind"] in ("article", "qa", None) or isinstance(body["kind"], str)


def test_source_endpoint_unknown_id_is_404(client):
    c, _ = client
    resp = c.get("/api/source/does-not-exist")
    assert resp.status_code == 404


def test_source_endpoint_works_after_disposable_caches_cleared(client):
    """Only the snapshot store is required; no other cache dependency."""
    c, deps = client
    deps.snapshot_store.put(_fake_snapshot("p2"), fingerprint_digest="cafef00d")
    # Simulate "disposable caches cleared" by not touching anything except
    # the snapshot store; the endpoint must still resolve the passage.
    resp = c.get("/api/source/p2")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


def test_status_endpoint_reports_health_and_config(client):
    c, _ = client
    resp = c.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm" in body
    assert "archives" in body
    assert "model_path" in body
    assert "context_ceiling" in body


def test_status_endpoint_with_session_id_reports_token_usage(client):
    c, _ = client
    sid = _sid(c)
    resp = c.get(f"/api/status?session_id={sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert "tokens_used" in body
    assert "headroom" in body


# ---------------------------------------------------------------------------
# Loopback-only binding
# ---------------------------------------------------------------------------


def test_build_uvicorn_config_binds_loopback_by_default():
    from types import SimpleNamespace

    class _Server:
        host = "127.0.0.1"
        port = 8420
    cfg = SimpleNamespace(server=_Server())  # only .server.host/.port are read
    uv_cfg = build_uvicorn_config(cfg)
    assert uv_cfg.host == "127.0.0.1" or uv_cfg["host"] == "127.0.0.1"
