"""RED tests for tutor.app.compose.build_deps -- wiring the real llama-server
client, ResearchEngine, SnapshotStore, session store, calc, Budget, and the
turn_runner/status_provider adapters into an AppDeps, all exercisable
without a live llama-server or a real ZIM archive.
"""

from __future__ import annotations

import dataclasses
import socket
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from tutor.app.compose import build_deps
from tutor.app.llm_client import StreamEvent
from tutor.app.main import create_app
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

    def stream_chat(self, messages, *, tools=None, cancel=None):
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


def test_no_file_in_tutor_app_imports_retrieval_zim_directly():
    app_dir = REPO_ROOT / "tutor" / "app"
    offenders = []
    for path in app_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "retrieval.zim" in text:
            offenders.append(str(path))
    assert offenders == []
