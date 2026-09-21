"""RED tests: the student profile's grade level is meant to reach the
rendered system prompt (HANDOFF.md ``Reading level`` follow-up, and
``docs/HANDOFF_2026-09-19_original.md`` sec 2) so the tutor pitches for a
10-16 year-old. ``Profile.prompt_summary()`` (tutor/app/profiles.py)
already renders a bounded sentence containing the grade level, and
``agent_loop.build_messages`` already appends ``session.profile_summary``
onto the system text (tutor/app/agent_loop.py) -- but nothing in
``tutor.app.compose`` ever fetches the profile and sets
``session.profile_summary`` for a real lesson. These tests pin that the
grade level actually appears in the FIRST rendered message sent to the
LLM, for both a brand-new lesson and a resumed one, using a fake LLM
client that records the exact messages it was called with (no live LLM
calls).
"""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path

from fastapi.testclient import TestClient

from tutor.app.compose import build_deps
from tutor.app.llm_client import StreamEvent
from tutor.app.main import create_app
from tutor.settings import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TOML = REPO_ROOT / "config" / "dev.toml"


def _closed_port() -> int:
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


def _cfg(tmp_path: Path):
    cfg = load_config(DEV_TOML)
    closed_port = _closed_port()
    server = dataclasses.replace(cfg.server, host="127.0.0.1", port=closed_port)
    app_cfg = dataclasses.replace(
        cfg.app,
        data_dir=(tmp_path / "data").resolve(),
        registry_path=_write_registry(tmp_path).resolve(),
    )
    return dataclasses.replace(cfg, server=server, app=app_cfg)


class _RecordingFakeLlm:
    """A fake LLM client that records every call's messages verbatim,
    matching the ``stream_chat`` signature ``tutor.app.compose`` calls
    against (see ``tests/test_compose.py``'s ``_FakeLlm``)."""

    def __init__(self, events):
        self._events = events
        self.calls: list[list[dict]] = []

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def health(self) -> bool:
        return True

    def stream_chat(
        self,
        messages,
        *,
        max_tokens=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        cancel=None,
        temperature=None,
    ):
        self.calls.append(list(messages))
        yield from self._events


class _FakeResearchEngine:
    def research(self, query, *, topic_hint=None, keywords=None):
        return {"passages": []}


def _done_events():
    return [
        StreamEvent(kind="token", text="Answer."),
        StreamEvent(kind="done", finish_reason="stop", usage={}),
    ]


def test_new_lesson_system_prompt_includes_profile_grade_level(tmp_path):
    cfg = _cfg(tmp_path)
    fake_llm = _RecordingFakeLlm(_done_events())
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())
    app = create_app(deps)

    with TestClient(app) as client:
        profile_id = client.post(
            "/api/profiles",
            json={
                "display_name": "Ada",
                "grade_level": 6,
                "subjects": ["science"],
                "reading_level": "grade6",
            },
        ).json()["id"]
        created = client.post(
            "/api/lesson", json={"profile_id": profile_id, "subject": "general"}
        ).json()
        session_id = created["session_id"]

        with client.stream(
            "POST", f"/api/session/{session_id}/turn", json={"text": "What is a cell?"}
        ) as resp:
            assert resp.status_code == 200
            "".join(resp.iter_text())

    assert fake_llm.calls, "the LLM was never called"
    system_message = fake_llm.calls[0][0]
    assert system_message["role"] == "system"
    assert "(grade 6)" in system_message["content"], system_message["content"]


def test_resumed_lesson_system_prompt_includes_profile_grade_level(tmp_path):
    cfg = _cfg(tmp_path)
    fake_llm = _RecordingFakeLlm(_done_events())
    deps = build_deps(cfg, llm=fake_llm, research_engine=_FakeResearchEngine())
    app = create_app(deps)

    with TestClient(app) as client:
        profile_id = client.post(
            "/api/profiles",
            json={
                "display_name": "Ada",
                "grade_level": 6,
                "subjects": ["science"],
                "reading_level": "grade6",
            },
        ).json()["id"]
        created = client.post(
            "/api/lesson", json={"profile_id": profile_id, "subject": "general"}
        ).json()
        lesson_id = created["lesson_id"]
        first_session_id = created["session_id"]

        with client.stream(
            "POST",
            f"/api/session/{first_session_id}/turn",
            json={"text": "What is a cell?"},
        ) as resp:
            "".join(resp.iter_text())

        resumed = client.post(f"/api/lesson/{lesson_id}/resume").json()
        resumed_session_id = resumed["session_id"]

        with client.stream(
            "POST",
            f"/api/session/{resumed_session_id}/turn",
            json={"text": "Tell me more."},
        ) as resp:
            "".join(resp.iter_text())

    assert len(fake_llm.calls) >= 2
    resumed_call_system_message = fake_llm.calls[-1][0]
    assert resumed_call_system_message["role"] == "system"
    assert "(grade 6)" in resumed_call_system_message["content"], (
        resumed_call_system_message["content"]
    )
