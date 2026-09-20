"""RED tests for status/metrics additions (WP-C4, M5 gate: "status page").

Spec sources:
- docs/plan/offline_tutor_spec_v0.3.md §12 UI: "Status | Model loaded,
  archives healthy, embedding index present, warm/cold cache state;
  derived from local checks only."
- §15 Logging and evaluation: "Logging fields from v0.2 §15 are
  retained. Added: `route` (`action` / `preretrieve` /
  `preretrieve+followup`), `calc_calls`, `eviction_reprefill` events,
  prompt-cache hit tokens per request, and GPU clock/temperature
  samples." These fields must appear in the per-turn JSON-lines log.
- Privacy: the spec does not use the word "privacy" verbatim, but §11
  "learner state kept small and explicit" and the citation-snapshot
  model (store extracts, not raw model chat transcripts, as the
  long-term record) together with there being no stated requirement to
  retain verbatim student wording motivate the contract decision made
  here (task brief: "no student free text in logs if the spec's privacy
  section says so"): the turns.jsonl log stores metrics/route/ids, never
  the student's raw `user_text`, matching "kept small and explicit".
  This is a contract decision, not a verbatim spec quote -- flagged in
  the final report as spec-undefined.
- Plan WP-C4 / M5 row: "status page" as one of the M5 deliverables.

Fakes only, no network, no GPU. utf-8 everywhere, pathlib only.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tutor.app.main import AppDeps, create_app
from tutor.app.turn_log import TurnLogger


def _count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class _FakeSessions:
    def create(self):
        return "sess-1"

    def get(self, session_id):
        return {"subject": None} if session_id == "sess-1" else None

    def set_subject(self, session_id, subject):
        return session_id == "sess-1"


class _FakeSnapshotStore:
    def get(self, passage_id):
        return None


def _fake_turn_runner(session_id, user_input, emit, cancel):
    emit("done", {"text": "ok"})


class TestStatusExposesResourcesAndLastTurnMetrics:
    def test_status_endpoint_includes_resource_samples(self):
        def status_provider(session_id=None):
            return {
                "model_loaded": True,
                "archives_healthy": True,
                "embedding_index_present": True,
                "cache_state": "warm",
                "resources": {
                    "rss_mb": 512.0,
                    "open_files": 12,
                    "thread_count": 4,
                    "child_process_count": 1,
                },
                "last_turn": {
                    "route": "preretrieve",
                    "first_token_ms": 850,
                    "tokens_per_second": 22.5,
                    "cached_tokens": 300,
                },
            }

        deps = AppDeps(
            turn_runner=_fake_turn_runner,
            snapshot_store=_FakeSnapshotStore(),
            status_provider=status_provider,
            sessions=_FakeSessions(),
            subjects=["math"],
        )
        app = create_app(deps)
        client = TestClient(app)
        response = client.get("/api/status")
        assert response.status_code == 200
        body = response.json()
        assert "resources" in body
        assert body["resources"]["rss_mb"] == 512.0
        assert "last_turn" in body
        assert body["last_turn"]["route"] == "preretrieve"
        assert body["last_turn"]["tokens_per_second"] == 22.5
        assert body["last_turn"]["cached_tokens"] == 300

    def test_status_endpoint_reports_local_checks_only_fields(self):
        # spec §12: "Model loaded, archives healthy, embedding index
        # present, warm/cold cache state; derived from local checks only."
        def status_provider(session_id=None):
            return {
                "model_loaded": True,
                "archives_healthy": True,
                "embedding_index_present": False,
                "cache_state": "cold",
                "resources": {},
                "last_turn": None,
            }

        deps = AppDeps(
            turn_runner=_fake_turn_runner,
            snapshot_store=_FakeSnapshotStore(),
            status_provider=status_provider,
            sessions=_FakeSessions(),
            subjects=["math"],
        )
        app = create_app(deps)
        client = TestClient(app)
        body = client.get("/api/status").json()
        assert body["embedding_index_present"] is False
        assert body["cache_state"] == "cold"


class TestTurnLoggerWritesJsonLines:
    def test_writes_one_utf8_json_line_per_turn(self, tmp_path):
        log_path = tmp_path / "logs" / "turns.jsonl"
        logger = TurnLogger(log_path)

        logger.log_turn(
            lesson_id="lesson-1",
            route="preretrieve",
            calc_calls=1,
            research_calls=1,
            eviction_reprefill=False,
            cached_tokens=120,
            tokens_used=800,
            first_token_ms=900,
            tokens_per_second=18.0,
            gpu_clock_mhz=1500,
            gpu_temp_c=62,
        )
        logger.log_turn(
            lesson_id="lesson-1",
            route="action",
            calc_calls=0,
            research_calls=0,
            eviction_reprefill=True,
            cached_tokens=0,
            tokens_used=200,
            first_token_ms=100,
            tokens_per_second=25.0,
            gpu_clock_mhz=1500,
            gpu_temp_c=63,
        )

        assert log_path.exists()
        raw = log_path.read_bytes()
        # utf-8 decodable
        text = raw.decode("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            # spec §15 required fields, retained + added.
            for key in (
                "route",
                "calc_calls",
                "research_calls",
                "eviction_reprefill",
                "cached_tokens",
                "tokens_used",
                "first_token_ms",
                "tokens_per_second",
                "gpu_clock_mhz",
                "gpu_temp_c",
            ):
                assert key in record, f"missing logging field: {key}"

    def test_route_field_restricted_to_spec_values(self, tmp_path):
        logger = TurnLogger(tmp_path / "logs" / "turns.jsonl")
        with pytest.raises(ValueError):
            logger.log_turn(
                lesson_id="lesson-1",
                route="not-a-real-route",
                calc_calls=0,
                research_calls=0,
                eviction_reprefill=False,
                cached_tokens=0,
                tokens_used=10,
                first_token_ms=10,
                tokens_per_second=1.0,
                gpu_clock_mhz=None,
                gpu_temp_c=None,
            )

    def test_no_student_free_text_recorded_in_log(self, tmp_path):
        # Contract decision (see module docstring): the logger's API has
        # no parameter for student free text at all, so it structurally
        # cannot appear in a logged record.
        import inspect

        logger = TurnLogger(tmp_path / "logs" / "turns.jsonl")
        params = set(inspect.signature(logger.log_turn).parameters)
        assert "user_text" not in params
        assert "text" not in params
        assert "message" not in params

    def test_appends_across_reopen(self, tmp_path):
        log_path = tmp_path / "logs" / "turns.jsonl"
        logger1 = TurnLogger(log_path)
        logger1.log_turn(
            lesson_id="lesson-1",
            route="action",
            calc_calls=0,
            research_calls=0,
            eviction_reprefill=False,
            cached_tokens=0,
            tokens_used=10,
            first_token_ms=10,
            tokens_per_second=1.0,
            gpu_clock_mhz=None,
            gpu_temp_c=None,
        )
        del logger1

        logger2 = TurnLogger(log_path)
        logger2.log_turn(
            lesson_id="lesson-1",
            route="action",
            calc_calls=0,
            research_calls=0,
            eviction_reprefill=False,
            cached_tokens=0,
            tokens_used=20,
            first_token_ms=20,
            tokens_per_second=2.0,
            gpu_clock_mhz=None,
            gpu_temp_c=None,
        )

        raw_lines = log_path.read_text(encoding="utf-8").splitlines()
        lines = [line for line in raw_lines if line.strip()]
        assert len(lines) == 2
