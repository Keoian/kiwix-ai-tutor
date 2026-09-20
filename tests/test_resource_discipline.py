"""RED tests for tutor.app.resources (WP-C4, M5 gate: "30-minute lesson
within resource limits").

Spec sources:
- docs/plan/offline_tutor_spec_v0.3.md §10 (Memory, scheduling,
  cancellation) resource table: "Retrieval worker | <= 512 MiB target, 1
  GiB hard cgroup limit", "Embedding model | <= 150 MiB resident", "App
  RAM cache | 128 MiB", "OS headroom | >= 2 GiB MemAvailable",
  "Inference host threads | 2", "Retrieval threads | 1 for
  Xapian/libzim (worker), 1 for BM25/embedding (app)". The spec gives no
  single "app process RSS ceiling" number, so ResourceLimits.app_rss_mb
  is a contract decision made here (task brief): comfortably above the
  documented 128 MiB app cache + embedding's 150 MiB, generous headroom
  under the machine's stated 2 GiB `MemAvailable` floor.
- §15 acceptance table: "Resource stability | 30-minute lesson: no OOM,
  no queue growth, no sustained swap, no thermal collapse below the
  decode gate."
- §8.1 eviction: "eviction_reprefill" events; the 32K profile (plan
  §0.2) uses a much larger ceiling than the small one this test injects
  to force eviction quickly, matching WP-C1's own note: "The eviction
  path is exercised in tests by injecting a small ceiling (e.g. 6K)".
- Plan WP-C4: "Resource measurements on Windows are indicative only; the
  numbers that count are re-taken on the Dell in M6" -- so this test's
  thresholds are generous/indicative, not the M6 gate numbers.

No GPU/server, no network; process/child/thread/handle counts observed
via psutil against the *current* process only (fakes only, no real
llama-server or retrieval worker).
"""

from __future__ import annotations

from tutor.app.agent_loop import run_turn
from tutor.app.llm_client import StreamEvent
from tutor.app.prompt import Budget
from tutor.app.resources import ResourceLimits, ResourceMonitor
from tutor.app.session import Session


def _count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class _FakeUserInput:
    def __init__(self, text):
        self.kind = "text"
        self.text = text
        self.action = None


class _FakeResearchEngine:
    def research(self, query, topic_hint=None):
        return {
            "status": "ok",
            "passages": [
                {"id": f"pid-{query}", "label": "S1", "text": "A fact " * 50}
            ],
        }


class _FakeLLM:
    """Streams a short fixed answer with no tool calls, so run_turn
    completes in one pass per turn -- fast enough for a 60-turn
    simulation to stay well under the `slow` marker's 20s cutoff."""

    def stream_chat(self, messages, *, tools=None, cancel=None, max_tokens=None):
        yield StreamEvent(kind="token", text="The answer is short [S1].")
        yield StreamEvent(kind="done", finish_reason="stop")


class _FakeCalc:
    def evaluate(self, expression):
        return {"ok": True, "result": "0"}


def _noop_emit(event):
    pass


class TestResourceMonitorSampling:
    def test_sample_reports_rss_mb(self):
        monitor = ResourceMonitor()
        sample = monitor.sample()
        assert sample.rss_mb > 0

    def test_sample_reports_open_files_and_threads(self):
        monitor = ResourceMonitor()
        sample = monitor.sample()
        assert sample.open_files >= 0
        assert sample.thread_count >= 1
        assert sample.child_process_count >= 0

    def test_check_returns_no_violations_when_within_limits(self):
        # Huge limits so a normal test process trivially passes.
        limits = ResourceLimits(
            app_rss_mb=100_000, max_open_files=100_000, max_child_processes=100_000
        )
        monitor = ResourceMonitor(limits=limits)
        violations = monitor.check()
        assert violations == []

    def test_check_flags_rss_violation(self):
        limits = ResourceLimits(app_rss_mb=1, max_open_files=100_000, max_child_processes=100_000)
        monitor = ResourceMonitor(limits=limits)
        violations = monitor.check()
        assert any("rss" in v.lower() for v in violations)


class TestSimulatedThirtyMinuteLesson:
    def test_sixty_turns_stay_within_ceiling_with_bounded_rss_growth(self):
        ceiling = 6000
        margin = 200
        budget = Budget.scaled(ceiling)
        session = Session(count_tokens=_count_tokens, subject_hint="math")
        session.log.append_system("You are a tutor. " * 5)

        monitor = ResourceMonitor()
        llm = _FakeLLM()
        research_engine = _FakeResearchEngine()
        calc = _FakeCalc()

        eviction_events = []
        rss_at_turn: dict[int, float] = {}
        child_counts: list[int] = []
        open_files: list[int] = []

        for i in range(60):
            user_input = _FakeUserInput(f"Question number {i} about fractions and decimals?")
            result = run_turn(
                session,
                user_input,
                llm=llm,
                research_engine=research_engine,
                calc=calc,
                budget=budget,
                emit=_noop_emit,
                cancel=None,
            )
            assert result.status == "ok"

            eviction_events.extend(result.events)

            tokens_used = session.log.tokens_used()
            assert tokens_used <= ceiling - margin, (
                f"turn {i}: tokens_used={tokens_used} exceeds ceiling-margin"
            )

            sample = monitor.sample()
            if i in (10, 60 - 1):
                rss_at_turn[i] = sample.rss_mb
            child_counts.append(sample.child_process_count)
            open_files.append(sample.open_files)

        assert len(eviction_events) > 0, "expected eviction to fire at least once over 60 turns"

        rss_growth = rss_at_turn[59] - rss_at_turn[10]
        assert rss_growth < 30, f"RSS grew {rss_growth} MB between turn 10 and turn 60"

        assert max(child_counts) == min(child_counts), "child process count grew during the lesson"
        assert max(open_files) - min(open_files) <= 5, "open file/handle count grew unexpectedly"

    def test_prefix_property_holds_between_evictions(self):
        # prompt.py: "the property resumes holding for all subsequent
        # appends" after an eviction. Verify serialize_messages before an
        # append (post-eviction) is a strict prefix of after.
        from tutor.app.prompt import serialize_messages

        ceiling = 6000
        budget = Budget.scaled(ceiling)
        session = Session(count_tokens=_count_tokens, subject_hint="math")
        session.log.append_system("You are a tutor.")
        llm = _FakeLLM()
        research_engine = _FakeResearchEngine()
        calc = _FakeCalc()

        saw_eviction = False
        for i in range(30):
            user_input = _FakeUserInput(f"Question {i}?")
            run_turn(
                session,
                user_input,
                llm=llm,
                research_engine=research_engine,
                calc=calc,
                budget=budget,
                emit=_noop_emit,
                cancel=None,
            )
            before = serialize_messages(session.log.render())
            # simulate the next append starting from this exact state
            after_user = _FakeUserInput(f"Question {i}-followup?")
            result = run_turn(
                session,
                after_user,
                llm=llm,
                research_engine=research_engine,
                calc=calc,
                budget=budget,
                emit=_noop_emit,
                cancel=None,
            )
            after = serialize_messages(session.log.render())
            if result.events:
                # Eviction is the sole, explicit exception to the
                # byte-prefix property (a head edit): it is allowed --
                # even expected -- to break the prefix for this one turn.
                saw_eviction = True
            else:
                assert after.startswith(before), f"prefix property broke at turn {i}"

        assert saw_eviction, "expected at least one eviction over 30 turns to exercise this path"
