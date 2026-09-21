"""RED/GREEN tests for docs/review_2026-09-20_pass2.md findings.

Covers:
- Finding 1: a tool-call exception, an LLM error mid-stream after a tool
  call, and a cancel mid-tool-loop must never leave ``PromptLog`` with a
  dangling ``tool_calls`` assistant message and no matching tool result.
  ``PromptLog.validate()`` is the authoritative check; a second, normal
  turn afterward must still succeed.
- Finding 2: ``topic_hint`` must not defeat the abstention/coverage gate
  for a genuinely off-topic question, but must still rescue a truly
  elliptical follow-up.
- Finding 3: ``ResearchEngine._response_cache`` must be bounded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

import tests.test_research as _tr
from tutor.app.agent_loop import run_turn
from tutor.app.llm_client import StreamEvent
from tutor.app.prompt import Budget, PromptLog
from tutor.app.session import Session

_engine = _tr._engine
registry_toml = _tr.registry_toml
snapshot_store = _tr.snapshot_store

pytestmark = []


def _count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@dataclass
class _UserInput:
    kind: str
    text: str | None = None
    action: str | None = None


class FakeLlmClient:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls: list[list[dict]] = []

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
        self.calls.append([dict(m) for m in messages])
        script = self._scripts.pop(0)
        yield from script


class FakeCalc:
    def evaluate(self, expression: str, **kwargs):
        return {"ok": True, "result": "4"}


def _token(text: str) -> StreamEvent:
    return StreamEvent(kind="token", text=text)


def _done(finish_reason="stop") -> StreamEvent:
    return StreamEvent(kind="done", finish_reason=finish_reason, usage={})


def _error_event() -> StreamEvent:
    return StreamEvent(kind="error", text="boom")


def _final(text: str) -> list[StreamEvent]:
    return [_token(text), _done()]


def _tool_call(name: str, arguments: dict, call_id="call_1") -> list[StreamEvent]:
    return [
        StreamEvent(kind="tool_call", id=call_id, name=name, arguments_json=json.dumps(arguments)),
        _done(finish_reason="tool_calls"),
    ]


def _mk_session() -> tuple[Session, Budget]:
    return Session(count_tokens=_count_tokens, subject_hint="math"), Budget()


class _OkResearchEngine:
    def __init__(self):
        self.calls = 0

    def research(self, query, *, topic_hint=None, keywords=None):
        self.calls += 1

        @dataclass
        class _Passage:
            label: str
            passage_id: str
            title: str = "Title"
            path: str = "A/Title"
            text: str = "Evidence text. " * 10
            kind: str = "article"

        @dataclass
        class _Response:
            status: str = "ok"
            passages: list = field(default_factory=list)

        pid = f"pid-{self.calls}"
        return _Response(passages=[_Passage(label=f"S{self.calls}", passage_id=pid)])


class _RaisingResearchEngine:
    """Raises on every call -- simulates a ZimWorker timeout/exception not
    swallowed at a lower layer."""

    def research(self, query, *, topic_hint=None, keywords=None):
        raise RuntimeError("archive worker exploded")


# ---------------------------------------------------------------------------
# Finding 1: log corruption on tool-call error / mid-turn failure.
# ---------------------------------------------------------------------------


def test_tool_raises_leaves_log_valid_and_second_turn_succeeds():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            [_token("Let me check."), *_tool_call("research", {"query": "moons of mars"})],
            _final("Here is what I found without the tool."),
            _final("Second turn answer [S1]."),
        ]
    )
    research = _RaisingResearchEngine()
    calc = FakeCalc()

    result1 = run_turn(
        session,
        _UserInput(kind="action", action="explain"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )
    assert result1.status == "ok"
    session.log.validate()  # must not raise

    ok_research = _OkResearchEngine()
    result2 = run_turn(
        session,
        _UserInput(kind="action", action="explain-again"),
        llm=llm,
        research_engine=ok_research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )
    assert result2.status == "ok"
    session.log.validate()


def test_llm_error_mid_stream_after_a_tool_call_leaves_log_valid():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("calc", {"expression": "2+2"}),
            [_error_event()],
            _final("Recovered answer [S1]."),
        ]
    )
    research = _OkResearchEngine()
    calc = FakeCalc()

    result1 = run_turn(
        session,
        _UserInput(kind="action", action="explain"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )
    # The tool call itself completed and was logged validly; the *next*
    # model call then errored mid-stream, which must not touch the log at
    # all (it returns before any further log mutation).
    assert result1.status == "error"
    session.log.validate()

    result2 = run_turn(
        session,
        _UserInput(kind="action", action="explain-again"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )
    assert result2.status == "ok"
    session.log.validate()


def test_cancel_mid_tool_loop_leaves_log_valid_and_second_turn_succeeds():
    import threading

    session, budget = _mk_session()
    cancel = threading.Event()

    class _CancellingResearchEngine:
        def research(self, query, *, topic_hint=None, keywords=None):
            cancel.set()  # simulate cancellation arriving while dispatching
            raise AssertionError("should not be reached: cancel checked first")

    llm = FakeLlmClient(
        [
            [
                StreamEvent(
                    kind="tool_call",
                    id="call_1",
                    name="calc",
                    arguments_json=json.dumps({"expression": "2+2"}),
                ),
                StreamEvent(
                    kind="tool_call",
                    id="call_2",
                    name="research",
                    arguments_json=json.dumps({"query": "moons"}),
                ),
                _done(finish_reason="tool_calls"),
            ],
        ]
    )
    calc = FakeCalc()

    # Pre-set the cancel flag before the second (research) tool call is
    # dispatched by monkeypatching calc.evaluate to set it after call 1.
    orig_evaluate = calc.evaluate

    def _evaluate_then_cancel(expression, **kwargs):
        result = orig_evaluate(expression, **kwargs)
        cancel.set()
        return result

    calc.evaluate = _evaluate_then_cancel

    research = _OkResearchEngine()

    result = run_turn(
        session,
        _UserInput(kind="action", action="explain"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        cancel=cancel,
    )
    assert result.status == "cancelled"
    session.log.validate()

    # A fresh, non-cancelled second turn on the same session must still
    # succeed -- the session/lesson is not wedged.
    cancel2 = threading.Event()
    llm2 = FakeLlmClient([_final("All good [S1].")])
    result2 = run_turn(
        session,
        _UserInput(kind="action", action="explain-again"),
        llm=llm2,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        cancel=cancel2,
    )
    assert result2.status == "ok"
    session.log.validate()


def test_prompt_log_validate_rejects_dangling_tool_calls():
    log = PromptLog(_count_tokens)
    log.append_system("sys")
    log.append_user("hi")
    log.append_assistant_tool_calls(
        [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]
    )
    with pytest.raises(ValueError):
        log.validate()


def test_prompt_log_repair_fixes_dangling_tool_calls():
    log = PromptLog(_count_tokens)
    log.append_system("sys")
    log.append_user("hi")
    log.append_assistant_tool_calls(
        [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]
    )
    assert log.is_valid() is False
    repaired = log.repair()
    assert repaired is True
    log.validate()  # now valid


def test_lesson_store_resume_repairs_broken_persisted_log(tmp_path):
    from tutor.app.lesson_state import LessonStore

    store = LessonStore(tmp_path / "lessons.db")
    lesson_id = store.start_lesson(profile_id="p1", subject="math")
    session = store.new_session(lesson_id, count_tokens=_count_tokens)
    session.log.append_system("sys")
    session.log.append_user("hi")
    session.log.append_assistant_tool_calls(
        [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]
    )
    store.save_session(lesson_id, session)

    resumed = store.resume(lesson_id, count_tokens=_count_tokens)
    resumed.log.validate()  # must not raise: repaired on resume

    # And the repair was persisted, not just in-memory.
    resumed_again = store.resume(lesson_id, count_tokens=_count_tokens)
    resumed_again.log.validate()
    store.close()


# ---------------------------------------------------------------------------
# Finding 2: topic_hint must not defeat abstention.
# ---------------------------------------------------------------------------


def test_off_topic_question_with_matching_topic_hint_is_still_empty(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research(
        "What is the flibbertigibbetopolis effect?", topic_hint="Pythagorean theorem"
    )
    assert resp.status == "empty"
    assert resp.passages == []
    assert resp.coverage["weak"] is True


def test_elliptical_question_with_topic_hint_still_finds_article(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("What about its moons?", topic_hint="Pythagorean theorem")
    assert resp.status in ("ok", "partial")
    assert resp.passages
    assert resp.passages[0].path == "pythagorean_theorem"


def test_coverage_terms_off_topic_query_not_rescued_by_topic_hint():
    from tutor.retrieval.research import _coverage_terms

    # "capital of France" has real content terms of its own (about a
    # different subject entirely) -- the topic_hint must never be folded
    # in for a query like this (finding 2's core scenario).
    terms = _coverage_terms("What is the capital of France?", topic_hint="Photosynthesis")
    assert "photosynthesis" not in terms


def test_coverage_terms_elliptical_query_rescued_by_topic_hint():
    from tutor.retrieval.research import _coverage_terms

    # An elliptical follow-up with essentially no content of its own is
    # still rescued by the topic_hint, whether it's a bare noun ("its
    # moons?") or a generic continuation phrase ("tell me more").
    terms = _coverage_terms("What about its moons?", topic_hint="Volcano")
    assert "volcano" in terms
    assert "moons" in terms

    terms2 = _coverage_terms("Can you tell me more about it?", topic_hint="Volcano")
    assert "volcano" in terms2


def test_best_coverage_ignores_topic_hint_terms_argument():
    from tutor.retrieval.research import _best_coverage

    # _best_coverage takes whatever term set _coverage_terms already
    # decided on; a candidate that only matches via a topic_hint token not
    # present in `query_terms` must stay weak here.
    candidates = [
        {
            "score": 1.0,
            "title": "Photosynthesis",
            "text": "Photosynthesis is how plants make food using sunlight.",
        }
    ]
    cov = _best_coverage(
        candidates,
        {"capital", "france"},
        frozenset({"photosynthesis"}),
    )
    assert cov["weak"] is True


# ---------------------------------------------------------------------------
# Finding 3: bounded LRU response cache.
# ---------------------------------------------------------------------------


def test_response_cache_is_bounded(registry_toml, snapshot_store, tmp_path):
    from tests.test_research import _engine
    from tutor.retrieval.research import _RESPONSE_CACHE_MAXSIZE

    engine = _engine(registry_toml, snapshot_store, tmp_path)
    for i in range(_RESPONSE_CACHE_MAXSIZE + 20):
        engine.research(f"distinct question number {i}")

    assert len(engine._response_cache) <= _RESPONSE_CACHE_MAXSIZE


# ---------------------------------------------------------------------------
# Pass-2b: elliptical recall vs. hint-only title-match rescue.
#
# Both properties are required together (docs/retrieval_baseline.md
# "Pass-2 fix" trade-off is not accepted): candidate generation/ranking
# still sees query + topic_hint terms (`_query_terms`, unchanged); the
# coverage/abstention gate now judges the QUESTION'S OWN content terms
# against the candidate's fetched text, with a topic_hint term allowed to
# satisfy `title_match` only when that same candidate's own-term text
# coverage *also* clears the threshold -- a title match on hint terms
# alone is never sufficient on its own.
# ---------------------------------------------------------------------------


def test_compute_coverage_plural_singular_normalization_is_not_weak():
    from tutor.retrieval.research import compute_coverage

    coverage = compute_coverage(
        query_terms={"moon"},
        title="Something else entirely",
        text="Jupiter has many moons orbiting it.",
    )
    assert coverage["term_coverage"] == 1.0
    assert coverage["weak"] is False


def test_compute_coverage_hint_title_match_requires_own_coverage_threshold():
    from tutor.retrieval.research import compute_coverage

    # Own content terms share nothing with the candidate's text -- a
    # topic_hint term matching the title alone must not rescue it (review
    # pass 2, finding 2's shape, at the compute_coverage level directly).
    coverage = compute_coverage(
        query_terms={"won", "1998", "world", "cup"},
        title="Volcano",
        text="A volcano is an opening that lets hot magma escape.",
        hint_terms=frozenset({"volcano"}),
    )
    assert coverage["title_match"] is False
    assert coverage["weak"] is True


def test_compute_coverage_hint_title_match_allowed_when_own_coverage_meets_threshold():
    from tutor.retrieval.research import compute_coverage

    # Own terms already clear the threshold on their own merits; the hint
    # term also matching the title is then reported truthfully.
    coverage = compute_coverage(
        query_terms={"magma", "opening", "escape"},
        title="Volcano",
        text="A volcano is an opening that lets hot magma escape.",
        hint_terms=frozenset({"volcano"}),
    )
    assert coverage["weak"] is False
    assert coverage["title_match"] is True


def test_zero_own_term_continuation_with_hint_finds_hinted_article(
    registry_toml, snapshot_store, tmp_path
):
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research("Can you tell me more about it?", topic_hint="Pythagorean theorem")
    assert resp.status in ("ok", "partial")
    assert resp.passages
    assert resp.passages[0].path == "pythagorean_theorem"


def test_own_term_text_coverage_does_not_need_hint_at_all():
    # "hypotenuse" is the question's own content term (present verbatim in
    # the pythagorean_theorem fixture text); coverage over own terms alone
    # (no topic_hint) already clears the threshold -- the coverage
    # *decision* (as opposed to candidate ranking, which is a separate
    # concern) never needs the hint's help here.
    from tutor.retrieval.research import compute_coverage

    coverage = compute_coverage(
        query_terms={"hypotenuse"},
        title="Pythagorean theorem",
        text="Pythagoras -- a squared plus b squared equals c squared "
        "describes right triangles. Legs a and b. Hypotenuse c.",
    )
    assert coverage["weak"] is False


def test_hint_only_title_match_with_uncovered_own_terms_is_empty(
    registry_toml, snapshot_store, tmp_path
):
    # Task's literal example shape: a multi-term own question about a
    # totally different subject than the topic_hint must still abstain,
    # even though the hint's own article is in the candidate pool. Own
    # terms are nonsense words unrelated to *any* fixture article's text,
    # so this isolates the hint-only-title-match gate rather than a
    # coincidental real-word overlap.
    engine = _engine(registry_toml, snapshot_store, tmp_path)
    resp = engine.research(
        "Who won the 1998 flibbertigibbetopolis wobblecup?",
        topic_hint="Pythagorean theorem",
    )
    assert resp.status == "empty"
    assert resp.passages == []
