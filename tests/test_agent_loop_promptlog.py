"""RED/GREEN tests for run_turn's integration with the session's
append-only PromptLog (M5).

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §6 (turn flow)
and §8 (append-only log layout, eviction, evidence dedupe by passage id);
docs/M3_report.md (OpenAI tool_calls wire-format replay shape).

These tests use a real ``tutor.app.session.Session`` (backed by a real
``PromptLog``), a scripted fake LLM (asserting on the actual messages
received call-to-call), and fake research/calc collaborators -- no
network, no real model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tutor.app.agent_loop import run_turn
from tutor.app.llm_client import StreamEvent
from tutor.app.prompt import Budget, serialize_messages
from tutor.app.session import Session


def _count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@dataclass
class _UserInput:
    kind: str
    text: str | None = None
    action: str | None = None


class FakeLlmClient:
    def __init__(self, scripts: list[list[StreamEvent]]):
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

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeResearchEngine:
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
            text: str = "Some evidence text about the topic in some detail. " * 15
            kind: str = "article"

        @dataclass
        class _Response:
            status: str = "ok"
            passages: list = field(default_factory=list)

        pid = f"pid-{self.calls}"
        return _Response(passages=[_Passage(label=f"S{self.calls}", passage_id=pid)])


class FakeCalc:
    def evaluate(self, expression: str, **kwargs):
        return {"ok": True, "result": "4"}


def _token(text: str) -> StreamEvent:
    return StreamEvent(kind="token", text=text)


def _done(finish_reason="stop") -> StreamEvent:
    return StreamEvent(kind="done", finish_reason=finish_reason, usage={})


def _final(text: str) -> list[StreamEvent]:
    return [_token(text), _done()]


def _tool_call(name: str, arguments: dict, call_id="call_1") -> list[StreamEvent]:
    return [
        StreamEvent(kind="tool_call", id=call_id, name=name, arguments_json=json.dumps(arguments)),
        _done(finish_reason="tool_calls"),
    ]


def _mk_session(ceiling: int | None = None) -> Session:
    budget = Budget.scaled(ceiling) if ceiling else Budget()
    session = Session(count_tokens=_count_tokens, subject_hint="math")
    return session, budget


def test_run_turn_log_order_user_evidence_assistant():
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Water boils at 100C [S1].")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="At what temperature does water boil?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )

    rendered = session.log.render()
    roles = [m["role"] for m in rendered]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "tool" in roles
    assert roles[-1] == "assistant"


def test_turn_two_messages_start_with_turn_one_messages_byte_for_byte_when_no_eviction():
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Answer one [S1]."), _final("Answer two [S2].")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="Question one?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )
    call1_messages = llm.calls[0]

    run_turn(
        session,
        _UserInput(kind="text", text="Question two?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )
    call2_messages = llm.calls[1]

    before_bytes = serialize_messages(call1_messages)
    after_bytes = serialize_messages(call2_messages)
    assert after_bytes.startswith(before_bytes)


def test_small_ceiling_forces_eviction_before_model_call_and_reports_event():
    session, budget = _mk_session(ceiling=6000)
    session.log.append_system("You are a tutor. " * 5)
    llm = FakeLlmClient([_final("Answer [S1].") for _ in range(30)])
    research = FakeResearchEngine()
    calc = FakeCalc()

    saw_eviction = False
    for i in range(30):
        result = run_turn(
            session,
            _UserInput(kind="text", text=f"Question {i} about fractions and decimals in depth?"),
            llm=llm,
            research_engine=research,
            calc=calc,
            budget=budget,
            emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )
        assert result.status == "ok"
        if result.events:
            saw_eviction = True
            for ev in result.events:
                assert ev.kind == "eviction_reprefill"

        max_allowed = budget.ceiling - budget.generation - budget.margin
        # the messages actually sent to the model this turn (last call)
        sent_tokens = sum(_count_tokens(json.dumps(m)) for m in llm.calls[-1])
        assert sent_tokens <= max_allowed + 200  # small slack for JSON overhead

    assert saw_eviction, "expected eviction to fire at least once with a 6K ceiling"


def test_evicted_evidence_still_cited_is_protected_in_next_model_call():
    session, budget = _mk_session(ceiling=6000)
    llm = FakeLlmClient([_final(f"Answer {i} [S{i + 1}].") for i in range(30)])
    research = FakeResearchEngine()
    calc = FakeCalc()

    for i in range(30):
        run_turn(
            session,
            _UserInput(kind="text", text=f"Question {i} about a big topic in fractions?"),
            llm=llm,
            research_engine=research,
            calc=calc,
            budget=budget,
            emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )

    rendered = session.log.render()
    cited_labels = set()
    for m in rendered:
        if m.get("role") == "assistant":
            cited_labels.update(m.get("cited_labels") or [])
    for label in cited_labels:
        assert any(label in json.dumps(m) for m in rendered)


def test_evidence_not_resent_duplicate_across_turns_when_same_passage():
    class _FixedResearch:
        def research(self, query, *, topic_hint=None, keywords=None):
            @dataclass
            class _Passage:
                label: str = "S1"
                passage_id: str = "shared-pid"
                title: str = "Title"
                path: str = "A/Title"
                text: str = (
                    "the shared fact text has many more than eight words in "
                    "it so a pointer snippet never equals the full passage"
                )
                kind: str = "article"

            @dataclass
            class _Response:
                status: str = "ok"
                passages: list = field(default_factory=lambda: [_Passage()])

            return _Response()

    session, budget = _mk_session()
    llm = FakeLlmClient([_final("A [S1]."), _final("B [S1] again.")])
    research = _FixedResearch()
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="Q1?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="Q2?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )

    full_text = json.dumps(session.log.render())
    # docs/passage_reuse.md: a passage id already held is pasted as a
    # short pointer on the second turn, not repasted in full.
    assert full_text.count("eight words in it") == 1
    assert "already shown above" in full_text


def test_action_appends_user_entry_and_no_evidence():
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Here is a simpler explanation.")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="action", action="simpler"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )

    assert research.calls == 0
    rendered = session.log.render()
    roles = [m["role"] for m in rendered]
    assert "user" in roles
    assert not any(m.get("passages") for m in rendered)


def test_oversized_newest_packet_is_trimmed_not_crashed():
    class _HugeResearch:
        def research(self, query, *, topic_hint=None, keywords=None):
            @dataclass
            class _Passage:
                label: str
                passage_id: str
                title: str = "Title"
                path: str = "A/Title"
                text: str = "word " * 4000
                kind: str = "article"

            @dataclass
            class _Response:
                status: str = "ok"
                passages: list = field(
                    default_factory=lambda: [
                        _Passage(label="S1", passage_id="p1"),
                        _Passage(label="S2", passage_id="p2"),
                        _Passage(label="S3", passage_id="p3"),
                    ]
                )

            return _Response()

    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Answer despite huge evidence.")])
    research = _HugeResearch()
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="Explain everything."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )

    assert result.status == "ok"


def test_turn_logger_receives_eviction_events(tmp_path):
    from tutor.app.turn_log import TurnLogger

    session, budget = _mk_session(ceiling=6000)
    llm = FakeLlmClient([_final(f"Answer {i} [S1].") for i in range(30)])
    research = FakeResearchEngine()
    calc = FakeCalc()
    logger = TurnLogger(tmp_path / "turns.jsonl")

    eviction_seen = False
    for i in range(30):
        result = run_turn(
            session,
            _UserInput(kind="text", text=f"Question {i} about fractions in great depth?"),
            llm=llm,
            research_engine=research,
            calc=calc,
            budget=budget,
            emit=lambda e: None,
        rewrite_on_followup=False,
        model_writes_search=False,
    )
        logger.log_turn(
            lesson_id="lesson-1",
            route=result.route,
            calc_calls=result.calc_calls,
            research_calls=result.research_calls,
            eviction_reprefill=bool(result.events),
            cached_tokens=result.cached_tokens or 0,
            tokens_used=session.log.tokens_used(),
            first_token_ms=None,
            tokens_per_second=None,
            gpu_clock_mhz=None,
            gpu_temp_c=None,
        )
        if result.events:
            eviction_seen = True

    lines = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert any(r["eviction_reprefill"] for r in records)
    assert eviction_seen
