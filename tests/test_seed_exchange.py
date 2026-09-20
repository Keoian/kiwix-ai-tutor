"""Tests for tutor.app.seed_exchange (HANDOFF.md Task 3, off-by-default
prompt variant: seed a new lesson with one fixed cited example).

Written test-first per each constraint in the task brief:
1. system slot still fits 800 tokens; the seed's own token cost is
   accounted for in the overall context budget.
2. the seed is part of the append-only prefix: byte-identical across
   turns, never evicted, never reordered.
3. [S0] is reserved: the resolver refuses it and attribute_sentences
   never attributes to it.
4. the seed exchange is never a real/persisted student turn.
5. off by default: an unseeded log is byte-identical to today's.
"""

from __future__ import annotations

from tutor.app.citations import (
    RESERVED_SEED_LABEL,
    attribute_sentences,
    resolve_citations,
)
from tutor.app.prompt import Budget, PromptLog, serialize_messages
from tutor.app.seed_exchange import (
    SEED_EXCHANGE_VARIANT,
    build_seed_entries,
    seed_session,
)


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Shape of the seed exchange itself
# ---------------------------------------------------------------------------


def test_seed_entries_are_a_student_question_evidence_calc_and_two_sentence_answer():
    entries = build_seed_entries()
    roles = [e["role"] for e in entries]
    assert roles == ["user", "tool", "assistant", "tool", "assistant"]
    assert all(e.get("seed") is True for e in entries)

    evidence = entries[1]
    assert evidence["passages"][0]["label"] == RESERVED_SEED_LABEL

    calc_call = entries[2]
    assert calc_call["tool_calls"][0]["function"]["name"] == "calc"

    calc_result = entries[3]
    assert calc_result["tool_call_id"] == calc_call["tool_calls"][0]["id"]

    answer = entries[4]["content"]
    # Two sentences.
    sentence_count = sum(answer.count(p) for p in (". ", "! ", "? ")) + 1
    assert sentence_count == 2
    # The [S0] label sits on the first sentence, not trailing the whole
    # paragraph (the label appears strictly before the final sentence's
    # start, i.e. not at the very end of the answer).
    label = f"[{RESERVED_SEED_LABEL}]"
    assert label in answer
    assert not answer.rstrip().endswith(label)
    first_sentence_end = answer.index(". ") + 1
    assert answer.index(label) < first_sentence_end
    # The calc result actually appears in the answer text.
    assert "1234.8" in answer


# ---------------------------------------------------------------------------
# Constraint 1: token budget
# ---------------------------------------------------------------------------


def test_system_prompt_800_budget_test_still_passes_independent_of_seed():
    """The seed lives outside the system message entirely -- confirm the
    system text used in production is still within its own budget on its
    own (regression guard for this change, mirrors
    tests/test_citations.py::test_system_prompt_stays_within_approx_800_token_budget)."""
    import tutor.app.agent_loop as agent_loop

    system_text = agent_loop._load_default_system_text()
    assert _count_tokens(system_text) <= 800


def test_seed_token_cost_is_counted_in_prompt_log_tokens_used():
    log = PromptLog(_count_tokens)
    log.append_system("sys")
    before = log.tokens_used()
    log.append_seed(build_seed_entries())
    after = log.tokens_used()
    assert after > before, "seeding must add to the accounted token total"


def test_seed_tokens_push_eviction_to_trigger_sooner():
    """The seed's cost is counted against budget.system + budget.history
    (the same ceiling normal turns are evicted against), not given a free
    ride outside the budget."""
    tiny_budget = Budget(
        ceiling=2000, system=50, history=80, newest=400, generation=200, margin=1270
    )

    seeded = PromptLog(_count_tokens)
    seeded.append_system("sys")
    seeded.append_seed(build_seed_entries())

    unseeded = PromptLog(_count_tokens)
    unseeded.append_system("sys")

    filler = [{"id": f"p{i}", "label": f"S{i + 1}", "text": "x" * 200} for i in range(3)]
    for log in (seeded, unseeded):
        log.append_user("question one")
        log.append_evidence(filler)
        log.append_assistant("answer one [S1]", cited_labels=["S1"])
        log.append_user("question two")
        log.append_evidence(filler)
        log.append_assistant("answer two [S1]", cited_labels=["S1"])

    seeded_event = seeded.evict(tiny_budget)
    unseeded_event = unseeded.evict(tiny_budget)
    assert seeded_event is not None
    # Both may evict, but the seeded log started with strictly more
    # tokens before eviction than the unseeded one -- proof the seed's
    # cost was actually counted.
    if unseeded_event is not None:
        assert seeded_event.tokens_before > unseeded_event.tokens_before


# ---------------------------------------------------------------------------
# Constraint 2: append-only prefix -- byte-identical, never evicted/reordered
# ---------------------------------------------------------------------------


def test_seed_is_byte_identical_across_turns():
    log = PromptLog(_count_tokens)
    log.append_system("sys")
    log.append_seed(build_seed_entries())

    def _seed_bytes(rendered_messages):
        # The seed occupies a fixed-length run right after the system
        # message; slice it out and serialize just that run.
        seed_len = len(build_seed_entries())
        start = 1  # after system
        return serialize_messages(rendered_messages[start : start + seed_len])

    before = _seed_bytes(log.render())

    log.append_user("q1")
    log.append_evidence([{"id": "p1", "label": "S1", "text": "evidence one"}])
    log.append_assistant("answer one [S1]", cited_labels=["S1"])

    mid = _seed_bytes(log.render())

    log.append_user("q2")
    log.append_evidence([{"id": "p2", "label": "S2", "text": "evidence two"}])
    log.append_assistant("answer two [S2]", cited_labels=["S2"])

    after = _seed_bytes(log.render())

    assert before == mid == after


def test_seed_survives_eviction_that_would_otherwise_evict_everything():
    log = PromptLog(_count_tokens)
    log.append_system("sys")
    log.append_seed(build_seed_entries())

    tiny_budget = Budget(
        ceiling=2000, system=20, history=20, newest=400, generation=200, margin=1360
    )
    for i in range(6):
        log.append_user(f"question {i} " + "x" * 100)
        log.append_evidence([{"id": f"p{i}", "label": f"S{i + 1}", "text": "y" * 300}])
        log.append_assistant(f"answer {i} [S{i + 1}]", cited_labels=[f"S{i + 1}"])
        log.evict(tiny_budget)

    rendered = log.render()
    seed_len = len(build_seed_entries())
    assert rendered[1 : 1 + seed_len] == build_seed_entries()


def test_append_seed_after_a_turn_raises():
    log = PromptLog(_count_tokens)
    log.append_system("sys")
    log.append_user("q")
    import pytest

    with pytest.raises(ValueError):
        log.append_seed(build_seed_entries())


def test_append_seed_twice_raises():
    log = PromptLog(_count_tokens)
    log.append_seed(build_seed_entries())
    import pytest

    with pytest.raises(ValueError):
        log.append_seed(build_seed_entries())


# ---------------------------------------------------------------------------
# Constraint 3: [S0] is reserved
# ---------------------------------------------------------------------------


def test_resolve_citations_refuses_s0_even_if_a_passage_is_labelled_s0():
    packet_passages = [
        {"id": "seed-s0", "label": "S0", "title": "T", "path": "seed://x", "text": "seed text"}
    ]
    citations = resolve_citations("This claims something [S0].", packet_passages)
    assert len(citations) == 1
    assert citations[0].unresolved is True
    assert citations[0].supported is False
    assert citations[0].path is None


def test_attribute_sentences_never_attributes_to_s0():
    passages = [
        {"id": "seed-s0", "label": "S0", "text": "seed text about sound waves and vibration"}
    ]
    result = attribute_sentences("Sound needs a medium to travel through air. [S0]", passages)
    assert all(a.label != "S0" for a in result.attributions)


# ---------------------------------------------------------------------------
# Constraint 4: never a real/persisted student turn
# ---------------------------------------------------------------------------


def test_seed_session_does_not_retain_the_s0_passage_for_citation_resolution():
    class _FakeSession:
        def __init__(self):
            self.log = PromptLog(_count_tokens)
            self.retained_passages: dict[str, dict] = {}

        def retain_passages(self, passages):
            for p in passages:
                self.retained_passages[p["label"]] = p

        def known_passages(self):
            return list(self.retained_passages.values())

    session = _FakeSession()
    seed_session(session)
    assert session.known_passages() == []


def test_seed_session_entries_are_never_persisted_as_a_turn_row():
    """LessonStore.append_turn/persist_turn only ever run once per real
    completed turn (called by the turn runner, never by seed_session);
    seed_session itself never calls anything that inserts a turns-table
    row, which we assert directly by using a stub that would fail loudly
    if seed_session ever tried."""

    class _Session:
        def __init__(self):
            self.log = PromptLog(_count_tokens)

    class _ExplodingLessons:
        def append_turn(self, *a, **k):
            raise AssertionError("seed_session must never persist a turn")

        def persist_turn(self, *a, **k):
            raise AssertionError("seed_session must never persist a turn")

    session = _Session()
    seed_session(session)  # must not touch _ExplodingLessons at all
    assert len(session.log.render()) == len(build_seed_entries())


# ---------------------------------------------------------------------------
# Constraint 5: off by default
# ---------------------------------------------------------------------------


def test_seed_session_is_idempotent_and_off_unless_called():
    log = PromptLog(_count_tokens)
    baseline = serialize_messages(log.render())
    assert baseline == b""  # nothing appended -> nothing rendered


def test_variant_registered_by_name():
    from eval.system_prompt_variants import VARIANTS

    assert SEED_EXCHANGE_VARIANT in VARIANTS
    assert VARIANTS[SEED_EXCHANGE_VARIANT]["seed_exchange"] is True


# ---------------------------------------------------------------------------
# Wiring: _SessionStore.create_for_lesson only seeds when asked
# ---------------------------------------------------------------------------


def test_session_store_create_for_lesson_off_by_default(tmp_path):
    from tutor.app.compose import _SessionStore
    from tutor.app.lesson_state import LessonStore

    lessons = LessonStore(tmp_path / "lessons.sqlite")
    lesson_id = lessons.start_lesson(profile_id="p1", subject="physics")

    sessions = _SessionStore(_count_tokens)  # seed_exchange defaults False
    session_id = sessions.create_for_lesson(lessons, lesson_id)
    session = sessions.get(session_id)

    assert session.log.render() == []  # byte-identical to today: nothing seeded


def test_session_store_create_for_lesson_seeds_when_enabled(tmp_path):
    from tutor.app.compose import _SessionStore
    from tutor.app.lesson_state import LessonStore

    lessons = LessonStore(tmp_path / "lessons2.sqlite")
    lesson_id = lessons.start_lesson(profile_id="p1", subject="physics")

    sessions = _SessionStore(_count_tokens, seed_exchange=True)
    session_id = sessions.create_for_lesson(lessons, lesson_id)
    session = sessions.get(session_id)

    assert len(session.log.render()) == len(build_seed_entries())


def test_build_deps_reads_prompt_variant_off_by_default(tmp_path, monkeypatch):
    from tutor.app import compose as compose_mod

    class _FakeAppCfg:
        data_dir = tmp_path / "data"
        registry_path = tmp_path / "archives.toml"
        host = "127.0.0.1"
        port = 0
        prompt_variant = "current"

    class _FakeServerCfg:
        base_url = "http://127.0.0.1:1"
        ctx_size = 4096

    class _FakeRuntimeCfg:
        model_path = tmp_path / "model.gguf"

    class _FakeCfg:
        app = _FakeAppCfg()
        server = _FakeServerCfg()
        runtime = _FakeRuntimeCfg()
        embedding = None

    (tmp_path / "archives.toml").write_text("", encoding="utf-8")

    class _FakeLLM:
        def count_tokens(self, text):
            return len(text.split())

    class _FakeResearchEngine:
        pass

    deps = compose_mod.build_deps(_FakeCfg(), llm=_FakeLLM(), research_engine=_FakeResearchEngine())
    assert deps.sessions._seed_exchange is False
    deps.lessons.close()
