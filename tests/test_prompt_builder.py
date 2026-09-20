"""Unit tests for tutor.app.prompt (Budget, PromptLog, eviction, prefix property).

Pure/no-network: token counting is injected as a simple callable.
"""

from __future__ import annotations

import random

import pytest

from tutor.app.prompt import (
    Budget,
    EvictionEvent,
    PromptLog,
    PromptOverflow,
    serialize_messages,
)


def _count_tokens(text: str) -> int:
    """Deterministic, injectable token counter: ~1 token per 4 chars, min 1."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_default_budget_matches_32k_table():
    budget = Budget()
    assert budget.ceiling == 32768
    assert budget.system == 800
    assert budget.history == 22000
    assert budget.newest == 2500
    assert budget.generation == 2000
    assert budget.margin == 5468


def test_default_budget_sums_to_ceiling():
    budget = Budget()
    total = budget.system + budget.history + budget.newest + budget.generation + budget.margin
    assert total == budget.ceiling == 32768


def test_budget_scaled_small_ceiling_for_eviction_tests():
    budget = Budget.scaled(6000)
    assert budget.ceiling == 6000
    total = budget.system + budget.history + budget.newest + budget.generation + budget.margin
    assert total == 6000


# ---------------------------------------------------------------------------
# PromptLog basic append / render
# ---------------------------------------------------------------------------


def test_log_starts_empty_and_renders_no_messages():
    log = PromptLog(count_tokens=_count_tokens)
    assert log.render() == []


def test_append_system_must_be_first_and_only_once():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("You are a tutor.")
    rendered = log.render()
    assert rendered[0]["role"] == "system"
    assert rendered[0]["content"] == "You are a tutor."

    with pytest.raises((ValueError, RuntimeError)):
        log.append_system("Second system message.")


def test_append_system_after_other_entries_raises():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_user("hi")
    with pytest.raises((ValueError, RuntimeError)):
        log.append_system("late system")


def test_chronological_append_order_is_preserved_in_render():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    log.append_user("question one")
    log.append_evidence(passages=[{"id": "p1", "label": "S1", "text": "fact one"}])
    log.append_assistant("answer one", cited_labels=["S1"])
    rendered = log.render()
    roles = [m["role"] for m in rendered]
    assert roles[0] == "system"
    assert roles[1] == "user"
    # evidence is a tool-result-shaped message somewhere before the assistant reply
    assert "assistant" in roles
    assert roles.index("assistant") == len(roles) - 1


def test_append_calc_tool_result():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    log.append_user("what is 2+2")
    log.append_calc_result(expression="2+2", result="4")
    log.append_assistant("The answer is 4.", cited_labels=[])
    rendered = log.render()
    assert any("2+2" in str(m) or "4" in str(m) for m in rendered)


def test_appending_never_mutates_earlier_entries():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    log.append_user("q1")
    before = log.render()
    before_copy = [dict(m) for m in before]
    log.append_evidence(passages=[{"id": "p1", "label": "S1", "text": "fact"}])
    log.append_assistant("a1", cited_labels=["S1"])
    for original, snapshot in zip(before, before_copy, strict=True):
        assert original == snapshot


def test_tokens_used_increases_with_appends():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    t0 = log.tokens_used()
    log.append_user("a reasonably long question about photosynthesis")
    t1 = log.tokens_used()
    assert t1 > t0


def test_headroom_decreases_as_tokens_used_increases():
    budget = Budget.scaled(6000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    h0 = log.headroom(budget)
    log.append_user("another question with a bit more text than before here")
    h1 = log.headroom(budget)
    assert h1 < h0


# ---------------------------------------------------------------------------
# Byte-prefix property (core acceptance)
# ---------------------------------------------------------------------------


def test_serialize_messages_is_deterministic_bytes():
    msgs = [{"role": "system", "content": "sys"}]
    assert isinstance(serialize_messages(msgs), bytes)
    assert serialize_messages(msgs) == serialize_messages(msgs)


def test_render_before_is_byte_prefix_of_render_after_single_append():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    log.append_user("q1")
    before = serialize_messages(log.render())
    log.append_evidence(passages=[{"id": "p1", "label": "S1", "text": "fact"}])
    after = serialize_messages(log.render())
    assert after.startswith(before)


def test_byte_prefix_property_holds_over_randomized_append_sequence():
    """Property-style test: 200 random appends without eviction; at every
    step, the previous render's serialization must be a strict byte
    prefix of the current render's serialization."""
    rng = random.Random(0)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("You are an offline tutor.")

    prev_bytes = serialize_messages(log.render())
    evidence_ids = []

    for i in range(200):
        action = rng.choice(["user_evidence_assistant", "user_calc_assistant"])
        question = f"question number {i} about topic {rng.randint(0, 9)}"
        log.append_user(question)

        if action == "user_evidence_assistant":
            pid = f"p{rng.randint(0, 20)}"
            label = f"S{len(evidence_ids) + 1}" if pid not in evidence_ids else "S1"
            if pid not in evidence_ids:
                evidence_ids.append(pid)
            log.append_evidence(
                passages=[{"id": pid, "label": label, "text": f"fact text for {pid} " * 3}]
            )
            log.append_assistant(f"answer {i} citing {label}", cited_labels=[label])
        else:
            log.append_calc_result(expression=f"{i}+1", result=str(i + 1))
            log.append_assistant(f"the calc result for turn {i} is {i + 1}", cited_labels=[])

        cur_bytes = serialize_messages(log.render())
        assert cur_bytes.startswith(prev_bytes), f"prefix property broke at step {i}"
        assert cur_bytes != prev_bytes
        prev_bytes = cur_bytes


# ---------------------------------------------------------------------------
# Head eviction
# ---------------------------------------------------------------------------


def _fill_log_until_over_budget(log: PromptLog, budget: Budget, n_turns: int = 30):
    for i in range(n_turns):
        log.append_user(f"question {i} " * 10)
        log.append_evidence(
            passages=[{"id": f"p{i}", "label": f"S{i}", "text": f"evidence text {i} " * 20}]
        )
        log.append_assistant(f"answer {i} citing S{i}" * 3, cited_labels=[f"S{i}"])


def test_evict_returns_none_when_under_budget():
    budget = Budget.scaled(32768)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    log.append_user("q1")
    log.append_assistant("a1", cited_labels=[])
    assert log.evict(budget) is None


def test_evict_removes_oldest_turns_first_and_never_the_system_entry():
    budget = Budget.scaled(6000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_until_over_budget(log, budget)

    event = log.evict(budget)
    assert event is not None
    assert isinstance(event, EvictionEvent)
    assert event.kind == "eviction_reprefill"
    assert event.evicted_turns >= 1
    assert event.tokens_before > event.tokens_after

    rendered = log.render()
    assert rendered[0]["role"] == "system"
    # the very first user turn (question 0) should be gone
    assert not any("question 0 " in str(m) for m in rendered)


def test_evict_protects_cited_evidence_of_retained_assistant_messages():
    budget = Budget.scaled(6000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_until_over_budget(log, budget)

    log.evict(budget)
    rendered = log.render()

    # find labels cited by any retained assistant message
    cited_labels = set()
    for m in rendered:
        if m.get("role") == "assistant":
            cited_labels.update(m.get("cited_labels", []))

    # every cited label's evidence text must still be present somewhere in the log
    for label in cited_labels:
        assert any(label in str(m) for m in rendered)


def test_evict_dedupes_evidence_by_passage_id_no_duplicate_text():
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    log.append_user("q1")
    log.append_evidence(passages=[{"id": "shared", "label": "S1", "text": "the shared fact text"}])
    log.append_assistant("a1 citing S1", cited_labels=["S1"])

    log.append_user("q2")
    # re-adding the same passage id should not duplicate the text
    log.append_evidence(passages=[{"id": "shared", "label": "S1", "text": "the shared fact text"}])
    log.append_assistant("a2 citing S1 again", cited_labels=["S1"])

    rendered = log.render()
    full_text = "".join(str(m) for m in rendered)
    assert full_text.count("the shared fact text") == 1


def test_evict_breaks_prefix_property_exactly_once_then_holds_again():
    budget = Budget.scaled(6000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_until_over_budget(log, budget)

    before_eviction = serialize_messages(log.render())
    log.evict(budget)
    after_eviction = serialize_messages(log.render())

    # the eviction is a head edit: prefix property is intentionally broken here
    assert not after_eviction.startswith(before_eviction)

    # but resumes holding for subsequent appends
    post_evict_snapshot = serialize_messages(log.render())
    log.append_user("a fresh question after eviction")
    log.append_assistant("a fresh answer", cited_labels=[])
    latest = serialize_messages(log.render())
    assert latest.startswith(post_evict_snapshot)


def test_evict_never_removes_system_even_under_extreme_pressure():
    budget = Budget.scaled(6000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_until_over_budget(log, budget, n_turns=60)

    for _ in range(5):
        log.evict(budget)

    rendered = log.render()
    assert rendered[0]["role"] == "system"
    assert sum(1 for m in rendered if m["role"] == "system") == 1


# ---------------------------------------------------------------------------
# Newest question + newest packet overflow (2,500 token budget)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Two-stage eviction: uncited evidence text dropped before whole turns
# ---------------------------------------------------------------------------


def _fill_log_with_uncited_evidence(log: PromptLog, n_turns: int = 30):
    """Every turn's evidence is never cited by that turn's own assistant
    reply (unlike ``_fill_log_until_over_budget``), so stage 1 should be
    able to reclaim space without evicting any whole turn."""
    for i in range(n_turns):
        log.append_user(f"question {i} " * 10)
        log.append_evidence(
            passages=[{"id": f"p{i}", "label": f"S{i}", "text": f"evidence text {i} " * 20}]
        )
        log.append_assistant(f"answer {i} with no citation at all" * 3, cited_labels=[])


def test_stage1_alone_stubs_uncited_evidence_oldest_first_no_turn_evicted():
    budget = Budget.scaled(7000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_with_uncited_evidence(log, n_turns=30)

    event = log.evict(budget)
    assert event is not None
    assert event.evicted_turns == 0
    assert event.dropped_uncited_passages > 0
    assert event.dropped_uncited_tokens > 0

    rendered = log.render()
    assert rendered[0]["role"] == "system"
    # the very first turn's question is still present (no whole turn evicted)
    assert any("question 0 " in str(m) for m in rendered)
    # its evidence text, however, has been stubbed out
    assert "evidence text 0 " not in str(rendered)
    assert any("dropped to save space" in str(m) for m in rendered)


def test_stage1_never_drops_evidence_cited_by_a_retained_assistant_message():
    budget = Budget.scaled(7000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    # First turn's evidence IS cited; the rest are not.
    log.append_user("question 0 " * 10)
    log.append_evidence(passages=[{"id": "p0", "label": "S0", "text": "evidence text 0 " * 20}])
    log.append_assistant("answer 0 citing S0" * 3, cited_labels=["S0"])
    _fill_log_with_uncited_evidence(log, n_turns=29)

    log.evict(budget)
    rendered = log.render()
    assert "evidence text 0 " in str(rendered)


def test_stage2_still_fires_when_stage1_is_not_enough():
    budget = Budget.scaled(4200)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_with_uncited_evidence(log, n_turns=30)

    event = log.evict(budget)
    assert event is not None
    # tiny budget: dropping evidence text alone can't be enough, whole
    # turns must also go.
    assert event.evicted_turns >= 1
    assert log.tokens_used() <= budget.system + budget.history


def test_stage1_keeps_log_valid_and_prefix_breaks_exactly_once():
    budget = Budget.scaled(7000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_with_uncited_evidence(log, n_turns=30)

    before_eviction = serialize_messages(log.render())
    log.evict(budget)
    log.validate()  # still a replayable message list
    after_eviction = serialize_messages(log.render())
    assert not after_eviction.startswith(before_eviction)

    post_evict_snapshot = serialize_messages(log.render())
    log.append_user("a fresh question after stage-1 eviction")
    log.append_assistant("a fresh answer", cited_labels=[])
    latest = serialize_messages(log.render())
    assert latest.startswith(post_evict_snapshot)


def test_stubbed_passage_is_resent_in_full_under_a_new_label_on_re_retrieval():
    budget = Budget.scaled(7000)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")
    _fill_log_with_uncited_evidence(log, n_turns=30)
    log.evict(budget)

    rendered_before = log.render()
    assert "evidence text 0 " not in str(rendered_before)

    # Research returns passage p0 again; the host assigns it a new label.
    log.append_user("a follow-up question that re-triggers research on p0")
    log.append_evidence(
        passages=[{"id": "p0", "label": "S_new", "text": "evidence text 0 " * 20}]
    )
    log.append_assistant("answer citing the re-sent passage", cited_labels=["S_new"])

    rendered_after = log.render()
    full_text = "".join(str(m) for m in rendered_after)
    # re-sent in full under the new label
    assert full_text.count("evidence text 0 ") >= 1
    # the old label is not resolvable as a citation target (never re-sent
    # under S0 -- the host must flag S0 as unresolved if the model cites it)
    assert not any(
        m.get("passages") and any(p["label"] == "S0" for p in m["passages"])
        for m in rendered_after
    )


def test_newest_turn_over_budget_raises_prompt_overflow():
    budget = Budget.scaled(32768)
    log = PromptLog(count_tokens=_count_tokens)
    log.append_system("sys")

    huge_question = "word " * 4000  # far more than 2500 tokens at ~1/4 chars
    huge_packet_text = "evidence " * 4000

    with pytest.raises(PromptOverflow):
        log.append_user(huge_question)
        log.append_evidence(
            passages=[{"id": "big", "label": "S1", "text": huge_packet_text}],
            budget=budget,
        )
