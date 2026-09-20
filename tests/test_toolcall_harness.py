"""Contract tests for eval/toolcall_harness.py (RED: module does not exist yet)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.toolcall_harness import (
    Outcome,
    RequestResult,
    ScriptedRequest,
    load_script,
    render_markdown,
    run,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "eval" / "questions" / "toolcall_script.json"


def _valid_research_call():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "research", "arguments": json.dumps({"query": "photosynthesis"})},
    }


def _valid_calc_call():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "calc", "arguments": json.dumps({"expression": "2+2"})},
    }


def _malformed_call_bad_json():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "calc", "arguments": "{not json"},
    }


def _malformed_call_unknown_tool():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"query": "x"})},
    }


# ---------------------------------------------------------------------------
# classify_response


def test_classify_parsed_call_research():
    message = {"content": None, "tool_calls": [_valid_research_call()]}
    assert Outcome.PARSED_CALL is not None  # ensure enum member exists
    from eval.toolcall_harness import classify_response

    assert classify_response(message) == Outcome.PARSED_CALL


def test_classify_parsed_call_calc():
    from eval.toolcall_harness import classify_response

    message = {"content": None, "tool_calls": [_valid_calc_call()]}
    assert classify_response(message) == Outcome.PARSED_CALL


def test_classify_malformed_bad_json_arguments():
    from eval.toolcall_harness import classify_response

    message = {"content": None, "tool_calls": [_malformed_call_bad_json()]}
    assert classify_response(message) == Outcome.MALFORMED_CALL


def test_classify_malformed_unknown_tool():
    from eval.toolcall_harness import classify_response

    message = {"content": None, "tool_calls": [_malformed_call_unknown_tool()]}
    assert classify_response(message) == Outcome.MALFORMED_CALL


def test_classify_malformed_one_bad_one_good():
    from eval.toolcall_harness import classify_response

    message = {
        "content": None,
        "tool_calls": [_valid_research_call(), _malformed_call_unknown_tool()],
    }
    assert classify_response(message) == Outcome.MALFORMED_CALL


def test_classify_content_none_with_tool_calls_is_parsed():
    from eval.toolcall_harness import classify_response

    message = {"content": None, "tool_calls": [_valid_calc_call()]}
    assert classify_response(message) == Outcome.PARSED_CALL


def test_classify_prose_call_xml_style_tag():
    from eval.toolcall_harness import classify_response

    message = {
        "content": '<tool_call>{"name": "research", "arguments": {"query": "x"}}</tool_call>',
        "tool_calls": None,
    }
    assert classify_response(message) == Outcome.PROSE_CALL


def test_classify_prose_call_json_shaped_text():
    from eval.toolcall_harness import classify_response

    message = {
        "content": 'I will call {"name": "calc", "arguments": {"expression": "2+2"}} now.',
        "tool_calls": None,
    }
    assert classify_response(message) == Outcome.PROSE_CALL


def test_classify_prose_call_function_syntax_research():
    from eval.toolcall_harness import classify_response

    message = {"content": 'Let me do research(query="mitosis")', "tool_calls": None}
    assert classify_response(message) == Outcome.PROSE_CALL


def test_classify_prose_call_function_syntax_calc():
    from eval.toolcall_harness import classify_response

    message = {"content": "calc(expression='2+2') should give us the answer", "tool_calls": None}
    assert classify_response(message) == Outcome.PROSE_CALL


def test_classify_prose_call_fenced_json_block():
    from eval.toolcall_harness import classify_response

    content = '```json\n{"name": "research", "arguments": {"query": "x"}}\n```'
    message = {"content": content, "tool_calls": None}
    assert classify_response(message) == Outcome.PROSE_CALL


def test_classify_no_call_plain_prose():
    from eval.toolcall_harness import classify_response

    message = {"content": "The mitochondria is the powerhouse of the cell.", "tool_calls": None}
    assert classify_response(message) == Outcome.NO_CALL


def test_classify_no_call_word_research_in_ordinary_prose():
    from eval.toolcall_harness import classify_response

    message = {
        "content": "Historians research primary sources before drawing conclusions.",
        "tool_calls": None,
    }
    assert classify_response(message) == Outcome.NO_CALL


def test_classify_no_call_word_calculate_in_ordinary_prose():
    from eval.toolcall_harness import classify_response

    message = {
        "content": "To calculate the area, multiply length by width.",
        "tool_calls": None,
    }
    assert classify_response(message) == Outcome.NO_CALL


def test_classify_no_call_empty_tool_calls_list_is_no_call():
    from eval.toolcall_harness import classify_response

    message = {"content": "Here is the answer.", "tool_calls": []}
    assert classify_response(message) == Outcome.NO_CALL


# ---------------------------------------------------------------------------
# ScriptedRequest / load_script


def test_scripted_request_is_frozen_dataclass():
    req = ScriptedRequest(
        id="q1", prompt="What is mitosis?", expect_call=True, expected_tool="research"
    )
    assert req.id == "q1"
    with pytest.raises(AttributeError):
        req.id = "other"


def test_load_script_reads_json_file(tmp_path):
    data = [
        {"id": "q1", "prompt": "hi", "expect_call": False, "expected_tool": None},
        {"id": "q2", "prompt": "what is 2+2", "expect_call": True, "expected_tool": "calc"},
    ]
    path = tmp_path / "script.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    requests = load_script(path)
    assert len(requests) == 2
    assert all(isinstance(r, ScriptedRequest) for r in requests)
    assert requests[0].id == "q1"
    assert requests[0].expect_call is False
    assert requests[1].expected_tool == "calc"


def test_real_toolcall_script_has_20_entries_balanced():
    assert SCRIPT_PATH.exists(), f"expected scripted questions at {SCRIPT_PATH}"
    requests = load_script(SCRIPT_PATH)
    assert len(requests) == 20

    ids = [r.id for r in requests]
    assert len(set(ids)) == 20

    expect_true = [r for r in requests if r.expect_call is True]
    expect_false = [r for r in requests if r.expect_call is False]
    assert len(expect_true) == 10
    assert len(expect_false) == 10

    for r in expect_false:
        assert r.expected_tool is None

    tool_counts: dict[str, int] = {}
    for r in expect_true:
        assert r.expected_tool in {"research", "calc"}
        tool_counts[r.expected_tool] = tool_counts.get(r.expected_tool, 0) + 1

    assert tool_counts.get("research", 0) >= 3
    assert tool_counts.get("calc", 0) >= 3


# ---------------------------------------------------------------------------
# summarize


def _req(i, expect_call=True, expected_tool="research"):
    return ScriptedRequest(
        id=f"q{i}",
        prompt=f"prompt {i}",
        expect_call=expect_call,
        expected_tool=expected_tool,
    )


def test_summarize_empty_results_raises_value_error():
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_counts_basic():
    results = [
        RequestResult(request=_req(1), outcome=Outcome.PARSED_CALL, tool_names=("research",)),
        RequestResult(
            request=_req(2, expect_call=False, expected_tool=None),
            outcome=Outcome.NO_CALL,
            tool_names=(),
        ),
    ]
    summary = summarize(results)
    assert summary.total == 2
    assert summary.parsed == 1
    assert summary.no_call == 1
    assert summary.malformed == 0
    assert summary.prose_call == 0
    assert summary.correct_decisions == 2
    assert summary.wrong_tool == 0
    assert summary.malformed_rate == 0.0
    assert summary.grammar_path_required is False


def test_summarize_malformed_rate_boundary_at_5_percent_true():
    results = []
    for i in range(19):
        results.append(
            RequestResult(request=_req(i), outcome=Outcome.PARSED_CALL, tool_names=("research",))
        )
    results.append(
        RequestResult(request=_req(19), outcome=Outcome.MALFORMED_CALL, tool_names=())
    )
    summary = summarize(results)
    assert summary.total == 20
    assert summary.malformed_rate == pytest.approx(1 / 20)
    assert summary.grammar_path_required is True


def test_summarize_malformed_rate_boundary_at_0_percent_false():
    results = [
        RequestResult(request=_req(i), outcome=Outcome.PARSED_CALL, tool_names=("research",))
        for i in range(20)
    ]
    summary = summarize(results)
    assert summary.malformed_rate == 0.0
    assert summary.grammar_path_required is False


def test_summarize_prose_call_counts_toward_malformed_rate():
    results = [
        RequestResult(request=_req(i), outcome=Outcome.PARSED_CALL, tool_names=("research",))
        for i in range(19)
    ]
    results.append(RequestResult(request=_req(19), outcome=Outcome.PROSE_CALL, tool_names=()))
    summary = summarize(results)
    assert summary.prose_call == 1
    assert summary.malformed_rate == pytest.approx(1 / 20)
    assert summary.grammar_path_required is True


def test_summarize_prose_and_malformed_count_as_incorrect_decisions():
    results = [
        RequestResult(request=_req(1), outcome=Outcome.MALFORMED_CALL, tool_names=()),
        RequestResult(request=_req(2), outcome=Outcome.PROSE_CALL, tool_names=()),
    ]
    summary = summarize(results)
    assert summary.correct_decisions == 0


def test_summarize_wrong_tool_counted():
    req = _req(1, expect_call=True, expected_tool="calc")
    result = RequestResult(request=req, outcome=Outcome.PARSED_CALL, tool_names=("research",))
    summary = summarize([result])
    assert summary.wrong_tool == 1
    assert summary.correct_decisions == 0


def test_summarize_errors_counted_and_included_in_total():
    results = [
        RequestResult(request=_req(1), outcome=Outcome.ERROR, tool_names=()),
        RequestResult(
            request=_req(2, expect_call=False, expected_tool=None),
            outcome=Outcome.NO_CALL,
            tool_names=(),
        ),
    ]
    summary = summarize(results)
    assert summary.total == 2
    assert summary.errors == 1


# ---------------------------------------------------------------------------
# render_markdown


def test_render_markdown_contains_ids_and_decision_line():
    req = _req(1, expect_call=True, expected_tool="research")
    result = RequestResult(request=req, outcome=Outcome.PARSED_CALL, tool_names=("research",))
    summary = summarize([result])
    md = render_markdown(summary, [result])
    assert "q1" in md
    assert "grammar_path_required" in md
    assert "malformed_rate" in md
    assert str(summary.total) in md


# ---------------------------------------------------------------------------
# run


def test_run_posts_expected_payload_shape():
    captured_payloads = []

    def fake_post(payload):
        captured_payloads.append(payload)
        message = {
            "content": "The mitochondria is the powerhouse of the cell.",
            "tool_calls": None,
        }
        return {"choices": [{"message": message}]}

    requests = [
        ScriptedRequest(id="q1", prompt="What is a cell?", expect_call=False, expected_tool=None)
    ]
    results = run(requests, fake_post)

    assert len(results) == 1
    assert len(captured_payloads) == 1
    payload = captured_payloads[0]

    from tutor.tools.schemas import TOOLS

    assert payload["tools"] == TOOLS
    assert payload["stream"] is False
    assert isinstance(payload["messages"], list)
    roles = [m["role"] for m in payload["messages"]]
    assert "system" in roles
    assert "user" in roles
    user_messages = [m for m in payload["messages"] if m["role"] == "user"]
    assert any("What is a cell?" in m["content"] for m in user_messages)


def test_run_records_outcome_and_tool_names_for_parsed_call():
    def fake_post(_payload):
        return {
            "choices": [
                {"message": {"content": None, "tool_calls": [_valid_research_call()]}}
            ]
        }

    requests = [
        ScriptedRequest(
            id="q1", prompt="Tell me about mitosis", expect_call=True, expected_tool="research"
        )
    ]
    results = run(requests, fake_post)
    assert results[0].outcome == Outcome.PARSED_CALL
    assert results[0].tool_names == ("research",)


def test_run_continues_after_post_raises_and_marks_error():
    def flaky_post(payload):
        user_msg = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        if "boom" in user_msg:
            raise RuntimeError("connection reset")
        return {"choices": [{"message": {"content": "fine", "tool_calls": None}}]}

    requests = [
        ScriptedRequest(id="q1", prompt="boom please", expect_call=False, expected_tool=None),
        ScriptedRequest(id="q2", prompt="a normal question", expect_call=False, expected_tool=None),
    ]
    results = run(requests, flaky_post)
    assert len(results) == 2
    assert results[0].outcome == Outcome.ERROR
    assert results[1].outcome == Outcome.NO_CALL
