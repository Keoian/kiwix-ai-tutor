"""Live integration tests against the real dev llama-server.

These require a running llama-server per config/dev.toml (see
docs/toolcall_verification.md for measured server behavior). They are
skipped automatically when the server's /health endpoint is unreachable, so
they never block offline/CI runs. All prompts are kept tiny (< 600 tokens)
because large prefill is pathologically slow on the dev GPU.

Run explicitly with: python -m pytest tests/test_llm_live.py -m integration
"""

from __future__ import annotations

import threading
import time

import pytest

from tutor.app.llm_client import LlamaClient
from tutor.settings import load_config
from tutor.tools.schemas import TOOLS, validate_tool_call

pytestmark = pytest.mark.integration

_CONFIG_PATHS = ["config/dev.toml", "config/dev.granite.toml"]


def _make_client(config_path: str) -> LlamaClient:
    config = load_config(config_path)
    return LlamaClient(config.server.base_url, timeout_s=30)


def _skip_unless_healthy(client: LlamaClient):
    if not client.health():
        pytest.skip("llama-server /health unreachable; skipping live integration test")


@pytest.fixture(params=_CONFIG_PATHS)
def live_client(request):
    client = _make_client(request.param)
    _skip_unless_healthy(client)
    return client


def test_stream_one_sentence_answer_yields_tokens_and_done(live_client):
    messages = [
        {
            "role": "system",
            "content": "You are a concise tutor. Answer in exactly one short sentence.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]
    events = list(
        live_client.stream_chat(messages, max_tokens=32, temperature=0.0)
    )
    token_events = [e for e in events if e.kind == "token"]
    done_events = [e for e in events if e.kind == "done"]

    assert len(token_events) > 0
    assert "".join(e.text for e in token_events).strip() != ""
    assert len(done_events) == 1
    assert done_events[0].finish_reason is not None


def test_calc_demanding_prompt_yields_valid_tool_call(live_client):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tutor. Always call the calc tool before stating any "
                "numeric result beyond single-digit arithmetic."
            ),
        },
        {"role": "user", "content": "What is 47 times 89?"},
    ]
    events = list(
        live_client.stream_chat(
            messages, tools=TOOLS, max_tokens=64, temperature=0.0
        )
    )
    tool_events = [e for e in events if e.kind == "tool_call"]

    assert len(tool_events) >= 1, "expected model to call a tool for a multi-digit multiplication"
    call = tool_events[0]
    result = validate_tool_call(call.name, call.arguments_json)
    assert result.ok, f"tool call failed schema validation: {result.error}"
    assert call.name == "calc"


def test_prompt_cache_second_request_reports_high_cached_tokens(live_client):
    base_messages = [
        {
            "role": "system",
            "content": (
                "You are a concise offline tutor helping a student learn about "
                "basic astronomy. Keep every answer to one short sentence and "
                "end with a brief check-for-understanding question so the "
                "student can confirm they followed along."
            ),
        },
        {
            "role": "user",
            "content": (
                "Can you briefly explain why the Moon appears to change shape "
                "over the course of a month, in simple terms a beginner would "
                "understand?"
            ),
        },
    ]

    first_events = list(
        live_client.stream_chat(base_messages, max_tokens=48, temperature=0.0)
    )
    first_done = next(e for e in first_events if e.kind == "done")
    first_prompt_tokens = first_done.usage.get("prompt_tokens")
    assert first_prompt_tokens is not None and first_prompt_tokens > 0

    first_answer = "".join(e.text for e in first_events if e.kind == "token")
    followup_question = "Thanks, one more short question: is the far side always dark?"
    followup_messages = base_messages + [
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": followup_question},
    ]

    second_events = list(
        live_client.stream_chat(followup_messages, max_tokens=48, temperature=0.0)
    )
    second_done = next(e for e in second_events if e.kind == "done")
    cached_tokens = second_done.usage.get("cached_tokens")

    assert cached_tokens is not None, (
        "server did not report cached_tokens; check field name in "
        "docs/toolcall_verification.md"
    )
    assert cached_tokens >= 0.8 * first_prompt_tokens


def test_cancel_mid_stream_returns_within_two_seconds(live_client):
    messages = [
        {"role": "user", "content": "Write a long, detailed essay about the water cycle."}
    ]
    cancel = threading.Event()

    def _cancel_soon():
        time.sleep(0.3)
        cancel.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()

    start = time.monotonic()
    events = list(
        live_client.stream_chat(
            messages, max_tokens=500, temperature=0.0, cancel=cancel
        )
    )
    elapsed = time.monotonic() - start

    assert elapsed < 2.0
    done_events = [e for e in events if e.kind == "done"]
    assert len(done_events) == 1
    assert done_events[0].finish_reason == "cancelled"
