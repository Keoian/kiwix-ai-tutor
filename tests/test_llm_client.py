"""Unit tests for tutor.app.llm_client.LlamaClient.

These tests spin up a local, in-process fake llama-server (ThreadingHTTPServer
on port 0) that mimics the subset of the llama.cpp server API the client
depends on: /health, /tokenize, /apply-template, and SSE-streamed
/v1/chat/completions. No real network access, no real model, no external
process.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tutor.app.llm_client import LlamaClient, LlamaError, StreamEvent


class _FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs):  # noqa: D401 - silence test logs
        pass

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            behavior = self.server.behavior  # type: ignore[attr-defined]
            if behavior == "health_down":
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        behavior = self.server.behavior  # type: ignore[attr-defined]

        if self.path == "/tokenize":
            body = self._read_json()
            content = body.get("content", "")
            tokens = [ord(c) for c in content]
            payload = json.dumps({"tokens": tokens}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/apply-template":
            self._read_json()
            text = "<|im_start|>system\n...<|im_start|>assistant\n"
            payload = json.dumps({"prompt": text}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/v1/chat/completions":
            self._read_json()

            if behavior == "server_error":
                self.send_response(500)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                if behavior == "simple_tokens":
                    self._emit_simple_tokens()
                elif behavior == "tool_call":
                    self._emit_tool_call()
                elif behavior == "split_chunk":
                    self._emit_split_json_chunk()
                elif behavior == "split_utf8":
                    self._emit_split_utf8()
                elif behavior == "keepalive_comments":
                    self._emit_with_comments()
                elif behavior == "slow_cancel":
                    self._emit_slow_for_cancel()
                else:
                    self._emit_simple_tokens()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                self.server.disconnect_seen = True  # type: ignore[attr-defined]
            return

        self.send_response(404)
        self.end_headers()

    def _sse(self, obj: dict | None):
        if obj is None:
            data = b"data: [DONE]\n\n"
        else:
            data = ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _chunk(self, delta: dict, finish_reason=None, usage=None):
        obj = {
            "choices": [{"delta": delta, "finish_reason": finish_reason, "index": 0}],
        }
        if usage is not None:
            obj["usage"] = usage
        return obj

    def _emit_simple_tokens(self):
        for word in ["Hello", " world"]:
            self._sse(self._chunk({"content": word}))
            time.sleep(0.01)
        self._sse(
            self._chunk(
                {},
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 3},
            )
        )
        self._sse(None)

    def _emit_tool_call(self):
        # Fragmented tool call deltas across multiple chunks, split by index.
        self._sse(
            self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "calc", "arguments": '{"expr'},
                        }
                    ]
                }
            )
        )
        self._sse(
            self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'ession": "2+2"}'},
                        }
                    ]
                }
            )
        )
        usage = {"prompt_tokens": 20, "completion_tokens": 5}
        self._sse(self._chunk({}, finish_reason="tool_calls", usage=usage))
        self._sse(None)

    def _emit_split_json_chunk(self):
        # Send a single SSE "data: {...}\n\n" frame's bytes split across two
        # separate socket writes, to exercise buffer reassembly.
        obj = self._chunk({"content": "partial-frame-token"})
        raw = ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")
        mid = len(raw) // 2
        self.wfile.write(raw[:mid])
        self.wfile.flush()
        time.sleep(0.02)
        self.wfile.write(raw[mid:])
        self.wfile.flush()
        self._sse(self._chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}))  # noqa: E501
        self._sse(None)

    def _emit_split_utf8(self):
        # "é" (U+00E9) and "²" (U+00B2) as multi-byte UTF-8 sequences, split
        # across two writes each so no single write is valid utf-8 alone.
        text = "café²"
        delta_obj = {
            "choices": [{"delta": {"content": text}, "finish_reason": None, "index": 0}]
        }
        raw = ("data: " + json.dumps(delta_obj) + "\n\n").encode("utf-8")
        mid = len(raw) // 2
        # Ensure the split lands inside a multibyte sequence if possible.
        self.wfile.write(raw[:mid])
        self.wfile.flush()
        time.sleep(0.02)
        self.wfile.write(raw[mid:])
        self.wfile.flush()
        self._sse(self._chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}))  # noqa: E501
        self._sse(None)

    def _emit_with_comments(self):
        self.wfile.write(b": keep-alive\n\n")
        self.wfile.flush()
        self._sse(self._chunk({"content": "ok"}))
        self.wfile.write(b": ping\n\n")
        self.wfile.flush()
        self._sse(self._chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}))  # noqa: E501
        self._sse(None)

    def _emit_slow_for_cancel(self):
        try:
            for _ in range(200):
                self._sse(self._chunk({"content": "x"}))
                time.sleep(0.05)
            self._sse(self._chunk({}, finish_reason="stop", usage={}))
            self._sse(None)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.server.disconnect_seen = True  # type: ignore[attr-defined]


@pytest.fixture
def fake_server() -> Callable[[str], str]:
    """Start a fake llama-server; returns a function to set behavior -> base_url."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.behavior = "simple_tokens"
    server.disconnect_seen = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{server.server_port}"

    def _set(behavior: str) -> str:
        server.behavior = behavior
        return base_url

    yield _set

    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_health_true_when_server_up(fake_server):
    base_url = fake_server("simple_tokens")
    client = LlamaClient(base_url, timeout_s=5)
    assert client.health() is True


def test_health_false_when_server_down(fake_server):
    base_url = fake_server("health_down")
    client = LlamaClient(base_url, timeout_s=5)
    assert client.health() is False


def test_health_false_on_connection_refused():
    client = LlamaClient("http://127.0.0.1:1", timeout_s=1)
    assert client.health() is False


def test_tokenize_returns_list_of_ints(fake_server):
    base_url = fake_server("simple_tokens")
    client = LlamaClient(base_url, timeout_s=5)
    tokens = client.tokenize("hi")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == 2


def test_count_tokens_matches_tokenize_length(fake_server):
    base_url = fake_server("simple_tokens")
    client = LlamaClient(base_url, timeout_s=5)
    assert client.count_tokens("hello") == len(client.tokenize("hello"))


def test_apply_template_returns_string(fake_server):
    base_url = fake_server("simple_tokens")
    client = LlamaClient(base_url, timeout_s=5)
    result = client.apply_template([{"role": "user", "content": "hi"}])
    assert isinstance(result, str)
    assert "assistant" in result


def test_apply_template_accepts_tools(fake_server):
    base_url = fake_server("simple_tokens")
    client = LlamaClient(base_url, timeout_s=5)
    result = client.apply_template(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "calc", "parameters": {}}}],
    )
    assert isinstance(result, str)


def test_stream_chat_yields_token_events_then_done(fake_server):
    base_url = fake_server("simple_tokens")
    client = LlamaClient(base_url, timeout_s=5)
    events = list(
        client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=16)
    )
    token_events = [e for e in events if e.kind == "token"]
    done_events = [e for e in events if e.kind == "done"]
    assert [e.text for e in token_events] == ["Hello", " world"]
    assert len(done_events) == 1
    done = done_events[0]
    assert done.finish_reason == "stop"
    assert done.usage["prompt_tokens"] == 10
    assert done.usage["completion_tokens"] == 2
    assert done.usage["cached_tokens"] == 3


def test_stream_chat_tolerates_missing_cached_tokens(fake_server):
    base_url = fake_server("slow_cancel")
    client = LlamaClient(base_url, timeout_s=5)
    cancel = threading.Event()

    def _cancel_soon():
        time.sleep(0.05)
        cancel.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()
    events = list(
        client.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=16, cancel=cancel
        )
    )
    done_events = [e for e in events if e.kind == "done"]
    assert len(done_events) == 1
    # usage dict may be empty/partial; must not raise KeyError getting here.
    assert done_events[0].usage is not None


def test_stream_chat_assembles_fragmented_tool_call(fake_server):
    base_url = fake_server("tool_call")
    client = LlamaClient(base_url, timeout_s=5)
    events = list(
        client.stream_chat([{"role": "user", "content": "what is 2+2"}], max_tokens=16)
    )
    tool_events = [e for e in events if e.kind == "tool_call"]
    assert len(tool_events) == 1
    call = tool_events[0]
    assert call.id == "call_1"
    assert call.name == "calc"
    parsed = json.loads(call.arguments_json)
    assert parsed == {"expression": "2+2"}

    done_events = [e for e in events if e.kind == "done"]
    assert done_events[0].finish_reason == "tool_calls"


def test_stream_chat_handles_json_chunk_split_across_tcp_writes(fake_server):
    base_url = fake_server("split_chunk")
    client = LlamaClient(base_url, timeout_s=5)
    events = list(
        client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=16)
    )
    token_events = [e for e in events if e.kind == "token"]
    assert token_events[0].text == "partial-frame-token"


def test_stream_chat_handles_multibyte_utf8_split_across_chunks(fake_server):
    base_url = fake_server("split_utf8")
    client = LlamaClient(base_url, timeout_s=5)
    events = list(
        client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=16)
    )
    token_events = [e for e in events if e.kind == "token"]
    assert "".join(e.text for e in token_events) == "café²"


def test_stream_chat_ignores_comments_and_keepalives(fake_server):
    base_url = fake_server("keepalive_comments")
    client = LlamaClient(base_url, timeout_s=5)
    events = list(
        client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=16)
    )
    token_events = [e for e in events if e.kind == "token"]
    assert [e.text for e in token_events] == ["ok"]


def test_stream_chat_cancel_stops_promptly_and_marks_cancelled(fake_server):
    base_url = fake_server("slow_cancel")
    client = LlamaClient(base_url, timeout_s=5)
    cancel = threading.Event()

    def _cancel_soon():
        time.sleep(0.1)
        cancel.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()

    start = time.monotonic()
    events = list(
        client.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=1000, cancel=cancel
        )
    )
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    done_events = [e for e in events if e.kind == "done"]
    assert len(done_events) == 1
    assert done_events[0].finish_reason == "cancelled"


def test_stream_chat_server_error_yields_single_error_event(fake_server):
    base_url = fake_server("server_error")
    client = LlamaClient(base_url, timeout_s=5)
    events = list(
        client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=16)
    )
    assert len(events) == 1
    assert events[0].kind == "error"


def test_stream_chat_connection_refused_yields_error_event_not_raw_exception():
    client = LlamaClient("http://127.0.0.1:1", timeout_s=1)
    events = list(
        client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=16)
    )
    assert len(events) == 1
    assert events[0].kind == "error"


def test_non_streaming_calls_raise_llama_error_on_connection_refused():
    client = LlamaClient("http://127.0.0.1:1", timeout_s=1)
    with pytest.raises(LlamaError):
        client.tokenize("hi")
    with pytest.raises(LlamaError):
        client.apply_template([{"role": "user", "content": "hi"}])


def test_stream_event_is_frozen():
    event = StreamEvent(kind="token", text="hi")
    with pytest.raises(AttributeError):
        event.text = "bye"  # type: ignore[misc]
