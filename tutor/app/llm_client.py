"""HTTP client for the llama.cpp `llama-server` used by the offline tutor.

This module treats the server as an opaque HTTP black box: it depends only
on the documented endpoints (``/health``, ``/tokenize``, ``/apply-template``,
``/v1/chat/completions`` with SSE streaming) and on the response *fields*
observed and recorded in docs/toolcall_verification.md. It never branches on
model name or backend identity.

Only the Python standard library is used: ``http.client``/``urllib`` for
transport, ``json`` for (de)serialization, and ``dataclasses``/``threading``
for the small amount of client-side state.

Streaming and cancellation
---------------------------
``stream_chat`` opens one HTTP connection per call and reads the
``text/event-stream`` response body in small chunks, polling the raw
socket with ``select`` between reads so cancellation can be observed
promptly without ever performing a read call that itself times out (which
would make CPython refuse to read from that socket file object again).
Bytes are decoded with an incremental UTF-8 decoder (SSE frames, and
even individual multi-byte UTF-8 sequences, may be split across TCP writes
or reads). A caller-supplied ``threading.Event`` is polled between reads; a
frame boundary is never assumed to align with a socket read boundary. When
cancellation is observed, the connection is closed immediately (no attempt
to drain the remainder of the response), and a single ``done`` event with
``finish_reason="cancelled"`` is yielded.

Field normalization
--------------------
The server reports cached prompt-cache tokens as either a top-level
``usage.cached_tokens`` (observed with some request shapes and in these
unit tests' fake server) or nested under
``usage.prompt_tokens_details.cached_tokens`` (observed live in
docs/toolcall_verification.md). ``stream_chat`` normalizes both into a
single ``cached_tokens`` key on the ``usage`` dict of the ``done`` event
when either form is present.
"""

from __future__ import annotations

import codecs
import http.client
import json
import select
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from urllib.parse import urlsplit

_POLL_TIMEOUT_S = 0.05
# Read one byte at a time: HTTPResponse.read(amt) blocks until *amt* bytes
# have arrived (it is not a "read whatever is available now" call), which
# would defeat prompt cancellation and prompt readability polling for
# small, slowly-trickling SSE frames. A byte at a time is cheap at SSE
# token volumes and works uniformly whether or not the response uses
# chunked transfer-encoding.
_READ_CHUNK = 1


class LlamaError(Exception):
    """Raised by non-streaming LlamaClient calls on connection/server errors."""


@dataclass(frozen=True)
class StreamEvent:
    """One event from :meth:`LlamaClient.stream_chat`.

    ``kind`` is one of ``"token"``, ``"tool_call"``, ``"done"``, or
    ``"error"``. Fields not relevant to a given kind are ``None``.
    """

    kind: str
    text: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    id: str | None = None
    name: str | None = None
    arguments_json: str | None = None
    error: str | None = None
    message: str | None = None


class LlamaClient:
    """Minimal client for the llama.cpp server's chat/completions API."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        parts = urlsplit(base_url)
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port or 80
        self._timeout_s = timeout_s

    def _connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout_s)

    def health(self) -> bool:
        try:
            conn = self._connect()
            try:
                conn.request("GET", "/health")
                resp = conn.getresponse()
                status = resp.status
                return status == 200
            finally:
                conn.close()
        except OSError:
            return False

    def _post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        try:
            conn = self._connect()
            try:
                conn.request(
                    "POST",
                    path,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                    },
                )
                resp = conn.getresponse()
                if resp.status != 200:
                    conn.close()
                    raise LlamaError(f"{path} returned HTTP {resp.status}")
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
            finally:
                conn.close()
        except OSError as exc:
            raise LlamaError(f"failed to reach llama-server at {path}: {exc}") from exc

    def tokenize(self, text: str) -> list[int]:
        result = self._post_json("/tokenize", {"content": text})
        return list(result.get("tokens", []))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def apply_template(
        self, messages: Iterable[dict], tools: list[dict] | None = None
    ) -> str:
        payload: dict = {"messages": list(messages)}
        if tools is not None:
            payload["tools"] = tools
        result = self._post_json("/apply-template", payload)
        return str(result.get("prompt", ""))

    def stream_chat(
        self,
        messages: Iterable[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        # `stream_options.include_usage` is required to get a `usage` chunk
        # at all in streaming mode on this server; without it, only the
        # llama.cpp-specific `timings` object is sent (see
        # docs/llm_client.md). Both `usage` and `timings` are still just
        # extra fields on an otherwise standard SSE chunk.
        payload: dict = {
            "messages": list(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools is not None:
            payload["tools"] = tools
        body = json.dumps(payload).encode("utf-8")

        try:
            conn = self._connect()
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            # Capture the raw socket now: HTTPConnection may null out
            # ``conn.sock`` once getresponse() sees the server intends to
            # close the connection, but this reference stays valid.
            sock = conn.sock
            resp = conn.getresponse()
        except OSError as exc:
            yield StreamEvent(kind="error", error=str(exc))
            return

        if resp.status != 200:
            conn.close()
            yield StreamEvent(kind="error", error=f"HTTP {resp.status}")
            return

        try:
            yield from self._consume_stream(resp, sock, cancel)
        finally:
            conn.close()

    def _consume_stream(
        self,
        resp: http.client.HTTPResponse,
        sock,
        cancel: threading.Event | None,
    ) -> Iterator[StreamEvent]:
        # Poll readability with `select` rather than a per-call socket
        # timeout: once a socket read times out, CPython marks its file
        # object as poisoned ("cannot read from timed out object") and it
        # can no longer be read even after data arrives. `select` lets us
        # wait briefly for data (to check `cancel` between waits) without
        # ever performing a blocking read that can time out.
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        tool_calls: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: dict = {}
        cancelled = False
        done_marker_seen = False

        while not done_marker_seen:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            if sock is not None:
                try:
                    readable, _, _ = select.select([sock], [], [], _POLL_TIMEOUT_S)
                except OSError:
                    break
                if not readable:
                    continue
            try:
                chunk = resp.read(_READ_CHUNK)
            except OSError:
                break
            if not chunk:
                break

            buffer += decoder.decode(chunk)
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                for raw_line in frame.split("\n"):
                    line = raw_line.strip("\r")
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        done_marker_seen = True
                        continue
                    obj = json.loads(data)
                    choices = obj.get("choices") or [{}]
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    content = delta.get("content")
                    if content:
                        yield StreamEvent(kind="token", text=content)

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        func = tc.get("function") or {}
                        if func.get("name"):
                            slot["name"] = func["name"]
                        if func.get("arguments"):
                            slot["arguments"] += func["arguments"]

                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr
                    if obj.get("usage") is not None:
                        usage = obj["usage"]
                if done_marker_seen:
                    break

        final_finish_reason = "cancelled" if cancelled else (finish_reason or "stop")

        for idx in sorted(tool_calls):
            slot = tool_calls[idx]
            yield StreamEvent(
                kind="tool_call",
                id=slot["id"],
                name=slot["name"],
                arguments_json=slot["arguments"],
            )

        normalized_usage = dict(usage or {})
        if "cached_tokens" not in normalized_usage:
            details = normalized_usage.get("prompt_tokens_details")
            if isinstance(details, dict) and "cached_tokens" in details:
                normalized_usage["cached_tokens"] = details["cached_tokens"]

        yield StreamEvent(kind="done", finish_reason=final_finish_reason, usage=normalized_usage)
