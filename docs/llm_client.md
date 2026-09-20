# llm_client: LlamaClient live verification

`tutor.app.llm_client.LlamaClient` talks to the llama.cpp `llama-server` as an
opaque HTTP black box (stdlib `http.client`/`json`/`threading`/`dataclasses`
only). This records the endpoints and response fields it actually relies on,
plus live numbers observed against `config/dev.toml` on 2026-09-20.

## Endpoints used

- `GET /health` — liveness check; any non-200 or connection error -> `False`.
- `POST /tokenize` — `{"content": str}` -> `{"tokens": [int, ...]}`.
- `POST /apply-template` — `{"messages": [...], "tools"?: [...]}` ->
  `{"prompt": str}`.
- `POST /v1/chat/completions` with `"stream": true` — OpenAI-compatible SSE
  chat completions.

## Fields relied on

- Streaming chunk: `choices[0].delta.content` (token text),
  `choices[0].delta.tool_calls[].{index,id,function.{name,arguments}}`
  (fragmented tool-call deltas, assembled by `index`), `choices[0].finish_reason`.
- **`stream_options: {"include_usage": true}` must be sent in the request**,
  or this server never emits a `usage` object in streaming mode at all — only
  its own `timings` object (see below). With it set, a final chunk with
  `"choices": []` carries `usage.prompt_tokens`, `usage.completion_tokens`,
  and cached-token count under **`usage.prompt_tokens_details.cached_tokens`**
  (the OpenAI-shaped field; there is no top-level `usage.cached_tokens` from
  this server). `LlamaClient` normalizes this: if `usage.cached_tokens` is
  absent but `usage.prompt_tokens_details.cached_tokens` is present, it is
  copied up to a top-level `cached_tokens` key on the `done` event's `usage`
  dict, so callers only ever need to check one key.
- The server also sends its own `timings` object (`cache_n`, `prompt_n`,
  `prompt_ms`, etc.) on the final content-bearing chunk; `LlamaClient` does
  not depend on it, but it corroborates the cache-hit numbers below.
- `data: [DONE]` marks the end of the SSE stream.

## Byte-prefix / incremental-decode property

SSE frames (and even individual multi-byte UTF-8 sequences) can be split
across TCP writes or `socket.recv` calls. `LlamaClient` buffers decoded text
(via `codecs.getincrementaldecoder("utf-8")`, never `bytes.decode` on a raw
chunk) and only extracts complete `"\n\n"`-terminated frames from that
buffer, leaving any remainder for the next read. This is exercised by
`tests/test_llm_client.py::test_stream_chat_handles_json_chunk_split_across_tcp_writes`
and `::test_stream_chat_handles_multibyte_utf8_split_across_chunks`, which
use a fake server that deliberately splits a single SSE frame's bytes (once
mid-frame, once mid-UTF-8-multibyte-sequence) across two separate socket
writes with a delay in between.

## Cancellation

`stream_chat` polls the raw socket with `select.select(..., timeout=0.05s)`
between reads, checking the caller's `threading.Event` each iteration, and
closes the connection immediately once cancellation is observed (finish
reason `"cancelled"`). Two implementation pitfalls found and worked around:

1. `http.client.HTTPConnection.getresponse()` may null out `conn.sock` once
   it sees the server intends to close the connection (`Connection: close`),
   so the raw socket must be captured right after `conn.request()`, before
   calling `getresponse()`.
2. `HTTPResponse.read(amt)` is not a "read whatever is available" call — it
   blocks until `amt` bytes have arrived (or EOF). Using a large `amt`
   (e.g. 4096) with slowly-trickling SSE tokens meant a single `read()` call
   could block for seconds waiting to accumulate that much data, even though
   `select()` had already reported the socket readable. Reading one byte at
   a time (`_READ_CHUNK = 1`) fixes this at negligible cost for SSE token
   volumes and works the same whether or not the response uses chunked
   transfer-encoding.
3. Setting a socket-level `settimeout()` and catching `socket.timeout` in a
   read-retry loop was tried first and rejected: once a `recv()` call times
   out, CPython marks that socket file object as poisoned and raises
   `OSError: cannot read from timed out object` on the next read attempt,
   even after data has arrived. `select()` avoids this because it never
   performs a timing-out read itself.

## Live numbers observed (config/dev.toml, Bonsai-8B-Q1_0, 2026-09-20)

Two-request prompt-cache check (system + user turn, then the same history
plus a follow-up turn):

| request | prompt_tokens | cached_tokens | completion_tokens |
|---|---|---|---|
| 1st (cold) | 26 | 1 | 17 |
| 2nd (repeated prefix + follow-up) | 56 | 22 | 17 |

22/26 ≈ 85% of the first request's full prompt was served from cache on the
second request (comfortably above the 80% threshold
`tests/test_llm_live.py::test_prompt_cache_second_request_reports_high_cached_tokens`
checks), consistent with `timings.cache_n`/`prompt_n` behavior already
recorded in `docs/toolcall_verification.md`.

Cancel-mid-stream latency: cancelling ~0.3s into a 500-max-token generation
returned control in **0.33s elapsed** (well under the 2s the live test
allows and the 1s the unit test's synthetic slow-stream allows), with
`finish_reason == "cancelled"`.

## Unit vs. live test scope

`tests/test_llm_client.py` is fully offline (in-process fake server, no
network/model). `tests/test_llm_live.py` (`pytest -m integration`) requires
`config/dev.toml`'s server to be healthy and is auto-skipped otherwise; all
4 of its tests passed in this verification run (one-sentence answer, calc
tool-call schema validation, prompt-cache hit ratio, and cancel latency).
