# Tool-call verification (WP eval)

Measured against the live dev llama-server (`config/dev.toml`, Bonsai-8B-Q1_0,
`--jinja` on) on 2026-09-19. Script: `eval/questions/toolcall_script.json`
(20 prompts). Command:

```
python -m eval.toolcall_harness --config config/dev.toml --out docs/toolcall_results.md
```

Run once, as instructed; not re-tuned. Raw responses: `docs/toolcall_results.json`.
Full per-row table: `docs/toolcall_results.md`.

## Results (measured, single run, n=20)

| parsed | malformed | prose-shaped | no-call | errors | correct decisions | wrong tool |
|---|---|---|---|---|---|---|
| 10 | 0 | 0 | 10 | 0 | 20 | 0 |

- malformed_rate = (malformed + prose_call) / total = 0/20 = **0.00%**
- All 5 research-expecting prompts called `research`; all 5 calc-expecting prompts
  called `calc`; all 10 no-call prompts (greetings, thanks, "explain more simply",
  opinion/meta, single-digit arithmetic, rephrase requests) correctly produced no
  tool call.

## Decision

Per plan §9 policy (malformed_rate > 2% -> grammar-constrained path;
otherwise native `--jinja` parsing): **malformed_rate = 0% <= 2%, so the
native `--jinja` chat-template tool-call parsing path is chosen.** No
grammar/JSON-schema constraint is required for this build/model at this
sample size. (Caveat: n=20 is small; if malformed calls appear later at
scale, re-run this harness before deciding to switch.)

If the grammar path were needed, it would be invoked via llama-server's
built-in constrained decoding flags (measured from `llama-server.exe --help`):
`-j/--json-schema SCHEMA` or `-jf/--json-schema-file FILE` to constrain
generation to a specific tool's argument schema, or `--grammar`/`--grammar-file`
for a hand-written GBNF grammar covering the `<tool_call>{"name":...,
"arguments":...}</tool_call>` envelope. This would need per-call schema
selection (research vs calc) rather than a single static grammar for the
whole tools array.

## Other measured items

**`/tokenize` and `/detokenize` round-trip** (measured): `POST /tokenize`
with `{"content": "Hello world, this is a test."}` returned
`{"tokens":[9707,1879,11,419,374,264,1273,13]}`; `POST /detokenize` with
those tokens returned `{"content":"Hello world, this is a test."}` —
exact round-trip.

**`/apply-template`** (measured): without `tools`, the returned prompt is a
plain ChatML string ending `...<|im_start|>assistant\n<think>\n\n</think>\n\n`
(an already-closed, empty think block — i.e. the template pre-empties
reasoning). With `tools` passed, the system prompt is expanded with a
`# Tools` section, an XML `<tools>...</tools>` block containing the JSON
function schema(s), and instructions to reply with
`<tool_call>{"name": ..., "arguments": ...}</tool_call>`. So `tools` measurably
changes the rendered prompt (adds ~a paragraph of tool-use instructions plus
the schema serialization); the assistant-priming suffix (empty `<think>`
block) is unchanged.

**Thinking off by default** (measured): the `/apply-template` output already
contains a closed, empty `<think></think>` block before generation starts,
and live chat-completion responses (e.g. "What is the capital of France?")
contained plain `content` with no `reasoning_content` field and no `<think>`
tags in the text, and likewise no reasoning content appeared in the 20-prompt
harness run. Thinking is off by default, consistent with plan expectations.

**Reasoning-budget flags** (measured from `llama-server.exe --help`, read-only):
the Bonsai fork build exposes a full reasoning-control surface:
`--reasoning-format FORMAT` (e.g. `deepseek`, populates
`message.reasoning_content`), `-rea/--reasoning [on|off|auto]`,
`--reasoning-effort LEVEL`, `--reasoning-budget N` (token budget for
thinking; `-1` unrestricted, `0` immediate end), `--reasoning-budget-message
MESSAGE`, and `--reasoning-preserve/--no-reasoning-preserve`. None of these
are passed in `config/dev.toml` today, which is consistent with the observed
default-off behavior above.

**Prompt-cache evidence** (measured): sending the identical request
(`"What is the capital of France?"`, temperature 0) twice in a row:
- Request 1: `usage.prompt_tokens_details.cached_tokens = 1`,
  `timings.cache_n = 1`, `timings.prompt_n = 18`.
- Request 2 (identical payload): `cached_tokens = 18`, `cache_n = 18`,
  `prompt_n = 1`, and `prompt_ms` dropped from ~207ms to ~30ms.

This confirms the server is reusing the KV cache for the repeated prompt
prefix (`slots` is enabled in `config/dev.toml`).

## Measured vs inferred

- Measured: harness run counts/table above, malformed_rate, tokenize/detokenize
  round-trip, `/apply-template` prompt shape with and without `tools`, absence
  of `reasoning_content`/`<think>` content in live responses, `--help` flag
  list, and the two-request prompt-cache comparison.
- Inferred: that 0% malformed will hold at larger sample sizes or with more
  adversarial prompts; that no reasoning-budget flag is needed for this model
  size/latency target (not exercised — only confirmed the flags exist).

## Server status

`llama-server` was already healthy at `http://127.0.0.1:8080/health` at the
start and was left running (not stopped) at the end.
