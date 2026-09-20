# M0 report — dev runtime verified (2026-09-19)

| Gate item | Result | Evidence |
|---|---|---|
| `C:\git\bonsai` server reachable, unmodified | pass | `docs/dev_runtime.md`, build `b10716-ee2aeae98` |
| `--jinja` tool calls, `research`/`calc` schemas, ≤ 2% malformed on 20 scripted requests | pass: 10 parsed, 0 malformed, 0 prose-shaped, 10 no-call, 20/20 correct decisions, 0 wrong tool | `docs/toolcall_verification.md`, raw `docs/toolcall_results.json` |
| Grammar decision | native `--jinja` parsing; grammar path not needed. `--grammar`, `--json-schema` exist in this build as fallback | same |
| `/tokenize`, `/apply-template` | both work; tokenize/detokenize round-trips exactly; `tools` changes the template | same |
| Thinking off by default | confirmed (closed empty `<think></think>` in template; no `reasoning_content`) ; `--reasoning-budget N` exists | same |
| Baseline pp/tg | pre-measured 175 t/s / 31.9 tok/s; this WP re-measured decode 30.1 tok/s on one request. llama-bench not re-run | `docs/dev_runtime.md` |

Caveat: n=20, single run, scripted prompts that are fairly explicit. 0/20 malformed bounds the true
rate only loosely (95% upper bound ≈ 14%). The agent loop (WP-C2) still validates every call and
never parses prose; M6 repeats this on the Dell for Q1_0 and Q2_0.
