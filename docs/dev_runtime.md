# Dev runtime notes (WP-A0)

This file records how the dev llama-server is launched and what was observed
when it was brought up for this work package, plus reference numbers
gathered previously. Sections are labelled by provenance.

**Current default**: `config/dev.toml` now runs IBM Granite 4.0 H-Tiny
(bake-off winner), not the Bonsai-8B Q1_0 example below. The launch command
and numbers below are historical, from when `dev.toml` was the Bonsai
profile; that profile is preserved at `config/dev.bonsai-q1.toml`. See
`docs/bakeoff_dev/granite_flags.md` for the Granite launch flags.

## Launch command (measured by this WP)

Produced by `Config.server.to_argv()` from `config/dev.toml`, run via
`scripts/serve_dev.ps1` (`python -m tutor.settings --argv config/dev.toml`
resolves the binary path + argv; the script just execs them):

```
C:\git\bonsai\bin\llama-server.exe
  -m C:\git\bonsai\models\Bonsai-8B-Q1_0.gguf
  --host 127.0.0.1 --port 8080
  -ngl 99 -fa on
  -c 32768
  -np 1
  -ctk q8_0 -ctv q8_0
  --temp 0.5 --top-p 0.9 --top-k 20
  --jinja --slots
```

Port: 8080. Flags: flash attention on, jinja templating on, `/slots`
endpoint enabled, KV cache q8_0/q8_0, context 32768, n_gpu_layers 99
(full offload), parallel slots 1.

## Startup log and /props (measured by this WP, 2026-09-19)

- Server build reported by `/props` and in chat responses'
  `system_fingerprint`: `b10716-ee2aeae98` — matches the pre-measured
  reference build below.
- `/health` returned `{"status":"ok"}` almost immediately; model load took
  well under 30 s on this run (log timestamps show ~2.1 s from process start
  to "model loaded").
- The startup log (verbosity 3, stderr) did **not** print an explicit
  Vulkan/device banner line at this verbosity level, so the exact backend
  string and device name were not captured directly from this run's log.
  This is inferred from `/props`/model metadata and the pre-measured
  reference (Vulkan, `-ngl 99`, this machine has one GPU: Radeon Pro 5500M)
  rather than observed in the log text.
- `/props`: `n_ctx` (slot) = 32768, `model_ftype` = "Q1_0", `n_slots` = 1,
  `chat_template_caps` shows tool-call support, ChatML-style template
  (`<|im_start|>`/`<|im_end|>`), `bos_token` = ",", `eos_token` = "<|im_end|>".
- `/v1/models`: `n_ctx` 32768, `n_ctx_train` 65536, `n_embd` 4096,
  `n_params` 8,188,548,096, `size` 1,152,704,128 bytes, `ftype` "Q1_0".

## Chat completion timing (measured by this WP)

One `/v1/chat/completions` call, `max_tokens: 64`, temperature 0.5, short
prompt ("Say hello in one short sentence."):

- prompt: 19 tokens, 205.7 ms, **92.4 t/s**
- decode: 3 tokens (model stopped early), 66.5 ms, **30.1 tok/s**

Decode tok/s is close to the pre-measured 31.9 tok/s reference below; prompt
t/s differs from the pre-measured 175 t/s figure but this run's prompt was
only 19 tokens (mostly fixed template overhead dominates at that length),
so it is not a like-for-like comparison with the reference short-context
measurement.

## Pre-measured reference data (measured 2026-09-19 before this work package)

- Model: `C:\git\bonsai\models\Bonsai-8B-Q1_0.gguf`, 1.16 GB, 8.19B params,
  SHA-256 `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54`.
- Server build `b10716-ee2aeae98` (PrismML fork, Vulkan).
- VRAM: 3,584 MiB at 32k context with q8_0 KV cache (1,016 MiB weights +
  2,448 MiB KV cache + 120 MiB compute, per `start-server-32k.ps1` header).
- Decode 31.9 tok/s and prompt 175 t/s at short context.
- Machine: i9-9980HK, Radeon Pro 5500M 8 GB, Windows 11, AMD driver
  32.0.12019.1028.
- **Flash-attention trap**: with `-fa on`, Vulkan prefill collapses with
  context depth — about 21 t/s at 4k, 11 t/s at 8k, and a 25,200-token
  prompt did not finish in 40 minutes. With `-fa off`, prefill at that same
  long-context regime was measured at 82.7 t/s, but decode throughput drops
  in exchange. Because chat replies are decode-bound, `-fa on` is the
  default here (see `config/dev.toml` and `start-server.ps1`'s own comment),
  but this means **integration tests must stay well under 8k tokens of
  prompt** or prefill time becomes impractical.

`llama-bench` (pp512 / pp2048 / tg128) was **not** re-run for this work
package: the reference numbers above already exist, and re-running
`llama-bench` is a long GPU job (the flash-attention trap above shows
prefill-heavy benchmarks can take tens of minutes at this context size),
which this work package's scope does not require.

## Ling 3.0 Tiny (2026-09-21)

Made the default dev model this session (owner: "Let's set default to
Ling") — `config/dev.toml` is now this profile; the prior default,
Granite 4.0 H-Tiny, moved to `config/dev.granite.toml` as the fallback.
Config: `config/dev.ling.toml` — inclusionAI Ling 3.0 Tiny (7.9B total /
1.3B active MoE, hybrid linear attention, MIT licence), bartowski
Q4_K_M GGUF (4.92 GB), `ctx_size` 32768, `parallel = 1`,
`cache_type_k`/`cache_type_v` q8_0, `n_gpu_layers` 99, `flash_attn` true.

**Thinking off**: added via `extra_args = ["--reasoning-budget", "0"]`,
not the JSON `--chat-template-kwargs` form. A tutor turn must not spend
hundreds of hidden reasoning tokens before answering. **PowerShell
quoting pitfall**: the JSON kwargs form (e.g.
`--chat-template-kwargs '{"thinking": false}'`) loses its quotes when
Windows PowerShell 5.1 passes it through to the native `llama-server.exe`
(via `scripts/serve_dev.ps1`), and the server then refuses to start. The
flag form (`--reasoning-budget 0`) sidesteps this entirely and is what
`config/dev.ling.toml` uses.

**Measured this session, this dev machine only** (not the delivery Dell
laptop, GTX 1060 Max-Q 6 GB — unverified there):

- ~4.63 GiB model share of VRAM at 32K context.
- ~25 tok/s writing (generation).
- ~40-200 tok/s reading (prompt processing; varies with prompt depth).
- Mixed K/V cache types were refused by this build when tried; not
  independently re-verified with numbers in a doc, so treat this as an
  observation, not a measured table row.
- `cached_tokens` was confirmed to grow monotonically call-over-call
  within a lesson once the tools schema was unified to one byte-identical
  `research` schema per call (`84f24a6`) — before that fix, the forced
  round and the answer round sent different schemas and the whole lesson
  had to be re-read from scratch every call.
- Live smoke (`data/ling_skip_smoke.py`) found Ling's `needs_search`
  decision inconsistent rep-to-rep and never populating the optional
  `question` field at all in the reps recorded — see
  `docs/rewrite_on_weak_evidence.md` for the full negative-result
  writeup; this is why the host topic gate (`tutor.app.topic_gate`) now
  makes the decline/chat decision in host code instead of relying on the
  model.
- Normal turns run roughly 30-40s end to end on this machine
  (prefill-bound: ~9s for the forced call, ~12-25s reading evidence) —
  not verified on the Dell.

## Inferred / not directly observed

- Exact Vulkan device name string was not captured in this run's log (see
  above); it is inferred from the reference machine description and prior
  session data, not observed in this run's log text.
- SHA-256 of the model file was not recomputed for this WP; it is carried
  over from the pre-measured reference.
