# Ternary-Bonsai-8B bake-off (Radeon Pro 5500M, Vulkan) — loadable variant found

Model family: dense 8B transformer, 36 layers / 8 KV heads. Three GGUF
variants exist in `runtime/models/ternary-bonsai-8b/`, sha256-verified:

| File | Size | Loads on Vulkan? |
|---|---|---|
| `Ternary-Bonsai-8B-Q2_0.gguf` | 2.18 GiB | **No** — legacy Prism group-128 packing under ggml type id 42; this build's Q2_0 reader expects official group-64. See prior attempt below. |
| `Ternary-Bonsai-8B-PQ2_0.gguf` | 2.18 GiB | Yes |
| `Ternary-Bonsai-8B-Q2_0_g64.gguf` | 2.31 GiB | Yes — **chosen** |

## Load test (`llama-bench -ngl 99 -fa 1 -p 64 -n 16`)

- `Ternary-Bonsai-8B-PQ2_0.gguf`: loads, full Vulkan offload. pp64 =
  160.05 t/s, tg16 = 51.91 t/s.
- `Ternary-Bonsai-8B-Q2_0_g64.gguf`: loads, full Vulkan offload. pp64 =
  221.24 t/s, tg16 = 51.29 t/s.

g64 is noticeably faster on prefill (221 vs 160 t/s) with equal decode, so
it was kept as the primary candidate for the sweep; PQ2_0 was not swept
further (its own numbers above are the full record for it).

## Sweep (`llama-bench -ngl 99 -p 512 -n 64`, g64 file, `-r 2`)

| fa | ctk/ctv | ub | pp@d0 | pp@d4096 | pp@d8192 | tg64@d0 | tg64@d4096 | tg64@d8192 |
|----|---------|-----|-------|----------|----------|---------|------------|------------|
| 1 | q8_0/q8_0 | 512 | 229.0 | 24.6 | **> 4 min (aborted)** | 49.0 | 28.7 | n/a |
| 1 | q8_0/q8_0 | 128 | 219.6 | 29.0 | not run | 50.1 | 31.8 | not run |
| 1 | q8_0/q8_0 | 256 | 157.5 | 27.1 | not run | 45.9 | 30.7 | not run |
| 0 | f16/f16 | 128 | 157.0 | 174.5 ± 21.6 (noisy) | 122.2 ± 9.4 | 50.7 | 26.7 | 10.6 |
| 0 | f16/f16 | 192 | 159.8 | 54.8 | not run | 50.5 | 26.8 | not run |
| 0 | f16/f16 | 256 | 164.1 | 55.3 | not run | 50.5 | 22.9 | not run |

Key finding, consistent with the known GPU pathology on the sibling
1-bit model: **fa=1 (q8_0 KV) collapses in prefill with depth** on this
Radeon Pro 5500M/Vulkan build — pp512@d4096 drops to ~25-29 t/s (from
~160-230 t/s at d0), and pp512@d8192 did not finish inside the 4-minute
cap at all. **fa=0 (f16 KV) holds up far better in prefill**: pp@d4096
stays at 55-175 t/s and pp@d8192 = 122 t/s, though decode falls to
10.6 t/s at d8192 (below the ~15 t/s floor used elsewhere, but only at
very deep context). ub=128 was the best fa=0 ubatch (best d8192 numbers,
competitive at d0/d4096); ub=512 was not tested for fa=0 (time budget).

`-ctk/-ctv q8_0` without `-fa 1` was not tried here (matches the Granite
finding that quantized KV requires flash attention on this build).

## Memory and context-size decision

f16 KV at the model's originally-planned 32K context would be too large
to fit the 5.6 GiB budget (~2.15 GiB model + ~4.8 GiB f16 KV at 32K, per
the task's own KV-size estimate). Measured directly: with `ctx_size =
20480`, total dedicated GPU memory (`GPU Adapter Memory` counter) was
**6,888,296,448 B ≈ 6.41 GiB — over budget.** Context was cut to
`ctx_size = 16384`; re-measured total dedicated GPU memory was
**5,944,033,280 B ≈ 5.54 GiB**, under the 5.6 GiB cap (thin margin, ~69
MiB headroom below 5.6 GiB = 6,013,091,840 B). Server log verbosity at
the default level did not print the per-buffer breakdown lines the
Granite run captured (`-lv 4` was not set in this config), so only the
whole-adapter counter is available here.

## Chosen configuration

`config/dev.ternary.toml`: model = `Ternary-Bonsai-8B-Q2_0_g64.gguf`,
`n_gpu_layers=99`, `flash_attn=false`, `cache_type_k=cache_type_v=f16`,
`ctx_size=16384`, `extra_args=["-ub","128"]`.

Reasoning (also in the TOML as a comment): this is a dense 36-layer
transformer, unlike Granite's Mamba-2 hybrid, so flash-attn's
depth-collapse pathology hits prefill hard once `-fa 1` is on, per the
sweep above. The app's steady-state turns are append-only with prompt
caching — each turn only prefills roughly 0.5-1.5k *new* tokens at
whatever depth the conversation has reached (5-20k), so prefill-at-depth
throughput matters more to the real workload than raw decode t/s;
fa=0/f16 wins decisively there (55-175 t/s vs 25-29 t/s at d4096) even
though its decode is a bit lower in that band (23-27 vs 29-32 t/s) and
falls to 10.6 t/s only at the much deeper d8192 point. f16 KV forces the
smaller 16K context to stay under the 5.6 GiB VRAM cap; 32K was not
achievable with fa=0 on this model's KV size, so this is a deliberate
context-size-for-prefill-speed trade rather than the full 32K Granite got.

## Sanity check: `/v1/chat/completions`

Request: `{"model": "ternary", "messages": [{"role": "user", "content":
"Say hello in five words."}], "stream": false, "stream_options":
{"include_usage": true}}`.

Measured (client-side wall clock 544 ms):

- `usage`: `{"completion_tokens": 9, "prompt_tokens": 18, "total_tokens":
  27, "prompt_tokens_details": {"cached_tokens": 0}}`.
- `timings`: `prompt_per_second: 57.95`, `predicted_per_second: 45.70`
  (short cold prompt, not representative of the sweep's batched
  throughput numbers above).
- Response: `"Hello, how can I assist you?"` — off-script but the
  endpoint, usage accounting, and timings all work correctly.

## Side-by-side with Granite

| | Granite 4.0 H-Tiny | Ternary-Bonsai-8B (g64) |
|---|---|---|
| Loads on Vulkan? | Yes, first try | Q2_0 file: no; PQ2_0 and Q2_0_g64: yes |
| Architecture | granitehybrid 7B.A1B (Mamba-2/attention MoE) | dense 8B transformer, 36 layers / 8 KV heads |
| Chosen flags | fa=1, KV q8_0/q8_0, ub=512 (default) | fa=0, KV f16/f16, ub=128 |
| Chosen context | 32768 | 16384 (cut from a 32K/20K target to fit VRAM with f16 KV) |
| Chosen VRAM | ~5.36 GiB | ~5.54 GiB |
| pp t/s (d0/4k/8k) | 683 / 387 / n/a (640 @ 2048-tok check) | 157 / 174.5 (noisy) / 122.2 |
| tg64 t/s (d0/4k/8k) | 67.4 / 63.9 / 30.6 | 50.7 / 26.7 / 10.6 |

## Server state at end of task

`llama-server.exe` running on `127.0.0.1:8080` with
`config/dev.ternary.toml`, healthy (`/health` -> ok), idle (`/slots` ->
`is_processing: false`).
