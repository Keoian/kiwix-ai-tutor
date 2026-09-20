# Granite 4.0 H-Tiny bake-off (Radeon Pro 5500M, Vulkan)

Model: `runtime/models/granite-4.0-h-tiny/granite-4.0-h-tiny-Q4_K_M.gguf`
(granitehybrid 7B.A1B, Mamba-2/attention hybrid MoE, 6.94B total params,
Q4_K_M, 3.94 GiB on disk). Binary: `C:\git\bonsai\bin\llama-server.exe` /
`llama-bench.exe` (build `b10716 ee2aeae98`, Vulkan backend, AMD Radeon Pro
5500M, 8176 MiB total / ~7337-7346 MiB free at idle).

Config: `config/dev.granite.toml` (copy of `config/dev.toml`; same
`runtime_dir`, model_path repointed at the Granite gguf, port 8080, ctx
32768, `[embedding].enabled = false`).

## Does it load on Vulkan?

Yes, first try, full GPU offload, no CPU fallback needed. Load time ~5.1 s.
No `n_gpu_layers=0` fallback test was needed (step 2's "if load fails"
branch did not trigger).

## Memory breakdown (measured, from server log at `-lv 4`, ctx=32768, ngl=99, fa=on, KV q8_0/q8_0)

```
n_layer = 40 (+ 1 output layer) -> offloaded 41/41 layers to GPU
CPU_Mapped model buffer size    =   120.59 MiB   (non-offloaded remainder)
Vulkan0 model buffer size       =  4031.55 MiB
Vulkan0 KV buffer size          =   136.00 MiB
Vulkan0 RS (recurrent-state) sz =    55.37 MiB   (Mamba-2 conv/ssm state)
Vulkan0 compute buffer size     =    78.34 MiB
Vulkan_Host compute buffer size =    38.09 MiB
```

Cross-check, `(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage')`
while the server sat idle at 32K ctx: **5,751,201,792 B ≈ 5.36 GiB** total
dedicated GPU memory in use. Comfortably under the 5.6 GiB budget (6 GiB
delivery machine).

## Sweep (llama-bench, `-ngl 99 -r 2`, ≤3 min per run, server stopped during bench)

| fa | ctk/ctv | ub  | pp@d0 (t/s) | pp@d4096 (t/s) | tg64@d0 (t/s) | tg64@d4096 (t/s) |
|----|---------|-----|-------------|-----------------|----------------|-------------------|
| 0  | f16 (default) | 256 | 248.5 | 219.6 | 68.1 | 29.6 |
| 0  | f16 (default) | 512 | **696.2** | **678.8** | 67.8 | **63.9** |
| 1  | q8_0/q8_0 | 256 | 248.9 | 194.6 | 67.5 | 30.0 |
| 1  | q8_0/q8_0 | 512 | 683.1 | 386.7 | 67.4 | 63.9 |

Extra depth-8192 check, fa=0/f16/ub512/p2048: pp2048@d0 = 706.6 t/s,
pp2048@d8192 = 640.4 t/s, tg64@d0 = 68.0 t/s, tg64@d8192 = 30.6 t/s.

**Important constraint found during the sweep**: `-ctk q8_0 -ctv q8_0`
without `-fa 1` fails outright on this build (`llama_bench: error: failed
to create context`) -- quantized KV cache requires flash attention here.
So f16-KV/fa=off is not a real alternative once q8_0 KV is required to fit
the 5.6 GiB VRAM budget at 32K context; the only viable "small KV" branch
of the sweep is fa=1 + q8_0/q8_0.

ub=512 (the llama-server default, not set explicitly in the TOML) clearly
beats ub=256 in every case, especially at depth (2-3x on pp@d4096).

## Chosen configuration

`config/dev.granite.toml`: `n_gpu_layers=99` (full offload), `flash_attn=true`,
`cache_type_k=cache_type_v=q8_0`, `ctx_size=32768`, default ubatch (512).
This is required (not just fastest) for the q8_0 KV path, and its numbers
(pp 683/387 t/s @ d0/d4096, tg 67.4/63.9 t/s @ d0/d4096, ~5.36 GiB VRAM)
clear the "decode >= 15 t/s" bar by 4x and stay under the 5.6 GiB cap.

## Sanity check: `/v1/chat/completions`

Request: `{"model": "granite", "messages": [{"role": "user", "content":
"Say hello in five words."}], "stream": false, "stream_options":
{"include_usage": true}}` (see `docs/llm_client.md` on `stream_options`).

Measured (client-side wall clock 2032 ms, includes cold single-request
overhead -- not comparable to the batched llama-bench numbers above):

- `usage`: `{"completion_tokens": 10, "prompt_tokens": 36, "total_tokens":
  46, "prompt_tokens_details": {"cached_tokens": 0}}` -- usage and
  cached-token fields present as expected.
- `timings`: `prompt_per_second: 21.06`, `predicted_per_second: 34.84`
  (short prompt, so per-token overhead dominates; not representative of
  throughput at depth -- see the sweep table for that).
- Response: `"Hello, how are you doing today?"` (model answered off-script
  but the endpoint, usage accounting, and timings all work correctly).

## Server state at end of task

`llama-server.exe` running on `127.0.0.1:8080` with `config/dev.granite.toml`,
healthy (`/health` -> `{"status":"ok"}`), idle (`/slots` -> `is_processing:
false`).
