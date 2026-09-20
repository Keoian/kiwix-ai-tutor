# Offline School Tutor — Implementation Plan

**Version:** 1.1
**Date:** 2026-09-19
**Inputs:** `offline_tutor_spec_v0.3.md` (architecture, budgets, gates) and `offline_tutor_kiwix_reuse_plan.md` (donor code and algorithms)
**Audience:** Claude Code, working in a fresh repository, with Tyler reviewing at milestone boundaries

**Changes in 1.1:** development moves to a MacBook Pro (Radeon Pro 5500M 8 GB) running Windows, using an already-installed Bonsai runtime at `C:\git\bonsai`. The Dell Inspiron 7577 (Linux Mint, GTX 1060 Max-Q 6 GB) becomes the port-and-tune target after development is complete. Context profile is **32K with q8_0 KV on both machines** — the configuration Tyler tuned for the Dell's 6 GiB; the spec's 16K target is superseded. Milestones, workstream A, machine assignments, cross-platform requirements, and the model-selection decision point are updated accordingly.

---

## 0. Machines, model, and corrections to the inputs

### 0.1 Machines

| Role | Machine | OS | GPU / backend | Notes |
|---|---|---|---|---|
| **Development** | MacBook Pro | Windows (the `C:\git\bonsai` path implies Boot Camp) | Radeon Pro 5500M 8 GB — **Vulkan** backend of the Prism fork (no CUDA, no Metal under Windows) | All of M0–M5 happen here |
| **Production target** | Dell Inspiron 7577 | Linux Mint | GTX 1060 Max-Q 6 GB — CUDA 12.x, `sm_61` | Port, tune, and accept in M6–M7 |
| Optional | GTX 1070 desktop | — | CUDA | Available for calibration benchmarks and sidecar builds if convenient; not required |

The tutor app talks to llama-server through its OpenAI-compatible HTTP API and never links against the runtime, so the backend difference (Vulkan on the MacBook, CUDA on the Dell) is invisible to application code.

**Both Windows and Linux are supported platforms, not a dev/prod split.** Development happens Windows-first, natively (no WSL2 — disk on the MacBook is limited), with the cross-platform discipline in §5 applied from the first commit so that the Linux port at M6 is a verification step, not a porting effort. The installer, launch scripts, and offline-acceptance script exist for both OSes; the Dell remains the machine where performance gates and the definitive offline acceptance are run, because it is the constrained target.

### 0.2 The development model

**File:** `Bonsai-8B-Q1_0.gguf`, context 32,768, measured **3,584 MiB VRAM** on the 5500M. That footprint (~1.15 GB weights + ~2.3 GB f16 KV at 32K + buffers) identifies it as the **1-bit Bonsai 8B** — the release the Ternary-Bonsai-8B model card labels "1-bit Bonsai 8B (prior)."

This matters for one reason: the spec's model is **Ternary-Bonsai-8B Q2_0** (2.03 GiB), and the publisher's own table puts the 1-bit release meaningfully behind it — average 70.5 vs 75.5, MMLU-Redux 65.7 vs 72.6, **BFCL (tool calling) 65.7 vs 73.9**. Tool-call reliability is the number the tutor lives on.

**Decision:** develop against Q1_0 as installed — it is running, it is fast, and it exercises every code path. Treat the model file as a config value. At the Dell tuning milestone (M6), run the tool-call verification and bake-off on **both** Q1_0 and Q2_0 and pick on measured tool-call integrity and rubric score. Q2_0 adds ~1 GB of VRAM on the Dell (still resident with room at 16K/q8_0), so the choice is about quality, not fit. Nothing in M1–M5 should assume either model.

**Context:** **32K on both machines.** Tyler tuned this configuration for the Dell's 6 GiB specifically: Q1_0 weights (~1.15 GB) + q8_0 KV at 32K (~2.3 GB) + buffers = the measured 3,584 MiB, leaving ~2.4 GiB headroom on the Dell. If the M6 bake-off selects Q2_0 instead, the same configuration is ~4.9 GiB — still resident with Optimus in on-demand mode, with less margin; A3 measures it. The spec's §8.2 budget table is superseded by the 32K profile below and will be updated in v0.4:

| Allocation | 32K profile |
|---|---:|
| System, tool schemas, profile | 800 |
| History incl. evidence tool results | 22,000 |
| Newest question + newest packet | 2,500 |
| Reserved generation (+400 if bounded thinking is enabled) | 2,000 |
| Safety margin | 5,468 |
| **Total** | **32,768** |

Evidence packet budget defaults to 2,000 tokens. Two consequences of running the context this large on Pascal, neither a reason to shrink it: decode slows as the KV fills (at a full 32K the per-token KV read is ~2× the weight read for Q1_0, so expect roughly half the short-context tok/s), and prefill of a large uncached prompt is the expensive operation — which makes the append-only prompt layout (WP-C1) more important, not less.

### 0.3 The existing runtime installation

`C:\git\bonsai` is a working, configured installation and is treated as **read-only**.

- For everyday development, launch llama-server from `C:\git\bonsai` exactly as it is configured today. The app reads the server's host/port from `config/dev.toml`.
- If the runtime itself needs changes — a different build flag, a patched chat template, a fork update, a reasoning-budget flag, grammar support — copy the entire installation into `<repo>/runtime/bonsai/` (gitignored) and make the changes there. `C:\git\bonsai` is never modified.
- `scripts/runtime_init.ps1` performs that copy, records the source path and a checksum manifest of what was copied, and writes `runtime/bonsai/ORIGIN.md` so anyone can see what diverged from the original.
- Application code never hard-codes either path; `config/dev.toml` has `runtime_dir` and `model_path`.

### 0.4 Archives on the development machine

The full collection lives at `D:\kiwix` on an external 5 TB HDD. The registry below is what gets registered for the tutor; everything else on the drive is ignored (Stack Overflow, Server Fault, gaming, movies, TV Tropes, etc. are not school content).

| Tier | Archive | Size | Location | Notes |
|---|---|---|---|---|
| **1** | `wikipedia_en_simple_all_maxi` | ~2 GB | `C:` SSD — **to download** | Primary. The only archive whose latency numbers count on the MacBook. |
| **2** | `wikipedia_en_all_maxi_2023-10.zim` | 103 GB | `D:\kiwix` | Encyclopedic fallback. Text extraction ignores the images, so retrieval matches the `_nopic` build. 2023-10 edition — label "outdated information" eval questions against it. |
| **3 — textbook** | `courses.lumenlearning.com_en_all_2021-03.zim` | 12 GB | `D:\kiwix` | **High priority.** OER course content: chapters, learning objectives, worked examples. Subject-tag on import from the course structure. |
| **3 — textbook** | `wikibooks_en_all_maxi` | ~4 GB | **to download** (`wikibooks_af` on the drive is Afrikaans) | Textbook-style supplements. |
| **3 — textbook** | `wikiversity_en_all_maxi_2021-03.zim` | 2.5 GB | `D:\kiwix` | Mixed quality; register, low weight. |
| **3 — Q&A** | `math.stackexchange.com_en_all_2023-08.zim` | 5.4 GB | `D:\kiwix` | Use OpenZIM MCP's Stack Exchange preset (clean Q&A rendering). Prefer accepted answers. The tutor labels SE evidence as a contributor's answer, not encyclopedic fact. |
| **3 — Q&A** | `matheducators.stackexchange.com_en_all_2023-10.zim` | 51 MB | `D:\kiwix` | Teaching-focused; small; worth a slightly higher weight for "how do I explain X" |
| **3 — Q&A** | `physics`, `chemistry`, `biology`, `history`, `english`, `linguistics`, `literature`, `puzzling` `.stackexchange.com_*.zim` | 55 MB – 1.3 GB each | `D:\kiwix` | Same preset and labeling rule. Subject tag from the site name. |
| **3 — how-to** | `wikihow_en_maxi_2023-03.zim` | 51 GB | `D:\kiwix` | Procedural content. Low weight; useful for "how do I write a lab report" style questions. |
| **Source-viewer only** | `gutenberg_en_all_2023-12.zim` | 74 GB | `D:\kiwix` | Primary literary texts for English class. Not searched for evidence in v1; the tutor can link a student to the book. Revisit if literature questions need quotations. |
| **Sample before registering** | `khanacademy_en_all_2023-03.zim` | 180 GB | `D:\kiwix` | Mostly video. Spend one hour at WP-B2 extracting sample pages to see whether transcripts or exercise text exist. If not, it is a link target at most. |
| Ignore | `booksdash`, `opentextbooks` (751 KB — a stub), `freecodecamp`, everything else | — | — | Not school content or not enough content to matter. |

Rules that follow from this collection:

- **Search order** is tier 1 → tier 2 → tier 3 by subject tag, with tier 3 consulted only when tier 1+2 coverage flags are weak or when the question is procedural ("how do I…") or pedagogical ("how would you explain…"). This keeps ordinary factual questions on the fast SSD archive.
- **Subject routing** becomes a v1 feature, not a later one (the reuse plan deferred it). With ten Stack Exchange sites plus Lumen, a "Roman Empire" question must not open `math.stackexchange`. The registry's subject tags plus the tutor's current-subject hint drive the archive allowlist per request; the model never selects archives.
- **Q&A rendering** uses the OpenZIM MCP Stack Exchange preset; passages are built per answer, with the accepted answer first and score recorded as metadata. Evidence from Q&A archives is rendered to the model with a `[Q&A]` marker so the tutor can qualify it.
- **Latency:** every log line carries the archive's storage class (`ssd` / `hdd`). Only tier-1 numbers feed the gates on the MacBook. The worker deadlines (3 s / 8 s) stay fixed; `partial` on a cold HDD read is the correct behavior and a free test of that path.
- **Edition ages** are all 2021–2024. That is fine for school content and gives concrete dates for the "outdated" eval category.
- **On the Dell:** tier 1 and the Lumen archive go on the SSD; the Q&A set (< 10 GB total) fits on the SSD too; `wikipedia_en_all_*` goes on the SSD as `_nopic` if disk allows, otherwise stays on an external drive with a documented latency penalty; Gutenberg and Khan stay external and are source-viewer only. Finalize at D1.

### 0.5 Corrections to the reuse plan

| Item | Finding | Action |
|---|---|---|
| OpenZIM MCP version | The reuse plan says it reviewed **v3.3.4**. The repository's latest published release is **v2.5.3** (2026-07-01). | Pin to the latest tag that actually exists at checkout. Record tag and SHA in `THIRD_PARTY_NOTICES.md`. |
| OpenZIM MCP file layout | Named modules (`bundle.py`, `synthesize.py`, `zim/search.py`, …) were not independently verified. | WP-B1 starts with an inventory mapping every reuse-plan reference to a real path; gaps are reimplemented. |
| Zim-Indexer license | None found. | Design reference only. No copied code. |

---

## 1. Milestones

| # | Milestone | Machine | Exit gate | Blocks |
|---|---|---|---|---|
| **M0** | Dev runtime verified | MacBook | Existing `C:\git\bonsai` server reachable; `--jinja` tool-call parsing works with the `research`/`calc` schemas (≤ 2% malformed on 20 scripted requests, else grammar path chosen); `/tokenize` and `/apply-template` verified; thinking off by default; baseline pp/tg recorded on the 5500M | M3 |
| **M1** | Vendored ZIM core | MacBook | Validate, open, fingerprint, search, resolve redirects, fetch entry, build `ArticleBundle` on the Simple Wikipedia ZIM; attribution present; runs on Windows | M2 |
| **M2** | Lexical retrieval baseline | MacBook | `research()` returns a budgeted packet with stable IDs/offsets; eval set runs; baseline metrics recorded | M3, M4 |
| **M3** | One tutoring turn | MacBook | Pre-retrieve → prompt → answer → citation → source-viewer highlight end to end; `calc` works; scripted routing lessons pass; two-call cap enforced | M5 |
| **M4** | Dense sidecar + hybrid | MacBook | Resumable, fingerprinted embedding pass; hybrid RRF beats lexical baseline on tuning without hurting held-out | M5 |
| **M5** | Lesson-complete app | MacBook | Profiles, lesson state, append-only prompt with eviction under the 32K profile, status page, 30-minute lesson within resource limits (Windows measurements) | M6 |
| **M6** | Dell port and model tuning | Dell | Fork built for `sm_61`; app runs on Linux unchanged; Phase 0 go/no-go (0 offloaded layers at 32K/q8_0, pp2048 ≥ 100, sustained tg128 ≥ 20 at minute 10 short-context); tool-call verification + bake-off on Q1_0 and Q2_0; model chosen; Linux resource limits measured | M7 |
| **M7** | Offline acceptance | Dell | Installer bundle with manifest/hashes; cold start with outbound blocked; held-out eval, arithmetic set, teaching rubric, real-student session | ship |

M0 and M1/M2 are independent and run in parallel on the MacBook. The Pascal-specific risks (kernel support, residency, prefill speed) are deferred to M6 by design — they cannot be retired on the MacBook, and nothing in M1–M5 depends on them.

---

## 2. Repository layout

```
tutor/
  app/
    main.py, session.py, prompt.py, agent_loop.py, llm_client.py, citations.py, routes.py
  retrieval/
    zim/        # vendored/adapted from OpenZIM MCP (MIT): archive, resolve, search, bundle, content, worker, models
    hybrid/     # ours; Zim-Indexer ideas reimplemented: lexical, dense, rrf, passages, ranking, diversity, packer
    index/      # simplewiki_build, simplewiki_store, manifest
    registry.py, research.py, cache.py
  tools/
    research_tool.py, calc_tool.py, schemas.py
  ui/
    index.html, app.js, app.css
  dev/
    mcp_adapter.py
  eval/
    questions/, run_retrieval_eval.py, run_turn_eval.py, rubric.md
  config/
    dev.toml        # MacBook: runtime_dir=C:\git\bonsai (or runtime/bonsai), model=Bonsai-8B-Q1_0.gguf, ctx=32768, kv=q8_0
    dell.toml       # Dell: runtime_dir=runtime/llama.cpp, model=<chosen at M6>, ctx=32768, kv=q8_0
  runtime/          # gitignored
    bonsai/         # copy of C:\git\bonsai, only if changes are needed (scripts/runtime_init.ps1)
    llama.cpp/      # Dell: fork source + sm_61 build (scripts/build_fork.sh)
  scripts/
    runtime_init.ps1          # copy C:\git\bonsai → runtime/bonsai with ORIGIN.md + checksums
    serve_dev.ps1             # launch llama-server from the configured runtime_dir (Windows)
    serve.sh                  # same, Linux
    build_fork.sh             # Dell, sm_61
    phase0_bench.sh / .ps1    # llama-bench sequence, both OSes
    install.py                # bundle assembly, manifest, hashes, embedding pass
    offline_acceptance.sh     # Dell only
  tests/
    fixtures/                 # ~50-article fixture ZIM, truncated copy, non-ZIM
  THIRD_PARTY_NOTICES.md
  pyproject.toml
```

Rules: `retrieval/zim/` carries upstream MIT headers. `retrieval/hybrid/` is ours. `app/` never imports `retrieval/zim/` directly; everything goes through `retrieval/research.py`. `runtime/` is never committed.

---

## 3. Work packages

Sizes are rough Claude Code session counts.

### Workstream A — Model and runtime

**WP-A0 — Wire up the existing dev runtime** (MacBook, ½ session)
Confirm `C:\git\bonsai` launches with the current configuration; capture the exact command line, port, context, and flags it uses into `config/dev.toml` and `scripts/serve_dev.ps1`. Record the fork commit and backend (Vulkan) from the server's startup log. Run `llama-bench` once for pp512/pp2048/tg128 on the 5500M as a dev-machine reference. **Do not modify `C:\git\bonsai`.** Deliverable: `docs/dev_runtime.md`.

**WP-A1 — Tool-call and template verification** (MacBook, 1 session)
Against the dev server with `--jinja`: 20 scripted requests using the `research`/`calc` schemas (10 should call, 10 should not). Count parsed calls, malformed calls, prose-that-looks-like-a-call. Verify thinking is off by default; check whether a reasoning-budget flag exists in this fork build. Verify `/tokenize` and `/apply-template`. If `--jinja` or any needed flag is not in the current `C:\git\bonsai` launch configuration, that is the first trigger for `scripts/runtime_init.ps1` — copy, then change the copy. Deliverable: results table; grammar-constraint decision. **Gate for M0.**

**WP-A2 — Dell fork build** (Dell, 1 session, M6)
Clone `PrismML-Eng/llama.cpp` (`prism` branch) into `runtime/llama.cpp`; grep the CUDA sources for the Q2_0/Q1_0 kernel types before building; build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61` on CUDA 12.x. If the CUDA kernels for the chosen format are absent, stop and report — this is the fallback trigger. Deliverable: `scripts/build_fork.sh`, build log, commit SHA.

**WP-A3 — Dell Phase 0 benchmark** (Dell, 1 session, M6)
Prime on-demand check; coherence test; `llama-bench` at 32K/q8_0 for **both** Q1_0 and Q2_0, at depth 0 and at depth 24K (`-d 24576`) so the KV-filled decode rate is measured, not guessed; 10-minute sustained loop with clock/temp logging. Deliverable: `docs/phase0_report.md` with the go/no-go table for each model.

**WP-A4 — Dell tool-call verification and bake-off** (Dell, 1 session, M6)
Re-run WP-A1's 20 scripted requests on the Dell build for Q1_0 and Q2_0. Then the bake-off on the 30 tuning questions with real retrieval packets: Q1_0 non-thinking, Q2_0 non-thinking, Q2_0 with a 400-token thinking budget, and Qwen 3.5 4B Q6_K on stock llama.cpp as the fallback control. Score on the teaching rubric and tool-call integrity. Deliverable: `docs/bakeoff.md`; chosen model written into `config/dell.toml`.

### Workstream B — Retrieval (MacBook; Simple Wikipedia ZIM on local SSD)

Unchanged in substance from plan 1.0; Windows-specific notes added.

**WP-B1 — Donor inventory and vendoring** (1–2 sessions). Check out OpenZIM MCP at its latest real tag; `docs/donor_inventory.md` mapping reuse-plan references to real paths; vendor the smallest cohesive units into `retrieval/zim/`; write `THIRD_PARTY_NOTICES.md`. Confirm `python-libzim` installs from a wheel on Windows for the pinned Python; if it doesn't, that is a blocker to report immediately, not to work around. Acceptance: `from tutor.retrieval.zim import archive` imports with only `libzim`, an HTML parser, and stdlib, on Windows.

**WP-B2 — Archive validation, registry, fingerprint** (1 session). Structured status for valid / truncated / renamed non-ZIM / no full-text index. Fingerprint from size + mtime + ZIM UUID + edition metadata. Paths handled with `pathlib` throughout — no string concatenation, no assumptions about separators or drive letters.

**WP-B3 — Resolve, search, worker** (2 sessions). Redirect chain with cycle detection; full-text, title, snippets; libzim child-process worker. **Windows note:** `multiprocessing` uses `spawn`, not `fork` — the worker must be launchable from a clean interpreter with explicit arguments; no inherited state. Hard-kill is `Process.kill()`/`terminate()`; the "only one worker alive" assertion uses `psutil`. Acceptance as in 1.0, verified on Windows; re-verified on Linux at M6.

**WP-B4 — ArticleBundle and offsets** (2 sessions). Five fixture pages; section offsets round-trip. Text is handled as Unicode throughout with explicit `encoding="utf-8"` on every file open — Windows default encodings will silently corrupt offsets otherwise.

**WP-B5 — Passages, citations, snapshots** (1–2 sessions). Passage ID from fingerprint + path + section + extractor version + text hash + span. Snapshot store. Acceptance: exact highlight after disposable caches are cleared.

**WP-B6 — Lexical `research()` end to end** (2 sessions). Full pipeline, versioned response, deadlines through the worker, caches. Runs `eval/run_retrieval_eval.py`; baseline recorded in `docs/retrieval_baseline.md`. **Gate for M2.**

**WP-B7 — Dense sidecar build** (1–2 sessions). Embedding model choice: prefer a GGUF embedding model served by the same llama.cpp build (one runtime to package, works on Vulkan and CUDA alike) over `onnxruntime`, unless the fork's embedding path proves unreliable. Resumable, checkpointed, fingerprinted build; fp16 flat file; brute-force top-k. Build on the MacBook GPU. Acceptance: resume-without-duplicates; query < 50 ms CPU.

**WP-B8 — Hybrid fusion and ranking flags** (1–2 sessions). Dense candidates into RRF; `title_boost`, `mention_penalty`, `heading_affinity`, `lead_augmentation` behind flags, off by default, each justified by an eval table. **Gate for M4.**

### Workstream C — Tutor app (MacBook)

**WP-C1 — llama-server client and prompt builder** (1–2 sessions). Streaming chat with tools, cancel, `/tokenize`, `/apply-template`. Strict chronological log; token accounting against the 32K profile in §0.2; head eviction protecting cited triples; `eviction_reprefill` event. The eviction path is exercised in tests by injecting a small ceiling (e.g. 6K), since a 32K lesson rarely reaches it naturally. Acceptance: byte-prefix property proven by unit test; prompt-cache hit tokens confirmed live against the dev server.

**WP-C2 — Agent loop and tools** (1–2 sessions). Pre-retrieve on free text; skip on actions; `research` cap 2, `calc` cap 4; schema validation; never parse prose. **`calc_tool.py` on Windows:** the `resource` module does not exist — enforce the 1 s wall clock with subprocess timeout and the memory cap with `psutil` polling (or a Job Object); keep the interface identical so the Linux build can add `RLIMIT_AS` at M6. Acceptance: the six scripted lessons from 1.0.

**WP-C3 — UI and source viewer** (1–2 sessions). Static HTML/JS, SSE, stop, selector, action buttons, `[S#]` chips, highlighted source viewer, status panel. Kiosk launch: `scripts/kiosk.ps1` (Edge/Chrome `--kiosk`) for dev, `scripts/kiosk.sh` for the Dell. **Gate for M3.**

**WP-C4 — Profiles, lesson state, resource discipline** (1 session). As in 1.0. Resource measurements on Windows are indicative only; the numbers that count are re-taken on the Dell in M6. **Gate for M5.**

### Workstream D — Port, packaging, acceptance (Dell)

**WP-D0 — Linux port** (Dell, 1 session, M6)
Clone the repo on the Dell; install pinned wheels; run the full unit and integration suite on Linux. Expected work: path edge cases, `spawn` vs `fork` defaults, `resource` limits in the calculator, kiosk script, service launch. Any Windows-only assumption found here is fixed in the shared code, not patched around. Acceptance: green test suite on both OSes from the same commit.

**WP-D1 — Installer and manifest** (Dell, 1–2 sessions; Windows variant verified on the MacBook). `install.py` is one cross-platform script: bundle the platform's runtime build (Dell fork with `sm_61`; the Windows Vulkan build from `runtime/bonsai`), the chosen GGUF plus fallback, embedding model, wheels for that platform, UI, stopword list; SHA-256 everything; register archives; `Archive.check()` once; import the sidecar built on the MacBook (fingerprint must match) or rebuild; write manifest; GPU-mode warning (Prime on Linux). Acceptance: fresh Mint install → installer → app starts with no network; the same script produces a working Windows install on the MacBook.

**WP-D2 — Offline acceptance and evaluation** (Dell, 1–2 sessions). Outbound block via `nftables` on Linux (Windows Firewall rule via `platform_/windows.py` for a best-effort Windows run), cleared caches, cold start, full lesson, zero tutor-originated connections; held-out retrieval eval, arithmetic set, turn eval; rubric on 20 transcripts; real-student session. The Dell run is the one that gates shipping. **Gate for M7.**

---

## 4. Order of work

```
MacBook ──────────────────────────────────────────────────────────────
Week 1   A0 → A1 → [M0]              ║  B1 → B2 → B3
Week 2                               ║  B4 → B5 → B6 → [M2 report]
Week 3   C1 → C2 → C3 → [M3 report]  ║  B7
Week 4   C4 → [M5]                   ║  B8 → [M4 report]

Dell ─────────────────────────────────────────────────────────────────
Week 5   D0 → A2 → A3 → A4 → [M6 report: model chosen]
Week 6   D1 → D2 → [M7]
```

Enforced rules: no embeddings before the lexical baseline (B7 waits on B6's report); no model wiring before retrieval is measured (C2 waits on M2); no Dell work before M5 except an early smoke run of D0 if a Windows-only assumption is suspected mid-stream.

---

## 5. Cross-platform requirements

Windows and Linux are both shipping targets. These rules apply from the first commit and are enforced by CI, not by review:

- `pathlib.Path` everywhere; no `os.path.join` with hard-coded separators; no drive-letter assumptions; config paths may be absolute on either OS.
- Every file open specifies `encoding="utf-8"`; every subprocess pipe is decoded explicitly.
- `multiprocessing.set_start_method("spawn")` is the *only* start method used, on both OSes, so behavior is identical.
- No `resource`, `fcntl`, `signal.SIGKILL`, or `os.fork` in shared code. Platform-specific bits live behind `tutor/platform_/` with a `windows.py` and a `linux.py` implementing the same small interface: calculator process limits, kiosk launch, outbound-network block for acceptance testing, service/autostart registration.
- SQLite in WAL mode with explicit `check_same_thread=False` handling; no reliance on file-locking semantics that differ between NTFS and ext4.
- CI runs the full unit suite and the fixture-ZIM integration tests on both a Windows and a Linux runner from day one. A commit that passes on one OS and fails on the other does not merge.
- The llama-server client treats the server as a black box: no assumptions about backend, only about the HTTP API and the response fields used.
- Scripts come in pairs (`.ps1` / `.sh`) with identical behavior, or as a single Python script when the logic is non-trivial. `install.py`, `phase0_bench`, and `offline_acceptance` are Python with a platform switch, not shell.
- Nothing is tested only on the MacBook and assumed on the Dell. WP-D0 re-runs every suite and every scripted lesson on Linux before any Dell tuning starts.

---

## 6. Testing strategy

Unchanged from 1.0 (fixture ZIM built with `python-libzim`'s writer; fast unit tests; marked integration tests; eval scripts at M2/M4/M6/M7; determinism check), with two additions:

- Every test that touches the filesystem or a subprocess runs on both OSes in CI.
- The one-turn integration test is parameterized by `config/*.toml` so the same test runs against the MacBook dev server and the Dell server.

---

## 7. Dependencies to pin

| Dependency | Note |
|---|---|
| Prism fork, as installed in `C:\git\bonsai` | Record the commit from the startup log at WP-A0; that is the dev pin |
| Prism fork, `prism` branch at M6 | Dell pin; may be newer than the dev pin — note any template/flag differences |
| CUDA 12.x + Pascal-capable driver | Dell only |
| `python-libzim` | Must have Windows and Linux wheels for the pinned Python; verify at WP-B1 |
| HTML parser | Match OpenZIM MCP's lockfile at the pinned tag |
| BM25 | Hand-rolled, ~60 lines, deterministic |
| Embedding model | GGUF via the same llama.cpp build, first choice |
| `sympy`, `psutil`, FastAPI + uvicorn | — |
| Python | OpenZIM MCP's `.python-version`, provided `python-libzim` has wheels for it on both OSes |

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Q1_0's weaker tool calling (BFCL 65.7) produces malformed calls during dev | Medium | Wasted debugging of "bugs" that are model behavior | WP-A1 measures it first; grammar constraints if > 2%; M6 bake-off decides Q1_0 vs Q2_0 on the Dell |
| Fork's CUDA kernels for the chosen format don't exist or don't build for `sm_61` | Medium | Model change late in the project | Deferred to M6 by design; fallback (Qwen 3.5 4B on stock llama.cpp) is API-identical; app code is untouched |
| Windows-only assumptions leak into shared code | High without discipline | Dell port stalls | §5 rules; dual-OS CI from the first commit; D0 as an explicit port step |
| `python-libzim` has no Windows wheel for the pinned Python | Low–medium | Blocks all of workstream B on the MacBook | Check at WP-B1 before anything else; if absent, pick the nearest Python with wheels on both OSes |
| Decode rate at a full 32K KV on Pascal is lower than short-context benchmarks suggest | Medium | Late-lesson turns feel slower | A3 benchmarks at depth 24K; if the filled-context rate is below ~10 tok/s, start a new lesson at a subject change rather than shrinking the context globally |
| Prefill on Pascal at the low end | Medium | First-token latency | Append-only prompt; reduce evidence budget first |
| OpenZIM MCP internals don't match the reuse plan | Medium | Vendoring slower | B1 inventory; reimplement gaps |
| Thermal throttling on the Dell | Medium | Decode below gate | A3 sustained run; ops note for fan/undervolt |
| Sympy sandbox escape | Low | Code execution | Whitelist parser, subprocess, timeouts; Linux adds RLIMIT at M6 |

---

## 9. Spec amendments to fold into v0.4

As in 1.0 (§4, §6, §7.2, §7.4, §11, §15, new §17 third-party code), plus:

- §2 Hardware: add a "development environment" row for the MacBook/5500M/Windows and state that all performance gates are Dell-only.
- §3.1 Model: record that development used 1-bit Bonsai 8B Q1_0 and that the production model is chosen at M6 between Q1_0 and Ternary Q2_0 on measured tool-call integrity and rubric score.
- §8.2 Budget: replace the 8K/16K tables with the 32K profile from this plan's §0.2; note that the model/KV configuration was tuned for the Dell's 6 GiB at 32K and that Q2_0 also fits at that setting.

---

## 10. Operating instructions for Claude Code

1. Start with WP-A0 and WP-B1 in parallel. Neither depends on the other.
2. **`C:\git\bonsai` is read-only.** If the runtime needs a change, run `scripts/runtime_init.ps1` first, then change the copy in `runtime/bonsai/`, then point `config/dev.toml` at it. Never edit in place.
3. Treat the model as a config value. No code path may branch on the model name.
4. The 32K/q8_0 configuration is the profile on both machines. The prompt builder reads the ceiling from config; tests exercise eviction by injecting a smaller one.
5. Every work package ends with its acceptance test passing on Windows and a note in `docs/`. Reports at M0, M2, M3, M4, M5 go to Tyler before the next milestone; M6 and M7 happen on the Dell.
6. Windows and Linux are both targets. Follow §5 from the first commit; the dual-OS CI is the enforcement. A Windows-only shortcut found at D0 is a regression, not a porting task.
7. Vendor the smallest unit that works; keep upstream MIT headers; update `THIRD_PARTY_NOTICES.md` in the same commit. Nothing from Zim-Indexer is copied.
8. Ranking flags default off; a flag is enabled only by a commit that includes its eval table.
9. Do not build a generic RAG framework. One retrieval entry point (`research()`), two model-facing tools (`research`, `calc`).
10. The dev MCP adapter exists for Claude Code's own testing; it never enters the production process tree.
11. When a budget, cap, or threshold is in doubt, the spec wins; when an algorithm is in doubt, the reuse plan's donor wins; when they disagree, stop and ask.
