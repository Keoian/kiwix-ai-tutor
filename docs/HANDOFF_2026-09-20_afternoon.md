# Handoff: offline school tutor (kiwix-ai-tutor)

Written 2026-09-20 ~15:00 for the session that continues after a `/clear`. You have no prior
context. Read this, then `docs/bakeoff_dev/README.md`. The previous handoff is kept at
`docs/HANDOFF_2026-09-19_original.md`; its house rules still apply and are restated in §2.

**You are the orchestrator. Sonnet sub-agents write the code. Tests come first.**

---

## 1. Where the project stands

M0–M3 done. M4 built but its gate FAILED on held-out (dense retrieval is off by default). M5 built
and measured on Windows. M6/M7 need the Dell. 826 unit tests pass, ruff clean, CI (GitHub Actions,
windows + ubuntu) green as of the last check. Everything is pushed; latest work commit `cdee690`,
then this handoff.

| Report | What it holds |
|---|---|
| `docs/M0_report.md` … `docs/M5_report.md`, `docs/M5_notes.md` | Milestone gates with evidence |
| `docs/retrieval_baseline.md` | Retrieval v1→v6 with before/after tables (tuning split) |
| `docs/M4_report.md`, `docs/hybrid_eval.md` | Why hybrid/dense is disabled |
| `docs/citation_experiment.md` | Every citation measurement, incl. the evidence-dump failure |
| `docs/bakeoff_dev/README.md` | **Three-model comparison table** (+ per-model files) |
| `docs/review_2026-09-20.md`, `…_pass2.md` | Two code reviews with resolutions |
| `docs/dense_sidecar.md` | Dense index build, CPU embedding server |

### The model decision (2026-09-20)

**Granite 4.0 H-Tiny Q4_K_M is the dev model going forward.** Bonsai Q1_0 (binary) cited nothing
under identical conditions and answers from memory; Ternary Bonsai cites but needs fa-off/f16 KV on
this GPU, so only 16K context fits and late-lesson first-token p95 was 75 s.

10-minute lesson soak, same code:

| | Bonsai Q1_0 | Ternary Q2_0_g64 | **Granite H-Tiny** |
|---|---|---|---|
| Turns / errors | 18 / 0 | 29 / 0 | **62 / 0** |
| First token p50 (p95) | 25.1 s (56.9) | 8.5 s (74.8) | **3.0 s (11.2)** |
| Answers carrying a citation | 0.00 | 0.95 (old metric) | **1.00** |
| Citations passing the support check | 0.00 | not measured | **0.00** |
| Context fitting ≤ 5.6 GiB | 32K | 16K | 32K |

Granite: 5.36 GiB VRAM at 32K, pp 683 t/s (387 at 4k depth), tg 67 t/s, prompt cache works across
turns despite Mamba layers, tool calls 0/20 malformed and 19/20 correct decisions.

**Granite's defect:** it cites 100% of lesson answers but **0% pass the support check** — it
staples one `[S1]` to the end of a 4–6 sentence paragraph instead of the claim it backs. Cold
single-turn questions: cited 0.44, supported 0.11. It does USE the evidence (reproduced the helium
infobox figures exactly, 3/3). Also: 6/8 calc answers right in one soak; sometimes pitches above a
10–16 year-old; did not call `calc` for a °C→°F conversion.

## 2. Working agreement (unchanged — the user's hard constraints)

- **Sub-agents: `model: "sonnet"` on every `Agent` call. Never `subagent_type: "fork"`.**
- **TDD:** failing test first; the orchestrator verifies by running things, not by trusting reports.
- **Conserve orchestrator output and context.** Delegate anything a sub-agent can do, including
  commits and pushes. Ask sub-agents for short final reports.
- **Pushing is pre-authorised** for this project (user, 2026-09-19). Commit as Keoian's noreply
  address (already set repo-locally). End commit messages with the attribution lines the session
  reminder gives you.
- **Never delete files.** Move junk to `D:\_trash_kiwix-ai-tutor\` and append a line to its
  `TRASH_LOG.txt` (timestamp | original | new | why). An autoclassifier blocks deletes.
- `C:\git\bonsai` is read-only. (One agent ran `git lfs pull` inside it to materialise
  `llama-bench.exe`; `git status` there is clean. Tell agents: no commands inside that directory.)
- Never hard-kill `llama-server` mid-prompt (suspected AMD driver reset). Stop it only when
  `/slots` shows idle. The user has given permission to stop/restart it and the app at will.
- The spec wins on budgets/caps/thresholds; the reuse plan wins on algorithms; if they conflict, ask.
- Writing for the user: lead with the outcome, measured numbers not adjectives, say what was
  inferred versus measured, say plainly what was not verified.

### Lessons about driving Sonnet sub-agents (learned the hard way)

- **Keep tasks small.** Three times an agent handed back a large multi-part task untouched. One
  concern per agent; split "build" from "measure live".
- Tell them the unit suite takes **~3 minutes — use a shell timeout ≥ 600000 ms**. One agent
  reported "295 passed" from a partial run; the true count is 826.
- **Spawn rule:** the ZIM worker uses multiprocessing `spawn`. Any repro must be a real `.py` file
  with `if __name__ == "__main__":`. A `python -` / `python -c` script silently returns
  `status: partial` after ~6 s. Set `PYTHONIOENCODING=utf-8` when printing (text has `−` and `°`).
- When two agents share the tree: forbid `git stash/checkout/restore/reset/clean` and deleting
  files in EVERY prompt, and give each agent an explicit list of files it owns. (One agent's
  `git stash` reverted another's edits; another's `rm -rf` removed a colleague's scratch output.)
- Scratch files go in `data/` (gitignored), never the repo root.
- Tell them the real archive EXISTS (give the `ls` line) — one agent wrongly concluded it didn't.
- Held-out split: never let an agent run it. The 30 original held-out questions have been observed
  several times and are no longer clean; the 12 `student_phrasing` held-out items were run once.
- Agents' own completion reports are claims. Twice a "done" report hid an unmet acceptance
  criterion that a 20-second orchestrator check exposed.

## 3. Machine state right now

| Thing | State |
|---|---|
| llama-server :8080 | **Granite**, `config/dev.granite.toml`, healthy |
| Tutor app :8420 | running against Granite, full registry (`config/archives.dev.toml`, touches `D:`) |
| Embedding server :8081 | CPU-only bge-small, idle; only needed if `[embedding].enabled = true` |
| `config/dev.toml` | **still points at Bonsai Q1_0** — see task 1 |
| Models | `runtime/models/granite-4.0-h-tiny/`; `runtime/models/ternary-bonsai-8b/` (3 variants; only `Q2_0_g64` and `PQ2_0` load on this build); `runtime/models/embedding.gguf` |
| Archives | `C:\kiwix\` Simple Wikipedia (tier 1) + Wikibooks; `D:\Kiwix\` everything else (external HDD — the user cannot move the laptop while it is spun up; use `config/archives.simplewiki_only.toml` to keep `D:` idle) |
| Dense index | `runtime/simplewiki_dense/` (281,270 vectors) — **stale**: the extractor is now `zim-bundle-v2`. Rebuild (~3 h; CPU is as fast as GPU) before re-enabling hybrid |
| Trash | `D:\_trash_kiwix-ai-tutor\` — safe for the user to delete |

Run it: `.\scripts\serve_dev.ps1 -Config config\dev.granite.toml`, then
`python -m tutor.app.main --config config\dev.granite.toml` → http://127.0.0.1:8420.

## 4. Tasks to implement, in order

Each is sized for one red→green pair of Sonnet agents unless noted. Measure live after 2 and 3.

### Task 1 — Make Granite the default dev config (small)

Point `config/dev.toml` at Granite (suggest: `dev.toml` becomes the Granite profile and the Bonsai
one is kept as `config/dev.bonsai-q1.toml`). `tests/test_settings.py` pins dev.toml to
`Bonsai-8B-Q1_0.gguf`, a runtime-relative model path and temp 0.5 — update those tests first. The
source guard (no "bonsai" / `C:\` inside `tutor/settings.py`) stays. Update `docs/dev_runtime.md`
and `README.md`. No code may branch on model name.

### Task 2 — Host-side sentence-level attribution (the main one)

Problem: models cite sloppily (Granite) or not at all; the product's promise is "check the tutor
against the book". Stop depending on the model's label placement.

- In `tutor/app/citations.py`: after the answer completes, split it into sentences; for each
  sentence find the best-supporting passage in the turn's packet plus retained evidence, reusing
  the existing `is_supported` logic (tokenizer from `tutor.retrieval.hybrid.lexical`; numbers count
  as terms; require a minimum overlap). Output `attributions: [{sentence_span, passage_id, label,
  score}]` and `unbacked_spans`.
- **The host never edits the model's text and never inserts `[S#]` into it.** Attribution is a
  separate layer: a new SSE event (or fields on `citations`), rendered in the UI as a subtle marker
  per backed sentence that opens the source viewer on the supporting passage, and a distinct style
  plus legend for "the tutor's own words — not checked against the library". `textContent` only;
  keep `tests/test_ui_static.py` green.
- Numbers deserve special care: a sentence containing a figure that appears in NO passage gets
  flagged (`unbacked_number`) — this is the wrong-from-memory case. A figure matching a key-fact
  passage links to it.
- Keep the model's own `[S#]` chips working; show model-cited versus host-found distinctly.
- Extend scoring in `eval/run_turn_eval.py` and `eval/run_lesson_soak.py`:
  `backed_sentence_rate`, `unbacked_number_rate`. Fixtures: the owner's 11-passage dump answer
  (already in `tests/test_citations.py`) and Granite's trailing-`[S1]` paragraphs (see
  `data/granite_soak10_v2.turns.json` if still present; else regenerate with a 3-minute soak).
- Spec check first: read spec §11–§12 on source-backed vs computed vs tutor's-own statements and
  follow its wording; note any amendment for v0.4.

### Task 3 — Seed new lessons with one cited example (small, then measure)

Hypothesis (untested): inside a lesson the model keeps citing because its earlier cited answers sit
in the context as examples; cold questions lack that (Granite cited 0.44 cold vs 1.00 in-lesson).
Add ONE short synthetic exchange at the start of a new lesson's prompt log showing a two-sentence
answer with the label attached to the specific sentence it supports, plus one calc use.
Constraints: fits the 800-token system slot (there is a test); is part of the append-only prefix
(never changes within a lesson); its evidence passage must never be citable for real questions —
reserve a label such as `[S0]` that the resolver refuses, and test that. Measure with
`python -m eval.run_turn_eval … --variants current,<new>` on the same 18 tuning questions; adopt
only if cited-and-supported improves by ≥ 0.15 without raising `evidence_dump_rate`. Record in
`docs/citation_experiment.md`.

### Task 4 — Calculator misses (investigate, then fix)

Granite got 6/8 calc items in its first 10-minute soak and never called `calc` for °C→°F. Find the
two misses (re-run a short soak; per-turn answers now dump to `data/*.turns.json`): wrong tool
arguments, no tool call, or a right call followed by wrong final text? Candidate host fixes:
mention unit conversion in the `calc` tool description (`tutor/tools/schemas.py`); have the host
flag an answer whose number disagrees with the calc result it was given.

### Task 5 — Retrieval latency (worker batching)

Tuning mean 1.6 s versus the spec's ≤ 1 s warm target: about 20 worker round-trips per request,
76% of the time in IPC wait (`docs/retrieval_baseline.md`, "Baseline v5"). Add a `multi` op to
`tutor/retrieval/zim/worker.py` (several searches / estimated-match lookups per round trip) and use
it in `research.py`. Acceptance: tuning mean ≤ 1.0 s, no recall change, deadlines still enforced.

### Smaller follow-ups (any order)

- `docs/bakeoff_dev/README.md` flags a contradiction about Q1_0's tool-call malformed rate (0/20 in
  `docs/toolcall_verification.md` vs "20/20" in the table in `ternary.md`). Almost certainly a typo
  in `ternary.md`; verify against `docs/toolcall_results.json` and fix the doc.
- Reading level: Granite drifts above 10–16 yo (abstract algebra, "hex-3-ene"). The profile's grade
  level is meant to reach the system prompt; check that it does. Measure before changing prompts.
- `student_phrasing` held-out recall@5 was 0.50, the weakest category. There is no spelling
  tolerance for a misspelt KEY term ("heluim"); title-suggestion search may help. Tune on tuning only.
- Spec v0.4 amendments to write up: two-stage eviction (uncited evidence dropped before whole
  turns) vs §8's whole-triple wording; evidence budget is a cap with a relevance cutoff; key-fact
  infobox passages; passage-ID wording (spec line 294 vs the reuse plan); the model section.
- M6 prep: Granite runs on STOCK llama.cpp, which removes the plan's biggest M6 risk (Prism-fork
  CUDA kernels on `sm_61`). `config/dell.toml` does not exist yet. If Ternary is ever revisited,
  its GGUF layout (group-64 vs legacy) must match whatever build lands on the Dell.

## 5. Open questions for the user

1. Confirm Granite replaces Bonsai in the plan (plan §0.2 and spec §3.1 name Bonsai; M6's bake-off
   was meant to choose between Q1_0 and Q2_0). Granite is Apache-2.0 — record it in
   `THIRD_PARTY_NOTICES.md` once confirmed.
2. M4: enlarge the eval set and build a fresh held-out split before revisiting hybrid retrieval?
   `docs/M4_report.md` lists the options. Not urgent — lexical plus key facts is working.
3. There is no curriculum: a "lesson" is one conversation on one subject. A guided path through
   the Lumen course archive would be new scope, not in the plan.
