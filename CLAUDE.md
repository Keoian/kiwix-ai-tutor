# kiwix-ai-tutor — mission and ground rules

## Mission

An **offline tutor for a student with no internet**. Everything is local: an offline model
(`llama-server` running a small model; Ling 3.0 Tiny is the current default dev model,
`config/dev.toml`, since 2026-09-21, with Granite 4.0 H-Tiny as the fallback config,
`config/dev.granite.toml`), an offline Kiwix ZIM library
(Wikipedia, OER textbooks, Stack Exchange), and an app that retrieves evidence from that library,
gives it to the model as citable sources, and **teaches rather than just answers**.

The students are children. Expect bad spelling, bad grammar, and elliptical follow-ups
("is it a molecule?"). The tutor must cope with that rather than rely on well-formed input.

## Deployment assumptions (owner-stated, 2026-09-21 — do not design against these)

- **The server is the machine with the interactive session.** One box runs the model, the library,
  the app and the browser. There is no network and no remote client.
- **One student at a time.** Lessons never interleave. `llama-server` stays at one slot
  (`parallel = 1`). A new lesson wiping the previous lesson's KV cache costs nothing; the prompt
  cache only matters *within* one lesson, which is why the lesson prompt log is **append-only**.
  Do not propose multi-slot serving, slot pinning, per-user caches or other concurrency work.
- Measurement scripts must run lessons **strictly sequentially** — parallel runs against the single
  slot produce spurious results.
- **Windows and Linux are both shipping targets.** Development is on Windows; the delivery machine
  is a Dell laptop with a GTX 1060 Max-Q 6 GB (README says Linux Mint). Old GPUs are slow at large-prompt prefill, so
  keep prompts cache-friendly.

## Minimum hardware

- **Minimum: a GPU with 6 GB of VRAM.** The delivery target is a **Dell laptop with a GTX 1060 Max-Q
  (6 GB)** (owner-stated 2026-09-21) — a power-limited mobile Pascal part, slower than this dev
  machine's card; that laptop is the floor. The model,
  its context and the KV cache must fit in that budget — do not make choices that need more.
- It **can run on CPU only, but it will be slow**. CPU-only is a fallback, not a target.

## What "good" means here

- Answers are grounded in the library. The host attributes each sentence to a source independently
  of the model's own labels and marks what was found, not found, computed, or an unbacked number.
  The host never edits the model's text; the UI renders with `textContent` only.
- When the library has nothing, the tutor says so plainly instead of answering confidently from its
  own head. A wrong-but-confident answer is the worst outcome.
- Two model-facing tools only: `research` and `calc`.
- Measured numbers over adjectives; say what was inferred versus measured and what was not verified.

## Where things are governed

- `docs/plan/offline_tutor_implementation_plan.md` — primary: milestones and order of work.
- `docs/plan/offline_tutor_spec_v0.3.md` — wins on budgets, caps and thresholds.
- `docs/plan/offline_tutor_kiwix_reuse_plan.md` — wins on algorithm choices.
- `HANDOFF.md` — current state, measured numbers, the working agreement and task order. Read it first.
- `README.md` — layout and development commands.

## Hard rules for any agent working here

- Never delete files; move junk to `D:\_trash_kiwix-ai-tutor\` and log it in its `TRASH_LOG.txt`.
- `C:\git\bonsai` is read-only; run no commands inside it.
- Never hard-kill `llama-server` mid-prompt; stop it only when `/slots` shows idle. Never stop the
  owner's running app — use a second in-process instance or another port for measurement.
- Never run the held-out eval split.
- No `git stash/checkout/restore/reset/clean/revert`. Commit by explicit pathspec.
- Tests must never read gitignored `data/`; scratch files live only in `data/`.
- TDD. The unit suite takes ~4 minutes and prints no "N passed" line — capture the return code.
- `tutor/app/` never imports `tutor/retrieval/zim/` directly; go through `tutor/retrieval/research.py`.
