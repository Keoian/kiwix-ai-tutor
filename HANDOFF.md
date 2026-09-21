# Handoff: offline school tutor (kiwix-ai-tutor)

Written 2026-09-21 ~02:30 for the session that continues after a `/clear`. You have no prior
context. Read this, then `docs/bakeoff_dev/README.md`. The previous handoff is kept at
`docs/HANDOFF_2026-09-20_afternoon.md`; the one before that at
`docs/HANDOFF_2026-09-19_original.md`. House rules still apply, restated in §2.

**You are the orchestrator. Sonnet sub-agents write the code. Tests come first.**

---

## 1. Where the project stands

Granite is now the default dev config (`config/dev.toml`; Bonsai kept as
`config/dev.bonsai-q1.toml`). This session built host-side sentence-level attribution, a
forced query-rewrite round on weak evidence, retrieval v7→v15, a computed-statement check,
generation caps, and several UI fixes. 58 commits since the last handoff (`git log --oneline
2f49b35..HEAD`). CI: Windows green; **ubuntu still fails** (see §4 task 2) — do not claim CI
is green this session.

| Report | What it holds |
|---|---|
| `docs/attribution_design.md` | Statement kinds (source-backed/computed/own-example), UI marker wording |
| `docs/attribution_measure.md` | Live 3-min soak: backed_sentence_rate 0.692; citation-rate 44/44→0/12 explained as model variance |
| `docs/citation_experiment.md` | Every citation measurement, incl. seed-exchange A/B and its reversal |
| `docs/soak_v3_analysis.md` | Long-answer cyclic loops, label spam, calc misses, unbacked_number false positives |
| `docs/calc_investigation.md` | Why Granite never calls `calc` voluntarily; ranked fixes |
| `docs/unsourced_claims_measure.md` | 0/10 unbacked specifics contradicted; n too small to trust |
| `docs/retrieval_baseline.md` | v7–v15 tables (worker batching → fallback composition) |
| `docs/retrieval_latency_profile.md` | Snippet generation is 96% of child-side time |
| `docs/worker_batching_design.md` | `multi` op design, ~21→4-6 round trips |
| `docs/rewrite_on_weak_evidence.md` | Forced rewrite mechanism, RRF merge, latency, not-found instruction |
| `docs/rewrite_probe_measure.md` | Assessor false-strong fix; probe A/B before/after |
| `docs/plan/spec_v0.4_amendments.md` | 7 proposed spec diffs, awaiting owner approval |
| `docs/bakeoff_dev/README.md` | Three-model comparison table (Bonsai Q1_0 / Ternary / Granite) |

### What changed and what it measured

- **Attribution** (`tutor/app/citations.py`, `attribute_sentences`): host splits answers into
  sentences and attributes each to the best-supporting passage independently of the model's own
  `[S#]` labels, which Granite places sloppily (one label stapled to the end of a 4–6 sentence
  paragraph). New `attributions` SSE event; UI markers ● found / ○ not found / ⚠ unbacked number
  / ✓ computed, with wording that says "found / not found in the sources the tutor looked up" —
  never claims a library-wide search happened. Measured on a live 3-min soak: `backed_sentence_rate
  0.692` (12 factual turns; one turn looked like a false negative from cross-turn retrieval
  variance — not re-tuned, flagged for more data).
- **Computed-statement check**: host re-verifies stated arithmetic with the calc evaluator
  (`tutor/tools/calc_tool`), read-only/additive. Granite still never calls `calc` voluntarily in
  lessons (confirmed: `calc_calls == 0` on every scripted arithmetic turn in two soaks, and the
  parser drops nothing — see `docs/calc_investigation.md`); this host check only catches
  disagreement in *stated* arithmetic, it does not force a tool call.
- **Generation cap (2000 tokens) + repetition guard**, extended to catch period-N (2–8) cycles:
  `docs/soak_v3_analysis.md` measured three runaway turns (11,271–11,588 chars, 66–86 s) that were
  cyclic paraphrase loops the old consecutive-repeat guard couldn't see; 60% of that soak's citation
  labels (221/365) were invented past the real evidence set, all inside those three loop turns.
- **Profile grade level reaching the system prompt**: was a real bug (never looked up); fixed
  (commit `e89f419` in the prior session's line, confirmed still wired this session — v3 soak still
  read above target level despite the pin, so wiring alone didn't fix tone; unresolved, see §4).
- **Seed-exchange prompt variant**: adopted after a single-turn A/B (`cited-and-supported` 0.093 →
  0.352, n=54, gate was ≥0.15) — **then reverted to off-by-default the same day** because in-lesson
  soaks contradicted it: per-label supported fraction was 0.0% (v2) and 6.8% (v3), both far below
  the A/B's pooled 0.352 (`docs/soak_v3_analysis.md` §3). Inferred, not confirmed, why: the A/B is
  18 cold single-turn questions primed by the seed; the soak is a real accumulating lesson with
  longer answers penalized harder by the strict overlap check. `AppConfig.prompt_variant` default
  is `"current"`.
- **Retrieval v7→v15** (`docs/retrieval_baseline.md`): worker `multi` batching (~21 round trips
  down to a design target of 4–6), per-request op memo, deferred/cached snippet text, lxml parser
  (0 mismatches over 10,869 articles vs the old parser), spelling fallback, compound split/join,
  morphological variants, and v15's fix so these fallbacks **compose** on one shared, current term
  list instead of each one working off a stale token list. Tuning recall@1/3/5 **0.595/0.690/0.762**
  and MRR **0.645** unchanged across all of v12–v15 (measured, byte-identical each time). Mean
  latency: v8 profiling found snippet-building for discarded candidates was 96.2% of child-side
  time (407.9 ms of ~424 ms per `search_fulltext` call); after fixes, tuning mean latency is
  **0.824–1.014 s warm** across v12–v15 runs (target ≤1.0 s mostly met, v15's own run measured
  1.014 s — right at the line, not cleanly under it); cold cache 1.222 s (v12). kid_phrasing
  gold-in-top-5 went 2/18 (v14) → **4/18** (v15, after the squarefoot compose fix).
- **Evidence assessor**: rarity-weighted-ish via a curated generic-word list (real IDF was scoped
  out as a larger change), topic-in-title-or-exact-phrase requirement, unknown terms count as
  uncovered. On the tuning split's 10 known misses, the assessor still calls **7 of 10 "strong"**
  (`docs/rewrite_probe_measure.md`'s confusion table, row "B False": strong=7, weak=1, empty=2) —
  the rewrite round only fires on weak/empty, so those 7 never get rewritten.
- **Forced `research` rewrite round on weak/empty evidence** (`docs/rewrite_on_weak_evidence.md`):
  append-only host note in the student's own turn text (never a second message), `tool_choice`
  forcing a `research` call with a `response_format` JSON-schema fallback, RRF merge (k=60,
  article-then-passage level) with a title-match boost and an incidental/generic-title penalty,
  re-assessment against the rewrite's own terms plus any already-healthy original terms. Setting
  `app.rewrite_on_weak_evidence` default **ON**. Added wall time settled at ~5 s (forced call 2–3 s
  + batched `research_many`, bounded by the slowest query, 3–4 s) after two follow-up fixes — the
  earlier ~8.3 s/~27 s numbers were pre-fix or a cold-run outlier, not the steady state. Prompt
  cache survives the extra turn: one live run measured turn 2's `cached_tokens` **1716** within one
  token of turn 1's `total_tokens` **1717** (not 1457/1458 — see report footer, that number does
  not appear in the repo).
- **UI**: fixed-viewport layout, single working citation chips, safe DOM-only markdown including
  tables/headings/lists, working indicator with live `status` SSE stages, "Show more of the
  article" expander (`GET /api/source/{passage_id}/context`), a muted "Searched for: ..." line.
  UI JS now **executes** under pytest via py_mini_racer (`tests/test_ui_js_exec.py`) — this catches
  real bugs the old static-only check couldn't (commit `9680c9f`'s message says fixes were made
  from what it found) — but **nobody has looked at the UI in a real browser** since these fixes;
  that is today's first task.

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
- `C:\git\bonsai` is read-only. No commands inside that directory.
- Never hard-kill `llama-server` mid-prompt (suspected AMD driver reset). Stop it only when
  `/slots` shows idle. The user has given permission to stop/restart it and the app at will.
- The spec wins on budgets/caps/thresholds; the reuse plan wins on algorithms; if they conflict, ask.
- Writing for the user: lead with the outcome, measured numbers not adjectives, say what was
  inferred versus measured, say plainly what was not verified.

### Lessons about driving Sonnet sub-agents (learned the hard way)

- **Keep tasks small.** Agents have handed back large multi-part tasks untouched more than once.
  One concern per agent; split "build" from "measure live" — every measurement doc this session
  was its own agent, separate from the doc/code that built the feature it measured.
- Tell them the unit suite takes ~3 minutes — use a shell timeout ≥ 600000 ms. Piping pytest
  through `tail` hides its exit code and this project prints no "N passed" line: capture the
  return code explicitly, don't infer pass/fail from truncated output.
- **An agent's background processes die when it hands back.** Tell every agent to run long
  operations in the **foreground**, in chunks of ≤ 9 minutes, and make any long-running driver
  script resumable (checkpoint to a file, re-run picks up where it left off) rather than relying on
  a background process surviving past hand-back.
- **Commit by explicit pathspec** (`git commit <paths>`, never `git add -A`/`git commit -a`) —
  concurrent agents' staged files got swept into each other's commits twice this session.
- **Tests must never read gitignored `data/`** — this broke CI once; scratch/measurement data
  belongs in `data/` but must stay out of the test-collected path.
- The orchestrator should **run the full suite itself** after any risky merge, not trust an agent's
  "suite green" claim — those claims were wrong or unverified several times this session.
- `gh` is not installed in this environment; the unauthenticated GitHub API works for read-only
  checks (e.g. `GET /repos/Keoian/kiwix-ai-tutor/check-runs/{job_id}/annotations` to read a CI
  job's diagnostic annotations without a token) but is capped at 60 requests/hour — budget calls.
- When two agents share the tree: forbid `git stash/checkout/restore/reset/clean` and deleting
  files in EVERY prompt, and give each agent an explicit list of files it owns.
- Tell an agent explicitly "do this yourself, no sub-agents" when the task must not be
  re-delegated further — otherwise it may spawn its own sub-agents and diffuse accountability.
- Held-out split: never let an agent run it; it has been observed enough times to be no longer clean.
- Agents' own completion reports are claims, not verification. Twice this session a "done" report
  hid an unmet acceptance criterion (a stale token list not being updated, a metric computed over a
  different denominator than claimed) that a direct orchestrator check exposed.

## 3. Machine state right now

| Thing | State |
|---|---|
| llama-server :8080 | Granite, healthy |
| Tutor app :8420 | Started by the owner ~22:00 on 2026-09-20 — **runs code from BEFORE most of this evening's server-side changes.** Restart before trusting anything it serves: stop the two python processes for it, then `python -m tutor.app.main --config config\dev.toml` from the repo root. The permission classifier blocks agents from stopping the owner's own processes — if you need a live instance for measurement instead of restarting the owner's, start a second one on another port (pattern: `data/dev.soak8421.toml`, see `docs/attribution_measure.md` for how one was started/stopped cleanly). |
| `pip install -e ".[dev]"` | Needed to pick up lxml, pytest-timeout, mini-racer — run this before trusting local test/tooling results if the venv predates today. |
| Dense index | `runtime/simplewiki_dense/` — still stale; hybrid still off (open problem #9, unchanged from before this session). |

## 4. Tasks to implement, in order

Each is sized for one agent. Split build from measure as two separate agents where a task has both.

### Task 1 — Owner's morning browser test (do this first)

Nobody has opened the UI in a real browser since the UI fixes in commits `599c58a`, `4a1b9ce`,
`019a305`, `9680c9f`. The owner tests it this morning. Be ready to receive their report and fix
what they find; do not assume the pytest-JS-exec suite being green means the browser experience is
right.

### Task 2 — CI ubuntu root cause (blocking, known open problem)

The ubuntu job fails deterministically around test ~446 with `sqlite3.OperationalError: disk I/O
error` in a store `__init__`, then stalls. The fd-exhaustion hypothesis was **refuted by
measurement** (commits `4310b1b`, `de77a20`, `f442d07`); read those commits and the CI workflow's
own diagnostic annotations before re-guessing. Read annotations unauthenticated:
`GET /repos/Keoian/kiwix-ai-tutor/check-runs/{job_id}/annotations` (60 req/h limit — budget it).
Windows is green. Retry/WAL-fallback code was added to the SQLite stores while chasing this
(`69c8ea3`/`0f0d1e1`-era commits) and **may be masking rather than fixing** the real cause — review
it critically, don't assume it's correct because tests currently pass on Windows. Check
`git log -- .github tests/conftest.py` first — a CI-focused agent may still have been mid-task when
this handoff was written.

### Task 3 — Re-run the end-to-end probe A/B after v15 + assessor fixes

`data/probe_rewrite_ab.py` is resumable. It has **not** been re-run since retrieval v15 (fallback
composition) and the assessor false-strong fix landed. The last numbers on record
(`docs/rewrite_probe_measure.md`, "AFTER" columns, 16 questions) predate both. Re-run ON vs OFF,
compare false_strong / gold_in_evidence / rewrite_rate / mean wall time against that table.

### Task 4 — Out-of-library probe set + compliance measurement

`eval/questions/out_of_library_probes.jsonl` does not exist yet. The tiered policy (explain freely
when sources were found; mark unfound specifics; say "couldn't find it" when the library has
nothing) has only been measured for the "sources were found" case
(`docs/unsourced_claims_measure.md`: 8/10 unbacked specifics were actually in the full article
anyway, 0/10 contradicted, 0/15 backed-control contradicted — but n is tiny and skews to
well-covered stock topics). The out-of-library behavior (does it actually say "couldn't find it" and
avoid inventing specifics when there really is nothing) is **not measured at all**. Build the probe
set, measure compliance, then tune the not-found instruction in
`tutor/app/agent_loop.py` (`_REWRITE_HOST_NOTE`-adjacent) if it's not compliant enough.

### Task 5 — Assessor false-strong on the 7 remaining tuning misses

`docs/rewrite_probe_measure.md`'s own recommendation #3: surface `term_matches`/`estimated_matches`
on `ResearchResponse` so `assess_evidence` can use real per-term rarity instead of the curated
~20-word generic-word list, which was deferred as a larger change this session. This is what's
needed to get past the 7/10 false-strong tuning misses in task above.

### Task 6 — Lesson-context-aware pre-search for elliptical follow-ups

Known gap: pre-search for a follow-up like "What soil do I use?" retrieves generic "Soil" rather
than the lesson's actual topic. Pre-search should fold in lesson context, not just the raw
follow-up text.

### Task 7 — Single-word phonetic spelling fallback

"fotosinthesis", "dinasors" still fail retrieval-only (status comes back `empty`): the existing
spelling fallback needs ≥1 zero-match term of length ≥4 among *several* content terms to anchor on;
a single misspelled word with nothing else in the query has no anchor. `docs/retrieval_baseline.md`
v15's own residual note flags this as pre-existing and unrelated to the v15 fix. The rewrite round
(task 3) may catch some of these already — measure before adding retrieval-side complexity.

### Task 8 — Upgrade unbacked-specific markers by checking the full article

`docs/unsourced_claims_measure.md`'s policy (c): withhold a number/name only if it's also absent
from the full top article (not just the indexed passage). Only measured on 8/10 checked specifics
(all present in the full article, 0 suppressed); 2/10 were never checked. Wire this as a real
upgrade path for the UI markers rather than a one-off measurement script.

### Task 9 — 10-minute soak with everything on

Compare against the v2/v3 soaks in `docs/soak_v3_analysis.md` now that attribution, computed-check,
rewrite-on-weak-evidence, and the repetition-cycle guard are all live together. Watch specifically
for: does the period-N guard actually stop the 11K-char cyclic loops; does the rewrite round change
the citation/attribution rates measured in `docs/attribution_measure.md`.

### Task 10 — M6 prep

`config/dell.toml` does not exist yet. Granite runs on stock llama.cpp (no Prism-fork CUDA kernels
needed), which removes M6's biggest risk from the prior plan.

## 5. Open questions for the owner

1. Confirm Granite replaces Bonsai in the plan/spec (plan §0.2 and spec §3.1 still name Bonsai).
   Record Apache-2.0 in `THIRD_PARTY_NOTICES.md` once confirmed — check first whether it's already
   recorded, this was not verified this session.
2. Approve or reject the 7 spec v0.4 amendment proposals in `docs/plan/spec_v0.4_amendments.md`
   (two-stage eviction wording, evidence-budget-as-cap, key-fact infobox passages, passage-ID
   wording, Granite-replaces-Bonsai, host-inferred attribution as a separate layer, reserved `[S0]`
   seed label).
3. Change passage chunking from 400-char word-packed chunks (cut mid-sentence,
   `tutor/retrieval/hybrid/passages.py`) to sentence boundaries? Left unchanged this session because
   it would change indexed text/eval numbers and require a dense-index rebuild — needs an explicit
   go-ahead since it invalidates the tuning baseline.
4. M4 eval-set enlargement / fresh held-out split before revisiting hybrid retrieval — not urgent,
   lexical plus key facts is working, but unaddressed again this session.
5. Curriculum scope: still no guided path through the archive, just one conversation per subject —
   confirm this remains out of scope.
