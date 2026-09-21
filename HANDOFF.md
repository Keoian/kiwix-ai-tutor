# Handoff: offline school tutor (kiwix-ai-tutor)

Written 2026-09-21 morning (updated after commits `9ed5633`, `fa3d685`, `f589170`). You have no
prior context. Read this, then `docs/bakeoff_dev/README.md`. The previous handoff is kept at
`docs/HANDOFF_2026-09-20_afternoon.md`; the one before that at
`docs/HANDOFF_2026-09-19_original.md`. House rules still apply, restated in §2.

**You are the orchestrator. Sonnet sub-agents write the code. Tests come first.**

### Do this first: owner's browser test

The owner was about to test the UI in a **real browser** on the morning of 2026-09-21. Before
trusting anything the app serves, **restart it**: stop the two python processes (the app and its
ZIM worker child), then from `C:\git\kiwix-ai-tutor` run
`python -m tutor.app.main --config config\dev.toml` (run `pip install -e ".[dev]"` first if the
venv predates today, for lxml/mini-racer). Nobody has verified layout/CSS/clicks/live streaming in
a real browser — only JS *logic* executes under pytest (`tests/test_ui_js_exec.py`), and the
orchestrator's Chrome tools reach a browser that **cannot** open the owner's `127.0.0.1`. Wait for
the owner's report and fix what they find.

### Owner preferences (unchanged)

Two-sentence status updates until told otherwise. Sub-agent driven, Sonnet-only, conserve
orchestrator context (full house rules in §2). Prior-session decisions: tiered unsourced-answer
policy adopted; host-driven rewrite lives in the lesson log (cache stability, not a second
message); lxml parser approved.

---

## 1. Where the project stands

Granite is now the default dev config (`config/dev.toml`; Bonsai kept as
`config/dev.bonsai-q1.toml`). The prior session built host-side sentence-level attribution, a
forced query-rewrite round on weak evidence, retrieval v7→v15, a computed-statement check,
generation caps, and several UI fixes. Since then this morning: the end-to-end probe A/B was
re-run on current code (rescued 6 questions, lost none), the out-of-library not-found behavior was
measured for the first time (n=14), and the evidence assessor got a v3 pass (multi-word topic
integrity, constraint terms; co-occurrence tried and rejected). See "What's new since the last
handoff" below. CI: Windows green; **ubuntu status is UNKNOWN, presumed still failing** — a CI
agent worked for hours and its final report never arrived (see §4 task 2) — do not claim CI is
green this session.

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
| `docs/rewrite_probe_measure.md` | Assessor false-strong fix; probe A/B before/after; assessor v3 labelled-set results |
| `docs/plan/spec_v0.4_amendments.md` | 7 proposed spec diffs, awaiting owner approval |
| `docs/bakeoff_dev/README.md` | Three-model comparison table (Bonsai Q1_0 / Ternary / Granite) |
| `docs/out_of_library_measure.md` | Does the tutor say "couldn't find it" when the library has nothing? n=14 |

### What's new since the last handoff (`git log --oneline e57d0a1..HEAD`)

1. **End-to-end probe A/B re-run** (`docs/rewrite_probe_measure.md`, "End-to-end probe A/B on
   current code"; commit `fa3d685`). 28 kid-phrasing/misspelling questions, live Granite. ON: gold
   in evidence 17/28, rewrite fired 10/28, backed rate **0.68**, mean wall **10.1 s**. OFF: 11/28,
   backed rate 0.39, mean wall 8.9 s. Rescued (gold absent OFF, present ON): kid07, kid14, msp02,
   msp05, msp06, msp09 — lost none; all 10 fired rewrites were correct. 8 turns/arm still rated
   "strong" with gold absent (same false-strong issue as item 3). Old open problem "(3) end-to-end
   probe A/B NOT re-run" and old task 3 are now **done**.
2. **Out-of-library probe set + measurement** (`docs/out_of_library_measure.md`,
   `eval/questions/out_of_library_probes.jsonl`; commit `9ed5633`). n=14. Said "couldn't find it"
   plainly: **8/14**. Invented nothing on 4/4 questions about non-existent things. But **5/14**
   answered confidently and wrongly because `assess_evidence` rated topically-wrong evidence
   "strong" (1994 Rose Bowl answered from the 1998 game; blue-whale flipper hairs fabricated from
   Whale/Hair pages; "coldest survived" answered with Earth's coldest recorded temp). One answer
   looped evidence passages ~6x verbatim (backed count 79) — a repetition bug, **not yet fixed**.
   Old open problem (4) and old task 4 are done as *measurement*; tuning remains (new task 4).
3. **Evidence assessor v2/v3** (commits `d86a541`, `ab10b25`, `f589170`;
   `docs/rewrite_probe_measure.md` §"Assessor confusion on the tuning split" and §"Assessor v3 —
   labelled set"). Labelled cache `data/assessor_labelled_cache.json` (84 qs, gitignored; rebuild
   `PYTHONPATH=. PYTHONIOENCODING=utf-8 python data/assessor_labelled.py --build`, score with
   `... --score`). False-strong **20 → 16**; false-weak within budget (sw42 new, msp03
   pre-existing, sw46 a labelling artifact). Adopted: multi-word topic integrity, number/roman
   equivalence, filler names in `QUESTION_SHAPE_FILLERS`. **Rejected:** same-passage co-occurrence
   (~14 new false-weak). Tuning retrieval eval re-run after `f589170` (needs `--out`, e.g.
   `python -m eval.run_retrieval_eval --registry config/archives.simplewiki_only.toml --questions
   eval/questions/simplewiki_questions.jsonl --split tuning --out data/x.md`): recall@1/3/5
   **0.595/0.690/0.762**, MRR **0.645**, mean **1.020 s**, p95 **2.687 s** — unchanged (expected;
   assessment is downstream of retrieval ranking). Remaining false-strong ids are in the doc.
   **Main quality bottleneck**: tuning misses, the probe A/B, and out-of-library all trace
   wrong-but-confident answers to assessor false-strong.

### What changed and what it measured (prior session, before `e57d0a1`)

- **Attribution** (`tutor/app/citations.py`, `attribute_sentences`): host splits answers into
  sentences and attributes each to the best-supporting passage independently of the model's own
  `[S#]` labels, which Granite places sloppily (one label stapled to the end of a 4–6 sentence
  paragraph). New `attributions` SSE event; UI markers ● found / ○ not found / ⚠ unbacked number
  / ✓ computed, with wording that says "found / not found in the sources the tutor looked up" —
  never claims a library-wide search happened. Measured on a live 3-min soak: `backed_sentence_rate
  0.692` (12 factual turns; one turn looked like a false negative from cross-turn retrieval
  variance — not re-tuned, flagged for more data).
- **Computed-statement check**: host re-verifies stated arithmetic with the calc evaluator
  (`tutor/tools/calc_tool`), read-only/additive. Granite still never calls `calc` voluntarily
  (`calc_calls == 0` every scripted turn — `docs/calc_investigation.md`); only catches disagreement
  in *stated* arithmetic, doesn't force a tool call.
- **Generation cap (2000 tokens) + repetition guard**, extended to catch period-N (2–8) cycles:
  `docs/soak_v3_analysis.md` measured three runaway turns (11,271–11,588 chars, 66–86 s) of cyclic
  paraphrase loops the old consecutive-repeat guard couldn't see. (The ool10 repetition bug found
  this morning, task 4 below, looks like a related-but-distinct gap in this same guard family.)
- **Profile grade level reaching the system prompt**: fixed (commit `e89f419`); v3 soak still read
  above target level despite the pin — wiring alone didn't fix tone, unresolved.
- **Seed-exchange prompt variant**: adopted after a single-turn A/B, then **reverted to
  off-by-default the same day** because in-lesson soaks contradicted it (`docs/soak_v3_analysis.md`
  §3). `AppConfig.prompt_variant` default is `"current"`.
- **Retrieval v7→v15** (`docs/retrieval_baseline.md`): worker `multi` batching (~21→4-6 round
  trips), per-request op memo, deferred/cached snippet text, lxml parser (0 mismatches/10,869
  articles), spelling fallback, compound split/join, morphological variants composing via v15's
  fix. Tuning recall@1/3/5 0.595/0.690/0.762, MRR 0.645, unchanged through v12–v15 and still after
  `f589170` (assessment is downstream of ranking). Mean latency 0.824–1.014 s warm; cold 1.222 s.
  kid_phrasing gold-in-top-5 2/18 (v14) → 4/18 (v15).
- **Evidence assessor (v1)**: rarity-weighted-ish via a curated generic-word list, topic-in-
  title-or-exact-phrase requirement. Superseded by v2/v3 this morning — see "What's new" and task 3.
- **Forced `research` rewrite round on weak/empty evidence** (`docs/rewrite_on_weak_evidence.md`):
  append-only host note in the student's own turn (never a second message), forced `research` tool
  call, RRF merge (k=60) with title-match boost and generic-title penalty, re-assessment against
  the rewrite's terms plus healthy original terms. `app.rewrite_on_weak_evidence` default **ON**.
  This morning's fuller end-to-end soak (item 1 above) measured 10.1 s mean wall with rewrite on vs
  8.9 s off across 28 questions, 10 rewrites fired.
- **UI**: fixed-viewport layout, working citation chips, safe DOM-only markdown, live `status` SSE
  stages, "Show more of the article" expander, a muted "Searched for: ..." line. UI JS **executes**
  under pytest via py_mini_racer (`tests/test_ui_js_exec.py`) — but **nobody has looked at the UI in
  a real browser** since these fixes; that is task 1 below (and the box at the top of this file).

## 2. Working agreement (unchanged — the user's hard constraints)

- **Sub-agents: `model: "sonnet"` on every `Agent` call. Never `subagent_type: "fork"`.**
- **TDD:** failing test first; the orchestrator verifies by running things, not by trusting reports.
- **Conserve orchestrator output and context.** Delegate anything a sub-agent can do, including
  commits and pushes. Ask sub-agents for short final reports.
- **Pushing is pre-authorised** (user, 2026-09-19). Commit as Keoian's noreply address (already set
  repo-locally). End commit messages with the attribution lines the session reminder gives you.
- **Never delete files.** Move junk to `D:\_trash_kiwix-ai-tutor\` and append a line to its
  `TRASH_LOG.txt` (timestamp | original | new | why). An autoclassifier blocks deletes.
- `C:\git\bonsai` is read-only. No commands inside that directory.
- Never hard-kill `llama-server` mid-prompt (suspected AMD driver reset). Stop it only when
  `/slots` shows idle. The user has given permission to stop/restart it and the app at will.
- The spec wins on budgets/caps/thresholds; the reuse plan wins on algorithms; if they conflict, ask.
- Writing for the user: lead with the outcome, measured numbers not adjectives, say what was
  inferred versus measured, say plainly what was not verified.

### Lessons about driving Sonnet sub-agents (learned the hard way)

- **Keep tasks small — this keeps recurring.** Agents have repeatedly handed back large multi-part
  tasks **untouched**, citing budget, including several this morning. Split build / measure / doc
  into separate small agents; every measurement doc so far was its own agent, separate from the
  code that built the feature it measured. Tell each agent to start coding within its first few
  tool calls, and say explicitly "do this yourself, no sub-agents" when re-delegation isn't wanted.
  Partial, committed, green progress beats an untouched hand-back.
- Tell them the unit suite takes ~3 minutes — use a shell timeout ≥ 600000 ms. Piping pytest
  through `tail` hides its exit code and this project prints no "N passed" line: capture the
  return code explicitly, don't infer pass/fail from truncated output.
- **An agent's background processes die when it hands back.** Tell every agent to run long
  operations in the **foreground**, in chunks of ≤ 9 minutes, and make any long-running driver
  script resumable (checkpoint to a file, re-run picks up where it left off) rather than relying on
  a background process surviving past hand-back.
- **Commit by explicit pathspec** (`git commit <paths>`, never `git add -A`/`git commit -a`) —
  concurrent agents' staged files got swept into each other's commits twice. **Tests must never
  read gitignored `data/`** — this broke CI once; scratch/measurement data must stay out of the
  test-collected path.
- The orchestrator should **run the full suite itself** after any risky merge, not trust an agent's
  "suite green" claim — those claims were wrong or unverified several times this session.
- `gh` is not installed; the unauthenticated GitHub API works read-only (e.g.
  `GET /repos/Keoian/kiwix-ai-tutor/check-runs/{job_id}/annotations`) but is capped at 60 req/h —
  budget calls (it was exhausted once this session already, see task 2).
- When two agents share the tree: forbid `git stash/checkout/restore/reset/clean` and deleting
  files in EVERY prompt, and give each agent an explicit list of files it owns. Held-out split:
  never let an agent run it — observed enough times to be no longer clean.
- Agents' own completion reports are claims, not verification. Twice a "done" report hid an unmet
  acceptance criterion (a stale token list not updated, a metric over a wrong denominator) that a
  direct orchestrator check exposed. Long-running live measurement runs are best started by the
  **orchestrator itself** in a background shell, which survives across turns — an agent's
  background job dies the moment it hands back.

## 3. Machine state right now

| Thing | State |
|---|---|
| llama-server :8080 | Granite, healthy |
| Tutor app :8420 | Stale — predates this morning's assessor v3 / out-of-library changes. Restart before trusting anything it serves (see "Do this first" box at the top). The permission classifier blocks agents from stopping the owner's own processes — if you need a live instance for measurement instead, start a second one on another port (pattern: `data/dev.soak8421.toml`, see `docs/attribution_measure.md`). |
| `pip install -e ".[dev]"` | Needed for lxml, pytest-timeout, mini-racer — run before trusting local results if the venv predates today. |
| Dense index | `runtime/simplewiki_dense/` — still stale; hybrid still off (unchanged, open question 4 below). |

## 4. Tasks to implement, in order (re-ordered this morning)

Each is sized for one agent. Split build from measure as two separate agents where a task has both.

### Task 1 — Fix whatever the owner reports from the browser test (do this first)

Nobody has opened the UI in a real browser since the UI fixes in commits `599c58a`, `4a1b9ce`,
`019a305`, `9680c9f`. See the "Do this first" box at the top of this file for the restart
procedure. Be ready to receive the owner's report and fix what they find; do not assume the
pytest-JS-exec suite being green means the browser experience is right.

### Task 2 — CI ubuntu root cause (blocking, known open problem)

The ubuntu job fails deterministically around test ~446 with `sqlite3.OperationalError: disk I/O
error` in a store `__init__`, then stalls. The fd-exhaustion hypothesis was **refuted by
measurement** (commits `4310b1b`, `de77a20`, `f442d07`). A CI-focused agent then worked for hours
adding retry/backoff to all three SQLite stores plus more diagnostics — but its **final report
never arrived**, and the orchestrator could not read the latest run because the unauthenticated
GitHub API (60 req/h) was exhausted. **State of ubuntu CI is UNKNOWN, presumed still failing.**
Windows is green. First step next session: read the latest run's jobs + annotations (URLs/recipe
above), then critically review whether the SQLite retry code the agent added is masking a real
problem rather than fixing it — don't assume it's correct just because Windows currently passes.
Check `git log -- .github tests/conftest.py tutor -- '*sqlite*'` for what actually landed.

### Task 3 — Assessor false-strong (the main quality bottleneck)

Tuning misses, the probe A/B, and out-of-library measurement (see "What's new" above) all trace
wrong-but-confident answers to the assessor rating topically-wrong-but-lexically-similar evidence
"strong." Ideas from `docs/rewrite_probe_measure.md` and `docs/out_of_library_measure.md`: (a)
expose real per-term match counts (`term_matches`) on the research response for rarity weighting
instead of the curated generic-word list; (b) add year/entity constraint terms (catches the
1994-vs-1998 Rose Bowl case); (c) a host-side **post-check**: when a finished answer has specifics
(numbers/dates/names) with zero backed sentences, show a stronger caution and/or regenerate once
with the not-found instruction. Co-occurrence was tried and **rejected** — don't re-try it as-is.

### Task 4 — Evidence-dump / verbatim-repetition guard

The ool10 "biggest number" probe answer repeated all 5 evidence passages ~6x verbatim (backed count
reported as 79) — `docs/out_of_library_measure.md` §"Investigate the ool10 repetition bug" flags a
decoding/dedupe bug in the citation or streaming path, distinct from not-found-compliance and
**not yet fixed**. Reproduce it and fix or extend the repetition-cycle guard from the prior session
(`docs/soak_v3_analysis.md`).

### Task 5 — Lesson-context-aware pre-search for elliptical follow-ups

Known gap: pre-search for a follow-up like "What soil do I use?" retrieves generic "Soil" rather
than the lesson's actual topic. Fold lesson context into pre-search, not just the raw follow-up text.

### Task 6 — Single-word phonetic spelling fallback

"fotosinthesis", "dinasors" still fail retrieval-only (status comes back `empty`): the spelling
fallback needs ≥1 zero-match term of length ≥4 among several content terms to anchor on; a single
misspelled word alone has no anchor (`docs/retrieval_baseline.md` v15's residual note). The model
rewrite already rescues these end-to-end at ~5 s (task 3's mechanism; the probe A/B's kid07/msp05/
msp06/msp09 rescues are this class) — measure whether retrieval-side complexity is worth adding.

### Task 7 — Upgrade unbacked-specific markers by checking the full article

`docs/unsourced_claims_measure.md`'s policy (c): withhold a number/name only if it's also absent
from the full top article (not just the indexed passage). Only measured on 8/10 checked specifics
(all present in the full article, 0 suppressed); 2/10 were never checked. Wire this as a real
upgrade path for the UI markers rather than a one-off measurement script.

### Task 8 — 10-minute soak with everything on

Compare against the v2/v3 soaks in `docs/soak_v3_analysis.md` now that attribution, computed-check,
rewrite-on-weak-evidence, the repetition guard, and assessor v3 are all live — **not run since the
rewrite landed**. Watch: does the period-N guard stop 11K-char cyclic loops (and ool10-style
evidence-dump repetition, task 4); does the rewrite round change rates in `docs/attribution_measure.md`.

### Task 9 — M6 prep

`config/dell.toml` does not exist yet. Granite runs on stock llama.cpp (no Prism-fork CUDA kernels
needed), removing M6's biggest risk from the prior plan.

## 5. Open questions for the owner

1. Confirm Granite replaces Bonsai in the plan/spec (plan §0.2 and spec §3.1 still name Bonsai).
   Record Apache-2.0 in `THIRD_PARTY_NOTICES.md` once confirmed — check first whether already recorded.
2. Approve or reject the 7 spec v0.4 amendment proposals in `docs/plan/spec_v0.4_amendments.md`
   (two-stage eviction wording, evidence-budget-as-cap, key-fact infobox passages, passage-ID
   wording, Granite-replaces-Bonsai, host-inferred attribution as a separate layer, reserved `[S0]` seed label).
3. Change passage chunking from 400-char word-packed chunks (cut mid-sentence,
   `tutor/retrieval/hybrid/passages.py`) to sentence boundaries? Unchanged — would require a
   dense-index rebuild and invalidate the tuning baseline; needs explicit go-ahead.
4. M4 eval-set enlargement / fresh held-out split before revisiting hybrid retrieval — not urgent,
   lexical plus key facts is working, but still unaddressed.
5. Curriculum scope: still no guided path through the archive, just one conversation per subject — confirm this remains out of scope.
