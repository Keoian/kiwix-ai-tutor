# Handoff: offline school tutor (kiwix-ai-tutor)

Written 2026-09-21, end of session (46 commits since `ec6d46d`; `git log --oneline
ec6d46d..HEAD`). You have no prior context. Read this, then §2. The previous handoff
(this morning) is archived at `docs/HANDOFF_2026-09-21_morning.md`; before that
`docs/HANDOFF_2026-09-20_afternoon.md`; before that `docs/HANDOFF_2026-09-19_original.md`.

**You are the orchestrator. Sonnet sub-agents write the code. Tests come first.**

## Do this first: real-browser verification, none of today's UI landed live

The owner restarted the app on `0a3ced6` and was about to test in a **real browser**.
Nobody — not the owner, not any agent — has verified today's UI changes in a real
browser: the status line lit continuously from Send to first token with per-stage
timings (`7a5de27`, `8d769e8`), the seconds counter, the new "conversion checked" wording
(`fe67847`), hidden fake `[S#]`/`[Sources: ...]` labels (`8f26f82`, `334b7ab`, `f0f8755`),
and the host-decline reply path (`2827e69`, `route: "declined"`). Only JS *logic* runs
under pytest (`tests/test_ui_js_exec.py`); the orchestrator's browser tools cannot reach
the owner's `127.0.0.1`. The orchestrator verified `0a3ced6` (the state the owner is
testing) with a full `python -m pytest -q` — **rc=0, live tests included** — and
`ruff check .` clean. Wait for the owner's report and fix what they find; do not assume
green tests mean the browser experience is right.

## Evening session (2026-09-21): host-driven "Did you mean X?" for unknown words

**Trigger case (owner, live):** "what is a ardweeno" (= Arduino) got "I couldn't find
anything". Root cause was NOT the retrieval-v16 filter changes: `ardweeno`→`arduino` is
Damerau distance 3 (spelling fallback ceiling is 2) and Arduino never enters the
title-suggest candidate pool (every prefix down to `ardw` suggests nothing; `ard` returns
20 titles without it). And Ling, inside the tutor prompt with thinking off, does not
recognise the word (it proposed "ardeweno"/"ardwolf"/"Arctic tern" depending on framing).

**What landed (uncommitted at the time of writing — see git log for the commit):**
- `ResearchResponse.unknown_terms` (`tutor/retrieval/research.py`): zero-match content
  terms that neither the spelling nor the compound fallback could fix. Verified on the
  real archives: `ardweeno`, `fotosinthesis`, `vaxeens`, `conputer` flagged; `arduino` not.
- `tutor/app/clarify.py` + `agent_loop.py` (`app.clarify_unknown_words`, default True):
  after the raw pre-search, if the first unknown term has non-strong evidence, the host
  asks the model ONE constrained question in a **fixed, system-less, single-message
  context** (`A kid typed: "<msg>". The word '<w>' is misspelled. Which real thing did
  they most likely mean by it? Reply exactly as: NAME | five-word description.`), then a
  bare-word framing as a second try. A candidate is only offered if the library has an
  article with that exact title (checked via `research_engine.research(name)` — no zim
  import). Reply is host-written: `I don't know the word "ardweeno". Did you mean
  Arduino, <description>? Say yes, or type the right word — or say no and tell me what
  it is.` Route `clarify`, ~1–2 s, no search, no answer round.
- Pending-state handling on the next turn: clean yes/no matched by a tiny host list
  (`clarify.classify_reply`; legitimate because the host itself asked a yes/no question —
  not a follow-up detector). "yes" → the normal turn runs on the message with the word
  replaced ("what is Arduino"). "no" → "OK. Tell me what a "ardweeno" is or does, in your
  own words." Anything else (a description, "no its the thing for robots", a re-typed
  word) → re-ask the model with the owner's framing *You asked the kid "Did you mean
  Arduino…?" and they answered "…"*: a new gated candidate → ask again (max 2 offers);
  the same candidate from a ≥4-word description → re-offer it once ("From that, the
  closest thing in the library is still Arduino. Say yes…"); a short reply → fresh normal
  turn (so typing "arduino" just works). A given-up word is remembered per lesson so the
  fallback turn does not immediately re-ask.
- `tutor/app/main.py`: `tutor.*` loggers now emit INFO when run directly, so the terminal
  shows `presearch unknown_terms=[...]` and `clarify word=... raw=... -> ...` per turn.

**Why the fixed context (measured, `data/ardweeno_framings.py`):** the owner saw it work
once and fail twice in a browser. Mirrored exactly (`data/ardweeno_bare.py`: bare
`/api/session` sessions ×3 in one process) — the clarify call, when it rode the lesson
log, returned `Arduino` in a lesson session (seed exchange present) and `Ardenna | A
seabird` in a bare session; 16 kid misspellings × 3 framings: sentence framing with NO
system prompt gated 11/16 real titles, the same framing under a one-line system prompt
lost `ardweeno`. After the change: bare sessions 3/3, lesson sessions 3/3, and the
no→describe→re-offer→"yeah" branch resolves to a real Arduino answer.

**Owner-verified in a real browser (2026-09-21 evening, after `8fa1653`):** "what is
ardweeno" → Did you mean Arduino → "No" → describe → "its a computer" → re-offer →
"yes" → fully sourced Arduino answer. Both branches confirmed live by the owner.

**Prompt wording attempt that was tried and reverted:** telling the model in the
weak/empty tool text and a system-prompt bullet to ask "Did you mean X?" itself. Live it
produced the right *shape* with useless content ("did you mean 'ardeweno'?") and, worse,
model-side "did you mean Arduino?" questions the host had no state for, so "yeah"
dead-ended. Removed; the host owns this now.

**Findings for the next session (not fixed):**
1. **A page refresh calls `POST /api/session`, which creates a bare session with no
   lesson** — turns are never persisted (`data/lessons.sqlite` had 15 lessons, 0 turns),
   no seed exchange, no profile summary. "New lesson" only inserts a lesson row and
   never switches the chat to it (owner: "seemingly doesn't do anything"). Owner also
   wants lessons **named** (dozens of "general" rows, no rename). UI job.
2. **Worst-case output observed live:** with the clarify gate failing, the normal path on
   "what is ardweeno" (turn 3 of the bare probe) had the model write queries
   `Ardweeno Raspberry Pi`, the assessor rated the Raspberry Pi hits **strong**, and the
   answer confidently invented "Ardweeno is a custom firmware image for the Raspberry
   Pi 4 B". The assessor certifies "strong" when the unknown head term is absent from
   every passage — that co-occurrence check should require the head noun.
3. **Code blocks get sentence attribution marks** (owner's live lesson: `delay(1000);⚠`
   flagged as an unbacked number inside a ```cpp block). Attribution should skip fenced
   code entirely.
4. A "look it up on the official Arduino website" suggestion was marked ● (backed) because
   "Arduino" is in the evidence — but the student has no internet. Same attribution pass
   as item 3.
5. The pre-search on "what is ardweeno" corrected the model's own `ardweno`→`arduino`
   (run 2 in the owner's browser) and still answered "not found" — unexplained.

## Owner decisions today (2026-09-21)

- **One student at a time, one offline machine; the server IS the interactive machine.**
  "The server is the machine with the interactive session" — no concurrency work, ever
  (no multi-slot serving, slot pinning, per-user caches). `parallel = 1` stays.
- **Delivery target = Dell laptop, GTX 1060 Max-Q, 6 GB VRAM.** Minimum hardware is a 6 GB
  GPU; CPU-only works but is slow — a fallback, not a target.
- **Ling 3.0 Tiny is the default dev model.** "Let's set default to Ling" — `config/dev.toml`
  now is the Ling profile; Granite 4.0 H-Tiny moved to `config/dev.granite.toml` as the
  fallback.
- **The LLM writes every search, even if slower.** `model_writes_search` default True,
  including turn 1 — accuracy gain was worth the added wall time (see §3a).
- **No word-list heuristics for detecting follow-ups.** Kids' spelling defeats them; three
  attempts to make the model self-declare "no search needed" all failed too (see §3f).
- **"The blue dots ARE the citations."** Stop asking the model to write `[S#]` labels —
  the host's own sentence-level attribution already does the job regardless of whether the
  model labels anything.
- **System-prompt budget 800 -> 2000 tokens** (spec v0.4 amendment item 8), so the
  search-writing / no-specifics / body-topics worked examples fit without trimming; paid
  once per lesson via the prompt cache, not per turn.
- **No-specifics-without-source rule, with exemplars**: names/numbers/dates only from the
  library, never from memory, on weak/empty evidence.
- **Body/sex topics: clinical, library-only, mention a parent.** Decline outright — and
  perform **no search at all** — for explicit/record-style requests, even under a
  jailbreak/role-play wrapper. No value judgements either way, even when a source has them.
  Verbatim: "Clinical and library-only, and mention asking a parent. I don't want the model
  to confidently exclaim inappropriate sexual content to my kids, even if they try to
  jailbreak it. It should politely decline and then not perform a search. If they have good
  enough reasoning, then it should perform a search but it should still answer without any
  judgement and in as factual and dry a way as possible."
- **Ubuntu CI is handed to a separate Linux agent** — do not work on it here.
- **Hold CI/full-suite churn while the owner iterates on models.**

## 3. What changed today

### (a) Follow-up handling
Model-written search fires on every turn including turn 1 (`app.model_writes_search`,
default True since `950eef2`): live measurement (`docs/model_writes_search_measure.md`,
8 lessons/2 arms/2 reps/80 turns) found `gold_in_evidence` 25/40 -> 34/40 (62.5% -> 85.0%)
and mean wall time 7.39s -> 9.08s (**+1.69s/turn**), controls unregressed (backed rate
drops only 0.058, inside the 0.1 tolerance). Restate-question-last (`cbd8791`, default
True) killed a stray leading "Yes," on open follow-up turns: 9/10 -> 1/10
(`docs/followup_answer_shape.md`). Passage-reuse pointers instead of re-pasting held
evidence (`e516be2`/`9f1185d`) cut prompt tokens ~16% by turn 6 (6090 vs 7260,
`docs/passage_reuse.md`). Duplicate `[S#]` labels in forced-rewrite packets fixed
(`5db7277`). A "concise follow-up note" (`8dd0918`) was adopted then **reverted**
(`6c5d83c`): wording alone over-corrected. A standalone `question` field was added to the
forced tool call so the model can resolve pronouns/referents before writing keyword
queries (`1d1cde6`/`612c21f`) — live smoke found Ling never actually populates it, only
`queries` (negative result, `docs/rewrite_on_weak_evidence.md`).

### (b) Retrieval / assessor
Retrieval v16 (`2e62c35`): measure/superlative modifiers ("biggest", "fastest") excluded
from the topic-candidate pool, junk titles ("The Biggest Loser", "Long Island") stopped
appearing in evidence; tuning recall@1/3/5 unchanged at 0.595/0.690/0.762 (assessment is
downstream of ranking, not retrieval). Assessor v4 (`cf48e81`): a co-occurrence gate
requiring the head noun and a superlative cue in the *same* passage before certifying
"strong" evidence for a superlative question; labelled-set regression unchanged
(false-strong 16, false-weak 3 — same ids before/after). Root cause found for why
"longest river"-style questions returned empty: see `docs/rewrite_probe_measure.md`.

### (c) Trust / attribution
Evidence-dump false collapse fixed — a normal short cited answer was mis-flagged as a
dump (`186b2e3`). Invented `[S#]`-shaped labels and fake `[Sources: ...]`/`[citation
needed]` blocks are now hidden from the UI rather than rendered as if real (`8f26f82`,
`334b7ab`, `f0f8755`). The source-backed mark (`is_supported`) now gates on specifics —
once a sentence states a number or a proper name, every such specific must appear
verbatim in the current turn's evidence text, not just loose word overlap
(`141bfb8`, `docs/attribution_design.md` — this is what caught the invented "Vitus
Andronicus" / "-70C" case). "Conversion checked" wording stops a bare "checked by the
calculator" phrase from reading as an independent fact-check (`fe67847`). A Unicode
minus sign used to crash `computed_check` and fail the whole turn with no answer at all —
fixed, checkers now never fail a turn (`bfb9954`).

### (d) Prompt cache
The forced round and the answer round used to send two different `tools` schemas, which
llama-server renders near the top of the prompt — every call re-read the whole lesson
from scratch. Unified into one `research` schema sent byte-identical on every call in a
turn (`84f24a6`); live-verified on Ling that `cached_tokens` grows monotonically
call-over-call instead of resetting. Research guidance (worked examples, the
`needs_search` rule) moved out of the per-turn host note and into the system prompt, read
once per lesson via the cache — the per-turn note dropped from ~360 tokens/turn to under
60 (`2d8f39d`). The host note was further scoped to the research tool call only, so an
obedient model does not echo `question`/`queries` back into its visible answer
(`c21e9f0`).

### (e) UI
A status line now stays lit continuously from Send to first token, with per-stage timings
threaded through (`7a5de27`); a tool event no longer clears the working status bubble
mid-turn (`8d769e8`). Not verified in a real browser (see the box above).

### (f) Decision path: search, skip, or decline
Settings and a decision table for skip/question/backfill landed (`2cd378c`, `2002053`,
`5bba3bb`). **Three separate attempts failed** to get the model itself to decide "no
search needed" or "decline" reliably: a `needs_search` boolean with worked examples, more
worked examples added to the system prompt, and a separate `reply_directly` tool with
`tool_choice: "required"` — all failed on "What's the longest human penis?", which always
went to `research` and got answered with measurements even with that exact case given as
a literal example. Because of this, the decision moved out of the model entirely: the
host topic gate (`2827e69`) classifies every message as decline/chat/normal in pure host
code before any search or LLM call — declines cost 0 LLM calls, 0 searches, ~0.02s; chat
messages skip search and ask the model once.

### (g) No-specifics rule + body-topics section
`app.no_specifics_without_source` and `app.child_safe_body_topics` added, both default
True (`249e945`), with a live check on six lessons (`00f6e49`,
`docs/rewrite_on_weak_evidence.md` "No specifics without a source"). Results: the
weak/empty evidence tail wording measurably held (no invented specifics, ended with a
lookup suggestion on 3 of 3 such turns) — but the **strong-evidence path has no such
guard and depends solely on the model following the system-prompt section**, which it did
not on the very first live turn tested (invented "Vitus Andronicus" and "-70C" even with
strong evidence and the new section present). This is why the strong-evidence tail later
got its own no-specifics reminder line (`2002053`, "Bug 3" in the decision-path job).

### (h) Ling 3.0 Tiny trial
`config/dev.ling.toml` added (`612c21f`), thinking switched off via `--reasoning-budget 0`
(`ecbd46e`) — the JSON `--chat-template-kwargs` form loses its quotes under Windows
PowerShell 5.1 and the server refuses to start, so the flag form is required on this
platform. Measured on this dev machine only (not the Dell): **~4.63 GiB** model share of
VRAM at 32K context, **~25 tok/s** writing/generation, **~40-200 tok/s** reading/prompt
processing (varies with depth). A build limitation was hit trying mixed K/V cache types;
not independently re-verified with numbers in this session's docs, so treat as owner/agent
observation rather than a measured table row.

### (i) Defaults flipped today
System-prompt budget 800 -> 2000 (`fd70bdf`, spec v0.4 amendment item 8; history/margin
rescaled to keep the same ~80/20 split); assembled system prompt measured against this
budget by `tests/test_citations.py`/`tests/test_seed_exchange.py` (not measured here as a
single number in a doc — the test only asserts it stays <= 2000). Ling made the default
dev model (`f26f3b6`). `model_writes_citations` default False — stop asking the model to
write `[S#]` labels (`c24fdf6`; owner: "I don't think we need the [S1] stuff"). Eviction
fixture fixed after the budget raise (`0a3ced6`) — full suite green, rc=0, live tests
included.

## 4. Machine state right now

| Thing | State |
|---|---|
| llama-server :8080 | Ling 3.0 Tiny running (Granite stopped), one slot (`parallel=1`) |
| Tutor app :8420 | The owner's — **never stop it**. For measurement, start a second instance on another port (pattern: `config/dev.soak8421.toml`-style, see `docs/attribution_measure.md`) |
| Dense index | `runtime/simplewiki_dense/` — still stale, unchanged |

Scratch measurement scripts worth keeping in `data/` (gitignored, not committed):
`data/ling_cache_probe.py`, `data/ling_check.py`, `data/ling_forced_raw.py`,
`data/ling_forced_stream.py`, `data/ling_prefill.py`, `data/ling_skip_smoke.py`,
`data/ling_status_trace.py`, `data/ling_toolchoice_probe.py`, `data/ling_turn.py`,
`data/mws_measure.py`, `data/restate_measure.py`, `data/followup_shape_measure.py` (and
its `2_wc`/`2_wd`/`2_we`/`2_wf` variants), `data/topic_gate_live_check.py`.

## 5. Working agreement (unchanged — the user's hard constraints, plus today's lessons)

- **Sub-agents: `model: "sonnet"` on every `Agent` call. Never `subagent_type: "fork"`.**
- **TDD:** failing test first; the orchestrator verifies by running things, not by
  trusting reports.
- **Conserve orchestrator output and context.** Delegate anything a sub-agent can do,
  including commits and pushes. Ask sub-agents for short final reports.
- **Pushing is pre-authorised** (user, 2026-09-19). Commit as Keoian's noreply address.
  End commit messages with the attribution lines the session reminder gives you.
- **Never delete files.** Move junk to `D:\_trash_kiwix-ai-tutor\` and append a line to
  its `TRASH_LOG.txt`.
- `C:\git\bonsai` is read-only. No commands inside that directory.
- Never hard-kill `llama-server` mid-prompt. Stop it only when `/slots` shows idle.
- **Commit by explicit pathspec only** (`git commit <paths>`). **Never `git add -A`/
  `git commit -a`/`--amend`** — an agent did both today and commingled another agent's
  edits into its own commit.
- **One agent, one file set.** List the exact files each agent owns in every brief; two
  agents sharing a file set collided again this session.
- **One job per agent, not a multi-part brief.** Agents ran out of budget on multi-job
  briefs repeatedly today — split build/measure/doc into separate small agents, same as
  the prior session's lesson, and it is still worth repeating because it still happened.
- **Live measurements must be strictly sequential on the single slot.** Parallel runs, or
  the owner chatting against the same server during a measurement, produce spurious
  "not found" results and meaningless timings.
- **Starlette's `TestClient` buffers SSE** — per-event timestamps recorded from it are
  meaningless. Use the `timings` dict in the `done` event instead.
- **`llama-server`'s `timings.cache_n` (and the client-side `prompt_n`/`cached_tokens`
  bookkeeping) is what proves cache reuse** — not visual inspection of latency alone.
- **An agent's own transcript/output file must not be read by the orchestrator** — trust
  the agent's final report, don't re-read its raw tool output.

## 6. Known open problems, in priority order

1. **The decline reply gets parroted onto later answers in the same lesson.** Seen live:
   after a decline, unrelated later answers in the same lesson (puberty, fastest-animal)
   ended with the decline sentence tacked on. Idea: keep declined exchanges out of the
   model-visible log entirely, or mark them host-only so the model never sees them as
   something to echo.
2. **Ling never sets `needs_search: false` / never picks a direct-reply path** for
   "How fast am I?"-type turns that are about the student, not the library — only chatter
   ("lol ok thanks") and topic-gate declines are host-certain today; everything else still
   depends on a model decision that has failed three separate mechanisms so far (see §3f).
3. **Normal turns run ~30-40s on Ling** (prefill-bound; roughly a 9s forced call plus
   12-25s reading evidence) — measured on this dev machine, **not verified on the Dell**.
4. **Ling's fit on the 6 GB laptop is unverified.** ~4.63 GiB model share measured here;
   no comparable desktop-vs-Dell VRAM comparison number exists in today's docs — not
   measured.
5. **"What's the largest molecule?" still fails when the model doesn't know the answer.**
   No query-writing scheme fixes this: the model never writes "titin" or "macromolecule"
   in its queries, so titin never enters evidence even with model-written search on.
6. **Re-asking the same question still replays the prior answer** rather than treating it
   as a fresh request.
7. **False positive: "How long is a blue whale's penis?" gets declined** by the topic
   gate — a genuine biology question caught by the same anatomy+measure-cue combination
   the unsafe case needs. Accepted trade-off (owner), not a bug to fix blindly.
8. **Dense index is stale** (unchanged from prior sessions).
9. **No real-browser verification** of anything landed today (see the box at the top).
10. **The "84 vs 42 questions" reconciliation** flagged in a prior handoff is still
    unresolved — not touched this session.
11. **Eval citation-rate metrics (`cited_rate`/`uncited_rate`) are less meaningful now**
    that labels are not requested by default (`app.model_writes_citations=False`) — the
    scripts (`eval/run_turn_eval.py`, `eval/run_lesson_soak.py`,
    `eval/attribution_scoring.py`) were not rewired to score host-inferred attribution
    instead; known limitation, not fixed.

## 7. Open questions for the owner

1. **Dell OS**: README says Linux Mint; the owner separately mentioned a Windows video
   driver for that machine — which is actually correct?
2. Whether to measure Ling vs Granite vs Bonsai 2 on the eight-lesson set from a separate
   config — owner said no for now.
3. The 5 remaining spec v0.4 amendment items still awaiting approval (items 1-7 minus
   item 8, which is now owner-approved and implemented) — see
   `docs/plan/spec_v0.4_amendments.md`.
4. Curriculum scope — still no guided path through the archive, one conversation per
   subject; confirm this remains out of scope.
5. Sentence-boundary chunking (vs. today's 400-char word-packed chunks that cut
   mid-sentence) — would require a dense-index rebuild and invalidate the tuning
   baseline; needs explicit go-ahead.
