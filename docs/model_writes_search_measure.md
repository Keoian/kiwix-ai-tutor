# Measuring `app.model_writes_search` (ON vs OFF) on HEAD

`app.model_writes_search` (added in c00fb2e, default False) makes the
model write 1-3 short article-style search queries on EVERY turn,
including turn 1, replacing the raw pre-search / follow-up rewrite for
that turn. This measures it live against HEAD (Retrieval v16 + assessor
v4 already on main) using data/mws_measure.py, an in-process
TestClient against the real llama-server on :8080 -- never the owner's
app on :8420.

Methodology: 8 lessons (the owner's real failing conversations, L1-L5
superlative questions, L6/L7 controls that already work, L8 spelling
errors), 2 arms (OFF = defaults, ON = `model_writes_search=True`), 2
reps each = 32 lesson-runs, 80 turns total (40 per arm). Per turn:
model-written queries, evidence titles actually retained into the
prompt this turn, `gold_in_evidence` (a gold-article title substring
present, case-insensitive), backed_sentence_rate, wall time, a crude
correctness flag (answer mentions the expected entity).

**Recording bug found and fixed mid-run** (caught by the orchestrator
before the doc was written, not by me): my first version of
`data/mws_measure.py` scanned `session.log._turns` for
`{"role": "tool", "passages": [...]}` entries to get evidence titles.
That shape is only ever produced by the raw pre-search / plain
weak-evidence-rewrite path (`PromptLog.append_evidence`). On the
model-writes-search / follow-up forced-rewrite path
(`_run_forced_rewrite_round` in `tutor/app/agent_loop.py`), the tool
result appended to the prompt is plain rendered text via
`log.append_tool_result(tool_call_id=..., content=tool_text)` -- never a
`passages`-shaped entry -- so every ON-arm turn (and every OFF-arm
follow-up turn) silently recorded `evidence_titles = []` regardless of
what was actually shown to the model, incorrectly reading as
`gold_in_evidence=False` across the board (e.g. L4 turn 1 ON with
queries `['Mount Everest', 'Mount Kilimanjaro', 'Mount Fuji']` recorded
zero evidence titles, which is impossible for a real "Mount Everest"
search). Fixed by snapshotting `session.retained_passages` (keyed by
label) before/after each turn and diffing on `(label, text)` -- this is
populated by `_retain_passages()` on *every* evidence path, forced round
included, specifically so citations can be resolved after prompt-log
eviction, and was the only place that survives both code paths. The
full 32-lesson-run measurement below is from the **re-run after the
fix** (`data/mws_measure.json`); the buggy first pass is kept at
`data/mws_measure.buggy_titles.json.bak` for reference and was
discarded from analysis entirely.

## Aggregate (40 turns per arm)

| | n | gold_in_evidence | crude-correct | mean backed_sentence_rate | mean wall_s |
|---|---|---|---|---|---|
| OFF | 40 | 25 (62.5%) | 6 | 0.811 | 7.39 |
| ON  | 40 | 34 (85.0%) | 9 | 0.841 | 9.08 |

## L1-L5 (superlative lessons, the target failure mode), all turns (28/arm)

| | gold_in_evidence | crude-correct | backed | wall_s |
|---|---|---|---|---|
| OFF | 17/28 (60.7%) | 6 | 0.826 | 7.63 |
| ON  | 22/28 (78.6%) | 9 | 0.854 | 8.43 |

## L1-L5 turn 1 only (10/arm) -- the exact target case

| | gold_in_evidence | crude-correct | backed | wall_s |
|---|---|---|---|---|
| OFF | 2/10 | 6 | 0.642 | 6.64 |
| ON  | 6/10 | 9 | 0.867 | 11.03 |

Turn-1 crude-correct counts as "6/9" not "6/10" or "9/10" because the
crude flag only applies to L1-L5 (n.a. elsewhere); OFF got the entity
right in its *answer text* on 6/10 turn-1s even when the gold article
was not literally in evidence (model already knew "blue whale",
"peregrine falcon", "Everest", "Nile" from pretraining) -- ON got 9/10
right and grounded them in evidence.

## Controls L6/L7 (must not regress)

| | gold_in_evidence | backed | wall_s |
|---|---|---|---|
| OFF | 6/8 | 0.919 | 5.87 |
| ON  | 8/8 | 0.861 | 10.81 |

Backed rate drops 0.058 (within the 0.1 tolerance in the brief); gold
presence improves, not regresses.

## L8 (spelling errors)

| | gold_in_evidence | backed | wall_s |
|---|---|---|---|
| OFF | 2/4 | 0.487 | 8.71 |
| ON  | 4/4 | 0.714 | 10.11 |

## Per-question table, the five superlative turn-1 questions

| Lesson | question | OFF gold | OFF correct | ON gold | ON correct | ON queries |
|---|---|---|---|---|---|---|
| L1 (both reps) | What's the largest molecule? | False | None | False | titin | `['DNA','protein','lipid']` |
| L2 rep0 | What's the biggest animal? | False | blue whale | True | blue whale | `['Blue whale','Giant squid','Largest animal on Earth']` |
| L2 rep1 | " | False | blue whale | True | blue whale | `['Blue whale','largest animals']` |
| L3 rep0 | What's the fastest bird? | False | peregrine | True | peregrine | `['peregrine falcon speed','peregrine falcon hunting speed','peregrine falcon diving speed']` |
| L3 rep1 | " | False | peregrine | False | None | `[]` (model produced no usable queries this rep) |
| L4 rep0 | What's the tallest mountain? | False | everest | True | everest | `['Mount Everest height','K2 height','Mount Everest facts']` |
| L4 rep1 | " | False | everest | False | everest | `['tallest mountain']` (too generic; still answered right from pretraining, not grounded) |
| L5 (both reps) | What's the longest river? | True | None | True | nile | `['Nile','Amazon River','Yangtze River']` |

L1 is the one lesson where ON never recovers the gold article
(Titin/Macromolecule/Polymer): the model consistently writes
`DNA/protein/lipid`, never `titin` or `macromolecule` -- it does not
know the trick answer to "what's the largest molecule" is Titin, so no
query-writing scheme fixes this without also fixing the model's prior.
The answer text does correctly say "titin" though (crude-correct
passes) -- likely pretrained knowledge, not evidence-grounded, since the
evidence titles are DNA/Protein/Lipid, none of which is Titin.

## Bad or degenerate model queries, verbatim

- L3 rep1 turn 1, L2 rep0 turn 3, L4 rep1 turn 3, L6 rep0 turn 2, L7
  rep1 turns 1 and 2, L8 rep0 turn 2: model produced `queries = []`
  (forced round found no usable tool call after `_clip_model_written_queries`,
  falling back to the not-found path). None of these were malformed
  sentences or leftover "biggest"/"fastest" wording -- the model simply
  emitted no tool call, or one that validated to nothing.
- L4 rep1 turn 1: query `'tallest mountain'` -- not a wrong guess, but
  too generic/measure-shaped rather than a specific article title, and
  it did not retrieve "Mount Everest" itself (gold_in_evidence=False)
  even though the model's own answer happened to say "Everest" anyway.
- No case was found where a query still contained the raw superlative
  word ("biggest", "fastest", "tallest", "longest") verbatim AND was
  multi-word/sentence-shaped -- when the model degenerated, it either
  produced a generic single term (`'tallest mountain'`) or nothing.
- No junk-title contamination (Longest Ride / Biggest Loser / Coffee
  Morning / Fastest lap / Long Island / tallest buildings / Molecule
  Man / Animal Crossing / Anime) was observed in ANY turn's retained
  evidence, ON or OFF, across all 80 turns.

## What was measured vs inferred

Measured directly, per turn, from the live TestClient run against
llama-server on :8080: model-written queries (from the SSE
`attributions` event's `evidence.rewritten_queries`), evidence titles
(diffed `session.retained_passages` before/after the turn -- see bug
note above), `gold_in_evidence`, `research_calls` (SSE `done` event),
backed_sentence_rate and unbacked count (`attribute_sentences` against
`session.known_passages()`), wall seconds, answer text/tokens, citation
presence, not-found phrase presence, junk-title presence.

Inferred/approximate: the "crude correctness" flag is a literal
substring match on the expected entity name in the answer text, not a
judged correctness score -- it says nothing about whether the rest of
the answer is accurate, well-cited, or age-appropriate. "Number of LLM
calls" was not recorded as a separate field; `research_calls` (host
research-engine invocations) was recorded instead and is a reasonable
proxy but not identical (a turn with 1 model-authored multi-query
research_many call is not the same accounting as raw LLM stream calls).

Not verified: correctness beyond the single expected-entity substring
(no human/LLM judge scored full-answer quality); effect on multi-lesson
soak stability or memory over a full lesson session (each lesson-run
starts a fresh profile/session); the model's behavior on lessons outside
this list; any interaction with `restate_question_last`/R2 beyond what
falls out of ON replacing the follow-up round on turn >= 2 (both arms
here already default `restate_question_last=True` per cbd8791).

## Decision

**Adopt.** ON clearly beats OFF on `gold_in_evidence` and the crude
correctness flag on L1-L5 in both reps (turn-1: 6/10 -> 9/10 correct,
2/10 -> 6/10 grounded in gold evidence; all-turns: 17/28 -> 22/28 gold,
6 -> 9 correct), does not hurt the L6/L7 controls (gold 6/8 -> 8/8;
backed_sentence_rate drops only 0.058, inside the 0.1 tolerance), and
mean added wall time per turn is 9.08 - 7.39 = 1.69s, well under the
~6s budget. `model_writes_search` default flipped to `True` in
`tutor/settings.py` (`AppConfig.model_writes_search` and
`load_config`'s `_app_bool` default), `tutor/app/compose.py`
(`_make_turn_runner` default and `build_deps`'s `getattr` fallback), and
`tutor/app/agent_loop.py` (`run_turn`'s keyword default), following the
same pattern cbd8791 used for `restate_question_last`. Updated
`tests/test_settings_app.py` (default-True assertion, configurability
test now sets `false` to prove it's overridable) and
`tests/test_agent_loop_model_writes_search.py` (the "off reproduces old
bytes" test now passes `model_writes_search=False` explicitly instead
of relying on the default).

L1 remains unsolved either way -- "largest molecule = Titin" is a
model-knowledge gap this setting cannot fix by itself, since the model
never writes a query containing "titin" or "macromolecule."
