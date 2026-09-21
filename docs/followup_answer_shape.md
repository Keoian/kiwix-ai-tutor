# Follow-up answer shape: why every follow-up got the same essay

Problem (docs/passage_reuse.md, "Measured"): in a lesson, follow-ups like
"Is it a molecule?", "Yes but is it a molecule?", "Tell me about DNA"
produced near-identical long "Composition / Structure / Replication /
Function ... In summary" essays, regardless of what was actually asked.
A yes/no question should get "Yes -- because ..." in 1-3 sentences; a
follow-up should add what is NEW, not restate the earlier answer.

## What Granite actually saw (Step 1)

Traced the exact prompt built for a follow-up turn
(`tutor/app/agent_loop.py`, `run_turn` + `_run_forced_rewrite_round`):

- **System prompt** (`tutor/app/system_prompt.txt`): generic tutoring
  rules (cite sources, use calc for arithmetic, "keep your reply within
  the token budget ... roughly 800 tokens"). Nothing here pushes an
  essay shape, and nothing here says a short answer is acceptable --
  the 800-token budget line, if anything, reads as permission to use a
  lot of the budget every turn.
- **Follow-up host note** (`_FOLLOWUP_HOST_NOTE`, appended to the
  student's own message on every turn >= 2): tells the model to rewrite
  the query resolving pronouns, and ends with "answer directly first
  (yes or no, if it is a yes/no question), then explain."
- **Evidence preface / `strong_suffix`** (`_FOLLOWUP_DIRECTNESS_NOTE`,
  appended right after the merged evidence in the forced-rewrite round's
  tool result): "give a direct answer first (yes or no, if it is a
  yes/no question), then explain using the sources above."

So the host was already asking for a direct answer first on turn >= 2.
What it never said, anywhere: *don't repeat what you already told the
student earlier in this lesson*, or *a yes/no confirmation can be very
short*. With the same evidence passages present turn after turn (the
same DNA passage, reused via `app.reuse_prior_passages`) and no
instruction against restating, the model's default behavior -- and
Granite 4.0 H-Tiny's apparent prior for "explain a science topic" ==
"structured multi-section essay" -- won out: turns 2-5 of a baseline
lesson came back **byte-identical or near-identical** ("Yes, DNA
(deoxyribonucleic acid) is a molecule." + the same ~120-token essay),
Jaccard overlap up to 1.00 turn-to-turn.

Conclusion: the essay shape was not caused by the system prompt or the
evidence-rendering function -- it was a **missing instruction**: the
existing directness note asked for "answer first" but never asked for
"don't repeat" or "a yes/no answer can be brief."

## Experiment (Step 2)

Script: `data/followup_shape_measure.py` (pattern copied from
`data/passage_reuse_measure.py`), live against llama-server on :8080,
in-process `TestClient`, one lesson of 7 turns run 2x per arm:

```
1. What's the largest molecule?
2. What about DNA?
3. Is it a molecule?
4. Yes but is it a molecule?
5. Tell me about DNA
6. How does it copy itself?
7. Why does that matter?
```

Two arms (both keep the log append-only; only the forced round's
`strong_suffix` text changes):

- **baseline**: today's `_FOLLOWUP_DIRECTNESS_NOTE` ("answer directly
  first ... then explain").
- **enhanced**: `_FOLLOWUP_CONCISE_NOTE` -- same direct-answer-first
  instruction, plus "add ONLY what is new -- do not restate points you
  already made earlier in this lesson", "if the question only asks for
  a yes/no confirmation, 1-3 sentences total is enough", and "if the
  question asks for a fuller explanation ... still give a real
  explanation, not one line, citing sources."

(Candidate (b), a system-prompt-only variant via
`eval/system_prompt_variants.py`, was not run: Step 1 showed the gap was
in the per-turn note the model reads right next to the evidence it is
about to answer from, not in the always-present system prompt, so the
per-turn note was the more targeted lever and the one actually tested
end-to-end live.)

### Measured (2 reps each, `data/followup_shape_measure_20260921_084457.json`)

Answer tokens per turn:

| turn | question | baseline rep0 | baseline rep1 | enhanced rep0 | enhanced rep1 |
|---|---|---|---|---|---|
| 1 | largest molecule | 45 | 63 | 80 | 40 |
| 2 | what about DNA | 121 | 124 | 53 | 39 |
| 3 | is it a molecule? | 120 | 124 | 32 | 40 |
| 4 | yes but is it a molecule? | 120 | 124 | 31 | 35 |
| 5 | tell me about DNA | 120 | 124 | 57 | 35 |
| 6 | how does it copy itself? | 114 | 95 | 41 | 59 |
| 7 | why does that matter? | 123 | 95 | 36 | 28 |

Jaccard overlap of turn 3/4/5 vs turn 2 (same-topic follow-up
restating the earlier answer):

| | baseline rep0 | baseline rep1 | enhanced rep0 | enhanced rep1 |
|---|---|---|---|---|
| turn 3 vs 2 | 0.844 | 1.000 | 0.395 | 0.960 |
| turn 4 vs 2 | 0.844 | 1.000 | 0.333 | 0.704 |
| turn 4 vs 3 | 1.000 | 1.000 | 0.708 | 0.741 |
| turn 5 vs 4 | (=1.000 vs 2,3,4 all) | (=1.000 vs 2,3,4 all) | 0.588 | 1.000 |

First sentence, every baseline rep, turns 2-5: **"Yes, DNA
(deoxyribonucleic acid) is a molecule."** followed by the same
~120-124-token essay word-for-word (Jaccard 1.00 in rep1, 0.84-1.00 in
rep0). Enhanced cut those same turns to 31-60 tokens, still opening
with a direct "Yes, DNA is a molecule" but not re-deriving the whole
essay. Turn 1 (a real "what is X" question) and turns 6-7 ("how does it
copy itself" / "why does that matter" -- both need real explanations)
stayed substantive under the enhanced note (28-59 tokens, citing [S1]
where evidence was shown) rather than collapsing to one line --
guarding against the over-correction risk named in the task.
`backed_sentence_rate` stayed at 1.0 for both arms on most turns
(one enhanced rep0 dip to 0.5 on turns 3-4, where the very short 1-2
sentence answer left one clause unattributed; not seen in rep1).

**Measured, not just inferred**: the token-count collapse and
Jaccard-overlap drop above are both directly measured from the live
runs. **Not verified**: whether a real student perceives the shorter
answers as more helpful (no human eval was run); the system-prompt-only
variant (b) and the "both" combination (c) were not measured
separately, since the per-turn note alone already produced a clear,
large effect and Step 1's tracing pointed at the note as the lever, not
the system prompt.

## Decision (Step 3)

Adopted **the enhanced per-turn note**, wired behind
`AppConfig.concise_followup_note` (`tutor/settings.py`, `[app]` TOML key
`concise_followup_note`, default `True` -- same pattern as
`rewrite_on_followup`). `False` reproduces the prior
`_FOLLOWUP_DIRECTNESS_NOTE` byte-for-byte. Threaded through
`tutor/app/agent_loop.py` (`run_turn` picks `_FOLLOWUP_CONCISE_NOTE` vs
`_FOLLOWUP_DIRECTNESS_NOTE` as the forced round's `strong_suffix`) and
`tutor/app/compose.py` (`_make_turn_runner` / the `create_app` call
site). Unit tests in `tests/test_agent_loop_followup.py`
(`test_concise_followup_note_default_on_lands_on_followup_turn`,
`test_concise_followup_note_off_reproduces_plain_directness_note`) use
fake-LLM doubles to assert the note text lands in the forced round's
tool result on turn >= 2 (never turn 1), and that the `False` setting
reproduces the old wording exactly -- no live LLM in the test suite.

Not done in this pass (left for a follow-up if wanted): the (b)
system-prompt variant and (c) combo arms; a held-out/non-DNA topic
lesson to check the effect generalizes; human-perceived-helpfulness
eval.

## Iteration 2 (negative result)

An orchestrator review of the live lesson found iteration 1's adopted
note over-corrected: on the same DNA lesson, "Tell me about DNA" (turn
5) got a one-line non-answer ("Yes, DNA is a molecule."), and the open
questions "How does it copy itself?" (turn 6) and "Why does that
matter?" (turn 7) both started with a spurious **"Yes,"** even though
neither is a yes/no question -- the model was over-generalizing "answer
directly first" into "always open with Yes/No."

### Arms tried

All keep the log append-only and only change `_FOLLOWUP_CONCISE_NOTE`'s
wording (scripts: `data/followup_shape_measure2.py` + three throwaway
single-file add-ons for wordings c/d/e/f, not committed -- all scratch
in `data/`, gitignored). Two lessons: the original DNA lesson (7 turns)
plus a non-DNA volcano lesson (`"What is a volcano?"`, `"Are they
dangerous?"`, `"Tell me about the biggest one?"`, `"Why do they
erupt?"`, `"Is lava hot?"`) to check generalization.

- **a**: "decide yes/no first, else answer what was asked, else give a
  real explanation" (closest to the orchestrator's suggested wording).
- **b**: same idea, phrased as "classify only the latest message."
- **c**: "only say Yes/No if the message grammatically starts with
  Is/Are/Does/.../Have"; otherwise never start with Yes/No.
- **d**: c reworded, with an explicit "even if the underlying fact
  happens to be a yes/no fact" caveat.
- **e**: c/d reworded again with lesson-specific examples embedded.
- **f** (started, aborted mid-run on the orchestrator's stop instruction
  before it produced any results): a 3-case version of e with an
  explicit sentence-count floor for the "tell me about" case.

### Measured: first sentence per turn, DNA lesson (`data/followup_shape_measure2_20260921_085943.json`, rep0 shown; rep1 in the same file)

| turn | question | a | b | c | d | e |
|---|---|---|---|---|---|---|
| 3 | Is it a molecule? | "Yes, DNA is a molecule." | "I wasn't able to locate..." | "DNA is a molecule." (no Yes) | "DNA is a molecule." (no Yes) | "I wasn't able to find... DNA is a molecule." |
| 4 | Yes but is it a molecule? | "Yes, DNA is a molecule." | "I wasn't able to locate..." | "DNA is a molecule." (no Yes) | "DNA is a molecule." (no Yes) | "I wasn't able to find..." |
| 5 | Tell me about DNA | **"Yes, DNA is a molecule."** (still a non-answer) | "Yes, DNA (deoxyribonucleic acid) is a molecule." | "DNA is a molecule." (flattened, same as turns 3-4, not a real explanation) | same flattening as c | same flattening as c |
| 6 | How does it copy itself? | **"Yes, DNA copies itself..."** (spurious Yes) | "I wasn't able to locate..." then real content | "DNA copies itself..." (no Yes -- correct) | "DNA copies itself..." (correct) | "I wasn't able to find... DNA replication is..." |
| 7 | Why does that matter? | **"Yes, DNA replication matters..."** (spurious Yes) | "DNA replication is crucial..." (correct) | "DNA replication is important..." (correct) | "DNA replication is important..." (correct) | "I wasn't able to find..." |

Volcano lesson: the same spurious-Yes pattern repeated for arm a
("Tell me about the biggest one" -> "Yes, the biggest volcano is Mauna
Loa..."; "Why do they erupt?" -> "Yes, volcanoes erupt because...").
Arms c/d/e on the volcano lesson came back "I wasn't able to find a
clear definition of 'volcano'..." on **every turn including turn 1**
(which has no follow-up note at all).

### Conclusion

**Wording alone did not fix this.** Every arm traded one failure mode
for another:

- Arm a (and the iteration-1 default) reliably says "Yes" on real
  yes/no turns, but bleeds "Yes," onto every open turn (10/10 open
  turns wrong across both lessons/reps).
- Arms c/d/e stop the spurious "Yes," on open turns, but then either
  drop the literal "Yes" token on real yes/no turns too, or (c/d)
  flatten "Tell me about DNA" into the same 1-sentence answer as "Is it
  a molecule?" instead of a real explanation, or (e) prepend an
  irrelevant "I wasn't able to find..." disclaimer before still
  answering from evidence anyway.

This points at a stronger effect than any note wording can steer around
on this model/size: Granite 4.0 H-Tiny appears to anchor on **its own
previous answer** in the transcript and partially re-emits it
regardless of the current instruction, rather than reasoning fresh from
the newly-resolved question each turn. The default is turned back
**off** (`concise_followup_note: bool = False` in
`tutor/settings.py`); the flag and both wordings stay in the codebase
for further experimentation, but the shipped behavior is the
pre-existing, plainer `_FOLLOWUP_DIRECTNESS_NOTE`.

### Item 3: was evidence actually present on the "I wasn't able to find" turns, or a harness bug?

**Measured** (from `answer_text` stored this iteration, `has_citation`
and `backed_sentence_rate` fields in the checkpoint JSON):

- `b/dna rep0` turns 2-5: the model's own text, right after "I wasn't
  able to locate a specific answer about DNA," goes on to give a
  **"Direct answer:"** section with specific, evidence-consistent
  detail ("a long, double-stranded polymer made of nucleotide units
  that store genetic information..."), and turn 5's `backed_sentence_rate`
  is 0.8 (turn 2: 0.778). That rate is computed by
  `citations.attribute_sentences` against `session.known_passages()`,
  which only returns a nonzero rate if the answer text actually overlaps
  passage text held by the session -- so on those turns, evidence **was**
  present and substantially used, and "I wasn't able to locate..." is a
  rhetorical preamble the model adds before answering anyway, not a
  true absence of evidence. This is measured, not inferred.
- `c/d/e dna` turns 2-7: `has_citation` is False and
  `backed_sentence_rate` is 0.0 on every turn, yet the answer content
  is specific and topically correct (matches known DNA facts). Whether
  evidence was present in the prompt as full text, as an
  "(already shown above)" pointer, or was genuinely missing is
  **unknown/not measured this iteration** -- `data/followup_shape_measure2.py`
  only recorded the final answer text and `backed_sentence_rate`, not
  the rendered prompt log's tool messages (unlike
  `data/passage_reuse_measure.py`'s `_count_passage_lines` helper, which
  this script did not reuse). No conclusion can be drawn from this data
  about whether `app.reuse_prior_passages` pointer-only evidence is
  what causes the "wasn't able to find" framing; a repeat with the
  rendered log captured per turn is the direct way to check that.
- `c/volcano`, `d/volcano`, `e/volcano`: **every turn**, including turn
  1 (which has no follow-up note or forced-rewrite round of any kind --
  it is the plain pre-search path), came back "I wasn't able to
  find a clear definition of 'volcano'..." A live direct probe of
  `research_engine.research("What is a volcano?")` against the same
  registry, run immediately after, returned 6 real passages (Volcano
  article content) without issue. Since turn 1's behavior cannot depend
  on any of arms a-f's wording, this is measured evidence of
  **run-to-run flakiness in evidence assessment for this run**
  (`assess_evidence`/the forced-rewrite query-rewrite step, both of
  which involve their own live model calls with no fixed seed), not a
  wording-caused or `app.reuse_prior_passages`-caused regression -- but
  the root cause of that flakiness itself was not diagnosed further
  this iteration (stopped per the orchestrator's instruction before
  running the planned diagnostic).

### Next hypothesis to test

The student's resolved, standalone question is currently the first
thing the model reads in the follow-up turn (inside the host note,
before the query-rewrite tool call), then the evidence block comes
after, with the model's own prior answers still visible earlier in the
transcript. A plausible fix: after the evidence block, restate the
resolved standalone question (the model's own rewritten query from the
forced round, e.g. `"Is DNA a molecule?"` or `"How does DNA copy
itself?"`) as the **last line** of the turn, right before the model
generates -- on the theory that recency in the transcript matters more
than instruction wording for this model, and that the model is
currently defaulting to whatever question shape is nearest in its
context (which, without a restated question, is its own most recent
prior answer). This was not tried in this pass; note wording changes
alone are not the correct lever.

## Iteration 3 (built, not live-measured this pass)

Implemented the "restate the resolved question last" fix candidate
from the previous section, gated behind two new settings threaded like
`rewrite_on_followup` (`tutor/settings.py`, `tutor/app/agent_loop.py`,
`tutor/app/compose.py`):

- `app.restate_question_last` (default `False`): on a follow-up turn's
  forced-rewrite round, when the round's evidence comes back `strong`,
  append one line as the LAST thing in that tool result, after the
  evidence text and after `strong_suffix`:
  `The student is now asking: "<original message>" (meaning: <the
  round's own rewritten query>). Answer THIS question.` Never appended
  on turn 1 (no forced round exists yet), and not appended on the
  separate weak-evidence forced-rewrite path (only on the follow-up
  path, since the doc's hypothesis is specifically about follow-up
  turns burying the question behind evidence).
- `app.restate_question_instruction` (default `False`, only meaningful
  together with the above): appends one more sentence right after the
  restated question -- "If it is a yes/no question start with Yes or
  No; otherwise just answer it. Add what is new; do not repeat your
  earlier answer." This is the "R2" arm; "R1" is the restated question
  alone with no extra instruction sentence.
- Both default `False` and reproduce prior byte-for-byte behavior when
  off (`_restate_question_line` in `tutor/app/agent_loop.py`).

Unit tests (fake-LLM doubles, no live model) in
`tests/test_agent_loop_followup.py`:
`test_restate_question_last_off_reproduces_plain_bytes`,
`test_restate_question_last_r1_lands_on_followup_turn_only` (asserts
the line uses the original text + the round's own rewritten query, is
absent on turn 1, and is the last thing in the tool result), and
`test_restate_question_r2_adds_instruction_sentence`. Settings tests in
`tests/test_settings_app.py` cover the TOML default (`False`/`False`)
and override.

### Measured (`data/restate_measure.py`, `data/restate_measure.json`, 2 reps x 2 lessons x 3 arms = 12 lesson runs, live against llama-server on :8080)

Arms: **baseline** (both settings `False`, today's default), **R1**
(`restate_question_last=True`), **R2** (`restate_question_last=True`,
`restate_question_instruction=True`). Lessons: the DNA 7-turn lesson and
a new Volcano 5-turn lesson (`"What is a volcano?"`, `"Are they
dangerous?"`, `"Tell me about the biggest one"`, `"Why do they
erupt?"`, `"Is lava hot?"`).

First sentence / tokens / Jaccard-vs-previous-answer, DNA lesson, rep0
(rep1 in the checkpoint JSON tells the same story):

| turn | question | baseline | R1 | R2 |
|---|---|---|---|---|
| 2 | What about DNA? | "Yes, DNA ... is a molecule." (674 tok) | "Yes, DNA ... is a molecule." (672 tok) | "DNA is indeed a large molecule." (754 tok) |
| 3 | Is it a molecule? | "Yes, DNA is a molecule." (629 tok, J=0.96 vs t2) | "Yes, DNA ... is a molecule." (672 tok, J=1.00 vs t2) | "Yes, DNA is a molecule." (778 tok, J=0.29 vs t2) |
| 4 | Yes but is it a molecule? | same as t3 (629 tok, J=1.00 vs t3) | same as t3 (672 tok, J=1.00 vs t3) | same as t3 (778 tok, J=1.00 vs t3) |
| 5 | Tell me about DNA | **"Yes,** DNA ... is a molecule." (327 tok, J=0.50 vs t4) | "DNA ... is a molecule that contains..." (no Yes; 665 tok, but J=0.995 vs t4 -- essentially the same essay) | **"Yes,** DNA ... is a molecule." (778 tok, J=1.00 vs t4) |
| 6 | How does it copy itself? | **"Yes,** DNA replicates itself through..." (293 tok) | "DNA copies itself through a process called DNA replication." (no Yes; 856 tok) | "DNA copies itself through..." (no Yes; 717 tok) |
| 7 | Why does that matter? | **"Yes,** DNA replication is important..." (287 tok) | "DNA copying itself matters because..." (no Yes; 842 tok) | "DNA replication is crucial because..." (no Yes; 656 tok) |

Aggregate spurious-"Yes"-on-an-open-turn rate (turns 5-7 DNA, turns 3-4
Volcano; 10 open-turn instances total across 2 lessons x 2 reps):
**baseline 9/10**, **R1 1/10**, **R2 8/10**. R1 clearly, repeatably
fixes the iteration-2 failure mode (open turns bleeding a spurious
"Yes,"); R2's extra instruction sentence perversely brings the spurious
Yes back almost to baseline levels.

But R1 does **not** clear the doc's own success bar for open turns:
"Tell me about DNA" (turn 5) stopped starting with "Yes," but its
Jaccard overlap vs the immediately preceding answer was 0.995 (rep0)
and 0.877-0.886 (rep1) -- essentially the same essay restated, not the
"well under 0.8, add only what's new" the criteria call for. And no
arm -- baseline, R1, or R2 -- ever produced a *short* yes/no
confirmation: turns 3/4 ("Is it a molecule?" / "Yes but is it a
molecule?") stayed 629-778 tokens with Jaccard 0.96-1.00 vs the prior
turn in every arm, baseline included. `restate_question_last` changes
*whether the model opens with "Yes,"* on later turns; it does not touch
the separate, still-unsolved problem (named in iteration 2's
conclusion) of the model re-emitting its own prior full-length answer
regardless of what a yes/no turn actually needs.

Volcano lesson replicated the same pattern: R1's open turns ("Tell me
about the biggest one", "Why do they erupt?") did not start with
"Yes," in either rep; baseline and R2 mostly did. `has_citation`
([S#] present) was inconsistent across all three arms on both lessons
(true on DNA turn 1 in every arm; false on most Volcano turns in every
arm) -- evidence rendering/citation habits did not visibly change with
the restate settings either way. No "wasn't able to find" text was seen
on a turn with usable evidence in R1 or R2; one baseline rep (Volcano,
"Are they dangerous?") answered "I could not find information about
the danger of volcanoes..." despite `backed_sentence_rate=0.667`
(evidence was present) -- a pre-existing baseline flaw, not something
R1/R2 introduced or fixed.

**Measured, not inferred**: the spurious-Yes counts, token counts, and
Jaccard values above are all read directly from
`data/restate_measure.json` (full answer text, evidence-appended
content, and `backed_sentence_rate` stored per turn). **Not verified**:
why R2's extra instruction sentence reverses R1's improvement (not
diagnosed -- would need a prompt-content diff, which the checkpoint's
`appended_content` field has but this pass did not analyze further);
human-perceived helpfulness; any topic beyond DNA/Volcano.

### Decision

Neither R1 nor R2 clearly meets the doc's success criteria on both
lessons in both reps -- R1 fixes the spurious-"Yes"-on-open-turns
failure but not the near-verbatim-repeat failure on either open or
yes/no turns, and R2 is worse than R1 on the metric R1 does fix. Per
the task's decision rule, **defaults stay `False`**
(`app.restate_question_last`, `app.restate_question_instruction`); no
changes to `tutor/settings.py`, `tutor/app/agent_loop.py`,
`tutor/app/compose.py`, or their tests. The restate-question-last
mechanism is a real, measured partial improvement (open turns stop
saying "Yes," ~9x less often) but not a complete fix, and turning it on
by default would trade one measured problem for a smaller one rather
than solving the doc's actual goal. The still-open problem for a future
iteration: DNA/Volcano yes/no confirmation turns and DNA turn 5 replay
the full previous essay almost verbatim in every arm tried so far,
including the current default -- restating the question did not change
that; only the model's disposition to re-emit its own prior answer did.
