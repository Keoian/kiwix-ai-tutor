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
