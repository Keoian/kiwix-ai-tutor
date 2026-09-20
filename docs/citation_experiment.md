# Citation-rate experiment (2026-09-20)

Problem (measured, `docs/M3_report.md`): against the live 1-bit Bonsai-8B
model, the tutor emitted an `[S#]` citation in **0 of 3** live runs on a
factual fixture question even though evidence was supplied. Citations are
the product's core promise (spec §11/§12). This experiment tunes only
what the HOST controls -- system prompt text, evidence rendering, evidence
placement, and answer-turn sampling temperature -- never model-name
branching, never parsing prose as a tool call, and never fabricating a
citation the model did not write.

## Method

`eval/run_turn_eval.py` drives the REAL turn path
(`tutor.app.agent_loop.run_turn` over a real `LlamaClient` against the
real dev llama-server, `config/dev.toml`) and a real `ResearchEngine`
over the libzim fixture archive built by `tests/zim_fixtures.py`. It
skips cleanly if `/health` is down.

**Question set.** `eval/questions/fixture_questions.jsonl` has 12
questions, but the fixture ZIM (`tests/zim_fixtures.py`'s
`_RICH_ARTICLES`) only contains 5 of the corresponding articles
(`pythagorean_theorem`, `algebra_basics`, `history_of_mathematics`,
`geometry_intro`, `erdos_number`); `load_questions()` filters to those 5
(q01-q05) so every question can actually be answered from the archive
under test. The other 7 (photosynthesis, water cycle, WWII, ...) have no
matching content here and were out of scope for this harness.

**Scoring** (`score_answer`, mechanical, no judgment calls): does the
answer contain >=1 `[S#]` label; do all labels resolve against the
retained passages (`tutor.app.citations.resolve_citations`); does a
resolved citation's `path` match the question's `expected_paths`; does
the answer show a teaching behaviour (a follow-up question or step
marker); answer length in characters.

**Run size.** 5 questions x 1 run per variant (the task brief's target of
12 was not reachable -- see question-set note above -- and 5 questions
already reproduced the M3 baseline finding). Total GPU time for all 6
initial variants plus 1 follow-up combination: **~6.6 minutes**
(336s + 57s), well under the ~40-minute budget.

## Variants

| Variant | Change | citation_rate | all_resolve_rate | on_topic_rate | taught_rate | avg_len (chars) |
|---|---|---:|---:|---:|---:|---:|
| baseline | current system prompt + evidence rendering, default temperature | 0.20 | 0.20 | 0.20 | 1.00 | 1279 |
| a_evidence_reminder | evidence block gets a trailing line "Cite the sources you use like [S1]." (label already at line start) | **0.60** | **0.60** | **0.60** | 0.80 | 1164 |
| b_one_shot_example | one-shot cited-answer example appended to the system prompt | 0.40 | 0.40 | 0.40 | 0.80 | 1029 |
| c_short_prompt | short, imperative, numbered-rules system prompt | 0.00 | 0.00 | 0.00 | 1.00 | 1129 |
| d_temperature_0.2 | answer-turn sampling temperature 0.2 (vs config default 0.5), everything else baseline | 0.00 | 0.00 | 0.00 | 1.00 | 1138 |
| combo | short prompt + one-shot example + evidence reminder + temperature 0.2 | 0.20 | 0.20 | 0.20 | 1.00 | 1119 |
| e_reminder_plus_oneshot (follow-up) | baseline prompt + one-shot example + evidence reminder | 0.40 | 0.40 | 0.40 | 1.00 | 1315 |

Raw numbers reproduced in `eval/run_turn_eval.py`'s own output; the table
above is that script's `render_report` output pasted in verbatim.

## What was adopted, and why

**`a_evidence_reminder` alone** -- the trailing "Cite the sources you use
like [S1]." line appended to the evidence tool-result, nothing else
changed. It was the single largest lever measured (0.20 -> 0.60) and,
notably, *combining* it with other changes never beat it alone:
- `combo` (short prompt + one-shot + reminder + temp 0.2) scored only
  0.20 -- the short prompt variant on its own also scored 0.00, so it
  appears to actively hurt this model's citation behaviour, and stacking
  it back onto the reminder erased the reminder's gain.
- `e_reminder_plus_oneshot` (reminder + one-shot, no short prompt, no
  temp change) scored 0.40 -- the one-shot example alone also underwhelms
  (0.40) and adding it to the reminder made things worse than the
  reminder alone, not better.

This is a real (if small-N) finding: for this model, a short in-band
reminder at the point evidence is shown outperforms front-loaded
instructions (one-shot examples, a shorter system prompt) and outperforms
a lower sampling temperature. Adopted as `tutor.app.citations
.render_evidence`'s new default (see its docstring and
`tests/test_citations.py`
::test_render_evidence_appends_citation_reminder_after_non_empty_packet,
written test-first).

The system prompt (`tutor/app/system_prompt.txt`) is left unchanged --
neither the one-shot example nor the shorter prompt beat the baseline, so
there is no measured basis to change it. `config/dev.toml`'s sampling
defaults are also left unchanged (temperature 0.2 for the answer turn
scored 0.00 here, i.e. worse than the configured 0.5).

## `tests/test_turn_live.py` update

`test_factual_question_pre_retrieves_streams_and_resolves_citations`'s
citation assertion stays `xfail(strict=False)`, not a hard assertion:
60% is above the old 0% but still short of the task's 80% (>= 2 of 3 attempts)
bar for converting it to a real assertion. The xfail reason string now
cites the measured 3/5 (60%) rate and this document instead of the old
0/3 M3 number.

## What remains model-limited

Even with the best host-side variant, citation is not reliable (60%, not
100%), and the two "make the model read more instruction" levers (a
one-shot example, a shorter prompt) did not help or actively hurt -- that
looks like a genuine 1-bit-quantization instruction-following ceiling on
this checkpoint (Bonsai-8B-Q1_0), not a host wiring gap: the plumbing
(evidence -> `[S#]` -> `resolve_citations` -> source viewer) already
works end to end whenever the model does emit a label (`docs/M3_report.md`).
This should be re-measured with the Q2_0 checkpoint on the Dell hardware
at M6 -- higher-bit quantization is expected to follow instructions (both
the reminder and possibly the one-shot/short-prompt levers this run
penalized) more reliably, and the same harness (`eval/run_turn_eval.py`)
can be pointed at that config unchanged.

## Commands to reproduce

```powershell
# Ensure llama-server is healthy:
curl http://127.0.0.1:8080/health

# Run the full variant sweep (skips cleanly if the server is down):
python -m eval.run_turn_eval

# Run this experiment's own unit tests (fakes, no network):
python -m pytest tests/test_run_turn_eval.py tests/test_citations.py -q -p no:warnings

# Run the live plumbing test (now updated xfail reason):
python -m pytest -m integration tests/test_turn_live.py -v -p no:warnings
```
