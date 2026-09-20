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

## 2026-09-20 follow-up: evidence-dump failure

Real failure from the project owner's session, verbatim (paraphrased
below, exact shape reproduced as a test fixture -- see
`tests/test_citations.py::_helium_dump_fixture` and
`tests/test_compose.py::test_turn_runner_flags_unsupported_citations_and_evidence_dump`):

> Question: "Output the boiling point of helium in celsius and
> farenheit." Retrieval (being fixed separately by another workstream)
> returned 11 passages about computer "output" devices, logic gates and
> multiplexers -- a lexical false-positive on the word "output" in the
> question. The model answered from memory ("-268.92°C or -452.04°F",
> slightly wrong -- the true value is -268.93°C / -452.07°F -- and it did
> not call `calc`), then wrote "Here's the reasoning:" followed by a
> bullet list that copied out all 11 off-topic passages verbatim as
> "[S1]: ..." ... "[S11]: ...", and ended the final sentence with
> "[S1][S2]...[S11]" stacked together. Every label *resolved* (all 11
> passages existed in the packet) so the host's old citation handling
> treated this as a fully-cited answer, even though not one of them
> supports anything the answer claims about helium.

### Three host-side defects fixed here

1. **"Resolves" != "supports".** `tutor.app.citations.Citation` gains a
   `supported: bool` field. `resolve_citations` computes it with
   `is_supported(sentence, passage_text)`: the sentence/bullet carrying
   the label is tokenized with `tutor.retrieval.hybrid.lexical.tokenize`
   (reused, not reimplemented) and compared against the cited passage's
   text; supported if they share >= 2 content terms (numbers count, since
   `tokenize` keeps digit runs) or >= 30% of the sentence's own content
   terms are shared. A sentence that merely quotes its passage back would
   pass this check on its own -- but see (2): when that quoting is part of
   a detected evidence dump, those specific citations are overridden back
   to `supported=False` (`tutor.app.citations._dump_signal`), since
   regurgitating the source is not the same as a legitimate citation
   quoting it. The `citations` SSE event (`tutor.app.compose`) now also
   carries `unsupported_labels`, and the `done` event carries
   `citation_quality`: `"ok"` (citations exist and all support), `"uncited"`
   (no `[S#]` label at all), or `"unsupported"` (at least one label present
   but not supporting). The host still never adds, removes, or rewrites a
   label in the model's own text -- `supported` is metadata alongside the
   citation, never a mutation of the answer.

2. **Evidence dumping.** `tutor.app.citations.detect_evidence_dump` flags
   an answer as `evidence_dump: true` (also in the `done` event) on either
   of two mechanical signals: (a) label-stacking -- a single
   sentence/bullet carrying 4 or more distinct `[S#]` labels (the
   "...[S1][S2]...[S11]" trailer), or (b) wholesale copying -- at least 3
   distinct cited sentences/bullets each have >= 80% of their own content
   terms contained in the passage their label resolves to. `tutor/ui/app.js`
   never edits the model's text; when `evidence_dump` is true it collapses
   the already-rendered answer behind a "Show the tutor's pasted sources"
   toggle button (`textContent` only, DOM nodes built the same way as the
   rest of the citation rendering -- `tests/test_ui_static.py` stays
   green, no `innerHTML` anywhere), and when every citation on the turn is
   unsupported it adds a plain note, "The tutor's sources did not match
   this question."

3. **Wording.** The evidence reminder adopted for the 0.20 -> 0.60
   citation-rate win (`render_evidence`'s trailing line) plausibly
   encouraged this failure: "Cite the sources you use like [S1]" reads as
   "cite all of them," not "cite only what's relevant." It is now: "Cite
   only a source that actually supports the sentence, like [S1]. Do not
   list or copy the sources. If none of them answers the question, say
   so." `tutor/app/system_prompt.txt` mirrors the same instruction, plus
   two additions: say plainly when the evidence doesn't cover the
   question and answer only what you're confident of as your own
   knowledge; and always use the `calc` tool for unit conversions (the
   owner's transcript did a Celsius/Fahrenheit conversion from memory
   instead). The prompt stays within the ~800-token slot
   (`tests/test_citations.py::test_system_prompt_stays_within_approx_800_token_budget`,
   a length/3.5 approximation -- the same fallback
   `tutor.app.compose._make_count_tokens` uses when the real tokenizer is
   unreachable).

### Mechanical scoring extended

`eval/run_turn_eval.py`'s `score_answer`/`aggregate` gain
`all_supported`/`supported_citation_rate` (citation exists, resolves, and
is supported -- strictly stronger than `all_resolve_rate`) and
`evidence_dump`/`evidence_dump_rate` (from `AnswerRecord.evidence_dump`,
which `run_variant` now populates via `detect_evidence_dump`).
`render_report`'s table gained both columns. All test-first, fakes only,
no network (`tests/test_run_turn_eval.py`).

### Status: live re-measurement DONE (2026-09-20)

Re-measured against the real `config/dev.toml` server (Bonsai-8B-Q1_0),
the real `simplewiki` archive (`config/archives.simplewiki_only.toml`,
no fixture ZIM), and 18 held-out-of-fixture questions from
`eval/questions/simplewiki_questions.jsonl` (`--split tuning
--categories direct,why_how,student_phrasing --n 18`), each a fresh
single-turn session. `eval/run_turn_eval.py` was extended first
(test-first, `tests/test_run_turn_eval.py`) with `--registry
--questions --split --categories --n --variants --out` and a JSON
per-answer dump (`data/citation_run_2026-09-20.json`) carrying
`calc_calls`, `first_token_ms`, `total_ms`, `prompt_tokens`,
`passages_in_packet`, `uncited`, in addition to the existing scoring
fields.

**Bug found and fixed first:** the harness's `LlamaClient` was
constructed with `timeout_s=10`, far too short for real-archive prompt
prefill on this GPU. The first RUN A pass returned the student-safe
error message ("Sorry, I ran into a problem...") for roughly half the
18 questions -- those were HTTP timeouts, not model refusals. Raised to
`timeout_s=120` (`eval/run_turn_eval.py::_build_live_collaborators`)
and re-ran; zero timeout errors after the fix. All numbers below are
from the fixed-timeout run.

**RUN A -- shipped configuration (`current`: real system prompt,
default `render_evidence`, default temperature), n=18:**

| Variant | n | citation_rate | all_resolve_rate | supported_citation_rate | evidence_dump_rate | on_topic_rate | taught_rate | first_token p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 18 | 0.11 | 0.11 | 0.00 | 0.00 | 0.11 | 0.28 | see data/citation_run_2026-09-20.json |

Cited-and-supported = 0/18. This is far below the 0.5 gate in the task
brief, so RUN C ran (below). Reading the per-answer JSON dump: the
model overwhelmingly answers real-lesson factual questions from its own
parametric knowledge without emitting any `[S#]` label at all (16/18
answers `uncited`), even though every turn was `route=preretrieve`
(real evidence was supplied, 6 passages/packet). The two questions that
did cite worked correctly (one supported, e.g. sw63's nitrogen boiling
point cleanly cited `[S1]`). This reproduces the same failure shape as
the 0.889-uncited measurement from the real 30-minute lesson --
confirming that was not an artifact of that particular session, and
that the reworded reminder/system-prompt did **not** move the needle on
real-archive, real-lesson questions the way the fixture-only 5-question
A/B (0.20 -> 0.60) suggested it would.

**RUN C -- two further host-side levers, same 18 questions (triggered
because RUN A's supported rate was below 0.5):**

| Variant | n | citation_rate | all_resolve_rate | supported_citation_rate | evidence_dump_rate | on_topic_rate | taught_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| c_short_prompt | 18 | 0.06 | 0.06 | 0.00 | 0.00 | 0.06 | 0.22 |
| d_temperature_0.2 | 18 | 0.11 | 0.11 | 0.11 | 0.00 | 0.06 | 0.22 |

Neither variant beats `current`'s 0.00 supported rate by the required
>= 0.15 margin (`c_short_prompt` is worse; `d_temperature_0.2` gains
only 0.11). **Nothing was adopted.** The shipped `current` configuration
(system prompt + evidence-reminder wording from the 2026-09-20
evidence-dump follow-up above) remains the default; `tests/
test_turn_live.py`'s citation xfail reason was updated to this
measurement (below).

**RUN B -- owner's exact helium question, 3x, `current` variant:**

Question: "Output the boiling point of helium in celsius and
farenheit." Against `config/archives.simplewiki_only.toml` (real
simplewiki ZIM). All three runs: `uncited=True` (no `[S#]` label),
`calc_calls=0` (calc was NOT invoked for the C->F conversion despite
the system prompt's explicit instruction to always use calc for unit
conversions), `passages_in_packet=6`, route `preretrieve`.

Verbatim answers:

1. "The boiling point of helium is approximately **-268.9C** (or
   **-452.0F**). This information is based on the fact that helium has
   the lowest boiling point of all the elements, and its boiling point
   is significantly lower than that of water."
2. "The boiling point of helium is **-268.92C** or **-452.04F**. This
   is based on the information that helium does not usually react with
   anything else and has the lowest boiling point among all elements.
   The boiling point of a substance depends on the surrounding air
   pressure. For helium, this means its boiling point is significantly
   lower than that of water, which boils at **100C** (212F) at
   standard pressure."
3. "The boiling point of helium is -268.9C (which is approximately
   -452.0F). This is the lowest boiling point among all elements."

Checked what the retrieval packet actually contained
(`data/check_helium_packet.py`, scratch): the `Helium` passages
retrieved (`S1`, `S2`) state qualitatively that "It has the lowest
boiling point of all the elements" but carry **no numeric boiling-point
figure** at all -- Simple Wikipedia's Helium article text in this
packet doesn't surface an infobox value the retrieval pulled in. So the
model's numbers are from its own parametric knowledge, not the
evidence, which is consistent with it not citing (there was nothing to
cite for the specific figure) -- but the system prompt also asks the
model to say plainly when it's answering from its own knowledge rather
than the evidence, and none of the three runs did that either.

On accuracy: true values are -268.93 C / -452.07 F. The three runs
(-268.9/-268.92/-268.9 C, -452.0/-452.04/-452.0 F) are all within
about 0.03-0.07 of the true Celsius figure and 0.03-0.07 of the true
Fahrenheit figure -- close, plausibly memorized, not evidence-grounded,
and not run through `calc` as instructed (the C->F conversion is
internally consistent across all three answers, e.g. -268.92C ->
-452.04F is the correct linear conversion, so the model's arithmetic
here happens to be right even without calc -- but the instruction to
use calc was not followed).

### What was adopted, and what remains model-limited

**Adopted: nothing new.** The shipped `current` system prompt +
evidence-reminder wording remains the host's default; RUN C's two
alternatives did not clear the required improvement bar. The host-side
timeout fix (`timeout_s=10 -> 120` in the eval harness) was adopted
into the harness itself since it was a measurement bug, not a citation
lever, and it does not touch any code that ships to students.

**Model-limited, not host-limited, going into M6:** on real-archive,
real-lesson factual questions, the shipped 1-bit-8B model
overwhelmingly answers from its own parametric knowledge and does not
reach for the supplied `[S#]` evidence at all (89% uncited here, in
line with the 0.889 figure from the real 30-minute lesson), even when
the evidence block, the system prompt, and a lower sampling temperature
all explicitly ask it to cite only what's supported and say so when
evidence doesn't answer. Three host-controlled levers have now been
tried against live real-archive traffic (evidence-reminder wording,
system-prompt length, temperature) without closing this gap. This is
the standing item to carry into the Q2_0 comparison at M6: the citation
behavior needs to be checked again there, but the working hypothesis is
that this is a quantization/capability ceiling of the current 1-bit-8B
model on citation-following, not a host-prompt-wording problem the
current levers can still move.

## 2026-09-20 Task 3: seed-exchange variant (built, not yet measured)

Added an OFF-by-default prompt variant, `seed_exchange_s0`
(`tutor.app.seed_exchange`), that seeds a brand-new lesson with one fixed
synthetic exchange before the first real turn: a student question, an
evidence passage labelled `[S0]`, a two-sentence answer with `[S0]`
attached to the specific sentence it supports (not trailing the
paragraph), and one `calc` tool use for a m/s -> km/h unit conversion
whose result is used in the answer. Pitched at a 10-16 year-old
(thunder/lightning + sound speed).

Implementation, test-first (`tests/test_seed_exchange.py`):
- `tutor.app.prompt.PromptLog` gained `append_seed`, a dedicated `_seed`
  list distinct from `_turns`: set at most once, before any real turn,
  rendered right after the system message, and never touched by
  `evict()` -- so it is never evicted, never reordered, and trivially
  byte-identical across every subsequent turn. Its tokens are still
  counted by `tokens_used()` like any other message, so its cost is
  accounted for in the same `budget.system + budget.history` ceiling
  `evict()` already enforces.
- `tutor.app.citations.RESERVED_SEED_LABEL` ("S0") is refused
  unconditionally by `resolve_citations` (always `unresolved=True`, even
  if a passage happens to be labelled "S0") and excluded from
  `attribute_sentences`' `by_label` lookup, so it can never resolve to a
  passage, open the source viewer, or receive a sentence attribution.
  Real evidence numbering (`Session.allocate_label`) starts at S1 and
  never collides with it.
- The seed is off by default: `[app] prompt_variant` (new `AppConfig`
  field, default `"current"`) must equal
  `tutor.app.seed_exchange.SEED_EXCHANGE_VARIANT` for
  `tutor.app.compose.build_deps`/`_SessionStore.create_for_lesson` to
  call `seed_session` on a brand-new lesson; an unseeded log's `render()`
  is unchanged (byte-identical to before this change). `seed_session`
  never calls `session.retain_passages` for the S0 passage and never goes
  through `LessonStore.append_turn`/`persist_turn`, so it is never a
  citable passage for a real question and never shows up as a persisted
  student turn or in the live SSE transcript (it is injected directly
  into the log, not emitted).
- Registered for `eval/run_turn_eval.py` as `seed_exchange_s0` in
  `eval/system_prompt_variants.VARIANTS` (`"seed_exchange": True`,
  threaded through `run_variant`'s `make_session()` step).

**Not yet measured** (build-only task; live measurement is a follow-up).
To measure against the same 18 tuning questions used for RUN A/RUN C
above:

```powershell
python -m eval.run_turn_eval --registry config/archives.simplewiki_only.toml `
  --questions eval/questions/simplewiki_questions.jsonl --split tuning `
  --categories direct,why_how,student_phrasing --n 18 `
  --variants current,seed_exchange_s0
```

Adopt only if `supported_citation_rate` improves by >= 0.15 over
`current` without raising `evidence_dump_rate`, per the task brief's gate.
