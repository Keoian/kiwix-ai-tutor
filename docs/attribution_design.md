# Attribution: statement kinds and naming

Spec: `docs/plan/offline_tutor_spec_v0.3.md` §14 (Teaching behavior), exact wording:

> The system prompt distinguishes three kinds of statement the tutor can make:
> **source-backed** (cite `[S#]`), **computed** (show the `calc` expression),
> and **tutor's own example/derivation** (say so). The UI styles them differently.

§11/§12 carry the mechanics this rests on: `[S#]` labels resolved by the host and
rejected if unresolved; the source viewer highlights the exact cited sentence(s)
using the passage's stored character offsets; the chat pane distinguishes
source-backed/computed/own-example styles (already referenced in
`tutor/app/agent_loop.py`'s `uncited` docstring).

## Display wording (2026-09-20 owner feedback)

The internal names below ("source-backed", "computed", "own-example") stay as
the spec's three statement kinds and the names used in code. What the UI
*shows* the student is more precise about what actually happened: the host
checks a sentence against the passages retrieved/retained for that turn, not
"the whole library". A marker never claims a library-wide search was done
when it wasn't:

- ● source-backed → "Found in the sources the tutor looked up"
- ○ own-example/unbacked → "Not found in the sources the tutor looked up —
  this may be the tutor's own knowledge"
- ⚠ unbacked_number → "This number is not in the sources the tutor looked up
  — double-check it"
- ✓ computed/verified → "Checked by the calculator"

The turn-level note (shown when nothing on the turn was host-backed) also
distinguishes "no passages came back at all for this question" from "passages
came back but none of them backed this answer" — see the `attributions`
event's additive `passages_available` field in `tutor/app/compose.py`.

## Names used in code/UI

Matching the spec's own three terms rather than inventing new ones:

- **source-backed** — a sentence attributed to a passage (labeled `[S#]` or not,
  per Granite's staple-one-citation-per-paragraph defect, `docs/citation_experiment.md`).
  Code: `AttributionResult.attributions`, each item's `model_cited` distinguishes
  "sentence itself carries the label" from "host inferred it from term overlap".
- **computed** — out of scope for `attribute_sentences`; already handled by
  `calc_calls`/`calc` rendering (§14, `tutor/tools/calc_tool`). Not touched here.
- **tutor's own example/derivation** — the residual: no passage attributed, no
  figure, i.e. `unbacked_spans` reason `"unbacked"`. `"unbacked_number"` is a new,
  stricter sub-case (a figure with no support anywhere); the UI should style it
  more strongly than a generic own-example since an invented number is riskier.

## Specifics gate (2026-09-21, owner-reported live bug)

**What.** The owner watched the tutor fabricate a person and two temperatures
("Vitus Andronicus, a Roman soldier ... -70°C (-94°F)"), and the host then
marked those sentences ● "found in the sources" against a real passage that
was only about cold/temperature in general -- no library passage anywhere
named any such person or stated either figure. `is_supported` was passing
the sentence purely on generic word overlap ("cold", "survived",
"temperatures" all appear in the real passage too).

Fix: once a sentence carries a SPECIFIC -- a number, or a proper name -- the
overlap check is no longer enough. `is_supported(sentence, passage_text,
all_passages_text="")` now additionally requires every specific the
sentence states to actually appear in the evidence text (`passage_text`
plus `all_passages_text`, the concatenation of every passage in the
*current* turn's packet -- `resolve_citations`/`attribute_sentences`/
`_best_supporting_passage` all pass their already-received passage list
through, so no new plumbing/parameters were added at any public boundary).
A number must match the exact figure (reusing the existing `_figures`
extraction and its Unicode-minus/label-noise normalization) except a
unit-conversion OUTPUT riding along in parentheses right after its source
number (the "-94°F" in "-70°C (-94°F)") -- that output doesn't need its own
citation, but the ORIGINAL number ("-70") still does. A name is a
capitalised multi-word span, or a single capitalised word that is not
sentence-initial and not in a short stoplist (months, days, "I") --
matched case-insensitively, and a surname-only/one-token match against the
evidence counts too. A sentence with no specifics at all keeps the
pre-existing overlap-only behavior unchanged.

The evidence set is always the CURRENT turn's retrieved/retained passages
only -- never the tutor's own earlier text in the same lesson. Nothing in
`is_supported`/`resolve_citations`/`attribute_sentences` reads prior turns'
answer text; widening the evidence set to "the lesson so far" would let a
fabrication launder itself into "found" the second time the model repeats
it (exactly what happened live: turn 2 asked about "Vitus Andronicus" and
the model repeated turn 1's invention).

**What is NOT verified.** This is a specifics gate, not a full fact-check:
a paraphrased claim with no numbers and no proper names (e.g. "cold
temperatures can be survived for a time") is still judged by loose overlap
alone, same as before. The gate only tightens things once a sentence stakes
out something concrete enough to check mechanically.

## Amendment needed for spec v0.4

§14's three-way split assumes the model marks its own kind reliably ("say so").
Measured behavior (Granite, `docs/citation_experiment.md`) is that it does not:
one `[S1]` staples to the end of a multi-sentence paragraph, leaving prior
sentences formally uncited despite being drawn from the same passage. v0.4 should
add: the host may *infer* source-backed attribution for an unlabeled sentence from
term overlap with an already-cited passage in the same answer (this module), but
this inference is UI-visible (`model_cited=False`) and never fabricates or rewrites
a `[S#]` label in the stored answer text, per §11's "host never fabricates a
citation" rule already upheld by `agent_loop.TurnResult.uncited`.

## Model-written labels are no longer requested (owner decision, 2026-09-21)

Owner: "I don't think we need the [S1] stuff" -- the host's own per-sentence
attribution (`attribute_sentences`, amendment 6 above) already links each sentence to
the passage that backs it, via host-inferred citation independent of whether the model
itself writes an `[S#]` label. Asking the model to write labels is therefore no longer
necessary, so `app.model_writes_citations` (default **False**) removes the instruction
to write one:

- The system prompt's "Sourcing and citations" bullet asking the model to write the
  numbered label (`e.g. [S1] or [S1, S2]`) is omitted by default; it is still available
  (`app.model_writes_citations = true`, byte-identical to before) via
  `tutor.app.agent_loop.build_system_text`.
- The evidence-tail reminder (`tutor.app.citations.citation_reminder_text`) drops only
  the "cite ... like [S1]" clause by default, keeping "do not list or copy the
  sources" and "if none of them answers the question, say so".
- Passages in the evidence packet are still labelled `[S1]`... either way -- the host
  and UI (source viewer, chips) need the ids regardless of whether the model is asked
  to write them.
- If the model writes an `[S#]` label anyway (nothing stops it), citation resolution,
  chips, and invented-label hiding all keep working unchanged -- none of that code was
  removed, only the instruction to write one in the first place.
- `eval/run_turn_eval.py`, `eval/run_lesson_soak.py`, and
  `eval/attribution_scoring.py` still score presence of a model `[S#]` label
  (`cited_rate`/`uncited_rate`); with the setting off by default that metric measures
  something the model is no longer being asked to do, so it is expected to trend
  toward the host-inferred-only case. Left as-is (not rewired to score host-inferred
  attribution instead) -- noted here as a known limitation of those scripts' current
  metric under the new default, not fixed in this change.
