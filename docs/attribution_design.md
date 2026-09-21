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
