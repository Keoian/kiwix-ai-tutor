# Spec v0.4 amendment proposals

Status: proposal for the project owner to approve or reject. This document does not
edit `docs/plan/offline_tutor_spec_v0.3.md`; it lists diffs against it. Numbers below
are measured unless marked "inferred" or "pending". Section numbers refer to v0.3.

## 1. Two-stage eviction, not whole-triple eviction

**Spec, §8.1:** "When the prompt would overflow, drop the oldest `[user, tool,
assistant]` triples from the head of the history until it fits, keeping the system
region and any triple whose evidence is cited by the last two answers."

**Implementation:** `PromptLog.evict()` in `tutor/app/prompt.py` (lines ~382-422+) does
two stages, per its own docstring: stage 1 drops only the passage *text* of uncited
evidence tool-results within the oldest turns first (replacing content with a stub,
keeping the message list schema-valid), and only proceeds to stage 2 (dropping whole
`[user, tool, assistant]` triples) if stage 1 alone did not free enough tokens. Both
stages are logged as one `EvictionEvent` (one head edit, one re-prefill), matching the
spec's re-prefill-cost reasoning but not its wording, which only describes whole-triple
eviction.

**Proposed replacement (§8.1):** "When the prompt would overflow, evict in two stages
within the oldest turns first: (1) drop the passage text of any tool-result entry not
cited by a retained assistant message, replacing it with a stub, so a later
re-retrieval re-fetches it under a new label; (2) if stage 1 does not free enough
tokens, drop whole oldest `[user, tool, assistant]` triples, keeping the system region
and any triple whose evidence is cited by the last two answers. Both stages count as
one eviction event."

**Why:** ships as-built; the docstring records this was "the project owner's approved
change" but the spec text was never updated. Not independently re-confirmed with the
owner in this session — flagged for owner sign-off since it says "approved" without a
citable decision record.

## 2. Evidence budget is a cap, not a target

**Spec, §7.2 step 7 / §8.2:** "Pack. Greedy by relevance × new facet coverage / token
cost..." and "Evidence packet budget defaults to 1,200 tokens at 8K and 1,800 at 16K,"
worded as if the packer fills the budget.

**Implementation:** `tutor/retrieval/research.py`, `_apply_relevance_cutoff()` (~line
251) and its call site (~line 1566), applied before `pack()` (`tutor/retrieval/hybrid/
packer.py`). Docstring: "the 2,000-token evidence budget (`pack`'s `budget_tokens`) is
a CAP, not a target." A passage is kept only if its score is at least a fraction
(`_PACKING_RELEVANCE_FRACTION_DEFAULT = 0.25`) of the top passage's score AND it covers
an own content term of the question, with an infobox exemption from the term-overlap
requirement for quantity/unit questions (see amendment 3), hard-capped at
`_PACKING_MAX_PASSAGES_DEFAULT = 6` passages, tuned on the tuning split
(`docs/retrieval_baseline.md`, "Baseline v5" item 3).

**Proposed replacement (§7.2 step 7):** "Pack. Before packing, apply a relevance
cutoff: drop candidate passages scoring below a fraction of the top passage's score
unless exempt (see key-fact exemption below), and cap the passage count. Then pack
greedily by relevance × new facet coverage / token cost into the evidence budget. The
evidence budget is a cap on packet size, not a target the packer tries to fill."

**Why:** the "2,000-token" figure in the docstring is a comment, not this project's
config value (config default is 1,800 at 16K, matching §8.2) — the packer's behavior
(stop early on relevance falloff) is the substantive change, independent of the exact
budget number.

## 3. Key-fact infobox passages

**Spec:** no infobox-specific passage type; §7.2 step 5 describes only heading-aware
80-180 word passage groups.

**Implementation:** `build_key_fact_passages()` in `tutor/retrieval/hybrid/passages.py`
(~line 176), called from `research.py` (~line 1350) for the top-2 scored articles only.
Docstring: "Baseline v6 ('infobox key facts'): a matching infobox row is short (a
label/value pair)... Compact, citable passages for infobox rows whose KEY shares an own
content term." These bypass BM25/diversity ranking (`research.py` line ~1308 comment)
and get their own relevance-cutoff exemption when the query asks for a quantity/unit
(`_asks_for_quantity()`, `research.py` ~line 244, matches "how many/much/long/far/
fast/old/tall/hot/cold/high/deep", "degrees", "percent", "%", "number of").

**Proposed addition (new §7.2 step 4a, after Extract):** "Key-fact passages. For the
top-2 scored articles, build one short passage per infobox row whose label shares a
content term with the question, independent of the heading-aware passage groups in
step 5. These bypass ranking and are exempt from the relevance-cutoff's term-overlap
requirement (but not its score-fraction requirement) when the question asks for a
quantity or unit."

**Why:** ships as-built (Baseline v6, measured on tuning per `docs/retrieval_baseline.md`);
not previously described in the spec at all.

## 4. Passage-ID wording (spec vs. reuse plan)

**Spec, §11:** "immutable passage IDs derived from edition + path + extractor version +
content."

**Reuse plan, §11 ("Section 11 — Citations"):** "archive fingerprint + canonical entry
path + section ID + extractor version + passage text hash / character span."

**Implementation:** `_passage_id()` in `tutor/retrieval/hybrid/passages.py` (~line 93)
hashes: fingerprint digest, archive ID, canonical path, section index, extractor
version, sha256(text), start, end — a 32-hex sha256 digest of that joined payload. This
matches the reuse plan's fuller list (fingerprint, path, section, extractor version,
text hash, span), not the spec's shorter "edition + path + extractor version + content"
(the spec text omits section and character span as separate inputs, though "content"
could be read to subsume the span).

**Proposed replacement (§11):** align the spec's phrase to the reuse plan and the code:
"immutable passage IDs derived from archive fingerprint + canonical entry path +
section index + extractor version + passage text hash + character span." Per the
working agreement, the reuse plan wins on algorithms where the two conflict — this
amendment makes the spec text match that resolution rather than leaving an
inconsistency for a future reader.

**Why:** `docs/bundle_and_passages.md` already documents this as "the same rule as the
spec's shorter summary, at a fuller level of detail" — this amendment just moves that
reconciliation into the spec itself.

## 5. Model section: Granite 4.0 H-Tiny Q4_K_M replaces Bonsai as dev model

**Spec, §3.1:** "Primary: Ternary Bonsai 8B (Q2_0)... Publisher: prism-ml... Apache
2.0. Requires the PrismML-Eng/llama.cpp fork."

**Measured basis** (`docs/bakeoff_dev/README.md`, three-model bake-off table, this
session's `granite_soak10_v2.md` / `granite.md` / `granite_flags.md` /
`granite_citations.md`): Granite 4.0 H-Tiny Q4_K_M runs on stock llama.cpp (no fork),
5.36 GiB VRAM at 32K context with q8_0/q8_0 KV, pp 683-706 t/s at depth 0 (versus
Ternary's 157-229 t/s), tg 67.8-68.1 t/s at depth 0, 62 turns / 0 errors in a 10-minute
soak with first-token p50 2.969 s / p95 11.212 s (versus Bonsai Q1_0's 25.132 s / 56.855 s
and Ternary's 8.532 s / 74.803 s), 0/20 malformed tool calls with 19/20 correct routing
decisions, 100% of soak factual answers carrying a citation label (versus Ternary's
95.2% under the older presence-only metric and Bonsai Q1_0's 0%).

**Granite's measured defect, carried forward, not resolved by this amendment:** 0% of
those same soak citations pass the stricter sentence-level support check
(`is_supported`) — Granite staples one `[S1]` to the end of a 4-6 sentence paragraph
rather than the sentence it backs (see amendment 6, which is the host-side response to
this defect). Cold single-turn: cited 0.44, supported 0.11 (18-question set,
`granite_citations.md`).

**Proposed replacement (§3.1), marked pending owner confirmation:** replace "Primary:
Ternary Bonsai 8B (Q2_0)" with "Primary (dev): Granite 4.0 H-Tiny Q4_K_M, Apache-2.0,
runs on stock llama.cpp — no fork dependency," carrying forward the bake-off numbers
above as the basis, and demote Ternary Bonsai Q2_0 and Bonsai Q1_0 to the §3.3
fallback/stretch table (both measured worse: Q1_0 cites nothing under identical
conditions and answers from memory; Ternary needs fa-off/f16 KV on this GPU, fitting
only 16K context with 75 s late-lesson first-token p95).

**Why pending:** `HANDOFF.md` §5 open question 1 asks the owner to explicitly confirm
Granite replaces Bonsai in the plan (plan §0.2 and spec §3.1 both still name Bonsai;
M6's bake-off was originally meant to choose between Q1_0 and Q2_0, not introduce a
third model). This amendment proposes the wording change contingent on that
confirmation, and separately flags that Granite's license (Apache-2.0) needs an entry
in `THIRD_PARTY_NOTICES.md` once confirmed — this document does not add that entry.

## 6. Host-inferred sentence-level attribution as a separate layer

**Spec, §14:** "The system prompt distinguishes three kinds of statement the tutor can
make: source-backed (cite `[S#]`), computed (show the `calc` expression), and tutor's
own example/derivation (say so). The UI styles them differently." This assumes the
model reliably marks its own kind — measured behavior says it does not (amendment 5).

**Implementation:** `attribute_sentences()` in `tutor/app/citations.py` (~line 352)
splits the answer into sentences and, for each, either accepts the model's own `[S#]`
label (`model_cited=True`) or infers the best-supporting passage from term overlap
(`model_cited=False`), reusing `is_supported()` (~line 109). Sentences with neither go
to `unbacked_spans`, with reason `"unbacked_number"` (~line 414) instead of plain
`"unbacked"` when the sentence contains a figure absent from every passage in the
packet — the flag called for in the task brief. The host emits this as a distinct SSE
event, `attributions` (`tutor/app/compose.py` line 375, `emit("attributions",
attributions_payload)`), separate from the existing `citations` event, and persists it
in its own `attributions_json` column (`tutor/app/lesson_state.py`). The host never
edits the model's stored answer text or inserts labels into it — `docs/
attribution_design.md` states this explicitly and ties it to `agent_loop.TurnResult
.uncited`'s existing "host never fabricates a citation" rule.

**Proposed addition (§14, replacing the assumption that the model self-marks reliably):**
"The host does not rely solely on the model to mark statement kind. After the answer
completes, the host splits it into sentences and attributes each one: a sentence
carrying the model's own `[S#]` label is source-backed as marked; an unlabeled sentence
may be attributed by the host to a supporting passage already in the turn's evidence,
based on term overlap, and is shown to the student as host-inferred, not model-cited;
a sentence attributed to nothing is the tutor's own words, and a sentence containing a
figure that matches no passage is flagged as an unverified number. The host never edits
the model's answer text or inserts labels into it — attribution is a separate,
UI-visible layer (a distinct SSE event, or fields alongside `citations`)."

**Why:** ships as-built and directly addresses the measured defect in amendment 5
(Granite: 100% cited, 0% sentence-supported by the mechanical check, `docs/bakeoff_dev/
README.md`'s "By-eye read of `supported`" section, which found the mechanical flag
"about right, arguably slightly lenient").

## 7. Reserved `[S0]` seed label

**Spec:** no mention of a reserved/seed passage label; §7.1 numbers `[S#]` labels
starting from real evidence only.

**Implementation:** `tutor.app.seed_exchange` (per `docs/citation_experiment.md`,
"2026-09-20 Task 3") seeds a new lesson with one fixed synthetic exchange labelled
`[S0]`, off by default (`AppConfig.prompt_variant` must equal `SEED_EXCHANGE_VARIANT`).
`RESERVED_SEED_LABEL = "S0"` in `tutor/app/citations.py` is refused unconditionally by
`resolve_citations` (always `unresolved=True`) and excluded from `attribute_sentences`'
label lookup, so it can never resolve to a real passage. Real evidence numbering
(`Session.allocate_label`) starts at `S1` and never collides with it. This variant has
been built and tested (`tests/test_seed_exchange.py`) but **not yet measured** against
the 18-question tuning set; the task brief's adoption gate is `supported_citation_rate`
improving by ≥ 0.15 without raising `evidence_dump_rate`.

**Proposed addition (§7.1 or new §7.5):** "The label `S0` is reserved and never
resolvable to a real evidence passage. An implementation may seed a new lesson's prompt
history with a fixed synthetic example exchange using `[S0]`, to demonstrate correct
per-sentence citation placement; it is part of the append-only prefix, never re-rendered
or altered within a lesson. Available, off by default; adopted then reverted the same
day -- see citation_experiment.md. Adopted 2026-09-20 on the single-turn A/B (pooled
n=54: cited-and-supported 0.09 -> 0.35 (+0.259), evidence_dump_rate tied at 0.04,
paired 19 wins / 5 losses / 30 ties), then reverted the same day when paired 6-minute
in-lesson soaks contradicted it (current: 34 turns, median answer 332 chars, cited_rate
1.000; seed: 32 turns, median answer 512 chars, cited_rate 0.000 -- 0 labels in the
whole lesson). `current` (unseeded) is the default again; `seed_exchange_s0` stays
selectable and tested."

**Why adopted, then reverted:** the single-turn A/B on the 18-question tuning set (3
runs, pooled n=54) met the task brief's gate (>= 0.15 `supported_citation_rate`
improvement, no `evidence_dump_rate` increase), so the seed was adopted as the default.
The same day, paired in-lesson soaks (same script, current vs seed) showed the seed
lengthens answers and does not make lesson citation placement reliable, so the default
was reverted to `current`; see docs/citation_experiment.md, "Seed exchange A/B" and
"In-lesson check and reversal", for the full numbers.
