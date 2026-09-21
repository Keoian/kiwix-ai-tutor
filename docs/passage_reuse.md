# Passage reuse across turns (`app.reuse_prior_passages`)

## What

Before this change, every turn's evidence packet pasted the full text of
every retrieved passage into that turn's tool message, even when a
passage's exact text was already sitting earlier in the same lesson's
append-only prompt log (`tutor.app.prompt.PromptLog`). A student grilling
one topic repeatedly (three follow-up questions all about DNA, say) got
the same passage text re-pasted verbatim on every turn: the model tended
to re-write the same explanatory essay instead of answering the specific
follow-up, `[S#]` labels kept colliding (`tutor/retrieval/hybrid/packer.py`
numbers each research response's labels from `S1`, independent of any
prior call), and the prompt log filled with duplicate text years before
the token budget forced real eviction.

With `app.reuse_prior_passages` (default **on**), `PromptLog.append_evidence`
now tracks which passage ids currently have their full text present
somewhere in the log (`PromptLog.held_ids()`, backed by the existing
`_seen_ids` bookkeeping). When a turn's packet contains a passage whose id
is already held:

- its full text is **not** pasted again;
- a one-line pointer is pasted in its place instead, e.g.:
  `[S2] (already shown above) DNA — Deoxyribonucleic acid is the molecule that…`
- the passage still gets its own `[S#]` label line in *this* turn's
  message (just short), so the model doesn't have to scroll back to find
  a citable label for it.

If **every** passage in a turn's packet is already held, a short host
note is appended to that turn's evidence preface:

> [Host note: every source above was already shown earlier in this
> lesson. Answer the student's specific question briefly using the
> sources already shown above; do not re-summarise the topic.]

A passage that was evicted (`PromptLog.evict`, spec §8) — meaning its
full text is no longer actually anywhere in the log, whether because
stage 1 dropped it as an uncited passage or stage 2 dropped its whole
turn and it was not cited enough to be re-protected — is **not** treated
as held. `held_ids()` is recomputed at the end of every `evict()` call
from the log's actual current contents (protected passages + surviving
turns), so a later turn re-pastes that passage's full text again, exactly
as before this change. Eviction itself, and the log's append-only/
byte-prefix guarantees, are unchanged; the pointer/note text is written
once, like everything else in a turn's entry, and never edited
afterwards.

Everything is append-only: an already-appended turn's entries are never
rewritten. `PromptLog.held_ids()` is rebuilt from the stored entries on
lesson resume (`tutor.app.lesson_state._rebuild_prompt_log`, unchanged —
it already walked every tool message's passages into `_seen_ids`), so the
held set survives a restart correctly.

Citation/attribution is untouched: `_retain_passages` (agent_loop.py)
always hands `Session.retain_passages` the **original**, full-text packet
returned by the research engine, before any pointer substitution. Host
attribution (`tutor.app.citations.attribute_sentences`) and the UI's
source chips read from `Session.known_passages()`, which is built from
that retained full-text store — never from what actually got pasted into
the prompt log. Only the text pasted into the model's prompt is
shortened; what the host uses to check and display citations is always
the full passage, for both new and pointer passages.

`app.reuse_prior_passages = false` reproduces the pre-existing behaviour
byte-for-byte: `PromptLog.append_evidence`'s original code path (silently
drop a passage whose id is already in `_seen_ids`, no pointer, no note)
runs unchanged.

## Forced-rewrite rounds are now covered

Since `d97f2b6` forced a query-rewrite round on every follow-up turn
(`docs/rewrite_on_weak_evidence.md`, `_run_forced_rewrite_round` in
`tutor/app/agent_loop.py`), that path ran on effectively every turn after
the lesson's first — but it built its own tool-result string directly via
`render_evidence(...)` and appended it with `PromptLog.append_tool_result`,
not `PromptLog.append_evidence`, so it never went through the held-id/
pointer bookkeeping above. In practice this meant reuse barely applied:
passages surfaced by a forced-rewrite round were re-pasted in full on
every later turn, and were never recorded as held either.

This is now fixed. Both `PromptLog.append_evidence` and
`_run_forced_rewrite_round`'s tool-result construction go through one
shared helper, `tutor.app.prompt.render_evidence_with_reuse`: given a
turn's passages, the log's current `held_ids()`, and the
`reuse_prior_passages` flag, it returns the rendered content (pointer
lines for already-held passages, the "already shown above" line format,
and the all-held host note when every passage this call is already
held), the newly-seen ids, and the display passages actually shown.
`PromptLog.append_evidence` uses it and updates `self._seen_ids` itself;
`_run_forced_rewrite_round` calls it directly (passing `log.held_ids()`)
and then calls the new `PromptLog.mark_held(newly_seen_ids)` to update the
same bookkeeping from outside `append_evidence`, since it appends via
`append_tool_result` (a plain string keyed by `tool_call_id`), not a
passages-shaped message.

A citation reminder line is appended by the shared helper itself in the
forced-round path (`citation_reminder=_CITATION_REMINDER`), since that
path's tool-result text is a literal string sent straight to the model,
unlike `append_evidence`'s passages-shaped log entry, whose reminder is
added later at wire-render time by `citations.render_evidence` inside
`_to_wire_messages`.

`reuse_prior_passages=False` still reproduces old-bytes behaviour for the
forced round too: `render_evidence_with_reuse`'s `False` branch drops an
already-held passage outright (no pointer, no note), and
`_run_forced_rewrite_round` falls back to the original
`render_evidence({"passages": merged_passages})` call whenever
`reuse_prior_passages` is false or the session has no `PromptLog`
(`use_log` false — some fakes in unit tests, and non-log message-list
sessions generally, which have no held-id concept to consult).

One consequence worth calling out: a follow-up turn's raw pre-search
packet is never appended to the log directly (see
`docs/rewrite_on_weak_evidence.md` — it is only ever merged in as
*backfill* behind the forced round's own rewritten-query passages,
`_lead_with_backfill`), so it was never at risk of being pasted in full
twice within the same turn even before this fix; that invariant is
covered by
`tests/test_agent_loop_followup.py::test_no_passage_pasted_twice_in_full_within_one_followup_turn`.
See also
`tests/test_agent_loop_followup.py::test_followup_forced_round_pastes_pointer_for_held_passage`
and `::test_followup_forced_round_off_setting_pastes_full_text_every_time`.

## Backfill (not implemented)

The task considered backfilling a turn's freed slots (from passages that
became pointers) with the next-ranked *unseen* passages, if cheap.
`tutor.retrieval.research.ResearchEngine.research()` has no
`exclude`/`offset` parameter to ask for "the next passages after the ones
I already have" without re-ranking from scratch — and this task
deliberately does not add one to the retrieval layer. So backfill is
**skipped**: a turn whose packet is entirely held passages gets *only*
the pointer lines (and the host note) — it does not automatically pull in
new passages to fill the space the pointers freed up.

## What is NOT yet measured

Whether the on-device model (Granite, via llama-server) actually attends
to a passage's full text several turns back in the context as reliably as
it does to text freshly pasted at the end of the prompt is unmeasured.
Pointer lines assume the model will look back at the earlier full text
when it sees "(already shown above)" — if long-range attention within the
same context window degrades meaningfully compared to recency, a
pointer-only turn could produce a *worse* answer than a full re-paste
would have, trading duplicate text for a less-grounded answer. This
should be checked with a live-model measurement (not run as part of this
change — see the sibling handoff note about llama-server timing runs in
progress) before assuming pointers are strictly better than the old
re-paste behaviour for the model's actual answer quality, as opposed to
just being cheaper on tokens.

## Measured (2026-09-21, live, `app.reuse_prior_passages` ON vs OFF)

Live run against the dev `llama-server` on `:8080` (Granite, 32K profile,
`config/archives.simplewiki_only.toml`), in-process via `TestClient`
(`data/passage_reuse_measure.py`, checkpointed to
`data/passage_reuse_measure_v2_20260921_090000.json`, 2 reps each
setting). One 6-turn lesson, deliberately grilling one topic the way a
student would:

1. "What's the largest molecule?"
2. "What about DNA?"
3. "Is it a molecule?"
4. "Yes but is it a molecule?"
5. "Tell me about DNA"
6. "How does it copy itself?"

`rewrite_on_followup` (the unconditional forced-rewrite round, `d97f2b6`)
was left at its default (on), so this measures reuse under exactly the
condition the scope note above flagged: the forced round is the path
that actually runs on every one of turns 2-6.

### Per-turn averages (2 reps)

| Turn | ON prompt tok | OFF prompt tok | ON cached tok | OFF cached tok | ON full/pointer | OFF full/pointer | ON wall s | OFF wall s | ON backed rate | OFF backed rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1837 | 1837 | 1105 | 1105 | 6 / 0 | 6 / 0 | 10.5 | 12.5 | 0.33-0.67 | 0.43-0.60 |
| 2 | 2769 | 2914 | 2084 | 2151 | 7 / 1 | 8 / 0 | 8.5 | 9.0 | 1.0 | 1.0 |
| 3 | 3503 | 3979 | 3110 | 3216 | 2 / 6 | 8 / 0 | 7.1 | 7.9 | 1.0 | 1.0 |
| 4 | 4243 | 5074 | 3836 | 4305 | 2 / 6 | 8 / 0 | 10.3 | 11.3 | 1.0 | 1.0 |
| 5 | 5006 | 6096 | 4568 | 5396 | 3-4 / 3 | 7-8 / 0 | 8.0 | 9.3 | 1.0 | 1.0 |
| 6 | 6090 | 7260 | 5344 | 6421 | 5-6 / 2-3 | 8 / 0 | 11.4 | 12.4 | 0.83-1.0 | 1.0 |

("full/pointer" = passages pasted in full vs as one-line pointers this
turn, counted from the rendered log's `[S#]` lines across both evidence
shapes — `PromptLog.append_evidence`'s `"passages"` messages and the
forced round's plain-string tool results.)

By turn 6, ON is running **~16% fewer prompt tokens** than OFF (6090 vs
7260) and correspondingly fewer cached tokens carried forward (5344 vs
6421) — the gap that was supposed to open up once the fix in this change
routed the forced round through the same pointer bookkeeping. `passages_pointer`
is 0 for every OFF turn (confirms OFF really does behave like the
pre-reuse code: no pointers, ever) and rises from turn 3 onward for ON,
confirming the forced round is in fact using pointers now, not just the
pre-search path (`test_followup_forced_round_pastes_pointer_for_held_passage`
covers this at the unit level; this is the live confirmation).

Wall-clock is consistently ~1-2s faster per turn under ON from turn 2
onward (e.g. turn 6: 11.4s vs 12.4s) — consistent with, but not fully
explained by, the smaller prompt; no attempt was made here to attribute
the wall-time delta to prompt-processing vs generation (see
`data/followup_latency_profile.py` for that kind of component
breakdown, not re-run here).

`backed_sentence_rate` (fraction of the answer's sentences
`citations.attribute_sentences` could attribute to a known passage) was
1.0 for both settings on every turn except turn 1 (weak/mixed — the
"largest molecule" question doesn't have one clean textbook answer in
this archive, independent of reuse) and turn 6 for one ON rep (0.833,
one un-attributed sentence in a `[S#]`-cited "DNA replicates itself
through..." answer). No systematic ON-vs-OFF difference in citation
grounding was observed.

**Is the tutor still rewriting the same essay?** Word-overlap (Jaccard)
between each DNA-topic turn's answer and every earlier DNA-topic answer
in the same rep:

- Turns 3-5 ("Is it a molecule?" / "Yes but is it a molecule?" / "Tell me
  about DNA") are near-identical single-sentence answers
  ("Yes, DNA is a molecule.") in **both** ON and OFF — Jaccard 0.97-1.0
  between them in 3 of 4 reps. This is not a reuse-setting effect: the
  model gives the same terse answer regardless of whether it's looking at
  pointers or fresh full text, because the question itself keeps asking
  the same yes/no thing. One OFF rep and one ON rep show lower overlap
  (0.17-0.42) on turn 3, i.e. some rep-to-rep variance exists but it
  doesn't correlate with the setting.
- Turn 6 ("How does it copy itself?") reliably breaks the loop in every
  rep, both settings — the answer shifts to "DNA replication" content
  with Jaccard 0.12-0.42 against the earlier turns. So the concern in the
  original scope note (pointers causing worse long-range attention to
  older full text, visible as continued essay-rewriting) was not observed:
  the repetition present is a model/question-phrasing behavior already
  present in the OFF baseline, not something reuse introduces or worsens.

### What was and was not verified

**Measured directly:** prompt/cached token counts, full-vs-pointer
passage counts per turn (confirming the Part-A fix actually changes
forced-round behavior live, not just in unit tests), wall-clock per turn,
`backed_sentence_rate`, and cross-turn answer-text overlap — all live
against the real model, 2 reps per setting.

**Inferred, not independently isolated:** the wall-clock improvement is
plausible from the smaller prompt but was not decomposed into
prompt-processing vs generation time here (would need the
`followup_latency_profile.py`-style per-call instrumentation split by
`timings.prompt_ms`/`predicted_ms`, not done in this run).

**Not verified / still open** (same caveat as the "What is NOT yet
measured" section above, now narrower): whether the model's citation
*correctness* (not just `backed_sentence_rate`, which only checks
whether a supporting passage exists, not whether the model's stated
reasoning is right) degrades when a passage is several turns back as a
pointer vs freshly pasted — this run's `backed_sentence_rate` staying at
1.0 for both settings is a good sign but is a coarse proxy, and only 2
reps of one 6-turn lesson on one topic pair (molecule/DNA) were run;
this is not a claim that reuse is quality-neutral across topics or
longer lessons approaching eviction.

**Recommendation: keep `app.reuse_prior_passages` ON.** It measurably
reduces prompt/cached tokens from turn 3 onward once routed through the
forced-rewrite round (this change's whole point — before it, the setting
barely applied since the forced round runs on every follow-up turn), wall
time is not worse (mildly better in this run), and no citation-quality or
repetition regression was observed in the live comparison. Nothing here
found the "long-range attention to pointer text" risk the original scope
note flagged; if regressed citation correctness is spotted in normal use
later, `app.reuse_prior_passages = false` is a one-line config revert to
the byte-identical old behavior with no other code path changes.
