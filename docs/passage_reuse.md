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
