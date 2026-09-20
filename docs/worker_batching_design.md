# Worker batching design (`multi` op)

## Current round-trips (worst case, one archive)
`search_fulltext`+`search_titles` (2) + fallback per-token searches (up to
`_MAX_FALLBACK_SEARCHES`=6) + `estimated_matches` per entity term (up to
`_MAX_ENTITY_TERM_SEARCHES`=4) + entity `search_titles`+snippet
`search_fulltext` pairs (up to `_ENTITY_TITLE_LIMIT`=3 each, 6) +
relaxed-AND `search_fulltext` (up to 2) + topic_hint `search_titles` (1) +
`fetch_entry` per top article (up to `_TOP_N_ARTICLES`=6). Sum ~21, matching
the ~20 measured in `docs/retrieval_baseline.md` "Baseline v5" (each op
pays full pipe send/poll/recv even for near-zero work).

## `multi` wire format
Child op `("multi", {"ops": [(op, kwargs), ...], "deadline_s": t})`. Reply
`("ok"|"partial", results, None)`; `results` is a list, same order/length
as input, each entry `{"status": "ok"|"error", "value", "error"}`. A
raising sub-op is caught (same try/except as today per op) and reported
only in its own entry, never aborting the rest of the batch.

## Deadline across a batch
Not preemptible mid-op (libzim isn't interruptible): the child checks
elapsed time between sub-ops against `deadline_s`; once exceeded, unrun
sub-ops get `error: "deadline exceeded before this sub-op"` and the outer
status is `"partial"`. Parent's `conn.poll` kill/restart still covers one
wedged sub-op. Batch reuses `_PER_OP_DEADLINE_CAP_S` (3.0s) as its deadline;
cap length at 16 sub-ops (above the ~10 worst case per archive below).

## Call sites to batch in research.py
1. `_search_with_fallback` initial fulltext+titles (2->1).
2. Per-term `estimated_matches` loop, up to 4 (4->1).
3. Entity title+snippet loop, up to 6 (6->1).
4. `fetch_entry` loop over `top_paths`, up to 6 (6->1).

Target: ~21 -> **4-6** round-trips (conditional fallback/relaxed-AND paths
stay separate; they depend on earlier results).
