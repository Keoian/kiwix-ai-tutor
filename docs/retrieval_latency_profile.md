# Retrieval latency: child-side profile (Baseline v8, investigate-only)

Measured with `data/profile_child_v8.py`: captured the real op sequence
`research.py` issues (via a wrapped `ZimWorker.request`) for all 42
`simplewiki_questions.jsonl` tuning questions, then replayed those ops
in-process (no IPC) against `C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim`,
timing each sub-op with `time.perf_counter`. Held-out split never touched.

## (1) Per-op-type share of child time

| op | calls | calls/req | mean ms | total ms | share |
|---|---|---|---|---|---|
| search_fulltext | 205 | 4.88 | 198.7 | 40,739 | **96.2%** |
| search_titles | 252 | 6.00 | 4.1 | 1,021 | 2.4% |
| fetch_entry | 252 | 6.00 | 1.3 | 320 | 0.8% |
| estimated_matches | 93 | 2.21 | 2.7 | 252 | 0.6% |

Total replayed: 42,332 ms / 42 requests = 1,008 ms/request (matches the
~1.31 s live mean once IPC/parent overhead is added back).

Root cause isolated (`data/snippet_cost_v8.py`, one real fulltext hit set,
20 kept hits): `_resolve_hits(..., with_snippet=True)` = **407.9 ms**;
`with_snippet=False` = **0.09 ms**. `search_fulltext`'s own Xapian query is
near-free; nearly all its cost is `fetch_entry` + bs4 text extraction done
*inside* `_resolve_hits` for every one of the up to 20 kept hits, run for
every `search_fulltext` call (including the small single-result entity
searches, which also pass `with_snippet=True`).

## (2) Duplicate work

- Within a request: **108 duplicate (op, key) calls** out of 802 total
  (13.5%) — e.g. the entity `estimated_matches`/`search_titles`/
  `search_fulltext` loop re-issuing a term already looked up earlier in the
  same request.
- Across the 42 requests: **185 repeated (op, key) pairs** (23%) — mostly
  shared entity terms across different tuning questions. Real user traffic
  repetition rate is unmeasured ("needs eval"); this number is this
  question set only.

## (3) Concurrency / GIL

Two `search_fulltext` calls, two threads vs serial, same open `Archive`:
serial 684.3 ms, threaded 561.1 ms → **1.22x speedup**, not ~2x. libzim/Xapian
does **not** fully release the GIL during a search call — threads inside one
worker process would give only partial overlap. True parallelism needs a
second worker **process**, which reintroduces IPC overhead (today's
already-identified 76% wait) rather than removing it.

## (4) Ops whose result rarely survives to top-5

The entity `estimated_matches` loop (93 calls, 2.7 ms mean, 0.6% share) and
the ~14 discarded fulltext hits per call whose snippets are built and then
never packed are cheap individually but not free: snippet-building for
discarded candidates is exactly the 407.9 ms cost above. Whether skipping
snippets for hits outside the eventual top-N changes recall — **needs
eval** (rerun `eval/run_retrieval_eval.py --split tuning`), not guessed here.

## Ranked candidate fixes

1. **Defer snippet generation to only the final kept candidates** (compute
   `with_snippet=False` in `_resolve_hits` during the initial `search_fulltext`
   /`multi` fan-out, build snippets afterward only for the articles that
   survive ranking). Estimated saving: **~400 ms/request** (measured basis:
   `data/snippet_cost_v8.py`, one representative fulltext call). Recall
   risk: none expected — ranking uses hit rank/title, not snippet text, but
   confirm no code path reads `.snippet` pre-ranking. Size: small (one
   function's call site + a follow-up snippet pass).
2. **Memoize (op, key) results within one request** before batching `multi`
   ops, so a term already resolved this request is never re-sent to the
   child. Estimated saving: **~136 ms/request** (measured basis: 108/802
   duplicate calls x 42,332 ms total / 42 requests). Recall risk: none (same
   result, just not recomputed). Size: small (dict cache keyed by (op, query
   term/path) in `research.py`'s op-batching helpers).
3. **Cross-request LRU cache for `search_fulltext`/`estimated_matches`**
   keyed by (archive fingerprint, query, limit), same pattern as the
   existing `_idf_cache`. Estimated saving: unmeasured for real traffic
   ("needs eval" — 185/802 dup rate is this 42-question set only, not
   representative of production query diversity). Recall risk: none if
   invalidated on fingerprint change. Size: small–medium.
4. **Second worker process for independent ops** (real parallelism, since
   threads only got 1.22x). Estimated saving: up to ~2x on the batched
   independent ops, but reintroduces IPC coordination cost already flagged
   as 76% of wall time — net saving uncertain without a live measurement.
   Recall risk: none (same ops, different scheduling). Size: large.
