# Retrieval baseline (WP-B/eval)

## Pipeline as built

`tutor.retrieval.research.ResearchEngine.research()` is the single entry
point. Per request:

1. **Route archives** by tier/subject: `Registry.for_subject(topic_hint)`
   (tier 1 + tier 2 always, tier 3 only when `topic_hint` matches a
   subject), further gated so tier 3 is consulted only for
   procedural/pedagogical phrasing ("how do I ...", "how would you
   explain ...", etc. -- see `_is_procedural`).
2. **Per archive, via `ZimWorker`** (out-of-process, hard-deadline
   bounded): full-text (Xapian) search and title-suggestion search over
   the pinned tokenizer's joined query. If the joined query returns
   nothing (a natural-language question can zero out Xapian's
   AND-of-terms parser on a single stray non-corpus word), falls back to
   merging each token's own single-term hits.
3. **Fuse article rankings** (fulltext ranking + title ranking) via RRF
   and fetch the top-N entries (`_TOP_N_ARTICLES = 8`) through the
   worker.
4. **Build bundles** (`zim.bundle.build_bundle`, single HTML parse) and
   **split into passages** (`hybrid.passages.split_passages`, never
   crossing a section boundary, exact `text[start:end]` offsets).
5. **BM25** (`hybrid.lexical.BM25`, hand-rolled, positive IDF) over the
   candidate passage pool for this archive, using the same pinned
   tokenizer (lowercase, unicode-aware `\w+`, English stopwords dropped).
6. **RRF** (`hybrid.rrf.rrf_fuse`) of (article rank, BM25 rank) per
   passage, `k=60`, all weights 1, deterministic first-appearance tie
   break.
7. Candidates from all consulted archives are merged and sorted by fused
   score, then **diversity-capped** (`hybrid.diversity.cap_per_article`,
   default 2 passages/article) and **packed** to the token budget
   (`hybrid.packer.pack`, greedy in rank order, skip-too-large-and-
   continue, `estimate_tokens = ceil(len(text)/3.5)`).
8. Every packed passage is **snapshotted** (`SnapshotStore.put`) before
   being returned.
9. Response carries `version`, `status` (`ok`/`partial`/`empty`/`error`),
   `coverage` (`{"weak": bool}`), `archives_consulted` (each with `id` and
   `storage` class), and `timings`.

Ranking flags (`RankingFlags`: `title_boost`, `mention_penalty`,
`heading_affinity`, `lead_augmentation`) all default OFF; when off each
hook in `hybrid/ranking.py` is a pure no-op, so the pipeline above runs
unmodified. This is intentionally not a generic ranking framework -- one
entry point, minimal optional hooks.

Every log line emitted by the pipeline (`logging`, JSON-ish dict) carries
the archive's storage class (`ssd`/`hdd`) alongside its id, tier, and hit
counts.

## Parameters

| Parameter | Value |
|---|---|
| BM25 k1 | 1.2 |
| BM25 b | 0.75 |
| BM25 IDF | positive (floored at 0) |
| RRF k | 60 |
| RRF weights | 1 (all rankings) |
| Full-text candidate limit / archive | 20 |
| Title-suggestion candidate limit / archive | 10 |
| Top-N articles fetched / archive | 8 |
| Diversity cap | 2 passages / article |
| Token estimator | `ceil(len(text) / 3.5)` chars-per-token |
| Default token budget | 2000 |
| Default hard deadline | 8.0 s |
| Per-op deadline cap | 3.0 s |

## Fixture-eval numbers (this test run)

`eval/questions/fixture_questions.jsonl` (12 questions) against the
in-memory fixture ZIM built by `tests/zim_fixtures.py`:

| metric | recall@1 | recall@3 | recall@5 | mrr | mean latency (s) | n |
|---|---|---|---|---|---|---|
| overall | 0.917 | 1.000 | 1.000 | 0.958 | 0.141 | 12 |
| direct | 0.900 | 1.000 | 1.000 | 0.950 | 0.141 | 10 |
| why_how | 1.000 | 1.000 | 1.000 | 1.000 | 0.140 | 2 |

(`tests/test_retrieval_eval.py::test_end_to_end_fixture_recall_at_5_meets_bar`
asserts `recall@5 >= 0.75`; the fixture run above clears that bar.)

## Real tier-1 baseline: PENDING

`C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim` is **not yet
available** -- only a `.part` file exists at that path (~797 MB so far,
download in progress via `fetch.log`). No evaluation was run against it;
this baseline will be extended once the download completes and the
`.part` suffix is gone.
