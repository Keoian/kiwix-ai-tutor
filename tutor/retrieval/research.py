"""Single entry point: research() -- the whole hybrid retrieval pipeline.

Ours: no donor code. Pipeline per request (see docs/plan/
offline_tutor_kiwix_reuse_plan.md and offline_tutor_spec_v0.3.md §7):
route archives by tier/subject -> per archive via worker: fulltext + title
candidates (deadline-bounded) -> fetch top-N entries -> build bundles ->
split passages -> BM25 over candidate passages -> RRF of (article rank,
BM25 rank) -> diversity cap -> pack to budget -> snapshot every returned
passage -> versioned response with coverage flags, timings,
archives_consulted (including storage class).

This module intentionally implements exactly one entry point,
:class:`ResearchEngine` / :meth:`ResearchEngine.research`. It is not a
generic RAG framework: ranking flags (:class:`RankingFlags`) default OFF
and are documented no-op hooks (see ``tutor.retrieval.hybrid.ranking``)
unless a caller explicitly enables one.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tutor.retrieval.hybrid.dense import DenseIndex
from tutor.retrieval.hybrid.diversity import cap_per_article
from tutor.retrieval.hybrid.lexical import BM25, tokenize
from tutor.retrieval.hybrid.packer import estimate_tokens, pack
from tutor.retrieval.hybrid.passages import split_passages
from tutor.retrieval.hybrid.ranking import (
    RankingFlags,
    apply_heading_affinity,
    apply_lead_augmentation,
    apply_mention_penalty,
    apply_title_boost,
)
from tutor.retrieval.hybrid.rrf import rrf_fuse
from tutor.retrieval.registry import ArchiveEntry, Registry
from tutor.retrieval.snapshots import SnapshotStore
from tutor.retrieval.zim.archive import fingerprint as fingerprint_archive
from tutor.retrieval.zim.bundle import build_bundle
from tutor.retrieval.zim.worker import WorkerResult, ZimWorker

logger = logging.getLogger(__name__)

RESPONSE_VERSION = 1

_FULLTEXT_LIMIT = 20
_TITLE_LIMIT = 10
_TOP_N_ARTICLES = 6  # spec §7.2 step 4: "Top 6 articles (cap 10)."
_TOP_N_ARTICLES_CAP = 10
_DIVERSITY_CAP = 2
_DEFAULT_BUDGET_TOKENS = 2000
_DEFAULT_HARD_DEADLINE_S = 8.0
_DEFAULT_SOFT_DEADLINE_S = 3.0
_PER_OP_DEADLINE_CAP_S = 3.0
_MIN_OP_DEADLINE_S = 0.05
_QUEUE_SLACK_S = 0.05
_THREAD_JOIN_TIMEOUT_S = 0.5

# Reuse plan §7.2 "Dense article top 16": the dense sidecar's own
# candidate list, fused at article level with the lexical/title rankings
# below (not blended with raw cosine/BM25 scores directly).
_DENSE_ARTICLE_TOP = 16
_EMBED_QUERY_TIMEOUT_S = 1.0

_PROCEDURAL_RE = re.compile(r"\bhow (do|would|can|should) (i|you|we)\b", re.IGNORECASE)

# Coverage-flag threshold (spec §7.3 "coverage signals are explainable").
# Tuned on the tuning split of eval/questions/simplewiki_questions.jsonl --
# see docs/retrieval_baseline.md "Baseline v2" for the before/after tables.
_COVERAGE_TERM_THRESHOLD = 0.6


def _is_procedural(query: str) -> bool:
    """Whether ``query`` reads as a "how do I ..."-style procedural ask."""
    return bool(_PROCEDURAL_RE.search(query))


def _query_terms(query: str, topic_hint: str | None) -> set[str]:
    """Content terms used for coverage checks: the query plus, per spec §5/§7.1,
    the host-supplied ``topic_hint`` (current subject), which is what lets an
    elliptical follow-up ("what about its moons?") still be judged covered.
    """
    terms = set(tokenize(query))
    if topic_hint:
        terms |= set(tokenize(topic_hint))
    return terms


# Coverage is checked over the top few candidates, not just the single
# best-scored one: RRF/BM25 rank-1 is not always the semantically correct
# passage (see docs/retrieval_baseline.md failure analysis), so requiring
# *only* rank-1 to look relevant would wipe out perfectly good evidence
# sitting at rank 2-3. Tuned on the tuning split alongside the term
# threshold.
_COVERAGE_CANDIDATES_CHECKED = 3


def _best_coverage(
    candidates: list[dict[str, Any]],
    query_terms: set[str],
    topic_hint_terms: frozenset[str],
) -> dict[str, Any]:
    """The least-weak :func:`compute_coverage` result among the top-scored
    ``_COVERAGE_CANDIDATES_CHECKED`` candidates (any one of them being a
    strong hit is enough to call the query covered). ``topic_hint_terms``
    is accepted for symmetry with the query-term merge already done in
    ``query_terms`` (see ``_query_terms``); it is not otherwise needed
    here since a topic_hint-driven title match already shows up as a
    normal ``title_match`` once its tokens are unioned into the query.
    """
    del topic_hint_terms
    if not candidates:
        return compute_coverage(query_terms, None, None)
    top = sorted(candidates, key=lambda c: -c["score"])[:_COVERAGE_CANDIDATES_CHECKED]
    best: dict[str, Any] | None = None
    for c in top:
        cov = compute_coverage(query_terms, c["title"], c["text"])
        if not cov["weak"]:
            return cov
        if best is None or cov["term_coverage"] > best["term_coverage"]:
            best = cov
    assert best is not None
    return best


def compute_coverage(
    query_terms: set[str], title: str | None, text: str | None
) -> dict[str, Any]:
    """Explainable coverage flags for a single candidate.

    Per spec §7.3 ("coverage signals are explainable (facets represented,
    entities missing, limits hit)") and §6 ("tier 2 only if coverage flags
    are weak after tier 1"). ``weak`` is True when there is no candidate at
    all, or the candidate shares no query/topic-hint terms with the
    query's content words (``query_terms`` already includes the
    host-supplied topic_hint's tokens -- see ``_query_terms``) and its
    title does not match any of them either.
    """
    if not query_terms or title is None or text is None:
        return {"term_coverage": 0.0, "title_match": False, "weak": True}

    passage_terms = set(tokenize(text))
    title_terms = set(tokenize(title))
    term_coverage = len(query_terms & passage_terms) / len(query_terms)
    title_match = bool(query_terms & title_terms)
    weak = (not title_match) and term_coverage < _COVERAGE_TERM_THRESHOLD
    return {"term_coverage": term_coverage, "title_match": title_match, "weak": weak}


@dataclass(frozen=True)
class ResearchPassage:
    """One packed, citable passage returned by :meth:`ResearchEngine.research`."""

    label: str
    passage_id: str
    archive_id: str
    title: str
    path: str
    heading_path: tuple[str, ...]
    text: str
    start: int
    end: int
    score: float
    kind: str
    estimated_tokens: int
    # Subset of ("lexical", "dense"): which candidate-generation path(s)
    # surfaced this passage's article. Defaults to ("lexical",) so the
    # field is always present in to_dict() output even when dense
    # retrieval was never configured.
    sources: tuple[str, ...] = ("lexical",)


@dataclass
class ResearchResponse:
    """Versioned response of one :meth:`ResearchEngine.research` call."""

    version: int
    status: str  # "ok" | "partial" | "empty" | "error"
    passages: list[ResearchPassage]
    coverage: dict[str, Any]
    archives_consulted: list[dict[str, Any]]
    timings: dict[str, Any] = field(default_factory=dict)
    dense_used: bool = False
    dense_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "passages": [dataclasses.asdict(p) for p in self.passages],
            "coverage": self.coverage,
            "archives_consulted": self.archives_consulted,
            "timings": self.timings,
            "dense_used": self.dense_used,
            "dense_note": self.dense_note,
        }


def _call_worker(worker: Any, op: str, *, deadline_s: float, **kwargs: Any) -> WorkerResult | None:
    """Call ``worker.request`` off-thread so a misbehaving worker can never
    block the caller past ``deadline_s`` (+ small slack), regardless of what
    the worker implementation itself does with its own ``deadline_s``.
    """
    result_box: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_box.put(worker.request(op, deadline_s=deadline_s, **kwargs))
        except Exception as exc:  # noqa: BLE001 - reported as a worker error
            result_box.put(
                WorkerResult(status="error", value=None, error=str(exc), elapsed_s=0.0)
            )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    try:
        return result_box.get(timeout=deadline_s + _QUEUE_SLACK_S)
    except queue.Empty:
        # The worker call outlived its deadline. Kill the child so the
        # in-flight request unblocks (its `conn.poll`/`conn.recv` inside
        # `worker.request` returns instead of being abandoned mid-call),
        # then join the thread briefly so the next `_call_worker` on this
        # worker cannot overlap with a still-running previous request.
        interrupt = getattr(worker, "interrupt", None)
        interrupted = False
        if callable(interrupt):
            try:
                interrupt()
                interrupted = True
            except Exception:  # noqa: BLE001 - best-effort
                pass
        if interrupted:
            # Killing the child should unblock the in-flight request almost
            # immediately; join briefly to avoid overlapping the next call
            # to this worker with a request thread that is still finishing.
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        return None


def _call_bounded(
    fn: Callable[[], Any], *, timeout_s: float
) -> tuple[Any, str | None]:
    """Call ``fn`` off-thread, bounded by ``timeout_s``.

    Returns ``(value, None)`` on success or ``(None, note)`` on timeout or
    exception. Used for ``embed_query``, which -- unlike the ZIM worker --
    has no ``interrupt()`` hook, so a timed-out call's thread is simply
    abandoned (daemon thread; it cannot outlive the process and its result
    is discarded).
    """
    result_box: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_box.put(("ok", fn()))
        except Exception as exc:  # noqa: BLE001 - reported as a note
            result_box.put(("error", str(exc)))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    try:
        status, value = result_box.get(timeout=max(timeout_s, _MIN_OP_DEADLINE_S))
    except queue.Empty:
        return None, "embed_query timed out"
    if status == "error":
        return None, f"embed_query failed: {value}"
    return value, None


class ResearchEngine:
    """The single research() entry point over a :class:`Registry` of archives."""

    def __init__(
        self,
        registry: Registry,
        *,
        snapshot_store: SnapshotStore,
        cache_dir: Path,
        worker_factory: Any = None,
        ranking_flags: RankingFlags | None = None,
        hard_deadline_s: float = _DEFAULT_HARD_DEADLINE_S,
        soft_deadline_s: float = _DEFAULT_SOFT_DEADLINE_S,
        default_budget_tokens: int = _DEFAULT_BUDGET_TOKENS,
        top_n_articles: int = _TOP_N_ARTICLES,
        dense_indexes: Mapping[str, DenseIndex] | None = None,
        embed_query: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self._registry = registry
        self._snapshot_store = snapshot_store
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._worker_factory = worker_factory or (lambda path: ZimWorker(path))
        self._ranking_flags = ranking_flags or RankingFlags()
        # Keyed by archive id (Registry.ArchiveEntry.id). A dense index for
        # an archive not in the registry, or whose manifest fingerprint no
        # longer matches that archive's current fingerprint, is ignored
        # with a note -- never fatal (see ``_dense_hits_for``).
        self._dense_indexes: dict[str, DenseIndex] = dict(dense_indexes or {})
        self._embed_query = embed_query
        self._hard_deadline_s = hard_deadline_s
        self._soft_deadline_s = soft_deadline_s
        self._default_budget_tokens = default_budget_tokens
        # Spec §7.2 step 4: "Top 6 articles (cap 10)" -- the spec's cap wins
        # over any caller-supplied config value.
        self._top_n_articles = min(max(top_n_articles, 1), _TOP_N_ARTICLES_CAP)
        self._workers: dict[str, Any] = {}
        self._fingerprints: dict[str, str] = {}
        self._response_cache: dict[tuple[Any, ...], ResearchResponse] = {}

    def close(self) -> None:
        for worker in self._workers.values():
            try:
                worker.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._workers.clear()

    def _get_worker(self, entry: ArchiveEntry) -> Any:
        worker = self._workers.get(entry.id)
        if worker is None:
            worker = self._worker_factory(entry.path)
            self._workers[entry.id] = worker
        return worker

    def _fingerprint_digest(self, entry: ArchiveEntry) -> str:
        digest = self._fingerprints.get(entry.id)
        if digest is None:
            digest = fingerprint_archive(entry.path).digest
            self._fingerprints[entry.id] = digest
        return digest

    def _route(self, query: str, topic_hint: str | None) -> list[ArchiveEntry]:
        entries = self._registry.for_subject(topic_hint)
        if not _is_procedural(query):
            entries = [e for e in entries if e.tier != 3]
        return entries

    def _dense_hits_for(
        self, entry: ArchiveEntry, query_vec: Sequence[float] | None
    ) -> tuple[list[str], str | None]:
        """Dense top-K article paths for ``entry``, or ``([], note)``.

        Never raises: a missing dense index, a stale (fingerprint
        mismatch) sidecar, a sidecar with no ``paths.txt`` (pre-backfill),
        or a search-time error are all reported as a note and treated as
        "no dense candidates for this archive" rather than a fatal error.
        """
        if query_vec is None:
            return [], None
        index = self._dense_indexes.get(entry.id)
        if index is None:
            return [], None
        try:
            archive_digest = self._fingerprint_digest(entry)
            if index.manifest.archive_digest != archive_digest:
                return [], (
                    f"dense sidecar for {entry.id!r} is stale (fingerprint mismatch); ignored"
                )
            if index.paths is None:
                return [], f"dense sidecar for {entry.id!r} has no paths.txt; ignored"
            hits = index.search(list(query_vec), k=_DENSE_ARTICLE_TOP)
        except Exception as exc:  # noqa: BLE001 - dense is best-effort
            return [], f"dense search failed for {entry.id!r}: {exc}"
        paths: list[str] = []
        for hit in hits:
            path = hit[1] if len(hit) == 3 else None
            if path is not None and path not in paths:
                paths.append(path)
        return paths, None

    def _process_archive(
        self,
        entry: ArchiveEntry,
        query: str,
        remaining: Any,
        keywords: list[str] | None = None,
        topic_hint: str | None = None,
        query_vec: Sequence[float] | None = None,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        worker = self._get_worker(entry)
        timed_out = False

        # Xapian's own query parser chokes on stopword/punctuation-heavy
        # natural-language questions (an all-stopword or all-AND-required
        # query can yield zero hits even when the topic is clearly present),
        # so search uses the same pinned tokenizer as BM25 rather than the
        # raw question string.
        tokens = tokenize(query)
        # Spec §7.2 step 1: "plus the model keywords if present" -- append
        # the (already-validated) model-supplied keywords to the lexical
        # query terms used for candidate generation.
        if keywords:
            for kw in keywords:
                tokens.extend(tokenize(kw))
        # Spec §5 step 3 / §7.1: the host-supplied topic_hint (current
        # subject) is the referent for an elliptical follow-up ("what about
        # its moons?"), so it feeds candidate search and ranking exactly
        # like model-supplied keywords do.
        topic_hint_tokens = tokenize(topic_hint) if topic_hint else []
        tokens.extend(topic_hint_tokens)
        search_query = " ".join(tokens)

        def _op_deadline() -> float:
            return max(_MIN_OP_DEADLINE_S, min(_PER_OP_DEADLINE_CAP_S, remaining()))

        def _search(op: str, joined_query: str, limit: int) -> tuple[list[Any], bool]:
            res = _call_worker(
                worker, op, deadline_s=_op_deadline(), query=joined_query, limit=limit
            )
            if res is None or res.status != "ok":
                return [], True
            return (res.value or []), False

        def _search_with_fallback(op: str, limit: int) -> tuple[list[Any], bool, bool]:
            # Xapian's query parser ANDs bare terms together, so a natural-
            # language question with a stray non-corpus word (e.g. "tell",
            # "about") can zero out an otherwise-good query. When the joined
            # query returns nothing, fall back to merging each token's own
            # single-term hits (deduplicated, first-seen order). The
            # returned ``used_fallback`` flag is carried onto each result
            # (see the "used_fallback" key below) purely for observability
            # -- coverage itself is judged by term/title overlap (see
            # ``compute_coverage``), not by which search path found it.
            hits, failed = _search(op, search_query, limit)
            if hits or failed or not tokens:
                return hits, failed, False
            merged: list[Any] = []
            seen: set[str] = set()
            any_failed = False
            for term in tokens:
                term_hits, term_failed = _search(op, term, limit)
                any_failed = any_failed or term_failed
                for hit in term_hits:
                    if hit.path not in seen:
                        seen.add(hit.path)
                        merged.append(hit)
            return merged[:limit], any_failed, True

        fulltext_hits, fulltext_failed, fulltext_fallback = _search_with_fallback(
            "search_fulltext", _FULLTEXT_LIMIT
        )
        title_hits, title_failed, title_fallback = _search_with_fallback(
            "search_titles", _TITLE_LIMIT
        )
        timed_out = timed_out or fulltext_failed or title_failed
        # Based on the full-text fallback only: a full-text AND-of-terms
        # search failing outright is the strong signal that the query's
        # content words never co-occur in any article. Title-suggestion
        # search is inherently fuzzy/prefix-based and can independently
        # return a single-word match (e.g. "tell" -> "Tell Me It's Real")
        # without ever needing its own fallback, so it is not treated as
        # corroborating evidence here.
        used_fallback = fulltext_fallback

        logger.info(
            {
                "event": "archive_searched",
                "archive_id": entry.id,
                "storage": entry.storage,
                "tier": entry.tier,
                "fulltext_hits": len(fulltext_hits),
                "title_hits": len(title_hits),
                "timed_out": timed_out,
            }
        )

        dense_paths, dense_note = self._dense_hits_for(entry, query_vec)

        if not fulltext_hits and not title_hits and not dense_paths:
            return [], timed_out, dense_note

        # Reuse plan §7.2: three independent rankings (dense semantic
        # search, title, body/full-text) fused via RRF at the article
        # level, not blended as raw scores.
        fused_articles = rrf_fuse(
            [[h.path for h in fulltext_hits], [h.path for h in title_hits], dense_paths]
        )
        top_paths = [path for path, _ in fused_articles[: self._top_n_articles]]
        lexical_paths = {h.path for h in fulltext_hits} | {h.path for h in title_hits}
        dense_paths_set = set(dense_paths)

        # Spec §5 step 3: topic_hint is "the current subject" -- i.e. an
        # entity the lesson is already about, not merely a keyword to
        # search for. An elliptical follow-up's own words ("what made it
        # explode?") may not out-rank an unrelated but lexically closer
        # article, so the topic_hint's own title is looked up directly and
        # guaranteed a slot in the candidate pool rather than left to
        # compete purely on the bare query's fused rank.
        if topic_hint and topic_hint.strip() not in top_paths:
            hint_hits, hint_failed = _search("search_titles", topic_hint, 1)
            timed_out = timed_out or hint_failed
            if hint_hits:
                hint_path = hint_hits[0].path
                lexical_paths.add(hint_path)
                if hint_path in top_paths:
                    top_paths.remove(hint_path)
                top_paths = [hint_path, *top_paths][: self._top_n_articles + 1]

        article_rank = {path: i for i, path in enumerate(top_paths)}

        fingerprint_digest = self._fingerprint_digest(entry)

        candidate_passages: list[Any] = []
        for path in top_paths:
            if remaining() <= 0:
                timed_out = True
                break
            fetch_deadline = max(_MIN_OP_DEADLINE_S, min(_PER_OP_DEADLINE_CAP_S, remaining()))
            fetch_res = _call_worker(worker, "fetch_entry", deadline_s=fetch_deadline, path=path)
            if fetch_res is None or fetch_res.status != "ok":
                timed_out = True
                continue
            fetched = fetch_res.value
            bundle = build_bundle(fetched.html, path=fetched.path, title=fetched.title)
            candidate_passages.extend(
                split_passages(bundle, fingerprint_digest=fingerprint_digest, archive_id=entry.id)
            )

        if not candidate_passages:
            return [], timed_out, dense_note

        tokenized_query = tokenize(query) + topic_hint_tokens
        bm25 = BM25([tokenize(p.text) for p in candidate_passages])
        bm25_scores = bm25.scores(tokenized_query)
        bm25_order = sorted(
            range(len(candidate_passages)), key=lambda i: (-bm25_scores[i], i)
        )
        bm25_ranking_ids = [candidate_passages[i].passage_id for i in bm25_order]

        article_order = sorted(
            range(len(candidate_passages)),
            key=lambda i: article_rank.get(candidate_passages[i].path, len(top_paths)),
        )
        article_ranking_ids = [candidate_passages[i].passage_id for i in article_order]

        fused = rrf_fuse([article_ranking_ids, bm25_ranking_ids])
        passages_by_id = {p.passage_id: p for p in candidate_passages}

        results: list[dict[str, Any]] = []
        for passage_id, score in fused:
            p = passages_by_id[passage_id]
            sources: list[str] = []
            if p.path in lexical_paths:
                sources.append("lexical")
            if p.path in dense_paths_set:
                sources.append("dense")
            if not sources:
                sources.append("lexical")
            results.append(
                {
                    "passage_id": p.passage_id,
                    "archive_id": p.archive_id,
                    "title": p.title,
                    "path": p.path,
                    "heading_path": p.heading_path,
                    "text": p.text,
                    "start": p.start,
                    "end": p.end,
                    "score": score,
                    "kind": entry.kind,
                    "used_fallback": used_fallback,
                    "sources": tuple(sorted(sources)),
                }
            )
        return results, timed_out, dense_note

    def research(
        self,
        query: str,
        *,
        keywords: list[str] | None = None,
        budget_tokens: int | None = None,
        deadline_s: float | None = None,
        soft_deadline_s: float | None = None,
        topic_hint: str | None = None,
    ) -> ResearchResponse:
        keywords_key = tuple(keywords) if keywords else None
        cache_key = (query, keywords_key, budget_tokens, deadline_s, topic_hint)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            return dataclasses.replace(cached, timings={**cached.timings, "cache_hit": True})

        started = time.monotonic()
        hard_deadline = deadline_s if deadline_s is not None else self._hard_deadline_s
        soft_deadline = soft_deadline_s if soft_deadline_s is not None else self._soft_deadline_s
        # The soft deadline can never exceed the hard one.
        soft_deadline = min(soft_deadline, hard_deadline)

        def remaining() -> float:
            return hard_deadline - (time.monotonic() - started)

        def soft_elapsed() -> bool:
            return (time.monotonic() - started) >= soft_deadline

        archives = self._route(query, topic_hint)
        # Spec §6: "Default search order: tier 1 -> tier 2 only if coverage
        # flags are weak after tier 1." Tier 3 (subject/procedural-gated) is
        # already filtered by _route and is consulted alongside tier 1.
        primary_archives = [e for e in archives if e.tier != 2]
        fallback_archives = [e for e in archives if e.tier == 2]

        consulted: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        any_timeout = False
        query_terms = _query_terms(query, topic_hint)
        topic_hint_terms = frozenset(tokenize(topic_hint)) if topic_hint else frozenset()

        # Embed the query at most once per request, bounded by whatever
        # time remains before the soft deadline (never longer than
        # ``_EMBED_QUERY_TIMEOUT_S``). Any failure -- no embedder
        # configured, timeout, or the embedder raising -- degrades to
        # lexical-only retrieval; it never fails the whole request.
        query_vec: Sequence[float] | None = None
        dense_used = False
        dense_notes: list[str] = []
        if self._dense_indexes and self._embed_query is not None:
            soft_remaining = max(0.0, soft_deadline - (time.monotonic() - started))
            budget = min(_EMBED_QUERY_TIMEOUT_S, soft_remaining)
            if budget <= 0:
                dense_notes.append(
                    "dense skipped: soft deadline already elapsed before embed_query"
                )
            else:
                query_vec, note = _call_bounded(
                    lambda: self._embed_query(query), timeout_s=budget
                )
                if note is not None:
                    dense_notes.append(note)
                    query_vec = None

        def _consult(entries: list[ArchiveEntry]) -> None:
            nonlocal any_timeout, dense_used
            for entry in entries:
                if remaining() <= 0:
                    any_timeout = True
                    break
                # Spec §7.4: soft deadline (3s) -- once elapsed, stop
                # consulting further archives and return what is ready as
                # "partial" rather than waiting all the way to the hard
                # deadline.
                if soft_elapsed():
                    any_timeout = True
                    break
                consulted.append({"id": entry.id, "storage": entry.storage, "tier": entry.tier})
                cands, timed_out, dense_note = self._process_archive(
                    entry,
                    query,
                    remaining,
                    keywords=keywords,
                    topic_hint=topic_hint,
                    query_vec=query_vec,
                )
                any_timeout = any_timeout or timed_out
                if dense_note is not None:
                    dense_notes.append(dense_note)
                if any(c.get("sources") and "dense" in c["sources"] for c in cands):
                    dense_used = True
                candidates.extend(cands)

        _consult(primary_archives)

        if fallback_archives and not soft_elapsed() and remaining() > 0:
            preliminary_coverage = _best_coverage(candidates, query_terms, topic_hint_terms)
            if preliminary_coverage["weak"]:
                _consult(fallback_archives)

        candidates.sort(key=lambda c: -c["score"])
        # WP-B8 ranking refinements (reuse plan §7.5-7.6): documented
        # no-ops unless their flag is enabled, applied here -- after the
        # per-archive RRF fusion and the global score sort, before the
        # diversity cap and packing so a promoted passage can still win a
        # per-article slot. With every flag off (the default) each
        # ``apply_*`` call returns its input unchanged, so this is
        # byte-identical to the pre-WP-B8 pipeline.
        query_terms_list = list(query_terms)
        candidates = apply_title_boost(
            candidates, query_terms_list, enabled=self._ranking_flags.title_boost
        )
        candidates = apply_mention_penalty(
            candidates, query_terms_list, enabled=self._ranking_flags.mention_penalty
        )
        candidates = apply_heading_affinity(
            candidates, query_terms_list, enabled=self._ranking_flags.heading_affinity
        )
        candidates = apply_lead_augmentation(
            candidates, enabled=self._ranking_flags.lead_augmentation
        )
        capped = cap_per_article(
            candidates, key=lambda c: c["path"], max_per_article=_DIVERSITY_CAP
        )
        budget = budget_tokens if budget_tokens is not None else self._default_budget_tokens
        packed = pack(capped, budget_tokens=budget, count_tokens=estimate_tokens)

        coverage = _best_coverage(packed, query_terms, topic_hint_terms)

        passages: list[ResearchPassage] = []
        if not coverage["weak"]:
            for entry_dict in packed:
                rp = ResearchPassage(
                    label=entry_dict["label"],
                    passage_id=entry_dict["passage_id"],
                    archive_id=entry_dict["archive_id"],
                    title=entry_dict["title"],
                    path=entry_dict["path"],
                    heading_path=tuple(entry_dict["heading_path"]),
                    text=entry_dict["text"],
                    start=entry_dict["start"],
                    end=entry_dict["end"],
                    score=entry_dict["score"],
                    kind=entry_dict["kind"],
                    estimated_tokens=estimate_tokens(entry_dict["text"]),
                    sources=tuple(entry_dict.get("sources", ("lexical",))),
                )
                passages.append(rp)
                self._snapshot_store.put(
                    rp, fingerprint_digest=self._fingerprints.get(rp.archive_id, "")
                )

        if coverage["weak"]:
            status = "partial" if any_timeout else "empty"
        else:
            status = "partial" if any_timeout else "ok"

        elapsed = time.monotonic() - started
        response = ResearchResponse(
            version=RESPONSE_VERSION,
            status=status,
            passages=passages,
            coverage=coverage,
            archives_consulted=consulted,
            timings={"elapsed_s": elapsed, "cache_hit": False},
            dense_used=dense_used,
            dense_note="; ".join(dense_notes) if dense_notes else None,
        )
        self._response_cache[cache_key] = response
        return response
