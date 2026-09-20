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
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tutor.retrieval.hybrid.dense import DenseIndex
from tutor.retrieval.hybrid.diversity import cap_per_article
from tutor.retrieval.hybrid.lexical import (
    BM25,
    rank_terms_by_rarity,
    singularize,
    strip_instruction_words,
    tokenize,
)
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
_MAX_FALLBACK_SEARCHES = 6

# Baseline v4 (see docs/retrieval_baseline.md "Baseline v4"): even when the
# joined all-terms query returns *some* hits, they can all be wrong (every
# word happens to co-occur in some unrelated article) while the one article
# that is actually the entity in question ("Helium") never appears because
# it lacks one word (a misspelling, or a unit symbol like "°C" instead
# of the word "celsius"). Real IDF (libzim's own `getEstimatedMatches`,
# corpus-wide -- not just this request's local hit counts) picks out the
# rarest/most specific query terms, which are always run as their own title
# search (finds "Helium" directly) and as a small top-k-rarest-terms
# full-text query (a relaxed AND that can still succeed even though the
# *full* AND query would need one more, absent, term). Bounded the same way
# as the existing per-token fallback so a long question can never blow the
# soft deadline.
_MAX_ENTITY_TERM_SEARCHES = 4
_ENTITY_TITLE_LIMIT = 3
_ENTITY_TITLE_RESULTS = 3
_ENTITY_RELAXED_AND_SIZES = (2, 3)
_ENTITY_GUARANTEED_SLOTS = 2
_IDF_CACHE_MAXSIZE = 4096

# Reuse plan §7.2 "Dense article top 16": the dense sidecar's own
# candidate list, fused at article level with the lexical/title rankings
# below (not blended with raw cosine/BM25 scores directly).
_DENSE_ARTICLE_TOP = 16
_EMBED_QUERY_TIMEOUT_S = 1.0

# Bound on ResearchEngine._response_cache: an unbounded dict keyed by raw
# query text grows for the life of a long-running process (review pass 2,
# finding 4). An `OrderedDict` capped at this size and evicted LRU-style
# keeps memory bounded without needing a TTL.
_RESPONSE_CACHE_MAXSIZE = 256

_PROCEDURAL_RE = re.compile(r"\bhow (do|would|can|should) (i|you|we)\b", re.IGNORECASE)

# Coverage-flag threshold (spec §7.3 "coverage signals are explainable").
# Tuned on the tuning split of eval/questions/simplewiki_questions.jsonl --
# see docs/retrieval_baseline.md "Baseline v2" for the before/after tables.
_COVERAGE_TERM_THRESHOLD = 0.6


def _is_procedural(query: str) -> bool:
    """Whether ``query`` reads as a "how do I ..."-style procedural ask."""
    return bool(_PROCEDURAL_RE.search(query))


def _query_terms(query: str, topic_hint: str | None) -> set[str]:
    """Content terms used for *candidate generation and ranking* (BM25 query
    expansion, title boost, etc.): the query plus, per spec §5/§7.1, the
    host-supplied ``topic_hint`` (current subject). NOT used for the
    abstention/coverage gate -- see :func:`_coverage_terms`, since unioning
    the topic_hint in there let a subject hint alone satisfy coverage for an
    off-topic question (review pass 2, finding 2).
    """
    terms = set(tokenize(strip_instruction_words(query)))
    if topic_hint:
        terms |= set(tokenize(topic_hint))
    return terms


# A query is treated as "elliptical" (too little content of its own to
# judge coverage on) when it has at most this many content-bearing tokens
# of its own, once both ``tokenize``'s stopwords/pronouns AND a small set
# of generic conversational-continuation fillers (below) are stripped --
# e.g. "what about its moons?" -> {"moons"}, "can you tell me more about
# it?" -> {} (both "tell" and "more" are fillers). Only then does the
# host-supplied topic_hint get folded into the *coverage* term set. An
# ordinary question about a different subject, like "what's the capital of
# France?" -> {"s", "capital", "france"} or "what is the
# flibbertigibbetopolis effect?" -> {"flibbertigibbetopolis", "effect"},
# keeps real content terms of its own once fillers are stripped and must
# never be rescued by a topic_hint that merely happens to be the current
# subject's own title (review pass 2, finding 2).
_ELLIPTICAL_TERM_COUNT = 1

# Generic conversational-continuation fillers: words that signal "keep
# going on the current subject" rather than naming a subject of their own.
# ``tokenize``'s stopword list already strips grammatical words (the, is,
# about, ...); this small extra set catches the handful of common verbs a
# "tell me more" / "explain further" style follow-up uses that aren't
# stopwords but also aren't content words for coverage purposes.
_CONTINUATION_FILLERS = frozenset({"tell", "more", "explain", "elaborate", "continue"})


def _elliptical_term_count(query: str) -> int:
    return len(frozenset(tokenize(strip_instruction_words(query))) - _CONTINUATION_FILLERS)


def _own_term_count(query: str) -> int:
    """Number of the question's OWN content terms (instruction words
    stripped, ``topic_hint`` excluded) -- used only to gate the coverage
    floor (see ``_OWN_TERM_FLOOR_MIN_COUNT``), never to decide whether the
    topic_hint may be folded in (that is ``_elliptical_term_count``)."""
    return len(frozenset(tokenize(strip_instruction_words(query))))


def _coverage_terms(query: str, topic_hint: str | None = None) -> frozenset[str]:
    """Content terms used for the abstention/coverage gate: the query's own
    terms only, except for the elliptical case (see
    ``_ELLIPTICAL_TERM_COUNT``) where the topic_hint's terms are folded in
    too so a follow-up like "what about its moons?" or "can you tell me
    more about it?" can still be judged covered against the current
    subject's article. A query with more content terms of its own is
    judged strictly on those -- a topic_hint can never single-handedly
    manufacture coverage for it (that was the bug: a subject hint alone
    making an off-topic candidate look "covered")."""
    own = frozenset(tokenize(strip_instruction_words(query)))
    if topic_hint and _elliptical_term_count(query) <= _ELLIPTICAL_TERM_COUNT:
        return own | frozenset(tokenize(topic_hint))
    return own


# Coverage is checked over the top few candidates, not just the single
# best-scored one: RRF/BM25 rank-1 is not always the semantically correct
# passage (see docs/retrieval_baseline.md failure analysis), so requiring
# *only* rank-1 to look relevant would wipe out perfectly good evidence
# sitting at rank 2-3. Tuned on the tuning split alongside the term
# threshold.
_COVERAGE_CANDIDATES_CHECKED = 3

# The documented "generic single-word title" hole (docs/retrieval_baseline.md):
# a bare ``title_match`` used to be enough to call a candidate covered even
# when almost none of the question's own words appeared anywhere in it --
# e.g. "Output the boiling point of helium..." title-matching "Output"
# (Input/output, Logic gate, ...) while "helium" never showed up. Once the
# question has at least this many of its OWN content terms (topic_hint
# excluded -- see ``_own_term_count``), a title match on an own term alone
# is only honored if own-term TEXT coverage also clears this floor. Tuned
# on the tuning split of eval/questions/simplewiki_questions.jsonl -- see
# docs/retrieval_baseline.md "Baseline v3".
_OWN_TERM_COVERAGE_FLOOR = 0.34
_OWN_TERM_FLOOR_MIN_COUNT = 3


def _best_coverage(
    candidates: list[dict[str, Any]],
    query_terms: set[str],
    topic_hint_terms: frozenset[str],
    own_term_count: int = 0,
) -> dict[str, Any]:
    """The least-weak :func:`compute_coverage` result among the top-scored
    ``_COVERAGE_CANDIDATES_CHECKED`` candidates (any one of them being a
    strong hit is enough to call the query covered). ``query_terms`` is
    already the right term set for this call -- see ``_coverage_terms``,
    which decides whether the topic_hint's terms belong in it (elliptical
    queries only) *before* this function ever runs, since that decision
    depends on the query text alone, not on any particular candidate.
    ``topic_hint_terms`` is forwarded to :func:`compute_coverage`, which
    may let a hint term satisfy ``title_match`` -- but only gated on this
    same candidate's own-term text coverage also clearing the threshold
    (pass-2b fix: a title match on hint terms alone is never sufficient by
    itself).
    """
    if not candidates:
        return compute_coverage(
            query_terms, None, None, hint_terms=topic_hint_terms, own_term_count=own_term_count
        )
    top = sorted(candidates, key=lambda c: -c["score"])[:_COVERAGE_CANDIDATES_CHECKED]
    best: dict[str, Any] | None = None
    for c in top:
        cov = compute_coverage(
            query_terms,
            c["title"],
            c["text"],
            hint_terms=topic_hint_terms,
            own_term_count=own_term_count,
        )
        if not cov["weak"]:
            return cov
        if best is None or cov["term_coverage"] > best["term_coverage"]:
            best = cov
    assert best is not None
    return best


def compute_coverage(
    query_terms: set[str],
    title: str | None,
    text: str | None,
    hint_terms: frozenset[str] | None = None,
    own_term_count: int = 0,
) -> dict[str, Any]:
    """Explainable coverage flags for a single candidate.

    Per spec §7.3 ("coverage signals are explainable (facets represented,
    entities missing, limits hit)") and §6 ("tier 2 only if coverage flags
    are weak after tier 1"). ``weak`` is True when there is no candidate at
    all, or the candidate shares no ``query_terms`` with the candidate's
    text/title -- ``query_terms`` here is whatever the caller decided is
    the right coverage term set (see ``_coverage_terms``; for the
    elliptical-fold case it already includes the topic_hint's tokens).

    ``hint_terms``, when given, is the *separate* topic_hint token set
    (pass-2b fix, docs/retrieval_baseline.md "Pass-2 fix"): a hint term
    matching the title is only allowed to satisfy ``title_match`` when
    ``query_terms``' own text coverage against this same candidate
    *already* clears ``_COVERAGE_TERM_THRESHOLD`` on its own -- a title
    match on hint terms alone (with the question's own content terms
    otherwise uncovered) is never sufficient by itself. Matching tolerates
    simple plural/singular differences (``singularize``) so "moon" in the
    query matches "moons" in the text.
    """
    if not query_terms or title is None or text is None:
        return {"term_coverage": 0.0, "title_match": False, "weak": True}

    passage_terms = {singularize(t) for t in tokenize(text)}
    title_terms = {singularize(t) for t in tokenize(title)}
    norm_query_terms = {singularize(t) for t in query_terms}
    term_coverage = len(norm_query_terms & passage_terms) / len(norm_query_terms)
    title_match = bool(norm_query_terms & title_terms)
    if (
        title_match
        and own_term_count >= _OWN_TERM_FLOOR_MIN_COUNT
        and term_coverage < _OWN_TERM_COVERAGE_FLOOR
    ):
        # A title match on one own term (often a generic single word like
        # "Output" or "Effect") is not enough when the question has several
        # other own content terms and almost none of them show up anywhere
        # in this candidate's text -- see _OWN_TERM_COVERAGE_FLOOR.
        title_match = False
    if not title_match and hint_terms:
        norm_hint_terms = {singularize(t) for t in hint_terms}
        if term_coverage >= _COVERAGE_TERM_THRESHOLD and (norm_hint_terms & title_terms):
            title_match = True
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
        self._response_cache: OrderedDict[tuple[Any, ...], ResearchResponse] = OrderedDict()
        # Real corpus-IDF signal (libzim `getEstimatedMatches`), cached per
        # (archive fingerprint, term) since the same term is looked up
        # across many requests and archive contents only change when the
        # ZIM itself changes (fingerprint changes too). Bounded LRU so a
        # long-running process's memory does not grow without limit.
        self._idf_cache: OrderedDict[tuple[str, str], int] = OrderedDict()

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

    def _term_matches(
        self, entry: ArchiveEntry, worker: Any, term: str, *, deadline_s: float
    ) -> int:
        """Cached real corpus-IDF signal for ``term`` in ``entry``'s archive:
        libzim's own `getEstimatedMatches`, not a local per-request hit
        count. Any failure (timeout, worker error) degrades to ``0`` --
        "no rarity signal for this term" -- never fatal, and is not
        cached (a transient failure should not poison the cache for the
        life of the process).
        """
        key = (self._fingerprint_digest(entry), term)
        cached = self._idf_cache.get(key)
        if cached is not None:
            self._idf_cache.move_to_end(key)
            return cached
        res = _call_worker(worker, "estimated_matches", deadline_s=deadline_s, term=term)
        if res is None or res.status != "ok" or res.value is None:
            return 0
        matches = int(res.value)
        self._idf_cache[key] = matches
        self._idf_cache.move_to_end(key)
        while len(self._idf_cache) > _IDF_CACHE_MAXSIZE:
            self._idf_cache.popitem(last=False)
        return matches

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
        tokens = tokenize(strip_instruction_words(query))
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

        # Bound on extra per-token worker searches across BOTH the fulltext
        # and title fallback passes combined, per archive per request (item
        # 3): the fallback must never blow the soft deadline just because a
        # long natural-language question has many tokens.
        fallback_searches_used = 0

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
            # language question with a stray non-corpus/misspelt word (e.g.
            # "farenheit") can zero out an otherwise-good query. When the
            # joined query returns nothing, fall back to per-token searches
            # (bounded, see ``_MAX_FALLBACK_SEARCHES``, so extra worker
            # round-trips stay within the soft deadline) and rank the
            # resulting ARTICLES by (a) how many distinct query terms they
            # matched -- more is better -- and (b) the rarity of the rarest
            # of those terms (fewer hits of its own -- see
            # ``rank_terms_by_rarity``), so a specific/rare term like
            # "helium" outweighs a generic one like "output" or "point", and
            # a term with zero hits at all (a misspelling) is dropped
            # outright rather than diluting the merge. The returned
            # ``used_fallback`` flag is carried onto each result (see the
            # "used_fallback" key below) purely for observability --
            # coverage itself is judged by term/title overlap (see
            # ``compute_coverage``), not by which search path found it.
            hits, failed = _search(op, search_query, limit)
            if hits or failed or not tokens:
                return hits, failed, False
            nonlocal fallback_searches_used
            term_hits_by_term: dict[str, list[Any]] = {}
            term_counts: dict[str, int] = {}
            any_failed = False
            for term in tokens:
                if fallback_searches_used >= _MAX_FALLBACK_SEARCHES:
                    break
                fallback_searches_used += 1
                term_hits, term_failed = _search(op, term, limit)
                any_failed = any_failed or term_failed
                term_hits_by_term[term] = term_hits
                term_counts[term] = len(term_hits)
            rarity_order = rank_terms_by_rarity(term_counts)
            rarity_rank = {term: i for i, term in enumerate(rarity_order)}
            matched_terms: dict[str, set[str]] = {}
            first_hit: dict[str, Any] = {}
            first_seen_order: dict[str, int] = {}
            for term in rarity_order:
                for hit in term_hits_by_term[term]:
                    if hit.path not in first_hit:
                        first_hit[hit.path] = hit
                        first_seen_order[hit.path] = len(first_seen_order)
                        matched_terms[hit.path] = set()
                    matched_terms[hit.path].add(term)
            ordered_paths = sorted(
                first_hit,
                key=lambda p: (
                    -len(matched_terms[p]),
                    min(rarity_rank[t] for t in matched_terms[p]),
                    first_seen_order[p],
                ),
            )
            merged = [first_hit[p] for p in ordered_paths]
            return merged[:limit], any_failed, True

        fulltext_hits, fulltext_failed, fulltext_fallback = _search_with_fallback(
            "search_fulltext", _FULLTEXT_LIMIT
        )
        title_hits, title_failed, title_fallback = _search_with_fallback(
            "search_titles", _TITLE_LIMIT
        )
        timed_out = timed_out or fulltext_failed or title_failed

        def _dedupe_by_path(hits: list[Any]) -> list[Any]:
            seen: set[str] = set()
            out: list[Any] = []
            for h in hits:
                if h.path not in seen:
                    seen.add(h.path)
                    out.append(h)
            return out

        # Always (not only when the all-terms query returned zero hits, see
        # Baseline v4 above) add "entity candidates": title-search each of
        # the rarest own query terms directly, and run a relaxed AND of just
        # the top-k rarest terms. This is the fix for the case where the
        # full AND query *does* return hits (so the existing zero-hits-only
        # fallback above never runs) but they are all wrong because the
        # true entity article lacks one word the question used (a
        # misspelling, or a symbol like degC where the question spelled out
        # "celsius"). Bounded to a few extra worker calls total.
        guaranteed_entity_paths: list[str] = []
        if tokens and remaining() > 0:
            unique_tokens = list(dict.fromkeys(tokens))
            term_matches: dict[str, int] = {}
            for i, term in enumerate(unique_tokens):
                if i >= _MAX_ENTITY_TERM_SEARCHES or remaining() <= 0:
                    break
                term_matches[term] = self._term_matches(
                    entry, worker, term, deadline_s=_op_deadline()
                )
            # Rarest (fewest corpus-wide matches) first; a term with zero
            # matches anywhere in the archive (a misspelling) carries no
            # rarity signal and is dropped, same rationale as
            # ``rank_terms_by_rarity``.
            rarity_order = sorted(
                (t for t in term_matches if term_matches[t] > 0),
                key=lambda t: (term_matches[t], unique_tokens.index(t)),
            )
            entity_title_hits: list[Any] = []
            for term in rarity_order[:_ENTITY_TITLE_LIMIT]:
                hits, failed = _search("search_titles", term, _ENTITY_TITLE_RESULTS)
                timed_out = timed_out or failed
                if hits and term in rarity_order[:_ENTITY_GUARANTEED_SLOTS]:
                    guaranteed_entity_paths.append(hits[0].path)
                entity_title_hits.extend(hits)
            entity_fulltext_hits: list[Any] = []
            for k in _ENTITY_RELAXED_AND_SIZES:
                if len(rarity_order) >= k:
                    joined = " ".join(rarity_order[:k])
                    hits, failed = _search("search_fulltext", joined, _FULLTEXT_LIMIT)
                    timed_out = timed_out or failed
                    entity_fulltext_hits.extend(hits)
            # Entity hits are prepended: they are the strongest, most
            # specific signal (rare/entity terms only), so they should win
            # ties in the RRF fusion below over generic fallback hits.
            title_hits = _dedupe_by_path([*entity_title_hits, *title_hits])
            fulltext_hits = _dedupe_by_path([*entity_fulltext_hits, *fulltext_hits])
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

        # Guarantee a slot for each of the (up to _ENTITY_GUARANTEED_SLOTS)
        # rarest own query terms' own title-search hit -- see Baseline v4
        # above. RRF fusion alone under-ranks a candidate that is only in
        # one input list (title) against candidates present in both
        # fulltext and title lists (e.g. every generic "Boiling ..." title
        # that also turns up in the full-text AND-of-all-terms query), even
        # though the title hit is the far more specific/relevant one. Same
        # mechanism as the topic_hint guaranteed slot just below.
        # Only a genuinely MISSING entity candidate is inserted -- one
        # already present in ``top_paths`` keeps its fused rank rather than
        # being force-promoted to the front, so a query that was already
        # answered correctly (the common case) is never perturbed by this
        # mechanism; it only rescues the specific failure mode above.
        for entity_path in reversed(guaranteed_entity_paths):
            if entity_path in top_paths:
                continue
            lexical_paths.add(entity_path)
            top_paths.insert(0, entity_path)
        top_paths = top_paths[: self._top_n_articles + len(guaranteed_entity_paths)]

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
            self._response_cache.move_to_end(cache_key)
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
        coverage_terms = _coverage_terms(query, topic_hint)
        topic_hint_terms = frozenset(tokenize(topic_hint)) if topic_hint else frozenset()
        own_term_count = _own_term_count(query)

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
            preliminary_coverage = _best_coverage(
                candidates, coverage_terms, topic_hint_terms, own_term_count
            )
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

        coverage = _best_coverage(packed, coverage_terms, topic_hint_terms, own_term_count)

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
        self._response_cache.move_to_end(cache_key)
        while len(self._response_cache) > _RESPONSE_CACHE_MAXSIZE:
            self._response_cache.popitem(last=False)
        return response
