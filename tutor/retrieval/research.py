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
import os
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
from tutor.retrieval.hybrid.passages import build_key_fact_passages, split_passages
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


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# Baseline v9 (docs/retrieval_baseline.md), candidate B: only build a real
# snippet for the top-N hits (by Xapian rank) of each search_fulltext call;
# remaining hits score with an empty snippet. This CHANGES ranking, since
# ``_score_articles``/``hit_meta`` reads ``.snippet`` of every hit -- OFF
# (``None``) unless this env var is set, which keeps today's behavior
# byte-identical by default.
_SNIPPET_TOP_N = _env_int("TUTOR_RETRIEVAL_SNIPPET_TOP_N")


def _fulltext_kwargs(query: str, limit: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"query": query, "limit": limit}
    if _SNIPPET_TOP_N is not None:
        kwargs["snippet_top_n"] = _SNIPPET_TOP_N
    return kwargs


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


# Baseline v5 elliptical fix: the plain term-count threshold above (<=1)
# missed real elliptical phrasings that still carry several of their own
# content words -- "What made it explode like that?" (own terms {"made",
# "explode", "like"}, count 3) and "Who wrote that play about the two
# lovers who die?" (count 5). Raising the bare count threshold to cover
# these would also wrongly fold the topic_hint into an ordinary,
# unrelated question of similar length (e.g. "What is the capital of
# France?", also 3 own terms -- see
# ``test_coverage_terms_off_topic_query_not_rescued_by_topic_hint``,
# which must stay green). The actual signal distinguishing a real
# follow-up from an ordinary question is anaphora: an elliptical query
# refers back to something ("it", "that", "this", "again", ...) rather
# than naming its own subject outright. None of "capital of France" /
# "flibbertigibbetopolis effect" contain such a word; every tuning/
# held-out elliptical item does.
_ANAPHORA_RE = re.compile(r"\b(it|its|this|that|those|them|again)\b", re.IGNORECASE)


def _has_anaphora(query: str) -> bool:
    return bool(_ANAPHORA_RE.search(query))


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
    if topic_hint and (
        _elliptical_term_count(query) <= _ELLIPTICAL_TERM_COUNT or _has_anaphora(query)
    ):
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


_QUANTITY_RE = re.compile(
    r"\bhow (many|much|long|far|fast|old|tall|hot|cold|high|deep)\b"
    r"|\bdegrees?\b|\bpercent\b|%|\bnumber of\b",
    re.IGNORECASE,
)

# Baseline v5 item 3 (relevance-cutoff packing) defaults, tuned on the
# tuning split only -- see docs/retrieval_baseline.md "Baseline v5" for the
# before/after packets-per-response and recall@5-on-packed numbers.
_PACKING_RELEVANCE_FRACTION_DEFAULT = 0.25
_PACKING_MAX_PASSAGES_DEFAULT = 6


def _asks_for_quantity(query: str) -> bool:
    """Whether ``query`` reads as asking for a quantity/unit (spec-approved
    infobox exemption for the relevance cutoff below)."""
    return bool(_QUANTITY_RE.search(query))


def _apply_relevance_cutoff(
    passages: list[dict[str, Any]],
    own_terms: frozenset[str],
    top2_paths: frozenset[str],
    *,
    fraction: float,
    max_passages: int,
    quantity_query: bool,
) -> list[dict[str, Any]]:
    """Stop packing when relevance falls off (Baseline v5 item 3): the
    2,000-token evidence budget (``pack``'s ``budget_tokens``) is a CAP, not
    a target. ``passages`` must already be sorted best-first (as
    ``cap_per_article``'s output is). A passage is kept only if (a) its
    score is at least ``fraction`` of the top passage's score, AND (b) it
    covers at least one of the question's own content terms -- EXCEPT an
    infobox passage of one of the top-2 scored articles, which is exempt
    from (b) when the question asks for a quantity/unit (an infobox's
    label/value rows rarely echo the question's own wording verbatim, e.g.
    "Boiling point: -269 C" for "what is helium's boiling point in
    celsius?"). At least one passage is always kept when ``passages`` is
    non-empty (an "ok" response must return SOME evidence), and packing is
    hard-capped at ``max_passages`` regardless of how many would otherwise
    clear the bar.
    """
    if not passages:
        return []
    top_score = passages[0]["score"]
    norm_own = {singularize(t) for t in own_terms}
    kept: list[dict[str, Any]] = []
    for p in passages:
        if len(kept) >= max_passages:
            break
        text_terms = {singularize(t) for t in tokenize(p["text"])}
        covers_own_term = bool(norm_own & text_terms)
        is_exempt_infobox = (
            quantity_query
            and p["path"] in top2_paths
            and "Infobox" in tuple(p.get("heading_path", ()))
        )
        score_ok = p["score"] >= fraction * top_score if top_score > 0 else True
        if score_ok and (covers_own_term or is_exempt_infobox):
            kept.append(p)
    if not kept:
        kept = [passages[0]]
    return kept


_OF_ENTITY_RE = re.compile(r"\bof\s+([a-z]+)", re.IGNORECASE)
_POSSESSIVE_ENTITY_RE = re.compile(r"\b([a-z]+)'s\b", re.IGNORECASE)
_UNIT_PREP_RE = re.compile(r"\bin\s+([a-z]+)\b", re.IGNORECASE)

_ENTITY_ROLE_WEIGHT = 2.5
_UNIT_ROLE_WEIGHT = 0.3


def _term_role_weights(query: str) -> dict[str, float]:
    """Cheap syntactic role heuristic (Baseline v5): English marks "the
    boiling point OF X" as naming X the entity being asked about, and "in
    Y" (a bare noun following "in") as naming Y a unit/modifier, not the
    entity -- e.g. "the boiling point of helium in celsius" is a question
    about helium, expressed in celsius, not the reverse. A possessive
    ("helium's boiling point") is the same "X is the entity" marker in a
    different word order. This does not replace corpus-IDF (a term can be
    both rare AND syntactically marked as the entity, which is the common
    case), it re-weights it: an entity-marked term's contribution to the
    scorer is boosted, a unit-marked term's is discounted, everything else
    stays neutral (weight 1.0).
    """
    weights: dict[str, float] = {}
    for m in _OF_ENTITY_RE.finditer(query):
        weights[singularize(m.group(1).lower())] = _ENTITY_ROLE_WEIGHT
    for m in _POSSESSIVE_ENTITY_RE.finditer(query):
        weights[singularize(m.group(1).lower())] = _ENTITY_ROLE_WEIGHT
    for m in _UNIT_PREP_RE.finditer(query):
        term = singularize(m.group(1).lower())
        if term not in weights:  # "of"/possessive entity marking always wins
            weights[term] = _UNIT_ROLE_WEIGHT
    return weights


def _idf_weight(term: str, matches: dict[str, int]) -> float:
    """Cheap, monotonic IDF-shaped weight from a corpus-wide hit count:
    rarer terms (fewer ``matches``) score higher. Terms this request never
    looked up a real count for (outside the small IDF-lookup budget) get a
    fixed "moderately common" default rather than 0 -- an unknown-rarity
    term still contributes to coordination, just without a rarity bonus.
    """
    count = matches.get(term)
    if count is None:
        # Unknown rarity (outside the small per-request IDF-lookup budget,
        # see ``_MAX_ENTITY_TERM_SEARCHES``): treat as moderately common
        # rather than as a strong (and unearned) rarity signal -- this
        # must stay well below any real looked-up weight for a genuinely
        # rare term, or an un-looked-up common term (e.g. a stray token
        # that crowded a rare one out of the lookup budget) can wrongly
        # outscore it.
        return 0.0005
    if count <= 0:
        return 0.0
    return 1.0 / count


def _score_articles(
    hit_meta: dict[str, tuple[str, str]],
    own_terms: set[str],
    idf_matches: dict[str, int],
    role_weights: dict[str, float] | None = None,
) -> list[str]:
    """Score every candidate article in ``hit_meta`` (path -> (title,
    lead-text-or-snippet)) by IDF-weighted coordination of the question's
    OWN terms over (title, lead) plus a title bonus, and return paths best
    first.

    This replaces Baseline v4's "guarantee a front slot for the rarest-
    term title hit" mechanic (docs/retrieval_baseline.md "Baseline v4"),
    which fixed the Helium regression by force-inserting one hit ahead of
    the RRF-fused order -- correct for that one case, but wrong in
    general: it can only ever promote, never actually rank candidates
    against each other, so it also demoted rank-1 accuracy elsewhere
    (Baseline v5 recall@1 regression). Scoring instead lets every article
    compete on the same signal: a rare term fully covering a short title
    (e.g. "Helium") outscores a common term merely appearing in a longer
    title (e.g. "Boiling point"), and a half-covered longer title (e.g.
    "Celsius Holdings") scores lower still -- see
    ``docs/retrieval_baseline.md`` "Baseline v5" for worked examples.

    ``role_weights`` (see :func:`_term_role_weights`) re-weights specific
    own terms by their syntactic role in the question ("of X" / "X's"
    marks X as the entity; "in Y" marks Y as a unit/modifier) -- without
    it, pure corpus-IDF coordination alone cannot separate "the boiling
    point of helium in celsius" (about helium) from a corpus where
    "celsius" happens to have fewer raw hits than "helium" (Baseline v5
    case A).
    """
    role_weights = role_weights or {}
    norm_own = {singularize(t) for t in own_terms}
    # ``idf_matches`` is keyed by the raw (un-singularized) term the caller
    # looked up (e.g. "celsius"); title/lead matching below works in
    # singularized space (so "moon"/"moons" match), so the IDF lookup must
    # too, or a term whose singular form differs from its raw form (e.g.
    # "celsius" -> "celsiu") silently misses its real weight and falls back
    # to the generic default for every such term.
    idf_matches = {singularize(t): v for t, v in idf_matches.items()}

    def _weight(term: str) -> float:
        return _idf_weight(term, idf_matches) * role_weights.get(term, 1.0)

    scored: list[tuple[str, float]] = []
    for path, (title, lead) in hit_meta.items():
        title_terms = {singularize(t) for t in tokenize(title)}
        lead_terms = {singularize(t) for t in tokenize(lead)} if lead else set()
        matched_title = norm_own & title_terms
        matched_lead = norm_own & (lead_terms - title_terms)
        coordination = sum(_weight(t) for t in matched_title | matched_lead)
        title_coverage_frac = (len(matched_title) / len(title_terms)) if title_terms else 0.0
        title_bonus = title_coverage_frac * sum(_weight(t) for t in matched_title)
        scored.append((path, coordination + title_bonus))
    scored.sort(key=lambda ps: -ps[1])
    return [path for path, _ in scored]


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


def _call_worker_dispatch(
    worker: Any, deadline_s: float, call: Callable[[], WorkerResult]
) -> WorkerResult | None:
    """Run ``call`` (a closure invoking ``worker.request``) off-thread so a
    misbehaving worker can never block the caller past ``deadline_s`` (+
    small slack), regardless of what the worker implementation itself does
    with its own ``deadline_s``. Shared by :func:`_call_worker` (single op)
    and :func:`_call_worker_multi` (batched ``multi`` op, see
    ``docs/worker_batching_design.md``).
    """
    result_box: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_box.put(call())
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
        # then join the thread briefly so the next dispatch on this worker
        # cannot overlap with a still-running previous request.
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


def _memo_key(worker: Any, op: str, kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Key for the per-request (worker, op, args) -> result memo (Fix 2,
    docs/retrieval_latency_profile.md "Ranked candidate fixes" #2).
    Scoped by ``id(worker)`` too, not just ``(op, kwargs)``, so two
    different archives' workers (which can legitimately return different
    results for the same op/kwargs) never share a cache slot.
    """
    return (id(worker), op, tuple(sorted(kwargs.items())))


def _call_worker(
    worker: Any,
    op: str,
    *,
    deadline_s: float,
    memo: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> WorkerResult | None:
    """Call ``worker.request`` off-thread; see :func:`_call_worker_dispatch`.

    When ``memo`` is given, an ``(worker, op, kwargs)`` combination already
    seen (successfully) this request is returned from the memo instead of
    being re-sent to the worker (Fix 2). Only successful (``status ==
    "ok"``) results are memoized; a request/timeout/error is always retried.
    """
    if memo is not None:
        key = _memo_key(worker, op, kwargs)
        cached = memo.get(key)
        if cached is not None:
            return WorkerResult(
                status=cached["status"], value=cached["value"], error=cached["error"],
                elapsed_s=0.0,
            )
    result = _call_worker_dispatch(
        worker, deadline_s, lambda: worker.request(op, deadline_s=deadline_s, **kwargs)
    )
    if memo is not None and result is not None and result.status == "ok":
        key = _memo_key(worker, op, kwargs)
        memo[key] = {"status": result.status, "value": result.value, "error": result.error}
    return result


def _call_worker_multi(
    worker: Any,
    ops: list[tuple[str, dict[str, Any]]],
    *,
    deadline_s: float,
    memo: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> WorkerResult | None:
    """Batch ``ops`` into a single ``multi`` round-trip (see
    ``docs/worker_batching_design.md``); off-thread the same way as
    :func:`_call_worker`. ``result.value`` (when not ``None``) is a list,
    same order/length as ``ops``, of ``{"status", "value", "error"}`` dicts.

    When ``memo`` is given (Fix 2, docs/retrieval_latency_profile.md
    "Ranked candidate fixes" #2), any sub-op whose ``(worker, op, kwargs)``
    already has a successful cached result -- either from an earlier call
    this request, or from an earlier, identical sub-op elsewhere in this
    same ``ops`` list -- is answered from the memo, and only the genuinely
    new sub-ops are sent to the worker (deduplicated by key, so two
    identical uncached sub-ops in one batch are still only sent once).
    Errors are never memoized, so a failed sub-op is retried next time.
    Deadlines are unaffected: this only ever removes round-trips, it never
    changes ``deadline_s`` or skips the deadline check on a real call.
    """
    if not ops:
        return WorkerResult(status="ok", value=[], error=None, elapsed_s=0.0)
    if memo is None:
        return _call_worker_dispatch(
            worker, deadline_s, lambda: worker.request("multi", deadline_s=deadline_s, ops=ops)
        )

    keys = [_memo_key(worker, op, kwargs) for op, kwargs in ops]
    result_slots: list[dict[str, Any] | None] = [memo.get(k) for k in keys]

    send_ops: list[tuple[str, dict[str, Any]]] = []
    send_index_by_key: dict[tuple[Any, ...], int] = {}
    for i, key in enumerate(keys):
        if result_slots[i] is None and key not in send_index_by_key:
            send_index_by_key[key] = len(send_ops)
            send_ops.append(ops[i])

    if send_ops:
        fresh = _call_worker_dispatch(
            worker, deadline_s, lambda: worker.request("multi", deadline_s=deadline_s, ops=send_ops)
        )
        if fresh is None or fresh.status not in ("ok", "partial"):
            err_msg = "worker call timed out" if fresh is None else (fresh.error or "multi failed")
            fresh_values: list[dict[str, Any]] = [
                {"status": "error", "value": None, "error": err_msg} for _ in send_ops
            ]
        else:
            fresh_values = fresh.value
        for key, send_idx in send_index_by_key.items():
            sub = fresh_values[send_idx]
            if sub["status"] == "ok":
                memo[key] = sub
        for i, key in enumerate(keys):
            if result_slots[i] is None:
                result_slots[i] = fresh_values[send_index_by_key[key]]

    overall_status = "ok" if all(r["status"] == "ok" for r in result_slots) else "partial"
    return WorkerResult(status=overall_status, value=result_slots, error=None, elapsed_s=0.0)


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
        packing_relevance_fraction: float = _PACKING_RELEVANCE_FRACTION_DEFAULT,
        packing_max_passages: int = _PACKING_MAX_PASSAGES_DEFAULT,
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
        self._packing_relevance_fraction = packing_relevance_fraction
        self._packing_max_passages = packing_max_passages
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
        memo: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], bool, str | None, list[dict[str, Any]]]:
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
            extra = (
                {"snippet_top_n": _SNIPPET_TOP_N}
                if op == "search_fulltext" and _SNIPPET_TOP_N is not None
                else {}
            )
            res = _call_worker(
                worker,
                op,
                deadline_s=_op_deadline(),
                memo=memo,
                query=joined_query,
                limit=limit,
                **extra,
            )
            if res is None or res.status != "ok":
                return [], True
            return (res.value or []), False

        def _search_with_fallback(
            op: str, limit: int, initial_hits: list[Any], initial_failed: bool
        ) -> tuple[list[Any], bool, bool]:
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
            hits, failed = initial_hits, initial_failed
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

        # Call site 1 (docs/worker_batching_design.md): the initial
        # fulltext+titles searches over the same joined ``search_query``
        # are independent of each other, so they go out as one ``multi``
        # round-trip instead of two. Only this first attempt is batched --
        # the conditional per-token/fallback searches below still depend on
        # these results, so they stay separate calls.
        initial_ops = [
            ("search_fulltext", _fulltext_kwargs(search_query, _FULLTEXT_LIMIT)),
            ("search_titles", {"query": search_query, "limit": _TITLE_LIMIT}),
        ]
        initial_res = _call_worker_multi(worker, initial_ops, deadline_s=_op_deadline(), memo=memo)

        def _unpack_initial(index: int) -> tuple[list[Any], bool]:
            if initial_res is None or initial_res.status not in ("ok", "partial"):
                return [], True
            sub_result = initial_res.value[index]
            if sub_result["status"] != "ok":
                return [], True
            return (sub_result["value"] or []), False

        fulltext_initial_hits, fulltext_initial_failed = _unpack_initial(0)
        title_initial_hits, title_initial_failed = _unpack_initial(1)

        fulltext_hits, fulltext_failed, fulltext_fallback = _search_with_fallback(
            "search_fulltext", _FULLTEXT_LIMIT, fulltext_initial_hits, fulltext_initial_failed
        )
        title_hits, title_failed, title_fallback = _search_with_fallback(
            "search_titles", _TITLE_LIMIT, title_initial_hits, title_initial_failed
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
        # Baseline v5 (docs/retrieval_baseline.md "Baseline v5"): these
        # entity-term lookups now feed the ARTICLE SCORER (``_score_articles``)
        # below instead of force-inserting a "guaranteed slot" ahead of the
        # RRF-fused order -- see that function's docstring for why
        # slot-forcing regressed recall@1.
        term_matches: dict[str, int] = {}
        rarity_order: list[str] = []
        role_weights = _term_role_weights(query)
        if tokens and remaining() > 0:
            # Single-character tokens are almost always tokenizer artifacts
            # (e.g. "helium's" -> "helium", "s") rather than real content
            # words; letting one occupy a scarce IDF-lookup slot can crowd
            # out a genuinely rare term the scorer needs.
            unique_tokens = [t for t in dict.fromkeys(tokens) if len(t) > 1]
            # Call site 2 (docs/worker_batching_design.md): up to
            # ``_MAX_ENTITY_TERM_SEARCHES`` independent ``estimated_matches``
            # lookups, cache-checked first (same as the unbatched
            # ``_term_matches`` path), batched into one ``multi`` round-trip
            # for whichever terms are not already cached.
            candidate_terms = (
                unique_tokens[:_MAX_ENTITY_TERM_SEARCHES] if remaining() > 0 else []
            )
            uncached_terms: list[str] = []
            for term in candidate_terms:
                cache_key = (self._fingerprint_digest(entry), term)
                cached = self._idf_cache.get(cache_key)
                if cached is not None:
                    self._idf_cache.move_to_end(cache_key)
                    term_matches[term] = cached
                else:
                    uncached_terms.append(term)
            if uncached_terms and remaining() > 0:
                matches_res = _call_worker_multi(
                    worker,
                    [("estimated_matches", {"term": t}) for t in uncached_terms],
                    deadline_s=_op_deadline(),
                    memo=memo,
                )
                for i, term in enumerate(uncached_terms):
                    sub_result = (
                        matches_res.value[i]
                        if matches_res is not None
                        and matches_res.status in ("ok", "partial")
                        else None
                    )
                    if (
                        sub_result is not None
                        and sub_result["status"] == "ok"
                        and sub_result["value"] is not None
                    ):
                        matches = int(sub_result["value"])
                        term_matches[term] = matches
                        cache_key = (self._fingerprint_digest(entry), term)
                        self._idf_cache[cache_key] = matches
                        self._idf_cache.move_to_end(cache_key)
                        while len(self._idf_cache) > _IDF_CACHE_MAXSIZE:
                            self._idf_cache.popitem(last=False)
                    else:
                        # Degrade to "no rarity signal" (same as
                        # ``_term_matches`` on a single-op failure); never
                        # cached, so a transient failure can't poison the
                        # cache for the process lifetime.
                        term_matches[term] = 0
            else:
                for term in uncached_terms:
                    term_matches[term] = 0
            # Rarest (fewest corpus-wide matches) first; a term with zero
            # matches anywhere in the archive (a misspelling) carries no
            # rarity signal and is dropped, same rationale as
            # ``rank_terms_by_rarity``.
            rarity_order = sorted(
                (t for t in term_matches if term_matches[t] > 0),
                key=lambda t: (term_matches[t], unique_tokens.index(t)),
            )
            entity_title_hits: list[Any] = []
            entity_snippet_hits: list[Any] = []
            # Call site 3 (docs/worker_batching_design.md): title+snippet
            # for each of the top rarest terms are independent, so all of
            # them (up to ``_ENTITY_TITLE_LIMIT`` terms x 2 ops) go out as
            # one ``multi`` round-trip. `search_titles` never returns a
            # snippet (real worker, confirmed against the live archive --
            # only the FakeWorker test fixture happened to synthesize one),
            # so the scorer below would see title-only coordination for
            # these candidates and fall back to raw corpus-IDF magnitude
            # alone -- exactly the Baseline v5 case A bug ("Celsius" has
            # fewer real corpus hits than "Helium", so it would win on IDF
            # alone with no lead-text tiebreaker). One extra single-term
            # full-text search per rarest term gets a real snippet
            # (lead-text proxy) for the same top article.
            entity_terms = rarity_order[:_ENTITY_TITLE_LIMIT]
            if entity_terms:
                entity_ops: list[tuple[str, dict[str, Any]]] = []
                for term in entity_terms:
                    entity_ops.append(
                        ("search_titles", {"query": term, "limit": _ENTITY_TITLE_RESULTS})
                    )
                    entity_ops.append(("search_fulltext", {"query": term, "limit": 1}))
                entity_res = _call_worker_multi(
                    worker, entity_ops, deadline_s=_op_deadline(), memo=memo
                )
                for i in range(len(entity_terms)):
                    if entity_res is None or entity_res.status not in ("ok", "partial"):
                        timed_out = True
                        continue
                    title_sub = entity_res.value[2 * i]
                    snip_sub = entity_res.value[2 * i + 1]
                    if title_sub["status"] == "ok":
                        entity_title_hits.extend(title_sub["value"] or [])
                    else:
                        timed_out = True
                    if snip_sub["status"] == "ok":
                        entity_snippet_hits.extend(snip_sub["value"] or [])
                    else:
                        timed_out = True
            # Rarity picks WHICH terms to search directly (a good discovery
            # signal: a rare term is likely to be the entity itself), but
            # is not itself the right signal to rank the resulting entity
            # hits against each other -- e.g. a unit word like "celsius"
            # can have fewer corpus-wide hits than the actual answer entity
            # "helium" while still being the wrong article (see Baseline
            # v5's case A). Reorder these specific candidates by the same
            # article scorer used for final fusion so the entity list itself
            # already reflects title+lead coordination, not raw rarity.
            if entity_title_hits:
                own_terms_for_entities = frozenset(
                    tokenize(strip_instruction_words(query))
                )
                snippet_by_path = {h.path: h.snippet for h in entity_snippet_hits if h.snippet}
                entity_hit_meta = {
                    h.path: (
                        h.title,
                        snippet_by_path.get(h.path) or getattr(h, "snippet", "") or "",
                    )
                    for h in entity_title_hits
                }
                entity_score_order = _score_articles(
                    entity_hit_meta, own_terms_for_entities, term_matches, role_weights
                )
                rank_of = {p: i for i, p in enumerate(entity_score_order)}
                entity_title_hits = sorted(
                    entity_title_hits, key=lambda h: rank_of.get(h.path, len(rank_of))
                )
            # Prefer the snippet-bearing full-text hit for a path present in
            # both lists (it's the same article; the full-text hit is the
            # one with real lead text) while keeping the score-based order
            # just established above.
            order_index = {h.path: i for i, h in enumerate(entity_title_hits)}
            entity_title_hits = _dedupe_by_path([*entity_snippet_hits, *entity_title_hits])
            entity_title_hits.sort(key=lambda h: order_index.get(h.path, len(order_index)))
            # Latency (Baseline v5, item 4): the relaxed-AND query over the
            # rarest terms is only useful when the all-terms query's own
            # top-3 hits are missing the rarest term's title-matched
            # article outright -- if it is already there, the extra
            # worker round-trip(s) below cannot change the outcome (the
            # scorer will rank it on its own merits) and are skipped.
            top3_fulltext_paths = {h.path for h in fulltext_hits[:3]}
            already_present = any(
                h.path in top3_fulltext_paths
                for term in rarity_order[:_ENTITY_TITLE_LIMIT]
                for h in entity_title_hits
                if h.title  # any hit found by this term search
            )
            # ``entity_snippet_hits`` are themselves real full-text hits
            # (the single-term ``search_fulltext`` call above), so they
            # belong in the full-text ranking too -- not only used as a
            # snippet source for scoring the title hits -- or the entity's
            # article can end up voted for by only 2 of the 3 fused
            # rankings (title, scored) while several generic co-occurring
            # articles are voted for by all 3, letting count-of-lists beat
            # the scorer's actual judgment (Baseline v5 case A).
            entity_fulltext_hits: list[Any] = [*entity_snippet_hits]
            if not already_present:
                for k in _ENTITY_RELAXED_AND_SIZES:
                    if len(rarity_order) >= k:
                        joined = " ".join(rarity_order[:k])
                        hits, failed = _search("search_fulltext", joined, _FULLTEXT_LIMIT)
                        timed_out = timed_out or failed
                        entity_fulltext_hits.extend(hits)
            # Entity hits are prepended so they contribute to (and, via the
            # scorer, can win) the RRF fusion below alongside the generic
            # fallback hits -- never force-inserted ahead of it.
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
            return [], timed_out, dense_note, []

        # Baseline v5: a 4th ranking -- ARTICLE SCORING (``_score_articles``)
        # -- is fused alongside the reuse plan's three (dense semantic
        # search, title, body/full-text), all via the same RRF, rather than
        # any one of them force-inserting a slot ahead of the others (see
        # ``_score_articles``'s docstring). The scorer uses each hit's own
        # title + search snippet (a lead-text proxy already returned by the
        # worker, so this costs no extra round-trip) and the question's OWN
        # terms only (not keywords/topic_hint -- those already shape which
        # articles appear as hits at all via ``search_query`` above).
        own_terms = frozenset(tokenize(strip_instruction_words(query)))
        hit_meta: dict[str, tuple[str, str]] = {}
        for h in (*fulltext_hits, *title_hits):
            if h.path not in hit_meta:
                hit_meta[h.path] = (h.title, getattr(h, "snippet", "") or "")
        scored_paths = (
            _score_articles(hit_meta, own_terms, term_matches, role_weights) if hit_meta else []
        )
        # The fulltext/title lists themselves are also re-sorted by this
        # same score before fusion (not just added as a 4th input): both
        # can otherwise carry an internal order from a much cruder
        # heuristic (the per-token fallback's own "matched term count,
        # then LOCAL rarity" merge -- see ``_search_with_fallback``), which
        # can rank several generic articles matching common terms (e.g.
        # every "Boiling ..." title matching both "boiling" and "point")
        # ahead of the one true entity article the scorer identifies via
        # corpus-IDF + syntactic role + lead-text coordination. Re-sorting
        # in place (rather than only adding one more equal-weight ranking)
        # means the fused order reflects the scorer's judgment throughout,
        # while still fusing -- via ``rrf_fuse`` below -- with the
        # independent title/dense rankings rather than force-inserting
        # anything ahead of them.
        if hit_meta:
            score_rank = {p: i for i, p in enumerate(scored_paths)}
            fulltext_hits = sorted(
                fulltext_hits, key=lambda h: score_rank.get(h.path, len(score_rank))
            )
            title_hits = sorted(title_hits, key=lambda h: score_rank.get(h.path, len(score_rank)))

        fused_articles = rrf_fuse(
            [
                [h.path for h in fulltext_hits],
                [h.path for h in title_hits],
                dense_paths,
                scored_paths,
            ]
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

        # Baseline v6 ("infobox key facts"): the top-2 SCORED articles for
        # this archive (``top_paths``, already fused/scored above) get their
        # bundle kept around so key-fact passages can be built from it below
        # -- these bypass BM25/diversity/relevance-cutoff entirely (see
        # ``research()``), so they must never be built for a low-ranked
        # article that merely happened to be fetched.
        key_fact_top_paths = top_paths[:2]
        key_fact_bundles: dict[str, Any] = {}

        # Call site 4 (docs/worker_batching_design.md): fetching every top
        # article is independent, so all of ``top_paths`` (up to
        # ``_top_n_articles + 1``) go out as one ``multi`` round-trip
        # instead of one ``fetch_entry`` per path.
        candidate_passages: list[Any] = []
        if top_paths and remaining() > 0:
            fetch_deadline = max(_MIN_OP_DEADLINE_S, min(_PER_OP_DEADLINE_CAP_S, remaining()))
            fetch_res = _call_worker_multi(
                worker,
                [("fetch_entry", {"path": path}) for path in top_paths],
                deadline_s=fetch_deadline,
                memo=memo,
            )
            if fetch_res is None or fetch_res.status not in ("ok", "partial"):
                timed_out = True
            else:
                if fetch_res.status == "partial":
                    timed_out = True
                for path, sub_result in zip(top_paths, fetch_res.value, strict=True):
                    if sub_result["status"] != "ok":
                        timed_out = True
                        continue
                    fetched = sub_result["value"]
                    bundle = build_bundle(fetched.html, path=fetched.path, title=fetched.title)
                    if path in key_fact_top_paths:
                        key_fact_bundles[path] = bundle
                    candidate_passages.extend(
                        split_passages(
                            bundle, fingerprint_digest=fingerprint_digest, archive_id=entry.id
                        )
                    )
        elif top_paths:
            timed_out = True

        own_terms_for_key_facts = frozenset(tokenize(strip_instruction_words(query)))
        key_fact_results: list[dict[str, Any]] = []
        for path in key_fact_top_paths:
            b = key_fact_bundles.get(path)
            if b is None:
                continue
            for kf in build_key_fact_passages(
                b,
                own_terms_for_key_facts,
                fingerprint_digest=fingerprint_digest,
                archive_id=entry.id,
            ):
                key_fact_results.append(
                    {
                        "passage_id": kf.passage_id,
                        "archive_id": kf.archive_id,
                        "title": kf.title,
                        "path": kf.path,
                        "heading_path": kf.heading_path,
                        "text": kf.text,
                        "start": kf.start,
                        "end": kf.end,
                        "score": float("inf"),
                        "kind": entry.kind,
                        "used_fallback": used_fallback,
                        "sources": ("lexical",),
                    }
                )

        if not candidate_passages:
            return [], timed_out, dense_note, key_fact_results

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
        return results, timed_out, dense_note, key_fact_results

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
        key_fact_candidates: list[dict[str, Any]] = []
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

        # Fix 2 (docs/retrieval_latency_profile.md "Ranked candidate fixes"
        # #2): one memo dict for the whole request, shared across every
        # archive consulted (keyed by worker identity too -- see
        # ``_memo_key`` -- so different archives' workers never collide),
        # so a duplicate (op, kwargs) issued anywhere in this request hits
        # the worker at most once.
        op_memo: dict[tuple[Any, ...], dict[str, Any]] = {}

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
                cands, timed_out, dense_note, key_facts = self._process_archive(
                    entry,
                    query,
                    remaining,
                    keywords=keywords,
                    topic_hint=topic_hint,
                    query_vec=query_vec,
                    memo=op_memo,
                )
                any_timeout = any_timeout or timed_out
                if dense_note is not None:
                    dense_notes.append(dense_note)
                if any(c.get("sources") and "dense" in c["sources"] for c in cands):
                    dense_used = True
                candidates.extend(cands)
                key_fact_candidates.extend(key_facts)

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
        # Baseline v5 item 3 (relevance-cutoff packing, project-owner
        # approved -- docs/retrieval_baseline.md "Baseline v5"): the token
        # budget is a CAP, not a target. Stop packing when relevance falls
        # off rather than filling the whole budget with weak passages.
        top2_paths: list[str] = []
        for c in candidates:
            if c["path"] not in top2_paths:
                top2_paths.append(c["path"])
            if len(top2_paths) >= 2:
                break
        own_terms_for_packing = frozenset(tokenize(strip_instruction_words(query)))
        cutoff_passages = _apply_relevance_cutoff(
            capped,
            own_terms_for_packing,
            frozenset(top2_paths),
            fraction=self._packing_relevance_fraction,
            max_passages=self._packing_max_passages,
            quantity_query=_asks_for_quantity(query),
        )
        # Baseline v6 ("infobox key facts"): key-fact passages (already
        # restricted to the top-2 scored articles per archive when they
        # were built -- see ``_process_archive``) are packed FIRST, ahead
        # of the normally-ranked passages, and are exempt from the
        # diversity cap and relevance-fraction cutoff above -- but they
        # still count against the token budget, and the max-passages cap
        # is only raised by however many of them there are (max +2).
        seen_key_fact_ids: set[str] = set()
        unique_key_facts: list[dict[str, Any]] = []
        for kf in key_fact_candidates:
            if kf["passage_id"] not in seen_key_fact_ids:
                seen_key_fact_ids.add(kf["passage_id"])
                unique_key_facts.append(kf)
        unique_key_facts = unique_key_facts[:2]
        combined_passages = [*unique_key_facts, *cutoff_passages]
        budget = budget_tokens if budget_tokens is not None else self._default_budget_tokens
        packed = pack(combined_passages, budget_tokens=budget, count_tokens=estimate_tokens)

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
