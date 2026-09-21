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

from tutor.retrieval.assessment import assess_evidence
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
from tutor.retrieval.registry import ArchiveEntry, Registry, RegistryError
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

# Baseline v11 (docs/retrieval_baseline.md "Baseline v11"): spelling-
# tolerant fallback for a rare/misspelt KEY term. Only engages when the
# normal path is weak -- a content term with <= this many corpus-wide
# estimated matches -- so a correctly spelt rare term (a small but real
# corpus count) is never touched. Bounded to a couple of terms/request so a
# long, mostly-misspelt question can't blow the soft deadline.
_SPELL_ZERO_MATCH_CEILING = 0
_SPELL_MIN_TERM_LEN = 4
_MAX_SPELL_CORRECTIONS = 2
_SPELL_SUGGEST_LIMIT = 20
_SPELL_MAX_EDIT_DISTANCE = 2
# libzim's title-suggestion search matches on a TITLE PREFIX, not on
# arbitrary edit distance -- so it only surfaces "Helium" for a query
# literally spelled "heluim" when the divergence point (here "u"/"i"
# transposed) falls late enough. Real single-edit typos on a real corpus
# (measured, see docs/retrieval_baseline.md "Baseline v11") often diverge
# a few characters before the end, so the query is retried against
# progressively shorter prefixes of the misspelt term (never below
# ``_SPELL_MIN_PREFIX_LEN``) until the true title's shared prefix is
# reached. All variants for all terms go out as one ``multi`` batch.
_SPELL_PREFIX_DROPS = (0, 1, 2, 3, 4)
_SPELL_MIN_PREFIX_LEN = 3
# A correction candidate must clear this many corpus-wide matches itself,
# or it is not meaningfully more real than the misspelling it would
# replace (both would otherwise look like "no signal").
_SPELL_MIN_CORRECTED_MATCHES = 3
# Set to "0" to force the fallback off entirely (e.g. to reproduce pre-v11
# behavior exactly for an A/B comparison).
_SPELL_FALLBACK_ENABLED = os.environ.get("TUTOR_RETRIEVAL_SPELLING_FALLBACK", "1") != "0"

# Baseline v13 (docs/retrieval_baseline.md "Baseline v13"): kids fuse
# ("squarefoot", "solarsystem") or split ("photo synthesis", "earth
# quake") compound words. Both directions are tried ONLY when this
# archive's first pass already looks weak (see the call site: no fulltext/
# title hits at all, or a term has ~zero corpus-wide matches) -- never on
# an otherwise-strong result. A SPLIT candidate is accepted only when BOTH
# halves individually clear this many corpus-wide matches (otherwise a
# split of a genuinely rare-but-real term into two meaningless fragments
# would look "healthy" by accident); among qualifying splits the one whose
# phrase query has the most matches wins. A JOIN candidate (adjacent-word
# pair) is accepted only when the two words together, as an exact phrase,
# have (near) zero matches -- i.e. that literal pairing basically never
# occurs -- while the fused or hyphenated form does.
_COMPOUND_MIN_HALF_LEN = 4
# Measured on the tuning split (docs/retrieval_baseline.md "Baseline v13"):
# a short (3-4 char) junk fragment from splitting a genuine misspelling
# (e.g. "dinasors" -> "dina"/"sors") can still show a deceptively large
# raw estimated_matches count (Xapian's query parser treats a very short
# term almost like a stopword/substring match), so a low bar like the
# spelling fallback's ``_SPELL_MIN_CORRECTED_MATCHES`` (3) let false
# splits through. 200 was the smallest threshold that rejected every
# false split found on the tuning split's misspelling probes while still
# accepting every real compound half (e.g. "square"/8109, "foot"/2515,
# "solar"/"system").
_COMPOUND_MIN_HALF_MATCHES = 200
_COMPOUND_JOIN_PHRASE_CEILING = 0
_MAX_COMPOUND_TERMS = 2
# A fused compound word ("solarsystem") is a real, if unusual, index
# token and can carry a small number of incidental matches (measured on
# the tuning split: "solarsystem" -> 3) even though it is not the genuine
# term -- looser than the spelling fallback's strict-zero
# ``_SPELL_ZERO_MATCH_CEILING`` so this case is still tried for a split.
_COMPOUND_NEAR_ZERO_CEILING = 5

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


def _damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Restricted Damerau-Levenshtein edit distance (insert/delete/
    substitute/adjacent-transpose), with an early bail-out once the true
    distance is provably > ``max_dist`` (length gap alone) -- this is only
    ever called against a small, already-bounded candidate list, but stays
    cheap regardless.
    """
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


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
    corrected_terms: dict[str, str] = field(default_factory=dict)
    # Baseline v13 (docs/retrieval_baseline.md "Baseline v13"): additive
    # evidence-strength signal (see tutor.retrieval.assessment) for the app
    # layer to branch on later (weak/empty -> ask the model to rewrite the
    # query). Never changes status/passages above; None only if assessment
    # was skipped (e.g. a cached/legacy response built without it).
    assessment: Any = None

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
            "corrected_terms": self.corrected_terms,
            "assessment": self.assessment.to_dict() if self.assessment is not None else None,
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


class ArticleUnavailable(Exception):
    """Raised by :meth:`ResearchEngine.fetch_article_text` when the archive
    or entry backing a stored passage cannot be read right now (archive
    missing/unavailable, worker timeout/crash, or the entry no longer
    resolves). Never carries a filesystem path -- the caller (a route)
    turns this into a clean 503 for the UI."""


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

    def fetch_article_text(
        self, archive_id: str, path: str, *, deadline_s: float = 5.0
    ) -> str:
        """Re-render the article at ``path`` in ``archive_id`` to plain text
        via the SAME extraction (``tutor.retrieval.zim.bundle.build_bundle``,
        which calls ``content.render_text``) that produced the passages cut
        from it, so a stored passage's ``text`` is guaranteed to be an exact
        substring of the returned text at the same offsets.

        Goes through this engine's own worker pool (``_get_worker``), so it
        shares deadline handling and process reuse with ``research()``.
        Raises :class:`ArticleUnavailable` -- never a lower-level exception
        -- for an unknown archive id, a dead/timed-out worker, or a missing
        entry, so callers (routes) can map this uniformly to a 503.
        """
        from tutor.retrieval.zim.bundle import build_bundle

        try:
            entry = self._registry.get(archive_id)
        except RegistryError as exc:
            raise ArticleUnavailable(str(exc)) from exc
        worker = self._get_worker(entry)
        result = worker.request("fetch_entry", deadline_s=deadline_s, path=path)
        if result.status != "ok" or result.value is None:
            raise ArticleUnavailable(result.error or f"fetch_entry status={result.status}")
        fetched = result.value
        bundle = build_bundle(fetched.html, path=fetched.path, title=fetched.title)
        return bundle.text

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

    def _correct_spelling(
        self,
        worker: Any,
        entry: ArchiveEntry,
        terms: list[str],
        *,
        deadline_s: float,
        memo: dict[tuple[Any, ...], dict[str, Any]] | None,
    ) -> dict[str, tuple[str, int]]:
        """Baseline v11: try to correct each of ``terms`` (already known to
        have ~zero corpus-wide matches) using the archive's own title-
        suggestion search as the candidate source -- no wordlist/dependency
        added. For each term: title-suggest it (libzim's suggestion search
        is itself edit-distance tolerant, so a misspelling like "heluim"
        still often surfaces "Helium" as a title), collect candidate words
        from the returned titles within Damerau distance <=
        ``_SPELL_MAX_EDIT_DISTANCE`` of the term, then pick whichever
        candidate has the highest real ``estimated_matches`` -- a genuine
        rare-but-correctly-spelt term simply won't have a close, much-more-
        common neighbor and is correctly left uncorrected.

        Returns ``{term: (corrected_term, corrected_matches)}`` for terms a
        correction was found for; terms with no qualifying candidate are
        omitted entirely (never a fatal error).
        """
        # (term, prefix) pairs to query, longest prefix (drop=0, the term
        # itself) first -- see ``_SPELL_PREFIX_DROPS``.
        variants: list[tuple[str, str]] = []
        for term in terms:
            for drop in _SPELL_PREFIX_DROPS:
                prefix = term[: len(term) - drop]
                if len(prefix) < _SPELL_MIN_PREFIX_LEN:
                    break
                variants.append((term, prefix))
        suggest_res = _call_worker_multi(
            worker,
            [("search_titles", {"query": p, "limit": _SPELL_SUGGEST_LIMIT}) for _, p in variants],
            deadline_s=deadline_s,
            memo=memo,
        )
        candidates_by_term: dict[str, set[str]] = {t: set() for t in terms}
        for i, (term, _prefix) in enumerate(variants):
            sub_result = (
                suggest_res.value[i]
                if suggest_res is not None and suggest_res.status in ("ok", "partial")
                else None
            )
            if sub_result is None or sub_result["status"] != "ok":
                continue
            for hit in sub_result["value"] or []:
                for word in tokenize(hit.title):
                    word = word.lower()
                    if len(word) < _SPELL_MIN_TERM_LEN or word == term:
                        continue
                    if _damerau_levenshtein(term, word, _SPELL_MAX_EDIT_DISTANCE) <= (
                        _SPELL_MAX_EDIT_DISTANCE
                    ):
                        candidates_by_term[term].add(word)
        all_candidates = sorted({w for words in candidates_by_term.values() for w in words})
        if not all_candidates:
            return {}
        matches_res = _call_worker_multi(
            worker,
            [("estimated_matches", {"term": w}) for w in all_candidates],
            deadline_s=deadline_s,
            memo=memo,
        )
        matches_by_candidate: dict[str, int] = {}
        for i, word in enumerate(all_candidates):
            sub_result = (
                matches_res.value[i]
                if matches_res is not None and matches_res.status in ("ok", "partial")
                else None
            )
            if (
                sub_result is not None
                and sub_result["status"] == "ok"
                and sub_result["value"] is not None
            ):
                matches_by_candidate[word] = int(sub_result["value"])
        corrections: dict[str, tuple[str, int]] = {}
        for term, words in candidates_by_term.items():
            best_word = None
            best_matches = 0
            for word in words:
                matches = matches_by_candidate.get(word, 0)
                if matches > best_matches:
                    best_matches = matches
                    best_word = word
            if best_word is not None and best_matches >= _SPELL_MIN_CORRECTED_MATCHES:
                corrections[term] = (best_word, best_matches)
        return corrections

    def _batch_estimated_matches(
        self,
        worker: Any,
        queries: list[str],
        *,
        deadline_s: float,
        memo: dict[tuple[Any, ...], dict[str, Any]] | None,
    ) -> dict[str, int]:
        """One ``multi`` round-trip of ``estimated_matches`` for each
        distinct query string in ``queries`` (order-preserving dedupe).
        Missing/failed sub-results degrade to ``0`` ("no signal"), same as
        every other estimated-matches use in this module.
        """
        uniq = list(dict.fromkeys(queries))
        if not uniq:
            return {}
        res = _call_worker_multi(
            worker,
            [("estimated_matches", {"term": q}) for q in uniq],
            deadline_s=deadline_s,
            memo=memo,
        )
        out: dict[str, int] = {}
        for i, q in enumerate(uniq):
            sub_result = (
                res.value[i] if res is not None and res.status in ("ok", "partial") else None
            )
            if (
                sub_result is not None
                and sub_result["status"] == "ok"
                and sub_result["value"] is not None
            ):
                out[q] = int(sub_result["value"])
            else:
                out[q] = 0
        return out

    def _correct_compounds(
        self,
        worker: Any,
        entry: ArchiveEntry,
        zero_terms: list[str],
        ordered_tokens: list[str],
        *,
        deadline_s: float,
        memo: dict[tuple[Any, ...], dict[str, Any]] | None,
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Baseline v13: deterministic compound-word query variants for a
        weak/empty first pass -- see the constants above for the exact
        acceptance rule for each direction. Returns ``(corrected_terms,
        term_matches)`` where ``corrected_terms`` maps the original
        fused/split text to the corrected phrase/word (for
        ``ResearchResponse.corrected_terms``, so the UI/model can say
        "Searching for: square foot gardening"), and ``term_matches`` maps
        each real word surfaced (both halves of a winning split, or the
        winning joined/hyphenated form) to its own corpus-wide
        ``estimated_matches`` -- merged by the caller into the normal
        per-request ``term_matches`` so these real words flow through the
        existing rarest-term title/full-text search machinery unchanged.
        """
        corrections: dict[str, str] = {}
        match_counts: dict[str, int] = {}

        # SPLITS: e.g. "squarefoot" -> ("square", "foot").
        split_plan: dict[str, list[tuple[str, str]]] = {}
        split_queries: list[str] = []
        for term in zero_terms[:_MAX_COMPOUND_TERMS]:
            if len(term) < 2 * _COMPOUND_MIN_HALF_LEN:
                continue
            cands = [
                (term[:i], term[i:])
                for i in range(_COMPOUND_MIN_HALF_LEN, len(term) - _COMPOUND_MIN_HALF_LEN + 1)
            ]
            if cands:
                split_plan[term] = cands
                for left, right in cands:
                    split_queries.extend([left, right, f'"{left} {right}"'])
        if split_plan and deadline_s > 0:
            counts = self._batch_estimated_matches(
                worker, split_queries, deadline_s=deadline_s, memo=memo
            )
            for term, cands in split_plan.items():
                best: tuple[str, str] | None = None
                best_phrase_matches = -1
                for left, right in cands:
                    if (
                        counts.get(left, 0) >= _COMPOUND_MIN_HALF_MATCHES
                        and counts.get(right, 0) >= _COMPOUND_MIN_HALF_MATCHES
                    ):
                        phrase_matches = counts.get(f'"{left} {right}"', 0)
                        if phrase_matches > best_phrase_matches:
                            best_phrase_matches = phrase_matches
                            best = (left, right)
                if best is not None:
                    left, right = best
                    corrections[term] = f"{left} {right}"
                    match_counts[left] = counts.get(left, 0)
                    match_counts[right] = counts.get(right, 0)

        # JOINS: e.g. "sun" + "flower" -> "sunflower" (also tries the
        # hyphenated form, "photo-synthesis"-style).
        seen_pairs: set[tuple[str, str]] = set()
        pairs: list[tuple[str, str]] = []
        for a, b in zip(ordered_tokens, ordered_tokens[1:], strict=False):
            if len(a) < _COMPOUND_MIN_HALF_LEN or len(b) < _COMPOUND_MIN_HALF_LEN:
                continue
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            pairs.append((a, b))
        pairs = pairs[:_MAX_COMPOUND_TERMS]
        if pairs and deadline_s > 0:
            join_queries: list[str] = []
            for a, b in pairs:
                join_queries.extend([f"{a}{b}", f'"{a} {b}"', f"{a}-{b}"])
            counts = self._batch_estimated_matches(
                worker, join_queries, deadline_s=deadline_s, memo=memo
            )
            for a, b in pairs:
                joined, phrase, hyphen = f"{a}{b}", f'"{a} {b}"', f"{a}-{b}"
                phrase_matches = counts.get(phrase, 0)
                joined_matches = counts.get(joined, 0)
                hyphen_matches = counts.get(hyphen, 0)
                if phrase_matches <= _COMPOUND_JOIN_PHRASE_CEILING and (
                    joined_matches > 0 or hyphen_matches > 0
                ):
                    key = f"{a} {b}"
                    if joined_matches >= hyphen_matches:
                        corrections[key] = joined
                        match_counts[joined] = joined_matches
                    else:
                        corrections[key] = hyphen
                        match_counts[hyphen] = hyphen_matches

        return corrections, match_counts

    def _process_archive(
        self,
        entry: ArchiveEntry,
        query: str,
        remaining: Any,
        keywords: list[str] | None = None,
        topic_hint: str | None = None,
        query_vec: Sequence[float] | None = None,
        memo: dict[tuple[Any, ...], dict[str, Any]] | None = None,
        corrected_terms_out: dict[str, str] | None = None,
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
        correction_terms: set[str] = set()
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
            # Baseline v11 (docs/retrieval_baseline.md "Baseline v11"): a
            # term with zero (or near-zero) corpus-wide matches is either a
            # genuinely absent word or a misspelt KEY term (e.g. "heluim"
            # for "helium") -- the two look identical from IDF alone. Try a
            # spelling correction ONLY for these weak-signal terms (the
            # normal path is left untouched for every term that already has
            # a real rarity signal), using the archive's own title-
            # suggestion search as the candidate source so no wordlist/
            # dependency is added.
            zero_terms = [
                t
                for t in candidate_terms
                if term_matches.get(t, 0) <= _SPELL_ZERO_MATCH_CEILING
                and len(t) >= _SPELL_MIN_TERM_LEN
            ]
            spelling_corrections: dict[str, tuple[str, int]] = {}
            if _SPELL_FALLBACK_ENABLED and zero_terms and remaining() > 0:
                spelling_corrections = self._correct_spelling(
                    worker,
                    entry,
                    zero_terms[:_MAX_SPELL_CORRECTIONS],
                    deadline_s=_op_deadline(),
                    memo=memo,
                )
            correction_terms: set[str] = set()
            if spelling_corrections:
                for orig, (fixed, matches) in spelling_corrections.items():
                    term_matches[fixed] = matches
                    correction_terms.add(fixed)
                    if corrected_terms_out is not None:
                        corrected_terms_out[orig] = fixed
            # A fused compound word ("solarsystem") is still a real (if
            # unusual) token to the archive's search index -- it can have a
            # handful of incidental corpus-wide matches (e.g. 3, from
            # running-text typos elsewhere) even though it is NOT the
            # genuine, common term. Splitting therefore uses a looser
            # near-zero ceiling than the spelling fallback's strict-zero
            # one; a term the spelling fallback already fixed (e.g.
            # "dinasors" -> "dinosaurs") is a genuine misspelling, not a
            # fused compound, so it is excluded from the split candidate
            # list (trying to also SPLIT it risks a spurious accidental
            # split of two short, individually-real-but-unrelated fragments
            # overwriting a correct spelling fix).
            near_zero_terms = [
                t
                for t in candidate_terms
                if term_matches.get(t, 0) <= _COMPOUND_NEAR_ZERO_CEILING
                and len(t) >= 2 * _COMPOUND_MIN_HALF_LEN
                and t not in spelling_corrections
            ]
            # Baseline v13: compound-word (fused/split) variants -- only
            # tried when this archive's first pass already looks weak
            # (literally no fulltext/title hits from the joined query), a
            # term has ~zero matches, or a term is a near-zero-match fused
            # candidate, exactly like the spelling fallback above; never on
            # an otherwise-strong result. The join direction (adjacent
            # word pairs -- see ``_correct_compounds``) still needs at
            # least one weak signal to engage too, even though neither
            # half of a bad pairing is itself near-zero (both "sun" and
            # "flower" are common) -- gated the same way rather than
            # unconditionally, to bound worker calls to weak requests only.
            compound_weak_signal = (
                bool(zero_terms) or bool(near_zero_terms) or (not fulltext_hits and not title_hits)
            )
            if _SPELL_FALLBACK_ENABLED and compound_weak_signal and remaining() > 0:
                compound_corrections, compound_matches = self._correct_compounds(
                    worker,
                    entry,
                    near_zero_terms[:_MAX_COMPOUND_TERMS],
                    tokens,
                    deadline_s=_op_deadline(),
                    memo=memo,
                )
                for word, matches in compound_matches.items():
                    if matches > 0:
                        term_matches[word] = matches
                        correction_terms.add(word)
                if compound_corrections and corrected_terms_out is not None:
                    for orig, fixed in compound_corrections.items():
                        corrected_terms_out[orig] = fixed
                if compound_corrections and remaining() > 0:
                    # Rebuild the joined query with each fused/split token
                    # swapped for its corrected phrase/word (e.g.
                    # "squarefoot garden right way" ->
                    # "square foot garden right way") and re-issue the same
                    # fulltext+title search the normal path already does --
                    # this is what actually lets the AND-style joined query
                    # find "Square foot gardening" (a single-term entity
                    # title search for "square" or "foot" alone does not).
                    corrected_tokens: list[str] = []
                    for tok in tokens:
                        if tok in compound_corrections:
                            corrected_tokens.extend(compound_corrections[tok].split())
                        else:
                            corrected_tokens.append(tok)
                    for orig_pair, fixed in compound_corrections.items():
                        if " " in orig_pair:
                            a, b = orig_pair.split(" ", 1)
                            for i in range(len(corrected_tokens) - 1):
                                if corrected_tokens[i] == a and corrected_tokens[i + 1] == b:
                                    corrected_tokens[i : i + 2] = [fixed]
                                    break
                    corrected_query = " ".join(corrected_tokens)
                    if corrected_query and corrected_query != search_query:
                        compound_res = _call_worker_multi(
                            worker,
                            [
                                (
                                    "search_fulltext",
                                    _fulltext_kwargs(corrected_query, _FULLTEXT_LIMIT),
                                ),
                                (
                                    "search_titles",
                                    {"query": corrected_query, "limit": _TITLE_LIMIT},
                                ),
                            ],
                            deadline_s=_op_deadline(),
                            memo=memo,
                        )
                        if compound_res is not None and compound_res.status in ("ok", "partial"):
                            ft_sub = compound_res.value[0]
                            ti_sub = compound_res.value[1]
                            if ft_sub["status"] == "ok" and ft_sub["value"]:
                                fulltext_hits = _dedupe_by_path([*fulltext_hits, *ft_sub["value"]])
                            if ti_sub["status"] == "ok" and ti_sub["value"]:
                                title_hits = _dedupe_by_path([*title_hits, *ti_sub["value"]])
            # Rarest (fewest corpus-wide matches) first; a term with zero
            # matches anywhere in the archive (a misspelling) carries no
            # rarity signal and is dropped, same rationale as
            # ``rank_terms_by_rarity``. Corrected terms (not in
            # ``unique_tokens``) sort ahead of same-frequency original
            # terms -- they are a strong, deliberately-surfaced signal.
            rarity_order = sorted(
                (t for t in term_matches if term_matches[t] > 0),
                key=lambda t: (
                    term_matches[t],
                    unique_tokens.index(t) if t in unique_tokens else -1,
                ),
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
                    {*tokenize(strip_instruction_words(query)), *correction_terms}
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
        own_terms = frozenset(
            {*tokenize(strip_instruction_words(query)), *correction_terms}
        )
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
        # Baseline v11: spelling-tolerant fallback records any corrections
        # made across all archives consulted this request (e.g.
        # {"heluim": "helium"}) so the UI/model can say "showing results
        # for helium" -- see ``_correct_spelling``.
        corrected_terms_out: dict[str, str] = {}

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
                    corrected_terms_out=corrected_terms_out,
                )
                any_timeout = any_timeout or timed_out
                if dense_note is not None:
                    dense_notes.append(dense_note)
                if any(c.get("sources") and "dense" in c["sources"] for c in cands):
                    dense_used = True
                candidates.extend(cands)
                key_fact_candidates.extend(key_facts)

        _consult(primary_archives)

        # Baseline v11: a spelling correction found during archive
        # consultation (e.g. "photosynthsis" -> "photosynthesis") must
        # also count for the coverage gate below, or a passage that only
        # ever contains the CORRECTED spelling (which is all real articles
        # do) looks like weak/no coverage of the question's own terms and
        # gets dropped as "empty" even though the right article was found.
        if corrected_terms_out:
            coverage_terms = coverage_terms | frozenset(corrected_terms_out.values())

        if fallback_archives and not soft_elapsed() and remaining() > 0:
            preliminary_coverage = _best_coverage(
                candidates, coverage_terms, topic_hint_terms, own_term_count
            )
            if preliminary_coverage["weak"]:
                _consult(fallback_archives)
                if corrected_terms_out:
                    coverage_terms = coverage_terms | frozenset(corrected_terms_out.values())

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
            corrected_terms=dict(corrected_terms_out),
        )
        response.assessment = assess_evidence(query, response)
        self._response_cache[cache_key] = response
        self._response_cache.move_to_end(cache_key)
        while len(self._response_cache) > _RESPONSE_CACHE_MAXSIZE:
            self._response_cache.popitem(last=False)
        return response

    def research_many(
        self,
        queries: Sequence[str],
        *,
        keywords: list[str] | None = None,
        budget_tokens: int | None = None,
        deadline_s: float | None = None,
        soft_deadline_s: float | None = None,
        topic_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run several queries against :meth:`research` as one batched
        request, sharing a single deadline across the whole batch and this
        engine's own per-archive worker pool / IDF/response caches (both
        already persistent on ``self``, so concurrent calls reuse the same
        warm worker processes instead of paying per-call startup cost).

        Each query runs in its own thread: per-query wall time is itself
        dominated by blocking IPC waits on the out-of-process ZIM worker
        (``docs/retrieval_baseline.md``'s profiling note), which release the
        GIL, so N queries against distinct/pooled workers cost close to the
        slowest single query rather than the sum of all of them -- unlike
        the previous sequential-call path this replaces.

        Returns a list, same order/length as ``queries``, of
        ``{"query": str, "status": "ok" | "error", "response":
        ResearchResponse | None, "error": str | None}`` -- a per-query
        failure (exception or the shared deadline already elapsed) never
        raises or drops a slot, so callers can always zip results back
        against their original queries.
        """
        if not queries:
            return []

        batch_deadline = deadline_s if deadline_s is not None else self._hard_deadline_s
        batch_started = time.monotonic()

        def _run_one(q: str) -> dict[str, Any]:
            remaining = batch_deadline - (time.monotonic() - batch_started)
            if remaining <= 0:
                return {
                    "query": q,
                    "status": "error",
                    "response": None,
                    "error": "deadline elapsed",
                }
            try:
                resp = self.research(
                    q,
                    keywords=keywords,
                    budget_tokens=budget_tokens,
                    deadline_s=remaining,
                    soft_deadline_s=soft_deadline_s,
                    topic_hint=topic_hint,
                )
                return {"query": q, "status": "ok", "response": resp, "error": None}
            except Exception as exc:  # pragma: no cover - defensive, per-query isolation
                return {"query": q, "status": "error", "response": None, "error": str(exc)}

        results: list[dict[str, Any] | None] = [None] * len(queries)
        threads = []
        for i, q in enumerate(queries):

            def _target(idx=i, query=q):
                results[idx] = _run_one(query)

            t = threading.Thread(target=_target, daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=batch_deadline + _QUEUE_SLACK_S)

        return [
            r
            if r is not None
            else {"query": q, "status": "error", "response": None, "error": "timed out"}
            for r, q in zip(results, queries, strict=True)
        ]
