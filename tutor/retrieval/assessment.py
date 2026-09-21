"""Evidence-strength assessment for a completed retrieval result.

Ours: no donor code. Separate from :mod:`tutor.retrieval.research`'s own
internal coverage gate (``_best_coverage``, which decides "ok" vs "empty"/
"partial" status and is already load-bearing for existing behaviour) --
this module is a NEW, additive signal (``ResearchResponse.assessment``)
that the app layer can branch on later (owner decision 2026-09-20: weak
evidence should trigger cheap deterministic query variants, then a
model-forced query rewrite). It must never change ``research()``'s
existing status/passages behaviour.

Thresholds below are tuned ONLY on the 42-question TUNING split of
eval/questions/simplewiki_questions.jsonl (``python -m
eval.run_retrieval_eval --split tuning``), never on held-out data. See
docs/retrieval_baseline.md "Baseline v13" for the measured confusion
counts this was chosen from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tutor.retrieval.hybrid.lexical import singularize, strip_instruction_words, tokenize

# Extra "question-shape" words that survive ``tokenize``'s stopword list
# (which already drops how/what/do/does/...) but still carry no content of
# their own for THIS purpose -- coverage is about the topic being asked
# about, not the phrasing used to ask about it.
_QUESTION_FILLERS = frozenset(
    {
        "right",
        "way",
        "ways",
        "kind",
        "kinds",
        "sort",
        "sorts",
        "best",
        "good",
        "proper",
        "properly",
        "correctly",
    }
)

# Baseline v13 (measured on the 42-question tuning split, see
# docs/retrieval_baseline.md "Baseline v13"): a coverage fraction (of the
# question's own key terms, stem-matched, found in the top passages'
# title+text) at or above this is "strong"; below it (with at least one
# passage present) is "weak". Chosen as the value that called every
# tuning gold-in-top-5 case "strong" while calling every observed miss
# "weak" or "empty" (a miss called "strong" is the costly error this
# threshold is set to avoid).
_COVERAGE_STRONG_THRESHOLD = 0.5

# How many of the top-ranked passages (already best-first) count toward
# coverage -- matches ``research._COVERAGE_CANDIDATES_CHECKED``'s rationale:
# the single top passage is not always the semantically right one.
_TOP_PASSAGES_CHECKED = 3

# NOTE on score-based floor: BM25/RRF-fused passage scores were checked
# across tuning questions of very different lengths/term-rarity and are
# NOT comparable in magnitude across queries (a short, common-word query
# and a long, rare-term query produce scores on different scales with no
# shared floor that separates strong from weak without also cutting good
# short-query results). Per the task's own instruction ("if they aren't
# [comparable], don't use them"), this module does NOT use an absolute
# score floor -- only term coverage and title-match, both scale-free.


def _key_terms(question: str) -> frozenset[str]:
    """The question's own content terms: ``strip_instruction_words`` +
    ``tokenize`` (drops English stopwords, including how/what/do/does),
    then drop the extra question-shape fillers above (right/way/...)."""
    return frozenset(tokenize(strip_instruction_words(question))) - _QUESTION_FILLERS


def _passage_title_text(passage: Any) -> tuple[str, str]:
    title = passage.get("title", "") if isinstance(passage, dict) else getattr(passage, "title", "")
    text = passage.get("text", "") if isinstance(passage, dict) else getattr(passage, "text", "")
    return title or "", text or ""


def _passage_terms(passage: Any) -> frozenset[str]:
    title, text = _passage_title_text(passage)
    return frozenset(singularize(t) for t in tokenize(f"{title} {text}"))


# Generic single-word terms that, on their own, are common enough across the
# archive that covering them alone must never be enough to certify a passage
# as being about the question's actual topic (case: "square foot gardening"
# -> key terms square/foot/garden, and articles titled "Garden"/"Square"/
# "Foot"/"Chromatica" (a song containing the word "foot") each trivially
# cover one of these while being entirely off-topic). This is a coarse,
# curated stand-in for real corpus-frequency (IDF) weighting; see
# docs/rewrite_probe_measure.md for why full IDF plumbing was deferred.
_GENERIC_SINGLE_WORDS = frozenset(
    {
        "garden",
        "foot",
        "feet",
        "square",
        "work",
        "works",
        "make",
        "makes",
        "food",
        "place",
        "water",
        "play",
        "plant",
        "plants",
        "people",
        "thing",
        "things",
        "part",
        "parts",
        "time",
        "guide",
        "basics",
        "step",
        "steps",
    }
)


def _topic_candidates(
    key_terms: frozenset[str], corrected_terms: Mapping[str, str] | None
) -> list[str]:
    """Ordered (longest-first, then alphabetical for a stable tie-break)
    candidates for the question's "main topic" phrase: corrected phrases
    (e.g. "squarefoot" -> "square foot") if any correction happened,
    otherwise non-generic key terms, otherwise every key term (better than
    nothing when every term is generic)."""
    if corrected_terms:
        phrases = [v.lower() for v in corrected_terms.values() if v]
        if phrases:
            return sorted(phrases, key=lambda p: (-len(p), p))
    candidates = [t.lower() for t in key_terms if t.lower() not in _GENERIC_SINGLE_WORDS]
    pool = candidates or [t.lower() for t in key_terms]
    return sorted(pool, key=lambda p: (-len(p), p))


def _topic_phrase(key_terms: frozenset[str], corrected_terms: Mapping[str, str] | None) -> str:
    """The single longest topic candidate (see ``_topic_candidates``), kept
    for callers that just want a label -- ``assess_evidence`` itself now
    picks among ALL candidates via ``_select_topic_phrase`` below."""
    candidates = _topic_candidates(key_terms, corrected_terms)
    return candidates[0] if candidates else ""


def _phrase_in_passage(topic_phrase: str, passage: Any) -> bool:
    """True if ``topic_phrase`` (possibly multi-word, e.g. "square foot")
    appears in THIS passage's TITLE, or as an exact phrase in its TEXT, or
    -- to tolerate plural/singular mismatches (e.g. "volcanoes" vs the
    title "Volcano") and reordered multi-word phrases (e.g. title words in
    a different order than the phrase) -- every word of the phrase,
    singularized, appears somewhere in this SAME passage's title+text.
    Words matched across DIFFERENT passages does NOT satisfy this -- that
    is exactly the fused-term-split false-strong failure mode this guards
    against."""
    title, text = _passage_title_text(passage)
    title_l, text_l = title.lower(), text.lower()
    if topic_phrase in title_l or topic_phrase in text_l:
        return True
    phrase_words = [singularize(w) for w in topic_phrase.split() if w]
    if not phrase_words:
        return False
    passage_words = {singularize(w) for w in tokenize(f"{title} {text}")}
    return all(w in passage_words for w in phrase_words)


def _topic_present(topic_phrase: str, passages: list[Any]) -> bool:
    """True if ``topic_phrase`` is present (see ``_phrase_in_passage``) in
    some single top passage."""
    if not topic_phrase:
        return True
    return any(_phrase_in_passage(topic_phrase, p) for p in passages[:_TOP_PASSAGES_CHECKED])


def _select_topic_phrase(
    key_terms: frozenset[str],
    corrected_terms: Mapping[str, str] | None,
    passages: list[Any],
) -> tuple[str, bool]:
    """Pick the best "main topic" candidate: the longest candidate that is
    actually present (see ``_topic_present``) in the top passages, so a
    generic/incidental longest word (a verb like "explain", a question-
    shape noun like "difference") doesn't get committed to ahead of a real
    entity term the passages DO cover. Falls back to the single longest
    candidate (old behaviour) -- not found -- when none of them match,
    which is exactly the true-miss case (e.g. a nonsense topic) this
    check exists to still catch."""
    candidates = _topic_candidates(key_terms, corrected_terms)
    if not candidates:
        return "", True
    for candidate in candidates:
        if _topic_present(candidate, passages):
            return candidate, True
    return candidates[0], False


@dataclass(frozen=True)
class EvidenceAssessment:
    """Result of :func:`assess_evidence`."""

    level: str  # "strong" | "weak" | "empty"
    reasons: list[str] = field(default_factory=list)
    key_terms: frozenset[str] = field(default_factory=frozenset)
    covered_terms: frozenset[str] = field(default_factory=frozenset)
    coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "key_terms": sorted(self.key_terms),
            "covered_terms": sorted(self.covered_terms),
            "coverage": self.coverage,
        }


def assess_evidence(
    question: str,
    result: Any,
    *,
    rewritten_queries: list[str] | None = None,
    healthy_terms: Any = None,
    corrected_terms: Mapping[str, str] | None = None,
) -> EvidenceAssessment:
    """Assess how well ``result`` (anything with a ``.passages`` sequence
    of objects/dicts exposing ``title``/``text``) answers ``question``.

    ``empty``: no passages at all.
    ``weak``: passages exist, but the question's own key content terms
        (stems/plurals normalised via ``singularize``, e.g. "garden" and
        "gardening" both reduce to a form that lets "garden" match) are
        poorly covered by the top ``_TOP_PASSAGES_CHECKED`` passages'
        title+text -- coverage below ``_COVERAGE_STRONG_THRESHOLD``.
    ``strong``: otherwise.

    ``rewritten_queries`` (optional, additive): when given (e.g. by the
    forced-rewrite re-assessment round in ``tutor/app/agent_loop.py``), the
    terms checked for coverage are the union of each rewritten query's own
    key content terms (a misspelt/fused original term like "squarefoot"
    can never be covered by good passages once the model has already
    corrected it to "square foot", so re-checking the ORIGINAL question's
    terms would permanently pin coverage low) plus, from ``healthy_terms``
    if given, any of the original question's own key terms that already
    had a healthy match count there (kept so a correctly-spelt original
    term is not silently dropped just because a rewrite happened).
    ``healthy_terms`` may be any iterable of term strings, or a mapping of
    term -> truthy "healthy" flag / estimated-match count (only truthy
    entries are kept); anything falsy for a term drops it. With no
    ``rewritten_queries``, behaviour is unchanged (original question terms
    only) -- fully backward compatible.
    """
    passages = getattr(result, "passages", None)
    if passages is None and isinstance(result, dict):
        passages = result.get("passages")
    passages = passages or []

    if rewritten_queries:
        key_terms: frozenset[str] = frozenset()
        for rq in rewritten_queries:
            key_terms |= _key_terms(rq)
        if healthy_terms is not None:
            if isinstance(healthy_terms, Mapping):
                healthy = frozenset(t for t, v in healthy_terms.items() if v)
            else:
                healthy = frozenset(healthy_terms)
            key_terms |= _key_terms(question) & healthy
    else:
        key_terms = _key_terms(question)

    if not passages:
        return EvidenceAssessment(
            level="empty",
            reasons=["no passages returned"],
            key_terms=key_terms,
            covered_terms=frozenset(),
            coverage=0.0,
        )

    if not key_terms:
        # No content terms of our own to check coverage against (e.g. a
        # pure follow-up with no topic hint folded in here) -- can't call
        # this weak on term coverage, so treat presence of evidence as
        # strong (existing research()-level coverage gate already handles
        # the abstention decision for such queries).
        return EvidenceAssessment(
            level="strong",
            reasons=["no key terms to check; passages present"],
            key_terms=key_terms,
            covered_terms=frozenset(),
            coverage=1.0,
        )

    norm_key_terms = {singularize(t) for t in key_terms}
    covered_norm: set[str] = set()
    for passage in passages[:_TOP_PASSAGES_CHECKED]:
        covered_norm |= norm_key_terms & _passage_terms(passage)

    coverage = len(covered_norm) / len(norm_key_terms) if norm_key_terms else 1.0
    covered_terms = frozenset(t for t in key_terms if singularize(t) in covered_norm)

    topic_phrase, topic_ok = _select_topic_phrase(key_terms, corrected_terms, list(passages))

    reasons = [f"coverage {coverage:.2f} vs threshold {_COVERAGE_STRONG_THRESHOLD}"]
    if topic_phrase:
        reasons.append(
            f"topic phrase {topic_phrase!r} {'found' if topic_ok else 'NOT found'} "
            "in top passages' title/text"
        )

    if coverage >= _COVERAGE_STRONG_THRESHOLD and topic_ok:
        reasons.append("verdict: strong")
        return EvidenceAssessment(
            level="strong",
            reasons=reasons,
            key_terms=key_terms,
            covered_terms=covered_terms,
            coverage=coverage,
        )
    if coverage >= _COVERAGE_STRONG_THRESHOLD and not topic_ok:
        reasons.append(
            "verdict: weak (coverage alone met by generic/incidental terms; "
            "main topic absent from title/text)"
        )
    else:
        reasons.append(f"missing key terms: {sorted(key_terms - covered_terms)}")
        reasons.append("verdict: weak")
    return EvidenceAssessment(
        level="weak",
        reasons=reasons,
        key_terms=key_terms,
        covered_terms=covered_terms,
        coverage=coverage,
    )
