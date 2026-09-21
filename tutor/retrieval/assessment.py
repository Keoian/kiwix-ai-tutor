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

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tutor.retrieval.hybrid.lexical import (
    QUESTION_SHAPE_FILLERS as _QUESTION_FILLERS,
)
from tutor.retrieval.hybrid.lexical import is_superlative_word as _is_superlative_word
from tutor.retrieval.hybrid.lexical import singularize, strip_instruction_words, tokenize

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
        # Assessor v3 additions (docs/rewrite_probe_measure.md "Assessor
        # v3"): single words that are the generic HALF of a multi-word
        # named topic seen in the labelled set -- "War"/"World" alone
        # doesn't prove "World War Two" any more than "Garden" alone
        # proves "square foot gardening"; same for the other halves of
        # "French Revolution", "rain forest", "planets in the Solar
        # System".
        "war",
        "world",
        "revolution",
        "rain",
        "forest",
        "planet",
        "planets",
        "solar",
        "system",
        "blue",
        "whale",
    }
)


# Assessor v3 (docs/rewrite_probe_measure.md "Assessor v3"): number-word /
# roman-numeral / digit equivalence classes, so a phrase like "world war
# two" is recognised as matching a title that spells it "World War II" or
# "World War 2" -- and, just as importantly, recognised as NOT matching a
# title that spells a DIFFERENT number ("World War I"), which a bare
# any-2-words-of-the-title check would wrongly accept.
_NUMBER_EQUIVALENCE_GROUPS: list[frozenset[str]] = [
    frozenset({"1", "one", "i", "first"}),
    frozenset({"2", "two", "ii", "second"}),
    frozenset({"3", "three", "iii", "third"}),
    frozenset({"4", "four", "iv", "fourth"}),
    frozenset({"5", "five", "v", "fifth"}),
    frozenset({"6", "six", "vi", "sixth"}),
]


def _number_group(word: str) -> frozenset[str] | None:
    for group in _NUMBER_EQUIVALENCE_GROUPS:
        if word in group:
            return group
    return None


def _is_bare_year_or_number(word: str) -> bool:
    return word.isdigit() and len(word) >= 3


# Structural/instructional verbs and connectives that survive
# ``strip_instruction_words`` + ``tokenize`` (they're real English content
# words, just not NAME/topic words) but must never be glued into a
# multi-word topic-integrity phrase or picked as a "rarest term" for
# co-occurrence -- otherwise "Newton's laws of motion explain how things
# move" turns into one giant bogus phrase candidate, and "difference"/
# "between" (long words, but pure question-shape, not content) get
# treated as the two "rarest" terms of a difference-between question.
# Curated the same way as ``_GENERIC_SINGLE_WORDS`` above (no per-term
# corpus-frequency data plumbed through yet); see docs/rewrite_probe_
# measure.md "Assessor v3" for the measurement this was tuned against.
_STRUCTURAL_WORDS = frozenset(
    {
        "caused",
        "cause",
        "causes",
        "made",
        "make",
        "makes",
        "wrote",
        "write",
        "written",
        "explain",
        "explains",
        "describe",
        "describes",
        "output",
        "convert",
        "understand",
        "happen",
        "happens",
        "happened",
        "forms",
        "form",
        "different",
        "difference",
        "between",
        "survived",
        "survive",
        "survives",
        "move",
        "moves",
        "live",
        "lives",
        "living",
    }
)


# Assessor v4 (docs/rewrite_probe_measure.md "Assessor v4 -- superlatives"):
# superlative/comparative size/speed/age modifiers ("biggest", "fastest",
# "largest", ...) are common English adjectives reused across unrelated
# archive articles (a reality show titled "The Biggest Loser", a
# motorsport "Fastest lap" stat) -- a bare match on the modifier word
# alone, with no real connection to the question's actual topic, must
# never by itself certify a passage as covering the question. These are
# MODIFIERS, not topic terms: the topic is the head noun ("animal",
# "bird", "molecule"), and a superlative question only counts as
# genuinely answered when a passage contains BOTH the head noun AND a
# superlative/extreme cue word IN THE SAME passage (see
# ``_superlative_gate_ok`` below). Retrieval v16: ``_SUPERLATIVE_WORDS``,
# ``_NON_SUPERLATIVE_EST_WORDS`` and ``_is_superlative_word`` now live in
# ``tutor.retrieval.hybrid.lexical`` (imported above) so the retrieval
# pipeline (``tutor.retrieval.research``) shares the exact same modifier
# definition -- only re-exported here under their original names so the
# rest of this module (and any external caller importing them from here)
# is unaffected. What follows historically documented the explicit list
# (covers irregular forms leading with a consonant-doubling or "good"/
# "bad"-style
# suppletion, none of which the conservative "-est" rule below would
# catch), then a conservative regex rule for regular "-est" adjectives
# guarded by an exceptions list of common English words that merely END
# in "-est" without being a superlative at all ("forest", "interest",
# ...).
_SUPERLATIVE_CUE_PHRASES = (
    "most massive",
    "record holder",
    "record-holding",
    "record breaking",
    "record-breaking",
)


def _passage_has_superlative_cue(passage: Any) -> bool:
    """True if this passage's title+text contains any superlative/extreme
    cue word or phrase (see ``_SUPERLATIVE_WORDS`` and
    ``_SUPERLATIVE_CUE_PHRASES``)."""
    title, text = _passage_title_text(passage)
    combined = f"{title} {text}"
    lowered = combined.lower()
    for phrase in _SUPERLATIVE_CUE_PHRASES:
        if phrase in lowered:
            return True
    tokens = {singularize(t) for t in tokenize(combined)}
    return any(_is_superlative_word(t) for t in tokens)


def _superlative_gate_ok(
    key_terms: frozenset[str], passages: list[Any]
) -> tuple[bool, list[str]]:
    """Assessor v4 co-occurrence gate for superlative questions (Assessor
    v3's generic same-passage co-occurrence rule was tried project-wide
    and REJECTED for ~14 new false-weak; this is deliberately NARROW --
    it applies only when the question itself contains a superlative/
    comparative modifier, not to every question).

    A superlative question's head noun(s) are the question's key terms
    minus the modifier words themselves. The gate passes if some one of
    the top-checked passages contains BOTH a head-noun term AND a
    superlative/extreme cue, in that SAME passage -- title match OR body
    text, either counts, since the point here is only to rule out
    coverage satisfied piecemeal by unrelated passages (a modifier word
    alone in one passage's title, a head noun alone in a different,
    unrelated passage). Returns ``(gate_ok, modifier_words_found)``; when
    no modifier is present in the question at all, the gate trivially
    passes (this check is inert for non-superlative questions).
    """
    modifiers = [t for t in key_terms if _is_superlative_word(t)]
    if not modifiers:
        return True, []
    head_terms = frozenset(
        singularize(t) for t in key_terms if not _is_superlative_word(t) and len(t) > 1
    )
    if not head_terms:
        # A question that is ENTIRELY modifier words carries no head noun
        # to anchor on; nothing here to gate.
        return True, modifiers
    for passage in passages[:_TOP_PASSAGES_CHECKED]:
        passage_terms = _passage_terms(passage)
        if not (head_terms & passage_terms):
            continue
        if _passage_has_superlative_cue(passage):
            return True, modifiers
    return False, modifiers


_RAW_WORD_RE = re.compile(r"[A-Za-z']+")


def _multiword_phrase_candidates(question: str, key_terms: frozenset[str]) -> list[str]:
    """Consecutive-content-word phrase candidates (Assessor v3 signal
    (a): "multi-word topic integrity"): runs of words that are literally
    ADJACENT in the RAW question text (nothing at all in between -- not
    even a stopword) and each one of ``key_terms``, then every contiguous
    sub-window of length >= 2 within each run (longest first) -- e.g.
    "What caused World War Two?" -> run [world, war, two] ("caused" is a
    structural verb, excluded) -> candidate "world war two". Requiring
    RAW adjacency (rather than adjacency after ``tokenize`` drops
    stopwords) is what tells apart a genuine compound-name phrase like
    "rain forest" from a comparison like "a noun AND a verb" or "the
    capital OF the Moon" -- those have real words between the two content
    terms in the original question, so they never form a run here and
    fall through to the old single-word candidate logic unaffected. A
    question with no adjacent multi-word run yields no candidates here."""
    raw_tokens = [w.lower() for w in _RAW_WORD_RE.findall(question)]
    key_lower = {t.lower() for t in key_terms}
    runs: list[list[str]] = []
    current: list[str] = []
    for tok in raw_tokens:
        if tok in key_lower and tok not in _STRUCTURAL_WORDS:
            current.append(tok)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    # Cap the window at 3 words: real multi-word NAMES in these probes
    # ("world war two", "french revolution", "rain forest", "blue whale",
    # "rose bowl") are 2-3 words; longer surviving runs are usually a
    # noun phrase plus incidental adjacent content words, not a single
    # named topic, and gluing them all together only produces candidates
    # that can never realistically match a title.
    candidates: list[str] = []
    for run in runs:
        n = len(run)
        if n < 2:
            continue
        # If the run carries a number/year word (a "World War Two"-style
        # distinguishing suffix), never generate a shorter sub-window
        # that drops it -- "world war" alone (from the run [world, war,
        # two]) would trivially title-match "World War I" and silently
        # defeat the whole point of the constraint-terms check below.
        number_positions = {
            i
            for i, w in enumerate(run)
            if _number_group(w) is not None or _is_bare_year_or_number(w)
        }
        for length in range(min(n, 3), 1, -1):
            for start in range(0, n - length + 1):
                window = set(range(start, start + length))
                if number_positions and not number_positions.issubset(window):
                    continue
                candidates.append(" ".join(run[start : start + length]))
    # Longest first (stable within same length: first-seen order, i.e.
    # left-to-right in the question, which is as good a tie-break as any).
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return sorted(ordered, key=lambda p: -len(p.split()))


def _phrase_words_match_passage(phrase_words: list[str], passage: Any) -> tuple[int, bool]:
    """(title_word_match_count, constraint_ok) for one passage: how many
    of ``phrase_words`` (singularized, number-equivalence aware) appear in
    the passage's TITLE, and whether every number/year word in the phrase
    is satisfied (an equivalent found somewhere in title+text) -- signal
    (b) "constraint terms": a title matching the generic words of a
    multi-word phrase but carrying a DIFFERENT number/year does not count
    ("World War I" for "world war two"; a 1998 game for a "1994" question)."""
    title, text = _passage_title_text(passage)
    title_words = {singularize(w) for w in tokenize(title)}

    title_count = 0
    constraint_ok = True
    for w in phrase_words:
        group = _number_group(w)
        if group is not None:
            # Constraint terms (signal (b)): a distinguishing number/
            # ordinal word must match IN THE TITLE, not merely be
            # mentioned somewhere in the body text -- an article titled
            # "World War I" mentions "World War II" in passing, which
            # must not certify it as covering a "World War Two" question.
            if title_words & group:
                title_count += 1
            else:
                constraint_ok = False
        elif _is_bare_year_or_number(w):
            if w in title_words:
                title_count += 1
            else:
                constraint_ok = False
        else:
            sw = singularize(w)
            if sw in title_words:
                title_count += 1
    return title_count, constraint_ok


def _multiword_phrase_in_passage(phrase: str, passage: Any) -> bool:
    """Strict multi-word check (signal (a)+(b)): a top-3 passage counts as
    carrying this phrase only if its TITLE contains >= 2 of the phrase's
    words (number-equivalence aware) *and* every number/year word in the
    phrase is satisfied somewhere in that passage, OR the exact phrase
    (words in this order, adjacent) appears literally in the passage
    text. Deliberately stricter than ``_phrase_in_passage``'s loose
    "words present anywhere in the passage" fallback, which is exactly
    what let a passage merely mentioning "eight planets in the Solar
    System" in passing certify "planets solar system" as covered."""
    phrase_words = [w for w in phrase.split() if w]
    if len(phrase_words) < 2:
        return _phrase_in_passage(phrase, passage)
    title, text = _passage_title_text(passage)
    # If every non-number word of the phrase is one of the curated
    # generic words ("war"/"world", "rain"/"forest", "solar"/"system",
    # "blue"/"whale", ...), the exact-phrase-in-text shortcut is skipped:
    # a big general article on the generic topic (Whale, Planet) will
    # often happen to mention the specific phrase once in passing (e.g.
    # "...such as the blue whale...", "...eight planets in the Solar
    # System...") without actually being ABOUT that specific thing, so
    # only a real title match is trusted here. A phrase with at least one
    # non-generic word (e.g. "french revolution": "french" isn't generic)
    # keeps the exact-text-phrase shortcut, since a passage that spells
    # out the whole specific phrase verbatim is good evidence.
    all_generic = all(
        w.lower() in _GENERIC_SINGLE_WORDS
        for w in phrase_words
        if _number_group(w) is None and not _is_bare_year_or_number(w)
    )
    if not all_generic:
        # Word-boundary regex rather than a padded-space substring check,
        # so e.g. "Boiling point: 77.355 K" (colon, no space, right after
        # the phrase) still counts as the exact phrase "boiling point".
        phrase_re = re.compile(
            r"\b" + r"\s+".join(re.escape(w) for w in phrase_words) + r"\b", re.IGNORECASE
        )
        if phrase_re.search(text) or phrase_re.search(title):
            return True
    title_count, constraint_ok = _phrase_words_match_passage(phrase_words, passage)
    return constraint_ok and title_count >= 2


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
    # Assessor v4: a bare superlative/comparative modifier ("biggest",
    # "fastest", ...) must never be picked as the question's "main topic"
    # phrase on its own -- it is a near-universal English adjective that
    # trivially title-matches unrelated articles ("The Biggest Loser").
    # Also drop stray single-character tokens (e.g. "s" leaking in from an
    # apostrophe contraction like "What's" -- ``tokenize`` splits on \w+,
    # which doesn't include the apostrophe): a length-1 token is never a
    # genuine topic word, and worse, ``_phrase_in_passage``'s substring
    # check trivially "finds" a single letter in almost any text.
    non_modifier = [
        t for t in key_terms if not _is_superlative_word(t.lower()) and len(t) > 1
    ]
    candidates = [
        t.lower()
        for t in non_modifier
        if t.lower() not in _GENERIC_SINGLE_WORDS
    ]
    pool = candidates or [t.lower() for t in non_modifier] or [t.lower() for t in key_terms]
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
    question: str = "",
) -> tuple[str, bool]:
    """Pick the best "main topic" candidate: the longest candidate that is
    actually present (see ``_topic_present``) in the top passages, so a
    generic/incidental longest word (a verb like "explain", a question-
    shape noun like "difference") doesn't get committed to ahead of a real
    entity term the passages DO cover. Falls back to the single longest
    candidate (old behaviour) -- not found -- when none of them match,
    which is exactly the true-miss case (e.g. a nonsense topic) this
    check exists to still catch.

    Assessor v3 signal (a) "multi-word topic integrity": when
    ``corrected_terms`` gave no phrase and the question itself contains a
    multi-word content run ("world war two", "french revolution", "rain
    forest", "blue whale", ...), that run's candidates are checked FIRST,
    with the strict ``_multiword_phrase_in_passage`` check (title >= 2
    words + constraint terms, or exact adjacent phrase in text) -- never
    the loose any-order-anywhere-in-passage fallback. If none of them are
    satisfied, the topic is NOT found, even if some single generic word
    from the phrase (e.g. "revolution", "planets") trivially matches a
    passage title on its own; a single-word-only question (no multi-word
    run) is unaffected and falls through to the old candidate logic."""
    if not corrected_terms and question:
        multiword = _multiword_phrase_candidates(question, key_terms)
        if multiword:
            for phrase in multiword:
                if any(
                    _multiword_phrase_in_passage(phrase, p)
                    for p in passages[:_TOP_PASSAGES_CHECKED]
                ):
                    return phrase, True
            # No multi-word candidate matched. Before forcing weak, check
            # whether the OLD single-word logic would have trusted a
            # candidate that is NOT one of the structural words this
            # signal exists to distrust (e.g. "explode" for "What made it
            # explode like that?", "eruptions" for "volcanoe eruptions",
            # "snake" for "how many legs does a snake have") -- a
            # fallback resting on a genuine, specific content word is
            # left alone; only a bare number word is disqualified here
            # (the number-vs-title-numeral tension is exactly what
            # ``_phrase_words_match_passage``'s constraint check exists
            # for, and a lone number word carries no topic content of its
            # own either way).
            # Exception: if the best (longest) candidate is made ENTIRELY
            # of curated generic words ("rain"+"forest", "blue"+"whale",
            # "world"+"war"+"two") -- i.e. neither half is a specific
            # content word we'd otherwise trust alone -- and some top
            # passage's title already carries at least one of those
            # generic words (a related-but-wrong entity: "Rain"/"Forest"
            # for "rain forest", "Whale" for "blue whale"), that's a
            # clear enough "wrong specific topic, only generic overlap"
            # signature that no single-word fallback should reopen it.

            def _all_generic_near_miss(phrase: str) -> bool:
                words = phrase.split()
                all_generic = all(
                    w in _GENERIC_SINGLE_WORDS
                    for w in words
                    if _number_group(w) is None and not _is_bare_year_or_number(w)
                )
                if not all_generic:
                    return False
                return any(
                    _phrase_words_match_passage(words, p)[0] >= 1
                    for p in passages[:_TOP_PASSAGES_CHECKED]
                )

            if not any(_all_generic_near_miss(phrase) for phrase in multiword):
                trusted_terms = frozenset(
                    t for t in key_terms if t.lower() not in _STRUCTURAL_WORDS
                )
                for candidate in _topic_candidates(trusted_terms, None):
                    if _number_group(candidate.lower()) is not None or _is_bare_year_or_number(
                        candidate.lower()
                    ):
                        continue
                    if _topic_present(candidate, passages):
                        return candidate, True
            return multiword[0], False

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

    topic_phrase, topic_ok = _select_topic_phrase(
        key_terms, corrected_terms, list(passages), question
    )

    reasons = [f"coverage {coverage:.2f} vs threshold {_COVERAGE_STRONG_THRESHOLD}"]
    if topic_phrase:
        reasons.append(
            f"topic phrase {topic_phrase!r} {'found' if topic_ok else 'NOT found'} "
            "in top passages' title/text"
        )
    question_modifiers = sorted(t for t in key_terms if _is_superlative_word(t))
    if question_modifiers:
        reasons.append(
            f"superlative modifier(s) {question_modifiers!r} detected -- excluded "
            "from topic-phrase selection and coverage-alone credit (Assessor v4)"
        )

    if coverage >= _COVERAGE_STRONG_THRESHOLD and topic_ok:
        gate_ok, modifiers = _superlative_gate_ok(key_terms, list(passages))
        if not gate_ok:
            reasons.append(
                "verdict: weak (superlative modifier "
                f"{sorted(modifiers)!r} matched separately from the head "
                "noun topic; no passage has both in the same passage)"
            )
            return EvidenceAssessment(
                level="weak",
                reasons=reasons,
                key_terms=key_terms,
                covered_terms=covered_terms,
                coverage=coverage,
            )
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
