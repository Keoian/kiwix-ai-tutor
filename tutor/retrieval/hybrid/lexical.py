"""Tokenizer and hand-rolled BM25 for passage-level lexical ranking.

Ours: no donor code. See docs/plan/offline_tutor_kiwix_reuse_plan.md §4
("passage BM25", "pinned tokenizer") and offline_tutor_spec_v0.3.md §7.2
step 6 ("BM25 over the request's passage pool (k1 1.2, b 0.75, positive
IDF, pinned tokenizer)").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS_PATH = Path(__file__).with_name("stopwords_en.txt")
_INSTRUCTION_WORDS_PATH = Path(__file__).with_name("instruction_words_en.txt")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    text = _STOPWORDS_PATH.read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


@lru_cache(maxsize=1)
def _instruction_words() -> frozenset[str]:
    text = _INSTRUCTION_WORDS_PATH.read_text(encoding="utf-8")
    return frozenset(
        line.strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def tokenize(text: str) -> list[str]:
    """Lowercase, unicode-aware tokenize, dropping English stopwords."""
    stopwords = _stopwords()
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if t not in stopwords]


# Extra "question-shape" words that survive ``tokenize``'s stopword list
# (which already drops how/what/do/does/...) but still carry no content of
# their own -- neither for coverage judgment (see
# ``tutor.retrieval.assessment._key_terms``, the original home of this set)
# nor for building a search AND-query (Baseline v15: "how to squarefoot
# garden the RIGHT WAY" -- the corrected AND query "square foot garden
# right way" never matches "Square foot gardening" because that article
# contains neither "right" nor "way"). Shared here so both call sites use
# exactly the same list.
QUESTION_SHAPE_FILLERS = frozenset(
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
        # Assessor v3 (docs/rewrite_probe_measure.md "Assessor v3"): stop-
        # name noise -- common words that happen to be titles in the
        # archive ("Guy Fieri") but never carry topic content of their
        # own in a question like "what's the name of the guy who...".
        "guy",
        "guys",
        "name",
        "names",
    }
)


# Wrapper phrases that signal "this is an instruction to the model", not a
# topic: "tell me", "give me", "show me", "can you explain/describe/tell/
# show/give/list", and a bare "please". Stripped from anywhere in the text
# (not just the lead) since they never carry topic content of their own.
_INSTRUCTION_PHRASE_RE = re.compile(
    r"\b(?:tell|give|show)\s+me\b"
    r"|\bcan\s+you\s+(?:explain|describe|tell|show|give|list)\b"
    r"|\bplease\b",
    re.IGNORECASE,
)
_LEADING_WORD_RE = re.compile(r"^\s*([A-Za-z]+)\b")


def strip_instruction_words(text: str) -> str:
    """Strip instruction/imperative wrapping from ``text`` for query and
    coverage term extraction (see ``instruction_words_en.txt``).

    Two independent things are stripped: (1) known wrapper phrases like
    "tell me" / "can you explain" anywhere in the text, and (2) a leading
    imperative verb from ``instruction_words_en.txt`` (e.g. "Output the
    boiling point of helium...") -- but ONLY when the text still has other
    content left afterwards, so a real topic query like "output device" or
    "What is output?" (where the instruction word is the sole content term,
    or not in leading position) is left untouched. Idempotent; safe to call
    on already-stripped text.
    """
    stripped = _INSTRUCTION_PHRASE_RE.sub(" ", text)
    match = _LEADING_WORD_RE.match(stripped)
    if match and match.group(1).lower() in _instruction_words():
        rest = stripped[match.end() :]
        if tokenize(rest):
            stripped = rest
    return stripped


def rank_terms_by_rarity(term_hit_counts: dict[str, int]) -> list[str]:
    """Order ``term_hit_counts``' keys by rarity (fewest search hits first),
    dropping terms with zero hits entirely (misspelt/OOV terms like
    "farenheit" that never occur in the corpus at all).

    Used by the candidate-generation fallback (see ``research.py``) to
    prefer specific/rare terms (e.g. "helium") over generic/common ones
    (e.g. "output", "point") when the all-terms query returns nothing and
    candidates must be built from a relaxed, coordination-ranked query.
    Ties keep first-seen (insertion) order for determinism.
    """
    nonzero = [(term, count) for term, count in term_hit_counts.items() if count > 0]
    order = {term: i for i, term in enumerate(term_hit_counts)}
    nonzero.sort(key=lambda tc: (tc[1], order[tc[0]]))
    return [term for term, _ in nonzero]


_SIBILANT_ES_SUFFIXES = ("ses", "xes", "zes", "ches", "shes")


def singularize(token: str) -> str:
    """Minimal, deterministic plural -> singular normalisation.

    No stemmer dependency (donor-code constraint) -- just the handful of
    simple English plural shapes retrieval coverage matching needs to
    tolerate ("moon"/"moons"): trailing "-ies" -> "-y", trailing "-es"
    after a sibilant ("boxes" -> "box", "churches" -> "church"), and a
    bare trailing "-s" otherwise (but never for a double-"s" ending like
    "class", nor for short tokens where stripping would be unsafe/noisy).
    Already-singular input is returned unchanged; the function is
    idempotent.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(_SIBILANT_ES_SUFFIXES):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


# Superlative/comparative size/speed/age modifiers ("biggest", "fastest",
# "longest", ...) are common English adjectives reused across unrelated
# archive articles (a reality show titled "The Biggest Loser", a movie
# "The Longest Ride", a motorsport "Fastest lap" stat) -- a bare match on
# the modifier word alone, with no real connection to the question's
# actual topic, must never by itself certify a passage as covering the
# question, nor be treated as a required/topic term for search or
# coverage. These are MODIFIERS, not topic terms: the topic is the head
# noun ("animal", "bird", "molecule", "river"). Moved here (retrieval v16,
# docs/retrieval_baseline.md "v16") from ``tutor.retrieval.assessment``
# (Assessor v4) so both the assessor and the retrieval pipeline
# (``tutor.retrieval.research``) share exactly one definition without an
# import cycle (``assessment`` already imports this module; ``research``
# already imports ``assessment``). See ``is_superlative_word`` and
# ``question_modifier_terms`` below.
SUPERLATIVE_WORDS = frozenset(
    {
        "biggest",
        "bigger",
        "largest",
        "larger",
        "smallest",
        "smaller",
        "tallest",
        "taller",
        "longest",
        "longer",
        "shortest",
        "shorter",
        "fastest",
        "faster",
        "slowest",
        "slower",
        "oldest",
        "older",
        "youngest",
        "younger",
        "heaviest",
        "heavier",
        "lightest",
        "lighter",
        "hottest",
        "hotter",
        "coldest",
        "colder",
        "highest",
        "higher",
        "lowest",
        "lower",
        "deepest",
        "deeper",
        "widest",
        "wider",
        "narrowest",
        "narrower",
        "strongest",
        "stronger",
        "loudest",
        "louder",
        "brightest",
        "brighter",
        "most",
        "least",
        "best",
        "worst",
        "worse",
        "better",
        "first",
        "last",
        "record",
    }
)

# Common English words that end in "-est" but are NOT a superlative
# adjective -- the conservative regex rule below must never treat these as
# modifiers.
NON_SUPERLATIVE_EST_WORDS = frozenset(
    {
        "forest",
        "interest",
        "interested",
        "interests",
        "harvest",
        "request",
        "requests",
        "honest",
        "modest",
        "protest",
        "protests",
        "manifest",
        "suggest",
        "suggests",
        "digest",
        "contest",
        "contests",
        "invest",
        "invests",
        "quest",
        "quests",
        "wrest",
        "chest",
        "guest",
        "guests",
        "nest",
        "nests",
        "pest",
        "pests",
        "rest",
        "test",
        "tests",
        "vest",
        "vests",
        "west",
        "zest",
        "arrest",
        "arrests",
        "crest",
        "crests",
    }
)


def is_superlative_word(word: str) -> bool:
    """True if ``word`` (already lowercased, or not -- lowercased here) is a
    superlative/comparative size/speed/age modifier -- see
    ``SUPERLATIVE_WORDS`` above for why these must never count as a topic
    term on their own."""
    w = word.lower()
    if w in SUPERLATIVE_WORDS:
        return True
    # Suffix check (not just exact membership): a compound noun like
    # "rainforest" ends in the exact letters of the non-superlative word
    # "forest" without being an exact match to it, and must be excluded
    # the same way.
    if any(w.endswith(exc) for exc in NON_SUPERLATIVE_EST_WORDS):
        return False
    # Conservative regular "-est" rule: at least 6 letters (rules out
    # "best"/"rest"-length false positives not already in the exceptions
    # list) and not one of the curated non-superlative exceptions above.
    return len(w) >= 6 and w.endswith("est")


# "how long/tall/old/far/big/fast/heavy/deep/high/many/much" -- a measure
# word is only a MODIFIER (not a topic term) when it directly follows
# "how" in the raw question text: "How long is DNA?" (modifier "long") vs.
# "Tell me about the long jump." ("long" is real question content there,
# not a bare measure-question modifier). Retrieval v16.
_HOW_MEASURE_WORDS = (
    "long",
    "tall",
    "old",
    "far",
    "big",
    "fast",
    "heavy",
    "deep",
    "high",
    "many",
    "much",
)
_HOW_MEASURE_RE = re.compile(
    r"\bhow\s+(" + "|".join(_HOW_MEASURE_WORDS) + r")\b", re.IGNORECASE
)


def question_modifier_terms(question: str) -> frozenset[str]:
    """The set of MODIFIER words (lowercased) present in ``question``: any
    bare superlative/comparative word (see ``is_superlative_word``) among
    ``question``'s tokens, plus any "how X" measure word (see
    ``_HOW_MEASURE_WORDS``) that directly follows "how" in the raw text.
    Shared by both ``tutor.retrieval.assessment`` and
    ``tutor.retrieval.research`` (retrieval v16) so a modifier is defined
    identically everywhere it matters: never a required/topic term for
    search, title-suggest, title boost, or the coverage gate, and never
    enough on its own to certify a passage as covering the question.
    Empty for a question with no modifier at all (e.g. "What is the
    capital of France?"), which keeps every non-modifier question's
    retrieval byte-identical to before this function existed.
    """
    mods = {t for t in tokenize(question) if is_superlative_word(t)}
    mods |= {m.lower() for m in _HOW_MEASURE_RE.findall(question)}
    return frozenset(mods)


class BM25:
    """Hand-rolled BM25 over pre-tokenized documents (positive IDF)."""

    def __init__(self, docs: list[list[str]], *, k1: float = 1.2, b: float = 0.75) -> None:
        self._docs = docs
        self._k1 = k1
        self._b = b
        self._doc_lens = [len(d) for d in docs]
        self._avgdl = (sum(self._doc_lens) / len(docs)) if docs else 0.0
        self._doc_freqs: list[Counter[str]] = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        self._df = df
        self._n = len(docs)

    def _idf(self, term: str) -> float:
        n_q = self._df.get(term, 0)
        return max(0.0, math.log(1 + (self._n - n_q + 0.5) / (n_q + 0.5)))

    def scores(self, query_terms: list[str]) -> list[float]:
        if not self._docs:
            return []
        idfs = {term: self._idf(term) for term in set(query_terms)}
        results: list[float] = []
        for i, doc_len in enumerate(self._doc_lens):
            total = 0.0
            for term in query_terms:
                idf = idfs.get(term, 0.0)
                if idf == 0.0:
                    continue
                freq = self._doc_freqs[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self._k1 * (1 - self._b + self._b * doc_len / self._avgdl)
                total += idf * (freq * (self._k1 + 1)) / denom
            results.append(total)
        return results
