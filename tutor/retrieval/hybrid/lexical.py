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
