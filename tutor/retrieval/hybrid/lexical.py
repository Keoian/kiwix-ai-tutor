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


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    text = _STOPWORDS_PATH.read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def tokenize(text: str) -> list[str]:
    """Lowercase, unicode-aware tokenize, dropping English stopwords."""
    stopwords = _stopwords()
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if t not in stopwords]


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
