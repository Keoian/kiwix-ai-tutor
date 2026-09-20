"""RED tests for tutor.retrieval.hybrid.lexical (tokenize + hand-rolled BM25).

Module does not exist yet; every test here should fail with
ModuleNotFoundError until implemented. See docs/plan/
offline_tutor_kiwix_reuse_plan.md section 4 ("passage BM25", "pinned
tokenizer") and offline_tutor_spec_v0.3.md §7.2 step 6:
"BM25 over the request's passage pool (k1 1.2, b 0.75, positive IDF,
pinned tokenizer)".

Contract decisions made here (spec/reuse-plan silent or in conflict):
- tokenize() lowercases, is unicode-aware (str.isalnum per char via regex
  \\w with UNICODE), and drops tokens found in the stopword list loaded
  from tutor/retrieval/hybrid/stopwords_en.txt (this file, created as
  fixture-adjacent contract data, mirrors the eval question file's role).
- BM25 uses "positive IDF" per spec line above: IDF is floored at 0 (never
  negative), which the classic Robertson-Sparck-Jones formula can produce
  for terms in more than half the corpus.
"""

from __future__ import annotations

import math

from tutor.retrieval.hybrid.lexical import BM25, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("The Quick Fox") == ["quick", "fox"]


def test_tokenize_drops_stopwords():
    toks = tokenize("this is the cat and the hat")
    assert "the" not in toks
    assert "is" not in toks
    assert toks == ["cat", "hat"]


def test_tokenize_unicode_aware():
    # Erdős must tokenize as one word, not split on the ő.
    assert tokenize("Erdős number") == ["erdős", "number"]


def test_tokenize_deterministic():
    text = "Repeated repeated words words here here"
    assert tokenize(text) == tokenize(text)


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_bm25_hand_computed_tiny_corpus():
    # Tiny corpus, hand-computed against the standard BM25 formula:
    #   IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))   [positive IDF form]
    #   score(D,q) = IDF(q) * (f(q,D)*(k1+1)) / (f(q,D) + k1*(1 - b + b*|D|/avgdl))
    docs = [
        ["cat", "sat", "mat"],       # doc0
        ["dog", "sat", "log"],       # doc1
        ["cat", "cat", "dog", "run"],  # doc2
    ]
    bm25 = BM25(docs, k1=1.2, b=0.75)
    scores = bm25.scores(["cat"])

    n = 3
    avgdl = (3 + 3 + 4) / 3
    k1, b = 1.2, 0.75

    def idf(n_q: int) -> float:
        return max(0.0, math.log(1 + (n - n_q + 0.5) / (n_q + 0.5)))

    idf_cat = idf(2)  # "cat" appears in doc0 and doc2

    def score(freq: int, doclen: int) -> float:
        if freq == 0:
            return 0.0
        return idf_cat * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doclen / avgdl))

    expected0 = score(1, 3)
    expected1 = score(0, 3)
    expected2 = score(2, 4)

    assert abs(scores[0] - expected0) < 1e-9
    assert abs(scores[1] - expected1) < 1e-9
    assert abs(scores[2] - expected2) < 1e-9


def test_bm25_ties_broken_by_input_order():
    docs = [["a", "b"], ["a", "b"]]
    bm25 = BM25(docs, k1=1.2, b=0.75)
    scores = bm25.scores(["a"])
    assert scores[0] == scores[1]
    ranked = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
    assert ranked == [0, 1]


def test_bm25_deterministic_across_instances():
    docs = [["x", "y", "z"], ["y", "z"], ["x"]]
    s1 = BM25(docs).scores(["x", "y"])
    s2 = BM25(docs).scores(["x", "y"])
    assert s1 == s2


def test_bm25_unknown_query_term_yields_zero_contribution():
    docs = [["cat", "sat"], ["dog", "ran"]]
    bm25 = BM25(docs)
    scores = bm25.scores(["nonexistent"])
    assert scores == [0.0, 0.0]


def test_bm25_empty_corpus_returns_empty_scores():
    bm25 = BM25([])
    assert bm25.scores(["anything"]) == []


# ---------------------------------------------------------------------------
# singularize() -- minimal, deterministic plural-stripping helper used by
# the retrieval coverage gate (research.py) to tolerate "moon"/"moons"
# style mismatches between a question's own content terms and article
# text, without pulling in a stemmer dependency (docs/retrieval_baseline.md
# "Pass-2 fix").
# ---------------------------------------------------------------------------


def test_singularize_strips_simple_plural_s():
    from tutor.retrieval.hybrid.lexical import singularize

    assert singularize("moons") == "moon"
    assert singularize("triangles") == "triangle"


def test_singularize_handles_ies_plural():
    from tutor.retrieval.hybrid.lexical import singularize

    assert singularize("theories") == "theory"


def test_singularize_handles_es_after_sibilant():
    from tutor.retrieval.hybrid.lexical import singularize

    assert singularize("boxes") == "box"
    assert singularize("churches") == "church"


def test_singularize_leaves_already_singular_words_alone():
    from tutor.retrieval.hybrid.lexical import singularize

    assert singularize("moon") == "moon"
    assert singularize("class") == "class"  # double-s, not a plural
    assert singularize("gas") == "gas"  # short word, left alone


def test_singularize_is_deterministic_and_idempotent():
    from tutor.retrieval.hybrid.lexical import singularize

    assert singularize(singularize("moons")) == singularize("moons")
