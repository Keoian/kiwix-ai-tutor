"""RED tests for tutor.retrieval.hybrid.{rrf,diversity,packer}.

Modules do not exist yet; every test here should fail with
ModuleNotFoundError until implemented. See docs/plan/
offline_tutor_kiwix_reuse_plan.md §4.1 ("Reciprocal Rank Fusion"), §4.2
("Budget enforcement": token budget, max passages, max passages/article,
sentence-boundary trimming, deterministic ordering, "truncated/limit hit"
status) and offline_tutor_spec_v0.3.md §7.2 steps 3, 6, 7 (RRF
`1/(60+rank)` all weights 1; "<= 2 passages per article unless coverage
requires more"; greedy packer trimmed at sentence boundaries).

Contract decisions:
- rrf_fuse tie-break is "first appearance across the rankings, scanning
  rankings in order, then within a ranking by rank" since neither doc
  specifies exact tie order beyond "deterministic".
- diversity cap default is 2 per spec §7.2 step 6 ("<= 2 passages per
  article").
- packer's default token estimator: reuse plan doesn't specify a formula;
  spec is silent on the exact divisor. Chosen: ceil(len(text)/3.5)
  (documented approximation for English text at ~3.5 chars/token).
"""

from __future__ import annotations

import math

from tutor.retrieval.hybrid.diversity import cap_per_article
from tutor.retrieval.hybrid.packer import estimate_tokens, pack
from tutor.retrieval.hybrid.rrf import rrf_fuse

# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------


def test_rrf_fuse_basic_scores():
    rankings = [["a", "b", "c"], ["b", "a", "c"]]
    fused = rrf_fuse(rankings, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert set(ids) == {"a", "b", "c"}
    scores = dict(fused)
    expected_a = 1 / (60 + 0) + 1 / (60 + 1)
    expected_b = 1 / (60 + 1) + 1 / (60 + 0)
    assert abs(scores["a"] - expected_a) < 1e-12
    assert abs(scores["b"] - expected_b) < 1e-12
    # a and b tie in total score; b should rank first because it appears
    # at rank 0 in the second ranking and rank 1 in the first, while a is
    # the mirror -- but first-appearance across rankings (scanning ranking
    # 0 first) means "a" (rank 0 in ranking 0) appears before "b".
    assert ids[0] in ("a", "b")


def test_rrf_fuse_deterministic_tie_break_first_appearance():
    rankings = [["x", "y"], ["y", "x"]]
    fused1 = rrf_fuse(rankings, k=60)
    fused2 = rrf_fuse(rankings, k=60)
    assert fused1 == fused2
    # x and y have identical total RRF score (symmetric); tie broken by
    # first appearance scanning ranking 0 first => x before y.
    ids = [d for d, _ in fused1]
    assert ids == ["x", "y"]


def test_rrf_fuse_document_only_in_one_ranking():
    rankings = [["a", "b"], ["c"]]
    fused = rrf_fuse(rankings, k=60)
    scores = dict(fused)
    assert abs(scores["c"] - 1 / (60 + 0)) < 1e-12
    assert abs(scores["a"] - 1 / (60 + 0)) < 1e-12


def test_rrf_fuse_empty_rankings():
    assert rrf_fuse([], k=60) == []


# ---------------------------------------------------------------------------
# diversity.cap_per_article
# ---------------------------------------------------------------------------


def test_cap_per_article_default_two_preserves_order():
    passages = [
        {"article": "A", "id": "a1"},
        {"article": "A", "id": "a2"},
        {"article": "A", "id": "a3"},
        {"article": "B", "id": "b1"},
    ]
    result = cap_per_article(passages, key=lambda p: p["article"])
    ids = [p["id"] for p in result]
    assert ids == ["a1", "a2", "b1"]


def test_cap_per_article_custom_max():
    passages = [
        {"article": "A", "id": "a1"},
        {"article": "A", "id": "a2"},
        {"article": "A", "id": "a3"},
    ]
    result = cap_per_article(passages, key=lambda p: p["article"], max_per_article=1)
    assert [p["id"] for p in result] == ["a1"]


def test_cap_per_article_no_articles_dropped_when_under_cap():
    passages = [{"article": "A", "id": "a1"}, {"article": "B", "id": "b1"}]
    result = cap_per_article(passages, key=lambda p: p["article"])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# packer.estimate_tokens / pack
# ---------------------------------------------------------------------------


def test_estimate_tokens_matches_documented_formula():
    text = "a" * 35
    assert estimate_tokens(text) == math.ceil(35 / 3.5)


def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_pack_never_exceeds_budget():
    passages = [{"text": "word " * 10} for _ in range(20)]
    packed = pack(passages, budget_tokens=50, count_tokens=lambda t: len(t.split()))
    total = sum(count for count in (len(p["text"].split()) for p in packed))
    assert total <= 50


def test_pack_greedy_rank_order_skips_too_large_and_continues():
    passages = [
        {"text": "short one"},          # 2 tokens, fits
        {"text": "x " * 100},           # 100 tokens, too big for remaining budget, skip
        {"text": "short two"},          # 2 tokens, fits after skip
    ]
    packed = pack(passages, budget_tokens=10, count_tokens=lambda t: len(t.split()))
    texts = [p["text"] for p in packed]
    assert "x " * 100 not in texts
    assert any("short one" in t for t in texts)
    assert any("short two" in t for t in texts)


def test_pack_assigns_stable_sequential_labels_in_packed_order():
    passages = [{"text": "one"}, {"text": "two"}, {"text": "three"}]
    packed = pack(passages, budget_tokens=100, count_tokens=lambda t: len(t.split()))
    labels = [p["label"] for p in packed]
    assert labels == ["S1", "S2", "S3"]


def test_pack_empty_passages_returns_empty():
    assert pack([], budget_tokens=100, count_tokens=len) == []


def test_pack_zero_budget_returns_empty():
    passages = [{"text": "anything"}]
    packed = pack(passages, budget_tokens=0, count_tokens=lambda t: len(t.split()))
    assert packed == []
