"""Reciprocal Rank Fusion of multiple rankings.

Ours: no donor code. See docs/plan/offline_tutor_kiwix_reuse_plan.md §4.1
("Reciprocal Rank Fusion") and offline_tutor_spec_v0.3.md §7.2 step 3
(RRF ``1/(60+rank)``, all weights 1).
"""

from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse ``rankings`` (each a list of doc ids, best first) via RRF.

    Ties are broken deterministically by first appearance, scanning
    rankings in order and, within a ranking, by rank.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order_counter = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order_counter
                order_counter += 1

    ordered_ids = sorted(scores.keys(), key=lambda d: (-scores[d], first_seen[d]))
    return [(doc_id, scores[doc_id]) for doc_id in ordered_ids]
