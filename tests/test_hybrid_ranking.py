"""WP-B8: ranking flags are no-ops unless enabled, and each moves ordering
in its documented direction when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

from tutor.retrieval.hybrid.ranking import (
    RankingFlags,
    apply_heading_affinity,
    apply_lead_augmentation,
    apply_mention_penalty,
    apply_title_boost,
)


@dataclass(frozen=True)
class _P:
    passage_id: str
    title: str = ""
    text: str = ""
    heading_path: tuple[str, ...] = ()
    start: int = 0


def test_ranking_flags_default_off():
    flags = RankingFlags()
    assert flags == RankingFlags(False, False, False, False)


def test_title_boost_off_is_identity():
    passages = [_P("a", title="Cats"), _P("b", title="Photosynthesis")]
    assert apply_title_boost(passages, ["photosynthesis"], enabled=False) == passages


def test_title_boost_on_promotes_title_match():
    passages = [_P("a", title="Cats"), _P("b", title="Photosynthesis")]
    out = apply_title_boost(passages, ["photosynthesis"], enabled=True)
    assert [p.passage_id for p in out] == ["b", "a"]


def test_mention_penalty_off_is_identity():
    passages = [_P("a", text="photosynthesis " * 20), _P("b", text="photosynthesis once")]
    assert apply_mention_penalty(passages, ["photosynthesis"], enabled=False) == passages


def test_mention_penalty_on_demotes_overmentioned():
    over = _P("a", text="photosynthesis " * 20)
    normal = _P("b", text="photosynthesis is how plants make food")
    out = apply_mention_penalty([over, normal], ["photosynthesis"], enabled=True)
    assert [p.passage_id for p in out] == ["b", "a"]


def test_heading_affinity_off_is_identity():
    passages = [_P("a", heading_path=("History",)), _P("b", heading_path=("Photosynthesis",))]
    assert apply_heading_affinity(passages, ["photosynthesis"], enabled=False) == passages


def test_heading_affinity_on_promotes_heading_match():
    passages = [_P("a", heading_path=("History",)), _P("b", heading_path=("Photosynthesis",))]
    out = apply_heading_affinity(passages, ["photosynthesis"], enabled=True)
    assert [p.passage_id for p in out] == ["b", "a"]


def test_lead_augmentation_off_is_identity():
    passages = [_P("a", start=40), _P("b", start=0)]
    assert apply_lead_augmentation(passages, enabled=False) == passages


def test_lead_augmentation_on_promotes_lead_passage():
    passages = [_P("a", start=40), _P("b", start=0)]
    out = apply_lead_augmentation(passages, enabled=True)
    assert [p.passage_id for p in out] == ["b", "a"]


def test_ranking_functions_accept_dicts():
    passages = [
        {"passage_id": "a", "title": "Cats"},
        {"passage_id": "b", "title": "Photosynthesis"},
    ]
    out = apply_title_boost(passages, ["photosynthesis"], enabled=True)
    assert [p["passage_id"] for p in out] == ["b", "a"]
