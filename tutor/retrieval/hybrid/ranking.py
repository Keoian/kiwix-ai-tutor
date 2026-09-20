"""Optional ranking-signal hooks: documented no-ops unless enabled.

Ours: no donor code. Per docs/plan/offline_tutor_implementation_plan.md
§10.8-9, ranking flags exist but default OFF and do nothing unless
explicitly enabled by the caller. Each hook here returns its input
score/ordering unchanged when its flag is off; when a flag is on it
applies the minimal documented adjustment. These are intentionally
small: this project is not a generic ranking framework.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def apply_title_boost(
    passages: Sequence[Any], query_terms: Sequence[str], *, enabled: bool
) -> list[Any]:
    """No-op unless ``enabled``; a minimal title-term-overlap boost otherwise."""
    if not enabled:
        return list(passages)
    query_set = {t.lower() for t in query_terms}

    def boost_key(p: Any) -> tuple[int, int]:
        title_terms = {t.lower() for t in str(getattr(p, "title", "")).split()}
        overlap = len(query_set & title_terms)
        return (-overlap, 0)

    return sorted(passages, key=boost_key)


def apply_mention_penalty(passages: Sequence[Any], *, enabled: bool) -> list[Any]:
    """No-op unless ``enabled`` (hook reserved for future implementation)."""
    return list(passages)


def apply_heading_affinity(passages: Sequence[Any], *, enabled: bool) -> list[Any]:
    """No-op unless ``enabled`` (hook reserved for future implementation)."""
    return list(passages)


def apply_lead_augmentation(passages: Sequence[Any], *, enabled: bool) -> list[Any]:
    """No-op unless ``enabled`` (hook reserved for future implementation)."""
    return list(passages)
