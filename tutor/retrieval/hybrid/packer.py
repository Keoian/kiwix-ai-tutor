"""Greedy token-budget packer with sequential passage labels.

Ours: no donor code. See offline_tutor_spec_v0.3.md §7.2 step 7 (greedy
packer trimmed at sentence boundaries; here, whole passages are packed
greedily in rank order and any passage that would exceed the remaining
budget is skipped rather than trimmed, since callers already produce
passages sized within ``_MAX_PASSAGE_CHARS``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Documented approximation: ceil(len(text) / 3.5)."""
    if not text:
        return 0
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def pack(
    passages: list[Mapping[str, Any]],
    *,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Greedily pack ``passages`` (already rank-ordered) into ``budget_tokens``.

    A passage whose token count exceeds the *remaining* budget is skipped
    (not trimmed) and packing continues with the next passage. Labels
    ``S1``, ``S2``, ... are assigned sequentially in packed order.
    """
    if budget_tokens <= 0:
        return []
    packed: list[dict[str, Any]] = []
    remaining = budget_tokens
    for passage in passages:
        tokens = count_tokens(passage["text"])
        if tokens > remaining:
            continue
        remaining -= tokens
        entry = dict(passage)
        entry["label"] = f"S{len(packed) + 1}"
        packed.append(entry)
    return packed
