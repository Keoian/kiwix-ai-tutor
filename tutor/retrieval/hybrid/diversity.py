"""Per-article diversity capping.

Ours: no donor code. See offline_tutor_spec_v0.3.md §7.2 step 6
("<= 2 passages per article unless coverage requires more").
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def cap_per_article[T](
    passages: Sequence[T], *, key: Callable[[T], Any], max_per_article: int = 2
) -> list[T]:
    """Keep at most ``max_per_article`` items per ``key(item)``, preserving order."""
    counts: dict[Any, int] = {}
    result: list[T] = []
    for passage in passages:
        article_key = key(passage)
        count = counts.get(article_key, 0)
        if count >= max_per_article:
            continue
        counts[article_key] = count + 1
        result.append(passage)
    return result
