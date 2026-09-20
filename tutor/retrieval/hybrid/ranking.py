"""Optional ranking-signal hooks: documented no-ops unless enabled.

Ours: no donor code. Per docs/plan/offline_tutor_implementation_plan.md
WP-B8 and docs/plan/offline_tutor_kiwix_reuse_plan.md §7.5-7.6 and
"Step 6 - Add ranking refinements one at a time", four ranking flags
exist, default OFF, and do nothing unless explicitly enabled by the
caller. Each hook returns its input ordering unchanged when its flag
is off; when a flag is on it applies the minimal documented
adjustment. These are intentionally small: this project is not a
generic ranking framework.

Flags (see ``RankingFlags``):

- ``title_boost``: promote passages whose title shares terms with the
  query (reuse plan §7.5, navigational/title boost).
- ``mention_penalty``: demote passages that merely mention the query
  terms many times relative to their length -- a proxy for
  list/disambiguation pages that "mention twenty times" without
  substantive treatment (reuse plan line 25, "mention penalties").
- ``heading_affinity``: promote passages whose section heading shares
  terms with the query (reuse plan "heading affinity").
- ``lead_augmentation``: promote each article's lead passage ahead of
  its other passages, so a deep passage that would otherwise be
  ambiguous is preceded by its article's lead context (reuse plan
  §7.6, "attach article lead").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RankingFlags:
    """Independently switchable ranking refinements. All default off."""

    title_boost: bool = False
    mention_penalty: bool = False
    heading_affinity: bool = False
    lead_augmentation: bool = False


def _terms(text: str) -> set[str]:
    return {t.lower() for t in str(text).split()}


def _field(p: Any, name: str, default: Any = "") -> Any:
    """Read ``name`` from a passage, whether it is an object or a mapping."""
    if isinstance(p, dict):
        return p.get(name, default)
    return getattr(p, name, default)


def apply_title_boost(
    passages: Sequence[Any], query_terms: Sequence[str], *, enabled: bool
) -> list[Any]:
    """No-op unless ``enabled``; a minimal title-term-overlap boost otherwise."""
    if not enabled:
        return list(passages)
    query_set = {t.lower() for t in query_terms}

    def boost_key(p: Any) -> tuple[int, int]:
        title_terms = _terms(_field(p, "title"))
        overlap = len(query_set & title_terms)
        return (-overlap, 0)

    return sorted(passages, key=boost_key)


def apply_mention_penalty(
    passages: Sequence[Any],
    query_terms: Sequence[str],
    *,
    enabled: bool,
    threshold: int = 5,
) -> list[Any]:
    """No-op unless ``enabled``.

    Demotes passages whose raw query-term mention count exceeds
    ``threshold`` -- these tend to be list/disambiguation pages that
    mention a term many times without substantive treatment. Passages
    at or below the threshold keep their relative order; passages
    above it are pushed to the end, ordered by ascending mention
    count (least-overpenalized first).
    """
    if not enabled:
        return list(passages)
    query_set = {t.lower() for t in query_terms}

    def mention_count(p: Any) -> int:
        text = str(_field(p, "text", _field(p, "content", "")))
        words = [w.lower() for w in text.split()]
        return sum(1 for w in words if w in query_set)

    def penalty_key(p: Any) -> tuple[int, int]:
        count = mention_count(p)
        over = count > threshold
        return (1 if over else 0, count if over else 0)

    indexed = list(enumerate(passages))
    indexed.sort(key=lambda ip: (*penalty_key(ip[1]), ip[0]))
    return [p for _, p in indexed]


def apply_heading_affinity(
    passages: Sequence[Any], query_terms: Sequence[str], *, enabled: bool
) -> list[Any]:
    """No-op unless ``enabled``; promotes passages whose heading overlaps query terms."""
    if not enabled:
        return list(passages)
    query_set = {t.lower() for t in query_terms}

    def affinity_key(p: Any) -> tuple[int, int]:
        heading = _field(p, "heading", None)
        if heading is None:
            heading_path = _field(p, "heading_path", ())
            heading = " ".join(heading_path) if heading_path else ""
        heading_terms = _terms(heading)
        overlap = len(query_set & heading_terms)
        return (-overlap, 0)

    return sorted(passages, key=affinity_key)


def apply_lead_augmentation(passages: Sequence[Any], *, enabled: bool) -> list[Any]:
    """No-op unless ``enabled``; promotes each article's lead passage first."""
    if not enabled:
        return list(passages)

    def lead_key(p: Any) -> tuple[int, int]:
        is_lead = _field(p, "is_lead", None)
        if is_lead is None:
            is_lead = _field(p, "start", None) == 0
        return (0 if is_lead else 1, 0)

    indexed = list(enumerate(passages))
    indexed.sort(key=lambda ip: (*lead_key(ip[1]), ip[0]))
    return [p for _, p in indexed]
