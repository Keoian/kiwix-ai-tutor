"""Shared helper: sentence-attribution counts for eval harnesses.

Wraps ``tutor.app.citations.attribute_sentences`` (host-side sentence-level
attribution, docs/attribution_design.md) so ``eval/run_turn_eval.py`` and
``eval/run_lesson_soak.py`` can both derive the same two metrics from an
answer's text plus the turn's passages, without duplicating the counting
logic:

- ``backed_sentence_rate`` = attributed sentences / (attributed + unbacked
  sentences), computed per turn here; the harnesses micro-average it
  themselves across turns (sum of numerators / sum of denominators, not an
  average of per-turn rates) so a handful of long answers don't get
  outweighed by many short ones.
- ``unbacked_number_rate`` = fraction of turns with >=1 ``unbacked_number``
  span; ``has_unbacked_number`` here is the per-turn boolean the harnesses
  average.

Pure, network-free: ``attribute_sentences`` itself is pure.
"""

from __future__ import annotations

from tutor.app.citations import attribute_sentences


def sentence_attribution_counts(answer_text: str, passages: list[dict]) -> dict:
    """Per-turn attribution counts for ``answer_text`` against ``passages``.

    Returns a dict with:
    - ``attributed_sentences``: count of attributed sentence spans.
    - ``unbacked_sentences``: count of unbacked spans (any reason).
    - ``has_unbacked_number``: True if any unbacked span's reason is
      ``"unbacked_number"``.
    - ``backed_sentence_rate``: ``attributed / (attributed + unbacked)`` for
      *this turn*, or ``None`` if the denominator is zero (e.g. the whole
      answer was short non-claims, or empty).
    """
    result = attribute_sentences(answer_text, passages)
    attributed = len(result.attributions)
    unbacked = len(result.unbacked_spans)
    has_unbacked_number = any(u.reason == "unbacked_number" for u in result.unbacked_spans)
    denom = attributed + unbacked
    return {
        "attributed_sentences": attributed,
        "unbacked_sentences": unbacked,
        "has_unbacked_number": has_unbacked_number,
        "backed_sentence_rate": (attributed / denom) if denom else None,
    }


def sentence_attribution_counts_from_event(event: dict) -> dict:
    """Per-turn attribution counts derived from the server's own
    ``attributions`` SSE event (``tutor/app/compose.py``, added 2026-09-20:
    ``{"attributions": [...], "unbacked": [...]}``, dumped verbatim by the
    host from ``tutor.app.citations.attribute_sentences``), instead of
    recomputing attribution from ``passages`` client-side. Preferred
    whenever the event is available (e.g. a live soak, where ``passages``
    is otherwise empty -- see ``eval/run_lesson_soak.py``'s ``TurnRecord``),
    since it reflects exactly what the host computed for that turn, not a
    client-side re-derivation.

    Same return shape as ``sentence_attribution_counts``.
    """
    attributions = event.get("attributions") or []
    unbacked = event.get("unbacked") or []
    attributed = len(attributions)
    unbacked_count = len(unbacked)
    has_unbacked_number = any(u.get("reason") == "unbacked_number" for u in unbacked)
    denom = attributed + unbacked_count
    return {
        "attributed_sentences": attributed,
        "unbacked_sentences": unbacked_count,
        "has_unbacked_number": has_unbacked_number,
        "backed_sentence_rate": (attributed / denom) if denom else None,
    }


def micro_average_backed_sentence_rate(counts: list[dict]) -> float | None:
    """Micro-average ``backed_sentence_rate`` across turns: sum of
    ``attributed_sentences`` over sum of ``(attributed_sentences +
    unbacked_sentences)``. ``None`` if every turn has a zero denominator
    (or ``counts`` is empty)."""
    total_attributed = sum(c["attributed_sentences"] for c in counts)
    total_denom = total_attributed + sum(c["unbacked_sentences"] for c in counts)
    if total_denom == 0:
        return None
    return total_attributed / total_denom


def unbacked_number_rate(counts: list[dict]) -> float | None:
    """Fraction of turns (``counts``, one per turn) with >=1
    ``unbacked_number`` span. ``None`` if ``counts`` is empty."""
    if not counts:
        return None
    return sum(1 for c in counts if c["has_unbacked_number"]) / len(counts)
