"""Host-side repetition-loop detector for streamed answers.

Defect (docs/citation_experiment.md, "Seed exchange A/B (2026-09-20)"):
three of 108 eval answers were 110K-145K-character repetition loops (the
model repeating the same table row or sentence verbatim thousands of
times) because generation had no output cap and no loop guard. This
module implements the guard: it looks at the *tail* of the answer
produced so far and reports whether it has degenerated into the same
line/sentence repeated consecutively, so the caller (``tutor.app.
agent_loop.run_turn``) can stop generation and trim the answer back to
the first occurrence of the repeated unit.

This never changes model sampling (no ``repeat_penalty`` etc. -- that
would change model behaviour and needs separate measurement, see the
report for WP "Bound generation"). It is purely a host-side read of the
text already produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A "unit" is one line, or (within a line) one sentence, whichever the
# text naturally splits on. Splitting on both newlines and
# sentence-ending punctuation covers both observed failure modes: a
# table repeated row-by-row (each row is its own line) and a sentence
# repeated inline without newlines.
_UNIT_SPLIT_RE = re.compile(r"(\n+|(?<=[.!?])\s+)")

# Repeat the SAME normalised unit at least this many times consecutively
# before calling it a loop. 4 was picked because legitimate structured
# answers (a 3-item bullet list that happens to reuse a template
# sentence, e.g. three "X is a good example of Y." lines) commonly repeat
# a pattern 2-3 times; requiring 4 avoids flagging those while still
# catching the observed loops, which repeated 29x-1534x on the trailing
# span before this guard existed.
DEFAULT_MIN_REPEATS = 4

# Units shorter than this many characters (after normalising whitespace)
# are never counted as a loop, even if repeated many times -- otherwise a
# short, legitimately-repeated list item like "- yes" or "- no" across a
# multi-row true/false table would trip the guard. 20 characters is
# roughly "- yes, it is true." -- long enough that a genuine short list
# item usually falls under it, short enough to still catch the observed
# one-sentence and one-table-row loops (all well over 20 characters).
DEFAULT_MIN_UNIT_CHARS = 20


def _normalise(unit: str) -> str:
    return " ".join(unit.split()).lower()


@dataclass(frozen=True)
class RepetitionLoop:
    """A detected repetition loop.

    ``trimmed_text`` is ``text`` cut back to the end of the first
    occurrence of the repeated unit (i.e. with the redundant repeats
    dropped); ``unit`` is the normalised repeated unit, kept for logging.
    ``repeat_count`` is how many consecutive repeats were found at the
    point of detection.
    """

    trimmed_text: str
    unit: str
    repeat_count: int


def find_repetition_loop(
    text: str,
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    min_unit_chars: int = DEFAULT_MIN_UNIT_CHARS,
) -> RepetitionLoop | None:
    """Return a :class:`RepetitionLoop` if the tail of ``text`` is the
    same normalised line/sentence repeated ``>= min_repeats`` times
    consecutively (ignoring units shorter than ``min_unit_chars`` once
    whitespace-normalised), else ``None``.
    """
    if not text:
        return None

    parts = _UNIT_SPLIT_RE.split(text)
    # parts alternates content, separator, content, separator, ...
    units = parts[0::2]
    normalised = [_normalise(u) for u in units]

    # Walk back from the end counting a consecutive run of the same
    # normalised, long-enough unit.
    last_idx = len(units) - 1
    while last_idx >= 0 and normalised[last_idx] == "":
        last_idx -= 1
    if last_idx < 0:
        return None

    tail_norm = normalised[last_idx]
    if len(tail_norm) < min_unit_chars:
        return None

    run_len = 0
    idx = last_idx
    while idx >= 0 and normalised[idx] == tail_norm:
        run_len += 1
        idx -= 1

    if run_len < min_repeats:
        return None

    first_idx = last_idx - run_len + 1  # index of the first occurrence in the run

    # Rebuild the text through the end of the first occurrence, i.e.
    # units[0..first_idx] plus the separator immediately following
    # units[first_idx] (there always is one here, since a repeat of the
    # same unit follows it).
    unit_pos = 2 * first_idx  # index into `parts` of the unit itself
    trimmed = "".join(parts[: unit_pos + 2])

    return RepetitionLoop(trimmed_text=trimmed, unit=tail_norm, repeat_count=run_len)
