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

# Minimum number of *full periods* a period-N cycle (N >= 2) must repeat
# before it is called a loop. Measured against three real runaway answers
# (docs/soak_v3_analysis.md, "Three huge turns"): they cycle a fixed set of
# N distinct paraphrase sentences dozens of times, differing turn-to-turn
# only by an incrementing invented citation label ([S12], [S17], ...). 3
# periods keeps this well clear of a legitimate structure (e.g. a 2-3 item
# pattern used twice) while catching the observed cycles early.
DEFAULT_MIN_PERIODS = 3

# Cycle period lengths (in units) to check, smallest first so a shorter,
# more specific period wins over a longer one that would also happen to
# match (e.g. a period-2 cycle also trivially satisfies as "period 4").
_CYCLE_PERIODS = range(2, 9)

_LABEL_RE = re.compile(r"\[S\d+\]")

# Note: an earlier version of this normaliser also collapsed every digit
# run to a placeholder (per the original bug report's wording, "the real
# loops differ only by an incrementing label/number"). Measurement showed
# the three real loops already match after label-stripping alone (their
# cycled sentences are byte-identical apart from the label); blanket digit
# collapsing was dropped because it also erases the one thing that makes a
# legitimate 8-step numbered list ("Step 1: ...", "Step 2: ...") or a
# times-table answer distinct, causing false positives on both.


def _normalise(unit: str) -> str:
    return " ".join(unit.split()).lower()


def _normalise_for_cycle(unit: str) -> str:
    """Normalise a unit for cycle-detection: like ``_normalise`` but also
    strips ``[S123]``-style citation labels, so units that differ only by
    an incrementing invented label (the real observed loops) still compare
    equal. See the module-level note by ``_LABEL_RE`` for why digit runs
    are NOT also blanked. Applied uniformly (also used for the plain
    consecutive-repeat check) since it is a strict widening of
    ``_normalise`` -- it never makes two units that were previously equal
    compare unequal."""
    unit = _LABEL_RE.sub("", unit)
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


def _trim_to_unit(parts: list[str], unit_idx: int) -> str:
    """Rebuild the original text through the end of ``units[unit_idx]``
    (i.e. ``units[0..unit_idx]`` plus the separator immediately following
    it, which always exists here since a repeat follows)."""
    unit_pos = 2 * unit_idx  # index into `parts` of the unit itself
    return "".join(parts[: unit_pos + 2])


def find_repetition_loop(
    text: str,
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    min_unit_chars: int = DEFAULT_MIN_UNIT_CHARS,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> RepetitionLoop | None:
    """Return a :class:`RepetitionLoop` if the tail of ``text`` has
    degenerated into a repeating cycle, else ``None``. Two shapes are
    detected, both over units normalised with labels/citation markers
    stripped and digit runs collapsed (so units that differ only by an
    incrementing ``[S#]`` label or number still compare equal --
    docs/soak_v3_analysis.md, "Three huge turns"):

    1. The same unit repeated consecutively ``>= min_repeats`` times
       (period 1 -- the original, narrower defect).
    2. A period-N cycle (``N`` in 2..8) of near-identical units repeating
       ``>= min_periods`` full periods.

    Units shorter than ``min_unit_chars`` (measured on the *original*,
    unnormalised unit) are never counted as a loop on their own -- this
    keeps a legitimate short list (or a times-table answer, whose rows
    differ only by number and would otherwise collapse to the same
    normalised unit) from tripping the guard.
    """
    if not text:
        return None

    parts = _UNIT_SPLIT_RE.split(text)
    # parts alternates content, separator, content, separator, ...
    units = parts[0::2]
    raw_lens = [len(" ".join(u.split())) for u in units]
    normalised = [_normalise_for_cycle(u) for u in units]

    # Skip trailing units that are empty or too short to judge on their
    # own -- most commonly a mid-sentence truncation where generation
    # stopped (cap hit, guard fired) partway through the next unit. The
    # cycle we care about lives in the units before that fragment.
    last_idx = len(units) - 1
    while last_idx >= 0 and (
        normalised[last_idx] == "" or raw_lens[last_idx] < min_unit_chars
    ):
        last_idx -= 1
    if last_idx < 0:
        return None

    # Shape 1: consecutive repeats of the same unit (period 1).
    tail_norm = normalised[last_idx]
    run_len = 0
    idx = last_idx
    while idx >= 0 and normalised[idx] == tail_norm and raw_lens[idx] >= min_unit_chars:
        run_len += 1
        idx -= 1
    if run_len >= min_repeats:
        first_idx = last_idx - run_len + 1
        return RepetitionLoop(
            trimmed_text=_trim_to_unit(parts, first_idx),
            unit=tail_norm,
            repeat_count=run_len,
        )

    # Shape 2: a period-N cycle of near-identical units, smallest N first.
    # The very tail of the text is often a mid-unit truncation (generation
    # stopped partway through the next cycled unit), which would not
    # exactly match its period-mate -- so if the true last unit doesn't
    # start a match for any period, retry from a few units earlier.
    max_tail_skip = min(3, last_idx)
    for tail_skip in range(0, max_tail_skip + 1):
        end_idx = last_idx - tail_skip
        if end_idx < 0:
            break
        for period in _CYCLE_PERIODS:
            if end_idx - period < 0:
                continue
            matched = 0
            idx = end_idx
            while (
                idx - period >= 0
                and normalised[idx] == normalised[idx - period]
                and raw_lens[idx] >= min_unit_chars
                and raw_lens[idx - period] >= min_unit_chars
            ):
                matched += 1
                idx -= 1
            # `matched` counts pairwise equalities; the cyclic run spans
            # `matched + period` units (the trailing period plus each
            # earlier period it was checked against).
            if matched < period * (min_periods - 1):
                continue
            run_units = matched + period
            first_idx = end_idx - run_units + 1
            # Trim to the end of the first full period of the cycle.
            cycle_end_idx = first_idx + period - 1
            return RepetitionLoop(
                trimmed_text=_trim_to_unit(parts, cycle_end_idx),
                unit="|".join(normalised[first_idx : first_idx + period]),
                repeat_count=run_units // period,
            )

    return None
