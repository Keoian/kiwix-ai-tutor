"""Citation-rate experiment variants for ``eval/run_turn_eval.py``.

Each variant changes exactly one host-controlled lever (or, for
``combo``, the winning combination): the system prompt text, the
evidence-rendering function, or the answer-turn sampling temperature.
None of these fabricate a citation or parse prose as a tool call; they
only change what the host shows the model and how it is asked to
respond. See docs/citation_experiment.md for the measured results.
"""

from __future__ import annotations

from pathlib import Path

_BASELINE_SYSTEM_TEXT = (
    Path(__file__).parent.parent / "tutor" / "app" / "system_prompt.txt"
).read_text(encoding="utf-8").strip()

_SHORT_SYSTEM_TEXT = (
    "You are an offline tutor. Rules:\n"
    "1. Cite every fact from the evidence with its label, e.g. [S1]. "
    "Never invent a label.\n"
    "2. Use calc before stating arithmetic beyond single digits.\n"
    "3. Teach: ask a guiding question or give steps, don't just answer.\n"
    "4. Say plainly when an example is your own, not from the evidence.\n"
    "5. Keep replies under 800 tokens."
)

_ONE_SHOT_EXAMPLE = (
    "\n\nExample\n"
    "Evidence: [S1] Water boils at 100 degrees Celsius at sea-level "
    "atmospheric pressure.\n"
    'Student: "At what temperature does water boil?"\n'
    "Tutor: \"Water boils at 100 C at sea level [S1]. Why do you think "
    "it might boil at a different temperature on a mountain?\""
)

_EVIDENCE_REMINDER = "\n\nCite the sources you use like [S1]."


def _render_evidence_with_reminder(packet: dict) -> str:
    """Same rendering as ``tutor.app.citations.render_evidence`` (label at
    the start of each passage), plus a one-line reminder appended at the
    end of the block."""
    from tutor.app.citations import render_evidence

    body = render_evidence(packet)
    if not body:
        return body
    return body + _EVIDENCE_REMINDER


VARIANTS: dict[str, dict] = {
    "baseline": {
        "system_text": _BASELINE_SYSTEM_TEXT,
    },
    "a_evidence_reminder": {
        "system_text": _BASELINE_SYSTEM_TEXT,
        "render_evidence_fn": _render_evidence_with_reminder,
    },
    "b_one_shot_example": {
        "system_text": _BASELINE_SYSTEM_TEXT + _ONE_SHOT_EXAMPLE,
    },
    "c_short_prompt": {
        "system_text": _SHORT_SYSTEM_TEXT,
    },
    "d_temperature_0.2": {
        "system_text": _BASELINE_SYSTEM_TEXT,
        "temperature": 0.2,
    },
    "combo": {
        "system_text": _SHORT_SYSTEM_TEXT + _ONE_SHOT_EXAMPLE,
        "render_evidence_fn": _render_evidence_with_reminder,
        "temperature": 0.2,
    },
    "e_reminder_plus_oneshot": {
        "system_text": _BASELINE_SYSTEM_TEXT + _ONE_SHOT_EXAMPLE,
        "render_evidence_fn": _render_evidence_with_reminder,
    },
}
