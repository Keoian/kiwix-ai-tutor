"""Synthetic seed exchange: an OFF-by-default prompt variant (HANDOFF.md
Task 3, docs/citation_experiment.md) that seeds a new lesson's prompt log
with one short, fixed example turn before any real student turn, so the
model has an in-context, correctly-cited answer to imitate even on a cold
lesson (Granite cited 0.44 cold vs 1.00 in-lesson per the task brief's
hypothesis).

Design constraints (each covered by a test in ``tests/test_seed_exchange.py``):

1. Fits the 2000-token system budget together with the rest of the seed
   cost being accounted for in the overall context budget -- the seed is
   NOT part of the system message; it is its own fixed run of messages
   appended via ``PromptLog.append_seed``, so its tokens are counted by
   ``PromptLog.tokens_used`` like any other message (see that module).
2. Part of the append-only prefix: ``PromptLog.append_seed`` stores it in
   a dedicated ``_seed`` list, set exactly once and never touched by
   ``PromptLog.evict`` -- never evicted, never reordered, byte-identical
   across every turn of the lesson.
3. Uses the reserved label ``tutor.app.citations.RESERVED_SEED_LABEL``
   ("S0"), which the citation resolver refuses unconditionally and which
   ``attribute_sentences`` never attributes to -- real evidence numbering
   (``Session.allocate_label``) starts at S1 and never collides with it.
4. Every entry is tagged ``"seed": True`` so the UI transcript builder and
   turn persistence (``LessonStore.append_turn`` / ``persist_turn``, which
   only ever record real completed turns) can recognize and skip it; it
   is never emitted over the turn-runner's SSE ``emit`` channel either,
   so it never appears in a live transcript.
5. Off by default: ``seed_session`` is only ever called when a variant
   name asks for it (``SEED_EXCHANGE_VARIANT``, wired through
   ``tutor.app.compose.build_deps`` from ``[app] prompt_variant`` in
   config, and through ``eval.system_prompt_variants.VARIANTS`` for the
   eval harness) -- an unseeded lesson's prompt is byte-identical to
   today's.

Pitched at a 10-16 year-old: a plain physics question (sound vs. light)
with a `calc` unit conversion (m/s -> km/h), matching this repo's system
prompt's own instruction to always use `calc` for unit conversions.
"""

from __future__ import annotations

from tutor.app.citations import RESERVED_SEED_LABEL

SEED_EXCHANGE_VARIANT = "seed_exchange_s0"

_SEED_QUESTION = (
    "Why does thunder always come after lightning, and how fast is that "
    "in kilometers per hour if my book says sound travels at 343 meters "
    "per second?"
)

_SEED_EVIDENCE_TEXT = (
    "Sound travels through air as a wave of vibrating molecules bumping "
    "into each other, which takes time and is much slower than light. "
    "Light does not need any molecules to travel through, so it reaches "
    "your eyes almost instantly, while the sound of the thunder arrives "
    "moments later."
)

_SEED_CALC_EXPRESSION = "343 * 3.6"
_SEED_CALC_RESULT = "1234.8"

_SEED_ANSWER = (
    "Thunder comes after lightning because sound has to travel as a wave "
    "through vibrating air molecules, which takes much longer than light "
    f"needs to reach your eyes [{RESERVED_SEED_LABEL}]. Using your book's "
    f"value of 343 meters per second, that works out to about "
    f"{_SEED_CALC_RESULT} kilometers per hour."
)

_SEED_CALC_TOOL_CALL_ID = "seed-calc-1"


def build_seed_entries() -> list[dict]:
    """Build the fixed seed-exchange messages, following the exact wire
    shapes real turns use (``tutor.app.prompt.PromptLog``'s internal
    message shapes, later rendered to the model's chat/tool-call template
    by ``tutor.app.agent_loop._to_wire_messages``/``render_evidence``,
    same as any real turn): a user question, an evidence tool message
    carrying one passage labelled ``[S0]``, an assistant ``calc`` tool
    call, its tool result, and a two-sentence final answer with the
    ``[S0]`` label attached to the specific sentence it supports (the
    first sentence), not trailing the paragraph."""
    return [
        {
            "role": "user",
            "content": _SEED_QUESTION,
            "seed": True,
        },
        {
            "role": "tool",
            "passages": [
                {
                    "id": "seed-s0",
                    "label": RESERVED_SEED_LABEL,
                    "title": "How Sound Travels",
                    "path": "seed://how-sound-travels",
                    "text": _SEED_EVIDENCE_TEXT,
                    "kind": "article",
                }
            ],
            "seed": True,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _SEED_CALC_TOOL_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "calc",
                        "arguments": f'{{"expression": "{_SEED_CALC_EXPRESSION}"}}',
                    },
                }
            ],
            "seed": True,
        },
        {
            "role": "tool",
            "tool_call_id": _SEED_CALC_TOOL_CALL_ID,
            "content": f'{{"ok": true, "result": "{_SEED_CALC_RESULT}"}}',
            "seed": True,
        },
        {
            "role": "assistant",
            "content": _SEED_ANSWER,
            "cited_labels": [RESERVED_SEED_LABEL],
            "seed": True,
        },
    ]


def seed_session(session) -> None:
    """Seed ``session.log`` with the fixed exchange, if it has not been
    seeded already (idempotent no-op on a session whose log already has
    turns or a seed, so it is always safe to call once at lesson-start
    wiring). No-op for a session/log that doesn't support ``append_seed``
    (some fakes in unit tests)."""
    log = getattr(session, "log", None)
    append_seed = getattr(log, "append_seed", None)
    if append_seed is None:
        return
    if getattr(log, "_seed", None) or getattr(log, "_turns", None):
        return
    append_seed(build_seed_entries())
