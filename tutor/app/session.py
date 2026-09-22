"""Per-lesson tutoring session state.

Holds the append-only :class:`tutor.app.prompt.PromptLog`, evidence
retained so far (for citation resolution across turns), the current
subject hint used to steer retrieval, and stable [S#] label allocation
across the whole lesson (spec §8: labels are numbered and stable for the
lesson, never reused for a different passage within it).
"""

from __future__ import annotations

from collections.abc import Callable

from tutor.app.prompt import PromptLog


class Session:
    """Mutable state for one tutoring lesson (one PromptLog's worth of
    conversation)."""

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        *,
        subject_hint: str | None = None,
        profile_summary: str | None = None,
    ):
        self.log = PromptLog(count_tokens)
        self.subject_hint = subject_hint
        self.profile_summary = profile_summary
        self.history: list[dict] = []
        self.retained_passages: dict[str, dict] = {}
        self._next_label_num = 1
        self.pending_clarify: dict | None = None
        """Set by ``tutor.app.agent_loop`` when a "Did you mean X?" clarify
        question is pending the student's reply (owner request 2026-09-21):
        ``{"word", "candidate", "description", "message", "round"}``.
        ``None`` when no clarify question is outstanding."""

    def allocate_label(self) -> str:
        """Allocate the next stable [S#] label for this lesson."""
        label = f"S{self._next_label_num}"
        self._next_label_num += 1
        return label

    def retain_passages(self, passages: list[dict]) -> None:
        """Remember passages (keyed by label) so citations can be resolved
        later in the lesson even after prompt-log eviction."""
        for passage in passages:
            self.retained_passages[passage["label"]] = passage

    def known_passages(self) -> list[dict]:
        return list(self.retained_passages.values())
