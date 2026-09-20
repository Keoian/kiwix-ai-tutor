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

    def __init__(self, count_tokens: Callable[[str], int], *, subject_hint: str | None = None):
        self.log = PromptLog(count_tokens)
        self.subject_hint = subject_hint
        self.history: list[dict] = []
        self.retained_passages: dict[str, dict] = {}
        self._next_label_num = 1

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
