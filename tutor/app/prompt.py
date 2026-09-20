"""Chronological prompt log and token budget for the offline tutor.

Layout and eviction follow docs/plan/offline_tutor_spec_v0.3.md §8, but the
concrete token table is the 32K profile from
docs/plan/offline_tutor_implementation_plan.md §0.2 (system 800 / history
22000 / newest 2500 / generation 2000 / margin 5468), which supersedes the
older 8K/16K tables in spec §8.2 per that plan's Deltas section.

Design notes
------------
``PromptLog`` stores entries in strict chronological append order (a system
message, then a sequence of ``[user, evidence*, assistant]`` turns).
``serialize_messages`` renders a message list to a deterministic byte
string (one sorted-key compact JSON object per line); as long as entries are
only appended, ``serialize_messages(log.render())`` before an append is a
strict byte-prefix of the serialization after it. This "byte-prefix
property" is what lets a caller trust that appending does not require
re-prefilling anything already sent to the model.

Eviction (``PromptLog.evict``) is the sole, explicit exception: it is a
*head* edit that drops the oldest turns and, when necessary, re-inserts
("reprefill") evidence for passages still cited by assistant turns that
survive. Evicting is intentionally allowed to break the byte-prefix
property for that one edit (logged as an ``EvictionEvent`` of kind
``"eviction_reprefill"``); the property resumes holding for all subsequent
appends.

Overflow behavior: if a single newest turn (the latest user question plus
its evidence packet) does not fit within ``Budget.newest`` tokens on its
own, ``append_evidence`` raises :class:`PromptOverflow` rather than
silently trimming or truncating the turn. Silent trimming of the newest,
not-yet-answered turn would risk quietly discarding facts the model is
about to be asked to cite, so overflow is a caller-visible error instead.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


class PromptOverflow(Exception):
    """Raised when the newest turn cannot fit within Budget.newest tokens."""


@dataclass(frozen=True)
class EvictionEvent:
    kind: str
    evicted_turns: int
    tokens_before: int
    tokens_after: int


@dataclass(frozen=True)
class Budget:
    """Token budget for one request, matching the 32K profile by default."""

    ceiling: int = 32768
    system: int = 800
    history: int = 22000
    newest: int = 2500
    generation: int = 2000
    margin: int = 5468

    @classmethod
    def scaled(cls, ceiling: int) -> Budget:
        """Build a budget for a different ceiling, summing exactly to
        ``ceiling``.

        ``system``, ``newest``, and ``generation`` are fixed per-request
        overheads (a system prompt, one incoming turn, one generated
        reply) that do not shrink with the ceiling; only ``history``
        (accumulated past turns) and ``margin`` (safety buffer) scale,
        split in the default profile's ~80/20 ratio. This means a small
        ceiling (e.g. 6K) yields a small ``history`` allowance, which is
        exactly what the eviction unit tests rely on to force eviction
        without needing an enormous transcript.
        """
        default = cls()
        fixed = default.system + default.newest + default.generation
        remaining = max(0, ceiling - fixed)
        history_ratio = default.history / (default.history + default.margin)
        history = int(remaining * history_ratio)
        margin = remaining - history
        return cls(
            ceiling=ceiling,
            system=default.system,
            history=history,
            newest=default.newest,
            generation=default.generation,
            margin=margin,
        )

    @classmethod
    def for_ceiling(cls, ctx_size: int) -> Budget:
        """Alias for :meth:`scaled`, named for building from
        ``cfg.server.ctx_size``."""
        return cls.scaled(ctx_size)


def serialize_messages(messages: Sequence[dict]) -> bytes:
    """Deterministic byte serialization: one sorted-key JSON object per
    line. Appending messages to the end of a list only ever appends bytes
    to the end of this serialization (the "byte-prefix property")."""
    lines = [json.dumps(m, sort_keys=True, separators=(",", ":")) for m in messages]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    return text.encode("utf-8")


class PromptLog:
    """Strict chronological log of a tutoring conversation."""

    def __init__(self, count_tokens: Callable[[str], int]) -> None:
        self.count_tokens = count_tokens
        self._system: dict | None = None
        self._turns: list[list[dict]] = []
        self._protected: list[dict] = []
        self._seen_ids: set[str] = set()

    # -- append API ---------------------------------------------------

    def append_system(self, text: str) -> None:
        if self._system is not None or self._turns:
            raise ValueError(
                "system message must be the first entry and appended only once"
            )
        self._system = {"role": "system", "content": text}

    def append_user(self, text: str) -> None:
        self._turns.append([{"role": "user", "content": text}])

    def append_evidence(
        self,
        passages: Iterable[dict],
        budget: Budget | None = None,
    ) -> None:
        if not self._turns:
            raise RuntimeError("append_evidence requires an open turn (append_user first)")
        current_turn = self._turns[-1]

        new_passages = []
        for p in passages:
            if p["id"] in self._seen_ids:
                continue
            new_passages.append(p)

        content = "\n".join(f"[{p['label']}] {p['text']}" for p in new_passages)

        if budget is not None:
            user_text = current_turn[0].get("content", "") if current_turn else ""
            combined = f"{user_text}\n{content}"
            if self.count_tokens(combined) > budget.newest:
                raise PromptOverflow(
                    f"newest turn exceeds budget.newest={budget.newest} tokens"
                )

        current_turn.append(
            {
                "role": "tool",
                "passages": [dict(p) for p in new_passages],
            }
        )
        for p in new_passages:
            self._seen_ids.add(p["id"])

    def append_calc_result(self, *, expression: str, result: str) -> None:
        if not self._turns:
            raise RuntimeError("append_calc_result requires an open turn (append_user first)")
        self._turns[-1].append(
            {"role": "tool", "content": f"calc({expression}) = {result}"}
        )

    def append_assistant(self, text: str, cited_labels: Iterable[str]) -> None:
        if not self._turns:
            raise RuntimeError("append_assistant requires an open turn (append_user first)")
        self._turns[-1].append(
            {"role": "assistant", "content": text, "cited_labels": list(cited_labels)}
        )

    # -- render / accounting -------------------------------------------

    def render(self) -> list[dict]:
        messages: list[dict] = []
        if self._system is not None:
            messages.append(dict(self._system))
        for p in self._protected:
            messages.append({"role": "tool", "passages": [dict(p)]})
        for turn in self._turns:
            for entry in turn:
                messages.append(dict(entry))
        return messages

    @staticmethod
    def _text_of(message: dict) -> str:
        if message.get("passages"):
            return "\n".join(
                f"[{p['label']}] {p['text']}" for p in message["passages"]
            )
        return message.get("content", "") or ""

    def tokens_used(self) -> int:
        return sum(self.count_tokens(self._text_of(m)) for m in self.render())

    def headroom(self, budget: Budget) -> int:
        return budget.ceiling - self.tokens_used()

    # -- eviction --------------------------------------------------------

    def evict(self, budget: Budget) -> EvictionEvent | None:
        operating_ceiling = budget.system + budget.history
        tokens_before = self.tokens_used()
        if tokens_before <= operating_ceiling:
            return None

        removed_entries: list[dict] = []
        evicted_turns = 0
        while self.tokens_used() > operating_ceiling and len(self._turns) > 1:
            removed = self._turns.pop(0)
            removed_entries.extend(removed)
            evicted_turns += 1

        if evicted_turns == 0:
            return None

        needed_labels: set[str] = set()
        for turn in self._turns:
            for entry in turn:
                if entry.get("role") == "assistant":
                    needed_labels.update(entry.get("cited_labels") or [])

        have_labels = {p["label"] for p in self._protected}
        for turn in self._turns:
            for entry in turn:
                if entry.get("role") == "tool":
                    for p in entry.get("passages", []) or []:
                        have_labels.add(p["label"])

        for entry in removed_entries:
            if entry.get("role") != "tool":
                continue
            for p in entry.get("passages", []) or []:
                if p["label"] in needed_labels and p["label"] not in have_labels:
                    self._protected.append(dict(p))
                    have_labels.add(p["label"])

        tokens_after = self.tokens_used()
        return EvictionEvent(
            kind="eviction_reprefill",
            evicted_turns=evicted_turns,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
