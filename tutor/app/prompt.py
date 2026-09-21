"""Chronological prompt log and token budget for the offline tutor.

Layout and eviction follow docs/plan/offline_tutor_spec_v0.3.md §8, but the
concrete token table is the 32K profile from
docs/plan/offline_tutor_implementation_plan.md §0.2, originally system 800
/ history 22000 / newest 2500 / generation 2000 / margin 5468, which
supersedes the older 8K/16K tables in spec §8.2 per that plan's Deltas
section.

Owner-approved 2026-09-21 (see docs/plan/spec_v0.4_amendments.md item 8):
the ``system`` slot is raised from 800 to 2000 tokens to fit the
assembled system prompt's worked examples (search-writing, no-specifics,
and body-topics rules, read once per lesson and cached). ``history`` and
``margin`` are rescaled to keep the same ~80/20 split of what remains
after the fixed ``system``/``newest``/``generation`` overheads (the same
computation ``Budget.scaled`` performs), so the 32K profile still sums to
``ceiling`` exactly: system 2000 / history 21038 / newest 2500 /
generation 2000 / margin 5230.

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


_POINTER_WORD_COUNT = 8

_ALL_HELD_INSTRUCTION = (
    "[Host note: every source above was already shown earlier in this "
    "lesson. Answer the student's specific question briefly using the "
    "sources already shown above; do not re-summarise the topic.]"
)


def render_evidence_with_reuse(
    passages: Iterable[dict],
    held_ids: Iterable[str],
    reuse_prior_passages: bool,
    *,
    citation_reminder: str | None = None,
) -> tuple[str, list[str], list[dict], str | None]:
    """Shared pointer-substitution + rendering logic behind
    ``app.reuse_prior_passages`` (docs/passage_reuse.md), used by both
    ``PromptLog.append_evidence`` (the pre-search/model-initiated research
    path) and the forced query-rewrite round
    (``tutor.app.agent_loop._run_forced_rewrite_round``), so both paths
    honour the setting identically instead of the forced round bypassing
    held-id bookkeeping entirely as it used to.

    Returns ``(content, newly_seen_ids, display_passages, note)``:

    - ``content`` is ``display_passages`` rendered as ``[S#] text`` lines
      (pointer text for already-held passages when
      ``reuse_prior_passages`` is set), with the all-held host note
      appended when every passage given was already in ``held_ids``, and
      ``citation_reminder`` appended last when given and something was
      actually rendered.
    - ``newly_seen_ids`` are the ids from ``passages`` not already in
      ``held_ids`` -- the caller is responsible for recording these as
      held afterwards (``PromptLog.append_evidence`` does so on its own
      ``_seen_ids``; the forced-rewrite round calls
      ``PromptLog.mark_held``).
    - ``display_passages`` is what was actually shown this call (full
      text or pointer), for callers that also need to store it (e.g. as
      a ``PromptLog`` entry's ``"passages"``).

    When ``reuse_prior_passages`` is false, this reproduces the original
    pre-reuse behaviour byte-for-byte: an already-held passage is dropped
    outright (no pointer, no note), matching ``app.reuse_prior_passages =
    false``'s documented "old bytes" guarantee.
    """
    passages = list(passages)
    held = frozenset(held_ids)
    if not reuse_prior_passages:
        display_passages = [p for p in passages if p["id"] not in held]
        newly_seen_ids = [p["id"] for p in display_passages]
        note = None
    else:
        display_passages = []
        newly_seen_ids = []
        held_count = 0
        for p in passages:
            if p["id"] in held:
                held_count += 1
                display_passages.append({**p, "text": _pointer_text(p)})
            else:
                display_passages.append(dict(p))
                newly_seen_ids.append(p["id"])
        all_held = bool(passages) and held_count == len(passages)
        note = _ALL_HELD_INSTRUCTION if (all_held and display_passages) else None

    content = "\n".join(f"[{p['label']}] {p['text']}" for p in display_passages)
    if note:
        content = f"{content}\n{note}" if content else note
    if citation_reminder and display_passages:
        content = f"{content}\n{citation_reminder}" if content else citation_reminder
    return content, newly_seen_ids, display_passages, note


def _pointer_text(passage: dict) -> str:
    """One-line pointer for a passage already fully pasted earlier in the
    log: cites where it came from without repeating its text."""
    words = (passage.get("text") or "").split()
    snippet = " ".join(words[:_POINTER_WORD_COUNT])
    ellipsis = "…" if len(words) > _POINTER_WORD_COUNT else ""
    title = passage.get("title") or ""
    return f"(already shown above) {title} — {snippet}{ellipsis}".strip()


@dataclass(frozen=True)
class EvictionEvent:
    kind: str
    evicted_turns: int
    tokens_before: int
    tokens_after: int
    dropped_uncited_passages: int = 0
    dropped_uncited_tokens: int = 0


@dataclass(frozen=True)
class Budget:
    """Token budget for one request, matching the 32K profile by default."""

    ceiling: int = 32768
    system: int = 2000
    history: int = 21038
    newest: int = 2500
    generation: int = 2000
    margin: int = 5230

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
        self._seed: list[dict] = []
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

    def append_seed(self, entries: Iterable[dict]) -> None:
        """Seed the lesson with a fixed synthetic exchange, appended once,
        before any real turn. Distinct from ``_turns``: ``evict`` never
        looks at ``self._seed``, so it is never evicted and never
        reordered, and since it is set exactly once it is trivially
        byte-identical across every subsequent turn of the lesson (part of
        the append-only prefix, per the module docstring's byte-prefix
        property). Each entry should carry ``"seed": True`` so callers
        (the UI transcript builder, turn persistence) can recognize and
        skip these messages -- they are never a real student turn."""
        if self._turns:
            raise ValueError("seed must be appended before any real turn")
        if self._seed:
            raise ValueError("seed already appended")
        self._seed = [dict(e) for e in entries]

    def append_user(self, text: str) -> None:
        self._turns.append([{"role": "user", "content": text}])

    def held_ids(self) -> frozenset[str]:
        """Passage ids whose full text is currently present somewhere in
        this lesson's prompt log -- i.e. not (yet) evicted. Survives
        resume (``_rebuild_prompt_log`` repopulates ``_seen_ids`` from the
        stored entries) and respects eviction (``evict`` removes an id
        from this set once its text is no longer actually present, so a
        later turn re-pastes it in full instead of pointing at text that
        is no longer there)."""
        return frozenset(self._seen_ids)

    def mark_held(self, ids: Iterable[str]) -> None:
        """Record ``ids`` as held (their full text is now present
        somewhere in the log) without going through ``append_evidence``.
        Used by the forced-rewrite round
        (``tutor.app.agent_loop._run_forced_rewrite_round``), which
        appends its own tool-result text via ``append_tool_result``
        (a plain string, keyed by ``tool_call_id``) rather than
        ``append_evidence``'s passages-shaped message, but must still
        update the same held-id bookkeeping so a later turn's pointer
        substitution (via ``held_ids()``) knows these passages' full text
        is already in the log."""
        for pid in ids:
            self._seen_ids.add(pid)

    def append_evidence(
        self,
        passages: Iterable[dict],
        budget: Budget | None = None,
        reuse_prior_passages: bool = False,
    ) -> None:
        if not self._turns:
            raise RuntimeError("append_evidence requires an open turn (append_user first)")
        current_turn = self._turns[-1]

        content, newly_seen_ids, display_passages, note = render_evidence_with_reuse(
            passages, self._seen_ids, reuse_prior_passages
        )

        if budget is not None:
            user_text = current_turn[0].get("content", "") if current_turn else ""
            combined = f"{user_text}\n{content}"
            if self.count_tokens(combined) > budget.newest:
                raise PromptOverflow(
                    f"newest turn exceeds budget.newest={budget.newest} tokens"
                )

        entry: dict = {
            "role": "tool",
            "passages": [dict(p) for p in display_passages],
        }
        if note:
            entry["note"] = note
        current_turn.append(entry)
        for pid in newly_seen_ids:
            self._seen_ids.add(pid)

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

    def append_assistant_tool_calls(self, tool_calls: Iterable[dict]) -> None:
        """Append an assistant message carrying tool calls, in the OpenAI
        wire shape (``{"id","type":"function","function":{"name",
        "arguments"}}``). Used to replay a tool-calling turn verbatim on
        the next model call, per docs/M3_report.md's tool_calls replay
        shape fix."""
        if not self._turns:
            raise RuntimeError(
                "append_assistant_tool_calls requires an open turn (append_user first)"
            )
        self._turns[-1].append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [dict(tc) for tc in tool_calls],
            }
        )

    def append_tool_result(self, *, tool_call_id: str, content: str) -> None:
        """Append a tool-result message for a specific tool call id (role
        "tool", ``tool_call_id`` set) -- distinct from ``append_evidence``'s
        passages-shaped tool message."""
        if not self._turns:
            raise RuntimeError("append_tool_result requires an open turn (append_user first)")
        self._turns[-1].append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    # -- render / accounting -------------------------------------------

    def render(self) -> list[dict]:
        messages: list[dict] = []
        if self._system is not None:
            messages.append(dict(self._system))
        for e in self._seed:
            messages.append(dict(e))
        for p in self._protected:
            messages.append({"role": "tool", "passages": [dict(p)]})
        for turn in self._turns:
            for entry in turn:
                messages.append(dict(entry))
        return messages

    @staticmethod
    def _text_of(message: dict) -> str:
        if message.get("passages"):
            text = "\n".join(
                f"[{p['label']}] {p['text']}" for p in message["passages"]
            )
            if message.get("note"):
                text = f"{text}\n{message['note']}" if text else message["note"]
            return text
        if message.get("tool_calls"):
            return json.dumps(message["tool_calls"], sort_keys=True)
        return message.get("content", "") or ""

    def tokens_used(self) -> int:
        return sum(self.count_tokens(self._text_of(m)) for m in self.render())

    def headroom(self, budget: Budget) -> int:
        return budget.ceiling - self.tokens_used()

    # -- validation / repair ----------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` unless the rendered message list is a valid,
        replayable OpenAI message list: a system message (if any) is first,
        and every assistant ``tool_calls`` entry is immediately followed by
        exactly its own tool-result entries (one per call id, no fewer, no
        more, no interleaving). See review pass 2, finding 1: a tool-call
        exception or a mid-turn failure that leaves a dangling assistant
        ``tool_calls`` entry with no matching tool result produces a message
        list that OpenAI-compatible ``/v1/chat/completions`` endpoints (like
        llama-server) reject with a 400, permanently wedging the session."""
        messages = self.render()
        n = len(messages)
        i = 0
        if i < n and messages[i].get("role") == "system":
            i += 1
        for j in range(i, n):
            if messages[j].get("role") == "system":
                raise ValueError(
                    f"system message must be first; found a second one at index {j}"
                )
        while i < n:
            message = messages[i]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                expected_ids = {tc["id"] for tc in message["tool_calls"]}
                i += 1
                seen_ids: set[str] = set()
                unlabeled_count = 0
                # A tool result normally carries the ``tool_call_id`` it
                # answers (``append_tool_result``), but a research
                # follow-up's evidence packet (``append_evidence``) is a
                # plain ``{"role": "tool", "passages": [...]}`` message with
                # no id of its own -- it still counts as satisfying one of
                # this turn's call ids, just not a *named* one.
                while i < n and messages[i].get("role") == "tool":
                    call_id = messages[i].get("tool_call_id")
                    if call_id is not None:
                        seen_ids.add(call_id)
                    else:
                        unlabeled_count += 1
                    i += 1
                if not seen_ids.issubset(expected_ids):
                    raise ValueError(
                        "assistant tool_calls entry "
                        f"{sorted(expected_ids)} matched by unexpected tool "
                        f"result id(s) {sorted(seen_ids - expected_ids)}"
                    )
                if len(seen_ids) + unlabeled_count != len(expected_ids):
                    raise ValueError(
                        "assistant tool_calls entry "
                        f"{sorted(expected_ids)} not matched exactly by its "
                        f"tool results (found {len(seen_ids)} id-matched + "
                        f"{unlabeled_count} unlabeled)"
                    )
                continue
            i += 1

    def is_valid(self) -> bool:
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def repair(self) -> bool:
        """If the log ends with a dangling assistant ``tool_calls`` entry
        missing some of its tool results (e.g. a turn interrupted by a
        crash before every dispatched call got a result, then persisted in
        that broken shape -- see ``LessonStore.resume``), append a
        synthetic error tool result for each missing call id so the log
        renders a valid message list again. Returns True if a repair was
        made, False if the log was already valid (or empty)."""
        if self.is_valid():
            return False
        if not self._turns:
            return False
        last_turn = self._turns[-1]
        for j in range(len(last_turn) - 1, -1, -1):
            entry = last_turn[j]
            if entry.get("role") == "assistant" and entry.get("tool_calls"):
                expected_ids = {tc["id"] for tc in entry["tool_calls"]}
                trailing = last_turn[j + 1 :]
                seen_ids = {
                    e["tool_call_id"]
                    for e in trailing
                    if e.get("role") == "tool" and e.get("tool_call_id") is not None
                }
                unlabeled_count = sum(
                    1
                    for e in trailing
                    if e.get("role") == "tool" and e.get("tool_call_id") is None
                )
                unmatched = sorted(expected_ids - seen_ids)
                # Unlabeled tool messages (evidence packets) already cover
                # that many of the unmatched ids; only fill the rest.
                still_missing = unmatched[unlabeled_count:]
                for missing_id in still_missing:
                    self.append_tool_result(
                        tool_call_id=missing_id,
                        content=json.dumps(
                            {
                                "ok": False,
                                "error": "turn interrupted before this tool "
                                "call completed",
                            }
                        ),
                    )
                return True
        return False

    # -- eviction --------------------------------------------------------

    def _cited_labels(self) -> set[str]:
        labels: set[str] = set()
        for turn in self._turns:
            for entry in turn:
                if entry.get("role") == "assistant":
                    labels.update(entry.get("cited_labels") or [])
        return labels

    def evict(self, budget: Budget) -> EvictionEvent | None:
        """Two-stage head eviction (spec §8, extended per the project
        owner's approved change: uncited evidence is cheaper to lose than
        whole turns, so it goes first).

        Stage 1: within the *oldest* turns first, drop the passage TEXT of
        evidence entries never cited by any retained assistant message,
        replacing the tool-result content with a short stub. This keeps
        the message list valid (every assistant ``tool_calls`` entry is
        still followed by its tool result) but is no longer "retained
        evidence": the passage id is forgotten so a later re-retrieval
        re-sends it in full (under a new label -- the old label stays
        unresolved).

        Stage 2 (only if stage 1 was not enough): evict whole oldest
        turns, as before. Both stages together are logged as a single
        ``EvictionEvent`` -- still exactly one head edit, breaking the
        byte-prefix property once."""
        operating_ceiling = budget.system + budget.history
        tokens_before = self.tokens_used()
        if tokens_before <= operating_ceiling:
            return None

        # Decided once, up front: a label a retained assistant message
        # cites right now is never dropped in stage 1, regardless of the
        # order stage 1 happens to visit entries in.
        needed_labels = self._cited_labels()

        dropped_uncited_passages = 0
        dropped_uncited_tokens = 0

        for turn in self._turns:
            if self.tokens_used() <= operating_ceiling:
                break
            for idx, entry in enumerate(turn):
                if self.tokens_used() <= operating_ceiling:
                    break
                if entry.get("role") != "tool" or not entry.get("passages"):
                    continue
                passages = entry["passages"]
                cited = [p for p in passages if p.get("label") in needed_labels]
                uncited = [p for p in passages if p.get("label") not in needed_labels]
                if not uncited:
                    continue
                for p in uncited:
                    dropped_uncited_passages += 1
                    dropped_uncited_tokens += self.count_tokens(p.get("text", "") or "")
                    self._seen_ids.discard(p["id"])
                if cited:
                    turn[idx] = {**entry, "passages": [dict(p) for p in cited]}
                else:
                    turn[idx] = {
                        "role": "tool",
                        "content": (
                            f"[evidence dropped to save space: {len(uncited)} "
                            "passages, never cited]"
                        ),
                    }

        removed_entries: list[dict] = []
        evicted_turns = 0
        if self.tokens_used() > operating_ceiling:
            while self.tokens_used() > operating_ceiling and len(self._turns) > 1:
                removed = self._turns.pop(0)
                removed_entries.extend(removed)
                evicted_turns += 1

        if evicted_turns == 0 and dropped_uncited_passages == 0:
            return None

        needed_labels = self._cited_labels()
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

        # ``_seen_ids`` must reflect exactly which passage ids' full text
        # is still actually present after this eviction (stage 1's
        # discards above already did this for individually-dropped
        # passages; a whole evicted turn -- stage 2 -- otherwise left its
        # ids stranded in ``_seen_ids`` even though the text is gone
        # unless reprefilled into ``_protected`` above). Recomputing from
        # the truth is what ``held_ids()`` relies on for passage-reuse
        # pointers to only ever point at text that is genuinely still
        # there.
        self._seen_ids = {p["id"] for p in self._protected}
        for turn in self._turns:
            for entry in turn:
                if entry.get("role") == "tool":
                    for p in entry.get("passages", []) or []:
                        self._seen_ids.add(p["id"])

        tokens_after = self.tokens_used()
        return EvictionEvent(
            kind="eviction_reprefill",
            evicted_turns=evicted_turns,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            dropped_uncited_passages=dropped_uncited_passages,
            dropped_uncited_tokens=dropped_uncited_tokens,
        )
