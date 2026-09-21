"""Lesson lifecycle and append-only prompt-log persistence (WP-C4).

Spec sources: docs/plan/offline_tutor_spec_v0.3.md §8.1 (the prompt log is
a strict chronological, append-only-within-a-lesson log) and the plan's
risk register ("start a new lesson at a subject change rather than
shrinking the context globally"). §15 logging fields (route, calc_calls,
eviction_reprefill events, prompt-cache hit tokens) are recorded per turn
here so the status/turn-log layer can read them back.

``LessonStore`` persists two things per lesson: the lifecycle/turn-metric
rows (for the resume list and status page) and the exact byte-serializable
prompt log (for byte-identical resume, per ``tutor.app.prompt``'s
byte-prefix property). SQLite WAL + ``check_same_thread=False`` + a lock,
matching ``tutor.retrieval.snapshots.SnapshotStore``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tutor.app.prompt import PromptLog
from tutor.app.session import Session
from tutor.retrieval._sqlite_retry import execute_with_retry as _execute_with_retry


@dataclass(frozen=True)
class Lesson:
    id: str
    profile_id: str
    subject: str
    ended: bool


@dataclass(frozen=True)
class Turn:
    lesson_id: str
    subject: str
    route: str
    calc_calls: int
    research_calls: int
    citation_passage_ids: list[str]
    tokens_used: int
    cached_tokens: int
    user_text: str | None = None
    action: str | None = None
    eviction_events: list[dict] | None = None
    attributions: dict | None = None
    truncated: str | None = None
    """``None``, ``"repetition"``, or ``"max_tokens"`` -- see
    ``tutor.app.agent_loop.TurnResult.truncated``."""


def _rebuild_prompt_log(count_tokens: Callable[[str], int], entries: list[dict]) -> PromptLog:
    """Reconstruct a ``PromptLog`` whose ``render()`` reproduces ``entries``
    exactly, from the flat message list ``PromptLog.render()`` produces."""
    log = PromptLog(count_tokens)
    i = 0
    if entries and entries[0].get("role") == "system":
        log._system = dict(entries[0])
        i = 1

    # Seed exchange: a fixed run of entries marked "seed": True, appended
    # once before any real turn (tutor.app.prompt.PromptLog.append_seed).
    seed_entries: list[dict] = []
    while i < len(entries) and entries[i].get("seed"):
        seed_entries.append(dict(entries[i]))
        i += 1
    log._seed = seed_entries

    # Protected passages: tool messages (one passage each) that precede
    # the first user turn.
    while i < len(entries) and entries[i].get("role") == "tool" and "passages" in entries[i]:
        # Only protected entries sit here (before any turn starts); a
        # turn's own tool/evidence entries always follow a user entry.
        passage = dict(entries[i]["passages"][0])
        log._protected.append(passage)
        log._seen_ids.add(passage["id"])
        i += 1

    turns: list[list[dict]] = []
    current: list[dict] | None = None
    for entry in entries[i:]:
        entry = dict(entry)
        if entry.get("role") == "user":
            current = [entry]
            turns.append(current)
        else:
            if current is None:
                # Defensive: shouldn't happen for a log built only via the
                # public append API, but avoid dropping data silently.
                current = []
                turns.append(current)
            current.append(entry)
        if entry.get("role") == "tool" and entry.get("passages"):
            for passage in entry["passages"]:
                log._seen_ids.add(passage["id"])

    log._turns = turns
    return log


class LessonStore:
    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(str(Path(db_path)), check_same_thread=False)
        with self._lock:
            try:
                _execute_with_retry(self.connection, "PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                _execute_with_retry(self.connection, "PRAGMA journal_mode=DELETE")
            _execute_with_retry(
                self.connection,
                """
                CREATE TABLE IF NOT EXISTS lessons (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    ended INTEGER NOT NULL DEFAULT 0,
                    prompt_log_json TEXT
                )
                """,
            )
            _execute_with_retry(
                self.connection,
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    lesson_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    user_text TEXT,
                    action TEXT,
                    route TEXT NOT NULL,
                    calc_calls INTEGER NOT NULL,
                    research_calls INTEGER NOT NULL,
                    citation_passage_ids TEXT NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL,
                    eviction_events TEXT,
                    attributions_json TEXT,
                    truncated TEXT,
                    seq INTEGER NOT NULL
                )
                """,
            )
            # Older databases created before host-side attribution (2026-09-20)
            # lack this column; add it if missing rather than forcing a
            # migration on every dev/CI machine that already has a
            # lessons.sqlite from before this change.
            existing_cols = {
                row[1] for row in self.connection.execute("PRAGMA table_info(turns)")
            }
            if "attributions_json" not in existing_cols:
                self.connection.execute("ALTER TABLE turns ADD COLUMN attributions_json TEXT")
            # Older databases created before the generation-bounding /
            # repetition-guard fix (2026-09-20) lack this column too;
            # same additive-migration pattern as attributions_json above.
            if "truncated" not in existing_cols:
                self.connection.execute("ALTER TABLE turns ADD COLUMN truncated TEXT")
            self.connection.commit()

    # -- lifecycle -------------------------------------------------------

    def start_lesson(self, *, profile_id: str, subject: str) -> str:
        lesson_id = uuid.uuid4().hex
        with self._lock:
            self.connection.execute(
                "INSERT INTO lessons (id, profile_id, subject, ended) VALUES (?, ?, ?, 0)",
                (lesson_id, profile_id, subject),
            )
            self.connection.commit()
        return lesson_id

    def _get_lesson_row(self, lesson_id: str):
        with self._lock:
            return self.connection.execute(
                "SELECT id, profile_id, subject, ended, prompt_log_json FROM lessons WHERE id = ?",
                (lesson_id,),
            ).fetchone()

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        """Public accessor for a lesson's lifecycle fields (id, profile_id,
        subject, ended), used to look up a lesson's owning profile without
        exposing the raw row (e.g. to fetch the profile's grade level for
        the system prompt -- see ``tutor.app.compose._SessionStore``)."""
        row = self._get_lesson_row(lesson_id)
        if row is None:
            return None
        return Lesson(id=row[0], profile_id=row[1], subject=row[2], ended=bool(row[3]))

    def end_lesson(self, lesson_id: str) -> None:
        with self._lock:
            self.connection.execute("UPDATE lessons SET ended = 1 WHERE id = ?", (lesson_id,))
            self.connection.commit()

    def list_lessons(self, *, profile_id: str) -> list[Lesson]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT id, profile_id, subject, ended FROM lessons WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
        return [
            Lesson(id=row[0], profile_id=row[1], subject=row[2], ended=bool(row[3]))
            for row in rows
        ]

    # -- turns -------------------------------------------------------------

    def append_turn(
        self,
        lesson_id: str,
        *,
        subject: str,
        route: str,
        calc_calls: int,
        research_calls: int,
        citation_passage_ids: list[str],
        tokens_used: int,
        cached_tokens: int,
        user_text: str | None = None,
        action: str | None = None,
        eviction_events: list[dict] | None = None,
        attributions: dict | None = None,
        truncated: str | None = None,
    ) -> None:
        row = self._get_lesson_row(lesson_id)
        if row is None:
            raise ValueError(f"unknown lesson: {lesson_id}")
        lesson_subject = row[2]
        if subject != lesson_subject:
            raise ValueError(
                f"subject change ({lesson_subject!r} -> {subject!r}) requires a new lesson"
            )

        turn_id = uuid.uuid4().hex
        with self._lock:
            seq = self.connection.execute(
                "SELECT COUNT(*) FROM turns WHERE lesson_id = ?", (lesson_id,)
            ).fetchone()[0]
            self.connection.execute(
                """
                INSERT INTO turns
                    (id, lesson_id, subject, user_text, action, route, calc_calls,
                     research_calls, citation_passage_ids, tokens_used, cached_tokens,
                     eviction_events, attributions_json, truncated, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    lesson_id,
                    subject,
                    user_text,
                    action,
                    route,
                    calc_calls,
                    research_calls,
                    json.dumps(citation_passage_ids),
                    tokens_used,
                    cached_tokens,
                    json.dumps(eviction_events) if eviction_events is not None else None,
                    json.dumps(attributions) if attributions is not None else None,
                    truncated,
                    seq,
                ),
            )
            self.connection.commit()

    def persist_turn(
        self,
        lesson_id: str,
        session: Session,
        *,
        subject: str,
        route: str,
        calc_calls: int,
        research_calls: int,
        citation_passage_ids: list[str],
        tokens_used: int,
        cached_tokens: int,
        user_text: str | None = None,
        action: str | None = None,
        eviction_events: list[dict] | None = None,
        attributions: dict | None = None,
        truncated: str | None = None,
    ) -> None:
        """Atomically persist one completed turn: the turn's metric row
        AND the lesson's current prompt-log snapshot (``session.log``), in
        a single SQLite transaction. This is the app-side counterpart to
        ``append_turn`` + ``save_session`` called separately -- doing both
        writes under one commit means a crash mid-persist can never leave
        the turn row and the prompt log out of sync with each other; it
        can only ever lose the whole (not-yet-committed) turn, exactly
        like a crash before persistence started at all.
        """
        row = self._get_lesson_row(lesson_id)
        if row is None:
            raise ValueError(f"unknown lesson: {lesson_id}")
        lesson_subject = row[2]
        if subject != lesson_subject:
            raise ValueError(
                f"subject change ({lesson_subject!r} -> {subject!r}) requires a new lesson"
            )

        turn_id = uuid.uuid4().hex
        entries = session.log.render()
        with self._lock:
            seq = self.connection.execute(
                "SELECT COUNT(*) FROM turns WHERE lesson_id = ?", (lesson_id,)
            ).fetchone()[0]
            self.connection.execute(
                """
                INSERT INTO turns
                    (id, lesson_id, subject, user_text, action, route, calc_calls,
                     research_calls, citation_passage_ids, tokens_used, cached_tokens,
                     eviction_events, attributions_json, truncated, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    lesson_id,
                    subject,
                    user_text,
                    action,
                    route,
                    calc_calls,
                    research_calls,
                    json.dumps(citation_passage_ids),
                    tokens_used,
                    cached_tokens,
                    json.dumps(eviction_events) if eviction_events is not None else None,
                    json.dumps(attributions) if attributions is not None else None,
                    truncated,
                    seq,
                ),
            )
            self.connection.execute(
                "UPDATE lessons SET prompt_log_json = ? WHERE id = ?",
                (json.dumps(entries), lesson_id),
            )
            self.connection.commit()

    def list_turns(self, lesson_id: str) -> list[Turn]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT lesson_id, subject, user_text, action, route, calc_calls,
                       research_calls, citation_passage_ids, tokens_used, cached_tokens,
                       eviction_events, attributions_json, truncated
                FROM turns WHERE lesson_id = ? ORDER BY seq ASC
                """,
                (lesson_id,),
            ).fetchall()
        return [
            Turn(
                lesson_id=row[0],
                subject=row[1],
                user_text=row[2],
                action=row[3],
                route=row[4],
                calc_calls=row[5],
                research_calls=row[6],
                citation_passage_ids=json.loads(row[7]),
                tokens_used=row[8],
                cached_tokens=row[9],
                eviction_events=json.loads(row[10]) if row[10] is not None else None,
                attributions=json.loads(row[11]) if row[11] is not None else None,
                truncated=row[12],
            )
            for row in rows
        ]

    # -- prompt-log persistence / resume ---------------------------------

    def new_session(self, lesson_id: str, *, count_tokens: Callable[[str], int]) -> Session:
        row = self._get_lesson_row(lesson_id)
        if row is None:
            raise ValueError(f"unknown lesson: {lesson_id}")
        return Session(count_tokens, subject_hint=row[2])

    def save_session(self, lesson_id: str, session: Session) -> None:
        entries = session.log.render()
        with self._lock:
            self.connection.execute(
                "UPDATE lessons SET prompt_log_json = ? WHERE id = ?",
                (json.dumps(entries), lesson_id),
            )
            self.connection.commit()

    def resume(self, lesson_id: str, *, count_tokens: Callable[[str], int]) -> Session:
        row = self._get_lesson_row(lesson_id)
        if row is None:
            raise ValueError(f"unknown lesson: {lesson_id}")
        subject = row[2]
        prompt_log_json = row[4]
        entries = json.loads(prompt_log_json) if prompt_log_json else []

        session = Session(count_tokens, subject_hint=subject)
        log = _rebuild_prompt_log(count_tokens, entries)
        # Repair-on-resume: a log persisted mid-turn-interruption (e.g. the
        # process crashed after a tool-call exception, before pass-2 fix #1
        # existed) can be stuck with a dangling assistant tool_calls entry
        # and no matching tool result. Repair it here so an already-broken
        # persisted lesson becomes usable again instead of staying wedged
        # forever (review pass 2, finding 1).
        session.log = log
        if log.repair():
            self.save_session(lesson_id, session)
        return session

    def close(self) -> None:
        self.connection.close()
