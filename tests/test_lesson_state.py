"""RED tests for tutor.app.lesson_state.LessonStore (WP-C4, M5 gate).

Spec sources:
- docs/plan/offline_tutor_spec_v0.3.md §8.1: the prompt is a "strict
  chronological log that is append-only within a lesson"; system region
  is "fixed for the session". §8.1 eviction: "On a subject change the
  host may start a new lesson (fresh history) instead."
- docs/plan/offline_tutor_implementation_plan.md risk register (line
  ~283): "start a new lesson at a subject change rather than shrinking
  the context globally" -- adopted here as the "new lesson on subject
  change" rule the task brief references.
- Spec §15 logging fields: "route (action / preretrieve /
  preretrieve+followup), calc_calls, eviction_reprefill events,
  prompt-cache hit tokens per request" -- turn records must be able to
  carry these.
- Spec §11: citation snapshots / learner state "archived/deleted with
  their sessions as a unit" -- lessons are the persisted unit here.

SQLite must be WAL + check_same_thread=False handling.
"""

from __future__ import annotations

import sqlite3

import pytest

from tutor.app.lesson_state import LessonStore
from tutor.app.prompt import serialize_messages


def _count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _make_profile_id(tmp_path) -> str:
    from tutor.app.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles.sqlite3")
    return store.create(
        display_name="Ada", grade_level=7, subjects=["math"], reading_level="grade7"
    )


class TestLessonLifecycle:
    def test_start_lesson_returns_id(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        assert isinstance(lesson_id, str)
        assert lesson_id

    def test_list_lessons_for_profile(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        id1 = store.start_lesson(profile_id=profile_id, subject="math")
        id2 = store.start_lesson(profile_id=profile_id, subject="history")
        lessons = store.list_lessons(profile_id=profile_id)
        ids = {lesson.id for lesson in lessons}
        assert ids == {id1, id2}

    def test_end_lesson_marks_ended(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        store.end_lesson(lesson_id)
        lessons = store.list_lessons(profile_id=profile_id)
        [lesson] = [item for item in lessons if item.id == lesson_id]
        assert lesson.ended is True

    def test_new_lesson_required_on_subject_change(self, tmp_path):
        # plan risk register: "start a new lesson at a subject change
        # rather than shrinking the context globally". Appending a turn
        # whose subject differs from the lesson's subject is rejected --
        # the caller (agent loop / routes) must start a new lesson instead
        # of mutating the existing one's subject mid-lesson.
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        with pytest.raises(ValueError):
            store.append_turn(
                lesson_id,
                subject="history",
                user_text="When was the war?",
                route="preretrieve",
                calc_calls=0,
                research_calls=1,
                citation_passage_ids=[],
                tokens_used=100,
                cached_tokens=0,
            )


class TestTurnAppending:
    def test_append_turn_records_metrics(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        store.append_turn(
            lesson_id,
            subject="math",
            user_text="What is 2+2?",
            route="action",
            calc_calls=1,
            research_calls=0,
            citation_passage_ids=["p1", "p2"],
            tokens_used=250,
            cached_tokens=100,
        )
        turns = store.list_turns(lesson_id)
        assert len(turns) == 1
        turn = turns[0]
        assert turn.route == "action"
        assert turn.calc_calls == 1
        assert turn.research_calls == 0
        assert turn.citation_passage_ids == ["p1", "p2"]
        assert turn.tokens_used == 250
        assert turn.cached_tokens == 100

    def test_append_turn_accepts_action_without_user_text(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        store.append_turn(
            lesson_id,
            subject="math",
            action="hint",
            route="action",
            calc_calls=0,
            research_calls=0,
            citation_passage_ids=[],
            tokens_used=50,
            cached_tokens=0,
        )
        [turn] = store.list_turns(lesson_id)
        assert turn.action == "hint"

    def test_append_turn_records_eviction_events(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        store.append_turn(
            lesson_id,
            subject="math",
            user_text="Explain fractions",
            route="preretrieve",
            calc_calls=0,
            research_calls=1,
            citation_passage_ids=["p1"],
            tokens_used=6000,
            cached_tokens=0,
            eviction_events=[
                {
                    "kind": "eviction_reprefill",
                    "evicted_turns": 2,
                    "tokens_before": 7000,
                    "tokens_after": 5500,
                }
            ],
        )
        [turn] = store.list_turns(lesson_id)
        assert turn.eviction_events == [
            {
                "kind": "eviction_reprefill",
                "evicted_turns": 2,
                "tokens_before": 7000,
                "tokens_after": 5500,
            }
        ]


class TestResumeRebuildsByteIdenticalLog:
    def test_resume_prompt_log_renders_byte_identical(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")

        session = store.new_session(lesson_id, count_tokens=_count_tokens)
        session.log.append_system("You are a tutor.")
        session.log.append_user("What is 2+2?")
        session.log.append_evidence(
            [{"id": "pid-1", "label": "S1", "text": "Two plus two equals four."}]
        )
        session.log.append_assistant("It's 4.", cited_labels=["S1"])

        store.append_turn(
            lesson_id,
            subject="math",
            user_text="What is 2+2?",
            route="action",
            calc_calls=0,
            research_calls=0,
            citation_passage_ids=["pid-1"],
            tokens_used=session.log.tokens_used(),
            cached_tokens=0,
        )
        store.save_session(lesson_id, session)

        original_bytes = serialize_messages(session.log.render())

        resumed_session = store.resume(lesson_id, count_tokens=_count_tokens)
        resumed_bytes = serialize_messages(resumed_session.log.render())

        assert resumed_bytes == original_bytes

    def test_resume_survives_process_restart_reopen(self, tmp_path):
        db_path = tmp_path / "lessons.sqlite3"
        profile_id = _make_profile_id(tmp_path)

        store1 = LessonStore(db_path)
        lesson_id = store1.start_lesson(profile_id=profile_id, subject="math")
        session = store1.new_session(lesson_id, count_tokens=_count_tokens)
        session.log.append_system("You are a tutor.")
        session.log.append_user("Hello")
        session.log.append_assistant("Hi there.", cited_labels=[])
        store1.save_session(lesson_id, session)
        original_bytes = serialize_messages(session.log.render())
        del store1

        store2 = LessonStore(db_path)
        resumed = store2.resume(lesson_id, count_tokens=_count_tokens)
        assert serialize_messages(resumed.log.render()) == original_bytes

    def test_append_only_property_holds_after_resume_and_append(self, tmp_path):
        # The byte-prefix property (prompt.py docstring): appending more
        # after a resume must keep the pre-resume bytes as a strict prefix.
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        session = store.new_session(lesson_id, count_tokens=_count_tokens)
        session.log.append_system("You are a tutor.")
        session.log.append_user("Hello")
        session.log.append_assistant("Hi.", cited_labels=[])
        store.save_session(lesson_id, session)
        before_bytes = serialize_messages(session.log.render())

        resumed = store.resume(lesson_id, count_tokens=_count_tokens)
        resumed.log.append_user("How are you?")
        resumed.log.append_assistant("Doing well.", cited_labels=[])
        after_bytes = serialize_messages(resumed.log.render())

        assert after_bytes.startswith(before_bytes)


class TestPersistTurnAtomic:
    """``persist_turn`` (app-side persistence, used by
    tutor.app.compose._persist_turn) writes the turn row and the prompt
    log snapshot in one transaction."""

    def test_persist_turn_writes_turn_row_and_prompt_log_together(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        session = store.new_session(lesson_id, count_tokens=_count_tokens)
        session.log.append_system("You are a tutor.")
        session.log.append_user("What is 2+2?")
        session.log.append_assistant("It's 4.", cited_labels=[])

        store.persist_turn(
            lesson_id,
            session,
            subject="math",
            route="action",
            calc_calls=0,
            research_calls=0,
            citation_passage_ids=[],
            tokens_used=session.log.tokens_used(),
            cached_tokens=0,
            user_text="What is 2+2?",
        )

        [turn] = store.list_turns(lesson_id)
        assert turn.route == "action"

        resumed = store.resume(lesson_id, count_tokens=_count_tokens)
        assert serialize_messages(resumed.log.render()) == serialize_messages(
            session.log.render()
        )

    def test_persist_turn_rejects_subject_change_without_writing_anything(self, tmp_path):
        profile_id = _make_profile_id(tmp_path)
        store = LessonStore(tmp_path / "lessons.sqlite3")
        lesson_id = store.start_lesson(profile_id=profile_id, subject="math")
        session = store.new_session(lesson_id, count_tokens=_count_tokens)
        session.log.append_system("hi")
        session.log.append_user("hello")

        with pytest.raises(ValueError):
            store.persist_turn(
                lesson_id,
                session,
                subject="history",
                route="action",
                calc_calls=0,
                research_calls=0,
                citation_passage_ids=[],
                tokens_used=0,
                cached_tokens=0,
            )

        assert store.list_turns(lesson_id) == []


class TestLessonStoreSqliteDiscipline:
    def test_database_is_wal_mode(self, tmp_path):
        db_path = tmp_path / "lessons.sqlite3"
        LessonStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()
