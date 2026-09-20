"""RED tests for tutor.app.profiles.ProfileStore (WP-C4, M5 gate).

Spec sources:
- docs/plan/offline_tutor_spec_v0.3.md line ~36: "Student profiles
  selectable between sessions." §12 UI: "Student selector | Between
  sessions only; loads profile and last lesson summary".
- docs/plan/offline_tutor_implementation_plan.md WP-C4 / M5 row: "Profiles,
  lesson state, append-only prompt with eviction under the 32K profile,
  status page, 30-minute lesson within resource limits".
- The 32K profile budget table (plan §0.2): "System, tool schemas,
  profile | 800" tokens total for that whole slot -- the spec does not
  give profile-only sub-budget, so this test asserts the profile summary
  alone stays well under the full 800-token slot (contract decision made
  here, not spelled out verbatim by the spec).
- Spec does not enumerate exact profile fields; §12 mentions "student
  selector ... loads profile"; this test derives a minimal field set
  (display_name, grade_level, subjects, reading_level/preferences) per
  the task brief and stores nothing beyond that (no PII beyond a display
  name, consistent with "learner state kept small and explicit", §11).

SQLite must be WAL + check_same_thread=False handling per repo-wide
resource-discipline rules.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from tutor.app.profiles import ProfileError, ProfileStore


def _fake_counter(text: str) -> int:
    # Deterministic, cheap stand-in for the runtime tokenizer: ~1 token
    # per 4 characters, rounded up. Good enough to assert a bound.
    return max(1, (len(text) + 3) // 4)


class TestProfileStoreBasics:
    def test_create_and_get_roundtrip(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Ada",
            grade_level=7,
            subjects=["math", "history"],
            reading_level="grade7",
        )
        got = store.get(profile_id)
        assert got.display_name == "Ada"
        assert got.grade_level == 7
        assert got.subjects == ["math", "history"]
        assert got.reading_level == "grade7"

    def test_unicode_display_name_roundtrips(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Zoë 李雷",
            grade_level=5,
            subjects=["science"],
            reading_level="grade5",
        )
        got = store.get(profile_id)
        assert got.display_name == "Zoë 李雷"

    def test_list_returns_all_created_profiles(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        id1 = store.create(
            display_name="A", grade_level=3, subjects=["math"], reading_level="grade3"
        )
        id2 = store.create(
            display_name="B", grade_level=4, subjects=["math"], reading_level="grade4"
        )
        listed_ids = {p.id for p in store.list()}
        assert listed_ids == {id1, id2}

    def test_update_persists_changes(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Ada", grade_level=7, subjects=["math"], reading_level="grade7"
        )
        store.update(profile_id, subjects=["math", "art"], reading_level="grade8")
        got = store.get(profile_id)
        assert got.subjects == ["math", "art"]
        assert got.reading_level == "grade8"

    def test_delete_removes_profile(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Ada", grade_level=7, subjects=["math"], reading_level="grade7"
        )
        store.delete(profile_id)
        assert store.get(profile_id) is None

    def test_get_unknown_profile_returns_none(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        assert store.get("does-not-exist") is None

    def test_persists_across_reopen(self, tmp_path):
        db_path = tmp_path / "profiles.sqlite3"
        store1 = ProfileStore(db_path)
        profile_id = store1.create(
            display_name="Ada", grade_level=7, subjects=["math"], reading_level="grade7"
        )
        del store1

        store2 = ProfileStore(db_path)
        got = store2.get(profile_id)
        assert got is not None
        assert got.display_name == "Ada"


class TestProfileValidation:
    def test_rejects_empty_display_name(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        with pytest.raises(ProfileError):
            store.create(display_name="", grade_level=7, subjects=["math"], reading_level="grade7")

    def test_rejects_negative_grade_level(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        with pytest.raises(ProfileError):
            store.create(
                display_name="Ada", grade_level=-1, subjects=["math"], reading_level="grade7"
            )

    def test_rejects_non_list_subjects(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        with pytest.raises(ProfileError):
            store.create(display_name="Ada", grade_level=7, subjects="math", reading_level="grade7")

    def test_rejects_unknown_sensitive_fields(self, tmp_path):
        # The spec's "learner state kept small and explicit" (§11) and the
        # profile selector described in §12 give no basis for storing
        # anything beyond name/grade/subjects/reading-level/preferences;
        # ProfileStore.create must reject arbitrary extra kwargs rather
        # than silently persisting unspecified (possibly sensitive) data.
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        with pytest.raises(TypeError):
            store.create(
                display_name="Ada",
                grade_level=7,
                subjects=["math"],
                reading_level="grade7",
                home_address="123 Main St",
            )


class TestProfilePromptSummary:
    def test_prompt_summary_is_bounded_by_injected_counter(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Ada",
            grade_level=7,
            subjects=["math", "history", "science", "art", "music"],
            reading_level="grade7",
            preferences="likes worked examples, dislikes long reading",
        )
        profile = store.get(profile_id)
        summary = profile.prompt_summary()
        assert isinstance(summary, str)
        # Plan §0.2: the whole "system, tool schemas, profile" slot is 800
        # tokens for the 32K profile. The profile summary is only one part
        # of that slot; bound it well under the full slot.
        assert _fake_counter(summary) <= 200

    def test_prompt_summary_mentions_display_name_and_subjects(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Ada", grade_level=7, subjects=["math"], reading_level="grade7"
        )
        summary = store.get(profile_id).prompt_summary()
        assert "Ada" in summary
        assert "math" in summary

    def test_prompt_summary_excludes_unstated_fields(self, tmp_path):
        # No field beyond display_name/grade_level/subjects/reading_level/
        # preferences is ever stored, so the summary cannot leak anything
        # else either.
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        profile_id = store.create(
            display_name="Ada", grade_level=7, subjects=["math"], reading_level="grade7"
        )
        profile = store.get(profile_id)
        allowed = {"id", "display_name", "grade_level", "subjects", "reading_level", "preferences"}
        if hasattr(profile, "__dataclass_fields__"):
            stored_fields = set(profile.__dataclass_fields__)
        else:
            stored_fields = set(vars(profile).keys())
        assert stored_fields <= allowed


class TestProfileStoreSqliteDiscipline:
    def test_database_is_wal_mode(self, tmp_path):
        db_path = tmp_path / "profiles.sqlite3"
        ProfileStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()

    def test_usable_from_multiple_threads(self, tmp_path):
        # check_same_thread=False handling: the store must not raise
        # sqlite3.ProgrammingError when used from a worker thread other
        # than the one that constructed it.
        store = ProfileStore(tmp_path / "profiles.sqlite3")
        errors = []

        def worker():
            try:
                pid = store.create(
                    display_name="Worker", grade_level=6, subjects=["math"], reading_level="grade6"
                )
                store.get(pid)
            except Exception as exc:  # noqa: BLE001 - capture for assertion
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert errors == []
