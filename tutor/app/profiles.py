"""Student profile store (WP-C4).

Spec sources: docs/plan/offline_tutor_spec_v0.3.md line ~36 ("Student
profiles selectable between sessions") and §12 ("Student selector |
Between sessions only; loads profile and last lesson summary"). The
field set (display_name, grade_level, subjects, reading_level,
preferences) is a contract decision (see tests/test_profiles.py
docstring): learner state is kept small and explicit (§11), no PII
beyond a display name.

SQLite is opened WAL + ``check_same_thread=False`` with an internal lock,
matching the pattern in ``tutor.retrieval.snapshots.SnapshotStore``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path


class ProfileError(Exception):
    """Raised for invalid profile field values."""


@dataclass(frozen=True)
class Profile:
    id: str
    display_name: str
    grade_level: int
    subjects: list[str]
    reading_level: str
    preferences: str | None = None

    def prompt_summary(self) -> str:
        """A short, bounded natural-language summary for the system
        prompt's profile slot (well under the 800-token system/tools/
        profile budget slot -- see plan §0.2)."""
        subjects = ", ".join(self.subjects)
        parts = [
            f"Student: {self.display_name} (grade {self.grade_level}).",
            f"Subjects: {subjects}.",
            f"Reading level: {self.reading_level}.",
        ]
        if self.preferences:
            parts.append(f"Preferences: {self.preferences}.")
        return " ".join(parts)


class ProfileStore:
    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(str(Path(db_path)), check_same_thread=False)
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    grade_level INTEGER NOT NULL,
                    subjects TEXT NOT NULL,
                    reading_level TEXT NOT NULL,
                    preferences TEXT
                )
                """
            )
            self.connection.commit()

    def create(
        self,
        *,
        display_name: str,
        grade_level: int,
        subjects: list[str],
        reading_level: str,
        preferences: str | None = None,
    ) -> str:
        if not display_name:
            raise ProfileError("display_name must not be empty")
        if grade_level < 0:
            raise ProfileError("grade_level must not be negative")
        if not isinstance(subjects, list):
            raise ProfileError("subjects must be a list")

        profile_id = uuid.uuid4().hex
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO profiles
                    (id, display_name, grade_level, subjects, reading_level, preferences)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    display_name,
                    grade_level,
                    json.dumps(subjects),
                    reading_level,
                    preferences,
                ),
            )
            self.connection.commit()
        return profile_id

    def _row_to_profile(self, row) -> Profile:
        return Profile(
            id=row[0],
            display_name=row[1],
            grade_level=row[2],
            subjects=json.loads(row[3]),
            reading_level=row[4],
            preferences=row[5],
        )

    def get(self, profile_id: str) -> Profile | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT id, display_name, grade_level, subjects, reading_level, preferences
                FROM profiles WHERE id = ?
                """,
                (profile_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_profile(row)

    def list(self) -> list[Profile]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT id, display_name, grade_level, subjects, reading_level, preferences
                FROM profiles
                """
            ).fetchall()
        return [self._row_to_profile(row) for row in rows]

    def update(self, profile_id: str, **fields) -> None:
        allowed = {"display_name", "grade_level", "subjects", "reading_level", "preferences"}
        unknown = set(fields) - allowed
        if unknown:
            raise TypeError(f"unknown profile field(s): {sorted(unknown)}")

        current = self.get(profile_id)
        if current is None:
            raise ProfileError(f"unknown profile: {profile_id}")

        merged = {
            "display_name": fields.get("display_name", current.display_name),
            "grade_level": fields.get("grade_level", current.grade_level),
            "subjects": fields.get("subjects", current.subjects),
            "reading_level": fields.get("reading_level", current.reading_level),
            "preferences": fields.get("preferences", current.preferences),
        }
        if not merged["display_name"]:
            raise ProfileError("display_name must not be empty")
        if merged["grade_level"] < 0:
            raise ProfileError("grade_level must not be negative")
        if not isinstance(merged["subjects"], list):
            raise ProfileError("subjects must be a list")

        with self._lock:
            self.connection.execute(
                """
                UPDATE profiles
                SET display_name = ?, grade_level = ?, subjects = ?,
                    reading_level = ?, preferences = ?
                WHERE id = ?
                """,
                (
                    merged["display_name"],
                    merged["grade_level"],
                    json.dumps(merged["subjects"]),
                    merged["reading_level"],
                    merged["preferences"],
                    profile_id,
                ),
            )
            self.connection.commit()

    def delete(self, profile_id: str) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()
