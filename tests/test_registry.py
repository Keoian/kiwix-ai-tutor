"""RED tests for ``tutor.retrieval.registry`` (WP-B2).

Loads ``[[archive]]`` tables from a TOML config into a frozen registry of
archive entries, without requiring the referenced ZIM files to exist on
disk (existence/validity is a separate, explicit ``validate_all`` step).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.registry import (
    _VALID_KINDS,
    _VALID_STORAGE,
    _VALID_TIERS,
    ArchiveEntry,
    Registry,
    RegistryError,
    load_registry,
)
from tutor.retrieval.zim.archive import ArchiveState, ArchiveStatus

REPO_ROOT = Path(__file__).resolve().parents[1]

# Subject tags the retrieval layer's tier-3 ``for_subject`` routing is
# designed to match against a host-supplied ``topic_hint`` (see
# ``tutor/retrieval/research.py``'s ``_route``). ``Registry`` itself does not
# validate ``subjects`` values (they're host-supplied free text at query
# time), so this is a config-side guard against typos: every subject tag used
# in ``config/archives.dev.toml`` must be drawn from this known vocabulary,
# so a teacher's/topic-hint string has a real chance of matching it.
KNOWN_SUBJECTS = frozenset(
    {
        # Present before the 2026-09-22 zim_library_recommendations.md additions.
        "math",
        "physics",
        "chemistry",
        "biology",
        "history",
        "english",
        "linguistics",
        "literature",
        "puzzling",
        "religion",
        # Added 2026-09-22 alongside the new archives.
        "datascience",
        "statistics",
        "cs",
        "medicine",
        "economics",
        "philosophy",
        "psychology",
        "earthscience",
        "engineering",
        "law",
        "politics",
    }
)


def _write_toml(tmp_path: Path, body: str) -> Path:
    toml_path = tmp_path / "archives.toml"
    toml_path.write_text(body, encoding="utf-8")
    return toml_path


# --------------------------------------------------------------------------
# Basic loading
# --------------------------------------------------------------------------


def test_load_registry_basic(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "a1"
path = "a1.zim"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"

[[archive]]
id = "a2"
path = "a2.zim"
tier = 3
kind = "qa"
subjects = ["math"]
storage = "hdd"
""",
    )
    registry = load_registry(toml_path)
    assert isinstance(registry, Registry)
    assert len(registry.archives) == 2
    entry = registry.get("a1")
    assert isinstance(entry, ArchiveEntry)
    assert entry.tier == 1
    assert entry.kind == "encyclopedia"
    assert entry.storage == "ssd"
    assert entry.weight == 1.0
    assert entry.searchable is True


def test_relative_paths_resolve_against_toml_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "cfg"
    subdir.mkdir()
    toml_path = _write_toml(
        subdir,
        """
[[archive]]
id = "a1"
path = "../zims/a1.zim"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"
""",
    )
    registry = load_registry(toml_path)
    entry = registry.get("a1")
    assert entry.path == (subdir / "../zims/a1.zim").resolve()
    assert entry.path.is_absolute()


def test_defaults_for_weight_and_searchable(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "a1"
path = "a1.zim"
tier = 2
kind = "textbook"
subjects = []
storage = "hdd"
weight = 2.5
searchable = false
""",
    )
    registry = load_registry(toml_path)
    entry = registry.get("a1")
    assert entry.weight == 2.5
    assert entry.searchable is False


# --------------------------------------------------------------------------
# for_subject ordering / filtering
# --------------------------------------------------------------------------


def _subject_toml(tmp_path: Path) -> Path:
    return _write_toml(
        tmp_path,
        """
[[archive]]
id = "tier1"
path = "t1.zim"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"

[[archive]]
id = "tier2"
path = "t2.zim"
tier = 2
kind = "encyclopedia"
subjects = []
storage = "hdd"

[[archive]]
id = "tier3_math"
path = "t3m.zim"
tier = 3
kind = "qa"
subjects = ["math"]
storage = "ssd"

[[archive]]
id = "tier3_history"
path = "t3h.zim"
tier = 3
kind = "qa"
subjects = ["history"]
storage = "ssd"

[[archive]]
id = "viewer"
path = "v.zim"
tier = 3
kind = "viewer_only"
subjects = ["math"]
storage = "hdd"
searchable = false
""",
    )


def test_for_subject_orders_tier1_then_tier2_then_matching_tier3(tmp_path: Path) -> None:
    registry = load_registry(_subject_toml(tmp_path))
    result = registry.for_subject("math")
    ids = [e.id for e in result]
    assert ids == ["tier1", "tier2", "tier3_math"]


def test_for_subject_none_excludes_all_tier3(tmp_path: Path) -> None:
    registry = load_registry(_subject_toml(tmp_path))
    result = registry.for_subject(None)
    ids = [e.id for e in result]
    assert ids == ["tier1", "tier2"]


def test_for_subject_excludes_non_searchable(tmp_path: Path) -> None:
    registry = load_registry(_subject_toml(tmp_path))
    result = registry.for_subject("math")
    ids = [e.id for e in result]
    assert "viewer" not in ids


def test_for_subject_non_matching_subject_excludes_tier3(tmp_path: Path) -> None:
    registry = load_registry(_subject_toml(tmp_path))
    result = registry.for_subject("chemistry")
    ids = [e.id for e in result]
    assert ids == ["tier1", "tier2"]


# --------------------------------------------------------------------------
# validate_all
# --------------------------------------------------------------------------


def test_validate_all_never_raises_for_missing_files(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "missing_one"
path = "does_not_exist.zim"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"
""",
    )
    registry = load_registry(toml_path)
    statuses = registry.validate_all()
    assert isinstance(statuses, dict)
    assert isinstance(statuses["missing_one"], ArchiveStatus)
    assert statuses["missing_one"].state is ArchiveState.MISSING


# --------------------------------------------------------------------------
# Error cases
# --------------------------------------------------------------------------


def test_duplicate_id_raises(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "dup"
path = "a.zim"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"

[[archive]]
id = "dup"
path = "b.zim"
tier = 2
kind = "encyclopedia"
subjects = []
storage = "hdd"
""",
    )
    with pytest.raises(RegistryError):
        load_registry(toml_path)


def test_bad_tier_raises(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "a"
path = "a.zim"
tier = 4
kind = "encyclopedia"
subjects = []
storage = "ssd"
""",
    )
    with pytest.raises(RegistryError):
        load_registry(toml_path)


def test_bad_kind_raises(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "a"
path = "a.zim"
tier = 1
kind = "not_a_real_kind"
subjects = []
storage = "ssd"
""",
    )
    with pytest.raises(RegistryError):
        load_registry(toml_path)


def test_bad_storage_raises(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "a"
path = "a.zim"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "cloud"
""",
    )
    with pytest.raises(RegistryError):
        load_registry(toml_path)


def test_missing_key_raises(tmp_path: Path) -> None:
    toml_path = _write_toml(
        tmp_path,
        """
[[archive]]
id = "a"
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"
""",
    )
    with pytest.raises(RegistryError):
        load_registry(toml_path)


# --------------------------------------------------------------------------
# Real dev config
# --------------------------------------------------------------------------


def test_real_dev_archives_toml_loads() -> None:
    toml_path = REPO_ROOT / "config" / "archives.dev.toml"
    registry = load_registry(toml_path)
    ids = {e.id for e in registry.archives}
    assert "simplewiki" in ids
    assert "enwiki" in ids
    assert "lumen" in ids
    assert "math_se" in ids

    simplewiki = registry.get("simplewiki")
    assert simplewiki.tier == 1
    assert simplewiki.storage == "ssd"

    enwiki = registry.get("enwiki")
    assert enwiki.tier == 2
    assert enwiki.storage == "hdd"

    math_se = registry.get("math_se")
    assert math_se.kind == "qa"
    assert "math" in math_se.subjects


def test_real_dev_archives_toml_ids_are_unique() -> None:
    # ``load_registry`` already raises ``RegistryError`` on a duplicate id
    # (see ``test_duplicate_id_raises``), so a successful load of the real
    # config already proves this; asserted explicitly too for a direct,
    # named regression signal on the real file.
    toml_path = REPO_ROOT / "config" / "archives.dev.toml"
    registry = load_registry(toml_path)
    ids = [e.id for e in registry.archives]
    assert len(ids) == len(set(ids))


def test_real_dev_archives_toml_paths_are_unique() -> None:
    toml_path = REPO_ROOT / "config" / "archives.dev.toml"
    registry = load_registry(toml_path)
    paths = [str(e.path) for e in registry.archives]
    assert len(paths) == len(set(paths))


def test_real_dev_archives_toml_entries_have_valid_fields() -> None:
    """Every entry's required fields are present and hold valid values.

    Does NOT check that ``path`` exists on disk -- tests must never depend
    on gitignored ``data/`` or on host-specific drives like ``D:\\Kiwix``;
    that is what ``Registry.validate_all`` is for, exercised elsewhere
    against fixtures, not against the real dev config.
    """
    toml_path = REPO_ROOT / "config" / "archives.dev.toml"
    registry = load_registry(toml_path)
    assert len(registry.archives) > 0
    for entry in registry.archives:
        assert isinstance(entry.id, str) and entry.id
        assert entry.tier in _VALID_TIERS
        assert entry.kind in _VALID_KINDS
        assert entry.storage in _VALID_STORAGE
        assert isinstance(entry.subjects, tuple)
        assert all(isinstance(s, str) and s for s in entry.subjects)
        assert entry.weight > 0
        assert isinstance(entry.searchable, bool)
        assert entry.path.is_absolute()


def test_real_dev_archives_toml_subjects_are_known() -> None:
    toml_path = REPO_ROOT / "config" / "archives.dev.toml"
    registry = load_registry(toml_path)
    unknown: dict[str, tuple[str, ...]] = {}
    for entry in registry.archives:
        bad = tuple(s for s in entry.subjects if s not in KNOWN_SUBJECTS)
        if bad:
            unknown[entry.id] = bad
    assert not unknown, f"archive entries with unrecognized subjects: {unknown}"
