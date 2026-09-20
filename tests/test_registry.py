"""RED tests for ``tutor.retrieval.registry`` (WP-B2).

Loads ``[[archive]]`` tables from a TOML config into a frozen registry of
archive entries, without requiring the referenced ZIM files to exist on
disk (existence/validity is a separate, explicit ``validate_all`` step).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.registry import (
    ArchiveEntry,
    Registry,
    RegistryError,
    load_registry,
)
from tutor.retrieval.zim.archive import ArchiveState, ArchiveStatus

REPO_ROOT = Path(__file__).resolve().parents[1]


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
