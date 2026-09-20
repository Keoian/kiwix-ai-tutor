"""RED tests for ``tutor.retrieval.zim.archive`` (WP-B1 / WP-B2).

Vendored/adapted from the header-signature functions in openzim-mcp's
``zim/archive.py`` (see docs/donor_inventory.md (c)/(d)) plus a WP-B2
``validate_archive``/``fingerprint``/``open_archive`` surface written fresh
for the tutor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tutor.retrieval.zim.archive import (
    ArchiveError,
    ArchiveFingerprint,
    ArchiveState,
    ArchiveStatus,
    fingerprint,
    has_zim_signature,
    is_truncated_zim,
    open_archive,
    validate_archive,
)

# --------------------------------------------------------------------------
# has_zim_signature / is_truncated_zim
# --------------------------------------------------------------------------


def test_has_zim_signature_true_for_real_zim(fixture_zim: Path) -> None:
    assert has_zim_signature(fixture_zim) is True


def test_has_zim_signature_false_for_not_a_zim(not_a_zim: Path) -> None:
    assert has_zim_signature(not_a_zim) is False


def test_has_zim_signature_false_for_missing_path(tmp_path: Path) -> None:
    assert has_zim_signature(tmp_path / "does_not_exist.zim") is False


def test_is_truncated_zim_true_for_truncated(truncated_zim: Path) -> None:
    assert is_truncated_zim(truncated_zim) is True


def test_is_truncated_zim_false_for_full_archive(fixture_zim: Path) -> None:
    assert is_truncated_zim(fixture_zim) is False


def test_is_truncated_zim_false_for_not_a_zim(not_a_zim: Path) -> None:
    # A stray text file is not-a-ZIM, not a truncated one; the two must stay
    # distinguishable (donor v3.3.1 field report, see archive.py docstring).
    assert is_truncated_zim(not_a_zim) is False


# --------------------------------------------------------------------------
# validate_archive
# --------------------------------------------------------------------------


def test_validate_archive_valid(fixture_zim: Path) -> None:
    status = validate_archive(fixture_zim)
    assert status.state is ArchiveState.VALID
    assert status.path == fixture_zim
    assert status.has_fulltext_index is True
    assert status.article_count is not None
    assert status.article_count > 40


def test_validate_archive_truncated(truncated_zim: Path) -> None:
    status = validate_archive(truncated_zim)
    assert status.state is ArchiveState.TRUNCATED


def test_validate_archive_not_zim(not_a_zim: Path) -> None:
    status = validate_archive(not_a_zim)
    assert status.state is ArchiveState.NOT_ZIM


def test_validate_archive_missing(tmp_path: Path) -> None:
    status = validate_archive(tmp_path / "nope.zim")
    assert status.state is ArchiveState.MISSING


def test_validate_archive_no_fulltext_index(noindex_zim: Path) -> None:
    status = validate_archive(noindex_zim)
    assert status.state is ArchiveState.NO_FULLTEXT_INDEX
    assert status.has_fulltext_index is False


def test_validate_archive_never_raises(
    tmp_path: Path, not_a_zim: Path, truncated_zim: Path
) -> None:
    for candidate in (tmp_path / "missing.zim", not_a_zim, truncated_zim):
        validate_archive(candidate)  # must not raise


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_is_stable_for_same_file(fixture_zim: Path) -> None:
    fp1 = fingerprint(fixture_zim)
    fp2 = fingerprint(fixture_zim)
    assert isinstance(fp1, ArchiveFingerprint)
    assert fp1.digest == fp2.digest


def test_fingerprint_same_uuid_different_digest_for_copy(fixture_zim: Path, tmp_path: Path) -> None:
    import shutil
    import time

    copy_path = tmp_path / "copy.zim"
    shutil.copyfile(fixture_zim, copy_path)
    # Force a different mtime so the copy's fingerprint differs even though
    # its bytes (and hence UUID) are identical.
    time.sleep(0.01)
    original_fp = fingerprint(fixture_zim)
    copy_fp = fingerprint(copy_path)

    assert original_fp.uuid == copy_fp.uuid
    assert original_fp.digest != copy_fp.digest


def test_fingerprint_digest_independent_of_path_string(fixture_zim: Path, tmp_path: Path) -> None:
    # Renaming/relocating without touching mtime or bytes must not change the
    # part of the digest that identifies content-plus-mtime; only the actual
    # stat (size/mtime) and UUID/edition metadata may vary the digest -
    # never the path string itself.
    import os

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    other_path = other_dir / "renamed_but_same_bytes.zim"
    os.link(fixture_zim, other_path) if hasattr(os, "link") else None
    if not other_path.exists():
        import shutil

        shutil.copyfile(fixture_zim, other_path)
        stat = fixture_zim.stat()
        os.utime(other_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    fp_original = fingerprint(fixture_zim)
    fp_other = fingerprint(other_path)
    assert fp_original.uuid == fp_other.uuid
    # Same size + mtime + uuid + edition metadata -> same digest, regardless
    # of the path strings differing.
    assert fp_original.digest == fp_other.digest


def test_fingerprint_has_edition_metadata(fixture_zim: Path) -> None:
    fp = fingerprint(fixture_zim)
    assert fp.language == "eng"
    assert fp.name == "fixture_en_school"
    assert fp.title
    assert fp.date == "2026-01-01"


# --------------------------------------------------------------------------
# open_archive
# --------------------------------------------------------------------------


def test_open_archive_yields_libzim_archive(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    with open_archive(fixture_zim) as archive:
        assert isinstance(archive, Archive)
        assert archive.has_main_entry


def test_open_archive_usable_when_no_fulltext_index(noindex_zim: Path) -> None:
    # NO_FULLTEXT_INDEX archives are usable, just flagged - open_archive must
    # not raise for them.
    with open_archive(noindex_zim) as archive:
        assert archive.has_main_entry


@pytest.mark.parametrize("bad_fixture_name", ["not_a_zim", "truncated_zim"])
def test_open_archive_raises_archive_error_for_bad_states(
    bad_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    bad_path: Path = request.getfixturevalue(bad_fixture_name)
    with pytest.raises(ArchiveError) as excinfo:
        with open_archive(bad_path):
            pass
    assert isinstance(excinfo.value.status, ArchiveStatus)


def test_open_archive_raises_archive_error_for_missing(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError):
        with open_archive(tmp_path / "missing.zim"):
            pass


# --------------------------------------------------------------------------
# WP-B1 acceptance: import hygiene
# --------------------------------------------------------------------------

_ALLOWED_NON_STDLIB_IMPORTS = {"libzim", "bs4", "soupsieve", "tutor", "typing_extensions"}


def test_archive_module_import_hygiene() -> None:
    script = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import tutor.retrieval.zim.archive\n"
        "after = set(sys.modules)\n"
        "new_top_level = {m.split('.')[0] for m in (after - before)}\n"
        "stdlib = set(sys.stdlib_module_names)\n"
        "non_stdlib_new = new_top_level - stdlib\n"
        "print(sorted(non_stdlib_new))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    non_stdlib_new = set(eval(result.stdout.strip()))
    assert non_stdlib_new.issubset(_ALLOWED_NON_STDLIB_IMPORTS), non_stdlib_new


# --------------------------------------------------------------------------
# Licence header / third-party notices hygiene
# --------------------------------------------------------------------------

_ZIM_PKG_DIR = Path(__file__).resolve().parents[1] / "tutor" / "retrieval" / "zim"
_ADAPTED_MARKER = "Adapted from openzim-mcp"


def test_vendored_files_with_adapted_marker_carry_licence_attribution() -> None:
    py_files = [p for p in _ZIM_PKG_DIR.glob("*.py") if p.name != "__init__.py"]
    assert py_files, "expected vendored modules under tutor/retrieval/zim/"
    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8")
        if _ADAPTED_MARKER in text:
            assert "MIT" in text, f"{py_file} has the adapted marker but no MIT mention"
            assert "Cameron Rye" in text, f"{py_file} has the adapted marker but no author credit"


def test_archive_and_resolve_carry_the_adapted_marker() -> None:
    for name in ("archive.py", "resolve.py"):
        text = (_ZIM_PKG_DIR / name).read_text(encoding="utf-8")
        assert _ADAPTED_MARKER in text, f"{name} must credit openzim-mcp"


def test_third_party_notices_mentions_openzim_mcp_and_libzim() -> None:
    notices = (Path(__file__).resolve().parents[1] / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    assert "openzim-mcp" in notices
    assert "v3.3.4" in notices
    assert "9358db06f205bb0b95cc938c68405851a0e205a8" in notices
    assert "libzim" in notices
    assert "GPL" in notices
