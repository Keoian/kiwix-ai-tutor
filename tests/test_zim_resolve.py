"""RED tests for ``tutor.retrieval.zim.resolve`` (WP-B1 / WP-B3 groundwork).

Ported semantics from openzim-mcp's ``zim/redirects.py`` (see
docs/donor_inventory.md (c)): strict redirect-chain resolution with cycle
and over-depth detection. libzim's own writer refuses to keep redirect
loops/dangling redirects in a built archive (see tests/zim_fixtures.py
module docstring), so cycle/over-depth behavior here is exercised against
fake archive/entry objects rather than a built fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.zim.resolve import (
    EntryNotFound,
    RedirectCycle,
    RedirectTooDeep,
    ResolvedEntry,
    resolve_entry,
)

# --------------------------------------------------------------------------
# Against the real fixture ZIM
# --------------------------------------------------------------------------


def test_resolve_direct_article_zero_hops(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    resolved = resolve_entry(archive, "pythagorean_theorem")
    assert isinstance(resolved, ResolvedEntry)
    assert resolved.requested_path == "pythagorean_theorem"
    assert resolved.final_path == "pythagorean_theorem"
    assert resolved.hops == ()


def test_resolve_simple_redirect_one_hop(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    resolved = resolve_entry(archive, "redirect_a")
    assert resolved.requested_path == "redirect_a"
    assert resolved.final_path == "pythagorean_theorem"
    assert len(resolved.hops) == 1
    assert resolved.hops[0] == "redirect_a"


def test_resolve_chain_two_hops(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    resolved = resolve_entry(archive, "redirect_b")
    assert resolved.final_path == "pythagorean_theorem"
    assert len(resolved.hops) == 2
    assert resolved.hops[0] == "redirect_b"
    assert resolved.hops[1] == "redirect_a"


def test_resolve_missing_path_raises_entry_not_found(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    with pytest.raises(EntryNotFound):
        resolve_entry(archive, "this_path_does_not_exist")


# --------------------------------------------------------------------------
# Fake archive/entry objects for cycle / over-depth cases libzim itself
# refuses to persist (see module docstring).
# --------------------------------------------------------------------------


class _FakeEntry:
    def __init__(self, path: str, archive: _FakeArchive) -> None:
        self.path = path
        self._archive = archive

    @property
    def is_redirect(self) -> bool:
        return self.path in self._archive.redirects

    def get_redirect_entry(self) -> _FakeEntry:
        target = self._archive.redirects[self.path]
        return self._archive.get_entry_by_path(target)


class _FakeArchive:
    """Minimal stand-in exposing has_entry_by_path/get_entry_by_path."""

    def __init__(self, redirects: dict[str, str], terminal_paths: set[str]) -> None:
        self.redirects = redirects
        self._terminal_paths = terminal_paths

    def has_entry_by_path(self, path: str) -> bool:
        return path in self.redirects or path in self._terminal_paths

    def get_entry_by_path(self, path: str) -> _FakeEntry:
        if not self.has_entry_by_path(path):
            raise KeyError(path)
        return _FakeEntry(path, self)


def test_resolve_cycle_raises_redirect_cycle() -> None:
    archive = _FakeArchive(redirects={"a": "b", "b": "a"}, terminal_paths=set())
    with pytest.raises(RedirectCycle):
        resolve_entry(archive, "a")


def test_resolve_chain_longer_than_max_hops_raises_redirect_too_deep() -> None:
    # 10-hop chain, max_hops=8: chain 0->1->2->...->9->"end"
    redirects = {str(i): str(i + 1) for i in range(10)}
    archive = _FakeArchive(redirects=redirects, terminal_paths={"end"})
    redirects["9"] = "end"  # ensure it terminates if hops were unbounded
    with pytest.raises(RedirectTooDeep):
        resolve_entry(archive, "0", max_hops=8)


def test_resolve_chain_within_max_hops_succeeds() -> None:
    redirects = {"0": "1", "1": "2", "2": "final"}
    archive = _FakeArchive(redirects=redirects, terminal_paths={"final"})
    resolved = resolve_entry(archive, "0", max_hops=8)
    assert resolved.final_path == "final"
    assert resolved.hops == ("0", "1", "2")


def test_resolve_fake_missing_path_raises_entry_not_found() -> None:
    archive = _FakeArchive(redirects={}, terminal_paths={"real"})
    with pytest.raises(EntryNotFound):
        resolve_entry(archive, "missing")
