"""Strict redirect-chain resolution for ZIM entries.

Adapted from openzim-mcp
Upstream path: openzim_mcp/zim/redirects.py
Tag: v3.3.4
Commit: 9358db06f205bb0b95cc938c68405851a0e205a8
MIT License
Copyright (c) 2025-2026 Cameron Rye

What was changed: ``resolve_redirect_chain`` was renamed ``resolve_entry``
and adapted to look the starting entry up by path (via
``archive.has_entry_by_path``/``get_entry_by_path``) rather than receiving
an already-resolved entry; it returns a tutor-owned ``ResolvedEntry``
dataclass recording the requested path, final path, and the hop path
sequence instead of raising a single generic archive error, distinguishing
``EntryNotFound``, ``RedirectCycle``, and ``RedirectTooDeep``. The
best-effort variant and the openzim-mcp config/exception coupling were
dropped; the max-hop bound is a plain parameter (default 25) instead of a
config constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_HOPS = 25


class ZimResolveError(Exception):
    """Base class for resolution failures."""


class EntryNotFound(ZimResolveError):
    """Raised when the requested path does not exist in the archive."""


class RedirectCycle(ZimResolveError):
    """Raised when a redirect chain revisits a path it has already seen."""


class RedirectTooDeep(ZimResolveError):
    """Raised when a redirect chain exceeds ``max_hops``."""


@dataclass(frozen=True)
class ResolvedEntry:
    """Result of following ``requested_path`` to its non-redirect target."""

    requested_path: str
    final_path: str
    entry: Any
    hops: tuple[str, ...]


def resolve_entry(archive: Any, path: str, *, max_hops: int = DEFAULT_MAX_HOPS) -> ResolvedEntry:
    """Follow ``path``'s redirect chain in ``archive`` to its canonical target.

    Bounded by ``max_hops`` with seen-path cycle detection, matching the
    donor's ``resolve_redirect_chain`` strict semantics. Raises
    :class:`EntryNotFound` if ``path`` does not exist, :class:`RedirectCycle`
    if a chain revisits a path, and :class:`RedirectTooDeep` if it exceeds
    ``max_hops``.
    """
    if not archive.has_entry_by_path(path):
        raise EntryNotFound(path)

    entry = archive.get_entry_by_path(path)
    hops: list[str] = []
    seen: set[str] = set()
    for _ in range(max_hops):
        if not getattr(entry, "is_redirect", False):
            return ResolvedEntry(
                requested_path=path,
                final_path=entry.path,
                entry=entry,
                hops=tuple(hops),
            )
        if entry.path in seen:
            raise RedirectCycle(entry.path)
        seen.add(entry.path)
        hops.append(entry.path)
        entry = entry.get_redirect_entry()

    if getattr(entry, "is_redirect", False):
        raise RedirectTooDeep(f"Redirect chain too deep (>{max_hops}) starting at {path}")
    return ResolvedEntry(
        requested_path=path,
        final_path=entry.path,
        entry=entry,
        hops=tuple(hops),
    )
