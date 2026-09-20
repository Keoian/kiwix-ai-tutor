"""Archive registry: loads ``[[archive]]`` tables from a TOML config file.

Pure tutor code — nothing here is vendored from openzim-mcp.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from tutor.retrieval.zim.archive import ArchiveStatus, fingerprint, validate_archive

_VALID_TIERS = {1, 2, 3}
_VALID_KINDS = {"encyclopedia", "textbook", "qa", "howto", "viewer_only"}
_VALID_STORAGE = {"ssd", "hdd"}


class RegistryError(Exception):
    """Raised for malformed or invalid archive registry configuration."""


@dataclass(frozen=True)
class ArchiveEntry:
    id: str
    path: Path
    tier: int
    kind: str
    subjects: tuple[str, ...]
    storage: str
    weight: float = 1.0
    searchable: bool = True


@dataclass(frozen=True)
class Registry:
    archives: tuple[ArchiveEntry, ...]

    def get(self, archive_id: str) -> ArchiveEntry:
        for entry in self.archives:
            if entry.id == archive_id:
                return entry
        raise RegistryError(f"Unknown archive id: {archive_id}")

    def for_subject(self, subject: str | None) -> list[ArchiveEntry]:
        """Tier 1 archives, then tier 2, then tier 3 archives matching ``subject``.

        Non-searchable archives are excluded entirely.
        """
        result: list[ArchiveEntry] = []
        for tier in (1, 2):
            result.extend(
                e for e in self.archives if e.tier == tier and e.searchable
            )
        if subject is not None:
            result.extend(
                e
                for e in self.archives
                if e.tier == 3 and e.searchable and subject in e.subjects
            )
        return result

    def validate_all(self) -> dict[str, ArchiveStatus]:
        return {entry.id: validate_archive(entry.path) for entry in self.archives}

    def fingerprint_digest(self, archive_id: str) -> str:
        """The current on-disk fingerprint digest of ``archive_id``'s archive.

        Lets callers outside ``tutor.retrieval`` (e.g. ``tutor.app.compose``,
        which never imports ``tutor.retrieval.zim`` directly) check a dense
        sidecar's manifest against the archive's *current* fingerprint
        without reaching into the ZIM layer themselves.
        """
        return fingerprint(self.get(archive_id).path).digest


def _require(table: dict, key: str, toml_path: Path) -> object:
    if key not in table:
        raise RegistryError(f"{toml_path}: archive entry missing required key {key!r}")
    return table[key]


def load_registry(toml_path: Path) -> Registry:
    """Load and validate a registry from ``toml_path``.

    Relative ``path`` values are resolved against ``toml_path``'s directory.
    Never checks whether the referenced ZIM files exist; use
    ``Registry.validate_all`` for that.
    """
    toml_path = Path(toml_path)
    try:
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RegistryError(f"Cannot load {toml_path}: {exc}") from exc

    base_dir = toml_path.resolve().parent
    raw_entries = data.get("archive", [])
    entries: list[ArchiveEntry] = []
    seen_ids: set[str] = set()

    for raw in raw_entries:
        archive_id = _require(raw, "id", toml_path)
        if archive_id in seen_ids:
            raise RegistryError(f"{toml_path}: duplicate archive id {archive_id!r}")
        seen_ids.add(archive_id)

        raw_path = _require(raw, "path", toml_path)
        tier = _require(raw, "tier", toml_path)
        kind = _require(raw, "kind", toml_path)
        subjects = _require(raw, "subjects", toml_path)
        storage = _require(raw, "storage", toml_path)

        if tier not in _VALID_TIERS:
            raise RegistryError(f"{toml_path}: archive {archive_id!r} has invalid tier {tier!r}")
        if kind not in _VALID_KINDS:
            raise RegistryError(f"{toml_path}: archive {archive_id!r} has invalid kind {kind!r}")
        if storage not in _VALID_STORAGE:
            raise RegistryError(
                f"{toml_path}: archive {archive_id!r} has invalid storage {storage!r}"
            )

        resolved_path = (base_dir / raw_path).resolve()

        entries.append(
            ArchiveEntry(
                id=archive_id,
                path=resolved_path,
                tier=tier,
                kind=kind,
                subjects=tuple(subjects),
                storage=storage,
                weight=float(raw.get("weight", 1.0)),
                searchable=bool(raw.get("searchable", True)),
            )
        )

    return Registry(archives=tuple(entries))
