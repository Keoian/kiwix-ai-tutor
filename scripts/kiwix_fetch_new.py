#!/usr/bin/env python3
"""Download NEW ZIM files listed in scripts/zim_wanted.txt into D:\\Kiwix.

This is a sibling of kiwix_update_zims.py, NOT a replacement for it and NOT
something that touches its state. There may be another process running
kiwix_update_zims.py against D:\\Kiwix\\_update_state.json /
D:\\Kiwix\\_update_log.txt at the same time this script runs. This script:

  - Uses its OWN state file (D:\\Kiwix\\_additions_state.json) and OWN log
    (D:\\Kiwix\\_additions_log.txt). It never reads or writes the
    "_update_*" files.
  - Never moves or deletes anything in D:\\Kiwix or D:\\Kiwix-2023. Every
    name in zim_wanted.txt is a ZIM series we do not have yet, so there is
    nothing to replace -- we only ever add new files.
  - Downloads strictly sequentially (one curl process at a time), smallest
    file first, so it does not starve the other, already-running download.

Catalog / meta4 logic (OPDS fetch+parse, meta4 sidecar size+sha256
verification, curl invocation) is reused from kiwix_update_zims.py by
import -- see that module's docstring for the quirks it documents (catalog
`length` can be slightly stale; the .meta4 sidecar is authoritative).

Each wanted line is a Kiwix catalog **name** (the OPDS <name> element),
which is a different key than the "<basename>_<YYYY-MM>" filename stem
kiwix_update_zims.py resolves against for already-held files. This script
resolves by exact <name> match instead.

Safe to Ctrl-C and re-run: state is durable, and skip/resume decisions are
re-checked against the real filesystem (existing file + correct size) each
run, not just against the state file.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kiwix_update_zims as base  # noqa: E402  (reuse catalog/meta4/curl logic)

# --------------------------------------------------------------------------
# Configuration -- deliberately separate from kiwix_update_zims.py's files.
# --------------------------------------------------------------------------

LOCAL_DIR = base.LOCAL_DIR  # D:\Kiwix -- same download destination directory
WANTED_PATH = Path(__file__).resolve().parent / "zim_wanted.txt"
STATE_PATH = LOCAL_DIR / "_additions_state.json"
LOG_PATH = LOCAL_DIR / "_additions_log.txt"

FORBIDDEN_PATHS = {
    LOCAL_DIR / "_update_state.json",
    LOCAL_DIR / "_update_log.txt",
}


def _assert_not_forbidden() -> None:
    assert STATE_PATH not in FORBIDDEN_PATHS
    assert LOG_PATH not in FORBIDDEN_PATHS
    assert STATE_PATH != base.STATE_PATH
    assert LOG_PATH != base.LOG_PATH


_assert_not_forbidden()


# --------------------------------------------------------------------------
# Wanted list
# --------------------------------------------------------------------------


def read_wanted(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


# --------------------------------------------------------------------------
# State (own file -- never shared with kiwix_update_zims.py)
# --------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_state(state: dict) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def update_state(state: dict, name: str, **fields) -> None:
    entry = state.setdefault(name, {})
    entry.update(fields)
    entry["updated"] = base.now_iso()
    save_state(state)


# --------------------------------------------------------------------------
# Resolve wanted `name` -> catalog entry (exact <name> match, latest date)
# --------------------------------------------------------------------------


def resolve_by_name(
    name: str, full_cache: "base.FullCatalogCache", log
) -> Optional["base.CatalogEntry"]:
    entries: list[base.CatalogEntry] = []
    try:
        data = base.fetch_catalog(name=name)
        entries = [e for e in base.parse_entries(data) if e.name == name]
    except (urllib.error.URLError, OSError) as ex:
        log(f"  catalog query failed for name={name!r}: {ex}")

    if not entries:
        entries = [e for e in full_cache.get(log) if e.name == name]

    if not entries:
        return None

    entries.sort(key=lambda e: (e.fdate or ""))
    return entries[-1]


# --------------------------------------------------------------------------
# Plan item
# --------------------------------------------------------------------------


@dataclass
class WantedItem:
    name: str
    entry: Optional["base.CatalogEntry"]
    filename: Optional[str]  # <fbase>_<fdate>.zim
    url: Optional[str]  # direct .zim URL (meta4 URL minus suffix)
    meta4_url: Optional[str]
    catalog_length: Optional[int]
    meta_size: Optional[int]
    meta_sha: Optional[str]
    action: str  # to_download | already_held | not_found | resolve_error


def resolve_all(names: list[str], state: dict, log) -> list[WantedItem]:
    full_cache = base.FullCatalogCache()
    items: list[WantedItem] = []
    for name in names:
        log(f"resolving {name!r} ...")
        try:
            entry = resolve_by_name(name, full_cache, log)
        except Exception as ex:  # keep going across names
            log(f"  [{name}] resolve error: {ex!r}")
            update_state(state, name, status="resolve_error", error=repr(ex))
            items.append(WantedItem(name, None, None, None, None, None, None, None,
                                     "resolve_error"))
            continue

        if entry is None or not entry.fbase or not entry.fdate:
            log(f"  [{name}] NOT FOUND in catalog")
            update_state(state, name, status="not_found", error="no catalog entry")
            items.append(WantedItem(name, None, None, None, None, None, None, None,
                                     "not_found"))
            continue

        filename = f"{entry.fbase}_{entry.fdate}.zim"
        meta4_url = entry.href
        direct_url = (
            meta4_url[: -len(".meta4")] if meta4_url.endswith(".meta4") else meta4_url
        )

        meta_size, meta_sha = base.fetch_meta4_info(meta4_url, log) if meta4_url else (None, None)

        final_path = LOCAL_DIR / filename
        expected_size = meta_size or entry.length
        if final_path.exists() and expected_size and final_path.stat().st_size == expected_size:
            log(f"  [{name}] already held at correct size -> {filename}")
            update_state(state, name, status="already_held", filename=filename,
                         size=expected_size, error=None)
            items.append(WantedItem(name, entry, filename, direct_url, meta4_url,
                                     entry.length, meta_size, meta_sha, "already_held"))
            continue

        update_state(state, name, status="resolved", filename=filename,
                     size=expected_size, url=direct_url, error=None)
        items.append(WantedItem(name, entry, filename, direct_url, meta4_url,
                                 entry.length, meta_size, meta_sha, "to_download"))
    return items


# --------------------------------------------------------------------------
# Download one item (sequential curl, size+sha256 verify, atomic rename)
# --------------------------------------------------------------------------


def process_item(item: WantedItem, state: dict, log) -> None:
    name = item.name
    size_estimate = item.meta_size or item.catalog_length

    update_state(state, name, status="downloading", filename=item.filename,
                 url=item.url, size=size_estimate, started=base.now_iso())

    free = base.disk_free_bytes(str(LOCAL_DIR.drive) + "\\")
    needed = (size_estimate or 0) + base.FREE_SPACE_MARGIN_BYTES
    if free < needed:
        log(f"[{name}] SKIP: not enough free space "
            f"(need {base.human_gb(needed)}, have {base.human_gb(free)})")
        update_state(state, name, status="skipped_no_space", error=None)
        return

    part_path = LOCAL_DIR / (item.filename + ".part")
    log(f"[{name}] downloading -> {part_path.name} ({base.human_gb(size_estimate)})")
    ok = base.curl_download(item.url, part_path, log)
    if not ok:
        log(f"[{name}] FAILED after {base.MAX_CURL_ATTEMPTS} curl attempts")
        update_state(state, name, status="error", error="curl failed after max attempts",
                     bytes_downloaded=part_path.stat().st_size if part_path.exists() else 0)
        return

    meta_size, meta_sha = item.meta_size, item.meta_sha
    if meta_size is None and item.meta4_url:
        meta_size, meta_sha = base.fetch_meta4_info(item.meta4_url, log)
    expected_size = meta_size or item.catalog_length
    actual_size = part_path.stat().st_size

    if expected_size is not None and actual_size != expected_size:
        log(f"[{name}] SIZE MISMATCH: expected {expected_size}, got {actual_size}")
        update_state(state, name, status="error",
                     error=f"size mismatch: expected {expected_size} got {actual_size}",
                     bytes_downloaded=actual_size)
        return

    sha_ok = None
    if meta_sha:
        log(f"[{name}] verifying sha256...")
        actual_sha = base.sha256_file(part_path)
        sha_ok = actual_sha == meta_sha
        if not sha_ok:
            log(f"[{name}] SHA256 MISMATCH: expected {meta_sha}, got {actual_sha}")
            update_state(state, name, status="error",
                         error=f"sha256 mismatch: expected {meta_sha} got {actual_sha}",
                         bytes_downloaded=actual_size, sha_ok=False)
            return

    final_path = LOCAL_DIR / item.filename
    os.replace(part_path, final_path)
    log(f"[{name}] verified and renamed -> {final_path.name}"
        + (" (sha256 ok)" if sha_ok else " (size only, no sha256 published)"))

    update_state(state, name, status="complete", bytes_downloaded=actual_size,
                 sha_ok=sha_ok, finished=base.now_iso(), error=None)
    log(f"[{name}] done")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="resolve and print the plan only")
    ap.add_argument("--limit", type=int, default=None, help="only download the first N (by size)")
    args = ap.parse_args(argv)

    log = base.Logger(LOG_PATH, dry_run=args.dry_run)
    _assert_not_forbidden()

    names = read_wanted(WANTED_PATH)
    log(f"{len(names)} wanted name(s) read from {WANTED_PATH}")

    state = load_state()
    items = resolve_all(names, state, log)

    to_download = [it for it in items if it.action == "to_download"]
    to_download.sort(key=lambda it: (
        (it.meta_size or it.catalog_length) is None,
        it.meta_size or it.catalog_length or 0,
    ))
    if args.limit is not None:
        to_download = to_download[: args.limit]

    total_bytes = sum((it.meta_size or it.catalog_length or 0) for it in to_download)
    log(f"plan: {len(to_download)} file(s) to download, "
        f"total {base.human_gb(total_bytes)} "
        f"(of {len(items)} wanted names)")

    if args.dry_run:
        for it in to_download:
            print(f"  {it.name:<55} {base.human_gb(it.meta_size or it.catalog_length):>10}  {it.filename}")
        not_found = [it.name for it in items if it.action in ("not_found", "resolve_error")]
        if not_found:
            print("Not found / resolve errors:")
            for n in not_found:
                print(f"  - {n}")
        return 0

    try:
        for it in to_download:
            try:
                process_item(it, state, log)
            except KeyboardInterrupt:
                raise
            except Exception as ex:  # keep going across items
                log(f"[{it.name}] ERROR: {ex!r}")
                update_state(state, it.name, status="error", error=repr(ex))
    except KeyboardInterrupt:
        log("interrupted by user (Ctrl-C); state saved, safe to re-run")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
