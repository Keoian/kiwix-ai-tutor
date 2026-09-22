#!/usr/bin/env python3
"""Update every ZIM in D:\\Kiwix to its latest Kiwix release, keeping old files.

Stdlib only (plus the `curl` binary for the actual file transfer -- no
aria2/wget). Designed for Windows, but the logic has no Windows-only calls
except the default drive letters, which are configurable below.

Behaviour is documented in the module-level docstrings of each function.
See the task that produced this script for the full spec. Short version:

  1. Enumerate D:\\Kiwix\\*.zim, parse "<basename>_<YYYY-MM>.zim".
  2. For each basename, ask the Kiwix OPDS catalog
     (https://library.kiwix.org/catalog/v2/entries) for the latest release.
  3. Skip if already up to date or an identical file is already present.
  4. Otherwise download with curl (resumable, retried), verify size (and
     sha-256 when the .meta4 sidecar publishes one), then move the old file
     to D:\\Kiwix-2023 and atomically rename the new one into place.
  5. Never delete anything. State is durable in _update_state.json so the
     script is safe to Ctrl-C and re-run.

Catalog quirk discovered while writing this script (2026-09-22): the OPDS
<link length="..."> attribute on the acquisition link can be a few hundred
bytes off from the real file size (observed: catalog said 259175424 bytes,
the .meta4 sidecar and the real HTTP Content-Length both agreed on
259175193 bytes). The .meta4 sidecar's <size> element matches the real
download every time it was checked, so this script treats the *catalog*
`length` as a planning estimate only (dry-run table, free-space check) and
always fetches the small .meta4 sidecar before verifying a completed
download, using its <size> (and <hash type="sha-256"> when present) as the
authoritative check.

Also discovered: download.kiwix.org no longer serves directory listings
(the whole tree 301-redirects to the marketing site hub.kiwix.org), so the
"scrape the directory listing" fallback described in the original spec is
implemented but will typically find nothing; the real fallback that works
is fetching the *unfiltered* OPDS catalog (count=-1, no name filter) once
per run and searching it in memory for an exact basename match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

LOCAL_DIR = Path(r"D:\Kiwix")
ARCHIVE_DIR = Path(r"D:\Kiwix-2023")
STATE_PATH = LOCAL_DIR / "_update_state.json"
LOG_PATH = LOCAL_DIR / "_update_log.txt"

CATALOG_BASE = "https://library.kiwix.org/catalog/v2/entries"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}

FLAVOUR_SUFFIXES = ["nopic", "maxi", "mini"]  # tried in this order when stripping

FREE_SPACE_MARGIN_BYTES = 5 * 1024 ** 3  # 5 GB
CHUNK_SIZE = 16 * 1024 * 1024  # 16 MiB, for sha256 hashing

CURL_CANDIDATES = [
    "curl",  # PATH (covers Windows' built-in System32\curl.exe too)
    r"C:\Program Files\Git\mingw64\bin\curl.exe",
    r"C:\msys64\mingw64\bin\curl.exe",
    "/mingw64/bin/curl",
]

NAME_DATE_RE = re.compile(r"^(?P<base>.+)_(?P<date>\d{4}-\d{2})$")
# Kiwix occasionally re-releases a ZIM within the same calendar month, e.g.
# "...en_all_2026-07a.zim" for the second 2026-07 release. Catalog hrefs can
# carry that trailing letter; local filenames (per this script's own naming,
# and per the task spec) never do, so date *comparisons* use YYYY-MM only.
CATALOG_DATE_RE = re.compile(r"^(?P<base>.+)_(?P<date>\d{4}-\d{2}[a-z]?)$")

MAX_CURL_ATTEMPTS = 10
CURL_RETRY_SLEEP_SECONDS = 15
HTTP_TIMEOUT_SECONDS = 60


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def human_gb(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "?"
    return f"{num_bytes / 1024 ** 3:.2f} GB"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Logger:
    def __init__(self, log_path: Path, dry_run: bool):
        self.log_path = log_path
        self.dry_run = dry_run

    def __call__(self, msg: str) -> None:
        line = f"{now_iso()} {msg}"
        print(line)
        if self.dry_run:
            return
        try:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            # Logging must never crash the run.
            pass


def get_curl() -> str:
    for cand in CURL_CANDIDATES:
        found = shutil.which(cand)
        if found:
            return found
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        "No curl executable found (tried PATH and known MSYS2/Git-for-Windows "
        "locations). Install curl or add it to PATH."
    )


# --------------------------------------------------------------------------
# Local file inventory
# --------------------------------------------------------------------------


@dataclass
class LocalZim:
    path: Path
    basename: str
    date: str  # YYYY-MM
    size: int


def scan_local(only: Optional[str] = None) -> list[LocalZim]:
    """Enumerate D:\\Kiwix\\*.zim, parse "<basename>_<YYYY-MM>.zim"."""
    out = []
    for p in sorted(LOCAL_DIR.glob("*.zim")):
        m = re.match(r"^(?P<base>.+)_(?P<date>\d{4}-\d{2})\.zim$", p.name)
        if not m:
            continue  # not a dated release file (e.g. a stray .zim.part); skip
        base = m.group("base")
        if only and only.lower() not in base.lower():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        out.append(LocalZim(path=p, basename=base, date=m.group("date"), size=size))
    return out


# --------------------------------------------------------------------------
# OPDS catalog
# --------------------------------------------------------------------------


def http_get(url: str, timeout: int = HTTP_TIMEOUT_SECONDS) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "kiwix-ai-tutor-zim-updater/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_catalog(name: Optional[str] = None) -> bytes:
    if name:
        url = f"{CATALOG_BASE}?count=-1&name={urllib.parse.quote(name)}"
    else:
        url = f"{CATALOG_BASE}?count=-1"
    return http_get(url)


@dataclass
class CatalogEntry:
    name: str
    flavour: str
    href: str  # acquisition link, typically *.zim.meta4
    length: Optional[int]
    category: str
    fbase: Optional[str] = None  # basename portion parsed back out of href
    fdate: Optional[str] = None


def derive_fbase_date(href: str) -> tuple[Optional[str], Optional[str]]:
    fname = href.rsplit("/", 1)[-1]
    for suf in (".zim.meta4", ".zim"):
        if fname.endswith(suf):
            fname = fname[: -len(suf)]
            break
    m = CATALOG_DATE_RE.match(fname)
    if not m:
        return None, None
    return m.group("base"), m.group("date")


def category_from_href(href: str) -> str:
    m = re.search(r"/zim/([^/]+)/", href)
    return m.group(1) if m else "unknown"


def parse_entries(xml_bytes: bytes) -> list[CatalogEntry]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    out = []
    for e in root.findall("a:entry", ATOM_NS):
        name = e.findtext("a:name", default="", namespaces=ATOM_NS) or ""
        flavour = e.findtext("a:flavour", default="", namespaces=ATOM_NS) or ""
        link = None
        for l in e.findall("a:link", ATOM_NS):
            if l.get("rel") == "http://opds-spec.org/acquisition/open-access":
                link = l
                break
        if link is None or not link.get("href"):
            continue
        href = link.get("href")
        length_attr = link.get("length")
        length = int(length_attr) if length_attr and length_attr.isdigit() else None
        fbase, fdate = derive_fbase_date(href)
        out.append(
            CatalogEntry(
                name=name,
                flavour=flavour,
                href=href,
                length=length,
                category=category_from_href(href),
                fbase=fbase,
                fdate=fdate,
            )
        )
    return out


class FullCatalogCache:
    """Fetches the unfiltered catalog (count=-1, no name) at most once."""

    def __init__(self):
        self._entries: Optional[list[CatalogEntry]] = None
        self._error: Optional[str] = None

    def get(self, log) -> list[CatalogEntry]:
        if self._entries is None and self._error is None:
            try:
                log("fetching full OPDS catalog (fallback, one-time)...")
                data = fetch_catalog(name=None)
                self._entries = parse_entries(data)
                log(f"full catalog: {len(self._entries)} entries")
            except (urllib.error.URLError, OSError) as ex:
                self._error = str(ex)
                self._entries = []
        return self._entries or []


def try_directory_listing_fallback(basename: str, log) -> Optional[CatalogEntry]:
    """Best-effort scrape of https://download.kiwix.org/zim/<category>/ .

    As of 2026-09-22 download.kiwix.org 301-redirects its whole tree to the
    hub.kiwix.org marketing site, so this normally finds nothing -- it is
    kept as a documented, harmless no-op fallback in case that changes.
    """
    guesses = [
        "stack_exchange", "wikipedia", "wikibooks", "wikiversity", "wikihow",
        "other", "gutenberg", "ifixit", "zimit", "freecodecamp", "khanacademy",
    ]
    for cat in guesses:
        url = f"https://download.kiwix.org/zim/{cat}/"
        try:
            html = http_get(url, timeout=20).decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError):
            continue
        # Look for "<basename>_YYYY-MM(.zim)" links, take the max date.
        pattern = re.escape(basename) + r"_(\d{4}-\d{2})\.zim"
        dates = re.findall(pattern, html)
        if not dates:
            continue
        latest = max(dates)
        href = f"{url}{basename}_{latest}.zim.meta4"
        return CatalogEntry(
            name=basename, flavour="", href=href, length=None,
            category=cat, fbase=basename, fdate=latest,
        )
    return None


def resolve_basename(
    basename: str, full_cache: FullCatalogCache, log
) -> Optional[CatalogEntry]:
    """Find the catalog entry whose reconstructed filename == basename exactly."""
    queries = [basename]
    for suf in FLAVOUR_SUFFIXES:
        if basename.endswith("_" + suf):
            queries.append(basename[: -(len(suf) + 1)])

    tried = set()
    for q in queries:
        if q in tried:
            continue
        tried.add(q)
        try:
            data = fetch_catalog(name=q)
        except (urllib.error.URLError, OSError) as ex:
            log(f"  catalog query failed for name={q!r}: {ex}")
            continue
        for entry in parse_entries(data):
            if entry.fbase == basename:
                return entry

    # Fallback: search the full unfiltered catalog once.
    for entry in full_cache.get(log):
        if entry.fbase == basename:
            return entry

    # Fallback: directory listing scrape (see docstring -- usually a no-op).
    return try_directory_listing_fallback(basename, log)


# --------------------------------------------------------------------------
# meta4 sidecar (authoritative size + sha256)
# --------------------------------------------------------------------------


def fetch_meta4_info(meta4_url: str, log) -> tuple[Optional[int], Optional[str]]:
    try:
        data = http_get(meta4_url, timeout=30).decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as ex:
        log(f"  could not fetch meta4 sidecar {meta4_url}: {ex}")
        return None, None
    m_size = re.search(r"<size>(\d+)</size>", data)
    m_sha = re.search(r'<hash type="sha-256">([0-9a-fA-F]{64})</hash>', data)
    size = int(m_size.group(1)) if m_size else None
    sha = m_sha.group(1).lower() if m_sha else None
    return size, sha


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# State file
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


def update_state(state: dict, basename: str, **fields) -> None:
    entry = state.setdefault(basename, {})
    entry.update(fields)
    entry["updated"] = now_iso()
    save_state(state)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass
class PlanItem:
    basename: str
    local: Optional[LocalZim]
    entry: Optional[CatalogEntry]
    action: str  # up_to_date | done | to_download | no_release_found
    new_filename: Optional[str] = None
    length: Optional[int] = None
    url: Optional[str] = None
    meta4_url: Optional[str] = None


def build_plan(locals_: list[LocalZim], log) -> list[PlanItem]:
    full_cache = FullCatalogCache()
    plan = []
    for lz in locals_:
        entry = resolve_basename(lz.basename, full_cache, log)
        if entry is None or not entry.fdate:
            plan.append(PlanItem(basename=lz.basename, local=lz, entry=None,
                                  action="no_release_found"))
            continue

        new_filename = f"{lz.basename}_{entry.fdate}.zim"
        meta4_url = entry.href
        direct_url = meta4_url[: -len(".meta4")] if meta4_url.endswith(".meta4") else meta4_url

        if entry.fdate[:7] == lz.date:
            plan.append(PlanItem(basename=lz.basename, local=lz, entry=entry,
                                  action="up_to_date", new_filename=new_filename,
                                  length=entry.length, url=direct_url, meta4_url=meta4_url))
            continue

        existing_new = LOCAL_DIR / new_filename
        if existing_new.exists() and entry.length and existing_new.stat().st_size == entry.length:
            plan.append(PlanItem(basename=lz.basename, local=lz, entry=entry,
                                  action="done", new_filename=new_filename,
                                  length=entry.length, url=direct_url, meta4_url=meta4_url))
            continue

        plan.append(PlanItem(basename=lz.basename, local=lz, entry=entry,
                              action="to_download", new_filename=new_filename,
                              length=entry.length, url=direct_url, meta4_url=meta4_url))
    return plan


def print_plan_table(plan: list[PlanItem]) -> None:
    print(f"{'basename':<55} {'local':<8} {'latest':<8} {'size':>10}  action       url")
    print("-" * 140)
    for item in plan:
        local_date = item.local.date if item.local else "?"
        latest_date = item.entry.fdate if item.entry else "-"
        size = human_gb(item.length)
        url = item.url or ""
        print(f"{item.basename:<55} {local_date:<8} {latest_date:<8} {size:>10}  {item.action:<12} {url}")

    counts: dict[str, int] = {}
    total_download_bytes = 0
    not_found = []
    for item in plan:
        counts[item.action] = counts.get(item.action, 0) + 1
        if item.action == "to_download" and item.length:
            total_download_bytes += item.length
        if item.action == "no_release_found":
            not_found.append(item.basename)

    print()
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Total to download: {human_gb(total_download_bytes)}")
    if not_found:
        print("Not found:")
        for b in not_found:
            print(f"  - {b}")


# --------------------------------------------------------------------------
# Download / verify / move
# --------------------------------------------------------------------------


def disk_free_bytes(drive: str) -> int:
    return shutil.disk_usage(drive).free


def curl_download(url: str, dest_part: Path, log) -> bool:
    curl_bin = get_curl()
    cmd = [
        curl_bin, "-L", "-C", "-",
        "--retry", "20", "--retry-all-errors", "--retry-delay", "15",
        "--fail", "--speed-limit", "1000", "--speed-time", "120",
        "-o", str(dest_part), url,
    ]
    for attempt in range(1, MAX_CURL_ATTEMPTS + 1):
        log(f"  curl attempt {attempt}/{MAX_CURL_ATTEMPTS}: {dest_part.name}")
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            raise
        except OSError as ex:
            log(f"  curl failed to launch: {ex}")
            rc = -1
        if rc == 0:
            return True
        log(f"  curl exited {rc}")
        if attempt < MAX_CURL_ATTEMPTS:
            time.sleep(CURL_RETRY_SLEEP_SECONDS)
    return False


def move_old_file(old_path: Path, log) -> str:
    """Move old_path to ARCHIVE_DIR. Returns 'moved', 'missing' or 'move_deferred'."""
    if not old_path.exists():
        return "missing"
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / old_path.name
    try:
        if dest.exists():
            log(f"  archive already has {dest.name}, leaving {old_path} in place")
            return "move_deferred"
        shutil.move(str(old_path), str(dest))
        log(f"  moved old file -> {dest}")
        return "moved"
    except PermissionError as ex:
        log(f"  old file locked, deferring move: {ex}")
        return "move_deferred"
    except OSError as ex:
        log(f"  move failed ({ex}), deferring")
        return "move_deferred"


def process_download(item: PlanItem, state: dict, log) -> None:
    basename = item.basename
    update_state(state, basename, status="downloading", old=item.local.path.name if item.local else None,
                 new=item.new_filename, url=item.url, length=item.length,
                 started=now_iso())

    free = disk_free_bytes(str(LOCAL_DIR.drive) + "\\")
    needed = (item.length or 0) + FREE_SPACE_MARGIN_BYTES
    if free < needed:
        log(f"[{basename}] SKIP: not enough free space "
            f"(need {human_gb(needed)}, have {human_gb(free)})")
        update_state(state, basename, status="skipped_no_space", error=None)
        return

    part_path = LOCAL_DIR / (item.new_filename + ".part")
    log(f"[{basename}] downloading -> {part_path.name} ({human_gb(item.length)})")
    ok = curl_download(item.url, part_path, log)
    if not ok:
        log(f"[{basename}] FAILED after {MAX_CURL_ATTEMPTS} curl attempts")
        update_state(state, basename, status="error", error="curl failed after max attempts",
                     bytes_downloaded=part_path.stat().st_size if part_path.exists() else 0)
        return

    # Authoritative size/hash from the meta4 sidecar (catalog `length` can be
    # slightly stale -- see module docstring).
    meta_size, meta_sha = fetch_meta4_info(item.meta4_url, log) if item.meta4_url else (None, None)
    expected_size = meta_size or item.length
    actual_size = part_path.stat().st_size

    if expected_size is not None and actual_size != expected_size:
        log(f"[{basename}] SIZE MISMATCH: expected {expected_size}, got {actual_size}")
        update_state(state, basename, status="error",
                     error=f"size mismatch: expected {expected_size} got {actual_size}",
                     bytes_downloaded=actual_size)
        return

    sha_ok = None
    if meta_sha:
        log(f"[{basename}] verifying sha256...")
        actual_sha = sha256_file(part_path)
        sha_ok = actual_sha == meta_sha
        if not sha_ok:
            log(f"[{basename}] SHA256 MISMATCH: expected {meta_sha}, got {actual_sha}")
            update_state(state, basename, status="error",
                         error=f"sha256 mismatch: expected {meta_sha} got {actual_sha}",
                         bytes_downloaded=actual_size, sha_ok=False)
            return

    final_path = LOCAL_DIR / item.new_filename
    os.replace(part_path, final_path)
    log(f"[{basename}] verified and renamed -> {final_path.name}"
        + (" (sha256 ok)" if sha_ok else " (size only, no sha256 published)"))

    move_status = "missing"
    if item.local is not None:
        move_status = move_old_file(item.local.path, log)

    update_state(state, basename, status="complete", bytes_downloaded=actual_size,
                 sha_ok=sha_ok, move_status=move_status, finished=now_iso(), error=None)
    log(f"[{basename}] done (move_status={move_status})")


def retry_deferred_moves(state: dict, log) -> None:
    any_found = False
    for basename, entry in state.items():
        if entry.get("move_status") != "move_deferred":
            continue
        any_found = True
        old_name = entry.get("old")
        if not old_name:
            continue
        old_path = LOCAL_DIR / old_name
        status = move_old_file(old_path, log)
        entry["move_status"] = status
        entry["updated"] = now_iso()
        save_state(state)
        log(f"[{basename}] retry-move -> {status}")
    if not any_found:
        log("no deferred moves to retry")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="resolve and print the plan only")
    ap.add_argument("--only", default=None, help="substring filter on basename")
    ap.add_argument("--retry-moves", action="store_true", help="only re-attempt deferred moves")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N downloads")
    args = ap.parse_args(argv)

    log = Logger(LOG_PATH, dry_run=args.dry_run)

    if args.retry_moves:
        state = load_state()
        retry_deferred_moves(state, log)
        return 0

    locals_ = scan_local(only=args.only)
    if not locals_:
        print("No matching local .zim files found under", LOCAL_DIR)
        return 1

    log(f"resolving {len(locals_)} local basenames against the Kiwix catalog...")
    plan = build_plan(locals_, log)

    if args.dry_run:
        print_plan_table(plan)
        return 0

    to_download = [p for p in plan if p.action == "to_download"]
    to_download.sort(key=lambda p: (p.length is None, p.length or 0))
    if args.limit is not None:
        to_download = to_download[: args.limit]

    log(f"plan: {len(to_download)} file(s) to download "
        f"(of {len(plan)} local basenames)")

    state = load_state()
    try:
        for item in to_download:
            try:
                process_download(item, state, log)
            except KeyboardInterrupt:
                raise
            except Exception as ex:  # keep going across basenames
                log(f"[{item.basename}] ERROR: {ex!r}")
                update_state(state, item.basename, status="error", error=repr(ex))
    except KeyboardInterrupt:
        log("interrupted by user (Ctrl-C); state saved, safe to re-run")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
