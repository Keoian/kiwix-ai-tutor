"""Build a ZIM of the LDS standard works from a public-domain scripture corpus.

Source data: `beandog/lds-scriptures` (https://github.com/beandog/lds-scriptures),
`json/lds-scriptures-json.txt` -- a flat JSON list of verse rows (one dict per
verse: volume_title, book_title, chapter_number, verse_number, verse_title,
scripture_text). That repo's README.txt states plainly:

    Copyright: public domain

for the scripture text (King James Bible Old + New Testament, Book of Mormon,
Doctrine and Covenants, Pearl of Great Price). See docs/lds_theology_offline_options.md
for the full licensing research.

This script does NOT include General Conference talks or any other LDS Church
material -- those are copyrighted and out of scope (see the doc above).

Output layout (paths inside the ZIM):
    index.html                          top-level index, all 4 volumes
    bible/index.html                    King James Bible volume index (OT + NT)
    bible/<book-slug>/index.html        book index (chapter list)
    bible/<book-slug>/<chapter>.html    chapter page (verses)
    bom/index.html                      Book of Mormon volume index
    bom/<book-slug>/index.html
    bom/<book-slug>/<chapter>.html
    dc/index.html                       Doctrine and Covenants (single "book" ==
                                         volume, so sections sit directly under dc/)
    dc/<section>.html
    pgp/index.html                      Pearl of Great Price volume index
    pgp/<book-slug>/index.html
    pgp/<book-slug>/<chapter>.html

Usage:
    python scripts/build_lds_scriptures_zim.py <lds-scriptures-json.txt> <output.zim>

Requires `pip install libzim` (python-libzim). If that wheel is unavailable on
your platform, see the FALLBACK note in main() below: this script can also be
asked to dump the HTML tree to a directory (--html-dir) instead of building a
ZIM, so the ZIM can be built later on Linux with the same libzim package or
with zimwriterfs.
"""

from __future__ import annotations

import argparse
import datetime
import html as html_lib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# --- Volume/book layout -----------------------------------------------------

# Maps the corpus's `volume_title` to (zim directory, display volume title,
# optional sub-heading used on the merged Bible index page).
VOLUME_MAP = {
    "Old Testament": ("bible", "King James Bible", "Old Testament"),
    "New Testament": ("bible", "King James Bible", "New Testament"),
    "Book of Mormon": ("bom", "Book of Mormon", None),
    "Doctrine and Covenants": ("dc", "Doctrine and Covenants", None),
    "Pearl of Great Price": ("pgp", "Pearl of Great Price", None),
}

# Volumes whose single "book" is identical to the volume itself (no book-level
# subdirectory/index -- chapters/sections sit directly under the volume dir).
FLAT_VOLUME_DIRS = {"dc"}

TOP_INDEX_ORDER = ["bible", "bom", "dc", "pgp"]


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def esc(text: str) -> str:
    return html_lib.escape(text, quote=False)


def page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{esc(title)}</title>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


class Page:
    """One HTML entry to add to the ZIM (or write to disk for the fallback)."""

    __slots__ = ("path", "title", "html")

    def __init__(self, path: str, title: str, html: str):
        self.path = path
        self.title = title
        self.html = html


def load_verses(json_path: Path) -> list[dict]:
    with json_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"unexpected corpus shape in {json_path}")
    required = {"volume_title", "book_title", "chapter_number", "verse_number", "scripture_text"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"corpus rows missing expected fields: {missing}")
    return rows


def build_structure(rows: list[dict]) -> dict:
    """Group flat verse rows into volume -> book -> chapter -> [verses]."""
    volumes: dict[str, dict] = {}
    for row in rows:
        vol_title = row["volume_title"]
        if vol_title not in VOLUME_MAP:
            raise ValueError(f"unrecognized volume_title in corpus: {vol_title!r}")
        vol_dir, vol_display, sub_heading = VOLUME_MAP[vol_title]
        vol = volumes.setdefault(
            vol_dir, {"display": vol_display, "books": {}, "book_order": [], "sub_of": {}}
        )
        book_title = row["book_title"]
        if book_title not in vol["books"]:
            vol["books"][book_title] = {"chapters": {}, "chapter_order": []}
            vol["book_order"].append(book_title)
            vol["sub_of"][book_title] = sub_heading
        book = vol["books"][book_title]
        chap_num = row["chapter_number"]
        if chap_num not in book["chapters"]:
            book["chapters"][chap_num] = []
            book["chapter_order"].append(chap_num)
        book["chapters"][chap_num].append(row)
    return volumes


def chapter_page_and_title(vol_dir: str, vol_display: str, book_title: str, chap_num: int) -> tuple[str, str]:
    """Return (zim path, page title) for one chapter/section."""
    if vol_dir in FLAT_VOLUME_DIRS:
        path = f"{vol_dir}/{chap_num}.html"
        title = f"{book_title} {chap_num}"
    else:
        book_slug = slugify(book_title)
        path = f"{vol_dir}/{book_slug}/{chap_num}.html"
        title = f"{book_title} {chap_num} - {vol_display}"
    return path, title


def generate_pages(volumes: dict) -> list[Page]:
    pages: list[Page] = []

    # Top-level index.
    vol_links = []
    for vol_dir in TOP_INDEX_ORDER:
        vol = volumes.get(vol_dir)
        if not vol:
            continue
        vol_links.append(f'<li><a href="{vol_dir}/index.html">{esc(vol["display"])}</a></li>')
    top_body = (
        "<h1>LDS Standard Works</h1>\n"
        "<p>Public-domain scripture text: the King James Bible, the Book of Mormon, "
        "the Doctrine and Covenants, and the Pearl of Great Price. "
        "Does not include General Conference talks or other copyrighted Church material.</p>\n"
        "<ul>\n" + "\n".join(vol_links) + "\n</ul>\n"
    )
    pages.append(Page("index.html", "LDS Standard Works", page("LDS Standard Works", top_body)))

    for vol_dir in TOP_INDEX_ORDER:
        vol = volumes.get(vol_dir)
        if not vol:
            continue
        vol_display = vol["display"]

        if vol_dir in FLAT_VOLUME_DIRS:
            # Single "book" == volume: list sections directly.
            assert len(vol["book_order"]) == 1
            book_title = vol["book_order"][0]
            book = vol["books"][book_title]
            chap_nums = sorted(book["chapter_order"])
            items = "\n".join(
                f'<li><a href="{vol_dir}/{n}.html">{esc(book_title)} {n}</a></li>' for n in chap_nums
            )
            vol_body = (
                f"<h1>{esc(vol_display)}</h1>\n"
                f'<p><a href="../index.html">LDS Standard Works</a></p>\n'
                f"<ol>\n{items}\n</ol>\n"
            )
            pages.append(Page(f"{vol_dir}/index.html", vol_display, page(vol_display, vol_body)))

            for n in chap_nums:
                verses = sorted(book["chapters"][n], key=lambda r: r["verse_number"])
                path, title = chapter_page_and_title(vol_dir, vol_display, book_title, n)
                verse_items = "\n".join(
                    f'<li value="{v["verse_number"]}">{esc(v["scripture_text"])}</li>' for v in verses
                )
                nav = f'<p><a href="index.html">{esc(vol_display)}</a> | <a href="../index.html">LDS Standard Works</a></p>\n'
                body = f"<h1>{esc(title)}</h1>\n{nav}<ol>\n{verse_items}\n</ol>\n"
                pages.append(Page(path, title, page(title, body)))
            continue

        # Normal nested volume (bible / bom / pgp): book subdirectories.
        sub_groups: dict[str | None, list[str]] = {}
        for book_title in vol["book_order"]:
            heading = vol["sub_of"][book_title]
            sub_groups.setdefault(heading, []).append(book_title)

        vol_sections = []
        for heading, book_titles in sub_groups.items():
            links = "\n".join(
                f'<li><a href="{slugify(bt)}/index.html">{esc(bt)}</a></li>' for bt in book_titles
            )
            if heading:
                vol_sections.append(f"<h2>{esc(heading)}</h2>\n<ul>\n{links}\n</ul>")
            else:
                vol_sections.append(f"<ul>\n{links}\n</ul>")
        vol_body = (
            f"<h1>{esc(vol_display)}</h1>\n"
            f'<p><a href="../index.html">LDS Standard Works</a></p>\n'
            + "\n".join(vol_sections)
            + "\n"
        )
        pages.append(Page(f"{vol_dir}/index.html", vol_display, page(vol_display, vol_body)))

        for book_title in vol["book_order"]:
            book = vol["books"][book_title]
            book_slug = slugify(book_title)
            chap_nums = sorted(book["chapter_order"])
            book_title_full = f"{book_title} - {vol_display}"
            chap_items = "\n".join(
                f'<li><a href="{n}.html">{book_title} {n}</a></li>' for n in chap_nums
            )
            book_body = (
                f"<h1>{esc(book_title_full)}</h1>\n"
                f'<p><a href="index.html">{esc(vol_display)}</a> | <a href="../../index.html">LDS Standard Works</a></p>\n'
                f"<ol>\n{chap_items}\n</ol>\n"
            )
            pages.append(
                Page(
                    f"{vol_dir}/{book_slug}/index.html",
                    book_title_full,
                    page(book_title_full, book_body),
                )
            )

            for n in chap_nums:
                verses = sorted(book["chapters"][n], key=lambda r: r["verse_number"])
                path, title = chapter_page_and_title(vol_dir, vol_display, book_title, n)
                verse_items = "\n".join(
                    f'<li value="{v["verse_number"]}">{esc(v["scripture_text"])}</li>' for v in verses
                )
                nav = (
                    f'<p><a href="index.html">{esc(book_title)}</a> | '
                    f'<a href="../index.html">{esc(vol_display)}</a> | '
                    f'<a href="../../index.html">LDS Standard Works</a></p>\n'
                )
                body = f"<h1>{esc(title)}</h1>\n{nav}<ol>\n{verse_items}\n</ol>\n"
                pages.append(Page(path, title, page(title, body)))

    return pages


def write_html_tree(pages: Iterable[Page], out_dir: Path) -> None:
    """Fallback path: write the same pages as a static HTML tree for a later
    Linux-side ZIM build (e.g. with zimwriterfs or this same script's libzim
    path run on Linux)."""
    for p in pages:
        dest = out_dir / p.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(p.html, encoding="utf-8")


def build_zim(pages: list[Page], out_path: Path, date: datetime.date) -> None:
    from libzim.writer import Creator, Hint, Item, StringProvider

    class HtmlItem(Item):
        def __init__(self, p: Page):
            super().__init__()
            self._p = p

        def get_path(self) -> str:
            return self._p.path

        def get_title(self) -> str:
            return self._p.title

        def get_mimetype(self) -> str:
            return "text/html"

        def get_contentprovider(self):
            return StringProvider(self._p.html)

        def get_hints(self):
            return {Hint.FRONT_ARTICLE: True}

    with Creator(str(out_path)).config_indexing(True, "eng") as creator:
        creator.set_mainpath("index.html")
        for p in pages:
            creator.add_item(HtmlItem(p))
        creator.add_metadata("Title", "LDS Standard Works")
        creator.add_metadata(
            "Description",
            "Public-domain LDS standard works: King James Bible, Book of Mormon, "
            "Doctrine and Covenants, Pearl of Great Price. No General Conference talks.",
        )
        creator.add_metadata("Language", "eng")
        creator.add_metadata("Creator", "beandog/lds-scriptures (Mormon Documentation Project)")
        creator.add_metadata("Publisher", "kiwix-ai-tutor")
        creator.add_metadata("Date", date)
        creator.add_metadata("Name", "lds_scriptures_en_all")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path, help="path to lds-scriptures-json.txt")
    ap.add_argument("output", type=Path, help="output .zim path (must not already exist)")
    ap.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="FALLBACK: also/instead write the static HTML tree here (for a Linux-side "
        "ZIM build if the libzim wheel is unavailable on this platform)",
    )
    ap.add_argument("--skip-zim", action="store_true", help="only write --html-dir, skip building the .zim")
    args = ap.parse_args()

    rows = load_verses(args.json_path)
    volumes = build_structure(rows)
    pages = generate_pages(volumes)
    print(f"generated {len(pages)} pages from {len(rows)} verses", file=sys.stderr)

    if args.html_dir:
        write_html_tree(pages, args.html_dir)
        print(f"wrote HTML tree to {args.html_dir}", file=sys.stderr)

    if args.skip_zim:
        return 0

    if args.output.exists():
        print(f"refusing to overwrite existing file: {args.output}", file=sys.stderr)
        return 1

    try:
        import libzim  # noqa: F401
    except ImportError:
        print(
            "libzim is not importable on this platform. Falling back: writing the "
            "HTML tree only. Build the ZIM later (e.g. on Linux) with the same "
            "script's build_zim() or with zimwriterfs.",
            file=sys.stderr,
        )
        if not args.html_dir:
            fallback_dir = args.output.with_suffix("")
            write_html_tree(pages, fallback_dir)
            print(f"wrote HTML tree to {fallback_dir}", file=sys.stderr)
        return 1

    build_zim(pages, args.output, datetime.date.today())
    size = args.output.stat().st_size
    print(f"built {args.output} ({size:,} bytes, {len(pages)} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
