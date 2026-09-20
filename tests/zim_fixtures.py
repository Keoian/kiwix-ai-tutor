"""ZIM fixture builder used by ``tests/conftest.py``.

Builds small, deterministic ZIM archives with ``libzim.writer.Creator`` so
the WP-B1/WP-B2 tests exercise real libzim binaries rather than mocks.

Note on redirects (see docs/donor_inventory.md and the WP-B1 task brief):
libzim's ``Creator`` does NOT raise when a dangling redirect or a
redirect loop is added with ``add_redirection`` — both are accepted at
add-time and silently *dropped* by libzim itself when the archive is
finalized (confirmed empirically: libzim prints "Detect dangling
redirects" / "Detect loops and/or blind chains of redirects" and removes
the offending entries during ``Creator.__exit__``). That means a built
fixture ZIM can never contain a redirect cycle or a dangling redirect for
our resolver to exercise — cycle/over-deep-chain behavior is instead unit
tested against fake archive/entry objects in ``tests/test_zim_resolve.py``.
This module only builds well-formed redirects (a simple one and a 2-hop
chain), which libzim keeps.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import libzim.writer as zw

LANGUAGE = "eng"
FIXTURE_NAME = "fixture_en_school"

# A handful of hand-written, richer articles: headings, an infobox table,
# internal links, a list, and deliberately non-ASCII text so encoding bugs
# show up immediately instead of silently mangling offsets.
_RICH_ARTICLES: dict[str, str] = {
    "pythagorean_theorem": """
<html><head><title>Pythagorean theorem</title></head><body>
<h1>Pythagorean theorem</h1>
<table class="infobox">
<tr><th>Field</th><td>Geometry</td></tr>
<tr><th>Discoverer</th><td>Pythagoras</td></tr>
</table>
<h2>Statement</h2>
<p>Pythagoras &mdash; a&sup2;+b&sup2;=c&sup2; describes right triangles.</p>
<h3>Proof sketch</h3>
<p>See also <a href="algebra_basics">algebra basics</a> and
<a href="geometry_intro">geometry intro</a>.</p>
<ul><li>Legs a and b</li><li>Hypotenuse c</li><li>Right angle</li></ul>
</body></html>
""",
    "algebra_basics": """
<html><head><title>Algebra basics</title></head><body>
<h1>Algebra basics</h1>
<h2>Variables</h2>
<p>A naïve first look at variables and expressions.</p>
<h2>Equations</h2>
<p>Linked from <a href="pythagorean_theorem">Pythagorean theorem</a>.</p>
<ul><li>Addition</li><li>Subtraction</li><li>Multiplication</li></ul>
</body></html>
""",
    "history_of_mathematics": """
<html><head><title>History of mathematics</title></head><body>
<h1>History of mathematics</h1>
<table class="infobox">
<tr><th>Notable figure</th><td>Erdős</td></tr>
<tr><th>Era</th><td>20th century</td></tr>
</table>
<h2>Ancient roots</h2>
<p>Includes work later attributed to Pythagoras.</p>
<h3>Modern era</h3>
<p>Paul Erdős published prolifically.</p>
<ul><li>Number theory</li><li>Combinatorics</li></ul>
</body></html>
""",
    "geometry_intro": """
<html><head><title>Geometry intro</title></head><body>
<h1>Geometry intro</h1>
<h2>Shapes</h2>
<p>Triangles, circles, and the naïve intuition that shapes are simple.</p>
<h2>Theorems</h2>
<p>See <a href="pythagorean_theorem">Pythagorean theorem</a>.</p>
<ul><li>Points</li><li>Lines</li><li>Planes</li></ul>
</body></html>
""",
    "erdos_number": """
<html><head><title>Erdős number</title></head><body>
<h1>Erdős number</h1>
<table class="infobox">
<tr><th>Named after</th><td>Paul Erdős</td></tr>
</table>
<h2>Definition</h2>
<p>Collaboration distance to Erdős in
<a href="history_of_mathematics">history of mathematics</a>.</p>
<h3>Trivia</h3>
<p>Most working mathematicians have a small Erdős number.</p>
<ul><li>0 for Erdős himself</li><li>1 for co-authors</li></ul>
</body></html>
""",
}

_SIMPLE_TOPICS = [
    "photosynthesis", "cell_biology", "world_war_two", "the_water_cycle",
    "fractions", "decimals", "the_solar_system", "plate_tectonics",
    "the_french_revolution", "chemical_bonds", "the_periodic_table",
    "newtons_laws", "the_civil_war", "ecosystems", "grammar_basics",
    "poetry_forms", "the_renaissance", "ancient_egypt", "ancient_rome",
    "ancient_greece", "computer_basics", "the_internet", "electricity",
    "magnetism", "sound_waves", "light_and_optics", "the_human_heart",
    "the_digestive_system", "genetics_basics", "evolution_basics",
    "weather_basics", "climate_basics", "rivers_and_lakes", "oceans",
    "mountains", "deserts", "rainforests", "the_industrial_revolution",
    "the_cold_war", "the_united_nations", "government_basics",
    "economics_basics", "supply_and_demand", "the_stock_market",
    "map_reading", "latitude_and_longitude", "musical_notation",
    "art_history", "the_scientific_method", "measurement_units",
]


class _HtmlItem(zw.Item):
    def __init__(self, path: str, title: str, content: str) -> None:
        super().__init__()
        self._path = path
        self._title = title
        self._content = content.encode("utf-8")

    def get_path(self) -> str:
        return self._path

    def get_title(self) -> str:
        return self._title

    def get_mimetype(self) -> str:
        return "text/html"

    def get_contentprovider(self) -> zw.StringProvider:
        return zw.StringProvider(self._content.decode("utf-8"))

    def get_hints(self) -> dict:
        return {zw.Hint.FRONT_ARTICLE: True}


def _simple_page(topic: str) -> str:
    title = topic.replace("_", " ").title()
    return (
        f"<html><head><title>{title}</title></head><body>"
        f"<h1>{title}</h1><h2>Overview</h2>"
        f"<p>{title} is a school topic. It links to "
        f'<a href="pythagorean_theorem">Pythagorean theorem</a>.</p>'
        f"<ul><li>Fact one</li><li>Fact two</li></ul>"
        f"</body></html>"
    )


def build_zim(
    zim_path: Path,
    *,
    indexing: bool = True,
    include_rich: bool = True,
    n_simple: int | None = None,
) -> Path:
    """Build a fixture ZIM at ``zim_path`` and return it.

    ``indexing=False`` builds the ``noindex_zim`` variant. ``n_simple``
    limits the number of generated simple articles (used for the small
    ``noindex_zim`` fixture).
    """
    zim_path = Path(zim_path)
    zim_path.parent.mkdir(parents=True, exist_ok=True)

    creator = zw.Creator(str(zim_path))
    creator.config_indexing(indexing, LANGUAGE)
    with creator:
        if include_rich:
            for path, html in _RICH_ARTICLES.items():
                creator.add_item(_HtmlItem(path, path.replace("_", " ").title(), html))

        topics = _SIMPLE_TOPICS if n_simple is None else _SIMPLE_TOPICS[:n_simple]
        for topic in topics:
            creator.add_item(_HtmlItem(topic, topic.replace("_", " ").title(), _simple_page(topic)))

        main_path = "pythagorean_theorem" if include_rich else (topics[0] if topics else None)
        if main_path is not None:
            creator.set_mainpath(main_path)

        if include_rich:
            # Simple redirect: redirect_a -> pythagorean_theorem (0 extra hops
            # beyond the redirect itself: resolving it is 1 hop).
            creator.add_redirection(
                "redirect_a", "Redirect A", "pythagorean_theorem", {}
            )
            # 2-hop chain: redirect_b -> redirect_a -> pythagorean_theorem.
            creator.add_redirection("redirect_b", "Redirect B", "redirect_a", {})

        creator.add_metadata("Title", "Fixture: school topics (English)")
        creator.add_metadata("Language", LANGUAGE)
        creator.add_metadata("Date", "2026-01-01")
        creator.add_metadata("Name", FIXTURE_NAME)
        creator.add_metadata("Creator", "kiwix-ai-tutor test suite")
        creator.add_metadata("Publisher", "kiwix-ai-tutor")
        creator.add_metadata("Description", "Synthetic fixture ZIM for WP-B1/WP-B2 tests.")

    return zim_path


def build_truncated_zim(source: Path, dest: Path, *, fraction: float = 0.6) -> Path:
    """Copy ``source`` to ``dest`` but cut it to ``fraction`` of its bytes."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = Path(source).read_bytes()
    cut = int(len(data) * fraction)
    dest.write_bytes(data[:cut])
    return dest


def build_not_a_zim(dest: Path) -> Path:
    """Write a plain text file with a ``.zim`` extension."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "This is not a ZIM archive. It is a plain text file renamed to "
        "end in .zim to test signature detection.\n",
        encoding="utf-8",
    )
    return dest


def copy_zim(source: Path, dest: Path) -> Path:
    """Byte-identical copy of ``source`` at ``dest`` (used for fingerprint tests)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return dest
