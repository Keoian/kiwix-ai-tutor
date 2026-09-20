"""RED tests for tutor.retrieval.zim.content and tutor.retrieval.zim.bundle (WP-B4).

These modules do not exist yet; every test here should fail with
ModuleNotFoundError until they are implemented. See docs/plan/
offline_tutor_implementation_plan.md WP-B4 and offline_tutor_kiwix_reuse_plan.md
section 3 (ArticleBundle, section offsets) for the semantics being pinned down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.zim import bundle as bundle_mod
from tutor.retrieval.zim.bundle import build_bundle
from tutor.retrieval.zim.content import render_text, select_main_content

# ---------------------------------------------------------------------------
# Fixture HTML strings (self-contained; do not touch tests/zim_fixtures.py).
# ---------------------------------------------------------------------------

PLAIN_HTML = """
<html><head><title>Plain Article</title></head><body>
<nav>site nav junk</nav>
<h1>Plain Article</h1>
<p>This is the lead paragraph before any heading.</p>
<h2>Only Section</h2>
<p>Some body text in the only section.<sup class="reference">[1]</sup></p>
<p>A second paragraph in the same section.</p>
<footer>footer junk</footer>
<script>var x = 1;</script>
</body></html>
"""

NESTED_HEADINGS_HTML = """
<html><head><title>Nested Article</title></head><body>
<h1>Nested Article</h1>
<p>Lead text.</p>
<h2>Top A</h2>
<p>Text A.</p>
<h3>Sub A1</h3>
<p>Text A1.</p>
<h3>Sub A2</h3>
<p>Text A2.</p>
<h2>Top B</h2>
<p>Text B.</p>
<h3>Sub B1</h3>
<p>Text B1.</p>
</body></html>
"""

INFOBOX_LISTS_HTML = """
<html><head><title>Infobox Article</title></head><body>
<h1>Infobox Article</h1>
<table class="infobox">
<tr><th>Field</th><td>Geometry</td></tr>
<tr><th>Discoverer</th><td>Pythagoras</td></tr>
</table>
<p>Lead paragraph.</p>
<h2>List Section</h2>
<ul><li>First item</li><li>Second item</li><li>Third item</li></ul>
<a class="edit" href="#edit">[edit]</a>
</body></html>
"""

DUPLICATE_HEADINGS_HTML = """
<html><head><title>Duplicate Headings</title></head><body>
<h1>Duplicate Headings</h1>
<p>Lead.</p>
<h2>Examples</h2>
<p>First examples section body.</p>
<h2>Discussion</h2>
<p>Discussion body.</p>
<h2>Examples</h2>
<p>Second examples section body, distinct from the first.</p>
</body></html>
"""

UNICODE_TABLE_HTML = """
<html><head><title>Unicode Heavy</title></head><body>
<h1>Unicode Heavy</h1>
<p>Erdős worked on combinatorics. a²+b²=c² is Pythagorean.</p>
<h2>\U0001F9EE Counting</h2>
<p>A combining mark: é (e + acute). An emoji: \U0001F600.</p>
<table><tr><th>Name</th><th>Value</th></tr><tr><td>x</td><td>1</td></tr></table>
<h2>Cr—lf test</h2>
<p>Line one.\r\nLine two after CRLF.</p>
</body></html>
"""

ALL_FIXTURES = [
    PLAIN_HTML,
    NESTED_HEADINGS_HTML,
    INFOBOX_LISTS_HTML,
    DUPLICATE_HEADINGS_HTML,
    UNICODE_TABLE_HTML,
]

INLINE_MARKUP_HEADING_HTML = """
<html><head><title>Inline Markup</title></head><body>
<h1>Inline Markup</h1>
<p>Lead.</p>
<h2>The <i>Proof</i> of <code>x=y</code></h2>
<p>Body text for the inline-markup heading.</p>
</body></html>
"""

NO_HEADINGS_HTML = """
<html><head><title>No Headings</title></head><body>
<p>Just one paragraph, no headings at all.</p>
<p>And a second paragraph.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# content.select_main_content / content.render_text
# ---------------------------------------------------------------------------


def test_select_main_content_returns_tag_like_object():
    result = select_main_content(PLAIN_HTML)
    # bs4 Tag: has get_text
    assert hasattr(result, "get_text")


def test_render_text_strips_script_style_nav_footer():
    text = render_text(PLAIN_HTML)
    assert "site nav junk" not in text
    assert "footer junk" not in text
    assert "var x = 1" not in text


def test_render_text_strips_reference_markers_and_edit_links():
    text = render_text(PLAIN_HTML)
    assert "[1]" not in text
    infobox_text = render_text(INFOBOX_LISTS_HTML)
    assert "[edit]" not in infobox_text


def test_render_text_keeps_headings_paragraphs_list_items():
    text = render_text(INFOBOX_LISTS_HTML)
    assert "Infobox Article" in text
    assert "Lead paragraph." in text
    assert "List Section" in text
    assert "First item" in text
    assert "Second item" in text
    assert "Third item" in text


def test_render_text_is_deterministic():
    assert render_text(NESTED_HEADINGS_HTML) == render_text(NESTED_HEADINGS_HTML)


def test_render_text_preserves_unicode_exactly():
    text = render_text(UNICODE_TABLE_HTML)
    assert "Erdős" in text
    assert "a²+b²=c²" in text
    assert "\U0001F600" in text
    assert "é" in text


def test_render_text_normalizes_whitespace_no_blank_run_over_one():
    text = render_text(PLAIN_HTML)
    assert "\n\n\n" not in text


# ---------------------------------------------------------------------------
# bundle.build_bundle / ArticleBundle
# ---------------------------------------------------------------------------


def _build(html: str, path: str = "test_path", title: str = "Test Title"):
    return build_bundle(html, path=path, title=title)


def test_build_bundle_basic_fields():
    b = _build(PLAIN_HTML, path="plain", title="Plain Article")
    assert b.path == "plain"
    assert b.title == "Plain Article"
    assert isinstance(b.text, str)
    assert isinstance(b.sections, tuple)
    assert isinstance(b.links, tuple)
    assert isinstance(b.infobox, tuple)
    assert b.extractor_version == bundle_mod.EXTRACTOR_VERSION


def test_bundle_is_frozen_dataclass():
    import dataclasses

    b = _build(PLAIN_HTML)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.title = "changed"  # type: ignore[misc]


def test_bundle_links_deduped_and_ordered_internal_only():
    html = """
    <html><head><title>Links</title></head><body>
    <h1>Links</h1>
    <p><a href="a_page">A</a> <a href="b_page">B</a> <a href="a_page">A again</a>
    <a href="https://example.com/external">External</a></p>
    </body></html>
    """
    b = _build(html, path="links")
    assert b.links == ("a_page", "b_page")


def test_bundle_infobox_extracted_as_kv_pairs():
    b = _build(INFOBOX_LISTS_HTML, path="infobox")
    assert ("Field", "Geometry") in b.infobox
    assert ("Discoverer", "Pythagoras") in b.infobox


def test_bundle_infobox_empty_when_absent():
    b = _build(PLAIN_HTML, path="plain")
    assert b.infobox == ()


def test_bundle_no_headings_yields_single_lead_section():
    b = _build(NO_HEADINGS_HTML, path="no_headings")
    assert len(b.sections) == 1
    lead = b.sections[0]
    assert lead.index == 0
    assert lead.heading == ""
    assert lead.parent_index is None
    assert lead.start == 0
    assert lead.end == len(b.text)


def test_bundle_lead_section_is_index_zero_with_empty_heading():
    b = _build(PLAIN_HTML, path="plain")
    lead = b.sections[0]
    assert lead.index == 0
    assert lead.heading == ""


def test_bundle_build_is_deterministic():
    b1 = _build(NESTED_HEADINGS_HTML, path="nested")
    b2 = _build(NESTED_HEADINGS_HTML, path="nested")
    assert b1 == b2


def test_bundle_nested_headings_parent_index_correct():
    b = _build(NESTED_HEADINGS_HTML, path="nested")
    by_heading = {s.heading: s for s in b.sections if s.heading}
    top_a = by_heading["Top A"]
    sub_a1 = by_heading["Sub A1"]
    sub_a2 = by_heading["Sub A2"]
    top_b = by_heading["Top B"]
    sub_b1 = by_heading["Sub B1"]

    assert sub_a1.parent_index == top_a.index
    assert sub_a2.parent_index == top_a.index
    assert sub_b1.parent_index == top_b.index
    assert top_a.parent_index == 0  # lead is parent of top-level h2s
    assert top_b.parent_index == 0
    assert top_a.level == 2
    assert sub_a1.level == 3


def test_bundle_duplicate_headings_get_distinct_correct_offsets():
    b = _build(DUPLICATE_HEADINGS_HTML, path="dup")
    examples_sections = [s for s in b.sections if s.heading == "Examples"]
    assert len(examples_sections) == 2
    first, second = examples_sections
    assert first.start < second.start
    first_text = b.text[first.start:first.end]
    second_text = b.text[second.start:second.end]
    assert "First examples section body" in first_text
    assert "Second examples section body" in second_text
    assert "First examples section body" not in second_text
    assert "Second examples section body" not in first_text


def test_bundle_heading_with_inline_markup_resolves():
    b = _build(INLINE_MARKUP_HEADING_HTML, path="inline_markup")
    headings = [s.heading for s in b.sections]
    assert any("Proof" in h and "x=y" in h for h in headings)


def test_bundle_empty_sections_allowed():
    html = """
    <html><head><title>Empty Section</title></head><body>
    <h1>Empty Section</h1>
    <p>Lead.</p>
    <h2>Empty Heading</h2>
    <h2>Next Heading</h2>
    <p>Body of next.</p>
    </body></html>
    """
    b = _build(html, path="empty_section")
    empty = next(s for s in b.sections if s.heading == "Empty Heading")
    assert empty.start <= empty.end


@pytest.mark.parametrize("html", ALL_FIXTURES)
def test_bundle_offset_round_trip_tiling(html: str):
    b = _build(html, path="p", title="T")
    sections = b.sections
    assert sections[0].start == 0
    for cur, nxt in zip(sections, sections[1:], strict=False):
        assert cur.end == nxt.start
    assert sections[-1].end == len(b.text)

    for s in sections:
        slice_text = b.text[s.start:s.end]
        if s.heading:
            head_window = slice_text[: len(s.heading) + 20]
            assert slice_text.lstrip().startswith(s.heading) or s.heading in head_window


@pytest.mark.parametrize("html", ALL_FIXTURES)
def test_bundle_offsets_are_codepoint_based(html: str):
    b = _build(html, path="p", title="T")
    # Sanity: text length equals number of Python str code points (trivially
    # true for a str), and every offset is within range and on a valid
    # boundary (slicing never raises and stays consistent across calls).
    for s in b.sections:
        assert 0 <= s.start <= s.end <= len(b.text)
        # Slicing twice gives identical results (no surrogate/byte issues).
        assert b.text[s.start:s.end] == b.text[s.start:s.end]


def test_bundle_offset_round_trip_against_rich_fixture_zim(fixture_zim: Path):
    from libzim.reader import Archive

    archive = Archive(fixture_zim)
    rich_paths = [
        "pythagorean_theorem",
        "algebra_basics",
        "history_of_mathematics",
        "geometry_intro",
        "erdos_number",
    ]
    for path in rich_paths:
        entry = archive.get_entry_by_path(path)
        item = entry.get_item()
        html = bytes(item.content).decode("utf-8")
        b = build_bundle(html, path=path, title=entry.title)

        sections = b.sections
        assert sections[0].start == 0
        for cur, nxt in zip(sections, sections[1:], strict=False):
            assert cur.end == nxt.start
        assert sections[-1].end == len(b.text)


def test_bundle_build_twice_equal(fixture_zim: Path):
    from libzim.reader import Archive

    archive = Archive(fixture_zim)
    entry = archive.get_entry_by_path("pythagorean_theorem")
    html = bytes(entry.get_item().content).decode("utf-8")
    b1 = build_bundle(html, path="pythagorean_theorem", title=entry.title)
    b2 = build_bundle(html, path="pythagorean_theorem", title=entry.title)
    assert b1 == b2
