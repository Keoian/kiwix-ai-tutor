"""RED tests for tutor.retrieval.hybrid.passages (WP-B5, ours not vendored).

Module does not exist yet; every test here should fail with
ModuleNotFoundError until implemented. See docs/plan/
offline_tutor_implementation_plan.md WP-B5 ("Passage ID from fingerprint +
path + section + extractor version + text hash + span") and
offline_tutor_kiwix_reuse_plan.md line ~1293-1310.
"""

from __future__ import annotations

from tutor.retrieval.hybrid.passages import build_key_fact_passages, split_passages
from tutor.retrieval.zim.bundle import build_bundle

FINGERPRINT = "deadbeef" * 8  # 64 hex chars, arbitrary stand-in digest
ARCHIVE_ID = "fixture_en_school"


def _bundle(html: str, path: str = "p", title: str = "T"):
    return build_bundle(html, path=path, title=title)


SIMPLE_HTML = """
<html><head><title>Simple</title></head><body>
<h1>Simple</h1>
<p>Lead paragraph text that is reasonably short.</p>
<h2>Section One</h2>
<p>Body of section one. It has a couple of sentences. Here is another one.</p>
<h2>Section Two</h2>
<p>Body of section two, distinct content entirely from section one.</p>
</body></html>
"""

NESTED_HTML = """
<html><head><title>Nested</title></head><body>
<h1>Nested</h1>
<p>Lead.</p>
<h2>Parent</h2>
<p>Parent body text.</p>
<h3>Child</h3>
<p>Child body text, under the parent section, for heading path testing.</p>
</body></html>
"""

LONG_SECTION_HTML = """
<html><head><title>Long</title></head><body>
<h1>Long</h1>
<p>Lead.</p>
<h2>Big Section</h2>
<p>""" + (" ".join([f"Sentence number {i} in a long section." for i in range(200)])) + """</p>
</body></html>
"""

TINY_SECTION_HTML = """
<html><head><title>Tiny</title></head><body>
<h1>Tiny</h1>
<p>Lead.</p>
<h2>Tiny Section</h2>
<p>Short.</p>
</body></html>
"""

EMPTY_SECTION_HTML = """
<html><head><title>EmptySection</title></head><body>
<h1>EmptySection</h1>
<p>Lead.</p>
<h2>Nothing Here</h2>
<h2>Next</h2>
<p>Body of next section.</p>
</body></html>
"""


def test_split_passages_returns_list():
    b = _bundle(SIMPLE_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    assert isinstance(passages, list)
    assert len(passages) > 0


def test_passage_text_matches_bundle_slice_always():
    b = _bundle(SIMPLE_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    for p in passages:
        assert b.text[p.start:p.end] == p.text


def test_passages_never_cross_section_boundaries():
    b = _bundle(SIMPLE_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    section_by_index = {s.index: s for s in b.sections}
    for p in passages:
        sec = section_by_index[p.section_index]
        assert sec.start <= p.start
        assert p.end <= sec.end


def test_passages_tile_section_without_gaps_except_whitespace():
    b = _bundle(LONG_SECTION_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    big_section = next(s for s in b.sections if s.heading == "Big Section")
    in_section = sorted(
        (p for p in passages if p.section_index == big_section.index),
        key=lambda p: p.start,
    )
    assert len(in_section) > 1  # section is long enough to split
    head_gap = b.text[big_section.start:in_section[0].start]
    assert in_section[0].start == big_section.start or head_gap.strip() == ""
    for cur, nxt in zip(in_section, in_section[1:], strict=False):
        gap = b.text[cur.end:nxt.start]
        assert gap.strip() == ""
    tail_gap = b.text[in_section[-1].end:big_section.end]
    assert in_section[-1].end == big_section.end or tail_gap.strip() == ""


def test_heading_path_reflects_nesting():
    b = _bundle(NESTED_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    child_section = next(s for s in b.sections if s.heading == "Child")
    child_passages = [p for p in passages if p.section_index == child_section.index]
    assert child_passages
    for p in child_passages:
        assert p.heading_path == ("Parent", "Child")


def test_tiny_section_yields_a_passage():
    b = _bundle(TINY_SECTION_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    tiny_section = next(s for s in b.sections if s.heading == "Tiny Section")
    tiny_passages = [p for p in passages if p.section_index == tiny_section.index]
    assert len(tiny_passages) == 1


def test_whitespace_only_or_heading_only_section_yields_no_passages():
    b = _bundle(EMPTY_SECTION_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    empty_section = next(s for s in b.sections if s.heading == "Nothing Here")
    empty_passages = [p for p in passages if p.section_index == empty_section.index]
    assert empty_passages == []


def test_passage_fields_present():
    b = _bundle(SIMPLE_HTML, path="my_path", title="Simple")
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    p = passages[0]
    assert p.archive_id == ARCHIVE_ID
    assert p.path == "my_path"
    assert p.title == "Simple"
    assert isinstance(p.passage_id, str)
    assert isinstance(p.heading_path, tuple)


def test_passage_id_is_32_hex_chars():
    b = _bundle(SIMPLE_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    for p in passages:
        assert len(p.passage_id) == 32
        int(p.passage_id, 16)  # raises if not hex


def test_passage_id_deterministic_for_identical_inputs():
    b1 = _bundle(SIMPLE_HTML, path="p1")
    b2 = _bundle(SIMPLE_HTML, path="p1")
    p1 = split_passages(b1, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    p2 = split_passages(b2, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    assert [p.passage_id for p in p1] == [p.passage_id for p in p2]


def test_passage_id_changes_with_fingerprint_digest():
    b = _bundle(SIMPLE_HTML)
    p1 = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    other_fp = "cafebabe" * 8
    p2 = split_passages(b, fingerprint_digest=other_fp, archive_id=ARCHIVE_ID)
    assert p1[0].passage_id != p2[0].passage_id


def test_passage_id_changes_with_path():
    b1 = _bundle(SIMPLE_HTML, path="path_a")
    b2 = _bundle(SIMPLE_HTML, path="path_b")
    p1 = split_passages(b1, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    p2 = split_passages(b2, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    assert p1[0].passage_id != p2[0].passage_id


def test_passage_id_changes_with_archive_id():
    b = _bundle(SIMPLE_HTML)
    p1 = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id="archive_one")
    p2 = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id="archive_two")
    assert p1[0].passage_id != p2[0].passage_id


def test_passage_id_changes_with_extractor_version(monkeypatch):
    import tutor.retrieval.zim.bundle as bundle_mod

    b1 = _bundle(SIMPLE_HTML)
    p1 = split_passages(b1, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)

    monkeypatch.setattr(bundle_mod, "EXTRACTOR_VERSION", "zzz-different-version")
    b2 = build_bundle(SIMPLE_HTML, path="p", title="T")
    assert b2.extractor_version != b1.extractor_version
    p2 = split_passages(b2, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    assert p1[0].passage_id != p2[0].passage_id


def test_passage_id_changes_with_text_hash():
    # Two bundles differing only in body text (same path/section/version)
    # must yield different passage ids for the corresponding passage.
    html_a = SIMPLE_HTML
    html_b = SIMPLE_HTML.replace(
        "Body of section one. It has a couple of sentences. Here is another one.",
        "Completely different body content for section one entirely.",
    )
    ba = _bundle(html_a, path="same_path")
    bb = _bundle(html_b, path="same_path")
    pa = split_passages(ba, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    pb = split_passages(bb, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    sec_a = next(p for p in pa if "Section One" in p.heading_path)
    sec_b = next(p for p in pb if "Section One" in p.heading_path)
    assert sec_a.passage_id != sec_b.passage_id


HELIUM_HTML = """
<html><head><title>Helium</title></head><body>
<h1>Helium</h1>
<table class="infobox">
<tr><th>Symbol</th><td>He</td></tr>
<tr><th>Melting point</th><td>0.95 K</td></tr>
<tr><th>Boiling point</th><td>4.222 K (-268.928 C, -452.070 F)</td></tr>
<tr><th>Density</th><td>0.1786 g/L</td></tr>
<tr><th>Discovered by</th><td>Pierre Janssen</td></tr>
</table>
<h2>Overview</h2>
<p>Helium is a chemical element that is a colourless, odourless gas.</p>
</body></html>
"""

NO_INFOBOX_HTML = SIMPLE_HTML


def test_key_fact_single_line_match():
    b = _bundle(HELIUM_HTML, path="helium", title="Helium")
    passages = build_key_fact_passages(
        b, frozenset({"boiling"}), fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID
    )
    assert len(passages) == 1
    p = passages[0]
    assert p.text == "Boiling point: 4.222 K (-268.928 C, -452.070 F)"
    assert p.heading_path == ("Infobox",)
    assert b.text[p.start:p.end] == p.text


def test_key_fact_no_match_returns_empty():
    b = _bundle(HELIUM_HTML, path="helium", title="Helium")
    passages = build_key_fact_passages(
        b, frozenset({"xylophone"}), fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID
    )
    assert passages == []


def test_key_fact_no_infobox_returns_empty():
    b = _bundle(NO_INFOBOX_HTML)
    passages = build_key_fact_passages(
        b, frozenset({"section"}), fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID
    )
    assert passages == []


def test_key_fact_multi_line_run_is_one_passage():
    # "point" matches both "Melting point" and "Boiling point", which are
    # adjacent infobox rows -- one contiguous passage, not two.
    b = _bundle(HELIUM_HTML, path="helium", title="Helium")
    passages = build_key_fact_passages(
        b, frozenset({"point"}), fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID
    )
    assert len(passages) == 1
    p = passages[0]
    assert p.text == "Melting point: 0.95 K\nBoiling point: 4.222 K (-268.928 C, -452.070 F)"
    assert b.text[p.start:p.end] == p.text


def test_key_fact_non_contiguous_matches_yield_separate_passages():
    # "Symbol" and "Discovered by" both match on "d"-free terms picked so
    # they are non-adjacent rows -- two separate runs/passages.
    b = _bundle(HELIUM_HTML, path="helium", title="Helium")
    passages = build_key_fact_passages(
        b,
        frozenset({"symbol", "discovered"}),
        fingerprint_digest=FINGERPRINT,
        archive_id=ARCHIVE_ID,
    )
    assert len(passages) == 2
    assert passages[0].text == "Symbol: He"
    assert passages[1].text == "Discovered by: Pierre Janssen"
    for p in passages:
        assert b.text[p.start:p.end] == p.text


def test_key_fact_offsets_round_trip_for_all_rows():
    b = _bundle(HELIUM_HTML, path="helium", title="Helium")
    all_terms = frozenset(
        {"symbol", "melting", "boiling", "point", "density", "discovered"}
    )
    passages = build_key_fact_passages(
        b, all_terms, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID
    )
    assert passages
    for p in passages:
        assert b.text[p.start:p.end] == p.text
        assert p.passage_id and len(p.passage_id) == 32


def test_key_fact_max_lines_budget_respected():
    b = _bundle(HELIUM_HTML, path="helium", title="Helium")
    all_terms = frozenset(
        {"symbol", "melting", "boiling", "point", "density", "discovered"}
    )
    passages = build_key_fact_passages(
        b,
        all_terms,
        fingerprint_digest=FINGERPRINT,
        archive_id=ARCHIVE_ID,
        max_lines=2,
    )
    total_lines = sum(p.text.count("\n") + 1 for p in passages)
    assert total_lines <= 2


def test_passage_id_changes_with_span():
    # A section split into two passages must give each a distinct id,
    # even sharing path/section_index/version/fingerprint (span differs).
    b = _bundle(LONG_SECTION_HTML)
    passages = split_passages(b, fingerprint_digest=FINGERPRINT, archive_id=ARCHIVE_ID)
    big_section = next(s for s in b.sections if s.heading == "Big Section")
    in_section = [p for p in passages if p.section_index == big_section.index]
    assert len(in_section) >= 2
    ids = {p.passage_id for p in in_section}
    assert len(ids) == len(in_section)
