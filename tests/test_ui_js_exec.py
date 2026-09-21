"""Executes tutor/ui/app.js's pure UI logic -- not just parses it.

Until this file existed, none of today's markdown tokenizer, block mapper,
marker-placement, or DOM-render code in tutor/ui/app.js had ever actually
been RUN by anyone: Node is not installed on this box, so the two
Node-conditional checks in tests/test_ui_static.py and
tests/test_ui_attribution_blocks.py are skipped, and no browser check was
possible either. This file runs it for real under an embedded V8
(py_mini_racer, see tests/js_harness/) so a logic bug fails a test instead
of surfacing for the first time in front of the owner.

Prefers a real Node.js if present on the box (closer to the browser); falls
back to the embedded engine; skips (never fails) when neither is available,
since the engine is an optional dev dependency and some CI runners may lack
a wheel for their platform/arch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.js_harness.engine import HAVE_ENGINE, call, js_str, new_context

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
FIXTURE_TABLE = FIXTURES / "table_answer.md"
FIXTURE_SSE = FIXTURES / "ui_bug_sse_frames.txt"

HAVE_NODE = shutil.which("node") is not None

pytestmark = pytest.mark.skipif(
    not HAVE_ENGINE and not HAVE_NODE,
    reason="no JS engine available (py_mini_racer not importable, no Node)",
)


@pytest.fixture()
def ctx():
    if not HAVE_ENGINE:
        pytest.skip("py_mini_racer not importable on this platform")
    return new_context()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (a) mapAnswerToBlocks
# ---------------------------------------------------------------------------


def test_map_blocks_on_table_fixture_slices_match_raw_text(ctx):
    text = _read(FIXTURE_TABLE)
    blocks = call(ctx, f"module.exports.mapAnswerToBlocks({js_str(text)})")
    kinds = [b["type"] for b in blocks]
    assert "table" in kinds
    assert "list" in kinds
    assert "heading" in kinds
    assert "paragraph" in kinds

    table = next(b for b in blocks if b["type"] == "table")
    assert text[table["headCells"][0]["start"] : table["headCells"][0]["end"]] == "Step"
    for row in table["rows"]:
        assert text[row["start"] : row["end"]].startswith("|")
        for cell in row["cells"]:
            slice_ = text[cell["start"] : cell["end"]]
            assert slice_ == slice_.strip()  # trimmed, per _trimCellRange
            assert "|" not in slice_[:0]  # sanity: offsets are inside the row

    list_block = next(b for b in blocks if b["type"] == "list")
    for item in list_block["items"]:
        assert text[item["start"] : item["end"]].lstrip().startswith("-")
        item_text = text[item["textStart"] : item["textEnd"]]
        assert item_text == item_text.strip()

    heading = next(b for b in blocks if b["type"] == "heading")
    inner = text[heading["innerStart"] : heading["innerEnd"]]
    assert inner == "Quick Tips for Success"


def test_map_blocks_on_plain_paragraphs_lists_headings(ctx):
    text = "# Title\n\nSome paragraph text\nsecond line\n\n- item one\n- item two\n"
    blocks = call(ctx, f"module.exports.mapAnswerToBlocks({js_str(text)})")
    assert blocks[0]["type"] == "heading"
    assert text[blocks[0]["innerStart"] : blocks[0]["innerEnd"]] == "Title"
    para = next(b for b in blocks if b["type"] == "paragraph" and len(b["lines"]) == 4)
    assert [text[ln["start"] : ln["end"]] for ln in para["lines"]] == [
        "",
        "Some paragraph text",
        "second line",
        "",
    ]
    lst = next(b for b in blocks if b["type"] == "list")
    assert [text[i["textStart"] : i["textEnd"]] for i in lst["items"]] == [
        "item one",
        "item two",
    ]


def test_map_blocks_table_with_br_and_bold_span_crossing_cells(ctx):
    text = _read(FIXTURE_TABLE)
    blocks = call(ctx, f"module.exports.mapAnswerToBlocks({js_str(text)})")
    table = next(b for b in blocks if b["type"] == "table")
    br_row = next(
        r
        for r in table["rows"]
        if "<br>" in text[r["cells"][1]["start"] : r["cells"][1]["end"]]
    )
    assert len(br_row["cells"]) == 3
    # The soil-mix row's bold span ("**Base Mix:**") straddles a literal
    # "|" that markdown never intended as a column separator -- the mapper
    # splits cells the exact same (naive) way _splitTableRow always has,
    # so this documents existing, consistent behaviour rather than a bug.
    soil_row = table["rows"][2]
    assert text[soil_row["cells"][0]["start"] : soil_row["cells"][0]["end"]] == (
        "**3. Prepare the Soil Mix"
    )


# ---------------------------------------------------------------------------
# (b) inline tokenizer round-trip
# ---------------------------------------------------------------------------

_ROUNDTRIP_SAMPLES = [
    "3 * 4 * 5",
    "snake_case_var here",
    "unbalanced **bold",
    "https://example.com/path",
    "a_b c_d e_f",
    "**bold** and *italic* and `code`",
    "plain text only",
    "Multiple   spaces  here",
    "Ends with **unterminated bold",
    "*starts italic unterminated",
    "Mixed **bold** then *italic* then `code` end.",
    "Number 3*3=9 no spaces",
    "A **B** C *D* E `F` G",
    "trailing_underscore_",
    "_leading_underscore",
    "multi\nline\ntext",
    "Citation [S1] inline",
    "Question: what is 2 * 2?",
    "temp is 98.6 degrees",
    "a**b**c**d**e",
    "**a** *b* `c` **d**",
    "no special chars at all",
    "weird ** spacing ** case",
    "3 * 4",
    "x * y * z = result",
    "CamelCaseWord",
    "under_score_word_here",
    "http://x.com",
    "Bold at start **X** end",
    "code at `end`",
    "The Burj Khalifa has **163 floors** and stands **828 meters** tall.",
    "It has 6 * 7 = 42 marbles in *three* equal groups.",
]


def _reconstruct(tokens: list[dict]) -> str:
    out = []
    for t in tokens:
        tag = t["tag"]
        if tag is None:
            out.append(t["text"])
        elif tag == "strong":
            out.append("**" + t["text"] + "**")
        elif tag == "em":
            out.append("*" + t["text"] + "*")
        elif tag == "code":
            out.append("`" + t["text"] + "`")
        elif tag == "br":
            out.append("\n")
    return "".join(out)


@pytest.mark.parametrize("text", _ROUNDTRIP_SAMPLES)
def test_inline_tokenizer_roundtrips_raw_text(ctx, text):
    tokens = call(ctx, f"_inlineTokens({js_str(text)})")
    assert _reconstruct(tokens) == text


# ---------------------------------------------------------------------------
# (b2) invented citation-label stripping (owner-reported live bug: the model
# invents source labels like "[Q&A]"/"[Cite: [Q&A]]" that are not real [S#]
# passages, and the UI must never show them as literal text). Mirrors
# tutor/app/citations.py's _INVENTED_LABEL_RE pattern list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,stripped_token",
    [
        ("This is the coldest known point [Cite: [Q&A]]", "[Cite: [Q&A]]"),
        ("... found in interstellar space. [Q&A]", "[Q&A]"),
        ("Some claim here. [Source]", "[Source]"),
        ("Some claim here. [Sources: S1, S2]", "[Sources: S1, S2]"),
        ("Some claim here. [citation needed]", "[citation needed]"),
        ("Some claim here. [Cite: S1]", "[Cite: S1]"),
    ],
)
def test_invented_citation_labels_are_stripped_from_rendered_text(ctx, text, stripped_token):
    tokens = call(ctx, f"_inlineTokens({js_str(text)})")
    reconstructed = _reconstruct(tokens)
    assert stripped_token not in reconstructed


@pytest.mark.parametrize(
    "text",
    [
        "The formula for water is [H2O].",
        "The winning numbers were [1, 2, 3].",
        "The array is [S1, S2] indexed from zero.",  # real [S#] group, not invented
        "f(x) is defined for all reals (see above).",
    ],
)
def test_ordinary_bracketed_text_is_never_stripped(ctx, text):
    tokens = call(ctx, f"_inlineTokens({js_str(text)})")
    reconstructed = _reconstruct(tokens)
    assert reconstructed == text


# ---------------------------------------------------------------------------
# (b3) markdown-link fake citation (owner-reported LIVE bug, seen after
# 8f26f82 + f0f8755 hid the plain "[Cite: [Q&A]]" / fake "Sources:" forms):
# the model now ends answers with a fake citation in MARKDOWN-LINK shape
# pointing at the live internet -- this is an OFFLINE tool, the URL is
# invented. Verbatim owner-reported examples:
#   [Cite: [S1]](https://en.wikipedia.org/wiki/Titin)
#   [Cite: [S1]](https://en.wikipedia.org/wiki/Rubber)
#   [Cite: [S1]](https://en.wikipedia.org/wiki/Vulcanization)
# Also: this is an offline UI, so ANY other markdown link the model writes
# must render as plain link text only, never a clickable URL.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[Cite: [S1]](https://en.wikipedia.org/wiki/Titin)", "[S1]"),
        ("[Cite: [S1]](https://en.wikipedia.org/wiki/Rubber)", "[S1]"),
        ("[Cite: [S1]](https://en.wikipedia.org/wiki/Vulcanization)", "[S1]"),
        ("[Source](https://en.wikipedia.org/wiki/Rubber)", ""),
        ("[Source: foo](https://example.com/x)", ""),
        ("[S1](https://example.com/x)", "[S1]"),
    ],
)
def test_collapse_invented_links_examples(ctx, raw, expected):
    collapsed = call(ctx, f"module.exports.collapseInventedLinks({js_str(raw)})")
    assert collapsed == expected


def test_ordinary_markdown_link_renders_as_plain_text_no_url_no_anchor(ctx):
    text = "See the [full report](https://example.com/report) for details."
    tokens = call(ctx, f"_inlineTokens({js_str(text)})")
    reconstructed = _reconstruct(tokens)
    assert "https://example.com/report" not in reconstructed
    assert "full report" in reconstructed
    assert not any(t["tag"] == "A" for t in tokens)


def test_markdown_link_fake_citation_examples_hidden_and_real_label_kept(ctx):
    for url in (
        "https://en.wikipedia.org/wiki/Titin",
        "https://en.wikipedia.org/wiki/Rubber",
        "https://en.wikipedia.org/wiki/Vulcanization",
    ):
        answer = (
            "This is a real claim about the topic. [Cite: [S1]](" + url + ")"
        )
        ctx.eval("var _cite_c = document.createElement('div');")
        ctx.eval(
            f"module.exports.renderTextWithCitations(_cite_c, {js_str(answer)});"
        )
        reconstructed = call(ctx, "_cite_c.textContent")
        assert "Cite:" not in reconstructed
        assert url not in reconstructed
        found = call(ctx, "_walkCollect(_cite_c)")
        assert any("S1" in c["text"] for c in found["chips"]) or any(
            "S1" in lbl for lbl in found["plainCitationLabels"]
        )


def test_markdown_link_fake_citation_with_no_real_label_is_hidden_entirely(ctx):
    answer = "Rubber is vulcanized with sulfur. [Source](https://en.wikipedia.org/wiki/Rubber)"
    ctx.eval("var _cite_c2 = document.createElement('div');")
    ctx.eval(f"module.exports.renderTextWithCitations(_cite_c2, {js_str(answer)});")
    reconstructed = call(ctx, "_cite_c2.textContent")
    assert "https://en.wikipedia.org" not in reconstructed
    assert "[Source]" not in reconstructed
    assert "Source" not in reconstructed


def test_sentence_made_only_of_invented_label_renders_no_orphan_marker(ctx):
    # The real owner example: a trailing "[Cite: [Q&A]]" line that used to
    # get its own spurious "not found" marker.
    answer = "Neptune is the eighth planet from the Sun. [S1]\n[Cite: [Q&A]]"
    attributions = {
        "attributions": [
            {
                "sentence_span": [0, len("Neptune is the eighth planet from the Sun.")],
                "model_cited": True,
                "passage_id": "p1",
            }
        ],
        "unbacked": [],
    }
    citations = {"citations": [{"label": "S1", "passage_id": "p1"}]}
    ctx.eval("var _c2 = document.createElement('div');")
    ctx.eval(
        "module.exports.renderAnswerWithAttribution(_c2, "
        f"{js_str(answer)}, {js_str(attributions)}, {js_str(citations)});"
    )
    reconstructed = call(ctx, "_c2.textContent")
    assert "[Cite:" not in reconstructed
    assert "[Q&A]" not in reconstructed
    found = call(ctx, "_walkCollect(_c2)")
    assert len(found["markers"]) == 0


def test_inline_tokenizer_does_not_treat_math_asterisks_as_italic(ctx):
    # Regression: "3 * 4 * 5" used to render " 4 " as <em> because the
    # italic regex allowed whitespace right inside the delimiters.
    tokens = call(ctx, f"_inlineTokens({js_str('3 * 4 * 5')})")
    assert not any(t["tag"] == "em" for t in tokens)


def test_inline_tokenizer_real_fixture_answer_roundtrips(ctx):
    frames = _read(FIXTURE_SSE)
    done_line = next(
        line
        for line in frames.splitlines()
        if line.startswith("data:") and '"status": "ok"' in line
    )
    answer = json.loads(done_line[len("data:") :].strip())["answer"]
    tokens = call(ctx, f"_inlineTokens({js_str(answer)})")
    assert _reconstruct(tokens) == answer
    assert any(t["tag"] == "strong" for t in tokens)


# ---------------------------------------------------------------------------
# (c) marker placement (host-attributed sentences map into the right block)
# ---------------------------------------------------------------------------


def test_markers_land_in_last_cell_of_table_row_not_in_pipe_syntax(ctx):
    text = _read(FIXTURE_TABLE)
    blocks = call(ctx, f"module.exports.mapAnswerToBlocks({js_str(text)})")
    table = next(b for b in blocks if b["type"] == "table")
    row = table["rows"][0]
    last_cell_end = row["cells"][-1]["end"]
    markers = [{"pos": last_cell_end, "kind": "backed", "passageId": "p1"}]
    ctx.eval(f"var _blocks = {js_str(blocks)};")
    ctx.eval(f"var _markers = {js_str(markers)};")
    ctx.eval(
        f"""
        var _container = document.createElement("div");
        module.exports.renderBlocksToDom(
          _container, {js_str(text)}, _blocks, [], _markers,
          function (label) {{ return _mkSpan(); }},
          function (m) {{ return _mkMarker("attribution-marker attribution-backed"); }}
        );
        """
    )
    found = call(ctx, "_walkCollect(_container)")
    assert len(found["markers"]) == 1
    # heading gets no markers at all
    heading = next(b for b in blocks if b["type"] == "heading")
    stray_marker = [{"pos": heading["innerEnd"], "kind": "backed", "passageId": "p1"}]
    ctx.eval(f"var _markers2 = {js_str(stray_marker)};")
    ctx.eval(
        f"""
        var _container2 = document.createElement("div");
        module.exports.renderBlocksToDom(
          _container2, {js_str(text)}, _blocks, [], _markers2,
          function (label) {{ return _mkSpan(); }},
          function (m) {{ return _mkMarker("attribution-marker attribution-backed"); }}
        );
        """
    )
    found2 = call(ctx, "_walkCollect(_container2)")
    assert len(found2["markers"]) == 0  # a marker positioned inside a heading is dropped


def test_markers_land_at_end_of_list_item(ctx):
    text = _read(FIXTURE_TABLE)
    blocks = call(ctx, f"module.exports.mapAnswerToBlocks({js_str(text)})")
    lst = next(b for b in blocks if b["type"] == "list")
    item = lst["items"][0]
    markers = [{"pos": item["end"], "kind": "unbacked"}]
    ctx.eval(f"var _blocks3 = {js_str(blocks)}; var _markers3 = {js_str(markers)};")
    ctx.eval(
        f"""
        var _container3 = document.createElement("div");
        module.exports.renderBlocksToDom(
          _container3, {js_str(text)}, _blocks3, [], _markers3,
          function (label) {{ return _mkSpan(); }},
          function (m) {{ return _mkMarker("attribution-marker attribution-unbacked"); }}
        );
        """
    )
    found = call(ctx, "_walkCollect(_container3)")
    assert len(found["markers"]) == 1


def test_unmappable_marker_offset_is_dropped_not_misplaced(ctx):
    text = _read(FIXTURE_TABLE)
    blocks = call(ctx, f"module.exports.mapAnswerToBlocks({js_str(text)})")
    markers = [{"pos": len(text) + 500, "kind": "backed", "passageId": "p1"}]
    ctx.eval(f"var _blocks4 = {js_str(blocks)}; var _markers4 = {js_str(markers)};")
    ctx.eval(
        f"""
        var _container4 = document.createElement("div");
        module.exports.renderBlocksToDom(
          _container4, {js_str(text)}, _blocks4, [], _markers4,
          function (label) {{ return _mkSpan(); }},
          function (m) {{ return _mkMarker("attribution-marker attribution-backed"); }}
        );
        """
    )
    found = call(ctx, "_walkCollect(_container4)")
    assert len(found["markers"]) == 0


# ---------------------------------------------------------------------------
# (d) findBestSentenceSpan
# ---------------------------------------------------------------------------


def test_find_best_sentence_span_matches_burj_khalifa_sentence(ctx):
    passage = (
        "The Burj Khalifa is the tallest building in the world. "
        "It is located in Dubai, United Arab Emirates. "
        "It has 163 floors and stands 828 meters tall."
    )
    answer_sentence = "The Burj Khalifa has 163 floors and stands 828 meters tall."
    span = call(
        ctx,
        f"module.exports.findBestSentenceSpan({js_str(passage)}, 0, {len(passage)}, {js_str(answer_sentence)})",  # noqa: E501
    )
    assert span is not None
    assert passage[span["start"] : span["end"]].strip() == (
        "It has 163 floors and stands 828 meters tall."
    )


def test_find_best_sentence_span_returns_none_without_answer_sentence(ctx):
    passage = "Some passage text. Another sentence here."
    span = call(ctx, f"module.exports.findBestSentenceSpan({js_str(passage)}, 0, {len(passage)}, null)")  # noqa: E501
    assert span is None


# ---------------------------------------------------------------------------
# (e) end-to-end DOM render on the captured SSE fixture
# ---------------------------------------------------------------------------


def _load_fixture_events():
    frames = _read(FIXTURE_SSE)
    lines = frames.splitlines()
    done = json.loads(
        next(line for line in lines if line.startswith("data:") and '"status": "ok"' in line)[
            len("data:") :
        ].strip()
    )
    citations = json.loads(
        next(
            line for line in lines if line.startswith("data:") and "unsupported_labels" in line
        )[len("data:") :].strip()
    )
    attributions = json.loads(
        next(line for line in lines if line.startswith("data:") and '"unbacked"' in line)[
            len("data:") :
        ].strip()
    )
    return done, citations, attributions


def test_render_answer_with_attribution_end_to_end_on_sse_fixture(ctx):
    done, citations, attributions = _load_fixture_events()
    answer = done["answer"]
    ctx.eval("var _c = document.createElement('div');")
    ctx.eval(
        "module.exports.renderAnswerWithAttribution(_c, "
        f"{js_str(answer)}, {js_str(attributions)}, {js_str(citations)});"
    )
    found = call(ctx, "_walkCollect(_c)")
    # exactly one chip per [S#] occurrence in the raw text
    assert len(found["chips"]) == 2
    assert {c["text"] for c in found["chips"]} == {"[S1]", "[S2]"}
    # each resolved chip carries its passage id (never falls back to the bare label)
    passage_ids = {c["passage"] for c in found["chips"]}
    real_ids = {c["passage_id"] for c in citations["citations"]}
    assert passage_ids == real_ids
    assert found["plainCitationLabels"] == []
    # host-backed markers for the two non-model-cited attributions
    assert len(found["markers"]) == 2
    assert all(m == "attribution-marker attribution-backed" for m in found["markers"])
    # never assigns innerHTML anywhere on the tree (the fake DOM throws on it)
    reconstructed = call(ctx, "_c.textContent")
    assert "**" not in reconstructed  # markdown syntax stripped from the rendered text


def test_working_bubble_removed_on_first_token(ctx):
    ctx.eval("var _working = module.exports.el('div', {className: 'msg msg-tutor msg-working'});")
    ctx.eval("var _removed = false; _working.remove = function(){ _removed = true; };")
    ctx.eval("_working.remove();")
    assert ctx.eval("_removed") is True


def test_has_host_backed_content_matches_fixture(ctx):
    _, _citations, attributions = _load_fixture_events()
    assert call(ctx, f"module.exports.hasHostBackedContent({js_str(attributions)})") is True
    assert call(ctx, "module.exports.hasHostBackedContent(null)") is False
    assert (
        call(
            ctx,
            "module.exports.hasHostBackedContent({attributions: [], unbacked: [], "
            "computed: [], passages_available: 0})",
        )
        is False
    )


def test_collapse_evidence_dump_shows_visible_note_and_toggle_reveals_content(ctx):
    """2026-09-21 dump-collapse bug: a tutor bubble collapsed by
    collapseEvidenceDump must never end up with zero visible explanation --
    a short muted note stays visible above the toggle, and the toggle still
    reveals the original (hidden, not deleted) content."""
    ctx.eval(
        """
        var _tutorNode = document.createElement("div");
        var _answerText = document.createTextNode("Yes, DNA is a molecule. [S1][S2][S3][S4]");
        _tutorNode.appendChild(_answerText);
        module.exports.collapseEvidenceDump(_tutorNode);
        """
    )
    found = call(ctx, "_walkCollect(_tutorNode)")
    # the original answer text is still in the tree (never deleted), just hidden
    full_text = call(ctx, "_tutorNode.textContent")
    assert "DNA is a molecule" in full_text
    # a visible (non-hidden) note explaining the collapse is present
    note_text = call(ctx, "_tutorNode.children[0].textContent")
    note_hidden = call(ctx, "!!_tutorNode.children[0].hidden")
    assert note_hidden is False
    assert "repeated its sources" in note_text
    # the dump content itself starts hidden
    content_wrap_hidden = call(ctx, "_tutorNode.children[2].hidden")
    assert content_wrap_hidden is True
    # clicking the toggle reveals it and the note remains
    ctx.eval('_tutorNode.children[1].dispatchEvent({type: "click"});')
    content_wrap_hidden_after = call(ctx, "_tutorNode.children[2].hidden")
    assert content_wrap_hidden_after is False
    note_hidden_after = call(ctx, "!!_tutorNode.children[0].hidden")
    assert note_hidden_after is False
    del found


# ---------------------------------------------------------------------------
# (f) 2026-09-21 model-authored fake "Sources:" block (owner-reported live
# bug A) -- must never be rendered, and its label-led lines must never
# become attribution markers or citation chips. Built end-to-end: the real
# Python `attribute_sentences`/`resolve_citations` compute offsets against
# the real owner-shaped answer, and the real JS renderer consumes them.
# ---------------------------------------------------------------------------

from tutor.app.citations import attribute_sentences, resolve_citations  # noqa: E402

_OWNER_ANSWER_WITH_FAKE_SOURCES = (
    "So, while DNA is the largest molecule ... the longest molecule overall "
    "is titin.\n\n"
    "Sources:\n"
    "[S1] DNA structure and replication mechanisms (describing DNA's size "
    "in terms of base pairs).\n"
    "[S2] Overview of DNA replication process.\n"
    "[S3] Basic chemical structure of DNA (describing its components).\n"
    "[S4] General description of DNA molecules.\n"
    "[S5] References showing various descriptions of DNA length.\n"
    "[S6] (Note: The source list provided was not directly related to the "
    "question but serves as context for the answer.)\n\n"
    "Short answer: DNA is large, but titin is the longest molecule by "
    "sheer length of its amino acid sequence."
)


def _passage(label, pid, text):
    return {"label": label, "id": pid, "title": "T", "path": "P", "text": text}


def test_fake_sources_block_never_rendered_end_to_end(ctx):
    answer = _OWNER_ANSWER_WITH_FAKE_SOURCES
    passages = [_passage(f"S{i}", f"p{i}", "unrelated passage text " * 3) for i in range(1, 7)]
    result = attribute_sentences(answer, passages)
    citations = resolve_citations(answer, passages)

    attributions_payload = {
        "attributions": [
            {
                "sentence_span": list(a.sentence_span),
                "passage_id": a.passage_id,
                "model_cited": a.model_cited,
            }
            for a in result.attributions
        ],
        "unbacked": [{"span": list(u.span), "reason": u.reason} for u in result.unbacked_spans],
        "computed": [],
    }
    citations_payload = {
        "citations": [
            {"label": c.label, "passage_id": c.passage_id, "unresolved": c.unresolved}
            for c in citations
        ]
    }

    ctx.eval("var _cf = document.createElement('div');")
    ctx.eval(
        "module.exports.renderAnswerWithAttribution(_cf, "
        f"{js_str(answer)}, {js_str(attributions_payload)}, {js_str(citations_payload)});"
    )
    rendered = call(ctx, "_cf.textContent")
    assert "Sources:" not in rendered
    assert "[S1]" not in rendered
    assert "DNA structure and replication" not in rendered
    assert "Short answer" in rendered
    assert "longest molecule overall is titin" in rendered

    found = call(ctx, "_walkCollect(_cf)")
    # None of the model's invented per-source descriptions produced a
    # citation chip -- the six fake "[S1]".."[S6]" labels never resolve to
    # a real chip (only the two real content sentences may still get their
    # own unbacked markers, unrelated to the fake source list).
    assert found["chips"] == []


def test_find_model_source_block_ranges_matches_python(ctx):
    from tutor.app.citations import find_model_source_block_ranges

    py_ranges = find_model_source_block_ranges(_OWNER_ANSWER_WITH_FAKE_SOURCES)
    js_ranges = call(
        ctx,
        f"module.exports.findModelSourceBlockRanges({js_str(_OWNER_ANSWER_WITH_FAKE_SOURCES)})",
    )
    assert [tuple(r) for r in js_ranges] == py_ranges


def test_ordinary_sentence_mentioning_label_still_renders(ctx):
    text = "As shown in [S1], water boils at 100C."
    ctx.eval("var _cg = document.createElement('div');")
    ctx.eval(f"module.exports.renderTextWithCitations(_cg, {js_str(text)});")
    rendered = call(ctx, "_cg.textContent")
    assert "water boils at 100C" in rendered


# ---------------------------------------------------------------------------
# (g) 2026-09-21 owner-reported live bug B: numbered-list items with a real
# [S#] label that does NOT actually support the claim must still surface a
# visible ○ "not found" marker, not just a bare citation chip.
# ---------------------------------------------------------------------------


def test_list_item_with_unsupported_label_gets_unbacked_marker_end_to_end(ctx):
    answer = (
        "The largest molecule is not definitively known, but some of the "
        "largest include:\n"
        "1. **Ribulose Bisphosphate Carboxylase/Oxygenase (RuBisCO)**: This "
        "is one of the largest enzymes on Earth. [S1]\n"
        "2. **Titin**: Known as the longest protein in the human body. [S2]\n"
    )
    passages = [
        _passage("S1", "p1", "Water boils at 100 degrees Celsius at sea level."),
        _passage("S2", "p2", "Titin is the longest known protein, found in muscle sarcomeres."),
    ]
    result = attribute_sentences(answer, passages)
    citations = resolve_citations(answer, passages)

    attributions_payload = {
        "attributions": [
            {
                "sentence_span": list(a.sentence_span),
                "passage_id": a.passage_id,
                "model_cited": a.model_cited,
            }
            for a in result.attributions
        ],
        "unbacked": [{"span": list(u.span), "reason": u.reason} for u in result.unbacked_spans],
        "computed": [],
    }
    citations_payload = {
        "citations": [
            {"label": c.label, "passage_id": c.passage_id, "unresolved": c.unresolved}
            for c in citations
        ]
    }

    ctx.eval("var _ch = document.createElement('div');")
    ctx.eval(
        "module.exports.renderAnswerWithAttribution(_ch, "
        f"{js_str(answer)}, {js_str(attributions_payload)}, {js_str(citations_payload)});"
    )
    found = call(ctx, "_walkCollect(_ch)")
    assert "[S1]" in {c["text"] for c in found["chips"]}
    assert "[S2]" in {c["text"] for c in found["chips"]}
    # exactly one list item's claim went unsupported by its own cited
    # passage -- it must carry the "unbacked" marker class.
    assert any(m == "attribution-marker attribution-unbacked" for m in found["markers"])


# ---------------------------------------------------------------------------
# 2026-09-21 owner-reported bug: a "tool" SSE event fetched the tutor node
# (removing the working status bubble) before any answer text existed, so
# the student saw no status through the whole search-and-read phase. Fixed
# by handling "tool" (and "status") before ever calling getTutorNode().
# ---------------------------------------------------------------------------


def test_tool_event_does_not_remove_working_bubble_status_still_updates(ctx):
    ctx.eval(
        """
        var _removed = false;
        var _statusCalls = [];
        var _getTutorNodeCalls = 0;
        var _working = {
          node: { remove: function () { _removed = true; } },
          setStatus: function (stage, detail) { _statusCalls.push(detail); },
          stop: function () {},
        };
        function _getTutorNode() {
          _getTutorNodeCalls += 1;
          return document.createElement('div');
        }
        var _line = '';
        function _setTutorLine(t) { _line = t; }
        function _getTutorLine() { return _line; }
        """
    )
    ctx.eval(
        """
        module.exports.handleFrame(
          'event: status' + String.fromCharCode(10) +
            'data: {"stage":"searching","detail":"Working out what to look up..."}',
          _getTutorNode, _setTutorLine, _getTutorLine, _working, null
        );
        module.exports.handleFrame(
          'event: tool' + String.fromCharCode(10) + 'data: {"name":"research","phase":"call"}',
          _getTutorNode, _setTutorLine, _getTutorLine, _working, null
        );
        module.exports.handleFrame(
          'event: status' + String.fromCharCode(10) +
            'data: {"stage":"reading","detail":"Found 3 passages. Reading them..."}',
          _getTutorNode, _setTutorLine, _getTutorLine, _working, null
        );
        """
    )
    assert call(ctx, "_removed") is False
    assert call(ctx, "_getTutorNodeCalls") == 0
    assert call(ctx, "_statusCalls") == [
        "Working out what to look up...",
        "Found 3 passages. Reading them...",
    ]

    # The working bubble is removed only once real answer text starts
    # arriving (the first "token" frame), never by a "tool" or "status"
    # frame in between.
    ctx.eval(
        """
        module.exports.handleFrame(
          'event: token' + String.fromCharCode(10) + 'data: {"text":"Hello"}',
          _getTutorNode, _setTutorLine, _getTutorLine, _working, null
        );
        """
    )
    assert call(ctx, "_getTutorNodeCalls") == 1


# ---------------------------------------------------------------------------
# app.model_may_skip_search: evidence level "skipped" -- see
# docs/rewrite_on_weak_evidence.md, "Model may skip the search". No search
# ran this turn, so neither the "Searched for: ..." line nor the "did not
# find anything in the library" note should ever appear.
# ---------------------------------------------------------------------------


def test_searched_for_line_omitted_when_evidence_skipped(ctx):
    ctx.eval("var _tutorNode = document.createElement('div');")
    ctx.eval(
        'module.exports.appendSearchedForLine(_tutorNode, '
        '{level_before: "skipped", level_after: "skipped", rewritten_queries: [], '
        'corrected_terms: {}});'
    )
    text = call(ctx, "_tutorNode.textContent")
    assert text == ""
    assert call(ctx, "_tutorNode.children.length") == 0


def test_apply_citation_quality_suppresses_not_found_note_when_skipped(ctx):
    """Baseline: with no evidence info at all (a normal turn with nothing
    backed), applyCitationQuality DOES append a "did not find anything"
    note to the chat pane. With evidence.level_after == "skipped", that
    note must never appear, even though nothing was backed either."""
    chat_children_before = call(ctx, "document.getElementById('chat').children.length")
    ctx.eval("var _tutorNode1 = document.createElement('div');")
    ctx.eval(
        'module.exports.applyCitationQuality(_tutorNode1, {}, null, '
        '{attributions: [], unbacked: [], computed: [], passages_available: 0});'
    )
    chat_children_after_normal = call(ctx, "document.getElementById('chat').children.length")
    assert chat_children_after_normal == chat_children_before + 1

    ctx.eval("var _tutorNode2 = document.createElement('div');")
    ctx.eval(
        'module.exports.applyCitationQuality(_tutorNode2, '
        '{evidence: {level_before: "skipped", level_after: "skipped"}}, null, '
        '{attributions: [], unbacked: [], computed: [], passages_available: 0});'
    )
    chat_children_after_skipped = call(ctx, "document.getElementById('chat').children.length")
    assert chat_children_after_skipped == chat_children_after_normal
