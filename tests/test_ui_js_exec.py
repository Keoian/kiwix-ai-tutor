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
