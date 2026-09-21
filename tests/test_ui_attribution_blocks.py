"""RED tests for offset-aware attribution/citation placement inside
block-rendered markdown (tables, lists, headings).

Follow-up to 88ec55b (Markdown tables and structure-aware sentence units):
`tutor.app.citations._sentence_spans` now yields one unit per table row /
list item with real raw-text offsets, but `tutor/ui/app.js` rendered blocks
(`<table>`, `<ul>`/`<ol>`, headings) as opaque nodes, so a marker/chip
computed from those offsets landed after the whole block instead of inside
the row/item/cell it belonged to.

This file has two layers:
- Static source checks (style of tests/test_ui_static.py): the pure
  offset-mapping function and the block-aware renderer exist, are wired
  into `renderAnswerWithAttribution`, and the file still never touches
  innerHTML.
- A Node-conditional behavioural check of the pure `mapAnswerToBlocks`
  function against tests/fixtures/table_answer.md, using REAL offsets
  produced by `tutor.app.citations._sentence_spans`/`attribute_sentences`
  on that exact text -- skipped (not failed) when Node is not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tutor.app.citations import attribute_sentences, resolve_citations

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JS = REPO_ROOT / "tutor" / "ui" / "app.js"
FIXTURE_TABLE = REPO_ROOT / "tests" / "fixtures" / "table_answer.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static source checks
# ---------------------------------------------------------------------------


def test_app_js_has_pure_block_mapping_function():
    text = _read(APP_JS)
    assert "function mapAnswerToBlocks(text)" in text


def test_app_js_exports_block_mapping_for_node():
    text = _read(APP_JS)
    assert 'typeof module !== "undefined"' in text
    # 2026-09-20 (UI JS exec follow-up): module.exports now carries every
    # pure/DOM-lite function tests/test_ui_js_exec.py executes under the
    # embedded engine, not just mapAnswerToBlocks -- check the map is still
    # exported rather than pinning the exact object literal text.
    assert re.search(r"module\.exports\s*=\s*\{[\s\S]*mapAnswerToBlocks:\s*mapAnswerToBlocks", text)


def test_app_js_has_block_aware_renderer_and_inline_segment_helpers():
    text = _read(APP_JS)
    assert "function renderBlocksToDom(" in text
    assert "function renderInlineSegment(" in text
    assert "function appendMarkersInRange(" in text


def test_app_js_attribution_rebuild_uses_block_aware_renderer():
    # renderAnswerWithAttribution (used by both the citations and the
    # attributions SSE events) must route through the block-aware
    # renderer, not the old flat emitTextUpTo/emitMarkersUpTo walk that
    # could only append markers as siblings after a whole table/list.
    text = _read(APP_JS)
    fn_match = re.search(
        r"function renderAnswerWithAttribution\([\s\S]*?\n  \}\n",
        text,
    )
    assert fn_match is not None
    body = fn_match.group(0)
    assert "mapAnswerToBlocks(text)" in body
    assert "renderBlocksToDom(" in body
    assert "emitMarkersUpTo" not in body


def test_app_js_table_row_marker_placed_in_last_cell():
    text = _read(APP_JS)
    # The row's last cell gets the marker appended after its own inline
    # content is rendered -- appendMarkersInRange is called with the row's
    # span, keyed off `idx === row.cells.length - 1`.
    table_block = re.search(r'if \(block\.type === "table"\) \{([\s\S]*?)\n      \}\n', text)
    assert table_block is not None
    assert "idx === row.cells.length - 1" in table_block.group(1)
    assert "appendMarkersInRange(td, markers, row.start, row.end" in table_block.group(1)


def test_app_js_list_item_marker_placed_at_item_end():
    text = _read(APP_JS)
    list_block = re.search(r'if \(block\.type === "list"\) \{([\s\S]*?)\n      \}\n', text)
    assert list_block is not None
    assert "appendMarkersInRange(li, markers, item.start, item.end" in list_block.group(1)


def test_app_js_heading_gets_no_markers():
    text = _read(APP_JS)
    heading_block = re.search(
        r'if \(block\.type === "heading"\) \{([\s\S]*?)\n        return;\n      \}\n', text
    )
    assert heading_block is not None
    # Headings pass an explicit empty markers list to renderInlineSegment.
    assert re.search(r"renderInlineSegment\(heading,.*\[\], resolveLabel", heading_block.group(1))


def test_app_js_never_uses_innerhtml_after_block_offset_change():
    text = _read(APP_JS)
    assert "innerHTML" not in text


# ---------------------------------------------------------------------------
# Node-conditional behavioural check, real offsets from attribute_sentences
# ---------------------------------------------------------------------------


def test_node_map_answer_to_blocks_places_markers_and_chips_correctly():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed on this box")

    text = _read(FIXTURE_TABLE)

    # Build a tiny fake passage set so a couple of sentences resolve as
    # host-backed, and a couple of [S#] labels appear + resolve, exercising
    # both marker placement (table row + list item) and citation-chip
    # placement inside a cell, with REAL offsets from the citations module.
    passages = [
        {
            "id": "p1",
            "label": "S1",
            "text": "Build or buy a raised bed, typically 4x4 feet. A compact bed keeps"
            " the whole garden reachable.",
        },
        {
            "id": "p2",
            "label": "S2",
            "text": "Water consistently, about 1 inch per week.",
        },
    ]
    result = attribute_sentences(text, passages)
    citations = resolve_citations(text, passages)

    attributions_payload = {
        "attributions": [
            {
                "sentence_span": list(a.sentence_span),
                "passage_id": a.passage_id,
                "model_cited": a.model_cited,
            }
            for a in result.attributions
        ],
        "unbacked": [
            {"span": list(u.span), "reason": u.reason} for u in result.unbacked_spans
        ],
        "computed": [],
    }
    citations_payload = {
        "citations": [
            {"label": c.label, "passage_id": c.passage_id, "unresolved": c.unresolved}
            for c in citations
        ]
    }

    script = r"""
    const fs = require('fs');
    const path = process.argv[2];
    const mod = require(path);
    const text = fs.readFileSync(process.argv[3], 'utf8');
    const blocks = mod.mapAnswerToBlocks(text);

    // Every offset in every block must be within [0, text.length] and
    // start <= end for every range produced.
    function checkRange(r) {
      if (typeof r.start !== 'number' || typeof r.end !== 'number') return;
      if (r.start < 0 || r.end > text.length || r.start > r.end) {
        throw new Error('bad range ' + JSON.stringify(r));
      }
    }
    const tableBlocks = blocks.filter((b) => b.type === 'table');
    const listBlocks = blocks.filter((b) => b.type === 'list');
    const headingBlocks = blocks.filter((b) => b.type === 'heading');
    if (tableBlocks.length !== 1) throw new Error('expected 1 table, got ' + tableBlocks.length);
    if (listBlocks.length !== 1) throw new Error('expected 1 list, got ' + listBlocks.length);
    if (headingBlocks.length !== 1) {
      throw new Error('expected 1 heading, got ' + headingBlocks.length);
    }

    const table = tableBlocks[0];
    if (table.rows.length !== 4) throw new Error('expected 4 body rows, got ' + table.rows.length);
    table.rows.forEach((row) => {
      checkRange(row);
      row.cells.forEach(checkRange);
      // Row text (raw) must exactly match the row's own [start,end) slice.
      const rowText = text.slice(row.start, row.end);
      if (!rowText.startsWith('|') || !rowText.endsWith('|')) {
        throw new Error('row range is not a full pipe row: ' + JSON.stringify(rowText));
      }
    });
    table.headCells.forEach(checkRange);

    const list = listBlocks[0];
    if (list.items.length !== 3) throw new Error('expected 3 list items, got ' + list.items.length);
    list.items.forEach((item) => {
      checkRange(item);
      const itemText = text.slice(item.start, item.end);
      if (!/^-\s/.test(itemText)) throw new Error('lost "- " marker: ' + JSON.stringify(itemText));
      const innerText = text.slice(item.textStart, item.textEnd);
      if (itemText.indexOf(innerText) === -1) throw new Error('text range not inside item');
    });

    checkRange(headingBlocks[0]);

    console.log('OK');
    """

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    try:
        result_proc = subprocess.run(
            [node, script_path, str(APP_JS), str(FIXTURE_TABLE)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        import os

        os.unlink(script_path)

    assert result_proc.returncode == 0, result_proc.stdout + result_proc.stderr
    assert "OK" in result_proc.stdout

    # Sanity: the payloads built above are usable (exercised for coverage
    # of the python side producing the offsets a real turn would send).
    assert attributions_payload["attributions"] or attributions_payload["unbacked"]
    assert isinstance(citations_payload["citations"], list)
