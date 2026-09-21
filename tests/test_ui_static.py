"""RED tests for the static UI assets (WP-C3).

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §12 (UI table:
chat, stop, subject selector, simpler/deeper/hint, source viewer, status)
and §13-15 (privacy/offline: no external URLs, fonts, or CDNs -- everything
served by the tutor app itself, loopback only) plus
docs/plan/offline_tutor_implementation_plan.md WP-C3 (static HTML/JS, SSE
consumption, stop, subject selector, action buttons, [S#] chips, highlighted
source viewer, status panel, scripts/kiosk.ps1 + scripts/kiosk.sh with
"--kiosk").

These tests only read files from disk; nothing under tutor/ is touched by
this test file itself (it asserts on files another workstream is expected
to create).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "tutor" / "ui"
SCRIPTS_DIR = REPO_ROOT / "scripts"

INDEX_HTML = UI_DIR / "index.html"
APP_JS = UI_DIR / "app.js"
APP_CSS = UI_DIR / "app.css"
KIOSK_PS1 = SCRIPTS_DIR / "kiosk.ps1"
KIOSK_SH = SCRIPTS_DIR / "kiosk.sh"

_EXTERNAL_URL_RE = re.compile(r"(https?://(?!127\.0\.0\.1)[^\s\"'()]+)|(//cdn[^\s\"'()]*)")

_REQUIRED_IDS = [
    "chat",
    "input",
    "send",
    "stop",
    "subject",
    "btn-simpler",
    "btn-deeper",
    "btn-hint",
    "btn-research",
    "source-viewer",
    "status-panel",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------


def test_index_html_exists():
    assert INDEX_HTML.is_file()


def test_app_js_exists():
    assert APP_JS.is_file()


def test_app_css_exists():
    assert APP_CSS.is_file()


def test_kiosk_ps1_exists():
    assert KIOSK_PS1.is_file()


def test_kiosk_sh_exists():
    assert KIOSK_SH.is_file()


# ---------------------------------------------------------------------------
# Offline rule: no external hosts referenced anywhere
# ---------------------------------------------------------------------------


def test_index_html_has_no_external_references():
    text = _read(INDEX_HTML)
    assert not _EXTERNAL_URL_RE.search(text), "index.html must not reference external hosts"


def test_app_js_has_no_external_references():
    text = _read(APP_JS)
    assert not _EXTERNAL_URL_RE.search(text), "app.js must not reference external hosts"


def test_app_css_has_no_external_references():
    text = _read(APP_CSS)
    assert not _EXTERNAL_URL_RE.search(text), "app.css must not reference external hosts"


# ---------------------------------------------------------------------------
# index.html required elements
# ---------------------------------------------------------------------------


def test_index_html_contains_required_element_ids():
    text = _read(INDEX_HTML)
    for element_id in _REQUIRED_IDS:
        assert re.search(rf'id=["\']{re.escape(element_id)}["\']', text), (
            f"index.html missing id={element_id!r}"
        )


# ---------------------------------------------------------------------------
# app.js behavior
# ---------------------------------------------------------------------------


def test_app_js_consumes_sse_or_fetch_stream():
    text = _read(APP_JS)
    assert "EventSource" in text or "ReadableStream" in text or "getReader" in text


def test_app_js_handles_all_named_events():
    text = _read(APP_JS)
    for event_name in ("token", "tool", "citations", "done", "error"):
        assert event_name in text, f"app.js does not reference event {event_name!r}"


def test_app_js_renders_citation_chips():
    text = _read(APP_JS)
    has_render_fn = "renderCitations" in text
    has_regex_pattern = bool(re.search(r"\\\[S\\d", text)) or "[S" in text
    assert has_render_fn or has_regex_pattern


def test_app_js_renders_unresolved_labels_as_plain_text_not_a_chip():
    # An [S#] label that resolves to no evidence passage (e.g. [S89] when
    # only S1-S5 exist) must render as plain muted text, never a clickable
    # citation-chip <button> (docs/soak_v3_analysis.md, label spam finding).
    text = _read(APP_JS)
    assert "unresolved" in text
    assert "renderUnresolvedLabel" in text


def test_app_js_excludes_unresolved_labels_from_citation_quality_counts():
    text = _read(APP_JS)
    assert re.search(r"function applyCitationQuality[\s\S]*?unresolved", text)


def test_app_js_never_uses_innerhtml():
    text = _read(APP_JS)
    assert "innerHTML" not in text


def test_app_js_uses_textcontent_for_untrusted_text():
    text = _read(APP_JS)
    assert "textContent" in text


# ---------------------------------------------------------------------------
# 2026-09-20 evidence-dump follow-up: the UI collapses a detected evidence
# dump behind a toggle and shows a plain note when every citation on a turn
# is unsupported (docs/citation_experiment.md). textContent/DOM-node only,
# same as the rest of the citation rendering -- no innerHTML anywhere.
# ---------------------------------------------------------------------------


def test_app_js_reacts_to_evidence_dump_flag():
    text = _read(APP_JS)
    assert "evidence_dump" in text


def test_app_js_offers_a_toggle_to_show_pasted_sources():
    text = _read(APP_JS)
    assert "show the tutor's pasted sources" in text.lower()


def test_app_js_shows_unsupported_sources_note():
    # 2026-09-20 attribution-driven note follow-up: the old wording ("did
    # not match this question") was shown even when the host DID back a
    # sentence to a relevant passage, just under a mislabeled [S#] --
    # actively wrong. The note now only fires when nothing was checked
    # against the library at all, worded accordingly, and is driven by the
    # `attributions` event via hasHostBackedContent rather than by
    # citation_quality/unsupported_labels alone.
    text = _read(APP_JS)
    assert "none of this answer was found in the sources the tutor looked up" in text.lower()
    assert "the tutor did not find anything in the library for this question" in text.lower()
    assert "hasHostBackedContent" in text
    assert "passages_available" in text


def test_app_js_still_never_uses_innerhtml_after_dump_handling():
    text = _read(APP_JS)
    assert "innerHTML" not in text


# ---------------------------------------------------------------------------
# 2026-09-20 attribution follow-up: host-side sentence-level attribution
# (docs/attribution_design.md). The UI never edits the model's text or
# inserts [S#] itself -- it renders a subtle marker per host-backed
# sentence (opens the source viewer), a distinct style for unbacked
# sentences, a stronger one for unbacked_number, and a legend using the
# spec's own statement names.
# ---------------------------------------------------------------------------


def test_app_js_handles_attributions_event():
    text = _read(APP_JS)
    assert "attributions" in text


def test_app_js_distinguishes_unbacked_number():
    text = _read(APP_JS)
    assert "unbacked_number" in text or "unbacked-number" in text


def test_app_js_opens_source_viewer_for_host_backed_attribution():
    text = _read(APP_JS)
    assert "openSourceViewer" in text
    assert "passage_id" in text


def test_app_js_attribution_handling_uses_textcontent_only():
    text = _read(APP_JS)
    assert "innerHTML" not in text


def test_index_html_has_attribution_legend():
    # 2026-09-20 wording follow-up: the host checks each sentence against
    # the passages retrieved/retained FOR THIS TURN, not "the whole
    # library" -- the legend must say what actually happened.
    text = _read(INDEX_HTML)
    lower = text.lower()
    assert "found in the sources the tutor looked up" in lower
    assert "not found in the sources the tutor looked up" in lower
    assert "not in the sources the tutor looked up" in lower


def test_app_css_has_attribution_marker_styles():
    text = _read(APP_CSS)
    assert "attribution" in text.lower()


# ---------------------------------------------------------------------------
# 2026-09-20 computed-statement follow-up (docs/calc_investigation.md fix
# #1): the host verifies stated arithmetic against the calc evaluator and
# renders a checked/mismatch marker, never editing the model's own text.
# ---------------------------------------------------------------------------


def test_app_js_handles_computed_items():
    text = _read(APP_JS)
    assert "computed" in text
    assert "checked" in text.lower()


def test_app_js_shows_calculator_mismatch_text():
    text = _read(APP_JS)
    assert "The calculator gets" in text


def test_app_js_computed_handling_uses_textcontent_only():
    text = _read(APP_JS)
    assert "innerHTML" not in text


def test_index_html_has_computed_legend_entry():
    text = _read(INDEX_HTML)
    lower = text.lower()
    assert "computed" in lower
    assert "checked by the calculator" in lower


def test_app_css_has_computed_marker_styles():
    text = _read(APP_CSS)
    assert "attribution-computed-verified" in text
    assert "attribution-computed-mismatch" in text


# ---------------------------------------------------------------------------
# kiosk launch scripts
# ---------------------------------------------------------------------------


def test_kiosk_ps1_mentions_kiosk_flag():
    text = _read(KIOSK_PS1)
    assert "--kiosk" in text


def test_kiosk_sh_mentions_kiosk_flag():
    text = _read(KIOSK_SH)
    assert "--kiosk" in text


# ---------------------------------------------------------------------------
# 2026-09-20 owner bug report follow-up: fixed-viewport layout, one chip
# per citation (using the citations event's real passage_id, never a
# duplicate/broken chip appended alongside it), safe minimal markdown, and
# sentence-level marker highlighting. Fixture is a real SSE dump captured
# from the running app (data/dump_ui_bug_sse.py), copied into
# tests/fixtures/ -- this file never reads tests/../data/.
# ---------------------------------------------------------------------------

FIXTURE_SSE = REPO_ROOT / "tests" / "fixtures" / "ui_bug_sse_frames.txt"


def test_ui_bug_sse_fixture_exists_and_shows_the_reported_shape():
    text = _read(FIXTURE_SSE)
    assert "event: citations" in text
    assert "event: attributions" in text
    # The captured turn reproduces the owner's report: two model [S#]
    # labels the resolver marks unsupported by the old citation_quality
    # rule, while the host's own attribution DID back two sentences to
    # those same passages (model_cited: true) -- exactly the situation the
    # new note logic (item 3/7) must not call "did not match".
    assert '"unsupported_labels": ["S1", "S2"]' in text
    assert '"model_cited": true' in text


def test_app_js_rebuilds_from_raw_text_on_citations_and_attributions_no_append():
    # bug #2 root cause: the "attributions" handler used to call
    # renderCitations() a SECOND time after renderAnswerWithAttribution()
    # already drew a chip for the same [S#] occurrence -- one chip built
    # without a passage_id (broken -- "Source not found") and one correct.
    # Both event handlers must now call the same rebuild-from-raw-text
    # function and nothing else.
    text = _read(APP_JS)
    assert re.search(
        r'eventName === "citations"[\s\S]*?renderAnswerWithAttribution\(',
        text,
    )
    attributions_block = re.search(
        r'else if \(eventName === "attributions"\) \{([\s\S]*?)\} else if',
        text,
    )
    assert attributions_block is not None
    assert "renderCitations(tutorNode" not in attributions_block.group(1)


def test_app_js_citation_chip_uses_real_passage_id_not_bare_label():
    # The rebuilt chip must look the passage_id up from the citations
    # event's label -> passage_id map, never fall back to the bare label
    # (that fallback is what produced the "Source not found" chip).
    text = _read(APP_JS)
    assert "passageIdByLabel" in text
    assert re.search(r"renderCitationChip\(label, passageIdByLabel\[label\]\)", text)


def test_app_js_has_safe_markdown_tokenizer():
    text = _read(APP_JS)
    assert "appendMarkdownText" in text
    assert 'el("strong"' in text
    assert 'el("em"' in text
    assert 'el("code"' in text
    assert "innerHTML" not in text


def test_app_js_sentence_level_highlight_helpers_present():
    text = _read(APP_JS)
    assert "findBestSentenceSpan" in text
    assert "sentenceText" in text


def test_app_css_uses_fixed_viewport_shell():
    text = _read(APP_CSS)
    assert "100dvh" in text
    assert "#chat {" in text
    assert "overflow-y: auto" in text


def test_index_html_status_panel_is_collapsible():
    text = _read(INDEX_HTML)
    assert re.search(r'<details id=["\']status-panel["\']', text)


def test_node_markdown_and_chip_tokenizer_behaviour():
    """Behavioural check of appendMarkdownText/renderAnswerWithAttribution
    against the real captured SSE payload, run under Node if available on
    this box; skipped (not failed) when Node is not installed, per the
    task's "no JS test runner" fallback."""
    import json
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not installed on this box")

    frames_text = _read(FIXTURE_SSE)
    done_line = [
        line
        for line in frames_text.splitlines()
        if line.startswith("data:") and '"status": "ok"' in line
    ][0]
    done_data = json.loads(done_line[len("data:"):].strip())
    answer = done_data["answer"]

    script = r"""
    const fs = require('fs');
    global.document = {
      createElement(tag) {
        return {
          tagName: tag,
          children: [],
          textContent: '',
          appendChild(child) { this.children.push(child); },
          setAttribute() {},
        };
      },
      createTextNode(text) {
        return { textContent: text, children: [] };
      },
    };
    const src = fs.readFileSync(process.argv[2], 'utf8');
    // Pull out just the two pure functions under test without running the
    // whole IIFE (which touches a real DOM this harness does not have).
    const markdownMatch = src.match(/function appendMarkdownText[\s\S]*?\n  \}\n/);
    const elMatch = src.match(/function el\(tag, opts\)[\s\S]*?\n  \}\n/);
    const markdownReMatch = src.match(/const MARKDOWN_RE = [^\n]+\n/);
    eval(markdownReMatch[0] + elMatch[0] + markdownMatch[0]);
    const container = document.createElement('div');
    appendMarkdownText(container, process.argv[3]);
    const kinds = container.children.map((c) => c.tagName || 'text');
    console.log(JSON.stringify(kinds));
    """
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run(
            [node, script_path, str(APP_JS), answer],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        import os

        os.unlink(script_path)
    assert result.returncode == 0, result.stderr
    kinds = json.loads(result.stdout.strip())
    assert "strong" in kinds  # **Burj Khalifa** etc. render as real <strong> nodes
    assert not any(k == "text" and "**" in k for k in kinds)
