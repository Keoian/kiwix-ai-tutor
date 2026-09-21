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
    text = _read(APP_JS)
    assert "did not match this question" in text.lower()


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
    text = _read(INDEX_HTML)
    lower = text.lower()
    assert "source-backed" in lower
    assert "not checked against the library" in lower
    assert "not found in the library" in lower


def test_app_css_has_attribution_marker_styles():
    text = _read(APP_CSS)
    assert "attribution" in text.lower()


# ---------------------------------------------------------------------------
# kiosk launch scripts
# ---------------------------------------------------------------------------


def test_kiosk_ps1_mentions_kiosk_flag():
    text = _read(KIOSK_PS1)
    assert "--kiosk" in text


def test_kiosk_sh_mentions_kiosk_flag():
    text = _read(KIOSK_SH)
    assert "--kiosk" in text
