"""Load a JS engine capable of executing tutor/ui/app.js's pure UI logic
without Node.

Preference order: Node (if present on this box) is NOT used here -- this
harness is the no-Node fallback, using an embedded V8 (py_mini_racer). If
py_mini_racer is not importable (e.g. no wheel for this platform), callers
should skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "tutor" / "ui" / "app.js"
FAKE_DOM_JS = Path(__file__).resolve().parent / "fake_dom.js"

try:
    from py_mini_racer import MiniRacer  # type: ignore

    HAVE_ENGINE = True
except Exception:  # pragma: no cover - exercised only when the wheel is missing
    MiniRacer = None  # type: ignore
    HAVE_ENGINE = False


def new_context() -> MiniRacer:
    """Return a MiniRacer context with app.js loaded and its pure functions
    exposed on the global `module.exports`."""
    mr = MiniRacer()
    mr.eval(FAKE_DOM_JS.read_text(encoding="utf-8"))
    mr.eval(APP_JS.read_text(encoding="utf-8"))
    # Small JS-side helpers used by several tests below, kept here (not in
    # app.js) since they exist only to observe the fake DOM tree.
    mr.eval(
        r"""
        function _inlineTokens(text) {
          var c = document.createElement("div");
          module.exports.appendInlineMarkdown(c, text);
          return c.children.map(function (ch) {
            return { tag: ch.tagName, text: ch.textContent };
          });
        }
        function _walkCollect(node) {
          var chips = [];
          var plainCitationLabels = [];
          var markers = [];
          (function walk(n) {
            var cls = n.className || "";
            if (cls.indexOf("citation-chip") >= 0) {
              chips.push({ text: n.textContent, passage: n.attrs["data-passage-id"] });
            }
            if (cls.indexOf("citation-label-unresolved") >= 0) {
              plainCitationLabels.push(n.textContent);
            }
            if (cls.indexOf("attribution-marker") >= 0) {
              markers.push(cls);
            }
            (n.children || []).forEach(walk);
          })(node);
          return { chips: chips, plainCitationLabels: plainCitationLabels, markers: markers };
        }
        """
    )
    return mr


def call(mr: MiniRacer, expr: str) -> Any:
    """Evaluate a JS expression and return it JSON-round-tripped into Python."""
    return json.loads(mr.eval(f"JSON.stringify({expr})"))


def js_str(value: Any) -> str:
    return json.dumps(value)
