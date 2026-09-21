"""Baseline v12: BeautifulSoup parser backend selection.

Proven byte-identical over 10,869 real Simple Wikipedia + Wikibooks
articles (plus all tuning-question gold articles) for both call sites that
use ``HTML_PARSER`` (search.py:_extract_text and bundle.py:build_bundle) --
see docs/retrieval_baseline.md, "Baseline v12". These tests cover the
selection logic itself: lxml preferred when importable, safe fallback to
html.parser otherwise, so the app keeps working on a machine without lxml.
"""
from __future__ import annotations

import builtins

import pytest

import tutor.retrieval.zim.content as content_mod


def test_detect_bs_parser_prefers_lxml_when_available():
    pytest.importorskip("lxml")
    assert content_mod._detect_bs_parser() == "lxml"


def test_detect_bs_parser_falls_back_without_lxml(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lxml":
            raise ImportError("simulated: lxml not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert content_mod._detect_bs_parser() == "html.parser"


def test_module_still_works_when_html_parser_is_forced(monkeypatch):
    """The fallback path must keep producing correct output, not just avoid
    crashing: force HTML_PARSER to html.parser and confirm build_bundle
    still renders sane text from a small fixture."""
    import tutor.retrieval.zim.bundle as bundle_mod

    monkeypatch.setattr(content_mod, "HTML_PARSER", "html.parser")
    monkeypatch.setattr(bundle_mod, "HTML_PARSER", "html.parser")
    html = "<html><body><article><h2>Section</h2><p>Hello world.</p></article></body></html>"
    bundle = bundle_mod.build_bundle(html, path="A/x", title="x")
    assert "Hello world." in bundle.text
