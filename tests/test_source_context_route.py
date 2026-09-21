"""RED tests for ``GET /api/source/{passage_id}/context`` (owner-requested
"show more of the article" reader affordance).

Uses a fake ``research_engine`` (only ``fetch_article_text`` is exercised)
so these tests never touch a real ZIM archive or worker process -- the
route's contract with ``tutor.app.source_view.build_context_view`` is
covered in depth by tests/test_source_viewer.py; this file covers the HTTP
layer: 404 for an unknown passage, 503 (clean JSON) when the archive is
unavailable, paging via before/after query params, and that no path/archive
parameter is ever accepted from the request (the route resolves those only
through the stored snapshot).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tutor.app.main import AppDeps, create_app
from tutor.retrieval.research import ArticleUnavailable
from tutor.retrieval.snapshots import SnapshotStore


@dataclass
class _Passage:
    passage_id: str
    archive_id: str = "wiki_demo"
    path: str = "A/Water"
    title: str = "Water"
    heading_path: tuple = ("Water", "Properties")
    start: int = 0
    end: int = 0
    text: str = ""


def _long_article(passage_text="It boils at 100C at sea level."):
    sentences = [f"Sentence number {i} goes here." for i in range(80)]
    before = " ".join(sentences[:40])
    after = " ".join(sentences[40:])
    text = f"{before} {passage_text} {after}"
    start = text.index(passage_text)
    end = start + len(passage_text)
    return text, start, end


class _FakeResearchEngine:
    def __init__(self, article_text=None, *, fail=False):
        self.article_text = article_text
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def fetch_article_text(self, archive_id, path, *, deadline_s=5.0):
        self.calls.append((archive_id, path))
        if self.fail:
            raise ArticleUnavailable("archive not available")
        return self.article_text

    def close(self):
        pass


def _fake_status_provider(session_id=None):
    return {"llm": {"healthy": True}}


class _FakeSessions:
    def create(self):
        return "sess-1"

    def get(self, session_id):
        return {}


def _make_app(tmp_path: Path, engine=None):
    store = SnapshotStore(tmp_path / "snaps.sqlite3")
    deps = AppDeps(
        turn_runner=lambda *a, **k: None,
        snapshot_store=store,
        status_provider=_fake_status_provider,
        sessions=_FakeSessions(),
        subjects=["general"],
        research_engine=engine,
    )
    app = create_app(deps)
    return app, deps


@pytest.fixture
def article_text():
    return _long_article()


def test_context_endpoint_returns_expanded_text(tmp_path, article_text):
    text, start, end = article_text
    engine = _FakeResearchEngine(article_text=text)
    app, deps = _make_app(tmp_path, engine=engine)
    with TestClient(app) as c:
        deps.snapshot_store.put(
            _Passage("p1", start=start, end=end, text=text[start:end]),
            fingerprint_digest="deadbeef",
        )
        resp = c.get("/api/source/p1/context")
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "Water"
        p = body["passage"]
        assert body["text"][p["start"] : p["end"]] == text[start:end]
        assert "more_before" in body
        assert "more_after" in body
        assert "text_start" in body
    deps.close()


def test_context_endpoint_unknown_passage_is_404(tmp_path):
    engine = _FakeResearchEngine(article_text="x")
    app, deps = _make_app(tmp_path, engine=engine)
    with TestClient(app) as c:
        resp = c.get("/api/source/does-not-exist/context")
        assert resp.status_code == 404
    deps.close()


def test_context_endpoint_archive_unavailable_is_clean_503(tmp_path, article_text):
    text, start, end = article_text
    engine = _FakeResearchEngine(fail=True)
    app, deps = _make_app(tmp_path, engine=engine)
    with TestClient(app) as c:
        deps.snapshot_store.put(
            _Passage("p1", start=start, end=end, text=text[start:end]),
            fingerprint_digest="deadbeef",
        )
        resp = c.get("/api/source/p1/context")
        assert resp.status_code == 503
        body = resp.json()
        assert "book is not available" in body["message"].lower()
    deps.close()


def test_context_endpoint_pages_with_before_after_params(tmp_path, article_text):
    text, start, end = article_text
    engine = _FakeResearchEngine(article_text=text)
    app, deps = _make_app(tmp_path, engine=engine)
    with TestClient(app) as c:
        deps.snapshot_store.put(
            _Passage("p1", start=start, end=end, text=text[start:end]),
            fingerprint_digest="deadbeef",
        )
        small = c.get("/api/source/p1/context", params={"before": 20, "after": 20}).json()
        big = c.get("/api/source/p1/context", params={"before": 500, "after": 500}).json()
        assert len(big["text"]) > len(small["text"])
    deps.close()


def test_context_endpoint_accepts_no_path_or_archive_parameter(tmp_path, article_text):
    """The passage's archive/path come only from the stored snapshot -- a
    caller cannot smuggle a different path/archive_id in via query params
    to read an arbitrary entry (no path-traversal surface)."""
    text, start, end = article_text
    engine = _FakeResearchEngine(article_text=text)
    app, deps = _make_app(tmp_path, engine=engine)
    with TestClient(app) as c:
        deps.snapshot_store.put(
            _Passage("p1", start=start, end=end, text=text[start:end]),
            fingerprint_digest="deadbeef",
        )
        resp = c.get(
            "/api/source/p1/context",
            params={"path": "../../etc/passwd", "archive_id": "other"},
        )
        assert resp.status_code == 200
        # The fake engine was still called with the snapshot's own
        # archive_id/path, never anything from the query string.
        assert engine.calls == [("wiki_demo", "A/Water")]
    deps.close()


def test_context_endpoint_404_when_research_engine_not_configured(tmp_path, article_text):
    text, start, end = article_text
    app, deps = _make_app(tmp_path, engine=None)
    with TestClient(app) as c:
        deps.snapshot_store.put(
            _Passage("p1", start=start, end=end, text=text[start:end]),
            fingerprint_digest="deadbeef",
        )
        resp = c.get("/api/source/p1/context")
        assert resp.status_code == 404
    deps.close()
