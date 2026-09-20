"""Unit tests for tutor.retrieval.index.embedding_client.EmbeddingClient.

Same fake-server pattern as tests/test_llm_client.py: an in-process
ThreadingHTTPServer on port 0 stands in for llama-server's
/v1/embeddings endpoint. No real network access, no real model.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tutor.retrieval.index.embedding_client import EmbeddingClient, EmbeddingError


class _FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs):  # noqa: D401 - silence test logs
        pass

    def do_POST(self):
        behavior = self.server.behavior  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode("utf-8"))

        if behavior == "server_error":
            self.send_response(500)
            self.end_headers()
            return

        if behavior == "malformed":
            payload = b"not json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if behavior == "wrong_count":
            data = [{"index": 0, "embedding": [0.1, 0.2]}]
            payload = json.dumps({"data": data}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # default: echo deterministic vectors, one per input, out of order
        # on the wire to exercise index-based reordering.
        texts = body.get("input", [])
        data = [
            {"index": i, "embedding": [float(len(t)), float(i)]}
            for i, t in enumerate(texts)
        ]
        data.reverse()
        payload = json.dumps({"data": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.behavior = "default"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _base_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_embed_empty_list_returns_empty(fake_server):
    client = EmbeddingClient(_base_url(fake_server))
    assert client.embed([]) == []


def test_embed_returns_vectors_in_input_order(fake_server):
    client = EmbeddingClient(_base_url(fake_server))
    vectors = client.embed(["ab", "abcd", "a"])
    assert vectors == [[2.0, 0.0], [4.0, 1.0], [1.0, 2.0]]


def test_embed_server_error_raises(fake_server):
    fake_server.behavior = "server_error"
    client = EmbeddingClient(_base_url(fake_server))
    with pytest.raises(EmbeddingError):
        client.embed(["x"])


def test_embed_malformed_response_raises(fake_server):
    fake_server.behavior = "malformed"
    client = EmbeddingClient(_base_url(fake_server))
    with pytest.raises(EmbeddingError):
        client.embed(["x"])


def test_embed_wrong_count_raises(fake_server):
    fake_server.behavior = "wrong_count"
    client = EmbeddingClient(_base_url(fake_server))
    with pytest.raises(EmbeddingError):
        client.embed(["x", "y"])


def test_embed_unreachable_server_raises():
    client = EmbeddingClient("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(EmbeddingError):
        client.embed(["x"])
