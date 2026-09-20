"""HTTP client for llama-server's ``/v1/embeddings`` endpoint (WP-B7).

Ours: no donor code. Mirrors ``tutor.app.llm_client.LlamaClient`` in spirit
(stdlib-only transport, treats the server as an opaque HTTP black box) but
is deliberately much smaller: one blocking request/response call, no
streaming, no cancellation.
"""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit


class EmbeddingError(Exception):
    """Raised when the embedding server cannot be reached or errors out."""


class EmbeddingClient:
    """Blocking client for a llama-server instance started with ``--embedding``."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        parts = urlsplit(self.base_url)
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._https = parts.scheme == "https"

    def _connect(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=self._timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, returning one vector per input in the same order."""
        if not texts:
            return []
        body = json.dumps({"input": texts}).encode("utf-8")
        conn = self._connect()
        try:
            conn.request(
                "POST",
                "/v1/embeddings",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise EmbeddingError(
                    f"embedding server returned HTTP {resp.status}: {raw[:500]!r}"
                )
        except OSError as exc:
            raise EmbeddingError(
                f"could not reach embedding server at {self.base_url}: {exc}"
            ) from exc
        finally:
            conn.close()

        try:
            payload = json.loads(raw.decode("utf-8"))
            rows = payload["data"]
            ordered = sorted(rows, key=lambda r: r.get("index", 0))
            vectors = [row["embedding"] for row in ordered]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EmbeddingError(f"malformed embedding response: {exc}") from exc

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding server returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors
