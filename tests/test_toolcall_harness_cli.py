"""Tests for the eval.toolcall_harness CLI (Part 2: writes markdown + JSON report)."""

from __future__ import annotations

import json
from pathlib import Path

from eval import toolcall_harness

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "dev.toml"


def test_main_writes_markdown_and_json_report(tmp_path, monkeypatch):
    def fake_http_post(base_url, payload, timeout=180.0):
        assert base_url.startswith("http://")
        assert payload["max_tokens"] == 256
        assert "temperature" in payload
        message = {"content": "Here is the answer.", "tool_calls": None}
        return {"choices": [{"message": message}]}

    monkeypatch.setattr(toolcall_harness, "_http_post", fake_http_post)

    out_path = tmp_path / "results.md"
    exit_code = toolcall_harness.main(["--config", str(CONFIG_PATH), "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    md = out_path.read_text(encoding="utf-8")
    assert "grammar_path_required" in md
    assert "malformed_rate" in md

    json_path = out_path.with_suffix(".json")
    assert json_path.exists()
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) == 20
    assert {"id", "prompt", "expect_call", "expected_tool", "outcome", "tool_names"} <= set(
        raw[0].keys()
    )
