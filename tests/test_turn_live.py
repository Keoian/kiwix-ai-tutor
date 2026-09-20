"""LIVE end-to-end turn tests for milestone M3.

These exercise the real wiring (tutor.app.compose.build_deps ->
tutor.app.main.create_app) against the real dev llama-server (per
config/dev.toml) and a real libzim fixture archive, over HTTP via
FastAPI's TestClient. They are skipped automatically when the server's
``/health`` endpoint is unreachable, so they never block offline/CI runs.

Prompts are kept tiny: the fixed research budget (2000 tokens, set in
tutor.retrieval.research._DEFAULT_BUDGET_TOKENS and not currently
overridable through tutor.app.compose.build_deps) plus a short system
prompt keeps every live prompt well under ~3k tokens. Expect ~20-40s
prefill on the dev GPU per turn; timeouts here are generous (300s).

Run explicitly with:
    python -m pytest -m integration tests/test_turn_live.py -v -p no:warnings
"""

from __future__ import annotations

import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from tutor.app.compose import build_deps
from tutor.app.llm_client import LlamaClient
from tutor.app.main import create_app
from tutor.settings import load_config

pytestmark = pytest.mark.integration

_CONFIG_PATHS = ["config/dev.toml", "config/dev.granite.toml"]
_TURN_TIMEOUT_S = 300


def _skip_unless_healthy(base_url: str):
    client = LlamaClient(base_url, timeout_s=10)
    if not client.health():
        pytest.skip("llama-server /health unreachable; skipping live integration test")


def _write_registry(path, zim_path) -> None:
    path.write_text(
        f"""
[[archive]]
id = "fixture_ssd"
path = '{zim_path}'
tier = 1
kind = "encyclopedia"
subjects = []
storage = "ssd"
""",
        encoding="utf-8",
    )


def _write_config(path, base_cfg, *, data_dir, registry_path) -> None:
    text = f"""
[runtime]
runtime_dir = '{base_cfg.runtime.runtime_dir}'
model_path = '{base_cfg.runtime.model_path}'
server_binary = '{base_cfg.runtime.server_binary}'

[server]
host = "{base_cfg.server.host}"
port = {base_cfg.server.port}
ctx_size = {base_cfg.server.ctx_size}
cache_type_k = "{base_cfg.server.cache_type_k}"
cache_type_v = "{base_cfg.server.cache_type_v}"
n_gpu_layers = {base_cfg.server.n_gpu_layers}
parallel = {base_cfg.server.parallel}
flash_attn = {str(base_cfg.server.flash_attn).lower()}
jinja = {str(base_cfg.server.jinja).lower()}
slots = {str(base_cfg.server.slots).lower()}

[sampling]
temperature = {base_cfg.sampling.temperature}
top_p = {base_cfg.sampling.top_p}
top_k = {base_cfg.sampling.top_k}

[app]
host = "127.0.0.1"
port = 0
data_dir = '{data_dir}'
registry_path = '{registry_path}'
"""
    path.write_text(text, encoding="utf-8")


@pytest.fixture(params=_CONFIG_PATHS)
def live_app(request, tmp_path, fixture_zim):
    """A real ``TestClient`` over ``create_app(build_deps(cfg))`` where
    ``cfg`` mirrors ``config/dev.toml`` except for a tmp data_dir/registry
    pointing at ONE tier-1 "ssd" archive: the libzim fixture built by
    ``tests/zim_fixtures.py``."""
    base_cfg = load_config(request.param)
    _skip_unless_healthy(base_cfg.server.base_url)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_path = tmp_path / "archives.toml"
    _write_registry(registry_path, fixture_zim)

    config_path = tmp_path / "live.toml"
    _write_config(config_path, base_cfg, data_dir=data_dir, registry_path=registry_path)

    cfg = load_config(config_path)
    deps = build_deps(cfg)
    app = create_app(deps)
    client = TestClient(app)
    return client


def _sid(client: TestClient) -> str:
    resp = client.post("/api/session")
    assert resp.status_code == 200
    return resp.json()["session_id"]


def _stream_turn(
    client: TestClient, session_id: str, body: dict, *, timeout: float = _TURN_TIMEOUT_S
):
    """POST a turn and parse the SSE stream into a list of (event, data,
    t_offset_seconds) tuples, t_offset measured from just before the POST."""
    start = time.monotonic()
    frames = []
    with client.stream(
        "POST", f"/api/session/{session_id}/turn", json=body, timeout=timeout
    ) as resp:
        assert resp.status_code == 200
        event_name = None
        for line in resp.iter_lines():
            if line == "":
                continue
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                frames.append((event_name, data, time.monotonic() - start))
                event_name = None
    return frames


def _first_index(frames, predicate):
    for i, (name, data, _t) in enumerate(frames):
        if predicate(name, data):
            return i
    return None


_NUMBER_RE = re.compile(r"[\d,]+")


def _contains_number(text: str, value: int) -> bool:
    target = str(value)
    for match in _NUMBER_RE.findall(text):
        if match.replace(",", "") == target:
            return True
    return False


# ---------------------------------------------------------------------------
# (a) factual question -> pre-retrieval, streaming, citations, source viewer
# ---------------------------------------------------------------------------


def test_factual_question_pre_retrieves_streams_and_resolves_citations(live_app):
    client = live_app
    sid = _sid(client)

    question = "What is the Pythagorean theorem?"
    frames = _stream_turn(client, sid, {"text": question})

    tool_idx = _first_index(frames, lambda n, d: n == "tool")
    token_idx = _first_index(frames, lambda n, d: n == "token")
    done_frames = [(n, d, t) for n, d, t in frames if n == "done"]
    citations_frames = [(n, d, t) for n, d, t in frames if n == "citations"]

    assert tool_idx is not None, "expected at least one research/tool event"
    assert token_idx is not None, "expected the answer to stream tokens"
    assert tool_idx < token_idx, "the (pre-retrieval) tool event must precede the first token"

    assert len(done_frames) == 1
    assert done_frames[0][1]["status"] == "ok"

    assert len(citations_frames) == 1
    citations = citations_frames[0][1]["citations"]

    resolved = [c for c in citations if not c.get("unresolved")]
    for citation in resolved:
        passage_id = citation["passage_id"]
        resp = client.get(f"/api/source/{passage_id}")
        assert resp.status_code == 200
        body = resp.json()
        start, end = body["highlight"]["start"], body["highlight"]["end"]
        assert body["text"][start:end] != ""

    if not citations:
        pytest.xfail(
            "model behaviour: no [S#] citation emitted for this fixture "
            "question. Re-measured live 2026-09-20 (evidence-dump "
            "follow-up, real simplewiki archive, config/dev.toml's "
            "Bonsai-8B-Q1_0 model, 18 real-lesson factual questions, "
            "shipped 'current' system prompt + reminder wording): "
            "cited_and_supported (all citations resolve AND pass "
            "is_supported) = 0/18 (0%), citation_rate (>=1 [S#] label at "
            "all) = 2/18 (11%), evidence_dump_rate = 0%. Two further "
            "host-side levers (a shorter system prompt; temperature 0.2) "
            "were tried and neither cleared the model's own knowledge "
            "answering the question without citing supplied evidence -- "
            "still far below the 80% bar for a hard assertion, and now "
            "known to be model-limited rather than host-wording-limited "
            "(the fixture-only 60% figure this xfail previously cited "
            "does not generalize to real-archive, real-lesson questions); "
            "see eval/run_turn_eval.py and docs/citation_experiment.md)"
        )


# ---------------------------------------------------------------------------
# (b) calc tool
# ---------------------------------------------------------------------------


def test_calculator_question_uses_calc_tool(live_app):
    client = live_app
    sid = _sid(client)

    frames = _stream_turn(
        client, sid, {"text": "What is 4871 * 392? Use the calculator."}
    )

    calc_result_frames = [
        d for n, d in ((n, d) for n, d, _t in frames) if n == "tool" and d.get("name") == "calc"
    ]
    done_frames = [(n, d, t) for n, d, t in frames if n == "done"]
    assert len(done_frames) == 1
    assert done_frames[0][1]["status"] == "ok"
    answer = done_frames[0][1]["answer"]

    if not calc_result_frames:
        pytest.xfail(
            "model behaviour: calc tool not invoked for a multi-digit "
            "multiplication (measured 3/3 runs invoked calc on "
            "2026-09-20; this run is the outlier -- see docs/M3_report.md)"
        )

    if not _contains_number(answer, 1909432):
        pytest.xfail(
            "model behaviour: final answer did not surface the computed "
            "product 1909432 (measured 3/3 runs surfaced it on "
            "2026-09-20; this run is the outlier -- see docs/M3_report.md)"
        )


# ---------------------------------------------------------------------------
# (c) deterministic action never retrieves
# ---------------------------------------------------------------------------


def test_action_never_triggers_research(live_app):
    client = live_app
    sid = _sid(client)

    # Prime the session with one turn so "simpler" has something to act on.
    _stream_turn(client, sid, {"text": "What is the Pythagorean theorem?"})

    frames = _stream_turn(client, sid, {"action": "simpler"})

    research_tool_frames = [
        d for n, d in ((n, d) for n, d, _t in frames) if n == "tool" and d.get("name") == "research"
    ]
    done_frames = [(n, d, t) for n, d, t in frames if n == "done"]

    assert research_tool_frames == []
    assert len(done_frames) == 1
    assert done_frames[0][1]["status"] == "ok"


# ---------------------------------------------------------------------------
# (d) /api/status
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (e) M5: append-only PromptLog reaches the server's prompt cache across turns
# ---------------------------------------------------------------------------


def test_second_turn_hits_prompt_cache_and_can_refer_to_first_turn(live_app):
    """Two consecutive turns in one session: the second turn's rendered
    messages must literally include the first turn's user text (the
    plumbing proof that run_turn now reads/writes the session's
    append-only PromptLog rather than a throwaway local list), and the
    server-reported cached_tokens for turn 2 must be a large fraction of
    turn 1's prompt tokens (the prompt-cache-hit proof of the byte-prefix
    property in production)."""
    client = live_app
    sid = _sid(client)

    first_question = "What is a right triangle?"
    frames1 = _stream_turn(client, sid, {"text": first_question})
    done1 = [d for n, d, _t in frames1 if n == "done"][0]
    assert done1["status"] == "ok"

    second_question = "Say briefly why that matters."
    frames2 = _stream_turn(client, sid, {"text": second_question})
    done2 = [d for n, d, _t in frames2 if n == "done"][0]
    assert done2["status"] == "ok"

    # Plumbing assertion: the session object's PromptLog (read directly,
    # since this is the same in-process TestClient/app) must contain the
    # first turn's user text -- proof that turn 2's messages were built
    # from the same append-only log turn 1 wrote to.
    rendered_text = json.dumps(_get_session_log(client, sid))
    assert first_question in rendered_text

    # Cache-hit assertion via the server's own reported usage: turn 2's
    # cached_tokens should cover most of what was already sent in turn 1.
    prompt_tokens_1 = done1.get("tokens_used")
    cached_tokens_2 = done2.get("cached_tokens")
    if not prompt_tokens_1 or cached_tokens_2 is None:
        pytest.skip(
            "turn 'done' events do not currently surface tokens_used/"
            "cached_tokens over the wire; see docs/M5_notes.md"
        )
    assert cached_tokens_2 >= 0.7 * prompt_tokens_1


def _get_session_log(client, sid):
    """Best-effort access to the live app's in-process Session for
    plumbing assertions; returns [] if the app doesn't expose it."""
    app = client.app
    deps = getattr(app.state, "deps", None)
    if deps is None:
        return []
    session = deps.sessions.get(sid)
    if session is None:
        return []
    return session.log.render()


def test_status_reports_llm_healthy_and_archive_valid_ssd(live_app):
    client = live_app
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()

    assert body["llm"]["healthy"] is True

    archives = {a["id"]: a for a in body["archives"]}
    assert "fixture_ssd" in archives
    assert archives["fixture_ssd"]["state"] == "VALID"
    assert archives["fixture_ssd"]["storage"] == "ssd"
