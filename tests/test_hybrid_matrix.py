"""Tests for eval.tools.hybrid_matrix: selection, gate verdict, per-flag
verdicts, and rendering are pure logic tested against fake EvalReports;
the engine-running part (``run_matrix``'s ``engine_builder``) is injected
so these never need a real ZIM archive or embedding server. One
end-to-end smoke test exercises the whole driver against the fixture ZIM
with a tiny sidecar built by the deterministic fake embedder (see
tests/test_simplewiki_build.py), still with no network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.run_retrieval_eval import EvalReport, Question, build_engine
from eval.tools.hybrid_matrix import (
    _BASELINE_LABEL,
    _HYBRID_LABEL,
    ConfigSpec,
    flag_verdicts,
    gate_verdict,
    hybrid_configs,
    pick_best_hybrid,
    render_matrix_markdown,
    run_matrix,
)
from tutor.retrieval.index.simplewiki_build import build_index


def _report(recall5, mrr, recall1=None, n=10) -> EvalReport:
    r1 = recall1 if recall1 is not None else recall5
    return EvalReport(
        recall_at_k={1: r1, 3: recall5, 5: recall5},
        mrr=mrr,
        mean_latency_s=0.01,
        n=n,
        p95_latency_s=0.02,
        dense_used_rate=0.0,
    )


def _configs_by_label():
    return {c.label: c for c in hybrid_configs()}


# ---------------------------------------------------------------------------
# hybrid_configs
# ---------------------------------------------------------------------------


def test_hybrid_configs_has_baseline_plus_four_flags_plus_no_flag_hybrid():
    configs = hybrid_configs()
    labels = [c.label for c in configs]
    assert _BASELINE_LABEL in labels
    assert _HYBRID_LABEL in labels
    assert len(labels) == 6
    assert configs[0].dense is False
    assert all(c.dense for c in configs[1:])


# ---------------------------------------------------------------------------
# pick_best_hybrid
# ---------------------------------------------------------------------------


def test_pick_best_hybrid_picks_highest_recall5():
    configs_by_label = _configs_by_label()
    reports = {
        _HYBRID_LABEL: _report(0.80, 0.70),
        "hybrid+title_boost": _report(0.90, 0.70),
        "hybrid+mention_penalty": _report(0.70, 0.90),
        "hybrid+heading_affinity": _report(0.80, 0.60),
        "hybrid+lead_augmentation": _report(0.80, 0.60),
    }
    assert pick_best_hybrid(reports, configs_by_label) == "hybrid+title_boost"


def test_pick_best_hybrid_breaks_recall5_tie_with_mrr():
    configs_by_label = _configs_by_label()
    reports = {
        _HYBRID_LABEL: _report(0.80, 0.70),
        "hybrid+title_boost": _report(0.80, 0.85),
        "hybrid+mention_penalty": _report(0.80, 0.60),
        "hybrid+heading_affinity": _report(0.80, 0.60),
        "hybrid+lead_augmentation": _report(0.80, 0.60),
    }
    assert pick_best_hybrid(reports, configs_by_label) == "hybrid+title_boost"


def test_pick_best_hybrid_full_tie_prefers_fewer_flags():
    configs_by_label = _configs_by_label()
    reports = {
        _HYBRID_LABEL: _report(0.80, 0.70),
        "hybrid+title_boost": _report(0.80, 0.70),
        "hybrid+mention_penalty": _report(0.80, 0.70),
        "hybrid+heading_affinity": _report(0.80, 0.70),
        "hybrid+lead_augmentation": _report(0.80, 0.70),
    }
    assert pick_best_hybrid(reports, configs_by_label) == _HYBRID_LABEL


def test_pick_best_hybrid_never_returns_baseline():
    configs_by_label = _configs_by_label()
    reports = {
        _BASELINE_LABEL: _report(0.99, 0.99),
        _HYBRID_LABEL: _report(0.10, 0.10),
        "hybrid+title_boost": _report(0.10, 0.10),
        "hybrid+mention_penalty": _report(0.10, 0.10),
        "hybrid+heading_affinity": _report(0.10, 0.10),
        "hybrid+lead_augmentation": _report(0.10, 0.10),
    }
    assert pick_best_hybrid(reports, configs_by_label) != _BASELINE_LABEL


# ---------------------------------------------------------------------------
# gate_verdict
# ---------------------------------------------------------------------------


def test_gate_verdict_pass_when_tuning_improves_and_heldout_not_worse():
    baseline_tuning = _report(0.70, 0.60)
    chosen_tuning = _report(0.80, 0.65)
    baseline_heldout = _report(0.70, 0.60)
    chosen_heldout = _report(0.75, 0.62)
    assert gate_verdict(baseline_tuning, chosen_tuning, baseline_heldout, chosen_heldout) == "PASS"


def test_gate_verdict_fail_when_tuning_does_not_improve():
    baseline_tuning = _report(0.80, 0.70)
    chosen_tuning = _report(0.80, 0.70)
    baseline_heldout = _report(0.70, 0.60)
    chosen_heldout = _report(0.75, 0.62)
    assert gate_verdict(baseline_tuning, chosen_tuning, baseline_heldout, chosen_heldout) == "FAIL"


def test_gate_verdict_fail_when_heldout_recall_regresses():
    baseline_tuning = _report(0.70, 0.60)
    chosen_tuning = _report(0.80, 0.65)
    baseline_heldout = _report(0.70, 0.60)
    chosen_heldout = _report(0.60, 0.62)
    assert gate_verdict(baseline_tuning, chosen_tuning, baseline_heldout, chosen_heldout) == "FAIL"


def test_gate_verdict_fail_when_heldout_mrr_regresses():
    baseline_tuning = _report(0.70, 0.60)
    chosen_tuning = _report(0.80, 0.65)
    baseline_heldout = _report(0.70, 0.60)
    chosen_heldout = _report(0.75, 0.50)
    assert gate_verdict(baseline_tuning, chosen_tuning, baseline_heldout, chosen_heldout) == "FAIL"


def test_gate_verdict_pass_when_heldout_exactly_equal():
    baseline_tuning = _report(0.70, 0.60)
    chosen_tuning = _report(0.80, 0.65)
    baseline_heldout = _report(0.70, 0.60)
    chosen_heldout = _report(0.70, 0.60)
    assert gate_verdict(baseline_tuning, chosen_tuning, baseline_heldout, chosen_heldout) == "PASS"


# ---------------------------------------------------------------------------
# flag_verdicts
# ---------------------------------------------------------------------------


def test_flag_verdicts_enable_only_flags_beating_no_flag_hybrid():
    configs_by_label = _configs_by_label()
    reports = {
        _HYBRID_LABEL: _report(0.80, 0.70),
        "hybrid+title_boost": _report(0.85, 0.70),  # beats -> enable
        "hybrid+mention_penalty": _report(0.75, 0.70),  # worse -> keep off
        "hybrid+heading_affinity": _report(0.80, 0.71),  # tie recall, better mrr -> enable
        "hybrid+lead_augmentation": _report(0.80, 0.69),  # tie recall, worse mrr -> keep off
    }
    verdicts = flag_verdicts(reports, configs_by_label)
    assert verdicts["title_boost"] == "enable"
    assert verdicts["mention_penalty"] == "keep off"
    assert verdicts["heading_affinity"] == "enable"
    assert verdicts["lead_augmentation"] == "keep off"


# ---------------------------------------------------------------------------
# render_matrix_markdown
# ---------------------------------------------------------------------------


def test_render_matrix_markdown_contains_gate_and_chosen_and_flag_table():
    tuning = {
        _BASELINE_LABEL: _report(0.70, 0.60),
        _HYBRID_LABEL: _report(0.80, 0.65),
        "hybrid+title_boost": _report(0.85, 0.70),
        "hybrid+mention_penalty": _report(0.75, 0.60),
        "hybrid+heading_affinity": _report(0.80, 0.65),
        "hybrid+lead_augmentation": _report(0.80, 0.65),
    }
    heldout = {
        _BASELINE_LABEL: _report(0.70, 0.60),
        "hybrid+title_boost": _report(0.78, 0.68),
    }
    verdicts = {
        "title_boost": "enable",
        "mention_penalty": "keep off",
        "heading_affinity": "keep off",
        "lead_augmentation": "keep off",
    }
    md = render_matrix_markdown(
        tuning_reports=tuning,
        heldout_reports=heldout,
        chosen_label="hybrid+title_boost",
        gate="PASS",
        flag_verdicts_by_name=verdicts,
    )
    assert "hybrid+title_boost" in md
    assert "PASS" in md
    assert "title_boost | enable" in md
    assert "mention_penalty | keep off" in md
    assert _BASELINE_LABEL in md


# ---------------------------------------------------------------------------
# run_matrix orchestration with an injected fake engine_builder.
# ---------------------------------------------------------------------------


@pytest.fixture
def three_questions() -> list[Question]:
    return [
        Question("q1", "Q1?", ["a"], "direct", "tuning"),
        Question("q2", "Q2?", ["b"], "direct", "tuning"),
        Question("q3", "Q3?", ["c"], "direct", "heldout"),
    ]


@dataclass
class _FakePassage:
    path: str


class _FakeResponse:
    def __init__(self, passages) -> None:
        self.passages = passages
        self.status = "ok"
        self.dense_used = False


class _FakeEngine:
    """Every config gets a perfect score except title_boost, which is best."""

    def __init__(self, spec: ConfigSpec) -> None:
        self._spec = spec

    def research(self, query, **kwargs):
        expected = {"Q1?": "a", "Q2?": "b", "Q3?": "c"}[query]
        if self._spec.label == "hybrid+title_boost":
            return _FakeResponse([_FakePassage(expected)])
        if self._spec.dense:
            return _FakeResponse([_FakePassage("wrong"), _FakePassage(expected)])
        return _FakeResponse([_FakePassage("wrong"), _FakePassage("wrong2")])


def test_run_matrix_end_to_end_with_fake_engine_builder(three_questions):
    tuning_reports, heldout_reports, chosen_label, gate, verdicts = run_matrix(
        three_questions, engine_builder=_FakeEngine
    )
    assert set(tuning_reports) == {c.label for c in hybrid_configs()}
    assert chosen_label == "hybrid+title_boost"
    assert set(heldout_reports) == {_BASELINE_LABEL, "hybrid+title_boost"}
    assert gate == "PASS"
    assert verdicts["title_boost"] == "enable"


# ---------------------------------------------------------------------------
# Full offline smoke test: real fixture ZIM, real (tiny) dense sidecar built
# by the deterministic fake embedder, driven through the real build_engine.
# Never touches a network or ports 8080/8081.
# ---------------------------------------------------------------------------

_DENSE_DIM = 8


def _fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:_DENSE_DIM]])
    return out


def test_run_matrix_smoke_test_over_fixture_zim(tmp_path, fixture_zim, monkeypatch):
    # Build a tiny real sidecar for the fixture archive.
    sidecar_dir = tmp_path / "sidecar"
    build_index(
        fixture_zim,
        sidecar_dir,
        _fake_embed,
        archive_id="fixture",
        embedding_model_name="fake-embedder-v1",
        embedding_model_sha256="0" * 64,
        dim=_DENSE_DIM,
        batch_size=3,
        checkpoint_every=1,
    )

    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        "\n".join(
            [
                "[[archive]]",
                'id = "fixture"',
                f'path = "{fixture_zim.as_posix()}"',
                "tier = 1",
                'kind = "encyclopedia"',
                "subjects = []",
                'storage = "ssd"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    questions_path = (
        Path(__file__).resolve().parent.parent / "eval" / "questions" / "fixture_questions.jsonl"
    )
    raw_lines = questions_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines if line.strip()]
    # fixture_questions.jsonl is all "tuning" (no held-out split); relabel a
    # couple of rows as held-out so the driver has something to run there
    # too, without needing a second question file for this smoke test.
    for row in rows[:2]:
        row["split"] = "heldout"
    tmp_questions = tmp_path / "questions.jsonl"
    tmp_questions.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    from eval.run_retrieval_eval import load_questions
    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.snapshots import SnapshotStore

    questions = load_questions(tmp_questions)
    registry = load_registry(registry_path)

    # Never a real network call: EmbeddingClient.embed is monkeypatched to
    # the same deterministic fake used to build the sidecar, so no port
    # (8080/8081 or otherwise) is ever touched.
    from tutor.retrieval.index import embedding_client as embedding_client_module

    monkeypatch.setattr(
        embedding_client_module.EmbeddingClient, "embed", lambda self, texts: _fake_embed(texts)
    )

    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    try:
        def engine_builder(spec: ConfigSpec):
            return build_engine(
                registry,
                snapshot_store=store,
                cache_dir=tmp_path / "cache" / spec.label,
                dense_dir=sidecar_dir if spec.dense else None,
                embed_url="http://127.0.0.1:59999" if spec.dense else None,
                ranking_flags=spec.flags,
            )

        tuning_reports, heldout_reports, chosen_label, gate, verdicts = run_matrix(
            questions, engine_builder=engine_builder
        )
    finally:
        store.close()

    assert set(tuning_reports) == {c.label for c in hybrid_configs()}
    assert _BASELINE_LABEL in heldout_reports
    assert chosen_label in heldout_reports
    assert gate in ("PASS", "FAIL")
    assert set(verdicts) == {
        "title_boost",
        "mention_penalty",
        "heading_affinity",
        "lead_augmentation",
    }

    md = render_matrix_markdown(
        tuning_reports=tuning_reports,
        heldout_reports=heldout_reports,
        chosen_label=chosen_label,
        gate=gate,
        flag_verdicts_by_name=verdicts,
    )
    assert "M4 gate" in md
