"""RED tests for eval.run_retrieval_eval (recall@k / MRR harness).

Module does not exist yet; every test here should fail with
ModuleNotFoundError until implemented. See
docs/plan/offline_tutor_spec_v0.3.md §15 ("Candidate and extraction
recall: Labeled article in top-64 fused candidates >= 95% ...") and
docs/plan/offline_tutor_kiwix_reuse_plan.md's research-pipeline sections
for the underlying research() engine this eval drives.

Contract decisions:
- Question file format is JSONL, one object per line, fields exactly
  {id, question, expected_paths, category, split}: neither doc specifies
  a file format for the eval set, so JSONL was chosen to match the
  project's existing line-oriented log/config conventions.
- `evaluate()` scores recall@k against `expected_paths` matched by the
  returned passages' `.path` attribute (any expected path appearing
  anywhere in the top-k passages counts as a hit for that k) -- spec §15
  talks about "labeled article in top-N fused candidates", which this
  approximates using the final packed passages since ResearchEngine does
  not expose raw fused candidates as a public contract.
- MRR uses the rank of the first passage whose path is in
  expected_paths (1-indexed), 0 contribution if absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.run_retrieval_eval import (
    FlagsError,
    build_engine,
    evaluate,
    load_questions,
    parse_flags,
    render_markdown,
)


@dataclass
class _FakePassage:
    path: str


class _FakeResponse:
    def __init__(self, passages, status="ok"):
        self.passages = passages
        self.status = status


class _FakeEngine:
    """Deterministic fake: returns a canned ranking per question id."""

    def __init__(self, rankings: dict[str, list[str]], statuses: dict[str, str] | None = None):
        self._rankings = rankings
        self._statuses = statuses or {}
        self.calls = []

    def research(self, query, **kwargs):
        self.calls.append(query)
        paths = self._rankings.get(query, [])
        status = self._statuses.get(query, "ok")
        return _FakeResponse([_FakePassage(p) for p in paths], status=status)


def _write_questions(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_load_questions_parses_jsonl(tmp_path):
    rows = [
        _row("q1", "Q1?", ["a"], category="direct", split="tuning"),
        _row("q2", "Q2?", ["b"], category="why_how", split="heldout"),
    ]
    path = _write_questions(tmp_path, rows)
    questions = load_questions(path)
    assert len(questions) == 2
    assert questions[0].id == "q1"
    assert questions[0].expected_paths == ["a"]
    assert questions[1].split == "heldout"
    assert questions[0].context is None


def test_load_questions_parses_optional_context_field(tmp_path):
    rows = [
        {
            "id": "q1",
            "question": "What about its moons?",
            "expected_paths": ["Jupiter"],
            "category": "elliptical",
            "split": "tuning",
            "context": "Jupiter",
        }
    ]
    path = _write_questions(tmp_path, rows)
    questions = load_questions(path)
    assert questions[0].context == "Jupiter"


def test_evaluate_passes_context_as_topic_hint(tmp_path):
    captured = {}

    class _HintEngine:
        def research(self, query, **kwargs):
            captured["topic_hint"] = kwargs.get("topic_hint")
            return _FakeResponse([_FakePassage("a")])

    questions = load_questions_from_rows(
        [
            {
                "id": "q1",
                "question": "What about its moons?",
                "expected_paths": ["a"],
                "category": "elliptical",
                "split": "tuning",
                "context": "Jupiter",
            }
        ]
    )
    evaluate(_HintEngine(), questions)
    assert captured["topic_hint"] == "Jupiter"


def test_evaluate_recall_at_k_perfect_rank1():
    engine = _FakeEngine({"Q1?": ["a", "x", "y", "z", "w"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions)
    assert report.recall_at_k[1] == 1.0
    assert report.recall_at_k[3] == 1.0
    assert report.recall_at_k[5] == 1.0
    assert report.mrr == 1.0


def test_evaluate_recall_at_k_rank_three_only_counts_from_k3():
    engine = _FakeEngine({"Q1?": ["x", "y", "a", "z", "w"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions)
    assert report.recall_at_k[1] == 0.0
    assert report.recall_at_k[3] == 1.0
    assert report.recall_at_k[5] == 1.0
    assert abs(report.mrr - (1 / 3)) < 1e-9


def test_evaluate_miss_scores_zero_recall_and_mrr():
    engine = _FakeEngine({"Q1?": ["x", "y", "z"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions)
    assert report.recall_at_k[5] == 0.0
    assert report.mrr == 0.0


def test_evaluate_unanswerable_scored_correct_when_status_empty():
    engine = _FakeEngine({"Q1?": []}, statuses={"Q1?": "empty"})
    questions = load_questions_from_rows([_row("q1", "Q1?", [], category="unanswerable")])
    report = evaluate(engine, questions)
    assert report.recall_at_k[1] == 1.0
    assert report.recall_at_k[5] == 1.0
    assert report.mrr == 1.0


def test_evaluate_unanswerable_scored_correct_when_status_partial_with_no_passages():
    # A real off-corpus query can hit the soft deadline while tier 2 is
    # consulted (weak tier-1 coverage) and still correctly find nothing;
    # that "partial + no passages" is still "no coverage", not a miss.
    engine = _FakeEngine({"Q1?": []}, statuses={"Q1?": "partial"})
    questions = load_questions_from_rows([_row("q1", "Q1?", [], category="unanswerable")])
    report = evaluate(engine, questions)
    assert report.recall_at_k[1] == 1.0


def test_evaluate_unanswerable_scored_wrong_when_status_ok_with_passages():
    engine = _FakeEngine({"Q1?": ["x", "y"]}, statuses={"Q1?": "ok"})
    questions = load_questions_from_rows([_row("q1", "Q1?", [], category="unanswerable")])
    report = evaluate(engine, questions)
    assert report.recall_at_k[5] == 0.0
    assert report.mrr == 0.0


def test_evaluate_per_category_breakdown():
    engine = _FakeEngine(
        {"Q1?": ["a"], "Q2?": ["x", "y"]}
    )
    questions = load_questions_from_rows(
        [
            _row("q1", "Q1?", ["a"], category="direct"),
            _row("q2", "Q2?", ["b"], category="why_how"),
        ]
    )
    report = evaluate(engine, questions)
    assert report.per_category["direct"].recall_at_k[1] == 1.0
    assert report.per_category["why_how"].recall_at_k[5] == 0.0


def test_evaluate_mean_latency_is_nonnegative():
    engine = _FakeEngine({"Q1?": ["a"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions)
    assert report.mean_latency_s >= 0.0


def test_render_markdown_returns_string_with_recall():
    engine = _FakeEngine({"Q1?": ["a"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions)
    md = render_markdown(report)
    assert isinstance(md, str)
    assert "recall" in md.lower()


# ---------------------------------------------------------------------------
# End-to-end over the real fixture ZIM + fixture question file.
# ---------------------------------------------------------------------------


def test_end_to_end_fixture_recall_at_5_meets_bar(tmp_path, fixture_zim):
    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.research import ResearchEngine
    from tutor.retrieval.snapshots import SnapshotStore

    toml_path = tmp_path / "registry.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[[archive]]",
                'id = "tier1"',
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
    registry = load_registry(toml_path)
    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    engine = ResearchEngine(registry, snapshot_store=store, cache_dir=tmp_path / "cache")

    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    questions_path = eval_dir / "questions" / "fixture_questions.jsonl"
    questions = load_questions(questions_path)
    assert len(questions) == 12

    report = evaluate(engine, questions)
    store.close()

    assert report.recall_at_k[5] >= 0.75


# ---------------------------------------------------------------------------
# --flags parsing
# ---------------------------------------------------------------------------


def test_parse_flags_none_returns_all_off():
    flags = parse_flags(None)
    assert flags.title_boost is False
    assert flags.mention_penalty is False
    assert flags.heading_affinity is False
    assert flags.lead_augmentation is False


def test_parse_flags_single_flag():
    flags = parse_flags("title_boost")
    assert flags.title_boost is True
    assert flags.mention_penalty is False


def test_parse_flags_multiple_flags_comma_separated():
    flags = parse_flags("title_boost, heading_affinity")
    assert flags.title_boost is True
    assert flags.heading_affinity is True
    assert flags.mention_penalty is False


def test_parse_flags_unknown_flag_raises():
    with pytest.raises(FlagsError):
        parse_flags("not_a_real_flag")


# ---------------------------------------------------------------------------
# dense_used_rate / label / p95
# ---------------------------------------------------------------------------


class _DenseAwareResponse:
    def __init__(self, passages, dense_used):
        self.passages = passages
        self.status = "ok"
        self.dense_used = dense_used


class _DenseAwareEngine:
    def __init__(self, dense_used_by_query):
        self._dense_used = dense_used_by_query

    def research(self, query, **kwargs):
        return _DenseAwareResponse([_FakePassage("a")], self._dense_used[query])


def test_evaluate_reports_dense_used_rate():
    engine = _DenseAwareEngine({"Q1?": True, "Q2?": False})
    questions = load_questions_from_rows(
        [_row("q1", "Q1?", ["a"]), _row("q2", "Q2?", ["a"])]
    )
    report = evaluate(engine, questions)
    assert report.dense_used_rate == pytest.approx(0.5)


def test_evaluate_label_is_carried_into_report_and_markdown():
    engine = _FakeEngine({"Q1?": ["a"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions, label="hybrid+title_boost")
    assert report.label == "hybrid+title_boost"
    md = render_markdown(report)
    assert "hybrid+title_boost" in md


def test_render_markdown_includes_dense_used_rate():
    engine = _FakeEngine({"Q1?": ["a"]})
    questions = load_questions_from_rows([_row("q1", "Q1?", ["a"])])
    report = evaluate(engine, questions)
    md = render_markdown(report)
    assert "dense_used" in md


# ---------------------------------------------------------------------------
# build_engine: lexical-only when no dense-dir/embed-url given, and skips
# archives whose sidecar fingerprint doesn't match (never fatal).
# ---------------------------------------------------------------------------


def test_build_engine_without_dense_args_has_no_dense_indexes(tmp_path, fixture_zim):
    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.snapshots import SnapshotStore

    toml_path = tmp_path / "registry.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[[archive]]",
                'id = "tier1"',
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
    registry = load_registry(toml_path)
    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    try:
        engine = build_engine(registry, snapshot_store=store, cache_dir=tmp_path / "cache")
        assert engine._dense_indexes == {}
    finally:
        store.close()


def test_build_engine_skips_archive_with_no_sidecar_present(tmp_path, fixture_zim):
    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.snapshots import SnapshotStore

    toml_path = tmp_path / "registry.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[[archive]]",
                'id = "tier1"',
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
    registry = load_registry(toml_path)
    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    try:
        engine = build_engine(
            registry,
            snapshot_store=store,
            cache_dir=tmp_path / "cache",
            dense_dir=tmp_path / "no_such_sidecar",
            embed_url="http://127.0.0.1:1",
        )
        assert engine._dense_indexes == {}
        assert engine._embed_query is None
    finally:
        store.close()


def _row(qid, question, expected_paths, category="direct", split="tuning"):
    return {
        "id": qid,
        "question": question,
        "expected_paths": expected_paths,
        "category": category,
        "split": split,
    }


def load_questions_from_rows(rows):
    """Test-local helper: build Question objects without touching disk."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = _write_questions(Path(td), rows)
        return load_questions(path)
