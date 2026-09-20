"""Citation-rate experiment harness (task: raise the tutor's [S#] citation
rate against the live 1-bit 8B model by tuning host-controlled levers
only: system prompt, evidence rendering, evidence placement, sampling
temperature). See docs/citation_experiment.md for the write-up.

This module has two halves:

- Pure, network-free scoring/report functions (``score_answer``,
  ``aggregate``, ``render_report``) -- unit tested with fakes in
  ``tests/test_run_turn_eval.py``.
- A live runner (``run_variant``, ``main``) that drives the REAL turn path
  (``tutor.app.agent_loop.run_turn`` over a real ``LlamaClient`` and a
  real ``ResearchEngine`` against the libzim fixture archive) for each
  variant, against ``config/dev.toml``. Skipped/aborted cleanly if
  ``/health`` is down.

Run explicitly:
    python -m eval.run_turn_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tutor.app.citations import extract_labels, resolve_citations

_QUESTIONS_PATH = Path(__file__).parent / "questions" / "fixture_questions.jsonl"

# Only these articles actually exist in tests/zim_fixtures.py's fixture ZIM
# (see that module's _RICH_ARTICLES); the rest of fixture_questions.jsonl
# (photosynthesis, water cycle, WWII, ...) has no corresponding archive
# content here and is out of scope for this harness.
_FIXTURE_ARTICLE_IDS = {
    "pythagorean_theorem",
    "algebra_basics",
    "history_of_mathematics",
    "geometry_intro",
    "erdos_number",
}

_TEACHING_MARKERS = ("?", "try", "step", "first,", "next,")


def load_questions(path: Path = _QUESTIONS_PATH) -> list[dict]:
    """Load questions whose ``expected_paths`` are covered by the fixture
    ZIM, in file order."""
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if any(p in _FIXTURE_ARTICLE_IDS for p in row.get("expected_paths", [])):
                questions.append(row)
    return questions


@dataclass
class AnswerRecord:
    """One scored turn."""

    question_id: str
    question: str
    expected_paths: list[str]
    answer_text: str
    citations: list[dict] = field(default_factory=list)


def score_answer(record: AnswerRecord) -> dict:
    """Mechanical, deterministic scoring of one answer. Never guesses
    intent; every field is a simple derived fact."""
    labels = extract_labels(record.answer_text)
    has_citation = len(labels) >= 1

    resolved = [c for c in record.citations if not c.get("unresolved")]
    unresolved = [c for c in record.citations if c.get("unresolved")]
    all_resolve = has_citation and not unresolved

    on_topic = any(
        c.get("path") in record.expected_paths for c in resolved if c.get("path")
    )

    lowered = record.answer_text.lower()
    taught = any(marker in lowered for marker in _TEACHING_MARKERS)

    return {
        "question_id": record.question_id,
        "has_citation": has_citation,
        "n_labels": len(labels),
        "all_resolve": all_resolve,
        "n_unresolved": len(unresolved),
        "on_topic_citation": on_topic,
        "answer_len_chars": len(record.answer_text),
        "taught": taught,
    }


def aggregate(scores: list[dict]) -> dict:
    """Roll up per-question ``score_answer`` dicts into variant-level
    rates. Empty input yields zeroed rates rather than raising."""
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "citation_rate": 0.0,
            "all_resolve_rate": 0.0,
            "on_topic_rate": 0.0,
            "taught_rate": 0.0,
            "avg_answer_len_chars": 0.0,
        }
    return {
        "n": n,
        "citation_rate": sum(s["has_citation"] for s in scores) / n,
        "all_resolve_rate": sum(s["all_resolve"] for s in scores) / n,
        "on_topic_rate": sum(s["on_topic_citation"] for s in scores) / n,
        "taught_rate": sum(s["taught"] for s in scores) / n,
        "avg_answer_len_chars": sum(s["answer_len_chars"] for s in scores) / n,
    }


def render_report(variant_summaries: dict[str, dict]) -> str:
    """Render a Markdown table, one row per variant, in insertion order."""
    header = (
        "| Variant | n | citation_rate | all_resolve_rate | on_topic_rate "
        "| taught_rate | avg_len |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for name, summary in variant_summaries.items():
        rows.append(
            f"| {name} | {summary['n']} | {summary['citation_rate']:.2f} "
            f"| {summary['all_resolve_rate']:.2f} | {summary['on_topic_rate']:.2f} "
            f"| {summary['taught_rate']:.2f} | {summary['avg_answer_len_chars']:.0f} |"
        )
    return "\n".join([header, *rows])


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------


def run_variant(
    questions: list[dict],
    *,
    make_session: Callable[[], Any],
    llm: Any,
    research_engine: Any,
    calc: Any,
    budget: Any,
    system_text: str,
    render_evidence_fn: Callable[[dict], str] | None = None,
    temperature: float | None = None,
) -> list[dict]:
    """Run every question through the real ``run_turn`` path for one
    variant, patching only the host-controlled knobs
    (``tutor.app.agent_loop``'s system text / evidence renderer /
    temperature), and return per-question ``score_answer`` dicts."""
    import tutor.app.agent_loop as agent_loop
    from tutor.app.compose import _UserInputLike

    orig_render_evidence = agent_loop.render_evidence
    if render_evidence_fn is not None:
        agent_loop.render_evidence = render_evidence_fn  # type: ignore[attr-defined]

    scores = []
    try:
        for row in questions:
            session = make_session()
            user_input = _UserInputLike(kind="text", text=row["question"])
            result = agent_loop.run_turn(
                session,
                user_input,
                llm=llm,
                research_engine=research_engine,
                calc=calc,
                budget=budget,
                emit=lambda _obj: None,
                system_text_override=system_text,
                temperature=temperature,
            )
            citations = [
                {
                    "label": c.label,
                    "path": c.path,
                    "unresolved": c.unresolved,
                }
                for c in resolve_citations(result.answer_text, session.known_passages())
            ]
            record = AnswerRecord(
                question_id=row["id"],
                question=row["question"],
                expected_paths=row["expected_paths"],
                answer_text=result.answer_text,
                citations=citations,
            )
            scores.append(score_answer(record))
    finally:
        agent_loop.render_evidence = orig_render_evidence

    return scores


def _build_live_collaborators(config_path: str, zim_path: Path, tmp_path: Path):
    from tutor.app.llm_client import LlamaClient
    from tutor.app.prompt import Budget
    from tutor.app.session import Session
    from tutor.retrieval.registry import Registry, load_registry
    from tutor.retrieval.research import ResearchEngine
    from tutor.retrieval.snapshots import SnapshotStore
    from tutor.settings import load_config

    cfg = load_config(config_path)
    llm = LlamaClient(cfg.server.base_url, timeout_s=10)
    if not llm.health():
        return None

    registry_path = tmp_path / "archives.toml"
    registry_path.write_text(
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
    registry: Registry = load_registry(registry_path)
    snapshot_store = SnapshotStore(tmp_path / "snapshots.sqlite")
    research_engine = ResearchEngine(
        registry, snapshot_store=snapshot_store, cache_dir=tmp_path / "research_cache"
    )
    budget = Budget.for_ceiling(cfg.server.ctx_size)

    def count_tokens(text: str) -> int:
        try:
            return llm.count_tokens(text)
        except Exception:  # noqa: BLE001
            return max(1, len(text) // 4)

    def make_session():
        return Session(count_tokens)

    class _CalcTool:
        def evaluate(self, expression: str) -> dict:
            from tutor.tools import calc_tool

            return calc_tool.evaluate(expression)

    return {
        "llm": llm,
        "research_engine": research_engine,
        "calc": _CalcTool(),
        "budget": budget,
        "make_session": make_session,
    }


def main(argv: list[str] | None = None) -> int:
    import shutil
    import tempfile

    from tests.zim_fixtures import build_zim

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.toml")
    args = parser.parse_args(argv)

    questions = load_questions()

    tmp = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp)
        zim_path = build_zim(tmp_path / "fixture.zim", indexing=True)
        collaborators = _build_live_collaborators(args.config, zim_path, tmp_path)
        if collaborators is None:
            print("llama-server /health unreachable; skipping live experiment.")
            return 0

        from eval.system_prompt_variants import VARIANTS

        summaries: dict[str, dict] = {}
        t0 = time.monotonic()
        for name, variant in VARIANTS.items():
            scores = run_variant(
                questions,
                make_session=collaborators["make_session"],
                llm=collaborators["llm"],
                research_engine=collaborators["research_engine"],
                calc=collaborators["calc"],
                budget=collaborators["budget"],
                system_text=variant["system_text"],
                render_evidence_fn=variant.get("render_evidence_fn"),
                temperature=variant.get("temperature"),
            )
            summaries[name] = aggregate(scores)
            print(f"[{name}] done, elapsed {time.monotonic() - t0:.0f}s")

        print(render_report(summaries))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
