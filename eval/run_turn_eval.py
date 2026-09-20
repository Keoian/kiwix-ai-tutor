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

from tutor.app.citations import detect_evidence_dump, extract_labels, resolve_citations

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


def load_questions(path: Path = _QUESTIONS_PATH, *, filter_to_fixture: bool = True) -> list[dict]:
    """Load questions in file order. When ``filter_to_fixture`` (default,
    used by the fixture-ZIM path) only rows whose ``expected_paths`` are
    covered by the fixture ZIM are kept; real-archive question files (e.g.
    ``eval/questions/simplewiki_questions.jsonl``) pass ``filter_to_fixture=
    False`` since there is no fixture-article allowlist for them."""
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not filter_to_fixture or any(
                p in _FIXTURE_ARTICLE_IDS for p in row.get("expected_paths", [])
            ):
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
    evidence_dump: bool = False
    """2026-09-20 evidence-dump follow-up (docs/citation_experiment.md):
    the host's ``tutor.app.citations.detect_evidence_dump`` verdict for
    this answer, already computed by the caller (``run_variant`` below)
    since it needs the retrieval packet's passages, which this record does
    not otherwise carry."""


def score_answer(record: AnswerRecord) -> dict:
    """Mechanical, deterministic scoring of one answer. Never guesses
    intent; every field is a simple derived fact."""
    labels = extract_labels(record.answer_text)
    has_citation = len(labels) >= 1

    resolved = [c for c in record.citations if not c.get("unresolved")]
    unresolved = [c for c in record.citations if c.get("unresolved")]
    all_resolve = has_citation and not unresolved

    # 2026-09-20 evidence-dump follow-up: a citation "resolving" is not the
    # same as it "supporting" the sentence that carries it (see
    # tutor.app.citations.is_supported / Citation.supported). All labels
    # here have to resolve *and* be marked supported for the answer to
    # count as fully citation-supported.
    all_supported = has_citation and not unresolved and all(
        c.get("supported") for c in record.citations
    )

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
        "all_supported": all_supported,
        "n_unresolved": len(unresolved),
        "on_topic_citation": on_topic,
        "answer_len_chars": len(record.answer_text),
        "taught": taught,
        "evidence_dump": record.evidence_dump,
        "uncited": not has_citation,
    }


def filter_questions(
    rows: list[dict],
    *,
    split: str | None = None,
    categories: list[str] | None = None,
    n: int | None = None,
) -> list[dict]:
    """Filter loaded question rows by ``split``/``categories`` (both
    optional, applied in that order), then take a deterministic first-``n``
    (in the input's own order) if ``n`` is given. Pure, no I/O."""
    out = rows
    if split is not None:
        out = [r for r in out if r.get("split") == split]
    if categories is not None:
        cats = set(categories)
        out = [r for r in out if r.get("category") in cats]
    if n is not None:
        out = out[:n]
    return out


def aggregate(scores: list[dict]) -> dict:
    """Roll up per-question ``score_answer`` dicts into variant-level
    rates. Empty input yields zeroed rates rather than raising."""
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "citation_rate": 0.0,
            "all_resolve_rate": 0.0,
            "supported_citation_rate": 0.0,
            "evidence_dump_rate": 0.0,
            "on_topic_rate": 0.0,
            "taught_rate": 0.0,
            "avg_answer_len_chars": 0.0,
        }
    return {
        "n": n,
        "citation_rate": sum(s["has_citation"] for s in scores) / n,
        "all_resolve_rate": sum(s["all_resolve"] for s in scores) / n,
        "supported_citation_rate": sum(s["all_supported"] for s in scores) / n,
        "evidence_dump_rate": sum(s["evidence_dump"] for s in scores) / n,
        "on_topic_rate": sum(s["on_topic_citation"] for s in scores) / n,
        "taught_rate": sum(s["taught"] for s in scores) / n,
        "avg_answer_len_chars": sum(s["answer_len_chars"] for s in scores) / n,
    }


def render_report(variant_summaries: dict[str, dict]) -> str:
    """Render a Markdown table, one row per variant, in insertion order."""
    header = (
        "| Variant | n | citation_rate | all_resolve_rate | supported_citation_rate "
        "| evidence_dump_rate | on_topic_rate | taught_rate | avg_len |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for name, summary in variant_summaries.items():
        rows.append(
            f"| {name} | {summary['n']} | {summary['citation_rate']:.2f} "
            f"| {summary['all_resolve_rate']:.2f} | {summary['supported_citation_rate']:.2f} "
            f"| {summary['evidence_dump_rate']:.2f} | {summary['on_topic_rate']:.2f} "
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
    details = []
    try:
        for row in questions:
            session = make_session()
            user_input = _UserInputLike(kind="text", text=row["question"])

            first_token_ms: float | None = None
            t_start = time.monotonic()

            def _timed_emit(evt, _t_start=t_start):
                nonlocal first_token_ms
                if first_token_ms is None and getattr(evt, "kind", None) == "token":
                    first_token_ms = (time.monotonic() - _t_start) * 1000.0

            result = agent_loop.run_turn(
                session,
                user_input,
                llm=llm,
                research_engine=research_engine,
                calc=calc,
                budget=budget,
                emit=_timed_emit,
                system_text_override=system_text,
                temperature=temperature,
            )
            total_ms = (time.monotonic() - t_start) * 1000.0

            known_passages = session.known_passages()
            resolved_citations = resolve_citations(result.answer_text, known_passages)
            citations = [
                {
                    "label": c.label,
                    "path": c.path,
                    "unresolved": c.unresolved,
                    "supported": c.supported,
                }
                for c in resolved_citations
            ]
            record = AnswerRecord(
                question_id=row["id"],
                question=row["question"],
                expected_paths=row["expected_paths"],
                answer_text=result.answer_text,
                citations=citations,
                evidence_dump=detect_evidence_dump(
                    result.answer_text, resolved_citations, known_passages
                ),
            )
            scored = score_answer(record)
            scored["calc_calls"] = result.calc_calls
            scored["first_token_ms"] = first_token_ms
            scored["total_ms"] = total_ms
            scored["prompt_tokens"] = result.prompt_tokens
            scored["passages_in_packet"] = len(known_passages)
            scores.append(scored)
            details.append(
                {
                    **scored,
                    "question": row["question"],
                    "answer_text": result.answer_text,
                    "citations": citations,
                    "route": result.route,
                }
            )
    finally:
        agent_loop.render_evidence = orig_render_evidence

    run_variant.last_details = details  # type: ignore[attr-defined]
    return scores


def _build_live_collaborators(
    config_path: str, zim_path: Path | None, tmp_path: Path, registry_path: Path | None = None
):
    from tutor.app.llm_client import LlamaClient
    from tutor.app.prompt import Budget
    from tutor.app.session import Session
    from tutor.retrieval.registry import Registry, load_registry
    from tutor.retrieval.research import ResearchEngine
    from tutor.retrieval.snapshots import SnapshotStore
    from tutor.settings import load_config

    cfg = load_config(config_path)
    # 2026-09-20 live re-measurement: 10s was far too short for real-archive
    # prompt prefill on this GPU (harness note: "large prompt prefill is
    # slow") -- RUN A's first pass showed a wall of "Sorry, I ran into a
    # problem" answers that were actually HTTP timeouts, not model
    # failures. 120s covers a cold prefill plus generation comfortably.
    llm = LlamaClient(cfg.server.base_url, timeout_s=120)
    if not llm.health():
        return None

    if registry_path is None:
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.toml")
    parser.add_argument(
        "--registry", default=None, help="Archive registry TOML; default: temp fixture registry"
    )
    parser.add_argument(
        "--questions", default=None, help="Questions JSONL; default: fixture question set"
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--categories", default=None, help="Comma-separated category allowlist")
    parser.add_argument(
        "--n", type=int, default=None, help="Deterministic first-n after filtering"
    )
    parser.add_argument(
        "--variants", default=None, help="Comma-separated variant names; default: all"
    )
    parser.add_argument(
        "--out", default=None, help="Markdown report path; also writes a sibling .json dump"
    )
    args = parser.parse_args(argv)

    use_fixture = args.registry is None
    if args.questions is not None:
        questions = load_questions(Path(args.questions), filter_to_fixture=use_fixture)
    else:
        questions = load_questions()

    categories = args.categories.split(",") if args.categories else None
    questions = filter_questions(questions, split=args.split, categories=categories, n=args.n)

    tmp = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp)
        zim_path = None
        if use_fixture:
            from tests.zim_fixtures import build_zim

            zim_path = build_zim(tmp_path / "fixture.zim", indexing=True)
        collaborators = _build_live_collaborators(
            args.config,
            zim_path,
            tmp_path,
            registry_path=Path(args.registry) if args.registry else None,
        )
        if collaborators is None:
            print("llama-server /health unreachable; skipping live experiment.")
            return 0

        from eval.system_prompt_variants import VARIANTS

        available = dict(VARIANTS)
        available["current"] = {"system_text": None}

        if args.variants:
            names = args.variants.split(",")
            variant_items = [(n, available[n]) for n in names if n in available]
        else:
            variant_items = list(available.items())

        summaries: dict[str, dict] = {}
        all_details: dict[str, list] = {}
        t0 = time.monotonic()
        for name, variant in variant_items:
            system_text = variant.get("system_text")
            if system_text is None:
                import tutor.app.agent_loop as agent_loop

                system_text = agent_loop._load_default_system_text()
            scores = run_variant(
                questions,
                make_session=collaborators["make_session"],
                llm=collaborators["llm"],
                research_engine=collaborators["research_engine"],
                calc=collaborators["calc"],
                budget=collaborators["budget"],
                system_text=system_text,
                render_evidence_fn=variant.get("render_evidence_fn"),
                temperature=variant.get("temperature"),
            )
            summaries[name] = aggregate(scores)
            all_details[name] = getattr(run_variant, "last_details", [])
            print(f"[{name}] done, elapsed {time.monotonic() - t0:.0f}s")

        report = render_report(summaries)
        print(report)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report + "\n", encoding="utf-8")
            json_path = Path("data") / (out_path.stem + ".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(all_details, indent=2), encoding="utf-8")
            print(f"wrote {out_path} and {json_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
