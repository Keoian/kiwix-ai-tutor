"""Recall@k / MRR evaluation harness for :class:`tutor.retrieval.research.ResearchEngine`.

Ours: no donor code. See docs/plan/offline_tutor_spec_v0.3.md §15
("Candidate and extraction recall") for the metric this approximates.
Question file format is JSONL, one object per line, with exactly the
fields ``{id, question, expected_paths, category, split}``.

CLI:
    python -m eval.run_retrieval_eval --registry config/archives.dev.toml \\
        --questions eval/questions/fixture_questions.jsonl \\
        --out docs/retrieval_baseline.md [--split tuning|heldout]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_KS = (1, 3, 5)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected_paths: list[str]
    category: str
    split: str
    # Host-supplied conversational context (spec §5 step 3:
    # research(query=message, topic_hint=current_subject)). Populated for
    # elliptical follow-ups whose referent lives in prior turns; None for
    # every other category.
    context: str | None = None


def load_questions(path: Path) -> list[Question]:
    """Parse a JSONL question file into :class:`Question` objects."""
    questions: list[Question] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions.append(
                Question(
                    id=row["id"],
                    question=row["question"],
                    expected_paths=list(row["expected_paths"]),
                    category=row.get("category", "direct"),
                    split=row.get("split", "tuning"),
                    context=row.get("context"),
                )
            )
    return questions


@dataclass
class EvalReport:
    """Aggregate recall@k / MRR / latency, optionally broken down by category."""

    recall_at_k: dict[int, float]
    mrr: float
    mean_latency_s: float
    n: int = 0
    per_category: dict[str, EvalReport] = field(default_factory=dict)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_one(
    engine: Any, question: Question, ks: tuple[int, ...]
) -> tuple[dict[int, float], float, float]:
    started = time.monotonic()
    response = engine.research(question.question, topic_hint=question.context)
    elapsed = time.monotonic() - started

    if not question.expected_paths:
        # Unanswerable/absent items: correct iff the engine reports no (or
        # weak) coverage rather than confidently returning passages. See
        # docs/plan/offline_tutor_spec_v0.3.md §15 "ambiguity/false-premise/
        # absent" category. "partial" with no passages also counts:
        # against the real tier-2 archives, an off-corpus query with weak
        # tier-1 coverage can legitimately hit the soft deadline while tier
        # 2 is consulted and still correctly find nothing -- that is still
        # "no coverage", not a wrong confident answer.
        status = getattr(response, "status", "ok")
        correct = status in ("empty", "partial") and not response.passages
        value = 1.0 if correct else 0.0
        return {k: value for k in ks}, value, elapsed

    paths = [p.path for p in response.passages]
    hit_rank: int | None = None
    for i, path in enumerate(paths):
        if path in question.expected_paths:
            hit_rank = i + 1
            break
    recall = {k: (1.0 if hit_rank is not None and hit_rank <= k else 0.0) for k in ks}
    reciprocal_rank = (1.0 / hit_rank) if hit_rank else 0.0
    return recall, reciprocal_rank, elapsed


def evaluate(
    engine: Any, questions: list[Question], ks: tuple[int, ...] = _DEFAULT_KS
) -> EvalReport:
    """Run ``engine.research`` over every question and aggregate metrics."""
    buckets: dict[str, dict[str, Any]] = {}
    overall_recall: dict[int, list[float]] = {k: [] for k in ks}
    overall_rr: list[float] = []
    overall_latency: list[float] = []

    for question in questions:
        recall, rr, elapsed = _score_one(engine, question, ks)
        for k in ks:
            overall_recall[k].append(recall[k])
        overall_rr.append(rr)
        overall_latency.append(elapsed)

        bucket = buckets.setdefault(
            question.category, {"recall": {k: [] for k in ks}, "rr": [], "latency": []}
        )
        for k in ks:
            bucket["recall"][k].append(recall[k])
        bucket["rr"].append(rr)
        bucket["latency"].append(elapsed)

    per_category = {
        category: EvalReport(
            recall_at_k={k: _mean(v["recall"][k]) for k in ks},
            mrr=_mean(v["rr"]),
            mean_latency_s=_mean(v["latency"]),
            n=len(v["rr"]),
        )
        for category, v in buckets.items()
    }

    return EvalReport(
        recall_at_k={k: _mean(overall_recall[k]) for k in ks},
        mrr=_mean(overall_rr),
        mean_latency_s=_mean(overall_latency),
        n=len(questions),
        per_category=per_category,
    )


def render_markdown(report: EvalReport) -> str:
    """Render ``report`` as a small Markdown document."""
    lines = ["# Retrieval evaluation report", ""]
    ks = sorted(report.recall_at_k)
    header = (
        "| metric | " + " | ".join(f"recall@{k}" for k in ks) + " | mrr | mean latency (s) | n |"
    )
    sep = "|" + "---|" * (len(ks) + 3)
    lines.append(header)
    lines.append(sep)
    lines.append(
        "| overall | "
        + " | ".join(f"{report.recall_at_k[k]:.3f}" for k in ks)
        + f" | {report.mrr:.3f} | {report.mean_latency_s:.3f} | {report.n} |"
    )
    for category, sub in sorted(report.per_category.items()):
        lines.append(
            f"| {category} | "
            + " | ".join(f"{sub.recall_at_k[k]:.3f}" for k in ks)
            + f" | {sub.mrr:.3f} | {sub.mean_latency_s:.3f} | {sub.n} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["tuning", "heldout"], default=None)
    args = parser.parse_args(argv)

    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.research import ResearchEngine
    from tutor.retrieval.snapshots import SnapshotStore

    questions = load_questions(args.questions)
    if args.split is not None:
        questions = [q for q in questions if q.split == args.split]

    registry = load_registry(args.registry)
    cache_dir = args.out.resolve().parent / ".retrieval_eval_cache"
    snapshot_db = cache_dir / "snapshots.sqlite3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(snapshot_db)
    try:
        engine = ResearchEngine(registry, snapshot_store=store, cache_dir=cache_dir / "cache")
        report = evaluate(engine, questions)
    finally:
        store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
