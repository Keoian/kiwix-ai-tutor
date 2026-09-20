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
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from tutor.retrieval.hybrid.ranking import RankingFlags

_DEFAULT_KS = (1, 3, 5)

_RANKING_FLAG_NAMES = tuple(f.name for f in fields(RankingFlags))


class FlagsError(Exception):
    """Raised for an unknown ``--flags`` name."""


def parse_flags(flags_csv: str | None) -> RankingFlags:
    """Parse a comma-separated ``--flags`` value into a :class:`RankingFlags`.

    Every name must be a field of ``RankingFlags`` (validated against
    ``dataclasses.fields``, not a hardcoded list, so a new flag added there
    is automatically accepted here). ``None`` or ``""`` means "all off".
    """
    if not flags_csv:
        return RankingFlags()
    names = [n.strip() for n in flags_csv.split(",") if n.strip()]
    unknown = [n for n in names if n not in _RANKING_FLAG_NAMES]
    if unknown:
        raise FlagsError(
            f"unknown ranking flag(s) {unknown!r}; valid flags are {list(_RANKING_FLAG_NAMES)}"
        )
    return RankingFlags(**{n: True for n in names})


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
    dense_used_rate: float = 0.0
    p95_latency_s: float = 0.0
    label: str | None = None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_one(
    engine: Any, question: Question, ks: tuple[int, ...]
) -> tuple[dict[int, float], float, float, bool]:
    started = time.monotonic()
    response = engine.research(question.question, topic_hint=question.context)
    elapsed = time.monotonic() - started
    dense_used = bool(getattr(response, "dense_used", False))

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
        return {k: value for k in ks}, value, elapsed, dense_used

    paths = [p.path for p in response.passages]
    hit_rank: int | None = None
    for i, path in enumerate(paths):
        if path in question.expected_paths:
            hit_rank = i + 1
            break
    recall = {k: (1.0 if hit_rank is not None and hit_rank <= k else 0.0) for k in ks}
    reciprocal_rank = (1.0 / hit_rank) if hit_rank else 0.0
    return recall, reciprocal_rank, elapsed, dense_used


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def evaluate(
    engine: Any,
    questions: list[Question],
    ks: tuple[int, ...] = _DEFAULT_KS,
    *,
    label: str | None = None,
) -> EvalReport:
    """Run ``engine.research`` over every question and aggregate metrics."""
    buckets: dict[str, dict[str, Any]] = {}
    overall_recall: dict[int, list[float]] = {k: [] for k in ks}
    overall_rr: list[float] = []
    overall_latency: list[float] = []
    overall_dense_used: list[bool] = []

    for question in questions:
        recall, rr, elapsed, dense_used = _score_one(engine, question, ks)
        for k in ks:
            overall_recall[k].append(recall[k])
        overall_rr.append(rr)
        overall_latency.append(elapsed)
        overall_dense_used.append(dense_used)

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
            p95_latency_s=_p95(v["latency"]),
        )
        for category, v in buckets.items()
    }

    return EvalReport(
        recall_at_k={k: _mean(overall_recall[k]) for k in ks},
        mrr=_mean(overall_rr),
        mean_latency_s=_mean(overall_latency),
        n=len(questions),
        per_category=per_category,
        dense_used_rate=_mean([1.0 if d else 0.0 for d in overall_dense_used]),
        p95_latency_s=_p95(overall_latency),
        label=label,
    )


def render_markdown(report: EvalReport) -> str:
    """Render ``report`` as a small Markdown document."""
    lines = ["# Retrieval evaluation report", ""]
    if report.label:
        lines.append(f"Configuration: **{report.label}**")
        lines.append("")
    lines.append(f"dense_used rate: {report.dense_used_rate:.3f}")
    lines.append("")
    ks = sorted(report.recall_at_k)
    header = (
        "| metric | "
        + " | ".join(f"recall@{k}" for k in ks)
        + " | mrr | mean latency (s) | p95 latency (s) | n |"
    )
    sep = "|" + "---|" * (len(ks) + 4)
    lines.append(header)
    lines.append(sep)
    lines.append(
        "| overall | "
        + " | ".join(f"{report.recall_at_k[k]:.3f}" for k in ks)
        + f" | {report.mrr:.3f} | {report.mean_latency_s:.3f}"
        f" | {report.p95_latency_s:.3f} | {report.n} |"
    )
    for category, sub in sorted(report.per_category.items()):
        lines.append(
            f"| {category} | "
            + " | ".join(f"{sub.recall_at_k[k]:.3f}" for k in ks)
            + f" | {sub.mrr:.3f} | {sub.mean_latency_s:.3f}"
            f" | {sub.p95_latency_s:.3f} | {sub.n} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_engine(
    registry: Any,
    *,
    snapshot_store: Any,
    cache_dir: Path,
    dense_dir: Path | None = None,
    embed_url: str | None = None,
    ranking_flags: RankingFlags | None = None,
) -> Any:
    """Build a :class:`~tutor.retrieval.research.ResearchEngine` for eval use.

    When ``dense_dir``/``embed_url`` are both given, opens a
    :class:`DenseIndex` for *every* archive in ``registry`` whose sidecar
    at ``dense_dir`` has a fingerprint matching that archive (others are
    silently skipped -- an eval run over a registry with archives the
    sidecar was never built for should still run, lexical-only, for those
    archives) and wires an :class:`EmbeddingClient` against ``embed_url``
    as ``embed_query``.
    """
    from tutor.retrieval.hybrid.dense import DenseIndex, DenseIndexError
    from tutor.retrieval.index.embedding_client import EmbeddingClient
    from tutor.retrieval.research import ResearchEngine

    dense_indexes: dict[str, Any] = {}
    embed_query = None
    if dense_dir is not None and embed_url is not None:
        for entry in registry.archives:
            try:
                digest = registry.fingerprint_digest(entry.id)
                index = DenseIndex.open(dense_dir, archive_digest=digest)
            except DenseIndexError:
                continue
            if index.paths is not None:
                dense_indexes[entry.id] = index
        if dense_indexes:
            client = EmbeddingClient(embed_url)
            embed_query = lambda text: client.embed([text])[0]  # noqa: E731

    return ResearchEngine(
        registry,
        snapshot_store=snapshot_store,
        cache_dir=cache_dir,
        dense_indexes=dense_indexes or None,
        embed_query=embed_query,
        ranking_flags=ranking_flags,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["tuning", "heldout"], default=None)
    parser.add_argument("--dense-dir", type=Path, default=None)
    parser.add_argument("--embed-url", type=str, default=None)
    parser.add_argument(
        "--flags",
        type=str,
        default=None,
        help=f"comma-separated subset of {list(_RANKING_FLAG_NAMES)}",
    )
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args(argv)

    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.snapshots import SnapshotStore

    try:
        ranking_flags = parse_flags(args.flags)
    except FlagsError as exc:
        parser.error(str(exc))
        return 2

    questions = load_questions(args.questions)
    if args.split is not None:
        questions = [q for q in questions if q.split == args.split]

    registry = load_registry(args.registry)
    cache_dir = args.out.resolve().parent / ".retrieval_eval_cache"
    snapshot_db = cache_dir / "snapshots.sqlite3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(snapshot_db)
    try:
        engine = build_engine(
            registry,
            snapshot_store=store,
            cache_dir=cache_dir / "cache",
            dense_dir=args.dense_dir,
            embed_url=args.embed_url,
            ranking_flags=ranking_flags,
        )
        report = evaluate(engine, questions, label=args.label)
    finally:
        store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
