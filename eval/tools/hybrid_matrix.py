"""M4 gate driver: lexical baseline vs. dense hybrid (+ each ranking flag).

Ours: no donor code. Implements docs/plan/offline_tutor_implementation_plan.md's
M4 gate ("hybrid RRF beats lexical baseline on tuning without hurting
held-out") and Sec 10.8's rule ("a flag is enabled only by a commit that
includes its eval table") as a single reproducible driver + markdown
report, rather than a manual judgment call.

What it runs, on the TUNING split only:
    - ``lexical-v2``       -- no dense sidecar at all (the existing baseline)
    - ``hybrid``           -- dense sidecar, all four ranking flags off
    - ``hybrid+<flag>``    -- dense sidecar, exactly one ranking flag on,
                              once per flag in ``RankingFlags``

The best of those five configurations (recall@5, then MRR, then recall@1;
ties broken toward fewer flags, then by flag name) is the "chosen"
configuration. HELD-OUT is then run exactly once each for ``lexical-v2``
and the chosen configuration -- never for the four flags individually, so
held-out cannot be used, even by accident, to pick among them.

Verdict logic (mechanical, no judgment call):
    - M4 gate: PASS iff the chosen configuration beats the baseline on
      tuning recall@5 OR tuning MRR, AND held-out recall@5 and held-out
      MRR are each >= baseline - 0.0 (i.e. not worse).
    - Per-flag verdict: "enable" iff that flag's own tuning report is >=
      the no-flag hybrid tuning report on recall@5 (ties broken by MRR);
      "keep off" otherwise. This is what Sec 10.8 means by requiring an
      eval table before a flag may be flipped on.

CLI:
    python -m eval.tools.hybrid_matrix --registry config/archives.simplewiki_only.toml \\
        --questions eval/questions/simplewiki_questions.jsonl \\
        --dense-dir runtime/simplewiki_dense --embed-url http://127.0.0.1:8081 \\
        --out docs/hybrid_eval.md
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from eval.run_retrieval_eval import EvalReport, Question, build_engine, evaluate, load_questions
from tutor.retrieval.hybrid.ranking import RankingFlags

_FLAG_NAMES: tuple[str, ...] = tuple(f.name for f in fields(RankingFlags))
_BASELINE_LABEL = "lexical-v2"
_HYBRID_LABEL = "hybrid"


@dataclass(frozen=True)
class ConfigSpec:
    """One row of the matrix: a label, whether dense is enabled, and flags."""

    label: str
    dense: bool
    flags: RankingFlags = RankingFlags()

    @property
    def n_flags(self) -> int:
        return sum(1 for f in fields(self.flags) if getattr(self.flags, f.name))


def hybrid_configs() -> list[ConfigSpec]:
    """The fixed matrix of configurations run on the tuning split."""
    configs = [ConfigSpec(_BASELINE_LABEL, dense=False), ConfigSpec(_HYBRID_LABEL, dense=True)]
    for name in _FLAG_NAMES:
        configs.append(
            ConfigSpec(f"hybrid+{name}", dense=True, flags=RankingFlags(**{name: True}))
        )
    return configs


# ---------------------------------------------------------------------------
# Pure logic: selection, gate verdict, per-flag verdicts, rendering.
# ---------------------------------------------------------------------------


def _better(a: EvalReport, b: EvalReport) -> bool:
    """Whether tuning report ``a`` beats ``b`` by recall@5, then MRR, then recall@1."""
    if a.recall_at_k[5] != b.recall_at_k[5]:
        return a.recall_at_k[5] > b.recall_at_k[5]
    if a.mrr != b.mrr:
        return a.mrr > b.mrr
    return a.recall_at_k[1] > b.recall_at_k[1]


def pick_best_hybrid(
    tuning_reports: dict[str, EvalReport], configs_by_label: dict[str, ConfigSpec]
) -> str:
    """The best of the ``hybrid*`` tuning configurations (never the baseline).

    Ties (equal recall@5, MRR, and recall@1) go to fewer flags, then to
    the earlier flag name in ``RankingFlags`` field order -- both purely
    to make the pick deterministic, never as a quality judgment.
    """
    candidates = [
        label for label in tuning_reports if configs_by_label[label].label != _BASELINE_LABEL
    ]
    if not candidates:
        raise ValueError("no hybrid configurations to choose from")

    def sort_key(label: str) -> tuple:
        report = tuning_reports[label]
        spec = configs_by_label[label]
        return (
            -report.recall_at_k[5],
            -report.mrr,
            -report.recall_at_k[1],
            spec.n_flags,
            label,
        )

    return min(candidates, key=sort_key)


def gate_verdict(
    baseline_tuning: EvalReport,
    chosen_tuning: EvalReport,
    baseline_heldout: EvalReport,
    chosen_heldout: EvalReport,
) -> str:
    """"PASS"/"FAIL" for the M4 gate, computed mechanically (no judgment call).

    PASS iff the chosen configuration beats the baseline on tuning
    recall@5 OR tuning MRR, AND held-out recall@5 and MRR are each not
    worse than the baseline's (>= baseline - 0.0).
    """
    improved_tuning = (
        chosen_tuning.recall_at_k[5] > baseline_tuning.recall_at_k[5]
        or chosen_tuning.mrr > baseline_tuning.mrr
    )
    held_out_not_worse = (
        chosen_heldout.recall_at_k[5] >= baseline_heldout.recall_at_k[5]
        and chosen_heldout.mrr >= baseline_heldout.mrr
    )
    return "PASS" if (improved_tuning and held_out_not_worse) else "FAIL"


def flag_verdicts(
    tuning_reports: dict[str, EvalReport], configs_by_label: dict[str, ConfigSpec]
) -> dict[str, str]:
    """Per-flag "enable"/"keep off", each justified by its own tuning table.

    A flag is "enable" iff its ``hybrid+<flag>`` tuning report is at least
    as good as the no-flag ``hybrid`` tuning report on recall@5 (ties
    broken by MRR); "keep off" otherwise. Matches Sec 10.8: a flag is
    enabled only by a commit carrying the eval table that justifies it.
    """
    baseline = tuning_reports[_HYBRID_LABEL]
    verdicts: dict[str, str] = {}
    for name in _FLAG_NAMES:
        label = f"hybrid+{name}"
        report = tuning_reports[label]
        if report.recall_at_k[5] != baseline.recall_at_k[5]:
            enable = report.recall_at_k[5] > baseline.recall_at_k[5]
        else:
            enable = report.mrr >= baseline.mrr
        verdicts[name] = "enable" if enable else "keep off"
    return verdicts


def _fmt_row(name: str, report: EvalReport) -> str:
    ks = sorted(report.recall_at_k)
    cells = " | ".join(f"{report.recall_at_k[k]:.3f}" for k in ks)
    return (
        f"| {name} | {cells} | {report.mrr:.3f} | {report.mean_latency_s:.3f}"
        f" | {report.p95_latency_s:.3f} | {report.dense_used_rate:.3f} | {report.n} |"
    )


def _fmt_table(reports: dict[str, EvalReport], order: list[str]) -> list[str]:
    ks = sorted(next(iter(reports.values())).recall_at_k)
    header = (
        "| configuration | "
        + " | ".join(f"recall@{k}" for k in ks)
        + " | mrr | mean latency (s) | p95 latency (s) | dense_used rate | n |"
    )
    sep = "|" + "---|" * (len(ks) + 6)
    lines = [header, sep]
    lines.extend(_fmt_row(label, reports[label]) for label in order)
    return lines


def render_matrix_markdown(
    *,
    tuning_reports: dict[str, EvalReport],
    heldout_reports: dict[str, EvalReport],
    chosen_label: str,
    gate: str,
    flag_verdicts_by_name: dict[str, str],
) -> str:
    """Render the whole matrix report as markdown."""
    lines = ["# Hybrid retrieval matrix (M4 gate)", ""]
    lines.append(f"Chosen configuration: **{chosen_label}**")
    lines.append("")
    lines.append(f"## M4 gate verdict: **{gate}**")
    lines.append("")
    lines.append("## Tuning split")
    lines.append("")
    tuning_order = [_BASELINE_LABEL, _HYBRID_LABEL] + [f"hybrid+{n}" for n in _FLAG_NAMES]
    lines.extend(_fmt_table(tuning_reports, tuning_order))
    lines.append("")

    lines.append("## Held-out split (run once, baseline + chosen only)")
    lines.append("")
    heldout_order = [_BASELINE_LABEL, chosen_label]
    lines.extend(_fmt_table(heldout_reports, heldout_order))
    lines.append("")

    lines.append("## Per-category recall@5 (tuning)")
    lines.append("")
    categories = sorted(
        {c for r in tuning_reports.values() for c in r.per_category}
    )
    if categories:
        lines.append("| configuration | " + " | ".join(categories) + " |")
        lines.append("|" + "---|" * (len(categories) + 1))
        for label in tuning_order:
            report = tuning_reports[label]
            cells = " | ".join(
                f"{report.per_category[c].recall_at_k[5]:.3f}" if c in report.per_category else "-"
                for c in categories
            )
            lines.append(f"| {label} | {cells} |")
        lines.append("")

    lines.append("## Per-flag verdicts")
    lines.append("")
    lines.append("| flag | verdict | justified by |")
    lines.append("|---|---|---|")
    for name in _FLAG_NAMES:
        lines.append(
            f"| {name} | {flag_verdicts_by_name[name]} | hybrid+{name} vs. hybrid, above |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration: engine-running part, injectable for tests.
# ---------------------------------------------------------------------------

EngineBuilder = Callable[[ConfigSpec], Any]


def _default_engine_builder(
    registry: Any,
    *,
    snapshot_store: Any,
    cache_dir: Path,
    dense_dir: Path | None,
    embed_url: str | None,
) -> EngineBuilder:
    def build(spec: ConfigSpec) -> Any:
        return build_engine(
            registry,
            snapshot_store=snapshot_store,
            cache_dir=cache_dir / spec.label,
            dense_dir=dense_dir if spec.dense else None,
            embed_url=embed_url if spec.dense else None,
            ranking_flags=spec.flags,
        )

    return build


def run_matrix(
    questions: list[Question],
    *,
    engine_builder: EngineBuilder,
) -> tuple[dict[str, EvalReport], dict[str, EvalReport], str, str, dict[str, str]]:
    """Run the whole matrix over already-loaded ``questions``.

    ``engine_builder(spec) -> engine`` is called once per configuration
    that needs to be evaluated (five times for tuning, twice for
    held-out); it is the only part of this function that ever touches a
    real registry, ZIM archive, or embedding server, so tests can inject
    a fake to exercise the selection/verdict/rendering logic, or a fake
    over the fixture ZIM + a deterministic embedder for a full offline
    smoke test.
    """
    tuning_questions = [q for q in questions if q.split == "tuning"]
    heldout_questions = [q for q in questions if q.split == "heldout"]

    configs = hybrid_configs()
    configs_by_label = {c.label: c for c in configs}

    tuning_reports: dict[str, EvalReport] = {}
    for spec in configs:
        engine = engine_builder(spec)
        tuning_reports[spec.label] = evaluate(engine, tuning_questions, label=spec.label)

    chosen_label = pick_best_hybrid(tuning_reports, configs_by_label)

    heldout_reports: dict[str, EvalReport] = {}
    for label in (_BASELINE_LABEL, chosen_label):
        spec = configs_by_label[label]
        engine = engine_builder(spec)
        heldout_reports[label] = evaluate(engine, heldout_questions, label=label)

    gate = gate_verdict(
        tuning_reports[_BASELINE_LABEL],
        tuning_reports[chosen_label],
        heldout_reports[_BASELINE_LABEL],
        heldout_reports[chosen_label],
    )
    verdicts = flag_verdicts(tuning_reports, configs_by_label)

    return tuning_reports, heldout_reports, chosen_label, gate, verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--embed-url", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from tutor.retrieval.registry import load_registry
    from tutor.retrieval.snapshots import SnapshotStore

    questions = load_questions(args.questions)
    registry = load_registry(args.registry)

    cache_dir = args.out.resolve().parent / ".hybrid_matrix_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(cache_dir / "snapshots.sqlite3")
    try:
        engine_builder = _default_engine_builder(
            registry,
            snapshot_store=store,
            cache_dir=cache_dir / "cache",
            dense_dir=args.dense_dir,
            embed_url=args.embed_url,
        )
        tuning_reports, heldout_reports, chosen_label, gate, verdicts = run_matrix(
            questions, engine_builder=engine_builder
        )
    finally:
        store.close()

    markdown = render_matrix_markdown(
        tuning_reports=tuning_reports,
        heldout_reports=heldout_reports,
        chosen_label=chosen_label,
        gate=gate,
        flag_verdicts_by_name=verdicts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
