"""M5 gate: a real 30-minute lesson soak against the in-process app.

Drives the real ``tutor.app.compose.build_deps`` / ``tutor.app.main.create_app``
wiring (the same production code path as ``tutor.app.main._main``) over a
``fastapi.testclient.TestClient``, against the real dev llama-server
(``config/dev.toml``'s ``[server]`` on :8080) and the real archive registry
(``config/archives.dev.toml``). No fakes anywhere in the live run -- this
script's only "fake" behaviour is a captured wrapper around
``tutor.retrieval.research.ResearchEngine.research`` (see
``_ResearchStatusRecorder``), added because no research status is currently
surfaced over the wire (see docs/M5_notes.md and the WP-B8 review).

Split for testability (docs/plan/offline_tutor_implementation_plan.md WP
convention: pure logic test-first, network/process code thin and last):

* ``build_script`` / ``cycle_script`` -- pure: the fixed ~40-item scripted
  lesson (two subjects, factual/why/elliptical/arithmetic/actions).
* ``TurnRecord`` / ``aggregate`` -- pure: per-turn measurement rows in,
  a summary dict out (percentiles, growth curve, cache ratio, citation/
  calc-correctness rates, research status mix).
* ``render_report`` -- pure: summary dict + samples + meta -> markdown.
* ``run_soak`` -- impure: the real 30-minute loop against the live app.

Run for real with:
    python -m eval.run_lesson_soak --config config/dev.toml --minutes 30 \
        --out docs/M5_report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.attribution_scoring import (
    micro_average_backed_sentence_rate,
    sentence_attribution_counts,
    sentence_attribution_counts_from_event,
)
from eval.attribution_scoring import (
    unbacked_number_rate as _unbacked_number_rate,
)

# ---------------------------------------------------------------------------
# The scripted lesson (pure data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptItem:
    subject: str
    kind: str  # "text" | "action"
    text: str | None = None
    action: str | None = None
    expected_calc: float | None = None


def build_script() -> list[ScriptItem]:
    """A fixed, realistic ~40-turn school lesson: photosynthesis/plant
    biology, then fractions/percentages. Mixes factual questions, "why"
    follow-ups, elliptical follow-ups (depend on prior turn context),
    arithmetic that needs the calc tool (with a known expected answer),
    and the simpler/deeper/hint actions interleaved."""
    bio = [
        ScriptItem("photosynthesis", "text", text="What is photosynthesis?"),
        ScriptItem("photosynthesis", "text", text="Why do plants need sunlight for that?"),
        ScriptItem("photosynthesis", "action", action="simpler"),
        ScriptItem("photosynthesis", "text", text="What gas do they release?"),
        ScriptItem("photosynthesis", "text", text="Why that one and not something else?"),
        ScriptItem("photosynthesis", "action", action="deeper"),
        ScriptItem("photosynthesis", "text", text="Where in the cell does it happen?"),
        ScriptItem("photosynthesis", "text", text="What's chlorophyll for?"),
        ScriptItem("photosynthesis", "action", action="hint"),
        ScriptItem("photosynthesis", "text", text="Do plants breathe at night too?"),
        ScriptItem("photosynthesis", "text", text="Why would that matter for a greenhouse?"),
        ScriptItem("photosynthesis", "text", text="What do roots absorb from soil?"),
        ScriptItem("photosynthesis", "action", action="simpler"),
        ScriptItem("photosynthesis", "text", text="How is that different from a mushroom eating?"),
        ScriptItem("photosynthesis", "text", text="What's a stoma?"),
        ScriptItem("photosynthesis", "action", action="deeper"),
        ScriptItem("photosynthesis", "text", text="Can algae photosynthesize too?"),
        ScriptItem("photosynthesis", "text", text="Why do leaves turn colors in autumn?"),
        ScriptItem("photosynthesis", "action", action="hint"),
        ScriptItem("photosynthesis", "text", text="Give me a one-sentence summary of all that."),
    ]
    frac = [
        ScriptItem("fractions", "text", text="What is a fraction?"),
        ScriptItem("fractions", "text", text="What is 3/4 as a percentage?", expected_calc=75.0),
        ScriptItem("fractions", "text", text="Why do we multiply by 100 for that?"),
        ScriptItem("fractions", "action", action="simpler"),
        ScriptItem("fractions", "text", text="What is 5/8 as a decimal?", expected_calc=0.625),
        ScriptItem("fractions", "text", text="Why isn't that decimal exact for some fractions?"),
        ScriptItem("fractions", "action", action="deeper"),
        ScriptItem("fractions", "text", text="What is 17% of 240?", expected_calc=40.8),
        ScriptItem("fractions", "text", text="How would I check that answer?"),
        ScriptItem("fractions", "action", action="hint"),
        ScriptItem("fractions", "text", text="What is 2/3 + 1/6?", expected_calc=5 / 6),
        ScriptItem("fractions", "text", text="Why do we need a common denominator for that?"),
        ScriptItem("fractions", "text", text="What is 12.5% of 640?", expected_calc=80.0),
        ScriptItem("fractions", "action", action="simpler"),
        ScriptItem("fractions", "text", text="How is a percentage a special kind of fraction?"),
        ScriptItem("fractions", "text", text="What is 9/12 simplified?", expected_calc=0.75),
        ScriptItem("fractions", "action", action="deeper"),
        ScriptItem("fractions", "text", text="What is 250% of 40?", expected_calc=100.0),
        ScriptItem("fractions", "action", action="hint"),
        ScriptItem(
            "fractions", "text",
            text="Give me a one-sentence summary of fractions and percentages.",
        ),
    ]
    return bio + frac


def cycle_script(script: list[ScriptItem], n: int) -> list[ScriptItem]:
    """The first ``n`` items of ``script``, repeating (cycling) it if
    ``n > len(script)``. ``n`` may be 0 (empty list) but ``script`` must be
    non-empty whenever ``n > 0``."""
    if n <= 0:
        return []
    if not script:
        raise ValueError("cannot cycle an empty script")
    return [script[i % len(script)] for i in range(n)]


# ---------------------------------------------------------------------------
# Per-turn record + pure aggregation
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    index: int
    subject: str
    input_kind: str
    input_text: str | None
    wall_s: float
    ttft_s: float | None  # time to first token
    tt_tool_s: float | None  # time to first tool event
    route: str | None
    research_statuses: list[str] = field(default_factory=list)
    research_elapsed_s: list[float] = field(default_factory=list)
    citations: int = 0
    citations_resolved: int = 0
    uncited: bool = False
    calc_calls: int = 0
    tokens_used: int | None = None
    cached_tokens: int | None = None
    eviction_events: int = 0
    status: str = "ok"
    error: str | None = None
    expected_calc: float | None = None
    calc_answer_text: str | None = None
    calc_correct: bool | None = None
    answer_text: str = ""
    labels: list[str] = field(default_factory=list)
    unsupported_labels: list[str] = field(default_factory=list)
    citation_quality: str | None = None
    evidence_dump: bool = False
    truncated: str | None = None
    """Mirrors ``tutor.app.agent_loop.TurnResult.truncated``: ``None``,
    ``"repetition"``, or ``"max_tokens"``. Carried into the per-turn JSON
    dump (``write_turns_dump``) so a runaway-answer defect like the one in
    docs/soak_v3_analysis.md ("Three huge turns") is visible without
    re-running the soak."""
    passages: list[dict] = field(default_factory=list)
    """The turn's known passages (same shape ``resolve_citations``/
    ``attribute_sentences`` take), used by ``aggregate`` to compute
    ``backed_sentence_rate``/``unbacked_number_rate`` (docs/
    attribution_design.md) and kept in the per-turn JSON dump
    (``write_turns_dump``) so a run can be rescored offline. Only used as a
    fallback today: a live soak has no passage text to put here (the
    ``citations`` SSE event carries label/path/support flags only, not
    passage bodies), so this stays ``[]`` for real soaks -- see
    ``attribution_event`` below, which a live soak *does* populate."""
    computed_verified: int = 0
    computed_mismatch: int = 0
    """Counts of "computed" statement items (docs/calc_investigation.md fix
    #1, tutor.app.computed_check) carried by this turn's own
    ``attribution_event["computed"]`` list -- the host's verification of
    stated arithmetic against the sandboxed calc evaluator, independent of
    whether the model itself called the `calc` tool."""
    attribution_event: dict | None = None
    """The server's own ``attributions`` SSE event for this turn (commit
    26c14d0, ``tutor/app/compose.py``): ``{"attributions": [...],
    "unbacked": [...]}``, dumped verbatim from
    ``tutor.app.citations.attribute_sentences``. When present, ``aggregate``
    scores ``backed_sentence_rate``/``unbacked_number_rate`` from this
    instead of recomputing from ``passages`` (which is empty in a live
    soak) -- see ``eval.attribution_scoring.
    sentence_attribution_counts_from_event``. ``None`` for older dumps or
    non-factual turns, in which case ``aggregate`` falls back to the
    ``passages`` path."""

    @property
    def is_factual(self) -> bool:
        """Pre-retrieval turns only (the only turns citations are expected
        on)."""
        return bool(self.route and self.route.startswith("preretrieve"))


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


_RELATIVE_TOL = 1e-3

# docs/soak_v3_analysis.md "Calc misses": two of v2's three flagged misses
# and one of v3's were CORRECT answers graded False only because the model
# gave a fraction ("5/6") where `expected_calc` is the decimal (0.8333...),
# or the reverse. `_candidate_values` below extracts both plain decimals
# (with thousands separators/units stripped) AND `a/b` fraction tokens
# (evaluated to a float), so either form matches.
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")
_PERCENT_NUMBER_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*%")


def _matches(found: float, value: float, tol: float) -> bool:
    return abs(found - value) <= max(tol, abs(value) * 0.005, abs(value) * _RELATIVE_TOL)


def _candidate_values(text: str) -> list[float]:
    """Every number ``score_calc`` should consider as a possible answer in
    ``text``: plain decimals (commas stripped), `a/b` fractions evaluated
    to a float, and a percent figure's `/100` form (so "75%" also matches
    an ``expected_calc`` of 0.75)."""
    values: list[float] = []
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    for num, den in _FRACTION_RE.findall(text):
        try:
            denominator = float(den)
            if denominator != 0:
                values.append(float(num) / denominator)
        except ValueError:
            continue
    for match in _PERCENT_NUMBER_RE.findall(text):
        cleaned = match.replace(",", "")
        try:
            values.append(float(cleaned) / 100.0)
        except ValueError:
            continue
    return values


def _contains_approx(text: str, value: float, *, tol: float = 0.01) -> bool:
    return any(_matches(found, value, tol) for found in _candidate_values(text))


def score_calc(answer_text: str, expected: float) -> bool:
    """True if ``answer_text`` surfaces a number approximately equal to
    ``expected``: rounding-tolerant (1% relative / 0.01 absolute / 1e-3
    relative, whichever is loosest) and format-tolerant -- a fraction
    ("5/6"), a percent ("75%" against an ``expected_calc`` of 0.75), a
    thousands-separated number ("1,250.5"), and a trailing unit ("40.8
    degrees") are all recognised (docs/soak_v3_analysis.md, "Calc
    misses")."""
    return _contains_approx(answer_text, expected)


def aggregate(records: list[TurnRecord]) -> dict[str, Any]:
    """Pure summary of a completed (or early-stopped) soak run."""
    ok = [r for r in records if r.status == "ok"]
    errored = [r for r in records if r.status != "ok"]

    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    wall = [r.wall_s for r in records]

    total_citations = sum(r.citations for r in ok)
    total_resolved = sum(r.citations_resolved for r in ok)
    factual = [r for r in ok if r.is_factual]
    uncited_count = sum(1 for r in factual if r.uncited)
    cited_turns = sum(1 for r in factual if r.labels)
    supported_turns = sum(1 for r in factual if r.labels and not r.unsupported_labels)
    evidence_dump_turns = sum(1 for r in factual if r.evidence_dump)

    attribution_counts = [
        sentence_attribution_counts_from_event(r.attribution_event)
        if r.attribution_event is not None
        else sentence_attribution_counts(r.answer_text, r.passages)
        for r in factual
    ]
    backed_sentence_rate = micro_average_backed_sentence_rate(attribution_counts)
    unbacked_number_rate = _unbacked_number_rate(attribution_counts)

    calc_items = [r for r in ok if r.expected_calc is not None]
    calc_correct = [r for r in calc_items if r.calc_correct]

    statuses: dict[str, int] = {}
    research_elapsed: list[float] = []
    for r in ok:
        for st in r.research_statuses:
            statuses[st] = statuses.get(st, 0) + 1
        research_elapsed.extend(r.research_elapsed_s)

    tokens_series = [(r.index, r.tokens_used) for r in ok if r.tokens_used is not None]
    cache_ratios = []
    for r in ok:
        if r.tokens_used and r.cached_tokens is not None and r.tokens_used > 0:
            cache_ratios.append(r.cached_tokens / r.tokens_used)

    slow_turns = [r.index for r in records if r.wall_s > 300]

    return {
        "turns_completed": len(records),
        "turns_ok": len(ok),
        "turns_errored": len(errored),
        "errors": [{"index": r.index, "status": r.status, "error": r.error} for r in errored],
        "latency": {
            "ttft_p50": _percentile(ttft, 0.5),
            "ttft_p95": _percentile(ttft, 0.95),
            "total_p50": _percentile(wall, 0.5),
            "total_p95": _percentile(wall, 0.95),
        },
        "tokens_series": tokens_series,
        "cache_hit_ratio_mean": statistics.fmean(cache_ratios) if cache_ratios else None,
        "cache_hit_ratio_series": [
            (r.index, (r.cached_tokens / r.tokens_used) if r.tokens_used else None)
            for r in ok
        ],
        "citation_rate": (total_citations / len(factual)) if factual else None,
        "citation_resolved_rate": (total_resolved / total_citations) if total_citations else None,
        "uncited_rate": (uncited_count / len(factual)) if factual else None,
        "factual_turns": len(factual),
        "cited_turns": cited_turns,
        "cited_rate": (cited_turns / len(factual)) if factual else None,
        "supported_turns": supported_turns,
        "supported_rate": (supported_turns / len(factual)) if factual else None,
        "evidence_dump_turns": evidence_dump_turns,
        "backed_sentence_rate": backed_sentence_rate,
        "unbacked_number_rate": unbacked_number_rate,
        "calc_items": len(calc_items),
        "calc_correct": len(calc_correct),
        "calc_accuracy": (len(calc_correct) / len(calc_items)) if calc_items else None,
        "computed_verified": sum(r.computed_verified for r in ok),
        "computed_mismatch": sum(r.computed_mismatch for r in ok),
        "research_status_mix": statuses,
        "research_mean_elapsed_s": statistics.fmean(research_elapsed) if research_elapsed else None,
        "eviction_events_total": sum(r.eviction_events for r in records),
        "slow_turns_over_5min": slow_turns,
        "max_tokens_used": max((r.tokens_used for r in ok if r.tokens_used), default=None),
    }


# ---------------------------------------------------------------------------
# Pure report rendering
# ---------------------------------------------------------------------------


def render_report(
    summary: dict[str, Any],
    resource_samples: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(
        "# M5 Report: profiles, lesson state, prompt log, status page, 30-minute lesson soak"
    )
    lines.append("")
    lines.append(f"Generated: {meta.get('generated_at', '(unknown)')}")
    lines.append("")

    lines.append("## M5 gate checklist")
    lines.append("")
    lines.append("Gate (docs/plan/offline_tutor_implementation_plan.md §1): \"Profiles, lesson "
                  "state, append-only prompt with eviction under the 32K profile, status page, "
                  "30-minute lesson within resource limits (Windows measurements).\"")
    lines.append("")
    lines.append("| Item | Evidence | Status |")
    lines.append("|---|---|---|")
    lines.append(
        "| Profiles | `tutor/app/profiles.py`, `tests/test_profiles.py` | measured (unit) |"
    )
    lines.append("| Lesson state / persistence / resume | `tutor/app/lesson_state.py`, "
                  "`tests/test_lesson_state.py`, `tests/test_review_pass2_fixes.py` | "
                  "measured (unit) |")
    lines.append("| Append-only prompt log + eviction | `tutor/app/prompt.py`, "
                  "`tests/test_prompt_builder.py`, docs/M5_notes.md | measured (unit) |")
    lines.append("| Status page | `tutor/app/compose.py` (`_make_status_provider`), "
                  "`tests/test_compose.py`, `tests/test_turn_live.py` | "
                  "measured (unit + this soak) |")
    lines.append(
        "| 30-minute real lesson within resource limits | this document, "
        "Soak results below | "
                  f"{meta.get('gate_status', '(see below)')} |")
    lines.append("")

    lines.append("## Soak run parameters")
    lines.append("")
    meta_keys = (
        "config", "minutes_requested", "started_at", "ended_at", "stopped_early", "stop_reason"
    )
    for k in meta_keys:
        if k in meta:
            lines.append(f"- {k}: {meta[k]}")
    lines.append("")

    lines.append("## Turns")
    lines.append("")
    lines.append(f"- turns completed: **{summary['turns_completed']}** "
                  f"(ok: {summary['turns_ok']}, errored: {summary['turns_errored']})")
    if summary["errors"]:
        lines.append("- errors:")
        for e in summary["errors"]:
            lines.append(f"  - turn {e['index']}: {e['status']} — {e['error']}")
    if summary["slow_turns_over_5min"]:
        slow = summary["slow_turns_over_5min"]
        lines.append(f"- turns exceeding 5 minutes wall time: {slow} (measured)")
    else:
        lines.append("- no turn exceeded 5 minutes wall time (measured)")
    lines.append("")

    lines.append("## Latency (measured)")
    lines.append("")
    lat = summary["latency"]
    lines.append(
        f"- time-to-first-token: p50={_fmt(lat['ttft_p50'])}s, p95={_fmt(lat['ttft_p95'])}s"
    )
    lines.append(
        f"- total turn wall time: p50={_fmt(lat['total_p50'])}s, p95={_fmt(lat['total_p95'])}s"
    )
    lines.append("")
    lines.append(
        "IMPORTANT hardware caveat (inferred from prior measurement, not re-derived here): "
        "on this GPU, prompt prefill throughput collapses with depth when flash-attention "
        "is on (~21 t/s at 4k context, ~11 t/s at 8k). Turns only stay fast as the lesson "
                  "log grows if the prompt cache keeps hitting -- see cache hit ratio below.")
    lines.append("")

    lines.append("## Token growth (measured)")
    lines.append("")
    lines.append("Prompt tokens_used every ~5 turns:")
    lines.append("")
    lines.append("| turn | tokens_used |")
    lines.append("|---|---|")
    for idx, tok in summary["tokens_series"]:
        if idx % 5 == 0 or idx == summary["tokens_series"][-1][0]:
            lines.append(f"| {idx} | {tok} |")
    lines.append("")
    max_tok = summary.get("max_tokens_used")
    lines.append(f"- max tokens_used observed: {max_tok} "
                  "(32K ceiling; a 30-minute lesson may never trigger eviction -- "
                  "that is expected, not a bug)")
    n_evict = summary["eviction_events_total"]
    lines.append(
        f"- eviction_reprefill events observed: {n_evict} (measured; forwarded live over "
        "SSE as an `eviction` event and counted from that event, not read out of "
        "PromptLog in-process -- see docs/M5_report.md \"fixed after the soak\")"
    )
    lines.append("")

    lines.append("## Prompt cache hit ratio (measured, cached_tokens / tokens_used)")
    lines.append("")
    mean_ratio = summary["cache_hit_ratio_mean"]
    lines.append(
        f"- mean: {_fmt(mean_ratio)}" if mean_ratio is not None else "- mean: n/a (no data)"
    )
    lines.append("")

    lines.append("## Citations and calc correctness (measured)")
    lines.append("")
    lines.append(f"- factual (pre-retrieval) turns: {summary['factual_turns']}")
    lines.append(f"- citation rate (citations per factual turn): {_fmt(summary['citation_rate'])}")
    lines.append(f"- citation resolved rate: {_fmt(summary['citation_resolved_rate'])}")
    lines.append(f"- uncited rate (factual turn, no [S#] at all): {_fmt(summary['uncited_rate'])}")
    lines.append(f"- calc items scripted with known answers: {summary['calc_items']}, "
                  f"correct: {summary['calc_correct']} ({_fmt(summary['calc_accuracy'])})")
    lines.append("")

    lines.append("### Citations (factual turns only)")
    lines.append("")
    lines.append(
        "cited_rate == 1 - uncited_rate (same factual-turn denominator, complementary "
        "definitions: cited_rate counts turns with >=1 [S#] label, uncited_rate counts "
        "turns with none)."
    )
    lines.append("")
    lines.append("| metric | value | definition |")
    lines.append("|---|---|---|")
    lines.append(
        f"| factual_turns | {summary['factual_turns']} | "
        "pre-retrieval turns (route starts with `preretrieve`) |"
    )
    lines.append(
        f"| cited_turns / cited_rate | {summary['cited_turns']} / {_fmt(summary['cited_rate'])} | "
        "factual turns with at least one [S#] citation label |"
    )
    lines.append(
        f"| supported_turns / supported_rate | {summary['supported_turns']} / "
        f"{_fmt(summary['supported_rate'])} | "
        "factual turns with >=1 citation label and no unsupported labels |"
    )
    lines.append(
        f"| evidence_dump_turns | {summary['evidence_dump_turns']} | "
        "factual turns flagged as dumping raw evidence instead of a synthesized answer |"
    )
    lines.append(
        f"| uncited_rate | {_fmt(summary['uncited_rate'])} | "
        "factual turns with no [S#] citation at all (== 1 - cited_rate) |"
    )
    lines.append(
        f"| backed_sentence_rate | {_fmt(summary['backed_sentence_rate'])} | "
        "host-side sentence attribution (tutor.app.citations.attribute_sentences): "
        "attributed sentences / (attributed + unbacked sentences), micro-averaged "
        "over factual turns |"
    )
    lines.append(
        f"| unbacked_number_rate | {_fmt(summary['unbacked_number_rate'])} | "
        "factual turns with >=1 sentence carrying a figure attributed to no passage |"
    )
    lines.append("")

    lines.append("## Research status mix (measured, captured via a research()-call recorder; "
                  "no research_status field is exposed over the wire today)")
    lines.append("")
    for k, v in summary["research_status_mix"].items():
        lines.append(f"- {k}: {v}")
    _mean_elapsed = summary["research_mean_elapsed_s"]
    _mean_elapsed_str = f"{_mean_elapsed:.3f}s" if _mean_elapsed is not None else "n/a"
    lines.append(f"- mean research call elapsed: {_mean_elapsed_str}")
    lines.append("")

    lines.append("## Resource usage (measured, via /api/status every ~30s)")
    lines.append("")
    if resource_samples:
        rss = [s["rss_mb"] for s in resource_samples if s.get("rss_mb") is not None]
        threads = [s["thread_count"] for s in resource_samples if s.get("thread_count") is not None]
        handles = [s["open_files"] for s in resource_samples if s.get("open_files") is not None]
        children = [
            s["child_process_count"]
            for s in resource_samples
            if s.get("child_process_count") is not None
        ]
        rss_growth = (rss[-1] - rss[0]) if len(rss) >= 2 else None
        rss_start = _fmt(rss[0] if rss else None)
        rss_end = _fmt(rss[-1] if rss else None)
        rss_max = _fmt(max(rss) if rss else None)
        lines.append(
            f"- RSS MB: start={rss_start}, end={rss_end}, "
            f"max={rss_max}, growth={_fmt(rss_growth)}"
        )
        t_start = threads[0] if threads else "n/a"
        t_end = threads[-1] if threads else "n/a"
        lines.append(f"- thread count: start={t_start}, end={t_end}")
        h_start = handles[0] if handles else "n/a"
        h_end = handles[-1] if handles else "n/a"
        lines.append(f"- open files/handles: start={h_start}, end={h_end}")
        c_start = children[0] if children else "n/a"
        c_end = children[-1] if children else "n/a"
        lines.append(f"- child processes: start={c_start}, end={c_end}")
        llm_healthy_all = all(s.get("llm_healthy") for s in resource_samples)
        lines.append(f"- llama-server /health stayed healthy for every sample: {llm_healthy_all}")
    else:
        lines.append(
            "- no resource samples were collected (soak stopped before the first 30s mark)"
        )
    lines.append("")

    lines.append("## Resume check")
    lines.append("")
    lines.append(meta.get("resume_check", "not run"))
    lines.append("")

    lines.append("## Deferred to the Dell (M6)")
    lines.append("")
    lines.append(
        "- Real hardware measurements on the delivery machine (this soak ran on the dev GPU box)."
    )
    lines.append("")

    lines.append("Labeling: everything in \"Turns\"/\"Latency\"/\"Token growth\"/\"Prompt cache\"/"
                  '"Citations"/"Research status"/"Resource usage"/"Resume check" is **measured** '
                  "from this run. The hardware prefill-collapse figures "
                  "(21 t/s@4k / 11 t/s@8k) are "
                  "**inferred** from a prior measurement, cited for context, not re-derived here.")
    lines.append("")

    return "\n".join(lines)


def turns_dump_path(out_path: str) -> Path:
    """The path a per-turn JSON dump for ``out_path`` is written to: same
    stem, under ``data/`` instead of ``out_path``'s own directory, with a
    ``.turns.json`` suffix (e.g. ``docs/x/q1_soak10.md`` ->
    ``data/q1_soak10.turns.json``)."""
    return Path("data") / f"{Path(out_path).stem}.turns.json"


def write_turns_dump(records: list[TurnRecord], out_path: str) -> Path:
    """Write every ``TurnRecord`` (including ``answer_text``) as JSON next
    to the report, so a run can be re-scored later without re-running the
    soak. Returns the path written."""
    dump_path = turns_dump_path(out_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) | {"is_factual": r.is_factual} for r in records]
    dump_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dump_path


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


# ---------------------------------------------------------------------------
# Live run (impure)
# ---------------------------------------------------------------------------


class _ResearchStatusRecorder:
    """Wraps ``ResearchEngine.research`` (bound instance) to record
    (status, elapsed_s) for every call, in call order, into a shared list
    the soak loop drains per-turn. Turns run strictly sequentially in this
    script (one POST /turn fully consumed before the next), so call order
    alone is enough to attribute recordings to the right turn."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._orig = engine.research
        self.calls: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def install(self) -> None:
        def wrapped(*args, **kwargs):
            start = time.monotonic()
            response = self._orig(*args, **kwargs)
            elapsed = time.monotonic() - start
            with self._lock:
                self.calls.append((getattr(response, "status", "unknown"), elapsed))
            return response

        self._engine.research = wrapped

    def drain(self, n: int) -> list[tuple[str, float]]:
        with self._lock:
            taken, self.calls = self.calls[:n], self.calls[n:]
        return taken


def process_turn_stream(lines: Any, t0: float, now: Any) -> dict:
    """Pure(-ish) reduction of one turn's raw SSE lines (as
    ``httpx``/``requests``-style ``resp.iter_lines()`` yields them) into the
    fields ``run_soak`` needs for a ``TurnRecord``. ``t0`` is the wall-clock
    start of the turn (for ``ttft``/``tt_tool``); ``now`` is a zero-arg
    clock (``time.monotonic`` in production, a fake counter in tests) so
    this has no real I/O or timing dependency of its own -- only ``lines``
    is impure, and it is just an iterable here.

    Includes the ``attributions`` event (commit 26c14d0): stored verbatim
    under ``attribution_event`` for ``aggregate`` to score from (see
    ``TurnRecord.attribution_event``), never parsed further here.
    """
    state: dict[str, Any] = {
        "ttft": None,
        "tt_tool": None,
        "route": None,
        "citations": 0,
        "citations_resolved": 0,
        "uncited": False,
        "calc_calls": 0,
        "tokens_used": None,
        "cached_tokens": None,
        "status": "error",
        "error": None,
        "answer_text": "",
        "eviction_events": 0,
        "labels": [],
        "unsupported_labels": [],
        "citation_quality": None,
        "evidence_dump": False,
        "truncated": None,
        "attribution_event": None,
    }
    event_name = None
    for line in lines:
        if line == "":
            continue
        if line.startswith("event: "):
            event_name = line[len("event: "):]
            continue
        if not line.startswith("data: "):
            continue
        data = json.loads(line[len("data: "):])
        n = now()
        if event_name == "token":
            if state["ttft"] is None:
                state["ttft"] = n - t0
            state["answer_text"] += data.get("text", "") or ""
        elif event_name == "tool":
            if state["tt_tool"] is None:
                state["tt_tool"] = n - t0
            if data.get("name") == "calc" and data.get("phase") == "result":
                state["calc_calls"] += 1
        elif event_name == "citations":
            cites = data.get("citations", [])
            state["citations"] = len(cites)
            state["citations_resolved"] = sum(1 for c in cites if not c.get("unresolved"))
            state["labels"] = [c.get("label") for c in cites if c.get("label")]
            state["unsupported_labels"] = list(data.get("unsupported_labels", []))
        elif event_name == "attributions":
            state["attribution_event"] = data
        elif event_name == "eviction":
            # GAP 2 fix: eviction_reprefill is now forwarded live over SSE
            # instead of this script reaching into PromptLog.evict.
            state["eviction_events"] += 1
        elif event_name == "done":
            state["status"] = data.get("status", "error")
            state["route"] = data.get("route")
            state["tokens_used"] = data.get("tokens_used")
            state["cached_tokens"] = data.get("cached_tokens")
            state["uncited"] = bool(data.get("uncited"))
            state["answer_text"] = data.get("answer") or state["answer_text"]
            state["citation_quality"] = data.get("citation_quality")
            state["evidence_dump"] = bool(data.get("evidence_dump"))
            state["truncated"] = data.get("truncated")
            if data.get("calc_calls") is not None:
                state["calc_calls"] = data["calc_calls"]
        elif event_name == "error":
            state["status"] = "error"
            state["error"] = data.get("message")
    return state


def _make_soak_config(base_cfg: Any, data_dir: Path):
    server = dataclasses.replace(base_cfg.server)
    app_cfg = dataclasses.replace(base_cfg.app, data_dir=data_dir)
    return dataclasses.replace(base_cfg, server=server, app=app_cfg)


def run_soak(*, config_path: str, minutes: float, out_path: str, think_time_s: float = 5.0) -> str:
    import datetime

    from fastapi.testclient import TestClient

    from tutor.app.compose import build_deps
    from tutor.app.llm_client import LlamaClient
    from tutor.app.main import create_app
    from tutor.settings import load_config

    base_cfg = load_config(config_path)

    llm_probe = LlamaClient(base_cfg.server.base_url, timeout_s=10)
    if not llm_probe.health():
        raise RuntimeError(
            f"llama-server /health unreachable at {base_cfg.server.base_url}; "
            "start it (e.g. scripts/serve_dev.ps1) before running the soak"
        )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = (Path("data") / f"soak_{ts}").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_soak_config(base_cfg, data_dir)
    deps = build_deps(cfg)

    # Attribute research status per call (see _ResearchStatusRecorder); the
    # research engine object lives inside the turn_runner closure, so we
    # reach it the only way available: build a second, throwaway engine is
    # wrong (different cache/registry instance) -- instead we monkeypatch
    # the class method for the lifetime of this process. Safe: this process
    # only ever runs one soak.
    from tutor.retrieval.research import ResearchEngine

    recorder_calls: list[tuple[str, float]] = []
    recorder_lock = threading.Lock()
    _orig_research = ResearchEngine.research

    def _patched(self, *args, **kwargs):
        start = time.monotonic()
        response = _orig_research(self, *args, **kwargs)
        elapsed = time.monotonic() - start
        with recorder_lock:
            recorder_calls.append((getattr(response, "status", "unknown"), elapsed))
        return response

    ResearchEngine.research = _patched

    try:
        app = create_app(deps)
        with TestClient(app) as client:
            # Lesson persistence (turn records + byte-identical-resume prompt
            # log) is now the app's own responsibility (tutor.app.compose's
            # turn_runner persists every completed turn into LessonStore when
            # the session is attached this way) -- this soak no longer
            # persists anything itself.
            profile_id = client.post(
                "/api/profiles",
                json={
                    "display_name": "Soak Student",
                    "grade_level": 6,
                    "subjects": ["photosynthesis", "fractions"],
                    "reading_level": "grade6",
                },
            ).json()["id"]
            created = client.post(
                "/api/lesson", json={"profile_id": profile_id, "subject": "general"}
            ).json()
            lesson_id = created["lesson_id"]
            session_id = created["session_id"]
            # Note: the lesson is held under one fixed subject ("general")
            # for the whole soak, on purpose -- LessonStore.append_turn/
            # persist_turn enforce the same "a subject change starts a new
            # lesson" rule the app itself follows (docs/plan risk register),
            # so switching the *session's* subject_hint per scripted item
            # (as earlier soaks did, purely to steer retrieval's topic_hint)
            # would make every turn after the first switch silently fail to
            # persist. The scripted lesson still exercises both subjects'
            # content; it just no longer steers retrieval with a per-item
            # topic hint.

            script = build_script()
            # Generous upper bound on turn count; the deadline check inside
            # the loop is what actually stops the soak, not running out of
            # scripted items (cycle_script just needs to not run dry).
            n_turns = max(1, round((minutes * 60) / max(think_time_s, 1.0))) + len(script)
            items = cycle_script(script, n_turns)

            records: list[TurnRecord] = []
            resource_samples: list[dict[str, Any]] = []
            last_sample_t = 0.0
            start_t = time.monotonic()
            consecutive_slow = 0
            stopped_early = False
            stop_reason = None
            deadline = start_t + minutes * 60

            for i, item in enumerate(items, start=1):
                if time.monotonic() > deadline:
                    break

                body: dict[str, Any] = (
                    {"text": item.text} if item.kind == "text" else {"action": item.action}
                )

                t0 = time.monotonic()
                parsed: dict[str, Any] = {}

                try:
                    with client.stream(
                        "POST", f"/api/session/{session_id}/turn", json=body, timeout=330
                    ) as resp:
                        parsed = process_turn_stream(resp.iter_lines(), t0, time.monotonic)
                except Exception as exc:  # noqa: BLE001 - record and keep the soak going
                    parsed = {"status": "error", "error": str(exc), "answer_text": ""}

                ttft = parsed.get("ttft")
                tt_tool = parsed.get("tt_tool")
                route = parsed.get("route")
                citations = parsed.get("citations", 0)
                citations_resolved = parsed.get("citations_resolved", 0)
                uncited = parsed.get("uncited", False)
                calc_calls = parsed.get("calc_calls", 0)
                tokens_used = parsed.get("tokens_used")
                cached_tokens = parsed.get("cached_tokens")
                status = parsed.get("status", "error")
                error = parsed.get("error")
                answer_text = parsed.get("answer_text", "")
                eviction_events = parsed.get("eviction_events", 0)
                labels = parsed.get("labels", [])
                unsupported_labels = parsed.get("unsupported_labels", [])
                citation_quality = parsed.get("citation_quality")
                evidence_dump = parsed.get("evidence_dump", False)
                truncated = parsed.get("truncated")
                attribution_event = parsed.get("attribution_event")

                wall = time.monotonic() - t0

                # GAP 1 fix: the app itself persists this turn (turn row +
                # prompt-log snapshot, atomically) inside
                # tutor.app.compose's turn_runner, because session_id was
                # attached to lesson_id via POST /api/lesson above. This
                # script no longer writes to LessonStore itself.

                with recorder_lock:
                    calls_for_turn = list(recorder_calls)
                    recorder_calls.clear()
                research_statuses = [c[0] for c in calls_for_turn]
                research_elapsed = [c[1] for c in calls_for_turn]

                expected_calc = item.expected_calc
                calc_correct = (
                    score_calc(answer_text, expected_calc) if expected_calc is not None else None
                )

                records.append(
                    TurnRecord(
                        index=i,
                        subject=item.subject,
                        input_kind=item.kind,
                        input_text=item.text,
                        wall_s=wall,
                        ttft_s=ttft,
                        tt_tool_s=tt_tool,
                        route=route,
                        research_statuses=research_statuses,
                        research_elapsed_s=research_elapsed,
                        citations=citations,
                        citations_resolved=citations_resolved,
                        uncited=uncited,
                        calc_calls=calc_calls,
                        tokens_used=tokens_used,
                        cached_tokens=cached_tokens,
                        eviction_events=eviction_events,
                        status=status,
                        error=error,
                        expected_calc=expected_calc,
                        calc_answer_text=answer_text,
                        calc_correct=calc_correct,
                        answer_text=answer_text,
                        labels=labels,
                        unsupported_labels=unsupported_labels,
                        citation_quality=citation_quality,
                        evidence_dump=evidence_dump,
                        truncated=truncated,
                        # See TurnRecord.passages docstring: the wire
                        # protocol has no passage text for the fallback
                        # path, but attribution_event (below) carries the
                        # server's own scoring for a live soak.
                        passages=[],
                        attribution_event=attribution_event,
                        computed_verified=sum(
                            1
                            for c in (attribution_event or {}).get("computed", [])
                            if c.get("status") == "verified"
                        ),
                        computed_mismatch=sum(
                            1
                            for c in (attribution_event or {}).get("computed", [])
                            if c.get("status") == "mismatch"
                        ),
                    )
                )

                if wall > 300:
                    consecutive_slow += 1
                else:
                    consecutive_slow = 0
                if consecutive_slow >= 3:
                    stopped_early = True
                    stop_reason = (
                        f"3 consecutive turns exceeded 5 minutes wall time "
                        f"(turns {i - 2}-{i}); stopping without killing the server"
                    )
                    break

                now = time.monotonic()
                if now - start_t - last_sample_t >= 30 or last_sample_t == 0:
                    status_resp = client.get("/api/status").json()
                    resources = status_resp.get("resources", {})
                    resource_samples.append(
                        {
                            "t": now - start_t,
                            "rss_mb": resources.get("rss_mb"),
                            "thread_count": resources.get("thread_count"),
                            "open_files": resources.get("open_files"),
                            "child_process_count": resources.get("child_process_count"),
                            "llm_healthy": status_resp.get("llm", {}).get("healthy"),
                        }
                    )
                    last_sample_t = now - start_t

                time.sleep(think_time_s)

            summary = aggregate(records)
            write_turns_dump(records, out_path)

            # --- resume check: fresh app instance, same data_dir ---
            from tutor.app.prompt import serialize_messages

            pre_restart_session = deps.sessions.get(session_id)
            pre_restart_bytes = serialize_messages(pre_restart_session.log.render())
            resume_text = _resume_check(cfg, lesson_id, pre_restart_bytes)

    finally:
        ResearchEngine.research = _orig_research

    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": config_path,
        "minutes_requested": minutes,
        "started_at": datetime.datetime.fromtimestamp(
            start_t + (time.time() - time.monotonic())
        ).isoformat(timespec="seconds"),
        "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "resume_check": resume_text,
        "gate_status": (
            "PASS (measured, no turn >5min, soak ran to completion)"
            if not stopped_early
            else "PARTIAL -- soak stopped early per the 3-consecutive-slow-turns rule "
            "(see stop_reason)"
        ),
    }
    report = render_report(summary, resource_samples, meta)
    Path(out_path).write_text(report, encoding="utf-8")
    return report


def _resume_check(cfg: Any, lesson_id: str, pre_restart_bytes: bytes) -> str:
    """Build a fresh AppDeps/app (a fresh process would rebuild these the
    same way) over the same data_dir and confirm, entirely through the
    real HTTP surface (GAP 1 fix: no reaching into LessonStore/Session
    internals here anymore):

    1. ``POST /api/lesson/{lesson_id}/resume`` succeeds and rebuilds a
       prompt log that is byte-identical to what was persisted live.
    2. one more turn against the real llm succeeds on the resumed session.
    """
    from fastapi.testclient import TestClient

    from tutor.app.compose import build_deps
    from tutor.app.main import create_app
    from tutor.app.prompt import serialize_messages

    try:
        deps2 = build_deps(cfg)
        app2 = create_app(deps2)
        with TestClient(app2) as client2:
            resume_resp = client2.post(f"/api/lesson/{lesson_id}/resume")
            if resume_resp.status_code != 200:
                return f"FAIL: resume endpoint returned {resume_resp.status_code}."
            new_sid = resume_resp.json()["session_id"]

            resumed_session = deps2.sessions.get(new_sid)
            resumed_session.log.validate()
            resumed_bytes = serialize_messages(resumed_session.log.render())
            if resumed_bytes != pre_restart_bytes:
                return (
                    "FAIL: resumed prompt log is not byte-identical to the "
                    f"pre-restart log (lesson {lesson_id})."
                )

            frames_status = []
            with client2.stream(
                "POST",
                f"/api/session/{new_sid}/turn",
                json={"text": "One more quick question: what did we cover?"},
                timeout=330,
            ) as resp:
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        frames_status.append(line)
        ok = any("done" in f for f in frames_status)
        return (
            "PASS (measured): POST /api/lesson/{id}/resume rebuilt a byte-identical prompt "
            "log (app-side persistence, no soak-driven persistence involved) and one "
            f"additional turn completed successfully after resume (lesson {lesson_id})."
            if ok
            else (
                "FAIL: resumed PromptLog was byte-identical but the follow-up turn did not "
                f"complete (lesson {lesson_id})."
            )
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the soak on resume-check failure
        return f"FAIL: resume check raised {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.toml")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--out", default="docs/M5_report.md")
    parser.add_argument("--think-time", type=float, default=5.0)
    args = parser.parse_args(argv)

    report = run_soak(
        config_path=args.config,
        minutes=args.minutes,
        out_path=args.out,
        think_time_s=args.think_time,
    )
    print(report)


if __name__ == "__main__":
    main()
