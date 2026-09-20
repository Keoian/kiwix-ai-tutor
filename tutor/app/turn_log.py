"""Append-only, privacy-conscious per-turn JSON-lines logger (WP-C4).

Spec sources: docs/plan/offline_tutor_spec_v0.3.md §15 ("Logging fields
from v0.2 §15 are retained. Added: route (action / preretrieve /
preretrieve+followup), calc_calls, eviction_reprefill events,
prompt-cache hit tokens per request, and GPU clock/temperature
samples.").

Privacy (contract decision, see tests/test_status_metrics.py docstring):
``log_turn`` structurally has no parameter for student free text, so it
cannot appear in the log -- only metrics, route, and ids. utf-8, one
compact JSON object per line, pathlib only.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

_ALLOWED_ROUTES = {"action", "preretrieve", "preretrieve+followup"}


class TurnLogger:
    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_turn(
        self,
        *,
        lesson_id: str,
        route: str,
        calc_calls: int,
        research_calls: int,
        eviction_reprefill: bool,
        cached_tokens: int,
        tokens_used: int,
        first_token_ms: int | None,
        tokens_per_second: float | None,
        gpu_clock_mhz: int | None,
        gpu_temp_c: int | None,
    ) -> None:
        if route not in _ALLOWED_ROUTES:
            raise ValueError(f"unknown route: {route!r}")

        record = {
            "lesson_id": lesson_id,
            "route": route,
            "calc_calls": calc_calls,
            "research_calls": research_calls,
            "eviction_reprefill": eviction_reprefill,
            "cached_tokens": cached_tokens,
            "tokens_used": tokens_used,
            "first_token_ms": first_token_ms,
            "tokens_per_second": tokens_per_second,
            "gpu_clock_mhz": gpu_clock_mhz,
            "gpu_temp_c": gpu_temp_c,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
