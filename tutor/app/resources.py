"""Resource discipline: sampling and limit checks for the app process
(WP-C4).

Spec sources: docs/plan/offline_tutor_spec_v0.3.md §10 resource table
(retrieval worker <=512 MiB target / 1 GiB hard limit, embedding model
<=150 MiB resident, app RAM cache 128 MiB, OS headroom >=2 GiB
MemAvailable) and §15 acceptance ("30-minute lesson: no OOM, no queue
growth, no sustained swap, no thermal collapse"). The spec gives no single
app-process RSS ceiling number; ``ResourceLimits.app_rss_mb`` default is a
contract decision (task brief) set comfortably above the documented
128 MiB app cache + 150 MiB embedding budget, well under the 2 GiB
``MemAvailable`` floor.

Uses psutil only, against the *current* process (and its live children,
if any) -- no shelling out to vendor GPU tools here (that stays out of
scope for this work package; GPU clock/temp fields elsewhere are simply
``None`` when no such sampler is wired in).
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceLimits:
    app_rss_mb: float = 600.0
    max_open_files: int = 200
    max_child_processes: int = 4


@dataclass(frozen=True)
class ResourceSample:
    rss_mb: float
    open_files: int
    thread_count: int
    child_process_count: int


class ResourceMonitor:
    """Samples the current process's resource usage via psutil."""

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self._process = psutil.Process()

    def sample(self) -> ResourceSample:
        rss_mb = self._process.memory_info().rss / (1024 * 1024)

        try:
            open_files = len(self._process.open_files())
        except (psutil.Error, OSError):
            open_files = 0

        try:
            thread_count = self._process.num_threads()
        except (psutil.Error, OSError):
            thread_count = 1

        try:
            child_process_count = len(self._process.children(recursive=True))
        except (psutil.Error, OSError):
            child_process_count = 0

        return ResourceSample(
            rss_mb=rss_mb,
            open_files=open_files,
            thread_count=thread_count,
            child_process_count=child_process_count,
        )

    def check(self) -> list[str]:
        sample = self.sample()
        violations: list[str] = []
        if sample.rss_mb > self.limits.app_rss_mb:
            violations.append(
                f"rss_mb {sample.rss_mb:.1f} exceeds limit {self.limits.app_rss_mb:.1f}"
            )
        if sample.open_files > self.limits.max_open_files:
            violations.append(
                f"open_files {sample.open_files} exceeds limit {self.limits.max_open_files}"
            )
        if sample.child_process_count > self.limits.max_child_processes:
            violations.append(
                f"child_process_count {sample.child_process_count} exceeds limit "
                f"{self.limits.max_child_processes}"
            )
        return violations
