"""Linux implementation of the tutor.platform_ interface.

``resource`` is POSIX-only and must never be imported at module scope
(this module is still imported, unused, on Windows test runs), so every
use of it is deferred to inside a function body.
"""

from __future__ import annotations


def limit_child_process(proc_or_pid, *, memory_limit_mb: int) -> None:
    """Set RLIMIT_AS on the current process before a fork/spawn-exec, or
    best-effort on an already-running child via /proc if supported.

    This is only meaningful when called in the child itself (e.g. as a
    ``preexec_fn``); calling it on an arbitrary already-running pid is a
    best-effort no-op since POSIX has no cross-process RLIMIT_AS setter.
    """
    import os
    import resource

    if isinstance(proc_or_pid, int) and proc_or_pid != os.getpid():
        return None

    limit_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    return None


def kiosk_command(url: str) -> list[str]:
    """Command line to launch the browser in kiosk mode against ``url``."""
    return ["chromium-browser", "--kiosk", url, "--noerrdialogs", "--disable-infobars"]


def outbound_block_supported() -> bool:
    """Whether this platform module can install an outbound-block rule
    (an nftables rule, per WP-D2). Not implemented yet."""
    return False
