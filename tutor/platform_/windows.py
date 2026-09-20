"""Windows implementation of the tutor.platform_ interface.

No POSIX-only APIs are used here (they are not even importable cleanly
on Windows); memory enforcement for child processes is left to psutil
polling done by the caller (see tutor/tools/calc_tool.py), not to this
module.
"""

from __future__ import annotations


def limit_child_process(proc_or_pid, *, memory_limit_mb: int) -> None:
    """No-op on Windows: there is no cheap way to set a hard RLIMIT_AS
    equivalent before the child starts. Memory is instead enforced by the
    caller polling psutil and killing the child on breach."""
    return None


def kiosk_command(url: str) -> list[str]:
    """Command line to launch the browser in kiosk mode against ``url``."""
    return ["cmd", "/c", "start", "msedge", "--kiosk", url, "--edge-kiosk-type=fullscreen"]


def outbound_block_supported() -> bool:
    """Whether this platform module can install an outbound-block rule
    (a Windows Firewall rule, per WP-D2). Not implemented yet."""
    return False
