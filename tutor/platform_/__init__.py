"""Small platform abstraction: windows.py / linux.py implementing the same
interface (docs/plan/offline_tutor_spec_v0.3.md §5).

``get_platform()`` returns the module for the current OS: windows on
``sys.platform == "win32"``, linux otherwise. Both modules expose exactly
three callables: ``limit_child_process``, ``kiosk_command``, and
``outbound_block_supported``.
"""

from __future__ import annotations

import sys
from types import ModuleType


def get_platform() -> ModuleType:
    if sys.platform == "win32":
        from tutor.platform_ import windows

        return windows
    from tutor.platform_ import linux

    return linux
