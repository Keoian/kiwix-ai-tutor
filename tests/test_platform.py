"""RED tests for the tutor.platform_ package.

Authoritative source: docs/plan/offline_tutor_spec_v0.3.md §5 (platform_
package: windows.py / linux.py implementing the same small interface) and
docs/plan/offline_tutor_implementation_plan.md WP-C2 / plan §5 (POSIX-only
modules -- `resource`, `fcntl`, `os.fork`, `signal.SIGKILL` -- banned
everywhere except behind tutor/platform_/linux.py).

Contract decisions made here (spec silent):
- `get_platform()` returns the *module* for the current OS: windows on
  win32, linux otherwise (spec's OS matrix is Windows dev / Linux ship;
  no third OS is in scope).
- The shared interface is exactly three callables:
  `limit_child_process(proc_or_pid, *, memory_limit_mb)`,
  `kiosk_command(url) -> list[str]`, `outbound_block_supported() -> bool`,
  per the task brief (WP-D2 mentions a Windows Firewall rule and an
  nftables rule as the two concrete backends `limit_child_process`/
  `outbound_block_supported` must eventually front).
- Both modules must be importable on both OSes (this test file runs on
  whichever OS CI happens to be), so `linux.py` must defer `import
  resource` to inside function bodies rather than at module scope.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTOR_ROOT = REPO_ROOT / "tutor"

BANNED_TOKENS = ("import resource", "import fcntl", "os.fork", "SIGKILL")


def _all_tutor_py_files() -> list[Path]:
    return sorted(TUTOR_ROOT.rglob("*.py"))


def test_platform_package_importable():
    import tutor.platform_  # noqa: F401


def test_windows_module_imports_cleanly():
    import tutor.platform_.windows  # noqa: F401


def test_linux_module_imports_cleanly():
    # Must import on ANY OS, including this test run on Windows: linux.py
    # must not import `resource` at module scope.
    import tutor.platform_.linux  # noqa: F401


def test_get_platform_returns_windows_or_linux_module():
    from tutor.platform_ import get_platform

    module = get_platform()
    assert module.__name__ in ("tutor.platform_.windows", "tutor.platform_.linux")


def test_get_platform_picks_windows_on_win32(monkeypatch):
    from tutor.platform_ import get_platform

    monkeypatch.setattr("sys.platform", "win32")
    module = get_platform()
    assert module.__name__ == "tutor.platform_.windows"


def test_get_platform_picks_linux_when_not_win32(monkeypatch):
    from tutor.platform_ import get_platform

    monkeypatch.setattr("sys.platform", "linux")
    module = get_platform()
    assert module.__name__ == "tutor.platform_.linux"


def test_both_modules_expose_the_same_public_interface():
    import tutor.platform_.linux as linux_mod
    import tutor.platform_.windows as windows_mod

    required = {"limit_child_process", "kiosk_command", "outbound_block_supported"}
    win_public = {n for n in dir(windows_mod) if not n.startswith("_")}
    lin_public = {n for n in dir(linux_mod) if not n.startswith("_")}

    assert required <= win_public
    assert required <= lin_public

    for name in required:
        assert callable(getattr(windows_mod, name))
        assert callable(getattr(linux_mod, name))


def test_kiosk_command_returns_list_of_strings_on_both_modules():
    import tutor.platform_.linux as linux_mod
    import tutor.platform_.windows as windows_mod

    for module in (windows_mod, linux_mod):
        command = module.kiosk_command("http://127.0.0.1:8080/")
        assert isinstance(command, list)
        assert all(isinstance(part, str) for part in command)
        assert len(command) >= 1


def test_outbound_block_supported_returns_bool_on_both_modules():
    import tutor.platform_.linux as linux_mod
    import tutor.platform_.windows as windows_mod

    for module in (windows_mod, linux_mod):
        assert isinstance(module.outbound_block_supported(), bool)


def test_limit_child_process_accepts_pid_and_memory_kwarg_on_current_platform():
    import os

    from tutor.platform_ import get_platform

    module = get_platform()
    # Must not raise merely for being invoked with the current process's own
    # pid and a generous memory limit (a soft-touch smoke test, not asserting
    # actual enforcement here -- enforcement is exercised by test_calc_tool).
    module.limit_child_process(os.getpid(), memory_limit_mb=512)


@pytest.mark.parametrize(
    "path", _all_tutor_py_files(), ids=lambda p: str(p.relative_to(TUTOR_ROOT))
)
def test_no_posix_only_tokens_outside_linux_module(path: Path):
    if path.name == "linux.py" and path.parent.name == "platform_":
        pytest.skip("linux.py is the one file allowed to use POSIX-only APIs")

    source = path.read_text(encoding="utf-8")
    for token in BANNED_TOKENS:
        assert token not in source, f"{path} contains banned POSIX-only token {token!r}"


def test_linux_module_defers_resource_import_into_function_bodies():
    source = (TUTOR_ROOT / "platform_" / "linux.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    module_level_resource_import = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "resource":
                    module_level_resource_import = True
        if isinstance(node, ast.ImportFrom) and node.module == "resource":
            module_level_resource_import = True

    assert not module_level_resource_import, (
        "linux.py must import `resource` lazily inside function bodies so the "
        "module still imports on Windows"
    )
