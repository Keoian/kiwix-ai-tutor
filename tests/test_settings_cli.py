"""Tests for the `python -m tutor.settings --argv <config.toml>` CLI.

The CLI prints the resolved server binary path on the first line, then one
argv element per line, so shell scripts can build a command line without
duplicating flag-building logic from tutor/settings.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TOML = REPO_ROOT / "config" / "dev.toml"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tutor.settings", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_prints_binary_then_argv_lines():
    result = _run_cli("--argv", str(DEV_TOML))

    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert len(lines) > 1
    assert lines[0].endswith("llama-server.exe")
    assert "-m" in lines[1:]
    assert "--host" in lines[1:]
    assert "8080" in lines[1:]


def test_cli_errors_on_missing_file():
    result = _run_cli("--argv", str(REPO_ROOT / "config" / "does_not_exist.toml"))
    assert result.returncode != 0


def test_cli_errors_on_bad_usage():
    result = _run_cli("--wrong-flag")
    assert result.returncode != 0
