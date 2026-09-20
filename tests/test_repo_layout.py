"""Guards on the repository's shape and its house rules.

These are cheap and they fail loudly when a later commit quietly drops one of the
constraints that the plan says must hold from the first commit (§2, §5, §10).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _read(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def test_package_layout_matches_plan_section_2() -> None:
    for package in [
        "tutor",
        "tutor/app",
        "tutor/retrieval",
        "tutor/retrieval/zim",
        "tutor/retrieval/hybrid",
        "tutor/retrieval/index",
        "tutor/tools",
        "tutor/dev",
        "tutor/platform_",
    ]:
        assert (REPO / package / "__init__.py").is_file(), f"{package} is not an importable package"

    for directory in ["config", "scripts", "eval", "tests/fixtures", "tutor/ui"]:
        assert (REPO / directory).is_dir(), f"{directory} is missing"


def test_large_binaries_and_the_runtime_are_never_committable() -> None:
    ignored = _read(".gitignore")
    for pattern in ["runtime/", "*.gguf", "*.zim"]:
        assert pattern in ignored, f"{pattern} must stay in .gitignore (plan §10.2)"


def test_ci_runs_on_both_a_windows_and_a_linux_runner() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "windows-latest" in workflow
    assert "ubuntu-latest" in workflow


def test_portability_lint_rules_are_configured() -> None:
    config = tomllib.loads(_read("pyproject.toml"))
    lint = config["tool"]["ruff"]["lint"]

    # pathlib-only and explicit-encoding rules (plan §5).
    assert "PTH" in lint["select"]
    assert "PLW1514" in lint["select"]

    banned = lint["flake8-tidy-imports"]["banned-api"]
    for module in ["resource", "fcntl", "os.fork", "signal.SIGKILL"]:
        assert module in banned, f"{module} must stay banned from shared code (plan §5)"
