"""RED tests for the `python -m tutor.app.main --config <path>` CLI.

Only argument parsing and the injectable `serve` seam are tested here: no
live uvicorn server is started (no network, per the task rules for this
piece of work).
"""

from __future__ import annotations

from pathlib import Path

from tutor.app.main import _main, _parse_args

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TOML = REPO_ROOT / "config" / "dev.toml"


def test_parse_args_reads_config_flag():
    args = _parse_args(["--config", str(DEV_TOML)])
    assert Path(args.config) == DEV_TOML


def test_main_calls_serve_with_host_and_port_from_app_config():
    calls = []

    def fake_serve(app, *, host, port):
        calls.append((host, port))

    rc = _main(["--config", str(DEV_TOML)], serve=fake_serve)

    assert rc == 0
    assert len(calls) == 1
    host, port = calls[0]
    assert host == "127.0.0.1"
    assert port == 8420


def test_main_errors_on_missing_config_file(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    calls = []

    def fake_serve(app, *, host, port):
        calls.append((host, port))

    rc = _main(["--config", str(missing)], serve=fake_serve)

    assert rc != 0
    assert calls == []
