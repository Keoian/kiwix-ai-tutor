"""Tests for the optional [server].extra_args passthrough.

Some launch flags (e.g. -lv/--verbosity for capturing buffer-size log
lines during the Granite bake-off) have no dedicated ServerConfig field.
`[server].extra_args` is an optional list of raw strings appended verbatim
to argv by `to_argv`, after all the flags settings.py already knows about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.settings import ConfigError, load_config

BASE_TOML = """
[runtime]
runtime_dir = "{runtime_dir}"
model_path = "models/some-model.gguf"
server_binary = "bin/llama-server"

[server]
host = "127.0.0.1"
port = 8080
ctx_size = 4096
cache_type_k = "q8_0"
cache_type_v = "q8_0"
n_gpu_layers = 99
parallel = 1
flash_attn = true
jinja = true
slots = true
{extra}

[sampling]
temperature = 0.5
top_p = 0.9
top_k = 20
"""


def _write(tmp_path: Path, extra: str) -> Path:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text(
        BASE_TOML.format(runtime_dir=runtime_dir.as_posix(), extra=extra),
        encoding="utf-8",
    )
    return toml_path


def test_extra_args_absent_by_default(tmp_path):
    cfg = load_config(_write(tmp_path, ""))
    argv = cfg.server.to_argv(cfg.runtime, cfg.sampling)
    assert "-lv" not in argv


def test_extra_args_appended_verbatim_after_known_flags(tmp_path):
    cfg = load_config(_write(tmp_path, 'extra_args = ["-lv", "4"]'))
    argv = cfg.server.to_argv(cfg.runtime, cfg.sampling)
    assert argv[-2:] == ["-lv", "4"]


def test_extra_args_must_be_list_of_strings(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "extra_args = [1, 2]"))


def test_extra_args_must_be_a_list(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, 'extra_args = "-lv 4"'))
