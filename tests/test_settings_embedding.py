"""Contract for the optional [embedding] config table (WP-B7).

Test-first per WP-B7 rules: this file is written before tutor/settings.py
gains embedding support.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.settings import Config, ConfigError, load_config

_BASE = """
[runtime]
runtime_dir = "{runtime_dir}"
model_path = "model.gguf"
server_binary = "server.bin"

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

[sampling]
temperature = 0.5
top_p = 0.9
top_k = 20
"""


def _write(tmp_path: Path, extra: str) -> Path:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    path = tmp_path / "config.toml"
    text = _BASE.format(runtime_dir=str(runtime_dir).replace("\\", "\\\\")) + extra
    path.write_text(text, encoding="utf-8")
    return path


def test_no_embedding_table_is_none(tmp_path: Path):
    cfg = load_config(_write(tmp_path, ""))
    assert isinstance(cfg, Config)
    assert cfg.embedding is None


def test_embedding_table_parsed(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
"""
    cfg = load_config(_write(tmp_path, extra))
    assert cfg.embedding is not None
    assert cfg.embedding.host == "127.0.0.1"
    assert cfg.embedding.port == 8081
    assert cfg.embedding.dim == 384
    # M4 gate FAILED on held-out (see docs/M4_report.md): dense/hybrid must
    # not be a silent default, so [embedding].enabled defaults to False when
    # omitted from the config.
    assert cfg.embedding.enabled is False
    assert cfg.embedding.model_path.name == "embedding.gguf"
    # relative model_path resolves against [runtime].runtime_dir
    assert cfg.embedding.model_path.parent.name == "models"
    assert cfg.embedding.base_url == "http://127.0.0.1:8081"
    # CPU-by-default fields (measured 2026-09-20: CPU is as fast as GPU here
    # and frees VRAM for the chat model), with defaults when omitted.
    assert cfg.embedding.n_gpu_layers == 0
    assert cfg.embedding.threads == 4
    assert cfg.embedding.ctx_size == 512
    assert cfg.embedding.sidecar_dir.name == "simplewiki_dense"
    assert cfg.embedding.sidecar_dir.parent.name == "runtime"
    assert cfg.embedding.archive_id == "simplewiki"


def test_embedding_table_explicit_cpu_fields(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
n_gpu_layers = 5
threads = 8
ctx_size = 1024
sidecar_dir = "runtime/other_dense"
archive_id = "otherwiki"
"""
    cfg = load_config(_write(tmp_path, extra))
    assert cfg.embedding.n_gpu_layers == 5
    assert cfg.embedding.threads == 8
    assert cfg.embedding.ctx_size == 1024
    assert cfg.embedding.sidecar_dir.name == "other_dense"
    assert cfg.embedding.archive_id == "otherwiki"


def test_embedding_table_rejects_bad_threads(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
threads = "four"
"""
    with pytest.raises(ConfigError, match="threads"):
        load_config(_write(tmp_path, extra))


def test_embedding_table_rejects_bad_n_gpu_layers(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
n_gpu_layers = "zero"
"""
    with pytest.raises(ConfigError, match="n_gpu_layers"):
        load_config(_write(tmp_path, extra))


def test_embedding_table_requires_dim(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
"""
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, extra))


def test_embedding_table_rejects_non_positive_dim(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 0
"""
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, extra))


def test_embedding_table_enabled_explicit_true(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
enabled = true
"""
    cfg = load_config(_write(tmp_path, extra))
    assert cfg.embedding.enabled is True


def test_embedding_table_rejects_bad_enabled(tmp_path: Path):
    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
enabled = "yes"
"""
    with pytest.raises(ConfigError, match="enabled"):
        load_config(_write(tmp_path, extra))


def test_real_dev_toml_has_embedding_table():
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "config" / "dev.toml")
    assert cfg.embedding is not None
    assert cfg.embedding.dim == 384
    assert cfg.embedding.port == 8081
    # M4 gate FAILED on held-out (docs/M4_report.md): dev.toml keeps dense
    # disabled by default until the gate passes.
    assert cfg.embedding.enabled is False


def test_cli_argv_embedding(tmp_path: Path, capsys):
    from tutor.settings import _main

    extra = """
[embedding]
host = "127.0.0.1"
port = 8081
model_path = "models/embedding.gguf"
dim = 384
"""
    config_path = _write(tmp_path, extra)
    rc = _main(["--argv-embedding", str(config_path)])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("server.bin")
    assert "--embedding" in out
    assert "8081" in out
    # CPU-by-default flags (see docs/dense_sidecar.md, "runs on the CPU").
    assert "-ngl" in out
    assert out[out.index("-ngl") + 1] == "0"
    assert "-t" in out
    assert out[out.index("-t") + 1] == "4"
    assert "-c" in out
    assert out[out.index("-c") + 1] == "512"


def test_cli_argv_embedding_without_table_errors(tmp_path: Path, capsys):
    from tutor.settings import _main

    config_path = _write(tmp_path, "")
    rc = _main(["--argv-embedding", str(config_path)])
    assert rc == 1
