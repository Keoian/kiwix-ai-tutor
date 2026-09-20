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
    assert cfg.embedding.model_path.name == "embedding.gguf"
    # relative model_path resolves against [runtime].runtime_dir
    assert cfg.embedding.model_path.parent.name == "models"
    assert cfg.embedding.base_url == "http://127.0.0.1:8081"


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


def test_real_dev_toml_has_embedding_table():
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "config" / "dev.toml")
    assert cfg.embedding is not None
    assert cfg.embedding.dim == 384
    assert cfg.embedding.port == 8081


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


def test_cli_argv_embedding_without_table_errors(tmp_path: Path, capsys):
    from tutor.settings import _main

    config_path = _write(tmp_path, "")
    rc = _main(["--argv-embedding", str(config_path)])
    assert rc == 1
