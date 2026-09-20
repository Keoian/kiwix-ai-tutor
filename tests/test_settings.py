"""Tests defining the contract for tutor.settings.load_config / Config.

These tests are RED by design: tutor/settings.py and config/dev.toml do not
exist yet. A later change is expected to add both.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tutor.settings import Config, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TOML = REPO_ROOT / "config" / "dev.toml"
SETTINGS_PY = REPO_ROOT / "tutor" / "settings.py"


def _flag_value(argv: list[str], flag: str) -> str:
    """Return the value immediately following `flag` in argv, or raise."""
    idx = argv.index(flag)
    return argv[idx + 1]


def _write_toml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Real repo config file
# ---------------------------------------------------------------------------


def test_real_dev_toml_loads_expected_values():
    cfg = load_config(DEV_TOML)

    assert isinstance(cfg, Config)
    assert cfg.server.ctx_size == 32768
    assert cfg.server.cache_type_k == "q8_0"
    assert cfg.server.cache_type_v == "q8_0"
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8080
    assert cfg.server.jinja is True
    assert cfg.server.slots is True
    assert cfg.server.parallel == 1
    assert cfg.sampling.temperature == 0.5
    assert cfg.sampling.top_p == 0.9
    assert cfg.sampling.top_k == 20
    assert cfg.runtime.model_path.name == "granite-4.0-h-tiny-Q4_K_M.gguf"


def test_real_dev_toml_base_url():
    cfg = load_config(DEV_TOML)
    assert cfg.server.base_url == "http://127.0.0.1:8080"


def test_real_dev_bonsai_q1_toml_loads_expected_values():
    cfg = load_config(REPO_ROOT / "config" / "dev.bonsai-q1.toml")

    assert isinstance(cfg, Config)
    assert cfg.server.ctx_size == 32768
    assert cfg.server.cache_type_k == "q8_0"
    assert cfg.server.cache_type_v == "q8_0"
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8080
    assert cfg.server.jinja is True
    assert cfg.server.slots is True
    assert cfg.server.parallel == 1
    assert cfg.sampling.temperature == 0.5
    assert cfg.sampling.top_p == 0.9
    assert cfg.sampling.top_k == 20
    assert cfg.runtime.model_path.name == "Bonsai-8B-Q1_0.gguf"


# ---------------------------------------------------------------------------
# 2. Runtime path is read from config, not hard-coded
# ---------------------------------------------------------------------------


def test_runtime_dir_is_read_from_config_relative_model_path(tmp_path):
    made_up_runtime = tmp_path / "made-up-runtime-dir"
    made_up_runtime.mkdir()

    toml_path = tmp_path / "custom.toml"
    _write_toml(
        toml_path,
        f"""
[runtime]
runtime_dir = "{made_up_runtime.as_posix()}"
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

[sampling]
temperature = 0.5
top_p = 0.9
top_k = 20
""",
    )

    cfg = load_config(toml_path)

    assert cfg.runtime.runtime_dir == made_up_runtime
    assert cfg.runtime.model_path == made_up_runtime / "models" / "some-model.gguf"
    assert cfg.runtime.server_binary == made_up_runtime / "bin" / "llama-server"


def test_absolute_model_path_is_preserved(tmp_path):
    made_up_runtime = tmp_path / "another-runtime-dir"
    made_up_runtime.mkdir()
    absolute_model = tmp_path / "elsewhere" / "abs-model.gguf"
    absolute_model.parent.mkdir()

    toml_path = tmp_path / "custom_abs.toml"
    _write_toml(
        toml_path,
        f"""
[runtime]
runtime_dir = "{made_up_runtime.as_posix()}"
model_path = "{absolute_model.as_posix()}"
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

[sampling]
temperature = 0.5
top_p = 0.9
top_k = 20
""",
    )

    cfg = load_config(toml_path)

    assert cfg.runtime.model_path == absolute_model
    assert cfg.runtime.server_binary == made_up_runtime / "bin" / "llama-server"


# ---------------------------------------------------------------------------
# 3. Source-level guard: no hard-coded runtime path or model name in code
# ---------------------------------------------------------------------------


def test_settings_source_has_no_hardcoded_runtime_or_model_name():
    text = SETTINGS_PY.read_text(encoding="utf-8")
    assert "bonsai" not in text.lower()
    assert "C:\\" not in text


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------


def _valid_toml_dict_text(tmp_path: Path) -> str:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    return f"""
[runtime]
runtime_dir = "{runtime_dir.as_posix()}"
model_path = "model.gguf"
server_binary = "llama-server"

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


def test_missing_required_key_raises_config_error_naming_key(tmp_path):
    text = _valid_toml_dict_text(tmp_path).replace('port = 8080\n', '')
    toml_path = _write_toml(tmp_path / "missing_key.toml", text)

    with pytest.raises(ConfigError) as excinfo:
        load_config(toml_path)
    assert "port" in str(excinfo.value)


def test_missing_table_raises_config_error(tmp_path):
    text = _valid_toml_dict_text(tmp_path)
    # Remove the entire [sampling] table.
    text = text.split("[sampling]")[0]
    toml_path = _write_toml(tmp_path / "missing_table.toml", text)

    with pytest.raises(ConfigError) as excinfo:
        load_config(toml_path)
    assert "sampling" in str(excinfo.value).lower()


def test_missing_file_raises_config_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.toml"
    with pytest.raises(ConfigError):
        load_config(missing_path)


def test_wrong_type_port_raises_config_error(tmp_path):
    text = _valid_toml_dict_text(tmp_path).replace(
        "port = 8080", 'port = "8080"'
    )
    toml_path = _write_toml(tmp_path / "wrong_type.toml", text)

    with pytest.raises(ConfigError):
        load_config(toml_path)


def test_non_positive_ctx_size_raises_config_error(tmp_path):
    text = _valid_toml_dict_text(tmp_path).replace("ctx_size = 4096", "ctx_size = 0")
    toml_path = _write_toml(tmp_path / "bad_ctx.toml", text)

    with pytest.raises(ConfigError):
        load_config(toml_path)


@pytest.mark.parametrize("bad_cache_type", ["q5_1", "fp16", ""])
def test_invalid_cache_type_raises_config_error(tmp_path, bad_cache_type):
    text = _valid_toml_dict_text(tmp_path).replace(
        'cache_type_k = "q8_0"', f'cache_type_k = "{bad_cache_type}"'
    )
    toml_path = _write_toml(tmp_path / "bad_cache.toml", text)

    with pytest.raises(ConfigError):
        load_config(toml_path)


# ---------------------------------------------------------------------------
# 5. to_argv
# ---------------------------------------------------------------------------


def test_to_argv_contains_expected_flag_value_pairs():
    cfg = load_config(DEV_TOML)
    argv = cfg.server.to_argv(cfg.runtime, cfg.sampling)

    assert _flag_value(argv, "-m") == str(cfg.runtime.model_path)
    assert _flag_value(argv, "--host") == cfg.server.host
    assert _flag_value(argv, "--port") == "8080"
    assert _flag_value(argv, "-ngl") == "99"
    assert _flag_value(argv, "-fa") == "on"
    assert _flag_value(argv, "-c") == "32768"
    assert _flag_value(argv, "-np") == "1"
    assert _flag_value(argv, "-ctk") == "q8_0"
    assert _flag_value(argv, "-ctv") == "q8_0"
    assert _flag_value(argv, "--temp") == "0.5"
    assert _flag_value(argv, "--top-p") == "0.9"
    assert _flag_value(argv, "--top-k") == "20"
    assert "--jinja" in argv
    assert "--slots" in argv


def test_to_argv_flash_attn_false_gives_off(tmp_path):
    text = _valid_toml_dict_text(tmp_path).replace(
        "flash_attn = true", "flash_attn = false"
    )
    toml_path = _write_toml(tmp_path / "fa_off.toml", text)
    cfg = load_config(toml_path)

    argv = cfg.server.to_argv(cfg.runtime, cfg.sampling)
    assert _flag_value(argv, "-fa") == "off"


def test_to_argv_jinja_false_omits_flag(tmp_path):
    text = _valid_toml_dict_text(tmp_path).replace("jinja = true", "jinja = false")
    toml_path = _write_toml(tmp_path / "jinja_off.toml", text)
    cfg = load_config(toml_path)

    argv = cfg.server.to_argv(cfg.runtime, cfg.sampling)
    assert "--jinja" not in argv


# ---------------------------------------------------------------------------
# 6. Immutability
# ---------------------------------------------------------------------------


def test_config_is_immutable():
    cfg = load_config(DEV_TOML)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.server = cfg.server  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.server.port = 9090  # type: ignore[misc]
