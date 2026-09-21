"""RED tests for the optional [app] table on tutor.settings.Config.

New config knobs used by tutor.app.compose (WP-C4): host/port for uvicorn,
plus data_dir and registry_path resolved relative to the repo root (the
config file's parent's parent) when given as relative paths.
"""

from __future__ import annotations

from pathlib import Path

from tutor.settings import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TOML = REPO_ROOT / "config" / "dev.toml"


def _write_toml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


_BASE = """
[runtime]
runtime_dir = 'C:\\\\git\\\\bonsai'
model_path = "models/Bonsai-8B-Q1_0.gguf"
server_binary = "bin/llama-server.exe"

[server]
host = "127.0.0.1"
port = 8080
ctx_size = 32768
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


def test_real_dev_toml_has_app_table_with_defaults_or_values():
    cfg = load_config(DEV_TOML)
    assert cfg.app.host
    assert isinstance(cfg.app.port, int)
    assert cfg.app.data_dir.is_absolute()
    assert cfg.app.registry_path.is_absolute()


def test_app_table_defaults_when_missing(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = _write_toml(config_dir / "dev.toml", _BASE)

    cfg = load_config(config_path)

    assert cfg.app.host == "127.0.0.1"
    assert cfg.app.port == 8420
    # data_dir defaults to "data" resolved relative to repo root, i.e. the
    # config file's parent's parent.
    assert cfg.app.data_dir == (tmp_path / "data").resolve()
    assert cfg.app.registry_path == (tmp_path / "config" / "archives.dev.toml").resolve()
    # Answer/output token cap defaults to 2000, matching
    # tutor.app.prompt.Budget.generation (see settings.py docstring).
    assert cfg.app.answer_max_tokens == 2000
    # Adopted 2026-09-20, then reverted the same day (docs/citation_experiment.md,
    # "Seed exchange A/B" / "In-lesson check and reversal"): in-lesson soaks
    # contradicted the single-turn A/B, so the default is "current" again.
    # "seed_exchange_s0" stays fully selectable and tested.
    assert cfg.app.prompt_variant == "current"
    assert cfg.app.rewrite_on_weak_evidence is True
    assert cfg.app.rewrite_on_followup is True
    # docs/followup_answer_shape.md, "Iteration 2 (negative result)":
    # the stronger follow-up note over-corrected (a spurious "Yes," tic
    # on open questions, flattened "Tell me about X" answers in some
    # rewordings), so the default is False -- the flag stays selectable.
    assert cfg.app.concise_followup_note is False
    # docs/followup_answer_shape.md, "Iteration 3": not yet adopted --
    # both default to False.
    assert cfg.app.restate_question_last is False
    assert cfg.app.restate_question_instruction is False


def test_app_table_restate_question_settings_are_configurable(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    text = _BASE + """
[app]
restate_question_last = true
restate_question_instruction = true
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.restate_question_last is True
    assert cfg.app.restate_question_instruction is True


def test_app_table_concise_followup_note_is_configurable(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    text = _BASE + """
[app]
concise_followup_note = true
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.concise_followup_note is True


def test_app_table_rewrite_on_weak_evidence_is_configurable(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    text = _BASE + """
[app]
rewrite_on_weak_evidence = false
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.rewrite_on_weak_evidence is False


def test_app_table_rewrite_on_followup_is_configurable(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    text = _BASE + """
[app]
rewrite_on_followup = false
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.rewrite_on_followup is False


def test_app_table_answer_max_tokens_is_configurable(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    text = _BASE + """
[app]
answer_max_tokens = 512
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.answer_max_tokens == 512


def test_app_table_explicit_values_are_honored(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    text = _BASE + """
[app]
host = "0.0.0.0"
port = 9000
data_dir = "custom_data"
registry_path = "config/archives.custom.toml"
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.host == "0.0.0.0"
    assert cfg.app.port == 9000
    assert cfg.app.data_dir == (tmp_path / "custom_data").resolve()
    assert cfg.app.registry_path == (tmp_path / "config" / "archives.custom.toml").resolve()


def test_app_table_absolute_paths_are_kept_as_is(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    abs_data_dir = tmp_path / "elsewhere" / "data"
    # Use a TOML literal string (single quotes) so backslashes on Windows
    # are not treated as escape sequences.
    text = _BASE + f"""
[app]
data_dir = '{abs_data_dir}'
"""
    config_path = _write_toml(config_dir / "dev.toml", text)

    cfg = load_config(config_path)

    assert cfg.app.data_dir == abs_data_dir.resolve()
