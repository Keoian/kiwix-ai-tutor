"""Configuration loading for the tutor's dev llama-server launcher.

Reads a TOML config file into frozen dataclasses and exposes ``to_argv`` to
build the llama-server command line. No runtime path or model name is
hard-coded here; everything comes from the config file passed to
``load_config``.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

_VALID_CACHE_TYPES = {"q8_0", "q4_0", "f16"}


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    runtime_dir: Path
    model_path: Path
    server_binary: Path


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    ctx_size: int
    cache_type_k: str
    cache_type_v: str
    n_gpu_layers: int
    parallel: int
    flash_attn: bool
    jinja: bool
    slots: bool

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_argv(self, runtime: RuntimeConfig, sampling: SamplingConfig) -> list[str]:
        argv = [
            "-m", str(runtime.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-fa", "on" if self.flash_attn else "off",
            "-c", str(self.ctx_size),
            "-np", str(self.parallel),
            "-ctk", self.cache_type_k,
            "-ctv", self.cache_type_v,
            "--temp", str(sampling.temperature),
            "--top-p", str(sampling.top_p),
            "--top-k", str(sampling.top_k),
        ]
        if self.jinja:
            argv.append("--jinja")
        if self.slots:
            argv.append("--slots")
        return argv


@dataclass(frozen=True)
class Config:
    runtime: RuntimeConfig
    server: ServerConfig
    sampling: SamplingConfig


def _require_table(data: dict, name: str) -> dict:
    table = data.get(name)
    if table is None:
        raise ConfigError(f"missing required table: [{name}]")
    if not isinstance(table, dict):
        raise ConfigError(f"[{name}] must be a table")
    return table


def _require_key(table: dict, table_name: str, key: str):
    if key not in table:
        raise ConfigError(f"missing required key '{key}' in [{table_name}]")
    return table[key]


def _require_type(value, expected: type, table_name: str, key: str):
    # bool is a subclass of int in Python, so guard against port=true style mistakes
    # and against numbers where a string is required, and vice versa.
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{key}' in [{table_name}] must be a boolean")
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{key}' in [{table_name}] must be an integer")
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{key}' in [{table_name}] must be a number")
        value = float(value)
    elif expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{key}' in [{table_name}] must be a string")
    return value


def _get(table: dict, table_name: str, key: str, expected: type):
    value = _require_key(table, table_name, key)
    return _require_type(value, expected, table_name, key)


def load_config(path: Path) -> Config:
    path = Path(path)
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    runtime_table = _require_table(data, "runtime")
    server_table = _require_table(data, "server")
    sampling_table = _require_table(data, "sampling")

    runtime_dir_str = _get(runtime_table, "runtime", "runtime_dir", str)
    model_path_str = _get(runtime_table, "runtime", "model_path", str)
    server_binary_str = _get(runtime_table, "runtime", "server_binary", str)

    runtime_dir = Path(runtime_dir_str)

    model_path = Path(model_path_str)
    if not model_path.is_absolute():
        model_path = runtime_dir / model_path

    server_binary = Path(server_binary_str)
    if not server_binary.is_absolute():
        server_binary = runtime_dir / server_binary

    runtime = RuntimeConfig(
        runtime_dir=runtime_dir,
        model_path=model_path,
        server_binary=server_binary,
    )

    host = _get(server_table, "server", "host", str)
    port = _get(server_table, "server", "port", int)
    ctx_size = _get(server_table, "server", "ctx_size", int)
    cache_type_k = _get(server_table, "server", "cache_type_k", str)
    cache_type_v = _get(server_table, "server", "cache_type_v", str)
    n_gpu_layers = _get(server_table, "server", "n_gpu_layers", int)
    parallel = _get(server_table, "server", "parallel", int)
    flash_attn = _get(server_table, "server", "flash_attn", bool)
    jinja = _get(server_table, "server", "jinja", bool)
    slots = _get(server_table, "server", "slots", bool)

    if ctx_size <= 0:
        raise ConfigError("ctx_size must be a positive integer")
    if cache_type_k not in _VALID_CACHE_TYPES:
        raise ConfigError(
            f"cache_type_k must be one of {sorted(_VALID_CACHE_TYPES)}, got {cache_type_k!r}"
        )
    if cache_type_v not in _VALID_CACHE_TYPES:
        raise ConfigError(
            f"cache_type_v must be one of {sorted(_VALID_CACHE_TYPES)}, got {cache_type_v!r}"
        )

    server = ServerConfig(
        host=host,
        port=port,
        ctx_size=ctx_size,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        n_gpu_layers=n_gpu_layers,
        parallel=parallel,
        flash_attn=flash_attn,
        jinja=jinja,
        slots=slots,
    )

    temperature = _get(sampling_table, "sampling", "temperature", float)
    top_p = _get(sampling_table, "sampling", "top_p", float)
    top_k = _get(sampling_table, "sampling", "top_k", int)

    sampling = SamplingConfig(temperature=temperature, top_p=top_p, top_k=top_k)

    return Config(runtime=runtime, server=server, sampling=sampling)


def _main(argv: list[str]) -> int:
    """Tiny CLI: `python -m tutor.settings --argv <config.toml>`.

    Prints the resolved server binary path on the first line, followed by
    one argv element per line, so a shell script can build a command line
    without duplicating flag-building logic.
    """
    if len(argv) != 2 or argv[0] != "--argv":
        print("usage: python -m tutor.settings --argv <config.toml>", file=sys.stderr)
        return 2

    config_path = Path(argv[1])
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(cfg.runtime.server_binary))
    for item in cfg.server.to_argv(cfg.runtime, cfg.sampling):
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
