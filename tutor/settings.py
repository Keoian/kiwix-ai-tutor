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
    extra_args: tuple[str, ...] = ()

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
        argv.extend(self.extra_args)
        return argv


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    data_dir: Path
    registry_path: Path
    prompt_variant: str = "current"
    """Selects a prompt-assembly variant by name. Adopted 2026-09-20
    (docs/citation_experiment.md, "Seed exchange A/B"): the seed-exchange
    variant (``tutor.app.seed_exchange.SEED_EXCHANGE_VARIANT``) was made the
    default on the single-turn A/B (pooled n=54, cited-and-supported
    0.09->0.35, evidence_dump tied at 0.04, paired W19/L5/T30) -- then
    reverted the same day (see docs/citation_experiment.md "In-lesson check
    and reversal"): paired in-lesson soaks showed the seed lengthens answers
    and does not improve lesson citation placement, so ``"current"``
    (today's byte-identical, unseeded prompt) is the default again.
    ``seed_exchange_s0`` stays fully selectable and tested. See
    ``eval.system_prompt_variants.VARIANTS`` for the registered values."""
    answer_max_tokens: int = 2000
    """Output token cap passed as ``max_tokens`` on every generation call
    (app agent loop and eval harnesses), including tool-call rounds.
    Default 2000 matches ``tutor.app.prompt.Budget.generation`` -- the
    reserved-generation-tokens figure already used in the
    ``rendered_prompt_tokens + reserved_generation_tokens + safety_margin
    <= configured_context`` accounting (docs/plan/offline_tutor_spec_v0.3.md
    §8, line ~239). The spec does not separately name an answer/output
    token budget; reusing this number avoids inventing a second one. See
    docs/citation_experiment.md, "Seed exchange A/B (2026-09-20)", for the
    unbounded-generation defect this closes."""
    rewrite_on_weak_evidence: bool = True
    """When the deterministic pre-search for a factual turn is assessed
    ``weak``/``empty`` (``tutor.retrieval.assessment.assess_evidence``),
    force one model-written query-rewrite tool call before answering
    (see docs/rewrite_on_weak_evidence.md) rather than trusting the model
    to volunteer a search on its own. ``False`` reproduces today's
    behaviour byte-for-byte (see
    tests/test_agent_loop_rewrite.py::test_setting_off_is_byte_identical)."""
    rewrite_on_followup: bool = True
    """On every turn after the lesson's first, force a model-written
    query-rewrite round BEFORE trusting the raw pre-search (see
    docs/rewrite_on_weak_evidence.md, "Follow-up rewrite"), regardless of
    how strong the raw pre-search looks -- deliberately unconditional
    (never gated by a word-list/pronoun detector: a student's own
    grammar/spelling can't be relied on to signal an unresolved
    reference like "it"/"that"). The raw pre-search still always runs and
    its passages are kept only as backfill behind the rewrite's own
    results. ``False`` reproduces today's behaviour (no follow-up
    rewrite; the existing weak/empty-evidence rewrite still applies)."""
    reuse_prior_passages: bool = True
    """When a turn's evidence packet contains a passage whose full text
    is already present earlier in the lesson's prompt log (tracked by
    passage id, and re-derived on resume -- see
    ``tutor.app.prompt.PromptLog``), paste only a short pointer line for
    it instead of the full text again (see docs/passage_reuse.md). A
    passage evicted from the log (its text no longer actually present) is
    treated as not-held and re-pasted in full. If every passage in a
    turn's packet is already held, a short host instruction is appended
    telling the model to answer the specific question from the sources
    already shown rather than re-summarising. Citation/attribution
    resolution always sees the full passage text regardless of this
    setting -- only what is pasted into the prompt log is shortened.
    ``False`` reproduces today's behaviour byte-for-byte."""
    restate_question_last: bool = True
    """On a follow-up turn's forced-rewrite round, append a one-line
    restatement of the resolved standalone question (the model's own
    rewritten query from that round) as the LAST thing in the evidence
    tool result, right before the model generates -- see
    docs/followup_answer_shape.md, "Iteration 3". Theory: recency in the
    transcript matters more than instruction wording for this model, and
    without a restated question nearest the end of the prompt, it
    defaults to anchoring on its own most recent prior answer instead of
    the newly-resolved question. ``False`` reproduces today's behaviour
    byte-for-byte. Only applies on turn >= 2 (a follow-up rewrite round);
    never appended on turn 1."""
    restate_question_instruction: bool = False
    """Only meaningful when ``restate_question_last`` is True: also
    append one extra instruction sentence after the restated question
    ("If it is a yes/no question start with Yes or No; otherwise just
    answer it. Add what is new; do not repeat your earlier answer.") --
    the "R2" arm in docs/followup_answer_shape.md, "Iteration 3", vs.
    "R1" (the restated question alone, no extra instruction)."""
    concise_followup_note: bool = False
    """Use the longer follow-up host note (``_FOLLOWUP_CONCISE_NOTE``,
    docs/followup_answer_shape.md) instead of the plain
    ``_FOLLOWUP_DIRECTNESS_NOTE``. Iteration 1 measured this note cutting
    same-topic follow-up answers from ~120 near-duplicate tokens
    (baseline, Jaccard up to 1.0 turn-to-turn) to ~30-60 tokens. Iteration
    2 (docs/followup_answer_shape.md, "Iteration 2 (negative result)")
    found this over-corrected: on turns 6-7 of the same lesson it
    prefixed open, non-yes/no questions ("How does it copy itself?",
    "Why does that matter?") with a spurious "Yes,", and several
    negative-result rewordings tried to fix that instead flattened
    "Tell me about DNA" into a 1-sentence non-answer or dropped citations
    entirely -- wording alone did not fix the underlying "the model
    parrots its own earlier answer" defect. Left in the codebase and
    still selectable (``True``) for further experimentation, but
    **default is False**: reproduces the plain, pre-existing
    ``_FOLLOWUP_DIRECTNESS_NOTE`` wording byte-for-byte."""
    model_writes_search: bool = True
    """Force a model-written ``research`` tool call BEFORE answering on
    EVERY turn, including turn 1, rather than only on weak evidence
    (``rewrite_on_weak_evidence``) or turn >= 2 (``rewrite_on_followup``,
    which this replaces for that turn rather than adding to). Owner
    decision (see docs/rewrite_on_weak_evidence.md, "Model writes every
    search"): the deterministic pre-search word-matches the student's RAW
    text, which on turn 1 already produces wrong-topic hits no rewrite
    ever gets a chance to fix (e.g. "What's the largest molecule?" ->
    "Molecule Man"; "What's the biggest animal?" -> "The Biggest Loser").
    The model is told to write short, article-title-like queries, not
    full sentences. **Default is False** -- not yet measured against a
    live model; the measuring agent decides whether to flip it."""
    model_may_skip_search: bool = True
    """Only meaningful when ``model_writes_search`` is True: also let the
    model set ``needs_search: false`` on the same forced ``research`` call
    to skip the search entirely for this turn (see
    docs/rewrite_on_weak_evidence.md, "Model may skip the search"). Owner
    report: after a lesson about the fastest animal and Usain Bolt, the
    student asked "How fast am I?" then "But what about me personally?"
    -- no library search can ever answer a question about the student's
    own speed, so the tutor should just answer conversationally instead of
    running (and re-running) the same Usain Bolt search. **Default is
    True** -- ``False`` reproduces today's bytes exactly (the forced call
    still always runs one search)."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """WP-B7: config for a second llama-server started with ``--embedding``.

    Model path resolves against ``[runtime].runtime_dir`` the same way
    ``[runtime].model_path`` does, so the embedding model can live in the
    same runtime checkout without repeating the runtime dir. Optional --
    a config with no ``[embedding]`` table simply has no dense sidecar
    support (scripts/serve_embed.* require it; the chat server does not).
    """

    host: str
    port: int
    model_path: Path
    dim: int
    n_gpu_layers: int
    threads: int
    ctx_size: int
    sidecar_dir: Path
    archive_id: str
    enabled: bool

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class Config:
    runtime: RuntimeConfig
    server: ServerConfig
    sampling: SamplingConfig
    app: AppConfig
    embedding: EmbeddingConfig | None


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

    extra_args_value = server_table.get("extra_args", [])
    if not isinstance(extra_args_value, list) or not all(
        isinstance(item, str) for item in extra_args_value
    ):
        raise ConfigError("'extra_args' in [server] must be a list of strings")
    extra_args = tuple(extra_args_value)

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
        extra_args=extra_args,
    )

    temperature = _get(sampling_table, "sampling", "temperature", float)
    top_p = _get(sampling_table, "sampling", "top_p", float)
    top_k = _get(sampling_table, "sampling", "top_k", int)

    sampling = SamplingConfig(temperature=temperature, top_p=top_p, top_k=top_k)

    # [app] is optional; every key defaults, and relative data_dir/
    # registry_path values resolve against the repo root, i.e. the config
    # file's parent's parent (config/dev.toml -> repo root).
    app_table = data.get("app", {})
    if not isinstance(app_table, dict):
        raise ConfigError("[app] must be a table")

    repo_root = path.resolve().parent.parent

    def _app_str(key: str, default: str) -> str:
        value = app_table.get(key, default)
        return _require_type(value, str, "app", key)

    def _app_int(key: str, default: int) -> int:
        value = app_table.get(key, default)
        return _require_type(value, int, "app", key)

    def _app_bool(key: str, default: bool) -> bool:
        value = app_table.get(key, default)
        return _require_type(value, bool, "app", key)

    def _app_path(key: str, default: str) -> Path:
        value = app_table.get(key, default)
        value = _require_type(value, str, "app", key)
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate.resolve()

    app = AppConfig(
        host=_app_str("host", "127.0.0.1"),
        port=_app_int("port", 8420),
        data_dir=_app_path("data_dir", "data"),
        registry_path=_app_path("registry_path", "config/archives.dev.toml"),
        prompt_variant=_app_str("prompt_variant", "current"),
        answer_max_tokens=_app_int("answer_max_tokens", 2000),
        rewrite_on_weak_evidence=_app_bool("rewrite_on_weak_evidence", True),
        rewrite_on_followup=_app_bool("rewrite_on_followup", True),
        reuse_prior_passages=_app_bool("reuse_prior_passages", True),
        concise_followup_note=_app_bool("concise_followup_note", False),
        restate_question_last=_app_bool("restate_question_last", True),
        restate_question_instruction=_app_bool("restate_question_instruction", False),
        model_writes_search=_app_bool("model_writes_search", True),
        model_may_skip_search=_app_bool("model_may_skip_search", True),
    )

    # [embedding] is optional, like [app]; when present every key is
    # required (no sensible default for a model path or port pair that
    # must not collide with [server].port).
    embedding_table = data.get("embedding")
    embedding: EmbeddingConfig | None = None
    if embedding_table is not None:
        if not isinstance(embedding_table, dict):
            raise ConfigError("[embedding] must be a table")
        emb_host = _get(embedding_table, "embedding", "host", str)
        emb_port = _get(embedding_table, "embedding", "port", int)
        emb_model_path_str = _get(embedding_table, "embedding", "model_path", str)
        emb_dim = _get(embedding_table, "embedding", "dim", int)
        emb_model_path = Path(emb_model_path_str)
        if not emb_model_path.is_absolute():
            emb_model_path = runtime_dir / emb_model_path
        if emb_dim <= 0:
            raise ConfigError("'dim' in [embedding] must be a positive integer")

        def _emb_int(key: str, default: int) -> int:
            value = embedding_table.get(key, default)
            return _require_type(value, int, "embedding", key)

        def _emb_str(key: str, default: str) -> str:
            value = embedding_table.get(key, default)
            return _require_type(value, str, "embedding", key)

        def _emb_bool(key: str, default: bool) -> bool:
            value = embedding_table.get(key, default)
            return _require_type(value, bool, "embedding", key)

        emb_n_gpu_layers = _emb_int("n_gpu_layers", 0)
        emb_threads = _emb_int("threads", 4)
        emb_ctx_size = _emb_int("ctx_size", 512)
        emb_archive_id = _emb_str("archive_id", "simplewiki")
        # Default False: the M4 held-out gate FAILED (docs/M4_report.md), so
        # dense/hybrid retrieval must never be a silent default. A config
        # must opt in explicitly with `enabled = true`.
        emb_enabled = _emb_bool("enabled", False)
        emb_sidecar_dir_str = _emb_str("sidecar_dir", "runtime/simplewiki_dense")
        emb_sidecar_dir = Path(emb_sidecar_dir_str)
        if not emb_sidecar_dir.is_absolute():
            emb_sidecar_dir = repo_root / emb_sidecar_dir
        emb_sidecar_dir = emb_sidecar_dir.resolve()

        embedding = EmbeddingConfig(
            host=emb_host,
            port=emb_port,
            model_path=emb_model_path,
            dim=emb_dim,
            n_gpu_layers=emb_n_gpu_layers,
            threads=emb_threads,
            ctx_size=emb_ctx_size,
            sidecar_dir=emb_sidecar_dir,
            archive_id=emb_archive_id,
            enabled=emb_enabled,
        )

    return Config(runtime=runtime, server=server, sampling=sampling, app=app, embedding=embedding)


def _main(argv: list[str]) -> int:
    """Tiny CLI: `python -m tutor.settings --argv <config.toml>`.

    Prints the resolved server binary path on the first line, followed by
    one argv element per line, so a shell script can build a command line
    without duplicating flag-building logic.
    """
    if len(argv) != 2 or argv[0] not in ("--argv", "--argv-embedding"):
        print(
            "usage: python -m tutor.settings (--argv|--argv-embedding) <config.toml>",
            file=sys.stderr,
        )
        return 2

    mode = argv[0]
    config_path = Path(argv[1])
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if mode == "--argv":
        print(str(cfg.runtime.server_binary))
        for item in cfg.server.to_argv(cfg.runtime, cfg.sampling):
            print(item)
        return 0

    # --argv-embedding: same binary, a minimal argv for an embedding server
    # (no sampling flags -- embeddings do not sample).
    if cfg.embedding is None:
        print("error: config has no [embedding] table", file=sys.stderr)
        return 1
    print(str(cfg.runtime.server_binary))
    for item in [
        "-m", str(cfg.embedding.model_path),
        "--host", cfg.embedding.host,
        "--port", str(cfg.embedding.port),
        "--embedding",
        "-ngl", str(cfg.embedding.n_gpu_layers),
        "-t", str(cfg.embedding.threads),
        "-c", str(cfg.embedding.ctx_size),
    ]:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
