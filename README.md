# kiwix-ai-tutor

An offline tutoring app for a student with no internet. A local `llama-server` runs a small model;
the app retrieves evidence from Kiwix ZIM archives (offline Wikipedia, OER textbooks, Stack
Exchange), feeds it to the model as citations, and teaches rather than answers.

Two model-facing tools only: `research` and `calc`.

**Both Windows and Linux are shipping targets**, not a dev/prod split. Development happens on
Windows; the delivery machine is a Dell with a GTX 1060 6 GB running Linux Mint.

## Documents

| File | What it governs |
|---|---|
| [`docs/plan/offline_tutor_implementation_plan.md`](docs/plan/offline_tutor_implementation_plan.md) | **Primary.** Milestones, work packages, order of work, cross-platform rules. |
| [`docs/plan/offline_tutor_spec_v0.3.md`](docs/plan/offline_tutor_spec_v0.3.md) | Architecture, budgets, gates. Wins on budgets, caps and thresholds. |
| [`docs/plan/offline_tutor_kiwix_reuse_plan.md`](docs/plan/offline_tutor_kiwix_reuse_plan.md) | Donor code and algorithms. Wins on algorithm choices. |
| [`HANDOFF.md`](HANDOFF.md) | Machine state, measured runtime numbers, working agreement. |

## Layout

```
tutor/          the importable package: app, retrieval, tools, ui, dev, platform_
config/         dev.toml (Windows dev machine, Ling 3.0 Tiny default since
                2026-09-21; dev.granite.toml is the Granite 4.0 H-Tiny
                fallback, dev.bonsai-q1.toml the Bonsai Q1_0 alternative),
                dell.toml (delivery machine)
scripts/        launch, build, install and benchmark scripts, .ps1/.sh in pairs
eval/           question sets and the eval runners
tests/          unit tests, plus integration tests marked `integration`
runtime/        gitignored: local llama.cpp builds and model weights
```

`tutor/app/` never imports `tutor/retrieval/zim/` directly — everything goes through
`tutor/retrieval/research.py`.

## Development

```
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -m "not integration"
```

`ruff` carries the machine-checkable half of the cross-platform rules: `pathlib` instead of
`os.path`, an explicit `encoding=` on every `open()`, and a hard ban on `resource`, `fcntl`,
`os.fork` and `signal.SIGKILL` outside `tutor/platform_/`.

Tests that need a live `llama-server`, a real ZIM archive, or a performance number are marked
`integration` and run on the developer's machines, not in CI.
