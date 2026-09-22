"""Tutor app entrypoint: FastAPI app assembly (WP-C3).

``create_app(deps)`` wires the static UI, the CSP header, and the API
routes from ``tutor.app.routes`` onto a single FastAPI app. ``AppDeps`` is
the injection point the routes and this module depend on; it holds only
plain callables/interfaces so the app can be exercised in tests without a
real model, archive, or network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from tutor.app.routes import build_router

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"

_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self'; connect-src 'self'"
)


@dataclass
class AppDeps:
    turn_runner: Callable[..., Any]
    snapshot_store: Any
    status_provider: Callable[..., dict]
    sessions: Any
    subjects: list[str]
    # Optional (WP-C4): profiles / lessons / resource monitoring. Left as
    # None by any deps built before this work package; routes guard on
    # their presence rather than requiring them.
    profiles: Any = None
    lessons: Any = None
    turn_logger: Any = None
    resource_monitor: Any = None
    # Optional: the ResearchEngine, exposed so routes can re-fetch article
    # text for the source-viewer "show more" context endpoint. None for any
    # deps built before this was added; the route guards on its presence.
    research_engine: Any = None

    def close(self) -> None:
        """Release resources opened by ``build_deps`` (SQLite connections).

        Idempotent-ish best-effort: each store's ``close`` is independent,
        so one failing doesn't stop the others from closing. Safe to call
        on deps built with fakes that lack a ``close`` method.
        """
        for store in (self.snapshot_store, self.profiles, self.lessons, self.research_engine):
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> AppDeps:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        return response


def create_app(deps: AppDeps) -> FastAPI:
    app = FastAPI()
    app.state.deps = deps

    app.add_middleware(_CSPMiddleware)

    app.include_router(build_router(deps))

    app.mount("/static", StaticFiles(directory=_UI_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(_UI_DIR / "index.html", media_type="text/html")

    return app


def build_uvicorn_config(cfg: Any, app: Any = None):
    """Build a loopback-bound uvicorn config from a ``tutor.settings.Config``."""
    import uvicorn

    return uvicorn.Config(
        app=app if app is not None else "tutor.app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
    )


def _default_serve(app: FastAPI, *, host: str, port: int) -> None:
    """Run ``app`` under uvicorn on ``host``/``port``. Only exercised for
    real by the run scripts, never by the unit test suite (`serve` is an
    injectable seam for tests)."""
    import logging

    import uvicorn

    # The tutor's own INFO lines (host topic gate, clarify diagnostics:
    # "presearch unknown_terms=..." / "clarify word=... raw=... -> ...")
    # are the only way to see WHY a turn took the path it did without a
    # debugger. Root stays at WARNING so third-party chatter is unchanged.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("tutor").setLevel(logging.INFO)
    uvicorn.run(app, host=host, port=port)


def _parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m tutor.app.main")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file (e.g. config/dev.toml).",
    )
    return parser.parse_args(argv)


def _main(argv: list[str], *, serve: Callable[..., None] = _default_serve) -> int:
    import sys

    from tutor.settings import ConfigError, load_config

    args = _parse_args(argv)

    try:
        cfg = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    from tutor.app.compose import build_deps

    deps = build_deps(cfg)
    app = create_app(deps)
    serve(app, host=cfg.app.host, port=cfg.app.port)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
