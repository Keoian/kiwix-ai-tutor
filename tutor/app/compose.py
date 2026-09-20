"""Wires the tutor app's real collaborators into an ``AppDeps`` (WP-C4).

``build_deps(cfg)`` builds the production ``LlamaClient``, ``ResearchEngine``
(over the on-disk archive registry), ``SnapshotStore``, an in-memory session
store, the sandboxed ``calc`` tool, a ``Budget`` derived from
``cfg.server.ctx_size``, a ``turn_runner`` adapting
``tutor.app.agent_loop.run_turn`` to the routes' ``(session_id, user_input,
emit, cancel)`` signature, and a ``status_provider`` reporting LLM health,
archive validation state, the model file name, the context ceiling, and
per-session token usage.

Nothing here ever imports the retrieval package's low-level ZIM layer
directly: archive validation goes through ``Registry.validate_all()`` and
archive/worker construction goes through ``tutor.retrieval.research.ResearchEngine``.
A failed archive (missing file, truncated, not a ZIM, no fulltext index) is
never fatal to ``build_deps`` -- it is simply excluded from the
``ResearchEngine``'s registry and reported (by id/state/storage) in
``status_provider``'s output.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import threading
from dataclasses import dataclass
from typing import Any

from tutor.app.citations import resolve_citations
from tutor.app.lesson_state import LessonStore
from tutor.app.llm_client import LlamaClient, LlamaError, StreamEvent
from tutor.app.main import AppDeps
from tutor.app.profiles import ProfileStore
from tutor.app.prompt import Budget
from tutor.app.resources import ResourceMonitor
from tutor.app.session import Session
from tutor.app.turn_log import TurnLogger
from tutor.retrieval.registry import Registry, RegistryError, load_registry
from tutor.retrieval.research import ResearchEngine
from tutor.retrieval.snapshots import SnapshotStore
from tutor.tools import calc_tool

_STUDENT_SAFE_ERROR = (
    "Sorry, I ran into a problem answering that. Please try asking again."
)

_TOKEN_CACHE_SIZE = 4096


def _make_count_tokens(llm: Any):
    """Token counting through ``llm.count_tokens`` (i.e. ``/tokenize``),
    LRU-cached, falling back to ``ceil(len(text) / 3.5)`` when the server
    cannot be reached."""

    @functools.lru_cache(maxsize=_TOKEN_CACHE_SIZE)
    def _count(text: str) -> int:
        try:
            return llm.count_tokens(text)
        except LlamaError:
            return math.ceil(len(text) / 3.5)
        except Exception:  # noqa: BLE001 - never let token counting crash a turn
            return math.ceil(len(text) / 3.5)

    return _count


def _load_registry_safely(registry_path) -> Registry:
    try:
        return load_registry(registry_path)
    except RegistryError:
        return Registry(archives=())


def _valid_registry(registry: Registry) -> Registry:
    """The subset of ``registry`` whose archives pass validation -- the one
    handed to ``ResearchEngine`` so a missing/broken archive can never make
    a research call blow up."""
    statuses = registry.validate_all()
    valid = tuple(
        entry for entry in registry.archives if statuses[entry.id].state.value == "valid"
    )
    return Registry(archives=valid)


class _CalcTool:
    """Adapts ``tutor.tools.calc_tool.evaluate`` to the ``calc.evaluate(...)``
    interface ``tutor.app.agent_loop.run_turn`` expects."""

    def evaluate(self, expression: str) -> dict:
        return calc_tool.evaluate(expression)


class _SessionStore:
    """Minimal in-memory session store matching the routes' contract:
    ``create() -> str``, ``get(session_id) -> Session | None``, and
    ``set_subject(session_id, subject) -> bool``."""

    def __init__(self, count_tokens) -> None:
        self._count_tokens = count_tokens
        self._sessions: dict[str, Session] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def create(self) -> str:
        with self._lock:
            self._counter += 1
            session_id = f"sess-{self._counter}"
            self._sessions[session_id] = Session(self._count_tokens)
        return session_id

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def set_subject(self, session_id: str, subject: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.subject_hint = subject
        return True


@dataclass
class _UserInputLike:
    """Structural stand-in accepted by ``run_turn``: any object with
    ``kind``/``text``/``action`` attributes, matching
    ``tutor.app.routes._UserInput``."""

    kind: str
    text: str | None = None
    action: str | None = None


def _make_turn_runner(*, sessions: _SessionStore, llm, research_engine, calc, budget):
    from tutor.app.agent_loop import run_turn

    def turn_runner(session_id, user_input, emit, cancel) -> None:
        session = sessions.get(session_id)
        if session is None:
            emit("error", {"message": _STUDENT_SAFE_ERROR})
            return

        def adapter(obj: Any) -> None:
            if isinstance(obj, StreamEvent):
                if obj.kind == "token":
                    emit("token", {"text": obj.text or ""})
                elif obj.kind == "tool_call":
                    emit("tool", {"name": obj.name, "phase": "call"})
                # "error" and "done" StreamEvents carry no UI-facing frame
                # of their own here: the turn's final status (below) is
                # what the student sees, always via a student-safe message.
                return
            if isinstance(obj, dict) and obj.get("kind") == "tool_result":
                emit(
                    "tool",
                    {"name": obj.get("name"), "ok": obj.get("ok", True), "phase": "result"},
                )

        try:
            result = run_turn(
                session,
                user_input,
                llm=llm,
                research_engine=research_engine,
                calc=calc,
                budget=budget,
                emit=adapter,
                cancel=cancel,
            )
        except Exception:  # noqa: BLE001 - never leak a traceback to the student
            emit("error", {"message": _STUDENT_SAFE_ERROR})
            return

        if result.status == "ok":
            citations = resolve_citations(result.answer_text, session.known_passages())
            emit(
                "citations",
                {"citations": [dataclasses.asdict(c) for c in citations]},
            )
            emit(
                "done",
                {
                    "status": "ok",
                    "answer": result.answer_text,
                    "route": result.route,
                    "research_calls": result.research_calls,
                    "calc_calls": result.calc_calls,
                    "cached_tokens": result.cached_tokens or 0,
                    "tokens_used": session.log.tokens_used(),
                    "uncited": result.uncited,
                },
            )
        elif result.status == "cancelled":
            emit("done", {"status": "cancelled"})
        else:
            emit("error", {"message": result.answer_text or _STUDENT_SAFE_ERROR})

    return turn_runner


def _make_status_provider(
    *,
    llm,
    registry: Registry,
    sessions: _SessionStore,
    budget: Budget,
    model_name: str,
    resource_monitor: ResourceMonitor | None = None,
):
    def status_provider(session_id: str | None = None) -> dict:
        healthy = llm.health()
        statuses = registry.validate_all()
        storage_by_id = {entry.id: entry.storage for entry in registry.archives}
        archives = [
            {
                "id": archive_id,
                "state": status.state.value.upper(),
                "storage": storage_by_id.get(archive_id),
            }
            for archive_id, status in statuses.items()
        ]

        result: dict = {
            "llm": {"healthy": healthy},
            "archives": archives,
            "model_path": model_name,
            "context_ceiling": budget.ceiling,
            "tokens_used": None,
            "headroom": None,
            "last_eviction": None,
        }

        if resource_monitor is not None:
            sample = resource_monitor.sample()
            result["resources"] = {
                "rss_mb": sample.rss_mb,
                "open_files": sample.open_files,
                "thread_count": sample.thread_count,
                "child_process_count": sample.child_process_count,
            }

        if session_id is not None:
            session = sessions.get(session_id)
            if session is not None:
                result["tokens_used"] = session.log.tokens_used()
                result["headroom"] = session.log.headroom(budget)
        return result

    return status_provider


def build_deps(cfg: Any, *, llm: Any = None, research_engine: Any = None) -> AppDeps:
    """Build the production ``AppDeps`` for ``cfg`` (a ``tutor.settings.Config``).

    ``llm`` and ``research_engine`` may be overridden (tests only) to inject
    fakes without a live llama-server or a real ZIM archive.
    """
    data_dir = cfg.app.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if llm is None:
        llm = LlamaClient(cfg.server.base_url)

    full_registry = _load_registry_safely(cfg.app.registry_path)

    snapshot_store = SnapshotStore(data_dir / "snapshots.sqlite")

    if research_engine is None:
        engine_registry = _valid_registry(full_registry)
        research_engine = ResearchEngine(
            engine_registry,
            snapshot_store=snapshot_store,
            cache_dir=data_dir / "research_cache",
        )

    count_tokens = _make_count_tokens(llm)
    sessions = _SessionStore(count_tokens)
    calc = _CalcTool()
    budget = Budget.for_ceiling(cfg.server.ctx_size)

    profiles = ProfileStore(data_dir / "profiles.sqlite")
    lessons = LessonStore(data_dir / "lessons.sqlite")
    turn_logger = TurnLogger(data_dir / "logs" / "turns.jsonl")
    resource_monitor = ResourceMonitor()

    turn_runner = _make_turn_runner(
        sessions=sessions,
        llm=llm,
        research_engine=research_engine,
        calc=calc,
        budget=budget,
    )
    status_provider = _make_status_provider(
        llm=llm,
        registry=full_registry,
        sessions=sessions,
        budget=budget,
        model_name=cfg.runtime.model_path.name,
        resource_monitor=resource_monitor,
    )

    return AppDeps(
        turn_runner=turn_runner,
        snapshot_store=snapshot_store,
        status_provider=status_provider,
        sessions=sessions,
        subjects=["general", "math", "history"],
        profiles=profiles,
        lessons=lessons,
        turn_logger=turn_logger,
        resource_monitor=resource_monitor,
    )
