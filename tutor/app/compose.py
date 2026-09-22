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
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

from tutor.app.citations import (
    _figures,
    attribute_sentences,
    detect_evidence_dump,
    resolve_citations,
)
from tutor.app.computed_check import check_computed_statements
from tutor.app.lesson_state import LessonStore
from tutor.app.llm_client import LlamaClient, LlamaError, StreamEvent
from tutor.app.main import AppDeps
from tutor.app.profiles import ProfileStore
from tutor.app.prompt import Budget
from tutor.app.resources import ResourceMonitor
from tutor.app.session import Session
from tutor.app.turn_log import TurnLogger
from tutor.retrieval.hybrid.dense import DenseIndex, DenseIndexError
from tutor.retrieval.index.embedding_client import EmbeddingClient
from tutor.retrieval.registry import Registry, RegistryError, load_registry
from tutor.retrieval.research import ResearchEngine
from tutor.retrieval.snapshots import SnapshotStore
from tutor.tools import calc_tool

_STUDENT_SAFE_ERROR = (
    "Sorry, I ran into a problem answering that. Please try asking again."
)

_logger = logging.getLogger(__name__)

_TOKEN_CACHE_SIZE = 4096

# Short timeout for the per-query embedding call: research()'s own
# ``embed_query`` bound (see tutor.retrieval.research) is 1s, so a longer
# client-level timeout here would never actually be reached -- but it
# still matters that a hung TCP connection to a dead embedding server
# cannot block a request past that budget.
_EMBED_CLIENT_TIMEOUT_S = 2.0

_DENSE_UNAVAILABLE_STATUS = {
    "available": False,
    "reason": "not configured",
    "rows": None,
    "model": None,
}


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


def _open_dense(cfg: Any, engine_registry: Registry) -> tuple[dict[str, DenseIndex], Any, dict]:
    """Best-effort dense sidecar wiring for ``build_deps``.

    Never raises: any problem (no ``[embedding]`` config, archive not in
    the *valid* registry, sidecar directory missing manifest.json or
    paths.txt, stale fingerprint, corrupt sidecar) is reported as a
    ``dense`` status dict with ``available: False`` and a human-readable
    ``reason``, and the caller falls back to lexical-only retrieval. This
    never talks to the embedding server: reachability is checked lazily,
    per query, by ``ResearchEngine`` itself (see ``embed_query`` there),
    which is also what keeps this function safe to call with no network
    access in unit tests.
    """
    emb = getattr(cfg, "embedding", None)
    if emb is None:
        return {}, None, dict(_DENSE_UNAVAILABLE_STATUS)

    if not getattr(emb, "enabled", False):
        # M4 gate FAILED on held-out (docs/M4_report.md): hybrid beat
        # lexical-v2 on tuning but recall@1/MRR got worse on held-out, so
        # dense retrieval must never be a silent default. A config must
        # opt in explicitly with [embedding].enabled = true.
        return {}, None, {
            "available": False,
            "reason": "disabled in config",
            "rows": None,
            "model": None,
        }

    try:
        entry = engine_registry.get(emb.archive_id)
    except RegistryError:
        return {}, None, {
            "available": False,
            "reason": f"archive {emb.archive_id!r} is not a valid registered archive",
            "rows": None,
            "model": None,
        }

    manifest_path = emb.sidecar_dir / "manifest.json"
    paths_path = emb.sidecar_dir / "paths.txt"
    if not manifest_path.exists() or not paths_path.exists():
        return {}, None, {
            "available": False,
            "reason": f"no dense sidecar at {emb.sidecar_dir} (missing manifest.json/paths.txt)",
            "rows": None,
            "model": None,
        }

    try:
        digest = engine_registry.fingerprint_digest(entry.id)
        index = DenseIndex.open(emb.sidecar_dir, archive_digest=digest)
    except DenseIndexError as exc:
        return {}, None, {"available": False, "reason": str(exc), "rows": None, "model": None}

    if index.paths is None:
        return {}, None, {
            "available": False,
            "reason": f"dense sidecar at {emb.sidecar_dir} has no paths.txt",
            "rows": None,
            "model": None,
        }

    client = EmbeddingClient(emb.base_url, timeout=_EMBED_CLIENT_TIMEOUT_S)

    def embed_query(text: str) -> list[float]:
        return client.embed([text])[0]

    status = {
        "available": True,
        "reason": None,
        "rows": len(index.ids),
        "model": index.manifest.embedding_model_name,
    }
    return {entry.id: index}, embed_query, status


class _CalcTool:
    """Adapts ``tutor.tools.calc_tool.evaluate`` to the ``calc.evaluate(...)``
    interface ``tutor.app.agent_loop.run_turn`` expects."""

    def evaluate(self, expression: str) -> dict:
        return calc_tool.evaluate(expression)


class _SessionStore:
    """Minimal in-memory session store matching the routes' contract:
    ``create() -> str``, ``get(session_id) -> Session | None``, and
    ``set_subject(session_id, subject) -> bool``."""

    def __init__(self, count_tokens, *, seed_exchange: bool = False, profiles: Any = None) -> None:
        self._count_tokens = count_tokens
        self._sessions: dict[str, Session] = {}
        self._lesson_by_session: dict[str, str] = {}
        self._counter = 0
        self._lock = threading.Lock()
        # Off by default (docs/citation_experiment.md Task 3): only true
        # when ``[app] prompt_variant`` selects
        # ``tutor.app.seed_exchange.SEED_EXCHANGE_VARIANT``. A brand-new
        # lesson's session then gets the fixed synthetic seed exchange
        # before any real turn; a resumed lesson's log already carries
        # (or doesn't carry) whatever it was seeded with at creation, so
        # this flag is never applied on resume.
        self._seed_exchange = seed_exchange
        # Optional ProfileStore: when given, a lesson-attached session's
        # ``profile_summary`` is populated from the lesson's owning
        # profile (HANDOFF.md "Reading level" follow-up -- the profile's
        # grade level is meant to reach the system prompt). ``None``
        # (the default, and every plain ``create()`` session) leaves
        # ``profile_summary`` unset, matching ``Session``'s own default.
        self._profiles = profiles

    def _profile_summary_for(self, lessons: Any, lesson_id: str) -> str | None:
        if self._profiles is None:
            return None
        lesson = lessons.get_lesson(lesson_id)
        if lesson is None or not lesson.profile_id:
            return None
        profile = self._profiles.get(lesson.profile_id)
        if profile is None:
            return None
        return profile.prompt_summary()

    def create(self) -> str:
        with self._lock:
            self._counter += 1
            session_id = f"sess-{self._counter}"
            self._sessions[session_id] = Session(self._count_tokens)
        return session_id

    def _register(self, session: Session, lesson_id: str) -> str:
        with self._lock:
            self._counter += 1
            session_id = f"sess-{self._counter}"
            self._sessions[session_id] = session
            self._lesson_by_session[session_id] = lesson_id
        return session_id

    def create_for_lesson(self, lessons: Any, lesson_id: str) -> str:
        """Create a fresh session attached to ``lesson_id`` (a brand-new
        lesson, no prior turns): every completed turn on this session is
        then persisted into ``lessons`` by the turn runner."""
        session = lessons.new_session(lesson_id, count_tokens=self._count_tokens)
        session.profile_summary = self._profile_summary_for(lessons, lesson_id)
        if self._seed_exchange:
            from tutor.app.seed_exchange import seed_session

            seed_session(session)
        return self._register(session, lesson_id)

    def resume_lesson(self, lessons: Any, lesson_id: str) -> str:
        """Rebuild ``lesson_id``'s prompt log from persisted state and
        attach it to a fresh session id. Raises ``ValueError`` for an
        unknown lesson (``LessonStore.resume``'s own contract)."""
        session = lessons.resume(lesson_id, count_tokens=self._count_tokens)
        session.profile_summary = self._profile_summary_for(lessons, lesson_id)
        return self._register(session, lesson_id)

    def lesson_for(self, session_id: str) -> str | None:
        return self._lesson_by_session.get(session_id)

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


def _make_turn_runner(
    *,
    sessions: _SessionStore,
    llm,
    research_engine,
    calc,
    budget,
    lessons: Any = None,
    last_eviction: dict | None = None,
    rewrite_on_weak_evidence: bool = True,
    rewrite_on_followup: bool = True,
    reuse_prior_passages: bool = True,
    concise_followup_note: bool = False,
    restate_question_last: bool = True,
    restate_question_instruction: bool = False,
    model_writes_search: bool = True,
    model_may_skip_search: bool = True,
    no_specifics_without_source: bool = True,
    child_safe_body_topics: bool = True,
    host_topic_gate: bool = True,
    model_writes_citations: bool = False,
    clarify_unknown_words: bool = True,
):
    from tutor.app.agent_loop import run_turn

    if last_eviction is None:
        last_eviction = {}

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
            if isinstance(obj, dict) and obj.get("kind") == "status":
                # Additive turn-progress event (2026-09-20 UI follow-up):
                # never fails a turn -- emission is best-effort and purely
                # informational, older clients/eval harnesses ignore it.
                try:
                    emit("status", {"stage": obj.get("stage"), "detail": obj.get("detail")})
                except Exception:  # noqa: BLE001 - status is best-effort
                    pass
                return
            if isinstance(obj, dict) and obj.get("kind") == "eviction_reprefill":
                # Emitted synchronously by run_turn before the model call
                # that follows the eviction (see tutor.app.agent_loop and
                # tutor.app.prompt.PromptLog.evict). Forwarded verbatim
                # over SSE and remembered for /api/status.
                payload = {
                    "evicted_turns": obj.get("evicted_turns"),
                    "tokens_before": obj.get("tokens_before"),
                    "tokens_after": obj.get("tokens_after"),
                    "dropped_uncited_passages": obj.get("dropped_uncited_passages", 0),
                    "dropped_uncited_tokens": obj.get("dropped_uncited_tokens", 0),
                }
                last_eviction[session_id] = payload
                emit("eviction", payload)
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
                rewrite_on_weak_evidence=rewrite_on_weak_evidence,
                rewrite_on_followup=rewrite_on_followup,
                reuse_prior_passages=reuse_prior_passages,
                concise_followup_note=concise_followup_note,
                restate_question_last=restate_question_last,
                restate_question_instruction=restate_question_instruction,
                model_writes_search=model_writes_search,
                model_may_skip_search=model_may_skip_search,
                no_specifics_without_source=no_specifics_without_source,
                child_safe_body_topics=child_safe_body_topics,
                host_topic_gate=host_topic_gate,
                model_writes_citations=model_writes_citations,
                clarify_unknown_words=clarify_unknown_words,
            )
        except Exception:  # noqa: BLE001 - never leak a traceback to the student
            emit("error", {"message": _STUDENT_SAFE_ERROR})
            return

        citation_passage_ids: list[str] = []
        attributions_payload: dict | None = None
        if result.status == "ok" and result.route == "declined":
            # Host topic gate (docs/rewrite_on_weak_evidence.md, "Host
            # topic gate"): a fixed host-written reply, never attributed
            # or citation-checked -- it never claims a source and must
            # never be marked "not found".
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
                    "uncited": False,
                    "citation_quality": "n/a",
                    "evidence_dump": False,
                    "truncated": result.truncated,
                    "evidence": result.evidence,
                    "timings": result.timings,
                },
            )
        elif result.status == "ok":
            known_passages = session.known_passages()
            citations = resolve_citations(result.answer_text, known_passages)
            citation_passage_ids = [c.passage_id for c in citations if c.passage_id]
            unsupported_labels = [c.label for c in citations if not c.supported]
            evidence_dump = detect_evidence_dump(
                result.answer_text, citations, known_passages
            )
            if not citations:
                citation_quality = "uncited"
            elif unsupported_labels:
                citation_quality = "unsupported"
            else:
                citation_quality = "ok"
            emit(
                "citations",
                {
                    "citations": [dataclasses.asdict(c) for c in citations],
                    "unsupported_labels": unsupported_labels,
                },
            )
            # Host-side sentence-level attribution (2026-09-20 follow-up,
            # docs/attribution_design.md): a separate SSE event, sent
            # before "done", never editing the model's own answer text.
            # Failure here must never fail an otherwise-successful turn.
            try:
                emit(
                    "status",
                    {"stage": "checking", "detail": "Checking the answer against the sources..."},
                )
            except Exception:  # noqa: BLE001 - status is best-effort
                pass
            computed_items: list[dict] = []
            try:
                question_text = (
                    user_input.text if getattr(user_input, "kind", None) == "text" else None
                )
                computed_items = check_computed_statements(result.answer_text, question_text)
            except Exception:  # noqa: BLE001 - computed-check is best-effort
                _logger.exception("check_computed_statements failed for session %s", session_id)
                computed_items = []
            try:
                attribution_result = attribute_sentences(result.answer_text, known_passages)
            except Exception:  # noqa: BLE001 - attribution is best-effort
                _logger.exception("attribute_sentences failed for session %s", session_id)
            else:
                unbacked_spans = _drop_verified_number_flags(
                    attribution_result.unbacked_spans, result.answer_text, computed_items
                )
                attributions_payload = {
                    "attributions": [
                        dataclasses.asdict(a) for a in attribution_result.attributions
                    ],
                    "unbacked": [dataclasses.asdict(u) for u in unbacked_spans],
                    "computed": computed_items,
                    # Additive (2026-09-20 UI wording follow-up): lets the UI
                    # tell "the library had nothing at all for this question"
                    # apart from "some passages came back but none of them
                    # backed anything the model said" -- two different notes.
                    # Does not change citation_quality or any soak/eval metric.
                    "passages_available": len(known_passages),
                    "evidence": result.evidence,
                }
                emit("attributions", attributions_payload)
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
                    "citation_quality": citation_quality,
                    "evidence_dump": evidence_dump,
                    "truncated": result.truncated,
                    "evidence": result.evidence,
                    "timings": result.timings,
                },
            )
        elif result.status == "cancelled":
            emit("done", {"status": "cancelled"})
        else:
            emit("error", {"message": result.answer_text or _STUDENT_SAFE_ERROR})

        if lessons is not None:
            lesson_id = sessions.lesson_for(session_id)
            if lesson_id is not None:
                _persist_turn(
                    lessons,
                    lesson_id,
                    session,
                    user_input=user_input,
                    result=result,
                    citation_passage_ids=citation_passage_ids,
                    attributions=attributions_payload,
                )

    return turn_runner


def _drop_verified_number_flags(unbacked_spans, answer_text, computed_items):
    """Drop an ``"unbacked_number"`` flag for a span whose figure is the
    ``stated``/``computed`` value of a VERIFIED computed item (docs/
    calc_investigation.md fix #1): a number the host's own calc-backed
    check already confirmed is not "invented", even though no evidence
    passage contains it. Post-filter (rather than threading verified
    numbers into ``attribute_sentences`` itself) per docs/
    attribution_design.md -- keeps that function a pure, computed-check-
    agnostic sentence/passage overlap check."""
    if not computed_items:
        return unbacked_spans
    verified_values = [
        float(v)
        for item in computed_items
        if item.get("status") == "verified"
        for v in (item.get("stated"), item.get("computed"))
        if v is not None
    ]
    if not verified_values:
        return unbacked_spans

    def _matches_verified(fig: str) -> bool:
        try:
            fv = float(fig)
        except ValueError:
            return False
        return any(
            abs(fv - vv) <= max(abs(vv), 1e-9) * 1e-3 or round(fv, 3) == round(vv, 3)
            for vv in verified_values
        )

    kept = []
    for u in unbacked_spans:
        if u.reason == "unbacked_number":
            span_text = answer_text[u.span[0] : u.span[1]]
            if any(_matches_verified(fig) for fig in _figures(span_text)):
                continue
        kept.append(u)
    return kept


def _persist_turn(
    lessons,
    lesson_id,
    session,
    *,
    user_input,
    result,
    citation_passage_ids,
    attributions: dict | None = None,
) -> None:
    """Best-effort, atomic per-turn persistence for a session attached to a
    lesson (see ``routes.create_lesson``/``routes.resume_lesson``). A log
    left invalid by a crashed tool call is repaired first (mirroring
    ``LessonStore.resume``'s own repair-on-resume), so byte-identical
    resume never has to replay a dangling ``tool_calls`` entry. A subject
    change on the session (``subject != lesson.subject``) is the one
    expected failure here -- per the existing ``LessonStore`` rule it
    requires starting a new lesson, so it is swallowed rather than
    crashing an already-answered turn.
    """
    log = getattr(session, "log", None)
    if log is not None and hasattr(log, "repair") and not log.is_valid():
        log.repair()

    eviction_events = (
        [dataclasses.asdict(e) for e in result.events] if result.events else None
    )
    try:
        lessons.persist_turn(
            lesson_id,
            session,
            subject=session.subject_hint or "",
            route=result.route,
            calc_calls=result.calc_calls,
            research_calls=result.research_calls,
            citation_passage_ids=citation_passage_ids,
            tokens_used=session.log.tokens_used(),
            cached_tokens=result.cached_tokens or 0,
            user_text=user_input.text if user_input.kind == "text" else None,
            action=user_input.action if user_input.kind == "action" else None,
            eviction_events=eviction_events,
            attributions=attributions,
            truncated=result.truncated,
        )
    except ValueError:
        # Subject changed mid-lesson: the existing LessonStore rule is
        # "start a new lesson", not silently corrupt this one.
        pass


def _make_status_provider(
    *,
    llm,
    registry: Registry,
    sessions: _SessionStore,
    budget: Budget,
    model_name: str,
    resource_monitor: ResourceMonitor | None = None,
    dense_status: dict | None = None,
    last_eviction: dict | None = None,
):
    dense_status = dict(dense_status) if dense_status is not None else dict(
        _DENSE_UNAVAILABLE_STATUS
    )
    if last_eviction is None:
        last_eviction = {}

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
            "dense": dense_status,
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
            if session_id in last_eviction:
                result["last_eviction"] = dict(last_eviction[session_id])
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

    dense_status = dict(_DENSE_UNAVAILABLE_STATUS)
    if research_engine is None:
        engine_registry = _valid_registry(full_registry)
        dense_indexes, embed_query, dense_status = _open_dense(cfg, engine_registry)
        research_engine = ResearchEngine(
            engine_registry,
            snapshot_store=snapshot_store,
            cache_dir=data_dir / "research_cache",
            dense_indexes=dense_indexes,
            embed_query=embed_query,
        )

    count_tokens = _make_count_tokens(llm)
    from tutor.app.seed_exchange import SEED_EXCHANGE_VARIANT

    profiles = ProfileStore(data_dir / "profiles.sqlite")
    prompt_variant = getattr(cfg.app, "prompt_variant", "current")
    sessions = _SessionStore(
        count_tokens,
        seed_exchange=prompt_variant == SEED_EXCHANGE_VARIANT,
        profiles=profiles,
    )
    calc = _CalcTool()
    budget = Budget.for_ceiling(cfg.server.ctx_size)

    lessons = LessonStore(data_dir / "lessons.sqlite")
    turn_logger = TurnLogger(data_dir / "logs" / "turns.jsonl")
    resource_monitor = ResourceMonitor()
    last_eviction: dict[str, dict] = {}

    turn_runner = _make_turn_runner(
        sessions=sessions,
        llm=llm,
        research_engine=research_engine,
        calc=calc,
        budget=budget,
        lessons=lessons,
        last_eviction=last_eviction,
        rewrite_on_weak_evidence=getattr(cfg.app, "rewrite_on_weak_evidence", True),
        rewrite_on_followup=getattr(cfg.app, "rewrite_on_followup", True),
        reuse_prior_passages=getattr(cfg.app, "reuse_prior_passages", True),
        concise_followup_note=getattr(cfg.app, "concise_followup_note", False),
        restate_question_last=getattr(cfg.app, "restate_question_last", True),
        restate_question_instruction=getattr(cfg.app, "restate_question_instruction", False),
        model_writes_search=getattr(cfg.app, "model_writes_search", True),
        model_may_skip_search=getattr(cfg.app, "model_may_skip_search", True),
        no_specifics_without_source=getattr(cfg.app, "no_specifics_without_source", True),
        child_safe_body_topics=getattr(cfg.app, "child_safe_body_topics", True),
        host_topic_gate=getattr(cfg.app, "host_topic_gate", True),
        model_writes_citations=getattr(cfg.app, "model_writes_citations", False),
        clarify_unknown_words=getattr(cfg.app, "clarify_unknown_words", True),
    )
    status_provider = _make_status_provider(
        llm=llm,
        registry=full_registry,
        sessions=sessions,
        budget=budget,
        model_name=cfg.runtime.model_path.name,
        resource_monitor=resource_monitor,
        dense_status=dense_status,
        last_eviction=last_eviction,
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
        research_engine=research_engine,
    )
