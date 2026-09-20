"""HTTP routes for the tutor app (WP-C3).

Session lifecycle, SSE-streamed turns, the source viewer endpoint, and the
status endpoint. Mounted onto the FastAPI app built by
``tutor.app.main.create_app``.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tutor.app.source_view import build_source_view

_ALLOWED_ACTIONS = {"simpler", "deeper", "hint", "research_this"}

_SENTINEL = object()


@dataclass
class _UserInput:
    kind: str
    text: str | None = None
    action: str | None = None


def build_router(deps: Any) -> APIRouter:
    router = APIRouter()

    # session_id -> threading.Event (cancel) for the turn currently running,
    # if any. Guarded by _turns_lock for atomic check-and-set (concurrency
    # guard) and for stop() lookups.
    active_turns: dict[str, threading.Event] = {}
    turns_lock = threading.Lock()

    @router.post("/api/session")
    def create_session():
        session_id = deps.sessions.create()
        return {"session_id": session_id}

    @router.post("/api/session/{session_id}/subject")
    def set_subject(session_id: str, body: dict):
        subject = body.get("subject")
        ok = deps.sessions.set_subject(session_id, subject)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown session")
        return {"ok": True}

    @router.post("/api/session/{session_id}/turn")
    async def turn(session_id: str, request: Request):
        if deps.sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown session")

        body = await request.json()
        text = body.get("text")
        action = body.get("action")

        if text is not None and action is not None:
            raise HTTPException(status_code=422, detail="text and action are mutually exclusive")
        if text is None and action is None:
            raise HTTPException(status_code=422, detail="one of text or action is required")
        if action is not None and action not in _ALLOWED_ACTIONS:
            raise HTTPException(status_code=422, detail=f"unknown action: {action!r}")

        with turns_lock:
            if session_id in active_turns:
                raise HTTPException(status_code=409, detail="turn already in progress")
            cancel = threading.Event()
            active_turns[session_id] = cancel

        kind = "text" if text is not None else "action"
        user_input = _UserInput(kind=kind, text=text, action=action)

        frame_queue: queue.Queue = queue.Queue()

        def emit(event_name: str, data: dict) -> None:
            frame_queue.put((event_name, data))

        def run() -> None:
            try:
                deps.turn_runner(session_id, user_input, emit, cancel)
            finally:
                frame_queue.put(_SENTINEL)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def stream():
            try:
                while True:
                    item = frame_queue.get()
                    if item is _SENTINEL:
                        break
                    event_name, data = item
                    yield f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
            finally:
                with turns_lock:
                    if active_turns.get(session_id) is cancel:
                        del active_turns[session_id]

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.post("/api/session/{session_id}/stop")
    def stop(session_id: str):
        with turns_lock:
            cancel = active_turns.get(session_id)
        if cancel is not None:
            cancel.set()
        return {"ok": True}

    @router.get("/api/source/{passage_id}")
    def source(passage_id: str):
        snapshot = deps.snapshot_store.get(passage_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="unknown passage")
        view = build_source_view(snapshot)
        payload = asdict(view) if hasattr(view, "__dataclass_fields__") else dict(view.__dict__)
        payload["heading_path"] = list(view.heading_path)
        payload["highlight"] = {"start": view.highlight[0], "end": view.highlight[1]}
        return JSONResponse(payload)

    @router.get("/api/status")
    def status(session_id: str | None = None):
        return deps.status_provider(session_id=session_id)

    @router.get("/api/profiles")
    def list_profiles():
        if deps.profiles is None:
            raise HTTPException(status_code=404, detail="profiles not configured")
        return {
            "profiles": [
                {
                    "id": p.id,
                    "display_name": p.display_name,
                    "grade_level": p.grade_level,
                    "subjects": p.subjects,
                    "reading_level": p.reading_level,
                }
                for p in deps.profiles.list()
            ]
        }

    @router.post("/api/profiles")
    def create_profile(body: dict):
        if deps.profiles is None:
            raise HTTPException(status_code=404, detail="profiles not configured")
        profile_id = deps.profiles.create(
            display_name=body.get("display_name", ""),
            grade_level=body.get("grade_level", 0),
            subjects=body.get("subjects", []),
            reading_level=body.get("reading_level", ""),
            preferences=body.get("preferences"),
        )
        return {"id": profile_id}

    @router.get("/api/profiles/{profile_id}/lessons")
    def list_lessons(profile_id: str):
        if deps.lessons is None:
            raise HTTPException(status_code=404, detail="lessons not configured")
        return {
            "lessons": [
                {"id": lesson.id, "subject": lesson.subject, "ended": lesson.ended}
                for lesson in deps.lessons.list_lessons(profile_id=profile_id)
            ]
        }

    @router.post("/api/profiles/{profile_id}/lessons")
    def start_lesson(profile_id: str, body: dict):
        if deps.lessons is None:
            raise HTTPException(status_code=404, detail="lessons not configured")
        lesson_id = deps.lessons.start_lesson(
            profile_id=profile_id, subject=body.get("subject", "")
        )
        return {"id": lesson_id}

    return router
