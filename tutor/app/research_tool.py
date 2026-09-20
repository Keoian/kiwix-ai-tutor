"""Thin adapter from a validated `research` tool call to rendered evidence
text: validated args -> ResearchEngine.research -> tool-result text.

This module intentionally does no validation of its own -- argument
validation is tutor.tools.schemas.validate_tool_call's job, called by the
agent loop before this adapter is ever reached (see
tutor/app/agent_loop.py).
"""

from __future__ import annotations

from tutor.app.citations import render_evidence


def _passage_to_dict(passage) -> dict:
    if isinstance(passage, dict):
        return dict(passage)
    return {
        "id": getattr(passage, "passage_id", None) or getattr(passage, "id", None),
        "label": getattr(passage, "label", None),
        "title": getattr(passage, "title", None),
        "path": getattr(passage, "path", None),
        "text": getattr(passage, "text", ""),
        "kind": getattr(passage, "kind", "article"),
    }


def run_research(research_engine, arguments: dict, *, topic_hint: str | None = None) -> str:
    """Call ``research_engine.research`` with validated ``arguments`` and
    return the rendered evidence text to hand back to the model as a tool
    result."""
    response = research_engine.research(
        arguments["query"],
        topic_hint=topic_hint,
        keywords=arguments.get("keywords"),
    )
    passages = getattr(response, "passages", None)
    if passages is None and isinstance(response, dict):
        passages = response.get("passages", [])
    packet = {"passages": [_passage_to_dict(p) for p in (passages or [])]}
    return render_evidence(packet)
