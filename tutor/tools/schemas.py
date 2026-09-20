"""Tool schemas and argument validation for the tutor's two tools.

``research`` and ``calc`` are exposed to the model via OpenAI-style function
tool definitions (see docs/plan/offline_tutor_spec_v0.3.md §9). This module
defines those schemas and a strict validator for the JSON arguments a model
emits in a tool call, used by both the production tool-call path and the
offline eval harness (eval/toolcall_harness.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

RESEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "research",
        "description": (
            "Look up facts in the offline Kiwix archive. Use this before asserting "
            "any fact you are not certain of, so the answer can be source-backed."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language question or topic to look up.",
                },
                "keywords": {
                    "type": "array",
                    "description": "Optional extra keywords to refine the search.",
                    "items": {"type": "string"},
                },
            },
        },
    },
}

CALC_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": (
            "Evaluate an arithmetic expression in a sandbox. Use this before stating "
            "any numeric result beyond single-digit arithmetic. Also accepts a single "
            "equation form 'solve(<linear/quadratic equation in x>)', e.g. "
            "'solve(2*x + 3 = 11)' -> \"x = 4\", or 'solve(x**2 - 5*x + 6 = 0)' -> "
            "\"x = 2 or x = 3\". When a solved root is not already a plain integer or "
            "exact fraction, a decimal approximation is appended, e.g. "
            "\"x = sqrt(2) ≈ 1.41421356237\"."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["expression"],
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The arithmetic expression to evaluate, e.g. '2 + 2', or a "
                        "'solve(...)' equation in x, e.g. 'solve(2*x + 3 = 11)'."
                    ),
                },
            },
        },
    },
}

TOOLS: list[dict] = [RESEARCH_TOOL, CALC_TOOL]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    arguments: dict | None
    error: str | None


_SCHEMAS: dict[str, dict] = {
    "research": RESEARCH_TOOL["function"]["parameters"],
    "calc": CALC_TOOL["function"]["parameters"],
}


def _validate_string_field(value: object, field: str) -> str | None:
    if not isinstance(value, str):
        return f"'{field}' must be a string"
    if not value.strip():
        return f"'{field}' must not be empty or whitespace-only"
    return None


def validate_tool_call(name: str, arguments_json: str) -> ValidationResult:
    """Validate a tool call's name and JSON-encoded arguments against its schema."""
    schema = _SCHEMAS.get(name)
    if schema is None:
        return ValidationResult(ok=False, arguments=None, error=f"unknown tool: {name}")

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return ValidationResult(ok=False, arguments=None, error=f"invalid JSON: {exc}")

    if not isinstance(arguments, dict):
        return ValidationResult(ok=False, arguments=None, error="arguments must be a JSON object")

    required = schema.get("required", [])
    for field in required:
        if field not in arguments:
            error = f"missing required field: {field}"
            return ValidationResult(ok=False, arguments=None, error=error)

    allowed_keys = set(schema.get("properties", {}))
    extra_keys = set(arguments) - allowed_keys
    if extra_keys:
        return ValidationResult(
            ok=False, arguments=None, error=f"unknown extra keys: {sorted(extra_keys)}"
        )

    if name == "research":
        error = _validate_string_field(arguments["query"], "query")
        if error:
            return ValidationResult(ok=False, arguments=None, error=error)
        if "keywords" in arguments:
            keywords = arguments["keywords"]
            if not isinstance(keywords, list) or not all(
                isinstance(item, str) for item in keywords
            ):
                return ValidationResult(
                    ok=False, arguments=None, error="'keywords' must be an array of strings"
                )
    elif name == "calc":
        error = _validate_string_field(arguments["expression"], "expression")
        if error:
            return ValidationResult(ok=False, arguments=None, error=error)

    return ValidationResult(ok=True, arguments=arguments, error=None)
