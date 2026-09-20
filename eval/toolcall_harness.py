"""Offline eval harness for tool-call reliability against a running llama-server.

Runs a fixed script of prompts (eval/questions/toolcall_script.json) against
the chat completions endpoint, classifies each response, and produces a
markdown summary. See docs/plan/offline_tutor_spec_v0.3.md §9 for the tool
contract this exercises.

This module is not part of the shipped tutor; it is a development-time
measurement tool (plan §9, offline acceptance).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from tutor.settings import load_config
from tutor.tools.schemas import TOOLS, validate_tool_call

SYSTEM_PROMPT = (
    "You are a school tutor. Call the `research` tool to look up facts you are "
    "not certain of. Call the `calc` tool for any arithmetic beyond single-digit "
    "addition or subtraction. Otherwise, answer the student directly."
)

_MALFORMED_RATE_THRESHOLD = 0.02

_PROSE_CALL_PATTERNS = [
    re.compile(r"<tool_call>"),
    re.compile(r"```json"),
    re.compile(r'"name"\s*:\s*"(research|calc)"'),
    re.compile(r"\bresearch\s*\("),
    re.compile(r"\bcalc\s*\("),
]


class Outcome(Enum):
    PARSED_CALL = "parsed_call"
    MALFORMED_CALL = "malformed_call"
    PROSE_CALL = "prose_call"
    NO_CALL = "no_call"
    ERROR = "error"


@dataclass(frozen=True)
class ScriptedRequest:
    id: str
    prompt: str
    expect_call: bool
    expected_tool: str | None


@dataclass(frozen=True)
class RequestResult:
    request: ScriptedRequest
    outcome: Outcome
    tool_names: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True)
class Summary:
    total: int
    parsed: int
    malformed: int
    prose_call: int
    no_call: int
    errors: int
    correct_decisions: int
    wrong_tool: int
    malformed_rate: float
    grammar_path_required: bool


def load_script(path: Path) -> list[ScriptedRequest]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        ScriptedRequest(
            id=item["id"],
            prompt=item["prompt"],
            expect_call=item["expect_call"],
            expected_tool=item.get("expected_tool"),
        )
        for item in data
    ]


def _looks_like_prose_call(content: str) -> bool:
    return any(pattern.search(content) for pattern in _PROSE_CALL_PATTERNS)


def classify_response(message: dict[str, Any]) -> Outcome:
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", "")
            result = validate_tool_call(name, arguments)
            if not result.ok:
                return Outcome.MALFORMED_CALL
        return Outcome.PARSED_CALL

    content = message.get("content") or ""
    if _looks_like_prose_call(content):
        return Outcome.PROSE_CALL
    return Outcome.NO_CALL


def _tool_names(message: dict[str, Any]) -> tuple[str, ...]:
    tool_calls = message.get("tool_calls") or []
    return tuple(call.get("function", {}).get("name") for call in tool_calls)


def _build_payload(
    request: ScriptedRequest, sampling: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt},
        ],
        "tools": TOOLS,
        "stream": False,
        "max_tokens": 256,
    }
    if sampling:
        payload.update(sampling)
    return payload


def run(
    requests: list[ScriptedRequest],
    post: Callable[[dict[str, Any]], dict[str, Any]],
    sampling: dict[str, Any] | None = None,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    for request in requests:
        payload = _build_payload(request, sampling)
        try:
            response = post(payload)
            message = response["choices"][0]["message"]
        except Exception as exc:  # noqa: BLE001 - a live server can fail in any way
            results.append(
                RequestResult(request=request, outcome=Outcome.ERROR, tool_names=(), error=str(exc))
            )
            continue
        outcome = classify_response(message)
        results.append(
            RequestResult(request=request, outcome=outcome, tool_names=_tool_names(message))
        )
    return results


def summarize(results: list[RequestResult]) -> Summary:
    if not results:
        raise ValueError("cannot summarize an empty result set")

    total = len(results)
    parsed = sum(1 for r in results if r.outcome is Outcome.PARSED_CALL)
    malformed = sum(1 for r in results if r.outcome is Outcome.MALFORMED_CALL)
    prose_call = sum(1 for r in results if r.outcome is Outcome.PROSE_CALL)
    no_call = sum(1 for r in results if r.outcome is Outcome.NO_CALL)
    errors = sum(1 for r in results if r.outcome is Outcome.ERROR)

    correct_decisions = 0
    wrong_tool = 0
    for r in results:
        if r.outcome is Outcome.ERROR:
            continue
        req = r.request
        if req.expect_call:
            if r.outcome is Outcome.PARSED_CALL:
                if len(r.tool_names) == 1 and r.tool_names[0] == req.expected_tool:
                    correct_decisions += 1
                else:
                    wrong_tool += 1
        else:
            if r.outcome is Outcome.NO_CALL:
                correct_decisions += 1

    malformed_rate = (malformed + prose_call) / total
    grammar_path_required = malformed_rate > _MALFORMED_RATE_THRESHOLD

    return Summary(
        total=total,
        parsed=parsed,
        malformed=malformed,
        prose_call=prose_call,
        no_call=no_call,
        errors=errors,
        correct_decisions=correct_decisions,
        wrong_tool=wrong_tool,
        malformed_rate=malformed_rate,
        grammar_path_required=grammar_path_required,
    )


def render_markdown(summary: Summary, results: list[RequestResult]) -> str:
    lines = [
        "# Tool-call harness results",
        "",
        "| id | expect_call | expected_tool | outcome | tool_names |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.request.id} | {r.request.expect_call} | {r.request.expected_tool} "
            f"| {r.outcome.value} | {', '.join(n for n in r.tool_names if n)} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- total: {summary.total}",
        f"- parsed: {summary.parsed}",
        f"- malformed: {summary.malformed}",
        f"- prose_call: {summary.prose_call}",
        f"- no_call: {summary.no_call}",
        f"- errors: {summary.errors}",
        f"- correct_decisions: {summary.correct_decisions}",
        f"- wrong_tool: {summary.wrong_tool}",
        f"- malformed_rate: {summary.malformed_rate:.4f}",
        f"- grammar_path_required: {summary.grammar_path_required}",
    ]
    return "\n".join(lines) + "\n"


def _http_post(base_url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local dev server
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the tool-call reliability eval harness.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "questions" / "toolcall_script.json",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    requests = load_script(args.script)
    sampling = {
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "top_k": config.sampling.top_k,
    }

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        return _http_post(config.server.base_url, payload)

    results = run(requests, post, sampling)
    summary = summarize(results)
    markdown = render_markdown(summary, results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    json_out = args.out.with_suffix(".json")
    raw = [
        {
            "id": r.request.id,
            "prompt": r.request.prompt,
            "expect_call": r.request.expect_call,
            "expected_tool": r.request.expected_tool,
            "outcome": r.outcome.value,
            "tool_names": list(r.tool_names),
            "error": r.error,
        }
        for r in results
    ]
    json_out.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
