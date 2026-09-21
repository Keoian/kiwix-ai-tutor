"""Investigation repro (HANDOFF.md Task 4 / docs/soak_v3_analysis.md item 5,
docs/calc_investigation.md): pins down whether the `calc` tool is actually
offered to the LLM on the turn shapes where the Granite soak recorded
``calc_calls == 0`` for a scripted arithmetic turn.

Finding under test: ``run_turn`` (tutor/app/agent_loop.py) always passes the
same ``TOOLS = [RESEARCH_TOOL, CALC_TOOL]`` list (tutor/tools/schemas.py) to
every ``llm.stream_chat`` call regardless of route, and never passes a
``tool_choice`` at all -- ``tutor.app.llm_client.LlamaClient.stream_chat``
has no ``tool_choice`` parameter to pass one to. There is no separate
"arithmetic route": a pure-arithmetic free-text question and a factual
question needing a unit conversion both take the single ``preretrieve``
route (host always calls ``research`` first, then always offers both tools
with implicit default tool_choice). So on both turn shapes below, `calc` IS
offered in the exact same request shape -- if the soak model never calls it,
that is the model declining an offered tool, not the host withholding it.
"""

from __future__ import annotations

from tests.test_agent_loop import (
    FakeCalc,
    FakeResearchEngine,
    _budget,
    _final_answer_script,
    _mk_session,
    _UserInput,
)
from tutor.app.agent_loop import run_turn
from tutor.app.llm_client import StreamEvent
from tutor.tools.schemas import CALC_TOOL, RESEARCH_TOOL, TOOLS


class _CapturingLlmClient:
    """Like FakeLlmClient but also records the ``tools`` kwarg (and whether
    a ``tool_choice`` kwarg was ever passed at all) so this test can inspect
    the exact request payload shape sent to the model."""

    def __init__(self, script):
        self._script = list(script)
        self.tools_calls: list = []
        self.received_tool_choice = False

    def stream_chat(self, messages, *, max_tokens=None, tools=None, cancel=None,
                     temperature=None, **kwargs):
        self.tools_calls.append(tools)
        if "tool_choice" in kwargs:
            self.received_tool_choice = True
        yield from self._script
        self._script = [StreamEvent(kind="done", finish_reason="stop")]


def _run(user_text: str, answer_text: str) -> _CapturingLlmClient:
    llm = _CapturingLlmClient(_final_answer_script(answer_text))
    research = FakeResearchEngine()
    calc = FakeCalc()
    run_turn(
        _mk_session(),
        _UserInput(kind="text", text=user_text),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda evt: None,
        model_writes_search=False,
    )
    return llm


def test_calc_is_offered_on_a_pure_arithmetic_turn():
    """('a') What is 12.5% of 640? -- a scripted soak calc turn (soak v3
    turn index 32, docs/soak_v3_analysis.md item 5) that recorded
    calc_calls == 0. The request payload sent to the model still contains
    the calc tool."""
    llm = _run("What is 12.5% of 640?", "12.5% of 640 is 80.")
    assert len(llm.tools_calls) == 1
    tools = llm.tools_calls[0]
    assert tools == TOOLS
    names = {t["function"]["name"] for t in tools}
    assert names == {"research", "calc"}
    assert CALC_TOOL in tools
    assert RESEARCH_TOOL in tools


def test_calc_is_offered_on_a_factual_turn_needing_a_conversion():
    """('b') A pre-retrieved factual turn that needs a unit conversion
    (e.g. Celsius to Fahrenheit) takes the same ``preretrieve`` route as
    (a) -- same TOOLS array, same implicit tool_choice."""
    llm = _run(
        "The recipe says the oven should be 200 degrees Celsius. What is that "
        "in Fahrenheit?",
        "200C is 392F [S1].",
    )
    assert len(llm.tools_calls) == 1
    tools = llm.tools_calls[0]
    assert tools == TOOLS
    names = {t["function"]["name"] for t in tools}
    assert names == {"research", "calc"}


def test_run_turn_never_passes_tool_choice():
    """The host never constrains tool_choice at all (not even implicitly
    to "auto" as an explicit kwarg) -- confirms candidate fix #3
    (tool_choice="required" on arithmetic routing) is not yet implemented
    anywhere in this call path."""
    llm = _run("What is 12.5% of 640?", "12.5% of 640 is 80.")
    assert llm.received_tool_choice is False
