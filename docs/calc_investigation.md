# Calc tool investigation: offered but declined

**Question:** does Granite skip `calc` on lesson arithmetic turns because the
app doesn't offer/encourage it, or because the model chooses not to call it?

## Findings

1. `tutor/app/agent_loop.py:272-278` always calls `llm.stream_chat(messages,
   tools=TOOLS, ...)` with the full, constant `TOOLS = [RESEARCH_TOOL,
   CALC_TOOL]` (`tutor/tools/schemas.py:79`), for every route. There is no
   separate "arithmetic route" -- a pure-arithmetic free-text question and a
   factual question needing a unit conversion both take the single
   `preretrieve` route (`agent_loop.py:213-230`): host always calls
   `research` first, then always offers both tools. Pinned by
   `tests/test_calc_tool_offering.py` (new): both turn shapes send an
   identical `tools` payload, `{"research", "calc"}`, matching `TOOLS`
   exactly.
2. No `tool_choice` is ever passed. `LlamaClient.stream_chat`
   (`tutor/app/llm_client.py:149`) has no `tool_choice` parameter at all --
   confirmed by `test_run_turn_never_passes_tool_choice`. The server sees
   implicit default tool_choice ("auto"), same as the harness.
3. Harness comparison (`eval/toolcall_harness.py`,
   `docs/bakeoff_dev/granite.md`): same `TOOLS` schema import, same
   Granite chat template (`<tools>...</tools>` XML block + `<tool_call>`
   JSON), no meaningful shape difference found. The harness's calc items
   are short, isolated single-turn prompts with no preceding
   research/evidence context; production calc turns arrive deep in an
   accumulating multi-turn lesson log with several prior evidence/answer
   turns already in context -- a plausible but unconfirmed source of the
   gap (nothing here can test in-context effects without a live call).
4. System prompt (`tutor/app/system_prompt.txt`, "Arithmetic" section) is
   explicit: never assert non-single-digit arithmetic without calling
   `calc`, and never compute a unit conversion from memory. The seed
   exchange (`tutor/app/seed_exchange.py`, off by default,
   `SEED_EXCHANGE_VARIANT`) does model one correct `calc` call in the exact
   wire tool-call format real turns use (`build_seed_entries`), but per its
   own docstring it is **not enabled** in the config used for the soaks --
   so the soak's lesson log carries no in-context calc example at all.
5. `data/granite_soak10_v3.turns.json`: every `expected_calc` turn (indices
   21, 24, 27, 30, 32) has `route == "preretrieve"` and `calc_calls == 0`.
   Scanned every turn's `answer_text` for tool-call-looking text
   (`<tool_call>`, `"name": "calc"`, `calc(`) -- **none found**; the model
   never emits a tool call as prose that the parser might drop, it simply
   never emits a tool_call event at all on these turns.
6. `docs/bakeoff_dev/granite.md` "parsed 11/20" (`granite_toolcalls.md`):
   of the harness's 20 items, 11 got `outcome: parsed_call` (a real,
   well-formed tool_call the host's parser recognized), 9 got `no_call`.
   Cross-referencing the per-item table: the 5 `research_*` and 5 `calc_*`
   items (`expect_call: True`) all show `parsed_call` (10/10), plus one
   `no_call_single_digit_sub` item unexpectedly also parsed a `calc` call
   (11th). The other 9 `no_call_*` items (greetings, thanks, opinions,
   etc., where `expect_call: False`) correctly got no tool call. So
   "11/20 parsed" is **not** "9/20 calls silently dropped by our parser" --
   it means only 11 of the 20 *items* triggered any tool call at all
   (correctly, for the 10 that should, plus 1 false positive); the
   remaining 9 correctly had no call to parse. `malformed: 0` confirms the
   parser dropped nothing. "19/20 correct decisions" = 20 minus that one
   false-positive single-digit-subtraction call.

## Conclusion

No app bug: `calc` is offered, in the correct format, with an explicit
system-prompt instruction, on both turn shapes. Per HANDOFF Task 4/analysis
item 5, this is the model declining an offered tool in a multi-turn lesson
context that the isolated harness doesn't reproduce. No product code
changed; only the pinning test (`tests/test_calc_tool_offering.py`) was
added.

## Ranked candidate fixes

1. **Host-side check: flag an answer's number disagreeing with a
   host-computed result.** Compute the expected value server-side (same
   sandbox as `calc.evaluate`) for a small set of detectable patterns
   (`N% of M`, `A + B` with fractions, `X degC to degF`) and flag/annotate
   when the model's stated figure doesn't match, without editing the
   model's text (same non-invasive posture as Task 2's attribution layer).
   Risk: low -- read-only, additive UI flag; pattern coverage is narrow and
   needs upkeep. No A/B needed (a correctness/safety net, not a wording
   change); needs a small regression fixture set of known-good/known-wrong
   arithmetic answers.
2. **Mention percentages/unit conversion explicitly in the `calc` tool's
   `description`** (currently says "non-trivial numeric result" and
   "convert between units" already, but not "percentage") and/or strengthen
   the system prompt's "Arithmetic" section with a percentage example.
   Risk: low, cheap to try, but this exact instruction already exists
   almost verbatim and the soak model still skipped it -- weak prior that
   more wording moves it. Needs an A/B (same eval harness as Task 3) since
   it changes model behavior, not host logic, and could regress latency or
   citation rate.
3. **Route arithmetic-looking turns with `tool_choice="required"` or a
   restricted tool set.** Requires adding `tool_choice` plumbing through
   `LlamaClient.stream_chat` (does not exist today) and reliable turn-shape
   detection (regex/heuristic for "N% of M", fractions, unit conversions) to
   force-route without misfiring on turns that only mention a number in
   passing. Risk: medium -- new code path, new failure mode (forcing a tool
   call on a turn that didn't need one, or misclassifying), latency cost
   from an always-required call. Needs an A/B and misclassification testing
   before adoption; highest payoff if it works, highest engineering cost.
