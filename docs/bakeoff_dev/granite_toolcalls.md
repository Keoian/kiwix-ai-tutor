# Tool-call harness results

| id | expect_call | expected_tool | outcome | tool_names |
|---|---|---|---|---|
| research_1 | True | research | parsed_call | research |
| research_2 | True | research | parsed_call | research |
| research_3 | True | research | parsed_call | research |
| research_4 | True | research | parsed_call | research |
| research_5 | True | research | parsed_call | research |
| calc_1 | True | calc | parsed_call | calc |
| calc_2 | True | calc | parsed_call | calc |
| calc_3 | True | calc | parsed_call | calc |
| calc_4 | True | calc | parsed_call | calc |
| calc_5 | True | calc | parsed_call | calc |
| no_call_greeting | False | None | no_call |  |
| no_call_thanks | False | None | no_call |  |
| no_call_simplify | False | None | no_call |  |
| no_call_opinion | False | None | no_call |  |
| no_call_meta | False | None | no_call |  |
| no_call_single_digit_add | False | None | no_call |  |
| no_call_single_digit_sub | False | None | parsed_call | calc |
| no_call_rephrase | False | None | no_call |  |
| no_call_goodbye | False | None | no_call |  |
| no_call_encourage | False | None | no_call |  |

## Summary

- total: 20
- parsed: 11
- malformed: 0
- prose_call: 0
- no_call: 9
- errors: 0
- correct_decisions: 19
- wrong_tool: 0
- malformed_rate: 0.0000
- grammar_path_required: False

## Additional verification (measured against config/dev.granite.toml, IBM Granite 4.0 H-Tiny Q4_K_M)

Method follows docs/toolcall_verification.md, repeated against the Granite server.

**`/tokenize` and `/detokenize` round-trip** (measured): `POST /tokenize` with
`{"content": "Hello world, this is a test."}` returned
`{"tokens":[9906,1917,11,420,374,264,1296,13]}` (different vocab from the
Bonsai/Qwen tokenizer, as expected); `POST /detokenize` on those tokens
returned `{"content":"Hello world, this is a test."}` — exact round-trip.

**`/apply-template`** (measured): Granite's chat template uses
`<|start_of_role|>...<|end_of_role|>...<|end_of_text|>` framing, not ChatML,
and there is **no** `<think>` scaffolding at all (no empty think block is
pre-emitted). Without `tools`, the rendered prompt is a plain system+user
turn. With `tools` passed, the system message is replaced with a
tool-instruction preamble ("You are a helpful assistant with access to the
following tools...") plus a `<tools>...</tools>` XML block containing the
JSON function schema(s), and instructions to reply with
`<tool_call>{"name": ..., "arguments": ...}</tool_call>`. So `tools`
measurably changes the rendered prompt, same mechanism as the previous model.

**Thinking off by default** (measured): a live `/v1/chat/completions` call
("What is the capital of France?", temperature 0) returned plain
`content: "The capital of France is Paris."` with no `reasoning_content`
field and no `<think>` tags anywhere in the response or in the rendered
template. Thinking is off by default for this build/model.

**Prompt-cache spot check** (measured, single repeated-prefix request):
`prompt_tokens=37`, `prompt_tokens_details.cached_tokens=22`,
`timings.cache_n=22`, `timings.prompt_n=15` — partial prefix reuse from the
system-prompt/template portion on a cold cache; see granite_citations.md /
the live-app test run for turn-2-vs-turn-1 cache behavior in a real
multi-turn conversation.
