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
| no_call_single_digit_sub | False | None | no_call |  |
| no_call_rephrase | False | None | no_call |  |
| no_call_goodbye | False | None | no_call |  |
| no_call_encourage | False | None | no_call |  |

## Summary

- total: 20
- parsed: 10
- malformed: 0
- prose_call: 0
- no_call: 10
- errors: 0
- correct_decisions: 20
- wrong_tool: 0
- malformed_rate: 0.0000
- grammar_path_required: False

## Additional verification (measured against config/dev.ternary.toml, Ternary-Bonsai-8B Q2_0_g64)

Method follows docs/toolcall_verification.md, repeated against the Ternary server.

**`/tokenize` and `/detokenize` round-trip** (measured): `POST /tokenize` with
`{"content": "Hello world, this is a test."}` returned
`{"tokens":[9707,1879,11,419,374,264,1273,13]}` (Qwen-family vocab);
`POST /detokenize` on those tokens returned
`{"content":"Hello world, this is a test."}` — exact round-trip.

**`/apply-template`** (measured): Ternary-Bonsai's chat template is ChatML
(`<|im_start|>role\n...<|im_end|>`), same family as Bonsai. Without `tools`,
the rendered prompt is a plain system+user turn, but note it **does**
pre-emit an empty `<think>\n\n</think>\n\n` scaffold at the end of the
prompt (unlike Granite, which has no think scaffolding at all). With `tools`
passed, the system message gets a `# Tools` section appended with a
`<tools>...</tools>` XML block containing the JSON function schema and
instructions to reply with `<tool_call>{"name": ..., "arguments":
...}</tool_call>` — same mechanism as Granite/Bonsai. So `tools`
measurably changes the rendered prompt.

**Thinking off by default** (measured): a live `/v1/chat/completions` call
("What is the capital of France?", temperature 0) returned plain
`content: "The capital of France is Paris."` with no `reasoning_content`
field and no literal `<think>` tags in the returned content (the empty
think scaffold in the template resolves to nothing in the output). Thinking
is off by default for this build/model despite the empty scaffold being
present in the raw prompt.

**Prompt-cache spot check** (measured, two-call repeated-prefix request):
call 1: `prompt_tokens=32`, `cached_tokens=1`, `cache_n=1`, `prompt_n=31`
(cold cache); call 2 (identical system+user prefix): `prompt_tokens=32`,
`cached_tokens=31`, `cache_n=31`, `prompt_n=1` — full prefix reuse on the
repeated system+user turn.
