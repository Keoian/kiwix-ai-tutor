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
