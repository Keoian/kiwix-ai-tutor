# Out-of-library measure: does the tutor say it couldn't find it?

Measured (not inferred): 14 probes, `data/out_of_library_run.py` (copy of
`data/probe_rewrite_ab.py`, rewrite ON only), real archive, Granite on
:8080. n=14 is small; percentages are illustrative, not statistically
robust. Full records: `data/out_of_library_run.jsonl`.

## Per-question table

| id | level before→after | (a) said couldn't find | (b) unlabeled memory specifics | (c) labeled as memory? | notes |
|---|---|---|---|---|---|
| ool01 coldest survived | empty→**strong** | no | 0 (backed, but wrong fact) | n/a | Cold/Negative-temp passages describe coldest *recorded* Earth temp (Vostok, -89.2°C, 1983); answer presents this as "coldest a human survived" — wrong claim, confidently sourced |
| ool02 1994 Rose Bowl attendance | empty→**strong** | no | 0 (backed, wrong game) | n/a | rewrite found "**1998** Rose Bowl", answer states 101,219 as the **1994** figure — off-topic passage presented as the answer |
| ool03 Millbrook pop. 2023 | strong→strong | n/a (answerable) | 0 | n/a | correctly answers with 2020 census figure, caveats "as of that census" — good |
| ool04 2026 World Series | empty→weak | yes | 0 | n/a | clean decline |
| ool05 news yesterday | empty→weak | yes | 0 | offers memory but doesn't give any | good |
| ool06 Zanzibar regatta (fake) | empty→weak | yes | 0 | n/a | clean decline, no invention |
| ool07 unobtainium boiling pt (fake) | **strong**→strong | yes (fictional) | 0 | n/a | best answer: explains material is fictional, gives no fabricated number |
| ool08 Fitbit Charge 3 reset | empty→empty | yes | 0 | offers if wanted | good |
| ool09 invented homework | **strong**→strong | no | 4 (Prussia, 19th century, Pestalozzi, US) | **no** | assessor said "strong" on off-topic Guy Fieri/Invention pages, so the not-found instruction never fires; model falls back to memory and states specifics as fact |
| ool10 biggest number | **strong**→strong | partial | 0 clean specifics, but broken | n/a | answer degenerates into ~6x verbatim repetition of all 5 off-topic passages (Whole wheat flour, Biggest Loser, etc.) — separate quality bug, reported below |
| ool11 school Saturday on Mars | empty→empty | yes | 0 | n/a | good, explicit "not confident giving an answer from memory" |
| ool12 blue whale flipper hairs | **strong**→strong | no | 1 ("approximately 4 hairs") | **no** | assessor "strong" on Whale/Whaling/Hair pages; model invents a specific number with no basis and no memory label |
| ool13 grains of sand, Bondi | empty→**strong** | yes | 0 | n/a | rewrite found off-topic Sand dollar/Wood grain pages but model correctly says "unknown", doesn't use them |
| ool14 Details FC keeper (fake) | empty→**strong** | yes | 0 | n/a | rewrite found off-topic football pages, correctly says no match, no invention |

## Compliance rates (n=14)

- (a) said plainly it couldn't find it: **8/14 unambiguous yes** (57%); 1 more (ool03) didn't need to since the info was genuinely on-topic; 5 answered confidently and wrongly without saying so.
- (b) mean unlabeled-memory-specifics (numbers/dates/names) per answer: **5/14 ≈ 0.36**, concentrated in 2 answers (ool09: 4, ool12: 1).
- (c) labeled memory content as unchecked when offered: 0/2 offers actually used the label requirement correctly (ool05, ool08 offered but didn't supply; ool09/ool12 supplied without labeling).
- (d) invented an answer to a thing that does not exist (ool06, ool07, ool11, ool14): **0/4** — all four declined correctly. The instruction works when the assessor actually returns weak/empty.
- Off-topic passages presented as the answer: **3/14 clear cases** (ool02, ool10, ool12), plus ool01 borderline (right domain, wrong fact). All 4 share one root cause: `assess_evidence` returned **strong** on passages that are lexically similar but topically wrong, so `_not_found_tool_text` (tutor/app/agent_loop.py:350) is never appended to the tool result — the model sees ordinary "you have evidence" framing and treats it as license to answer.

## Worst 3 answers (quoted, ≤3 lines each)

1. ool02: "The attendance at the 1994 Rose Bowl was **101,219**. This information is supported by the source [S6]..." — S6 was the 1998 Rose Bowl.
2. ool12: "A blue whale has approximately **4 hairs** on its left flipper." — no such fact in any retrieved passage (Whale/Whaling/Hair, all off-topic).
3. ool01: "The coldest temperature a human being has ever survived is **−89.2 °C**... at Vostok Station... 21 July 1983." — that is Earth's coldest *recorded* temperature, not a human-survival record; presented with full confidence.

## What triggered good vs bad behavior

Good behavior (8/14) happened exactly when the forced rewrite ran and the
merged re-assessment still came back weak/empty, so `_not_found_tool_text`
was injected into the tool result — the model then reliably declined and,
in the 4 nonexistent-thing cases, invented nothing.

Bad behavior (ool01, 02, 09, 10, 12) happened every time `assess_evidence`
returned **strong** on lexically-close but topically wrong passages
(1998≠1994 Rose Bowl; Guy Fieri/"Invention" for "invented homework"; Whale/
Hair for whale-flipper-hair count; Whole-wheat-flour/Biggest-Loser for
"biggest number"). In all of these the not-found instruction is **never
given** — the tool result looks like ordinary strong evidence, so the
model answers plainly, sometimes from the (wrong) passages, sometimes
lapsing into unlabeled memory.

## Ranked recommended changes (expected effect)

1. **Add a softer caution even on "strong" evidence when passage titles are lexically off-topic from the question's key terms** (e.g. year/number mismatch, named-entity mismatch) — would have caught ool01/02/09/12, the majority of bad answers.
2. **Host-side post-check**: if `level_after` is anything but strong, or the "strong" passages don't share the question's proper nouns, flag/ask-for-regeneration when the finished answer contains a number, date, or name with 0 backed_sentence_rate for that span — targets ool01, ool02, ool12 directly.
3. **Extend `_not_found_tool_text` wording** to also fire (in a lighter form) when the model is about to state a number/date not present verbatim in any passage, even when overall level is "strong" — closes the ool12 gap without needing a second retrieval pass.
4. **Investigate the ool10 repetition bug** separately (not a not-found-compliance issue): the answer for "biggest number" repeats all 5 passages' text ~6x verbatim, backed count reported as 79 — looks like a decoding/dedupe bug in the citation or streaming path, worth its own repro.

n=14; measured directly from full answer text and evidence payload, not sampled or summarized by another model.
