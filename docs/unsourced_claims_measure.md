# Measure: how often Granite's unsourced specifics are wrong

Offline, no live LLM calls. Data: `seed_eval_run{1,2,3}.json` (`current`,
54 answers), `soak6_current.turns.json` (34, attribution precomputed),
`granite_soak10_v2.turns.json` (62; retrieval re-run offline via
`config/archives.simplewiki_only.toml`, lexical only). No answer >4000 chars.

## Sweep results
`attribute_sentences` over every answer flagged sentences with a checkable
specific (number/date/measurement or proper-noun). Raw: **39 unbacked**,
223 backed (60 sampled as control). Most were false positives: Markdown
headers (`**Krebs cycle**`), law/theorem-name restatements, and arithmetic
the model computed itself (17% of 240 = 40.8). After removing those:
**n=10 genuine unbacked specifics**, ~15 in the 60-backed sample — n is
very small, illustrative not a stable estimate.

## Grading (MEASURED via `ResearchEngine.fetch_article_text`; no
`not_in_library_article` case arose, so no MODEL-JUDGED labels needed)
Unbacked (n=10): Bastille "1789" / "14 July 1789" (French_Revolution:
confirmed, exact date present); Versailles->resentment->fascism and
Versailles->Hitler/Nazi Party (World_War_II: both confirmed, paraphrased);
Louis XVI "modernize" framing (named throughout, not contradicted);
Hawaiian hot-spot volcanism (confirmed via Mauna Loa passage). 2 more
("~34 ATP" for respiration, lithosphere/asthenosphere mechanism) were
generic textbook claims not checked against a fetched article (out of
scope here) but not contradicted by anything found.

**Result: 0/10 contradicted, 0/10 likely_false, 8/10 supported_by_article
(retrieval/chunking miss, not model error), 2/10 unverified-not-checked.**
Backed control: spot-checked ~15 genuine specifics (WWII 1939,
Rhineland/Anschluss/Poland, Mount St. Helens "May 18 1980",
Koryaksky/Kamchatka, Solar System "~4.6 billion years", Pythagorean
attribution) against backing passages — **0/15 contradicted** (false-
assurance rate measured 0 here; n tiny).

Numbers vs names: 2/10 date-led (both Bastille), 8/10 name-led. All 10
came from **cold single-turn** answers to broad history/science questions;
in-lesson unbacked hits were all arithmetic, not sourcing.

## Clearest wrong claims
**None found** among the 10 unbacked + 15 backed graded. Sample skews to
stock questions (French Revolution, WWII, volcanoes) simplewiki covers
well and Granite likely knows well — absence here isn't proof it never
fabricates elsewhere.

## Policy precision/recall (0 wrong claims present in this sample)
- **(a) mark only:** stops 0 wrong (none existed); suppresses nothing;
  recall/precision undefined (0/0).
- **(b) withhold numbers/names absent from retrieved passages:** withholds
  all 10; catches 0/0 wrong; cost = **8 correct facts suppressed**
  (Bastille dates, Versailles/Hitler chain, Louis XVI, Hawaiian volcanism)
  for a pure retrieval miss, plus 2 unverified.
- **(c) withhold only if also absent from full top article:** of the 8
  checked, all were present -> **0 suppressed**; the 2 unchecked need a
  real fetch. Far fewer false suppressions than (b), one extra fetch/claim.

**Bottom line:** zero fabricated unbacked specifics surfaced here — every
genuine miss was retrieval/chunking, not a Granite error, favoring (a)/(c)
over (b). n=10 (0 wrong) is far too small to conclude Granite doesn't
fabricate; needs a larger, more adversarial sample before trusting this.
