# ZIM library recommendations

Catalog snapshot: `https://library.kiwix.org/catalog/v2/entries?count=-1`, fetched 2026-09-22
(3,630 total entries; 925 in English).

## What the library already covers

`D:\Kiwix` currently holds 65 ZIMs (plus a `D:\Kiwix-2023` archive of older releases): the full
English Wikipedia (`wikipedia_en_all_maxi`, 2023-10), Wikiversity, Wikihow, an Afrikaans
Wikibooks (not English), Project Gutenberg (`gutenberg_en_all`), Khan Academy (2023-03),
Lumen Learning OER courses, opentextbooks_en_all, freeCodeCamp (JavaScript track), and roughly
50 Stack Exchange sites covering programming/sysadmin (Ask Ubuntu, Stack Overflow, Super User,
Server Fault, Unix & Linux, Software Engineering, Security, etc.), the hard sciences
(math, physics, chemistry, biology, engineering, mechanics), and a long tail of hobby sites
(3D printing, gaming, woodworking, travel, worldbuilding, etc.). AI Stack Exchange, English
Language & Usage, Linguistics, and Academia are already held.

The gaps are: no English-language dictionary or open-textbook collection at scale, no dedicated
medicine reference, no data-science/statistics/theoretical-CS Stack Exchange sites, and several
"general college knowledge" subjects (economics, philosophy, psychology, law) are entirely
unrepresented.

## Tier 1 — download first (35.55 GB)

Directly on-mission: fills the biggest real gaps (dictionary, open textbooks, medicine, the
Stack Exchange sites a ML/stats/CS-focused tutor needs, subjects children need for spelling and
vocabulary).

| Name | Title | Size GB | Articles | Why it matters |
|---|---|---:|---:|---|
| `wikibooks_en_all` (maxi) | Wikibooks | 5.75 | 118,571 | English open textbooks — the library only has an Afrikaans Wikibooks (`wikibooks_af_all`), not the English one. Directly covers "OER textbooks" from the mission statement. |
| `wiktionary_en_all` (nopic) | Wiktionary | 8.53 | 9,129,949 | Full English dictionary/thesaurus. Students have bad spelling and grammar per the mission brief; nothing in the library currently answers "what does this word mean" or "how do you spell X" directly. |
| `datascience.stackexchange.com_en_all` | Data Science Q&A | 0.25 | 66,764 | Core ML/data-science reference; the library has AI.SE but nothing on data science itself. |
| `stats.stackexchange.com_en_all` | Cross Validated | 1.52 | 338,066 | The canonical statistics Q&A site — statistics is explicitly in scope and the library has none. |
| `cs.stackexchange.com_en_all` | Computer Science Q&A | 0.26 | 84,391 | CS fundamentals/theory Q&A, distinct from the practical dev sites already held (Stack Overflow, Software Engineering). |
| `medicalsciences.stackexchange.com_en_all` | Medical Sciences Q&A | 0.06 | 15,161 | Medicine is an explicit priority subject; the library currently has zero medical Q&A content. |
| `wikipedia_en_medicine` (maxi) | WikiMed Medical Encyclopedia | 2.06 | 362,501 | Curated, illustrated medical-topic slice of Wikipedia — the standard offline medical encyclopedia used by Kiwix deployments. |
| `medlineplus.gov_en_all` | MedlinePlus | 1.81 | 17,791 | NIH consumer health reference — plain-language, trustworthy medical information, complements WikiMed's encyclopedic tone. |
| `economics.stackexchange.com_en_all` | Economics Q&A | 0.11 | 26,751 | Named subject in scope, zero current coverage. |
| `philosophy.stackexchange.com_en_all` | Philosophy Q&A | 0.20 | 42,692 | Named subject in scope, zero current coverage. |
| `psychology.stackexchange.com_en_all` | Psychology & Neuroscience Q&A | 0.06 | 15,043 | General college knowledge + genetics/medicine-adjacent (neuroscience). |
| `ell.stackexchange.com_en_all` | English Language Learners Q&A | 0.44 | 154,184 | Directly addresses the audience's stated weak spot — grammar/usage help pitched at learners, not native-speaker pedants. |
| `libretexts.org_en_math` | Mathematics LibreTexts | 0.77 | 38,289 | Open, structured college-level textbook content — closest thing in the catalog to OpenStax/MIT OCW (see "not available" note below). |
| `libretexts.org_en_phys` | Physics LibreTexts | 0.52 | 18,480 | Same — physics is explicitly in scope. |
| `libretexts.org_en_chem` | Chemistry LibreTexts | 2.03 | 89,351 | Same — chemistry is explicitly in scope. |
| `libretexts.org_en_bio` | Biology LibreTexts | 2.09 | 27,000 | Same — biology/genetics is explicitly in scope. |
| `libretexts.org_en_stats` | Statistics LibreTexts | 0.20 | 11,511 | Textbook-style companion to Cross Validated above. |
| `libretexts.org_en_human` | Humanities LibreTexts | 4.13 | 70,113 | Covers history, philosophy, languages, literature — several named subjects in one collection. |
| `libretexts.org_en_socialsci` | Social Sciences LibreTexts | 1.91 | 51,998 | Covers economics, psychology, sociology, political science as structured coursework. |
| `libretexts.org_en_geo` | Geosciences LibreTexts | 1.11 | 10,080 | Earth science / physical geography, general college knowledge. |
| `libretexts.org_en_eng` | Engineering LibreTexts | 0.63 | 19,644 | Complements the existing engineering Stack Exchange sites with textbook material. |
| `libretexts.org_en_med` | Medicine LibreTexts | 1.11 | 23,171 | Textbook-style companion to the medical references above. |

## Tier 2 — worth having (21.96 GB)

Solid, on-mission additions that are more specialized, more niche, or a bigger download than
Tier 1, but still clearly worth having.

| Name | Title | Size GB | Articles | Why it matters |
|---|---|---:|---:|---|
| `cstheory.stackexchange.com_en_all` | Theoretical Computer Science | 0.07 | 22,182 | Deeper CS theory than `cs.stackexchange`; niche but cheap. |
| `bioinformatics.stackexchange.com_en_all` | Bioinformatics Q&A | 0.05 | 10,963 | Genetics-adjacent, directly named as a priority in the brief. |
| `astronomy.stackexchange.com_en_all` | Astronomy Q&A | 0.18 | 26,773 | Physics-adjacent, general science. |
| `earthscience.stackexchange.com_en_all` | Earth Science Q&A | 0.12 | 14,035 | General science, complements Geosciences LibreTexts. |
| `space.stackexchange.com_en_all` | Space Exploration Q&A | 0.37 | 31,822 | Physics/engineering-adjacent. |
| `law.stackexchange.com_en_all` | Law Q&A | 0.17 | 51,487 | Named subject (general college knowledge), zero current coverage. |
| `politics.stackexchange.com_en_all` | Politics Q&A | 0.20 | 30,472 | General college knowledge; reasoned Q&A format rather than punditry. |
| `skeptics.stackexchange.com_en_all` | Skeptics Q&A | 0.17 | 21,778 | Evaluates claims with cited evidence — fits the "teach, don't just answer confidently" ethos. |
| `crypto.stackexchange.com_en_all` | Cryptography Q&A | 0.17 | 53,948 | CS/math intersection, not covered by the existing security-focused sites. |
| `codereview.stackexchange.com_en_all` | Code Review Q&A | 0.51 | 136,321 | Practical CS learning content, complements Stack Overflow. |
| `mathoverflow.net_en_all` | MathOverflow | 0.79 | 218,048 | Research-level math Q&A, complements the already-held `math.stackexchange.com`. |
| `languagelearning.stackexchange.com_en_all` | Language Learning Q&A | 0.01 | 3,116 | Tiny; general study-skills value for language learning. |
| `scicomp.stackexchange.com_en_all` | Computational Science Q&A | 0.09 | 20,097 | CS/math/science intersection. |
| `mattermodeling.stackexchange.com_en_all` | Matter Modeling Q&A | 0.05 | 7,585 | Chemistry/materials-adjacent, as requested. |
| `hsm.stackexchange.com_en_all` | History of Science and Math Q&A | 0.05 | 9,231 | History-of-science angle on named subjects. |
| `wikipedia_en-simple_all` (maxi) | Wikipedia in Simple English | 3.26 | 396,632 | Plain-language encyclopedia — useful fallback for a struggling reader alongside full Wikipedia. |
| `wikisource_en_all` (nopic) | Wikisource | 11.23 | 4,662,306 | Primary-source texts (historical documents, classic literature) for history/literature/languages. |
| `proofwiki.org_en_all` (maxi) | ProofWiki | 0.17 | 109,137 | Worked math proofs, complements Math.SE/MathOverflow/Mathematics LibreTexts. |
| `mdwiki_en_all` (maxi) | MDWiki Medical Encyclopedia | 2.15 | 363,797 | Independent community medical wiki, supplements WikiMed/MedlinePlus rather than duplicating them. |
| `libretexts.org_en_biz` | Business LibreTexts | 0.78 | 27,655 | Economics-adjacent college subject. |
| `libretexts.org_en_workforce` | Workforce LibreTexts | 0.48 | 9,568 | Study-skills/career-readiness adjacent. |
| `libretexts.org_en_k12` | K-12 Education LibreTexts | 0.34 | 5,083 | Age-appropriate material for the younger end of the audience. |
| `milneopentextbooks.org_en_all` | Milne Open Textbooks | 0.39 | 583 | Small open-textbook collection distinct from LibreTexts/Wikibooks. |
| `freecodecamp_en_rosetta-code` | Rosetta Code | 0.01 | 161 | CS practice; cheap add. |
| `freecodecamp_en_javascript-algorithms-and-data-structures` | FreeCodeCamp JavaScript (algorithms/DS) | 0.01 | 300 | CS practice, complements the JS track already held. |
| `freecodecamp_en_coding-interview-prep` | FreeCodeCamp Interview Prep | 0.01 | 57 | CS practice; cheap add. |
| `freecodecamp_en_all` | FreeCodeCamp | 0.01 | 996 | Remaining freeCodeCamp curriculum not in the existing JS-only ZIM. |
| `devdocs_en_python` | Python Docs | 0.004 | 497 | ML/data-science tooling reference (Python is the field's default language). |
| `devdocs_en_numpy` | NumPy Docs | 0.005 | 2,227 | ML/data-science tooling reference. |
| `devdocs_en_pandas` | pandas Docs | 0.005 | 2,177 | ML/data-science tooling reference. |
| `devdocs_en_scikit-learn` | scikit-learn Docs | 0.05 | 933 | ML-specific reference, directly named as a priority. |
| `devdocs_en_pytorch` | PyTorch Docs | 0.01 | 2,445 | ML-specific reference, directly named as a priority. |
| `devdocs_en_tensorflow` | TensorFlow Docs | 0.01 | 4,799 | ML-specific reference, directly named as a priority. |
| `devdocs_en_matplotlib` | Matplotlib Docs | 0.03 | 1,062 | Data-science tooling reference. |
| `devdocs_en_r` | R Docs | 0.005 | 2,589 | Statistics tooling reference, complements Cross Validated. |
| `devdocs_en_statsmodels` | Statsmodels Docs | 0.01 | 3,460 | Statistics/ML tooling reference. |

## Tier 3 — optional / large (34.35 GB)

Not core to the mission, or large relative to their value for this audience. Downloading them
is a judgment call, not a recommendation.

| Name | Title | Size GB | Articles | Why it's optional |
|---|---|---:|---:|---|
| `cheatography.com_en_all` | Cheatography | 10.65 | 94,088 | Large, crowdsourced cheat-sheet site of uneven and unvetted quality; poor fit for a "grounded, citable evidence" tutor. |
| `gis.stackexchange.com_en_all` | Geographic Information Systems | 1.98 | 252,544 | Professional GIS tooling, niche for this audience. |
| `dba.stackexchange.com_en_all` | Database Administrators Q&A | 0.65 | 178,114 | Professional/niche, already have Stack Overflow and Server Fault for general dev needs. |
| `tex.stackexchange.com_en_all` | TeX - LaTeX Q&A | 4.22 | 379,230 | Useful only for advanced math/science typesetting; large for the value. |
| `quant.stackexchange.com_en_all` | Quantitative Finance Q&A | 0.15 | 38,886 | Niche finance-math intersection, not core to the stated subjects. |
| `quantumcomputing.stackexchange.com_en_all` | Quantum Computing Q&A | 0.09 | 19,323 | Advanced/specialized beyond general college physics/CS. |
| `or.stackexchange.com_en_all` | Operations Research Q&A | 0.03 | 7,543 | Niche applied-math specialty. |
| `proofassistants.stackexchange.com_en_all` | Proof Assistants Q&A | 0.01 | 2,708 | Extremely niche (formal verification tooling). |
| `wikispecies_en_all` (nopic) | Wikispecies | 1.43 | 1,643,146 | Taxonomic name database, not article prose — low tutoring value per GB. |

### Religion and theology

Checked per owner request. Nothing here is core to the mission (ML/genetics/medicine/general
college knowledge); listed for completeness only, and none of it is in `zim_wanted.txt`.

| Name | Title | Size GB | Articles | Note |
|---|---|---:|---:|---|
| `gotquestions.org_en_all` | Got Questions? — Bible Questions Answered | 14.14 | 21,482 | Evangelical Christian apologetics Q&A site. Large for a single-viewpoint theological source. |
| `christianity.stackexchange.com_en_all` | Christianity Q&A | 0.18 | 30,322 | General Christianity Q&A. |
| `hermeneutics.stackexchange.com_en_all` | Biblical Hermeneutics Q&A | 0.18 | 30,046 | Biblical-text interpretation Q&A; closest thing to Bible-study material in the catalog (no actual Bible-text ZIM exists — see below). |
| `judaism.stackexchange.com_en_all` | Mi Yodeya Q&A | 0.28 | 55,114 | Judaism Q&A. |
| `islam.stackexchange.com_en_all` | Islam Q&A | 0.08 | 27,605 | Islam Q&A. |
| `hinduism.stackexchange.com_en_all` | Hinduism Q&A | 0.19 | 27,592 | Hinduism Q&A. |
| `buddhism.stackexchange.com_en_all` | Buddhism Q&A | 0.08 | 14,303 | Buddhism Q&A. |

`gutenberg_en_lcc-b` ("Project Gutenberg Library — Philosophy, psychology, religion", 5.93 GB,
6,180 books) was also found but is **not listed above** because it is a subset of the full
Project Gutenberg corpus (`gutenberg_en_all`, 80,656 books) that the library already holds in
full — downloading it would duplicate books already on disk.

## Already covered, skipped

- Full English Wikipedia, Wikiversity, Wikihow, Project Gutenberg, Khan Academy, Lumen Learning,
  `opentextbooks_en_all`, and ~50 Stack Exchange sites (programming, sysadmin, hard sciences,
  hobbies) are already held — not re-listed even though the catalog offers newer releases of
  several of them (that's an update, handled by `scripts/kiwix_update_zims.py`, not a new
  acquisition).
- `wikibooks_af_all` is held, but that is **Afrikaans** Wikibooks, not English — the English
  `wikibooks_en_all` above is genuinely new.
- `writers.stackexchange.com_en_all` (held, 2023-11) no longer appears in the catalog under that
  name; the site appears to have been renamed to `writing.stackexchange.com` at some point after
  2023. Not a new-ZIM recommendation, but worth flagging since the existing updater script
  resolves basenames by exact name match and may report "no release found" for it.
- All religion/theology candidates above duplicate nothing already held (the library holds no
  religion-specific content at all before this search).

## Not available in the catalog

- **No LDS / Church of Jesus Christ of Latter-day Saints content of any kind.** Searched both the
  `name` field and the `title`/`summary` fields for: `lds`, `mormon`, `latter-day` / `latter day`,
  `churchofjesuschrist`, `gospel`, `conference` (talks), `come follow me`, `book of mormon`,
  `doctrine and covenants`, `pearl of great price`, `joseph smith`, `temple`, `missionary` — zero
  matches across all 3,630 entries in any language. There is no `lds.org` or
  `churchofjesuschrist.org` scrape, no Gospel Library content, and no LDS scripture text in the
  Kiwix catalog. The closest available alternatives, if any religious reference is wanted, are the
  six religion Stack Exchange sites and `gotquestions.org_en_all` listed under "Religion and
  theology" above — none of them are LDS-specific.
- **No standalone Bible-text ZIM** (any translation) and **no Quran-text ZIM** — searched `bible`,
  `scripture`, `quran`/`qur'an`. `hermeneutics.stackexchange.com` (Biblical interpretation Q&A)
  and `gotquestions.org` (Christian apologetics) are the closest things to Bible-adjacent content.
- **No dedicated "encyclopedia of religion" or Wikipedia religion-subset ZIM** — unlike
  `wikipedia_en_medicine`, `wikipedia_en_physics`, `wikipedia_en_chemistry`, etc., there is no
  `wikipedia_en_religion`.
- **No MIT OpenCourseWare and no OpenStax collection** — searched `openstax`, `opencourseware`,
  `ocw`, `mit.edu`. LibreTexts (Tier 1/2 above) is the closest equivalent the catalog actually
  offers: openly-licensed, subject-organized college textbook content.
- **No Khan Academy variant beyond the single `khanacademy_en_all` already held** — the catalog's
  current `khanacademy_en_all` is dramatically larger (167.6 GB) than what's on disk (2023-03),
  which is an update question, not a new-acquisition one.
