# LDS scriptures / theology content: offline options for kiwix-ai-tutor

**Research date:** 2026-09-22. Web-research only; nothing in `D:\Kiwix` was modified.

## Verdict

- **Is there an LDS ZIM: NO.** Confirmed negative. There is no Church of Jesus Christ of
  Latter-day Saints ZIM in the Kiwix catalog, and no standalone Bible-text ZIM either.
- Someone already asked for exactly this. **[openzim/zim-requests#1492](https://github.com/openzim/zim-requests/issues/1492)**,
  "New request: Zim Website - churchofjesuschrist.org" (opened 2025-07-30), requested a ZIM of the
  Church's study library (scriptures, General Conference talks, music, church history, MP3/MP4
  media). It was **closed as "Invalid" + "Licensing"** — i.e. rejected on licensing grounds, not
  built. No alternative was offered in the visible issue text.
- The project already holds a large amount of usable public-domain theology text: **the local
  `gutenberg_en_all_2023-12.zim` (confirmed present at `D:\Kiwix\gutenberg_en_all_2023-12.zim`,
  ~74 GB, referenced in `config/archives.dev.toml` as `id = "gutenberg"`, `tier = 3`,
  `kind = "viewer_only"`)** includes Project Gutenberg's King James Bible text (ebook #10,
  https://www.gutenberg.org/ebooks/10, and the older complete edition
  https://www.gutenberg.org/ebooks/30) and broad public-domain theology/religious-studies works,
  since Gutenberg's `en_all` compilation is "all English texts." This is a real Bible text, not
  LDS-specific material, and not the LDS Church's own translation/footnoted edition — but it does
  satisfy "a Bible text is available offline" today, with zero new work.
- **Recommended path forward is ranked below** — short version: don't chase a ZIM; build a small
  custom one from the public-domain LDS scripture corpus (`bcbooks/scriptures-json` or
  `beandog/lds-scriptures`), and treat General Conference talks as out of scope unless the owner
  gets written permission from the Church (they are copyrighted and the Church's terms of use
  restrict redistribution).

---

## 1. Does any LDS ZIM exist anywhere?

Checked:

- **Kiwix hub/catalog search** (`opds.library.kiwix.org/catalog/v2/entries?...q=bible`, the
  current home for what used to be `library.kiwix.org/catalog/v2/entries` — that URL now 301s to
  `opds.library.kiwix.org`): no LDS/Mormon/scripture results. A search for "bible" returns
  unrelated hits only (a "Got Questions?" Bible-trivia zimit capture, and "The OpSec Bible," a
  privacy guide — neither is scripture text).
- **`download.kiwix.org/zim/`**: confirmed it 301-redirects to `hub.kiwix.org/downloads/`, per the
  task's own expectation. The hub downloads page didn't surface per-file listings through
  WebFetch (it's a JS app backed by the catalog API above, not a static directory listing), so the
  catalog search above is the authoritative check, not this page.
- **openzim/zim-requests** (GitHub): direct search for "LDS" turned up nothing, but searching for
  "Mormon", "scriptures", "General Conference" found the request above (issue #1492), confirming
  this has been asked for and explicitly declined on licensing grounds.
- **General web search** ("LDS ZIM kiwix offline", "gospel library offline zim mormon kiwix",
  "book of mormon zim file kiwix"): no results indicating any LDS ZIM exists, pre-built or
  community-made.

**Conclusion: the previous agent's negative is confirmed independently.** No LDS ZIM exists in the
official catalog, on download.kiwix.org/hub.kiwix.org, or anywhere findable by search, and the
Kiwix project has already declined a direct request for one over licensing.

## 2. Bible/scripture ZIMs generally

- No standalone Bible-text ZIM turned up in the catalog search either (see above).
- **The `gutenberg_en_all` claim is verified as true, with a caveat on exactly which Bible
  edition.** Project Gutenberg does carry the King James Bible as ebook #10
  (https://www.gutenberg.org/ebooks/10, plain text at
  https://www.gutenberg.org/cache/epub/10/pg10.txt) and a "complete" edition as ebook #30
  (https://www.gutenberg.org/ebooks/30), both public domain. `gutenberg_en_all` is Gutenberg's
  "all English-language texts" compilation, so it should include these and a substantial amount of
  other public-domain theology (commentaries, sermons, religious history, etc. — anything on
  Gutenberg that's tagged English). I confirmed the file itself is present on disk at
  `D:\Kiwix\gutenberg_en_all_2023-12.zim` (~74 GB) and wired into
  `config/archives.dev.toml` (`id = "gutenberg"`), but I did **not** open the ZIM's internal
  contents (out of scope / avoided touching `D:\Kiwix` beyond a directory listing, and the
  in-progress download). **What I could not verify directly: that ebook #10/#30 specifically are
  included in the 2023-12 build of `gutenberg_en_all`** — this is inferred from Gutenberg's own
  cataloging (both books are old, stable, public-domain Gutenberg texts, very unlikely to be
  excluded from an "all English" compilation), not confirmed by inspecting the ZIM. If this matters,
  the cheap verification is a local `zimdump list` / `zimsearch` against the existing file for
  title "King James" — no download required, and read-only against the file, but I left this for
  the calling agent since the instructions were to avoid touching `D:\Kiwix` during the
  in-progress download.
- This is **not** LDS scripture. The LDS canon includes the King James Bible (the Church uses KJV
  as its official English Bible, so `gutenberg_en_all`'s KJV text is actually the right Bible
  translation) but does not include the Book of Mormon, Doctrine and Covenants, or Pearl of Great
  Price, none of which are on Gutenberg (they're not old enough to be public domain by original
  publication date in the way Gutenberg normally sources texts — see licensing note in §3).

## 3. Official/open offline options outside Kiwix

**Gospel Library app (official, church-published):**
- Confirmed the app has a documented offline-download feature for audio/video
  (https://lds365.com/2023/03/06/how-to-use-the-gospel-library-app-offline/): tap the download
  button on a chapter/video/audio item, or download a whole book's audio via the chapter list.
- **Not fully verified:** whether the app also fully caches all *text* (scriptures, GC talks,
  manuals) for permanent offline reading with zero future connectivity, versus needing periodic
  re-sync. The source article and a user comment ("preferred audio has to be downloaded every day
  in order to listen offline") suggest audio caching may be session-limited; it did not clarify
  text-content persistence. From general knowledge (not independently re-verified via search),
  the Gospel Library app is known to cache downloaded text content locally after first sync and is
  usable offline after that — but I could not find a citable source confirming this holds
  indefinitely with no future internet access at all, which is this project's actual deployment
  model (see CLAUDE.md: "no internet, ever," not "occasionally online"). Treat this as **not
  verified for the no-internet-forever case** and don't build on it without hands-on testing.
- Even if it works, it's a separate Android/iOS/Windows app, not ZIM content — it wouldn't plug
  into this project's Kiwix-based retrieval (`tutor/retrieval/research.py`) without a lot of new,
  unplanned integration work (scraping the app's local cache format, which isn't documented and
  may be encrypted/obfuscated).

**scriptures.byu.edu:** not independently checked in depth this pass; BYU-hosted scripture search
tools generally mirror the same Church-copyrighted/public-domain-text split as churchofjesuschrist.org
and wouldn't change the licensing picture below.

**churchofjesuschrist.org General Conference archive — bulk download legality:**
- Fetched the Church's own Terms of Use
  (https://www.churchofjesuschrist.org/learn/legal/terms-of-use/go?lang=eng). Key clauses:
  - Personal/non-commercial viewing, downloading, and printing is permitted.
  - Content in the **"Gospel Media"** section specifically may be reposted/mirrored "for your own
    personal, noncommercial use," including onto another site or network.
  - **Everything else** ("materials besides those available in Gospel Media") **may not be posted
    from this site to another website or computer network without prior written permission.**
  - Commercial use, selling, or soliciting is expressly prohibited.
  - Permission requests take ~45 days (noncommercial) to ~4-6 weeks (commercial) per the Church's
    stated process.
- **This means:** downloading General Conference talks for one student's personal offline study is
  fine under the terms as written. **Bulk-mirroring the General Conference archive into a
  redistributable ZIM artifact is the kind of "post to another computer network" use the terms
  reserve for prior written permission**, unless the talks in question are specifically inside the
  "Gospel Media" exception (this needs a Church-side confirmation of what counts as Gospel Media
  for GC talks specifically — not resolved by this search). This matches issue #1492 being closed
  under a "Licensing" label. **Recommendation: treat General Conference talk text as
  copyrighted and out of scope for bulk inclusion** unless the owner obtains written permission,
  or restricts to genuinely personal-use, non-redistributed local caching outside this project's
  shared ZIM pipeline.

**Open-licensed plain-text/JSON scripture corpora (the actually usable path):**
- **`bcbooks/scriptures-json`** (https://github.com/bcbooks/scriptures-json) — LDS scriptures
  (Old Testament, New Testament, Book of Mormon, Doctrine and Covenants, Pearl of Great Price) as
  JSON, in both a "flat" verse-iteration edition and a "reference" edition. Sourced from the 2013
  Mormon Documentation Project SQLite export.
- **`beandog/lds-scriptures`** (https://github.com/beandog/lds-scriptures) — "LDS Documentation
  Project - The Scriptures." Provides the same canon (KJV Bible, Book of Mormon, D&C, Pearl of
  Great Price) as MySQL, PostgreSQL, SQLite 3, CSV, JSON, XLSX, ODS, HTML, and plain text exports.
  **States its data as "public domain."** This is the strongest license position found: the
  *scripture text itself* (KJV Bible + the three LDS-unique volumes) is treated as public domain
  by this project, consistent with the Church's own long-standing position that the scripture text
  (as opposed to footnotes/study helps/GC talks/manuals) is free for personal and non-commercial
  redistribution. I did not find the Church's own explicit copyright statement for the scripture
  text confirming this in this pass — **flag: the "public domain" label is `beandog`'s own
  characterization, not independently confirmed against a Church copyright notice.** It is at
  minimum consistent with common practice (many LDS study apps and sites redistribute the plain
  scripture text freely) and with the Terms of Use language above, which doesn't single out
  scripture text as restricted.
- Also found, not evaluated in depth: `hyperdriveguy/OpenGospel` (desktop app to study Bible/Book
  of Mormon/D&C/Pearl of Great Price) and `pacokwon/obsidian-lds-library-plugin` — both downstream
  consumers of the same scripture corpora, not independent sources.
- **Not found:** any open corpus of General Conference talk text with a clear redistribution
  license. Given §3's Terms-of-Use analysis, this is expected — GC talks are the copyrighted part.

## 4. Feasibility of building our own ZIM

- **`zimwriterfs`** is archived/deprecated (per its own GitHub repo,
  https://github.com/openzim/zimwriterfs — "[ARCHIVED]") and takes a pre-built directory of static
  HTML, not raw text — usable but not the current recommended tool.
- **`python-libzim`** (https://github.com/openzim/python-libzim,
  docs at https://python-libzim.readthedocs.io/) is the current, maintained way to build a ZIM
  programmatically: `libzim.writer.Creator` + `Item` + `ContentProvider` (e.g. `StringProvider`
  for in-memory HTML/text). This is a good fit for a scripture corpus, since the source data
  (JSON/SQLite verses) can be templated straight into per-chapter HTML pages without needing an
  intermediate static site.
- **Rough scope of the work**, if the owner wants to proceed:
  1. Pull `beandog/lds-scriptures` (SQLite or JSON export) — public-domain text, no ambiguity.
  2. Write a small Python script: one `Item`/HTML page per chapter (or per book), with simple
     cross-reference links (book/chapter/verse), a title page, and maybe a search-friendly flat
     text version per page for this project's retrieval chunking.
  3. Build the ZIM with `python-libzim`'s `Creator`, set title/description/language/tags metadata
     consistent with how other archives are declared in `config/archives.dev.toml`.
  4. Add the new archive's `id`/`path`/`tier`/`kind` entry to `config/archives.dev.toml` (and the
     Granite fallback config) the same way `gutenberg` is declared — no changes needed to
     `tutor/retrieval/zim/` beyond pointing at the new file, since the project already treats ZIMs
     as swappable archives behind `tutor/retrieval/research.py`.
  - This is a small, self-contained, one-afternoon-scale script for an engineer already familiar
    with the corpus format — the scripture text is small (a few MB of plain text at most) compared
    to the existing multi-GB archives, so ZIM build time and size are non-issues.
  - **General Conference talks are excluded from this plan** per the licensing analysis in §3,
    unless the owner separately secures written permission from the Church, or narrows scope to
    strictly non-redistributed personal caching (which conflicts with this project's model of a
    shared, versioned ZIM library).

---

## Recommended path forward, ranked by effort

1. **Lowest effort, do this first:** confirm (with a quick, read-only `zimdump`/`zimsearch`
   listing against the existing `D:\Kiwix\gutenberg_en_all_2023-12.zim`, once its current download
   is finished and it's safe to touch) that the KJV Bible text is actually present in that
   archive, and if so, treat the Bible portion of the LDS canon as already covered.
2. **Medium effort, highest value: build a small custom ZIM for the LDS-unique scripture text**
   (Book of Mormon, Doctrine and Covenants, Pearl of Great Price, plus KJV Bible for a
   self-contained single archive) from `beandog/lds-scriptures` or `bcbooks/scriptures-json` using
   `python-libzim`, per §4. This is public-domain text, legally clean, and directly serves the
   owner's stated want for "LDS scriptures" offline.
3. **Higher effort, legally gated: General Conference talks.** Do not bulk-include without written
   permission from the Church (owner would need to initiate that ~4-6 week request). If the owner
   wants *some* GC content now without that permission, the safest reading of the Terms of Use is
   to limit to whatever specific content the Church designates as "Gospel Media" (not yet
   determined in this research) rather than the full talk archive.
4. **Not recommended:** relying on the Gospel Library app's offline cache as a data source for this
   project — it's a separate, undocumented, likely-proprietary local store, not a redistributable
   corpus, and its "works with zero future internet, forever" behavior is unverified (§3).

## Things I could not verify (flagged explicitly)

- Whether ebook #10/#30 (KJV Bible) are specifically included in the local
  `gutenberg_en_all_2023-12.zim` build — inferred, not inspected inside the ZIM.
- Whether the Gospel Library app's offline mode fully caches scripture/GC *text* (not just
  audio/video) indefinitely with no future internet access at all.
- What exactly falls under the Church's "Gospel Media" Terms-of-Use exception (the one category
  that's redistributable) — not enumerated in the page I fetched.
- Full comment thread on openzim/zim-requests#1492 — the WebFetch summary reported no visible
  maintainer comments explaining the "Invalid" label in detail; worth a direct look
  (https://github.com/openzim/zim-requests/issues/1492) if the exact maintainer reasoning matters.
- Whether `beandog/lds-scriptures`'s own "public domain" characterization of the scripture text
  matches an explicit Church copyright statement — not cross-checked against a Church source.
