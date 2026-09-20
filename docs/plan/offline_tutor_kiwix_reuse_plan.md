# Offline Tutor — Kiwix / ZIM Reuse Plan for Claude Code

**Purpose:** Implementation handoff for the offline school tutor.  
**Goal:** Reuse mature Kiwix/ZIM retrieval code and proven retrieval ideas instead of rebuilding archive access, search, parsing, citation, and failure-handling from scratch.

This document assumes the tutor architecture in `offline_tutor_spec_v0.3.md`: local `llama-server`, a Python tutor app, automatic pre-retrieval on each factual question, at most one model-requested follow-up retrieval, deterministic host-side routing where possible, and a local calculator tool.

---

# 1. Executive direction

Do **not** base the tutor on `jeffreyrampineda/kiwix-wiki-mcp-server`. It is useful as a tiny MCP/tool-shape reference, but its implementation is intentionally simple: it talks to `kiwix-serve` over HTTP, parses Kiwix search HTML, and strips article HTML with regex/basic tag removal.

Instead:

1. **Use `cameronrye/openzim-mcp` as the primary implementation donor.**
   - It already solves most of the hard ZIM-specific plumbing we need.
   - It uses native `libzim`.
   - It has robust archive validation, search, redirect resolution, caching, article structure extraction, section offsets, citations, result budgeting, and multi-archive behavior.
   - It is MIT licensed.
   - We should vendor/adapt the useful internal modules into the tutor instead of running its full MCP stack in production.

2. **Use `OscSanto/Zim-Indexer` as a design donor for hybrid retrieval.**
   - Its retrieval architecture is extremely close to our planned Simple-Wikipedia index.
   - It has good ideas for structured chunking, BM25 + dense retrieval, RRF, diversity caps, mention penalties, and title/navigation boosts.
   - **Do not copy source directly unless licensing is clarified.** No license file was found during review. Reimplement these algorithms independently from the described behavior.

3. **Use smaller Kiwix LLM/MCP repos only for narrow ideas.**
   - `mozanunal/llm-tools-kiwix`: useful search-and-collect abstraction, archive discovery.
   - `yitch/kiwix-llm`: useful ingestion/checkpoint/index bookkeeping ideas.
   - `gulamsabri/kiwix-rag`: useful resumable batch-indexing and collection-routing ideas.
   - `OscillateLabsLLC/kiwix-mcp`: useful Kiwix HTTP compatibility knowledge if we ever need `kiwix-serve`, but native `libzim` remains preferred.
   - `scottyphillips/kiwix-mcp`: useful token-efficient "search with snippets before full fetch" pattern.
   - `ThinkInAI-Hackathon/zim-mcp-server`: simple native-ZIM MCP reference; less mature than OpenZIM MCP.

The intended end-state remains the v0.3 architecture: **the tutor app owns retrieval in-process**, with a long-lived libzim worker where needed. MCP remains development-only.

---

# 2. Primary donor: `cameronrye/openzim-mcp`

Repository:

`https://github.com/cameronrye/openzim-mcp`

Reviewed release: **v3.3.4**  
License: **MIT**

This is the most important repo. Treat it as the source of truth for mature ZIM handling.

## 2.1 Archive opening, validation, and health checks

### Source

`openzim_mcp/zim/archive.py`

### Reuse/adapt

Port or adapt the following concepts and implementation:

- `ZIM_MAGIC`
- ZIM header probing
- `_declared_zim_size(...)`
- `has_zim_signature(...)`
- `is_truncated_zim(...)`
- `zim_signature_error(...)`
- `unreadable_zim_warning(...)`
- archive integrity/check handling
- archive open timeout behavior
- normalized exceptions around bad/corrupt/unreadable archives
- libzim cache tuning hooks
- safe archive metadata/introspection logic

### Why we need it

The tutor installer/startup must be able to distinguish:

- valid ZIM
- corrupted ZIM
- partially downloaded/truncated ZIM
- wrong file renamed `.zim`
- permission failure
- archive that opens but lacks a full-text index

Do not reduce all failures to "archive unavailable".

### Tutor integration

Create something like:

```text
tutor/retrieval/zim/archive.py
```

with:

- `validate_archive(path)`
- `open_archive(path)`
- `archive_capabilities(path)`
- `archive_fingerprint(path)`
- `configure_libzim_cache(...)`

Return structured status, not exceptions, at the tutor boundary.

### Important adaptation

We do **not** need OpenZIM MCP's full server configuration, HTTP transport, MCP resource system, subscriptions, or rate limiter.

Keep only the archive/data-access layer.

---

## 2.2 Redirect handling and path resolution

### Sources

- `openzim_mcp/zim/redirects.py`
- `openzim_mcp/zim/content.py`
- smart-retrieval behavior documented in:
  - `website/src/content/docs/smart-retrieval.mdx`

### Reuse/adapt

Port:

- bounded redirect-chain resolution
- redirect cycle detection
- maximum redirect depth
- normalized path handling
- URL-decoding / alternate spelling probes
- path-mapping cache keyed by archive identity
- search fallback for guessed paths
- storing resolved canonical paths

The OpenZIM MCP fallback sequence is worth preserving conceptually:

1. exact lookup
2. alternate spelling probes
3. redirect following
4. cached mapping
5. search-derived resolution when direct access fails

### Why we need it

Kiwix/ZIM article paths are not uniform across archive families and versions. The tutor should never rely on "Wikipedia article X must be at `/A/X`".

This also makes citation IDs stable because citations should use the resolved canonical path, not a guessed path.

### Tutor integration

Add:

```text
tutor/retrieval/zim/resolve.py
```

Possible API:

```python
resolve_entry(archive, requested_path) -> ResolvedEntry
```

where `ResolvedEntry` contains:

- requested path
- canonical path
- redirect chain
- title
- content type

Cache the mapping using the archive fingerprint so archive replacement automatically invalidates stale path mappings.

---

## 2.3 Native full-text and title search

### Sources

- `openzim_mcp/zim/search.py`
- `openzim_mcp/zim/archive.py`
- `libzim.search.Searcher`
- `libzim.search.Query`
- `libzim.suggestion.SuggestionSearcher`

### Reuse/adapt

Port the native libzim search approach, including:

- full-text search
- title-first search
- suggestions/title lookup
- snippets
- result limits
- deterministic ordering
- namespace/content filtering where useful
- cross-archive result handling
- failure behavior when an archive has no full-text index

Look specifically for the logic used by:

- `search_top_k(...)`
- title matching helpers
- suggestion search
- snippet generation
- search-result normalization

### Why we need it

Our v0.3 spec currently says the retrieval layer will use Xapian through libzim. OpenZIM MCP already implements that correctly.

Do **not** reimplement Xapian query parsing ourselves unless an actual retrieval test proves OpenZIM MCP's search surface is insufficient.

### Tutor integration

Create:

```text
tutor/retrieval/zim/search.py
```

with a deliberately small API:

```python
search_fulltext(archive, query, limit)
search_title(archive, query, limit)
get_search_snippet(hit)
```

The tutor's higher-level retrieval pipeline decides how to fuse/rank them.

---

## 2.4 Search result snippets

### Sources

- `openzim_mcp/zim/search.py`
- `openzim_mcp/synthesize.py`

### Reuse/adapt

Preserve the concept of using libzim/search snippets as an inexpensive first relevance signal.

OpenZIM MCP's synthesize path takes existing search snippets and converts them into candidate passages before doing section attribution.

### Why we need it

For many direct school questions, we can avoid opening and fully parsing ten articles.

A cheap flow is:

```text
Xapian result
→ snippet
→ relevance filtering
→ open/parse only the top few articles
```

This supports our latency target on the i5-7300HQ.

---

# 3. Article parsing and structural extraction

## 3.1 `EntryBundle` concept — high priority

### Source

`openzim_mcp/bundle.py`

### Reuse/adapt

This is one of the most valuable pieces in the entire repo.

Port the `EntryBundle` idea:

A single parse of an HTML article should produce:

- canonical entry path
- title
- MIME/content type
- rendered plain text or Markdown
- word count
- character count
- headings/sections
- section IDs
- section parent relationships
- section character start/end offsets
- outbound/internal/media links if desired
- infobox data if desired

Important functions/concepts:

- `extract_entry_bundle(...)`
- `get_or_build_bundle(...)`
- `_compute_section_offsets(...)`
- `_locate_heading_text(...)`
- heading normalization / decorated-heading fallback
- stable unique section IDs
- parent section tracking
- cacheable bundle result

### Why we need it

Our tutor spec requires:

- exact evidence passages
- citation IDs
- source viewer
- highlighting the cited sentence
- heading-aware passage grouping
- saved character offsets

The `EntryBundle` structure gives us the correct intermediate representation for all of that.

### Tutor integration

Create:

```text
tutor/retrieval/zim/bundle.py
```

Define our own dataclasses/Pydantic models, but preserve the semantics:

```python
ArticleBundle
SectionMeta
LinkBuckets
InfoboxData
```

The rest of the tutor retrieval pipeline should consume `ArticleBundle`, not raw HTML.

---

## 3.2 Main-content selection and HTML cleanup

### Sources

- `openzim_mcp/content_processor.py`
- `openzim_mcp/bundle.py`

### Reuse/adapt

Port or adapt:

- `select_main_content(...)`
- script/style/navigation removal
- main-content landmark detection
- HTML → readable text/Markdown rendering
- oversized table handling
- infobox extraction
- link extraction
- archive-type-specific cleanup if it is not overly coupled to the MCP app

Do not use naive:

```python
BeautifulSoup(html).get_text()
```

for production retrieval.

### Why we need it

Wikipedia and other ZIM archives contain:

- navigation chrome
- reference sections
- sidebars
- infoboxes
- tables
- edit UI remnants
- archive-family-specific structure

Poor cleaning poisons both lexical retrieval and embeddings.

---

## 3.3 Section-aware citation attribution

### Source

`openzim_mcp/synthesize.py`

### Reuse/adapt

High-value functions/concepts:

- `_extract_passages(...)`
- `_locate_passage(...)`
- `_normalize_ws(...)`
- bold-marker stripping/remapping
- `_attribute_sections(...)`
- deepest/smallest containing section wins
- fallback to article-level citation when exact section attribution fails

OpenZIM MCP accounts for subtle differences between search-snippet rendering and whole-article rendering. Preserve that robustness.

### Why we need it

Our citation should ideally be:

```text
archive + canonical article path + section + passage offset
```

rather than merely:

```text
Wikipedia: Photosynthesis
```

This is essential for the source viewer and "highlight exact cited sentence" behavior.

---

# 4. Multi-source fusion and result budgeting

## 4.1 Reciprocal Rank Fusion (RRF)

### Source

`openzim_mcp/synthesize.py`

### Reuse directly/adapt

Function:

```python
_rrf_fuse(...)
```

Behavior:

```text
score(document) = Σ 1 / (k + rank)
```

with `k = 60`.

Also preserve deterministic tie-breaking.

### Where we use it

Our retrieval has multiple candidate sources:

- Xapian full-text
- title search
- Simple-Wikipedia dense index
- possibly multiple archives

Fuse rank positions instead of trying to compare unrelated raw scores.

### Tutor implementation

Use our own public name, e.g.:

```python
rrf_fuse(rankings, k=60)
```

This is simple enough that we can test exhaustively.

---

## 4.2 Budget enforcement

### Sources

- `openzim_mcp/synthesize.py`
- `openzim_mcp/tool_schemas.py`

### Reuse/adapt

Steal the concept that retrieval itself owns the evidence budget.

The LLM should not receive arbitrarily long article bodies.

Our packer should enforce:

- token budget
- max passages
- max passages/article
- sentence-boundary trimming
- deterministic ordering
- source metadata preserved after trimming
- explicit "truncated / limit hit" status

This aligns with the v0.3 `evidence_budget_tokens` contract.

---

## 4.3 Partial-success semantics

### Sources

Across OpenZIM MCP, especially:

- `openzim_mcp/synthesize.py`
- archive/search/structure modules
- tool error response patterns

### Reuse/adapt

The retrieval system should be allowed to return:

```text
partial success
```

instead of converting one failed archive/read into total failure.

Our response should explicitly record:

- archives searched
- archives failed
- limits hit
- timeout
- partial extraction
- missing index
- fallback route used

This maps directly to the tutor spec's `status`, `reason_codes`, `coverage`, and `followup_recommended` fields.

---

# 5. Caching ideas to reuse

## 5.1 Article bundle cache

### Source

`openzim_mcp/bundle.py`

Cache parsed `ArticleBundle`s.

Key should include:

- archive identity/fingerprint
- canonical entry path
- extractor/render version
- compact/full render mode if both exist

---

## 5.2 Path-resolution cache

### Source

OpenZIM MCP smart retrieval.

Cache:

```text
requested path → resolved canonical path
```

Keyed by archive fingerprint.

---

## 5.3 Search-result/snippet cache

### Source

OpenZIM MCP cache architecture.

Cache hot queries and article render fragments.

For the single-student tutor this can stay small.

---

## 5.4 Archive replacement invalidation

### Source

OpenZIM MCP cache key strategy.

Use:

- file size
- modified timestamp
- archive edition metadata
- extractor version

Prefer a stable explicit `archive_fingerprint` stored in our own metadata DB.

When the archive changes:

- invalidate path mappings
- invalidate article bundles
- invalidate passage cache
- rebuild embedding sidecar when applicable

This directly supports the v0.3 archive edition/fingerprint requirement.

---

# 6. Libzim process isolation

### Source ideas

- `openzim_mcp/zim/archive.py`
- our existing v0.3 architecture

OpenZIM MCP uses separate processes for operations that can block the Python event loop/GIL, including integrity checking.

Our design already proposes a long-lived libzim child worker.

### Reuse/adapt

Implement a worker protocol around:

- open archive
- search
- read entry
- resolve redirect
- basic metadata

The parent owns:

- deadline
- hard kill
- restart
- request ID
- stale-result rejection

### Important

Do **not** put dense embeddings or the LLM in this worker.

Keep the worker narrow and disposable.

---

# 7. Secondary design donor: `OscSanto/Zim-Indexer`

Repository:

`https://github.com/OscSanto/Zim-Indexer`

**Licensing status:** no `LICENSE` file was found during review.  
**Rule:** reimplement the ideas below; do not copy source verbatim unless licensing is clarified.

This repo is especially useful for the Simple-Wikipedia dense sidecar that OpenZIM MCP intentionally does not currently provide as its default architecture.

---

## 7.1 Structure-aware article extraction

### Source

`indexer/extract.py`

### Ideas to reimplement

Functions/concepts:

- `_clean_text(...)`
- `_split_sentences(...)`
- `_semantic_units(...)`
- `_extract_infobox(...)`
- `extract(...)`

Useful behavior:

- separate article lead from sections
- keep section boundaries
- preserve h2/h3 hierarchy
- turn long paragraphs into sentence-bounded semantic units
- merge very short adjacent units
- skip junk sections such as references/external links
- treat infobox as structured facts rather than flattening it into prose

### How it fits our design

Our current spec says passages should be 80–180 words and heading-aware.

Use OpenZIM MCP's `ArticleBundle` as the source representation, then apply a Zim-Indexer-inspired chunker to the rendered section bodies.

Do not maintain two unrelated HTML parsers if avoidable.

Preferred pipeline:

```text
OpenZIM ArticleBundle
→ section bodies
→ our semantic-unit chunker
→ passages
```

---

## 7.2 Dense + lexical fusion

### Source

`indexer/query.py`

### Ideas to reimplement

Use three independent rankings:

1. dense semantic search
2. article-title BM25
3. passage/body BM25

Then combine with RRF.

This is better than blending raw cosine score and BM25 score directly.

### Tutor adaptation

For Simple Wikipedia:

- dense: article-level or passage-level embedding sidecar
- lexical: native Xapian full-text
- title: libzim title/suggestion result

Potential first version:

```text
Dense article top 16
Xapian top 16
Title top 8
→ RRF
→ open top 6 articles
→ extract passages
→ passage BM25
→ pack evidence
```

This is close to v0.3 but removes unnecessary custom indexing where libzim already provides a good index.

---

## 7.3 Mention-strength penalty

### Source

`indexer/query.py` → `_mention_strength(...)`

### Reimplement conceptually

Problem:

A query term may occur once incidentally near the end of a chunk, making the chunk look relevant lexically even though the passage is not actually about the topic.

Possible scoring adjustment:

- term appears once → mild penalty
- term appears only late in passage → mild penalty
- repeated / early mention → no penalty

### Use carefully

Keep this behind an evaluation flag.

Do not assume it improves school QA until measured on the held-out set.

---

## 7.4 Per-article diversity cap

### Source

`indexer/query.py` → `_diversity_cap(...)`

### Reimplement

Before final evidence packing:

```text
max 2 passages/article by default
```

unless additional passages cover missing facets.

This already matches the intent in v0.3 and prevents one broad Wikipedia article from monopolizing the entire evidence packet.

---

## 7.5 Navigational/title boost

### Source

`indexer/query.py` → `_nav_boost(...)`

### Reimplement

If a query strongly matches an article title, promote its lead/primary passage.

Example:

```text
"What is photosynthesis?"
```

should strongly favor:

```text
Photosynthesis
```

over a page that happens to mention photosynthesis twenty times.

This is especially useful for school questions.

Again: evaluation-gated, not hard-coded dogma.

---

## 7.6 Lead augmentation / section augmentation

### Source

`indexer/query.py`

### Reimplement selectively

When a passage from deep inside an article is selected, optionally attach:

- article lead
- first paragraph of the section

This can provide missing referents/context.

For our tutor, use only if the passage would otherwise be ambiguous.

Do not blindly prepend text because our context budget is tight.

---

# 8. SQLite / FTS ideas from Zim-Indexer

### Source

`indexer/db.py`

### Ideas worth reimplementing

- SQLite WAL mode
- persistent metadata DB
- contentless FTS5 tables
- title FTS
- passage FTS
- incremental indexing metadata
- row IDs stable across restarts
- embedding-state flags
- article/chunk relationship tables

### But simplify

We already have Xapian inside the ZIM.

Do **not** duplicate full Wikipedia into a second giant SQLite text index unless evaluation proves we need it.

Use SQLite primarily for:

- Simple-Wikipedia embedding metadata
- article IDs / vector IDs
- archive fingerprints
- citation snapshots
- cache metadata
- student/lesson state

---

# 9. Small donor: `mozanunal/llm-tools-kiwix`

Repository:

`https://github.com/mozanunal/llm-tools-kiwix`

License: **Apache 2.0**

## Reuse ideas

### 9.1 Automatic archive discovery

Scan configured folders and register discovered `.zim` files.

Our installer can then validate and register them automatically.

### 9.2 `search_and_collect` abstraction

The repo exposes a combined:

```text
search → fetch matching article content
```

operation.

That reinforces our host-side retrieval design.

For Bonsai, do not expose ten low-level tools if the host can deterministically do the work.

### Tutor adaptation

Production model tools remain:

- `research(query, keywords[])`
- `calc(expression)`

Everything else stays internal.

---

# 10. Small donor: `yitch/kiwix-llm`

Repository:

`https://github.com/yitch/kiwix-llm`

License: **MIT**

## Reuse ideas

- ingestion manifest
- persistent index bookkeeping
- model/index version tracking
- source metadata attached to vectors
- local-only citation-bearing RAG
- optional local Kiwix browser/source viewer
- restartable service layout

### Do not adopt

- ChromaDB as a requirement
- Ollama dependency
- re-extracting all archives into a second corpus
- Mac-specific runtime architecture

Our Simple-Wikipedia sidecar should remain much smaller and simpler.

---

# 11. Small donor: `gulamsabri/kiwix-rag`

Repository:

`https://github.com/gulamsabri/kiwix-rag`

Licensing was not confirmed during review; treat as a design reference unless verified.

## Reuse ideas

### Resumable batch indexing

Embedding/index generation should:

- checkpoint progress
- resume after interruption
- record archive fingerprint
- record embedding model version
- record completed article ID/range
- atomically mark index complete

This matters on the slow Dell.

### Collection/subject routing

The project routes searches toward relevant content groups.

We do **not** need this for Simple Wikipedia initially, but it could become useful later when adding:

- Wikibooks
- Wikipedia for Schools
- medicine/science-specific archives
- Stack Exchange education/technical archives

Do not add this complexity in v1 unless multiple archive families become a demonstrated retrieval problem.

---

# 12. Kiwix HTTP compatibility references

These are lower priority because we prefer native libzim.

---

## 12.1 `OscillateLabsLLC/kiwix-mcp`

Repository:

`https://github.com/OscillateLabsLLC/kiwix-mcp`

Useful if we ever support external `kiwix-serve`.

Documented Kiwix HTTP quirks include:

- OPDS catalog parsing
- full-text search HTML, not JSON
- book scoping
- path-prefixed URL schemes
- multiple Kiwix versions

Steal these compatibility lessons if an HTTP adapter is added later.

Do not put HTTP `kiwix-serve` in the core tutor path.

---

## 12.2 `scottyphillips/kiwix-mcp`

Repository:

`https://github.com/scottyphillips/kiwix-mcp`

Useful pattern:

```text
search_with_snippets
→ inspect several cheap previews
→ fetch only the relevant full article
```

Our retrieval pipeline should preserve this economy even though it uses native libzim.

---

# 13. Original linked repo: `jeffreyrampineda/kiwix-wiki-mcp-server`

Repository:

`https://github.com/jeffreyrampineda/kiwix-wiki-mcp-server`

## What is worth borrowing

Only the simple external tool shape:

- `search_wiki`
- `get_article`
- `list_libraries`
- clean MCP/Zod parameter definitions
- friendly structured error messages

This can inspire the **development-only MCP adapter**.

## What not to use

Do not reuse its production retrieval internals:

- HTTP call to `kiwix-serve`
- regex extraction of search results
- regex/basic tag stripping for article body
- full article return with weak structural understanding

Our native libzim pipeline should be substantially stronger.

---

# 14. Recommended final tutor retrieval architecture

Implement the following:

```text
Student question
    ↓
Tutor app
    ↓
Host-side query normalization
    ↓
┌───────────────────────────────────────────────┐
│ Candidate retrieval                           │
│                                               │
│  libzim Xapian full-text ─────────────┐       │
│  libzim title search ─────────────────┼─ RRF  │
│  Simple-Wiki dense sidecar ───────────┘       │
└───────────────────────────────────────────────┘
    ↓
Top candidate articles
    ↓
OpenZIM-derived ArticleBundle parser
    ↓
Section-aware semantic units/passages
    ↓
Passage BM25 / lightweight scoring
    ↓
Diversity cap + title/heading affinity
    ↓
Evidence packer with exact token budget
    ↓
Stable passage IDs + section + offsets
    ↓
Bonsai
```

For tier-2 full Wikipedia:

```text
Xapian + title only
```

at first.

Do not embed full Wikipedia in v1.

---

# 15. Proposed source tree

```text
tutor/
  retrieval/
    __init__.py

    zim/
      archive.py
      resolve.py
      search.py
      bundle.py
      content.py
      worker.py
      models.py

    hybrid/
      dense.py
      rrf.py
      lexical.py
      ranking.py
      diversity.py
      passages.py
      packer.py

    index/
      simplewiki_build.py
      simplewiki_store.py
      manifest.py

    research.py
    citations.py
    cache.py

  tools/
    research_tool.py
    calc_tool.py

  app/
    ...
```

---

# 16. What to copy vs. what to rewrite

## Safe to vendor/adapt from `cameronrye/openzim-mcp`

Because it is MIT licensed, we can reuse code with required copyright/license notice.

High-value candidates:

- archive validation
- redirect handling
- libzim search wrappers
- smart path retrieval
- HTML/main-content processing
- `EntryBundle` extraction
- heading/section offset logic
- section attribution
- RRF implementation
- result budgeting patterns
- cache key strategy
- partial-failure behavior
- archive health metadata

Prefer copying the smallest cohesive units rather than dragging the entire MCP server dependency graph into the tutor.

Keep attribution in a `THIRD_PARTY_NOTICES.md`.

---

## Reimplement from descriptions, not code

From `OscSanto/Zim-Indexer` unless license is later confirmed:

- semantic chunking
- title BM25 + body BM25 + dense RRF concept
- mention-strength penalty
- per-article diversity cap
- title/navigation boost
- optional lead/section augmentation

---

## Inspiration only

From the smaller repos:

- MCP tool naming
- archive discovery
- search-and-collect abstraction
- resumable index jobs
- source browser UX
- snippet-first fetching

---

# 17. Features from OpenZIM MCP that we should explicitly NOT port

Avoid accidental feature creep.

Do not port unless later required:

- MCP protocol server
- HTTP transport
- authentication
- CORS
- subscriptions
- resource URIs
- prompts
- server health endpoint surface
- general rate limiting
- multi-user concurrency machinery
- inbound link-graph sidecars
- Stack Exchange-specific archive presets
- cross-encoder reranker
- Docker/distribution system
- broad namespace-browsing UI
- generic "query intent parser"

The tutor is a single-user appliance. Keep it narrow.

---

# 18. Changes to `offline_tutor_spec_v0.3.md`

Claude should update the spec after implementation planning.

## Section 4 — Components

Replace the generic "retrieval library" description with:

- native libzim code adapted from OpenZIM MCP
- OpenZIM-derived `ArticleBundle`
- Simple-Wikipedia dense sidecar
- host-owned RRF/ranking/packing

---

## Section 6 — Archives

Add:

- OpenZIM-style archive signature/truncation validation
- canonical archive fingerprint
- archive capability probe:
  - full-text index
  - title index
  - metadata
  - namespace scheme

---

## Section 7 — Retrieval

Clarify the pipeline:

1. normalize query
2. Xapian full-text candidates
3. title candidates
4. Simple-Wikipedia dense candidates
5. RRF fuse
6. resolve canonical paths / redirects
7. parse only top article candidates into `ArticleBundle`
8. section-aware passage creation
9. passage ranking
10. diversity / heading-title boosts
11. exact token packing
12. stable citations

---

## Section 7.4 — Deadlines

The worker should own libzim operations that could wedge or block.

Parent process enforces:

- soft deadline
- hard deadline
- process restart
- partial response if usable work already exists

---

## Section 11 — Citations

Define passage IDs from:

```text
archive fingerprint
+ canonical entry path
+ section ID
+ extractor version
+ passage text hash / character span
```

Store:

- title
- canonical path
- section title
- section ID
- char start/end
- exact rendered passage
- edition/fingerprint

This makes the source viewer deterministic even if disposable caches are cleared.

---

# 19. Implementation order for Claude Code

## Step 1 — Vendor the minimal OpenZIM core

Bring in/adapt only enough to:

- validate a ZIM
- open it
- search it
- resolve redirects
- fetch an entry
- build an `ArticleBundle`

Add upstream MIT attribution immediately.

Do not begin embeddings yet.

### Acceptance

Given a Simple Wikipedia ZIM:

- validate it
- report fingerprint/capabilities
- search "photosynthesis"
- resolve result path
- parse article
- print lead + section tree
- retrieve one section by ID

---

## Step 2 — Citation/offset fidelity

Implement:

- passage extraction from a bundle
- passage → deepest containing section
- stable passage IDs
- character offsets
- source-viewer snapshot

### Acceptance

A passage shown to the model can be highlighted exactly in the saved article rendering.

---

## Step 3 — Lexical-only tutor retrieval

Before embeddings, get this working:

```text
query
→ Xapian
→ title search
→ RRF
→ parse top articles
→ passage rank
→ pack
```

Run the retrieval evaluation set.

This gives us a baseline.

---

## Step 4 — Add Simple-Wikipedia embeddings

Build the small dense sidecar.

Recommended first implementation:

- one vector per article:
  - title
  - lead
  - optionally first ~400 words
- 384 dimensions
- fp16 or another compact representation
- brute-force cosine/dot product initially
- sidecar keyed by archive fingerprint + embedding model version

Do not introduce ChromaDB unless brute force actually misses the latency gate.

---

## Step 5 — Add hybrid RRF

Fuse:

- dense article ranking
- Xapian full text
- title ranking

Measure.

---

## Step 6 — Add ranking refinements one at a time

Behind flags:

- diversity cap
- navigational/title boost
- mention-strength penalty
- heading affinity
- optional lead augmentation

Each feature must improve the tuning set without harming held-out retrieval.

Do not enable all of them blindly.

---

## Step 7 — Wire into tutor lifecycle

Only after retrieval is measured:

```text
student question
→ pre-retrieve
→ evidence packet
→ Bonsai request
→ optional second research call
```

Keep the production model tool schema tiny.

---

# 20. Evaluation we should steal as a mindset

`OscSanto/Zim-Indexer` has the right philosophy: retrieval should be evaluated independently of LLM answer quality.

Add or preserve metrics for:

- Hit@1
- Hit@3
- Hit@5
- Hit@10
- MRR
- correct article in candidate set
- correct article in parsed/extracted set
- correct passage in final evidence packet
- evidence budget utilization
- number of articles opened per query
- latency per stage
- cache hit/miss
- archive reads/decompressions

This lets us know whether a bad tutor answer was:

```text
retrieval failure
ranking failure
packing failure
or model failure
```

instead of blaming the LLM for everything.

---

# 21. Attribution / licensing checklist

Create:

```text
THIRD_PARTY_NOTICES.md
```

At minimum document:

## OpenZIM MCP

Repository:

`https://github.com/cameronrye/openzim-mcp`

License:

MIT

If source is copied/adapted, preserve the upstream copyright and MIT notice.

## `llm-tools-kiwix`

Repository:

`https://github.com/mozanunal/llm-tools-kiwix`

License:

Apache 2.0

Only include its code if we actually copy/adapt anything.

## `yitch/kiwix-llm`

Repository:

`https://github.com/yitch/kiwix-llm`

License:

MIT

Only include if actual source is reused.

## Zim-Indexer

Repository:

`https://github.com/OscSanto/Zim-Indexer`

No license was found during review.

Treat as research/design inspiration only until licensing is confirmed.

---

# 22. Highest-value code to inspect first

Claude Code should inspect these in this order:

### OpenZIM MCP

1. `openzim_mcp/bundle.py`
2. `openzim_mcp/synthesize.py`
3. `openzim_mcp/zim/search.py`
4. `openzim_mcp/zim/content.py`
5. `openzim_mcp/zim/redirects.py`
6. `openzim_mcp/zim/archive.py`
7. `openzim_mcp/content_processor.py`
8. `openzim_mcp/zim/structure.py`
9. `openzim_mcp/cache.py`
10. `openzim_mcp/tool_schemas.py`

### Zim-Indexer — design reference

1. `indexer/extract.py`
2. `indexer/query.py`
3. `indexer/db.py`
4. `indexer/embed.py`
5. `indexer/index.py`
6. `indexer/schema.sql`
7. `evaluate.py`

### Other references

- `mozanunal/llm-tools-kiwix`
- `yitch/kiwix-llm`
- `gulamsabri/kiwix-rag`
- `OscillateLabsLLC/kiwix-mcp`
- `scottyphillips/kiwix-mcp`
- `jeffreyrampineda/kiwix-wiki-mcp-server`

---

# 23. Bottom line

The tutor should **not** implement ZIM access from scratch.

The best approach is:

> **OpenZIM MCP for the mature ZIM/archive/search/parsing/citation plumbing + a small custom dense Simple-Wikipedia sidecar + a deliberately simple host-owned retrieval loop.**

That preserves the strongest parts of the current v0.3 architecture while removing a large amount of risky custom code.

The biggest reusable pieces are:

1. OpenZIM archive validation and canonical-path handling
2. native libzim Xapian/title search
3. `ArticleBundle` / section extraction
4. section-aware citation offsets
5. RRF fusion
6. cache/fingerprint invalidation
7. partial-failure semantics
8. Zim-Indexer-inspired structured chunking
9. Zim-Indexer-inspired dense + lexical RRF
10. diversity/title/mention ranking heuristics, enabled only after measurement

Claude Code should first build a **lexical-only vertical slice from these reused pieces**, benchmark it, then add the Simple-Wikipedia embedding sidecar. Do not start by building a giant generic RAG framework.
