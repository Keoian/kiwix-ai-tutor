# Donor Inventory — WP-B1 Research

Status: research/inventory only. No production code was written or vendored as
part of this document. All findings below were verified by running commands
and reading source on 2026-09-19 unless explicitly marked "inferred".

---

## (a) `python-libzim` wheel findings

**Verdict: GO on both platforms.**

- Windows: `python -m pip install libzim --only-binary=:all:` installs a
  prebuilt wheel from PyPI for Python 3.12 on win_amd64. Verified by actual
  install (took several retries — see "Gaps and surprises").
  - Wheel: `libzim-3.13.0-cp312-cp312-win_amd64.whl`
  - `pip show libzim` → Version 3.13.0, License-Expression `GPL-3.0-or-later`,
    home page `https://github.com/openzim/python-libzim`.
  - Verified imports all succeed in the same interpreter:
    `import libzim`, `from libzim.reader import Archive`,
    `from libzim.writer import Creator`,
    `from libzim.search import Query, Searcher`,
    `from libzim.suggestion import SuggestionSearcher`.
  - `libzim` has no `__version__` attribute (confirmed — `hasattr` is False);
    version must be read via `importlib.metadata.version("libzim")` or
    `pip show`, not `libzim.__version__`.
- Linux: confirmed via PyPI JSON API (`https://pypi.org/pypi/libzim/json`)
  rather than a full download (downloads kept timing out — see below).
  Available cp312 wheels for 3.13.0 (current) and 3.9.0 (openzim-mcp's floor
  pin):
  - `libzim-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
  - `libzim-3.13.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`
  - `libzim-3.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
  - (aarch64 equivalent also exists for 3.9.0)
  - The plan's suggested `--platform manylinux_2_28_x86_64` tag is not the
    literal wheel platform tag (the actual tag is the compound
    `manylinux_2_27_x86_64.manylinux_2_28_x86_64`), but pip's platform
    compatibility resolution accepts `manylinux_2_28_x86_64` as a request
    tag correctly per PEP 600 — a full `pip download` invocation with that
    exact flag was attempted three times and each attempt hit
    `ReadTimeoutError` from `files.pythonhosted.org` before completing (see
    surprises). The JSON-API check is a positive confirmation of wheel
    *existence*, not a downloaded/imported confirmation on Linux (that would
    require an actual Linux box, which was not available in this Windows
    session).

No source build was attempted or needed.

---

## (b) OpenZIM MCP donor pin

- Clone location: `C:\git\kiwix-ai-tutor\runtime\donors\openzim-mcp`
  (inside the gitignored `runtime/` tree; nothing here is committed).
- **Newest real tag found: `v3.3.4`** (confirmed via `git tag --sort=-creatordate`).
  This directly contradicts the plan document's stated "v2.5.3 as newest as
  of 2026-07-01" expectation — the upstream project has released many more
  versions since (3.0.0 → 3.3.4) that the plan's author apparently never saw.
  The other value mentioned as possibly-inaccurate, `v3.3.4`, **does exist**
  and is in fact the newest tag — so of the two candidate "newest" values
  floating around, `v3.3.4` is correct and `v2.5.3` is stale/wrong.
- Checked out commit: `9358db06f205bb0b95cc938c68405851a0e205a8`
  (`chore: release 3.3.4 (#438)`).
- License: `LICENSE` file present, **MIT**, copyright "(c) 2025-2026 Cameron Rye".
  Full standard MIT text, no additional restrictions.
- `.python-version`: `3.12`
- Parser/library pins from `pyproject.toml`:
  - `libzim>=3.9.0,<4.0`
  - `beautifulsoup4>=4.14.3,<5.0`
  - `html2text>=2025.4.15,<2027.0`
  - Both BeautifulSoup (`html.parser` built-in parser, not lxml) and
    html2text are used — BeautifulSoup for DOM manipulation and section
    extraction (`content_processor.py`, `bundle.py`), html2text-*shaped*
    markdown output is produced by `content_processor._render_soup_to_text`
    (confirmed by reading the file: it imports both `html2text` and `bs4`,
    and comments throughout `bundle.py` describe matching "html2text's
    escaping/whitespace behavior", strongly suggesting the renderer wraps or
    mimics html2text's conventions even though it also does its own
    BeautifulSoup-based extraction). This is a two-parser dependency, not
    one, which the reuse plan's "do not maintain two unrelated HTML parsers"
    guidance (§3.2) directly warns against.

---

## (c) Reuse-plan reference → real location table

Format: plan line → symbol/path in repo at v3.3.4 → approx LOC of the
containing file → non-stdlib/non-libzim/non-parser imports it pulls in →
verdict. "Verified" = read the actual code. "NOT FOUND" = grepped and not
present under that name.

| Plan ref (line) | Real path @ v3.3.4 | LOC (file) | Imports beyond stdlib/libzim/parser | Verdict |
|---|---|---|---|---|
| `ZIM_MAGIC` (L62) | `openzim_mcp/zim/archive.py:105` | 1906 | `openzim_mcp.config`, `.constants`, `.exceptions`, `.defaults`, `bs4` (indirect via module), `ProcessPoolExecutor`/`multiprocessing` | vendor with adaptation |
| `_declared_zim_size` (L63) | `zim/archive.py:157` | (same file) | none beyond stdlib | vendor as-is |
| `has_zim_signature` (L64) | `zim/archive.py:212` | (same file) | none beyond stdlib | vendor as-is |
| `is_truncated_zim` (L65) | `zim/archive.py:173` | (same file) | none beyond stdlib | vendor as-is |
| `zim_signature_error` (L66) | `zim/archive.py:140` | (same file) | none beyond stdlib | vendor as-is |
| `unreadable_zim_warning` (L67) | `zim/archive.py:242` | (same file) | none beyond stdlib | vendor as-is |
| archive open timeout / integrity check / process isolation (L68-72) | `zim/archive.py` `_VALIDATION_POOL`/`_validation_pool()` (~line 269-296) | 1906 | `concurrent.futures.ProcessPoolExecutor`, `multiprocessing` | vendor with adaptation (the pool machinery is intertwined with `OpenZimMcpConfig`/`CacheConfig`) |
| redirect resolution (L117, L126-141) | `zim/redirects.py:31` `resolve_redirect_chain`, `:60` `best_effort_redirect_chain` | 96 | none beyond stdlib/libzim | vendor as-is — smallest, cleanest file in the whole donor |
| smart-retrieval fallback sequence (L119-141) | `zim/content.py` (redirect/path-mapping logic) + `website/src/content/docs/smart-retrieval.mdx` (docs, not code) | 2418 | `openzim_mcp.cache`, `.config`, `.security.PathValidator`, `.pagination.Cursor`, `.recovery_advice`, `.error_messages`, `bs4`, `soupsieve` | vendor with heavy adaptation — this file is not just resolution, it is the whole content-fetch/paging/rewrite path |
| native full-text/title search, `Searcher`/`Query`/`SuggestionSearcher` (L179-227) | `zim/search.py` | **4887** | `openzim_mcp.config`, `.cache`, `.constants`, `.defaults`, `.text_utils`, `.title_promotion`, `.preset_data`, `.error_messages`, `.recovery_advice`, `bs4` (crawl-artefact detection) | vendor with heavy adaptation — largest file in the repo; `search_top_k` at line 4772 is deep inside a much larger mixin class |
| `search_top_k` (L201) | `zim/search.py:4772` (method, not free function — part of a mixin) | (same file) | inherits from `_ArchiveAccessMixin`, `_SearchMixin` composition | reimplement — extracting a single method out of a 4887-line mixin file that depends on cache/config/preset internals is not practical; reimplement the ~20-30 lines of actual libzim `Searcher`/`Query` calling convention directly |
| search snippets → passage candidates (L237-244) | `synthesize.py` (whole file) | 2451 | `openzim_mcp.bundle`, `.tool_schemas`, `.constants`, `.text_utils` | vendor with adaptation for `_rrf_fuse`/`_extract_passages`/`_normalize_ws`; reimplement the rest |
| `EntryBundle` / `extract_entry_bundle` (L269-334) | `bundle.py` (whole file) | 693 | `openzim_mcp.tool_schemas` (TypedDicts only), `bs4`, `.content_processor.select_main_content` | vendor as-is/adapt — cleanest high-value file; only real external coupling is the `tool_schemas` TypedDict imports, which are trivially replaced with our own dataclasses |
| `_compute_section_offsets` (L296) | `bundle.py:432` | (same file) | none beyond stdlib/re | vendor as-is |
| `_locate_heading_text` (L296) | `bundle.py:367` | (same file) | none beyond stdlib/re | vendor as-is |
| `select_main_content` (L348) | `content_processor.py:1245` | **2561** | `bs4` (`BeautifulSoup`, `Comment`, `NavigableString`, `Tag`), `soupsieve`, `html2text` | vendor with adaptation — function itself is bounded, but the file it lives in is 2561 lines of tightly coupled helpers (`_strip_furniture_sections`, `_strip_in_page_nav`, `_flatten_multiline_table_cells`, infobox extraction, meta-tag extraction, markdown rendering) that `select_main_content` and `extract_entry_bundle` both call into |
| `_extract_passages`, `_locate_passage`, `_normalize_ws`, `_attribute_sections` (L391-397) | `synthesize.py:153`, `:251`, `:203`, `:326` | 2451 | `openzim_mcp.bundle`, `.tool_schemas` | vendor with adaptation — functions are individually small and mostly self-contained, but file-level imports pull in bundle/tool_schemas types |
| `_rrf_fuse` (L432) | `synthesize.py:60` | (same file) | none beyond stdlib | vendor as-is — smallest, most independent unit found besides `redirects.py` |
| budget enforcement / evidence packing (L468-491) | `synthesize.py`, `tool_schemas.py` | 2451 / 887 | `tool_schemas.py` is pure TypedDicts, no runtime deps | reimplement the packer logic from the described contract; the TypedDict shapes in `tool_schemas.py` are safe to read as a reference but not worth vendoring since the tutor needs its own models anyway |
| partial-success semantics (L495-526) | spread across `synthesize.py`, `exceptions.py`, `responses.py` | n/a | `openzim_mcp.exceptions`, `.responses` | reimplement — this is a cross-cutting pattern, not a portable module |
| article bundle cache / cache key strategy (L531-598) | `bundle.py` (`_bundle_cache_key`, `archive_stat_token`, `_BUNDLE_KEY_PREFIX`, `_RENDER_EPOCH`) + `cache.py` | 693 / 1007 | none beyond stdlib for the bundle-side key functions; `cache.py` (not inspected line-by-line) is a general-purpose cache layer | vendor the key-derivation *functions* as-is; reimplement (or skip) the generic `OpenZimMcpCache` class itself — plan explicitly says drop caching layers |
| libzim process isolation / worker (L602-636) | `zim/archive.py` `_VALIDATION_POOL` and friends | 1906 | `multiprocessing`, `concurrent.futures` | reimplement — donor's worker is narrowly for the integrity-check subprocess, not a general libzim RPC worker; the tutor's `worker.py` needs a broader protocol (open/search/read/resolve) that doesn't exist as a unit anywhere in this repo |
| Stack Exchange preset (plan L1220, listed as something to *avoid* porting) | `archive_types.py` (`ArchiveType` literal incl. `"stackexchange"`, SE host-detection heuristics ~line 15-87) | not counted (small file) | none beyond stdlib | not needed — plan explicitly says not to port this, and it exists confirming the donor does have it |
| cross-encoder reranker (plan L1221, also explicitly excluded) | `rerank.py` | 302 | `openzim_mcp.constants`, `.zim.search.demote_crawl_artefacts`, optional `fastembed` extra | not needed — present, confirms exclusion list is accurate |

Not found at all under the names the plan gives:
- `configure_libzim_cache(...)` (L101) — NOT FOUND as a standalone function.
  libzim cache tuning appears inline inside `zim/archive.py`'s archive-open
  path rather than as a separately named helper.
- `archive_capabilities(path)` / `archive_fingerprint(path)` (L98-100) —
  NOT FOUND as named functions; the *concept* (archive stat/fingerprint) is
  present via `bundle.archive_stat_token` but there is no single
  `archive_fingerprint` API matching the plan's proposed tutor-side name.
  These are tutor-side names the plan invented, not upstream symbols, which
  is consistent with the plan's own phrasing ("Create something like...").
- `get_search_snippet(hit)` (L226) — NOT FOUND under this name; snippet
  handling is inlined inside `zim/search.py`'s much larger search mixin
  methods, not exposed as one small function.

Reference count: **~24 distinct plan references checked**; **21 resolved to
real code** (several to the same donor file), **3 not found under the
stated name** (the tutor-side API names the plan proposes do not exist
upstream verbatim — expected, since the plan explicitly frames them as
"create something like").

---

## (d) Recommended vendoring units → `tutor/retrieval/zim/`

Given the actual coupling discovered (very different from what the plan's
compact per-topic write-up implies), the smallest *honestly* cohesive units
are:

- **`archive.py`** — port the pure header/signature functions verbatim
  (`ZIM_MAGIC`, `_declared_zim_size`, `has_zim_signature`, `is_truncated_zim`,
  `zim_signature_error`, `unreadable_zim_warning`). These have zero
  cross-module coupling. Everything else in the donor's `zim/archive.py`
  (config objects, cache config, process-pool validation) must be stripped
  and reimplemented narrowly against `libzim.reader.Archive` directly.
- **`resolve.py`** — port `zim/redirects.py` almost verbatim (96 lines,
  no internal coupling). This is the single cleanest donor file found.
- **`search.py`** — do **not** attempt to extract `search_top_k` from the
  4887-line `zim/search.py` mixin. Reimplement a small wrapper directly
  against `libzim.search.Query`/`Searcher`/`SuggestionSearcher`, using the
  donor only as a reading reference for query-construction and
  result-limiting conventions.
- **`bundle.py`** — the best single vendoring candidate in the repo. Port
  `extract_entry_bundle`, `_compute_section_offsets`, `_locate_heading_text`,
  `_match_decorated_heading_line`, `_strip_md_inline_decorations`,
  `archive_stat_token`, and the bundle cache-key functions, replacing the
  `openzim_mcp.tool_schemas` TypedDict imports with tutor-owned
  `ArticleBundle`/`SectionMeta`/`LinkBuckets`/`InfoboxData` dataclasses.
  Depends on `content.py`'s `select_main_content`/`_build_headings`, which
  must come along (see next item).
- **`content.py`** — port only `select_main_content` and the specific
  helper functions `bundle.py` calls (`_build_headings`,
  `extract_html_links`, `extract_infobox`, `_render_soup_to_text`) out of
  `content_processor.py` (2561 lines). Do not port the file wholesale — it
  contains meta-tag extraction, archive-type-specific cleanup, and markdown
  edge-case handling for MCP resource rendering that the tutor does not need.
- **`worker.py`** — no real donor exists for this. There is no
  general-purpose libzim RPC worker in openzim-mcp; only a narrow
  single-purpose `ProcessPoolExecutor` used for archive integrity checks.
  This module must be designed from scratch, using the plan's own §6
  description (open/search/read/resolve worker protocol) as the spec, and
  only borrowing the archive-open-timeout idea from `zim/archive.py`.
- **`models.py`** — write fresh; use donor `tool_schemas.py` purely as a
  reading reference for field names/shapes (it is pure TypedDicts, safe to
  read, not worth vendoring since Pydantic/dataclass conventions differ).

**Must be stripped from every vendored file:**
- `openzim_mcp.config.OpenZimMcpConfig` / `CacheConfig` — the whole config
  system (server config, transport config, resource limits).
- `openzim_mcp.cache.OpenZimMcpCache` — generic TTL cache layer; the tutor
  needs its own narrow bundle/path caches keyed by archive fingerprint per
  plan §5, not this general-purpose class.
- `openzim_mcp.security.PathValidator` — MCP-specific path/permission
  sandboxing tied to the server's resource-exposure model.
- `openzim_mcp.rate_limiter`, `.subscriptions`, `.http_app`, `.server`,
  `.server_state`, `.mcp_envelope`, `.sdk_compat` — the entire MCP protocol
  server/transport stack (confirmed present as separate files: `server.py`,
  `http_app.py`, `subscriptions.py`, `rate_limiter.py`).
- `openzim_mcp.rerank` — cross-encoder reranker (plan explicitly excludes).
- `openzim_mcp.archive_types` Stack Exchange preset detection (plan
  explicitly excludes).
- `openzim_mcp.linkgraph.*` — inbound link-graph sidecar (plan explicitly
  excludes).
- `openzim_mcp.intent_parser` (2767 lines!) — generic query-intent parser
  (plan explicitly excludes; also by far one of the largest files in the
  repo, confirming this exclusion matters).
- `openzim_mcp.onboarding`, `.instructions`, `.tool_schemas` MCP-specific
  wrapper types, `.responses.ToolErrorPayload` — MCP tool-response envelope
  conventions, not needed for an in-process library.

**HTML parser to standardize on:** BeautifulSoup with the built-in
`html.parser` backend (already what the donor pins per its own comment
"pinned so the dependency footprint stays minimal" in `content_processor.py`
line 152). Treat `html2text` as optional/skippable — the donor's actual
DOM-walk-based renderer (`_render_soup_to_text`) does the real work;
html2text's presence in `pyproject.toml` needs closer confirmation before
deciding whether the tutor needs it too (not fully resolved — see gaps).

---

## (e) Gaps and surprises

1. **Biggest surprise: this is not a "primary donor with a few modules"
   codebase — it's a ~23,000+ line MCP server with heavy internal coupling.**
   The reuse plan's per-topic write-ups (§2.1-§4.3) read as if each concept
   maps to a small, extractable function. In reality several of the named
   "modules" are single files of 2,000-4,900 lines (`zim/search.py` 4887,
   `intent_parser.py` 2767, `content_processor.py` 2561, `synthesize.py`
   2451, `zim/content.py` 2418, `zim/structure.py` 2106, `zim/archive.py`
   1906) that internally depend on the project's config/cache/security
   system. Only two files in the whole codebase (`zim/redirects.py` at 96
   lines, and the free functions in `bundle.py`) are cleanly self-contained.
   "Vendor the smallest cohesive unit" in practice means porting a handful
   of standalone functions plus rewriting the surrounding plumbing, not
   copying files.

2. **Version mismatch confirmed and resolved in the donor's favor of
   `v3.3.4`.** The plan document itself says "Reviewed release: v3.3.4" in
   its own §2 header while also saying elsewhere (per this task's brief)
   that v2.5.3 was expected as newest as of 2026-07-01. `git tag` shows the
   real tag history runs 2.6.x → 2.7.0 → 3.0.0 → 3.1.x → 3.2.x → 3.3.4, i.e.
   the project moved fast and 3.3.4 genuinely is newest — the plan's
   internal "v2.5.3" expectation is simply stale/wrong, not a fabricated
   version.

3. **The pip install for `libzim` was highly unreliable on this network** —
   four separate `pip install`/`pip download` invocations failed with either
   `ReadTimeoutError` mid-download or (twice) a hash mismatch against PyPI's
   published sha256 for the same 16MB wheel, each time with a *different*
   wrong hash, consistent with truncated/corrupted downloads rather than a
   tampered package. Install succeeded on a later retry (4th attempt in a
   loop). This is a real operational risk for CI/setup scripts on this
   network and should be handled with retry loops, not treated as a security
   incident.

4. **Whether `html2text` is actually still load-bearing at runtime, or is a
   vestigial pin, was not fully resolved.** `bundle.py`'s comments describe
   matching "html2text's whitespace handling" and "html2text's backslash
   escaping" as behavior to replicate, which reads as if the renderer used
   to be html2text-based and was replaced by a custom BeautifulSoup-based
   renderer that intentionally mimics html2text's output shape for
   backward-compatible section-offset math. `content_processor.py` does
   `import html2text` at module level (confirmed), but time did not permit
   tracing every call site to confirm whether it's invoked at runtime or
   only referenced in dead/legacy code paths. Recommend the vendoring
   implementer check this directly before deciding whether the tutor's
   `content.py` needs html2text as a dependency at all — the plan's
   instruction to avoid two parsers may already be satisfied by the donor,
   or may not be.

5. **`OscSanto/Zim-Indexer`** (plan §7) was **not** cloned or inspected as
   part of this task — the assignment scope (STEP 2/3) was OpenZIM MCP only.
   Its licensing-unclear status and "reimplement, don't copy" instruction
   from the plan stand unverified by this pass.

6. **libzim's own license is GPL-3.0-or-later** (per `pip show libzim`),
   distinct from `python-libzim`'s wrapper/binding license and from
   openzim-mcp's MIT license. This wasn't explicitly flagged in the reuse
   plan and should be accounted for in `THIRD_PARTY_NOTICES.md` alongside
   the MIT notice for openzim-mcp, since the tutor will depend on libzim
   directly regardless of what's vendored from openzim-mcp.
