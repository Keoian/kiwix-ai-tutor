# ArticleBundle, passages, and citation snapshots (WP-B4/B5)

## The offset model

`ArticleBundle.text` (`tutor/retrieval/zim/models.py`) is one plain-text
rendering of an article, produced by a single DOM pass
(`tutor/retrieval/zim/content.py::_render_with_headings`). Every
`SectionMeta.start`/`end` is a Python-string **code-point offset** into that
same `text` — not a byte offset, not a grapheme-cluster count. Sections tile
`text` exactly: `sections[0].start == 0`, `sections[i].end ==
sections[i+1].start` for every consecutive pair, and `sections[-1].end ==
len(text)`. `Passage.start`/`end` (`tutor/retrieval/hybrid/passages.py`) are
offsets in the same coordinate space, further constrained to never cross a
section's `[start, end)`.

The offsets are produced **online**: as the renderer walks the DOM emitting
text, it records each heading's start position at the moment it appends
that heading's text, rather than rendering everything to a string first and
then searching for headings in it. `text[start:end]` is therefore correct
by construction, with no possibility of a heading locator matching the
wrong occurrence of a repeated or common heading title.

Only `<h2>`–`<h6>` open a new section. An `<h1>` is the article title, not a
section boundary, and is folded into section 0 (the "lead"), whose
`heading` is always `""`. Nesting (`level`, `parent_index`) is computed with
a simple level stack seeded with the lead as an implicit level-1 ancestor;
tiling itself does not depend on nesting — every section, at any depth,
simply ends where the next one (in document order) begins.

## Vendored vs. rewritten

`tutor/retrieval/zim/content.py` and `tutor/retrieval/zim/bundle.py` are
headed as adapted from the openzim-mcp donor (`openzim_mcp/content_processor.py`,
`openzim_mcp/bundle.py`, tag v3.3.4). In practice very little donor code
survives verbatim:

- **Kept (the idea, reimplemented in a few lines):** `select_main_content`'s
  landmark-priority rule — try `<article>`, then `<main>`, then
  `[role=main]`, and use a match only when there is exactly one and it
  carries visible text.
- **Kept (the shape, not the code):** "one HTML parse produces one bundle
  value that downstream code slices" — `build_bundle`'s top-level structure.
- **Dropped and rewritten:** the donor locates sections by rendering to
  html2text Markdown and then regex-searching for each heading's text back
  inside it (`_compute_section_offsets`, `_locate_heading_text`,
  `_match_decorated_heading_line`, `_strip_md_inline_decorations`). That
  approach cannot *guarantee* `text[start:end]` correctness — a duplicate or
  recurring heading can resolve to the wrong occurrence, which the donor's
  own extensive comments document as a repeated real-world bug source. This
  project's acceptance criterion is exact `text[start:end]` slicing, so
  offsets are instead recorded online during a single-pass renderer
  (`_render_with_headings`), as described above.
- **Rewritten from scratch, no donor code:** infobox extraction
  (`bundle._extract_infobox`, a small `table.infobox` KV-row scan) and
  internal-link extraction (`content.iter_internal_links`). The donor's
  versions return nested TypedDicts and cover a much larger surface (media
  links, external-link classification, infobox section headers) this
  project does not need.
- **Not carried over at all:** the donor's caching layer
  (`get_or_build_bundle`, `archive_stat_token`, cache-key epoch bumps),
  html2text-based compact/raw table rendering, and search-snippet
  highlighting.

`tutor/retrieval/hybrid/passages.py` and `tutor/retrieval/snapshots.py` are
ours — no donor code.

## Passage ID formula

```
passage_id = sha256(
    fingerprint_digest + archive_id + path + section_index
    + extractor_version + sha256(passage_text) + start + end
)[:32]   # 32 hex chars
```

(see `tutor.retrieval.hybrid.passages._passage_id`). Every input that should
change the id does: the archive's fingerprint digest, which archive, which
article path, which section, which extractor code produced the bundle,
what the passage's text actually says, and where in the section it sits
(so two passages from the same long section get distinct ids even though
everything else about them matches).

> Spec v0.3 line 294 summarises the ID as edition + path + extractor version
> + content; the reuse plan (~l.1293) and implementation plan WP-B5 give the
> fuller formula used here. Treated as the same rule at different levels of
> detail, not a conflict; to be reconciled in spec v0.4.

## Snapshot store schema

`tutor.retrieval.snapshots.SnapshotStore` opens a SQLite database in WAL
mode (`check_same_thread=False`, one connection, one internal lock — safe
for one process with multiple threads, not for multiple processes) with a
single table:

```sql
CREATE TABLE snapshots (
    passage_id          TEXT PRIMARY KEY,
    archive_id          TEXT NOT NULL,
    path                TEXT NOT NULL,
    title               TEXT NOT NULL,
    heading_path        TEXT NOT NULL,  -- "\x1f"-joined heading titles
    start               INTEGER NOT NULL,
    end_offset          INTEGER NOT NULL,
    text                TEXT NOT NULL,
    fingerprint_digest  TEXT NOT NULL,
    created_at          TEXT NOT NULL   -- ISO-8601 UTC
)
```

`put` is an upsert (`INSERT ... ON CONFLICT(passage_id) DO UPDATE`), so
re-citing the same passage is idempotent. This table is the durable citation
record; it is never cleared alongside disposable index/render caches, which
is what lets `locate_in_text` re-find an exact citation's text even after a
cache wipe or a re-render of the archive: it tries the stored `(start,
end)` first, then falls back to a plain substring search for the stored
text, and only returns `None` when the text is genuinely gone.
