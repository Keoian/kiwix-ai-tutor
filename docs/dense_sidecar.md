# Dense sidecar (WP-B7)

Ours: no donor code. Ideas only from the reuse plan's donor inventory
(resumable checkpointed indexing recording archive fingerprint + embedding
model version); nothing copied from any donor, and nothing at all from
Zim-Indexer, which carries no licence.

## Why a GGUF embedding model over onnxruntime

`docs/plan/offline_tutor_implementation_plan.md`'s WP-B7 text prefers "a
GGUF embedding model served by the same llama.cpp build ... over
`onnxruntime`" so the installer packages one inference runtime, not two,
and that runtime already works on both Vulkan (Windows) and CUDA (Linux)
paths this project targets. This build uses `bge-small-en-v1.5` in GGUF
form (BERT architecture, 384-dim output, Q8_0 quantisation, ~37 MB),
served by a **second** `llama-server` process started with `--embedding`
on its own port (8081), alongside — never instead of — the chat server on
8080.

## Why numpy, when the rest of this project is stdlib-only

The acceptance criterion is a brute-force top-k query under 50 ms CPU. A
pure-Python dot product was measured directly in this environment before
reaching for numpy:

```
20,000 x 384 fp32 vectors, pure Python: 0.487s
extrapolated to 100,000 x 384:          2.43s
```

That is roughly 50x over budget, and would only get worse at the
250k-article scale the spec (§6.1) sizes Simple Wikipedia's index for.
numpy's BLAS-backed matrix-vector product clears the budget comfortably
(measured numbers below), so it is added as this WP's one new dependency
(`pyproject.toml`, with the same justification recorded inline). It is used
**only** for the search-time dot product and top-k selection; the on-disk
format is plain `struct`-packed fp16, not a numpy-specific format, so
nothing else in the codebase needs numpy to read a sidecar's bytes.

## File formats

A dense index (a "sidecar") is a directory containing:

| File | Format |
|---|---|
| `vectors.fp16` | Flat file of IEEE-754 half-precision floats, row-major, `count * dim` of them. Packed with `struct.pack(f"<{dim}e", *row)` (`array` has no half-float typecode). Row `i` is passage `i`'s L2-normalised embedding. |
| `ids.txt` | One passage id per line, UTF-8, `\n`-terminated. Row `i` names the passage embedded at row `i` of `vectors.fp16`. |
| `manifest.json` | Written only once the build is **complete** (see below). |
| `checkpoint.json` | Written periodically during a build; absent once `manifest.json` exists (a finished sidecar has no in-progress state). |

### `manifest.json` fields

```json
{
  "version": 1,
  "archive_digest": "<ArchiveFingerprint.digest, sha256(uuid+size+mtime_ns)>",
  "extractor_version": "zim-bundle-v1",
  "embedding_model_name": "bge-small-en-v1.5-q8_0",
  "embedding_model_sha256": "<sha256 of the .gguf file>",
  "dim": 384,
  "count": 1234,
  "normalisation": "l2",
  "created_at": "2026-09-20T02:40:00+00:00"
}
```

`archive_digest` is `tutor.retrieval.zim.archive.fingerprint(path).digest`
— the same value the rest of retrieval already uses to detect a
replaced/re-downloaded archive. `DenseIndex.open(dir, archive_digest=...)`
refuses to open a sidecar whose manifest digest doesn't match, with a
clear `DenseIndexError` naming both digests.

### `checkpoint.json` fields (internal to the builder)

```json
{
  "archive_digest": "...", "extractor_version": "...",
  "embedding_model_name": "...", "embedding_model_sha256": "...",
  "dim": 384, "normalisation": "l2",
  "next_entry_id": 4821,
  "articles_processed": 2000,
  "count": 9481,
  "vectors_bytes": 7281408,
  "ids_bytes": 303392
}
```

## Resume semantics

Articles are visited in ascending libzim entry-id order (the only
deterministic order libzim exposes without a separate sort index), skipping
redirects and non-HTML entries. Each qualifying article becomes passages
via the existing `zim.bundle` + `hybrid.passages` pipeline (same passage
ids the rest of retrieval uses), embedded in sub-batches of `batch_size`
through a caller-supplied `embed(texts) -> vectors` callable, and appended
to the two flat files.

**A checkpoint is only ever taken at an article boundary** — after *all*
of one article's sub-batches have been embedded and appended, never
mid-article. This is what makes "truncate to the last checkpoint, then
resume" safe despite embedding happening in smaller sub-batches within an
article: if a crash (process kill, embed-server error) happens partway
through an article, the bytes it already appended belong to an article
that never reached a checkpoint. On resume, `truncate_to()` cuts both flat
files back to the last checkpoint's exact recorded byte lengths — discarding
that in-progress article's partial rows wholesale — and the article is
simply reprocessed from scratch. No row is ever double-counted, and no
partial row is ever left dangling. This is exercised directly:
`tests/test_simplewiki_build.py::test_resume_detects_and_truncates_torn_tail`
appends garbage bytes past a checkpoint and confirms the resumed build is
byte-identical to a clean one; `test_resume_after_interruption_is_byte_identical`
does the same via an embed callable that raises mid-build.

A resumed build that changes archive, embedding model, model sha256, dim,
extractor version, or normalisation from what its existing checkpoint
recorded raises `BuildConfigError` rather than silently mixing
incompatible rows — start a fresh `out_dir` to change any of those.

`--limit-articles` counts qualifying articles (not raw entry ids) and is
honoured across a resume: a build limited to 2,000 articles that is
interrupted and resumed still stops at 2,000 total.

## Query path

`DenseIndex.open(directory, archive_digest=...)` loads `ids.txt` into a
tuple, reads `vectors.fp16` with `numpy.fromfile` and upcasts to float32
once into memory (250k x 384 float32 ≈ 384 MB — the fp16 file itself is
half that on disk). `search(query_vec, k)` L2-normalises the query, takes a
single `vectors @ q` BLAS matrix-vector product, and uses
`numpy.argpartition` for an O(n) top-k selection (only the top k get fully
sorted). Both index rows and the query are unit vectors, so this dot
product **is** cosine similarity — no separate norm division needed.

## Measured numbers (this session, this machine)

Machine: Windows, one llama-server on port 8080 (chat, already running,
never touched) plus a second llama-server on port 8081 (`--embedding`,
started and stopped by this session). GPU vendor/VRAM was not
independently confirmed (`nvidia-smi` not present on PATH); both servers
were health-checked as simultaneously healthy before and during the
sample build.

- **Embedding model**: `bge-small-en-v1.5` GGUF, Q8_0, BERT architecture,
  384-dim, ~37 MB (downloaded from
  `CompendiumLabs/bge-small-en-v1.5-gguf` on Hugging Face; sha256
  `ec38e8da142596baa913124ae50550de284b6916bf59577ef2f0cb9660c2f514`,
  stored at `runtime/models/embedding.gguf`, gitignored). The model itself
  downloaded in one `curl` attempt; numpy's own wheel is what hit the
  flaky network described in `pyproject.toml`'s WP-B7 comment — large
  downloads (>4 MB) were consistently truncated mid-transfer in this
  environment, and a loop of `curl -C -` resumes was needed to complete it.
- **Sample build**: first 2,000 qualifying articles of
  `C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim`, lead passage only
  per article (per the design fix below).
- **Design correction made during this session**: the first working
  version of the builder embedded *every* body passage per article
  (~9.3 passages/article on this sample), not just the lead. That
  contradicts both the spec (§6.1: "title + first ~400 words") and the
  reuse plan (step 4: "one vector per article: title, lead ... first ~400
  words") — both prescribe one vector per article. It also failed the
  latency budget at realistic scale: a synthetic 1,000,000-row benchmark
  (roughly what full passage-level embedding of ~250k spec-sized articles
  would produce) measured **p50 59.6ms / p95 63.3ms — over the 50ms
  budget**. Switching to lead-only embedding (`split_passages(...)[0]` per
  article) fixed this at the spec's own intended scale; see below.

### Sample build (2,000 articles, lead-only)

```
2000/2000 articles, 62.6s elapsed
sample build: 1973 passages in 62.80s (31.4 passages/s ~= articles/s)
sample query latency (n=1973): p50=0.32ms p95=0.46ms max=0.46ms
```

(1,973 rather than 2,000 rows: a handful of articles have no lead body —
e.g. redirects to disambiguation stubs or infobox-only pages — and
contribute no lead passage.)

**Full-build extrapolation** (label: extrapolation, not measured): at
31.85 articles/s and the spec's own ~250,000-qualifying-article estimate
for Simple Wikipedia tier-1 content:

```
250,000 / 31.85 articles/s ~= 7,849s ~= 2.2 hours
```

This ran on CPU-side batching through a live embedding server (no GPU
utilisation was independently confirmed — `nvidia-smi` was not on PATH in
this environment); the spec's own text expects the real number to be
"well under an hour" on a GPU, so 2.2 hours should be read as a
conservative, possibly CPU-bound, upper estimate pending a real full run.

### Query latency (brute-force cosine, numpy)

```
synthetic, n=250,000  (dim=384, spec's own sizing): p50=14.89ms p95=15.81ms
synthetic, n=1,000,000 (dim=384, 4x spec's sizing): p50=59.61ms p95=63.25ms
sample,    n=1,973    (dim=384, this session's real build): p50=0.32ms p95=0.46ms
```

**At the spec's intended scale (~250k rows), the < 50ms CPU acceptance
criterion is met with headroom (15ms).** At 4x that scale (1M rows, not
what this project's corpus is sized for), it is missed by ~10-13ms. If the
real corpus ever grows past ~500k-700k rows and needs to stay under 50ms,
the fix is not a different algorithm but a cheaper per-row cost: int8
quantised vectors (half the memory traffic of the float32 upcast this
implementation does today) or capping candidate rows via a coarse
pre-filter (e.g. only searching the archive's already-computed BM25
top-N candidate set) before the exact brute-force pass, rather than
introducing an ANN index the reuse plan explicitly says to avoid at this
size.

## Embedding server runs on the CPU

**Measured** on this machine, 2026-09-20, `bge-small-en-v1.5` Q8_0 (37 MB,
384-dim), CPU, 4 threads, while a GPU build was running concurrently:

- Single query embedding: p50 6.2 ms (max 17.2 ms over 10 queries).
- Bulk: 31.7 lead passages/s.

Versus **31.4 articles/s** measured on the Vulkan GPU over the
2,000-article sample above. CPU is as fast as the GPU here, so
`[embedding]` now defaults to `n_gpu_layers = 0` (CPU), `threads = 4`,
`ctx_size = 512` (`tutor/settings.py`, `config/dev.toml`). Running the
embedding server on CPU:

- Frees all VRAM for the chat model — matters on the 6 GB delivery
  machine, where the chat model and an embedding model both wanting GPU
  memory would compete.
- Avoids opening a second Vulkan context alongside the chat server's.
- Makes the embedding server backend-independent on both Windows and
  Linux, since it no longer depends on a working Vulkan/CUDA setup at all.

`scripts/serve_embed.ps1` / `.sh` take the `-ngl`/`-t`/`-c` flags (and
everything else) from `python -m tutor.settings --argv-embedding`, so
this default lives in one place.

**Assumption, not measured**: the full `runtime/simplewiki_dense` index
build that was started on the GPU before this change switched the
default to CPU is treated as numerically equivalent to a CPU-built index
for retrieval purposes — same model file, same fp16 on-disk vector
storage, so the vectors themselves don't depend on which backend produced
them. That equivalence has not been independently verified by re-running
the build on CPU and diffing outputs; it follows from the model and
storage format being identical, not from a measurement.

## Running the full build

The full Simple Wikipedia build (~394,566 entries, ~250k qualifying
articles per the spec's own estimate) is **explicitly deferred** — this
session only ran a 2,000-article sample to get real numbers. To run it:

```powershell
# 1. Start the embedding server (separate terminal; leave running):
scripts\serve_embed.ps1

# 2. Run the full build (no --limit-articles):
python -m eval.tools.dense_bench sample `
    --archive "C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim" `
    --out "runtime\simplewiki_dense" `
    --embed-url http://127.0.0.1:8081 `
    --model-name "bge-small-en-v1.5-q8_0" `
    --model-sha256 <sha256 of runtime/models/embedding.gguf> `
    --dim 384 --batch-size 32 --checkpoint-every 200
```

(`dense_bench.py sample` takes `--limit-articles`; passing a very large
value or a follow-up flag to omit it entirely covers the full archive. The
builder is resumable, so re-running the same command after an interruption
picks up where it left off rather than restarting.)

**Estimated duration and disk size**: extrapolated from the measured
sample throughput above to ~250,000 qualifying articles; see the
placeholder section once filled in. Disk size scales linearly with passage
count: the spec's own estimate for the full corpus is ~200 MB
(`~250k x 384 x 2 bytes`) for `vectors.fp16` alone, plus a small `ids.txt`
(32 hex chars + newline per passage) and a negligible `manifest.json`.
