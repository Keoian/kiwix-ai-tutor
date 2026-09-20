"""Resumable dense sidecar builder for a ZIM archive (WP-B7).

Ours: no donor code (ideas only from the reuse plan's donor inventory --
resumable batch indexing with a checkpoint recording archive fingerprint,
embedding model version, and completed-range; no code copied from any
donor, least of all Zim-Indexer which carries no licence at all).

Articles are visited in ascending libzim entry-id order -- the only order
libzim exposes deterministically without building a separate sort index --
skipping redirects and non-HTML entries. Per the spec (Sec 6.1) and the
reuse plan's step 4, this indexes **one vector per article**: the lead
passage only (title's section, capped at ``hybrid.passages``'s existing
~400-char passage limit) -- not every body passage. This keeps the sidecar
at the spec's sized-for-it ~250k rows for Simple Wikipedia rather than the
several-x-larger row count full passage-level embedding would produce,
which is what keeps brute-force query latency under the WP-B7 acceptance
budget at that scale (see docs/dense_sidecar.md's measured numbers).

Lead passages are gathered into batches of ``batch_size`` **articles**
(crossing article boundaries freely, unlike a hypothetical multi-passage
scheme) and embedded through a caller-supplied ``embed`` callable one
batch at a time. Because there is at most one passage per article, a batch
is always a whole number of complete articles: ``embed()`` either returns
successfully and the whole batch is appended, or it raises and *nothing*
for that batch was appended. A checkpoint is taken after every
``checkpoint_every`` batches, recording the exact byte lengths the flat
files should have and the entry id to resume from. This is what makes
"truncate to the last checkpoint, then resume" produce a byte-identical
result to an uninterrupted run: a crash never leaves a partially-appended
row, only whole appended-and-uncheckpointed batches, which
``truncate_to()`` discards wholesale on resume before those articles are
simply reprocessed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from libzim.reader import Archive

from tutor.retrieval.hybrid.passages import Passage, split_passages
from tutor.retrieval.index.manifest import (
    MANIFEST_VERSION,
    BuildCheckpoint,
    DenseManifest,
    read_checkpoint,
    write_json_atomic,
)
from tutor.retrieval.index.simplewiki_store import (
    CHECKPOINT_FILENAME,
    IDS_FILENAME,
    MANIFEST_FILENAME,
    PATHS_FILENAME,
    VECTORS_FILENAME,
    append_bytes,
    append_ids,
    file_size,
    pack_vectors,
    read_ids,
    truncate_to,
)
from tutor.retrieval.zim.archive import fingerprint as fingerprint_archive
from tutor.retrieval.zim.bundle import EXTRACTOR_VERSION, build_bundle

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]


class BuildConfigError(Exception):
    """Raised when an existing checkpoint doesn't match this build's config."""


@dataclass(frozen=True)
class _Article:
    entry_id: int
    path: str
    title: str
    html: str


def _iter_articles(archive: Archive, *, start_entry_id: int) -> Iterator[_Article]:
    """Yield qualifying (non-redirect, text/html) articles in entry-id order."""
    total = archive.all_entry_count
    for entry_id in range(start_entry_id, total):
        entry = archive._get_entry_by_id(entry_id)  # noqa: SLF001 - only iteration API libzim exposes
        if entry.is_redirect:
            continue
        item = entry.get_item()
        if not item.mimetype.startswith("text/html"):
            continue
        html = bytes(item.content).decode("utf-8", errors="replace")
        yield _Article(entry_id=entry_id, path=entry.path, title=entry.title, html=html)


def _normalise(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def build_index(
    archive_path: Path,
    out_dir: Path,
    embed: EmbedFn,
    *,
    archive_id: str,
    embedding_model_name: str,
    embedding_model_sha256: str,
    dim: int,
    batch_size: int = 32,
    checkpoint_every: int = 10,
    limit_articles: int | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> DenseManifest:
    """Build (or resume) a dense sidecar for ``archive_path`` into ``out_dir``.

    ``embed`` must be deterministic given its input for the resume test's
    byte-identical guarantee to hold (a real embedding server is
    deterministic at temperature-free inference; the unit tests use a fake
    deterministic embedder).

    ``limit_articles`` counts *qualifying* articles (redirects and non-HTML
    entries excluded), not raw libzim entry ids, and is honoured across a
    resume: a build limited to 2000 articles that is interrupted and
    resumed still stops at 2000 total.
    """
    archive_path = Path(archive_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    digest = fingerprint_archive(archive_path).digest
    normalisation = "l2"

    checkpoint_path = out_dir / CHECKPOINT_FILENAME
    vectors_path = out_dir / VECTORS_FILENAME
    ids_path = out_dir / IDS_FILENAME
    paths_path = out_dir / PATHS_FILENAME

    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint is not None:
        if (
            checkpoint.archive_digest != digest
            or checkpoint.extractor_version != EXTRACTOR_VERSION
            or checkpoint.embedding_model_name != embedding_model_name
            or checkpoint.embedding_model_sha256 != embedding_model_sha256
            or checkpoint.dim != dim
            or checkpoint.normalisation != normalisation
        ):
            raise BuildConfigError(
                f"existing checkpoint at {checkpoint_path} was built with different "
                "archive/model/extractor settings; start a fresh out_dir to change them"
            )
        # A crash may have appended bytes after the last checkpoint was
        # written; truncate both flat files back to the checkpointed truth
        # before resuming so no row is ever counted twice.
        truncate_to(vectors_path, checkpoint.vectors_bytes)
        truncate_to(ids_path, checkpoint.ids_bytes)
        # paths.txt is row-aligned with ids.txt/vectors.fp16. An old
        # checkpoint from before paths.txt existed has ``paths_bytes==0``
        # (see BuildCheckpoint's default) and no paths.txt on disk;
        # truncate_to tolerates a missing file at target size 0.
        truncate_to(paths_path, checkpoint.paths_bytes)
        start_entry_id = checkpoint.next_entry_id
        count = checkpoint.count
        articles_processed = checkpoint.articles_processed
    else:
        # Fresh build: start the flat files from empty regardless of any
        # stray bytes left by an unrelated prior run in this directory.
        for path in (vectors_path, ids_path, paths_path):
            if path.exists():
                truncate_to(path, 0)
            else:
                path.touch()
        start_entry_id = 0
        count = 0
        articles_processed = 0

    archive = Archive(str(archive_path))

    def _checkpoint() -> None:
        write_json_atomic(
            checkpoint_path,
            BuildCheckpoint(
                archive_digest=digest,
                extractor_version=EXTRACTOR_VERSION,
                embedding_model_name=embedding_model_name,
                embedding_model_sha256=embedding_model_sha256,
                dim=dim,
                normalisation=normalisation,
                next_entry_id=next_entry_id,
                articles_processed=articles_processed,
                count=count,
                vectors_bytes=file_size(vectors_path),
                ids_bytes=file_size(ids_path),
                paths_bytes=file_size(paths_path),
            ).to_dict(),
        )

    next_entry_id = start_entry_id
    batches_since_checkpoint = 0
    start_time = time.monotonic()

    pending: list[tuple[int, Passage]] = []  # (article's next_entry_id, lead passage)

    def _flush() -> None:
        """Embed and append one batch of whole articles' lead passages.

        Nothing is appended unless ``embed()`` returns successfully for the
        *entire* batch, so a crash here never leaves a partial row -- only
        a batch that was never appended at all, safe to reprocess wholesale
        on resume.
        """
        nonlocal count, next_entry_id, batches_since_checkpoint
        if not pending:
            return
        texts = [p.text for _, p in pending]
        vectors = embed(texts)
        if len(vectors) != len(pending):
            raise RuntimeError(f"embed() returned {len(vectors)} vectors for {len(pending)} texts")
        vectors = [_normalise(v) for v in vectors]
        append_bytes(vectors_path, pack_vectors(vectors, dim))
        append_ids(ids_path, [p.passage_id for _, p in pending])
        append_ids(paths_path, [p.path for _, p in pending])
        count += len(pending)
        next_entry_id = pending[-1][0]
        pending.clear()
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_every:
            _checkpoint()
            batches_since_checkpoint = 0

    for article in _iter_articles(archive, start_entry_id=start_entry_id):
        if limit_articles is not None and articles_processed >= limit_articles:
            next_entry_id = article.entry_id
            break
        bundle = build_bundle(article.html, path=article.path, title=article.title)
        passages = split_passages(bundle, fingerprint_digest=digest, archive_id=archive_id)
        if passages:
            pending.append((article.entry_id + 1, passages[0]))

        articles_processed += 1
        if len(pending) >= batch_size:
            _flush()

        if progress_cb is not None:
            elapsed = time.monotonic() - start_time
            progress_cb(articles_processed, limit_articles or archive.all_entry_count, elapsed)
    else:
        # Ran off the end of the archive (no --limit-articles cutoff): there
        # is nothing left to resume from. If articles are still pending,
        # the final _flush() below overwrites this with the same value
        # (the last article's entry id + 1 == all_entry_count).
        next_entry_id = archive.all_entry_count

    _flush()
    _checkpoint()

    manifest = DenseManifest(
        version=MANIFEST_VERSION,
        archive_digest=digest,
        extractor_version=EXTRACTOR_VERSION,
        embedding_model_name=embedding_model_name,
        embedding_model_sha256=embedding_model_sha256,
        dim=dim,
        count=count,
        normalisation=normalisation,
        created_at=datetime.now(UTC).isoformat(),
    )
    write_json_atomic(out_dir / MANIFEST_FILENAME, manifest.to_dict())
    return manifest


def backfill_paths(
    archive_path: Path,
    out_dir: Path,
    *,
    archive_id: str,
    force: bool = False,
) -> int:
    """Recompute ``paths.txt`` for a sidecar built before paths.txt existed.

    Re-walks ``archive_path`` in exactly the builder's deterministic
    order (see ``_iter_articles``), without calling any embedder,
    recomputing each qualifying article's lead-passage id and asserting
    it equals the corresponding row of the existing ``ids.txt`` --
    row-by-row, aborting loudly on the first mismatch (a mismatch means
    either the archive or the extractor has changed since the sidecar was
    built, and the recomputed paths would silently mislabel rows).

    ``archive_id`` must be the same value used to build the sidecar (it
    feeds the passage-id hash, see ``passages._passage_id``); the id
    comparison will fail loudly if a wrong value is supplied.

    Refuses to run while a build looks to be in progress -- a checkpoint
    present with no completed manifest -- unless ``force`` is set,
    matching the safety rule the builder itself applies to its own
    resume path.

    Returns the number of paths written.
    """
    archive_path = Path(archive_path)
    out_dir = Path(out_dir)
    checkpoint_path = out_dir / CHECKPOINT_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME
    if checkpoint_path.exists() and not manifest_path.exists() and not force:
        raise BuildConfigError(
            f"a build appears to be in progress at {out_dir} (checkpoint present, "
            "no manifest present); refusing to backfill paths.txt without force=True"
        )

    digest = fingerprint_archive(archive_path).digest
    existing_ids = read_ids(out_dir / IDS_FILENAME)

    archive = Archive(str(archive_path))
    paths: list[str] = []
    for article in _iter_articles(archive, start_entry_id=0):
        bundle = build_bundle(article.html, path=article.path, title=article.title)
        article_passages = split_passages(bundle, fingerprint_digest=digest, archive_id=archive_id)
        if not article_passages:
            continue
        lead = article_passages[0]
        row = len(paths)
        if row >= len(existing_ids):
            break
        if lead.passage_id != existing_ids[row]:
            raise BuildConfigError(
                f"backfill mismatch at row {row}: ids.txt has {existing_ids[row]!r}, "
                f"recomputed id for article {article.path!r} is {lead.passage_id!r} "
                "-- refusing to write a possibly misaligned paths.txt "
                "(archive, extractor version, or archive_id may have changed)"
            )
        paths.append(lead.path)

    if len(paths) != len(existing_ids):
        raise BuildConfigError(
            f"backfill produced {len(paths)} paths but ids.txt has {len(existing_ids)} rows "
            "-- refusing to write a misaligned paths.txt"
        )

    tmp_path = out_dir / (PATHS_FILENAME + ".tmp")
    data = ("".join(f"{p}\n" for p in paths)).encode("utf-8")
    tmp_path.write_bytes(data)
    tmp_path.replace(out_dir / PATHS_FILENAME)
    return len(paths)
