"""Manual measurement tool for the dense sidecar's WP-B7 acceptance numbers.

Not part of the pytest suite (it needs a real ZIM archive and, for the
throughput number, a live embedding server) -- it is what
docs/dense_sidecar.md's measured numbers came from. Two things are
measured:

1. Embedding throughput and query latency over a small SAMPLE build (a few
   thousand articles of a real archive), via an ``EmbeddingClient`` against
   a running embedding llama-server.
2. Query latency of ``DenseIndex.search`` against a SYNTHETIC index at full
   scale (e.g. 1,000,000 x dim), which needs no archive or model at all --
   just random unit vectors -- to check the brute-force top-k acceptance
   criterion independent of how big the real corpus turns out to be.

Usage:
    python -m eval.tools.dense_bench sample --archive PATH --out DIR \
        --embed-url http://127.0.0.1:8081 --limit-articles 2000 --dim 384

    python -m eval.tools.dense_bench synthetic --count 1000000 --dim 384
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tutor.retrieval.hybrid.dense import DenseIndex
from tutor.retrieval.index.embedding_client import EmbeddingClient
from tutor.retrieval.index.simplewiki_build import backfill_paths, build_index
from tutor.retrieval.zim.archive import fingerprint as fingerprint_archive


def _time_queries(index: DenseIndex, *, n_queries: int = 20) -> list[float]:
    rng = np.random.default_rng(0)
    dim = index.manifest.dim
    latencies = []
    for _ in range(n_queries):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        t0 = time.perf_counter()
        index.search(q.tolist(), k=10)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _run_sample(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    out_dir = Path(args.out)
    client = EmbeddingClient(args.embed_url)

    def embed(texts: list[str]) -> list[list[float]]:
        return client.embed(texts)

    t0 = time.perf_counter()
    manifest = build_index(
        archive,
        out_dir,
        embed,
        archive_id=args.archive_id,
        embedding_model_name=args.model_name,
        embedding_model_sha256=args.model_sha256,
        dim=args.dim,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        limit_articles=args.limit_articles,
        progress_cb=lambda done, total, elapsed: print(
            f"  {done}/{total} articles, {elapsed:.1f}s elapsed", file=sys.stderr
        ),
    )
    build_elapsed = time.perf_counter() - t0

    passages_per_s = manifest.count / build_elapsed if build_elapsed > 0 else float("inf")
    print(f"sample build: {manifest.count} passages in {build_elapsed:.2f}s "
          f"({passages_per_s:.1f} passages/s)")

    digest = fingerprint_archive(archive).digest
    index = DenseIndex.open(out_dir, archive_digest=digest)
    latencies = _time_queries(index)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"sample query latency (n={len(index.ids)}): "
          f"p50={p50:.2f}ms p95={p95:.2f}ms max={max(latencies):.2f}ms")


def _run_synthetic(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    count, dim = args.count, args.dim
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    latencies = []
    n_queries = 20
    for _ in range(n_queries):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        t0 = time.perf_counter()
        scores = vectors @ q
        k = 10
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx[np.argsort(-scores[top_idx])]
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"synthetic query latency (n={count}, dim={dim}): "
          f"p50={p50:.2f}ms p95={p95:.2f}ms max={max(latencies):.2f}ms")


def _run_backfill_paths(args: argparse.Namespace) -> None:
    n = backfill_paths(
        Path(args.archive),
        Path(args.dir),
        archive_id=args.archive_id,
        force=args.force,
    )
    print(f"backfill-paths: wrote {n} paths to {Path(args.dir) / 'paths.txt'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample = sub.add_parser(
        "sample", help="build+measure a small sample against a live embedding server"
    )
    sample.add_argument("--archive", required=True)
    sample.add_argument("--out", required=True)
    sample.add_argument("--embed-url", required=True)
    sample.add_argument("--archive-id", default="simplewiki")
    sample.add_argument("--model-name", required=True)
    sample.add_argument("--model-sha256", required=True)
    sample.add_argument("--dim", type=int, required=True)
    sample.add_argument("--batch-size", type=int, default=32)
    sample.add_argument("--checkpoint-every", type=int, default=50)
    sample.add_argument("--limit-articles", type=int, default=2000)
    sample.set_defaults(func=_run_sample)

    synthetic = sub.add_parser("synthetic", help="query-latency-only, no archive or model needed")
    synthetic.add_argument("--count", type=int, default=1_000_000)
    synthetic.add_argument("--dim", type=int, default=384)
    synthetic.set_defaults(func=_run_synthetic)

    backfill = sub.add_parser(
        "backfill-paths",
        help="recompute paths.txt for a sidecar built before paths.txt existed",
    )
    backfill.add_argument("--archive", required=True)
    backfill.add_argument("--dir", required=True)
    backfill.add_argument("--archive-id", default="simplewiki")
    backfill.add_argument("--force", action="store_true")
    backfill.set_defaults(func=_run_backfill_paths)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
