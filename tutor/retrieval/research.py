"""Single entry point: research() -- the whole hybrid retrieval pipeline.

Ours: no donor code. Pipeline per request (see docs/plan/
offline_tutor_kiwix_reuse_plan.md and offline_tutor_spec_v0.3.md §7):
route archives by tier/subject -> per archive via worker: fulltext + title
candidates (deadline-bounded) -> fetch top-N entries -> build bundles ->
split passages -> BM25 over candidate passages -> RRF of (article rank,
BM25 rank) -> diversity cap -> pack to budget -> snapshot every returned
passage -> versioned response with coverage flags, timings,
archives_consulted (including storage class).

This module intentionally implements exactly one entry point,
:class:`ResearchEngine` / :meth:`ResearchEngine.research`. It is not a
generic RAG framework: ranking flags (:class:`RankingFlags`) default OFF
and are documented no-op hooks (see ``tutor.retrieval.hybrid.ranking``)
unless a caller explicitly enables one.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tutor.retrieval.hybrid.diversity import cap_per_article
from tutor.retrieval.hybrid.lexical import BM25, tokenize
from tutor.retrieval.hybrid.packer import estimate_tokens, pack
from tutor.retrieval.hybrid.passages import split_passages
from tutor.retrieval.hybrid.rrf import rrf_fuse
from tutor.retrieval.registry import ArchiveEntry, Registry
from tutor.retrieval.snapshots import SnapshotStore
from tutor.retrieval.zim.archive import fingerprint as fingerprint_archive
from tutor.retrieval.zim.bundle import build_bundle
from tutor.retrieval.zim.worker import WorkerResult, ZimWorker

logger = logging.getLogger(__name__)

RESPONSE_VERSION = 1

_FULLTEXT_LIMIT = 20
_TITLE_LIMIT = 10
_TOP_N_ARTICLES = 8
_DIVERSITY_CAP = 2
_DEFAULT_BUDGET_TOKENS = 2000
_DEFAULT_HARD_DEADLINE_S = 8.0
_PER_OP_DEADLINE_CAP_S = 3.0
_MIN_OP_DEADLINE_S = 0.05
_QUEUE_SLACK_S = 0.05

_PROCEDURAL_RE = re.compile(r"\bhow (do|would|can|should) (i|you|we)\b", re.IGNORECASE)


def _is_procedural(query: str) -> bool:
    """Whether ``query`` reads as a "how do I ..."-style procedural ask."""
    return bool(_PROCEDURAL_RE.search(query))


@dataclass(frozen=True)
class RankingFlags:
    """Optional ranking signal toggles. All default OFF (see hybrid.ranking)."""

    title_boost: bool = False
    mention_penalty: bool = False
    heading_affinity: bool = False
    lead_augmentation: bool = False


@dataclass(frozen=True)
class ResearchPassage:
    """One packed, citable passage returned by :meth:`ResearchEngine.research`."""

    label: str
    passage_id: str
    archive_id: str
    title: str
    path: str
    heading_path: tuple[str, ...]
    text: str
    start: int
    end: int
    score: float
    kind: str
    estimated_tokens: int


@dataclass
class ResearchResponse:
    """Versioned response of one :meth:`ResearchEngine.research` call."""

    version: int
    status: str  # "ok" | "partial" | "empty" | "error"
    passages: list[ResearchPassage]
    coverage: dict[str, Any]
    archives_consulted: list[dict[str, Any]]
    timings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "passages": [dataclasses.asdict(p) for p in self.passages],
            "coverage": self.coverage,
            "archives_consulted": self.archives_consulted,
            "timings": self.timings,
        }


def _call_worker(worker: Any, op: str, *, deadline_s: float, **kwargs: Any) -> WorkerResult | None:
    """Call ``worker.request`` off-thread so a misbehaving worker can never
    block the caller past ``deadline_s`` (+ small slack), regardless of what
    the worker implementation itself does with its own ``deadline_s``.
    """
    result_box: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_box.put(worker.request(op, deadline_s=deadline_s, **kwargs))
        except Exception as exc:  # noqa: BLE001 - reported as a worker error
            result_box.put(
                WorkerResult(status="error", value=None, error=str(exc), elapsed_s=0.0)
            )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    try:
        return result_box.get(timeout=deadline_s + _QUEUE_SLACK_S)
    except queue.Empty:
        return None


class ResearchEngine:
    """The single research() entry point over a :class:`Registry` of archives."""

    def __init__(
        self,
        registry: Registry,
        *,
        snapshot_store: SnapshotStore,
        cache_dir: Path,
        worker_factory: Any = None,
        ranking_flags: RankingFlags | None = None,
        hard_deadline_s: float = _DEFAULT_HARD_DEADLINE_S,
        default_budget_tokens: int = _DEFAULT_BUDGET_TOKENS,
    ) -> None:
        self._registry = registry
        self._snapshot_store = snapshot_store
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._worker_factory = worker_factory or (lambda path: ZimWorker(path))
        self._ranking_flags = ranking_flags or RankingFlags()
        self._hard_deadline_s = hard_deadline_s
        self._default_budget_tokens = default_budget_tokens
        self._workers: dict[str, Any] = {}
        self._fingerprints: dict[str, str] = {}
        self._response_cache: dict[tuple[Any, ...], ResearchResponse] = {}

    def close(self) -> None:
        for worker in self._workers.values():
            try:
                worker.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._workers.clear()

    def _get_worker(self, entry: ArchiveEntry) -> Any:
        worker = self._workers.get(entry.id)
        if worker is None:
            worker = self._worker_factory(entry.path)
            self._workers[entry.id] = worker
        return worker

    def _fingerprint_digest(self, entry: ArchiveEntry) -> str:
        digest = self._fingerprints.get(entry.id)
        if digest is None:
            digest = fingerprint_archive(entry.path).digest
            self._fingerprints[entry.id] = digest
        return digest

    def _route(self, query: str, topic_hint: str | None) -> list[ArchiveEntry]:
        entries = self._registry.for_subject(topic_hint)
        if not _is_procedural(query):
            entries = [e for e in entries if e.tier != 3]
        return entries

    def _process_archive(
        self, entry: ArchiveEntry, query: str, remaining: Any
    ) -> tuple[list[dict[str, Any]], bool]:
        worker = self._get_worker(entry)
        timed_out = False

        # Xapian's own query parser chokes on stopword/punctuation-heavy
        # natural-language questions (an all-stopword or all-AND-required
        # query can yield zero hits even when the topic is clearly present),
        # so search uses the same pinned tokenizer as BM25 rather than the
        # raw question string.
        tokens = tokenize(query)
        search_query = " ".join(tokens)

        def _op_deadline() -> float:
            return max(_MIN_OP_DEADLINE_S, min(_PER_OP_DEADLINE_CAP_S, remaining()))

        def _search(op: str, joined_query: str, limit: int) -> tuple[list[Any], bool]:
            res = _call_worker(
                worker, op, deadline_s=_op_deadline(), query=joined_query, limit=limit
            )
            if res is None or res.status != "ok":
                return [], True
            return (res.value or []), False

        def _search_with_fallback(op: str, limit: int) -> tuple[list[Any], bool]:
            # Xapian's query parser ANDs bare terms together, so a natural-
            # language question with a stray non-corpus word (e.g. "tell",
            # "about") can zero out an otherwise-good query. When the joined
            # query returns nothing, fall back to merging each token's own
            # single-term hits (deduplicated, first-seen order).
            hits, failed = _search(op, search_query, limit)
            if hits or failed or not tokens:
                return hits, failed
            merged: list[Any] = []
            seen: set[str] = set()
            any_failed = False
            for term in tokens:
                term_hits, term_failed = _search(op, term, limit)
                any_failed = any_failed or term_failed
                for hit in term_hits:
                    if hit.path not in seen:
                        seen.add(hit.path)
                        merged.append(hit)
            return merged[:limit], any_failed

        fulltext_hits, fulltext_failed = _search_with_fallback("search_fulltext", _FULLTEXT_LIMIT)
        title_hits, title_failed = _search_with_fallback("search_titles", _TITLE_LIMIT)
        timed_out = timed_out or fulltext_failed or title_failed

        logger.info(
            {
                "event": "archive_searched",
                "archive_id": entry.id,
                "storage": entry.storage,
                "tier": entry.tier,
                "fulltext_hits": len(fulltext_hits),
                "title_hits": len(title_hits),
                "timed_out": timed_out,
            }
        )

        if not fulltext_hits and not title_hits:
            return [], timed_out

        fused_articles = rrf_fuse([[h.path for h in fulltext_hits], [h.path for h in title_hits]])
        top_paths = [path for path, _ in fused_articles[:_TOP_N_ARTICLES]]
        article_rank = {path: i for i, path in enumerate(top_paths)}

        fingerprint_digest = self._fingerprint_digest(entry)

        candidate_passages: list[Any] = []
        for path in top_paths:
            if remaining() <= 0:
                timed_out = True
                break
            fetch_deadline = max(_MIN_OP_DEADLINE_S, min(_PER_OP_DEADLINE_CAP_S, remaining()))
            fetch_res = _call_worker(worker, "fetch_entry", deadline_s=fetch_deadline, path=path)
            if fetch_res is None or fetch_res.status != "ok":
                timed_out = True
                continue
            fetched = fetch_res.value
            bundle = build_bundle(fetched.html, path=fetched.path, title=fetched.title)
            candidate_passages.extend(
                split_passages(bundle, fingerprint_digest=fingerprint_digest, archive_id=entry.id)
            )

        if not candidate_passages:
            return [], timed_out

        tokenized_query = tokenize(query)
        bm25 = BM25([tokenize(p.text) for p in candidate_passages])
        bm25_scores = bm25.scores(tokenized_query)
        bm25_order = sorted(
            range(len(candidate_passages)), key=lambda i: (-bm25_scores[i], i)
        )
        bm25_ranking_ids = [candidate_passages[i].passage_id for i in bm25_order]

        article_order = sorted(
            range(len(candidate_passages)),
            key=lambda i: article_rank.get(candidate_passages[i].path, len(top_paths)),
        )
        article_ranking_ids = [candidate_passages[i].passage_id for i in article_order]

        fused = rrf_fuse([article_ranking_ids, bm25_ranking_ids])
        passages_by_id = {p.passage_id: p for p in candidate_passages}

        results: list[dict[str, Any]] = []
        for passage_id, score in fused:
            p = passages_by_id[passage_id]
            results.append(
                {
                    "passage_id": p.passage_id,
                    "archive_id": p.archive_id,
                    "title": p.title,
                    "path": p.path,
                    "heading_path": p.heading_path,
                    "text": p.text,
                    "start": p.start,
                    "end": p.end,
                    "score": score,
                    "kind": entry.kind,
                }
            )
        return results, timed_out

    def research(
        self,
        query: str,
        *,
        budget_tokens: int | None = None,
        deadline_s: float | None = None,
        topic_hint: str | None = None,
    ) -> ResearchResponse:
        cache_key = (query, budget_tokens, deadline_s, topic_hint)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            return dataclasses.replace(cached, timings={**cached.timings, "cache_hit": True})

        started = time.monotonic()
        hard_deadline = deadline_s if deadline_s is not None else self._hard_deadline_s

        def remaining() -> float:
            return hard_deadline - (time.monotonic() - started)

        archives = self._route(query, topic_hint)
        consulted: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        any_timeout = False

        for entry in archives:
            if remaining() <= 0:
                any_timeout = True
                break
            consulted.append({"id": entry.id, "storage": entry.storage, "tier": entry.tier})
            cands, timed_out = self._process_archive(entry, query, remaining)
            any_timeout = any_timeout or timed_out
            candidates.extend(cands)

        candidates.sort(key=lambda c: -c["score"])
        capped = cap_per_article(
            candidates, key=lambda c: c["path"], max_per_article=_DIVERSITY_CAP
        )
        budget = budget_tokens if budget_tokens is not None else self._default_budget_tokens
        packed = pack(capped, budget_tokens=budget, count_tokens=estimate_tokens)

        passages: list[ResearchPassage] = []
        for entry_dict in packed:
            rp = ResearchPassage(
                label=entry_dict["label"],
                passage_id=entry_dict["passage_id"],
                archive_id=entry_dict["archive_id"],
                title=entry_dict["title"],
                path=entry_dict["path"],
                heading_path=tuple(entry_dict["heading_path"]),
                text=entry_dict["text"],
                start=entry_dict["start"],
                end=entry_dict["end"],
                score=entry_dict["score"],
                kind=entry_dict["kind"],
                estimated_tokens=estimate_tokens(entry_dict["text"]),
            )
            passages.append(rp)
            self._snapshot_store.put(
                rp, fingerprint_digest=self._fingerprints.get(rp.archive_id, "")
            )

        if passages:
            status = "partial" if any_timeout else "ok"
        else:
            status = "partial" if any_timeout else "empty"

        elapsed = time.monotonic() - started
        response = ResearchResponse(
            version=RESPONSE_VERSION,
            status=status,
            passages=passages,
            coverage={"weak": len(passages) == 0},
            archives_consulted=consulted,
            timings={"elapsed_s": elapsed, "cache_hit": False},
        )
        self._response_cache[cache_key] = response
        return response
