"""The tutor's per-turn agent loop: receive -> route -> pre-retrieve ->
build prompt -> inference -> tool dispatch -> inference -> finish.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §6 (turn flow;
the research cap message: "If the model requests a second `research`
after the cap, the host returns a tool result stating the cap is reached
and the model answers the supported portion or asks for clarification.")
and docs/plan/offline_tutor_implementation_plan.md WP-C2 (`research` cap 2
INCLUDING the host pre-retrieval, `calc` cap 4, schema validation, never
parse prose as a tool call).

The six scripted lessons (this module's own enumeration, see
tests/test_agent_loop.py for the authoritative list under test):

1. Free-text question: host pre-retrieves before the first model call.
2. The model's first reply is a `research` follow-up: executed as call 2
   of the research cap.
3. A third research request hits the cap: the host returns a cap-reached
   tool result instead of calling the real research engine again.
4. A deterministic action never retrieves.
5. Arithmetic free text: the model calls `calc` (cap 4 per turn, separate
   from the research cap).
6. A malformed tool call (bad JSON, unknown name, schema violation) is
   validated and, on failure, never executed -- the host returns a
   validation-error tool result instead. Prose that merely looks like a
   tool call is never parsed as one: only a "tool_call" StreamEvent from
   the LLM client is ever treated as a tool call.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from tutor.app.citations import _CITATION_REMINDER, extract_labels, render_evidence
from tutor.app.prompt import PromptOverflow, render_evidence_with_reuse
from tutor.app.repetition_guard import find_repetition_loop
from tutor.retrieval.assessment import assess_evidence
from tutor.retrieval.hybrid.lexical import singularize, tokenize
from tutor.tools.schemas import FORCED_RESEARCH_TOOL, TOOLS, validate_tool_call

RESEARCH_CAP = 2
CALC_CAP = 4

_STUDENT_SAFE_ERROR = (
    "Sorry, I ran into a problem answering that. Please try asking again."
)

_SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


class _UnionCancel:
    """Duck-typed ``threading.Event``-alike (only ``is_set()`` is used by
    ``LlamaClient.stream_chat``) that is set when ANY of the given events
    is set. Lets the repetition-loop guard (below) trigger the exact same
    clean-stream-close path as a real user cancel -- closing the HTTP
    connection to llama-server properly, never killing the server --
    while still letting ``run_turn`` tell the two apart afterwards by
    checking the original ``cancel`` event directly."""

    def __init__(self, *events) -> None:
        self._events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)


def _load_default_system_text() -> str:
    """Read the host's default system prompt from ``system_prompt.txt``
    (kept in a plain-text file so it's easy to review/diff independently
    of the code, and so ``eval/run_turn_eval.py`` can swap variants in
    without touching this module's source)."""
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@dataclass
class TurnResult:
    status: str  # "ok" | "error" | "cancelled"
    answer_text: str
    route: str
    research_calls: int = 0
    calc_calls: int = 0
    cached_tokens: int | None = 0
    prompt_tokens: int | None = None
    events: list = field(default_factory=list)
    uncited: bool = False
    """True when the route was a factual pre-retrieval (evidence was
    supplied) but the model's final answer contains no [S#] label at all.
    Per spec §11/§12 the host never fabricates a citation the model did
    not write; this flag only tells the UI to show an "uncited" notice
    (§12's chat pane already distinguishes source-backed/computed/own-
    example statement styles, so this is the same kind of provenance
    signal, not a new citation)."""
    truncated: str | None = None
    """``None`` (not truncated), ``"repetition"`` (the host-side
    repetition-loop guard in tutor.app.repetition_guard stopped
    generation and trimmed the answer), or ``"max_tokens"`` (the server's
    own ``finish_reason == "length"``, i.e. the ``max_tokens`` cap was
    hit). Citations and attributions still run on the (possibly trimmed)
    ``answer_text``; the host never fabricates content to fill in what
    was cut."""
    timings: dict = field(
        default_factory=lambda: {
            "forced_call": 0.0,
            "searches": 0.0,
            "second_round": 0.0,
            "answer_prefill": 0.0,
            "answer_generation": 0.0,
        }
    )
    """Wall-clock seconds spent per turn stage (see docs/
    rewrite_on_weak_evidence.md, "Model may skip the search" /
    per-stage timing addendum): ``forced_call`` (the model deciding what
    to search, first round), ``searches`` (running the research engine
    against the model's queries), ``second_round`` (the whole weak-
    evidence extra round, call + searches; 0.0 when it never fires),
    ``answer_prefill`` (answer request sent -> first token), and
    ``answer_generation`` (first token -> stream done). Always present
    and non-negative; a slow prompt-read on constrained hardware shows
    up here as a large ``answer_prefill``."""
    evidence: dict | None = None
    """Additive (see docs/rewrite_on_weak_evidence.md): ``None`` for
    non-factual routes (action/greeting-shaped turns never pre-retrieve).
    For a factual (``preretrieve*``) route, always a dict with
    ``level_before``/``level_after`` (the pre- and post-rewrite
    ``assess_evidence`` levels -- identical when no rewrite ran),
    ``rewritten_queries`` (the host-executed query list, ``[]`` when no
    rewrite ran), and ``corrected_terms`` (the merged research response's
    spelling/compound corrections dict)."""


def _passage_to_dict(passage) -> dict:
    if isinstance(passage, dict):
        return dict(passage)
    return {
        "id": getattr(passage, "passage_id", None) or getattr(passage, "id", None),
        "label": getattr(passage, "label", None),
        "title": getattr(passage, "title", None),
        "path": getattr(passage, "path", None),
        "text": getattr(passage, "text", ""),
        "kind": getattr(passage, "kind", "article"),
    }


def _retain_passages(session, packet: dict) -> None:
    """Hand a research packet's passages to ``session.retain_passages``
    (if the session supports it) so citations can be resolved later from
    ``session.known_passages()`` -- including after prompt-log eviction.
    Sessions without the method (some fakes in unit tests) are a no-op."""
    retain = getattr(session, "retain_passages", None)
    if retain is not None:
        retain(packet.get("passages", []))


def _packet_from_response(response) -> dict:
    passages = getattr(response, "passages", None)
    if passages is None and isinstance(response, dict):
        passages = response.get("passages", [])
    return {"passages": [_passage_to_dict(p) for p in (passages or [])]}


def _to_wire_messages(rendered: list[dict]) -> list[dict]:
    """Translate ``PromptLog.render()``'s internal message shapes into the
    OpenAI-compatible wire shapes llama-server's ``/v1/chat/completions``
    accepts: evidence's ``{"role": "tool", "passages": [...]}`` becomes a
    plain ``{"role": "tool", "content": <rendered evidence text>}``, and
    an assistant entry's internal ``cited_labels`` bookkeeping field
    (never part of the wire schema) is dropped."""
    wire: list[dict] = []
    for message in rendered:
        if message.get("passages") is not None:
            wire.append(
                {"role": message["role"], "content": render_evidence(message)}
            )
        elif message.get("role") == "assistant" and "cited_labels" in message:
            wire.append({"role": "assistant", "content": message.get("content")})
        else:
            wire.append(message)
    return wire


def _trim_and_append_evidence(
    log, passages: list[dict], budget, *, reuse_prior_passages: bool = False
) -> None:
    """Append ``passages`` as an evidence packet, trimming the
    lowest-ranked (last) passages first if the full packet does not fit
    ``budget.newest`` -- never crashing the turn on PromptOverflow.

    ``reuse_prior_passages`` (see docs/passage_reuse.md and
    ``tutor.settings.AppConfig.reuse_prior_passages``): when set, a
    passage whose full text is already present earlier in the log
    (``log.held_ids()``) is pasted as a short pointer line instead of
    being re-pasted in full."""
    remaining = list(passages)
    while True:
        try:
            log.append_evidence(
                remaining, budget=budget, reuse_prior_passages=reuse_prior_passages
            )
            return
        except PromptOverflow:
            if not remaining:
                raise
            remaining = remaining[:-1]


# The bracketed note the host appends to the student's own message text
# (append-only: written once, as part of the single user message that
# gets logged, never edited afterwards) when a forced rewrite round is
# about to run. Kept short since it eats into the same turn's
# ``budget.newest`` allowance. See docs/rewrite_on_weak_evidence.md.
_REWRITE_HOST_NOTE = (
    " [Host note: the library search for this question came back weak. "
    "Call research with 1-3 short rewritten queries: fix spelling, "
    "split or join fused words, use the topic's standard name, and "
    "resolve any \"it\"/\"that\" from the lesson so far.]"
)

# Host note used on every turn after the first (see
# docs/rewrite_on_weak_evidence.md, "Follow-up rewrite"): the raw
# pre-search ran on the literal words of the question, which for a
# follow-up turn can retrieve the wrong topic entirely (e.g. "Is it a
# molecule?" retrieves the "Molecule" article, when the lesson has been
# about DNA). Fires unconditionally on turn >= 2 -- deliberately NOT
# gated by any word-list/pronoun detector, since a student's own grammar
# or spelling can't be relied on to signal a reference.
_FOLLOWUP_HOST_NOTE = (
    " [Host note: call research with 1-3 short standalone rewritten "
    "queries for this question: resolve any pronoun/reference (\"it\", "
    "\"that\", \"they\", ...) using the lesson so far, and fix spelling. "
    "For example if the lesson has been about DNA and the student asks "
    "\"Is it a molecule?\", search for \"Is DNA a molecule\", not "
    "\"molecule\". Then answer the student's actual question as it "
    "relates to the lesson so far: answer directly first (yes or no, if "
    "it is a yes/no question), then explain.]"
)

# Host note for ``app.model_writes_search`` (default True, adopted --
# see docs/model_writes_search_measure.md; originally see
# docs/rewrite_on_weak_evidence.md, "Model writes every search"): fires on
# EVERY turn, including turn 1, replacing both the weak-evidence rewrite
# and the follow-up rewrite for that turn. Root cause this targets: the
# deterministic pre-search word-matches the student's RAW text, so
# "What's the largest molecule?" retrieves "Molecule Man" (a comic
# character) and "What's the biggest animal?" retrieves "The Biggest
# Loser" (a TV show) -- describing words and full sentences are exactly
# what trip up the word-matcher. Tells the model to write short,
# title-like queries instead of sentences/questions.
_MODEL_WRITES_SEARCH_HOST_NOTE = (
    " [Host note, for the research tool call ONLY (never repeat any of "
    "this in your answer to the student): call research. FIRST give \"question\": the student's "
    "latest message rewritten as one standalone question, replacing "
    "\"it\"/\"that\"/\"they\"/\"the other ones\"/etc. with what it refers "
    "to in the lesson so far, and fixing spelling. E.g. lesson about tires "
    "and rubber, student says \"They're a single molecule?\" -> question "
    "\"Is a tire a single molecule?\"; lesson about titin, student says "
    "\"What are the other ones?\" -> question \"What other very long "
    "molecules are there besides titin?\"; student says \"calvinize the "
    "rubber\" -> question \"Does vulcanizing rubber make a tire a single "
    "molecule?\", queries \"Vulcanization\", \"Rubber\", \"Tire\". THEN "
    "give 1-3 short \"queries\" for that question, like encyclopedia "
    "article titles or key terms, NOT full sentences. Leave out describing words "
    "like \"biggest\"/\"longest\"/\"fastest\"/\"how long\" unless part of "
    "a real title. If you already know the likely answer, make that one "
    "of the queries. Don't repeat last turn's exact queries unless the "
    "question is the same. After the library results arrive, answer the "
    "student in plain sentences; do not write \"question\" or "
    "\"queries\" in the answer.]"
)

# Small cap on the forced rewrite call's own output -- it only needs to
# emit one tool call, never prose.
_FORCED_REWRITE_MAX_TOKENS = 96

# Cap on the number of passages kept after merging all rewritten queries'
# results, matching the normal single-query evidence packet size so the
# rewritten evidence is no more expensive than an ordinary research call.
_MERGE_CAP = 8


# Reciprocal-rank-fusion constant, matching the standard/existing RRF used
# elsewhere in retrieval (tutor/retrieval/hybrid/rrf.py, RRF k=60).
_RRF_K = 60

_TITLE_BOOST = 0.5
_GENERIC_TITLE_PENALTY = 0.5


def _title_terms(title: str) -> frozenset[str]:
    return frozenset(singularize(t) for t in tokenize(title or ""))


def _is_generic_or_disambiguation(
    title: str,
    title_terms: frozenset[str],
    query_term_union: frozenset[str] | None = None,
) -> bool:
    """A title that is a disambiguation page, shares only a single generic
    content term with the query (e.g. "Garden", "Square"), or otherwise
    only INCIDENTALLY overlaps the rewritten queries (e.g. "6 Foot 7 Foot",
    a song whose title happens to contain "foot" but is not itself built
    entirely of query terms and shares only that one word with them), is a
    weak, non-specific hit that should never outrank a multi-term article
    match on the actual topic. The incidental-overlap check only applies
    when ``query_term_union`` (the union of every rewritten query's own
    terms) is given and the title is NOT itself a subset of it -- a title
    that IS a full subset already gets ``_TITLE_BOOST`` instead and must
    never also be penalized here."""
    if "disambiguation" in (title or "").lower():
        return True
    if len(title_terms) <= 1:
        return True
    if query_term_union and not (title_terms <= query_term_union):
        overlap = title_terms & query_term_union
        if len(overlap) <= 1:
            return True
    return False


def _merge_dedupe_passages(
    passage_lists: list[list[dict]],
    cap: int,
    *,
    rewritten_queries: list[str] | None = None,
) -> list[dict]:
    """Merge several passage lists (one per rewritten query) via
    reciprocal-rank fusion (RRF, k=60) computed at ARTICLE level (so an
    article found -- at any rank -- by more than one rewritten query
    reliably outranks an article only one query happened to find), then
    ordered passage-level within each article by the same fusion applied
    to passages. A title-match boost is applied when an article's own
    title terms are fully contained in one of the rewritten queries (e.g.
    "Square foot gardening" is a substring-of-terms match for "square foot
    gardening basics"); a penalty is applied to disambiguation pages and
    titles that carry only a single, generic content term (e.g. "Garden",
    "Square"), so a specific multi-query hit always beats a generic
    single-term title. Capped to ``cap`` passages; ties are broken
    deterministically by (best original rank, title, passage id)."""
    query_term_sets = [
        frozenset(singularize(t) for t in tokenize(q)) for q in (rewritten_queries or [])
    ]

    article_rrf: dict[str, float] = {}
    article_hits: dict[str, int] = {}
    article_best_rank: dict[str, int] = {}
    article_title: dict[str, str] = {}
    passage_rrf: dict[str, float] = {}
    passage_best_rank: dict[str, int] = {}
    by_id: dict[str, dict] = {}

    for passages in passage_lists:
        seen_articles_this_list: set[str] = set()
        for rank, passage in enumerate(passages):
            pid = passage.get("id")
            if pid is None:
                continue
            title = passage.get("title") or ""
            by_id.setdefault(pid, passage)
            passage_rrf[pid] = passage_rrf.get(pid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if pid not in passage_best_rank or rank < passage_best_rank[pid]:
                passage_best_rank[pid] = rank

            if title not in seen_articles_this_list:
                seen_articles_this_list.add(title)
                article_hits[title] = article_hits.get(title, 0) + 1
                article_rrf[title] = article_rrf.get(title, 0.0) + 1.0 / (_RRF_K + rank + 1)
                article_title[title] = title
                if title not in article_best_rank or rank < article_best_rank[title]:
                    article_best_rank[title] = rank

    query_term_union: frozenset[str] = (
        frozenset().union(*query_term_sets) if query_term_sets else frozenset()
    )

    def _article_score(title: str) -> float:
        terms = _title_terms(title)
        score = article_rrf.get(title, 0.0)
        if query_term_sets and terms and any(terms <= qts for qts in query_term_sets):
            score += _TITLE_BOOST
        if _is_generic_or_disambiguation(title, terms, query_term_union):
            score -= _GENERIC_TITLE_PENALTY
        return score

    # A generic/disambiguation/incidental-overlap title (e.g. a song whose
    # title happens to share one word with the query) must sort AFTER
    # every non-generic hit regardless of how many rewritten queries
    # happened to surface it -- vote count alone would otherwise let it
    # beat the one real, on-topic article a single query found (the
    # smoke-test bug this fixes: "6 Foot 7 Foot" found by 2/3 rewritten
    # queries outranking "Square foot gardening" found by only 1). Ties
    # within each of those two tiers still resolve by vote count, then
    # score, then best rank, then title, exactly as before.
    def _is_generic_tier(title: str) -> bool:
        return _is_generic_or_disambiguation(title, _title_terms(title), query_term_union)

    article_order = sorted(
        article_hits.keys(),
        key=lambda t: (
            _is_generic_tier(t),
            -article_hits[t],
            -_article_score(t),
            article_best_rank[t],
            t,
        ),
    )
    article_rank = {t: i for i, t in enumerate(article_order)}

    ordered_ids = sorted(
        by_id.keys(),
        key=lambda pid: (
            article_rank.get(by_id[pid].get("title") or "", len(article_order)),
            -passage_rrf.get(pid, 0.0),
            passage_best_rank.get(pid, 0),
            str(pid),
        ),
    )
    return [by_id[pid] for pid in ordered_ids[:cap]]


def _lead_with_backfill(
    lead_passages: list[dict], backfill_passages: list[dict], cap: int
) -> list[dict]:
    """Merge a follow-up-rewrite query's passages (``lead_passages``, kept
    in their own rank order) ahead of the raw pre-search's passages
    (``backfill_passages``), rather than an equal reciprocal-rank-fusion
    blend: for a follow-up turn the raw pre-search ran on the literal
    (possibly wrong-topic) question text, so it only ever fills in
    remaining slots the rewrite's own results didn't use, never
    outranks them. De-duplicated by passage id; capped to ``cap``."""
    seen: set[str] = set()
    merged: list[dict] = []
    for passage in lead_passages:
        pid = passage.get("id")
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        merged.append(passage)
    for passage in backfill_passages:
        if len(merged) >= cap:
            break
        pid = passage.get("id")
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        merged.append(passage)
    return merged[:cap]


def _relabel_sequential(passages: list[dict]) -> list[dict]:
    """Return a copy of ``passages`` with ``label`` reassigned to
    ``S1``..``Sn`` in list order, discarding whatever label each passage
    arrived with (assigned independently by its own originating
    retrieval call, and not guaranteed unique across calls -- see
    ``_run_forced_rewrite_round``)."""
    return [{**p, "label": f"S{i + 1}"} for i, p in enumerate(passages)]


@dataclass
class _MergedResult:
    """Minimal ``.passages`` holder so ``assess_evidence`` (which only
    ever reads ``result.passages``) can be re-run against the merged,
    de-duplicated evidence from all of a forced rewrite's queries."""

    passages: list[dict]


def _not_found_tool_text(*, searched_for: list[str], level_after: str) -> str:
    """The instruction text appended as the forced rewrite's tool result
    when the merged, re-assessed evidence is still weak/empty. Tells the
    model plainly to say it could not find this in the library, suggest a
    better way to ask (or a related topic actually in the evidence, if
    any), and -- only if it offers anything from memory -- to keep it
    brief, clearly label it as from memory and unchecked, and give no
    specific numbers/dates/names."""
    return (
        "No good match was found in the library for this question, even "
        f"after rewriting the search ({level_after} evidence). Tell the "
        "student plainly that you could not find this in the library. "
        "Suggest a better way to ask, or point to a related topic that IS "
        "covered by any sources given earlier in this lesson, if one "
        "exists. Only if you choose to add anything from your own general "
        "knowledge: keep it brief, clearly say it is from memory and "
        "unchecked, and do not give any specific numbers, dates, or names."
    )


_MODEL_QUERY_MAX_WORDS = 8


def _clip_model_written_queries(queries: list[str]) -> list[str]:
    """Validate/clip queries the MODEL wrote (``app.model_writes_search``):
    at most 3 queries, each stripped of surrounding quotes and a trailing
    "?", clipped to ``_MODEL_QUERY_MAX_WORDS`` words. Anything that becomes
    empty after cleaning is dropped. Callers fall back to today's
    behaviour when this returns an empty list."""
    cleaned: list[str] = []
    for raw in (queries or [])[:3]:
        if not isinstance(raw, str):
            continue
        q = raw.strip()
        q = q.strip("\"'").strip()
        if q.endswith("?"):
            q = q[:-1].strip()
        if not q:
            continue
        words = q.split()
        if len(words) > _MODEL_QUERY_MAX_WORDS:
            q = " ".join(words[:_MODEL_QUERY_MAX_WORDS])
        if q:
            cleaned.append(q)
    return cleaned


_MODEL_QUESTION_MAX_WORDS = 30


def _clip_model_written_question(question) -> str | None:
    """Clip the model's optional standalone ``question`` argument
    (``app.model_writes_search``) to a single line of at most
    ``_MODEL_QUESTION_MAX_WORDS`` words, for safe display in the restate
    line. Returns ``None`` when there is nothing usable."""
    if not isinstance(question, str):
        return None
    q = " ".join(question.split())
    q = q.strip().strip("\"'").strip()
    if not q:
        return None
    words = q.split()
    if len(words) > _MODEL_QUESTION_MAX_WORDS:
        q = " ".join(words[:_MODEL_QUESTION_MAX_WORDS])
    return q or None


def _forced_research_tool_call(llm, messages: list[dict], *, cancel):
    """Force the model to answer this turn's next completion with exactly
    one ``research`` tool call, primarily via the OpenAI-compatible
    ``tool_choice`` request field; if the server rejects that field
    (a non-2xx/error StreamEvent), fall back to a JSON-schema/grammar
    constrained plain-text completion asking for the same shape and
    synthesize an equivalent tool-call.

    Returns ``(tool_call_id, queries, arguments_json, question)`` or
    ``None`` if the model produced nothing usable (both paths failed, or
    the forced call returned no queries) -- callers treat ``None`` the
    same as "still weak" and skip straight to the not-found outcome.
    ``question`` is the model's optional standalone-question argument
    (``app.model_writes_search``), unvalidated/unclipped, or ``None`` when
    absent or when the fallback (non-tool-call) path was used.
    """
    forced_tool_choice = {"type": "function", "function": {"name": "research"}}
    text_parts: list[str] = []
    tool_call = None
    errored = False
    for evt in llm.stream_chat(
        messages,
        tools=[FORCED_RESEARCH_TOOL],
        tool_choice=forced_tool_choice,
        cancel=cancel,
        max_tokens=_FORCED_REWRITE_MAX_TOKENS,
    ):
        if evt.kind == "tool_call":
            tool_call = evt
        elif evt.kind == "token":
            text_parts.append(evt.text or "")
        elif evt.kind == "error":
            errored = True
            break

    if tool_call is not None:
        validation = validate_tool_call("research", tool_call.arguments_json or "{}")
        if validation.ok:
            queries = validation.arguments.get("queries") or (
                [validation.arguments["query"]] if validation.arguments.get("query") else []
            )
            if queries:
                question = validation.arguments.get("question")
                return tool_call.id, queries, tool_call.arguments_json, question

    if not errored:
        # The server accepted tool_choice but the model still did not
        # produce a usable call (e.g. it emitted only prose despite the
        # forced choice) -- no fallback request will do better here, so
        # give up on the rewrite for this turn.
        return None

    # Fallback path: tool_choice was rejected outright by the server.
    # Ask for a small constrained JSON object instead, via the OpenAI
    # -compatible ``response_format`` (json_schema) field, and synthesize
    # a tool call locally from the parsed result.
    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "research_queries",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "queries"],
                "properties": {
                    "question": {"type": "string", "maxLength": 300},
                    "queries": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 80},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                },
            },
        },
    }
    fallback_text_parts: list[str] = []
    for evt in llm.stream_chat(
        messages,
        cancel=cancel,
        max_tokens=_FORCED_REWRITE_MAX_TOKENS,
        response_format=schema_format,
    ):
        if evt.kind == "token":
            fallback_text_parts.append(evt.text or "")
        elif evt.kind == "error":
            return None

    raw = "".join(fallback_text_parts).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    queries = parsed.get("queries") if isinstance(parsed, dict) else None
    if not queries or not isinstance(queries, list):
        return None
    queries = [q for q in queries if isinstance(q, str) and q.strip()][:3]
    if not queries:
        return None
    fallback_question = parsed.get("question") if isinstance(parsed, dict) else None
    fallback_args: dict = {"queries": queries}
    if isinstance(fallback_question, str) and fallback_question.strip():
        fallback_args["question"] = fallback_question
    arguments_json = json.dumps(fallback_args)
    return "forced-rewrite-fallback", queries, arguments_json, fallback_question


# Short instruction appended after strong merged evidence on a follow-up
# turn's forced-rewrite round: answer the student's actual question
# directly first, then explain. Kept separate from ``_FOLLOWUP_HOST_NOTE``
# (which the model reads before it searches) so it lands right next to
# the evidence the model is about to answer from.
_FOLLOWUP_DIRECTNESS_NOTE = (
    "Answer the student's question as it relates to the lesson so far: "
    "give a direct answer first (yes or no, if it is a yes/no question), "
    "then explain using the sources above."
)

# Measured stronger variant (docs/followup_answer_shape.md, "enhanced"
# arm of data/followup_shape_measure.py): adds the "only what is new" /
# "1-3 sentences for a yes/no question" instructions the plain
# ``_FOLLOWUP_DIRECTNESS_NOTE`` was missing, which is what let the model
# fall back to re-emitting the same "Composition / Structure /
# Replication / Function" essay turn after turn regardless of the
# question. Gated by ``app.concise_followup_note`` (default False -- see
# docs/followup_answer_shape.md, "Iteration 2 (negative result)").
_FOLLOWUP_CONCISE_NOTE = (
    "Answer the student's question as it relates to the lesson so far: "
    "give a direct answer first (yes or no, in one clause, if it is a "
    "yes/no question), then add ONLY what is new -- do not restate points "
    "you already made earlier in this lesson. If the question only asks "
    "for a yes/no confirmation, 1-3 sentences total is enough; stop there "
    "rather than repeating the earlier explanation. If the question asks "
    "for a fuller explanation (e.g. \"tell me about X\", \"how does X "
    "work\"), still give a real explanation, not one line, citing sources "
    "like [S1]."
)


# "Iteration 3" fix candidate (docs/followup_answer_shape.md): restate
# the resolved standalone question as the LAST thing appended to the
# forced-rewrite round's evidence tool result, using the model's own
# rewritten query from that same round. Gated by
# ``app.restate_question_last`` (default True; adopted, see
# docs/followup_answer_shape.md "Decision (orchestrator)").
_RESTATE_QUESTION_R2_INSTRUCTION = (
    "If it is a yes/no question start with Yes or No; otherwise just "
    "answer it. Add what is new; do not repeat your earlier answer."
)


def _restate_question_line(original_text: str, meaning: str | None) -> str:
    """``meaning`` is a standalone restatement of the student's question --
    either the model's own ``question`` argument (``app.model_writes_search``
    mode) or, for the legacy follow-up-rewrite mode (where the rewritten
    queries ARE standalone questions, e.g. "Is DNA a molecule"), the first
    rewritten query. When ``meaning`` is falsy/absent the "(meaning: ...)"
    clause is omitted entirely rather than falling back to showing raw
    keyword search queries, which are not a restatement of the question."""
    if meaning:
        return (
            f'The student is now asking: "{original_text}" (meaning: {meaning}). '
            "Answer THIS question."
        )
    return f'The student is now asking: "{original_text}". Answer THIS question.'


def _has_prior_turns(session, use_log: bool, log, messages: list[dict] | None) -> bool:
    """True when the lesson already has at least one earlier turn, i.e.
    this is not the student's first message. Checked BEFORE this turn's
    own user message is appended."""
    if use_log:
        rendered = log.render()
        return any(m.get("role") == "user" for m in rendered)
    return any(m.get("role") == "user" for m in (messages or []))


def _run_forced_rewrite_round(
    *,
    llm,
    log,
    messages: list[dict],
    use_log: bool,
    cancel,
    user_input,
    session,
    research_engine,
    assessment,
    corrected_terms: dict,
    emit,
    backfill_passages: list[dict] | None = None,
    strong_suffix: str = "",
    reuse_prior_passages: bool = True,
    restate_question_text: str | None = None,
    restate_question_instruction: bool = False,
    clip_model_queries: bool = False,
    require_question_for_restate: bool = False,
    second_round: bool = False,
    timing: dict | None = None,
):
    """Run one forced ``research`` tool-call round (see
    ``_forced_research_tool_call``), append the resulting assistant
    tool-call + tool-result messages to the log/messages (append-only),
    and return ``(level_after, rewritten_queries, research_calls_delta)``.

    When ``backfill_passages`` is given, the rewrite's own passages LEAD
    the merged evidence and ``backfill_passages`` only fill remaining
    slots (``_lead_with_backfill``) -- used for a follow-up turn, where
    the raw pre-search ran on the question's literal (possibly
    wrong-topic) text. Otherwise the rewrite's own queries are merged
    against each other by RRF only (``_merge_dedupe_passages``), matching
    the original weak-evidence rewrite behaviour.

    ``require_question_for_restate``, when True (turn 1 under
    ``app.model_writes_search``, which has no prior turn to restate a
    question against otherwise), suppresses the restate line entirely
    unless the model's forced call also supplied a usable ``question``.
    """
    research_calls_delta = 0
    emit(
        {
            "kind": "status",
            "stage": "planning",
            "detail": (
                "Trying a different search..."
                if second_round
                else "Working out what to look up..."
            ),
        }
    )
    wire_messages = _to_wire_messages(log.render()) if use_log else messages
    _t0 = time.monotonic()
    forced = _forced_research_tool_call(llm, wire_messages, cancel=cancel)
    if timing is not None:
        timing["forced_call"] = timing.get("forced_call", 0.0) + (time.monotonic() - _t0)
    model_question: str | None = None
    if forced is not None:
        model_question = _clip_model_written_question(forced[3])
    if forced is not None and clip_model_queries:
        tool_call_id, raw_queries, _arguments_json, _raw_question = forced
        cleaned_queries = _clip_model_written_queries(raw_queries)
        if not cleaned_queries:
            # Nothing usable after validation/clipping -- fall back to
            # today's not-found behaviour exactly as if the model had
            # produced no usable call at all.
            forced = None
            model_question = None
        else:
            forced = (
                tool_call_id,
                cleaned_queries,
                json.dumps({"queries": cleaned_queries}),
                _raw_question,
            )
    elif forced is not None:
        # Legacy (non-model_writes_search) forced-rewrite modes never ask
        # the model for a standalone ``question`` -- even if one somehow
        # showed up, ignore it so the restate line keeps using the
        # rewritten query (already a standalone question in that mode),
        # matching today's behaviour exactly.
        model_question = None

    if forced is None:
        merged_response = _MergedResult([])
        new_assessment = assess_evidence(user_input.text, merged_response)
        level_after = new_assessment.level
        synth_id = "forced-rewrite-none"
        tool_call_message = {
            "id": synth_id,
            "type": "function",
            "function": {"name": "research", "arguments": json.dumps({"queries": []})},
        }
        if use_log:
            log.append_assistant_tool_calls([tool_call_message])
        else:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": [tool_call_message]}
            )
        tool_text = _not_found_tool_text(searched_for=[], level_after=level_after)
        if use_log:
            log.append_tool_result(tool_call_id=synth_id, content=tool_text)
        else:
            messages.append({"role": "tool", "tool_call_id": synth_id, "content": tool_text})
        emit({"kind": "tool_result", "name": "research", "ok": False})
        emit(
            {
                "kind": "status",
                "stage": "not_found",
                "detail": (
                    "Nothing in the library on this. Answering from what I know..."
                ),
            }
        )
        return level_after, [], research_calls_delta

    tool_call_id, rewritten_queries, arguments_json, _model_question_raw = forced
    emit(
        {
            "kind": "status",
            "stage": "searching",
            "detail": f"Searching the library for: {', '.join(rewritten_queries)}",
        }
    )
    if use_log:
        log.append_assistant_tool_calls(
            [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "research", "arguments": arguments_json},
                }
            ]
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "research", "arguments": arguments_json},
                    }
                ],
            }
        )

    _t_search0 = time.monotonic()
    many_fn = getattr(research_engine, "research_many", None)
    if many_fn is not None:
        batch_results = many_fn(
            rewritten_queries, topic_hint=getattr(session, "subject_hint", None)
        )
    else:
        batch_results = [
            {
                "query": q,
                "status": "ok",
                "response": research_engine.research(
                    q, topic_hint=getattr(session, "subject_hint", None)
                ),
                "error": None,
            }
            for q in rewritten_queries
        ]
    if timing is not None:
        timing["searches"] = timing.get("searches", 0.0) + (time.monotonic() - _t_search0)

    passage_lists = []
    for item in batch_results:
        sub_response = item.get("response")
        if item.get("status") != "ok" or sub_response is None:
            continue
        research_calls_delta += 1
        sub_packet = _packet_from_response(sub_response)
        passage_lists.append(sub_packet["passages"])
        sub_corrected = getattr(sub_response, "corrected_terms", None) or {}
        corrected_terms.update(dict(sub_corrected))

    lead_passages = _merge_dedupe_passages(
        passage_lists, _MERGE_CAP, rewritten_queries=rewritten_queries
    )
    if backfill_passages is not None:
        merged_passages = _lead_with_backfill(lead_passages, backfill_passages, _MERGE_CAP)
    else:
        merged_passages = lead_passages
    # Each passage's ``label`` was assigned independently by whichever
    # upstream retrieval call fetched it (each one numbers its own
    # results starting at "S1"), so merging several such calls' results
    # into one packet -- a rewrite's own lead passages plus the raw
    # pre-search's backfill passages -- can (and in practice does) land
    # the SAME "[S#]" label on two different passages in the same
    # packet. Relabel S1..Sn sequentially in final displayed order right
    # here, before anything renders or retains these passages, so every
    # downstream consumer (the rendered evidence text, the pointer lines
    # inside it, ``_retain_passages``, and the citation/attribution
    # lookup for this turn) sees one single, consistent, collision-free
    # label per passage.
    merged_passages = _relabel_sequential(merged_passages)
    merged_response = _MergedResult(merged_passages)
    healthy_terms = {
        t: True for t in getattr(assessment, "covered_terms", frozenset()) or []
    }
    new_assessment = assess_evidence(
        user_input.text,
        merged_response,
        rewritten_queries=rewritten_queries,
        healthy_terms=healthy_terms,
        corrected_terms=corrected_terms,
    )
    level_after = new_assessment.level

    _retain_passages(session, {"passages": merged_passages})
    searched_for_line = f"Searched for: {', '.join(rewritten_queries)}"
    if level_after == "strong":
        emit(
            {
                "kind": "status",
                "stage": "reading",
                "detail": f"Found {len(merged_passages)} passages. Reading them...",
            }
        )
        if use_log and reuse_prior_passages:
            # Route the forced round's own tool-result text through the
            # same pointer-substitution/held-id bookkeeping as
            # ``_trim_and_append_evidence`` (docs/passage_reuse.md), so a
            # passage already pasted in full earlier in this lesson --
            # including by this turn's own pre-search packet, if that was
            # appended before this round ran -- gets a one-line pointer
            # here instead of a full re-paste. This bypassed that
            # bookkeeping entirely before (see docs/passage_reuse.md's
            # "forced-rewrite rounds are not covered" note).
            evidence_text, newly_seen_ids, _display, _note = render_evidence_with_reuse(
                merged_passages,
                log.held_ids(),
                True,
                citation_reminder=_CITATION_REMINDER,
            )
            log.mark_held(newly_seen_ids)
        else:
            evidence_text = render_evidence({"passages": merged_passages})
        tool_text = f"{searched_for_line}\n{evidence_text}"
        if strong_suffix:
            tool_text = f"{tool_text}\n\n{strong_suffix}"
        if restate_question_text is not None and (
            not require_question_for_restate or model_question
        ):
            meaning = model_question if clip_model_queries else (
                rewritten_queries[0] if rewritten_queries else None
            )
            restate_line = _restate_question_line(restate_question_text, meaning)
            if restate_question_instruction:
                restate_line = f"{restate_line} {_RESTATE_QUESTION_R2_INSTRUCTION}"
            tool_text = f"{tool_text}\n\n{restate_line}"
    else:
        tool_text = (
            f"{searched_for_line}\n"
            + _not_found_tool_text(searched_for=rewritten_queries, level_after=level_after)
        )
    if use_log:
        log.append_tool_result(tool_call_id=tool_call_id, content=tool_text)
    else:
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_text})
    emit({"kind": "tool_result", "name": "research", "ok": level_after == "strong"})

    return level_after, rewritten_queries, research_calls_delta


def run_turn(
    session,
    user_input,
    *,
    llm,
    research_engine,
    calc,
    budget,
    emit,
    cancel: threading.Event | None = None,
    system_text_override: str | None = None,
    temperature: float | None = None,
    rewrite_on_weak_evidence: bool = True,
    rewrite_on_followup: bool = True,
    reuse_prior_passages: bool = True,
    concise_followup_note: bool = False,
    restate_question_last: bool = True,
    restate_question_instruction: bool = False,
    model_writes_search: bool = True,
) -> TurnResult:
    research_calls = 0
    timings = {
        "forced_call": 0.0,
        "searches": 0.0,
        "second_round": 0.0,
        "answer_prefill": 0.0,
        "answer_generation": 0.0,
    }
    calc_calls = 0
    followup_research_used = False
    events: list = []

    log = getattr(session, "log", None)
    use_log = log is not None and hasattr(log, "append_user")

    system_text = (
        system_text_override
        if system_text_override is not None
        else _load_default_system_text()
    )
    profile_summary = getattr(session, "profile_summary", None)
    if profile_summary:
        system_text = f"{system_text} {profile_summary}"

    if use_log:
        try:
            log.append_system(system_text)
        except ValueError:
            pass  # already appended on a prior turn of this session
    else:
        messages: list[dict] = [{"role": "system", "content": system_text}]

    evidence_summary: dict | None = None
    if user_input.kind == "action":
        route = "action"
        if use_log:
            log.append_user(f"[action:{user_input.action}]")
        else:
            messages.append({"role": "user", "content": f"[action:{user_input.action}]"})
    else:
        route = "preretrieve"
        has_prior_turns = _has_prior_turns(
            session, use_log, log, None if use_log else messages
        )
        emit({"kind": "status", "stage": "searching", "detail": "Looking in the library..."})
        response = research_engine.research(
            user_input.text, topic_hint=getattr(session, "subject_hint", None)
        )
        research_calls += 1
        packet = _packet_from_response(response)
        assessment = getattr(response, "assessment", None)
        level_before = assessment.level if assessment is not None else "strong"
        corrected_terms = dict(getattr(response, "corrected_terms", None) or {})

        # Follow-up rewrite (docs/rewrite_on_weak_evidence.md, "Follow-up
        # rewrite"): fires unconditionally on any turn after the first,
        # regardless of how strong the raw pre-search looks -- a student's
        # own grammar/spelling can't be relied on to signal an unresolved
        # reference, so this is NOT gated by any word-list/pronoun
        # detector. The raw pre-search still always runs (above) and its
        # passages are kept as backfill only.
        # ``model_writes_search`` (docs/rewrite_on_weak_evidence.md, "Model
        # writes every search") fires unconditionally on EVERY turn,
        # including turn 1, and replaces the follow-up rewrite on turn
        # >= 2 rather than running both.
        do_model_writes_search = model_writes_search
        do_followup = rewrite_on_followup and has_prior_turns and not do_model_writes_search

        if do_model_writes_search:
            user_text = user_input.text + _MODEL_WRITES_SEARCH_HOST_NOTE
        elif do_followup:
            user_text = user_input.text + _FOLLOWUP_HOST_NOTE
        elif rewrite_on_weak_evidence and level_before in ("weak", "empty"):
            user_text = user_input.text + _REWRITE_HOST_NOTE
        else:
            user_text = user_input.text
        if use_log:
            log.append_user(user_text)
        else:
            messages.append({"role": "user", "content": user_text})

        level_after = level_before
        rewritten_queries: list[str] = []
        followup_ran = False

        if do_model_writes_search or do_followup:
            # The raw pre-search's evidence is never appended directly on
            # a turn where the model writes its own search -- either it
            # ran on the question's literal text (turn 1) or on a
            # follow-up's literal, possibly wrong-topic text. It is only
            # ever used as backfill inside the forced round's own merge.
            # On turn 1 there is no prior turn to restate a question
            # against, so the restatement line only ever applies when
            # there ARE prior turns (matches the follow-up path).
            followup_level, followup_queries, delta = _run_forced_rewrite_round(
                llm=llm,
                log=log,
                messages=messages if not use_log else None,
                use_log=use_log,
                cancel=cancel,
                user_input=user_input,
                session=session,
                research_engine=research_engine,
                assessment=assessment,
                corrected_terms=corrected_terms,
                emit=emit,
                backfill_passages=packet["passages"],
                strong_suffix=(
                    ""
                    if do_model_writes_search and not has_prior_turns
                    else (
                        _FOLLOWUP_CONCISE_NOTE
                        if concise_followup_note
                        else _FOLLOWUP_DIRECTNESS_NOTE
                    )
                ),
                reuse_prior_passages=reuse_prior_passages,
                restate_question_text=(
                    user_input.text
                    if restate_question_last and (has_prior_turns or do_model_writes_search)
                    else None
                ),
                restate_question_instruction=restate_question_instruction,
                clip_model_queries=do_model_writes_search,
                require_question_for_restate=not has_prior_turns,
                timing=timings,
            )
            research_calls += delta
            # _run_forced_rewrite_round always leaves the log in a valid,
            # fully-closed state (assistant tool_calls + matching tool
            # result) even when the model produced no usable queries --
            # so once attempted, this turn's evidence has already been
            # decided one way or another; never fall through to a plain
            # packet append afterwards, only possibly a second (weak-
            # evidence) round below.
            followup_ran = True
            level_before = followup_level
            level_after = followup_level
            rewritten_queries = followup_queries

        do_rewrite = rewrite_on_weak_evidence and level_before in ("weak", "empty")

        if not followup_ran and not do_rewrite:
            _retain_passages(session, packet)
            emit(
                {
                    "kind": "status",
                    "stage": "reading",
                    "detail": f"Reading {len(packet.get('passages') or [])} sources...",
                }
            )
            if use_log:
                _trim_and_append_evidence(
                    log,
                    packet["passages"],
                    budget,
                    reuse_prior_passages=reuse_prior_passages,
                )
            else:
                evidence_text = render_evidence(packet)
                messages.append(
                    {"role": "tool", "content": f"[research results]\n{evidence_text}"}
                )
            emit({"kind": "tool_result", "name": "research"})
        elif do_rewrite:
            # One extra forced-rewrite round -- the original weak-evidence
            # mechanism, capped at one extra round even after a follow-up
            # round already ran (append-only, cache-safe: this is just
            # another tool round appended after whatever came before).
            _t_second0 = time.monotonic()
            second_round_timing: dict = {}
            level_after, rewritten_queries, delta = _run_forced_rewrite_round(
                llm=llm,
                log=log,
                messages=messages if not use_log else None,
                use_log=use_log,
                cancel=cancel,
                user_input=user_input,
                session=session,
                research_engine=research_engine,
                assessment=assessment,
                corrected_terms=corrected_terms,
                emit=emit,
                reuse_prior_passages=reuse_prior_passages,
                second_round=True,
                timing=second_round_timing,
            )
            timings["second_round"] = time.monotonic() - _t_second0
            research_calls += delta
        # else: followup_ran and not do_rewrite -- the follow-up round's
        # own strong result already fully handled this turn's evidence.

        if level_after in ("weak", "empty"):
            emit(
                {
                    "kind": "status",
                    "stage": "not_found",
                    "detail": (
                        "Nothing in the library on this. "
                        "Answering from what I know..."
                    ),
                }
            )

        evidence_summary = {
            "level_before": level_before,
            "level_after": level_after,
            "rewritten_queries": rewritten_queries,
            "corrected_terms": corrected_terms,
        }

    while True:
        if cancel is not None and cancel.is_set():
            return TurnResult(
                status="cancelled",
                answer_text="",
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
                timings=timings,
            )

        if use_log:
            eviction_event = log.evict(budget)
            if eviction_event is not None:
                events.append(eviction_event)
                emit(
                    {
                        "kind": "eviction_reprefill",
                        "evicted_turns": eviction_event.evicted_turns,
                        "tokens_before": eviction_event.tokens_before,
                        "tokens_after": eviction_event.tokens_after,
                        "dropped_uncited_passages": eviction_event.dropped_uncited_passages,
                        "dropped_uncited_tokens": eviction_event.dropped_uncited_tokens,
                    }
                )
            messages = _to_wire_messages(log.render())

        answer_text_parts: list[str] = []
        tool_calls: list = []
        finish_reason = "stop"
        usage: dict = {}
        errored = False
        truncated: str | None = None
        trimmed_answer_text: str | None = None
        # The guard's own cancel signal, ORed with the caller's `cancel`
        # (if any) so LlamaClient.stream_chat closes the connection the
        # same clean way a user-initiated stop does (see _UnionCancel).
        guard_cancel = threading.Event()
        stream_cancel = _UnionCancel(cancel, guard_cancel)

        emit({"kind": "status", "stage": "thinking", "detail": "Writing an answer..."})
        _t_request_sent = time.monotonic()
        _first_token_time: float | None = None
        for evt in llm.stream_chat(
            messages,
            tools=TOOLS,
            cancel=stream_cancel,
            temperature=temperature,
            max_tokens=budget.generation,
        ):
            if evt.kind == "token":
                if _first_token_time is None:
                    _first_token_time = time.monotonic()
                    timings["answer_prefill"] += _first_token_time - _t_request_sent
                answer_text_parts.append(evt.text or "")
                emit(evt)
                if truncated is None:
                    loop = find_repetition_loop("".join(answer_text_parts))
                    if loop is not None:
                        truncated = "repetition"
                        trimmed_answer_text = loop.trimmed_text
                        guard_cancel.set()
            elif evt.kind == "tool_call":
                tool_calls.append(evt)
                emit(evt)
            elif evt.kind == "error":
                errored = True
                emit(evt)
                break
            elif evt.kind == "done":
                finish_reason = evt.finish_reason or "stop"
                usage = evt.usage or {}
                if finish_reason == "length" and truncated is None:
                    truncated = "max_tokens"

        if _first_token_time is not None:
            timings["answer_generation"] += time.monotonic() - _first_token_time

        if errored:
            return TurnResult(
                status="error",
                answer_text=_STUDENT_SAFE_ERROR,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
                timings=timings,
            )

        # A "cancelled" finish_reason from the guard's own union event
        # (not the caller's `cancel`) is not a real user cancellation --
        # it's the repetition guard stopping generation cleanly. Only
        # treat it as a cancelled turn when the caller actually asked to
        # cancel.
        if finish_reason == "cancelled" and (cancel is None or not cancel.is_set()):
            finish_reason = "stop"

        if finish_reason == "cancelled":
            return TurnResult(
                status="cancelled",
                answer_text="".join(answer_text_parts),
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
                timings=timings,
            )

        if not tool_calls:
            answer_text = (
                trimmed_answer_text if truncated == "repetition" else "".join(answer_text_parts)
            )
            labels = extract_labels(answer_text)
            if use_log:
                log.append_assistant(answer_text, cited_labels=labels)
            uncited = route.startswith("preretrieve") and not labels
            return TurnResult(
                status="ok",
                answer_text=answer_text,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                cached_tokens=usage.get("cached_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens"),
                events=events,
                uncited=uncited,
                truncated=truncated,
                evidence=evidence_summary,
                timings=timings,
            )

        # Record the assistant's tool-call turn, then dispatch each call.
        wire_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments_json,
                },
            }
            for tc in tool_calls
        ]
        if use_log:
            log.append_assistant_tool_calls(wire_tool_calls)
        else:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": wire_tool_calls}
            )

        def _append_tool_message(tool_call_id: str, content: str, _messages=messages) -> None:
            if use_log:
                log.append_tool_result(tool_call_id=tool_call_id, content=content)
            else:
                _messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                )

        turn_cancelled = False
        for index, tc in enumerate(tool_calls):
            # Review pass 2, finding 1: once the assistant's tool_calls
            # message has been appended (above), every id it names MUST get
            # a matching tool-result entry -- on a raised exception, on a
            # cancel mid-loop, or on normal dispatch -- or the log is left
            # with a dangling tool_calls message that the next turn cannot
            # replay (OpenAI-compatible endpoints reject it, wedging the
            # session/lesson permanently). So this loop body never lets an
            # exception propagate past a single tool call, and a cancel
            # mid-loop fills in the remaining not-yet-dispatched ids with a
            # synthetic "cancelled" result before returning, instead of
            # leaving them unfilled.
            if cancel is not None and cancel.is_set():
                turn_cancelled = True
                for remaining_tc in tool_calls[index:]:
                    _append_tool_message(
                        remaining_tc.id,
                        json.dumps({"ok": False, "error": "turn cancelled"}),
                    )
                    emit({"kind": "tool_result", "name": remaining_tc.name, "ok": False})
                break

            try:
                validation = validate_tool_call(tc.name, tc.arguments_json or "{}")
                if not validation.ok:
                    _append_tool_message(tc.id, f"error: invalid tool call: {validation.error}")
                    emit({"kind": "tool_result", "name": tc.name, "ok": False})
                    continue

                if tc.name == "research":
                    emit(
                        {
                            "kind": "status",
                            "stage": "tool",
                            "detail": "Searching again...",
                        }
                    )
                    if research_calls >= RESEARCH_CAP:
                        _append_tool_message(
                            tc.id,
                            f"research cap of {RESEARCH_CAP} calls reached for this "
                            "turn; answer with the supported portion or ask for "
                            "clarification.",
                        )
                    else:
                        response = research_engine.research(
                            validation.arguments["query"],
                            topic_hint=getattr(session, "subject_hint", None),
                            keywords=validation.arguments.get("keywords"),
                        )
                        research_calls += 1
                        followup_research_used = True
                        packet = _packet_from_response(response)
                        _retain_passages(session, packet)
                        if use_log:
                            _trim_and_append_evidence(
                                log,
                                packet["passages"],
                                budget,
                                reuse_prior_passages=reuse_prior_passages,
                            )
                        else:
                            evidence_text = render_evidence(packet)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"[research results]\n{evidence_text}",
                                }
                            )
                    emit({"kind": "tool_result", "name": "research"})
                elif tc.name == "calc":
                    emit(
                        {
                            "kind": "status",
                            "stage": "tool",
                            "detail": "Using the calculator...",
                        }
                    )
                    if calc_calls >= CALC_CAP:
                        _append_tool_message(
                            tc.id, f"calc cap of {CALC_CAP} calls reached for this turn."
                        )
                    else:
                        calc_result = calc.evaluate(validation.arguments["expression"])
                        calc_calls += 1
                        _append_tool_message(tc.id, json.dumps(calc_result))
                    emit({"kind": "tool_result", "name": "calc"})
                else:
                    # Unreachable: validate_tool_call already rejects unknown
                    # tool names, but keep the loop total in case of future
                    # tool additions with no dispatch branch yet.
                    _append_tool_message(tc.id, f"error: no dispatcher for tool {tc.name}")
            except Exception as exc:  # noqa: BLE001 - never leave the log dangling
                # research/calc (or evidence trimming) raised -- e.g. a
                # ZimWorker timeout not swallowed at a lower layer. Give
                # this call's id a student-safe, model-safe error result
                # instead of letting the exception propagate: propagating
                # here would leave `log` with an assistant tool_calls
                # message and no result for this id, which is exactly the
                # corruption this fix exists to prevent. Every dispatch
                # branch above only appends its tool message *after*
                # successfully computing a result, so reaching this except
                # means no message for `tc.id` has been appended yet.
                _append_tool_message(
                    tc.id, json.dumps({"ok": False, "error": str(exc)})
                )
                emit({"kind": "tool_result", "name": tc.name, "ok": False})

        if use_log:
            log.validate()

        if turn_cancelled:
            return TurnResult(
                status="cancelled",
                answer_text="".join(answer_text_parts),
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
                timings=timings,
            )

        if followup_research_used and route == "preretrieve":
            route = "preretrieve+followup"
