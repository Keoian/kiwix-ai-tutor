"""Host-side detector for an elliptical follow-up question: one that only
makes sense read against the lesson so far (an unresolved pronoun/
demonstrative reference, or too few content words of its own to search on
directly).

Pure, no I/O: ``is_elliptical_followup`` takes the raw question text and
whether the lesson has any prior turns, and returns a bool. See
docs/rewrite_on_weak_evidence.md ("Elliptical follow-up rewrite") for how
the host uses this to force a query-rewrite round instead of trusting the
raw pre-search, which otherwise retrieves whatever article the dangling
pronoun happens to name literally (e.g. "Is it a molecule?" retrieving the
"Molecule" article instead of "Is DNA a molecule?").
"""

from __future__ import annotations

import re

from tutor.retrieval.hybrid.lexical import QUESTION_SHAPE_FILLERS, _stopwords, tokenize

# Pronouns/demonstratives whose referent can only be resolved from the
# lesson so far -- never meaningful search terms on their own.
_UNRESOLVED_REFERENCE_WORDS = frozenset(
    {
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "that",
        "this",
        "those",
        "these",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "one",
    }
)

# Sentence openers that only ever introduce a follow-up tied to whatever
# was just discussed, regardless of what (if anything) follows them.
_FOLLOWUP_OPENER_RE = re.compile(
    r"^\s*(what about|how about|why is that|why's that|why|and)\b", re.IGNORECASE
)

# A well-formed "What is X?"/"Who is X?"-shaped definitional question names
# its own topic in full and is never elliptical on term count alone (only
# the pronoun/opener rules above can still flag it, e.g. "What is it?").
# Without this exclusion the <=1-content-term fallback below would
# misfire on ordinary standalone questions like "What is photosynthesis?"
# whose only content word is the topic noun itself.
_WELL_FORMED_DEFINITION_RE = re.compile(
    r"^\s*(what|who|which|where|when)\s+(is|are|was|were)\b", re.IGNORECASE
)


def _content_terms(question: str) -> list[str]:
    """Tokens left after stripping stopwords and the retrieval layer's own
    question-shape filler words (``QUESTION_SHAPE_FILLERS``,
    tutor/retrieval/hybrid/lexical.py) -- the same filtering used to judge
    whether a query string carries enough of its own topic content."""
    stop = _stopwords()
    terms = []
    for tok in tokenize(question):
        low = tok.lower()
        if low in stop or low in QUESTION_SHAPE_FILLERS or low in _UNRESOLVED_REFERENCE_WORDS:
            continue
        terms.append(low)
    return terms


def is_elliptical_followup(question: str, has_prior_turns: bool) -> bool:
    """True only when ``has_prior_turns`` and the question either contains
    an unresolved pronoun/demonstrative reference, opens with a bare
    follow-up phrase ("what about", "and", "why", ...), or has at most one
    content term of its own once stopwords/filler are removed."""
    if not has_prior_turns:
        return False
    if not question or not question.strip():
        return False

    tokens = [t.lower() for t in tokenize(question)]
    if any(t in _UNRESOLVED_REFERENCE_WORDS for t in tokens):
        return True
    if _FOLLOWUP_OPENER_RE.match(question):
        return True
    if not _WELL_FORMED_DEFINITION_RE.match(question) and len(_content_terms(question)) <= 1:
        return True
    return False
