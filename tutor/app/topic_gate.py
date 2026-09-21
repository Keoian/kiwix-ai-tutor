"""Host topic gate (docs/rewrite_on_weak_evidence.md, "Host topic gate").

Pure, no-I/O text classification the host runs BEFORE any LLM call and
BEFORE the raw pre-search, so the decision to decline a message never
depends on a model choosing to decline -- see the owner's stated reason
(HANDOFF/task context, 2026-09-21): a small local model cannot reliably be
talked into declining a jailbreak-wrapped sexual-content request, so the
decline must be decided by host code.

``classify_message`` returns one of:

* ``"decline"`` -- explicit sexual content, or a clinical anatomy/sex term
  combined with a record/measurement/sensational cue (see
  ``_EXPLICIT_TERMS`` / ``_SENSITIVE_TERMS`` / ``_has_record_cue`` below).
  Role-play/jailbreak wrapper text ("pretend you're a doctor", "for a
  school project", "ignore your rules") never changes this verdict --
  classification runs on the content terms wherever they appear in the
  message.
* ``"chat"`` -- no real content terms at all (thanks/greetings/acks).
  Conservative: anything not confidently chatter falls through to
  ``"normal"`` (a missed chat only costs one wasted search).
* ``"normal"`` -- everything else; today's behaviour, untouched.

Known accepted false positive (owner-approved, see docs/
rewrite_on_weak_evidence.md): "How long is a blue whale's penis?" declines
under rule (b) even though it is a genuine biology question, because it
combines a sensitive-anatomy term with a superlative/measure cue exactly
like the unsafe case does. The gate cannot tell those apart without more
context than a single message provides, and the owner accepted trading
this rare false positive for reliably declining the unsafe case.
"""

from __future__ import annotations

import re

from tutor.retrieval.hybrid.lexical import question_modifier_terms

# ---------------------------------------------------------------------------
# Term lists. Both lists below are DELIBERATELY SMALL and owner-editable --
# see the module docstring and the owner's stated policy. Do not "complete"
# these lists with every synonym you can think of; a short, reviewable list
# the owner can audit and extend is the point, not exhaustive coverage.
# ---------------------------------------------------------------------------

# Explicit sexual-content terms: presence of ANY of these, anywhere in the
# message (even inside a role-play/jailbreak wrapper), is enough on its own
# to decline. These are never words a clinical biology question needs.
_EXPLICIT_TERMS: frozenset[str] = frozenset(
    {
        "porn",
        "porno",
        "pornography",
        "pornographic",
        "erotic",
        "erotica",
        "sex story",
        "sex stories",
        "nude",
        "nudes",
        "naked",
        "naked pictures",
        "nude pictures",
        "nude photos",
        "fetish",
        "blowjob",
        "handjob",
        "cumshot",
        "anal sex",
        "cunnilingus",
        "fellatio",
        "cock",
        "cum",
        "horny",
        "sexy",
        "slut",
        "whore",
    }
)

# Clinical anatomy/sex terms: fine on their own ("What is a penis?"), only
# a decline trigger when paired with a record/measurement/sensational cue
# (rule (b) below).
_SENSITIVE_TERMS: frozenset[str] = frozenset(
    {
        "penis",
        "vagina",
        "vulva",
        "breast",
        "breasts",
        "boobs",
        "testicle",
        "testicles",
        "genital",
        "genitals",
        "sex",
        "orgasm",
        "erection",
        "masturbation",
        "masturbate",
        "condom",
        "intercourse",
        "ejaculate",
        "ejaculation",
    }
)

# Extra record/measurement/sensational cue PHRASES not already covered by
# ``question_modifier_terms`` (which gives us "longest"/"biggest"/... and
# "how long/big/tall/..."). Whole-phrase, case-insensitive.
_EXTRA_CUE_PHRASES: frozenset[str] = frozenset(
    {
        "world record",
        "average size",
        "how large",
        "how to",
        "pictures of",
        "show me",
        "hottest",
        "sexiest",
        "inches",
        "size",
    }
)

# Chatter tokens: a message reduced to nothing but these (after lower-
# casing and stripping punctuation/emoji) is "chat", not "normal".
_CHAT_WORDS: frozenset[str] = frozenset(
    {
        "ok",
        "okay",
        "k",
        "lol",
        "haha",
        "thanks",
        "thankyou",
        "thx",
        "ty",
        "cool",
        "nice",
        "wow",
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "bye",
        "hi",
        "hello",
        "hey",
        "gotit",
        "isee",
        "oh",
    }
)

# Simple leetspeak / substitution normalization -- tolerant of "p0rn",
# "s3x", "$exy" etc. Deliberately small: this is meant to defeat the most
# obvious character substitutions, not act as a full obfuscation decoder.
_LEET_MAP = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a"}
)

# Collapses spaced-out single letters ("s e x y" -> "sexy", "p o r n" ->
# "porn") so simple spacing tricks don't defeat whole-word matching below.
_SPACED_LETTERS_RE = re.compile(r"\b(?:[a-z]\s+){2,}[a-z]\b")


def _normalize(text: str) -> str:
    t = text.lower().translate(_LEET_MAP)
    t = _SPACED_LETTERS_RE.sub(lambda m: m.group(0).replace(" ", ""), t)
    return t


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", normalized) is not None


def _has_record_cue(text: str, normalized: str) -> bool:
    if question_modifier_terms(text):
        return True
    return any(_contains_phrase(normalized, p) for p in _EXTRA_CUE_PHRASES)


def classify_message(text: str) -> str:
    """Classify a single student message. Pure function, no I/O.

    Returns ``"decline"``, ``"chat"``, or ``"normal"``. See module
    docstring for the full decision rules.
    """
    if text is None or not text.strip():
        return "normal"

    normalized = _normalize(text)

    if any(_contains_phrase(normalized, t) for t in _EXPLICIT_TERMS):
        return "decline"

    if any(_contains_phrase(normalized, t) for t in _SENSITIVE_TERMS):
        if _has_record_cue(text, normalized):
            return "decline"

    stripped = re.sub(r"[^a-z\s']", " ", normalized)
    stripped = (
        stripped.replace("thank you", "thankyou")
        .replace("got it", "gotit")
        .replace("i see", "isee")
    )
    tokens = stripped.split()
    if tokens and all(tok in _CHAT_WORDS for tok in tokens):
        return "chat"

    return "normal"


def classify_reason(text: str) -> tuple[str, str]:
    """Like ``classify_message`` but also returns a short, structured
    reason string for the host's own logging -- e.g. ``"explicit_term"``
    or ``"sensitive_plus_cue"`` -- NEVER the matched term or the term
    list itself (see docs/rewrite_on_weak_evidence.md, "Host topic
    gate": the owner wants to know which rule fired, not a log full of
    the explicit-term list)."""
    if text is None or not text.strip():
        return "normal", "empty"

    normalized = _normalize(text)

    if any(_contains_phrase(normalized, t) for t in _EXPLICIT_TERMS):
        return "decline", "explicit_term"

    if any(_contains_phrase(normalized, t) for t in _SENSITIVE_TERMS):
        if _has_record_cue(text, normalized):
            return "decline", "sensitive_plus_cue"

    verdict = classify_message(text)
    if verdict == "chat":
        return "chat", "chatter_only"
    return "normal", "no_trigger"


# Fixed, owner-editable host reply for a declined message. Streamed as the
# tutor's own answer -- never an error, never a "not found" note.
DECLINE_REPLY: str = (
    "That's not something I can help with. If you're curious about how the "
    "body works, I can look up the biology for you — and a parent or "
    "trusted adult is a good person to ask too."
)
