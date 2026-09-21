"""Dependency-free morphological variant generation for a single word.

Purely lexical/rule-based: no wordlist, no stemmer library, no network or
archive access. Used by ``tutor.retrieval.research`` to widen a weak/empty
lexical search (e.g. the query says "square foot garden" but the article is
titled "Square foot gardening") -- see that module's ``_expand_morph_variants``
for how candidates returned here are filtered by real corpus-wide
``estimated_matches`` before ever being used in a query.
"""

from __future__ import annotations

_MIN_VARIANT_LEN = 3
_MAX_VARIANTS = 8

# Doubled-final-consonant set for the "-ing"/"-ed" reverse rules (run ->
# running, running -> run): only consonants that commonly double in English
# CVC monosyllables/stems are considered, to avoid absurd reversals of words
# that merely end in a doubled letter for other reasons (e.g. "add").
_DOUBLING_CONSONANTS = set("bdgklmnprt")
_VOWELS = set("aeiou")


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _forward_variants(term: str) -> list[str]:
    """Base word -> inflected/derived forms."""
    out: list[str] = []
    n = len(term)

    # --- plural (singular -> plural) ---
    if term.endswith("y") and n > 1 and term[-2] not in _VOWELS:
        out.append(term[:-1] + "ies")
    elif term.endswith(("s", "x", "z", "ch", "sh")):
        out.append(term + "es")
    else:
        out.append(term + "s")

    # --- -ing ---
    if term.endswith("e") and not term.endswith("ee") and n > 2:
        out.append(term[:-1] + "ing")  # freeze -> freezing
    elif (
        3 <= n <= 5
        and term[-1] in _DOUBLING_CONSONANTS
        and term[-2] in _VOWELS
        and term[-3] not in _VOWELS
    ):
        out.append(term + term[-1] + "ing")  # run -> running
    else:
        out.append(term + "ing")

    # --- -ed ---
    if term.endswith("e") and n > 2:
        out.append(term + "d")
    elif (
        3 <= n <= 5
        and term[-1] in _DOUBLING_CONSONANTS
        and term[-2] in _VOWELS
        and term[-3] not in _VOWELS
    ):
        out.append(term + term[-1] + "ed")
    elif term.endswith("y") and n > 1 and term[-2] not in _VOWELS:
        out.append(term[:-1] + "ied")
    else:
        out.append(term + "ed")

    # --- -er ---
    if term.endswith("e") and n > 2:
        out.append(term + "r")
    else:
        out.append(term + "er")

    # --- -ion / -ation / -tion ---
    if term.endswith("t"):
        out.append(term + "ion")  # erupt -> eruption
    elif term.endswith("e") and n > 2:
        out.append(term[:-1] + "ion")  # evaporate -> evaporation
        out.append(term[:-1] + "ation")
    else:
        out.append(term + "ation")

    # --- -ment ---
    out.append(term + "ment")

    # --- -ly ---
    if term.endswith("y") and n > 1 and term[-2] not in _VOWELS:
        out.append(term[:-1] + "ily")
    else:
        out.append(term + "ly")

    return out


def _reverse_variants(term: str) -> list[str]:
    """Inflected/derived form -> plausible base word(s)."""
    out: list[str] = []
    n = len(term)

    # --- plural -> singular ---
    if term.endswith("ies") and n > 4:
        out.append(term[:-3] + "y")  # gardenies-like -> gardeny (harmless)
    if term.endswith("es") and n > 4:
        out.append(term[:-2])
    if term.endswith("s") and not term.endswith("ss") and n > 3:
        out.append(term[:-1])

    # --- -ing -> base ---
    if term.endswith("ing") and n > 5:
        stem = term[:-3]
        # doubled consonant: running -> run
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] in _DOUBLING_CONSONANTS:
            out.append(stem[:-1])
        else:
            out.append(stem)
            out.append(stem + "e")  # freezing -> freeze

    # --- -ed -> base ---
    if term.endswith("ied") and n > 5:
        out.append(term[:-3] + "y")
    elif term.endswith("ed") and n > 4:
        stem = term[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] in _DOUBLING_CONSONANTS:
            out.append(stem[:-1])
        else:
            out.append(stem)
            out.append(stem + "e")

    # --- -er -> base ---
    if term.endswith("er") and n > 4:
        out.append(term[:-2])
        out.append(term[:-1])

    # --- -ation / -tion / -ion -> base (erupt/eruption, evaporate/evaporation) ---
    if term.endswith("ation") and n > 6:
        out.append(term[:-3] + "e")  # evaporation -> evaporate
        out.append(term[:-5])
    if term.endswith("tion") and n > 5:
        out.append(term[:-3])  # eruption -> erupt
    if term.endswith("ion") and n > 4 and not term.endswith("tion"):
        out.append(term[:-3])
        out.append(term[:-3] + "e")

    # --- -ment -> base ---
    if term.endswith("ment") and n > 5:
        out.append(term[:-4])

    # --- -ly -> base ---
    if term.endswith("ily") and n > 4:
        out.append(term[:-3] + "y")
    elif term.endswith("ly") and n > 3:
        out.append(term[:-2])

    return out


def morph_variants(term: str) -> list[str]:
    """Dependency-free morphological variants of ``term``, both directions.

    Returns a de-duplicated list WITHOUT ``term`` itself, shortest-edit
    (by length delta from ``term``) candidates first, at most
    ``_MAX_VARIANTS`` entries, never a variant shorter than
    ``_MIN_VARIANT_LEN`` characters. Pure/dependency-free: no lookups, no
    I/O -- callers (see ``research._expand_morph_variants``) are
    responsible for verifying a candidate is a real word in the target
    corpus (via ``estimated_matches``) before using it in a query.
    """
    term = term.lower().strip()
    if len(term) < _MIN_VARIANT_LEN:
        return []

    candidates = [*_forward_variants(term), *_reverse_variants(term)]
    candidates = [
        c for c in _dedupe_ordered(candidates) if c != term and len(c) >= _MIN_VARIANT_LEN
    ]
    candidates.sort(key=lambda c: abs(len(c) - len(term)))
    return candidates[:_MAX_VARIANTS]
