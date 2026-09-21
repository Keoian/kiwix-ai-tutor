"""Unit tests for tutor.retrieval.morph.morph_variants (pure function, no
worker/archive access -- see tutor/retrieval/morph.py for the rules)."""

from __future__ import annotations

from tutor.retrieval.morph import morph_variants


def test_never_includes_input_term():
    for term in ("garden", "gardening", "run", "running", "erupt", "eruption"):
        assert term not in morph_variants(term)


def test_deduplicated():
    for term in ("garden", "running", "freeze", "dinosaur"):
        variants = morph_variants(term)
        assert len(variants) == len(set(variants))


def test_max_eight_variants():
    for term in ("garden", "run", "freeze", "erupt", "evaporate"):
        assert len(morph_variants(term)) <= 8


def test_no_short_variants():
    for term in ("garden", "run", "freeze", "erupt"):
        assert all(len(v) >= 3 for v in morph_variants(term))


def test_shortest_edit_first():
    variants = morph_variants("garden")
    deltas = [abs(len(v) - len("garden")) for v in variants]
    assert deltas == sorted(deltas)


def test_no_explosion_on_short_words():
    for term in ("gas", "bus", "is"):
        assert len(morph_variants(term)) <= 8


PAIRS_FORWARD = [
    ("garden", "gardens"),
    ("garden", "gardening"),
    ("run", "running"),
    ("run", "runs"),
    ("freeze", "freezing"),
    ("freeze", "freezes"),
    ("dinosaur", "dinosaurs"),
    ("erupt", "erupted"),
    ("erupt", "eruption"),
    ("evaporate", "evaporation"),
    ("baby", "babies"),
    ("box", "boxes"),
    ("bus", "buses"),
    ("quick", "quickly"),
    ("happy", "happily"),
    ("bake", "baked"),
    ("bake", "baker"),
    ("plant", "planted"),
    ("plant", "planting"),
    ("govern", "government"),
]

PAIRS_REVERSE = [
    ("gardens", "garden"),
    ("gardening", "garden"),
    ("running", "run"),
    ("runs", "run"),
    ("freezing", "freeze"),
    ("freezes", "freeze"),
    ("dinosaurs", "dinosaur"),
    ("erupted", "erupt"),
    ("eruption", "erupt"),
    ("evaporation", "evaporate"),
    ("babies", "baby"),
    ("boxes", "box"),
    ("quickly", "quick"),
    ("happily", "happy"),
    ("baked", "bake"),
    ("planted", "plant"),
    ("planting", "plant"),
    ("government", "govern"),
]


def test_forward_pairs():
    missing = []
    for base, expected in PAIRS_FORWARD:
        if expected not in morph_variants(base):
            missing.append((base, expected))
    assert not missing, missing


def test_reverse_pairs():
    missing = []
    for inflected, expected in PAIRS_REVERSE:
        if expected not in morph_variants(inflected):
            missing.append((inflected, expected))
    assert not missing, missing


def test_negative_no_bogus_variant_for_gas():
    # "gas" is short and should not explode into nonsense via the -e drop
    # rule etc; just make sure it doesn't crash and stays small/plausible.
    variants = morph_variants("gas")
    assert "ga" not in variants


def test_negative_short_word_not_shortened_below_floor():
    for term in ("run", "bus", "gas"):
        for v in morph_variants(term):
            assert len(v) >= 3


def test_empty_for_too_short_input():
    assert morph_variants("is") == []
    assert morph_variants("a") == []
