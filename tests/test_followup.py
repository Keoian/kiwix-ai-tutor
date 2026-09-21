"""Tests for tutor.app.followup.is_elliptical_followup (pure detector, no
I/O). See docs/rewrite_on_weak_evidence.md, "Elliptical follow-up
rewrite"."""

from __future__ import annotations

import pytest

from tutor.app.followup import is_elliptical_followup


@pytest.mark.parametrize(
    "question",
    [
        "Is it a molecule?",
        "What about DNA?",
        "Why?",
        "How big are they?",
    ],
)
def test_positive_elliptical_followups(question):
    assert is_elliptical_followup(question, has_prior_turns=True) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is photosynthesis?",
        "Is DNA a molecule?",
    ],
)
def test_negative_standalone_questions(question):
    assert is_elliptical_followup(question, has_prior_turns=True) is False


def test_first_turn_never_elliptical_even_with_pronoun():
    assert is_elliptical_followup("Is it raining?", has_prior_turns=False) is False
