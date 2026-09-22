"""Tests that the tutor never tells a student (who has no internet) to
look something up outside the offline Kiwix library.

Owner-reported, live, 2026-09-21: the tutor told a student to "look it up
on the official Arduino website." The only resource the student has is
this app and its offline library, so every model-facing instruction that
used to say "look up next" / "look up hypothermia or Antarctica" must
instead point at a related topic *in this library* to ask the tutor
about, or at a teacher/parent -- never an external website, search
engine, app, or "the internet."

Covers:
- ``system_prompt.txt`` (via ``build_system_text``) carries an explicit
  no-internet rule.
- ``_NAMES_NUMBERS_SECTION`` and both branches of ``_not_found_tool_text``
  (the model-facing host notes on a not-found search) never say "look up
  next", "website", "online", or "internet".
"""

from __future__ import annotations

from tutor.app.agent_loop import (
    _NAMES_NUMBERS_SECTION,
    _not_found_tool_text,
    build_system_text,
)


def _normalize(text: str) -> str:
    """Collapse whitespace (including the fixed line-wrap in
    system_prompt.txt) so a rule can be matched as one logical sentence
    regardless of exactly where its lines wrap."""
    return " ".join(text.split())


# The rule added to system_prompt.txt's "Sourcing and citations" section
# -- the one place "website"/"search engine"/"app" are allowed to appear,
# since it is the rule *forbidding* sending the student to one. Matched
# whitespace-insensitively against the assembled prompt (see
# ``_normalize``) so it is not coupled to the file's exact line-wrap.
_NO_INTERNET_RULE = _normalize(
    """The student has no internet. Never send them to a website, search
    engine, or app outside this library. If the library lacks it, say so
    and suggest a related topic to ask about, or a teacher or parent."""
)

_FORBIDDEN = ("look up next", "website", "online", "internet")


def test_assembled_system_prompt_contains_no_internet_rule():
    text = _normalize(build_system_text())
    assert _NO_INTERNET_RULE in text


def test_assembled_system_prompt_has_no_forbidden_phrases_outside_the_rule():
    text = _normalize(build_system_text())
    remainder = text.replace(_NO_INTERNET_RULE, "")
    assert "look up next" not in remainder
    assert "website" not in remainder
    assert "online" not in remainder
    # "internet" also appears once in the mission-statement opening line
    # ("...with no internet access...") -- that is a true, harmless
    # statement of fact about the student's situation, not an
    # instruction to go online, so it is allowed to remain exactly once.
    assert remainder.count("internet") <= 1


def test_names_numbers_section_has_no_forbidden_phrases():
    for phrase in _FORBIDDEN:
        assert phrase not in _NAMES_NUMBERS_SECTION, phrase


def test_names_numbers_section_suggests_a_library_topic_not_a_lookup():
    assert "ask" in _NAMES_NUMBERS_SECTION
    assert "hypothermia or Antarctica" in _NAMES_NUMBERS_SECTION


def test_not_found_tool_text_empty_branch_has_no_forbidden_phrases():
    text = _not_found_tool_text(
        searched_for=[], level_after="empty", no_specifics_without_source=True
    )
    for phrase in _FORBIDDEN:
        assert phrase not in text, phrase
    assert "ask about next" in text


def test_not_found_tool_text_weak_branch_has_no_forbidden_phrases():
    text = _not_found_tool_text(
        searched_for=[], level_after="weak", no_specifics_without_source=True
    )
    for phrase in _FORBIDDEN:
        assert phrase not in text, phrase


def test_not_found_tool_text_old_bytes_branch_has_no_forbidden_phrases():
    for level in ("weak", "empty"):
        text = _not_found_tool_text(
            searched_for=[], level_after=level, no_specifics_without_source=False
        )
        for phrase in _FORBIDDEN:
            assert phrase not in text, phrase
