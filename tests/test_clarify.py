"""Unit tests for tutor.app.clarify: pure parsing/classification helpers
behind the host-driven "Did you mean X?" clarify feature (owner request
2026-09-21). No LLM, no retrieval -- see tests/test_agent_loop_clarify.py
for the wired-up behaviour."""

from __future__ import annotations

from tutor.app.clarify import classify_reply, parse_candidate


def test_parse_candidate_bar_form():
    result = parse_candidate(
        "Arduino | A small single-board computer used for building electronics"
    )
    assert result == (
        "Arduino",
        "a small single-board computer used for building electronics",
    )


def test_parse_candidate_description_never_ends_on_a_dangling_word():
    # Live Ling output clipped at 8 words read "...used for building
    # electronic?" -- the clip must back off to a whole phrase.
    name, description = parse_candidate(
        "Arduino | A small computer used for building electronic projects."
    )
    assert description == "a small computer used for building electronic projects"
    name, description = parse_candidate("Arduino | A small computer that is used for")
    assert description == "a small computer"
    # Acronyms keep their case.
    assert parse_candidate("LED | LED light source.")[1] == "LED light source"


def test_parse_candidate_bar_form_clips_long_description():
    result = parse_candidate(
        "Arduino | one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    name, description = result
    assert name == "Arduino"
    assert len(description.split()) == 12


def test_parse_candidate_no_bar_short_line():
    assert parse_candidate("Zorblax") == ("Zorblax", "")


def test_parse_candidate_no_bar_quoted_string():
    assert parse_candidate('The student likely meant "dinosaur"') == ("dinosaur", "")


def test_parse_candidate_long_sentence_is_none():
    assert (
        parse_candidate(
            "I think the student probably meant something related to small "
            "computers used for electronics projects and robotics kits"
        )
        is None
    )


def test_parse_candidate_empty_is_none():
    assert parse_candidate("") is None
    assert parse_candidate("   ") is None


def test_classify_reply_yes_variants():
    assert classify_reply("Yes!") == "yes"
    assert classify_reply("yeah") == "yes"
    assert classify_reply("Yep.") == "yes"


def test_classify_reply_no_variants():
    assert classify_reply("nope") == "no"
    assert classify_reply("No") == "no"


def test_classify_reply_description_is_none():
    assert classify_reply("No, its like a little computer") is None


def test_classify_reply_retyped_word_is_none():
    assert classify_reply("arduino") is None


def test_classify_reply_empty_is_none():
    assert classify_reply("") is None
    assert classify_reply(None) is None
