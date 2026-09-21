"""Tests for tutor.retrieval.assessment.assess_evidence (pure, no I/O)."""

from __future__ import annotations

from types import SimpleNamespace

from tutor.retrieval.assessment import assess_evidence


def _passage(title: str, text: str):
    return SimpleNamespace(title=title, text=text)


class _Result:
    def __init__(self, passages):
        self.passages = passages


def test_empty_when_no_passages():
    result = _Result([])
    assessment = assess_evidence("What is the capital of France?", result)
    assert assessment.level == "empty"


def test_strong_when_top_passage_covers_key_terms():
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening is a way to plan a small vegetable garden.",
            )
        ]
    )
    assessment = assess_evidence("how to squarefoot garden the right way?", result)
    # "squarefoot" itself won't match (it's fused), but "garden"/"gardening"
    # should stem-match via singularize, giving decent coverage once the
    # fused-word variant search (research.py) has actually found the
    # right article's title/text.
    assert assessment.level in ("strong", "weak")
    assert "garden" in {t for t in assessment.key_terms}


def test_false_strong_fused_term_split_across_generic_titled_passages():
    """Real bottleneck case: 'squarefoot garden' splits into the generic
    words 'square'/'foot'/'garden', each trivially covered by an unrelated,
    generically-titled passage -- coverage alone crosses 0.5 but none of
    these passages is actually about square-foot gardening."""
    result = _Result(
        [
            _passage("Garden", "A garden is a planned space set aside for plants."),
            _passage("Foot", "The foot is an anatomical structure of vertebrates."),
            _passage("Chromatica", "Chromatica is an album; one song mentions a foot."),
        ]
    )
    assessment = assess_evidence(
        "how to squarefoot garden the right way?",
        result,
        corrected_terms={"squarefoot": "square foot"},
    )
    assert assessment.level == "weak"
    assert any("topic phrase" in r for r in assessment.reasons)


def test_strong_when_corrected_topic_phrase_actually_present():
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening is a way to plan a small vegetable garden.",
            )
        ]
    )
    assessment = assess_evidence(
        "how to squarefoot garden the right way?",
        result,
        corrected_terms={"squarefoot": "square foot"},
    )
    assert assessment.level == "strong"


def test_weak_when_passages_are_off_topic():
    result = _Result(
        [_passage("Unrelated topic", "This article is about something else entirely.")]
    )
    assessment = assess_evidence("how does photosynthesis work in plants?", result)
    assert assessment.level == "weak"
    assert assessment.coverage < 0.5


def test_stem_match_garden_gardening():
    result = _Result([_passage("Gardening", "Gardening is the practice of growing plants.")])
    assessment = assess_evidence("what is square foot gardening?", result)
    assert "garden" in assessment.covered_terms or "gardening" in assessment.covered_terms
    assert assessment.coverage > 0.0


def test_question_fillers_excluded_from_key_terms():
    assessment = assess_evidence("how to squarefoot garden the right way?", _Result([]))
    assert "right" not in assessment.key_terms
    assert "way" not in assessment.key_terms


def test_no_key_terms_treated_as_strong_when_passages_present():
    result = _Result([_passage("Something", "Some text.")])
    assessment = assess_evidence("what about it?", result)
    assert assessment.level == "strong"


def test_only_checks_top_passages():
    # 4th passage covers the term, but only top 3 are checked.
    passages = [
        _passage("Irrelevant one", "nothing"),
        _passage("Irrelevant two", "nothing"),
        _passage("Irrelevant three", "nothing"),
        _passage("Volcano", "A volcano is an opening in the Earth's crust."),
    ]
    result = _Result(passages)
    assessment = assess_evidence("how do volcano eruptions happen?", result)
    assert assessment.level == "weak"


def test_to_dict_roundtrip_keys():
    result = _Result([_passage("Volcano", "A volcano erupts lava.")])
    assessment = assess_evidence("how do volcanoes erupt?", result)
    d = assessment.to_dict()
    assert set(d) == {"level", "reasons", "key_terms", "covered_terms", "coverage"}


def test_rewritten_queries_terms_replace_fused_original_terms():
    """The reassessment call after a forced rewrite must judge coverage
    against the REWRITTEN queries' own terms, not the original fused/
    misspelt "squarefoot" -- see docs/rewrite_on_weak_evidence.md."""
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening is a method for planning small vegetable "
                "gardens using a grid.",
            )
        ]
    )
    rewritten_queries = [
        "how to square foot garden the right way",
        "square foot gardening basics",
        "square foot gardening guide step by step",
    ]
    rewritten = assess_evidence(
        "how to squarefoot garden the right way?",
        result,
        rewritten_queries=rewritten_queries,
    )
    # The fused original term pins coverage low or ambiguous; the rewritten
    # queries' own (correctly spelt) terms should give a clearly strong
    # verdict on the same evidence.
    assert rewritten.level == "strong"
    assert "squarefoot" not in rewritten.key_terms


def test_rewritten_queries_with_healthy_original_terms_kept():
    """A correctly-spelt original-question term with a healthy match is
    kept alongside the rewritten queries' terms, not dropped."""
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening basics for a raised bed vegetable garden.",
            )
        ]
    )
    assessment = assess_evidence(
        "square foot garden basics",
        result,
        rewritten_queries=["square foot gardening basics"],
        healthy_terms={"garden": True, "basics": True},
    )
    assert assessment.level == "strong"


# --- Assessor v3 (docs/rewrite_probe_measure.md "Assessor v3") --------
# Real cached cases from data/assessor_labelled_cache.json (84-question
# labelled set), texts inlined -- see that doc's measure for the full
# before/after confusion tables. Signals under test: (a) multi-word topic
# integrity, (b) constraint (number/year) terms, (c) stop-name filler
# words reusing QUESTION_SHAPE_FILLERS. Signal (d) (same-passage
# co-occurrence) was tried and REJECTED: it fixed 2 out-of-library cases
# but cost ~14 new false-weak verdicts on the tuning/probe should-strong
# rows, far over budget -- see that doc section for the numbers.


def test_sw21_world_war_two_not_world_war_one():
    """sw21: 'World War I' partially covers "world war two" (2 of 3
    words) but carries the WRONG number -- must not certify as strong
    just because "world"/"war" trivially match."""
    result = _Result(
        [
            _passage(
                "World War I",
                "World War II began in 1939, World War I was called the Great War, "
                "or the World War. 135 countries took part in World War I, and "
                "almost 10 million people died while fighting.",
            ),
            _passage(
                "World War I",
                "how many people it killed and how much damage it caused. "
                'They hoped it would be "the war to end all wars". Instead it '
                "led to another, larger, world war 21 years later.",
            ),
            _passage("List of field guns", "France | World War I\n75 | Canon de 75 mle GP1"),
        ]
    )
    assessment = assess_evidence("What caused World War Two?", result)
    assert assessment.level == "weak"


def test_sw22_french_revolution_residual_false_strong():
    """sw22: 'Revolution' (generic article) happens to spell out "the
    French Revolution (1789)" verbatim in its text while discussing
    revolutions generally -- this genuinely satisfies the exact-phrase-
    in-text branch of the multi-word check, so this remains a KNOWN,
    documented residual false-strong (see docs/rewrite_probe_measure.md
    "Assessor v3"), not a guard for a fix. Recorded here so a future
    change that accidentally makes this MORE wrong (e.g. matching on a
    totally unrelated passage) is still caught."""
    result = _Result(
        [
            _passage(
                "Revolution",
                "The Soviet Union was started by the Russian Revolution, which "
                "killed millions, and the country later fell apart in another "
                "revolution without much fighting. However, in the French "
                "Revolution (1789), there was much bloodshed, which included "
                "the Reign of Terror.",
            ),
        ]
    )
    assessment = assess_evidence("Why did the French Revolution happen?", result)
    assert assessment.level == "strong"  # documented residual, not desired


def test_sw65_solar_system_residual_false_strong():
    """sw65: similarly a residual -- 'Planet' mentions "the Solar System"
    while discussing planets generally, and "planets" alone (a curated
    generic word) is still trusted as a last-resort single-word fallback
    once the multi-word phrase fails outright. Documented, not asserted
    as fixed."""
    result = _Result(
        [
            _passage(
                "Planet",
                "Jupiter is the biggest planet in the Solar System, while the "
                "smallest planet in the Solar System is Mercury.",
            ),
            _passage(
                "Planet",
                "There are eight planets in the Solar System. Pluto used to be "
                "called a planet, but in August 2006 the IAU decided it was a "
                "dwarf planet instead.",
            ),
            _passage("List of planets", "This is a list of two types of planets."),
        ]
    )
    assessment = assess_evidence("List the planets in the Solar System.", result)
    assert assessment.level == "strong"  # documented residual, not desired


def test_kid04_rain_forest_not_generic_rain_and_forest_separately():
    """kid04: "Rain" and "Forest" are separate generic single-word
    articles -- neither is "rain forest" the named habitat, and the
    two-word compound never appears as one phrase either. Signal (a)
    fixes this one."""
    result = _Result(
        [
            _passage(
                "Rain",
                "Rain falling. Rainy day. Rain is a kind of precipitation. "
                "Precipitation is any kind of water that falls from clouds.",
            ),
            _passage(
                "Forest",
                "A forest is a piece of land with many trees. Forests are an "
                "ecosystem which includes many plants and animals. Many "
                "animals live in forests and need them to survive.",
            ),
            _passage("Rain", "Rain is part of the water cycle."),
        ]
    )
    assessment = assess_evidence("what animals live in the rain forest?", result)
    assert assessment.level == "weak"


def test_ool09_guy_and_name_are_stop_words_not_a_topic():
    """ool09: "Guy Fieri" trivially matches the filler word "guy" (now in
    QUESTION_SHAPE_FILLERS) and "Invention" covers "invented" alone --
    neither passage is actually about who invented homework."""
    result = _Result(
        [
            _passage("Guy Fieri", "Guy Ramsay Fieri is an American restaurateur."),
            _passage(
                "Invention",
                "An invention is a new thing that someone has made. The computer "
                'was an invention when it was first made. We say when it was "invented".',
            ),
            _passage("Guy Fieri", "Guy Fieri is a true Bow-Tie guy."),
        ]
    )
    assessment = assess_evidence("whats the name of the guy who invented homework", result)
    assert assessment.level == "weak"
    assert "guy" not in assessment.key_terms
    assert "name" not in assessment.key_terms


def test_ool12_blue_whale_not_generic_whale_article():
    """ool12: a general "Whale"/"Whaling" article happens to mention "the
    blue whale" in passing, and coverage is otherwise met by unrelated
    generic terms -- neither passage says anything about hairs or a
    flipper. Signal (a), with "blue"/"whale" curated as generic, fixes
    this one."""
    result = _Result(
        [
            _passage(
                "Whale",
                "Some countries, such as Iceland and Japan, do not have these "
                "laws. In other countries, only Eskimos and some American "
                "Indians may legally kill whales such as the blue whale and "
                "beluga whale.",
            ),
            _passage(
                "Whale",
                "Whales are a group of cetacean mammals that live in the ocean. "
                "Like other mammals, they breathe oxygen from the air, have a "
                "small amount of hair, and are warm blooded.",
            ),
            _passage(
                "Whaling",
                "Several kinds of whales became almost extinct. Whales are "
                "protected by conservation laws that stop people from killing "
                "too many of them.",
            ),
        ]
    )
    assessment = assess_evidence(
        "How many hairs does a blue whale have on its left flipper?", result
    )
    assert assessment.level == "weak"


def test_rose_bowl_out_of_library_empty_stays_weak():
    """The 1994 Rose Bowl case: the archive has no passages at all for
    this out-of-library question -- already correctly "empty" before
    Assessor v3, kept here as a guard against any future signal
    accidentally requiring passages to call something weak."""
    result = _Result([])
    assessment = assess_evidence("What was the attendance at the 1994 Rose Bowl?", result)
    assert assessment.level == "empty"


# --- Guards: real should-be-strong tuning cases a naive version of the
# multi-word/constraint rules would have broken. -----------------------


def test_guard_water_cycle_compound_phrase_matches_title():
    """sw04: "water cycle" IS a genuine adjacent multi-word compound and
    the title matches it outright -- must stay strong."""
    result = _Result(
        [
            _passage(
                "Water cycle",
                "The water cycle, or hydrological cycle, is the cycle that "
                "water goes through on Earth.",
            ),
            _passage(
                "Water cycle",
                "This is the process that water starts and ends in the water "
                "cycle. The cycle starts when water underground evaporates.",
            ),
            _passage("Thermodynamic cycle", "A thermodynamic cycle is a series of processes."),
        ]
    )
    assessment = assess_evidence("What is the water cycle?", result)
    assert assessment.level == "strong"


def test_guard_volcano_singular_title_for_singular_question():
    """sw10: a plain single-word gold title ("Volcano") with no
    multi-word run in the question at all -- unaffected by signal (a)."""
    result = _Result(
        [
            _passage(
                "Volcano",
                "A volcano is a mountain that has lava coming out from a magma "
                "chamber under the ground.",
            ),
            _passage("Volcano", "An extinct volcano has not erupted in the past 10,000 years."),
            _passage("Volcano (1997 movie)", "Volcano on IMDb."),
        ]
    )
    assessment = assess_evidence("What is a volcano?", result)
    assert assessment.level == "strong"


def test_guard_helium_boiling_point_infobox_colon_adjacency():
    """sw61/sw62: "boiling point" is a genuine adjacent bigram and the
    infobox text spells it "Boiling point: 4.222 K" (colon, no space) --
    the word-boundary regex, not a padded-space substring check, must
    still find it. Also a single generic-looking gold title (Helium)."""
    result = _Result(
        [
            _passage(
                "Helium",
                "Melting point: 0.95 K (-272.20 C) (at 2.5 MPa)\n"
                "Boiling point: 4.222 K (-268.928 C, -452.070 F)",
            ),
            _passage("Helium", "Triple point: 2.177 K, 5.043 kPa"),
            _passage(
                "Helium",
                "It has the lowest boiling point of all the elements. It is "
                "the second most common element in the universe.",
            ),
        ]
    )
    assessment = assess_evidence(
        "Output the boiling point of helium in celsius and farenheit.", result
    )
    assert assessment.level == "strong"


def test_guard_nitrogen_boiling_point_repeated_title():
    """sw63: same shape as the Helium guard, with every top passage
    sharing the same (single-word, chemistry-generic) title."""
    result = _Result(
        [
            _passage(
                "Nitrogen",
                "Melting point: 63.15 K (-210.00 C)\nBoiling point: 77.355 K "
                "(-195.795 C, -320.431 F)",
            ),
            _passage("Nitrogen", "Triple point: 63.151 K, 12.52 kPa"),
            _passage(
                "Nitrogen",
                "Nitrogen is a colorless odorless gas at normal temperature.",
            ),
        ]
    )
    assessment = assess_evidence("Tell me the boiling point of nitrogen.", result)
    assert assessment.level == "strong"


def test_guard_noun_and_verb_split_across_two_generic_titles():
    """sw34: "a noun AND a verb" -- the raw-adjacency requirement means
    "noun"/"verb" never form a multi-word run (they're separated by
    "and" in the raw text), so this is untouched by signal (a) and stays
    strong via the two single-word titles, exactly as before."""
    result = _Result(
        [
            _passage(
                "Noun",
                "Nouns: truth, danger, happiness. Abstract Nouns: things that "
                "cannot be interacted with physically.",
            ),
            _passage(
                "Noun",
                "These words usually do not go with other kinds of words like "
                "verbs or adverbs.",
            ),
            _passage("Verb", "walking | She is walking home; past participle: walked."),
        ]
    )
    assessment = assess_evidence("How are a noun and a verb different?", result)
    assert assessment.level == "strong"


def test_v4_biggest_animal_unrelated_titles_not_strong():
    """Turn-1 evidence bug (docs/rewrite_probe_measure.md 'Assessor v4'):
    'biggest' alone (a bare superlative modifier) must never certify a
    passage as covering 'What's the biggest animal?' just because the word
    literally appears in an unrelated title."""
    result = _Result(
        [
            _passage("The Biggest Loser", "A reality show about weight loss and fitness."),
            _passage(
                "World's Biggest Coffee Morning",
                "An annual fundraising event held every year.",
            ),
        ]
    )
    assessment = assess_evidence("What's the biggest animal?", result)
    assert assessment.level != "strong"


def test_v4_fastest_bird_disjoint_passages_not_strong():
    """'fastest' matches a motorsport article and 'bird' matches a state-
    birds list -- disjoint passages, neither actually about the fastest
    bird -- must not be strong even though naive coverage is 1.00."""
    result = _Result(
        [
            _passage("Fastest lap", "The fastest lap record in Formula One racing."),
            _passage("List of U.S. state birds", "Each U.S. state has an official bird."),
        ]
    )
    assessment = assess_evidence("What's the fastest bird?", result)
    assert assessment.level != "strong"


def test_v4_largest_molecule_no_answer_passage_not_strong():
    """No titin/macromolecule passage at all; 'molecule' passages mention
    sugar/bonding, not size, and 'largest' never co-occurs with 'molecule'
    in the same passage."""
    result = _Result(
        [
            _passage("Molecule", "This is a sugar molecule. Carbon atoms are made blue."),
            _passage("Molecule", "Bonding: for a molecule to exist, atoms have to stick."),
            _passage("Water", "Water has a specific heat capacity."),
        ]
    )
    assessment = assess_evidence("What's the largest molecule?", result)
    assert assessment.level != "strong"


def test_v4_smallest_planet_stays_strong():
    """Mercury passage genuinely says 'smallest planet' -- head noun and
    superlative cue co-occur in the same passage -- must stay strong."""
    result = _Result(
        [
            _passage(
                "Mercury (planet)",
                "Mercury is the smallest planet in the Solar System.",
            ),
        ]
    )
    assessment = assess_evidence("What is the smallest planet?", result)
    assert assessment.level == "strong"


def test_v4_tallest_mountain_stays_strong():
    """Everest passage says 'tallest mountain' -- head noun + cue
    co-occur -- must stay strong even though the modifier itself must
    never count toward coverage alone."""
    result = _Result(
        [
            _passage(
                "Mount Everest",
                "Mount Everest is the tallest mountain in the world.",
            ),
        ]
    )
    assessment = assess_evidence("What's the tallest mountain?", result)
    assert assessment.level == "strong"


def test_v4_non_superlative_question_unaffected():
    """A plain (non-superlative) question must behave exactly as before:
    real topic coverage in a real passage is still strong."""
    result = _Result(
        [_passage("Volcano", "A volcano is an opening in the Earth's crust.")]
    )
    assessment = assess_evidence("What is a volcano?", result)
    assert assessment.level == "strong"


def test_v4_superlative_downgrade_has_reason():
    result = _Result(
        [
            _passage("The Biggest Loser", "A reality show about weight loss."),
        ]
    )
    assessment = assess_evidence("What's the biggest animal?", result)
    assert assessment.level != "strong"
    assert any("superlative" in r.lower() for r in assessment.reasons)


def test_no_rewritten_queries_is_backward_compatible():
    """Omitting the new parameters reproduces the exact old signature's
    behaviour."""
    result = _Result([_passage("Volcano", "A volcano is an opening in the crust.")])
    old = assess_evidence("What is a volcano?", result)
    new = assess_evidence("What is a volcano?", result, rewritten_queries=None)
    assert old.level == new.level
    assert old.key_terms == new.key_terms
