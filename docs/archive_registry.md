# Real archive validation (config/archives.dev.toml)

Produced by running `python -m pytest -m integration tests/test_registry_real_archives.py`
against the archives actually present on this machine on 2026-09-19. `simplewiki` and
`wikibooks` point at `C:\kiwix\...` paths that do not exist yet on this machine (aspirational
future downloads) and were skipped, as `Registry.validate_all` never raises for missing files.
No `Archive.check()` integrity scan was run (D:\Kiwix is a slow HDD); this is signature +
open + metadata only.

| id | state | has_fulltext_index | article_count | uuid |
|---|---|---|---|---|
| simplewiki | MISSING (skipped) | | | |
| enwiki | VALID | True | 6827783 | 73e044f1-2361-7d9d-3a1a-6be0b6eab420 |
| lumen | VALID | True | 3931 | 16c4d5a7-60c9-abbb-f06d-80933958f962 |
| math_se | VALID | True | 1947704 | 8d98fc2e-469d-a1e4-725b-97a2883eef46 |
| wikibooks | MISSING (skipped) | | | |
| wikiversity | VALID | True | 42755 | 0474d4a1-c018-5d14-bb92-5fa05f1c89ca |
| matheducators_se | VALID | True | 8412 | d994c979-5dd8-5cf5-334f-e0837d621e81 |
| physics_se | VALID | True | 321582 | 35e15094-a413-52a3-5e42-103597c867cb |
| chemistry_se | VALID | True | 69812 | 49504fe2-e9c1-0202-28d6-2b50b59d4916 |
| biology_se | VALID | True | 46354 | 10947d5c-cae1-c43b-4fc2-3060769f11d9 |
| history_se | VALID | True | 27264 | 4aa7bbaf-8546-374b-be07-885a721bc3aa |
| english_se | VALID | True | 234818 | 8ce6003e-7a68-a297-0e43-e4c9406bd305 |
| linguistics_se | VALID | True | 19086 | 348e4ac5-4654-1da1-64c3-b814bfc3820e |
| literature_se | VALID | True | 12792 | 16d6fe80-653a-5386-3687-e7f391117abb |
| puzzling_se | VALID | True | 43525 | 6b94602d-f4dd-6ea7-20f5-7158b0ae6392 |
| wikihow | VALID | True | 122463 | 4f1c17e6-cd27-3060-ac5b-31448348c6dc |
| gutenberg | VALID | True | 132769 | dde3b47e-903d-b824-28db-699d9d523790 |

Every archive present on D:\Kiwix validated as VALID with a full-text index; none were
NO_FULLTEXT_INDEX or corrupt.

## simplewiki tier-1 archive (M2, 2026-09-20)

`C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim` (3.4 GB, SSD) now exists and was
validated with `tutor.retrieval.zim.archive.validate_archive` + `fingerprint`:

| id | state | has_fulltext_index | article_count | uuid | digest (sha256, truncated) |
|---|---|---|---|---|---|
| simplewiki | VALID | True | 394566 | 7623b2f2-ca9e-08b6-c254-d17e4b9131e7 | 63e73b397230ad9fc5a22040173fd1375a0bba3... |

Metadata: language `eng`, name `wikipedia_en_simple_all`, title "Wikipedia in simple
English", date `2026-05-10`.
