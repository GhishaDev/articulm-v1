"""Sentence-level splitting, deduplication and leakage prevention.

docs/03 requires a 90/5/5 sentence-level split after deduplication with no
near-duplicate leakage; docs/09 rejects any run where train/val leakage exists.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from random import Random

import pytest
import yaml

from articulm.config import ConfigError, SplitConfig, load_data_config, to_plain_dict
from articulm.data import split as split_module
from articulm.data.schema import load_samples, parse_sample
from articulm.data.split import (
    SPLIT_NAMES,
    SplitError,
    analyse_duplicates,
    assign_groups,
    bottom_k_sketch,
    exact_signatures,
    jaccard,
    phoneme_sequence,
    shingle_hashes,
    split_samples,
    text_signature,
    verify_no_leakage,
    write_split,
)


def _token(phoneme: str, tone: int = 1, viseme: int = 3) -> dict:
    return {
        "phoneme": phoneme,
        "language": "zh",
        "surface_tone": tone,
        "stress": 0,
        "syllable_role": "nucleus",
        "articulatory": {"type": "vowel", "height": "high", "backness": "front"},
        "boundary": {"word_start": True, "word_end": True, "boundary_type": "none"},
        "labels": {
            "viseme_id": viseme,
            "strength": 60.0,
            "viseme_source": "unit_test",
            "strength_source": "pseudo_strength_v1",
        },
    }


def _make_samples(data_config, specs: list[tuple[str, list[str]]]):
    """Build samples from (sample_id, [phonemes]) pairs."""
    return [
        parse_sample(
            {
                "schema_version": "articulm_v1_sample_v1",
                "sample_id": sample_id,
                "text": "".join(phonemes),
                "tokens": [_token(p) for p in phonemes],
            },
            data_config,
        )
        for sample_id, phonemes in specs
    ]


@pytest.fixture
def dev_config(tmp_path, data_config):
    """A data config writing splits under tmp_path."""
    raw = to_plain_dict(data_config)
    raw["train_path"] = str(tmp_path / "train.jsonl")
    raw["validation_path"] = str(tmp_path / "validation.jsonl")
    raw["test_path"] = str(tmp_path / "test.jsonl")
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"data": raw}), encoding="utf-8")
    return load_data_config(path)


@pytest.fixture
def many_samples(data_config):
    """200 genuinely unrelated sentences.

    Each is drawn from its own seeded PRNG. Cyclic shifts of one pattern would
    *not* work here: they share most of their n-grams and are correctly treated
    as near-duplicates.
    """
    alphabet = "aieoudmnstxbpfglkzhcr"
    specs = []
    for index in range(200):
        rng = Random(10_000 + index)
        length = rng.randint(18, 40)
        phonemes = [rng.choice(alphabet) for _ in range(length)]
        specs.append((f"s{index:04d}", phonemes))
    return _make_samples(data_config, specs)


# -- signatures and shingles ----------------------------------------------


def test_phoneme_sequence_and_text_signature(all_samples):
    sample = all_samples[0]
    assert phoneme_sequence(sample) == " ".join(t.phoneme for t in sample.tokens)
    assert text_signature(sample)


def test_normalized_text_is_preferred_identity(fixture_paths, data_config):
    """The corpus with a recorded normalisation must key on the normalised form."""
    samples = load_samples(fixture_paths["zh"], data_config)
    with_normalisation = next(s for s in samples if s.normalized_text)
    assert text_signature(with_normalisation) == " ".join(
        with_normalisation.normalized_text.split()
    )


def test_exact_signatures_cover_text_and_phonemes(all_samples):
    signatures = exact_signatures(all_samples[0])
    assert any(s.startswith("phonemes:") for s in signatures)
    assert any(s.startswith("text:") for s in signatures)


def test_shingles_of_short_sequence_fall_back_to_whole(data_config):
    short, = _make_samples(data_config, [("x", ["a", "i"])])
    assert len(shingle_hashes(short, size=4)) == 1


def test_shingle_count_matches_ngram_count(data_config):
    sample, = _make_samples(data_config, [("x", list("abcdefgh"))])
    # 8 phonemes, 4-grams -> 5 distinct windows
    assert len(shingle_hashes(sample, size=4)) == 5


def _silence_token(phoneme: str) -> dict:
    token = _token(phoneme)
    token["articulatory"] = {"type": "silence"}
    token["syllable_role"] = "silence"
    token["labels"] = {
        "viseme_id": 16,
        "strength": 0.0,
        "viseme_source": "unit_test",
        "strength_source": "pseudo_strength_v1",
    }
    return token


def test_shingles_exclude_silence_tokens(data_config):
    """Sentence-final silence tokens must not create shingle collisions.

    ``∅``/``-`` are identical in every sentence; hashing them makes the final
    4-gram a near-universal collision and defeats near-duplicate recall.
    """
    base = parse_sample(
        {"sample_id": "base", "text": "ai", "tokens": [_token("a"), _token("i")]},
        data_config,
    )
    with_silence = parse_sample(
        {
            "sample_id": "s",
            "text": "ai",
            "tokens": [_token("a"), _token("i"), _silence_token("-"), _silence_token("∅")],
        },
        data_config,
    )
    assert shingle_hashes(with_silence, size=4) == shingle_hashes(base, size=4)


def test_bottom_k_sketch_is_the_k_smallest():
    assert bottom_k_sketch([9, 1, 5, 3, 7], 3) == (1, 3, 5)
    assert bottom_k_sketch([2, 1], 8) == (1, 2)


def test_jaccard_bounds():
    assert jaccard(frozenset({1, 2}), frozenset({1, 2})) == 1.0
    assert jaccard(frozenset({1}), frozenset({2})) == 0.0
    assert jaccard(frozenset({1, 2}), frozenset({2, 3})) == pytest.approx(1 / 3)
    assert jaccard(frozenset(), frozenset()) == 1.0


# -- duplicate analysis ---------------------------------------------------


def test_distinct_sentences_form_singleton_groups(data_config, many_samples):
    analysis = analyse_duplicates(many_samples, data_config)
    assert analysis.num_sentences == len(many_samples)
    assert analysis.num_redundant_sentences == 0
    assert all(len(group) == 1 for group in analysis.groups)


def test_exact_duplicates_are_grouped(data_config):
    samples = _make_samples(
        data_config,
        [("a", list("aieou")), ("b", list("aieou")), ("c", list("uoeia"))],
    )
    analysis = analyse_duplicates(samples, data_config)
    groups = {tuple(group) for group in analysis.groups}
    assert (0, 1) in groups
    assert (2,) in groups
    assert analysis.num_exact_duplicate_pairs >= 1


def _long_varied_sequence(length: int, offset: int = 0) -> list[str]:
    """A non-repeating phoneme sequence, so 4-grams are mostly distinct.

    A repeating pattern like ``aieou`` x3 collapses to very few distinct
    shingles, which makes Jaccard behave nothing like it does on real text.
    """
    alphabet = "aieoudmnstx"
    return [
        alphabet[(index * 7 + index // len(alphabet) + offset) % len(alphabet)]
        for index in range(length)
    ]


def test_near_duplicates_are_grouped(data_config):
    base = _long_varied_sequence(90)
    near = base.copy()
    near[45] = "z"  # a single edit inside a long sentence
    samples = _make_samples(
        data_config,
        [("base", base), ("near", near), ("other", _long_varied_sequence(90, offset=3))],
    )
    analysis = analyse_duplicates(samples, data_config)
    group_of = {index: gi for gi, group in enumerate(analysis.groups) for index in group}
    assert group_of[0] == group_of[1], "near duplicate was not grouped"
    assert group_of[2] != group_of[0]
    assert analysis.num_near_duplicate_pairs >= 1


def test_near_duplicate_recall_depends_on_shingle_overlap(data_config):
    """Characterisation: one edit in a *short* sentence stays under threshold.

    A 4-gram edit changes up to 4 shingles, so on short sentences the Jaccard
    drop is large. Lowering the threshold recovers those pairs. Documented
    rather than hidden, because it bounds what leakage control can catch.
    """
    base = _long_varied_sequence(12)
    near = base.copy()
    near[-1] = "z"
    samples = _make_samples(data_config, [("base", base), ("near", near)])

    strict = dataclasses.replace(
        data_config,
        split=dataclasses.replace(
            data_config.split, near_duplicate_jaccard_threshold=0.95
        ),
    )
    assert analyse_duplicates(samples, strict).num_near_duplicate_pairs == 0

    lenient = dataclasses.replace(
        data_config,
        split=dataclasses.replace(
            data_config.split, near_duplicate_jaccard_threshold=0.5
        ),
    )
    assert analyse_duplicates(samples, lenient).num_near_duplicate_pairs >= 1


def test_default_threshold_favours_recall(data_config):
    """The default must be below 0.9: missing a leak costs more than
    over-grouping (see SplitConfig)."""
    assert data_config.split.near_duplicate_jaccard_threshold <= 0.85


def test_near_duplicate_detection_can_be_disabled(data_config):
    base = _long_varied_sequence(90)
    near = base.copy()
    near[45] = "z"
    samples = _make_samples(data_config, [("base", base), ("near", near)])
    disabled = dataclasses.replace(
        data_config,
        split=dataclasses.replace(data_config.split, prevent_near_duplicate_leakage=False),
    )
    analysis = analyse_duplicates(samples, disabled)
    assert analysis.num_near_duplicate_pairs == 0
    assert analysis.num_candidate_pairs_compared == 0
    assert all(len(group) == 1 for group in analysis.groups)


def test_oversized_buckets_are_skipped_and_reported(data_config):
    """Skipping must be visible, not silent."""
    samples = _make_samples(
        data_config, [(f"s{i}", list("aieouaieou")) for i in range(20)]
    )
    capped = dataclasses.replace(
        data_config,
        split=dataclasses.replace(data_config.split, near_duplicate_max_bucket_size=2),
    )
    analysis = analyse_duplicates(samples, capped)
    assert analysis.num_skipped_buckets > 0
    assert analysis.largest_skipped_bucket >= 3
    assert analysis.num_sentences_in_skipped_buckets > 0


def test_empty_corpus_is_rejected(data_config):
    with pytest.raises(SplitError, match="empty corpus"):
        analyse_duplicates([], data_config)


# -- assignment -----------------------------------------------------------


def test_split_hits_target_ratios(data_config, many_samples):
    result = split_samples(many_samples, data_config)
    actual = result.actual_ratios()
    assert actual["train"] == pytest.approx(0.90, abs=0.02)
    assert actual["validation"] == pytest.approx(0.05, abs=0.02)
    assert actual["test"] == pytest.approx(0.05, abs=0.02)


def test_every_sentence_is_assigned_exactly_once(data_config, many_samples):
    result = split_samples(many_samples, data_config)
    everything = [i for name in SPLIT_NAMES for i in result.indices[name]]
    assert sorted(everything) == list(range(len(many_samples)))


def test_split_is_deterministic(data_config, many_samples):
    first = split_samples(many_samples, data_config)
    second = split_samples(many_samples, data_config)
    assert first.indices == second.indices


def test_different_seed_changes_the_assignment(data_config, many_samples):
    other = dataclasses.replace(
        data_config, split=dataclasses.replace(data_config.split, seed=999)
    )
    first = split_samples(many_samples, data_config)
    second = split_samples(many_samples, other)
    assert first.indices != second.indices
    # ...but both stay valid partitions.
    for result in (first, second):
        everything = [i for name in SPLIT_NAMES for i in result.indices[name]]
        assert sorted(everything) == list(range(len(many_samples)))


def test_duplicate_groups_never_span_splits(data_config, many_samples):
    """The core leakage guarantee, with heavy injected duplication."""
    duplicated = list(many_samples)
    for index in range(0, 100, 3):
        clone = copy.deepcopy(many_samples[index])
        duplicated.append(dataclasses.replace(clone, sample_id=f"dup_{index}"))

    result = split_samples(duplicated, data_config)
    assert result.analysis.num_redundant_sentences > 0
    assert verify_no_leakage(duplicated, result, data_config) == []


def test_verify_detects_a_corrupted_split(data_config):
    samples = _make_samples(
        data_config, [("a", list("aieou")), ("b", list("aieou"))]
    )
    result = split_samples(samples, data_config)
    # Force the exact-duplicate pair apart.
    broken = dataclasses.replace(
        result, indices={"train": [0], "validation": [1], "test": []}
    )
    problems = verify_no_leakage(samples, broken, data_config)
    assert problems
    assert any("spans splits" in p or "appears in splits" in p for p in problems)


def test_verify_detects_a_sentence_in_two_splits(data_config, many_samples):
    result = split_samples(many_samples, data_config)
    broken = dataclasses.replace(
        result,
        indices={
            "train": result.indices["train"],
            "validation": [result.indices["train"][0]],
            "test": [],
        },
    )
    problems = verify_no_leakage(many_samples, broken, data_config)
    assert any("appears in both" in p for p in problems)


def test_zero_ratio_split_receives_nothing(data_config, many_samples):
    no_test = dataclasses.replace(
        data_config,
        split=SplitConfig(train_ratio=0.9, validation_ratio=0.1, test_ratio=0.0),
    )
    result = split_samples(many_samples, no_test)
    assert result.indices["test"] == []
    assert len(result.indices["train"]) + len(result.indices["validation"]) == len(
        many_samples
    )


def test_all_zero_ratios_are_rejected():
    with pytest.raises(SplitError, match="at least one split ratio"):
        assign_groups([[0], [1]], {"train": 0.0, "validation": 0.0, "test": 0.0}, seed=1)


def test_large_groups_are_placed_first(data_config):
    """A cluster larger than the validation quota must not overshoot it."""
    groups = [[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]]
    assigned = assign_groups(
        groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, seed=0
    )
    assert len(assigned["train"]) == 8
    assert sorted(assigned["validation"] + assigned["test"]) == [8, 9]


def test_drop_duplicates_keeps_one_per_group(data_config):
    samples = _make_samples(
        data_config,
        [("a", list("aieou")), ("b", list("aieou")), ("c", list("aieou")), ("d", list("ddddd"))],
    )
    result = split_samples(samples, data_config, drop_duplicates=True)
    kept = [i for name in SPLIT_NAMES for i in result.indices[name]]
    assert len(kept) == 2
    assert len(result.dropped_duplicate_indices) == 2


def test_split_result_serialises(data_config, many_samples):
    payload = split_samples(many_samples, data_config).as_dict()
    json.dumps(payload)
    assert payload["counts"]["train"] > 0
    assert "num_near_duplicate_pairs" in payload
    assert "num_skipped_buckets" in payload


# -- writing --------------------------------------------------------------


def test_write_split_copies_lines_verbatim(tmp_path, fixture_dir):
    source = fixture_dir / "sample_zh.jsonl"
    original = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    destination = tmp_path / "subset.jsonl"
    written = write_split(source, [0, 2], destination)
    assert written == 2
    result = [line for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Byte-identical: splitting must never re-serialise and alter semantics.
    assert result == [original[0], original[2]]


def test_write_split_rejects_out_of_range_selection(tmp_path, fixture_dir):
    with pytest.raises(SplitError, match="wrote 0 records"):
        write_split(fixture_dir / "sample_zh.jsonl", [99], tmp_path / "x.jsonl")


# -- config validation ----------------------------------------------------


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("near_duplicate_jaccard_threshold", 0.0, "jaccard_threshold"),
        ("near_duplicate_jaccard_threshold", 1.5, "jaccard_threshold"),
        ("near_duplicate_shingle_size", 0, "shingle_size"),
        ("near_duplicate_sketch_size", 0, "sketch_size"),
        ("near_duplicate_max_bucket_size", 1, "max_bucket_size"),
    ],
)
def test_invalid_split_settings_fail_fast(tmp_path, data_config, field, value, message):
    raw = to_plain_dict(data_config)
    raw["split"][field] = value
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"data": raw}), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_data_config(path)


# -- CLI ------------------------------------------------------------------


def _write_diverse_corpus(path, count: int) -> None:
    """A corpus of genuinely distinct sentences, one JSONL record each."""
    with path.open("w", encoding="utf-8") as fh:
        for index in range(count):
            phonemes = _long_varied_sequence(8 + index % 9, offset=index)
            record = {
                "schema_version": "articulm_v1_sample_v1",
                "sample_id": f"c{index:04d}",
                "text": f"sentence-{index}-" + "".join(phonemes),
                "tokens": [_token(p) for p in phonemes],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_split_cli_writes_all_three_splits(tmp_path, dev_config, capsys):
    corpus = tmp_path / "corpus.jsonl"
    _write_diverse_corpus(corpus, 60)
    config_path = tmp_path / "data.yaml"
    exit_code = split_module.main(
        [
            "--config",
            str(config_path),
            "--input",
            str(corpus),
            "--report-out",
            str(tmp_path / "report.json"),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "leakage check passed" in output

    for name in SPLIT_NAMES:
        path = tmp_path / f"{name}.jsonl"
        assert path.is_file(), name
    total = sum(
        len([1 for line in (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])
        for name in SPLIT_NAMES
    )
    assert total == 60

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["leakage_check"] == "passed"
    assert report["counts"]["train"] > report["counts"]["validation"]


def test_split_cli_dry_run_writes_nothing(tmp_path, dev_config, capsys):
    corpus = tmp_path / "corpus.jsonl"
    _write_diverse_corpus(corpus, 60)
    exit_code = split_module.main(
        ["--config", str(tmp_path / "data.yaml"), "--input", str(corpus), "--dry-run"]
    )
    capsys.readouterr()
    assert exit_code == 0
    assert not (tmp_path / "train.jsonl").exists()


def test_split_cli_fails_when_a_targeted_split_is_empty(tmp_path, dev_config, capsys):
    """A degenerate corpus must not silently produce a 100/0/0 split."""
    corpus = tmp_path / "corpus.jsonl"
    phonemes = _long_varied_sequence(20)
    with corpus.open("w", encoding="utf-8") as fh:
        for index in range(30):
            record = {
                "schema_version": "articulm_v1_sample_v1",
                "sample_id": f"same{index:03d}",
                "text": "identical",
                "tokens": [_token(p) for p in phonemes],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    exit_code = split_module.main(
        ["--config", str(tmp_path / "data.yaml"), "--input", str(corpus), "--dry-run"]
    )
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "have a positive target ratio but received no sentences" in output
    assert "RATIO WARNINGS" in output
    assert "is EMPTY" in output


def test_allow_empty_splits_overrides_the_failure(tmp_path, dev_config, capsys):
    corpus = tmp_path / "corpus.jsonl"
    phonemes = _long_varied_sequence(20)
    with corpus.open("w", encoding="utf-8") as fh:
        for index in range(30):
            record = {
                "schema_version": "articulm_v1_sample_v1",
                "sample_id": f"same{index:03d}",
                "text": "identical",
                "tokens": [_token(p) for p in phonemes],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    exit_code = split_module.main(
        [
            "--config",
            str(tmp_path / "data.yaml"),
            "--input",
            str(corpus),
            "--dry-run",
            "--allow-empty-splits",
        ]
    )
    capsys.readouterr()
    assert exit_code == 0


def test_degenerate_corpus_reports_warnings_in_the_result(data_config):
    samples = _make_samples(
        data_config, [(f"same{i}", _long_varied_sequence(20)) for i in range(30)]
    )
    result = split_samples(samples, data_config)
    assert result.empty_targeted_splits
    assert result.warnings
    assert any("is EMPTY" in warning for warning in result.warnings)
    # Leakage control still holds: the group was not broken up.
    assert verify_no_leakage(samples, result, data_config) == []


def test_healthy_corpus_has_no_warnings(data_config, many_samples):
    result = split_samples(many_samples, data_config)
    assert result.warnings == []
    assert result.empty_targeted_splits == []


def test_split_cli_rejects_a_bad_corpus(tmp_path, dev_config, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"sample_id": "x", "tokens": [{"phoneme": "a", "language": "zh"}]}) + "\n",
        encoding="utf-8",
    )
    exit_code = split_module.main(
        ["--config", str(tmp_path / "data.yaml"), "--input", str(bad)]
    )
    assert exit_code == 2
    assert "FAILED" in capsys.readouterr().out
