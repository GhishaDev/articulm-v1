"""Schema parsing and label-leakage rejection, using the doc fixtures."""

from __future__ import annotations

import copy

import pytest

from articulm.data.schema import (
    NA,
    SchemaError,
    load_samples,
    parse_sample,
)


def _minimal_token() -> dict:
    return {
        "phoneme": "n",
        "language": "zh",
        "surface_tone": 2,
        "stress": 0,
        "syllable_role": "onset",
        "articulatory": {
            "type": "consonant",
            "height": None,
            "backness": None,
            "rounded": None,
            "place": "alveolar",
            "manner": "nasal",
            "voiced": True,
            "aspirated": False,
        },
        "boundary": {
            "word_start": True,
            "word_end": False,
            "phrase_start": True,
            "phrase_end": False,
            "boundary_type": "none",
        },
        "labels": {
            "viseme_id": 14,
            "strength": 62.0,
            "viseme_source": "rule",
            "strength_source": "pseudo_strength_v1",
        },
    }


def _minimal_sample(**overrides) -> dict:
    sample = {
        "schema_version": "articulm_v1_sample_v1",
        "sample_id": "unit_001",
        "text": "你",
        "tokens": [_minimal_token()],
    }
    sample.update(overrides)
    return sample


# -- fixtures parse ---------------------------------------------------------


def test_all_fixtures_parse(fixture_paths, data_config):
    for name, path in fixture_paths.items():
        samples = load_samples(path, data_config)
        assert samples, f"{name} fixture is empty"
        for sample in samples:
            assert sample.has_labels
            assert len(sample.tokens) >= 1


def test_documented_nihao_sample_matches_docs(fixture_paths, data_config):
    """The canonical docs/12 sample keeps its documented structure."""
    samples = load_samples(fixture_paths["zh"], data_config)
    nihao = next(s for s in samples if s.sample_id == "zh_nihao_001")
    assert [t.phoneme for t in nihao.tokens] == ["n", "i", "x", "a"]
    assert [t.surface_tone for t in nihao.tokens] == [2, 2, 3, 3]
    assert all(t.stress == 0 for t in nihao.tokens)
    assert all(t.language == "zh" for t in nihao.tokens)
    assert [t.syllable_role for t in nihao.tokens] == [
        "onset",
        "nucleus",
        "onset",
        "nucleus",
    ]
    assert nihao.tokens[-1].boundary.phrase_end == "true"
    assert nihao.tokens[-1].boundary.boundary_type == "major"
    # Null articulatory fields normalise to [NA], not to a silent default.
    assert nihao.tokens[0].articulatory.height == NA
    assert nihao.tokens[1].articulatory.place == NA


def test_english_fixture_conventions(fixture_paths, data_config):
    samples = load_samples(fixture_paths["en"], data_config)
    for sample in samples:
        for token in sample.tokens:
            assert token.language == "en"
            assert token.surface_tone == 0
            assert token.stress in (0, 1, 2)


def test_mixed_fixture_switches_language_within_a_sentence(fixture_paths, data_config):
    samples = load_samples(fixture_paths["mixed"], data_config)
    languages = {token.language for sample in samples for token in sample.tokens}
    assert languages == {"zh", "en"}


# -- label range validation -------------------------------------------------


@pytest.mark.parametrize("viseme_id", [-1, 18, 99])
def test_out_of_range_viseme_is_rejected(data_config, viseme_id):
    sample = _minimal_sample()
    sample["tokens"][0]["labels"]["viseme_id"] = viseme_id
    with pytest.raises(SchemaError, match="viseme_id"):
        parse_sample(sample, data_config)


@pytest.mark.parametrize("strength", [-0.1, 100.1, 1000.0])
def test_out_of_range_strength_is_rejected(data_config, strength):
    sample = _minimal_sample()
    sample["tokens"][0]["labels"]["strength"] = strength
    with pytest.raises(SchemaError, match="strength"):
        parse_sample(sample, data_config)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_strength_are_rejected(data_config, bad):
    sample = _minimal_sample()
    sample["tokens"][0]["labels"]["strength"] = bad
    with pytest.raises(SchemaError, match="NaN/Inf"):
        parse_sample(sample, data_config)


def test_viseme_and_strength_boundaries_are_inclusive(data_config):
    for viseme_id, strength in ((0, 0.0), (17, 100.0)):
        sample = _minimal_sample()
        sample["tokens"][0]["labels"]["viseme_id"] = viseme_id
        sample["tokens"][0]["labels"]["strength"] = strength
        parsed = parse_sample(sample, data_config)
        assert parsed.tokens[0].labels is not None
        assert parsed.tokens[0].labels.viseme_id == viseme_id
        assert parsed.tokens[0].labels.strength == strength


# -- language conventions ---------------------------------------------------


@pytest.mark.parametrize("tone", [0, 6, -1])
def test_chinese_tone_must_be_1_to_5(data_config, tone):
    sample = _minimal_sample()
    sample["tokens"][0]["surface_tone"] = tone
    with pytest.raises(SchemaError, match="Chinese surface_tone"):
        parse_sample(sample, data_config)


def test_chinese_stress_must_be_zero(data_config):
    sample = _minimal_sample()
    sample["tokens"][0]["stress"] = 1
    with pytest.raises(SchemaError, match="Chinese stress"):
        parse_sample(sample, data_config)


def test_english_tone_must_be_zero(data_config):
    sample = _minimal_sample()
    sample["tokens"][0].update({"language": "en", "surface_tone": 3, "stress": 1})
    with pytest.raises(SchemaError, match="English surface_tone"):
        parse_sample(sample, data_config)


def test_english_stress_must_be_0_1_2(data_config):
    sample = _minimal_sample()
    sample["tokens"][0].update({"language": "en", "surface_tone": 0, "stress": 3})
    with pytest.raises(SchemaError, match="English stress"):
        parse_sample(sample, data_config)


def test_unsupported_language_is_rejected(data_config):
    sample = _minimal_sample()
    sample["tokens"][0]["language"] = "ja"
    with pytest.raises(SchemaError, match=r"language.*supported"):
        parse_sample(sample, data_config)


# -- leakage ---------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("viseme_id", 2),
        ("strength", 80.0),
        ("shapeV2", "mouth_open"),
        ("Talk", 1),
        ("raw_value", 100),
    ],
)
def test_target_fields_at_feature_level_are_rejected(data_config, field, value):
    sample = _minimal_sample()
    sample["tokens"][0][field] = value
    with pytest.raises(SchemaError, match="must not appear as encoder features"):
        parse_sample(sample, data_config)


def test_leakage_inside_a_features_block_is_rejected(data_config):
    sample = _minimal_sample()
    sample["tokens"][0]["features"] = {"phoneme": "a", "viseme_id": 2}
    with pytest.raises(SchemaError, match="must not appear as encoder features"):
        parse_sample(sample, data_config)


def test_teacher_metadata_is_allowed_but_never_parsed_as_a_feature(data_config):
    sample = _minimal_sample(
        teacher_metadata={"raw_value": 100, "shapeV2": "x", "duration": 0.08}
    )
    sample["tokens"][0]["labels"]["strength_source"] = "pseudo_strength_v1"
    parsed = parse_sample(sample, data_config)
    assert parsed.teacher_metadata["raw_value"] == 100
    token = parsed.tokens[0]
    # The token dataclass has no field that could carry teacher metadata.
    assert not hasattr(token, "raw_value")
    assert not hasattr(token, "duration")
    assert not hasattr(token, "shapeV2")


def test_raw_value_relabelled_as_human_gold_is_rejected(data_config):
    """docs/12 section 12: a website rule value must not be called Human Gold."""
    sample = _minimal_sample()
    sample["tokens"][0]["labels"].update(
        {"strength": 100.0, "strength_source": "human", "raw_value": 100}
    )
    with pytest.raises(SchemaError, match="must not be relabelled as Human Gold"):
        parse_sample(sample, data_config)


def test_pseudo_strength_is_not_reported_as_human_gold(fixture_paths, data_config):
    samples = load_samples(fixture_paths["zh"], data_config)
    for sample in samples:
        for token in sample.tokens:
            assert token.labels is not None
            assert token.labels.strength_source == "pseudo_strength_v1"
            assert token.labels.is_human_gold_strength is False


# -- structural validation -------------------------------------------------


def test_missing_tokens_list_is_rejected(data_config):
    with pytest.raises(SchemaError, match="missing 'tokens'"):
        parse_sample({"sample_id": "x", "phoneme": "a"}, data_config)


def test_empty_tokens_list_is_rejected(data_config):
    with pytest.raises(SchemaError, match="must not be empty"):
        parse_sample(_minimal_sample(tokens=[]), data_config)


def test_token_label_count_mismatch_is_rejected(data_config):
    sample = _minimal_sample()
    second = copy.deepcopy(sample["tokens"][0])
    second.pop("labels")
    sample["tokens"].append(second)
    with pytest.raises(SchemaError, match="missing 'labels'"):
        parse_sample(sample, data_config)


def test_labels_optional_for_inference_input(data_config):
    sample = _minimal_sample()
    sample["tokens"][0].pop("labels")
    parsed = parse_sample(sample, data_config, require_labels=False)
    assert parsed.tokens[0].labels is None
    assert parsed.has_labels is False


def test_schema_version_mismatch_is_rejected(data_config):
    sample = _minimal_sample(schema_version="articulm_v0_legacy")
    with pytest.raises(SchemaError, match="schema_version"):
        parse_sample(sample, data_config)


def test_sequence_longer_than_max_seq_len_is_rejected(data_config):
    sample = _minimal_sample()
    sample["tokens"] = [copy.deepcopy(sample["tokens"][0]) for _ in range(data_config.max_seq_len + 1)]
    with pytest.raises(SchemaError, match="exceeds"):
        parse_sample(sample, data_config)


def test_unknown_articulatory_field_is_rejected(data_config):
    sample = _minimal_sample()
    sample["tokens"][0]["articulatory"]["mystery"] = "x"
    with pytest.raises(SchemaError, match="unknown fields"):
        parse_sample(sample, data_config)


def test_missing_phoneme_is_rejected(data_config):
    sample = _minimal_sample()
    sample["tokens"][0]["phoneme"] = "  "
    with pytest.raises(SchemaError, match="phoneme"):
        parse_sample(sample, data_config)
