"""Vocabulary stability, reserved indices and round-tripping."""

from __future__ import annotations

import pytest

from articulm.data.schema import ARTICULATORY_FIELDS, BOUNDARY_FIELDS
from articulm.data.vocab import (
    FEATURE_KEYS,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    CategoricalVocabulary,
    FeatureVocabulary,
    VocabError,
    build_vocabulary,
)


def test_feature_keys_cover_every_documented_field():
    assert FEATURE_KEYS[:5] == (
        "phoneme",
        "language",
        "surface_tone",
        "stress",
        "syllable_role",
    )
    for name in ARTICULATORY_FIELDS:
        assert f"articulatory.{name}" in FEATURE_KEYS
    for name in BOUNDARY_FIELDS:
        assert f"boundary.{name}" in FEATURE_KEYS
    assert len(FEATURE_KEYS) == 5 + len(ARTICULATORY_FIELDS) + len(BOUNDARY_FIELDS) == 18
    assert len(set(FEATURE_KEYS)) == len(FEATURE_KEYS)


def test_no_label_field_is_ever_a_vocabulary_field():
    forbidden = {"viseme_id", "strength", "shapeV2", "Talk", "raw_value", "duration", "timing"}
    assert forbidden.isdisjoint(set(FEATURE_KEYS))


def test_every_field_reserves_pad_and_unk(vocab):
    for key in FEATURE_KEYS:
        table = vocab.fields[key]
        assert table.tokens[PAD_ID] == PAD_TOKEN
        assert table.tokens[UNK_ID] == UNK_TOKEN
        assert len(table) >= 2


def test_phoneme_vocabulary_includes_pad_and_unk(vocab):
    phoneme = vocab.fields["phoneme"]
    assert PAD_TOKEN in phoneme
    assert UNK_TOKEN in phoneme
    assert phoneme.encode(PAD_TOKEN) == PAD_ID
    assert phoneme.encode(UNK_TOKEN) == UNK_ID


def test_unseen_token_maps_to_unk(vocab):
    assert vocab.fields["phoneme"].encode("!!definitely_not_a_phoneme!!") == UNK_ID
    assert vocab.unknown_phoneme("!!definitely_not_a_phoneme!!") is True


def test_closed_sets_are_seeded_from_spec_not_from_corpus(all_samples, data_config):
    """Tone/stress ids must not depend on which sentences were scanned."""
    full = build_vocabulary(all_samples, data_config)
    subset = build_vocabulary(all_samples[:1], data_config)
    for key in (
        "language",
        "surface_tone",
        "stress",
        "syllable_role",
        "boundary.boundary_type",
    ):
        assert full.fields[key].tokens == subset.fields[key].tokens, key


def test_tone_and_stress_cover_documented_values(vocab):
    tones = vocab.fields["surface_tone"]
    for value in ("0", "1", "2", "3", "4", "5"):
        assert value in tones
    stresses = vocab.fields["stress"]
    for value in ("0", "1", "2"):
        assert value in stresses


def test_encode_token_produces_one_id_per_field(all_samples, vocab):
    token = all_samples[0].tokens[0]
    ids = vocab.encode_token(token)
    assert set(ids) == set(FEATURE_KEYS)
    for key, value in ids.items():
        assert 0 <= value < len(vocab.fields[key]), key
        # A real token never encodes to PAD.
        assert value != PAD_ID, key


def test_save_load_round_trip(vocab, tmp_path):
    path = tmp_path / "vocab.json"
    vocab.save(path)
    reloaded = FeatureVocabulary.load(path)
    assert reloaded.sizes() == vocab.sizes()
    assert reloaded.viseme_classes == vocab.viseme_classes
    for key in FEATURE_KEYS:
        assert reloaded.fields[key].tokens == vocab.fields[key].tokens


def test_load_rejects_wrong_format_version(vocab, tmp_path):
    raw = vocab.to_dict()
    raw["format_version"] = "articulm_v0_vocab"
    path = tmp_path / "vocab.json"
    import json

    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(VocabError, match="format_version"):
        FeatureVocabulary.load(path)


def test_categorical_vocabulary_rejects_missing_reserved_tokens():
    with pytest.raises(VocabError, match="must be"):
        CategoricalVocabulary(name="x", tokens=["a", "b", "c"])


def test_categorical_vocabulary_rejects_duplicates():
    with pytest.raises(VocabError, match="duplicate"):
        CategoricalVocabulary(name="x", tokens=[PAD_TOKEN, UNK_TOKEN, "a", "a"])


def test_add_is_idempotent():
    table = CategoricalVocabulary.build("x", ["a"])
    first = table.add("b")
    assert table.add("b") == first
    assert len(table) == 4


def test_decode_round_trips(vocab):
    table = vocab.fields["phoneme"]
    for index in range(len(table)):
        assert table.encode(table.decode(index)) == index


def test_feature_vocabulary_rejects_missing_field(vocab):
    fields = dict(vocab.fields)
    fields.pop("phoneme")
    with pytest.raises(VocabError, match="missing vocabularies"):
        FeatureVocabulary(
            fields=fields, viseme_classes=16, strength_min=0.0, strength_max=100.0
        )
