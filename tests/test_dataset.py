"""Dataset encoding: sentence-level units, label separation, weights."""

from __future__ import annotations

import pytest
import torch

from articulm.data.dataset import PhonemeSequenceDataset, encode_sample
from articulm.data.schema import load_samples
from articulm.data.vocab import FEATURE_KEYS, PAD_ID


def test_one_item_is_one_sentence(dataset, all_samples):
    assert len(dataset) == len(all_samples)
    for item, sample in zip(dataset.encoded, all_samples, strict=True):
        assert item.length == len(sample.tokens)
        assert item.sample_id == sample.sample_id


def test_no_sentence_is_split_into_independent_rows(dataset):
    """Multi-token sentences stay whole; a length-1 dataset item would mean
    the sentence had exactly one phoneme, not that it was split."""
    assert max(dataset.lengths) > 1
    assert dataset.num_tokens == sum(dataset.lengths)


def test_feature_ids_shape_and_range(dataset, vocab):
    sizes = vocab.sizes()
    for item in dataset.encoded:
        assert item.feature_ids.shape == (item.length, len(FEATURE_KEYS))
        assert item.feature_ids.dtype == torch.long
        for index, key in enumerate(FEATURE_KEYS):
            column = item.feature_ids[:, index]
            assert int(column.min()) >= 0
            assert int(column.max()) < sizes[key]
            # Real tokens never encode to PAD.
            assert int(column.min()) != PAD_ID


def test_labels_are_normalised_to_0_1(dataset):
    for item in dataset.encoded:
        assert item.viseme_ids is not None
        assert item.strength is not None
        assert item.viseme_ids.dtype == torch.long
        assert int(item.viseme_ids.min()) >= 0
        assert int(item.viseme_ids.max()) <= 17
        assert float(item.strength.min()) >= 0.0
        assert float(item.strength.max()) <= 1.0


def test_strength_normalisation_matches_raw_labels(all_samples, vocab):
    sample = all_samples[0]
    encoded = encode_sample(sample, vocab)
    assert encoded.strength is not None
    for index, token in enumerate(sample.tokens):
        assert token.labels is not None
        assert float(encoded.strength[index]) == pytest.approx(
            token.labels.strength / 100.0, abs=1e-6
        )


def test_source_weights_are_applied_per_token(all_samples, vocab):
    weighted = encode_sample(
        all_samples[0], vocab, source_weights={"pseudo_strength_v1": 0.3}
    )
    assert weighted.strength_weight is not None
    assert torch.allclose(
        weighted.strength_weight, torch.full_like(weighted.strength_weight, 0.3)
    )


def test_unknown_source_defaults_to_weight_one(all_samples, vocab):
    encoded = encode_sample(all_samples[0], vocab, source_weights={"human": 1.0})
    assert encoded.strength_weight is not None
    assert torch.allclose(
        encoded.strength_weight, torch.ones_like(encoded.strength_weight)
    )


def test_pseudo_labels_are_not_flagged_human_gold(dataset):
    for item in dataset.encoded:
        assert item.human_gold_strength is not None
        assert not bool(item.human_gold_strength.any())


def test_human_gold_flag_is_set_for_human_source(all_samples, vocab, data_config):
    import copy
    import dataclasses

    from articulm.data.schema import TokenLabels

    sample = all_samples[0]
    tokens = list(copy.deepcopy(sample).tokens)
    tokens[0] = dataclasses.replace(
        tokens[0],
        labels=TokenLabels(
            viseme_id=2, strength=74.0, viseme_source="human", strength_source="human"
        ),
    )
    patched = dataclasses.replace(sample, tokens=tuple(tokens))
    encoded = encode_sample(patched, vocab)
    assert encoded.human_gold_strength is not None
    assert bool(encoded.human_gold_strength[0])
    assert not bool(encoded.human_gold_strength[1:].any())


def test_inference_samples_carry_no_label_tensors(fixture_paths, data_config, vocab):
    import json

    with fixture_paths["zh"].open("r", encoding="utf-8") as fh:
        record = json.loads(fh.readline())
    for token in record["tokens"]:
        token.pop("labels")

    from articulm.data.schema import parse_sample

    sample = parse_sample(record, data_config, require_labels=False)
    encoded = encode_sample(sample, vocab)
    assert encoded.viseme_ids is None
    assert encoded.strength is None
    assert encoded.strength_weight is None


def test_slice_metadata_is_populated(dataset):
    for item in dataset.encoded:
        assert len(item.phonemes) == item.length
        assert len(item.languages) == item.length
        assert len(item.surface_tones) == item.length
        assert len(item.stresses) == item.length
        assert len(item.syllable_roles) == item.length
        assert len(item.phrase_positions) == item.length


def test_from_jsonl_and_subset(fixture_paths, data_config, vocab):
    loaded = PhonemeSequenceDataset.from_jsonl(fixture_paths["zh"], data_config, vocab)
    assert len(loaded) == 3
    subset = loaded.subset(2)
    assert len(subset) == 2
    assert subset.encoded[0].sample_id == loaded.encoded[0].sample_id
    assert subset.vocab is loaded.vocab


def test_empty_dataset_is_rejected(vocab):
    with pytest.raises(ValueError, match="at least one sample"):
        PhonemeSequenceDataset([], vocab)


def test_encoding_is_deterministic(all_samples, vocab):
    first = encode_sample(all_samples[0], vocab)
    second = encode_sample(all_samples[0], vocab)
    assert torch.equal(first.feature_ids, second.feature_ids)
    assert first.viseme_ids is not None and second.viseme_ids is not None
    assert torch.equal(first.viseme_ids, second.viseme_ids)


def test_all_fixture_sets_load_together(fixture_paths, data_config):
    total = 0
    for path in fixture_paths.values():
        total += len(load_samples(path, data_config))
    assert total == 5


# ------------------------------------------------------------ encode cache

def test_encode_cache_roundtrip(all_samples, vocab, fixture_paths, data_config, tmp_path):
    """Second construction with the same corpus/vocab hits the cache and
    returns identical tensors; a changed corpus is a miss, never stale."""
    import os
    import shutil

    source = tmp_path / "train.jsonl"
    shutil.copy(next(iter(fixture_paths.values())), source)
    corpus = load_samples(source, data_config)
    cache = tmp_path / "train.enc-cache.pt"

    first = PhonemeSequenceDataset(corpus, vocab, cache_path=cache, source_path=source)
    assert first.cache_state == "saved"
    assert cache.is_file()

    second = PhonemeSequenceDataset(corpus, vocab, cache_path=cache, source_path=source)
    assert second.cache_state == "hit"
    for a, b in zip(first.encoded, second.encoded, strict=True):
        assert torch.equal(a.feature_ids, b.feature_ids)
        assert a.length == b.length
        assert a.sample_id == b.sample_id

    # Touching the corpus invalidates the key: stale encodings must not load.
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1))
    third = PhonemeSequenceDataset(corpus, vocab, cache_path=cache, source_path=source)
    assert third.cache_state == "saved"


def test_encode_cache_disabled_without_argument(all_samples, vocab):
    plain = PhonemeSequenceDataset(all_samples, vocab)
    assert plain.cache_state == "disabled"


def test_corrupt_cache_falls_back_to_encoding(all_samples, vocab, fixture_paths, data_config, tmp_path):
    import shutil

    source = tmp_path / "train.jsonl"
    shutil.copy(next(iter(fixture_paths.values())), source)
    corpus = load_samples(source, data_config)
    cache = tmp_path / "bad.enc-cache.pt"
    cache.write_bytes(b"not a torch file")

    ds = PhonemeSequenceDataset(corpus, vocab, cache_path=cache, source_path=source)
    # The corrupt file is a miss; a fresh cache is written over it.
    assert ds.cache_state == "saved"
    assert len(ds.encoded) == len(corpus)
