"""Padding, masking and batching — the padding-leakage contract."""

from __future__ import annotations

import pytest
import torch

from articulm.data.collator import (
    IGNORE_INDEX,
    Batch,
    DynamicTokenBatchSampler,
    PhonemeCollator,
    build_dataloader,
    length_bucket,
)
from articulm.data.vocab import FEATURE_KEYS, PAD_ID


@pytest.fixture
def batch(dataset) -> Batch:
    return PhonemeCollator()(list(dataset.encoded))


def test_batch_shapes(batch, dataset):
    batch_size = len(dataset)
    max_len = max(dataset.lengths)
    assert batch.feature_ids.shape == (batch_size, max_len, len(FEATURE_KEYS))
    assert batch.attention_mask.shape == (batch_size, max_len)
    assert batch.loss_mask.shape == (batch_size, max_len)
    assert batch.viseme_targets is not None
    assert batch.viseme_targets.shape == (batch_size, max_len)
    assert batch.strength_targets is not None
    assert batch.strength_targets.shape == (batch_size, max_len)


def test_attention_mask_matches_true_lengths(batch, dataset):
    for row, length in enumerate(dataset.lengths):
        assert bool(batch.attention_mask[row, :length].all())
        assert not bool(batch.attention_mask[row, length:].any())
        assert int(batch.lengths[row]) == length


def test_loss_mask_never_exceeds_attention_mask(batch):
    assert not bool((batch.loss_mask & ~batch.attention_mask).any())
    assert batch.num_supervised_tokens == batch.num_real_tokens


def test_padded_positions_use_pad_id(batch, dataset):
    for row, length in enumerate(dataset.lengths):
        padded = batch.feature_ids[row, length:]
        if padded.numel():
            assert bool((padded == PAD_ID).all())


def test_padded_viseme_targets_use_ignore_index(batch, dataset):
    assert batch.viseme_targets is not None
    for row, length in enumerate(dataset.lengths):
        tail = batch.viseme_targets[row, length:]
        if tail.numel():
            assert bool((tail == IGNORE_INDEX).all())
        head = batch.viseme_targets[row, :length]
        assert int(head.min()) >= 0
        assert int(head.max()) <= 17


def test_padded_strength_targets_and_weights_are_zero(batch, dataset):
    assert batch.strength_targets is not None
    assert batch.strength_weight is not None
    for row, length in enumerate(dataset.lengths):
        assert float(batch.strength_targets[row, length:].abs().sum()) == 0.0
        assert float(batch.strength_weight[row, length:].abs().sum()) == 0.0


def test_documented_padding_example(dataset, vocab):
    """docs/12 section 8: a 4-token and a 7-token sequence pad to 7."""
    short = next(item for item in dataset.encoded if item.length == 4)
    long_item = max(dataset.encoded, key=lambda item: item.length)
    collated = PhonemeCollator()([short, long_item])
    assert collated.max_length == long_item.length
    assert collated.attention_mask[0].tolist() == (
        [True] * 4 + [False] * (long_item.length - 4)
    )
    assert collated.attention_mask[1].tolist() == [True] * long_item.length
    assert collated.loss_mask[0].tolist() == collated.attention_mask[0].tolist()


def test_flat_metadata_aligns_with_masked_row_major_order(batch, dataset):
    """Slice arrays must line up with `tensor[mask]` selection order."""
    total = sum(dataset.lengths)
    assert len(batch.phonemes) == total
    assert batch.slices is not None
    for name, values in batch.slices.items():
        assert len(values) == total, name

    expected = [p for item in dataset.encoded for p in item.phonemes]
    assert list(batch.phonemes) == expected
    expected_languages = [lang for item in dataset.encoded for lang in item.languages]
    assert list(batch.slices["language"]) == expected_languages


def test_masked_selection_order_matches_metadata(batch, dataset):
    assert batch.viseme_targets is not None
    selected = batch.viseme_targets[batch.loss_mask]
    expected = torch.cat([item.viseme_ids for item in dataset.encoded])
    assert torch.equal(selected, expected)


def test_mixing_labelled_and_unlabelled_is_rejected(dataset, all_samples, vocab):
    import dataclasses

    from articulm.data.dataset import encode_sample

    labelled = dataset.encoded[0]
    unlabelled = dataclasses.replace(
        encode_sample(all_samples[1], vocab),
        viseme_ids=None,
        strength=None,
        strength_weight=None,
        human_gold_strength=None,
    )
    with pytest.raises(ValueError, match="mix labelled and unlabelled"):
        PhonemeCollator()([labelled, unlabelled])


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="empty batch"):
        PhonemeCollator()([])


def test_sequence_over_max_seq_len_is_rejected(dataset):
    collator = PhonemeCollator(max_seq_len=3)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        collator(list(dataset.encoded))


def test_batch_to_device_keeps_metadata(batch):
    moved = batch.to("cpu")
    assert moved.sample_ids == batch.sample_ids
    assert moved.phonemes == batch.phonemes
    assert moved.slices == batch.slices
    assert torch.equal(moved.attention_mask, batch.attention_mask)


# -- batching strategies ----------------------------------------------------


def test_fixed_sample_dataloader_covers_every_sentence(dataset):
    loader = build_dataloader(
        dataset, strategy="fixed_samples", batch_size=2, shuffle=False
    )
    seen = [sid for batch in loader for sid in batch.sample_ids]
    assert sorted(seen) == sorted(item.sample_id for item in dataset.encoded)


def test_dynamic_token_batching_respects_budget(dataset):
    budget = 100
    sampler = DynamicTokenBatchSampler(
        dataset.lengths, budget, shuffle=False
    )
    batches = list(sampler)
    covered = sorted(index for batch in batches for index in batch)
    assert covered == list(range(len(dataset)))
    for indices in batches:
        max_len = max(dataset.lengths[i] for i in indices)
        assert max_len * len(indices) <= budget


def test_dynamic_batching_rejects_unfittable_budget(dataset):
    with pytest.raises(ValueError, match="cannot hold the longest"):
        DynamicTokenBatchSampler(dataset.lengths, 2)


def test_dynamic_sampler_shuffle_is_seed_stable(dataset):
    first = DynamicTokenBatchSampler(dataset.lengths, 200, shuffle=True, seed=7)
    second = DynamicTokenBatchSampler(dataset.lengths, 200, shuffle=True, seed=7)
    assert list(first) == list(second)


def test_dynamic_sampler_epoch_reorders_without_changing_coverage(dataset):
    """set_epoch changes batch order only; the partition itself is stable."""
    sampler = DynamicTokenBatchSampler(dataset.lengths, 200, shuffle=True, seed=7)
    epoch_zero = sorted(tuple(batch) for batch in sampler)
    sampler.set_epoch(1)
    epoch_one = sorted(tuple(batch) for batch in sampler)
    assert epoch_zero == epoch_one
    covered = sorted(index for batch in epoch_one for index in batch)
    assert covered == list(range(len(dataset)))


def test_dynamic_sampler_epoch_changes_order_on_many_batches():
    """With enough batches, consecutive epochs must not be identical."""
    lengths = list(range(1, 41))
    sampler = DynamicTokenBatchSampler(lengths, 60, shuffle=True, seed=7)
    epoch_zero = list(sampler)
    assert len(epoch_zero) > 5
    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero


def test_dynamic_dataloader_produces_valid_batches(dataset):
    loader = build_dataloader(
        dataset,
        strategy="dynamic_phoneme_tokens",
        max_phoneme_tokens_per_batch=200,
        shuffle=False,
    )
    total = 0
    for batch in loader:
        total += batch.batch_size
        assert batch.max_length * batch.batch_size <= 200
        assert not bool((batch.loss_mask & ~batch.attention_mask).any())
    assert total == len(dataset)


def test_unknown_strategy_is_rejected(dataset):
    with pytest.raises(ValueError, match="unknown batching strategy"):
        build_dataloader(dataset, strategy="magic")


@pytest.mark.parametrize(
    "length,expected",
    [(1, "len_1_8"), (8, "len_1_8"), (9, "len_9_16"), (40, "len_33_64"), (500, "len_129_plus")],
)
def test_length_buckets(length, expected):
    assert length_bucket(length) == expected
