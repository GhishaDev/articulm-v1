"""Metric correctness, slice breakdowns and padding exclusion."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from articulm.data.collator import SLICE_FIELDS, PhonemeCollator
from articulm.model.articulm_v1 import ArticuLMV1
from articulm.training.metrics import (
    MetricsAccumulator,
    masked_accuracy,
    masked_strength_mae,
)


@pytest.fixture
def batch(dataset):
    return PhonemeCollator()(list(dataset.encoded))


@pytest.fixture
def model(tiny_model_config, vocab) -> ArticuLMV1:
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    model.eval()
    return model


# -- classification metrics ------------------------------------------------


def test_perfect_predictions_give_perfect_scores():
    accumulator = MetricsAccumulator(num_classes=18)
    targets = np.array([0, 3, 14, 15, 2])
    strength = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    accumulator.update(targets, targets, strength, strength)
    report = accumulator.compute()
    assert report.viseme.accuracy == pytest.approx(1.0)
    assert report.viseme.macro_f1 == pytest.approx(1.0)
    assert report.viseme.weighted_f1 == pytest.approx(1.0)
    assert report.strength.mae == pytest.approx(0.0)
    assert report.strength.rmse == pytest.approx(0.0)
    assert report.strength.median_absolute_error == pytest.approx(0.0)


def test_all_wrong_predictions_give_zero_accuracy():
    accumulator = MetricsAccumulator(num_classes=18)
    targets = np.array([0, 1, 2, 3])
    predictions = np.array([4, 5, 6, 7])
    values = np.zeros(4)
    accumulator.update(predictions, targets, values, values)
    report = accumulator.compute()
    assert report.viseme.accuracy == pytest.approx(0.0)
    assert report.viseme.macro_f1 == pytest.approx(0.0)


def test_accuracy_matches_hand_computation():
    accumulator = MetricsAccumulator(num_classes=4)
    targets = np.array([0, 0, 1, 1, 2, 3])
    predictions = np.array([0, 1, 1, 1, 2, 0])
    values = np.zeros(6)
    accumulator.update(predictions, targets, values, values)
    report = accumulator.compute()
    # correct: index 0, 2, 3, 4 -> 4/6
    assert report.viseme.accuracy == pytest.approx(4 / 6)


def test_confusion_matrix_is_target_by_prediction():
    accumulator = MetricsAccumulator(num_classes=3)
    targets = np.array([0, 0, 1, 2])
    predictions = np.array([0, 1, 1, 0])
    values = np.zeros(4)
    accumulator.update(predictions, targets, values, values)
    matrix = accumulator.compute().viseme.confusion_matrix
    assert matrix[0] == [1, 1, 0]
    assert matrix[1] == [0, 1, 0]
    assert matrix[2] == [1, 0, 0]


def test_per_class_precision_recall_f1():
    accumulator = MetricsAccumulator(num_classes=2)
    #      target: 0 0 0 1 1
    #  prediction: 0 0 1 1 0
    targets = np.array([0, 0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1, 0])
    values = np.zeros(5)
    accumulator.update(predictions, targets, values, values)
    viseme = accumulator.compute().viseme
    # class 0: tp=2, predicted=3 -> precision 2/3; actual=3 -> recall 2/3
    assert viseme.per_class_precision[0] == pytest.approx(2 / 3)
    assert viseme.per_class_recall[0] == pytest.approx(2 / 3)
    # class 1: tp=1, predicted=2 -> precision 1/2; actual=2 -> recall 1/2
    assert viseme.per_class_precision[1] == pytest.approx(0.5)
    assert viseme.per_class_recall[1] == pytest.approx(0.5)
    assert viseme.support == [3, 2]


def test_macro_average_ignores_absent_classes():
    """A class with no examples must not drag macro-F1 toward zero."""
    accumulator = MetricsAccumulator(num_classes=18)
    targets = np.array([0, 0, 1, 1])
    values = np.zeros(4)
    accumulator.update(targets, targets, values, values)
    report = accumulator.compute()
    assert report.viseme.macro_f1 == pytest.approx(1.0)
    assert report.viseme.support[2] == 0


def test_weighted_f1_uses_support():
    accumulator = MetricsAccumulator(num_classes=2)
    targets = np.array([0] * 9 + [1])
    predictions = np.array([0] * 9 + [0])
    values = np.zeros(10)
    accumulator.update(predictions, targets, values, values)
    viseme = accumulator.compute().viseme
    # class 0 f1 = 2*0.9*1/(1.9); class 1 f1 = 0; weighted by 9 and 1.
    expected = (9 * (2 * 0.9 * 1.0 / 1.9) + 1 * 0.0) / 10
    assert viseme.weighted_f1 == pytest.approx(expected)


# -- regression metrics ----------------------------------------------------


def test_mae_rmse_and_median_are_correct():
    accumulator = MetricsAccumulator(num_classes=2)
    targets = np.array([0, 0, 0, 0])
    predicted_strength = np.array([10.0, 20.0, 30.0, 40.0])
    target_strength = np.array([0.0, 0.0, 0.0, 0.0])
    accumulator.update(targets, targets, predicted_strength, target_strength)
    strength = accumulator.compute().strength
    assert strength.mae == pytest.approx(25.0)
    assert strength.rmse == pytest.approx(np.sqrt((100 + 400 + 900 + 1600) / 4))
    assert strength.median_absolute_error == pytest.approx(25.0)
    assert strength.count == 4


def test_per_viseme_mae():
    accumulator = MetricsAccumulator(num_classes=3)
    targets = np.array([0, 0, 1])
    predicted = np.array([10.0, 30.0, 5.0])
    actual = np.array([0.0, 0.0, 0.0])
    accumulator.update(targets, targets, predicted, actual)
    per_viseme = accumulator.compute().strength.per_viseme_mae
    assert per_viseme[0] == pytest.approx(20.0)
    assert per_viseme[1] == pytest.approx(5.0)
    assert 2 not in per_viseme


def test_metrics_are_reported_in_0_100_units(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18, strength_scale=100.0)
    accumulator.update_from_batch(output, batch)
    report = accumulator.compute()
    # A sigmoid model vs 0..100 labels cannot exceed 100 MAE.
    assert 0.0 <= report.strength.mae <= 100.0


# -- padding exclusion -----------------------------------------------------


def test_metrics_count_only_unpadded_tokens(model, batch, dataset):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    report = accumulator.compute()
    assert report.num_tokens == sum(dataset.lengths)
    assert report.num_tokens < batch.batch_size * batch.max_length
    assert report.num_sequences == batch.batch_size


def test_corrupt_padded_targets_do_not_change_metrics(model, dataset):
    batch = PhonemeCollator()(list(dataset.encoded))
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)

    clean = MetricsAccumulator(num_classes=18)
    clean.update_from_batch(output, batch)
    reference = clean.compute()

    poisoned = dataclasses.replace(
        batch,
        viseme_targets=batch.viseme_targets.clone(),
        strength_targets=batch.strength_targets.clone(),
    )
    padded = ~batch.attention_mask
    poisoned.viseme_targets[padded] = 5
    poisoned.strength_targets[padded] = 7.0

    dirty = MetricsAccumulator(num_classes=18)
    dirty.update_from_batch(output, poisoned)
    after = dirty.compute()

    assert after.viseme.accuracy == pytest.approx(reference.viseme.accuracy)
    assert after.strength.mae == pytest.approx(reference.strength.mae)
    assert after.num_tokens == reference.num_tokens


def test_batching_does_not_change_metrics(model, dataset):
    """Streaming in several batches must equal one big batch."""
    collator = PhonemeCollator()
    single = collator(list(dataset.encoded))
    with torch.no_grad():
        output = model(single.feature_ids, single.attention_mask)
    whole = MetricsAccumulator(num_classes=18)
    whole.update_from_batch(output, single)
    reference = whole.compute()

    streamed = MetricsAccumulator(num_classes=18)
    for item in dataset.encoded:
        chunk = collator([item])
        with torch.no_grad():
            chunk_output = model(chunk.feature_ids, chunk.attention_mask)
        streamed.update_from_batch(chunk_output, chunk)
    incremental = streamed.compute()

    assert incremental.num_tokens == reference.num_tokens
    assert incremental.viseme.accuracy == pytest.approx(reference.viseme.accuracy, abs=1e-5)
    assert incremental.strength.mae == pytest.approx(reference.strength.mae, abs=1e-3)


# -- slices ----------------------------------------------------------------


def test_all_documented_slices_are_produced(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    report = accumulator.compute()
    for field in ("language", "surface_tone", "stress", "syllable_role", "phrase_position", "length_bucket"):
        assert field in report.slices, field
        assert report.slices[field]
    assert set(SLICE_FIELDS) >= set(report.slices)


def test_slice_token_counts_sum_to_total(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    report = accumulator.compute()
    for field, values in report.slices.items():
        total = sum(int(stats["num_tokens"]) for stats in values.values())
        assert total == report.num_tokens, field


def test_language_slice_matches_fixture_composition(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    languages = accumulator.compute().slices["language"]
    assert set(languages) == {"zh", "en"}


def test_misaligned_slice_metadata_is_rejected():
    accumulator = MetricsAccumulator(num_classes=4)
    targets = np.array([0, 1, 2])
    values = np.zeros(3)
    with pytest.raises(ValueError, match="misaligned"):
        accumulator.update(
            targets, targets, values, values, slices={"language": ("zh", "zh")}
        )


def test_scalar_summary_keys(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    summary = accumulator.compute().scalar_summary("val_")
    assert set(summary) == {
        "val_viseme_accuracy",
        "val_viseme_macro_f1",
        "val_viseme_weighted_f1",
        "val_strength_mae",
        "val_strength_rmse",
    }


def test_report_serialises_to_plain_data(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    payload = accumulator.compute().as_dict(include_details=True)
    import json

    text = json.dumps(payload)
    assert "confusion_matrix" in text
    assert "per_viseme_mae" in text


def test_empty_accumulator_does_not_crash():
    report = MetricsAccumulator(num_classes=18).compute()
    assert report.num_tokens == 0
    assert report.viseme.accuracy == 0.0
    assert report.strength.count == 0


# -- quick logging helpers -------------------------------------------------


def test_masked_accuracy_matches_full_metric(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    quick = masked_accuracy(output.viseme_logits, batch.viseme_targets, batch.loss_mask)
    accumulator = MetricsAccumulator(num_classes=18)
    accumulator.update_from_batch(output, batch)
    assert quick == pytest.approx(accumulator.compute().viseme.accuracy, abs=1e-6)


def test_masked_strength_mae_matches_full_metric(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    quick = masked_strength_mae(
        output.strength_norm, batch.strength_targets, batch.loss_mask, 100.0
    )
    accumulator = MetricsAccumulator(num_classes=18, strength_scale=100.0)
    accumulator.update_from_batch(output, batch)
    assert quick == pytest.approx(accumulator.compute().strength.mae, rel=1e-4)


def test_quick_helpers_return_zero_on_empty_mask(model, batch):
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    empty = torch.zeros_like(batch.loss_mask)
    assert masked_accuracy(output.viseme_logits, batch.viseme_targets, empty) == 0.0
    assert (
        masked_strength_mae(output.strength_norm, batch.strength_targets, empty) == 0.0
    )
