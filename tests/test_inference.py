"""ModelPredictor: checkpoint loading and baseline-decoding prediction."""

from __future__ import annotations

from pathlib import Path

import pytest

from articulm.inference import ModelPredictor
from articulm.model.articulm_v1 import ArticuLMV1
from articulm.training.checkpoint import TrainingState, save_checkpoint
from articulm.visemes import VISEME_NAMES


@pytest.fixture
def saved_checkpoint(
    tiny_model_config, model_config, data_config, vocab, tmp_path
) -> Path:
    """A real (untrained) checkpoint saved from the tiny model + fixtures."""
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    return save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(global_step=42, best_metric=0.9, best_step=42),
        seed=7,
    )


def test_load_restores_model_vocab_and_config(saved_checkpoint, vocab):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    assert isinstance(predictor.model, ArticuLMV1)
    assert predictor.vocab.to_dict() == vocab.to_dict()
    assert predictor.num_viseme_classes == 18
    assert str(predictor.device) == "cpu"


def test_predict_samples_returns_names_by_default(saved_checkpoint, all_samples):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    results = predictor.predict_samples(all_samples)

    assert len(results) == len(all_samples)
    for result, sample in zip(results, all_samples, strict=True):
        assert result["sample_id"] == sample.sample_id
        assert len(result["outputs"]) == len(sample.tokens)
        for item, token in zip(result["outputs"], sample.tokens, strict=True):
            assert item["phoneme"] == token.phoneme
            assert isinstance(item["viseme"], str)
            assert item["viseme"] in VISEME_NAMES
            assert "viseme_id" not in item
            assert 0.0 <= item["strength"] <= 100.0


def test_predict_samples_can_return_numeric_ids(saved_checkpoint, all_samples):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    results = predictor.predict_samples(all_samples, label_names=False)
    for result in results:
        for item in result["outputs"]:
            assert isinstance(item["viseme_id"], int)
            assert 0 <= item["viseme_id"] < 18
            assert "viseme" not in item


def test_predict_is_deterministic(saved_checkpoint, all_samples):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    first = predictor.predict_samples(all_samples)
    second = predictor.predict_samples(all_samples)
    assert first == second


def test_predict_jsonl_from_fixture(saved_checkpoint, fixture_paths):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    path = next(iter(fixture_paths.values()))
    results = predictor.predict_jsonl(path)
    assert results
    for result in results:
        assert result["outputs"]
        assert all("strength" in o and "viseme" in o for o in result["outputs"])


def test_predictor_matches_infer_cli_decoding(saved_checkpoint, fixture_paths):
    """The programmatic API and the CLI must share one decoding path."""
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    path = next(iter(fixture_paths.values()))
    api = predictor.predict_jsonl(path, label_names=False)

    # Reuse the same samples parse + predict_samples the CLI uses.
    from articulm.infer import load_inference_samples, predict_samples

    samples = load_inference_samples(path, predictor.data_config)
    cli = predict_samples(
        predictor.model,
        samples,
        predictor.vocab,
        device=predictor.device,
        batch_size=16,
        max_seq_len=predictor.data_config.max_seq_len,
        strength_scale=predictor.strength_scale,
        round_strength=1,
    )
    assert api == cli
