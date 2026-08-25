"""predict_api_response: tdoge API response -> model predictions."""

from __future__ import annotations

from pathlib import Path

import pytest

from articulm.inference import ModelPredictor
from articulm.model.articulm_v1 import ArticuLMV1
from articulm.service import predict_api_response
from articulm.training.checkpoint import TrainingState, save_checkpoint
from articulm.visemes import VISEME_NAMES


@pytest.fixture
def saved_checkpoint(tiny_model_config, data_config, vocab, tmp_path) -> Path:
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    return save_checkpoint(
        tmp_path / "best.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(global_step=0),
        seed=0,
    )


def _response() -> dict:
    return {
        "text": "你好",
        "spokenText": "你好",
        "visemes": [
            {"ipa": "n", "word": "你", "wordIndex": 0, "charIndex": 0,
             "shapeV2": "316_THin", "startPercent": 0.1, "endPercent": 0.4},
            {"ipa": "i", "word": "你", "wordIndex": 0, "charIndex": 0,
             "shapeV2": "303_Idea", "startPercent": 0.4, "endPercent": 0.7},
            {"ipa": "x", "word": "好", "wordIndex": 1, "charIndex": 1,
             "shapeV2": "318_KAKA", "startPercent": 0.7, "endPercent": 0.9},
            {"ipa": "a", "word": "好", "wordIndex": 1, "charIndex": 1,
             "shapeV2": "302_High", "startPercent": 0.9, "endPercent": 1.0},
        ],
    }


def test_predict_api_response_structure_and_ranges(saved_checkpoint):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    out = predict_api_response(_response(), predictor)

    assert out["text"] == "你好"
    assert len(out["visemes"]) == 4
    for v in out["visemes"]:
        assert v["shapeV2"] in VISEME_NAMES
        assert 0.0 <= v["strength"] <= 100.0
        assert v["startPercent"] is not None
        assert v["wordIndex"] is not None


def test_predict_api_response_preserves_timing_and_indices(saved_checkpoint):
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    out = predict_api_response(_response(), predictor)
    src = _response()["visemes"]
    for pred, ref in zip(out["visemes"], src, strict=True):
        assert pred["startPercent"] == ref["startPercent"]
        assert pred["endPercent"] == ref["endPercent"]
        assert pred["wordIndex"] == ref["wordIndex"]
        assert pred["ipa"] == ref["ipa"]


def test_predict_api_response_accepts_minimal_request(saved_checkpoint):
    """Only word/wordIndex/charIndex/ipa are required; timing is omitted."""
    predictor = ModelPredictor.load(saved_checkpoint, device="cpu")
    request = {
        "text": "你好",
        "visemes": [
            {"word": "你", "wordIndex": 0, "charIndex": 0, "ipa": "n"},
            {"word": "你", "wordIndex": 0, "charIndex": 0, "ipa": "i"},
            {"word": "好", "wordIndex": 1, "charIndex": 0, "ipa": "x"},
            {"word": "好", "wordIndex": 1, "charIndex": 0, "ipa": "a"},
        ],
    }
    out = predict_api_response(request, predictor)
    assert len(out["visemes"]) == 4
    for v in out["visemes"]:
        assert v["shapeV2"] in VISEME_NAMES
        assert "strength" in v and "word" in v
        assert "startPercent" not in v  # caller didn't send timing
        assert "endPercent" not in v
