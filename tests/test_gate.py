"""Tiny-overfit acceptance gate (docs/09 M2).

The gate is what stands between a broken implementation and a long GPU run, so
its failure modes matter as much as its success path.
"""

from __future__ import annotations

import json

import pytest

from articulm import gate as gate_module
from articulm.gate import (
    GateCriteria,
    check_checkpoints,
    evaluate_summary,
    find_summary,
)


def _passing_summary(**overrides) -> dict:
    summary = {
        "run_id": "unit_test",
        "global_step": 2000,
        "epochs_completed": 250,
        "final_train_loss": 6.7e-05,
        "final_train_viseme_accuracy": 1.0,
        "final_train_strength_mae": 0.27,
        "best_metric": 1.0,
        "best_step": 1000,
        "validation": {
            "val_viseme_accuracy": 1.0,
            "val_viseme_macro_f1": 1.0,
            "val_strength_mae": 0.21,
            "val_strength_rmse": 0.26,
        },
        "checkpoint_dir": "runs/x/checkpoints",
        "stopped_because": "max_steps_reached",
        "saw_non_finite_loss": False,
    }
    summary.update(overrides)
    return summary


# -- happy path -----------------------------------------------------------


def test_a_fitted_run_passes():
    result = evaluate_summary(_passing_summary())
    assert result.passed
    assert result.failures == []
    assert "GATE PASSED" in result.render()


def test_result_serialises():
    payload = evaluate_summary(_passing_summary()).as_dict()
    json.dumps(payload)
    assert payload["passed"] is True
    assert all(check["passed"] for check in payload["checks"])


# -- each failure mode is caught individually ----------------------------


def test_non_finite_loss_fails_the_gate():
    result = evaluate_summary(_passing_summary(saw_non_finite_loss=True))
    assert not result.passed
    assert any("NaN/Inf" in check.name for check in result.failures)


def test_nan_final_loss_fails_the_gate():
    result = evaluate_summary(_passing_summary(final_train_loss=float("nan")))
    assert not result.passed
    assert any("finite" in check.name for check in result.failures)


def test_underfit_viseme_accuracy_fails_the_gate():
    result = evaluate_summary(_passing_summary(final_train_viseme_accuracy=0.42))
    assert not result.passed
    failure = next(c for c in result.failures if "viseme train accuracy" in c.name)
    assert "0.42" in failure.detail


def test_large_strength_error_fails_the_gate():
    result = evaluate_summary(_passing_summary(final_train_strength_mae=13.3))
    assert not result.passed
    assert any("strength error" in check.name for check in result.failures)


def test_zero_steps_fails_the_gate():
    result = evaluate_summary(_passing_summary(global_step=0))
    assert not result.passed
    assert any("actually ran" in check.name for check in result.failures)


def test_missing_validation_fails_by_default():
    """A run that never validated cannot demonstrate the fit."""
    result = evaluate_summary(_passing_summary(validation=None))
    assert not result.passed
    assert any("validation metrics present" in c.name for c in result.failures)


def test_missing_validation_can_be_waived():
    criteria = GateCriteria(require_validation=False)
    assert evaluate_summary(_passing_summary(validation=None), criteria).passed


def test_bad_validation_accuracy_fails_the_gate():
    summary = _passing_summary()
    summary["validation"]["val_viseme_accuracy"] = 0.3
    result = evaluate_summary(summary)
    assert not result.passed
    assert any("validation viseme accuracy" in c.name for c in result.failures)


def test_absent_metric_keys_fail_rather_than_pass_silently():
    summary = _passing_summary()
    del summary["final_train_viseme_accuracy"]
    result = evaluate_summary(summary)
    assert not result.passed


def test_string_metric_does_not_crash_the_gate():
    result = evaluate_summary(_passing_summary(final_train_strength_mae="n/a"))
    assert not result.passed


# -- thresholds are configurable -----------------------------------------


def test_stricter_threshold_can_reject_a_default_pass():
    summary = _passing_summary(final_train_viseme_accuracy=0.96)
    assert evaluate_summary(summary, GateCriteria()).passed
    strict = GateCriteria(min_viseme_accuracy=0.99)
    assert not evaluate_summary(summary, strict).passed


def test_looser_mae_threshold_can_accept_a_default_failure():
    summary = _passing_summary(final_train_strength_mae=8.0)
    assert not evaluate_summary(summary, GateCriteria()).passed
    loose = GateCriteria(max_strength_mae=10.0)
    assert evaluate_summary(summary, loose).passed


# -- checkpoint checks ----------------------------------------------------


def test_missing_checkpoints_are_reported(tmp_path):
    from articulm.gate import GateResult

    result = GateResult()
    check_checkpoints(tmp_path, result)
    assert not result.passed
    assert len(result.failures) == 2


def test_corrupt_checkpoint_is_reported(tmp_path):
    from articulm.gate import GateResult

    (tmp_path / "last.pt").write_bytes(b"not a checkpoint")
    (tmp_path / "best.pt").write_bytes(b"not a checkpoint")
    result = GateResult()
    check_checkpoints(tmp_path, result)
    assert not result.passed


def test_real_checkpoints_load(tmp_path, tiny_model_config, data_config, vocab):
    from articulm.gate import GateResult
    from articulm.model.articulm_v1 import ArticuLMV1
    from articulm.training.checkpoint import TrainingState, save_checkpoint

    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    for name in ("last.pt", "best.pt"):
        save_checkpoint(
            tmp_path / name,
            model=model,
            model_config=tiny_model_config,
            data_config=data_config,
            vocab=vocab,
            state=TrainingState(global_step=7, epoch=1),
            seed=42,
        )
    result = GateResult()
    check_checkpoints(tmp_path, result)
    assert result.passed
    assert all("step=7" in check.detail for check in result.checks)


# -- CLI ------------------------------------------------------------------


def _write_run(tmp_path, summary: dict):
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return tmp_path


def test_find_summary_locates_the_file(tmp_path):
    _write_run(tmp_path, _passing_summary())
    assert find_summary(tmp_path).is_file()


def test_find_summary_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="did the run finish"):
        find_summary(tmp_path)


def test_cli_exit_zero_on_pass(tmp_path, capsys):
    _write_run(tmp_path, _passing_summary())
    code = gate_module.main(
        ["--run-dir", str(tmp_path), "--skip-checkpoint-checks"]
    )
    assert code == 0
    assert "GATE PASSED" in capsys.readouterr().out


def test_cli_exit_one_on_fail(tmp_path, capsys):
    _write_run(tmp_path, _passing_summary(final_train_viseme_accuracy=0.1))
    code = gate_module.main(
        ["--run-dir", str(tmp_path), "--skip-checkpoint-checks"]
    )
    assert code == 1
    output = capsys.readouterr().out
    assert "GATE FAILED" in output
    assert "Do not start a long training run" in output


def test_cli_exit_two_when_the_run_never_finished(tmp_path, capsys):
    code = gate_module.main(["--run-dir", str(tmp_path)])
    assert code == 2
    assert "GATE FAILED" in capsys.readouterr().out


def test_cli_writes_a_json_verdict(tmp_path, capsys):
    _write_run(tmp_path, _passing_summary())
    out = tmp_path / "verdict.json"
    gate_module.main(
        [
            "--run-dir",
            str(tmp_path),
            "--skip-checkpoint-checks",
            "--json-out",
            str(out),
        ]
    )
    capsys.readouterr()
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["passed"] is True
    assert verdict["checks"]


def test_cli_accepts_a_summary_path_directly(tmp_path, capsys):
    _write_run(tmp_path, _passing_summary())
    code = gate_module.main(
        [
            "--summary",
            str(tmp_path / "metrics" / "training_summary.json"),
            "--skip-checkpoint-checks",
        ]
    )
    capsys.readouterr()
    assert code == 0


def test_cli_threshold_overrides_reach_the_criteria(tmp_path, capsys):
    _write_run(tmp_path, _passing_summary(final_train_viseme_accuracy=0.96))
    assert (
        gate_module.main(
            [
                "--run-dir",
                str(tmp_path),
                "--skip-checkpoint-checks",
                "--min-accuracy",
                "0.99",
            ]
        )
        == 1
    )
    capsys.readouterr()
