"""End-to-end CLI smoke tests: train -> evaluate -> infer -> export.

These run a real (very short) training job on the fixtures so the documented
commands are exercised, not just the library internals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import articulm.evaluate as evaluate_cli
import articulm.export as export_cli
import articulm.infer as infer_cli
import articulm.train as train_cli
from articulm.data import validate as validate_cli


@pytest.fixture
def tiny_data_config(tmp_path, fixture_dir) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "schema_version": "articulm_v1_sample_v1",
                    "train_path": str(fixture_dir / "sample_zh.jsonl"),
                    "validation_path": str(fixture_dir / "sample_mixed.jsonl"),
                    "test_path": str(fixture_dir / "sample_en.jsonl"),
                    "max_seq_len": 128,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def small_model_config(tmp_path, model_config) -> Path:
    """A genuinely small model so the CLI test stays fast."""
    from articulm.config import to_plain_dict

    raw = to_plain_dict(model_config)
    raw["input"]["max_seq_len"] = 128
    raw["input"]["embedding_dims"] = {
        "phoneme": 16,
        "language": 2,
        "surface_tone": 4,
        "stress": 2,
        "syllable_role": 4,
        "articulatory": {
            "type": 2,
            "height": 2,
            "backness": 2,
            "rounded": 1,
            "place": 3,
            "manner": 3,
            "voiced": 1,
            "aspirated": 1,
        },
        "boundary": 5,
    }
    raw["input"]["fused_input_dim"] = 48
    raw["fusion"]["output_dim"] = 64
    raw["transformer"].update(
        {"hidden_size": 64, "num_layers": 2, "num_heads": 4, "head_dim": 16, "ffn_size": 128}
    )
    raw["viseme_head"]["hidden_size"] = 32
    raw["strength_head"]["input_dim"] = 96
    raw["strength_head"]["hidden_dims"] = [32, 16]

    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"model": raw}), encoding="utf-8")
    return path


@pytest.fixture
def train_config(tmp_path, small_model_config, tiny_data_config) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "cli_smoke",
                    "seed": 42,
                    "output_dir": str(tmp_path / "runs"),
                },
                "model_config": str(small_model_config),
                "data_config": str(tiny_data_config),
                "training": {
                    "stage": "smoke_test",
                    "precision": "fp32",
                    "max_steps": 4,
                    "optimizer": {"type": "adamw", "learning_rate": 1.0e-3},
                    "scheduler": {"type": "cosine", "warmup_ratio": 0.25},
                    "batching": {
                        "strategy": "fixed_samples",
                        "batch_size": 2,
                        "gradient_accumulation_steps": 1,
                    },
                    "gradient_clip_norm": 1.0,
                    "loss": {
                        "viseme": {"weight": 1.0, "label_smoothing": 0.05},
                        "strength": {"weight": 0.3, "source_weights": {"pseudo_strength_v1": 1.0}},
                    },
                    "evaluation": {"every_steps": 4},
                    "checkpoint": {"every_steps": 4, "keep_last_n": 1},
                    "logging": {"every_steps": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_validate_data_cli(tiny_data_config, tmp_path, capsys):
    exit_code = validate_cli.main(
        ["--config", str(tiny_data_config), "--json-out", str(tmp_path / "report.json")]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Phoneme Tokens:" in output
    assert "Viseme Classes:" in output
    assert "Human Gold strength:" in output
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert set(report) == {"train", "validation", "test"}
    assert report["train"]["num_sentences"] == 3


def test_train_dry_run_reports_without_training(train_config, tmp_path, capsys):
    run_dir = tmp_path / "dry"
    exit_code = train_cli.main(
        ["--config", str(train_config), "--dry-run", "--run-dir", str(run_dir)]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    for expected in (
        "pre-flight report",
        "Total parameters:",
        "Precision:",
        "Est. activations:",
        "Checkpoint dir:",
        "Local Conv:",
    ):
        assert expected in output, expected
    assert not (run_dir / "checkpoints").exists()
    # Run inputs are still persisted for provenance.
    assert (run_dir / "model_config.yaml").is_file()
    assert (run_dir / "vocab" / "feature_vocab.json").is_file()


@pytest.fixture
def trained_run(train_config, tmp_path, capsys) -> Path:
    run_dir = tmp_path / "run"
    exit_code = train_cli.main(
        ["--config", str(train_config), "--run-dir", str(run_dir), "--device", "cpu"]
    )
    capsys.readouterr()
    assert exit_code == 0
    return run_dir


def test_train_cli_produces_expected_artifacts(trained_run):
    assert (trained_run / "checkpoints" / "last.pt").is_file()
    assert (trained_run / "checkpoints" / "best.pt").is_file()
    assert (trained_run / "model_config.yaml").is_file()
    assert (trained_run / "data_config.yaml").is_file()
    assert (trained_run / "train_config.yaml").is_file()
    assert (trained_run / "vocab" / "feature_vocab.json").is_file()
    assert (trained_run / "logs" / "events.jsonl").is_file()

    summary = json.loads(
        (trained_run / "metrics" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["global_step"] == 4
    assert summary["saw_non_finite_loss"] is False
    assert summary["validation"] is not None


def test_training_event_log_is_structured(trained_run):
    lines = (trained_run / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    stages = {record["stage"] for record in records}
    assert {"config", "data_loaded", "vocab", "trainer_init", "train_step"} <= stages
    for record in records:
        assert "run_id" in record and "elapsed_s" in record
    step_events = [r for r in records if r["stage"] == "train_step"]
    assert step_events
    for event in step_events:
        assert {"loss", "viseme_loss", "strength_loss", "lr", "tokens_per_s"} <= set(event)


def test_resume_from_checkpoint(train_config, trained_run, tmp_path, capsys):
    exit_code = train_cli.main(
        [
            "--config",
            str(train_config),
            "--run-dir",
            str(tmp_path / "resumed"),
            "--device",
            "cpu",
            "--resume",
            str(trained_run / "checkpoints" / "last.pt"),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "resumed from checkpoint" in output


def test_init_from_warm_starts_weights_with_fresh_state(
    train_config, trained_run, tmp_path, capsys
):
    run_dir = tmp_path / "finetune"
    exit_code = train_cli.main(
        [
            "--config",
            str(train_config),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--init-from",
            str(trained_run / "checkpoints" / "last.pt"),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "warm start from checkpoint" in output

    records = [
        json.loads(line)
        for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    init_events = [r for r in records if r["stage"] == "init_from"]
    assert init_events, "expected an init_from event"
    assert init_events[0]["source_global_step"] == 4
    # Fresh training state: the fine-tune run logs its own steps from 1.
    step_events = [r for r in records if r["stage"] == "train_step"]
    assert [e["step"] for e in step_events] == [2, 4]

    summary = json.loads(
        (run_dir / "metrics" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["global_step"] == 4


def test_resume_and_init_from_are_mutually_exclusive(trained_run, train_config):
    with pytest.raises(SystemExit):
        train_cli.main(
            [
                "--config",
                str(train_config),
                "--resume",
                str(trained_run / "checkpoints" / "last.pt"),
                "--init-from",
                str(trained_run / "checkpoints" / "last.pt"),
            ]
        )


def test_init_from_rejects_vocabulary_mismatch(
    train_config, trained_run, tmp_path, capsys
):
    vocab_path = tmp_path / "mutated_vocab.json"
    vocab = json.loads(
        (trained_run / "vocab" / "feature_vocab.json").read_text(encoding="utf-8")
    )
    vocab["fields"]["phoneme"] = vocab["fields"]["phoneme"][:-1]
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    with pytest.raises(Exception, match="misaligned"):
        train_cli.main(
            [
                "--config",
                str(train_config),
                "--run-dir",
                str(tmp_path / "finetune"),
                "--device",
                "cpu",
                "--vocab",
                str(vocab_path),
                "--init-from",
                str(trained_run / "checkpoints" / "last.pt"),
            ]
        )
    capsys.readouterr()


def test_evaluate_cli_writes_all_artifacts(trained_run, tmp_path, fixture_dir, capsys):
    out_dir = tmp_path / "eval"
    exit_code = evaluate_cli.main(
        [
            "--checkpoint",
            str(trained_run / "checkpoints" / "best.pt"),
            "--data",
            str(fixture_dir / "sample_zh.jsonl"),
            "--label-set",
            "synthetic",
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "label_set=synthetic" in output
    assert "macro F1" in output

    for name in (
        "metrics.json",
        "per_class.csv",
        "confusion_matrix.csv",
        "strength_report.csv",
        "failure_cases.jsonl",
    ):
        assert (out_dir / name).is_file(), name

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["label_set"] == "synthetic"
    assert "must be reported separately" in metrics["note"]
    assert len(metrics["viseme"]["confusion_matrix"]) == 18
    assert metrics["num_tokens"] == 65


def test_evaluate_keeps_label_sets_separate(trained_run, tmp_path, fixture_dir, capsys):
    """Two label sets must produce two independent reports."""
    checkpoint = str(trained_run / "checkpoints" / "best.pt")
    for label_set, name in (("synthetic", "sample_zh.jsonl"), ("human_gold", "sample_en.jsonl")):
        out_dir = tmp_path / f"eval_{label_set}"
        evaluate_cli.main(
            [
                "--checkpoint",
                checkpoint,
                "--data",
                str(fixture_dir / name),
                "--label-set",
                label_set,
                "--out-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
        )
        capsys.readouterr()
        payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert payload["label_set"] == label_set
    a = json.loads((tmp_path / "eval_synthetic" / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "eval_human_gold" / "metrics.json").read_text(encoding="utf-8"))
    assert a["num_tokens"] != b["num_tokens"]


def test_infer_cli_output_schema(trained_run, tmp_path, fixture_dir, capsys):
    # Inference input carries no labels at all.
    source = fixture_dir / "sample_zh.jsonl"
    stripped = tmp_path / "infer_in.jsonl"
    with stripped.open("w", encoding="utf-8") as out:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for token in record["tokens"]:
                token.pop("labels", None)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_path = tmp_path / "predictions.jsonl"
    exit_code = infer_cli.main(
        [
            "--checkpoint",
            str(trained_run / "checkpoints" / "best.pt"),
            "--input",
            str(stripped),
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ]
    )
    capsys.readouterr()
    assert exit_code == 0

    records = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3
    for record in records:
        assert set(record) == {"sample_id", "text", "outputs"}
        for token in record["outputs"]:
            # Exactly the documented per-phoneme output schema.
            assert set(token) == {"phoneme", "viseme_id", "strength"}
            assert isinstance(token["viseme_id"], int)
            assert 0 <= token["viseme_id"] <= 17
            assert 0.0 <= token["strength"] <= 100.0


def test_inference_token_count_matches_input(trained_run, tmp_path, fixture_dir, capsys):
    from articulm.config import load_data_config
    from articulm.data.schema import load_samples

    checkpoint = str(trained_run / "checkpoints" / "best.pt")
    out_path = tmp_path / "pred.jsonl"
    infer_cli.main(
        [
            "--checkpoint",
            checkpoint,
            "--input",
            str(fixture_dir / "sample_mixed.jsonl"),
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ]
    )
    capsys.readouterr()
    data_cfg = load_data_config(
        Path(__file__).resolve().parent.parent / "config" / "data_v1.yaml"
    )
    samples = load_samples(fixture_dir / "sample_mixed.jsonl", data_cfg)
    records = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for sample, record in zip(samples, records, strict=True):
        assert len(record["outputs"]) == len(sample.tokens)
        assert [t["phoneme"] for t in record["outputs"]] == [
            t.phoneme for t in sample.tokens
        ]


def test_export_torchscript_with_metadata(trained_run, tmp_path, capsys):
    out = tmp_path / "model.ts"
    exit_code = export_cli.main(
        [
            "--checkpoint",
            str(trained_run / "checkpoints" / "best.pt"),
            "--out",
            str(out),
            "--format",
            "torchscript",
        ]
    )
    capsys.readouterr()
    assert exit_code == 0
    assert out.is_file()

    metadata_path = out.with_suffix(out.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "config_hash",
        "phoneme_vocab_size",
        "feature_vocab_version",
        "feature_vocab_hash",
        "data_schema_version",
        "checkpoint_step",
        "seed",
    ):
        assert key in metadata, key


def test_exported_torchscript_matches_pytorch(trained_run, tmp_path, capsys):
    import torch

    from articulm.export import ArticuLMExportWrapper, example_inputs
    from articulm.model.articulm_v1 import ArticuLMV1
    from articulm.training.checkpoint import load_checkpoint

    out = tmp_path / "model.ts"
    export_cli.main(
        [
            "--checkpoint",
            str(trained_run / "checkpoints" / "best.pt"),
            "--out",
            str(out),
            "--format",
            "torchscript",
        ]
    )
    capsys.readouterr()

    loaded = load_checkpoint(str(trained_run / "checkpoints" / "best.pt"))
    model = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
    model.load_state_dict(loaded.payload["model_state_dict"], strict=True)
    wrapper = ArticuLMExportWrapper(model)
    wrapper.eval()

    inputs = example_inputs(loaded.vocab, batch_size=2, seq_len=16)
    traced = torch.jit.load(str(out))
    with torch.no_grad():
        reference = wrapper(*inputs)
        exported = traced(*inputs)
    assert torch.allclose(reference[0], exported[0], atol=1e-5)
    assert torch.allclose(reference[1], exported[1], atol=1e-4)


def test_dispatcher_lists_commands(capsys):
    from articulm.__main__ import main

    assert main([]) == 2
    output = capsys.readouterr().out
    for command in ("train", "evaluate", "infer", "export", "validate-data"):
        assert command in output


def test_dispatcher_rejects_unknown_command(capsys):
    from articulm.__main__ import main

    assert main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().out


def test_val_composite_metric_in_events_and_best_selection(
    train_config, trained_run, tmp_path, capsys
):
    """val_composite is logged on every validation and can drive best selection."""
    import yaml

    cfg = yaml.safe_load(Path(train_config).read_text(encoding="utf-8"))
    cfg["training"]["checkpoint"]["save_best_by"] = "val_composite"
    cfg["training"]["checkpoint"]["best_composite_alpha"] = 1.0
    composite_config = tmp_path / "train_composite.yaml"
    composite_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    run_dir = tmp_path / "composite"
    exit_code = train_cli.main(
        ["--config", str(composite_config), "--run-dir", str(run_dir), "--device", "cpu"]
    )
    assert exit_code == 0
    capsys.readouterr()

    records = [
        json.loads(line)
        for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validations = [r for r in records if r["stage"] == "validation"]
    assert validations
    for event in validations:
        assert "val_composite" in event
        expected = (
            event["val_viseme_macro_f1"]
            - event["val_strength_mae"] / 100.0
        )
        assert abs(event["val_composite"] - expected) < 1e-6

    best_events = [r for r in records if r["stage"] == "checkpoint_best"]
    assert best_events and all(e["metric"] == "val_composite" for e in best_events)

    summary = json.loads(
        (run_dir / "metrics" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["best_metric"] is not None
