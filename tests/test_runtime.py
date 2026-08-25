"""Precision policy, seeding and structured logging.

The V100 rule is non-negotiable: V100-class GPUs use fp16, never bf16.
These tests exercise the decision function against synthetic HardwareInfo so
they run on any host.
"""

from __future__ import annotations

import json

import pytest
import torch

from articulm.runtime import (
    FP16_ONLY_COMPUTE_CAPABILITIES,
    HardwareInfo,
    PrecisionError,
    StructuredLogger,
    describe_hardware,
    make_run_id,
    resolve_device,
    resolve_precision,
    set_seed,
)


def _gpu(name: str, capability: tuple[int, int], bf16: bool) -> HardwareInfo:
    return HardwareInfo(
        device=torch.device("cuda"),
        device_kind="cuda",
        device_name=name,
        compute_capability=capability,
        total_memory_bytes=32 * 1024**3,
        torch_version=torch.__version__,
        cuda_version="12.1",
        bf16_supported=bf16,
    )


V100 = _gpu("Tesla V100-SXM2-32GB", (7, 0), bf16=False)
A100 = _gpu("NVIDIA A100-SXM4-40GB", (8, 0), bf16=True)
H100 = _gpu("NVIDIA H100 80GB HBM3", (9, 0), bf16=True)
CPU = HardwareInfo(
    device=torch.device("cpu"),
    device_kind="cpu",
    device_name="cpu",
    compute_capability=None,
    total_memory_bytes=None,
    torch_version=torch.__version__,
    cuda_version=None,
    bf16_supported=False,
)


def test_v100_capability_is_in_the_fp16_only_set():
    assert (7, 0) in FP16_ONLY_COMPUTE_CAPABILITIES


def test_auto_on_v100_selects_fp16_not_bf16():
    plan = resolve_precision("auto", V100)
    assert plan.name == "fp16"
    assert plan.autocast_dtype == torch.float16
    assert plan.use_grad_scaler is True


def test_explicit_bf16_on_v100_is_rejected():
    with pytest.raises(PrecisionError, match="V100-class GPUs must use fp16"):
        resolve_precision("bf16", V100)


@pytest.mark.parametrize("hardware", [A100, H100], ids=["a100", "h100"])
def test_auto_on_a100_h100_prefers_bf16(hardware):
    plan = resolve_precision("auto", hardware)
    assert plan.name == "bf16"
    assert plan.autocast_dtype == torch.bfloat16
    # bf16 has fp32 dynamic range, so no loss scaling is needed.
    assert plan.use_grad_scaler is False


@pytest.mark.parametrize("hardware", [A100, H100], ids=["a100", "h100"])
def test_explicit_bf16_is_allowed_on_modern_gpus(hardware):
    assert resolve_precision("bf16", hardware).name == "bf16"


def test_explicit_fp16_works_everywhere_on_cuda():
    for hardware in (V100, A100, H100):
        plan = resolve_precision("fp16", hardware)
        assert plan.name == "fp16"
        assert plan.use_grad_scaler is True


def test_fp32_is_honoured_on_gpu():
    plan = resolve_precision("fp32", A100)
    assert plan.name == "fp32"
    assert plan.autocast_enabled is False
    assert plan.use_grad_scaler is False


def test_cpu_falls_back_to_fp32_with_a_reason():
    plan = resolve_precision("auto", CPU)
    assert plan.name == "fp32"
    assert plan.autocast_enabled is False
    assert "cpu" in plan.reason


def test_mixed_precision_request_on_cpu_degrades_not_crashes():
    plan = resolve_precision("fp16", CPU)
    assert plan.name == "fp32"
    assert "falling back to fp32" in plan.reason


def test_unknown_precision_is_rejected():
    with pytest.raises(PrecisionError, match="unknown precision"):
        resolve_precision("int4", A100)


def test_precision_plan_serialises():
    payload = resolve_precision("auto", A100).as_dict()
    assert payload["precision"] == "bf16"
    assert payload["grad_scaler"] is False
    assert payload["reason"]


# -- hardware description --------------------------------------------------


def test_describe_hardware_runs_on_this_host():
    info = describe_hardware()
    assert info.device_kind in {"cpu", "cuda", "mps"}
    assert info.torch_version == torch.__version__
    payload = info.as_dict()
    assert set(payload) >= {"device", "device_kind", "bf16_supported"}


def test_resolve_device_honours_explicit_request():
    assert resolve_device("cpu") == torch.device("cpu")


def test_this_host_precision_plan_is_consistent():
    hardware = describe_hardware()
    plan = resolve_precision("auto", hardware)
    if hardware.device_kind != "cuda":
        assert plan.name == "fp32"
    else:
        assert plan.name in {"fp16", "bf16"}


# -- seeding ---------------------------------------------------------------


def test_set_seed_makes_torch_deterministic():
    set_seed(99)
    first = torch.randn(8)
    set_seed(99)
    assert torch.equal(first, torch.randn(8))


def test_set_seed_covers_python_and_numpy():
    import random

    import numpy as np

    set_seed(7)
    values = (random.random(), float(np.random.rand()))
    set_seed(7)
    assert values == (random.random(), float(np.random.rand()))


# -- structured logging ----------------------------------------------------


def test_structured_logger_writes_jsonl(tmp_path):
    logger = StructuredLogger("run_x", log_dir=tmp_path)
    logger.event("train_step", "step done", step=3, loss=0.25, tokens_per_s=1234.5)
    assert logger.jsonl_path is not None
    lines = logger.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "run_x"
    assert record["stage"] == "train_step"
    assert record["step"] == 3
    assert record["loss"] == pytest.approx(0.25)
    assert "elapsed_s" in record


def test_reserved_log_keys_cannot_be_shadowed(tmp_path):
    """A caller field named `stage` must not overwrite the record's stage."""
    logger = StructuredLogger("run_r", log_dir=tmp_path)
    logger.event("train_step", stage="synthetic_pretraining", run_id="hacked")
    assert logger.jsonl_path is not None
    record = json.loads(logger.jsonl_path.read_text(encoding="utf-8").strip())
    assert record["stage"] == "train_step"
    assert record["run_id"] == "run_r"
    assert record["field_stage"] == "synthetic_pretraining"
    assert record["field_run_id"] == "hacked"


def test_structured_logger_appends(tmp_path):
    logger = StructuredLogger("run_y", log_dir=tmp_path)
    for step in range(3):
        logger.event("train_step", step=step)
    assert logger.jsonl_path is not None
    assert len(logger.jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_structured_logger_without_dir_does_not_write(tmp_path):
    logger = StructuredLogger("run_z")
    logger.event("train_step", step=1)
    assert logger.jsonl_path is None
    assert not list(tmp_path.iterdir())


def test_run_id_includes_experiment_name():
    run_id = make_run_id("articulm_v1_50m_baseline", timestamp=0)
    assert run_id.startswith("articulm_v1_50m_baseline_")
    assert len(run_id) > len("articulm_v1_50m_baseline_")


def test_run_ids_within_the_same_second_are_distinct():
    """Two runs launched together must not share a run directory."""
    first = make_run_id("exp", timestamp=1000.100)
    second = make_run_id("exp", timestamp=1000.700)
    assert first != second


def test_logger_reinstalls_its_handler_for_a_reused_name(tmp_path, capsys):
    """A second logger of the same name must write to the current stream."""
    StructuredLogger("same_name", log_dir=tmp_path / "a")
    capsys.readouterr()
    StructuredLogger("same_name", log_dir=tmp_path / "b").event("resume", "resumed")
    assert "resumed" in capsys.readouterr().out
