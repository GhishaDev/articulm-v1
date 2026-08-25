"""Device, precision, seeding and structured logging.

Precision policy (docs/11_training_operations.md — non-negotiable):

```text
V100 (sm_70)        -> fp16.  BF16 is never selected automatically, and an
                       explicit bf16 request on V100 is rejected.
A100/H100 (sm_80+)  -> bf16 when the runtime reports support, else fp16.
CPU / MPS           -> fp32.
```
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# GPUs whose tensor cores predate bf16. Selecting bf16 here is a silent
# correctness/performance trap, so it is refused outright.
FP16_ONLY_COMPUTE_CAPABILITIES = frozenset({(7, 0), (7, 2), (7, 5)})


class PrecisionError(ValueError):
    pass


@dataclass(frozen=True)
class HardwareInfo:
    device: torch.device
    device_kind: str
    device_name: str
    compute_capability: tuple[int, int] | None
    total_memory_bytes: int | None
    torch_version: str
    cuda_version: str | None
    bf16_supported: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "device_kind": self.device_kind,
            "device_name": self.device_name,
            "compute_capability": (
                f"{self.compute_capability[0]}.{self.compute_capability[1]}"
                if self.compute_capability
                else None
            ),
            "total_memory_gb": (
                round(self.total_memory_bytes / (1024**3), 2)
                if self.total_memory_bytes
                else None
            ),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "bf16_supported": self.bf16_supported,
        }


@dataclass(frozen=True)
class PrecisionPlan:
    name: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": self.name,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": str(self.autocast_dtype) if self.autocast_dtype else None,
            "grad_scaler": self.use_grad_scaler,
            "reason": self.reason,
        }


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a device: explicit request, else CUDA > MPS > CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_hardware(device: torch.device | None = None) -> HardwareInfo:
    device = device or resolve_device()
    kind = device.type

    name = "cpu"
    capability: tuple[int, int] | None = None
    total_memory: int | None = None
    bf16 = False

    if kind == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        name = properties.name
        capability = (properties.major, properties.minor)
        total_memory = properties.total_memory
        bf16 = bool(torch.cuda.is_bf16_supported()) and capability not in FP16_ONLY_COMPUTE_CAPABILITIES
    elif kind == "mps":
        name = "Apple MPS"

    return HardwareInfo(
        device=device,
        device_kind=kind,
        device_name=name,
        compute_capability=capability,
        total_memory_bytes=total_memory,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        bf16_supported=bf16,
    )


def resolve_precision(requested: str, hardware: HardwareInfo) -> PrecisionPlan:
    """Turn a config precision setting into a concrete autocast plan."""
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise PrecisionError(f"unknown precision {requested!r}")

    fp32 = PrecisionPlan(
        name="fp32",
        autocast_enabled=False,
        autocast_dtype=None,
        use_grad_scaler=False,
        reason="",
    )

    if hardware.device_kind != "cuda":
        if requested in {"fp16", "bf16"}:
            return PrecisionPlan(
                name="fp32",
                autocast_enabled=False,
                autocast_dtype=None,
                use_grad_scaler=False,
                reason=(
                    f"requested {requested} but device is {hardware.device_kind}; "
                    "mixed precision is CUDA-only in this trainer, falling back to fp32"
                ),
            )
        return PrecisionPlan(
            **{**fp32.__dict__, "reason": f"{hardware.device_kind} device runs fp32"}
        )

    if requested == "fp32":
        return PrecisionPlan(**{**fp32.__dict__, "reason": "fp32 requested explicitly"})

    if requested == "bf16":
        if not hardware.bf16_supported:
            raise PrecisionError(
                f"bf16 requested but {hardware.device_name} "
                f"(sm_{hardware.compute_capability}) does not support it. "
                "V100-class GPUs must use fp16."
            )
        return PrecisionPlan(
            name="bf16",
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            use_grad_scaler=False,
            reason=f"bf16 requested and supported on {hardware.device_name}",
        )

    if requested == "fp16":
        return PrecisionPlan(
            name="fp16",
            autocast_enabled=True,
            autocast_dtype=torch.float16,
            use_grad_scaler=True,
            reason=f"fp16 requested on {hardware.device_name}",
        )

    # requested == "auto"
    if hardware.bf16_supported:
        return PrecisionPlan(
            name="bf16",
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            use_grad_scaler=False,
            reason=f"auto: {hardware.device_name} supports bf16",
        )
    return PrecisionPlan(
        name="fp16",
        autocast_enabled=True,
        autocast_dtype=torch.float16,
        use_grad_scaler=True,
        reason=(
            f"auto: {hardware.device_name} "
            f"(sm_{hardware.compute_capability}) is fp16-only"
        ),
    )


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------


_RESERVED_EVENT_KEYS = frozenset({"run_id", "stage", "elapsed_s", "message"})


class StructuredLogger:
    """Emit human-readable console lines plus a JSONL event log.

    Every event carries ``run_id``, ``stage`` and a monotonic ``elapsed_s`` so
    training curves and per-stage latency can be recovered from the file.
    """

    def __init__(
        self,
        run_id: str,
        *,
        log_dir: str | Path | None = None,
        stream: Any = None,
        level: int = logging.INFO,
    ) -> None:
        self.run_id = run_id
        self.started_at = time.monotonic()
        self._logger = logging.getLogger(f"articulm.{run_id}")
        self._logger.setLevel(level)
        self._logger.propagate = False
        # Always install a fresh handler. Reusing one from an earlier logger of
        # the same name would write this run's events to that run's stream.
        for existing in list(self._logger.handlers):
            self._logger.removeHandler(existing)
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._logger.addHandler(handler)

        self._jsonl_path: Path | None = None
        if log_dir is not None:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = directory / "events.jsonl"

    @property
    def jsonl_path(self) -> Path | None:
        return self._jsonl_path

    def event(self, stage: str, message: str = "", /, **fields: Any) -> None:
        # Reserved keys own their names; a caller field of the same name is
        # prefixed rather than allowed to shadow the record's own stage.
        safe_fields = {
            (f"field_{key}" if key in _RESERVED_EVENT_KEYS else key): value
            for key, value in fields.items()
        }
        record = {
            "run_id": self.run_id,
            "stage": stage,
            "elapsed_s": round(time.monotonic() - self.started_at, 3),
            **safe_fields,
        }
        if message:
            record["message"] = message

        rendered = " ".join(
            f"{key}={_render(value)}"
            for key, value in record.items()
            if key not in {"run_id", "message"}
        )
        self._logger.info("[%s] %s %s", stage, message, rendered)

        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def make_run_id(experiment_name: str, timestamp: float | None = None) -> str:
    """Timestamped run id.

    Milliseconds are included so two runs launched in the same second do not
    collide on the same run directory or log file.
    """
    moment = timestamp if timestamp is not None else time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(moment))
    milliseconds = int((moment % 1) * 1000)
    return f"{experiment_name}_{stamp}_{milliseconds:03d}"
