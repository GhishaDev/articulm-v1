"""Tiny-overfit acceptance gate.

Encodes the M2 criteria from ``docs/09_acceptance_criteria.md`` as an
executable check, so "the gate passed" is a verdict from data rather than an
impression from reading a log:

```text
M2 — Tiny overfit:
- no NaN/Inf
- correct masks
- Viseme loss strongly decreases
- Strength loss strongly decreases
- save/load parity
```

Usage:

```bash
python -m articulm.gate --run-dir runs/articulm_v1_50m_tiny_overfit/<run_id>
python -m articulm.gate --summary path/to/training_summary.json --min-accuracy 0.99
```

Exit code 0 means every check passed. Any other value means a long training
run must not be started.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    CheckpointError,
    load_checkpoint,
)


@dataclass(frozen=True)
class GateCriteria:
    """Thresholds that operationalise "approaches a near-perfect fit"."""

    min_viseme_accuracy: float = 0.95
    max_strength_mae: float = 5.0
    require_finite_loss: bool = True
    require_validation: bool = True
    require_checkpoints: bool = True


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class GateResult:
    checks: list[GateCheck] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(GateCheck(name=name, passed=passed, detail=detail))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def render(self) -> str:
        lines = ["=" * 72, "Tiny-overfit gate (docs/09 M2)", "=" * 72]
        lines += [check.render() for check in self.checks]
        lines.append("=" * 72)
        if self.passed:
            lines.append("GATE PASSED — a long training run may start.")
        else:
            lines.append(
                f"GATE FAILED ({len(self.failures)} of {len(self.checks)} checks). "
                "Do not start a long training run; debug the implementation first."
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def evaluate_summary(
    summary: dict[str, Any], criteria: GateCriteria | None = None
) -> GateResult:
    """Check a ``training_summary.json`` payload against the gate criteria."""
    criteria = criteria or GateCriteria()
    result = GateResult()

    step = summary.get("global_step", 0)
    result.add(
        "training actually ran",
        isinstance(step, int) and step > 0,
        f"global_step={step}",
    )

    if criteria.require_finite_loss:
        saw_non_finite = summary.get("saw_non_finite_loss")
        result.add(
            "no NaN/Inf during training",
            saw_non_finite is False,
            f"saw_non_finite_loss={saw_non_finite}",
        )

        loss = summary.get("final_train_loss")
        result.add(
            "final train loss is finite",
            _finite(loss),
            f"final_train_loss={loss}",
        )

    accuracy = summary.get("final_train_viseme_accuracy")
    result.add(
        "viseme train accuracy approaches a perfect fit",
        _finite(accuracy) and accuracy >= criteria.min_viseme_accuracy,
        f"final_train_viseme_accuracy={accuracy} "
        f"(need >= {criteria.min_viseme_accuracy})",
    )

    mae = summary.get("final_train_strength_mae")
    result.add(
        "strength error converged",
        _finite(mae) and mae <= criteria.max_strength_mae,
        f"final_train_strength_mae={mae} "
        f"(need <= {criteria.max_strength_mae}, 0..100 units)",
    )

    if criteria.require_validation:
        validation = summary.get("validation")
        if not isinstance(validation, dict) or not validation:
            result.add(
                "validation metrics present",
                False,
                "no validation block in the summary; the gate cannot confirm "
                "the fit was measured",
            )
        else:
            result.add("validation metrics present", True, f"{sorted(validation)}")
            val_accuracy = validation.get("val_viseme_accuracy")
            result.add(
                "validation viseme accuracy approaches a perfect fit",
                _finite(val_accuracy) and val_accuracy >= criteria.min_viseme_accuracy,
                f"val_viseme_accuracy={val_accuracy} "
                f"(need >= {criteria.min_viseme_accuracy})",
            )
            val_mae = validation.get("val_strength_mae")
            result.add(
                "validation strength error converged",
                _finite(val_mae) and val_mae <= criteria.max_strength_mae,
                f"val_strength_mae={val_mae} (need <= {criteria.max_strength_mae})",
            )

    return result


def check_checkpoints(checkpoint_dir: str | Path, result: GateResult) -> None:
    """Confirm the run left loadable ``best.pt`` and ``last.pt`` behind."""
    directory = Path(checkpoint_dir)
    for name in (LAST_CHECKPOINT_NAME, BEST_CHECKPOINT_NAME):
        path = directory / name
        if not path.is_file():
            result.add(f"{name} exists", False, f"missing {path}")
            continue
        try:
            loaded = load_checkpoint(path)
        except CheckpointError as exc:
            result.add(f"{name} loads", False, str(exc))
            continue
        except Exception as exc:
            # Deliberately broad. A truncated, corrupt or wrong-format file
            # surfaces as UnpicklingError, RuntimeError, OSError and more. The
            # gate's job is to return a verdict, so any load failure becomes a
            # failed check rather than a traceback out of the launcher.
            result.add(
                f"{name} loads",
                False,
                f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            continue
        result.add(
            f"{name} loads",
            True,
            f"step={loaded.state.global_step} epoch={loaded.state.epoch} "
            f"seed={loaded.seed} params={loaded.extra.get('parameters')}",
        )


def find_summary(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "metrics" / "training_summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no training summary at {path}; did the run finish?"
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.gate",
        description="Verify the tiny-overfit acceptance gate (docs/09 M2).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="run directory produced by articulm.train")
    source.add_argument("--summary", help="path to a training_summary.json")
    parser.add_argument("--checkpoint-dir", default=None, help="override checkpoint dir")
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--max-strength-mae", type=float, default=5.0)
    parser.add_argument(
        "--skip-validation-checks",
        action="store_true",
        help="accept a run that produced no validation metrics",
    )
    parser.add_argument(
        "--skip-checkpoint-checks",
        action="store_true",
        help="do not try to load best.pt / last.pt",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        summary_path = (
            Path(args.summary) if args.summary else find_summary(args.run_dir)
        )
    except FileNotFoundError as exc:
        print(f"GATE FAILED: {exc}")
        return 2

    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    criteria = GateCriteria(
        min_viseme_accuracy=args.min_accuracy,
        max_strength_mae=args.max_strength_mae,
        require_validation=not args.skip_validation_checks,
        require_checkpoints=not args.skip_checkpoint_checks,
    )
    result = evaluate_summary(summary, criteria)

    if criteria.require_checkpoints:
        checkpoint_dir = args.checkpoint_dir or summary.get("checkpoint_dir")
        if checkpoint_dir:
            check_checkpoints(checkpoint_dir, result)
        else:
            result.add(
                "checkpoint directory known",
                False,
                "summary has no checkpoint_dir; pass --checkpoint-dir",
            )

    print(f"summary: {summary_path}")
    print(result.render())

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(
                {"summary": str(summary_path), **result.as_dict()},
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote {args.json_out}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
