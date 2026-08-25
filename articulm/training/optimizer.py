"""AdamW construction with weight-decay and split-LR parameter groups.

Biases, LayerNorm weights and embedding tables are excluded from weight decay
(standard transformer practice). ``head_learning_rate`` gives the output heads
their own LR, which is what docs/04 recommends for Human Gold fine-tuning.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import OptimizerConfig


def _is_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
    """Whether weight decay should apply to this parameter."""
    if parameter.ndim <= 1:  # biases and LayerNorm weights
        return False
    if ".tables." in name or name.startswith("embeddings."):
        return False
    return "learned_positions" not in name


def build_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig,
    *,
    head_parameter_names: tuple[str, ...] = (),
) -> torch.optim.Optimizer:
    """Build AdamW with decay / no-decay and backbone / head groups."""
    if cfg.type.lower() != "adamw":
        raise ValueError(f"unsupported optimizer {cfg.type!r}")

    head_names = set(head_parameter_names)
    head_lr = cfg.head_learning_rate if cfg.head_learning_rate is not None else cfg.learning_rate

    groups: dict[str, dict[str, object]] = {
        "backbone_decay": {"params": [], "weight_decay": cfg.weight_decay, "lr": cfg.learning_rate},
        "backbone_no_decay": {"params": [], "weight_decay": 0.0, "lr": cfg.learning_rate},
        "head_decay": {"params": [], "weight_decay": cfg.weight_decay, "lr": head_lr},
        "head_no_decay": {"params": [], "weight_decay": 0.0, "lr": head_lr},
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        section = "head" if name in head_names else "backbone"
        decay = "decay" if _is_decay_parameter(name, parameter) else "no_decay"
        groups[f"{section}_{decay}"]["params"].append(parameter)  # type: ignore[union-attr]

    param_groups = [
        {"name": key, **value}
        for key, value in groups.items()
        if value["params"]  # type: ignore[index]
    ]
    if not param_groups:
        raise ValueError("no trainable parameters found")

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=tuple(cfg.betas),  # type: ignore[arg-type]
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def parameter_group_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, object]]:
    """Human-readable group sizes, for the pre-run report."""
    return [
        {
            "name": group.get("name", f"group_{index}"),
            "num_parameters": sum(p.numel() for p in group["params"]),
            "num_tensors": len(group["params"]),
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
