"""Helpers for skipping eager-only input validation during graph capture.

The model asserts things like "every sequence has at least one unpadded
token". Those checks read tensor *values*, which ``torch.jit.trace`` and
``torch.export`` cannot represent — they either bake in a constant or fail
outright. Validation is therefore skipped while a graph is being captured and
kept everywhere else, including all training and eager inference.
"""

from __future__ import annotations

import torch


def is_graph_capture() -> bool:
    """True while TorchScript tracing, torch.export or torch.compile runs."""
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return True
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        for probe in ("is_exporting", "is_compiling"):
            check = getattr(compiler, probe, None)
            if check is not None and check():
                return True
    return False
