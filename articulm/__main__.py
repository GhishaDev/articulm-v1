"""Sub-command dispatcher: ``python -m articulm <command> [...]``.

The documented workflow uses the module CLIs directly
(``python -m articulm.train ...``); this is a convenience front-end that
forwards to the same entry points.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

COMMANDS: dict[str, str] = {
    "train": "articulm.train",
    "evaluate": "articulm.evaluate",
    "infer": "articulm.infer",
    "export": "articulm.export",
    "validate-data": "articulm.data.validate",
    "split-data": "articulm.data.split",
    "gate": "articulm.gate",
}


def _load(module_name: str) -> Callable[[Sequence[str] | None], int]:
    from importlib import import_module

    module = import_module(module_name)
    return module.main  # type: ignore[attr-defined,no-any-return]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m articulm <command> [options]\n")
        print("commands:")
        for name, module in COMMANDS.items():
            print(f"  {name:<14} -> python -m {module}")
        return 0 if args else 2

    command = args[0]
    if command not in COMMANDS:
        print(f"unknown command {command!r}; expected one of {sorted(COMMANDS)}")
        return 2
    return _load(COMMANDS[command])(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
