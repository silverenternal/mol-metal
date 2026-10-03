# -*- coding: utf-8 -*-
"""Console-script entry point for the ``molmetal`` package.

Mirrors the OmegaFold ``omegafold/__main__.py`` pattern: a
``main()`` callable registered via ``console_scripts`` in
``pyproject.toml``.  Invoked as ``molmetal <subcommand>`` after
``uv pip install -e .`` (or via ``python -m molmetal``).

Subcommands are the canonical 4 from
:mod:`molmetal.scripts.pre_flight_check`:

* ``preflight``  — run pre-flight v3 (CPU-only; 25 → 59+ gates).
* ``aggregate``  — run :mod:`molmetal.scripts.audit_metrics_disk`.
* ``benchmark``  — run the standard 5-seed × 50-epoch protocol
  (Blackwell/RTX-5090 GPU path per AGENTS.md §5.1; USER_OK-gated).
* ``demo``       — run the synthetic batch demo at ``main.py``
  (the MolFlow-Triton Euler/ODE integration smoke).

This module intentionally ships thin — the heavy lifting lives in the
canonical scripts under :mod:`molmetal.scripts`.  We dispatch by
subcommand and exit-code propagate from the called script.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

# Subcommand dispatch table — mapping ``subcommand`` → module + callable.
# Each entry imports lazily to keep ``import molmetal`` cheap when the
# user only wants the package metadata (version + ROCm env-var
# bootstrap).
_SUBCOMMANDS = {
    "preflight": ("molmetal.scripts.pre_flight_check", "main"),
    "aggregate": ("molmetal.scripts.audit_metrics_disk", "main"),
    "demo": ("__main__", "_demo_main"),
}


def _import_subcommand(name: str):
    """Lazy import for the ``subcommand`` dispatch table."""
    module_name, callable_name = _SUBCOMMANDS[name]
    if module_name == "__main__":
        # ``demo`` is implemented in this file (was the original
        # main.py behaviour, preserved here so the legacy entry
        # point keeps working).
        return globals()[callable_name]
    import importlib

    mod = importlib.import_module(module_name)
    return getattr(mod, callable_name)


def _demo_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the original ``main.py`` end-to-end demo via import.

    The synthetic batch + EGNN + Triton-Euler ODE integration smoke
    from the legacy ``main.py`` lives here so the new
    ``molmetal`` console script covers the demo subcommand.
    """
    # The original ``main.py`` is a free function (not a module).
    # We import its module path by reading from the project root.
    project_root = _resolve_project_root()
    sys.path.insert(0, str(project_root))
    import main as _legacy_main  # type: ignore[import-not-found]

    _legacy_main.main()
    return 0


def _resolve_project_root() -> "Path":
    """Walk up from this file's directory until ``pyproject.toml`` is found."""
    from pathlib import Path

    here = Path(__file__).resolve().parent.parent  # molmetal/ → project root
    if (here / "pyproject.toml").exists():
        return here
    # Fallback: walk further up (sandbox-friendly)
    for ancestor in here.parents:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here  # last-resort: trust the layout


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level ``molmetal`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="molmetal",
        description=(
            "Mol-Metal CLI: precious-metal anticancer drug design framework. "
            "Mirrors OmegaFold's console-script entry pattern (omegafold/__main__.py)."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the molmetal package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subparsers.add_parser(
        "preflight",
        help="Run pre-flight v3 gate suite (CPU-only; ~75s wall).",
    )
    subparsers.add_parser(
        "aggregate",
        help="Aggregate per-metric JSON files into the by_metric/ dashboard.",
    )
    subparsers.add_parser(
        "demo",
        help="Run the MolFlow-Triton synthetic batch + EGNN + ODE integration smoke.",
    )
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point registered in ``pyproject.toml``.

    Mirrors ``omegafold.__main__.main`` shape but lighter — the
    OmegaFold entry does a full model + fasta2inputs run; Mol-Metal's
    dispatch surface is subcommand-routed to its canonical scripts.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from molmetal import __version__

        print(f"molmetal {__version__}")
        return 0

    if args.subcommand is None:
        parser.print_help()
        return 0

    subcommand_main = _import_subcommand(args.subcommand)
    result = subcommand_main()
    # If the subcommand returns an int, propagate as exit code; else 0.
    return int(result) if isinstance(result, int) else 0


__all__ = ["build_parser", "cli_main", "_demo_main", "_import_subcommand"]


if __name__ == "__main__":
    raise SystemExit(cli_main())
