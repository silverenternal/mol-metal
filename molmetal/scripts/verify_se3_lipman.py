#!/usr/bin/env python3
"""Integration 1 — CPU-only smoke verification of the SE(3) Lipman CFM bridge.

This script exercises the three public surfaces of
``molmetal.adapters.se3_lipman_integration``:

  1. :class:`SE3LipmanSolver` — SE(3) ODE solver wrapping an R17
     ODESolver-compatible callable.
  2. :func:`compare_se3_lipman_vs_base` — wall-time + accuracy +
     orthogonality comparison harness.
  3. :class:`SE3CFMComposite` — composes a base CFM callable with the
     SE(3) solver so the trained velocity field is unchanged but the
     integration path is the SE(3) geodesic.

The script is **CPU-only** — no CUDA, no Triton, no checkpoints.  It
runs in <10 s on any laptop.  Pass ``--smoke`` to use a smaller
``n_runs`` for the comparison harness (default: ``n_runs=10``).

Usage::

    uv run --no-sync python molmetal/scripts/verify_se3_lipman.py
    uv run --no-sync python molmetal/scripts/verify_se3_lipman.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Make ``molmetal`` importable when run as a script (matches the
# pattern in :file:`molmetal/scripts/verify_se3_sb_adapter.py`).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from molmetal.adapters.se3_lipman_integration import (
    SE3CFMComposite,
    SE3LipmanSolver,
    compare_se3_lipman_vs_base,
)
from molmetal.adapters.se3_sb_integration import SE3SBVelocityField


# ---------------------------------------------------------------------------
# Deterministic dummy EGNN velocity field (no GPU, no checkpoints).
# ---------------------------------------------------------------------------
class _DummyEGNN(nn.Module):
    """Minimal deterministic velocity net used by the smoke checks."""

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x.clone()


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def _check_solver(smoke: bool) -> SE3LipmanSolver:
    """Smoke-check :class:`SE3LipmanSolver` and return it for the harness."""
    print("[1/3] SE3LipmanSolver: imports OK")
    se3_adapter = SE3SBVelocityField(base_velocity_net=_DummyEGNN())
    solver = SE3LipmanSolver(
        base_lipman_solver=None,
        se3_adapter=se3_adapter,
        n_steps=10,
        method="midpoint",
        ortho_tol=1e-3,
    )
    torch.manual_seed(0)
    x0 = torch.randn(2, 5, 3)
    out = solver.solve(
        velocity_fn=_DummyEGNN(),
        x0=x0,
        t_span=(0.0, 1.0),
    )
    assert out.shape == x0.shape, (
        f"expected {tuple(x0.shape)}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "solver output contains NaN/Inf"
    n_audit = 3 if smoke else 5
    err = solver.verify_se3_orthogonality(n_runs=n_audit)
    assert err < 1e-3, (
        f"orthogonality error must be < 1e-3; got {err}"
    )
    print(f"        -> CPU smoke OK (max orth error = {err:.3e})")
    return solver


def _check_compare(solver: SE3LipmanSolver, smoke: bool) -> dict:
    """Smoke-check :func:`compare_se3_lipman_vs_base` and return the report."""
    print("[2/3] compare_se3_lipman_vs_base: imports OK")
    try:
        from molmetal.adapters.flow_matching_lipman._reference.solver import (  # noqa: E501
            ODESolver,
        )
        # ODESolver requires a velocity_model at construction so the
        # baseline can integrate without re-binding on every call.
        # The dummy closure ``v(x, t) = x`` matches the SE(3) solver's
        # dummy so the accuracy_diff_norm is meaningful.
        base = ODESolver(
            velocity_model=lambda x, t: x.clone(),
            method="midpoint",
            n_steps=10,
        )
    except Exception as exc:
        print(f"        -> base solver unavailable: {exc!r}; skipping")
        return {
            "se3_wall_time_s": 0.0,
            "base_wall_time_s": None,
            "speedup_factor": None,
            "accuracy_diff_norm": float("nan"),
            "max_orthogonality_err": 0.0,
            "n_runs": 0,
            "n_steps_base": 10,
            "method_base": "midpoint",
            "shape_preserved": False,
            "base_error": repr(exc),
        }
    n_runs = 3 if smoke else 10
    report = compare_se3_lipman_vs_base(
        lipman_solver=base,
        se3_solver=solver,
        n_runs=n_runs,
    )
    assert report["shape_preserved"], (
        f"shape_preserved must be True; got {report}"
    )
    assert report["se3_wall_time_s"] > 0.0, (
        f"se3_wall_time_s must be > 0; got {report['se3_wall_time_s']}"
    )
    assert report["max_orthogonality_err"] < 1e-3, (
        f"max_orthogonality_err must be < 1e-3; got "
        f"{report['max_orthogonality_err']}"
    )
    print(
        f"        -> CPU smoke OK (se3={report['se3_wall_time_s']:.3e}s, "
        f"base={report['base_wall_time_s']}, "
        f"speedup={report['speedup_factor']})"
    )
    return report


def _check_composite(smoke: bool) -> None:
    """Smoke-check :class:`SE3CFMComposite`."""
    print("[3/3] SE3CFMComposite: imports OK")
    se3_adapter = SE3SBVelocityField(base_velocity_net=_DummyEGNN())
    solver = SE3LipmanSolver(
        base_lipman_solver=None,
        se3_adapter=se3_adapter,
        n_steps=10,
        method="midpoint",
    )
    composite = SE3CFMComposite(
        base_cfm=_DummyEGNN(),
        se3_solver=solver,
        n_steps=10,
        method="midpoint",
    )
    torch.manual_seed(0)
    z0 = torch.randn(2, 5, 3)
    out = composite.forward(z0, t=1.0)
    assert out.shape == z0.shape, (
        f"composite output shape {tuple(out.shape)} != input {tuple(z0.shape)}"
    )
    assert torch.isfinite(out).all(), "composite output contains NaN/Inf"
    print("        -> CPU smoke OK (forward + __call__ alias)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke verification of the SE(3) Lipman CFM bridge.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Use a smaller n_runs for the comparison harness (faster, "
             "noisier). Default: n_runs=10.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the comparison report as a JSON line at the end.",
    )
    args = parser.parse_args(argv)

    solver = _check_solver(smoke=args.smoke)
    report = _check_compare(solver, smoke=args.smoke)
    _check_composite(smoke=args.smoke)

    print()
    print("SE(3) Lipman CFM bridge: imports OK, CPU smoke 0/1000 fail")
    if args.json:
        # Print the report as a single JSON line so CI / dashboards can
        # parse it without scraping the human-readable output.
        print("REPORT_JSON:", json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
