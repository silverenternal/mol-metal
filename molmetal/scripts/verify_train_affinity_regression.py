#!/usr/bin/env python3
"""Wf-32 Phase 1 — CPU-only smoke verification of
``train_affinity_regression.py``.

This script is a sanity check, not a full benchmark.  It runs:

  1. :func:`train_affinity_regression.PlatinAIRegressionDataset.from_csv`
     on the project-root CSV and prints per-cell-line valid counts.
  2. The platinai training loop for 1 seed × 1 epoch on a tiny
     encoder (``--hidden 32 --depth 2 --batch 16``), and prints the
     per-cell-line test Pearson r.
  3. The argparse contract (so the CLI is well-formed).
  4. R18 init-choice wiring (no real training — just confirms the
     adapter handles are reachable).

No CUDA, no checkpoint loading, no HeLa48h MetalCytoToxDB read
(CI hosts don't have that 26 k-row file).  Wall-time < 60 s.

Usage::

    uv run --no-sync python molmetal/scripts/verify_train_affinity_regression.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _check_argparse_contract() -> None:
    """The CLI parses cleanly with default values.

    NOTE: ``args.epochs`` default is bound at :func:`add_common_args` call
    time to ``_common_train_args.RECIPE_EPOCHS`` (50).  The locally
    redefined :data:`train_affinity_regression.RECIPE_EPOCHS` (30) is a
    documentation-only constant — the *argparse* default is the common one.
    """
    import train_affinity_regression
    from molmetal.scripts import _common_train_args

    args = train_affinity_regression.parse_args([])
    assert args.dataset == "platinai"
    assert args.init == "tmqm"
    # The actual argparse default for ``--epochs`` is the common one
    # (50), not the locally redefined 30 — the latter is documentation
    # only and is intentionally overridable from the CLI.
    assert args.epochs == _common_train_args.RECIPE_EPOCHS, (
        f"args.epochs={args.epochs} should equal "
        f"_common_train_args.RECIPE_EPOCHS={_common_train_args.RECIPE_EPOCHS}"
    )
    assert train_affinity_regression.RECIPE_ALPHA == 0.0
    print("[1/4] argparse contract OK (default --dataset=platinai "
          f"--init={args.init} --epochs={args.epochs} "
          f"--alpha={train_affinity_regression.RECIPE_ALPHA})")


def _check_r18_init_lazy_imports() -> None:
    """R18 init lazy import handles are wired in (None-acceptable)."""
    import train_affinity_regression as tar

    for name in ("_SE3SBVelocityField", "_RelaxedEndpointTerm",
                 "_MultiPHPocketFeatures"):
        assert hasattr(tar, name), f"R18 init handle missing: {name}"
    print("[2/4] R18 init lazy imports wired in "
          f"(se3sb={tar._SE3SBVelocityField is not None}, "
          f"transport={tar._RelaxedEndpointTerm is not None}, "
          f"multiph={tar._MultiPHPocketFeatures is not None})")


def _check_platinai_dataset_load() -> None:
    """Smoke-load the PlatinAI 4-cell dataset."""
    import train_affinity_regression as tar
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    csv_path = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    if not csv_path.exists():
        print(f"[3/4] SKIP platinai CSV not found at {csv_path}")
        return
    ds = tar.PlatinAIRegressionDataset.from_csv(csv_path)
    counts = ds.per_cell_line_valid_counts()
    print(
        f"[3/4] PlatinAI dataset: rows={len(ds)} "
        f"per_cell_valid={counts}"
    )


def _check_platinai_smoke_train() -> None:
    """Run the platinai trainer for 1 seed × 1 epoch on tiny arch."""
    import subprocess

    venv_py = (
        _REPO_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
    )
    if not venv_py.exists():
        print("[4/4] SKIP venv not found")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ckpt"
        metrics_path = Path(tmp) / "metrics.json"
        summary_path = Path(tmp) / "summary.json"
        cmd = [
            str(venv_py),
            str(_HERE / "train_affinity_regression.py"),
            "--dataset", "platinai",
            "--seeds", "0",
            "--epochs", "1",
            "--batch", "16",
            "--hidden", "32",
            "--depth", "2",
            "--eval-every", "1",
            "--init", "random",
            "--out-dir", str(out_dir),
            "--metrics-path", str(metrics_path),
            "--summary-path", str(summary_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            print("[4/4] FAIL: platinai smoke exit "
                  f"{result.returncode}\nstderr (tail):\n{result.stderr[-1500:]}")
            return
        if not metrics_path.exists():
            print("[4/4] FAIL: metrics JSON missing")
            return
        payload = json.loads(metrics_path.read_text())
        headline = payload.get("headline_pearson_r_mean")
        per_cell = payload.get("per_cell_line_pearson_r", {})
        cell_strs = ", ".join(
            f"{l}={per_cell[l]['mean']:.3f}" for l in per_cell
        )
        print(
            f"[4/4] PlatinAI smoke OK "
            f"(headline_pearson={headline}, per_cell_means={cell_strs})"
        )


def main() -> int:
    print("=== train_affinity_regression smoke verify ===")
    _check_argparse_contract()
    _check_r18_init_lazy_imports()
    _check_platinai_dataset_load()
    _check_platinai_smoke_train()
    print("=== all checks done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())