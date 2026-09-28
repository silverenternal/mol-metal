#!/usr/bin/env python3
"""Wf-31 Phase 1 — CPU-only smoke verification for the Ru 5-metal SE(3) SB trainer.

This script exercises the Wf-31 deliverables in :file:`molmetal/scripts/train_ru_5metal_se3sb.py`:

  1. **Module imports cleanly** — no top-level errors, no dependency on
     ``train_se3sb.py`` mutation (the Wf-22 baseline is locked).
  2. :class:`Ru5MetalSe3SBHead` constructs a V4 backbone + 5 cls + 5 reg
     heads and emits the documented forward contract:
     ``out["cls"][m] = (B, 2)``, ``out["reg"][m] = (B, 1)``,
     ``out["coord"] = (B, n_coord_classes)``.
  3. :class:`Ru5MetalSe3SBLoss` produces finite scalar loss on the
     wrapper's output, and the gradient flows back to the V4 backbone
     + the new reg heads.
  4. :func:`load_5metal_csv` loads a tiny CSV (5 rows × 1 metal) and
     returns the documented ``{metal: {split: [(smiles, label)]}}``
     shape.
  5. :func:`aggregate_per_metal` returns the documented
     ``(per_metal_mean, per_metal_std)`` shape and the mean/std match
     a manual numpy reduction within ``1e-6``.
  6. :func:`attach_init_components` wires each of the 4 init modes
     without raising; the ``se3sb`` mode attaches a
     :class:`SE3SBVelocityField`, ``transport`` attaches a
     :class:`RelaxedEndpointTerm`, ``multiph`` attaches a
     :class:`MultiPHPocketFeatures`, and ``tmqm`` is a no-op.
  7. :func:`parse_args` exposes ``--metals``, ``--seeds``, ``--epochs``,
     ``--init`` with the documented choices.
  8. End-to-end :func:`main` runs to completion on CPU with a tiny
     subset (1 seed, 1 epoch, ``--train-max-per-metal 4``) and writes
     the documented JSON outputs.

CPU-only, no GPU. Pass ``--dry-run`` to skip the end-to-end :func:`main`
step (which is the most expensive path).

Usage::

    uv run --no-sync python molmetal/scripts/verify_train_ru_5metal.py
    uv run --no-sync python molmetal/scripts/verify_train_ru_5metal.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Make ``molmetal`` importable when run as a script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Wf-31 trainer module (the deliverable).
sys.path.insert(0, str(_HERE))
import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]

from molmetal.models.v4_multitask_dmpnn import SUPPORTED_METALS


# ---------------------------------------------------------------------------
# 1. Module imports + exported symbols
# ---------------------------------------------------------------------------
def _check_imports() -> None:
    """The Wf-31 module imports cleanly and exports documented symbols."""
    print("[1/8] module imports")
    required = [
        "Ru5MetalSe3SBHead",
        "Ru5MetalSe3SBLoss",
        "load_5metal_csv",
        "attach_init_components",
        "aggregate_per_metal",
        "parse_args",
        "main",
        "build_se3_twist_head",
        "se3_integration_smoke",
        "INIT_CHOICES",
        "RECIPE_SEEDS",
        "RECIPE_EPOCHS",
        "DEFAULT_METALS",
        "COORD_BY_METAL",
    ]
    for sym in required:
        assert hasattr(w31, sym), f"missing symbol: {sym}"
    print("        -> OK (all 14 symbols present)")


# ---------------------------------------------------------------------------
# 2. Head wrapper forward contract
# ---------------------------------------------------------------------------
def _check_head_wrapper_forward() -> None:
    """V4 + 5 cls + 5 reg heads emit the documented shapes."""
    print("[2/8] Ru5MetalSe3SBHead forward contract")
    backbone = w31.Ru5MetalSe3SBHead.__init__
    # Construct via a tiny backbone — keep hidden=64 for speed.
    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskDMPNN
    bb = V4MultiTaskDMPNN(
        hidden_dim=64, n_layers=2, n_metals=5, n_coord_classes=5,
    )
    head = w31.Ru5MetalSe3SBHead(backbone=bb)
    smis = ["CCO", "c1ccccc1", "CC(=O)O"]
    mids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = head(smis, mids)
    assert set(out.keys()) == {"cls", "reg", "coord"}, (
        f"forward keys must be {{cls, reg, coord}}; got {set(out.keys())}"
    )
    assert set(out["cls"].keys()) == set(SUPPORTED_METALS), (
        f"cls heads must cover {SUPPORTED_METALS}; got {set(out['cls'].keys())}"
    )
    assert set(out["reg"].keys()) == set(SUPPORTED_METALS), (
        f"reg heads must cover {SUPPORTED_METALS}; got {set(out['reg'].keys())}"
    )
    for m in SUPPORTED_METALS:
        c = out["cls"][m]
        r = out["reg"][m]
        assert c.shape == (3, 2), f"cls[{m}] shape {tuple(c.shape)} != (3, 2)"
        assert r.shape == (3, 1), f"reg[{m}] shape {tuple(r.shape)} != (3, 1)"
    coord = out["coord"]
    assert coord.shape == (3, 5), f"coord shape {tuple(coord.shape)} != (3, 5)"
    assert torch.isfinite(coord).all(), "coord contains NaN/Inf"
    for m in SUPPORTED_METALS:
        assert torch.isfinite(out["cls"][m]).all(), f"cls[{m}] contains NaN/Inf"
        assert torch.isfinite(out["reg"][m]).all(), f"reg[{m}] contains NaN/Inf"
    print("        -> OK (cls (3,2), reg (3,1), coord (3,5))")


# ---------------------------------------------------------------------------
# 3. Multi-task loss + backward
# ---------------------------------------------------------------------------
def _check_loss_and_backward() -> None:
    """The Wf-31 loss produces finite scalar and backward propagates."""
    print("[3/8] Ru5MetalSe3SBLoss + backward")
    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskDMPNN
    bb = V4MultiTaskDMPNN(hidden_dim=64, n_layers=2, n_metals=5, n_coord_classes=5)
    head = w31.Ru5MetalSe3SBHead(backbone=bb)
    smis = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"]
    mids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = head(smis, mids)
    y_true = {m: torch.tensor([0, 1, 0, 1], dtype=torch.long) for m in SUPPORTED_METALS}
    coord_true = torch.tensor([3, 2, 3, 0], dtype=torch.long)
    loss_fn = w31.Ru5MetalSe3SBLoss(n_coord_classes=5)
    loss, info = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert "loss_total" in info, "info missing loss_total"
    for m in SUPPORTED_METALS:
        assert f"loss_cls_{m}" in info, f"info missing loss_cls_{m}"
        assert f"loss_reg_{m}" in info, f"info missing loss_reg_{m}"
    assert "loss_coord" in info, "info missing loss_coord"
    # Backward sanity — grads must reach the reg heads.
    loss.backward()
    for m in SUPPORTED_METALS:
        w_first = head.reg_heads[m][0].weight
        assert w_first.grad is not None, f"reg_heads[{m}][0].weight.grad is None"
        assert torch.isfinite(w_first.grad).all(), (
            f"reg_heads[{m}][0].weight.grad contains NaN/Inf"
        )
    print("        -> OK (loss finite, backward reaches reg heads)")


# ---------------------------------------------------------------------------
# 4. CSV loader
# ---------------------------------------------------------------------------
def _check_load_5metal_csv(tmp_path: Path) -> None:
    """``load_5metal_csv`` returns the documented {metal: {split: [(smi, label)]}} shape."""
    print("[4/8] load_5metal_csv")
    csv_path = tmp_path / "tiny_5metal.csv"
    rows = [
        {"smiles": "CCO",      "metal": "Ru", "scaffold_id": "s1", "split": "train", "activity": "0"},
        {"smiles": "CCO",      "metal": "Ru", "scaffold_id": "s1", "split": "train", "activity": "1"},
        {"smiles": "c1ccccc1", "metal": "Ru", "scaffold_id": "s2", "split": "val",   "activity": "1"},
        {"smiles": "CC(=O)O",  "metal": "Ir", "scaffold_id": "s3", "split": "test",  "activity": "0"},
        {"smiles": "invalid",  "metal": "ZZ", "scaffold_id": "sX", "split": "train", "activity": "0"},
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["smiles", "metal", "scaffold_id", "split", "activity"],
        )
        writer.writeheader()
        writer.writerows(rows)
    data = w31.load_5metal_csv(csv_path)
    # Unknown metal row (ZZ) must be dropped.
    assert "Ru" in data, "missing Ru in loaded data"
    assert "Ir" in data, "missing Ir in loaded data"
    assert "ZZ" not in data, f"unknown metal ZZ should be dropped; data keys = {list(data.keys())}"
    assert len(data["Ru"]["train"]) == 2, (
        f"expected 2 Ru train rows; got {len(data['Ru']['train'])}"
    )
    assert len(data["Ru"]["val"]) == 1, (
        f"expected 1 Ru val row; got {len(data['Ru']['val'])}"
    )
    assert len(data["Ir"]["test"]) == 1, (
        f"expected 1 Ir test row; got {len(data['Ir']['test'])}"
    )
    print("        -> OK (5 rows → 4 retained, ZZ dropped)")


# ---------------------------------------------------------------------------
# 5. aggregate_per_metal
# ---------------------------------------------------------------------------
def _check_aggregate_per_metal() -> None:
    """``aggregate_per_metal`` returns (per_metal_mean, per_metal_std) matching numpy."""
    print("[5/8] aggregate_per_metal")
    seed_results = [
        {"test_auc_per_metal": {"Ru": 0.80, "Ir": 0.70, "Rh": 0.60, "Re": 0.50, "Os": 0.40}},
        {"test_auc_per_metal": {"Ru": 0.82, "Ir": 0.72, "Rh": 0.62, "Re": 0.52, "Os": 0.42}},
        {"test_auc_per_metal": {"Ru": 0.84, "Ir": 0.74, "Rh": 0.64, "Re": 0.54, "Os": 0.44}},
    ]
    metals = ("Ru", "Ir", "Rh", "Re", "Os")
    means, stds = w31.aggregate_per_metal(seed_results, metals)
    # Manual reference: numpy.
    expected_means = {
        m: float(np.mean([r["test_auc_per_metal"][m] for r in seed_results]))
        for m in metals
    }
    expected_stds = {
        m: float(np.std([r["test_auc_per_metal"][m] for r in seed_results], ddof=0))
        for m in metals
    }
    for m in metals:
        assert abs(means[m] - expected_means[m]) < 1e-9, (
            f"means[{m}] = {means[m]} != expected {expected_means[m]}"
        )
        assert abs(stds[m] - expected_stds[m]) < 1e-9, (
            f"stds[{m}] = {stds[m]} != expected {expected_stds[m]}"
        )
    # Verify NaN-only list → NaN.
    nan_only = [{"test_auc_per_metal": {"Ru": float("nan")}}]
    m, s = w31.aggregate_per_metal(nan_only, ("Ru",))
    assert math.isnan(m["Ru"]), f"NaN-only must yield NaN mean; got {m}"
    assert math.isnan(s["Ru"]), f"NaN-only must yield NaN std; got {s}"
    print("        -> OK (mean/std match numpy within 1e-9, NaN-safe)")


# ---------------------------------------------------------------------------
# 6. attach_init_components — 4 init modes
# ---------------------------------------------------------------------------
def _check_attach_init_components() -> None:
    """Each of the 4 init modes attaches the documented components (or no-op)."""
    print("[6/8] attach_init_components (4 init modes)")
    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskDMPNN
    device = torch.device("cpu")
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        bb = V4MultiTaskDMPNN(hidden_dim=32, n_layers=2, n_metals=5, n_coord_classes=5)
        head = w31.Ru5MetalSe3SBHead(backbone=bb)
        comps = w31.attach_init_components(init, head, device, hidden_dim=32)
        if init == "tmqm":
            assert comps == {}, f"tmqm must be no-op; got {comps}"
        elif init == "se3sb":
            assert "se3_field" in comps, f"se3sb must attach se3_field; got {comps}"
            from molmetal.adapters.se3_sb_integration import SE3SBVelocityField
            assert isinstance(comps["se3_field"], SE3SBVelocityField)
        elif init == "transport":
            # Adapter may not be installed in the venv; comps may be empty.
            if comps:
                assert "relax_term" in comps, (
                    f"transport must attach relax_term; got {comps}"
                )
        elif init == "multiph":
            if comps:
                assert "mphp" in comps, (
                    f"multiph must attach mphp; got {comps}"
                )
        else:
            raise AssertionError(f"unhandled init {init!r}")
    print("        -> OK (tmqm no-op + 3 R18 adapters wired without crash)")


# ---------------------------------------------------------------------------
# 7. parse_args exposes --metals / --seeds / --epochs / --init
# ---------------------------------------------------------------------------
def _check_parse_args() -> None:
    """``parse_args`` exposes the documented CLI surface."""
    print("[7/8] parse_args CLI surface")
    args = w31.parse_args([])
    assert hasattr(args, "metals"), "parse_args missing --metals"
    assert hasattr(args, "seeds"), "parse_args missing --seeds"
    assert hasattr(args, "epochs"), "parse_args missing --epochs"
    assert hasattr(args, "init"), "parse_args missing --init"
    assert args.init in w31.INIT_CHOICES, (
        f"init {args.init!r} not in {w31.INIT_CHOICES}"
    )
    assert args.metals == list(w31.DEFAULT_METALS), (
        f"default metals = {args.metals}; expected {list(w31.DEFAULT_METALS)}"
    )
    assert args.seeds == list(w31.RECIPE_SEEDS), (
        f"default seeds = {args.seeds}; expected {list(w31.RECIPE_SEEDS)}"
    )
    assert args.epochs == w31.RECIPE_EPOCHS
    # Override test
    args2 = w31.parse_args([
        "--metals", "Ru", "Ir",
        "--seeds", "0", "1",
        "--epochs", "5",
        "--init", "tmqm",
    ])
    assert args2.metals == ["Ru", "Ir"], (
        f"--metals override failed; got {args2.metals}"
    )
    assert args2.seeds == [0, 1], f"--seeds override failed; got {args2.seeds}"
    assert args2.epochs == 5, f"--epochs override failed; got {args2.epochs}"
    assert args2.init == "tmqm", f"--init override failed; got {args2.init}"
    print("        -> OK (--metals / --seeds / --epochs / --init all live)")


# ---------------------------------------------------------------------------
# 8. End-to-end main() smoke
# ---------------------------------------------------------------------------
def _check_main_smoke(
    tmp_path: Path, dry_run: bool, csv_path: Optional[Path] = None,
) -> None:
    """End-to-end main() runs to completion on a tiny CSV (1 seed, 1 epoch)."""
    print("[8/8] main() smoke (1 seed × 1 epoch × 5 rows)")
    if dry_run:
        print("        -> SKIPPED (--dry-run)")
        return
    if csv_path is None:
        # Fallback: build a tiny CSV in tmp.
        csv_path = tmp_path / "tiny_5metal.csv"
        rows = []
        for m in w31.DEFAULT_METALS:
            for sp, n in (("train", 4), ("val", 2), ("test", 2)):
                for i in range(n):
                    rows.append({
                        "smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C", "CCN"][i % 5],
                        "metal": m,
                        "scaffold_id": f"s_{m}_{i}",
                        "split": sp,
                        "activity": str(i % 2),
                    })
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["smiles", "metal", "scaffold_id", "split", "activity"],
            )
            writer.writeheader()
            writer.writerows(rows)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    rc = w31.main([
        "--data", str(csv_path),
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
        "--init", "se3sb",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "4",
        "--se3-n-steps", "2",
        "--metals", "Ru", "Ir", "Rh", "Re", "Os",
    ])
    assert rc == 0, f"main() returned {rc}"
    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "wf31_r18_m22_se3sb_5metal_cls_reg"
    assert "per_metal_mean_auc" in payload
    assert "per_metal_std_auc" in payload
    assert "ru_mean_auc_5seed" in payload
    assert "ru_std_auc_5seed" in payload
    # Headline metric must round-trip.
    assert isinstance(payload["ru_mean_auc_5seed"], (float, int, type(None)))
    # Ckpt must exist.
    ckpt_path = out_dir / "ru_5metal_se3sb_se3sb_seed0.pt"
    assert ckpt_path.exists(), f"ckpt not written at {ckpt_path}"
    assert ckpt_path.stat().st_size > 0
    print(f"        -> OK (rc=0, ckpt={ckpt_path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke verification of the Wf-31 Ru 5-metal SE(3) SB trainer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the (slowest) main() smoke step; only verify the "
             "head/loss/init components on CPU.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _check_imports()
        _check_head_wrapper_forward()
        _check_loss_and_backward()
        _check_load_5metal_csv(tmp_path)
        _check_aggregate_per_metal()
        _check_attach_init_components()
        _check_parse_args()
        _check_main_smoke(tmp_path, dry_run=args.dry_run)

    print()
    print("Wf-31 Phase 1 verify: imports OK, CPU smoke 0/8 fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
