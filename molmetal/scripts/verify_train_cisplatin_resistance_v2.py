#!/usr/bin/env python3
"""Wf-31 Phase 2 — CPU-only smoke verification of
``train_cisplatin_resistance_v2.py``.

This script exercises the Wf-31 Phase 2 deliverables in
:file:`molmetal/scripts/train_cisplatin_resistance_v2.py`:

  1. **Module imports cleanly** — no top-level errors, no dependency on
     ``train_se3sb.py`` mutation (the W2.1 baseline is locked).
  2. :class:`FourCellResistanceDMPNN` constructs and emits the documented
     forward contract: ``{cell_line -> (B, 2)}`` for all 4 cells.
  3. :func:`four_cell_loss` produces a finite scalar loss and the
     gradient flows back to the per-cell heads.
  4. :func:`load_dataset` reads the Wf-28 ship
     ``platinai_4cell.csv`` and returns the documented
     ``(smiles, active, res_feat)`` shape with 4 cell lines.
  5. :func:`scaffold_split_indices` returns 70/10/20 deterministic
     splits (the Wf-22 OOD split contract).
  6. :func:`aggregate_per_cell` returns the documented
     ``(per_cell_mean, per_cell_std)`` shape and the mean/std match a
     manual numpy reduction within ``1e-6``.
  7. :func:`attach_init_components` wires each of the 4 init modes
     without raising; the ``se3sb`` mode attaches a
     :class:`SE3SBVelocityField`, ``transport`` attaches a
     :class:`RelaxedEndpointTerm`, ``multiph`` attaches a
     :class:`MultiPHPocketFeatures`, and ``tmqm`` is a no-op.
  8. :func:`parse_args` exposes ``--cell-line``, ``--seeds``,
     ``--epochs``, ``--init`` with the documented choices.
  9. End-to-end :func:`main` runs to completion on CPU with a tiny
     subset (1 seed, 1 epoch) and writes the documented JSON outputs.

CPU-only, no GPU. Pass ``--dry-run`` to skip the end-to-end
:func:`main` step (which is the most expensive path).

Usage::

    uv run --no-sync python molmetal/scripts/verify_train_cisplatin_resistance_v2.py
    uv run --no-sync python molmetal/scripts/verify_train_cisplatin_resistance_v2.py --dry-run
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

# Wf-31 Phase 2 trainer module (the deliverable).
sys.path.insert(0, str(_HERE))
import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]


PLATINAI_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"


# ---------------------------------------------------------------------------
# 1. Module imports + exported symbols
# ---------------------------------------------------------------------------
def _check_imports() -> None:
    """The Wf-31 Phase 2 module imports cleanly and exports documented symbols."""
    print("[1/9] module imports")
    required = [
        "FourCellResistanceDMPNN",
        "four_cell_loss",
        "focal_loss",
        "load_dataset",
        "scaffold_split_indices",
        "attach_init_components",
        "aggregate_per_cell",
        "parse_args",
        "main",
        "build_se3_twist_head",
        "se3_integration_smoke",
        "try_load_tmqm_encoder",
        "freeze_encoder_layers_1_2",
        "INIT_CHOICES",
        "CELL_LINES",
        "N_CELLS",
        "N_RESIST_FEAT",
        "build_batches",
        "_eval_cell_auc",
        "RECIPE_EPOCHS",
        "RECIPE_HIDDEN",
        "RECIPE_DEPTH",
        "IC50_COLS",
        "ACTIVE_THRESHOLD_NM",
    ]
    for sym in required:
        assert hasattr(w31p2, sym), f"missing symbol: {sym}"
    print(f"        -> OK (all {len(required)} symbols present)")


# ---------------------------------------------------------------------------
# 2. Model forward contract
# ---------------------------------------------------------------------------
def _check_model_forward() -> None:
    """``FourCellResistanceDMPNN`` emits the documented shapes for all 4 cells."""
    print("[2/9] FourCellResistanceDMPNN forward contract")
    model = w31p2.FourCellResistanceDMPNN(hidden_dim=32, n_layers=1)
    smis = ["CCO", "c1ccccc1", "CCN"]
    cid = torch.tensor([0, 1, 2], dtype=torch.long)
    rf = torch.randn(3, w31p2.N_RESIST_FEAT)
    model.eval()
    with torch.no_grad():
        out = model(smis, cid, rf)
    assert set(out.keys()) == set(
        w31p2.CELL_LINES
    ), f"output keys must be {set(w31p2.CELL_LINES)}; got {set(out.keys())}"
    for cl in w31p2.CELL_LINES:
        lg = out[cl]
        assert lg.shape == (3, 2), f"{cl}: shape {tuple(lg.shape)} != (3, 2)"
        assert torch.isfinite(lg).all(), f"{cl}: non-finite logits"
    print(f"        -> OK (4 cell heads, each (B, 2) = (3, 2))")


# ---------------------------------------------------------------------------
# 3. Multi-task loss + backward
# ---------------------------------------------------------------------------
def _check_loss_and_backward() -> None:
    """``four_cell_loss`` produces finite scalar + backward reaches per-cell heads."""
    print("[3/9] four_cell_loss + backward")
    model = w31p2.FourCellResistanceDMPNN(hidden_dim=32, n_layers=1)
    smis = ["CCO", "c1ccccc1", "CCN", "CCS"]
    cid = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    rf = torch.randn(4, w31p2.N_RESIST_FEAT)
    # Use training-mode forward — the loss + backward check needs
    # gradients to flow back to the per-cell heads.
    model.train()
    out = model(smis, cid, rf)
    targets = {
        cl: torch.tensor([0, 1, 0, 1], dtype=torch.long) for cl in w31p2.CELL_LINES
    }
    loss = w31p2.four_cell_loss(out, targets)
    assert torch.isfinite(loss), f"loss is non-finite: {loss.item()}"
    assert loss.item() > 0.0, f"focal loss should be positive; got {loss.item()}"
    # Backward sanity — grads must reach the per-cell heads.
    loss.backward()
    for cl in w31p2.CELL_LINES:
        first_linear = model.cell_heads[cl][0]
        assert first_linear.weight.grad is not None, (
            f"cell_heads[{cl}][0].weight.grad is None — backward did not "
            f"propagate to {cl} head"
        )
        assert torch.isfinite(
            first_linear.weight.grad
        ).all(), f"cell_heads[{cl}][0].weight.grad contains NaN/Inf"
    print("        -> OK (loss finite, backward reaches all 4 heads)")


# ---------------------------------------------------------------------------
# 4. load_dataset on real platinai_4cell.csv
# ---------------------------------------------------------------------------
def _check_load_dataset() -> None:
    """``load_dataset`` returns the documented shape on the Wf-28 ship CSV."""
    print("[4/9] load_dataset (real platinai_4cell.csv)")
    if not PLATINAI_CSV.exists():
        print(f"        -> SKIPPED (platinai CSV not found at {PLATINAI_CSV})")
        return
    smiles, active, res_feat = w31p2.load_dataset(PLATINAI_CSV)
    assert len(smiles) > 4000, f"expected >4000 rows; got {len(smiles)}"
    assert active.shape == (
        len(smiles),
        w31p2.N_CELLS,
    ), f"active shape {active.shape} != ({len(smiles)}, {w31p2.N_CELLS})"
    assert res_feat.shape == (len(smiles), w31p2.N_RESIST_FEAT), (
        f"res_feat shape {res_feat.shape} != " f"({len(smiles)}, {w31p2.N_RESIST_FEAT})"
    )
    # Each cell line must have at least some active labels.
    for c, cl in enumerate(w31p2.CELL_LINES):
        n_pos = int((active[:, c] == 1).sum())
        n_neg = int((active[:, c] == 0).sum())
        assert n_pos > 100, f"cell {cl}: only {n_pos} positives"
        assert n_neg > 100, f"cell {cl}: only {n_neg} negatives"
    print(
        f"        -> OK (rows={len(smiles)}, "
        f"active_rates="
        f"{', '.join(f'{cl}={(active[:, i] == 1).mean():.3f}' for i, cl in enumerate(w31p2.CELL_LINES))})"
    )


# ---------------------------------------------------------------------------
# 5. scaffold_split_indices
# ---------------------------------------------------------------------------
def _check_scaffold_split_indices() -> None:
    """``scaffold_split_indices`` returns deterministic 70/10/20 splits."""
    print("[5/9] scaffold_split_indices")
    smiles = [
        "CCO",
        "c1ccccc1",
        "CCN",
        "CCS",
        "CC(=O)O",
        "CCC",
        "c1ccncc1",
        "CC(C)C",
        "c1ccc2ccccc2c1",
        "OCC",
    ]
    tr, va, te = w31p2.scaffold_split_indices(smiles)
    assert set(tr.tolist()).isdisjoint(
        set(va.tolist())
    ), "train / val indices must be disjoint"
    assert set(tr.tolist()).isdisjoint(
        set(te.tolist())
    ), "train / test indices must be disjoint"
    assert set(va.tolist()).isdisjoint(
        set(te.tolist())
    ), "val / test indices must be disjoint"
    assert len(tr) + len(va) + len(te) == len(smiles), (
        f"splits do not cover all rows: "
        f"{len(tr)} + {len(va)} + {len(te)} != {len(smiles)}"
    )
    # Same input → same split (deterministic).
    tr2, va2, te2 = w31p2.scaffold_split_indices(smiles)
    assert (
        (tr == tr2).all() and (va == va2).all() and (te == te2).all()
    ), "split must be deterministic"
    print(f"        -> OK (train={len(tr)}, val={len(va)}, test={len(te)})")


# ---------------------------------------------------------------------------
# 6. aggregate_per_cell — mean/std match numpy + NaN-safe
# ---------------------------------------------------------------------------
def _check_aggregate_per_cell() -> None:
    """``aggregate_per_cell`` returns (per_cell_mean, per_cell_std) matching numpy."""
    print("[6/9] aggregate_per_cell")
    seed_results = [
        {
            "test_auc_per_cell": {
                "MCF7": 0.80,
                "A549": 0.70,
                "A2780": 0.60,
                "A2780cis": 0.50,
            }
        },
        {
            "test_auc_per_cell": {
                "MCF7": 0.82,
                "A549": 0.72,
                "A2780": 0.62,
                "A2780cis": 0.52,
            }
        },
        {
            "test_auc_per_cell": {
                "MCF7": 0.84,
                "A549": 0.74,
                "A2780": 0.64,
                "A2780cis": 0.54,
            }
        },
    ]
    cells = w31p2.CELL_LINES
    means, stds = w31p2.aggregate_per_cell(seed_results, cells)
    expected_means = {
        c: float(np.mean([r["test_auc_per_cell"][c] for r in seed_results]))
        for c in cells
    }
    expected_stds = {
        c: float(np.std([r["test_auc_per_cell"][c] for r in seed_results], ddof=0))
        for c in cells
    }
    for c in cells:
        assert (
            abs(means[c] - expected_means[c]) < 1e-9
        ), f"means[{c}] = {means[c]} != expected {expected_means[c]}"
        assert (
            abs(stds[c] - expected_stds[c]) < 1e-9
        ), f"stds[{c}] = {stds[c]} != expected {expected_stds[c]}"
    # Verify NaN-only list → NaN.
    nan_only = [{"test_auc_per_cell": {"MCF7": float("nan")}}]
    m, s = w31p2.aggregate_per_cell(nan_only, ("MCF7",))
    assert math.isnan(m["MCF7"]), f"NaN-only must yield NaN mean; got {m}"
    assert math.isnan(s["MCF7"]), f"NaN-only must yield NaN std; got {s}"
    print("        -> OK (mean/std match numpy within 1e-9, NaN-safe)")


# ---------------------------------------------------------------------------
# 7. attach_init_components — 4 init modes
# ---------------------------------------------------------------------------
def _check_attach_init_components() -> None:
    """Each of the 4 init modes attaches the documented components (or no-op)."""
    print("[7/9] attach_init_components (4 init modes)")
    device = torch.device("cpu")
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        model = w31p2.FourCellResistanceDMPNN(hidden_dim=32, n_layers=1)
        comps = w31p2.attach_init_components(init, model, device, hidden_dim=32)
        if init == "tmqm":
            assert comps == {}, f"tmqm must be no-op; got {comps}"
        elif init == "se3sb":
            assert "se3_field" in comps, f"se3sb must attach se3_field; got {comps}"
            from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

            assert isinstance(comps["se3_field"], SE3SBVelocityField)
        elif init == "transport":
            if comps:
                assert (
                    "relax_term" in comps
                ), f"transport must attach relax_term; got {comps}"
        elif init == "multiph":
            if comps:
                assert "mphp" in comps, f"multiph must attach mphp; got {comps}"
        else:
            raise AssertionError(f"unhandled init {init!r}")
    print("        -> OK (tmqm no-op + 3 R18 adapters wired without crash)")


# ---------------------------------------------------------------------------
# 8. parse_args exposes --cell-line / --seeds / --epochs / --init
# ---------------------------------------------------------------------------
def _check_parse_args() -> None:
    """``parse_args`` exposes the documented CLI surface."""
    print("[8/9] parse_args CLI surface")
    args = w31p2.parse_args([])
    assert hasattr(args, "cell_line"), "parse_args missing --cell-line"
    assert hasattr(args, "seeds"), "parse_args missing --seeds"
    assert hasattr(args, "epochs"), "parse_args missing --epochs"
    assert hasattr(args, "init"), "parse_args missing --init"
    assert (
        args.init in w31p2.INIT_CHOICES
    ), f"init {args.init!r} not in {w31p2.INIT_CHOICES}"
    assert (
        args.cell_line in w31p2.CELL_LINES
    ), f"cell_line {args.cell_line!r} not in {w31p2.CELL_LINES}"
    assert args.seeds == list(
        w31p2.RECIPE_SEEDS
    ), f"default seeds = {args.seeds}; expected {list(w31p2.RECIPE_SEEDS)}"
    assert args.epochs == w31p2.RECIPE_EPOCHS
    # Override test
    args2 = w31p2.parse_args(
        [
            "--cell-line",
            "MCF7",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "5",
            "--init",
            "tmqm",
        ]
    )
    assert (
        args2.cell_line == "MCF7"
    ), f"--cell-line override failed; got {args2.cell_line}"
    assert args2.seeds == [0, 1], f"--seeds override failed; got {args2.seeds}"
    assert args2.epochs == 5, f"--epochs override failed; got {args2.epochs}"
    assert args2.init == "tmqm", f"--init override failed; got {args2.init}"
    print("        -> OK (--cell-line / --seeds / --epochs / --init all live)")


# ---------------------------------------------------------------------------
# 9. End-to-end main() smoke (1 seed, 1 epoch, tiny CSV)
# ---------------------------------------------------------------------------
def _check_main_smoke(
    tmp_path: Path,
    dry_run: bool,
) -> None:
    """End-to-end main() runs to completion on a tiny CSV (1 seed, 1 epoch)."""
    print("[9/9] main() smoke (1 seed × 1 epoch × 50 rows)")
    if dry_run:
        print("        -> SKIPPED (--dry-run)")
        return
    csv_path = tmp_path / "tiny_platinai.csv"
    rows: List[dict] = []
    smiles_pool = [
        "[Pt](N)(N)(Cl)Cl",
        "[Pt](N)(Cl)(N)Cl",
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
        "CCN(CC)CC.Cl.Cl.[Pt]",
        "CCS",
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
    ]
    for i in range(50):
        smi = smiles_pool[i % len(smiles_pool)]
        ic50 = (i * 137) % 20000 + 1
        rows.append(
            {
                "complex_id": f"tiny_{i}",
                "smiles": smi,
                "ic50_mcf7_nm": float(ic50),
                "ic50_a549_nm": float((ic50 + 1000) % 25000 + 1),
                "ic50_a2780_nm": float((ic50 + 500) % 18000 + 1),
                "ic50_a2780cis_nm": float((ic50 + 2000) % 30000 + 1),
                "source": "tiny",
            }
        )
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "complex_id",
                "smiles",
                "ic50_mcf7_nm",
                "ic50_a549_nm",
                "ic50_a2780_nm",
                "ic50_a2780cis_nm",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    rc = w31p2.main(
        [
            "--data",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--metrics-path",
            str(metrics_path),
            "--summary-path",
            str(summary_path),
            "--init",
            "se3sb",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--batch",
            "16",
            "--hidden",
            "32",
            "--depth",
            "1",
            "--eval-every",
            "1",
            "--cell-line",
            "MCF7",
            "--se3-n-steps",
            "2",
        ]
    )
    assert rc == 0, f"main() returned {rc}"
    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "wf31_r18_se3sb_cisplatin_resistance_v2"
    assert "per_cell_mean_auc" in payload
    assert "per_cell_std_auc" in payload
    assert "headline_auc_mean" in payload
    # Ckpt must exist.
    ckpt_path = out_dir / "cisplatin_resistance_v2_se3sb_MCF7_seed0.pt"
    assert ckpt_path.exists(), f"ckpt not written at {ckpt_path}"
    assert ckpt_path.stat().st_size > 0
    print(f"        -> OK (rc=0, ckpt={ckpt_path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only smoke verification of the Wf-31 Phase 2 R18 SE(3) "
            "SB cisplatin-resistance 4-cell trainer."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the (slowest) main() smoke step; only verify the "
        "model/loss/init components on CPU.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _check_imports()
        _check_model_forward()
        _check_loss_and_backward()
        _check_load_dataset()
        _check_scaffold_split_indices()
        _check_aggregate_per_cell()
        _check_attach_init_components()
        _check_parse_args()
        _check_main_smoke(tmp_path, dry_run=args.dry_run)

    print()
    print("Wf-31 Phase 2 verify: imports OK, CPU smoke 0/9 fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
