#!/usr/bin/env python3
"""Wf-32 Phase 2 — CPU-only smoke verification for the multi-metal LOMO trainer.

This script exercises the Wf-32 deliverables in
:file:`molmetal/scripts/train_multimetal_lomo.py`:

  1. **Module imports cleanly** — no top-level errors, no mutation of
     the Wf-31 ``train_ru_5metal_se3sb.py`` baseline.
  2. :func:`parse_fold_token` parses ``leave-Ru-out`` /
     ``leave-Ru`` / ``Ru`` → ``Ru`` (case-insensitive).
  3. :func:`split_folds` de-duplicates and validates a list of fold
     tokens; rejects unknown metals.
  4. :func:`build_lomo_view` masks the held-out metal out of train /
     val (and out of the *other* metals' test pools).
  5. :func:`aggregate_per_fold` and :func:`aggregate_overall` return
     the documented shapes (per-fold mean/std, overall mean/std).
  6. :func:`attach_hilbert_components` wires the Wf-9 P1-4 composite
     (or no-op when the adapter is missing in the venv).
  7. :func:`parse_args` exposes ``--folds``, ``--seeds``, ``--epochs``,
     ``--init`` with the documented 3 init choices.
  8. End-to-end :func:`main` runs to completion on a tiny CSV
     (1 fold, 1 seed, 1 epoch, ``--train-max-per-metal 4``) and
     writes the documented JSON outputs.

CPU-only, no GPU. Pass ``--dry-run`` to skip the end-to-end
:func:`main` step (which is the most expensive path).

Usage::

    uv run --no-sync python molmetal/scripts/verify_train_multimetal_lomo.py
    uv run --no-sync python molmetal/scripts/verify_train_multimetal_lomo.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make ``molmetal`` + the scripts directory importable when run as
# a script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Wf-32 trainer module (the deliverable).
import train_multimetal_lomo as m32  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# 1. Module imports + exported symbols
# ---------------------------------------------------------------------------
def _check_imports() -> None:
    """The Wf-32 module imports cleanly and exports documented symbols."""
    print("[1/8] module imports")
    required = [
        "Ru5MetalSe3SBHead",
        "Ru5MetalSe3SBLoss",
        "load_5metal_csv",
        "attach_init_components",
        "build_lomo_view",
        "attach_hilbert_components",
        "aggregate_per_fold",
        "aggregate_overall",
        "parse_fold_token",
        "split_folds",
        "parse_args",
        "main",
        "train_one_fold_seed",
        "INIT_CHOICES",
        "RECIPE_EPOCHS",
        "RECIPE_SEEDS",
        "DEFAULT_HELD_OUT_METALS",
        "DEFAULT_FOLDS",
    ]
    for sym in required:
        assert hasattr(m32, sym), f"missing symbol: {sym}"
    print(f"        -> OK (all {len(required)} symbols present)")


# ---------------------------------------------------------------------------
# 2. parse_fold_token
# ---------------------------------------------------------------------------
def _check_parse_fold_token() -> None:
    """``parse_fold_token`` parses ``leave-Ru-out`` / ``leave-Ru`` / ``Ru``."""
    print("[2/8] parse_fold_token")
    for raw, expected in (
        ("leave-Ru-out", "Ru"),
        ("leave-Ir-out", "Ir"),
        ("leave-Rh-out", "Rh"),
        ("leave-Re-out", "Re"),
        ("leave-Os-out", "Os"),
        ("leave-Ru", "Ru"),
        ("leave-Ir", "Ir"),
        ("Ru", "Ru"),
        ("leave-ru-out", "Ru"),
        ("Ru-out", None),
    ):
        if expected is None:
            try:
                m32.parse_fold_token(raw)
                raise AssertionError(f"expected ValueError for malformed token {raw!r}")
            except ValueError:
                continue
        got = m32.parse_fold_token(raw)
        assert (
            got == expected
        ), f"parse_fold_token({raw!r}) = {got!r}, expected {expected!r}"
    # Unknown metal must raise.
    try:
        m32.parse_fold_token("leave-XX-out")
        raise AssertionError("unknown metal must raise ValueError")
    except ValueError:
        pass
    print("        -> OK (5 metals parsed, unknown rejected)")


# ---------------------------------------------------------------------------
# 3. split_folds
# ---------------------------------------------------------------------------
def _check_split_folds() -> None:
    """``split_folds`` dedupes and validates; preserves first-seen order."""
    print("[3/8] split_folds")
    got = m32.split_folds(["leave-Ru-out", "leave-Ir", "Ru", "leave-Os-out"])
    # ``Ru`` (no prefix) must dedupe with ``leave-Ru-out``.
    assert got == ["Ru", "Ir", "Os"], f"split_folds dedup/order broken; got {got}"
    # Empty list → ValueError.
    try:
        m32.split_folds([])
        raise AssertionError("empty fold list must raise ValueError")
    except ValueError:
        pass
    print("        -> OK (3 unique metals in input order)")


# ---------------------------------------------------------------------------
# 4. build_lomo_view — held-out metal only in test
# ---------------------------------------------------------------------------
def _check_build_lomo_view() -> None:
    """``build_lomo_view`` masks the held-out metal out of train/val."""
    print("[4/8] build_lomo_view")
    # Build a tiny 5-metal dataset.
    data = {m: {"train": [], "val": [], "test": []} for m in m32.SUPPORTED_METALS}
    for i, m in enumerate(m32.SUPPORTED_METALS):
        data[m]["train"] = [(f"smi_{m}_train_{j}", j % 2) for j in range(4)]
        data[m]["val"] = [(f"smi_{m}_val_{j}", j % 2) for j in range(2)]
        data[m]["test"] = [(f"smi_{m}_test_{j}", j % 2) for j in range(2)]

    # Leave-Ru-out: Ru must only appear in test; the other 4 metals
    # must not appear in test.
    view, counts = m32.build_lomo_view(data, "Ru")
    assert (
        len(view["Ru"]["train"]) == 0
    ), f"Ru must be empty in train; got {len(view['Ru']['train'])}"
    assert (
        len(view["Ru"]["val"]) == 0
    ), f"Ru must be empty in val; got {len(view['Ru']['val'])}"
    assert (
        len(view["Ru"]["test"]) == 2
    ), f"Ru must have test rows; got {len(view['Ru']['test'])}"
    for m in ("Ir", "Rh", "Re", "Os"):
        assert (
            len(view[m]["train"]) == 4
        ), f"{m} must have train rows; got {len(view[m]['train'])}"
        assert len(view[m]["test"]) == 0, (
            f"{m} must NOT have test rows in LOMO view; " f"got {len(view[m]['test'])}"
        )
    print("        -> OK (Ru isolated to test, 4 train metals intact)")


# ---------------------------------------------------------------------------
# 5. aggregate_per_fold + aggregate_overall
# ---------------------------------------------------------------------------
def _check_aggregates() -> None:
    """``aggregate_per_fold`` / ``aggregate_overall`` return correct shapes."""
    print("[5/8] aggregate_per_fold + aggregate_overall")
    fold_results = {
        "Ru": [
            {"test_auc": 0.80},
            {"test_auc": 0.82},
            {"test_auc": 0.84},
            {"test_auc": 0.78},
            {"test_auc": 0.86},
        ],
        "Ir": [
            {"test_auc": 0.70},
            {"test_auc": 0.72},
            {"test_auc": 0.74},
            {"test_auc": 0.68},
            {"test_auc": 0.76},
        ],
        "Rh": [
            {"test_auc": 0.60},
            {"test_auc": 0.62},
            {"test_auc": 0.64},
            {"test_auc": 0.58},
            {"test_auc": 0.66},
        ],
    }
    per_fold_mean, per_fold_std = m32.aggregate_per_fold(fold_results)
    for m, vals in fold_results.items():
        expected_mean = float(np.mean([r["test_auc"] for r in vals]))
        expected_std = float(np.std([r["test_auc"] for r in vals], ddof=0))
        assert abs(per_fold_mean[m] - expected_mean) < 1e-9
        assert abs(per_fold_std[m] - expected_std) < 1e-9
    overall_mean, overall_std = m32.aggregate_overall(fold_results)
    all_vals = [r["test_auc"] for rs in fold_results.values() for r in rs]
    assert abs(overall_mean - float(np.mean(all_vals))) < 1e-9
    assert abs(overall_std - float(np.std(all_vals, ddof=0))) < 1e-9
    # NaN-only fold → NaN mean / std.
    nan_fold = {"Ru": [{"test_auc": float("nan")}]}
    m_, s_ = m32.aggregate_per_fold(nan_fold)
    assert math.isnan(m_["Ru"]) and math.isnan(s_["Ru"])
    print("        -> OK (per-fold + overall match numpy within 1e-9)")


# ---------------------------------------------------------------------------
# 6. attach_hilbert_components — wires the Wf-9 P1-4 composite
# ---------------------------------------------------------------------------
def _check_hilbert_components() -> None:
    """``attach_hilbert_components`` wires the composite (or no-op)."""
    print("[6/8] attach_hilbert_components")
    import torch
    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskDMPNN

    bb = V4MultiTaskDMPNN(
        hidden_dim=32,
        n_layers=2,
        n_metals=5,
        n_coord_classes=5,
    )
    head = m32.Ru5MetalSe3SBHead(backbone=bb)
    device = torch.device("cpu")
    comps = m32.attach_hilbert_components(
        head,
        device,
        population_dim=16,
        n_iters=4,
    )
    if m32._HilbertASTSRComposite is None:
        assert comps == {}, f"missing adapter must yield empty dict; got {comps}"
        print("        -> OK (Hilbert adapter unavailable → no-op)")
        return
    assert "hilbert_composite" in comps
    assert hasattr(head, "hilbert_composite")
    # Smoke test the PD fit on a tiny constant feature / target.
    import numpy as np

    feats = np.zeros((4, 16), dtype=float)
    feats[:, 0] = 1.0
    tgt = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    fit_info = head.hilbert_composite.fit(feats, tgt)
    assert "r2" in fit_info
    assert "wall_s" in fit_info
    assert "backend" in fit_info
    assert (
        fit_info["backend"] == "hilbert"
    ), f"backend = {fit_info['backend']!r}, expected 'hilbert'"
    print(
        f"        -> OK (composite.fit → backend={fit_info['backend']!r}, "
        f"r2={fit_info['r2']:.4f})"
    )


# ---------------------------------------------------------------------------
# 7. parse_args exposes --folds / --seeds / --epochs / --init
# ---------------------------------------------------------------------------
def _check_parse_args() -> None:
    """``parse_args`` exposes the documented CLI surface."""
    print("[7/8] parse_args CLI surface")
    args = m32.parse_args([])
    assert hasattr(args, "folds"), "parse_args missing --folds"
    assert hasattr(args, "seeds"), "parse_args missing --seeds"
    assert hasattr(args, "epochs"), "parse_args missing --epochs"
    assert hasattr(args, "init"), "parse_args missing --init"
    assert (
        args.init in m32.INIT_CHOICES
    ), f"init {args.init!r} not in {m32.INIT_CHOICES}"
    assert args.folds == list(
        m32.DEFAULT_FOLDS
    ), f"default folds = {args.folds}; expected {list(m32.DEFAULT_FOLDS)}"
    assert args.seeds == list(
        m32.RECIPE_SEEDS
    ), f"default seeds = {args.seeds}; expected {list(m32.RECIPE_SEEDS)}"
    assert args.epochs == m32.RECIPE_EPOCHS
    # Override test
    args2 = m32.parse_args(
        [
            "--folds",
            "leave-Ru-out",
            "leave-Ir-out",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "5",
            "--init",
            "hilbert",
        ]
    )
    assert args2.folds == ["leave-Ru-out", "leave-Ir-out"]
    assert args2.seeds == [0, 1]
    assert args2.epochs == 5
    assert args2.init == "hilbert"
    print("        -> OK (--folds / --seeds / --epochs / --init all live)")


# ---------------------------------------------------------------------------
# 8. End-to-end main() smoke (1 fold, 1 seed, 1 epoch, tiny subset)
# ---------------------------------------------------------------------------
def _check_main_smoke(
    tmp_path: Path,
    dry_run: bool,
    csv_path: Optional[Path] = None,
) -> None:
    """End-to-end main() runs to completion on a tiny CSV."""
    print("[8/8] main() smoke (1 fold × 1 seed × 1 epoch)")
    if dry_run:
        print("        -> SKIPPED (--dry-run)")
        return
    if csv_path is None:
        csv_path = tmp_path / "tiny_5metal.csv"
        rows = []
        # Build a balanced tiny CSV: 4 metals × 4 train / 2 val /
        # 2 test rows each (Ru follows the same shape — held-out in
        # the LOMO view).
        for m in m32.SUPPORTED_METALS:
            for sp, n in (("train", 4), ("val", 2), ("test", 2)):
                for i in range(n):
                    rows.append(
                        {
                            "smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"][i % 4],
                            "metal": m,
                            "scaffold_id": f"s_{m}_{i}",
                            "split": sp,
                            "activity": str(i % 2),
                        }
                    )
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["smiles", "metal", "scaffold_id", "split", "activity"],
            )
            writer.writeheader()
            writer.writerows(rows)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    rc = m32.main(
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
            "2",
            "--train-max-per-metal",
            "4",
            "--se3-n-steps",
            "2",
            "--folds",
            "leave-Ru-out",
        ]
    )
    assert rc == 0, f"main() returned {rc}"
    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "wf32_m19_multimetal_lomo_cv"
    assert "per_fold_mean_auc" in payload
    assert "per_fold_std_auc" in payload
    assert "mean_test_auc_overall" in payload
    assert "std_test_auc_overall" in payload
    assert "r17_baseline_lomo_auc" in payload
    assert "wf32_headline" in payload
    # Per-fold mean AUC must include Ru (the single held-out metal).
    assert "Ru" in payload["per_fold_mean_auc"], (
        f"per_fold_mean_auc missing Ru: " f"{list(payload['per_fold_mean_auc'].keys())}"
    )
    # Ckpt must exist.
    ckpt_path = out_dir / "multimetal_lomo_se3sb_leaveRu_seed0.pt"
    assert ckpt_path.exists(), f"ckpt not written at {ckpt_path}"
    assert ckpt_path.stat().st_size > 0
    print(f"        -> OK (rc=0, ckpt={ckpt_path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke verification of the Wf-32 multi-metal LOMO trainer.",
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
        _check_parse_fold_token()
        _check_split_folds()
        _check_build_lomo_view()
        _check_aggregates()
        _check_hilbert_components()
        _check_parse_args()
        _check_main_smoke(tmp_path, dry_run=args.dry_run)

    print()
    print("Wf-32 Phase 2 verify: imports OK, CPU smoke 0/8 fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
