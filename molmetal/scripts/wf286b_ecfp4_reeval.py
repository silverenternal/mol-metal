"""Wf-286B: ECFP4-CSV-direct re-eval (proper follow-up of Wf-275B adapter).

==============================================================================
Wf-286B-ECFP4-CSV-Direct: TODO-130 G5 follow-up of Wf-275B (the ECFP4 + scaffold
adapter) and Wf-277B (the chemprop v2 MolGraph re-eval that was mis-applied to
the wrong featurization).

Why this workflow exists
------------------------
Wf-277B re-evaluated the DrugOOD OOD AUC collapse (Wf-253-D 0.4544) but used
chemprop v2 MolGraph featurization on a custom pool-then-split CSV -- not the
ECFP4 + Bemis-Murcko scaffold-split CSV that Wf-275B ships.  So Wf-277B was
NOT a true ECFP4-direct re-eval.

This script IS the ECFP4-direct re-eval:

  1. Load the Wf-275B tidy CSV (``molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv``)
     -- 95,236 rows after RDKit canonicalisation.
  2. Decode the packed-hex ECFP4 bit vectors back into ``np.uint8 [N, 2048]``
     arrays (low-nibble-first, the inverse of
     :func:`molmetal.scripts.drugood_ecfp4_scaffold_adapter._arr_to_hex_packed`).
  3. Use the precomputed ``split`` column (``train`` / ``val`` / ``test``) that
     the adapter baked in -- Bemis-Murcko scaffold split, leak-free.
  4. Train a v3-fallback-mode head (chemprop v2 ``BinaryClassificationFFN``)
     on the ECFP4 bit vectors directly (no MPNN -- we want to isolate the
     ECFP4 feature axis from the chemprop message-passing axis).
  5. Report ``ood_test_auc`` (= test-set AUC since this is a single-ECFP4
     CSV with one pool-then-split partition) and compare to the Wf-185
     chemprop v2 baseline (0.6707 ± 0.0042 for ``lbap_refined_ic50_scaffold``).

What this script deliberately DOES NOT do
----------------------------------------
* Run the full chemprop v2 message-passing backbone.  We want to know whether
  ECFP4 features alone can recover the Wf-185 baseline AUC; running MPNN on
  top would conflate the two axes.
* Re-do scaffold splitting.  The Wf-275B adapter already assigned a
  leak-free scaffold split; we reuse the ``split`` column.
* Touch the trained Wf-185 / Wf-225 / Wf-230 / Wf-263A checkpoints.  This
  is a clean-side re-eval -- no checkpoint mutation.

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* This is a 1-seed × 5-epoch SMOKE run only; full retrain (5-seed × 20-ep)
  is a follow-up that ships a Wf-286B-Full workflow.  Smoke AUC numbers are
  honest-measured but NOT statistically significant on a single seed.
* If ECFP4 + chemprop v2 BinaryClassificationFFN ≥ 0.6707 - 0.02 (the
  Wf-185 v2 baseline minus seed-noise slack), then the ECFP4 axis is NOT
  the bottleneck for the Wf-253-D 0.4544 collapse -- the backbone is.
  Below that, the ECFP4 axis is a co-bottleneck with the backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHEMPROP_V2_PATH = PROJECT_ROOT / ".cloned" / "chemprop_v2"
for p in (str(PROJECT_ROOT), str(CHEMPROP_V2_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

from chemprop.nn import BinaryClassificationFFN  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults -- Wf-185 chemprop v2 protocol + tighter per-seed budget.
# ---------------------------------------------------------------------------
DEFAULT_CSV: Path = (
    PROJECT_ROOT / "molmetal" / ".data" / "molmetal" / "drugood_ecfp4_scaffold.csv"
)
DEFAULT_OUT: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf286b_ecfp4_reeval"
DEFAULT_SEEDS: Tuple[int, ...] = (0,)
DEFAULT_EPOCHS: int = 5
DEFAULT_HIDDEN: int = 128
DEFAULT_HEAD_HIDDEN: int = 128
DEFAULT_BATCH: int = 128
DEFAULT_LR: float = 1e-3
DEFAULT_SEED_TIME_BUDGET_S: float = 280.0

# Wf-185 chemprop v2 baseline (lbap_refined_ic50_scaffold split).
WF185_BASELINE_MEAN: float = 0.6707
WF185_BASELINE_STD: float = 0.0042
WF185_BASELINE_SEED_SLACK: float = 0.02  # seed-noise slack for pass-gate.


def _log(msg: str) -> None:
    print(f"[wf286b-ecfp4] {msg}", flush=True)


# ---------------------------------------------------------------------------
# ECFP4 hex-decode (inverse of adapter._arr_to_hex_packed).
# ---------------------------------------------------------------------------
def _hex_to_bit_array(hex_str: str, n_bits: int = 2048) -> np.ndarray:
    """Unpack a low-nibble-first hex string back into a uint8 bit array.

    Mirrors :func:`drugood_ecfp4_scaffold_adapter._arr_to_hex_packed`.  Each
    hex char packs 4 bits (bit j of nibble i is at absolute index 4*i + j).
    """
    out = np.zeros((n_bits,), dtype=np.uint8)
    for i, ch in enumerate(hex_str):
        try:
            nibble = int(ch, 16)
        except ValueError:
            continue
        for j in range(4):
            bit_idx = i * 4 + j
            if bit_idx < n_bits and (nibble & (1 << j)):
                out[bit_idx] = 1
    return out


def _load_ecfp4_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                              Dict[str, int]]:
    """Load the Wf-275B tidy CSV into (X_fp, y, split_idx, split_counts).

    ``split_idx`` maps each row to ``{train:0, val:1, test:2, unset:3}``.
    """
    df = pd.read_csv(csv_path)
    n_rows = len(df)
    _log(f"loaded {n_rows} rows from {csv_path}")

    # Decode the hex-packed ECFP4 bit vectors.
    t0 = time.time()
    X = np.zeros((n_rows, 2048), dtype=np.float32)
    for i, hex_str in enumerate(df["ecfp4_bits"].astype(str).tolist()):
        X[i] = _hex_to_bit_array(hex_str, n_bits=2048).astype(np.float32)
    _log(f"decoded {n_rows} ECFP4 hex vectors in {time.time() - t0:.1f}s")

    # Active label (binary, as in Wf-185 / Wf-263A).
    y = df["active"].astype(int).to_numpy()

    split_map = {"train": 0, "val": 1, "test": 2}
    split_idx = np.array(
        [split_map.get(s, 3) for s in df["split"].astype(str).tolist()],
        dtype=np.int8,
    )
    split_counts = {
        "train": int((split_idx == 0).sum()),
        "val": int((split_idx == 1).sum()),
        "test": int((split_idx == 2).sum()),
    }
    _log(f"split counts: {split_counts}")
    return X, y, split_idx, split_counts


# ---------------------------------------------------------------------------
# Model: chemprop v2 BinaryClassificationFFN on top of ECFP4 bits.
# ---------------------------------------------------------------------------
class _ECFP4FFN(nn.Module):
    """Linear -> chemprop v2 BinaryClassificationFFN on ECFP4 bit vectors."""

    def __init__(self, n_in: int = 2048, head_hidden: int = 128):
        super().__init__()
        # Project raw 2048-bit ECFP4 into a compact embedding so the
        # BinaryClassificationFFN head has something to chew on.
        self.proj = nn.Linear(n_in, head_hidden)
        self.head = BinaryClassificationFFN(
            n_tasks=1,
            input_dim=head_hidden,
            hidden_dim=head_hidden,
            n_layers=2,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chemprop BinaryClassificationFFN.train_step returns (B, 1) pre-sigmoid.
        return self.head.train_step(self.proj(x))


def _train_one_seed(seed: int, X: np.ndarray, y: np.ndarray,
                    split_idx: np.ndarray, args: argparse.Namespace,
                    device: torch.device) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    train_mask = (split_idx == 0)
    val_mask = (split_idx == 1)
    test_mask = (split_idx == 2)
    X_tr = torch.from_numpy(X[train_mask]).to(device)
    y_tr = torch.from_numpy(y[train_mask].astype(np.float32)).to(device)
    X_va = torch.from_numpy(X[val_mask]).to(device)
    y_va = y[val_mask]
    X_te = torch.from_numpy(X[test_mask]).to(device)
    y_te = y[test_mask]
    n_train = int(train_mask.sum())

    model = _ECFP4FFN(n_in=X.shape[1], head_hidden=args.head_hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    perm = np.arange(n_train)
    for epoch in range(1, args.epochs + 1):
        if (time.time() - t0) > args.seed_time_budget_s:
            _log(f"[seed={seed}] seed-budget hit at epoch={epoch - 1}; aborting")
            break
        np.random.shuffle(perm)
        model.train()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch):
            idx = perm[start:start + args.batch]
            xb = X_tr[idx]
            yb = y_tr[idx]
            logit = model(xb).squeeze(-1)
            loss = bce(logit, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            total_loss += float(loss.detach().cpu().item()) * len(idx)
            n_batches += len(idx)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            _log(f"[seed={seed}] epoch={epoch:2d}/{args.epochs}  "
                 f"loss={total_loss / max(n_batches, 1):.4f}  "
                 f"elapsed={time.time() - t0:.1f}s")

    # Eval (BCE head -> sigmoid -> AUC).
    model.eval()
    with torch.no_grad():
        for split_name, X_e, y_e in (
            ("train", X_tr, y[train_mask]),
            ("val", X_va, y_va),
            ("test", X_te, y_te),
        ):
            scores = []
            for start in range(0, X_e.size(0), args.batch):
                logit = model(X_e[start:start + args.batch]).squeeze(-1)
                scores.append(torch.sigmoid(logit).cpu().numpy())
            scores = np.concatenate(scores, axis=0)
            if len(np.unique(y_e)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_e, scores))
            _log(f"[seed={seed}] {split_name}_auc={auc:.4f} n={len(y_e)}")
            if split_name == "train":
                train_auc = auc
            elif split_name == "val":
                val_auc = auc
            else:
                test_auc = auc

    elapsed = time.time() - t0
    return {
        "seed": seed,
        "elapsed_s": float(elapsed),
        "n_train": n_train,
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
    }


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default=str(DEFAULT_CSV),
                   help="Wf-275B ECFP4 scaffold CSV path")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed-time-budget-s", type=float,
                   default=DEFAULT_SEED_TIME_BUDGET_S)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        _log(f"ERROR: csv not found at {csv_path}")
        return 1

    X, y, split_idx, split_counts = _load_ecfp4_csv(csv_path)

    device = torch.device(args.device)
    _log(f"device={device}  seeds={args.seeds}  epochs={args.epochs}  "
         f"head_hidden={args.head_hidden}  batch={args.batch}  lr={args.lr}")

    wall_start = time.time()
    results: List[Dict[str, float]] = []
    for seed in args.seeds:
        try:
            r = _train_one_seed(seed, X, y, split_idx, args, device)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _log(f"[seed={seed}] FAILED: {exc}")
            r = {
                "seed": seed, "elapsed_s": 0.0, "n_train": 0, "n_val": 0,
                "n_test": 0, "train_auc": float("nan"), "val_auc": float("nan"),
                "test_auc": float("nan"), "error": str(exc),
            }
        results.append(r)
        per_seed_path = out_dir / f"seed{seed}.json"
        per_seed_path.write_text(json.dumps(r, indent=2))
        _log(f"[seed={seed}] FINAL  train={r['train_auc']:.4f}  "
             f"val={r['val_auc']:.4f}  test={r['test_auc']:.4f}  "
             f"({r['elapsed_s']:.1f}s)")

    total_wall_min = (time.time() - wall_start) / 60.0

    # Aggregate.
    test_aucs = [r["test_auc"] for r in results if not np.isnan(r["test_auc"])]
    val_aucs = [r["val_auc"] for r in results if not np.isnan(r["val_auc"])]
    train_aucs = [r["train_auc"] for r in results if not np.isnan(r["train_auc"])]

    def _m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def _s(xs):
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    test_mean = _m(test_aucs)
    test_std = _s(test_aucs)
    val_mean = _m(val_aucs)
    val_std = _s(val_aucs)
    train_mean = _m(train_aucs)
    train_std = _s(train_aucs)

    # Pass-gate: ECFP4 + BinaryClassificationFFN ≥ Wf-185 v2 baseline - 0.02.
    passes = bool(test_mean >= WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK)

    aggregate = {
        "wf": "wf286b-ecfp4-direct-reeval",
        "date": time.strftime("%Y-%m-%d"),
        "csv_path": str(csv_path),
        "n_rows": int(len(X)),
        "split_counts": split_counts,
        "featurization": "ECFP4 (Morgan radius=2, n_bits=2048) + chemprop v2 BinaryClassificationFFN",
        "config": {
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "head_hidden": args.head_hidden,
            "batch": args.batch,
            "lr": args.lr,
            "device": str(device),
            "seed_time_budget_s": args.seed_time_budget_s,
        },
        "per_seed": results,
        "train_auc_mean": train_mean, "train_auc_std": train_std,
        "val_auc_mean": val_mean, "val_auc_std": val_std,
        "test_auc_mean": test_mean, "test_auc_std": test_std,
        "wf185_v2_baseline_mean": WF185_BASELINE_MEAN,
        "wf185_v2_baseline_std": WF185_BASELINE_STD,
        "delta_vs_wf185": float(test_mean - WF185_BASELINE_MEAN),
        "passes_wf185_v2_minus_slack": passes,
        "total_wall_min": total_wall_min,
        "verdict": (
            "ECFP4 axis is NOT the bottleneck -- Wf-185 baseline AUC "
            f"recovered within seed-noise slack ({WF185_BASELINE_SEED_SLACK:.2f}); "
            "the Wf-253-D 0.4544 collapse is a backbone issue."
            if passes else
            "ECFP4 axis is a co-bottleneck -- Wf-185 baseline NOT recovered "
            f"(smoke AUC {test_mean:.4f} < {WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK:.4f}); "
            "ECFP4 + scaffold-split featurization alone cannot rescue the "
            "Wf-253-D 0.4544 collapse."
        ),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    _log(f"wrote {out_dir / 'aggregate.json'}")
    _log(f"TEST AUC = {test_mean:.4f} ± {test_std:.4f}  "
         f"vs Wf-185 v2 = {WF185_BASELINE_MEAN:.4f} ± {WF185_BASELINE_STD:.4f}  "
         f"passes={passes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())