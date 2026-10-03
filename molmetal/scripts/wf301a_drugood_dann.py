"""Wf-301A: DrugOOD scaffold-split DANN re-eval to lift Wf-253-D 0.4544 collapse.

==============================================================================
Wf-301A (TODO-130 G5 follow-up of Wf-253-D / Wf-286B):

Per Wf-253-D, chemprop v3 forward-pass on DrugOOD scaffold ood_test split
yielded aggregate AUC = 0.4544 (BELOW_RANDOM).  Per Wf-286B, ECFP4 alone
recovers AUC ≈ 0.8820 on the Wf-275B tidy CSV (single seed smoke).  The
ECFP4 axis is therefore NOT the bottleneck; the bottleneck is the MPNN
backbone axis or the v3-fallback config path.

Wf-301A hypothesis
------------------
Add DANN (Domain-Adversarial Neural Network, Ganin et al. 2016) gradient-
reversal adversarial head to the chemprop v2 BinaryClassificationFFN
training, with the per-row **scaffold Bemis-Murcko bin** as the domain
label (DrugOOD core has no per-row metal annotation, so the scaffold bin
is the only sensible adversarial domain signal).

If DANN improves test_auc above the Wf-286B baseline (and at minimum above
the field-avg 0.55-0.65 + Wf-185 v2 baseline 0.6707), then the Wf-253-D
0.4544 collapse is addressable with a domain-invariant ECFP4 backbone +
DANN scaffold-binning.  Otherwise, the bottleneck moves to the v3
backbone axis (already excluded per Wf-286B + Wf-285B).

Inputs
------
* molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv (Wf-275B adapter
  output, 95,236 rows, 0 leak, Bemis-Murcko scaffold split
  precomputed).

Outputs
-------
* molmetal/reports/wf301a_drugood_dann/aggregate.json
* molmetal/reports/wf301a_drugood_dann/seed<N>.json  (per-seed)
* molmetal/reports/wf301a_drugood_dann/dann_report.md  (this script's
  primary deliverable)

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* 3-seed × 5-epoch SMOKE run; full retrain (5-seed × 20-ep) is a follow-up
  if smoke clears the field-avg + Wf-185 v2 baseline.
* Domain label = scaffold bin (top-K most-frequent Bemis-Murcko scaffolds,
  remainder collapsed to a single OOV bin).  Different from a per-metal
  DANN domain (Wf-285B), but the only honest adversarial signal for the
  single-assay DrugOOD core subset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
# Defaults — Wf-185 chemprop v2 protocol + 5-seed smoke budget.
# ---------------------------------------------------------------------------
DEFAULT_CSV: Path = (
    PROJECT_ROOT / "molmetal" / ".data" / "molmetal" / "drugood_ecfp4_scaffold.csv"
)
DEFAULT_OUT: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf301a_drugood_dann"
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EPOCHS: int = 5
DEFAULT_HIDDEN: int = 300
DEFAULT_DEPTH: int = 3
DEFAULT_HEAD_HIDDEN: int = 300
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_SEED_TIME_BUDGET_S: float = 600.0
DEFAULT_DANN_LAMBDA: float = 1.0
DEFAULT_DANN_N_DOMAINS: int = 16  # top-15 scaffold bins + 1 OOV bin
DEFAULT_USE_DANN: bool = True

# Wf-185 chemprop v2 baseline (lbap_refined_ic50_scaffold split).
WF185_BASELINE_MEAN: float = 0.6707
WF185_BASELINE_STD: float = 0.0042
WF185_BASELINE_SEED_SLACK: float = 0.02
# Field-avg target per TODO-130 G5 honest reporting.
FIELD_AVG_TARGET_LO: float = 0.55
FIELD_AVG_TARGET_HI: float = 0.65


def _log(msg: str) -> None:
    print(f"[wf301a-dann] {msg}", flush=True)


# ---------------------------------------------------------------------------
# ECFP4 hex-decode (inverse of Wf-275B adapter._arr_to_hex_packed).
# ---------------------------------------------------------------------------
def _hex_to_bit_array(hex_str: str, n_bits: int = 2048) -> np.ndarray:
    """Unpack a low-nibble-first hex string back into a uint8 bit array."""
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


def _load_ecfp4_csv(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Load the Wf-275B tidy CSV.

    Returns
    -------
    X : (N, 2048) float32 ECFP4 bit vectors
    y : (N,) int8 active labels
    split_idx : (N,) int8 {0=train, 1=val, 2=test}
    scaffold_id : (N,) int64 scaffold bin id (top-K + OOV = K+1 bins)
    split_counts : dict
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

    # Scaffold bin id (top-K + OOV).  The scaffold column is a hex SMILES
    # string; bin by frequency.
    scaffold_strs = df["scaffold"].astype(str).tolist()
    counts = Counter(scaffold_strs)
    top_k = DEFAULT_DANN_N_DOMAINS - 1  # 15 most-frequent scaffolds + OOV
    top_scaffolds = set(s for s, _ in counts.most_common(top_k))
    scaffold_id = np.zeros(n_rows, dtype=np.int64)
    for i, s in enumerate(scaffold_strs):
        scaffold_id[i] = (
            (list(top_scaffolds).index(s) if s in top_scaffolds else top_k)
            if s in top_scaffolds
            else top_k
        )
    n_unique = len(set(scaffold_id.tolist()))
    _log(f"scaffold binning: top-{top_k} + OOV = {n_unique} bins")
    return X, y, split_idx, scaffold_id, split_counts


# ---------------------------------------------------------------------------
# Gradient reversal layer (Ganin et al. 2016 DANN).
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambda_ * g, None


def grad_reverse(x, lambda_=1.0):
    return _GradReverse.apply(x, lambda_)


# ---------------------------------------------------------------------------
# Model: ECFP4 -> Linear -> BinaryClassificationFFN (+ optional DANN head).
# ---------------------------------------------------------------------------
class _ECFP4DANN(nn.Module):
    """ECFP4 -> Linear -> BinaryClassificationFFN + DANN domain head.

    DANN head: gradient-reversed linear -> [n_domains] classifier.
    """

    def __init__(
        self,
        n_in: int = 2048,
        head_hidden: int = 300,
        n_domains: int = 16,
        use_dann: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(n_in, head_hidden)
        self.head = BinaryClassificationFFN(
            n_tasks=1,
            input_dim=head_hidden,
            hidden_dim=head_hidden,
            n_layers=2,
            dropout=0.0,
        )
        self.use_dann = use_dann
        if use_dann:
            self.domain_head = nn.Linear(head_hidden, n_domains)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x)
        logit = self.head.train_step(z).squeeze(-1)
        if self.use_dann:
            z_rev = grad_reverse(z, lambda_=1.0)
            domain_logits = self.domain_head(z_rev)
        else:
            domain_logits = torch.zeros(x.size(0), 1, device=x.device)
        return logit, domain_logits


def _train_one_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    split_idx: np.ndarray,
    scaffold_id: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    train_mask = split_idx == 0
    val_mask = split_idx == 1
    test_mask = split_idx == 2
    X_tr = torch.from_numpy(X[train_mask]).to(device)
    y_tr = torch.from_numpy(y[train_mask].astype(np.float32)).to(device)
    s_tr = torch.from_numpy(scaffold_id[train_mask]).to(device)
    X_va = torch.from_numpy(X[val_mask]).to(device)
    y_va = y[val_mask]
    X_te = torch.from_numpy(X[test_mask]).to(device)
    y_te = y[test_mask]
    n_train = int(train_mask.sum())

    n_domains = int(scaffold_id.max() + 1)
    model = _ECFP4DANN(
        n_in=X.shape[1],
        head_hidden=args.head_hidden,
        n_domains=n_domains,
        use_dann=args.use_dann,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

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
            idx = perm[start : start + args.batch]
            xb = X_tr[idx]
            yb = y_tr[idx]
            sb = s_tr[idx]
            logit, domain_logits = model(xb)
            loss = bce(logit, yb)
            if args.use_dann:
                # DANN loss with GRL applied inside the model (lambda=1.0).
                loss = loss + args.dann_lambda * ce(domain_logits, sb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            total_loss += float(loss.detach().cpu().item()) * len(idx)
            n_batches += len(idx)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            _log(
                f"[seed={seed}] epoch={epoch:2d}/{args.epochs}  "
                f"loss={total_loss / max(n_batches, 1):.4f}  "
                f"elapsed={time.time() - t0:.1f}s"
            )

    # Eval.
    model.eval()
    out: Dict[str, float] = {"seed": seed}
    with torch.no_grad():
        for split_name, X_e, y_e in (
            ("train", X_tr, y[train_mask]),
            ("val", X_va, y_va),
            ("test", X_te, y_te),
        ):
            scores = []
            for start in range(0, X_e.size(0), args.batch):
                logit, _ = model(X_e[start : start + args.batch])
                scores.append(torch.sigmoid(logit).cpu().numpy())
            scores = np.concatenate(scores, axis=0)
            if len(np.unique(y_e)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_e, scores))
            out[f"{split_name}_auc"] = float(auc)
            _log(f"[seed={seed}] {split_name}_auc={auc:.4f} n={len(y_e)}")

    out.update(
        {
            "elapsed_s": float(time.time() - t0),
            "n_train": n_train,
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "use_dann": bool(args.use_dann),
            "n_domains": n_domains,
        }
    )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--seed-time-budget-s", type=float, default=DEFAULT_SEED_TIME_BUDGET_S
    )
    p.add_argument(
        "--use-dann", dest="use_dann", action="store_true", default=DEFAULT_USE_DANN
    )
    p.add_argument("--no-dann", dest="use_dann", action="store_false")
    p.add_argument("--dann-lambda", type=float, default=DEFAULT_DANN_LAMBDA)
    p.add_argument(
        "--data-csv", default=None, help="Alias for --csv (Wf-301A spec compat)."
    )
    p.add_argument(
        "--chemprop-mode",
        default="v3-fallback",
        help="Cosmetic only: v3-fallback is the canonical mode "
        "for chemprop v2 BinaryClassificationFFN (matches "
        "Wf-263A-Audit verdict).",
    )
    args = p.parse_args(argv)

    # Handle --data-csv alias.
    csv_arg = args.data_csv if args.data_csv else args.csv
    csv_path = Path(csv_arg)
    if not csv_path.exists():
        _log(f"ERROR: csv not found at {csv_path}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, split_idx, scaffold_id, split_counts = _load_ecfp4_csv(csv_path)
    device = torch.device(args.device)
    _log(
        f"device={device}  seeds={args.seeds}  epochs={args.epochs}  "
        f"head_hidden={args.head_hidden}  batch={args.batch}  "
        f"lr={args.lr}  use_dann={args.use_dann}  "
        f"chemprop_mode={args.chemprop_mode}"
    )

    wall_start = time.time()
    results: List[Dict[str, float]] = []
    for seed in args.seeds:
        try:
            r = _train_one_seed(seed, X, y, split_idx, scaffold_id, args, device)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _log(f"[seed={seed}] FAILED: {exc}")
            r = {"seed": seed, "error": str(exc)}
        results.append(r)
        per_seed_path = out_dir / f"seed{seed}.json"
        per_seed_path.write_text(json.dumps(r, indent=2))
        if "test_auc" in r:
            _log(
                f"[seed={seed}] FINAL  train={r['train_auc']:.4f}  "
                f"val={r['val_auc']:.4f}  test={r['test_auc']:.4f}  "
                f"({r['elapsed_s']:.1f}s)"
            )

    total_wall_min = (time.time() - wall_start) / 60.0

    # Aggregate.
    test_aucs = [
        r["test_auc"]
        for r in results
        if "test_auc" in r and not np.isnan(r["test_auc"])
    ]
    val_aucs = [
        r["val_auc"] for r in results if "val_auc" in r and not np.isnan(r["val_auc"])
    ]
    train_aucs = [
        r["train_auc"]
        for r in results
        if "train_auc" in r and not np.isnan(r["train_auc"])
    ]

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

    # Pass-gates.
    passes_field_avg = bool(test_mean >= FIELD_AVG_TARGET_LO)
    passes_wf185_minus_slack = bool(
        test_mean >= WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK
    )
    passes_wf185 = bool(test_mean >= WF185_BASELINE_MEAN)

    if passes_wf185:
        verdict = (
            f"PASS_WF185 -- DANN {test_mean:.4f} >= Wf-185 baseline "
            f"{WF185_BASELINE_MEAN:.4f}; the Wf-253-D 0.4544 collapse is "
            "addressable by ECFP4 + scaffold-bin DANN."
        )
    elif passes_wf185_minus_slack:
        verdict = (
            f"PASS_FIELD_AVG -- DANN {test_mean:.4f} >= field-avg lo "
            f"{FIELD_AVG_TARGET_LO:.2f} and within seed-noise slack of "
            f"Wf-185 v2 ({WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK:.4f}); "
            "the collapse is partially addressable."
        )
    else:
        verdict = (
            f"HONEST_NEGATIVE -- DANN {test_mean:.4f} < field-avg lo "
            f"{FIELD_AVG_TARGET_LO:.2f}; DANN scaffold-binning does NOT "
            "rescue the Wf-253-D 0.4544 collapse; backbone axis (or "
            "v3-fallback config path) remains the bottleneck."
        )

    aggregate = {
        "wf": "wf301a-drugood-dann",
        "date": time.strftime("%Y-%m-%d"),
        "csv_path": str(csv_path),
        "n_rows": int(len(X)),
        "split_counts": split_counts,
        "featurization": "ECFP4 (Morgan radius=2, n_bits=2048) + chemprop v2 BinaryClassificationFFN + DANN scaffold-bin adversarial head",
        "config": {
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "hidden": args.hidden,
            "depth": args.depth,
            "head_hidden": args.head_hidden,
            "batch": args.batch,
            "lr": args.lr,
            "device": str(device),
            "seed_time_budget_s": args.seed_time_budget_s,
            "use_dann": bool(args.use_dann),
            "dann_lambda": float(args.dann_lambda),
            "n_domains": int(scaffold_id.max() + 1),
            "chemprop_mode": str(args.chemprop_mode),
        },
        "per_seed": results,
        "n_seeds_completed": int(len(test_aucs)),
        "train_auc_mean": train_mean,
        "train_auc_std": train_std,
        "val_auc_mean": val_mean,
        "val_auc_std": val_std,
        "test_auc_mean": test_mean,
        "test_auc_std": test_std,
        "wf185_v2_baseline_mean": WF185_BASELINE_MEAN,
        "wf185_v2_baseline_std": WF185_BASELINE_STD,
        "field_avg_target_lo": FIELD_AVG_TARGET_LO,
        "field_avg_target_hi": FIELD_AVG_TARGET_HI,
        "delta_vs_wf185": float(test_mean - WF185_BASELINE_MEAN),
        "passes_field_avg_target": passes_field_avg,
        "passes_wf185_v2_minus_slack": passes_wf185_minus_slack,
        "passes_wf185_v2": passes_wf185,
        "total_wall_min": total_wall_min,
        "verdict": verdict,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    _log(f"wrote {out_dir / 'aggregate.json'}")
    _log(
        f"TEST AUC = {test_mean:.4f} ± {test_std:.4f}  "
        f"vs Wf-185 v2 = {WF185_BASELINE_MEAN:.4f} ± {WF185_BASELINE_STD:.4f}  "
        f"verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
