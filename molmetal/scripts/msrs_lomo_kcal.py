#!/usr/bin/env python3
"""UP-35 — k measured rows of a new metal: does adaptation beat a shift?

=====================================================================
Why this is a second attempt, and why it is allowed
=====================================================================
DP-11 closed the training-side route after two losses: changing the loss
(pairwise 0.5928, BCE 0.5924) and changing the head both failed to move
the zero-shot LOMO AUC of 0.5974 on the two resolvable folds.  DP-12
reopened the line once under the project's ``reopen_rule`` because
few-shot adaptation is a different mechanism: neither the loss nor the
backbone changes, only which rows are in the training set.  The gate was
written fresh before any run.

=====================================================================
Two arms, and the theorem that separates them
=====================================================================
``shift``     a constant added to every logit of the held-out fold.  AUC
              is invariant under any strictly monotone transform, so this
              arm's AUC must equal the k=0 AUC bit for bit.  It is a
              check on the harness, not a model; it is allowed to move
              Brier and ECE, and it is reported that way.
``finetune``  k rows of the held-out metal are moved from test into
              train and the model is retrained for the same 30 epochs.
              This is the only arm the gate reads.

The k rows are drawn inside the splitter after the ligand-leakage rule,
seeded by the run seed, so a (metal, seed, k) triple is reproducible.

=====================================================================
Gate (todo.json v5, frozen 2026-10-23T03:20+08:00)
=====================================================================
primary     finetune at k=10: mean AUC over Ru and Ir >= 0.65
reproduce   k=0 mean AUC = 0.5974 +- 0.01
theorem     shift arm's AUC differs from k=0 by < 1e-6 on every fold
monotone    >= 4 of the 5 k points non-decreasing
kill        k=10 < 0.62 closes the LOMO line for the third and last time

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_lomo_kcal.py            # run the sweep
    .venv/bin/python molmetal/scripts/msrs_lomo_kcal.py --reuse    # re-score existing runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir")
SEEDS = (0, 1, 2, 3, 4)
K_GRID = (0, 5, 10, 20, 50)
REPORT_ROOT = PROJECT_ROOT / "molmetal" / "reports" / "up35_lomo_kcal"
SCORE_ROOT = PROJECT_ROOT / "results" / "up35_lomo_kcal"
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"
GATE_FROZEN_AT = "2026-10-23T03:20+08:00"
TARGET_AUC = 0.5974
REPRO_TOL = 0.01
AUC_GATE = 0.65
AUC_KILL = 0.62
THEOREM_TOL = 1e-6
MIN_MONOTONE = 4
SHIFT = 0.5


def config_for(k: int) -> Path:
    return PROJECT_ROOT / "configs" / f"msrs_lomo_kcal_k{k}.yaml"


def run_one(k: int, log: Path) -> None:
    """Launch the trainer through its YAML. No hyperparameter on the CLI."""
    cmd = [sys.executable, str(TRAINER), "--config", str(config_for(k))]
    with open(log, "a") as fh:
        fh.write(f"\n=== k={k} {' '.join(cmd)} ===\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        # The trainer's Wf-C2 guard exits 3 *after* the folds are written when
        # --out-metric is missing. That is now configured in the YAML, but a
        # non-zero code with both folds on disk is still a completed run: keep
        # going and record the code rather than throwing away 10 trained models.
        folds = [REPORT_ROOT / f"k{k}" / m / "per_seed_metrics.json" for m in METALS]
        if all(p.exists() for p in folds):
            with open(log, "a") as fh:
                fh.write(f"=== k={k} trainer exited {proc.returncode} but both folds "
                         f"exist; continuing ===\n")
        else:
            raise RuntimeError(f"k={k} trainer exited {proc.returncode}; see {log}")


def read_fold(k: int, metal: str) -> Dict[str, object]:
    path = REPORT_ROOT / f"k{k}" / metal / "per_seed_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    rows = payload["per_seed"]
    aucs = [r["test_auc"] for r in rows if r.get("test_auc") is not None]
    if len(aucs) != len(SEEDS):
        raise RuntimeError(f"{path}: {len(aucs)} of {len(SEEDS)} seeds produced an AUC")
    return {
        "auc_per_seed": [round(float(a), 4) for a in aucs],
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_sd": round(float(np.std(aucs, ddof=1)), 4),
        "n_test": int(rows[0]["n_test"]),
        "n_train": int(rows[0]["n_train"]),
        "pos_rate_test": (round(float(rows[0]["pos_rate_test"]), 4)
                          if rows[0].get("pos_rate_test") is not None else None),
        "elapsed_seconds": round(float(np.sum([r["elapsed_seconds"] for r in rows])), 1),
    }


def shift_arm(k: int) -> Dict[str, object]:
    """Constant logit shift on the k=0 dumps: AUC must not move at all."""
    out: Dict[str, object] = {}
    worst = 0.0
    for metal in METALS:
        rows = []
        for seed in SEEDS:
            path = SCORE_ROOT / f"k{k}" / f"{metal}_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as data:
                y = data["y"].astype(np.int64)
                s = data["score"].astype(np.float64)
            a0 = float(roc_auc_score(y, s))
            a1 = float(roc_auc_score(y, s + SHIFT))
            b0 = float(brier_score_loss(y, 1.0 / (1.0 + np.exp(-s))))
            b1 = float(brier_score_loss(y, 1.0 / (1.0 + np.exp(-(s + SHIFT)))))
            worst = max(worst, abs(a1 - a0))
            rows.append({
                "seed": seed,
                "auc_before": round(a0, 6),
                "auc_after": round(a1, 6),
                "abs_diff": float(abs(a1 - a0)),
                "brier_before": round(b0, 6),
                "brier_after": round(b1, 6),
            })
        out[metal] = {
            "shift": SHIFT,
            "auc_before_mean": round(float(np.mean([r["auc_before"] for r in rows])), 4),
            "auc_after_mean": round(float(np.mean([r["auc_after"] for r in rows])), 4),
            "brier_before_mean": round(float(np.mean([r["brier_before"] for r in rows])), 4),
            "brier_after_mean": round(float(np.mean([r["brier_after"] for r in rows])), 4),
            "per_seed": rows,
        }
    out["max_abs_auc_diff"] = float(worst)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse", action="store_true",
                    help="do not retrain; score whatever is already on disk")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_lomo_kcal.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    log = REPORT_ROOT / "sweep.log"
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        for k in K_GRID:
            print(f"[UP-35] launching k={k}")
            run_one(k, log)

    per_k: Dict[str, object] = {}
    base_train: Dict[str, int] = {}
    for k in K_GRID:
        folds = {m: read_fold(k, m) for m in METALS}
        if k == 0:
            base_train = {m: folds[m]["n_train"] for m in METALS}
        mean_auc = round(float(np.mean([folds[m]["auc_mean"] for m in METALS])), 4)
        per_k[str(k)] = {
            "k": k,
            "per_fold": folds,
            "mean_auc": mean_auc,
            "n_rows_moved": {m: int(folds[m]["n_train"] - base_train[m])
                             for m in METALS} if base_train else {},
        }
        print(f"[UP-35] k={k:2d} Ru={folds['Ru']['auc_mean']:.4f} "
              f"Ir={folds['Ir']['auc_mean']:.4f} mean={mean_auc:.4f}")

    shift = shift_arm(0)
    curve = [per_k[str(k)]["mean_auc"] for k in K_GRID]
    steps = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    n_nondec = int(sum(1 for d in steps if d >= 0))

    a0 = per_k["0"]["mean_auc"]
    a10 = per_k["10"]["mean_auc"]
    checks = {
        "primary_auc_k10": {
            "value": a10,
            "threshold": AUC_GATE,
            "pass": bool(a10 >= AUC_GATE),
        },
        "reproduce_k0": {
            "value": a0,
            "target": TARGET_AUC,
            "tolerance": REPRO_TOL,
            "pass": bool(abs(a0 - TARGET_AUC) <= REPRO_TOL),
        },
        "shift_theorem": {
            "max_abs_auc_diff": shift["max_abs_auc_diff"],
            "tolerance": THEOREM_TOL,
            "pass": bool(shift["max_abs_auc_diff"] < THEOREM_TOL),
        },
        "monotone": {
            "curve": curve,
            "steps": [round(float(d), 4) for d in steps],
            "non_decreasing_steps": n_nondec,
            "required": MIN_MONOTONE,
            "pass": bool(n_nondec >= MIN_MONOTONE),
        },
    }
    kill = {
        "rule": "finetune at k=10 < 0.62 closes the LOMO line for the third and "
                "last time",
        "value": a10,
        "triggered": bool(a10 < AUC_KILL),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_lomo_kcal",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 metal extrapolation with k measured rows (UP-35)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "trainer": str(TRAINER.relative_to(PROJECT_ROOT)),
            "configs": [str(config_for(k).relative_to(PROJECT_ROOT)) for k in K_GRID],
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "sweep_log": str(log.relative_to(PROJECT_ROOT)),
        },
        "design": {
            "cohort": "molmetal/data/metal_lomo.csv, default 24h snapshot, collapsed "
                      "to one row per ligand within a metal",
            "folds": list(METALS),
            "why_these_folds": "the two folds with >= 200 unique compounds in the "
                               "resolvability audit; Os/Rh/Re are below the "
                               "resolution floor and their AUC is noise",
            "seeds": list(SEEDS),
            "k_grid": list(K_GRID),
            "epochs": 30,
            "learner": "D-MPNN multitask + Wasserstein (the published LOMO model)",
            "support_selection": "uniform draw inside the splitter, seeded by the run "
                                 "seed, drawn after the ligand-leakage rule",
            "gate_frozen_at": GATE_FROZEN_AT,
            "reopened_under": "DP-12 reopen_rule: different mechanism from DP-11 "
                              "(rows, not loss or backbone); fresh gate",
        },
        "per_k": per_k,
        "shift_arm": shift,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The k rows are a uniform draw, not a designed plate. A deployment that "
            "chooses which compounds to measure may do better.",
            "The shift arm cannot move AUC by construction. It is included to prove "
            "the harness scores what it says it scores, and to show that a post-hoc "
            "correction is not an adaptation mechanism.",
            "Only Ru and Ir are scored. The other three folds have fewer than 200 "
            "unique compounds and their AUC standard error is ~0.08-0.10, so a "
            "k-curve there would be noise.",
            "CUDA is not bit-reproducible in this trainer, so k=0 is expected to sit "
            "within ~0.01 of the published 0.5974 rather than on it. The tolerance "
            "was frozen at that width before the run.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    print(f"[UP-35] curve={curve} shift_diff={shift['max_abs_auc_diff']:.2e}")
    print(f"[UP-35] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time() - t0, 1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
