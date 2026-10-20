#!/usr/bin/env python3
"""C6 — does averaging the five seeds' test scores clear the 0.6 LOMO target?

The published LOMO number is the mean of per-seed AUCs: five independent AUCs
averaged after the fact.  That is not what a deployed ensemble does.  A seed
ensemble averages the *scores* and then computes one AUC, and the two differ
whenever the seeds' rankings disagree — averaging AUCs throws away the fact
that four seeds can outvote one.

This script scores both, from the per-row dumps written by
``train_metal_lomo_wasserstein.py --dump-test-scores-dir``:

* ``mean_of_seed_aucs`` — the published method, recomputed from the dumps as a
  consistency check against ``metal_lomo_auc_wasserstein.json`` (a small gap is
  expected: the trainer is not bit-reproducible on CUDA);
* ``auc_of_mean_score`` — the ensemble.

Both are reported against the 0.6 target, and both for the resolvable folds
(``n_unique >= MIN_UNIQUE_FOR_GATE``) as well as all five.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/lomo_seed_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from molmetal.scripts.lomo_resolvability_audit import hanley_se  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Os", "Rh", "Re")
TARGET = 0.6


def load_fold(directory: Path, metal: str) -> Dict[str, np.ndarray]:
    paths = sorted(directory.glob(f"{metal}_seed*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {metal}_seed*.npz under {directory}")
    ys, scores = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            ys.append(data["y"].astype(np.int64))
            scores.append(data["score"].astype(np.float64))
    first = ys[0]
    for path, y in zip(paths[1:], ys[1:]):
        if not np.array_equal(first, y):
            raise ValueError(f"{path}: test labels differ from {paths[0]} (row order?)")
    return {"y": first, "scores": np.column_stack(scores), "paths": [p.name for p in paths]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lomo_seed_ensemble",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "lomo_resolvability_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "lomo_seed_ensemble.json",
    )
    args = parser.parse_args(argv)

    audit = json.loads(args.audit.read_text())
    by_metal = {f["metal"]: f for f in audit["folds"]}

    folds: List[Dict[str, object]] = []
    for metal in METALS:
        fold = load_fold(args.dir, metal)
        y, scores = fold["y"], fold["scores"]
        # The AUC is computed on the DUMPED rows, so the binomial n that sets
        # the Hanley-McNeil standard error is the dumped row count, not the
        # ligand-collapsed count in the audit.  The collapsed count stays in the
        # record as the *effective-n* criterion for resolvability.
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        per_seed = [float(roc_auc_score(y, scores[:, j])) for j in range(scores.shape[1])]
        mean_score = scores.mean(axis=1)
        ensemble = float(roc_auc_score(y, mean_score))
        se = hanley_se(ensemble, n_pos, n_neg)
        ref = by_metal.get(metal, {})
        folds.append(
            {
                "metal": metal,
                "n_test_rows": int(len(y)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_unique_ligands": int(ref.get("n_unique", -1)),
                "n_seeds": int(scores.shape[1]),
                "mean_of_seed_aucs": round(float(np.mean(per_seed)), 4),
                "seed_sd": round(float(np.std(per_seed, ddof=1)), 4),
                "auc_of_mean_score": round(ensemble, 4),
                "ensemble_gain": round(ensemble - float(np.mean(per_seed)), 4),
                "hanley_se": round(se, 4),
                "ci95_hanley": [round(ensemble - 1.96 * se, 4), round(ensemble + 1.96 * se, 4)],
                "published_auc_mean": ref.get("auc_mean"),
                "reproduction_gap": round(float(np.mean(per_seed)) - ref["auc_mean"], 4)
                if ref.get("auc_mean") is not None
                else None,
                "resolvable": bool(ref.get("resolvable", False)),
                "files": fold["paths"],
            }
        )

    def aggregate(key: str, only_resolvable: bool) -> float:
        rows = [f for f in folds if (f["resolvable"] or not only_resolvable)]
        return round(float(np.mean([f[key] for f in rows])), 4)

    aggregates = {
        "all_metals": {
            "mean_of_seed_aucs": aggregate("mean_of_seed_aucs", False),
            "auc_of_mean_score": aggregate("auc_of_mean_score", False),
        },
        "resolvable_only": {
            "mean_of_seed_aucs": aggregate("mean_of_seed_aucs", True),
            "auc_of_mean_score": aggregate("auc_of_mean_score", True),
            "metals": [f["metal"] for f in folds if f["resolvable"]],
        },
    }

    published = audit["aggregates"]["all_metals"]["value"]
    reproduced = aggregates["all_metals"]["mean_of_seed_aucs"]
    payload = {
        "artifact": "lomo_seed_ensemble",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/lomo_seed_ensemble.py",
            "git_commit": _git_commit(),
            "score_dumps": str(args.dir.relative_to(PROJECT_ROOT)) if args.dir.is_relative_to(PROJECT_ROOT) else str(args.dir),
            "source_artifact": "metrics/by_metric/lomo_resolvability_audit.json",
        },
        "design": {
            "target_overall_auc": TARGET,
            "mean_of_seed_aucs": "published method: AUC per seed, then arithmetic mean",
            "auc_of_mean_score": "seed ensemble: average the scores, then one AUC",
            "resolvability_rule": audit["design"]["rule"],
            "min_unique_for_gate": audit["design"]["min_unique_for_gate"],
        },
        "folds": folds,
        "aggregates": aggregates,
        "consistency": {
            "published_all_metals": published,
            "reproduced_all_metals": reproduced,
            "abs_diff": round(abs(published - reproduced), 4),
            "per_fold_gap": {f["metal"]: f["reproduction_gap"] for f in folds},
            "note": (
                "A gap is expected: the trainer runs on CUDA without a fixed "
                "algorithm seed, so a fold's AUC moves by ~1e-2 between runs "
                "(Re, n=74, moved 0.085). A gap is NOT evidence that the dump "
                "covers a different cohort; that would need a fold-size check, "
                "and the dumped row counts are recorded above."
            ),
            "max_abs_gap": round(
                max(abs(f["reproduction_gap"] or 0.0) for f in folds), 4
            ),
        },
        "verdict": {
            "target_met_all_metals": bool(aggregates["all_metals"]["auc_of_mean_score"] >= TARGET),
            "target_met_resolvable_only": bool(aggregates["resolvable_only"]["auc_of_mean_score"] >= TARGET),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for f in folds:
        print(
            f"[c6b] {f['metal']:2s} seed-mean={f['mean_of_seed_aucs']:.4f} "
            f"ensemble={f['auc_of_mean_score']:.4f} (+{f['ensemble_gain']:+.4f}) "
            f"ci={f['ci95_hanley']} resolvable={f['resolvable']}"
        )
    print(
        f"[c6b] all={aggregates['all_metals']['auc_of_mean_score']} "
        f"resolvable={aggregates['resolvable_only']['auc_of_mean_score']} target={TARGET}"
    )
    print(f"[c6b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
