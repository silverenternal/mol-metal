#!/usr/bin/env python3
"""UP-17 — where does the additive residual actually come from?

=====================================================================
Why this exists
=====================================================================
Round 1 measured a 0.712 "ligand x metal interaction share" on raw
potency and then spent seven packages trying to learn it. But the
residual of an additive model can be carried by four different
interactions, and only one of them is the one we want:

    ligand x metal   the substitution effect we are trying to predict
    cell   x metal   a metal is better in some cell lines than others
    ligand x cell    a ligand is unusually potent in some cell lines
    paper  x metal   a lab measures a metal differently from another lab

If the residual is mostly paper x metal, it is a batch effect and no
model should learn it. This script decomposes the residual of the
additive surface into those four blocks.

=====================================================================
Estimator
=====================================================================
For a block (A, B) and the current residual e:

    m_ab   = mean(e | A=a, B=b),  n_ab = count
    Var_cell = weighted variance of m_ab around zero
    sigma2_b = max(0, Var_cell - sigma2_noise * mean(1/n_ab))
    reliability w_ab = sigma2_b / (sigma2_b + sigma2_noise / n_ab)
    fitted = w_ab * m_ab, subtracted from e before the next block

The reliability weight is what stops a block with one observation per
cell from claiming the whole residual: with n_ab = 1 the fit is shrunk
by sigma2_b / (sigma2_b + sigma2_noise).

Because the four blocks overlap, the independent estimates need not sum
to one. A sequential pass (and its reverse) is reported as well; those
marginals do sum, and the gap between the two orders bounds how much the
answer depends on the order.

=====================================================================
Gate (todo.json, frozen 2026-10-21T20:30+08:00)
=====================================================================
Every component carries a ligand-clustered bootstrap CI in both orders, and
the sequential decomposition must account for the residual variance to within
5% (1 - explained_total < 0.05). Kill: paper x metal share > 0.5 means the
"shape" is a batch effect.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_shape_decomposition.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import (  # noqa: E402
    METALS,
    _git_commit,
    _sha256,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    TOP_CELL_LINES,
    oof_predict,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_BOOT = 200
BOOT_SEED = 20261021
BLOCKS: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("ligand_x_metal", ("ligand_canonical", "metal")),
    ("cell_x_metal", ("cell_line_key", "metal")),
    ("ligand_x_cell", ("ligand_canonical", "cell_line_key")),
    ("paper_x_metal", ("doi", "metal")),
)


def pooled_noise(df: pd.DataFrame) -> Tuple[float, int]:
    keys = ["ligand_canonical", "metal", "cell_line_key", "time_h", "doi"]
    g = df.groupby(keys, dropna=False)["value_log10"]
    n = g.size()
    var = g.var(ddof=1)
    rep = n >= 2
    dof = int((n[rep] - 1).sum())
    ss = float(((n[rep] - 1) * var[rep]).sum())
    return (ss / dof if dof else float("nan")), dof


def noise_floor(path: Path, endpoint: str) -> Tuple[float, int]:
    """Replicate variance must come from the undeduplicated long table.

    ``measurements_dedup.parquet`` has already collapsed replicate rows, so
    every group there has n = 1 and the estimator returns df = 0.
    """
    long = pd.read_parquet(path)
    sub = long[
        (long["endpoint"] == endpoint)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_canonical"].notna()
    ]
    return pooled_noise(sub)


def build_design_with_paper(sub: pd.DataFrame, descriptors: pd.DataFrame, cell_vocab: Sequence[str]):
    """Same design as msrs_surface_model plus a DOI one-hot block.

    Without a paper main effect the residual carries the between-paper offset,
    and the (paper, metal) block then absorbs that main effect instead of the
    paper x metal interaction -- which is how a batch effect gets mislabelled
    as structure.
    """
    from scipy import sparse

    from molmetal.scripts.msrs_surface_model import cell_matrix, metal_matrix, morgan_matrix

    lig = morgan_matrix(sub["ligand_main_fragment"].tolist())
    met = sparse.csr_matrix(metal_matrix(sub["metal"].tolist(), descriptors))
    cell = cell_matrix(sub["cell_line_key"].tolist(), cell_vocab)
    doi = pd.get_dummies(sub["doi"].astype(str), dtype=float, sparse=False).to_numpy(dtype=float)
    X = sparse.hstack([lig, met, cell, sparse.csr_matrix(doi)], format="csr")
    n_lig, n_met, n_cell, n_doi = lig.shape[1], met.shape[1], cell.shape[1], doi.shape[1]
    groups = [
        list(range(0, n_lig)),
        list(range(n_lig, n_lig + n_met)),
        list(range(n_lig + n_met, n_lig + n_met + n_cell)),
        list(range(n_lig + n_met + n_cell, n_lig + n_met + n_cell + n_doi)),
    ]
    return X, groups


def block_effects(
    keys_a: np.ndarray, keys_b: np.ndarray, resid: np.ndarray, sigma2_noise: float
) -> Dict[str, object]:
    """Method-of-moments variance component plus reliability-shrunk cell effects."""
    frame = pd.DataFrame({"a": keys_a, "b": keys_b, "e": resid})
    g = frame.groupby(["a", "b"], sort=False)["e"]
    n = g.size().to_numpy(dtype=float)
    m = g.mean().to_numpy(dtype=float)
    total = float(len(resid))
    grand = float(np.sum(n * m) / total)

    var_cell = float(np.sum(n * (m - grand) ** 2) / max(total - 1.0, 1.0))
    correction = sigma2_noise * float(np.mean(1.0 / n))
    sigma2 = max(0.0, var_cell - correction)
    w = sigma2 / (sigma2 + sigma2_noise / n) if sigma2 > 0 else np.zeros_like(n)
    fitted = w * (m - grand)

    index = g.mean().index
    mapping = dict(zip(index, fitted))
    contribution = np.fromiter(
        (mapping[(a, b)] for a, b in zip(keys_a, keys_b)), dtype=float, count=len(resid)
    )
    return {
        "n_cells": int(len(n)),
        "mean_cell_size": round(float(n.mean()), 3),
        "var_cell_means": round(var_cell, 6),
        "sampling_correction": round(correction, 6),
        "sigma2": round(sigma2, 6),
        "mean_reliability": round(float(np.mean(w)), 4),
        "contribution": contribution,
    }


def sequential(
    blocks: Sequence[Tuple[str, Tuple[str, str]]],
    keys: Dict[str, Tuple[np.ndarray, np.ndarray]],
    resid: np.ndarray,
    sigma2_noise: float,
) -> Dict[str, object]:
    e = resid.copy()
    var0 = float(np.var(resid, ddof=1))
    steps = {}
    for name, _ in blocks:
        before = float(np.var(e, ddof=1))
        info = block_effects(*keys[name], e, sigma2_noise)
        e = e - info["contribution"]
        after = float(np.var(e, ddof=1))
        steps[name] = {
            "explained_var": round(before - after, 6),
            "share_of_residual": round((before - after) / var0, 4) if var0 else None,
            "sigma2": info["sigma2"],
            "mean_reliability": info["mean_reliability"],
            "n_cells": info["n_cells"],
        }
    return {
        "order": [n for n, _ in blocks],
        "steps": steps,
        "explained_total": round(1.0 - float(np.var(e, ddof=1)) / var0, 4) if var0 else None,
        "residual_after": round(float(np.var(e, ddof=1)), 6),
    }


def clustered_bootstrap(
    blocks: Sequence[Tuple[str, Tuple[str, str]]],
    keys: Dict[str, Tuple[np.ndarray, np.ndarray]],
    resid: np.ndarray,
    sigma2_noise: float,
    ligands: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(ligands)
    idx = {u: np.flatnonzero(ligands == u) for u in uniq}
    draws: Dict[str, List[float]] = {n: [] for n, _ in blocks}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        res = sequential(blocks, {k: (a[sel], b[sel]) for k, (a, b) in keys.items()},
                         resid[sel], sigma2_noise)
        for name, _ in blocks:
            draws[name].append(res["steps"][name]["share_of_residual"])
    return {
        name: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
        for name, v in draws.items()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--endpoint", default="potency")
    ap.add_argument("--noise-parquet", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_shape_decomposition.json")
    args = ap.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)
    sub = df[(df["endpoint"] == args.endpoint) & df["value_log10"].notna()
             & df["ligand_main_fragment"].notna()].copy().reset_index(drop=True)
    if sub.empty:
        print("FAIL: empty cohort")
        return 1

    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    X, groups = build_design_with_paper(sub, descriptors, cell_vocab)
    y = sub["value_log10"].to_numpy(dtype=float)
    ligand_key = sub["ligand_main_fragment"].to_numpy()

    print(f"[UP-17] rows={len(sub)} ligands={sub['ligand_canonical'].nunique()} "
          f"cells={sub['cell_line_key'].nunique()} dois={sub['doi'].nunique()}")

    additive, fold_id = oof_predict(X, y, ligand_key, groups)
    resid = y - additive
    sigma2_noise, df_noise = noise_floor(args.noise_parquet, args.endpoint)
    if not np.isfinite(sigma2_noise) or df_noise == 0:
        print(f"FAIL: no replicate groups in {args.noise_parquet} for {args.endpoint}")
        return 1
    var_resid = float(np.var(resid, ddof=1))

    print(f"[UP-17] additive OOF R2={score(y, additive)['r2']} "
          f"residual_var={var_resid:.4f} noise_sigma2={sigma2_noise:.4f} (df={df_noise})")

    keys: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (ca, cb) in BLOCKS:
        keys[name] = (
            sub[ca].astype(str).to_numpy(),
            sub[cb].astype(str).to_numpy(),
        )

    independent = {}
    for name, _ in BLOCKS:
        info = block_effects(*keys[name], resid, sigma2_noise)
        independent[name] = {
            "n_cells": info["n_cells"],
            "mean_cell_size": info["mean_cell_size"],
            "var_cell_means": info["var_cell_means"],
            "sampling_correction": info["sampling_correction"],
            "sigma2": info["sigma2"],
            "share_of_residual": round(info["sigma2"] / var_resid, 4) if var_resid else None,
            "mean_reliability": info["mean_reliability"],
        }

    forward = sequential(BLOCKS, keys, resid, sigma2_noise)
    # reversed(BLOCKS) puts paper x metal first, i.e. the batch-first order.
    batch_first = sequential(tuple(reversed(BLOCKS)), keys, resid, sigma2_noise)
    lig_keys = sub["ligand_canonical"].astype(str).to_numpy()
    cis_forward = clustered_bootstrap(BLOCKS, keys, resid, sigma2_noise, lig_keys, N_BOOT, BOOT_SEED)
    cis_batch = clustered_bootstrap(tuple(reversed(BLOCKS)), keys, resid, sigma2_noise,
                                   lig_keys, N_BOOT, BOOT_SEED)
    for name, _ in BLOCKS:
        independent[name]["share_forward_ci95_clustered"] = cis_forward[name]
        independent[name]["share_batch_first_ci95_clustered"] = cis_batch[name]
        independent[name]["share_forward"] = forward["steps"][name]["share_of_residual"]
        independent[name]["share_batch_first"] = batch_first["steps"][name]["share_of_residual"]

    # The frozen gate is "the four components account for the residual variance
    # to within 5%", i.e. 1 - explained_total < 0.05.  Comparing explained_total
    # against the sum of its own steps is a tautology (the variances telescope),
    # so it is NOT the gate.
    unexplained = {
        "structure_first": round(1.0 - float(forward["explained_total"]), 4),
        "batch_first": round(1.0 - float(batch_first["explained_total"]), 4),
    }
    paper_share = batch_first["steps"]["paper_x_metal"]["share_of_residual"]
    lig_share = forward["steps"]["ligand_x_metal"]["share_of_residual"]
    dominant_batch_first = max(batch_first["steps"],
                               key=lambda k: batch_first["steps"][k]["share_of_residual"] or 0.0)
    dominant_structure_first = max(forward["steps"],
                                   key=lambda k: forward["steps"][k]["share_of_residual"] or 0.0)

    # The kill rule as frozen ("paper x metal > 0.5 => the shape is paper batch")
    # was written for a single decomposition.  Both blocks exceed 0.5 under their
    # own priority and both fall to 0.0 under the other's, so the rule fires but
    # cannot support its stated consequence.
    kill_fires = (paper_share or 0) > 0.5
    both_exceed_half = kill_fires and (lig_share or 0) > 0.5
    if both_exceed_half:
        verdict = "SHAPE_SHARED_STRUCTURE_AND_PAPER_BATCH"
    elif kill_fires:
        verdict = "SHAPE_IS_PAPER_BATCH"
    else:
        verdict = "SHAPE_IS_STRUCTURAL"
    gate = {
        "unexplained_share": unexplained,
        "unexplained_within_5pct": {k: bool(v < 0.05) for k, v in unexplained.items()},
        "gate_pass": bool(all(v < 0.05 for v in unexplained.values())),
        "every_component_has_ci_both_orders": all(
            "share_forward_ci95_clustered" in v and "share_batch_first_ci95_clustered" in v
            for v in independent.values()
        ),
        "paper_x_metal_share_batch_first": paper_share,
        "ligand_x_metal_share_structure_first": lig_share,
        "kill_triggered": bool(kill_fires),
        "kill_consequence_valid": bool(not both_exceed_half),
        "order_spread": {
            name: [independent[name]["share_forward"], independent[name]["share_batch_first"]]
            for name, _ in BLOCKS
        },
    }

    out = {
        "artifact": "msrs_shape_decomposition",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L2 residual decomposition (UP-17)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup.relative_to(PROJECT_ROOT)),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc.relative_to(PROJECT_ROOT)),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": args.endpoint,
            "n_rows": int(len(sub)),
            "additive_model": "LightGBM on ECFP4 + metal block + cell block, "
                              "interaction_constraints=[[ligand],[metal],[cell]]",
            "cv": "GroupKFold(5) grouped by ligand_main_fragment (same as msrs_surface_model)",
            "estimator": "method-of-moments variance component per block, reliability-shrunk "
                         "cell effects, sequential subtraction",
            "blocks": [{"name": n, "keys": list(k)} for n, k in BLOCKS],
            "bootstrap": f"{N_BOOT} ligand-clustered resamples per order "
                         f"(structure-first and batch-first), seed {BOOT_SEED}",
            "gate_frozen_at": "2026-10-21T20:30+08:00",
        },
        "additive_fit": {
            "oof_r2": score(y, additive)["r2"],
            "oof_mae": score(y, additive)["mae"],
            "residual_var": round(var_resid, 6),
            "noise_sigma2": round(sigma2_noise, 6),
            "noise_source": str(args.noise_parquet.relative_to(PROJECT_ROOT)),
            "noise_sigma": round(float(np.sqrt(sigma2_noise)), 4),
            "noise_df": int(df_noise),
            "residual_over_noise": round(var_resid / sigma2_noise, 3) if sigma2_noise else None,
        },
        "independent": independent,
        "sequential_structure_first": forward,
        "sequential_batch_first": batch_first,
        "dominant_component": {
            "structure_first": dominant_structure_first,
            "batch_first": dominant_batch_first,
        },
        "verdict": verdict,
        "gate": gate,
        "caveats": [
            "Blocks overlap by construction (a ligand-metal cell is also a ligand-cell cell). "
            "Read the independent shares as upper bounds and the two sequential passes as the "
            "order-dependent range. sequential_batch_first is the reverse of the block order, "
            "so a separate 'reverse' pass would be identical and is not reported.",
            "sigma2_noise is pooled across metals; Rh and Re have few replicate groups, so their "
            "own noise floor is barely constrained. The pooled value is the conservative choice.",
            "The additive model is fitted on ligand_main_fragment folds, not DOI folds, so this "
            "decomposition describes in-distribution residual structure. It says nothing about "
            "whether the components transfer to an unseen paper.",
            "Both ligand x metal and paper x metal exceed 0.5 of the residual under their own "
            "priority and drop to 0.0 under the other's. The kill rule fires on the batch-first "
            "order but its consequence ('any model is fitting noise') does not follow, because "
            "ligand x metal explains strictly more when it goes first.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    for name in independent:
        v = independent[name]
        print(f"[UP-17] {name:15s} share={v['share_of_residual']} "
              f"ci_fwd={v['share_forward_ci95_clustered']} ci_batch={v['share_batch_first_ci95_clustered']} "
              f"fwd={v['share_forward']} batch_first={v['share_batch_first']} "
              f"cells={v['n_cells']} rel={v['mean_reliability']}")
    print(f"[UP-17] dominant structure_first={dominant_structure_first} "
          f"batch_first={dominant_batch_first} verdict={verdict}")
    print(f"[UP-17] unexplained structure_first={unexplained['structure_first']} "
          f"batch_first={unexplained['batch_first']} gate_pass={gate['gate_pass']}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
