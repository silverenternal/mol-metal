#!/usr/bin/env python3
"""MET-6 — is the gain actually carried by metal information?

MET-2 measured a representation gain of +0.0544 R2 on the ligand protocol
(+0.0423 on DOI) when the state and coordination blocks are added to the
ECFP4 + categorical baseline.  That number says the *blocks* help; it does not
say the help comes from metal identity.  Three confounds remain:

1. the metal **label** is also present as a LightGBM categorical code, so a
   block gain could be re-encoding something the code already had;
2. the state block (oxidation state, charge, d-electrons, valence) is nearly a
   deterministic function of the metal, so "metal awareness" may be nothing
   more than "metal one-hot with extra steps";
3. the ligand ECFP4 block could already be doing the work, with the metal
   blocks only absorbing residual variance.

This script removes one block at a time and then permutes metal information to
build a null distribution.

Arms (all on the same folds, same seed, same LightGBM params as MET-2)::

    full              ECFP4 + state + coord + metal_code + cell_code
    no_metal_code     full - metal categorical code
    no_state          full - state
    no_coord          full - coord
    no_state_coord    full - state - coord          (= MET-2 "ligand" arm)
    metal_blind       full - metal_code - state - coord
    no_complex        full - ECFP4                  (structure-blind control)
    cell_blind        full - cell categorical code  (non-metal control)
    ligand_only       ECFP4 only

Two permutation nulls, 100 draws each, same folds and seed:

``metal_label``  permute the metal label, rebuild only the categorical code.
                 Tests whether the *label* adds anything beyond state/coord.
``metal_block``  permute the metal code + state + coord rows together.
                 Tests whether the whole metal-derived block carries signal;
                 this is the null for the metal-blind gap.

p-values are (1 + #{null >= observed}) / (1 + n_draws).

Output: ``metrics/by_metric/msrs_metal_ablation.json``.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_metal_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_generalization import scheme_folds  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    ENDPOINT,
    LIGAND_ONLY,
    SEED,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    native_codes,
    state_block,
)
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi")
N_PERM = 100
PRIMARY_PROTOCOL = "ligand"
MIN_DROP_OK = 0.05
P_GATE = 0.01
N_BOOT = 2000

#: name -> (complex, state, coord, metal_code, cell_code)
ARMS: Dict[str, tuple[bool, bool, bool, bool, bool]] = {
    "full": (True, True, True, True, True),
    "no_metal_code": (True, True, True, False, True),
    "no_state": (True, False, True, True, True),
    "no_coord": (True, True, False, True, True),
    "no_state_coord": (True, False, False, True, True),
    "metal_blind": (True, False, False, False, True),
    "no_complex": (False, True, True, True, True),
    "cell_blind": (True, True, True, True, False),
    "ligand_only": (True, False, False, False, False),
}


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def arm_matrix(
    name: str,
    complex_block: sparse.csr_matrix,
    state: np.ndarray,
    coord: sparse.csr_matrix,
    codes: np.ndarray,
) -> tuple[sparse.csr_matrix, Optional[np.ndarray]]:
    """Return ``(X, cat_codes)`` for one ablation arm.

    ``cat_codes`` keeps the column order [metal, cell] when both survive, so a
    single-column arm stays a ``(n, 1)`` array and LightGBM still sees it as a
    categorical feature.
    """
    use_complex, use_state, use_coord, use_metal, use_cell = ARMS[name]
    parts: List[sparse.csr_matrix] = []
    if use_complex:
        parts.append(complex_block)
    if use_state:
        parts.append(sparse.csr_matrix(state))
    if use_coord:
        parts.append(coord)
    if not parts:
        raise ValueError(f"arm {name!r} has no features")
    X = sparse.hstack(parts, format="csr")

    keep = [i for i, flag in enumerate((use_metal, use_cell)) if flag]
    cat = codes[:, keep] if keep else None
    return X, cat


def oof_predict(
    X: sparse.csr_matrix,
    cat: Optional[np.ndarray],
    y: np.ndarray,
    fold_id: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    empty = sparse.csr_matrix((len(y), 0), dtype=np.float32)
    return fit_arm(
        X,
        empty,
        empty,
        None,
        0,
        False,
        y,
        fold_id,
        cat_codes=cat,
        params_override={"random_state": int(seed)},
    )


def arm_r2(
    X: sparse.csr_matrix,
    cat: Optional[np.ndarray],
    y: np.ndarray,
    fold_id: np.ndarray,
    seed: int = SEED,
) -> tuple[float, np.ndarray]:
    pred = oof_predict(X, cat, y, fold_id, seed=seed)
    ev = fold_id >= 0
    return float(score(y[ev], pred[ev])["r2"]), pred


# ---------------------------------------------------------------------------
# Group bootstrap
# ---------------------------------------------------------------------------
def group_bootstrap_delta(
    y: np.ndarray,
    pred_full: np.ndarray,
    pred_arm: np.ndarray,
    groups: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Dict[str, object]:
    """CI for R2(full) - R2(arm), resampling the grouping key, not rows."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        yb, fb, ab = y[idx], pred_full[idx], pred_arm[idx]
        ss_tot = float(np.sum((yb - yb.mean()) ** 2))
        if ss_tot <= 0:
            deltas[b] = np.nan
            continue
        r2_full = 1.0 - float(np.sum((yb - fb) ** 2)) / ss_tot
        r2_arm = 1.0 - float(np.sum((yb - ab) ** 2)) / ss_tot
        deltas[b] = r2_full - r2_arm
    good = deltas[np.isfinite(deltas)]
    return {
        "n_boot": int(len(good)),
        "n_groups": int(len(uniq)),
        "ci95": [round(float(np.percentile(good, 2.5)), 4),
                 round(float(np.percentile(good, 97.5)), 4)],
        "mean": round(float(good.mean()), 4),
        "p_one_sided": round(float(np.mean(good <= 0.0)), 4),
    }


# ---------------------------------------------------------------------------
# Permutation nulls
# ---------------------------------------------------------------------------
# The workers are module-level and read a process-global context.  With the
# Linux ``fork`` start method the child inherits the context by copy-on-write,
# so the 21k x 2074 sparse matrix is never pickled.
_CTX: Dict[str, object] = {}


def _init_ctx(ctx: Dict[str, object]) -> None:
    global _CTX
    _CTX = ctx


def permute_metal_label(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle column 0 (metal) only; cell code and row order stay put."""
    out = codes.copy()
    out[:, 0] = out[rng.permutation(len(out)), 0]
    return out


def permute_metal_block(
    codes: np.ndarray,
    state: np.ndarray,
    coord: sparse.csr_matrix,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Shuffle metal code, state and coord rows together.

    The same row permutation is applied to all three so the metal-derived
    features stay internally consistent with each other; what breaks is their
    association with the ligand, cell line and outcome.
    """
    perm = rng.permutation(len(codes))
    out_codes = codes.copy()
    out_codes[:, 0] = out_codes[perm, 0]
    return out_codes, state[perm], coord[perm]


def permutation_pvalue(observed: float, null: Sequence[float]) -> float:
    null = np.asarray(null, dtype=float)
    return float((1.0 + np.sum(null >= observed)) / (1.0 + len(null)))


def _perm_label_worker(i: int) -> float:
    c = _CTX
    rng = np.random.default_rng(int(c["seed"]) + 10_000 + i)
    codes = permute_metal_label(c["codes"], rng)
    X, cat = arm_matrix("full", c["complex"], c["state"], c["coord"], codes)
    pred = oof_predict(X, cat, c["y"], c["fold"], seed=int(c["seed"]))
    ev = c["fold"] >= 0
    return float(score(c["y"][ev], pred[ev])["r2"])


def _perm_block_worker(i: int) -> float:
    c = _CTX
    rng = np.random.default_rng(int(c["seed"]) + 20_000 + i)
    codes, state, coord = permute_metal_block(
        c["codes"], c["state"], c["coord"], rng
    )
    X, cat = arm_matrix("full", c["complex"], state, coord, codes)
    pred = oof_predict(X, cat, c["y"], c["fold"], seed=int(c["seed"]))
    ev = c["fold"] >= 0
    return float(score(c["y"][ev], pred[ev])["r2"])


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------
def run_protocol(
    sub: pd.DataFrame,
    y: np.ndarray,
    complex_block: sparse.csr_matrix,
    state: np.ndarray,
    coord: sparse.csr_matrix,
    codes: np.ndarray,
    protocol: str,
    seed: int,
) -> Dict[str, object]:
    fold_id, groups, info = scheme_folds(sub, protocol)
    ev = fold_id >= 0
    t0 = time.time()

    arms: Dict[str, object] = {}
    preds: Dict[str, np.ndarray] = {}
    for name in ARMS:
        X, cat = arm_matrix(name, complex_block, state, coord, codes)
        r2, pred = arm_r2(X, cat, y, fold_id, seed=seed)
        preds[name] = pred
        arms[name] = {
            "r2": round(r2, 4),
            "mae": round(float(score(y[ev], pred[ev])["mae"]), 4),
            "n_features": int(X.shape[1]),
            "n_cat_features": 0 if cat is None else int(cat.shape[1]),
        }
        print(f"    {name:16s} R2={r2:.4f} feats={X.shape[1]}")

    full = preds["full"]
    for name in ARMS:
        if name == "full":
            continue
        arms[name]["delta_r2_vs_full"] = round(
            float(arms["full"]["r2"] - arms[name]["r2"]), 4
        )
        arms[name]["delta_ci"] = group_bootstrap_delta(
            y[ev], full[ev], preds[name][ev], groups[ev], seed=seed
        )

    # Reference arm for the label null: the metal code is the only block removed.
    return {
        "fold_info": info,
        "arms": arms,
        "wall_s": round(time.time() - t0, 1),
    }


def run_permutations(
    sub: pd.DataFrame,
    y: np.ndarray,
    complex_block: sparse.csr_matrix,
    state: np.ndarray,
    coord: sparse.csr_matrix,
    codes: np.ndarray,
    protocol: str,
    observed_full: float,
    observed_metal_blind: float,
    n_perm: int,
    seed: int,
    workers: int = 1,
) -> Dict[str, object]:
    fold_id, _, _ = scheme_folds(sub, protocol)
    ev = fold_id >= 0
    gap_obs = observed_full - observed_metal_blind

    _init_ctx(
        {
            "codes": codes,
            "state": state,
            "coord": coord,
            "complex": complex_block,
            "y": y,
            "fold": fold_id,
            "seed": int(seed),
        }
    )

    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            label_null = pool.map(_perm_label_worker, range(n_perm))
            block_null = pool.map(_perm_block_worker, range(n_perm))
    else:
        label_null = [_perm_label_worker(i) for i in range(n_perm)]
        block_null = [_perm_block_worker(i) for i in range(n_perm)]

    label_gap = np.asarray(label_null) - observed_metal_blind
    block_gap = np.asarray(block_null) - observed_metal_blind
    return {
        "n_perm": int(n_perm),
        "protocol": protocol,
        "observed_full_r2": round(float(observed_full), 4),
        "observed_metal_blind_r2": round(float(observed_metal_blind), 4),
        "observed_gap": round(float(gap_obs), 4),
        "metal_label": {
            "null_r2_mean": round(float(np.mean(label_null)), 4),
            "null_r2_ci95": [round(float(np.percentile(label_null, 2.5)), 4),
                             round(float(np.percentile(label_null, 97.5)), 4)],
            "p_value": round(permutation_pvalue(observed_full, label_null), 4),
            "note": "state and coord kept; tests the categorical label alone",
        },
        "metal_block": {
            "null_r2_mean": round(float(np.mean(block_null)), 4),
            "null_r2_ci95": [round(float(np.percentile(block_null, 2.5)), 4),
                             round(float(np.percentile(block_null, 97.5)), 4)],
            "null_gap_mean": round(float(np.mean(block_gap)), 4),
            "p_value": round(permutation_pvalue(gap_obs, block_gap), 4),
            "note": "metal code + state + coord permuted together; null for the metal-blind gap",
        },
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_ablation.json",
    )
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligands")
    parser.add_argument("--skip-permutations", action="store_true")
    args = parser.parse_args(argv)

    t_start = time.time()
    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(
        sub["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    codes = native_codes(
        sub["metal"].astype(str).tolist(),
        sub["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )
    bundle = build(sub, LIGAND_ONLY)
    state, state_names = state_block(sub)
    coord, coord_names, coverage = coordination_block(sub, include_geometry=True)

    print(
        f"[met6] rows={len(sub)} ligands={sub['ligand_canonical'].nunique()} "
        f"state={len(state_names)} coord={len(coord_names)} "
        f"complex={bundle.X.shape[1]}"
    )

    protocols: Dict[str, object] = {}
    for protocol in args.protocols:
        print(f"[met6] protocol={protocol}")
        protocols[protocol] = run_protocol(
            sub, y, bundle.X, state, coord, codes, protocol, args.seed
        )

    primary = protocols[PRIMARY_PROTOCOL]["arms"]
    drop = float(primary["metal_blind"]["delta_r2_vs_full"])
    perm: Dict[str, object] = {}
    if not args.skip_permutations and args.n_perm > 0:
        perm = run_permutations(
            sub,
            y,
            bundle.X,
            state,
            coord,
            codes,
            PRIMARY_PROTOCOL,
            observed_full=float(primary["full"]["r2"]),
            observed_metal_blind=float(primary["metal_blind"]["r2"]),
            n_perm=args.n_perm,
            seed=args.seed,
            workers=args.workers,
        )
        print(
            f"[met6] permutation: label p={perm['metal_label']['p_value']} "
            f"block p={perm['metal_block']['p_value']} "
            f"(gap={perm['observed_gap']})"
        )

    p_block = perm.get("metal_block", {}).get("p_value")
    passed = bool(drop >= MIN_DROP_OK and p_block is not None and p_block < P_GATE)
    verdict = "PASS" if passed else "NEGATIVE"

    payload = {
        "artifact": "msrs_metal_ablation",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "ligand_spec_hash": bundle.spec_hash,
        },
        "design": {
            "split_schema": "ligand and doi GroupKFold(5); consumes MET-2's partition code",
            "ligand_source": "ligand_canonical",
            "arm_ligand_source": {name: "ligand_canonical" for name in ARMS},
            "rows": int(len(sub)),
            "protocols": list(args.protocols),
            "primary_protocol": PRIMARY_PROTOCOL,
            "arms": {name: dict(zip(
                ("complex", "state", "coord", "metal_code", "cell_code"), flags
            )) for name, flags in ARMS.items()},
            "state_columns": state_names,
            "coord_columns": coord_names,
            "donor_coverage": coverage,
            "n_perm": int(args.n_perm),
            "seed": int(args.seed),
            "min_drop_ok": MIN_DROP_OK,
            "p_gate": P_GATE,
            "model": "LightGBM, LGB_PARAMS from msrs_surface_model (n_jobs=1, deterministic)",
            "gnn_note": (
                "The GNN arm is not re-run. MET-1 showed the tree is the decision "
                "variable (GNN alone loses to the tree in 10/10 combinations); the "
                "GNN consumes the same state/coord blocks, so a block-level "
                "conclusion from the tree bounds what the fusion can recover."
            ),
        },
        "protocols": protocols,
        "permutation": perm,
        "verdict": verdict,
        "verdict_rule": (
            f"{PRIMARY_PROTOCOL}: R2(full) - R2(metal_blind) >= {MIN_DROP_OK} "
            f"AND metal_block permutation p < {P_GATE}"
        ),
        "gate_measurement": {
            "metal_blind_drop": round(drop, 4),
            "metal_block_p": p_block,
        },
    }
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"[met6] verdict={verdict} drop={drop:.4f} p_block={p_block} "
        f"wall={time.time() - t_start:.0f}s -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
