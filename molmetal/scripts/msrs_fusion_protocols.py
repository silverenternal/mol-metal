#!/usr/bin/env python3
"""STR-2b — does the tree+GNN fusion survive the four protocols?

FIX-M2 measured ``blend_equal`` at +0.0273 R2 on a ligand-grouped partition
and MET-2 then showed the tree falls from 0.5556 to 0.3332 under DOI
grouping.  Neither result answers the question the paper actually needs:
*does the fusion lift the paper-held-out number?*  This script takes the
tree OOF from ``msrs_seed_ensemble`` and the GNN OOF from
``msrs_dmpnn_protocols`` — both produced on the same folds — and evaluates

    tree            LightGBM on ligand+both (MET-2)
    gnn             D-MPNN + metal state (STR-2a)
    blend_equal     0.5/0.5, no fitted parameters (primary)
    blend_foldwise  w fitted on the other folds
    stack_foldwise  a*tree + b*gnn + c fitted on the other folds

Predictions are first averaged over the available seeds, so a single seed's
noise cannot decide the verdict.  Every delta carries a paired DOI-cluster
bootstrap CI: the same resampled DOIs are used for both models, because a
marginal CI on R2 is far too wide to detect the effect (MET-8: MDE 0.0643).

Gate (fixed before the numbers were seen, ``todo.json`` STR-2):

    DOI stack R2 >= 0.36 and paired delta-R2 CI low > 0.

Usage::

    .venv/bin/python molmetal/scripts/msrs_fusion_protocols.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_hybrid import fit_linear, fit_one_weight  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
TREE_ARM = "ligand+both"
GNN_ARM = "gnn_metal_state"
N_BOOT = 2000
BOOT_SEED = 20261019
DOI_STACK_GATE = 0.36
DOI_STACK_KILL = 0.34
MIN_GAIN = 0.01

# Columns that must agree row-for-row between every OOF file.
IDENTITY = (
    "y",
    "metal",
    "cell_line_key",
    "doi",
    "time_h",
    "ligand_canonical",
    "ligand_main_fragment",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_seed_files(directory: Path, pattern: str) -> List[Path]:
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern!r} under {directory}")
    return paths


def stack_seeds(paths: Sequence[Path], protocols: Sequence[str], arm: str) -> Dict[str, np.ndarray]:
    """Load one arm from several seeds; verify identity columns and folds."""
    per_seed: List[Dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            per_seed.append({k: data[k] for k in data.files})

    first = per_seed[0]
    for path, arrays in zip(paths[1:], per_seed[1:]):
        for key in IDENTITY:
            if key not in first or key not in arrays:
                continue
            if not np.array_equal(first[key], arrays[key]):
                raise ValueError(f"{path}: column {key!r} differs from {paths[0]}")
        for proto in protocols:
            fkey = f"{proto}__fold"
            if not np.array_equal(first[fkey], arrays[fkey]):
                raise ValueError(f"{path}: {fkey} differs from {paths[0]}")

    out: Dict[str, np.ndarray] = {k: first[k] for k in IDENTITY if k in first}
    for proto in protocols:
        out[f"{proto}__fold"] = first[f"{proto}__fold"].astype(int)
        preds = []
        for arrays in per_seed:
            key = f"{proto}__{arm}"
            if key not in arrays:
                raise KeyError(f"{paths[0].parent}: missing {key!r}")
            preds.append(arrays[key].astype(float))
        out[f"{proto}__pred"] = np.nanmean(np.stack(preds, axis=1), axis=1)
    out["n_seeds"] = np.array([len(paths)])
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
def paired_r2_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Dict[str, object]:
    """Paired cluster bootstrap of ``R2(b) - R2(a)``.

    Resamples *clusters* (DOIs), not rows, and applies the same draw to both
    models so the difference is paired.  R2 uses the resampled mean, which is
    the quantity the protocol actually reports.
    """
    frame = pd.DataFrame({"c": clusters, "y": y, "a": pred_a, "b": pred_b})
    frame = frame[np.isfinite(frame["a"]) & np.isfinite(frame["b"])]
    codes, uniques = pd.factorize(frame["c"], sort=True)
    n_clusters = len(uniques)

    def r2(yv: np.ndarray, pv: np.ndarray) -> float:
        resid = float(np.sum((yv - pv) ** 2))
        total = float(np.sum((yv - yv.mean()) ** 2))
        return 1.0 - resid / total if total > 0 else float("nan")

    yv = frame["y"].to_numpy()
    av = frame["a"].to_numpy()
    bv = frame["b"].to_numpy()
    point = r2(yv, bv) - r2(yv, av)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    groups = [np.flatnonzero(codes == k) for k in range(n_clusters)]
    for i in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[j] for j in pick])
        draws[i] = r2(yv[idx], bv[idx]) - r2(yv[idx], av[idx])

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "delta_r2": round(float(point), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n_clusters": int(n_clusters),
        "n_rows": int(len(frame)),
        "n_boot": int(n_boot),
        "note": "paired cluster bootstrap; positive = b better than a",
    }


# ---------------------------------------------------------------------------
# Fold-wise stacking
# ---------------------------------------------------------------------------
def foldwise(
    y: np.ndarray,
    p_tree: np.ndarray,
    p_gnn: np.ndarray,
    folds: np.ndarray,
    kind: str,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    pred = np.full(len(y), np.nan)
    params: List[Dict[str, float]] = []
    for k in sorted(set(int(v) for v in folds[folds >= 0])):
        train = (folds != k) & np.isfinite(p_tree) & np.isfinite(p_gnn)
        test = folds == k
        if kind == "one_weight":
            w = fit_one_weight(y[train], p_tree[train], p_gnn[train])
            pred[test] = w * p_tree[test] + (1.0 - w) * p_gnn[test]
            params.append({"fold": int(k), "w_tree": round(w, 4)})
        elif kind == "linear":
            a, b, c = fit_linear(y[train], p_tree[train], p_gnn[train])
            pred[test] = a * p_tree[test] + b * p_gnn[test] + c
            params.append(
                {"fold": int(k), "a_tree": round(a, 4), "b_gnn": round(b, 4), "c": round(c, 4)}
            )
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return pred, params


# ---------------------------------------------------------------------------
# One protocol
# ---------------------------------------------------------------------------
def run_protocol(
    data: Dict[str, np.ndarray], protocol: str
) -> Tuple[Dict[str, object], Dict[str, object]]:
    y = data["y"].astype(float)
    folds = data[f"{protocol}__fold"]
    p_tree = data[f"{protocol}__tree"]
    p_gnn = data[f"{protocol}__gnn"]

    evaluated = (folds >= 0) & np.isfinite(p_tree) & np.isfinite(p_gnn)
    yv = y[evaluated]
    tree = p_tree[evaluated]
    gnn = p_gnn[evaluated]
    fold_v = folds[evaluated]
    doi_v = data["doi"][evaluated].astype(str)

    equal = 0.5 * (tree + gnn)
    one_w, one_params = foldwise(yv, tree, gnn, fold_v, "one_weight")
    linear, linear_params = foldwise(yv, tree, gnn, fold_v, "linear")
    if np.isnan(one_w).any() or np.isnan(linear).any():
        raise RuntimeError(f"{protocol}: fold-wise stacking left rows unscored")

    preds = {
        "tree": tree,
        "gnn": gnn,
        "blend_equal": equal,
        "blend_foldwise": one_w,
        "stack_foldwise": linear,
    }
    scores: Dict[str, object] = {}
    for name, pred in preds.items():
        entry = score(yv, pred)
        if name != "tree":
            entry["vs_tree"] = paired_r2_delta(yv, tree, pred, doi_v)
        scores[name] = entry

    deltas = {
        name: scores[name]["vs_tree"]["delta_r2"]
        for name in ("gnn", "blend_equal", "blend_foldwise", "stack_foldwise")
    }
    best = max(
        ("blend_equal", "blend_foldwise", "stack_foldwise"),
        key=lambda name: scores[name]["r2"],
    )
    entry = {
        "n_evaluated": int(evaluated.sum()),
        "n_unscored": int((~evaluated).sum()),
        "arms": scores,
        "delta_vs_tree": deltas,
        "best_blend": best,
        "weights": {"one_weight": one_params, "linear": linear_params},
    }
    diag = {
        "residual_correlation": round(
            float(np.corrcoef(yv - tree, yv - gnn)[0, 1]), 4
        ),
        "tree_r2": scores["tree"]["r2"],
        "gnn_r2": scores["gnn"]["r2"],
    }
    return entry, diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_seed_ensemble",
    )
    parser.add_argument(
        "--gnn-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_dmpnn_protocols",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_fusion_protocols.json",
    )
    args = parser.parse_args(argv)

    tree_paths = load_seed_files(args.tree_dir, "oof_seed*.npz")
    gnn_paths = load_seed_files(args.gnn_dir, "oof_seed*.npz")
    tree = stack_seeds(tree_paths, PROTOCOLS, TREE_ARM)
    gnn = stack_seeds(gnn_paths, PROTOCOLS, GNN_ARM)

    for key in IDENTITY:
        if key in tree and key in gnn and not np.array_equal(tree[key], gnn[key]):
            raise ValueError(f"tree and GNN OOF disagree on {key!r}")
    for proto in PROTOCOLS:
        if not np.array_equal(tree[f"{proto}__fold"], gnn[f"{proto}__fold"]):
            raise ValueError(f"tree and GNN OOF disagree on {proto}__fold")

    data = dict(tree)
    for proto in PROTOCOLS:
        data[f"{proto}__tree"] = tree[f"{proto}__pred"]
        data[f"{proto}__gnn"] = gnn[f"{proto}__pred"]

    protocols: Dict[str, object] = {}
    diagnostics: Dict[str, object] = {}
    for proto in PROTOCOLS:
        entry, diag = run_protocol(data, proto)
        protocols[proto] = entry
        diagnostics[proto] = diag
        print(
            f"[str2b] {proto:9s} tree={entry['arms']['tree']['r2']:.4f} "
            f"gnn={entry['arms']['gnn']['r2']:.4f} "
            f"equal={entry['arms']['blend_equal']['r2']:.4f} "
            f"stack={entry['arms']['stack_foldwise']['r2']:.4f} "
            f"d_stack={entry['delta_vs_tree']['stack_foldwise']:+.4f} "
            f"[{entry['arms']['stack_foldwise']['vs_tree']['ci_low']:+.4f},"
            f"{entry['arms']['stack_foldwise']['vs_tree']['ci_high']:+.4f}]"
        )

    doi = protocols["doi"]["arms"]["stack_foldwise"]
    doi_r2 = float(doi["r2"])
    doi_ci = doi["vs_tree"]
    if doi_r2 >= DOI_STACK_GATE and doi_ci["ci_low"] > 0:
        verdict = "PASS"
    elif doi_r2 < DOI_STACK_KILL or (
        doi_ci["ci_low"] <= 0 <= doi_ci["ci_high"] and doi_ci["delta_r2"] < MIN_GAIN
    ):
        verdict = "KILL"
    else:
        verdict = "PARTIAL"

    payload = {
        "artifact": "msrs_fusion_protocols",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": _display_path(Path(__file__)),
            "git_commit": _git_commit(),
            "tree_seeds": [_display_path(p) for p in tree_paths],
            "gnn_seeds": [_display_path(p) for p in gnn_paths],
            "tree_seed_sha256": [_sha256(p) for p in tree_paths],
            "gnn_seed_sha256": [_sha256(p) for p in gnn_paths],
        },
        "design": {
            "tree_arm": TREE_ARM,
            "gnn_arm": GNN_ARM,
            "n_tree_seeds": len(tree_paths),
            "n_gnn_seeds": len(gnn_paths),
            "protocols": list(PROTOCOLS),
            "primary_arm": "blend_equal",
            "gate_arm": "stack_foldwise",
            "doi_stack_gate": DOI_STACK_GATE,
            "doi_stack_kill": DOI_STACK_KILL,
            "min_gain": MIN_GAIN,
            "bootstrap": {
                "n_boot": N_BOOT,
                "seed": BOOT_SEED,
                "cluster": "doi",
                "paired": True,
            },
            "note": (
                "Predictions are seed-averaged before scoring. blend_equal fits "
                "nothing and is the primary arm; the gate uses stack_foldwise "
                "because that is what STR-2 pre-registered."
            ),
        },
        "protocols": protocols,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "verdict_arm": {
            "protocol": "doi",
            "arm": "stack_foldwise",
            "r2": doi_r2,
            "delta_r2": doi_ci["delta_r2"],
            "ci": [doi_ci["ci_low"], doi_ci["ci_high"]],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[str2b] verdict={verdict} doi_stack_r2={doi_r2:.4f}")
    print(f"[str2b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
