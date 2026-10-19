#!/usr/bin/env python3
"""STR-4b — five-seed maccs_rdkit arm, and its fusion with the D-MPNN.

The single-seed screening run (``msrs_feature_expansion.py``) found one arm
that clears the DOI gate: ECFP4 + MACCS(167) + 20 RDKit descriptors, DOI R2
0.3706 vs the tree's 0.3332.  A single seed is not enough for a headline, so
this script:

1. averages the five per-seed OOF files and reports mean +/- sd plus the
   seed-averaged R2;
2. recomputes the paired DOI-cluster bootstrap against the seed-averaged tree;
3. reports the delta sign accuracy (the other half of the pre-registered bar);
4. fuses the maccs tree with the existing five-seed GNN OOF and reports the
   same numbers, because the project's best model is the fusion.

Usage::

    .venv/bin/python molmetal/scripts/msrs_feature_expansion_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_fusion_protocols import (  # noqa: E402
    _same,
    foldwise,
    paired_r2_delta,
)
from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
ARM = "maccs_rdkit"
TREE_ARM = "ligand+both"
GNN_ARM = "gnn_metal_state"
IDENTITY = (
    "y",
    "metal",
    "cell_line_key",
    "doi",
    "time_h",
    "ligand_canonical",
    "ligand_main_fragment",
)
N_BOOT = 2000
DOI_GATE = 0.36
SIGN_GATE = 0.65


def load_seeds(directory: Path, arm: str) -> Dict[str, object]:
    paths = sorted(directory.glob("oof_seed*.npz"))
    if not paths:
        raise FileNotFoundError(f"no oof_seed*.npz under {directory}")
    loaded = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            loaded.append({k: data[k] for k in data.files})
    first = loaded[0]
    for path, arrays in zip(paths[1:], loaded[1:]):
        for key in IDENTITY:
            if key in first and key in arrays and not _same(key, first[key], arrays[key]):
                raise ValueError(f"{path}: {key!r} differs from {paths[0]}")
        for protocol in PROTOCOLS:
            key = f"{protocol}__fold"
            if not np.array_equal(first[key], arrays[key]):
                raise ValueError(f"{path}: {key} differs from {paths[0]}")

    out: Dict[str, object] = {
        "paths": paths,
        "y": first["y"].astype(float),
        "doi": np.asarray(first["doi"]).astype(str),
        "metal": np.asarray(first["metal"]).astype(str),
        "cell_line_key": np.asarray(first["cell_line_key"]).astype(str),
        "ligand_canonical": np.asarray(first["ligand_canonical"]).astype(str),
        "time_h": first["time_h"].astype(float),
    }
    for protocol in PROTOCOLS:
        out[f"{protocol}__fold"] = first[f"{protocol}__fold"].astype(int)
        out[f"{protocol}__preds"] = np.column_stack(
            [a[f"{protocol}__{arm}"].astype(float) for a in loaded]
        )
    return out


def seed_average(preds: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(preds, axis=1)


def sign_accuracy(y: np.ndarray, pred: np.ndarray, ia: np.ndarray, ib: np.ndarray, fold: np.ndarray) -> Dict[str, object]:
    obs = y[ib] - y[ia]
    prd = pred[ib] - pred[ia]
    valid = (fold[ia] >= 0) & (fold[ib] >= 0) & np.isfinite(prd)
    keep = valid & (np.abs(obs) >= GATE_THRESHOLD)
    if not keep.any():
        return {"n_pairs": 0, "sign_accuracy": None}
    return {
        "n_pairs": int(keep.sum()),
        "sign_accuracy": round(float(np.mean(np.sign(prd[keep]) == np.sign(obs[keep]))), 4),
        "mae_delta": round(float(np.mean(np.abs(prd[keep] - obs[keep]))), 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_feature_expansion",
    )
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
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_feature_expansion_5seed.json",
    )
    parser.add_argument(
        "--fusion-out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_feature_fusion.json",
    )
    args = parser.parse_args(argv)

    maccs = load_seeds(args.dir, ARM)
    tree = load_seeds(args.tree_dir, TREE_ARM)
    gnn = load_seeds(args.gnn_dir, GNN_ARM)
    y = maccs["y"]
    doi = maccs["doi"]

    frame = pd.DataFrame(
        {
            "ligand_canonical": maccs["ligand_canonical"],
            "cell_line_key": maccs["cell_line_key"],
            "doi": doi,
            "time_h": maccs["time_h"],
            "metal": maccs["metal"],
        }
    )
    pairs = pair_table(frame)
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()

    protocols: Dict[str, object] = {}
    for protocol in PROTOCOLS:
        fold = maccs[f"{protocol}__fold"]
        ev = fold >= 0
        preds = maccs[f"{protocol}__preds"]
        ens = seed_average(preds)
        base = seed_average(tree[f"{protocol}__preds"])
        finite = ev & np.isfinite(ens) & np.isfinite(base)

        per_seed = [score(y[ev], preds[ev, j])["r2"] for j in range(preds.shape[1])]
        entry = {
            "per_seed_r2": [round(float(v), 4) for v in per_seed],
            "r2_mean": round(float(np.mean(per_seed)), 4),
            "r2_sd": round(float(np.std(per_seed, ddof=1)), 4),
            "r2_seed_averaged": round(float(score(y[ev], ens[ev])["r2"]), 4),
            "mae_seed_averaged": round(float(score(y[ev], ens[ev])["mae"]), 4),
            "vs_tree": paired_r2_delta(y[finite], base[finite], ens[finite], doi[finite], n_boot=N_BOOT),
            "delta_sign": sign_accuracy(y, ens, ia, ib, fold),
        }

        stack, _params = foldwise(y, ens, seed_average(gnn[f"{protocol}__preds"]), fold, "linear")
        finite_f = finite & np.isfinite(stack)
        entry["fusion"] = {
            "r2": round(float(score(y[ev], stack[ev])["r2"]), 4),
            "vs_tree": paired_r2_delta(y[finite_f], base[finite_f], stack[finite_f], doi[finite_f], n_boot=N_BOOT),
            "delta_sign": sign_accuracy(y, stack, ia, ib, fold),
        }
        protocols[protocol] = entry
        print(
            f"[str4b] {protocol:9s} maccs R2={entry['r2_seed_averaged']:.4f} "
            f"(seed {entry['r2_mean']:.4f}+/-{entry['r2_sd']:.4f}) "
            f"d={entry['vs_tree']['delta_r2']:+.4f} "
            f"[{entry['vs_tree']['ci_low']:+.4f},{entry['vs_tree']['ci_high']:+.4f}] "
            f"sign={entry['delta_sign']['sign_accuracy']} | "
            f"fused R2={entry['fusion']['r2']:.4f} "
            f"d={entry['fusion']['vs_tree']['delta_r2']:+.4f} "
            f"sign={entry['fusion']['delta_sign']['sign_accuracy']}"
        )

    doi_entry = protocols["doi"]
    passed = (
        doi_entry["r2_seed_averaged"] >= DOI_GATE
        and doi_entry["vs_tree"]["ci_low"] > 0
    )
    sign_ok = (
        doi_entry["delta_sign"]["sign_accuracy"] is not None
        and doi_entry["delta_sign"]["sign_accuracy"] >= SIGN_GATE
    )
    verdict = "PASS" if passed else ("PARTIAL" if doi_entry["r2_seed_averaged"] >= 0.34 else "KILL")

    payload = {
        "artifact": "msrs_feature_expansion_5seed",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_feature_expansion_ensemble.py",
            "git_commit": _git_commit(),
            "seeds": [p.name for p in maccs["paths"]],
            "tree_dir": str(args.tree_dir),
            "gnn_dir": str(args.gnn_dir),
        },
        "design": {
            "arm": ARM,
            "n_seeds": len(maccs["paths"]),
            "protocols": list(PROTOCOLS),
            "doi_gate": DOI_GATE,
            "sign_gate": SIGN_GATE,
            "sign_threshold": GATE_THRESHOLD,
            "bootstrap": {"n_boot": N_BOOT, "cluster": "doi", "paired": True},
            "fusion": "stack_foldwise on the seed-averaged maccs tree and GNN predictions",
        },
        "protocols": protocols,
        "verdict": verdict,
        "verdict_detail": {
            "doi_r2": doi_entry["r2_seed_averaged"],
            "doi_delta": doi_entry["vs_tree"]["delta_r2"],
            "doi_ci": [doi_entry["vs_tree"]["ci_low"], doi_entry["vs_tree"]["ci_high"]],
            "doi_sign": doi_entry["delta_sign"]["sign_accuracy"],
            "r2_gate": passed,
            "sign_gate": sign_ok,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    fusion_payload = {
        "artifact": "msrs_feature_fusion",
        "version": "1.0",
        "generated_utc": payload["generated_utc"],
        "provenance": payload["provenance"],
        "design": payload["design"],
        "protocols": {
            p: {
                "tree_maccs_r2": protocols[p]["r2_seed_averaged"],
                "fused_r2": protocols[p]["fusion"]["r2"],
                "vs_tree": protocols[p]["fusion"]["vs_tree"],
                "delta_sign": protocols[p]["fusion"]["delta_sign"],
            }
            for p in PROTOCOLS
        },
        "verdict": "PASS"
        if protocols["doi"]["fusion"]["r2"] >= DOI_GATE
        and protocols["doi"]["fusion"]["vs_tree"]["ci_low"] > 0
        else "PARTIAL",
    }
    atomic_write_file(args.fusion_out, json.dumps(fusion_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"[str4b] verdict={verdict} doi_r2={doi_entry['r2_seed_averaged']} sign={doi_entry['delta_sign']['sign_accuracy']}")
    print(f"[str4b] wrote {args.out}")
    print(f"[str4b] wrote {args.fusion_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
