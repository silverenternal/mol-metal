#!/usr/bin/env python3
"""MSRS-EXT-4 — is a learned metal embedding identifiable at all?

=====================================================================
The question
=====================================================================
EXT-2 asked whether static descriptors can place a metal's level and answered
no (KILL: the only positive LOOCV R2 was a single column on one endpoint, it
was not distinguishable from the 120-permutation null, and it inverted on the
second endpoint).

The natural alternative is to learn the metal vector from the data instead of
curating it: treat potency as a ligand x metal matrix, factor it, and read the
metal loadings off the factors.  That is how collaborative filtering handles a
new user.

This script asks the prior question, which the plan did not: **is the matrix
identified?**  A latent metal vector can only be estimated from ligands that
were measured against more than one metal.  If almost every ligand was tested
against exactly one metal, the metal columns share no rows, and no factorisation
can separate "this metal" from "these ligands".

=====================================================================
What is measured
=====================================================================
    coverage      how many ligands appear with >=2, >=3, and all 5 metals
    spectrum      singular values of the ligand-centred matrix, on the
                  overlapping subset only (imputation is reported, not hidden)
    loo_average   leave one metal out, predict its residual row as the mean of
                  the other four metals' residual rows.  This is the best a
                  factorisation can do when the held-out loading is unidentified.

Gate (pre-registered in todo.json, EXT-4): beat EXT-2's best arm by >= 0.03 R2.
An identification failure is reported as KILL with the coverage numbers that
cause it, not as a modelling result.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_metal_embedding.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
ENDPOINTS = ("potency", "cisplatin_ratio")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((pred - y) ** 2) / denom)


def analyse(piv: pd.DataFrame) -> Dict[str, object]:
    coverage = piv.notna().sum(axis=1)
    cells = {m: int(piv[m].notna().sum()) for m in METALS if m in piv.columns}
    n_lig = int(len(piv))
    overlap = piv[coverage >= 2].copy()

    out: Dict[str, object] = {
        "n_ligands": n_lig,
        "n_ligands_ge2_metals": int((coverage >= 2).sum()),
        "n_ligands_ge3_metals": int((coverage >= 3).sum()),
        "n_ligands_all5_metals": int((coverage == 5).sum()),
        "share_ge2": round(float((coverage >= 2).mean()), 4),
        "cells_per_metal": cells,
    }

    if len(overlap) < 20:
        out["spectrum"] = {"status": "INSUFFICIENT", "n_overlap": int(len(overlap))}
        out["loo_average"] = {"status": "INSUFFICIENT"}
        return out

    # Ligand-centre on the observed cells, then fill the missing cells with 0
    # (i.e. "this ligand has no metal-specific offset").  The fill is stated so
    # the spectrum is not mistaken for a clean factorisation.
    centred = overlap.sub(overlap.mean(axis=1), axis=0)
    n_missing = int(centred.isna().sum().sum())
    filled = centred.fillna(0.0).to_numpy(dtype=float)
    u, s, vt = np.linalg.svd(filled, full_matrices=False)
    var = s ** 2
    out["spectrum"] = {
        "n_overlap_ligands": int(len(overlap)),
        "n_missing_cells_filled_with_zero": n_missing,
        "singular_values": [round(float(x), 4) for x in s],
        "explained_variance_ratio": [round(float(x / var.sum()), 4) for x in var],
        "rank_upper_bound": int(min(filled.shape[0], filled.shape[1])),
        "note": "rank is bounded by 5 (metals) before centring and by 4 after; "
                "a 5-metal matrix cannot support more than four latent dimensions.",
    }

    # Leave one metal out: the held-out loading is unidentified, so the best
    # available guess is the mean of the other four residual rows.
    loo = {}
    for m in METALS:
        if m not in overlap.columns:
            continue
        obs = centred[m].dropna()
        if len(obs) < 10:
            loo[m] = {"status": "INSUFFICIENT", "n": int(len(obs))}
            continue
        others = [c for c in overlap.columns if c != m]
        pred = centred.loc[obs.index, others].mean(axis=1).to_numpy(dtype=float)
        true = obs.to_numpy(dtype=float)
        loo[m] = {
            "status": "OK",
            "n": int(len(true)),
            "r2": round(r2(pred, true), 4),
            "mae": round(float(np.mean(np.abs(pred - true))), 4),
            "corr": round(float(np.corrcoef(pred, true)[0, 1]), 4) if len(true) > 2 else None,
        }
    ok = {m: v for m, v in loo.items() if v.get("status") == "OK"}
    out["loo_average"] = {
        "per_metal": loo,
        "mean_r2": round(float(np.mean([v["r2"] for v in ok.values()])), 4) if ok else None,
        "n_positive": int(sum(1 for v in ok.values() if v["r2"] > 0)),
        "n_metals": len(ok),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--ext2",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_descriptor_level.json",
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_embedding.json"
    )
    args = parser.parse_args(argv)

    if not args.dedup.exists():
        print(f"FAIL: missing {args.dedup}")
        return 1

    df = pd.read_parquet(args.dedup)
    endpoints: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sub = df[(df["endpoint"] == ep) & df["value_log10"].notna()]
        piv = sub.groupby(["ligand_main_fragment", "metal"])["value_log10"].mean().unstack()
        endpoints[ep] = analyse(piv)
        e = endpoints[ep]
        print(
            f"[{ep}] ligands={e['n_ligands']} >=2 metals={e['n_ligands_ge2_metals']} "
            f"({e['share_ge2']:.1%}) all5={e['n_ligands_all5_metals']}"
        )

    ext2_best = None
    if args.ext2.exists():
        d2 = json.load(open(args.ext2, encoding="utf-8"))
        ext2_best = d2.get("best_loo_r2")
    loo_means = [endpoints[ep]["loo_average"].get("mean_r2") for ep in ENDPOINTS]
    loo_means = [x for x in loo_means if x is not None]
    best_here = max(loo_means) if loo_means else None

    identified = all(e["share_ge2"] >= 0.20 for e in endpoints.values())
    beats = (
        best_here is not None and ext2_best is not None and best_here >= ext2_best + 0.03
    )
    verdict = "PASS" if (identified and beats) else "KILL"

    out = {
        "artifact": "msrs_metal_embedding",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 metal embedding identifiability",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "matrix": "ligand_main_fragment x metal, mean value_log10 per cell",
            "gate": "coverage >= 20% of ligands with >=2 metals AND LOOCV R2 >= EXT-2 best + 0.03",
            "ext2_best_loo_r2": ext2_best,
            "loo_estimator": "held-out residual row predicted as the mean of the other four metals' residual rows",
        },
        "verdict": verdict,
        "best_loo_r2": best_here,
        "identified": identified,
        "endpoints": endpoints,
        "caveats": [
            "A latent metal vector is only estimable from ligands measured against more than one "
            "metal. The coverage numbers here are the binding constraint, not the factoriser.",
            "The singular-value spectrum is computed after filling missing cells with zero "
            "(no metal-specific offset). The number of filled cells is reported; the spectrum "
            "is descriptive, not a decomposition of measured variation.",
            "The leave-one-metal-out estimator is deliberately the weakest possible one, because "
            "a held-out metal's loading is unidentified. Any stronger estimator would need "
            "descriptors, which is EXT-2 and which failed.",
            "Rank is bounded by 4 after ligand-centring: five metals cannot support more than "
            "four latent dimensions, so an 8-dimensional embedding is unidentifiable by construction.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"\njson: {args.out}")
    print(f"VERDICT {verdict}  identified={identified}  best_loo_r2={best_here}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
