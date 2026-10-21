#!/usr/bin/env python3
"""UP-10 — how much paired evidence does each metal pair actually have?

=====================================================================
Why this exists
=====================================================================
Round 1 (UP-1..UP-9) tried seven ways to learn the ligand x metal
interaction. All seven failed. None of them asked the prior question:
**does the corpus contain the comparison at all?**

A metal-substitution effect is only identified when the same ligand is
measured against two metals *in the same cell line*, otherwise the
difference is confounded with the cell line's own potency scale. This
script counts those units, per metal pair, per endpoint, and per
metal-out scenario.

=====================================================================
What counts as support
=====================================================================
A *unit* is one (ligand_canonical, cell_line_key) group measured under at
least two metals. For each metal pair the unit contributes

    Delta = mean(y | metal_a) - mean(y | metal_b)

The noise on Delta is the pooled within-replicate variance of the two
metals (same ligand, metal, cell, duration, DOI). ``separability`` is
|mean Delta| divided by that noise SD: below 1 the typical substitution
is smaller than the assay's own replicate scatter.

=====================================================================
Gate (todo.json, frozen 2026-10-21T20:30+08:00)
=====================================================================
Artifact written with every endpoint x every metal-out scenario present.
A pair with fewer than ``MIN_GROUPS`` units is marked UNSUPPORTED and may
not enter any later training set.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_support_audit.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import (  # noqa: E402
    ENDPOINTS,
    METALS,
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

MIN_GROUPS = 20
N_BOOT = 1000
BOOT_SEED = 20261021


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = (
        df["endpoint"].isin(ENDPOINTS)
        & df["value_log10"].notna()
        & (df["censoring"].astype(str) == "none")
        & df["ligand_canonical"].notna()
    )
    return df[keep].copy().reset_index(drop=True)


def metal_noise(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Pooled within-replicate variance per metal (same ligand, cell, time, DOI)."""
    keys = ["ligand_canonical", "metal", "cell_line_key", "time_h", "doi"]
    g = df.groupby(keys, dropna=False)["value_log10"]
    n = g.size()
    var = g.var(ddof=1)
    rep = n >= 2
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for metal in METALS:
        sel = rep & (n.index.get_level_values("metal") == metal)
        dof = int((n[sel] - 1).sum())
        ss = float(((n[sel] - 1) * var[sel]).sum())
        out[metal] = {
            "n_replicate_groups": int(sel.sum()),
            "df": dof,
            "sigma2": round(ss / dof, 6) if dof else None,
            "sigma": round(float(np.sqrt(ss / dof)), 4) if dof else None,
        }
    return out


def unit_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ligand, cell) with the per-metal means and counts."""
    g = (
        df.groupby(["ligand_canonical", "cell_line_key", "metal"], dropna=False)["value_log10"]
        .agg(["mean", "size"])
        .reset_index()
    )
    return g


def pair_stats(
    units: pd.DataFrame,
    m1: str,
    m2: str,
    noise: Dict[str, Dict[str, Optional[float]]],
    endpoint: str,
) -> Dict[str, object]:
    a = units[units["metal"] == m1].rename(columns={"mean": "y1", "size": "n1"})
    b = units[units["metal"] == m2].rename(columns={"mean": "y2", "size": "n2"})
    j = a.merge(b, on=["ligand_canonical", "cell_line_key"], how="inner")
    if j.empty:
        return {
            "pair": f"{m1}-{m2}",
            "endpoint": endpoint,
            "n_groups": 0,
            "status": "EMPTY",
        }
    delta = (j["y1"] - j["y2"]).to_numpy(dtype=float)
    s2_1 = noise.get(m1, {}).get("sigma2")
    s2_2 = noise.get(m2, {}).get("sigma2")
    noise_var = float((s2_1 or 0.0) + (s2_2 or 0.0))
    var_obs = float(np.var(delta, ddof=1)) if len(delta) > 1 else float("nan")

    # Sampling noise on Delta includes 1/n of the replicate variance per row.
    samp = float(np.mean((1.0 / j["n1"].to_numpy()) + (1.0 / j["n2"].to_numpy()))) * noise_var / 2.0
    inter = max(0.0, var_obs - samp) if np.isfinite(var_obs) else float("nan")

    # Ligand-clustered bootstrap of the interaction share.
    rng = np.random.default_rng(BOOT_SEED)
    ligs = j["ligand_canonical"].to_numpy()
    uniq = np.unique(ligs)
    idx = {u: np.flatnonzero(ligs == u) for u in uniq}
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        v = float(np.var(delta[sel], ddof=1))
        boot.append(max(0.0, v - samp) / v if v > 0 else 0.0)
    boot_arr = np.asarray(boot)
    ci = (
        round(float(np.quantile(boot_arr, 0.025)), 4),
        round(float(np.quantile(boot_arr, 0.975)), 4),
    )

    mean_delta = float(np.mean(delta))
    noise_sd = float(np.sqrt(noise_var)) if noise_var > 0 else float("nan")
    return {
        "pair": f"{m1}-{m2}",
        "endpoint": endpoint,
        "n_groups": int(len(j)),
        "n_rows": int((j["n1"] + j["n2"]).sum()),
        "n_ligands": int(j["ligand_canonical"].nunique()),
        "n_cells": int(j["cell_line_key"].nunique()),
        "median_abs_delta": round(float(np.median(np.abs(delta))), 4),
        "mean_delta": round(mean_delta, 4),
        "sd_delta": round(float(np.sqrt(var_obs)), 4) if np.isfinite(var_obs) else None,
        "noise_sd_delta": round(noise_sd, 4) if np.isfinite(noise_sd) else None,
        "separability": round(abs(mean_delta) / noise_sd, 3) if noise_sd and np.isfinite(noise_sd) else None,
        "interaction_share": round(inter / var_obs, 4) if np.isfinite(inter) and var_obs > 0 else None,
        "interaction_share_ci95": list(ci),
        "status": "OK" if len(j) >= MIN_GROUPS else "UNSUPPORTED",
        "min_groups_required": MIN_GROUPS,
    }


def endpoint_audit(df: pd.DataFrame, endpoint: str) -> Dict[str, object]:
    sub = df[df["endpoint"] == endpoint]
    noise = metal_noise(sub)
    units = unit_table(sub)
    pairs = [
        pair_stats(units, m1, m2, noise, endpoint)
        for m1, m2 in itertools.combinations(METALS, 2)
    ]
    observed = [p for p in pairs if p["status"] == "OK"]
    ligand_span = units.groupby("ligand_canonical")["metal"].nunique()
    return {
        "n_rows": int(len(sub)),
        "n_ligands": int(sub["ligand_canonical"].nunique()),
        "n_cells": int(sub["cell_line_key"].nunique()),
        "n_dois": int(sub["doi"].nunique()),
        "n_ligands_ge2_metals": int((ligand_span >= 2).sum()),
        "share_ligands_ge2_metals": round(float((ligand_span >= 2).mean()), 4),
        "noise_floor_per_metal": noise,
        "pairs": pairs,
        "n_pairs_observed": len(observed),
        "n_pairs_total": len(pairs),
        "observed_pairs": [p["pair"] for p in observed],
        "unsupported_pairs": [p["pair"] for p in pairs if p["status"] != "OK"],
    }


def metal_out_scenarios(endpoint_audit: Dict[str, object]) -> Dict[str, object]:
    """Which pairs remain available when one metal is held out for testing."""
    ok = {
        tuple(sorted(p["pair"].split("-"))): p["n_groups"]
        for p in endpoint_audit["pairs"]
        if p["status"] == "OK"
    }
    out = {}
    for held in METALS:
        train = [m for m in METALS if m != held]
        avail = {}
        for m1, m2 in itertools.combinations(train, 2):
            key = tuple(sorted((m1, m2)))
            avail["-".join(key)] = int(ok.get(key, 0))
        out[held] = {
            "train_metals": train,
            "available_pairs": avail,
            "n_pairs_with_evidence": int(sum(v >= MIN_GROUPS for v in avail.values())),
            "n_groups_total": int(sum(v for v in avail.values() if v >= MIN_GROUPS)),
        }
    return out


def lomo_audit(path: Path) -> Dict[str, object]:
    """The classification cohort (LOMO) uses SMILES rather than a canonical ligand id."""
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    d = pd.read_csv(path, usecols=["SMILES_Ligands", "Metal", "Cell_line", "DOI"])
    d = d.dropna(subset=["SMILES_Ligands", "Metal"])
    span = d.groupby("SMILES_Ligands")["Metal"].nunique()
    pair_counts = {}
    for m1, m2 in itertools.combinations(METALS, 2):
        a = d[d["Metal"] == m1]
        b = d[d["Metal"] == m2]
        shared = set(a["SMILES_Ligands"]) & set(b["SMILES_Ligands"])
        pair_counts[f"{m1}-{m2}"] = int(len(shared))
    return {
        "status": "OK",
        "path": str(path.relative_to(PROJECT_ROOT)),
        "n_rows": int(len(d)),
        "n_ligands": int(d["SMILES_Ligands"].nunique()),
        "n_ligands_ge2_metals": int((span >= 2).sum()),
        "share_ligands_ge2_metals": round(float((span >= 2).mean()), 4),
        "shared_ligands_per_pair": pair_counts,
        "caveat": "SMILES strings are not canonicalised, so shared-ligand counts are a lower "
        "bound; this cohort also has no duration key, so no cell-matched Delta is defined.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--lomo", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_support_audit.json")
    args = ap.parse_args(argv)

    for p in (args.long,):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = load_long(args.long)
    endpoints = OrderedDict((ep, endpoint_audit(df, ep)) for ep in ENDPOINTS)

    scenarios = {
        ep: metal_out_scenarios(res) for ep, res in endpoints.items() if res["n_rows"] > 0
    }

    pot = endpoints.get("potency", {})
    summary = {
        "pairs_observed_potency": pot.get("observed_pairs", []),
        "pairs_unsupported_potency": pot.get("unsupported_pairs", []),
        "share_ligands_ge2_metals_potency": pot.get("share_ligands_ge2_metals"),
        "min_groups_required": MIN_GROUPS,
    }

    out = {
        "artifact": "msrs_support_audit",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L0 support audit (UP-10)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "unit": "(ligand_canonical, cell_line_key) measured under >=2 metals",
            "delta": "mean(y|metal_a) - mean(y|metal_b) within the unit",
            "noise": "pooled within-replicate variance per metal (same ligand, metal, cell, duration, DOI)",
            "interaction_share": "max(0, Var(Delta) - sampling noise) / Var(Delta)",
            "separability": "|mean Delta| / noise SD",
            "bootstrap": f"{N_BOOT} ligand-clustered resamples, seed {BOOT_SEED}",
            "min_groups": MIN_GROUPS,
            "gate": "every endpoint x every metal-out scenario reported; pairs below min_groups are UNSUPPORTED",
        },
        "summary": summary,
        "endpoints": endpoints,
        "metal_out_scenarios": scenarios,
        "lomo_classification": lomo_audit(args.lomo),
        "caveats": [
            "A unit is cell-matched but not DOI-matched: the same ligand measured under two metals "
            "in two different papers still counts, so part of Delta can be a paper effect. The "
            "interaction_share subtracts replicate noise, not between-paper drift.",
            "n_groups counts units, not independent ligands; a ligand measured in ten cell lines "
            "contributes ten units. n_ligands is reported next to it.",
            "Pairs below 20 units are marked UNSUPPORTED. That threshold is a rule of thumb for "
            "estimating a variance, not a power calculation.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"[UP-10] endpoints: " + ", ".join(
        f"{ep}({r['n_rows']} rows, {r['n_pairs_observed']}/{r['n_pairs_total']} pairs)"
        for ep, r in endpoints.items() if r["n_rows"] > 0
    ))
    for p in pot.get("pairs", []):
        print(f"[UP-10] {p['pair']:8s} n_groups={p.get('n_groups', 0):4d} "
              f"ligands={p.get('n_ligands', 0):4d} share={p.get('interaction_share')} "
              f"sep={p.get('separability')} {p['status']}")
    print(f"[UP-10] potency supported pairs: {summary['pairs_observed_potency']}")
    print(f"[UP-10] potency unsupported: {summary['pairs_unsupported_potency']}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
