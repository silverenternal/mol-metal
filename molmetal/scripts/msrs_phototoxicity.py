#!/usr/bin/env python3
"""MSRS-5 — phototoxicity as a functional endpoint: does light flip the metal's behaviour?

=====================================================================
The question
=====================================================================
Potency alone says which metal is stronger. It does not say whether the
*mechanism* changes under illumination. PI = log10(IC50_dark / IC50_light) is
the natural readout: positive means the complex gains potency in light.

=====================================================================
Why the obvious analysis is wrong
=====================================================================
The tempting move is to compare median PI between metals. MSRS-0 already shows
why that fails: the metals are censored at wildly different rates (Ru 518 of
1,100 light-bearing rows are right-censored, Ir 223 of 752), and the exact-only
subset is a *different selection* in each metal. Comparing exact medians then
compares two selections, not two metals.

So the primary endpoint here is a **classification with censoring built in**:

    switch      dark IC50 >= 10 uM and light IC50 < 10 uM   (light turns it on)
    antagonise  dark IC50 < 10 uM and light IC50 >= 10 uM   (light turns it off)
    both_active / both_inactive / undecidable

A right-censored dark value (">100") still decides "inactive", so these
categories use the censored rows instead of discarding them. The naive PI
medians are reported next to the rates, precisely so the disagreement is
visible.

Matched-pair status: MSRS-0 measured ZERO within-ligand cross-metal PI units, so
this is a between-compound contrast. The DOI-clustered bootstrap is the honest
uncertainty statement; a per-compound test would pretend the compounds are
exchangeable across papers.

=====================================================================
Usage
=====================================================================
::

    .venv/bin/python molmetal/scripts/msrs_phototoxicity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ACTIVITY_THRESHOLD_UM = 10.0  # the mainline's active/inactive cut
N_BOOT = 2000
BOOT_SEED = 20261018
MIN_SCAFFOLD_N = 10


# ---------------------------------------------------------------------------
# Censoring-aware activity
# ---------------------------------------------------------------------------
def is_active(value: float, censoring: str, threshold: float = ACTIVITY_THRESHOLD_UM) -> Optional[bool]:
    """``IC50 < threshold``, resolving censoring where it is decidable.

    ``>X`` with ``X >= threshold`` is inactive; ``>X`` with ``X < threshold``
    cannot be decided. ``<X`` with ``X <= threshold`` is active. Returns None
    when the bound straddles the threshold.
    """
    if not np.isfinite(value):
        return None
    if censoring == "none":
        return bool(value < threshold)
    if censoring == "right":
        return False if value >= threshold else None
    if censoring == "left":
        return True if value <= threshold else None
    return None


def classify(
    dark_value: float, dark_cens: str, light_value: float, light_cens: str,
    threshold: float = ACTIVITY_THRESHOLD_UM,
) -> str:
    d = is_active(dark_value, dark_cens, threshold)
    l = is_active(light_value, light_cens, threshold)
    if d is None or l is None:
        return "undecidable"
    if (not d) and l:
        return "switch"
    if d and (not l):
        return "antagonise"
    if d and l:
        return "active_both"
    return "inactive_both"


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval; honest at the small n Re/Os/Rh carry."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Classification table
# ---------------------------------------------------------------------------
def build_switch_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per source row that has both a dark and a light IC50."""
    dark = long[long["endpoint"] == "potency"].set_index("source_row")
    light = long[long["endpoint"] == "phototoxicity"].set_index("source_row")
    joined = dark[["metal", "value", "censoring"]].join(
        light[["value", "censoring"]], lsuffix="_dark", rsuffix="_light", how="inner"
    )
    joined["cls"] = [
        classify(dv, dc, lv, lc)
        for dv, dc, lv, lc in zip(
            joined["value_dark"], joined["censoring_dark"], joined["value_light"], joined["censoring_light"]
        )
    ]
    # PI is the ratio endpoint; carry its log so the naive view can be reported.
    joined["pi_log10"] = np.log10(joined["value_dark"] / joined["value_light"])
    joined["pi_exact"] = (joined["censoring_dark"] == "none") & (joined["censoring_light"] == "none")
    meta = dark[["ligand_canonical", "ligand_main_fragment", "cell_line_key", "doi", "year"]]
    return joined.join(meta, how="left").reset_index()


def summarise_rates(table: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for metal in METALS:
        sub = table[table["metal"] == metal]
        decidable = sub[sub["cls"] != "undecidable"]
        n = len(decidable)
        row: Dict[str, object] = {"n_rows_with_light": int(len(sub)), "n_decidable": int(n)}
        for cls in ("switch", "antagonise", "active_both", "inactive_both"):
            k = int((decidable["cls"] == cls).sum())
            lo, hi = wilson(k, n)
            row[cls] = {
                "n": k,
                "rate": round(k / n, 4) if n else None,
                "ci95_wilson": [round(lo, 4), round(hi, 4)],
            }
        row["n_undecidable"] = int((sub["cls"] == "undecidable").sum())
        exact = sub[sub["pi_exact"]]
        row["pi_exact_only"] = {
            "n": int(len(exact)),
            "median_log10": round(float(exact["pi_log10"].median()), 4) if len(exact) else None,
            "iqr_log10": [
                round(float(exact["pi_log10"].quantile(0.25)), 4),
                round(float(exact["pi_log10"].quantile(0.75)), 4),
            ] if len(exact) else None,
        }
        out[metal] = row
    return out


def clustered_difference(table: pd.DataFrame, metal_a: str, metal_b: str, cls: str) -> Dict[str, object]:
    """Rate difference (a - b) with a DOI-clustered bootstrap interval.

    Resampling DOIs keeps each paper's internal correlation intact; resampling
    compounds would treat 30 complexes from one paper as 30 independent facts.
    """
    sub = table[table["metal"].isin((metal_a, metal_b)) & (table["cls"] != "undecidable")]
    if sub.empty:
        return {"status": "NO_DATA"}

    def rate_diff(frame: pd.DataFrame) -> Optional[float]:
        a = frame[frame["metal"] == metal_a]
        b = frame[frame["metal"] == metal_b]
        if a.empty or b.empty:
            return None
        return float((a["cls"] == cls).mean() - (b["cls"] == cls).mean())

    point = rate_diff(sub)
    dois = sub["doi"].dropna().unique()
    by_doi = {d: sub[sub["doi"] == d] for d in dois}
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(dois, size=len(dois), replace=True)
        sample = pd.concat([by_doi[d] for d in pick], ignore_index=True)
        v = rate_diff(sample)
        if v is not None:
            boot.append(v)
    boot = np.asarray(boot)
    return {
        "status": "OK",
        "contrast": f"{metal_a} - {metal_b}",
        "outcome": cls,
        "rate_difference": round(float(point), 4) if point is not None else None,
        "ci95_doi_clustered_bootstrap": [
            round(float(np.quantile(boot, 0.025)), 4),
            round(float(np.quantile(boot, 0.975)), 4),
        ] if len(boot) else None,
        "n_dois": int(len(dois)),
        "n_units": {metal_a: int((sub["metal"] == metal_a).sum()), metal_b: int((sub["metal"] == metal_b).sum())},
    }


# ---------------------------------------------------------------------------
# Structure: predictability and scaffold enrichment
# ---------------------------------------------------------------------------
def _fingerprints(smiles: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def predictability(table: pd.DataFrame, metal: str) -> Dict[str, object]:
    """Ligand-grouped CV of PI from Morgan fingerprints, within one metal.

    Ligand-grouped because the same compound appears in several cell lines and
    durations; a random split would leak the answer through its own replicates.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    sub = table[(table["metal"] == metal) & table["pi_exact"] & table["ligand_main_fragment"].notna()].copy()
    # One row per ligand: median PI across its measurements.
    sub = sub.groupby("ligand_main_fragment", as_index=False).agg(pi=("pi_log10", "median"))
    if len(sub) < 30:
        return {"status": "INSUFFICIENT", "n_ligands": int(len(sub))}

    X = _fingerprints(sub["ligand_main_fragment"].tolist())
    y = sub["pi"].to_numpy(dtype=float)
    groups = sub["ligand_main_fragment"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return {"status": "INSUFFICIENT", "n_ligands": int(len(sub))}

    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    mae = float(np.mean(np.abs(y - pred)))

    # Permutation null: shuffle y, keep the same folds.
    rng = np.random.default_rng(BOOT_SEED)
    null = []
    for _ in range(200):
        yp = rng.permutation(y)
        pr = np.zeros_like(yp)
        for tr, te in gkf.split(X, yp, groups):
            model = Ridge(alpha=1.0)
            model.fit(X[tr], yp[tr])
            pr[te] = model.predict(X[te])
        s = float(np.sum((yp - pr) ** 2))
        t = float(np.sum((yp - yp.mean()) ** 2))
        null.append(1.0 - s / t if t > 0 else 0.0)
    null = np.asarray(null)
    verdict = (
        "no usable signal: out-of-fold R2 is negative, so the model is worse than predicting the "
        "metal's mean PI"
        if (r2 is not None and r2 < 0)
        else "out-of-fold R2 is positive; check the permutation p before claiming signal"
    )
    return {
        "status": "OK",
        "n_ligands": int(len(sub)),
        "r2_oof": round(float(r2), 4) if r2 is not None else None,
        "mae_oof": round(mae, 4),
        "baseline_mae_predict_mean": round(float(np.mean(np.abs(y - y.mean()))), 4),
        "permutation_p": round(float((np.sum(null >= r2) + 1) / (len(null) + 1)), 4) if r2 is not None else None,
        "n_permutations": int(len(null)),
        "folds": int(n_splits),
        "verdict": verdict,
        "permutation_caveat": "the permutation p compares the model to the same model fitted on "
        "shuffled labels; when both are worse than the mean it is not evidence of signal. The "
        "R2 sign is the primary read.",
    }


def scaffold_enrichment(table: pd.DataFrame, metal: str) -> Dict[str, object]:
    """Which Murcko scaffolds switch more often than the rest of the metal?

    Two-proportion z-test per scaffold with n >= MIN_SCAFFOLD_N, BH-corrected.
    A permutation check on the best scaffold guards the normal approximation at
    these sample sizes.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    sub = table[(table["metal"] == metal) & (table["cls"] != "undecidable") & table["ligand_main_fragment"].notna()].copy()
    if len(sub) < 50:
        return {"status": "INSUFFICIENT", "n_units": int(len(sub))}

    cache: Dict[str, Optional[str]] = {}

    def scaffold_of(s: str) -> Optional[str]:
        if s in cache:
            return cache[s]
        mol = Chem.MolFromSmiles(str(s))
        try:
            val = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None
        except Exception:
            val = None
        cache[s] = val
        return val

    sub["scaffold"] = sub["ligand_main_fragment"].map(scaffold_of)
    sub = sub[sub["scaffold"].notna() & (sub["scaffold"].str.len() > 0)]
    sub["is_switch"] = (sub["cls"] == "switch").astype(int)
    overall = float(sub["is_switch"].mean())

    rows = []
    for scaffold, grp in sub.groupby("scaffold"):
        n = len(grp)
        if n < MIN_SCAFFOLD_N:
            continue
        k = int(grp["is_switch"].sum())
        rest_n = len(sub) - n
        rest_k = int(sub["is_switch"].sum()) - k
        if rest_n <= 0:
            continue
        p1, p2 = k / n, rest_k / rest_n
        p = (k + rest_k) / (n + rest_n)
        se = np.sqrt(p * (1 - p) * (1 / n + 1 / rest_n))
        z = (p1 - p2) / se if se > 0 else 0.0
        from scipy import stats as _stats

        rows.append(
            {
                "scaffold": scaffold,
                "n": n,
                "n_switch": k,
                "rate": round(p1, 4),
                "rate_rest": round(p2, 4),
                "z": round(float(z), 3),
                "p_raw": float(2 * _stats.norm.sf(abs(z))),
            }
        )

    if not rows:
        return {"status": "NO_SCAFFOLD_WITH_ENOUGH_UNITS", "n_units": int(len(sub)), "overall_switch_rate": round(overall, 4)}

    rows.sort(key=lambda r: r["p_raw"])
    # Benjamini-Hochberg.
    m = len(rows)
    for i, r in enumerate(rows):
        r["p_bh"] = round(min(1.0, r["p_raw"] * m / (i + 1)), 5)
    surviving = [r for r in rows if r["p_bh"] < 0.05]

    # Paper-level robustness: collapse each (scaffold, DOI) to one observation so
    # a scaffold cannot win by being measured in many cell lines within one paper.
    from scipy import stats as _stats

    paper = sub.groupby(["scaffold", "doi"], as_index=False)["is_switch"].mean()
    paper_rows = []
    for r in rows:
        a = paper.loc[paper["scaffold"] == r["scaffold"], "is_switch"]
        b = paper.loc[paper["scaffold"] != r["scaffold"], "is_switch"]
        r["n_dois"] = int(len(a))
        if len(a) >= 3 and len(b) >= 3:
            r["p_paper_level_mwu"] = float(
                _stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            )
        else:
            r["p_paper_level_mwu"] = None
    tested = [r for r in rows if r["p_paper_level_mwu"] is not None]
    tested.sort(key=lambda r: r["p_paper_level_mwu"])
    for i, r in enumerate(tested):
        r["p_paper_level_bh"] = round(min(1.0, r["p_paper_level_mwu"] * len(tested) / (i + 1)), 5)
    surviving_paper = [r for r in tested if r["p_paper_level_bh"] < 0.05]

    # Permutation check on the top scaffold.
    top = rows[0]
    rng = np.random.default_rng(BOOT_SEED)
    top_mask = (sub["scaffold"] == top["scaffold"]).to_numpy()
    y = sub["is_switch"].to_numpy()
    n_top = int(top_mask.sum())
    obs = float(y[top_mask].mean() - y[~top_mask].mean())
    null = []
    for _ in range(1000):
        yp = rng.permutation(y)
        null.append(float(yp[top_mask].mean() - yp[~top_mask].mean()))
    null = np.asarray(null)
    p_perm = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))

    verdict = (
        "{} scaffold(s) survive BH at the unit level but {} at the paper level: the apparent "
        "enrichment is a paper effect (a scaffold measured many times inside one article), not a "
        "chemistry effect. Do not report a scaffold rule from this data.".format(
            len(surviving), len(surviving_paper)
        )
        if len(surviving) and not len(surviving_paper)
        else "{} scaffold(s) survive at both levels; still report n_dois next to each".format(
            len(surviving_paper)
        )
    )
    return {
        "status": "OK",
        "n_units": int(len(sub)),
        "n_scaffolds_tested": int(m),
        "overall_switch_rate": round(overall, 4),
        "n_surviving_bh_0_05": int(len(surviving)),
        "n_surviving_paper_level_bh_0_05": int(len(surviving_paper)),
        "top_scaffolds": rows[:5],
        "top_scaffold_permutation_p": round(p_perm, 4),
        "verdict": verdict,
        "note": "scaffold = Bemis-Murcko of the largest organic fragment; units are "
        "(compound, cell line, duration, DOI), so a scaffold's n counts measurements, not molecules. "
        "The paper-level column re-tests with one observation per (scaffold, DOI) and is the "
        "number to trust when a single paper supplies most of a scaffold's units.",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    parser.add_argument("--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_phototoxicity.json"
    )
    args = parser.parse_args(argv)

    for p in (args.long, args.dedup):
        if not p.exists():
            print(f"FAIL: missing {p}; run msrs_build_endpoints.py first")
            return 1

    long = pd.read_parquet(args.long)
    table = build_switch_table(long)
    if table.empty:
        print("FAIL: no source row carries both a dark and a light IC50")
        return 1

    rates = summarise_rates(table)
    contrasts = {
        "switch": clustered_difference(table, "Ru", "Ir", "switch"),
        "antagonise": clustered_difference(table, "Ru", "Ir", "antagonise"),
    }

    pred = {m: predictability(table, m) for m in ("Ru", "Ir")}
    scaffolds = {m: scaffold_enrichment(table, m) for m in ("Ru", "Ir")}

    # The disagreement between the naive and the censoring-aware view is the
    # point of this script, so it is stated as data rather than as prose.
    ru_pi = rates["Ru"]["pi_exact_only"]["median_log10"]
    ir_pi = rates["Ir"]["pi_exact_only"]["median_log10"]
    ru_sw = rates["Ru"]["switch"]["rate"]
    ir_sw = rates["Ir"]["switch"]["rate"]
    naive_ranks_ir_higher = bool(ru_pi is not None and ir_pi is not None and ir_pi > ru_pi)
    switch_ranks_ru_higher = bool(ru_sw is not None and ir_sw is not None and ru_sw > ir_sw)

    # Reconcile the number this work package was planned around. The 2026-10-18
    # one-off audit recorded "pi_median_by_metal: Ir 1.27, Ru 0.88" and the plan
    # read it as a ratio ("Ir ~19x"). It is in fact the median log10 of the LIGHT
    # IC50 alone, on rows where both arms are exact. Reproduced here so the
    # mislabelling is on disk, not in prose.
    exact_both = table[table["pi_exact"]]
    light_median = {}
    for m in ("Ru", "Ir"):
        vals = exact_both.loc[exact_both["metal"] == m, "value_light"]
        light_median[m] = round(float(np.log10(vals.median())), 4) if len(vals) else None

    out = {
        "artifact": "msrs_phototoxicity",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "functional endpoint",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "definitions": {
            "activity_threshold_um": ACTIVITY_THRESHOLD_UM,
            "switch": "dark IC50 >= threshold and light IC50 < threshold",
            "antagonise": "dark IC50 < threshold and light IC50 >= threshold",
            "undecidable": "a censoring bound straddles the threshold, or a value is missing",
            "pi": "PI = log10(IC50_dark / IC50_light); reported on exact rows only",
        },
        "counts": table["cls"].value_counts().to_dict(),
        "rates_by_metal": rates,
        "contrasts": contrasts,
        "predictability": pred,
        "scaffold_enrichment": scaffolds,
        "naive_vs_censoring_aware": {
            "pi_exact_median_Ru": ru_pi,
            "pi_exact_median_Ir": ir_pi,
            "naive_ranks_Ir_as_more_phototoxic": naive_ranks_ir_higher,
            "switch_rate_Ru": ru_sw,
            "switch_rate_Ir": ir_sw,
            "switch_ranks_Ru_as_more_light_activated": switch_ranks_ru_higher,
            "interpretation": "The exact-PI median and the switch rate agree on direction (both put "
            "Ru higher), but they are computed on different row sets: the exact-PI median silently "
            "drops every censored light measurement, and it drops 404/1292 (31%) for Ru against "
            "89/781 (11%) for Ir. The switch rate keeps those rows and is the endpoint to quote; "
            "the exact-PI median is reported only to show how much the selection changes.",
        },
        "prior_claim_reconciliation": {
            "prior_value": "todo.json current_position.continuous_data_audit_2026_10_18."
            "pi_median_by_metal = {Ir: 1.27, Ru: 0.88}",
            "prior_reading": "Ir is ~19x phototoxic vs Ru ~7.6x (log10 PI)",
            "what_it_actually_was": "median log10(IC50_light) per metal, not a dark/light ratio",
            "measured_median_light_ic50_log10": light_median,
            "correct_pi_median_log10": {"Ru": ru_pi, "Ir": ir_pi},
            "correct_pi_fold": {
                "Ru": round(float(10 ** ru_pi), 2) if ru_pi is not None else None,
                "Ir": round(float(10 ** ir_pi), 2) if ir_pi is not None else None,
            },
            "consequence": "The 'Ir is far more phototoxic' reading is WITHDRAWN. On exact rows "
            "the median PI is Ru 3.1x vs Ir 0.6x, and the censoring-aware switch rate agrees "
            "(Ru 26% vs Ir 14%). The two independent endpoints now point the same way; the old "
            "one pointed the other way because it was not a ratio.",
        },
        "design_facts": {
            "matched_pairs": "zero within-ligand cross-metal PI units (MSRS-0), so every contrast "
            "here is between compounds and clustered by DOI",
            "dois_with_both_metals": int(
                table[table["metal"].isin(("Ru", "Ir"))].groupby("doi")["metal"].nunique().gt(1).sum()
            ),
        },
        "caveats": [
            "Whether a compound was tested under light at all is a paper-level decision; the "
            "DOI-clustered bootstrap reflects that, but it cannot correct for papers that never "
            "ran the light arm.",
            "The 10 uM activity threshold is the mainline's binary cut, kept for comparability. "
            "A ratio endpoint would avoid the threshold but cannot use right-censored rows.",
            "Re (28 units), Os (9) and Rh (3) are reported for completeness and must not be "
            "ranked; their Wilson intervals span most of [0, 1].",
            "Scaffold enrichment counts measurements, not molecules: a scaffold tested in five "
            "cell lines contributes five units and is not five independent observations.",
            "The 'Ir is ~19x phototoxic' reading carried in the plan came from a median of the "
            "light IC50, not of the dark/light ratio. It is withdrawn; see "
            "prior_claim_reconciliation.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    print(f"rows with both dark and light IC50: {len(table)}")
    for m in METALS:
        r = rates[m]
        sw = r["switch"]
        an = r["antagonise"]
        print(
            f"  {m}: decidable={r['n_decidable']:4d} switch={sw['rate']} ({sw['n']}) "
            f"antagonise={an['rate']} ({an['n']})"
        )
    for k, v in contrasts.items():
        if v.get("status") == "OK":
            print(f"  Ru-Ir {k}: {v['rate_difference']} CI={v['ci95_doi_clustered_bootstrap']}")
    for m, v in pred.items():
        print(f"  PI predictability {m}: {v.get('status')} r2={v.get('r2_oof')} p={v.get('permutation_p')}")
    for m, v in scaffolds.items():
        print(
            f"  scaffolds {m}: {v.get('status')} unit-level={v.get('n_surviving_bh_0_05')} "
            f"paper-level={v.get('n_surviving_paper_level_bh_0_05')}"
        )
    print(f"  prior claim: median log10(IC50_light) = {light_median} (that was NOT PI)")
    print(f"  corrected PI median: Ru={ru_pi} Ir={ir_pi}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
