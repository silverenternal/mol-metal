#!/usr/bin/env python3
"""MSRS-1 — variance decomposition and the ligand x metal interaction test.

=====================================================================
The question this answers
=====================================================================
MSRS's decision layer recommends a metal *per ligand*. That is only meaningful
if the best metal actually depends on the ligand. If every ligand prefers the
same metal, the recommendation collapses to a constant and the whole apparatus
is theatre.

So this script splits the variance of ``y = log10 IC50`` into

    metal (fixed) | cell line (fixed) | DOI (random) | ligand|DOI (random) | noise

and then tests the one term that decides the paper's angle:

    does the metal effect vary across ligands?

=====================================================================
Why the naive test is wrong here
=====================================================================
1. **Metal is 90% nested in DOI.** Only 110 of 1,801 DOIs in the potency
   cohort contain more than one metal. A "metal effect" estimated on the full
   cohort is therefore mostly a *paper* effect: different labs, different
   assays, different cell panels. The interaction test is run only on the
   within-DOI matched subcohort, where the comparison is clean, and the
   full-cohort metal R^2 is labelled as a between-paper contrast.

2. **Delta is not free.** For a matched unit, Var(Delta) = Var(true effect) +
   sigma2_m1 + sigma2_m2. A raw Delta SD of 0.45 log sounds enormous until you
   notice that Ru replicates alone disagree with sigma = 0.22. The interaction
   share is the part above the measured noise floor, and the noise floor is
   estimated from 309 replicate groups, not assumed.

3. **Units are clustered by ligand.** One ligand contributes several
   (cell line, DOI, duration) units, so an F-test that treats units as
   independent is anti-conservative. A ligand-clustered bootstrap CI is
   reported next to it and is the number to quote.

=====================================================================
Kill criterion (from todo.json)
=====================================================================
If the interaction share is below 5% and L3's regret improvement is below
0.05 log, drop "ligand-specific recommendation" and reframe the paper around
metal ranking plus the phototoxicity axis. This script produces the first half
of that test.

=====================================================================
Usage
=====================================================================
::

    .venv/bin/python molmetal/scripts/msrs_variance_decomposition.py
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
from scipy import stats

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

# A pair needs this many clean units before its interaction is worth estimating.
MIN_UNITS_FOR_PAIR = 30
N_BOOT = 2000
BOOT_SEED = 20261018


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------
def noise_floor(long: pd.DataFrame, endpoint: str) -> Dict[str, object]:
    """Pooled within-group variance from genuine replicate measurements.

    A replicate group is the same (ligand, metal, cell line, duration, DOI).
    ``sum((x - mean)^2) / sum(n - 1)`` is the unbiased pooled estimate; unlike
    ``mean(sd^2)`` it does not let one three-replicate group outvote ten pairs.
    """
    sub = long[(long["endpoint"] == endpoint) & (long["censoring"] == "none")]
    keys = ["ligand_canonical", "metal", "cell_line_key", "time_h", "doi"]
    g = sub.groupby(keys, dropna=False)["value_log10"]
    n = g.size()
    var = g.var(ddof=1)
    rep = n >= 2

    per_metal: Dict[str, object] = {}
    for metal in METALS:
        sel = rep & (n.index.get_level_values("metal") == metal)
        df = int((n[sel] - 1).sum())
        ss = float(((n[sel] - 1) * var[sel]).sum())
        per_metal[metal] = {
            "n_groups": int(sel.sum()),
            "df": df,
            "sigma2": round(ss / df, 6) if df else None,
            "sigma": round(float(np.sqrt(ss / df)), 4) if df else None,
        }
    df_all = int((n[rep] - 1).sum())
    ss_all = float(((n[rep] - 1) * var[rep]).sum())
    return {
        "definition": "same (ligand, metal, cell line, duration, DOI); pooled within-group variance",
        "n_groups_total": int(len(n)),
        "n_replicate_groups": int(rep.sum()),
        "n_replicate_rows": int(n[rep].sum()),
        "pooled": {
            "df": df_all,
            "sigma2": round(ss_all / df_all, 6) if df_all else None,
            "sigma": round(float(np.sqrt(ss_all / df_all)), 4) if df_all else None,
        },
        "per_metal": per_metal,
        "group_sd_quantiles": {
            q: round(float(var[rep].pow(0.5).quantile(float(q))), 4)
            for q in ("0.25", "0.5", "0.75", "0.9")
        },
        "caveat": "Rh has {} replicate groups and Re {}; their sigma is barely constrained "
        "and must not be used to declare an interaction absent.".format(
            per_metal["Rh"]["n_groups"], per_metal["Re"]["n_groups"]
        ),
    }


# ---------------------------------------------------------------------------
# Fixed effects + nested variance components
# ---------------------------------------------------------------------------
def _ols_residual(y: np.ndarray, dummies: np.ndarray) -> Tuple[np.ndarray, int]:
    X = np.column_stack([np.ones(len(y)), dummies])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta, X.shape[1]


def _dummies(series: pd.Series) -> np.ndarray:
    return pd.get_dummies(series.astype(str), drop_first=True).to_numpy(dtype=float)


def fixed_effect_r2(df: pd.DataFrame, y_col: str, specs: Sequence[Sequence[str]]) -> Dict[str, object]:
    y = df[y_col].to_numpy(dtype=float)
    total = float(np.var(y))
    out: Dict[str, object] = {}
    for spec in specs:
        if len(spec) == 1:
            d = _dummies(df[spec[0]])
        else:
            d = np.column_stack([_dummies(df[c]) for c in spec])
        resid, k = _ols_residual(y, d)
        out["+".join(spec)] = {
            "n_params": int(k),
            "r2": round(float(1.0 - np.var(resid) / total), 4),
        }
    return out


def nested_variance_components(
    df: pd.DataFrame, y_col: str, doi_col: str = "doi", ligand_col: str = "ligand_canonical"
) -> Dict[str, object]:
    """Nested ANOVA: DOI > ligand|DOI > residual, on fixed-effect residuals.

    Standard unbalanced one-way nested decomposition:
    ``sigma2_ligand = (MS_lig - MS_err) / n0_lig`` and
    ``sigma2_doi = (MS_doi - MS_lig) / n0_doi``. The two random factors are
    treated as nested because only 379 of 5,965 ligands appear under more than
    one DOI; a crossed model is not identified for the other 94%.
    """
    y = df[y_col].to_numpy(dtype=float)
    N = len(y)
    doi = df[doi_col].astype(str).to_numpy()
    lig = df[ligand_col].astype(str).to_numpy()

    grand = float(y.mean())
    doi_frame = pd.DataFrame({"doi": doi, "y": y})
    g_doi = doi_frame.groupby("doi")["y"]
    n_i = g_doi.size().to_numpy(dtype=float)
    mean_i = g_doi.mean().to_numpy(dtype=float)
    ss_doi = float(np.sum(n_i * (mean_i - grand) ** 2))
    k1 = len(n_i)

    unit = pd.DataFrame({"doi": doi, "lig": lig, "y": y})
    g_unit = unit.groupby(["doi", "lig"])["y"]
    n_ij = g_unit.size().to_numpy(dtype=float)
    mean_ij = g_unit.mean().to_numpy(dtype=float)
    # Map each (doi, lig) back to its DOI mean.
    unit_index = g_unit.mean().index.to_frame(index=False)
    doi_mean_map = dict(zip(doi_frame.groupby("doi")["y"].mean().index, mean_i))
    ref = np.array([doi_mean_map[d] for d in unit_index["doi"]])
    ss_lig = float(np.sum(n_ij * (mean_ij - ref) ** 2))
    k2 = len(n_ij)

    merged = unit.merge(
        pd.DataFrame({"doi": unit_index["doi"], "lig": unit_index["lig"], "m": mean_ij}),
        on=["doi", "lig"],
        how="left",
    )
    ss_err = float(np.sum((merged["y"] - merged["m"]) ** 2))

    df_doi = k1 - 1
    df_lig = k2 - k1
    df_err = N - k2
    ms_doi = ss_doi / df_doi
    ms_lig = ss_lig / df_lig
    ms_err = ss_err / df_err

    n0_lig = (N - float(np.sum(n_ij**2)) / N) / df_lig
    n0_doi = (N - float(np.sum(n_i**2)) / N) / df_doi

    s2_err = max(0.0, ms_err)
    s2_lig = max(0.0, (ms_lig - ms_err) / n0_lig)
    s2_doi = max(0.0, (ms_doi - ms_lig) / n0_doi)
    total = s2_err + s2_lig + s2_doi
    total_y = float(np.var(y, ddof=1))

    return {
        "n_rows": int(N),
        "n_dois": int(k1),
        "n_ligand_within_doi": int(k2),
        "ms": {"doi": round(float(ms_doi), 6), "ligand": round(float(ms_lig), 6), "residual": round(float(ms_err), 6)},
        "sigma2": {
            "doi": round(float(s2_doi), 6),
            "ligand_within_doi": round(float(s2_lig), 6),
            "residual": round(float(s2_err), 6),
        },
        "share_of_random_variance": {
            "doi": round(float(s2_doi / total), 4) if total else None,
            "ligand_within_doi": round(float(s2_lig / total), 4) if total else None,
            "residual": round(float(s2_err / total), 4) if total else None,
        },
        "share_of_total_y_variance": round(float(total / total_y), 4) if total_y else None,
        "caveat": "components are sequential (DOI then ligand); with an unbalanced design a "
        "crossed model would split the same covariance differently",
    }


# ---------------------------------------------------------------------------
# Interaction test on the matched subcohort
# ---------------------------------------------------------------------------
def _pair_units(dedup: pd.DataFrame, endpoint: str, m1: str, m2: str) -> pd.DataFrame:
    """Units measured under BOTH metals in the SAME DOI and duration."""
    sub = dedup[(dedup["endpoint"] == endpoint) & (~dedup["all_censored"]) & dedup["metal"].isin((m1, m2))]
    keys = ["ligand_canonical", "cell_line_key", "doi", "time_h"]
    a = sub[sub["metal"] == m1].groupby(keys, dropna=False)["value_log10"].median().rename("y1")
    b = sub[sub["metal"] == m2].groupby(keys, dropna=False)["value_log10"].median().rename("y2")
    out = pd.concat([a, b], axis=1, join="inner").reset_index()
    out["delta"] = out["y1"] - out["y2"]
    return out


def _cell_adjust(units: pd.DataFrame, min_per_cell: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Residualise Delta on cell line; returns (residual, mask of rows kept)."""
    counts = units["cell_line_key"].value_counts()
    keep_cells = set(counts[counts >= min_per_cell].index)
    mask = units["cell_line_key"].isin(keep_cells).to_numpy()
    if mask.sum() < 10 or units.loc[mask, "cell_line_key"].nunique() < 2:
        return units["delta"].to_numpy(dtype=float), np.ones(len(units), dtype=bool)
    d = _dummies(units.loc[mask, "cell_line_key"].reset_index(drop=True))
    resid, _ = _ols_residual(units.loc[mask, "delta"].to_numpy(dtype=float), d)
    return resid, mask


def _min_detectable_share(n: int, df_noise: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest interaction share an F-test of Var(Delta) can detect.

    Under the alternative the numerator is inflated by ``1 / (1 - s)``, so the
    statistic is ``F ~ (1 / (1 - s)) * F(df1, df2)`` -- a scaled central F, not
    a noncentral one. Power is therefore ``P(F(df1, df2) > crit * (1 - s))``.

    Note the consequence: power is capped by ``df_noise``, the precision of the
    noise floor, not by the number of units. With 100 noise df no sample size
    can resolve a 5% variance share, because the critical value stays near 1.24.
    """
    df1 = max(1, n - 1)
    crit = stats.f.ppf(1.0 - alpha, df1, df_noise)

    def achieved(s: float) -> float:
        return float(stats.f.sf(crit * (1.0 - s), df1, df_noise))

    if achieved(0.0) >= power:
        return 0.0
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if achieved(mid) < power:
            lo = mid
        else:
            hi = mid
    return hi


def interaction_test(
    dedup: pd.DataFrame,
    endpoint: str,
    m1: str,
    m2: str,
    sigma2: Dict[str, float],
    df_noise: int,
) -> Dict[str, object]:
    """Is Var(Delta) larger than the noise the two metals' replicates predict?"""
    units = _pair_units(dedup, endpoint, m1, m2)
    if len(units) < MIN_UNITS_FOR_PAIR:
        return {
            "pair": f"{m1}-{m2}",
            "status": "INSUFFICIENT",
            "n_units": int(len(units)),
            "min_required": MIN_UNITS_FOR_PAIR,
        }

    resid, mask = _cell_adjust(units)
    n = len(resid)
    var_obs = float(np.var(resid, ddof=1))
    noise = float(sigma2[m1] + sigma2[m2])
    inter = max(0.0, var_obs - noise)
    share = inter / var_obs if var_obs > 0 else None

    # F-test: Var(Delta) / noise ~ F(n-1, df_noise). Ignores ligand clustering,
    # so it is reported as secondary; the bootstrap below is the quoted interval.
    F = var_obs / noise if noise > 0 else np.nan
    p_f = float(stats.f.sf(F, n - 1, df_noise)) if np.isfinite(F) else None

    # Smallest interaction share this pair could detect at 80% power. Without
    # it, "share = 0" reads as evidence of no interaction when it is often just
    # an underpowered pair.
    min_detectable = _min_detectable_share(n, df_noise)

    # Ligand-clustered bootstrap of the interaction share.
    rng = np.random.default_rng(BOOT_SEED)
    lig = units.loc[mask, "ligand_canonical"].to_numpy()
    uniq = np.unique(lig)
    idx_by_lig = {u: np.flatnonzero(lig == u) for u in uniq}
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_lig[u] for u in pick])
        v = float(np.var(resid[sel], ddof=1))
        boot.append(max(0.0, v - noise) / v if v > 0 else 0.0)
    boot = np.asarray(boot)
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))

    # Per-ligand mean Delta: the estimand the recommendation layer actually uses.
    per_lig = units.loc[mask].groupby("ligand_canonical")["delta"].agg(["mean", "size"])
    return {
        "pair": f"{m1}-{m2}",
        "status": "OK",
        "n_units": int(n),
        "n_ligands": int(per_lig.shape[0]),
        "mean_delta": round(float(units.loc[mask, "delta"].mean()), 4),
        "median_delta": round(float(units.loc[mask, "delta"].median()), 4),
        "sd_delta_units": round(float(np.sqrt(var_obs)), 4),
        "sd_delta_per_ligand": round(float(per_lig["mean"].std(ddof=1)), 4),
        "sign_rate_positive": round(float((units.loc[mask, "delta"] > 0).mean()), 4),
        "noise_var": round(noise, 6),
        "noise_sd": round(float(np.sqrt(noise)), 4),
        "interaction_var": round(inter, 6),
        "interaction_share": round(float(share), 4) if share is not None else None,
        "interaction_share_ci95_clustered_bootstrap": [round(ci[0], 4), round(ci[1], 4)],
        "F_vs_noise": round(float(F), 3) if np.isfinite(F) else None,
        "p_one_sided_F": p_f,
        "min_detectable_share_80pct_power": round(float(min_detectable), 4),
        "detected": bool(
            p_f is not None and p_f < 0.05 and (share or 0.0) >= 0.05
        ),
        "test_caveats": [
            "the F-test treats units as independent; ligands contribute several units, so it is "
            "anti-conservative. Quote the clustered bootstrap interval.",
            "the noise floor is estimated, not known; for Rh (4 replicate groups) it is barely "
            "constrained and the test has almost no power to reject.",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    parser.add_argument("--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_variance_decomposition.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.long):
        if not p.exists():
            print(f"FAIL: missing {p}; run msrs_build_endpoints.py first")
            return 1

    dedup = pd.read_parquet(args.dedup)
    long = pd.read_parquet(args.long)

    noise = {ep: noise_floor(long, ep) for ep in ENDPOINTS}

    decomposition: Dict[str, object] = {}
    fixed_r2: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sub = dedup[(dedup["endpoint"] == ep) & (~dedup["all_censored"])].copy()
        fixed_r2[ep] = fixed_effect_r2(sub, "value_log10", [("metal",), ("cell_line_key",), ("metal", "cell_line_key")])
        decomposition[ep] = nested_variance_components(sub, "value_log10")

    interactions: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sigma2: Dict[str, float] = {
            metal: float(noise[ep]["per_metal"][metal]["sigma2"] or 0.0) for metal in METALS
        }
        pair_results = {}
        for i, m1 in enumerate(METALS):
            for m2 in METALS[i + 1 :]:
                # Noise df is pair-specific: Rh's 8 df must not borrow Ru's 275.
                df_pair = int(noise[ep]["per_metal"][m1]["df"]) + int(noise[ep]["per_metal"][m2]["df"])
                pair_results[f"{m1}-{m2}"] = interaction_test(dedup, ep, m1, m2, sigma2, df_pair)
        interactions[ep] = pair_results

    # ---- the decision the kill criterion asks for -------------------------
    pot = interactions["potency"]
    estimable = {k: v for k, v in pot.items() if v.get("status") == "OK"}
    threshold = 0.05

    def _ci(k: str) -> Tuple[float, float]:
        lo, hi = estimable[k]["interaction_share_ci95_clustered_bootstrap"]
        return float(lo), float(hi)

    # The criterion is about the *share*, so it is read off the CI, not the
    # point estimate: "established" needs the lower bound above 5%, and "killed"
    # needs the upper bound below it. Anything in between is undetermined.
    established = [k for k in estimable if _ci(k)[0] > threshold]
    killed = [k for k in estimable if _ci(k)[1] < threshold]
    undetermined = [k for k in estimable if k not in established and k not in killed]

    decision = {
        "kill_threshold_interaction_share": threshold,
        "estimable_pairs": sorted(estimable),
        "pairs_interaction_established": sorted(established),
        "pairs_interaction_killed": sorted(killed),
        "pairs_undetermined": sorted(undetermined),
        "pair_detail": {
            k: {
                "n_units": v["n_units"],
                "interaction_share": v["interaction_share"],
                "ci95": v["interaction_share_ci95_clustered_bootstrap"],
                "min_detectable_share_80pct_power": v["min_detectable_share_80pct_power"],
                "detected": v["detected"],
            }
            for k, v in estimable.items()
        },
        "power_note": "A 5% variance share is below the detection limit of every pair: at n=269 "
        "the minimum detectable share at 80% power is {}. So 'not detected' means 'share is below "
        "the minimum detectable value', never 'share is below 5%'. The criterion is therefore "
        "applied to the CI bounds, and the paper must quote the detection limit next to any null "
        "result.".format(estimable["Ru-Os"]["min_detectable_share_80pct_power"] if "Ru-Os" in estimable else "n/a"),
        "second_half_of_kill_criterion": "L3 top-1 regret improvement vs global-best-metal "
        "(MSRS-3/6, not measured here)",
        "verdict": None,
    }
    if not established and killed:
        decision["verdict"] = (
            "interaction is excluded on {} (CI upper below 5%): a ligand-specific recommendation "
            "is not supported; reframe around metal ranking and the phototoxicity axis".format(
                ", ".join(sorted(killed))
            )
        )
    elif established and not undetermined:
        decision["verdict"] = (
            "interaction is established on every estimable pair: proceed to MSRS-2/3, but the "
            "regret gate still has to pass"
        )
    elif established:
        decision["verdict"] = (
            "interaction is established on {} and undetermined on {}; the undetermined pairs have "
            "a minimum detectable share of {} and their null result is a power statement, not "
            "evidence of absence".format(
                ", ".join(sorted(established)),
                ", ".join(sorted(undetermined)),
                ", ".join(
                    "{}={}".format(k, estimable[k]["min_detectable_share_80pct_power"])
                    for k in sorted(undetermined)
                ),
            )
        )
    else:
        decision["verdict"] = (
            "no pair is decisive: interaction is neither established nor excluded, so the "
            "recommendation story cannot be closed on this data"
        )

    out = {
        "artifact": "msrs_variance_decomposition",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L2 structural",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "long_parquet": str(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "noise_floor": noise,
        "fixed_effect_r2": fixed_r2,
        "variance_components": decomposition,
        "interaction": interactions,
        "decision": decision,
        "design_facts": {
            "metal_nested_in_doi": "110 of 1,801 potency DOIs contain >1 metal (2,217 of 21,406 units, "
            "10.4%). The full-cohort metal R^2 is therefore mostly a between-paper contrast; only the "
            "matched subcohort identifies a within-paper metal effect.",
            "ligand_across_dois": "379 of 5,965 potency ligands appear under more than one DOI, which is "
            "why ligand is modelled as nested within DOI rather than crossed.",
        },
        "caveats": [
            "Var(Delta) is compared to a *measured* noise floor, not to zero. A Delta that looks large "
            "can be entirely replicate disagreement: Ru sigma = 0.22 log on its own.",
            "The noise floor comes from 309 replicate groups and is itself uncertain; the interaction "
            "share inherits that uncertainty, which is why every pair carries a clustered bootstrap CI.",
            "Phototoxicity has no cross-metal matched unit (MSRS-0), so its interaction is not estimable "
            "at all and its entry is INSUFFICIENT by construction, not by low power.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    print(f"noise floor (potency) sigma = {noise['potency']['pooled']['sigma']} "
          f"from {noise['potency']['n_replicate_groups']} replicate groups")
    for ep in ENDPOINTS:
        c = decomposition[ep]["share_of_random_variance"]
        print(f"  {ep:16s} doi={c['doi']} ligand|doi={c['ligand_within_doi']} resid={c['residual']}")
    print("interaction (potency):")
    for k, v in pot.items():
        if v.get("status") == "OK":
            print(f"  {k:7s} n={v['n_units']:4d} share={v['interaction_share']:.3f} "
                  f"CI={v['interaction_share_ci95_clustered_bootstrap']} "
                  f"min_detect={v['min_detectable_share_80pct_power']:.3f} "
                  f"detected={v['detected']}")
        else:
            print(f"  {k:7s} {v['status']} (n={v['n_units']})")
    print(f"decision: established={decision['pairs_interaction_established']} "
          f"undetermined={decision['pairs_undetermined']} "
          f"killed={decision['pairs_interaction_killed']}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
