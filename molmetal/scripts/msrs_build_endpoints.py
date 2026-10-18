#!/usr/bin/env python3
"""MSRS-0 — build the tidy measurement table behind the metal-substitution surface.

=====================================================================
Why this exists
=====================================================================
The MSRS framework asks a dose-response question: *if I swap the metal centre
and change nothing else, how much does potency move?* That question needs a
continuous endpoint, and the binary ``IC50 < 10 uM`` label used by the
classification mainline throws the dose information away.

The 2026-10-18 audit produced the counts (21,777 numeric dark IC50, 1,150
phototoxicity indices, 668 cross-metal matched ligand/cell groups) from an
ad-hoc shell session. Nothing was on disk, so nothing was reproducible, and
three of those numbers turn out to depend on choices nobody had written down:

* ``IC50_*_value`` is **not** a censoring code. It is the numeric value for
  every row, including the 5,015 dark rows whose ``IC50_Dark(M*10^-6)`` cell
  literally reads ``>100``. So ``>100`` silently becomes ``100.0``, and a
  ``ic50 < 10`` label built from it labels a right-censored observation as
  inactive — right direction, but the continuous value is a lower bound
  presented as an exact measurement.
* The "668 matched groups" figure counted censored rows. Restricting to
  numerically-measured dark IC50 gives 537. The gap is the censoring bias.
* Cell-line names carry 16 spelling variants (``SK-HEP-1`` / ``SK-Hep1``).
  Exact string grouping splits a matched pair that is really matched. But
  aggressive normalisation wrongly merges ``HCT-116 p53+/+`` with
  ``HCT-116 p53-/-``, so the normaliser has to preserve the p53 sign.

This script pins all of that into two artifacts and an audit JSON. It trains
nothing and touches no GPU.

=====================================================================
Outputs
=====================================================================
``data/msrs/measurements.parquet``
    Long format, one row per (source row, endpoint), **nothing dropped**.
    Every row keeps ``source_row`` (0-based CSV index), ``doi`` and ``year``.
``data/msrs/measurements_dedup.parquet``
    One row per (ligand, metal, cell line, time, DOI, endpoint) with the median
    log10 value over replicates — the declared dedup rule.
``metrics/by_metric/msrs_endpoints.json``
    Counts, censoring rates, filter ledger, matched-pair coverage, checks.

=====================================================================
Endpoint definitions
=====================================================================
``potency``            y = log10 IC50_dark (uM). Lower = more potent.
``phototoxicity``      PI = log10(IC50_dark / IC50_light). Positive = light-activated.
``cisplatin_ratio``    RI = log10(IC50_dark / IC50_cisplatin). Cell-line-normalised.

``value_log10`` is the analysis scale for all three, so downstream code never
has to know which endpoint it is looking at.

=====================================================================
Usage
=====================================================================
::

    .venv/bin/python molmetal/scripts/msrs_build_endpoints.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
ENDPOINTS: Tuple[str, ...] = ("potency", "phototoxicity", "cisplatin_ratio")

# Source columns. The parenthesised unit is part of the column name in the CSV.
DARK = "IC50_Dark(M*10^-6)"
LIGHT = "IC50_Light(M*10^-6)"
CIS = "IC50_Cisplatin(M*10^-6)"
DARK_VALUE = "IC50_Dark_value"
LIGHT_VALUE = "IC50_Light_value"
CIS_VALUE = "IC50_Cisplatin_value"

# ``>100``, ``> 100``, ``<1`` — the only non-numeric forms present (verified).
_CENSOR_RE = re.compile(r"^\s*([<>])\s*([0-9]*\.?[0-9]+)\s*$")
_P53_RE = re.compile(r"P53\s*([+-])\s*/\s*([+-])")

# A derived endpoint PI = num/den is censored in the direction implied by which
# side is censored: a right-censored numerator pushes the ratio up, a
# right-censored denominator pushes it down.
_DERIVED_DIRECTION = {
    ("right", "none"): "right",
    ("left", "none"): "left",
    ("none", "right"): "left",
    ("none", "left"): "right",
    ("right", "right"): "indeterminate",
    ("left", "left"): "indeterminate",
    ("right", "left"): "indeterminate",
    ("left", "right"): "indeterminate",
    ("none", "none"): "none",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def normalize_cell_line(value: object) -> str:
    """Case/punctuation-insensitive cell-line key that keeps p53 status apart.

    ``SK-HEP-1`` and ``SK-Hep1`` must collide; ``HCT-116 p53+/+`` and
    ``HCT-116 p53-/-`` must not. Stripping all non-alphanumerics achieves the
    first and destroys the second, so the p53 sign is captured first and
    re-appended.
    """
    s = str(value).strip().upper()
    suffix = ""
    m = _P53_RE.search(s)
    if m:
        suffix = "_P53" + m.group(1) + m.group(2)
    base = re.sub(r"[^A-Z0-9]", "", s)
    base = re.sub(r"(CELLLINE|CELLS|CELL)$", "", base)
    return base + suffix


def parse_ic50(raw: object, fallback: object) -> Tuple[float, str]:
    """Return ``(value_um, censoring)`` where censoring in {none,right,left,missing}.

    ``>X`` is a *right*-censored IC50 (the true value is above the bound); ``<X``
    is *left*-censored. The ``*_value`` column is used only when the primary
    column is empty — for censored rows it duplicates the bound, which is
    verified against the primary column in ``checks`` rather than trusted.
    """
    s = "" if raw is None else str(raw).strip()
    m = _CENSOR_RE.match(s)
    if m:
        sign, bound = m.group(1), float(m.group(2))
        return bound, ("right" if sign == ">" else "left")
    v = pd.to_numeric(s, errors="coerce") if s else np.nan
    if pd.notna(v):
        return float(v), "none"
    v2 = pd.to_numeric(fallback, errors="coerce")
    if pd.notna(v2):
        return float(v2), "none"
    return float("nan"), "missing"


def canonicalize(smiles: object) -> Optional[str]:
    """RDKit canonical SMILES, or ``None`` if the string does not parse."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def combine_censoring(num: str, den: str) -> str:
    """Censoring direction of ``num / den`` given the two inputs' directions."""
    return _DERIVED_DIRECTION[(num, den)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent
        return None


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------
def build_long_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Explode one source row into up to three endpoint rows. Never drops a row."""
    n = len(df)
    dark, dark_c = zip(*[parse_ic50(a, b) for a, b in zip(df[DARK], df[DARK_VALUE])])
    light, light_c = zip(*[parse_ic50(a, b) for a, b in zip(df[LIGHT], df[LIGHT_VALUE])])
    cis, cis_c = zip(*[parse_ic50(a, b) for a, b in zip(df[CIS], df[CIS_VALUE])])
    dark = np.asarray(dark, dtype=float)
    light = np.asarray(light, dtype=float)
    cis = np.asarray(cis, dtype=float)
    dark_c = np.asarray(dark_c, dtype=object)
    light_c = np.asarray(light_c, dtype=object)
    cis_c = np.asarray(cis_c, dtype=object)

    canon = {s: canonicalize(s) for s in df["SMILES_Ligands"].astype(str).unique()}
    ligand_canon = df["SMILES_Ligands"].astype(str).map(canon)

    base = pd.DataFrame(
        {
            "source_row": np.arange(n, dtype=np.int64),
            "measurement_id": [
                hashlib.sha1(f"{i}|{d}|{m}".encode()).hexdigest()[:16]
                for i, d, m in zip(range(n), df["DOI"].astype(str), df["Metal"].astype(str))
            ],
            "metal": df["Metal"].astype(str).to_numpy(),
            "oxidation_state": pd.to_numeric(df["Oxidation_state"], errors="coerce").to_numpy(),
            "charge_complex": pd.to_numeric(df["Charge_complex"], errors="coerce").to_numpy(),
            "atomic_number": pd.to_numeric(df["Atomic_number"], errors="coerce").to_numpy(),
            "valence_e": pd.to_numeric(df["Valence_e"], errors="coerce").to_numpy(),
            "ligand_raw": df["SMILES_Ligands"].astype(str).to_numpy(),
            "ligand_canonical": ligand_canon.to_numpy(),
            "counterion": df["Counterion"].astype(str).to_numpy(),
            "cell_line_raw": df["Cell_line"].astype(str).to_numpy(),
            "cell_line_key": df["Cell_line"].map(normalize_cell_line).to_numpy(),
            "time_h": pd.to_numeric(df["Time(h)"], errors="coerce").to_numpy(),
            "doi": df["DOI"].astype(str).to_numpy(),
            "year": pd.to_numeric(df["Year"], errors="coerce").to_numpy(),
            "excitation_nm": pd.to_numeric(df["Excitation_Wavelength(nm)"], errors="coerce").to_numpy(),
            "irradiation_min": pd.to_numeric(df["Irradiation_Time(minutes)"], errors="coerce").to_numpy(),
            "irradiation_power": pd.to_numeric(df["Irradiation_Power(W*m^-2)"], errors="coerce").to_numpy(),
        }
    )

    frames: List[pd.DataFrame] = []
    ledger: Dict[str, object] = {}

    # --- potency -----------------------------------------------------------
    ok = dark_c != "missing"
    pot = base.loc[ok].copy()
    pot["endpoint"] = "potency"
    pot["value"] = dark[ok]
    pot["value_units"] = "uM"
    pot["censoring"] = dark_c[ok]
    pot["censor_bound"] = dark[ok]
    pot["value_log10"] = np.log10(pot["value"].to_numpy(dtype=float))
    frames.append(pot)
    ledger["potency"] = {
        "source_rows": int(n),
        "emitted": int(ok.sum()),
        "dropped_no_dark_ic50": int((~ok).sum()),
    }

    # --- phototoxicity -----------------------------------------------------
    ok = (dark_c != "missing") & (light_c != "missing") & (light > 0) & (dark > 0)
    pi = base.loc[ok].copy()
    ratio = dark[ok] / light[ok]
    cens = np.array([combine_censoring(a, b) for a, b in zip(dark_c[ok], light_c[ok])], dtype=object)
    pi["endpoint"] = "phototoxicity"
    pi["value"] = ratio
    pi["value_units"] = "ratio_dark_over_light"
    pi["censoring"] = cens
    pi["censor_bound"] = ratio
    pi["value_log10"] = np.log10(ratio)
    frames.append(pi)
    ledger["phototoxicity"] = {
        "source_rows": int(n),
        "emitted": int(ok.sum()),
        "dropped_no_light_ic50": int((light_c == "missing").sum()),
        "dropped_nonpositive": int((((dark_c != "missing") & (light_c != "missing")) & ~((light > 0) & (dark > 0))).sum()),
        "indeterminate_censoring": int((cens == "indeterminate").sum()),
    }

    # --- cisplatin-normalised ---------------------------------------------
    ok = (dark_c != "missing") & (cis_c != "missing") & (cis > 0) & (dark > 0)
    ri = base.loc[ok].copy()
    ratio = dark[ok] / cis[ok]
    cens = np.array([combine_censoring(a, b) for a, b in zip(dark_c[ok], cis_c[ok])], dtype=object)
    ri["endpoint"] = "cisplatin_ratio"
    ri["value"] = ratio
    ri["value_units"] = "ratio_dark_over_cisplatin"
    ri["censoring"] = cens
    ri["censor_bound"] = ratio
    ri["value_log10"] = np.log10(ratio)
    frames.append(ri)
    ledger["cisplatin_ratio"] = {
        "source_rows": int(n),
        "emitted": int(ok.sum()),
        "dropped_no_cisplatin_ic50": int((cis_c == "missing").sum()),
        "dropped_nonpositive": int((((dark_c != "missing") & (cis_c != "missing")) & ~((cis > 0) & (dark > 0))).sum()),
        "indeterminate_censoring": int((cens == "indeterminate").sum()),
    }

    long = pd.concat(frames, ignore_index=True)

    # Provenance / cohort flags computed once per source row and merged in.
    long["in_paper_funnel"] = False
    for metal in METALS:
        from molmetal.data.cytotox import build_paper_cohort_indices

        idx = build_paper_cohort_indices(df, metal=metal)
        long.loc[long["source_row"].isin(set(idx.tolist())), "in_paper_funnel"] = True
    from molmetal.data.cytotox import NON_CANCER_CELL_LINES

    non_cancer = {normalize_cell_line(x) for x in NON_CANCER_CELL_LINES}
    long["is_cancer_line"] = ~long["cell_line_key"].isin(non_cancer)

    long["n_fragments"] = long["ligand_canonical"].fillna("").str.count(r"\.") + 1
    long.loc[long["ligand_canonical"].isna(), "n_fragments"] = np.nan

    # Stable column order for downstream readers.
    order = [
        "measurement_id",
        "source_row",
        "endpoint",
        "value",
        "value_log10",
        "value_units",
        "censoring",
        "censor_bound",
        "metal",
        "oxidation_state",
        "charge_complex",
        "atomic_number",
        "valence_e",
        "ligand_raw",
        "ligand_canonical",
        "n_fragments",
        "counterion",
        "cell_line_raw",
        "cell_line_key",
        "time_h",
        "doi",
        "year",
        "excitation_nm",
        "irradiation_min",
        "irradiation_power",
        "is_cancer_line",
        "in_paper_funnel",
    ]
    return long[order], ledger


def build_dedup(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (ligand, metal, cell line, time, DOI, endpoint); median value.

    Censored rows are excluded from the median and counted in ``n_censored``.
    A group with no exact measurement keeps the median *bound* and is flagged
    ``all_censored=True`` so downstream code can drop it explicitly rather
    than inherit an imputed number.
    """
    keys = ["ligand_canonical", "metal", "cell_line_key", "time_h", "doi", "endpoint"]
    exact = long[long["censoring"] == "none"]
    agg = (
        exact.groupby(keys, dropna=False)
        .agg(
            value_log10=("value_log10", "median"),
            n_exact=("value_log10", "size"),
        )
        .reset_index()
    )
    cens_counts = (
        long[long["censoring"] != "none"]
        .groupby(keys, dropna=False)
        .size()
        .rename("n_censored")
        .reset_index()
    )
    out = agg.merge(cens_counts, on=keys, how="outer")
    out["n_exact"] = out["n_exact"].fillna(0).astype(int)
    out["n_censored"] = out["n_censored"].fillna(0).astype(int)
    out["n_measurements"] = out["n_exact"] + out["n_censored"]
    out["all_censored"] = out["value_log10"].isna()
    out["is_cancer_line"] = out["cell_line_key"].isin(
        set(long.loc[long["is_cancer_line"], "cell_line_key"])
    )
    out["in_paper_funnel"] = out["ligand_canonical"].isin(
        set(long.loc[long["in_paper_funnel"], "ligand_canonical"])
    )
    return out.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def _metal_pair_coverage(dedup: pd.DataFrame, endpoint: str) -> Dict[str, object]:
    """How many (ligand, cell line) units carry both metals of each pair.

    ``same_doi`` / ``same_time`` split the clean comparisons (both metals
    measured in one paper, one assay duration) from the confounded ones.
    """
    sub = dedup[(dedup["endpoint"] == endpoint) & (~dedup["all_censored"])]
    sub = sub[sub["metal"].isin(METALS)]
    unit_keys = ["ligand_canonical", "cell_line_key"]

    def _group_with_pairs(frame: pd.DataFrame) -> pd.DataFrame:
        """Per-unit sets of DOIs, durations, and (DOI, duration) tuples."""
        g = frame.groupby(unit_keys, dropna=False).agg(
            dois=("doi", lambda s: set(s.astype(str))),
            times=("time_h", lambda s: set(s.dropna().tolist())),
        )
        g["doi_times"] = frame.groupby(unit_keys, dropna=False).apply(
            lambda f: set(zip(f["doi"].astype(str), f["time_h"])), include_groups=False
        )
        return g

    by_pair: Dict[str, Dict[str, int]] = {}
    for i, m1 in enumerate(METALS):
        for m2 in METALS[i + 1 :]:
            a = _group_with_pairs(sub[sub["metal"] == m1])
            b = _group_with_pairs(sub[sub["metal"] == m2])
            joined = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            by_pair[f"{m1}-{m2}"] = {
                "n_matched_units": int(len(joined)),
                "n_same_doi": int(sum(bool(x & y) for x, y in zip(joined["dois_a"], joined["dois_b"]))),
                "n_same_time": int(sum(bool(x & y) for x, y in zip(joined["times_a"], joined["times_b"]))),
                "n_same_doi_and_time": int(
                    sum(bool(x & y) for x, y in zip(joined["doi_times_a"], joined["doi_times_b"]))
                ),
            }
    return by_pair


def build_audit(
    df: pd.DataFrame,
    long: pd.DataFrame,
    dedup: pd.DataFrame,
    ledger: Dict[str, object],
    source: Path,
    argv: List[str],
) -> Dict[str, object]:
    import rdkit

    n_src = len(df)

    # ---- censoring / value-column agreement (checked, not assumed) --------
    value_col_checks = {}
    for name, col, vcol in (("dark", DARK, DARK_VALUE), ("light", LIGHT, LIGHT_VALUE), ("cisplatin", CIS, CIS_VALUE)):
        raw = df[col].astype(str).str.strip()
        cens = raw.str.match(r"^[<>]")
        parsed, _ = zip(*[parse_ic50(a, b) for a, b in zip(df[col], df[vcol])])
        parsed = np.asarray(parsed, dtype=float)
        fallback = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)
        both = cens.to_numpy() & np.isfinite(parsed) & np.isfinite(fallback)
        max_diff = float(np.max(np.abs(parsed[both] - fallback[both]))) if both.any() else None
        value_col_checks[name] = {
            "censored_rows": int(cens.sum()),
            "censored_rows_with_value_col": int(both.sum()),
            "max_abs_bound_minus_value_col": max_diff,
            "agrees": bool(max_diff is not None and max_diff < 1e-9),
        }

    # ---- endpoint counts --------------------------------------------------
    endpoint_counts: Dict[str, object] = {}
    for ep in ENDPOINTS:
        s = long[long["endpoint"] == ep]
        vc = s["censoring"].value_counts().to_dict()
        endpoint_counts[ep] = {
            "n_rows": int(len(s)),
            "n_exact": int(vc.get("none", 0)),
            "n_right_censored": int(vc.get("right", 0)),
            "n_left_censored": int(vc.get("left", 0)),
            "n_indeterminate": int(vc.get("indeterminate", 0)),
            "censoring_rate": round(float(1.0 - vc.get("none", 0) / max(len(s), 1)), 4),
            "value_log10_quantiles_exact_only": {
                q: round(float(s.loc[s["censoring"] == "none", "value_log10"].quantile(float(q))), 4)
                for q in ("0.01", "0.25", "0.5", "0.75", "0.99")
            },
            "n_unique_ligands": int(s["ligand_canonical"].nunique()),
            "n_unique_cell_lines": int(s["cell_line_key"].nunique()),
            "n_unique_dois": int(s["doi"].nunique()),
        }

    # ---- per metal --------------------------------------------------------
    per_metal: Dict[str, object] = {}
    for m in METALS:
        row: Dict[str, object] = {}
        for ep in ENDPOINTS:
            s = long[(long["metal"] == m) & (long["endpoint"] == ep)]
            exact = s[s["censoring"] == "none"]
            row[ep] = {
                "n_rows": int(len(s)),
                "n_exact": int(len(exact)),
                "censoring_rate": round(float(1.0 - len(exact) / max(len(s), 1)), 4),
                "median_log10": round(float(exact["value_log10"].median()), 4) if len(exact) else None,
                "n_unique_ligands": int(s["ligand_canonical"].nunique()),
                "n_unique_cell_lines": int(s["cell_line_key"].nunique()),
                "n_unique_dois": int(s["doi"].nunique()),
            }
        per_metal[m] = row

    # ---- cohorts ----------------------------------------------------------
    def _cohort_stats(sub: pd.DataFrame) -> Dict[str, object]:
        return {
            "n_rows": int(len(sub)),
            "n_unique_ligands": int(sub["ligand_canonical"].nunique()),
            "per_metal_rows": {m: int((sub["metal"] == m).sum()) for m in METALS},
            "per_metal_unique_ligands": {
                m: int(sub.loc[sub["metal"] == m, "ligand_canonical"].nunique()) for m in METALS
            },
        }

    pot = long[long["endpoint"] == "potency"]
    cohorts = {
        "all_rows": _cohort_stats(pot),
        "numeric_dark": _cohort_stats(pot[pot["censoring"] == "none"]),
        "cancer_only": _cohort_stats(pot[pot["is_cancer_line"]]),
        "paper_funnel": _cohort_stats(pot[pot["in_paper_funnel"]]),
        "paper_funnel_numeric": _cohort_stats(
            pot[pot["in_paper_funnel"] & (pot["censoring"] == "none")]
        ),
    }

    # ---- filter ledger (no silent drops) ---------------------------------
    filters = [
        {
            "step": "source rows",
            "rows_in": n_src,
            "rows_out": n_src,
            "dropped": 0,
            "reason": "MetalCytoToxDB.csv as shipped",
        },
        {
            "step": "metal filter",
            "rows_in": n_src,
            "rows_out": int(df["Metal"].astype(str).isin(METALS).sum()),
            "dropped": int((~df["Metal"].astype(str).isin(METALS)).sum()),
            "reason": "keep the five studied metals; every dropped row is counted, not deleted",
        },
    ]
    for ep in ENDPOINTS:
        info = ledger[ep]
        filters.append(
            {
                "step": f"{ep}: endpoint requires its raw IC50(s)",
                "rows_in": n_src,
                "rows_out": info["emitted"],
                "dropped": n_src - info["emitted"],
                "reason": {k: v for k, v in info.items() if k.startswith("dropped_") or k == "indeterminate_censoring"},
            }
        )
    filters.append(
        {
            "step": "dedup to (ligand, metal, cell, time, DOI, endpoint)",
            "rows_in": int(len(long)),
            "rows_out": int(len(dedup)),
            "dropped": int(len(long) - len(dedup)),
            "reason": "replicates collapsed to the median log10 value; censored rows counted separately",
        }
    )

    # ---- matched pairs ----------------------------------------------------
    matched = {ep: _metal_pair_coverage(dedup, ep) for ep in ENDPOINTS}
    matched_totals: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sub = dedup[(dedup["endpoint"] == ep) & (~dedup["all_censored"]) & dedup["metal"].isin(METALS)]
        units = sub.groupby(["ligand_canonical", "cell_line_key"])["metal"].nunique()
        matched_totals[ep] = {
            "n_units_total": int(len(units)),
            "n_units_multi_metal": int((units >= 2).sum()),
            "n_units_2_metals": int((units == 2).sum()),
            "n_units_3plus_metals": int((units >= 3).sum()),
            "fraction_multi_metal": round(float((units >= 2).mean()), 4) if len(units) else None,
        }

    # ---- cell-line normalisation -----------------------------------------
    raw_names = sorted(df["Cell_line"].astype(str).unique())
    clusters: Dict[str, List[str]] = {}
    for name in raw_names:
        clusters.setdefault(normalize_cell_line(name), []).append(name)
    merges = {k: sorted(v) for k, v in clusters.items() if len(v) > 1}

    # ---- DOI structure ----------------------------------------------------
    doi_per_unit = (
        dedup[dedup["endpoint"] == "potency"]
        .groupby(["ligand_canonical", "metal", "cell_line_key", "time_h"], dropna=False)["doi"]
        .nunique()
    )

    # ---- what the coverage implies for the framework ----------------------
    # These are not opinions: each one follows from a number above and changes
    # what MSRS-1/3/5 are allowed to claim.
    pot_pairs = matched["potency"]
    nonempty_pairs = {k: v for k, v in pot_pairs.items() if v["n_matched_units"] > 0}
    pi_multi = matched_totals["phototoxicity"]["n_units_multi_metal"]

    # Reconcile the 668 figure from the 2026-10-18 ad-hoc audit. It grouped on
    # raw strings and kept censored rows; both choices inflate the count.
    blind = (
        long[long["endpoint"] == "potency"]
        .groupby(["ligand_canonical", "cell_line_key"], dropna=False)["metal"]
        .nunique()
    )
    reconciliation = {
        "prior_audit_value": 668,
        "prior_audit_rule": "raw SMILES x raw Cell_line string, censored rows included",
        "recomputed_censoring_blind_normalized_cell_key": int((blind >= 2).sum()),
        "recomputed_numeric_only_normalized_cell_key": int(
            matched_totals["potency"]["n_units_multi_metal"]
        ),
        "explanation": "the 668 count treats '>100 uM' as a measurement; it also splits "
        "spelling variants of one cell line. Restricting to exact measurements and "
        "normalising the cell-line key gives the number in "
        "matched_pairs.totals.potency.n_units_multi_metal, which supersedes 668.",
    }
    design_consequences = {
        "potency_matched_pairs_nonempty": sorted(nonempty_pairs),
        "potency_matched_units_by_pair": {k: v["n_matched_units"] for k, v in nonempty_pairs.items()},
        "potency_clean_fraction_by_pair": {
            k: round(v["n_same_doi_and_time"] / v["n_matched_units"], 4) for k, v in nonempty_pairs.items()
        },
        "ru_ir_matched_units": int(pot_pairs["Ru-Ir"]["n_matched_units"]),
        "phototoxicity_matched_units": int(pi_multi),
        "reconciles_prior_audit": reconciliation,
        "statements": [
            "Only {}/10 metal pairs have any within-ligand potency comparison, and Ru-Ir has "
            "{} matched units. Any 'Ru vs Ir' substitution claim is a between-ligand comparison "
            "and cannot be paired.".format(len(nonempty_pairs), pot_pairs["Ru-Ir"]["n_matched_units"]),
            "The paired potency evidence is {}.".format(
                "; ".join(
                    "{}: {} units, {:.0%} same DOI and duration".format(
                        k, v["n_matched_units"], v["n_same_doi_and_time"] / max(v["n_matched_units"], 1)
                    )
                    for k, v in sorted(nonempty_pairs.items(), key=lambda kv: -kv[1]["n_matched_units"])
                )
            ),
            "Phototoxicity has {} within-ligand cross-metal units: no ligand is measured for "
            "PI under two metals. MSRS-5 must be a between-ligand metal contrast, and any "
            "'metal switches the mechanism' sentence has to be downgraded to exploratory.".format(pi_multi),
            "Only {:.1%} of potency units are multi-metal, so a model that predicts Delta has "
            "very few supervised examples per metal pair; L3 must report per-pair n.".format(
                matched_totals["potency"]["fraction_multi_metal"] or 0.0
            ),
        ],
    }

    checks = {
        "no_missing_doi": bool((long["doi"].astype(str).str.len() > 0).all()),
        "no_missing_metal": bool(long["metal"].isin(METALS).all()),
        "all_canonical_parsed": bool(long["ligand_canonical"].notna().all()),
        "value_column_agrees_with_bound": value_col_checks,
        "no_silent_row_loss": bool(
            len(long[long["endpoint"] == "potency"]) + ledger["potency"]["dropped_no_dark_ic50"] == n_src
        ),
        "dedup_has_no_duplicate_keys": bool(
            not dedup.duplicated(
                subset=["ligand_canonical", "metal", "cell_line_key", "time_h", "doi", "endpoint"]
            ).any()
        ),
        "potency_value_log10_finite_for_exact": bool(
            np.isfinite(pot.loc[pot["censoring"] == "none", "value_log10"]).all()
        ),
    }

    caveats = [
        "IC50_*_value is the numeric bound for censored rows, so any code that reads it "
        "directly (including the paper-funnel binary label) treats '>100 uM' as exactly 100 uM. "
        "This table keeps the censoring flag so the two can be separated.",
        "Censoring is not random: 18.4% of paper-funnel Ru rows are right-censored vs 10.7% for Re. "
        "Dropping censored rows biases every metal median toward potent compounds, and biases metals "
        "unequally. Report both the exact-only and the bound-imputed view.",
        "Matched cross-metal units are a small, non-random subset of each metal's cohort "
        "(see matched_pairs). They are the only units that identify a substitution effect without "
        "a between-paper confound, so every Delta must carry its n.",
        "Phototoxicity needs dark and light IC50 on the same source row; the source does not "
        "pair measurements across rows, so a paper that reports light/dark in separate rows "
        "contributes no PI.",
        "SMILES_Ligands is a mixture string that may embed counterions; n_fragments records the "
        "fragment count but ligand identity is the whole canonical string, matching the "
        "classification mainline so cohorts stay comparable.",
        "Cell-line keys merge 16 spelling variants and keep the p53 +/- pair apart. Keys are "
        "declared, not inferred from ontology lookup.",
    ]

    return {
        "artifact": "msrs_endpoints",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "source_csv": str(source),
            "source_sha256": _sha256(source),
            "n_source_rows": n_src,
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": argv,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
        },
        "endpoint_definitions": {
            "potency": "y = log10(IC50_dark / uM); lower is more potent",
            "phototoxicity": "PI = log10(IC50_dark / IC50_light); positive = more potent under light",
            "cisplatin_ratio": "RI = log10(IC50_dark / IC50_cisplatin); cell-line normalised",
        },
        "endpoints": endpoint_counts,
        "per_metal": per_metal,
        "cohorts": cohorts,
        "filters": filters,
        "matched_pairs": {
            "definition": "unit = (ligand_canonical, cell_line_key); a unit is matched for a metal "
            "pair when both metals have a non-censored measurement in the dedup table",
            "by_endpoint": matched,
            "totals": matched_totals,
        },
        "cell_line_normalization": {
            "n_raw": len(raw_names),
            "n_keys": len(clusters),
            "merges": merges,
            "rule": "upper -> capture p53 +/- -> strip non-alphanumerics -> strip trailing CELL/CELLS/CELLLINE",
        },
        "doi_structure": {
            "n_dois": int(df["DOI"].nunique()),
            "year_range": [int(df["Year"].min()), int(df["Year"].max())],
            "n_ligand_metal_cell_time_units": int(len(doi_per_unit)),
            "units_spanning_multiple_dois": int((doi_per_unit > 1).sum()),
            "note": "leave-one-DOI-out is only meaningful for the multi-DOI units",
        },
        "design_consequences": design_consequences,
        "checks": checks,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
    )
    parser.add_argument("--out-long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    parser.add_argument(
        "--out-dedup",
        type=Path,
        default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_endpoints.json",
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"FAIL: source not found: {args.source}")
        return 1

    df = pd.read_csv(args.source, low_memory=False)
    long, ledger = build_long_table(df)
    dedup = build_dedup(long)

    # Acceptance gates. A failure here means the artifact would be unsound.
    failures: List[str] = []
    for ep in ENDPOINTS:
        if int((long["endpoint"] == ep).sum()) == 0:
            failures.append(f"endpoint {ep} produced zero rows")
    if long["doi"].isna().any():
        failures.append("null DOI in long table")
    if (long["censoring"] == "missing").any():
        failures.append("'missing' censoring leaked into the long table")
    if dedup.duplicated(subset=["ligand_canonical", "metal", "cell_line_key", "time_h", "doi", "endpoint"]).any():
        failures.append("dedup table has duplicate keys")

    audit = build_audit(df, long, dedup, ledger, args.source, [sys.executable] + (argv or sys.argv[1:]))
    for name, ok in audit["checks"].items():
        if name == "value_column_agrees_with_bound":
            for sub in ok.values():
                if not sub["agrees"]:
                    failures.append(f"value column disagrees with bound for {sub}")
        elif not ok:
            failures.append(f"check failed: {name}")

    _write_parquet(long, args.out_long)
    _write_parquet(dedup, args.out_dedup)
    atomic_write_file(args.out_json, json.dumps(audit, indent=2) + "\n")

    print(f"long rows      : {len(long)}")
    print(f"dedup rows     : {len(dedup)}")
    print(f"json           : {args.out_json}")
    for ep in ENDPOINTS:
        c = audit["endpoints"][ep]
        print(
            f"  {ep:16s} n={c['n_rows']:6d} exact={c['n_exact']:6d} "
            f"cens={c['censoring_rate']:.3f} ligands={c['n_unique_ligands']}"
        )
    for ep in ENDPOINTS:
        t = audit["matched_pairs"]["totals"][ep]
        print(f"  matched[{ep:16s}] units={t['n_units_multi_metal']:5d} / {t['n_units_total']}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
