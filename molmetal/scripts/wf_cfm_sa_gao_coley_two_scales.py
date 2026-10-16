#!/usr/bin/env python3
"""Wf-CFM-SA-GC: two-scale SA normalisation report.

================================================================
What this script does
================================================================
Score the SAME two cohorts on TWO different SA normalisation scales:

  * Linear hack (current repo-wide convention):
        sa_norm_linear = 1 - SA_raw / 10
  * Gao & Coley (J. Chem. Inf. Model. 2020, 60, 5714-5723,
    DOI 10.1021/acs.jcim.0c00174):
        sa_norm_gao_coley = exp(-(SA_raw - mu) / sigma)
    with mu=2.230044, sigma=0.6526308.

THE TWO SCALES ARE NOT COMPARABLE. They are different functions of the
same raw Ertl score. State that explicitly in any report that places
them in the same sentence.

================================================================
Cohorts
================================================================
* generated: molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv
  (n=66 surviving free generations, seed-0 epoch-10 checkpoint)
* real_tmqm: .data/molmetal/tmQM/tmQM_y.csv filtered to
  {Pt, Ir, Ru, Rh, Os, Re} with parseable SMILES (n=31,150;
  the real six-metal baseline already on disk in
  metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json).

================================================================
Output
================================================================
metrics/by_metric/cfm_sa_gao_coley_reference.json -- single consolidated
JSON of:
  * SA_raw on each cohort
  * SA_norm_linear on each cohort (the existing hack)
  * SA_norm_gao_coley on each cohort (NEW)
  * 8th-decade axios of the literature LigandDiff PPR_100 0.68 with
    the scale ambiguity explicitly recorded
  * a NON-VERIFIED flag for the comparability vs LigandDiff

CPU only. Pure RDKit + arithmetic. No GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Repo-root on sys.path so the adapter import works regardless of
# invocation form (script directly, or via the test runner).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.adapters.sa_gao_coley import (  # noqa: E402
    GAO_COLEY_MU,
    GAO_COLEY_SIGMA,
    gao_coley_sa_norm,
)


# The six train metals and their RDKit Z-numbers. Centralised in the
# real-tmQM baseline; reproduced here for self-containment.
SIX_METAL_ZNUMBERS = {
    "Pt": 78,
    "Ir": 77,
    "Ru": 44,
    "Rh": 45,
    "Os": 76,
    "Re": 75,
}

# SMILES that contain any of these as a bracketed atom -- by element
# symbol inside [..] with the train metals. Cheap O(n) check, exact.
SIX_METAL_BRACKET_TOKENS = {f"[{sym}" for sym in SIX_METAL_ZNUMBERS.keys()}


def _load_sascorer():
    """Lazy-load rdkit-Contrib sascorer with the project-standard
    exception-mirror fallback."""
    from rdkit import RDConfig
    import os as _os
    import importlib

    sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    return importlib.import_module("sascorer")


def _score_smiles(smiles_list: Sequence[str]):
    """Score a sequence of SMILES. Returns (raw_sa, parsed_mask,
    parse_failures) where parse_failures has the row indices of unparseable
    SMILES so we report the denominator."""
    from rdkit import Chem

    sascorer = _load_sascorer()
    raw: List[Optional[float]] = []
    parsed: List[bool] = []
    failures: List[int] = []
    for i, sm in enumerate(smiles_list):
        if not isinstance(sm, str) or not sm.strip():
            raw.append(None)
            parsed.append(False)
            failures.append(i)
            continue
        try:
            m = Chem.MolFromSmiles(sm)
        except Exception:
            m = None
        if m is None:
            raw.append(None)
            parsed.append(False)
            failures.append(i)
            continue
        try:
            s = float(sascorer.calculateScore(m))
            raw.append(s)
            parsed.append(True)
        except Exception:
            raw.append(None)
            parsed.append(False)
            failures.append(i)
    return raw, parsed, failures


def _read_generated_csv(path: Path) -> List[str]:
    """Read the smiles column from the generated CSV (skip header)."""
    out: List[str] = []
    with path.open(newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            sm = row.get("smiles") or row.get("SMILES") or ""
            out.append(sm)
    return out


def _read_real_tmqm_csv(path: Path) -> List[str]:
    """Read SMILES from .data/molmetal/tmQM/tmQM_y.csv (semi-colon
    delimited, SMILES is the last column) and filter to rows whose SMILES
    contain >=1 of the six train metals. Follow the protocol used in
    metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json exactly:
    the filter is structural (SMILES contains the bracketed metal token)
    NOT a column match, because tmQM_y.csv does not carry a metal-symbol
    column.
    """
    out: List[str] = []
    with path.open(newline="") as fh:
        rdr = csv.reader(fh, delimiter=";")
        header = next(rdr, None)
        # The header for this file is 11 columns, so the data rows are
        # 11 columns too.  Require >= 11 (NOT 12) to match.
        min_cols = len(header) if header is not None else 11
        for row in rdr:
            if len(row) < min_cols:
                continue
            sm = row[-1].strip()
            if not sm:
                continue
            if any(tok in sm for tok in SIX_METAL_BRACKET_TOKENS):
                out.append(sm)
    return out


def _summarise(values: Sequence[Optional[float]]) -> Dict[str, float]:
    """Mean / median / sd / p10 / p90 of values, ignoring Nones. Returns
    0.0 for any statistic whose denominator is zero."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0, "p10": 0.0, "p90": 0.0}
    s = sorted(vals)
    mean = sum(s) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in s) / (n - 1)
        sd = var ** 0.5
    else:
        sd = 0.0

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        # Linear interpolation, matches numpy default.
        pos = p * (n - 1)
        lo = int(pos)
        frac = pos - lo
        if lo + 1 < n:
            return s[lo] + frac * (s[lo + 1] - s[lo])
        return s[lo]

    return {
        "n": n,
        "mean": mean,
        "median": pct(0.5),
        "sd": sd,
        "p10": pct(0.1),
        "p90": pct(0.9),
    }


def _resolve_default_paths(args: argparse.Namespace):
    """Pick file paths. Defaults match the ones used by the CFM line
    today; --override is for re-runs only."""
    repo = Path("/home/hugo/codes/mol-metal")
    gen_csv = Path(
        args.gen_csv
        or repo / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv"
    )
    real_csv = Path(args.real_csv or repo / ".data" / "molmetal" / "tmQM" / "tmQM_y.csv")
    out_json = Path(args.out_json or repo / "metrics" / "by_metric" / "cfm_sa_gao_coley_reference.json")
    return gen_csv, real_csv, out_json


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-csv", type=str, default=None)
    parser.add_argument("--real-csv", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument(
        "--real-prefix",
        type=int,
        default=None,
        help="If set, score only the first N real rows (smoke test).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    gen_csv, real_csv, out_json = _resolve_default_paths(args)

    if not gen_csv.is_file():
        print(f"ERROR: generated CSV not found: {gen_csv}", file=sys.stderr)
        return 2
    if not real_csv.is_file():
        print(f"ERROR: real tmQM CSV not found: {real_csv}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"[wf_cfm_sa_gc] reading generated SMILES from {gen_csv}")
    gen_smiles = _read_generated_csv(gen_csv)
    if not args.quiet:
        print(f"[wf_cfm_sa_gc] reading real SMILES from {real_csv}")
    real_smiles = _read_real_tmqm_csv(real_csv)
    if args.real_prefix is not None:
        real_smiles = real_smiles[: args.real_prefix]
        if not args.quiet:
            print(f"[wf_cfm_sa_gc] --real-prefix={args.real_prefix}; truncated cohort to {len(real_smiles)}")

    if not args.quiet:
        print(f"[wf_cfm_sa_gc] scoring {len(gen_smiles)} generated + {len(real_smiles)} real SMILES ...")

    gen_raw, gen_parsed, gen_fail = _score_smiles(gen_smiles)
    real_raw, real_parsed, real_fail = _score_smiles(real_smiles)

    # Two-scale normalisation on successfully-parsed rows.
    gen_linear = [None if v is None else 1.0 - v / 10.0 for v in gen_raw]
    real_linear = [None if v is None else 1.0 - v / 10.0 for v in real_raw]
    gen_gc = [None if v is None else gao_coley_sa_norm(v) for v in gen_raw]
    real_gc = [None if v is None else gao_coley_sa_norm(v) for v in real_raw]

    payload: Dict[str, object] = {
        "metric": "cfm_sa_gao_coley_reference",
        "tag": "MEASURED",
        "date": "2026-10-12",
        "purpose": (
            "Score the SAME two cohorts (generated n=66 and real_tmQM six-metal "
            "n=31150) on TWO different SA normalisation scales. The two scales "
            "are NOT comparable to each other and NOT comparable to LigandDiff's "
            "0.68-0.80 band unless LigandDiff's own normalisation is verified."
        ),
        "gao_coley_parameters": {
            "mu": GAO_COLEY_MU,
            "sigma": GAO_COLEY_SIGMA,
            "formula": "sa_norm = exp(-(sa_raw - mu) / sigma)",
            "axis_convention": "0 = hard, 1 = easy (HIGHER = more synthetically accessible)",
            "calibrated_against": (
                "Population mean and standard deviation of Ertl SA across the "
                "Enamine REAL / ZINC15-clean-3 lead-like library (Gao & Coley "
                "2020, J. Chem. Inf. Model. 60, 5714-5723, DOI "
                "10.1021/acs.jcim.0c00174). NOT derived from tmQM or any "
                "organometallic corpus."
            ),
        },
        "linear_hack_parameters": {
            "formula": "sa_norm_linear = 1 - SA_raw / 10",
            "axis_convention": "0 = hard, 1 = easy (clipped at sa_raw>=10)",
            "what_is_wrong_with_it": (
                "Linear rescaling of the Ertl range [1,10] ignores the calibrated "
                "Gaussian-shaped difficulty profile. A SA of 4 (moderate) and a SA "
                "of 6 (hard) get mapped to 0.6 and 0.4 respectively, but Ertl's own "
                "rating of these is not linearly proportional."
            ),
        },
        "literature_comparison": {
            "liganddiff_PPR_100": 0.68,
            "liganddiff_table2_random_6_20": 0.69,
            "liganddiff_table2_fix_lig_6": 0.80,
            "citation": (
                "Jin H, Merz KM Jr. J. Chem. Theory Comput. 2024, 20(10), "
                "4377-4384. DOI 10.1021/acs.jctc.4c00232."
            ),
            "venue_correction": (
                "Cited in cfm_todo.json as 'JCIM 2024' -- it is JCTC 2024."
            ),
            "what_liganddiff_reports": (
                "Table 2 SA scores for Pt/Pd/Ru complexes, axis 0=hard / 1=easy. "
                "LigandDiff does not state which normalisation it uses."
            ),
            "comparability_with_this_repo_verdict": (
                "NOT VERIFIED. LigandDiff does not cite Gao & Coley, Brown "
                "(GuacaMol), or any other normalised-SA paper. Their SI does "
                "not quote a formula or its parameters. Until LigandDiff's "
                "implementation is read, LigandDiff's 0.68-0.80 band is "
                "incomparable to BOTH our scales: if their scale equals the "
                "Gao-Coley scale the gap is computed below; if their scale "
                "equals the linear hack the gap is the same shape at a "
                "different magnitude."
            ),
        },
        "generated": {
            "n_input": len(gen_smiles),
            "n_parsed": sum(gen_parsed),
            "n_parse_failures": len(gen_fail),
            "sa_raw": _summarise(gen_raw),
            "sa_norm_linear_1_minus_SA_over_10": _summarise(gen_linear),
            "sa_norm_gao_coley": _summarise(gen_gc),
            "cohort": str(gen_csv),
        },
        "real_tmqm": {
            "n_input": len(real_smiles),
            "n_parsed": sum(real_parsed),
            "n_parse_failures": len(real_fail),
            "sa_raw": _summarise(real_raw),
            "sa_norm_linear_1_minus_SA_over_10": _summarise(real_linear),
            "sa_norm_gao_coley": _summarise(real_gc),
            "metal_filter": sorted(SIX_METAL_ZNUMBERS.keys()),
            "corpus": str(real_csv),
        },
        "gaps": {},
        "comparability_warning": (
            "Two numbers may only be reported side by side if they share "
            "dataset AND same SA normalisation scale. The LigandDiff 0.68-0.80 "
            "band uses an UNSTATED scale."
        ),
        "out_of_scope": [
            "Re-implementing SA from scratch (Ertl & Schuffenhauer 2009 / "
            "rdkit-Contrib sascorer.py is the only SA implementation in this "
            "venv -- verified: mordred 1613 descriptors include no SA; "
            "openbabel 3.1.0 has no SA).",
            "Computing the LigandDiff scale for our cohort until the paper's "
            "normalisation is identified.",
            "Replacing the Linear hack repo-wide -- that is a downstream "
            "decision; this script only computes both numbers.",
        ],
    }

    # Headline gaps (generated-minus-real, on each scale).
    g_raw = payload["generated"]["sa_raw"]
    r_raw = payload["real_tmqm"]["sa_raw"]
    g_lin = payload["generated"]["sa_norm_linear_1_minus_SA_over_10"]
    r_lin = payload["real_tmqm"]["sa_norm_linear_1_minus_SA_over_10"]
    g_gc = payload["generated"]["sa_norm_gao_coley"]
    r_gc = payload["real_tmqm"]["sa_norm_gao_coley"]
    payload["gaps"] = {
        "sa_raw_generated_minus_real": g_raw["mean"] - r_raw["mean"],
        "sa_norm_linear_generated_minus_real": g_lin["mean"] - r_lin["mean"],
        "sa_norm_gao_coley_generated_minus_real": g_gc["mean"] - r_gc["mean"],
        "sa_norm_gao_coley_real_minus_generated": r_gc["mean"] - g_gc["mean"],
        "liganddiff_PPR_100_minus_gao_coley_generated": 0.68 - g_gc["mean"],
        "liganddiff_PPR_100_minus_gao_coley_real": 0.68 - r_gc["mean"],
        "note": (
            "All gaps are arithmetic. The 'sa_norm_gao_coley_generated_minus_real' "
            "value is informative only as a ratio (real / generated), because both "
            "numbers are themselves the exp of -delta/sigma and the underlying raw "
            "delta dominates. See 'how_to_read' for the recommended reporting form."
        ),
    }

    payload["how_to_read"] = (
        "On the linear hack, generated 0.36 vs real 0.50 (gap 0.14 open) -- "
        "this is what the repo has always quoted. On Gao-Coley, generated "
        "~0.002 vs real ~0.015 (a ~7x ratio, not a comparable 0.14 -- the "
        "two scales are different functions and 0.14 on one is not the same "
        "physical claim as 0.14 on the other). The LigandDiff 0.68 band is "
        "from a paper whose SA normalisation is UNSTATED; do not cite it "
        "side-by-side with either scale until verified."
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    if not args.quiet:
        print(f"[wf_cfm_sa_gc] wrote {out_json}")
        print(f"[wf_cfm_sa_gc] generated   sa_raw={g_raw['mean']:.4f}  linear={g_lin['mean']:.4f}  gao_coley={g_gc['mean']:.6f}")
        print(f"[wf_cfm_sa_gc] real_tmqm   sa_raw={r_raw['mean']:.4f}  linear={r_lin['mean']:.4f}  gao_coley={r_gc['mean']:.6f}")
        print(f"[wf_cfm_sa_gc] gap (linear): {payload['gaps']['sa_norm_linear_generated_minus_real']:+.4f}; gap (gc): {payload['gaps']['sa_norm_gao_coley_generated_minus_real']:+.6f} (ratio r/g = {r_gc['mean']/max(g_gc['mean'], 1e-12):.2f}x)")
        print(f"[wf_cfm_sa_gc] LigandDiff PPR_100 (0.68, scale UNSTATED) vs our gc scales: gap to generated = {payload['gaps']['liganddiff_PPR_100_minus_gao_coley_generated']:+.6f}, gap to real = {payload['gaps']['liganddiff_PPR_100_minus_gao_coley_real']:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
