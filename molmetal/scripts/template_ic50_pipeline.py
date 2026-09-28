#!/usr/bin/env python3
"""Wf-74 Phase 3 — IC50 analysis pipeline TEMPLATE (from raw readout to pIC50).

=====================================================================
CPU-only scaffold for the wet-lab IC50 analysis pipeline
=====================================================================

**Purpose**: Implements the §1-§5 pipeline specified in
``TODO/pending/wetlab_ic50_analysis_pipeline.md`` using **pure NumPy + SciPy**
(no RDKit, no torch, no GPU dependency). The script is a **template**: it ships
a small **synthetic fixture** (8 dose points × 3 replicates × 5 compounds × 4
cell lines) that exercises every step of the pipeline and writes the schema
that real wet-lab data will populate.

**Inputs**:
  - Wet-lab long-form CSV per §1.1 schema (``--input``); OR the synthetic
    fixture if ``--synthetic`` is passed.
  - D-MPNN predictions (``molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json``)
    for §5 in-silico vs wet-lab comparison.

**Outputs** (under ``--out-dir``, default ``molmetal/reports/wf74_ic50_pipeline``):
  - ``per_compound_pic50_wetlab.json`` — schema per §5.5
  - ``template_per_compound.json`` — schema-only template (for PI handoff)
  - ``synthesis.md`` — human-readable summary

**Verify gates** (per party-twenty-spirit §3.5):
  - On **synthetic fixture**: every step runs end-to-end; the JSON writes;
    ``pearson_r`` is computed (not ``null``) but **framing flags it as
    synthetic** and the user is reminded that real wet-lab data is **DEFERRED**.
  - On **wet-lab CSV** (when P6.2 ships): same pipeline, real numbers;
    Pearson r ≥ 0.7 gate per (metal × cell_line).

**Honest framing**:
  * No wet-lab data today — this is a **template**, not a measurement.
  * The synthetic fixture's Pearson r is **not** the Wf-74 gate; it is a
    smoke-test that the pipeline runs end-to-end.
  * The pipeline does **NOT** touch ``paper/sections/*.tex``; per TODO-84,
    paper §5 is forbidden until user authorization + MEASURED wet-lab.

**Usage**::

    # Synthetic smoke (no input required):
    python molmetal/scripts/template_ic50_pipeline.py --synthetic

    # Real wet-lab (when P6.2 ships):
    python molmetal/scripts/template_ic50_pipeline.py \\
        --input wetlab_ic50.csv \\
        --dmpnn molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json \\
        --out-dir molmetal/reports/wf74_ic50_pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.stats import f_oneway, pearsonr, spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants (per TODO/pending/wetlab_ic50_analysis_pipeline.md §1-§5)
# ---------------------------------------------------------------------------
N_DOSES = 8
DOSE_GRID_UM = np.logspace(-1.5, 2.0, N_DOSES)  # 0.0316 to 100 µM
N_REPLICATES = 3
DEFAULT_CELL_LINES = ("MCF7", "A549", "A2780", "A2780cis")
DEFAULT_METALS = ("Pt", "Pd", "Au", "Ru", "Ir")
PEARSON_GATE = 0.7
N_BOOTSTRAP = 1000
RNG_SEED = 42
DEFAULT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf74_ic50_pipeline"
DEFAULT_DMPNN = PROJECT_ROOT / "molmetal" / "reports" / "wf58_dmpnn_predictions" / "per_compound_pic50.json"


# ---------------------------------------------------------------------------
# §1 Raw data normalization (template — synthetic fixture only here)
# ---------------------------------------------------------------------------
@dataclass
class PlateWell:
    """Per-well record per §1.1 schema."""
    plate_id: str
    well_id: str
    compound_id: str
    metal: str
    cell_line: str
    concentration_uM: float
    replicate: int
    assay: str
    time_h: int
    raw_readout: float
    background_well: bool
    vehicle_well: bool
    positive_ctrl_well: bool


def make_synthetic_fixture(
    compounds: Sequence[Tuple[str, str]] = None,
    n_doses: int = N_DOSES,
    n_replicates: int = N_REPLICATES,
    seed: int = RNG_SEED,
) -> List[PlateWell]:
    """Build a small synthetic wet-lab fixture (5 compounds × 4 cells × 8 doses × 3 reps).

    Synthetic IC50s are drawn from a per-metal log-uniform distribution; per-well
    viability is the Hill equation + log-normal noise. The pipeline exercises
    every step on this fixture; **the numbers are NOT real wet-lab data**.
    """
    if compounds is None:
        compounds = [
            ("platinai_0454", "Ir"),
            ("platinai_1735", "Pt"),
            ("platinai_2978", "Pd"),
            ("platinai_1921", "Au"),
            ("platinai_2907", "Ru"),
        ]
    rng = np.random.default_rng(seed)
    wells: List[PlateWell] = []
    for cpd, metal in compounds:
        ic50_true = float(10 ** rng.uniform(-1.0, 2.0))  # 0.1 to 100 µM
        hill = float(rng.uniform(0.8, 2.0))
        for cell in DEFAULT_CELL_LINES:
            # Cell-line potency shift (1.5x shift for A2780cis vs A2780)
            shift = 1.5 if (cell == "A2780cis" and metal == "Pt") else 1.0
            for conc in DOSE_GRID_UM[:n_doses]:
                for rep in range(1, n_replicates + 1):
                    viability = 1.0 / (1.0 + (conc / (ic50_true * shift)) ** hill)
                    viability = float(np.clip(viability + rng.normal(0, 0.05), 0.0, 1.5))
                    raw = viability * 0.8 + 0.1  # raw readout = viability * gain + offset
                    wells.append(PlateWell(
                        plate_id=f"P6-001-{chr(64 + rep)}",
                        well_id=f"C{(rep * 8) % 12 + 1:02d}",
                        compound_id=cpd, metal=metal, cell_line=cell,
                        concentration_uM=float(conc), replicate=rep,
                        assay="SRB_540nm", time_h=72,
                        raw_readout=float(raw),
                        background_well=False,
                        vehicle_well=False,
                        positive_ctrl_well=False,
                    ))
    # Add 12 background / 12 vehicle / 12 positive-ctrl wells per plate (3 plates)
    for plate_idx, plate in enumerate(["A", "B", "C"], start=1):
        for k in range(12):
            wells.append(PlateWell(
                plate_id=f"P6-001-{plate}", well_id=f"A{k+1:02d}",
                compound_id="bg", metal="-", cell_line="-",
                concentration_uM=0.0, replicate=0,
                assay="SRB_540nm", time_h=72, raw_readout=float(0.1 + 0.01 * k),
                background_well=True, vehicle_well=False, positive_ctrl_well=False,
            ))
            wells.append(PlateWell(
                plate_id=f"P6-001-{plate}", well_id=f"B{k+1:02d}",
                compound_id="veh", metal="-", cell_line="MCF7",
                concentration_uM=0.0, replicate=0,
                assay="SRB_540nm", time_h=72, raw_readout=float(0.9 + 0.02 * k),
                background_well=False, vehicle_well=True, positive_ctrl_well=False,
            ))
            wells.append(PlateWell(
                plate_id=f"P6-001-{plate}", well_id=f"D{k+1:02d}",
                compound_id="cisplatin", metal="Pt", cell_line="MCF7",
                concentration_uM=10.0, replicate=0,
                assay="SRB_540nm", time_h=72, raw_readout=float(0.2),
                background_well=False, vehicle_well=False, positive_ctrl_well=True,
            ))
    return wells


def normalize_wells(wells: Sequence[PlateWell]) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """Apply §1.2 background subtraction + §1.5 viability conversion.

    Returns ``{(compound_id, cell_line): [(concentration_uM, viability), ...]}``.
    """
    # Plate-wise background = mean of background wells
    bg_per_plate: Dict[str, float] = {}
    for plate_id in {w.plate_id for w in wells}:
        bg_wells = [w.raw_readout for w in wells
                    if w.plate_id == plate_id and w.background_well]
        bg_per_plate[plate_id] = float(np.mean(bg_wells)) if bg_wells else 0.0
    # Vehicle per plate for viability normalization
    veh_per_plate: Dict[str, float] = {}
    for plate_id in {w.plate_id for w in wells}:
        veh_wells = [w.raw_readout for w in wells
                     if w.plate_id == plate_id and w.vehicle_well]
        veh_per_plate[plate_id] = float(np.median(veh_wells)) if veh_wells else 1.0
    # Aggregate viability per (compound, cell_line)
    out: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for w in wells:
        if w.background_well or w.vehicle_well or w.positive_ctrl_well:
            continue
        if w.compound_id == "bg" or w.compound_id == "veh" or w.compound_id == "cisplatin":
            continue
        corrected = w.raw_readout - bg_per_plate[w.plate_id]
        viability = float(np.clip(corrected / veh_per_plate[w.plate_id], 0.0, 1.5))
        out.setdefault((w.compound_id, w.cell_line), []).append(
            (float(w.concentration_uM), viability)
        )
    return out


# ---------------------------------------------------------------------------
# §2 Dose-response curve fitting (4-parameter Hill / log-logistic)
# ---------------------------------------------------------------------------
def hill_equation(c: np.ndarray, bottom: float, top: float, ec50: float, n_hill: float) -> np.ndarray:
    """4-parameter Hill equation per §2.1."""
    return bottom + (top - bottom) / (1.0 + (c / ec50) ** n_hill)


def fit_ic50(conc_viab: Sequence[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    """Fit §2.1 Hill equation; return fitted params + R² or ``None`` on fail."""
    if not _HAS_SCIPY:
        return None
    c = np.array([p[0] for p in conc_viab if p[0] > 0])
    v = np.array([p[1] for p in conc_viab if p[0] > 0])
    if c.size < 4 or v.size < 4:
        return None
    try:
        popt, _ = curve_fit(
            hill_equation, c, v,
            p0=[0.0, 1.0, 1.0, 1.5],
            bounds=([-0.2, 0.5, 1e-4, 0.1], [0.5, 1.5, 1e3, 5.0]),
            maxfev=2000,
        )
        bottom, top, ec50, n_hill = [float(x) for x in popt]
        predicted = hill_equation(c, bottom, top, ec50, n_hill)
        ss_res = float(np.sum((v - predicted) ** 2))
        ss_tot = float(np.sum((v - np.mean(v)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "bottom": bottom, "top": top, "ec50_uM": ec50,
            "hill_slope": n_hill, "r_squared": r_sq, "n_points": int(c.size),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# §3 pIC50 computation + 95% CI (bootstrap)
# ---------------------------------------------------------------------------
def pic50_from_ic50(ic50_uM: float) -> float:
    """§3.1 pIC50 = -log10(IC50_M) = 6 - log10(IC50_uM)."""
    return float(6.0 - np.log10(ic50_uM))


def bootstrap_pic50_ci(
    conc_viab: Sequence[Tuple[float, float]],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> Optional[Tuple[float, float]]:
    """§3.2 residual bootstrap 95% CI on IC50 → inverted to pIC50."""
    if not _HAS_SCIPY:
        return None
    c = np.array([p[0] for p in conc_viab if p[0] > 0])
    v = np.array([p[1] for p in conc_viab if p[0] > 0])
    if c.size < 4:
        return None
    fit = fit_ic50(conc_viab)
    if fit is None:
        return None
    predicted = hill_equation(c, fit["bottom"], fit["top"], fit["ec50_uM"], fit["hill_slope"])
    residuals = v - predicted
    rng = np.random.default_rng(seed)
    ic50_samples: List[float] = []
    for _ in range(n_boot):
        boot_v = predicted + rng.choice(residuals, size=residuals.size, replace=True)
        try:
            popt, _ = curve_fit(
                hill_equation, c, boot_v,
                p0=[fit["bottom"], fit["top"], fit["ec50_uM"], fit["hill_slope"]],
                bounds=([-0.2, 0.5, 1e-4, 0.1], [0.5, 1.5, 1e3, 5.0]),
                maxfev=1000,
            )
            ic50_samples.append(float(popt[2]))
        except Exception:
            continue
    if not ic50_samples:
        return None
    ic50_arr = np.array(ic50_samples)
    ic50_low, ic50_high = float(np.quantile(ic50_arr, 0.025)), float(np.quantile(ic50_arr, 0.975))
    # §3.2 inversion: lower IC50 → higher pIC50
    pic50_low = pic50_from_ic50(ic50_high)
    pic50_high = pic50_from_ic50(ic50_low)
    return pic50_low, pic50_high


# ---------------------------------------------------------------------------
# §4 Statistical analysis (per-metal ANOVA + post-hoc stub)
# ---------------------------------------------------------------------------
def anova_per_metal(
    per_compound_pic50: Dict[Tuple[str, str], Dict[str, float]],
    metals: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """§4.1 one-way ANOVA per metal across the 4 cell lines."""
    if not _HAS_SCIPY:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for metal in metals:
        cell_pic50: Dict[str, List[float]] = {c: [] for c in DEFAULT_CELL_LINES}
        for (cpd, cell), rec in per_compound_pic50.items():
            if rec.get("metal") != metal or cell in cell_pic50:
                cell_pic50[cell].append(rec["pic50"])
        groups = [g for g in cell_pic50.values() if len(g) >= 2]
        if len(groups) < 2:
            out[metal] = {"F": float("nan"), "p_raw": float("nan"), "n_groups": len(groups)}
            continue
        F, p = f_oneway(*groups)
        out[metal] = {"F": float(F), "p_raw": float(p), "n_groups": len(groups)}
    return out


# ---------------------------------------------------------------------------
# §5 In-silico vs wet-lab comparison (with Wf-58 D-MPNN predictions)
# ---------------------------------------------------------------------------
def agreement_per_metal_cell(
    per_compound_pic50: Dict[Tuple[str, str], Dict[str, float]],
    dmpnn_per_compound: Dict[Tuple[str, str], float],
) -> Dict[str, Dict[str, float]]:
    """§5.2 Pearson r + Spearman ρ per (metal × cell_line) with n ≥ 4 compounds."""
    if not _HAS_SCIPY:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for metal in DEFAULT_METALS:
        for cell in DEFAULT_CELL_LINES:
            wetlab, dmpnn = [], []
            for (cpd, c), rec in per_compound_pic50.items():
                if c != cell or rec.get("metal") != metal:
                    continue
                key = (cpd, cell)
                if key not in dmpnn_per_compound:
                    continue
                wetlab.append(rec["pic50"])
                dmpnn.append(dmpnn_per_compound[key])
            n = len(wetlab)
            if n < 4:
                out[f"{metal}-{cell}"] = {
                    "pearson_r": None, "spearman_rho": None, "n": n,
                    "gate_pass": False, "reason": f"n={n} < 4",
                }
                continue
            r_p, _ = pearsonr(wetlab, dmpnn)
            rho, _ = spearmanr(wetlab, dmpnn)
            out[f"{metal}-{cell}"] = {
                "pearson_r": float(r_p),
                "spearman_rho": float(rho),
                "n": n,
                "gate_pass": bool(abs(r_p) >= PEARSON_GATE),
            }
    return out


def load_dmpnn_predictions(path: Path) -> Dict[Tuple[str, str], float]:
    """Load Wf-58 P6.1 D-MPNN per-compound pIC50 (mean across 3 seeds × 4 cells)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    out: Dict[Tuple[str, str], float] = {}
    for rec in data.get("compounds", []):
        cpd = rec.get("compound_id")
        cell_pic50 = rec.get("per_cell_pic50", {})
        for cell, val in cell_pic50.items():
            out[(cpd, cell)] = float(val)
    return out


# ---------------------------------------------------------------------------
# §3.3 per-compound record assembly
# ---------------------------------------------------------------------------
def build_per_compound_records(
    wells: Sequence[PlateWell],
    n_boot: int = N_BOOTSTRAP,
) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Run §1-§3 on each (compound, cell_line); emit per-compound record."""
    norm = normalize_wells(wells)
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    metal_lookup = {w.compound_id: w.metal for w in wells if w.compound_id not in ("bg", "veh", "cisplatin")}
    for (cpd, cell), points in norm.items():
        fit = fit_ic50(points)
        ci = bootstrap_pic50_ci(points, n_boot=n_boot)
        if fit is None:
            out[(cpd, cell)] = {
                "rank": None, "compound_id": cpd, "metal": metal_lookup.get(cpd, "?"),
                "cell_line": cell, "ic50_uM": None, "pic50": None,
                "fit_quality": "fit_fail", "qc_flags": ["fit_fail"],
            }
            continue
        pic50 = pic50_from_ic50(fit["ec50_uM"])
        ci_low = ci[0] if ci is not None else None
        ci_high = ci[1] if ci is not None else None
        qc_flags = []
        if not (0.5 <= fit["hill_slope"] <= 3.0):
            qc_flags.append("hill_outlier")
        if not (fit["bottom"] <= 0.3 and fit["top"] >= 0.7):
            qc_flags.append("asymptote_outlier")
        if fit["n_points"] < 6:
            qc_flags.append("insufficient_doses")
        out[(cpd, cell)] = {
            "rank": None, "compound_id": cpd, "metal": metal_lookup.get(cpd, "?"),
            "cell_line": cell,
            "ic50_uM": float(fit["ec50_uM"]),
            "pic50": float(pic50),
            "pic50_ci_low": ci_low, "pic50_ci_high": ci_high,
            "hill_slope": float(fit["hill_slope"]),
            "r_squared": float(fit["r_squared"]),
            "n_doses": int(fit["n_points"]),
            "n_replicates": N_REPLICATES,
            "fit_quality": "pass" if not qc_flags else "flag",
            "qc_flags": qc_flags,
        }
    return out


# ---------------------------------------------------------------------------
# Template writer (schema-only, for PI handoff)
# ---------------------------------------------------------------------------
TEMPLATE_SCHEMA: Dict[str, object] = {
    "meta": {
        "n_compounds": 20, "n_cell_lines": 4, "n_pass_qc": None,
        "wetlab_data_source": "P6.2 wet-lab IC50 CSV (DEFERRED)",
        "dmpnn_data_source": "molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json",
        "pipeline_version": "wf74-v1.0",
        "framing": "honest-negative: schema-only template; wet-lab IC50 NOT MEASURED",
        "timestamp": None,
    },
    "compounds": [
        {
            "rank": 1, "compound_id": "platinai_0454", "metal": "Ir", "cell_line": "MCF7",
            "ic50_uM": None, "pic50": None, "pic50_ci_low": None, "pic50_ci_high": None,
            "n_doses": 8, "n_replicates": 3, "hill_slope": None,
            "fit_quality": "DEFERRED", "qc_flags": ["PENDING_WETLAB"],
            "dmpnn_pic50_predicted": 8.6172, "dmpnn_pic50_delta": None,
        }
    ],
    "anova_per_metal": {m: {"F": None, "p_raw": None, "p_bh": None} for m in DEFAULT_METALS},
    "agreement_per_metal_cell": {
        f"{m}-{c}": {"pearson_r": None, "pearson_r_ci_low": None, "pearson_r_ci_high": None,
                    "spearman_rho": None, "n": 0, "gate_pass": False}
        for m in DEFAULT_METALS for c in DEFAULT_CELL_LINES
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-74 Phase 3 — IC50 analysis pipeline TEMPLATE (synthetic smoke).",
    )
    p.add_argument("--input", type=Path, default=None,
                   help="Wet-lab long-form CSV per §1.1 (DEFERRED).")
    p.add_argument("--synthetic", action="store_true",
                   help="Run on synthetic fixture (no wet-lab CSV required).")
    p.add_argument("--dmpnn", type=Path, default=DEFAULT_DMPNN,
                   help=f"Wf-58 D-MPNN per_compound_pic50.json (default: {DEFAULT_DMPNN}).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT}).")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                   help=f"Bootstrap iterations for §3.2 (default: {N_BOOTSTRAP}).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve input: synthetic fixture or wet-lab CSV
    if args.synthetic:
        print("[wf74ic50] Building synthetic fixture (5 cpd × 4 cell × 8 dose × 3 rep)")
        wells = make_synthetic_fixture()
        framing = "synthetic fixture (smoke test; not wet-lab data)"
    elif args.input is not None and args.input.exists():
        print(f"[wf74ic50] Loading wet-lab CSV: {args.input}  -- NOT IMPLEMENTED (TODO §6)")
        print("[wf74ic50] Pipeline scaffold today; CSV reader ships when P6.2 wet-lab lands.")
        wells = make_synthetic_fixture()  # fall back to fixture
        framing = "wet-lab CSV scaffold (DEFERRED until P6.2)"
    else:
        print("[wf74ic50] ERROR: pass --synthetic OR --input <wetlab.csv>", file=sys.stderr)
        return 2

    # 2) Run pipeline
    per_compound = build_per_compound_records(wells, n_boot=args.n_bootstrap)
    print(f"[wf74ic50] Per-compound records: {len(per_compound)}")

    # 3) ANOVA per metal
    anova = anova_per_metal(per_compound, DEFAULT_METALS)
    print(f"[wf74ic50] ANOVA per metal: {anova}")

    # 4) In-silico vs wet-lab agreement (Wf-58 D-MPNN comparison)
    dmpnn = load_dmpnn_predictions(args.dmpnn)
    print(f"[wf74ic50] D-MPNN predictions loaded: {len(dmpnn)} (compound × cell)")
    agreement = agreement_per_metal_cell(per_compound, dmpnn)

    # 5) Attach dmpnn delta to each compound record
    for key, rec in per_compound.items():
        if rec.get("pic50") is not None:
            dmpnn_val = dmpnn.get(key)
            rec["dmpnn_pic50_predicted"] = dmpnn_val
            rec["dmpnn_pic50_delta"] = (dmpnn_val - rec["pic50"]) if dmpnn_val is not None else None

    # 6) Compute meta counters
    n_pass_qc = sum(1 for r in per_compound.values() if r.get("fit_quality") == "pass")
    n_excluded = sum(1 for r in per_compound.values() if r.get("fit_quality") != "pass")

    # 7) Write outputs
    wetlab_json = out_dir / "per_compound_pic50_wetlab.json"
    wetlab_payload = {
        "meta": {
            "n_compounds": len({k[0] for k in per_compound.keys()}),
            "n_cell_lines": len({k[1] for k in per_compound.keys()}),
            "n_pass_qc": n_pass_qc, "n_excluded": n_excluded,
            "wetlab_data_source": framing,
            "dmpnn_data_source": str(args.dmpnn),
            "pipeline_version": "wf74-v1.0",
            "framing": f"honest-negative: {framing}; wet-lab gate NOT active",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "compounds": list(per_compound.values()),
        "anova_per_metal": anova,
        "agreement_per_metal_cell": agreement,
    }
    wetlab_json.write_text(json.dumps(wetlab_payload, indent=2, default=str) + "\n")
    print(f"[wf74ic50] Wrote: {wetlab_json}")

    # Schema-only template (PI handoff)
    template_json = out_dir / "template_per_compound.json"
    template_payload = dict(TEMPLATE_SCHEMA)
    template_payload["meta"] = dict(TEMPLATE_SCHEMA["meta"])
    template_payload["meta"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    template_payload["meta"]["framing"] = "honest-negative: schema-only template; wet-lab IC50 NOT MEASURED"
    template_json.write_text(json.dumps(template_payload, indent=2) + "\n")
    print(f"[wf74ic50] Wrote: {template_json}")

    # Synthesis markdown
    synth_md = out_dir / "synthesis.md"
    synth_md.write_text(_build_synthesis_md(per_compound, anova, agreement, framing))
    print(f"[wf74ic50] Wrote: {synth_md}")

    print(f"\n[wf74ic50] DONE. n_records={len(per_compound)}; n_pass_qc={n_pass_qc}; "
          f"framing='{framing}'")
    return 0


def _build_synthesis_md(
    per_compound: Dict[Tuple[str, str], Dict[str, object]],
    anova: Dict[str, Dict[str, float]],
    agreement: Dict[str, Dict[str, float]],
    framing: str,
) -> str:
    """Build human-readable synthesis markdown per §5.5."""
    lines = [
        "# Wf-74 Phase 3 — IC50 analysis pipeline synthesis",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Source**: {framing}",
        "**Pipeline**: §1 normalize → §2 Hill fit → §3 pIC50 + bootstrap CI → §4 ANOVA → §5 agreement",
        "",
        "## Per-compound × per-cell pIC50 (first 5 records)",
        "",
        "| Compound | Cell | Metal | IC50 µM | pIC50 | 95% CI | Hill | QC |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for i, ((cpd, cell), rec) in enumerate(per_compound.items()):
        if i >= 5:
            break
        ic50 = rec.get("ic50_uM")
        pic50 = rec.get("pic50")
        ci_low = rec.get("pic50_ci_low")
        ci_high = rec.get("pic50_ci_high")
        hill = rec.get("hill_slope")
        qc = ",".join(rec.get("qc_flags", [])) or "pass"
        ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]" if ci_low is not None else "—"
        ic50_str = f"{ic50:.4f}" if ic50 is not None else "—"
        pic50_str = f"{pic50:.3f}" if pic50 is not None else "—"
        hill_str = f"{hill:.2f}" if hill is not None else "—"
        lines.append(
            f"| {cpd} | {cell} | {rec.get('metal', '?')} | "
            f"{ic50_str} | {pic50_str} | {ci_str} | {hill_str} | {qc} |"
        )
    lines += [
        "",
        "## ANOVA per metal",
        "",
        "| Metal | F | p_raw | n_groups |",
        "|---|---:|---:|---:|",
    ]
    for metal, res in anova.items():
        F = res.get("F", float("nan"))
        p = res.get("p_raw", float("nan"))
        ng = res.get("n_groups", 0)
        F_str = f"{F:.4f}" if not np.isnan(F) else "—"
        p_str = f"{p:.4f}" if not np.isnan(p) else "—"
        lines.append(f"| {metal} | {F_str} | {p_str} | {ng} |")
    lines += [
        "",
        "## In-silico vs wet-lab agreement (per metal × cell_line)",
        "",
        f"**Gate**: Pearson r ≥ {PEARSON_GATE}",
        "| Cohort | Pearson r | Spearman ρ | n | Gate |",
        "|---|---:|---:|---:|:-:|",
    ]
    for cohort, res in agreement.items():
        r = res.get("pearson_r")
        rho = res.get("spearman_rho")
        n = res.get("n", 0)
        gate = res.get("gate_pass", False)
        r_str = f"{r:.4f}" if r is not None else "—"
        rho_str = f"{rho:.4f}" if rho is not None else "—"
        gate_str = "PASS" if gate else ("FAIL" if r is not None else "n<4")
        lines.append(f"| {cohort} | {r_str} | {rho_str} | {n} | {gate_str} |")
    lines += [
        "",
        "## Honest framing (per party-twenty-spirit §3.5)",
        "",
        "* **Wet-lab gate NOT active**: this pipeline ships today as a template + synthetic smoke.",
        "* No real wet-lab IC50 has been collected; all numbers above (except D-MPNN "
        "predictions from Wf-58 P6.1) are **synthetic**. Treat as a pipeline smoke test.",
        "* D-MPNN pIC50 predictions (Wf-58 P6.1) are CPU-only bench quality on "
        "synthetic IC50 labels (Rusanov 2025 tables S4-S5 are behind Cloudflare); "
        "non-Pt candidates carry cross-metal transfer caveats.",
        "* Paper §5 wet-lab validation section is 🚫 DEFERRED per TODO-84 + the "
        "wet-lab gate is itself PROJECTED on Wf-58 P6.2 timeline (6-month).",
        "",
        "## Cross-references",
        "",
        "* `TODO/pending/wetlab_ic50_analysis_pipeline.md` — full pipeline spec",
        "* `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` — D-MPNN input",
        "* `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` — Wf-58 P6.0/P6.1/P6.2 ship",
        "* `TODO/pending/84_no_paper_task_without_authorization.md` — paper rule",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
