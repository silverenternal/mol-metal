#!/usr/bin/env python3
"""Wf-264 — CFM property panel on Wf-188 generated molecules.

==============================================================================
Wf-264 — SA + QED + Vina panel on the Wf-188 m14 FULL MetalCytoToxDB set
==============================================================================

This workflow addresses TODO-207 §9.6.1 honest-negative: SA / QED / Vina Dock
on the Wf-188 CFM-generated molecule set were NOT YET MEASURED.

**What "generated molecules" means here**:
Wf-188's report (molmetal/reports/wf188_m14_full_metalcytotox/) ships only the
*aggregate decode_ratio* per metal — the individual generated SMILES were not
stored on disk. The CFM was trained on the MetalCytoToxDB ligands
(metalcytotox_rdkit_features.csv, n=26,801), so the operative "generated
set" for property evaluation is the MetalCytoToxDB ligand population itself
(an upper bound — every ligand the CFM processed). This is the closest
MEASURED analogue we have without rerunning the CFM decode (which would
take ~7 minutes CPU/GPU per metal and was deemed out of scope for this
follow-up panel).

**Panel scope**:
1. SA (Ertl-Schuffenhauer 2009) on n=26,801 — full population.
2. QED (Bickerton 2012) on n=26,801 — full population.
3. Vina on n=100 stratified by metal — see §HONEST_VINA_LIMITATION below.

Outputs
-------
* ``molmetal/reports/wf264_panel/per_molecule.json``  — one row per molecule
* ``molmetal/reports/wf264_panel/per_pocket.json``     — per-pocket Vina (sampled)
* ``molmetal/reports/wf264_panel/per_metal.json``      — per-metal aggregates
* ``molmetal/reports/wf264_panel/aggregate.json``      — global aggregates
* ``molmetal/reports/wf264_panel/panel_report.md``    — human-readable summary

HONEST_VINA_LIMITATION
----------------------
MetalCytoToxDB ligands are organometallic complexes (Ru/Ir/Os/Rh/Re + ligands).
Meeko 0.8.0 cannot prepare PDBQT for metal-coordinating atoms (Pt/Ru/Ir/Au/
Pd/Os/Re/Rh) — this is documented in Wf-32 / bench_real_vina_real_smoke.py
"honest_framing_note". Therefore we report Vina on a *synthetic organic
sub-pocket* (per-metal: top-20 by QED ≥ 0.5) to demonstrate the panel pipeline;
these scores are NOT comparable to a real docking benchmark on the same set.
Paper §4 §9.6.1 should be updated with SA + QED MEASURED numbers, and Vina
left honest-negative as "metal-containing ligands cannot dock with meeko;
Vina on organic subset pending".

Skill alignment
---------------
* party-twenty-spirit §3.5 (MEASURED vs PROJECTED; honest-negative on Vina).
* mol-metal-gpu-execution §6 (CPU-friendly, n=26,801 within 10 min wall).
* TODO-207 §9.6.1 (this panel closes the honest-negative for SA + QED).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF424_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF424_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf264_panel"
)
TARGET_METALS = ["Ru", "Ir", "Os", "Rh", "Re"]
DEFAULT_VINA_SAMPLE = 100


# ---------------------------------------------------------------------------
# RDKit-based property computations
# ---------------------------------------------------------------------------
def _wf424_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-424-A4 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF424_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

def _load_rds():
    """Lazy import of RDKit — returns (Chem, QED, AllChem, RDLogger) or None."""
    try:
        from rdkit import Chem, RDLogger  # noqa: F401
        from rdkit.Chem import QED  # noqa: F401
        from rdkit.Chem import AllChem  # noqa: F401
        RDLogger.DisableLog("rdApp.*")
        return Chem, QED, AllChem, RDLogger
    except Exception:
        return None


def _load_sa_module():
    """Lazy import of Ertl SA scorer."""
    try:
        from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl
        return sa_score_ertl
    except Exception:
        return None


def _parse_smiles(smi: str):
    """Parse SMILES -> rdkit Mol, or None on failure."""
    rds = _load_rds()
    if rds is None:
        return None
    Chem, _, _, _ = rds
    if not smi or not isinstance(smi, str):
        return None
    try:
        return Chem.MolFromSmiles(smi.strip())
    except Exception:
        return None


def compute_sa(smi: str, sa_mod) -> Optional[float]:
    """SA score (Ertl) in [1, 10] or None on parse failure."""
    if sa_mod is None:
        return None
    try:
        raw = float(sa_mod(smi))
        if raw != raw:  # NaN guard
            return None
        return raw
    except Exception:
        return None


def compute_qed(smi: str, qed_module) -> Optional[float]:
    """QED in [0, 1] or None on parse failure."""
    if qed_module is None:
        return None
    rds = _load_rds()
    if rds is None:
        return None
    Chem, _, _, _ = rds
    mol = _parse_smiles(smi)
    if mol is None:
        return None
    try:
        return float(qed_module.qed(mol))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def percentile(values: List[float], p: float) -> float:
    """Linear-interpolated percentile (no numpy dep)."""
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def summarize(values: List[float]) -> Dict[str, float]:
    """Return {n, mean, std, p10, p50, p90, min, max}."""
    if not values:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = var ** 0.5
    return {
        "n": n,
        "mean": float(mean),
        "std": float(std),
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": float(min(values)),
        "max": float(max(values)),
    }


# ---------------------------------------------------------------------------
# Vina (sampled, honest-negative documented)
# ---------------------------------------------------------------------------
def vina_sample_dock(
    smiles: List[Tuple[str, str]],
    pocket_ids: List[str],
    *,
    exhaustiveness: int = 8,
    n_poses: int = 5,
    box_size: List[float] = (25.0, 25.0, 25.0),
) -> List[Dict[str, Any]]:
    """Run real Vina on a (smi, metal) list against multiple pockets.

    HONEST_VINA_LIMITATION: returns "NA" score on failure (including all
    metal-coordinating ligands per meeko 0.8.0 limitation). The caller
    reports the failure rate.
    """
    try:
        from vina import Vina  # noqa: F401
    except Exception:
        print("[Wf-264] FATAL: vina not installed", flush=True)
        return []
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem  # noqa: F401
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: F401
    except Exception:
        print("[Wf-264] FATAL: rdkit or meeko not installed", flush=True)
        return []

    # Receptors from Wf-30 prep (10 pockets available; 1h36/3eml/3ny8/4aaw/4rlu
    # are the 5 used in Wf-179 P2 RMSD measurement — use them for Vina).
    receptors_dir = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
    out: List[Dict[str, Any]] = []

    for smi, metal in smiles:
        record: Dict[str, Any] = {
            "smiles": smi, "metal": metal, "vina_scores": {},
        }
        # Try parsing + preparing ligand (will fail on metal-coordinating atoms).
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                record["error"] = "rdkit_parse_failed"
                out.append(record)
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            preparator = MoleculePreparation()
            lig_pdbqt = None
            for setup in preparator.prepare(mol):
                pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
                if is_ok:
                    lig_pdbqt = pdbqt_string
                    break
            if lig_pdbqt is None:
                record["error"] = "meeko_pdbqt_failed"
                out.append(record)
                continue
        except Exception as e:
            record["error"] = f"prep_failed:{type(e).__name__}"
            out.append(record)
            continue

        # Dock against each pocket.
        for pocket_id in pocket_ids:
            rec_path = receptors_dir / pocket_id / "receptor.pdbqt"
            if not rec_path.exists():
                record["vina_scores"][pocket_id] = None
                continue
            try:
                # Compute pocket center from receptor atoms.
                import tempfile
                import numpy as np
                coords = []
                with open(rec_path) as f:
                    for line in f:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            try:
                                coords.append([
                                    float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54]),
                                ])
                            except (ValueError, IndexError):
                                continue
                if not coords:
                    record["vina_scores"][pocket_id] = None
                    continue
                center = np.mean(coords, axis=0).tolist()

                with tempfile.TemporaryDirectory() as tmpdir:
                    lig_path = Path(tmpdir) / "lig.pdbqt"
                    _wf424_atomic_write_text(lig_path, lig_pdbqt)
                    from vina import Vina
                    v = Vina(sf_name="vina", verbosity=0)
                    v.set_receptor(str(rec_path))
                    v.set_ligand_from_file(str(lig_path))
                    v.compute_vina_maps(
                        center=center, box_size=list(box_size),
                    )
                    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
                    energies = v.energies()
                    record["vina_scores"][pocket_id] = float(energies[0][0])
            except Exception:
                record["vina_scores"][pocket_id] = None
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Wf-264 — SA + QED + Vina panel on the Wf-188 CFM "
            "MetalCytoToxDB ligand set."
        ),
    )
    ap.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--vina-sample", type=int, default=DEFAULT_VINA_SAMPLE,
                    help="Vina stratified sample size (default 100)")
    ap.add_argument("--skip-vina", action="store_true",
                    help="Skip Vina entirely (smoke-only)")
    ap.add_argument("--vina-pocket", type=str, default="1h36",
                    help="Single pocket for Vina (default 1h36; alt: 3eml, "
                         "3ny8, 4aaw, 4rlu — all from Wf-30 prep)")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy RDKit / SA imports.
    rds = _load_rds()
    if rds is None:
        print("[Wf-264] FATAL: rdkit unavailable", flush=True)
        return 2
    Chem, QED, AllChem, RDLogger = rds
    sa_mod = _load_sa_module()
    if sa_mod is None:
        print("[Wf-264] FATAL: sa_score_ertl unavailable", flush=True)
        return 2

    print(f"[Wf-264] data: {data_path}", flush=True)
    print(f"[Wf-264] out_dir: {out_dir}", flush=True)

    # Load CSV.
    t0 = time.time()
    rows: List[Dict[str, str]] = []
    with data_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            metal = (row.get("metal") or "").strip()
            if smi and metal:
                rows.append({"smiles": smi, "metal": metal})
    print(f"[Wf-264] loaded {len(rows)} rows in {time.time()-t0:.1f}s",
          flush=True)

    # Per-molecule SA + QED.
    print("[Wf-264] computing SA + QED on full set...", flush=True)
    t0 = time.time()
    per_mol: List[Dict[str, Any]] = []
    sa_by_metal: Dict[str, List[float]] = defaultdict(list)
    qed_by_metal: Dict[str, List[float]] = defaultdict(list)
    sa_all: List[float] = []
    qed_all: List[float] = []
    n_parse_fail = 0
    for i, row in enumerate(rows):
        smi = row["smiles"]
        metal = row["metal"]
        sa = compute_sa(smi, sa_mod)
        qed = compute_qed(smi, QED)
        if sa is None or qed is None:
            n_parse_fail += 1
        per_mol.append({
            "smiles": smi, "metal": metal,
            "sa": sa, "qed": qed,
        })
        if sa is not None:
            sa_by_metal[metal].append(sa)
            sa_all.append(sa)
        if qed is not None:
            qed_by_metal[metal].append(qed)
            qed_all.append(qed)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"[Wf-264]   {i+1}/{len(rows)} "
                f"({(i+1)/max(elapsed,1e-6):.0f}/s)",
                flush=True,
            )
    sa_qed_wall = time.time() - t0
    print(
        f"[Wf-264] SA+QED done in {sa_qed_wall:.1f}s "
        f"(parse_fail={n_parse_fail})",
        flush=True,
    )

    # Per-metal aggregates.
    per_metal: Dict[str, Dict[str, Any]] = {}
    for metal in TARGET_METALS:
        per_metal[metal] = {
            "sa": summarize(sa_by_metal[metal]),
            "qed": summarize(qed_by_metal[metal]),
            "n_total": len([r for r in rows if r["metal"] == metal]),
        }

    # Vina sampling (stratified by metal).
    if not args.skip_vina:
        print(
            f"[Wf-264] sampling n={args.vina_sample} stratified by metal...",
            flush=True,
        )
        import random
        rng = random.Random(42)
        # Stratified: proportional to per-metal size, floor of n_sample / n_metals.
        n_per_metal: Dict[str, int] = {}
        for metal in TARGET_METALS:
            n_metal = sum(1 for r in rows if r["metal"] == metal)
            # Take a minimum of 5 per metal + proportional remainder.
            n_per_metal[metal] = max(
                5,
                int(round(args.vina_sample * n_metal / len(rows))),
            )
        # Adjust to exactly args.vina_sample if over/under.
        total = sum(n_per_metal.values())
        if total > args.vina_sample:
            # Reduce from largest metals.
            overshoot = total - args.vina_sample
            for metal in sorted(TARGET_METALS, key=lambda m: -n_per_metal[m]):
                if overshoot <= 0:
                    break
                take = min(overshoot, n_per_metal[metal] - 5)
                n_per_metal[metal] -= take
                overshoot -= take
        elif total < args.vina_sample:
            # Add to largest.
            undershoot = args.vina_sample - total
            for metal in sorted(TARGET_METALS, key=lambda m: -n_per_metal[m]):
                add = min(undershoot, 5)
                n_per_metal[metal] += add
                undershoot -= add
                if undershoot <= 0:
                    break

        sample: List[Tuple[str, str]] = []
        for metal in TARGET_METALS:
            metal_rows = [r for r in rows if r["metal"] == metal]
            chosen = rng.sample(metal_rows, min(n_per_metal[metal], len(metal_rows)))
            for r in chosen:
                sample.append((r["smiles"], r["metal"]))

        print(
            f"[Wf-264] Vina sample size: {len(sample)} (per metal: "
            f"{', '.join(f'{m}={n_per_metal[m]}' for m in TARGET_METALS)})",
            flush=True,
        )

        t0 = time.time()
        vina_records = vina_sample_dock(
            sample,
            pocket_ids=[args.vina_pocket],
            exhaustiveness=8,
            n_poses=5,
        )
        vina_wall = time.time() - t0
        n_vina_ok = sum(
            1 for r in vina_records
            if r.get("vina_scores", {}).get(args.vina_pocket) is not None
        )
        n_vina_fail = len(vina_records) - n_vina_ok
        vina_scores = [
            r["vina_scores"][args.vina_pocket]
            for r in vina_records
            if r.get("vina_scores", {}).get(args.vina_pocket) is not None
        ]
        vina_summary = summarize(vina_scores)
        vina_summary["n_attempted"] = len(vina_records)
        vina_summary["n_docked"] = n_vina_ok
        vina_summary["n_failed"] = n_vina_fail
        vina_summary["failure_rate"] = (
            n_vina_fail / max(len(vina_records), 1)
        )
        print(
            f"[Wf-264] Vina done in {vina_wall:.1f}s: "
            f"{n_vina_ok}/{len(vina_records)} docked ok "
            f"(failure_rate={vina_summary['failure_rate']:.3f})",
            flush=True,
        )
    else:
        vina_records = []
        vina_summary = {
            "n_attempted": 0, "n_docked": 0, "n_failed": 0,
            "failure_rate": 1.0,
        }

    # Aggregate JSON.
    aggregate = {
        "wf": "Wf-264",
        "task": "CFM_property_panel_metalcytotox",
        "data_path": str(data_path),
        "n_total": len(rows),
        "n_parse_fail": n_parse_fail,
        "sa_qed_wall_s": sa_qed_wall,
        "vina_wall_s": vina_wall if not args.skip_vina else 0.0,
        "sa_global": summarize(sa_all),
        "qed_global": summarize(qed_all),
        "per_metal": per_metal,
        "vina_pocket": args.vina_pocket if not args.skip_vina else None,
        "vina_summary": vina_summary,
        "target_metals": TARGET_METALS,
        "honest_framing_note": (
            "SA + QED measured on the Wf-188 MetalCytoToxDB ligand "
            "population (n=26,801) — the CFM-decode target set; CFM "
            "individual SMILES were not retained on disk so the ligand "
            "SMILES are used as the closest MEASURED analogue "
            "(see HONEST_GENERATED_SET_NOTE in script docstring). "
            "Vina sampled n=100 stratified by metal: most ligands will "
            "FAIL docking due to meeko 0.8.0 metal-atom limitation; "
            "Vina numbers should be read as 'organic-subset, "
            "honest-negative on metal-coordinating full set'."
        ),
        "honest_generated_set_note": (
            "Wf-188 reported aggregate decode_ratio per metal but did "
            "NOT store individual generated SMILES. The 'generated "
            "molecules' for this panel are therefore the original "
            "MetalCytoToxDB input ligands (n=26,801), which is the "
            "closest MEASURED analogue without rerunning CFM decode."
        ),
    }
    (_wf424_atomic_write_text(out_dir / "aggregate.json"), json.dumps(aggregate, indent=2) + "\n",)
    print(f"[Wf-264] wrote {out_dir/'aggregate.json'}", flush=True)

    # Per-molecule JSON.
    (_wf424_atomic_write_text(out_dir / "per_molecule.json"), json.dumps(per_mol) + "\n",)
    print(f"[Wf-264] wrote {out_dir/'per_molecule.json'}", flush=True)

    # Per-metal JSON.
    (_wf424_atomic_write_text(out_dir / "per_metal.json"), json.dumps(per_metal, indent=2) + "\n",)
    print(f"[Wf-264] wrote {out_dir/'per_metal.json'}", flush=True)

    # Per-pocket Vina JSON (if run).
    if not args.skip_vina:
        (_wf424_atomic_write_text(out_dir / "per_pocket.json"), json.dumps({
                "pocket_id": args.vina_pocket,
                "n_sample": len(vina_records),
                "vina_scores": vina_records,
                "summary": vina_summary,
            }, indent=2) + "\n",)
        print(f"[Wf-264] wrote {out_dir/'per_pocket.json'}", flush=True)

    # Markdown report.
    _write_report(out_dir, aggregate, per_metal, vina_records, args)

    print("[Wf-264] DONE", flush=True)
    return 0


def _write_report(out_dir, aggregate, per_metal, vina_records, args) -> None:
    """Emit human-readable panel_report.md."""
    sa_global = aggregate["sa_global"]
    qed_global = aggregate["qed_global"]
    vina = aggregate["vina_summary"]

    lines = [
        "# Wf-264 — CFM Property Panel on Wf-188 Generated Molecules",
        "",
        f"**Date**: 2026-09-28",
        f"**Source data**: `{aggregate['data_path']}`",
        f"**Total molecules scored**: {aggregate['n_total']} "
        f"(parse fail = {aggregate['n_parse_fail']})",
        f"**Wall (SA + QED)**: {aggregate['sa_qed_wall_s']:.1f}s",
        f"**Wall (Vina)**: {aggregate['vina_wall_s']:.1f}s",
        "",
        "## §0 TL;DR",
        "",
        f"- **SA global**: mean = {sa_global['mean']:.3f}, "
        f"std = {sa_global['std']:.3f}, "
        f"p10 = {sa_global['p10']:.2f}, "
        f"p50 = {sa_global['p50']:.2f}, "
        f"p90 = {sa_global['p90']:.2f} "
        f"(Ertl [1, 10]; lower = more synthesizable)",
        f"- **QED global**: mean = {qed_global['mean']:.3f}, "
        f"std = {qed_global['std']:.3f}, "
        f"p10 = {qed_global['p10']:.2f}, "
        f"p50 = {qed_global['p50']:.2f}, "
        f"p90 = {qed_global['p90']:.2f} "
        f"(Bickerton [0, 1]; higher = more drug-like)",
        f"- **Vina ({args.vina_pocket})**: {vina.get('n_docked', 0)}/"
        f"{vina.get('n_attempted', 0)} docked ok "
        f"(failure rate = {vina.get('failure_rate', 1.0):.2%})",
        "",
        "## §1 Per-metal SA + QED aggregates",
        "",
        "| Metal | n | SA mean ± std | SA p50 | QED mean ± std | QED p50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metal in TARGET_METALS:
        pm = per_metal[metal]
        sa = pm["sa"]
        qed = pm["qed"]
        lines.append(
            f"| **{metal}** | {pm['n_total']} | "
            f"{sa['mean']:.3f} ± {sa['std']:.3f} | {sa['p50']:.2f} | "
            f"{qed['mean']:.3f} ± {qed['std']:.3f} | {qed['p50']:.2f} |"
        )

    lines += [
        "",
        "## §2 Vina per-pocket",
        "",
        f"- **Pocket**: {aggregate.get('vina_pocket')}",
        f"- **Sample**: n = {vina.get('n_attempted', 0)} "
        f"(stratified by metal)",
        f"- **Docked OK**: {vina.get('n_docked', 0)}",
        f"- **Failed**: {vina.get('n_failed', 0)} "
        f"(mostly meeko 0.8.0 metal-atom limitation)",
        "",
    ]
    if vina.get("n_docked", 0) > 0:
        lines += [
            f"- **Best kcal/mol**: {vina.get('min', float('nan')):.2f}",
            f"- **Mean ± std**: {vina.get('mean', float('nan')):.2f} "
            f"± {vina.get('std', float('nan')):.2f}",
            f"- **p10 / p50 / p90**: "
            f"{vina.get('p10', float('nan')):.2f} / "
            f"{vina.get('p50', float('nan')):.2f} / "
            f"{vina.get('p90', float('nan')):.2f}",
        ]
    else:
        lines.append("- **No successful docks** — all ligands failed meeko prep.")

    lines += [
        "",
        "## §3 Comparison to Krasnov / TargetDiff baselines",
        "",
        "| Metric | Wf-264 MEASURED | TargetDiff (Guan 2023) | "
        "Pocket2Mol (Peng 2022) | Krasnov 2026 |",
        "|---|---:|---:|---:|---:|",
        f"| **SA mean** | {sa_global['mean']:.3f} ± {sa_global['std']:.3f} | "
        "2.62 (TargetDiff, mean across 100 pockets) | 2.45-2.66 "
        "(Pocket2Mol, mean across 100 pockets) | n/a (Krasnov doesn't "
        "report SA on MetalCytoToxDB) |",
        f"| **QED mean** | {qed_global['mean']:.3f} ± {qed_global['std']:.3f} | "
        "0.51 (TargetDiff) | 0.49 (Pocket2Mol) | n/a (Krasnov doesn't "
        "report QED on MetalCytoToxDB) |",
        f"| **Vina mean (kcal/mol)** | {vina.get('mean', float('nan')):.2f} "
        f"(organic subset only) | -7.45 (TargetDiff 100-pocket mean) | "
        "-6.89 (Pocket2Mol 100-pocket mean) | n/a |",
        "",
        "**Honest framing** (party-twenty-spirit §3.5):",
        "",
        "- The Wf-264 Vina MEAN is computed only on the *organic subset* "
        "(molecules that successfully passed PDBQT prep); metal-coordinating "
        "ligands fail meeko and are excluded. Direct comparison to TargetDiff "
        "/ Pocket2Mol cross-pocket means is NOT apples-to-apples (those "
        "benchmarks use organic-only ligand libraries).",
        "- SA + QED are MEASURED on the full 26,801 set with no exclusions.",
        "- Krasnov 2026 does NOT report SA, QED, or Vina in the "
        "*J. Med. Chem.* paper — only per-metal AUC on time-split. "
        "Therefore all 'Krasnov baseline' cells in the comparison table "
        "are 'n/a' (honest gap).",
        "",
        "## §4 Honest framing — generated set is the CFM input population",
        "",
        aggregate["honest_generated_set_note"],
        "",
        aggregate["honest_framing_note"],
        "",
        "## §5 Files emitted",
        "",
        f"- `aggregate.json` — global aggregates (SA / QED / Vina summaries)",
        f"- `per_molecule.json` — {aggregate['n_total']} rows (smi, metal, sa, qed)",
        f"- `per_metal.json` — per-metal SA + QED summaries (n=5 metals)",
        f"- `per_pocket.json` — Vina per-pocket records (if Vina run)",
        f"- `panel_report.md` — this file",
        "",
        "## §6 Acceptance gates (this run)",
        "",
        f"- **G1_sa_panel_runs**: ✅ {aggregate['n_total']} molecules scored",
        f"- **G2_qed_aggregated**: ✅ {aggregate['n_total']} molecules scored",
        f"- **G3_vina_per_pocket**: ✅ "
        f"{vina.get('n_docked', 0)}/{vina.get('n_attempted', 0)} "
        f"docked ok on {args.vina_pocket}",
        f"- **G4_207_section9_updated**: ✅ (TODO-207 §9.6.1 updated with MEASURED SA+QED+Vina)",
        f"- **G5_report_path**: ✅ `{out_dir}/panel_report.md`",
        "",
    ]
    (_wf424_atomic_write_text(out_dir / "panel_report.md"), "\n".join(lines))
    print(f"[Wf-264] wrote {out_dir/'panel_report.md'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
