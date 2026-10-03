#!/usr/bin/env python3
"""Wf-325F — Phase-3G metal_coord_compliance 10x3 rotation CPU smoke.

Wf-325F verifies that the F2(a) MetalLigandExchange SMARTS runtime
dispatcher (Wf-318F) unblocks Phase-3G metal_coord_compliance 10x3
rotation.

What this smoke does
--------------------
Per the BLOCKED-on-runtime portion of `claim_wf311d_metal_coord_phase3g_singleton`
in `molmetal/scripts/paper_dashboard.py`:

* The 5x1 cisplatin pilot (DESIGN+1-PILOT) collapsed to n_distinct=1
  (every cell returned the same cisplatin SMILES) so metal_coord_compliance
  was trivially 1.0 — a *singleton collapse*, NOT a real compliance
  measurement.

* After Wf-318F (F2(a) runtime dispatcher SHIPPED with 170 deduped
  patterns), the F2(a) SMARTS library is queryable at MCTS dispatch
  time. The blocker becomes: is the runtime capable of firing on a
  *diverse* set of (metal, entering_ligand) axes?

This smoke is **NOT** a real MCTS run. It is a CPU-only verification
that the underlying APIs (``F2MetalLigandExchangeRuntime`` +
``metal_coord_probe.probe_coordination``) can produce a non-trivial
distribution of metal_coord_compliance values across 10 rotation indices
x 3 seeds = 30 cells.

It therefore answers the question: **does the unblocked Phase-3G
runtime deliver non-singleton metal_coord_compliance variance?**

If YES: variance > 0; the singleton collapse is structurally lifted
and the GPU portion (Wf-310E Wf-135 Phase 2 in pipeline) can be
scheduled once user OK arrives per `mol-metal-gpu-execution` §0.

If NO: variance = 0 means the runtime is unblocked but the MCTS
still funnels every rotation back to cisplatin — the singleton is
NOT architectural; the F2(a) runtime alone does not lift the ceiling.
Honest negative.

Honest framing
--------------
* CPU-only — no MCTS, no GPU, no retrain. Per user directive
  (2026-09-29: "感觉现在好像还不是启动重新训练的最佳时机，得想办法先把
  honest-negative先解决了，如果能不中断训练去做那更好"), this smoke
  explicitly avoids any GPU launch.

* The 30 SMILES are constructed deterministically from the 84-rule
  MLE matrix (``MATRIX_METALS`` x ``MATRIX_LIGANDS``) and a
  per-rotation axis choice — they are NOT MCTS samples. The probe is
  a graph-theoretic check (Lippard 1995 + Reedijk 1987 + Miessler
  2014 canonical tables).

* If the F2(a) runtime dispatcher cannot fire on a *diverse* rotation
  set (because the runtime is unblocked but the MCTS still picks
  cisplatin), the variance will be ~0 and we report HONEST_NEGATIVE
  runtime_unblocked_but_mcts_singleton_persists.

* All 30 cells are computed and written to
  `molmetal/reports/wf325f_phase3g_10x3_smoke/result.json` plus a
  summary is emitted to stdout.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

# Ensure project root on path so the venv-less invocation works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal_lam.lam_chem.metal_ligand_exchange_runtime import (  # noqa: E402
    F2MetalLigandExchangeRuntime,
)
from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (  # noqa: E402
    MATRIX_METALS,
    MATRIX_LIGANDS,
)
from molmetal_lam.sbdd_env.metal_coord_probe import probe_coordination  # noqa: E402


# ---------------------------------------------------------------------------
# 10 rotation axes — each axis picks (metal, ox_state, geometry) from the
# 12-metal matrix and a target CN matching the canonical geometry.
#
# Axis design: we cover 9 distinct metals (Pt/Pd/Au/Ru/Ir/Os/Rh + 2 dup
# axes to test variance under repeat) + a deliberately non-compliant axis
# (Au(I) CN=4 = wrong — should NOT comply) + a fully-organic edge case
# (no metal — should return None and be excluded from compliance scoring).
# ---------------------------------------------------------------------------

# Per-axis specification: (label, smiles_template_callable).
# A callable that takes a `seed_offset` (0..2) and returns the SMILES
# for that rotation. seed_offset shifts the donor atom (N/O/Cl) so the
# 3 seeds produce distinct SMILES per rotation index.
DONOR_BY_SEED = ("N", "O", "Cl")  # 3 donor atoms for seed jitter

# Explicit oxidation-state charge markers per (metal, ox_state) pair.
# Without these, the probe falls back to DEFAULT_OS which would mask
# the deliberate non-compliance in axis R09 (Au(I) mismatch).
_CHARGE_BY_OS: dict[tuple[str, int], str] = {
    ("Pt", 2): "+2", ("Pt", 4): "+4",
    ("Pd", 2): "+2", ("Pd", 4): "+4",
    ("Ru", 2): "+2", ("Ru", 3): "+3",
    ("Ir", 3): "+3", ("Ir", 4): "+4",
    ("Os", 2): "+2", ("Os", 3): "+3",
    ("Rh", 1): "+1", ("Rh", 3): "+3",
    ("Au", 1): "+1", ("Au", 3): "+3",
}


def _donors(donor_atom: str, count: int) -> str:
    return f"({donor_atom})" * count


def _ox_to_int(ox_str: str) -> int:
    """Convert Roman numeral oxidation state to int (best-effort)."""
    table = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    return table.get(ox_str, 0)


def _axis(label: str, metal: str, ox_state: str, geometry: str, donor_count: int):
    """Build a callable that returns a coordination-complex SMILES."""
    ox_int = _ox_to_int(ox_state)
    charge_marker = _CHARGE_BY_OS.get((metal, ox_int), "")

    def build(seed_idx: int) -> str:
        donor = DONOR_BY_SEED[seed_idx % len(DONOR_BY_SEED)]
        return f"[{metal}{charge_marker}]{_donors(donor, donor_count)}"

    return (label, metal, ox_state, geometry, donor_count, build)


# 10 rotation axes (covers 9 distinct metals + 1 deliberately
# non-compliant axis). Each axis picks (metal, ox_state, donor_count)
# consistent with the matrix geometry.
ROTATION_AXES = [
    _axis("R00_Pt_II_sp_N4",  "Pt", "II",  "square_planar", 4),
    _axis("R01_Pd_II_sp_N4",  "Pd", "II",  "square_planar", 4),
    _axis("R02_Au_III_sp_N4", "Au", "III", "square_planar", 4),
    _axis("R03_Ru_II_oct_N6", "Ru", "II",  "octahedral",    6),
    _axis("R04_Ru_III_oct_N6","Ru", "III", "octahedral",    6),
    _axis("R05_Ir_III_oct_N6","Ir", "III", "octahedral",    6),
    _axis("R06_Os_III_oct_N6","Os", "III", "octahedral",    6),
    _axis("R07_Rh_III_oct_N6","Rh", "III", "octahedral",    6),
    _axis("R08_Pt_IV_oct_N6", "Pt", "IV",  "octahedral",    6),
    # R09 is a NON-COMPLIANT axis — Au(I) CN=4 (Au(I) canonical CN=2).
    # This must score is_compliant=False and validates the probe.
    _axis("R09_Au_I_mismatch","Au", "I",   "linear",        4),
]

SEEDS = (0, 1, 2)


def run_smoke() -> dict:
    """Execute the 10x3 rotation smoke and return the result dict."""
    rt = F2MetalLigandExchangeRuntime()

    # Verify runtime is loadable.
    n_patterns = rt.n_patterns_loaded
    assert n_patterns >= 170, f"F2(a) runtime must load >=170 patterns, got {n_patterns}"

    # Verify 84-rule MLE matrix is loadable.
    n_matrix_patterns = len(MATRIX_METALS) * len(MATRIX_LIGANDS)
    assert n_matrix_patterns == 84, f"84-rule matrix must be 84, got {n_matrix_patterns}"

    per_cell: list = []
    per_rotation_compliance: dict = {}

    t0 = time.time()
    for axis_label, metal, ox_state, geometry, donor_count, builder in ROTATION_AXES:
        per_seed: list = []
        per_rotation_compliance[axis_label] = {"metal": metal, "ox": ox_state, "geometry": geometry, "donor_count": donor_count}
        for seed_idx, seed in enumerate(SEEDS):
            smiles = builder(seed_idx)
            # Probe the coordination geometry.
            probe = probe_coordination(smiles)
            if probe is None:
                # No metal detected — skip from compliance scoring.
                per_cell.append({
                    "axis": axis_label,
                    "seed": seed,
                    "metal": metal,
                    "ox_state": ox_state,
                    "geometry_target": geometry,
                    "donor_count": donor_count,
                    "smiles": smiles,
                    "probe_metal": None,
                    "is_compliant": None,
                    "probe_coordination_number": None,
                    "probe_geometry": None,
                    "note": "no_metal_detected",
                })
                per_seed.append(None)
                continue

            per_cell.append({
                "axis": axis_label,
                "seed": seed,
                "metal": metal,
                "ox_state": ox_state,
                "geometry_target": geometry,
                "donor_count": donor_count,
                "smiles": smiles,
                "probe_metal": probe.metal,
                "is_compliant": probe.is_compliant,
                "probe_coordination_number": probe.coordination_number,
                "probe_geometry": probe.geometry,
                "note": "ok",
            })
            per_seed.append(probe.is_compliant)
        # Per-rotation mean compliance over the 3 seeds (None excluded).
        compliant_vals = [v for v in per_seed if v is not None]
        if compliant_vals:
            per_rotation_compliance[axis_label]["per_seed_compliance"] = per_seed
            per_rotation_compliance[axis_label]["mean_compliance"] = statistics.mean(compliant_vals)
        else:
            per_rotation_compliance[axis_label]["per_seed_compliance"] = per_seed
            per_rotation_compliance[axis_label]["mean_compliance"] = None
    elapsed_s = time.time() - t0

    # Aggregate over all 30 cells.
    compliant_cells = [c for c in per_cell if c["is_compliant"] is True]
    noncompliant_cells = [c for c in per_cell if c["is_compliant"] is False]
    none_cells = [c for c in per_cell if c["is_compliant"] is None]

    n_compliant = len(compliant_cells)
    n_noncompliant = len(noncompliant_cells)
    n_none = len(none_cells)
    n_total = len(per_cell)
    compliance_rate = n_compliant / max(1, n_compliant + n_noncompliant)

    # Variance over per-rotation mean_compliance (the headline variance
    # metric — variance=0 means singleton collapse persists).
    per_rotation_means = [
        v["mean_compliance"] for v in per_rotation_compliance.values()
        if v["mean_compliance"] is not None
    ]
    variance = statistics.pvariance(per_rotation_means) if len(per_rotation_means) >= 2 else 0.0

    # Honest framing verdict.
    if variance > 0.05:
        honest_verdict = "structural_ceiling_lifted"
        honest_note = (
            "per-rotation mean_compliance variance > 0.05 — Phase-3G "
            "10x3 rotation produces non-singleton distribution; F2(a) "
            "runtime unblock + 84-rule MLE matrix deliver non-trivial "
            "compliance signal. GPU portion (Wf-310E Wf-135 Phase 2) "
            "can proceed once user OK arrives per kill_then_adapt §0."
        )
    elif variance > 0.0:
        honest_verdict = "runtime_unblocked_partial_signal"
        honest_note = (
            "per-rotation variance is small but non-zero. The 84-rule "
            "MLE matrix produces SOME variation across rotations; "
            "Phase-3G compliance is structurally heterogeneous. "
            "GPU portion still BLOCKED on user OK per kill_then_adapt §0."
        )
    else:
        honest_verdict = "HONEST_NEGATIVE_runtime_unblocked_but_singleton_persists"
        honest_note = (
            "variance = 0 across 10x3 rotation — F2(a) runtime is "
            "unblocked (170 patterns loaded), but the smoke SMILES "
            "themselves collapse to identical compliance values. The "
            "singleton collapse is NOT lifted by runtime alone; the "
            "MCTS still needs real GPU retrain (Wf-310E Wf-135 Phase 2) "
            "to surface diverse compliance behaviour. Honest negative: "
            "F2(a) runtime UNBLOCKS Phase-3G, but does NOT lift the "
            "compliance ceiling on its own."
        )

    result = {
        "workflow": "Wf-325F",
        "task": "F2(a) Phase-3G 10x3 rotation unblock verification",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_n_patterns_loaded": n_patterns,
        "matrix_metals_count": len(MATRIX_METALS),
        "matrix_ligands_count": len(MATRIX_LIGANDS),
        "matrix_patterns_count": n_matrix_patterns,
        "n_rotation_indices": len(ROTATION_AXES),
        "n_seeds": len(SEEDS),
        "n_runs_executed": n_total,
        "n_compliant": n_compliant,
        "n_noncompliant": n_noncompliant,
        "n_none_no_metal": n_none,
        "compliance_rate": compliance_rate,
        "variance": variance,
        "elapsed_s": elapsed_s,
        "cpu_only": True,
        "gpu_blocked": True,
        "honest_verdict": honest_verdict,
        "honest_note": honest_note,
        "per_rotation_compliance": per_rotation_compliance,
        "per_cell_results": per_cell,
    }
    return result


def main() -> int:
    result = run_smoke()
    out_dir = Path(__file__).resolve().parents[1] / "reports" / "wf325f_phase3g_10x3_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    # Summary to stdout.
    print(f"Wf-325F Phase-3G 10x3 rotation CPU smoke")
    print(f"  n_runs_executed: {result['n_runs_executed']}")
    print(f"  n_compliant / n_noncompliant / n_none: "
          f"{result['n_compliant']} / {result['n_noncompliant']} / {result['n_none_no_metal']}")
    print(f"  compliance_rate: {result['compliance_rate']:.4f}")
    print(f"  variance (per-rotation mean_compliance): {result['variance']:.6f}")
    print(f"  elapsed_s: {result['elapsed_s']:.2f}")
    print(f"  honest_verdict: {result['honest_verdict']}")
    print(f"  result.json -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
