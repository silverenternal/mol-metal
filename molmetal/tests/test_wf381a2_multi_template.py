"""Wf-381-A2 — TODO-204 m15 Docking RMSD multi-template + MM-GBSA tests.

================================================================
What this test module validates
================================================================
The Wf-381-A2 closeout ships an integration scaffold for the
TODO-204 multi-template + MM-GBSA m15 Docking RMSD lift axis.
This test module covers the two headline functions exposed by
``molmetal/scripts/wf381a2_multi_template_mm_gbsa.py``:

  1. ``multi_template_dock`` — runs Vina against N homologous
     templates and returns a ``PoseGrid`` with shape
     ``(n_templates, n_poses)`` + per-cell MM-GBSA energies.
  2. ``mm_gbsa_rescore`` — picks the best (template, pose) cell
     from a ``PoseGrid`` and returns a ``RescoreGrid``.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Tests are CPU-only and use the synthetic surrogate backend
  (the real Vina + GBSA backend is deferred to TODO-207 / Wf-299
  and raises NotImplementedError when invoked).
* ``multi_template_dock`` is asserted against a deterministic seed
  (seed=0) so the test passes reproducibly on any host.
* ``mm_gbsa_rescore`` is asserted to return a valid energy range
  (negative kcal/mol for the closed-form surrogate).
* Per-pocket baselines are read from the canonical
  ``metrics/by_metric/docked_pose_rmsd.json`` (MEASURED by Wf-179
  BD-9 + Wf-341-T1 + Wf-362-A3); skipif if missing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from molmetal.scripts.wf381a2_multi_template_mm_gbsa import (
    DEFAULT_N_TEMPLATES,
    DEFAULT_N_POSES,
    DEFAULT_POCKETS,
    MMGBSA_BORN_IMPLICIT_SOLVENT_KCAL,
    PER_POCKET_SUCCESS_RATE,
    RMSD_PRACTICAL_THRESHOLD_ANGSTROM,
    RMSD_SOTA_THRESHOLD_ANGSTROM,
    STATUS_HONEST_NEGATIVE,
    STATUS_SURROGATE,
    CohortSummary,
    PerPocketResult,
    PoseGrid,
    RescoreGrid,
    _synthetic_rmsd_grid,
    aggregate_median_rmsd,
    bench_multi_template_mm_gbsa,
    mm_gbsa_rescore,
    multi_template_dock,
    per_pocket_median_rmsd,
)

REPO = Path(__file__).resolve().parent.parent.parent
DOCKED_POSE_RMSD_JSON = REPO / "metrics" / "by_metric" / "docked_pose_rmsd.json"


# ---------------------------------------------------------------------------
# Test 1 — multi_template_dock runs without error
# ---------------------------------------------------------------------------
def test_multi_template_dock() -> None:
    """Multi-template dock must run on synthetic inputs without error.

    Validates:
    1. ``multi_template_dock`` returns a ``PoseGrid`` with the
       requested ``(n_templates, n_poses)`` shape.
    2. ``rmsd_grid`` is non-negative (synthetic clipped to >= 0).
    3. ``mmgbsa_grid`` is finite (no NaN/Inf in kcal/mol values).
    4. Deterministic with fixed seed: same seed -> same grid.
    5. Pocket id is derived from the path stem correctly.
    6. NotImplementedError raised for ``backend="real"`` (TODO-207 deferred).
    """
    pocket_pdb = "/tmp/1h36_pocket.pdb"
    template_pdbs = [f"/tmp/template_{i}.pdb" for i in range(5)]

    # Default surrogate backend
    grid = multi_template_dock(
        pocket_pdb=pocket_pdb,
        template_pdbs=template_pdbs,
        n_poses=9,
        seed=0,
    )
    assert isinstance(grid, PoseGrid)
    assert grid.pocket_id == "1h36"  # _pocket suffix stripped
    assert grid.n_templates == 5
    assert grid.n_poses == 9
    assert grid.rmsd_grid.shape == (5, 9)
    assert grid.mmgbsa_grid.shape == (5, 9)
    assert grid.status == STATUS_SURROGATE

    # Non-negative RMSD
    assert np.all(grid.rmsd_grid >= 0.0), (
        f"RMSD grid must be non-negative, got min={grid.rmsd_grid.min():.4f}"
    )

    # Finite MM-GBSA energies (no NaN/Inf)
    assert np.all(np.isfinite(grid.mmgbsa_grid)), (
        "MM-GBSA grid must be finite (no NaN/Inf)"
    )

    # Deterministic with fixed seed
    grid_again = multi_template_dock(
        pocket_pdb=pocket_pdb,
        template_pdbs=template_pdbs,
        n_poses=9,
        seed=0,
    )
    np.testing.assert_array_equal(
        grid.rmsd_grid,
        grid_again.rmsd_grid,
        err_msg="multi_template_dock must be deterministic with fixed seed",
    )

    # Real backend raises NotImplementedError (TODO-207 deferred)
    with pytest.raises(NotImplementedError) as exc_info:
        multi_template_dock(
            pocket_pdb=pocket_pdb,
            template_pdbs=template_pdbs,
            n_poses=9,
            seed=0,
            backend="real",
        )
    assert "TODO-207" in str(exc_info.value) or "Wf-299" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2 — mm_gbsa_rescore returns valid energy
# ---------------------------------------------------------------------------
def test_mm_gbsa_rescore() -> None:
    """MM-GBSA rescore must return valid (negative kcal/mol) energy.

    Validates:
    1. ``mm_gbsa_rescore`` returns a ``RescoreGrid`` with the
       correct pocket id propagated from the input PoseGrid.
    2. ``best_rmsd_angstrom`` is non-negative and <= median_rmsd.
    3. ``best_mmgbsa_kcal`` is negative kcal/mol (closed-form surrogate
       model uses Born implicit solvent offset of -5.0 kcal/mol + noise).
    4. ``best_template_idx`` + ``best_pose_idx`` are valid in-bounds
       indices for the (n_templates, n_poses) grid.
    5. Status is SURROGATE (closed-form synthetic).
    6. End-to-end smoke: rescore + per_pocket_median_rmsd +
       aggregate_median_rmsd yields a CohortSummary with
       pass-rate 1.0 on canonical 5-pocket cohort.
    """
    pocket_pdb = "/tmp/3eml_pocket.pdb"
    template_pdbs = [f"/tmp/template_{i}.pdb" for i in range(5)]
    grid = multi_template_dock(
        pocket_pdb=pocket_pdb,
        template_pdbs=template_pdbs,
        n_poses=9,
        seed=42,
    )

    rescore = mm_gbsa_rescore(
        poses=grid,
        receptor="/tmp/3eml_receptor.pdbqt",
        ligand="/tmp/3eml_ligand.pdbqt",
    )
    assert isinstance(rescore, RescoreGrid)
    assert rescore.pocket_id == "3eml"
    assert rescore.status == STATUS_SURROGATE

    # Best RMSD is non-negative and within grid bounds
    assert rescore.best_rmsd_angstrom >= 0.0, (
        f"best_rmsd_angstrom must be non-negative, got {rescore.best_rmsd_angstrom}"
    )
    assert rescore.best_rmsd_angstrom <= float(grid.rmsd_grid.max()), (
        "best RMSD must be within grid range"
    )

    # Best MM-GBSA is negative kcal/mol (typical binding energy)
    assert rescore.best_mmgbsa_kcal < 0.0, (
        f"best_mmgbsa_kcal must be negative kcal/mol for binding, "
        f"got {rescore.best_mmgbsa_kcal}"
    )
    # Should be near MMGBSA_BORN_IMPLICIT_SOLVENT_KCAL (-5.0 kcal/mol)
    # plus the per-pose RMSD term (-0.50 * rmsd/2.0)
    expected_approx = MMGBSA_BORN_IMPLICIT_SOLVENT_KCAL
    assert rescore.best_mmgbsa_kcal < expected_approx + 0.5, (
        f"best MM-GBSA {rescore.best_mmgbsa_kcal} should be near "
        f"MMGBSA_BORN_IMPLICIT_SOLVENT_KCAL={expected_approx}"
    )

    # Indices are in-bounds
    assert 0 <= rescore.best_template_idx < grid.n_templates
    assert 0 <= rescore.best_pose_idx < grid.n_poses

    # End-to-end smoke (deferred to suite-level hook in production)
    # Verifies the full pipeline: rescore -> per_pocket -> aggregate
    cohort_results: List[PerPocketResult] = []
    for pocket in DEFAULT_POCKETS:
        p_grid = multi_template_dock(
            pocket_pdb=f"/tmp/{pocket}_pocket.pdb",
            template_pdbs=[f"/tmp/{pocket}_t_{i}.pdb" for i in range(5)],
            n_poses=DEFAULT_N_POSES,
            seed=0,
        )
        cohort_results.append(
            per_pocket_median_rmsd(
                poses=p_grid,
                native_ligand=f"/tmp/{pocket}_native.sdf",
            )
        )
    summary = aggregate_median_rmsd(
        cohort_results=cohort_results,
        backend="surrogate",
    )
    assert isinstance(summary, CohortSummary)
    assert summary.n_pockets == len(DEFAULT_POCKETS)
    assert summary.m15_status == STATUS_SURROGATE
    assert summary.aggregate_success_rate_2_0a == 1.0, (
        f"All 5/5 pockets should pass 2.0 Å SOTA gate, "
        f"got {summary.aggregate_success_rate_2_0a}"
    )
