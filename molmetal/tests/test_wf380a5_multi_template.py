"""Wf-380-A5 — TODO-204 m15 multi-template + MM-GBSA tests.

================================================================
What this test module validates
================================================================
The Wf-380-A5 closeout ships a **CPU-only scaffold** for two m15
Docking RMSD improvements deferred by the canonical Wf-179 BD-9 +
Wf-246-B + Wf-247-A + Wf-341-T1 single-template bench:

  1. **Multi-template docking** — run Vina against N homologous
     pocket templates and pick the best-of-N pose per pocket.
  2. **MM-GBSA rescoring** — closed-form surrogate of the free-energy
     refinement (Still 1990; Genheden & Ryde 2015 review).

The 2 tests below cover each axis independently:

  * ``test_multi_template_dock`` — multi-template runs without error
    and returns a grid with non-negative RMSD across all
    (template, pose) cells.
  * ``test_mm_gbsa_rescore`` — rescore function returns a valid
    (best_template_idx, best_pose_idx, best_rmsd, best_mmgbsa) tuple.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Tests are CPU-only and use ``numpy`` + the in-module closed-form
  surrogate.  No real meeko / Vina / GBSA binaries are invoked.
* The synthetic surrogate is documented in
  ``wf380a5_multi_template_mm_gbsa._synthetic_rmsd_grid`` and
  ``_synthetic_mmgbsa_kcal``; the lift axis (-0.10 to -0.15 A) is
  PROJECTED per the Wf-380-A5 brief and NOT MEASURED on real
  binaries.
* ``test_mm_gbsa_rescore`` also asserts that the rescore picks a
  *low-MMGBSA* pose and that the best RMSD is non-negative.
"""
from __future__ import annotations

import pytest

from molmetal.scripts.wf380a5_multi_template_mm_gbsa import (
    DEFAULT_N_POSES,
    DEFAULT_N_TEMPLATES,
    DEFAULT_POCKETS,
    RMSD_SOTA_THRESHOLD_ANGSTROM,
    STATUS_SURROGATE,
    aggregate_median_rmsd,
    bench_multi_template_mm_gbsa,
    mm_gbsa_rescore,
    multi_template_dock,
    per_pocket_median_rmsd,
)


# ---------------------------------------------------------------------------
# Test 1 — multi_template_dock runs without error and returns valid grid
# ---------------------------------------------------------------------------
def test_multi_template_dock() -> None:
    """Multi-template dock must return a non-negative RMSD grid + MM-GBSA grid.

    Steps:
      1. Call ``multi_template_dock`` with default n_templates / n_poses.
      2. Assert grid shape matches ``(n_templates, n_poses)``.
      3. Assert all RMSD values are >= 0 (no spurious negatives).
      4. Assert MM-GBSA grid shape matches RMSD grid shape.
      5. Assert MM-GBSA values are finite floats (no NaN / Inf).
    """
    rmsd_grid, mmgbsa_grid = multi_template_dock(
        pocket_id="1h36",
        baseline_rmsd=0.65,  # typical m15 baseline
        n_templates=DEFAULT_N_TEMPLATES,
        n_poses=DEFAULT_N_POSES,
        seed=0,
        backend="surrogate",
    )

    # Shape contract
    assert rmsd_grid.shape == (DEFAULT_N_TEMPLATES, DEFAULT_N_POSES)
    assert mmgbsa_grid.shape == (DEFAULT_N_TEMPLATES, DEFAULT_N_POSES)

    # RMSD must be non-negative (clipped surrogate).
    assert (rmsd_grid >= 0.0).all(), (
        f"unexpected negative RMSD values in synthetic grid: "
        f"min={rmsd_grid.min():.4f} A"
    )

    # MM-GBSA must be finite floats (no NaN / Inf).
    import math

    assert np_all_finite(mmgbsa_grid), (
        f"MM-GBSA grid contains non-finite values: "
        f"min={mmgbsa_grid.min():.4f} max={mmgbsa_grid.max():.4f}"
    )

    # Best RMSD across the grid should be in a sensible range
    # (synthetic surrogate gives approx 0.50-0.65 A on this baseline).
    best = float(rmsd_grid.min())
    assert 0.0 <= best <= RMSD_SOTA_THRESHOLD_ANGSTROM, (
        f"best RMSD {best:.3f} A unexpectedly outside 0 .. "
        f"{RMSD_SOTA_THRESHOLD_ANGSTROM:.1f} A band"
    )


# ---------------------------------------------------------------------------
# Test 2 — mm_gbsa_rescore returns valid (idx, idx, rmsd, energy) tuple
# ---------------------------------------------------------------------------
def test_mm_gbsa_rescore() -> None:
    """MM-GBSA rescore must return valid indices + non-negative RMSD + finite kcal/mol.

    Steps:
      1. Build a multi-template dock grid for one pocket.
      2. Call ``mm_gbsa_rescore``.
      3. Assert returned tuple has 4 elements with the expected types.
      4. Assert ``best_template_idx`` / ``best_pose_idx`` are valid indices.
      5. Assert ``best_rmsd_angstrom`` is non-negative and finite.
      6. Assert ``best_mmgbsa_kcal`` is finite.
      7. End-to-end ``per_pocket_median_rmsd`` also returns a record
         with the same SOTA-pass boolean plus a SURROGATE status.
    """
    rmsd_grid, mmgbsa_grid = multi_template_dock(
        pocket_id="3eml",
        baseline_rmsd=0.70,
        n_templates=DEFAULT_N_TEMPLATES,
        n_poses=DEFAULT_N_POSES,
        seed=42,
        backend="surrogate",
    )

    best_t, best_p, best_rmsd, best_dg = mm_gbsa_rescore(
        rmsd_grid=rmsd_grid,
        mmgbsa_grid=mmgbsa_grid,
    )

    # Tuple arity + types
    assert isinstance(best_t, int)
    assert isinstance(best_p, int)
    assert isinstance(best_rmsd, float)
    assert isinstance(best_dg, float)

    # Index validity
    assert 0 <= best_t < DEFAULT_N_TEMPLATES
    assert 0 <= best_p < DEFAULT_N_POSES

    # Value validity
    assert best_rmsd >= 0.0, f"best RMSD must be non-negative, got {best_rmsd}"
    import math

    assert math.isfinite(best_dg), f"best MM-GBSA must be finite, got {best_dg}"

    # The picked cell must match the argmin over the grid.
    import numpy as np

    flat_idx_expected = int(np.argmin(mmgbsa_grid))
    flat_idx_actual = best_t * DEFAULT_N_POSES + best_p
    assert flat_idx_actual == flat_idx_expected, (
        f"rescore picked ({best_t},{best_p}) -> flat {flat_idx_actual}; "
        f"argmin gave flat {flat_idx_expected}"
    )

    # End-to-end: ``per_pocket_median_rmsd`` matches rescore semantics.
    record = per_pocket_median_rmsd(
        pocket_id="3eml",
        baseline_rmsd=0.70,
        n_templates=DEFAULT_N_TEMPLATES,
        n_poses=DEFAULT_N_POSES,
        seed=42,
        backend="surrogate",
    )
    assert record.pocket == "3eml"
    assert record.best_template_idx == best_t
    assert record.best_mmgbsa_kcal == best_dg
    assert record.best_rmsd_angstrom == best_rmsd
    assert record.status == STATUS_SURROGATE
    assert isinstance(record.passes_2_0a_sota, bool)


# ---------------------------------------------------------------------------
# Helper — assert all values in a numpy array are finite
# ---------------------------------------------------------------------------
def np_all_finite(arr) -> bool:  # type: ignore[no-untyped-def]
    import numpy as np

    a = np.asarray(arr)
    return bool(np.all(np.isfinite(a)))