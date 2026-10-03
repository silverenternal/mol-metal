"""Wf-404-A2 — TODO-204 m15 multitemplate + top-K + MM-GBSA SCAFFOLD tests.

================================================================
What this test module validates
================================================================
The Wf-404-A2 closeout ships a **CPU-only SCAFFOLD** for the m15
multitemplate pose generation + top-K pose selection per template +
MM-GBSA rescoring axis, extending the Wf-380-A5 single-template
MM-GBSA scaffold with three concrete additions:

  1. **Multi-template pose generation** — 3 homologous templates per
     pocket from a synthetic ``TEMPLATE_SOURCES`` table.
  2. **Top-K pose selection per template** — narrow the
     (n_templates x n_poses) grid to a (n_templates x K) "best-K"
     subgrid ranked by MM-GBSA ΔG.
  3. **MM-GBSA rescoring on top-K + best-pose selection** — pick the
     (template, in-template-pose) cell with the lowest MM-GBSA
     kcal/mol across the top-K subgrid.

The 3 tests below cover each axis independently:

  * ``test_multi_template_loader`` — the multitemplate loader returns
    ``n_templates`` template records per pocket with valid template
    ids + structural families + synthetic Tanimoto similarity.
  * ``test_mm_gbsa_rescore`` — the MM-GBSA rescore picks the
    (template, top-K pose) cell with the lowest MM-GBSA kcal/mol,
    matching ``np.argmin`` over the grid.
  * ``test_best_pose_selection`` — the best-pose record reports a
    ``best_template_id`` + ``best_template_family`` + non-negative
    ``best_rmsd_angstrom`` + finite ``best_mmgbsa_kcal`` + a
    SURROGATE status (per party-twenty-spirit §3.5).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Tests are CPU-only and use ``numpy`` + the in-module closed-form
  surrogate (``_synthetic_rmsd_grid`` + ``_synthetic_mmgbsa_kcal``
  re-imported from Wf-380-A5).  No real meeko / Vina / GBSA binaries
  are invoked.
* The 3 templates per pocket are SYNTHETIC structural sources with
  placeholder Tanimoto similarity values; real RCSB PDB template
  sourcing is deferred to TODO-207 / Wf-299 launch.
* All numeric values are SURROGATE — the m15 MEASURED baseline
  ``metrics/by_metric/docked_pose_rmsd.json`` (median 0.6578 Å
  across 5 organic-cofactor pockets) is the source-of-truth baseline;
  the SCAFFOLD only simulates the expected lift via a closed-form
  linear surrogate anchored on the MEASURED baseline.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from molmetal.scripts.wf404a2_rmsd_multitemplate_scaffold import (
    DEFAULT_BACKEND,
    DEFAULT_N_TEMPLATES,
    DEFAULT_TOP_K,
    PerPocketMultitemplate,
    PerTemplateRecord,
    RMSD_PRACTICAL_THRESHOLD_ANGSTROM,
    RMSD_SOTA_THRESHOLD_ANGSTROM,
    STATUS_SURROGATE,
    TEMPLATE_SOURCES,
    MultitemplateSummary,
    _generate_pose_grid,
    _select_top_k_per_template,
    aggregate_multitemplate_rmsd,
    bench_multitemplate_mm_gbsa,
    load_baseline_per_pocket_rmsd,
    mm_gbsa_rescore_top_k,
    per_pocket_multitemplate_bench,
)


# ---------------------------------------------------------------------------
# Test 1 — multi_template_loader returns valid template records
# ---------------------------------------------------------------------------
def test_multi_template_loader() -> None:
    """The multitemplate loader must return ``n_templates`` template
    records per pocket, each with valid template id + structural
    family + synthetic Tanimoto in [0, 1].

    Steps:
      1. Run ``per_pocket_multitemplate_bench`` for each of the 5
         canonical pockets (1h36 + 3eml + 3ny8 + 4aaw + 4rlu).
      2. Assert the per-pocket record has ``n_templates`` template
         records.
      3. Assert each ``PerTemplateRecord`` has a valid template id +
         structural family + synthetic Tanimoto in [0, 1].
      4. Assert the top-K RMSD array length matches ``top_k``.
      5. Assert the top-K MM-GBSA array length matches ``top_k``.
      6. Assert all RMSD values are >= 0 (clipped surrogate).
      7. Assert all MM-GBSA values are finite floats (no NaN / Inf).
    """
    pockets = list(TEMPLATE_SOURCES.keys())
    assert len(pockets) == 5, (
        f"TEMPLATE_SOURCES should expose 5 canonical pockets; got {len(pockets)}"
    )

    for pocket_id in pockets:
        record: PerPocketMultitemplate = per_pocket_multitemplate_bench(
            pocket_id=pocket_id,
            baseline_rmsd=0.65,
            n_templates=DEFAULT_N_TEMPLATES,
            top_k=DEFAULT_TOP_K,
            seed=0,
            backend=DEFAULT_BACKEND,
        )
        # Per-pocket record invariants
        assert record.pocket == pocket_id
        assert record.n_templates == DEFAULT_N_TEMPLATES
        assert record.top_k == DEFAULT_TOP_K
        assert record.status == STATUS_SURROGATE

        # Per-template records
        assert len(record.per_template) == DEFAULT_N_TEMPLATES, (
            f"{pocket_id}: expected {DEFAULT_N_TEMPLATES} template records, "
            f"got {len(record.per_template)}"
        )

        for t_idx, t_rec in enumerate(record.per_template):
            # Template id is non-empty string.
            assert isinstance(t_rec.template_id, str) and len(t_rec.template_id) > 0, (
                f"{pocket_id}: template id at index {t_idx} is invalid: {t_rec.template_id!r}"
            )
            # Structural family is non-empty string.
            assert isinstance(t_rec.structural_family, str) and len(t_rec.structural_family) > 0, (
                f"{pocket_id}: structural family at index {t_idx} is invalid: "
                f"{t_rec.structural_family!r}"
            )
            # Synthetic Tanimoto in [0, 1].
            assert 0.0 <= t_rec.synthetic_tanimoto <= 1.0, (
                f"{pocket_id}: template[{t_idx}] synthetic Tanimoto "
                f"{t_rec.synthetic_tanimoto:.3f} outside [0, 1]"
            )
            # Top-K RMSD length matches top_k.
            assert len(t_rec.top_k_rmsds) == DEFAULT_TOP_K, (
                f"{pocket_id}: template[{t_idx}] top-K RMSD length "
                f"{len(t_rec.top_k_rmsds)} != {DEFAULT_TOP_K}"
            )
            # Top-K MM-GBSA length matches top_k.
            assert len(t_rec.top_k_mmgbsa_kcal) == DEFAULT_TOP_K, (
                f"{pocket_id}: template[{t_idx}] top-K MM-GBSA length "
                f"{len(t_rec.top_k_mmgbsa_kcal)} != {DEFAULT_TOP_K}"
            )
            # All RMSDs non-negative.
            assert (np.asarray(t_rec.top_k_rmsds) >= 0.0).all(), (
                f"{pocket_id}: template[{t_idx}] has negative RMSD values: "
                f"{list(t_rec.top_k_rmsds)}"
            )
            # All MM-GBSA values finite.
            assert np_all_finite(t_rec.top_k_mmgbsa_kcal), (
                f"{pocket_id}: template[{t_idx}] has non-finite MM-GBSA values"
            )

        # Best RMSD + MM-GBSA at the per-pocket level are non-negative + finite.
        assert record.best_rmsd_angstrom >= 0.0
        assert math.isfinite(record.best_mmgbsa_kcal)


# ---------------------------------------------------------------------------
# Test 2 — MM-GBSA rescore picks the argmin over the (n_templates, top_k) grid
# ---------------------------------------------------------------------------
def test_mm_gbsa_rescore() -> None:
    """MM-GBSA rescore must pick the (template, top-K pose) cell with
    the lowest MM-GBSA kcal/mol — matching ``np.argmin`` over the grid.

    Steps:
      1. Build a (n_templates, n_poses) pose grid via ``_generate_pose_grid``.
      2. Compute the top-K subgrid via ``_select_top_k_per_template``.
      3. Call ``mm_gbsa_rescore_top_k`` and unpack the 4-tuple.
      4. Assert the picked cell matches ``np.argmin(top_k_mmgbsa)``.
      5. Assert the picked RMSD equals ``top_k_rmsds[best_template_idx, best_pose_idx]``.
      6. Assert the picked MM-GBSA equals ``top_k_mmgbsa[best_template_idx, best_pose_idx]``.
      7. Assert the picked MM-GBSA is <= all other cells (strictly argmin).
    """
    # Build a synthetic pose grid for the canonical 1h36 pocket.
    n_templates = 4
    n_poses = 9
    rmsd_grid, mmgbsa_grid = _generate_pose_grid(
        pocket_id="1h36",
        baseline_rmsd=0.65,
        n_templates=n_templates,
        n_poses=n_poses,
        seed=42,
        backend=DEFAULT_BACKEND,
    )
    assert rmsd_grid.shape == (n_templates, n_poses)
    assert mmgbsa_grid.shape == (n_templates, n_poses)

    # Top-K selection (K=3).
    top_k = 3
    top_k_rmsds, top_k_mmgbsa = _select_top_k_per_template(
        rmsd_grid=rmsd_grid,
        mmgbsa_grid=mmgbsa_grid,
        top_k=top_k,
    )
    assert top_k_rmsds.shape == (n_templates, top_k)
    assert top_k_mmgbsa.shape == (n_templates, top_k)

    # Rescore.
    best_t, best_p, best_rmsd, best_dg = mm_gbsa_rescore_top_k(
        top_k_rmsds=top_k_rmsds,
        top_k_mmgbsa=top_k_mmgbsa,
    )

    # Tuple arity + types
    assert isinstance(best_t, int)
    assert isinstance(best_p, int)
    assert isinstance(best_rmsd, float)
    assert isinstance(best_dg, float)

    # Index validity
    assert 0 <= best_t < n_templates
    assert 0 <= best_p < top_k

    # The picked cell must match ``np.argmin`` over the grid.
    flat_idx_expected = int(np.argmin(top_k_mmgbsa))
    flat_idx_actual = best_t * top_k + best_p
    assert flat_idx_actual == flat_idx_expected, (
        f"rescore picked ({best_t},{best_p}) -> flat {flat_idx_actual}; "
        f"argmin gave flat {flat_idx_expected}"
    )

    # Value consistency.
    assert math.isclose(best_rmsd, float(top_k_rmsds[best_t, best_p]), rel_tol=1e-9)
    assert math.isclose(best_dg, float(top_k_mmgbsa[best_t, best_p]), rel_tol=1e-9)

    # MM-GBSA value at the picked cell is the global minimum (no ties expected with surrogate seed).
    assert best_dg == float(top_k_mmgbsa.min()), (
        f"picked MM-GBSA {best_dg:.4f} != grid minimum {top_k_mmgbsa.min():.4f}"
    )

    # The per-template top-K selection preserves the Vina/MM-GBSA ranking
    # invariant: the first (lowest-MMGBSA) cell in each template's top-K
    # subgrid is the grid minimum for that template.
    for t in range(n_templates):
        per_template_min = float(mmgbsa_grid[t, :].min())
        assert top_k_mmgbsa[t, 0] == per_template_min, (
            f"template[{t}] top-K MM-GBSA[0]={top_k_mmgbsa[t, 0]:.4f} "
            f"!= per-template grid min {per_template_min:.4f}"
        )

    # Exhaustive top_k=-1 returns all poses.
    full_rmsds, full_mmgbsa = _select_top_k_per_template(
        rmsd_grid=rmsd_grid,
        mmgbsa_grid=mmgbsa_grid,
        top_k=-1,
    )
    assert full_rmsds.shape == (n_templates, n_poses)
    assert full_mmgbsa.shape == (n_templates, n_poses)
    # The exhaustive top-K-0 cell equals the per-template argmin.
    for t in range(n_templates):
        assert full_mmgbsa[t, 0] == float(mm_gbsa_grid[t, :].min())


# ---------------------------------------------------------------------------
# Test 3 — best_pose_selection reports valid best record
# ---------------------------------------------------------------------------
def test_best_pose_selection() -> None:
    """Best-pose selection must populate a valid best-pose record with
    non-negative RMSD + finite MM-GBSA + SURROGATE status.

    Steps:
      1. Run the full top-level bench via ``bench_multitemplate_mm_gbsa``.
      2. Assert the summary has ``n_pockets`` records matching the
         canonical 5-pocket cohort.
      3. For each pocket, assert the best-template record reports:
         * ``best_template_id`` is a non-empty string (member of
           ``TEMPLATE_SOURCES[pocket_id]``).
         * ``best_template_family`` is a non-empty string.
         * ``best_rmsd_angstrom`` is in [0, RMSD_SOTA_THRESHOLD].
         * ``best_mmgbsa_kcal`` is finite.
         * ``passes_2_0a_sota`` is True (synthetic surrogate should
           guarantee this for the canonical m15 cohort).
      4. Assert the cohort aggregate ``aggregate_median_best_rmsd_angstrom``
         is in a sensible range (synthetic surrogate gives approx
         0.50-0.65 A on a 0.65 baseline).
      5. Assert the baseline loader populates non-empty per-pocket baseline.
    """
    summary: MultitemplateSummary = bench_multitemplate_mm_gbsa(
        pockets=tuple(TEMPLATE_SOURCES.keys()),
        n_templates=DEFAULT_N_TEMPLATES,
        top_k=DEFAULT_TOP_K,
        seed=0,
        backend=DEFAULT_BACKEND,
    )
    # Summary invariants
    assert summary.n_pockets == 5
    assert summary.n_templates == DEFAULT_N_TEMPLATES
    assert summary.top_k == DEFAULT_TOP_K
    assert summary.backend == DEFAULT_BACKEND
    assert summary.m15_status == STATUS_SURROGATE
    assert len(summary.per_pocket) == 5

    # Baseline loader invariant.
    baseline = load_baseline_per_pocket_rmsd()
    assert len(baseline) == 5, (
        f"baseline loader expected to populate 5 canonical pockets, "
        f"got {len(baseline)}: {list(baseline.keys())}"
    )
    for pocket_id in TEMPLATE_SOURCES.keys():
        assert pocket_id in baseline, f"missing baseline for {pocket_id}"
        assert baseline[pocket_id] > 0.0
    # Median baseline is the Wf-179 BD-9 + Wf-246-B + Wf-341-T1 MEASURED baseline.
    assert summary.baseline_median_best_rmsd_angstrom is not None
    assert 0.6 < summary.baseline_median_best_rmsd_angstrom < 0.7, (
        f"baseline median {summary.baseline_median_best_rmsd_angstrom:.4f} "
        f"outside expected Wf-179 BD-9 band 0.6-0.7"
    )

    # Per-pocket best-pose validation.
    valid_template_ids = {tid: fid for tid, fid, _ in TEMPLATE_SOURCES["1h36"]}
    for pocket_id in TEMPLATE_SOURCES.keys():
        pocket_valid_ids = {tid for tid, _, _ in TEMPLATE_SOURCES[pocket_id]}
        record = next(r for r in summary.per_pocket if r.pocket == pocket_id)

        # Best-template id is one of the per-pocket template ids.
        assert record.best_template_id in pocket_valid_ids, (
            f"{pocket_id}: best_template_id={record.best_template_id!r} "
            f"not in {sorted(pocket_valid_ids)}"
        )
        # Best-template family is non-empty string.
        assert isinstance(record.best_template_family, str)
        assert len(record.best_template_family) > 0
        # Best RMSD is non-negative and within the SOTA band.
        assert record.best_rmsd_angstrom >= 0.0
        assert record.best_rmsd_angstrom <= RMSD_SOTA_THRESHOLD_ANGSTROM, (
            f"{pocket_id}: best RMSD {record.best_rmsd_angstrom:.4f} "
            f"> SOTA gate {RMSD_SOTA_THRESHOLD_ANGSTROM}"
        )
        # MM-GBSA is finite.
        assert math.isfinite(record.best_mmgbsa_kcal)
        # 2.0 A SOTA gate passed (synthetic surrogate guarantee).
        assert record.passes_2_0a_sota is True
        # SURROGATE status preserved.
        assert record.status == STATUS_SURROGATE

    # Aggregate cohort invariants.
    assert summary.aggregate_success_rate_2_0a == 1.0, (
        f"all 5 canonical pockets should PASS 2.0 A SOTA gate; "
        f"got {summary.aggregate_success_rate_2_0a}"
    )
    assert summary.aggregate_success_rate_2_5a == 1.0
    # Synthetic surrogate median best RMSD should be in a sensible band.
    assert 0.0 <= summary.aggregate_median_best_rmsd_angstrom <= RMSD_SOTA_THRESHOLD_ANGSTROM, (
        f"aggregate median {summary.aggregate_median_best_rmsd_angstrom:.4f} "
        f"outside SOTA band"
    )
    # Synthetic-surrogate lift should be negative (improvement).
    assert summary.lift_angstrom is not None
    assert summary.lift_angstrom < 0.0, (
        f"expected surrogate lift to be negative (improvement), "
        f"got {summary.lift_angstrom:.4f}"
    )

    # Aggregate function sanity: aggregate_multitemplate_rmsd reproduces summary.
    rep_summary = aggregate_multitemplate_rmsd(
        per_pocket=summary.per_pocket,
        baseline_median=summary.baseline_median_best_rmsd_angstrom,
        backend=DEFAULT_BACKEND,
    )
    assert rep_summary.aggregate_median_best_rmsd_angstrom == pytest.approx(
        summary.aggregate_median_best_rmsd_angstrom, rel=1e-9
    )
    assert rep_summary.lift_angstrom == pytest.approx(summary.lift_angstrom, rel=1e-9)


# ---------------------------------------------------------------------------
# Helper — assert all values in a numpy array are finite
# ---------------------------------------------------------------------------
def np_all_finite(arr) -> bool:  # type: ignore[no-untyped-def]
    a = np.asarray(arr)
    return bool(np.all(np.isfinite(a)))
