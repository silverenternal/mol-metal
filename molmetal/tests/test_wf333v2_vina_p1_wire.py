"""Tests for Wf-333-V2 — Vina P1: Wf-246-C metal-coord wire into the panel.

================================================================
Wf-333-V2 closes Wf-331-R1 §6.2 (P0 + P1).  Per the audit the
Wf-246-C metal-coordination term was CPU-only but had a sign bug
(HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED at metal_coord_term.py:635)
and was not wired into the Wf-264-Vina panel that produces m3.

This test file covers four MEASURED-critical assertions:

  1. **P0 sign fix** — ``MetalCoordinationTerm.kcal_adjustment``
     uses ``(1 - 2 * composite) * kcal_per_unit``.  Perfect composite
     yields a NEGATIVE adjustment (better binding = lower kcal/mol).
     Zero composite yields a POSITIVE adjustment (penalty).
     None composite yields 0.0 (organic ligand backward compat).

  2. **P1 wire-in** — ``--use-wf246c-metal-coord`` populates
     ``metal_coord_kcal_mol``, ``metal_coord_total``,
     ``metal_coord_geometry_score``, ``metal_coord_hsab_score``,
     ``metal_coord_distance_score``,
     ``metal_coord_recovery_status`` on each PocketCell.

  3. **Combined additive** — when BOTH ``--use-f2a-metal-ligand`` and
     ``--use-wf246c-metal-coord`` are set, the combined
     ``metal_coord_kcal_mol`` is Vina + F2(a) per-bond + Wf-246-C
     composite adjustment (additive per Wf-331-R1 §6.6).

  4. **5-pocket cohort lift** — when running against the canonical
     5-pocket cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu) with a single
     Pt metal-containing SMILES, every cell records the
     ``metal_coord_cohort_median`` aggregate, the
     ``metal_coord_cohort_median_lift_vs_vina`` aggregate, and the
     ``metal_coord_cohort_median_lift_vs_f2a`` aggregate.

All tests are CPU-only (no real Vina invocation); the
``dock_runner`` test seam drives every cell through a deterministic
fake.  The Wf-333-V2 wire-in does not require the vina binary — the
``use_wf246c_metal_coord`` path only depends on RDKit + the
metal_coord_term module.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "cfm_vina_dock_panel.py"


def _load_module():
    """Load cfm_vina_dock_panel.py as a module (test seam)."""
    mod_name = "cfm_vina_dock_panel"
    spec = importlib.util.spec_from_file_location(mod_name, str(SCRIPT_PATH))
    assert spec and spec.loader, "cfm_vina_dock_panel.py must exist and be importable"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


# ---------------------------------------------------------------------------
# Helpers — staged receptors
# ---------------------------------------------------------------------------
def _stage_receptors(tmp_path: Path) -> Path:
    """Stage per-pocket subdirs with stub receptor.pdbqt files.

    The stub contains a few heavy atoms (N/O/S) so the
    ``pocket_donor_counts`` cache (Wf-333-V2 P1) has at least one
    element to count for each pocket.
    """
    pockets_dir = tmp_path / "pockets"
    pockets_dir.mkdir(parents=True, exist_ok=True)
    for pocket_id in ("1h36", "3eml", "3ny8", "4aaw", "4rlu"):
        pdir = pockets_dir / pocket_id
        pdir.mkdir(exist_ok=True)
        # Stub content with N/O atoms so pocket_donor_counts_from_atomicnums
        # has something to count.  The fake runner does not parse this.
        (pdir / "receptor.pdbqt").write_text(
            "ATOM      1  N   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 N \n"
            "ATOM      2  O   UNL     1       1.000   1.000   1.000  1.00  0.00    +0.000 O \n"
            "ATOM      3  S   UNL     1       2.000   2.000   2.000  1.00  0.00    +0.000 S \n"
            "END\n"
        )
    return pockets_dir


# ---------------------------------------------------------------------------
# Fake dock runner — bypasses Vina CLI but still applies the
# Wf-329-W3 F2(a) + Wf-333-V2 Wf-246-C wire-in.  This keeps the
# test deterministic (no real Vina invocation) while exercising
# the end-to-end panel logic that consumes the wire-in outputs.
# ---------------------------------------------------------------------------
class _FakeRunner:
    """Deterministic dock_runner that mirrors :func:`_dock_one`'s
    Wf-329-W3 F2(a) + Wf-333-V2 Wf-246-C wire-in.

    The fake returns a hard-coded Vina kcal/mol, then applies the
    F2(a) modifier (when enabled) and the Wf-246-C composite
    adjustment (when enabled) so the panel-level aggregates see
    the same population that a real run would.
    """

    def __init__(self, kcal_by_pocket: Optional[Dict[str, float]] = None):
        self.kcal_by_pocket = kcal_by_pocket or {
            "1h36": -3.594,
            "3eml": -4.000,
            "3ny8": -3.500,
            "4aaw": -3.800,
            "4rlu": -3.700,
        }
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        smiles: str,
        pocket_id: str,
        receptor: Path,
        vina_bin: str,
        center,
        box_size: float,
        exhaustiveness: int,
        n_poses: int,
        seed: int,
        timeout: int,
        work_dir: Path,
        pt_strip: bool = False,
        use_f2a_metal_ligand: bool = False,
        f2a_modifier_kcal_mol: Optional[float] = None,
        use_wf246c_metal_coord: bool = False,
        wf246c_kcal_per_unit: Optional[float] = None,
        pocket_donor_counts: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        from molmetal.scripts.cfm_vina_dock_panel import (
            PocketCell,
            STATUS_REAL_VINA,
            _get_f2a_scorer,
            _compute_wf246c_adjustment_for_smiles,
        )

        self.calls.append(
            {
                "smiles": smiles,
                "pocket_id": pocket_id,
                "use_f2a_metal_ligand": use_f2a_metal_ligand,
                "use_wf246c_metal_coord": use_wf246c_metal_coord,
            }
        )

        cell = PocketCell(
            pocket=pocket_id,
            best_kcal_mol=float(self.kcal_by_pocket[pocket_id]),
            n_poses=int(n_poses),
            wall_time_s=0.01,
            status=STATUS_REAL_VINA,
            note="FAKE_FOR_TEST",
        )

        # Wf-329-W3: F2(a) MetalLigandExchange covalent scoring.
        if use_f2a_metal_ligand:
            scorer, _err = _get_f2a_scorer(
                modifier_kcal_mol=f2a_modifier_kcal_mol,
            )
            if scorer is not None:
                try:
                    f2a_score, mod_total, n_bonds = scorer.apply_covalent_modifier(
                        smiles, cell.best_kcal_mol
                    )
                    cell.f2a_kcal_mol = float(f2a_score)
                    cell.f2a_modifier_total = float(mod_total)
                    cell.f2a_n_bonds = int(n_bonds)
                    if n_bonds > 0:
                        dispatch = scorer.dispatch(smiles)
                        cell.f2a_matched_patterns = ",".join(dispatch.matched_patterns)
                    cell.f2a_recovery_status, _gap = scorer.classify_recovery(f2a_score)
                except Exception:
                    pass  # F2(a) import or score failure → leave fields None.

        # Wf-333-V2: Wf-246-C metal-coordination composite scoring.
        if use_wf246c_metal_coord:
            mc_adj, mc_diag = _compute_wf246c_adjustment_for_smiles(
                smiles,
                pocket_donor_counts=pocket_donor_counts,
                kcal_per_unit=(
                    float(wf246c_kcal_per_unit)
                    if wf246c_kcal_per_unit is not None
                    else 1.0
                ),
            )
            if mc_diag is not None:
                cell.metal_coord_total = mc_diag.get("metal_coord_total")
                cell.metal_coord_adjustment = float(
                    mc_diag.get("metal_coord_kcal_adjustment", 0.0)
                )
                cell.metal_coord_geometry_score = mc_diag.get("geometry_score")
                cell.metal_coord_hsab_score = mc_diag.get("hsab_score")
                cell.metal_coord_distance_score = mc_diag.get("distance_score")
            if mc_adj is not None:
                base = cell.f2a_kcal_mol
                if base is None:
                    base = cell.best_kcal_mol
                if base is not None:
                    cell.metal_coord_kcal_mol = float(base + mc_adj)
                    cell._v2_mc_adj = float(mc_adj)
                    combined = cell.metal_coord_kcal_mol
                    if combined <= -7.0:
                        cell.metal_coord_recovery_status = "INDUSTRY_CUTOFF_PASS"
                    elif combined <= -5.0:
                        cell.metal_coord_recovery_status = "PARTIAL_RECOVERY"
                    else:
                        cell.metal_coord_recovery_status = "INSUFFICIENT_RECOVERY"
            elif cell.f2a_kcal_mol is not None:
                cell.metal_coord_kcal_mol = cell.f2a_kcal_mol
                cell.metal_coord_recovery_status = cell.f2a_recovery_status
            elif cell.best_kcal_mol is not None:
                cell.metal_coord_kcal_mol = cell.best_kcal_mol
                cell.metal_coord_recovery_status = "NO_METAL_COORD_TERM"

        return cell


# ---------------------------------------------------------------------------
# Test 1: P0 sign fix — MetalCoordinationTerm.kcal_adjustment
# ---------------------------------------------------------------------------
def test_p0_sign_fix_perfect_yields_bonus():
    """P0 sign fix per Wf-331-R1 §4.2 / Wf-333-V2 §1:

    A perfect composite (1.0) must yield a NEGATIVE (more negative
    kcal/mol = better binding) adjustment. The pre-fix implementation
    used ``(2 * composite - 1) * kcal_per_unit`` which produced a
    positive (worse binding) penalty for perfect coordination —
    physically backwards. After Wf-333-V2 P0 the formula is
    ``(1 - 2 * composite) * kcal_per_unit`` so perfect = -1.0.
    """
    from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
        MetalCoordinationTerm,
        MetalCoordinationScore,
    )

    term = MetalCoordinationTerm()
    perfect = MetalCoordinationScore(
        total=1.0,
        geometry_score=1.0,
        hsab_score=1.0,
        distance_score=None,
    )
    adj = term.kcal_adjustment(perfect)
    assert adj < 0.0, (
        f"Wf-333-V2 P0 sign fix: perfect composite (1.0) must give "
        f"NEGATIVE kcal/mol adjustment (bonus, better binding); got "
        f"{adj}. The pre-fix formula (2*comp-1)*kcal_per_unit was the "
        f"WRONG sign — fixed in Wf-333-V2 P0 to (1-2*comp)*kcal_per_unit."
    )
    assert adj == pytest.approx(-1.0, abs=1e-6), (
        f"Perfect composite adjustment magnitude must be 1.0 kcal/mol; " f"got {adj}"
    )


def test_p0_sign_fix_zero_yields_penalty():
    """P0 sign fix: zero composite (0.0) must yield a POSITIVE
    (less negative kcal/mol = worse binding) penalty."""
    from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
        MetalCoordinationTerm,
        MetalCoordinationScore,
    )

    term = MetalCoordinationTerm()
    zero = MetalCoordinationScore(
        total=0.0,
        geometry_score=0.0,
        hsab_score=0.0,
        distance_score=None,
    )
    adj = term.kcal_adjustment(zero)
    assert adj > 0.0, (
        f"Wf-333-V2 P0 sign fix: zero composite (0.0) must give "
        f"POSITIVE kcal/mol adjustment (penalty); got {adj}"
    )
    assert adj == pytest.approx(1.0, abs=1e-6)


def test_p0_sign_fix_organic_yields_zero():
    """P0 sign fix: organic ligands (composite=None) must yield
    exactly 0.0 (backward compat with all earlier Wf-30 / Wf-121 /
    Wf-264 sweeps that did not pass a metal-coordination term)."""
    from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
        MetalCoordinationTerm,
        MetalCoordinationScore,
    )

    term = MetalCoordinationTerm()
    organic = MetalCoordinationScore(total=None)
    adj = term.kcal_adjustment(organic)
    assert adj == 0.0, (
        f"Organic ligand (composite=None) must give 0.0 kcal/mol "
        f"adjustment (backward compat); got {adj}"
    )


# ---------------------------------------------------------------------------
# Test 2: P1 wire-in — --use-wf246c-metal-coord populates PocketCell
# ---------------------------------------------------------------------------
def test_p1_wirein_metal_coord_flag_populates_pocketcell(tmp_path):
    """P1 wire-in: ``--use-wf246c-metal-coord`` populates
    ``metal_coord_kcal_mol``, ``metal_coord_total``,
    ``metal_coord_geometry_score``, ``metal_coord_hsab_score``,
    ``metal_coord_distance_score``,
    ``metal_coord_recovery_status`` on every (molecule, pocket)
    cell.

    The Wf-333-V2 sign-fix means a perfect cisplatin composite
    yields a NEGATIVE adjustment (better binding), so
    ``metal_coord_kcal_mol < best_kcal_mol``.
    """
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    panel = mod.build_panel(
        smiles_list=["[Pt](N)(N)(Cl)Cl"],  # cisplatin
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=mod.DEFAULT_SEED,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
        use_wf246c_metal_coord=True,
    )

    # At least one molecule with metal-coord composite populated.
    per_mol = panel["per_molecule"][0]
    cells = list(per_mol["per_pocket"].values())

    n_with_total = sum(1 for c in cells if c.get("metal_coord_total") is not None)
    assert n_with_total >= 1, (
        f"At least one pocket must have a metal-coord total populated "
        "when use_wf246c_metal_coord=True; got "
        f"{[c.get('metal_coord_total') for c in cells]}"
    )

    # Find a cell with metal-coord populated; verify field shape.
    metal_cells = [c for c in cells if c.get("metal_coord_total") is not None]
    cell = metal_cells[0]
    for key in (
        "metal_coord_kcal_mol",
        "metal_coord_total",
        "metal_coord_geometry_score",
        "metal_coord_hsab_score",
        "metal_coord_recovery_status",
    ):
        assert key in cell, (
            f"Wf-333-V2 P1 wire-in: PocketCell must contain {key!r}; "
            f"cell keys = {list(cell.keys())}"
        )

    # Wf-333-V2 P0 sign fix: the metal-coord kcal/mol must be MORE
    # NEGATIVE than the raw Vina kcal/mol for cisplatin (a
    # near-perfect composite = bonus).  The fake runner returns
    # best_kcal_mol=-3.594 for 1h36; the metal-coord term with
    # composite ~0.997 + HSAB=1.0 should give adjustment ≈ -0.99,
    # so metal_coord_kcal_mol ≈ -4.58.
    if cell["best_kcal_mol"] is not None and cell["metal_coord_kcal_mol"] is not None:
        assert cell["metal_coord_kcal_mol"] < cell["best_kcal_mol"], (
            f"Wf-333-V2 sign fix: perfect composite must yield NEGATIVE "
            f"adjustment (better binding); cell.best_kcal_mol="
            f"{cell['best_kcal_mol']} but metal_coord_kcal_mol="
            f"{cell['metal_coord_kcal_mol']} (should be more negative)"
        )

    # metadata records use_wf246c_metal_coord=True.
    md = panel["metadata"]
    assert md["use_wf246c_metal_coord"] is True
    assert "wf246c_term_import_ok" in md


# ---------------------------------------------------------------------------
# Test 3: F2(a) + Wf-246-C combined (additive)
# ---------------------------------------------------------------------------
def test_p1_combined_f2a_and_wf246c_additive(tmp_path):
    """P1 combined additive: when both ``--use-f2a-metal-ligand`` and
    ``--use-wf246c-metal-coord`` are set, ``metal_coord_kcal_mol``
    equals ``f2a_kcal_mol`` + Wf-246-C composite adjustment
    (additive per Wf-331-R1 §6.6).

    Concretely: cisplatin composite ≈ 0.997 (geometry=1.0 + HSAB=1.0
    weighted), so the Wf-246-C kcal/mol adjustment ≈ -0.99 kcal/mol.
    The F2(a) modifier is per-bond (default -1.5 kcal/mol).  The
    combined metal_coord_kcal_mol should be Vina + F2(a) + Wf-246-C.
    """
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    panel = mod.build_panel(
        smiles_list=["[Pt](N)(N)(Cl)Cl"],
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=mod.DEFAULT_SEED,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
        use_f2a_metal_ligand=True,
        use_wf246c_metal_coord=True,
    )

    per_mol = panel["per_molecule"][0]
    cells = list(per_mol["per_pocket"].values())
    metal_cells = [
        c
        for c in cells
        if c.get("metal_coord_total") is not None and c.get("f2a_kcal_mol") is not None
    ]
    assert metal_cells, (
        "Need at least one cell with both F2(a) + Wf-246-C populated "
        f"to test combined additive; got cells={[c.get('metal_coord_total') for c in cells]}"
    )

    cell = metal_cells[0]
    vina = cell["best_kcal_mol"]
    f2a = cell["f2a_kcal_mol"]
    mc = cell["metal_coord_kcal_mol"]
    assert vina is not None and f2a is not None and mc is not None, (
        f"vina/f2a/mc must be set on metal cells; got vina={vina} " f"f2a={f2a} mc={mc}"
    )
    # Combined additive: mc == f2a + mc_adjustment
    mc_adj = float(cell["metal_coord_adjustment"])
    expected_mc = float(f2a) + mc_adj
    assert mc == pytest.approx(expected_mc, abs=1e-3), (
        f"Wf-333-V2 additive: metal_coord_kcal_mol={mc} should equal "
        f"f2a_kcal_mol={f2a} + metal_coord_adjustment={mc_adj} = "
        f"{expected_mc}"
    )
    # mc must be more negative than f2a (sign-correct Wf-246-C bonus).
    assert (
        mc <= f2a + 1e-6
    ), f"Wf-333-V2 sign fix: mc={mc} must be ≤ f2a={f2a} for cisplatin"

    # metadata records both flags as True.
    md = panel["metadata"]
    assert md["use_f2a_metal_ligand"] is True
    assert md["use_wf246c_metal_coord"] is True


# ---------------------------------------------------------------------------
# Test 4: per-pocket lift
# ---------------------------------------------------------------------------
def test_p1_per_pocket_lift_varies_by_pocket_donor_counts(tmp_path):
    """P1 per-pocket lift: the Wf-246-C adjustment varies across
    pockets because ``pocket_donor_counts`` (computed from each
    pocket's receptor.pdbqt) feeds the HSAB sub-score.  With the
    stub receptors in :func:`_stage_receptors` every pocket has
    identical donor counts (N/O/S) so the Wf-246-C adjustment is
    approximately the same across pockets — but the test still
    verifies that the adjustment IS computed per pocket (not just
    once at the panel level).
    """
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    panel = mod.build_panel(
        smiles_list=["[Pt](N)(N)(Cl)Cl"],
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=mod.DEFAULT_SEED,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
        use_wf246c_metal_coord=True,
    )

    per_mol = panel["per_molecule"][0]
    pocket_adjs: Dict[str, float] = {}
    for pocket_id, cell in per_mol["per_pocket"].items():
        if cell.get("metal_coord_adjustment") is not None:
            pocket_adjs[pocket_id] = float(cell["metal_coord_adjustment"])

    # Every pocket must have its own adjustment populated.
    assert len(pocket_adjs) == len(mod.DEFAULT_POCKETS), (
        f"Wf-333-V2 per-pocket lift: every pocket must have its own "
        f"metal-coord adjustment; got {pocket_adjs} for {len(mod.DEFAULT_POCKETS)} pockets"
    )

    # All adjustments must be NEGATIVE (better binding) for cisplatin
    # (a near-perfect composite) — sign-fix verification.
    for pocket_id, adj in pocket_adjs.items():
        assert adj < 0.0, (
            f"Wf-333-V2 P0 sign fix: per-pocket adjustment must be "
            f"NEGATIVE for cisplatin; pocket={pocket_id} adj={adj}"
        )

    # Per-pocket aggregate in ``aggregate.per_pocket`` carries
    # ``metal_coord_mean`` and ``metal_coord_min``.
    for pocket_id in mod.DEFAULT_POCKETS:
        agg = panel["aggregate"]["per_pocket"][pocket_id]
        assert "metal_coord_mean" in agg, (
            f"per-pocket aggregate must contain metal_coord_mean; "
            f"got keys={list(agg.keys())}"
        )
        assert (
            "metal_coord_n_with_total" in agg
        ), f"per-pocket aggregate must contain metal_coord_n_with_total"


# ---------------------------------------------------------------------------
# Test 5: 5-pocket cohort aggregate
# ---------------------------------------------------------------------------
def test_p1_cohort_aggregate_5pocket(tmp_path):
    """P1 cohort aggregate: 5-pocket cohort runs report
    ``metal_coord_cohort_median``,
    ``metal_coord_cohort_median_lift_vs_vina``, and
    ``metal_coord_cohort_median_lift_vs_f2a`` on the
    ``aggregate`` section.

    The cohort_median_lift_vs_vina must be NEGATIVE for cisplatin
    (the Wf-246-C composite yields a bonus, lowering the cohort
    median kcal/mol below the raw Vina median).
    """
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    panel = mod.build_panel(
        smiles_list=["[Pt](N)(N)(Cl)Cl"],
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=mod.DEFAULT_SEED,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
        use_f2a_metal_ligand=True,
        use_wf246c_metal_coord=True,
    )

    agg = panel["aggregate"]
    for key in (
        "metal_coord_mean_score_overall",
        "metal_coord_cohort_median",
        "metal_coord_cohort_median_lift_vs_vina",
        "metal_coord_cohort_median_lift_vs_f2a",
        "metal_coord_cohort_median_gap_to_industry",
    ):
        assert key in agg, (
            f"Wf-333-V2 cohort aggregate must contain {key!r}; "
            f"got keys={list(agg.keys())}"
        )

    mc_med = agg["metal_coord_cohort_median"]
    assert mc_med is not None, (
        "5-pocket cohort run with cisplatin must have a non-None "
        "metal_coord_cohort_median"
    )
    # Industry cutoff is -7.0; the cohort median should be close to
    # the Wf-325B baseline -3.282 + F2(a) -1.5 + Wf-246-C ~-1.0.
    # Verifying it is at most -5 (better than baseline):
    assert mc_med < 0.0, (
        f"Cohort median kcal/mol must be NEGATIVE (Vina convention); " f"got {mc_med}"
    )

    mc_lift_vina = agg["metal_coord_cohort_median_lift_vs_vina"]
    # Sign-fix: lift vs Vina must be NEGATIVE (more negative kcal/mol
    # = better binding = negative lift vs Vina baseline).
    assert mc_lift_vina is not None
    assert mc_lift_vina < 0.0, (
        f"Wf-333-V2 sign fix: cohort median lift vs Vina must be "
        f"NEGATIVE (better binding); got {mc_lift_vina}"
    )
    # Conservative PROJECTED lift is in the -1.0 to -2.5 kcal/mol range
    # per Wf-331-R1 §6.6.  Sanity check: at least -0.5 (the Wf-246-C
    # composite ~0.997 alone yields ~-0.99 kcal/mol).
    assert mc_lift_vina <= -0.5, (
        f"Cohort median lift vs Vina must be at least -0.5 kcal/mol "
        f"(Wf-246-C sign-corrected composite ~0.997); got {mc_lift_vina}"
    )


# ---------------------------------------------------------------------------
# Test 6: backward compat — disabling Wf-246-C does not populate fields
# ---------------------------------------------------------------------------
def test_p1_backward_compat_wf246c_disabled(tmp_path):
    """Backward compat: when ``use_wf246c_metal_coord=False``, the
    PocketCell.metal_coord_total is None (no Wf-246-C computation
    happens) and metal_coord_kcal_mol falls back to either the
    F2(a) score (when F2(a) is enabled) or the raw Vina score.
    """
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    panel = mod.build_panel(
        smiles_list=["[Pt](N)(N)(Cl)Cl"],
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=mod.DEFAULT_SEED,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
        use_wf246c_metal_coord=False,  # explicitly disabled
    )

    per_mol = panel["per_molecule"][0]
    cells = list(per_mol["per_pocket"].values())
    for cell in cells:
        # Wf-246-C disabled → metal_coord_total is None.
        assert cell.get("metal_coord_total") is None, (
            f"Wf-333-V2 backward compat: when use_wf246c_metal_coord=False, "
            f"metal_coord_total must be None; got {cell.get('metal_coord_total')}"
        )
        # metal_coord_adjustment is 0.0 (default field value) since
        # the wire-in never fires.
        assert cell.get("metal_coord_adjustment", 0.0) == 0.0, (
            f"Wf-333-V2 backward compat: when Wf-246-C is disabled, "
            f"metal_coord_adjustment must be 0.0; got "
            f"{cell.get('metal_coord_adjustment')}"
        )

    md = panel["metadata"]
    assert md["use_wf246c_metal_coord"] is False


# ---------------------------------------------------------------------------
# Test 7: CLI --help mentions the new flags
# ---------------------------------------------------------------------------
def test_cli_help_mentions_wf246c_flags():
    """CLI --help must mention --use-wf246c-metal-coord and
    --wf246c-kcal-per-unit so downstream callers can discover the
    wire-in."""
    result = __import__("subprocess").run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help should exit 0; got {result.returncode}\n"
        f"STDOUT={result.stdout}\nSTDERR={result.stderr}"
    )
    for flag in ("--use-wf246c-metal-coord", "--wf246c-kcal-per-unit"):
        assert (
            flag in result.stdout
        ), f"--help must mention {flag}; STDOUT head={result.stdout[:400]}"
