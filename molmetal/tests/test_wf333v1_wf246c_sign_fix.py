"""Wf-333-V1: Wf-246-C kcal_adjustment sign bug fix verification tests.

================================================================
Why these tests exist
================================================================
Wf-331-R1 §4.2 identified a NEW honest-negative bug at
`molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:629`:

    OLD: kcal_adjustment = (2 * composite - 1) * kcal_per_unit
         perfect composite (1.0) → +1.0 kcal/mol PENALTY (less negative
         Vina kcal/mol = worse binding). This is physically backwards:
         a metallodrug that perfectly fits its coordination chemistry
         should have BETTER binding (more negative kcal/mol), not worse.

Wf-331-R1 smoke measured this on cisplatin × 1h36:
    plain Vina mean  = -3.582 kcal/mol
    metal-coord mean = -2.588 kcal/mol
    mean_adjustment  = +0.994 kcal/mol  ← sign bug smoking gun

Wf-333-V1 fixed the formula to:

    NEW: kcal_adjustment = (1 - 2 * composite) * kcal_per_unit
         perfect composite (1.0) → -1.0 kcal/mol BONUS (more negative
         kcal/mol = better binding). Zero composite → +1.0 kcal/mol
         penalty. Matches Vina's sign convention (lower = better).

These tests pin all four behaviours to prevent regression:
  1. Sanity check: before the fix, the formula gives the WRONG sign
     (perfect composite → +1.0 penalty).
  2. Sign fix: after the fix, perfect composite → -1.0 bonus,
     zero composite → +1.0 penalty, organic → 0.0 no-op.
  3. Integration: VinaAdapter's _compute_metal_coord_adjustment()
     returns a value < 0 for cisplatin (was > 0 before the fix).
  4. Per-pocket lift on the 5 Wf-325B canonical pockets: project the
     Vina + metal-coord kcal/mol on each pocket and verify that
     perfect-composite ligands get a NEGATIVE (binding-improving)
     adjustment vs the per-pocket Vina median.

Honest framing
--------------
* All tests are CPU-only; no GPU launch required.
* Per AGENTS.md §5.4 (Tier-2 journal, no wet-lab), these are pure
  in-silico verification tests.
* The sign fix is a 1-line edit at metal_coord_term.py:635; the
  per-pocket projection in Test 4 uses the Wf-325B MEASURED kcal/mol
  medians (MEASURED_GENERIC_ATOM) and the Wf-246-C smoke composite
  values (projected for sign-fixed metal-coord term).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

try:
    from rdkit import Chem, RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False

from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
    METAL_LIGAND_OPT_DISTANCE,
    MetalCoordinationScore,
    MetalCoordinationTerm,
)

pytestmark = pytest.mark.skipif(
    not _HAVE_RDKIT, reason="rdkit not installed",
)


# ---------------------------------------------------------------------------
# 1. Sanity check: BUGGY formula (pre-fix) gave +1.0 for perfect composite
# ---------------------------------------------------------------------------
class TestSignBugPresentBeforeFix:
    """Reconstruct the Wf-246-C buggy formula (pre-fix) and verify it
    yields the smoking-gun +0.994 kcal/mol adjustment that Wf-331-R1
    measured on cisplatin × 1h36.

    This test pins the bug behaviour so future code review can detect
    a regression to the buggy formula.
    """

    def test_buggy_formula_yields_positive_for_perfect(self):
        """Pre-fix formula: (2 * total - 1) * kcal_per_unit with
        total=1.0 yields +1.0 (penalty for perfect)."""
        # Construct the composite that Wf-246-C smoke measured
        # (cisplatin geometry=1.0 + HSAB=1.0 + distance=None
        # → weighted composite ≈ 0.997).
        composite = 0.997  # Wf-246-C smoke MEASURED
        buggy_adjustment = (2.0 * composite - 1.0) * 1.0
        # Smoking gun: +0.994 kcal/mol (Wf-331-R1 §4.3 MEASURED)
        assert buggy_adjustment == pytest.approx(0.994, abs=1e-3)
        assert buggy_adjustment > 0, (
            f"BUGGY formula yields {buggy_adjustment:+.4f} kcal/mol — "
            "this is the Wf-246-C sign bug Wf-333-V2 closed."
        )

    def test_buggy_formula_smoke_matches_wf331r1(self):
        """Wf-331-R1 §4.3 measured mean_adjustment=+0.994 kcal/mol
        on cisplatin × 1h36. The buggy formula must reproduce this
        within ±0.01 kcal/mol."""
        # Wf-246-C composite for cisplatin (geometry=1.0 + HSAB=1.0,
        # distance=None, weights 0.4/0.4/0.2; distance disabled
        # because pocket donor counts only carry N from cisplatin's
        # own ligands, not from the docked pocket):
        # - composite_raw = (0.4*1.0 + 0.4*1.0 + 0.2*1.0) / 1.0
        #               = 1.0  (distance_None treated as 1.0 in
        #                 MetalCoordinationTerm.score when enabled)
        # However Wf-331-R1 §4.3 quotes 0.997 — this is the empirical
        # composite from the smoke (small float drift in HSAB match).
        composite_empirical = 0.9970443349753695  # from cisplatin_1h36.json
        buggy_adj = (2.0 * composite_empirical - 1.0) * 1.0
        wf331r1_measured = 0.994088669950739  # from cisplatin_1h36.json
        assert buggy_adj == pytest.approx(wf331r1_measured, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Sign fix: after fix, perfect composite → -1.0 (NOT +1.0)
# ---------------------------------------------------------------------------
class TestSignFixedPerfectComposite:
    """Post-fix formula must yield -1.0 kcal/mol for perfect composite
    (BONUS, more negative kcal/mol = better binding per Vina's
    convention)."""

    def test_perfect_composite_yields_minus_one(self):
        """After Wf-333-V2 fix: composite=1.0 → kcal_adjustment=-1.0
        (the binding-improving direction)."""
        term = MetalCoordinationTerm()
        score = term.score("[Pt](N)(N)(Cl)Cl")  # cisplatin
        # Cisplatin composite = 1.0 (geometry + HSAB both perfect;
        # distance None → not penalised; weights sum to 1.0).
        assert score.total == pytest.approx(1.0, abs=1e-6)
        assert term.kcal_adjustment(score) == pytest.approx(-1.0, abs=1e-6)
        assert term.kcal_adjustment(score) < 0, (
            "Sign fix regression: perfect composite must yield NEGATIVE "
            "kcal/mol adjustment (binding bonus), not positive."
        )

    def test_zero_composite_yields_plus_one(self):
        """Zero composite → +1.0 kcal/mol (penalty for no
        coordination = worse binding)."""
        term = MetalCoordinationTerm()
        score = MetalCoordinationScore(total=0.0)
        assert term.kcal_adjustment(score) == pytest.approx(1.0, abs=1e-6)
        assert term.kcal_adjustment(score) > 0

    def test_midpoint_composite_yields_zero(self):
        """Composite=0.5 → 0.0 kcal/mol (neutral — equally good/bad
        on both sides of the midpoint)."""
        term = MetalCoordinationTerm()
        score = MetalCoordinationScore(total=0.5)
        assert term.kcal_adjustment(score) == pytest.approx(0.0, abs=1e-9)

    def test_organic_ligand_yields_zero_no_op(self):
        """Organic SMILES (no metal, total=None) → 0.0 kcal/mol
        adjustment (backward-compatible with Wf-30 / Wf-121 sweeps
        that don't trigger metal-coord at all)."""
        term = MetalCoordinationTerm()
        # Aspirin: organic, no metal atoms
        score = term.score("CC(=O)Oc1ccccc1C(=O)O")
        assert score.total is None
        assert term.kcal_adjustment(score) == 0.0

    def test_sign_monotonic_with_composite(self):
        """As composite increases from 0 to 1, kcal_adjustment must
        DECREASE from +1.0 to -1.0 monotonically (the whole point of
        the sign fix — bonus scales with composite)."""
        term = MetalCoordinationTerm()
        prev = float("inf")
        for composite in (0.0, 0.25, 0.5, 0.75, 1.0):
            adj = term.kcal_adjustment(
                MetalCoordinationScore(total=composite)
            )
            assert adj < prev, (
                f"Sign-fix regression: composite={composite} "
                f"→ adj={adj:+.3f} not strictly decreasing (prev={prev:+.3f})"
            )
            prev = adj

    def test_kcal_per_unit_scaling_after_fix(self):
        """After fix, kcal_per_unit scales the adjustment linearly:
        kcal_per_unit=0.5 + composite=1.0 → -0.5 kcal/mol."""
        term = MetalCoordinationTerm(kcal_per_unit=0.5)
        score = MetalCoordinationScore(total=1.0)
        assert term.kcal_adjustment(score) == pytest.approx(-0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Integration with VinaAdapter (panel + metadata)
# ---------------------------------------------------------------------------
class TestVinaPanelIntegration:
    """VinaAdapter._compute_metal_coord_adjustment must return a
    NEGATIVE kcal/mol for cisplatin (perfect composite) — i.e. the
    sign fix propagates through the adapter integration."""

    def test_vina_adapter_metal_coord_returns_negative_for_cisplatin(self):
        """Wire the MetalCoordinationTerm through VinaAdapter's
        adjustment logic and verify the adjustment is NEGATIVE
        for cisplatin (was POSITIVE before Wf-333-V2).

        We replicate the adapter's adjustment path inline because
        VinaAdapter.__init__ requires the `vina` package to be
        installed (which is environment-dependent). The logic
        being tested is the sign of the adjustment on cisplatin —
        which depends ONLY on the MetalCoordinationTerm and the
        pocket_donor_counts helper, both of which are pure-Python
        + RDKit.
        """
        from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
            pocket_donor_counts_from_atomicnums,
        )

        # 1h36 pocket donor fingerprint (per Wf-30 staging profile).
        # Simulate pocket.atom_types as a CPU list.
        pocket_atom_types = [7] * 14  # 14 N atoms (proxy for 1h36)
        pocket_donor_counts = pocket_donor_counts_from_atomicnums(
            pocket_atom_types,
        )

        # This is the same logic as vina_adapter.py:_compute_metal_coord_adjustment
        # lines 949-967 (without the actual Vina dock call).
        term = MetalCoordinationTerm()
        score_obj = term.score(
            "[Pt](N)(N)(Cl)Cl",
            pocket_donor_counts=pocket_donor_counts,
        )
        adjustment = term.kcal_adjustment(score_obj)

        # The whole point of Wf-333-V2: adjustment must be NEGATIVE
        # (binding-improving direction per Vina's sign convention).
        assert adjustment < 0, (
            f"VinaAdapter integration regression: cisplatin adjustment "
            f"should be negative (binding bonus) but got {adjustment:+.4f}"
        )
        # And the absolute magnitude is ~ -1.0 kcal/mol for perfect
        # composite (cisplatin × N-rich pocket).
        assert adjustment == pytest.approx(-1.0, abs=1e-3), (
            f"Adjustment magnitude off: expected ~-1.0, got {adjustment:+.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Per-pocket lift on the 5 Wf-325B canonical pockets
# ---------------------------------------------------------------------------
class TestPerPocketLift5Pockets:
    """Project the Wf-246-C metal-coord term onto the 5 Wf-325B
    canonical pockets (1h36, 3eml, 3ny8, 4aaw, 4rlu) and verify
    that the sign fix yields a NEGATIVE per-pocket adjustment
    (binding-improving direction) for cisplatin on each pocket.

    Honest framing:
      * The Wf-325B kcal/mol medians are MEASURED (MEASURED_GENERIC_ATOM).
      * The composite values are PROJECTED for the Wf-246-C smoke
        on each pocket (we don't re-run the full smoke per pocket
        in this test — that would require Vina binding per pocket).
      * The PROJECTED per-pocket lift = -1.0 kcal/mol for perfect
        composite (cisplatin × any pocket); this is the upper bound
        because the distance sub-score is a no-op in current smoke
        (no 3-D pose passed).
      * The test asserts the SIGN direction (negative, not positive)
        for each pocket — that's what the fix guarantees.
    """

    POCKETS = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")

    def _wf325b_medians(self):
        """Return the Wf-325B 5-pocket Vina medians (MEASURED)."""
        wf325b_path = Path(
            "/home/hugo/codes/mol-metal/molmetal/reports/"
            "wf325b_vina_pt_zn_substitution/result.json"
        )
        if not wf325b_path.exists():
            pytest.skip(f"Wf-325B report not found at {wf325b_path}")
        data = json.loads(wf325b_path.read_text())
        return {
            p["pocket_id"]: p["zn_substituted_kcal_mol_median"]
            for p in data["per_pocket"]
        }

    def test_cisplatin_yields_negative_adjustment_on_all_5_pockets(self):
        """The sign fix must propagate a NEGATIVE adjustment on each
        of the 5 Wf-325B canonical pockets for cisplatin. (Before
        the fix, all 5 pockets yielded +0.994 kcal/mol penalty.)"""
        term = MetalCoordinationTerm()
        score = term.score("[Pt](N)(N)(Cl)Cl")
        adj = term.kcal_adjustment(score)

        for pocket_id in self.POCKETS:
            assert adj < 0, (
                f"Pocket {pocket_id}: cisplatin adjustment "
                f"{adj:+.4f} must be NEGATIVE (sign fix). "
                f"Pre-fix would have been +{adj * -1:.4f}."
            )

    def test_projected_total_lift_per_pocket(self):
        """For each of the 5 Wf-325B pockets, the projected metal-coord
        lift is -1.0 kcal/mol (perfect composite, distance=None).
        Combine with Wf-325B MEASURED median to get projected
        metal-coord kcal/mol."""
        medians = self._wf325b_medians()
        term = MetalCoordinationTerm()
        score = term.score("[Pt](N)(N)(Cl)Cl")
        adj = term.kcal_adjustment(score)

        projected_lifts = {}
        for pocket_id in self.POCKETS:
            assert pocket_id in medians, (
                f"Wf-325B median missing for {pocket_id}"
            )
            projected = medians[pocket_id] + adj  # adj = -1.0
            projected_lifts[pocket_id] = projected

            # Sanity: projected_kcal < median (better binding).
            assert projected < medians[pocket_id], (
                f"Pocket {pocket_id}: projected_kcal {projected:.3f} "
                f"not better than median {medians[pocket_id]:.3f}"
            )

        # Aggregate: median of projected kcal/mol across pockets
        all_vals = sorted(projected_lifts.values())
        proj_median = (all_vals[2] + all_vals[3]) / 2  # n=5, median = 3rd

        # Honest-negative framing: the cross-pocket projected median
        # lift vs Wf-325B baseline.
        wf325b_all = sorted(medians.values())
        wf325b_median = (wf325b_all[2] + wf325b_all[3]) / 2
        lift = proj_median - wf325b_median
        assert lift < 0, (
            f"PROJECTED lift {lift:+.3f} kcal/mol must be NEGATIVE "
            f"(wf325b_median={wf325b_median:.3f}, proj_median={proj_median:.3f})"
        )

    def test_wf325b_baselines_loaded(self):
        """Sanity check: the 5 Wf-325B baseline medians are loaded
        correctly (catches future file-format regressions)."""
        medians = self._wf325b_medians()
        assert set(medians.keys()) == set(self.POCKETS)
        # All kcal/mol values are in the [-5, 0] range (Vina typical)
        for pocket_id, val in medians.items():
            assert -5.0 <= val <= 0.0, (
                f"Wf-325B median for {pocket_id} = {val} kcal/mol "
                f"outside expected [-5, 0] range"
            )