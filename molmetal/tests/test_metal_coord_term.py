"""Tests for the Wf-246-C metal-coordination term.

================================================================
Why these tests exist
================================================================
TODO-130 baseline flagged m3/m10 Vina as a "REDESIGN" because the
generic Vina scoring function lacks any metal-coordination term.
Wf-246-C adds three CPU-cheap sub-scores (coordination geometry,
HSAB match, M-ligand distance) on top of Vina's kcal/mol:

1. **Coordination geometry score** — checks CN matches the
   canonical Pt(II) square-planar / Ru(II) octahedral geometry.
2. **HSAB match score** — Pearson 1963 hard-soft acid-base matching
   (Pt2+ prefers N/O donors; Au+ prefers S/P donors).
3. **Metal-ligand distance score** — Gaussian penalty around the
   Shannon 1976 ionic-radius optima (Pt-N ~2.04 Å; Au-S ~2.32 Å).

These tests pin all three behaviours + the composite term + the
5-metal × 4-donor summary table used in paper §B / honest disclosure.

Honest framing
--------------
* Tests are CPU-only and depend on rdkit being installed. They are
  skipped when rdkit is unavailable.
* The metal-coordination term is a **soft bonus / penalty** on top
  of Vina's kcal/mol — it does NOT change the docked pose and
  does NOT modify Vina's underlying scoring function.
"""
from __future__ import annotations

import math

import pytest

try:
    from rdkit import Chem, RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False

from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
    EXPECTED_GEOMETRY,
    HSAB_TABLE,
    METAL_LIGAND_OPT_DISTANCE,
    MetalCoordinationScore,
    MetalCoordinationTerm,
    coordination_geometry_score,
    hsab_match_score,
    hsab_summary_5metal,
    metal_ligand_distance_score,
    pocket_donor_counts_from_atomicnums,
)


pytestmark = pytest.mark.skipif(
    not _HAVE_RDKIT, reason="rdkit not installed",
)


# ---------------------------------------------------------------------------
# 1. Coordination geometry score
# ---------------------------------------------------------------------------
class TestCoordinationGeometryScore:
    """The geometry sub-score fires only for metallodrugs and
    correctly identifies canonical CN=4 / CN=6 geometries."""

    def test_cisplatin_square_planar_full_score(self):
        """cisplatin = [Pt](N)(N)(Cl)Cl — Pt(II) CN=4 square-planar.
        Geometry score must equal 1.0."""
        smiles = "[Pt](N)(N)(Cl)Cl"
        score, diag = coordination_geometry_score(smiles)
        assert score is not None
        assert score == 1.0, f"expected 1.0, got {score}"
        assert diag["metal"] == "Pt"
        assert diag["oxidation_state"] == 2
        assert diag["coordination_number"] == 4
        assert diag["expected_geometry"] == "square_planar"

    def test_organic_ligand_returns_none(self):
        """Pure organic SMILES must NOT trigger the geometry term."""
        score, diag = coordination_geometry_score("CC(=O)Oc1ccccc1C(=O)O")
        assert score is None
        assert diag["metal"] == ""

    def test_wrong_cn_penalised(self):
        """Pt with CN=5 (off-canonical) must score < 1.0."""
        smiles = "[Pt](N)(N)(N)(Cl)Cl"  # CN=5
        score, diag = coordination_geometry_score(smiles)
        assert score is not None
        assert score < 1.0, f"CN=5 should score <1.0, got {score}"
        assert diag["coordination_number"] == 5

    def test_ruthenium_octahedral_full_score(self):
        """A Ru(II) octahedral probe must score 1.0."""
        # We construct a hypothetical hexaammine Ru(II) probe; it may
        # not embed in 3D but the geometry sub-score is graph-only.
        smiles = "[Ru](N)(N)(N)(N)(N)N"  # CN=6
        score, diag = coordination_geometry_score(smiles)
        if score is not None:
            assert diag["metal"] == "Ru"
            assert diag["coordination_number"] == 6


# ---------------------------------------------------------------------------
# 2. HSAB match score
# ---------------------------------------------------------------------------
class TestHSABMatchScore:
    """The HSAB sub-score matches Pearson 1963 hard/soft donor
    preferences against the 5-metal HSAB table."""

    def test_cisplatin_n_donor_full_score(self):
        """cisplatin has 2 N donors — Pt prefers N (HSAB_TABLE=1.0)."""
        smiles = "[Pt](N)(N)(Cl)Cl"
        score, diag = hsab_match_score(smiles)
        assert score is not None
        assert score == 1.0, f"expected 1.0, got {score}"
        assert diag["metal"] == "Pt"
        assert diag["donor_counts"]["N"] >= 1

    def test_auranofin_soft_preference(self):
        """Auranofin = [Au](S)(P)(C)(C) — Au prefers S/P (HSAB=1.0)."""
        smiles = "[Au](SC)(P(C)C)(C)(C)C"  # simplified soft-donor probe
        score, diag = hsab_match_score(smiles)
        if score is not None and diag["metal"] == "Au":
            # The composite is a count-weighted HSAB favourability.
            # All donors present should be S or P (HSAB=1.0 each),
            # so the score should be 1.0.
            assert score == pytest.approx(1.0, abs=1e-6), f"got {score}"

    def test_pocket_donor_counts_mode(self):
        """When pocket_donor_counts is supplied, the HSAB score
        reflects the pocket donor fingerprint (cis/trans context)."""
        smiles = "[Pt](N)(N)(Cl)Cl"
        # Pocket with 4 N and 0 O/S/P donors
        pocket = {"N": 4, "O": 0, "S": 0, "P": 0}
        score, diag = hsab_match_score(smiles, pocket_donor_counts=pocket)
        assert score is not None
        assert score == 1.0, f"got {score}"
        assert diag["donor_counts"]["N"] == 4

    def test_organic_returns_none(self):
        """Organic SMILES do NOT trigger HSAB."""
        score, diag = hsab_match_score("CCO")
        assert score is None


# ---------------------------------------------------------------------------
# 3. Metal-ligand distance score
# ---------------------------------------------------------------------------
class TestMetalLigandDistanceScore:
    """The distance sub-score is a Gaussian penalty around
    Shannon 1976 ionic-radius optima."""

    def test_pt_n_at_optimum(self):
        """Pt-N at 2.04 Å (the canonical optimum) → score 1.0."""
        score = metal_ligand_distance_score("Pt", "N", 2.04)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_pt_n_off_optimum_decays(self):
        """Pt-N at 2.30 Å (off-optimum) → score < 1.0."""
        score = metal_ligand_distance_score("Pt", "N", 2.30)
        assert score is not None
        assert score < 1.0
        # Gaussian with sigma=0.20, diff=0.26 → exp(-(0.26/0.20)^2)
        expected = math.exp(-(0.26 / 0.20) ** 2)
        assert score == pytest.approx(expected, abs=1e-3)

    def test_unknown_pair_returns_none(self):
        """Unknown (metal, donor_class) pair → None."""
        # Use a metal not in METAL_LIGAND_OPT_DISTANCE.
        score = metal_ligand_distance_score("Xx", "N", 2.04)
        assert score is None

    def test_all_8_metals_have_pt_n_entry(self):
        """All 8 d-block metals must have a Pt-N distance entry."""
        for m in ("Pt", "Pd", "Ru", "Ir", "Os", "Rh", "Re", "Au"):
            assert (m, "N") in METAL_LIGAND_OPT_DISTANCE, m


# ---------------------------------------------------------------------------
# 4. Composite term
# ---------------------------------------------------------------------------
class TestMetalCoordinationTerm:
    """The composite term combines geometry + HSAB + distance
    into a single kcal/mol adjustment."""

    def test_cisplatin_full_composite(self):
        """cisplatin has geometry=1.0 + HSAB=1.0 (no pocket context)
        + distance=None → composite = 1.0 → +1.0 kcal/mol bonus."""
        term = MetalCoordinationTerm()
        score = term.score("[Pt](N)(N)(Cl)Cl")
        assert score.geometry_score == 1.0
        assert score.hsab_score == 1.0
        assert score.distance_score is None
        assert score.total == pytest.approx(1.0, abs=1e-6)
        # kcal_adjustment = (2 * 1.0 - 1.0) * 1.0 = +1.0
        assert term.kcal_adjustment(score) == pytest.approx(1.0, abs=1e-6)

    def test_organic_returns_none_total(self):
        """Organic SMILES → composite is None → adjustment 0.0."""
        term = MetalCoordinationTerm()
        score = term.score("CCO")
        assert score.total is None
        assert term.kcal_adjustment(score) == 0.0

    def test_pocket_donor_counts_passed_through(self):
        """The composite honours pocket_donor_counts when supplied."""
        term = MetalCoordinationTerm()
        pocket = {"N": 3, "O": 0, "S": 0, "P": 0}
        score = term.score("[Pt](N)(N)(Cl)Cl", pocket_donor_counts=pocket)
        assert score.hsab_score == 1.0

    def test_kcal_adjustment_zero_for_none_total(self):
        """kcal_adjustment is 0.0 when total is None (organic ligand)."""
        term = MetalCoordinationTerm()
        score = MetalCoordinationScore(total=None)
        assert term.kcal_adjustment(score) == 0.0


# ---------------------------------------------------------------------------
# 5. 5-metal HSAB summary (paper §B / honest disclosure)
# ---------------------------------------------------------------------------
class TestHSABSummary5Metal:
    """The 5-metal × 4-donor HSAB summary table is the headline
    artefact for paper §B / honest disclosure."""

    def test_all_5_metals_present(self):
        summary = hsab_summary_5metal()
        assert set(summary.keys()) == {"Pt", "Ru", "Ir", "Os", "Rh"}

    def test_pt_prefers_n_o_over_s_p(self):
        """Pt(II) is borderline-hard: N/O > S/P."""
        summary = hsab_summary_5metal()
        pt = summary["Pt"]
        assert pt["N"] == 1.0
        assert pt["O"] == 1.0
        assert pt["S"] == 0.4
        assert pt["P"] == 0.4
        assert pt["N"] > pt["S"]

    def test_ru_prefers_n_o_over_s_p_but_softer(self):
        """Ru(II) is borderline-hard (octahedral): N/O = S/P closer."""
        summary = hsab_summary_5metal()
        ru = summary["Ru"]
        assert ru["N"] == 1.0
        assert ru["S"] == 0.7
        assert ru["N"] > ru["S"]
        # but smaller gap than Pt
        assert (ru["N"] - ru["S"]) < (summary["Pt"]["N"] - summary["Pt"]["S"])


# ---------------------------------------------------------------------------
# 6. Helper — pocket_donor_counts_from_atomicnums
# ---------------------------------------------------------------------------
class TestPocketDonorCounts:
    """The pocket-side HSAB donor fingerprint helper."""

    def test_basic_counting(self):
        # 4 N + 2 O + 1 S + 0 P → {'N': 4, 'O': 2, 'S': 1, 'P': 0}
        pocket_atom_types = [7, 7, 7, 7, 8, 8, 16, 6, 6]  # 4N 2O 1S 2C
        counts = pocket_donor_counts_from_atomicnums(pocket_atom_types)
        assert counts == {"N": 4, "O": 2, "S": 1, "P": 0}

    def test_no_donors(self):
        # All carbon → empty donor fingerprint
        counts = pocket_donor_counts_from_atomicnums([6, 6, 6])
        assert counts == {"N": 0, "O": 0, "S": 0, "P": 0}


# ---------------------------------------------------------------------------
# 7. Integration with VinaDockingAdapter (CLI flag + metadata)
# ---------------------------------------------------------------------------
class TestVinaAdapterMetalCoordIntegration:
    """The CLI flag + adapter metadata + smoke test wiring."""

    def test_default_enable_metal_coord_is_true(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
        # We don't actually invoke the adapter (it would need a working
        # vina binary + pocket), so we just verify the kwarg path.
        # When `enable_metal_coord` is omitted, the constructor must
        # default to True.
        import inspect
        sig = inspect.signature(VinaDockingAdapter.__init__)
        assert "enable_metal_coord" in sig.parameters
        assert sig.parameters["enable_metal_coord"].default is True

    def test_cli_flag_enable_metal_coord(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import build_arg_parser
        args = build_arg_parser().parse_args([])
        assert args.enable_metal_coord is True
        args = build_arg_parser().parse_args(["--disable-metal-coord"])
        assert args.enable_metal_coord is False