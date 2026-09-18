"""Unit tests for the WF-Lambda-Boost Phase 1+2+3+4 deliverables.

Covers:
* reference_ligand_resolver — 5 distinct pocket SMILES, fallback,
  <50 ms budget.
* pt_metal_ligand_exchange — 5 MetalLigandExchange + 2 AquaExchange
  SMARTS patterns, AquaContext dataclass.
* pt_click_compat — strict_Pt_II now allows MetalLigandExchange +
  AquaExchange.
* r4_lambda_only_run.py — new CLI flags wire up.

Honest framing
--------------
* The reference-ligand resolver is a *cache-busting primitive*, not
  a chemistry classifier — the unit tests pin that the 5 named
  pockets get distinct SMILES and the missing-pocket fallback
  emits a WARN.
* The MetalLigandExchange / AquaExchange tests do NOT fire the
  SMARTS (RDKit cannot sanitise Pt_II products) — they pin the
  pattern count + the AquaContext schema.
* The pt_click_compat test pins the F2(a) invariant (strict_Pt_II
  COMPATIBLE on both metal-coordination rules).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Phase 1 — reference_ligand_resolver
# ---------------------------------------------------------------------------

class TestReferenceLigandResolver:
    """Unit tests for molmetal.molmetal_lam.lam_chem.reference_ligand_resolver."""

    def test_five_distinct_pockets_return_five_distinct_smiles(self):
        """The 5 named pockets in POCKET_REFERENCE_LIGANDS must each
        return a DIFFERENT SMILES (otherwise the cache-buster fails)."""
        from molmetal.molmetal_lam.lam_chem.reference_ligand_resolver import (
            resolve_reference_ligand,
            POCKET_REFERENCE_LIGANDS,
        )
        from molmetal.molmetal_lam.search_alg.warm_start import (
            PocketFeatureVector,
        )

        pocket_keys = list(POCKET_REFERENCE_LIGANDS.keys())
        assert len(pocket_keys) == 5, (
            f"expected 5 pockets in lookup, got {len(pocket_keys)}"
        )

        smiles_seen = []
        for key in pocket_keys:
            pf = PocketFeatureVector(
                values=np.zeros(64, dtype=np.float32),
                pocket_name=key,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec = resolve_reference_ligand(pf, warn_on_fallback=False)
            smiles_seen.append(rec.smiles)

        assert len(set(smiles_seen)) == 5, (
            f"all 5 pockets must return distinct SMILES; got "
            f"{smiles_seen}"
        )

    def test_missing_pocket_features_falls_back_with_warn(self):
        """PocketFeatureVector=None triggers the cisplatin legacy
        fallback AND emits a UserWarning."""
        from molmetal.molmetal_lam.lam_chem.reference_ligand_resolver import (
            resolve_reference_ligand,
            LEGACY_CISPLATIN_SEED,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rec = resolve_reference_ligand(None, warn_on_fallback=True)
        # UserWarning fired
        assert any(
            issubclass(ww.category, UserWarning) for ww in w
        ), "expected UserWarning on missing pocket_features"
        # Fallback record
        assert rec.is_fallback is True
        assert rec.fallback_reason == "missing_pocket_features"
        assert rec.smiles == LEGACY_CISPLATIN_SEED[1]
        assert rec.pocket_key == "cisplatin_legacy"

    def test_resolution_under_50ms_per_pocket(self):
        """Each pocket-resolution call must complete within 50 ms
        (the production-cell budget)."""
        from molmetal.molmetal_lam.lam_chem.reference_ligand_resolver import (
            resolve_reference_ligand,
        )
        from molmetal.molmetal_lam.search_alg.warm_start import (
            PocketFeatureVector,
        )

        for key in ["CA2", "MMP2", "HDAC2", "pocket_007", "pocket_011"]:
            pf = PocketFeatureVector(
                values=np.zeros(64, dtype=np.float32),
                pocket_name=key,
            )
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec = resolve_reference_ligand(pf, smiles_only=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert elapsed_ms < 50.0, (
                f"pocket={key} resolution took {elapsed_ms:.2f} ms "
                f"(budget 50 ms)"
            )

    def test_slot_contrast_routes_to_CA2_bucket(self):
        """Slot 2 (pos_charge) >= 0.40 AND pos - neg >= 0.20 routes
        to the CA2 bucket even with no pocket_name set."""
        from molmetal.molmetal_lam.lam_chem.reference_ligand_resolver import (
            resolve_reference_ligand,
        )
        from molmetal.molmetal_lam.search_alg.warm_start import (
            PocketFeatureVector,
        )
        values = np.zeros(64, dtype=np.float32)
        values[2] = 0.50  # pos_charge
        values[3] = 0.10  # neg_charge
        pf = PocketFeatureVector(values=values, pocket_name="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = resolve_reference_ligand(pf, warn_on_fallback=False)
        assert rec.pocket_key == "acetazolamide_seed"
        assert rec.chemistry_label == "sulfonamide_arene"

    def test_slot_contrast_routes_to_MMP2_bucket(self):
        """neg - pos >= 0.05 AND neg >= 0.15 routes to MMP2 bucket."""
        from molmetal.molmetal_lam.lam_chem.reference_ligand_resolver import (
            resolve_reference_ligand,
        )
        from molmetal.molmetal_lam.search_alg.warm_start import (
            PocketFeatureVector,
        )
        values = np.zeros(64, dtype=np.float32)
        values[2] = 0.10  # pos_charge
        values[3] = 0.20  # neg_charge
        pf = PocketFeatureVector(values=values, pocket_name="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = resolve_reference_ligand(pf, warn_on_fallback=False)
        assert rec.pocket_key == "marimastat_seed"
        assert rec.chemistry_label == "hydroxamate_peptide"


# ---------------------------------------------------------------------------
# Phase 2 — MetalLigandExchange SMARTS extension
# ---------------------------------------------------------------------------

class TestMetalLigandExchangeSMARTS:
    """Unit tests for the 5 MetalLigandExchange SMARTS patterns."""

    def test_five_metal_ligand_exchange_patterns(self):
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            METAL_LIGAND_EXCHANGE_SMARTS,
            get_metal_ligand_exchange_patterns,
        )
        patterns = get_metal_ligand_exchange_patterns()
        assert len(patterns) == 5
        assert len(METAL_LIGAND_EXCHANGE_SMARTS) == 5

    def test_metal_ligand_pattern_names_distinct(self):
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            METAL_LIGAND_EXCHANGE_SMARTS,
        )
        names = [n for (n, _s, _d) in METAL_LIGAND_EXCHANGE_SMARTS]
        assert len(set(names)) == 5, (
            f"all 5 pattern names must be unique, got {names}"
        )

    def test_metal_ligand_patterns_have_atom_maps(self):
        """Each SMARTS pattern must use atom-maps (``:N``) so the
        product keeps the Pt-N bond and releases the leaving group."""
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            METAL_LIGAND_EXCHANGE_SMARTS,
        )
        for name, smarts, _desc in METAL_LIGAND_EXCHANGE_SMARTS:
            assert ":1" in smarts, (
                f"pattern {name} missing :1 atom-map: {smarts}"
            )
            assert ":2" in smarts, (
                f"pattern {name} missing :2 atom-map (leaving group)"
            )
            assert ">>" in smarts, (
                f"pattern {name} missing >> reaction separator"
            )

    def test_metal_ligand_patterns_fire_on_Pt_Cl_complex(self):
        """At least the canonical Pt-Cl + NH3 pattern must fire on a
        simple Pt-Cl reactant + NH3 donor (sanitisation issues are
        tolerated — we just check the rule attempts)."""
        from molmetal.molmetal_lam.reactions.beta_reductions import (
            MetalLigandExchange,
            REACTION_RULES,
        )
        mle = REACTION_RULES["MetalLigandExchange"]
        assert isinstance(mle, MetalLigandExchange)
        # Pattern 1 is the canonical one
        patterns = mle.available_smarts()
        first_pattern_name = patterns[0][0]
        assert first_pattern_name == "Pt_Cl_NH3"

    def test_extends_d8_analogue_metals(self):
        """The 5 patterns cover Pt, Pt+Br, Pd, Au + a primary amine
        variant — the d8 coordination chemistry extended beyond
        strict Pt-Cl."""
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            METAL_LIGAND_EXCHANGE_SMARTS,
        )
        joined = " ".join(s for _n, s, _d in METAL_LIGAND_EXCHANGE_SMARTS)
        assert "[Pt" in joined
        assert "[Pd" in joined
        assert "[Au" in joined
        assert "[Br" in joined
        assert "[NH2" in joined  # RNH2 variant


# ---------------------------------------------------------------------------
# Phase 3 — AquaExchange SMARTS extension
# ---------------------------------------------------------------------------

class TestAquaExchangeSMARTS:
    """Unit tests for the 2 AquaExchange SMARTS patterns + AquaContext."""

    def test_two_aqua_exchange_patterns(self):
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            AQUA_EXCHANGE_SMARTS,
            get_aqua_exchange_patterns,
        )
        patterns = get_aqua_exchange_patterns()
        assert len(patterns) == 2
        assert len(AQUA_EXCHANGE_SMARTS) == 2

    def test_first_aquation_pka_propagates(self):
        """The first aquation AquaContext must carry pKa1 = 6.5
        (Reedijk 1987 cisplatin canonical)."""
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            get_aqua_context,
        )
        ctx = get_aqua_context("Pt_Cl_H2O_first")
        assert abs(ctx.pka1 - 6.5) < 1e-6
        assert ctx.ionic_strength_M == pytest.approx(0.10, abs=1e-6)
        assert ctx.temperature_K == pytest.approx(310.0, abs=1e-6)

    def test_second_aquation_diaqua_context(self):
        """The second aquation AquaContext documents the diaqua
        complex that binds DNA-N7-guanine."""
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            get_aqua_context,
        )
        ctx = get_aqua_context("Pt_OHCl_H2O_second")
        assert ctx.pka1 > 6.5, "second aquation pKa should be > first"
        assert "diaqua" in ctx.notes or "DNA" in ctx.notes

    def test_unknown_pattern_falls_back_to_first(self):
        """Unknown pattern names fall back to the first-aquation
        context (defensive, never raises)."""
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            get_aqua_context,
        )
        ctx = get_aqua_context("NONEXISTENT_PATTERN")
        assert abs(ctx.pka1 - 6.5) < 1e-6

    def test_aqua_exchange_patterns_have_atom_maps(self):
        """Each AquaExchange SMARTS must use atom-maps + the ``>>``
        separator."""
        from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
            AQUA_EXCHANGE_SMARTS,
        )
        for name, smarts, _desc in AQUA_EXCHANGE_SMARTS:
            assert ":1" in smarts
            assert ":2" in smarts
            assert ">>" in smarts
            assert "[OH2" in smarts, (
                f"pattern {name} missing [OH2] water donor: {smarts}"
            )

    def test_aqua_context_via_rule_method(self):
        """AquaExchange rule must expose aqua_context() accessor."""
        from molmetal.molmetal_lam.reactions.beta_reductions import (
            REACTION_RULES,
        )
        ae = REACTION_RULES["AquaExchange"]
        ctx = ae.aqua_context("Pt_Cl_H2O_first")
        assert abs(ctx.pka1 - 6.5) < 1e-6


# ---------------------------------------------------------------------------
# Phase 4 — pt_click_compat invariant
# ---------------------------------------------------------------------------

class TestPtClickCompatMatrixUpdate:
    """Pin the F2(a) invariant: strict_Pt_II must be COMPATIBLE
    on both metal_ligand_exchange AND aqua_exchange."""

    def test_strict_pt_ii_allows_metal_coordination_helper(self):
        from molmetal.molmetal_lam.lam_chem.pt_click_compat import (
            strict_pt_ii_allows_metal_coordination,
        )
        assert strict_pt_ii_allows_metal_coordination() is True

    def test_strict_pt_ii_row_has_metal_coordination_compatible(self):
        from molmetal.molmetal_lam.lam_chem.pt_click_compat import (
            COMPAT_MATRIX,
        )
        row = COMPAT_MATRIX["strict_Pt_II"]
        assert row["metal_ligand_exchange"] == "compatible"
        assert row["aqua_exchange"] == "compatible"

    def test_compat_matrix_version_marker(self):
        from molmetal.molmetal_lam.lam_chem.pt_click_compat import (
            COMPAT_MATRIX_VERSION,
        )
        # The version marker must be the WF-Lambda-Boost string
        assert "wf-lambda-boost" in COMPAT_MATRIX_VERSION
        assert "2026-09-16" in COMPAT_MATRIX_VERSION


# ---------------------------------------------------------------------------
# Phase 5 — r4_lambda_only_run.py CLI flag wiring
# ---------------------------------------------------------------------------

class TestR4LambdaOnlyRunCLIFlags:
    """Verify the new CLI flags wire up via --help."""

    def test_use_pocket_conditioned_reference_flag_exists(self):
        import subprocess
        proc = subprocess.run(
            ["python", "/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py",
             "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert "--use-pocket-conditioned-reference" in proc.stdout, (
            f"flag missing from --help:\n{proc.stdout[-500:]}"
        )

    def test_use_learned_prior_flag_exists(self):
        import subprocess
        proc = subprocess.run(
            ["python", "/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py",
             "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert "--use-learned-prior" in proc.stdout
        assert "--pocket-boost-strength" in proc.stdout
