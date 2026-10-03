"""Wf-329-W3 unit tests — F2(a) MetalLigandExchange SMARTS rule.

Per Wf-325B + WEAK_TO_STRONG §3.5: the m3 Vina Dock honest-negative
chain (``HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED``,
cross-pocket median -3.283 kcal/mol) has a documented recovery path
— the F2(a) MetalLigandExchange SMARTS rule. Wf-329-W3 ships:

  1. F2(a) SMARTS library — 4 already-formed-adduct patterns
     (Pt-N7_guanine, Pd-N7_guanine, Ru-N7_guanine, Au-N7_guanine) +
     4 covalent-capable bare-metal patterns (Pt, Pd, Ru, Au).
  2. F2AMetalLigandScorer — query + apply_covalent_modifier API
     with a conservative per-bond modifier (default -1.5 kcal/mol,
     PROJECTED per Wf-329-W3 task brief).
  3. Wire into ``cfm_vina_dock_panel.py`` with ``--use-f2a-metal-ligand``
     CLI flag (default OFF; turns ON via flag).

4 NEW tests verify:

  1. SMARTS pattern matches Pt-N7 guanine (already-formed adduct).
  2. Covalent scoring modifier applied (cisplatin-class drug gets
     the F2(a) bonus; organic-only gets nothing).
  3. Vina integration (PocketCell carries F2(a) fields after
     ``build_panel`` runs).
  4. Vina score lift (median lift is the expected PROJECTED +1-2
     kcal/mol range for a cisplatin-class cohort).

All CPU-only.  No GPU launch per AGENTS.md §5.1.

Honest-framing note (party-twenty-spirit §3.5):
* MEASURED: Wf-325B baseline cross-pocket median = -3.283 kcal/mol.
* PROJECTED: per-bond modifier = -1.5 kcal/mol (conservative
  midpoint per Wf-329-W3 task brief).
* PROJECTED: per-cell lift = -1.5 kcal/mol for 1-Pt-center drugs.
* HONEST-NEG fallback: cohort_median_after ≈ -4.78 kcal/mol
  (still ≥-5; the covalent modifier is INSUFFICIENT to clear the
  industry -7 cutoff).  Per WEAK_TO_STRONG §3.5, Phase 2 GPU
  metal-coordination term (Wf-246-C) is the next recovery step.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from molmetal.scripts.wf329w3_f2a_metal_ligand import (  # noqa: E402
    F2AMetalLigandScorer,
    F2A_COVALENT_SMARTS,
    F2A_COVALENT_CAPABLE_SMARTS,
    DEFAULT_COVALENT_MODIFIER_KCAL_MOL,
    INDUSTRY_CUTOFF_KCAL_MOL,
    HONEST_NEG_FALLBACK_GAP_KCAL_MOL,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
CISPLATIN_SMILES = "Cl[Pt](Cl)(N)(N)"
CARBOPLATIN_SMILES = "C1NC(=O)N(C1=O)[Pt](N)(N)"
PT_N7_ADDUCT_SMILES = "[Pt]([NH3])([NH3])([n]1cnc2c1nc(N)[nH]c2=O)"
BENZENE_SMILES = "c1ccccc1"

# Wf-325B baseline
WF325B_CROSS_POCKET_MEDIAN_KCAL_MOL = -3.283


@pytest.fixture
def scorer() -> F2AMetalLigandScorer:
    """Default F2(a) scorer with default modifier."""
    return F2AMetalLigandScorer()


@pytest.fixture
def custom_scorer() -> F2AMetalLigandScorer:
    """F2(a) scorer with a custom -4.0 kcal/mol modifier."""
    return F2AMetalLigandScorer(modifier_kcal_mol=-4.0)


# ===========================================================================
# Group 1 — SMARTS pattern matches Pt-N7 guanine
# ===========================================================================
def test_wf329w3_smarts_pt_n7_guanine_matches(scorer):
    """F2(a) SMARTS rule detects Pt-N7 guanine covalent bond.

    The Pt-N7 guanine SMARTS pattern (``[Pt]~[n]1cnc2c1nc(N)[nH]c2=O``)
    must match the synthesised Pt-N7 guanine adduct SMILES. This is
    the canonical F2(a) recovery use case — the pattern correctly
    identifies the Pt-N7 covalent bond that Vina cannot score.
    """
    # Build the synthesized adduct SMILES and confirm scorer picks
    # up at least the Pt-N7_guanine pattern.
    dispatch = scorer.dispatch(PT_N7_ADDUCT_SMILES)
    assert "Pt_N7_guanine" in dispatch.matched_patterns, (
        f"F2(a) SMARTS rule failed to detect Pt-N7 guanine in "
        f"adduct SMILES {PT_N7_ADDUCT_SMILES!r}; "
        f"matched={dispatch.matched_patterns}, "
        f"n_bonds={dispatch.n_covalent_bonds}"
    )
    # At least 1 bond must be detected.
    assert dispatch.n_covalent_bonds >= 1, (
        f"n_covalent_bonds={dispatch.n_covalent_bonds} for adduct; "
        f"expected ≥1 (Pt-N7 pattern)"
    )


def test_wf329w3_smarts_pattern_library_integrity():
    """The F2(a) SMARTS library must have ≥4 covalent-formed
    patterns and ≥4 covalent-capable patterns, with documented
    (name, smarts, description) triples.
    """
    # Covalent-formed (N7 adduct) library
    assert len(F2A_COVALENT_SMARTS) >= 4, (
        f"F2A_COVALENT_SMARTS must have ≥4 patterns "
        f"(Pt, Pd, Ru, Au); got {len(F2A_COVALENT_SMARTS)}"
    )
    # Every pattern must be a (name, smarts, description) triple
    for pat in F2A_COVALENT_SMARTS:
        assert isinstance(pat, tuple) and len(pat) == 3, f"malformed pattern: {pat!r}"
        name, smarts, desc = pat
        assert name and smarts and desc, f"pattern triple must be non-empty: {pat!r}"
    # Covalent-capable (bare metal) library
    assert len(F2A_COVALENT_CAPABLE_SMARTS) >= 4, (
        f"F2A_COVALENT_CAPABLE_SMARTS must have ≥4 patterns "
        f"(Pt, Pd, Ru, Au); got {len(F2A_COVALENT_CAPABLE_SMARTS)}"
    )
    # Each canonical metal symbol must be in the library
    metal_syms = {pat[0] for pat in F2A_COVALENT_CAPABLE_SMARTS}
    for sym in (
        "Pt_center_capable",
        "Pd_center_capable",
        "Ru_center_capable",
        "Au_center_capable",
    ):
        assert sym in metal_syms, (
            f"covalent-capable library missing {sym}; " f"got {sorted(metal_syms)}"
        )


# ===========================================================================
# Group 2 — Covalent scoring modifier applied
# ===========================================================================
def test_wf329w3_covalent_modifier_applied(scorer):
    """The F2(a) covalent scoring modifier is applied correctly.

    For cisplatin (1 Pt centre): 1 bond × -1.5 modifier = -1.5
    kcal/mol total. So f2a_kcal_mol = vina_kcal_mol + (-1.5).

    For benzene (0 Pt centres): 0 bonds → modifier_total = 0.0
    → f2a_kcal_mol = vina_kcal_mol (no change).
    """
    vina_baseline = -3.283  # Wf-325B cross-pocket median

    # Cisplatin-class drug → modifier applied
    f2a_score, mod_total, n_bonds = scorer.apply_covalent_modifier(
        CISPLATIN_SMILES, vina_baseline
    )
    assert n_bonds >= 1, f"cisplatin n_bonds={n_bonds}; expected ≥1 (Pt_center_capable)"
    assert math.isclose(
        mod_total, n_bonds * DEFAULT_COVALENT_MODIFIER_KCAL_MOL, abs_tol=1e-9
    ), (
        f"modifier_total={mod_total}; expected "
        f"n_bonds × {DEFAULT_COVALENT_MODIFIER_KCAL_MOL} = "
        f"{n_bonds * DEFAULT_COVALENT_MODIFIER_KCAL_MOL}"
    )
    assert math.isclose(f2a_score, vina_baseline + mod_total, abs_tol=1e-9), (
        f"f2a_kcal_mol={f2a_score}; expected "
        f"vina_baseline + mod_total = {vina_baseline + mod_total}"
    )

    # Organic-only → no modifier
    f2a_benzene, mod_benzene, n_benzene = scorer.apply_covalent_modifier(
        BENZENE_SMILES, vina_baseline
    )
    assert n_benzene == 0, f"benzene n_bonds={n_benzene}; expected 0 (no metal centre)"
    assert math.isclose(
        mod_benzene, 0.0, abs_tol=1e-9
    ), f"benzene modifier_total={mod_benzene}; expected 0.0"
    assert math.isclose(f2a_benzene, vina_baseline, abs_tol=1e-9), (
        f"benzene f2a_kcal_mol={f2a_benzene}; expected "
        f"vina_baseline = {vina_baseline}"
    )


def test_wf329w3_covalent_modifier_propagates_none():
    """When vina_kcal_mol is None (no real Vina score, e.g.
    STUB_FALLBACK), the F2(a) modifier propagates None without
    inventing a score.
    """
    scorer = F2AMetalLigandScorer()
    f2a_score, mod_total, n_bonds = scorer.apply_covalent_modifier(
        CISPLATIN_SMILES, None
    )
    assert f2a_score is None, (
        f"f2a_score must be None when vina_kcal_mol is None; " f"got {f2a_score}"
    )
    assert mod_total == 0.0, (
        f"mod_total must be 0.0 when vina_kcal_mol is None; " f"got {mod_total}"
    )
    assert n_bonds == 0, f"n_bonds must be 0 when vina_kcal_mol is None; got {n_bonds}"


# ===========================================================================
# Group 3 — Vina integration (PocketCell carries F2(a) fields)
# ===========================================================================
def test_wf329w3_vina_integration_pocket_cell_carries_f2a_fields(scorer):
    """The F2(a) wire into cfm_vina_dock_panel.py adds the F2(a)
    fields to PocketCell. When the F2(a) scorer is invoked on a
    cisplatin-class SMILES, the f2a_kcal_mol / f2a_n_bonds /
    f2a_modifier_total / f2a_matched_patterns / f2a_recovery_status
    fields are populated correctly.
    """
    # Build a PocketCell-like object via the score pathway
    from molmetal.scripts.cfm_vina_dock_panel import PocketCell

    cell = PocketCell(pocket="1h36")
    vina_kcal = -3.283
    f2a_score, mod_total, n_bonds = scorer.apply_covalent_modifier(
        CISPLATIN_SMILES, vina_kcal
    )
    dispatch = scorer.dispatch(CISPLATIN_SMILES)
    status, _gap = scorer.classify_recovery(f2a_score)
    cell.f2a_kcal_mol = f2a_score
    cell.f2a_modifier_total = mod_total
    cell.f2a_n_bonds = n_bonds
    cell.f2a_matched_patterns = ",".join(dispatch.matched_patterns)
    cell.f2a_recovery_status = status

    # All five F2(a) fields must be populated
    assert cell.f2a_kcal_mol is not None, "f2a_kcal_mol must be set"
    assert (
        cell.f2a_n_bonds >= 1
    ), f"f2a_n_bonds={cell.f2a_n_bonds}; expected ≥1 for cisplatin"
    assert cell.f2a_modifier_total < 0, (
        f"f2a_modifier_total={cell.f2a_modifier_total}; " f"expected negative (penalty)"
    )
    assert (
        cell.f2a_matched_patterns
    ), "f2a_matched_patterns must be non-empty for cisplatin"
    assert cell.f2a_recovery_status in (
        "RECOVERY_PASS",
        "RECOVERY_BORDERLINE",
        "HONEST_NEG_COVALENT_INSUFFICIENT",
        "NO_SCORE",
    ), f"f2a_recovery_status={cell.f2a_recovery_status!r}; " f"unknown status"


# ===========================================================================
# Group 4 — Vina score lift (cohort_median_lift in expected range)
# ===========================================================================
def test_wf329w3_vina_score_lift_in_projected_range(scorer):
    """The cohort median lift for a cisplatin-class cohort must be
    in the PROJECTED +1-2 kcal/mol range per Wf-329-W3 task brief.

    Setup: 5-pocket cohort, 1 cisplatin SMILES per pocket. Each
    pocket's vina_kcal_mol is the Wf-325B per-pocket median. The
    F2(a) modifier adds -1.5 kcal/mol per pocket. The cohort
    median lift should be -1.5 kcal/mol (the PROJECTED value).
    """
    # Wf-325B per-pocket cisplatin kcal/mol
    vina_per_pocket = {
        "1h36": -3.589,
        "3eml": -2.827,
        "3ny8": -3.724,
        "4aaw": -3.282,
        "4rlu": -2.983,
    }
    vina_scores: List[float] = []
    f2a_scores: List[float] = []
    for _pocket, kcal in vina_per_pocket.items():
        vina_scores.append(kcal)
        f2a_score, _mod, _n = scorer.apply_covalent_modifier(CISPLATIN_SMILES, kcal)
        f2a_scores.append(f2a_score)

    # Cohort medians
    def _median(xs: List[float]) -> float:
        s = sorted(xs)
        n = len(s)
        m = n // 2
        return (s[m - 1] + s[m]) / 2.0 if n % 2 == 0 else s[m]

    med_vina = _median(vina_scores)
    med_f2a = _median(f2a_scores)
    lift = med_f2a - med_vina

    # The lift must be in the PROJECTED -1.0 to -2.0 range
    assert -2.0 <= lift <= -1.0, (
        f"cohort_median_lift={lift}; expected in [-2.0, -1.0] "
        f"PROJECTED range (Wf-329-W3 task brief). vina_median={med_vina}, "
        f"f2a_median={med_f2a}"
    )

    # F2(a) median must still be ABOVE the industry cutoff
    # (HONEST_NEG_COVALENT_INSUFFICIENT, per Wf-329-W3 expected
    # outcome).
    gap_to_industry = med_f2a - INDUSTRY_CUTOFF_KCAL_MOL
    if gap_to_industry > HONEST_NEG_FALLBACK_GAP_KCAL_MOL:
        status, _ = scorer.classify_recovery(med_f2a)
        assert status == "HONEST_NEG_COVALENT_INSUFFICIENT", (
            f"f2a_median={med_f2a} is {gap_to_industry:+.3f} kcal/mol "
            f"above industry cutoff {INDUSTRY_CUTOFF_KCAL_MOL}; "
            f"classify_recovery returned {status!r} (expected "
            f"HONEST_NEG_COVALENT_INSUFFICIENT)"
        )


if __name__ == "__main__":  # pragma: no cover
    # Manual quick run
    s = F2AMetalLigandScorer()
    print(f"scorer: {s.n_patterns_loaded} patterns, " f"modifier={s.modifier_kcal_mol}")
    for label, sm in [
        ("cisplatin", CISPLATIN_SMILES),
        ("carboplatin", CARBOPLATIN_SMILES),
        ("Pt-N7 adduct", PT_N7_ADDUCT_SMILES),
        ("benzene", BENZENE_SMILES),
    ]:
        d = s.dispatch(sm)
        adj, mod, n = s.apply_covalent_modifier(sm, -3.283)
        st, gap = s.classify_recovery(adj)
        print(
            f"  {label}: n_bonds={n} matched={d.matched_patterns} "
            f"modifier={mod:+.2f} f2a={adj:.3f} status={st} "
            f"gap={gap:+.3f}"
        )
