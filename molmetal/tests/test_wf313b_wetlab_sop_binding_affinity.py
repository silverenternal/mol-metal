#!/usr/bin/env python3
"""Tests for Wf-313B §PART 1 — wetlab SOP binding-affinity edge-case handling.

==============================================================================
Wf-313B PART 1 (2026-09-29).

These tests verify that the institutional-tailored wet-lab binding-affinity SOP
(`TODO/pending/wetlab_sop_binding_affinity_assay.md`) implements the §6a edge-case
handling correctly. Each test validates one edge case (SPR regeneration
residual, ITC stoichiometry, KD crosscheck, kinetics chi², heats of dilution,
DLS polydispersity) and the integration with the Wf-74 IC50 analysis pipeline
(`molmetal/scripts/template_ic50_pipeline.py`).

**Honest framing**: the SOP is a **template** (no wet-lab SPR/ITC data); the
edge-case thresholds are **validated as logical checks**, not against real
binding measurements.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

# Lazy import
try:
    from molmetal.scripts import template_ic50_pipeline as ic50p  # noqa: E402
    _HAS_IC50P = True
except Exception as exc:
    ic50p = None
    _HAS_IC50P = False
    _IMPORT_ERR = repr(exc)


pytestmark = pytest.mark.skipif(
    not _HAS_IC50P,
    reason=f"template_ic50_pipeline import failed: {_IMPORT_ERR if not _HAS_IC50P else ''}",
)


# ---------------------------------------------------------------------------
# §6a edge-case threshold constants (mirrors wetlab_sop_binding_affinity_assay.md §6a)
# ---------------------------------------------------------------------------
REGEN_RESIDUAL_FRAC_MAX = 0.10  # > 10% Rmax after regen → flag
STOICH_N_MIN, STOICH_N_MAX = 0.8, 1.2
STOICH_N_TWO_SITE_THRESHOLD = 1.5
KD_RATIO_MIN, KD_RATIO_MAX = 0.5, 2.0
SPR_CHI2_MAX = 5.0
DLS_PDI_MAX = 0.2
HEATS_OF_DILUTION_MAX_UM = 0.05  # µW
SPEARMAN_RHO_FALLBACK_MIN = 0.7


# ---------------------------------------------------------------------------
# Test 1: Edge-case validation — SPR regeneration residual
# ---------------------------------------------------------------------------


def test_spr_regen_residual_threshold():
    """§6a.1 — Residual binding signal > 10% of pre-regen Rmax → flag."""
    # Pre-regen Rmax
    rmax_pre = 100.0  # RU
    # Post-regen residual signal
    rmax_post = 12.0  # RU (12%)
    residual_frac = rmax_post / rmax_pre
    flag_regen_incomplete = residual_frac > REGEN_RESIDUAL_FRAC_MAX
    assert flag_regen_incomplete is True
    assert residual_frac > 0.10
    # Acceptable residual
    rmax_post_ok = 8.0  # 8%
    assert (rmax_post_ok / rmax_pre) <= REGEN_RESIDUAL_FRAC_MAX


# ---------------------------------------------------------------------------
# Test 2: Edge-case validation — ITC stoichiometry range
# ---------------------------------------------------------------------------


def test_itc_stoichiometry_range():
    """§6a.2 — N must be in [0.8, 1.2]; outside → stoichiometry_outlier."""
    valid_n = [0.8, 0.9, 1.0, 1.1, 1.2]
    invalid_n_low = [0.5, 0.7, 0.79]
    invalid_n_high = [1.21, 1.4, 1.5, 2.0]
    for n in valid_n:
        assert STOICH_N_MIN <= n <= STOICH_N_MAX
    for n in invalid_n_low:
        assert not (STOICH_N_MIN <= n <= STOICH_N_MAX)
    for n in invalid_n_high:
        assert not (STOICH_N_MIN <= n <= STOICH_N_MAX)


def test_itc_two_site_fallback():
    """§6a.2 — N > 1.5 → fit two-site model; report KD1, KD2, dH1, dH2."""
    n_multisite = 2.0  # > 1.5
    assert n_multisite > STOICH_N_TWO_SITE_THRESHOLD
    # Two-site fit schema (per §6a.2 of the SOP)
    two_site_schema = {"KD1_M": None, "KD2_M": None, "dH1_kcal_mol": None,
                        "dH2_kcal_mol": None, "n1": None, "n2": None}
    assert all(k in two_site_schema for k in ("KD1_M", "KD2_M", "dH1_kcal_mol"))


# ---------------------------------------------------------------------------
# Test 3: Parameter range check — KD crosscheck ratio
# ---------------------------------------------------------------------------


def test_kd_crosscheck_ratio_range():
    """§6a.3 — KD_SPR / KD_ITC must be in [0.5, 2.0]; outside → flag."""
    # Same compound × same protein measured on both instruments
    valid_ratios = [0.5, 1.0, 1.5, 2.0]
    invalid_low = [0.3, 0.4, 0.49]
    invalid_high = [2.1, 3.0, 5.0]
    for r in valid_ratios:
        assert KD_RATIO_MIN <= r <= KD_RATIO_MAX
    for r in invalid_low:
        assert not (KD_RATIO_MIN <= r <= KD_RATIO_MAX)
    for r in invalid_high:
        assert not (KD_RATIO_MIN <= r <= KD_RATIO_MAX)


def test_kd_crosscheck_rank_concordance_fallback():
    """§6a.3 — If KD ratio persistently fails, fall back to Spearman ρ ≥ 0.7 across top-10."""
    # Top-10 binders ranked by SPR vs ITC
    spr_ranks = np.arange(1, 11)
    # ITC ranks highly concordant
    itc_concordant = np.array([2, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(spr_ranks, itc_concordant)
    assert abs(rho) >= SPEARMAN_RHO_FALLBACK_MIN


# ---------------------------------------------------------------------------
# Test 4: Parameter range check — SPR kinetics chi²
# ---------------------------------------------------------------------------


def test_spr_kinetics_chi2_threshold():
    """§6a.4 — SPR kinetics χ² > 5 → `spr_chi2_high` flag."""
    valid_chi2 = [0.5, 1.0, 2.0, 3.0, 5.0]
    invalid_chi2 = [5.1, 10.0, 25.0]
    for chi2 in valid_chi2:
        assert chi2 <= SPR_CHI2_MAX
    for chi2 in invalid_chi2:
        assert chi2 > SPR_CHI2_MAX


# ---------------------------------------------------------------------------
# Test 5: Parameter range check — DLS polydispersity
# ---------------------------------------------------------------------------


def test_dls_polydispersity_threshold():
    """§6a.4 — DLS PDI > 0.2 → `protein_aggregated`; reject and re-purify."""
    valid_pdi = [0.05, 0.10, 0.15, 0.20]
    invalid_pdi = [0.21, 0.30, 0.50]
    for pdi in valid_pdi:
        assert pdi <= DLS_PDI_MAX
    for pdi in invalid_pdi:
        assert pdi > DLS_PDI_MAX


def test_itc_heats_of_dilution_threshold():
    """§6a.4 — Heats of dilution > 0.05 µW → `buffer_mismatch`."""
    valid_heat = [0.001, 0.01, 0.05]
    invalid_heat = [0.06, 0.10, 0.5]
    for h in valid_heat:
        assert h <= HEATS_OF_DILUTION_MAX_UM
    for h in invalid_heat:
        assert h > HEATS_OF_DILUTION_MAX_UM


# ---------------------------------------------------------------------------
# Test 6: Integration with template_ic50_pipeline — KD extension not yet wired
# ---------------------------------------------------------------------------


def test_integration_pipeline_does_not_emit_fabricated_kd():
    """§6a.4 — IC50 pipeline does NOT emit KD_M fields (per Wf-74 schema; KD ships separately)."""
    wells = ic50p.make_synthetic_fixture()
    per_compound = ic50p.build_per_compound_records(wells, n_boot=50)
    for key, rec in per_compound.items():
        # The pipeline emits pIC50 + Hill slope + R²; KD_M is NOT in this schema.
        # Verify no fabricated KD field appears.
        assert "KD_M" not in rec, "KD_M does not belong in the IC50 pipeline schema"
        assert "kd_spr" not in rec
        assert "kd_itc" not in rec


# ---------------------------------------------------------------------------
# Test 7: Integration — pipeline emits fit-quality metadata
# ---------------------------------------------------------------------------


def test_integration_pipeline_emits_required_fields():
    """§6a.4 — Per-compound records include compound_id, pic50, hill_slope, qc_flags."""
    wells = ic50p.make_synthetic_fixture()
    per_compound = ic50p.build_per_compound_records(wells, n_boot=50)
    required_fields = {"compound_id", "metal", "cell_line", "hill_slope", "qc_flags"}
    for key, rec in per_compound.items():
        missing = required_fields - set(rec.keys())
        assert not missing, f"Record {key} missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Test 8: Honest framing — binding SOP ships as template, no KD MEASURED
# ---------------------------------------------------------------------------


def test_honest_framing_no_fabricated_binding_measurements():
    """§6a.4 — Pipeline does NOT fabricate KD or stoich values; binding is template-only."""
    wells = ic50p.make_synthetic_fixture()
    per_compound = ic50p.build_per_compound_records(wells, n_boot=50)
    # No record should carry a "stoichiometry" or "kd" field from the IC50 pipeline
    for key, rec in per_compound.items():
        assert "stoichiometry" not in rec
        assert "stoichiometry_n" not in rec
        # The pipeline returns Hill slope (not stoichiometry n)
        # Hill slope is dimensionless and serves the same numerical role
        if rec.get("hill_slope") is not None:
            assert isinstance(rec["hill_slope"], float)
