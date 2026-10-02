"""Pytest: Wf-313E — MECHANISM_METRIC_MAPPING M22-M44 documentation extension.

This test module pins the Wf-313E contract:

1. Each of the 23 R18 mechanisms (M22-M44) has a documented entry in
   :file:`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`.
2. Each entry carries the required fields: mechanism description,
   file/module path, theorem anchor, affected metrics, target gate.
3. The three wave-level cross-mapping tables (W1 / W2 / W3) are
   populated.
4. The §4 regression-debug / failure-mode table covers all 20 metrics
   in :file:`TODO/02_metrics/METRICS_TARGETS.md`.
5. The ablation design table (§7) has at least one row per new
   mechanism family (M22-M28 / M29-M35 / M36-M44).

These are **documentation-only** tests — they do NOT run any GPU
training. All 23 source modules are CPU-only shipping per
``TODO/03_plans/R18_PLAN.md`` (Blackwell single-card §risk deferred).

**Honest framing** (per AGENTS.md §1.5 / party-twenty-spirit §3.5):
- All 23 R18 mechanism modules are MEASURED-shipped at the
  documentation + import level (308 / 308 tests pass per
  ``MECHANISM_METRIC_MAPPING.md`` §6 footer).
- Per-mechanism target gates are PROJECTED (R18 GPU retrain is
  deferred — see ``TODO/03_plans/R18_PLAN.md`` Blackwell single-card
  §risk). Tests do NOT assert GPU-trained values.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAPPING_PATH = Path(__file__).resolve().parents[2] / "TODO" / "02_metrics" / "MECHANISM_METRIC_MAPPING.md"
METRICS_PATH = Path(__file__).resolve().parents[2] / "TODO" / "02_metrics" / "METRICS_TARGETS.md"

# All 23 new R18 mechanisms grouped by wave.
M22_M28_WAVE1 = ["M22", "M23", "M24", "M25", "M26", "M27", "M28"]
M29_M35_WAVE2 = ["M29", "M30", "M31", "M32", "M33", "M34", "M35"]
M36_M44_WAVE3 = ["M36", "M37", "M38", "M39", "M40", "M41", "M42", "M43", "M44"]
ALL_NEW_MECHANISMS = M22_M28_WAVE1 + M29_M35_WAVE2 + M36_M44_WAVE3

# 20 metrics from METRICS_TARGETS.md §2.
ALL_METRICS = [f"m{i}" for i in range(1, 21)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mapping_text() -> str:
    """Read the MECHANISM_METRIC_MAPPING.md file once per module."""
    assert MAPPING_PATH.exists(), f"missing {MAPPING_PATH}"
    return MAPPING_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metrics_text() -> str:
    """Read the METRICS_TARGETS.md file once per module."""
    assert METRICS_PATH.exists(), f"missing {METRICS_PATH}"
    return METRICS_PATH.read_text(encoding="utf-8")


def _section(text: str, anchor: str) -> str:
    """Return the substring from ``anchor`` to the next ``## `` heading."""
    idx = text.find(anchor)
    assert idx >= 0, f"anchor {anchor!r} not found"
    rest = text[idx + len(anchor):]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


# ---------------------------------------------------------------------------
# W1 — M22-M28 (7 mechanisms)
# ---------------------------------------------------------------------------


def test_m22_se3_sb(mapping_text: str) -> None:
    """M22 SE(3) Schrödinger Bridge: entry + affected metrics m3 / m11 / m15."""
    body = _section(mapping_text, "### M22 SE(3) Schrödinger Bridge solver")
    assert "Mahmood 2026" in body, "missing theorem anchor"
    for tag in ("m3", "m11", "m15"):
        assert tag in body, f"M22 should affect {tag}"
    assert "PROJECTED" in body, "must label target gate as PROJECTED (GPU-blocked)"


def test_m23_transport_relaxation(mapping_text: str) -> None:
    """M23 Transport Relaxation: KL endpoint penalty + Sinkhorn-5-iter."""
    body = _section(mapping_text, "### M23 Transport Relaxation")
    assert "Jiang-Xu-Zhang 2026" in body, "missing theorem anchor"
    for tag in ("m14", "m1"):
        assert tag in body, f"M23 should affect {tag}"
    assert "PROJECTED" in body


def test_m24_multi_parameter_ph(mapping_text: str) -> None:
    """M24 Multi-Parameter Persistence: 2-PH filtration + PLL pipeline."""
    body = _section(mapping_text, "### M24 Multi-Parameter Persistence")
    assert "Kim & Jung 2026" in body or "Kim &amp; Jung 2026" in body, "missing theorem anchor"
    for tag in ("m7", "m11"):
        assert tag in body, f"M24 should affect {tag}"


def test_m25_hilbert_primal_dual(mapping_text: str) -> None:
    """M25 Hilbert primal-dual: Chambolle-Pock O(1/k) on L²[0,1] Fourier basis."""
    body = _section(mapping_text, "### M25 Hilbert primal-dual optimizer")
    assert "Chambolle-Pock" in body or "Sun X.-K." in body, "missing theorem anchor"
    assert "ASTSR" in body, "M25 replaces / augments ASTSR R² baseline"


def test_m26_score_repellent_mc(mapping_text: str) -> None:
    """M26 Score-Repellent Monte Carlo: Burch-Bardsley 2022 Langevin MCMC."""
    body = _section(mapping_text, "### M26 Score-Repellent Monte Carlo")
    assert "Burch" in body or "score-repellent" in body.lower(), "missing theorem anchor"
    for tag in ("m4", "m7"):
        assert tag in body, f"M26 should affect {tag}"


def test_m27_w_flow(mapping_text: str) -> None:
    """M27 W-Flow ODE solver: Lelarasmee 1982 waveform relaxation."""
    body = _section(mapping_text, "### M27 W-Flow ODE solver")
    assert "Lelarasmee" in body or "waveform" in body.lower(), "missing theorem anchor"
    assert "100×" in body or "100x" in body, "expected speedup target 100×"


def test_m28_macroguide_ph(mapping_text: str) -> None:
    """M28 MacroGuide PH: Vietoris-Rips ring-size topology features."""
    body = _section(mapping_text, "### M28 MacroGuide PH")
    assert "Vietoris" in body or "macroguide2026" in body, "missing theorem anchor"
    for tag in ("m7", "m11"):
        assert tag in body, f"M28 should affect {tag}"


# ---------------------------------------------------------------------------
# W2 — M29-M35 (7 mechanisms)
# ---------------------------------------------------------------------------


def test_m29_coordination_aware_gnn(mapping_text: str) -> None:
    """M29 Coordination-Aware GNN: coordination-class gated messages."""
    body = _section(mapping_text, "### M29 Coordination-Aware GNN")
    assert "CoordinationAwareGNNLayer" in body or "coordination-class" in body.lower(), "missing module anchor"
    for tag in ("m3", "m21"):
        assert tag in body, f"M29 should affect {tag}"


def test_m30_quotient_space_diffusion(mapping_text: str) -> None:
    """M30 Quotient-Space Diffusion: SE(3) by construction."""
    body = _section(mapping_text, "### M30 Quotient-Space Diffusion")
    assert "quotient" in body.lower(), "missing mechanism description"
    assert "SE(3)" in body or "SE(3) " in body, "missing SE(3) anchor"
    for tag in ("m3", "m15"):
        assert tag in body, f"M30 should affect {tag}"


def test_m31_quantum_semantics(mapping_text: str) -> None:
    """M31 Quantum semantics: symplectic point per atom (paper §3 only)."""
    body = _section(mapping_text, "### M31 Quantum semantics")
    assert "symplectic" in body.lower() or "麻小南" in body or "Ma X.-N." in body, "missing theorem anchor"
    assert "paper §3" in body, "M31 is paper §3 only (no metric uplift)"


def test_m32_hochschild_cohomology(mapping_text: str) -> None:
    """M32 Hochschild cohomology: β-reduction cochain (paper §3 only)."""
    body = _section(mapping_text, "### M32 Hochschild cohomology")
    assert "Hochschild" in body, "missing mechanism name"
    assert "李思" in body or "Li S." in body, "missing theorem anchor"
    assert "paper §3" in body, "M32 is paper §3 only"


def test_m33_variance_controlled_puct(mapping_text: str) -> None:
    """M33 VarianceControlledPUCT: variance penalty in PUCT."""
    body = _section(mapping_text, "### M33 VarianceControlledPUCT")
    assert "VarianceControlledPUCT" in body, "missing class anchor"
    assert "variance" in body.lower(), "missing mechanism description"
    assert "Burch" in body, "missing theorem anchor"


def test_m34_fractional_ode(mapping_text: str) -> None:
    """M34 Fractional ODE: Caputo derivative, Grünwald-Letnikov."""
    body = _section(mapping_text, "### M34 Fractional ODE")
    assert "Caputo" in body, "missing fractional-order anchor"
    assert "Grünwald" in body or "Grunwald" in body, "missing discretization anchor"


def test_m35_multiph_pll_integration(mapping_text: str) -> None:
    """M35 Multi-PH + PLL signature integration: stacking strict-monotonicity."""
    body = _section(mapping_text, "### M35 Multi-PH + PLL signature integration")
    assert "MultiPHPLL" in body or "multi-PH" in body, "missing module anchor"
    for tag in ("m7", "m11"):
        assert tag in body, f"M35 should affect {tag}"


# ---------------------------------------------------------------------------
# W3 — M36-M44 (9 mechanisms)
# ---------------------------------------------------------------------------


def test_m36_skeleton_jump(mapping_text: str) -> None:
    """M36 Skeleton jump / fragment-swap: Bemis-Murcko FragmentLibrary."""
    body = _section(mapping_text, "### M36 Skeleton jump / fragment-swap")
    assert "FragmentLibrary" in body, "missing module anchor"
    assert "Bemis" in body or "Murcko" in body, "missing theorem anchor"
    for tag in ("m7", "m11"):
        assert tag in body, f"M36 should affect {tag}"


def test_m37_md_relaxation_stub(mapping_text: str) -> None:
    """M37 MD relaxation stub: SimpleForceField LJ-12-6 truncated."""
    body = _section(mapping_text, "### M37 MD relaxation stub")
    assert "SimpleForceField" in body, "missing module anchor"
    assert "UFF" in body or "Rappé" in body or "central-diff" in body, "missing force-field anchor"
    for tag in ("m3", "m15"):
        assert tag in body, f"M37 should affect {tag}"


def test_m38_fg_compat_veto(mapping_text: str) -> None:
    """M38 FG-compat / leave-group veto: 84-cell matrix + VetoAugmentedReducer."""
    body = _section(mapping_text, "### M38 FG-compat / leave-group veto")
    assert "FGCompatMatrix" in body, "missing module anchor"
    assert "84-cell" in body or "84" in body, "missing matrix size anchor"
    for tag in ("m11", "m13"):
        assert tag in body, f"M38 should affect {tag}"


def test_m39_layered_generation(mapping_text: str) -> None:
    """M39 Layered generation: scaffold → attach → decorate 3-stage."""
    body = _section(mapping_text, "### M39 Layered generation")
    assert "ScaffoldStage" in body and "AttachStage" in body and "DecorateStage" in body, (
        "missing 3-stage orchestrator"
    )
    for tag in ("m7", "m11"):
        assert tag in body, f"M39 should affect {tag}"


def test_m40_buyable_reactant_gate(mapping_text: str) -> None:
    """M40 Buyable-reactant gate: AiZynth-as-gate with SMARTS fallback."""
    body = _section(mapping_text, "### M40 Buyable-reactant gate")
    assert "BuyableReactantGate" in body, "missing module anchor"
    assert "AiZynth" in body, "missing oracle anchor"
    assert "m12" in body, "M40 should affect m12 (synthesis rate)"


def test_m41_ood_protocol(mapping_text: str) -> None:
    """M41 OOD protocol: Tanimoto<0.4 + MMseqs2 stub (PARTIAL)."""
    body = _section(mapping_text, "### M41 OOD protocol")
    assert "Tanimoto" in body, "missing Tanimoto anchor"
    assert "MMseqs2" in body, "missing MMseqs2 anchor (even if stub)"
    assert "PARTIAL" in body, "M41 marked as PARTIAL shipping"


def test_m42_non_click_medchem(mapping_text: str) -> None:
    """M42 Non-click med-chem rules: peptide + Boc + reductive amination."""
    body = _section(mapping_text, "### M42 Non-click med-chem rules")
    assert "NonClickMedChemRegistry" in body, "missing module anchor"
    assert "peptide" in body.lower() or "Boc" in body, "missing reaction class"
    for tag in ("m11", "m12"):
        assert tag in body, f"M42 should affect {tag}"


def test_m43_bergman_kernel(mapping_text: str) -> None:
    """M43 Bergman kernel: bounded symmetric domain d-orbital features."""
    body = _section(mapping_text, "### M43 Bergman kernel")
    assert "Bergman" in body, "missing mechanism name"
    assert "Cartan" in body or "R_II" in body, "missing Cartan domain anchor"
    assert "Ma X.-N." in body or "麻小南" in body, "missing theorem anchor"


def test_m44_bass_martingale(mapping_text: str) -> None:
    """M44 Bass martingale: mean-preserving martingale coupling (T1-stable-gated)."""
    body = _section(mapping_text, "### M44 Bass martingale")
    assert "BassMartingale" in body, "missing module anchor"
    assert "Beiglböck" in body or "Beiglbock" in body or "Bass 1997" in body, "missing theorem anchor"
    assert "T1-stable-gated" in body or "T1 stability" in body, "missing gating condition"


# ---------------------------------------------------------------------------
# Cross-mapping tables (3 waves) and §4 regression-debug table
# ---------------------------------------------------------------------------


def test_wave1_cross_mapping_table(mapping_text: str) -> None:
    """W1 cross-mapping table: M22-M28 → affected metrics matrix populated."""
    body = _section(mapping_text, "### R18 Wave 1 cross-mapping")
    for mech in M22_M28_WAVE1:
        assert mech in body, f"{mech} missing from W1 cross-mapping table"


def test_wave2_cross_mapping_table(mapping_text: str) -> None:
    """W2 cross-mapping table: M29-M35 → affected metrics matrix populated."""
    body = _section(mapping_text, "### R18 Wave 2 cross-mapping")
    for mech in M29_M35_WAVE2:
        assert mech in body, f"{mech} missing from W2 cross-mapping table"


def test_wave3_cross_mapping_table(mapping_text: str) -> None:
    """W3 cross-mapping table: M36-M44 → affected metrics matrix populated."""
    body = _section(mapping_text, "### R18 Wave 3 cross-mapping")
    for mech in M36_M44_WAVE3:
        assert mech in body, f"{mech} missing from W3 cross-mapping table"


def test_regression_debug_table(mapping_text: str) -> None:
    """§4 + §12 (Wf-313E) regression-debug tables cover all 20 metrics.

    Each row maps a regression symptom → mechanisms to check.
    Verifies that all 20 metrics from METRICS_TARGETS.md are covered
    by at least one entry across the §4 (R17 baseline) and §12
    (Wf-313E comprehensive) tables.
    """
    body_4 = _section(mapping_text, "## 4. 失败模式诊断")
    body_12 = _section(mapping_text, "## 12. 完整 regression-debug 表")
    covered = set(re.findall(r"\bm\d+\b", body_4)) | set(re.findall(r"\bm\d+\b", body_12))
    missing = [m for m in ALL_METRICS if m not in covered]
    assert not missing, (
        f"§4+§12 regression-debug tables do not cover: {missing}; "
        "per AGENTS.md §1.5 honest framing, every metric needs a debug path"
    )


def test_section_12_status_tags(mapping_text: str) -> None:
    """§12 regression-debug table uses MEASURED/PROJECTED/BLOCKED/HONEST-NEGATIVE tags.

    Per AGENTS.md §1.5 + party-twenty-spirit §3.5, every debug row
    must declare its measurement status honestly.
    """
    body = _section(mapping_text, "## 12. 完整 regression-debug 表")
    for tag in ("MEASURED", "PROJECTED", "BLOCKED", "HONEST-NEGATIVE"):
        assert tag in body, f"§12 must use {tag} tag for honest framing"


def test_section_12_covers_future_metrics(mapping_text: str) -> None:
    """§12 explicitly addresses m18 / m19 / m20 (R20-P3 / P4 / P6 future)."""
    body = _section(mapping_text, "## 12. 完整 regression-debug 表")
    for future_m in ("m18", "m19", "m20"):
        assert future_m in body, f"§12 must cover future metric {future_m}"


def test_wave_regression_diagnostic_tables(mapping_text: str) -> None:
    """Each wave carries its own failure-mode diagnostic table."""
    for anchor in (
        "### R18 Wave 1 失败模式诊断",
        "### R18 Wave 2 失败模式诊断",
        "### R18 Wave 3 失败模式诊断",
    ):
        body = _section(mapping_text, anchor)
        assert "自查命令" in body, f"{anchor} missing self-check commands"


def test_all_23_mechanisms_present(mapping_text: str) -> None:
    """All 23 R18 mechanism entries (M22-M44) appear in the file."""
    for mech in ALL_NEW_MECHANISMS:
        # Each mechanism has a dedicated `### M## Name` subsection.
        pattern = rf"### {re.escape(mech)}\b"
        assert re.search(pattern, mapping_text), (
            f"mechanism {mech} missing dedicated subsection"
        )


def test_pareto_insight_aggregations(mapping_text: str) -> None:
    """Each wave has a '按 metric 聚合' insight table (Pareto 80/20)."""
    for anchor in (
        "### R18 Wave 1 关键 insight",
        "### R18 Wave 2 关键 insight",
        "### R18 Wave 3 关键 insight",
    ):
        body = _section(mapping_text, anchor)
        assert "Pareto" in body or "主要 mechanisms" in body, (
            f"{anchor} missing Pareto aggregation"
        )


# ---------------------------------------------------------------------------
# Honest framing: PROJECTED labels are required (per AGENTS.md §1.5)
# ---------------------------------------------------------------------------


def test_projected_tagging_consistency(mapping_text: str) -> None:
    """GPU-blocked targets carry PROJECTED tag (no aspirational numbers)."""
    # Every new-mechanism subsection body should mention PROJECTED OR
    # `paper §3 only` (M31, M32 are math-only).
    math_only = {"M31", "M32"}
    for mech in ALL_NEW_MECHANISMS:
        pattern = rf"### {re.escape(mech)}\b"
        m = re.search(pattern, mapping_text)
        assert m is not None
        body = _section(mapping_text, m.group(0))
        if mech in math_only:
            assert "paper §3" in body, (
                f"{mech} must be tagged paper §3 only (no GPU metric uplift)"
            )
        else:
            assert "PROJECTED" in body, (
                f"{mech} GPU-blocked target gate must be tagged PROJECTED "
                "per AGENTS.md §1.5 / party-twenty-spirit §3.5"
            )
