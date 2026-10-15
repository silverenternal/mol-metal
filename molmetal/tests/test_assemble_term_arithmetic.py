"""§I.2 AssembleTerm arithmetic_mean regression — Wf-A44 §6.71f.

Per `docs/CFM_WEAKNESSES_2026_10_13.md` §I.2 + `docs/HONEST_NEGATIVE_CATALOG.md` §6.71f:
the ``assemble_term`` honest-negative is the product formulation underflowing to
~5e-9 even at weight=1.0 (12-step diagonal product is order 1e-12). The
arithmetic_mean default at ``molmetal/loss/aggregation_losses.py:47`` already
avoids the underflow — this test pins the arithmetic path so the default never
silently regresses to the product formulation under future refactors.

Test layout (3 tests, all CPU-only, ~1 sec total):
  (a) arithmetic_mean default does NOT underflow — gradient > 1e-6
  (b) product formulation IS the documented underflow (not attempting fix; pins §I.2)
  (c) hinge_tau=-1.0 disables hinge (no clamp)

Per AGENTS.md §5.7: no NEW flag introduced (existing ``assemble_formulation``
arg); smoke verified via direct unit test rather than 50-step trainer run.
"""

from __future__ import annotations

import torch

from molmetal.loss.aggregation_losses import AssembleTerm


def _build_fake_inputs(n_atoms: int = 5, n_edges: int = 8, seed: int = 0):
    """Build (logits_per_edge, edge_index) with deterministic shape."""
    g = torch.Generator().manual_seed(seed)
    # 5 bond classes (0 = no bond, 1..4 = bond types) per edge.
    logits = torch.randn(n_edges, 5, generator=g)
    # edge_index = (2, n_edges) src/dst pairs.
    src = torch.randint(0, n_atoms, (n_edges,), generator=g)
    dst = torch.randint(0, n_atoms, (n_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return logits, edge_index


def test_arithmetic_mean_default_no_underflow() -> None:
    """Default formulation=arithmetic_mean must produce a non-trivial gradient."""
    logits, edge_index = _build_fake_inputs(n_atoms=5, n_edges=8)
    term = AssembleTerm(weight=1.0, enabled=True, formulation="arithmetic_mean")
    val = term.compute(logits, edge_index, n_atoms=5)
    val_abs = float(val.abs())
    # arithmetic_mean of 12 diagonal slices ~ P_bond diagonal mean ~ 1/12 of edge weight.
    # At weight=1.0 we expect |val| ≫ 1e-6 (the §I.2 underflow threshold).
    assert val_abs > 1e-6, (
        f"§I.2 underflow detected on arithmetic_mean path: |val| = {val_abs:.3e} "
        f"< 1e-6 (the §I.2 threshold). arithmetic_mean default at line 47 of "
        f"aggregation_losses.py should not underflow."
    )


def test_product_formulation_documented_underflow() -> None:
    """Product formulation DOES underflow (pins the §I.2 honest-negative)."""
    logits, edge_index = _build_fake_inputs(n_atoms=5, n_edges=8)
    term = AssembleTerm(weight=1.0, enabled=True, formulation="product")
    val = term.compute(logits, edge_index, n_atoms=5)
    val_abs = float(val.abs())
    # product of 12 P_bond diagonal values ≈ (P_bond ~ 0.5)^12 ≈ 2.4e-4 if P_bond
    # is uniform; in practice with softmax over 5 classes on randn logits, P_bond
    # diagonal is ~5e-2 so 12-step product is ~5e-2 ** 12 ≈ 2.4e-21. The CFM_WEAKNESSES
    # §I.2 reports ~5e-9 (likely a softmax-temperature artifact in the shipped
    # trainer). Either way, well below the §I.2 1e-6 threshold. We pin < 1e-6.
    assert val_abs < 1e-6, (
        f"§I.2 product formulation unexpectedly does NOT underflow: |val| = "
        f"{val_abs:.3e}. If this is a real fix, the §I.2 honest-negative in "
        f"CFM_WEAKNESSES should be marked RESOLVED."
    )


def test_hinge_tau_default_disabled() -> None:
    """hinge_tau=-1.0 disables hinge (no clamp); arithmetic_mean path unaffected."""
    logits, edge_index = _build_fake_inputs(n_atoms=5, n_edges=8)
    term_disabled = AssembleTerm(weight=1.0, enabled=True, hinge_tau=-1.0)
    term_enabled = AssembleTerm(weight=1.0, enabled=True, hinge_tau=0.1)
    val_disabled = term_disabled.compute(logits, edge_index, n_atoms=5)
    val_enabled = term_enabled.compute(logits, edge_index, n_atoms=5)
    # Hinge tau=0.1 clamps values BELOW 0.1 to zero (`loss_val = clamp(loss - tau, min=0)`).
    # So the enabled path can only DECREASE or match the disabled path (it cannot
    # increase because the clamp is a one-sided floor at zero). Both values are
    # non-negative (clamp min=0); the enabled path is ≤ the disabled path
    # elementwise (modulo the rng between calls — but `compute` is deterministic
    # for identical inputs, so val_disabled should be ≥ val_enabled).
    assert torch.all(val_enabled <= val_disabled + 1e-6), (
        f"hinge_tau=0.1 should not produce values > hinge_tau=-1.0 path "
        f"(hinge is monotonic non-increasing: it clamps below tau to zero)."
    )
    # Also: with tau=-1.0, no clamp happens. With tau=0.1, the value should
    # either match (if disabled path ≥ 0.1) or be zero (if disabled path < 0.1).
    # Both must be non-negative (clamp min=0).
    assert float(val_disabled) >= 0.0
    assert float(val_enabled) >= 0.0
