"""Wf-A73 (2026-10-18) — Operator 1 (per-class logit adjustment, Menon 2020)
regression tests.

Background: 95% of candidate pairs are BOND_NO_BOND in the 5-metal cohort.
Vanilla CE is minimised by predicting 'no bond' almost everywhere. The
existing `bond_class_weights` (inverse-frequency power 0.5) is a weaker
rebalancer. The principled upgrade is **per-class logit adjustment**:
add ``tau * log(pi_c)`` to the logit before CE so the loss sees the
OPTIMAL decision boundary under known class prior (Menon, Jayasumana,
Rawat et al. ICLR 2021, arXiv:2007.07314).

The 5 tests below cover:
  1. Identity / no-op paths
  2. The textbook tau=1 case on a 95/5 prior
  3. Amplification at tau>1
  4. tau<1 attenuation
  5. Gradient flow + numerical sanity
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _load_trainer():
    """Load the trainer module fresh (it has heavy imports)."""
    spec = importlib.util.spec_from_file_location("_t_wfa73", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_wfa73"] = mod
    spec.loader.exec_module(mod)
    return mod


# Load once at module import (test session is short).
_TRAINER = _load_trainer()
_ADJUST = _TRAINER._apply_logit_adjustment


# 5-class empirical prior matching the Wf-A70 W3 smoke
# (95% NO_BOND + 1.5% SINGLE + 0.5% DOUBLE + 0.04% TRIPLE + 2.96% AROMATIC
#  from results/cfm_5metal_wfa70_w3_smoke; rounded to 4 decimals).
_PRIOR_5METAL = torch.tensor([0.9500, 0.0150, 0.0050, 0.0004, 0.0296])
_LOGITS_BATCH = torch.tensor([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [-0.5, 0.0, 0.5, 1.0, 1.5],
])


# ---------------------------------------------------------------------------
# Test 1 — identity / no-op paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tau, prior",
    [
        (0.0, _PRIOR_5METAL),       # tau=0: explicit no-op
        (1.0, None),                # prior=None: implicit no-op
        (0.5, None),                # prior=None: implicit no-op
    ],
)
def test_apply_logit_adjustment_identity(tau: float, prior) -> None:
    """tau=0 OR prior=None must return the input logits unchanged (no
    allocation, no dtype change, no device move)."""
    out = _ADJUST(_LOGITS_BATCH, prior, tau)
    assert torch.equal(out, _LOGITS_BATCH), (
        f"WF-A73 Op1: _apply_logit_adjustment must be a no-op when "
        f"tau=0 OR prior is None.  tau={tau}, prior is None: {prior is None}."
    )


# ---------------------------------------------------------------------------
# Test 2 — the textbook tau=1 case on a 95/5 prior
# ---------------------------------------------------------------------------
def test_apply_logit_adjustment_textbook_tau_1() -> None:
    """At tau=1, the per-class logit shift is exactly ``log(pi_c)``;
    the minority classes get a large NEGATIVE push (because log(small)
    is a large negative number), so the head has to overcome the
    prior before predicting a minority class.  This is what makes
    Menon 2020 the OPTIMAL decision boundary under known prior.

    We verify the closed-form values (no torch ops in the assert path,
    so the test is a pure arithmetic check).
    """
    out = _ADJUST(_LOGITS_BATCH, _PRIOR_5METAL, tau=1.0)
    expected_shift = _PRIOR_5METAL.log()  # (5,)
    expected = _LOGITS_BATCH + expected_shift  # (2, 5)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"WF-A73 Op1: tau=1 shift must equal log(pi_c).  "
        f"max abs diff: {(out - expected).abs().max().item():.6f}"
    )
    # Specific: NO_BOND prior=0.95 → shift -0.0513 (small)
    # TRIPLE prior=0.0004 → shift -7.824 (large negative push)
    no_bond_shift = float(_PRIOR_5METAL.log()[0])
    triple_shift = float(_PRIOR_5METAL.log()[3])
    assert -0.06 < no_bond_shift < -0.04, (
        f"NO_BOND log-prior should be ~-0.051, got {no_bond_shift:.4f}"
    )
    assert -8.0 < triple_shift < -7.5, (
        f"TRIPLE log-prior should be ~-7.82, got {triple_shift:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — amplification at tau>1
# ---------------------------------------------------------------------------
def test_apply_logit_adjustment_tau_amplifies() -> None:
    """tau>1 multiplies the per-class shift; the NO_BOND vs minority
    gap widens monotonically with tau."""
    out_tau_1 = _ADJUST(_LOGITS_BATCH, _PRIOR_5METAL, tau=1.0)
    out_tau_2 = _ADJUST(_LOGITS_BATCH, _PRIOR_5METAL, tau=2.0)
    out_tau_3 = _ADJUST(_LOGITS_BATCH, _PRIOR_5METAL, tau=3.0)
    # NO_BOND (col 0) vs TRIPLE (col 3) gap is monotone in tau
    gap_1 = (out_tau_1[:, 0] - out_tau_1[:, 3]).mean().item()
    gap_2 = (out_tau_2[:, 0] - out_tau_2[:, 3]).mean().item()
    gap_3 = (out_tau_3[:, 0] - out_tau_3[:, 3]).mean().item()
    assert gap_1 < gap_2 < gap_3, (
        f"WF-A73 Op1: NO_BOND-vs-TRIPLE gap must grow monotonically in tau. "
        f"got gap(tau=1)={gap_1:.3f}, gap(tau=2)={gap_2:.3f}, gap(tau=3)={gap_3:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — attenuation at tau<1
# ---------------------------------------------------------------------------
def test_apply_logit_adjustment_tau_attenuates() -> None:
    """tau<1 brings the head closer to the unadjusted (vanilla CE)
    operating point.  The minority-vs-NO_BOND gap shrinks toward the
    input-logit gap as tau -> 0.  Small tau < large tau < (vanilla CE)
    is the expected ordering: |gap(tau=0.1) - gap(input)| is the
    smallest (most attenuated)."""
    out_tau_0_5 = _ADJUST(_LOGITS_BATCH, _PRIOR_5METAL, tau=0.5)
    out_tau_0_1 = _ADJUST(_LOGITS_BATCH, _PRIOR_5METAL, tau=0.1)
    # The input-logit gap (col 0 - col 3) across the 2-row batch
    out_tau_0 = _LOGITS_BATCH[:, 0] - _LOGITS_BATCH[:, 3]
    gap_0 = out_tau_0.mean().item()
    gap_0_5 = (out_tau_0_5[:, 0] - out_tau_0_5[:, 3]).mean().item()
    gap_0_1 = (out_tau_0_1[:, 0] - out_tau_0_1[:, 3]).mean().item()
    # As tau -> 0, the output gap -> input-logit gap
    assert abs(gap_0_1 - gap_0) < abs(gap_0_5 - gap_0), (
        f"WF-A73 Op1: smaller tau must attenuate the prior shift closer to "
        f"the input-logit gap ({gap_0:.3f}).  got gap(tau=0.1)={gap_0_1:.3f} "
        f"(delta={abs(gap_0_1 - gap_0):.3f}), gap(tau=0.5)={gap_0_5:.3f} "
        f"(delta={abs(gap_0_5 - gap_0):.3f}).  The tau=0.1 delta should be "
        f"the smaller of the two."
    )


# ---------------------------------------------------------------------------
# Test 5 — gradient flow + numerical sanity
# ---------------------------------------------------------------------------
def test_apply_logit_adjustment_gradient_flow() -> None:
    """The shift must be differentiable (the loss uses it directly) and
    numerically stable for realistic pi_c in [1e-4, 1.0].  The clamp at
    1e-12 inside the helper defends against future cohort changes where
    a class is missing entirely."""
    logits_g = _LOGITS_BATCH.clone().requires_grad_(True)
    out = _ADJUST(logits_g, _PRIOR_5METAL, tau=1.0)
    loss = out.sum()
    loss.backward()
    assert logits_g.grad is not None, "gradient must flow through _apply_logit_adjustment"
    assert torch.isfinite(logits_g.grad).all(), (
        f"gradient must be finite; got min={logits_g.grad.min().item():.3e}, "
        f"max={logits_g.grad.max().item():.3e}"
    )
    # Gradient w.r.t. logits is the IDENTITY matrix (sum of constants has
    # gradient 1 w.r.t. each input element).  The shift is constant in
    # the logits, so d/dlogits[shift] = 1.
    expected_grad = torch.ones_like(_LOGITS_BATCH)
    assert torch.allclose(logits_g.grad, expected_grad, atol=1e-5), (
        f"WF-A73 Op1: gradient w.r.t. logits must be identity (shift is "
        f"constant in logits).  max abs diff: "
        f"{(logits_g.grad - expected_grad).abs().max().item():.6f}"
    )

    # Defensive: pi_c = 0.0 should not blow up (clamp at 1e-12)
    prior_degenerate = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    out_degenerate = _ADJUST(_LOGITS_BATCH, prior_degenerate, tau=1.0)
    assert torch.isfinite(out_degenerate).all(), (
        f"WF-A73 Op1: degenerate prior (single class) must remain finite. "
        f"max abs: {out_degenerate.abs().max().item():.3e}"
    )
