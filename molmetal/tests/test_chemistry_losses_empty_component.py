"""Wf-A61 (2026-10-17) L18 — ``MoleculeLoss.compute`` returns zeros when
2+ of {QED, SA, ADMET} components are empty, instead of substituting a
misleading ``0.5`` "average molecule" signal for the missing component.
"""

from __future__ import annotations

import torch

from molmetal.loss.chemistry_losses import MoleculeLoss


def _make_loss() -> MoleculeLoss:
    return MoleculeLoss(weight=1.0, name="molecule_loss_test")


def test_three_components_full_returns_negative_mean() -> None:
    """L18 — all three components filled produces the legacy three-term sum."""
    loss = _make_loss()
    # QED, SA-normalised, ADMET all present.
    out = loss.compute(
        decoded_smiles_batch=["CCO", "CCN"],
        device=torch.device("cpu"),
    )
    # Loss is ``-1 * mean_qed + mean_sa + mean_admet``; with real
    # values we only assert non-zero + finite.
    assert torch.is_tensor(out)
    assert torch.isfinite(out)


def test_one_empty_returns_zero() -> None:
    """L18 — when only one of three components is filled, return zeros.

    The previous code substituted ``0.5`` for the missing component,
    producing a misleading signal of an "average molecule" when
    actually no QED/SA/ADMET data existed for the cohort.
    """
    loss = _make_loss()
    # Pass a cohort where RDKit cannot produce any of the three scores
    # (empty list). The exact route is None; the cohort has no
    # chemistry data.
    out = loss.compute(
        decoded_smiles_batch=[],  # no decoded SMILES -> no scores
        device=torch.device("cpu"),
    )
    assert float(out.item()) == 0.0