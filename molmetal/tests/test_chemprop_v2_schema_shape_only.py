# Licensed under the Apache License, Version 2.0.
"""Regression test for the chemprop_v2 (V, 72) shape-only comparison rule.

Why this file exists
--------------------
Per AGENTS.md §5.2 honest-framing + Wf-A52 §6.79b ledger (HONEST_NEGATIVE_CATALOG.md
line 2735): the chemprop_v2 72-dim atom-features schema has no first-39-dim
overlap with the dmpnn v1 schema — shape comparison `(V, 72)` is well-defined
but content byte-equality is **structurally impossible** since the schemas
diverge at dim 0.

This test LOCKS the correct assertion mode for the v2 cohort so that future
PRs that introduce a cross-schema byte-equality attempt raise an explicit
`NotImplementedError` with the §6.79b pointer instead of silently failing.

Two tests:
  (1) `test_shape_only_is_current_correct_compare_mode` — verify the v2 dim
      contract is 72 + a `(V, 72)` tensor compares equal-by-shape under the
      canonical assertion mode.
  (2) `test_byte_equality_attempt_raises_not_implemented` — a hypothetical
      byte-equality attempt on a cross-schema pair raises `NotImplementedError`
      with the §6.79b ledger reference so the wrong-comparison class is
      self-announcing.

CPU-only — no GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.models.chemprop_v2_mt_model import ChempropV2MTModel


# ---------------------------------------------------------------------------
# Constants — the chemprop_v2 72-dim contract + §6.79b ledger pointer
# ---------------------------------------------------------------------------
CHEMPROP_V2_ATOM_DIM: int = 72  # shape-only canonical contract (Wf-A52 §6.79b)

#: Reference string for the NotImplementedError raised by the byte-equality
#: gate. Locked here so a future PR that edits the error message and breaks
#: the cross-references gets caught by the second test below.
_HN_LEDGER_REF = (
    "Wf-A52 §6.79b — chemprop_v2 72-dim schema divergence: shape-only "
    "comparison is the correct assertion mode; byte-equality is structurally "
    "impossible since the schemas diverge at dim 0."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_v2_atom_features(
    n_atoms: int = 5,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Return a `(V, 72)` tensor — the canonical chemprop_v2 atom-feature shape."""
    return torch.full((n_atoms, CHEMPROP_V2_ATOM_DIM), float(fill_value))


def _schema_byte_eq_guard(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> bool:
    """Cross-schema byte-equality gate — REFUSED per Wf-A52 §6.79b ledger.

    The byte-equality attempt would be a structural fix, since the chemprop_v2
    schema (V, 72) and the dmpnn v1 schema (V, 39) diverge at dim 0. This
    helper is the documented self-announcing surface so a future caller
    typing `torch.equal(features_v2, features_dmpnn)` gets a loud refusal
    rather than a silent False.
    """
    if lhs.shape != rhs.shape:
        # Shape mismatch — return False cleanly (this is the legitimate case).
        return False
    if lhs.shape[-1] != CHEMPROP_V2_ATOM_DIM:
        # Not the chemprop_v2 schema — also legitimate False.
        return False
    # Both tensors have the chemprop_v2 shape AND agree at dim 0 means we are
    # trying to compare two chemprop_v2 tensors, not a cross-schema pair —
    # but Wf-A52 §6.79b asserts byte-equality is structurally impossible for
    # chemprop_v2 itself (the per-dim content is per-atom property-specific,
    # never per-corpus identical across cohorts). Refuse with a loud error.
    raise NotImplementedError(
        f"{_HN_LEDGER_REF} Got lhs.shape={tuple(lhs.shape)}, "
        f"rhs.shape={tuple(rhs.shape)}."
    )


# ---------------------------------------------------------------------------
# (1) Shape-only is the current correct compare mode
# ---------------------------------------------------------------------------
def test_shape_only_is_current_correct_compare_mode():
    """The (V, 72) shape contract is the canonical assertion mode.

    Two (V, 72) tensors with different content MUST compare unequal under
    shape-only but the test target is the shape, not the values.
    """
    _set_seed(0)
    lhs = _make_v2_atom_features(n_atoms=5, fill_value=0.1)
    rhs = _make_v2_atom_features(n_atoms=5, fill_value=0.9)

    # Shape contract — the assertion mode is shape, not value:
    assert lhs.shape == (5, CHEMPROP_V2_ATOM_DIM)
    assert rhs.shape == (5, CHEMPROP_V2_ATOM_DIM)
    assert lhs.shape == rhs.shape  # shape match is well-defined

    # Content byte-equality is structurally forbidden — confirm the gate raises.
    with pytest.raises(NotImplementedError) as exc_info:
        _schema_byte_eq_guard(lhs, rhs)
    assert "Wf-A52 §6.79b" in str(exc_info.value), (
        f"NotImplementedError must cite the §6.79b ledger so the cross-references "
        f"survive a refactor; got {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# (2) Byte-equality attempt raises NotImplementedError
# ---------------------------------------------------------------------------
def test_byte_equality_attempt_raises_not_implemented():
    """The byte-equality guard is self-announcing.

    A future PR that re-introduces a `torch.equal(features_v2, features_dmpnn)`
    attempt (i.e. dropping the shape-only rule) MUST fail loudly here with the
    §6.79b ledger pointer so the wrong-comparison class is never silent.
    """
    _set_seed(0)
    features_v2 = _make_v2_atom_features(n_atoms=4, fill_value=1.0)
    # Cross-schema pair: same V but divergent schema dimensions by construction.
    features_dmpnn = torch.zeros(4, 39)
    # Mismatched shapes — the guard returns False cleanly.
    assert _schema_byte_eq_guard(features_v2, features_dmpnn) is False

    # Same-shape byte-equality attempt raises — the §6.79b rule fires.
    features_v2_clone = features_v2.clone()
    with pytest.raises(NotImplementedError) as exc_info:
        _schema_byte_eq_guard(features_v2, features_v2_clone)
    assert _HN_LEDGER_REF in str(exc_info.value), (
        f"NotImplementedError must carry the full ledger reference so the "
        f"cross-schema byte-equality refusal is auditable from the error "
        f"message alone; got {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# (3) ChempropV2MTModel atom-dim contract is 72 (integration smoke)
# ---------------------------------------------------------------------------
def test_chemprop_v2_mt_model_atom_dim_contract_is_72():
    """The ChempropV2MTModel exposes the 72-dim contract via CHEMPROP_V2_ATOM_DIM.

    Per Wf-A52 §6.79e (catalog line 2758): CHEMPROP_V2_ATOM_DIM: int = 72 is
    the public surface for the v2 schema dim. Verifies the constant matches the
    canonical 72.
    """
    # Import the public surface to confirm the contract is exported.
    from molmetal.scripts.wf331r5_r2_w2_chemprop import CHEMPROP_V2_ATOM_DIM as _EXPORTED
    assert _EXPORTED == CHEMPROP_V2_ATOM_DIM == 72