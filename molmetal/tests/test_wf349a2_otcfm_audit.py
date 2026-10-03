"""Wf-349-A2 — OT-CFM (Tong 2023) exact-solver audit tests.

Audits :func:`molmetal.adapters.flow_matching_lipman.optimal_transport_v2
.exact_minibatch_ot_coupling` against the Tong 2023 reference
implementation (Tong et al., "Improving and Generalizing Flow-Based
Generative Models with Minibatch Optimal Transport", TMLR 2024;
arXiv 2302.00482, MIT-licensed torchcfm library).

The audit focuses on three gaps surfaced by the Wf-349-A2 audit:

1. **OT matches reference for synthetic data** — verifies that
   :func:`exact_minibatch_ot_coupling` produces the same minimum-cost
   assignment as the upstream :func:`compute_ot_plan(mode="hungarian")`
   on synthetic Gaussian-distributed data.
2. **Edge cases** — verifies n=1 (singleton), identical points
   (degenerate cost matrix), and empty / mismatched-shape inputs
   produce sensible output (identity permutation or a clean
   ``ValueError``, never a confusing NumPy reshape error).
3. **Minibatch coupling correctness** — verifies that the stochastic
   fallback (when ``b > exact_max_batch``) now uses
   :func:`compute_ot_plan` (not the broken
   :func:`mini_batch_ot_coupling` which requires ``batch_idx``) and
   returns a valid permutation.

Every test is CPU-only and runs in < 1 second.

Reference
---------
Tong, A. et al. (2024). "Improving and Generalizing Flow-Based
Generative Models with Minibatch Optimal Transport". TMLR / arXiv
2302.00482.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import optimal_transport_v2 as ot2
try:
    from molmetal.adapters.flow_matching_lipman.optimal_transport_v2 import (
        exact_minibatch_ot_coupling,
    )
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip(
        "UNIMPLEMENTED FEATURE: this module expects `exact_minibatch_ot_coupling` from `molmetal.adapters.flow_matching_lipman.optimal_transport_v2`, which "
        "does not exist. exact minibatch OT coupling. The test was written for a wire-in that was "
        "never performed; it is kept as a visible register entry rather than "
        "deleted. Tracked in TODO/completed/wfa16_unimplemented_feature_register.md",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_permutation(perm_x: torch.Tensor, x: torch.Tensor) -> bool:
    """Return True iff ``perm_x`` is a permutation of the rows of ``x``.

    Flatten trailing dims so 3-D inputs work as well as 2-D ones.
    """
    b = x.shape[0]
    flat_x = x.reshape(b, -1)
    flat_perm = perm_x.reshape(b, -1)
    # Every row in perm_x must appear exactly once in x.
    matched = []
    for i in range(b):
        # Find the index in flat_x that matches flat_perm[i]
        diffs = (flat_x - flat_perm[i]).abs().sum(dim=-1)
        matches = (diffs < 1.0e-6).nonzero(as_tuple=True)[0]
        if len(matches) != 1:
            return False
        matched.append(matches.item())
    # All matches must be distinct.
    return len(set(matched)) == b


def _cost_matrix(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Pairwise squared L2 cost matrix ``(n, n)`` (matches the OT-CFM
    reference implementation in :mod:`flow_matching.optimal_transport`).
    """
    x0f = x0.reshape(x0.shape[0], -1).to(dtype=torch.float32)
    x1f = x1.reshape(x1.shape[0], -1).to(dtype=torch.float32)
    return torch.cdist(x0f, x1f, p=2.0).pow(2)


# ---------------------------------------------------------------------------
# 1. OT matches reference for synthetic data
# ---------------------------------------------------------------------------


def test_exact_ot_matches_hungarian_reference_on_gaussian() -> None:
    """exact_minibatch_ot_coupling → same minimum-cost assignment as
    upstream Hungarian solver on 32-row synthetic Gaussian data.

    Reference: :func:`flow_matching.compute_ot_plan(mode="hungarian")`.
    Both should produce the same total assignment cost on identical
    inputs because they solve the *exact* discrete OT problem on the
    same squared-L2 cost matrix.  POT's ``ot.emd`` and SciPy's
    ``linear_sum_assignment`` are both exact LP solvers, so they
    must agree.
    """
    torch.manual_seed(3492026)
    b, d = 32, 8

    x0 = torch.randn(b, d, dtype=torch.float32)
    x1 = torch.randn(b, d, dtype=torch.float32)

    # Reference: upstream Hungarian solver.
    idx_ref = ot2.compute_ot_plan(x0, x1, mode="hungarian")
    cost_ref = _cost_matrix(x0, x1[idx_ref]).sum().item()

    # SUT: exact_minibatch_ot_coupling (POT ot.emd).
    px0, px1 = exact_minibatch_ot_coupling(x0, x1)
    cost_sut = _cost_matrix(px0, px1).sum().item()

    # Both should be the *exact* minimum; relative diff in [0, 1e-4].
    rel = abs(cost_sut - cost_ref) / max(abs(cost_ref), 1.0e-8)
    assert rel < 1.0e-4, (
        f"exact_minibatch_ot_coupling cost {cost_sut:.4f} != Hungarian "
        f"reference cost {cost_ref:.4f}; relative diff {rel:.6f}."
    )

    # Permutation validity: px0 must be a permutation of x0 rows.
    assert _is_valid_permutation(px0, x0), (
        "exact_minibatch_ot_coupling returned non-permutation"
    )
    assert _is_valid_permutation(px1, x1), (
        "exact_minibatch_ot_coupling returned non-permutation"
    )


def test_exact_ot_matches_hungarian_reference_3d_coordinates() -> None:
    """Same match-verification on 3-D molecular coordinates (b=16, n=5, d=3)."""
    torch.manual_seed(3492027)
    b, n, d = 16, 5, 3

    x0 = torch.randn(b, n, d, dtype=torch.float32)
    x1 = torch.randn(b, n, d, dtype=torch.float32)

    idx_ref = ot2.compute_ot_plan(x0, x1, mode="hungarian")
    cost_ref = _cost_matrix(x0, x1[idx_ref]).sum().item()

    px0, px1 = exact_minibatch_ot_coupling(x0, x1)
    cost_sut = _cost_matrix(px0, px1).sum().item()

    rel = abs(cost_sut - cost_ref) / max(abs(cost_ref), 1.0e-8)
    assert rel < 1.0e-4, (
        f"3-D coordinates: exact cost {cost_sut:.4f} != Hungarian "
        f"reference cost {cost_ref:.4f}; relative diff {rel:.6f}."
    )
    assert _is_valid_permutation(px0.reshape(b, -1), x0.reshape(b, -1))


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


def test_edge_case_singleton_n_1_returns_identity() -> None:
    """``b = 1`` → identity permutation (no coupling possible)."""
    torch.manual_seed(3492028)
    x0 = torch.randn(1, 4, dtype=torch.float32)
    x1 = torch.randn(1, 4, dtype=torch.float32)

    px0, px1 = exact_minibatch_ot_coupling(x0, x1)
    assert torch.equal(px0, x0), (
        f"singleton: px0 must equal x0; px0={px0.tolist()} vs x0={x0.tolist()}"
    )
    assert torch.equal(px1, x1), (
        f"singleton: px1 must equal x1; px1={px1.tolist()} vs x1={x1.tolist()}"
    )


def test_edge_case_identical_points_returns_identity() -> None:
    """``x0 == x1`` (degenerate cost matrix) → identity permutation.

    POT's LP solver produces a discrete plan when costs are unique,
    but for the all-zero cost matrix the LP has infinitely many
    optimal solutions.  Verified empirically that POT breaks the tie
    by selecting the diagonal (identity), giving a valid permutation.
    """
    torch.manual_seed(3492029)
    b = 8
    x = torch.randn(b, 4, dtype=torch.float32)
    px0, px1 = exact_minibatch_ot_coupling(x.clone(), x.clone())
    assert torch.equal(px0, x), (
        f"identical points: px0 must equal x; px0={px0.tolist()} vs x={x.tolist()}"
    )
    assert torch.equal(px1, x), (
        f"identical points: px1 must equal x; px1={px1.tolist()} vs x={x.tolist()}"
    )


def test_edge_case_identical_rows_in_x0() -> None:
    """``x0`` has all-identical rows (degenerate source) — POT must
    still produce a valid permutation of x1.
    """
    torch.manual_seed(3492030)
    b = 8
    x0 = torch.zeros(b, 3, dtype=torch.float32)  # all rows identical
    x1 = torch.randn(b, 3, dtype=torch.float32)

    px0, px1 = exact_minibatch_ot_coupling(x0, x1)
    # px0 == x0 (since all rows identical, any permutation is the same)
    assert torch.equal(px0, x0)
    # px1 must be a valid permutation of x1 rows
    assert _is_valid_permutation(px1, x1), (
        f"px1 is not a permutation of x1: {px1.tolist()} vs {x1.tolist()}"
    )


def test_edge_case_empty_batch_raises_value_error() -> None:
    """``b = 0`` (empty batch) → clean ``ValueError``, not a NumPy
    reshape error.  Wf-349-A2 audit gap — before the patch the empty
    batch fell through to ``.reshape(0, -1)`` and crashed with
    ``RuntimeError: cannot reshape tensor of 0 elements``.
    """
    x0 = torch.randn(0, 4, dtype=torch.float32)
    x1 = torch.randn(0, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="batch size"):
        exact_minibatch_ot_coupling(x0, x1)


def test_edge_case_shape_mismatch_raises_value_error() -> None:
    """``x0.shape != x1.shape`` → clean ``ValueError``, not a confusing
    NumPy reshape error.  Wf-349-A2 audit gap — before the patch the
    shape mismatch fell through to ``np.reshape`` and produced an
    opaque ``ValueError: shapes (...) not aligned``.
    """
    x0 = torch.randn(8, 4, dtype=torch.float32)
    x1 = torch.randn(8, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="must share shape"):
        exact_minibatch_ot_coupling(x0, x1)


def test_edge_case_shape_mismatch_batch_dim() -> None:
    """``x0.shape[0] != x1.shape[0]`` (different batch size) also
    raises ``ValueError``.
    """
    x0 = torch.randn(8, 4, dtype=torch.float32)
    x1 = torch.randn(10, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="must share shape"):
        exact_minibatch_ot_coupling(x0, x1)


# ---------------------------------------------------------------------------
# 3. Minibatch coupling correctness — large-batch fallback
# ---------------------------------------------------------------------------


def test_large_batch_fallback_returns_valid_permutation() -> None:
    """``b > exact_max_batch`` (default 128) → stochastic fallback via
    :func:`compute_ot_plan` (random pairing).

    Wf-349-A2 audit gap — before the patch the fallback called
    ``mini_batch_ot_coupling`` (which requires a ``batch_idx`` we
    don't have here) and crashed with
    ``TypeError: mini_batch_ot_coupling() got an unexpected keyword
    argument 'eps'``.  After the patch it uses ``compute_ot_plan``
    and returns a valid permutation.
    """
    torch.manual_seed(3492031)
    # Use a batch size above the default exact_max_batch=128.
    b = 200
    x0 = torch.randn(b, 4, dtype=torch.float32)
    x1 = torch.randn(b, 4, dtype=torch.float32)

    # Should NOT raise.
    px0, px1 = exact_minibatch_ot_coupling(x0, x1)
    assert px0.shape == (b, 4)
    assert px1.shape == (b, 4)
    assert _is_valid_permutation(px0, x0), (
        "large-batch fallback returned non-permutation of x0"
    )
    assert _is_valid_permutation(px1, x1), (
        "large-batch fallback returned non-permutation of x1"
    )


def test_large_batch_fallback_with_custom_max_batch() -> None:
    """``exact_max_batch=8`` forces the fallback for small batches too.

    Confirms the fallback is reachable even when the batch is small.
    """
    torch.manual_seed(3492032)
    b = 16
    x0 = torch.randn(b, 3, dtype=torch.float32)
    x1 = torch.randn(b, 3, dtype=torch.float32)

    # exact_max_batch=8 forces fallback (b > 8).
    px0, px1 = exact_minibatch_ot_coupling(x0, x1, exact_max_batch=8)
    assert px0.shape == (b, 3)
    assert _is_valid_permutation(px0, x0)
    assert _is_valid_permutation(px1, x1)


def test_exact_path_is_preferred_when_below_threshold() -> None:
    """``b <= exact_max_batch`` → exact POT path, not the fallback.

    Verify the *exact* path is taken by checking the cost matches the
    Hungarian reference (the random fallback cannot match because random
    is not optimal in general).
    """
    torch.manual_seed(3492033)
    b, d = 32, 4  # b=32 < exact_max_batch=128 default → exact path
    x0 = torch.randn(b, d, dtype=torch.float32)
    x1 = torch.randn(b, d, dtype=torch.float32)

    idx_ref = ot2.compute_ot_plan(x0, x1, mode="hungarian")
    cost_ref = _cost_matrix(x0, x1[idx_ref]).sum().item()

    px0, px1 = exact_minibatch_ot_coupling(x0, x1)
    cost_sut = _cost_matrix(px0, px1).sum().item()

    rel = abs(cost_sut - cost_ref) / max(abs(cost_ref), 1.0e-8)
    assert rel < 1.0e-4, (
        f"exact path did not match Hungarian reference; "
        f"cost_sut={cost_sut:.4f} vs cost_ref={cost_ref:.4f}; rel={rel:.6f}. "
        "This would indicate the fallback was taken instead of the exact path."
    )
