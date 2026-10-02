"""Wf-424-A3 — scaffold_conditioning 2D auto-detection tests.

=====================================================================
Closes the Wf-396-A3 CRITICAL FINDING: the function ``scaffold_condition``
in ``molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py``
previously raised ``ValueError`` immediately when called with 2D
``(E, V)`` logits, which is the shape the Wf-388-A2 trainer passes via
``all_logits.view(-1, V)`` when ``--scaffold-conditioning`` is True
and the loss is bond-CE on a ``(B·N, V)`` tensor.

The fix (per Wf-424-A3) auto-detects 2D inputs and reshapes them to
``(B, 1, V)`` internally so the per-sample bias broadcasting logic
is uniform with the canonical 3D ``(B, N, V)`` path.  The 2D shape is
preserved on return (``squeeze(1)`` on output).

Constraints (per AGENTS.md §5 + party-twenty-spirit §3.5):
* CPU-only (no GPU launches).
* No ``paper/*`` touches (TODO-84).
* No wet-lab planning (§10).
* No new pre-flight gates added (test-only changes).

Test contracts (3 REQUIRED by Wf-424-A3 brief + 4 EXTRAS for full
shape parity coverage):

1. ``test_3d_input`` — 3D ``(B, N, V)`` logits + ``(B,)`` scaffold_idx
   + ``(n_scaffolds, V)`` bias produces a 3D output with the bias
   added to each (B, :) slice.

2. ``test_2d_input_reshaped`` — 2D ``(B, V)`` logits + ``(B,)`` scaffold_idx
   + ``(n_scaffolds, V)`` bias produces a 2D output with the bias
   added along the vocab dim (the canonical Wf-388-A2 trainer
   invocation path).

3. ``test_error_clear`` — 4D and 1D inputs both raise ``ValueError``
   with informative messages.

4. ``test_2d_identity_path`` — when ``scaffold_bias is None`` the
   function returns the 2D input logits bit-exact unchanged.

5. ``test_3d_identity_path`` — same identity contract for 3D inputs.

6. ``test_2d_per_sample_bias_broadcast`` — verify that the per-sample
   bias actually selects per-row when scaffold_idx has multiple
   distinct values.

8. ``test_scaffold_idx_shape_mismatch_raises`` — 2D logits with
   wrong-shape scaffold_idx raises ``ValueError``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from molmetal.adapters.flow_matching_lipman import scaffold_conditioning as _sc


# ---------------------------------------------------------------------------
# 3 REQUIRED tests (per Wf-424-A3 brief)
# ---------------------------------------------------------------------------
def test_3d_input() -> None:
    """Canonical 3D ``(B, N, V)`` path — bias broadcasts over N."""
    torch = pytest.importorskip("torch")

    B, N, V, n_scaffolds = 4, 3, 5, 16
    logits = torch.randn(B, N, V)
    scaffold_idx = torch.zeros(B, dtype=torch.long)
    scaffold_bias = torch.zeros(n_scaffolds, V)
    # Mark column 2 for scaffold_idx == 0
    scaffold_bias[0, 2] = 1.0

    out = _sc.scaffold_condition(
        logits, scaffold_idx, n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
    )

    # Shape preserved
    assert out.shape == (B, N, V), f"shape mismatch: {tuple(out.shape)}"
    # Bias applied to column 2 of every (B, :) slice for samples with idx=0
    expected = logits.clone()
    expected[:, :, 2] += 1.0
    assert torch.allclose(out, expected), "3D bias broadcast incorrect"


def test_2d_input_reshaped() -> None:
    """2D ``(B, V)`` path — what the Wf-388-A2 trainer passes.

    This is the canonical Wf-424-A3 fix: a 2D logits tensor
    (the flattened token-stream shape used by ``bond-CE``) must NOT
    raise ``ValueError``; it must auto-reshape to ``(B, 1, V)``
    internally, apply the per-sample bias, and squeeze back to
    ``(B, V)`` on return.
    """
    torch = pytest.importorskip("torch")

    B, V, n_scaffolds = 8, 5, 256
    logits = torch.randn(B, V)
    scaffold_idx = torch.zeros(B, dtype=torch.long)
    scaffold_bias = torch.zeros(n_scaffolds, V)
    scaffold_bias[0, 0] = 1.0  # only column 0 affected for samples with idx=0

    out = _sc.scaffold_condition(
        logits, scaffold_idx, n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
    )

    # Shape preserved (2D → 2D)
    assert out.shape == (B, V), (
        f"2D auto-detect failed: input {tuple(logits.shape)} -> "
        f"output {tuple(out.shape)} (expected (B, V))"
    )
    # Bias applied to column 0 of every row (all samples have idx=0)
    expected = logits.clone()
    expected[:, 0] += 1.0
    assert torch.allclose(out, expected), "2D bias broadcast incorrect"


def test_error_clear() -> None:
    """4D and 1D inputs both raise ``ValueError`` with clear messages."""
    torch = pytest.importorskip("torch")

    n_scaffolds, V = 16, 5
    scaffold_bias = torch.zeros(n_scaffolds, V)

    # 4D input
    logits_4d = torch.randn(2, 3, 4, V)
    with pytest.raises(ValueError) as exc_info:
        _sc.scaffold_condition(
            logits_4d, torch.zeros(2, dtype=torch.long),
            n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
        )
    err_4d = str(exc_info.value)
    assert "(B, V)" in err_4d or "(B, N, V)" in err_4d, (
        f"4D error message should mention accepted shapes: {err_4d!r}"
    )

    # 1D input
    logits_1d = torch.randn(V)
    with pytest.raises(ValueError) as exc_info:
        _sc.scaffold_condition(
            logits_1d, torch.zeros(1, dtype=torch.long),
            n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
        )
    err_1d = str(exc_info.value)
    assert "(B, V)" in err_1d or "(B, N, V)" in err_1d, (
        f"1D error message should mention accepted shapes: {err_1d!r}"
    )


# ---------------------------------------------------------------------------
# 4 EXTRAS — full shape parity coverage (Wf-424-A3 docstring promises)
# ---------------------------------------------------------------------------
def test_2d_identity_path() -> None:
    """``scaffold_bias=None`` returns 2D logits bit-exact unchanged."""
    torch = pytest.importorskip("torch")

    B, V = 8, 5
    logits = torch.randn(B, V)
    scaffold_idx = torch.zeros(B, dtype=torch.long)
    out = _sc.scaffold_condition(
        logits, scaffold_idx, n_scaffolds=256, scaffold_bias=None,
    )
    assert out.shape == (B, V)
    assert torch.equal(out, logits), "identity path not bit-exact for 2D"


def test_3d_identity_path() -> None:
    """``scaffold_bias=None`` returns 3D logits bit-exact unchanged."""
    torch = pytest.importorskip("torch")

    B, N, V = 4, 3, 5
    logits = torch.randn(B, N, V)
    scaffold_idx = torch.zeros(B, dtype=torch.long)
    out = _sc.scaffold_condition(
        logits, scaffold_idx, n_scaffolds=256, scaffold_bias=None,
    )
    assert out.shape == (B, N, V)
    assert torch.equal(out, logits), "identity path not bit-exact for 3D"


def test_2d_per_sample_bias_broadcast() -> None:
    """Per-row bias selection when scaffold_idx has multiple distinct values.

    Each row's vocab gets the bias of the scaffold at that row's
    ``scaffold_idx``.  This is the canonical use case of the
    per-batch scaffold conditioning.
    """
    torch = pytest.importorskip("torch")

    B, V, n_scaffolds = 4, 3, 16
    logits = torch.zeros(B, V)
    scaffold_bias = torch.zeros(n_scaffolds, V)
    # Distinguish scaffolds: scaffold 0 has [1,0,0], scaffold 1 has [0,1,0],
    # scaffold 2 has [0,0,1], scaffold 3 has [1,1,1]
    scaffold_bias[0] = torch.tensor([1.0, 0.0, 0.0])
    scaffold_bias[1] = torch.tensor([0.0, 1.0, 0.0])
    scaffold_bias[2] = torch.tensor([0.0, 0.0, 1.0])
    scaffold_bias[3] = torch.tensor([1.0, 1.0, 1.0])
    scaffold_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    out = _sc.scaffold_condition(
        logits, scaffold_idx, n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    assert torch.allclose(out, expected), (
        f"per-row bias selection failed: {out} vs {expected}"
    )


def test_scaffold_idx_shape_mismatch_raises() -> None:
    """scaffold_idx whose dim != 1 or whose length != B raises clearly."""
    torch = pytest.importorskip("torch")

    n_scaffolds, V = 16, 5
    scaffold_bias = torch.zeros(n_scaffolds, V)

    # 2D logits with wrong-length scaffold_idx
    logits_2d = torch.randn(8, 5)
    bad_idx = torch.zeros(5, dtype=torch.long)  # length 5, but B=8
    with pytest.raises(ValueError) as exc_info:
        _sc.scaffold_condition(
            logits_2d, bad_idx, n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
        )
    err = str(exc_info.value)
    assert "scaffold_idx" in err, (
        f"scaffold_idx error message should mention scaffold_idx: {err!r}"
    )

    # 2D logits with 2D scaffold_idx (dim=2 instead of dim=1)
    bad_idx_2d = torch.zeros(8, 1, dtype=torch.long)
    with pytest.raises(ValueError) as exc_info:
        _sc.scaffold_condition(
            logits_2d, bad_idx_2d, n_scaffolds=n_scaffolds, scaffold_bias=scaffold_bias,
        )
    err_2d = str(exc_info.value)
    assert "scaffold_idx" in err_2d, (
        f"scaffold_idx dim error message should mention scaffold_idx: {err_2d!r}"
    )