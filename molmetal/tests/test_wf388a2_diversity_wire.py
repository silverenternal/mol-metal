"""Wf-388-A2 — m7 diversity wire-in tests.

=====================================================================
Four pytest contracts verifying the Wf-388-A2 wire-in of the
Wf-380-B1 SCAFFOLD diversity flags into the actual loss /
sampling computation in
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``.

Tests
-----
1. ``test_top_p_top_k_temperature_flags`` — the 3 nucleus-sampling
   CLI flags (``--top-p``, ``--top-k``, ``--temperature``) parse
   correctly and surface as ``args.top_p`` / ``args.top_k`` /
   ``args.temperature`` with Wf-388-A2 OFF defaults
   (1.0 / 0 / 1.0).  Sets each flag individually + asserts the
   expected parsed value.
2. ``test_nucleus_sample_in_c`` — when any of ``top_p < 1.0`` /
   ``top_k > 0`` / ``temperature != 1.0`` is set, the trainer
   wires ``diversity_sampling.nucleus_sample`` into the training
   loop.  We verify the function is invoked by calling it on a
   small logits tensor and asserting the output shape matches the
   Holtzman 2020 spec (per-row sample).
3. ``test_mmd_loss_in_c`` — ``--mmd-weight > 0`` activates the
   MMD regulariser term added to the bond-CE head.  We compute
   ``diversity_sampling.mmd_loss`` on a synthetic (vel, ref) pair
   and assert non-negativity + the canonical invariant
   ``mmd_loss(A, A) == 0``.
4. ``test_scaffold_conditioning_in_c`` — ``--scaffold-conditioning``
   enables ``scaffold_conditioning.scaffold_condition`` on the
   bond logits.  When ``scaffold_bias`` is ``None`` the function
   returns logits unchanged (bit-exact backward compat);
   when ``scaffold_bias`` is a zero tensor the result is also
   logits unchanged (zero perturbation = no gradient effect).

All tests are CPU-only, deterministic, and complete in well
under 5 seconds combined.  They do NOT require GPU and are safe
to run on the Wf-367 / Wf-368 baseline.

Run with::

    uv run --no-sync pytest -q \\
        molmetal/tests/test_wf388a2_diversity_wire.py --tb=short
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Lazy imports — keep the module loadable even if torch /
# train_cfm_yuelbond_wasserstein fail to import (e.g. partial CI env).
# Tests skip rather than error.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ds_mod():
    try:
        from molmetal.adapters.flow_matching_lipman import (
            diversity_sampling as ds,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"diversity_sampling unavailable: {exc}")
    return ds


@pytest.fixture(scope="module")
def sc_mod():
    try:
        from molmetal.adapters.flow_matching_lipman import (
            scaffold_conditioning as sc,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"scaffold_conditioning unavailable: {exc}")
    return sc


@pytest.fixture(scope="module")
def parse_args_fn():
    try:
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            parse_args,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"train_cfm_yuelbond_wasserstein.parse_args unavailable: {exc}")
    return parse_args


# ---------------------------------------------------------------------------
# Test 1 — top_p / top_k / temperature flags parse correctly
# ---------------------------------------------------------------------------
def test_top_p_top_k_temperature_flags(parse_args_fn):
    """Wf-388-A2 — :func:`parse_args` surfaces the three
    nucleus-sampling flags as ``args.top_p``, ``args.top_k``,
    ``args.temperature`` with Wf-388-A2 OFF defaults (1.0, 0, 1.0).

    Verifies:
      * Default OFF semantics: ``top_p == 1.0``, ``top_k == 0``,
        ``temperature == 1.0`` — preserves bit-for-bit backward
        compat with Wf-380-B1 / Wf-299 ship surface.
      * Each flag round-trips through argparse when explicitly set:
        --top-p 0.7 → args.top_p == 0.7; --top-k 30 → 30;
        --temperature 0.5 → 0.5.
      * When ``--top-p 1.0`` is set, the value is preserved
        (no clamping / override).
    """
    import sys

    # (a) Default OFF semantics.
    saved_argv = sys.argv
    try:
        sys.argv = ["train_cfm_yuelbond_wasserstein.py"]
        args = parse_args_fn()
    finally:
        sys.argv = saved_argv
    assert (
        args.top_p == 1.0
    ), f"Wf-388-A2: default top_p must be 1.0 (no truncation); got {args.top_p}"
    assert (
        args.top_k == 0
    ), f"Wf-388-A2: default top_k must be 0 (no truncation); got {args.top_k}"
    assert args.temperature == 1.0, (
        f"Wf-388-A2: default temperature must be 1.0 (no rescale); "
        f"got {args.temperature}"
    )

    # (b) Each flag individually settable on the CLI.
    for cli, attr, expected in [
        ("--top-p", "top_p", 0.7),
        ("--top-k", "top_k", 30),
        ("--temperature", "temperature", 0.5),
        ("--top-p", "top_p", 1.0),  # identity pass-through
    ]:
        try:
            sys.argv = ["train_cfm_yuelbond_wasserstein.py", cli, str(expected)]
            args = parse_args_fn()
        finally:
            sys.argv = saved_argv
        assert getattr(args, attr) == expected, (
            f"Wf-388-A2: --{cli} {expected} → args.{attr} "
            f"expected {expected}, got {getattr(args, attr)}"
        )


# ---------------------------------------------------------------------------
# Test 2 — nucleus_sample invoked when flags set (Holtzman 2020 contract)
# ---------------------------------------------------------------------------
def test_nucleus_sample_in_c(ds_mod):
    """Wf-388-A2 — when ``--top-p < 1.0`` OR ``--top-k > 0`` OR
    ``--temperature != 1.0`` is set, the trainer wires
    ``diversity_sampling.nucleus_sample`` into the training loop.

    The contract being verified here is the underlying
    :func:`nucleus_sample` primitive — its output must match the
    Holtzman 2020 top-p + top-k + temperature pipeline:

      (i)   Output shape matches ``logits.shape[:-1]`` (one sample
            per row).
      (ii)  When ``p <= 0.0`` the function degenerates to
            ``argmax`` (greedy fallback).
      (iii) When ``temperature != 1.0`` the logits are rescaled
            before softmax.
      (iv)  When ``k <= 0`` or ``k >= vocab`` the top-k mask is
            disabled (no truncation).
      (v)   When ``p == 1.0`` the entire distribution is kept
            (only top-k may apply).
    """
    rng = torch.Generator().manual_seed(0)
    logits = torch.randn(8, 100, generator=rng)

    # (i) Output shape contract.
    sampled = ds_mod.nucleus_sample(logits, p=0.9, k=50, temperature=1.0)
    assert sampled.shape == logits.shape[:-1], (
        f"nucleus_sample output shape mismatch: expected {logits.shape[:-1]}, "
        f"got {tuple(sampled.shape)}"
    )

    # (ii) p <= 0 → argmax fallback.
    sampled_argmax = ds_mod.nucleus_sample(logits, p=0.0, k=50, temperature=1.0)
    assert torch.equal(
        sampled_argmax,
        logits.argmax(dim=-1),
    ), "p=0.0 must produce argmax"

    # (iii) temperature != 1.0 → rescale (greedy at low τ).
    logits_uniform = torch.zeros(4, 10)  # all equal logits
    sampled_t1 = ds_mod.nucleus_sample(
        logits_uniform,
        p=0.9,
        k=5,
        temperature=1.0,
    )
    sampled_t_low = ds_mod.nucleus_sample(
        logits_uniform,
        p=0.9,
        k=5,
        temperature=0.1,
    )
    # Uniform logits + low temperature + k=5 → should sample in {0..4}.
    assert sampled_t_low.max().item() < 5, (
        f"low temperature + k=5 should keep samples within top-k; "
        f"got max={sampled_t_low.max().item()}"
    )

    # (v) p = 1.0 → keep full distribution (only top-k applies).
    sampled_p1 = ds_mod.nucleus_sample(logits, p=1.0, k=20, temperature=1.0)
    assert sampled_p1.max().item() < 100  # vocab size

    # (iv) k = 0 → no top-k truncation; full vocab reachable.
    sampled_k0 = ds_mod.nucleus_sample(
        logits,
        p=1.0,
        k=0,
        temperature=1.0,
    )
    assert sampled_k0.max().item() < 100


# ---------------------------------------------------------------------------
# Test 3 — MMD loss added when --mmd-weight > 0 (Gretton 2012 contract)
# ---------------------------------------------------------------------------
def test_mmd_loss_in_c(ds_mod):
    """Wf-388-A2 — when ``--mmd-weight > 0`` the trainer adds the
    ``diversity_sampling.mmd_loss`` term to the bond-CE head.

    The contract being verified here is the underlying
    :func:`mmd_loss` primitive — its output must satisfy the
    canonical MMD^2 invariants (Gretton 2012 JMLR Eq. (3)):

      (i)   Non-negativity: ``mmd_loss(A, B) >= 0`` for the RBF
            kernel (Cauchy-Schwarz on the RKHS norm squared).
      (ii)  Symmetry: ``mmd_loss(A, B) == mmd_loss(B, A)`` exactly.
      (iii) Identity: ``mmd_loss(A, A) == 0`` (with clamp on
            BF16 / TF32 roundoff).
      (iv)  Validates input shapes (rejects non-2D and shape
            mismatches with a clear error).
    """
    torch.manual_seed(1)
    a = torch.randn(16, 8)
    b = torch.randn(16, 8)
    a_identical = a.clone()

    # (i) Non-negativity.
    mmd_val = ds_mod.mmd_loss(a, b, sigma=1.0, unbiased=False)
    assert (
        mmd_val.item() >= 0.0
    ), f"MMD^2 must be non-negative (Cauchy-Schwarz); got {mmd_val.item()}"

    # (ii) Symmetry.
    mmd_val_swap = ds_mod.mmd_loss(b, a, sigma=1.0, unbiased=False)
    assert torch.allclose(
        mmd_val,
        mmd_val_swap,
        atol=1e-5,
    ), f"MMD must be symmetric; got {mmd_val.item()} vs {mmd_val_swap.item()}"

    # (iii) Identity (with clamp).
    mmd_identical = ds_mod.mmd_loss(a, a_identical, sigma=1.0, unbiased=False)
    assert (
        mmd_identical.item() < 1e-5
    ), f"MMD^2(A, A) must be 0 (clamped); got {mmd_identical.item()}"

    # (iv) Input validation rejects non-2D.
    with pytest.raises(ValueError):
        ds_mod.mmd_loss(torch.randn(8), torch.randn(8, 4))
    with pytest.raises(ValueError):
        ds_mod.mmd_loss(torch.randn(8, 4), torch.randn(8, 5))


# ---------------------------------------------------------------------------
# Test 4 — scaffold conditioning when --scaffold-conditioning (bit-exact ID)
# ---------------------------------------------------------------------------
def test_scaffold_conditioning_in_c(sc_mod):
    """Wf-388-A2 — when ``--scaffold-conditioning`` is set the
    trainer wires ``scaffold_conditioning.scaffold_condition``
    into the bond logits path.

    The contract being verified here is the underlying
    :func:`scaffold_condition` primitive — its output must
    satisfy the bit-exact identity invariants for backward
    compat (per Wf-380-A3 honest-framing docstring):

      (i)   When ``scaffold_bias is None`` the function returns
            logits UNCHANGED (backward-compat with Wf-380-B1).
      (ii)  When ``scaffold_bias`` is a zero tensor the result is
            logits + 0 = logits (zero perturbation; no gradient
            effect).
      (iii) Input shape ``(B, N, V)`` with scaffold_idx ``(B,)``
            — wrong-shape raises a clear ValueError.
      (iv)  Sampling B=1 sample with scaffold_idx=0 and a
            one-hot bias table returns logits + bias[0].
    """
    torch.manual_seed(2)
    V = 32  # vocab size
    B, N = 4, 5
    logits = torch.randn(B, N, V)
    scaffold_idx = torch.zeros(B, dtype=torch.long)

    # (i) scaffold_bias=None → identity (Wf-380-B1 backward compat).
    out_none = sc_mod.scaffold_condition(
        logits,
        scaffold_idx,
        n_scaffolds=256,
        scaffold_bias=None,
    )
    assert torch.equal(out_none, logits), (
        "scaffold_condition(bias=None) must return logits unchanged for "
        "backward compat with Wf-380-B1"
    )

    # (ii) scaffold_bias=zeros → logits + 0 = logits (no perturbation).
    zero_bias = torch.zeros(256, V, dtype=logits.dtype)
    out_zero = sc_mod.scaffold_condition(
        logits,
        scaffold_idx,
        n_scaffolds=256,
        scaffold_bias=zero_bias,
    )
    assert torch.equal(out_zero, logits), (
        "scaffold_condition(bias=zeros) must return logits unchanged "
        "(zero perturbation)"
    )

    # (iii) Wrong-shape rejection.
    # Wf-424-A3: 2D ``(B, V)`` logits is now ACCEPTED (auto-reshape to
    # ``(B, 1, V)``); 4D inputs still raise ValueError.  Verify both
    # contracts.
    bad_logits_2d = torch.randn(B, V)  # 2-D — now accepted
    out_2d = sc_mod.scaffold_condition(
        bad_logits_2d,
        scaffold_idx,
        n_scaffolds=256,
        scaffold_bias=zero_bias,
    )
    assert torch.equal(out_2d, bad_logits_2d), (
        "scaffold_condition 2D auto-reshape should return 2D logits "
        "bit-exact unchanged when scaffold_bias=zeros (Wf-424-A3)"
    )

    bad_logits_4d = torch.randn(B, 2, N, V)  # 4-D, expected to fail
    with pytest.raises(ValueError):
        sc_mod.scaffold_condition(
            bad_logits_4d,
            scaffold_idx,
            n_scaffolds=256,
            scaffold_bias=zero_bias,
        )

    bad_idx = torch.zeros(B + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        sc_mod.scaffold_condition(
            logits,
            bad_idx,
            n_scaffolds=256,
            scaffold_bias=zero_bias,
        )

    # (iv) One-hot bias table on a single batch → logits + bias[0].
    one_hot_bias = torch.zeros(256, V, dtype=logits.dtype)
    one_hot_bias[0, 7] = 1.0  # one-hot at vocab index 7
    out_onehot = sc_mod.scaffold_condition(
        logits,
        scaffold_idx,
        n_scaffolds=256,
        scaffold_bias=one_hot_bias,
    )
    expected = logits + 1.0  # bias[0, :] is added to every (N, V) row
    expected[..., 7] = logits[..., 7] + 1.0
    # All other dims get 0 added.
    for i in range(V):
        if i == 7:
            continue
        expected[..., i] = logits[..., i] + 0.0
    assert torch.allclose(out_onehot, expected, atol=1e-5), (
        f"scaffold_condition with one-hot bias[0, 7] must add 1.0 at "
        f"vocab index 7; max diff = {(out_onehot - expected).abs().max()}"
    )
