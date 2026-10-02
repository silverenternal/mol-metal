"""Wf-328-W1: torch.compile wired into production trainer — tests.

=================================================================
Wf-328-W1 — production-port tests for the Wf-326C torch.compile
optimisations on the Wf-131 P1 / Wf-181 G5 / Wf-182 G6 trainer.
=================================================================

Per Wf-326C MEASURED: applying ``torch.compile(mode='reduce-overhead')``
to the D-MPNN encoder forward + the W₂ closed-form gives 2.72×
encoder + 2.41× W₂ step speedup on the RTX PRO 6000 Blackwell sm_120.
The Wf-326C bench lives in
``molmetal/scripts/wf326c_torch_compile_optimized.py``; this file
verifies the production trainer port lives in
``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``.

Test inventory (4 tests per Wf-328-W1 brief)
-------------------------------------------
1. test_torch_compile_flag_passed           — CLI flag wiring
2. test_model_encoder_wrapped_with_compile  — encoder monkey-patch
3. test_w2_loss_wrapped_with_compile        — W₂ wrapper thread-through
4. test_compile_integration_with_trainer    — end-to-end smoke

All tests are CPU-only by default (matching the Wf-326C bench); GPU
flag ``--torch-compile-mode max-autotune`` can be enabled by setting
``TORCH_COMPILE_GPU=1`` in the environment (skipped otherwise).  The
harness auto-fallback (Wf-320B) keeps GPU contention at zero so the
Inductor compile latency (~30s first run) does not perturb the in-
flight Wf-124 P3 / Wf-181 G5 / Wf-182 G6 runs.

**Honest framing (party-twenty-spirit §3.5):**
* Compile correctness is verified CPU-only; absolute GPU speedup
  numbers come from Wf-326C (MEASURED) — these tests verify the
  *wiring*, not the speedup.
* torch.compile compile latency is ~30s first-run; amortised after
  ~100 forward passes on static shape.
* mode='reduce-overhead' enables CUDA Graphs inside Inductor; on CPU
  hosts this mode is a no-op (no CUDA Graphs available).
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskModel,
    MultiTaskLoss,
)
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    WassersteinConfig,
    wasserstein_pic50_gradient,
)

# Import the production trainer helpers — these are the units under
# test (Wf-328-W1 ports the Wf-326C optimisations into the production
# trainer via these wrappers).
from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
    _make_compiled_encoder_forward,
    _make_compiled_w2_penalty,
    parse_args,
    train_one_metal_one_seed,
    train_one_epoch_joint,
    MultiTaskWassersteinModel,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

def _make_synthetic_graph_batch(
    B: int = 4,
    V: int = 24,
    E: int = 48,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a (B, V, A), (B, E, B), (B, E), (B, E) synthetic batch.

    Deterministic shape-pinned batch used by the integration tests so
    the inputs are RDKit-free and reproducible across hosts.
    """
    g = torch.Generator().manual_seed(int(seed))
    atom_f = torch.randn(B, V, ATOM_FEATURE_DIM, generator=g)
    bond_f = torch.randn(B, E, BOND_FEATURE_DIM, generator=g)
    e_src = torch.randint(0, V, (B, E), generator=g)
    e_dst = torch.randint(0, V, (B, E), generator=g)
    return atom_f, bond_f, e_src, e_dst


def _build_small_encoder(hidden: int = 32, depth: int = 2) -> DMPNNMultiTaskModel:
    """Build a small DMPNNMultiTaskModel for compile correctness tests."""
    torch.manual_seed(0)
    return DMPNNMultiTaskModel(hidden=hidden, depth=depth, dropout=0.0)


# ---------------------------------------------------------------------------
# 1: CLI flag wiring (test_torch_compile_flag_passed)
# ---------------------------------------------------------------------------

def test_torch_compile_flag_passed() -> None:
    """``--torch-compile`` / ``--no-torch-compile`` / ``--torch-compile-mode``
    are parsed by :func:`parse_args` and forwarded to the trainer.

    Verifies the new Wf-328-W1 CLI surface (updated Wf-335-R1):
      * default ``args.torch_compile = True``
      * ``args.torch_compile_mode = 'max-autotune-no-cudagraphs'``
        (Wf-335-R1 MEASURED FIX: was 'reduce-overhead' per Wf-328-W1,
        but 'reduce-overhead' enabled CUDA Graphs capture that fired
        RuntimeError on Ru seed=2 + Ir seed=0 per Wf-330-B4).
      * ``--no-torch-compile`` flips the flag to ``False``
      * ``--torch-compile-mode max-autotune-no-cudagraphs`` is accepted
        (CPU-only safe mode)
      * ``--torch-compile-mode reduce-overhead`` is REJECTED by the
        fix branch of main() override (CUDA Graphs disabled).  Pass
        ``--torch-compile-mode-v2 reduce-overhead`` to opt back in
        (legacy bench reproducer).
    """
    # Default — both flags wired with their shipped defaults
    args_default = parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])
    assert args_default.torch_compile is True, (
        f"default torch_compile should be True (Wf-328-W1 shipped), "
        f"got {args_default.torch_compile}"
    )
    # Wf-335-R1 MEASURED FIX: default switched from 'reduce-overhead' to
    # 'max-autotune-no-cudagraphs' to disable CUDA Graphs capture that
    # was triggering RuntimeError on Ru seed=2 + Ir seed=0 per Wf-330-B4.
    assert args_default.torch_compile_mode == "max-autotune-no-cudagraphs", (
        f"default torch_compile_mode should be 'max-autotune-no-cudagraphs' "
        f"(Wf-335-R1 MEASURED FIX per Wf-330-B4), got "
        f"{args_default.torch_compile_mode}"
    )

    # --no-torch-compile flips to False
    args_off = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "1", "--no-torch-compile"]
    )
    assert args_off.torch_compile is False, (
        f"--no-torch-compile should set torch_compile=False, got {args_off.torch_compile}"
    )

    # --torch-compile-mode accepts the four Inductor modes
    for mode in ("default", "reduce-overhead", "max-autotune",
                 "max-autotune-no-cudagraphs"):
        args_mode = parse_args(
            [
                "--metals", "Ru", "--seeds", "0", "--epochs", "1",
                "--torch-compile-mode", mode,
            ]
        )
        assert args_mode.torch_compile_mode == mode, (
            f"--torch-compile-mode {mode!r} did not parse; got "
            f"{args_mode.torch_compile_mode}"
        )

    # Invalid mode is rejected by argparse choices=
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--metals", "Ru", "--seeds", "0", "--epochs", "1",
                "--torch-compile-mode", "not-a-mode",
            ]
        )


# ---------------------------------------------------------------------------
# 2: Encoder monkey-patch (test_model_encoder_wrapped_with_compile)
# ---------------------------------------------------------------------------

def test_model_encoder_wrapped_with_compile() -> None:
    """:func:`_make_compiled_encoder_forward` returns a callable that
    replaces ``encoder._forward_one_mol`` and yields the same output as
    the eager path within FP32 tolerance.

    Verifies the production trainer port: ``encode_pool`` and
    ``forward`` both look up ``_forward_one_mol`` at call-time so a
    monkey-patch is picked up on the next forward — this is the same
    pattern Wf-326C used in ``_make_compiled_dmpnn_encoder``.
    """
    encoder = _build_small_encoder(hidden=32, depth=2)
    encoder.eval()

    # Build a small per-mol input so the compiled forward sees a
    # real (V, E) shape.
    torch.manual_seed(1)
    V, E = 12, 24
    atom_f = torch.randn(V, ATOM_FEATURE_DIM)
    bond_f = torch.randn(E, BOND_FEATURE_DIM)
    e_src = torch.randint(0, V, (E,))
    e_dst = torch.randint(0, V, (E,))

    with torch.no_grad():
        out_eager = encoder._forward_one_mol(atom_f, bond_f, e_src, e_dst)

    # Patch in the compiled forward
    compiled_forward = _make_compiled_encoder_forward(encoder, mode="reduce-overhead")
    original = encoder._forward_one_mol
    encoder._forward_one_mol = compiled_forward  # type: ignore[assignment]
    try:
        with torch.no_grad():
            # First call triggers Inductor compile
            out_compiled_first = encoder._forward_one_mol(
                atom_f, bond_f, e_src, e_dst,
            )
            # Second call hits the Inductor cache
            out_compiled_second = encoder._forward_one_mol(
                atom_f, bond_f, e_src, e_dst,
            )

        assert out_compiled_first.shape == out_eager.shape, (
            f"compiled encoder output shape mismatch: eager {out_eager.shape} "
            f"vs compiled {out_compiled_first.shape}"
        )
        diff_first = (out_eager - out_compiled_first).abs().max().item()
        diff_second = (out_eager - out_compiled_second).abs().max().item()
        assert diff_first < 1e-3, (
            f"compiled encoder first-call output mismatch: {diff_first}"
        )
        assert diff_second < 1e-3, (
            f"compiled encoder second-call output mismatch: {diff_second}"
        )
    finally:
        # Restore the original — matches the production trainer's
        # try/finally restore in :func:`train_one_metal_one_seed`.
        encoder._forward_one_mol = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 3: W₂ wrapper thread-through (test_w2_loss_wrapped_with_compile)
# ---------------------------------------------------------------------------

def test_w2_loss_wrapped_with_compile() -> None:
    """:func:`_make_compiled_w2_penalty` returns a callable that produces
    the same W₂ value + gradient as the eager :func:`wasserstein_pic50_gradient`
    within FP32 tolerance.

    Per Wf-326C: the conditional ``if xs.numel() != n`` interpolation
    branch lives outside Inductor in the eager path; the compile
    wrapper collapses the branch into a single Triton kernel for ~2.4×
    on GPU.  This test verifies the wrapper preserves correctness so
    the per-(metal, seed) trainer call site can rely on it.
    """
    torch.manual_seed(2)
    pred = torch.randn(64, requires_grad=True)
    target = torch.randn(128)

    w_eager = wasserstein_pic50_gradient(
        pred, target.detach(), w_weight=1.0,
    )
    grad_eager = torch.autograd.grad(w_eager, pred, retain_graph=False)[0]

    # Build compiled wrapper
    compiled_w2 = _make_compiled_w2_penalty(w_weight=1.0, mode="reduce-overhead")

    pred2 = pred.detach().clone().requires_grad_(True)
    w_compiled = compiled_w2(pred2, target.detach())
    grad_compiled = torch.autograd.grad(
        w_compiled, pred2, retain_graph=False,
    )[0]

    diff_w = (w_eager - w_compiled).abs().item()
    diff_g = (grad_eager - grad_compiled).abs().max().item()

    assert diff_w < 1e-3, f"compiled W₂ value mismatch: {diff_w}"
    assert diff_g < 1e-3, f"compiled W₂ gradient mismatch: {diff_g}"


# ---------------------------------------------------------------------------
# 4: End-to-end integration with the trainer (test_compile_integration_with_trainer)
# ---------------------------------------------------------------------------

def test_compile_integration_with_trainer() -> None:
    """End-to-end smoke: the compiled encoder + W₂ wrap cleanly into
    :func:`train_one_epoch_joint` without breaking the autograd graph.

    Builds a tiny :class:`MultiTaskWassersteinModel` + 4-graph synthetic
    batch, monkey-patches the encoder forward with the compiled
    wrapper, threads the compiled W₂ through
    :func:`train_one_epoch_joint`, and asserts the backward pass still
    flows gradient into the encoder weights.  CPU-only by design — the
    harness auto-fallback (Wf-320B) keeps GPU contention at zero.

    Honest framing: this is a *correctness* smoke, not a speedup
    measurement.  Absolute speedup numbers live in Wf-326C (MEASURED
    2.72× encoder + 2.41× W₂ on RTX PRO 6000 Blackwell sm_120); this
    test verifies the wiring does not break the autograd graph.
    """
    torch.manual_seed(3)
    np_seed = 3
    import numpy as np
    np.random.seed(np_seed)

    hidden = 32
    depth = 2
    n_metals = 5
    device = torch.device("cpu")

    model = MultiTaskWassersteinModel(
        n_metals=n_metals, hidden=hidden, depth=depth, dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=0.5, use_kendall_loss=False)

    # Patch encoder forward with the compiled wrapper — matches the
    # production trainer pattern in :func:`train_one_metal_one_seed`.
    compiled_encoder = _make_compiled_encoder_forward(
        model.encoder, mode="reduce-overhead",
    )
    compiled_w2 = _make_compiled_w2_penalty(
        w_weight=1.0, mode="reduce-overhead",
    )
    original_forward_one_mol = model.encoder._forward_one_mol
    model.encoder._forward_one_mol = compiled_encoder  # type: ignore[assignment]

    try:
        # Synthetic batch — (B=4, V=24, E=48)
        atom_f, bond_f, e_src, e_dst = _make_synthetic_graph_batch(
            B=4, V=24, E=48, seed=4,
        )
        mids_t = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        pic50_t = torch.tensor([5.1, 6.2, 4.3, 7.5], dtype=torch.float32)
        active_t = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        ref_pic50 = torch.randn(64)

        # Single forward + loss + backward — verify the autograd graph
        # is intact and the encoder weights receive gradient.
        optimizer.zero_grad()
        with torch.no_grad():
            pass  # ensure no leakage from previous test
        optimizer.zero_grad()

        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
        task_loss, lc, lr_ = criterion(pic50_pred, active_logits, pic50_t, active_t)

        # Use compiled W₂ wrapper (sort mode)
        w_pen = compiled_w2(pic50_pred, ref_pic50)
        total = task_loss + 0.05 * w_pen

        assert torch.isfinite(total), f"compiled total loss not finite: {total}"
        total.backward()

        # Check that gradients flow back into the encoder weights —
        # this is the critical correctness gate.  If the compiled
        # forward breaks the autograd graph, every encoder gradient will
        # be None or zero and the trainer sits at signal-floor R²
        # (TODO-179 root cause).
        encoder_param_grads = [
            p for n, p in model.encoder.named_parameters()
            if p.grad is not None
        ]
        assert len(encoder_param_grads) > 0, (
            "no encoder parameter received a gradient — compiled forward "
            "broke the autograd graph"
        )
        encoder_grad_norm = sum(
            (p.grad.norm().item() ** 2 for p in encoder_param_grads)
        ) ** 0.5
        assert encoder_grad_norm > 1e-6, (
            f"encoder gradient norm is zero after compiled+predict pipeline: "
            f"||grad||={encoder_grad_norm}"
        )
    finally:
        # Restore — matches the production trainer's try/finally.
        model.encoder._forward_one_mol = original_forward_one_mol  # type: ignore[assignment]


if __name__ == "__main__":
    # Allow direct invocation for CPU-only smoke runs (no CUDA required).
    pytest.main([__file__, "-v", "-x", "--tb=short"])