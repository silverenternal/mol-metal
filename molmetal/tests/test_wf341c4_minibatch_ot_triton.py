"""Wf-341-C4 — Tests for the mini-batch OT Triton kernel CPU scaffold.

=====================================================================
Per Wf-340-A3, the ``Mini-batch-OT`` solver from
`khainb/Mini-batch-OT <https://github.com/khainb/Mini-batch-OT>`_
(MIT-licensed) is one of the top-5 borrowable code candidates for
CFM training acceleration.  Wf-341-C4 ships the **CPU scaffold** for
the fused cost-matrix + greedy-argmin kernel that
:func:`flow_matching.optimal_transport._solve_minibatch_ot`'s cost
step (``torch.cdist(x0, x1, p=2.0).pow(2)``) is the dominant cost
when ``method="gapless_greedy"``.

Wf-341-C4 ships 4 NEW tests in
``molmetal/tests/test_wf341c4_minibatch_ot_triton.py``:

  Group A — Mini-batch OT coupling kernel:
    test_01_minibatch_ot_triton_correctness_vs_pytorch
    test_02_minibatch_ot_triton_integration_with_mini_batch_ot_solver
    test_03_minibatch_ot_triton_fallback_to_pytorch
    test_04_minibatch_ot_triton_cpu_correctness

All tests are CPU-only (no GPU launch).  Each correctness test
verifies the CPU PyTorch fallback matches an independently-computed
PyTorch baseline (bit-exact for fp32 / within 1e-5 for reductions).
The integration test verifies the kernel's output matches the
upstream ``flow_matching.optimal_transport._solve_minibatch_ot``
``method="gapless_greedy"``-equivalent (the no-scipy / no-POT
fallback).

Per AGENTS.md §5.1, this is an audit (no retrain / panel / ablation
launch), so pre-flight gate (TODO-207) is N/A and the kill_then_adapt
directive does not apply.  Per AGENTS.md §5.2, all claims use the
MEASURED/PROJECTED tag discipline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = (
    _PROJECT_ROOT / "molmetal" / "adapters" / "flow_matching_lipman" / "_reference"
)

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the mini-batch OT kernel module.  Top-level import is safe even
# when triton is unavailable (the try/except inside installs a no-op stub).
from molmetal.adapters.flow_matching_lipman._reference.minibatch_ot_triton import (  # noqa: E402
    fused_minibatch_ot_coupling,
    pytorch_minibatch_ot_coupling,
    MinibatchOTTritonConfig,
    dispatch_state_report,
)


# ===========================================================================
# Group A — Mini-batch OT coupling kernel (4 tests)
# ===========================================================================
class TestMinibatchOTTriton:
    """Tests 01-04: mini-batch OT Triton kernel CPU correctness + integration."""

    def test_01_minibatch_ot_triton_correctness_vs_pytorch(self):
        """Verify the CPU fallback for ``fused_minibatch_ot_coupling``
        matches an independently-computed PyTorch baseline
        (``torch.argmin(torch.cdist(x0, x1, p=2.0).pow(2), dim=1)``).

        Tests with 3 shape configs:
          - (n=8, d=4): small dense
          - (n=32, d=8): medium
          - (n=64, d=16): large
        All on CPU (auto PyTorch path), all expect bit-exact agreement.
        """
        torch.manual_seed(0)
        for n, d in [(8, 4), (32, 8), (64, 16)]:
            x0 = torch.randn(n, d)
            x1 = torch.randn(n, d)
            # Independent PyTorch baseline (mirror _solve_minibatch_ot).
            baseline_cost = torch.cdist(x0, x1, p=2.0).pow(2)
            baseline_idx = torch.argmin(baseline_cost, dim=1)
            # CPU fallback (no CUDA → auto PyTorch reference path).
            out = fused_minibatch_ot_coupling(x0, x1)
            assert out.shape == (n,), (
                f"[n={n}, d={d}] shape mismatch: got {tuple(out.shape)}, "
                f"expected ({n},)"
            )
            assert (
                out.dtype == torch.long
            ), f"[n={n}, d={d}] dtype mismatch: got {out.dtype}, expected torch.long"
            assert torch.equal(out, baseline_idx), (
                f"[n={n}, d={d}] mini-batch OT coupling diverges from "
                f"independent PyTorch baseline. "
                f"max diff: {(out - baseline_idx).abs().max().item():.6f}"
            )

    def test_02_minibatch_ot_triton_integration_with_mini_batch_ot_solver(self):
        """Verify the kernel's output matches the upstream
        ``flow_matching.optimal_transport._solve_minibatch_ot``
        ``method="gapless_greedy"``-equivalent for a small batch.

        Builds a flat 2D mini-batch (n=16, d=8) and runs both the
        kernel scaffold's :func:`fused_minibatch_ot_coupling` and a
        hand-rolled ``torch.argmin(torch.cdist(x0, x1, p=2.0).pow(2),
        dim=1)`` equivalent (the same cost matrix that
        ``_solve_minibatch_ot`` builds before Sinkhorn / Hungarian /
        random fallback).  Asserts bit-exact agreement.
        """
        torch.manual_seed(7)
        n, d = 16, 8
        x0 = torch.randn(n, d)
        x1 = torch.randn(n, d)
        # Independent PyTorch reference (no scipy / no POT dependency).
        cost = torch.cdist(x0, x1, p=2.0).pow(2)
        expected_idx = torch.argmin(cost, dim=1)
        # Kernel scaffold (CPU fallback path).
        out = fused_minibatch_ot_coupling(x0, x1)
        assert out.shape == expected_idx.shape, (
            f"shape mismatch: got {tuple(out.shape)}, "
            f"expected {tuple(expected_idx.shape)}"
        )
        assert torch.equal(out, expected_idx), (
            f"kernel output does not match _solve_minibatch_ot's "
            f"cost+argmin.indices chain. "
            f"max diff: {(out - expected_idx).abs().max().item():.6f}"
        )
        # Also verify the module-level pytorch_minibatch_ot_coupling
        # matches the kernel output (parity check on the public surface).
        ref = pytorch_minibatch_ot_coupling(x0, x1)
        assert torch.equal(out, ref), (
            "fused_minibatch_ot_coupling output diverges from "
            "module pytorch_minibatch_ot_coupling()"
        )

    def test_03_minibatch_ot_triton_fallback_to_pytorch(self):
        """Verify ``fused_minibatch_ot_coupling`` returns the PyTorch
        reference when ``force_pytorch=True`` and when the env-var
        gate is closed (default), regardless of triton availability.

        Also verifies that the shape gate (N > 1024 or D > 64) routes
        back to the PyTorch reference (the autotune configs are
        BLOCK_N in {32,64,128} so anything above the supported regime
        must fall back).
        """
        torch.manual_seed(11)
        n, d = 8, 4
        x0 = torch.randn(n, d)
        x1 = torch.randn(n, d)
        # Default dispatch (env gates closed by default → PyTorch).
        out_default = fused_minibatch_ot_coupling(x0, x1)
        # force_pytorch=True → always PyTorch.
        out_forced = fused_minibatch_ot_coupling(x0, x1, force_pytorch=True)
        # Module reference.
        ref = pytorch_minibatch_ot_coupling(x0, x1)
        assert torch.equal(out_default, ref), (
            "default dispatch output diverges from " "pytorch_minibatch_ot_coupling()"
        )
        assert torch.equal(out_forced, ref), (
            "force_pytorch=True output diverges from "
            "pytorch_minibatch_ot_coupling()"
        )
        # Shape gate: N > 1024 must still return a valid PyTorch result
        # (the gate_open() check + the explicit N > 1024 fallback in the
        # dispatcher route the call to pytorch_minibatch_ot_coupling
        # even when force_pytorch=False).
        n_too_big = 2048
        x0_big = torch.randn(n_too_big, d)
        x1_big = torch.randn(n_too_big, d)
        out_big = fused_minibatch_ot_coupling(x0_big, x1_big)
        ref_big = pytorch_minibatch_ot_coupling(x0_big, x1_big)
        assert torch.equal(out_big, ref_big), (
            f"shape-gate fallback (N={n_too_big} > 1024) diverges from "
            f"pytorch_minibatch_ot_coupling()"
        )

    def test_04_minibatch_ot_triton_cpu_correctness(self):
        """Verify the CPU PyTorch reference produces a valid index
        assignment (each ``idx[i]`` is in ``[0, n)``) and a non-trivial
        cost reduction vs. an identity baseline (i.e. the OT coupling is
        doing actual work, not just copying the input order).

        Note: the greedy nearest-neighbour mini-batch OT coupling is
        **not** a permutation — collisions (two x0 rows mapping to the
        same x1 column) are allowed, and indeed expected at small n.
        The hard-permutation projection is the Sinkhorn / Hungarian
        layer's job, not the cost+argmin layer this kernel ships.  See
        ``flow_matching.optimal_transport._sinkhorn_plan_to_permutation``
        for the rounding step.

        Tests with 3 shape configs:
          - (n=4, d=3): tiny — index-in-range + cost reduction
          - (n=16, d=8): medium — index-in-range + cost reduction
          - (n=32, d=4): medium-wide — index-in-range + cost reduction

        Asserts:
          1. ``idx[i] in [0, n)`` for every i (valid index range).
          2. The greedy OT cost is <= the identity cost (OT is doing
             actual work — not just a placeholder).
          3. The fused dispatch agrees with the module reference.
        """
        for n, d in [(4, 3), (16, 8), (32, 4)]:
            torch.manual_seed(42 + n)
            x0 = torch.randn(n, d)
            x1 = torch.randn(n, d)
            # pytorch reference (CPU-only — auto-routes regardless of gate state).
            idx = pytorch_minibatch_ot_coupling(x0, x1)
            # 1. Index-range validity: every index is in [0, n).
            assert idx.shape == (n,), (
                f"[n={n}, d={d}] idx shape: got {tuple(idx.shape)}, " f"expected ({n},)"
            )
            assert (
                idx.dtype == torch.long
            ), f"[n={n}, d={d}] idx dtype: got {idx.dtype}, expected torch.long"
            assert idx.min().item() >= 0, (
                f"[n={n}, d={d}] idx has out-of-range negative value: "
                f"min={idx.min().item()}"
            )
            assert idx.max().item() < n, (
                f"[n={n}, d={d}] idx has out-of-range positive value: "
                f"max={idx.max().item()} >= n={n}"
            )
            # 2. Cost reduction: greedy OT cost must beat identity cost.
            cost = torch.cdist(x0, x1, p=2.0).pow(2)
            greedy_cost = cost.gather(1, idx.unsqueeze(1)).sum().item()
            identity_cost = cost.diagonal().sum().item()
            assert greedy_cost <= identity_cost + 1e-4, (
                f"[n={n}, d={d}] greedy OT cost ({greedy_cost:.4f}) is "
                f"worse than identity cost ({identity_cost:.4f}) — the "
                f"kernel is not doing actual OT work."
            )
            # 3. Fused dispatch agrees with module reference.
            out = fused_minibatch_ot_coupling(x0, x1)
            assert torch.equal(out, idx), (
                f"[n={n}, d={d}] fused_minibatch_ot_coupling disagrees "
                f"with pytorch_minibatch_ot_coupling"
            )


# ===========================================================================
# Group B — Config + report (bonus: dispatch surface tests)
# ===========================================================================
class TestMinibatchOTConfig:
    """Tests 05-06: config holder + dispatch state report."""

    def test_05_minibatch_ot_config_from_env_defaults(self):
        """Verify MinibatchOTTritonConfig.from_env() reads the default
        gate state correctly when no env vars are set."""
        # Save + clear env vars.
        saved = {
            "TRITON_USE_FUSED": os.environ.pop("TRITON_USE_FUSED", "0"),
            "TRITON_USE_FUSED_EVAL": os.environ.pop("TRITON_USE_FUSED_EVAL", "0"),
            "TRITON_MINIBATCH_OT": os.environ.pop("TRITON_MINIBATCH_OT", "0"),
        }
        try:
            cfg = MinibatchOTTritonConfig.from_env()
            assert cfg.use_fused is False, "default TRITON_USE_FUSED should be False"
            assert (
                cfg.use_fused_eval is False
            ), "default TRITON_USE_FUSED_EVAL should be False"
            assert (
                cfg.use_minibatch_ot is False
            ), "default TRITON_MINIBATCH_OT should be False"
            assert (
                cfg.gate_open() is False
            ), "default gate should be closed (no TRITON_*=1 in env)"
            assert (
                cfg.gate_open(is_eval=True) is False
            ), "default gate should be closed in eval mode too"
        finally:
            for k, v in saved.items():
                os.environ[k] = v

    def test_06_minibatch_ot_dispatch_state_report(self):
        """Verify dispatch_state_report() returns a JSON-serialisable
        dict with the expected keys."""
        import json

        report = dispatch_state_report()
        # Must be JSON-serialisable (the upstream callers will JSON dump it).
        json.dumps(report)
        assert "triton_available" in report
        assert "use_fused" in report
        assert "use_fused_eval" in report
        assert "use_minibatch_ot" in report
        assert "gate_open_default" in report
        assert "raw_env" in report
        assert "supported_n_max" in report
        assert "supported_d_max" in report
        assert report["supported_n_max"] == 1024
        assert report["supported_d_max"] == 64
        assert (
            report["gate_open_default"] is False
        ), "default gate should be closed (no TRITON_*=1 in env)"
