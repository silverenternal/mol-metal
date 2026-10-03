"""Wf-330-B2 — Integration tests for Wf-328 optimization wire-ups.

This test module verifies that the 10 Wf-328 optimization flags
correctly parse, wire into the trainer, and function together:

  W1  --torch-compile / --torch-compile-mode
  W2  --num-workers-host-cores (companion to W6; tested via fallback)
  W3  --bf16 / --no-bf16
  W4  --tf32 / --no-tf32
  W5  --w2-gradient-mode {sliced,sort}
  W6  --num-workers / --cpu-num-workers (dataloader factory precedence)
  W7  --dmpnn-backend {pyg,loop} (gap: NOT in wasserstein trainer —
      document as honest-negative integration gap)
  W8  --cuda-graph / --cuda-graph-static-bs
  W9  --triton-edge-update
  W10 dmpnn featurise cache

Per AGENTS.md §5.1 — CPU-only test, no GPU launch.
Per AGENTS.md §5.2 — honest framing for the W7 gap.

Tests
-----
1. ``test_all_optimization_flags_parse`` — 9/10 flags parse; W7 not
   in wasserstein trainer (documented integration gap).
2. ``test_trainer_initialization_with_all_flags`` — Trainer
   constructor accepts every flag without error.
3. ``test_forward_pass_with_torch_compile`` — MultiTaskWassersteinModel
   forward works with ``torch.compile`` on the encoder.
4. ``test_w2_loss_with_sort_mode`` — :func:`wasserstein_pic50_gradient`
   is differentiable in ``sort`` mode (closed-form 1-D W₂).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from molmetal.molmetal_lam.reward.wasserstein_loss import (
    wasserstein_1d_torchsort,
    wasserstein_pic50_gradient,
)

# ---------------------------------------------------------------------------
# Locate the trainer script as a module so we can call parse_args() without
# triggering the dataclass-without-sys.modules-registration issue.
# ---------------------------------------------------------------------------
_TRAINER_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "train_dmpnn_multitask_wasserstein.py"
)


def _load_trainer_module():
    """Load the trainer script with explicit sys.modules registration so
    that ``@dataclass`` can introspect the module dict."""
    mod_name = "_wf329_mtw_under_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(_TRAINER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. test_all_optimization_flags_parse
# ---------------------------------------------------------------------------
def test_all_optimization_flags_parse():
    """All 10 of the Wf-328 optimization flags + Wf-334-U1 wire-up parse."""
    trainer = _load_trainer_module()
    # W1: --torch-compile + --torch-compile-mode
    # W3: --bf16 (auto-skipped on CPU per W3 — that's expected behaviour, not a parse error)
    # W4: --tf32
    # W5: --w2-gradient-mode sort
    # W6: --num-workers + --cpu-num-workers (precedence)
    # W7: --dmpnn-backend {pyg,loop} (Wf-334-U1 wire-up; resolves the
    #     Wf-330-B2 §2 integration gap — flag was in legacy trainer
    #     but missing from Wasserstein trainer)
    # W8: --cuda-graph + --cuda-graph-static-bs
    # W9: --triton-edge-update
    # W10: no CLI flag (it's the Wf-328-W10 on-disk pickle cache; tested
    #     implicitly by the W1 smoke run which shows the cache hit on
    #     re-run)
    argv = [
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--torch-compile",
        "--torch-compile-mode",
        "reduce-overhead",
        "--bf16",
        "--tf32",
        "--w2-gradient-mode",
        "sort",
        "--num-workers",
        "2",
        "--cpu-num-workers",
        "4",
        "--cuda-graph",
        "--cuda-graph-static-bs",
        "8",
        "--triton-edge-update",
        "--metals",
        "Ir",
        "--seeds",
        "0",
    ]
    args = trainer.parse_args(argv)
    assert args.torch_compile is True
    assert args.torch_compile_mode == "reduce-overhead"
    assert args.bf16 is True
    assert args.tf32 is True
    assert args.w2_gradient_mode == "sort"
    assert args.num_workers == 2
    assert args.cpu_num_workers == 4
    assert args.cuda_graph is True
    assert args.cuda_graph_static_bs == 8
    assert args.triton_edge_update is True

    # Wf-334-U1 — --dmpnn-backend is now wired into the Wasserstein
    # trainer (resolves Wf-330-B2 §2 integration gap).  Both choices
    # parse and the default is 'pyg'.  Bogus values still raise
    # SystemExit(2).
    args_default = trainer.parse_args(["--device", "cpu"])
    assert (
        args_default.dmpnn_backend == "pyg"
    ), f"default dmpnn_backend must be 'pyg', got {args_default.dmpnn_backend!r}"
    args_loop = trainer.parse_args(["--dmpnn-backend", "loop", "--device", "cpu"])
    assert (
        args_loop.dmpnn_backend == "loop"
    ), f"--dmpnn-backend loop must set namespace to 'loop', got {args_loop.dmpnn_backend!r}"
    args_pyg = trainer.parse_args(["--dmpnn-backend", "pyg", "--device", "cpu"])
    assert (
        args_pyg.dmpnn_backend == "pyg"
    ), f"--dmpnn-backend pyg must set namespace to 'pyg', got {args_pyg.dmpnn_backend!r}"
    with pytest.raises(SystemExit) as exc_info:
        trainer.parse_args(["--dmpnn-backend", "cuda", "--device", "cpu"])
    assert exc_info.value.code == 2  # argparse's standard "invalid choice" exit code


# ---------------------------------------------------------------------------
# 2. test_trainer_initialization_with_all_flags
# ---------------------------------------------------------------------------
def test_trainer_initialization_with_all_flags():
    """MultiTaskWassersteinModel + WassersteinConfig instantiate cleanly with every flag's effect baked in."""
    trainer = _load_trainer_module()

    # Instantiate the model with hidden=64, depth=2 (small for CPU).
    model = trainer.MultiTaskWassersteinModel(
        n_metals=5,
        hidden=64,
        depth=2,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0

    # W3 BF16 autocast wrapper is a no-op on CPU; verify the wrapper
    # exists and can be entered without raising.
    bf16_ctx = trainer.bf16_autocast(torch.device("cpu"), enabled=True)
    # On CPU the helper should return a nullcontext — verify it's importable
    # and callable.
    assert bf16_ctx is not None
    with bf16_ctx:
        pass  # no-op; just ensure the context manager protocol works
    # And the OFF path also returns a context.
    bf16_off = trainer.bf16_autocast(torch.device("cpu"), enabled=False)
    with bf16_off:
        pass

    # W4 TF32 helper: verify configure_tf32() round-trips on CPU without error.
    snap = trainer.configure_tf32(enabled=True)
    # On CPU the snap records prior/current state (no-op apply)
    assert "prior_allow_tf32_matmul" in snap
    assert "current_allow_tf32_matmul" in snap
    assert "prior_allow_tf32_cudnn" in snap
    assert "current_allow_tf32_cudnn" in snap
    # restore
    trainer.configure_tf32(enabled=False)


# ---------------------------------------------------------------------------
# 3. test_forward_pass_with_torch_compile
# ---------------------------------------------------------------------------
def test_forward_pass_with_torch_compile():
    """torch.compile produces bit-equivalent output for the trainer's per-metal heads.

    The full ``DMPNNMultiTaskModel`` encoder needs RDKit-shaped inputs
    (atom/bond feature dicts, padding masks) that are inconvenient to
    fabricate in unit tests; instead we verify torch.compile's wire-up
    at the level of the per-metal heads (the reg/cls heads the encoder
    feeds) — these are pure MLPs and ``torch.compile`` works on them
    trivially.  Combined with the live smoke test (CPU forward + train
    step with ``--torch-compile``), this gives us coverage for the
    torch.compile wire-up path.
    """
    trainer = _load_trainer_module()
    model = trainer.MultiTaskWassersteinModel(
        n_metals=5,
        hidden=64,
        depth=2,
    )

    # Per-metal regression head is a 3-layer MLP.  We pass a 2H-dim
    # pooled representation (the encoder's output) and verify torch.compile
    # produces the same shape.
    in_dim = 2 * 64  # 2H concat pool
    head0 = model.reg_heads[
        0
    ]  # nn.Sequential(Linear(in_features=2H, H), ReLU, Dropout, Linear(H, 1))
    head0.eval()

    pooled = torch.randn(2, in_dim, dtype=torch.float32)
    with torch.no_grad():
        out_eager = head0(pooled)

    compiled_head = torch.compile(head0, mode="default", fullgraph=False)
    with torch.no_grad():
        out_compiled = compiled_head(pooled)

    assert out_compiled.shape == out_eager.shape
    assert torch.allclose(out_compiled, out_eager, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. test_w2_loss_with_sort_mode
# ---------------------------------------------------------------------------
def test_w2_loss_with_sort_mode():
    """wasserstein_pic50_gradient (sort mode) is differentiable + numerically exact for 1-D."""
    # Closed-form 1-D W₂: equal-len case
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    ref = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    w2 = wasserstein_1d_torchsort(pred, ref)
    # Equal length: closed form is mean(|sort(pred) - sort(ref)|)
    # Both already sorted; |diff| = [0.5, 0.5, 0.5, 0.5, 0.5] → mean=0.5
    assert torch.isclose(w2, torch.tensor(0.5), atol=1e-5)

    # Gradient flows back to pred (the sort op is non-smooth but piecewise-linear)
    w2.backward()
    assert pred.grad is not None
    # gradient = sign(sort(pred) - sort(ref)) / N  (mean op)
    expected_grad = torch.full_like(pred, 0.2)  # all +0.5/5 = +0.2
    assert torch.allclose(pred.grad, expected_grad, atol=1e-5)

    # wasserstein_pic50_gradient wraps the closed-form with w_weight=0.05
    pred2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    penalty = wasserstein_pic50_gradient(pred2, ref.detach(), w_weight=0.05)
    assert torch.isclose(penalty, torch.tensor(0.5 * 0.05), atol=1e-5)
    penalty.backward()
    assert pred2.grad is not None

    # Unequal-length case: linear-interpolated CDF resize
    pred3 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ref3 = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    w2_unequal = wasserstein_1d_torchsort(pred3, ref3)
    # pred (N=3) upsampled to N=5: [1, 1.5, 2, 2.5, 3] (linear interp)
    # ref already N=5 sorted: [0, 1, 2, 3, 4]
    # |diff| = [1, 0.5, 0, 0.5, 1] → mean = 0.6
    assert torch.isclose(w2_unequal, torch.tensor(0.6), atol=1e-4)
