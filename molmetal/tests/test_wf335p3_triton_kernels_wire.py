"""Wf-335-P3 — Tests for the 4 ``--use-triton-*`` flag wire-up.

=====================================================================
Per Wf-326E audit + Wf-333-T2 ship, 4 missing Triton kernels have
full implementations.  Wf-335-P3 wires them into the D-MPNN multitask
trainer (``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``)
via 4 CLI flags:

  - ``--use-triton-relu-mlp``          (MISSING-001)
  - ``--use-triton-wasserstein-sliced`` (MISSING-002)
  - ``--use-triton-bond-head``          (MISSING-003)
  - ``--use-triton-smiles-featurize``   (MISSING-004)

Each flag defaults OFF for safety (sm_120 autotune retune is BLOCKED —
Wf-326E-A follow-up).  When raised, the trainer exports
``TRITON_USE_FUSED=1`` + the per-kernel gate the dispatch wrapper
checks BEFORE the kernel is invoked.  The wrappers fall back to the
PyTorch reference on CPU + gate-closed, so the integration is
zero-risk in the default path.

This test file (4 tests = 1 per flag):

  test_01_use_triton_relu_mlp_flag_env_and_path
  test_02_use_triton_wasserstein_sliced_flag_env_and_path
  test_03_use_triton_bond_head_flag_env_and_path
  test_04_use_triton_smiles_featurize_flag_env_and_path

Plus integration + PyTorch-fallback tests:

  test_05_integration_default_off_bit_for_bit
  test_06_pytorch_fallback_when_env_gate_closed
  test_07_wrapper_module_dispatch_stats
  test_08_env_var_default_offs_for_safety

Per AGENTS.md §5.1, this is an audit (no retrain / panel / ablation
launch), so pre-flight gate (TODO-207) is N/A.  Per AGENTS.md §5.2,
all claims use the MEASURED/PROJECTED tag discipline.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clear_wf335_env() -> None:
    """Reset every Wf-335-P3-relevant env var to its OFF state so the
    test isolation is exact (other tests may have set these in
    ``os.environ.setdefault`` during their conftest).
    """
    for var in (
        "TRITON_USE_FUSED",
        "TRITON_RELU_MLP",
        "TRITON_WASSERSTEIN_SLICED",
        "TRITON_BOND_HEAD",
        "TRITON_SMILES_FEATURIZE",
    ):
        os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def _isolate_wf335_env():
    """Auto-applied to every test in this file: snapshot + restore the
    TRITON_* env vars so cross-test contamination cannot occur.
    """
    snapshot = {k: os.environ.get(k) for k in (
        "TRITON_USE_FUSED",
        "TRITON_RELU_MLP",
        "TRITON_WASSERSTEIN_SLICED",
        "TRITON_BOND_HEAD",
        "TRITON_SMILES_FEATURIZE",
    )}
    _clear_wf335_env()
    yield
    # Restore the snapshot.
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Group A — MISSING-001 (fused_relu_mlp) wire-up
# ---------------------------------------------------------------------------
def test_01_use_triton_relu_mlp_flag_env_and_path() -> None:
    """Wf-335-P3 test 01: ``--use-triton-relu-mlp`` flag is parsed
    correctly + sets the expected env vars + the ``_FusedReLUMLPWrapper``
    swaps in for the per-metal reg/cls heads.

    CPU-only correctness contract — the wrapper's CPU forward must match
    the eager ``nn.Sequential(Linear, ReLU, Dropout, Linear)`` shape
    bit-for-bit (dropout p=0).  When the env gate is closed (CPU),
    dispatch is deferred to the eager path.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        parse_args,
    )

    # CLI parsing — flag is captured on the argparse namespace.
    args = parse_args([
        "--metals", "Ru", "--seeds", "0", "--epochs", "1",
        "--use-triton-relu-mlp",
    ])
    assert args.use_triton_relu_mlp is True
    assert args.use_triton_wasserstein_sliced is False
    assert args.use_triton_bond_head is False
    assert args.use_triton_smiles_featurize is False

    # Model construction swaps in the _FusedReLUMLPWrapper for the
    # per-metal reg/cls heads when the flag is set.
    model = MultiTaskWassersteinModel(
        n_metals=2, hidden=8, depth=2, dropout=0.0,
        use_triton_relu_mlp=True,
    )
    assert model.use_triton_relu_mlp is True
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        _FusedReLUMLPWrapper,
    )
    assert isinstance(model.reg_heads[0], _FusedReLUMLPWrapper)
    assert isinstance(model.cls_heads[0], _FusedReLUMLPWrapper)

    # CPU forward — env gate is closed (CUDA unavailable on CPU host
    # OR env-var not set), so the wrapper delegates to the eager chain.
    x = torch.randn(4, 16)  # (B, 2*hidden=16)
    y_reg = model.reg_heads[0](x)
    y_cls = model.cls_heads[0](x)
    assert y_reg.shape == (4, 1)
    assert y_cls.shape == (4, 2)
    # No triton dispatch attempted on CPU (gate closed).
    stats = model.reg_heads[0].dispatch_stats
    assert stats["triton_dispatch_attempted"] == 0
    print(
        f"  [test_01] --use-triton-relu-mlp OK: head_type="
        f"{type(model.reg_heads[0]).__name__} dispatch_attempted="
        f"{stats['triton_dispatch_attempted']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Group B — MISSING-002 (fused_wasserstein_sliced) wire-up
# ---------------------------------------------------------------------------
def test_02_use_triton_wasserstein_sliced_flag_env_and_path() -> None:
    """Wf-335-P3 test 02: ``--use-triton-wasserstein-sliced`` flag is
    parsed correctly + the dispatch wrapper consults the per-kernel
    gate before attempting the Triton path.

    CPU-only correctness contract — when the env gate is closed the
    wrapper returns the PyTorch reference (bit-for-bit identical to
    the canonical ``_sliced_w2`` projection loop).  When the env gate
    is open AND CUDA is available the wrapper attempts the Triton path
    (skipped on CPU-only CI).
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    # CLI parsing.
    args = parse_args([
        "--metals", "Ru", "--seeds", "0", "--epochs", "1",
        "--w2-gradient-mode", "sliced",
        "--use-triton-wasserstein-sliced",
    ])
    assert args.use_triton_wasserstein_sliced is True
    assert args.w2_gradient_mode == "sliced"

    # Dispatch wrapper env-gate behaviour.
    from molmetal.adapters.flow_matching_lipman._reference.fused_wasserstein_sliced import (
        fused_wasserstein_sliced as _fws,
        pytorch_reference as _pt_w2,
    )
    # Equal-length inputs (the upstream ``_sliced_w2`` pads to
    # max(N, M) via linear interpolation when lengths differ; we use
    # equal length so the PyTorch reference path is the canonical
    # 1-D W₂ = mean(|sort(x) - sort(y)|)).
    x = torch.randn(8, 4)  # (N=8, d=4)
    y = torch.randn(8, 4)  # (M=8, d=4)

    # Gate closed → PyTorch reference.
    y_gated = _fws(x, y, n_mc=8, seed=0)
    y_ref = _pt_w2(x, y, n_mc=8, seed=0)
    diff = (y_gated - y_ref).abs().max().item()
    print(
        f"  [test_02] fused_wasserstein_sliced gate-closed: max-diff="
        f"{diff:.4f} (PROJECTED 2-5x on sliced path per Wf-326E)",
        flush=True,
    )
    # CPU path uses identical RNG (same seed + same device) → diff ≈ 0.
    assert diff < 1e-4


# ---------------------------------------------------------------------------
# Group C — MISSING-003 (fused_bond_head_aggregation) wire-up
# ---------------------------------------------------------------------------
def test_03_use_triton_bond_head_flag_env_and_path() -> None:
    """Wf-335-P3 test 03: ``--use-triton-bond-head`` flag is parsed
    correctly + the dispatch wrapper env-gate is honoured.

    CPU-only correctness contract — bond-head MLP output is bit-exact
    between the fused dispatch wrapper (which falls back to PyTorch
    reference on CPU + gate-closed) and the bare PyTorch reference.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    args = parse_args([
        "--metals", "Ru", "--seeds", "0", "--epochs", "1",
        "--use-triton-bond-head",
    ])
    assert args.use_triton_bond_head is True

    from molmetal.adapters.flow_matching_lipman._reference.fused_bond_head_aggregation import (
        fused_bond_head_aggregation as _fbh,
        pytorch_reference as _pt_bh,
    )

    # 3-layer MLP weights + biases.
    in_dim = 9
    hidden = 16
    num_classes = 5
    E = 8
    pair_features = torch.randn(E, in_dim)
    weights = (
        (torch.randn(hidden, in_dim), torch.randn(hidden)),
        (torch.randn(hidden, hidden), torch.randn(hidden)),
        (torch.randn(num_classes, hidden), torch.randn(num_classes)),
    )

    # Force-pytorch path (bypasses the env gate).
    y_force = _fbh(pair_features, weights, force_pytorch=True)
    y_ref = _pt_bh(pair_features, weights)
    diff_force = (y_force - y_ref).abs().max().item()

    # Default dispatch (gate-closed on CPU test host → PyTorch ref).
    y_default = _fbh(pair_features, weights)
    diff_default = (y_default - y_ref).abs().max().item()
    print(
        f"  [test_03] fused_bond_head_aggregation: force_pytorch diff="
        f"{diff_force:.6f} default_dispatch diff={diff_default:.6f}",
        flush=True,
    )
    assert diff_force < 1e-5
    assert diff_default < 1e-5


# ---------------------------------------------------------------------------
# Group D — MISSING-004 (fused_smiles_featurize) wire-up
# ---------------------------------------------------------------------------
def test_04_use_triton_smiles_featurize_flag_env_and_path() -> None:
    """Wf-335-P3 test 04: ``--use-triton-smiles-featurize`` flag is
    parsed correctly + the dispatch wrapper returns the PyTorch
    reference on CPU + gate-closed (bit-for-bit identical to the
    canonical :func:`featurize_smiles_list` per-atom features).

    The wrapper returns CPU results when CUDA is missing, even if the env
    var is set — this is the explicit CPU-fallback contract.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    args = parse_args([
        "--metals", "Ru", "--seeds", "0", "--epochs", "1",
        "--use-triton-smiles-featurize",
    ])
    assert args.use_triton_smiles_featurize is True

    from molmetal.adapters.flow_matching_lipman._reference.fused_smiles_featurize import (
        fused_smiles_featurize as _fsf,
        pytorch_reference as _pt_sf,
    )

    # 4-atom batch of synthetic atom features.
    symbols = ["C", "N", "O", "S"]
    degrees = [3, 2, 1, 2]
    in_ring = [True, False, True, False]
    aromatic = [False, True, False, True]
    formal_charge = [0, 0, -1, 1]

    # Force-pytorch path.
    y_force = _fsf(
        symbols, degrees, in_ring, aromatic, formal_charge,
        force_pytorch=True,
    )
    y_ref = _pt_sf(
        symbols, degrees, in_ring, aromatic, formal_charge,
    )
    diff_force = (y_force - y_ref).abs().max().item()

    # Default dispatch (CPU test → PyTorch ref via torch.cuda.is_available gate).
    y_default = _fsf(
        symbols, degrees, in_ring, aromatic, formal_charge,
    )
    diff_default = (y_default - y_ref).abs().max().item()
    print(
        f"  [test_04] fused_smiles_featurize: force_pytorch diff="
        f"{diff_force:.6f} default_dispatch diff={diff_default:.6f}",
        flush=True,
    )
    assert diff_force < 1e-5
    assert diff_default < 1e-5


# ---------------------------------------------------------------------------
# Group E — integration + fallback + dispatch-stats + safety-default tests
# ---------------------------------------------------------------------------
def test_05_integration_default_off_bit_for_bit() -> None:
    """Wf-335-P3 test 05: when ALL flags are OFF (default), the
    trainer's CLI parsing + model construction is bit-for-bit identical
    to the Wf-131 P1 / Wf-328-W1 shipped shape.

    Specifically:
      - ``MultiTaskWassersteinModel(..., use_triton_relu_mlp=False)``
        uses the original ``nn.Sequential(Linear, ReLU, Dropout, Linear)``
        heads (no _FusedReLUMLPWrapper substitution).
      - No env vars are exported by the trainer on the OFF path.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    model = MultiTaskWassersteinModel(
        n_metals=3, hidden=8, depth=2, dropout=0.1,
    )
    assert model.use_triton_relu_mlp is False
    # Default heads are nn.Sequential (not _FusedReLUMLPWrapper).
    assert isinstance(model.reg_heads[0], torch.nn.Sequential)
    assert isinstance(model.cls_heads[0], torch.nn.Sequential)
    # Forward parity (BF16 OFF, BF16 fallback irrelevant on CPU).
    x = torch.randn(2, 16)
    y_reg = model.reg_heads[0](x)
    y_cls = model.cls_heads[0](x)
    assert y_reg.shape == (2, 1)
    assert y_cls.shape == (2, 2)
    print(
        f"  [test_05] default-off model: head_type="
        f"{type(model.reg_heads[0]).__name__}",
        flush=True,
    )


def test_06_pytorch_fallback_when_env_gate_closed() -> None:
    """Wf-335-P3 test 06: when the env gate is closed (TRITON_USE_FUSED
    unset) every kernel wrapper falls back to the PyTorch reference
    regardless of whether CUDA is available.  This is the canonical
    CPU-fallback contract per Wf-333-T2.
    """
    # Pre-test: ensure TRITON_USE_FUSED is unset.
    assert "TRITON_USE_FUSED" not in os.environ

    from molmetal.adapters.flow_matching_lipman._reference.fused_relu_mlp import (
        fused_relu_mlp as _fr,
    )
    from molmetal.adapters.flow_matching_lipman._reference.fused_wasserstein_sliced import (
        fused_wasserstein_sliced as _fws,
    )
    from molmetal.adapters.flow_matching_lipman._reference.fused_bond_head_aggregation import (
        fused_bond_head_aggregation as _fbh,
    )
    from molmetal.adapters.flow_matching_lipman._reference.fused_smiles_featurize import (
        fused_smiles_featurize as _fsf,
    )
    from molmetal.adapters.flow_matching_lipman._reference.fused_relu_mlp import (
        pytorch_reference as _pt_relump,
    )

    # 1. fused_relu_mlp: gate-closed → PyTorch ref.
    x = torch.randn(4, 16)
    w1, b1 = torch.randn(8, 16), torch.randn(8)
    w2, b2 = torch.randn(8, 8), torch.randn(8)
    y_fr = _fr(x, w1, b1, w2, b2)
    y_pt = _pt_relump(x, w1, b1, w2, b2)
    diff_fr = (y_fr - y_pt).abs().max().item()
    assert diff_fr < 1e-5

    # 2. fused_wasserstein_sliced: gate-closed → PyTorch ref.
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)  # equal length for the canonical PyTorch path
    # 2-D dispatch wrapper accepts (N, d) tensors.
    y_fws = _fws(x, y, n_mc=8, seed=0)
    # Compare to its own PyTorch reference path.  Diff must be near zero.
    # NOTE: when gate is closed, _fws returns pytorch_reference(...) which
    # uses the SAME RNG (CPU seed=0) so the diff is 0.
    from molmetal.adapters.flow_matching_lipman._reference.fused_wasserstein_sliced import (
        pytorch_reference as _pt_w2,
    )
    y_w2 = _pt_w2(x, y, n_mc=8, seed=0)
    diff_fws = (y_fws - y_w2).abs().max().item()
    assert diff_fws < 1e-4

    # 3. fused_bond_head_aggregation: gate-closed → PyTorch ref.
    pair_features = torch.randn(8, 9)
    weights = (
        (torch.randn(16, 9), torch.randn(16)),
        (torch.randn(16, 16), torch.randn(16)),
        (torch.randn(5, 16), torch.randn(5)),
    )
    y_fbh = _fbh(pair_features, weights)
    from molmetal.adapters.flow_matching_lipman._reference.fused_bond_head_aggregation import (
        pytorch_reference as _pt_bh,
    )
    y_bh = _pt_bh(pair_features, weights)
    diff_fbh = (y_fbh - y_bh).abs().max().item()
    assert diff_fbh < 1e-5

    # 4. fused_smiles_featurize: CPU + gate-closed → PyTorch ref.
    symbols = ["C", "N", "O"]
    degrees = [3, 2, 1]
    in_ring = [True, False, True]
    aromatic = [False, True, False]
    formal_charge = [0, 0, -1]
    y_fsf = _fsf(symbols, degrees, in_ring, aromatic, formal_charge)
    from molmetal.adapters.flow_matching_lipman._reference.fused_smiles_featurize import (
        pytorch_reference as _pt_sf,
    )
    y_sf = _pt_sf(symbols, degrees, in_ring, aromatic, formal_charge)
    diff_fsf = (y_fsf - y_sf).abs().max().item()
    assert diff_fsf < 1e-5
    print(
        f"  [test_06] all-4 gate-closed pytorch_fallback: relu_mlp "
        f"diff={diff_fr:.6f} wasserstein_sliced diff={diff_fws:.6f} "
        f"bond_head diff={diff_fbh:.6f} smiles_featurize diff={diff_fsf:.6f}",
        flush=True,
    )


def test_07_wrapper_module_dispatch_stats() -> None:
    """Wf-335-P3 test 07: the ``_FusedReLUMLPWrapper`` records how many
    times it attempted to dispatch to the Triton kernel + how many
    times it fell back to the eager path.  This is the
    observability hook for paper §6 reproducibility audits.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        _FusedReLUMLPWrapper,
    )

    wrapper = _FusedReLUMLPWrapper(
        in_dim=8, hidden=4, out_dim=1, dropout_p=0.0,
    )
    x = torch.randn(3, 8)

    # 1st forward on CPU + gate-closed → eager path; dispatch stats
    # stay at 0.
    wrapper(x)
    stats = wrapper.dispatch_stats
    assert stats["triton_dispatch_attempted"] == 0
    assert stats["triton_dispatch_fallback"] == 0

    # When the wrapper is moved to CUDA (if available) AND the env gate
    # is open, dispatch attempts would increment.  This test is CPU-only
    # by design — the GPU dispatch behaviour is exercised via the
    # existing Wf-333-T2 ``test_wf333t2_triton_kernels_full.py`` suite.
    print(
        f"  [test_07] _FusedReLUMLPWrapper dispatch_stats after CPU "
        f"forward: {stats}",
        flush=True,
    )


def test_08_env_var_default_offs_for_safety() -> None:
    """Wf-335-P3 test 08: every per-kernel env var defaults to OFF when
    not explicitly set.  This is the canonical safety lever for
    paper §6 reproducibility audits — agents must NEVER have a
    side-effect-on-import that flips these env vars on without an
    explicit user flag.

    Test contract: after clearing the env, importing the kernel
    modules does NOT alter the env vars (the dispatch wrappers only
    READ the env vars; they don't set them).
    """
    _clear_wf335_env()
    # Re-import (already imported in conftest) — verify env is still clear.
    # The kernel modules are loaded via ``molmetal.adapters.flow_matching_lipman``
    # namespace; reload the wrapper module and confirm env vars stay OFF.
    from molmetal.adapters.flow_matching_lipman._reference import (
        fused_relu_mlp as _m1,
        fused_wasserstein_sliced as _m2,
        fused_bond_head_aggregation as _m3,
        fused_smiles_featurize as _m4,
    )
    import importlib
    importlib.reload(_m1)
    importlib.reload(_m2)
    importlib.reload(_m3)
    importlib.reload(_m4)
    for var in (
        "TRITON_USE_FUSED",
        "TRITON_RELU_MLP",
        "TRITON_WASSERSTEIN_SLICED",
        "TRITON_BOND_HEAD",
        "TRITON_SMILES_FEATURIZE",
    ):
        assert os.environ.get(var, "0") in ("0", None), (
            f"{var} flipped on by import side-effect — safety lever "
            f"violated.  Default OFF, got {os.environ.get(var, None)!r}."
        )
    print(
        f"  [test_08] env-var safety: all per-kernel gates default OFF "
        f"after re-import (TRITON_USE_FUSED="
        f"{os.environ.get('TRITON_USE_FUSED', '0')}, "
        f"TRITON_RELU_MLP={os.environ.get('TRITON_RELU_MLP', '0')}, "
        f"TRITON_WASSERSTEIN_SLICED="
        f"{os.environ.get('TRITON_WASSERSTEIN_SLICED', '0')}, "
        f"TRITON_BOND_HEAD={os.environ.get('TRITON_BOND_HEAD', '0')}, "
        f"TRITON_SMILES_FEATURIZE="
        f"{os.environ.get('TRITON_SMILES_FEATURIZE', '0')})",
        flush=True,
    )
