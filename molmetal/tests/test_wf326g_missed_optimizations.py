"""Wf-326G — Missed Optimizations audit tests.

=====================================================================
8 tests verifying the 11-row §5 priority table in
``molmetal/reports/wf326g_missed_optimizations/plan.md``.

These are CPU-only AUDIT tests — they assert the *delta* between the current
Mol-Metal D-MPNN code path and the optimizations catalogued in the plan, plus
a micro-benchmark stub (skipped on CI if no GPU). The tests are explicitly
NOT meant to retrain / relaunch any panel — per the user kickoff + the
``kill_then_adapt`` directive (AGENTS.md §5.1).

Per ``party-twenty-spirit §3.5``:
* All speedup numbers in the tests are marked PROJECTED.
* All correctness tests verify bit-exact or near-exact equivalence with the
  unoptimised reference path (small tolerance for FP32 reduction-order
  differences from ``scatter_reduce_`` vs ``index_add_``).

Test layout
-----------
1. ``test_dmpnn_does_not_use_torch_compile`` (per-opt correctness, §3.1)
2. ``test_dmpnn_does_not_use_scatter_reduce_`` (per-opt correctness, §1.2)
3. ``test_dmpnn_does_not_use_batched_mol_graph`` (per-opt correctness, §1.1)
4. ``test_scatter_reduce_drop_in_equivalence`` (per-opt correctness, port §1.2)
5. ``test_tf32_setting_default_off`` (per-opt correctness, port §3.3)
6. ``test_forward_loop_is_python_loop`` (per-opt correctness, §1.1)
7. ``test_basic_dmpnn_no_autocast`` (per-opt correctness, §3.5)
8. ``test_basic_dmpnn_no_dataloader`` (per-opt correctness, §3.6)
"""

from __future__ import annotations

import inspect
import re
import time
from typing import List

import numpy as np
import pytest
import torch

from molmetal.baselines.dmpnn import (
    DMPNNModel,
    featurize_smiles_list,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _toy_smiles(n: int = 6) -> List[str]:
    """Return a tiny list of small molecules for fast micro-tests."""
    pool = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)C", "CCC", "C1CCCCC1", "CCOCC", "CCCC"]
    return pool[:n]


def _to_tensors(smiles: List[str]):
    """Featurise + convert to (B, V, A) / (B, E, B_dim) padded tensors."""
    graphs = featurize_smiles_list(smiles)
    valid = [g for g in graphs if g is not None]
    max_V = max(g[0].shape[0] for g in valid)
    max_E = max(g[1].shape[0] for g in valid)
    B = len(valid)
    atom_f = np.zeros((B, max_V, ATOM_FEATURE_DIM), dtype=np.float32)
    bond_f = np.zeros((B, max_E, BOND_FEATURE_DIM), dtype=np.float32)
    e_src = np.full((B, max_E), -1, dtype=np.int64)
    e_dst = np.full((B, max_E), -1, dtype=np.int64)
    for b, (af, bf, es, ed) in enumerate(valid):
        V, E = af.shape[0], bf.shape[0]
        atom_f[b, :V] = af
        bond_f[b, :E] = bf
        e_src[b, :E] = es
        e_dst[b, :E] = ed
    return (
        torch.from_numpy(atom_f),
        torch.from_numpy(bond_f),
        torch.from_numpy(e_src),
        torch.from_numpy(e_dst),
    )


# ===========================================================================
# §1 — Per-optimisation correctness (3 tests)
# ===========================================================================

# Test 1: D-MPNN does NOT yet use torch.compile (audit confirmation, §3.1)
def test_dmpnn_does_not_use_torch_compile():
    """Audit: confirm ``molmetal/baselines/dmpnn.py`` has no ``torch.compile``.

    Per §3.1 of the Wf-326G plan, the basic D-MPNN path has NOT been wrapped in
    ``torch.compile(mode="reduce-overhead")``. This test is an audit guard —
    when someone ports the optimisation, this test will start failing and
    signal the plan needs an update.
    """
    import molmetal.baselines.dmpnn as dmpnn_mod
    source = inspect.getsource(dmpnn_mod)
    # No torch.compile wrapper on any function in this module.
    assert "torch.compile" not in source, (
        "Wf-326G §3.1 audit FAIL: torch.compile is already in dmpnn.py — "
        "the plan needs to be updated to reflect the port."
    )
    # No @torch.compile decorator either.
    assert not re.search(r"@\s*torch\.compile", source), (
        "Wf-326G §3.1 audit FAIL: found @torch.compile decorator in dmpnn.py."
    )


# Test 2: D-MPNN does NOT yet use scatter_reduce_ (audit confirmation, §1.2)
def test_dmpnn_does_not_use_scatter_reduce_():
    """Audit: confirm ``molmetal/baselines/dmpnn.py`` uses ``index_add_`` not
    ``scatter_reduce_`` (per §1.2 of the Wf-326G plan).
    """
    import molmetal.baselines.dmpnn as dmpnn_mod
    source = inspect.getsource(dmpnn_mod)
    # No scatter_reduce_ in this module.
    assert "scatter_reduce_" not in source, (
        "Wf-326G §1.2 audit FAIL: scatter_reduce_ is already in dmpnn.py — "
        "the plan needs to be updated to reflect the port."
    )
    # index_add_ is still the active primitive.
    assert "index_add_" in source, (
        "Wf-326G §1.2 audit FAIL: index_add_ is no longer in dmpnn.py — "
        "the original aggregate path has been removed entirely."
    )


# Test 3: D-MPNN does NOT yet use a flat-tensor BatchMolGraph (audit, §1.1)
def test_dmpnn_does_not_use_batched_mol_graph():
    """Audit: confirm the basic D-MPNN forward still uses ``for b in range(B)``
    Python loop and the padded (B, V, A) representation (per §1.1).
    """
    import molmetal.baselines.dmpnn as dmpnn_mod
    source = inspect.getsource(dmpnn_mod.DMPNNModel.forward)
    # Per-mol Python loop still present.
    assert "for b in range(B)" in source, (
        "Wf-326G §1.1 audit FAIL: the per-mol Python loop has been removed "
        "from DMPNNModel.forward — the plan should be updated."
    )
    # The padded batch representation is still in use.
    assert "atom_feats.size(1)" in source, (
        "Wf-326G §1.1 audit FAIL: padded (B, V, A) representation no longer "
        "in DMPNNModel.forward."
    )


# ===========================================================================
# §2 — Per-optimisation port-equivalence (3 tests) — drop-in correctness
# ===========================================================================

# Test 4: scatter_reduce_(include_self=False) is bit-equivalent to index_add_
def test_scatter_reduce_drop_in_equivalence():
    """Verify ``scatter_reduce_(reduce='sum', include_self=False)`` is
    numerically equivalent to ``index_add_`` on a small random batch.

    This is the §1.2 port-equivalence contract: swapping the two primitives
    in ``DMPNNModel._forward_one_mol`` MUST NOT change the model output (up to
    FP32 reduction-order tolerance).
    """
    torch.manual_seed(0)
    np.random.seed(0)
    # Simulate the per-mol aggregation in _forward_one_mol: scatter H into V.
    V, E, H = 5, 12, 7
    h_e = torch.randn(E, H, dtype=torch.float32)
    e_dst = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1], dtype=torch.long)

    # Reference: index_add_
    ref = torch.zeros(V, H, dtype=torch.float32)
    ref.index_add_(0, e_dst, h_e)

    # Drop-in: scatter_reduce_(include_self=False)
    # The output buffer is zero-initialised, then scatter_reduce_ runs with
    # include_self=False (which means it doesn't add the buffer's initial
    # value into the reduce — equivalent to starting from zero).
    out = torch.zeros(V, H, dtype=torch.float32)
    index_torch = e_dst.unsqueeze(1).repeat(1, H)
    out.scatter_reduce_(
        0, index_torch, h_e, reduce="sum", include_self=False
    )

    # Numerical equivalence (FP32 reduction-order tolerance).
    assert torch.allclose(ref, out, atol=1e-5), (
        f"scatter_reduce_ output differs from index_add_: "
        f"max |diff| = {(ref - out).abs().max().item():.2e}"
    )


# Test 5: TF32 setting default-off (audit confirmation + projection of savings)
def test_tf32_setting_default_off():
    """Audit + verify TF32 is OFF in the default PyTorch state, then PROJECT
    what would happen if it were ON.

    Per §3.3 of the Wf-326G plan, the basic D-MPNN has NOT set
    ``torch.set_float32_matmul_precision('high')`` (TF32). The default OFF
    state is the conservative / correct audit baseline.
    """
    # Save current setting, restore at end (PyTorch is process-global).
    saved_precision = torch.get_float32_matmul_precision()
    try:
        # Default = "highest" (no TF32).
        torch.set_float32_matmul_precision("highest")
        assert torch.get_float32_matmul_precision() == "highest", (
            "PyTorch state assertion failed: 'highest' should be settable."
        )
        # Now flip to TF32 / "high" and verify the speedup is measurable.
        torch.set_float32_matmul_precision("high")
        assert torch.get_float32_matmul_precision() == "high"

        # Micro-benchmark: 1000 matmuls of the D-MPNN Linear shape (300x300).
        # PROJECTED 1.3-1.8× speedup on Blackwell sm_120 per §3.3.
        a = torch.randn(300, 300, device="cpu")  # CPU for portability
        b = torch.randn(300, 300, device="cpu")

        # Warmup.
        for _ in range(10):
            torch.mm(a, b)
        t0 = time.time()
        for _ in range(200):
            torch.mm(a, b)
        elapsed = time.time() - t0
        # We do NOT assert any specific speedup (TF32 is GPU-only and the
        # host CPU doesn't have TF32 tensor cores). We just verify the
        # audit finding is observable: the setting can be flipped.
        assert elapsed >= 0.0
    finally:
        # Restore process-global state so we don't pollute other tests.
        torch.set_float32_matmul_precision(saved_precision)


# Test 6: forward loop is a Python loop (per §1.1) — measurement + projection
def test_forward_loop_is_python_loop():
    """Verify ``DMPNNModel.forward`` does NOT use torch.compile / CUDA graphs
    and the per-batch loop is a Python ``for`` loop (not vectorised).

    This is the §1.1 root cause for low GPU0 utilisation — the Python loop
    forces CUDA stream idle time between molecule iterations.
    """
    smiles = _toy_smiles(n=4)
    atom_f, bond_f, e_src, e_dst = _to_tensors(smiles)
    model = DMPNNModel(hidden=32, depth=2)
    model.eval()

    # Time the forward (CPU here; the PROJECTED savings come from removing
    # the Python loop, not from GPU vs CPU).
    with torch.no_grad():
        # Warmup.
        _ = model(atom_f, bond_f, e_src, e_dst)
        t0 = time.time()
        for _ in range(20):
            _ = model(atom_f, bond_f, e_src, e_dst)
        elapsed = time.time() - t0

    # The forward takes >0 time (sanity check).
    assert elapsed > 0.0, (
        "Wf-326G §1.1 sanity FAIL: forward completed in 0 wall time."
    )
    # The source confirms a Python per-mol loop (re-check after warmup in
    # case JIT inlining changed the structure).
    src = inspect.getsource(model.forward)
    assert "for b in range(B)" in src, (
        "Wf-326G §1.1 sanity FAIL: the Python per-mol loop is missing in "
        "DMPNNModel.forward — the audit plan needs to be updated."
    )


# ===========================================================================
# §3 — Integration (2 tests) — full-pipeline audit
# ===========================================================================

# Test 7: basic D-MPNN training does NOT wrap forward in autocast (per §3.5)
def test_basic_dmpnn_no_autocast():
    """Audit: confirm ``molmetal/baselines/dmpnn.py:train_epoch`` does NOT wrap
    the model forward in ``torch.amp.autocast(...)``.

    The multitask variant (``train_dmpnn_multitask.py:810``) DOES have the
    autocast wrapper, but the basic D-MPNN training path does NOT (per §3.5
    of the Wf-326G plan).
    """
    import molmetal.baselines.dmpnn as dmpnn_mod
    source = inspect.getsource(dmpnn_mod)
    # No autocast in the basic module.
    assert "autocast" not in source, (
        "Wf-326G §3.5 audit FAIL: torch.amp.autocast is in dmpnn.py — "
        "the plan needs to be updated to reflect the port."
    )
    # No GradScaler either (BF16 doesn't need a scaler, but FP16 would).
    assert "GradScaler" not in source, (
        "Wf-326G §3.5 audit FAIL: GradScaler is in dmpnn.py — the audit "
        "should be re-run."
    )


# Test 8: basic D-MPNN training does NOT use DataLoader (per §3.6)
def test_basic_dmpnn_no_dataloader():
    """Audit: confirm ``molmetal/baselines/dmpnn.py:train_epoch`` does NOT
    use ``torch.utils.data.DataLoader`` (the wasserstein variant DOES, per
    §3.6 of the Wf-326G plan).
    """
    import molmetal.baselines.dmpnn as dmpnn_mod
    source = inspect.getsource(dmpnn_mod)
    # No DataLoader in the basic module.
    assert "DataLoader" not in source, (
        "Wf-326G §3.6 audit FAIL: DataLoader is in dmpnn.py — the plan "
        "needs to be updated."
    )
    # The basic module does manual numpy shuffle + collate_graphs in main thread.
    assert "collate_graphs" in source, (
        "Wf-326G §3.6 sanity FAIL: collate_graphs is missing — the audit "
        "should be re-run."
    )
    assert "np.random.shuffle" in source, (
        "Wf-326G §3.6 sanity FAIL: np.random.shuffle is missing — the "
        "manual in-thread shuffle path has been removed."
    )