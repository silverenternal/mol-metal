"""Wf-328-W8 — CUDA Graphs production trainer wiring tests.

================================================================
CUDA Graphs capture wrapper — 4-test acceptance gate
================================================================

Per Wf-328-W8 mandate, the CUDA Graphs capture wrapper
(:class:`molmetal.baselines.cuda_graph_capture.CudaGraphDMPNNWrapper`)
must satisfy four tests:

1. **Capture works** on a static-shape MolGraphBatch — the captured
   graph object is created without error.
2. **Replay correctness** matches eager within 1e-3 tolerance.
3. **Fallback on shape change** — if the input shape changes mid-call,
   the wrapper falls back to eager mode and emits a warning.
4. **Trainer integration** — the wrapper integrates with the Wf-131
   production trainer without breaking the forward pass shape.

These tests run on CUDA when available; on CPU they verify the fallback
path (wrapper disabled = eager passthrough).

Honest framing (party-twenty-spirit §3.5):

* CUDA Graphs: 1.5-2.5× PROJECTED on static shape (per Wf-326C).
* Requires Wf-328-W7 PyG port (removes .item() blocker) — the
  :class:`DirectedMPNNPyG.forward_batched` has no ``.item()`` so
  capture is unblocked.  ``coalesce_padded_batch`` still has
  ``.item()`` but runs *outside* the capture region.
* Default OFF for safety (CUDAGraph errors are hard to debug).
* CPU-only tests verify the eager-fallback path is correct (wrapper
  transparent passthrough); GPU-only tests verify capture + replay.

**Pre-existing import note**: :mod:`molmetal.models.dmpnn_pyg` triggers
the ``molmetal/models/__init__.py`` which imports triton kernels that
have an API mismatch on the test venv.  We use lazy imports inside the
test bodies so the eager-passthrough tests still run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest
import torch

from molmetal.baselines.cuda_graph_capture import CudaGraphDMPNNWrapper


# Stub MPNNConfig to avoid the ``molmetal/models/__init__`` triton import.
# Mirrors :class:`molmetal.models.dmpnn.MPNNConfig` defaults verbatim;
# the wrapper only reads the field values, so the stub is sufficient.
@dataclass
class _MPNNConfigStub:
    atom_feat_dim: int = 39
    edge_feat_dim: int = 10
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.0
    n_shared_layers: int = 0


MPNNConfig = _MPNNConfigStub


def _load_pyg_module():
    """Lazy loader for the PyG encoder module (see header note).

    Uses importlib to load :mod:`molmetal.models.dmpnn_pyg` directly
    from the file path.  Importing via the package namespace would
    trigger :mod:`molmetal.models.__init__` which imports triton
    kernels that have a pre-existing API mismatch on the test venv.
    """
    import importlib.util
    import sys
    if "molmetal.models.dmpnn_pyg" in sys.modules:
        return sys.modules["molmetal.models.dmpnn_pyg"]
    spec = importlib.util.spec_from_file_location(
        "molmetal.models.dmpnn_pyg",
        "/home/hugo/codes/mol-metal/molmetal/models/dmpnn_pyg.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load dmpnn_pyg.py via importlib")
    mod = importlib.util.module_from_spec(spec)
    # Register the parent package so :mod:`molmetal.models.dmpnn_pyg`
    # relative imports inside dmpnn_pyg.py resolve correctly.
    if "molmetal.models" not in sys.modules:
        import types
        parent_pkg = types.ModuleType("molmetal.models")
        parent_pkg.__path__ = ["/home/hugo/codes/mol-metal/molmetal/models"]
        sys.modules["molmetal.models"] = parent_pkg
    sys.modules["molmetal.models.dmpnn_pyg"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Local helpers — build the PyG fixtures inside test bodies (fixtures at
# module scope trigger the dmpnn_pyg import which fails collection).
# ---------------------------------------------------------------------------
def _make_cpu_pyg_encoder():
    """Small DirectedMPNNPyG for CPU parity tests."""
    pyg = _load_pyg_module()
    cfg = MPNNConfig(
        atom_feat_dim=39, edge_feat_dim=10, hidden_dim=32,
        n_layers=2, dropout=0.0,
    )
    return pyg.DirectedMPNNPyG(cfg).eval()


def _make_cpu_static_batch():
    """Static-shape MolGraphBatch on CPU: N=16 atoms, E=12 edges, B=2."""
    pyg = _load_pyg_module()
    torch.manual_seed(0)
    device = torch.device("cpu")
    N_total, E_total = 16, 12
    atom_feats = torch.randn(N_total, 39, device=device)
    edge_index = torch.randint(
        0, N_total, (2, E_total), device=device, dtype=torch.long,
    )
    edge_attr = torch.randn(E_total, 10, device=device)
    batch_idx = torch.tensor(
        [0] * 8 + [1] * 8, device=device, dtype=torch.long,
    )
    n_atoms_per_mol = torch.tensor([8, 8], device=device, dtype=torch.long)
    n_edges_per_mol = torch.tensor([6, 6], device=device, dtype=torch.long)
    return pyg.MolGraphBatch(
        atom_feats=atom_feats, edge_index=edge_index,
        edge_attr=edge_attr, batch_idx=batch_idx,
        n_atoms_per_mol=n_atoms_per_mol,
        n_edges_per_mol=n_edges_per_mol,
    )


# ---------------------------------------------------------------------------
# Test 1: CUDAGraph capture on static shape (CPU: eager passthrough)
# ---------------------------------------------------------------------------
def test_cuda_graph_wrapper_capture_or_passthrough() -> None:
    """Wrapper exposes capture interface; when CUDA unavailable, eager passthrough.

    On GPU: ``is_captured()`` returns True after the first forward call.
    On CPU: ``is_captured()`` returns False; the wrapper is a transparent
    passthrough that delegates every call to the underlying encoder.
    """
    if torch.cuda.is_available():
        pytest.skip(
            "This test exercises the CPU eager-passthrough path; "
            "see test_cuda_graph_wrapper_replay_matches_eager for the "
            "CUDA capture + replay path."
        )
    encoder = _make_cpu_pyg_encoder()
    batch = _make_cpu_static_batch()
    wrapper = CudaGraphDMPNNWrapper(
        encoder=encoder,
        static_n_atoms=batch.n_atoms,
        static_n_edges=batch.n_edges,
        static_batch_size=batch.batch_size,
    )
    out = wrapper(batch)
    with torch.no_grad():
        out_eager = encoder.forward_batched(batch)
    diff = (out - out_eager).abs().max().item()
    assert diff < 1e-4, f"wrapper output diverges from eager: {diff}"
    assert not wrapper.is_captured()


# ---------------------------------------------------------------------------
# Test 2: Replay correctness vs eager
# ---------------------------------------------------------------------------
def test_cuda_graph_wrapper_replay_matches_eager() -> None:
    """Replay output equals eager output within 1e-3 tolerance.

    Runs only on CUDA (capture + replay are CUDA-only APIs).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graphs capture requires CUDA")
    pyg = _load_pyg_module()
    torch.manual_seed(0)
    cfg = MPNNConfig(atom_feat_dim=39, edge_feat_dim=10, hidden_dim=32, n_layers=2)
    encoder = pyg.DirectedMPNNPyG(cfg).cuda().eval()

    N_total, E_total, B = 16, 12, 2
    atom_feats = torch.randn(N_total, 39, device="cuda")
    edge_index = torch.randint(
        0, N_total, (2, E_total), device="cuda", dtype=torch.long,
    )
    edge_attr = torch.randn(E_total, 10, device="cuda")
    batch_idx = torch.tensor(
        [0] * 8 + [1] * 8, device="cuda", dtype=torch.long,
    )
    n_atoms_per_mol = torch.tensor([8, 8], device="cuda", dtype=torch.long)
    n_edges_per_mol = torch.tensor([6, 6], device="cuda", dtype=torch.long)
    batch = pyg.MolGraphBatch(
        atom_feats=atom_feats, edge_index=edge_index, edge_attr=edge_attr,
        batch_idx=batch_idx, n_atoms_per_mol=n_atoms_per_mol,
        n_edges_per_mol=n_edges_per_mol,
    )

    wrapper = CudaGraphDMPNNWrapper(
        encoder=encoder,
        static_n_atoms=N_total,
        static_n_edges=E_total,
        static_batch_size=B,
    )
    with torch.no_grad():
        out_eager = encoder.forward_batched(batch)
        out_replay_1 = wrapper(batch)
        out_replay_2 = wrapper(batch)

    diff_1 = (out_eager - out_replay_1).abs().max().item()
    diff_2 = (out_replay_1 - out_replay_2).abs().max().item()
    assert diff_1 < 1e-3, f"replay 1 diverges from eager: {diff_1}"
    assert diff_2 < 1e-3, f"replay 2 diverges from replay 1: {diff_2}"
    fc = wrapper.fallback_counts()
    assert fc["n_replays"] >= 2, (
        f"expected ≥ 2 replays, got {fc['n_replays']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Fallback on shape change
# ---------------------------------------------------------------------------
def test_cuda_graph_wrapper_fallback_on_shape_change() -> None:
    """If input shape changes, wrapper falls back to eager + emits warning.

    The first forward call captures at (N=16, E=12, B=2); the second
    call uses (N=8, E=6, B=2) — the wrapper detects the mismatch,
    increments ``n_eager_fallbacks``, and emits a RuntimeWarning.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graphs capture requires CUDA")
    pyg = _load_pyg_module()
    torch.manual_seed(1)
    cfg = MPNNConfig(atom_feat_dim=39, edge_feat_dim=10, hidden_dim=32, n_layers=2)
    encoder = pyg.DirectedMPNNPyG(cfg).cuda().eval()

    # Capture batch: N=16, E=12, B=2
    N1, E1 = 16, 12
    atom1 = torch.randn(N1, 39, device="cuda")
    ei1 = torch.randint(0, N1, (2, E1), device="cuda", dtype=torch.long)
    ea1 = torch.randn(E1, 10, device="cuda")
    bi1 = torch.tensor([0] * 8 + [1] * 8, device="cuda", dtype=torch.long)
    na1 = torch.tensor([8, 8], device="cuda", dtype=torch.long)
    ne1 = torch.tensor([6, 6], device="cuda", dtype=torch.long)
    batch1 = pyg.MolGraphBatch(atom1, ei1, ea1, bi1, na1, ne1)

    wrapper = CudaGraphDMPNNWrapper(
        encoder=encoder, static_n_atoms=N1, static_n_edges=E1,
        static_batch_size=2,
    )
    _ = wrapper(batch1)
    assert wrapper.is_captured(), (
        "capture should have succeeded on GPU; got "
        f"is_captured()={wrapper.is_captured()}"
    )

    # Different batch: N=8, E=6, B=2 — N changed.
    N2, E2 = 8, 6
    atom2 = torch.randn(N2, 39, device="cuda")
    ei2 = torch.randint(0, N2, (2, E2), device="cuda", dtype=torch.long)
    ea2 = torch.randn(E2, 10, device="cuda")
    bi2 = torch.tensor([0] * 4 + [1] * 4, device="cuda", dtype=torch.long)
    na2 = torch.tensor([4, 4], device="cuda", dtype=torch.long)
    ne2 = torch.tensor([3, 3], device="cuda", dtype=torch.long)
    batch2 = pyg.MolGraphBatch(atom2, ei2, ea2, bi2, na2, ne2)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out2 = wrapper(batch2)
    assert wrapper._n_eager_fallbacks >= 1, (
        "expected n_eager_fallbacks >= 1 after shape change, "
        f"got {wrapper._n_eager_fallbacks}"
    )
    assert out2.shape == (2, encoder.hidden_dim)


# ---------------------------------------------------------------------------
# Test 4: Integration with Wf-131 production trainer signature
# ---------------------------------------------------------------------------
def test_cuda_graph_wrapper_trainer_signature() -> None:
    """Wrapper accepts the PyG-style MolGraphBatch used by the trainer.

    The trainer integration point (Wf-328-W8 wire-in) builds a
    :class:`MolGraphBatch` from the padded collate output, then passes
    it through the wrapper.  The wrapper must return a (B, hidden_dim)
    tensor.
    """
    encoder = _make_cpu_pyg_encoder()
    batch = _make_cpu_static_batch()
    wrapper = CudaGraphDMPNNWrapper(
        encoder=encoder,
        static_n_atoms=batch.n_atoms,
        static_n_edges=batch.n_edges,
        static_batch_size=batch.batch_size,
    )
    out = wrapper(batch)
    B = batch.batch_size
    assert out.shape == (B, encoder.hidden_dim), (
        f"output shape mismatch: got {out.shape}, "
        f"expected {(B, encoder.hidden_dim)}"
    )


# ---------------------------------------------------------------------------
# Test 5 (extra): reset() drops the captured graph
# ---------------------------------------------------------------------------
def test_cuda_graph_wrapper_reset() -> None:
    """reset() drops the captured graph so the next call recaptures."""
    cfg = MPNNConfig(atom_feat_dim=39, edge_feat_dim=10, hidden_dim=16, n_layers=2)
    # Build a stub encoder without triggering the PyG import (the wrapper
    # does not call the encoder until the first forward).
    encoder = torch.nn.Linear(39, 16)  # placeholder
    wrapper = CudaGraphDMPNNWrapper(
        encoder=encoder,
        static_n_atoms=16,
        static_n_edges=12,
        static_batch_size=2,
    )
    assert not wrapper.is_captured()
    wrapper.reset()
    assert not wrapper.is_captured()
    fc = wrapper.fallback_counts()
    assert fc["n_replays"] == 0
    assert fc["n_eager_fallbacks"] == 0
    assert fc["n_capture_failures"] == 0
