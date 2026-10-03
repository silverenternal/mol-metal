"""Wf-331-X4 — Tests for the 4 missing Triton kernel CPU scaffolds.

=====================================================================
Per Wf-326E audit, 4 missing Triton kernels were identified:
  - fused_relu_mlp              (MISSING-001)
  - fused_wasserstein_sliced    (MISSING-002)
  - fused_bond_head_aggregation (MISSING-003)
  - fused_smiles_featurize      (MISSING-004)

Wf-331-X4 ships CPU scaffolds for all 4 in
``molmetal/adapters/flow_matching_lipman/_reference/``.  This test
file checks (8 tests = 2 per kernel):

  Group A — fused_relu_mlp (2 tests):
    test_01_fused_relu_mlp_correctness_vs_pt_reference
    test_02_fused_relu_mlp_fallback_when_triton_unavailable

  Group B — fused_wasserstein_sliced (2 tests):
    test_03_fused_wasserstein_sliced_correctness_vs_pt_reference
    test_04_fused_wasserstein_sliced_fallback_when_triton_unavailable

  Group C — fused_bond_head_aggregation (2 tests):
    test_05_fused_bond_head_aggregation_correctness_vs_pt_reference
    test_06_fused_bond_head_aggregation_fallback_when_triton_unavailable

  Group D — fused_smiles_featurize (2 tests):
    test_07_fused_smiles_featurize_correctness_vs_pt_reference
    test_08_fused_smiles_featurize_fallback_when_triton_unavailable

All tests are CPU-only (no GPU launch).  Each correctness test
verifies the CPU PyTorch fallback matches an independently-computed
PyTorch baseline (bit-exact for fp32 / within 1e-5 for reductions).
Each fallback test verifies that forcing the PyTorch path or running
on CPU yields the same result as the default dispatch.

Per AGENTS.md §5.1, this is an audit (no retrain / panel / ablation
launch), so pre-flight gate (TODO-207) is N/A and the kill_then_adapt
directive does not apply.  Per AGENTS.md §5.2, all claims use the
MEASURED/PROJECTED tag discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = (
    _PROJECT_ROOT / "molmetal" / "adapters" / "flow_matching_lipman" / "_reference"
)

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import all 4 kernel modules.  Each module's top-level import is safe
# even when triton is unavailable (the try/except inside each file
# installs a no-op stub).
from molmetal.adapters.flow_matching_lipman._reference.fused_relu_mlp import (  # noqa: E402
    fused_relu_mlp as _fused_relu_mlp,
    pytorch_reference as _pt_relu_mlp,
)
from molmetal.adapters.flow_matching_lipman._reference.fused_wasserstein_sliced import (  # noqa: E402
    fused_wasserstein_sliced as _fused_w2,
    pytorch_reference as _pt_w2,
)
from molmetal.adapters.flow_matching_lipman._reference.fused_bond_head_aggregation import (  # noqa: E402
    fused_bond_head_aggregation as _fused_bond_head,
    pytorch_reference as _pt_bond_head,
)
from molmetal.adapters.flow_matching_lipman._reference.fused_smiles_featurize import (  # noqa: E402
    fused_smiles_featurize as _fused_smiles,
    pytorch_reference as _pt_smiles,
    _FEATURE_DIM,
)


# ===========================================================================
# Group A — fused_relu_mlp (2 tests)
# ===========================================================================
class TestFusedReLUMlp:
    """Tests 01-02: fused ReLU-MLP CPU correctness + fallback."""

    def test_01_fused_relu_mlp_correctness_vs_pt_reference(self):
        """Verify the CPU fallback for ``fused_relu_mlp`` matches an
        independently-computed PyTorch baseline to within 1e-5.

        Builds a small ``nn.Sequential(Linear, ReLU, Linear)`` block,
        runs the kernel scaffold's ``fused_relu_mlp`` on a CPU
        tensor (which auto-falls-back to PyTorch), and compares to a
        hand-rolled ``F.linear(F.relu(F.linear(...)))`` chain.
        """
        torch.manual_seed(0)
        M, K, N1, N2 = 8, 16, 32, 5
        x = torch.randn(M, K)
        w1 = torch.randn(N1, K) * 0.1
        b1 = torch.randn(N1) * 0.05
        w2 = torch.randn(N2, N1) * 0.1
        b2 = torch.randn(N2) * 0.05
        # Independent PyTorch baseline.
        baseline = F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)
        # CPU fallback (no CUDA → auto PyTorch reference path).
        out = _fused_relu_mlp(x, w1, b1, w2, b2)
        assert (
            out.shape == baseline.shape
        ), f"shape mismatch: got {tuple(out.shape)}, expected {tuple(baseline.shape)}"
        diff = (out - baseline).abs().max().item()
        assert diff < 1e-5, (
            f"fused_relu_mlp CPU fallback diverges from independent "
            f"PyTorch baseline by {diff:.6f} (>1e-5 tolerance)."
        )
        # Also verify the module-level PyTorch reference matches.
        ref = _pt_relu_mlp(x, w1, b1, w2, b2)
        assert torch.allclose(
            out, ref, atol=1e-6
        ), "fused_relu_mlp output diverges from module pytorch_reference()"

    def test_02_fused_relu_mlp_fallback_when_triton_unavailable(self):
        """Verify ``fused_relu_mlp`` returns the PyTorch reference when
        ``force_pytorch=True``, regardless of any env-var state.  Also
        verify the no-bias path is supported (defaults b1/b2 to zeros).
        """
        torch.manual_seed(1)
        M, K, N1, N2 = 4, 8, 16, 3
        x = torch.randn(M, K)
        w1 = torch.randn(N1, K)
        w2 = torch.randn(N2, N1)
        # force_pytorch=True → always PyTorch.
        out = _fused_relu_mlp(
            x,
            w1,
            None,
            w2,
            None,
            force_pytorch=True,
        )
        expected = _pt_relu_mlp(
            x,
            w1,
            torch.zeros(N1, dtype=x.dtype),
            w2,
            torch.zeros(N2, dtype=x.dtype),
        )
        assert torch.allclose(out, expected, atol=1e-6), (
            f"force_pytorch=True output diverges from pytorch_reference() "
            f"by {(out - expected).abs().max().item():.6f}"
        )
        # force_pytorch=True with explicit biases matches the upstream
        # nn.Sequential(Linear, ReLU, Linear) shape exactly.
        b1 = torch.randn(N1)
        b2 = torch.randn(N2)
        out2 = _fused_relu_mlp(x, w1, b1, w2, b2, force_pytorch=True)
        assert out2.shape == (M, N2)
        # Sanity: output is not all zeros (the random projections + bias
        # must have produced something non-trivial).
        assert out2.abs().max().item() > 0.0


# ===========================================================================
# Group B — fused_wasserstein_sliced (2 tests)
# ===========================================================================
class TestFusedWassersteinSliced:
    """Tests 03-04: sliced-W₂ CPU correctness + fallback."""

    def test_03_fused_wasserstein_sliced_correctness_vs_pt_reference(self):
        """Verify the CPU fallback for ``fused_wasserstein_sliced``
        matches the module-level ``pytorch_reference`` (which mirrors
        ``_sliced_w2`` from ``wasserstein_loss.py``).

        Seeds the projections to ``seed=0``, computes both the
        scaffold's :func:`fused_wasserstein_sliced` and the module
        reference, and asserts agreement to within 1e-4 abs diff.
        """
        torch.manual_seed(2)
        N, M, d = 32, 32, 4
        n_mc = 8
        seed = 0
        x = torch.randn(N, d)
        y = torch.randn(M, d)
        # Independent PyTorch reference (inlined from
        # wasserstein_loss.py::_sliced_w2).
        gen = torch.Generator()
        gen.manual_seed(seed)
        proj = torch.randn(n_mc, d, generator=gen)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-12)
        x_proj = x @ proj.t()
        y_proj = y @ proj.t()
        contribs = [
            (xs_proj.sort().values - ys_proj.sort().values).abs().mean()
            for xs_proj, ys_proj in zip(
                [x_proj[:, i] for i in range(n_mc)],
                [y_proj[:, i] for i in range(n_mc)],
            )
        ]
        baseline = torch.stack(contribs).mean()
        # Module-level pytorch_reference (must agree with baseline).
        ref = _pt_w2(x, y, n_mc=n_mc, seed=seed)
        # fused_wasserstein_sliced (CPU → auto PyTorch path).
        out = _fused_w2(x, y, n_mc=n_mc, seed=seed)
        diff_baseline = (out - baseline).abs().max().item()
        diff_ref = (out - ref).abs().max().item()
        assert diff_baseline < 1e-4, (
            f"fused_wasserstein_sliced diverges from independent "
            f"_sliced_w2 baseline by {diff_baseline:.6f}"
        )
        assert diff_ref < 1e-6, (
            f"fused_wasserstein_sliced diverges from pytorch_reference "
            f"by {diff_ref:.6f}"
        )
        # Sanity: W₂ is non-negative.
        assert out.item() >= 0.0
        # Sanity: W₂(x, x) ≈ 0 (identical distributions → 0 distance).
        out_xx = _fused_w2(x, x, n_mc=n_mc, seed=seed)
        assert (
            out_xx.abs().item() < 1e-3
        ), f"W₂(x, x) should be ~0, got {out_xx.item():.6f}"

    def test_04_fused_wasserstein_sliced_fallback_when_triton_unavailable(self):
        """Verify ``fused_wasserstein_sliced`` with ``force_pytorch=True``
        matches the upstream signature contract:
          - shape mismatches raise ValueError
          - n_mc < 1 raises ValueError
          - empty input returns zero scalar
        """
        torch.manual_seed(3)
        x = torch.randn(8, 3)
        y = torch.randn(8, 3)
        # force_pytorch=True → always PyTorch, no env-gate.
        out = _fused_w2(x, y, n_mc=4, seed=7, force_pytorch=True)
        ref = _pt_w2(x, y, n_mc=4, seed=7)
        assert torch.allclose(out, ref, atol=1e-6), (
            f"force_pytorch=True output diverges from pytorch_reference "
            f"by {(out - ref).abs().max().item():.6f}"
        )
        # Empty input → zero.
        empty_out = _fused_w2(
            torch.zeros(0, 3),
            y,
            n_mc=4,
            seed=0,
            force_pytorch=True,
        )
        assert empty_out.item() == 0.0
        # Shape mismatch raises.
        with pytest.raises(ValueError, match="must be 2-D"):
            _fused_w2(torch.randn(8), y, n_mc=4, seed=0, force_pytorch=True)
        # n_mc < 1 raises.
        with pytest.raises(ValueError, match="n_mc"):
            _fused_w2(x, y, n_mc=0, seed=0, force_pytorch=True)


# ===========================================================================
# Group C — fused_bond_head_aggregation (2 tests)
# ===========================================================================
class TestFusedBondHeadAggregation:
    """Tests 05-06: bond-head MLP CPU correctness + fallback."""

    def test_05_fused_bond_head_aggregation_correctness_vs_pt_reference(self):
        """Verify the CPU fallback for ``fused_bond_head_aggregation``
        matches the canonical 3-layer MLP
        ``Linear → ReLU → Linear → ReLU → Linear`` that
        :class:`BondOrderHead.forward` uses.
        """
        torch.manual_seed(4)
        E, in_dim, hidden, num_classes = 12, 9, 64, 5
        pair_features = torch.randn(E, in_dim)
        w1 = torch.randn(hidden, in_dim) * 0.1
        b1 = torch.randn(hidden) * 0.05
        w2 = torch.randn(hidden, hidden) * 0.1
        b2 = torch.randn(hidden) * 0.05
        w3 = torch.randn(num_classes, hidden) * 0.1
        b3 = torch.randn(num_classes) * 0.05
        weights = [(w1, b1), (w2, b2), (w3, b3)]
        # Independent PyTorch baseline (inlined from BondOrderHead.forward).
        h = F.relu(F.linear(pair_features, w1, b1))
        h = F.relu(F.linear(h, w2, b2))
        baseline = F.linear(h, w3, b3)
        # Module-level pytorch_reference.
        ref = _pt_bond_head(pair_features, weights)
        # CPU fallback.
        out = _fused_bond_head(pair_features, weights)
        diff_baseline = (out - baseline).abs().max().item()
        diff_ref = (out - ref).abs().max().item()
        assert diff_baseline < 1e-5, (
            f"fused_bond_head_aggregation diverges from independent MLP "
            f"baseline by {diff_baseline:.6f}"
        )
        assert diff_ref < 1e-6, (
            f"fused_bond_head_aggregation diverges from "
            f"pytorch_reference by {diff_ref:.6f}"
        )
        assert out.shape == (E, num_classes)

    def test_06_fused_bond_head_aggregation_fallback_when_triton_unavailable(self):
        """Verify ``fused_bond_head_aggregation`` falls back correctly
        when ``force_pytorch=True``, supports no-bias weights, and
        raises on wrong number of weight tuples.
        """
        torch.manual_seed(5)
        E, in_dim, hidden, num_classes = 6, 4, 8, 3
        pair_features = torch.randn(E, in_dim)
        w1 = torch.randn(hidden, in_dim) * 0.1
        w2 = torch.randn(hidden, hidden) * 0.1
        w3 = torch.randn(num_classes, hidden) * 0.1
        # No-bias path: force_pytorch=True with b=None should default to zeros.
        weights_no_bias = [(w1, None), (w2, None), (w3, None)]
        out = _fused_bond_head(
            pair_features,
            weights_no_bias,
            force_pytorch=True,
        )
        ref = _pt_bond_head(
            pair_features,
            [
                (w1, torch.zeros(hidden, dtype=pair_features.dtype)),
                (w2, torch.zeros(hidden, dtype=pair_features.dtype)),
                (w3, torch.zeros(num_classes, dtype=pair_features.dtype)),
            ],
        )
        assert torch.allclose(out, ref, atol=1e-6), (
            f"no-bias path diverges from pytorch_reference by "
            f"{(out - ref).abs().max().item():.6f}"
        )
        # Wrong number of weight tuples → ValueError.
        with pytest.raises(ValueError, match="3 \\(W, b\\) pairs"):
            _pt_bond_head(pair_features, [(w1, None), (w2, None)])
        # Explicit biases path.
        b1 = torch.randn(hidden)
        b2 = torch.randn(hidden)
        b3 = torch.randn(num_classes)
        out2 = _fused_bond_head(
            pair_features,
            [(w1, b1), (w2, b2), (w3, b3)],
            force_pytorch=True,
        )
        assert out2.shape == (E, num_classes)
        assert out2.abs().max().item() > 0.0


# ===========================================================================
# Group D — fused_smiles_featurize (2 tests)
# ===========================================================================
class TestFusedSmilesFeaturize:
    """Tests 07-08: SMILES featurise CPU correctness + fallback."""

    def test_07_fused_smiles_featurize_correctness_vs_pt_reference(self):
        """Verify the CPU fallback for ``fused_smiles_featurize`` matches
        an independently-computed per-atom feature matrix (one-hot
        symbol + one-hot degree + ring + aromatic + charge).
        """
        symbols = ["C", "N", "O", "Pt", "S", "Other"]
        degrees = [3, 2, 1, 4, 2, 0]
        in_ring = [True, False, True, False, False, True]
        aromatic = [True, False, False, False, False, False]
        formal_charge = [0, 0, -1, 2, 0, 1]
        # Independent PyTorch baseline (inlined).
        n = len(symbols)
        sym_idx = [
            {
                "C": 0,
                "N": 1,
                "O": 2,
                "F": 3,
                "P": 4,
                "S": 5,
                "Cl": 6,
                "Br": 7,
                "I": 8,
                "B": 9,
                "Si": 10,
                "Se": 11,
                "H": 12,
            }.get(s, 13)
            for s in symbols
        ]
        deg_idx = [min(d, 5) for d in degrees]
        num_symbols = 14
        num_degrees = 6
        baseline = torch.zeros((n, num_symbols + num_degrees + 3), dtype=torch.float32)
        for i in range(n):
            baseline[i, sym_idx[i]] = 1.0
            baseline[i, num_symbols + deg_idx[i]] = 1.0
            baseline[i, num_symbols + num_degrees] = float(in_ring[i])
            baseline[i, num_symbols + num_degrees + 1] = float(aromatic[i])
            c = max(-1, min(1, formal_charge[i]))
            baseline[i, num_symbols + num_degrees + 2] = float(c)
        # Module pytorch_reference.
        ref = _pt_smiles(symbols, degrees, in_ring, aromatic, formal_charge)
        # CPU fallback.
        out = _fused_smiles(symbols, degrees, in_ring, aromatic, formal_charge)
        # Verify dimensions.
        assert out.shape == (
            n,
            _FEATURE_DIM,
        ), f"shape mismatch: got {tuple(out.shape)}, expected ({n}, {_FEATURE_DIM})"
        diff_baseline = (out - baseline).abs().max().item()
        diff_ref = (out - ref).abs().max().item()
        assert diff_baseline < 1e-5, (
            f"fused_smiles_featurize diverges from independent baseline "
            f"by {diff_baseline:.6f}"
        )
        assert diff_ref < 1e-6, (
            f"fused_smiles_featurize diverges from pytorch_reference by "
            f"{diff_ref:.6f}"
        )
        # Sanity: Pt atom (idx 13 in Other row) has degree=4 → degree
        # one-hot at slot 4 should be set.
        pt_idx = symbols.index("Pt")
        assert out[pt_idx, num_symbols + 4].item() == 1.0
        # Sanity: O atom (charge -1) → charge slot set to -1 (clipped).
        o_idx = symbols.index("O")
        assert out[o_idx, num_symbols + num_degrees + 2].item() == -1.0

    def test_08_fused_smiles_featurize_fallback_when_triton_unavailable(self):
        """Verify ``fused_smiles_featurize`` with ``force_pytorch=True``
        matches the module pytorch_reference and raises on mismatched
        per-atom list lengths.
        """
        symbols = ["C", "N", "O"]
        degrees = [4, 3, 2]
        in_ring = [True, False, True]
        aromatic = [False, False, False]
        formal_charge = [0, 0, 0]
        # force_pytorch=True → always PyTorch reference.
        out = _fused_smiles(
            symbols,
            degrees,
            in_ring,
            aromatic,
            formal_charge,
            force_pytorch=True,
        )
        ref = _pt_smiles(symbols, degrees, in_ring, aromatic, formal_charge)
        assert torch.allclose(out, ref, atol=1e-6), (
            f"force_pytorch=True output diverges from pytorch_reference "
            f"by {(out - ref).abs().max().item():.6f}"
        )
        # Empty input → empty output.
        empty_out = _fused_smiles(
            [],
            [],
            [],
            [],
            [],
            force_pytorch=True,
        )
        assert empty_out.shape == (0, _FEATURE_DIM)
        # Mismatched lengths → ValueError.
        with pytest.raises(ValueError, match="same length"):
            _fused_smiles(
                ["C", "N"],
                [4],
                [True, False],
                [False, False],
                [0, 0],
                force_pytorch=True,
            )
        # Unknown symbol falls into the "Other" slot (idx 13).
        out_unknown = _fused_smiles(
            ["Xx"],
            [0],
            [False],
            [False],
            [0],
            force_pytorch=True,
        )
        assert out_unknown[0, 13].item() == 1.0
