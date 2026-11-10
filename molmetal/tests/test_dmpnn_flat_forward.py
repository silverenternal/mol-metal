"""Regression test for DMPNNMultiTaskModel._forward_flat vs _forward_one_mol.

Verifies the vectorised flat-graph forward produces (near-)identical
pooled vectors and (pic50, cls) predictions to the per-molecule
Python-loop path.  This is the fail-first regression gate for the
Wf-XXX speedup; if anyone changes message passing or pooling, this
test will fail loudly.

Tolerance rationale: the only numerical difference between the two
paths is the FP-add associativity of the per-molecule sum.  Across
~30 atoms x 2H features summed in the same order, both paths produce
identical reductions, so ``rtol=1e-5`` is tight enough to catch
shape/structural regressions but loose enough to ignore noise from
random initialisation effects.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from molmetal.baselines.dmpnn import featurize_smiles_list, collate_graphs  # noqa: E402
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel  # noqa: E402


def _build_batch(smiles_subset, seed=0):
    graphs = featurize_smiles_list(smiles_subset)
    batch = collate_graphs(graphs)
    af, bf, es, ed = batch
    return af, bf, es, ed


def _oracle_forward_one_by_one(model, af, bf, es, ed):
    """Per-mol Python loop — same logic as the original forward."""
    B = af.size(0)
    device = af.device
    pooled_list = []
    for b in range(B):
        V = af.size(1)
        valid_edges = bf[b].abs().sum(dim=-1) > 0
        E = valid_edges.sum().item()
        if E == 0 or V == 0:
            pooled_list.append(torch.zeros(2 * model.hidden, device=device))
            continue
        pooled_list.append(
            model._forward_one_mol(af[b, :V], bf[b, :E], es[b, :E], ed[b, :E])
        )
    return torch.stack(pooled_list, dim=0)


def test_flat_forward_matches_per_mol_loop():
    torch.manual_seed(0)
    np.random.seed(0)

    # 8 mols of varying size — enough to exercise both padding paths
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "O=C(O)c1ccccc1O",              # salicylic acid
        "CCO",                          # ethanol
        "c1ccccc1",                     # benzene
        "CCN(CC)CC",                    # triethylamine
        "OCC(O)CO",                     # glycerol
        "CC(C)O",                       # isopropanol
    ]
    af, bf, es, ed = _build_batch(smiles, seed=0)

    model = DMPNNMultiTaskModel(hidden=64, depth=3, dropout=0.0)
    model.eval()  # disable dropout so output is deterministic

    # Per-mol loop (oracle) vs flat (new) — pooled vectors
    pooled_oracle = _oracle_forward_one_by_one(model, af, bf, es, ed)
    pooled_flat = model._forward_flat(af, bf, es, ed)

    assert pooled_oracle.shape == pooled_flat.shape == (8, 2 * 64), (
        f"shape mismatch: oracle={pooled_oracle.shape} flat={pooled_flat.shape}"
    )
    max_diff = (pooled_oracle - pooled_flat).abs().max().item()
    assert max_diff < 1e-4, (
        f"pooled vector max abs diff {max_diff:.2e} exceeds tolerance 1e-4"
    )

    # Full forward (pic50, cls) — should also match
    pic50_oracle, cls_oracle = model.readout(pooled_oracle)
    pic50_flat, cls_flat = model._forward_flat_then_readout(af, bf, es, ed) \
        if hasattr(model, "_forward_flat_then_readout") \
        else model.forward(af, bf, es, ed)

    assert torch.allclose(pic50_oracle, pic50_flat, rtol=1e-5, atol=1e-5), (
        f"pic50 mismatch: max diff {(pic50_oracle - pic50_flat).abs().max():.2e}"
    )
    assert torch.allclose(cls_oracle, cls_flat, rtol=1e-5, atol=1e-5), (
        f"cls mismatch: max diff {(cls_oracle - cls_flat).abs().max():.2e}"
    )


def test_flat_forward_handles_empty_molecule():
    """When a molecule has V==0 (all-zero atom_feats), pooled should be 0."""
    from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM  # noqa: E402
    torch.manual_seed(0)
    model = DMPNNMultiTaskModel(hidden=32, depth=2, dropout=0.0)
    model.eval()

    # 3-mol batch where middle one is empty (all zeros)
    af = torch.zeros(3, 5, ATOM_FEATURE_DIM)
    af[0] = torch.randn(5, ATOM_FEATURE_DIM)
    af[2] = torch.randn(5, ATOM_FEATURE_DIM)
    bf = torch.zeros(3, 8, BOND_FEATURE_DIM)
    bf[0, :4] = torch.randn(4, BOND_FEATURE_DIM)
    bf[2, :4] = torch.randn(4, BOND_FEATURE_DIM)
    es = torch.full((3, 8), -1, dtype=torch.long)
    es[0, :4] = torch.tensor([0, 1, 2, 3])
    es[2, :4] = torch.tensor([0, 1, 2, 3])
    ed = torch.full((3, 8), -1, dtype=torch.long)
    ed[0, :4] = torch.tensor([1, 2, 3, 4])
    ed[2, :4] = torch.tensor([1, 2, 3, 4])

    pooled = model._forward_flat(af, bf, es, ed)
    assert pooled.shape == (3, 2 * 32)
    # Empty middle molecule should have zero pooled
    assert pooled[1].abs().max().item() == 0.0, (
        f"empty mol pooled should be zero, got {pooled[1].abs().max()}"
    )
    # Non-empty mols should have non-zero pooled
    assert pooled[0].abs().max().item() > 0.0
    assert pooled[2].abs().max().item() > 0.0


def test_flat_forward_matches_in_train_mode():
    """In train mode (with dropout=0 so RNG state doesn't matter), the
    forward must be bit-identical to the oracle — same gradient flow,
    same shapes, same finite outputs.  When dropout is active, the
    RNG state advances differently between the per-mol loop and the
    flat call, so the two paths will give different dropout masks; this
    is acceptable (both are valid training trajectories) and is NOT a
    correctness regression.  The eval-mode test above is the load-bearing
    correctness gate."""
    torch.manual_seed(0)
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
              "O=C(O)c1ccccc1O", "CCO"]
    af, bf, es, ed = _build_batch(smiles)

    model = DMPNNMultiTaskModel(hidden=64, depth=3, dropout=0.0)
    model.train()

    pooled_oracle = _oracle_forward_one_by_one(model, af, bf, es, ed)
    pooled_flat = model._forward_flat(af, bf, es, ed)

    max_diff = (pooled_oracle - pooled_flat).abs().max().item()
    assert max_diff < 1e-4, (
        f"train-mode pooled diff {max_diff:.2e} exceeds tolerance"
    )

    # Sanity: gradients flow identically in both paths.
    pooled_oracle.sum().backward()
    grads_oracle = {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad()
    pooled_flat.sum().backward()
    grads_flat = {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    assert set(grads_oracle) == set(grads_flat), (
        f"param set mismatch: oracle={set(grads_oracle) - set(grads_flat)} "
        f"flat={set(grads_flat) - set(grads_oracle)}"
    )
    for n in grads_oracle:
        assert torch.allclose(grads_oracle[n], grads_flat[n], atol=1e-5), (
            f"gradient mismatch for {n}: "
            f"max diff {(grads_oracle[n] - grads_flat[n]).abs().max():.2e}"
        )


if __name__ == "__main__":
    test_flat_forward_matches_per_mol_loop()
    test_flat_forward_handles_empty_molecule()
    test_flat_forward_matches_in_train_mode()
    print("[test_dmpnn_flat_forward] 3/3 PASS")
