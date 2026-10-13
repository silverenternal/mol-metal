"""Regression: the D-MPNN pooled readout must mask zero-padding atoms.

``MetalCytotoxModel.forward`` (``molmetal/scripts/finetune_metalcytotox_from_tmqm.py``)
pools over ``atom_feats[b, :V]`` where ``V`` is the *padded* width.  When a batch
is padded to ``max_atoms``, every molecule carries its zero-filled padding rows
into the sum pool, and ``relu(atom_emb(0))`` is a non-zero bias vector — so the
padding is not neutral.  MEASURED on the Ru dedup cohort: median ligand 35 heavy
atoms padded to 120, i.e. ~68% of pooled slots are padding.

``MaskedMetalCytotoxModel`` (``molmetal/scripts/dmpnn_within_metal_auc.py``) uses
each molecule's true non-zero atom/edge count instead.  These tests pin the
property that matters: the masked model's output for a molecule must be
*independent of the batch's padding width*, which the unmasked model's is not.

CPU-only; no data files; tiny tensors.
"""

from __future__ import annotations

import torch

from molmetal.scripts.dmpnn_within_metal_auc import MaskedMetalCytotoxModel
from molmetal.scripts.finetune_metalcytotox_from_tmqm import MetalCytotoxModel
from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def _toy_graph(n_atoms: int, n_edges: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    atom = torch.rand(n_atoms, ATOM_FEATURE_DIM, generator=g)
    bond = torch.rand(n_edges, BOND_FEATURE_DIM, generator=g)
    src = torch.tensor([i % n_atoms for i in range(n_edges)], dtype=torch.long)
    dst = torch.tensor([(i + 1) % n_atoms for i in range(n_edges)], dtype=torch.long)
    return atom, bond, src, dst


def _pad_to(graph, max_atoms: int, max_edges: int):
    atom, bond, src, dst = graph
    V, E = atom.size(0), bond.size(0)
    af = torch.zeros(max_atoms, ATOM_FEATURE_DIM)
    bf = torch.zeros(max_edges, BOND_FEATURE_DIM)
    es = torch.zeros(max_edges, dtype=torch.long)
    ed = torch.zeros(max_edges, dtype=torch.long)
    af[:V] = atom
    bf[:E] = bond
    es[:E] = src
    ed[:E] = dst
    return af.unsqueeze(0), bf.unsqueeze(0), es.unsqueeze(0), ed.unsqueeze(0)


def test_masked_output_is_padding_width_invariant():
    """A molecule's masked logits do not change when the batch pad width grows."""
    torch.manual_seed(0)
    model = MaskedMetalCytotoxModel().eval()
    graph = _toy_graph(9, 8)

    out_small = model(*_pad_to(graph, 9, 8))
    out_wide = model(*_pad_to(graph, 40, 60))
    assert torch.allclose(out_small, out_wide, atol=1e-5), (
        "masked readout must ignore zero-padding rows; "
        f"got {out_small.tolist()} vs {out_wide.tolist()}"
    )


def test_unmasked_output_does_change_with_padding():
    """The property the fix addresses: the reused model is NOT pad-invariant.

    If this ever starts passing, the upstream ``MetalCytotoxModel`` gained a mask
    and ``MaskedMetalCytotoxModel`` is redundant — worth knowing.
    """
    torch.manual_seed(0)
    model = MetalCytotoxModel().eval()
    graph = _toy_graph(9, 8)

    out_small = model(*_pad_to(graph, 9, 8))
    out_wide = model(*_pad_to(graph, 40, 60))
    assert not torch.allclose(out_small, out_wide, atol=1e-5), (
        "expected the unmasked model's logits to shift with the pad width "
        "(relu(atom_emb(0)) bias enters the sum pool)"
    )


def test_masked_batch_matches_per_molecule_forward():
    """A padded batch equals the sum of the individual masked forwards."""
    torch.manual_seed(0)
    model = MaskedMetalCytotoxModel().eval()
    g1 = _toy_graph(7, 6, seed=1)
    g2 = _toy_graph(12, 11, seed=2)

    # batch of two, padded to the larger width
    a1, b1, s1, d1 = _pad_to(g1, 12, 11)
    a2, b2, s2, d2 = _pad_to(g2, 12, 11)
    af = torch.cat([a1, a2], dim=0)
    bf = torch.cat([b1, b2], dim=0)
    es = torch.cat([s1, s2], dim=0)
    ed = torch.cat([d1, d2], dim=0)
    batch_out = model(af, bf, es, ed)

    solo1 = model(*_pad_to(g1, 12, 11))
    solo2 = model(*_pad_to(g2, 12, 11))
    assert torch.allclose(batch_out, torch.cat([solo1, solo2], dim=0), atol=1e-5)
