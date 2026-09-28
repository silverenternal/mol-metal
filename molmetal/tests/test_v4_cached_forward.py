"""Tests for ``V4MultiTaskDMPNN.forward_cached`` (R20 P3 §2 pre-featurized path).

Goal:
    ``V4MultiTaskDMPNN.forward`` originally takes a SMILES list and calls
    ``_featurize_batch`` (RDKit on CPU) per step. This makes the
    pre_featurized tensors in ``molmetal/data/pre_featurized/`` useless.
    The new ``forward_cached`` method accepts pre-featurized tensors
    directly and skips RDKit entirely.

Three properties are verified here:

  (i)   ``forward_cached`` does NOT call ``rdkit.Chem.MolFromSmiles``.
  (ii)  ``forward_cached`` produces numerically similar pooled embeddings
        to ``forward`` for the same SMILES (after pre-featurising them
        with the underlying ``GraphFeaturizer`` so both paths share the
        upstream featurisation result).
  (iii) ``use_cached_encoding`` toggles the cached-vs-SMILES branch in
        ``forward()`` — default ``False`` keeps the SMILES path, opt-in
        ``True`` enables the cached dispatch.

All tests are CPU-only — no GPU required.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from molmetal.data.featurize import GraphFeaturizer
from molmetal.models.v4_multitask_dmpnn import (
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


SMILES_3 = ["CCO", "c1ccccc1", "CC(=O)O"]


def _make_model(
    hidden_dim: int = 64,
    n_layers: int = 2,
    n_metals: int = 5,
    n_coord_classes: int = 5,
    use_cached_encoding: bool = False,
) -> V4MultiTaskDMPNN:
    """Small model for unit tests."""
    _set_seed(0)
    return V4MultiTaskDMPNN(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
        use_cached_encoding=use_cached_encoding,
    )


def _featurize_smiles_to_tensors(smiles_list):
    """Run GraphFeaturizer on SMILES → padded (atom_features, edge_index,
    edge_attr, atom_mask, batch_idx) tensors.

    Mirrors what ``DMPNNEncoder._featurize_batch`` does internally so we
    can call the same downstream code path that ``forward_cached``
    exercises — but without going through RDKit in the test path (we
    only need *one* featurisation to build the cached-input tensors).
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    featurizer = GraphFeaturizer()
    B = len(smiles_list)
    atom_feats, edge_indices, edge_attrs, n_atoms_list = [], [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit parse failed for {smi!r}")
        feat = featurizer(mol)
        atom_feats.append(feat["x"])
        edge_indices.append(feat["edge_index"])
        edge_attrs.append(feat["edge_attr"])
        n_atoms_list.append(int(feat["n_atoms"]))
    N_max = max(max(n_atoms_list), 1)
    E_max = max(max(ei.shape[1] for ei in edge_indices), 1)

    h_atom = torch.zeros(B, N_max, atom_feats[0].shape[1], dtype=torch.float32)
    edge_index = torch.zeros(B, 2, E_max, dtype=torch.long)
    edge_attr = torch.zeros(B, E_max, edge_attrs[0].shape[1], dtype=torch.float32)
    atom_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for b in range(B):
        af = torch.from_numpy(atom_feats[b]).float()
        n_a = af.size(0)
        h_atom[b, :n_a] = af
        atom_mask[b, :n_a] = True
        ei = torch.from_numpy(edge_indices[b]).long()
        n_e = ei.size(1)
        edge_index[b, :, :n_e] = ei
        ea = torch.from_numpy(edge_attrs[b]).float()
        n_ea = ea.size(0)
        edge_attr[b, :n_ea] = ea
    return h_atom, edge_index, edge_attr, atom_mask


# ---------------------------------------------------------------------------
# (i) forward_cached skips RDKit
# ---------------------------------------------------------------------------
def test_forward_cached_skips_rdkit():
    """``forward_cached`` must not call ``rdkit.Chem.MolFromSmiles``.

    Even though we still need *some* featurisation to produce the
    tensors that go into ``forward_cached``, that featurisation must
    happen *before* ``forward_cached`` is called. The method itself
    must not invoke RDKit.
    """
    m = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    h_atom, edge_index, edge_attr, atom_mask = _featurize_smiles_to_tensors(
        SMILES_3
    )

    with patch("rdkit.Chem.MolFromSmiles") as mock_mol:
        out = m.forward_cached(
            h_atom, edge_index, metal_ids,
            atom_mask=atom_mask, edge_attr=edge_attr,
        )
        # RDKit was NOT called by forward_cached
        assert mock_mol.call_count == 0, (
            f"forward_cached unexpectedly called rdkit.Chem.MolFromSmiles "
            f"{mock_mol.call_count} times"
        )

    # Output shape sanity
    assert set(out.keys()) == set(SUPPORTED_METALS) | {"coord"}
    for met in SUPPORTED_METALS:
        assert out[met].shape == (3, 2)
    assert out["coord"].shape == (3, 5)


# ---------------------------------------------------------------------------
# (ii) forward_cached matches forward (numerical equivalence)
# ---------------------------------------------------------------------------
def test_forward_cached_matches_forward():
    """Cached and SMILES paths produce numerically similar pooled embeddings
    for the same molecules.

    We featurise the SMILES once via the standard featurizer to build
    the cached inputs, then compare the cached ``h = encoder(...)``
    output to the SMILES ``h = encoder(...)`` output.  Both should be
    equal because the upstream featurisation is shared.
    """
    _set_seed(0)
    m = _make_model()
    m.eval()

    with torch.no_grad():
        # SMILES path — invokes _featurize_batch (RDKit) internally
        h_smiles = m.encode(SMILES_3)              # (3, hidden_dim)

        # Cached path — feed pre-featurized tensors
        h_atom, edge_index, edge_attr, atom_mask = _featurize_smiles_to_tensors(
            SMILES_3
        )
        h_cached = m.encode_cached(
            h_atom, edge_index, atom_mask=atom_mask, edge_attr=edge_attr
        )

    assert h_smiles.shape == h_cached.shape == (3, m.hidden_dim)
    # Equal — same featuriser, same weights, deterministic ops.
    assert torch.allclose(h_smiles, h_cached, atol=1e-5), (
        f"Cached and SMILES encoder outputs differ: max diff = "
        f"{(h_smiles - h_cached).abs().max().item():.3e}"
    )

    # Full forward path — logits should also match
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        out_smiles = m(SMILES_3, metal_ids)
        out_cached = m.forward_cached(
            h_atom, edge_index, metal_ids,
            atom_mask=atom_mask, edge_attr=edge_attr,
        )
    for met in SUPPORTED_METALS:
        assert torch.allclose(out_smiles[met], out_cached[met], atol=1e-5)
    assert torch.allclose(out_smiles["coord"], out_cached["coord"], atol=1e-5)


# ---------------------------------------------------------------------------
# (iii) use_cached_encoding toggle
# ---------------------------------------------------------------------------
def test_use_cached_toggle():
    """``use_cached_encoding`` flag wires into the constructor and changes
    the cached-vs-SMILES routing path in ``forward()``.

    Default ``False``: ``forward(smiles_list, ...)`` runs the SMILES
    path; ``forward(tensor, ...)`` raises (must call ``forward_cached``
    explicitly or flip the flag).

    ``True``: ``forward(tensor, ..., edge_index=...)`` dispatches to the
    cached path (no RDKit); ``forward(smiles_list, ...)`` still works
    (backward compat).
    """
    _set_seed(0)

    # ----- Default: use_cached_encoding=False -----
    m_default = _make_model()
    assert m_default.use_cached_encoding is False, (
        "Default use_cached_encoding should be False (backward compat)"
    )
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    # SMILES path works
    out_smiles = m_default(SMILES_3, metal_ids)
    assert out_smiles["coord"].shape == (3, 5)
    # Tensor path is rejected — must use forward_cached explicitly
    h_atom, edge_index, edge_attr, atom_mask = _featurize_smiles_to_tensors(
        SMILES_3
    )
    with pytest.raises(RuntimeError, match="use_cached_encoding"):
        m_default(h_atom, metal_ids, edge_index=edge_index)

    # ----- Opt-in: use_cached_encoding=True -----
    m_cached = _make_model(use_cached_encoding=True)
    assert m_cached.use_cached_encoding is True, (
        "Opt-in use_cached_encoding should be True"
    )
    # Cached path is now reachable through forward()
    out_cached = m_cached(h_atom, metal_ids, edge_index=edge_index,
                          atom_mask=atom_mask, edge_attr=edge_attr)
    assert set(out_cached.keys()) == set(SUPPORTED_METALS) | {"coord"}
    assert out_cached["coord"].shape == (3, 5)
    # SMILES path still works (backward compat)
    out_smiles_cached = m_cached(SMILES_3, metal_ids)
    assert out_smiles_cached["coord"].shape == (3, 5)


# ---------------------------------------------------------------------------
# (bonus) encoder.forward_cached shape + missing edge_attr graceful
# ---------------------------------------------------------------------------
def test_forward_cached_handles_missing_edge_attr():
    """When ``edge_attr`` is None, ``forward_cached`` substitutes zeros —
    no crash, valid output shape, finite values."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    h_atom, edge_index, _, atom_mask = _featurize_smiles_to_tensors(SMILES_3)

    with patch("rdkit.Chem.MolFromSmiles") as mock_mol:
        out = m.forward_cached(
            h_atom, edge_index, metal_ids,
            atom_mask=atom_mask, edge_attr=None,
        )
        assert mock_mol.call_count == 0

    for met in SUPPORTED_METALS:
        assert out[met].shape == (3, 2)
        assert torch.isfinite(out[met]).all()
    assert out["coord"].shape == (3, 5)
    assert torch.isfinite(out["coord"]).all()


# ---------------------------------------------------------------------------
# (bonus) backward still flows through the cached path
# ---------------------------------------------------------------------------
def test_forward_cached_backward_flows():
    """``loss.backward()`` succeeds on the cached path — gradient reaches
    the shared encoder AND every metal head."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    h_atom, edge_index, edge_attr, atom_mask = _featurize_smiles_to_tensors(
        SMILES_3
    )

    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskLoss

    out = m.forward_cached(
        h_atom, edge_index, metal_ids,
        atom_mask=atom_mask, edge_attr=edge_attr,
    )
    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))
    loss_fn = V4MultiTaskLoss()
    loss, _ = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss)
    loss.backward()

    # Encoder received gradient
    enc_grad = m.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0.0
    # Every metal head received gradient
    for met in SUPPORTED_METALS:
        g = m.metal_heads[met][0].weight.grad
        assert g is not None and g.abs().sum().item() > 0.0