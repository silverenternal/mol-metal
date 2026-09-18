"""Canonical graph comparison and bounded-memory GPU nearest Morgan neighbors."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonical_graph(mol):
    """Preserve stereo/isotopes/charge/fragments; remove maps and ordinary H."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('empty or invalid molecule')
    mol = Chem.Mol(mol)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('wildcard atom')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_smiles(smiles):
    # Generic scaffold novelty ignores substituents and stereochemistry but
    # retains atom/bond identity. Empty (acyclic) scaffolds are not novel rings.
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)


def morgan_fingerprints(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False)
    return np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
                       for s in smiles], dtype=np.uint8).reshape(-1, 2048)


def nearest_tanimoto(query, reference, *, device='cuda:0', chunk_size=4096,
                     query_chunk_size=128):
    """Return first-index deterministic ties; stream reference chunks to device.

    Input fingerprints must be binary. Empty references have no neighbors;
    empty-zero bit-vector pairs have Tanimoto 1 by explicit convention.
    """
    import torch
    if chunk_size < 1 or query_chunk_size < 1:
        raise ValueError('chunk sizes must be positive')
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError('fingerprints must be 2D with matching dimensions')
    if not np.isin(query, [0, 1]).all() or not np.isin(reference, [0, 1]).all():
        raise ValueError('fingerprints must be binary')
    target = torch.device(device)
    # Validate explicit GPU requests even for empty inputs; no CPU fallback.
    torch.empty(0, device=target)
    scores = np.full(len(query), np.nan, dtype=np.float64)
    indices = np.full(len(query), -1, dtype=np.int64)
    if not len(reference):
        return scores, indices
    with torch.inference_mode():
        for qstart in range(0, len(query), query_chunk_size):
            qstop = min(len(query), qstart + query_chunk_size)
            q = torch.as_tensor(query[qstart:qstop], dtype=torch.float32, device=target)
            qsize = q.sum(1)
            best = torch.full((len(q),), -1., device=target)
            best_idx = torch.full((len(q),), -1, dtype=torch.int64, device=target)
            for start in range(0, len(reference), chunk_size):
                r = torch.tensor(reference[start:start+chunk_size], dtype=torch.float32, device=target)
                inter = q @ r.T
                union = qsize[:, None] + r.sum(1)[None, :] - inter
                sim = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
                values, local = sim.max(1)
                improved = values > best
                best_idx = torch.where(improved, local + start, best_idx)
                best = torch.maximum(best, values)
            scores[qstart:qstop] = best.cpu().numpy()
            indices[qstart:qstop] = best_idx.cpu().numpy()
    return scores, indices
