"""RDKit fingerprints on CPU, exact binary Tanimoto reductions on torch GPU.

Explicit GPU requests never silently fall back. Chunking bounds pairwise GPU
memory, and invalid inputs are reported rather than presented as diversity.
"""
from __future__ import annotations


def molecular_diversity(smiles, *, device='auto', chunk_size=512):
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    resolved = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    target = torch.device(resolved)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints, valid_indices, canonical = [], [], []
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text.strip() else None
        if mol is None or mol.GetNumAtoms() == 0 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            continue
        fingerprints.append(generator.GetFingerprintAsNumPy(mol))
        valid_indices.append(i)
        canonical.append(Chem.MolToSmiles(mol))
    report = dict(n_input=len(smiles), n_valid=len(fingerprints), n_invalid=len(smiles)-len(fingerprints),
                  n_unique=len(set(canonical)), valid_indices=valid_indices,
                  device=str(target), fingerprint_backend='RDKit Morgan radius=2 bits=2048 CPU',
                  similarity_backend='torch binary Tanimoto', mean_pairwise_tanimoto=None, diversity=None)
    # Allocate on requested device even with no pairs, to expose unavailable GPU.
    total, pairs = torch.zeros((), device=target, dtype=torch.float64), 0
    if len(fingerprints) < 2:
        return report
    fp = torch.as_tensor(np.stack(fingerprints), device=target, dtype=torch.float32)
    sizes = fp.sum(dim=1)
    for start in range(0, len(fp), chunk_size):
        stop = min(start + chunk_size, len(fp))
        for other in range(start, len(fp), chunk_size):
            end = min(other + chunk_size, len(fp))
            intersection = fp[start:stop] @ fp[other:end].T
            union = sizes[start:stop, None] + sizes[None, other:end] - intersection
            similarity = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
            if other == start:
                indices = torch.triu_indices(stop-start, end-other, offset=1, device=target)
                values = similarity[indices[0], indices[1]]
            else:
                values = similarity.flatten()
            total += values.sum(dtype=torch.float64)
            pairs += values.numel()
    mean = float((total / pairs).item())
    report.update(n_pairs=pairs, mean_pairwise_tanimoto=mean, diversity=1.-mean)
    return report
