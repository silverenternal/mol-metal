"""Pure batch-collation helper for multi-metal D-MPNN training —
extracted from `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
(lines 1056-1090 in the pre-split trainer; the trainer re-exports
`collate_multimetal` for backwards compatibility).

Wf-??? monolith-split audit (2026-10-05): this function is a pure leaf:
no module-global reads, no `self.*`, only numpy + torch + a lazy
import of `molmetal.baselines.dmpnn.collate_graphs`. Safe to extract;
behaviour-preserving move (function body unchanged).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


def collate_multimetal(
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    pic50: np.ndarray,
    active: np.ndarray,
):
    """Collate a multi-metal batch.

    Returns ``(atom_feats, bond_feats, edge_src, edge_dst, metal_ids_t,
    pic50_t, active_t)`` — any with ``None`` graph is dropped.
    """
    from molmetal.baselines.dmpnn import collate_graphs

    valid = []
    for g, m, p, a in zip(graphs, metal_ids, pic50, active):
        if g is None:
            continue
        # NaN check that handles both Python floats and numpy scalars
        try:
            if math.isnan(float(p)):
                continue
        except (TypeError, ValueError):
            continue
        valid.append((g, m, p, a))
    if not valid:
        return None
    g_list = [v[0] for v in valid]
    batch_data = collate_graphs(g_list)
    if batch_data is None:
        return None
    atom_f, bond_f, e_src, e_dst = batch_data
    metal_ids_t = torch.tensor([v[1] for v in valid], dtype=torch.long)
    pic50_t = torch.tensor([v[2] for v in valid], dtype=torch.float32)
    active_t = torch.tensor([v[3] for v in valid], dtype=torch.long)
    return atom_f, bond_f, e_src, e_dst, metal_ids_t, pic50_t, active_t
