# SPDX-License-Identifier: Apache-2.0
"""Where does a CFM training step actually spend its time? (probe, not a run)

WHY
---
cfm_r9_fix_v2 ran 29,681 s for 11 epochs on n_train=26715 at batch_size=16,
i.e. 1670 optimizer steps per epoch. That is ~45 min/epoch and it is what
makes the full corpus expensive. Before anyone proposes an optimization, the
question is whether the cost is in the bond head's PER-CLOUD message passing
(`DMPNNBondHead.forward` -> `forward_per_atom`, one cloud at a time) or
somewhere else — because those need different fixes.

A comment in pretrain_coordination.py claims block-diagonal batching is "~20x
faster on ROCm". That is an UNMEASURED comment, written for AMD hardware.
This host is NVIDIA (RTX PRO 6000 Blackwell). It is measured here or not at
all.

SCOPE — read the numbers as ratios, not absolutes
-------------------------------------------------
This is a synthetic micro-probe on random clouds of realistic size. It does
not train, does not touch a checkpoint, and does not produce a metric. It
answers exactly one question: how do per-cloud and block-diagonal-batched
forward+backward compare on THIS host, and what fraction of a step the head
occupies. Absolute seconds here are not a prediction of a real run's wall
clock; the ratio is the transferable part.

Equivalence is not assumed either: the batched path is only useful if it
computes the same thing. So the probe also checks that the batched per-atom
hidden states MATCH the concatenated per-cloud ones. If they diverge, the
speedup is a wrong answer rather than a fast one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_CLOUDS = 16          # cfm.yaml's post-fix batch_size
ATOMS_PER_CLOUD = 40   # tmQM complexes: mean heavy-atom count is in this range
HIDDEN = 128           # cfm.yaml model.hidden_dim
LAYERS = 3
WARMUP = 3
ITERS = 12


def make_clouds(n: int, atoms: int, device) -> List:
    from molmetal.models.bond_head import AtomCloud

    g = torch.Generator(device="cpu").manual_seed(0)
    clouds = []
    for _ in range(n):
        pos = torch.randn(atoms, 3, generator=g) * 1.5
        # Realistic-ish heavy-atom mix, biased to the metals in the cohort.
        z = torch.tensor(
            [78] + torch.randint(1, 17, (atoms - 1,), generator=g).tolist(),
            dtype=torch.long,
        )
        clouds.append(AtomCloud(positions=pos.to(device), atomic_numbers=z.to(device)))
    return clouds


def timeit(fn, device) -> float:
    for _ in range(WARMUP):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from molmetal.models.dmpnn import MPNNConfig, DirectedMPNN

    cfg = MPNNConfig(
        atom_feat_dim=HIDDEN, edge_feat_dim=3, hidden_dim=HIDDEN,
        n_layers=LAYERS, dropout=0.0,
    )
    net = DirectedMPNN(cfg).to(device)

    clouds = make_clouds(N_CLOUDS, ATOMS_PER_CLOUD, device)
    n_atoms = sum(int(c.atomic_numbers.shape[0]) for c in clouds)

    # --- edges, built ONCE and shared by both arms ---
    # Deliberately the SAME construction the head itself uses (upper-triangle
    # pairs under the cutoff, then BI-DIRECTIONAL, per the DMPNNBondHead
    # comment). Both arms consume these identical edges, so the batched arm is
    # nothing but the concatenation with an atom offset — equivalence holds by
    # construction, and the only thing being measured is 16 kernel launches
    # versus 1.
    #
    # An earlier version went through the adapter's radius_edge_index and
    # reported a 1.52x speedup with equivalence max|d| = 1.33. That number was
    # worthless: radius_edge_index returns per-row LOCAL indices, and the
    # // M and % M used to move them into the flat space corrupted the edge
    # set, so the two arms were not computing the same function. A speedup
    # measured across different graphs is a wrong answer, not a fast one.
    cutoff = 3.5
    offsets, run = [], 0
    for c in clouds:
        offsets.append(run)
        run += int(c.atomic_numbers.shape[0])
    n_atoms = run
    pos_all = torch.cat([c.positions for c in clouds], dim=0)

    per_cloud_ei, per_cloud_ea = [], []
    for ci, c in enumerate(clouds):
        m = int(c.atomic_numbers.shape[0])
        ii, jj = torch.triu_indices(m, m, offset=1, device=device)
        d = (c.positions[jj] - c.positions[ii]).norm(dim=-1).clamp(min=1.0)
        keep = d < cutoff
        ii, jj, d = ii[keep], jj[keep], d[keep]
        # LOCAL indices. forward_per_atom sets N = h_atom.size(0) and indexes
        # h_v[src], so handing it global indices (up to n_atoms-1) against a
        # 40-row h is an out-of-bounds device assert — which is exactly what an
        # earlier version of this probe hit.
        per_cloud_ei.append(
            torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0)
        )
        d_bi = torch.cat([d, d])
        per_cloud_ea.append(
            torch.stack(
                [d_bi, torch.zeros_like(d_bi), (d_bi < 1.7).float()], dim=-1
            )
        )
    # Batched arm only: shift each cloud's edges into the flat atom space.
    edge_index = torch.cat(
        [ei + offsets[i] for i, ei in enumerate(per_cloud_ei)], dim=1
    )
    edge_attr = torch.cat(per_cloud_ea, dim=0)
    h_all = torch.randn(n_atoms, HIDDEN, device=device, requires_grad=True)

    def per_cloud():
        outs = []
        for i, c in enumerate(clouds):
            m = int(c.atomic_numbers.shape[0])
            outs.append(net.forward_per_atom(
                h_all[offsets[i]:offsets[i] + m], per_cloud_ei[i], per_cloud_ea[i]
            ))
        torch.cat(outs, 0).square().mean().backward()

    def batched():
        h = h_all.detach().clone().requires_grad_(True)
        net.forward_per_atom(h, edge_index, edge_attr).square().mean().backward()

    t_per = timeit(per_cloud, device)
    t_bat = timeit(batched, device)

    # --- equivalence: batched must equal the per-cloud concatenation ---
    with torch.no_grad():
        h = h_all.detach().clone()
        hb = net.forward_per_atom(h, edge_index, edge_attr)
        parts = []
        for i, c in enumerate(clouds):
            m = int(c.atomic_numbers.shape[0])
            parts.append(net.forward_per_atom(
                h[offsets[i]:offsets[i] + m], per_cloud_ei[i], per_cloud_ea[i]
            ))
        hc = torch.cat(parts, 0)
        max_abs = float((hb - hc).abs().max())

    print()
    print(f"  clouds={N_CLOUDS}  atoms={n_atoms}  edges={edge_index.shape[1]}  hidden={HIDDEN}")
    print(f"  per-cloud  forward+backward : {t_per*1000:8.2f} ms")
    print(f"  block-diag forward+backward : {t_bat*1000:8.2f} ms")
    print(f"  speedup                    : {t_per/t_bat:8.2f}x")
    print(f"  equivalence max|Δ|          : {max_abs:.3e}   "
          f"({'MATCH' if max_abs < 1e-4 else 'MISMATCH — speedup would be a wrong answer'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())