#!/usr/bin/env python
"""Wf-A107 — reproduce the two ring-prior gradient defects on synthetic clouds.

Every number in `molmetal/loss/ring_template.py`'s Wf-A107 docstrings and in
`docs/HONEST_NEGATIVE_CATALOG.md` §6.151 is produced by this script.  It runs
on CPU in a couple of seconds and writes a JSON so the prose stays traceable
(AGENTS.md §5.2).

Two probes:

  probe_1_gradient_collapse
      A 20-atom cloud whose candidate graph is complete.  The bond
      probability is swept from 0.50 (no hard edges, cold start) to 0.90
      (dense hard graph, the over-produced-3-ring regime Wf-A96 measured).
      For each p we report, per formulation, the loss and the gradient norm
      d(loss)/d(bond_logits).  The claim under test: the ``count`` form's
      gradient SHRINKS as cycles appear, while ``per_atom`` does not.

  probe_2_triangle_correction
      A 6-atom soft hexagon (p=0.35) with one weak chord (p=0.01).  The old
      triangle double-count correction subtracted ``0.5 * raw[3]``; on a
      soft graph a triangle contributes ``0.5 * (p1 p2 p3)**2``.  We report
      raw 6-count, both corrections, the resulting 6-count, and whether the
      6-ring term still has a gradient.

Usage:
    python molmetal/scripts/wfa107_ring_gradient_probe.py \
        --out metrics/by_metric/wfa107_ring_gradient_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from molmetal.loss.ring_template import (  # noqa: E402
    RingTemplatePrior,
    RingTemplatePriorConfig,
    _cycle_counts,
)

NO_BOND_CLASS = 0


def _logits_for(bond_prob, n_pairs):
    """5-class logits whose bond probability (1 - P(NO_BOND)) is bond_prob."""
    p = float(bond_prob)
    logits = torch.zeros((n_pairs, 5), dtype=torch.float64)
    # softmax(logit_0, 0, 0, 0, 0)[0] = 1 - p  =>  logit_0 = log(4(1-p)/p)
    logits[:, NO_BOND_CLASS] = math.log(4.0 * (1.0 - p) / p)
    return logits


def _complete_pair_index(n_atoms):
    rows, cols = [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rows.append(i)
            cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


def _hexagon_truth(n_atoms=20, offset=0):
    """Truth = one 6-ring (0-1-2-3-4-5-0), the ring the prior should promote."""
    rows = [(offset + k, offset + (k + 1) % 6) for k in range(6)]
    return torch.tensor([[a for a, _ in rows], [b for _, b in rows]], dtype=torch.long)


def _hard_triangle_count(soft, pair_index, n_atoms):
    hard = (soft > 0.5).to(soft.dtype)
    return float(_cycle_counts(hard, pair_index, (3,), n_atoms)[3])


def probe_1_gradient_collapse(n_atoms=20, weight=0.01):
    pair_index = _complete_pair_index(n_atoms)
    truth = _hexagon_truth(n_atoms)
    n_pairs = pair_index.shape[1]
    out = {"n_atoms": n_atoms, "n_pairs": n_pairs, "weight": weight, "arms": []}
    for p in (0.50, 0.60, 0.70, 0.80, 0.90):
        row = {"bond_prob": p}
        for formulation in ("count", "per_atom"):
            logits = _logits_for(p, n_pairs).requires_grad_(True)
            prior = RingTemplatePrior(
                weight=weight,
                config=RingTemplatePriorConfig(formulation=formulation),
            )
            loss = prior.compute(logits, pair_index, n_atoms, truth)
            loss.backward()
            soft = 1.0 - torch.softmax(logits.detach(), dim=-1)[:, NO_BOND_CLASS]
            row[formulation] = {
                "loss": float(loss.detach()),
                "grad_norm": float(logits.grad.norm()),
                "hard_triangles": _hard_triangle_count(soft, pair_index, n_atoms),
                "hard_rings_per_mol": float(
                    sum(_cycle_counts((soft > 0.5).to(soft.dtype),
                                      pair_index, (3, 4, 5, 6), n_atoms).values())
                ),
            }
        c = row["count"]["grad_norm"]
        pa = row["per_atom"]["grad_norm"]
        row["per_atom_over_count_grad"] = (pa / c) if c > 0 else None
        out["arms"].append(row)
    return out


def probe_2_triangle_correction(hex_p=0.35, chord_p=0.35):
    """6-atom hexagon + one chord closing a triangle; old vs corrected subtraction.

    The chord (0-2) makes atoms 0-1-2 a triangle whose three edges are all in
    the hexagon.  That is the realistic case: in a dense predicted graph most
    hexagon edges also sit in a partial triangle.  With chord_p == hex_p the
    old ``0.5 * raw[3]`` correction exceeds the hexagon's own 6-count and the
    clamp zeroes the term.
    """
    n_atoms = 6
    edges = [(k, (k + 1) % 6) for k in range(6)]  # the hexagon
    edges.append((0, 2))                          # chord -> triangle 0-1-2
    hex_edge_idx = 3                              # edge (3,4): in the hexagon, NOT in the triangle
    chord_idx = 6
    pair_index = torch.tensor([[a for a, _ in edges], [b for _, b in edges]],
                              dtype=torch.long)
    probs = [hex_p] * 6 + [chord_p]
    p_dir = torch.tensor(probs, dtype=torch.float64)

    # Rebuild the internals of _cycle_counts so both corrections can be shown.
    src, dst = pair_index[0], pair_index[1]
    d_src = torch.cat([src, dst])
    d_dst = torch.cat([dst, src])
    p_all = torch.cat([p_dir, p_dir])
    continues = d_dst.unsqueeze(1) == d_src.unsqueeze(0)
    backtracks = d_dst.unsqueeze(0) == d_src.unsqueeze(1)
    b_mat = (continues & ~backtracks).to(p_dir.dtype) * p_all.unsqueeze(0)
    power = b_mat
    raw = {}
    for k in range(2, 7):
        power = power @ b_mat
        raw[k] = float(torch.diagonal(power).sum() / (2.0 * k))

    w = torch.zeros((n_atoms, n_atoms), dtype=p_dir.dtype)
    w[d_src, d_dst] = p_all
    w2 = w * w
    tri_sq = float(torch.trace(w2 @ w2 @ w2) / 6.0)
    tri_unsq = float(torch.trace(w @ w @ w) / 6.0)

    old_value = max(raw[6] - 0.5 * tri_unsq, 0.0)
    new_value = max(raw[6] - 0.5 * tri_sq, 0.0)

    def grad_of_six_count(use_squared: bool):
        """Return (grad on hex edge, grad on chord, value) for the 6-count."""
        logits = torch.zeros((len(probs), 5), dtype=torch.float64)
        for i, p in enumerate(probs):
            logits[i, NO_BOND_CLASS] = math.log(4.0 * (1.0 - p) / p)
        logits.requires_grad_(True)
        soft = 1.0 - torch.softmax(logits, dim=-1)[:, NO_BOND_CLASS]
        if use_squared:
            six = _cycle_counts(soft, pair_index, (6,), n_atoms)[6]
        else:
            # Emulate the pre-A107 form: raw[6] - 0.5 * (triangle product,
            # unsquared).  Rebuilt here because the shipped code no longer
            # exposes the old path.
            src2, dst2 = pair_index[0], pair_index[1]
            d_src2 = torch.cat([src2, dst2])
            d_dst2 = torch.cat([dst2, src2])
            p2 = torch.cat([soft, soft])
            cont = d_dst2.unsqueeze(1) == d_src2.unsqueeze(0)
            back = d_dst2.unsqueeze(0) == d_src2.unsqueeze(1)
            bm = (cont & ~back).to(soft.dtype) * p2.unsqueeze(0)
            pw = bm
            for _ in range(2, 7):
                pw = pw @ bm
            raw6 = torch.diagonal(pw).sum() / 12.0
            ww = torch.zeros((n_atoms, n_atoms), dtype=soft.dtype)
            ww[d_src2, d_dst2] = p2
            tri = torch.trace(ww @ ww @ ww) / 6.0
            six = (raw6 - 0.5 * tri).clamp(min=0.0)
        six.backward()
        g = logits.grad[:, NO_BOND_CLASS]
        return float(g[hex_edge_idx]), float(g[chord_idx]), float(six.detach())

    g_hex_new, g_chord_new, v_new = grad_of_six_count(True)
    g_hex_old, g_chord_old, v_old = grad_of_six_count(False)

    return {
        "n_atoms": n_atoms,
        "hexagon_prob": hex_p,
        "chord_prob": chord_p,
        "hex_edge_index": hex_edge_idx,
        "chord_edge_index": chord_idx,
        "raw_cycle_counts": raw,
        "triangle_term_unsquared": tri_unsq,
        "triangle_term_squared": tri_sq,
        "old_correction": 0.5 * tri_unsq,
        "new_correction": 0.5 * tri_sq,
        "old_six_count": old_value,
        "new_six_count": new_value,
        "hexagon_own_six_count": hex_p ** 6,
        "old_grad_hex_edge": g_hex_old,
        "new_grad_hex_edge": g_hex_new,
        "old_grad_chord": g_chord_old,
        "new_grad_chord": g_chord_new,
        "old_term_alive": bool(abs(g_hex_old) > 0.0),
        "new_term_alive": bool(abs(g_hex_new) > 0.0),
        "shipped_value_matches_new": bool(abs(v_new - new_value) < 1e-12),
        "shipped_value_matches_old": bool(abs(v_old - old_value) < 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/by_metric/wfa107_ring_gradient_probe.json")
    args = ap.parse_args()

    payload = {
        "workflow": "Wf-A107",
        "kind": "synthetic gradient probe (CPU)",
        "probe_1_gradient_collapse": probe_1_gradient_collapse(),
        "probe_2_triangle_correction": probe_2_triangle_correction(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
