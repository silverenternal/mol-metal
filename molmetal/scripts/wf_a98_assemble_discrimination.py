#!/usr/bin/env python3
"""Wf-A98 (2026-10-19) -- can the assemble term SEE fragmentation at all?

WHY THIS EXISTS
---------------
Wf-A97 measured the decode side: 100% of decoded clouds are fragmented
(mean 4.94 heavy-atom components) and the connectivity prior cannot bridge
them because the valence budget is already spent intra-fragment.  The
conclusion was "the fix has to be upstream", i.e. in training.

The only upstream connectivity term in the repo is the assemble term
(escape_time / product on a lazy random walk).  Before spending a GPU run on
it, two things must hold, and neither has been measured:

  (1) the statistic must SEPARATE a fragmented hard graph from a connected
      one on this cohort -- if a 5-component decode scores the same as the
      truth graph, the term is blind and no weight will help;
  (2) the hinge must not floor the term to zero everywhere.

(2) is already known to fail: the shipped tau = 0.30 sits above the model's
own stay_mean, so assemble_loss is exactly 0.0 every epoch.  This probe
measures (1), and re-derives tau on THIS cohort instead of importing tmQM's.

WHAT IT MEASURES, per cloud
---------------------------
  truth        stay_mean on the cloud's OWN Wiberg graph (cloud.bonds)
  soft         stay_mean on the model's softmax graph after ODE integration
  hard         stay_mean on the model's DECODED graph (p in {0,1})
  n_components heavy-atom components of the decoded graph

The diagnostic that decides the question is the OVERLAP between the truth
distribution and the hard distribution.  A useful term needs
hard(fragmented) > truth(connected) by a margin; a blind term has them
interleaved.

No training, no gradient.  One ODE integration per cloud.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]

# One-hot logit magnitude for a synthetic p in {0, 1}.  softmax([0, 20]) gives
# p_bond = 1 - 2.06e-9, which is "hard" to float precision without being inf.
_HARD = 20.0


def _adjacency(edge_index, weights, n_atoms: int) -> torch.Tensor:
    """Symmetric (N, N) soft adjacency from an upper-triangle edge list."""
    adj = torch.zeros((n_atoms, n_atoms), dtype=torch.float32)
    if edge_index is None or int(edge_index.numel()) == 0:
        return adj
    i = edge_index[0].long().cpu()
    j = edge_index[1].long().cpu()
    w = weights.float().cpu()
    keep = (i >= 0) & (j >= 0) & (i < n_atoms) & (j < n_atoms)
    adj.index_put_((i[keep], j[keep]), w[keep], accumulate=True)
    adj.index_put_((j[keep], i[keep]), w[keep], accumulate=True)
    return adj


def _lambda2_ratio(adj: torch.Tensor) -> float:
    """Size-normalised algebraic connectivity of the NORMALISED Laplacian.

    Why this statistic and not stay_mean: see the module docstring.  In short,
    a molecule of five saturated fragments is locally DENSE, so a lazy walk
    leaves each fragment quickly and stay_mean comes out LOW -- i.e. the
    statistic rewards the exact failure mode.  The algebraic connectivity of
    the normalised Laplacian is zero iff the graph is disconnected, whatever
    its density, which is the property the term needs.

    Divided by the path-graph reference 1 - cos(pi/n) (the normalised
    Laplacian lambda2 of P_n), so the number is dimensionless and comparable
    across molecule sizes: ~1 for a chain, 0 for any disconnected graph.
    """
    n = int(adj.shape[0])
    if n < 3:
        return 0.0
    d = adj.sum(dim=1)
    dinv = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
    eye = torch.eye(n, dtype=adj.dtype)
    lap = eye - dinv.unsqueeze(1) * adj * dinv.unsqueeze(0)
    ev = torch.linalg.eigvalsh((lap + lap.t()) * 0.5)
    lam2 = float(ev[1].clamp(min=0.0))
    ref = 1.0 - math.cos(math.pi / n)
    return lam2 / ref if ref > 0 else 0.0


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _logits_from_orders(orders: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """One-hot logits (E, 5) from integer class ids, hard at p in {0, 1}."""
    e = int(orders.shape[0])
    out = torch.full((e, n_classes), -_HARD, dtype=torch.float32)
    out[torch.arange(e), orders.long().clamp(0, n_classes - 1)] = _HARD
    return out


def _components(n_atoms: int, bonds) -> List[List[int]]:
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _o in bonds:
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[rb] = ra
    groups: Dict[int, List[int]] = {}
    for i in range(n_atoms):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "min": s[0],
        "p50": s[len(s) // 2],
        "p90": s[int(0.9 * (len(s) - 1))],
        "max": s[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-clouds", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--method", default="midpoint")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tau", type=float, default=0.30)
    ns = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(ns.config.read_text(encoding="utf-8")) or {}
    ckpt = Path(cfg["model"]["ckpt"])

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
    from wf_blkc_l3_guidance_probe import integrate, load_clouds, load_model

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.assemble_connectivity import (
        mean_stay_probability,
    )
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _aromatic_bias_vector,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a, state = load_model(ckpt, device)
    clouds = load_clouds(a, ns.n_clouds)
    sampled = integrate(
        model,
        clouds,
        device,
        method=ns.method,
        n_steps=ns.n_steps,
        seed=ns.seed,
        noise_sigma=float(a.get("fm_noise_sigma", 1.0)),
        guidance_strength=0.0,
    )
    dec = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=int(a.get("hidden_dim", 128)),
        n_layers=int(a.get("n_layers", 3)),
        decision=str(a.get("bond_decision", "threshold")),
        decision_threshold=float(a.get("bond_decision_threshold", 0.5)),
        class_logit_bias=_aromatic_bias_vector(dict(a)),
        connectivity_prior=True,
        connectivity_prior_min_logit=float(
            a.get("connectivity_prior_min_logit", 0.0)
        ),
        dative_max_distance=float(a.get("dative_distance_cutoff", 2.6)),
    )

    rows: List[Dict[str, Any]] = []
    for cloud, c in zip(clouds, sampled):
        zs_t = torch.tensor(c["zs"], dtype=torch.long)
        ac = AtomCloud(positions=c["positions"], atomic_numbers=zs_t)
        res = dec.decode(ac)
        yres = res.yuelbond_result
        zs = [int(z) for z in c["zs"]]
        n = len(zs)

        # --- truth: the cloud's own Wiberg graph -----------------------
        truth_stay = None
        if getattr(cloud, "bonds", None) is not None and cloud.bonds.numel():
            bt = getattr(cloud, "bond_types", None)
            if bt is not None and int(bt.numel()) == int(cloud.bonds.shape[1]):
                truth_stay = mean_stay_probability(
                    _logits_from_orders(bt),
                    cloud.bonds,
                    cloud.atomic_numbers.long().cpu(),
                )

        # --- model soft graph on the sampled geometry ------------------
        soft_stay = None
        if yres.edge_index is not None and int(yres.edge_index.numel()):
            soft_stay = mean_stay_probability(
                yres.logits.detach().cpu(), yres.edge_index.cpu(), zs_t
            )

        # --- model HARD decoded graph ----------------------------------
        bo = res.decoded.bond_orders
        comps = _components(n, bo)
        comps_heavy = [g for g in comps if any(zs[i] != 1 for i in g)]
        hard_stay = None
        if yres.edge_index is not None and int(yres.edge_index.numel()):
            ei = yres.edge_index.cpu()
            bonded = torch.zeros(int(ei.shape[1]), dtype=torch.long)
            lut = {}
            for k in range(int(ei.shape[1])):
                lut[(int(ei[0, k]), int(ei[1, k]))] = k
                lut[(int(ei[1, k]), int(ei[0, k]))] = k
            for i, j, _o in bo:
                k = lut.get((int(i), int(j)))
                if k is not None:
                    bonded[k] = 1
            hard_stay = mean_stay_probability(
                _logits_from_orders(bonded), ei, zs_t
            )

        # --- lambda2 on the same three graphs --------------------------
        truth_l2 = None
        if getattr(cloud, "bonds", None) is not None and cloud.bonds.numel():
            _bt = getattr(cloud, "bond_types", None)
            if _bt is not None and int(_bt.numel()) == int(cloud.bonds.shape[1]):
                _w = (_bt.long().clamp(0, 4) != 0).float()
                truth_l2 = _lambda2_ratio(
                    _adjacency(cloud.bonds, _w, int(cloud.atomic_numbers.shape[0]))
                )
        soft_l2 = None
        hard_l2 = None
        if yres.edge_index is not None and int(yres.edge_index.numel()):
            _ei = yres.edge_index.cpu()
            _pb = (1.0 - torch.softmax(yres.logits.detach().cpu().float(), dim=-1)[:, 0])
            soft_l2 = _lambda2_ratio(_adjacency(_ei, _pb, n))
            _bonded = torch.zeros(int(_ei.shape[1]))
            _lut = {}
            for _k in range(int(_ei.shape[1])):
                _lut[(int(_ei[0, _k]), int(_ei[1, _k]))] = _k
                _lut[(int(_ei[1, _k]), int(_ei[0, _k]))] = _k
            for _i, _j, _o in bo:
                _k = _lut.get((int(_i), int(_j)))
                if _k is not None:
                    _bonded[_k] = 1.0
            hard_l2 = _lambda2_ratio(_adjacency(_ei, _bonded, n))

        rows.append(
            {
                "n_atoms": n,
                "n_components": len(comps_heavy),
                "truth_stay": truth_stay,
                "soft_stay": soft_stay,
                "hard_stay": hard_stay,
                "truth_l2": truth_l2,
                "soft_l2": soft_l2,
                "hard_l2": hard_l2,
                "truth_pen": None if truth_stay is None else max(0.0, truth_stay - ns.tau),
                "soft_pen": None if soft_stay is None else max(0.0, soft_stay - ns.tau),
                "hard_pen": None if hard_stay is None else max(0.0, hard_stay - ns.tau),
            }
        )

    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    truth = col("truth_stay")
    soft = col("soft_stay")
    hard = col("hard_stay")
    frag_hard = [
        r["hard_stay"] for r in rows
        if r["n_components"] > 1 and r["hard_stay"] is not None
    ]
    conn_hard = [
        r["hard_stay"] for r in rows
        if r["n_components"] <= 1 and r["hard_stay"] is not None
    ]

    # Corpus-derived tau on THIS cohort, using the same rule the published
    # artifact used for tmQM (max * 1.13, rounded up to 0.01) so the two are
    # comparable: 0.2649 * 1.13 = 0.2993 -> 0.30 reproduces the shipped value.
    tau_corpus = None
    if truth:
        tau_corpus = math.ceil(max(truth) * 1.13 * 100.0) / 100.0

    truth_l2 = col("truth_l2")
    hard_l2 = col("hard_l2")
    soft_l2 = col("soft_l2")
    frag_l2 = [
        r["hard_l2"] for r in rows
        if r["n_components"] > 1 and r["hard_l2"] is not None
    ]
    conn_l2 = [
        r["hard_l2"] for r in rows
        if r["n_components"] <= 1 and r["hard_l2"] is not None
    ]

    report = {
        "metric": "wfa98_assemble_discrimination",
        "tag": "MEASURED",
        "date": "2026-10-19",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "n_clouds": len(rows),
        "tau_shipped": ns.tau,
        "stay_mean": {
            "truth": _stats(truth),
            "model_soft": _stats(soft),
            "model_hard": _stats(hard),
            "hard_fragmented": _stats(frag_hard),
            "hard_connected": _stats(conn_hard),
        },
        "tau_derived_this_cohort": tau_corpus,
        "hinge_penalty_at_shipped_tau": {
            "truth_nonzero_frac": (
                sum(1 for v in truth if v > ns.tau) / len(truth) if truth else None
            ),
            "hard_nonzero_frac": (
                sum(1 for v in hard if v > ns.tau) / len(hard) if hard else None
            ),
        },
        "lambda2_ratio": {
            "truth": _stats(truth_l2),
            "model_soft": _stats(soft_l2),
            "model_hard": _stats(hard_l2),
            "hard_fragmented": _stats(frag_l2),
            "hard_connected": _stats(conn_l2),
            "frac_hard_disconnected": (
                sum(1 for v in hard_l2 if v < 1e-3) / len(hard_l2)
                if hard_l2 else None
            ),
            "frac_truth_disconnected": (
                sum(1 for v in truth_l2 if v < 1e-3) / len(truth_l2)
                if truth_l2 else None
            ),
            # The question the term must answer: does the statistic put a
            # fragmented decode below a connected truth graph?  lambda2 is
            # 0 for ANY disconnected graph, so this should be a clean yes.
            "median_hard_fragmented": statistics.median(frag_l2) if frag_l2 else None,
            "median_hard_connected": statistics.median(conn_l2) if conn_l2 else None,
            "median_truth": statistics.median(truth_l2) if truth_l2 else None,
        },
        "separation": {
            "median_hard_fragmented": statistics.median(frag_hard) if frag_hard else None,
            "median_hard_connected": statistics.median(conn_hard) if conn_hard else None,
            "median_truth": statistics.median(truth) if truth else None,
            "blind": (
                None
                if not (frag_hard and truth)
                else bool(statistics.median(frag_hard) <= max(truth))
            ),
        },
        "rows": rows,
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = ns.out.with_suffix(ns.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(tmp, ns.out)
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
