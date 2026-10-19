#!/usr/bin/env python3
"""Wf-A97 (2026-10-19) -- why the connectivity prior does not connect.

THE SHORT BOARD THIS EXISTS FOR
-------------------------------
Both Wf-A96 measurements end with the same number:

    L3 decode (64 clouds):      single_fragment_ratio = 0.047
    OPT-5 5-epoch gen eval:     gen_single_fragment_ratio = 0.000 (every epoch)

with connectivity_prior = on in both runs.  A postprocess whose whole job
is to join components is on, and the molecules are still fragments.  The
gate metrics call this "decode_rate 1.000", which counts sanitised strings.

WHAT IT MEASURES
----------------
For each decoded cloud:
  * components before/after the prior, and bonds_added_connectivity,
  * the MINIMUM inter-component atom-atom distance,
  * whether any inter-component pair exists in the head's own edge_index
    (the ONLY pairs the prior is allowed to bridge),
  * the remaining valence at the two closest atoms of each component pair.

The three candidate diagnoses are distinguishable:
  (a) no candidate pair crosses components  -> the prior cannot act, ever;
      the model puts fragments further apart than the head's cutoff;
  (b) a candidate exists but valence is exhausted -> the selector already
      spent the budget on intra-fragment bonds;
  (c) a candidate exists and valence is free -> the prior's own rule is
      broken and the bug is in _bridge_components.

No training.  One forward ODE integration per cloud.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-clouds", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--method", default="midpoint")
    ap.add_argument("--seed", type=int, default=1234)
    ns = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(ns.config.read_text(encoding="utf-8")) or {}
    ckpt = Path(cfg["model"]["ckpt"])

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
    from wf_blkc_l3_guidance_probe import integrate, load_clouds, load_model

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
        _CLASS_VALENCE_COST,
    )
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _aromatic_bias_vector,
        decode_chemistry,
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
        ac = AtomCloud(
            positions=c["positions"],
            atomic_numbers=torch.tensor(c["zs"], dtype=torch.long),
        )
        res = dec.decode(ac)
        bo = res.decoded.bond_orders
        zs = [int(z) for z in c["zs"]]
        n = len(zs)
        comps = _components(n, bo)
        comps_heavy = [g for g in comps if any(zs[i] != 1 for i in g)]
        pos = c["positions"].detach().cpu()

        remaining = [dec._cap_for_z(int(z)) for z in zs]
        for i, j, order in bo:
            cost = _CLASS_VALENCE_COST.get(int(order), 1.0)
            remaining[i] -= cost
            remaining[j] -= cost

        ei = res.yuelbond_result.edge_index
        cand = set()
        if ei is not None and ei.numel():
            for k in range(ei.shape[1]):
                i, j = int(ei[0][k]), int(ei[1][k])
                if i != j:
                    cand.add((i, j) if i < j else (j, i))

        best = None
        for x in range(len(comps_heavy)):
            for y in range(x + 1, len(comps_heavy)):
                for i in comps_heavy[x]:
                    for j in comps_heavy[y]:
                        d = float((pos[i] - pos[j]).norm())
                        key = (i, j) if i < j else (j, i)
                        rec = {
                            "d": d,
                            "i": i,
                            "j": j,
                            "in_candidates": key in cand,
                            "budget": min(remaining[i], remaining[j]),
                        }
                        if best is None or d < best["d"]:
                            best = rec
        info = decode_chemistry(res.decoded.smiles or "") or {}
        rows.append(
            {
                "n_atoms": n,
                "n_components": len(comps_heavy),
                "bonds_added_connectivity": int(res.bonds_added_connectivity),
                "n_fragments": info.get("n_fragments"),
                "closest_inter_component": best,
            }
        )

    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    frag = [r for r in rows if r["n_components"] > 1]
    report = {
        "metric": "wfa97_fragmentation_diagnosis",
        "tag": "MEASURED",
        "date": "2026-10-19",
        "checkpoint": str(ckpt),
        "n_clouds": len(rows),
        "connectivity_prior": True,
        "summary": {
            "frac_fragmented": len(frag) / len(rows) if rows else None,
            "mean_components": _mean("n_components"),
            "mean_bonds_added_connectivity": _mean("bonds_added_connectivity"),
            "n_clouds_with_zero_bridges": sum(
                1 for r in frag if r["bonds_added_connectivity"] == 0
            ),
            "frac_fragmented_with_candidate_bridge": (
                sum(
                    1
                    for r in frag
                    if r["closest_inter_component"]
                    and r["closest_inter_component"]["in_candidates"]
                )
                / len(frag)
                if frag
                else None
            ),
            "frac_fragmented_with_candidate_and_budget": (
                sum(
                    1
                    for r in frag
                    if r["closest_inter_component"]
                    and r["closest_inter_component"]["in_candidates"]
                    and r["closest_inter_component"]["budget"] >= 1.0
                )
                / len(frag)
                if frag
                else None
            ),
            "median_closest_inter_component_distance": (
                statistics.median(
                    r["closest_inter_component"]["d"]
                    for r in frag
                    if r["closest_inter_component"]
                )
                if frag
                else None
            ),
        },
        "rows": rows,
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = ns.out.with_suffix(ns.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(tmp, ns.out)
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
