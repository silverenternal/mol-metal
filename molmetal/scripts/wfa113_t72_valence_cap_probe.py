# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T72 — does the decoder's valence cap fit the corpus it decodes?

WHY THIS EXISTS
---------------
``YuelBondBondDecoder._select_bonds`` enforces two ceilings on every bond it
keeps, and both are per-ELEMENT constants:

  * ``_COVALENT_VALENCE`` (cfm_decoder_yuelbond.py:135) — Pt 4, Ru 6, Ir 6,
    Rh 6, Os 6, Re 6.
  * ``_geometry_constraint_for_z`` — square-planar for Pt/Pd (max_bonds 4),
    octahedral for Ru/Ir/Rh/Os/Re.

The corpus is tmQM: DFT-optimised CSD structures, coordination numbers 2-7.
If a cloud's true metal degree exceeds its cap, the selector can keep at most
``cap`` bonds to that metal no matter what the head predicts.  The remaining
donors are then left unbonded, the decoded molecule is multi-fragment, and
``molecule_is_valid`` (single fragment + seed metal) is False by construction.

This probe measures the gap on the same 400-cloud cohort the other T6x probes
use, so the number is comparable rather than a fresh sample.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t72_valence_cap_probe.py \
        --config configs/c8_fullgen_b64_2026_10_24.yaml \
        --out metrics/by_metric/wfa113_t72_valence_cap.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

_REPO = Path(__file__).resolve().parents[2]


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t72_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t72_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=400)
    args = ap.parse_args()

    tr = _load_trainer()
    sys.argv = ["probe", "--config", args.config, "--device", "cpu",
                "--out-dir", "/tmp/wfa113_t72", "--ckpt-dir", "/tmp/wfa113_t72"]
    cfg = tr.parse_args()

    from molmetal.data.tmqm_clouds import load_tmqm_clouds as _load_tmqm
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        covalent_valence_table,
    )
    from molmetal.molmetal_lam.priors.metal_geometry import (
        _geometry_constraint_for_z,
    )

    root = str(getattr(cfg, "tmqm_dir", ".data/molmetal/tmQM"))
    metals = list(getattr(cfg, "tmqm_metals", ["Pt"]))
    cn = int(getattr(cfg, "tmqm_coord_number", -1))
    clouds = _load_tmqm(
        root,
        metals=metals,
        coord_number=None if cn < 0 else cn,
        n_clouds=int(args.n_clouds),
        seed=int(cfg.seed),
        split="train",
        val_frac=float(getattr(cfg, "tmqm_val_frac", 0.2)),
        min_per_metal=int(getattr(cfg, "tmqm_min_per_metal", 0)),
        split_kind=str(getattr(cfg, "tmqm_split_kind", "random")),
    )

    cap = covalent_valence_table()
    # The caps the decoder shipped before T72.  Kept as a literal so the
    # artifact can state the before/after without a git checkout.
    CAPS_BEFORE = {78: 4, 44: 6, 77: 6, 45: 6, 76: 6, 75: 6}
    Z_SYM = {78: "Pt", 44: "Ru", 77: "Ir", 45: "Rh", 76: "Os", 75: "Re", 46: "Pd"}

    # per-metal: degree histogram, cap, n over cap, bonds lost
    deg_hist: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_clouds_by_metal: collections.Counter = collections.Counter()
    n_over_cap: collections.Counter = collections.Counter()
    n_over_before: collections.Counter = collections.Counter()
    bonds_total: collections.Counter = collections.Counter()
    bonds_lost: collections.Counter = collections.Counter()
    bonds_lost_before: collections.Counter = collections.Counter()
    examples: List[Dict[str, Any]] = []
    # cross-tab: does the geometry gate bind before the valence gate?
    geom_binds: collections.Counter = collections.Counter()
    # pi-bound detection: metal bonded to >= 2 carbons of the SAME ring.
    # That is the eta6-arene / eta5-Cp signature, and it is what makes the
    # metal degree exceed the octahedral cap without the complex actually
    # having more than 6 ligands.
    mc_bond_hist: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_pi_bound: collections.Counter = collections.Counter()
    pi_degree: List[int] = []

    def _ring_of_carbons(b: torch.Tensor, z: torch.Tensor) -> Dict[int, int]:
        """Map carbon index -> ring id for all C-C rings of size 4..8."""
        c_idx = [int(v) for v in torch.nonzero(z == 6).flatten().tolist()]
        cset = set(c_idx)
        adj: Dict[int, set] = {i: set() for i in c_idx}
        for k in range(b.shape[1]):
            a, bb = int(b[0, k]), int(b[1, k])
            if a in cset and bb in cset and a != bb:
                adj[a].add(bb)
                adj[bb].add(a)
        ring_id: Dict[int, int] = {}
        seen_edges = set()
        rid = 0
        for u in c_idx:
            for v in list(adj[u]):
                e = (min(u, v), max(u, v))
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                # BFS from u to v with this edge removed
                prev = {u: None}
                q = [u]
                found = False
                while q and not found:
                    nq = []
                    for x in q:
                        for y in adj[x]:
                            if (x == u and y == v) or (x == v and y == u):
                                continue
                            if y in prev:
                                continue
                            prev[y] = x
                            if y == v:
                                found = True
                                break
                            nq.append(y)
                        if found:
                            break
                    q = nq
                if not found:
                    continue
                path = [v]
                cur = prev[v]
                while cur is not None and cur != u:
                    path.append(cur)
                    cur = prev[cur]
                size = len(path) + 1
                if 4 <= size <= 8:
                    for a in path + [u]:
                        ring_id.setdefault(a, rid)
                    rid += 1
        return ring_id

    for ci, c in enumerate(clouds):
        z = c.atomic_numbers.long()
        midx = (
            int(c.metal_atom_index.item())
            if getattr(c, "metal_atom_index", None) is not None
            else -1
        )
        if midx < 0:
            continue
        mz = int(z[midx].item())
        sym = Z_SYM.get(mz, str(mz))
        bonds = getattr(c, "bonds", None)
        types = getattr(c, "bond_types", None)
        if bonds is None or types is None:
            continue
        b = bonds.long()
        at_metal = (b[0] == midx) | (b[1] == midx)
        deg = int(at_metal.sum().item())
        n_clouds_by_metal[sym] += 1
        deg_hist[sym][deg] += 1
        bonds_total[sym] += deg
        c_cap = cap.get(mz, 4)
        c_before = CAPS_BEFORE.get(mz, c_cap)
        gc = _geometry_constraint_for_z(mz)
        c_geom = int(gc.max_bonds) if gc is not None else None
        # Before T72 the effective ceiling was min(valence, geometry).  The
        # geometry gate is gone now, so the ceiling is the valence cap alone.
        eff_before = min(c_before, c_geom) if c_geom is not None else c_before
        eff = c_cap
        # --- pi-bound signature -------------------------------------------
        z_l = z
        mc = [int(b[0, k]) if int(b[1, k]) == midx else int(b[1, k])
              for k in range(b.shape[1])
              if (int(b[0, k]) == midx or int(b[1, k]) == midx)
              and int(z_l[int(b[0, k]) if int(b[1, k]) == midx else int(b[1, k])]) == 6]
        mc_bond_hist[sym][len(mc)] += 1
        if len(mc) >= 2:
            ring_id = _ring_of_carbons(b, z_l)
            by_ring: Dict[int, int] = collections.Counter()
            for a in mc:
                if a in ring_id:
                    by_ring[ring_id[a]] += 1
            if by_ring and max(by_ring.values()) >= 2:
                n_pi_bound[sym] += 1
                pi_degree.append(deg)
        if deg > eff_before:
            n_over_before[sym] += 1
            bonds_lost_before[sym] += deg - eff_before
        if deg > eff:
            n_over_cap[sym] += 1
            bonds_lost[sym] += deg - eff
            if len(examples) < 12:
                examples.append(
                    {
                        "cloud": ci,
                        "metal": sym,
                        "degree": deg,
                        "valence_cap": c_cap,
                        "geometry_max_bonds": c_geom,
                        "effective_cap": eff,
                        "lost": deg - eff,
                    }
                )
        geom_binds[sym + ":" + ("geom<val" if c_geom is not None and c_geom < c_cap else "val<=geom")] += 1

    out: Dict[str, Any] = {
        "probe": "wfa113_t72_valence_cap",
        "config": args.config,
        "n_clouds": len(clouds),
        "per_metal": {},
        "examples": examples,
        "geom_vs_valence": dict(sorted(geom_binds.items())),
    }
    tot_clouds = tot_over = tot_over_before = 0
    tot_bonds = tot_lost = tot_lost_before = 0
    for sym in sorted(n_clouds_by_metal):
        nc = n_clouds_by_metal[sym]
        tot_clouds += nc
        tot_over += n_over_cap[sym]
        tot_bonds += bonds_total[sym]
        tot_lost += bonds_lost[sym]
        tot_over_before += n_over_before[sym]
        tot_lost_before += bonds_lost_before[sym]
        mz = {v: k for k, v in Z_SYM.items()}[sym]
        out["per_metal"][sym] = {
            "n_clouds": nc,
            "valence_cap": cap.get(mz, 4),
            "valence_cap_before_t72": c_before,
            "degree_hist": dict(sorted(deg_hist[sym].items())),
            "degree_mean": round(bonds_total[sym] / max(nc, 1), 3),
            "n_clouds_over_effective_cap": n_over_cap[sym],
            "frac_clouds_over_effective_cap": round(n_over_cap[sym] / max(nc, 1), 4),
            "n_clouds_over_cap_before_t72": n_over_before[sym],
            "metal_bonds_unrepresentable_before_t72": bonds_lost_before[sym],
            "metal_bonds_total": bonds_total[sym],
            "metal_bonds_unrepresentable": bonds_lost[sym],
            "frac_metal_bonds_unrepresentable": round(
                bonds_lost[sym] / max(bonds_total[sym], 1), 4
            ),
            "n_metal_carbon_bonds_hist": dict(sorted(mc_bond_hist[sym].items())),
            "n_clouds_pi_bound": n_pi_bound[sym],
            "frac_clouds_pi_bound": round(n_pi_bound[sym] / max(nc, 1), 4),
        }
    out["pooled"] = {
        "n_clouds": tot_clouds,
        "n_clouds_over_effective_cap": tot_over,
        "frac_clouds_over_effective_cap": round(tot_over / max(tot_clouds, 1), 4),
        "metal_bonds_total": tot_bonds,
        "metal_bonds_unrepresentable": tot_lost,
        "frac_metal_bonds_unrepresentable": round(tot_lost / max(tot_bonds, 1), 4),
        "n_clouds_over_cap_before_t72": tot_over_before,
        "frac_clouds_over_cap_before_t72": round(tot_over_before / max(tot_clouds, 1), 4),
        "metal_bonds_unrepresentable_before_t72": tot_lost_before,
        "frac_metal_bonds_unrepresentable_before_t72": round(
            tot_lost_before / max(tot_bonds, 1), 4
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(json.dumps(out["pooled"], indent=1))
    for sym, row in out["per_metal"].items():
        print(sym, "cap", row["valence_cap"], "deg", row["degree_hist"],
              "over", row["n_clouds_over_effective_cap"], "/", row["n_clouds"],
              "lost", row["metal_bonds_unrepresentable"], "/", row["metal_bonds_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
