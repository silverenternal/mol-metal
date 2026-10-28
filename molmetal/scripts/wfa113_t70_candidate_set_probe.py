# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T70 — the bond head's candidate set at train vs at decode.

WHY THIS EXISTS
---------------
The trainer's default is ``flow.bond_candidate_source: truth_only``
(``flow_matching_lipman/__init__.py:3092``): every candidate edge is read
from ``m.bonds``, so the per-pair cross-entropy is 100% positive by
construction and class 0 (``BOND_NO_BOND``) is never in the target.

The decoder builds its candidates differently
(``yuelbond_decoder.py:871``): ``triu_indices(n, n, offset=1)`` masked by
``d_raw < cutoff`` with ``cutoff = 3.5`` (``:623``).  That set is mostly
NO_BOND.  Wf-A67 measured the consequence — ``bond_f1 = 0.265``.

Wf-A69 added ``radius_graph`` + ``subsample_neg_ratio`` as an opt-in.  This
probe measures, on the same cohort the production runs train on, exactly
how far apart the two sets are, and what the ``radius_graph`` arm would
lose if it were switched on at the documented 2.4 A:

  * how many truth bonds sit at or beyond 2.4 A (dropped from the target
    under ``radius_graph``, because the candidate filter is the only thing
    that decides what is labelled);
  * how many sit at or beyond 3.5 A (invisible at decode as well);
  * the positive rate in each of the three candidate sets;
  * the overlap between the 2.4 A and 3.5 A sets.

CPU only.  No model is constructed.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t70_candidate_set_probe.py \
        --config configs/c8_fullgen_b64_2026_10_24.yaml \
        --out metrics/by_metric/wfa113_t70_candidate_sets.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]

DONOR_Z = frozenset({7, 8, 15, 16, 9, 17, 35, 53})
METAL_Z = frozenset({78, 44, 77, 76, 45, 75, 46})  # Pt Ru Ir Os Rh Re Pd

TRAIN_CUTOFF = 2.4   # flow.bond_candidate_cutoff default
DECODE_CUTOFF = 3.5  # YuelBondDecoderHead.cutoff default


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t70_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t70_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _quantiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    t = torch.tensor(vals, dtype=torch.float64)
    q = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], dtype=torch.float64)
    out = torch.quantile(t, q).tolist()
    return {
        "min": float(t.min()),
        "p01": out[0],
        "p10": out[1],
        "median": out[2],
        "p90": out[3],
        "p99": out[4],
        "max": float(t.max()),
        "mean": float(t.mean()),
    }


def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=400)
    args = ap.parse_args()

    tr = _load_trainer()
    sys.argv = ["probe", "--config", args.config, "--device", "cpu",
                "--out-dir", "/tmp/wfa113_t70", "--ckpt-dir", "/tmp/wfa113_t70"]
    cfg = tr.parse_args()

    root = str(getattr(cfg, "tmqm_dir", ".data/molmetal/tmQM"))
    metals = list(getattr(cfg, "tmqm_metals", ["Pt"]))
    cn = int(getattr(cfg, "tmqm_coord_number", -1))
    from molmetal.data.tmqm_clouds import load_tmqm_clouds as _load_tmqm

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

    # ---- accumulators -------------------------------------------------
    n_clouds = 0
    n_truth_bonds_total = 0
    n_truth_ge_train = 0        # truth bond d >= 2.4  -> dropped by radius_graph
    n_truth_ge_decode = 0       # truth bond d >= 3.5  -> invisible at decode too
    truth_dists: List[float] = []
    # per-cloud candidate counts
    per_cloud: List[Dict[str, Any]] = []
    label_hist_truth = collections.Counter()
    # decode-set composition
    dec_cand_total = 0
    dec_pos_total = 0
    rad_cand_total = 0
    rad_pos_total = 0
    # overlap
    jac_num = 0
    jac_den = 0
    # metal-specific: are M-L bonds at risk of the 2.4 A filter?
    ml_dists: List[float] = []
    ml_ge_train = 0
    ml_ge_decode = 0
    # dative-by-rule pairs that are NOT in m.bonds (a decode-only positive)
    rule_dative_not_in_truth = 0
    rule_dative_in_truth = 0
    # truth bonds whose RDKit type maps to class 0
    truth_label_zero = 0
    # positive rate distribution
    dec_pos_rate: List[float] = []
    rad_pos_rate: List[float] = []

    for cloud in clouds:
        if cloud.bonds is None or cloud.bonds.shape[1] == 0:
            continue
        n = int(cloud.atomic_numbers.shape[0])
        if n < 2:
            continue
        n_clouds += 1
        coords = cloud.positions.detach().cpu().double()
        z = cloud.atomic_numbers.detach().cpu().tolist()
        src = cloud.bonds[0].detach().cpu().tolist()
        dst = cloud.bonds[1].detach().cpu().tolist()
        btype = (
            cloud.bond_types.detach().cpu().tolist()
            if cloud.bond_types is not None else [1] * len(src)
        )

        truth: Dict[Tuple[int, int], float] = {}
        for k in range(len(src)):
            i, j = int(src[k]), int(dst[k])
            if i < 0 or j < 0 or i >= n or j >= n:
                continue
            key = _pair_key(i, j)
            d = float((coords[key[0]] - coords[key[1]]).norm())
            if key not in truth:
                truth[key] = d
            else:
                truth[key] = min(truth[key], d)
            label_hist_truth[int(btype[k])] += 1
            if int(btype[k]) not in (1, 2, 3, 4):
                truth_label_zero += 1

        # decode candidate set: all pairs < 3.5
        dmat = torch.cdist(coords, coords)
        iu = torch.triu_indices(n, n, offset=1)
        dec_set = set()
        rad_set = set()
        for k in range(iu.shape[1]):
            i, j = int(iu[0, k]), int(iu[1, k])
            d = float(dmat[i, j])
            if d < DECODE_CUTOFF:
                dec_set.add((i, j))
            if d < TRAIN_CUTOFF:
                rad_set.add((i, j))

        truth_keys = set(truth.keys())
        for key, d in truth.items():
            n_truth_bonds_total += 1
            truth_dists.append(d)
            if d >= TRAIN_CUTOFF:
                n_truth_ge_train += 1
            if d >= DECODE_CUTOFF:
                n_truth_ge_decode += 1
            zi, zj = z[key[0]], z[key[1]]
            if (zi in METAL_Z) != (zj in METAL_Z):
                ml_dists.append(d)
                if d >= TRAIN_CUTOFF:
                    ml_ge_train += 1
                if d >= DECODE_CUTOFF:
                    ml_ge_decode += 1

        n_dec = len(dec_set)
        n_dec_pos = len(dec_set & truth_keys)
        n_rad = len(rad_set)
        n_rad_pos = len(rad_set & truth_keys)
        dec_cand_total += n_dec
        dec_pos_total += n_dec_pos
        rad_cand_total += n_rad
        rad_pos_total += n_rad_pos
        if n_dec:
            dec_pos_rate.append(n_dec_pos / n_dec)
        if n_rad:
            rad_pos_rate.append(n_rad_pos / n_rad)
        union = dec_set | rad_set
        inter = dec_set & rad_set
        jac_num += len(inter)
        jac_den += len(union)

        # dative-by-rule pairs within the decode set
        for (i, j) in dec_set:
            zi, zj = z[i], z[j]
            rule = ((zi in DONOR_Z and zj in METAL_Z)
                    or (zi in METAL_Z and zj in DONOR_Z))
            if rule:
                if (i, j) in truth_keys:
                    rule_dative_in_truth += 1
                else:
                    rule_dative_not_in_truth += 1

        per_cloud.append({
            "n_atoms": n,
            "n_truth": len(truth_keys),
            "n_dec_cand": n_dec,
            "n_dec_pos": n_dec_pos,
            "n_rad_cand": n_rad,
            "n_rad_pos": n_rad_pos,
            "n_truth_dropped_by_2p4": sum(1 for d in truth.values() if d >= TRAIN_CUTOFF),
        })

    out: Dict[str, Any] = {
        "probe": "wfa113_t70_candidate_sets",
        "config": args.config,
        "n_clouds": n_clouds,
        "cutoffs": {"train_radius_graph": TRAIN_CUTOFF,
                    "decode": DECODE_CUTOFF},
        "truth_bonds": {
            "total": n_truth_bonds_total,
            "distance": _quantiles(truth_dists),
            "n_ge_train_cutoff": n_truth_ge_train,
            "frac_ge_train_cutoff": (n_truth_ge_train / n_truth_bonds_total
                                     if n_truth_bonds_total else None),
            "n_ge_decode_cutoff": n_truth_ge_decode,
            "frac_ge_decode_cutoff": (n_truth_ge_decode / n_truth_bonds_total
                                      if n_truth_bonds_total else None),
            "label_hist_rdkit_int": dict(sorted(label_hist_truth.items())),
            "n_label_maps_to_no_bond": truth_label_zero,
        },
        "metal_ligand_bonds": {
            "total": len(ml_dists),
            "distance": _quantiles(ml_dists),
            "n_ge_train_cutoff": ml_ge_train,
            "n_ge_decode_cutoff": ml_ge_decode,
        },
        "candidate_sets": {
            "truth_only": {
                "n_cand": n_truth_bonds_total,
                "n_pos": n_truth_bonds_total - truth_label_zero,
                "positive_rate": (1.0 - truth_label_zero / n_truth_bonds_total
                                  if n_truth_bonds_total else None),
                "n_neg": truth_label_zero,
            },
            "radius_graph_2p4": {
                "n_cand": rad_cand_total,
                "n_pos": rad_pos_total,
                "positive_rate": (rad_pos_total / rad_cand_total
                                  if rad_cand_total else None),
                "per_cloud_positive_rate": _quantiles(rad_pos_rate),
                "n_truth_dropped": n_truth_bonds_total - rad_pos_total,
            },
            "decode_3p5": {
                "n_cand": dec_cand_total,
                "n_pos": dec_pos_total,
                "positive_rate": (dec_pos_total / dec_cand_total
                                  if dec_cand_total else None),
                "per_cloud_positive_rate": _quantiles(dec_pos_rate),
            },
            "jaccard_2p4_vs_3p5": (jac_num / jac_den if jac_den else None),
        },
        "dative_by_rule_in_decode_set": {
            "in_truth_bonds": rule_dative_in_truth,
            "not_in_truth_bonds": rule_dative_not_in_truth,
        },
        "per_cloud": per_cloud,
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2) + "\n")

    cs = out["candidate_sets"]
    tb = out["truth_bonds"]
    ml = out["metal_ligand_bonds"]
    print(f"clouds={n_clouds} truth_bonds={tb['total']} "
          f"(d median {tb['distance']['median']:.3f})")
    print(f"truth_only: pos_rate={cs['truth_only']['positive_rate']:.4f} "
          f"n_neg={cs['truth_only']['n_neg']}")
    print(f"radius_graph@2.4: cand={cs['radius_graph_2p4']['n_cand']} "
          f"pos_rate={cs['radius_graph_2p4']['positive_rate']:.4f} "
          f"dropped_truth={cs['radius_graph_2p4']['n_truth_dropped']}")
    print(f"decode@3.5: cand={cs['decode_3p5']['n_cand']} "
          f"pos_rate={cs['decode_3p5']['positive_rate']:.4f}")
    print(f"jaccard(2.4,3.5)={cs['jaccard_2p4_vs_3p5']:.4f}")
    print(f"truth bonds >=2.4: {tb['n_ge_train_cutoff']} "
          f"({100*tb['frac_ge_train_cutoff']:.2f}%)  "
          f">=3.5: {tb['n_ge_decode_cutoff']} "
          f"({100*tb['frac_ge_decode_cutoff']:.2f}%)")
    print(f"M-L bonds={ml['total']} >=2.4: {ml['n_ge_train_cutoff']} "
          f">=3.5: {ml['n_ge_decode_cutoff']}")
    print(f"rule-dative in decode set: in_truth={rule_dative_in_truth} "
          f"not_in_truth={rule_dative_not_in_truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
