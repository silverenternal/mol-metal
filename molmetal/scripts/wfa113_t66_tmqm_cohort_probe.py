# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T66 — the tmQM cohort, measured on the same axes as the CSV one.

WHY THIS EXISTS
---------------
T57/T58/T60/T61/T63/T64 all measured ``build_csv_cohort`` — the ETKDG path
that embeds each ligand fragment independently and then places the metal at a
bond length from its top donor.  T63 concluded "72% of clouds have exactly one
donor inside 2.6 A" and called the coordination sphere fake.

But the trainer has TWO cohorts (``--dataset {tmqm,csv,synthetic}``), and the
only run that ever produced a generated pool (``c8_fullgen``) trained on
**tmQM**, whose coordinates are DFT-optimised CSD structures.  If the tmQM
cohort already has a real coordination sphere, then T63 is a defect of the CSV
path only, and the production path does not need a relaxation engine.

This probe answers that with the same statistics, on the same metal set.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t66_tmqm_cohort_probe.py \
        --config configs/c8_fullgen_b64_2026_10_24.yaml \
        --out metrics/by_metric/wfa113_t66_tmqm_cohort.json
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

DONOR_Z = frozenset({7, 8, 15, 16, 9, 17, 35, 53})


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t66_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t66_trainer"] = mod
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
                "--out-dir", "/tmp/wfa113_t66", "--ckpt-dir", "/tmp/wfa113_t66"]
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

    n = len(clouds)
    cn_hist: collections.Counter = collections.Counter()
    donors_total = 0
    n_no_metal = 0
    n_clouds_lt1 = 0
    pairs_lt1 = 0
    min_pair_vals: List[float] = []
    ml_min_vals: List[float] = []
    cls_hist: collections.Counter = collections.Counter()
    dative_edges = 0
    edges_at_metal = 0
    aromatic = 0
    n_bonds_none = 0
    m_metal = collections.Counter()

    for c in clouds:
        pos = c.positions.float()
        z = c.atomic_numbers.long()
        midx = int(c.metal_atom_index.item()) if c.metal_atom_index is not None else -1
        if midx < 0:
            n_no_metal += 1
            continue
        m_metal[int(z[midx].item())] += 1
        d = torch.linalg.norm(pos - pos[midx], dim=-1)
        mask = torch.ones_like(d, dtype=torch.bool)
        mask[midx] = False
        # donors = N/O/S/P/halides (same set the builder uses)
        is_donor = torch.zeros_like(mask)
        for zz in DONOR_Z:
            is_donor |= z == zz
        is_donor[midx] = False
        near = mask & is_donor & (d <= 2.6)
        k = int(near.sum().item())
        cn_hist[k] += 1
        donors_total += k
        if k:
            ml_min_vals.append(float(d[near].min().item()))

        # closest non-bonded pair over all atoms
        dd = torch.cdist(pos, pos)
        dd.fill_diagonal_(float("inf"))
        mn = float(dd.min().item())
        min_pair_vals.append(mn)
        n_pairs = int((dd < 1.0).sum().item()) // 2
        if n_pairs:
            n_clouds_lt1 += 1
            pairs_lt1 += n_pairs

        bonds = getattr(c, "bonds", None)
        types = getattr(c, "bond_types", None)
        if bonds is None or types is None:
            n_bonds_none += 1
            continue
        t = types.long()
        cls_hist.update(t.tolist())
        aromatic += int((t == 4).sum().item())
        b = bonds.long()
        edges_at_metal += int(((b[0] == midx) | (b[1] == midx)).sum().item())
        dp = getattr(c, "is_dative_pair", None)
        if dp is not None:
            dative_edges += int(dp.sum().item())

    def _pct(vals, q):
        if not vals:
            return None
        s = sorted(vals)
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[i]

    out: Dict[str, Any] = {
        "probe": "wfa113_t66_tmqm_cohort",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "config": str(args.config),
        "tmqm_dir": root,
        "metals_requested": metals,
        "coord_number_filter": cn,
        "n_clouds": n,
        "n_clouds_without_metal_index": n_no_metal,
        "metal_atomic_numbers": {str(k): v for k, v in sorted(m_metal.items())},
        "donors_within_2p6_histogram": {str(k): v for k, v in sorted(cn_hist.items())},
        "donors_within_2p6_mean": donors_total / max(n - n_no_metal, 1),
        "clouds_with_one_donor_fraction": cn_hist.get(1, 0) / max(n - n_no_metal, 1),
        "clouds_with_ge4_donors_fraction": (
            sum(v for k, v in cn_hist.items() if k >= 4) / max(n - n_no_metal, 1)
        ),
        "metal_ligand_min_A": {
            "min": min(ml_min_vals) if ml_min_vals else None,
            "p10": _pct(ml_min_vals, 0.10),
            "median": _pct(ml_min_vals, 0.50),
            "p90": _pct(ml_min_vals, 0.90),
            "max": max(ml_min_vals) if ml_min_vals else None,
        },
        "min_pair_A": {
            "min": min(min_pair_vals) if min_pair_vals else None,
            "p01": _pct(min_pair_vals, 0.01),
            "median": _pct(min_pair_vals, 0.50),
            "max": max(min_pair_vals) if min_pair_vals else None,
        },
        "clouds_with_pair_below_1A": n_clouds_lt1,
        "pairs_below_1A": pairs_lt1,
        "clouds_without_bonds": n_bonds_none,
        "bond_class_histogram": {str(k): v for k, v in sorted(cls_hist.items())},
        "aromatic_fraction": aromatic / max(sum(cls_hist.values()), 1),
        "edges_at_metal": edges_at_metal,
        "dative_flags_true": dative_edges,
        "dative_pairs": dative_edges // 2,
        "dative_pairs_per_cloud": (dative_edges // 2) / max(n, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
