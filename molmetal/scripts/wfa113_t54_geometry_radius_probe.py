"""Wf-A113 (T54) — the geometry-consistency term has never seen a real radius.

``molmetal/loss/geometry_consistency.py`` scores a predicted geometry against
``eq = r_i + r_j`` with a band of ``[0.85, 1.25] * eq``.  The radii come from
``Batch.atomic_radius``, which ``Batch.collate_fn`` fills with
``_per_atom_field(cloud, "atomic_radius", n, PAD_R)``.  ``AtomCloud`` has no
``atomic_radius`` field and no loader sets one, so the lookup always misses and
every atom gets ``PAD_R = 0.0``, which the loss then clamps to 0.3 A.  ``eq`` is
therefore 0.6 A for every pair of every element.

This probe measures what that does to the TRUE bond graph on the training
cohort: the fraction of real bonds inside the band, and the squared deviation
the term applies, under the uniform 0.3 A radius and under Cordero.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/wfa113_t54_geometry_radius_probe.py \
        --config configs/wfa108l_geom_smoke_2026_10_25.yaml \
        --n-clouds 64 --out metrics/by_metric/wfa113_t54_geometry_radius.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_TRAINER = _REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

BOND_LO, BOND_HI = 0.85, 1.25
UNIFORM_R = 0.3  # geometry_consistency.py clamps radius to this


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-clouds", type=int, default=64)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(argv)

    tr = _load("_t54_trainer", _TRAINER)
    dec = _load(
        "_t54_decoder",
        _REPO / "molmetal" / "molmetal_lam" / "lam_chem" / "yuelbond_decoder.py",
    )

    sys.argv = ["probe", "--config", str(ns.config), "--device", "cpu",
                "--out-dir", "/tmp/wfa113_t54", "--ckpt-dir", "/tmp/wfa113_t54"]
    args = tr.parse_args()
    clouds = tr.build_csv_cohort(
        str(args.data_csv),
        split=str(getattr(args, "data_split", "train")),
        n_clouds=int(ns.n_clouds),
        seed=int(args.seed),
        metal_placement=str(getattr(args, "metal_placement", "bonded")),
        dative_distance_cutoff=float(getattr(args, "dative_distance_cutoff", 2.6)),
    )

    rows = []
    for cloud in clouds:
        if cloud.bonds is None or int(cloud.bonds.shape[1]) == 0:
            continue
        x = cloud.positions.float()
        z = cloud.atomic_numbers.long()
        i = cloud.bonds[0].long()
        j = cloud.bonds[1].long()
        d = (x[i] - x[j]).norm(dim=-1)
        r_true = dec._rcov_per_atom(z)
        eq_true = (r_true[i] + r_true[j]).clamp_min(1e-6)
        eq_unif = torch.full_like(eq_true, 2.0 * UNIFORM_R)
        for tag, eq in (("uniform_0p3", eq_unif), ("cordero", eq_true)):
            rel = d / eq
            dev = (BOND_LO - rel).clamp_min(0.0) + (rel - BOND_HI).clamp_min(0.0)
            rows.append({
                "tag": tag,
                "n_bonds": int(d.numel()),
                "in_band": int(((rel >= BOND_LO) & (rel <= BOND_HI)).sum()),
                "dev_sq_sum": float(dev.pow(2).sum()),
                "rel_mean": float(rel.mean()),
            })

    agg = {}
    for tag in ("uniform_0p3", "cordero"):
        sub = [r for r in rows if r["tag"] == tag]
        n = sum(r["n_bonds"] for r in sub)
        inb = sum(r["in_band"] for r in sub)
        devsq = sum(r["dev_sq_sum"] for r in sub)
        rel_mean = sum(r["rel_mean"] * r["n_bonds"] for r in sub) / max(n, 1)
        agg[tag] = {
            "n_bonds": n,
            "in_band": inb,
            "in_band_fraction": inb / max(n, 1),
            "mean_squared_deviation": devsq / max(2 * n, 1),
            "rel_mean": rel_mean,
        }

    # ---- the term the trainer actually computes -------------------------
    # Run the real function on the real batch, once per radius convention, so
    # the artifact carries the objective value and not only the geometry.
    from molmetal.loss.geometry_consistency import geometry_consistency

    terms = {}
    for tag in ("uniform_0p3", "cordero"):
        radii = []
        for cloud in clouds:
            z = cloud.atomic_numbers.long()
            if tag == "cordero":
                radii.append(dec._rcov_per_atom(z))
            else:
                radii.append(torch.full((int(z.shape[0]),), UNIFORM_R))
        batch = tr.Batch.collate_fn(clouds).to(torch.device("cpu"))
        batch.atomic_radius = torch.cat(radii).float()
        out = geometry_consistency(
            batch.positions.float(),
            batch.atomic_numbers.long(),
            batch.batch_idx,
            batch.bonds,
            batch.atomic_radius,
            int(batch.n_clouds),
            pair_cloud_idx=batch.pair_cloud_idx,
        )
        terms[tag] = {
            "bond": float(out["bond"]),
            "clash": float(out["clash"]),
            "metal": float(out["metal"]),
            "n_bond_pairs": float(out["n_bond_pairs"]),
            "n_clash_pairs": float(out["n_clash_pairs"]),
            "n_metal_pairs": float(out["n_metal_pairs"]),
        }

    payload = {
        "metric": "wfa113_t54_geometry_radius",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "why": (
            "geometry_consistency_weight > 0 turns on a term whose equilibrium "
            "length eq = r_i + r_j is meant to be element-specific. AtomCloud "
            "has no atomic_radius field and no loader sets one, so "
            "Batch.collate_fn fills PAD_R = 0.0 for every atom and the loss "
            "clamps it to 0.3 A: eq = 0.6 A for every pair of every element."
        ),
        "artifact_for": "todo.json T54 / D54",
        "config": str(ns.config),
        "n_clouds": len(clouds),
        "rows": agg,
        "geometry_consistency_term": terms,
        "verdict": (
            "If in_band_fraction is ~0 under uniform_0p3 and ~1 under Cordero, "
            "the term penalises EVERY true bond as too long and would pull the "
            "predicted geometry toward 0.5-0.75 A."
        ),
    }
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
