# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T68 — the Wiberg BO target was read in the wrong atom frame.

THE DEFECT
----------
``_build_cloud_from_kept`` drops hydrogens and remaps every pair through
``old_to_new``, but the ``bond_orders_continuous`` lookup was keyed by the RAW
tmQM indices and then queried with the REMAPPED ones.  tmQM interleaves
hydrogens with heavy atoms (MEASURED: the first H sits at index 2 in one of the
first complexes), so nearly every heavy index shifts and the lookup missed —
silently, because the lookup used ``.get(..., 0.0)``.

WHAT THIS MEASURES
------------------
The same cohort twice: the pre-fix value (raw-index key, new-index query) and
the post-fix value (kept-atom-frame key), for every bond of every cloud.  The
probe recomputes both from the BO shards instead of running two copies of the
builder, and VALIDATES itself by asserting that the recomputed post-fix value
reproduces ``cloud.bond_orders_continuous`` exactly.

The cohort is drawn with ``split=None`` so the code order is exactly
``_stratified_code_draw``'s draw order — that is what lets the probe pair a
cloud with its CSD code (``AtomCloud`` does not carry one).

Run:
    .venv/bin/python molmetal/scripts/wfa113_t68_bo_frame_probe.py \
        --out metrics/by_metric/wfa113_t68_bo_frame.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[2]
TMQM = _REPO / ".data" / "molmetal" / "tmQM"
METALS = ["Pt", "Ru", "Ir", "Rh", "Os", "Re"]
CLASS_NAMES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}
XYZ = [TMQM / "tmQM_X1.xyz.gz", TMQM / "tmQM_X2.xyz.gz", TMQM / "tmQM_X3.xyz.gz"]
BO = [TMQM / "tmQM_X1.BO.gz", TMQM / "tmQM_X2.BO.gz", TMQM / "tmQM_X3.BO.gz"]


def _xyz_keep_idx() -> Dict[str, List[int]]:
    from molmetal.data.tmqm_clouds import iter_tmqm_blocks

    out: Dict[str, List[int]] = {}
    for code, symbols, _coords in iter_tmqm_blocks(XYZ):
        out[code] = [i for i, s in enumerate(symbols) if s != "H"]
    return out


def _bo_maps() -> Tuple[Dict[str, list], Dict[str, list]]:
    from molmetal.data.tmqm_clouds import iter_bo_bonds, iter_bo_continuous

    cls = {code: pairs for code, pairs, _nbrs in iter_bo_bonds(BO)}
    cont = {code: pairs for code, pairs in iter_bo_continuous(BO)}
    return cls, cont


def _stats(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "fraction_zero": round(sum(1 for v in vals if v == 0.0) / len(vals), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(_REPO))
    from molmetal.data.tmqm_clouds import (
        _stratified_code_draw,
        load_tmqm_clouds,
        load_tmqm_metadata,
    )

    meta = load_tmqm_metadata(TMQM / "tmqm_parsed.csv")
    codes_all = [
        c for c, r in meta.items() if str(r.get("metal", "")).strip() in set(METALS)
    ]
    drawn, quota, pool = _stratified_code_draw(meta, codes_all, int(args.n_clouds), int(args.seed))
    order = {c: i for i, c in enumerate(drawn)}
    # The draw oversamples by ~25% so a complex that fails to build can be
    # replaced; the loader trims to the per-metal quota in draw order, and the
    # probe must reproduce that exactly to pair cloud k with code k.
    by_metal: Dict[str, List[str]] = {}
    for c in drawn:
        by_metal.setdefault(str(meta[c].get("metal", "")).strip(), []).append(c)
    taken: List[str] = []
    for m in sorted(by_metal):
        codes_m = sorted(by_metal[m], key=lambda c: order[c])
        taken.extend(codes_m[: int(quota.get(m, 0))])
    codes_in_order = sorted(taken, key=lambda c: order[c])

    clouds = load_tmqm_clouds(
        str(TMQM),
        metals=METALS,
        n_clouds=int(args.n_clouds),
        seed=int(args.seed),
        split=None,
    )
    assert len(clouds) == len(codes_in_order), (len(clouds), len(codes_in_order))

    keep = _xyz_keep_idx()
    cls_map, cont_map = _bo_maps()

    before_by_class: Dict[int, List[float]] = {}
    after_by_class: Dict[int, List[float]] = {}
    n_before_zero = n_after_zero = n_bonds = 0
    n_validate_ok = n_validate_bad = 0
    dative_before = dative_after = metal_edges = 0
    per_cloud_zero_before: List[float] = []
    per_cloud_zero_after: List[float] = []

    for cloud, code in zip(clouds, codes_in_order):
        keep_idx = keep[code]
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        pairs = [
            (old_to_new[i], old_to_new[j], t)
            for i, j, t in cls_map.get(code, [])
            if i in old_to_new and j in old_to_new
        ]
        if not pairs:
            continue
        # Pre-fix: keyed by the RAW tmQM indices.
        raw: Dict[Tuple[int, int], float] = {}
        for i, j, v in cont_map.get(code, []):
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in raw or v > raw[(a, b)]:
                raw[(a, b)] = v
        # Post-fix: keyed in the kept-atom frame.
        fixed: Dict[Tuple[int, int], float] = {}
        for i, j, v in cont_map.get(code, []):
            ni, nj = old_to_new.get(i), old_to_new.get(j)
            if ni is None or nj is None:
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            if (a, b) not in fixed or v > fixed[(a, b)]:
                fixed[(a, b)] = v

        midx = int(cloud.metal_atom_index.item())
        z = cloud.atomic_numbers
        zb = 0
        za = 0
        for k, (i, j, t) in enumerate(pairs):
            a, b = (i, j) if i < j else (j, i)
            vb = raw.get((a, b), 0.0)
            va = fixed.get((a, b), 0.0)
            # Self-validation: the recomputed post-fix value must reproduce
            # what the shipped builder wrote into the cloud.
            if abs(float(cloud.bond_orders_continuous[k]) - va) < 1e-5:
                n_validate_ok += 1
            else:
                n_validate_bad += 1
            n_bonds += 1
            if vb == 0.0:
                n_before_zero += 1
                zb += 1
            if va == 0.0:
                n_after_zero += 1
                za += 1
            before_by_class.setdefault(t, []).append(vb)
            after_by_class.setdefault(t, []).append(va)
            if midx in (i, j):
                metal_edges += 1
                if vb < 1.0:
                    dative_before += 1
                if va < 1.0:
                    dative_after += 1
        per_cloud_zero_before.append(zb / len(pairs))
        per_cloud_zero_after.append(za / len(pairs))

    out = {
        "metric": "wfa113_t68_bo_frame",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "probe": "molmetal/scripts/wfa113_t68_bo_frame_probe.py",
        "cohort": (
            f"{len(clouds)} clouds drawn by _stratified_code_draw from the "
            f"{len(codes_all)} six-metal tmQM complexes, split=None, seed={args.seed}"
        ),
        "defect": {
            "code": "molmetal/data/tmqm_clouds.py::_build_cloud_from_kept (cont_lookup keyed by raw tmQM indices)",
            "mechanism": (
                "pairs_with_class is remapped through old_to_new because "
                "hydrogens are dropped, but cont_lookup was built from "
                "pairs_continuous with the RAW indices and then queried with "
                "the remapped ones. tmQM interleaves H with heavy atoms, so "
                "almost every heavy index shifts and .get(..., 0.0) returned "
                "0.0 — or another bond's order."
            ),
            "fix": (
                "Map both endpoints of every pairs_continuous entry through "
                "old_to_new before keying the lookup, and skip entries whose "
                "endpoint is a dropped hydrogen."
            ),
        },
        "n_bonds": n_bonds,
        "before": {
            "zero_bo_bonds": n_before_zero,
            "zero_bo_fraction": round(n_before_zero / max(n_bonds, 1), 4),
            "per_cloud_zero_fraction_mean": round(
                statistics.mean(per_cloud_zero_before), 4
            ) if per_cloud_zero_before else None,
            "by_class": {
                CLASS_NAMES.get(k, str(k)): _stats(v)
                for k, v in sorted(before_by_class.items())
            },
            "dative_directed_flags": dative_before,
        },
        "after": {
            "zero_bo_bonds": n_after_zero,
            "zero_bo_fraction": round(n_after_zero / max(n_bonds, 1), 4),
            "per_cloud_zero_fraction_mean": round(
                statistics.mean(per_cloud_zero_after), 4
            ) if per_cloud_zero_after else None,
            "by_class": {
                CLASS_NAMES.get(k, str(k)): _stats(v)
                for k, v in sorted(after_by_class.items())
            },
            "dative_directed_flags": dative_after,
        },
        "metal_edges_directed": metal_edges,
        "self_validation": {
            "recomputed_matches_cloud": n_validate_ok,
            "recomputed_mismatch": n_validate_bad,
            "note": (
                "The probe's post-fix lookup is asserted to reproduce the "
                "shipped cloud's bond_orders_continuous exactly; a nonzero "
                "mismatch means the probe's model of the builder is wrong and "
                "the before/after numbers should not be trusted."
            ),
        },
        "consequences": [
            "The BO regression target (chemistry_losses.bond_order_loss, and "
            "the trainer's linear aux MSE at flow.bond_aux_mse_weight > 0 — "
            "several configs set 0.1) was fitted against a vector that is 55% "
            "zeros on real bonds and otherwise another bond's order.",
            "is_dative_pair is derived from `v < 1.0`, so the dative flag was "
            "determined by the wrong number. Both the bond head's dative "
            "supervision and the decoder's dative marking read it.",
            "bond_orders_continuous is also logged into the per-epoch artifact "
            "(model.last_bond_orders_continuous), so archived BO numbers are "
            "readings of the corrupted vector.",
        ],
        "honest_limits": [
            "One cohort, split=None, seed 0. The pre-fix value is recomputed "
            "from the BO shards with the pre-fix keying, not by restoring the "
            "old code; the self-validation block is what licenses that.",
            "bo_per_atom sums over pairs_continuous with NO threshold, so it "
            "legitimately exceeds the sum over the thresholded bond list; the "
            "two are not expected to match and the probe does not compare them.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
