# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T69 — ``coord_geom_class`` was classified from another atom's degree.

THE DEFECT
----------
``_build_cloud_from_kept`` reads

    metal_cn = len(bo_info.per_atom_nbr_zsets.get(metal_idx_in_kept, []))

``per_atom_nbr_zsets`` is keyed by the RAW tmQM index; ``metal_idx_in_kept`` is
an index in the kept (hydrogen-dropped) frame.  So the geometry class was
classified from whichever atom happened to sit at that raw index — usually a
carbon — and the metal's real coordination number never entered the lookup.

The field has NO consumer anywhere in the repo today (``AtomCloud.
coord_geom_class`` is written and never read, and ``_to_device_cloud`` drops
it), so this fix changes no trained number.  It is recorded because the field
is documented as a MOD5 conditioning slot: wiring it up against the current
values would feed the model a carbon's degree under the name "metal geometry".

Run:
    .venv/bin/python molmetal/scripts/wfa113_t69_coord_geom_frame_probe.py \
        --out metrics/by_metric/wfa113_t69_coord_geom_frame.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[2]
TMQM = _REPO / ".data" / "molmetal" / "tmQM"
METALS = ["Pt", "Ru", "Ir", "Rh", "Os", "Re"]
XYZ = [TMQM / "tmQM_X1.xyz.gz", TMQM / "tmQM_X2.xyz.gz", TMQM / "tmQM_X3.xyz.gz"]
BO = [TMQM / "tmQM_X1.BO.gz", TMQM / "tmQM_X2.BO.gz", TMQM / "tmQM_X3.BO.gz"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(_REPO))
    from molmetal.data.coord_geometry import classify_coord_geometry
    from molmetal.data.tmqm_clouds import (
        TMQM_METAL_Z,
        _stratified_code_draw,
        iter_bo_bonds,
        iter_tmqm_blocks,
        load_tmqm_clouds,
        load_tmqm_metadata,
    )

    meta = load_tmqm_metadata(TMQM / "tmqm_parsed.csv")
    codes_all = [
        c for c, r in meta.items() if str(r.get("metal", "")).strip() in set(METALS)
    ]
    drawn, quota, _pool = _stratified_code_draw(
        meta, codes_all, int(args.n_clouds), int(args.seed)
    )
    order = {c: i for i, c in enumerate(drawn)}
    by_metal: Dict[str, List[str]] = {}
    for c in drawn:
        by_metal.setdefault(str(meta[c].get("metal", "")).strip(), []).append(c)
    taken: List[str] = []
    for m in sorted(by_metal):
        taken.extend(sorted(by_metal[m], key=lambda c: order[c])[: int(quota.get(m, 0))])
    codes_in_order = sorted(taken, key=lambda c: order[c])

    keep: Dict[str, List[int]] = {}
    for code, symbols, _coords in iter_tmqm_blocks(XYZ):
        if code in order:
            keep[code] = [i for i, s in enumerate(symbols) if s != "H"]
    nbrs = {code: z for code, _pairs, z in iter_bo_bonds(BO) if code in order}

    clouds = load_tmqm_clouds(
        str(TMQM), metals=METALS, n_clouds=int(args.n_clouds), seed=int(args.seed), split=None
    )
    assert len(clouds) == len(codes_in_order)

    before = collections.Counter()
    after = collections.Counter()
    cn_before = collections.Counter()
    cn_after = collections.Counter()
    same_cn = 0
    for cloud, code in zip(clouds, codes_in_order):
        row = meta[code]
        metal_z = TMQM_METAL_Z.get(str(row.get("metal", "")).strip())
        if metal_z is None:
            continue
        kept = keep[code]
        old_to_new = {old: new for new, old in enumerate(kept)}
        midx_new = next(
            (new for new, z in enumerate(cloud.atomic_numbers.tolist()) if z == metal_z),
            None,
        )
        if midx_new is None:
            continue
        raw_idx = kept[midx_new]
        cn_bug = len(nbrs.get(code, {}).get(midx_new, []))          # kept index, raw dict
        cn_ok = len(nbrs.get(code, {}).get(raw_idx, []))            # raw index, raw dict
        cn_before[cn_bug] += 1
        cn_after[cn_ok] += 1
        same_cn += int(cn_bug == cn_ok)
        before[str(classify_coord_geometry(cn_bug, int(metal_z)))] += 1
        after[str(classify_coord_geometry(cn_ok, int(metal_z)))] += 1
        # The shipped field must agree with the fixed computation.
        assert str(cloud.coord_geom_class) == str(
            classify_coord_geometry(cn_ok, int(metal_z))
        ), (code, cloud.coord_geom_class, cn_ok)

    out = {
        "metric": "wfa113_t69_coord_geom_frame",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "probe": "molmetal/scripts/wfa113_t69_coord_geom_frame_probe.py",
        "cohort": f"{len(clouds)} clouds, split=None, seed={args.seed}",
        "defect": {
            "code": "molmetal/data/tmqm_clouds.py::_build_cloud_from_kept (coord_geom_class)",
            "mechanism": (
                "metal_cn was read from per_atom_nbr_zsets with "
                "metal_idx_in_kept (kept-atom frame) while the dict is keyed by "
                "the raw tmQM index, so the degree of an arbitrary atom was "
                "classified as the metal's coordination number."
            ),
            "fix": "Look up keep_idx[metal_idx_in_kept] instead.",
        },
        "n_clouds": len(clouds),
        "metal_cn_matches_between_frames": same_cn,
        "metal_cn_before": {str(k): v for k, v in sorted(cn_before.items())},
        "metal_cn_after": {str(k): v for k, v in sorted(cn_after.items())},
        "class_before": dict(before.most_common()),
        "class_after": dict(after.most_common()),
        "consumer": "NONE — AtomCloud.coord_geom_class is written and never read; "
                    "_to_device_cloud does not carry it.",
        "impact": (
            "No trained number changes today.  The fix matters because the field "
            "is documented as a MOD5 conditioning slot: wiring it up against the "
            "pre-fix values would feed the model a carbon's degree under the name "
            "'metal coordination geometry'."
        ),
        "honest_limits": [
            "metal_cn counts every neighbour listed in the Wiberg BO block (up to "
            "12), not the thresholded bond list, so it can exceed the bond graph's "
            "metal degree.",
            "The class is None whenever (Z, CN) is unmapped; tmQM genuinely has "
            "7- and 8-coordinate complexes, so None is not an error.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
