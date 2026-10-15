#!/usr/bin/env python3
"""Merge per-(metal, seed) ``dmpnn_within_metal_auc`` slices into one artifact.

``dmpnn_within_metal_auc.py`` runs metals x split-seeds sequentially, so a
5-metal x 5-seed paper-funnel run is ~6 h on one core.  The run is embarrassingly
parallel in (metal, split_seed), so the launch splits it into one process per
pair; each writes a slice report with a single metal and a single split seed.

This script unions those slices back into the canonical artifact:

  * top-level metadata (cohort, arch, fold protocol, ...) is taken from the
    slices — they all come from the same config, and the script asserts that the
    metadata blocks agree rather than silently trusting the first file;
  * ``per_metal[M]`` is rebuilt by concatenating every slice's ``per_split``
    rows and recomputing the aggregates exactly as ``_run_metal`` does
    (mean over per-fold means, population std, paired XGB delta);
  * ``split_seeds`` becomes the sorted union across slices.

MISSING != dropped (AGENTS.md §5.2): a metal with no slice is reported as
``status: SKIPPED`` with a reason, never omitted from the artifact.

Usage
-----
::

    .venv/bin/python molmetal/scripts/merge_within_metal_slices.py \
        --slices-dir molmetal/reports/wm_pf_slices \
        --out-metric metrics/by_metric/dmpnn_within_metal_auc_paper_funnel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Top-level keys that must be identical across slices (same config => same run).
_METADATA_KEYS = (
    "metric", "MEASURED_vs_PROJECTED", "unit", "scope", "source", "data_source",
    "cohort", "cohort_spec", "fold_protocol", "dmpnn_arch", "xgb_reference",
    "device", "krasnov_within_metal_auc", "min_unique_for_gate",
)


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _merge_metal(slices: List[Dict[str, object]], metal: str) -> Dict[str, object]:
    """Rebuild one metal's record from its slices (mirrors ``_run_metal``)."""
    first = slices[0]["per_metal"][metal]
    per_split: List[Dict[str, object]] = []
    all_dmpnn: List[float] = []
    all_xgb: List[float] = []
    seeds: List[int] = []
    for s in slices:
        rec = s["per_metal"][metal]
        for row in rec.get("per_split", []):
            per_split.append(row)
            seeds.append(int(row["split_seed"]))
            for f in row.get("per_fold", []):
                if f.get("status") == "MEASURED":
                    all_dmpnn.append(float(f["dmpnn_auc"]))
                    all_xgb.append(float(f["xgb_auc"]))
    per_split.sort(key=lambda r: int(r["split_seed"]))

    entry = dict(first)
    entry.pop("wall_sec", None)
    if not all_dmpnn:
        entry["status"] = "SKIPPED"
        # A metal skipped before any fold ran (n too small / single-class) already
        # carries a more specific reason; keep it rather than flattening to the
        # generic "no measurable folds" (AGENTS.md §5.2 — the reason is the result).
        entry["reason"] = first.get("reason") or "no measurable folds"
        entry["per_split"] = per_split
        entry["split_seeds"] = sorted(set(seeds))
        return entry

    dmpnn_mean = float(np.mean(all_dmpnn))
    xgb_mean = float(np.mean(all_xgb))
    paired42 = next(
        (
            {
                "dmpnn_auc": row["dmpnn_auc"],
                "xgb_auc": row["xgb_auc"],
                "delta_dmpnn_minus_xgb": row["delta_dmpnn_minus_xgb"],
            }
            for row in per_split
            if int(row["split_seed"]) == 42
        ),
        None,
    )
    entry.update(
        {
            "status": "MEASURED",
            "split_seeds": sorted(set(seeds)),
            "n_folds_measured": int(len(all_dmpnn)),
            "dmpnn_auc": dmpnn_mean,
            "dmpnn_auc_across_folds_std": float(np.std(all_dmpnn)),
            "xgb_auc_same_folds": xgb_mean,
            "delta_dmpnn_minus_xgb": float(dmpnn_mean - xgb_mean),
            "dmpnn_beats_xgb": bool(dmpnn_mean > xgb_mean),
            "paired_split_seed_42": paired42,
            "per_split": per_split,
        }
    )
    if entry.get("krasnov_reported_auc") is not None:
        entry["dmpnn_beats_krasnov"] = bool(
            dmpnn_mean > float(entry["krasnov_reported_auc"])
        )
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slices-dir", default="molmetal/reports/wm_pf_slices")
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/dmpnn_within_metal_auc_paper_funnel.json",
    )
    args = p.parse_args(argv)

    slices_dir = PROJECT_ROOT / args.slices_dir
    files = sorted(slices_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no slice JSONs under {slices_dir}")

    slices = [_load(f) for f in files]

    # Metadata must agree across slices — same config, so any disagreement is a
    # real bug (a stale slice from a different run), not a formatting nit.
    ref = slices[0]
    for key in _METADATA_KEYS:
        for s in slices[1:]:
            if s.get(key) != ref.get(key):
                raise SystemExit(
                    f"slice metadata disagreement on '{key}': "
                    f"{s.get(key)!r} != {ref.get(key)!r} "
                    f"(a slice is from a different run)"
                )

    metals: List[str] = []
    for s in slices:
        for m in s.get("per_metal", {}):
            if m not in metals:
                metals.append(m)
    metals.sort()

    per_metal: Dict[str, object] = {}
    for m in metals:
        ms = [s for s in slices if m in s.get("per_metal", {})]
        per_metal[m] = _merge_metal(ms, m)

    out = {k: ref.get(k) for k in ref}
    out.pop("per_metal", None)
    out["per_metal"] = per_metal
    out["split_seeds"] = sorted(
        {int(sd) for s in slices for sd in s.get("split_seeds", [])}
    )
    # A single slice's honest_negative block only knows that one metal.  Take the
    # config-independent prose from the reference slice but recompute the two
    # per-metal lists over the FULL merged table (AGENTS.md §5.2).
    hn = dict(ref.get("honest_negative", {}))
    hn["unresolvable_metals"] = sorted(
        m for m, v in per_metal.items() if not v.get("resolvable", False)
    )
    hn["dmpnn_below_xgb"] = sorted(
        m
        for m, v in per_metal.items()
        if v.get("status") == "MEASURED" and not v.get("dmpnn_beats_xgb", True)
    )
    out["honest_negative"] = hn
    out["merged_from_slices"] = {
        "dir": args.slices_dir,
        "n_slice_files": len(files),
        "slice_files": [f.name for f in files],
        "note": (
            "One slice per (metal, split_seed); unioned by "
            "merge_within_metal_slices.py.  Aggregates recomputed exactly as "
            "_run_metal does."
        ),
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = PROJECT_ROOT / args.out_metric
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    for m in metals:
        e = per_metal[m]
        if e.get("status") == "MEASURED":
            print(
                f"{m:>4} n={e['n_unique']:5d} dmpnn={e['dmpnn_auc']:.4f}"
                f"±{e['dmpnn_auc_across_folds_std']:.4f} xgb={e['xgb_auc_same_folds']:.4f}"
                f" seeds={e['split_seeds']} folds={e['n_folds_measured']}"
            )
        else:
            print(f"{m:>4} SKIPPED: {e.get('reason')}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
