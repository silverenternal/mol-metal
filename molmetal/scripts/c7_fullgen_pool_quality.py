#!/usr/bin/env python3
"""Per-epoch quality of a free-generation pool: fragments, rings, validity.

WHY THIS EXISTS
---------------
``c7_scale_matched_pools.py`` answers the C7 gate (SA <= 4.5, novelty) but says
nothing about whether the molecules got *worse* while the SA rate stayed flat.
The 18-epoch full-corpus run emits 704 molecules per epoch, and the epoch
column is preserved in ``generated_smiles.csv``, so the trajectory is already
on disk — it just was never tabulated.

WHAT IT MEASURES, PER EPOCH
---------------------------
  * ``n_rows`` / ``n_parse`` / ``validity`` — RDKit-parsed fraction,
  * ``single_fragment_rate`` — share of rows with ``n_frags_raw == 1``
    (a molecule, not a bag of ions),
  * ``mean_largest_frag_frac`` — the trainer's own connectivity summary,
  * ``ring_size_distribution`` — atom rings by size, plus the 3-ring and
    6-ring shares of all rings,
  * ``n_unique_smiles`` / ``mean_heavy_atoms``.

Read-only. No model, no GPU. Numbers land in a JSON with MEASURED tags.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402


def _epoch_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    n_rows = len(rows)
    n_parse = 0
    n_single = 0
    n_frag_present = 0
    n_frac_present = 0
    frac_sum = 0.0
    heavy_sum = 0
    unique = set()
    ring_sizes: Counter = Counter()
    for row in rows:
        text = row.get("smiles", "")
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text else None
        if mol is None:
            continue
        n_parse += 1
        unique.add(Chem.MolToSmiles(mol))
        heavy_sum += mol.GetNumHeavyAtoms()
        frag = row.get("n_frags_raw", "")
        try:
            if int(float(frag)) == 1:
                n_single += 1
            n_frag_present += 1
        except (TypeError, ValueError):
            pass
        frac = row.get("largest_frag_frac", "")
        try:
            frac_sum += float(frac)
            n_frac_present += 1
        except (TypeError, ValueError):
            pass
        for ring in mol.GetRingInfo().AtomRings():
            ring_sizes[len(ring)] += 1
    total_rings = sum(ring_sizes.values())
    return {
        "n_rows": n_rows,
        "n_parse": n_parse,
        "validity": round(n_parse / n_rows, 4) if n_rows else None,
        # A pool written before the trainer emitted n_frags_raw has no
        # connectivity column at all; report None, not a flattering 0%.
        "single_fragment_rate": (
            round(n_single / n_frag_present, 4) if n_frag_present else None
        ),
        "mean_largest_frag_frac": round(frac_sum / n_frac_present, 4) if n_frac_present else None,
        "n_unique_smiles": len(unique),
        "mean_heavy_atoms": round(heavy_sum / n_parse, 3) if n_parse else None,
        "n_rings": total_rings,
        "ring_size_distribution": {str(k): v for k, v in sorted(ring_sizes.items())},
        "three_ring_share": round(ring_sizes.get(3, 0) / total_rings, 4) if total_rings else None,
        "six_ring_share": round(ring_sizes.get(6, 0) / total_rings, 4) if total_rings else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "c7_fullgen_pool_quality.json",
    )
    args = parser.parse_args(argv)

    path = args.run / "generated_smiles.csv"
    try:
        pool_label = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        # A pool outside the repo (a test fixture, an archived run).
        pool_label = str(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_epoch: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", "")))
        except (TypeError, ValueError):
            continue
        by_epoch.setdefault(epoch, []).append(row)

    per_epoch = {str(e): _epoch_summary(by_epoch[e]) for e in sorted(by_epoch)}
    overall = _epoch_summary(rows)
    order = sorted(per_epoch, key=int)
    first = per_epoch[order[0]] if order else {}
    last = per_epoch[order[-1]] if order else {}
    trend = None
    if first and last:
        trend = {
            "epoch_first": int(order[0]),
            "epoch_last": int(order[-1]),
            "validity": [first["validity"], last["validity"]],
            "single_fragment_rate": [first["single_fragment_rate"], last["single_fragment_rate"]],
            "three_ring_share": [first["three_ring_share"], last["three_ring_share"]],
        }

    payload = {
        "artifact": "c7_fullgen_pool_quality",
        "tag": "MEASURED",
        "generated_utc": None,
        "provenance": {
            "script": "molmetal/scripts/c7_fullgen_pool_quality.py",
            "git_commit": _git_commit(),
            "pool": pool_label,
            "n_rows": len(rows),
            "epochs": len(per_epoch),
        },
        "overall": overall,
        "per_epoch": per_epoch,
        "first_vs_last": trend,
        "reading": (
            "Single-fragment rate is the share of parsed molecules that came out "
            "as one connected component; three_ring_share is the fraction of all "
            "atom rings that are 3-membered. Both are descriptive: they say what "
            "the pool looks like, not whether it is useful."
        ),
    }
    from datetime import datetime, timezone

    payload["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[c7q] rows={len(rows)} epochs={len(per_epoch)} out={args.out}")
    for e in sorted(per_epoch, key=int):
        row = per_epoch[e]
        print(
            f"[c7q] ep{e:>2} valid={row['validity']} single={row['single_fragment_rate']} "
            f"r3={row['three_ring_share']} r6={row['six_ring_share']} "
            f"heavy={row['mean_heavy_atoms']} uniq={row['n_unique_smiles']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
