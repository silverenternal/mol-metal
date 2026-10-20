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
    bond_sum = 0
    n_bond_eligible = 0
    unique = set()
    ring_sizes: Counter = Counter()
    for row in rows:
        text = row.get("smiles", "")
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text else None
        if mol is None:
            continue
        n_parse += 1
        unique.add(Chem.MolToSmiles(mol))
        heavy = mol.GetNumHeavyAtoms()
        heavy_sum += heavy
        if heavy >= 2:
            bond_sum += mol.GetNumBonds()
            n_bond_eligible += 1
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
        # Bonds per heavy atom separates "how many bonds" from "which bonds".
        # A pool can match the real bond density and still put every bond in a
        # 3-membered ring.
        "bonds_per_heavy_atom": (
            round(bond_sum / heavy_sum, 4) if heavy_sum else None
        ),
        "n_rings": total_rings,
        "ring_size_distribution": {str(k): v for k, v in sorted(ring_sizes.items())},
        "three_ring_share": round(ring_sizes.get(3, 0) / total_rings, 4) if total_rings else None,
        "six_ring_share": round(ring_sizes.get(6, 0) / total_rings, 4) if total_rings else None,
    }


def _bond_head_trajectory(run_dir: Path) -> Dict[str, object] | None:
    """Teacher-forced bond metrics per epoch, from the run's own metrics file.

    These are computed on real held-out geometries (the head is given the true
    atoms and asked for the bonds), which is a different question from whether
    the sampler can connect a cloud. Putting them side by side is the point:
    if the head improves while the generated pools fragment more, the failure
    is in the sampling path, not in the head.
    """
    path = run_dir / "per_epoch_metrics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    rows = payload.get("per_epoch_metrics") or []
    keep = (
        "epoch",
        "bond_precision",
        "bond_recall",
        "bond_f1",
        "bond_pr_auc",
        "bond_n_scored",
    )
    out = [{k: row.get(k) for k in keep if k in row} for row in rows]
    return out or None


def _reference_stats(path: Path) -> Dict[str, object] | None:
    """The same numbers for real chemistry, so the pool has something to be wrong against.

    Without this, ``three_ring_share = 0.67`` reads as "lots of small rings",
    which sounds like a stylistic quirk. Against the corpus it is a different
    statement: the corpus has essentially no 3-membered rings at all.
    """
    if not path.exists():
        return None
    import pandas as pd

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_csv(path, low_memory=False)
    if "split" in frame.columns:
        frame = frame[frame["split"] == "train"]
    if "smiles" not in frame.columns:
        return None
    texts = frame["smiles"].dropna().astype(str)
    heavy_sum = 0
    bond_sum = 0
    n = 0
    ring_sizes: Counter = Counter()
    for text in texts:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        heavy = mol.GetNumHeavyAtoms()
        if heavy < 2:
            continue
        n += 1
        heavy_sum += heavy
        bond_sum += mol.GetNumBonds()
        for ring in mol.GetRingInfo().AtomRings():
            ring_sizes[len(ring)] += 1
    if not n:
        return None
    total_rings = sum(ring_sizes.values())
    return {
        "source": str(path),
        "n": n,
        "mean_heavy_atoms": round(heavy_sum / n, 3),
        "bonds_per_heavy_atom": round(bond_sum / heavy_sum, 4),
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
        "--reference",
        type=Path,
        default=PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
        help="real corpus to compare the pool against; skipped when absent",
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
        "bond_head_trajectory": _bond_head_trajectory(args.run),
        "reference": _reference_stats(args.reference),
        "reading": (
            "Single-fragment rate is the share of parsed molecules that came out "
            "as one connected component; three_ring_share is the fraction of all "
            "atom rings that are 3-membered. Both are descriptive: they say what "
            "the pool looks like, not whether it is useful. bond_head_trajectory "
            "is teacher-forced (true atoms, predicted bonds) and is NOT comparable "
            "to the pool's connectivity: it measures the head, the pool measures "
            "the sampler."
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
