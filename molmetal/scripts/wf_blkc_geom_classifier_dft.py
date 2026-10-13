#!/usr/bin/env python
"""Apply wf_blkc_geom_classifier.measure_clouds to REAL tmQM DFT frames.

This is the like-for-like baseline. The round-1 instrument
(``wf_blk_c_obj_metal_geometry.py::measure_dft_frame``) already does
this on REAL_DFT, but it was inline: a method on the
metal-geometry script. Here we call the new module's
``measure_clouds`` directly so the two paths (round 1's
``measure_dft_frame`` and this script's ``measure_clouds``) can be
compared byte-for-byte and any drift is caught.

The output JSON has the same shape as
``metrics/by_metric/cfm_metal_geometry.json::cohorts.REAL_DFT`` but is
written with the new module's own measurement path, so a future
reader can tell the two apart by which script produced it.

Usage
-----
    /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/wf_blkc_geom_classifier_dft.py --real-limit 3000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO / "molmetal/scripts"))
from wf_blkc_geom_classifier import (  # noqa: E402
    PLANARITY_CUT,
    TRAIN_METALS,
    measure_clouds,
    summarise_clouds,
)

#: Six train metals. Same list as round 1; pinned here so the cohort
#: filter cannot drift.
TRAIN_METAL_SYMS = list(TRAIN_METALS.keys())

TMQM_CSV = REPO / ".data/molmetal/tmQM/tmqm_parsed.csv"
#: All three shards. Round 1 only read X1; the loader fix in tmqm_clouds.py
#: (comment at line 116-121) means X2 and X3 are also available, but the
#: real-DFT cohort of 3000 is well within X1's 34970-frame capacity, so
#: we read X1 for a like-for-like baseline.
TMQM_XYZ = REPO / ".data/molmetal/tmQM/tmQM_X1.xyz.gz"


def _csd_from_comment(comment: str) -> Optional[str]:
    marker = "CSD_code = "
    if marker in comment:
        return comment.split(marker, 1)[1].split("|")[0].strip()
    return None


def _read_xyz_csd_codes(path: Path) -> set:
    """Every CSD code present in the xyz file. One streaming pass."""
    codes = set()
    with gzip.open(path, "rt") as stream:
        while True:
            header = stream.readline()
            if not header:
                return codes
            if not header.strip().isdigit():
                continue
            n_atoms = int(header.strip())
            comment = stream.readline()
            code = _csd_from_comment(comment)
            if code is not None:
                codes.add(code)
            for _ in range(n_atoms):
                stream.readline()


def _read_metal_csd_codes(
    csv_path: Path,
    metals: Sequence[str],
    limit: int,
    seed: int,
    restrict_to: Optional[set] = None,
):
    """CSD codes + metal for the six train metals, capped by a fixed seed."""
    keep = set(metals)
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("metal") not in keep or not row.get("smiles"):
                continue
            if restrict_to is not None and row["csd_code"] not in restrict_to:
                continue
            rows.append((row["csd_code"], row["metal"], row.get("coord_number")))
    rows.sort()
    if limit and len(rows) > limit:
        rnd = random.Random(seed)
        rows = sorted(rnd.sample(rows, limit))
    return rows


def _parse_xyz_frames(path: Path, want: Dict[str, str]):
    """Yield (csd_code, metal_symbol, [(sym, x, y, z), ...]) for wanted CSD codes."""
    with gzip.open(path, "rt") as stream:
        while True:
            header = stream.readline()
            if not header:
                return
            n_atoms = header.strip()
            if not n_atoms.isdigit():
                continue
            comment = stream.readline()
            code = _csd_from_comment(comment)
            if code is None or code not in want:
                for _ in range(int(n_atoms)):
                    stream.readline()
                continue
            atoms = []
            for _ in range(int(n_atoms)):
                line = stream.readline()
                parts = line.split()
                if len(parts) < 4:
                    atoms = []
                    break
                atoms.append(
                    (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
                )
            if atoms:
                yield code, want[code], atoms


def _atoms_to_clouds(atoms):
    """Convert a list of (sym, x, y, z) tuples to (positions, atomic_numbers)."""
    syms = [a[0] for a in atoms]
    pos = np.asarray([(a[1], a[2], a[3]) for a in atoms], dtype=float)
    zs = np.asarray(
        [_sym_to_z(s) for s in syms],
        dtype=int,
    )
    return pos, zs


def _sym_to_z(sym: str) -> int:
    """Symbol -> Z. Lazy import to keep this file chemistry-free at top."""
    if sym in TRAIN_METALS:
        return TRAIN_METALS[sym]
    common = {
        "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
        "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    }
    z = common.get(sym)
    if z is None:
        raise ValueError(f"unmapped element {sym!r} in tmQM frame")
    return z


def select_dft_population(limit: int, seed: int):
    """CSD codes for the DFT cohort: six train metals WITH coordinates."""
    available = _read_xyz_csd_codes(TMQM_XYZ)
    return _read_metal_csd_codes(
        TMQM_CSV, TRAIN_METAL_SYMS, limit, seed, restrict_to=available
    )


def measure_dft_frames(limit: int, seed: int) -> List[Dict[str, object]]:
    """REAL_DFT, measured through the new classifier."""
    rows = select_dft_population(limit, seed)
    want = {csd: metal for csd, metal, _cn in rows}
    records = []
    n_metal_mismatch = 0
    n_unmapped = 0
    for csd, metal, atoms in _parse_xyz_frames(TMQM_XYZ, want):
        try:
            pos, zs = _atoms_to_clouds(atoms)
        except ValueError:
            n_unmapped += 1
            continue
        # The metal listed in the csv is the curated label; we honour it
        # when locating the centre. If the curated metal is not actually
        # present in the geometry, we fall back to the first train metal
        # in the cloud -- the same behaviour round 1 took.
        if metal not in TRAIN_METALS or metal not in {a[0] for a in atoms}:
            n_metal_mismatch += 1
            rec = measure_clouds(pos, zs)  # no metal preference
        else:
            rec = measure_clouds(pos, zs, metal=metal)
        rec["csd_code"] = csd
        records.append(rec)
    print(
        f"[dft] {len(records)} measured; {n_metal_mismatch} metal mismatches; "
        f"{n_unmapped} unmapped-element skips"
    )
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-limit", type=int, default=3000)
    ap.add_argument("--sample-seed", type=int, default=20261012)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-records", type=Path, default=None)
    args = ap.parse_args(argv)

    records = measure_dft_frames(args.real_limit, args.sample_seed)
    summary = summarise_clouds(records, "REAL_DFT (new module, no ETKDG, no RDKit)")
    summary["n_requested"] = args.real_limit
    payload = {
        "metric": "cfm_geom_classifier_dft",
        "purpose": "REAL_DFT baseline for the new ETKDG-free classifier",
        "created": "2026-10-12",
        "protocol": {
            "module": "molmetal.scripts.wf_blkc_geom_classifier.measure_clouds",
            "donor_cutoff_A": 2.6,
            "trans_angle_cut_deg": 150.0,
            "planarity_cut": PLANARITY_CUT,
            "min_plausible_ml_bond_A": 1.5,
            "train_metals": TRAIN_METAL_SYMS,
            "sample_seed": args.sample_seed,
            "real_limit_requested": args.real_limit,
        },
        "cohort": summary,
        "interpretation_rules": [
            "This cohort IS the like-for-like baseline for a future GENERATED measurement that uses the same module on sampled clouds. It must be re-derived whenever the module's tolerance values change.",
            "Numbers should match cfm_metal_geometry.json::cohorts.REAL_DFT under the round-1 instrument measure_dft_frame; any drift indicates a tolerance change.",
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"[write] {args.out}")
    else:
        print(text)

    if args.dump_records:
        args.dump_records.parent.mkdir(parents=True, exist_ok=True)
        args.dump_records.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"[write] {args.dump_records}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
