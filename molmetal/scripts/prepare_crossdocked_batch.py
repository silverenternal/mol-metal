"""Prepare the first manifest receptors strictly and report every failure.

No docking, relaxed residue matching, atom deletion or source edits occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from molmetal.scripts.prepare_crossdocked_receptor import ROOT, manifest_pair, prepare_receptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "molmetal/data/crossdocked100_manifest.csv")
    parser.add_argument("--n-pockets", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "molmetal/reports/crossdocked_first10_preparation")
    parser.add_argument("--resolve-conformers", action="store_true",
                        help="After strict attempt, record altloc selection/equivalent O/OXT name normalization")
    args = parser.parse_args()
    if args.n_pockets < 1:
        parser.error("--n-pockets must be positive")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:args.n_pockets]
    if len(rows) != args.n_pockets:
        parser.error("manifest contains fewer rows than requested")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "scope": "strict preparation of the first manifest entries; not a physical pilot",
        "requested_pockets": args.n_pockets,
        "traceable_conformer_resolution": args.resolve_conformers,
        "pockets": [],
    }
    for row in rows:
        pocket_id = row["pocket_id"]
        started = time.monotonic()
        record = {"pocket_id": pocket_id, "passed": False}
        try:
            receptor, ligand = manifest_pair(args.manifest, pocket_id)
            source_hash = hashlib.sha256(receptor.read_bytes()).hexdigest()
            record["sources"] = {"receptor": str(receptor), "ligand": str(ligand),
                                 "receptor_sha256": source_hash,
                                 "ligand_sha256": hashlib.sha256(ligand.read_bytes()).hexdigest()}
            if args.resolve_conformers:
                from molmetal.scripts.receptor_preparation_for_evaluation import prepare_receptor_for_evaluation
                record["preparation"] = prepare_receptor_for_evaluation(receptor, args.out_dir / pocket_id)
            else:
                record["preparation"] = prepare_receptor(receptor, args.out_dir / pocket_id)
            record["source_unchanged"] = hashlib.sha256(receptor.read_bytes()).hexdigest() == source_hash
            record["passed"] = record["preparation"]["passed"] and record["source_unchanged"]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.monotonic() - started
        report["pockets"].append(record)
        report["succeeded"] = sum(item["passed"] for item in report["pockets"])
        report["failed"] = len(report["pockets"]) - report["succeeded"]
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{pocket_id}: {'prepared' if record['passed'] else 'FAILED'}", flush=True)
    print(json.dumps({"succeeded": report["succeeded"], "failed": report["failed"],
                      "report": str(args.out_dir / 'report.json')}))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
