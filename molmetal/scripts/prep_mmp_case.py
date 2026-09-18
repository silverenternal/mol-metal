"""MMP2/MMP9 case-study data prep — extract CrossDocked pairs by receptor PDB.

Iterates the CrossDocked2020 split (100 k train, 100 test) and selects
the pairs whose receptor (pocket) PDB matches the curated MMP2/MMP9
target list in :mod:`molmetal.data.mmp_targets`.  For each match we
write a JSON record of the form::

    {
      "receptor_pdb":       "1GKC",
      "target":             "MMP9",
      "pocket_pdb":         "/abs/path/to/..._pocket10.pdb",
      "ligand_sdf":         "/abs/path/to/...sdf",
      "pocket_pdb_relpath": "MMP9_HUMAN_36_109_0/...pdb",
      "ligand_sdf_relpath": "MMP9_HUMAN_36_109_0/...sdf",
      "affinity":           null,
      "split":              "train"
    }

Usage
-----
::

    source .venv/bin/activate
    cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.prep_mmp_case --target MMP2
    python -m molmetal.scripts.prep_mmp_case --target BOTH --splits train test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.crossdocked import (
    DEFAULT_ARCHIVE,
    DEFAULT_EXTRACT_DIR,
    CrossDockedDataset,
)
from molmetal.data.crossdocked_filter import filter_by_pdb
from molmetal.data.mmp_targets import (
    MMP2_TARGET,
    MMP9_TARGET,
    MMPTarget,
    all_targets,
    combined_pdb_ids,
    get_target,
)

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "molmetal" / "data" / "mmp_case_study"


# ---------------------------------------------------------------------------
# Entry serialisation
# ---------------------------------------------------------------------------
def _entry_to_record(entry, target: str) -> Dict[str, object]:
    """Convert a :class:`CrossDockedEntry` to a JSON-safe dict."""
    affinity = entry.affinity
    affinity_val: Optional[float] = None
    try:
        import math as _math

        if not (isinstance(affinity, float) and _math.isnan(affinity)):
            affinity_val = float(affinity)
    except Exception:
        affinity_val = None
    return {
        "receptor_pdb": entry.receptor_pdb,
        "target": target,
        "pocket_pdb": entry.pocket_pdb_path,
        "ligand_sdf": entry.ligand_sdf_path,
        "pocket_pdb_relpath": entry.pocket_pdb_relpath,
        "ligand_sdf_relpath": entry.ligand_sdf_relpath,
        "affinity": affinity_val,
        "split": entry.split,
    }


# ---------------------------------------------------------------------------
# Run prep for one target + one split
# ---------------------------------------------------------------------------
def run_for_target_split(
    target: MMPTarget,
    split: str,
    archive_path: Path,
    extracted_dir: Path,
    output_dir: Path,
    compute_stats: bool = True,
    max_entries: Optional[int] = None,
) -> Dict[str, object]:
    """Filter the dataset, write JSON, and return summary stats."""
    ds = CrossDockedDataset(
        archive_path=archive_path,
        extracted_dir=extracted_dir,
        split=split,
        auto_extract=True,
    )
    entries, stats = filter_by_pdb(
        ds,
        list(target.pdb_ids),
        compute_stats=compute_stats,
        max_entries=max_entries,
    )
    records = [_entry_to_record(e, target.name) for e in entries]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{target.name.lower()}_{split}.json"
    payload = {
        "target": target.name,
        "uniprot_id": target.uniprot_id,
        "full_name": target.full_name,
        "split": split,
        "n_entries": len(records),
        "pdb_ids_queried": list(target.pdb_ids),
        "missing_pdbs": sorted(stats.missing_pdbs),
        "stats": stats.as_dict(),
        "entries": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return {
        "target": target.name,
        "split": split,
        "n_entries": len(records),
        "missing_pdbs": sorted(stats.missing_pdbs),
        "mean_ligand_atoms": stats.mean_ligand_atoms,
        "mean_pocket_atoms": stats.mean_pocket_atoms,
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract MMP2/MMP9 CrossDocked2020 pairs by receptor PDB."
    )
    p.add_argument(
        "--target",
        default="MMP2",
        choices=["MMP2", "MMP9", "BOTH"],
        help="Which MMP target to extract (default: MMP2)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "test"],
        help="Which CrossDocked splits to scan (default: train)",
    )
    p.add_argument(
        "--archive",
        default=str(DEFAULT_ARCHIVE),
        help="Path to CrossDocked2020_cascadediff.zip",
    )
    p.add_argument(
        "--extracted-dir",
        default=str(DEFAULT_EXTRACT_DIR),
        help="Where to extract the split file + tarball",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Where to write per-target/split JSON files",
    )
    p.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap on entries per (target, split) for fast iteration",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip per-file ligand/pocket atom counting (faster)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    archive_path = Path(args.archive)
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)

    if args.target == "BOTH":
        targets = [MMP2_TARGET, MMP9_TARGET]
    else:
        targets = [get_target(args.target)]

    print(
        f"[mmp_case] target={args.target} splits={args.splits} "
        f"archive={archive_path}"
    )
    if not archive_path.exists():
        print(f"[mmp_case] WARNING: archive not found at {archive_path}")

    overall_t0 = time.time()
    summary: List[Dict[str, object]] = []

    for target in targets:
        for split in args.splits:
            t0 = time.time()
            print(
                f"\n[mmp_case] === {target.name} / split={split} ==="
            )
            print(
                f"[mmp_case]   querying {len(target.pdb_ids)} PDB ids: "
                f"{', '.join(target.pdb_ids)}"
            )
            try:
                result = run_for_target_split(
                    target=target,
                    split=split,
                    archive_path=archive_path,
                    extracted_dir=extracted_dir,
                    output_dir=output_dir,
                    compute_stats=not args.no_stats,
                    max_entries=args.max_entries,
                )
            except FileNotFoundError as e:
                print(f"[mmp_case] ERROR: {e}")
                print(
                    "[mmp_case] hint: extract the zip first, or set --archive "
                    "to the location of CrossDocked2020_cascadediff.zip"
                )
                return 2
            except Exception as e:
                print(f"[mmp_case] ERROR ({target.name}/{split}): {e}")
                return 3
            elapsed = time.time() - t0
            result["elapsed_seconds"] = elapsed
            summary.append(result)
            print(
                f"[mmp_case]   n_entries={result['n_entries']} "
                f"missing={result['missing_pdbs']} "
                f"mean_lig_atoms={result['mean_ligand_atoms']:.1f} "
                f"mean_poc_atoms={result['mean_pocket_atoms']:.1f} "
                f"elapsed={elapsed:.1f}s"
            )
            print(f"[mmp_case]   wrote {result['out_path']}")

    overall_elapsed = time.time() - overall_t0
    print(
        f"\n[mmp_case] DONE total={overall_elapsed:.1f}s "
        f"n_summary={len(summary)}"
    )
    # Also drop a combined summary file alongside the records
    if summary:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORTS_DIR / "mmp_case_study_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[mmp_case] wrote summary {summary_path}")
    return 0


__all__ = ["run_for_target_split"]


if __name__ == "__main__":
    raise SystemExit(main())