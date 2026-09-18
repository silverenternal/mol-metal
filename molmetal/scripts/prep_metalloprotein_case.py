"""Metalloprotein case-study data prep — audit + extract CrossDocked pairs.

Iterates the CrossDocked2020 split and selects the pairs whose receptor
PDB matches the catalogue in :mod:`molmetal.data.metalloprotein_targets`.
The output is one JSON per (family, split) plus a combined summary
file.  This script is the metalloprotein analogue of
:mod:`molmetal.scripts.prep_mmp_case`.

Usage
-----
::

    source .venv/bin/activate
    cd /home/hugo/codes/try_triton_on_rocm
    # Audit only (fast; just count matches):
    python -m molmetal.scripts.prep_metalloprotein_case --audit
    # Full extraction (writes JSON per family):
    python -m molmetal.scripts.prep_metalloprotein_case --target CA2
    python -m molmetal.scripts.prep_metalloprotein_case --target all --split train
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.crossdocked import (
    DEFAULT_ARCHIVE,
    DEFAULT_EXTRACT_DIR,
    CrossDockedDataset,
)
from molmetal.data.crossdocked_filter import filter_by_pdb
from molmetal.data.metalloprotein_targets import (
    METALLOPROTEIN_TARGETS,
    MetalloproteinTarget,
    combined_pdb_ids,
    get_target,
)

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "molmetal" / "data" / "metalloprotein_case_study"


# ---------------------------------------------------------------------------
# Entry serialisation
# ---------------------------------------------------------------------------
def _entry_to_record(entry, target_name: str) -> Dict[str, object]:
    affinity = entry.affinity
    affinity_val: Optional[float] = None
    if not (isinstance(affinity, float) and math.isnan(affinity)):
        affinity_val = float(affinity)
    return {
        "receptor_pdb": entry.receptor_pdb,
        "target": target_name,
        "pocket_pdb": entry.pocket_pdb_path,
        "ligand_sdf": entry.ligand_sdf_path,
        "pocket_pdb_relpath": entry.pocket_pdb_relpath,
        "ligand_sdf_relpath": entry.ligand_sdf_relpath,
        "affinity": affinity_val,
        "split": entry.split,
    }


# ---------------------------------------------------------------------------
# Audit (no disk writes)
# ---------------------------------------------------------------------------
def audit_coverage(
    archive_path: Path = DEFAULT_ARCHIVE,
    extracted_dir: Path = DEFAULT_EXTRACT_DIR,
    split: str = "train",
) -> Dict[str, object]:
    """Audit CrossDocked coverage of all metalloprotein families.

    Returns a JSON-serialisable dict::

        {
          "total_pairs":     100000,
          "split":           "train",
          "families": {
            "MMP2":  {"n_pdbs_queried": 6, "n_pdbs_matched": 0, "n_pairs": 0, "missing_pdbs": [...]},
            ...
          },
          "all_families_pdbs_matched": <int>,
          "all_families_pairs":       <int>,
        }
    """
    ds = CrossDockedDataset(
        archive_path=archive_path,
        extracted_dir=extracted_dir,
        split=split,
        auto_extract=True,
    )
    flat_pdbs = combined_pdb_ids()
    family_summary: Dict[str, Dict[str, object]] = {}

    for name, target in METALLOPROTEIN_TARGETS.items():
        wanted = list(target.pdb_ids)
        entries, stats = filter_by_pdb(
            ds, wanted, compute_stats=False, max_entries=10_000
        )
        family_summary[name] = {
            "metal": target.metal,
            "n_pdbs_queried": len(wanted),
            "n_pdbs_matched": len(stats.unique_receptor_pdbs),
            "n_pairs": int(stats.matching_pairs),
            "matched_pdbs": sorted(stats.unique_receptor_pdbs),
            "missing_pdbs": sorted(stats.missing_pdbs),
        }
        print(
            f"[audit] {name:8s} ({target.metal:2s})  "
            f"queried={len(wanted)} matched={len(stats.unique_receptor_pdbs)} "
            f"pairs={stats.matching_pairs}"
        )

    return {
        "total_pairs": int(len(ds)),
        "split": split,
        "families": family_summary,
        "all_families_pdbs_matched": sum(
            f["n_pdbs_matched"] for f in family_summary.values()
        ),
        "all_families_pairs": sum(f["n_pairs"] for f in family_summary.values()),
    }


# ---------------------------------------------------------------------------
# Per-family extraction
# ---------------------------------------------------------------------------
def run_for_target_split(
    target: MetalloproteinTarget,
    split: str,
    archive_path: Path,
    extracted_dir: Path,
    output_dir: Path,
    compute_stats: bool = True,
    max_entries: Optional[int] = None,
) -> Dict[str, object]:
    ds = CrossDockedDataset(
        archive_path=archive_path,
        extracted_dir=extracted_dir,
        split=split,
        auto_extract=True,
    )
    entries, stats = filter_by_pdb(
        ds, list(target.pdb_ids), compute_stats=compute_stats, max_entries=max_entries
    )
    records = [_entry_to_record(e, target.name) for e in entries]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{target.name.lower()}_{split}.json"
    payload = {
        "target": target.name,
        "uniprot_id": target.uniprot_id,
        "metal": target.metal,
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
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit + extract CrossDocked2020 pairs by metalloprotein family."
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Audit coverage (no JSON extraction).  Fast.",
    )
    p.add_argument(
        "--target",
        default="all",
        help="Family name (e.g. CA2, ACE, MMP2) or 'all' (default).",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="CrossDocked split to scan (default: train).",
    )
    p.add_argument(
        "--archive",
        default=str(DEFAULT_ARCHIVE),
        help="Path to CrossDocked2020_cascadediff.zip",
    )
    p.add_argument(
        "--extracted-dir",
        default=str(DEFAULT_EXTRACT_DIR),
        help="Where the extracted CrossDocked split + tarball live",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Where to write per-family/split JSON files",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip per-file ligand/pocket atom counting (faster)",
    )
    p.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap on entries per (family, split)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    archive_path = Path(args.archive)
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)

    if args.audit:
        print(
            f"[metallo] AUDIT split={args.split} archive={archive_path}"
        )
        audit = audit_coverage(
            archive_path=archive_path,
            extracted_dir=extracted_dir,
            split=args.split,
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "metalloprotein_coverage_audit.json"
        out.write_text(json.dumps(audit, indent=2))
        print(f"\n[metallo] wrote audit {out}")
        print(
            f"[metallo] total_pairs={audit['total_pairs']} "
            f"families_matched={audit['all_families_pdbs_matched']} "
            f"family_pairs={audit['all_families_pairs']}"
        )
        return 0

    if args.target == "all":
        targets = list(METALLOPROTEIN_TARGETS.values())
    else:
        targets = [get_target(args.target)]

    print(
        f"[metallo] target={args.target} split={args.split} "
        f"archive={archive_path}"
    )
    overall_t0 = time.time()
    summary: List[Dict[str, object]] = []
    for target in targets:
        t0 = time.time()
        try:
            result = run_for_target_split(
                target=target,
                split=args.split,
                archive_path=archive_path,
                extracted_dir=extracted_dir,
                output_dir=output_dir,
                compute_stats=not args.no_stats,
                max_entries=args.max_entries,
            )
        except FileNotFoundError as e:
            print(f"[metallo] ERROR: {e}")
            return 2
        elapsed = time.time() - t0
        result["elapsed_seconds"] = elapsed
        summary.append(result)
        print(
            f"[metallo]   {target.name:8s} ({target.metal:2s}) "
            f"n_entries={result['n_entries']} "
            f"missing={result['missing_pdbs']} "
            f"elapsed={elapsed:.1f}s"
        )
        print(f"[metallo]   wrote {result['out_path']}")

    print(
        f"\n[metallo] DONE total={time.time()-overall_t0:.1f}s "
        f"families={len(summary)}"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "metalloprotein_case_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[metallo] wrote summary {summary_path}")
    return 0


__all__ = ["audit_coverage", "run_for_target_split"]


if __name__ == "__main__":
    raise SystemExit(main())