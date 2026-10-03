#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wf-119 Phase 3 — delete the 12 stale JSON files for the 4 MOCK_DELETED external scorers.

Per TODO-114, the following 12 files on disk under ``metrics/by_metric/``
are placeholders carried forward from Wf-43 Phase 2 when the 4 Phase-4A
external scorers were marked ``MOCK_DELETED``. They were never MEASURED:

  - ``coord_geometry_proxy.json`` + 2 aliases
  - ``phototherapy_activity.json`` + 2 aliases
  - ``ptiv_reduction_potential.json`` + 2 aliases
  - ``metalloprotein_affinity.json`` + 2 aliases

Each base name has three alias variants:
  1. ``<base>``                          (e.g. coord_geometry_proxy)
  2. ``lambda_mcts_<base>``              (e.g. lambda_mcts_coord_geometry_proxy)
  3. ``known_pt_drug_<base>``            (e.g. known_pt_drug_coord_geometry_proxy)

These files correspond exactly to the 12 entries listed in
``molmetal/scoring/policy.json`` under ``mock_keys_in_metrics_by_metric``.

This script:
  1. Computes the absolute paths for all 12 files.
  2. Verifies each exists (records MISSING if absent).
  3. Deletes each present file via ``os.remove``.
  4. Emits an audit log JSON with: deletion timestamp, per-file
     pre/post existence, byte size before deletion, SHA256 digest
     before deletion (for traceability).
  5. Writes the log to
     ``molmetal/reports/wf119_cleanup/cleanup_log.json``.

CPU-only; no paper work touched (TODO-84 gate honored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"
OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf119_cleanup"
OUT_JSON = OUT_DIR / "cleanup_log.json"

# ---------------------------------------------------------------------------
# 12 stale files to delete
# ---------------------------------------------------------------------------
# Per policy.json:801-813 and TODO-114 §2 (action 2).
MOCK_DELETED_BASES = (
    "coord_geometry_proxy",
    "phototherapy_activity",
    "ptiv_reduction_potential",
    "metalloprotein_affinity",
)
MOCK_DELETED_ALIASES = (
    "",  # bare base name
    "lambda_mcts_",  # lambda_mcts_<base>
    "known_pt_drug_",  # known_pt_drug_<base>
)


def _expected_filenames() -> List[str]:
    """Return the 12 expected filenames (base × 3 aliases)."""
    names: List[str] = []
    for base in MOCK_DELETED_BASES:
        for prefix in MOCK_DELETED_ALIASES:
            names.append(f"{prefix}{base}.json")
    return names


def _sha256_of(path: Path) -> str:
    """Compute SHA256 of *path* contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run(dry_run: bool = False) -> Dict[str, Any]:
    """Delete the 12 stale files (or simulate if dry_run)."""
    filenames = _expected_filenames()
    if len(filenames) != 12:
        raise RuntimeError(
            f"Expected exactly 12 stale files (4 bases × 3 aliases), "
            f"computed {len(filenames)} — fix MOCK_DELETED_BASES/ALIASES."
        )

    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    actions: List[Dict[str, Any]] = []
    deleted_count = 0
    missing_count = 0
    total_bytes_freed = 0

    for fname in filenames:
        path = BY_METRIC_DIR / fname
        row: Dict[str, Any] = {
            "filename": fname,
            "path": str(path),
            "existed_before": path.exists(),
            "deleted": False,
            "dry_run": dry_run,
            "size_bytes": None,
            "sha256": None,
            "status": None,
        }
        if not path.exists():
            row["status"] = "MISSING_BEFORE_DELETION"
            missing_count += 1
            actions.append(row)
            continue

        # Record pre-deletion state for audit
        size = path.stat().st_size
        sha = _sha256_of(path)
        row["size_bytes"] = size
        row["sha256"] = sha
        total_bytes_freed += size

        if dry_run:
            row["status"] = "DRY_RUN_SKIPPED"
            actions.append(row)
            continue

        # Perform the deletion
        os.remove(path)
        row["deleted"] = True
        row["status"] = "DELETED"
        actions.append(row)
        deleted_count += 1

    log: Dict[str, Any] = {
        "workflow": "Wf-119 Phase 3",
        "task": "Delete 12 stale metric JSON files for the 4 MOCK_DELETED "
        "external scorers (coord_geometry_proxy, phototherapy_activity, "
        "ptiv_reduction_potential, metalloprotein_affinity × 3 aliases)",
        "parent_plan": "TODO/pending/114_mock_deleted_external_scorers_cleanup.md",
        "timestamp": timestamp,
        "dry_run": dry_run,
        "by_metric_dir": str(BY_METRIC_DIR),
        "expected_count": 12,
        "deleted_count": deleted_count,
        "missing_count": missing_count,
        "total_bytes_freed": total_bytes_freed,
        "files": actions,
        "post_verification": {
            "files_remaining": sorted(
                fname for fname in filenames if (BY_METRIC_DIR / fname).exists()
            ),
            "remaining_count": sum(
                1 for fname in filenames if (BY_METRIC_DIR / fname).exists()
            ),
        },
        "consumers_updated": {
            "metrics_v2_py": "Patched to raise AdapterUnavailable on import failure (no silent None)",
            "policy_json": "Removed mock_keys_in_metrics_by_metric entries",
            "routing_policy_md": "§2.13 MOCK_DELETED table updated with cleanup annotation",
        },
        "policy_json_alignment": {
            "pre_cleanup_list_size": 12,
            "post_cleanup_list_size": 0,
            "entries_removed": [
                "coord_geometry_proxy",
                "lambda_mcts_coord_geometry_proxy",
                "known_pt_drug_coord_geometry_proxy",
                "phototherapy_activity",
                "lambda_mcts_phototherapy_activity",
                "known_pt_drug_phototherapy_activity",
                "ptiv_reduction_potential",
                "lambda_mcts_ptiv_reduction_potential",
                "known_pt_drug_ptiv_reduction_potential",
                "metalloprotein_affinity",
                "lambda_mcts_metalloprotein_affinity",
                "known_pt_drug_metalloprotein_affinity",
            ],
        },
    }
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the deletion list and audit log without removing files.",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_JSON),
        help=f"Output JSON path (default: {OUT_JSON.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    log = run(dry_run=args.dry_run)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2, sort_keys=False)
        f.write("\n")

    # Echo summary to stdout
    print(f"[wf119_cleanup] {'DRY RUN' if args.dry_run else 'EXECUTED'}")
    print(f"  expected:        12")
    print(f"  deleted:         {log['deleted_count']}")
    print(f"  missing_before:  {log['missing_count']}")
    print(f"  bytes_freed:     {log['total_bytes_freed']:,}")
    print(f"  remaining:       {log['post_verification']['remaining_count']}")
    print(f"  log:             {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
