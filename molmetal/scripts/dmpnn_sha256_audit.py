"""Wf-120 Phase 1 — SHA256 audit for D-MPNN pIC50 checkpoints.

=====================================================================
Per-ckpt SHA256 audit for ``molmetal/checkpoints/dmpnn_pic50_*.pt``
=====================================================================

Reads every ``dmpnn_pic50_{metal}_seed{N}.pt`` checkpoint under
``molmetal/checkpoints/``, computes its SHA256, and verifies that:

  1. Every (metal, seed) pair has a unique SHA256 (i.e. seed0 and
     seed1 of the same metal are byte-different — defends against
     the silent copy-paste bug that produced the original
     ``dmpnn_pic50_rh_seed1.pt == dmpnn_pic50_rh_seed0.pt`` claim).
  2. Every reported SHA256 in ``metrics/by_metric/dmpnn_ckpt_status.json``
     matches the on-disk file (or — if the JSON lacks a ``sha256`` field —
     the script reports the missing fields so the JSON can be back-filled).

**Hardware**: CPU only (SHA256 + JSON read).

**Usage**
-----
::
    python molmetal/scripts/dmpnn_sha256_audit.py \\
        --ckpt-dir molmetal/checkpoints \\
        --status-json metrics/by_metric/dmpnn_ckpt_status.json \\
        --out molmetal/reports/wf120_p1_rh_retrain/sha256_audit.json

Exit code is non-zero if the audit finds duplicate SHAs or
JSON mismatches.

**Honest framing**
------------------
* SHA256 diversity is a **necessary** but **not sufficient** condition for
  true seed diversity — two checkpoints with different SHAs can still
  share the same trained weights if the trainer is buggy (e.g. always
  loads the same init state). But identical SHAs across ``seed0``/``seed1``
  is a **smoking gun** for the silent copy bug we are guarding against.
* ``dmpnn_pic50_{ru,ir,os,re}_seed0.pt`` and ``seed1.pt`` were trained in
  Wf-84 Phase 3 with 10 epochs; Wf-120 P1 only re-trains Rh seed1 (with
  30 epochs). For the other metals, SHA256 diversity is preserved by the
  original Wf-84 training, but the SHA256s here provide the audit
  snapshot used by downstream paper-grade reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Wf-120 P3 — import shared SHA256-guard helpers from the dependency-free
# module so the audit re-uses the canonical scan_all_ckpts /
# find_duplicate_shas implementations.
from molmetal.scripts._dmpnn_sha256_guard import (  # noqa: E402
    find_duplicate_shas as _guard_find_duplicate_shas,
)

DEFAULT_CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"
DEFAULT_STATUS_JSON = PROJECT_ROOT / "metrics" / "by_metric" / "dmpnn_ckpt_status.json"
FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")


def sha256_of_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Compute SHA256 of ``path`` streaming 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def discover_dmpnn_ckpts(ckpt_dir: Path) -> List[Path]:
    """Return ``dmpnn_pic50_*.pt`` paths under ``ckpt_dir``, sorted."""
    if not ckpt_dir.is_dir():
        return []
    return sorted(ckpt_dir.glob("dmpnn_pic50_*.pt"))


def parse_filename(p: Path) -> Optional[Tuple[str, int]]:
    """Parse ``dmpnn_pic50_{metal}_seed{N}.pt`` → (metal, seed) or None."""
    stem = p.stem  # dmpnn_pic50_{metal}_seed{N}
    parts = stem.split("_")
    if len(parts) < 4 or not parts[-1].startswith("seed"):
        return None
    seed_str = parts[-1][4:]
    if not seed_str.isdigit():
        return None
    metal = "_".join(parts[2:-1])  # metal might be "ru", "ir", "rh", "os", "re"
    # Capitalise for parity with status JSON
    metal_cap = metal.capitalize()
    if metal_cap not in FIVE_METALS:
        return None
    return metal_cap, int(seed_str)


def audit(ckpt_dir: Path, status_json: Optional[Path]) -> Dict:
    """Run the full audit; return a structured report dict.

    Wf-120 P3 extension — adds three new fields beyond the Phase 1 contract:

      * ``per_metal_seed_summary`` — ``{metal: {seed: sha256}}`` matrix
        for quick per-(metal, seed) lookup.
      * ``per_metal_diversity`` — ``{metal: {"n_ckpts": int, "n_unique_sha":
        int, "diversity_ratio": float}}``. ``diversity_ratio == 1.0``
        means every (metal, seed) has a unique SHA — the desired state.
      * ``simulated_post_save_guard`` — ``{"would_pass": bool,
        "blocking_duplicate_sha": Optional[str], "blocking_filenames":
        Optional[Tuple[str, ...]]}``. This simulates the Wf-120 P3
        post-save guard on the **current on-disk state**: if any
        duplicate SHA is present, the simulated guard reports the
        blocking SHA + filenames so the operator can manually delete
        one of them before re-running a training pass.
    """
    paths = discover_dmpnn_ckpts(ckpt_dir)
    rows: List[Dict] = []
    sha_index: Dict[str, List[str]] = defaultdict(list)  # sha → [filenames]
    parse_errors: List[str] = []

    for p in paths:
        parsed = parse_filename(p)
        if parsed is None:
            parse_errors.append(p.name)
            continue
        metal, seed = parsed
        sha = sha256_of_file(p)
        size = p.stat().st_size
        rows.append(
            {
                "filename": p.name,
                "metal": metal,
                "seed": seed,
                "size_bytes": size,
                "sha256": sha,
            }
        )
        sha_index[sha].append(p.name)

    # Identify duplicate SHAs (>= 2 filenames with same SHA) — these are the
    # smoking gun for the silent copy bug.
    duplicate_shas: Dict[str, List[str]] = {
        sha: names for sha, names in sha_index.items() if len(names) > 1
    }

    # JSON reconciliation
    json_mismatches: List[Dict] = []
    json_missing_sha: List[str] = []
    if status_json is not None and status_json.is_file():
        with status_json.open("r") as f:
            status = json.load(f)
        # Build filename → row index
        reported = {row["filename"]: row for row in status.get("checkpoints", [])}
        for row in rows:
            entry = reported.get(row["filename"])
            if entry is None:
                continue
            reported_sha = entry.get("sha256")
            if reported_sha is None:
                json_missing_sha.append(row["filename"])
                continue
            if reported_sha != row["sha256"]:
                json_mismatches.append(
                    {
                        "filename": row["filename"],
                        "json_sha256": reported_sha,
                        "fs_sha256": row["sha256"],
                    }
                )

    # ---- Wf-120 P3 extension: per-(metal, seed) summary + diversity + sim guard
    per_metal_seed_summary: Dict[str, Dict[str, str]] = defaultdict(dict)
    per_metal_diversity: Dict[str, Dict] = {}
    for row in rows:
        per_metal_seed_summary[row["metal"]][str(row["seed"])] = row["sha256"]

    for metal, seed_to_sha in per_metal_seed_summary.items():
        n_ckpts = len(seed_to_sha)
        n_unique = len(set(seed_to_sha.values()))
        per_metal_diversity[metal] = {
            "n_ckpts": n_ckpts,
            "n_unique_sha": n_unique,
            "diversity_ratio": (n_unique / n_ckpts) if n_ckpts else 0.0,
        }

    if duplicate_shas:
        first_sha, first_files = next(iter(sorted(duplicate_shas.items())))
        simulated_guard = {
            "would_pass": False,
            "blocking_duplicate_sha": first_sha,
            "blocking_filenames": sorted(first_files),
        }
    else:
        simulated_guard = {
            "would_pass": True,
            "blocking_duplicate_sha": None,
            "blocking_filenames": None,
        }
    # ---- end Wf-120 P3 extension

    report = {
        "audit_tag": "Wf-120 Phase 1 — D-MPNN pIC50 ckpt SHA256 audit",
        "ckpt_dir": str(ckpt_dir),
        "n_ckpts_scanned": len(rows),
        "n_parse_errors": len(parse_errors),
        "parse_errors": parse_errors,
        "n_duplicate_shas": len(duplicate_shas),
        "duplicate_shas": duplicate_shas,
        "n_json_mismatches": len(json_mismatches),
        "json_mismatches": json_mismatches,
        "n_json_missing_sha": len(json_missing_sha),
        "json_missing_sha": json_missing_sha,
        "checkpoints": rows,
        # Wf-120 P3 additions
        "per_metal_seed_summary": {
            m: dict(v) for m, v in per_metal_seed_summary.items()
        },
        "per_metal_diversity": per_metal_diversity,
        "simulated_post_save_guard": simulated_guard,
        "status": (
            "PASS"
            if (len(duplicate_shas) == 0 and len(json_mismatches) == 0)
            else "FAIL"
        ),
    }
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-120 Phase 1+3 — SHA256 audit + simulated P3 overwrite-guard "
            "for D-MPNN pIC50 ckpts."
        ),
    )
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the report JSON (default: stdout only).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = audit(args.ckpt_dir, args.status_json)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
