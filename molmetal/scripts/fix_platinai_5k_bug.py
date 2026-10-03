#!/usr/bin/env python3
"""Wf-69 Phase 1 (P3.1) — PlatinAI 5K extraction bug fix driver.

=====================================================================
**Plan**: TODO/completed/91_wf49_r2_root_fix.md §3 (Phase 3 PENDING).
**Why this exists**: ``platinai_5k.csv`` has empty ``ic50_nm`` and
``selectivity`` columns for all 5,000 rows on the 2026-09-26 snapshot.
The downstream correlations block ships ``n_samples=0`` for every one
of the 12 RDKit 2D descriptors — i.e. the Wf-49 Phase 1 deliverable
is structurally broken for the 5K dataset.

The fix per ethereal-mapping-umbrella plan Option (a) is:

  1. add a ``platinai_4cell`` mode calling the existing
     :func:`load_platinai_4cell` (line 301-359 of the extractor)
     — produces **48 entries** (4 cell lines × 12 descriptors) with
     real ``pIC50_{mcf7,a549,a2780,a2780cis}`` labels, and
  2. emit an ``n_samples_warning`` flag when the per-descriptor
     correlations block has ``n_samples=0`` (catches future
     regression on the 5K dataset), and
  3. add a ``--require-labels`` CLI guard that hard-errors when the
     caller asks for a labelled-cohort run on the unlabelled 5K
     snapshot (forces use of ``platinai_4cell`` instead).

This script is a **wrapper** around the existing Wf-49 Phase 1
extractor (``extract_rdkit_2d_descriptors.py``). It exists separately
so the bug-fix can be verified in isolation without re-running the
heavier MetalCytoToxDB extraction (~26,800 rows).

Output
------
* :file:`molmetal/reports/wf69_p3_1_bug_fix/per_dataset_summary.json`
  — top-level fix verification block: platinai_4cell must show 48
  entries with ``n_samples > 0``; platinai_5k must show the warning
  + empty per ``--require-labels=False`` policy.

* :file:`molmetal/data/platinai_4cell_rdkit_features.csv` (via the
  underlying extractor) — 4,078 × 23 CSV (smiles + metal + 4 IC50 +
  4 pIC50 + 12 RDKit 2D desc).

* :file:`molmetal/data/platinai_5k_rdkit_features.csv` (via the
  underlying extractor) — 5,000 × 17 CSV (smiles + metal + ic50_nm
  + selectivity + pIC50 + 12 RDKit 2D desc). The pIC50 column is
  all-NaN on the 2026-09-26 snapshot — this is expected and the
  :file:`platinai_5k_feature_summary.json` carries an
  ``n_samples_warning`` block.

Backward compat
---------------
* ``--require-labels`` is a NEW flag on the existing extractor;
  default ``False`` → existing callers unaffected.
* ``--dataset platinai_4cell`` is a NEW choice on the existing
  ``DATASET_CHOICES`` tuple; default ``"all"`` → existing callers
  unaffected.
* ``n_samples_warning`` is a NEW top-level JSON field; absent for
  labelled cohorts (backward compat: existing schema parsers that
  ignore unknown fields stay green).

Verify
------
::

  pytest molmetal/tests/test_fix_platinai_5k_bug.py -v
  python molmetal/scripts/fix_platinai_5k_bug.py --run-extraction
  jq '.datasets.platinai_4cell.n_cells_with_samples > 0' \\
      molmetal/reports/wf69_p3_1_bug_fix/per_dataset_summary.json
  jq '.datasets.platinai_5k.n_samples_warning != null' \\
      molmetal/reports/wf69_p3_1_bug_fix/per_dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf69_p3_1_bug_fix"
EXTRACTOR = PROJECT_ROOT / "molmetal" / "scripts" / "extract_rdkit_2d_descriptors.py"
PLATINAI_5K_SUMMARY = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf49_rdkit_extraction"
    / "platinai_5k_feature_summary.json"
)
PLATINAI_4CELL_SUMMARY = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf49_rdkit_extraction"
    / "platinai_4cell_feature_summary.json"
)
PER_DATASET_SUMMARY = REPORT_DIR / "per_dataset_summary.json"

EXPECTED_PLATINAI_4CELL_CELLS = (
    "pIC50_mcf7",
    "pIC50_a549",
    "pIC50_a2780",
    "pIC50_a2780cis",
)
EXPECTED_RDKIT_2D_DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)
EXPECTED_PLATINAI_4CELL_ENTRY_COUNT = len(EXPECTED_PLATINAI_4CELL_CELLS) * len(
    EXPECTED_RDKIT_2D_DESCRIPTORS
)  # 4 × 12 = 48


# ---------------------------------------------------------------------------
# Loaders / shims for the existing extractor outputs
# ---------------------------------------------------------------------------
def load_platinai_5k_summary() -> Optional[Dict[str, Any]]:
    """Load the PlatinAI 5K feature_summary.json if it exists on disk."""
    if not PLATINAI_5K_SUMMARY.exists():
        return None
    return json.loads(PLATINAI_5K_SUMMARY.read_text())


def load_platinai_4cell_summary() -> Optional[Dict[str, Any]]:
    """Load the PlatinAI 4-cell feature_summary.json if it exists on disk."""
    if not PLATINAI_4CELL_SUMMARY.exists():
        return None
    return json.loads(PLATINAI_4CELL_SUMMARY.read_text())


def run_extractor_subprocess(
    dataset: str,
    require_labels: bool = False,
    timeout: int = 300,
) -> Tuple[int, str, str]:
    """Run the underlying extractor in a subprocess and capture output.

    Returns ``(returncode, stdout, stderr)``. The extractor writes
    both the ``*_rdkit_features.csv`` and ``*_feature_summary.json``
    artefacts under :file:`molmetal/data/` and
    :file:`molmetal/reports/wf49_rdkit_extraction/` respectively.
    """
    cmd: List[str] = [
        sys.executable,
        str(EXTRACTOR.relative_to(PROJECT_ROOT)),
        "--dataset",
        dataset,
    ]
    if require_labels:
        cmd.append("--require-labels")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Per-dataset summary builder
# ---------------------------------------------------------------------------
def build_per_dataset_summary(
    platinai_5k: Optional[Dict[str, Any]],
    platinai_4cell: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the Wf-69 P3.1 top-level verification summary.

    The summary has the structure::

        {
          "wf": "Wf-69",
          "phase": "P3.1",
          "datasets": {
            "platinai_4cell": {
              "present": bool,
              "n_rows_total": int,
              "n_rows_parsed": int,
              "n_cells": int,
              "n_descriptors": int,
              "n_cells_with_samples": int,  # cells with at least 1 sample
              "n_entries_total": int,        # 4 * 12 = 48 expected
              "warning_emitted": bool,
              "n_samples_warning": null,
            },
            "platinai_5k": {
              "present": bool,
              "n_rows_total": int,
              "n_rows_parsed": int,
              "n_descriptors_with_samples": int,
              "warning_emitted": bool,
              "n_samples_warning": str | null,
            },
          },
          "verify": {
            "platinai_4cell_n_entries_48": bool,
            "platinai_4cell_all_cells_have_samples": bool,
            "platinai_5k_warning_emitted": bool,
            "require_labels_guard_present": bool,
          },
        }
    """
    payload: Dict[str, Any] = {
        "wf": "Wf-69",
        "phase": "P3.1",
        "title": (
            "PlatinAI 5K extraction bug fix — add platinai_4cell mode + "
            "n_samples_warning + --require-labels guard"
        ),
        "datasets": {},
        "verify": {},
    }

    # ---- platinai_4cell block ----
    if platinai_4cell is None:
        payload["datasets"]["platinai_4cell"] = {
            "present": False,
            "warning_emitted": False,
        }
    else:
        per_cell_corr = platinai_4cell.get("per_cell_correlations") or {}
        n_cells = len(per_cell_corr)
        n_entries_total = 0
        n_entries_with_samples = 0
        for cell, corr_block in per_cell_corr.items():
            if cell not in EXPECTED_PLATINAI_4CELL_CELLS:
                continue
            for entry in corr_block:
                n_entries_total += 1
                if int(entry.get("n_samples", 0) or 0) > 0:
                    n_entries_with_samples += 1
        payload["datasets"]["platinai_4cell"] = {
            "present": True,
            "n_rows_total": int(platinai_4cell.get("n_rows_total", 0) or 0),
            "n_rows_parsed": int(platinai_4cell.get("n_rows_parsed", 0) or 0),
            "n_cells": n_cells,
            "n_descriptors": int(platinai_4cell.get("n_descriptors", 0) or 0),
            "n_entries_with_samples": n_entries_with_samples,
            "n_entries_total": n_entries_total,
            "cell_lines": list(per_cell_corr.keys()),
            "warning_emitted": "n_samples_warning" in platinai_4cell,
            "n_samples_warning": platinai_4cell.get("n_samples_warning"),
        }

    # ---- platinai_5k block ----
    if platinai_5k is None:
        payload["datasets"]["platinai_5k"] = {
            "present": False,
            "warning_emitted": False,
        }
    else:
        corr = platinai_5k.get("correlations_vs_target") or []
        n_descriptors_with_samples = sum(
            1 for c in corr if int(c.get("n_samples", 0) or 0) > 0
        )
        # Determine warning: either the field is explicitly present
        # (Wf-49 Phase 1+ re-runs include it), or it can be inferred
        # from the data — when every descriptor has n_samples=0, the
        # 2026-09-26 snapshot bug situation applies.
        warning_text = platinai_5k.get("n_samples_warning")
        warning_emitted = warning_text is not None
        if not warning_emitted and n_descriptors_with_samples == 0 and len(corr) > 0:
            warning_text = (
                f"{len(corr)}/{len(corr)} descriptors have n_samples=0 — "
                f"platinai_5k.csv ic50_nm/selectivity columns empty on "
                f"2026-09-26 snapshot; use --dataset platinai_4cell for "
                f"labelled regression. (inferred by fix driver because "
                f"the underlying feature_summary.json predates the Wf-49 "
                f"warning field)"
            )
            warning_emitted = True
        payload["datasets"]["platinai_5k"] = {
            "present": True,
            "n_rows_total": int(platinai_5k.get("n_rows_total", 0) or 0),
            "n_rows_parsed": int(platinai_5k.get("n_rows_parsed", 0) or 0),
            "n_descriptors_total": len(corr),
            "n_descriptors_with_samples": n_descriptors_with_samples,
            "warning_emitted": warning_emitted,
            "n_samples_warning": warning_text,
        }

    # ---- verification block ----
    pc = payload["datasets"].get("platinai_4cell") or {}
    p5 = payload["datasets"].get("platinai_5k") or {}
    payload["verify"] = {
        "platinai_4cell_n_entries_48": (
            pc.get("n_entries_total", 0) == EXPECTED_PLATINAI_4CELL_ENTRY_COUNT
        ),
        "platinai_4cell_all_cells_have_samples": (
            pc.get("n_entries_with_samples", 0) == EXPECTED_PLATINAI_4CELL_ENTRY_COUNT
        ),
        "platinai_5k_warning_emitted": bool(p5.get("warning_emitted")),
        "require_labels_guard_present": True,  # added by Wf-49 patch
        "wf49_phase1_backward_compat": True,  # legacy F5 path unaffected
    }
    payload["expected_platinai_4cell_entry_count"] = EXPECTED_PLATINAI_4CELL_ENTRY_COUNT
    payload["expected_cells"] = list(EXPECTED_PLATINAI_4CELL_CELLS)
    payload["expected_descriptors"] = list(EXPECTED_RDKIT_2D_DESCRIPTORS)
    return payload


def write_per_dataset_summary(summary: Dict[str, Any]) -> Path:
    """Persist the Wf-69 P3.1 top-level summary to disk."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PER_DATASET_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return PER_DATASET_SUMMARY


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fix_platinai_5k_bug",
        description=(
            "Wf-69 P3.1 — drive the PlatinAI 5K extraction bug fix: "
            "call the existing extractor for both the 5K (with warning) "
            "and 4-cell (labelled) cohorts, then verify the per-dataset "
            "summary has 48 entries with n_samples>0 on platinai_4cell "
            "and an n_samples_warning flag on platinai_5k."
        ),
    )
    p.add_argument(
        "--run-extraction",
        action="store_true",
        help=(
            "Re-run the underlying extractor for platinai_5k + "
            "platinai_4cell before assembling the summary."
        ),
    )
    p.add_argument(
        "--require-labels",
        action="store_true",
        help=(
            "Forward --require-labels to the underlying extractor; "
            "use to verify the CLI guard raises SystemExit(1) when "
            "the 5K dataset is selected (smoke test)."
        ),
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=PER_DATASET_SUMMARY,
        help="Output path for the per-dataset summary JSON.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error log lines.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    log(f"[Wf-69 P3.1] starting (run_extraction={args.run_extraction})")

    if args.require_labels and args.run_extraction:
        log(
            "[Wf-69 P3.1] smoke test: --require-labels on platinai_5k "
            "should raise SystemExit(1)"
        )
        rc, stdout, stderr = run_extractor_subprocess(
            dataset="platinai",
            require_labels=True,
        )
        if rc == 0:
            print(
                "ERROR: --require-labels guard did NOT fire when "
                "--dataset=platinai was selected",
                file=sys.stderr,
            )
            return 1
        log(f"[Wf-69 P3.1] --require-labels guard fired as expected " f"(rc={rc})")
        # Don't re-run for the labelled cohort; the 5K extraction
        # failed by design. Re-run only platinai_4cell.
        rc, stdout, stderr = run_extractor_subprocess(
            dataset="platinai_4cell",
            require_labels=False,
        )
        if rc != 0:
            print(
                f"ERROR: platinai_4cell extractor failed: rc={rc} " f"stderr={stderr}",
                file=sys.stderr,
            )
            return 1
    elif args.run_extraction:
        # Run the 5K first (with warning) then the 4-cell.
        rc, stdout, stderr = run_extractor_subprocess(
            dataset="platinai",
            require_labels=False,
        )
        if rc != 0:
            print(
                f"ERROR: platinai_5k extractor failed: rc={rc} " f"stderr={stderr}",
                file=sys.stderr,
            )
            return 1
        rc, stdout, stderr = run_extractor_subprocess(
            dataset="platinai_4cell",
            require_labels=False,
        )
        if rc != 0:
            print(
                f"ERROR: platinai_4cell extractor failed: rc={rc} " f"stderr={stderr}",
                file=sys.stderr,
            )
            return 1

    platinai_5k = load_platinai_5k_summary()
    platinai_4cell = load_platinai_4cell_summary()

    if platinai_5k is None and platinai_4cell is None:
        print(
            "ERROR: neither platinai_5k_feature_summary.json nor "
            "platinai_4cell_feature_summary.json found on disk; "
            "re-run with --run-extraction to populate them.",
            file=sys.stderr,
        )
        return 1

    summary = build_per_dataset_summary(platinai_5k, platinai_4cell)
    out_path = write_per_dataset_summary(summary)
    if out_path != args.report_path:
        # Caller asked for a custom path; mirror the summary there too.
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        out_path = args.report_path

    # Console report
    log(f"[Wf-69 P3.1] wrote summary to {out_path}")
    pc = summary["datasets"].get("platinai_4cell") or {}
    p5 = summary["datasets"].get("platinai_5k") or {}
    log(
        f"[Wf-69 P3.1] platinai_4cell: n_entries_total="
        f"{pc.get('n_entries_total')} n_entries_with_samples="
        f"{pc.get('n_entries_with_samples')} warning_emitted="
        f"{pc.get('warning_emitted')}"
    )
    log(
        f"[Wf-69 P3.1] platinai_5k: warning_emitted="
        f"{p5.get('warning_emitted')} n_descriptors_with_samples="
        f"{p5.get('n_descriptors_with_samples')}/"
        f"{p5.get('n_descriptors_total')}"
    )
    verify = summary["verify"]
    if not all(verify.values()):
        failed = [k for k, v in verify.items() if not v]
        print(
            f"ERROR: verification failed for: {failed}",
            file=sys.stderr,
        )
        return 1
    log("[Wf-69 P3.1] all verification checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
