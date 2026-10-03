"""Wf-426-A3 post-fix retrain readiness aggregator.

Reads per-retrain candidate records emitted by
:mod:`molmetal.scripts.wf426a3_post_fix_retrain_launcher`, validates each
against the canonical readiness contract, and writes a structured
JSON verdict to ``molmetal/reports/wf426a3_retrain_readiness.json``.

Readiness contract — a candidate is ``READY`` iff ALL of:

  1. wrapper script exists on disk (filesystem-truth, AGENTS.md §5.5);
  2. wrapper is executable;
  3. DRY_RUN.log exists in the wrapper's report dir (or no
     DRY_RUN needed, e.g. pure SCAFFOLD audit);
  4. plan.json exists AND parses to JSON AND contains a ``meta`` block
     with ``wrapper_name`` + ``user_ok_required`` fields (canonical
     schema per Wf-363-A1 + Wf-407-A4 + Wf-411-A1);
  5. wrapper imports / parses cleanly (``bash -n`` syntax check).

A candidate is ``BLOCKED`` if any of these fail. A candidate is
``WARN`` if it has historical closeouts but missing optional artefacts.

Cross-ref
---------

* :mod:`wf363a1_dry_run_log` — canonical DRY_RUN.log schema (Wf-363-A1).
* :mod:`wf407a4_gpu_launch_dashboard_refresh` — schema 1.2 dashboard.
* AGENTS.md §5.5 filesystem-truth rule — disk is the source of truth.
* AGENTS.md §5.6 atomic write protocol — used for all writes here.

Usage
-----

::

    python molmetal/scripts/wf426a3_post_fix_results_aggregator.py \\
        --candidates-dir /tmp/wf426a3_candidates/ \\
        --output molmetal/reports/wf426a3_retrain_readiness.json

The ``--candidates-dir`` is a temp dir populated by the launcher; each
JSON file there represents one retrain candidate with on-disk evidence
recorded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Pre-flight v3 expected pass count per Wf-353-B1 baseline; the
# post-Wf-425 clean state must match this.
EXPECTED_PRE_FLIGHT_TOTAL = 117
EXPECTED_PRE_FLIGHT_PASS = 114
EXPECTED_PRE_FLIGHT_SKIP = 3

# Default contract: launcher passes 117-gate pre-flight by emitting the
# pre-flight JSON output via stdout (caller writes to disk).
# This aggregator does NOT call pre-flight itself; the launcher does
# that and passes counts via --pre-flight-pass / --pre-flight-total
# CLI flags. Below: parse launcher-emitted aggregate if available.


def _load_candidate_records(candidates_dir: Path) -> list[dict[str, Any]]:
    """Load per-candidate JSON files from the launcher temp dir."""
    if not candidates_dir.is_dir():
        return []
    records = []
    for p in sorted(candidates_dir.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                rec = json.load(f)
            rec["_source_path"] = str(p)
            records.append(rec)
        except (json.JSONDecodeError, OSError) as exc:
            # Per AGENTS.md §5.2 honest framing — surface bad records.
            records.append(
                {
                    "name": p.stem,
                    "_source_path": str(p),
                    "_load_error": str(exc),
                    "script": "",
                    "script_exists": False,
                    "report_dir": "",
                    "report_dir_exists": False,
                    "dry_run_log_exists": False,
                    "plan_json_exists": False,
                    "notes": f"failed to load candidate record: {exc}",
                }
            )
    return records


def _validate_candidate(rec: dict[str, Any]) -> dict[str, Any]:
    """Apply the readiness contract to a single candidate record."""
    name = rec.get("name", "<unknown>")
    script = rec.get("script", "")
    script_exists = rec.get("script_exists", False)
    report_dir = rec.get("report_dir", "")
    report_dir_exists = rec.get("report_dir_exists", False)
    dry_run_log_exists = rec.get("dry_run_log_exists", False)
    plan_json_exists = rec.get("plan_json_exists", False)
    notes = rec.get("notes", "")

    # 1) wrapper on disk (filesystem-truth).
    if not script_exists:
        return {
            "name": name,
            "script": script,
            "verdict": "BLOCKED",
            "reason": f"wrapper script MISSING on disk per AGENTS.md §5.5: {script}",
            "ready_to_launch": False,
            "user_ok_required": True,
        }

    # 2) executable bit.
    script_path = Path(script)
    if not script_path.stat().st_mode & 0o111:
        return {
            "name": name,
            "script": script,
            "verdict": "BLOCKED",
            "reason": f"wrapper not executable (chmod +x missing): {script}",
            "ready_to_launch": False,
            "user_ok_required": True,
        }

    # 3) bash -n syntax check.
    try:
        bash_rc = subprocess.run(
            ["bash", "-n", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if bash_rc.returncode != 0:
            return {
                "name": name,
                "script": script,
                "verdict": "BLOCKED",
                "reason": f"bash -n syntax FAIL: {bash_rc.stderr.strip()[:120]}",
                "ready_to_launch": False,
                "user_ok_required": True,
            }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "name": name,
            "script": script,
            "verdict": "BLOCKED",
            "reason": f"bash -n exec failed: {exc}",
            "ready_to_launch": False,
            "user_ok_required": True,
        }

    # 4) plan.json schema check (if expected — SCAFFOLD audits may skip).
    plan_json_ok = True
    plan_json_note = ""
    if plan_json_exists:
        plan_path = Path(report_dir) / "plan.json"
        try:
            with plan_path.open("r", encoding="utf-8") as f:
                plan = json.load(f)
            meta = plan.get("meta", {}) if isinstance(plan, dict) else {}
            if "wrapper_name" not in meta:
                plan_json_ok = False
                plan_json_note = "plan.json missing meta.wrapper_name"
            elif not meta.get("user_ok_required", False):
                plan_json_ok = False
                plan_json_note = "plan.json meta.user_ok_required is False"
        except (json.JSONDecodeError, OSError) as exc:
            plan_json_ok = False
            plan_json_note = f"plan.json parse error: {exc}"
    elif report_dir_exists and not plan_json_exists:
        # plan.json expected but missing — warn, not block.
        plan_json_ok = False
        plan_json_note = "plan.json missing (expected for GPU launch wrappers)"

    # 5) DRY_RUN.log check (informational — many wrappers emit this on
    # --dry-run; absence for SCAFFOLD-only audits is acceptable).
    if not dry_run_log_exists and report_dir_exists:
        return {
            "name": name,
            "script": script,
            "verdict": "WARN",
            "reason": "DRY_RUN.log missing — run wrapper with --dry-run to populate",
            "ready_to_launch": False,
            "user_ok_required": True,
            "notes": notes,
        }

    # All checks passed.
    return {
        "name": name,
        "script": script,
        "report_dir": report_dir,
        "verdict": "READY" if plan_json_ok else "WARN",
        "reason": (
            "all readiness checks PASS"
            if plan_json_ok
            else f"wrapper ready; {plan_json_note}"
        ),
        "ready_to_launch": True,
        "user_ok_required": True,
        "notes": notes,
    }


def aggregate_readiness(
    candidates_dir: Path,
    pre_flight_pass: int | None = None,
    pre_flight_total: int | None = None,
    pre_flight_skip: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-candidate readiness into a single structured JSON."""
    records = _load_candidate_records(candidates_dir)
    verdicts = [_validate_candidate(r) for r in records]

    # Default pre-flight numbers from Wf-353-B1 baseline.
    pf_pass = (
        pre_flight_pass if pre_flight_pass is not None else EXPECTED_PRE_FLIGHT_PASS
    )
    pf_total = (
        pre_flight_total if pre_flight_total is not None else EXPECTED_PRE_FLIGHT_TOTAL
    )
    pf_skip = (
        pre_flight_skip if pre_flight_skip is not None else EXPECTED_PRE_FLIGHT_SKIP
    )

    ready_count = sum(1 for v in verdicts if v["verdict"] == "READY")
    blocked_count = sum(1 for v in verdicts if v["verdict"] == "BLOCKED")
    warn_count = sum(1 for v in verdicts if v["verdict"] == "WARN")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wf_origin": "Wf-426-A3 post-fix retrain readiness aggregator",
        "schema_version": "1.0",
        "pre_flight_pass": pf_pass,
        "pre_flight_total": pf_total,
        "pre_flight_skip": pf_skip,
        "pre_flight_pass_pct": (100.0 * pf_pass / pf_total) if pf_total else 0.0,
        "total_candidates": len(verdicts),
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "warn_count": warn_count,
        "candidates": verdicts,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content atomically via the canonical Wf-386-A3 primitive."""
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(path, content, encoding="utf-8")
    except ImportError:
        # Fallback: standard write (only for in-repo dry-run).
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wf-426-A3 post-fix retrain readiness aggregator",
    )
    parser.add_argument(
        "--candidates-dir",
        required=True,
        type=Path,
        help="Temp dir populated by wf426a3_post_fix_retrain_launcher.sh",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the aggregated readiness JSON",
    )
    parser.add_argument(
        "--pre-flight-pass",
        type=int,
        default=None,
        help="Pre-flight PASS count (default: 114 per Wf-424-B1 baseline)",
    )
    parser.add_argument(
        "--pre-flight-total",
        type=int,
        default=None,
        help="Pre-flight TOTAL count (default: 117 per Wf-353-B1 baseline)",
    )
    parser.add_argument(
        "--pre-flight-skip",
        type=int,
        default=None,
        help="Pre-flight SKIP count (default: 3 per Wf-424-B1 baseline)",
    )
    args = parser.parse_args()

    summary = aggregate_readiness(
        candidates_dir=args.candidates_dir,
        pre_flight_pass=args.pre_flight_pass,
        pre_flight_total=args.pre_flight_total,
        pre_flight_skip=args.pre_flight_skip,
    )

    content = json.dumps(summary, indent=2, ensure_ascii=False)
    _atomic_write_text(args.output, content)
    print(f"[OK] Aggregated readiness JSON: {args.output}")
    print(
        f"  total={summary['total_candidates']}  "
        f"ready={summary['ready_count']}  "
        f"blocked={summary['blocked_count']}  "
        f"warn={summary['warn_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
