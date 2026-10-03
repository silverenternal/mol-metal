#!/usr/bin/env python3
"""_wrapper_audit.py — classifies every scripts/launch_wf*.sh wrapper as one of:

  - REAL:        USER_OK gate + DRY_RUN.log + plan.json emission + invokes an actual
                 trainer script (.venv/bin/python) via PYTHONPATH=REPO_ROOT
  - SCAFFOLD:    any wrapper that lacks the full 4-property canonical surface
                 (USER_OK + DRY_RUN.log + plan.json + canonical trainer call)
  - DUPLICATE:   target_overlap with another wrapper (detected via DOCK_PANEL
                 field emission inside the wrapper)

Per Wf-431-A2 FIX 2 (wrapper reclassification + single wrapper template) +
AGENTS.md §5.5 filesystem-truth + §5.6 atomic write + §5.9 hardcoded-PY rule.

Stdlib-only — no third-party deps. Reads every wrapper for lane >= 1, classifies
via regex matching against the canonical surface markers, emits structured JSON to
molmetal/reports/wrapper_audit.json, and returns non-zero exit code if any wrapper
violates the canonical surface (i.e. is SCAFFOLD or DUPLICATE).

Usage:
  python scripts/_wrapper_audit.py                 # default REPO_ROOT discovery
  python scripts/_wrapper_audit.py --strict       # exit 1 on any violation
  python scripts/_wrapper_audit.py --json-only    # console.log the JSON and exit 0

Cross-ref: scripts/_WRAPPER_TEMPLATE.sh is the canonical surface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS_DIR = REPO_ROOT / "scripts"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"
AUDIT_JSON = REPORTS_DIR / "wrapper_audit.json"

# canonical surface markers — must all be present for REAL classification.
# Patterns tuned to the REAL wrappers on disk (Wf-355-B1 USER_OK-gate family):
#   - `if [ "${USER_OK:-0}" != "1" ]` is the Wf-294B gate
#   - `PYTHONPATH=/home/hugo/codes/mol-metal ${PY}` is the Wf-431-A1 fix pattern
#   - `PY=/home/hugo/codes/mol-metal/.venv/bin/python` is the §5.9 hardcoded rule
# `export PYTHONPATH` is an acceptable alternative (catches wrappers that do
# `export PYTHONPATH=...` once at the top instead of inline).
CANONICAL_MARKERS = {
    "USER_OK gate": re.compile(r'USER_OK:-0}.*\s*!=\s*"1"', re.IGNORECASE),
    "DRY_RUN emission": re.compile(r"DRY_RUN\.log", re.IGNORECASE),
    "plan_json emission": re.compile(r"plan\.json", re.IGNORECASE),
    "pre_flight_check gate": re.compile(r"pre_flight_check\.py", re.IGNORECASE),
    "PYTHONPATH export": re.compile(
        r'PYTHONPATH\s*=\s*[/"\']?/home/hugo/codes/mol-metal(?:\s|\$|\")', re.IGNORECASE
    ),
    "PY hardcoded-via-bin/python": re.compile(
        r"(?:PY|py|python)[: =].{0,40}/home/hugo/codes/mol-metal/\.venv/bin/python",
        re.IGNORECASE,
    ),
    "USER_OK env propagation to trainer": re.compile(r"--user-ok", re.IGNORECASE),
}

# DUPLICATE detector — DOCK_PANEL references the same trainer
DOCK_PANEL_RE = re.compile(r'DOCK_PANEL|TRAINER\s*=\s*"?([\w_./-]+\.py)"?')


def discover_wrappers(scripts_dir: Path) -> list[Path]:
    """List every launch_wf*.sh wrapper on disk."""
    return sorted(scripts_dir.glob("launch_wf*.sh"))


def surface_check(path: Path) -> dict[str, bool]:
    """Check which canonical markers are present in a wrapper."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {name: False for name in CANONICAL_MARKERS} | {"_read_error": str(exc)}
    result = {
        name: bool(pattern.search(content))
        for name, pattern in CANONICAL_MARKERS.items()
    }
    # exclude the template from classification noise (it's the reference itself)
    result["_template"] = path.name.startswith("_")
    return result


def dock_panel(path: Path) -> str:
    """Extract DOCK_PANEL or TRAINER__name reference from a wrapper."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    m = DOCK_PANEL_RE.search(content)
    if m:
        # group(1) captures the explicit TRAINER value; if absent (DOCK_PANEL only), return ""
        return m.group(1) if m.group(1) else "<DOCK_PANEL-placeholder>"
    # Fall back: any .py reference in the trainer section
    m2 = re.search(r'(?:TRAINER|--trainer)\s+["\']?([\w_./-]+\.py)', content)
    return m2.group(1) if m2 else ""


def has_multiple_device_flags(path: Path) -> bool:
    """CANONICAL flag detection: return True if wrapper has > 1 --device flag invocation.

    Per AGENTS.md §5.5 filesystem-truth + Wf-A2 wrapper consolidation: a CANONICAL
    wrapper has exactly ONE --device flag invocation (the canonical entry-point).
    Multiple --device flags indicate DUPLICATE pattern (e.g. inline + for-loop).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return len(re.findall(r"--device\s+\w+", content)) > 1


def classify(surface: dict[str, bool], wrapper_path: Path) -> tuple[str, str]:
    """Classify a wrapper based on its canonical surface coverage.

    Returns (classification, evidence).

    Per Wf-A2 ADR docs-only sync: extend to factor in CANONICAL flag detection
    (multiple --device flags = DUPLICATE pattern per AGENTS.md §5.5 filesystem-truth).
    """
    if surface.get("_read_error"):
        return "DUPLICATE", f"read_error: {surface['_read_error']}"
    matched = [k for k, v in surface.items() if v and k != "_template"]
    missing = [k for k, v in surface.items() if not v and k != "_template"]
    if has_multiple_device_flags(wrapper_path):
        return (
            "DUPLICATE",
            "multiple --device flag invocations (>1) — CANONICAL flag pattern violated",
        )
    if len(matched) == len(CANONICAL_MARKERS):
        return "REAL", "all 7 canonical markers present"
    if surface.get("_template"):
        return "DUAL: TEMPLATE", "template file (excluded from classification)"
    return "SCAFFOLD", f"missing_markers: {",".join(missing)}"


def collect_duplicates(wrappers: list[Path]) -> dict[str, str]:
    """Map dock_panel → wrapper name (primary collision key)."""
    panels: dict[str, str] = {}
    for w in wrappers:
        if w.name.startswith("_"):
            continue
        panel = dock_panel(w)
        if not panel or panel.startswith("<"):
            continue
        if panel in panels:
            panels[panel] = panels[panel] + " + " + w.name
        else:
            panels[panel] = w.name
    return {k: v for k, v in panels.items() if " + " in v}


def audit(strict: bool = False) -> dict:
    """Run the full audit and emit structured JSON."""
    wrappers = discover_wrappers(SCRIPTS_DIR)
    records = []
    summary = {"REAL": 0, "SCAFFOLD": 0, "DUPLICATE": 0, "TEMPLATE": 0}
    for w in wrappers:
        surface = surface_check(w)
        cls, evidence = classify(surface, w)
        records.append(
            {
                "wrapper": w.name,
                "path": str(w),
                "size_bytes": w.stat().st_size,
                "mode_octal": oct(w.stat().st_mode & 0o777),
                "classification": cls,
                "evidence": evidence,
                "matched_markers": [
                    k for k, v in surface.items() if v and k != "_template"
                ],
                "missing_markers": [
                    k for k, v in surface.items() if not v and k != "_template"
                ],
            }
        )
        if w.name.startswith("_"):
            summary["TEMPLATE"] += 1
        elif cls.startswith("DUAL"):
            summary["TEMPLATE"] += 1
        else:
            summary[cls] = summary.get(cls, 0) + 1

    duplicates = collect_duplicates(wrappers)
    for record in records:
        record["duplicates_panel_with"] = [
            p
            for p, who in duplicates.items()
            if record["wrapper"] in who
            and record["wrapper"] not in who.replace(record["wrapper"], "")
        ]

    return {
        "audit_ts": datetime.now(timezone.utc).isoformat(),
        "wrapper_count": len(wrappers),
        "summary": summary,
        "duplicates": duplicates,
        "canonical_markers": list(CANONICAL_MARKERS.keys()),
        "template": "scripts/_WRAPPER_TEMPLATE.sh",
        "wrappers": records,
        "violations": [
            r["wrapper"]
            for r in records
            if r["classification"] in ("SCAFFOLD", "DUPLICATE")
        ],
    }


def write_audit_json(audit_obj: dict) -> None:
    """Atomic-write the audit JSON to molmetal/reports/wrapper_audit.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(AUDIT_JSON, json.dumps(audit_obj, indent=2, sort_keys=True))
    except ImportError:
        # Fallback: direct write_text (NOT atomic — only as a last resort if the
        # atomic primitive fails to import). Audit script is CI-safe.
        AUDIT_JSON.write_text(
            json.dumps(audit_obj, indent=2, sort_keys=True), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper audit per Wf-431-A2 FIX 2")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 on any SCAFFOLD/DUPLICATE"
    )
    parser.add_argument(
        "--json-only", action="store_true", help="console.log the JSON and exit 0"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="console.log the summary only"
    )
    args = parser.parse_args()

    audit_obj = audit(strict=args.strict)
    write_audit_json(audit_obj)

    if args.summary_only:
        s = audit_obj["summary"]
        total = audit_obj["wrapper_count"]
        print(
            json.dumps(
                {"summary": s, "total": total, "violations": audit_obj["violations"]},
                indent=2,
            )
        )
    elif args.json_only:
        print(json.dumps(audit_obj, indent=2, sort_keys=True))
    else:
        s = audit_obj["summary"]
        total = audit_obj["wrapper_count"]
        print(f"[wrapper_audit] {total} wrappers scanned → {s}")
        if audit_obj["violations"]:
            print(f"[wrapper_audit] {len(audit_obj['violations'])} violations:")
            for v in audit_obj["violations"]:
                print(f"  - {v}")
        print(f"[wrapper_audit] JSON: {AUDIT_JSON}")

    return 1 if args.strict and audit_obj["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
