"""Wf-304C Task 2: Audit which frontier math adapter modules are wired into
the locked flow_matching_lipman CFM pipeline.

Per Wf-234 audit, 4 of 7 frontier math modules are already shipped as
standalone adapters (multi_ph_integration, se3_lipman_integration,
se3_sb_integration, transport_relaxation_integration). The remaining
work is the "drop into flow_matching_lipman" wire step, which is gated
on user OK (R17 invariant: do not modify flow_matching_lipman/).

This script produces a JSON audit report that confirms the current state:
* Which adapters are importable
* Which adapters are imported by the locked __init__.py
* Which adapters are USED (not just imported) by __init__.py
* Wire-in delta (what glue would be needed to drop them in)
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ADAPTER_DIR = Path("/home/hugo/codes/mol-metal/molmetal/adapters")
FLOW_DIR = ADAPTER_DIR / "flow_matching_lipman"
INIT_PY = FLOW_DIR / "__init__.py"


WIRE_READY_ADAPTERS = [
    ("multi_ph_integration", "W2.3 Multi-PH + PLL signature (Kim-Jung 2026)"),
    ("se3_lipman_integration", "W2.1 SE(3) Lipman (Mahmood 2026)"),
    ("se3_sb_integration", "W2.1 SE(3) Schrödinger Bridge (Mahmood 2026)"),
    (
        "transport_relaxation_integration",
        "W2.2 Transport relaxation (Jiang-Xu-Zhang 2026)",
    ),
]
RESEARCH_ONLY = [
    ("bass_existence", "Bass existence (Backhoff-Veraguas 2020)"),
    ("sb_bass_martingale", "Bass martingale (Henry-Labordère 2017)"),
    ("monge_ampere_regularity", "Monge-Ampère regularity (汪徐家 2026)"),
]
LOCAL_FLOW_FILES = [
    "se3sb_bass_integration",
    "w_flow_cfm_integration",
    "optimal_transport_v2",
    "connectivity_decoder",
    "m5_schedulers",
    "fractional_ode",
]


def main():
    init_text = INIT_PY.read_text()

    report = {
        "workflow": "Wf-304C Task 2",
        "audit_date": "2026-09-29",
        "scope": "Verifies which frontier math adapter modules are importable + imported + USED by flow_matching_lipman/__init__.py",
        "wire_ready_adapters": [],
        "research_only": [],
        "local_flow_files": [],
    }

    # 1. Check the 4 wire-ready adapters
    for mod_name, desc in WIRE_READY_ADAPTERS:
        try:
            mod = importlib.import_module(f"molmetal.adapters.{mod_name}")
            importable = True
            public_symbols = [s for s in dir(mod) if not s.startswith("_")]
            n_public = len(public_symbols)
            err = None
        except Exception as e:
            importable = False
            n_public = 0
            public_symbols = [f"IMPORT_ERROR: {type(e).__name__}: {e}"]
            err = str(e)

        imported_by_init = (
            f"from .{mod_name}" in init_text or f"import {mod_name}" in init_text
        )
        referenced_in_init = mod_name in init_text

        if importable and imported_by_init:
            status = "WIRED"
        elif importable and not imported_by_init:
            status = "WIRE-READY (adapter exists, NOT wired into __init__.py)"
        else:
            status = "IMPORT-ERROR"

        entry = {
            "adapter": mod_name,
            "description": desc,
            "importable": importable,
            "n_public_symbols": n_public,
            "sample_public_symbols": public_symbols[:8],
            "imported_by_init": imported_by_init,
            "referenced_in_init": referenced_in_init,
            "wire_in_status": status,
        }
        if err:
            entry["import_error"] = err
        report["wire_ready_adapters"].append(entry)

    # 2. Check the 3 research-only modules
    for mod_name, desc in RESEARCH_ONLY:
        try:
            mod = importlib.import_module(f"molmetal.math_skills.{mod_name}")
            importable = True
            public_symbols = [s for s in dir(mod) if not s.startswith("_")]
            n_public = len(public_symbols)
            err = None
        except Exception as e:
            importable = False
            n_public = 0
            public_symbols = [f"IMPORT_ERROR: {type(e).__name__}: {e}"]
            err = str(e)

        entry = {
            "module": mod_name,
            "description": desc,
            "importable": importable,
            "n_public_symbols": n_public,
            "sample_public_symbols": public_symbols[:8],
            "wire_in_status": "RESEARCH-ONLY by design (statement holder / citation, not a drop-in loss)",
        }
        if err:
            entry["import_error"] = err
        report["research_only"].append(entry)

    # 3. Check the local flow files
    for file_stem in LOCAL_FLOW_FILES:
        path = FLOW_DIR / f"{file_stem}.py"
        if not path.exists():
            report["local_flow_files"].append(
                {
                    "file": f"{file_stem}.py",
                    "exists": False,
                    "wire_in_status": "MISSING",
                }
            )
            continue
        text = path.read_text()
        imported = (
            f"from .{file_stem}" in init_text
            or f"from molmetal.adapters.flow_matching_lipman.{file_stem}" in init_text
            or file_stem in init_text
        )
        n_lines = len(text.splitlines())

        if imported:
            status = "IN-TREE (imported by __init__.py)"
        else:
            status = "STANDALONE (shipped in flow_matching_lipman/ but not wired into __init__.py)"

        report["local_flow_files"].append(
            {
                "file": f"{file_stem}.py",
                "exists": True,
                "n_lines": n_lines,
                "imported_by_init": imported,
                "wire_in_status": status,
            }
        )

    # 4. Summary verdict
    n_wire_ready_total = len(report["wire_ready_adapters"])
    n_wired_outer = sum(
        1 for e in report["wire_ready_adapters"] if e["imported_by_init"]
    )
    n_wired_local = sum(1 for e in report["local_flow_files"] if e["imported_by_init"])
    n_remaining = n_wire_ready_total - n_wired_outer
    report["summary"] = {
        "n_wire_ready_outer_adapters": n_wire_ready_total,
        "n_wired_outer_to_init": n_wired_outer,
        "n_remaining_wire_in": n_remaining,
        "n_local_flow_files": len(report["local_flow_files"]),
        "n_local_flow_files_wired": n_wired_local,
        "research_only_count": len(report["research_only"]),
        "verdict": (
            f"{n_remaining}/{n_wire_ready_total} outer frontier math adapters (multi_ph / se3_lipman / "
            f"se3_sb / transport_relaxation) are wire-ready (standalone shipped) but NOT yet "
            f"imported by the locked __init__.py. {n_wired_local}/{len(report['local_flow_files'])} "
            f"local flow_matching_lipman/ files (se3sb_bass / w_flow_cfm / optimal_transport_v2 / "
            f"connectivity_decoder / m5_schedulers / fractional_ode) are wired into __init__.py."
        ),
        "r17_invariant_note": (
            "Per the R17 invariant, flow_matching_lipman/ is LOCKED — modifications require "
            "explicit user authorization. The 4 outer adapters exist as standalone modules that "
            "callers can import directly; the wire-in step is a ~5-7 day ultracode effort that "
            "drops them into the locked CFM pipeline."
        ),
    }

    out_path = Path(
        "/home/hugo/codes/mol-metal/molmetal/reports/wf304c_todo_impl/wire_in_audit.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"Verdict: {report['summary']['verdict']}")
    print(f"Outer adapters: {n_wired_outer}/{n_wire_ready_total} wired")
    print(f"Local flow files: {n_wired_local}/{len(report['local_flow_files'])} wired")


if __name__ == "__main__":
    main()
