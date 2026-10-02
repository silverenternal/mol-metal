#!/usr/bin/env python3
"""
Wf-313A relink helper — REORGANIZE (NOT re-extract) Phase 1 reference ligand data.

This script takes the MEASURED Phase 1 data shipped via Wf-139 P1
(`molmetal/data/pocket_ref_ligands/ligand_metadata.json`) and reorganizes
it from a flat layout:

    molmetal/data/pocket_ref_ligands/<pocket>_ligand.smi

into a per-pocket subdirectory layout that better supports Phase 2
redocking outputs (per-pocket RMSD reports, per-pocket PDBQT copies,
per-pocket InChI metadata sidecar):

    molmetal/data/m15_redesign_ref/pocket_<n>/ref_ligand.smi
    molmetal/data/m15_redesign_ref/pocket_<n>/ref_ligand_inchi.json

Honest framing per party-twenty-spirit §3.5:
- This is a REFACTOR, NOT a new extraction. Zero new SMILES are generated.
- All data is sourced verbatim from Wf-139 P1 MEASURED outputs.
- The extraction script + tests + InChIKeys remain unchanged.

Outputs:
1. molmetal/data/m15_redesign_ref/pocket_<pocket>/ref_ligand.smi
   - TSV 3-column: SMILES \\t InChIKey \\t pocket_id
2. molmetal/data/m15_redesign_ref/pocket_<pocket>/ref_ligand_inchi.json
   - Per-pocket InChIKey + canonical SMILES + formula + heavy_atom_count
3. molmetal/reports/wf313a_m15_redesign_phase1/extraction_summary.json
   - Aggregated Phase 1 + Wf-313A relink status with source_file/line citations

Verify:
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/wf313a_relink_pocket_ref_ligands.py
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf313a_m15_redesign_phase1.py -v
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Atomic write protocol — closes the "half-merged file state" failure mode
# (concurrent writers racing on the rename) and the "direct overwrite
# without risk check" failure mode (closeout-drafting scripts silently
# overwriting prior closeouts).  See `wf386a3_atomic_write.py` §0 +
# AGENTS.md §5.6 adoption rule.
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
    detect_truncation,
)

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
SOURCE_DIR = PROJECT_ROOT / "molmetal" / "data" / "pocket_ref_ligands"
SOURCE_METADATA = SOURCE_DIR / "ligand_metadata.json"
TARGET_DIR = PROJECT_ROOT / "molmetal" / "data" / "m15_redesign_ref"
REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf313a_m15_redesign_phase1"

# Wf-139 P1 closeout: TODO/completed/139_wf139_p5_cfm_decode_real.md
# Actually the Phase 1 reference ligands shipped via Wf-139 P1 in:
# - molmetal/data/pocket_ref_ligands/ligand_metadata.json
# - molmetal/reports/wf139_p1_ref_ligands/synthesis.md
WF139_P1_SYNTHESIS_MD = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf139_p1_ref_ligands" / "synthesis.md"
)
TODO_134 = PROJECT_ROOT / "TODO" / "completed" / "134_m15_docking_rmsd_redesign.md"


def relink_one_pocket(pocket: dict[str, Any]) -> dict[str, Any]:
    """Relink a single pocket's data from flat to per-pocket subdir layout."""
    pocket_id = pocket["pocket_id"]
    target_pocket_dir = TARGET_DIR / f"pocket_{pocket_id}"
    target_pocket_dir.mkdir(parents=True, exist_ok=True)

    smiles = pocket["smiles"]
    inchikey = pocket["inchikey"]

    # ref_ligand.smi (TSV 3-column: SMILES, InChIKey, pocket_id)
    smi_path = target_pocket_dir / "ref_ligand.smi"
    # Wf-410-A3 migration: atomic write + safe_overwrite_check so the
    # ref_ligand.smi survives concurrent relink runs and surfaces any
    # substantial-content overwrite before it happens.
    _smi_content = f"{smiles}\t{inchikey}\t{pocket_id}\n"
    safe_overwrite_check(smi_path, _smi_content)
    atomic_write_file(smi_path, _smi_content, encoding="utf-8")

    # ref_ligand_inchi.json (InChIKey sidecar)
    inchi_json = {
        "pocket_id": pocket_id,
        "smiles": smiles,
        "inchikey": inchikey,
        "formula": pocket.get("formula"),
        "atom_count": pocket.get("atom_count"),
        "heavy_atom_count": pocket.get("heavy_atom_count"),
        "ligand_resname": pocket.get("ligand_resname"),
        "extraction_kind": pocket.get("extraction_kind"),
        "source_path": pocket.get("source_path"),
        "source_sha256": pocket.get("source_sha256"),
        "ok": pocket.get("ok", True),
        "notes": pocket.get("notes", ""),
        # Provenance: cite Wf-139 P1 as the MEASURED source
        "source_wf": "Wf-139 P1",
        "source_metadata_json": str(SOURCE_METADATA.relative_to(PROJECT_ROOT)),
        "source_synthesis_md": str(
            WF139_P1_SYNTHESIS_MD.relative_to(PROJECT_ROOT)
        ),
        "wf313a_relink_utc": datetime.now(timezone.utc).isoformat(),
        "wf313a_relink_kind": "REFACTOR_FROM_FLAT_LAYOUT",
        # Phase 2 honest-negative marker
        "phase2_redocking_status": "BLOCKED",
        "phase2_recovery_path": (
            "conda-forge vina install OR MOLMETAL_VINA_BIN env override "
            "per Wf-84 P2 stub_warning (Wf-309A discovery: which vina → "
            "not found)"
        ),
    }
    inchi_path = target_pocket_dir / "ref_ligand_inchi.json"
    # Wf-410-A3 migration: atomic write + safe_overwrite_check so the
    # inchi sidecar JSON survives concurrent relink runs.
    _inchi_payload = json.dumps(inchi_json, indent=2, sort_keys=True) + "\n"
    safe_overwrite_check(inchi_path, _inchi_payload)
    atomic_write_file(inchi_path, _inchi_payload, encoding="utf-8")

    return {
        "pocket_id": pocket_id,
        "ok": pocket.get("ok", False),
        "smiles_path": str(smi_path.relative_to(PROJECT_ROOT)),
        "inchi_path": str(inchi_path.relative_to(PROJECT_ROOT)),
        "smiles_len_chars": len(smiles),
        "inchikey": inchikey,
        "heavy_atom_count": pocket.get("heavy_atom_count"),
    }


def write_extraction_summary(
    per_pocket: list[dict[str, Any]],
    source_metadata: dict[str, Any],
) -> Path:
    """Write the Wf-313A extraction_summary.json."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_pockets = len(per_pocket)
    n_ok = sum(1 for p in per_pocket if p["ok"])
    n_inchi_consistent = len({p["inchikey"] for p in per_pocket if p["ok"]})

    summary = {
        # Provenance
        "wf_id": "Wf-313A",
        "wf_kind": "RELINK_FROM_WF139_P1_MEASURED_DATA",
        "wf_purpose": (
            "Reorganize Wf-139 P1 MEASURED Phase 1 reference ligand "
            "SMILES/InChIKey data from a flat layout "
            "(`molmetal/data/pocket_ref_ligands/<pocket>_ligand.smi`) into a "
            "per-pocket subdirectory layout "
            "(`molmetal/data/m15_redesign_ref/pocket_<n>/`) that supports "
            "Phase 2 Vina redocking outputs."
        ),
        "source_wf": "Wf-139 P1",
        "source_metadata_json": str(SOURCE_METADATA.relative_to(PROJECT_ROOT)),
        "source_synthesis_md": str(
            WF139_P1_SYNTHESIS_MD.relative_to(PROJECT_ROOT)
        ),
        "todo134_status_reference": str(TODO_134.relative_to(PROJECT_ROOT)),
        # Honest-framing marker: this is a refactor, not new extraction
        "refactor_only": True,
        "no_new_smiles_generated": True,
        # Counts (all MEASURED via Wf-139 P1, just relinked)
        "n_pockets_extracted": n_pockets,
        "n_smiles_valid": n_ok,
        "n_smiles_failed": n_pockets - n_ok,
        "n_inchi_consistent": n_inchi_consistent,  # unique InChIKeys per pocket
        "n_inchi_collisions": n_ok - n_inchi_consistent,
        # Per-pocket entries
        "per_pocket": per_pocket,
        # Phase 2 honest-negative close
        "phase2_status": "BLOCKED",
        "phase2_reason": (
            "Vina 1.2.7 not installed (Wf-309A discovery: `which vina` → "
            "not found). Phase 2 redocking bench requires Vina python "
            "binding or CLI."
        ),
        "phase2_recovery_path": (
            "Option A: `conda install -c conda-forge vina` (~10 min wall). "
            "Option B: Set `MOLMETAL_VINA_BIN=/path/to/vina` env override "
            "per Wf-84 P2 stub_warning. "
            "Then re-run `molmetal/scripts/bench_docked_pose_rmsd.py` per "
            "TODO-134 §2 Phase 2."
        ),
        # Pre-flight status (caller should re-verify after run)
        "preflight_status": "PENDING_VERIFICATION_AT_CLOSE",
        # Generated
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = REPORT_DIR / "extraction_summary.json"
    # Wf-410-A3 migration: atomic write + safe_overwrite_check so the
    # extraction_summary.json survives concurrent relink runs and surfaces
    # any substantial-content overwrite before it happens.
    _summary_payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    safe_overwrite_check(out_path, _summary_payload)
    atomic_write_file(out_path, _summary_payload, encoding="utf-8")
    return out_path


def main() -> int:
    if not SOURCE_METADATA.exists():
        print(f"FAIL: source metadata not found: {SOURCE_METADATA}", file=sys.stderr)
        return 1
    if not WF139_P1_SYNTHESIS_MD.exists():
        print(
            f"FAIL: Wf-139 P1 synthesis.md not found: {WF139_P1_SYNTHESIS_MD}",
            file=sys.stderr,
        )
        return 1

    source_metadata = json.loads(SOURCE_METADATA.read_text())
    pockets = source_metadata["pockets"]

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    per_pocket = [relink_one_pocket(p) for p in pockets]
    summary_path = write_extraction_summary(per_pocket, source_metadata)

    # Stdout invariant per AGENTS.md §4
    n_pockets = len(per_pocket)
    n_ok = sum(1 for p in per_pocket if p["ok"])
    n_unique_inchi = len({p["inchikey"] for p in per_pocket if p["ok"]})
    print("PASS")
    print(f"  wf_id: Wf-313A (RELINK from Wf-139 P1 MEASURED)")
    print(f"  n_pockets_extracted: {n_pockets}")
    print(f"  n_smiles_valid: {n_ok}")
    print(f"  n_inchi_consistent: {n_unique_inchi} (unique InChIKeys)")
    print(f"  out_dir: {TARGET_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"  phase2_redocking_status: BLOCKED (Vina 1.2.7 not installed)")
    print(f"  preflight_status: PENDING_VERIFICATION_AT_CLOSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
