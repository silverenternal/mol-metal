"""Wf-126 Phase 1 — Stage 1h36/3eml/3ny8/4aaw/4rlu full (not partial).

This is a refresh pass that materialises one Meeko-prepared
``receptor.pdbqt`` per pocket under
``molmetal/reports/wf126_p1_staging/<pocket>/`` so the Wf-123 / Wf-126
phase-1 RMSD re-bench has a fresh, audited staging directory to read
from. It reuses ``mk_prepare_receptor.py`` exactly as the WF-30 staging
pass did, but focuses on the five pockets that drive m15 RMSD and
records an honest per-pocket provenance block in
``staging_status.json``.

Honest framing
--------------
The TODO-123 plan §0 reads "per-pocket numbers all = 1.241 — artifact of
partial staging". That diagnosis is *misleading*: ``rigid_fragment_rmsd``
is a **self-RMSD between two ETKDGv3 + UFF-optimised conformers of the
same molecule**, so the value is naturally identical across pockets (the
pocket doesn't enter the calculation). The 1.241 ceiling across
1h36/3eml/3ny8/4aaw/4rlu is therefore the *expected* value, not an
artifact of partial staging. The 5 pockets in question are already
READY under ``molmetal/reports/wf30_pocket_prep/<pocket>/`` (proven by
``WF-30 staging`` PASS). This script still re-stages them under
``wf126_p1_staging/`` to:

* give the Wf-126 close-out a self-contained directory,
* capture the audit trail (sha256, size, returncode, command) for the
  next reviewer,
* allow `--verify-only` to re-check that the on-disk artefacts still
  hash to the recorded values.

Usage
-----
    source env.sh            # wires UV_PROJECT_ENVIRONMENT + VIRTUAL_ENV
    uv run python -m molmetal.scripts.stage_5_pockets_full
    uv run python -m molmetal.scripts.stage_5_pockets_full --pocket 1h36
    uv run python -m molmetal.scripts.stage_5_pockets_full --skip-existing
    uv run python -m molmetal.scripts.stage_5_pockets_full --verify-only

Outputs
-------
* ``molmetal/reports/wf126_p1_staging/<pocket>/receptor.pdbqt``
* ``molmetal/reports/wf126_p1_staging/<pocket>/prep_meta.json``
* ``molmetal/reports/wf126_p1_staging/staging_status.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf126_p1_staging"
)
MOLMETAL_DATA_DIR = Path(
    __import__("os").environ.get(
        "MOLMETAL_DATA_DIR",
        str(PROJECT_ROOT / ".data" / "molmetal"),
    )
)


@dataclass(frozen=True)
class PocketSpec:
    """One pocket's source + destination."""

    pocket_id: str
    source_pdb: Optional[Path]
    metric_path: Optional[Path]
    notes: str = ""
    allow_bad_res: bool = True
    default_altloc: str = "A"
    set_template: str = ""


# The five pockets that drive m15 rigid_fragment_rmsd per-pocket
# numbers. Order matches TODO-123 §0 listing.
POCKETS: List[PocketSpec] = [
    PocketSpec(
        pocket_id="1h36",
        source_pdb=(
            MOLMETAL_DATA_DIR
            / "crossdocked"
            / "SQHC_ALIAD_1_631_0"
            / "1h36_A_rec_1h35_r01_lig_tt_min_0_pocket10.pdb"
        ),
        metric_path=(
            PROJECT_ROOT
            / "metrics"
            / "by_metric"
            / "vina_score_1h36.json"
        ),
        notes=(
            "SQHC_ALIAD (squalene-hopene cyclase). CrossDocked pocket10 "
            "cut. Already READY under wf30_pocket_prep/1h36/."
        ),
    ),
    PocketSpec(
        pocket_id="3eml",
        source_pdb=(
            MOLMETAL_DATA_DIR
            / "crossdocked"
            / "AA2AR_HUMAN_1_315_TM_0"
            / "3eml_A_rec_3eml_zma_lig_tt_min_0_pocket10.pdb"
        ),
        metric_path=(
            PROJECT_ROOT
            / "metrics"
            / "by_metric"
            / "vina_score_3eml.json"
        ),
        notes=(
            "AA2AR_HUMAN (A2A adenosine receptor TM). CrossDocked "
            "pocket10 cut. Already READY under wf30_pocket_prep/3eml/."
        ),
    ),
    PocketSpec(
        pocket_id="3ny8",
        source_pdb=(
            MOLMETAL_DATA_DIR
            / "crossdocked"
            / "ADRB2_HUMAN_26_357_TM_0"
            / "3ny8_A_rec_3d4s_tim_lig_tt_min_0_pocket10.pdb"
        ),
        metric_path=(
            PROJECT_ROOT
            / "metrics"
            / "by_metric"
            / "vina_score_3ny8.json"
        ),
        notes=(
            "ADRB2_HUMAN (β2 adrenergic receptor TM). CrossDocked "
            "pocket10 cut. Already READY under wf30_pocket_prep/3ny8/."
        ),
    ),
    PocketSpec(
        pocket_id="4aaw",
        source_pdb=(
            MOLMETAL_DATA_DIR
            / "crossdocked"
            / "GLMU_STRPN_2_459_0"
            / "4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb"
        ),
        metric_path=(
            PROJECT_ROOT
            / "metrics"
            / "by_metric"
            / "vina_score_4aaw.json"
        ),
        notes=(
            "GLMU_STRPN (glutamate mutarotase, R84 ligand). CrossDocked "
            "pocket10 cut. Already READY under wf30_pocket_prep/4aaw/."
        ),
    ),
    PocketSpec(
        pocket_id="4rlu",
        source_pdb=Path(
            "/home/hugo/codes/pocket/data/pdbbindpp-2020/extracted/"
            "pbpp-2020/4rlu/4rlu_protein.pdb"
        ),
        metric_path=(
            PROJECT_ROOT
            / "metrics"
            / "by_metric"
            / "vina_score_4rlu.json"
        ),
        notes=(
            "Trypsin HCC inhibitor. pdbbindpp-2020 full protein (not "
            "the pocket cut). 12 histidines forced to HIE. Already READY "
            "under wf30_pocket_prep/4rlu/."
        ),
        allow_bad_res=True,
        set_template=(
            "A:11=HIE,A:17=HIE,A:74=HIE,A:116=HIE,"
            "B:41=HIE,B:58=HIE,"
            "C:11=HIE,C:17=HIE,C:74=HIE,C:116=HIE,"
            "D:41=HIE,D:58=HIE"
        ),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_mk_prepare_receptor() -> str:
    """Locate ``mk_prepare_receptor.py`` on PATH; abort with a clear error."""
    found = shutil.which("mk_prepare_receptor.py")
    if not found:
        raise FileNotFoundError(
            "mk_prepare_receptor.py not on PATH; source env.sh so that "
            "uv routes to the shared CadButEaas venv that ships meeko."
        )
    return found


def _read_existing_wf30_status(pocket_id: str) -> Optional[Dict[str, object]]:
    """Best-effort read of the corresponding wf30_pocket_prep record."""
    status_path = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf30_pocket_prep"
        / "pocket_status.json"
    )
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return None
    for entry in payload.get("pockets", []):
        if entry.get("pocket_id") == pocket_id:
            return entry
    return None


def _verify_pdbqt(spec: PocketSpec, out_dir: Path) -> Dict[str, object]:
    """Re-hash the on-disk receptor.pdbqt and compare to the recorded one."""
    pdbqt_path = out_dir / "receptor.pdbqt"
    meta_path = out_dir / "prep_meta.json"
    record: Dict[str, object] = {
        "pocket_id": spec.pocket_id,
        "status": "FAILED",
        "error": None,
        "verified_sha_match": None,
    }
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        record["error"] = "receptor.pdbqt missing or empty"
        return record
    if not meta_path.exists():
        record["error"] = "prep_meta.json missing"
        return record
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        record["error"] = f"prep_meta.json unreadable: {exc}"
        return record
    on_disk_sha = _sha256(pdbqt_path)
    record["pdbqt_size_bytes"] = pdbqt_path.stat().st_size
    record["pdbqt_sha256"] = on_disk_sha
    record["meta_pdbqt_sha256"] = meta.get("pdbqt_sha256")
    record["meta_pdbqt_size_bytes"] = meta.get("pdbqt_size_bytes")
    if on_disk_sha == meta.get("pdbqt_sha256"):
        record["verified_sha_match"] = True
        record["status"] = "READY"
    else:
        record["verified_sha_match"] = False
        record["error"] = "pdbqt sha256 mismatch with prep_meta.json"
    return record


def prepare_one(
    spec: PocketSpec,
    out_dir: Path,
    skip_existing: bool,
    timeout_s: int = 120,
) -> Dict[str, object]:
    """Prepare a single pocket receptor; returns the per-pocket record."""
    started = time.monotonic()
    record: Dict[str, object] = {
        "pocket_id": spec.pocket_id,
        "requested_source": str(spec.source_pdb) if spec.source_pdb else None,
        "out_dir": str(out_dir),
        "pdbqt_path": None,
        "pdbqt_size_bytes": None,
        "pdbqt_sha256": None,
        "source_pdb": None,
        "source_pdb_sha256": None,
        "source_pdb_size_bytes": None,
        "command": None,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "elapsed_seconds": None,
        "status": "FAILED",
        "error": None,
        "notes": spec.notes,
        "wf30_audit": _read_existing_wf30_status(spec.pocket_id),
    }
    pdbqt_path = out_dir / "receptor.pdbqt"
    if (
        skip_existing
        and pdbqt_path.exists()
        and pdbqt_path.stat().st_size > 0
    ):
        record.update(
            status="SKIPPED",
            pdbqt_path=str(pdbqt_path),
            pdbqt_size_bytes=pdbqt_path.stat().st_size,
            pdbqt_sha256=_sha256(pdbqt_path),
            source_pdb=str(spec.source_pdb),
            elapsed_seconds=0.0,
        )
        if spec.source_pdb is not None and spec.source_pdb.exists():
            record["source_pdb_sha256"] = _sha256(spec.source_pdb)
            record["source_pdb_size_bytes"] = spec.source_pdb.stat().st_size
        return record
    if spec.source_pdb is None or not spec.source_pdb.exists():
        record["error"] = (
            f"source PDB missing on disk: {spec.source_pdb}"
        )
        return record
    record["source_pdb"] = str(spec.source_pdb)
    record["source_pdb_sha256"] = _sha256(spec.source_pdb)
    record["source_pdb_size_bytes"] = spec.source_pdb.stat().st_size
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "receptor"
    mk_script = _find_mk_prepare_receptor()
    cmd = [
        sys.executable,
        mk_script,
        "--read_pdb",
        str(spec.source_pdb),
        "-o",
        str(base),
        "-p",
        "-j",
        "--compute_charges",
        "--charge_model",
        "gasteiger",
        "--default_altloc",
        spec.default_altloc,
    ]
    if spec.allow_bad_res:
        cmd.append("--allow_bad_res")
    if spec.set_template:
        cmd.extend(["--set_template", spec.set_template])
    record["command"] = cmd
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        record["error"] = f"TimeoutExpired after {timeout_s}s"
        record["stderr_tail"] = (exc.stderr or "")[-1500:]
        record["elapsed_seconds"] = time.monotonic() - started
        return record
    record["returncode"] = proc.returncode
    record["stdout_tail"] = proc.stdout[-1500:]
    record["stderr_tail"] = proc.stderr[-1500:]
    if proc.returncode != 0:
        record["error"] = (
            f"mk_prepare_receptor.py exit={proc.returncode}; "
            "see stderr_tail for the template-match failure."
        )
        record["elapsed_seconds"] = time.monotonic() - started
        return record
    if (
        not pdbqt_path.exists()
        or pdbqt_path.stat().st_size == 0
    ):
        record["error"] = (
            "mk_prepare_receptor.py reported success but receptor.pdbqt "
            "is missing or empty."
        )
        record["elapsed_seconds"] = time.monotonic() - started
        return record
    record["pdbqt_path"] = str(pdbqt_path)
    record["pdbqt_size_bytes"] = pdbqt_path.stat().st_size
    record["pdbqt_sha256"] = _sha256(pdbqt_path)
    record["status"] = "READY"
    record["elapsed_seconds"] = time.monotonic() - started
    meta_path = out_dir / "prep_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "pocket_id": spec.pocket_id,
                "source_pdb": str(spec.source_pdb),
                "source_pdb_sha256": record["source_pdb_sha256"],
                "pdbqt_path": str(pdbqt_path),
                "pdbqt_sha256": record["pdbqt_sha256"],
                "pdbqt_size_bytes": record["pdbqt_size_bytes"],
                "command": cmd,
                "returncode": proc.returncode,
                "elapsed_seconds": record["elapsed_seconds"],
                "notes": spec.notes,
            },
            indent=2,
        )
        + "\n"
    )
    return record


def _summary(records: List[Dict[str, object]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "FAILED")
        summary[status] = summary.get(status, 0) + 1
    return summary


def _load_metric_value(spec: PocketSpec) -> Optional[Dict[str, object]]:
    """Best-effort lookup of the measured-pocket metric JSON."""
    if spec.metric_path is None or not spec.metric_path.exists():
        return None
    try:
        payload = json.loads(spec.metric_path.read_text())
    except json.JSONDecodeError:
        return None
    block = payload.get("value") or {}
    pocket_block = payload.get("pocket") or {}
    return {
        "metric_file": str(
            spec.metric_path.relative_to(PROJECT_ROOT)
        ),
        "metric_status": payload.get("status"),
        "mean": block.get("mean"),
        "unit": block.get("unit") or payload.get("unit"),
        "n_samples": payload.get("n_samples"),
        "engine": pocket_block.get("engine"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root for receptor.pdbqt files (default: %(default)s)",
    )
    parser.add_argument(
        "--pocket",
        action="append",
        default=None,
        help="Restrict to one or more pocket_id values (default: 5)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse any receptor.pdbqt already on disk; do not re-prepare",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip preparation; re-hash and validate on-disk artefacts",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=120,
        help="Per-pocket subprocess timeout (default: %(default)s)",
    )
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = POCKETS
    if args.pocket:
        wanted = set(args.pocket)
        selected = [spec for spec in POCKETS if spec.pocket_id in wanted]
        if not selected:
            parser.error(
                f"--pocket filter matched no pockets; valid ids: "
                f"{[spec.pocket_id for spec in POCKETS]}"
            )
    records: List[Dict[str, object]] = []
    for spec in selected:
        per_pocket_dir = args.out_dir / spec.pocket_id
        if args.verify_only:
            record = _verify_pdbqt(spec, per_pocket_dir)
            record["notes"] = spec.notes
            record["wf30_audit"] = _read_existing_wf30_status(spec.pocket_id)
        else:
            record = prepare_one(
                spec=spec,
                out_dir=per_pocket_dir,
                skip_existing=args.skip_existing,
                timeout_s=args.timeout_s,
            )
        record["metric_provenance"] = _load_metric_value(spec)
        records.append(record)
        status = record.get("status")
        print(
            f"{spec.pocket_id}: {status} "
            f"(pdbqt={record.get('pdbqt_size_bytes') or 'n/a'} bytes, "
            f"{float(record.get('elapsed_seconds') or 0.0):.1f}s)",
            flush=True,
        )
    summary = _summary(records)
    status_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "workflow": "wf126_p1",
        "out_dir": str(args.out_dir),
        "molmetal_data_dir": str(MOLMETAL_DATA_DIR),
        "verify_only": bool(args.verify_only),
        "summary": summary,
        "ready_count": summary.get("READY", 0),
        "expected_count": len(POCKETS),
        "honest_framing": {
            "diagnosis_in_todo_123": (
                "TODO-123 §0 attributes the uniform per-pocket "
                "rigid_fragment_rmsd = 1.241 to 'partial staging'. This "
                "is misleading: rigid_fragment_rmsd is a self-RMSD "
                "between two ETKDGv3 + UFF-optimised conformers of the "
                "same molecule (see metric description). The pocket "
                "does not enter the calculation, so identical 1.241 "
                "across pockets is the EXPECTED behaviour, not an "
                "artifact of partial staging. All 5 pockets are already "
                "READY under molmetal/reports/wf30_pocket_prep/."
            ),
            "what_this_pass_does": (
                "Re-stages 1h36/3eml/3ny8/4aaw/4rlu under "
                "molmetal/reports/wf126_p1_staging/ to give Wf-126 a "
                "self-contained, freshly-audited staging directory. "
                "Does NOT dock and does NOT mutate source PDBs."
            ),
            "stage_recipe": (
                "mk_prepare_receptor.py -p -j --compute_charges "
                "--charge_model gasteiger --default_altloc A "
                "--allow_bad_res (4rlu additionally forces 12 histidines "
                "to HIE via --set_template)."
            ),
        },
        "pockets": records,
    }
    status_path = args.out_dir / "staging_status.json"
    status_path.write_text(json.dumps(status_payload, indent=2) + "\n")
    print(json.dumps({"summary": summary, "status_json": str(status_path)}))
    expected = len(selected) if not args.verify_only else 0
    failed = summary.get("FAILED", 0)
    if args.verify_only:
        return 0 if failed == 0 else 1
    return 0 if summary.get("READY", 0) == expected and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())