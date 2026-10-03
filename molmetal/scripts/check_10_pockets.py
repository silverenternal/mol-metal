"""Wf-30 Phase 1 — Stage 10 CrossDocked pocket receptor PDBQT files.

This is a staging-only pass: for every target pocket we materialise one
Meeko-prepared ``receptor.pdbqt`` (static rigid receptor) under
``molmetal/reports/wf30_pocket_prep/<pocket>/`` so downstream B2-B5
workflows can dock against the same receptor file.

The list of 10 pockets is the union of:

* the 6 metric-measured pockets in ``metrics/by_metric/vina_score*.json``
  (1h36, 3eml, 3ny8, 4aaw, 4rlu, 830c)
* the 4 additional pockets named in the WF-30 task brief
  (1g9v, 2hnx, 4rn0, 1fmc)

We pick 4rn0 (HDAC8) and 1fmc (7-alpha-HSDH) because they are both
already named in ``molmetal/data/crossdocked100_manifest.csv`` (rows for
test_005 and test_006), they share the strict-preparation pathway used
in ``molmetal/scripts/prepare_crossdocked_receptor.py``, and their
existing receptor PDB is reproducible from the local
``MOLMETAL_DATA_DIR/crossdocked`` symlink tree. ``1bc8``, ``3f7r``,
``4de2``, ``4gmt`` and ``4jsz`` are not staged on this host, so the
candidate-shortlist reduces to ``1g9v``, ``2hnx`` plus two manifest entries.

Usage
-----
    source env.sh            # wires UV_PROJECT_ENVIRONMENT + VIRTUAL_ENV
    uv run python -m molmetal.scripts.check_10_pockets
    uv run python -m molmetal.scripts.check_10_pockets --pocket 830c
    uv run python -m molmetal.scripts.check_10_pockets --skip-existing

Outputs
-------
* ``molmetal/reports/wf30_pocket_prep/pocket_status.json``
* ``molmetal/reports/wf30_pocket_prep/pocket_prep_log.md``
* ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``
* ``molmetal/reports/wf30_pocket_prep/<pocket>/prep_meta.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
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
    source_pdb: Optional[Path] = None
    metric_path: Optional[Path] = None
    notes: str = ""
    allow_bad_res: bool = True
    default_altloc: str = "A"
    set_template: str = ""
    delete_residues: str = ""


def _candidate_paths(pocket_id: str) -> List[Path]:
    """Return possible on-disk paths for ``pocket_id`` in priority order.

    The chain letter is encoded in the filename and varies between targets
    (``_A_rec_*`` for most, ``_B_rec_*`` for 4rn0/1fmc); we therefore try
    ``_A_rec_*`` first and fall back to ``_B_rec_*`` so the same pocket_id
    works for any chain.
    """
    candidates: List[Path] = []
    if pocket_id == "830c":
        candidates.append(
            PROJECT_ROOT / "molmetal" / "data" / "mmp13_real" / "830c.pdb"
        )
    elif pocket_id == "4rlu":
        candidates.append(
            Path(
                "/home/hugo/codes/pocket/data/pdbbindpp-2020/extracted/"
                "pbpp-2020/4rlu/4rlu_protein.pdb"
            )
        )
    else:
        crossdocked_root = MOLMETAL_DATA_DIR / "crossdocked"
        if crossdocked_root.exists():
            for chain in ("A", "B"):
                # Prefer the ``_min_0`` minimised pose, then any
                # ``_pocket10.pdb`` for the same chain. If both empty,
                # try the next chain.
                pattern_min = (
                    f"{pocket_id}_{chain}_rec_*_min_0_pocket10.pdb"
                )
                hits = sorted(crossdocked_root.glob(f"*/{pattern_min}"))
                if hits:
                    candidates.append(hits[0])
                    break
                pattern_any = (
                    f"{pocket_id}_{chain}_rec_*_pocket10.pdb"
                )
                hits = sorted(crossdocked_root.glob(f"*/{pattern_any}"))
                if hits:
                    candidates.append(hits[0])
                    break
    return [path for path in candidates if path.exists()]


# Pockets to stage. ``metric_path`` is informational only — the JSON
# points to /mnt/disk1 which is not mounted on this host, so we cannot
# hash the source from there; the staging tree under
# ``MOLMETAL_DATA_DIR/crossdocked/`` is the live source of truth.
POCKETS: List[PocketSpec] = [
    PocketSpec(
        pocket_id="830c",
        source_pdb=Path(
            "/home/hugo/codes/mol-metal/molmetal/data/mmp13_real/830c.pdb"
        ),
        metric_path=PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / "vina_score.json",
        notes=(
            "MMP-13 hydroxamate (sulphone hydroxamate RS1). Single-pocket "
            "anchor for WF-30. Already docked successfully in R10 phase-1."
        ),
    ),
    PocketSpec(
        pocket_id="1h36",
        metric_path=PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / "vina_score_1h36.json",
        notes="SQHC_ALIAD (squalene-hopene cyclase) — CrossDocked test_022.",
    ),
    PocketSpec(
        pocket_id="3eml",
        metric_path=PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / "vina_score_3eml.json",
        notes="AA2AR_HUMAN (A2A adenosine receptor TM) — GPCR pocket.",
    ),
    PocketSpec(
        pocket_id="3ny8",
        metric_path=PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / "vina_score_3ny8.json",
        notes="ADRB2_HUMAN (β2 adrenergic receptor TM) — GPCR pocket.",
    ),
    PocketSpec(
        pocket_id="4aaw",
        metric_path=PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / "vina_score_4aaw.json",
        notes=(
            "GLMU_STRPN test_001 (glutamate mutarotase, R84 ligand). "
            "CrossDocked standard test entry."
        ),
    ),
    PocketSpec(
        pocket_id="4rlu",
        metric_path=PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / "vina_score_4rlu.json",
        notes=(
            "Trypsin HCC inhibitor (HCC pocket). Sourced from "
            "pdbbindpp-2020 (full receptor protein, not the pocket cut). "
            "12 histidines in the tetramer — all forced to HIE because "
            "Meeko cannot break the HIE/HID/HIP tie on its own."
        ),
        allow_bad_res=True,
        set_template=(
            "A:11=HIE,A:17=HIE,A:74=HIE,A:116=HIE,"
            "B:41=HIE,B:58=HIE,"
            "C:11=HIE,C:17=HIE,C:74=HIE,C:116=HIE,"
            "D:41=HIE,D:58=HIE"
        ),
    ),
    PocketSpec(
        pocket_id="1g9v",
        notes=(
            "HBA_HUMAN (haemoglobin subunit α). CrossDocked pocket10 cut. "
            "Selected because 1g9v appears in the task brief candidate "
            "short-list and is staged."
        ),
    ),
    PocketSpec(
        pocket_id="2hnx",
        notes=(
            "FABP4_HUMAN (fatty-acid binding protein 4). CrossDocked "
            "pocket10 cut. Selected because 2hnx appears in the task "
            "brief candidate short-list and is staged."
        ),
    ),
    PocketSpec(
        pocket_id="4rn0",
        notes=(
            "HDAC8_HUMAN (histone deacetylase 8). CrossDocked test_005. "
            "Selected as a metalloprotein-relevant pocket — HDAC8 has a "
            "Zn2+ active site."
        ),
    ),
    PocketSpec(
        pocket_id="1fmc",
        notes=(
            "HDHA_ECOLI (7-α-hydroxysteroid dehydrogenase). CrossDocked "
            "test_006. Already partly prepared as "
            "molmetal/reports/crossdocked_first10_resolved/test_006/"
            "receptor_source_names.pdbqt; this run materialises a fresh "
            "strict-mode pdbqt under the WF-30 dir."
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
            "uv routes to the project-local .venv that ships meeko."
        )
    return found


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
    }
    pdbqt_path = out_dir / "receptor.pdbqt"
    if skip_existing and pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
        record.update(
            status="SKIPPED",
            pdbqt_path=str(pdbqt_path),
            pdbqt_size_bytes=pdbqt_path.stat().st_size,
            pdbqt_sha256=_sha256(pdbqt_path),
            source_pdb=str(spec.source_pdb),
            elapsed_seconds=0.0,
        )
        return record
    if not spec.source_pdb:
        candidates = _candidate_paths(spec.pocket_id)
        if not candidates:
            record["error"] = (
                f"No staged PDB found for {spec.pocket_id}; searched "
                f"MOLMETAL_DATA_DIR={MOLMETAL_DATA_DIR} and project-local "
                "fallbacks."
            )
            return record
        spec = PocketSpec(
            pocket_id=spec.pocket_id,
            source_pdb=candidates[0],
            metric_path=spec.metric_path,
            notes=spec.notes,
            allow_bad_res=spec.allow_bad_res,
            default_altloc=spec.default_altloc,
            set_template=spec.set_template,
            delete_residues=spec.delete_residues,
        )
    if not spec.source_pdb.exists():
        record["error"] = f"source PDB missing on disk: {spec.source_pdb}"
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
        "-p",  # write pdbqt
        "-j",  # protonate
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
    if spec.delete_residues:
        cmd.extend(["--delete_residues", spec.delete_residues])
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
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        record["error"] = "mk_prepare_receptor.py reported success but receptor.pdbqt is missing or empty."
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


def _load_metrics_provenance() -> Dict[str, Dict[str, object]]:
    """Best-effort lookup of measured-pocket metadata for the prep log."""
    provenance: Dict[str, Dict[str, object]] = {}
    for spec in POCKETS:
        if spec.metric_path and spec.metric_path.exists():
            try:
                payload = json.loads(spec.metric_path.read_text())
                pocket_block = payload.get("pocket") or {}
                provenance[spec.pocket_id] = {
                    "metric_file": str(spec.metric_path.relative_to(PROJECT_ROOT)),
                    "metric_status": payload.get("status"),
                    "mean_kcal_mol": (payload.get("value") or {}).get("mean"),
                    "engine": pocket_block.get("engine"),
                    "exhaustiveness": pocket_block.get("exhaustiveness"),
                    "n_samples": payload.get("n_samples"),
                }
            except (OSError, json.JSONDecodeError):
                provenance[spec.pocket_id] = {"metric_file": str(spec.metric_path)}
    return provenance


def _summary(records: List[Dict[str, object]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "FAILED")
        summary[status] = summary.get(status, 0) + 1
    return summary


def _render_markdown(
    records: List[Dict[str, object]],
    provenance: Dict[str, Dict[str, object]],
    summary: Dict[str, int],
) -> str:
    lines: List[str] = []
    lines.append("# WF-30 Phase 1 — 10 CrossDocked pocket receptor prep log")
    lines.append("")
    lines.append(
        f"_Generated_: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append("")
    lines.append(
        "Per-pocket `mk_prepare_receptor.py` runs that materialise one "
        "static-receptor `receptor.pdbqt` per pocket under "
        "`molmetal/reports/wf30_pocket_prep/<pocket>/`. The script is "
        "staging-only: it does **not** dock and does **not** mutate the "
        "CrossDocked source PDBs."
    )
    lines.append("")
    lines.append("## Status summary")
    lines.append("")
    for status, count in sorted(summary.items()):
        lines.append(f"- **{status}**: {count}")
    lines.append("")
    lines.append(
        "Counts above are computed against the WF-30 list of 10 pockets. "
        "READY means a non-empty `receptor.pdbqt` is on disk; SKIPPED "
        "means `receptor.pdbqt` already existed and `--skip-existing` was "
        "passed; FAILED is documented below."
    )
    lines.append("")
    lines.append("## Per-pocket table")
    lines.append("")
    lines.append(
        "| pocket | status | pdbqt bytes | source bytes | source sha256 (first 12) | "
        "elapsed (s) |"
    )
    lines.append("|---|---|---:|---:|---|---:|")
    for record in records:
        sha = (record.get("source_pdb_sha256") or "")[:12]
        lines.append(
            "| `{pid}` | {status} | {pdbqt_b} | {src_b} | `{sha}` | {elapsed:.1f} |".format(
                pid=record["pocket_id"],
                status=record.get("status"),
                pdbqt_b=record.get("pdbqt_size_bytes") or "—",
                src_b=record.get("source_pdb_size_bytes") or "—",
                sha=sha or "—",
                elapsed=float(record.get("elapsed_seconds") or 0.0),
            )
        )
    lines.append("")
    lines.append("## Metric-measured pocket provenance")
    lines.append("")
    lines.append(
        "| pocket | metric file | metric status | mean kcal/mol | n samples |"
    )
    lines.append("|---|---|---|---:|---:|")
    for pocket_id, payload in provenance.items():
        lines.append(
            "| `{pid}` | `{file}` | {status} | {mean} | {n} |".format(
                pid=pocket_id,
                file=payload.get("metric_file", "—"),
                status=payload.get("metric_status", "—"),
                mean=payload.get("mean_kcal_mol", "—"),
                n=payload.get("n_samples", "—"),
            )
        )
    lines.append("")
    lines.append("## Per-pocket notes")
    lines.append("")
    for record in records:
        lines.append(f"### `{record['pocket_id']}` — {record.get('status')}")
        lines.append("")
        if record.get("error"):
            lines.append(f"- **error**: {record['error']}")
        if record.get("notes"):
            lines.append(f"- **notes**: {record['notes']}")
        if record.get("source_pdb"):
            lines.append(f"- **source_pdb**: `{record['source_pdb']}`")
        if record.get("pdbqt_path"):
            lines.append(f"- **pdbqt_path**: `{record['pdbqt_path']}`")
        if record.get("returncode") is not None:
            lines.append(f"- **returncode**: `{record['returncode']}`")
        if record.get("command"):
            lines.append(
                "- **command**: "
                + " ".join(str(part) for part in record["command"])
            )
        lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "```bash\n"
        "source env.sh\n"
        "uv run python -m molmetal.scripts.check_10_pockets --skip-existing\n"
        "```"
    )
    lines.append("")
    return "\n".join(lines)


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
        help="Restrict to one or more pocket_id values (default: all 10)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse any receptor.pdbqt already on disk; do not re-prepare",
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
        record = prepare_one(
            spec=spec,
            out_dir=per_pocket_dir,
            skip_existing=args.skip_existing,
            timeout_s=args.timeout_s,
        )
        records.append(record)
        status = record.get("status")
        print(
            f"{spec.pocket_id}: {status} "
            f"(pdbqt={record.get('pdbqt_size_bytes') or 'n/a'} bytes, "
            f"{float(record.get('elapsed_seconds') or 0.0):.1f}s)",
            flush=True,
        )
    summary = _summary(records)
    provenance = _load_metrics_provenance()
    status_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": str(args.out_dir),
        "molmetal_data_dir": str(MOLMETAL_DATA_DIR),
        "summary": summary,
        "pockets": records,
        "metric_provenance": provenance,
    }
    status_path = args.out_dir / "pocket_status.json"
    status_path.write_text(json.dumps(status_payload, indent=2) + "\n")
    log_path = args.out_dir / "pocket_prep_log.md"
    log_path.write_text(_render_markdown(records, provenance, summary))
    print(json.dumps({"summary": summary, "status_json": str(status_path)}))
    return 0 if summary.get("FAILED", 0) == 0 and summary.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())