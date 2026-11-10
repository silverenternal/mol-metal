#!/usr/bin/env python3
"""Wf-32 Phase 3 — 10-pocket × 5-candidate PoseBusters validity sweep.

Wf-43 Phase 3 audit (2026-09-26): the per-cell PoseBusters
evaluations already run on the real
``molmetal.validation.posebusters_runner.check_posebusters`` adapter
(no mock).  The OVERALL ``pb_pass_rate`` aggregate is also routed
through ``molmetal.scoring.score_any("pb_pass_rate", ...)`` so the
JSON audit trail records the tool name + backend + status
(``"MEASURED"``) per Wf-43 Phase 1's registry contract.

================================================================
Goal
================================================================
Run a structured PoseBusters (PB) validation experiment on the WF-30
ten-pocket cohort plus a 5-candidate ligand pool:

    10 pockets × 5 candidates = 50 PB evaluations
    × 26 checks (14 chemistry + 12 protein-aware) = 1300 PB checks

Each (pocket, candidate) cell is run with two modes:

* ``mol``     — chemistry + geometry only (14 checks).
* ``dock``    — adds protein-aware distance / clash checks (12 extra
                checks; total 26).

The script is the Wf-32 m8 PB-Valid deliverable.  It is intentionally
*read-mostly* — the docking trajectory is taken from the Wf-30 staging
directory (``receptor.pdbqt``) and the candidate SMILES are loaded from
``molmetal/scripts/_demo_5_metaldrugs.txt``.  The PoseBusters adapter
itself lives in ``molmetal/validation/posebusters_runner.py`` and
``molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py``.

Why a separate sweep (and not the in-loop adapter)?
---------------------------------------------------
The closed-loop + Vina outer-gate already call
:func:`check_posebusters` per candidate.  This sweep differs in three
ways:

1. **Coverage**: 50 candidates × 2 modes = 100 PB runs vs the closed-
   loop's 1 per candidate.  We want a *headline* pass-rate comparable
   to TargetDiff (0.94) and Uni-Mol v2 (0.75+).
2. **Mode split**: ``mol`` and ``dock`` are reported separately so the
   paper can quantify how much of the failure is protein-aware (clash
   / distance to cofactors) vs chemistry.
3. **JSON contract**: per-pocket + aggregate output is consumed by
   ``metrics/by_metric/pb_pass_rate.json`` (R17 1.000 smoke is the
   baseline this sweep must move or justify staying flat).

Inputs
------
* ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``
  (10 pockets, materialised by ``check_10_pockets.py``).
* ``molmetal/scripts/_demo_5_metaldrugs.txt`` — 5 reference SMILES.

Outputs
-------
* ``metrics/by_metric/wf32_pb_valid.json`` — per-(pocket, candidate,
  mode) pass / fail / skipped status + per-check failure names.
* ``metrics/by_metric/wf32_pb_valid_summary.json`` — per-pocket
  ``pb_pass_rate`` (mean ± std across 5 candidates) plus the
  overall ``pb_pass_rate`` (mean across 50 cells) for both modes.

Methods / modes
---------------
* ``mol``     — :func:`check_posebusters` on SMILES only (14 checks).
* ``dock``    — :func:`check_docked_pose` with the supplied receptor
                (26 checks).

Constraints
-----------
* CPU-only.  PB itself uses RDKit + numpy; no GPU.
* Graceful skip on missing ``posebusters`` package — the per-cell
  status records ``skipped=True`` so the closed-loop outer-gate pattern
  is preserved.
* No mutation of Wf-30 / Wf-31 / Wf-32 trainer modules.

Usage
-----
::

    source env.sh
    uv run --no-sync python -m molmetal.scripts.bench_pb_valid \\
        --methods mol dock \\
        --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc \\
        --n-candidates 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Repo-root path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_POCKETS: Tuple[str, ...] = (
    "830c",
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
    "1g9v",
    "2hnx",
    "4rn0",
    "1fmc",
)
# PB modes (mol = 14 chemistry checks, dock = 26 with protein-aware).
DEFAULT_PB_MODES: Tuple[str, ...] = ("mol", "dock")
# Trajectory-generator tags (matches Wf-30 docking methods).  Recorded
# as a metadata field so the PB-Valid JSON can be cross-referenced
# with the Wf-30 docking metrics; no separate sweep logic.
DEFAULT_TRAJECTORY_METHODS: Tuple[str, ...] = (
    "se3sb",
    "transport",
    "multiph",
    "baseline",
)
DEFAULT_METHOD: str = "se3sb"
DEFAULT_N_CANDIDATES: int = 5

# Output paths (overridable via CLI).
DEFAULT_OUT_DIR = _REPO_ROOT / "metrics" / "by_metric"
DEFAULT_METRICS_PATH = DEFAULT_OUT_DIR / "wf32_pb_valid.json"
DEFAULT_SUMMARY_PATH = DEFAULT_OUT_DIR / "wf32_pb_valid_summary.json"

# The DECLARATION OF RECORD for this sweep's scale.  It is NOT re-derived
# here: re-deriving it would re-create the very drift this path exists to
# prevent (a bench that declares its own target can always match itself).
# Wf-A127 (2026-10-31) — see catalog §6.239.
SPEC_TARGET_OF_RECORD_PATH = _REPO_ROOT / "molmetal" / "reports" / "wf32_m8_summary.json"


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    """Write ``payload`` as JSON via the atomic-write protocol (AGENTS.md §5.6).

    Wf-A127 (2026-10-31).  The bench used to call ``Path.write_text`` for
    both artifacts: a crash mid-write left a truncated JSON file that the
    registry would then treat as a changed source, and the ``sha`` recorded
    for the row would describe a half-written file.  ``atomic_write_file``
    writes to a temp file in the same directory and ``os.replace``s it, so
    the target is never observed half-written.
    """
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(path, json.dumps(payload, indent=2) + "\n")


def load_spec_target_of_record() -> Optional[Dict[str, object]]:
    """Return the declared sweep scale, copied VERBATIM from the record.

    Returns ``None`` when the declaration of record is absent or carries no
    ``spec_target`` — the caller then writes an explicit "declaration
    missing" block rather than a silent omission (an artifact that does not
    say what it was aiming at cannot be audited).
    """
    try:
        payload = json.loads(SPEC_TARGET_OF_RECORD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    spec = payload.get("spec_target") if isinstance(payload, dict) else None
    return dict(spec) if isinstance(spec, dict) else None


def build_honesty_blocks(
    *,
    n_evaluations: int,
    n_target_evaluations: int,
    per_evaluation: Sequence[Dict[str, object]],
    spec_target: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """The self-describing honesty blocks this artifact must always carry.

    Wf-A127 (2026-10-31).  Before this, the honesty blocks existed ONLY in
    the committed JSON — they were hand-written into
    ``metrics/by_metric/wf32_pb_valid.json`` by a wave, while the generator
    that rewrites that file wrote NEITHER of them.  The next regeneration
    therefore dropped the artifact's own honesty, silently, and the registry
    row went stale with the gap invisible.  The blocks are now COMPUTED from
    the run's own counts, so they cannot drift from what was measured.

    * ``spec_target`` — the declared scale, copied verbatim from
      :data:`SPEC_TARGET_OF_RECORD_PATH`.
    * ``wf432a1_pb_validity_deferred_block`` — what this file ACTUALLY
      measured, derived from ``per_evaluation``.
    """
    n_skipped = sum(
        1 for e in per_evaluation if str(e.get("status", "")).lower() == "skipped"
    )
    n_measured = int(n_evaluations) - n_skipped
    spec_cells = 0
    if spec_target is not None:
        try:
            spec_cells = int(spec_target.get("n_cells", 0))
        except (TypeError, ValueError):
            spec_cells = 0

    if spec_target is None:
        spec_block: Dict[str, object] = {
            "n_cells": None,
            "status": "DECLARATION OF RECORD MISSING",
            "declaration_of_record_path": str(SPEC_TARGET_OF_RECORD_PATH),
            "honest_framing": (
                f"The declaration of record ({SPEC_TARGET_OF_RECORD_PATH}) is "
                f"MISSING or carries no spec_target, so this artifact cannot "
                f"state what scale it was aiming at. It ran "
                f"{n_evaluations} evaluation(s) of a target of "
                f"{n_target_evaluations} cell(s). Do not cite this artifact "
                f"as a PoseBusters validity result."
            ),
        }
    else:
        spec_block = dict(spec_target)

    notes = sorted(
        {
            str(e.get("note", "")).strip()
            for e in per_evaluation
            if str(e.get("status", "")).lower() == "skipped" and e.get("note")
        }
    )
    note_clause = f" (per_evaluation note: '{notes[0]}')" if notes else ""

    if n_measured == 0 and n_evaluations > 0:
        honest_framing = (
            f"THIS ARTIFACT evaluated {n_evaluations} cell(s) out of the "
            f"{spec_cells}-cell sweep declared in spec_target (copied "
            f"verbatim from {SPEC_TARGET_OF_RECORD_PATH.name}), and ALL "
            f"{n_evaluations} evaluation(s) were SKIPPED{note_clause}, so "
            f"PoseBusters validity was NOT MEASURED (pb_valid=null, "
            f"n_checks=0, pass_rate=0.0). The {spec_cells}-cell sweep was "
            f"NEVER RUN — this file is the {n_evaluations}-cell smoke only. "
            f"Any SOTA-gate claim about a multi-pocket set belongs to "
            f"Wf-341-T1 §3 and is NOT reproduced here; do not cite this "
            f"artifact as a PoseBusters validity result."
        )
        status = (
            "DEFERRED HONEST-NEGATIVE — PoseBusters validity SKIPPED "
            "graceful preserved as honest-negative on validity NOT on "
            "accuracy"
        )
    elif n_measured == 0:
        honest_framing = (
            "THIS ARTIFACT ran ZERO evaluations, so PoseBusters validity was "
            "NOT MEASURED. Do not cite this artifact as a PoseBusters "
            "validity result."
        )
        status = "DEFERRED HONEST-NEGATIVE — zero evaluations were run"
    else:
        honest_framing = (
            f"THIS ARTIFACT evaluated {n_evaluations} cell(s) "
            f"({n_measured} measured, {n_skipped} skipped) against the "
            f"{spec_cells}-cell sweep declared in spec_target; "
            f"{max(spec_cells - n_evaluations, 0)} cell(s) of the declared "
            f"sweep were NOT run. The coverage is PARTIAL — read the "
            f"per_evaluation statuses before quoting a pass rate."
        )
        status = f"PARTIAL — {n_measured} measured / {n_skipped} skipped"

    deferred_block: Dict[str, object] = {
        "task_id": "A2_fix_step3_cat_b",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": (
            "PoseBusters validity was not measured for every declared cell; "
            "the shortfall is recorded here rather than averaged away."
        ),
        "closure_path": (
            "Wf-433+ USER_OK-gated follow-up with explicit meeko + autobabel "
            "pip install + Wf-433-A2 PoseBusters PbValid run on the "
            "5-pocket organic-cofactor set"
        ),
        "honest_framing": honest_framing,
        "n_evaluations": int(n_evaluations),
        "n_measured": n_measured,
        "n_skipped": n_skipped,
        "n_spec_cells": spec_cells,
    }
    return {
        "spec_target": spec_block,
        "wf432a1_pb_validity_deferred_block": deferred_block,
    }

# Pocket staging directory (shared with Wf-30 / Wf-31 — DO NOT mutate).
WF30_POCKET_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"

# Reference 5-candidate demo pool (shared with Wf-30 docking sweep).
DEFAULT_CANDIDATES_TXT = _REPO_ROOT / "molmetal" / "scripts" / "_demo_5_metaldrugs.txt"

# PoseBusters check counts (paper-grade, R17 ship).
N_PB_CHECKS_MOL: int = 14  # chemistry + geometry only
N_PB_CHECKS_DOCK: int = 26  # adds 12 protein-aware checks
N_PB_CHECKS_TOTAL: int = N_PB_CHECKS_DOCK  # for "26" headline number

log = logging.getLogger("bench_pb_valid")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One ligand candidate SMILES + a stable name (for JSON keys)."""

    name: str
    smiles: str


@dataclass(frozen=True)
class PocketBundle:
    """WF-30 pocket receptor (PDBQT) — minimal info for PB docking."""

    pocket_id: str
    pdbqt_path: Path
    pdbqt_exists: bool


@dataclass(frozen=True)
class PBResult:
    """One (pocket, candidate, pb_mode, trajectory_method) evaluation."""

    pocket_id: str
    candidate: str
    candidate_name: str
    method: str  # pb mode: "mol" | "dock"
    trajectory_method: str  # generation tag: se3sb / transport / ...
    status: str  # "passed" | "failed" | "skipped" | "error"
    pb_valid: Optional[bool]  # None when skipped
    n_checks: int
    n_passed: int
    n_failed: int
    n_skipped: int
    pass_rate: float  # n_passed / max(n_checks, 1)
    failed_checks: Tuple[str, ...]  # names of failing checks
    missing_checks: Tuple[str, ...]
    elapsed_s: float
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------
def load_default_candidates(path: Path) -> List[Candidate]:
    """Parse ``molmetal/scripts/_demo_5_metaldrugs.txt``.

    Format: one SMILES per line, optional ``# name`` after SMILES.
    Blank / comment-only lines are skipped.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Default candidate file not found at {path}; pass "
            "`--candidates-csv` to supply a different pool."
        )
    candidates: List[Candidate] = []
    with path.open("r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                smiles_part, _, comment = line.partition("#")
                smiles = smiles_part.strip()
                name = comment.strip() or f"cand_{len(candidates):02d}"
            else:
                smiles = line
                name = f"cand_{len(candidates):02d}"
            if not smiles:
                continue
            candidates.append(Candidate(name=name, smiles=smiles))
    if not candidates:
        raise ValueError(
            f"No candidates parsed from {path}; check the file is non-empty."
        )
    return candidates


def load_candidates_csv(path: Path) -> List[Candidate]:
    """Read a 2-column CSV with header ``name,smiles``."""
    candidates: List[Candidate] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "smiles"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"--candidates-csv must have columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get("name") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            candidates.append(
                Candidate(name=name or f"cand_{len(candidates):02d}", smiles=smiles)
            )
    if not candidates:
        raise ValueError(f"No candidates parsed from CSV {path}")
    return candidates


def select_candidates(
    pool: Sequence[Candidate],
    n: int,
) -> List[Candidate]:
    """Return the first ``n`` candidates (cap if pool is smaller)."""
    if n <= 0:
        return list(pool)
    return list(pool[: min(n, len(pool))])


# ---------------------------------------------------------------------------
# Pocket bundle loading
# ---------------------------------------------------------------------------
def load_pocket_bundle(pocket_id: str, pocket_dir: Path) -> PocketBundle:
    """Resolve the WF-30 receptor.pdbqt for a pocket.

    Unlike :func:`dock_10pockets_se3sb.load_pocket_bundle`, we do NOT
    parse ATOM/HETATM lines — PB's ``dock`` mode only needs the file
    path.  Returns ``pdbqt_exists=False`` when the receptor has not
    been staged so the caller can degrade gracefully to ``mol``-only.
    """
    pdbqt_path = pocket_dir / pocket_id / "receptor.pdbqt"
    return PocketBundle(
        pocket_id=pocket_id,
        pdbqt_path=pdbqt_path,
        pdbqt_exists=pdbqt_path.exists(),
    )


# ---------------------------------------------------------------------------
# PoseBusters dispatch
# ---------------------------------------------------------------------------
def _coerce_status(report: Dict[str, object]) -> Tuple[str, Optional[bool]]:
    """Map a PoseBusters report dict to (status, pb_valid)."""
    if bool(report.get("skipped", False)):
        return "skipped", None
    pb = report.get("pb_valid")
    if pb is None:
        # Backend absent or upstream parse failure → treat as skipped
        # (matches the closed-loop outer-gate pattern).
        return "skipped", None
    if isinstance(pb, bool):
        return ("passed" if pb else "failed"), bool(pb)
    # Anything else (NaN, strings, etc.) is a degraded "error".
    return "error", None


def _per_check_counts(
    report: Dict[str, object],
) -> Tuple[int, int, int, int, Tuple[str, ...], Tuple[str, ...]]:
    """Pull per-check counts and failing-check names out of a PB report.

    Returns ``(n_checks, n_passed, n_failed, n_skipped, failed, missing)``.
    Skipped counts are inferred from ``missing_checks`` (those checks
    were not run / not reported) and from any ``None`` entries in the
    per-check dict.
    """
    checks = report.get("checks", {}) or {}
    missing = tuple(report.get("missing_checks", ()) or ())
    failed_list = list(report.get("failures", []) or [])
    n_total = len(checks) + len(missing)
    n_passed = sum(1 for v in checks.values() if v is True)
    n_failed = sum(1 for v in checks.values() if v is False)
    n_skipped = sum(1 for v in checks.values() if v is None) + len(missing)
    # Failed names — prefer ``failures`` list when present (PB's
    # canonical ordering), fall back to keys whose value is False.
    failed_names = (
        tuple(failed_list)
        if failed_list
        else tuple(k for k, v in checks.items() if v is False)
    )
    if n_total == 0:
        n_total = N_PB_CHECKS_MOL if report.get("config") == "mol" else N_PB_CHECKS_DOCK
    return (
        int(n_total),
        int(n_passed),
        int(n_failed),
        int(n_skipped),
        failed_names,
        missing,
    )


def run_pb_one(
    method: str,
    candidate: Candidate,
    pocket: PocketBundle,
    trajectory_method: str = DEFAULT_METHOD,
) -> PBResult:
    """One (method, candidate, pocket) PoseBusters evaluation."""
    t0 = time.monotonic()
    # Defer the import so a missing ``posebusters`` package surfaces
    # here as a ``skipped`` cell rather than a top-level crash.
    try:
        from molmetal.validation.posebusters_runner import (
            check_posebusters,
            posebusters_available,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("posebusters_runner import failed: %s", exc)
        return PBResult(
            pocket_id=pocket.pocket_id,
            candidate=candidate.smiles,
            candidate_name=candidate.name,
            method=method,
            trajectory_method=trajectory_method,
            status="skipped",
            pb_valid=None,
            n_checks=0,
            n_passed=0,
            n_failed=0,
            n_skipped=0,
            pass_rate=0.0,
            failed_checks=(),
            missing_checks=(),
            elapsed_s=time.monotonic() - t0,
            note=f"posebusters_runner import failed: {exc}",
        )

    if not posebusters_available():
        return PBResult(
            pocket_id=pocket.pocket_id,
            candidate=candidate.smiles,
            candidate_name=candidate.name,
            method=method,
            trajectory_method=trajectory_method,
            status="skipped",
            pb_valid=None,
            n_checks=0,
            n_passed=0,
            n_failed=0,
            n_skipped=0,
            pass_rate=0.0,
            failed_checks=(),
            missing_checks=(),
            elapsed_s=time.monotonic() - t0,
            note="posebusters not installed",
        )

    if method == "mol":
        report = check_posebusters(candidate.smiles)
    elif method == "dock":
        if not pocket.pdbqt_exists:
            return PBResult(
                pocket_id=pocket.pocket_id,
                candidate=candidate.smiles,
                candidate_name=candidate.name,
                method=method,
                trajectory_method=trajectory_method,
                status="skipped",
                pb_valid=None,
                n_checks=0,
                n_passed=0,
                n_failed=0,
                n_skipped=0,
                pass_rate=0.0,
                failed_checks=(),
                missing_checks=(),
                elapsed_s=time.monotonic() - t0,
                note=f"receptor.pdbqt missing for {pocket.pocket_id}",
            )
        # ``check_docked_pose`` requires a single-conformer mol with
        # finite coordinates.  We embed the SMILES via the same
        # ETKDGv3+MMFF94 path as ``check_posebusters`` and pass the
        # resulting mol to ``check_docked_pose``.
        report = _run_dock_with_embed(candidate.smiles, pocket.pdbqt_path)
    else:
        raise ValueError(f"unknown method {method!r}; valid: {('mol', 'dock')}")

    status, pb_valid = _coerce_status(report)
    n_total, n_passed, n_failed, n_skipped, failed, missing = _per_check_counts(report)
    pass_rate = float(n_passed) / max(int(n_total), 1)
    elapsed = time.monotonic() - t0
    note = report.get("error") if status == "error" else None
    return PBResult(
        pocket_id=pocket.pocket_id,
        candidate=candidate.smiles,
        candidate_name=candidate.name,
        method=method,
        trajectory_method=trajectory_method,
        status=status,
        pb_valid=pb_valid,
        n_checks=n_total,
        n_passed=n_passed,
        n_failed=n_failed,
        n_skipped=n_skipped,
        pass_rate=pass_rate,
        failed_checks=failed,
        missing_checks=missing,
        elapsed_s=elapsed,
        note=note if note else None,
    )


def _run_dock_with_embed(smiles: str, receptor_path: Path) -> Dict[str, object]:
    """Embed a SMILES (ETKDGv3+MMFF94) and pass to ``check_docked_pose``.

    The PB ``dock`` config requires a single conformer on the input mol;
    we use the same RDKit path as :func:`check_posebusters` so the
    docked-pose check is comparable to the mol-mode check.
    """
    from molmetal.validation.posebusters_runner import check_docked_pose

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return {
                "pb_valid": False,
                "skipped": False,
                "config": "dock",
                "status": "invalid_input",
                "checks": {},
                "missing_checks": (),
                "failures": ["smiles_parse"],
                "n_conformers": 0,
            }
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status == -1:
            return {
                "pb_valid": False,
                "skipped": False,
                "config": "dock",
                "status": "embed_failed",
                "checks": {},
                "missing_checks": (),
                "failures": ["embed_3d"],
                "n_conformers": 0,
            }
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        Chem.RemoveHs(mol)
        return check_docked_pose(mol, str(receptor_path))
    except Exception as exc:  # pragma: no cover
        return {
            "pb_valid": False,
            "skipped": False,
            "config": "dock",
            "status": "error",
            "checks": {},
            "missing_checks": (),
            "failures": ["dock_dispatch_failed"],
            "error": str(exc),
            "n_conformers": 0,
        }


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep(
    pockets: Sequence[str],
    candidates: Sequence[Candidate],
    pb_modes: Sequence[str],
    pocket_dir: Path,
    trajectory_method: str = DEFAULT_METHOD,
) -> List[PBResult]:
    """Run the full (pockets × candidates × pb_modes) sweep.

    ``trajectory_method`` is recorded as a metadata field on every
    PBResult so the Wf-32 JSON can be cross-referenced with the Wf-30
    docking metrics (the Wf-30 sweep tags each candidate pose with
    one of ``se3sb`` / ``transport`` / ``multiph`` / ``baseline``).
    """
    valid = {"mol", "dock"}
    for m in pb_modes:
        if m not in valid:
            raise ValueError(f"unknown pb_mode {m!r}; valid: {sorted(valid)}")
    results: List[PBResult] = []
    n_total = len(pockets) * len(candidates) * len(pb_modes)
    counter = 0
    t_start = time.monotonic()
    for pocket_id in pockets:
        try:
            bundle = load_pocket_bundle(pocket_id, pocket_dir)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", pocket_id, exc)
            continue
        for cand in candidates:
            for mode in pb_modes:
                counter += 1
                res = run_pb_one(
                    mode, cand, bundle, trajectory_method=trajectory_method
                )
                results.append(res)
                if counter % max(1, n_total // 20) == 0 or counter == n_total:
                    elapsed = time.monotonic() - t_start
                    log.info(
                        "sweep %d/%d (%.0f%%) — %s %s pb=%s traj=%s status=%s "
                        "pass_rate=%.3f (%.1fs total)",
                        counter,
                        n_total,
                        100.0 * counter / max(1, n_total),
                        pocket_id,
                        cand.name,
                        mode,
                        trajectory_method,
                        res.status,
                        res.pass_rate,
                        elapsed,
                    )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(
    results: Iterable[PBResult],
) -> Tuple[Dict[str, Dict[str, object]], Dict[Tuple[str, str], Dict[str, float]]]:
    """Reduce the per-eval list into (per-pocket-per-method, aggregate).

    Aggregate key: ``(pocket_id, method)`` where ``method`` is the PB
    mode (``mol`` / ``dock``).  Pass rate is reported as the mean of
    per-cell pass rates (so a single skipped cell does not crater the
    average the way a sum-based fraction would).
    """
    by_pocket_method_rates: Dict[Tuple[str, str], List[float]] = {}
    by_pocket_method_status: Dict[Tuple[str, str], Dict[str, int]] = {}
    for res in results:
        key = (res.pocket_id, res.method)
        by_pocket_method_rates.setdefault(key, []).append(float(res.pass_rate))
        stats = by_pocket_method_status.setdefault(
            key,
            {"n_passed": 0, "n_failed": 0, "n_skipped": 0, "n_error": 0, "n_total": 0},
        )
        stats["n_total"] += 1
        if res.status == "passed":
            stats["n_passed"] += 1
        elif res.status == "failed":
            stats["n_failed"] += 1
        elif res.status == "skipped":
            stats["n_skipped"] += 1
        else:
            stats["n_error"] += 1
    aggregate: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, rates in by_pocket_method_rates.items():
        if not rates:
            continue
        aggregate[key] = {
            "n": int(len(rates)),
            "mean_pass_rate": float(statistics.fmean(rates)),
            "std_pass_rate": float(statistics.pstdev(rates)) if len(rates) > 1 else 0.0,
            "min_pass_rate": float(min(rates)),
            "max_pass_rate": float(max(rates)),
            **by_pocket_method_status[key],
        }
    per_pocket: Dict[str, Dict[str, object]] = {}
    for (pocket_id, method), stats in aggregate.items():
        per_pocket.setdefault(pocket_id, {})[method] = stats
    return per_pocket, aggregate


def overall_pass_rate(
    results: Iterable[PBResult],
    pb_mode: str,
    trajectory_method: Optional[str] = None,
) -> Optional[float]:
    """Headline ``pb_pass_rate`` for one (pb_mode, trajectory_method).

    When ``trajectory_method`` is provided the filter is restricted to
    PBResults whose tag matches (so ``se3sb``-generated candidates
    can be compared against ``transport``-generated ones).
    """
    rates = [
        r.pass_rate
        for r in results
        if r.method == pb_mode
        and (trajectory_method is None or r.trajectory_method == trajectory_method)
    ]
    if not rates:
        return None
    return float(statistics.fmean(rates))


def overall_pass_rate_via_score_any(
    results: Iterable[PBResult],
    pb_mode: str,
    trajectory_method: Optional[str] = None,
    *,
    pocket_pdb: Optional[str] = None,
) -> Dict[str, Any]:
    """Wf-43 Phase 3 — overall ``pb_pass_rate`` routed through ``score_any``.

    Returns a dict ``{"metric": "pb_pass_rate", "tool": str,
    "backend": str, "values": List[float], "n": int, "status": str}``
    suitable for direct inclusion in the metrics JSON.  When the
    PoseBusters adapter is missing on the host, returns
    ``status="skipped: AdapterUnavailable"`` — never silently
    substitutes a stub.

    The caller can additionally cross-check that the locally-computed
    ``overall_pass_rate(results, pb_mode, trajectory_method)`` agrees
    with the ``values[0]`` returned by the router (sanity gate).
    """
    from molmetal.scoring.score_any import AdapterUnavailable, score_any

    smiles = [
        r.candidate
        for r in results
        if r.method == pb_mode
        and (trajectory_method is None or r.trajectory_method == trajectory_method)
    ]
    if not smiles:
        return {
            "metric": "pb_pass_rate",
            "tool": "PoseBusters (R17 ship)",
            "backend": "python",
            "values": [],
            "n": 0,
            "status": "skipped: no_cells",
        }

    try:
        result = score_any(
            "pb_pass_rate",
            molecules=list(smiles),
            pocket=pocket_pdb,
        )
        return result
    except AdapterUnavailable as exc:
        return {
            "metric": "pb_pass_rate",
            "tool": "PoseBusters (R17 ship)",
            "backend": "python",
            "values": [],
            "n": 0,
            "status": f"skipped: {type(exc).__name__}",
        }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj):
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-32 Phase 3 — 10-pocket × 5-candidate PoseBusters validity sweep."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pockets", nargs="+", default=list(DEFAULT_POCKETS))
    p.add_argument(
        "--pb-modes",
        nargs="+",
        choices=("mol", "dock"),
        default=list(DEFAULT_PB_MODES),
        help="PoseBusters modes to run (mol = 14 chemistry checks; "
        "dock = 26 chemistry + protein-aware checks).",
    )
    p.add_argument(
        "--method",
        choices=DEFAULT_TRAJECTORY_METHODS,
        default=DEFAULT_METHOD,
        help="Trajectory-generator tag (matches Wf-30 docking methods). "
        "Recorded as a metadata field for cross-referencing with "
        "the Wf-30 docking metrics.",
    )
    p.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    p.add_argument(
        "--candidates-csv",
        type=Path,
        default=None,
        help="Optional 2-column CSV (name,smiles); default reads "
        "_demo_5_metaldrugs.txt.",
    )
    p.add_argument(
        "--candidates-txt",
        type=Path,
        default=DEFAULT_CANDIDATES_TXT,
    )
    p.add_argument(
        "--pocket-dir",
        type=Path,
        default=WF30_POCKET_DIR,
        help="WF-30 staging directory with <pocket>/receptor.pdbqt files.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )
    p.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-evaluation logging.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    log.info(
        "bench_pb_valid starting: pockets=%d pb_modes=%s "
        "trajectory_method=%s n_candidates=%d",
        len(args.pockets),
        args.pb_modes,
        args.method,
        args.n_candidates,
    )

    if args.candidates_csv is not None:
        candidates = load_candidates_csv(args.candidates_csv)
    else:
        candidates = load_default_candidates(args.candidates_txt)
    candidates = select_candidates(candidates, args.n_candidates)
    log.info("loaded %d candidates", len(candidates))

    results = run_sweep(
        pockets=args.pockets,
        candidates=candidates,
        pb_modes=args.pb_modes,
        pocket_dir=args.pocket_dir,
        trajectory_method=args.method,
    )

    n_total_cells = len(args.pockets) * len(candidates) * len(args.pb_modes)
    per_pocket, aggregate = aggregate_results(results)
    overall = {m: overall_pass_rate(results, m, args.method) for m in args.pb_modes}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "recipe": "wf32_pb_valid",
        "n_pockets": len(args.pockets),
        "n_candidates": len(candidates),
        "n_pb_modes": len(args.pb_modes),
        "trajectory_method": args.method,
        "n_evaluations": int(len(results)),
        "n_target_evaluations": int(n_total_cells),
        "pb_modes": list(args.pb_modes),
        "pockets": list(args.pockets),
        "candidates": [{"name": c.name, "smiles": c.smiles} for c in candidates],
        "n_pb_checks_mol": N_PB_CHECKS_MOL,
        "n_pb_checks_dock": N_PB_CHECKS_DOCK,
        "n_pb_checks_total": N_PB_CHECKS_TOTAL,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "per_evaluation": [
            {
                "pocket_id": r.pocket_id,
                "candidate": r.candidate,
                "candidate_name": r.candidate_name,
                "method": r.method,
                "trajectory_method": r.trajectory_method,
                "status": r.status,
                "pb_valid": r.pb_valid,
                "n_checks": r.n_checks,
                "n_passed": r.n_passed,
                "n_failed": r.n_failed,
                "n_skipped": r.n_skipped,
                "pass_rate": r.pass_rate,
                "failed_checks": list(r.failed_checks),
                "missing_checks": list(r.missing_checks),
                "elapsed_s": r.elapsed_s,
                "note": r.note,
            }
            for r in results
        ],
        "per_pocket_per_method": _to_jsonable(per_pocket),
        "aggregate": {f"{k[0]}|{k[1]}": _to_jsonable(v) for k, v in aggregate.items()},
        "overall_pb_pass_rate": _to_jsonable(overall),
    }
    # Wf-A127 (2026-10-31): the honesty blocks are COMPUTED here, not
    # hand-written into the JSON afterwards.  Before this, the artifact's
    # own honesty lived only in the committed file and every regeneration
    # silently dropped it (catalog §6.239).
    metrics_payload.update(
        build_honesty_blocks(
            n_evaluations=len(results),
            n_target_evaluations=n_total_cells,
            per_evaluation=list(metrics_payload["per_evaluation"]),
            spec_target=load_spec_target_of_record(),
        )
    )
    _write_json_atomic(args.metrics_path, metrics_payload)
    log.info("wrote %s", args.metrics_path)

    summary_payload = {
        "recipe": "wf32_pb_valid_summary",
        "n_evaluations": int(len(results)),
        "n_target_evaluations": int(n_total_cells),
        "n_pockets": len(args.pockets),
        "n_candidates": len(candidates),
        "pb_modes": list(args.pb_modes),
        "trajectory_method": args.method,
        "n_pb_checks_mol": N_PB_CHECKS_MOL,
        "n_pb_checks_dock": N_PB_CHECKS_DOCK,
        "per_pocket_per_method": {
            f"{k[0]}|{k[1]}": _to_jsonable(v) for k, v in aggregate.items()
        },
        "overall_pb_pass_rate": _to_jsonable(overall),
        "aggregate": {f"{k[0]}|{k[1]}": _to_jsonable(v) for k, v in aggregate.items()},
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    _write_json_atomic(args.summary_path, summary_payload)
    log.info("wrote %s", args.summary_path)

    # Headline log
    for mode, rate in overall.items():
        if rate is None:
            log.warning("overall %s pb_pass_rate = UNDEFINED (no cells)", mode)
        else:
            log.info("overall %s pb_pass_rate = %.4f", mode, rate)

    if not results:
        log.error("bench_pb_valid produced zero evaluations")
        return 1
    log.info("bench_pb_valid PASSED (%d cells)", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
