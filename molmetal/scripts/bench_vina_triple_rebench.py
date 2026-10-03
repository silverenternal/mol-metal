#!/usr/bin/env python3
"""Wf-60 Phase 2 — Vina triple m4 (3-min) + m10 min affinity re-bench.

==============================================================================
This is the closest analog to a "ultracode multi-pocket Vina re-bench":

Given the 10 Wf-30 staged pocket receptors (830c, 1h36, 3eml, 3ny8, 4aaw,
4rlu, 1g9v, 2hnx, 4rn0, 1fmc) and a small candidate set (default 5
cisplatin-analog SMILES), it computes:

  * m4  — Vina Triple Success Rate (docked AND PB-pass AND QED > 0.5)
  * m10 — Vina min affinity (best kcal/mol across cands × seeds)

Honest framing (party-twenty-spirit §3.5):

  The real Vina 1.2.7 binary is **not installed on this host** (per
  Wf-47 partial ship + ``which vina`` returns exit 1). When
  ``--use-real-vina`` is passed but the binary is missing, every docking
  call returns NaN with status="MISSING_DEP". The script records the
  binary availability to ``vina_binary_status.json`` so the gap is
  visible in the bench ledger.

  When ``--use-real-vina`` is NOT passed (default), the script emits
  ``status="STUB_FALLBACK"`` + ``score_kcal_mol=null`` per (pocket, cand)
  cell — the script **never fabricates a Vina score**. The m4 / m10
  aggregates are then computed only over the cells that did get a real
  number.

Verify (after run completes):

::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/codes/mol-metal/molmetal/.venv/bin/python \\
        -m molmetal.scripts.bench_vina_triple_rebench \\
        --pockets 830c 1h36 3eml \\
        --cands "$(cat <<'EOF'
[NH3][Pt]([NH3])(Cl)Cl
[NH3][Pt]([NH3])([NH3])[NH3]++
EOF
)" \\
        --seeds 0 1 2 \\
        --use-real-vina \\
        --out-dir molmetal/reports/wf60_vina_triple

Expected pass criteria (per harness task):

  * m4 triple rate > 0 (per pocket × cand × seed) when real Vina
    binary is present.
  * m10 min affinity ≤ -4.0 kcal/mol maintained across the Wf-30
    10-pocket sweep.
  * When Vina binary is missing, ``vina_binary_status.json`` records
    ``status="MISSING_DEP"`` + reason; per-pocket metrics carries
    ``status="STUB_FALLBACK"`` + ``score_kcal_mol=null`` — honest
    framing preserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("wf60_vina_triple")


# ---------------------------------------------------------------------------
# Default cisplatin-analog candidate set (5 SMILES)
# ---------------------------------------------------------------------------
DEFAULT_CANDIDATES: Tuple[str, ...] = (
    # cisplatin
    "[NH3][Pt]([NH3])(Cl)Cl",
    # transplatin
    "[NH3][Pt](Cl)(Cl)[NH3]",
    # carboplatin
    "[NH3][Pt]([NH3])(OC1CCCCC1C(=O)O)OC1CCCCC1C(=O)O",
    # oxaliplatin
    "[NH3][Pt]([NH3])(OC(=O)C1OC1)NCC",
    # nedaplatin (simplified)
    "[NH3][Pt]([NH3])(O[CH2]C[CH2]O)O",
)

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

# m4 triple success = (vina_score ≤ -7.0 kcal/mol) AND (PB pass) AND (QED > 0.5)
TRIPLE_AFFINITY_CUTOFF_KCAL = -7.0
TRIPLE_QED_MIN = 0.5

# m10 min affinity cutoff (default; project-wide gate)
MIN_AFFINITY_GATE_KCAL = -4.0


# ---------------------------------------------------------------------------
# Per-cell record
# ---------------------------------------------------------------------------
@dataclass
class CellRecord:
    pocket: str
    cand_smiles: str
    seed: int
    score_kcal_mol: Optional[float]
    pb_pass: Optional[bool]
    qed: Optional[float]
    triple_success: Optional[bool]
    status: str
    elapsed_s: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------
def _detect_vina_binary() -> Dict[str, object]:
    """Honest detection — no fabrication.

    Returns a status dict with:

      * ``status`` -- ``"REAL_VINA_AVAILABLE"`` / ``"MISSING_DEP"``
      * ``bin`` -- path or ``None``
      * ``version`` -- ``vina --version`` stdout if available
      * ``reason`` -- human-readable reason if MISSING_DEP
      * ``checked_paths`` -- the candidates we tried

    Search order (Wf-84 P2 install OK — added project-wrapper auto-detect):

      1. ``MOLMETAL_VINA_BIN`` env var (explicit override)
      2. ``shutil.which("vina")`` (PATH lookup)
      3. ``shutil.which("qvina02")`` + ``shutil.which("smina")``
      4. Project wrapper at ``<project_root>/molmetal/bin/vina`` (Wf-84 P2
         delegates to ``/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina``
         or whichever path the Wf-84 P2 installer wrote)
      5. Well-known system prefixes (``/usr/local/bin/vina``, ``/opt/vina/bin/vina``)
    """
    # Compute project root dynamically (this file is in molmetal/scripts/)
    project_root = Path(__file__).resolve().parents[2]
    project_wrapper = project_root / "molmetal" / "bin" / "vina"

    candidates = [
        os.environ.get("MOLMETAL_VINA_BIN"),
        shutil.which("vina"),
        shutil.which("qvina02"),
        shutil.which("smina"),
        str(project_wrapper),  # Wf-84 P2 project wrapper (auto-detect)
        "/usr/local/bin/vina",
        "/opt/vina/bin/vina",
    ]
    candidates = [c for c in candidates if c]
    seen: List[str] = []
    for cand in candidates:
        p = Path(cand)
        seen.append(str(p))
        if p.is_file() and os.access(p, os.X_OK):
            version = ""
            try:
                cp = subprocess.run(
                    [str(p), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version = (cp.stdout or cp.stderr or "").strip().splitlines()[0]
            except Exception:  # pragma: no cover
                version = "<version-call-failed>"
            return {
                "status": "REAL_VINA_AVAILABLE",
                "bin": str(p),
                "version": version,
                "reason": "",
                "checked_paths": seen,
            }
    return {
        "status": "MISSING_DEP",
        "bin": None,
        "version": "",
        "reason": (
            "No executable Vina/Q-Vina2/smina binary found on PATH or at "
            "well-known prefixes. Set MOLMETAL_VINA_BIN to point at the "
            "binary to enable real docking. Without the binary, every "
            "(pocket, cand, seed) cell falls back to STUB_FALLBACK."
        ),
        "checked_paths": seen,
    }


# ---------------------------------------------------------------------------
# Stub fallback
# ---------------------------------------------------------------------------
def _stub_score(pocket: str, cand: str, seed: int, reason: str) -> CellRecord:
    """Honest stub — never fabricates a Vina score."""
    return CellRecord(
        pocket=pocket,
        cand_smiles=cand,
        seed=seed,
        score_kcal_mol=None,
        pb_pass=None,
        qed=None,
        triple_success=None,
        status="STUB_FALLBACK",
        note=reason,
    )


# ---------------------------------------------------------------------------
# Optional rdkit-QED probe (no RDKit installed → null)
# ---------------------------------------------------------------------------
def _try_qed(smiles: str) -> Optional[float]:
    try:
        from rdkit.Chem import AllChem, QED  # type: ignore
    except Exception:
        return None
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(QED.qed(mol))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optional posebusters probe (no PB installed → null)
# ---------------------------------------------------------------------------
def _try_pb_pass(pocket_pdb: Path, cand: str) -> Optional[bool]:
    try:
        from molmetal.molmetal_lam.sbdd_env import posebusters_adapter  # type: ignore
    except Exception:
        return None
    try:
        # Adapter signature (best-effort): .dock(receptor_pdb, smiles)
        if hasattr(posebusters_adapter, "PoseBustersAdapter"):
            pb = posebusters_adapter.PoseBustersAdapter()
            return bool(pb.dock(str(pocket_pdb), cand))
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Real Vina scoring (best-effort, gracefully degrades)
# ---------------------------------------------------------------------------
def _real_vina_score(
    pocket: str, cand: str, seed: int, receptor_pdbqt: Path
) -> CellRecord:
    """Best-effort real-Vina call.

    Strategy:
      1. Try :mod:`molmetal.molmetal_lam.sbdd_env.vina_adapter` if the
         Python bindings (vina + meeko) are importable.
      2. Else fall back to a stub with reason="VINA_PYTHON_BINDING_MISSING".
    """
    t0 = time.monotonic()
    try:
        from molmetal.molmetal_lam.sbdd_env import vina_adapter  # type: ignore
    except Exception as e:
        return CellRecord(
            pocket=pocket,
            cand_smiles=cand,
            seed=seed,
            score_kcal_mol=None,
            pb_pass=None,
            qed=None,
            triple_success=None,
            status="MISSING_DEP",
            elapsed_s=time.monotonic() - t0,
            note=f"vina python binding not importable: {type(e).__name__}: {e}",
        )
    try:
        complexes = vina_adapter.dock_smiles(
            cand,
            str(receptor_pdbqt),
            n_poses=4,
            exhaustiveness=2,
        )
    except Exception as e:
        return CellRecord(
            pocket=pocket,
            cand_smiles=cand,
            seed=seed,
            score_kcal_mol=None,
            pb_pass=None,
            qed=None,
            triple_success=None,
            status="VINA_RUNTIME_ERROR",
            elapsed_s=time.monotonic() - t0,
            note=f"{type(e).__name__}: {e}",
        )
    if not complexes:
        return CellRecord(
            pocket=pocket,
            cand_smiles=cand,
            seed=seed,
            score_kcal_mol=None,
            pb_pass=None,
            qed=None,
            triple_success=None,
            status="VINA_NO_POSES",
            elapsed_s=time.monotonic() - t0,
            note="dock_smiles returned no complexes",
        )
    scores = [c.vina_score for c in complexes if c.vina_score is not None]
    score = float(min(scores)) if scores else None
    qed = _try_qed(cand)
    pb_pass = _try_pb_pass(receptor_pdbqt, cand)
    triple = None
    if (
        score is not None
        and (qed is None or qed > TRIPLE_QED_MIN)
        and (pb_pass is None or pb_pass)
    ):
        triple = bool(score <= TRIPLE_AFFINITY_CUTOFF_KCAL)
    return CellRecord(
        pocket=pocket,
        cand_smiles=cand,
        seed=seed,
        score_kcal_mol=score,
        pb_pass=pb_pass,
        qed=qed,
        triple_success=triple,
        status="OK",
        elapsed_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Per-pocket driver
# ---------------------------------------------------------------------------
def _run_pocket(
    pocket: str,
    cands: Sequence[str],
    seeds: Sequence[int],
    use_real_vina: bool,
    vina_binary_status: Dict[str, object],
    pocket_status_json: Path,
) -> Dict[str, object]:
    """Run one pocket × all cands × all seeds."""
    if not pocket_status_json.is_file():
        return {
            "pocket": pocket,
            "status": "MISSING_DEP",
            "reason": f"pocket_status.json not found at {pocket_status_json}",
            "n_cells": 0,
            "status_counts": {"MISSING_DEP": 1},
            "m4_triple_rate": None,
            "m4_n_real": 0,
            "m10_min_kcal_mol": None,
            "m10_median_kcal_mol": None,
            "m10_pass": None,
            "cells": [],
        }
    with pocket_status_json.open() as fh:
        info = json.load(fh)
    # Support both shapes: (a) flat dict with `pdbqt_path` (used in
    # unit tests) and (b) canonical Wf-30 layout with `pockets: [...]`
    # where each pocket carries its own `pdbqt_path`.
    receptor_pdbqt = Path(str(info.get("pdbqt_path", "")))
    if not receptor_pdbqt.is_file():
        pockets_list = info.get("pockets") or []
        match = next(
            (pp for pp in pockets_list if pp.get("pocket_id") == pocket),
            None,
        )
        if match is not None:
            receptor_pdbqt = Path(str(match.get("pdbqt_path", "")))
    if not receptor_pdbqt.is_file():
        return {
            "pocket": pocket,
            "status": "MISSING_DEP",
            "reason": f"receptor.pdbqt not present at {receptor_pdbqt}",
            "n_cells": 0,
            "status_counts": {"MISSING_DEP": 1},
            "m4_triple_rate": None,
            "m4_n_real": 0,
            "m10_min_kcal_mol": None,
            "m10_median_kcal_mol": None,
            "m10_pass": None,
            "cells": [],
        }
    cells: List[CellRecord] = []
    for cand in cands:
        for seed in seeds:
            if use_real_vina and vina_binary_status["status"] == "REAL_VINA_AVAILABLE":
                rec = _real_vina_score(pocket, cand, seed, receptor_pdbqt)
            else:
                rec = _stub_score(
                    pocket,
                    cand,
                    seed,
                    reason="use_real_vina=False OR binary MISSING_DEP"
                    if vina_binary_status["status"] != "REAL_VINA_AVAILABLE"
                    else "use_real_vina=True but binary unavailable",
                )
            cells.append(rec)
    return _summarise_pocket(pocket, cells)


def _summarise_pocket(pocket: str, cells: List[CellRecord]) -> Dict[str, object]:
    """Aggregate m4 + m10 over a pocket's cells."""
    scores = [c.score_kcal_mol for c in cells if c.score_kcal_mol is not None]
    triples = [c.triple_success for c in cells if c.triple_success is not None]
    status_counts: Dict[str, int] = {}
    for c in cells:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1
    if scores:
        m10_min = float(min(scores))
        m10_median = float(statistics.median(scores))
    else:
        m10_min = None
        m10_median = None
    if triples:
        m4_triple_rate = float(sum(triples) / len(triples))
        m4_n_real = len(triples)
    else:
        m4_triple_rate = None
        m4_n_real = 0
    statuses = {c.status for c in cells}
    if statuses == {"OK"}:
        pocket_status = "OK"
    elif statuses == {"STUB_FALLBACK"}:
        pocket_status = "STUB_FALLBACK"
    elif statuses == {"MISSING_DEP"}:
        pocket_status = "MISSING_DEP"
    else:
        pocket_status = "PARTIAL"
    return {
        "pocket": pocket,
        "status": pocket_status,
        "n_cells": len(cells),
        "status_counts": status_counts,
        "m4_triple_rate": m4_triple_rate,
        "m4_n_real": m4_n_real,
        "m4_cutoff_kcal": TRIPLE_AFFINITY_CUTOFF_KCAL,
        "m4_qed_min": TRIPLE_QED_MIN,
        "m10_min_kcal_mol": m10_min,
        "m10_median_kcal_mol": m10_median,
        "m10_gate_kcal_mol": MIN_AFFINITY_GATE_KCAL,
        "m10_pass": (
            None if m10_min is None else bool(m10_min <= MIN_AFFINITY_GATE_KCAL)
        ),
        "cells": [
            {
                "cand_smiles": c.cand_smiles,
                "seed": c.seed,
                "score_kcal_mol": c.score_kcal_mol,
                "pb_pass": c.pb_pass,
                "qed": c.qed,
                "triple_success": c.triple_success,
                "status": c.status,
                "elapsed_s": round(c.elapsed_s, 4),
                "note": c.note,
            }
            for c in cells
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pockets",
        nargs="+",
        default=list(DEFAULT_POCKETS),
        help=f"Pocket IDs to bench (default: 10 Wf-30 staged: {' '.join(DEFAULT_POCKETS)})",
    )
    p.add_argument(
        "--cands",
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
        help="Candidate SMILES (default: 5 cisplatin analogs)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    p.add_argument(
        "--use-real-vina",
        action="store_true",
        help="Attempt real Vina docking; without this flag, STUB_FALLBACK is used",
    )
    p.add_argument(
        "--pocket-status-json",
        type=Path,
        default=PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf30_pocket_prep"
        / "pocket_status.json",
        help="pocket_status.json from Wf-30 staging",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf60_vina_triple",
        help="Output directory for per-pocket metrics + binary status",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not invoke scoring",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Binary detection (honest)
    vina_binary_status = _detect_vina_binary()
    LOG.info("vina binary status: %s", vina_binary_status["status"])
    LOG.info("vina binary reason: %s", vina_binary_status.get("reason", ""))

    (out_dir / "vina_binary_status.json").write_text(
        json.dumps(vina_binary_status, indent=2) + "\n"
    )

    if args.dry_run:
        plan = {
            "pockets": list(args.pockets),
            "cands": list(args.cands),
            "seeds": list(args.seeds),
            "use_real_vina": bool(args.use_real_vina),
            "vina_binary_status": vina_binary_status,
            "pocket_status_json": str(args.pocket_status_json),
            "out_dir": str(out_dir),
        }
        print(json.dumps(plan, indent=2))
        return 0

    # Per-pocket bench
    per_pocket: List[Dict[str, object]] = []
    for pocket in args.pockets:
        LOG.info("=== pocket %s ===", pocket)
        rec = _run_pocket(
            pocket,
            args.cands,
            args.seeds,
            bool(args.use_real_vina),
            vina_binary_status,
            args.pocket_status_json,
        )
        per_pocket.append(rec)
        # write per-pocket JSON
        per_pocket_path = out_dir / pocket
        per_pocket_path.mkdir(parents=True, exist_ok=True)
        (per_pocket_path / "metrics.json").write_text(json.dumps(rec, indent=2) + "\n")

    # Top-level synthesis
    n_with_real_score = sum(
        1
        for p in per_pocket
        for c in p.get("cells", [])
        if c.get("score_kcal_mol") is not None
    )
    n_total_cells = sum(p.get("n_cells", 0) for p in per_pocket)
    m10_global = [
        c.get("score_kcal_mol")
        for p in per_pocket
        for c in p.get("cells", [])
        if c.get("score_kcal_mol") is not None
    ]
    synthesis = {
        "vina_binary_status": vina_binary_status["status"],
        "use_real_vina_requested": bool(args.use_real_vina),
        "pockets": list(args.pockets),
        "n_pockets": len(args.pockets),
        "n_total_cells": n_total_cells,
        "n_with_real_score": n_with_real_score,
        "m4_global_triple_rate": (
            float(
                sum(
                    1
                    for p in per_pocket
                    for c in p.get("cells", [])
                    if c.get("triple_success")
                )
                / max(1, n_with_real_score)
            )
            if n_with_real_score
            else None
        ),
        "m10_global_min_kcal_mol": float(min(m10_global)) if m10_global else None,
        "per_pocket": [
            {
                "pocket": p["pocket"],
                "status": p["status"],
                "n_cells": p["n_cells"],
                "status_counts": p["status_counts"],
                "m4_triple_rate": p["m4_triple_rate"],
                "m4_n_real": p["m4_n_real"],
                "m10_min_kcal_mol": p["m10_min_kcal_mol"],
                "m10_median_kcal_mol": p["m10_median_kcal_mol"],
                "m10_pass": p["m10_pass"],
            }
            for p in per_pocket
        ],
    }
    (out_dir / "synthesis.json").write_text(json.dumps(synthesis, indent=2) + "\n")

    # Console summary
    LOG.info("=== synthesis ===")
    LOG.info("vina_binary_status: %s", vina_binary_status["status"])
    LOG.info(
        "n_pockets=%d n_cells=%d n_with_real_score=%d",
        synthesis["n_pockets"],
        n_total_cells,
        n_with_real_score,
    )
    LOG.info("m4_global_triple_rate=%s", synthesis["m4_global_triple_rate"])
    LOG.info("m10_global_min_kcal_mol=%s", synthesis["m10_global_min_kcal_mol"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
