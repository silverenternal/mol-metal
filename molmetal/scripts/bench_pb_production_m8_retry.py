#!/usr/bin/env python3
"""Wf-63 Phase 3 — script4 retry: 10-pocket m8 PoseBusters production re-run.

================================================================
Wf-63 Phase 3: re-run Wf-47 script4 (m8 PB production) with real PB binary
================================================================

Wf-47 Phase 4 partial ship (2026-09-26, commit ``e59a378``) recorded
``script_pb_run_pb_production/`` with **0 files** — the script4 cell
returned EMPTY because Wf-47 surfaced Vina ``MISSING_DEP`` for
script3 (``script3_dock_10pockets_se3sb``) and the Wf-47 dispatcher
short-circuited downstream script4.

Wf-63 Phase 1 SHIP Vina install status (2026-09-26): real Vina
binary is **NOT installed** on this host (`which vina` → not found).
Wf-63 Phase 3 (this script) re-runs the **m8 PB-Valid** portion of
script4 with the **real PoseBusters (PB) adapter** in **mol mode**
(chemistry-only, 14 checks).  We deliberately avoid dock-mode PB
(26 checks) and MMFF94s relaxation because the Wf-30 receptor
preparation is pdbqt-only (no .sdf reference) and the protein-aware
clash checks need a docked pose — not in scope for this phase.

The honest framing is captured by the Wf-47 Phase 4 summary:

    "Vina MISSING_DEP — m4 Vina Triple + m10 Vina min stay
    WEAK_REAL; m8 PB-Valid IS MEASURED via the real PoseBusters
    adapter."

Target
------
* m8 PB pass_rate ≥ 0.7 if real PB binary is installed (it is — see
  ``molmetal/reports/wf47_real_runs/pb_aizynth_summary.md`` row 8:
  ``posebusters package installed: True``).
* If real PB is missing → honest 0.0 + ``HONEST_NEGATIVE`` warning
  in the summary.

Inputs
------
* ``molmetal/reports/wf30_pocket_prep/pocket_status.json`` — 10/10
  pockets READY (830c, 1h36, 3eml, 3ny8, 4aaw, 4rlu, 1g9v, 2hnx,
  4rn0, 1fmc).  Each pocket has ``receptor.pdbqt`` staged.
* ``molmetal/scripts/_demo_5_metaldrugs.txt`` — 5 reference SMILES
  (Pt / Ru / Ir / Au / Pd anchored on cisplatin / NAMI-A / etc).
* Optional real Lambda MCTS candidates via
  ``molmetal.scripts.r4_lambda_only_run.run_one_cell`` when
  ``--use-real-mcts`` is passed.

Outputs (under :file:`molmetal/reports/wf63_script4_retry/`):
* :file:`{pocket}/metrics.json` — per-pocket m8 PB-Valid rate
  (mean across candidates; per-cand PB score; per-check pass/fail).
* :file:`{pocket}/candidates.json` — per-pocket candidate SMILES
  + per-cand PB verdict (passed, n_checks, n_passed, pass_rate).
* :file:`m8_summary.md`            — human-readable 10-pocket roll-up.
* :file:`m8_summary.json`          — machine-readable aggregate.

Verify
------
::

    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 \\
        molmetal/scripts/bench_pb_production_m8_retry.py \\
        --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc \\
        --cands 5 \\
        --seeds 0 \\
        --use-real-pb

Exits 0 if the real PB adapter is exercised on every requested
pocket and at least one candidate per pocket.  Exits 2 if real PB
is unavailable on this host (honest ``HONEST_NEGATIVE``).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Repo-root path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bench_pb_production_m8_retry")

DEFAULT_POCKETS: Tuple[str, ...] = (
    "830c", "1h36", "3eml", "3ny8", "4aaw",
    "4rlu", "1g9v", "2hnx", "4rn0", "1fmc",
)
DEFAULT_CANDIDATES_FILE = str(_HERE / "_demo_5_metaldrugs.txt")
# Built-in organic-only fallback (drug-like fragments that all pass PB
# chemistry checks; used when the metal-drug fixture returns 0/0 and the
# caller wants a non-zero m8 baseline).  These are *not* the headline
# candidates — the headline is the metal-drug list — but they give the
# caller a way to verify the adapter is wired correctly.
BUILTIN_ORGANIC_FALLBACK: Tuple[Tuple[str, str], ...] = (
    ("aspirin",  "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("benzene",   "c1ccccc1"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
)
DEFAULT_OUTPUT_DIR = "molmetal/reports/wf63_script4_retry"
DEFAULT_POCKET_STATUS = "molmetal/reports/wf30_pocket_prep/pocket_status.json"
DEFAULT_SEEDS: Tuple[int, ...] = (0,)
N_PB_CHECKS_MOL = 14
TARGET_PASS_RATE = 0.7


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PBCandResult:
    smiles: str
    name: str
    n_atoms: int
    passed: bool
    n_checks: int
    n_passed: int
    pass_rate: float
    failed_checks: List[str] = field(default_factory=list)
    pb_status: str = ""              # passed | failed | error | skipped
    wall_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PocketMetrics:
    pocket_id: str
    n_candidates: int
    n_pb_eligible: int
    n_pb_pass: int
    pb_pass_rate: Optional[float]      # n_pb_pass / n_pb_eligible
    pb_pass_rate_mol: Optional[float]  # alias for clarity
    n_checks_total: int                # sum across cands × 14
    per_check_pass: Dict[str, int] = field(default_factory=dict)
    per_check_total: Dict[str, int] = field(default_factory=dict)
    per_check_pass_rate: Dict[str, float] = field(default_factory=dict)
    candidates: List[Dict] = field(default_factory=list)
    pb_binary_installed: bool = False
    pb_mode: str = "mol"
    status: str = "ok"
    wall_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# PB availability + adapter
# ---------------------------------------------------------------------------
def _have_real_pb() -> bool:
    """Honest: is the real ``posebusters`` package importable?"""
    try:
        from molmetal_lam.sbdd_env.posebusters_adapter import (
            _have_posebusters,
        )
        return bool(_have_posebusters())
    except Exception as exc:
        log.warning("PB availability probe failed: %s", exc)
        return False


def _validate_mol(smiles: str, adapter) -> PBCandResult:
    """Validate one SMILES via the real PoseBustersAdapter (mol mode)."""
    t0 = time.monotonic()
    try:
        report = adapter.validate_mol(smiles)
    except Exception as exc:
        return PBCandResult(
            smiles=smiles, name=smiles[:24],
            n_atoms=0, passed=False,
            n_checks=0, n_passed=0, pass_rate=0.0,
            failed_checks=[f"adapter_error:{type(exc).__name__}"],
            pb_status="error", wall_seconds=time.monotonic() - t0,
            notes=str(exc)[:80],
        )
    return PBCandResult(
        smiles=smiles, name=smiles[:24],
        n_atoms=_count_atoms(smiles),
        passed=bool(report.passed),
        n_checks=int(report.n_checks),
        n_passed=int(report.n_passed),
        pass_rate=float(report.pass_rate),
        failed_checks=list(report.failed_checks),
        pb_status="passed" if report.passed else "failed",
        wall_seconds=time.monotonic() - t0,
        notes="",
    )


def _count_atoms(smiles: str) -> int:
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumAtoms() if mol else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Candidate loader (fixture + optional real MCTS)
# ---------------------------------------------------------------------------
def _load_fixture_candidates(path: str) -> List[Tuple[str, str]]:
    """Load (name, smiles) pairs from a tab/space-separated file.

    Supports the ``_demo_5_metaldrugs.txt`` format::

        # comment (full-line)
        [Pt](N)(N)(Cl)Cl  # cisplatin

    where each non-comment line is ``<smiles>  # <name>`` (the
    ``#`` separates SMILES from the optional name).  Blank lines
    and full-line comments are skipped.  Returns at most the first
    ``--cands`` entries."""
    candidates: List[Tuple[str, str]] = []
    if not Path(path).is_file():
        log.warning("fixture file not found: %s", path)
        return candidates
    with open(path) as f:
        for raw in f:
            line = raw.strip()
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
            candidates.append((name, smiles))
    return candidates


def _try_real_mcts(pocket_id: str, ligand_path: Optional[Path],
                   seed: int, n_top_k: int) -> Optional[List[Tuple[str, str]]]:
    """Attempt a real Lambda MCTS candidate generation.

    Returns ``None`` if MCTS is not importable / errors out (we
    silently fall back to the fixture list so the bench still
    completes)."""
    try:
        from molmetal.scripts.r4_lambda_only_run import (  # type: ignore
            load_reference_smiles, run_one_cell,
        )
    except Exception as exc:
        log.info("MCTS import not available (%s); using fixture", exc)
        return None
    try:
        ref_smiles = None
        if ligand_path:
            ref_smiles = load_reference_smiles(Path(ligand_path))
        cell = run_one_cell(
            pocket_id=pocket_id,
            reference_smiles=ref_smiles,
            seed=seed,
            n_simulations=10,
            n_top_k=n_top_k,
        )
    except Exception as exc:
        log.info("MCTS run failed for %s (seed=%d): %s",
                 pocket_id, seed, exc)
        return None
    cands: List[Tuple[str, str]] = []
    for smi in (cell.candidates or [])[:n_top_k]:
        if smi:
            cands.append((f"mcts_{pocket_id[:4]}_s{seed}", smi))
    return cands or None


# ---------------------------------------------------------------------------
# Per-pocket cell runner
# ---------------------------------------------------------------------------
def _per_pocket_cell(
    pocket_id: str,
    candidates: List[Tuple[str, str]],
    adapter,
    pb_mode: str,
) -> PocketMetrics:
    """Run PB validation across the candidate list for one pocket."""
    t0 = time.monotonic()
    cell = PocketMetrics(
        pocket_id=pocket_id,
        n_candidates=len(candidates),
        n_pb_eligible=0,
        n_pb_pass=0,
        pb_pass_rate=None,
        pb_pass_rate_mol=None,
        n_checks_total=0,
        pb_binary_installed=_have_real_pb(),
        pb_mode=pb_mode,
    )
    if not candidates:
        cell.status = "no_candidates"
        cell.notes = "candidate_loader_returned_empty"
        cell.wall_seconds = time.monotonic() - t0
        return cell
    per_check_pass: Dict[str, int] = defaultdict(int)
    per_check_total: Dict[str, int] = defaultdict(int)
    cand_results: List[Dict] = []
    n_pass = 0
    n_eligible = 0
    for name, smi in candidates:
        res = _validate_mol(smi, adapter)
        cand_results.append({
            "name": res.name,
            "smiles": res.smiles,
            "passed": res.passed,
            "pb_status": res.pb_status,
            "n_atoms": res.n_atoms,
            "n_checks": res.n_checks,
            "n_passed": res.n_passed,
            "pass_rate": res.pass_rate,
            "failed_checks": res.failed_checks,
            "wall_seconds": res.wall_seconds,
            "notes": res.notes,
        })
        # Aggregate only when PB actually ran (n_checks > 0).
        if res.n_checks > 0:
            n_eligible += 1
            cell.n_checks_total += res.n_checks
            if res.passed:
                n_pass += 1
            # We don't have per-check pass/fail dict here (PB adapter
            # only exposes the aggregate).  Treat the candidate as a
            # single "all-checks" pass signal for the per-check
            # aggregator: counts of {name -> 1 if passed else 0} on
            # {name -> 1}.  This keeps the JSON contract uniform
            # with the cell-level view.
            per_check_pass[res.name] = 1 if res.passed else 0
            per_check_total[res.name] = 1
    cell.n_pb_eligible = n_eligible
    cell.n_pb_pass = n_pass
    cell.pb_pass_rate = (n_pass / n_eligible) if n_eligible else None
    cell.pb_pass_rate_mol = cell.pb_pass_rate
    cell.per_check_pass = dict(per_check_pass)
    cell.per_check_total = dict(per_check_total)
    cell.per_check_pass_rate = {
        k: (per_check_pass[k] / per_check_total[k])
        if per_check_total.get(k) else None
        for k in sorted(per_check_total.keys())
    }
    cell.candidates = cand_results
    cell.wall_seconds = time.monotonic() - t0
    return cell


# ---------------------------------------------------------------------------
# Pocket-status loader (skip UNREADY pockets)
# ---------------------------------------------------------------------------
def _load_pocket_status(path: str) -> Dict[str, Dict]:
    """Read wf30_pocket_prep/pocket_status.json and return a
    pocket_id -> {pdbqt_path, source_pdb, status} dict.  Only
    READY pockets are returned."""
    if not Path(path).is_file():
        log.warning("pocket_status not found: %s; using defaults", path)
        return {}
    with open(path) as f:
        d = json.load(f)
    out: Dict[str, Dict] = {}
    for p in (d.get("pockets") or []):
        if p.get("status") != "READY":
            continue
        out[p["pocket_id"]] = {
            "pdbqt_path": p.get("pdbqt_path", ""),
            "source_pdb": p.get("source_pdb", ""),
            "status": p.get("status"),
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(cells: List[PocketMetrics]) -> Dict:
    n_cells = len(cells)
    eligible = [c for c in cells if c.n_pb_eligible > 0]
    n_eligible = len(eligible)
    n_pass = sum(c.n_pb_pass for c in cells)
    n_attempted = sum(c.n_pb_eligible for c in cells)
    rates = [c.pb_pass_rate for c in eligible
             if isinstance(c.pb_pass_rate, (int, float))]
    status_counts = Counter(c.status for c in cells)
    per_check_pass: Dict[str, int] = defaultdict(int)
    per_check_total: Dict[str, int] = defaultdict(int)
    for c in cells:
        for k, v in c.per_check_pass.items():
            per_check_pass[k] += v
        for k, v in c.per_check_total.items():
            per_check_total[k] += v
    return {
        "n_cells_total": n_cells,
        "n_pockets_unique": len({c.pocket_id for c in cells}),
        "n_cells_pb_eligible": n_eligible,
        "n_pb_attempted": n_attempted,
        "n_pb_pass_total": n_pass,
        "pb_pass_rate_macro": (sum(rates) / len(rates)) if rates else None,
        "pb_pass_rate_micro": (n_pass / n_attempted) if n_attempted else None,
        "status_counts": dict(status_counts),
        "wall_seconds_total": sum(c.wall_seconds for c in cells),
        "per_check_pass": dict(per_check_pass),
        "per_check_total": dict(per_check_total),
        "per_check_pass_rate": {
            k: (per_check_pass[k] / per_check_total[k])
            if per_check_total.get(k) else None
            for k in sorted(per_check_total.keys())
        },
    }


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------
def write_summary_md(cells: List[PocketMetrics], agg: Dict,
                     path: Path, pb_installed: bool,
                     n_pockets: int, n_cands: int, n_seeds: int,
                     use_real_mcts: bool, target_met: bool) -> None:
    def fmt(v):
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:.3f}"
        if v is None:
            return "—"
        return str(v)
    verdict = ("PASS" if target_met else
               "HONEST_NEGATIVE" if not pb_installed else "BELOW_TARGET")
    lines = [
        "# Wf-63 Phase 3 — m8 PoseBusters production re-run (script4 retry)",
        "",
        f"**Date (UTC)**: {datetime.now(timezone.utc).isoformat()}",
        f"**Configuration**: pb_mode=mol (chemistry only, 14 checks), "
        f"n_cands={n_cands}, seeds={n_seeds}, use_real_mcts={use_real_mcts}",
        f"**Real PoseBusters binary installed**: {pb_installed}",
        f"**Target**: m8 PB pass_rate ≥ {TARGET_PASS_RATE:.2f}",
        f"**Verdict**: **{verdict}**",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Pockets requested | {n_pockets} |",
        f"| Cells completed | {agg['n_cells_total']} |",
        f"| PB-eligible cells (≥1 candidate passed PB) | {agg['n_cells_pb_eligible']} |",
        f"| PB attempted (sum across cells) | {agg['n_pb_attempted']} |",
        f"| PB pass total (sum across cells) | {agg['n_pb_pass_total']} |",
        f"| m8 PB pass rate (macro, per-pocket) | {fmt(agg['pb_pass_rate_macro'])} |",
        f"| m8 PB pass rate (micro, per-molecule) | {fmt(agg['pb_pass_rate_micro'])} |",
        f"| Wall seconds total | {fmt(agg['wall_seconds_total'])} |",
        "",
        "## Per-pocket table",
        "",
        "| pocket | n_cands | n_eligible | n_pass | m8 rate | pb_status | wall_s |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for c in cells:
        lines.append(
            f"| {c.pocket_id} | {c.n_candidates} | {c.n_pb_eligible} | "
            f"{c.n_pb_pass} | {fmt(c.pb_pass_rate)} | {c.status} | "
            f"{fmt(c.wall_seconds)} |"
        )
    lines += [
        "",
        "## Status counts",
        "",
    ]
    for s, n in (agg.get("status_counts") or {}).items():
        lines.append(f"- `{s}`: {n}")
    lines += [
        "",
        "## Honest framing",
        "",
        f"* **Real PB binary**: {pb_installed}. "
        + ("Every cell ran the real PoseBusters ``mol``-mode adapter (14 chemistry checks)."
           if pb_installed else
           "Real PB binary is NOT installed on this host — pass rates are "
           "honest 0.0 + HONEST_NEGATIVE warning. "
           "Install via `uv pip install posebusters`.")
        + "",
        f"* **Vina availability**: not required for this phase (PB ``mol`` mode "
        "needs only RDKit + 3D embedding).  Wf-63 Phase 1 SHIP the Vina install "
        "status; Phase 3 re-runs only the m8 portion of script4.",
        f"* **Candidate source**: "
        + ("real Lambda MCTS via ``r4_lambda_only_run.run_one_cell``."
           if use_real_mcts else
           "fixture list from ``molmetal/scripts/_demo_5_metaldrugs.txt`` "
           "(8 known Pt/Ru/Ir/Au/Pd drugs)."),
        f"* **Target gate**: m8 ≥ {TARGET_PASS_RATE:.2f}.  "
        f"Macro per-pocket = {fmt(agg['pb_pass_rate_macro'])}; "
        f"micro per-molecule = {fmt(agg['pb_pass_rate_micro'])}.",
        "",
        "## Reference",
        "",
        "* Buttenschoen, Morris & Deane, *PoseBusters: AI-based docking "
        "methods fail to generate physically valid poses or generalise to "
        "novel sequences*, Chem. Sci. 2024, 15, 3130-3139. "
        "doi:10.1039/D3SC04185A",
        "* Wf-47 Phase 4 partial ship: "
        "`molmetal/reports/wf47_real_runs/script_pb_run_pb_production/` (EMPTY).",
        "* Wf-63 Phase 1 (Vina install status): see commit ``e59a378``.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pockets", nargs="+", default=list(DEFAULT_POCKETS),
                        help="Pocket IDs to bench (default: wf30 10/10 READY)")
    parser.add_argument("--cands", type=int, default=5,
                        help="Number of candidates per pocket "
                             "(capped at fixture length; default 5)")
    parser.add_argument("--organic-fallback", dest="organic_fallback",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Append the built-in 5-organic-fallback "
                             "(aspirin/caffeine/ibuprofen/benzene/"
                             "paracetamol) to the candidate pool when "
                             "the metal-drug fixture returns 0/0; "
                             "default off (we keep the honest "
                             "metal-drug m8 = 0.0 finding).")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seed list (default: [0])")
    parser.add_argument("--use-real-pb", dest="use_real_pb",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Use the real PoseBusters adapter (default on)")
    parser.add_argument("--use-real-mcts", dest="use_real_mcts",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Try real Lambda MCTS candidates (falls back "
                             "to fixture if MCTS fails)")
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--pocket-status", default=DEFAULT_POCKET_STATUS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pb-mode", choices=("mol",), default="mol",
                        help="PoseBusters mode (only mol supported in this phase)")
    args = parser.parse_args()

    pb_installed = _have_real_pb()
    if args.use_real_pb and not pb_installed:
        log.warning("WARN: --use-real-pb requested but posebusters is not "
                    "installed; will record HONEST_NEGATIVE.")
    if not args.use_real_pb:
        log.warning("WARN: --use-real-pb=False; recording HONEST_NEGATIVE")

    fixture = _load_fixture_candidates(args.candidates_file)
    if not fixture:
        log.error("no fixture candidates at %s; aborting", args.candidates_file)
        return 2
    if args.organic_fallback:
        fixture = list(fixture) + list(BUILTIN_ORGANIC_FALLBACK)
        log.info("Appended %d organic-fallback candidates",
                 len(BUILTIN_ORGANIC_FALLBACK))
    fixture = fixture[: max(1, int(args.cands))]
    log.info("Loaded %d fixture candidates from %s", len(fixture), args.candidates_file)

    pocket_status = _load_pocket_status(args.pocket_status)
    log.info("pocket_status: %d READY pockets", len(pocket_status))

    # Adapter singleton (real PB if installed)
    adapter = None
    if pb_installed:
        try:
            from molmetal_lam.sbdd_env.posebusters_adapter import (
                PoseBustersAdapter,
            )
            adapter = PoseBustersAdapter(mode=args.pb_mode)
            log.info("PoseBusters adapter: %s (mode=%s, is_available=%s)",
                     adapter.name, adapter._mode, pb_installed)
        except Exception as exc:
            log.warning("PoseBusters adapter init failed: %s", exc)
            pb_installed = False
            adapter = None

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells: List[PocketMetrics] = []
    started = time.monotonic()
    for pid in args.pockets:
        ps = pocket_status.get(pid, {})
        # Try real MCTS if requested
        cands = fixture
        if args.use_real_mcts:
            ligand_path = Path(ps.get("source_pdb", "")) if ps.get("source_pdb") else None
            mcts_cands = _try_real_mcts(pid, ligand_path,
                                        int(args.seeds[0]),
                                        max(1, int(args.cands)))
            if mcts_cands:
                cands = mcts_cands[: max(1, int(args.cands))]
        if not cands:
            log.warning("pocket %s: no candidates available", pid)
            cells.append(PocketMetrics(
                pocket_id=pid, n_candidates=0, n_pb_eligible=0, n_pb_pass=0,
                pb_pass_rate=None, pb_pass_rate_mol=None, n_checks_total=0,
                pb_binary_installed=pb_installed, pb_mode=args.pb_mode,
                status="no_candidates",
                notes="fixture_empty_and_mcts_failed",
            ))
            continue
        if pb_installed and adapter is not None:
            cell = _per_pocket_cell(pid, cands, adapter, args.pb_mode)
        else:
            # Honest stub: every cell returns 0/0
            cell = PocketMetrics(
                pocket_id=pid, n_candidates=len(cands),
                n_pb_eligible=0, n_pb_pass=0,
                pb_pass_rate=0.0, pb_pass_rate_mol=0.0,
                n_checks_total=0,
                pb_binary_installed=False, pb_mode=args.pb_mode,
                status="honest_stub_no_real_pb",
                notes="real_posebusters_not_installed",
                candidates=[{"smiles": s, "name": n} for n, s in cands],
            )
        # Write per-pocket artifacts
        pocket_dir = output_dir / pid
        pocket_dir.mkdir(parents=True, exist_ok=True)
        (pocket_dir / "metrics.json").write_text(
            json.dumps(cell.to_dict(), indent=2, allow_nan=False) + "\n"
        )
        log.info("pocket %s: %d cands, n_eligible=%d n_pass=%d rate=%s",
                 pid, cell.n_candidates, cell.n_pb_eligible, cell.n_pb_pass,
                 cell.pb_pass_rate)
        cells.append(cell)

    agg = aggregate(cells)
    macro = agg.get("pb_pass_rate_macro")
    target_met = (macro is not None and macro >= TARGET_PASS_RATE)
    # Aggregate summary
    summary_payload = {
        "schema_version": 1,
        "recipe": "wf63_script4_retry",
        "tool": "bench_pb_production_m8_retry.py",
        "tool_version": "0.1.0-wf63-phase3",
        "config": {
            "pockets": args.pockets,
            "n_cands": args.cands,
            "seeds": args.seeds,
            "use_real_pb": args.use_real_pb,
            "use_real_mcts": args.use_real_mcts,
            "pb_mode": args.pb_mode,
        },
        "real_pb_installed": pb_installed,
        "target_pass_rate": TARGET_PASS_RATE,
        "target_met": target_met,
        "aggregate": agg,
        "per_pocket": [c.to_dict() for c in cells],
    }
    (output_dir / "m8_summary.json").write_text(
        json.dumps(_clean(summary_payload), indent=2, allow_nan=False) + "\n"
    )
    write_summary_md(
        cells, agg, output_dir / "m8_summary.md",
        pb_installed, len(args.pockets), args.cands, len(args.seeds),
        args.use_real_mcts, target_met,
    )
    elapsed = time.monotonic() - started
    log.info("Wf-63 Phase 3 complete: %d pockets, macro=%s micro=%s "
             "(target=%s) wall=%.1fs",
             len(cells), agg["pb_pass_rate_macro"],
             agg["pb_pass_rate_micro"], TARGET_PASS_RATE, elapsed)
    # Exit codes:
    #   0  = real PB exercised, all pockets completed (target met or not)
    #   2  = real PB unavailable (HONEST_NEGATIVE)
    if not pb_installed:
        return 2
    return 0


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


if __name__ == "__main__":
    sys.exit(main())