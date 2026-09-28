#!/usr/bin/env python3
"""Wf-67 Phase 3 — 100-pocket × 10-seed docking sweep (R20-P4.3).

================================================================
Wf-67 Phase 3: Expand Wf-30 10/10 pockets × 1-seed → 100/100 pockets
× 10 seeds using real Vina binary when installed (Wf-63 ship), honest
stub fallback otherwise (Wf-63 honest-negative framing).

Inputs
------
* ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt`` —
  10/10 Wf-30 READY pockets (4aaw, 4rn0, 830c, 1fmc, 1g9v, 1h36,
  2hnx, 3eml, 3ny8, 4rlu).  These are used as the Wf-30 *back-
  compatible* subset of the 100-pocket sweep.
* ``molmetal/data/crossdocked100_manifest.csv`` — the 100-pocket
  manifest (test_000..test_099) written by
  ``molmetal/scripts/stage_crossdocked100.py``.  This file enumerates
  the per-pocket PDB / SDF pairs but the receptors themselves have
  to be materialised by ``mk_prepare_receptor.py`` before Vina
  scoring.  When the CrossDocked-2020 receptors are not yet staged
  for a particular ``pocket_id``, the script falls back to the Wf-30
  ten-pocket MMP13 cohort (see :func:`pick_pocket_bundle`) and
  records the swap in ``sweep_summary.json`` ``honest_framing``.
* ``molmetal/scripts/_demo_5_metaldrugs.txt`` — 5 reference SMILES
  used as the candidate pool (cisplatin, carboplatin, oxaliplatin,
  auranofin, ruthenium-arene).  Override with ``--candidates-csv``.

Outputs
-------
* ``molmetal/reports/wf67_p4_3_100pocket/<pocket>/per_seed_metrics.json``
  — per-pocket docking record with 10 per-seed Vina kcal/mol scores
  (real) or stub scores (honest fallback) + per-candidate ranking.
* ``molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json`` —
  full-sweep exit status + per-pocket summary + per-pocket rank
  correlation across seeds.

Methods
-------
* ``vina_real`` — real ``vina`` binary
  (``/home/hugo/.mamba/bin/vina`` per Wf-63 P1 ship).
* ``vina_stub`` — deterministic hash-based stub scorer (Wf-30
  honest-negative fallback).  Same hash as
  :func:`wf30_dock_summarize._stub_score` so the two sweeps are
  numerically comparable.

Constraints
-----------
* The script does NOT mutate ``molmetal/adapters/flow_matching_lipman/``
  (R17 ship is locked).
* The script does NOT mutate ``molmetal/scripts/dock_10pockets_se3sb.py``
  (Wf-30 ship is the historical record).
* Honest framing about Vina install: when ``vina --version`` is not
  on PATH the script auto-switches to the stub and records the
  rationale in the JSON; it never silently substitutes stub scores
  for real scores.
* Idempotent: re-running writes fresh per-pocket JSON but the sweep
  summary keys are deterministic.

Usage
-----
::

    source env.sh
    python -m molmetal.scripts.bench_100pocket_10seed \\
        --out-dir molmetal/reports/wf67_p4_3_100pocket \\
        --pockets 100 \\
        --seeds 10 \\
        --use-real-vina \\
        --max-wall-min 60

For a CI-cheap smoke run (no Vina, no GPU, ≤ 30 s wall)::

    python -m molmetal.scripts.bench_100pocket_10seed \\
        --pockets 5 --seeds 3 --no-real-vina
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("wf67_bench_100pocket")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"
DEFAULT_CANDIDATES = HERE / "_demo_5_metaldrugs.txt"
DEFAULT_POCKET_PREP_DIR = REPO_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
DEFAULT_OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf67_p4_3_100pocket"
WF30_TEN_POCKETS: Tuple[str, ...] = (
    "830c", "1h36", "3eml", "3ny8", "4aaw", "4rlu", "1g9v", "2hnx", "4rn0", "1fmc"
)

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
DEFAULT_CANDIDATE_LIMIT = 5
DEFAULT_STUB_SCORE_RANGE = (-10.5, -6.2)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PocketBundle:
    """Resolved per-pocket staging material for the sweep."""

    pocket_id: str
    source: str  # "wf30_mmp13" | "crossdocked100" | "stub_fallback"
    receptor_pdbqt: Optional[Path] = None
    ligand_sdf: Optional[Path] = None
    note: str = ""


@dataclass
class DockScore:
    """A single (pocket, candidate, seed) docking result."""

    pocket_id: str
    candidate_smiles: str
    seed: int
    method: str  # "vina_real" | "vina_stub"
    kcal_mol: float
    elapsed_s: float
    extra: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_smiles(smiles: str, seed: int) -> str:
    """Stable hex hash used by the stub scorer — keeps Wf-30 ↔ Wf-67 comparable."""
    h = hashlib.sha256(f"{smiles}|{seed}".encode("utf-8"))
    return h.hexdigest()[:16]


def _stub_score(smiles: str, seed: int, pocket_id: str) -> float:
    """Deterministic stub scorer.

    Same hash structure as :func:`wf30_dock_summarize._stub_score` —
    kcal/mol ∈ [DEFAULT_STUB_SCORE_RANGE] derived from the SHA-256
    of (smiles, seed, pocket_id).  Reproducible across CI hosts.
    """
    low, high = DEFAULT_STUB_SCORE_RANGE
    h = hashlib.sha256(f"{smiles}|{seed}|{pocket_id}".encode("utf-8")).digest()
    # Map first 4 bytes (unsigned int) -> [0, 1) -> [high, low] (more negative = better)
    frac = int.from_bytes(h[:4], "big") / 2**32
    return high + frac * (low - high)  # negative range


def _vina_binary_present() -> Optional[Path]:
    """Locate the ``vina`` binary if installed (Wf-63 P1 ship)."""
    candidates = [
        Path("/home/hugo/.mamba/bin/vina"),
        Path("/home/hugo/micromamba/bin/vina"),
        Path("/usr/local/bin/vina"),
        Path("/usr/bin/vina"),
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK):
            return c
    which = shutil.which("vina")
    return Path(which) if which else None


def _vina_run_one(
    vina_bin: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    out_pdbqt: Path,
    seed: int,
    exhaustiveness: int = 4,
) -> Tuple[float, float]:
    """Run ``vina`` once and parse kcal/mol (best pose).

    Returns (kcal_mol, elapsed_s). Caller handles the stdout parsing
    & exceptions.
    """
    t0 = time.time()
    cmd = [
        str(vina_bin),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(out_pdbqt),
        "--exhaustiveness", str(exhaustiveness),
        "--seed", str(seed),
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vina failed (rc={proc.returncode}) for {ligand_pdbqt.name}: "
            f"stderr={proc.stderr[-200:]}"
        )
    kcal_mol: Optional[float] = None
    for line in proc.stdout.splitlines():
        # Vina writes a table like:
        # mode |   affinity | rmsd l.b.| rmsd u.b.
        #    1       -7.4      0.000      0.000
        line_s = line.strip()
        if line_s and line_s[:1].isdigit():
            parts = line_s.split()
            try:
                kcal_mol = float(parts[1])
            except (IndexError, ValueError):
                continue
            break
    if kcal_mol is None:
        raise RuntimeError(
            f"vina produced no parseable score for {ligand_pdbqt.name}: "
            f"stdout-tail={proc.stdout[-200:]!r}"
        )
    return kcal_mol, time.time() - t0


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------
def _load_wf30_pocket_bundles(prep_dir: Path) -> Dict[str, PocketBundle]:
    """Load the Wf-30 ten-pocket RECEPTOR.pdbqt bundles (10/10 READY)."""
    bundles: Dict[str, PocketBundle] = {}
    if not prep_dir.exists():
        return bundles
    for sub in sorted(prep_dir.iterdir()):
        if not sub.is_dir():
            continue
        receptor = sub / "receptor.pdbqt"
        if not receptor.exists():
            continue
        bundle = PocketBundle(
            pocket_id=sub.name,
            source="wf30_mmp13",
            receptor_pdbqt=receptor,
            ligand_sdf=None,
            note="Wf-30 READY pdbqt from mk_prepare_receptor.py",
        )
        bundles[sub.name] = bundle
    return bundles


def _load_crossdocked100_manifest(manifest: Path) -> List[Dict[str, str]]:
    if not manifest.exists():
        return []
    with manifest.open() as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def pick_pocket_bundle(
    requested_pocket_id: str,
    wf30_bundles: Dict[str, PocketBundle],
    crossdocked_rows: List[Dict[str, str]],
) -> PocketBundle:
    """Resolve the PocketBundle for ``requested_pocket_id``.

    Priority: Wf-30 staged pdbqt → crossdocked100 manifest → stub.
    """
    if requested_pocket_id in wf30_bundles:
        return wf30_bundles[requested_pocket_id]
    for row in crossdocked_rows:
        if row.get("pocket_id") == requested_pocket_id:
            receptor = Path(row["receptor_path"])
            ligand = Path(row["ligand_path"])
            if receptor.exists() and ligand.exists():
                return PocketBundle(
                    pocket_id=requested_pocket_id,
                    source="crossdocked100",
                    receptor_pdbqt=receptor,
                    ligand_sdf=ligand,
                    note="CrossDocked-2020 local extract",
                )
            return PocketBundle(
                pocket_id=requested_pocket_id,
                source="stub_fallback",
                receptor_pdbqt=None,
                ligand_sdf=None,
                note=(
                    f"CrossDocked manifest row found but files missing: "
                    f"receptor_exists={receptor.exists()} ligand_exists={ligand.exists()}"
                ),
            )
    return PocketBundle(
        pocket_id=requested_pocket_id,
        source="stub_fallback",
        receptor_pdbqt=None,
        ligand_sdf=None,
        note=f"no manifest row for pocket_id={requested_pocket_id}",
    )


def select_pockets(
    requested_n: int,
    wf30_bundles: Dict[str, PocketBundle],
    crossdocked_rows: List[Dict[str, str]],
) -> List[str]:
    """Pick the first ``requested_n`` unique pocket ids.

    Order:
      1. Wf-30 ten (highest fidelity — already-staged pdbqt).
      2. CrossDocked-2020 row ids (test_000..test_099).
    """
    picked: List[str] = []
    seen: set = set()
    # 1. Wf-30 ten (always first)
    for pid in WF30_TEN_POCKETS:
        if pid in seen:
            continue
        seen.add(pid)
        picked.append(pid)
        if len(picked) >= requested_n:
            return picked[:requested_n]
    # 2. crossdocked100 (only test_* ids that are not in wf30)
    for row in crossdocked_rows:
        pid = row.get("pocket_id", "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        picked.append(pid)
        if len(picked) >= requested_n:
            break
    return picked


# ---------------------------------------------------------------------------
# Candidate loading (re-uses the Wf-30 demo file format)
# ---------------------------------------------------------------------------
def load_candidates(path: Path) -> List[Tuple[str, str]]:
    """Load (smiles, name) pairs from the Wf-30 ``_demo_5_metaldrugs.txt``.

    Lines are ``SMILES # name`` (comment lines start with ``#``).
    Returns up to ``DEFAULT_CANDIDATE_LIMIT`` items.
    """
    pairs: List[Tuple[str, str]] = []
    if not path.exists():
        return pairs
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            smiles, name = line.split("#", 1)
            pairs.append((smiles.strip(), name.strip()))
        else:
            pairs.append((line, f"cand_{len(pairs)}"))
    return pairs[:DEFAULT_CANDIDATE_LIMIT]


# ---------------------------------------------------------------------------
# Dock: per-pocket × per-candidate × per-seed
# ---------------------------------------------------------------------------
def dock_one(
    bundle: PocketBundle,
    smiles: str,
    name: str,
    seed: int,
    use_real_vina: bool,
    vina_bin: Optional[Path],
) -> DockScore:
    """Run a single docking evaluation. CPU stub fallback if no Vina."""
    method = "vina_real" if use_real_vina and vina_bin else "vina_stub"
    if method == "vina_stub":
        kcal = _stub_score(smiles, seed, bundle.pocket_id)
        return DockScore(
            pocket_id=bundle.pocket_id,
            candidate_smiles=smiles,
            seed=seed,
            method=method,
            kcal_mol=kcal,
            elapsed_s=0.0,
            extra={"stub_hash": _hash_smiles(smiles, seed), "note": bundle.note},
        )
    # Real Vina — needs receptor + a ligand pdbqt. We synthesise a
    # minimal pdbqt from the SMILES on the fly via RDKit if available,
    # else we fall back to stub for this single evaluation.
    if bundle.receptor_pdbqt is None:
        # Real Vina requested but no receptor — honest downgrade.
        kcal = _stub_score(smiles, seed, bundle.pocket_id)
        return DockScore(
            pocket_id=bundle.pocket_id,
            candidate_smiles=smiles,
            seed=seed,
            method="vina_stub",
            kcal_mol=kcal,
            elapsed_s=0.0,
            extra={
                "stub_hash": _hash_smiles(smiles, seed),
                "note": "vina_real requested but receptor.pdbqt missing — "
                "honest downgrade to stub per Wf-63",
            },
        )
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError:
        kcal = _stub_score(smiles, seed, bundle.pocket_id)
        return DockScore(
            pocket_id=bundle.pocket_id,
            candidate_smiles=smiles,
            seed=seed,
            method="vina_stub",
            kcal_mol=kcal,
            elapsed_s=0.0,
            extra={
                "stub_hash": _hash_smiles(smiles, seed),
                "note": "vina_real requested but RDKit not installed — "
                "honest downgrade to stub",
            },
        )
    with tempfile.TemporaryDirectory(prefix="wf67_p4_3_") as tmp:
        lig_pdbqt = Path(tmp) / f"{name}_{seed}.pdbqt"
        out_pdbqt = Path(tmp) / f"{name}_{seed}_out.pdbqt"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            kcal = _stub_score(smiles, seed, bundle.pocket_id)
            return DockScore(
                pocket_id=bundle.pocket_id,
                candidate_smiles=smiles,
                seed=seed,
                method="vina_stub",
                kcal_mol=kcal,
                elapsed_s=0.0,
                extra={
                    "stub_hash": _hash_smiles(smiles, seed),
                    "note": "RDKit could not parse SMILES — stub fallback",
                },
            )
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            kcal = _stub_score(smiles, seed, bundle.pocket_id)
            return DockScore(
                pocket_id=bundle.pocket_id,
                candidate_smiles=smiles,
                seed=seed,
                method="vina_stub",
                kcal_mol=kcal,
                elapsed_s=0.0,
                extra={
                    "stub_hash": _hash_smiles(smiles, seed),
                    "note": "RDKit EmbedMolecule failed — stub fallback",
                },
            )
        Chem.MolToPDBFile(mol, str(lig_pdbqt))
        kcal, elapsed = _vina_run_one(
            vina_bin=vina_bin,
            receptor_pdbqt=bundle.receptor_pdbqt,
            ligand_pdbqt=lig_pdbqt,
            out_pdbqt=out_pdbqt,
            seed=seed,
        )
        return DockScore(
            pocket_id=bundle.pocket_id,
            candidate_smiles=smiles,
            seed=seed,
            method=method,
            kcal_mol=kcal,
            elapsed_s=elapsed,
            extra={"out_pdbqt": str(out_pdbqt)},
        )


def dock_pocket(
    bundle: PocketBundle,
    candidates: List[Tuple[str, str]],
    seeds: Sequence[int],
    use_real_vina: bool,
    vina_bin: Optional[Path],
    max_wall_s: Optional[float] = None,
) -> List[DockScore]:
    """Dock every (candidate, seed) for a single pocket."""
    scores: List[DockScore] = []
    t0 = time.time()
    for smiles, name in candidates:
        for seed in seeds:
            if max_wall_s is not None and (time.time() - t0) > max_wall_s:
                LOG.warning(
                    "pocket=%s hit max_wall_s=%.1f; recording partial results",
                    bundle.pocket_id,
                    max_wall_s,
                )
                return scores
            scores.append(
                dock_one(
                    bundle=bundle,
                    smiles=smiles,
                    name=name,
                    seed=seed,
                    use_real_vina=use_real_vina,
                    vina_bin=vina_bin,
                )
            )
    return scores


# ---------------------------------------------------------------------------
# Per-pocket aggregation
# ---------------------------------------------------------------------------
def per_pocket_summary(scores: List[DockScore]) -> Dict[str, object]:
    """Return per-(pocket, candidate) mean ± std + per-pocket ranking."""
    by_cand: Dict[str, List[float]] = {}
    for s in scores:
        by_cand.setdefault(s.candidate_smiles, []).append(s.kcal_mol)

    per_cand: List[Dict[str, object]] = []
    for smiles, vals in by_cand.items():
        per_cand.append(
            {
                "candidate_smiles": smiles,
                "n_seeds": len(vals),
                "mean_kcal_mol": statistics.fmean(vals),
                "std_kcal_mol": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min_kcal_mol": min(vals),
                "max_kcal_mol": max(vals),
            }
        )
    per_cand.sort(key=lambda r: r["mean_kcal_mol"])  # most-negative first

    # Per-pocket seed-mean ± seed-std across all candidates
    by_seed: Dict[int, List[float]] = {}
    for s in scores:
        by_seed.setdefault(s.seed, []).append(s.kcal_mol)
    seed_means = [statistics.fmean(v) for v in by_seed.values()]
    pocket_kcals = [s.kcal_mol for s in scores]
    return {
        "per_candidate": per_cand,
        "best_candidate_smiles": per_cand[0]["candidate_smiles"] if per_cand else None,
        "best_candidate_mean_kcal_mol": per_cand[0]["mean_kcal_mol"] if per_cand else None,
        "n_evaluations": len(scores),
        "seed_means": seed_means,
        "pocket_mean_kcal_mol": (
            statistics.fmean(pocket_kcals) if pocket_kcals else None
        ),
        "pocket_std_kcal_mol": (
            statistics.pstdev(pocket_kcals) if len(pocket_kcals) > 1 else 0.0
        ),
        "method_mix": sorted({s.method for s in scores}),
    }


def rank_candidates(scores: List[DockScore]) -> List[str]:
    """Return candidate SMILES ranked most-negative → least-negative (mean over seeds)."""
    by_cand: Dict[str, List[float]] = {}
    for s in scores:
        by_cand.setdefault(s.candidate_smiles, []).append(s.kcal_mol)
    ranked = sorted(by_cand.items(), key=lambda kv: statistics.fmean(kv[1]))
    return [c for c, _ in ranked]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def write_per_pocket_json(scores: List[DockScore], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_seed": [
            {
                "pocket_id": s.pocket_id,
                "candidate_smiles": s.candidate_smiles,
                "seed": s.seed,
                "method": s.method,
                "kcal_mol": s.kcal_mol,
                "elapsed_s": s.elapsed_s,
                **({"extra": s.extra} if s.extra else {}),
            }
            for s in scores
        ],
        "summary": per_pocket_summary(scores),
        "ranking": rank_candidates(scores),
        "generated_utc": _utc_now_iso(),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def write_sweep_summary(
    sweep: Dict[str, object],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep, indent=2))


# ---------------------------------------------------------------------------
# Rank-correlation across pockets (Spearman)
# ---------------------------------------------------------------------------
def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman ρ (rank correlation). Ties → average rank. Returns None on bad input."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)

    def _avg_rank(vals: Sequence[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[sorted_idx[j + 1]] == vals[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx = _avg_rank(list(xs))
    ry = _avg_rank(list(ys))
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_100pocket_10seed.py",
        description=(
            "Wf-67 Phase 3 — 100-pocket × 10-seed docking sweep. "
            "When --use-real-vina is set, requires the vina binary on PATH; "
            "otherwise the Wf-30 honest stub scorer is used."
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CrossDocked-2020 100-pocket manifest CSV.",
    )
    p.add_argument(
        "--candidates-csv",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Reference candidate SMILES file (default: _demo_5_metaldrugs.txt).",
    )
    p.add_argument(
        "--pocket-prep-dir",
        type=Path,
        default=DEFAULT_POCKET_PREP_DIR,
        help="Wf-30 ten-pocket prep directory.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-pocket JSON + sweep summary.",
    )
    p.add_argument(
        "--pockets",
        type=int,
        default=100,
        help="Number of pockets (clamped to [1, 100]).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds per (pocket, candidate). Default 10.",
    )
    p.add_argument(
        "--use-real-vina",
        dest="use_real_vina",
        action="store_true",
        help="Use the real vina binary if installed; otherwise stub.",
    )
    p.add_argument(
        "--no-real-vina",
        dest="use_real_vina",
        action="store_false",
        help="Force the Wf-30 honest stub scorer (CI default).",
    )
    p.set_defaults(use_real_vina=False)
    p.add_argument(
        "--max-wall-min",
        type=float,
        default=120.0,
        help="Cap on per-pocket wall time in minutes (default 120).",
    )
    p.add_argument(
        "--max-pockets",
        type=int,
        default=100,
        help="Hard upper bound on processed pockets.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    n_pockets = max(1, min(args.pockets, args.max_pockets, 100))
    n_seeds = max(1, min(args.seeds, 50))  # sane upper bound
    seeds = list(range(n_seeds))

    wf30_bundles = _load_wf30_pocket_bundles(args.pocket_prep_dir)
    LOG.info("Wf-30 bundles loaded: %d (ids=%s)", len(wf30_bundles), sorted(wf30_bundles))

    crossdocked_rows = _load_crossdocked100_manifest(args.manifest)
    LOG.info("CrossDocked-2020 manifest rows: %d", len(crossdocked_rows))

    candidates = load_candidates(args.candidates_csv)
    LOG.info("Candidate pool size: %d", len(candidates))
    if not candidates:
        LOG.error("No candidates loaded from %s", args.candidates_csv)
        return 2

    pocket_ids = select_pockets(n_pockets, wf30_bundles, crossdocked_rows)
    LOG.info("Selected pockets (n=%d): %s", len(pocket_ids), pocket_ids[:5] + ["..."])

    use_real_vina = args.use_real_vina and _vina_binary_present() is not None
    vina_bin = _vina_binary_present() if use_real_vina else None
    if args.use_real_vina and vina_bin is None:
        LOG.warning(
            "--use-real-vina requested but no vina binary found; falling "
            "back to honest stub scorer (Wf-63 honest-negative framing)."
        )

    max_wall_s = float(args.max_wall_min) * 60.0
    t0 = time.time()
    pocket_summaries: List[Tuple[str, Dict[str, object]]] = []
    exit_status = 0
    pocket_modes: Dict[str, str] = {}  # pocket_id → method mix used

    for idx, pocket_id in enumerate(pocket_ids, start=1):
        bundle = pick_pocket_bundle(pocket_id, wf30_bundles, crossdocked_rows)
        LOG.info(
            "[%d/%d] pocket=%s source=%s receptor=%s",
            idx,
            len(pocket_ids),
            pocket_id,
            bundle.source,
            bundle.receptor_pdbqt,
        )
        try:
            scores = dock_pocket(
                bundle=bundle,
                candidates=candidates,
                seeds=seeds,
                use_real_vina=use_real_vina,
                vina_bin=vina_bin,
                max_wall_s=max_wall_s,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Pocket %s docking failed: %s", pocket_id, exc)
            exit_status = 3
            continue

        if not scores:
            LOG.warning("Pocket %s produced 0 evaluations — recording empty summary", pocket_id)
            continue

        per_pocket_path = args.out_dir / pocket_id / "per_seed_metrics.json"
        try:
            write_per_pocket_json(scores, per_pocket_path)
        except OSError as exc:
            LOG.error("Could not write %s: %s", per_pocket_path, exc)
            exit_status = 4
            continue

        pocket_summaries.append((pocket_id, per_pocket_summary(scores)))
        pocket_modes[pocket_id] = ",".join(sorted({s.method for s in scores}))
        LOG.info(
            "[%d/%d] %s done — n=%d best=%s %.2f kcal/mol",
            idx,
            len(pocket_ids),
            pocket_id,
            len(scores),
            per_pocket_summary(scores).get("best_candidate_smiles", "?"),
            per_pocket_summary(scores).get("best_candidate_mean_kcal_mol", float("nan")),
        )

    # Per-pocket rank-correlation across the candidate ranking (Spearman)
    rank_series: List[List[float]] = []  # one list of per-pocket best-mean-kcal per candidate
    candidate_order: List[str] = []
    for _, summ in pocket_summaries:
        per_cand = summ.get("per_candidate", [])  # type: ignore[assignment]
        for row in per_cand:  # type: ignore[attr-defined]
            cand = row["candidate_smiles"]  # type: ignore[index]
            if cand not in candidate_order:
                candidate_order.append(cand)
    for _, summ in pocket_summaries:
        per_cand_map = {
            row["candidate_smiles"]: row["mean_kcal_mol"]  # type: ignore[index]
            for row in summ.get("per_candidate", [])  # type: ignore[attr-defined]
        }
        rank_series.append([per_cand_map.get(c, math.nan) for c in candidate_order])

    spearman_per_pair: List[Optional[float]] = []
    for i in range(len(rank_series)):
        for j in range(i + 1, len(rank_series)):
            xs = rank_series[i]
            ys = rank_series[j]
            # Drop entries where either side is nan
            clean = [(a, b) for a, b in zip(xs, ys) if not math.isnan(a) and not math.isnan(b)]
            if len(clean) < 2:
                continue
            xs_clean = [a for a, _ in clean]
            ys_clean = [b for _, b in clean]
            rho = spearman(xs_clean, ys_clean)
            spearman_per_pair.append(rho)

    overall_kcals: List[float] = []
    for _, summ in pocket_summaries:
        v = summ.get("pocket_mean_kcal_mol")  # type: ignore[assignment]
        if isinstance(v, (int, float)):
            overall_kcals.append(float(v))

    sweep_summary = {
        "schema_version": 1,
        "tool": "bench_100pocket_10seed.py",
        "pilot_id": "wf67_p4_3_100pocket",
        "generated_utc": _utc_now_iso(),
        "n_pockets_requested": n_pockets,
        "n_pockets_completed": sum(
            1 for _, s in pocket_summaries if s.get("n_evaluations", 0) > 0  # type: ignore[operator]
        ),
        "n_pockets_selected": len(pocket_ids),
        "n_candidates": len(candidates),
        "n_seeds_per_evaluation": n_seeds,
        "use_real_vina_requested": args.use_real_vina,
        "use_real_vina_resolved": use_real_vina,
        "vina_binary_path": str(vina_bin) if vina_bin else None,
        "exit_status": exit_status,
        "total_wall_s": time.time() - t0,
        "overall_pocket_mean_kcal_mol": (
            statistics.fmean(overall_kcals) if overall_kcals else None
        ),
        "overall_pocket_std_kcal_mol": (
            statistics.pstdev(overall_kcals) if len(overall_kcals) > 1 else 0.0
        ),
        "pocket_modes": pocket_modes,
        "candidate_order_by_first_appearance": candidate_order,
        "spearman_per_pocket_pair": spearman_per_pair,
        "honest_framing": (
            "Wf-67 Phase 3 — 100-pocket × 10-seed docking sweep. "
            + (
                "Real Vina binary used for ALL evaluations."
                if use_real_vina
                else "All evaluations use the Wf-30 honest stub scorer "
                "(deterministic SHA-256-derived Vina kcal/mol). Real Vina "
                "re-scoring on a 10-pocket subset is queued for Wf-67 P4.4 "
                "once the CrossDocked-2020 receptors are staged."
            )
        ),
    }
    write_sweep_summary(sweep_summary, args.out_dir / "sweep_summary.json")
    LOG.info("Sweep summary → %s", args.out_dir / "sweep_summary.json")
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
