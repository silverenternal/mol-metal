#!/usr/bin/env python3
"""bench_vina_per_pocket_closure.py — Wf-126 P3 / TODO-125 per-pocket Vina bench.

==============================================================================
Wf-126 Phase 3 (TODO-125 per-pocket Vina balance via Catherine closure-bias)
==============================================================================

This script measures the per-pocket ``vina_high_affinity`` rate on the
M2 pocket_closure_bias cohort (TODO-126 §3 M2) and compares it to the
historical per-pocket baseline:

* 1h36: 0.95 ✅
* 4aaw: 0.65
* 4rlu: 0.45
* 3ny8: 0.35
* 3eml: 0.05 ❌

(TODO-125 §0)

Per-pocket closure-bias generation (M2) + per-pocket Wasserstein
alignment (M1) + per-pocket Vina docking gives the Wf-126 P3 close-out
record.  Goal: **no pocket below 0.30** (TODO-125 §5 uniform floor).

Inputs
------
* ``molmetal/molmetal_lam/reward/pocket_closure_bias.py`` (M2, shipped
  Wf-123 P2): ``closure_bounded_smiles_generation(pocket_id, n=...)``.
* ``molmetal/molmetal_lam/reward/wasserstein_loss.py`` (M1, shipped
  Wf-123 P1): ``wasserstein_pocket_loss(candidate_dist, pocket_ligand_dist)``.
* ``molmetal/molmetal_lam/sbdd_env/vina_adapter.py``: ``VinaDockingAdapter``
  with real Vina binary + ``STUB_FALLBACK`` for environments without Vina.

Outputs
-------
* ``<out-dir>/<pocket>/vina_high_affinity.json`` — per-pocket docking
  record (per-seed Vina kcal/mol + per-cand high_affinity flag +
  per-pocket Wasserstein distance).
* ``<out-dir>/per_pocket_summary.json`` — aggregated per-pocket
  ``vina_high_affinity_rate`` + Wasserstein distance + rank correlation.
* ``<out-dir>/vina_run_summary.json`` — parameter echo + honest
  framing block + binary detection status.

Verify
------
* ``per_pocket_summary.json`` ``vina_high_affinity_rate >= 0.30`` for
  every pocket (the uniform floor — TODO-125 §5).
* ``per_pocket_summary.json`` ``3eml.vina_high_affinity_rate >= 0.30``
  (vs current 0.05).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 \\
    molmetal/.venv/bin/python \\
        molmetal/scripts/bench_vina_per_pocket_closure.py \\
        --pockets 1h36 3eml 3ny8 4aaw 4rlu \\
        --seeds 0 1 2 \\
        --out-dir molmetal/reports/wf126_p3_vina_per_pocket

Honest framing
--------------
The 0.95 vs 0.05 spread is a real distribution mismatch (TODO-125 §0).
This script does NOT make every pocket 0.95 — it biases generation
per pocket so that no pocket falls below the 0.30 uniform floor.
When the Vina binary is missing, the script records ``status="STUB_FALLBACK"``
and never fabricates a kcal/mol score.
"""

from __future__ import annotations

import argparse
import csv
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

LOG = logging.getLogger("wf126_p3_vina_per_pocket")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Vina high-affinity cutoff (kcal/mol).  Standard cutoff for
#: "high-affinity" binders (per pocket2mol_vs_lambda_1h36.py + TODO-125 §0).
HIGH_AFFINITY_KCAL_CUTOFF: float = -7.0

#: Uniform floor — every pocket should reach this rate (TODO-125 §5).
UNIFORM_FLOOR_HIGH_AFFINITY_RATE: float = 0.30

#: 3eml-specific stretch target (vs current 0.05).
TARGET_3EML_HIGH_AFFINITY_RATE: float = 0.30

#: Historical per-pocket baseline (TODO-125 §0).
HISTORICAL_BASELINE: Dict[str, float] = {
    "1h36": 0.95,
    "4aaw": 0.65,
    "4rlu": 0.45,
    "3ny8": 0.35,
    "3eml": 0.05,
}

#: Default pocket cohort (5/5 Wf-126 P1 staged).
DEFAULT_POCKETS: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")

#: Default seeds.
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Cohort size per pocket × seed.
DEFAULT_COHORT_SIZE: int = 50


#: Per-pocket curated reference ligand SMILES (literature-known
#: binders).  Used by :func:`_wasserstein_pocket_alignment` as the
#: reference distribution for the W₂ computation.  These are NOT the
#: raw outputs of any generation model — they are hand-curated
#: reference ligands that characterise the pocket chemistry.
_POCKET_REFERENCE_LIGANDS: Dict[str, Tuple[str, ...]] = {
    "1h36": (
        # MMP13 Zn-coordination inhibitors (sulfonamide warheads)
        "NS(=O)(=O)c1ccc(NC(=O)C)cc1",  # sulfanilamide derivative
        "O=C(NO)Cc1ccccc1",  # hydroxamic acid analog
        "OC(=O)Cc1ccc(N)cc1",  # aminophenyl-acetic acid
        "NS(=O)(=O)c1ccc(C(=O)NO)cc1",  # MMP13 inhibitor
    ),
    "3eml": (
        # DHFR folate-pocket binders (diaminopyrimidine + pteridine)
        "Nc1nc(N)c2nc(CNc3ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc3)cnc2n1",  # methotrexate
        "Nc1nc2[nH]cnc2c(=O)[nH]1",  # guanine (natural ligand)
        "Nc1cc(N)nc(N)n1",  # 2,4,6-triaminopyrimidine
        "O=C(O)c1cnc(N)nc1N",  # pteroic acid precursor
    ),
    "3ny8": (
        # β2 adrenergic receptor ligands (amine + aromatic)
        "CC(C)NCC(O)c1ccc(O)c(CO)c1",  # isoproterenol
        "OC(C(O)C1CCC(N)C(=O)O)CC(=O)O",  # carboxylic acid analog
        "c1ccc(CN)cc1",  # benzylamine
        "CC(C)c1ccc(C(O)CNC(C)C)cc1",  # albuterol-like
    ),
    "4aaw": (
        # Carbonic anhydrase II inhibitors (sulfonamide warheads)
        "NS(=O)(=O)c1ccccc1",  # benzenesulfonamide
        "NS(=O)(=O)c1ccc(N)cc1",  # sulfanilamide
        "NS(=O)(=O)c1ccc(Cl)cc1",  # chlorothiazide analog
        "NS(=O)(=O)c1ccc(C(=O)O)cc1",  # 4-carboxybenzenesulfonamide
    ),
    "4rlu": (
        # GST electrophilic pocket binders (thiol-reactive)
        "O=C1NC(=O)c2ccccc21",  # phthalimide
        "O=C1c2ccccc2C(=O)c2ccccc21",  # anthraquinone
        "Cc1cc(=O)[nH]c(=S)[nH]1",  # thio-pyrimidinone
        "O=C1c2ccccc2-c2ccccc21",  # fluorenone
    ),
}


# ---------------------------------------------------------------------------
# Per-cell record
# ---------------------------------------------------------------------------


@dataclass
class VinaCellRecord:
    """One docking cell record (pocket × cand × seed)."""

    pocket: str
    cand_smiles: str
    seed: int
    score_kcal_mol: Optional[float]
    vina_status: str  # "REAL_VINA" / "STUB_FALLBACK"
    high_affinity: Optional[bool]
    wall_time_s: float
    note: str = ""


# ---------------------------------------------------------------------------
# Vina binary detection
# ---------------------------------------------------------------------------


def _detect_vina_binary() -> Dict[str, object]:
    """Detect whether a Vina-compatible binary is available.

    Returns a status dict with ``status`` (``REAL_VINA_AVAILABLE`` /
    ``MISSING_DEP``), ``bin``, ``version``, ``reason``,
    ``checked_paths``.  Mirrors the Wf-60 detection logic so this
    script has consistent binary detection across the project.
    """
    project_root = Path(__file__).resolve().parents[2]
    project_wrapper = project_root / "molmetal" / "bin" / "vina"
    candidates = [
        os.environ.get("MOLMETAL_VINA_BIN"),
        shutil.which("vina"),
        shutil.which("qvina02"),
        shutil.which("smina"),
        str(project_wrapper),
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
            "No executable Vina/Q-Vina2/smina binary found on PATH or "
            "well-known prefixes. Set MOLMETAL_VINA_BIN to enable real "
            "docking. Without the binary, every (pocket, cand, seed) "
            "cell falls back to STUB_FALLBACK — see honest_framing."
        ),
        "checked_paths": seen,
    }


# ---------------------------------------------------------------------------
# Docking (real Vina when available, deterministic stub otherwise)
# ---------------------------------------------------------------------------


def _build_pocket(pocket_id: str):
    """Build a synthetic Pocket from the pocket-id.

    Used when Vina is not available — the stub fallback does not
    actually dock, but we still need a well-formed Pocket for the
    real-Vina code path when it becomes available.
    """
    try:
        from molmetal.domain import Pocket  # type: ignore
    except Exception:  # pragma: no cover
        return None
    n_atoms = 200
    coords = __import__("torch").zeros(n_atoms, 3, dtype=__import__("torch").float32)
    coords[:, 0] = __import__("torch").linspace(-15.0, 15.0, n_atoms)
    atom_types = __import__("torch").full((n_atoms,), 6, dtype=__import__("torch").long)
    return Pocket(
        pdb_id=pocket_id,
        coords=coords,
        atom_types=atom_types,
        residue_ids=__import__("torch").zeros(n_atoms, dtype=__import__("torch").long),
        chain_ids=__import__("torch").zeros(n_atoms, dtype=__import__("torch").long),
        mask=__import__("torch").ones(n_atoms, dtype=__import__("torch").bool),
        center=__import__("torch").tensor(
            [0.0, 0.0, 0.0], dtype=__import__("torch").float32
        ),
        radius=15.0,
    )


def _find_staged_receptor(pocket_id: str) -> Optional[Path]:
    """Locate the staged receptor PDBQT file for ``pocket_id``.

    Searches in priority order:
      1. ``molmetal/reports/wf126_p1_staging/<pocket>/receptor.pdbqt`` (Wf-126 P1)
      2. ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt`` (Wf-30)

    Returns ``None`` if no staged receptor is found.
    """
    project_root = Path(__file__).resolve().parents[2]
    for staging_dir in (
        "wf126_p1_staging",
        "wf30_pocket_prep",
    ):
        cand = (
            project_root
            / "molmetal"
            / "reports"
            / staging_dir
            / pocket_id
            / "receptor.pdbqt"
        )
        if cand.is_file():
            return cand
    return None


def _receptor_center(pdbqt_path: Path) -> Tuple[float, float, float]:
    """Compute the geometric centre of a PDBQT receptor's ATOM lines.

    Falls back to (0, 0, 0) on parse failure.
    """
    try:
        coords = []
        with pdbqt_path.open() as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                try:
                    coords.append(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ]
                    )
                except (ValueError, IndexError):
                    continue
        if not coords:
            return (0.0, 0.0, 0.0)
        m = __import__("numpy").mean(coords, axis=0)
        return (float(m[0]), float(m[1]), float(m[2]))
    except Exception:
        return (0.0, 0.0, 0.0)


def _dock_one(
    pocket_id: str,
    cand_smiles: str,
    seed: int,
    vina_status: Dict[str, object],
    exhaustiveness: int = 4,
) -> VinaCellRecord:
    """Dock one (pocket, cand, seed) cell.

    When ``vina_status["status"] == "REAL_VINA_AVAILABLE"``, this
    uses the real ``vina`` Python binding against the staged
    receptor PDBQT file (wf126_p1_staging or wf30_pocket_prep).
    When the binary is missing OR no staged receptor is found,
    the cell falls back to STUB_FALLBACK — ``score_kcal_mol=None``,
    ``status="STUB_FALLBACK"``.  The script never fabricates a
    kcal/mol score.
    """
    t0 = time.perf_counter()
    if vina_status["status"] != "REAL_VINA_AVAILABLE":
        return VinaCellRecord(
            pocket=pocket_id,
            cand_smiles=cand_smiles,
            seed=seed,
            score_kcal_mol=None,
            vina_status="STUB_FALLBACK",
            high_affinity=None,
            wall_time_s=time.perf_counter() - t0,
            note=(
                "vina binary missing — STUB_FALLBACK; "
                "honest framing: high_affinity=None"
            ),
        )

    # Try real Vina path with the staged receptor.
    receptor = _find_staged_receptor(pocket_id)
    if receptor is None:
        return VinaCellRecord(
            pocket=pocket_id,
            cand_smiles=cand_smiles,
            seed=seed,
            score_kcal_mol=None,
            vina_status="STUB_FALLBACK",
            high_affinity=None,
            wall_time_s=time.perf_counter() - t0,
            note=(
                f"no staged receptor.pdbqt for {pocket_id} — "
                "STUB_FALLBACK (run stage_5_pockets_full.py first)"
            ),
        )

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
        import vina  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return VinaCellRecord(
            pocket=pocket_id,
            cand_smiles=cand_smiles,
            seed=seed,
            score_kcal_mol=None,
            vina_status="STUB_FALLBACK",
            high_affinity=None,
            wall_time_s=time.perf_counter() - t0,
            note=f"import-failed: {exc!r}",
        )

    try:
        # 1. Build ligand PDBQT via RDKit + Meeko.
        mol = Chem.MolFromSmiles(cand_smiles)
        if mol is None:
            return VinaCellRecord(
                pocket=pocket_id,
                cand_smiles=cand_smiles,
                seed=seed,
                score_kcal_mol=None,
                vina_status="PARSE_FAIL",
                high_affinity=None,
                wall_time_s=time.perf_counter() - t0,
                note="RDKit parse failed",
            )
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=int(seed))
        AllChem.MMFFOptimizeMolecule(mol_h)
        preparator = MoleculePreparation()
        pdbqt_string: Optional[str] = None
        for setup in preparator.prepare(mol_h):
            s, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                pdbqt_string = s
                break
        if pdbqt_string is None:
            return VinaCellRecord(
                pocket=pocket_id,
                cand_smiles=cand_smiles,
                seed=seed,
                score_kcal_mol=None,
                vina_status="PDBQT_FAIL",
                high_affinity=None,
                wall_time_s=time.perf_counter() - t0,
                note="Meeko PDBQT generation failed",
            )

        # 2. Real Vina dock.
        center = _receptor_center(receptor)
        box_size = (20.0, 20.0, 20.0)
        v = vina.Vina(sf_name="vina", seed=int(seed), verbosity=0)
        v.set_receptor(str(receptor))
        v.set_ligand_from_string(pdbqt_string)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=1)
        energies = v.energies(n_poses=1)
        score = float(energies[0][0])
        return VinaCellRecord(
            pocket=pocket_id,
            cand_smiles=cand_smiles,
            seed=seed,
            score_kcal_mol=score,
            vina_status="REAL_VINA",
            high_affinity=(score <= HIGH_AFFINITY_KCAL_CUTOFF),
            wall_time_s=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.debug(
            "[vina_per_pocket] dock failed pocket=%s smi=%r: %s",
            pocket_id,
            cand_smiles[:24],
            exc,
        )
        return VinaCellRecord(
            pocket=pocket_id,
            cand_smiles=cand_smiles,
            seed=seed,
            score_kcal_mol=None,
            vina_status="DOCK_FAIL",
            high_affinity=None,
            wall_time_s=time.perf_counter() - t0,
            note=f"{type(exc).__name__}: {str(exc)[:120]}",
        )


# ---------------------------------------------------------------------------
# Wasserstein alignment
# ---------------------------------------------------------------------------


def _wasserstein_pocket_alignment(
    candidate_smiles: Sequence[str],
    pocket_id: str,
    n_features: int = 16,
) -> Dict[str, object]:
    """Compute M1 wasserstein_pocket_loss between candidate and pocket ligand.

    The candidate and pocket-ligand distributions are summarised by
    RDKit Morgan-fingerprint density vectors of length ``n_features``.
    This is a coarse alignment proxy (M1 wasserstein_pocket_loss
    expects a ``(N, d)`` tensor; we hash-count Morgan bits into the
    first ``n_features`` bins).  The exact loss value is not the
    point — the *trend* (decreasing over cohort refinements) is.

    Returns
    -------
    dict
        ``{"wasserstein_loss": float, "n_candidates": int,
        "n_pocket_ligands": int, "engine": "wasserstein_reg"}``.
    """
    try:
        import torch  # type: ignore
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_pocket_loss,
        )
    except Exception as exc:  # pragma: no cover
        return {
            "wasserstein_loss": None,
            "n_candidates": len(candidate_smiles),
            "n_pocket_ligands": 0,
            "engine": "wasserstein_reg",
            "note": f"import-failed: {exc!r}",
        }

    def _fp_dense(smi: str) -> Optional["torch.Tensor"]:
        if not smi:
            return None
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_features)
            return torch.tensor(
                [int(fp.GetBit(i)) for i in range(n_features)],
                dtype=torch.float32,
            )
        except Exception:
            return None

    cand_vecs = [v for v in (_fp_dense(s) for s in candidate_smiles) if v is not None]
    # Reference distribution: per-pocket curated reference ligands.
    # The :func:`per_pocket_pharmacophore_lookup` returns class names
    # (e.g. "pteridine") for `preferred_scaffolds` — those aren't
    # SMILES, so we maintain a parallel hardcoded table of real
    # reference ligands (literature-known binders) below.
    ref_ligs = _POCKET_REFERENCE_LIGANDS.get(pocket_id, ())
    if not ref_ligs:
        # Generic fallback (aspirin / adenine / indole).
        ref_ligs = (
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "Nc1ncnc2[nH]cnc12",  # adenine
            "c1ccc2[nH]ccc2c1",  # indole
        )
    ref_vecs = [v for v in (_fp_dense(s) for s in ref_ligs) if v is not None]

    if not cand_vecs or not ref_vecs:
        return {
            "wasserstein_loss": None,
            "n_candidates": int(len(cand_vecs)),
            "n_pocket_ligands": int(len(ref_vecs)),
            "engine": "wasserstein_reg",
            "note": "empty candidate or pocket-ligand distribution",
        }

    cand = torch.stack(cand_vecs)
    ref = torch.stack(ref_vecs)
    try:
        w2 = wasserstein_pocket_loss(
            cand,
            ref,
            pocket_id=pocket_id,
            n_mc=128,
            seed=0,
        )
        return {
            "wasserstein_loss": float(w2.detach().cpu().item()),
            "n_candidates": int(cand.shape[0]),
            "n_pocket_ligands": int(ref.shape[0]),
            "engine": "wasserstein_reg",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "wasserstein_loss": None,
            "n_candidates": int(cand.shape[0]),
            "n_pocket_ligands": int(ref.shape[0]),
            "engine": "wasserstein_reg",
            "note": f"compute-failed: {exc!r}",
        }


# ---------------------------------------------------------------------------
# Per-pocket runner
# ---------------------------------------------------------------------------


def _smiles_from_bounded(result) -> List[str]:
    """Extract SMILES list from a BoundedGenerationResult."""
    return [str(c["smiles"]) for c in result.candidates]


def _run_pocket(
    pocket_id: str,
    seeds: Sequence[int],
    cohort_size: int,
    vina_status: Dict[str, object],
    per_scaffold_cap: int = 10,
    max_depth: int = 2,
) -> Dict[str, object]:
    """Run one pocket through all seeds.

    Generates a closure-bounded cohort per seed, docks each SMILES
    into the pocket, and computes per-seed high_affinity rate +
    Wasserstein alignment.

    Returns
    -------
    dict
        Per-pocket record + per-cell records + aggregates.
    """
    from molmetal.molmetal_lam.reward.pocket_closure_bias import (
        closure_bounded_smiles_generation,
    )

    all_cells: List[VinaCellRecord] = []
    per_seed_summary: List[Dict[str, object]] = []
    cohort_summary: List[Dict[str, object]] = []
    wassert_summary: List[Dict[str, object]] = []

    for seed in seeds:
        LOG.info(
            "[vina_per_pocket] pocket=%s seed=%d — generating cohort",
            pocket_id,
            seed,
        )
        cohort = closure_bounded_smiles_generation(
            pocket_id=pocket_id,
            n=cohort_size,
            max_depth=max_depth,
            max_products=cohort_size * 4,
            include_metal_coord_bias=True,
        )
        cohort_summary.append(
            {
                "seed": int(seed),
                "n_requested": int(cohort_size),
                "n_emitted_raw": int(cohort.n_candidates_raw),
                "n_kept": int(cohort.n_candidates_kept),
                "catherine_pass_rate": float(cohort.catherine_pass_rate),
                "wall_time_s": float(cohort.wall_time_s),
                "engine": str(cohort.engine),
                "n_seeds": int(cohort.n_seeds),
            }
        )

        smiles = _smiles_from_bounded(cohort)
        LOG.info(
            "[vina_per_pocket] pocket=%s seed=%d — docking %d cands",
            pocket_id,
            seed,
            len(smiles),
        )
        seed_cells: List[VinaCellRecord] = []
        for smi in smiles:
            cell = _dock_one(
                pocket_id,
                smi,
                seed,
                vina_status,
                exhaustiveness=4,
            )
            seed_cells.append(cell)
            all_cells.append(cell)

        # Per-seed high_affinity rate.
        real_cells = [c for c in seed_cells if c.high_affinity is not None]
        if real_cells:
            n_high = sum(1 for c in real_cells if c.high_affinity)
            rate = n_high / len(real_cells)
        else:
            rate = None  # all stub — honest framing
        per_seed_summary.append(
            {
                "seed": int(seed),
                "n_docked": len(real_cells),
                "n_high_affinity": (
                    sum(1 for c in real_cells if c.high_affinity) if real_cells else 0
                ),
                "high_affinity_rate": (None if rate is None else float(round(rate, 6))),
                "score_min": (
                    None
                    if not real_cells
                    else float(min(c.score_kcal_mol for c in real_cells))
                ),
                "score_mean": (
                    None
                    if not real_cells
                    else float(statistics.fmean(c.score_kcal_mol for c in real_cells))
                ),
                "n_stub_fallback": sum(
                    1 for c in seed_cells if c.vina_status == "STUB_FALLBACK"
                ),
            }
        )

        # Wasserstein alignment per seed.
        w = _wasserstein_pocket_alignment(smiles, pocket_id)
        w["seed"] = int(seed)
        wassert_summary.append(w)

    # Per-pocket aggregate across seeds.
    real_rates = [
        s["high_affinity_rate"]
        for s in per_seed_summary
        if s["high_affinity_rate"] is not None
    ]
    agg_rate = float(statistics.fmean(real_rates)) if real_rates else None
    agg_real_cells = [c for c in all_cells if c.high_affinity is not None]

    return {
        "pocket_id": pocket_id,
        "n_seeds": len(seeds),
        "per_seed": per_seed_summary,
        "cohort_summary": cohort_summary,
        "wasserstein_summary": wassert_summary,
        "aggregate": {
            "high_affinity_rate_aggregated": (
                None if agg_rate is None else float(round(agg_rate, 6))
            ),
            "high_affinity_rate_aggregated_target": (UNIFORM_FLOOR_HIGH_AFFINITY_RATE),
            "pass_uniform_floor": (
                agg_rate is not None and agg_rate >= UNIFORM_FLOOR_HIGH_AFFINITY_RATE
            ),
            "historical_baseline": HISTORICAL_BASELINE.get(pocket_id),
            "delta_vs_historical": (
                None
                if agg_rate is None or HISTORICAL_BASELINE.get(pocket_id) is None
                else float(round(agg_rate - HISTORICAL_BASELINE[pocket_id], 6))
            ),
            "n_docked_total": len(agg_real_cells),
            "n_high_affinity_total": (
                sum(1 for c in agg_real_cells if c.high_affinity)
            ),
            "score_min_overall": (
                None
                if not agg_real_cells
                else float(min(c.score_kcal_mol for c in agg_real_cells))
            ),
            "score_mean_overall": (
                None
                if not agg_real_cells
                else float(statistics.fmean(c.score_kcal_mol for c in agg_real_cells))
            ),
            "n_stub_fallback_total": sum(
                1 for c in all_cells if c.vina_status == "STUB_FALLBACK"
            ),
        },
        "per_cell": [
            {
                "pocket": c.pocket,
                "cand_smiles": c.cand_smiles,
                "seed": int(c.seed),
                "score_kcal_mol": (
                    None
                    if c.score_kcal_mol is None
                    else float(round(c.score_kcal_mol, 4))
                ),
                "vina_status": c.vina_status,
                "high_affinity": c.high_affinity,
                "wall_time_s": float(round(c.wall_time_s, 4)),
                "note": c.note,
            }
            for c in all_cells
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-126 P3 / TODO-125 per-pocket Vina balance bench.  "
            "Generates per-pocket closure-bias cohorts, docks each "
            "candidate, and reports per-pocket high_affinity_rate."
        ),
    )
    p.add_argument(
        "--pockets",
        type=str,
        nargs="+",
        default=list(DEFAULT_POCKETS),
        help=(
            "Pocket identifiers.  Default: 1h36 3eml 3ny8 4aaw 4rlu "
            "(5/5 Wf-126 P1 staged)."
        ),
    )
    p.add_argument(
        "--cohort-size",
        type=int,
        default=DEFAULT_COHORT_SIZE,
        help=(
            "Per-pocket cohort size (per seed).  Default 50 (CPU "
            "tractable — full GPU run with n=500 is a follow-up)."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default 0 1 2).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="BFS depth budget (default 2).",
    )
    p.add_argument(
        "--per-scaffold-cap",
        type=int,
        default=10,
        help="Per-seed yield cap (default 10).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf126_p3_vina_per_pocket",
        help="Output directory.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    vina_status = _detect_vina_binary()
    LOG.info(
        "[vina_per_pocket] vina_binary_status=%s bin=%s",
        vina_status["status"],
        vina_status.get("bin"),
    )

    per_pocket_records: Dict[str, Dict[str, object]] = {}

    for pocket_id in args.pockets:
        LOG.info(
            "[vina_per_pocket] ===== pocket=%s =====",
            pocket_id,
        )
        rec = _run_pocket(
            pocket_id=pocket_id,
            seeds=args.seeds,
            cohort_size=args.cohort_size,
            vina_status=vina_status,
            per_scaffold_cap=args.per_scaffold_cap,
            max_depth=args.max_depth,
        )
        per_pocket_records[pocket_id] = rec

        # Per-pocket output JSON.
        pocket_dir = args.out_dir / pocket_id
        pocket_dir.mkdir(parents=True, exist_ok=True)
        pocket_json = pocket_dir / "vina_high_affinity.json"
        pocket_json.write_text(json.dumps(rec, indent=2, default=str))
        LOG.info(
            "[vina_per_pocket] pocket=%s wrote %s",
            pocket_id,
            pocket_json,
        )

        agg = rec["aggregate"]
        rate = agg["high_affinity_rate_aggregated"]
        if rate is not None:
            print(
                f"[vina_per_pocket] pocket={pocket_id} "
                f"high_affinity_rate={rate:.3f} "
                f"target>={UNIFORM_FLOOR_HIGH_AFFINITY_RATE:.2f} "
                f"pass_uniform_floor={agg['pass_uniform_floor']} "
                f"historical_baseline={agg['historical_baseline']} "
                f"delta_vs_historical={agg['delta_vs_historical']}",
                flush=True,
            )
        else:
            print(
                f"[vina_per_pocket] pocket={pocket_id} "
                f"high_affinity_rate=None (STUB_FALLBACK) "
                f"n_stub_fallback={agg['n_stub_fallback_total']}",
                flush=True,
            )

    # Aggregate summary across pockets.
    rates_by_pocket: Dict[str, Optional[float]] = {
        p: r["aggregate"]["high_affinity_rate_aggregated"]
        for p, r in per_pocket_records.items()
    }
    pass_all_floor = all(
        (r is not None and r >= UNIFORM_FLOOR_HIGH_AFFINITY_RATE)
        for r in rates_by_pocket.values()
    )
    pass_3eml = (
        rates_by_pocket.get("3eml") is not None
        and rates_by_pocket["3eml"] >= TARGET_3EML_HIGH_AFFINITY_RATE
    )

    summary = {
        "metadata": {
            "pockets": list(args.pockets),
            "seeds": list(args.seeds),
            "cohort_size": int(args.cohort_size),
            "max_depth": int(args.max_depth),
            "per_scaffold_cap": int(args.per_scaffold_cap),
            "engine_pocket_closure_bias": "pocket_closure_bias",
            "engine_wasserstein": "wasserstein_reg",
            "high_affinity_kcal_cutoff": HIGH_AFFINITY_KCAL_CUTOFF,
        },
        "vina_binary_status": vina_status,
        "per_pocket": {p: r["aggregate"] for p, r in per_pocket_records.items()},
        "per_pocket_wasserstein": {
            p: [
                {
                    "seed": int(w["seed"]),
                    "wasserstein_loss": w.get("wasserstein_loss"),
                    "n_candidates": int(w.get("n_candidates", 0)),
                    "n_pocket_ligands": int(w.get("n_pocket_ligands", 0)),
                }
                for w in r["wasserstein_summary"]
            ]
            for p, r in per_pocket_records.items()
        },
        "targets": {
            "uniform_floor_high_affinity_rate": UNIFORM_FLOOR_HIGH_AFFINITY_RATE,
            "3eml_stretch_target": TARGET_3EML_HIGH_AFFINITY_RATE,
            "historical_baseline": HISTORICAL_BASELINE,
        },
        "primary": {
            "pass_uniform_floor_all_pockets": bool(pass_all_floor),
            "pass_3eml_stretch": bool(pass_3eml),
            "n_pockets_meeting_floor": sum(
                1
                for r in rates_by_pocket.values()
                if r is not None and r >= UNIFORM_FLOOR_HIGH_AFFINITY_RATE
            ),
            "n_pockets_total": len(rates_by_pocket),
            "min_high_affinity_rate": (
                None
                if any(r is None for r in rates_by_pocket.values())
                else float(round(min(rates_by_pocket.values()), 6))
            ),
            "max_high_affinity_rate": (
                None
                if any(r is None for r in rates_by_pocket.values())
                else float(round(max(rates_by_pocket.values()), 6))
            ),
        },
        "honest_framing": {
            "vina_binary_status": vina_status["status"],
            "stub_fallback_explanation": (
                "When the Vina binary is missing, every (pocket, cand, "
                "seed) cell falls back to STUB_FALLBACK and "
                "high_affinity_rate=None.  No kcal/mol score is "
                "fabricated.  The uniform-floor verification then "
                "reports pass_uniform_floor_all_pockets=False and the "
                "per-pocket records preserve the stub rationale."
            ),
            "per_pocket_variance_acknowledgement": (
                "Per-pocket results legitimately differ.  The 0.30 "
                "uniform floor is the verification gate, not a target "
                "to equalise every pocket to 0.95 (TODO-125 §5)."
            ),
            "cohort_size_warning": (
                "Per-pocket cohort n=50 (per seed) is below the n>=500 "
                "target in TODO-125 §2.2; this is a CPU-only ship "
                "target.  Larger cohorts (n=500) are deferred to a "
                "follow-up GPU run."
            ),
        },
    }

    summary_path = args.out_dir / "per_pocket_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("[vina_per_pocket] wrote %s", summary_path)

    # Run summary (parameters + honest framing).
    run_summary = {
        "args": vars(args),
        "primary": summary["primary"],
        "vina_binary_status": vina_status["status"],
        "wall_time_s_total": float(
            sum(
                sum(c["wall_time_s"] for c in r["per_cell"])
                for r in per_pocket_records.values()
            )
        ),
    }
    run_summary_path = args.out_dir / "vina_run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2, default=str))
    LOG.info("[vina_per_pocket] wrote %s", run_summary_path)

    # Per-pocket CSV (smiles, pocket, seed, score, high_affinity).
    csv_path = args.out_dir / "per_pocket_vina.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "pocket",
                "seed",
                "smiles",
                "score_kcal_mol",
                "high_affinity",
                "vina_status",
                "wall_time_s",
            ]
        )
        for pocket_id, r in per_pocket_records.items():
            for c in r["per_cell"]:
                writer.writerow(
                    [
                        pocket_id,
                        c["seed"],
                        c["cand_smiles"],
                        c["score_kcal_mol"],
                        c["high_affinity"],
                        c["vina_status"],
                        c["wall_time_s"],
                    ]
                )
    LOG.info("[vina_per_pocket] wrote %s", csv_path)

    print(
        f"[vina_per_pocket] DONE  pockets={len(args.pockets)} "
        f"seeds={len(args.seeds)} pass_uniform_floor_all_pockets="
        f"{pass_all_floor} pass_3eml_stretch={pass_3eml} "
        f"vina_status={vina_status['status']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
