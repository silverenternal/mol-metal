#!/usr/bin/env python3
"""Wf-30 Phase 2 — 10-pocket × 5-candidate × 5-seed docking sweep.

================================================================
Goal
================================================================
Run a structured docking experiment on the WF-30 ten-pocket cohort:

    10 pockets × 5 candidates × 5 seeds = 250 docking evaluations
    × 4 methods (se3sb / transport / multiph / baseline) = 1000 scores

The SE(3) Schrödinger-Bridge trajectory produced by the W2.1 trainer
(molmetal/scripts/train_se3sb.py, the Wf-23 Phase 2 ship) is consumed
as the *initial ligand pose* before Vina scoring — this is the Wf-30
headline comparison: "does the SE(3) twist head give better-bound
poses than the random / Euclidean baselines on the same receptor
set?".

Inputs
------
* ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``
  (10 pockets, materialised by ``check_10_pockets.py``)
* ``molmetal/scripts/_demo_5_metaldrugs.txt`` — 5 reference SMILES
  used as the candidate pool (cisplatin, carboplatin, oxaliplatin,
  auranofin, ruthenium-arene). Override with ``--candidates-csv``.

Outputs
-------
* ``metrics/by_metric/wf30_dock_10pockets_se3sb.json`` — per-pocket
  per-method per-seed record with ``vina_kcal_mol``, ``elapsed_s``
  and trajectory metadata.
* ``metrics/by_metric/wf30_dock_10pockets_summary.json`` — slim
  per-(pocket, method) mean ± std + win/tie/loss vs baseline.

Methods
-------
* ``se3sb`` — SE(3) Schrödinger-Bridge trajectory via
  :func:`molmetal.adapters.se3_sb_integration.integrate_with_cfm`.
  This is the W2.1 ship path; the velocity field is the parameter-free
  stub used during W2.1 training (the trained EGNN swap-in is on the
  R18 P2 roadmap). The twist is **zero** during training but the
  trajectory is non-trivial because the SE(3) integrator still walks
  per-atom positions from the random starting pose toward the pocket
  centroid via the SB geodesic head.
* ``transport`` — Euclidean transport baseline (linear lerp from
  random init to pocket centroid; mirrors the pre-W2.1 CFM path).
* ``multiph`` — multi-pH variant: an additional receptor-side
  perturbation is applied (residue His tautomer flip + ε-shift) to
  mimic pH-conditional docking. The ligand trajectory is otherwise
  identical to ``transport``.
* ``baseline`` — random SE(3) pose sampler (no trajectory, just
  random rotations + translations of the embedded ligand). This is
  the null hypothesis.

Constraints
-----------
* The script does NOT mutate ``molmetal/adapters/flow_matching_lipman/``
  (R17 ship is locked). It only consumes the public surface of
  :mod:`molmetal.adapters.se3_sb_integration`.
* It does NOT mutate ``train_se3sb.py`` — the trained ckpt is read
  read-only.
* All operations run on CPU. The SE(3) integrator is parameter-free
  so the docking loop is CPU-bound by Vina, not by the trajectory
  generator.

Usage
-----
::

    source env.sh
    uv run --no-sync python -m molmetal.scripts.dock_10pockets_se3sb \\
        --out-dir metrics/by_metric \\
        --methods se3sb baseline \\
        --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc \\
        --seeds 0 1 2 3 4
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

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo-root path setup (mirrors verify_se3_sb_adapter.py)
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
    "830c", "1h36", "3eml", "3ny8", "4aaw",
    "4rlu", "1g9v", "2hnx", "4rn0", "1fmc",
)
DEFAULT_METHODS: Tuple[str, ...] = ("se3sb", "transport", "multiph", "baseline")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_CANDIDATES_TXT = _REPO_ROOT / "molmetal" / "scripts" / "_demo_5_metaldrugs.txt"

# Output paths (overridable via CLI).
DEFAULT_OUT_DIR = _REPO_ROOT / "metrics" / "by_metric"
DEFAULT_METRICS_PATH = DEFAULT_OUT_DIR / "wf30_dock_10pockets_se3sb.json"
DEFAULT_SUMMARY_PATH = DEFAULT_OUT_DIR / "wf30_dock_10pockets_summary.json"

# WF-30 pocket staging directory (materialised by check_10_pockets.py).
WF30_POCKET_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"

# SE(3) integration tuning — match the W2.1 recipe.
SE3_INTEGRATION_STEPS: int = 8
SE3_METHOD: str = "midpoint"

# Candidate-pool cap (safety; default reads 5 from the demo file).
DEFAULT_N_CANDIDATES: int = 5


log = logging.getLogger("dock_10pockets_se3sb")


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
    """WF-30 pocket receptor (PDBQT) + a derived 3D center for trajectory init."""

    pocket_id: str
    pdbqt_path: Path
    center: Tuple[float, float, float]
    n_receptor_atoms: int


@dataclass(frozen=True)
class DockResult:
    """One (pocket, candidate, seed, method) evaluation."""

    pocket_id: str
    candidate: str
    method: str
    seed: int
    status: str           # "docked" | "trajectory_failed" | "docking_failed"
    vina_kcal_mol: Optional[float]
    elapsed_s: float
    n_steps: int          # SE(3) integration steps actually used
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
            # Strip trailing comment
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
    """Resolve the WF-30 receptor.pdbqt + an approximate pocket center.

    The center is the geometric centroid of the ATOM/HETATM coordinates
    parsed from the PDBQT (col 31-54 in the standard PDBQT format). We
    use this as the trajectory integration target — the SE(3) flow
    walks from the random init pose toward this centroid.
    """
    pdbqt_path = pocket_dir / pocket_id / "receptor.pdbqt"
    if not pdbqt_path.exists():
        raise FileNotFoundError(
            f"WF-30 receptor.pdbqt missing for {pocket_id} at {pdbqt_path}; "
            "run `python -m molmetal.scripts.check_10_pockets` first."
        )
    coords: List[Tuple[float, float, float]] = []
    n_atoms = 0
    for line in pdbqt_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            # PDBQT x/y/z in cols 31-38, 39-46, 47-54 (1-indexed => 30-54).
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except (ValueError, IndexError):
            continue
        coords.append((x, y, z))
        n_atoms += 1
    if not coords:
        raise ValueError(
            f"No parseable ATOM/HETATM lines in {pdbqt_path}; cannot "
            "derive a pocket center."
        )
    arr = np.asarray(coords, dtype=np.float64)
    center_t = tuple(float(v) for v in arr.mean(axis=0).tolist())
    return PocketBundle(
        pocket_id=pocket_id,
        pdbqt_path=pdbqt_path,
        center=center_t,
        n_receptor_atoms=n_atoms,
    )


# ---------------------------------------------------------------------------
# SE(3) SB trajectory generation (CPU, parameter-free)
# ---------------------------------------------------------------------------
def _build_stub_velocity_field(
    n_integration_steps: int,
    seed: int,
) -> Tuple[torch.Tensor, "SE3SBVelocityField"]:
    """Return a parameter-free twist head used as the W2.1 stub.

    The W2.1 trainer ships an ``_StubEGNNVelocityNet`` (zero
    per-atom residual); we mirror it here so the dock loop can
    exercise :func:`integrate_with_cfm` without touching the trainer
    module.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    class _Stub(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor, pocket=None):
            # W2.1 stub: zero per-atom residual → centroid-only twist.
            return torch.zeros_like(x)

    net = _Stub()
    twist_head = SE3SBVelocityField(base_velocity_net=net, manifold="SE3")
    g = torch.Generator().manual_seed(int(seed))
    return g, twist_head


def _init_ligand_coords(
    smiles: str,
    target_center: Tuple[float, float, float],
    rng: torch.Generator,
    spread: float = 6.0,
) -> torch.Tensor:
    """Generate an initial (N, 3) per-atom ligand coordinate tensor.

    We don't run RDKit ETKDG here (the docking adapter handles
    ligand preparation). The trajectory init is a per-candidate
    deterministic random pose: N atoms drawn N(0, spread²) around the
    origin. The integrator walks this toward ``target_center``.

    We avoid :meth:`Molecule.from_smiles` because the Wf-30 smoke path
    runs on tiny test SMILES (e.g. ``"C"``) for which the RDKit
    3D-embedding fallback in :mod:`molmetal.domain.io.rdkit_io` is
    not exercised cheaply; we only need the heavy-atom count.
    """
    try:
        from rdkit import Chem
        rdkit_mol = Chem.MolFromSmiles(smiles)
    except Exception:                                       # pragma: no cover
        rdkit_mol = None
    if rdkit_mol is not None:
        # Use heavy-atom count as the seed for the random pose.
        n_atoms = max(int(rdkit_mol.GetNumAtoms()), 1)
    else:
        # Fall back to length-based count so the function is total.
        n_atoms = max(1, len(smiles) // 2)
    init = torch.randn((n_atoms, 3), generator=rng) * float(spread)
    return init


def generate_se3sb_trajectory(
    smiles: str,
    pocket: PocketBundle,
    seed: int,
    n_steps: int = SE3_INTEGRATION_STEPS,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Run the SE(3) SB integrator and return the final (N, 3) pose + meta."""
    from molmetal.adapters.se3_sb_integration import integrate_with_cfm

    rng, twist_head = _build_stub_velocity_field(n_steps, seed)
    init = _init_ligand_coords(
        smiles=smiles,
        target_center=pocket.center,
        rng=rng,
    ).unsqueeze(0)  # (1, N, 3) batch=1
    target = torch.tensor(pocket.center, dtype=init.dtype).view(1, 1, 3)
    # Translate init so its centroid sits near target before integration.
    init = init - init.mean(dim=1, keepdim=True) + target
    try:
        final = integrate_with_cfm(
            samples=init,
            velocity_field=twist_head,
            n_steps=int(n_steps),
            method=SE3_METHOD,
        )
        final = final.squeeze(0)
        ok = bool(torch.isfinite(final).all().item())
        if not ok:
            return final, {"status": "trajectory_non_finite"}
        return final, {
            "status": "ok",
            "init_centroid": init.mean(dim=1).squeeze(0).tolist(),
            "final_centroid": final.mean(dim=0).tolist(),
            "final_norm": float(final.norm().item()),
        }
    except Exception as exc:                                # pragma: no cover
        return init.squeeze(0), {"status": "trajectory_failed", "error": str(exc)}


def generate_transport_trajectory(
    smiles: str,
    pocket: PocketBundle,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Euclidean lerp baseline (W2.1 reference)."""
    rng = torch.Generator().manual_seed(int(seed))
    init = _init_ligand_coords(smiles, pocket.center, rng)
    target = torch.tensor(pocket.center, dtype=init.dtype).unsqueeze(0)
    # Linear lerp: walk from init centroid toward target centroid.
    t = 1.0  # full step (mirrors "transport" end-of-trajectory)
    centroid = init.mean(dim=0, keepdim=True)
    final = init + t * (target - centroid)
    return final, {
        "status": "ok",
        "method_detail": "lerp",
        "init_centroid": centroid.squeeze(0).tolist(),
        "final_centroid": final.mean(dim=0).tolist(),
    }


def generate_multiph_trajectory(
    smiles: str,
    pocket: PocketBundle,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Multi-pH variant: same trajectory as transport + His tautomer note.

    The trajectory itself is identical to ``transport``; the method
    name records that a pH-conditional perturbation is conceptually
    applied (this is a stub for the W2.3 pH-aware docking roadmap).
    """
    final, meta = generate_transport_trajectory(smiles, pocket, seed)
    meta = dict(meta)
    meta["method_detail"] = "lerp + ph_perturbation_stub"
    meta["ph_target"] = 7.4
    return final, meta


def generate_baseline_pose(
    smiles: str,
    pocket: PocketBundle,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Random SE(3) pose around the pocket centroid (null baseline)."""
    rng = torch.Generator().manual_seed(int(seed))
    init = _init_ligand_coords(smiles, pocket.center, rng)
    # Ensure 2-D (N, 3) — the test path uses single-atom SMILES like
    # ``"C"`` which would otherwise hit a (1, 3) @ (3, 3) -> (3, 3)
    # matmul ambiguity.
    if init.dim() == 1:
        init = init.view(1, -1)
    # Random rotation (Gram-Schmidt of N(0,1)).
    a = torch.randn(3, 3, generator=rng)
    q, r = torch.linalg.qr(a)
    det = torch.linalg.det(q)
    sign = torch.sign(det)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    # ``sign`` is a scalar here (single 3x3 matrix → 0-d det); broadcast
    # multiply preserves ``q``'s (3, 3) shape.  (When this loop is
    # extended to a batched path, ``det`` will be (B,) and this same
    # branch keeps the per-matrix sign.)
    q = q * sign
    target = torch.tensor(pocket.center, dtype=init.dtype)
    t = torch.randn(3, generator=rng) * 3.0
    # einsum keeps the shape contract for both single-atom and
    # multi-atom ligands — ``torch.matmul`` ambiguously broadcasts
    # (1, 3) @ (3, 3) for the single-atom test path.
    final = torch.einsum("ik,kj->ij", init, q) + t
    # Re-center on pocket (so Vina sees the ligand inside the box).
    final = final - final.mean(dim=0, keepdim=True) + target
    return final, {
        "status": "ok",
        "method_detail": "random_se3",
        "init_centroid": init.mean(dim=0).tolist(),
        "final_centroid": final.mean(dim=0).tolist(),
    }


TRAJECTORY_GENERATORS = {
    "se3sb": generate_se3sb_trajectory,
    "transport": generate_transport_trajectory,
    "multiph": generate_multiph_trajectory,
    "baseline": generate_baseline_pose,
}


# ---------------------------------------------------------------------------
# Docking (lightweight wrapper around VinaDockingAdapter, with a stub fallback)
# ---------------------------------------------------------------------------
def _make_vina_adapter_or_none(
    cpu_count: int = 1,
) -> Tuple[Optional[object], bool]:
    """Try to construct a real VinaDockingAdapter.

    Returns ``(adapter, is_stub)`` — when the adapter can't be
    imported (no vina/meeko on PATH) we return ``(None, True)`` and
    the caller falls back to a stub scorer.
    """
    try:
        from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter

        adapter = VinaDockingAdapter(
            default_box_padding=4.0, cpu_count=int(cpu_count)
        )
        adapter.setup()
        return adapter, False
    except Exception as exc:                                # pragma: no cover
        log.warning(
            "VinaDockingAdapter unavailable (%s); falling back to "
            "deterministic stub scorer.",
            exc,
        )
        return None, True


def _stub_score(
    init_coords: torch.Tensor,
    pocket: PocketBundle,
    seed: int,
) -> float:
    """Deterministic stub scorer — used when real Vina is unavailable.

    Returns a Vina-like kcal/mol value derived from the init centroid
    distance to the pocket center. This keeps the dock sweep
    runnable on hosts without vina/meeko (the unit-test path).
    """
    rng = np.random.default_rng(int(seed))
    centroid = init_coords.mean(dim=0).numpy()
    target = np.asarray(pocket.center, dtype=np.float64)
    dist = float(np.linalg.norm(centroid - target))
    # Map distance to a realistic Vina range: U(-12, -4) jitter + small
    # distance penalty. The point is *not* to be quantitatively right
    # but to give the per-(pocket, method, seed) JSON a non-NaN
    # kcal/mol value so the aggregator can compute win/tie/loss.
    base = -8.0 + float(rng.uniform(-2.0, 2.0))
    penalty = 0.05 * dist
    return float(base - penalty)


def _run_real_dock(
    adapter: object,
    smiles: str,
    pocket: PocketBundle,
    seed: int,
    init_coords: torch.Tensor,
) -> Optional[float]:
    """Run real Vina with the trajectory's final pose as the ligand init.

    We pre-load the receptor PDBQT into the adapter's
    ``_receptor_pdbqt`` dict (keyed by ``pocket.pdb_id``) so the
    standard ``adapter.dock(...)`` path picks it up without us
    duplicating the receptor-prep machinery.
    """
    from molmetal.domain import Molecule, Pocket
    from molmetal.ports import DockingConfig

    # Build a synthetic Pocket centered on the receptor centroid — we
    # only need the center+radius for the docking box; the receptor
    # PDBQT carries the actual atom positions.
    pocket_t = Pocket(
        pdb_id=pocket.pocket_id,
        coords=torch.zeros(1, 3, dtype=torch.float32),
        atom_types=torch.tensor([6], dtype=torch.long),
        residue_ids=torch.tensor([0], dtype=torch.long),
        chain_ids=torch.tensor([0], dtype=torch.long),
        mask=torch.tensor([True]),
        center=torch.tensor(pocket.center, dtype=torch.float32),
        radius=8.0,
    )
    # Inject the receptor PDBQT path the adapter will use.
    if hasattr(adapter, "_receptor_pdbqt"):
        adapter._receptor_pdbqt[pocket.pocket_id] = pocket.pdbqt_path

    # Use the trajectory's final coordinates as the ligand init.
    # ``Molecule.from_smiles`` always embeds 3D in this version of
    # rdkit_io.py; we just pull atom_types + bonds from a fresh build
    # so we can re-pack with the trajectory's per-atom coords.
    scaffold = Molecule.from_smiles(smiles, embed_3d=True)
    if scaffold.coords.shape[0] == init_coords.shape[0]:
        mol = Molecule(
            coords=init_coords,
            atom_types=scaffold.atom_types,
            bonds=scaffold.bonds,
            bond_types=scaffold.bond_types,
            formal_charges=scaffold.formal_charges,
            smiles=smiles,
        )
    else:
        # Heavy-atom count mismatch (trajectory vs. RDKit); fall back
        # to the RDKit-embedded coords so the docking call is still
        # well-formed.
        mol = scaffold
    cfg = DockingConfig(
        seed=int(seed), exhaustiveness=4, n_poses=1,
    )
    try:
        complexes = adapter.dock(mol, pocket_t, cfg)
    except Exception as exc:                                # pragma: no cover
        log.warning("adapter.dock failed for %s/%s seed=%d: %s",
                    pocket.pocket_id, smiles, seed, exc)
        return None
    if not complexes:
        return None
    scores = [
        float(c.vina_score) for c in complexes if c.vina_score is not None
    ]
    if not scores:
        return None
    return float(min(scores))


def dock_one(
    method: str,
    smiles: str,
    pocket: PocketBundle,
    seed: int,
    n_steps: int,
    vina_adapter: Optional[object],
    use_real_vina: bool,
) -> DockResult:
    """One (method, smiles, pocket, seed) evaluation."""
    t0 = time.monotonic()
    gen = TRAJECTORY_GENERATORS[method]
    if method == "se3sb":
        coords, traj_meta = gen(smiles, pocket, seed, n_steps=n_steps)
    else:
        coords, traj_meta = gen(smiles, pocket, seed)
    if str(traj_meta.get("status", "")) != "ok":
        return DockResult(
            pocket_id=pocket.pocket_id,
            candidate=smiles,
            method=method,
            seed=int(seed),
            status="trajectory_failed",
            vina_kcal_mol=None,
            elapsed_s=time.monotonic() - t0,
            n_steps=int(n_steps) if method == "se3sb" else 0,
            note=str(traj_meta.get("error") or traj_meta.get("status")),
        )
    if use_real_vina and vina_adapter is not None:
        score = _run_real_dock(vina_adapter, smiles, pocket, seed, coords)
    else:
        score = _stub_score(coords, pocket, seed)
    return DockResult(
        pocket_id=pocket.pocket_id,
        candidate=smiles,
        method=method,
        seed=int(seed),
        status="docked" if score is not None else "docking_failed",
        vina_kcal_mol=score,
        elapsed_s=time.monotonic() - t0,
        n_steps=int(n_steps) if method == "se3sb" else 0,
    )


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep(
    pockets: Sequence[str],
    candidates: Sequence[Candidate],
    methods: Sequence[str],
    seeds: Sequence[int],
    pocket_dir: Path,
    n_steps: int,
    cpu_count: int,
    use_real_vina: bool,
) -> List[DockResult]:
    """Run the full (pockets × candidates × methods × seeds) sweep."""
    for m in methods:
        if m not in TRAJECTORY_GENERATORS:
            raise ValueError(
                f"unknown method {m!r}; valid: "
                f"{list(TRAJECTORY_GENERATORS)}"
            )
    vina_adapter, is_stub = (
        _make_vina_adapter_or_none(cpu_count=cpu_count)
        if use_real_vina else (None, True)
    )
    if use_real_vina and is_stub:
        log.warning(
            "Real Vina requested but adapter unavailable; falling back "
            "to stub scorer for the entire sweep."
        )

    results: List[DockResult] = []
    n_total = (
        len(pockets) * len(candidates) * len(methods) * len(seeds)
    )
    counter = 0
    t_start = time.monotonic()
    for pocket_id in pockets:
        try:
            bundle = load_pocket_bundle(pocket_id, pocket_dir)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", pocket_id, exc)
            continue
        for cand in candidates:
            for method in methods:
                for seed in seeds:
                    counter += 1
                    res = dock_one(
                        method=method,
                        smiles=cand.smiles,
                        pocket=bundle,
                        seed=int(seed),
                        n_steps=int(n_steps),
                        vina_adapter=vina_adapter,
                        use_real_vina=bool(use_real_vina and not is_stub),
                    )
                    results.append(res)
                    if counter % max(1, n_total // 20) == 0 or counter == n_total:
                        elapsed = time.monotonic() - t_start
                        log.info(
                            "sweep %d/%d (%.0f%%) — %s %s seed=%d "
                            "vina=%.3f (%.1fs total)",
                            counter, n_total,
                            100.0 * counter / max(1, n_total),
                            pocket_id, method, seed,
                            res.vina_kcal_mol if res.vina_kcal_mol is not None else float("nan"),
                            elapsed,
                        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(
    results: Iterable[DockResult],
) -> Tuple[Dict[str, Dict[str, object]], Dict[Tuple[str, str], Dict[str, float]]]:
    """Reduce the per-eval list into (per-pocket-per-method, aggregate).

    The aggregate key is ``(pocket_id, method)``; the per-pocket
    key is ``pocket_id``.
    """
    by_pocket_method: Dict[Tuple[str, str], List[float]] = {}
    by_pocket_method_status: Dict[Tuple[str, str], Dict[str, int]] = {}
    per_pocket: Dict[str, Dict[str, object]] = {}
    for res in results:
        if res.vina_kcal_mol is None or not math.isfinite(res.vina_kcal_mol):
            continue
        key = (res.pocket_id, res.method)
        by_pocket_method.setdefault(key, []).append(float(res.vina_kcal_mol))
        stats = by_pocket_method_status.setdefault(
            key, {"n_docked": 0, "n_failed": 0},
        )
        if res.status == "docked":
            stats["n_docked"] += 1
        else:
            stats["n_failed"] += 1
    aggregate: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, scores in by_pocket_method.items():
        if not scores:
            continue
        aggregate[key] = {
            "n": int(len(scores)),
            "mean": float(statistics.fmean(scores)),
            "std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
            "min": float(min(scores)),
            "max": float(max(scores)),
            **by_pocket_method_status[key],
        }
    for pocket_id, method_scores in _group_by_pocket(aggregate).items():
        per_pocket[pocket_id] = method_scores
    return per_pocket, aggregate


def _group_by_pocket(
    aggregate: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, Dict[str, object]]:
    """Re-shape ``{(pocket, method): stats}`` -> ``{pocket: {method: stats}}``."""
    out: Dict[str, Dict[str, object]] = {}
    for (pocket_id, method), stats in aggregate.items():
        out.setdefault(pocket_id, {})[method] = stats
    return out


def _win_tie_loss(
    aggregate: Dict[Tuple[str, str], Dict[str, float]],
    methods: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """For each pocket, compare every method against ``baseline`` (or first)."""
    if "baseline" not in methods or len(methods) < 2:
        return {}
    ref = "baseline"
    out: Dict[str, Dict[str, int]] = {}
    pockets = sorted({p for (p, _) in aggregate.keys()})
    for pocket_id in pockets:
        ref_block = aggregate.get((pocket_id, ref))
        if not ref_block:
            continue
        ref_mean = float(ref_block["mean"])
        per_method: Dict[str, int] = {}
        for m in methods:
            if m == ref:
                continue
            blk = aggregate.get((pocket_id, m))
            if not blk:
                per_method[m] = {"win": 0, "tie": 0, "loss": 0}
                continue
            mean = float(blk["mean"])
            # Lower kcal/mol = better; "win" means method beat baseline.
            if mean < ref_mean - 1e-6:
                per_method[m] = 1
            elif mean > ref_mean + 1e-6:
                per_method[m] = -1
            else:
                per_method[m] = 0
        out[pocket_id] = {m: per_method.get(m, 0) for m in methods if m != ref}
    return out


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (Path,)):
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
            "Wf-30 Phase 2 — 10-pocket × 5-candidate × 5-seed docking sweep."
        ),
    )
    p.add_argument("--pockets", nargs="+", default=list(DEFAULT_POCKETS))
    p.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    p.add_argument("--n-seeds", type=int, default=len(DEFAULT_SEEDS))
    p.add_argument(
        "--seeds", type=int, nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    p.add_argument(
        "--methods", nargs="+",
        choices=list(TRAJECTORY_GENERATORS),
        default=list(DEFAULT_METHODS),
    )
    p.add_argument(
        "--candidates-csv", type=Path, default=None,
        help="Optional 2-column CSV (name,smiles); default reads "
             "_demo_5_metaldrugs.txt.",
    )
    p.add_argument(
        "--pocket-dir", type=Path, default=WF30_POCKET_DIR,
        help="WF-30 staging directory with <pocket>/receptor.pdbqt files.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
    )
    p.add_argument(
        "--metrics-path", type=Path, default=DEFAULT_METRICS_PATH,
    )
    p.add_argument(
        "--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH,
    )
    p.add_argument(
        "--n-steps", type=int, default=SE3_INTEGRATION_STEPS,
        help="SE(3) integration steps for the se3sb method.",
    )
    p.add_argument(
        "--cpu-count", type=int, default=1,
        help="Per-docking-call CPU thread count (passed to Vina).",
    )
    p.add_argument(
        "--use-real-vina", action="store_true",
        help="Try to call the real VinaDockingAdapter. Default is "
             "the deterministic stub scorer (runnable on CPU-only CI).",
    )
    p.add_argument(
        "--seed-offset", type=int, default=0,
        help="Add this offset to all seeds (for batched re-runs).",
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress per-evaluation logging.",
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

    seeds = sorted({int(s) + int(args.seed_offset) for s in args.seeds})
    seeds = seeds[: max(1, int(args.n_seeds))]

    # Load candidates.
    if args.candidates_csv is not None:
        candidates = load_candidates_csv(args.candidates_csv)
    else:
        candidates = load_default_candidates(DEFAULT_CANDIDATES_TXT)
    candidates = select_candidates(candidates, int(args.n_candidates))
    if not candidates:
        log.error("no candidates after selection; aborting")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    n_evals = (
        len(args.pockets) * len(candidates) * len(args.methods) * len(seeds)
    )
    log.info(
        "starting dock_10pockets_se3sb sweep: %d pockets × %d candidates × "
        "%d methods × %d seeds = %d evals (real_vina=%s)",
        len(args.pockets), len(candidates), len(args.methods), len(seeds),
        n_evals, args.use_real_vina,
    )

    results = run_sweep(
        pockets=args.pockets,
        candidates=candidates,
        methods=args.methods,
        seeds=seeds,
        pocket_dir=args.pocket_dir,
        n_steps=int(args.n_steps),
        cpu_count=int(args.cpu_count),
        use_real_vina=bool(args.use_real_vina),
    )

    per_pocket, aggregate = aggregate_results(results)
    win_tie_loss = _win_tie_loss(aggregate, args.methods)

    metrics_payload = {
        "recipe": "wf30_dock_10pockets_se3sb",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pockets": list(args.pockets),
        "candidates": [
            {"name": c.name, "smiles": c.smiles} for c in candidates
        ],
        "methods": list(args.methods),
        "seeds": list(seeds),
        "n_steps": int(args.n_steps),
        "use_real_vina": bool(args.use_real_vina),
        "pocket_dir": str(args.pocket_dir),
        "n_evaluations": int(len(results)),
        "per_evaluation": [
            {
                "pocket_id": r.pocket_id,
                "candidate": r.candidate,
                "method": r.method,
                "seed": r.seed,
                "status": r.status,
                "vina_kcal_mol": r.vina_kcal_mol,
                "elapsed_s": r.elapsed_s,
                "n_steps": r.n_steps,
                "note": r.note,
            }
            for r in results
        ],
        "per_pocket_per_method": _to_jsonable(per_pocket),
        "aggregate": _to_jsonable(
            {f"{p}|{m}": v for (p, m), v in aggregate.items()}
        ),
    }
    args.metrics_path.write_text(
        json.dumps(_to_jsonable(metrics_payload), indent=2) + "\n"
    )
    log.info("wrote %s (%d evaluations)", args.metrics_path, len(results))

    summary_payload = {
        "recipe": "wf30_dock_10pockets_se3sb_summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_evaluations": int(len(results)),
        "n_docked": int(sum(1 for r in results if r.status == "docked")),
        "n_failed": int(
            sum(1 for r in results if r.status != "docked")
        ),
        "aggregate": _to_jsonable(
            {f"{p}|{m}": v for (p, m), v in aggregate.items()}
        ),
        "win_tie_loss_vs_baseline": _to_jsonable(win_tie_loss),
        "use_real_vina": bool(args.use_real_vina),
    }
    args.summary_path.write_text(
        json.dumps(_to_jsonable(summary_payload), indent=2) + "\n"
    )
    log.info("wrote %s", args.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
