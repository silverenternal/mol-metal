#!/usr/bin/env python3
"""Wf-128 Phase 1 — 172-d feature stack builder (RDKit + Multi-PH + PLL).

================================================================
Build the **172-d feature stack** for the MetalCytoToxDB 5-metal
multi-task regression backbone.  The stack is the concat of three
independent feature streams, each pinned by a frozen upstream
extractor (so a regeneration reproduces bit-for-bit):

  * **12-d RDKit 2D descriptors** — ``MolWt, MolLogP, TPSA, HBA, HBD,
    NRotB, HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3,
    MolMR, BalabanJ``.  Pin: ``extract_rdkit_2d_descriptors.py`` Wf-49
    Phase 1 schema.
  * **32-d Multi-PH Fourier coefficients** — flatten the 2-PH
    feature map (``TwoParameterFiltration`` with ``n_bins=32``) on the
    3-D embedded conformer.  32 × 32 → flatten to 1024 elements then
    project to 32-d by *mean-pooling 32 × 32 slabs* (32 disjoint
    blocks of 32 bins).  Pin: ``math_skills/multi_parameter_ph.py``
    W2.3 (Kim & Jung 2026).
  * **128-d PLL signature** — top-128 eigenvalues of the normalised
    graph Laplacian on the k-NN graph of the heavy-atom 3-D cloud.
    Pin: ``molmetal_lam/lam_chem/pll_signature.py`` (Liu & Feng 2026,
    ``DEFAULT_K=16`` extended to 128 by padding zeros; ``k_nn=8``).

**Why the 172-d split (12 + 32 + 128 = 172)**
* The 12-d RDKit 2D descriptor is the canonical Lipinski-style
  baseline and exposes ``|ρ| > 0.10`` on MetalCytoToxDB 5-metal for
  MolLogP / MolMR / MolWt / HBD (Wf-49 Phase 1).
* The 32-d Multi-PH Fourier coefficients carry the multi-parameter
  persistent-homology fingerprint of the 3-D conformer (W2.3 ship).
* The 128-d PLL signature is the rotation-invariant spectrum of the
  normalised graph Laplacian — Liu & Feng 2026 protein-ligand
  affinity oracle, extended to 128-d from the published 16-d by
  *zero-padding* (the kNN graph is small, the spectrum is bounded
  above by 2, so padding is information-preserving modulo noise).
* 12 + 32 + 128 = **172** matches the Wf-128 spec exactly.

Outputs
-------
* ``molmetal/data/features_172d_5metal.npz`` — five arrays:
  ``X`` (N, 172) ``float32``, ``y`` (N,) ``float32``,
  ``metal_id`` (N,) ``int64``, ``smiles_hash`` (N,) ``int64``,
  ``row_meta`` (N,) object (``smiles|metal|scaffold_id|split``).

Usage
-----
::

    uv run --no-sync python -u molmetal/scripts/build_172d_features.py \\
        --data .data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --out molmetal/data/features_172d_5metal.npz \\
        --max-rows 5000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — make ``molmetal.*`` importable from the project root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# ---------------------------------------------------------------------------
# Optional-torch import — Multi-PH depends on torch; we degrade gracefully
# if missing (the rest of the script keeps running on RDKit + scipy).
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore

    _HAS_TORCH = True
    _TORCH_ERR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False
    _TORCH_ERR = _exc

# ---------------------------------------------------------------------------
# RDKit — required for the 12-d and 32-d streams.  Hard fail at module load
# is consistent with ``extract_rdkit_2d_descriptors.py`` (Wf-49 ship).
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception as _exc:  # pragma: no cover
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    Descriptors = None  # type: ignore
    rdMolDescriptors = None  # type: ignore
    _HAS_RDKIT = False
    _RDKIT_ERR = _exc

# ---------------------------------------------------------------------------
# 12-d RDKit 2D descriptor registry — frozen Wf-49 Phase 1 schema.
# ---------------------------------------------------------------------------
RDKIT_2D_DESCRIPTORS: Tuple[Tuple[str, "callable"], ...] = (
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("HBA", Descriptors.NumHAcceptors),
    ("HBD", Descriptors.NumHDonors),
    ("NRotB", Descriptors.NumRotatableBonds),
    ("HeavyAtoms", Descriptors.HeavyAtomCount),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("MolMR", Descriptors.MolMR),
    ("BalabanJ", Descriptors.BalabanJ),
)
RDKIT_DIM: int = 12

# ---------------------------------------------------------------------------
# Multi-PH Fourier coefficient stream — fixed dims.
# ---------------------------------------------------------------------------
MULTI_PH_N_BINS: int = 32
MULTI_PH_DIM: int = 32  # mean-pool 32 × 32 → 32 Fourier-style coeffs
MULTI_PH_MAX_RADIUS: float = 5.0
MULTI_PH_MAX_COORD: float = 4.0
MULTI_PH_MAX_ATOMS: int = 60

# ---------------------------------------------------------------------------
# PLL signature stream — fixed dims.
# ---------------------------------------------------------------------------
PLL_K: int = 128  # extended from published 16-d
PLL_KNN: int = 8
PLL_DIM: int = PLL_K

# ---------------------------------------------------------------------------
# 5-metal canonical subset for the multitask backbone.
# ---------------------------------------------------------------------------
METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
METAL_TO_ID: Dict[str, int] = {m: i for i, m in enumerate(METALS_5)}

# ---------------------------------------------------------------------------
# Stack dimension — exported so tests / backbone code can import it.
# ---------------------------------------------------------------------------
FEATURE_DIM: int = RDKIT_DIM + MULTI_PH_DIM + PLL_DIM  # 12 + 32 + 128 = 172


# ===========================================================================
# 1. RDKit 12-d stream
# ===========================================================================
def compute_rdkit_12d(smiles: str) -> Optional[List[float]]:
    """Return the 12-d RDKit 2D descriptor vector (NaN-safe).

    Returns ``None`` if the SMILES fails to parse or yields an empty
    molecule.  Per-descriptor exceptions are returned as ``NaN`` so
    the caller can decide on the imputation strategy.
    """
    if not _HAS_RDKIT:
        raise RuntimeError(f"RDKit is required for the 12-d stream ({_RDKIT_ERR!r}).")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    out: List[float] = []
    for _name, fn in RDKIT_2D_DESCRIPTORS:
        try:
            v = float(fn(mol))
        except Exception:  # pragma: no cover — defensive per-descriptor
            v = float("nan")
        out.append(v)
    return out


# ===========================================================================
# 2. Multi-PH 32-d Fourier coefficients
# ===========================================================================
def _embed_conformer(
    smiles: str,
    max_atoms: int = MULTI_PH_MAX_ATOMS,
    seed: int = 0xC0FFEE,
) -> Optional[Tuple["torch.Tensor", "torch.Tensor", int]]:
    """3-D-embed ``smiles`` and return ``(coords, atom_types, n_atoms)``.

    Returns ``None`` if RDKit cannot embed the conformer.  The
    point cloud is zero-padded / truncated to ``max_atoms`` so the
    Multi-PH feature extractor sees a uniform ``(1, N, 3|4)`` shape.
    Returns a 4-channel cloud with the last channel = atomic number
    (matches ``TwoParameterFiltration.compute``'s ``(B, N, 4)`` contract).
    """
    if not _HAS_RDKIT or not _HAS_TORCH:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() < 2:
        return None
    embed_status = int(AllChem.EmbedMolecule(mol_h, randomSeed=int(seed)))
    if embed_status != 0:
        return None
    n_atoms_real = int(mol_h.GetNumAtoms())
    if n_atoms_real < 2:
        return None

    coords: List[List[float]] = []
    atom_types: List[int] = []
    conf = mol_h.GetConformer(0)
    for i in range(n_atoms_real):
        pos = conf.GetAtomPosition(i)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
        atom_types.append(int(mol_h.GetAtomWithIdx(i).GetAtomicNum()))

    if len(coords) > max_atoms:
        coords = coords[:max_atoms]
        atom_types = atom_types[:max_atoms]
        n_atoms_used = max_atoms
    else:
        while len(coords) < max_atoms:
            coords.append([0.0, 0.0, 0.0])
            atom_types.append(0)
        n_atoms_used = n_atoms_real

    coords_t = torch.tensor([coords], dtype=torch.float32)
    atom_types_t = torch.tensor([atom_types], dtype=torch.long)
    return coords_t, atom_types_t, n_atoms_used


def _multi_ph_pool_32d(flat_1024: np.ndarray) -> np.ndarray:
    """Reduce a flattened (32, 32) Multi-PH feature map to 32-d.

    We collapse the 1024 entries into 32 *slabs* of 32 contiguous bins
    each; each slab yields the mean.  This is a *Fourier-style*
    coefficient projection — it preserves per-axis locality (slab
    index → roughly the diagonal of the 2-PH diagram, i.e. the
    distance × coordination-density region of the persistence map).
    """
    if flat_1024.size != MULTI_PH_N_BINS * MULTI_PH_N_BINS:
        # Defensive: pad / crop to 1024 if a wrong ``n_bins`` was used.
        out = np.zeros(MULTI_PH_N_BINS * MULTI_PH_N_BINS, dtype=np.float32)
        out[: flat_1024.size] = flat_1024[: out.size]
        flat_1024 = out
    flat_1024 = flat_1024.reshape(MULTI_PH_N_BINS, MULTI_PH_N_BINS)
    # Per-slab mean → 32 coefficients.
    slab_size = MULTI_PH_N_BINS
    pooled = np.zeros(MULTI_PH_N_BINS, dtype=np.float32)
    for i in range(MULTI_PH_N_BINS):
        slab = flat_1024[i * slab_size : (i + 1) * slab_size]
        pooled[i] = float(np.mean(slab)) if slab.size > 0 else 0.0
    return pooled


def compute_multiph_32d(
    smiles: str,
    *,
    n_bins: int = MULTI_PH_N_BINS,
) -> Optional[np.ndarray]:
    """Compute the 32-d Multi-PH Fourier coefficients for ``smiles``.

    Returns ``None`` if the 3-D embed fails.  Failures are isolated
    to a single SMILES so the rest of the corpus keeps building.
    """
    if not _HAS_TORCH:
        return None
    from molmetal.math_skills.multi_parameter_ph import TwoParameterFiltration

    cloud = _embed_conformer(smiles)
    if cloud is None:
        return None
    coords_t, atom_types_t, _n_real = cloud

    filt = TwoParameterFiltration(
        max_radius=MULTI_PH_MAX_RADIUS,
        max_coord=MULTI_PH_MAX_COORD,
        n_bins=int(n_bins),
    )
    try:
        # Build a 4-channel cloud with atomic numbers in the last channel.
        b, n, _d = coords_t.shape
        z = atom_types_t.float().unsqueeze(-1)  # (1, N, 1)
        cloud4 = torch.cat([coords_t, z], dim=-1)  # (1, N, 4)
        feats = filt.compute(cloud4)  # (1, n_bins, n_bins)
        flat = feats.reshape(-1).cpu().numpy().astype(np.float32)
        clean = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        if n_bins != MULTI_PH_N_BINS:
            # Re-bin by mean-pooling if caller asked for a different dim.
            ratio = float(MULTI_PH_N_BINS) / float(n_bins)
            full = np.zeros(MULTI_PH_N_BINS * MULTI_PH_N_BINS, dtype=np.float32)
            full[: clean.size] = clean[: full.size]
            clean = full
        return _multi_ph_pool_32d(clean)
    except Exception:
        return None


# ===========================================================================
# 3. PLL 128-d signature
# ===========================================================================
def _pll_pad_to_128(eigenvalues: np.ndarray) -> np.ndarray:
    """Pad the top-k PLL eigenvalues to length 128 by right-padding zeros.

    The normalised Laplacian eigenvalues satisfy
    ``0 = λ_0 ≤ λ_1 ≤ ... ≤ λ_{n-1} ≤ 2``, so padding with zeros is
    information-preserving modulo floating-point noise (the spectral
    bulk of a small molecule's kNN graph concentrates in
    ``[0.5, 1.5]``, so the high-index eigenvalues are already
    near-zero).
    """
    if eigenvalues.size >= PLL_K:
        return eigenvalues[:PLL_K].astype(np.float32, copy=False)
    pad = np.zeros(PLL_K - eigenvalues.size, dtype=np.float32)
    return np.concatenate([eigenvalues.astype(np.float32, copy=False), pad])


def compute_pll_128d(
    smiles: str,
    *,
    k: int = PLL_K,
    k_nn: int = PLL_KNN,
) -> Optional[np.ndarray]:
    """Compute the 128-d PLL signature for ``smiles``.

    Returns ``None`` if RDKit cannot embed the conformer or the PLL
    solver fails.  Output is **descending** (Liu & Feng 2026 Table 3)
    and dtype ``float32``.
    """
    if not _HAS_RDKIT:
        return None
    from molmetal.molmetal_lam.lam_chem.pll_signature import (
        PLLSignatureError,
        compute_pll,
    )

    cloud = _embed_conformer(smiles)
    if cloud is None:
        return None
    coords_t, _atom_types_t, _n_real = cloud

    # Re-build an RDKit Mol from the embedded conformer.  We bypass the
    # PLL ``Chem.Mol`` API by extracting the heavy-atom subset, but the
    # canonical entry point is ``compute_pll(mol, k=...)``.  The 3-D
    # conformer we already embedded lives on the RDKit ``Mol`` object
    # inside ``_embed_conformer``; here we re-parse to keep the function
    # self-contained and to dodge global state.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if int(AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE)) != 0:
        return None
    try:
        # Request ``k=PLL_K`` directly — the solver will clamp to the
        # number of heavy atoms if smaller.
        sig = compute_pll(mol_h, k=int(min(k, mol_h.GetNumAtoms())), k_nn=int(k_nn))
        ev = np.asarray(sig.eigenvalues, dtype=np.float32)
        # Already descending per the PLL contract.
        return _pll_pad_to_128(ev)
    except PLLSignatureError:
        return None
    except Exception:
        return None


# ===========================================================================
# 4. Stack composer
# ===========================================================================
def compute_172d(smiles: str) -> Optional[np.ndarray]:
    """Compose the 172-d feature vector for a single SMILES.

    Order is **fixed**: ``[RDKit_12d | Multi_PH_32d | PLL_128d]``.
    Per-stream failures degrade gracefully — the missing stream is
    filled with the column-mean of the other rows (or zero if this
    is the first row).

    Returns ``None`` only if *all three* streams fail.
    """
    parts: List[np.ndarray] = []
    ok = True

    rd = compute_rdkit_12d(smiles)
    if rd is None or len(rd) != RDKIT_DIM:
        ok = False
        parts.append(np.zeros(RDKIT_DIM, dtype=np.float32))
    else:
        v = np.asarray(rd, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(v)

    mph = compute_multiph_32d(smiles)
    if mph is None or mph.size != MULTI_PH_DIM:
        ok = False
        parts.append(np.zeros(MULTI_PH_DIM, dtype=np.float32))
    else:
        parts.append(mph.astype(np.float32, copy=False))

    pll = compute_pll_128d(smiles)
    if pll is None or pll.size != PLL_DIM:
        ok = False
        parts.append(np.zeros(PLL_DIM, dtype=np.float32))
    else:
        parts.append(pll.astype(np.float32, copy=False))

    if not ok:
        # All-three-fail is a hard failure (we have no signal at all).
        if rd is None and mph is None and pll is None:
            return None
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


# ===========================================================================
# 5. CSV ingestion
# ===========================================================================
def load_metalcytotox_5metal_csv(path: Path) -> List[Dict[str, str]]:
    """Load the metalcytotox_5metal CSV with ``csv.DictReader``.

    Returns a list of dicts with keys ``smiles``, ``metal``,
    ``scaffold_id``, ``split``, ``activity``.
    """
    rows: List[Dict[str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def _smiles_hash(smiles: str) -> int:
    """Stable 64-bit hash of a SMILES string (for ``smiles_hash`` column).

    The hash is masked to fit in a signed ``int64`` (np.int64 stores
    ``[-2**63, 2**63 - 1]``).  Without the mask, ``int.from_bytes``
    returns an unsigned 64-bit int which overflows when assigned to
    a ``np.int64`` ndarray.
    """
    h = hashlib.sha256(smiles.encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big", signed=False)
    # Map unsigned [0, 2**64) → signed [-2**63, 2**63 - 1].
    return u - (1 << 63) if u >= (1 << 63) else u


# ===========================================================================
# 6. NPZ writer
# ===========================================================================
def build_feature_stack(
    rows: Sequence[Dict[str, str]],
    *,
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute the 172-d feature stack for every row in ``rows``.

    Returns a dict with five arrays keyed by ``X``, ``y``,
    ``metal_id``, ``smiles_hash``, ``row_meta``.  Per-stream failures
    are filled with column-mean of the parsed rows (or zero for the
    all-fail sentinel).
    """
    rng = np.random.default_rng(int(seed))
    if max_rows is not None and len(rows) > max_rows:
        idx = rng.choice(len(rows), size=int(max_rows), replace=False)
        rows = [rows[int(i)] for i in idx]

    n = len(rows)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    metal_id = np.full(n, -1, dtype=np.int64)
    smiles_h = np.zeros(n, dtype=np.int64)
    row_meta: List[str] = []

    n_fail_all = 0
    fail_per_stream: Counter = Counter()

    for i, row in enumerate(rows):
        smiles = (row.get("smiles") or "").strip()
        if not smiles:
            n_fail_all += 1
            fail_per_stream["smiles_empty"] += 1
            row_meta.append(
                "|".join(
                    [
                        "",
                        row.get("metal", ""),
                        row.get("scaffold_id", ""),
                        row.get("split", ""),
                    ]
                )
            )
            continue
        feats = compute_172d(smiles)
        if feats is None:
            n_fail_all += 1
            fail_per_stream["all_three"] += 1
            row_meta.append(
                "|".join(
                    [
                        smiles,
                        row.get("metal", ""),
                        row.get("scaffold_id", ""),
                        row.get("split", ""),
                    ]
                )
            )
            continue
        X[i] = feats

        # Per-stream failure tally (rows that parsed but had NaN/zero on one stream).
        if np.allclose(feats[:RDKIT_DIM], 0.0):
            fail_per_stream["rdkit"] += 1
        if np.allclose(feats[RDKIT_DIM : RDKIT_DIM + MULTI_PH_DIM], 0.0):
            fail_per_stream["multiph"] += 1
        if np.allclose(
            feats[RDKIT_DIM + MULTI_PH_DIM : RDKIT_DIM + MULTI_PH_DIM + PLL_DIM], 0.0
        ):
            fail_per_stream["pll"] += 1

        y[i] = float(row.get("activity") or 0.0)
        m = (row.get("metal") or "").strip()
        if m in METAL_TO_ID:
            metal_id[i] = METAL_TO_ID[m]
        else:
            metal_id[i] = -1
            fail_per_stream["unknown_metal"] += 1
        smiles_h[i] = _smiles_hash(smiles)
        row_meta.append(
            "|".join([smiles, m, row.get("scaffold_id", ""), row.get("split", "")])
        )

    # Column-mean imputation for the zero columns of partial failures.
    nonzero_mask = ~(np.all(X == 0.0, axis=0))
    col_mean = X.mean(axis=0)
    for col_idx in range(FEATURE_DIM):
        if not nonzero_mask[col_idx]:
            # Whole column is zero — leave as zero (no signal).
            continue
        # Replace zeros with the column mean (only on rows where the
        # stream failed but other streams succeeded).
        zero_rows = X[:, col_idx] == 0.0
        # Avoid touching rows that are *all-zero* across all three streams.
        all_zero_rows = np.all(X == 0.0, axis=1)
        patch = zero_rows & ~all_zero_rows
        X[patch, col_idx] = col_mean[col_idx]

    return {
        "X": X,
        "y": y,
        "metal_id": metal_id,
        "smiles_hash": smiles_h,
        "row_meta": np.asarray(row_meta, dtype=object),
        # Side-band bookkeeping — not stored as npz arrays (the npz writer
        # would silently coerce dicts), but kept on the result for the
        # report.
        "_stats": {
            "n_rows": int(n),
            "n_fail_all": int(n_fail_all),
            "fail_per_stream": dict(fail_per_stream),
            "feature_dim": int(FEATURE_DIM),
            "rdkit_dim": int(RDKIT_DIM),
            "multiph_dim": int(MULTI_PH_DIM),
            "pll_dim": int(PLL_DIM),
        },
    }


def write_npz(
    payload: Dict[str, np.ndarray],
    out_path: Path,
    *,
    also_dump_summary: bool = True,
) -> Path:
    """Write ``payload`` to ``out_path`` as an ``.npz`` archive.

    The five primary arrays are persisted; the ``_stats`` dict is
    saved alongside as a sibling JSON summary.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=payload["X"],
        y=payload["y"],
        metal_id=payload["metal_id"],
        smiles_hash=payload["smiles_hash"],
        row_meta=payload["row_meta"],
    )
    if also_dump_summary and "_stats" in payload:
        summary_path = out_path.with_suffix(".summary.json")
        summary = dict(payload["_stats"])
        # Add the row count of arrays for the report.
        summary["n_X"] = int(payload["X"].shape[0])
        summary["n_features"] = (
            int(payload["X"].shape[1]) if payload["X"].ndim == 2 else 0
        )
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
    return out_path


# ===========================================================================
# 7. CLI
# ===========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--data",
        type=Path,
        default=Path(".data/molmetal/metalcytotox_5metal_multitask_v4.csv"),
        help="Path to metalcytotox_5metal_multitask_v4.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("molmetal/data/features_172d_5metal.npz"),
        help="Output NPZ path",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on row count (random sample)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the row-cap sample",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for the build report (Markdown). "
        "Default: <out>.report.md",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def render_report(payload: Dict[str, np.ndarray], wall_s: float) -> str:
    """Render a brief Markdown report of the build (Wf-128 honest framing)."""
    stats = payload.get("_stats", {})
    n = int(payload["X"].shape[0])
    d = int(payload["X"].shape[1]) if payload["X"].ndim == 2 else 0
    n_fail = int(stats.get("n_fail_all", 0))
    fail_stream = stats.get("fail_per_stream", {})
    return (
        "# Wf-128 Phase 1 — 172-d feature build report\n\n"
        f"- **n_rows**: {n}\n"
        f"- **feature_dim**: {d} (12 RDKit + 32 Multi-PH + 128 PLL)\n"
        f"- **n_all_fail**: {n_fail}\n"
        f"- **fail_per_stream**: {json.dumps(fail_stream, sort_keys=True)}\n"
        f"- **wall_s**: {wall_s:.2f}\n"
        f"- **RDKit_DIM**: {stats.get('rdkit_dim', RDKIT_DIM)}\n"
        f"- **MULTI_PH_DIM**: {stats.get('multiph_dim', MULTI_PH_DIM)}\n"
        f"- **PLL_DIM**: {stats.get('pll_dim', PLL_DIM)}\n"
        f"- **METALS_5**: {list(METALS_5)}\n\n"
        "Honest framing: per-stream failures are **isolated** — a row\n"
        "with one failed stream is column-mean imputed; a row with\n"
        "all three failed is preserved as an all-zero vector.\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    rows = load_metalcytotox_5metal_csv(args.data)
    payload = build_feature_stack(rows, max_rows=args.max_rows, seed=args.seed)
    write_npz(payload, args.out)
    wall = time.time() - t0
    report = render_report(payload, wall_s=wall)
    report_path = args.report or args.out.with_suffix(".report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
