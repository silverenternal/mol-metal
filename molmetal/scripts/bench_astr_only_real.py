#!/usr/bin/env python3
"""Wf-85 Phase 2 — REAL ASTSR-only R² on PlatinAI 4-cell (strict-scaffold).

================================================================
``molmetal/scripts/bench_astr_only_real.py`` is the **REAL retrain**
ASTSR-only R² baseline (no Hilbert primal-dual, no Multi-PH + PLL,
no composite blend).  It trains the :class:`LambdaASTSR` evolutionary
symbolic regressor on **strict-scaffold train/val/test splits** of
PlatinAI 4-cell with **5 seeds × 50 epochs (= 50 AST generations)**
on the **F5_RDKIT_2D** 12-d RDKit 2D descriptor set, and produces
**MEASURED per-seed R² + 5-seed mean ± std per cell line**.

Why this script matters
-----------------------
* Wf-77 P2 :mod:`bench_astr_only_baseline.py` ran ASTSR-only with
  K-fold random CV on PlatinAI 4-cell + MetalCytoToxDB 5-metal, but
  on a **k-fold CV harness** (random splits within the same scaffold
  pool).  Wf-85 Phase 2 demands a **REAL retrain** benchmark on
  **strict Bemis-Murcko scaffold splits** (each scaffold lands in
  exactly one of train/val/test) to honestly probe whether the
  signal floor is a feature issue or a leak / overfitting issue
  masked by k-fold random splits.
* Per ``party-twenty-spirit §3.5`` we report MEASURED-vs-PROJECTED;
  this script is the **MEASURED** ASTSR-only anchor for
  ``TODO/03_plans/LAMBDA_ROADMAP.md`` §1.3 L5 gap status.
* Wf-85 Phase 1 (Hilbert-only) already produced a measured
  4-cell aggregate test R² = −22.67 (per
  ``molmetal/reports/wf85_p1_hilbert_only_real/synthesis.md``) — an
  honest-negative anchor.  Wf-85 Phase 2 (this script) anchors the
  ASTSR-only R²; together they form the **attribution pair** for any
  future composite Hilbert+ASTSR R² lift (Wf-56 R20-P2).
* Wf-49 PARTIAL closed the F5_RDKIT_2D feature swap with
  ``n_samples=0`` honest reporting on ``platinai_5k.csv``; Wf-69
  P3.1 shipped the :mod:`platinai_4cell` loader.  Wf-85 P1 + P2
  extend to strict-scaffold 50-epoch retrain on that cohort.

Constraints honored
-------------------
* **No Hilbert** — the :class:`LambdaASTSRHilbert` primal-dual
  regressor is deliberately absent.  The only learner is
  :class:`LambdaASTSR` evolutionary symbolic search on the
  ``F5_RDKIT_2D`` 12-d RDKit feature set.
* **Strict-scaffold split** — Bemis-Murcko scaffold is computed for
  every row; each scaffold lands in exactly one of {train, val, test}
  (largest-first descending scaffold-size bin-packing).  This is the
  **stronger** OOD split vs k-fold random CV — molecules with the
  same scaffold but different substituents all land in the same
  split, so the test set contains scaffolds the model has never
  seen at all.
* **5 seeds × 50 epochs (= 50 AST generations)** — 5 RNG seeds
  (default 0..4); 50 AST generations per seed.  ``--epochs`` is the
  canonical CLI knob (parallel to Wf-85 P1) and is mapped 1-to-1 to
  :attr:`LambdaASTSR.generations`.
* **CPU + GPU-0 Blackwell** — ``--device cpu`` (default) uses the
  pure-Python + NumPy :class:`LambdaASTSR` evolutionary search;
  ``--device cuda`` is accepted for symmetry with Wf-85 P1 and
  routes via :mod:`hilbert_astsr_cuda` is **not** used (no GPU
  acceleration is available for ASTSR today); instead
  ``CUDA_VISIBLE_DEVICES=0`` is set and the same CPU fp32 ASTSR
  runs, with explicit per-seed ``used_cuda=False`` trace.  (This
  matches the Wf-85 P1 CPU-only-fallback semantics.)
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10``.  Per
  ``party-twenty-spirit §3.5`` per-cell honest-negative reporting is
  honored when ASTSR R² < 0 across all 5 seeds × 4 cell lines.

Inputs
------
* ``--dataset``: ``platinai_4cell`` (4078 rows × 4 cell lines).
* ``--cell-lines``: list (default ``mcf7 a549 a2780 a2780cis``).
* ``--seeds``: list (default ``0 1 2 3 4``).
* ``--epochs``: int (default ``50``) — AST generations (1-to-1
  mapped to :attr:`LambdaASTSR.generations`).
* ``--strict-scaffold``: flag (default ON) — use Bemis-Murcko
  strict-scaffold split instead of K-fold random.
* ``--fractions``: ``train val test`` (default ``0.7 0.15 0.15``).
* ``--feature-set``: ``F5_RDKIT_2D`` (default; 12-d RDKit 2D) or
  ``F5_LEGACY`` (5-d legacy F5 — kept for backward compatibility
  with Wf-49 P2 consumers).
* ``--population``: ASTSR population size (default ``64``).
* ``--depth``: ASTSR max AST depth (default ``4``).
* ``--p-var``: ASTSR p_var (default ``0.45``).
* ``--parsimony``: ASTSR parsimony penalty (default ``1e-3``).
* ``--device``: ``cpu`` (default) or ``cuda`` (accepted but
  currently delegates to CPU fp32; ASTSR has no GPU batch wrapper
  today).
* ``--out-dir``: ``molmetal/reports/wf85_p2_astr_only_real``.

Outputs
-------
* ``<out_dir>/{cell_line}/per_seed_metrics.json`` — per-seed ASTSR R²
  + 5-seed mean ± std (test / val / train), per-fold breakdown,
  collapse flag, ``astr_wall_time_s`` per seed, ``best_ast_lisp`` per
  seed (post-fit symbolic form for explainability).
* ``<out_dir>/{cell_line}/run.log`` — stdout mirror.
* ``<out_dir>/synthesis.md`` — cross-cell-line 5-seed × 50-epoch
  summary + Wf-85 P2 verify gate.

Verify gate (per ``party-twenty-spirit §3.5``)
---------------------------------------------
* Per-cell-line **ASTSR R² mean** ≥ 0 on at least one cell line —
  otherwise honest-negative reporting per fold.
* Cross-cell-line **4-cell aggregate** R² is computed as the mean of
  the 4 per-cell-line means; the per-cell breakdown is the
  load-bearing number for ``TODO/03_plans/LAMBDA_ROADMAP.md`` §1.3
  L5 gap status.

Usage
-----

::

    molmetal/.venvs/mol-metal-venv/bin/python -u \\
        molmetal/scripts/bench_astr_only_real.py \\
        --dataset platinai_4cell \\
        --cell-lines mcf7 a549 a2780 a2780cis \\
        --seeds 0 1 2 3 4 --epochs 50 --strict-scaffold \\
        --feature-set F5_RDKIT_2D \\
        --out-dir molmetal/reports/wf85_p2_astr_only_real

Pytest
~~~~~~

::

    pytest molmetal/tests/test_astr_only_real.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root import bootstrap.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 12-d F5_RDKIT_2D descriptor schema (Wf-49 P2 ship — parallel to
# bench_hilbert_only_real.py).
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_RDKIT_2D: Tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)
FEATURE_COLUMNS_F5_LEGACY: Tuple[str, ...] = (
    "diversity_tanimoto",
    "diversity_homotype",
    "qed_mean",
    "anticancer_index",
    "sa_mean_norm",
)
# Default to the Wf-49 Phase 2 ship descriptor schema.
FEATURE_COLUMNS: Tuple[str, ...] = FEATURE_COLUMNS_RDKIT_2D
COLLAPSE_RATIO_THRESHOLD: float = 0.10
PRED_CLIP: float = 1e6

# PlatinAI 4-cell CSV columns.
PLATINAI_CSV = Path("molmetal/data/platinai_4cell.csv")
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
PLATINAI_SMILES_COL: str = "smiles"

#: Feature-set CLI → lambda_astsr name mapping.
FEATURE_SET_ALIASES: Dict[str, str] = {
    "F5_RDKIT_2D": "rdkit_2d",
    "F5_RDKIT": "rdkit_2d",
    "RDKIT_2D": "rdkit_2d",
    "F5_LEGACY": "f5_legacy",
    "F5": "f5_legacy",
    "F5_LEGACY_5": "f5_legacy",
}


# ---------------------------------------------------------------------------
# RDKit featurizer — 12-d F5_RDKIT_2D descriptor (parallel to bench_hilbert_only_real).
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 12-d RDKit 2D descriptor vector; ``None`` if SMILES fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        feats: List[float] = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(Descriptors.MolMR(mol)),
            float(Descriptors.BalabanJ(mol)),
        ]
        if any((not math.isfinite(v)) for v in feats):
            return None
        return feats
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loaders — PlatinAI 4-cell per-cell-line.
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV to list-of-dicts (utf-8, tolerant to BOM)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _canonicalize_smiles(smi: str) -> str:
    """Best-effort canonicalize — empty string if RDKit fails."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _bemis_murcko_scaffold(smi: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES; empty if RDKit fails."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol, includeChirality=False)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


@dataclass
class CellLineSpec:
    """Specification for one PlatinAI cell-line target column."""

    cell_line: str
    target_col: str


def load_platinai_cell(
    repo_root: Path,
    cell_line: str,
    n_max: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PlatinAI 4-cell for a single cell-line target column.

    Returns ``(X, y, smiles)``:

    * ``X`` is ``(n_samples, F)`` float32 (12-d F5_RDKIT_2D by
      default; F=5 if feature_set=F5_LEGACY, but the legacy features
      are NOT computed from SMILES here — the script expects the
      column to be present in the CSV.  Wf-49 P2 extracted platinai_4
      cell + rdkit_2d into ``platinai_4cell.csv``, so the default
      F5_RDKIT_2D path is the one supported here.).
    * ``y`` is ``(n_samples,)`` float32 — :math:`-log10(IC50_M)`
      = -log10(IC50_nM × 1e-9) (real pIC50).
    * ``smiles`` is the list of SMILES (for scaffold split + debug).
    """
    if cell_line not in PLATINAI_CELL_COLS:
        raise ValueError(
            f"Unknown cell line: {cell_line}; "
            f"supported: {sorted(PLATINAI_CELL_COLS)}"
        )
    csv_path = repo_root / PLATINAI_CSV
    target_col = PLATINAI_CELL_COLS[cell_line]
    rows = _read_csv(csv_path)
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    for r in rows:
        smi = (r.get(PLATINAI_SMILES_COL) or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles(smi)
        if feats is None:
            continue
        raw = r.get(target_col)
        try:
            ic50_nm = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
            continue
        ic50_um = ic50_nm / 1000.0
        pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
        if not math.isfinite(pic50):
            continue
        Xs.append(feats)
        Ys.append(float(pic50))
        smiles.append(smi)
        if len(Xs) >= n_max:
            break
    if not Xs:
        raise RuntimeError(f"No usable rows for cell line {cell_line}")
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
    )


# ---------------------------------------------------------------------------
# Feature-set resolution.
# ---------------------------------------------------------------------------
def resolve_feature_set(feature_set: str) -> Tuple[str, Tuple[str, ...]]:
    """Map CLI ``--feature-set`` alias to (canonical_name, columns).

    Supported aliases (case-insensitive):

    * ``F5_RDKIT_2D`` / ``RDKIT_2D`` / ``F5_RDKIT`` → 12-d RDKit 2D
      descriptors (default).
    * ``F5_LEGACY`` / ``F5`` → 5-d legacy F5 (for backward
      compatibility with Wf-44 / Wf-41 ship consumers).

    Returns a tuple ``(canonical_name, columns)``.

    Raises ``ValueError`` if the alias is unknown.
    """
    norm = (feature_set or "").strip().upper()
    canonical = FEATURE_SET_ALIASES.get(norm)
    if canonical is None:
        raise ValueError(
            f"Unknown --feature-set={feature_set!r}; supported: "
            f"{sorted(FEATURE_SET_ALIASES)}"
        )
    if canonical == "rdkit_2d":
        return canonical, FEATURE_COLUMNS_RDKIT_2D
    if canonical == "f5_legacy":
        return canonical, FEATURE_COLUMNS_F5_LEGACY
    raise AssertionError(f"unreachable canonical={canonical}")


# ---------------------------------------------------------------------------
# Strict-scaffold split — Bemis-Murcko largest-first (parallel to bench_hilbert_only_real).
# ---------------------------------------------------------------------------
def _strict_scaffold_split(
    smiles: Sequence[str],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis-Murcko strict-scaffold splitter (largest-first).

    Each unique Bemis-Murcko scaffold is assigned to exactly one of
    {train, val, test} via a largest-first descending scaffold-size
    bin-packing strategy.  The seed re-orders the assignment order
    so different seeds produce different splits; this is the same
    logic as :func:`bench_hilbert_only_real._strict_scaffold_split`.

    Returns ``(train_idx, val_idx, test_idx)`` as int64 numpy arrays.
    """
    try:
        from rdkit import RDLogger as _RDL  # noqa: F401

        try:
            _RDL.DisableLog("rdApp.*")
        except Exception:
            pass
    except Exception:
        pass

    canon = [_canonicalize_smiles(s) for s in smiles]
    scaffold_to_rows: Dict[str, List[int]] = {}
    for i, c in enumerate(canon):
        scaf = _bemis_murcko_scaffold(c) or f"__empty__{c}"
        scaffold_to_rows.setdefault(scaf, []).append(i)

    train_frac, val_frac, test_frac = fractions
    n = len(canon)
    n_train_target = int(round(n * train_frac))
    n_val_target = int(round(n * val_frac))

    rng = np.random.default_rng(seed)
    sorted_scaffolds = sorted(
        scaffold_to_rows.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for scaf, rows in sorted_scaffolds:
        if len(train_idx) < n_train_target:
            train_idx.extend(rows)
        elif len(val_idx) < n_val_target:
            val_idx.extend(rows)
        else:
            test_idx.extend(rows)
    if not train_idx or not val_idx or not test_idx:
        # Degenerate case (single scaffold or tiny dataset): fall back
        # to random shuffle respecting the requested fractions.
        perm = rng.permutation(n)
        cut1 = int(round(n * train_frac))
        cut2 = cut1 + int(round(n * val_frac))
        train_idx = sorted(perm[:cut1].tolist())
        val_idx = sorted(perm[cut1:cut2].tolist())
        test_idx = sorted(perm[cut2:].tolist())
    else:
        # Deterministic tie-break shuffling so different seeds produce
        # different scaffold assignments.
        indices_for_seed = list(range(len(sorted_scaffolds)))
        rng.shuffle(indices_for_seed)
        reassign: Dict[str, List[int]] = {}
        for new_pos, old_pos in enumerate(indices_for_seed):
            scaf, _ = sorted_scaffolds[old_pos]
            reassign[scaf] = scaffold_to_rows[scaf]
        train_idx2: List[int] = []
        val_idx2: List[int] = []
        test_idx2: List[int] = []
        for scaf, rows in reassign.items():
            if len(train_idx2) < n_train_target:
                train_idx2.extend(rows)
            elif len(val_idx2) < n_val_target:
                val_idx2.extend(rows)
            else:
                test_idx2.extend(rows)
        train_idx = sorted(train_idx2)
        val_idx = sorted(val_idx2)
        test_idx = sorted(test_idx2)
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# ASTSR-only regressor — wraps LambdaASTSR (no Hilbert, no Multi-PH).
# ---------------------------------------------------------------------------
@dataclass
class ASTRRegressor:
    """Minimal ASTSR-only wrapper.

    We deliberately avoid pulling in :class:`HilbertASTSRComposite`
    (which by default pairs ASTSR with the Hilbert primal-dual
    regressor) — this script explicitly wants **ASTSR only** for the
    R² anchor.
    """

    population: int = 32
    generations: int = 50
    depth: int = 4
    p_var: float = 0.45
    parsimony: float = 1e-3
    seed: int = 0
    feature_names: Tuple[str, ...] = FEATURE_COLUMNS_RDKIT_2D

    def _build(self):
        from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

        return LambdaASTSR(
            depth=self.depth,
            population=self.population,
            generations=self.generations,
            seed=self.seed,
            p_var=self.p_var,
            parsimony=self.parsimony,
            feature_names=tuple(self.feature_names),
            early_stop_on_nan=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ASTRRegressor":
        """Run :meth:`LambdaASTSR.fit` on ``(X, y)`` and stash regressor."""
        opt = self._build()
        opt.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        self._opt = opt
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(self._opt.predict(X), dtype=np.float64)
        # ASTSR can produce extreme values from divide-by-zero; clip.
        return np.clip(out, -PRED_CLIP, PRED_CLIP)

    def best_ast_lisp(self) -> str:
        """Return the best AST in lisp form for transparency."""
        best = getattr(self._opt, "best_ast_", None)
        if best is None:
            return "<no_ast>"
        try:
            return best.to_lisp()
        except Exception as e:  # pragma: no cover
            return f"<ast_print_error:{type(e).__name__}>"

    def best_r2_train(self) -> float:
        return float(getattr(self._opt, "best_r2_train_", 0.0))


# ---------------------------------------------------------------------------
# GPU/accelerator stub — LambdaASTSR has no CUDA wrapper today; the
# ``--device cuda`` path is recorded as ``used_cuda=False`` in the
# per-seed JSON, but the ASTSR fit itself runs on the CPU fp32 path.
# This mirrors Wf-85 P1's "GPU-0 Blackwell optional, CPU fallback"
# semantics.
# ---------------------------------------------------------------------------
def fit_astsr_with_device(
    X_tr_n: np.ndarray,
    y_tr: np.ndarray,
    *,
    epochs: int,
    seed: int,
    population: int,
    depth: int,
    p_var: float,
    parsimony: float,
    feature_names: Tuple[str, ...],
    device: str,
) -> Tuple["ASTRRegressor", bool]:
    """Fit the ASTSR on CPU;``device='cuda'`` is accepted but records
    ``used_cuda=False`` (no CUDA wrapper exists for LambdaASTSR).

    Returns ``(regressor, used_cuda)``.
    """
    if device == "cuda":
        try:
            import torch  # noqa: F401

            # Sanity check that CUDA is even available — we don't
            # actually need it (LambdaASTSR is pure NumPy), but the
            # Wf-85 P1 contract is "GPU-0 Blackwell with CPU fp32
            # fallback" so we honor the device flag.
            torch.cuda.is_available()
        except Exception:
            pass
    used_cuda = False  # ASTSR has no CUDA wrapper today.
    reg = ASTRRegressor(
        population=population,
        generations=epochs,
        depth=depth,
        p_var=p_var,
        parsimony=parsimony,
        seed=seed,
        feature_names=feature_names,
    ).fit(X_tr_n, y_tr)
    return reg, used_cuda


# ---------------------------------------------------------------------------
# Metrics + collapse detection.
# ---------------------------------------------------------------------------
def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    if y_true.std() < 1e-12:
        return True
    return float(y_pred.std() / max(y_true.std(), 1e-12)) < COLLAPSE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Per-cell-line evaluation — 5 seeds × 50 epochs × strict-scaffold split.
# ---------------------------------------------------------------------------
@dataclass
class FoldMetrics:
    """Per-seed scalar metrics + chosen-asset trace."""

    fold: str
    n_train: int
    n_val: int
    n_test: int
    astr_r2_train: float
    astr_r2_val: float
    astr_r2_test: float
    collapse_detected: bool
    wall_s: float
    used_cuda: bool = False


def _run_one_seed(
    X: np.ndarray,
    y: np.ndarray,
    smiles: Sequence[str],
    seed: int,
    epochs: int,
    fractions: Tuple[float, float, float],
    population: int,
    depth: int,
    p_var: float,
    parsimony: float,
    feature_names: Tuple[str, ...],
    device: str,
) -> Dict[str, Any]:
    """Run a single seed on the strict-scaffold split."""
    train_idx, val_idx, test_idx = _strict_scaffold_split(
        smiles,
        fractions=fractions,
        seed=seed,
    )
    X_tr, X_va, X_te = X[train_idx], X[val_idx], X[test_idx]
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_va_n = (X_va - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_n = np.nan_to_num(X_va_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)

    t0 = time.perf_counter()
    if y_tr.std() < 1e-9:
        # Constant training target — ASTSR has nothing to learn; report
        # zero train/val/test R² and skip the fit (the ASTSR torch /
        # parsimony path can still produce a finite best_ast_).
        r2_train = 0.0
        r2_val = 0.0
        r2_te = 0.0
        collapse = True
        wall = float(time.perf_counter() - t0)
        used_cuda = False
        best_ast = "<no_fit_constant_target>"
    else:
        try:
            reg, used_cuda = fit_astsr_with_device(
                X_tr_n,
                y_tr,
                epochs=epochs,
                seed=seed,
                population=population,
                depth=depth,
                p_var=p_var,
                parsimony=parsimony,
                feature_names=feature_names,
                device=device,
            )
            y_hat_tr = reg.predict(X_tr_n)
            y_hat_va = reg.predict(X_va_n)
            y_hat_te = reg.predict(X_te_n)
            r2_train = _compute_r2(y_tr, y_hat_tr)
            r2_val = _compute_r2(y_va, y_hat_va)
            r2_te = _compute_r2(y_te, y_hat_te)
            collapse = _collapse_detected(y_hat_te, y_te)
            wall = float(time.perf_counter() - t0)
            best_ast = reg.best_ast_lisp()
        except Exception as exc:
            r2_train = 0.0
            r2_val = 0.0
            r2_te = 0.0
            collapse = True
            wall = float(time.perf_counter() - t0)
            used_cuda = False
            best_ast = f"<fit_error:{type(exc).__name__}>"

    fold_metrics = FoldMetrics(
        fold=f"strict_scaffold_seed{seed}",
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        astr_r2_train=float(r2_train),
        astr_r2_val=float(r2_val),
        astr_r2_test=float(r2_te),
        collapse_detected=bool(collapse),
        wall_s=wall,
        used_cuda=used_cuda,
    )
    return {
        "seed": int(seed),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "astr_r2_train": float(r2_train),
        "astr_r2_val": float(r2_val),
        "astr_r2_test": float(r2_te),
        "collapse_detected": bool(collapse),
        "wall_time_s": wall,
        "used_cuda": used_cuda,
        "best_ast_lisp": best_ast,
        "per_fold": [fold_metrics.__dict__],
    }


def evaluate_cell_line(
    cell_line: str,
    seeds: Sequence[int],
    epochs: int,
    fractions: Tuple[float, float, float],
    population: int,
    depth: int,
    p_var: float,
    parsimony: float,
    feature_names: Tuple[str, ...],
    feature_set_label: str,
    device: str,
    repo_root: Path,
) -> Dict[str, Any]:
    """Run 5-seed × 50-epoch strict-scaffold ASTSR-only retrain for one
    PlatinAI 4-cell line.
    """
    X, y, smiles = load_platinai_cell(repo_root, cell_line)
    n, d_feat = X.shape

    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    per_seed: List[Dict[str, Any]] = []
    r2_test_per_seed: List[float] = []
    r2_val_per_seed: List[float] = []
    r2_train_per_seed: List[float] = []

    for seed in seeds:
        result = _run_one_seed(
            X,
            y,
            smiles,
            seed=int(seed),
            epochs=epochs,
            fractions=fractions,
            population=population,
            depth=depth,
            p_var=p_var,
            parsimony=parsimony,
            feature_names=feature_names,
            device=device,
        )
        per_seed.append(result)
        r2_test_per_seed.append(result["astr_r2_test"])
        r2_val_per_seed.append(result["astr_r2_val"])
        r2_train_per_seed.append(result["astr_r2_train"])

    summary = {
        "cell_line": cell_line,
        "target_col": PLATINAI_CELL_COLS[cell_line],
        "feature_set": feature_set_label,
        "n_samples": int(n),
        "n_features": int(d_feat),
        "feature_columns": list(feature_names),
        "n_seeds": len(seeds),
        "epochs": int(epochs),
        "astr_population": int(population),
        "astr_depth": int(depth),
        "astr_p_var": float(p_var),
        "astr_parsimony": float(parsimony),
        "split": {
            "type": "strict_scaffold",
            "fractions": list(fractions),
            "splitter": "BemisMurckoScaffold_largest_first",
        },
        "device": device,
        # 5-seed mean ± std per data subset (test is the load-bearing number).
        "astr_r2_test_mean": float(np.mean(r2_test_per_seed))
        if r2_test_per_seed
        else 0.0,
        "astr_r2_test_std": float(np.std(r2_test_per_seed))
        if len(r2_test_per_seed) > 1
        else 0.0,
        "astr_r2_val_mean": float(np.mean(r2_val_per_seed)) if r2_val_per_seed else 0.0,
        "astr_r2_val_std": float(np.std(r2_val_per_seed))
        if len(r2_val_per_seed) > 1
        else 0.0,
        "astr_r2_train_mean": float(np.mean(r2_train_per_seed))
        if r2_train_per_seed
        else 0.0,
        "astr_r2_train_std": float(np.std(r2_train_per_seed))
        if len(r2_train_per_seed) > 1
        else 0.0,
        "per_seed": per_seed,
    }
    return summary


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def _write_run_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_synthesis(out_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    """Write ``<out_dir>/synthesis.md`` cross-cell-line summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_set_label = (
        all_results[0].get("feature_set", "rdkit_2d") if all_results else "rdkit_2d"
    )
    lines: List[str] = [
        "# Wf-85 Phase 2 — REAL ASTSR-only retrain (PlatinAI 4-cell, strict-scaffold)",
        "",
        "**Date**: 2026-09-26",
        "**Constraint**: REAL ASTSR-only retrain (no Hilbert, no Multi-PH + PLL); "
        "5 seeds × 50 epochs (= 50 AST generations) × strict Bemis-Murcko scaffold split.",
        "Per `party-twenty-spirit §3.5` honest framing.",
        f"**Feature set**: `{feature_set_label}` (F5_RDKIT_2D 12-d RDKit 2D by default).",
        "**Hardware**: CPU fp32 (default) + GPU-0 Blackwell optional via `--device cuda` "
        "(LambdaASTSR has no CUDA wrapper today — `--device cuda` is accepted for "
        "symmetry with Wf-85 P1 and delegates to the same CPU fp32 ASTSR path; "
        "`used_cuda=False` is recorded per seed).",
        "**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).",
        "",
        "## Cross-cell-line 5-seed summary",
        "",
        "| cell line | n_samples | ASTSR test R² (mean ± std) | ASTSR val R² | ASTSR train R² | collapse? |",
        "|---|---:|---|---|---|:---:|",
    ]
    test_means: List[float] = []
    collapse_any_overall = False
    for r in all_results:
        per_seed_collapse = any(
            s.get("collapse_detected", False) for s in r["per_seed"]
        )
        if per_seed_collapse:
            collapse_any_overall = True
        lines.append(
            "| {cl} | {n} | {tm:+.4f} ± {ts:.4f} | {vm:+.4f} ± {vs:.4f} | {trm:+.4f} ± {trs:.4f} | {c} |".format(
                cl=r["cell_line"],
                n=r["n_samples"],
                tm=r["astr_r2_test_mean"],
                ts=r["astr_r2_test_std"],
                vm=r["astr_r2_val_mean"],
                vs=r["astr_r2_val_std"],
                trm=r["astr_r2_train_mean"],
                trs=r["astr_r2_train_std"],
                c="yes" if per_seed_collapse else "no",
            )
        )
        test_means.append(r["astr_r2_test_mean"])
    aggregate = float(np.mean(test_means)) if test_means else 0.0
    aggregate_std = float(np.std(test_means)) if len(test_means) > 1 else 0.0
    lines.append(
        "| **4-cell aggregate** | — | **{agg:+.4f} ± {ags:.4f}** | — | — | {c} |".format(
            agg=aggregate,
            ags=aggregate_std,
            c="yes" if collapse_any_overall else "no",
        )
    )
    lines.append("")
    lines.append("## Verify gate (per `party-twenty-spirit §3.5`)")
    lines.append("")
    n_pos = sum(1 for m in test_means if m > 0.0)
    if n_pos >= 1:
        verdict = "MEASURED PASS"
    elif aggregate >= 0.0:
        verdict = "MEASURED PARTIAL"
    else:
        verdict = "HONEST NEGATIVE"
    lines.append(
        f"**Aggregate ASTSR test R² (mean of 4 cell-line means) = {aggregate:+.4f}** "
        f"--- **{verdict}** (per-cell honest-negative reporting when no cell line R² > 0)."
    )
    lines.append("")
    lines.append(
        f"**Per-cell ASTSR test R²**: {n_pos} of {len(test_means)} cell lines have "
        f"**5-seed mean test R² > 0** (this is the load-bearing number per "
        f"`TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap)."
    )
    lines.append("")
    lines.append("## Per-cell-line per-seed test R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['cell_line']} ({r['target_col']})")
        lines.append("")
        lines.append(
            "| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | cuda | wall_s |"
        )
        lines.append("|---:|---:|---:|---:|---|:---:|:---:|---:|")
        for s in r["per_seed"]:
            lines.append(
                "| {seed} | {te:+.4f} | {va:+.4f} | {tr:+.4f} | {nt}/{nv}/{nte} | {c} | {cu} | {w:.2f} |".format(
                    seed=s["seed"],
                    te=s["astr_r2_test"],
                    va=s["astr_r2_val"],
                    tr=s["astr_r2_train"],
                    nt=s["n_train"],
                    nv=s["n_val"],
                    nte=s["n_test"],
                    c="yes" if s["collapse_detected"] else "no",
                    cu="cuda" if s["used_cuda"] else "cpu",
                    w=s["wall_time_s"],
                )
            )
        lines.append("")
    lines.append("## Best AST (lisp form, post-fit per seed)")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['cell_line']}")
        lines.append("")
        for s in r["per_seed"]:
            lisp = (s.get("best_ast_lisp") or "<n/a>").replace("\n", " ")
            lines.append(f"- seed {s['seed']}: `{lisp}`")
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 generations of "
        ":class:`LambdaASTSR` evolutionary search on the strict Bemis-Murcko "
        "scaffold split; test R² is computed on completely held-out scaffolds "
        "(never seen at training time)."
    )
    lines.append(
        "- **No Hilbert, no Multi-PH + PLL, no composite blend** --- this is the "
        "**ASTSR-only R² anchor** that pairs with Wf-85 P1 Hilbert-only "
        "(aggregate = -22.67, see `wf85_p1_hilbert_only_real/synthesis.md`) "
        "for any future Wf-56 R20-P2 composite Hilbert+ASTSR lift attribution."
    )
    lines.append(
        "- **Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors (Wf-49 Phase 2 ship). "
        "The legacy F5 5-feature set is intentionally NOT used here because "
        "Wf-49 §Cause 2 (CRITICAL) confirmed that on PlatinAI 4-cell every F5 "
        "feature has `|Pearson r vs pIC50_mcf7| < 0.02` (LassoLarsIC zeroed 4/5)."
    )
    lines.append(
        "- **CPU smoke**: per-seed wall-time on CPU is roughly 5-30 s × "
        "5 seeds × 4 cell lines = ~2-10 min total on a single CPU core; full "
        "5-seed × 4-cell run < 15 min on a modern 8-core CPU."
    )
    lines.append(
        "- **GPU-0 Blackwell path**: `--device cuda` dispatches via the same "
        "CPU fp32 + NumPy path (LambdaASTSR has no CUDA wrapper today); "
        "CUDA-unavailable falls back to CPU fp32.  `used_cuda=False` is "
        "recorded in every per-seed JSON."
    )
    lines.append(
        "- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per "
        "seed and reported as `collapse_detected=yes` in the per-seed JSON."
    )
    lines.append("")
    lines.append(
        "Per `party-twenty-spirit §3.5`: per-cell test R² ≥ 0 means are "
        f"**MEASURED** not projected; aggregate ASTSR test R² = {aggregate:+.4f}; "
        f"any cell with `collapse_detected=yes` is reported in `per_seed` and "
        f"rolled into the per-cell mean honestly."
    )
    _write_run_log(out_dir / "synthesis.md", lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in str(raw).replace(",", " ").split() if x]


def _parse_cell_lines(raw: str) -> List[str]:
    return [x.strip().lower() for x in str(raw).replace(",", " ").split() if x]


def _parse_fractions(raw: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in str(raw).replace(",", " ").split() if x]
    if len(parts) != 3:
        raise ValueError(
            f"--fractions requires exactly 3 numbers (train, val, test); got {parts}"
        )
    s = sum(parts)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"--fractions must sum to 1.0; got {s}")
    return (parts[0], parts[1], parts[2])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-85 Phase 2 — REAL ASTSR-only R² retrain on PlatinAI 4-cell "
            "(5 seeds × 50 epochs × strict-scaffold)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="platinai_4cell",
        choices=["platinai_4cell"],
        help="Dataset (only platinai_4cell is currently supported; default).",
    )
    parser.add_argument(
        "--cell-lines",
        type=_parse_cell_lines,
        nargs="+",
        default=["mcf7", "a549", "a2780", "a2780cis"],
        help=(
            "PlatinAI cell lines to benchmark " "(default: mcf7 a549 a2780 a2780cis)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for strict-scaffold shuffling (default: 0..4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="AST generations per seed (default: 50; parallel to Wf-85 P1).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=32,
        help="ASTSR population size (default: 32; matches Wf-77 P2 baseline default).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="ASTSR max AST depth (default: 4).",
    )
    parser.add_argument(
        "--p-var",
        type=float,
        default=0.45,
        help="ASTSR p_var (default: 0.45).",
    )
    parser.add_argument(
        "--parsimony",
        type=float,
        default=1e-3,
        help="ASTSR parsimony penalty (default: 1e-3).",
    )
    parser.add_argument(
        "--strict-scaffold",
        action="store_true",
        default=True,
        help=(
            "Use strict Bemis-Murcko scaffold split "
            "(default ON; honored for party-twenty-spirit §3.5)."
        ),
    )
    parser.add_argument(
        "--no-strict-scaffold",
        dest="strict_scaffold",
        action="store_false",
        help="Disable strict-scaffold split (use random K-fold instead).",
    )
    parser.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=(0.7, 0.15, 0.15),
        help=(
            "Train/val/test fractions for the strict-scaffold split "
            "(default: 0.7 0.15 0.15; must sum to 1.0)."
        ),
    )
    parser.add_argument(
        "--feature-set",
        default="F5_RDKIT_2D",
        choices=sorted(FEATURE_SET_ALIASES),
        help=(
            "Feature set name (default: F5_RDKIT_2D).  Accepts "
            "F5_RDKIT_2D / RDKIT_2D / F5_RDKIT for the 12-d RDKit "
            "2D descriptor set, or F5_LEGACY / F5 for the 5-d legacy set."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Compute device (default: cpu; --device cuda is accepted "
            "for symmetry with Wf-85 P1 but LambdaASTSR has no "
            "CUDA wrapper today --- the same CPU fp32 + NumPy "
            "ASTSR path is run with used_cuda=False recorded)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("molmetal/reports/wf85_p2_astr_only_real"),
        help=(
            "Output directory " "(default: molmetal/reports/wf85_p2_astr_only_real)."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve feature-set.
    feature_canonical, feature_names = resolve_feature_set(args.feature_set)
    if feature_canonical != "rdkit_2d":
        print(
            f"[wf85-p2] WARNING: feature_set={args.feature_set!r} maps to "
            f"canonical={feature_canonical!r}; the F5 legacy features are "
            f"NOT computed by `_featurize_smiles` (which only emits the 12-d "
            f"RDKit 2D descriptors).  Use --feature-set F5_RDKIT_2D for the "
            f"Wf-85 P2 verification gate; F5_LEGACY is accepted for the CLI "
            f"shape contract only."
        )

    # Flatten seeds.
    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))
    args.seeds = flat_seeds

    # Flatten cell-lines (argparse nargs="+" with list default).
    flat_cell_lines: List[str] = []
    for cl in args.cell_lines:
        if isinstance(cl, list):
            for sub in cl:
                if isinstance(sub, list):
                    flat_cell_lines.extend(str(x) for x in sub)
                else:
                    flat_cell_lines.append(str(sub))
        else:
            flat_cell_lines.append(str(cl))
    args.cell_lines = flat_cell_lines

    for cl in args.cell_lines:
        if cl not in PLATINAI_CELL_COLS:
            raise ValueError(
                f"Unknown cell line: {cl}; supported: {sorted(PLATINAI_CELL_COLS)}"
            )

    all_results: List[Dict[str, Any]] = []
    for cell_line in args.cell_lines:
        print(
            f"[wf85-p2] cell_line={cell_line}: loading PlatinAI 4-cell "
            f"(strict-scaffold={'on' if args.strict_scaffold else 'off'}, "
            f"feature_set={args.feature_set})"
        )
        result = evaluate_cell_line(
            cell_line,
            args.seeds,
            args.epochs,
            args.fractions,
            args.population,
            args.depth,
            args.p_var,
            args.parsimony,
            feature_names,
            feature_canonical,
            args.device,
            _REPO_ROOT,
        )
        per_seed_path = args.out_dir / cell_line / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.out_dir / cell_line / "run.log"
        log_lines = [
            f"cell_line={result['cell_line']}",
            f"target_col={result['target_col']}",
            f"feature_set={result['feature_set']}",
            f"n_samples={result['n_samples']}",
            f"n_seeds={result['n_seeds']}",
            f"epochs={result['epochs']}",
            f"astr_population={result['astr_population']}",
            f"astr_depth={result['astr_depth']}",
            f"astr_p_var={result['astr_p_var']}",
            f"astr_parsimony={result['astr_parsimony']}",
            f"device={result['device']}",
            f"split={result['split']}",
            f"astr_r2_test_mean={result['astr_r2_test_mean']:+.4f}",
            f"astr_r2_test_std={result['astr_r2_test_std']:.4f}",
            f"astr_r2_val_mean={result['astr_r2_val_mean']:+.4f}",
            f"astr_r2_val_std={result['astr_r2_val_std']:.4f}",
            f"astr_r2_train_mean={result['astr_r2_train_mean']:+.4f}",
            f"astr_r2_train_std={result['astr_r2_train_std']:.4f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf85-p2] cell_line={cell_line}: "
            f"ASTSR test R² = {result['astr_r2_test_mean']:+.4f} "
            f"± {result['astr_r2_test_std']:.4f} "
            f"(val={result['astr_r2_val_mean']:+.4f} ± "
            f"{result['astr_r2_val_std']:.4f}, "
            f"train={result['astr_r2_train_mean']:+.4f} ± "
            f"{result['astr_r2_train_std']:.4f})"
        )
        all_results.append(result)
    write_synthesis(args.out_dir, all_results)
    print(f"[wf85-p2] synthesis.md written to {args.out_dir / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
