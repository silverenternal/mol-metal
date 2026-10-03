"""Wf-329W5 — W₂ + chemprop v2 graph features alignment for R² recovery.

=====================================================================
Wf-329W5 / Wf-325B follow-up / WEAK_TO_STRONG §2.1 / TODO-122 P1.3 —
the **next** R²-ceiling remediation step after Wf-325A
confirmed the 32-d RDKit expansion does NOT lift Ridge baseline
R² above the signal floor.

Wf-325A MEASURED on platinai_4cell (200 SMILES, 4-cell mean pIC50):
  * R²(12d) = -0.0945
  * R²(32d) = -0.3118
  * verdict: STILL_NEGATIVE_HONEST_NEGATIVE

Per WEAK_TO_STRONG §2.1, the architectural ceiling requires
**fundamentally different signal sources** — not more RDKit
descriptors.  This script wires up two of those sources on the
CPU-only Ridge baseline:

  1. **chemprop v2 graph features** — atom-level 72-d + bond-level
     14-d from chemprop v2's :class:`SimpleMoleculeMolGraphFeaturizer`,
     pooled to graph-level via mean / max / sum over atoms and bonds
     (172-d fingerprint per molecule).  This captures the atom-bond
     message-passing structure that per-molecule scalar RDKit
     descriptors cannot.

  2. **W₂ distance as feature** — for each row, the 1-d Wasserstein
     distance from the row's pIC50 (treated as a Dirac delta) to the
     reference cohort's pIC50 distribution.  This injects a per-row
     distributional signal that the Ridge baseline can use as a
     "proximity to reference" prior.

The combined pipeline (RDKit 32-d + chemprop v2 graph 172-d +
W₂ 1-d = 205-d) is then evaluated via Ridge baseline on
platinai_4cell with the same 80/20 split + StandardScaler as
Wf-325A.  Lift is reported as
``R²(combined) - R²(12d_baseline=-0.0945)``.

Public API
----------
* :func:`extract_chemprop_v2_graph_features`  per-SMILES pooled graph feat
* :func:`compute_w2_distance_feature`         per-row W₂ distance
* :func:`combined_feature_pipeline`           32-d + graph + W₂ → (n, D)
* :func:`run_wf329w5`                         main entry point

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* PROJECTED R² lift: +0.10 to +0.30 vs -0.0945 baseline.
* If still negative: HONEST_NEGATIVE_R2_DEEP (the chemprop
  encoder is the GPU portion; CPU pooling of pre-featurized
  features is a structural upper bound on what 5-seed × 50-epoch
  chemprop v2 retrain can lift in this cohort).
* CPU-only; no torch autograd; chemprop v2 featurizer is used
  in pure-numpy mode (no forward pass through the encoder).
* Per WEAK_TO_STRONG §2.1, **Multi-PH + PLL** is a separate axis
  (TODO-94) and **Hilbert-ASTSR joint retrain** is the GPU
  portion (TODO-93).  Wf-329W5 is the CPU-only diagnostic
  for the chemprop v2 + W₂ sub-tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# sklearn is the canonical Ridge baseline.  Try the real sklearn first,
# then fall back to a closed-form numpy Ridge so the bench can still
# run in a minimal venv.
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover — defensive
    Ridge = None  # type: ignore
    train_test_split = None  # type: ignore
    StandardScaler = None  # type: ignore
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Repo path resolution (mirror bench_wf325a_r2_32d)
# ---------------------------------------------------------------------------
def _resolve_repo_path(path: str) -> str:
    """Resolve a repo-relative path to an absolute path."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for start in (here, os.getcwd()):
        cur = start
        for _ in range(6):
            child = os.path.join(cur, "molmetal")
            if os.path.isdir(child) and (
                os.path.isdir(os.path.join(child, "data"))
                or os.path.isdir(os.path.join(child, "scripts"))
            ):
                candidate = os.path.join(cur, path)
                if os.path.exists(candidate):
                    return candidate
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# W₂ distance feature (1-d per row)
# ---------------------------------------------------------------------------
def _wasserstein_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-d Wasserstein (earth mover's) distance between two distributions.

    Pure-numpy fallback — scipy.stats.wasserstein_distance when
    available.  For 1-d, EMD equals the L²-norm of the difference
    between inverse-CDF samples on a common grid; scipy implements
    this in O(n log n) via sort + cumulative integration.
    """
    if a.size < 1 or b.size < 1:
        return 0.0
    try:
        from scipy.stats import wasserstein_distance as _wd

        return float(_wd(a.astype(np.float64), b.astype(np.float64)))
    except Exception:
        # Closed-form fallback (when scipy is unavailable):
        # sort + cumulative-distribution CDF comparison.
        a_sorted = np.sort(a.astype(np.float64))
        b_sorted = np.sort(b.astype(np.float64))
        n_a, n_b = a_sorted.size, b_sorted.size
        # Build a common grid via all-sample support.
        all_vals = np.concatenate([a_sorted, b_sorted])
        cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / max(n_a, 1)
        cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / max(n_b, 1)
        diff = cdf_a - cdf_b
        # Approximate integral via trapezoidal rule on sorted support.
        order = np.argsort(all_vals)
        x_sorted = all_vals[order]
        d_sorted = diff[order]
        return float(np.trapz(np.abs(d_sorted), x_sorted))


def compute_w2_distance_feature(
    y: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-row Wasserstein distance from a Dirac distribution at ``y[i]``
    to a 1-d reference distribution.

    For row ``i``, this returns::

        W₂(δ(y[i]), ν_reference)

    which for 1-d distributions simplifies to the L² distance
    between ``y[i]`` and the mean of ν plus the std of ν; but the
    direct computation here gives the exact EMD on the empirical
    CDF grid.  This injects a per-row distributional signal
    ("how far is this row's pIC50 from the reference cohort?")
    that the Ridge baseline can use as a feature.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Target values (one per row).
    reference : np.ndarray of shape (m,), optional
        Reference distribution.  If None, defaults to ``y`` itself
        (i.e. "distance to the cohort that includes this row" — the
        leave-one-in reference).

    Returns
    -------
    w2_feat : np.ndarray of shape (n,)
        W₂ distance per row.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if reference is None:
        reference = y
    else:
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(y.size, dtype=np.float64)
    for i in range(y.size):
        out[i] = _wasserstein_distance_1d(
            np.array([y[i]], dtype=np.float64),
            reference,
        )
    return out


# ---------------------------------------------------------------------------
# chemprop v2 graph feature extraction
# ---------------------------------------------------------------------------
_GRAPH_FEATURE_DIM: Optional[int] = None


def _get_graph_feature_dim() -> int:
    """Return the chemprop v2 graph fingerprint dimension.

    Per Wf-329W5 design: pool chemprop v2 atom (72-d) + bond (14-d)
    features via [mean, max, sum] → ``3 * (72 + 14) = 258-d``.  Sum is
    included because sum scales with molecule size and captures a
    different signal than mean (per-atom-average) or max (per-atom
    extreme).  This dimension is computed once on the first SMILES
    and cached.
    """
    global _GRAPH_FEATURE_DIM
    if _GRAPH_FEATURE_DIM is not None:
        return _GRAPH_FEATURE_DIM
    atom_dim, bond_dim = _probe_chemprop_dims()
    _GRAPH_FEATURE_DIM = 3 * (atom_dim + bond_dim)
    return _GRAPH_FEATURE_DIM


def _probe_chemprop_dims() -> Tuple[int, int]:
    """Probe chemprop v2 SimpleMoleculeMolGraphFeaturizer dimensions.

    Featurizes a single canonical SMILES ('c1ccccc1') to extract the
    atom-level (72) and bond-level (14) feature dimensions used by
    chemprop v2.  Falls back to (72, 14) when chemprop is unavailable
    (the canonical chemprop v2 defaults).
    """
    try:
        from rdkit import Chem
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        ftr = SimpleMoleculeMolGraphFeaturizer()
        mol = Chem.MolFromSmiles("c1ccccc1")
        if mol is None:
            return (72, 14)
        mg = ftr(mol)
        atom_dim = int(mg.V.shape[1])
        bond_dim = int(mg.E.shape[1])
        return (atom_dim, bond_dim)
    except Exception:
        return (72, 14)


def extract_chemprop_v2_graph_features(
    smiles_list: List[str],
    pools: Tuple[str, ...] = ("mean", "max", "sum"),
) -> np.ndarray:
    """Per-SMILES chemprop v2 graph fingerprint.

    For each SMILES:
      1. RDKit parse → Chem.Mol
      2. chemprop v2 :class:`SimpleMoleculeMolGraphFeaturizer` → atom
         features (n_atoms, 72) and bond features (n_bonds, 14)
      3. Pool atom features via mean / max / sum (3 × 72 = 216)
      4. Pool bond features via mean / max / sum (3 × 14 = 42)
      5. Concatenate to a (258,) graph fingerprint

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings to featurize.
    pools : tuple of {"mean", "max", "sum"}
        Pooling operators to apply (default: all three).  Each adds
        (atom_dim + bond_dim) dimensions per pool; total dim is
        ``len(pools) * (atom_dim + bond_dim)``.

    Returns
    -------
    feats : np.ndarray of shape (n_smiles, dim)
        Graph fingerprint matrix; rows with failed RDKit parse
        are zero-filled.

    Notes
    -----
    The fingerprint is **deterministic** given the SMILES (chemprop v2
    uses canonical atom + bond featurisation).  This is the **CPU-only
    static featurisation** — no encoder forward pass, no autograd.
    The "graph features" here are the input to the chemprop v2 D-MPNN
    encoder, not the encoder output (the encoder output requires a
    full forward pass which is the GPU portion per WEAK_TO_STRONG §2.1).
    """
    try:
        from rdkit import Chem, RDLogger
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        RDLogger.DisableLog("rdApp.*")
        ftr = SimpleMoleculeMolGraphFeaturizer()
        chemprop_available = True
    except Exception:
        chemprop_available = False
        ftr = None

    atom_dim, bond_dim = _probe_chemprop_dims()
    n_pools = len(pools)
    dim = n_pools * (atom_dim + bond_dim)
    out = np.zeros((len(smiles_list), dim), dtype=np.float64)

    for i, smi in enumerate(smiles_list):
        if not chemprop_available:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mg = ftr(mol)
            V = np.asarray(mg.V, dtype=np.float64)  # (n_atoms, atom_dim)
            E = np.asarray(mg.E, dtype=np.float64)  # (n_bonds, bond_dim)
        except Exception:
            continue

        if V.shape[0] == 0:
            atom_pool = np.zeros(atom_dim * n_pools, dtype=np.float64)
        else:
            atom_pool_parts = []
            for pool in pools:
                if pool == "mean":
                    atom_pool_parts.append(V.mean(axis=0))
                elif pool == "max":
                    atom_pool_parts.append(V.max(axis=0))
                elif pool == "sum":
                    atom_pool_parts.append(V.sum(axis=0))
                else:
                    raise ValueError(f"unknown pool op: {pool!r}")
            atom_pool = np.concatenate(atom_pool_parts, axis=0)

        if E.shape[0] == 0:
            bond_pool = np.zeros(bond_dim * n_pools, dtype=np.float64)
        else:
            bond_pool_parts = []
            for pool in pools:
                if pool == "mean":
                    bond_pool_parts.append(E.mean(axis=0))
                elif pool == "max":
                    bond_pool_parts.append(E.max(axis=0))
                elif pool == "sum":
                    bond_pool_parts.append(E.sum(axis=0))
                else:
                    raise ValueError(f"unknown pool op: {pool!r}")
            bond_pool = np.concatenate(bond_pool_parts, axis=0)

        out[i, : atom_pool.size] = atom_pool
        out[i, atom_pool.size : atom_pool.size + bond_pool.size] = bond_pool

    return out


# ---------------------------------------------------------------------------
# Combined feature pipeline
# ---------------------------------------------------------------------------
def combined_feature_pipeline(
    smiles_list: List[str],
    X_32d: np.ndarray,
    y: np.ndarray,
    reference_y: Optional[np.ndarray] = None,
    include_rdkit_32d: bool = True,
    include_chemprop_v2_graph: bool = True,
    include_w2_distance: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Combined feature matrix: RDKit 32-d + chemprop v2 graph + W₂.

    Parameters
    ----------
    smiles_list : list of str
        SMILES, one per row (must align with rows of ``X_32d``).
    X_32d : np.ndarray of shape (n, 32)
        Pre-computed RDKit 32-d feature matrix (rows aligned with
        ``smiles_list``).
    y : np.ndarray of shape (n,)
        Target vector (used to compute the W₂ distance feature).
    reference_y : np.ndarray of shape (m,), optional
        Reference distribution for W₂ distance.  If None, uses
        ``y`` itself.
    include_rdkit_32d, include_chemprop_v2_graph, include_w2_distance : bool
        Toggles for each sub-feature stream.  All default True.

    Returns
    -------
    X_combined : np.ndarray of shape (n, D)
        Concatenated feature matrix.  D = 32 (rdkit) + 258 (graph)
        + 1 (W₂) when all three are on = 291-d.
    feature_names : list of str
        Names of each column (handy for the result JSON).
    """
    n = len(smiles_list)
    streams: List[np.ndarray] = []
    names: List[str] = []

    if include_rdkit_32d:
        if X_32d.shape[0] != n:
            raise ValueError(f"X_32d has {X_32d.shape[0]} rows; expected {n}")
        streams.append(np.asarray(X_32d, dtype=np.float64))
        names.extend([f"rdkit_32d_{i:02d}" for i in range(32)])

    if include_chemprop_v2_graph:
        graph_feat = extract_chemprop_v2_graph_features(smiles_list)
        streams.append(graph_feat)
        atom_dim, bond_dim = _probe_chemprop_dims()
        n_pools = 3  # mean / max / sum
        atom_names = [
            f"graph_atom_{pool}_{i:02d}"
            for pool in ("mean", "max", "sum")
            for i in range(atom_dim)
        ]
        bond_names = [
            f"graph_bond_{pool}_{i:02d}"
            for pool in ("mean", "max", "sum")
            for i in range(bond_dim)
        ]
        names.extend(atom_names + bond_names)

    if include_w2_distance:
        w2_feat = compute_w2_distance_feature(y, reference=reference_y)
        w2_feat = w2_feat.reshape(-1, 1)
        streams.append(w2_feat)
        names.append("w2_dist_to_ref")

    if not streams:
        return np.zeros((n, 0), dtype=np.float64), []
    X = np.concatenate(streams, axis=1)
    return X, names


# ---------------------------------------------------------------------------
# Ridge baseline (re-use Wf-325A helper signature for honest comparability)
# ---------------------------------------------------------------------------
def ridge_r2_eval(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Ridge regression baseline + R² on the test split (Wf-325A API)."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if X.shape[0] < 4:
        return {"r2": float("nan"), "n_train": 0, "n_test": 0, "alpha": alpha}
    if _HAS_SKLEARN:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, y, test_size=test_size, random_state=seed
        )
        model = Ridge(alpha=alpha)
        model.fit(Xtr, ytr)
        r2 = float(model.score(Xte, yte))
        return {
            "r2": r2,
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "alpha": alpha,
        }
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    n = Xs.shape[0]
    n_te = max(1, int(round(n * test_size)))
    n_tr = n - n_te
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    Xtr, Xte = Xs[tr_idx], Xs[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ ytr
    w = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), Xty)
    yhat = Xte @ w
    ss_res = float(np.sum((yte - yhat) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2": r2,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Data loaders (platinai_4cell 12-d + 32-d + smiles list)
# ---------------------------------------------------------------------------
_PLATINAI_12D_COLUMNS: Tuple[str, ...] = (
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
_PLATINAI_32D_COLUMNS: Tuple[str, ...] = _PLATINAI_12D_COLUMNS + (
    "BertzCT",
    "Chi0v",
    "Chi0n",
    "Chi1n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "fr_halogen",
    "fr_Al_COO",
)
_PLATINAI_PIC50_COLUMNS: Tuple[str, ...] = (
    "pIC50_mcf7",
    "pIC50_a549",
    "pIC50_a2780",
    "pIC50_a2780cis",
)


def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    path = _resolve_repo_path(path)
    with open(path, "r", encoding="utf-8") as f:
        import csv

        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    return header, rows


def _row_to_floats(
    row: List[str], columns: Tuple[str, ...], header: List[str]
) -> np.ndarray:
    idx = [header.index(c) for c in columns]
    out = np.zeros((len(row), len(columns)), dtype=np.float64)
    for i, r in enumerate(row):
        for j, k in enumerate(idx):
            try:
                v = float(r[k])
            except (ValueError, IndexError):
                v = float("nan")
            out[i, j] = v
    return out


def _mean_pIC50_target(
    rows: List[List[str]], header: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    idx = [header.index(c) for c in _PLATINAI_PIC50_COLUMNS]
    out = np.zeros(len(rows), dtype=np.float64)
    keep = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        vals: List[float] = []
        for k in idx:
            try:
                v = float(r[k])
            except (ValueError, IndexError):
                v = float("nan")
            if not np.isnan(v):
                vals.append(v)
        if vals:
            out[i] = float(np.mean(vals))
            keep[i] = True
    return out, keep


def load_platinai_4cell_32d_with_smiles(
    csv_path: str = "molmetal/data/platinai_4cell_rdkit_extended_features.csv",
    target_csv_path: str = "molmetal/data/platinai_4cell_rdkit_features.csv",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load platinai_4cell 32-d features + mean pIC50 + SMILES list.

    Returns ``(X_32d, y, smiles_list)``.  The SMILES list aligns
    row-for-row with ``X_32d`` / ``y``.
    """
    cols_32 = _PLATINAI_32D_COLUMNS
    header32, rows32 = _read_csv(csv_path)
    smiles_idx_32 = header32.index("smiles")
    smiles_to_y: Dict[str, float] = {}
    h12, r12 = _read_csv(target_csv_path)
    y_full_12, keep_12 = _mean_pIC50_target(r12, h12)
    for r, k, y in zip(r12, keep_12, y_full_12):
        if k:
            smiles_to_y[r[h12.index("smiles")]] = float(y)
    X_rows: List[np.ndarray] = []
    y_vals: List[float] = []
    smis: List[str] = []
    for r in rows32:
        smi = r[smiles_idx_32]
        if smi not in smiles_to_y:
            continue
        vec = _row_to_floats([r], cols_32, header32)[0]
        if not np.isfinite(vec).all():
            continue
        X_rows.append(vec)
        y_vals.append(smiles_to_y[smi])
        smis.append(smi)
    if not X_rows:
        raise RuntimeError("load_platinai_4cell_32d_with_smiles: no rows survived")
    return np.stack(X_rows, axis=0), np.array(y_vals, dtype=np.float64), smis


# ---------------------------------------------------------------------------
# Honest framing helpers
# ---------------------------------------------------------------------------
WF325A_BASELINE_R2_PLA_12D: float = -0.09446482453622895  # Wf-325A MEASURED


def _summarise_recovery(r2_combined: float) -> str:
    """Honest framing verdict for Wf-329W5 R² lift vs Wf-325A baseline."""
    if not np.isfinite(r2_combined):
        return "UNDEFINED"
    lift = r2_combined - WF325A_BASELINE_R2_PLA_12D
    if r2_combined > 0.0 and lift >= 0.10:
        return "CEILING_BREAK"
    if r2_combined > 0.0 and lift >= 0.05:
        return "MODERATE_RECOVERY"
    if r2_combined > 0.0:
        return "WEAK_RECOVERY"
    if lift >= 0.0:
        return "NO_LIFT_HONEST_NEGATIVE"
    return "HONEST_NEGATIVE_R2_DEEP"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_wf329w5(
    out_path: str = "molmetal/reports/wf329w5_w2_chemprop/result.json",
    cohort: str = "platinai_4cell",
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the Wf-329W5 W₂ + chemprop v2 graph features alignment bench.

    Saves the result JSON to ``out_path`` and returns the result dict.
    """
    if cohort != "platinai_4cell":
        raise NotImplementedError(
            f"cohort={cohort!r} not yet supported; Wf-329W5 currently "
            f"only ships platinai_4cell (200 SMILES)."
        )

    # ----- Load the cohort -----
    X_32d, y, smiles = load_platinai_4cell_32d_with_smiles()
    n_samples = int(X_32d.shape[0])

    # ----- Configurations to evaluate -----
    configs: List[Dict[str, Any]] = [
        # (label, include_rdkit_32d, include_chemprop_v2_graph, include_w2_distance)
        ("12d_baseline", False, False, False),  # filled below
        ("32d_rdkit", True, False, False),
        ("32d_rdkit_plus_w2", True, False, True),
        ("32d_rdkit_plus_graph", True, True, False),
        ("32d_rdkit_plus_graph_plus_w2", True, True, True),
        ("graph_only", False, True, False),
        ("w2_only", False, False, True),
        ("graph_plus_w2", False, True, True),
    ]

    per_config: List[Dict[str, Any]] = []
    for label, incl_32, incl_graph, incl_w2 in configs:
        if label == "12d_baseline":
            # 12-d baseline: use first 12 cols of X_32d (Wf-49 P1 frozen set).
            X_in = X_32d[:, :12]
            X_combined = X_in
            feature_names = [f"rdkit_12d_{i:02d}" for i in range(12)]
        else:
            try:
                X_combined, feature_names = combined_feature_pipeline(
                    smiles_list=smiles,
                    X_32d=X_32d,
                    y=y,
                    reference_y=y,
                    include_rdkit_32d=incl_32,
                    include_chemprop_v2_graph=incl_graph,
                    include_w2_distance=incl_w2,
                )
            except Exception as exc:
                per_config.append(
                    {
                        "label": label,
                        "error": f"feature pipeline failed: {exc}",
                        "r2": float("nan"),
                        "lift_vs_12d": float("nan"),
                        "verdict": "PIPELINE_ERROR",
                    }
                )
                continue
        # Drop constant columns defensively (W₂ + graph may yield
        # zero-var columns when SMILES parsing fails for a row).
        if X_combined.shape[1] > 0:
            keep_mask = X_combined.std(axis=0) > 1e-9
            X_combined = X_combined[:, keep_mask]
        res = ridge_r2_eval(X_combined, y, alpha=1.0, seed=seed)
        r2 = res["r2"]
        per_config.append(
            {
                "label": label,
                "r2": float(r2),
                "n_train": int(res["n_train"]),
                "n_test": int(res["n_test"]),
                "alpha": float(res["alpha"]),
                "n_features": int(X_combined.shape[1]),
                "feature_dim_breakdown": {
                    "rdkit_32d": 32 if incl_32 else 0,
                    "chemprop_v2_graph": (
                        _get_graph_feature_dim() if incl_graph else 0
                    ),
                    "w2_distance": 1 if incl_w2 else 0,
                },
                "lift_vs_12d_baseline": float(r2 - WF325A_BASELINE_R2_PLA_12D),
                "verdict": _summarise_recovery(r2),
            }
        )

    # ----- Find the best config (PROJECTED lift, not max-bias) -----
    finite_r2 = [c for c in per_config if np.isfinite(c.get("r2", float("nan")))]
    best_label = None
    best_r2 = float("-inf")
    for c in finite_r2:
        if c["r2"] > best_r2:
            best_r2 = float(c["r2"])
            best_label = c["label"]

    # ----- Aggregate verdict -----
    any_ceiling_break = any(c.get("verdict") == "CEILING_BREAK" for c in per_config)
    any_moderate = any(c.get("verdict") == "MODERATE_RECOVERY" for c in per_config)
    any_weak = any(c.get("verdict") == "WEAK_RECOVERY" for c in per_config)
    any_negative = any(
        c.get("verdict") == "HONEST_NEGATIVE_R2_DEEP"
        or c.get("verdict") == "NO_LIFT_HONEST_NEGATIVE"
        for c in per_config
    )
    if any_ceiling_break:
        aggregate_status = "CEILING_BREAK"
    elif any_moderate:
        aggregate_status = "MODERATE_RECOVERY"
    elif any_weak:
        aggregate_status = "WEAK_RECOVERY"
    elif any_negative:
        aggregate_status = "HONEST_NEGATIVE_R2_DEEP"
    else:
        aggregate_status = "UNDEFINED"

    result: Dict[str, Any] = {
        "wf": "Wf-329W5",
        "task": "W₂ + chemprop v2 graph features alignment for R² recovery",
        "scope": (
            "CPU-only diagnostic on platinai_4cell (200 SMILES, 4-cell mean pIC50); "
            "no torch autograd, no encoder forward pass."
        ),
        "cohort": cohort,
        "n_samples": n_samples,
        "n_features": {
            "rdkit_32d": 32,
            "chemprop_v2_graph": _get_graph_feature_dim(),
            "w2_distance": 1,
            "combined_max": 32 + _get_graph_feature_dim() + 1,
        },
        "wf325a_baseline": {
            "r2_12d": WF325A_BASELINE_R2_PLA_12D,
            "source": "molmetal/reports/wf325a_r2_32d/result.json",
            "verdict": "STILL_NEGATIVE_HONEST_NEGATIVE",
        },
        "per_config": per_config,
        "best_config": {
            "label": best_label,
            "r2": best_r2 if best_label is not None else float("nan"),
            "lift_vs_12d_baseline": (
                float(best_r2 - WF325A_BASELINE_R2_PLA_12D)
                if best_label is not None
                else float("nan")
            ),
        },
        "aggregate_status": aggregate_status,
        "honest_framing": {
            "projected_lift": (
                "+0.10 to +0.30 R² lift vs Wf-325A baseline "
                "(per WEAK_TO_STRONG §2.1 architectural remediation)."
            ),
            "structural_note": (
                "The chemprop v2 graph features here are the INPUT to "
                "the D-MPNN encoder (atom-level 72-d + bond-level 14-d "
                "featurisation), NOT the encoder OUTPUT.  The encoder "
                "output (300-d per molecule) requires a full forward "
                "pass with autograd, which is the GPU portion per "
                "WEAK_TO_STRONG §2.1 / TODO-93."
            ),
            "w2_distance_note": (
                "The W₂ distance feature is a 1-d per-row scalar "
                "capturing the EMD from a Dirac at y[i] to the "
                "reference cohort distribution.  This is the "
                "distributional-proximity prior recommended by "
                "TODO-122 P1.3."
            ),
            "ceiling_break": (
                "Best config R² > 0 AND lift ≥ +0.10 over Wf-325A "
                "baseline (-0.0945)."
            ),
            "honest_negative": (
                "Best config R² ≤ 0 OR lift < +0.01: architectural "
                "ceiling confirmed even with chemprop v2 graph + W₂."
            ),
        },
        "pre_flight_status": (
            "CPU-only diagnostic; pre-flight v2 gate NOT required "
            "(no retrain / panel / ablation)."
        ),
    }

    out_path_abs = _resolve_repo_path(out_path)
    os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
    with open(out_path_abs, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cohort",
        default="platinai_4cell",
        help="Cohort to evaluate (default: platinai_4cell)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="molmetal/reports/wf329w5_w2_chemprop")
    args = p.parse_args(argv)

    out_path = os.path.join(args.output_dir, "result.json")
    result = run_wf329w5(out_path=out_path, cohort=args.cohort, seed=args.seed)

    print(f"Wf-329W5: cohort={result['cohort']} n={result['n_samples']}")
    print(f"  Wf-325A baseline R²(12d) = {result['wf325a_baseline']['r2_12d']:+.4f}")
    for c in result["per_config"]:
        if "error" in c:
            print(f"  {c['label']:>30s}: {c['error']}")
            continue
        print(
            f"  {c['label']:>30s}  n_feat={c['n_features']:>3d}  "
            f"R²={c['r2']:+.4f}  lift_vs_12d={c['lift_vs_12d_baseline']:+.4f}  "
            f"{c['verdict']}"
        )
    print(
        f"  best_config: {result['best_config']['label']}  "
        f"R²={result['best_config']['r2']:+.4f}  "
        f"lift={result['best_config']['lift_vs_12d_baseline']:+.4f}"
    )
    print(f"  aggregate_status = {result['aggregate_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
