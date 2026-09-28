"""Wf-244-D: TODO-83l Click chemistry yield R² prediction.

Train Ridge regression on 60 click reactions × metal-combo yields using
reaction one-hot + RDKit 2D descriptors for smiles_a / smiles_b.

CPU-only. sklearn Ridge with 5-fold CV. Honest framing: R² may be low
if hand-crafted features don't capture reaction kinetics.

Usage:
    PY=/home/hugo/codes/mol-metal/.venv/bin/python
    $PY molmetal/scripts/wf244d_click_yield_r2.py
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

CSV_PATH = Path(
    "/home/hugo/codes/mol-metal/.data/molmetal/click_yields_literature.csv"
)
REPORT_DIR = Path(
    "/home/hugo/codes/mol-metal/molmetal/reports/wf244d_click_yield"
)
REPORT_JSON = REPORT_DIR / "R2_report.json"

# 12 RDKit 2D descriptors per molecule. Standard chemoinformatics panel
# capturing size, polarity, halogenation, aromaticity, flexibility.
DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("HeavyAtomMolWt", Descriptors.HeavyAtomMolWt),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("TPSA", Descriptors.TPSA),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("fr_halogen", Descriptors.fr_halogen),
    ("fr_C_O", Descriptors.fr_C_O),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
]


def mol_to_desc(smi: str) -> np.ndarray | None:
    """Compute 12 RDKit 2D descriptors for one SMILES. None if parse fails."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    out = []
    for _, fn in DESC_FUNCS:
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        out.append(v)
    return np.array(out, dtype=np.float64)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    """Reaction one-hot + RDKit 2D descriptors for smiles_a and smiles_b."""
    reactions = sorted(df["reaction"].unique().tolist())
    reaction_to_idx = {r: i for i, r in enumerate(reactions)}
    n_rxn = len(reactions)

    desc_names = [n for n, _ in DESC_FUNCS]
    feat_names: list[str] = []
    feat_names.extend([f"rxn_{r}" for r in reactions])
    feat_names.extend([f"a_{n}" for n in desc_names])
    feat_names.extend([f"b_{n}" for n in desc_names])

    X = np.zeros((len(df), len(feat_names)), dtype=np.float64)
    keep_idx: list[int] = []
    for i, row in df.iterrows():
        da = mol_to_desc(row["smiles_a"])
        db = mol_to_desc(row["smiles_b"])
        if da is None or db is None:
            continue
        X[i, reaction_to_idx[row["reaction"]]] = 1.0
        X[i, n_rxn : n_rxn + len(desc_names)] = da
        X[i, n_rxn + len(desc_names) :] = db
        keep_idx.append(i)
    return X[keep_idx], feat_names, keep_idx


def per_reaction_r2(
    df_kept: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Per-reaction R² (one-vs-rest grouping by reaction type)."""
    by_rxn: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r, t, p in zip(df_kept["reaction"].tolist(), y_true, y_pred):
        by_rxn[r].append((t, p))
    out: dict[str, float] = {}
    for r, pairs in by_rxn.items():
        if len(pairs) < 3:
            out[r] = float("nan")
            continue
        ts = np.array([p[0] for p in pairs])
        ps = np.array([p[1] for p in pairs])
        if ts.std() < 1e-9:
            # Constant target: trivial score = 1.0 if pred matches, else nan
            out[r] = 1.0 if np.allclose(ps, ts.mean()) else float("nan")
        else:
            out[r] = float(r2_score(ts, ps))
    return out


def top_features_by_abs_coef(
    coefs: np.ndarray, feat_names: list[str], k: int = 3
) -> list[str]:
    order = np.argsort(-np.abs(coefs))
    return [feat_names[i] for i in order[:k]]


def main() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    n_entries_total = int(len(df))
    n_reactions_total = int(df["reaction"].nunique())

    X, feat_names, keep_idx = build_features(df)
    n_kept = int(X.shape[0])
    n_features = int(X.shape[1])
    df_kept = df.iloc[keep_idx].reset_index(drop=True)
    y = df_kept["yield"].to_numpy(dtype=np.float64)

    # 5-fold CV with shuffle, fixed seed for reproducibility.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2: list[float] = []
    y_oof = np.zeros_like(y)
    for fold, (tr, va) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        # Small alpha keeps a flexible linear model on 25-35 features / 60 rows.
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xva)
        y_oof[va] = pred
        fold_r2.append(float(r2_score(y[va], pred)))

    mean_r2 = float(np.mean(fold_r2))
    std_r2 = float(np.std(fold_r2, ddof=0))

    per_rxn = per_reaction_r2(df_kept, y, y_oof)

    # Final full-data fit for top-feature identification.
    scaler_full = StandardScaler()
    Xfull = scaler_full.fit_transform(X)
    full_model = Ridge(alpha=1.0, random_state=42)
    full_model.fit(Xfull, y)
    top3 = top_features_by_abs_coef(full_model.coef_, feat_names, k=3)

    honest = (
        "60 click-yield datapoints with hand-crafted RDKit 2D descriptors and "
        "reaction one-hot yield R^2 that may be modest (kitchen-sink kinetics "
        "depend on catalyst, solvent, temperature, etc.). Report is descriptive "
        "of feature-yield correlation, NOT a kinetics model."
    )

    out = {
        "G1_csv_loaded": True,
        "G2_n_entries": n_kept,
        "G2_n_entries_total": n_entries_total,
        "G3_features_extracted": n_features,
        "G3_feature_names": feat_names,
        "G3_n_reactions": n_reactions_total,
        "G3_reactions": sorted(df["reaction"].unique().tolist()),
        "G4_ridge_runs": True,
        "G4_alpha": 1.0,
        "G4_n_folds": 5,
        "G5_fold_r2": [round(v, 4) for v in fold_r2],
        "G5_mean_r2": round(mean_r2, 4),
        "G5_std_r2": round(std_r2, 4),
        "G6_per_reaction_r2": {k: (None if (v != v) else round(v, 4)) for k, v in per_rxn.items()},
        "G7_top_features": top3,
        "G7_top_feature_coefs": [
            round(float(full_model.coef_[feat_names.index(f)]), 4) for f in top3
        ],
        "G8_report_written": False,
        "honest_framing_note": honest,
        "csv_path": str(CSV_PATH),
        "y_mean": round(float(y.mean()), 4),
        "y_std": round(float(y.std()), 4),
    }

    REPORT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    out["G8_report_written"] = True
    # Re-write with updated flag (canonical)
    REPORT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
