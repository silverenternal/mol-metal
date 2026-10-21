#!/usr/bin/env python3
"""UP-9 — is a third, structurally different base learner additive on the DOI holdout?

`msrs_hybrid.json` already blends the tree and the GNN with fold-wise stacking, but
those two see the same ligand representation.  The one diversity axis never tried
is a **non-parametric** learner: Tanimoto kNN over Morgan fingerprints.  It cannot
beat the trees on its own, but a stack can still profit from it because its errors
are not the trees' errors.

Arms (all on the DOI holdout, fold assignment read from the OOF store so nothing
is re-split):

    ligand        the existing tree arm, ligand block only
    ligand_both   the existing tree arm, ligand + metal + cell
    knn           Tanimoto kNN over Morgan FP, k=25, fold-wise
    stack2        ridge on [ligand, ligand_both], fitted on the other four folds
    stack3        ridge on [ligand, ligand_both, knn]

Gate (frozen before the run, 2026-10-21):

    PASS  if stack3 DOI R2 >= 0.42
    FAIL  otherwise -> the ensemble is not additive at the level the headline
         needs, and the single best arm stays the citable number.

A second, softer clause from the plan — "no sub-metric below its best single arm
by more than 0.01" — is reported as ``stack3_vs_best_single``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

from msrs_surface_model import score  # noqa: E402

OOF = REPO / "results" / "msrs_generalization" / "oof.npz"
OUT = REPO / "metrics" / "by_metric" / "msrs_final_stack.json"
K_NEIGHBOURS = 25
GATE = {"doi_r2": 0.42, "single_arm_tolerance": 0.01, "frozen_at": "2026-10-21T15:10+08:00"}


def morgan(smiles: List[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def knn_predict(X: np.ndarray, y: np.ndarray, fold: np.ndarray, k: int) -> np.ndarray:
    """Tanimoto kNN, fold-wise: neighbours only from the other folds."""
    pred = np.full(len(y), np.nan)
    Xb = X.astype(bool)
    nbits = Xb.sum(axis=1)
    for f in np.unique(fold):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        if len(tr) == 0 or len(te) == 0:
            continue
        inter = Xb[te] @ Xb[tr].T.astype(np.int32)
        denom = nbits[te][:, None] + nbits[tr][None, :] - inter
        denom[denom == 0] = 1
        sim = inter / denom
        kk = min(k, sim.shape[1])
        idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(te))[:, None]
        w = sim[rows, idx]
        w_sum = w.sum(axis=1)
        safe = np.where(w_sum == 0, 1.0, w_sum)
        pred[te] = (w * y[tr][idx]).sum(axis=1) / safe
        # all-zero similarity: fall back to the training mean
        pred[te] = np.where(w_sum == 0, y[tr].mean(), pred[te])
    return pred


def stack(preds: List[np.ndarray], y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Fold-wise ridge on the other four folds; intercept unpenalised."""
    A = np.column_stack([p for p in preds] + [np.ones(len(y))])
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        tr = fold != f
        te = fold == f
        At, Ae = A[tr], A[te]
        reg = np.eye(At.shape[1]) * 1.0
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(At.T @ At + reg, At.T @ y[tr])
        out[te] = Ae @ coef
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path, default=OOF)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    z = np.load(args.oof, allow_pickle=True)
    y = z["y"].astype(float)
    fold = z["doi__fold"].astype(int)
    smiles = z["ligand_canonical"]
    print(f"[UP-9] rows={len(y)} folds={len(np.unique(fold))}")

    preds = {"ligand": z["doi__ligand"].astype(float),
             "ligand_both": z["doi__ligand+both"].astype(float)}

    t0 = time.time()
    X = morgan(list(smiles))
    print(f"[UP-9] morgan in {time.time() - t0:.1f}s  density={X.mean():.4f}")
    t0 = time.time()
    preds["knn"] = knn_predict(X, y, fold, K_NEIGHBOURS)
    print(f"[UP-9] knn in {time.time() - t0:.1f}s")

    results: Dict[str, object] = {name: score(y, p) for name, p in preds.items()}
    for name, entry in results.items():
        print(f"[UP-9] {name:12s} R2={entry['r2']:+.4f} MAE={entry['mae']:.4f}")

    results["stack2"] = score(y, stack([preds["ligand"], preds["ligand_both"]], y, fold))
    results["stack3"] = score(y, stack([preds["ligand"], preds["ligand_both"], preds["knn"]], y, fold))
    for name in ("stack2", "stack3"):
        print(f"[UP-9] {name:12s} R2={results[name]['r2']:+.4f} MAE={results[name]['mae']:.4f}")

    best_single = max(("ligand", "ligand_both", "knn"), key=lambda a: results[a]["r2"])
    best_single_r2 = results[best_single]["r2"]
    stack3_r2 = results["stack3"]["r2"]
    delta = round(stack3_r2 - best_single_r2, 4)
    verdict = "STACK_ADDS" if stack3_r2 >= GATE["doi_r2"] else "STACK_NOT_ADDITIVE"

    doc = {
        "artifact": "msrs_final_stack",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 ensemble / UP-9",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "oof": str(args.oof.relative_to(REPO)),
            "script": str(Path(__file__).relative_to(REPO)),
            "k_neighbours": K_NEIGHBOURS,
        },
        "design": {
            "split": "doi (fold assignment taken from the OOF store, not re-split)",
            "arms": {
                "ligand": "existing tree OOF, ligand block",
                "ligand_both": "existing tree OOF, ligand + metal + cell",
                "knn": f"Tanimoto kNN over Morgan r=2 2048 bits, k={K_NEIGHBOURS}",
                "stack2": "fold-wise ridge on [ligand, ligand_both]",
                "stack3": "fold-wise ridge on [ligand, ligand_both, knn]",
            },
            "gate": GATE,
        },
        "results": results,
        "best_single": best_single,
        "best_single_r2": best_single_r2,
        "stack3_vs_best_single": delta,
        "verdict": verdict,
        "verdict_rule": "PASS iff stack3 DOI R2 >= 0.42",
        "caveats": [
            "The DOI split here has tree OOF only; there is no GNN OOF on this split, so this "
            "is not the same blend as msrs_hybrid.json (which is fragment/canonical).",
            "The kNN arm is fitted on the same rows it predicts within a fold, using only the "
            "other four folds' labels; the fold assignment is DOI-grouped, so no paper "
            "contributes to its own prediction.",
            "stack2/stack3 are fitted on the other four folds, which is standard stacking but "
            "still mildly optimistic relative to a fully nested protocol.",
        ],
    }
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[UP-9] best_single={best_single} {best_single_r2:+.4f}  "
          f"stack3={stack3_r2:+.4f}  delta={delta:+.4f} -> {verdict}")
    print(f"json: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
