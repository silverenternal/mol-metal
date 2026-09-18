"""pic50_predictor — graph-neural pIC50 predictor for click-chemistry ligands.

================================================================
What this replaces
================================================================
The original ``molmetal/molmetal_lam/scripts/baselines.py`` used a tiny
sklearn MLP on Morgan fingerprints fitted on 100 cytotox labels — a
predictor with effectively zero generalisation power.  This module wraps
a properly trained Attentive D-MPNN regressor (see
``molmetal/checkpoints/dmpnn_atn_ru_pic50.pt``) that produces sensible
pIC50 values for *any* drug-like SMILES.

================================================================
Public API
================================================================
* :class:`AttentiveDMPNNPredictor`  -- load checkpoint + predict pIC50.
* :func:`predict_pic50(smiles)`     -- drop-in replacement for the
                                      previous sklearn MLP wrapper.

CLI
---
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -c "from molmetal.molmetal_lam.sbdd_env.pic50_predictor import predict_pic50; print(predict_pic50('O=C1CN(CC1)C(=O)c1ccc(cc1)Cl'))"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Path bootstrap so this file can be imported when the package is used as
# ``molmetal.molmetal_lam.sbdd_env.pic50_predictor`` from anywhere.
# Layout: pic50_predictor.py lives at
# PROJECT_ROOT/molmetal/molmetal_lam/sbdd_env/pic50_predictor.py
# so PROJECT_ROOT = 3 hops up from __file__.
_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import numpy as np
import torch
from rdkit import Chem, RDLogger

from molmetal.baselines.dmpnn_attentive import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    AttentiveDMPNNModel,
    featurize_smiles_list,
)

RDLogger.DisableLog("rdApp.*")

DEFAULT_CKPT = os.path.join(
    _PKG_PARENT, "molmetal", "checkpoints", "dmpnn_attn_ru_pic50.pt"
)

# Lazy singleton — created on first call, reused thereafter.
_PREDICTOR: Optional["AttentiveDMPNNPredictor"] = None


class AttentiveDMPNNPredictor:
    """Wrap a trained ``AttentiveDMPNNModel`` checkpoint as a pIC50 oracle.

    Parameters
    ----------
    ckpt : str
        Path to a torch checkpoint produced by
        ``molmetal.molmetal_lam.scripts.calibrate_pic50_predictor``.
    device : str | torch.device
        Inference device (default: best available).
    """

    def __init__(
        self,
        ckpt: str | os.PathLike = DEFAULT_CKPT,
        device: Optional[str] = None,
    ) -> None:
        self.ckpt = str(ckpt)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not os.path.exists(self.ckpt):
            raise FileNotFoundError(
                f"pIC50 checkpoint not found at {self.ckpt}. "
                f"Run `python -m molmetal.molmetal_lam.scripts.calibrate_pic50_predictor` "
                f"to train and save it."
            )

        bundle = torch.load(self.ckpt, map_location=self.device, weights_only=False)
        if isinstance(bundle, dict) and "state_dict" in bundle:
            self.model_kwargs = dict(bundle.get("model_kwargs", {}))
            self.meta = dict(bundle.get("meta", {}))
            self.test_metrics = dict(bundle.get("test_metrics", {}))
            state_dict = bundle["state_dict"]
        else:
            # Plain state_dict fallback.
            state_dict = bundle
            self.model_kwargs = {}
            self.meta = {}
            self.test_metrics = {}

        # Merge sensible defaults so a plain state_dict still loads.
        kw = {
            "atom_dim": ATOM_FEATURE_DIM,
            "bond_dim": BOND_FEATURE_DIM,
            "hidden": 128,
            "depth": 3,
            "dropout": 0.0,
        }
        kw.update(self.model_kwargs)

        self.model = AttentiveDMPNNModel(
            atom_dim=kw["atom_dim"],
            bond_dim=kw["bond_dim"],
            hidden=int(kw["hidden"]),
            depth=int(kw["depth"]),
            dropout=float(kw["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict_pic50(self, smiles: str) -> float:
        """Predict pIC50 for a single SMILES. Returns ``float("nan")`` on parse fail."""
        out = self.predict_pic50_batch([smiles])
        if not out:
            return float("nan")
        return float(out[0])

    @torch.no_grad()
    def predict_pic50_batch(
        self, smiles_list: List[str], batch_size: int = 16
    ) -> List[float]:
        """Predict pIC50 for a list of SMILES, returning one float per input."""
        if not smiles_list:
            return []
        graphs = featurize_smiles_list([str(s) for s in smiles_list])
        out: List[float] = []
        from molmetal.baselines.dmpnn_attentive import collate_graphs

        for start in range(0, len(graphs), batch_size):
            batch = graphs[start:start + batch_size]
            valid = [g for g in batch if g is not None]
            preds: List[float] = []
            if valid:
                batch_data = collate_graphs(valid)
                if batch_data is not None:
                    atom_f, bond_f, e_src, e_dst = batch_data
                    atom_f = atom_f.to(self.device)
                    bond_f = bond_f.to(self.device)
                    e_src = e_src.to(self.device)
                    e_dst = e_dst.to(self.device)
                    logits = self.model(atom_f, bond_f, e_src, e_dst)
                    preds = [float(x) for x in logits.cpu().numpy().tolist()]
            vi = 0
            for g in batch:
                if g is None:
                    out.append(float("nan"))
                else:
                    out.append(preds[vi])
                    vi += 1
        return out


def get_predictor(ckpt: Optional[str] = None) -> AttentiveDMPNNPredictor:
    """Return the cached singleton predictor (load on first call)."""
    global _PREDICTOR
    if _PREDICTOR is None or (ckpt is not None and str(ckpt) != _PREDICTOR.ckpt):
        _PREDICTOR = AttentiveDMPNNPredictor(ckpt=ckpt or DEFAULT_CKPT)
    return _PREDICTOR


def predict_pic50(smiles: str) -> float:
    """Module-level convenience wrapper — drop-in for the sklearn MLP API.

    Returns the predicted pIC50 as a float (typically in [3, 9] for
    drug-like molecules), or ``float("nan")`` if the SMILES cannot be
    parsed or the checkpoint is unavailable.
    """
    ckpt_path = DEFAULT_CKPT
    if not os.path.exists(ckpt_path):
        return float("nan")
    try:
        return get_predictor(ckpt=ckpt_path).predict_pic50(smiles)
    except Exception:
        return float("nan")


__all__ = [
    "AttentiveDMPNNPredictor",
    "get_predictor",
    "predict_pic50",
    "DEFAULT_CKPT",
]


if __name__ == "__main__":
    print(
        "Use `python -m molmetal.molmetal_lam.sbdd_env.pic50_predictor` "
        "or `predict_pic50(smiles)` from your own script."
    )
