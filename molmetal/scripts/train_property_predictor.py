"""Stub for training the EGNN property predictor.

**Not implemented yet.**  This file exists so the import path referenced
in :class:`molmetal.adapters.egnn_predictor.EGNNPropertyPredictor` is
resolvable.

Plan (to be filled in by a later milestone):

1. Load MetalCytoToxDB -> (ligand_smiles, pIC50_or_cell_growth) pairs.
2. For each complex in CrossDocked2020 or BindingMOAD, pair the
   co-crystallized ligand SMILES with the protein pocket and use the
   reported binding affinity as the regression target.
3. Featurise molecules + pockets into a joint graph.
4. Train an EGNN message-passing network with a regression head,
   minimise MSE on pIC50 (neg-log-of-Ki or neg-log-of-IC50).
5. Validate on a held-out split and dump the best checkpoint to
   ``molmetal/checkpoints/egnn_pic50_v0.pt``.

Usage (once implemented)::

    source .venv/bin/activate
    python -m molmetal.scripts.train_property_predictor \
        --csv /mnt/storage/data/molmetal/MetalCytoToxDB.csv \
        --epochs 50 \
        --batch-size 32 \
        --output molmetal/checkpoints/egnn_pic50_v0.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "[STUB] Train EGNN property predictor for pIC50 — not yet "
            "implemented.  See module docstring for the plan."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/mnt/storage/data/molmetal/MetalCytoToxDB.csv",
        help="Training data CSV path (currently unused).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=str,
        default="molmetal/checkpoints/egnn_pic50_v0.pt",
        help="Where to save the trained checkpoint.",
    )
    args = parser.parse_args()

    raise NotImplementedError(
        "train_property_predictor is a stub. See module docstring for "
        "the planned implementation (MetalCytoToxDB + CrossDocked2020 "
        "featurisation + EGNN regression head)."
    )


if __name__ == "__main__":
    main()
