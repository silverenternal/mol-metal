"""Data layer for MolFlow-Triton.

This package exposes the molecular datasets and 3D geometry transforms
used by training and evaluation pipelines:

* :class:`BaseMoleculeDataset` - abstract base class (subclasses
  :class:`torch.utils.data.Dataset`) that defines the contract shared
  by every concrete dataset.
* :class:`MetalCytoToxDataset` - tabular QSAR toxicity dataset loaded
  from ``data/MetalCytoToxDB.csv``.
* :class:`QM9Dataset` - QM9 small-molecule 3D dataset
  (~134k molecules, 12 regression targets; falls back to a
  ~500-molecule RDKit synthetic set if the download/parse fails).
* :func:`data.transforms.random_rotation_3d` - uniform random SO(3)
  rotation applied to ``(N, 3)`` coordinate tensors.
* :func:`data.transforms.random_translation_3d` - random translation
  in ``[-max, max]`` per axis.
* :class:`data.transforms.Compose` - torchvision-style composition
  helper that also threads ``coords`` through :class:`MoleculeSample`
  containers.

The module is intentionally framework-light: it owns no optimizer, no
training loop, no collate function.  See :mod:`scripts` for training.
"""

from __future__ import annotations

# Pull canonical names from the focused modules.  Importing
# ``.mol_dataset`` here too would be redundant, but we leave it as the
# back-compat shim entry documented below; ``from data.mol_dataset
# import ...`` keeps working for the rest of the repo.
from ._base import BaseMoleculeDataset, MoleculeSample
from .metalcytotox import MetalCytoToxDataset
from .qm9 import QM9Dataset
from .transforms import Compose, random_rotation_3d, random_translation_3d

__all__ = [
    # datasets
    "MoleculeSample",
    "BaseMoleculeDataset",
    "MetalCytoToxDataset",
    "QM9Dataset",
    # transforms
    "random_rotation_3d",
    "random_translation_3d",
    "Compose",
]

__version__ = "0.1.0"