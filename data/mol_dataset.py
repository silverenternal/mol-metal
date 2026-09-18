"""Backwards-compatible re-export shim for the MolFlow-Triton datasets.

The dataset implementations previously lived in this single module.  They
have since been split into focused files:

* :mod:`data._base` - :class:`MoleculeSample` and
  :class:`BaseMoleculeDataset`.
* :mod:`data.metalcytotox` - :class:`MetalCytoToxDataset`.
* :mod:`data.qm9` - :class:`QM9Dataset` (full ~134k implementation,
  not a placeholder).

This module re-exports all four public names so that
``from data.mol_dataset import ...`` keeps working for the existing
``scripts/``, ``utils/`` and test code without modification.
"""

from __future__ import annotations

from ._base import BaseMoleculeDataset, MoleculeSample
from .metalcytotox import MetalCytoToxDataset
from .qm9 import QM9Dataset

__all__ = [
    "MoleculeSample",
    "BaseMoleculeDataset",
    "MetalCytoToxDataset",
    "QM9Dataset",
]