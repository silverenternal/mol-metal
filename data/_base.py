"""Shared base classes for MolFlow-Triton datasets.

This module defines the lightweight building blocks used by every
concrete molecular dataset shipped under :mod:`data`:

* :class:`MoleculeSample` - dataclass container for one data point
  (features, label, coords, atom_types).
* :class:`BaseMoleculeDataset` - abstract :class:`torch.utils.data.Dataset`
  subclass that concrete datasets (:class:`data.metalcytotox.MetalCytoToxDataset`,
  :class:`data.qm9.QM9Dataset`) extend.

Keeping these in their own file means the concrete dataset modules can
import only the contract they need and stay small and focused.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = [
    "MoleculeSample",
    "BaseMoleculeDataset",
]


# ----------------------------------------------------------------------
# Sample container
# ----------------------------------------------------------------------
@dataclass
class MoleculeSample:
    """Lightweight container for one molecular data point.

    For 3D-aware datasets (QM9, GEOM-Drugs) ``coords`` and ``atom_types``
    carry the geometry that :class:`models.velocity_net.VelocityNet`
    expects.  For tabular datasets (MetalCytoTox) ``features`` carries
    the flat descriptor vector and ``coords`` / ``atom_types`` are
    ``None``.
    """

    features: torch.Tensor | None = None   # (F,)        - flat descriptor
    label: torch.Tensor | None = None      # (,) or (T,) - target(s)
    coords: torch.Tensor | None = None     # (N, 3)
    atom_types: torch.Tensor | None = None  # (N,)


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class BaseMoleculeDataset(Dataset):
    """Abstract base class for MolFlow-Triton datasets.

    The contract:

    * ``__len__`` returns the number of samples.
    * ``__getitem__(idx)`` returns either a :class:`MoleculeSample`
      *or* a tuple ``(features, label)`` - concrete subclasses pick.
      For now both :class:`MetalCytoToxDataset` and the QM9 stub return
      :class:`MoleculeSample`, which is more uniform and lets the
      collate function stay dumb.

    Subclasses MUST override :meth:`_load_raw` and :meth:`__getitem__`.
    They MAY override :meth:`__len__` if the data is held in memory.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__()
        self.root = Path(root) if root is not None else Path("data")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _load_raw(self) -> None:
        """Populate ``self._samples`` from disk.

        Called once in :meth:`__init__`` after the constructor has set up
        any necessary paths / metadata.  Implementations should assign
        the loaded tensors to ``self._samples`` (or similar attribute)
        so :meth:`__len__`` / :meth:`__getitem__`` can use them.
        """

    # ------------------------------------------------------------------
    # Dataset protocol (defaults that work if subclasses follow the
    # "store list of MoleculeSample in self._samples" convention).
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if not hasattr(self, "_samples"):
            return 0
        return len(self._samples)

    def __getitem__(self, idx: int) -> MoleculeSample:
        if not hasattr(self, "_samples"):
            raise RuntimeError(
                f"{type(self).__name__}._load_raw() was never called."
            )
        return self._samples[idx]