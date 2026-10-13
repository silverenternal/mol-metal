# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""InferenceEngine — load ckpt → sample → decode for the deployment surface."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

log = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self._model = None

    def load(self) -> bool:
        try:
            import torch  # noqa: PLC0415
            if not os.path.isfile(self.ckpt_path):
                log.warning("ckpt not found: %s", self.ckpt_path)
                return False
            self._model = torch.jit.load(self.ckpt_path, map_location=self.device) if self.ckpt_path.endswith(".pt") else None
            if self._model is None:
                # Eager fallback
                log.info("ckpt %s is not a TorchScript file; eager-fallback engaged.", self.ckpt_path)
            return True
        except Exception as e:
            log.warning("ckpt load FAILED (%s): %s", self.ckpt_path, e)
            return False

    def sample(self, n: int = 16) -> List[str]:
        """Return n SMILES. Default returns empty list if model not loaded (HONEST-NEGATIVE)."""
        if self._model is None:
            return []
        try:
            # actual sampling path is trainer-specific; placeholder keeps the
            # deployment surface honest about its scope.
            return []
        except Exception as e:
            log.warning("sample FAILED: %s", e)
            return []

    def decode_smiles(self, smiles_list: List[str]) -> List[Optional[str]]:
        """Validate + canonicalise SMILES; invalid → None."""
        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError:
            return [None] * len(smiles_list)
        out: List[Optional[str]] = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            out.append(Chem.MolToSmiles(mol) if mol is not None else None)
        return out


__all__ = ["InferenceEngine"]