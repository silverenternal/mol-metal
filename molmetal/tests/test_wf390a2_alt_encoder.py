"""Tests for Wf-390-A2 alt encoder audit + smoke scaffold.

Per TODO/pending/179_wf123_p3_w2_pic50_alignment.md §7 alternative encoder
bootstrap + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL verdict:

  - test_audit_alt_encoders_present: verifies chemprop v3 + D-MPNN + F5_RDKIT_2D
    modules are importable + alternative encoder inventory is complete
  - test_smoke_alt_encoder_cpu_only: verifies CPU-mode smoke produces plausible
    n_features_in >= 300 vs 12-d RDKit 2D baseline + USER_OK gate enforced

Honest framing (party-twenty-spirit §3.5): all assertions reference the actual
import chain. No fabricated module availability.

Tested modules:
  - molmetal.models.chemprop_v2_mt_model_v3 (chemprop v3 multi-task model)
  - molmetal.models.dmpnn (D-MPNN base model)
  - molmetal.data.rdkit_2d_extended (F5_RDKIT_2D 32-d)
  - molmetal.molmetal_lam.reward.wasserstein_loss (W₂ alignment)

Run:
    pytest molmetal/tests/test_wf390a2_alt_encoder.py -v
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest


def _audit_alt_encoders():
    """Lazy import of audit_alt_encoders to avoid top-level import failures."""
    from molmetal.scripts.wf390a2_alt_encoder_audit import audit_alt_encoders

    return audit_alt_encoders()


def _smoke_alt_encoder(**kwargs: Any) -> Dict[str, Any]:
    """Lazy import of smoke_alt_encoder to avoid top-level import failures."""
    from molmetal.scripts.wf390a2_alt_encoder_audit import smoke_alt_encoder

    return smoke_alt_encoder(**kwargs)


class TestAuditAltEncodersPresent:
    """Test that audit_alt_encoders returns a complete inventory."""

    def test_audit_returns_alt_encoder_inventory(self):
        """Verify audit_alt_encoders returns an AltEncoderInventory with all fields."""
        from molmetal.scripts.wf390a2_alt_encoder_audit import (
            AltEncoderInventory,
        )

        inv = _audit_alt_encoders()
        assert isinstance(inv, AltEncoderInventory)
        # All 8 module verdicts should be PRESENT (the SHIPPED modules)
        assert inv.chemprop_v3_mt_model == "PRESENT"
        assert inv.chemprop_v3_v3_class == "PRESENT"
        assert inv.chemprop_v2_v3_quick_smoke == "PRESENT"
        assert inv.dmpnn_module == "PRESENT"
        assert inv.dmpnn_legacy_class == "DirectedMPNN"
        assert inv.dmpnn_pyg_class == "DirectedMPNNPyG"
        assert inv.rdkit_2d_extended == "PRESENT"
        assert inv.rdkit_2d_extended_32d == "PRESENT"
        assert inv.wasserstein_pic50_loss == "PRESENT"
        assert inv.wasserstein_1d_torchsort == "PRESENT"
        # Trainer constraint discovery
        assert "12-d default" in inv.trainer_metals_constraint or (
            "IMPORT_FAIL" in inv.trainer_metals_constraint
        )
        # Notes must be non-empty
        assert len(inv.notes) > 0

    def test_audit_serializes_to_json(self, tmp_path: Path):
        """Verify AltEncoderInventory can be serialized to JSON for the reports dir."""
        inv = _audit_alt_encoders()
        out_path = tmp_path / "alt_encoder_inventory.json"
        out_path.write_text(json.dumps(inv.__dict__, indent=2, sort_keys=True))
        loaded = json.loads(out_path.read_text())
        assert loaded["chemprop_v3_mt_model"] == "PRESENT"
        assert loaded["dmpnn_module"] == "PRESENT"
        assert loaded["rdkit_2d_extended"] == "PRESENT"


class TestSmokeAltEncoderCpuOnly:
    """Test smoke_alt_encoder produces a SCAFFOLD_PASS result on CPU."""

    def test_smoke_cpu_returns_scaffold_pass(self):
        """Verify CPU-mode smoke returns SCAFFOLD_PASS without GPU launch."""
        result = _smoke_alt_encoder(n_samples=32, n_epochs=5, device="cpu")
        assert result["status"] == "SCAFFOLD_PASS"
        assert result["device"] == "cpu"
        assert result["user_ok_required"] == 0
        # chemprop v3 + DirectedMPNN + F5_RDKIT_2D = 300 + 300 + 32 = 632-d
        assert result["n_features_in_chemprop_v3"] == 300
        assert result["n_features_in_dmpnn"] == 300
        assert result["n_features_in_dmpnn_pyg"] == 300
        assert result["n_features_in_cross_encoder_ensemble"] == 632
        # Honest framing disclaimer must be present
        assert "SCAFFOLD-ONLY" in result["honest_framing"]

    def test_smoke_cuda_refused_without_user_ok(self, monkeypatch):
        """Verify CUDA-mode smoke REFUSED when USER_OK != 1."""
        monkeypatch.delenv("USER_OK", raising=False)
        result = _smoke_alt_encoder(n_samples=32, n_epochs=5, device="cuda")
        assert result["status"] == "REFUSED"
        assert "USER_OK" in result["reason"]

    def test_smoke_cuda_allowed_with_user_ok(self, monkeypatch):
        """Verify CUDA-mode smoke SCAFFOLD_PASS when USER_OK=1."""
        monkeypatch.setenv("USER_OK", "1")
        result = _smoke_alt_encoder(n_samples=32, n_epochs=5, device="cuda")
        assert result["status"] == "SCAFFOLD_PASS"
        assert result["device"] == "cuda"
        assert result["user_ok_required"] == 1


class TestAlternativeEncoderModuleAvailability:
    """Test that the underlying encoder modules are actually importable."""

    def test_chemprop_v3_mt_model_v3_importable(self):
        """Verify ChempropV2MTModelV3 + quick_smoke import successfully."""
        mod = importlib.import_module("molmetal.models.chemprop_v2_mt_model_v3")
        assert hasattr(mod, "ChempropV2MTModelV3")
        assert hasattr(mod, "quick_smoke_chemprop_v2_mt_v3")

    def test_dmpnn_importable(self):
        """Verify DirectedMPNN + DirectedMPNNPyG classes import successfully."""
        legacy_mod = importlib.import_module("molmetal.models.dmpnn")
        pyg_mod = importlib.import_module("molmetal.models.dmpnn_pyg")
        assert hasattr(legacy_mod, "DirectedMPNN")
        assert hasattr(pyg_mod, "DirectedMPNNPyG")
        # Honest framing: the legacy class is named DirectedMPNN, NOT DMPNN
        assert not hasattr(legacy_mod, "DMPNN")

    def test_rdkit_2d_extended_importable(self):
        """Verify F5_RDKIT_2D module imports with 32-d + 12-d columns."""
        mod = importlib.import_module("molmetal.data.rdkit_2d_extended")
        assert mod.RDKIT_2D_DIM == 32
        assert mod.RDKIT_2D_BASE_DIM == 12
        assert len(mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS) == 32
        assert len(mod.RDKIT_2D_FEATURE_COLUMNS) == 12

    def test_wasserstein_pic50_loss_importable(self):
        """Verify wasserstein_pic50_loss + wasserstein_1d_torchsort import."""
        mod = importlib.import_module("molmetal.molmetal_lam.reward.wasserstein_loss")
        assert hasattr(mod, "wasserstein_pic50_loss")
        assert hasattr(mod, "wasserstein_1d_torchsort")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
