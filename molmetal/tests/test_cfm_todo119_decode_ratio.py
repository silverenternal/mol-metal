"""Tests for the Wf-257-Apply TODO-119 decode_ratio measurement.

Covers the ``molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond``
``evaluate_decode_ratio`` smoke path used by the Wf-124 P2 ship:

* ``evaluate_decode_ratio`` is exposed and callable on synthetic
  Pt-click clouds.
* The smoke path is **non-mutating** (does not change model
  parameters) per the test_evaluate_decode_ratio_does_not_mutate_model_params
  test in ``test_cfm_yuelbond_wasserstein.py``.

This test module is **CPU-only**, runs in <2s, and serves as a
**smoke gate** for the pre-retrain checklist §3.1 (per TODO-119 P1.3
acceptance: ``decode_ratio`` is measurable on synthetic Pt-click
clouds at runtime).

Honest framing (party-twenty-spirit §3.5):
* Synthetic decode_ratio = 1.000 (Wf-124 P2 ship) is the **upper
  bound**; real-CFM decode_ratio 0.9980 / 0.9958 (Wf-181 / Wf-188) is
  the MEASURED value on MetalCytoToxDB.  This test asserts the
  synthetic upper bound path remains green so Wf-119 P1.3 has a
  known-good smoke gate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture(scope="module")
def cfm_module():
    import torch
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
        build_synthetic_pt_click_clouds,
        NUM_BOND_CLASSES,
    )
    return {
        "torch": torch,
        "YuelBondBondDecoder": YuelBondBondDecoder,
        "build_synthetic_pt_click_clouds": build_synthetic_pt_click_clouds,
        "NUM_BOND_CLASSES": NUM_BOND_CLASSES,
    }


class TestDecodeRatioMeasurement:
    def test_decode_ratio_evaluator_is_exposed(self, cfm_module):
        """evaluate_decode_ratio must be importable from the CFM script."""
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            evaluate_decode_ratio,
        )
        assert callable(evaluate_decode_ratio)

    def test_synthetic_clouds_build(self, cfm_module):
        """The synthetic Pt-click cohort builder is functional."""
        build = cfm_module["build_synthetic_pt_click_clouds"]
        clouds = build(n_clouds=8, jitter=0.20, seed=0)
        assert len(clouds) == 8
        # Each cloud is 5 atoms: 1 Pt + 2 Cl + 2 N (cisplatin layout).
        assert clouds[0].positions.shape == (5, 3)
        assert clouds[0].atomic_numbers.shape == (5,)

    def test_decode_ratio_with_initial_decoder_is_smoke(self, cfm_module):
        """End-to-end smoke: fresh decoder + 8 clouds → int count returned."""
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            _decode_with_rdkit_strict,
            CFMJointModel,
            YuelBondBondDecoder,
        )
        torch = cfm_module["torch"]
        build = cfm_module["build_synthetic_pt_click_clouds"]
        clouds = build(n_clouds=8, jitter=0.20, seed=0)
        device = torch.device("cpu")
        decoder = YuelBondBondDecoder()
        model = CFMJointModel(hidden_dim=8, n_layers=1).to(device)
        # _decode_with_rdkit_strict returns bool (single-cloud).
        ok = _decode_with_rdkit_strict(decoder, clouds[0])
        assert isinstance(ok, bool)
        # Use the full evaluate_decode_ratio path with a real model.
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            evaluate_decode_ratio,
        )
        n_ok, n_total = evaluate_decode_ratio(
            model=model,
            decoder=decoder,
            val_clouds=clouds,
            device=device,
            closure_bounded=False,
        )
        assert n_total == 8
        assert 0 <= n_ok <= 8


class TestDecodeRatioBackwardCompat:
    def test_metric_file_is_well_formed(self):
        """The ``decode_ratio.json`` metric file is parseable JSON."""
        import json
        # Try both possible locations (root + nested).
        for path_str in (
            "metrics/by_metric/decode_ratio.json",
            "molmetal/metrics/by_metric/decode_ratio.json",
        ):
            path = Path(_REPO) / path_str
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                assert "metric" in obj
                assert "measurements" in obj
                return
        pytest.skip("decode_ratio.json not found in either location")

    def test_target_decode_ratio_field_present(self, cfm_module):
        """Wf-124 P2 ship target_decode_ratio field is present."""
        import json
        report_path = (
            Path(_REPO)
            / "molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json"
        )
        if not report_path.exists():
            pytest.skip("Wf-124 P2 per_epoch_metrics.json not found")
        with open(report_path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        assert "best" in obj
        assert "decode_ratio" in obj["best"]
        # The ship target was >= 0.8; the MEASURED best was 1.0.
        assert obj["best"]["decode_ratio"] >= 0.8
        assert obj["best"].get("hit_target") is True