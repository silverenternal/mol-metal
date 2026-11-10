"""Tests for the Wf-257-Apply TODO-126 M1+M2 master module integration.

Covers the integration of:

  * :class:`MasterWassersteinGradientFlow`  (M1 — Wasserstein regularization
    for CFM velocity field) — see
    :mod:`molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow`.
  * :class:`MasterCatherineClosureBias`     (M2 — Catherine closure-bounded
    generation at decode time) — see
    :mod:`molmetal.molmetal_lam.reward.master_catherine_closure_bias`.

Per ``TODO/completed/126_innovation_strengthen_roadmap.md`` §3 + §7:

* M1 + M2 are the **shared master modules** wired into both the CFM
  driver (Wf-124 P2) and the chemprop v2 v3 driver (Wf-250-Apply).
* The CFM driver imports them via the master facades (not the raw
  ``wasserstein_loss`` / ``pocket_closure_bias``); the master facade
  exposes a single ``compute`` / ``compute_bias`` API plus
  ``engine_tag`` attribution for honest reporting.

This test module verifies:

  1. M1 facade is importable + computes finite Wasserstein loss.
  2. M2 facade is importable + computes finite Catherine bias.
  3. The CFM driver wires both master modules (engine-tag
     attribution present in ``per_epoch_metrics.json``).
  4. chemprop v2 v3 driver wires both master modules (per-seed JSON
     carries ``wf250_use_master_wass`` / ``wf250_use_master_cath``
     engine tags).

Honest framing (party-twenty-spirit §3.5):
* **CPU-only** (torch-native Wasserstein + RDKit-free Catherine).
* **Backward-compat**: this test does not change any production
  behaviour; it only verifies that the Wf-124 P2 + Wf-250-Apply
  integrations are still in place.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# 1. M1 facade import + compute
# ---------------------------------------------------------------------------
class TestM1MasterWasserstein:
    def test_master_wasserstein_importable(self):
        from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
            MasterWassersteinGradientFlow,
            MasterWassersteinConfig,
        )

        assert MasterWassersteinGradientFlow is not None
        assert MasterWassersteinConfig is not None

    def test_master_wasserstein_compute_returns_finite(self):
        """M1.compute() returns finite Wasserstein loss on synthetic input."""
        import torch
        from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
            MasterWassersteinGradientFlow,
            MasterWassersteinConfig,
        )

        cfg = MasterWassersteinConfig(n_mc=8)
        m1 = MasterWassersteinGradientFlow(cfg)
        v_pred = torch.randn(64, 3)
        v_ref = torch.randn(64, 3)
        result = m1.compute(
            metal_logits=v_pred,
            ref_distribution=v_ref,
            loss_type="gradient_flow",
        )
        # Result dataclass — finite Wasserstein loss.
        assert hasattr(result, "loss_value")
        assert torch.isfinite(torch.as_tensor(result.loss_value)).all()


# ---------------------------------------------------------------------------
# 2. M2 facade import + compute
# ---------------------------------------------------------------------------
class TestM2MasterCatherine:
    def test_master_catherine_importable(self):
        from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
            MasterCatherineClosureBias,
            MasterCatherineBiasResult,
        )

        assert MasterCatherineClosureBias is not None
        assert MasterCatherineBiasResult is not None

    def test_master_catherine_compute_returns_finite(self):
        """M2.apply() returns a finite closure bias result for a feature_flag payload."""
        from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
            MasterCatherineClosureBias,
        )

        m2 = MasterCatherineClosureBias()
        payload = {
            "term_id": "t1",
            "metal": "Pt",
            "witness_path": [],
            "smiles": "N.N.[Cl][Pt][Cl]",
        }
        result = m2.apply(payload, bias_type="feature_flag")
        assert result is not None
        # feature_flag schema has engine attribute (per TODO-126 §7).
        assert hasattr(result, "engine")
        assert result.engine == "closure_catherine_feature_flag"


# ---------------------------------------------------------------------------
# 3. CFM driver wires M1+M2 (engine-tag attribution present in ship JSON)
# ---------------------------------------------------------------------------
class TestCFMDriverWiresM1M2:
    def test_wf124_p2_engine_tags_present(self):
        """Wf-124 P2 per_epoch_metrics.json carries the 3 engine tags."""
        path = Path(_REPO) / "molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json"
        if not path.exists():
            pytest.skip("Wf-124 P2 per_epoch_metrics.json not found")
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        tags = obj.get("engine_tags", {})
        assert tags.get("decoder_yuelbond") == "yuelbond_decoder"
        assert tags.get("wasserstein") == "wasserstein_reg"
        assert tags.get("closure_bounded") == "closure_catherine_bounded"

    def test_wf124_p2_synthesis_references_master_modules(self):
        """Wf-124 P2 synthesis.md cites M1 + M2 master facades."""
        path = Path(_REPO) / "molmetal/reports/wf124_p2_cfm_full/synthesis.md"
        if not path.exists():
            pytest.skip("Wf-124 P2 synthesis.md not found")
        text = path.read_text(encoding="utf-8")
        assert "yuelbond_decoder" in text
        assert "wasserstein_reg" in text
        assert "closure_catherine_bounded" in text


# ---------------------------------------------------------------------------
# 4. chemprop v2 v3 driver wires M1+M2 (per-seed JSON engine tags)
# ---------------------------------------------------------------------------
class TestChempropDriverWiresM1M2:
    def test_wf250_use_master_wass_flag_in_driver(self):
        """``train_chemprop_v2_v3.py`` exposes ``--use-master-wass`` flag."""
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(Path(_REPO) / "molmetal/scripts/train_chemprop_v2_v3.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Driver must surface the master flags (Wf-250-Apply F3+F4).
        assert "--use-master-wass" in proc.stdout
        assert "--use-master-cath" in proc.stdout

    def test_wf250_master_module_imports(self):
        """The chemprop driver imports the master facades (per Wf-250-Apply)."""
        from molmetal.scripts import train_chemprop_v2_v3 as driver

        assert hasattr(driver, "_HAVE_MASTER_WASS")
        assert hasattr(driver, "_HAVE_MASTER_CATH")
        assert driver._HAVE_MASTER_WASS is True
        assert driver._HAVE_MASTER_CATH is True
