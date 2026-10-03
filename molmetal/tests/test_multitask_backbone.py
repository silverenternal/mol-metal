"""Tests for Wf-128 Phase 1 — multi-task 5-metal backbone.

================================================================
Pins the Wf-128 Phase 1 contract for the shared 5-metal multi-task
regression backbone.  Ten tests across four categories:

1. **Architecture manifest** — frozen config dataclasses + JSON
   schema (input_dim=172, trunk_hidden=256, trunk_latent=128,
   n_metals=5, metals=(Ru, Ir, Os, Rh, Re)).
2. **Forward pass shapes** — per-metal dict output, finite values,
   ``forward_all_metals`` shape (B, n_metals).
3. **Backbone + arch JSON round-trip** — the manifest written by
   ``write_arch_json`` re-loads to the same dict.
4. **Sanity check** — CPU wall-time measured, parameter count
   consistent with the manifest, sanity JSON has the expected keys.

All tests run on CPU in < 60 s.  No GPU dependencies required.

References
----------
* TODO/pending/95_r20_p4_3way_100pocket.md
* TODO/pending/122_platinai_4cell_r2_strengthen.md
* TODO/completed/91_wf41_3way_integration.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRIPTS = _REPO_ROOT / "molmetal/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import build_multitask_backbone as m  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_no_torch():
    if not m._HAS_TORCH:
        pytest.skip("Torch not available — backbone tests skipped")


# ---------------------------------------------------------------------------
# 1. Architecture manifest (the Wf-128 spec)
# ---------------------------------------------------------------------------
def test_default_config_input_dim_is_172() -> None:
    """Default config wires INPUT_DIM = 172 (matches the feature stack)."""
    cfg = m.MultiTaskBackboneConfig()
    assert cfg.input_dim == 172


def test_default_config_trunk_dims() -> None:
    """Default trunk is 172 → 256 → 128 (Wf-128 §1 spec)."""
    cfg = m.MultiTaskBackboneConfig()
    assert cfg.trunk_hidden == 256
    assert cfg.trunk_latent == 128


def test_default_config_per_metal_head() -> None:
    """Default head is 128 → 64 → 1 with dropout 0.2 + ReLU."""
    cfg = m.MultiTaskBackboneConfig()
    head = cfg.head
    assert head.in_dim == 128
    assert head.hidden_dim == 64
    assert head.out_dim == 1
    assert head.dropout == 0.2
    assert head.activation == "relu"


def test_default_config_metals_subset() -> None:
    """The 5-metal subset is Ru / Ir / Os / Rh / Re."""
    cfg = m.MultiTaskBackboneConfig()
    assert cfg.metals == ("Ru", "Ir", "Os", "Rh", "Re")
    assert cfg.n_metals == 5
    assert len(cfg.metals) == cfg.n_metals


def test_per_metal_head_config_validates_act() -> None:
    """``PerMetalHead`` raises on unknown activation at construction time."""
    _skip_if_no_torch()
    cfg_bad = m.PerMetalHeadConfig(activation="bogus")
    with pytest.raises(ValueError, match="unknown activation"):
        m.PerMetalHead(cfg_bad)


# ---------------------------------------------------------------------------
# 2. Forward pass shapes
# ---------------------------------------------------------------------------
def test_backbone_forward_returns_per_metal_dict() -> None:
    """``forward`` returns a dict keyed by metal name with ``(B, 1)`` outputs."""
    _skip_if_no_torch()
    import torch

    cfg = m.MultiTaskBackboneConfig()
    module = m.MultiTaskBackbone(cfg)
    module.eval()

    b = 8
    x = torch.randn(b, cfg.input_dim)
    mid = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        out = module(x, mid)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(cfg.metals)
    for metal in cfg.metals:
        assert out[metal].shape == (
            b,
            1,
        ), f"{metal} bad shape {tuple(out[metal].shape)}"


def test_backbone_forward_outputs_are_finite() -> None:
    """All per-metal outputs are finite on a random batch."""
    _skip_if_no_torch()
    import torch

    cfg = m.MultiTaskBackboneConfig()
    module = m.MultiTaskBackbone(cfg)
    module.eval()
    x = torch.randn(16, cfg.input_dim)
    mid = torch.zeros(16, dtype=torch.long)  # all Ru
    with torch.no_grad():
        out = module(x, mid)
    for metal, t in out.items():
        assert torch.isfinite(t).all().item(), f"{metal} has non-finite entries"


def test_backbone_forward_all_metals_shape() -> None:
    """``forward_all_metals`` returns ``(B, n_metals)`` (one per metal)."""
    _skip_if_no_torch()
    import torch

    cfg = m.MultiTaskBackboneConfig()
    module = m.MultiTaskBackbone(cfg)
    module.eval()
    x = torch.randn(4, cfg.input_dim)
    with torch.no_grad():
        all_m = module.forward_all_metals(x)
    assert all_m.shape == (4, cfg.n_metals)


def test_backbone_forward_validates_input_shape() -> None:
    """Wrong input dim raises ``ValueError`` (not silent)."""
    _skip_if_no_torch()
    import torch

    cfg = m.MultiTaskBackboneConfig()
    module = m.MultiTaskBackbone(cfg)
    x = torch.randn(2, cfg.input_dim - 1)  # wrong
    mid = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="input_dim"):
        module(x, mid)


# ---------------------------------------------------------------------------
# 3. Arch JSON round-trip
# ---------------------------------------------------------------------------
def test_write_arch_json_roundtrip(tmp_path: Path) -> None:
    """``write_arch_json`` produces a JSON that re-loads to the same dict."""
    cfg = m.MultiTaskBackboneConfig()
    out_path = tmp_path / "arch.json"
    m.write_arch_json(cfg, out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["input_dim"] == cfg.input_dim
    assert loaded["trunk_hidden"] == cfg.trunk_hidden
    assert loaded["trunk_latent"] == cfg.trunk_latent
    assert loaded["head"]["in_dim"] == cfg.head.in_dim
    assert loaded["head"]["hidden_dim"] == cfg.head.hidden_dim
    assert loaded["head"]["out_dim"] == cfg.head.out_dim
    assert loaded["metals"] == list(cfg.metals)
    assert loaded["n_metals"] == cfg.n_metals
    assert loaded["version"] == cfg.version


def test_write_arch_json_parent_dirs_created(tmp_path: Path) -> None:
    """``write_arch_json`` creates parent directories (no manual mkdir)."""
    cfg = m.MultiTaskBackboneConfig()
    out_path = tmp_path / "deep" / "nested" / "arch.json"
    m.write_arch_json(cfg, out_path)
    assert out_path.exists()


# ---------------------------------------------------------------------------
# 4. Sanity check
# ---------------------------------------------------------------------------
def test_run_sanity_records_param_count() -> None:
    """``run_sanity`` reports the total parameter count."""
    _skip_if_no_torch()
    cfg = m.MultiTaskBackboneConfig()
    sanity = m.run_sanity(cfg, batch_size=8, n_iters=5, device="cpu")
    assert sanity["torch_available"] is True
    assert sanity["param_count"] is not None
    # 172→256 + 256→128 trunk + 5 × (128→64 + 64→1) heads
    # = 44032 + 32768 + 5 × (8192 + 64) = 76768 + 41280 = 118 048-ish.
    assert sanity["param_count"] > 100_000
    assert sanity["param_count"] < 200_000


def test_run_sanity_records_per_metal_shapes_and_cpu_wall() -> None:
    """``run_sanity`` writes per-metal output shapes + CPU wall-time."""
    _skip_if_no_torch()
    cfg = m.MultiTaskBackboneConfig()
    sanity = m.run_sanity(cfg, batch_size=4, n_iters=3, device="cpu")
    assert "per_metal_output_shape_cpu" in sanity
    for metal in cfg.metals:
        assert sanity["per_metal_output_shape_cpu"][metal] == (4, 1)
    wall = sanity["cpu_wall"]
    assert wall["n_iters"] == 3
    assert wall["device"] == "cpu"
    assert wall["mean_ms"] > 0.0


def test_run_sanity_finite_cpu_flag() -> None:
    """``run_sanity`` flags per-metal finiteness on the CPU pass."""
    _skip_if_no_torch()
    cfg = m.MultiTaskBackboneConfig()
    sanity = m.run_sanity(cfg, batch_size=4, n_iters=3, device="cpu")
    assert sanity["per_metal_finite_cpu"] == {m_: True for m_ in cfg.metals}


def test_run_sanity_no_torch_returns_friendly_payload() -> None:
    """Without torch, ``run_sanity`` still returns a valid payload."""
    # We *do* have torch on CI, so we patch the module flag for this test.
    saved, m._HAS_TORCH = m._HAS_TORCH, False  # type: ignore[misc]
    try:
        cfg = m.MultiTaskBackboneConfig()
        sanity = m.run_sanity(cfg, batch_size=4, n_iters=1, device="cpu")
        assert sanity["torch_available"] is False
        assert sanity["param_count"] is None
        assert "torch_error" in sanity
    finally:
        m._HAS_TORCH = saved  # type: ignore[misc]
