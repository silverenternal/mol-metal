"""Wf-A50 §6.55g — Exercise the log-normal ODE time-grid wire-in end-to-end.

The Wf-A35-2 wire-in lives at
``molmetal/adapters/flow_matching_lipman/__init__.py:2915`` and the
nucleus-sampling wire-in at line 3168. Both are DEFAULT-pass-through
under the canonical CFM 5-metal config (``fm_t_dist="uniform"``,
``top_p=1.0``, ``top_k=0``, ``temperature=1.0``), so a green pytest
run on the shipped config alone is no MEASURED proof that the
wire-ins reach the adapter.

This test builds a real :class:`GenerationConfig` with the new
log-normal + nucleus-sampling hyperparameters, calls
:meth:`LipmanFlowMatchingAdapter._generate_impl` once, and asserts:

1. ``last_sampling_diagnostics["nucleus_sample_invocations"] > 0``
   (the opt-in audit counter incremented — the wire-in fired).
2. ``nucleus_sample`` itself was called (monkeypatched stub observed
   the call) — independent path-of-truth on the wire-in branch.
3. ``t_grid != torch.linspace(0, 1, n_steps + 1)`` (the legacy
   uniform grid was replaced; log-normal schedule is in effect).

Per AGENTS.md §5.7 this is a 50-step CPU smoke pattern: it runs on
CPU with ``hidden_dim=32, n_layers=2``, no GPU required, and
exercises only the paths the YAML is supposed to exercise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — build a tiny adapter + a fixed pocket so the call is real
# ---------------------------------------------------------------------------
def _tiny_adapter():
    """A real ``LipmanFlowMatchingAdapter`` on CPU (test runs in <2s)."""
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=32, n_layers=2, lr=5e-3)
    adapter.setup(device="cpu")
    return adapter


def _fixed_pocket(n_atoms: int = 8):
    """A deterministic ``Pocket`` for shape-stable sampling."""
    from molmetal.domain import Pocket

    g = torch.Generator().manual_seed(7)
    pcoords = torch.randn(n_atoms * 2, 3, generator=g) * 2.0
    return Pocket(
        pdb_id="test_wfA50_log_normal",
        coords=pcoords,
        atom_types=torch.randint(1, 18, (n_atoms * 2,), generator=g),
        residue_ids=torch.zeros(n_atoms * 2, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms * 2, dtype=torch.long),
        mask=torch.ones(n_atoms * 2, dtype=torch.bool),
        center=pcoords.mean(0),
        radius=6.0,
    )


def _no_op_solver(x_init, step_size, method, time_grid):
    """Return ``x_init`` unchanged — keeps the test under 2s and lets the
    downstream path see real tensors."""
    return x_init


# ---------------------------------------------------------------------------
# Test 1: log-normal time grid is selected when fm_t_dist != "uniform"
# ---------------------------------------------------------------------------
def test_log_normal_time_grid_replaces_uniform():
    """When ``fm_t_dist="log_normal"``, the legacy ``torch.linspace`` path
    is bypassed and ``log_normal_t_sampling`` produces a grid that is
    NOT bit-exactly equal to ``torch.linspace(0, 1, n_steps + 1)``.

    The check is structural (not a sample-wise distribution test):
    the new wire-in takes effect iff the time-grid branch entered the
    ``_fm_t_dist == "log_normal"`` arm.  A bit-exact-equal grid would
    imply the legacy path was taken silently — the very regression
    this test exists to catch.
    """
    from molmetal.ports import GenerationConfig

    adapter = _tiny_adapter()
    pocket = _fixed_pocket()

    cfg = GenerationConfig(
        n_samples=2,
        n_steps=8,
        seed=42,
        # Non-default values exercise the wire-in branches.
        fm_t_dist="log_normal",
        fm_log_normal_mu=-0.5,
        fm_log_normal_sigma=1.0,
    )
    # fixed_atom_types so the fixed-shape branch is taken (no shape drift).
    cfg.conditioning["fixed_atom_types"] = torch.randint(1, 10, (2, 8))

    # Replace the internal ODE solver with a no-op that records the
    # time grid it received — the test asserts the grid is NOT the
    # legacy uniform grid.
    captured: dict = {}

    class _CaptureODESolver:
        def __init__(self, velocity_model):
            self.velocity_model = velocity_model

        def sample(self, x_init, step_size, method, time_grid):
            captured["time_grid"] = time_grid.detach().clone()
            return x_init

    adapter._ODESolver = _CaptureODESolver

    with patch.object(adapter, "_bond_decoder", None, create=True):
        try:
            adapter.generate(pocket, cfg)
        except Exception:
            # Downstream bond-head / decoder may fail on the toy adapter
            # (random weights, mocked solver returns x_init).  We only
            # care that the time-grid branch fired before solver.sample.
            pass

    assert "time_grid" in captured, (
        "solver.sample was not called — the ODE integration path "
        "short-circuited before fm_t_dist was read"
    )
    grid = captured["time_grid"]
    legacy = torch.linspace(0.0, 1.0, cfg.n_steps + 1, device=grid.device)
    assert not torch.equal(grid, legacy), (
        f"time_grid is bit-exactly equal to torch.linspace(0,1,{cfg.n_steps + 1}) — "
        "the log_normal branch did NOT replace the legacy uniform grid"
    )
    # All grid points must lie in (0, 1) — sigmoid(N(mu, sigma)) support.
    assert (grid > 0).all() and (grid < 1).all(), (
        f"time_grid has values outside (0, 1) — log_normal_sampling "
        f"support violated: min={grid.min().item()}, max={grid.max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 2: nucleus-sample wire-in fires when top_p < 1.0
# ---------------------------------------------------------------------------
def test_nucleus_sample_branch_fires_when_top_p_below_one():
    """When ``top_p=0.85`` (non-default), the nucleus-sample wire-in
    fires and the audit counter ``_nucleus_sample_invocation_count``
    increments.  The default path (``top_p=1.0``) keeps the legacy
    ``torch.multinomial`` path bit-exact, so a green run with the
    default top_p proves nothing about the wire-in.

    MEASURED on the Wf-A35-2 ship surface: at the default
    (``top_p=1.0``) the invocation count stays at 0; the wire-in
    is a no-op.  This test asserts the non-default path increments.
    """
    from molmetal.ports import GenerationConfig

    adapter = _tiny_adapter()
    pocket = _fixed_pocket()

    cfg = GenerationConfig(
        n_samples=2,
        n_steps=4,  # fewer steps so the test stays under 5s on CPU
        seed=42,
        # ALL three nucleus flags set to non-default so the wire-in
        # branch is unambiguously entered (the code activates on the
        # OR: top_p < 1.0 OR top_k > 0 OR temperature != 1.0).
        top_p=0.85,
        top_k=30,
        temperature=0.7,
        # Uniform time grid (default) — this test is about nucleus-sample,
        # not the ODE time grid, so we keep the legacy path here.
        fm_t_dist="uniform",
    )
    cfg.conditioning["fixed_atom_types"] = torch.randint(1, 10, (2, 8))

    captured: dict = {}

    class _CaptureODESolver:
        def __init__(self, velocity_model):
            self.velocity_model = velocity_model

        def sample(self, x_init, step_size, method, time_grid):
            captured["x_init"] = x_init
            return x_init

    adapter._ODESolver = _CaptureODESolver

    with patch.object(adapter, "_bond_decoder", None, create=True):
        try:
            adapter.generate(pocket, cfg)
        except Exception:
            pass

    # After one call to ``_generate_impl`` the audit counter on the
    # adapter must have incremented.  Per Wf-A35-2 D2, the counter
    # is exposed via ``last_sampling_diagnostics`` and is the canonical
    # way to verify the wire-in fired without parsing stdout.
    diag = getattr(adapter, "last_sampling_diagnostics", None)
    assert diag is not None, (
        "last_sampling_diagnostics was never set — _generate_impl "
        "did not enter the nucleus-sampling branch (top_p=0.85 should "
        "have activated it but no diagnostics dict was emitted)"
    )
    assert diag.get("active") is True, (
        f"nucleus_sample wire-in reported active=False despite non-default "
        f"top_p/top_k/temperature: {diag!r}"
    )
    assert diag.get("nucleus_sample_invocations", 0) > 0, (
        f"nucleus_sample_invocations is 0 — the audit counter did not "
        f"increment, so the wire-in did not fire end-to-end: {diag!r}"
    )
    assert diag.get("top_p") == 0.85
    assert diag.get("top_k") == 30
    assert diag.get("temperature") == 0.7


# ---------------------------------------------------------------------------
# Test 3: monkeypatch on diversity_sampling.nucleus_sample observed
# ---------------------------------------------------------------------------
def test_nucleus_sample_primitive_is_called_when_top_p_below_one(monkeypatch):
    """Independent path-of-truth on the wire-in: monkeypatch
    ``diversity_sampling.nucleus_sample`` to record its invocation,
    run ``_generate_impl`` once, and assert the stub was called.

    This is the structural pin — it does not depend on the audit
    counter being maintained, and any future refactor that replaces
    the audit-counter check would still need this stub call.
    """
    from molmetal.ports import GenerationConfig

    adapter = _tiny_adapter()
    pocket = _fixed_pocket()

    cfg = GenerationConfig(
        n_samples=2,
        n_steps=4,
        seed=42,
        top_p=0.85,
        top_k=30,
        temperature=0.7,
        fm_t_dist="uniform",
    )
    cfg.conditioning["fixed_atom_types"] = torch.randint(1, 10, (2, 8))

    captured_calls: list = {}

    def _spy_nucleus_sample(logits, p, k, temperature, generator=None):
        captured_calls["p"] = float(p)
        captured_calls["k"] = int(k)
        captured_calls["temperature"] = float(temperature)
        # Return a uniform sample so downstream sees the expected shape.
        return torch.randint(0, logits.shape[-1], logits.shape[:-1])

    # Patch the symbol the adapter imports lazily (per Wf-A35-2 D2).
    monkeypatch.setattr(
        "molmetal.adapters.flow_matching_lipman.diversity_sampling.nucleus_sample",
        _spy_nucleus_sample,
    )

    class _CaptureODESolver:
        def __init__(self, velocity_model):
            self.velocity_model = velocity_model

        def sample(self, x_init, step_size, method, time_grid):
            return x_init

    adapter._ODESolver = _CaptureODESolver

    with patch.object(adapter, "_bond_decoder", None, create=True):
        try:
            adapter.generate(pocket, cfg)
        except Exception:
            pass

    assert captured_calls, (
        "monkeypatched nucleus_sample was never called — the wire-in "
        "did not reach diversity_sampling.nucleus_sample despite "
        "top_p=0.85 (non-default)"
    )
    assert captured_calls["p"] == 0.85
    assert captured_calls["k"] == 30
    assert captured_calls["temperature"] == 0.7


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
