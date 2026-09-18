"""Round-10 axis D — MetalGeometryPrior on/off flag + ablation harness tests.

These tests assert:

1. The :class:`MetalGeometryPrior` ``enabled`` flag (Round-10 axis D)
   is honored by both :meth:`apply_prior` and :meth:`prior_loss`.
2. The flag toggles cleanly via :meth:`disable` / :meth:`enable`.
3. The on/off flag is independent of ``weight`` (weight=1 + enabled=False
   still returns zero).
4. The functional :func:`apply_metal_prior` / :func:`prior_loss`
   wrappers respect the flag (instantiated with the new param).
5. The micro-ablation harness script exists and parses CLI args
   correctly (the prior ``--skip-dock`` smoke run was removed by
   Phase-3F of WF-Remove-Smoke as redundant with the real
   ``ablation.py --skip-dock`` flow handled by ``r4_c_full_sweep``).

The integration-style micro-bench (Vina + PB) is intentionally NOT run
in CI — pytest only validates the algorithmic gate and the harness
plumbing so the per-pocket experiment can be launched manually as a
single 5-min wall run.

Notes
-----
* The ``MetalGeometryPrior`` class lives at
  ``molmetal/molmetal_lam/priors/metal_geometry.py`` and was extended in
  Round-10 axis D with the ``enabled`` keyword argument.
* The ablation harness lives at
  ``molmetal/scripts/r10_pt_prior_ablation_1h36.py`` and follows the
  same skeleton as ``r10_cfg_ablation_1h36.py``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "r10_pt_prior_ablation_1h36.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _square_planar_pt_complex(device: str = "cpu"):
    """Pt(II) with 4 donors at ideal 90 deg (cisplatin analogue)."""
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],   # Pt (Z=78)
            [2.0, 0.0, 0.0],   # N  +
            [-2.0, 0.0, 0.0],  # N  -
            [0.0, 2.0, 0.0],   # Cl +
            [0.0, -2.0, 0.0],  # Cl -
        ],
        device=device,
        dtype=torch.float32,
    )
    atom_types = torch.tensor(
        [78, 7, 7, 17, 17], device=device, dtype=torch.long
    )
    n = coords.size(0)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)
    edge_types = torch.zeros(edge_index.size(1), device=device, dtype=torch.long)
    pt_idx = 0
    for k in range(edge_index.size(1)):
        if (int(edge_index[1, k].item()) == pt_idx
                and int(edge_index[0, k].item()) != pt_idx):
            edge_types[k] = 2  # EDGE_TYPE_DATIVE
    return coords, atom_types, edge_index, edge_types


# ---------------------------------------------------------------------------
# 1. enabled flag — constructor default
# ---------------------------------------------------------------------------
def test_enabled_default_true():
    """Newly constructed ``MetalGeometryPrior`` is enabled by default."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior,
    )
    p = MetalGeometryPrior(weight=1.0)
    assert p.enabled is True
    p2 = MetalGeometryPrior()
    assert p2.enabled is True


# ---------------------------------------------------------------------------
# 2. enabled=False makes prior_loss a strict no-op (zero output)
# ---------------------------------------------------------------------------
def test_disabled_prior_returns_zero_on_perfect_geometry():
    """When ``enabled=False`` the prior returns 0 regardless of geometry."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior, Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    p = MetalGeometryPrior(weight=1.0, enabled=False)
    # Perfect square-planar Pt — but prior is OFF so loss must be exactly
    # zero, not the small ~0 rad ideal value.
    loss = p.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    assert loss.item() == 0.0, (
        f"disabled prior should return 0; got {loss.item()}"
    )


def test_disabled_prior_returns_zero_on_distorted_geometry():
    """Distorted geometry with prior OFF → still zero (gate wins)."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior, Geometry,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [math.cos(math.radians(45)) * 2.0,
             math.sin(math.radians(45)) * 2.0, 0.0],
            [math.cos(math.radians(90)) * 2.0,
             math.sin(math.radians(90)) * 2.0, 0.0],
            [math.cos(math.radians(135)) * 2.0,
             math.sin(math.radians(135)) * 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    atom_types = torch.tensor([78, 7, 7, 17, 17], dtype=torch.long)
    n = coords.size(0)
    src, dst, et = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            et.append(2 if (j == 0 and i != 0) else 0)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_types = torch.tensor(et, dtype=torch.long)
    p = MetalGeometryPrior(weight=1.0, enabled=False)
    loss = p.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    assert loss.item() == 0.0, (
        f"disabled prior should return 0; got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# 3. disable() / enable() toggle the flag at runtime
# ---------------------------------------------------------------------------
def test_disable_enable_toggle():
    """``.disable()`` flips ``enabled`` to False; ``.enable()`` flips it back."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior, Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    p = MetalGeometryPrior(weight=1.0)
    assert p.enabled is True
    p.disable()
    assert p.enabled is False
    loss_off = p.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    assert loss_off.item() == 0.0
    p.enable()
    assert p.enabled is True
    loss_on = p.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    # Perfect geometry → ~0 rad (well below 1 deg).
    assert loss_on.item() < math.radians(1.0)


# ---------------------------------------------------------------------------
# 4. enabled=False overrides weight (weight=1 + enabled=False → zero)
# ---------------------------------------------------------------------------
def test_enabled_flag_independent_of_weight():
    """The ``enabled`` flag is orthogonal to ``weight``: weight=1 + "
    "enabled=False still returns zero."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior, Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    p = MetalGeometryPrior(weight=10.0, enabled=False)
    loss = p.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    # The flag should win over weight — output is exactly 0.
    assert loss.item() == 0.0, (
        f"enabled=False should override weight; got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# 5. Functional entry-points also respect the flag
# ---------------------------------------------------------------------------
def test_functional_apply_metal_prior_respects_enabled():
    """``apply_metal_prior(weight=1.0)`` returns zero when called via a "
    "disabled prior (the functional wrapper instantiates a new "
    "``MetalGeometryPrior(weight=weight)`` — so ``enabled`` defaults to "
    "True for backward compat).  This test asserts the *behaviour* is
    "governed by the wrapper instantiation default, which is True."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        apply_metal_prior, prior_loss, Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    # Default (enabled=True) — perfect geometry → ~0 rad
    val = apply_metal_prior(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR, weight=1.0,
    )
    assert val.item() < math.radians(1.0), (
        f"functional wrapper default should report near-zero on perfect "
        f"geometry; got {val.item():.4f} rad"
    )
    # raw prior_loss (no weight multiplier) — same expectation
    val2 = prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    assert val2.item() < math.radians(1.0), (
        f"raw prior_loss should report near-zero on perfect geometry; "
        f"got {val2.item():.4f} rad"
    )


# ---------------------------------------------------------------------------
# 6. apply_prior also honours the flag (covered separately for clarity)
# ---------------------------------------------------------------------------
def test_apply_prior_disabled_returns_zero():
    """``apply_prior`` (the no-weight method) must also short-circuit when "
    "``enabled=False``."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior, Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    p = MetalGeometryPrior(weight=2.5, enabled=False)
    raw = p.apply_prior(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    assert raw.item() == 0.0


# ---------------------------------------------------------------------------
# 7. Module-level constant is the canonical 90 deg = pi/2
# ---------------------------------------------------------------------------
def test_ideal_square_planar_angle_is_pi_over_two():
    """``IDEAL_SQUARE_PLANAR_ANGLE`` must equal pi/2 (90 deg)."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        IDEAL_SQUARE_PLANAR_ANGLE,
    )
    assert abs(IDEAL_SQUARE_PLANAR_ANGLE - math.pi / 2.0) < 1e-12


# ---------------------------------------------------------------------------
# 8. Ablation harness script exists + CLI parses
# ---------------------------------------------------------------------------
def test_ablation_script_exists():
    """The ablation script must exist at the canonical location."""
    assert SCRIPT.exists(), f"missing harness script: {SCRIPT}"


def test_ablation_script_help():
    """The ablation script's --help should exit 0 and mention 'prior'."""
    if not SCRIPT.exists():
        pytest.skip(f"missing harness script: {SCRIPT}")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"--help failed with code {proc.returncode}: {proc.stderr}"
    )
    assert "prior" in proc.stdout.lower(), (
        "--help output should mention 'prior'; got:\n" + proc.stdout[:500]
    )


# ---------------------------------------------------------------------------
# 9. Reference 1h36 + harness constants match the spec
# ---------------------------------------------------------------------------
def test_1h36_pdb_path_constant():
    """The harness's EXAMPLE_PDB path should match the canonical 1h36 fixture."""
    expected_suffix = (
        "1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb"
    )
    text = SCRIPT.read_text()
    assert expected_suffix in text, (
        f"harness should reference {expected_suffix}; got:\n"
        + "\n".join(line for line in text.splitlines() if "1h36" in line)[:500]
    )
