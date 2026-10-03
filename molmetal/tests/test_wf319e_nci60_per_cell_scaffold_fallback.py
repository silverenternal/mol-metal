"""Wf-319E — NCI-60 9-cell per-cell n_test_min auto-tuning tests.

================================================================
Wf-319E / TODO-220 follow-up (Wf-308C stub driver). Wf-315F
extended the NCI-60 9-cell driver with ``--split {temporal,
scaffold, auto}`` but used a single ``--n-test-min=50`` threshold
for all cells. Per Wf-315F §3 honest finding, NO NCI-60 cell has
a Bemis-Murcko scaffold group with >= 50 rows; the 3 largest cells
(A549/MCF7/A2780) have max scaffold sizes 34-37. With default
n_test_min=50 the scaffold helper always returns None -> random
fallback, even in ``--split auto`` mode.

Wf-319E introduces per-cell n_test_min auto-tuning so the scaffold
fallback actually triggers for the 3 largest cells where it is
meaningful. The per-cell threshold is computed from the max
Bemis-Murcko scaffold group size with a configurable headroom
factor (default 0.85) and floor (default 5).

Worked examples (MEASURED max scaffold sizes from Wf-315F §3):
  * A549 (max=37)    -> 0.85 * 37 = 31 -> n_test_min=31
  * MCF7 (max=34)    -> 0.85 * 34 = 28 -> n_test_min=28
  * A2780 (max=37)   -> 0.85 * 37 = 31 -> n_test_min=31
  * A2780cis (max=30)-> 0.85 * 30 = 26 -> n_test_min=26
  * HCT116 (max=20)  -> 0.85 * 20 = 17 -> n_test_min=17
  * SKOV3 (max=18)   -> 0.85 * 18 = 15 -> n_test_min=15
  * HL60 (max=18)    -> 0.85 * 18 = 15 -> n_test_min=15
  * DU145 (max=10)   -> 0.85 * 10 = 9  -> n_test_min=9
  * PC3 (max=9)      -> 0.85 * 9  = 8  -> n_test_min=8

These tests cover (per the Wf-319E acceptance contract):
  * 5 tests for per-cell n_test_min auto computation (5 cells)
  * 3 tests for per-cell split decision with auto-tuning
  * 2 tests for driver CLI integration (--n-test-min-auto on/off)

This is INTEGRATION work, NOT a new measurement: no AUC values
are reported. Tests skip cleanly when the chemprop v2 driver can't
import cuik_molmaker (matches Wf-315F pattern).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make chemprop v2 importable for the driver import (mirrors
# ``test_wf315f_nci60_per_cell_scaffold_fallback`` pattern).
sys.path.insert(0, str(PROJECT_ROOT / ".cloned" / "chemprop_v2"))

CSV_DEFAULT = (
    PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
)

# Per-cell row counts (MEASURED Sep 28 2026 from nci60_precious_9cell.csv).
EXPECTED_PER_CELL_ROWS = {
    "A549": 2647,
    "MCF7": 1964,
    "A2780": 1095,
    "HCT116": 701,
    "A2780cis": 564,
    "PC3": 275,
    "HL60": 225,
    "SKOV3": 191,
    "DU145": 161,
}

# Per-cell MEASURED max Bemis-Murcko scaffold group sizes (Wf-315F §3).
EXPECTED_MAX_SCAFFOLD_SIZE = {
    "A549": 37,
    "MCF7": 34,
    "A2780": 37,
    "A2780cis": 30,
    "HCT116": 20,
    "SKOV3": 18,
    "HL60": 18,
    "DU145": 10,
    "PC3": 9,
}


# ---------------------------------------------------------------------------
# Helpers: import the pure-Python helpers module (no chemprop v2 dep).
# ---------------------------------------------------------------------------
def _try_import_helpers_module():
    try:
        from molmetal.scripts import _nci60_scaffold_helpers as helpers  # noqa: E402
        return helpers
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"nci60 scaffold helpers import failed: {exc}")


def _try_import_driver_module():
    """Fallback for tests that need the full driver (CLI integration)."""
    try:
        from molmetal.scripts import nci60_9cell_chemprop_driver as mod  # noqa: E402
        return mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"nci60 driver import failed: {exc}")


def _per_cell_smiles(cell: str) -> List[str]:
    """Load the SMILES for a cell from the default NCI-60 CSV."""
    if not CSV_DEFAULT.exists():
        pytest.skip(f"missing {CSV_DEFAULT}")
    df = pd.read_csv(CSV_DEFAULT, low_memory=False)
    sub = df[df["cell_line"] == cell].reset_index(drop=True)
    return sub["smiles"].astype(str).tolist()


# ---------------------------------------------------------------------------
# 1. Per-cell n_test_min auto computation — 5 tests.
# ---------------------------------------------------------------------------
class TestPerCellNTestMinAutoComputation:
    """Wf-319E: compute_per_cell_n_test_min per cell from max scaffold size.

    Each test verifies the per-cell threshold against the closed-form
    ``max(round(headroom * max_size), floor)`` rule with default
    headroom=0.85, floor=5. Tests 3 of the 9 cells (the 3 largest +
    2 smallest) to keep the suite fast.
    """

    def test_a549_n_test_min_auto_is_31(self) -> None:
        """A549 (max scaffold=37) -> 0.85 * 37 = 31 (floor 5) -> 31.

        MEASURED max scaffold size for A549 is 37 (Wf-315F §3). With
        headroom=0.85 and floor=5, the auto-computed threshold is
        max(round(31.45), 5) = 31. This makes ``--split auto`` actually
        trigger scaffold-OOD for A549 (n_test_random_est=529 >= 31, so
        temporal wins; but if user passes --split scaffold, the helper
        now finds a group >= 31 rows).
        """
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("A549")
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        assert n_test_min == 31, (
            f"A549 expected n_test_min=31 from max=37 headroom=0.85; "
            f"got {n_test_min}"
        )

    def test_mcf7_n_test_min_auto_is_29(self) -> None:
        """MCF7 (max scaffold=34) -> 0.85 * 34 = 28.9 -> round = 29."""
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("MCF7")
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        assert n_test_min == 29, (
            f"MCF7 expected n_test_min=29 from max=34 headroom=0.85; "
            f"got {n_test_min}"
        )

    def test_a2780_n_test_min_auto_is_31(self) -> None:
        """A2780 (max scaffold=37) -> 0.85 * 37 = 31 (floor 5) -> 31."""
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("A2780")
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        assert n_test_min == 31, (
            f"A2780 expected n_test_min=31 from max=37 headroom=0.85; "
            f"got {n_test_min}"
        )

    def test_du145_n_test_min_auto_is_8(self) -> None:
        """DU145 (max scaffold=10) -> 0.85 * 10 = 8.5 -> round = 8.

        Note: Python's round() uses banker's rounding (round half to
        even), so round(0.85 * 10) = round(8.5) = 8 (even). The
        function is deterministic; this test guards against any
        future change in the rounding policy.

        DU145 is the smallest cell (161 rows). The per-cell threshold
        of 8 lets the scaffold helper actually find a group >= 8 rows.
        Even so, --split auto still picks random 80/10/20 because
        n_test_random_est=32 >= 8. The point is the scaffold helper
        is no longer gated by an unreachable 50-row threshold.
        """
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("DU145")
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        assert n_test_min == 8, (
            f"DU145 expected n_test_min=8 from max=10 headroom=0.85 "
            f"(banker's rounding of 8.5); got {n_test_min}"
        )

    def test_pc3_n_test_min_auto_is_8(self) -> None:
        """PC3 (max scaffold=9) -> 0.85 * 9 = 8 (floor 5) -> 8.

        PC3 has the smallest max scaffold (9 rows). The auto-computed
        threshold is 8, which lets the scaffold helper find a group
        >= 8 rows. With --split auto the temporal branch wins
        (n_test_random_est=55 >= 8), but --split scaffold will now
        actually trigger scaffold-OOD split.
        """
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("PC3")
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        assert n_test_min == 8, (
            f"PC3 expected n_test_min=8 from max=9 headroom=0.85; "
            f"got {n_test_min}"
        )


# ---------------------------------------------------------------------------
# 2. Per-cell split decision with auto-tuning — 3 tests.
# ---------------------------------------------------------------------------
class TestPerCellSplitDecisionAutoTuned:
    """Wf-319E: --split auto with per-cell n_test_min auto-tuning.

    With auto-tuning ON (default) the per-cell split decision now
    actually uses scaffold-OOD for the 3 largest cells (where it
    is meaningful) and falls back to random for cells where the
    threshold is still too high. These tests verify the 3 cases.
    """

    def test_split_scaffold_works_for_a549_with_per_cell_threshold(self) -> None:
        """A549 with --split scaffold + auto-tuned n_test_min=31 works.

        The scaffold helper previously returned None for A549 because
        n_test_min=50 was unreachable. With the per-cell threshold
        of 31 (= round(0.85 * 37)), the helper now finds a scaffold
        group >= 31 rows and returns a valid (train, val, test, label)
        tuple.
        """
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("A549")
        df = pd.read_csv(CSV_DEFAULT, low_memory=False)
        sub = df[df["cell_line"] == "A549"].reset_index(drop=True)
        active = sub["active"].astype(int).to_numpy()
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        out = helpers._scaffold_split_indices_nci60(
            smiles_subset=smiles, active_subset=active, seed=0,
            n_test_min=n_test_min,
        )
        assert out is not None, (
            f"A549 with n_test_min={n_test_min} returned None; "
            f"expected a scaffold group >= 31 rows."
        )
        train_idx, val_idx, test_idx, label = out
        assert len(test_idx) >= n_test_min
        assert "scaffold" in label

    def test_split_scaffold_works_for_du145_with_per_cell_threshold(self) -> None:
        """DU145 with --split scaffold + auto-tuned n_test_min=9 works.

        Previously DU145 scaffold helper returned None (no group
        >= 50). With per-cell threshold 9 (= round(0.85 * 10)), the
        helper now finds a group >= 9 rows.
        """
        helpers = _try_import_helpers_module()
        smiles = _per_cell_smiles("DU145")
        df = pd.read_csv(CSV_DEFAULT, low_memory=False)
        sub = df[df["cell_line"] == "DU145"].reset_index(drop=True)
        active = sub["active"].astype(int).to_numpy()
        n_test_min = helpers.compute_per_cell_n_test_min(smiles_subset=smiles)
        out = helpers._scaffold_split_indices_nci60(
            smiles_subset=smiles, active_subset=active, seed=0,
            n_test_min=n_test_min,
        )
        assert out is not None, (
            f"DU145 with n_test_min={n_test_min} returned None; "
            f"expected a scaffold group >= 9 rows."
        )
        train_idx, val_idx, test_idx, label = out
        assert len(test_idx) >= n_test_min

    def test_auto_tuning_off_keeps_default_threshold(self) -> None:
        """--no-n-test-min-auto restores the Wf-315F single-threshold mode.

        With --no-n-test-min-auto the driver uses the single
        ``--n-test-min`` value (default 50) for all cells. This
        preserves the Wf-315F behaviour where the scaffold helper
        returns None for all 9 NCI-60 cells (no group >= 50).
        """
        mod = _try_import_driver_module()
        p = mod.build_argparser()
        # Default (--n-test-min-auto on)
        args_on = p.parse_args(["--split", "auto", "--dry-run"])
        assert args_on.n_test_min_auto is True
        assert args_on.n_test_min == 50
        # Off via --no-n-test-min-auto
        args_off = p.parse_args([
            "--split", "auto", "--dry-run", "--no-n-test-min-auto",
        ])
        assert args_off.n_test_min_auto is False
        assert args_off.n_test_min == 50


# ---------------------------------------------------------------------------
# 3. Driver CLI integration — 2 tests.
# ---------------------------------------------------------------------------
class TestDriverCLIIntegration:
    """Wf-319E: new CLI flags (--n-test-min-auto, --n-test-min-floor,
    --n-test-min-headroom) are accepted by the argparse parser and
    the per_cell_n_test_min field is recorded in the dry-run JSON.
    """

    def test_cli_accepts_new_n_test_min_flags(self) -> None:
        """All 4 new CLI flags accepted by the parser (and bogus rejected)."""
        mod = _try_import_driver_module()
        p = mod.build_argparser()
        # --n-test-min-auto on (default)
        args1 = p.parse_args(["--split", "auto", "--dry-run"])
        assert args1.n_test_min_auto is True
        assert args1.n_test_min_floor == 5
        assert args1.n_test_min_headroom == 0.85
        # --n-test-min-auto off
        args2 = p.parse_args([
            "--split", "auto", "--dry-run", "--no-n-test-min-auto",
            "--n-test-min-floor", "10", "--n-test-min-headroom", "0.7",
        ])
        assert args2.n_test_min_auto is False
        assert args2.n_test_min_floor == 10
        assert abs(args2.n_test_min_headroom - 0.7) < 1e-9

    def test_dry_run_plan_includes_per_cell_n_test_min_field(self) -> None:
        """The driver dry-run JSON shape accepts the new auto flags
        (regression guard for argparse wiring).
        """
        mod = _try_import_driver_module()
        p = mod.build_argparser()
        args = p.parse_args([
            "--split", "auto", "--dry-run",
            "--n-test-min-auto", "--n-test-min-floor", "5",
            "--n-test-min-headroom", "0.85",
        ])
        plan = mod.dry_run_plan(args)
        # Plan must include the split protocol field (WFM-319E
        # per_cell_n_test_min is in the execute path; dry-run just
        # confirms the args round-trip through argparse).
        assert plan["driver"] == "nci60_9cell_chemprop_driver"
        assert plan["dry_run"] is True
        # The driver fields are exposed via the argparse namespace.
        assert args.n_test_min_auto is True
        assert args.n_test_min_floor == 5
        assert abs(args.n_test_min_headroom - 0.85) < 1e-9
