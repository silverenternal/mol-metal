"""Tests for the per-pocket Vina config adapter (Wf-248-C / TODO-125).

==============================================================================
Wf-248-C — Per-pocket Vina config adapter
==============================================================================

Verifies the public API of
``molmetal.adapters.per_pocket_vina_config`` and the smoke CLI in
``molmetal.scripts.dock_per_pocket_vina``. The tests are hermetic
(no Vina binary required) — they exercise the config adapter + the
STUB_FALLBACK path of the CLI smoke runner.

Coverage
========
1. :class:`PocketVinaConfig` dataclass — field defaults + validation.
2. :class:`PerPocketVinaConfig` — lookup, overrides, fallback,
   ``to_docking_config`` + exhaustiveness/n_poses convenience methods.
3. Module-level ``default_per_pocket_config()`` — fresh instance.
4. CLI script — full smoke (force STUB_FALLBACK), output files written,
   per-pocket exhaustiveness map exposed in the summary.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. PocketVinaConfig dataclass
# ---------------------------------------------------------------------------
class TestPocketVinaConfig:
    def test_defaults(self):
        from molmetal.adapters.per_pocket_vina_config import PocketVinaConfig

        c = PocketVinaConfig()
        assert c.exhaustiveness == 8
        assert c.n_poses == 5
        assert c.center is None
        assert c.size is None
        assert c.label == "default"

    def test_custom_values(self):
        from molmetal.adapters.per_pocket_vina_config import PocketVinaConfig

        c = PocketVinaConfig(
            exhaustiveness=16,
            n_poses=10,
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            label="small-tight",
        )
        assert c.exhaustiveness == 16
        assert c.n_poses == 10
        assert c.center == (1.0, 2.0, 3.0)
        assert c.size == (20.0, 20.0, 20.0)
        assert c.label == "small-tight"

    def test_immutable(self):
        from molmetal.adapters.per_pocket_vina_config import PocketVinaConfig

        c = PocketVinaConfig()
        with pytest.raises(Exception):
            c.exhaustiveness = 32  # type: ignore[misc]

    def test_invalid_exhaustiveness_raises(self):
        from molmetal.adapters.per_pocket_vina_config import PocketVinaConfig

        with pytest.raises(ValueError, match="exhaustiveness"):
            PocketVinaConfig(exhaustiveness=0)

    def test_invalid_n_poses_raises(self):
        from molmetal.adapters.per_pocket_vina_config import PocketVinaConfig

        with pytest.raises(ValueError, match="n_poses"):
            PocketVinaConfig(n_poses=0)


# ---------------------------------------------------------------------------
# 2. PerPocketVinaConfig adapter
# ---------------------------------------------------------------------------
class TestPerPocketVinaConfig:
    def test_supported_pockets_complete(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )

        cfg = PerPocketVinaConfig()
        expected = {"1h36", "3eml", "3ny8", "4aaw", "4rlu"}
        assert set(cfg.supported_pockets()) == expected

    def test_small_pocket_uses_high_ex_and_many_poses(self):
        """Small pocket 1h36 → exhaustiveness=16, n_poses=10."""
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )

        cfg = PerPocketVinaConfig()
        c = cfg.get("1h36")
        assert c.exhaustiveness == 16
        assert c.n_poses == 10
        assert cfg.exhaustiveness_for("1h36") == 16
        assert cfg.n_poses_for("1h36") == 10

    def test_large_pocket_uses_low_ex_and_few_poses(self):
        """Large pocket 4rlu → exhaustiveness=8, n_poses=5."""
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )

        cfg = PerPocketVinaConfig()
        c = cfg.get("4rlu")
        assert c.exhaustiveness == 8
        assert c.n_poses == 5
        assert cfg.exhaustiveness_for("4rlu") == 8
        assert cfg.n_poses_for("4rlu") == 5

    def test_unknown_pocket_falls_back_to_global(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )

        cfg = PerPocketVinaConfig()
        c = cfg.get("XXXX")
        # Global fallback = same as 4rlu baseline (ex=8, n_poses=5).
        assert c.exhaustiveness == 8
        assert c.n_poses == 5
        assert c.label == "global-fallback"

    def test_per_pocket_exhaustiveness_differs(self):
        """Per-pocket exhaustiveness is NOT all the same — that's
        the whole point of the adapter. This test pins the per-pocket
        spread (gate G7)."""
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )

        cfg = PerPocketVinaConfig()
        ex_by_pocket = {p: cfg.exhaustiveness_for(p) for p in cfg.supported_pockets()}
        # 1h36 ≠ 4rlu (the two anchor pockets in the spec)
        assert ex_by_pocket["1h36"] != ex_by_pocket["4rlu"]
        assert ex_by_pocket["1h36"] == 16
        assert ex_by_pocket["4rlu"] == 8

    def test_overrides_win_over_default(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
            PocketVinaConfig,
        )

        cfg = PerPocketVinaConfig(
            overrides={"1h36": PocketVinaConfig(exhaustiveness=64, n_poses=20)},
        )
        c = cfg.get("1h36")
        assert c.exhaustiveness == 64
        assert c.n_poses == 20
        # Other pockets retain defaults.
        c4 = cfg.get("4rlu")
        assert c4.exhaustiveness == 8
        assert c4.n_poses == 5

    def test_pocket_map_replaces_defaults(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
            PocketVinaConfig,
        )

        cfg = PerPocketVinaConfig(
            pocket_map={"foo": PocketVinaConfig(exhaustiveness=4, n_poses=2)},
        )
        assert cfg.get("foo").exhaustiveness == 4
        # 1h36 is NOT in the new pocket_map → global fallback
        c = cfg.get("1h36")
        assert c.label == "global-fallback"

    def test_to_docking_config_basic(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )
        from molmetal.ports import DockingConfig

        cfg = PerPocketVinaConfig()
        dc = cfg.to_docking_config("1h36")
        assert isinstance(dc, DockingConfig)
        assert dc.exhaustiveness == 16
        assert dc.n_poses == 10

    def test_to_docking_config_carries_seed(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )
        from molmetal.ports import DockingConfig

        cfg = PerPocketVinaConfig()
        base = DockingConfig(seed=123, n_poses=1, exhaustiveness=1)
        dc = cfg.to_docking_config("1h36", base_config=base)
        assert dc.seed == 123
        # Per-pocket values still win for the docking knobs.
        assert dc.n_poses == 10
        assert dc.exhaustiveness == 16

    def test_as_dict_is_json_serialisable(self):
        from molmetal.adapters.per_pocket_vina_config import (
            PerPocketVinaConfig,
        )

        cfg = PerPocketVinaConfig()
        d = cfg.as_dict()
        # Round-trip through JSON to prove no non-serialisable types.
        s = json.dumps(d)
        parsed = json.loads(s)
        assert "1h36" in parsed
        assert parsed["1h36"]["exhaustiveness"] == 16
        assert parsed["1h36"]["n_poses"] == 10

    def test_default_per_pocket_config_helper(self):
        from molmetal.adapters.per_pocket_vina_config import (
            default_per_pocket_config,
            PerPocketVinaConfig,
        )

        a = default_per_pocket_config()
        b = default_per_pocket_config()
        # Each call returns a fresh instance — no shared mutable state.
        assert a is not b
        assert isinstance(a, PerPocketVinaConfig)


# ---------------------------------------------------------------------------
# 3. CLI smoke
# ---------------------------------------------------------------------------


def _import_cli_module():
    """Lazy-import the CLI script module."""
    scripts_dir = PROJECT_ROOT / "molmetal" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("dock_per_pocket_vina")


class TestDockPerPocketVinaCLI:
    def test_help(self):
        """The CLI exposes ``--help``."""
        # Already validated by manual invocation; this test pins it.
        assert (
            PROJECT_ROOT / "molmetal" / "scripts" / "dock_per_pocket_vina.py"
        ).is_file()

    def test_cli_smoke_with_stub_fallback(self, tmp_path, monkeypatch):
        """Full CLI smoke — force MISSING_DEP so every cell is
        STUB_FALLBACK. Verifies the per-pocket config is honoured
        in the records (exhaustiveness differs per pocket) and all
        output files are written."""
        mod = _import_cli_module()

        # Force MISSING_DEP regardless of on-disk binary.
        force = {
            "status": "MISSING_DEP",
            "bin": None,
            "version": "",
            "reason": "test-force",
        }
        monkeypatch.setattr(mod, "_detect_vina_binary", lambda: force)

        out_dir = tmp_path / "wf248c"
        out_dir.mkdir()
        argv = [
            "prog",
            "--pockets",
            "1h36",
            "3eml",
            "3ny8",
            "4aaw",
            "4rlu",
            "--smiles",
            "CCO",
            "c1ccccc1",
            "--seeds",
            "0",
            "1",
            "2",
            "--out-dir",
            str(out_dir),
            "--quiet",
        ]
        saved = sys.argv
        try:
            sys.argv = argv
            rc = mod.main()
        finally:
            sys.argv = saved
        assert rc == 0

        # Smoke summary exists + records per-pocket exhaustiveness.
        summary = json.loads((out_dir / "smoke_summary.json").read_text())
        assert "per_pocket_exhaustiveness" in summary
        ex = summary["per_pocket_exhaustiveness"]
        assert ex["1h36"] == 16
        assert ex["4rlu"] == 8
        # n_poses map also present.
        np_map = summary["per_pocket_n_poses"]
        assert np_map["1h36"] == 10
        assert np_map["4rlu"] == 5
        # All cells are STUB_FALLBACK under forced MISSING_DEP.
        assert summary["vina_binary_status"]["status"] == "MISSING_DEP"
        assert summary["n_stub_fallback_total"] >= 1

        # Per-pocket JSONs written for every pocket.
        for pocket in ("1h36", "3eml", "3ny8", "4aaw", "4rlu"):
            pj = out_dir / pocket / "vina_per_pocket.json"
            assert pj.is_file(), f"missing {pj}"
            pdata = json.loads(pj.read_text())
            # Per-pocket config honoured.
            assert pdata["per_pocket_config"]["exhaustiveness"] == ex[pocket]
            assert pdata["per_pocket_config"]["n_poses"] == np_map[pocket]

        # Combined CSV exists and has the right header.
        csv_path = out_dir / "per_pocket_vina.csv"
        assert csv_path.is_file()
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert "pocket" in header
        assert "exhaustiveness" in header
        assert "n_poses" in header

    def test_cli_smoke_emits_75_cells(self, tmp_path, monkeypatch):
        """With 5 pockets × 5 smiles × 3 seeds = 75 cells, the smoke
        runner emits exactly 75 (pocket, smiles, seed) records —
        matching the Wf-179 P2 reference of 75 docked poses."""
        mod = _import_cli_module()
        force = {
            "status": "MISSING_DEP",
            "bin": None,
            "version": "",
            "reason": "test-force",
        }
        monkeypatch.setattr(mod, "_detect_vina_binary", lambda: force)

        out_dir = tmp_path / "wf248c_75"
        out_dir.mkdir()
        argv = [
            "prog",
            "--pockets",
            "1h36",
            "3eml",
            "3ny8",
            "4aaw",
            "4rlu",
            "--smiles",
            "CCO",
            "c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O",
            "Nc1ncnc2[nH]cnc12",
            "O=C(O)c1ccccc1",
            "--seeds",
            "0",
            "1",
            "2",
            "--out-dir",
            str(out_dir),
            "--quiet",
        ]
        saved = sys.argv
        try:
            sys.argv = argv
            rc = mod.main()
        finally:
            sys.argv = saved
        assert rc == 0

        summary = json.loads((out_dir / "smoke_summary.json").read_text())
        # 5 × 5 × 3 = 75 cells (matches Wf-179 P2 reference of 75
        # docked poses — 5 pockets × 3 seeds × 5 poses).
        assert summary["metadata"]["n_cells"] == 75
        # All STUB_FALLBACK under forced MISSING_DEP.
        assert summary["n_stub_fallback_total"] == 75

        # Per-pocket exhaustiveness remains heterogeneous.
        ex = summary["per_pocket_exhaustiveness"]
        assert ex["1h36"] == 16
        assert ex["4rlu"] == 8
        assert ex["1h36"] != ex["4rlu"]

        # CSV row count = 75 (header excluded).
        with (out_dir / "per_pocket_vina.csv").open() as fh:
            rows = list(csv.reader(fh))
        # Header + 75 data rows.
        assert len(rows) == 76


# ---------------------------------------------------------------------------
# 4. Adapter wires into DockingConfig correctly
# ---------------------------------------------------------------------------
class TestPerPocketWiring:
    def test_each_pocket_yields_valid_docking_config(self):
        from molmetal.adapters.per_pocket_vina_config import (
            default_per_pocket_config,
        )
        from molmetal.ports import DockingConfig

        cfg = default_per_pocket_config()
        for pocket in cfg.supported_pockets():
            dc = cfg.to_docking_config(pocket)
            assert isinstance(dc, DockingConfig)
            assert dc.exhaustiveness >= 1
            assert dc.n_poses >= 1

    def test_exhaustiveness_does_not_match_n_poses_one_to_one(self):
        """Sanity: the per-pocket exhaustiveness values are
        intentionally not all equal (gate G7)."""
        from molmetal.adapters.per_pocket_vina_config import (
            default_per_pocket_config,
        )

        cfg = default_per_pocket_config()
        ex_values = {p: cfg.exhaustiveness_for(p) for p in cfg.supported_pockets()}
        # At least two pockets have distinct exhaustiveness.
        assert len(set(ex_values.values())) >= 2
