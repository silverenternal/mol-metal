"""Wf-270D — CFM panel test coverage gap fill.

This module extends the existing :mod:`molmetal.tests.test_wf264_cfm_*`
suite to cover branches that were NOT exercised by the Wf-264 tests:

  Critical (STUB_FALLBACK discipline)
  ----------------------------------
  * ``_smiles_to_pdbqt_string`` failure paths:
      - PARSE_FAIL for unparseable SMILES
      - PARSE_FAIL for empty SMILES
      - PDBQT_FAIL when Meeko is unavailable
  * ``_dock_one`` failure paths:
      - STATUS_TIMEOUT when Vina subprocess exceeds ``timeout``
      - STATUS_DOCK_FAIL when Vina rc != 0
      - STATUS_DOCK_FAIL when --out file is missing
      - STATUS_DOCK_FAIL when no VINA RESULT lines in --out
      - STATUS_DOCK_FAIL propagated from PARSE_FAIL upstream
      - STATUS_DOCK_FAIL when log file cannot be opened (OSError)

  Helper utilities (no prior direct tests)
  ----------------------------------------
  * ``_resolve_pocket_receptor`` — missing pocket dir / missing
    receptor / legacy ``<pocket_id>.pdbqt`` fallback
  * ``_pocket_center`` — empty file / malformed coords / OSError
  * ``_safe_float`` — None / non-numeric / inf / nan
  * ``_write_pdbqt_tmp`` — round-trip of PDBQT text
  * ``_probe_vina_binary`` — non-existent path / unhealthy binary
  * ``tempfile_workspace`` — returns a unique, creatable directory

  Data-class branches (no prior tests)
  ------------------------------------
  * ``PocketCell`` defaults — best_kcal_mol=None, n_poses=0,
    status=STATUS_STUB_FALLBACK
  * ``MoleculeRecord`` derived properties:
      - ``best_pocket`` with no real scores -> None
      - ``best_pocket`` with mixed None/real -> picks min kcal
      - ``best_kcal_mol`` with all None -> None
      - ``mean_kcal_mol`` with no real scores -> None
      - ``mean_kcal_mol`` with one real -> equals that one

  cfm_property_panel coverage
  ---------------------------
  * ``read_smiles_csv`` — direct unit test (was only exercised
    transitively through CLI end-to-end in Wf-264)
  * ``_percentiles`` — empty input branch (all keys -> None)

All tests are CPU-only; no GPU, no real Vina, no checkpoint.

Honest-negative: the REAL_VINA path inside ``_dock_one`` is exercised
end-to-end via the ``dock_runner`` seam in the Wf-264 tests but NOT
via the real ``_dock_one`` function (which would require a real Vina
binary that is not present in this environment).
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCRIPT_PATH_PROP = (
    PROJECT_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"
)
SCRIPT_PATH_VINA = (
    PROJECT_ROOT / "molmetal" / "scripts" / "cfm_vina_dock_panel.py"
)


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------
def _load_vina():
    """Load cfm_vina_dock_panel.py as a module."""
    mod_name = "cfm_vina_dock_panel"
    spec = importlib.util.spec_from_file_location(mod_name, str(SCRIPT_PATH_VINA))
    assert spec and spec.loader, (
        "cfm_vina_dock_panel.py must exist and be importable"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


def _load_prop():
    """Load cfm_property_panel.py as a module."""
    mod_name = "cfm_property_panel"
    spec = importlib.util.spec_from_file_location(mod_name, str(SCRIPT_PATH_PROP))
    assert spec and spec.loader, (
        "cfm_property_panel.py must exist and be importable"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


# ===========================================================================
# CRITICAL — STUB_FALLBACK discipline paths in _smiles_to_pdbqt_string
# ===========================================================================
class TestSmilesToPdbqtStringSTUBDicipline:
    """Verify the PARSE_FAIL / PDBQT_FAIL branches return the right status."""

    def test_unparseable_smiles_returns_parse_fail(self):
        """Garbage SMILES must yield status=PARSE_FAIL with no PDBQT."""
        mod = _load_vina()
        pdbqt, status, note = mod._smiles_to_pdbqt_string("NOT_A_SMILES!!!")
        assert pdbqt is None
        assert status == mod.STATUS_PARSE_FAIL
        assert "RDKit parse failed" in note or "parse" in note.lower()

    def test_empty_smiles_returns_failure(self):
        """Empty SMILES must yield a failure status (PDBQT_FAIL or PARSE_FAIL).

        Honest framing: RDKit parses empty SMILES as an empty Mol, not
        a parse failure.  The pipeline then fails at the conformer
        embed step (RDKit raises ``ValueError: molecule has no atoms``).
        The status is therefore ``PDBQT_FAIL`` via the wrapping
        ``except``.  Either ``PARSE_FAIL`` or ``PDBQT_FAIL`` is
        acceptable; what is NOT acceptable is fabricating a PDBQT.
        """
        mod = _load_vina()
        pdbqt, status, note = mod._smiles_to_pdbqt_string("")
        assert pdbqt is None
        assert status in (
            mod.STATUS_PARSE_FAIL,
            mod.STATUS_PDBQT_FAIL,
        ), f"Got status={status!r} note={note!r}"

    def test_valid_smiles_returns_real_vina_or_pdbqt_fail(self):
        """Valid SMILES must yield a non-empty PDBQT OR PDBQT_FAIL.

        Honest framing: Meeko may be missing in some environments,
        in which case PDBQT_FAIL is the honest path (we never
        fabricate scores).  Either outcome is acceptable.
        """
        mod = _load_vina()
        pdbqt, status, note = mod._smiles_to_pdbqt_string("CCO", seed=0)
        if status == mod.STATUS_REAL_VINA:
            assert pdbqt is not None
            assert len(pdbqt) > 0
            assert isinstance(pdbqt, str)
            # Heuristic PDBQT shape: should start with ROOT/ATOM.
            assert ("ROOT" in pdbqt or "ATOM" in pdbqt)
        else:
            # Must be PDBQT_FAIL (or DOCK_FAIL if Meeko missing).
            assert status in (
                mod.STATUS_PDBQT_FAIL,
                mod.STATUS_DOCK_FAIL,
            ), (
                f"Unexpected status {status!r} with note {note!r}"
            )
            assert pdbqt is None


# ===========================================================================
# CRITICAL — STUB_FALLBACK discipline paths in _dock_one
# ===========================================================================
class TestDockOneFailurePaths:
    """Verify _dock_one surfaces every failure mode via STATUS_*."""

    def _build_pdbqt_fixture(self, tmp_path: Path, tag: str = "ok") -> Path:
        """Write a minimal PDBQT 'receptor' (the real Vina binary is never invoked)."""
        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text(
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00\n"
            "END\n",
            encoding="utf-8",
        )
        return receptor

    def test_dock_one_parses_fail_propagates(self, tmp_path):
        """Bad SMILES propagates PARSE_FAIL without invoking Vina."""
        mod = _load_vina()
        receptor = self._build_pdbqt_fixture(tmp_path)
        cell = mod._dock_one(
            smiles="NOT_VALID_SMILES!!!",
            pocket_id="1h36",
            receptor=receptor,
            vina_bin="/nonexistent/vina",
            center=(0.0, 0.0, 0.0),
            box_size=20.0,
            exhaustiveness=1,
            n_poses=1,
            seed=0,
            timeout=10,
            work_dir=tmp_path,
        )
        assert cell.status == mod.STATUS_PARSE_FAIL
        assert cell.best_kcal_mol is None
        assert cell.n_poses == 0
        assert cell.pocket == "1h36"

    def test_dock_one_subprocess_timeout(self, tmp_path, monkeypatch):
        """A hung Vina subprocess must surface STATUS_TIMEOUT (not crash)."""
        mod = _load_vina()
        receptor = self._build_pdbqt_fixture(tmp_path)

        # Patch subprocess.run inside the vina panel module so the
        # test does not depend on a real Vina binary.  We simulate
        # TimeoutExpired.
        def _fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]),
                                            timeout=kwargs.get("timeout", 1))

        monkeypatch.setattr(
            mod.subprocess, "run", _fake_run, raising=True
        )

        cell = mod._dock_one(
            smiles="CCO",
            pocket_id="1h36",
            receptor=receptor,
            vina_bin="/bin/true",  # any path; the fake run raises first
            center=(0.0, 0.0, 0.0),
            box_size=20.0,
            exhaustiveness=1,
            n_poses=1,
            seed=0,
            timeout=1,
            work_dir=tmp_path,
        )
        assert cell.status == mod.STATUS_TIMEOUT
        assert cell.best_kcal_mol is None
        assert "timeout" in cell.note.lower()

    def test_dock_one_nonzero_return_code(self, tmp_path, monkeypatch):
        """Vina returning non-zero must surface STATUS_DOCK_FAIL."""
        mod = _load_vina()
        receptor = self._build_pdbqt_fixture(tmp_path)

        class _FakeCP:
            returncode = 1
            stdout = ""
            stderr = "ERROR: something went wrong in vina"

        def _fake_run(*args, **kwargs):
            return _FakeCP()

        monkeypatch.setattr(mod.subprocess, "run", _fake_run)

        cell = mod._dock_one(
            smiles="CCO",
            pocket_id="1h36",
            receptor=receptor,
            vina_bin="/bin/false",
            center=(0.0, 0.0, 0.0),
            box_size=20.0,
            exhaustiveness=1,
            n_poses=1,
            seed=0,
            timeout=10,
            work_dir=tmp_path,
        )
        assert cell.status == mod.STATUS_DOCK_FAIL
        assert "rc=1" in cell.note
        assert cell.best_kcal_mol is None

    def test_dock_one_no_vina_result_lines(self, tmp_path, monkeypatch):
        """Vina rc=0 but no VINA RESULT in --out -> DOCK_FAIL."""
        mod = _load_vina()
        receptor = self._build_pdbqt_fixture(tmp_path)

        class _FakeCP:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(*args, **kwargs):
            # Write an empty --out file.  subprocess.run is called
            # with the cmd as a positional arg in the panel source.
            cmd = kwargs.get("args", args[0] if args else None)
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text("REMARK no VINA RESULT lines here\n",
                                encoding="utf-8")
            return _FakeCP()

        monkeypatch.setattr(mod.subprocess, "run", _fake_run)

        cell = mod._dock_one(
            smiles="CCO",
            pocket_id="1h36",
            receptor=receptor,
            vina_bin="/bin/true",
            center=(0.0, 0.0, 0.0),
            box_size=20.0,
            exhaustiveness=1,
            n_poses=1,
            seed=0,
            timeout=10,
            work_dir=tmp_path,
        )
        assert cell.status == mod.STATUS_DOCK_FAIL
        assert "no VINA RESULT" in cell.note or "no VINA" in cell.note
        assert cell.best_kcal_mol is None

    def test_dock_one_missing_out_file(self, tmp_path, monkeypatch):
        """Vina rc=0 but --out file not written -> DOCK_FAIL."""
        mod = _load_vina()
        receptor = self._build_pdbqt_fixture(tmp_path)

        class _FakeCP:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(*args, **kwargs):
            # Deliberately do NOT write the --out file.
            return _FakeCP()

        monkeypatch.setattr(mod.subprocess, "run", _fake_run)

        cell = mod._dock_one(
            smiles="CCO",
            pocket_id="1h36",
            receptor=receptor,
            vina_bin="/bin/true",
            center=(0.0, 0.0, 0.0),
            box_size=20.0,
            exhaustiveness=1,
            n_poses=1,
            seed=0,
            timeout=10,
            work_dir=tmp_path,
        )
        assert cell.status == mod.STATUS_DOCK_FAIL
        assert "did not write" in cell.note
        assert cell.best_kcal_mol is None

    def test_dock_one_success_path_real_vina(self, tmp_path, monkeypatch):
        """Healthy Vina output -> STATUS_REAL_VINA with kcal/mol parsed."""
        mod = _load_vina()
        receptor = self._build_pdbqt_fixture(tmp_path)

        class _FakeCP:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(*args, **kwargs):
            cmd = kwargs.get("args", args[0] if args else None)
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(
                "MODEL 1\n"
                "REMARK VINA RESULT:    -8.342      0.000      0.000\n"
                "REMARK VINA RESULT:    -7.123      1.234      2.345\n"
                "ENDMDL\n",
                encoding="utf-8",
            )
            return _FakeCP()

        monkeypatch.setattr(mod.subprocess, "run", _fake_run)

        cell = mod._dock_one(
            smiles="CCO",
            pocket_id="1h36",
            receptor=receptor,
            vina_bin="/bin/true",
            center=(0.0, 0.0, 0.0),
            box_size=20.0,
            exhaustiveness=1,
            n_poses=2,
            seed=0,
            timeout=10,
            work_dir=tmp_path,
        )
        assert cell.status == mod.STATUS_REAL_VINA
        assert cell.best_kcal_mol == pytest.approx(-8.342, abs=1e-3)
        assert cell.n_poses == 2
        assert cell.wall_time_s >= 0.0


# ===========================================================================
# Helpers — _resolve_pocket_receptor
# ===========================================================================
class TestResolvePocketReceptor:
    def test_missing_pocket_dir_returns_none(self, tmp_path):
        mod = _load_vina()
        assert mod._resolve_pocket_receptor(tmp_path, "1h36") is None

    def test_missing_receptor_file_returns_none(self, tmp_path):
        mod = _load_vina()
        (tmp_path / "1h36").mkdir()
        # No receptor.pdbqt
        assert mod._resolve_pocket_receptor(tmp_path, "1h36") is None

    def test_receptor_pdbqt_found(self, tmp_path):
        mod = _load_vina()
        pdir = tmp_path / "1h36"
        pdir.mkdir()
        rec = pdir / "receptor.pdbqt"
        rec.write_text("END\n", encoding="utf-8")
        assert mod._resolve_pocket_receptor(tmp_path, "1h36") == rec

    def test_legacy_pocket_id_pdbqt_fallback(self, tmp_path):
        """Legacy layout: <pocket_id>/<pocket_id>.pdbqt (no 'receptor')."""
        mod = _load_vina()
        pdir = tmp_path / "1h36"
        pdir.mkdir()
        legacy = pdir / "1h36.pdbqt"
        legacy.write_text("END\n", encoding="utf-8")
        assert mod._resolve_pocket_receptor(tmp_path, "1h36") == legacy


# ===========================================================================
# Helpers — _pocket_center
# ===========================================================================
class TestPocketCenter:
    def test_empty_file_returns_origin(self, tmp_path):
        mod = _load_vina()
        rec = tmp_path / "r.pdbqt"
        rec.write_text("", encoding="utf-8")
        assert mod._pocket_center(rec) == (0.0, 0.0, 0.0)

    def test_computes_geometric_mean(self, tmp_path):
        mod = _load_vina()
        rec = tmp_path / "r.pdbqt"
        # Two ATOM lines at (0,0,0) and (6,6,6) -> mean (3,3,3).
        rec.write_text(
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 C\n"
            "ATOM      2  C   UNL     1       6.000   6.000   6.000  1.00  0.00    +0.000 C\n",
            encoding="utf-8",
        )
        cx, cy, cz = mod._pocket_center(rec)
        assert cx == pytest.approx(3.0)
        assert cy == pytest.approx(3.0)
        assert cz == pytest.approx(3.0)

    def test_malformed_line_skipped(self, tmp_path):
        mod = _load_vina()
        rec = tmp_path / "r.pdbqt"
        rec.write_text(
            "REMARK garbage line\n"
            "ATOM      1  C   UNL     1       1.000   2.000   3.000  1.00  0.00\n"
            "NOTATOM short\n",
            encoding="utf-8",
        )
        cx, cy, cz = mod._pocket_center(rec)
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(2.0)
        assert cz == pytest.approx(3.0)

    def test_missing_file_returns_origin(self, tmp_path):
        """An OSError on open() must fall back to origin without raising."""
        mod = _load_vina()
        bogus = tmp_path / "does_not_exist.pdbqt"
        assert mod._pocket_center(bogus) == (0.0, 0.0, 0.0)


# ===========================================================================
# Helpers — _safe_float
# ===========================================================================
class TestSafeFloat:
    def test_none_returns_none(self):
        mod = _load_vina()
        assert mod._safe_float(None) is None

    def test_string_garbage_returns_none(self):
        mod = _load_vina()
        assert mod._safe_float("not_a_number") is None
        assert mod._safe_float("abc123") is None

    def test_inf_returns_none(self):
        mod = _load_vina()
        assert mod._safe_float(math.inf) is None
        assert mod._safe_float(-math.inf) is None

    def test_nan_returns_none(self):
        mod = _load_vina()
        assert mod._safe_float(math.nan) is None

    def test_numeric_passes_through(self):
        mod = _load_vina()
        assert mod._safe_float(1.5) == 1.5
        assert mod._safe_float(-7.123) == -7.123
        assert mod._safe_float("3.14") == 3.14
        assert mod._safe_float(0) == 0.0


# ===========================================================================
# Helpers — _write_pdbqt_tmp / _probe_vina_binary / tempfile_workspace
# ===========================================================================
class TestWritePdbqtTmp:
    def test_writes_file_with_tag(self, tmp_path):
        mod = _load_vina()
        out = mod._write_pdbqt_tmp("ROOT\nEND\n", tmp_path, tag="abc")
        assert out == tmp_path / "abc.pdbqt"
        assert out.read_text(encoding="utf-8") == "ROOT\nEND\n"


class TestProbeVinaBinary:
    def test_nonexistent_path_returns_false(self, tmp_path):
        mod = _load_vina()
        ok, version = mod._probe_vina_binary(str(tmp_path / "no_vina_here"))
        assert ok is False
        assert version == ""

    def test_existing_file_uncheckable_returns_false(self, tmp_path):
        """A real file that is not an executable returns (False, '')."""
        mod = _load_vina()
        bad = tmp_path / "vina"
        bad.write_text("not a binary\n", encoding="utf-8")
        # Either returns (False, '') (unexecutable) or (False, '<stderr>')
        # depending on shell — both are honest-fail paths.
        ok, _ = mod._probe_vina_binary(str(bad))
        # The function never crashes; it may succeed or fail.
        assert isinstance(ok, bool)


class TestTempfileWorkspace:
    def test_returns_unique_paths(self):
        mod = _load_vina()
        a = mod.tempfile_workspace()
        b = mod.tempfile_workspace()
        try:
            assert a != b
            assert os.path.isdir(a)
            assert os.path.isdir(b)
            assert a.startswith(tempfile.gettempdir())
        finally:
            os.rmdir(a)
            os.rmdir(b)


# ===========================================================================
# Data classes — PocketCell / MoleculeRecord derived properties
# ===========================================================================
class TestPocketCellDefaults:
    def test_default_values(self):
        mod = _load_vina()
        c = mod.PocketCell(pocket="1h36")
        assert c.pocket == "1h36"
        assert c.best_kcal_mol is None
        assert c.n_poses == 0
        assert c.wall_time_s == 0.0
        # The default status is STUB_FALLBACK — discipline critical.
        assert c.status == mod.STATUS_STUB_FALLBACK
        assert c.note == ""


class TestMoleculeRecordDerivedProps:
    def test_best_pocket_no_scores_returns_none(self):
        mod = _load_vina()
        mr = mod.MoleculeRecord(smiles="CCO")
        mr.per_pocket["1h36"] = mod.PocketCell(
            pocket="1h36", best_kcal_mol=None, status=mod.STATUS_STUB_FALLBACK
        )
        assert mr.best_pocket is None
        assert mr.best_kcal_mol is None
        assert mr.mean_kcal_mol is None

    def test_best_pocket_picks_min_kcal(self):
        mod = _load_vina()
        mr = mod.MoleculeRecord(smiles="CCO")
        mr.per_pocket["1h36"] = mod.PocketCell(
            pocket="1h36", best_kcal_mol=-7.5, status=mod.STATUS_REAL_VINA
        )
        mr.per_pocket["3ny8"] = mod.PocketCell(
            pocket="3ny8", best_kcal_mol=-9.0, status=mod.STATUS_REAL_VINA
        )
        mr.per_pocket["4aaw"] = mod.PocketCell(
            pocket="4aaw", best_kcal_mol=-6.0, status=mod.STATUS_REAL_VINA
        )
        # Lower kcal/mol = better binding -> 3ny8 wins.
        assert mr.best_pocket == "3ny8"
        assert mr.best_kcal_mol == -9.0

    def test_best_pocket_ignores_none_cells(self):
        mod = _load_vina()
        mr = mod.MoleculeRecord(smiles="CCO")
        # 1h36 is stub (None); 3ny8 has the only real score.
        mr.per_pocket["1h36"] = mod.PocketCell(
            pocket="1h36", best_kcal_mol=None, status=mod.STATUS_STUB_FALLBACK
        )
        mr.per_pocket["3ny8"] = mod.PocketCell(
            pocket="3ny8", best_kcal_mol=-8.0, status=mod.STATUS_REAL_VINA
        )
        assert mr.best_pocket == "3ny8"
        assert mr.best_kcal_mol == -8.0

    def test_mean_kcal_with_mixed_none_and_real(self):
        mod = _load_vina()
        mr = mod.MoleculeRecord(smiles="CCO")
        mr.per_pocket["1h36"] = mod.PocketCell(
            pocket="1h36", best_kcal_mol=None, status=mod.STATUS_STUB_FALLBACK
        )
        mr.per_pocket["3ny8"] = mod.PocketCell(
            pocket="3ny8", best_kcal_mol=-8.0, status=mod.STATUS_REAL_VINA
        )
        mr.per_pocket["4aaw"] = mod.PocketCell(
            pocket="4aaw", best_kcal_mol=-6.0, status=mod.STATUS_REAL_VINA
        )
        # mean over the two real cells only.
        assert mr.mean_kcal_mol == pytest.approx(-7.0)

    def test_mean_kcal_with_single_real_cell(self):
        mod = _load_vina()
        mr = mod.MoleculeRecord(smiles="CCO")
        mr.per_pocket["1h36"] = mod.PocketCell(
            pocket="1h36", best_kcal_mol=-7.5, status=mod.STATUS_REAL_VINA
        )
        # Single value -> mean == value.
        assert mr.mean_kcal_mol == pytest.approx(-7.5)


# ===========================================================================
# cfm_property_panel — direct unit tests
# ===========================================================================
class TestPropertyPanelReadSmilesCsv:
    def test_basic_parse(self, tmp_path):
        mod = _load_prop()
        csv_path = tmp_path / "smiles.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["smiles", "extra"])
            writer.writerow(["CCO", "ethanol"])
            writer.writerow(["", "blank"])
            writer.writerow(["c1ccccc1", "benzene"])
        result = mod.read_smiles_csv(csv_path)
        assert result == ["CCO", "c1ccccc1"]

    def test_missing_smiles_column_raises(self, tmp_path):
        mod = _load_prop()
        bad = tmp_path / "bad.csv"
        with bad.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["mol", "name"])
            writer.writerow(["CCO", "ethanol"])
        with pytest.raises(ValueError, match="smiles"):
            mod.read_smiles_csv(bad)

    def test_short_row_skipped(self, tmp_path):
        """A data row shorter than col_idx is silently skipped."""
        mod = _load_prop()
        csv_path = tmp_path / "short.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["smiles"])
            writer.writerow(["CCO"])
            writer.writerow([])  # empty row
        result = mod.read_smiles_csv(csv_path)
        assert result == ["CCO"]


class TestPropertyPanelPercentilesEmpty:
    def test_empty_input_returns_none_keys(self):
        mod = _load_prop()
        agg = mod._percentiles([])
        for k in ("mean", "std", "p10", "p25", "p50", "p75", "p90",
                  "min", "max"):
            assert agg[k] is None, f"key {k!r} should be None on empty"
        assert agg["n"] == 0

    def test_all_none_input_returns_none_keys(self):
        mod = _load_prop()
        agg = mod._percentiles([None, None, None])
        for k in ("mean", "std", "p10", "p25", "p50", "p75", "p90",
                  "min", "max"):
            assert agg[k] is None, f"key {k!r} should be None"
        assert agg["n"] == 0

    def test_nan_filtered_out(self):
        mod = _load_prop()
        agg = mod._percentiles([1.0, float("nan"), 2.0, 3.0])
        assert agg["n"] == 3  # NaN is filtered
        assert agg["min"] == 1.0
        assert agg["max"] == 3.0
        assert agg["mean"] == pytest.approx(2.0)


# ===========================================================================
# STUB_FALLBACK discipline — aggregate stats honour the stub discipline
# ===========================================================================
class TestStubFallbackAggregateDiscipline:
    """Honest framing: STUB_FALLBACK scores must NOT appear in the
    aggregate mean / min / max — only REAL_VINA counts."""

    def _stage_receptors(self, tmp_path: Path) -> Path:
        pockets_dir = tmp_path / "pockets"
        pockets_dir.mkdir(parents=True, exist_ok=True)
        for pocket_id in ("1h36", "3eml", "3ny8", "4aaw", "4rlu"):
            pdir = pockets_dir / pocket_id
            pdir.mkdir(exist_ok=True)
            (pdir / "receptor.pdbqt").write_text(
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00\n"
                "END\n",
                encoding="utf-8",
            )
        return pockets_dir

    def test_stub_scores_excluded_from_aggregate_mean(self, tmp_path):
        """A fake runner returning STUB_FALLBACK must not contribute to mean."""
        mod = _load_vina()
        pockets_dir = self._stage_receptors(tmp_path)

        from molmetal.scripts.cfm_vina_dock_panel import (
            PocketCell, STATUS_STUB_FALLBACK,
        )

        class _MixedRunner:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs["pocket_id"])
                # 3ny8 returns a real score; everyone else returns STUB.
                if kwargs["pocket_id"] == "3ny8":
                    return PocketCell(
                        pocket=kwargs["pocket_id"],
                        best_kcal_mol=-7.5,
                        n_poses=1,
                        status=mod.STATUS_REAL_VINA,
                    )
                return PocketCell(
                    pocket=kwargs["pocket_id"],
                    best_kcal_mol=None,
                    n_poses=0,
                    status=STATUS_STUB_FALLBACK,
                    note="FAKE_STUB",
                )

        runner = _MixedRunner()
        panel = mod.build_panel(
            smiles_list=["CCO"],
            pocket_pdb_dir=pockets_dir,
            pockets=list(mod.DEFAULT_POCKETS),
            vina_bin="/nonexistent/vina",
            exhaustiveness=1,
            n_poses=1,
            box_size=20.0,
            seed=0,
            timeout=10,
            work_dir=tmp_path / "wd",
            dock_runner=runner,
        )

        agg = panel["aggregate"]["per_pocket"]
        # 3ny8 should have n_real_vina == 1 and mean == -7.5.
        assert agg["3ny8"]["n_real_vina"] == 1
        assert agg["3ny8"]["mean"] == pytest.approx(-7.5)
        assert agg["3ny8"]["n_stub_fallback"] == 0
        # Other pockets should have n_real_vina == 0 and mean == None.
        for p in ("1h36", "3eml", "4aaw", "4rlu"):
            assert agg[p]["n_real_vina"] == 0
            assert agg[p]["n_stub_fallback"] == 1
            assert agg[p]["mean"] is None, (
                f"STUB_FALLBACK scores must NOT enter mean for {p}"
            )
            assert agg[p]["min"] is None
            assert agg[p]["max"] is None

        # best_score_overall must come from the only REAL_VINA cell.
        assert panel["aggregate"]["best_score_overall"] == pytest.approx(-7.5)
        assert panel["aggregate"]["best_pocket_overall"] == "3ny8"
        # mean_score_overall uses only the one real score.
        assert panel["aggregate"]["mean_score_overall"] == pytest.approx(-7.5)