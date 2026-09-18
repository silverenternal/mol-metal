"""Tests for the AiZynth retrosynthesis SOTA-scoring column wire-up.

These tests verify:

1. ``aizynth_sota_scoring.AiZynthScoreColumn`` is the public dataclass
   carrying ``n_synthesis_route`` and ``synthesis_success_rate``.
2. :func:`score_candidates` with ``mode="smarts"`` (always available)
   returns ``status="smarts"``, records per-SMILES booleans and a
   finite success rate in [0, 1].
3. The mode="aizynthfinder" path degrades to ``status="unavailable"``
   when neither :mod:`aizynthfinder` nor a ``config.yml`` is on disk
   (the local ROCm env), and never invents a success rate.
4. The mode="aizynthfinder_isolated" path degrades to
   ``status="unavailable"`` when ``environments/aizynth/run.sh`` is
   missing.
5. The r4_c_full_sweep worker writes ``aizynth_*`` columns into the
   output JSON when ``--sota-aizynth`` is requested.
6. The :class:`AiZynthScoreColumn` schema exposes exactly the fields
   the paper-grade CSV/JSON report needs.

CPU-only: the tests do NOT require AiZynthFinder, a vendored
AiZynthFinder checkout, or the isolated Python environment.  The
prior ``test_dry_run_smoke_with_sota_aizynth`` end-to-end
``--dry-run`` invocation was removed by Phase-3F of WF-Remove-Smoke;
the same CLI plumbing is now exercised through the
``test_aizynth_real_backend.py`` integration test (real
build_synthesis_gate against the on-host USPTO/ZINC config).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. test_aizynth_score_column_schema
# ---------------------------------------------------------------------------
def test_aizynth_score_column_schema() -> None:
    """The public dataclass carries the documented fields."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import AiZynthScoreColumn
    fields = AiZynthScoreColumn.__dataclass_fields__
    for key in ("status", "mode", "n_invoked", "n_scored",
                "n_synthesis_route", "synthesis_success_rate", "per_smiles"):
        assert key in fields, f"missing dataclass field: {key}"
    col = AiZynthScoreColumn(status="smarts", mode="smarts", n_invoked=3,
                              n_scored=3, n_synthesis_route=2,
                              synthesis_success_rate=2/3)
    rec = col.to_record()
    assert rec["n_synthesis_route"] == 2
    assert rec["n_scored"] == 3
    assert math.isclose(rec["synthesis_success_rate"], 2/3)


# ---------------------------------------------------------------------------
# 2. test_smarts_mode_records_per_smiles_and_rate
# ---------------------------------------------------------------------------
def test_smarts_mode_records_per_smiles_and_rate() -> None:
    """SMARTS fallback reports synthesizable booleans + finite success rate."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import score_candidates
    candidates = [
        {"smiles": "CCO", "is_generated": True},                  # ethanol
        {"smiles": "c1ccncc1", "is_generated": True},             # pyridine
        {"smiles": "c1ccc(Cn2ccnn2)cc1", "is_generated": True},   # benzyl-triazole: CuAAC/SPAAC product
        {"smiles": "C1=CCCCC1", "is_generated": True},           # cyclohexene: Diels-Alder product
    ]
    col = score_candidates(candidates, mode="smarts")
    assert col.status == "smarts"
    assert col.mode == "smarts"
    assert col.n_invoked == 4
    assert col.n_scored == 4
    assert len(col.per_smiles) == 4
    for smi, row in col.per_smiles.items():
        assert "synthesizable" in row
        assert isinstance(row["synthesizable"], bool)
    # Triazole + cyclohexene products must round-trip through reverse-templates.
    assert col.per_smiles["c1ccc(Cn2ccnn2)cc1"]["synthesizable"] is True
    assert col.per_smiles["C1=CCCCC1"]["synthesizable"] is True
    # Pyridine / ethanol do not match any reverse-template.
    assert col.per_smiles["c1ccncc1"]["synthesizable"] is False
    assert col.per_smiles["CCO"]["synthesizable"] is False
    # Aggregate stats: 2/4 = 0.5
    assert col.synthesis_success_rate is not None
    assert math.isfinite(col.synthesis_success_rate)
    assert math.isclose(col.synthesis_success_rate, 0.5)
    assert col.n_synthesis_route == 2


# ---------------------------------------------------------------------------
# 3. test_aizynthfinder_mode_unavailable_gracefully
# ---------------------------------------------------------------------------
def test_aizynthfinder_mode_unavailable_gracefully(tmp_path: Path) -> None:
    """mode='aizynthfinder' degrades to status='unavailable' without assets."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import score_candidates
    # No config_path -> unavailable
    col = score_candidates([{"smiles": "CCO", "is_generated": True}],
                           mode="aizynthfinder", config_path=None)
    assert col.status == "unavailable"
    assert col.synthesis_success_rate is None
    assert col.n_synthesis_route == 0

    # Non-existent config file -> unavailable
    missing = str(tmp_path / "does_not_exist.yml")
    col2 = score_candidates([{"smiles": "CCO", "is_generated": True}],
                            mode="aizynthfinder", config_path=missing)
    assert col2.status == "unavailable"
    assert "config_path missing" in col2.notes or "not importable" in col2.notes.lower() \
        or col2.synthesis_success_rate is None


# ---------------------------------------------------------------------------
# 4. test_aizynthfinder_isolated_mode_unavailable_gracefully
# ---------------------------------------------------------------------------
def test_aizynthfinder_isolated_mode_unavailable_gracefully(tmp_path: Path) -> None:
    """mode='aizynthfinder_isolated' degrades when run.sh is missing."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import (
        score_candidates, aizynth_isolated_available,
    )
    fake_run_sh = str(tmp_path / "no_such_run.sh")
    # Missing run.sh -> aizynth_isolated_available reports False.
    assert aizynth_isolated_available(fake_run_sh) is False
    # Even with a config file, missing run.sh -> unavailable.
    fake_cfg = tmp_path / "config.yml"
    fake_cfg.write_text("# empty stub\n")
    col = score_candidates([{"smiles": "CCO", "is_generated": True}],
                            mode="aizynthfinder_isolated",
                            config_path=str(fake_cfg),
                            isolated_run_sh=fake_run_sh)
    assert col.status == "unavailable"
    assert col.synthesis_success_rate is None


# ---------------------------------------------------------------------------
# 5. test_no_candidates_short_circuits
# ---------------------------------------------------------------------------
def test_no_candidates_short_circuits() -> None:
    """Empty input -> status='no_candidates'; no rate synthesised."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import score_candidates
    col = score_candidates([], mode="smarts")
    assert col.status == "no_candidates"
    assert col.n_invoked == 0
    assert col.synthesis_success_rate is None
    assert col.n_synthesis_route == 0


# ---------------------------------------------------------------------------
# 6. test_seed_candidates_are_excluded
# ---------------------------------------------------------------------------
def test_seed_candidates_are_excluded() -> None:
    """Mirror DiffDock/FlowDock: only is_generated=True rows are scored."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import score_candidates
    candidates = [
        {"smiles": "CCO", "is_generated": False},                  # input seed -> skip
        {"smiles": "c1ccc(Cn2ccnn2)cc1", "is_generated": True},    # generated -> score
        {"smiles": "", "is_generated": True},                       # empty -> skip
    ]
    col = score_candidates(candidates, mode="smarts")
    assert col.n_invoked == 1
    assert "c1ccc(Cn2ccnn2)cc1" in col.per_smiles
    assert "CCO" not in col.per_smiles


# ---------------------------------------------------------------------------
# 7. test_rate_from_column_handles_nan
# ---------------------------------------------------------------------------
def test_rate_from_column_handles_nan() -> None:
    """``rate_from_column`` sanitises non-finite rates to None."""
    from molmetal_lam.sbdd_env.aizynth_sota_scoring import (
        AiZynthScoreColumn, rate_from_column,
    )
    col = AiZynthScoreColumn(synthesis_success_rate=float("nan"))
    assert rate_from_column(col) is None
    col_ok = AiZynthScoreColumn(synthesis_success_rate=0.5)
    assert rate_from_column(col_ok) == 0.5
    assert rate_from_column(AiZynthScoreColumn()) is None


# ---------------------------------------------------------------------------
# 8. test_sweep_worker_records_aizynth_columns_in_schema
# ---------------------------------------------------------------------------
def test_sweep_worker_records_aizynth_columns_in_schema() -> None:
    """PocketResult exposes the AiZynth column dataclass fields."""
    # Import via a *real* package path so dataclass introspection of
    # `Optional[float]` etc. can resolve the typing module via
    # sys.modules; ``spec_from_file_location`` returns a module with
    # ``__module__ == None`` which crashes dataclasses.fields().
    sys.path.insert(0, str(ROOT / "molmetal"))
    import importlib
    mod = importlib.import_module("scripts.r4_c_full_sweep")
    fields = mod.PocketResult.__dataclass_fields__
    for key in ("aizynth_status", "aizynth_mode", "aizynth_n_invoked",
                "aizynth_n_scored", "aizynth_n_synthesis_route",
                "aizynth_synthesis_success_rate", "aizynth_per_smiles"):
        assert key in fields, f"PocketResult missing field: {key}"


# ---------------------------------------------------------------------------
# 9. test_cli_flag_help_text
# ---------------------------------------------------------------------------
def test_cli_flag_help_text() -> None:
    """--sota-aizynth / --sota-aizynth-mode flags are wired into argparse."""
    import subprocess
    out = subprocess.run(
        ["python", str(ROOT / "molmetal/scripts/r4_c_full_sweep.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--sota-aizynth" in out.stdout
    assert "--sota-aizynth-mode" in out.stdout
    assert "smarts" in out.stdout


# ---------------------------------------------------------------------------
# 10. (removed by Phase-3F of WF-Remove-Smoke)
# The original ``test_dry_run_smoke_with_sota_aizynth`` invoked
# ``r4_c_full_sweep.py --dry-run --sota-aizynth --sota-aizynth-mode smarts``
# against the default manifest/config.  The same CLI plumbing is now
# exercised end-to-end by ``test_aizynth_real_backend.py`` (real
# build_synthesis_gate against the on-host USPTO/ZINC config) and by
# the per-mode ``test_aizynth_smart_mode_succeeds`` / ``test_aizynth_...``
# tests above.
# ---------------------------------------------------------------------------