"""Tests for Wf-30 Phase 2 — ``dock_10pockets_se3sb.py``.

These tests pin the structural + CLI contract of the docking sweep
script without spinning up a real Vina installation:

1. ``test_script_imports`` — the script imports cleanly (no syntax
   errors, no circular imports).
2. ``test_argparse_defaults`` — every documented flag has its
   expected default.
3. ``test_pocket_subset`` — ``--pockets 830c 1h36`` restricts the
   sweep to exactly two pockets; the parsed namespace records the
   requested pocket list verbatim.
4. ``test_method_subset`` — ``--methods se3sb baseline`` restricts
   the method set to those two; the method list is validated against
   the supported enum.
5. ``test_default_pockets_count_is_10`` — the default pocket cohort
   is exactly the WF-30 ten-pocket list.
6. ``test_default_methods_count_is_4`` — the default method set is
   ``(se3sb, transport, multiph, baseline)``.
7. ``test_default_seeds_count_is_5`` — the default seed set is
   ``(0, 1, 2, 3, 4)``.
8. ``test_load_default_candidates_parses_5`` — the bundled
   ``_demo_5_metaldrugs.txt`` parses to exactly 5 candidates with
   unique SMILES.
9. ``test_load_candidates_csv_round_trip`` — a 2-column CSV
   round-trips through :func:`load_candidates_csv`.
10. ``test_select_candidates_caps_at_n`` — :func:`select_candidates`
    honours the ``--n-candidates`` cap.
11. ``test_pocket_bundle_missing_raises`` — when the receptor.pdbqt
    is missing for a requested pocket, :func:`load_pocket_bundle`
    raises :class:`FileNotFoundError` with the pocket_id in the
    message.
12. ``test_trajectory_generators_exposed`` — every documented method
    has a registered trajectory generator.
13. ``test_dock_one_returns_result_dataclass`` — :func:`dock_one`
    returns a :class:`DockResult` with the documented fields.
14. ``test_run_sweep_smoke_two_pockets`` — full :func:`run_sweep`
    completes on a tiny 2-pocket × 1-candidate × 2-seed × 2-method
    configuration in <30 s (CPU stub scorer).
15. ``test_metrics_path_written`` — :func:`main` writes both the
    metrics JSON and the summary JSON.

All tests run on CPU and use the deterministic stub scorer (no real
Vina required). The script's :func:`_stub_score` is what makes the
suite CI-cheap.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"

# The bundled candidate file is the default for the dock script.
DEFAULT_CANDIDATES_TXT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "_demo_5_metaldrugs.txt"
)


# ---------------------------------------------------------------------------
# Helpers — lazy module import (mirrors test_train_se3sb_smoke.py)
# ---------------------------------------------------------------------------
def _import_dock_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import dock_10pockets_se3sb  # type: ignore[import-not-found]

    return dock_10pockets_se3sb


# ---------------------------------------------------------------------------
# 1. Script imports cleanly
# ---------------------------------------------------------------------------
def test_script_imports() -> None:
    """The dock script imports without raising (no syntax / circular import)."""
    mod = _import_dock_module()
    assert hasattr(mod, "main")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "run_sweep")
    assert hasattr(mod, "aggregate_results")


# ---------------------------------------------------------------------------
# 2. argparse defaults
# ---------------------------------------------------------------------------
def test_argparse_defaults() -> None:
    """Every documented CLI flag has the expected default value."""
    mod = _import_dock_module()
    args = mod.parse_args([])
    assert args.n_candidates == mod.DEFAULT_N_CANDIDATES
    assert args.n_seeds == len(mod.DEFAULT_SEEDS)
    assert args.methods == list(mod.DEFAULT_METHODS)
    assert args.pockets == list(mod.DEFAULT_POCKETS)
    assert args.n_steps == mod.SE3_INTEGRATION_STEPS
    # Boolean defaults
    assert args.use_real_vina is False
    assert args.quiet is False
    assert args.seed_offset == 0


# ---------------------------------------------------------------------------
# 3. --pockets subset
# ---------------------------------------------------------------------------
def test_pocket_subset() -> None:
    """``--pockets 830c 1h36`` narrows the cohort to exactly those two."""
    mod = _import_dock_module()
    args = mod.parse_args(["--pockets", "830c", "1h36"])
    assert args.pockets == ["830c", "1h36"]
    assert len(args.pockets) == 2
    # Cross-product math: 2 × 5 × 4 × 5 = 200 default evals.
    n = (
        len(args.pockets)
        * mod.DEFAULT_N_CANDIDATES
        * len(mod.DEFAULT_METHODS)
        * len(mod.DEFAULT_SEEDS)
    )
    assert n == 200


# ---------------------------------------------------------------------------
# 4. --methods subset
# ---------------------------------------------------------------------------
def test_method_subset() -> None:
    """``--methods se3sb baseline`` restricts the method list to those two."""
    mod = _import_dock_module()
    args = mod.parse_args(["--methods", "se3sb", "baseline"])
    assert args.methods == ["se3sb", "baseline"]
    # All selected methods must be in the registered generators.
    for m in args.methods:
        assert m in mod.TRAJECTORY_GENERATORS


def test_methods_unknown_rejected() -> None:
    """Unknown method names must be rejected by argparse."""
    mod = _import_dock_module()
    with pytest.raises(SystemExit):
        mod.parse_args(["--methods", "not-a-method"])


# ---------------------------------------------------------------------------
# 5-7. Default cohort sizes
# ---------------------------------------------------------------------------
def test_default_pockets_count_is_10() -> None:
    """The default pocket cohort is exactly the WF-30 ten-pocket list."""
    mod = _import_dock_module()
    assert len(mod.DEFAULT_POCKETS) == 10
    # Pin the order so any accidental edit triggers a clear failure.
    assert mod.DEFAULT_POCKETS == (
        "830c",
        "1h36",
        "3eml",
        "3ny8",
        "4aaw",
        "4rlu",
        "1g9v",
        "2hnx",
        "4rn0",
        "1fmc",
    )


def test_default_methods_count_is_4() -> None:
    """The default method set has exactly 4 entries."""
    mod = _import_dock_module()
    assert len(mod.DEFAULT_METHODS) == 4
    assert mod.DEFAULT_METHODS == ("se3sb", "transport", "multiph", "baseline")


def test_default_seeds_count_is_5() -> None:
    """The default seed set has exactly 5 entries."""
    mod = _import_dock_module()
    assert len(mod.DEFAULT_SEEDS) == 5
    assert mod.DEFAULT_SEEDS == (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# 8. Default candidate pool parses to 5 unique SMILES
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not DEFAULT_CANDIDATES_TXT.exists(),
    reason=f"default candidate file not at {DEFAULT_CANDIDATES_TXT}",
)
def test_load_default_candidates_parses_5() -> None:
    """The bundled metal-drug demo parses to exactly 5 unique SMILES."""
    mod = _import_dock_module()
    candidates = mod.load_default_candidates(DEFAULT_CANDIDATES_TXT)
    assert len(candidates) == 5
    smiles = [c.smiles for c in candidates]
    assert len(set(smiles)) == 5, f"duplicate SMILES: {smiles}"
    # Each candidate has a non-empty name.
    assert all(c.name for c in candidates)


# ---------------------------------------------------------------------------
# 9. CSV candidate round-trip
# ---------------------------------------------------------------------------
def test_load_candidates_csv_round_trip(tmp_path: Path) -> None:
    """A 2-column CSV round-trips through :func:`load_candidates_csv`."""
    mod = _import_dock_module()
    csv_path = tmp_path / "cands.csv"
    csv_path.write_text("name,smiles\n" "foo,CCO\n" "bar,c1ccccc1\n")
    candidates = mod.load_candidates_csv(csv_path)
    assert len(candidates) == 2
    assert candidates[0].name == "foo"
    assert candidates[0].smiles == "CCO"
    assert candidates[1].name == "bar"
    assert candidates[1].smiles == "c1ccccc1"


def test_load_candidates_csv_rejects_missing_columns(tmp_path: Path) -> None:
    """CSV without the required columns must raise :class:`ValueError`."""
    mod = _import_dock_module()
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        mod.load_candidates_csv(csv_path)


# ---------------------------------------------------------------------------
# 10. select_candidates caps the pool
# ---------------------------------------------------------------------------
def test_select_candidates_caps_at_n() -> None:
    """:func:`select_candidates` honours the ``--n-candidates`` cap."""
    mod = _import_dock_module()
    pool = [mod.Candidate(name=f"c{i}", smiles="C") for i in range(10)]
    selected = mod.select_candidates(pool, n=3)
    assert len(selected) == 3
    # Cap larger than pool returns the full pool.
    selected = mod.select_candidates(pool, n=20)
    assert len(selected) == 10
    # n=0 returns the full pool (interpret as "no cap").
    selected = mod.select_candidates(pool, n=0)
    assert len(selected) == 10


# ---------------------------------------------------------------------------
# 11. Pocket bundle missing → clear error
# ---------------------------------------------------------------------------
def test_pocket_bundle_missing_raises(tmp_path: Path) -> None:
    """Missing receptor.pdbqt raises FileNotFoundError mentioning the id."""
    mod = _import_dock_module()
    with pytest.raises(FileNotFoundError) as excinfo:
        mod.load_pocket_bundle("nonexistent_pocket", pocket_dir=tmp_path)
    assert "nonexistent_pocket" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 12. Trajectory generators exposed
# ---------------------------------------------------------------------------
def test_trajectory_generators_exposed() -> None:
    """Every documented method has a registered trajectory generator."""
    mod = _import_dock_module()
    expected = {"se3sb", "transport", "multiph", "baseline"}
    assert expected.issubset(set(mod.TRAJECTORY_GENERATORS))
    # Each generator must be callable with (smiles, pocket, seed).
    for method in expected:
        assert callable(mod.TRAJECTORY_GENERATORS[method])


# ---------------------------------------------------------------------------
# 13. dock_one returns DockResult
# ---------------------------------------------------------------------------
def test_dock_one_returns_result_dataclass(tmp_path: Path) -> None:
    """:func:`dock_one` returns a :class:`DockResult` with the documented fields."""
    mod = _import_dock_module()
    # Stage a tiny receptor.pdbqt so the pocket bundle can be built.
    pocket_id = "test_pocket"
    pocket_dir = tmp_path / "wf30"
    (pocket_dir / pocket_id).mkdir(parents=True)
    (pocket_dir / pocket_id / "receptor.pdbqt").write_text(
        # Two dummy ATOM lines so the centroid + n_receptor_atoms parse.
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
        "ATOM      2  C   ALA A   2       3.000   3.000   3.000  1.00  0.00    +0.000 C \n"
    )
    bundle = mod.load_pocket_bundle(pocket_id, pocket_dir)
    res = mod.dock_one(
        method="baseline",
        smiles="C",  # methane
        pocket=bundle,
        seed=0,
        n_steps=4,
        vina_adapter=None,
        use_real_vina=False,
    )
    assert isinstance(res, mod.DockResult)
    assert res.pocket_id == pocket_id
    assert res.method == "baseline"
    assert res.seed == 0
    assert res.status in {"docked", "trajectory_failed", "docking_failed"}
    # The stub scorer always returns a finite kcal/mol for a valid pose.
    if res.status == "docked":
        assert res.vina_kcal_mol is not None
        # Vina kcal/mol should be in a realistic range.
        assert -15.0 <= float(res.vina_kcal_mol) <= 0.0


# ---------------------------------------------------------------------------
# 14. Full run_sweep smoke on a tiny cohort
# ---------------------------------------------------------------------------
def test_run_sweep_smoke_two_pockets(tmp_path: Path) -> None:
    """Full :func:`run_sweep` completes on a 2×1×2×2 cohort in <30 s."""
    mod = _import_dock_module()
    pocket_dir = tmp_path / "wf30"
    pocket_ids: List[str] = ["p1", "p2"]
    for pid in pocket_ids:
        (pocket_dir / pid).mkdir(parents=True)
        (pocket_dir / pid / "receptor.pdbqt").write_text(
            "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
            "ATOM      2  N   ALA A   2       4.000   4.000   4.000  1.00  0.00    +0.000 N \n"
        )
    candidates = [mod.Candidate(name="methane", smiles="C")]
    results = mod.run_sweep(
        pockets=pocket_ids,
        candidates=candidates,
        methods=("se3sb", "baseline"),
        seeds=(0, 1),
        pocket_dir=pocket_dir,
        n_steps=2,
        cpu_count=1,
        use_real_vina=False,
    )
    # 2 pockets × 1 candidate × 2 methods × 2 seeds = 8 evals.
    assert len(results) == 8
    # Each result must be a DockResult with finite elapsed time.
    for r in results:
        assert r.elapsed_s >= 0.0
        if r.status == "docked":
            assert r.vina_kcal_mol is not None


# ---------------------------------------------------------------------------
# 15. main() writes both JSON outputs
# ---------------------------------------------------------------------------
def test_metrics_path_written(tmp_path: Path) -> None:
    """:func:`main` writes both the metrics + summary JSON files."""
    mod = _import_dock_module()
    pocket_dir = tmp_path / "wf30"
    pocket_id = "p1"
    (pocket_dir / pocket_id).mkdir(parents=True)
    (pocket_dir / pocket_id / "receptor.pdbqt").write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
    )
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    candidates_csv = tmp_path / "cands.csv"
    candidates_csv.write_text("name,smiles\nmethane,C\n")
    rc = mod.main(
        [
            "--pockets",
            pocket_id,
            "--methods",
            "baseline",
            "--seeds",
            "0",
            "--n-seeds",
            "1",
            "--candidates-csv",
            str(candidates_csv),
            "--pocket-dir",
            str(pocket_dir),
            "--out-dir",
            str(tmp_path),
            "--metrics-path",
            str(metrics_path),
            "--summary-path",
            str(summary_path),
            "--quiet",
        ]
    )
    assert rc == 0
    assert metrics_path.exists()
    assert summary_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["recipe"] == "wf30_dock_10pockets_se3sb"
    assert payload["n_evaluations"] == 1
    assert "per_evaluation" in payload
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["recipe"] == "wf30_dock_10pockets_se3sb_summary"
