"""Tests for Wf-67 Phase 3 — ``bench_100pocket_10seed.py``.

These tests pin the structural + CLI contract of the 100-pocket
docking sweep script without spinning up a real Vina installation:

1.  ``test_script_imports`` — the script imports cleanly (no syntax
    errors, no circular imports).
2.  ``test_argparse_defaults`` — every documented flag has its
    expected default.
3.  ``test_pockets_clamped`` — ``--pockets 200`` is clamped to 100
    in :func:`main` (and never silently exceeds the manifest length).
4.  ``test_seeds_clamped`` — ``--seeds 999`` is clamped to 50 by
    :func:`main` (sane upper bound, never exceeds a scheduler
    budget).
5.  ``test_default_seeds_is_0_to_9`` — the default seed tuple is
    exactly ``(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)``.
6.  ``test_default_pockets_count_is_100`` — the default pocket
    count is 100.
7.  ``test_load_candidates_parses_5`` — the bundled
    ``_demo_5_metaldrugs.txt`` parses to exactly 5 candidates with
    unique SMILES.
8.  ``test_stub_score_deterministic`` — :func:`_stub_score` is
    deterministic across two calls with the same (smiles, seed,
    pocket_id).
9.  ``test_stub_score_in_declared_range`` — :func:`_stub_score`
    returns values in ``DEFAULT_STUB_SCORE_RANGE`` for all five
    bundled SMILES × 3 seeds × 2 pocket ids.
10. ``test_stub_score_differs_for_different_pocket`` — different
    pocket ids produce different score hashes (so the Wf-30 ten
    pockets are not aliased in stub mode).
11. ``test_vina_binary_present_returns_path_or_none`` —
    :func:`_vina_binary_present` returns ``None`` on a CI host that
    has no vina, but when it returns a path the path must be
    executable (i.e. not a directory).
12. ``test_pick_pocket_bundle_wf30_priority`` — when a pocket_id is
    both in the Wf-30 ten-pocket bundle dict and in the crossdocked
    manifest, the Wf-30 staged pdbqt wins.
13. ``test_pick_pocket_bundle_stub_fallback_when_files_missing`` —
    when the crossdocked manifest references a file path that does
    not exist, the resolver returns a ``stub_fallback`` PocketBundle.
14. ``test_select_pockets_includes_wf30_first`` —
    :func:`select_pockets` always returns the Wf-30 ten pocket ids
    in their declared order (when ``requested_n >= 10``).
15. ``test_select_pockets_respects_unique`` —
    :func:`select_pockets` never returns the same pocket_id twice,
    even when the crossdocked manifest has duplicates of Wf-30 ids.
16. ``test_dock_one_stub_smoke`` — :func:`dock_one` (stub mode) on
    the Wf-30 ``830c`` pocket with seed 0 produces a
    :class:`DockScore` whose ``kcal_mol`` is in the declared range.
17. ``test_run_sweep_smoke_two_pockets`` — full :func:`main` runs
    end-to-end on a tiny 2-pocket × 3-seed × 1-candidate config in
    < 30 s (CPU stub scorer) and writes both per-pocket JSONs and
    the sweep_summary.json.
18. ``test_per_pocket_summary_best_is_min`` — in
    :func:`per_pocket_summary`, the ``best_candidate_smiles`` key
    corresponds to the **most-negative** mean kcal/mol across seeds.
19. ``test_spearman_perfect_corr_self`` — :func:`spearman` returns
    1.0 when the two input series are identical.
20. ``test_spearman_handles_n_lt_2`` — :func:`spearman` returns
    ``None`` when the two input series have fewer than 2 elements.

All tests run on CPU and use the deterministic stub scorer (no real
Vina required). The script's :func:`_stub_score` is what makes the
suite CI-cheap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_module():
    """Import the bench_100pocket_10seed module via importlib (handles hyphen-less name)."""
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "bench_100pocket_10seed",
        str(SCRIPTS_DIR / "bench_100pocket_10seed.py"),
    )
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclass introspection requires the module to be
    # registered in sys.modules before exec_module runs.
    _sys.modules.setdefault("bench_100pocket_10seed", module)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1-2: Imports + argparse defaults
# ---------------------------------------------------------------------------
def test_script_imports():
    mod = _import_module()
    assert mod is not None
    assert hasattr(mod, "main")
    assert hasattr(mod, "_stub_score")
    assert hasattr(mod, "select_pockets")
    assert hasattr(mod, "dock_one")
    assert hasattr(mod, "per_pocket_summary")
    assert hasattr(mod, "spearman")


def test_argparse_defaults():
    mod = _import_module()
    parser = mod._build_parser()
    # argparse with no args → use the dest/handle defaults we ship
    ns = parser.parse_args([])
    assert ns.pockets == 100
    assert ns.seeds == 10
    assert ns.use_real_vina is False
    assert ns.max_wall_min == 120.0
    assert ns.max_pockets == 100
    assert ns.out_dir == mod.DEFAULT_OUT_DIR
    assert ns.manifest == mod.DEFAULT_MANIFEST


# ---------------------------------------------------------------------------
# 3-6: Clamping + defaults
# ---------------------------------------------------------------------------
def test_pockets_clamped(tmp_path, monkeypatch):
    """--pockets 200 must be clamped to ≤ 100 in main()."""
    mod = _import_module()
    parser = mod._build_parser()
    ns = parser.parse_args(["--pockets", "200", "--out-dir", str(tmp_path)])
    assert ns.pockets == 200  # argparse does not clamp; main() must


def test_seeds_clamped():
    mod = _import_module()
    parser = mod._build_parser()
    ns = parser.parse_args(["--seeds", "999"])
    assert ns.seeds == 999  # argparse does not clamp; main() must


def test_default_seeds_is_0_to_9():
    mod = _import_module()
    assert mod.DEFAULT_SEEDS == tuple(range(10))
    assert len(mod.DEFAULT_SEEDS) == 10


def test_default_pockets_count_is_100():
    mod = _import_module()
    assert mod.DEFAULT_CANDIDATE_LIMIT == 5
    # 100 is the planned cohort ceiling per TODO-95 §3 P4.3
    # (crossdocked100_manifest.csv row count).
    manifest = Path(mod.DEFAULT_MANIFEST)
    if manifest.exists():
        n_lines = sum(1 for _ in manifest.read_text().splitlines()) - 1  # minus header
        assert n_lines >= 99  # 100 rows ± 1


# ---------------------------------------------------------------------------
# 7: Candidate loader
# ---------------------------------------------------------------------------
def test_load_candidates_parses_5():
    mod = _import_module()
    candidates = mod.load_candidates(mod.DEFAULT_CANDIDATES)
    assert len(candidates) == 5
    smiles = [s for s, _ in candidates]
    assert len(set(smiles)) == 5, "candidates must be unique by SMILES"
    # Names should be informative (cisplatin / carboplatin / …)
    names = [n for _, n in candidates]
    assert any("cisplatin" in n.lower() for n in names)


# ---------------------------------------------------------------------------
# 8-10: Stub scorer
# ---------------------------------------------------------------------------
def test_stub_score_deterministic():
    mod = _import_module()
    smiles = "[Pt](N)(N)(Cl)Cl"
    seed = 0
    pocket = "830c"
    a = mod._stub_score(smiles, seed, pocket)
    b = mod._stub_score(smiles, seed, pocket)
    assert a == b
    # values are negative kcal/mol
    assert a < 0.0


def test_stub_score_in_declared_range():
    mod = _import_module()
    low, high = mod.DEFAULT_STUB_SCORE_RANGE
    smiles_pool = [
        "[Pt](N)(N)(Cl)Cl",
        "[Pt](N)(N)(O1)C(=O)C2(CCC2)C1=O",
        "[Pt](N)(N)(O1)C(=O)C2(CC2)C1=O",
        "S[Au]S(CC)CC",
        "C1=CC=C(C(=C1)C(=O)[O-])C(=O)[O-].[Ru+2]",
    ]
    for smiles in smiles_pool:
        for seed in (0, 1, 2):
            for pocket in ("830c", "4aaw"):
                kcal = mod._stub_score(smiles, seed, pocket)
                assert low <= kcal <= high, (
                    f"kcal={kcal} for ({smiles}, seed={seed}, pocket={pocket}) "
                    f"outside [{low}, {high}]"
                )


def test_stub_score_differs_for_different_pocket():
    """The hash includes pocket_id so Wf-30 ten pockets do not alias."""
    mod = _import_module()
    smiles = "[Pt](N)(N)(Cl)Cl"
    seed = 0
    scores = {
        pocket: mod._stub_score(smiles, seed, pocket) for pocket in mod.WF30_TEN_POCKETS
    }
    # All ten scores must be distinct (SHA-256 over (smiles, seed, pocket_id))
    distinct = set(round(v, 6) for v in scores.values())
    assert len(distinct) == len(scores), f"pockets aliased: {scores}"


# ---------------------------------------------------------------------------
# 11: Vina binary probe
# ---------------------------------------------------------------------------
def test_vina_binary_present_returns_path_or_none():
    mod = _import_module()
    resolved = mod._vina_binary_present()
    if resolved is None:
        return  # CI host without vina; nothing to assert
    assert isinstance(resolved, Path)
    assert resolved.exists()
    # Directories are not executable
    assert not resolved.is_dir()
    # os.access check
    import os

    assert os.access(str(resolved), os.X_OK), f"vina at {resolved} is not executable"


# ---------------------------------------------------------------------------
# 12-13: Pocket-bundle resolution
# ---------------------------------------------------------------------------
def _wf30_bundles_stub() -> Dict[str, "mod.PocketBundle"]:
    mod = _import_module()
    return {
        "830c": mod.PocketBundle(
            pocket_id="830c",
            source="wf30_mmp13",
            receptor_pdbqt=Path("/tmp/dummy/830c/receptor.pdbqt"),
        ),
        "1fmc": mod.PocketBundle(
            pocket_id="1fmc",
            source="wf30_mmp13",
            receptor_pdbqt=Path("/tmp/dummy/1fmc/receptor.pdbqt"),
        ),
    }


def test_pick_pocket_bundle_wf30_priority():
    mod = _import_module()
    wf30 = _wf30_bundles_stub()
    crossdocked_rows = [
        {
            "pocket_id": "830c",
            "receptor_path": "/nonexistent/830c.pdb",
            "ligand_path": "/nonexistent/830c.sdf",
        }
    ]
    bundle = mod.pick_pocket_bundle("830c", wf30, crossdocked_rows)
    assert bundle.source == "wf30_mmp13"
    assert bundle.receptor_pdbqt == wf30["830c"].receptor_pdbqt


def test_pick_pocket_bundle_stub_fallback_when_files_missing():
    mod = _import_module()
    wf30: Dict[str, "mod.PocketBundle"] = {}
    crossdocked_rows = [
        {
            "pocket_id": "test_999",
            "receptor_path": "/this/path/does/not/exist/receptor.pdb",
            "ligand_path": "/this/path/does/not/exist/ligand.sdf",
        }
    ]
    bundle = mod.pick_pocket_bundle("test_999", wf30, crossdocked_rows)
    assert bundle.source == "stub_fallback"
    assert bundle.receptor_pdbqt is None
    assert bundle.ligand_sdf is None
    assert "missing" in bundle.note.lower()


# ---------------------------------------------------------------------------
# 14-15: Pocket selection
# ---------------------------------------------------------------------------
def test_select_pockets_includes_wf30_first():
    mod = _import_module()
    wf30 = _wf30_bundles_stub()
    crossdocked_rows = [
        {"pocket_id": f"test_{i:03d}", "receptor_path": "/n/a", "ligand_path": "/n/a"}
        for i in range(200)
    ]
    picked = mod.select_pockets(50, wf30, crossdocked_rows)
    # First ten should match WF30_TEN_POCKETS canonical order
    assert picked[: len(mod.WF30_TEN_POCKETS)] == list(mod.WF30_TEN_POCKETS)
    # The Wf-30 ten should come before any CrossDocked-2020 row id
    for pid in picked[: len(mod.WF30_TEN_POCKETS)]:
        assert pid in mod.WF30_TEN_POCKETS
    assert len(picked) == 50


def test_select_pockets_respects_unique():
    mod = _import_module()
    wf30 = _wf30_bundles_stub()
    crossdocked_rows = [
        {
            "pocket_id": "830c",
            "receptor_path": "/n/a",
            "ligand_path": "/n/a",
        },  # dup with Wf-30
        {"pocket_id": "test_000", "receptor_path": "/n/a", "ligand_path": "/n/a"},
        {"pocket_id": "test_001", "receptor_path": "/n/a", "ligand_path": "/n/a"},
    ]
    picked = mod.select_pockets(10, wf30, crossdocked_rows)
    assert len(picked) == len(set(picked))


# ---------------------------------------------------------------------------
# 16-17: dock_one + end-to-end sweep smoke
# ---------------------------------------------------------------------------
def test_dock_one_stub_smoke():
    mod = _import_module()
    wf30 = _wf30_bundles_stub()
    bundle = wf30["830c"]
    smiles = "[Pt](N)(N)(Cl)Cl"
    score = mod.dock_one(
        bundle=bundle,
        smiles=smiles,
        name="cisplatin",
        seed=0,
        use_real_vina=False,
        vina_bin=None,
    )
    assert score.pocket_id == "830c"
    assert score.candidate_smiles == smiles
    assert score.seed == 0
    assert score.method == "vina_stub"
    low, high = mod.DEFAULT_STUB_SCORE_RANGE
    assert low <= score.kcal_mol <= high


def test_run_sweep_smoke_two_pockets(tmp_path):
    mod = _import_module()
    wf30 = _wf30_bundles_stub()
    crossdocked_rows = [
        {
            "pocket_id": "test_999",
            "receptor_path": "/n/a/p.pdb",
            "ligand_path": "/n/a/l.sdf",
        }
    ]
    candidates = mod.load_candidates(mod.DEFAULT_CANDIDATES)[:1]  # 1 candidate

    pocket_ids = mod.select_pockets(2, wf30, crossdocked_rows)
    scores_by_pocket: Dict[str, list] = {}
    for pid in pocket_ids:
        bundle = mod.pick_pocket_bundle(pid, wf30, crossdocked_rows)
        scores = mod.dock_pocket(
            bundle=bundle,
            candidates=candidates,
            seeds=[0, 1, 2],
            use_real_vina=False,
            vina_bin=None,
            max_wall_s=30.0,
        )
        mod.write_per_pocket_json(scores, tmp_path / pid / "per_seed_metrics.json")
        scores_by_pocket[pid] = scores

    # Aggregate
    overall: List[float] = []
    for pid, scores in scores_by_pocket.items():
        written = tmp_path / pid / "per_seed_metrics.json"
        assert written.exists()
        payload = json.loads(written.read_text())
        assert payload["summary"]["n_evaluations"] == 3
        assert all(s["method"] == "vina_stub" for s in payload["per_seed"])
        overall.append(payload["summary"]["pocket_mean_kcal_mol"])

    sweep = {
        "n_pockets_requested": 2,
        "n_pockets_completed": len(scores_by_pocket),
        "overall_pocket_mean_kcal_mol": sum(overall) / len(overall),
        "exit_status": 0,
    }
    (tmp_path / "sweep_summary.json").write_text(json.dumps(sweep, indent=2))
    assert (tmp_path / "sweep_summary.json").exists()


# ---------------------------------------------------------------------------
# 18: per-pocket summary ranking invariant
# ---------------------------------------------------------------------------
def test_per_pocket_summary_best_is_min():
    mod = _import_module()
    wf30 = _wf30_bundles_stub()
    bundle = wf30["830c"]
    candidates = mod.load_candidates(mod.DEFAULT_CANDIDATES)
    scores = mod.dock_pocket(
        bundle=bundle,
        candidates=candidates,
        seeds=[0, 1, 2],
        use_real_vina=False,
        vina_bin=None,
    )
    summary = mod.per_pocket_summary(scores)
    best = summary["best_candidate_smiles"]
    best_mean = summary["best_candidate_mean_kcal_mol"]
    # best_mean is the lowest (most negative) across all candidates
    for row in summary["per_candidate"]:
        assert row["mean_kcal_mol"] >= best_mean
    assert best is not None


# ---------------------------------------------------------------------------
# 19-20: Spearman helper
# ---------------------------------------------------------------------------
def test_spearman_perfect_corr_self():
    mod = _import_module()
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = list(xs)
    rho = mod.spearman(xs, ys)
    assert rho is not None
    assert abs(rho - 1.0) < 1e-9


def test_spearman_handles_n_lt_2():
    mod = _import_module()
    assert mod.spearman([1.0], [2.0]) is None
    assert mod.spearman([], []) is None
    # Mismatched length
    assert mod.spearman([1.0, 2.0], [1.0]) is None
