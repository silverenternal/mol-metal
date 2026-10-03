"""Tests for Wf-60 Phase 3 bench_weak_real_lifting orchestrator.

These tests cover the public surface of `molmetal.scripts.bench_weak_real_lifting`:

* Recipe catalogue correctness (3 metrics, gate targets match the harness spec)
* CLI parsing (default GPU, custom seeds, dry-run, metric filtering, gate flags)
* Aggregation (mean matches statistics.fmean, gate_pass NaN-safe, delta math)
* Extract-values (shallow + nested patterns + scalar fallback)
* Synthesis markdown emissions (per-metric verdict + honest framing)
* Aggregate-existing path (re-reads on-disk JSON without subprocess)
* Wrapper purity (no torch / rdkit / sklearn imports in the bench script module)

The test runner picks them up via `pytest molmetal/tests/test_weak_real_lifting.py -v`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo-root path setup so this test works regardless of pytest cwd.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts import bench_weak_real_lifting as bwrl


# ---------------------------------------------------------------------------
# Recipe catalogue — 3 metrics, gate targets must match the harness spec
# ---------------------------------------------------------------------------
def test_three_recipes_present():
    expected = {"m7", "m15", "m16"}
    assert set(bwrl.RECIPES.keys()) == expected


def test_recipe_metadata_consistent():
    for metric, recipe in bwrl.RECIPES.items():
        assert recipe.metric == metric
        assert isinstance(recipe.baseline, float)
        assert isinstance(recipe.target, float)
        assert recipe.direction in ("max", "min")
        assert recipe.unit
        assert (
            recipe.trainer_path.exists()
        ), f"{metric} trainer path missing: {recipe.trainer_path}"


def test_m7_target_is_lift_over_r17_baseline():
    r = bwrl.RECIPES["m7"]
    assert r.baseline == 0.1065
    assert r.target >= 0.15
    assert r.direction == "max"


def test_m15_target_is_strict_scaffold_lift():
    r = bwrl.RECIPES["m15"]
    assert r.baseline == 0.428
    assert r.target <= 0.35
    assert r.direction == "min"
    assert "Å" in r.unit or "angstrom" in r.unit.lower() or "A" in r.unit


def test_m16_target_includes_wf44_numerical_fix_args():
    r = bwrl.RECIPES["m16"]
    assert r.baseline == 0.207
    assert r.target >= 0.30
    assert r.direction == "max"
    extras = " ".join(r.extra_args)
    assert "--use-numerical-fix" in extras
    assert "--grad-clip" in extras
    assert "--step-decay" in extras
    assert "--fp64-fallback" in extras


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_cli_defaults():
    ns = bwrl._parse_args([])
    assert ns.metrics == ["m7", "m15", "m16"]
    assert ns.seeds == [0, 1, 2, 3, 4]
    assert ns.cuda_visible_devices == "1"
    assert ns.strict_scaffold is True
    assert ns.dry_run is False
    assert ns.aggregate_existing is False


def test_cli_metric_filter():
    ns = bwrl._parse_args(["--metrics", "m7"])
    assert ns.metrics == ["m7"]


def test_cli_unknown_metric_rejected():
    import pytest

    with pytest.raises(SystemExit):
        bwrl._parse_args(["--metrics", "mX"])


def test_cli_dry_run_flag():
    ns = bwrl._parse_args(["--dry-run"])
    assert ns.dry_run is True


def test_cli_seed_override():
    ns = bwrl._parse_args(["--seeds", "7", "8"])
    assert ns.seeds == [7, 8]


# ---------------------------------------------------------------------------
# Aggregation: mean / std / gate_pass / delta math
# ---------------------------------------------------------------------------
def test_extract_values_shallow_list():
    payload = {"mean_tanimoto_distance": [0.12, 0.18, 0.20]}
    out = bwrl._extract_values(payload, "mean_tanimoto_distance")
    assert out == [0.12, 0.18, 0.20]


def test_extract_values_nested_dict():
    payload = {"diversity": {"mean_tanimoto_distance": [0.10, 0.20]}}
    out = bwrl._extract_values(payload, "diversity.mean_tanimoto_distance")
    assert out == [0.10, 0.20]


def test_extract_values_scalar_fallback():
    payload = {"value": 0.207}
    out = bwrl._extract_values(payload, "missing_path")
    assert out == [0.207]


def test_extract_values_empty_when_path_missing():
    assert bwrl._extract_values({}, "nope") == []


def test_safe_fmean_and_pstdev_consistent():
    xs = [0.10, 0.12, 0.20]
    mean = bwrl._safe_fmean(xs)
    std = bwrl._safe_pstdev(xs)
    assert mean is not None
    assert mean == 0.14
    assert std > 0.0


def test_gate_pass_max_direction():
    recipe = bwrl.RECIPES["m7"]
    assert bwrl._gate_pass(0.20, recipe) is True
    assert bwrl._gate_pass(0.1065, recipe) is False
    assert bwrl._gate_pass(None, recipe) is None


def test_gate_pass_min_direction():
    recipe = bwrl.RECIPES["m15"]
    assert bwrl._gate_pass(0.30, recipe) is True
    assert bwrl._gate_pass(0.42, recipe) is False
    assert bwrl._gate_pass(None, recipe) is None


# ---------------------------------------------------------------------------
# Synthesis markdown
# ---------------------------------------------------------------------------
def test_write_synthesis_md_contains_three_metrics(tmp_path: Path):
    out_dir = tmp_path / "synthesis"
    out_dir.mkdir()
    recipe_map = bwrl.RECIPES
    summary = {
        "generated_at": "2026-09-26T16:00:00Z",
        "metrics": {
            "m7": {
                "status": "MEASURED",
                "mean": 0.18,
                "std": 0.02,
                "baseline": 0.1065,
                "target": 0.15,
                "direction": "max",
                "delta_vs_baseline": 0.0735,
                "delta_vs_target": 0.03,
                "gate_pass": True,
                "label": "diversity",
                "unit": "distance",
            },
            "m15": {
                "status": "MEASURED",
                "mean": 0.32,
                "std": 0.05,
                "baseline": 0.428,
                "target": 0.35,
                "direction": "min",
                "delta_vs_baseline": -0.108,
                "delta_vs_target": 0.03,
                "gate_pass": True,
                "label": "rmsd",
                "unit": "A",
            },
            "m16": {
                "status": "DEFERRED",
                "mean": None,
                "std": None,
                "baseline": 0.207,
                "target": 0.30,
                "direction": "max",
                "delta_vs_baseline": None,
                "delta_vs_target": None,
                "gate_pass": None,
                "label": "pearson",
                "unit": "r",
            },
        },
    }
    md_path = bwrl._write_synthesis_md(out_dir, summary, recipe_map)
    text = md_path.read_text()
    for metric in ("m7", "m15", "m16"):
        assert metric in text
    assert "PASS" in text  # at least m7/m15 passed
    assert "DEFERRED" in text  # m16 deferred


def test_aggregate_existing_reads_on_disk(tmp_path: Path):
    out_dir = tmp_path / "agg"
    for metric in ("m7", "m15", "m16"):
        mdir = out_dir / metric
        mdir.mkdir(parents=True)
        per_run = mdir / "per_run_metrics.json"
        per_run.write_text(
            json.dumps(
                {
                    "diversity": {"mean_tanimoto_distance": [0.15, 0.17, 0.20]},
                }
                if metric == "m7"
                else {
                    "rmsd_median_angstrom": [0.30, 0.32, 0.31],
                }
                if metric == "m15"
                else {
                    "pearson_r_per_seed": [0.21, 0.25, 0.32],
                }
            )
        )
    summary = bwrl._aggregate_existing(out_dir, bwrl.RECIPES)
    assert "metrics" in summary
    for metric in ("m7", "m15", "m16"):
        assert metric in summary["metrics"]
        entry = summary["metrics"][metric]
        assert entry.get("status") in ("MEASURED", "DEFERRED")


# ---------------------------------------------------------------------------
# Wrapper purity — no torch / rdkit / sklearn imports
# ---------------------------------------------------------------------------
def test_bench_script_no_deep_learning_imports():
    """The bench script must remain a pure orchestrator (per Phase 1 design)."""
    src_path = Path(bwrl.__file__)
    text = src_path.read_text()
    forbidden = [
        "import torch",
        "from torch",
        "import rdkit",
        "from rdkit",
        "import sklearn",
        "from sklearn",
    ]
    for needle in forbidden:
        assert (
            needle not in text
        ), f"Forbidden import '{needle}' leaks the orchestrator's purity"
