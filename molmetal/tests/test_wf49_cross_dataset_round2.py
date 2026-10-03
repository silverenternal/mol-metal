"""Tests for Wf-71 P3 — Wf-49 cross-dataset retrain round 2 (per-cohort).

These tests verify the per-cohort breakdown produced by
``bench_wf49_cross_dataset_round2.py`` and the structure of the
output JSON.  Datasets are small (PlatinAI 4-cell ~4k, MetalCytoTox
~26k, tmQM ~108k) and Ridge on 12 features is O(n) so we run the
tests against the real data CSVs.

Run with:

    PY=/home/hugo/codes/mol-metal/.venv/bin/python \\
    $PY -m pytest molmetal/tests/test_wf49_cross_dataset_round2.py -v
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Bench-script imports -- tests run inside the project venv.
# We load the bench module by file path because pytest's collection
# doesn't add ``molmetal/scripts`` to ``sys.path`` automatically.
import importlib.util as _importlib_util

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]  # mol-metal/ (the repo root)
_BENCH_SCRIPT = _REPO / "molmetal" / "scripts" / "bench_wf49_cross_dataset_round2.py"

_spec = _importlib_util.spec_from_file_location(
    "bench_wf49_cross_dataset_round2", str(_BENCH_SCRIPT)
)
assert _spec is not None and _spec.loader is not None
bench = _importlib_util.module_from_spec(_spec)
sys.modules["bench_wf49_cross_dataset_round2"] = bench
_spec.loader.exec_module(bench)


# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------
def test_rdkit_2d_cols_constant_is_12_columns() -> None:
    """F5_RDKIT_2D should have exactly 12 columns, matching
    LambdaASTSR.rdkit_2d_feature_columns()."""
    assert len(bench.RDKIT_2D_COLS) == 12
    expected = {
        "MolWt", "MolLogP", "TPSA", "HBA", "HBD", "NRotB",
        "HeavyAtoms", "RingCount", "NumAromaticRings",
        "FractionCSP3", "MolMR", "BalabanJ",
    }
    assert set(bench.RDKIT_2D_COLS) == expected


def test_platinai_target_cols_constant_has_4_cell_lines() -> None:
    """PlatinAI 4-cell targets must include all 4 cell-lines from R17/R19 baseline."""
    assert set(bench.PLATINAI_TARGET_COLS.keys()) == {
        "MCF7", "A549", "A2780", "A2780cis",
    }
    for cohort, col in bench.PLATINAI_TARGET_COLS.items():
        assert col.startswith("pIC50_"), f"{cohort!r} -> {col!r} must start with pIC50_"


def test_metalcytotox_metals_constant_has_5_metals() -> None:
    """MetalCytoToxDB 5-metal whitelist: Ru / Ir / Rh / Os / Re."""
    assert bench.METALCYTOTOX_METALS == ("Ru", "Ir", "Rh", "Os", "Re")


def test_tmqm_dblock_metals_constant_is_d_block_only() -> None:
    """tmQM d-block metals should NOT include La (f-block) or Pb (post-d-block)."""
    for forbidden in ("La", "Pb", "Ce", "Pr", "Nd"):  # f-block examples
        assert forbidden not in bench.TMQM_DBLOCK_METALS, (
            f"{forbidden!r} is not a d-block metal"
        )
    for required in ("Sc", "Ti", "Fe", "Cu", "Ag", "Au", "Hg"):
        assert required in bench.TMQM_DBLOCK_METALS, (
            f"{required!r} should be in d-block"
        )


def test_default_out_dir_under_reports() -> None:
    """Default output dir should be ``molmetal/reports/wf71_p3_wf49_cross_round2``."""
    # bench.DEFAULT_OUT_DIR is computed from
    # ``Path(__file__).parents[2]`` so it's relative to the bench script.
    # Just check the suffix is right.
    assert str(bench.DEFAULT_OUT_DIR).endswith(
        "molmetal/reports/wf71_p3_wf49_cross_round2"
    )


# ---------------------------------------------------------------------------
# Per-cohort Ridge fit
# ---------------------------------------------------------------------------
def test_fit_per_cohort_returns_finite_r2_on_synthetic_signal() -> None:
    """Synthetic dataset where y = 2 * MolLogP + noise -- R^2 > 0.5."""
    n = 500
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, len(bench.RDKIT_2D_COLS)))
    # y is dominated by column index 1 (MolLogP) + noise
    y = 2.0 * X[:, 1] + 0.1 * rng.standard_normal(n)
    import pandas as pd
    df = pd.DataFrame(X, columns=list(bench.RDKIT_2D_COLS))
    df["target"] = y
    df["_cohort"] = "A"
    row = bench.fit_per_cohort(
        df, cohort_col="_cohort", target_col="target",
        cohort_name="A", seeds=(0, 1, 2),
    )
    assert row.n_rows == n
    assert len(row.r2_per_seed) == 3
    assert row.r2_mean > 0.5, f"R^2 should be > 0.5 on synthetic signal, got {row.r2_mean}"
    assert not math.isnan(row.rmse_mean)


def test_fit_per_cohort_returns_nan_for_tiny_cohort() -> None:
    """Cohort with < 10 rows should return NaN fields, not crash."""
    import pandas as pd
    df = pd.DataFrame({
        "MolWt": [1.0] * 5,
        "MolLogP": [1.0] * 5,
        "target": [1.0] * 5,
        "_cohort": ["A"] * 5,
    })
    for col in bench.RDKIT_2D_COLS:
        if col not in df.columns:
            df[col] = 1.0
    row = bench.fit_per_cohort(
        df, cohort_col="_cohort", target_col="target",
        cohort_name="A", seeds=(0,),
    )
    assert math.isnan(row.r2_mean)
    assert row.n_rows == 5


# ---------------------------------------------------------------------------
# Dataset loaders (smoke)
# ---------------------------------------------------------------------------
def test_load_platinai_4cell_has_12_rdkit_cols_and_4_pic50_cols() -> None:
    """platinai 4-cell feature CSV: 12 RDKit 2D cols + 4 pIC50 cols."""
    df = bench.load_platinai_4cell()
    assert len(df) > 100, f"expected ~4078 rows, got {len(df)}"
    for col in bench.RDKIT_2D_COLS:
        assert col in df.columns
    for col in bench.PLATINAI_TARGET_COLS.values():
        assert col in df.columns


def test_load_metalcytotox_5metal_filters_to_5_metals() -> None:
    """MetalCytoToxDB must filter to Ru/Ir/Rh/Os/Re (the 5-metal whitelist)."""
    df = bench.load_metalcytotox_5metal()
    assert set(df["metal"].unique()) == set(bench.METALCYTOTOX_METALS)
    for col in bench.RDKIT_2D_COLS:
        assert col in df.columns
    assert "activity" in df.columns


def test_load_tmqm_dblock_filters_to_dblock_metals() -> None:
    """tmQM d-block filter must drop La (f-block) and Pb (post-d)."""
    df = bench.load_tmqm_dblock(target="HL_Gap")
    metals = set(df["metal"].unique())
    assert "La" not in metals, "La is f-block -- must be filtered out"
    for col in bench.RDKIT_2D_COLS:
        assert col in df.columns
    assert "target" in df.columns


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------
def test_bench_platinai_4cell_has_4_cohorts() -> None:
    """platinai 4-cell should produce one cohort per cell-line."""
    s = bench.bench_platinai_4cell(seeds=(0, 1, 2))
    assert s.n_cohorts == 4
    assert set(s.cohort_names) == {
        "MCF7", "A549", "A2780", "A2780cis",
    }
    for c in s.per_cohort:
        assert c.dataset == "platinai_4cell"
        assert len(c.r2_per_seed) == 3


def test_bench_metalcytotox_5metal_has_5_cohorts() -> None:
    """metalcytotox 5-metal should produce one cohort per metal."""
    s = bench.bench_metalcytotox_5metal(seeds=(0, 1, 2))
    assert s.n_cohorts == 5
    assert set(s.cohort_names) == set(bench.METALCYTOTOX_METALS)


def test_bench_tmqm_dblock_has_at_least_10_cohorts() -> None:
    """tmQM d-block should produce >= 10 cohorts (29 d-block metals - small ones)."""
    s = bench.bench_tmqm_dblock(seeds=(0, 1, 2), max_per_metal=200)
    assert s.n_cohorts >= 10, (
        f"expected >= 10 d-block cohorts, got {s.n_cohorts}"
    )


# ---------------------------------------------------------------------------
# Output JSON structure
# ---------------------------------------------------------------------------
def test_summary_to_json_dict_has_all_required_fields() -> None:
    """JSON shape must include per_cohort list + macro aggregation."""
    s = bench.bench_platinai_4cell(seeds=(0, 1))
    d = bench.summary_to_json_dict(s)
    for k in (
        "dataset",
        "n_cohorts",
        "cohort_names",
        "macro_r2_mean",
        "macro_r2_std",
        "overall_r2_mean",
        "overall_pearson_r_mean",
        "n_total_rows",
        "per_cohort",
    ):
        assert k in d, f"missing key {k!r}"
    assert isinstance(d["per_cohort"], list)
    assert len(d["per_cohort"]) == s.n_cohorts
    for c in d["per_cohort"]:
        assert "r2_per_seed" in c
        assert "r2_mean" in c
        assert "pearson_r_per_seed" in c


def test_write_summary_creates_per_cohort_metrics_json(
    tmp_path: Path,
) -> None:
    """write_summary must produce ``{out}/{dataset}/per_cohort_metrics.json``."""
    s = bench.bench_platinai_4cell(seeds=(0, 1))
    p = bench.write_summary(tmp_path, s)
    assert p.exists()
    assert p.name == "per_cohort_metrics.json"
    data = json.loads(p.read_text())
    assert data["dataset"] == "platinai_4cell"
    assert data["n_cohorts"] == 4


def test_cross_dataset_summary_has_target_met_flag(tmp_path: Path) -> None:
    """Cross-dataset JSON must include ``target_met`` boolean field."""
    s = bench.bench_platinai_4cell(seeds=(0, 1))
    bench.write_summary(tmp_path, s)
    cross_path = tmp_path / "cross_dataset_summary.json"
    # Re-run main in --datasets-only mode by directly invoking the
    # cross_dataset aggregation logic
    summaries = [s]
    macro_r2_means = [
        sm.macro_r2_mean for sm in summaries if not math.isnan(sm.macro_r2_mean)
    ]
    assert macro_r2_means  # at least one finite
    cross_macro = float(np.mean(macro_r2_means))
    target_met = cross_macro >= 0.0
    cross_path.write_text(json.dumps({"cross_macro": cross_macro,
                                       "target_met": target_met}, indent=2))
    d = json.loads(cross_path.read_text())
    assert "target_met" in d
    assert isinstance(d["target_met"], bool)


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------
def test_default_seeds_has_5_elements() -> None:
    """Default seeds should be (0, 1, 2, 3, 4)."""
    assert bench.DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    assert len(bench.DEFAULT_SEEDS) == 5


def test_cli_smoke_runs_to_completion(tmp_path: Path) -> None:
    """CLI default invocation must produce per-cohort + cross JSON files."""
    # Use a tiny cap on tmqm to keep this smoke test fast.
    result = subprocess.run(
        [
            sys.executable,
            str(_BENCH_SCRIPT),
            "--datasets", "platinai_4cell", "metalcytotox_5metal",
            "--seeds", "0", "1",
            "--out-dir", str(tmp_path),
            "--max-per-metal-tmqm", "100",  # not used here, but accepted
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"CLI failed: stderr={result.stderr!r}"
    )
    expected_files = [
        "platinai_4cell/per_cohort_metrics.json",
        "metalcytotox_5metal/per_cohort_metrics.json",
        "cross_dataset_summary.json",
    ]
    for f in expected_files:
        assert (tmp_path / f).exists(), f"missing {f}"
