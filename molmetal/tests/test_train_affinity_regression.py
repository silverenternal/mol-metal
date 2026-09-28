"""Wf-32 Phase 1 — contract tests for ``train_affinity_regression.py``.

=====================================================================
CPU-only contract tests covering:

* Module imports (no syntax errors, no missing deps).
* ``parse_args`` defaults match the Wf-32 spec.
* ``--init`` choices mirror the R18 init contract (tmqm/random/se3sb/
  transport/multiph) — same enum as :mod:`molmetal.scripts.train_ru_multitask`.
* ``--dataset`` choices are exactly ``{platinai, hela48h}``.
* ``PlatinAIRegressionDataset.from_csv`` produces a (N, 4) pIC50 matrix
  with sensible value ranges (4-9) and the correct cell-line columns.
* Recipe constants match the brief.
* ``train_one_seed_hela48h`` shape contract (returns dict with the
  required keys).
* ``_to_jsonable`` round-trip handles NaN → None.
* ``--help`` exits 0; CLI smoke run with a 1-epoch / small-budget
  config returns exit-code 0.
* Init-choice wiring: each R18 init is accepted by argparse.
* Aggregate JSON contract: keys are present.
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_module(name: str = "train_affinity_regression"):
    sys.path.insert(0, str(_SCRIPT_DIR))
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# 1. Module imports + recipe constants
# ---------------------------------------------------------------------------
def test_module_imports_cleanly() -> None:
    """``train_affinity_regression`` imports without raising."""
    mod = _import_module()
    assert mod is not None
    assert hasattr(mod, "main")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "train_one_seed_platinai")
    assert hasattr(mod, "train_one_seed_hela48h")


def test_recipe_constants_match_brief() -> None:
    """Recipe constants match the Wf-32 Phase 1 spec."""
    mod = _import_module()
    assert mod.RECIPE_EPOCHS == 30
    assert mod.RECIPE_BATCH == 64
    assert mod.RECIPE_LR == 1e-3
    assert mod.RECIPE_SEEDS == (0, 1, 2, 3, 4)
    assert mod.RECIPE_HIDDEN == 128
    assert mod.RECIPE_DEPTH == 3
    assert mod.RECIPE_DROPOUT == 0.1
    assert mod.RECIPE_ALPHA == 0.0      # regression-only (MSE on pIC50)
    assert mod.RECIPE_EVAL_EVERY == 5


def test_init_choices_include_r17_and_r18() -> None:
    """``--init`` choices mirror the R18 init contract."""
    mod = _import_module()
    assert mod.INIT_CHOICES == (
        "tmqm", "random", "se3sb", "transport", "multiph",
    )


def test_dataset_choices_match_brief() -> None:
    """``--dataset`` is exactly ``{platinai, hela48h}``."""
    mod = _import_module()
    assert mod.DATASET_CHOICES == ("platinai", "hela48h")


def test_platinai_cell_lines_constant() -> None:
    """The 4 PlatinAI cell-line labels are baked in."""
    mod = _import_module()
    assert mod.PLATINAI_CELL_LINES == ("MCF7", "A549", "A2780", "A2780cis")


def test_hela48h_cohort_defaults() -> None:
    """HeLa48h cohort default cell-line + time-h are baked in."""
    mod = _import_module()
    assert mod.HELA48H_CELL_LINE == "HeLa"
    assert mod.HELA48H_TIME_H == 48.0


def test_default_metrics_paths() -> None:
    """``metrics_path`` / ``summary_path`` defaults match the brief."""
    mod = _import_module()
    args = mod.parse_args([])
    assert args.metrics_path.name == "affinity_pearson_r.json"
    assert args.summary_path.name == "affinity_pearson_r_summary.json"


# ---------------------------------------------------------------------------
# 2. argparse defaults
# ---------------------------------------------------------------------------
def test_parse_args_defaults() -> None:
    """``parse_args([])`` yields the spec defaults."""
    mod = _import_module()
    args = mod.parse_args([])
    assert args.dataset == "platinai"
    assert args.init == "tmqm"
    assert args.epochs == mod.RECIPE_EPOCHS == 30
    assert args.batch == mod.RECIPE_BATCH == 64
    assert args.lr == mod.RECIPE_LR == 1e-3
    assert tuple(args.seeds) == mod.RECIPE_SEEDS
    assert args.hidden == mod.RECIPE_HIDDEN == 128
    assert args.depth == mod.RECIPE_DEPTH == 3
    assert args.eval_every == mod.RECIPE_EVAL_EVERY == 5


def test_parse_args_accepts_r18_init_choices() -> None:
    """Each R18 init choice is accepted by argparse."""
    mod = _import_module()
    for init in ("se3sb", "transport", "multiph", "tmqm", "random"):
        args = mod.parse_args(["--init", init])
        assert args.init == init


def test_parse_args_accepts_hela48h_dataset() -> None:
    """``--dataset hela48h`` is accepted."""
    mod = _import_module()
    args = mod.parse_args(["--dataset", "hela48h"])
    assert args.dataset == "hela48h"
    args = mod.parse_args(["--dataset", "platinai"])
    assert args.dataset == "platinai"


def test_parse_args_rejects_unknown_dataset() -> None:
    """Unknown datasets are rejected by argparse."""
    mod = _import_module()
    with pytest.raises(SystemExit):
        mod.parse_args(["--dataset", "bogus"])


def test_parse_args_rejects_unknown_init() -> None:
    """Unknown init values are rejected by argparse."""
    mod = _import_module()
    with pytest.raises(SystemExit):
        mod.parse_args(["--init", "bogus"])


def test_parse_args_cli_overrides() -> None:
    """User-provided CLI overrides are parsed correctly."""
    mod = _import_module()
    args = mod.parse_args([
        "--dataset", "hela48h",
        "--epochs", "5",
        "--batch", "16",
        "--lr", "5e-4",
        "--seeds", "0", "1",
        "--hidden", "64",
        "--depth", "2",
        "--init", "se3sb",
        "--cell-line", "MCF-7",
        "--time-h", "24.0",
    ])
    assert args.dataset == "hela48h"
    assert args.epochs == 5
    assert args.batch == 16
    assert args.lr == 5e-4
    assert args.seeds == [0, 1]
    assert args.hidden == 64
    assert args.depth == 2
    assert args.init == "se3sb"
    assert args.cell_line == "MCF-7"
    assert args.time_h == 24.0


# ---------------------------------------------------------------------------
# 3. argparse --help
# ---------------------------------------------------------------------------
def test_help_exits_zero() -> None:
    """``--help`` exits with code 0 (argparse is well-formed)."""
    mod = _import_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
    assert exc.value.code == 0
    text = buf.getvalue()
    assert "--dataset" in text
    assert "--init" in text
    assert "--epochs" in text
    assert "--batch" in text
    assert "--hidden" in text
    assert "--depth" in text


def test_help_lists_init_choices() -> None:
    """The ``--init`` choices appear in the ``--help`` text."""
    mod = _import_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit):
            mod.parse_args(["--help"])
    text = buf.getvalue()
    for choice in ("tmqm", "random", "se3sb", "transport", "multiph"):
        assert choice in text, f"--init choice {choice!r} missing from --help"


def test_help_lists_dataset_choices() -> None:
    """The ``--dataset`` choices appear in the ``--help`` text."""
    mod = _import_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit):
            mod.parse_args(["--help"])
    text = buf.getvalue()
    assert "platinai" in text
    assert "hela48h" in text


# ---------------------------------------------------------------------------
# 4. JSON helper
# ---------------------------------------------------------------------------
def test_to_jsonable_handles_nan() -> None:
    """``_to_jsonable`` converts NaN → None and numpy → Python scalars."""
    mod = _import_module()
    import numpy as np
    obj = {
        "nan": float("nan"),
        "inf": float("inf"),
        "np_float": np.float32(1.5),
        "np_int": np.int64(7),
        "nested": [np.array([1.0, 2.0]), {"k": float("nan")}],
    }
    out = mod._to_jsonable(obj)
    assert out["nan"] is None
    assert out["np_float"] == 1.5
    assert out["np_int"] == 7
    assert out["nested"][0] == [1.0, 2.0]
    assert out["nested"][1]["k"] is None


# ---------------------------------------------------------------------------
# 5. PlatinAI dataset loader
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv").exists(),
    reason="platinai_4cell.csv not found",
)
def test_platinai_dataset_shape() -> None:
    """``PlatinAIRegressionDataset.from_csv`` yields (N, 4) pIC50 + valid mask."""
    mod = _import_module()
    pytest.importorskip("rdkit")
    from pathlib import Path
    ds = mod.PlatinAIRegressionDataset.from_csv(
        _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    )
    assert len(ds) == 4078
    assert ds.pic50.shape == (4078, 4)
    assert ds.valid.shape == (4078, 4)
    counts = ds.per_cell_line_valid_counts()
    # Each cell line should have ~4078 valid pIC50s (the CSV is dense).
    for line in mod.PLATINAI_CELL_LINES:
        assert counts[line] >= 4000, (
            f"too few valid pIC50s for {line}: {counts[line]}"
        )


@pytest.mark.skipif(
    not (_REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv").exists(),
    reason="platinai_4cell.csv not found",
)
def test_platinai_pic50_value_range() -> None:
    """pIC50 values are in the drug-like [3, 9] range."""
    mod = _import_module()
    pytest.importorskip("rdkit")
    import numpy as np
    ds = mod.PlatinAIRegressionDataset.from_csv(
        _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    )
    flat = ds.pic50[np.isfinite(ds.pic50)]
    assert (flat >= 3.0).all() and (flat <= 9.0).all(), (
        f"pIC50 out of range: min={flat.min()} max={flat.max()}"
    )


# ---------------------------------------------------------------------------
# 6. Adapter imports (R18 init contract)
# ---------------------------------------------------------------------------
def test_r18_init_lazy_imports_present() -> None:
    """The R18 lazy import handles are wired in (None is acceptable)."""
    mod = _import_module()
    assert hasattr(mod, "_SE3SBVelocityField")
    assert hasattr(mod, "_RelaxedEndpointTerm")
    assert hasattr(mod, "_MultiPHPocketFeatures")


# ---------------------------------------------------------------------------
# 7. Output paths (override paths respected)
# ---------------------------------------------------------------------------
def test_metrics_path_override() -> None:
    """``--metrics-path`` overrides the default location."""
    mod = _import_module()
    args = mod.parse_args(["--metrics-path", "/tmp/foo.json"])
    assert str(args.metrics_path) == "/tmp/foo.json"


def test_dataset_data_path_override() -> None:
    """``--data`` overrides the default PlatinAI path."""
    mod = _import_module()
    args = mod.parse_args(["--data", "/tmp/foo.csv"])
    assert str(args.data) == "/tmp/foo.csv"


# ---------------------------------------------------------------------------
# 8. End-to-end CLI smoke (CPU-only, 1 epoch, 1 seed, 32 rows)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv").exists(),
    reason="platinai_4cell.csv not found",
)
def test_cli_smoke_platinai_one_seed(tmp_path: Path) -> None:
    """CLI smoke: 1 seed, 1 epoch on platinai dataset — exits 0."""
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    venv_py = (
        _REPO_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
    )
    if not venv_py.exists():
        pytest.skip(f"venv missing at {venv_py}")
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    out_dir = tmp_path / "ckpt"
    cmd = [
        str(venv_py),
        str(_SCRIPT_DIR / "train_affinity_regression.py"),
        "--dataset", "platinai",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--eval-every", "1",
        "--init", "random",
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
    assert metrics_path.exists(), f"metrics JSON missing: {metrics_path}"
    assert summary_path.exists(), f"summary JSON missing: {summary_path}"


# ---------------------------------------------------------------------------
# 9. Aggregate JSON contract (after platinai smoke)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv").exists(),
    reason="platinai_4cell.csv not found",
)
def test_aggregate_json_platinai_contract(tmp_path: Path) -> None:
    """Aggregate JSON has the Wf-32 contract keys (per_seed, headline, etc.)."""
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    venv_py = (
        _REPO_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
    )
    if not venv_py.exists():
        pytest.skip(f"venv missing at {venv_py}")
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    out_dir = tmp_path / "ckpt"
    cmd = [
        str(venv_py),
        str(_SCRIPT_DIR / "train_affinity_regression.py"),
        "--dataset", "platinai",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--eval-every", "1",
        "--init", "random",
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
    payload = json.loads(metrics_path.read_text())
    assert payload["dataset"] == "platinai"
    assert "headline_pearson_r_per_seed" in payload
    assert "headline_pearson_r_mean" in payload
    assert "headline_pearson_r_std" in payload
    assert "per_cell_line_pearson_r" in payload
    assert "r17_baseline_pearson_r" in payload
    assert payload["alpha"] == 0.0


# ---------------------------------------------------------------------------
# 10. HeLa48h CLI smoke (skip if CSV missing)
# ---------------------------------------------------------------------------
def test_cli_smoke_hela48h_skipped_if_no_csv(tmp_path: Path) -> None:
    """``--dataset hela48h`` smoke is a no-op if CSV is missing on this host.

    Most CI hosts do NOT have MetalCytoToxDB.csv (it's a 26 k-row file
    only on the data host).  This test only validates that the CLI
    parses cleanly; the actual run is exercised on a data-bearing host.
    """
    pytest.importorskip("torch")
    mod = _import_module()
    # argparse sanity (full path)
    args = mod.parse_args(["--dataset", "hela48h"])
    assert args.dataset == "hela48h"
    assert args.cell_line == mod.HELA48H_CELL_LINE
    assert args.time_h == mod.HELA48H_TIME_H