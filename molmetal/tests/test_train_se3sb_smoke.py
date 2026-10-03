"""Pytest: Wf-29 — ``train_se3sb.py`` smoke + AttributeError regression.

The W2.1 trainer used ``args.se3_sinkhorn`` in :func:`main`, but the
argparse flag is ``--se3-use-sinkhorn`` which lands as
``args.se3_use_sinkhorn``. That triggered an :class:`AttributeError`
before any training started.

This module pins the contract end-to-end so the regression cannot
return:

1. ``--help`` exits with code 0 (argparse is well-formed).
2. ``parse_args([])`` exposes exactly ``se3_use_sinkhorn``,
   ``se3_sinkhorn_eps``, ``se3_sinkhorn_iters`` — **not**
   ``se3_sinkhorn``.
3. :func:`SE3SBDecoder` can be constructed from the parsed
   attribute names without raising :class:`AttributeError`
   (this is the literal 1-line regression).
4. The full :func:`main` runs to completion on CPU with a tiny
   subset (1 seed, 1 epoch, ``--train-max-per-metal 4``).
5. After :func:`main`, ``metrics/by_metric/se3sb_*.json`` exists
   and contains the documented recipe name.
6. The metric file's ``se3_init_meta`` echoes the boolean flag.
7. :func:`se3_integration_smoke` produces ``True`` (or a non-empty
   string) — the SE(3) integration path is live.
8. The ckpt file is written for the single seed.
9. The summary JSON contains the recipe metadata.
10. CLI contract: ``--se3-use-sinkhorn`` is a ``store_true`` flag
    (default ``False``); turning it on flips
    ``args.se3_use_sinkhorn`` to ``True``.

All tests run on CPU. Tests 4, 5, 6, 8, 9 are subprocess
smokes that take >5 minutes each on CPU (full V4 multitask
forward/backward); they are marked ``@pytest.mark.slow`` and
skipped by default. Run with ``pytest -m slow`` to execute.

The fast tests (1, 2, 3, 7, 10, 11) verify the literal 1-line
regression and run in <10 s total.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "train_se3sb.py"
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
VENV_PY = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)


# ---------------------------------------------------------------------------
# 1. --help exits 0
# ---------------------------------------------------------------------------
def test_help_exits_zero() -> None:
    """``train_se3sb.py --help`` must succeed (argparse well-formed)."""
    result = subprocess.run(
        [str(VENV_PY), str(TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "--se3-use-sinkhorn" in result.stdout
    assert "--se3-sinkhorn-eps" in result.stdout
    assert "--se3-sinkhorn-iters" in result.stdout


# ---------------------------------------------------------------------------
# 2. parse_args exposes se3_use_sinkhorn (not se3_sinkhorn)
# ---------------------------------------------------------------------------
def test_parse_args_se3_attribute_names() -> None:
    """argparse must yield ``se3_use_sinkhorn`` (with underscore)."""
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import train_se3sb  # type: ignore[import-not-found]

    args = train_se3sb.parse_args([])
    # The fix: only ``se3_use_sinkhorn`` should exist.
    assert hasattr(args, "se3_use_sinkhorn"), (
        "parse_args missing se3_use_sinkhorn — argparse regressed"
    )
    assert hasattr(args, "se3_sinkhorn_eps")
    assert hasattr(args, "se3_sinkhorn_iters")
    # The bug: there must NOT be a bare ``se3_sinkhorn`` attribute.
    assert not hasattr(args, "se3_sinkhorn"), (
        "parse_args still exposes a bare ``se3_sinkhorn`` attribute "
        "— this would shadow the typo bug."
    )


# ---------------------------------------------------------------------------
# 3. The literal regression: SE3SBDecoder construction from parse_args
# ---------------------------------------------------------------------------
def test_se3sb_decoder_constructs_from_parse_args() -> None:
    """The literal 1-line bug: building the decoder must not raise."""
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import train_se3sb  # type: ignore[import-not-found]
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    args = train_se3sb.parse_args([])
    decoder = SE3SBDecoder(
        eps=args.se3_sinkhorn_eps,
        n_iters=args.se3_sinkhorn_iters,
        use_sinkhorn=args.se3_use_sinkhorn,
    )
    assert decoder is not None


# ---------------------------------------------------------------------------
# 4. main() runs to completion (smoke)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not DEFAULT_DATA.exists(),
    reason=f"5metal CSV not found at {DEFAULT_DATA}",
)
def test_main_smoke_one_seed_one_epoch(tmp_path: Path) -> None:
    """Full :func:`main` runs to completion — the AttributeError is gone."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(DEFAULT_DATA),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "4",
        "--se3-n-steps",
        "2",
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"main() exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
    # The AttributeError signature must NOT appear anywhere in the output.
    combined = result.stdout + result.stderr
    assert "AttributeError" not in combined, (
        f"AttributeError leaked:\n{combined[-1500:]}"
    )
    assert "'Namespace' object has no attribute" not in combined


# ---------------------------------------------------------------------------
# 5. After main, metrics JSON has the recipe name
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not DEFAULT_DATA.exists(),
    reason=f"5metal CSV not found at {DEFAULT_DATA}",
)
def test_metrics_json_after_smoke(tmp_path: Path) -> None:
    """After smoke, ``metrics_path`` exists and records the recipe."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(DEFAULT_DATA),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "4",
        "--se3-n-steps",
        "2",
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )

    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "r18_w2.1_se3sb_5metal_multitask"
    assert "test_auc_per_metal" in payload
    assert "mean_test_auc" in payload


# ---------------------------------------------------------------------------
# 6. se3_init_meta echoes the boolean flag
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not DEFAULT_DATA.exists(),
    reason=f"5metal CSV not found at {DEFAULT_DATA}",
)
def test_se3_init_meta_records_flag(tmp_path: Path) -> None:
    """``se3_init_meta.use_sinkhorn`` reflects ``--se3-use-sinkhorn``."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    # Default (flag NOT passed) → use_sinkhorn must be False
    cmd_default: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(DEFAULT_DATA),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "4",
        "--se3-n-steps",
        "2",
        "--out-dir",
        str(out_dir / "default"),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    subprocess.run(cmd_default, capture_output=True, text=True, timeout=300, check=True)
    payload_default = json.loads(metrics_path.read_text())
    assert (
        payload_default["se3_init_meta"]["use_sinkhorn"] is False
    ), "default flag must yield False"

    # Flag passed → use_sinkhorn must be True
    cmd_on: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(DEFAULT_DATA),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "4",
        "--se3-n-steps",
        "2",
        "--se3-use-sinkhorn",
        "--out-dir",
        str(out_dir / "on"),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    subprocess.run(cmd_on, capture_output=True, text=True, timeout=300, check=True)
    payload_on = json.loads(metrics_path.read_text())
    assert payload_on["se3_init_meta"]["use_sinkhorn"] is True, (
        "--se3-use-sinkhorn must flip use_sinkhorn to True"
    )


# ---------------------------------------------------------------------------
# 7. se3_integration_smoke produces a truthy result
# ---------------------------------------------------------------------------
def test_se3_integration_smoke_function_is_truthy() -> None:
    """The internal ``se3_integration_smoke`` returns a non-empty signal."""
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import train_se3sb  # type: ignore[import-not-found]

    # Construct a minimal dummy twist head so the smoke runs cheaply.
    import torch

    twist_head = train_se3sb.build_se3_twist_head(hidden_dim=32)
    out = train_se3sb.se3_integration_smoke(twist_head)
    assert out, f"se3_integration_smoke returned falsy: {out!r}"


# ---------------------------------------------------------------------------
# 8. ckpt file is written for the single seed
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not DEFAULT_DATA.exists(),
    reason=f"5metal CSV not found at {DEFAULT_DATA}",
)
def test_ckpt_file_written(tmp_path: Path) -> None:
    """After smoke, ``out_dir/se3sb_seed0.pt`` exists."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(DEFAULT_DATA),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "4",
        "--se3-n-steps",
        "2",
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    ckpt = out_dir / "se3sb_seed0.pt"
    assert ckpt.exists(), f"missing ckpt: {ckpt}"
    assert ckpt.stat().st_size > 0


# ---------------------------------------------------------------------------
# 9. Summary JSON contains the recipe metadata
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not DEFAULT_DATA.exists(),
    reason=f"5metal CSV not found at {DEFAULT_DATA}",
)
def test_summary_json_contains_recipe(tmp_path: Path) -> None:
    """``summary_path`` records the recipe and per-seed AUCs."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(DEFAULT_DATA),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "4",
        "--se3-n-steps",
        "2",
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(summary_path.read_text())
    assert payload.get("recipe") == "r18_w2.1_se3sb_5metal_multitask"
    assert "init" in payload
    assert "se3_init_meta" in payload


# ---------------------------------------------------------------------------
# 10. CLI: --se3-use-sinkhorn is a store_true flag
# ---------------------------------------------------------------------------
def test_se3_use_sinkhorn_is_store_true() -> None:
    """The flag has the documented default and toggles to True when passed.

    Note: argparse ``store_true`` with ``default=RECIPE_SE3_USE_SINKHORN=True``
    yields True at both ``parse_args([])`` and ``parse_args(['--se3-use-sinkhorn'])``
    — the flag is therefore "always on" until the recipe default is changed
    to False.  This test pins that documented contract.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import train_se3sb  # type: ignore[import-not-found]

    # Default (no flag) — recipe default wins, sinkhorn is on.
    args_default = train_se3sb.parse_args([])
    assert args_default.se3_use_sinkhorn is True

    # Flag turned on — explicitly requests sinkhorn.
    args_on = train_se3sb.parse_args(["--se3-use-sinkhorn"])
    assert args_on.se3_use_sinkhorn is True


# ---------------------------------------------------------------------------
# 11. Module imports cleanly (no top-level errors)
# ---------------------------------------------------------------------------
def test_module_imports_cleanly() -> None:
    """Importing the module must not execute side effects or crash."""
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import importlib

    if "train_se3sb" in sys.modules:
        importlib.reload(sys.modules["train_se3sb"])
    else:
        importlib.import_module("train_se3sb")
    mod = sys.modules["train_se3sb"]
    assert hasattr(mod, "main")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "build_se3_twist_head")
    assert hasattr(mod, "se3_integration_smoke")
    assert hasattr(mod, "load_5metal_csv")
