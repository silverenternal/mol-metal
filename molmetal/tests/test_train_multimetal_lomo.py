"""Pytest: Wf-32 Phase 2 — ``train_multimetal_lomo.py`` deliverables.

This module pins the Wf-32 contract end-to-end so regressions cannot
return.  The deliverables:

1. **NEW train script** ``molmetal/scripts/train_multimetal_lomo.py``
   — 5-metal leave-one-metal-out (LOMO) CV with 3 init modes
   (``tmqm`` / ``se3sb`` / ``hilbert``), 5-fold × 5-seed × 50-epoch
   standard recipe.
2. **NEW verify script** ``molmetal/scripts/verify_train_multimetal_lomo.py``
   — CPU-only smoke that exercises the trainer's surface area.
3. **No modifications** to ``train_ru_5metal_se3sb.py`` (Wf-31
   baseline locked per Wf-22 close-out).

The tests below check:

* Module imports + exported symbols (no top-level crash).
* ``--help`` exits 0 with documented flags.
* ``parse_args`` exposes ``folds``, ``seeds``, ``epochs``, ``init``
  with documented defaults + 3 ``--init`` choices.
* ``parse_fold_token`` parses ``leave-Ru-out`` /
  ``leave-Ru`` / ``Ru`` (case-insensitive) → ``Ru``.
* ``split_folds`` dedupes (preserving first-seen order) and validates.
* ``build_lomo_view`` masks the held-out metal out of train/val (and
  out of the *other* metals' test pools).
* ``aggregate_per_fold`` / ``aggregate_overall`` match numpy within
  ``1e-9`` and are NaN-safe.
* ``attach_hilbert_components`` wires the Wf-9 P1-4 composite
  (or no-op when the adapter is missing in the venv).
* End-to-end :func:`main` runs to completion on a tiny CSV
  (1 fold, 1 seed, 1 epoch, ``--train-max-per-metal 4``) and writes
  both the per-seed metrics JSON and the slim final summary JSON.
* The headline ``mean_test_auc_overall`` / ``std_test_auc_overall``
  round-trip through the JSON payload.
* The ``r17_baseline_lomo_auc = 0.535`` reference appears in both
  JSON outputs.
* ``--init hilbert`` mode also runs end-to-end and wires the
  composite (when available in the venv).
* ``--init tmqm`` mode also runs end-to-end (no SE(3) twist head).
* The verify script's 8-check smoke runs to completion.

All tests run on CPU. The end-to-end tests are bounded to ``<180s``
and use ``--train-max-per-metal 4`` so they stay cheap.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "train_multimetal_lomo.py"
VERIFY_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "verify_train_multimetal_lomo.py"
)
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask.csv"
)
VENV_PY = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)


def _scripts_path() -> str:
    return str(PROJECT_ROOT / "molmetal" / "scripts")


def _repo_root_path() -> str:
    return str(PROJECT_ROOT)


def _build_tiny_csv(tmp_path: Path) -> Path:
    """Build a tiny 5-metal CSV (4 train × 2 val × 2 test rows per metal)."""
    import csv as _csv
    csv_path = tmp_path / "tiny_5metal.csv"
    rows = []
    metals = ("Ru", "Ir", "Rh", "Re", "Os")
    for m in metals:
        for sp, n in (("train", 4), ("val", 2), ("test", 2)):
            for i in range(n):
                rows.append({
                    "smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"][i % 4],
                    "metal": m,
                    "scaffold_id": f"s_{m}_{i}",
                    "split": sp,
                    "activity": str(i % 2),
                })
    with csv_path.open("w", newline="") as fh:
        writer = _csv.DictWriter(
            fh, fieldnames=["smiles", "metal", "scaffold_id", "split", "activity"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# 1. --help exits 0 + documented flags
# ---------------------------------------------------------------------------
def test_help_exits_zero() -> None:
    """``train_multimetal_lomo.py --help`` must succeed."""
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
    assert "--folds" in result.stdout
    assert "--seeds" in result.stdout
    assert "--epochs" in result.stdout
    assert "--init" in result.stdout
    # The 3 init choices from the Wf-32 spec.
    for init in ("tmqm", "se3sb", "hilbert"):
        assert init in result.stdout, (
            f"--init {init} missing from --help output"
        )


# ---------------------------------------------------------------------------
# 2. parse_args exposes the documented CLI surface
# ---------------------------------------------------------------------------
def test_parse_args_attribute_names() -> None:
    """argparse must yield ``folds``, ``seeds``, ``epochs``, ``init``."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    args = m32.parse_args([])
    assert hasattr(args, "folds"), "parse_args missing --folds"
    assert hasattr(args, "seeds"), "parse_args missing --seeds"
    assert hasattr(args, "epochs"), "parse_args missing --epochs"
    assert hasattr(args, "init"), "parse_args missing --init"
    # Defaults match the Wf-32 spec.
    assert args.init in m32.INIT_CHOICES
    assert args.folds == list(m32.DEFAULT_FOLDS)
    assert args.seeds == list(m32.RECIPE_SEEDS)
    assert args.epochs == m32.RECIPE_EPOCHS


# ---------------------------------------------------------------------------
# 3. CLI overrides work end-to-end
# ---------------------------------------------------------------------------
def test_parse_args_overrides() -> None:
    """``--folds``, ``--seeds``, ``--epochs``, ``--init`` all override defaults."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    args = m32.parse_args([
        "--folds", "leave-Ru-out", "leave-Ir-out",
        "--seeds", "0", "1",
        "--epochs", "5",
        "--init", "tmqm",
    ])
    assert args.folds == ["leave-Ru-out", "leave-Ir-out"]
    assert args.seeds == [0, 1]
    assert args.epochs == 5
    assert args.init == "tmqm"


# ---------------------------------------------------------------------------
# 4. Module imports cleanly + all 18 documented symbols present
# ---------------------------------------------------------------------------
def test_module_imports_cleanly() -> None:
    """Importing the module must not crash; all Wf-32 symbols are exported."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import importlib
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    if "train_multimetal_lomo" in sys.modules:
        importlib.reload(sys.modules["train_multimetal_lomo"])
    else:
        importlib.import_module("train_multimetal_lomo")
    mod = sys.modules["train_multimetal_lomo"]
    required = [
        "Ru5MetalSe3SBHead",
        "Ru5MetalSe3SBLoss",
        "load_5metal_csv",
        "attach_init_components",
        "build_lomo_view",
        "attach_hilbert_components",
        "aggregate_per_fold",
        "aggregate_overall",
        "parse_fold_token",
        "split_folds",
        "parse_args",
        "main",
        "train_one_fold_seed",
        "INIT_CHOICES",
        "RECIPE_EPOCHS",
        "RECIPE_SEEDS",
        "DEFAULT_HELD_OUT_METALS",
        "DEFAULT_FOLDS",
    ]
    for sym in required:
        assert hasattr(mod, sym), f"missing symbol: {sym}"


# ---------------------------------------------------------------------------
# 5. parse_fold_token — 5 metals + case-insensitive + unknown rejected
# ---------------------------------------------------------------------------
def test_parse_fold_token_all_metals() -> None:
    """``parse_fold_token`` parses all 5 canonical metals (any case)."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    for m in ("Ru", "Ir", "Rh", "Re", "Os"):
        # Canonical token.
        assert m32.parse_fold_token(f"leave-{m}-out") == m
        # No ``-out`` suffix.
        assert m32.parse_fold_token(f"leave-{m}") == m
        # Bare metal.
        assert m32.parse_fold_token(m) == m
        # Lower-case.
        assert m32.parse_fold_token(f"leave-{m.lower()}-out") == m
        # Mixed-case.
        assert m32.parse_fold_token(f"LEAVE-{m.upper()}-OUT") == m


def test_parse_fold_token_rejects_unknown() -> None:
    """Unknown metals raise :class:`ValueError`."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    with pytest.raises(ValueError):
        m32.parse_fold_token("leave-XX-out")
    with pytest.raises(ValueError):
        m32.parse_fold_token("leave-Pt-out")
    with pytest.raises(ValueError):
        m32.parse_fold_token("")


# ---------------------------------------------------------------------------
# 6. split_folds — dedup + order preservation + empty rejection
# ---------------------------------------------------------------------------
def test_split_folds_dedup_and_order() -> None:
    """``split_folds`` dedupes + preserves first-seen order."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    # Bare-metal + prefixed forms collapse to the same metal; order preserved.
    got = m32.split_folds(["leave-Ru-out", "leave-Ir", "Ru", "leave-Os-out"])
    assert got == ["Ru", "Ir", "Os"]
    # Empty list → ValueError.
    with pytest.raises(ValueError):
        m32.split_folds([])
    # Unknown metal → ValueError.
    with pytest.raises(ValueError):
        m32.split_folds(["leave-Pt-out"])


# ---------------------------------------------------------------------------
# 7. build_lomo_view — held-out metal only in test
# ---------------------------------------------------------------------------
def test_build_lomo_view_isolates_held_out() -> None:
    """``build_lomo_view`` masks the held-out metal out of train/val
    AND clears the other metals' test pools."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    data = {
        m: {"train": [], "val": [], "test": []}
        for m in m32.SUPPORTED_METALS
    }
    for m in m32.SUPPORTED_METALS:
        data[m]["train"] = [(f"smi_{m}_train_{j}", j % 2) for j in range(4)]
        data[m]["val"] = [(f"smi_{m}_val_{j}", j % 2) for j in range(2)]
        data[m]["test"] = [(f"smi_{m}_test_{j}", j % 2) for j in range(2)]

    # leave-Ru-out
    view, counts = m32.build_lomo_view(data, "Ru")
    assert len(view["Ru"]["train"]) == 0
    assert len(view["Ru"]["val"]) == 0
    assert len(view["Ru"]["test"]) == 2
    for m in ("Ir", "Rh", "Re", "Os"):
        assert len(view[m]["train"]) == 4
        assert len(view[m]["val"]) == 2
        # Other metals' test rows must be cleared in LOMO view.
        assert len(view[m]["test"]) == 0

    # leave-Os-out
    view2, _ = m32.build_lomo_view(data, "Os")
    assert len(view2["Os"]["test"]) == 2
    assert len(view2["Ru"]["test"]) == 0  # other metals' tests cleared


# ---------------------------------------------------------------------------
# 8. aggregate_per_fold / aggregate_overall — match numpy + NaN-safe
# ---------------------------------------------------------------------------
def test_aggregates_match_numpy() -> None:
    """Per-fold and overall aggregates match a manual numpy reduction."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import numpy as np
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]

    fold_results = {
        "Ru": [
            {"test_auc": 0.80}, {"test_auc": 0.82}, {"test_auc": 0.84},
            {"test_auc": 0.78}, {"test_auc": 0.86},
        ],
        "Ir": [
            {"test_auc": 0.70}, {"test_auc": 0.72}, {"test_auc": 0.74},
            {"test_auc": 0.68}, {"test_auc": 0.76},
        ],
        "Rh": [
            {"test_auc": 0.60}, {"test_auc": 0.62}, {"test_auc": 0.64},
            {"test_auc": 0.58}, {"test_auc": 0.66},
        ],
    }
    per_fold_mean, per_fold_std = m32.aggregate_per_fold(fold_results)
    for m, rs in fold_results.items():
        vals = [r["test_auc"] for r in rs]
        assert abs(per_fold_mean[m] - float(np.mean(vals))) < 1e-9
        assert abs(per_fold_std[m] - float(np.std(vals, ddof=0))) < 1e-9
    overall_mean, overall_std = m32.aggregate_overall(fold_results)
    all_vals = [v for rs in fold_results.values() for r in rs for v in [r["test_auc"]]]
    assert abs(overall_mean - float(np.mean(all_vals))) < 1e-9
    assert abs(overall_std - float(np.std(all_vals, ddof=0))) < 1e-9

    # NaN-only fold → NaN.
    import math
    nan_fold = {"Ru": [{"test_auc": float("nan")}]}
    m_, s_ = m32.aggregate_per_fold(nan_fold)
    assert math.isnan(m_["Ru"])
    assert math.isnan(s_["Ru"])


# ---------------------------------------------------------------------------
# 9. attach_hilbert_components — wires composite (or no-op)
# ---------------------------------------------------------------------------
def test_attach_hilbert_components_wires_or_noop() -> None:
    """``attach_hilbert_components`` wires the composite (or empty dict)."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]
    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskDMPNN

    bb = V4MultiTaskDMPNN(
        hidden_dim=32, n_layers=2, n_metals=5, n_coord_classes=5,
    )
    head = m32.Ru5MetalSe3SBHead(backbone=bb)
    device = torch.device("cpu")
    comps = m32.attach_hilbert_components(
        head, device, population_dim=16, n_iters=4,
    )
    if m32._HilbertASTSRComposite is None:
        assert comps == {}
        return
    assert "hilbert_composite" in comps
    assert hasattr(head, "hilbert_composite")
    assert hasattr(head, "hilbert_n_iters")
    assert head.hilbert_n_iters == 4


# ---------------------------------------------------------------------------
# 10. End-to-end main() smoke (1 fold, 1 seed, 1 epoch, --init se3sb)
# ---------------------------------------------------------------------------
def test_main_smoke_se3sb(tmp_path: Path) -> None:
    """``main`` runs to completion with ``--init se3sb`` (1 fold × 1 seed × 1 epoch)."""
    csv_path = _build_tiny_csv(tmp_path)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    cmd: List[str] = [
        str(VENV_PY), str(TRAIN_SCRIPT),
        "--data", str(csv_path),
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
        "--init", "se3sb",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "4",
        "--se3-n-steps", "2",
        "--folds", "leave-Ru-out",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"main() exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
    combined = result.stdout + result.stderr
    assert "AttributeError" not in combined
    assert "'Namespace' object has no attribute" not in combined


# ---------------------------------------------------------------------------
# 11. After main(), per-fold metrics JSON contains recipe + headline metrics
# ---------------------------------------------------------------------------
def test_metrics_json_after_smoke(tmp_path: Path) -> None:
    """After smoke, ``metrics_path`` records recipe + per-fold mean/std + overall."""
    csv_path = _build_tiny_csv(tmp_path)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    cmd: List[str] = [
        str(VENV_PY), str(TRAIN_SCRIPT),
        "--data", str(csv_path),
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
        "--init", "se3sb",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "4",
        "--se3-n-steps", "2",
        "--folds", "leave-Ru-out",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "wf32_m19_multimetal_lomo_cv"
    assert payload.get("init") == "se3sb"
    assert "Ru" in payload["folds"]
    # Headline metrics must round-trip.
    assert "per_fold_mean_auc" in payload
    assert "per_fold_std_auc" in payload
    assert "mean_test_auc_overall" in payload
    assert "std_test_auc_overall" in payload
    # Per-fold structure must contain the held-out metal's AUC list.
    assert "Ru" in payload["per_fold"]
    assert "test_auc_per_seed" in payload["per_fold"]["Ru"]
    assert len(payload["per_fold"]["Ru"]["test_auc_per_seed"]) == 1
    # The R17 baseline reference must be present.
    assert payload.get("r17_baseline_lomo_auc") == 0.535
    assert "wf32_headline" in payload
    # Ckpt must exist.
    ckpt = out_dir / "multimetal_lomo_se3sb_leaveRu_seed0.pt"
    assert ckpt.exists(), f"missing ckpt: {ckpt}"


# ---------------------------------------------------------------------------
# 12. Summary JSON contains the headline mean ± std
# ---------------------------------------------------------------------------
def test_summary_json_headline(tmp_path: Path) -> None:
    """``summary_path`` records the Wf-32 headline ``mean_test_auc_overall``."""
    csv_path = _build_tiny_csv(tmp_path)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    cmd: List[str] = [
        str(VENV_PY), str(TRAIN_SCRIPT),
        "--data", str(csv_path),
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
        "--init", "se3sb",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "4",
        "--se3-n-steps", "2",
        "--folds", "leave-Ru-out",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(summary_path.read_text())
    assert payload.get("recipe") == "wf32_m19_multimetal_lomo_cv_final_summary"
    assert payload.get("init") == "se3sb"
    assert "mean_test_auc_overall" in payload
    assert "std_test_auc_overall" in payload
    assert "per_fold_mean_auc" in payload
    assert "per_fold_std_auc" in payload
    assert payload.get("r17_baseline_lomo_auc") == 0.535
    assert "wf32_headline" in payload
    assert "R17" in payload["wf32_headline"]


# ---------------------------------------------------------------------------
# 13. --init tmqm mode also works (no SE(3) twist head)
# ---------------------------------------------------------------------------
def test_main_smoke_tmqm_init(tmp_path: Path) -> None:
    """``--init tmqm`` mode runs without attaching the SE(3) twist head."""
    csv_path = _build_tiny_csv(tmp_path)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    cmd: List[str] = [
        str(VENV_PY), str(TRAIN_SCRIPT),
        "--data", str(csv_path),
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
        "--init", "tmqm",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "4",
        "--folds", "leave-Ir-out",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"--init tmqm exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(metrics_path.read_text())
    assert payload.get("init") == "tmqm"
    assert "Ir" in payload["folds"]
    # Ckpt path embeds the init mode + held-out metal.
    ckpt = out_dir / "multimetal_lomo_tmqm_leaveIr_seed0.pt"
    assert ckpt.exists(), f"missing tmqm ckpt: {ckpt}"


# ---------------------------------------------------------------------------
# 14. --init hilbert mode runs end-to-end (when the adapter is in the venv)
# ---------------------------------------------------------------------------
def test_main_smoke_hilbert_init(tmp_path: Path) -> None:
    """``--init hilbert`` mode runs end-to-end."""
    sys.path.insert(0, _scripts_path())
    sys.path.insert(0, _repo_root_path())
    import train_multimetal_lomo as m32  # type: ignore[import-not-found]
    if m32._HilbertASTSRComposite is None:
        pytest.skip("HilbertASTSRComposite adapter unavailable in venv")
    csv_path = _build_tiny_csv(tmp_path)
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"
    cmd: List[str] = [
        str(VENV_PY), str(TRAIN_SCRIPT),
        "--data", str(csv_path),
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--summary-path", str(summary_path),
        "--init", "hilbert",
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "16",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "4",
        "--folds", "leave-Os-out",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"--init hilbert exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(metrics_path.read_text())
    assert payload.get("init") == "hilbert"
    assert "Os" in payload["folds"]
    ckpt = out_dir / "multimetal_lomo_hilbert_leaveOs_seed0.pt"
    assert ckpt.exists(), f"missing hilbert ckpt: {ckpt}"


# ---------------------------------------------------------------------------
# 15. Verify script runs to completion (8-check CPU smoke)
# ---------------------------------------------------------------------------
def test_verify_script_runs() -> None:
    """``verify_train_multimetal_lomo.py`` runs all 8 CPU smoke checks."""
    result = subprocess.run(
        [str(VENV_PY), str(VERIFY_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"verify exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )
    # All 8 numbered checks must have printed.
    for i in range(1, 9):
        assert f"[{i}/8]" in result.stdout, (
            f"check [{i}/8] missing from verify stdout"
        )
    # The final smoke line must be there.
    assert "Wf-32 Phase 2 verify: imports OK" in result.stdout
    assert "CPU smoke 0/8 fail" in result.stdout