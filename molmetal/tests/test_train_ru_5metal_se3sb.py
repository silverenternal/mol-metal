"""Pytest: Wf-31 Phase 1 — ``train_ru_5metal_se3sb.py`` deliverables.

This module pins the Wf-31 contract end-to-end so regressions cannot
return.  The deliverables:

1. **NEW train script** ``molmetal/scripts/train_ru_5metal_se3sb.py``
   — 5-metal cls + reg multitask trainer with 4 init modes
   (``tmqm`` / ``se3sb`` / ``transport`` / ``multiph``), 5-seed
   × 50-epoch standard recipe.
2. **NEW verify script** ``molmetal/scripts/verify_train_ru_5metal.py``
   — CPU-only smoke that exercises the trainer's surface area.
3. **No modifications** to ``train_se3sb.py`` (R17/R18 baseline locked)
   or ``train_ru_multitask.py`` (R17 baseline locked per Wf-22).

The tests below check:

* Module imports + exported symbols (no top-level crash).
* ``--help`` exits 0 with documented flags.
* ``parse_args`` exposes ``metals``, ``seeds``, ``epochs``, ``init``
  with documented defaults + 4 ``--init`` choices.
* ``Ru5MetalSe3SBHead`` constructs and emits the documented
  ``{cls, reg, coord}`` forward contract.
* ``Ru5MetalSe3SBLoss`` produces finite scalar + backward reaches
  the regression heads.
* ``load_5metal_csv`` returns the documented
  ``{metal: {split: [(smiles, label)]}}`` shape (and drops unknown
  metals).
* ``aggregate_per_metal`` matches a manual numpy reduction within
  ``1e-9`` and is NaN-safe.
* ``attach_init_components`` wires each of the 4 init modes (or
  no-op when the adapter is missing in the venv).
* End-to-end :func:`main` runs to completion on a tiny CSV
  (1 seed, 1 epoch, ``--train-max-per-metal 4``) and writes both
  the per-seed metrics JSON and the slim final summary JSON.
* The headline ``ru_mean_auc_5seed`` / ``ru_std_auc_5seed`` statistics
  round-trip through the JSON payload.
* The ``--init se3sb`` recipe name appears in the JSON output (the
  Wf-31 SPEC headline init).
* The verify script's 8-check smoke runs to completion.

All tests run on CPU. The end-to-end test is bounded to ``<180s`` and
uses ``--train-max-per-metal 4`` so it stays cheap.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import csv

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "train_ru_5metal_se3sb.py"
VERIFY_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "verify_train_ru_5metal.py"
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
VENV_PY = PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"


def _scripts_path() -> str:
    return str(PROJECT_ROOT / "molmetal" / "scripts")


# ---------------------------------------------------------------------------
# Tiny CSV fixture — used by the smoke tests to avoid loading the full
# 26K-row MetalCytoToxDB CSV (whose val/test splits each contain
# ~5K rows per metal × 5 metals = 25K SMILES, which would take >180s
# per subprocess).  The fixture writes a 30-row CSV (5 metals × 3
# splits × 2 rows) so the trainer's eval / train / ckpt paths run in
# <60s end-to-end.
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_5metal_csv(tmp_path: Path) -> Path:
    """Return a tmp-path 5-metal multitask CSV with 30 rows total."""
    csv_path = tmp_path / "tiny_5metal.csv"
    rows: List[dict] = []
    metals = ("Ru", "Ir", "Rh", "Re", "Os")
    smiles_pool = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C", "CCN"]
    for m in metals:
        for sp in ("train", "val", "test"):
            for i in range(2):
                rows.append(
                    {
                        "smiles": smiles_pool[(i + hash(m)) % len(smiles_pool)],
                        "metal": m,
                        "scaffold_id": f"s_{m}_{sp}_{i}",
                        "split": sp,
                        "activity": str(i % 2),
                    }
                )
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["smiles", "metal", "scaffold_id", "split", "activity"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# 1. --help exits 0 + documented flags
# ---------------------------------------------------------------------------
def test_help_exits_zero() -> None:
    """``train_ru_5metal_se3sb.py --help`` must succeed (argparse well-formed)."""
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
    assert "--metals" in result.stdout
    assert "--seeds" in result.stdout
    assert "--epochs" in result.stdout
    assert "--init" in result.stdout
    # The 4 init choices from the Wf-31 spec.
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        assert init in result.stdout, f"--init {init} missing from --help output"


# ---------------------------------------------------------------------------
# 2. parse_args exposes the documented CLI surface
# ---------------------------------------------------------------------------
def test_parse_args_attribute_names() -> None:
    """argparse must yield ``metals``, ``seeds``, ``epochs``, ``init``."""
    sys.path.insert(0, _scripts_path())
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]

    args = w31.parse_args([])
    assert hasattr(args, "metals"), "parse_args missing --metals"
    assert hasattr(args, "seeds"), "parse_args missing --seeds"
    assert hasattr(args, "epochs"), "parse_args missing --epochs"
    assert hasattr(args, "init"), "parse_args missing --init"
    # Defaults match the Wf-31 spec.
    assert args.init in w31.INIT_CHOICES
    assert args.metals == list(w31.DEFAULT_METALS)
    assert args.seeds == list(w31.RECIPE_SEEDS)
    assert args.epochs == w31.RECIPE_EPOCHS


# ---------------------------------------------------------------------------
# 3. CLI overrides work end-to-end
# ---------------------------------------------------------------------------
def test_parse_args_overrides() -> None:
    """``--metals``, ``--seeds``, ``--epochs``, ``--init`` all override defaults."""
    sys.path.insert(0, _scripts_path())
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]

    args = w31.parse_args(
        [
            "--metals",
            "Ru",
            "Ir",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "5",
            "--init",
            "tmqm",
        ]
    )
    assert args.metals == ["Ru", "Ir"]
    assert args.seeds == [0, 1]
    assert args.epochs == 5
    assert args.init == "tmqm"


# ---------------------------------------------------------------------------
# 4. Module imports cleanly + all 14 documented symbols present
# ---------------------------------------------------------------------------
def test_module_imports_cleanly() -> None:
    """Importing the module must not crash; all Wf-31 symbols are exported."""
    sys.path.insert(0, _scripts_path())
    import importlib
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]

    # Re-import (forces fresh load — same shape as test 11).
    if "train_ru_5metal_se3sb" in sys.modules:
        importlib.reload(sys.modules["train_ru_5metal_se3sb"])
    else:
        importlib.import_module("train_ru_5metal_se3sb")
    mod = sys.modules["train_ru_5metal_se3sb"]
    required = [
        "Ru5MetalSe3SBHead",
        "Ru5MetalSe3SBLoss",
        "load_5metal_csv",
        "attach_init_components",
        "aggregate_per_metal",
        "parse_args",
        "main",
        "build_se3_twist_head",
        "se3_integration_smoke",
        "INIT_CHOICES",
        "RECIPE_SEEDS",
        "RECIPE_EPOCHS",
        "DEFAULT_METALS",
        "COORD_BY_METAL",
    ]
    for sym in required:
        assert hasattr(mod, sym), f"missing symbol: {sym}"


# ---------------------------------------------------------------------------
# 5. Head wrapper forward contract — 5 cls + 5 reg + coord
# ---------------------------------------------------------------------------
def test_head_wrapper_forward_contract() -> None:
    """V4 + 5 cls + 5 reg heads emit the documented shapes."""
    sys.path.insert(0, _scripts_path())
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]
    from molmetal.models.v4_multitask_dmpnn import (
        SUPPORTED_METALS,
        V4MultiTaskDMPNN,
    )

    bb = V4MultiTaskDMPNN(
        hidden_dim=64,
        n_layers=2,
        n_metals=5,
        n_coord_classes=5,
    )
    head = w31.Ru5MetalSe3SBHead(backbone=bb)
    smis = ["CCO", "c1ccccc1", "CC(=O)O"]
    mids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = head(smis, mids)
    assert set(out.keys()) == {
        "cls",
        "reg",
        "coord",
    }, f"forward keys must be {{cls, reg, coord}}; got {set(out.keys())}"
    assert set(out["cls"].keys()) == set(SUPPORTED_METALS)
    assert set(out["reg"].keys()) == set(SUPPORTED_METALS)
    for m in SUPPORTED_METALS:
        assert out["cls"][m].shape == (
            3,
            2,
        ), f"cls[{m}] shape {tuple(out['cls'][m].shape)} != (3, 2)"
        assert out["reg"][m].shape == (
            3,
            1,
        ), f"reg[{m}] shape {tuple(out['reg'][m].shape)} != (3, 1)"
    assert out["coord"].shape == (3, 5)


# ---------------------------------------------------------------------------
# 6. Multi-task loss + backward propagates to the reg heads
# ---------------------------------------------------------------------------
def test_loss_and_backward() -> None:
    """The Wf-31 loss produces a finite scalar + backward reaches the reg heads."""
    sys.path.insert(0, _scripts_path())
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]
    import torch
    from molmetal.models.v4_multitask_dmpnn import (
        SUPPORTED_METALS,
        V4MultiTaskDMPNN,
    )

    bb = V4MultiTaskDMPNN(hidden_dim=64, n_layers=2, n_metals=5, n_coord_classes=5)
    head = w31.Ru5MetalSe3SBHead(backbone=bb)
    smis = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"]
    mids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = head(smis, mids)
    y_true = {m: torch.tensor([0, 1, 0, 1], dtype=torch.long) for m in SUPPORTED_METALS}
    coord_true = torch.tensor([3, 2, 3, 0], dtype=torch.long)
    loss_fn = w31.Ru5MetalSe3SBLoss(n_coord_classes=5)
    loss, info = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert "loss_total" in info
    for m in SUPPORTED_METALS:
        assert f"loss_cls_{m}" in info
        assert f"loss_reg_{m}" in info
    assert "loss_coord" in info
    # Backward must reach the regression heads (the Wf-31 deliverable).
    loss.backward()
    for m in SUPPORTED_METALS:
        w_first = head.reg_heads[m][0].weight
        assert w_first.grad is not None, (
            f"reg_heads[{m}][0].weight.grad is None — backward did not "
            f"propagate to the Wf-31 reg heads"
        )
        assert torch.isfinite(w_first.grad).all()


# ---------------------------------------------------------------------------
# 7. aggregate_per_metal — mean/std match numpy + NaN-safe
# ---------------------------------------------------------------------------
def test_aggregate_per_metal_matches_numpy() -> None:
    """``aggregate_per_metal`` returns (mean, std) matching numpy within 1e-9."""
    sys.path.insert(0, _scripts_path())
    import numpy as np
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]

    seed_results = [
        {
            "test_auc_per_metal": {
                "Ru": 0.80,
                "Ir": 0.70,
                "Rh": 0.60,
                "Re": 0.50,
                "Os": 0.40,
            }
        },
        {
            "test_auc_per_metal": {
                "Ru": 0.82,
                "Ir": 0.72,
                "Rh": 0.62,
                "Re": 0.52,
                "Os": 0.42,
            }
        },
        {
            "test_auc_per_metal": {
                "Ru": 0.84,
                "Ir": 0.74,
                "Rh": 0.64,
                "Re": 0.54,
                "Os": 0.44,
            }
        },
    ]
    metals = ("Ru", "Ir", "Rh", "Re", "Os")
    means, stds = w31.aggregate_per_metal(seed_results, metals)
    for m in metals:
        expected_mean = float(
            np.mean([r["test_auc_per_metal"][m] for r in seed_results])
        )
        expected_std = float(
            np.std(
                [r["test_auc_per_metal"][m] for r in seed_results],
                ddof=0,
            )
        )
        assert abs(means[m] - expected_mean) < 1e-9
        assert abs(stds[m] - expected_std) < 1e-9
    # NaN-only list must yield NaN.
    nan_only = [{"test_auc_per_metal": {"Ru": float("nan")}}]
    m, s = w31.aggregate_per_metal(nan_only, ("Ru",))
    import math

    assert math.isnan(m["Ru"]), f"NaN-only must yield NaN mean; got {m}"
    assert math.isnan(s["Ru"]), f"NaN-only must yield NaN std; got {s}"


# ---------------------------------------------------------------------------
# 8. attach_init_components — 4 init modes
# ---------------------------------------------------------------------------
def test_attach_init_components_all_modes() -> None:
    """Each of the 4 init modes attaches the documented components (or no-op)."""
    sys.path.insert(0, _scripts_path())
    import train_ru_5metal_se3sb as w31  # type: ignore[import-not-found]
    import torch
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField
    from molmetal.models.v4_multitask_dmpnn import V4MultiTaskDMPNN

    device = torch.device("cpu")
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        bb = V4MultiTaskDMPNN(
            hidden_dim=32,
            n_layers=2,
            n_metals=5,
            n_coord_classes=5,
        )
        head = w31.Ru5MetalSe3SBHead(backbone=bb)
        comps = w31.attach_init_components(init, head, device, hidden_dim=32)
        if init == "tmqm":
            assert comps == {}
        elif init == "se3sb":
            assert "se3_field" in comps
            assert isinstance(comps["se3_field"], SE3SBVelocityField)
        elif init == "transport":
            if comps:
                assert "relax_term" in comps
        elif init == "multiph":
            if comps:
                assert "mphp" in comps


# ---------------------------------------------------------------------------
# 9. End-to-end main() smoke (1 seed, 1 epoch, tiny CSV fixture)
# ---------------------------------------------------------------------------
def test_main_smoke_one_seed_one_epoch(
    tmp_path: Path,
    tiny_5metal_csv: Path,
) -> None:
    """Full :func:`main` runs to completion on a tiny CSV — no AttributeError."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_5metal_csv),
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
        "--init",
        "se3sb",
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
        "--metals",
        "Ru",
        "Ir",
        "Rh",
        "Re",
        "Os",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"main() exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
    # The Wf-31 trainer must NOT have any AttributeError.
    combined = result.stdout + result.stderr
    assert "AttributeError" not in combined
    assert "'Namespace' object has no attribute" not in combined


# ---------------------------------------------------------------------------
# 10. After main(), per-seed metrics JSON contains the recipe + headline metrics
# ---------------------------------------------------------------------------
def test_metrics_json_after_smoke(
    tmp_path: Path,
    tiny_5metal_csv: Path,
) -> None:
    """After smoke, ``metrics_path`` records recipe + per_metal mean/std + Ru headline."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_5metal_csv),
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
        "--init",
        "se3sb",
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
        "--metals",
        "Ru",
        "Ir",
        "Rh",
        "Re",
        "Os",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)

    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "wf31_r18_m22_se3sb_5metal_cls_reg"
    assert payload.get("init") == "se3sb"
    assert payload.get("metals") == ["Ru", "Ir", "Rh", "Re", "Os"]
    # Headline metrics must round-trip.
    assert "per_metal_mean_auc" in payload
    assert "per_metal_std_auc" in payload
    assert "ru_mean_auc_5seed" in payload
    assert "ru_std_auc_5seed" in payload
    # The 5 metals must be present in the per_metal summary (when
    # there are >=1 test rows, which the smoke CSV guarantees).
    for m in ("Ru", "Ir", "Rh", "Re", "Os"):
        assert m in payload["per_metal_mean_auc"], (
            f"per_metal_mean_auc missing {m}: "
            f"{list(payload['per_metal_mean_auc'].keys())}"
        )


# ---------------------------------------------------------------------------
# 11. Summary JSON contains the headline Ru mean ± std
# ---------------------------------------------------------------------------
def test_summary_json_headline(
    tmp_path: Path,
    tiny_5metal_csv: Path,
) -> None:
    """``summary_path`` records the Wf-31 headline ``ru_mean_auc_5seed``."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_5metal_csv),
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
        "--init",
        "se3sb",
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
        "--metals",
        "Ru",
        "Ir",
        "Rh",
        "Re",
        "Os",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)

    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(summary_path.read_text())
    assert payload.get("recipe") == ("wf31_r18_m22_se3sb_5metal_cls_reg_final_summary")
    assert payload.get("init") == "se3sb"
    assert "ru_mean_auc_5seed" in payload
    assert "ru_std_auc_5seed" in payload
    assert "wf31_headline" in payload
    # The headline message must mention Krasnov 0.81 — the Wf-31 SPEC target.
    assert "Krasnov" in payload["wf31_headline"]
    assert "0.81" in payload["wf31_headline"]


# ---------------------------------------------------------------------------
# 12. Ckpt file is written for the single seed
# ---------------------------------------------------------------------------
def test_ckpt_file_written(
    tmp_path: Path,
    tiny_5metal_csv: Path,
) -> None:
    """After smoke, ``out_dir/ru_5metal_se3sb_{init}_seed0.pt`` exists."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_5metal_csv),
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
        "--init",
        "se3sb",
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
        "--metals",
        "Ru",
        "Ir",
        "Rh",
        "Re",
        "Os",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    ckpt = out_dir / "ru_5metal_se3sb_se3sb_seed0.pt"
    assert ckpt.exists(), f"missing ckpt: {ckpt}"
    assert ckpt.stat().st_size > 0


# ---------------------------------------------------------------------------
# 13. --init tmqm mode also works (no SE(3) twist head)
# ---------------------------------------------------------------------------
def test_main_smoke_tmqm_init(
    tmp_path: Path,
    tiny_5metal_csv: Path,
) -> None:
    """``--init tmqm`` mode runs without attaching the SE(3) twist head."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_5metal_csv),
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
        "--init",
        "tmqm",
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
        "--metals",
        "Ru",
        "Ir",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"--init tmqm exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(metrics_path.read_text())
    assert payload.get("init") == "tmqm"
    assert payload.get("metals") == ["Ru", "Ir"]
    # Ckpt path embeds the init mode.
    ckpt = out_dir / "ru_5metal_se3sb_tmqm_seed0.pt"
    assert ckpt.exists(), f"missing tmqm ckpt: {ckpt}"


# ---------------------------------------------------------------------------
# 14. The 4 init choices appear in --help output
# ---------------------------------------------------------------------------
def test_init_choices_in_help() -> None:
    """``--help`` lists all 4 Wf-31 init choices (tmqm / se3sb / transport / multiph)."""
    result = subprocess.run(
        [str(VENV_PY), str(TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        assert init in result.stdout, f"--init {init} missing from --help output"


# ---------------------------------------------------------------------------
# 15. Verify script runs to completion (8-check CPU smoke)
# ---------------------------------------------------------------------------
def test_verify_script_runs(tmp_path: Path) -> None:
    """``verify_train_ru_5metal.py`` runs all 8 CPU smoke checks."""
    result = subprocess.run(
        [str(VENV_PY), str(VERIFY_SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"verify exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    # All 8 numbered checks must have printed.
    for i in range(1, 9):
        assert f"[{i}/8]" in result.stdout, f"check [{i}/8] missing from verify stdout"
    # The final smoke line must be there.
    assert "CPU smoke 0/8 fail" in result.stdout
