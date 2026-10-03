"""Pytest: Wf-31 Phase 2 — ``train_cisplatin_resistance_v2.py`` deliverables.

This module pins the Wf-31 Phase 2 contract end-to-end so regressions
cannot return.  The deliverables:

1. **NEW train script** ``molmetal/scripts/train_cisplatin_resistance_v2.py``
   — R18 SE(3) SB cisplatin-resistance 4-cell classifier trainer with
   4 init modes (``tmqm`` / ``se3sb`` / ``transport`` / ``multiph``),
   10-dim chemoinformatic side features.
2. **NEW verify script** ``molmetal/scripts/verify_train_cisplatin_resistance_v2.py``
   — CPU-only smoke that exercises the trainer's surface area.
3. **No modifications** to ``train_se3sb.py`` (R17/R18 baseline locked),
   ``train_cisplatin_resistance.py`` (R17 3-cell baseline locked), or
   ``train_ru_5metal_se3sb.py`` (R18 W2.1 baseline locked).

The tests below check:

* Module imports + exported symbols (no top-level crash).
* ``--help`` exits 0 with documented flags.
* ``parse_args`` exposes ``cell_line``, ``seeds``, ``epochs``, ``init``
  with documented defaults + 4 ``--init`` choices.
* :class:`FourCellResistanceDMPNN` constructs and emits the documented
  ``{cell_line -> (B, 2)}`` forward contract.
* :func:`four_cell_loss` produces a finite scalar + backward reaches
  the per-cell heads.
* :func:`load_dataset` returns the documented
  ``(smiles, active, res_feat)`` shape from the Wf-28 ship CSV.
* :func:`aggregate_per_cell` matches a manual numpy reduction within
  ``1e-9`` and is NaN-safe.
* :func:`attach_init_components` wires each of the 4 init modes (or
  no-op when the adapter is missing in the venv).
* End-to-end :func:`main` runs to completion on a tiny CSV-like
  subset (1 seed, 1 epoch) and writes both the per-seed metrics JSON
  and the slim final summary JSON.
* The headline ``per_cell_mean_auc`` / ``per_cell_std_auc`` statistics
  round-trip through the JSON payload.
* The 4 init choices appear in ``--help`` output.
* The verify script's smoke runs to completion.

All tests run on CPU. The end-to-end test uses tiny arch and is
bounded to ``<180s``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_cisplatin_resistance_v2.py"
)
VERIFY_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "verify_train_cisplatin_resistance_v2.py"
)
DEFAULT_DATA = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
VENV_PY = PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"


def _scripts_path() -> str:
    return str(PROJECT_ROOT / "molmetal" / "scripts")


# ---------------------------------------------------------------------------
# Tiny CSV fixture — write a 50-row subset of the real platinai CSV so the
# trainer can run end-to-end without dragging in the full 4K-row file.
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_platinai_csv(tmp_path: Path) -> Path:
    """Write a tiny platinai_4cell-shaped CSV with 50 rows."""
    csv_path = tmp_path / "tiny_platinai.csv"
    rows: List[dict] = []
    smiles_pool = [
        "[Pt](N)(N)(Cl)Cl",  # cisplatin
        "[Pt](N)(Cl)(N)Cl",  # trans-platin
        "COc1c(OC)c(N(=O)=O)c(N(=O)=O)c(OC)c1OC.[Pt]",  # satraplatin-like
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1",  # triphenylphosphine
        "CCN(CC)CC.Cl.Cl.[Pt]",  # Pt(NEt3)2Cl2
        "c1ccncc1.c1ccncc1.Cl.Cl.[Pt]",  # Pt(pyridine)2Cl2
        "CCS",  # ethanethiol
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
    ]
    for i in range(50):
        smi = smiles_pool[i % len(smiles_pool)]
        # Synthesize 4 IC50 values, mix of active / inactive.
        for c in range(4):
            ic50 = (i * 137 + c * 911) % 20000 + 1
            rows.append(
                {
                    "complex_id": f"tiny_{i}",
                    "smiles": smi,
                    "ic50_mcf7_nm": float(ic50),
                    "ic50_a549_nm": float((ic50 + c * 1000) % 25000 + 1),
                    "ic50_a2780_nm": float((ic50 + c * 500) % 18000 + 1),
                    "ic50_a2780cis_nm": float((ic50 + c * 2000) % 30000 + 1),
                    "source": "tiny",
                }
            )
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "complex_id",
                "smiles",
                "ic50_mcf7_nm",
                "ic50_a549_nm",
                "ic50_a2780_nm",
                "ic50_a2780cis_nm",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# 1. --help exits 0 + documented flags
# ---------------------------------------------------------------------------
def test_help_exits_zero() -> None:
    """``train_cisplatin_resistance_v2.py --help`` must succeed."""
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
    assert "--cell-line" in result.stdout
    assert "--init" in result.stdout
    assert "--seeds" in result.stdout
    assert "--epochs" in result.stdout
    # The 4 init choices from the Wf-31 spec.
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        assert init in result.stdout, f"--init {init} missing from --help output"
    # The 4 cell-line choices.
    for cl in ("MCF7", "A549", "A2780", "A2780cis"):
        assert cl in result.stdout, f"--cell-line {cl} missing from --help output"


# ---------------------------------------------------------------------------
# 2. parse_args exposes the documented CLI surface
# ---------------------------------------------------------------------------
def test_parse_args_attribute_names() -> None:
    """argparse must yield ``cell_line``, ``seeds``, ``epochs``, ``init``."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    args = w31p2.parse_args([])
    assert hasattr(args, "cell_line"), "parse_args missing --cell-line"
    assert hasattr(args, "seeds"), "parse_args missing --seeds"
    assert hasattr(args, "epochs"), "parse_args missing --epochs"
    assert hasattr(args, "init"), "parse_args missing --init"
    # Defaults match the Wf-31 Phase 2 spec.
    assert args.init in w31p2.INIT_CHOICES
    assert args.cell_line in w31p2.CELL_LINES
    assert args.seeds == list(w31p2.RECIPE_SEEDS)
    assert args.epochs == w31p2.RECIPE_EPOCHS


# ---------------------------------------------------------------------------
# 3. CLI overrides work end-to-end
# ---------------------------------------------------------------------------
def test_parse_args_overrides() -> None:
    """``--cell-line``, ``--seeds``, ``--epochs``, ``--init`` override defaults."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    args = w31p2.parse_args(
        [
            "--cell-line",
            "MCF7",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "5",
            "--init",
            "se3sb",
        ]
    )
    assert args.cell_line == "MCF7"
    assert args.seeds == [0, 1]
    assert args.epochs == 5
    assert args.init == "se3sb"


# ---------------------------------------------------------------------------
# 4. Module imports cleanly + all documented symbols present
# ---------------------------------------------------------------------------
def test_module_imports_cleanly() -> None:
    """Importing the module must not crash; all Wf-31 Phase 2 symbols are exported."""
    sys.path.insert(0, _scripts_path())
    import importlib

    if "train_cisplatin_resistance_v2" in sys.modules:
        importlib.reload(sys.modules["train_cisplatin_resistance_v2"])
    else:
        importlib.import_module("train_cisplatin_resistance_v2")
    mod = sys.modules["train_cisplatin_resistance_v2"]
    required = [
        "FourCellResistanceDMPNN",
        "four_cell_loss",
        "focal_loss",
        "load_dataset",
        "scaffold_split_indices",
        "attach_init_components",
        "aggregate_per_cell",
        "parse_args",
        "main",
        "build_se3_twist_head",
        "se3_integration_smoke",
        "try_load_tmqm_encoder",
        "freeze_encoder_layers_1_2",
        "INIT_CHOICES",
        "CELL_LINES",
        "N_CELLS",
        "N_RESIST_FEAT",
    ]
    for sym in required:
        assert hasattr(mod, sym), f"missing symbol: {sym}"


# ---------------------------------------------------------------------------
# 5. Model forward contract — 4 per-cell heads, (B, 2) each
# ---------------------------------------------------------------------------
def test_model_forward_contract() -> None:
    """``FourCellResistanceDMPNN`` emits ``{cell -> (B, 2)}`` for all 4 cells."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    model = w31p2.FourCellResistanceDMPNN(hidden_dim=32, n_layers=1)
    smis = ["CCO", "c1ccccc1", "CCN"]
    cid = torch.tensor([0, 1, 2], dtype=torch.long)
    rf = torch.randn(3, w31p2.N_RESIST_FEAT)
    model.eval()
    with torch.no_grad():
        out = model(smis, cid, rf)
    # All 4 cell-line keys must be present.
    assert set(out.keys()) == set(
        w31p2.CELL_LINES
    ), f"output keys must be {set(w31p2.CELL_LINES)}; got {set(out.keys())}"
    for cl in w31p2.CELL_LINES:
        lg = out[cl]
        assert lg.shape == (3, 2), f"{cl}: shape {tuple(lg.shape)} != (3, 2)"
        assert torch.isfinite(lg).all(), f"{cl}: non-finite logits"


# ---------------------------------------------------------------------------
# 6. Multi-task loss + backward propagates to the per-cell heads
# ---------------------------------------------------------------------------
def test_four_cell_loss_and_backward() -> None:
    """``four_cell_loss`` produces a finite scalar + backward reaches the heads."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    model = w31p2.FourCellResistanceDMPNN(hidden_dim=32, n_layers=1)
    smis = ["CCO", "c1ccccc1", "CCN", "CCS"]
    cid = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    rf = torch.randn(4, w31p2.N_RESIST_FEAT)
    # Use training-mode forward — the loss + backward check needs
    # gradients to flow back to the per-cell heads.
    model.train()
    out = model(smis, cid, rf)
    targets = {
        cl: torch.tensor([0, 1, 0, 1], dtype=torch.long) for cl in w31p2.CELL_LINES
    }
    loss = w31p2.four_cell_loss(out, targets)
    assert torch.isfinite(loss), f"loss is non-finite: {loss.item()}"
    # Loss must be positive — focal loss > 0 for any non-trivial input.
    assert loss.item() > 0.0, f"focal loss should be positive; got {loss.item()}"
    # Backward must reach the per-cell heads.
    loss.backward()
    for cl in w31p2.CELL_LINES:
        first_linear = model.cell_heads[cl][0]
        assert first_linear.weight.grad is not None, (
            f"cell_heads[{cl}][0].weight.grad is None — backward did not "
            f"propagate to {cl} head"
        )
        assert torch.isfinite(
            first_linear.weight.grad
        ).all(), f"cell_heads[{cl}][0].weight.grad contains NaN/Inf"


# ---------------------------------------------------------------------------
# 7. load_dataset reads platinai_4cell.csv correctly
# ---------------------------------------------------------------------------
def test_load_dataset_on_real_csv() -> None:
    """``load_dataset`` returns the documented shape on the Wf-28 ship CSV."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    if not DEFAULT_DATA.exists():
        pytest.skip(f"platinai_4cell.csv not found at {DEFAULT_DATA}")
    smiles, active, res_feat = w31p2.load_dataset(DEFAULT_DATA)
    assert len(smiles) > 4000, f"expected >4000 rows; got {len(smiles)}"
    assert active.shape == (
        len(smiles),
        w31p2.N_CELLS,
    ), f"active shape {active.shape} != ({len(smiles)}, {w31p2.N_CELLS})"
    assert res_feat.shape == (len(smiles), w31p2.N_RESIST_FEAT), (
        f"res_feat shape {res_feat.shape} != " f"({len(smiles)}, {w31p2.N_RESIST_FEAT})"
    )
    # Active labels are 0/1 or -1 (missing).
    unique = np.unique(active)
    assert set(unique.tolist()).issubset(
        {-1, 0, 1}
    ), f"active labels must be in {{-1, 0, 1}}; got {unique.tolist()}"
    # Each cell line must have at least some active labels.
    for c in range(w31p2.N_CELLS):
        n_pos = int((active[:, c] == 1).sum())
        n_neg = int((active[:, c] == 0).sum())
        assert n_pos > 100, f"cell {c}: only {n_pos} positives"
        assert n_neg > 100, f"cell {c}: only {n_neg} negatives"


# ---------------------------------------------------------------------------
# 8. aggregate_per_cell — mean/std match numpy + NaN-safe
# ---------------------------------------------------------------------------
def test_aggregate_per_cell_matches_numpy() -> None:
    """``aggregate_per_cell`` returns (mean, std) matching numpy within 1e-9."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    seed_results = [
        {
            "test_auc_per_cell": {
                "MCF7": 0.80,
                "A549": 0.70,
                "A2780": 0.60,
                "A2780cis": 0.50,
            }
        },
        {
            "test_auc_per_cell": {
                "MCF7": 0.82,
                "A549": 0.72,
                "A2780": 0.62,
                "A2780cis": 0.52,
            }
        },
        {
            "test_auc_per_cell": {
                "MCF7": 0.84,
                "A549": 0.74,
                "A2780": 0.64,
                "A2780cis": 0.54,
            }
        },
    ]
    cells = w31p2.CELL_LINES
    means, stds = w31p2.aggregate_per_cell(seed_results, cells)
    for c in cells:
        expected_mean = float(
            np.mean([r["test_auc_per_cell"][c] for r in seed_results])
        )
        expected_std = float(
            np.std(
                [r["test_auc_per_cell"][c] for r in seed_results],
                ddof=0,
            )
        )
        assert (
            abs(means[c] - expected_mean) < 1e-9
        ), f"means[{c}] = {means[c]} != expected {expected_mean}"
        assert (
            abs(stds[c] - expected_std) < 1e-9
        ), f"stds[{c}] = {stds[c]} != expected {expected_std}"
    # NaN-only list must yield NaN.
    nan_only = [{"test_auc_per_cell": {"MCF7": float("nan")}}]
    m, s = w31p2.aggregate_per_cell(nan_only, ("MCF7",))
    import math

    assert math.isnan(m["MCF7"]), f"NaN-only must yield NaN mean; got {m}"
    assert math.isnan(s["MCF7"]), f"NaN-only must yield NaN std; got {s}"


# ---------------------------------------------------------------------------
# 9. attach_init_components — 4 init modes
# ---------------------------------------------------------------------------
def test_attach_init_components_all_modes() -> None:
    """Each of the 4 init modes attaches the documented components (or no-op)."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    device = torch.device("cpu")
    for init in ("tmqm", "se3sb", "transport", "multiph"):
        model = w31p2.FourCellResistanceDMPNN(hidden_dim=32, n_layers=1)
        comps = w31p2.attach_init_components(init, model, device, hidden_dim=32)
        if init == "tmqm":
            assert comps == {}, f"tmqm must be no-op; got {comps}"
        elif init == "se3sb":
            assert "se3_field" in comps, f"se3sb must attach se3_field; got {comps}"
            from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

            assert isinstance(comps["se3_field"], SE3SBVelocityField)
        elif init == "transport":
            if comps:
                assert (
                    "relax_term" in comps
                ), f"transport must attach relax_term; got {comps}"
        elif init == "multiph":
            if comps:
                assert "mphp" in comps, f"multiph must attach mphp; got {comps}"


# ---------------------------------------------------------------------------
# 10. End-to-end main() smoke (1 seed, 1 epoch, tiny CSV fixture)
# ---------------------------------------------------------------------------
def test_main_smoke_one_seed_one_epoch(
    tmp_path: Path,
    tiny_platinai_csv: Path,
) -> None:
    """Full :func:`main` runs to completion on a tiny CSV — no AttributeError."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_platinai_csv),
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
        "1",
        "--eval-every",
        "1",
        "--cell-line",
        "MCF7",
        "--se3-n-steps",
        "2",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"main() exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
    # The Wf-31 Phase 2 trainer must NOT have any AttributeError.
    combined = result.stdout + result.stderr
    assert "AttributeError" not in combined
    assert "'Namespace' object has no attribute" not in combined


# ---------------------------------------------------------------------------
# 11. After main(), per-seed metrics JSON contains the recipe + headline stats
# ---------------------------------------------------------------------------
def test_metrics_json_after_smoke(
    tmp_path: Path,
    tiny_platinai_csv: Path,
) -> None:
    """After smoke, ``metrics_path`` records recipe + per_cell mean/std + headline."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_platinai_csv),
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
        "1",
        "--eval-every",
        "1",
        "--cell-line",
        "MCF7",
        "--se3-n-steps",
        "2",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)

    assert metrics_path.exists(), f"metrics JSON not written at {metrics_path}"
    payload = json.loads(metrics_path.read_text())
    assert payload.get("recipe") == "wf31_r18_se3sb_cisplatin_resistance_v2"
    assert payload.get("init") == "se3sb"
    assert payload.get("cell_line_focus") == "MCF7"
    assert payload.get("cell_lines") == list(["MCF7", "A549", "A2780", "A2780cis"])
    # Headline metrics must round-trip.
    assert "per_cell_mean_auc" in payload
    assert "per_cell_std_auc" in payload
    assert "headline_auc_mean" in payload
    assert "headline_auc_std" in payload
    # The 4 cell lines must be present in the per_cell summary.
    for cl in ("MCF7", "A549", "A2780", "A2780cis"):
        assert cl in payload["per_cell_mean_auc"], (
            f"per_cell_mean_auc missing {cl}: "
            f"{list(payload['per_cell_mean_auc'].keys())}"
        )


# ---------------------------------------------------------------------------
# 12. Summary JSON contains the headline mean ± std
# ---------------------------------------------------------------------------
def test_summary_json_headline(
    tmp_path: Path,
    tiny_platinai_csv: Path,
) -> None:
    """``summary_path`` records the Wf-31 Phase 2 headline mean ± std."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_platinai_csv),
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
        "1",
        "--eval-every",
        "1",
        "--cell-line",
        "MCF7",
        "--se3-n-steps",
        "2",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)

    assert summary_path.exists(), f"summary JSON not written at {summary_path}"
    payload = json.loads(summary_path.read_text())
    assert payload.get("recipe") == (
        "wf31_r18_se3sb_cisplatin_resistance_v2_final_summary"
    )
    assert payload.get("init") == "se3sb"
    assert "headline_auc_mean" in payload
    assert "headline_auc_std" in payload
    assert "wf31_ph2_headline" in payload
    # The headline message must mention the 4-cell classifier.
    assert "4-cell" in payload["wf31_ph2_headline"]
    assert "cisplatin" in payload["wf31_ph2_headline"].lower()


# ---------------------------------------------------------------------------
# 13. Ckpt file is written for the single seed
# ---------------------------------------------------------------------------
def test_ckpt_file_written(
    tmp_path: Path,
    tiny_platinai_csv: Path,
) -> None:
    """After smoke, ``out_dir/cisplatin_resistance_v2_*_seed0.pt`` exists."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_platinai_csv),
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
        "1",
        "--eval-every",
        "1",
        "--cell-line",
        "MCF7",
        "--se3-n-steps",
        "2",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
    ckpt = out_dir / "cisplatin_resistance_v2_se3sb_MCF7_seed0.pt"
    assert ckpt.exists(), f"missing ckpt: {ckpt}"
    assert ckpt.stat().st_size > 0


# ---------------------------------------------------------------------------
# 14. --init tmqm mode also works (no SE(3) twist head)
# ---------------------------------------------------------------------------
def test_main_smoke_tmqm_init(
    tmp_path: Path,
    tiny_platinai_csv: Path,
) -> None:
    """``--init tmqm`` mode runs without attaching the SE(3) twist head."""
    out_dir = tmp_path / "ckpt"
    metrics_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.json"

    cmd: List[str] = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--data",
        str(tiny_platinai_csv),
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
        "1",
        "--eval-every",
        "1",
        "--cell-line",
        "A2780cis",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"--init tmqm exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(metrics_path.read_text())
    assert payload.get("init") == "tmqm"
    assert payload.get("cell_line_focus") == "A2780cis"
    # Ckpt path embeds the init mode + cell line.
    ckpt = out_dir / "cisplatin_resistance_v2_tmqm_A2780cis_seed0.pt"
    assert ckpt.exists(), f"missing tmqm ckpt: {ckpt}"


# ---------------------------------------------------------------------------
# 15. The 4 init choices appear in --help output
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
# 16. The 4 cell-line choices appear in --help output
# ---------------------------------------------------------------------------
def test_cell_line_choices_in_help() -> None:
    """``--help`` lists all 4 cell-line choices (MCF7 / A549 / A2780 / A2780cis)."""
    result = subprocess.run(
        [str(VENV_PY), str(TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    for cl in ("MCF7", "A549", "A2780", "A2780cis"):
        assert cl in result.stdout, f"--cell-line {cl} missing from --help output"


# ---------------------------------------------------------------------------
# 17. Verify script runs to completion
# ---------------------------------------------------------------------------
def test_verify_script_runs(tmp_path: Path) -> None:
    """``verify_train_cisplatin_resistance_v2.py`` runs the CPU smoke checks."""
    if not VERIFY_SCRIPT.exists():
        pytest.skip(f"verify script not found at {VERIFY_SCRIPT}")
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
    # The verify summary line must be present.
    assert "0 fail" in result.stdout or "OK" in result.stdout


# ---------------------------------------------------------------------------
# 18. SE(3) integration smoke — produces a truthy result
# ---------------------------------------------------------------------------
def test_se3_integration_smoke_is_truthy() -> None:
    """``se3_integration_smoke`` returns a non-empty signal."""
    sys.path.insert(0, _scripts_path())
    import train_cisplatin_resistance_v2 as w31p2  # type: ignore[import-not-found]

    twist_head = w31p2.build_se3_twist_head(hidden_dim=32)
    out = w31p2.se3_integration_smoke(twist_head, n_steps=2)
    assert out, f"se3_integration_smoke returned falsy: {out!r}"
