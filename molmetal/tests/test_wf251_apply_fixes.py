"""Wf-251 Apply — unit tests for the 5 CPU-only fixes.

=====================================================================
Each test verifies ONE fix ships correctly:

* F1: ``train_hilbert_astsr_end_to_end`` --feature-set rdkit_32d wires
  the F5_RDKIT_2D 32-d pre-built CSV (32 features per row).
* F2: ``dmpnn_multitask.MT_PROD_HIDDEN`` / ``MT_PROD_DEPTH`` constants
  are exported and instantiable on ``DMPNNMultiTaskModel``.
* F3: ``train_dmpnn_multitask_wasserstein --ablation-config`` accepts
  all 4 configs; ``run_mtl_ablation_grid`` smoke-runs in dry-run mode.
* F4: ``wasserstein_1d_torchsort`` is autograd-friendly (gradient flows
  back); ``wasserstein_pic50_gradient`` scales by ``w_weight``.
* F5: ``build_platinai_rusanov_5metal`` smoke-imports and produces the
  expected CSV shape on synthetic input.

Backward-compat
---------------
The default ``--feature-set rdkit_12d`` path is unchanged (12-d
features).  The default ``--ablation-config mtl_baseline`` is unchanged
(current Wf-131 P1 behaviour).  The default ``--w2-gradient-mode sliced``
is unchanged (current sliced-W₂ path).  ``DMPNN_HIDDEN=128`` is preserved
so the per-metal single-task baseline is unaffected.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Path(__file__).parent = tests/, parent.parent = molmetal/
# The actual repo root is one level above molmetal/ (the project root).
_PROJECT_ROOT = _REPO_ROOT.parent
_TESTS_DIR = _REPO_ROOT / "tests"


# ---------------------------------------------------------------------------
# F1 — F5_RDKIT_2D wire-in
# ---------------------------------------------------------------------------
def test_f1_feature_columns_32d_constant_exists():
    """FEATURE_COLUMNS_32D is exported with the canonical 32-d schema."""
    from molmetal.scripts.train_hilbert_astsr_end_to_end import (
        FEATURE_COLUMNS_32D,
        N_DESCRIPTORS_32D,
    )
    assert len(FEATURE_COLUMNS_32D) == 32
    assert N_DESCRIPTORS_32D == 32
    # First 12 columns match the 12-d baseline (Wf-49 P1 frozen order).
    from molmetal.data.rdkit_2d_extended import RDKIT_2D_FEATURE_COLUMNS
    assert list(FEATURE_COLUMNS_32D[:12]) == list(RDKIT_2D_FEATURE_COLUMNS)


def test_f1_load_f5_32d_csv_smoke(tmp_path):
    """load_f5_32d_csv parses a synthetic CSV with 32-d schema."""
    from molmetal.scripts.train_hilbert_astsr_end_to_end import (
        FEATURE_COLUMNS_32D,
        load_f5_32d_csv,
    )
    csv_path = tmp_path / "plat_fake.csv"
    header = ["smiles", "metal", "pIC50_mcf7"] + list(FEATURE_COLUMNS_32D)
    row = ["Cc1ccccc1", "Pt", "6.5"] + [
        # 12 base (Wf-49 P1 frozen)
        "92.14", "2.13", "0.0", "0", "0", "0", "6", "1", "1", "1.0", "40.0", "2.0",
        # 20 Wf-218 expansion
        "100.0", "5.0", "5.0", "3.0", "2.0", "8.0", "5.0", "1.0", "80.0",
        "0.1", "-0.1", "0.1", "0.1", "0", "0", "1", "0", "0", "0", "0",
    ]
    csv_path.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    smiles_list, X, y, rows_meta = load_f5_32d_csv(csv_path)
    assert len(smiles_list) == 1
    assert X.shape == (1, 32)
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(6.5, abs=1e-3)
    assert rows_meta[0]["metal"] == "Pt"


def test_f1_build_design_matrix_from_32d_smoke(tmp_path):
    """build_design_matrix_from_32d returns (smiles, X, y, stats, rows_meta)."""
    from molmetal.scripts.train_hilbert_astsr_end_to_end import (
        FEATURE_COLUMNS_32D,
        build_design_matrix_from_32d,
    )
    csv_path = tmp_path / "plat_fake.csv"
    header = ["smiles", "metal", "pIC50_mcf7"] + list(FEATURE_COLUMNS_32D)
    row1 = ["Cc1ccccc1", "Pt", "6.5"] + ["0"] * 32
    row2 = ["CCO", "Pt", "7.0"] + ["0"] * 32
    csv_path.write_text(
        ",".join(header) + "\n" + ",".join(row1) + "\n" + ",".join(row2) + "\n"
    )
    smiles_list, X, y, stats, rows_meta = build_design_matrix_from_32d(
        csv_path=csv_path,
    )
    assert len(smiles_list) == 2
    assert X.shape == (2, 32)
    assert y.shape == (2,)
    assert stats["n_parsed"] == 2
    assert rows_meta[0]["smiles"] == "Cc1ccccc1"


def test_f1_train_script_help_shows_feature_set_flag():
    """--feature-set flag appears in the trainer's help text."""
    script = _PROJECT_ROOT / "molmetal" / "scripts" / "train_hilbert_astsr_end_to_end.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert "--feature-set" in proc.stdout
    assert "rdkit_12d" in proc.stdout
    assert "rdkit_32d" in proc.stdout


# ---------------------------------------------------------------------------
# F2 — DMPNN_PROD_HIDDEN / MT_PROD_HIDDEN constants
# ---------------------------------------------------------------------------
def test_f2_mt_prod_constants_present():
    """MT_PROD_HIDDEN=300 / MT_PROD_DEPTH=3 are exported by dmpnn_multitask.

    Subprocess-based to avoid the pre-existing circular-import issue in
    ``molmetal.models._scatter`` that breaks direct imports under
    pytest's sys.path layout.  The CLI approach mirrors how downstream
    trainers consume the constants.
    """
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "from molmetal.baselines.dmpnn_multitask import "
            "MT_HIDDEN, MT_PROD_HIDDEN, MT_PROD_DEPTH; "
            "assert MT_PROD_HIDDEN == 300; "
            "assert MT_PROD_DEPTH == 3; "
            "assert MT_HIDDEN == 128; "
            "print('F2 OK', MT_HIDDEN, MT_PROD_HIDDEN, MT_PROD_DEPTH)",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert "F2 OK" in proc.stdout
    assert "128" in proc.stdout
    assert "300" in proc.stdout


def test_f2_dmpnn_multitask_model_instantiable_at_prod_backbone():
    """DMPNNMultiTaskModel instantiates with hidden=MT_PROD_HIDDEN (subprocess)."""
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "import torch; "
            "from molmetal.baselines.dmpnn_multitask import ("
            "DMPNNMultiTaskModel, MT_PROD_DEPTH, MT_PROD_HIDDEN); "
            "model = DMPNNMultiTaskModel("
            "hidden=MT_PROD_HIDDEN, depth=MT_PROD_DEPTH, dropout=0.1); "
            "n = sum(p.numel() for p in model.parameters() if p.requires_grad); "
            "assert n > 50_000, n; "
            "print('F2-INST OK', n)",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert "F2-INST OK" in proc.stdout


# ---------------------------------------------------------------------------
# F3 — --ablation-config flag + grid driver
# ---------------------------------------------------------------------------
def test_f3_train_script_help_shows_ablation_config_flag():
    """--ablation-config flag appears in the multitask trainer's help text."""
    script = _PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert "--ablation-config" in proc.stdout
    assert "mtl_baseline" in proc.stdout
    assert "mtl_structured" in proc.stdout
    assert "mtl_gradproj" in proc.stdout
    assert "single_task" in proc.stdout


def test_f3_grid_driver_help_and_dry_run(tmp_path):
    """run_mtl_ablation_grid --dry-run prints the per-cell commands."""
    script = _PROJECT_ROOT / "molmetal" / "scripts" / "run_mtl_ablation_grid.py"
    out_dir = tmp_path / "wf251_f3_grid"
    proc = subprocess.run(
        [
            sys.executable, str(script),
            "--configs", "mtl_baseline", "single_task",
            "--metals", "Ru",
            "--seeds", "0",
            "--epochs", "1",
            "--out-dir", str(out_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=120,
    )
    assert "DRY-RUN cell" in proc.stdout
    assert "mtl_baseline" in proc.stdout
    assert "single_task" in proc.stdout
    # Aggregate CSV should be emitted even in dry-run mode.
    assert (out_dir / "aggregate.csv").exists()
    assert (out_dir / "ablation_table.md").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_total_cells"] == 2  # 2 configs × 1 metal × 1 seed
    assert summary["dry_run"] is True


# ---------------------------------------------------------------------------
# F4 — wasserstein_1d_torchsort + wasserstein_pic50_gradient
# ---------------------------------------------------------------------------
def test_f4_wasserstein_1d_torchsort_gradient_flows():
    """Backward on wasserstein_1d_torchsort produces non-zero gradient."""
    import torch
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_1d_torchsort,
    )
    pred = torch.randn(64, requires_grad=True)
    target = torch.randn(64)
    w2 = wasserstein_1d_torchsort(pred, target)
    assert w2.dim() == 0
    assert torch.isfinite(w2).item()
    w2.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    # Gradient must be non-zero on at least one entry (otherwise the
    # penalty would silently zero out — the bug from TODO-179).
    assert pred.grad.abs().sum().item() > 0


def test_f4_wasserstein_pic50_gradient_scales_by_weight():
    """wasserstein_pic50_gradient returns w_weight * W₂."""
    import torch
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_1d_torchsort,
        wasserstein_pic50_gradient,
    )
    pred = torch.randn(64, requires_grad=True)
    target = torch.randn(64)
    w0 = wasserstein_1d_torchsort(pred, target)
    pred2 = pred.detach().clone().requires_grad_(True)
    w05 = wasserstein_pic50_gradient(pred2, target, w_weight=0.5)
    assert torch.isclose(w05, 0.5 * w0, atol=1e-5).item()


def test_f4_train_script_help_shows_w2_gradient_mode_flag():
    """--w2-gradient-mode flag appears in the multitask trainer's help text."""
    script = _PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert "--w2-gradient-mode" in proc.stdout
    assert "sliced" in proc.stdout
    assert "sort" in proc.stdout


# ---------------------------------------------------------------------------
# F5 — PlatinAI Rusanov-2025 xlsx loader
# ---------------------------------------------------------------------------
def test_f5_build_platinai_rusanov_5metal_help():
    """build_platinai_rusanov_5metal --help exposes the expected CLI."""
    script = _PROJECT_ROOT / "molmetal" / "scripts" / "build_platinai_rusanov_5metal.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert "--xlsx-path" in proc.stdout
    assert "--out-csv" in proc.stdout
    assert "--smiles-col" in proc.stdout
    assert "--ic50-col" in proc.stdout
    assert "--metal-col" in proc.stdout
    assert "--cell-line-col" in proc.stdout


def test_f5_synthetic_xlsx_round_trip(tmp_path):
    """End-to-end: write a synthetic xlsx with pIC50 column, run the
    loader, verify the output CSV has the expected schema."""
    pytest.importorskip("openpyxl")
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["SMILES", "IC50_nM", "Metal", "CellLine"])
    # Synthetic Pt complex SMILES (real examples from PlatinAI literature).
    ws.append(["[Pt](N)(N)Cl", "1000", "Pt", "MCF7"])  # pic50 = 3.0
    ws.append(["[Pt](N)(N)Cl", "100", "Pt", "MCF7"])   # pic50 = 4.0 (replicate)
    ws.append(["[Pt](N)(N)Cl", "100", "Pt", "A2780"])  # pic50 = 4.0
    ws.append(["CCO", "10000", "Pt", "MCF7"])           # pic50 = 2.0
    xlsx_path = tmp_path / "synthetic_rusanov.xlsx"
    wb.save(str(xlsx_path))
    out_csv = tmp_path / "out.csv"
    summary_path = tmp_path / "summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "molmetal" / "scripts" / "build_platinai_rusanov_5metal.py"),
            "--xlsx-path", str(xlsx_path),
            "--out-csv", str(out_csv),
            "--summary", str(summary_path),
            "--no-rdkit-validate",  # skip for synthetic
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=120,
    )
    assert out_csv.exists()
    lines = out_csv.read_text().strip().splitlines()
    assert lines[0] == "smiles,metal,cell_line,pic50,n_replicates"
    # The first two rows (MCF7 replicates of the same SMILES) should
    # aggregate to one row with n_replicates=2 and the median pIC50.
    mc7_rows = [ln for ln in lines[1:] if ",MCF7," in ln]
    assert len(mc7_rows) >= 2  # both Pt(MCF7) and EtOH(MCF7)
    # Verify summary JSON
    summary = json.loads(summary_path.read_text())
    assert summary["wf"] == "Wf-251-F5"
    assert summary["n_rows_total"] >= 4
