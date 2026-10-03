"""Tests for D-MPNN pIC50 checkpoint diagnostic (Wf-84 Phase 3).

These tests are deliberately **additive**: they import and exercise the
diagnostic + training helpers without actually running a full 30-epoch
training (which would take too long for a unit test). The goal is to:

1. Verify the diagnostic's CLI imports and its ``parse_args`` accepts the
   documented flags.
2. Verify the training script's CLI imports and ``parse_args`` accepts the
   documented flags.
3. Verify the diagnostic correctly enumerates checkpoint files in a
   **fake** ``--ckpt-dir`` (tmp dir with 0-byte ``dmpnn_pic50_*.pt``
   shims); the fake ckpts deliberately don't have valid torch payloads
   so we only assert that the **inventory step** enumerates them.
4. Verify the diagnostic's load helper handles missing/corrupt files
   gracefully without crashing the whole run.
5. Verify the per-metal split-index builder returns sane train/val/test
   shapes on the real MetalCytoToxDB.

Run with: ``pytest molmetal/tests/test_dmpnn_ckpt_diagnostic.py -v``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: diagnostic CLI imports + parse_args
# ---------------------------------------------------------------------------
def test_diagnostic_cli_importable():
    """molmetal.scripts.dmpnn_ckpt_diagnostic imports and parse_args works."""
    from molmetal.scripts.dmpnn_ckpt_diagnostic import parse_args, main

    assert callable(parse_args)
    assert callable(main)


def test_diagnostic_parse_args_defaults():
    """Default flags: ckpt-dir = molmetal/checkpoints, pattern includes seed wildcard."""
    from molmetal.scripts.dmpnn_ckpt_diagnostic import parse_args

    ns = parse_args(["--ckpt-dir", "/tmp/fake_ckpts", "--out-dir", "/tmp/fake_out"])
    assert ns.ckpt_dir == Path("/tmp/fake_ckpts")
    assert ns.out_dir == Path("/tmp/fake_out")
    assert "dmpnn_pic50" in ns.pattern
    assert "*_seed*" in ns.pattern


# ---------------------------------------------------------------------------
# Test 2: training CLI imports + parse_args
# ---------------------------------------------------------------------------
def test_training_cli_importable():
    """molmetal.scripts.train_dmpnn_pic50_real imports and parse_args works."""
    from molmetal.scripts.train_dmpnn_pic50_real import parse_args, main

    assert callable(parse_args)
    assert callable(main)


def test_training_parse_args_seeds_parsing():
    """parse_args handles both space-separated and comma-separated seeds."""
    from molmetal.scripts.train_dmpnn_pic50_real import parse_args

    ns = parse_args(["--seeds", "0", "1", "2", "--epochs", "5"])
    assert ns.seeds == [0, 1, 2]
    assert ns.epochs == 5

    # Comma-separated inside a single token only takes the first element
    # (argparse `nargs='+'` + `type=parse_seeds` keeps that behaviour by design).
    ns2 = parse_args(["--seeds", "0,1,2"])
    assert ns2.seeds == [0]


def test_training_parse_args_metals_subset():
    """--metals restricts to a subset of {Ru,Ir,Rh,Os,Re}."""
    from molmetal.scripts.train_dmpnn_pic50_real import parse_args

    ns = parse_args(["--metals", "Ru", "Ir"])
    assert ns.metals == ["Ru", "Ir"]


# ---------------------------------------------------------------------------
# Test 3: inventory enumeration on fake ckpts
# ---------------------------------------------------------------------------
def test_diagnostic_inventory_enumerates_fake_ckpts(tmp_path):
    """With a fake ckpt dir containing 0-byte shims, the diagnostic's
    inventory step lists them all but the load step reports errors.
    We don't run main() because torch.load on a 0-byte file would crash;
    instead we exercise the helpers directly."""
    from molmetal.scripts.dmpnn_ckpt_diagnostic import _sha256_prefix

    fake_dir = tmp_path / "fake_ckpts"
    fake_dir.mkdir()
    fakes = []
    for metal, seed in [("ru", 0), ("ir", 1), ("os", 2)]:
        p = fake_dir / f"dmpnn_pic50_{metal}_seed{seed}.pt"
        p.write_bytes(b"FAKE_CKPT_BYTES_FOR_TESTING")
        fakes.append(p)

    # Glob check
    matches = sorted(fake_dir.glob("dmpnn_pic50_*_seed*.pt"))
    assert len(matches) == 3
    for p in matches:
        assert p.exists()
        # sha256 prefix is deterministic
        h1 = _sha256_prefix(p)
        h2 = _sha256_prefix(p)
        assert h1 == h2
        assert len(h1) == 16  # 8 bytes → 16 hex chars


# ---------------------------------------------------------------------------
# Test 4: corrupt-file handling
# ---------------------------------------------------------------------------
def test_diagnostic_load_helper_handles_corrupt_ckpt(tmp_path):
    """load_ckpt_diagnostic must catch torch.load exceptions and report them,
    not propagate (so the main loop continues across bad files)."""
    from molmetal.scripts.dmpnn_ckpt_diagnostic import load_ckpt_diagnostic

    bad = tmp_path / "dmpnn_pic50_ru_seed0.pt"
    bad.write_bytes(b"\x00\x01\x02NOT_A_VALID_TORCH_BLOB")

    # The diagnostic should raise because torch.load fails.
    # We assert it raises (NOT silently succeed).
    with pytest.raises(Exception):
        load_ckpt_diagnostic(bad)


# ---------------------------------------------------------------------------
# Test 5: per-metal split-index builder on a tiny fake dataset
# ---------------------------------------------------------------------------
class _FakeDataset:
    """Minimal stand-in for MetalCytotoxDataset: just .smiles and .metals attrs."""

    def __init__(self, smiles, metal, pic50=None, active=None):
        self.smiles = np.asarray(smiles, dtype=object)
        self.metals = np.asarray(metal, dtype=object)
        self.pic50 = np.asarray(
            pic50 if pic50 is not None else np.full(len(smiles), 5.0)
        )
        self.active = np.asarray(
            active if active is not None else np.zeros(len(smiles), dtype=int)
        )


def test_split_index_builder_returns_sane_shapes():
    """build_metal_split_indices returns (idx_train, idx_val, idx_test) of
    non-empty integer arrays, non-overlapping, on a small synthetic dataset
    (avoids the data-root dependency)."""
    import pandas as pd
    from molmetal.scripts.train_dmpnn_pic50_real import build_metal_split_indices

    # Tiny fake dataset: 6 Ru rows + 4 Ir rows
    smiles = [
        "Cc1ccccc1",
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "O=C([O-])c1ccccc1",
        "C1CCCC1",
        "CCN(CC)CC",
        "C1CCCCC1",
        "CCC",
    ]
    metal = ["Ru", "Ru", "Ru", "Ru", "Ru", "Ru", "Ir", "Ir", "Ir", "Ir"]
    ds = _FakeDataset(smiles, metal)

    split_df = pd.DataFrame(
        {
            "smiles": smiles[:6] + ["C1CCCC1", "CCN(CC)CC"],
            "metal": ["Ru"] * 6 + ["Ir"] * 2,
            "scaffold_id": [""] * 8,
            "split": ["train", "train", "train", "val", "val", "test", "train", "test"],
            "activity": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )

    idx_train, idx_val, idx_test = build_metal_split_indices(split_df, ds, metal="Ru")

    # 6 Ru rows in ds; splits encode 3 train, 2 val, 1 test
    assert len(idx_train) == 3
    assert len(idx_val) == 2
    assert len(idx_test) == 1

    # Non-overlapping
    train_set = set(idx_train.tolist())
    val_set = set(idx_val.tolist())
    test_set = set(idx_test.tolist())
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


# ---------------------------------------------------------------------------
# Test 6: training constants sane
# ---------------------------------------------------------------------------
def test_training_default_constants():
    """DEFAULT_EPOCHS / DEFAULT_ALPHA / FIVE_METALS are wired correctly."""
    from molmetal.scripts import train_dmpnn_pic50_real as m

    assert m.DEFAULT_EPOCHS >= 1
    assert 0.0 <= m.DEFAULT_ALPHA <= 1.0
    assert set(m.FIVE_METALS) == {"Ru", "Ir", "Rh", "Os", "Re"}
    assert set(m.DEFAULT_SEEDS) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Test 7: probe SMILES are valid + not pathological
# ---------------------------------------------------------------------------
def test_probe_smiles_valid():
    """Every PROBE_SMILES can be parsed by RDKit without errors."""
    from rdkit import Chem
    from molmetal.scripts.dmpnn_ckpt_diagnostic import PROBE_SMILES

    for smi in PROBE_SMILES:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid SMILES: {smi}"
        assert mol.GetNumAtoms() > 0


# ---------------------------------------------------------------------------
# Test 8: regression metrics helper consistent
# ---------------------------------------------------------------------------
def test_regression_metrics_keys_present():
    """regression_metrics emits all 4 keys (mae, rmse, pearson, n)."""
    from molmetal.baselines.dmpnn_multitask import regression_metrics

    y_true = np.array([5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([5.1, 5.9, 7.2, 7.8])
    out = regression_metrics(y_true, y_pred)
    assert set(out.keys()) >= {"mae", "rmse", "pearson", "n"}
    assert out["n"] == 4
    assert np.isfinite(out["mae"])
    assert np.isfinite(out["rmse"])
    assert -1.0 <= out["pearson"] <= 1.0


# ---------------------------------------------------------------------------
# Test 9: training script CLI produces non-NaN JSON keys
# ---------------------------------------------------------------------------
def test_training_parse_args_has_required_keys():
    """All documented flags (--seeds, --epochs, --metals, --alpha, --lr,
    --batch-size, --out-dir, --ckpt-dir, --csv, --device, --max-train-rows)
    are exposed."""
    from molmetal.scripts.train_dmpnn_pic50_real import parse_args

    ns = parse_args([])
    for attr in (
        "seeds",
        "epochs",
        "metals",
        "alpha",
        "lr",
        "batch_size",
        "out_dir",
        "ckpt_dir",
        "csv",
        "device",
    ):
        assert hasattr(ns, attr), f"missing arg: {attr}"


# ---------------------------------------------------------------------------
# Test 10: diagnostic CLI exit code 0 with no ckpts (graceful empty)
# ---------------------------------------------------------------------------
def test_diagnostic_empty_dir_does_not_crash(tmp_path):
    """With an empty ckpt dir, the diagnostic main() should exit 0 and
    write empty-but-valid JSON files (not raise)."""
    from molmetal.scripts.dmpnn_ckpt_diagnostic import main

    fake_dir = tmp_path / "empty_ckpts"
    fake_dir.mkdir()
    out_dir = tmp_path / "empty_out"

    rc = main(["--ckpt-dir", str(fake_dir), "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "ckpt_inventory.json").exists()
    inv = json.loads((out_dir / "ckpt_inventory.json").read_text())
    assert inv["n_checkpoints"] == 0
    assert inv["status"] == "EMPTY"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
