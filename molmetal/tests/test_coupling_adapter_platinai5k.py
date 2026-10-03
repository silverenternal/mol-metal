"""Tests for the PlatinAI-5K CPU-only coupling-adapter pretraining.

Covers:

* :mod:`molmetal.scripts.tmqm_coupling_pretraining_cpu` — the
  pretraining script trains on PlatinAI 5K, saves the checkpoint,
  hits the ≤ 0.05 hold-out RMSE gate.
* :func:`molmetal.scripts.tmqm_coupling_pretraining_cpu.load_platinai5k_adapter`
  — the loader returns a handle whose ``embed`` and ``embed_pocket``
  methods produce (1000, 64) finite outputs on the 1K hold-out.
* Reconstruction RMSE on the 1K hold-out is ≤ 0.05 (the spec gate).

Background
----------
TODO-21 (Lambda × CFM coupling) defines a 64-d bridge between
lambda-calculus features and CFM (continuous flow matching)
features.  This module is the CPU-only pretraining step that
produces :file:`models/coupling_adapter_platinai5k.npz` — a
*non-stub* adapter the existing
:mod:`molmetal.molmetal_lam.lam_chem.coupling_adapter` can consume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 5K dataset loader
# ---------------------------------------------------------------------------
def test_load_platinai_5k_smiles_returns_5000() -> None:
    """The 5K loader returns ≥ 5000 SMILES (after auto-materialisation)."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    smiles, labels = pretrain.load_platinai_5k_smiles()
    assert len(smiles) >= 5000, f"only got {len(smiles)} SMILES"
    assert "ic50_nm" in labels
    assert "selectivity" in labels
    assert labels["ic50_nm"].shape[0] == len(smiles)
    assert labels["selectivity"].shape[0] == len(smiles)


def test_platinai_5k_csv_exists_or_is_built() -> None:
    """The CSV is auto-materialised at the canonical path."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    path = pretrain.ensure_platinai_5k_csv()
    assert path.exists(), f"CSV not built at {path}"
    import pandas as _pd

    df = _pd.read_csv(path)
    assert "smiles" in df.columns
    assert len(df) >= 5000


# ---------------------------------------------------------------------------
# Featurisers
# ---------------------------------------------------------------------------
def test_lambda_features_shape_and_finite() -> None:
    """32-d lambda features are float32, finite, deterministic."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    x1 = pretrain._lambda_features_one("c1ccccc1")
    x2 = pretrain._lambda_features_one("c1ccccc1")
    assert x1.shape == (pretrain.LAMBDA_DIM,)
    assert x1.dtype == np.float32
    assert np.isfinite(x1).all()
    np.testing.assert_array_equal(x1, x2)


def test_cfm_features_shape_and_finite() -> None:
    """32-d CFM features are float32, finite."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    t = pretrain._cfm_features_one("c1ccccc1")
    assert t.shape == (pretrain.CFM_DIM,)
    assert np.isfinite(t).all()
    # Two distinct molecules → distinct feature vectors.
    t2 = pretrain._cfm_features_one("CCO")
    assert not np.array_equal(t, t2)


def test_bridge_target_l2_normalised() -> None:
    """The 64-d target is unit-norm per row."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    x = (
        np.random.default_rng(0)
        .standard_normal((5, pretrain.LAMBDA_DIM))
        .astype(np.float32)
    )
    y = pretrain.make_bridge_target(np.zeros_like(x), lambda_features=x)
    assert y.shape == (5, pretrain.EMBED_DIM)
    norms = np.linalg.norm(y, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Pretraining CLI
# ---------------------------------------------------------------------------
def test_pretraining_smoke(tmp_path: Path) -> None:
    """Smoke test: 5 epochs, 200 mols → checkpoint + metadata."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    out = pretrain.train_on_platinai5k(
        out_dir=tmp_path,
        n_epochs=5,
        batch_size=64,
        lr=1e-3,
        seed=0,
        n_smiles=200,
        csv_path=pretrain.DEFAULT_PLATINAI_5K_CSV,
        rmse_gate=10.0,  # disable gate for smoke
        verbose=False,
    )
    npz_path = tmp_path / "coupling_adapter_platinai5k.npz"
    json_path = tmp_path / "coupling_adapter_platinai5k.json"
    assert npz_path.exists()
    assert json_path.exists()
    data = np.load(npz_path)
    assert "embed.weight" in data.files
    assert "embed.bias" in data.files
    assert "head.weight" in data.files
    assert "head.bias" in data.files
    assert "hidden.weight" in data.files
    assert "hidden.bias" in data.files
    assert "input_mean" in data.files
    assert "input_std" in data.files
    assert data["embed.weight"].shape == (pretrain.LAMBDA_DIM, pretrain.EMBED_DIM)
    assert data["head.weight"].shape == (pretrain.EMBED_DIM, pretrain.EMBED_DIM)
    metadata = json.loads(json_path.read_text())
    assert metadata["lambda_dim"] == pretrain.LAMBDA_DIM
    assert metadata["cfm_dim"] == pretrain.CFM_DIM
    assert metadata["embed_dim"] == pretrain.EMBED_DIM
    assert metadata["n_epochs"] == 5
    assert np.isfinite(metadata["final_train_loss"])


def test_pretraining_full_meets_rmse_gate(tmp_path: Path) -> None:
    """Full 100-epoch pretraining clears the 0.05 RMSE gate.

    Runs in ~10 s on a single CPU; uses the canonical
    :file:`models/` output directory but routes to ``tmp_path`` to
    keep tests hermetic.
    """
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    out = pretrain.train_on_platinai5k(
        out_dir=tmp_path,
        n_epochs=100,
        batch_size=64,
        lr=1e-3,
        seed=0,
        n_smiles=5000,
        csv_path=pretrain.DEFAULT_PLATINAI_5K_CSV,
        rmse_gate=0.05,
        verbose=False,
    )
    assert out["gate_pass"], (
        f"RMSE gate failed: rmse={out['rmse']:.4f} > 0.05; "
        f"final_train_loss={out['metadata'].final_train_loss:.6f}"
    )
    assert out["rmse"] <= 0.05
    # Sanity-check the saved metadata.
    md = out["metadata"]
    assert md.n_train_mols == 4000
    assert md.n_holdout_mols == 1000
    assert md.n_epochs == 100
    assert len(md.loss_history) == 100
    assert len(md.rmse_history) == 100
    assert md.loss_history[-1] < 0.01
    assert md.rmse_history[-1] <= 0.05


def test_pretraining_cli_runs(tmp_path: Path) -> None:
    """CLI exits 0 (gate passes)."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    rc = pretrain.main(
        [
            "--out-dir",
            str(tmp_path),
            "--epochs",
            "20",
            "--batch-size",
            "64",
            "--lr",
            "1e-3",
            "--quiet",
        ]
    )
    assert rc == 0
    assert (tmp_path / "coupling_adapter_platinai5k.npz").exists()
    assert (tmp_path / "coupling_adapter_platinai5k.json").exists()


# ---------------------------------------------------------------------------
# Loader / embed
# ---------------------------------------------------------------------------
def test_load_platinai5k_adapter_round_trip(tmp_path: Path) -> None:
    """Train a fresh adapter, load it, embed_pocket returns (64,)."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    pretrain.train_on_platinai5k(
        out_dir=tmp_path,
        n_epochs=20,
        batch_size=64,
        lr=1e-3,
        seed=0,
        n_smiles=500,
        csv_path=pretrain.DEFAULT_PLATINAI_5K_CSV,
        rmse_gate=10.0,
        verbose=False,
    )
    handle = pretrain.load_platinai5k_adapter(
        npz_path=tmp_path / "coupling_adapter_platinai5k.npz"
    )
    vec = handle.embed_pocket()
    assert vec.shape == (pretrain.EMBED_DIM,)
    assert np.isfinite(vec).all()


def test_load_platinai5k_adapter_embed_shape_1000_64(tmp_path: Path) -> None:
    """Verify ``embed`` produces (1000, 64) finite outputs on a 1K batch.

    Mirrors the spec gate from C-R17-C1 step (4).
    """
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    pretrain.train_on_platinai5k(
        out_dir=tmp_path,
        n_epochs=20,
        batch_size=64,
        lr=1e-3,
        seed=0,
        n_smiles=500,
        csv_path=pretrain.DEFAULT_PLATINAI_5K_CSV,
        rmse_gate=10.0,
        verbose=False,
    )
    handle = pretrain.load_platinai5k_adapter(
        npz_path=tmp_path / "coupling_adapter_platinai5k.npz"
    )
    smiles, _ = pretrain.load_platinai_5k_smiles(n_smiles=1000)
    X, _ = pretrain.featurise_corpus(smiles)
    X_n = (X - handle.input_mean) / handle.input_std
    out = handle.embed(X_n)
    assert out.shape == (1000, pretrain.EMBED_DIM)
    assert np.isfinite(out).all()


def test_load_platinai5k_adapter_reconstruction_rmse(tmp_path: Path) -> None:
    """Verify the (1000, 64) hold-out RMSE is ≤ 0.05 against the target."""
    from molmetal.scripts import tmqm_coupling_pretraining_cpu as pretrain

    pretrain.train_on_platinai5k(
        out_dir=tmp_path,
        n_epochs=100,
        batch_size=64,
        lr=1e-3,
        seed=0,
        n_smiles=5000,
        csv_path=pretrain.DEFAULT_PLATINAI_5K_CSV,
        rmse_gate=0.05,
        verbose=False,
    )
    handle = pretrain.load_platinai5k_adapter(
        npz_path=tmp_path / "coupling_adapter_platinai5k.npz"
    )
    # Reproduce the hold-out split deterministically.
    smiles, _ = pretrain.load_platinai_5k_smiles(n_smiles=5000, seed=0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(smiles))
    n_holdout = 1000
    holdout_idx = perm[:n_holdout]
    X_all, _ = pretrain.featurise_corpus(smiles)
    X_h = X_all[holdout_idx]
    # Normalise the lambda features the same way the training
    # script does so the bridge target matches what the model
    # was actually fit to.
    X_h_n = (X_h - handle.input_mean) / handle.input_std
    Y_h = pretrain.make_bridge_target(np.zeros_like(X_h_n), lambda_features=X_h_n)
    pred_h = handle.embed(X_h_n)
    rmse = float(np.sqrt(np.mean((pred_h - Y_h) ** 2)))
    assert rmse <= 0.05, f"reconstruction RMSE {rmse:.4f} exceeds gate 0.05"
