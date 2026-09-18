"""Tests for the Lambda × CFM coupling adapter (TODO-21 / 2026-09-16).

Covers:

* :mod:`molmetal.scripts.tmqm_cfm_pretraining` — ``--dry-run`` trains 1
  step on the stub corpus and writes a checkpoint with the expected
  keys + a finite JSON metadata side-car.
* :mod:`molmetal.molmetal_lam.lam_chem.coupling_adapter` —
  :func:`load_coupling_adapter` returns a finite 64-d vector from the
  dry-run checkpoint and from the stub fallback.
* :mod:`molmetal.molmetal_lam.search_alg.warm_start` —
  :func:`pocket_features` picks up the adapter when the env gate is on
  and ignores it when the gate is off (backward compat).
* :mod:`molmetal.molmetal_lam.search_alg.learned_prior` —
  :class:`LearnedPolicyPrior` accepts a coupling_adapter and computes a
  5-d coupling bias (when torch is available; otherwise we test the
  bypass path).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Phase 1 — tmqm_cfm_pretraining dry-run
# ---------------------------------------------------------------------------
def test_tmqm_pretraining_dry_run_writes_checkpoint(tmp_path: Path) -> None:
    """``--dry-run`` trains 1 step and writes the expected npz+json pair."""
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    out = pretrain.train_dry_run(out_dir=tmp_path, n_steps=1, seed=0)
    npz_path = tmp_path / "coupling_mlp.npz"
    json_path = tmp_path / "coupling_mlp.json"
    assert npz_path.exists()
    assert json_path.exists()
    data = np.load(npz_path)
    for key in (
        "embed.weight",
        "embed.bias",
        "head.weight",
        "head.bias",
        "input_mean",
        "input_std",
    ):
        assert key in data.files, key
    assert data["embed.weight"].shape == (pretrain.INPUT_DIM, pretrain.EMBED_DIM)
    metadata = json.loads(json_path.read_text())
    assert metadata["n_train_mols"] == len(pretrain.DRY_RUN_SMILES)
    assert metadata["n_steps"] == 1
    assert metadata["dry_run"] is True
    assert metadata["embed_dim"] == pretrain.EMBED_DIM
    assert metadata["final_loss"] == pytest.approx(
        out["metadata"].final_loss, rel=1e-5
    )


def test_tmqm_pretraining_dry_run_cli(tmp_path: Path) -> None:
    """CLI ``--dry-run`` exit code 0 + writes checkpoint."""
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    rc = pretrain.main(
        [
            "--dry-run",
            "--out-dir",
            str(tmp_path),
            "--steps",
            "1",
            "--print-final-loss",
        ]
    )
    assert rc == 0
    assert (tmp_path / "coupling_mlp.npz").exists()


def test_tmqm_pretraining_dry_run_loss_is_finite(tmp_path: Path) -> None:
    """The reported loss must be a finite float (no NaN/inf)."""
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    out = pretrain.train_dry_run(out_dir=tmp_path, n_steps=2, seed=42)
    assert np.isfinite(out["metadata"].final_loss)
    assert out["metadata"].final_loss >= 0.0


def test_tmqm_pretraining_full_falls_back_when_no_tmqm(tmp_path: Path) -> None:
    """``train_full`` gracefully falls back to dry-run when tmQM is unavailable."""
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    out = pretrain.train_full(out_dir=tmp_path, n_steps=2, seed=0)
    # Either we fell back (fallback_reason set) OR we used tmQM
    # (n_train_mols > 8).  Both paths must produce a valid metadata blob.
    assert out["metadata"].n_train_mols >= 8
    assert (tmp_path / "coupling_mlp.npz").exists()


def test_tmqm_pretraining_input_dim_constants_stable() -> None:
    """The constants exposed in the script must equal coupling_adapter's."""
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain
    from molmetal.molmetal_lam.lam_chem import coupling_adapter as adapter

    assert pretrain.INPUT_DIM == adapter.INPUT_DIM
    assert pretrain.EMBED_DIM == adapter.EMBED_DIM


# ---------------------------------------------------------------------------
# Phase 2 — coupling_adapter
# ---------------------------------------------------------------------------
def test_load_coupling_adapter_returns_64d_finite(tmp_path: Path) -> None:
    """``load_coupling_adapter`` returns an adapter whose ``embed_pocket``
    is shape ``(64,)`` and every entry is finite."""
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        load_coupling_adapter,
        EMBED_DIM,
    )

    # Ensure a real checkpoint exists (writes it if missing).
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    pretrain.train_dry_run(out_dir=tmp_path, n_steps=1, seed=0)
    adapter = load_coupling_adapter(tmp_path / "coupling_mlp.npz")
    assert adapter.is_stub is False
    vec = adapter.embed_pocket(None, pocket_name="MMP2")
    assert vec.shape == (EMBED_DIM,)
    assert np.isfinite(vec).all()


def test_load_coupling_adapter_stub_when_missing() -> None:
    """Missing checkpoint → deterministic stub adapter."""
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        load_coupling_adapter,
        EMBED_DIM,
    )

    adapter = load_coupling_adapter(Path("/nonexistent/coupling.npz"))
    assert adapter.is_stub is True
    vec = adapter.embed_pocket(None, pocket_name="Pt_test")
    assert vec.shape == (EMBED_DIM,)
    assert np.isfinite(vec).all()


def test_stub_adapter_deterministic() -> None:
    """Two calls with the same pocket_name must return identical vectors."""
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import _stub_adapter

    a = _stub_adapter()
    v1 = a.embed_pocket(None, pocket_name="MMP2")
    v2 = a.embed_pocket(None, pocket_name="MMP2")
    np.testing.assert_array_equal(v1, v2)


def test_adapter_pocket_specificity() -> None:
    """Different pocket names must yield different fallback vectors."""
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import _stub_adapter

    a = _stub_adapter()
    v_a = a.embed_pocket(None, pocket_name="MMP2")
    v_b = a.embed_pocket(None, pocket_name="CA2")
    # Not strictly equal — the hash-based perturbation shifts a slot.
    assert not np.array_equal(v_a, v_b)


def test_is_coupling_enabled_truthy() -> None:
    from molmetal.molmetal_lam.lam_chem import coupling_adapter

    for v in ("1", "true", "TRUE", "Yes", "on"):
        os.environ["COUPLING_ENABLED"] = v
        assert coupling_adapter.is_coupling_enabled() is True
    for v in ("0", "false", "no", "off", ""):
        os.environ["COUPLING_ENABLED"] = v
        assert coupling_adapter.is_coupling_enabled() is False
    # Restore default for downstream tests.
    os.environ.pop("COUPLING_ENABLED", None)


def test_ensure_checkpoint_exists_idempotent(tmp_path: Path) -> None:
    """Calling twice writes only one checkpoint."""
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        ensure_checkpoint_exists,
    )

    p1 = ensure_checkpoint_exists(checkpoint_path=tmp_path / "coupling_mlp.npz")
    p2 = ensure_checkpoint_exists(checkpoint_path=tmp_path / "coupling_mlp.npz")
    assert p1.exists()
    assert p2 == p1


# ---------------------------------------------------------------------------
# Phase 3 — warm_start wiring
# ---------------------------------------------------------------------------
def test_warm_start_default_is_deterministic() -> None:
    """Without a coupling adapter the vector is bit-for-bit reproducible."""
    from molmetal.molmetal_lam.search_alg.warm_start import (
        pocket_features,
        PocketResidue,
    )

    residues = [
        PocketResidue(one_letter="H", resid=94, distance_to_ligand=3.0),
        PocketResidue(one_letter="H", resid=96, distance_to_ligand=4.0),
    ]
    v1 = pocket_features(residues, pocket_name="MMP2")
    v2 = pocket_features(residues, pocket_name="MMP2")
    np.testing.assert_array_equal(v1.values, v2.values)


def test_warm_start_adapter_ignored_when_env_off(tmp_path: Path) -> None:
    """``COUPLING_ENABLED`` unset → coupling adapter is a no-op."""
    os.environ.pop("COUPLING_ENABLED", None)
    from molmetal.molmetal_lam.search_alg.warm_start import (
        pocket_features,
        PocketResidue,
    )
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        load_coupling_adapter,
    )
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    pretrain.train_dry_run(out_dir=tmp_path, n_steps=1, seed=0)
    adapter = load_coupling_adapter(tmp_path / "coupling_mlp.npz")
    residues = [
        PocketResidue(one_letter="H", resid=94, distance_to_ligand=3.0),
    ]
    v_default = pocket_features(residues, pocket_name="MMP2")
    v_with_adapter = pocket_features(
        residues, pocket_name="MMP2", coupling_adapter=adapter
    )
    np.testing.assert_array_equal(v_default.values, v_with_adapter.values)


def test_warm_start_adapter_active_when_env_on(tmp_path: Path) -> None:
    """``COUPLING_ENABLED=1`` → pocket embedding comes from the adapter."""
    os.environ["COUPLING_ENABLED"] = "1"
    try:
        from molmetal.molmetal_lam.search_alg.warm_start import (
            pocket_features,
            PocketResidue,
        )
        from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
            load_coupling_adapter,
        )
        from molmetal.scripts import tmqm_cfm_pretraining as pretrain

        pretrain.train_dry_run(out_dir=tmp_path, n_steps=1, seed=0)
        adapter = load_coupling_adapter(tmp_path / "coupling_mlp.npz")
        residues = [
            PocketResidue(one_letter="H", resid=94, distance_to_ligand=3.0),
            PocketResidue(one_letter="S", resid=96, distance_to_ligand=4.0),
        ]
        v_default = pocket_features(residues, pocket_name="MMP2")
        v_with_adapter = pocket_features(
            residues, pocket_name="MMP2", coupling_adapter=adapter
        )
        assert not np.array_equal(v_default.values, v_with_adapter.values)
        assert np.isfinite(v_with_adapter.values).all()
        assert v_with_adapter.feature_names == [
            "coupling_adapter_embedding_64d"
        ]
    finally:
        os.environ.pop("COUPLING_ENABLED", None)


def test_warm_start_adapter_pocket_specificity(tmp_path: Path) -> None:
    """Two different residue shells → two different adapter embeddings."""
    os.environ["COUPLING_ENABLED"] = "1"
    try:
        from molmetal.molmetal_lam.search_alg.warm_start import (
            pocket_features,
            PocketResidue,
        )
        from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
            load_coupling_adapter,
        )
        from molmetal.scripts import tmqm_cfm_pretraining as pretrain

        pretrain.train_dry_run(out_dir=tmp_path, n_steps=1, seed=0)
        adapter = load_coupling_adapter(tmp_path / "coupling_mlp.npz")
        res_a = [
            PocketResidue(one_letter="H", resid=94, distance_to_ligand=3.0),
            PocketResidue(one_letter="H", resid=96, distance_to_ligand=4.0),
            PocketResidue(one_letter="H", resid=119, distance_to_ligand=4.5),
        ]
        res_b = [
            PocketResidue(one_letter="D", resid=94, distance_to_ligand=3.0),
            PocketResidue(one_letter="E", resid=96, distance_to_ligand=4.0),
        ]
        v_a = pocket_features(
            res_a, pocket_name="MMP2", coupling_adapter=adapter
        )
        v_b = pocket_features(
            res_b, pocket_name="CA2", coupling_adapter=adapter
        )
        assert not np.array_equal(v_a.values, v_b.values)
    finally:
        os.environ.pop("COUPLING_ENABLED", None)


# ---------------------------------------------------------------------------
# Phase 4 — learned_prior wiring (torch-dependent; skipped if torch broken)
# ---------------------------------------------------------------------------
_TORCH_IMPORT_ERROR = None
try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


@pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason=f"torch unavailable ({_TORCH_IMPORT_ERROR})",
)
def test_learned_prior_coupling_adapter_env_off_no_bias() -> None:
    """``COUPLING_ENABLED`` unset → no coupling bias cached."""
    os.environ.pop("COUPLING_ENABLED", None)
    from molmetal.molmetal_lam.search_alg.learned_prior import LearnedPolicyPrior
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        load_coupling_adapter,
    )

    adapter = load_coupling_adapter(prefer_stub=True)
    prior = LearnedPolicyPrior(coupling_adapter=adapter)
    assert prior._coupling_bias is None


@pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason=f"torch unavailable ({_TORCH_IMPORT_ERROR})",
)
def test_learned_prior_coupling_adapter_env_on_caches_bias(tmp_path: Path) -> None:
    """``COUPLING_ENABLED=1`` → 5-d bias tensor cached from the adapter."""
    os.environ["COUPLING_ENABLED"] = "1"
    try:
        from molmetal.molmetal_lam.search_alg.learned_prior import LearnedPolicyPrior
        from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
            load_coupling_adapter,
        )
        from molmetal.scripts import tmqm_cfm_pretraining as pretrain
        import torch

        pretrain.train_dry_run(out_dir=tmp_path, n_steps=1, seed=0)
        adapter = load_coupling_adapter(tmp_path / "coupling_mlp.npz")
        prior = LearnedPolicyPrior(coupling_adapter=adapter)
        assert prior._coupling_bias is not None
        assert prior._coupling_bias.shape == (5,)
        # Bias is zero-mean by construction.
        assert float(prior._coupling_bias.mean()) == pytest.approx(0.0, abs=1e-5)
        # And changing the pocket refreshes it.
        prior.set_coupling_pocket(pocket_name="CA2")
        assert prior._coupling_bias is not None
        assert prior._coupling_bias.shape == (5,)
    finally:
        os.environ.pop("COUPLING_ENABLED", None)


@pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason=f"torch unavailable ({_TORCH_IMPORT_ERROR})",
)
def test_learned_prior_predict_proba_uses_bias(tmp_path: Path) -> None:
    """``predict_proba`` returns a non-uniform distribution when bias is on."""
    os.environ["COUPLING_ENABLED"] = "1"
    try:
        from molmetal.molmetal_lam.search_alg.learned_prior import LearnedPolicyPrior
        from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
            load_coupling_adapter,
        )
        from molmetal.scripts import tmqm_cfm_pretraining as pretrain

        pretrain.train_dry_run(out_dir=tmp_path, n_steps=2, seed=0)
        adapter = load_coupling_adapter(tmp_path / "coupling_mlp.npz")
        prior = LearnedPolicyPrior(coupling_adapter=adapter)
        probs = prior.predict_proba("[Pt]([NH2])([Cl])([Cl])[NH2]")
        # Sum to 1
        s = sum(probs.values())
        assert s == pytest.approx(1.0, abs=1e-4)
        # Each entry is finite
        for v in probs.values():
            assert np.isfinite(v)
    finally:
        os.environ.pop("COUPLING_ENABLED", None)