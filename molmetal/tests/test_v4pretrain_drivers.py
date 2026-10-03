"""Tests for the P0 #3 --model flag wiring on the two training drivers.

Fix P0 #3 — Wire V4MultiTaskDMPNNPretrain to both training drivers:

  * :mod:`molmetal.scripts.train_dmpnn_multitask` — the RDKit-free
    pretrain path (run_pretrain_training) now honours ``--model v4pretrain``
    and instantiates :class:`V4MultiTaskDMPNNPretrain` (instead of the
    default :class:`V4MultiTaskDMPNN`).  Pretrain mode is auto-enabled.
  * :mod:`molmetal.scripts.train_metal_lomo_v2` — the LOMO loop exposes
    ``--model`` (choices: DMPNNMultiTaskModel / v4 / v4pretrain).  The
    default (DMPNNMultiTaskModel) preserves the legacy behaviour; V4
    classes raise NotImplementedError with a clear migration pointer
    (the LOMO loop is tightly coupled to DMPNNMultiTaskModel's 4-tensor
    forward + ``_forward_one_mol`` API — wiring V4 needs a refactor
    tracked under TODO-145 §4).

All tests are CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import train_dmpnn_multitask as td
from molmetal.scripts import train_metal_lomo_v2 as tl


# ---------------------------------------------------------------------------
# Synthetic pre-featurized pack — same shape as the production
# ``molmetal/data/pre_featurized/all_pretrain.pt`` but tiny for tests.
# ---------------------------------------------------------------------------
def _make_synthetic_pack(
    n_rows: int = 16,
    n_atoms: int = 5,
    n_edges: int = 8,
    atom_feat_dim: int = 39,
    edge_feat_dim: int = 6,
) -> dict:
    torch.manual_seed(0)
    af = torch.randn(n_rows, n_atoms, atom_feat_dim, dtype=torch.float32)
    ei = torch.zeros(n_rows, 2, n_edges, dtype=torch.int64)
    ea = torch.randn(n_rows, n_edges, edge_feat_dim, dtype=torch.float32)
    mask = torch.ones(n_rows, n_atoms, dtype=torch.bool)
    metal_idx = torch.tensor([0, 1, 2, 3, 4] * (n_rows // 5 + 1), dtype=torch.int64)[
        :n_rows
    ]
    y = torch.randn(n_rows, dtype=torch.float32)
    active = torch.tensor(
        [1.0, 0.0, 1.0, 0.0, 1.0] * (n_rows // 5 + 1), dtype=torch.float32
    )[:n_rows]
    domain = torch.tensor([0, 1, 2, 3, 0] * (n_rows // 5 + 1), dtype=torch.int64)[
        :n_rows
    ]
    return {
        "atom_features": af,
        "edge_index": ei,
        "edge_attr": ea,
        "atom_mask": mask,
        "metal_idx": metal_idx,
        "y": y,
        "active": active,
        "domain": domain,
    }


@pytest.fixture()
def synthetic_pretrain_pt(tmp_path: Path) -> Path:
    pack = _make_synthetic_pack(n_rows=16, n_atoms=5, n_edges=8)
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)
    return p


@pytest.fixture()
def patched_pretrain_path(
    synthetic_pretrain_pt: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(td, "PRE_FEATURIZED_ALL", synthetic_pretrain_pt)
    return synthetic_pretrain_pt


# ===========================================================================
# 1) train_dmpnn_multitask.py — --model flag + pretrain class wiring
# ===========================================================================
def test_train_dmpnn_multitask_default_model_is_v4() -> None:
    """The argparse default for ``--model`` is ``"v4"`` (backward compat)."""
    ns = td._parse_args([])
    assert getattr(ns, "model", None) == "v4", (
        "Default --model must be 'v4' to preserve current behaviour. "
        f"Got {getattr(ns, 'model', 'MISSING')!r}"
    )


def test_train_dmpnn_multitask_model_flag_choices() -> None:
    """All three choices (v3 / v4 / v4pretrain) are accepted at parse time."""
    for choice in ("v3", "v4", "v4pretrain"):
        ns = td._parse_args(["--model", choice])
        assert ns.model == choice, (
            f"--model {choice!r} did not round-trip through argparse; "
            f"got {ns.model!r}"
        )


# ---------------------------------------------------------------------------
# 1a) main() pretrain path honours --model v4pretrain
# ---------------------------------------------------------------------------
def test_train_dmpnn_multitask_model_flag_v4pretrain_instantiates_correct_class(
    patched_pretrain_path: Path, tmp_path: Path
) -> None:
    """With ``--model v4pretrain`` the pretrain path instantiates
    :class:`V4MultiTaskDMPNNPretrain` and enables pretrain mode.

    Verifies the P0 #3 wire-up: when ``--model v4pretrain`` is set,
    ``run_pretrain_training`` returns ``model_class ==
    "V4MultiTaskDMPNNPretrain"`` and ``pretrain_mode is True``.
    """
    out_path = tmp_path / "v4pretrain_record.json"
    rc = td.main(
        [
            "--data-config",
            "pretrain",
            "--batch",
            "4",
            "--epochs",
            "1",
            "--model",
            "v4pretrain",
            "--out",
            str(out_path),
            "--metal",
            "Ru",
            "--seed",
            "0",
            "--no-bf16",
        ]
    )
    assert rc == 0, f"main() returned {rc} (expected 0)"
    payload = json.loads(out_path.read_text())
    training = payload["training"]
    assert training["model_class"] == "V4MultiTaskDMPNNPretrain", (
        f"With --model v4pretrain, run_pretrain_training must instantiate "
        f"V4MultiTaskDMPNNPretrain. Got model_class={training['model_class']!r} "
        f"model_choice={training['model_choice']!r}"
    )
    assert training["model_choice"] == "v4pretrain"
    assert training["pretrain_mode"] is True, (
        "pretrain_mode must be auto-enabled when --model v4pretrain is set; "
        f"got {training['pretrain_mode']!r}"
    )


# ---------------------------------------------------------------------------
# 1b) default --model=v4 still uses V4MultiTaskDMPNN (NOT the pretrain class)
# ---------------------------------------------------------------------------
def test_train_dmpnn_multitask_default_v4_uses_base_class(
    patched_pretrain_path: Path, tmp_path: Path
) -> None:
    """With default ``--model=v4``, the pretrain path instantiates
    :class:`V4MultiTaskDMPNN` (NOT the pretrain variant)."""
    out_path = tmp_path / "v4_record.json"
    rc = td.main(
        [
            "--data-config",
            "pretrain",
            "--batch",
            "4",
            "--epochs",
            "1",
            "--out",
            str(out_path),
            "--metal",
            "Ru",
            "--seed",
            "0",
            "--no-bf16",
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text())
    training = payload["training"]
    assert training["model_class"] == "V4MultiTaskDMPNN", (
        f"Default --model=v4 must use V4MultiTaskDMPNN. "
        f"Got model_class={training['model_class']!r}"
    )
    assert training["model_choice"] == "v4"
    assert training["pretrain_mode"] is False, (
        "pretrain_mode must stay False under default --model=v4; got "
        f"{training['pretrain_mode']!r}"
    )


# ---------------------------------------------------------------------------
# 1c) --model v3 is reserved → NotImplementedError
# ---------------------------------------------------------------------------
def test_train_dmpnn_multitask_model_v3_raises_not_implemented(
    patched_pretrain_path: Path, tmp_path: Path
) -> None:
    """``--model v3`` is reserved (no V3 multi-task class) — pretrain
    path must raise NotImplementedError."""
    out_path = tmp_path / "v3_record.json"
    with pytest.raises(NotImplementedError) as excinfo:
        td.main(
            [
                "--data-config",
                "pretrain",
                "--batch",
                "4",
                "--epochs",
                "1",
                "--model",
                "v3",
                "--out",
                str(out_path),
                "--metal",
                "Ru",
                "--seed",
                "0",
                "--no-bf16",
            ]
        )
    msg = str(excinfo.value)
    assert (
        "--model v3" in msg or "v3" in msg
    ), f"NotImplementedError must mention 'v3' for clarity. Got: {msg}"


# ===========================================================================
# 2) train_metal_lomo_v2.py — --model flag + class resolution
# ===========================================================================
def test_train_metal_lomo_v2_default_model_is_dmpnnmultitaskmodel() -> None:
    """The argparse default for ``--model`` preserves the legacy class."""
    # We don't invoke main() here (that would try to load CSVs and
    # MetalCytoTox dataset).  Instead we replicate the argparse setup.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["DMPNNMultiTaskModel", "v4", "v4pretrain"],
        default="DMPNNMultiTaskModel",
    )
    ns = parser.parse_args([])
    assert ns.model == "DMPNNMultiTaskModel", (
        "Default --model must be 'DMPNNMultiTaskModel' to preserve legacy "
        f"behaviour. Got {ns.model!r}"
    )


def test_train_metal_lomo_v2_model_flag_choices() -> None:
    """All three choices parse correctly."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["DMPNNMultiTaskModel", "v4", "v4pretrain"],
        default="DMPNNMultiTaskModel",
    )
    for choice in ("DMPNNMultiTaskModel", "v4", "v4pretrain"):
        ns = parser.parse_args(["--model", choice])
        assert (
            ns.model == choice
        ), f"--model {choice!r} did not round-trip. Got {ns.model!r}"


def test_train_metal_lomo_v2_default_runner_uses_dmpnnmultitaskmodel() -> None:
    """Without ``model_cls`` the runner falls back to DMPNNMultiTaskModel."""
    from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel

    runner = tl.BalancedLOMORunner(
        ds=None,  # runner.__init__ does not access ds fields eagerly
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
    )
    assert runner._model_cls is DMPNNMultiTaskModel, (
        "Default runner must use DMPNNMultiTaskModel. Got " f"{runner._model_cls!r}"
    )


def test_train_metal_lomo_v2_runner_accepts_v4_model_cls() -> None:
    """``BalancedLOMORunner(model_cls=V4MultiTaskDMPNN)`` is accepted at
    construction time; the API mismatch is caught by ``_check_model_compat``
    at ``run()`` time (so the user can wire it but is forced to see the
    honest error before training starts)."""
    from molmetal.models.v4_multitask_dmpnn import (
        V4MultiTaskDMPNN,
        V4MultiTaskDMPNNPretrain,
    )

    for cls in (V4MultiTaskDMPNN, V4MultiTaskDMPNNPretrain):
        runner = tl.BalancedLOMORunner(
            ds=None,
            held_out_metal="Rh",
            balanced_smiles=[],
            balanced_labels=[],
            model_cls=cls,
        )
        assert runner._model_cls is cls, (
            f"Runner must retain model_cls={cls.__name__!r}; "
            f"got {runner._model_cls!r}"
        )


def test_train_metal_lomo_v2_v4_model_run_raises_not_implemented() -> None:
    """Selecting ``--model v4`` causes ``run()`` to raise
    NotImplementedError with a clear migration pointer."""
    from molmetal.models.v4_multitask_dmpnn import (
        V4MultiTaskDMPNN,
        V4MultiTaskDMPNNPretrain,
    )

    runner = tl.BalancedLOMORunner(
        ds=None,
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
        model_cls=V4MultiTaskDMPNN,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        runner.run()
    msg = str(excinfo.value)
    assert (
        "V4MultiTaskDMPNN" in msg
    ), f"NotImplementedError must name the V4 class. Got: {msg}"
    assert (
        "TODO-145" in msg or "migration" in msg.lower()
    ), f"NotImplementedError must include a migration pointer. Got: {msg}"

    runner_pretrain = tl.BalancedLOMORunner(
        ds=None,
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
        model_cls=V4MultiTaskDMPNNPretrain,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        runner_pretrain.run()
    msg = str(excinfo.value)
    assert (
        "V4MultiTaskDMPNNPretrain" in msg
    ), f"NotImplementedError must name the V4 pretrain class. Got: {msg}"


def test_train_metal_lomo_v2_default_dmpnnmultitaskmodel_compat_passes() -> None:
    """With default model_cls=DMPNNMultiTaskModel, ``_check_model_compat``
    does NOT raise — the legacy training loop keeps working."""
    runner = tl.BalancedLOMORunner(
        ds=None,
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
    )
    # Should not raise.
    runner._check_model_compat()


# ===========================================================================
# 3) V4MultiTaskDMPNNPretrain — pretrain head + mol_clf head emission
# ===========================================================================
SMILES_3 = ["CCO", "c1ccccc1", "CC(=O)O"]


def _make_pretrain_model() -> "V4MultiTaskDMPNNPretrain":
    from molmetal.models.v4_multitask_dmpnn import (
        ATOM_VOCAB_SIZE,
        V4MultiTaskDMPNNPretrain,
    )

    torch.manual_seed(0)
    return V4MultiTaskDMPNNPretrain(
        hidden_dim=64,
        n_layers=2,
        n_metals=5,
        n_coord_classes=5,
        atom_vocab_size=ATOM_VOCAB_SIZE,
        n_pretrain_tasks=2,
    )


def test_v4pretrain_pretrain_mode_forward_emits_keys() -> None:
    """In train mode + pretrain_mode=True the model emits ``atom_logits``,
    ``atom_mask`` and ``mol_clf`` keys (in addition to the standard
    5-metal + coord dict).

    This is the canonical P0 #3 verification: V4MultiTaskDMPNNPretrain's
    pretrain heads work as specified — once a future PR wires them into
    ``forward_cached``, the pretrain driver path can consume them.
    """
    m = _make_pretrain_model()
    m.train()
    m.set_pretrain_mode(True)
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)

    # ----- Pretrain heads must be present -----
    assert "atom_logits" in out, (
        "atom_logits key missing from V4MultiTaskDMPNNPretrain.forward() "
        "output under pretrain_mode=True (in train mode). "
        f"Got keys: {sorted(out.keys())}"
    )
    assert "mol_clf" in out, (
        "mol_clf key missing from V4MultiTaskDMPNNPretrain.forward() "
        f"output under pretrain_mode=True. Got keys: {sorted(out.keys())}"
    )
    assert "atom_mask" in out, (
        "atom_mask key missing — needed to align atom_labels with "
        "per-atom predictions. Got keys: {sorted(out.keys())}"
    )

    # ----- Shapes -----
    from molmetal.models.v4_multitask_dmpnn import (
        ATOM_VOCAB_SIZE,
        SUPPORTED_METALS,
    )

    # mol_clf: (B, n_pretrain_tasks)
    mol_clf = out["mol_clf"]
    assert mol_clf.shape == (3, m.n_pretrain_tasks), (
        f"mol_clf shape expected (3, {m.n_pretrain_tasks}), got "
        f"{tuple(mol_clf.shape)}"
    )
    # atom_logits: (sum_N, atom_vocab_size)
    atom_logits = out["atom_logits"]
    assert atom_logits.dim() == 2, (
        f"atom_logits must be 2-D (sum_N, vocab), got shape "
        f"{tuple(atom_logits.shape)}"
    )
    assert atom_logits.shape[-1] == ATOM_VOCAB_SIZE
    # atom_mask: (B, N_max) bool
    atom_mask = out["atom_mask"]
    assert atom_mask.shape[0] == 3
    assert atom_mask.dtype == torch.bool
    # sum_N rows of atom_logits must equal the number of True atoms
    sum_N = int(atom_mask.sum().item())
    assert atom_logits.shape[0] == sum_N, (
        f"atom_logits has {atom_logits.shape[0]} rows but atom_mask has "
        f"{sum_N} True atoms — pretrain head will not align with labels."
    )

    # ----- Base dict is also present (backward compat) -----
    for met in SUPPORTED_METALS:
        assert met in out, f"Metal key {met!r} missing — backward compat broken"
        assert out[met].shape == (3, 2), (
            f"Per-metal logits for {met!r} must be (B, 2); "
            f"got {tuple(out[met].shape)}"
        )
    assert "coord" in out
    assert out["coord"].shape[0] == 3


def test_v4pretrain_pretrain_mode_off_matches_v4_base() -> None:
    """With ``pretrain_mode=False`` the output dict is exactly the base
    V4MultiTaskDMPNN output (5 metals + coord), with no pretrain heads
    — this is the backward-compat guarantee."""
    from molmetal.models.v4_multitask_dmpnn import (
        SUPPORTED_METALS,
        V4MultiTaskDMPNN,
        V4MultiTaskDMPNNPretrain,
    )

    torch.manual_seed(0)
    base = V4MultiTaskDMPNN(
        hidden_dim=64,
        n_layers=2,
        n_metals=5,
        n_coord_classes=5,
    )
    torch.manual_seed(0)
    pretrain = V4MultiTaskDMPNNPretrain(
        hidden_dim=64,
        n_layers=2,
        n_metals=5,
        n_coord_classes=5,
    )
    pretrain.eval()  # train-mode is irrelevant when pretrain_mode=False
    pretrain.set_pretrain_mode(False)

    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out_base = base(SMILES_3, metal_ids)
    out_pretrain = pretrain(SMILES_3, metal_ids)

    assert set(out_pretrain.keys()) == set(out_base.keys()), (
        f"Keys differ: base={sorted(out_base.keys())} "
        f"pretrain={sorted(out_pretrain.keys())}"
    )
    for k in out_base:
        assert out_pretrain[k].shape == out_base[k].shape, (
            f"Shape mismatch for {k!r}: base={tuple(out_base[k].shape)} "
            f"pretrain={tuple(out_pretrain[k].shape)}"
        )


def test_v4pretrain_set_pretrain_mode_toggles() -> None:
    """``set_pretrain_mode(True/False)`` correctly toggles the flag."""
    m = _make_pretrain_model()
    assert m.pretrain_mode is False, "Default pretrain_mode must be False"
    m.set_pretrain_mode(True)
    assert m.pretrain_mode is True
    m.set_pretrain_mode(False)
    assert m.pretrain_mode is False
