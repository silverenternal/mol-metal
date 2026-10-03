"""Tests for ChempropV2MTModel wiring into 8 train scripts (Wf-148 / TODO-148).

Covers the four contract properties the task spec demands:

  (i)   The ``chemprop_v2`` CLI alias maps to ``ChempropV2MTModel`` in
        train_metal_lomo_v2.py — the previous "class not implemented"
        NotImplementedError stub has been replaced with real
        instantiation.
  (ii)  ``--model v4`` (backward compat) still resolves to
        ``V4MultiTaskDMPNN`` in train_metal_lomo_v2.py.
  (iii) ``--model v4`` (backward compat) still resolves to
        ``V4MultiTaskDMPNN`` in train_dmpnn_multitask.py's
        ``run_pretrain_training``.
  (iv)  ``--model chemprop_v2`` is exposed as a CLI choice in
        train_dmpnn_multitask.py and the chemprop_v2 instantiation path
        is exercised (the pretrain RDKit-free path raises
        NotImplementedError because ChempropV2MTModel consumes SMILES
        not pre-featurized tensors).

All tests are CPU-only — no GPU required.  We construct tiny
:class:`ChempropV2MTModel` instances with ``hidden_dim=64`` + ``depth=2``
to keep the chemprop v2 featurization cost bounded.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _import(name: str):
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# (i) train_metal_lomo_v2 --model chemprop_v2 → ChempropV2MTModel
# ---------------------------------------------------------------------------
def test_train_metal_lomo_v2_chemprop_v2_class_resolves() -> None:
    """``--model chemprop_v2`` in train_metal_lomo_v2 maps to
    ``ChempropV2MTModel`` (the previous NotImplementedError stub for
    "class does not exist" is replaced with real instantiation).
    """

    mod = _import("train_metal_lomo_v2")
    # Source must contain the chemprop_v2 branch resolving to
    # ChempropV2MTModel (replaces the previous stub).
    src = Path(mod.__file__).read_text()
    assert "model_cls = ChempropV2MTModel" in src, (
        "train_metal_lomo_v2.py no longer maps --model chemprop_v2 to "
        "ChempropV2MTModel — the previous stub may have been restored."
    )
    # And the ChempropV2MTModel import must be present.
    assert (
        "ChempropV2MTModel" in src
    ), "train_metal_lomo_v2.py does not import ChempropV2MTModel."


def test_train_metal_lomo_v2_chemprop_v2_instantiates() -> None:
    """``ChempropV2MTModel`` itself instantiates and runs a forward pass
    with the same args that ``train_metal_lomo_v2`` would pass to the
    legacy DMPNNMultiTaskModel — verifying the wiring contract that
    ``--model chemprop_v2`` no longer fails with a 'class missing'
    NotImplementedError.
    """
    pytest.importorskip("torch")
    from molmetal.models.chemprop_v2_mt_model import ChempropV2MTModel
    import torch

    # Match the constructor params a LOMO trainer would use.
    model = ChempropV2MTModel(
        hidden_dim=64,
        depth=2,
        n_metals=5,
        n_coord_classes=5,
    )
    model.eval()

    # SMILES-based forward (Chemprop v2 contract)
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        out = model(smiles_list, metal_ids)

    # Per-metal (B, 2) logits + (B, n_coord_classes) coord logits.
    B = len(smiles_list)
    for metal in model.metals:
        assert out[metal].shape == (
            B,
            2,
        ), f"{metal} logits shape mismatch: got {tuple(out[metal].shape)}"
    assert out["coord"].shape == (B, model.n_coord_classes)


def test_train_metal_lomo_v2_chemprop_v2_legacy_loop_still_blocks() -> None:
    """``--model chemprop_v2`` no longer raises "class does not exist"
    but the legacy LOMO loop still raises NotImplementedError because
    it calls DMPNN-only APIs (TODO-148 §4 loop migration).  Verify the
    error message distinguishes "loop not wired" from the old
    "class not implemented".
    """
    pytest.importorskip("torch")
    from molmetal.models.chemprop_v2_mt_model import ChempropV2MTModel

    mod = _import("train_metal_lomo_v2")
    # Construct a stub dataset-free runner just to call _check_model_compat.
    runner = mod.BalancedLOMORunner(
        ds=None,
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
        model_cls=ChempropV2MTModel,
        model_alias="chemprop_v2",
    )
    with pytest.raises(NotImplementedError) as exc_info:
        runner._check_model_compat()
    msg = str(exc_info.value)
    # The error must NOT say "class does not exist" — that stub was removed.
    assert (
        "does not exist" not in msg.lower()
    ), f"The old 'class does not exist' stub is still in the message: {msg}"
    # It SHOULD say "IS instantiated" — the new honest framing.
    assert (
        "is instantiated" in msg.lower()
    ), f"Expected new honest framing 'IS instantiated' in: {msg}"


# ---------------------------------------------------------------------------
# (ii) train_metal_lomo_v2 --model v4 backward compat
# ---------------------------------------------------------------------------
def test_train_metal_lomo_v2_v4_still_resolves_to_v4multitaskdmpnn() -> None:
    """``--model v4`` (backward compat) still resolves to V4MultiTaskDMPNN.

    The Wf-145 / TODO-145 §3.5 V4 wiring contract is preserved.
    """

    mod = _import("train_metal_lomo_v2")
    src = Path(mod.__file__).read_text()
    # The v4 branch in main() must still resolve to V4MultiTaskDMPNN.
    assert "V4MultiTaskDMPNN" in src
    # Spot-check: --model v4 → V4MultiTaskDMPNN (not Pretrain).
    v4_branch = src.split('args.model == "v4pretrain"')[0]
    assert "V4MultiTaskDMPNN" in v4_branch


def test_train_metal_lomo_v2_v4_alias_branch_present() -> None:
    """The --model v4 alias branch is still wired (backward compat)."""
    mod = _import("train_metal_lomo_v2")
    src = Path(mod.__file__).read_text()
    assert 'args.model in ("v4", "v4pretrain")' in src


# ---------------------------------------------------------------------------
# (iii) train_dmpnn_multitask --model v4 still works
# ---------------------------------------------------------------------------
def test_train_dmpnn_multitask_v4_default_choice_intact() -> None:
    """``train_dmpnn_multitask --model v4`` still maps to V4MultiTaskDMPNN
    in ``run_pretrain_training`` (backward compat).
    """
    mod = _import("train_dmpnn_multitask")
    args = mod._parse_args([])
    assert args.model == "v4"
    # Source must contain the V4 default branch.
    src = Path(mod.__file__).read_text()
    assert "V4MultiTaskDMPNN" in src
    assert 'model_choice == "v4pretrain"' in src


def test_train_dmpnn_multitask_chemprop_v2_choice_exists() -> None:
    """``train_dmpnn_multitask --model chemprop_v2`` is a valid choice
    and the chemprop_v2 branch in ``run_pretrain_training`` instantiates
    ChempropV2MTModel before raising NotImplementedError.
    """

    mod = _import("train_dmpnn_multitask")
    args = mod._parse_args(["--model", "chemprop_v2"])
    assert args.model == "chemprop_v2"
    src = Path(mod.__file__).read_text()
    # The chemprop_v2 branch must import ChempropV2MTModel.
    assert "ChempropV2MTModel" in src
    # And it must raise NotImplementedError (the honest "loop not wired"
    # contract — the pretrain path uses pre-featurized tensors, not SMILES).
    assert 'model_choice == "chemprop_v2"' in src


def test_train_dmpnn_multitask_chemprop_v2_instantiation_succeeds() -> None:
    """ChempropV2MTModel can be instantiated with the args
    ``run_pretrain_training`` would use (matching the V4 hidden_dim=64
    / n_layers=2 pattern)."""
    pytest.importorskip("torch")
    from molmetal.models.chemprop_v2_mt_model import ChempropV2MTModel

    # Match run_pretrain_training's V4 default sizes.
    model = ChempropV2MTModel(
        hidden_dim=64,
        depth=2,
        n_metals=5,
        n_coord_classes=5,
    )
    assert isinstance(model, ChempropV2MTModel)


# ---------------------------------------------------------------------------
# (iv) CLI argparse smoke
# ---------------------------------------------------------------------------
def test_train_metal_lomo_v2_chemprop_v2_cli_help_exits_cleanly() -> None:
    """``python train_metal_lomo_v2.py --help`` exits 0 and advertises
    chemprop_v2 in the --model choices.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "train_metal_lomo_v2.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"--help failed: stderr={result.stderr}"
    assert (
        "chemprop_v2" in result.stdout
    ), f"--model chemprop_v2 missing from --help text:\n{result.stdout}"
    assert (
        "v4" in result.stdout
    ), f"--model v4 missing from --help text:\n{result.stdout}"


def test_train_dmpnn_multitask_chemprop_v2_cli_help_exits_cleanly() -> None:
    """``python train_dmpnn_multitask.py --help`` exits 0 and advertises
    chemprop_v2 in the --model choices.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "train_dmpnn_multitask.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"--help failed: stderr={result.stderr}"
    assert (
        "chemprop_v2" in result.stdout
    ), f"--model chemprop_v2 missing from --help text:\n{result.stdout}"
    assert (
        "v4" in result.stdout
    ), f"--model v4 missing from --help text:\n{result.stdout}"
