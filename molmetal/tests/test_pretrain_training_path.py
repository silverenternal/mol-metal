"""Tests for the pretrain RDKit-free GPU training path.

The pretrain path is wired into
``molmetal.scripts.train_dmpnn_multitask`` as a third ``--data-config``
option.  It loads the pre-featurized corpus produced by
``molmetal/scripts/pre_featurize_dmpnn.py`` and skips RDKit
``_featurize_batch`` entirely — the per-step CPU featurize bottleneck
(~186 ms at BATCH=256) is replaced by a TensorDataset index lookup.

These tests guard the contract:

  * The pre-featurized tensors load and the resulting DataLoader
    yields the documented 8-tuple batch layout
    ``(atom_features, edge_index, edge_attr, atom_mask, metal_idx,
       y, active, domain)``.
  * RDKit is NOT called during data loading — the loader never goes
    through ``Chem.MolFromSmiles`` or any RDKit featurizer.

Tests use a tiny synthetic ``.pt`` written to a temp file rather than
the 45 GB production file so they stay CPU-only and run in seconds.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports the script (parses args, exposes ``build_pretrain_dataloader``).
from molmetal.scripts import train_dmpnn_multitask as td


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_synthetic_pack(
    n_rows: int = 16,
    n_atoms: int = 5,
    n_edges: int = 8,
    atom_feat_dim: int = 39,
    edge_feat_dim: int = 6,
) -> dict:
    """Build a tiny ``all_pretrain.pt``-shaped dict for tests."""
    _set_seed(0)
    af = torch.randn(n_rows, n_atoms, atom_feat_dim, dtype=torch.float32)
    ei = torch.zeros(n_rows, 2, n_edges, dtype=torch.int64)
    ea = torch.randn(n_rows, n_edges, edge_feat_dim, dtype=torch.float32)
    mask = torch.ones(n_rows, n_atoms, dtype=torch.bool)
    metal_idx = torch.zeros(n_rows, dtype=torch.int64)
    y = torch.randn(n_rows, dtype=torch.float32)
    active = torch.zeros(n_rows, dtype=torch.float32)
    domain = torch.zeros(n_rows, dtype=torch.int64)
    smiles = ["CCO"] * n_rows
    return {
        "atom_features": af,
        "edge_index": ei,
        "edge_attr": ea,
        "atom_mask": mask,
        "metal_idx": metal_idx,
        "y": y,
        "active": active,
        "domain": domain,
        "smiles": smiles,
    }


@pytest.fixture()
def synthetic_pretrain_pt(tmp_path: Path) -> Path:
    """Write a small all_pretrain.pt to ``tmp_path`` and yield its path."""
    pack = _make_synthetic_pack(n_rows=16, n_atoms=5, n_edges=8)
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_pretrain_data_loads(synthetic_pretrain_pt: Path) -> None:
    """build_pretrain_dataloader reads the .pt and yields a 8-tuple batch."""
    loader = td.build_pretrain_dataloader(
        batch_size=4,
        shuffle=False,
        seed=42,
        path=synthetic_pretrain_pt,
    )
    assert isinstance(loader, DataLoader)
    # pin_memory=False contract: tensors already on host RAM, no DMA hop.
    assert loader.pin_memory is False
    # num_workers=0 contract: no forked featurization workers needed.
    assert loader.num_workers == 0
    # All 16 rows, batch=4 → 4 batches.
    assert len(loader) == 4
    batch = next(iter(loader))
    # 8-tuple layout: (atom_features, edge_index, edge_attr, atom_mask,
    #                  metal_idx, y, active, domain)
    # torch.utils.data.TensorDataset uses ``default_collate`` which
    # returns a ``list`` (not a tuple) — coerce to tuple before
    # unpacking so the assertion is robust across torch versions.
    if isinstance(batch, list):
        batch = tuple(batch)
    assert isinstance(batch, tuple)
    assert len(batch) == 8
    af, ei, ea, mask, mi, y, act, dom = batch
    assert af.shape == (4, 5, 39)
    assert ei.shape == (4, 2, 8)
    assert ea.shape == (4, 8, 6)
    assert mask.shape == (4, 5)
    assert mi.shape == (4,)
    assert y.shape == (4,)
    assert act.shape == (4,)
    assert dom.shape == (4,)


def test_pretrain_no_rdkit_called(synthetic_pretrain_pt: Path) -> None:
    """build_pretrain_dataloader must NOT call RDKit."""
    # We patch the RDKit symbols that the loader MIGHT reach for (Chem /
    # RDLogger / GraphFeaturizer / molmetal.data.featurize).  If any of
    # them get called we fail the test.
    with patch("rdkit.Chem.MolFromSmiles") as mock_mol:
        with patch("molmetal.data.featurize.GraphFeaturizer") as mock_feat:
            loader = td.build_pretrain_dataloader(
                batch_size=4,
                shuffle=False,
                seed=42,
                path=synthetic_pretrain_pt,
            )
            # Drain the loader to make sure no RDKit call sneaks in via
            # worker init / generator init / shuffle.
            for _ in loader:
                pass
            assert mock_mol.call_count == 0, (
                "RDKit Chem.MolFromSmiles was invoked during "
                "build_pretrain_dataloader — the pretrain path must "
                "be RDKit-free."
            )
            # GraphFeaturizer may be imported but should not be called.
            assert mock_feat.call_count == 0, (
                "GraphFeaturizer was instantiated during "
                "build_pretrain_dataloader — the pretrain path must "
                "read pre-featurized tensors only."
            )


def test_pretrain_max_rows_caps_corpus(synthetic_pretrain_pt: Path) -> None:
    """max_rows slices the corpus head so smoke runs stay cheap."""
    loader = td.build_pretrain_dataloader(
        batch_size=4,
        shuffle=False,
        seed=42,
        path=synthetic_pretrain_pt,
        max_rows=8,
    )
    # 8 rows × batch=4 → 2 batches.
    assert len(loader) == 2


def test_pretrain_missing_file_raises(tmp_path: Path) -> None:
    """A missing .pt must raise FileNotFoundError, not silently empty."""
    with pytest.raises(FileNotFoundError):
        td.build_pretrain_dataloader(
            batch_size=4,
            shuffle=False,
            seed=42,
            path=tmp_path / "does_not_exist.pt",
        )


def test_cli_data_config_accepts_pretrain() -> None:
    """The --data-config choices include 'pretrain'."""
    args = td._parse_args(["--data-config", "pretrain"])
    assert args.data_config == "pretrain"
    # default is still 'single' for backward compat
    args_default = td._parse_args([])
    assert args_default.data_config == "single"


def test_cli_batch_default_is_256() -> None:
    """--batch default is 256 and can be overridden (e.g. 2048 for Blackwell)."""
    args_default = td._parse_args([])
    assert args_default.batch == 256
    args_big = td._parse_args(["--batch", "2048"])
    assert args_big.batch == 2048


def test_pretrain_short_circuit_writes_json(
    synthetic_pretrain_pt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main(['--data-config', 'pretrain', '--out', ...])`` writes the
    pretrain_payload JSON and returns 0 after RUNNING the training loop
    (Wf-125 P0 #1 wiring: short-circuit was removed).

    Pre-flight: this test originally asserted the short-circuit
    behaviour (no training, just JSON write).  After the Wf-125 fix
    the pretrain path genuinely trains, so the JSON now also contains
    a ``training`` block with ``n_steps_total`` ≥ 1 and a recorded
    ``loss_decrease``.
    """
    # Substitute PRE_FEATURIZED_ALL with the synthetic file so we
    # don't try to load the 45 GB production file.
    monkeypatch.setattr(td, "PRE_FEATURIZED_ALL", synthetic_pretrain_pt)
    out_path = tmp_path / "pretrain_record.json"
    rc = td.main(
        [
            "--data-config",
            "pretrain",
            "--batch",
            "4",
            "--out",
            str(out_path),
            "--metal",
            "Ru",
            "--epochs",
            "1",
            "--no-bf16",
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text())
    # Loader config (legacy fields, kept for backward compat).
    assert payload["batch_size"] == 4
    assert payload["pin_memory"] is False
    assert payload["rdkit_called"] is False
    # NEW (Wf-125 fix): the training loop actually ran.
    assert (
        "training" in payload
    ), "payload missing 'training' block — short-circuit was not removed."
    training = payload["training"]
    assert int(training["n_steps_total"]) >= 1, (
        f"n_steps_total={training['n_steps_total']} < 1 — training loop "
        "did not run any steps."
    )
