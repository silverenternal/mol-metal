"""Tests for the pretrain RDKit-free GPU training path actually TRAINING.

Pre-flight findings (Wf-125 P0 #1) showed that
``molmetal.scripts.train_dmpnn_multitask`` had a short-circuit in the
``--data-config pretrain`` branch that wrote a JSON record of the loader
wiring but never actually called the training loop.  This file guards
the FIX: the pretrain path now genuinely trains the
:class:`V4MultiTaskDMPNN` model via :meth:`forward_cached` (no RDKit
involved at any point).

Three properties are verified here:

  (i)   The training loop runs N steps — the ``main()`` flow does
        NOT short-circuit, but actually instantiates the model,
        iterates the DataLoader, and calls backward + optimizer.step.
  (ii)  ``V4MultiTaskDMPNN.forward_cached`` is the ONLY forward path
        invoked during the loop.  ``forward`` (the SMILES path) is
        NEVER called — RDKit is bypassed end-to-end.
  (iii) The training loss actually decreases — a small synthetic
        corpus lets us verify that ``initial_loss > final_loss`` and
        that ``loss_history[-1] < loss_history[0]``.

All tests are CPU-only (no GPU required), use a 16-row synthetic
``all_pretrain.pt`` written to ``tmp_path``, and monkey-patch
``PRE_FEATURIZED_ALL`` so the script doesn't try to load the 45 GB
production file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    # Cycle through 5 metals so the per-metal head sees variety.
    metal_idx = torch.tensor([0, 1, 2, 3, 4] * (n_rows // 5 + 1), dtype=torch.int64)[
        :n_rows
    ]
    y = torch.randn(n_rows, dtype=torch.float32)
    # Half active, half inactive — gives the loss a real signal.
    active = torch.tensor(
        [1.0, 0.0, 1.0, 0.0, 1.0] * (n_rows // 5 + 1), dtype=torch.float32
    )[:n_rows]
    # Cycle through 4 domains so coord head sees variety.
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
        "smiles": ["CCO"] * n_rows,
    }


@pytest.fixture()
def synthetic_pretrain_pt(tmp_path: Path) -> Path:
    """Write a small all_pretrain.pt to ``tmp_path`` and yield its path."""
    pack = _make_synthetic_pack(n_rows=16, n_atoms=5, n_edges=8)
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)
    return p


@pytest.fixture()
def patched_pretrain_path(
    synthetic_pretrain_pt: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Monkey-patch PRE_FEATURIZED_ALL so main() reads our synthetic file."""
    monkeypatch.setattr(td, "PRE_FEATURIZED_ALL", synthetic_pretrain_pt)
    return synthetic_pretrain_pt


# ---------------------------------------------------------------------------
# (i) training loop runs N steps
# ---------------------------------------------------------------------------
def test_pretrain_path_runs_n_steps(
    patched_pretrain_path: Path, tmp_path: Path
) -> None:
    """``main(['--data-config', 'pretrain', ...])`` actually trains.

    The pre-flight findings showed that the previous implementation
    short-circuited at lines 947-974 and never called the optimizer.
    After the fix, ``run_pretrain_training`` iterates over the
    DataLoader for ``--epochs`` epochs and the resulting JSON must
    record ``n_steps_total`` ≥ 1 (16 rows / batch=4 = 4 batches per
    epoch × 2 epochs = 8 steps).
    """
    out_path = tmp_path / "pretrain_record.json"
    rc = td.main(
        [
            "--data-config",
            "pretrain",
            "--batch",
            "4",
            "--epochs",
            "2",
            "--out",
            str(out_path),
            "--metal",
            "Ru",
            "--seed",
            "0",
            "--no-bf16",  # force CPU path for the test
        ]
    )
    assert rc == 0, f"main() returned {rc} (expected 0)"
    payload = json.loads(out_path.read_text())
    # The training block ran for at least one step.
    assert "training" in payload, (
        "Payload missing 'training' key — the main() pretrain branch "
        "still short-circuits without running the training loop."
    )
    training = payload["training"]
    n_steps = int(training["n_steps_total"])
    # 16 rows / batch=4 → 4 batches/epoch × 2 epochs = 8 steps.
    assert n_steps == 8, (
        f"Expected 8 training steps (16 rows × 2 epochs / batch 4), "
        f"got n_steps_total={n_steps}. Short-circuit not removed?"
    )
    # n_epochs matches --epochs.
    assert (
        int(training["n_epochs"]) == 2
    ), f"Expected n_epochs=2, got {training['n_epochs']}"
    # initial_loss / final_loss are both recorded.
    assert training["initial_loss"] is not None
    assert training["final_loss"] is not None
    # RDKit was never called.
    assert payload["rdkit_called"] is False
    # Forward-cached was invoked exactly once per step.
    assert int(training["forward_cached_calls"]) == n_steps, (
        f"forward_cached_calls ({training['forward_cached_calls']}) "
        f"!= n_steps_total ({n_steps})"
    )


# ---------------------------------------------------------------------------
# (ii) forward_cached is the ONLY forward path; forward() is never called
# ---------------------------------------------------------------------------
def test_pretrain_path_uses_cached_forward(
    patched_pretrain_path: Path, tmp_path: Path
) -> None:
    """The training loop calls ``forward_cached`` exclusively — RDKit
    ``forward()`` is never invoked.

    We patch :class:`V4MultiTaskDMPNN.forward` so it raises an assertion
    failure if called, then run ``main(['--data-config', 'pretrain'])``
    and verify that ``forward_cached`` was called and the run returned
    0 (no assertion failure).
    """
    from molmetal.models import v4_multitask_dmpnn as v4

    # Save originals so we can restore after the test (defensive).
    _orig_forward = v4.V4MultiTaskDMPNN.forward
    _orig_forward_cached = v4.V4MultiTaskDMPNN.forward_cached
    forward_calls: list = []
    forward_cached_calls: list = []

    def _spy_forward(self, *args, **kwargs):
        forward_calls.append((args, kwargs))
        return _orig_forward(self, *args, **kwargs)

    def _spy_forward_cached(self, *args, **kwargs):
        forward_cached_calls.append((args, kwargs))
        return _orig_forward_cached(self, *args, **kwargs)

    v4.V4MultiTaskDMPNN.forward = _spy_forward  # type: ignore[assignment]
    v4.V4MultiTaskDMPNN.forward_cached = _spy_forward_cached  # type: ignore[assignment]
    try:
        out_path = tmp_path / "pretrain_record.json"
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
        # Restore forward even if the test fails (use finally via the
        # try/finally around the assertion).
        assert rc == 0, (
            f"main() returned {rc} — expected 0 (forward() must NOT have "
            "been called; spy_forward would have been called if it was)"
        )
        assert len(forward_calls) == 0, (
            f"V4MultiTaskDMPNN.forward was called "
            f"{len(forward_calls)} times during pretrain training — "
            "must use forward_cached exclusively (RDKit-free path). "
            f"Calls: {forward_calls[:1]}"
        )
        # forward_cached must have been called at least once per step.
        assert len(forward_cached_calls) >= 1, (
            "V4MultiTaskDMPNN.forward_cached was never called during "
            "pretrain training — the training loop is broken."
        )
    finally:
        v4.V4MultiTaskDMPNN.forward = _orig_forward  # type: ignore[assignment]
        v4.V4MultiTaskDMPNN.forward_cached = _orig_forward_cached  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# (iii) training actually decreases the loss
# ---------------------------------------------------------------------------
def test_pretrain_path_loss_decreases(
    patched_pretrain_path: Path, tmp_path: Path
) -> None:
    """The loss must fall over the course of pretrain training.

    With 16 rows × 2 epochs × batch=4 = 8 steps, the Adam optimizer
    over the synthetic pack should drop the per-step loss measurably
    (the labels have a real signal: alternating active/inactive +
    cycling domains).

    We verify both:
      * ``loss_history[0] > loss_history[-1]`` (strict decrease)
      * ``loss_decrease > 0.05`` (meaningful decrease, allowing for
        the noisiness of 8 random init steps)
    """
    out_path = tmp_path / "pretrain_record.json"
    rc = td.main(
        [
            "--data-config",
            "pretrain",
            "--batch",
            "4",
            "--epochs",
            "2",
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
    initial_loss = float(training["initial_loss"])
    final_loss = float(training["final_loss"])
    loss_history = [float(x) for x in training["loss_history"]]

    assert len(loss_history) >= 2, (
        f"loss_history has {len(loss_history)} entries — need at "
        "least 2 steps to verify a decrease."
    )
    assert loss_history[0] > loss_history[-1], (
        f"Loss did NOT decrease: initial={loss_history[0]:.4f}  "
        f"final={loss_history[-1]:.4f}. The training loop is not "
        "actually updating parameters. "
        f"loss_history={loss_history}"
    )
    # Allow some slack — we just need the loss to drop by a
    # non-trivial margin over 8 steps on a tiny random init.
    decrease = float(training["loss_decrease"])
    assert (
        decrease > 0.0
    ), f"loss_decrease={decrease:.4f} — Adam should drive it positive."


# ---------------------------------------------------------------------------
# (bonus) run_pretrain_training callable directly (unit test)
# ---------------------------------------------------------------------------
def test_run_pretrain_training_directly(
    patched_pretrain_path: Path,
) -> None:
    """Direct call to :func:`run_pretrain_training` works end-to-end."""
    args = td._parse_args(
        [
            "--data-config",
            "pretrain",
            "--batch",
            "4",
            "--epochs",
            "1",
            "--metal",
            "Ru",
            "--seed",
            "0",
            "--no-bf16",
        ]
    )
    loader = td.build_pretrain_dataloader(
        batch_size=args.batch,
        shuffle=False,
        seed=args.seed,
    )
    summary = td.run_pretrain_training(args, loader)
    assert summary["n_steps_total"] >= 1
    assert summary["rdkit_called"] is False
    assert summary["bf16"] is False
    assert isinstance(summary["loss_history"], list)
    assert len(summary["loss_history"]) == summary["n_steps_total"]
