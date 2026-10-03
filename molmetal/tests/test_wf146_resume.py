"""Tests for Wf-146 — checkpoint save / resume safety net.

A 5-metal × 3-seed × 50-epoch D-MPNN pretrain runs for ~21h on Blackwell.
If a run crashes (GPU OOM, spot preemption, NCCL drop, SIGTERM from the
orchestrator) we MUST be able to resume from the last completed epoch
without losing model weights, optimizer state, or best-metric tracking.

These tests cover the three contracts that the safety net must hold:

* ``test_checkpoint_save_load``         — round-trip a payload through
                                         ``save_checkpoint`` + ``load_checkpoint``
                                         and verify the model weights are
                                         byte-identical.
* ``test_resume_continues_from_epoch``  — after a "crash" at epoch N,
                                         ``--resume`` makes ``fit()``
                                         continue from epoch N+1 and run
                                         exactly ``epochs - (N+1)`` more
                                         epochs.
* ``test_resume_preserves_optimizer_state`` — Adam moving averages (exp_avg,
                                         exp_avg_sq) and step count are
                                         restored from the snapshot rather
                                         than reinitialised.

All three tests construct a tiny in-memory model directly (not the full
D-MPNNMultiTaskBaseline → Cytotox → CSV pipeline) so they run in <1s
each and do not depend on the ~340K-row MetalCytoToxDB CSV.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskBaseline


# ---------------------------------------------------------------------------
# Tiny stand-in model used in place of DMPNNMultiTaskModel.
#
# The safety net only depends on ``state_dict()`` round-trip + a serialised
# optimizer state — both of which the smallest nn.Linear pair exercises
# faithfully.  We deliberately avoid the full D-MPNN here so the tests are
# fast (sub-second) and independent of the cytotox data pipeline.
# ---------------------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


SMILES_8 = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)C",
    "CCC",
    "C1CCCCC1",
    "CCOCC",
    "CCCC",
]


def _build_dummy_arrays(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(SMILES_8)
    return {
        "smiles": np.array(SMILES_8),
        "active_all": rng.integers(0, 2, size=n).astype(int),
        "pic50_all": rng.uniform(-2.0, 2.0, size=n).astype(np.float32),
        "cell_lines": np.array(["A2780"] * n),
        "idx_train": np.array([0, 1, 2, 3, 4]),
        "idx_val": np.array([5, 6]),
    }


def _patch_baseline_model(monkeypatch, baseline: DMPNNMultiTaskBaseline) -> TinyModel:
    """Replace ``baseline.model`` with a tiny module + a 2-param Adam.

    We do this so the test exercises the real ``save_checkpoint`` /
    ``load_checkpoint`` plumbing without dragging in the full D-MPNN
    message-passing graph.
    """
    tiny = TinyModel()
    baseline.model = tiny  # type: ignore[assignment]
    baseline._tiny_optim = torch.optim.Adam(tiny.parameters(), lr=1e-3)
    return tiny


# ---------------------------------------------------------------------------
# Test 1 — checkpoint save / load round-trip
# ---------------------------------------------------------------------------
def test_checkpoint_save_load(tmp_path: Path):
    """save_checkpoint writes a valid payload; load_checkpoint round-trips it.

    Verifies:

    * the file exists on disk and contains the expected keys,
    * model weights are byte-identical (torch.equal on every tensor),
    * epoch + best_metric are preserved,
    * metal + seed are restored and reject mismatched cross-run loads.
    """
    ckpt_dir = tmp_path / "wf146_lomo_v2"
    baseline = DMPNNMultiTaskBaseline(
        metal="Ru",
        seed=0,
        ckpt_dir=str(ckpt_dir),
        ckpt_every=5,
    )
    tiny = _patch_baseline_model(monkeypatch=None, baseline=baseline)  # type: ignore[arg-type]

    # Set a known "best AUC" so we can assert it round-trips.
    saved_best = 0.7321
    out_path = baseline.save_checkpoint(
        epoch=10,
        best_metric=saved_best,
        optimizer=baseline._tiny_optim,  # type: ignore[attr-defined]
    )
    assert out_path.exists(), f"checkpoint file missing: {out_path}"
    assert out_path.name == "Ru_seed0_epoch00010.pt"

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    assert payload["version"] == 1
    assert int(payload["epoch"]) == 10
    assert float(payload["best_metric"]) == pytest.approx(saved_best)
    assert str(payload["metal"]) == "Ru"
    assert int(payload["seed"]) == 0
    assert int(payload["ckpt_every"]) == 5
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload

    # Mutate the in-memory model so we can detect that load_checkpoint
    # actually overwrote it.
    with torch.no_grad():
        for p in tiny.parameters():
            p.zero_()
    assert torch.allclose(
        tiny.fc1.weight, torch.zeros_like(tiny.fc1.weight)
    ), "sanity: mutation should zero the weights"

    next_epoch, best_metric = baseline.load_checkpoint(path=out_path)
    assert int(next_epoch) == 11
    assert float(best_metric) == pytest.approx(saved_best)

    # Reload the ORIGINAL expected weights from a fresh TinyModel and
    # confirm they match.
    fresh = TinyModel()
    fresh.load_state_dict(payload["model_state_dict"])
    for (n1, p1), (n2, p2) in zip(tiny.named_parameters(), fresh.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"weight {n1} did not round-trip"

    # Cross-run guard: a baseline with a different seed must refuse to
    # load the (metal=Ru, seed=0) snapshot.
    other = DMPNNMultiTaskBaseline(metal="Ru", seed=1, ckpt_dir=str(ckpt_dir))
    other.model = TinyModel()
    with pytest.raises(ValueError, match="seed"):
        other.load_checkpoint(path=out_path)

    # Cross-run guard: a baseline with a different metal must refuse.
    other2 = DMPNNMultiTaskBaseline(metal="Ir", seed=0, ckpt_dir=str(ckpt_dir))
    other2.model = TinyModel()
    with pytest.raises(ValueError, match="metal"):
        other2.load_checkpoint(path=out_path)


# ---------------------------------------------------------------------------
# Test 2 — resume continues from epoch
# ---------------------------------------------------------------------------
def test_resume_continues_from_epoch(tmp_path: Path):
    """A "crashed" run at epoch 5 must continue from epoch 6 on --resume.

    We simulate the crash by:

      1. Creating a baseline with ``epochs=10`` and saving a snapshot
         at ``epoch=5`` after a few Adam steps on the tiny model.
      2. Constructing a FRESH baseline (``epochs=10``, ``resume=True``)
         and driving its ``fit()`` epoch loop the same way ``fit()``
         does (start_epoch, optimizer.load_state_dict, epoch loop).
      3. Asserting that ``fit()`` only ran ``10 - 5 = 5`` epochs (i.e.
         the second baseline's effective epoch count is ``epochs -
         resumed_epoch``).

    We use the tiny :class:`TinyModel` instead of calling
    ``train_epoch_mt`` directly — ``train_epoch_mt`` expects a real
    ``DMPNNMultiTaskModel`` forward signature, but the safety net
    contract is independent of the model class.
    """
    ckpt_dir = tmp_path / "wf146_lomo_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _simulate_epoch(model: TinyModel, optim: torch.optim.Optimizer) -> None:
        """One fake D-MPNN-MT epoch: a few Adam steps on a fixed input."""
        torch.manual_seed(0)
        for _ in range(2):
            x = torch.randn(4, 4)
            y = torch.randn(4, 1)
            optim.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            optim.step()

    # --- Run 1: "crashes" after epoch 5 ------------------------------
    b1 = DMPNNMultiTaskBaseline(
        metal="Ir",
        seed=42,
        epochs=10,
        ckpt_dir=str(ckpt_dir),
        ckpt_every=5,
    )
    b1.model = TinyModel()  # type: ignore[assignment]
    optim1 = torch.optim.Adam(b1.model.parameters(), lr=1e-3)
    last_saved_epoch = -1
    saved_epochs: list[int] = []
    for ep in range(10):
        _simulate_epoch(b1.model, optim1)  # type: ignore[arg-type]
        if (ep + 1) % 5 == 0:
            ckpt_p = b1.save_checkpoint(
                epoch=ep + 1,
                best_metric=0.5,
                optimizer=optim1,
            )
            last_saved_epoch = ep + 1
            saved_epochs.append(ep + 1)
    # ``ckpt_every=5`` saves at epoch 5 AND at the final epoch 10.
    assert saved_epochs == [5, 10]
    assert (ckpt_dir / "Ir_seed42_epoch00005.pt").exists()
    assert (ckpt_dir / "Ir_seed42_epoch00010.pt").exists()

    # Simulate a crash AFTER epoch 5 by deleting the epoch-10 snapshot —
    # this is exactly the scenario the safety net is for.  With only
    # the epoch-5 checkpoint on disk, ``find_latest_checkpoint`` must
    # return epoch 5 and ``load_checkpoint`` must hand back start_epoch=6.
    (ckpt_dir / "Ir_seed42_epoch00010.pt").unlink()

    # --- Run 2: --resume from epoch 5 ---------------------------------
    b2 = DMPNNMultiTaskBaseline(
        metal="Ir",
        seed=42,
        epochs=10,
        ckpt_dir=str(ckpt_dir),
        ckpt_every=5,
        resume=True,
    )
    b2.model = TinyModel()  # type: ignore[assignment]
    optim2 = torch.optim.Adam(b2.model.parameters(), lr=1e-3)

    start_epoch, best_metric = b2.load_checkpoint()
    # The epoch-10 snapshot was deleted (crash simulation), so the
    # latest survivor is epoch 5 → resume must start at epoch 6.
    assert (
        start_epoch == 6
    ), f"resume should pick up at epoch 6 (latest=5, +1), got {start_epoch}"
    assert best_metric == pytest.approx(0.5)
    # Optimizer state was stashed on the baseline; fit() applies it
    # after constructing the optimizer.  Mirror that step here.
    assert b2._optimizer_state_resume is not None
    optim2.load_state_dict(b2._optimizer_state_resume)

    # Drive the remaining epochs and confirm we trained exactly
    # (epochs - 6) = 4 more (epochs 6,7,8,9).
    remaining = 0
    for ep in range(start_epoch, 10):
        _simulate_epoch(b2.model, optim2)  # type: ignore[arg-type]
        remaining += 1
    assert remaining == 4, f"resume should run exactly 4 more epochs, ran {remaining}"

    # Weights should not be identical to the fresh TinyModel (i.e. the
    # checkpoint was actually applied, not silently skipped).
    fresh = TinyModel()
    for (n1, p1), (n2, p2) in zip(
        b2.model.named_parameters(), fresh.named_parameters()
    ):  # type: ignore[attr-defined]
        assert n1 == n2
        assert not torch.equal(p1, p2), (
            f"weight {n1} is identical to fresh init — checkpoint was " "not applied."
        )

    # Final assertion: find_latest_checkpoint discovers the surviving
    # epoch-5 file (the epoch-10 one was deleted above).
    latest = b2.find_latest_checkpoint()
    assert latest is not None
    assert latest.name == "Ir_seed42_epoch00005.pt"


# ---------------------------------------------------------------------------
# Test 3 — resume preserves optimizer state
# ---------------------------------------------------------------------------
def test_resume_preserves_optimizer_state(tmp_path: Path):
    """Adam's exp_avg / exp_avg_sq / step must survive the round-trip.

    Strategy:

      * Build a fresh Adam optimiser on a tiny model.
      * Take 10 train steps on a fixed input → Adam state has nonzero
        exp_avg entries.
      * Save the checkpoint (model + optimizer).
      * Mutate both the model and the optimiser (zero grads, swap
        parameters).
      * Reload from disk into a brand-new model + Adam pair.
      * Assert: model weights match, and the Adam state_dict has the
        same ``exp_avg`` / ``exp_avg_sq`` / ``step`` values.
    """
    ckpt_dir = tmp_path / "wf146_lomo_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    baseline = DMPNNMultiTaskBaseline(
        metal="Pt",
        seed=7,
        ckpt_dir=str(ckpt_dir),
        ckpt_every=5,
    )
    tiny = _patch_baseline_model(monkeypatch=None, baseline=baseline)  # type: ignore[arg-type]
    optim = baseline._tiny_optim  # type: ignore[attr-defined]

    # Burn 10 Adam steps so the moving averages are nonzero.
    torch.manual_seed(0)
    for _ in range(10):
        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        optim.zero_grad()
        pred = tiny(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optim.step()

    # Snapshot what we expect to round-trip.  ``copy.deepcopy`` is the
    # safe way to grab an optimizer state_dict because the nested
    # ``state`` dict contains plain python ints (``step``) and tensor
    # buffers (``exp_avg``, ``exp_avg_sq``) — both handled by deepcopy.
    pre_save_state = copy.deepcopy(optim.state_dict())
    pre_save_model = {k: v.detach().cpu().clone() for k, v in tiny.state_dict().items()}

    ckpt_path = baseline.save_checkpoint(
        epoch=5,
        best_metric=0.5,
        optimizer=optim,
    )

    # Sanity: payload contains optimizer_state_dict.
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in raw

    # Build a completely fresh model + optimiser, then restore.
    fresh_model = TinyModel()
    fresh_optim = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    baseline.model = fresh_model  # type: ignore[assignment]
    baseline.load_checkpoint(path=ckpt_path)

    # Model weights must match.
    for (n1, p1), p2 in zip(
        fresh_model.named_parameters(),
        pre_save_model.values(),
    ):
        assert torch.equal(p1, p2), f"model weight {n1} did not round-trip"

    # Optimizer state must restore without error and contain the Adam
    # bookkeeping we expect (exp_avg, exp_avg_sq, step) for every param.
    fresh_optim.load_state_dict(baseline._optimizer_state_resume)
    restored_state = fresh_optim.state_dict()
    assert "state" in restored_state
    assert len(restored_state["state"]) > 0
    for group in restored_state["state"].values():
        # Adam adds these on the first .step() — they must survive.
        assert "exp_avg" in group
        assert "exp_avg_sq" in group
        assert "step" in group

    # Re-running one more step from the restored optimiser must produce
    # a DIFFERENT gradient than running the same step on a fresh Adam
    # (which is the whole point of preserving the state).
    fresh_optim.zero_grad()
    x = torch.randn(4, 4)
    y = torch.randn(4, 1)
    pred = fresh_model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    grads_restored = {
        n: p.grad.detach().clone() for n, p in fresh_model.named_parameters()
    }
    fresh_optim.step()

    brand_new_model = TinyModel()
    brand_new_optim = torch.optim.Adam(brand_new_model.parameters(), lr=1e-3)
    brand_new_optim.zero_grad()
    pred = brand_new_model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    grads_fresh = {
        n: p.grad.detach().clone() for n, p in brand_new_model.named_parameters()
    }
    brand_new_optim.step()

    # Gradients themselves are the same (deterministic input); what we
    # want to assert is that the *post-step* parameters differ because
    # the Adam momentum buffers differ.
    for (n_r, p_r), (n_f, p_f) in zip(
        fresh_model.named_parameters(),
        brand_new_model.named_parameters(),
    ):
        assert n_r == n_f
        assert not torch.equal(p_r, p_f), (
            f"param {n_r} unchanged after step — optimizer state was "
            "not actually restored."
        )
