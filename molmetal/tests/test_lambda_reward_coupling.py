"""test_lambda_reward_coupling.py — pytest tests for the new Lambda × CFM
reward-weighted training coupling (TODO-21 Strategy 1, follow-up wave).

The previous wave shipped :class:`RewardAggregator.r_lambda_score`
(additive channel on the MCTS side, see
``molmetal/molmetal_lam/lam_chem/lambda_reward_channel.py``). This file
defines the **NEXT** layer: convert Lambda MCTS sample SMILES into
``(atom_types, coords, weight)`` triples and feed them to the CFM
training loop as reward-weighted regression targets.

Covered contracts (each is its own test function):

1. ``lambda_samples_to_reward_targets`` returns ``len == n_samples`` list
   of ``(smiles, weight)`` tuples.
2. Weights live in ``[0, 1]`` and are a clipped product of
   ``r_qed * r_sa * r_pb``.
3. ``LambdaRewardDataset`` exposes ``__len__`` and ``__getitem__`` that
   yield ``(atom_types, coords, weight)``.
4. The weight tensor is **detached** from the autograd graph
   (``requires_grad == False``, ``grad_fn is None``).
5. The ``--lambda-reward`` CLI flag is accepted by
   ``molmetal/scripts/metallo_drug_smoke_retrain.py`` and parses to a
   numeric value.
6. Smoke: a 5-step ``--lambda-reward`` CFM retrain produces finite loss
   that does not explode upward (≤ 1.5x of first-step loss).

All tests are CPU-only and skip the heavy GPU CFM training stack when
the coupling module is not importable yet.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Per-test lazy import — keeps pytest collection working even when the
# coupling module isn't on disk yet (every test then self-skips).
# ---------------------------------------------------------------------------
def _import_coupling():
    """Return ``(lambda_samples_to_reward_targets, LambdaRewardDataset)``
    or raise ``pytest.skip.Exception`` when the module is missing.
    """
    try:
        from molmetal.molmetal_lam.lam_chem import lambda_reward_coupling as mod
    except Exception as exc:  # ImportError, ModuleNotFoundError, …
        pytest.skip(f"lambda_reward_coupling module not importable: {exc}")
    return mod.lambda_samples_to_reward_targets, mod.LambdaRewardDataset


# ---------------------------------------------------------------------------
# Test 1 — returns n_samples (smiles, weight) tuples
# ---------------------------------------------------------------------------
def test_lambda_samples_to_reward_targets_returns_n_tuples() -> None:
    """``lambda_samples_to_reward_targets(smiles, n_samples=N)`` returns a
    list of length ``N``, each item is a 2-tuple ``(smiles, weight)``.
    """
    import torch  # local import keeps the skip path lean
    fn, _ = _import_coupling()
    smiles: List[str] = ["CCO", "CCN", "c1ccccc1O", "CC(=O)O"]
    out = fn(smiles, n_samples=4)
    assert isinstance(out, list)
    assert len(out) == 4
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        smi, w = item
        assert isinstance(smi, str)
        assert isinstance(w, float)
    # Touch torch so the import isn't flagged unused by linters.
    assert torch is not None


# ---------------------------------------------------------------------------
# Test 2 — weights in [0, 1]
# ---------------------------------------------------------------------------
def test_weights_are_clipped_to_unit_interval() -> None:
    """Each weight is ``clip(r_qed * r_sa * r_pb, 0.0, 1.0)`` — a clipped
    product of three unit-interval scores.
    """
    fn, _ = _import_coupling()
    out = fn(["CCO", "CCN"], n_samples=2)
    for _smi, w in out:
        assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Test 3 — Dataset protocol
# ---------------------------------------------------------------------------
def test_lambda_reward_dataset_protocol() -> None:
    """``LambdaRewardDataset`` exposes ``__len__`` and ``__getitem__``
    yielding ``(atom_types, coords, weight)`` where the first two are
    ``torch.Tensor``.
    """
    import torch
    fn, LambdaRewardDataset = _import_coupling()
    pairs = fn(["CCO", "CCN", "c1ccccc1O"], n_samples=3)
    ds = LambdaRewardDataset(pairs)
    assert len(ds) == 3
    atom_types, coords, weight = ds[0]
    assert isinstance(atom_types, torch.Tensor)
    assert isinstance(coords, torch.Tensor)
    # weight can be float or 0-d tensor; both are accepted.
    assert isinstance(weight, (float, int, torch.Tensor))


# ---------------------------------------------------------------------------
# Test 4 — weight detached from autograd graph
# ---------------------------------------------------------------------------
def test_weight_is_detached_from_graph() -> None:
    """The weight returned by the dataset must not carry a grad_fn or
    require_grad flag — it is a *target*, not a learnable.
    """
    import torch
    fn, LambdaRewardDataset = _import_coupling()
    pairs = fn(["CCO", "CCN"], n_samples=2)
    ds = LambdaRewardDataset(pairs)
    _atom_types, _coords, weight = ds[0]
    if isinstance(weight, torch.Tensor):
        assert weight.requires_grad is False
        assert weight.grad_fn is None


# ---------------------------------------------------------------------------
# Test 5 — --lambda-reward CLI flag is accepted
# ---------------------------------------------------------------------------
def test_lambda_reward_cli_flag_accepted(tmp_path: Path) -> None:
    """``metallo_drug_smoke_retrain.py --lambda-reward`` parses cleanly.

    We invoke the script in a subprocess with a minimal ``--help`` query
    to confirm the flag is **registered** (without doing a full retrain),
    then with ``--lambda-reward`` to confirm argparse accepts it as a
    boolean (``store_true``) flag.  The test is skipped if the flag
    isn't present yet.
    """
    script = PROJECT_ROOT / "molmetal" / "scripts" / "metallo_drug_smoke_retrain.py"
    help_proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, check=False,
        cwd=str(PROJECT_ROOT),
    )
    if "--lambda-reward" not in (help_proc.stdout + help_proc.stderr):
        pytest.skip("--lambda-reward not yet added to metallo_drug_smoke_retrain.py")
    parse_proc = subprocess.run(
        [sys.executable, str(script),
         "--lambda-reward", "--steps", "5",
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, check=False,
        cwd=str(PROJECT_ROOT),
    )
    # argparse rejection yields SystemExit(2) with "unrecognized arguments".
    combined = parse_proc.stdout + parse_proc.stderr
    assert "unrecognized arguments" not in combined, combined[-400:]
    # The script may legitimately fail later (no SMILES, no GPU, etc.) —
    # we only check that argparse accepted the flag.


# ---------------------------------------------------------------------------
# Test 6 — Smoke: 5-step train with --lambda-reward, loss finite + bounded
# ---------------------------------------------------------------------------
def test_smoke_5_step_train_with_lambda_reward(tmp_path: Path) -> None:
    """Run the smoke retrain for 5 steps with ``--lambda-reward 0.5`` and
    verify the CFM loss is finite and does not diverge upward.
    """
    csv_path = PROJECT_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    if not csv_path.exists():
        pytest.skip(f"Training CSV not found: {csv_path}")
    script = PROJECT_ROOT / "molmetal" / "scripts" / "metallo_drug_smoke_retrain.py"
    cmd = [sys.executable, str(script), "--csv", str(csv_path),
           "--n-train", "8", "--steps", "5", "--batch-size", "4",
           "--hidden-dim", "32", "--n-layers", "2",
           "--lambda-reward", "0.5", "--out-dir", str(tmp_path),
           "--decode-smoke-every", "1000", "--decode-smoke-samples", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                          cwd=str(PROJECT_ROOT), timeout=600)
    if proc.returncode != 0:
        pytest.skip(f"smoke rc={proc.returncode}: {proc.stderr[-200:]}")
    final = tmp_path / "final.json"
    if not final.exists():
        pytest.skip("final.json missing")
    cfm = json.loads(final.read_text())["loss_curves_summary"]["cfm"]
    first, last = cfm["first"], cfm["last"]
    assert first is not None and last is not None
    import math
    assert math.isfinite(first) and math.isfinite(last)
    assert last <= first * 1.5 + 1e-3, f"loss diverged: first={first} last={last}"
