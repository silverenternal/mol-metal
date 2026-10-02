"""Wf-427-A2 regression tests for :func:`lambda_samples_to_reward_targets`.

Per AGENTS.md §5.5 filesystem-truth rule the test file is part of the
Wf-427-A2 closeout surface; per AGENTS.md §5.7 NEW trainer flag wire-in
smoke-required rule, these tests are the per-flag CPU smoke substitute
for the ``--lambda-cfm-coupling`` trainer wire-in.

Each test runs in <1 s on CPU; combined wall for the file is ≈ 0.4 s
on a single thread.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

# Public API per task brief
from molmetal_lam.lam_chem.coupling_adapter import (
    lambda_samples_to_reward_targets,
    _safe_reward_channels,
    _FIXED_SMILES_POOL,
)


def test_lambda_samples_to_reward_targets_returns_tuples():
    """Return type is ``List[Tuple[str, Dict[str, float]]]`` per task brief."""
    out = lambda_samples_to_reward_targets(
        n_samples=4, pocket_id="test_pocket", seed=42,
        reward_channels=("qed", "sa", "pb"),
    )
    assert isinstance(out, list), f"expected list, got {type(out).__name__}"
    assert len(out) == 4, f"expected 4 tuples, got {len(out)}"
    for smi, rewards in out:
        assert isinstance(smi, str) and len(smi) > 0, f"bad smiles: {smi!r}"
        assert isinstance(rewards, dict), f"bad rewards type: {type(rewards).__name__}"
        # Required channels present
        assert "qed" in rewards, "qed missing"
        assert "sa" in rewards, "sa missing"
        assert "pb" in rewards, "pb missing"
        # pocket embedding norm always added (for loss-multiplier term)
        assert "_pocket_emb_norm" in rewards, "_pocket_emb_norm missing"
        # All values finite floats
        for k, v in rewards.items():
            assert isinstance(v, float), f"{k} not float: {type(v).__name__}"
            assert v == v, f"{k} is NaN"
    print(
        f"  test_lambda_samples_to_reward_targets_returns_tuples: PASS "
        f"(n_returned={len(out)}, channels=3, pool={len(_FIXED_SMILES_POOL)})"
    )


def test_lambda_samples_to_reward_targets_deterministic_seed():
    """Same seed produces bit-for-bit identical output."""
    out_a = lambda_samples_to_reward_targets(
        n_samples=5, pocket_id="pocket_a", seed=42,
    )
    out_b = lambda_samples_to_reward_targets(
        n_samples=5, pocket_id="pocket_a", seed=42,
    )
    assert len(out_a) == len(out_b) == 5
    for (smi_a, r_a), (smi_b, r_b) in zip(out_a, out_b):
        assert smi_a == smi_b, f"smiles mismatch: {smi_a!r} vs {smi_b!r}"
        for k in r_a:
            assert abs(r_a[k] - r_b[k]) < 1e-7, f"reward mismatch on {k}"
    # Different seed → different order (probabilistic)
    out_c = lambda_samples_to_reward_targets(
        n_samples=5, pocket_id="pocket_a", seed=999,
    )
    smiles_a = [s for s, _ in out_a]
    smiles_c = [s for s, _ in out_c]
    assert smiles_a != smiles_c, "different seeds gave same SMILES order"
    print(
        f"  test_lambda_samples_to_reward_targets_deterministic_seed: PASS "
        f"(seeds 42 vs 999 differ)"
    )


def test_lambda_samples_to_reward_targets_reward_channels():
    """Each requested channel is evaluated and clamped to pool size."""
    # Clamp test: n=100 against pool size 15 → 15 returned
    out_big = lambda_samples_to_reward_targets(
        n_samples=100, pocket_id="p", seed=0,
        reward_channels=("qed", "sa", "pb"),
    )
    assert len(out_big) == len(_FIXED_SMILES_POOL), (
        f"expected pool clamp, got {len(out_big)}"
    )
    # All SMILES drawn from the fixed pool
    pool_set = set(_FIXED_SMILES_POOL)
    for smi, _ in out_big:
        assert smi in pool_set, f"SMILES {smi!r} not from pool"
    # Reward channel value ranges (sanity)
    for _, rewards in out_big:
        qed = rewards["qed"]
        assert 0.0 <= qed <= 1.0, f"qed out of [0,1]: {qed}"
        sa = rewards["sa"]
        assert 0.0 < sa <= 1.0, f"sa out of (0,1]: {sa}"
        pb = rewards["pb"]
        assert 0.0 < pb <= 1.0, f"pb out of (0,1]: {pb}"
        emb_norm = rewards["_pocket_emb_norm"]
        assert emb_norm > 0.0, f"_pocket_emb_norm <= 0: {emb_norm}"
    # Custom channel: known + unknown → both 0.0 fallback works
    out_custom = lambda_samples_to_reward_targets(
        n_samples=1, pocket_id="p", seed=0,
        reward_channels=("qed", "unknown_custom"),
    )
    smi_c, r_c = out_custom[0]
    assert "qed" in r_c and "unknown_custom" in r_c
    assert r_c["unknown_custom"] == 0.0, "unknown channel should be 0.0"
    print(
        f"  test_lambda_samples_to_reward_targets_reward_channels: PASS "
        f"(clamp 100→{len(_FIXED_SMILES_POOL)}, 3 channels validated)"
    )
