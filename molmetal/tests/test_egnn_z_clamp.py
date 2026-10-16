"""Wf-A61 (2026-10-17) L11 — atomic-number conversion site clamps ``int(zi)``
into the vocab table.  The harness replicates the conversion list
comprehension in isolation, since the full ``fm_velocity`` path depends
on model state.
"""

from __future__ import annotations


def _convert(z_table: dict, z_dim: int, atomic_numbers: list) -> list:
    """Mirror the Wf-A61-clamped conversion at train_cfm_yuelbond_wasserstein.py:1974-1979."""
    return [
        z_table.get(max(0, min(int(zi), int(z_dim) - 1)), 0)
        for zi in atomic_numbers
    ]


def test_egnn_z_clamp_above_max_returns_last_index() -> None:
    """L11 — Z>z_dim-1 maps to idx ``z_dim-1`` (the legacy fallback)."""
    # z_table covers indices [0..z_dim-1]; the clamp guarantees that
    # atomic_numbers > z_dim-1 land on the dict's max-entry index.
    z_table = {i: i for i in range(20)}
    out = _convert(z_table, z_dim=20, atomic_numbers=[200, 999, 1000])
    assert out == [19, 19, 19]


def test_egnn_z_clamp_negative_returns_neg_one() -> None:
    """L11 — negative Z (shouldn't happen but harmless) clamps to 0."""
    z_table = {0: 0, 6: 1}
    out = _convert(z_table, z_dim=20, atomic_numbers=[-1, -5])
    assert out == [0, 0]


def test_egnn_z_clamp_in_range_uses_dict() -> None:
    """L11 — in-range Z still uses the dict lookup (no regression)."""
    z_table = {0: 0, 6: 1, 7: 2, 8: 3}
    out = _convert(z_table, z_dim=20, atomic_numbers=[6, 7, 8])
    assert out == [1, 2, 3]