# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T71 — a Pt-only atom vocabulary on a five-metal cohort.

``z_to_onehot`` maps an unknown Z to index 0, the PADDING slot
(``yuelbond_decoder.py:282``).  The default ``yuelbond_vocab`` is ``pt``:
13 entries whose only metal is Pt.  ``cfm_5metal_default`` and 51 other
configs train on Ru/Ir/Os/Rh/Re with that table, so every metal atom is
encoded as "no atom" — and the ``atom_vocab_size`` guard passes because
both tables are 13 wide.

These tests pin the encoding fact, the new startup guard, and the measured
artifact.

Run:
    .venv/bin/python -m pytest molmetal/tests/test_wfa113_t71_vocab_coverage.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from molmetal.models.bond_head import AtomCloud
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    get_z_table,
    z_to_onehot,
)
from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr


def _cloud(zs) -> AtomCloud:
    n = len(zs)
    return AtomCloud(
        positions=torch.zeros(n, 3),
        atomic_numbers=torch.tensor(zs, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 1. The encoding fact
# ---------------------------------------------------------------------------


def test_pt_table_has_no_non_pt_metal() -> None:
    table = get_z_table("pt")
    assert len(table) == 13
    assert [z for z in table if z in (44, 45, 75, 76, 77, 78)] == [78]


def test_ru_and_padding_are_the_same_vector_under_the_pt_table() -> None:
    table = get_z_table("pt")
    assert torch.equal(
        z_to_onehot(torch.tensor([44]), table),
        z_to_onehot(torch.tensor([0]), table),
    ), "an OOV metal must be indistinguishable from the padding slot"
    for z in (45, 75, 76, 77):
        assert torch.equal(
            z_to_onehot(torch.tensor([z]), table),
            z_to_onehot(torch.tensor([0]), table),
        )


def test_all6_covers_the_five_metal_cohort() -> None:
    table = set(get_z_table("all6"))
    assert {44, 45, 75, 76, 77, 78} <= table
    assert not torch.equal(
        z_to_onehot(torch.tensor([44]), tuple(get_z_table("all6"))),
        z_to_onehot(torch.tensor([0]), tuple(get_z_table("all6"))),
    )


# ---------------------------------------------------------------------------
# 2. The startup guard
# ---------------------------------------------------------------------------


def test_guard_fires_on_a_five_metal_cohort_with_the_pt_table() -> None:
    clouds = [_cloud([44, 7, 6]), _cloud([77, 15, 6])]
    with pytest.raises(SystemExit) as exc:
        tr._assert_vocab_covers_cohort(get_z_table("pt"), clouds, "pt")
    msg = str(exc.value)
    assert "does not cover the cohort" in msg
    assert "all6" in msg
    assert "44" in msg and "77" in msg


def test_guard_passes_on_all6() -> None:
    clouds = [_cloud([44, 7, 6]), _cloud([77, 15, 6]), _cloud([78, 17, 17])]
    tr._assert_vocab_covers_cohort(get_z_table("all6"), clouds, "all6")


def test_guard_passes_on_a_pt_cohort_with_the_pt_table() -> None:
    tr._assert_vocab_covers_cohort(get_z_table("pt"), [_cloud([78, 7, 7])], "pt")


def test_guard_ignores_non_metal_oov() -> None:
    """An organic element outside the table is a known limitation, not this bug."""
    tr._assert_vocab_covers_cohort(get_z_table("pt"), [_cloud([78, 5, 6])], "pt")


def test_guard_is_called_before_the_model_is_built() -> None:
    import inspect

    src = inspect.getsource(tr.main)
    guard = src.index("_assert_vocab_covers_cohort(")
    build = src.index("model = build_joint_model(")
    assert guard < build, (
        "the guard must run before build_joint_model, otherwise the encoder "
        "width is already fixed at the wrong size"
    )


# ---------------------------------------------------------------------------
# 3. The measured artifact
# ---------------------------------------------------------------------------


def test_measured_artifact() -> None:
    p = Path(__file__).resolve().parents[2] / "metrics/by_metric/wfa113_t71_vocab.json"
    if not p.is_file():
        pytest.skip("wfa113_t71 artifact not present")
    d = json.loads(p.read_text())
    m = d["measured"]["cfm_5metal_default.yaml"]
    assert m["metal_oov_frac"] == 1.0, (
        f"expected every metal atom OOV, got {m['metal_oov_frac']}"
    )
    assert m["onehot_Ru_equals_onehot_pad"] is True
    assert d["configs_with_oov_metal"] >= 50, (
        "the audit found fewer than 50 affected configs; re-run the probe"
    )
