# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 / T54+T55+T56 — three silent substitutions in the data carrier.

THE DEFECTS
-----------
1. **Radius (T54).** ``Batch.collate_fn`` read
   ``getattr(cloud, "atomic_radius", None)``.  ``AtomCloud`` has no such field
   and no loader sets one, so every atom got ``PAD_R = 0.0``, which
   ``geometry_consistency`` clamps to 0.3 A.  The equilibrium length
   ``eq = r_i + r_j`` was 0.6 A for every pair of every element, so the
   bond-plausibility band ``[0.85, 1.25] * eq`` became [0.51, 0.75] A.
   MEASURED on 64 clouds of the 5-metal cohort: 34 of 3,438 true bonds (0.99%)
   were inside the band; with Cordero radii 2,966 (86.3%) are.

2. **Charge (T55).** The same function read ``"natural_charge"``; the field is
   ``natural_charges`` (plural).  ``getattr`` missed, every atom got
   ``PAD_CHARGE = 0.0``, and the natural-charge MSE regressed the head toward
   zero while reporting a falling term.  Latent: every shipped config sets the
   weight to 0.  ``_to_device_cloud`` also dropped the field, so the model's
   charge-conditioning input was always absent.

3. **Bond frame (T56).** ``geometry_consistency`` indexed the flat atom axis
   with ``Batch.bonds``, which ``collate_fn`` stores as per-cloud LOCAL
   indices (the frame ``coordination_number_from_bonds`` documents and uses).
   For every cloud except cloud 0 the adjacency landed on the wrong atoms.
   MEASURED on 64 clouds: 370 of 3,438 bond entries survived into ``adj``
   (10.8%) and ``~adj`` then marked real bonds as non-bonded — 2,582 "clash"
   pairs on ground-truth geometry.  A single-cloud batch is unaffected, which
   is why every unit test passed.

WHAT THIS FILE CHECKS
---------------------
The three carriers, the frame guard, and the artifact's quoted numbers.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "metrics" / "by_metric" / "wfa113_t54_geometry_radius.json"


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dec():
    return _load("_t54_dec", REPO / "molmetal" / "molmetal_lam" / "lam_chem" / "yuelbond_decoder.py")


@pytest.fixture(scope="module")
def tr():
    return _load("_t54_tr", REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py")


def _cloud(z, x=None, **kw):
    from molmetal.models.bond_head import AtomCloud

    z = torch.tensor(z, dtype=torch.long)
    if x is None:
        x = torch.arange(len(z), dtype=torch.float32)[:, None].repeat(1, 3)
    return AtomCloud(positions=x.float(), atomic_numbers=z, **kw)


# ------------------------------------------------------------- radius (T54)


def test_the_two_cordero_tables_cannot_drift(dec):
    from molmetal.data.batch import CORDERO_RCOV

    assert dict(CORDERO_RCOV) == dict(dec._CORDERO_RCOV)


def test_collate_fills_a_real_radius_for_every_atom():
    from molmetal.data.batch import Batch, CORDERO_RCOV

    b = Batch.collate_fn([_cloud([78, 6, 1])])
    want = [CORDERO_RCOV[78], CORDERO_RCOV[6], CORDERO_RCOV[1]]
    assert [round(v, 4) for v in b.atomic_radius.tolist()] == [round(v, 4) for v in want]
    # The old behaviour: all zeros.
    assert float(b.atomic_radius.min()) > 0.0


def test_a_cloud_supplied_radius_wins():
    from molmetal.data.batch import Batch

    c = _cloud([6, 6])
    c.atomic_radius = torch.tensor([2.5, 2.5])
    b = Batch.collate_fn([c])
    assert b.atomic_radius.tolist() == [2.5, 2.5]


# ------------------------------------------------------------- charge (T55)


def test_collate_reads_the_plural_field():
    from molmetal.data.batch import Batch

    c = _cloud([78, 6, 6], natural_charges=torch.tensor([0.42, -0.21, -0.21]))
    b = Batch.collate_fn([c])
    assert b.natural_charge.tolist() == pytest.approx([0.42, -0.21, -0.21], abs=1e-6)


def test_to_device_cloud_carries_natural_charges(tr):
    from molmetal.data.batch import PAD_CHARGE

    c = _cloud([78, 6, 6], natural_charges=torch.tensor([0.42, -0.21, -0.21]))
    out = tr._to_device_cloud(c, torch.device("cpu"))
    assert out.natural_charges is not None
    assert out.natural_charges.tolist() == pytest.approx([0.42, -0.21, -0.21], abs=1e-6)
    # A cloud without charges must not gain a sentinel.
    plain = tr._to_device_cloud(_cloud([78, 6, 6]), torch.device("cpu"))
    assert plain.natural_charges is None
    assert PAD_CHARGE == 0.0


# ---------------------------------------------------------- frame (T56)


def test_a_multi_cloud_batch_needs_pair_cloud_idx():
    from molmetal.loss.geometry_consistency import geometry_consistency

    with pytest.raises(ValueError, match="pair_cloud_idx"):
        geometry_consistency(
            torch.zeros((4, 3)),
            torch.tensor([6, 6, 6, 6]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([[0], [1]]),
            torch.full((4,), 0.76),
            2,
        )


def test_local_bond_indices_are_scored_in_their_own_cloud():
    from molmetal.loss.geometry_consistency import geometry_consistency

    # Two clouds, each a stretched C-C pair.  Both must be scored; before the
    # frame fix the second cloud's (0, 1) landed inside cloud 0.
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [13.0, 0.0, 0.0]]
    )
    out = geometry_consistency(
        x,
        torch.tensor([6, 6, 6, 6]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([[0, 0], [1, 1]]),
        torch.full((4,), 0.76),
        2,
        pair_cloud_idx=torch.tensor([0, 1]),
    )
    assert float(out["n_bond_pairs"]) == 2.0
    assert float(out["bond"]) > 0.0


def test_the_metal_term_scores_only_bonded_pairs():
    from molmetal.loss.geometry_consistency import geometry_consistency

    # Pt at the origin, one N bonded at 2.2 A, one N unbonded at 5.0 A.
    x = torch.tensor([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
    out = geometry_consistency(
        x,
        torch.tensor([78, 7, 7]),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([[0], [1]]),
        torch.tensor([1.36, 0.71, 0.71]),
        1,
    )
    assert float(out["n_metal_pairs"]) == 1.0
    assert float(out["metal"]) == 0.0


# ----------------------------------------------------------- the artifact


def test_artifact_quotes_what_the_code_comments_quote():
    assert ART.exists(), ART
    d = json.loads(ART.read_text())
    u, c = d["rows"]["uniform_0p3"], d["rows"]["cordero"]
    assert u["n_bonds"] == c["n_bonds"] == 3438
    assert u["in_band_fraction"] == pytest.approx(0.00989, abs=1e-4)
    assert c["in_band_fraction"] == pytest.approx(0.86271, abs=1e-4)
    assert u["mean_squared_deviation"] == pytest.approx(0.6371, abs=1e-3)
    assert c["mean_squared_deviation"] == pytest.approx(0.02157, abs=1e-4)
    # The term, after the frame + metal fixes, on the same cohort.
    t = d["geometry_consistency_term"]
    assert t["cordero"]["bond"] < t["uniform_0p3"]["bond"]
