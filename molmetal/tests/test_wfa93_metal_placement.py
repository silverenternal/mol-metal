"""Wf-A93 (2026-10-19) — metal placement in CSV-built clouds.

The defect this pins: RDKit embeds an isolated metal atom at the ORIGIN and
centres the ligand at the origin too, so ``legacy`` leaves the metal INSIDE
the ligand.  MEASURED on the 5-metal cohort: 61.8% of label-side dative bonds
were shorter than 1.0 A (median 0.366 A).  ``bonded`` places the metal 2.05 A
outside the top-ranked donor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v3.csv"


def _trainer():
    name = "_wfa93_trainer"
    if name in sys.modules:
        return sys.modules[name]
    path = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dative_distances(cloud) -> list[float]:
    flags = getattr(cloud, "is_dative_pair", None)
    bonds = getattr(cloud, "bonds", None)
    if flags is None or bonds is None:
        return []
    metal = int(cloud.metal_atom_index.item())
    b = bonds.tolist()
    f = flags.tolist()
    seen: set[int] = set()
    out: list[float] = []
    for k in range(len(f)):
        if not f[k]:
            continue
        i, j = b[0][k], b[1][k]
        donor = j if i == metal else (i if j == metal else None)
        if donor is None or donor in seen:
            continue
        seen.add(donor)
        out.append(float((cloud.positions[metal] - cloud.positions[donor]).norm()))
    return out


def _min_heavy_distance(cloud) -> float:
    metal = int(cloud.metal_atom_index.item())
    z = cloud.atomic_numbers.tolist()
    d = torch.norm(cloud.positions - cloud.positions[metal], dim=-1)
    return min(float(d[k]) for k in range(len(z)) if z[k] != 1 and k != metal)


# ---------------------------------------------------------------------------
# The placement helper
# ---------------------------------------------------------------------------
def test_place_metal_at_donor_moves_it_outside_the_ligand() -> None:
    t = _trainer()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    z = torch.tensor([44, 7, 6], dtype=torch.long)
    new, ok = t._place_metal_at_donor(pos, z, 0, 1)
    assert ok
    assert abs(float((new[0] - new[1]).norm()) - 2.05) < 1e-4
    assert float((new[0] - new[2]).norm()) >= 1.2 - 1e-4
    # The input is not mutated.
    assert float(pos[0].norm()) == 0.0


def test_place_metal_pushes_past_a_blocking_atom() -> None:
    t = _trainer()
    # Donor at x=2, another heavy atom just behind it at x=2.5.  The first
    # candidate (2.05 A outward, i.e. toward -x) is only 2.55 A from that
    # atom, so a 3.0 A clearance requirement forces the push loop to run.
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        dtype=torch.float32,
    )
    z = torch.tensor([44, 7, 6], dtype=torch.long)
    new, ok = t._place_metal_at_donor(pos, z, 0, 1, min_clearance=3.0)
    assert ok
    assert float((new[0] - new[2]).norm()) >= 3.0 - 1e-4
    assert float((new[0] - new[1]).norm()) > 2.05


def test_unknown_placement_mode_is_refused() -> None:
    t = _trainer()
    try:
        t.build_cloud_from_csv_row("C", "Ru", metal_placement="nope")
    except ValueError as exc:
        assert "metal_placement" in str(exc)
    else:
        raise AssertionError("an unknown placement mode must raise")


# ---------------------------------------------------------------------------
# End to end on the real cohort
# ---------------------------------------------------------------------------
def test_bonded_cohort_has_physical_dative_distances() -> None:
    t = _trainer()
    clouds = t.build_csv_cohort(str(CSV), split="val", n_clouds=8, seed=0)
    assert clouds
    n = 0
    for cloud in clouds:
        for d in _dative_distances(cloud):
            n += 1
            assert 1.8 <= d <= 2.6, f"unphysical dative distance {d:.3f} A"
        assert _min_heavy_distance(cloud) >= 1.0
    assert n > 0


def test_legacy_cohort_keeps_the_centroid_placement() -> None:
    t = _trainer()
    clouds = t.build_csv_cohort(
        str(CSV), split="val", n_clouds=8, seed=0, metal_placement="legacy"
    )
    assert clouds
    # Documents the pre-Wf-A93 defect rather than asserting it is good: the
    # metal sits at the ligand centroid, so some heavy atom is inside 1.0 A.
    assert any(_min_heavy_distance(c) < 1.0 for c in clouds)


def test_dative_distance_cutoff_is_actually_consumed() -> None:
    t = _trainer()
    narrow = t.build_csv_cohort(
        str(CSV), split="val", n_clouds=8, seed=0, dative_distance_cutoff=2.6
    )
    wide = t.build_csv_cohort(
        str(CSV), split="val", n_clouds=8, seed=0, dative_distance_cutoff=100.0
    )
    n_narrow = sum(len(_dative_distances(c)) for c in narrow)
    n_wide = sum(len(_dative_distances(c)) for c in wide)
    assert n_wide > n_narrow, (n_wide, n_narrow)
