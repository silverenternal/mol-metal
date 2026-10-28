# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T72 — the decoder's metal ceiling must cover the corpus it decodes.

WHY THIS FILE EXISTS
--------------------
``YuelBondBondDecoder._select_bonds`` keeps a metal-ligand bond only while the
metal's remaining budget covers it.  The budget starts at
``_COVALENT_VALENCE[Z]``.  For Ru/Ir/Rh/Os/Re that constant was 6 and for Pt 4.

MEASURED against ``tmqm_parsed.csv`` (33,429 rows), those are the corpus
MEDIAN coordination numbers, not ceilings:

    metal   n      MND p50  p95  p99   decoder cap before
    Pt     8275         4    6    6   4
    Ru     9800         6    9   10   6
    Ir     5232         6    8    9   6
    Rh     4640         6    8   10   6
    Os     1310         6    9   10   6
    Re     4172         6    8    9   6

A ceiling at the median refuses half the cohort by construction.  On the
400-cloud c8 cohort 29.75% of clouds exceed it and 230/2,323 (9.9%) of metal
bonds are unrepresentable (``metrics/by_metric/wfa113_t72_valence_cap.json``).

The physical reason the median is 6 while the tail runs to 10 is hapticity:
an eta6-arene contributes SIX metal-carbon bonds to the graph but ONE ligand
and three coordination sites.  53.4% of Ru clouds in the cohort are pi-bound.

What this file pins
-------------------
1. Every cohort metal's decoder cap is at or above the corpus MND p99.
2. Pt is the one deliberate exception (pinned at 4, the Pt(II) target).
3. An eta6-arene Ru complex with three extra donors keeps all nine bonds
   through ``_select_bonds`` — the case the old cap deleted.
4. A cap that is too high is still caught: Pd keeps 4 of 5 attempted bonds.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    _COVALENT_VALENCE,
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondResult  # noqa: E402
from molmetal.models.bond_head import (  # noqa: E402
    AtomCloud,
    BOND_SINGLE,
)

CORPUS = PROJECT_ROOT / ".data" / "molmetal" / "tmQM" / "tmqm_parsed.csv"

#: Z -> (symbol, is_pinned_by_chemistry)
COHORT = {
    78: "Pt",
    44: "Ru",
    77: "Ir",
    45: "Rh",
    76: "Os",
    75: "Re",
}
#: Pt is the one element whose cap is a chemistry decision, not the p99.
CHEMISTRY_PINNED = {78: 4}


def _mnd_percentiles() -> dict[str, list[int]]:
    if not CORPUS.is_file():
        pytest.skip(f"tmQM corpus not present at {CORPUS}")
    by: dict[str, list[int]] = defaultdict(list)
    with CORPUS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = row.get("metal")
            if sym not in set(COHORT.values()):
                continue
            try:
                by[sym].append(int(float(row["coord_number"])))
            except (KeyError, TypeError, ValueError):
                continue
    return by


def _pct(vals: list[int], q: float) -> int:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def _forced_single_result(z_list, pairs):
    n_atoms = len(z_list)
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    for k, (i, j) in enumerate(pairs):
        # Distinct distances so the sort order is deterministic; all well
        # inside any bonding cutoff.
        positions[j] = torch.tensor(
            [1.6 + 0.01 * k, 0.05 * k, 0.0], dtype=torch.float32
        )
    cloud = AtomCloud(
        positions=positions,
        atomic_numbers=torch.tensor(z_list, dtype=torch.long),
    )
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    logits = torch.zeros((len(pairs), 5), dtype=torch.float32)
    logits[:, BOND_SINGLE] = 5.0
    yres = YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
    )
    return cloud, yres


# ---------------------------------------------------------------------------
# 1. Caps vs corpus
# ---------------------------------------------------------------------------
def test_every_cohort_metal_cap_covers_the_corpus_p99():
    pcts = _mnd_percentiles()
    bad = []
    for z, sym in COHORT.items():
        if z in CHEMISTRY_PINNED:
            continue
        cap = int(_COVALENT_VALENCE[z])
        p99 = _pct(pcts[sym], 0.99)
        if cap < p99:
            bad.append((sym, cap, p99))
    assert not bad, (
        "decoder metal cap below the corpus MND p99, so >1% of legitimate "
        f"complexes cannot be assembled: {bad}"
    )


def test_caps_are_not_the_corpus_median():
    """The regression itself: a cap equal to the median refuses half the cohort."""
    pcts = _mnd_percentiles()
    median_capped = []
    for z, sym in COHORT.items():
        if z in CHEMISTRY_PINNED:
            continue
        if int(_COVALENT_VALENCE[z]) <= _pct(pcts[sym], 0.50):
            median_capped.append(sym)
    assert not median_capped, (
        f"cap <= corpus median for {median_capped}; the selector would refuse "
        "half the cohort by construction"
    )


def test_pt_cap_is_the_documented_chemistry_choice():
    """Pt is pinned at 4 on purpose — 81.8% of the Pt corpus is 4-coordinate."""
    assert int(_COVALENT_VALENCE[78]) == CHEMISTRY_PINNED[78]


# ---------------------------------------------------------------------------
# 2. The case the old cap deleted
# ---------------------------------------------------------------------------
def test_eta6_arene_ru_keeps_all_nine_bonds():
    """Ru + eta6-benzene (6 C) + 3 N: true degree 9, CN 6.

    The old cap of 6 kept 6 of 9 bonds, leaving the arene ring attached to
    nothing -- a disconnected fragment, so ``molecule_is_valid`` was False no
    matter what the head predicted.
    """
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [44] + [6] * 6 + [7] * 3
    pairs = [(0, k) for k in range(1, 10)]
    cloud, yres = _forced_single_result(z_list, pairs)
    bond_orders, _dative, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert n_kept == 9, (
        f"eta6-arene Ru must keep 9 metal bonds, got {n_kept} "
        f"(n_overvalent={n_overvalent})"
    )
    assert n_overvalent == 0


def test_arene_bonds_survive_even_when_ranked_last():
    """Same cloud, but the metal bonds carry the LOWEST confidence.

    Sorting is by ``max_logits`` descending, so a decode whose ligand-internal
    bonds outscore the metal bonds walks the metal last.  The cap must still
    admit the whole coordination sphere, otherwise the outcome depends on the
    head's confidence ordering rather than on chemistry.
    """
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [44] + [6] * 6 + [7] * 3
    pairs = [(0, k) for k in range(1, 10)]
    cloud, yres = _forced_single_result(z_list, pairs)
    # Metal bonds get a lower SINGLE logit than the rest; all still argmax.
    yres.logits[:] = 1.0
    yres.logits[:, BOND_SINGLE] = 1.5
    bond_orders, _dative, n_kept, _n_over = decoder._select_bonds(cloud, yres)
    assert n_kept == 9


# ---------------------------------------------------------------------------
# 3. The cap still refuses over-coordination
# ---------------------------------------------------------------------------
def test_pd_still_refuses_a_fifth_bond():
    """Raising the cohort metals must not raise Pd."""
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [46, 7, 7, 7, 7, 7]
    pairs = [(0, k) for k in range(1, 6)]
    cloud, yres = _forced_single_result(z_list, pairs)
    _bo, _d, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert (n_kept, n_overvalent) == (4, 1)


def test_pt_still_refuses_a_fifth_bond():
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [78, 7, 7, 7, 7, 7]
    pairs = [(0, k) for k in range(1, 6)]
    cloud, yres = _forced_single_result(z_list, pairs)
    _bo, _d, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert (n_kept, n_overvalent) == (4, 1)
