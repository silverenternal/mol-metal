"""Round-10 sanity harness for the Lambda chemistry metrics.

These tests are deliberately small and deterministic.  They print observed
values so ``pytest -s`` can be used as a metric report.
"""

from __future__ import annotations

import importlib
import random

import pytest

from molmetal_lam.atoms.combinators import METAL_ATOMS, PRIMITIVE_ATOMS, from_smiles
from molmetal_lam.bonds.application import (
    Bond, COVALENT, DATIVE, AROMATIC, HYDROGEN, FreeSiteLedger, distinct,
)
from molmetal_lam.tile_lib.fragment_pool import fragments_from_chembl_reactive
from molmetal_lam.tile_lib.property_tests import assert_library_well_formed


CLICK_HANDLES = ["CCN=[N+]=[N-]", "C#CC", "C=CC=C", "CCS", "CC(=O)O"]
EXPECTED_CN = {"Pt_II": 4, "Ru_II": 6, "Zn_II": 4, "Ir_III": 6}


def test_arity_hit_rate() -> None:
    pool = fragments_from_chembl_reactive()
    smiles = [t.smiles for t in pool] + CLICK_HANDLES
    rng = random.Random(10)
    sample = [rng.choice(smiles) for _ in range(200)]
    hits = total = 0
    for smi in sample:
        for atom in from_smiles(smi):
            total += 1
            expected = PRIMITIVE_ATOMS.get(atom.symbol, METAL_ATOMS.get(atom.symbol))
            hits += int(expected is not None and atom.arity == expected.arity)
    rate = hits / total if total else 0.0
    print(f"ARITY_HIT_RATE={rate:.3f} ({hits}/{total})")
    assert rate >= 0.95


def test_metal_geometry_ok() -> None:
    observed = {}
    for name, expected in EXPECTED_CN.items():
        atom = METAL_ATOMS[name]
        observed[name] = atom.arity
        assert atom.arity == expected
    ratio = sum(observed[n] == e for n, e in EXPECTED_CN.items()) / len(EXPECTED_CN)
    print(f"METAL_GEOMETRY_OK={ratio:.3f} {observed}")


def test_dative_fraction_coordination() -> None:
    bonds = []
    for name, count in EXPECTED_CN.items():
        ledger = FreeSiteLedger()
        metal = METAL_ATOMS[name]
        for _ in range(count):
            bonds.append(Bond.dative("NH3", metal, ledger=ledger))
    fraction = sum(b.kind == DATIVE for b in bonds) / len(bonds)
    print(f"DATIVE_FRACTION={fraction:.3f} ({len(bonds)} coordination bonds)")
    assert fraction == 1.0


def test_bond_kind_distribution() -> None:
    ledger = FreeSiteLedger()
    bonds = [
        Bond.covalent(distinct("C"), distinct("C"), ledger=ledger),
        Bond.dative("NH3", "Pt_II", ledger=ledger),
        *Bond.aromatic([distinct("C") for _ in range(3)], ledger=ledger),
        Bond.hydrogen(distinct("N"), distinct("O"), ledger=ledger, strict=False),
    ]
    counts = {kind: sum(b.kind == kind for b in bonds) for kind in (COVALENT, DATIVE, AROMATIC, HYDROGEN)}
    print(f"BOND_KIND_DISTRIBUTION={counts}")
    assert all(count > 0 for count in counts.values())


def test_click_rule_coverage() -> None:
    names = ("CuAAC", "SPAAC", "thiol-ene", "Suzuki", "amide coupling")
    try:
        rules = importlib.import_module("molmetal_lam.lam_chem.rules")
        exposed = {name.lower() for name in dir(rules)}
    except ModuleNotFoundError:
        exposed = set()
    found = sum(any(token in item for item in exposed) for token in ("cuaac", "spaac", "thiol", "suzuki", "amide"))
    print(f"CLICK_RULE_COVERAGE={found}/5; requested={names}; rules.py={'present' if exposed else 'missing'}")
    assert found == 5


def test_tile_pool_size() -> None:
    tiles = assert_library_well_formed(fragments_from_chembl_reactive())
    count = len(tiles)
    print(f"TILE_POOL_SIZE={count} (target 220 or 204)")
    assert count in (200, 204, 220)
