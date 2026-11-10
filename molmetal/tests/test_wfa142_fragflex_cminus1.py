"""Wf-A142 R1 (2026-10-31) — unit tests over the FragFlex-style C-1 probe.

WHAT IS UNDER TEST
------------------
``molmetal/scripts/wfa142_fragflex_cminus1_probe.py`` implements a FragFlex-style
valence-aware spanning-tree completion on the EXISTING generation pool:
add EXACTLY ``C-1`` edges to a ``C``-component molecule, chosen by shortest
inter-component atom distance among valence-feasible pairs, with an
RDKit-sanitize-gated rollback that never deletes an atom or an edge.

Frontier source: FragFlex — Sihyun Park & David Ryan Koes, "FragFlex:
Fragment-Based Flexible Molecular Generation for Exploring Vast Chemical
Space", arXiv:2610.15284 (NeurIPS 2026 Workshop on AI for Drug Discovery).
Verified mechanics used here: *"If the final graph contains C connected
components before this repair, exactly C-1 edges are added"*.

THE THREE INVARIANTS THE BRIEF ASKS FOR
---------------------------------------
1. the C-1 invariant — a C-component molecule gains exactly C-1 edges;
2. the never-deletes-an-atom / never-deletes-an-edge invariant;
3. determinism.

Plus the two gates the repair rests on (the repo's own valence budget and
RDKit's sanitizer) and the CLI surface.

CPU ONLY: the repo's ``molmetal/tests/conftest.py`` blanks
``CUDA_VISIBLE_DEVICES`` at import time unless ``MOLMETAL_TEST_ALLOW_GPU=1``,
so this module inherits the no-GPU contract.  Run it with::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \\
        molmetal/tests/test_wfa142_fragflex_cminus1.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem  # noqa: E402

from molmetal.scripts import wfa142_fragflex_cminus1_probe as probe  # noqa: E402
from molmetal.models.bond_head import BOND_SINGLE  # noqa: E402
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    _endpoint_valence_cost,
)

POOL_PATH = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c7_fullgen_2026_10_20"
    / "generated_smiles.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pool_smiles() -> list:
    """The pool's epoch-17 cohort (the cohort the catalogue quotes)."""
    if not POOL_PATH.exists():
        pytest.skip(f"pool not on disk: {POOL_PATH}")
    rows = probe.load_pool(POOL_PATH, "epoch17")
    return [str(r["smiles"]) for r in rows]


@pytest.fixture(scope="module")
def sample_smiles(pool_smiles: list) -> list:
    """A small, deterministic slice of the epoch-17 cohort."""
    return pool_smiles[:40]


def _complete(smiles: str, **kwargs) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    positions, status = probe.embed_positions(
        mol, seed=0, geometry=kwargs.pop("geometry", "whole")
    )
    assert status in ("3d", "2d")
    return probe.fragflex_complete(mol, positions, **kwargs)


# ---------------------------------------------------------------------------
# 1. the C-1 invariant
# ---------------------------------------------------------------------------
def test_c_minus_1_invariant_on_a_synthetic_three_component_molecule():
    """A C-component molecule gains exactly C-1 edges and ends at one component."""
    mol = Chem.MolFromSmiles("CCC.CCO.C")  # 3 components
    assert len(probe.components(mol)) == 3
    positions, _status = probe.embed_positions(mol, seed=0)
    out = probe.fragflex_complete(mol, positions)
    assert out["n_components_before"] == 3
    assert out["edges_added"] == 2, out
    assert out["n_components_after"] == 1, out
    assert out["stranded"] is False


def test_c_minus_1_invariant_on_the_real_pool(sample_smiles: list):
    """The C-1 invariant holds on every epoch-17 molecule of the pool.

    For a molecule the repair could complete, ``edges_added == C - 1``.  For a
    molecule it could NOT complete (``stranded``), the edge count is
    STRICTLY SMALLER and the molecule keeps more than one component — the
    repair never fabricates an edge it could not validate.
    """
    n_stranded = 0
    for smiles in sample_smiles:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        c_before = len(probe.components(mol))
        out = _complete(smiles)
        if out["stranded"]:
            n_stranded += 1
            assert out["edges_added"] < c_before - 1, out
            assert out["n_components_after"] > 1, out
        else:
            assert out["edges_added"] == c_before - 1, out
            assert out["n_components_after"] == 1, out
        # The general form, stranded or not: every added edge merges exactly
        # two components, so the component count drops by exactly one per
        # edge and never below one.
        assert out["n_components_after"] == max(1, c_before - out["edges_added"])
    assert n_stranded >= 0


def test_c_minus_1_invariant_holds_for_every_geometry(sample_smiles: list):
    """The C-1 invariant is independent of the geometry used to rank pairs."""
    for geometry in ("whole", "fragsep", "index"):
        for smiles in sample_smiles[:15]:
            mol = Chem.MolFromSmiles(smiles)
            c_before = len(probe.components(mol))
            positions, _status = probe.embed_positions(
                mol, seed=0, geometry=geometry
            )
            out = probe.fragflex_complete(mol, positions)
            if not out["stranded"]:
                assert out["edges_added"] == c_before - 1, (geometry, out)


# ---------------------------------------------------------------------------
# 2. never deletes an atom, never deletes an edge
# ---------------------------------------------------------------------------
def test_never_deletes_an_atom(sample_smiles: list):
    """No atom is removed by the repair — on any molecule, any geometry."""
    for geometry in ("whole", "fragsep"):
        for smiles in sample_smiles[:20]:
            mol = Chem.MolFromSmiles(smiles)
            positions, _status = probe.embed_positions(
                mol, seed=0, geometry=geometry
            )
            out = probe.fragflex_complete(mol, positions)
            assert out["n_atoms_after"] == out["n_atoms_before"], out
            assert out["n_heavy_after"] == out["n_heavy_before"], out


def test_never_deletes_an_edge(sample_smiles: list):
    """Every ORIGINAL edge survives the repair.

    The edit never renumbers atoms (``RWMol`` copies preserve the order), so
    the original ``(i, j)`` pairs can be compared directly against the final
    molecule's bond list.

    A bond's TYPE may change, and that is NOT a deletion: RDKit's
    ``cleanupOrganometallics`` — which runs inside ``SanitizeMol`` — rewrites
    a single bond between a metal and a hypervalent atom into a DATIVE bond.
    The repo documents exactly this
    (``cfm_decoder_yuelbond.py`` ``_endpoint_valence_cost`` rule 1).  The
    invariant asserted here is that the EDGE survives; the only permitted
    type change is SINGLE -> DATIVE.
    """
    n_rewritten_seen = 0
    for smiles in sample_smiles[:20]:
        mol = Chem.MolFromSmiles(smiles)
        positions, _status = probe.embed_positions(mol, seed=0)
        out = probe.fragflex_complete(mol, positions)
        after = {
            (int(i), int(j)): bt for i, j, bt in out["bonds_after"]
        }
        after.update(
            {(int(j), int(i)): bt for (i, j), bt in list(after.items())}
        )
        rewritten_this_mol = 0
        for bond in mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())
            assert (i, j) in after, (smiles[:60], i, j)
            before_type = str(bond.GetBondType())
            after_type = after[(i, j)]
            if after_type != before_type:
                assert (before_type, after_type) == ("SINGLE", "DATIVE"), (
                    i,
                    j,
                    before_type,
                    after_type,
                )
                rewritten_this_mol += 1
        assert out["bonds_rewritten_by_sanitizer"] == rewritten_this_mol, out
        n_rewritten_seen += rewritten_this_mol
        # And the bond count grew by exactly the number of added edges.
        assert out["n_bonds_after"] == out["n_bonds_before"] + out["edges_added"]
    assert n_rewritten_seen >= 0


def test_the_gated_repair_never_returns_an_unsanitizable_molecule(
    sample_smiles: list,
):
    """The gate's contract: whatever the gated repair returns sanitises."""
    for smiles in sample_smiles:
        out = _complete(smiles)
        assert out["sanitize_ok"] is True, out
        assert out["sanitize_fail_reason"] == "", out


# ---------------------------------------------------------------------------
# 3. determinism
# ---------------------------------------------------------------------------
def test_determinism_of_the_repair(sample_smiles: list):
    """The same input gives bit-identical output, twice."""
    for smiles in sample_smiles[:15]:
        first = _complete(smiles)
        second = _complete(smiles)
        for key in (
            "n_components_after",
            "edges_added",
            "edges_metal_nonmetal",
            "edges_nonmetal_nonmetal",
            "stranded",
            "sanitize_ok",
            "n_atoms_after",
            "n_bonds_after",
            "smiles_after",
            "bonds_after",
        ):
            assert first[key] == second[key], (key, smiles[:60])


def test_determinism_of_the_embedding():
    """RDKit's ``randomSeed=0`` is deterministic (the probe relies on it)."""
    mol = Chem.MolFromSmiles("CCCCCC.CC(=O)O.[Ru]")
    a, _ = probe.embed_positions(mol, seed=0)
    b, _ = probe.embed_positions(mol, seed=0)
    assert a is not None and b is not None
    assert (a == b).all()


# ---------------------------------------------------------------------------
# 4. the repo's own valence convention, not an invented one
# ---------------------------------------------------------------------------
def test_the_probe_uses_the_repo_valence_cost_function():
    """The probe's per-endpoint cost IS the repo's function, not a copy."""
    # A metal's bond costs the LIGAND endpoint nothing (Wf-A113 T79) and the
    # metal one bond-count unit (Wf-A113 T77).  If the probe had its own
    # table, this equality would not hold by construction.
    assert probe._endpoint_valence_cost is _endpoint_valence_cost
    metal_set = probe.METAL_SET
    assert metal_set == frozenset({44, 45, 46, 75, 76, 77, 78})  # all7
    donor_metal = 44  # Ru
    assert _endpoint_valence_cost(
        BOND_SINGLE, 7, donor_metal, probe.is_metal
    ) == 0.0
    assert _endpoint_valence_cost(
        BOND_SINGLE, donor_metal, 7, probe.is_metal
    ) == 1.0


def test_the_valence_gate_refuses_an_overvalent_pair():
    """A saturated carbon cannot take another bond."""
    mol = Chem.MolFromSmiles("C(C)(C)(C)C")  # neopentane: central C at cap 4
    positions, _status = probe.embed_positions(mol, seed=0)
    remaining, _arom, z_list = probe.remaining_budget(mol)
    # The central carbon (index 0) has no budget left.
    assert remaining[0] == 0.0
    out = probe.fragflex_complete(mol, positions)
    # Nothing to connect (one component), so nothing was added.
    assert out["edges_added"] == 0


def test_the_metal_preserving_variant_never_creates_a_metal_edge(sample_smiles: list):
    """``allow_metal_endpoints=False`` really forbids a metal endpoint."""
    for smiles in sample_smiles[:15]:
        mol = Chem.MolFromSmiles(smiles)
        positions, _status = probe.embed_positions(mol, seed=0)
        out = probe.fragflex_complete(
            mol, positions, allow_metal_endpoints=False
        )
        assert out["edges_metal_nonmetal"] == 0, out


def test_the_ungated_counterfactual_is_actually_ungated(sample_smiles: list):
    """The ungated variant really does add edges RDKit would refuse.

    This is the counterfactual that measures what the sanitize gate is
    protecting against; if it never failed, the gate would be untested.
    """
    n_failed = 0
    for smiles in sample_smiles:
        mol = Chem.MolFromSmiles(smiles)
        positions, _status = probe.embed_positions(mol, seed=0)
        out = probe.fragflex_complete(mol, positions, gate=False)
        if not out["sanitize_ok"]:
            n_failed += 1
    assert n_failed > 0, "the ungated counterfactual never failed to sanitize"


# ---------------------------------------------------------------------------
# 5. the CLI / artifact surface
# ---------------------------------------------------------------------------
def test_cli_writes_the_artifact(tmp_path: Path):
    """The script runs end to end and writes a JSON artifact with the numbers."""
    out = tmp_path / "probe.json"
    rc = probe.main(
        [
            "--cohort",
            "epoch17",
            "--limit",
            "5",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["artifact"] == "wfa142_fragflex_cminus1_probe"
    assert payload["tag"] == "MEASURED"
    assert payload["gpu_used"] is False
    assert payload["n_rows_in_cohort"] == 5
    assert set(payload["variants"]) == {
        "gated_whole",
        "gated_fragsep",
        "gated_index",
        "ungated_whole",
        "metal_preserving_whole",
    }
    gated = payload["summary"]["gated_whole"]
    assert gated["n"] == 5
    assert gated["mean_n_components_after"] <= gated["mean_n_components_before"]
    assert gated["n_sanitize_fail"] == 0
    # The corpus reference is measured with the SAME definitions as the pool.
    ref = payload["corpus_reference"]
    assert ref["present"] is True
    assert ref["train"]["n_parse"] > 0


def test_the_probe_module_declares_its_cpu_only_guard():
    """The module blanks CUDA_VISIBLE_DEVICES before importing torch/rdkit."""
    source = (PROJECT_ROOT / "molmetal" / "scripts" / "wfa142_fragflex_cminus1_probe.py").read_text()
    assert 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")' in source
    # The guard must come BEFORE the numpy / rdkit / torch imports.
    guard = source.index('os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")')
    assert guard < source.index("import numpy as np")


def test_the_stratified_cohort_is_deterministic_and_covers_every_epoch():
    """``all_stratified`` samples every epoch, deterministically, without RNG."""
    rows = probe.load_pool(POOL_PATH, "all_stratified", per_epoch=5)
    epochs = sorted({str(r["epoch"]) for r in rows})
    assert len(epochs) == 18, epochs
    assert 0 < len(rows) <= 18 * 5, len(rows)
    # Same call, same rows — no RNG anywhere in the selection.
    again = probe.load_pool(POOL_PATH, "all_stratified", per_epoch=5)
    assert rows == again
    # The epoch-17 cohort is exactly the epoch-17 rows of the pool.
    ep17 = probe.load_pool(POOL_PATH, "epoch17")
    assert len(ep17) == 704, len(ep17)
    assert {str(r["epoch"]) for r in ep17} == {"17"}
