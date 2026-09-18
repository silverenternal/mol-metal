"""Test for F5: verify molsimplify import state and document the decision.

molsimplify is intentionally NOT installed in our uv-managed .venv (see
molmetal/reports/f5_molsimplify_plan.md). This test asserts the import state
and verifies our current RDKit ETKDGv3 backend is functional on the same
class of complexes molsimplify would have handled.
"""

import pytest


def test_molsimplify_not_installed_in_uv_venv():
    """molsimplify is intentionally absent — see F5 plan for rationale.

    If this test starts failing (i.e. molsimplify becomes importable), the
    F5 plan should be revisited and the comparison script re-evaluated.
    """
    with pytest.raises(ImportError):
        import molsimplify  # noqa: F401


def test_rdkit_etkdgv3_still_works_as_fallback():
    """Confirm our actual Phase 0 backend still functions.

    The F5 plan keeps RDKit ETKDGv3 as the production embedder; this guards
    against silent backend regression.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        pytest.skip("rdkit not installed in this env")

    # Cisplatin canonical SMILES — the canonical test case from F5 plan / T4 B1.
    smiles = "N.N.Cl.Cl.[Pt]"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "RDKit failed to parse canonical cisplatin SMILES"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    params.maxIterations = 200

    cid = AllChem.EmbedMolecule(mol, params)
    assert cid >= 0, (
        "RDKit ETKDGv3 backend regression — embed failed on cisplatin "
        "(F5 plan §4 said this should succeed)"
    )
