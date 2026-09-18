"""Real graph/stoichiometry regressions for the ThiolEne search witness."""
from collections import Counter

import pytest
from rdkit import Chem

from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import REACTION_RULES


def term(smiles):
    return MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)


def elements(mol):
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())


@pytest.mark.parametrize('a,b', [
    ('Cc1ccc(-c2ccc(S)cc2)cc1', 'C#CCc1ccc(/C=C/C)cc1'),
    ('CS', 'C=CC'), ('c1ccccc1S', 'C=C'), ('CSC.CS', 'C=CC'),
    ('C=CCS', 'CC=C'),
])
def test_real_sulfur_carbon_join_and_total_atom_conservation(a, b):
    lhs, rhs = term(a), term(b)
    ma, mb = lhs.to_rdkit(), rhs.to_rdkit()
    before = lhs.canonical_smiles(), rhs.canonical_smiles()
    products = REACTION_RULES['ThiolEne'].reduce((lhs, rhs))
    assert len(products) == 1
    product = products[0].to_rdkit()
    assert len(Chem.GetMolFrags(product)) == len(Chem.GetMolFrags(ma))+len(Chem.GetMolFrags(mb))-1
    assert elements(product) == elements(ma) + elements(mb)
    # New bond must connect sulfur from reactant 1 to an original alkene
    # endpoint from reactant 2, not an arbitrary carbon elsewhere in the graph.
    offset = ma.GetNumAtoms()
    alkene_atoms = {idx+offset for bond in mb.GetBonds()
                    if bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetBeginAtom().GetAtomicNum() == 6
                    and bond.GetEndAtom().GetAtomicNum() == 6
                    for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())}
    cross_bonds = [bond for bond in product.GetBonds()
                   if (bond.GetBeginAtomIdx() < offset) != (bond.GetEndAtomIdx() < offset)]
    assert len(cross_bonds) == 1
    indices = [cross_bonds[0].GetBeginAtomIdx(), cross_bonds[0].GetEndAtomIdx()]
    sulfur_idx, carbon_idx = sorted(indices)
    assert product.GetAtomWithIdx(sulfur_idx).GetAtomicNum() == 16
    assert carbon_idx in alkene_atoms
    assert sum(b.GetBondType() == Chem.BondType.TRIPLE for b in product.GetBonds()) == sum(
        b.GetBondType() == Chem.BondType.TRIPLE for m in (ma, mb) for b in m.GetBonds())
    assert (lhs.canonical_smiles(), rhs.canonical_smiles()) == before


def test_terminal_alkene_gives_anti_markovnikov_connected_product():
    product = REACTION_RULES['ThiolEne'].reduce((term('CS'), term('C=CC')))[0]
    assert product.canonical_smiles() == 'CCCSC'


def test_explicit_hydrogen_atom_is_transferred_without_deletion():
    a, b = [MoleculeClosedTerm.from_rdkit(Chem.AddHs(Chem.MolFromSmiles(s)))
            for s in ('CS', 'C=CC')]
    product = REACTION_RULES['ThiolEne'].reduce((a, b))[0].to_rdkit()
    assert product.GetNumAtoms() == a.n_atoms + b.n_atoms
    assert elements(product) == elements(a.to_rdkit()) + elements(b.to_rdkit())
    assert Chem.MolToSmiles(Chem.RemoveHs(product)) == 'CCCSC'


@pytest.mark.parametrize('a,b', [
    ('CSC', 'C=CC'), ('CSSC', 'C=CC'), ('CS(=O)C', 'C=CC'),
    ('C[S-]', 'C=CC'), ('S', 'C=CC'), ('CS', 'CC'), ('CS', 'C#CC'),
    ('CCN=[N+]=[N-]', 'C=C'),
])
def test_non_thiol_and_non_alkene_are_not_applicable(a, b):
    rule = REACTION_RULES['ThiolEne']
    reactants = term(a), term(b)
    assert not rule.can_apply(*reactants)
    assert rule.reduce(reactants) == []


@pytest.mark.parametrize('name,a,b', [
    ('CuAAC', 'CCN=[N+]=[N-]', 'C#CC'),
    ('SPAAC', '[N-]=[N+]=NCc1ccccc1', 'C1CCCC#CCC1'),
    ('Suzuki', 'OB(O)c1ccccc1', 'Brc1ccccc1'),
    ('AmideCoupling', 'CC(=O)O', 'CN'),
])
def test_other_public_click_rules_produce_connected_main_product(name, a, b):
    products = REACTION_RULES[name].reduce((term(a), term(b)))
    assert products
    assert all(len(Chem.GetMolFrags(p.to_rdkit())) == 1 for p in products)


def test_spc_explicit_nitrogen_byproduct_is_retained():
    products = REACTION_RULES['SPC'].reduce((term('CCN=[N+]=[N-]'), term('CP')))
    assert products
    assert any(len(Chem.GetMolFrags(p.to_rdkit())) > 1 for p in products)
