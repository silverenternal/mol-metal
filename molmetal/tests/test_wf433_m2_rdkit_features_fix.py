"""BLK-L1 §5.7 smoke test: _rdkit_features populates the 7 previously-silent columns.

Per AGENTS.md §5.7 + the Wf-385-A1 §1 (FLAG_ONLY prevention), every trainer
flag / SCAFFOLD wire-in must have a CPU smoke that asserts the fix actually
populates what it claims.  This is the smoke for the L1 Step 1 fix on
``molmetal.scripts.wf433_m2_extended._rdkit_features``.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _fresh():
    """Re-import to defeat any cached module state."""
    import molmetal.scripts.wf433_m2_extended as m
    importlib.reload(m)
    return m


def test_nhohcount_uses_lipinski_not_rdmodelcriptors():
    m = _fresh()
    from molmetal.scripts.wf331r5_r2_w2_chemprop import PLATINAI_32D_COLUMNS
    # Aspirin has 1 OH (NHOHCount=1) and 4 N/O (NOCount=4)
    X = m._rdkit_features("CC(=O)Oc1ccccc1C(=O)O", PLATINAI_32D_COLUMNS)
    j_nhoh = PLATINAI_32D_COLUMNS.index("NHOHCount")
    j_no = PLATINAI_32D_COLUMNS.index("NOCount")
    assert X[j_nhoh] == 1.0, f"NHOHCount={X[j_nhoh]} (expected 1.0 for aspirin)"
    assert X[j_no] == 4.0, f"NOCount={X[j_no]} (expected 4.0 for aspirin)"


def test_fr_halogen_uses_fragments_module():
    m = _fresh()
    from molmetal.scripts.wf331r5_r2_w2_chemprop import PLATINAI_32D_COLUMNS
    # ClCCBr has 2 halogen atoms (fr_halogen=2)
    X_halogen = m._rdkit_features("ClCCBr", PLATINAI_32D_COLUMNS)
    X_acetate = m._rdkit_features("OC(=O)CC(=O)O", PLATINAI_32D_COLUMNS)
    j = PLATINAI_32D_COLUMNS.index("fr_halogen")
    assert X_halogen[j] == 2.0, f"ClCCBr fr_halogen={X_halogen[j]} (expected 2.0)"
    assert X_acetate[j] == 0.0, f"OC(=O)CC(=O)O fr_halogen={X_acetate[j]} (expected 0.0)"


def test_fr_al_coo_uses_fragments_module():
    m = _fresh()
    from molmetal.scripts.wf331r5_r2_w2_chemprop import PLATINAI_32D_COLUMNS
    # Malonic acid (OC(=O)CC(=O)O) has 2 fr_Al_COO
    X = m._rdkit_features("OC(=O)CC(=O)O", PLATINAI_32D_COLUMNS)
    j = PLATINAI_32D_COLUMNS.index("fr_Al_COO")
    assert X[j] == 2.0, f"malonic-acid fr_Al_COO={X[j]} (expected 2.0)"


def test_numheteroatoms_uses_rdmodelcriptors():
    m = _fresh()
    from molmetal.scripts.wf331r5_r2_w2_chemprop import PLATINAI_32D_COLUMNS
    # Aspirin: 4 heteroatoms (2 O in acetyl, 2 O in carboxyl)
    X = m._rdkit_features("CC(=O)Oc1ccccc1C(=O)O", PLATINAI_32D_COLUMNS)
    j = PLATINAI_32D_COLUMNS.index("NumHeteroatoms")
    assert X[j] == 4.0, f"aspirin NumHeteroatoms={X[j]} (expected 4.0)"


def test_no_nan_no_inf_post_fix():
    m = _fresh()
    from molmetal.scripts.wf331r5_r2_w2_chemprop import PLATINAI_32D_COLUMNS
    test_smi = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "O=C(O)c1ccccc1O",
        "ClCCBr",
        "OC(=O)CC(=O)O",
    ]
    X = np.array([m._rdkit_features(s, PLATINAI_32D_COLUMNS) for s in test_smi])
    assert not np.isnan(X).any(), "feature matrix has NaN"
    assert not np.isinf(X).any(), "feature matrix has Inf"


if __name__ == "__main__":
    test_nhohcount_uses_lipinski_not_rdmodelDescriptors()
    test_fr_halogen_uses_fragments_module()
    test_fr_al_coo_uses_fragments_module()
    test_numheteroatoms_uses_rdmodelDescriptors()
    test_no_nan_no_inf_post_fix()
    print("5/5 §5.7 smoke PASS")


# ===========================================================================
# Defect A / B / C / E — the silent-zero 32-d featurizer.
#
# The pre-fix _rdkit_features shipped 9 structurally-dead columns on the
# full 4078-row platinai_4cell cohort. Three independent causes:
#
#   A. Chi*/Kappa* looked up as rdMolDescriptors.Chi0v / .Kappa1.  Those
#      names do not exist; the rd* namespaces prefix with "Calc".  So the
#      getattr returned None and the column was silently 0.  (Two more
#      columns, BalabanJ and BertzCT, were dead for the same reason.)
#   B. The 4 PartialCharge columns summed atom *formal* charge, which is
#      0 for every atom of a neutral organic molecule — a quantity that
#      is not a partial charge.  The real quantity is Gasteiger, which
#      RDKit cannot parameterize on a Pt centre.
#   C. The trailing `else: out[j] = CalcMolWt(mol)` turned any unmatched
#      column name into a collinear duplicate of MolWt, so a typo became
#      a plausible-looking feature instead of an error.
#
# These tests each fail on the pre-fix code and pass after it.
# ===========================================================================


# Aspirin: chosen because a probe confirmed it exercises every column we
# assert below (Chi0v=6.98, Kappa1=9.25, Gasteiger max=+0.339, min=-0.478).
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# A real platinai_4cell platinum complex. RDKit's Descriptors.*
# PartialCharge return NaN on this molecule because Gasteiger has no
# parameters for the Pt(VI) centre — this is the case Defect B must handle
# by falling back rather than by emitting a constant zero.
PT_COMPLEX = (
    "COc1c(OC)c(N([O-])[O-])[c-](->[Pt+6]2(<-[Cl-])<-[n]3ccccc3-c3cccc[n]->23)"
    "c(N([O-])[O-])c1OC"
)


def _cols():
    from molmetal.scripts.wf331r5_r2_w2_chemprop import PLATINAI_32D_COLUMNS
    return PLATINAI_32D_COLUMNS


def test_all_shipped_columns_resolve_to_a_real_descriptor():
    """Defect C guard: no name in the shipped column list may be dead.

    This is the test that would have caught A, B and C.  Before the fix
    `getattr(rdMolDescriptors, "Chi0v")` returned None and the column was
    filled with 0 with no error anywhere.
    """
    m = _fresh()
    missing = m.unresolved_descriptor_columns(_cols())
    assert missing == [], (
        f"columns with no descriptor implementation: {missing}. Every name "
        "in the shipped 32-d list must resolve to something that runs."
    )


def test_chi_and_kappa_columns_are_non_zero():
    """Defect A: Chi*/Kappa* must resolve (the "Calc" prefix) and be non-zero."""
    m = _fresh()
    cols = _cols()
    X = m._rdkit_features(ASPIRIN, cols)
    # Probe-confirmed RDKit values on aspirin.
    expected = {
        "Chi0v": 6.981360,
        "Chi0n": 6.981360,
        "Chi1n": 3.617454,
        "Kappa1": 9.249606,
        "Kappa2": 3.709251,
        "Kappa3": 2.297415,
    }
    for name, want in expected.items():
        got = float(X[cols.index(name)])
        assert want != 0.0
        assert got == pytest.approx(want, abs=1e-4), (
            f"{name}={got} expected {want}; a zero here means the lookup "
            "missed and the column is dead."
        )


def test_balabanj_and_bertzct_are_non_zero():
    """Defect A, same root cause: CalcBalabanJ / CalcBertzCT also do not exist."""
    m = _fresh()
    cols = _cols()
    X = m._rdkit_features(ASPIRIN, cols)
    assert X[cols.index("BalabanJ")] != 0.0, "BalabanJ is a dead column"
    assert X[cols.index("BertzCT")] != 0.0, "BertzCT is a dead column"


def test_partial_charges_non_zero_on_a_neutral_molecule():
    """Defect B: formal charge is not partial charge.

    Aspirin is globally neutral, so the pre-fix code (summing
    GetFormalCharge) returned exactly 0.0 for all 4 columns.  Gasteiger
    gives a real charge distribution.
    """
    m = _fresh()
    cols = _cols()
    X = m._rdkit_features(ASPIRIN, cols)
    values = {c: float(X[cols.index(c)]) for c in
              ("MaxPartialCharge", "MinPartialCharge",
               "MaxAbsPartialCharge", "MinAbsPartialCharge")}
    for name, v in values.items():
        assert v != 0.0, (
            f"{name}=0.0 on a neutral molecule — the old formal-charge sum "
            "is structurally zero, not a measurement."
        )
    # And the signs must be physically sensible: some atom is electron-poor,
    # some electron-rich, so max > 0 > min and max_abs >= |min|.
    assert values["MaxPartialCharge"] > 0.0
    assert values["MinPartialCharge"] < 0.0
    assert values["MaxAbsPartialCharge"] >= abs(values["MinPartialCharge"]) - 1e-9


def test_partial_charge_fallback_is_counted_not_silent():
    """Defect B: the Pt fallback must be COUNTED, never a silent zero.

    On a platinum complex RDKit's own Descriptors.*PartialCharge are NaN
    (Gasteiger cannot parameterize the Pt centre).  The resolver must
    either produce a real value or record the molecule as failed — an
    uncounted 0.0 is indistinguishable downstream from a real neutral.
    """
    import molmetal.data.rdkit_2d_extended as r2d
    importlib.reload(r2d)
    from rdkit import Chem

    r2d.reset_partial_charge_stats()
    mol = Chem.MolFromSmiles(PT_COMPLEX)
    assert mol is not None
    vals = r2d._partial_charges(mol)
    stats = r2d.partial_charge_stats()

    assert stats["n_molecules"] == 1, "every molecule must be counted"
    tiered = (
        stats["n_rdkit_native"] + stats["n_gasteiger_full"]
        + stats["n_gasteiger_organic"] + stats["n_failed"]
    )
    assert tiered == stats["n_molecules"], (
        f"a molecule escaped the provenance counters: {stats}"
    )
    # RDKit's native descriptor cannot do this molecule, so the count must
    # land in one of the explicit fallback tiers.
    assert stats["n_rdkit_native"] == 0, (
        "expected RDKit's native partial charge to be undefined on a Pt "
        f"complex; stats={stats}"
    )
    fallback = stats["n_gasteiger_full"] + stats["n_gasteiger_organic"]
    assert fallback + stats["n_failed"] >= 1
    if stats["n_failed"]:
        # A counted failure emits 0.0 — but it is visible in the counters.
        assert vals == (0.0, 0.0, 0.0, 0.0)
    else:
        assert any(v != 0.0 for v in vals), (
            "tier says we computed charges but all four are zero"
        )


def test_uncomputable_molecule_is_counted_as_failed():
    """A molecule no charge model can handle must be COUNTED, not silent.

    One of the 3 real cohort failures: a Pt complex whose remaining
    atoms are Se / As, which Gasteiger also cannot parameterize, so every
    tier fails and the resolver must record `n_failed` and emit 0.0.
    """
    import molmetal.data.rdkit_2d_extended as r2d
    importlib.reload(r2d)
    from rdkit import Chem

    r2d.reset_partial_charge_stats()
    mol = Chem.MolFromSmiles(
        "C[As]1(C)c2ccccc2[S-]->[Pt+2]<-12<-[S-]c1ccccc1[As]->2(C)C"
    )
    assert mol is not None
    vals = r2d._partial_charges(mol)
    stats = r2d.partial_charge_stats()
    assert stats["n_molecules"] == 1
    assert stats["n_failed"] == 1, (
        f"a fully unparameterizable molecule must land in n_failed: {stats}"
    )
    assert vals == (0.0, 0.0, 0.0, 0.0)


def test_cohort_fallback_rate_is_reported_not_assumed():
    """The 4 charge columns must mostly fall back on a metallodrug cohort.

    If RDKit's native partial charges ever become usable on Pt, this test
    is the one that says so instead of the number silently changing.
    """
    import csv

    import molmetal.data.rdkit_2d_extended as r2d
    importlib.reload(r2d)
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = list(
        csv.DictReader(open("molmetal/data/platinai_4cell_rdkit_features.csv"))
    )
    r2d.reset_partial_charge_stats()
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        assert mol is not None
        r2d._partial_charges(mol)
    stats = r2d.partial_charge_stats()
    assert stats["n_molecules"] == len(rows)
    total = sum(
        stats[k] for k in
        ("n_rdkit_native", "n_gasteiger_full", "n_gasteiger_organic", "n_failed")
    )
    assert total == stats["n_molecules"], f"uncounted molecules: {stats}"
    # MEASURED on this host: RDKit's native descriptor handles 13/4078; the
    # rest must be accounted for by an explicit fallback tier or n_failed.
    assert stats["n_rdkit_native"] < 0.05 * len(rows), (
        f"RDKit native partial charge now covers {stats['n_rdkit_native']}/"
        f"{len(rows)} rows — the fallback path may no longer be the norm; "
        "re-measure and update this assertion deliberately."
    )


def test_unknown_column_raises_instead_of_becoming_a_molwt_duplicate():
    """Defect C: a typo must be loud, not a collinear duplicate of MolWt."""
    m = _fresh()
    with pytest.raises(m.UnresolvedDescriptorColumn):
        m._rdkit_features(ASPIRIN, ("MolWt", "Chi0v_typo", "Kappa9"))


def test_unresolved_helper_lists_only_the_bad_names():
    m = _fresh()
    assert m.unresolved_descriptor_columns(("MolWt", "NotADescriptor")) == [
        "NotADescriptor"
    ]
    assert m.unresolved_descriptor_columns(_cols()) == []


def test_parse_failure_still_returns_zeros():
    """The documented graceful contract for unparseable SMILES must survive."""
    m = _fresh()
    cols = _cols()
    X = m._rdkit_features("this is not a smiles", cols)
    assert X.shape == (len(cols),)
    assert np.all(X == 0.0)


def test_shipped_32d_csv_has_no_all_zero_column():
    """End-to-end: the on-disk artifact must not ship a dead column.

    Guards Defect D's regeneration too — if the CSV ever drops back to a
    truncated / partially-populated build, this fails.
    """
    import csv

    import pytest  # noqa: F401  (import kept local to the test)

    path = "molmetal/data/platinai_4cell_rdkit_extended_features.csv"
    cols = _cols()
    with open(path) as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    assert len(rows) >= 4000, (
        f"{path} has only {len(rows)} rows; the base 12-d file has 4078, so "
        "this CSV is a truncated build (Wf-251 ran it with --max-rows 200)."
    )
    for c in cols:
        assert c in header, f"{c} missing from {path} header"
    for c in cols:
        vals = np.array([float(r[c]) for r in rows])
        assert vals.max() - vals.min() > 0.0, (
            f"{c} is constant ({vals[0]}) across all {len(rows)} rows"
        )
