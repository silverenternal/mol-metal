# SPDX-License-Identifier: Apache-2.0
"""The metal-deletion control must actually delete the metal.

WHY THIS EXISTS
---------------
``strip_metals`` was first written with
``Chem.DeleteSubstructs(mol, smarts, onlyFrags=True)``. That call was VERIFIED
NOT TO DELETE ANYTHING: on ``[Pt](Cl)(Cl)(N)N`` the SMARTS matches once and
``DeleteSubstructs`` returns the molecule unchanged, Pt still in the string.

Left alone, the control would then compare "embedding with the metal" against
"embedding with the metal still there", find a repair rate of zero on both
cohorts, and print a confident VERDICT that the localisation does not survive
the control. The conclusion would have been an artefact of a deletion that
never happened -- and it would have looked exactly like a real finding.

So the deletion itself needs pinning. These tests assert the observable
effect, not the implementation: after stripping, no metal atom may remain.

RED CHECK: restore the ``DeleteSubstructs`` implementation and
``test_metal_is_actually_removed`` fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _mod():
    name = "wf_blkc_l5_metal_deletion_control"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(REPO / "molmetal" / "scripts" / f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _has_metal(smiles: str) -> bool:
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    z = {Chem.GetPeriodicTable().GetAtomicNumber(s) for s in _mod().TRAIN_METALS}
    return any(a.GetAtomicNum() in z for a in m.GetAtoms())


def test_metal_is_actually_removed():
    m = _mod()
    for smi, expect_n_atoms in (
        ("[Pt](Cl)(Cl)(N)N", 4),
        ("CC[Pt](Cl)(N)CC", 6),   # splits to CC.CC.Cl.N
        ("[Rh](Cl)(Cl)(Cl)(N)N", 5),
    ):
        out = m.strip_metals(smi)
        assert out is not None, f"strip_metals returned None for {smi}"
        assert not _has_metal(out), (
            f"strip_metals({smi!r}) returned {out!r}, which STILL CONTAINS a "
            "metal. The control would then measure the molecule against "
            "itself and report a repair rate of zero for the wrong reason."
        )
        from rdkit import Chem

        assert Chem.MolFromSmiles(out).GetNumAtoms() == expect_n_atoms, (
            f"strip_metals({smi!r}) -> {out!r}: expected {expect_n_atoms} atoms "
            "after removing exactly one metal"
        )


def test_all_six_train_metals_are_stripped():
    """The cohort is filtered to the same six metals; the deletion must match.

    A metal in TRAIN_METALS that the SMARTS misses would be compared against
    itself inside the filtered cohort, which is the failure this whole file
    exists to detect.
    """
    m = _mod()
    assert m.TRAIN_METALS == frozenset({"Pt", "Ru", "Ir", "Rh", "Os", "Re"}), (
        f"the six train metals changed to {sorted(m.TRAIN_METALS)}; the "
        "cohort filter and the deletion must be kept in step, and the "
        "localisation JSON this control checks was measured on these six"
    )
    for sym in sorted(m.TRAIN_METALS):
        smi = f"[{sym}](Cl)(N)N"
        out = m.strip_metals(smi)
        assert out is not None and not _has_metal(out), (
            f"{sym} survived strip_metals: {out!r}"
        )


def test_metal_free_molecule_returns_none_rather_than_a_copy():
    """Nothing to delete must NOT look like a successful repair.

    A failure that has no metal cannot be 'repaired by deleting the metal'.
    Returning the input unchanged would count it as repaired, which is the
    opposite of the truth, and would inflate the repair rate on any cohort
    that contains metal-free failures.
    """
    m = _mod()
    assert m.strip_metals("CCO") is None, (
        "strip_metals('CCO') returned a molecule; there was no metal to "
        "delete, and returning one would let a metal-free failure be counted "
        "as repaired"
    )


def test_result_is_sanitised_not_dangling():
    """Removing an atom must not leave neighbours with impossible valences."""
    m = _mod()
    from rdkit import Chem

    out = m.strip_metals("[Pt](Cl)(Cl)(N)N")
    assert out is not None
    mol = Chem.MolFromSmiles(out)
    assert mol is not None, f"stripped SMILES does not reparse: {out!r}"
    # A silent valence failure here would be indistinguishable from an
    # embeddability failure, so the two must never be conflated.
    Chem.SanitizeMol(mol)
    assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms()), (
        f"{out!r} retains radicals after sanitisation; an unsanitised graph "
        "fails to embed for a reason that has nothing to do with the metal"
    )


def test_control_cohort_is_real_tmqm_not_decoder_output():
    """The control must not be the generator measured against itself.

    The BLK-C-L4 audit showed ``real_tmqm_reference.csv`` is 636/636 decoder
    output carrying bracket-atom notation. Using it as the "real" cohort would
    make the control circular: our own bond head's output would set the
    baseline our own bond head is then judged against.
    """
    m = _mod()
    src = (REPO / "molmetal" / "scripts" / f"{m.__name__}.py").read_text()
    assert "tmqm_parsed.csv" in src, "the real cohort no longer comes from tmqm_parsed.csv"
    assert 'default=REPO / "real_tmqm_reference.csv"' not in src, (
        "the control cohort was switched to real_tmqm_reference.csv, which "
        "the BLK-C-L4 audit showed is decoder output, not real tmQM"
    )

def test_embeddable_reproduces_the_recorded_generated_number():
    """The control must measure the generated cohort the way the original did.

    The first version of `embeddable` omitted `Chem.AddHs` and reported an
    embed rate of **0.0 for both cohorts** — which produced a confident,
    entirely artefactual VERDICT that the L5 localisation "does not survive the
    control". The bug and the finding had the same shape, which is why the
    number has to be pinned rather than trusted.

    This asserts the control reproduces `metrics/by_metric/
    cfm_embed_failure_localisation.json`'s cohort header: 66 molecules,
    54 embeddable, 0.8182.
    """
    m = _mod()
    from rdkit import Chem

    csv_path = (
        REPO / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10"
        / "generated_smiles.csv"
    )
    if not csv_path.is_file():
        return  # cohort file absent; nothing to pin against
    import csv as _csv

    rows = [
        r for r in _csv.DictReader(csv_path.open(encoding="utf-8"))
        if (r.get("smiles") or "").strip()
    ]
    ok = sum(1 for r in rows if m.embeddable(Chem.MolFromSmiles(r["smiles"])))
    assert (len(rows), ok) == (66, 54), (
        f"the control measured {ok}/{len(rows)} embeddable on the generated "
        "cohort; the recorded protocol gives 54/66. A protocol drift here "
        "makes every downstream comparison against real tmQM meaningless, "
        "and it fails SILENTLY — the script still prints a verdict."
    )
