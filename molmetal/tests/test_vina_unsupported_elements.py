# SPDX-License-Identifier: Apache-2.0
"""Guard: every Vina-rejected tmQM element must survive the atype rewrite.

WHY THIS EXISTS
---------------
Three invalid-atom-type bugs have hit the Vina path, each one a
different class and each one silent at the call site:

1. ``atype == "M"`` only -- but ``chemotype_vdw=True`` makes meeko emit
   ``Re``/``Ru``/``Pt`` directly, so the rewrite was a no-op and every
   metallodrug (including the cisplatin control) died. Pinned by
   ``test_vina_metal_atype_rewrite.py``.
2. Element extractor returned UPPERCASE PDB atom names ("PT1") against a
   mixed-case table ("Pt"), so every ligand silently took the fallback.
   Same file.
3. Boron. ``B`` is in 6.22% of tmQM complexes and Vina 1.2.7 has no ``B``
   row, so it raised ``Atom type B is not a valid AutoDock type`` and
   aborted the entire 10-pocket sweep -- not one molecule, the run.
   THIS FILE covers bug 3 and its whole class.

THE RULE UNDER TEST
-------------------
Rewrite exactly the elements Vina 1.2.7 measurably REJECTS, and nothing
else. The rejection set is measured by asking Vina, not by reading a
table (``molmetal/scripts/vina_element_atype_probe.py``). Two
properties matter and both are pinned below:

* COMPLETENESS -- every rejected element ends up with an atype Vina
  accepts, so no ligand can abort a sweep.
* PRECISION -- every natively-accepted element is left byte-for-byte
  alone. A naive "anything that is not C/H/N/O" rule would rewrite
  Ni/Cu/Co/Cd/Hg/W/Se/Fe/Mn/Zn to ``Mg`` and destroy correct typing.

RED CHECKS
----------
* Narrowing the rewrite back to the metal-only branch fails
  ``test_rejected_elements_get_a_vina_acceptable_atype``.
* Widening the rule to "every rejected-or-not element" fails
  ``test_accepted_elements_are_never_rewritten``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent


def _mod():
    name = "bench_cisplatin_real_vina"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(REPO / "molmetal" / "scripts" / "bench_cisplatin_real_vina.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _overrides():
    sys.path.insert(0, str(REPO))
    from molmetal.molmetal_lam.sbdd_env import metal_pdbqt_overrides

    return metal_pdbqt_overrides


def _row(symbol: str) -> str:
    """A meeko-shaped PDBQT ATOM line carrying ``symbol`` in BOTH the
    atom-name field [12:16] and the atype field [77:79].

    Both fields are set because that is what meeko actually emits for an
    unsupported element, and because the rewrite keys off the atom-name
    field. The record name is padded to column 12 exactly so
    ``_extract_element_from_pdbqt_line`` -- which slices [12:16] -- does
    not see a shifted field.
    """
    line = ("ATOM  " + " " * 6).ljust(12) + symbol.ljust(4)
    line = line.ljust(30) + f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
    line = line.ljust(54) + f"{0.00:6.2f}{0.00:6.2f}"
    line = line.ljust(77) + symbol.ljust(2)
    return line + f" {symbol}\n"


def _ligand(symbol: str, tmp_path: Path) -> Path:
    p = tmp_path / f"lig_{symbol}.pdbqt"
    p.write_text("REMARK SMILES probe\nROOT\n" + _row(symbol) + "ENDROOT\nTORSDOF 0\n")
    return p


def _atypes(path: Path):
    return [
        l[77:79].strip()
        for l in path.read_text().splitlines()
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 79
    ]


# ---------------------------------------------------------------------------
# COMPLETENESS — every rejected element must become acceptable to Vina
# ---------------------------------------------------------------------------
def _accepted_by_vina(atypes) -> bool:
    """Ask Vina 1.2.7 to parse a ligand carrying ``atypes``.

    This is the same oracle ``vina_element_atype_probe`` uses, run at
    test time so the rewrite cannot drift away from what Vina accepts.
    Skips (rather than fails) when ``vina`` is not importable.

    Built by rewriting the ATYPE COLUMN of a real meeko ligand rather
    than by synthesising rows. A hand-rolled row omits the atom serial
    and the residue/chain fields, and Vina's rigid-body reader rejects
    it with "Atom number \"\" is not valid" -- a failure that has nothing
    to do with the atom type under test, so every symbol would look
    rejected.
    """
    vina = pytest.importorskip("vina")
    import tempfile

    template = (
        REPO / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
        / "830c" / "cisplatin.pdbqt"
    )
    if not template.exists():
        pytest.skip(f"meeko template missing: {template}")
    tmpl = template.read_text()
    tmpl_line = next(
        l for l in tmpl.splitlines(keepends=True) if l.startswith("ATOM")
    )
    body = tmpl
    for i, at in enumerate(atypes):
        if i == 0:
            new = tmpl_line[:77] + at.ljust(2)[:2] + tmpl_line[79:]
            body = tmpl.replace(tmpl_line, new)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.pdbqt"
        p.write_text(body)
        try:
            v = vina.Vina(sf_name="vina", cpu=1, verbosity=0)
            v.set_ligand_from_file(str(p))
        except Exception:
            return False
    return True


@pytest.mark.parametrize(
    "symbol",
    sorted(_overrides().VINA_REJECTED_ELEMENTS),
)
def test_rejected_elements_get_a_vina_acceptable_atype(symbol, tmp_path):
    """COMPLETENESS, over the FULL pipeline. Boron was the live failure;
    this covers all 23 measured-rejected elements.

    Runs rewrite THEN probe, because that is the sequence the sweep
    performs (bench_cisplatin_real_vina.py, the ``if chemotype_vdw:``
    block). Asserting on the rewrite alone would be wrong for the
    Pt-group: the rewrite deliberately emits the chemotype symbol
    (``Pt``), and the probe is what converts it to ``Mg``. Testing only
    the rewrite would either fail on a correct pipeline or, worse, force
    someone to "fix" the rewrite and break the passing
    ``test_chemotype_vdw_true_writes_Pt_not_Mg`` pin.
    """
    B = _mod()
    lig = _ligand(symbol, tmp_path)
    B._rewrite_metal_atype_for_vina(lig, chemotype_vdw=True)
    B._probe_vina_accepts_chemotype_atype(lig)
    atypes = _atypes(lig)
    assert len(atypes) == 1
    assert _accepted_by_vina(atypes), (
        f"{symbol}: after rewrite+probe the atype is still {atypes[0]!r}, "
        f"which Vina 1.2.7 rejects. Any rejected element that reaches "
        f"set_ligand_from_file aborts the ENTIRE sweep, not one molecule."
    )


def test_probe_catches_a_rejected_non_pt_group_metal(tmp_path):
    """A Vina-rejected metal OUTSIDE the Pt group must not survive.

    The probe used to scan only ``PT_GROUP_ELEMENTS``, so anything the
    chemotype branch emitted for Ti/Zr/Mo/... was invisible to it and
    would abort the run. End state is the same either way — Mg — but the
    rewrite now handles non-Pt-group rejected metals directly, and the
    probe's widened scan is the backstop for anything that slips past.
    """
    B = _mod()
    lig = _ligand("Ti", tmp_path)
    B._rewrite_metal_atype_for_vina(lig, chemotype_vdw=True)
    B._probe_vina_accepts_chemotype_atype(lig)
    assert _atypes(lig) == ["Mg"], (
        f"Ti is Vina-rejected and has no chemotype atype; it must end up Mg, "
        f"got {_atypes(lig)}"
    )


def test_boron_specifically_survives(tmp_path):
    """The exact error from the dead sweep: 'Atom type B is not a valid
    AutoDock type'."""
    B = _mod()
    lig = _ligand("B", tmp_path)
    out = B._rewrite_metal_atype_for_vina(lig)
    assert _atypes(out) == ["C"], (
        "boron must map to C (nearest Vina-accepted non-metal by covalent "
        f"radius, Rcov 0.84 -> 0.76), got {_atypes(out)}"
    )


def test_arsenic_maps_to_phosphorus(tmp_path):
    """Arsenic is group 15; P is the same group and Vina-accepted."""
    B = _mod()
    lig = _ligand("As", tmp_path)
    out = B._rewrite_metal_atype_for_vina(lig)
    assert _atypes(out) == ["P"]


def test_boron_does_not_become_magnesium(tmp_path):
    """The coherence guard. Mg is a divalent METAL ion; a boron atom
    must never inherit its VDW/charge assumption."""
    B = _mod()
    lig = _ligand("B", tmp_path)
    out = B._rewrite_metal_atype_for_vina(lig)
    assert _atypes(out) != ["Mg"], (
        "B->Mg would score a covalent p-block atom with a divalent-metal "
        "VDW/charge assumption"
    )


# ---------------------------------------------------------------------------
# PRECISION — natively-accepted elements must not be touched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "symbol",
    sorted(_overrides().VINA_ACCEPTED_ELEMENTS - {"H"}),
)
def test_accepted_elements_are_never_rewritten(symbol, tmp_path):
    """PRECISION. Ni/Cu/Co/Cd/Hg/W/Se/Fe/Mn/Zn are Vina-native. A naive
    'rewrite every metal' rule turns them into Mg and destroys a correct
    atype -- a silent scoring change, not a visible failure."""
    B = _mod()
    lig = _ligand(symbol, tmp_path)
    out = B._rewrite_metal_atype_for_vina(lig)
    assert _atypes(out) == [symbol], (
        f"{symbol} is accepted by Vina 1.2.7 and must pass through "
        f"unchanged; got {_atypes(out)}"
    )


# ---------------------------------------------------------------------------
# The measured table itself must not drift from the constants
# ---------------------------------------------------------------------------
def test_nonmetal_surrogates_are_all_vina_accepted():
    """A surrogate Vina itself rejects would reintroduce the abort."""
    ov = _overrides()
    assert set(ov.VINA_NONMETAL_SURROGATES) <= ov.VINA_REJECTED_ELEMENTS
    assert set(ov.VINA_NONMETAL_SURROGATES.values()) <= ov.VINA_ACCEPTED_ELEMENTS


def test_accepted_and_rejected_are_disjoint():
    ov = _overrides()
    assert not (ov.VINA_ACCEPTED_ELEMENTS & ov.VINA_REJECTED_ELEMENTS)


def test_pt_group_is_contained_in_rejected():
    """The Pt-group metals the rewrite has always handled are all in the
    measured rejection set -- otherwise the table would silently shrink
    the coverage the chemotype path already had."""
    ov = _overrides()
    assert set(ov.PT_GROUP_ELEMENTS) <= ov.VINA_REJECTED_ELEMENTS