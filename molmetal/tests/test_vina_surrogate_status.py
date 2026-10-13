# SPDX-License-Identifier: Apache-2.0
"""A ligand scored as a covalent stand-in must not be labelled a clean measurement.

WHY THIS EXISTS
---------------
On 2026-10-12 a third invalid-atom-type class was fixed in
``_rewrite_metal_atype_for_vina``: Vina 1.2.7 rejects 23 of the 44 elements
that occur in tmQM, so B was rewritten to C and As to P. That turned 22 hard
``PDBQT parsing error`` crashes into silent substitutions -- which is the fix
working -- but it also broke the ``status`` field the sweep records.

The status decision reads ``_probe_vina_accepts_chemotype_atype``, which scans
the POST-rewrite atype column for a symbol in ``VINA_REJECTED_ELEMENTS``. After
the rewrite a boron atom reads ``C`` and a titanium atom reads ``Mg``; neither
is in the rejected set, so the probe finds nothing and returns the bare
``fallback_atype``. That string does not end in ``-fallback``, so the caller
fell through to ``MEASURED_CHEMOTYPE_VDW``.

So the sweep recorded ligands scored as carbon and magnesium as having been
scored with their true chemotype. The rewrite's own docstring calls this the
exact hazard it warns about: "a wrong rewrite silently changes which molecule
Vina thinks it's scoring and still records it as a real measurement."

The information needed to label the record honestly is destroyed by the
rewrite itself, so it has to be captured during the rewrite. That is what the
``record`` parameter carries.

RED CHECK: removing the ``surrogate`` branch from the status decision makes
``test_surrogate_is_not_reported_as_chemotype`` fail. Reported in
docs/CFM_STATUS_2026_10_12.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def _row(atom_name: str, atype: str, elem: str = "") -> str:
    """A meeko-shaped PDBQT ATOM line (>= 79 cols).

    The atom name starts at index 12: ``"ATOM  "`` is 6 chars, PDB records pad
    to column 13. Building the name at index 8 shifts it two columns and makes
    ``_extract_element_from_pdbqt_line`` return "", which forces the fallback
    branch and would make every test here pass for the wrong reason.
    """
    line = ("ATOM  " + " " * 6).ljust(12) + atom_name.ljust(4)
    line = line.ljust(30) + f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
    line = line.ljust(54) + f"{1.00:6.2f}{0.00:6.2f}"
    line = line.ljust(77) + atype.ljust(2)
    return line + (" " + elem if elem else "") + "\n"


def _status_for(tmp_path: Path, body: str) -> str:
    """Run the production rewrite + the production status decision.

    The probe is stubbed to the answer the real probe gives after a rewrite
    (the bare ``fallback_atype``, because no rejected symbol survives in the
    file), so this pins the DECISION and not Vina's availability. Everything
    else is production code: ``_rewrite_metal_atype_for_vina`` fills ``subs``,
    ``_vina_record_status`` decides.
    """
    B = _mod()
    p = tmp_path / "lig.pdbqt"
    p.write_text(body + "TORSDOF 0.00\n")

    subs: list = []
    B._rewrite_metal_atype_for_vina(p, chemotype_vdw=True, record=subs)
    return B._vina_record_status(subs, "Mg")


def test_surrogate_is_not_reported_as_chemotype(tmp_path: Path):
    """The exact mislabelling: boron scored as carbon."""
    status = _status_for(tmp_path, _row("B1", "B", "B") + _row("C1", "C", "C"))
    assert status == "MEASURED_SURROGATE_ATOM", (
        f"a boron ligand rewritten to carbon was labelled {status!r}; Vina "
        "returned a real number for a molecule it believes is a hydrocarbon, "
        "and MEASURED_CHEMOTYPE_VDW claims otherwise"
    )


def test_arsenic_surrogate_is_not_reported_as_chemotype(tmp_path: Path):
    status = _status_for(tmp_path, _row("AS1", "As", "As") + _row("C1", "C", "C"))
    assert status == "MEASURED_SURROGATE_ATOM", (
        f"arsenic scored as phosphorus was labelled {status!r}"
    )


def test_rejected_metal_fallback_is_not_reported_as_chemotype(tmp_path: Path):
    """Titanium is Vina-rejected and falls to the Mg branch, not the surrogate
    branch -- but the label must still not be a clean chemotype measurement."""
    status = _status_for(tmp_path, _row("TI1", "Ti", "Ti") + _row("C1", "C", "C"))
    assert status != "MEASURED_CHEMOTYPE_VDW", (
        "a titanium ligand rewritten to magnesium was labelled a chemotype "
        "measurement"
    )


def test_true_chemotype_still_reports_chemotype(tmp_path: Path):
    """The fix must not downgrade a ligand Vina scored with its real symbol."""
    status = _status_for(tmp_path, _row("PT1", "M", "PT") + _row("C1", "C", "C"))
    assert status == "MEASURED_CHEMOTYPE_VDW", (
        "a platinum ligand scored with the real Pt atype must stay "
        f"MEASURED_CHEMOTYPE_VDW, got {status!r}"
    )


def test_record_captures_every_substitution_kind(tmp_path: Path):
    """``record`` is the only channel the label has, so it must be complete."""
    B = _mod()
    p = tmp_path / "lig.pdbqt"
    p.write_text(
        _row("PT1", "M", "PT")      # chemotype
        + _row("B1", "B", "B")      # surrogate
        + _row("C1", "C", "C")      # untouched
    )
    subs: list = []
    B._rewrite_metal_atype_for_vina(p, chemotype_vdw=True, record=subs)
    kinds = {k for _, _, k in subs}
    assert kinds == {"chemotype", "surrogate"}, (
        f"record captured kinds {kinds!r}; a carbon must not appear and both "
        "the chemotype and surrogate substitutions must"
    )
    assert ("B", "C", "surrogate") in subs, (
        f"boron substitution missing or wrong: {subs!r}"
    )