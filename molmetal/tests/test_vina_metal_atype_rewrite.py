# SPDX-License-Identifier: Apache-2.0
"""The Vina metal-atype rewrite must handle chemotype atypes, not just "M".

WHY THIS EXISTS
---------------
``_rewrite_metal_atype_for_vina`` used to match only ``atype == "M"``. But
``add_metal_atom_types(chemotype_vdw=True)`` -- the default the adapter passes
-- makes meeko emit the chemotype symbol directly ("Re", "Ru", "Pt", ...). So
the rewrite was a silent no-op, every metallodrug ligand died at
``set_ligand_from_file`` with "Atom type Re is not a valid AutoDock type", and
the docking sweep returned n=0 for every generated molecule AND for the
cisplatin positive control. A control that fails means the harness is broken,
not that the molecules are -- which is exactly how this was caught.

RED CHECK: narrowing the matcher back to ``atype == "M"`` makes
test_rewrites_chemotype_atype fail.
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


def _row(atom_name: str, atype: str, tail: str = "") -> str:
    """A meeko-shaped PDBQT ATOM line (>= 79 cols).

    The atom name occupies columns 12-16 (0-indexed [12:16]) with the record
    name in [0:6] = "ATOM  ", so the name starts at index 12 exactly. Building
    it as "ATOM  " + name would shift it by two and make
    ``_extract_element_from_pdbqt_line`` return "" -- which silently forces the
    fallback branch and makes the chemotype test pass for the wrong reason.
    """
    line = ("ATOM  " + " " * 6).ljust(12) + atom_name.ljust(4)
    line = line.ljust(30) + f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
    line = line.ljust(54) + f"{1.00:6.2f}{0.00:6.2f}"
    line = line.ljust(77) + atype.ljust(2)
    return line + " " + tail + "\n"


def test_rewrites_chemotype_atype(tmp_path: Path):
    """The exact shape that produced 'Atom type Re is not a valid AutoDock type'."""
    B = _mod()
    p = tmp_path / "lig.pdbqt"
    p.write_text(
        _row("RE1", "Re", "Re") + _row("C1", "C", "C") + "TORSDOF 0.00\n"
    )
    B._rewrite_metal_atype_for_vina(p, chemotype_vdw=False, fallback_atype="Mg")
    atypes = [ln[77:79].strip() for ln in p.read_text().splitlines()
              if ln.startswith(("ATOM", "HETATM"))]
    assert atypes[0] == "Mg", (
        f"chemotype atype 'Re' was not rewritten (got {atypes[0]!r}); "
        "Vina 1.2.7 has no Re row and set_ligand_from_file will reject it"
    )
    assert atypes[1] == "C", "a non-metal atom was rewritten"


def test_rewrites_generic_M(tmp_path: Path):
    """The original behaviour must be preserved."""
    B = _mod()
    p = tmp_path / "lig.pdbqt"
    p.write_text(_row("PT1", "M", "Pt"))
    B._rewrite_metal_atype_for_vina(p, chemotype_vdw=False, fallback_atype="Mg")
    assert p.read_text().splitlines()[0][77:79].strip() == "Mg"


def test_chemotype_mode_keeps_symbol(tmp_path: Path):
    """With chemotype_vdw=True the Pt-group symbol must survive the rewrite."""
    B = _mod()
    p = tmp_path / "lig.pdbqt"
    p.write_text(_row("PT1", "M", "Pt"))
    B._rewrite_metal_atype_for_vina(p, chemotype_vdw=True)
    assert p.read_text().splitlines()[0][77:79].strip() == "Pt"


def test_every_pt_group_element_is_covered():
    """No Pt-group element may fall through the matcher."""
    B = _mod()
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import PT_GROUP_ELEMENTS

    for el in PT_GROUP_ELEMENTS:
        assert el in B._PT_GROUP, f"{el} missing from the rewrite matcher"