# SPDX-License-Identifier: Apache-2.0
"""The Vina kcal mean must arrive with its SIZE CONFOUND attached.

THE DEFECT
----------
``cfm_vina_generated_sweep.py`` published one number per pocket --
``kcal_mean_over_docked`` -- and labelled its survivor conditioning well
(``_survivor_conditioned: true``, ``_mean_excludes: 14``). That label fixes the
DENOMINATOR problem. It does not touch a second one, which is larger.

MEASURED on the 128 docked records in
``molmetal/reports/cfm_vina_sweep_2026_10_12/per_record.jsonl``:

    r(kcal_mol, n_atoms_ligand) = +0.2953
    slope                         = +0.0538 kcal/mol per atom
    mean n_atoms_ligand           = 37.56     (bare cisplatin: 4 heavy atoms)

A flat 32-atom size shift alone predicts +1.72 kcal/mol. The published
-5.81 / -5.95 means therefore describe "a Mg-fallback stand-in of ~38 atoms",
not binding affinity, and nothing in the file said so. A reader quoting the
headline as affinity would be wrong, and the file gave them no way to know.

WHAT IS PINNED HERE
-------------------
That the confound travels WITH the number. The mean is not deleted -- it is
what was measured -- but a report that carries it without the caveat is the
defect, so the caveat is asserted as part of the output contract.

RED CHECK: delete the ``_kcal_is_size_confounded`` block and
``test_size_confound_travels_with_the_mean`` fails.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "molmetal" / "scripts" / "cfm_vina_generated_sweep.py"
PER_RECORD = (
    REPO
    / "molmetal"
    / "reports"
    / "cfm_vina_sweep_2026_10_12"
    / "per_record.jsonl"
)


def _module():
    if "cfm_vina_gen_sweep_test" in sys.modules:
        return sys.modules["cfm_vina_gen_sweep_test"]
    spec = importlib.util.spec_from_file_location(
        "cfm_vina_gen_sweep_test", str(SCRIPT)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["cfm_vina_gen_sweep_test"] = m
    spec.loader.exec_module(m)
    return m


def _real_records():
    if not PER_RECORD.is_file():
        import pytest

        pytest.skip(f"no Vina records on disk: {PER_RECORD}")
    return [
        json.loads(line)
        for line in PER_RECORD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summarise(records):
    # Call the PRODUCTION summariser by name rather than rediscovering it by
    # prefix: a `for name in dir(m) if name.startswith("summar")` loop would
    # silently pick the wrong function the day a second one appears, and the
    # test would keep passing against a code path nothing ships.
    return _module().summarise_pocket("3eml", records)


def test_size_confound_travels_with_the_mean():
    out = _summarise(_real_records())
    if out.get("kcal_mean_over_docked") is None:
        import pytest

        pytest.skip("no docked records to summarise")
    assert out.get("_kcal_is_size_confounded") is True, (
        "a kcal mean is being published with no statement that it is "
        "size-confounded. On this cohort r=+0.2953 against ligand size and a "
        "32-atom gap predicts +1.72 kcal/mol, so the number cannot be read as "
        "affinity."
    )
    text = out.get("_size_confound", "")
    assert "n_atoms_ligand" in text, "the caveat must name the confound variable"
    assert "37.56" in text, "the caveat must quote the cohort mean ligand size"
    assert "4" in text, "the caveat must quote the cisplatin comparison size"


def test_citable_line_excludes_absolute_kcal():
    """The admissibility rule must be in the file, not only in a report."""
    out = _summarise(_real_records())
    if out.get("kcal_mean_over_docked") is None:
        import pytest

        pytest.skip("no docked records to summarise")
    cit = out.get("_citable", "")
    assert cit, "no _citable statement emitted"
    assert "docked_fraction" in cit
    assert "Not citable" in cit
    assert "cisplatin" in cit


def test_survivor_labels_are_not_regressed_by_the_new_block():
    """The denominator labels must survive alongside the size caveat."""
    out = _summarise(_real_records())
    if out.get("kcal_mean_over_docked") is None:
        import pytest

        pytest.skip("no docked records to summarise")
    assert out.get("_survivor_conditioned") is True
    assert isinstance(out.get("_mean_excludes"), int)


def test_confound_block_absent_when_nothing_docked():
    """A pocket with no docked records must not claim a size confound.

    Built by FILTERING the real records down to the DOCK_FAILED ones rather
    than slicing the first N. An earlier version took rows[:5], which still
    contained three MEASURED entries, so the branch was never exercised and
    the test SKIPPED while appearing to cover it -- a skip that looks like a
    pass in a summary line.
    """
    failed = [r for r in _real_records() if r.get("status") == "DOCK_FAILED"]
    assert failed, "expected some DOCK_FAILED records in the real cohort"
    out = _summarise(failed)
    assert out["n_docked"] == 0
    assert out["kcal_mean_over_docked"] is None
    assert "_kcal_is_size_confounded" not in out, (
        "with no kcal value there is no mean to qualify; emitting the caveat "
        "anyway would attach a size argument to an absent measurement"
    )
    assert out["_survivor_conditioned"] is None