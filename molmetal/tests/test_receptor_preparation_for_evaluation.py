"""Conformer/name normalization must preserve a traceable physical receptor."""
import json

import pytest

from molmetal.scripts import receptor_preparation_for_evaluation as prep


def atom(name, xyz, alt="", occupancy=1.0, element="C", charge=None, serial=1):
    columns = list(" " * 80)
    for start, text in ((0, "ATOM  "), (6, f"{serial:5d}"), (12, f" {name:<3}"),
                        (16, alt or " "), (17, "ASN"), (21, "B"), (22, " 255"),
                        (30, f"{xyz[0]:8.3f}"), (38, f"{xyz[1]:8.3f}"), (46, f"{xyz[2]:8.3f}"),
                        (54, f"{occupancy:6.2f}")):
        columns[start:start + len(text)] = text
    if charge is None:
        columns[76:78] = f"{element:>2}"
    else:
        columns[70:76] = f"{charge:6.3f}"
        columns[77:79] = f"{element:<2}"
    return "".join(columns) + "\n"


def test_altloc_choice_is_residue_wide_not_best_atom_mix(tmp_path):
    original, derived = tmp_path / "original.pdb", tmp_path / "selected.pdb"
    content = (atom("CA", [0, 0, 0], "A", .4, serial=1)
               + atom("CB", [1, 0, 0], "A", .9, serial=2)
               + atom("CA", [2, 0, 0], "B", .7, serial=3)
               + atom("CB", [3, 0, 0], "B", .7, serial=4))
    original.write_text(content)
    report = prep.select_receptor_altloc(original, derived)
    assert report["choices"][0]["selected_altloc"] == "B"
    assert {record["altloc"] for record in report["selected_atom_records"]} == {"B"}
    assert len(report["excluded_alternative_atom_records"]) == 2
    assert all(line[16] == " " for line in derived.read_text().splitlines())
    assert original.read_text() == content


@pytest.mark.parametrize("labels,expected", [("BA", "A"), ("CB", "B")])
def test_occupancy_tie_prefers_a_then_lexicographic(tmp_path, labels, expected):
    original, derived = tmp_path / "original.pdb", tmp_path / "selected.pdb"
    original.write_text("".join(atom("CA", [i, 0, 0], alt, .5, serial=i + 1) for i, alt in enumerate(labels)))
    assert prep.select_receptor_altloc(original, derived)["choices"][0]["selected_altloc"] == expected


def test_high_occupancy_conformer_cannot_silently_omit_heavy_atom(tmp_path):
    original = tmp_path / "original.pdb"
    original.write_text(atom("CA", [0, 0, 0], "A", .9, serial=1)
                        + atom("CA", [1, 0, 0], "B", .1, serial=2)
                        + atom("CB", [2, 0, 0], "B", .1, serial=3))
    with pytest.raises(ValueError, match="lacks heavy atoms"):
        prep.select_receptor_altloc(original, tmp_path / "selected.pdb")


@pytest.fixture
def equivalent_oxygens(tmp_path):
    c, o, oxt = [15.773, 8.529, -2.507], [14.630, 8.366, -2.033], [16.405, 9.608, -2.454]
    original, prepared, derived = (tmp_path / name for name in ("source.pdb", "meeko.pdbqt", "restored.pdbqt"))
    original.write_text(atom("C", c) + atom("O", o, element="O") + atom("OXT", oxt, element="O"))
    prepared.write_text(atom("C", c, charge=.065)
                        + atom("O", oxt, element="OA", charge=-.538)
                        + atom("OXT", o, element="OA", charge=-.538))
    return original, prepared, derived


def test_equivalent_oxygen_names_are_restored_without_moving_atoms(equivalent_oxygens):
    original, prepared, derived = equivalent_oxygens
    before = prepared.read_text()
    with pytest.raises(ValueError, match="moved"):
        prep.audit_receptor(original, prepared)
    report = prep.restore_equivalent_terminal_oxygen_names(original, prepared, derived)
    assert report["audit"]["max_heavy_atom_displacement_A"] == 0
    assert len(report["mapping"]) == 2
    assert prepared.read_text() == before
    for left, right in zip(before.splitlines(), derived.read_text().splitlines()):
        assert left[:12] == right[:12]
        assert left[16:] == right[16:]


def test_oxygen_names_with_different_charges_cannot_be_repaired(equivalent_oxygens):
    original, prepared, derived = equivalent_oxygens
    prepared.write_text(prepared.read_text().replace("-0.538", "-0.300", 1))
    with pytest.raises(ValueError, match="Charges differ"):
        prep.restore_equivalent_terminal_oxygen_names(original, prepared, derived)


def test_real_missing_heavy_atoms_stay_failed(monkeypatch, tmp_path):
    original = tmp_path / "missing_sidechain.pdb"
    original.write_text(atom("CA", [0, 0, 0]))
    calls = []

    def strict(source, destination):
        calls.append(source)
        return {"passed": False, "returncode": 1, "error": "Missing CG OD1 OD2", "stderr": "template mismatch"}

    monkeypatch.setattr(prep, "prepare_receptor", strict)
    report = prep.prepare_receptor_for_evaluation(original, tmp_path / "report")
    assert report["passed"] is False
    assert report["error"] == "Missing CG OD1 OD2"
    assert calls == [original]
    assert json.loads((tmp_path / "report/preparation_report.json").read_text())["source_unchanged"]
