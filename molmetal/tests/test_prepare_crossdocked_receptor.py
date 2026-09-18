"""Guard against reporting synthesized receptor chemistry as real preparation."""
import csv
import subprocess

import pytest

from molmetal.scripts import prepare_crossdocked_receptor as prep


@pytest.fixture
def atom_pair(tmp_path):
    original = tmp_path / "input.pdb"
    prepared = tmp_path / "output.pdbqt"
    # Small fixtures verify validation logic, never generate experimental data.
    original.write_text(
        "ATOM   2873  C   VAL A 384      47.233  41.755  77.624  1.00 21.21         A C\n"
    )
    prepared.write_text(
        "ATOM      1  C   VAL A 384      47.233  41.755  77.624  1.00  0.00    +0.243 C \n"
    )
    return original, prepared


def test_heavy_atom_audit_accepts_preserved_identity_and_coordinates(atom_pair):
    audit = prep.audit_receptor(*atom_pair)
    assert audit["input_heavy_atoms"] == audit["output_heavy_atoms"] == 1
    assert audit["max_heavy_atom_displacement_A"] == 0


@pytest.mark.parametrize("old,new,error", [
    ("VAL", "ALA", "identities"),
    ("+0.243", "+0.000", "zero"),
    ("47.233", "48.233", "moved"),
    (" C \n", " NA\n", "element"),
])
def test_corrupted_preparation_is_rejected(atom_pair, old, new, error):
    original, prepared = atom_pair
    prepared.write_text(prepared.read_text().replace(old, new))
    with pytest.raises(ValueError, match=error):
        prep.audit_receptor(original, prepared)


def test_manifest_resolves_exact_pair_and_rejects_duplicates(tmp_path):
    receptor, ligand = tmp_path / "original.pdb", tmp_path / "reference.sdf"
    receptor.touch()
    ligand.touch()
    manifest = tmp_path / "manifest.csv"
    row = dict(pocket_id="test_001", receptor_path=str(receptor), ligand_path=str(ligand))
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    assert prep.manifest_pair(manifest, "test_001") == (receptor, ligand)
    with manifest.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=row).writerow(row)
    with pytest.raises(ValueError, match="exactly one"):
        prep.manifest_pair(manifest, "test_001")


def test_failed_strict_preparation_does_not_retry_unsafe_flags(monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, "", "unmatched residue")

    monkeypatch.setattr(prep.shutil, "which", lambda _: "/env/mk_prepare_receptor.py")
    monkeypatch.setattr(prep.subprocess, "run", run)
    result = prep.prepare_receptor(tmp_path / "input.pdb", tmp_path / "prepared")
    assert not result["passed"]
    assert result["stderr"] == "unmatched residue"
    assert len(commands) == 1
    assert "--compute_charges" in commands[0]
    assert commands[0][-2:] == ["--charge_model", "gasteiger"]
    assert not {"--delete_bad_res", "--forgive_extra_bonds", "--allow_bad_res"}.intersection(commands[0])
