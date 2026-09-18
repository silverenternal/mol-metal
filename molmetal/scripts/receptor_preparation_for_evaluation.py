"""Traceable receptor preparation: strict Meeko, conformer choice, name repair.

Source files are never modified. Missing heavy atoms are never invented or
dropped. Alternative conformers are selected at residue level by mean
occupancy, ties preferring A then lexicographic order. Meeko's terminal
carboxylate O/OXT naming permutation may be restored only when coordinates,
elements, AutoDock types and charges prove the two oxygens equivalent.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from molmetal.scripts.prepare_crossdocked_receptor import _atoms, audit_receptor, prepare_receptor


def select_receptor_altloc(original: Path, derived: Path) -> dict:
    """Write one explicitly selected conformer and return atom-level provenance."""
    if original.resolve() == derived.resolve():
        raise ValueError("Derived PDB must not overwrite original")
    lines = original.read_text().splitlines()
    residues = defaultdict(list)
    for index, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            occupancy = float(line[54:60])
            if not math.isfinite(occupancy) or not 0 <= occupancy <= 1:
                raise ValueError(f"Invalid occupancy at line {index + 1}")
            residues[(line[21:22], line[22:27].strip())].append((index, line, occupancy))
    if not residues:
        raise ValueError("No atoms in input receptor")
    choices, discarded, selected_records, excluded_records = [], set(), [], []
    for residue, records in residues.items():
        if len({line[17:20] for _, line, _ in records}) != 1:
            raise ValueError(f"Alternative residue chemistry requires explicit review: {residue}")
        alternatives = defaultdict(list)
        for index, line, occupancy in records:
            if line[16].strip():
                alternatives[line[16]].append((index, line, occupancy))
        mean_occupancy = {alt: sum(row[2] for row in rows) / len(rows)
                          for alt, rows in alternatives.items()}
        selected = min(mean_occupancy, key=lambda alt: (-mean_occupancy[alt], alt != "A", alt)) if alternatives else None
        if alternatives:
            choices.append({
                "chain": residue[0], "residue_id": residue[1],
                "residue_name": records[0][1][17:20].strip(), "selected_altloc": selected,
                "mean_occupancy_by_altloc": mean_occupancy,
                "atom_count_by_altloc": {key: len(value) for key, value in alternatives.items()},
            })
        all_heavy = {line[12:16].strip() for _, line, _ in records if line[76:78].strip().upper() != "H"}
        selected_heavy, selected_names = set(), set()
        for index, line, occupancy in records:
            keep = not line[16].strip() or line[16] == selected
            atom_name = line[12:16].strip()
            entry = {"source_line_number": index + 1, "serial": int(line[6:11]),
                     "chain": residue[0], "residue_id": residue[1],
                     "residue_name": line[17:20].strip(), "atom_name": atom_name,
                     "altloc": line[16].strip(), "occupancy": occupancy,
                     "element": line[76:78].strip(),
                     "coordinates": [float(line[start:start + 8]) for start in (30, 38, 46)]}
            if keep:
                if atom_name in selected_names:
                    raise ValueError(f"Duplicate selected atom {residue}/{atom_name}")
                selected_names.add(atom_name)
                if entry["element"].upper() != "H":
                    selected_heavy.add(atom_name)
                selected_records.append(entry)
            else:
                discarded.add(index)
                excluded_records.append(entry)
        if selected_heavy != all_heavy:
            raise ValueError(f"Chosen altloc lacks heavy atoms at {residue}: {sorted(all_heavy - selected_heavy)}")
    excluded_serials = {entry["serial"] for entry in excluded_records}
    output = []
    for index, line in enumerate(lines):
        if index in discarded:
            continue
        if line.startswith("CONECT"):
            serials = {int(line[start:start + 5]) for start in range(6, len(line), 5) if line[start:start + 5].strip()}
            if serials & excluded_serials:
                raise ValueError("CONECT references excluded conformer; explicit connectivity review required")
        if line.startswith("ANISOU") and int(line[6:11]) in excluded_serials:
            continue
        if line.startswith(("ATOM  ", "HETATM", "ANISOU")):
            line = line[:16] + " " + line[17:]
        output.append(line)
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text("\n".join(output) + "\n")
    return {
        "source": str(original.resolve()), "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "derived_pdb": str(derived.resolve()), "derived_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
        "policy": "one residue-wide altloc; highest mean occupancy; ties A first then lexicographic; retain shared atoms",
        "protocol_change": "selected conformer labels normalized to blank; only excluded alternatives may be removed",
        "choices": choices, "selected_atom_records": selected_records,
        "excluded_alternative_atom_records": excluded_records,
        "selected_atom_count": len(selected_records), "excluded_alternative_atom_count": len(excluded_records),
    }


def restore_equivalent_terminal_oxygen_names(source: Path, prepared: Path, derived: Path) -> dict:
    """Restore only a verified O/OXT naming permutation, preserving all physics."""
    if prepared.resolve() == derived.resolve():
        raise ValueError("Name repair must write a new PDBQT")
    original_atoms, prepared_atoms = _atoms(source, False), _atoms(prepared, True)
    before = {key: atom for key, atom in original_atoms.items() if atom["element"] != "H"}
    after = {key: atom for key, atom in prepared_atoms.items() if atom["element"] != "H"}
    if before.keys() != after.keys():
        raise ValueError("Atom identity mismatch; cannot repair by naming")
    changed = defaultdict(set)
    for key in before:
        if before[key]["element"] != after[key]["element"]:
            raise ValueError("Element mismatch; cannot repair by naming")
        if math.dist(before[key]["coords"], after[key]["coords"]) > 0.002:
            changed[key[:3]].add(key[3])
    if not changed:
        raise ValueError("No terminal-oxygen naming permutation found")
    mapping, evidence = {}, []
    for residue, names in changed.items():
        if names != {"O", "OXT"}:
            raise ValueError(f"Non-equivalent coordinate changes at {residue}: {sorted(names)}")
        oxygen, terminal, carbon = residue + ("O",), residue + ("OXT",), residue + ("C",)
        if carbon not in before:
            raise ValueError("Missing carboxyl carbon")
        for key, other in ((oxygen, terminal), (terminal, oxygen)):
            if before[key]["element"] != "O" or after[key]["atom_type"] != "OA":
                raise ValueError("Not an oxygen-acceptor naming permutation")
            if math.dist(after[key]["coords"], before[other]["coords"]) > 0.002:
                raise ValueError("Coordinates are not a pure O/OXT swap")
            if not 0.9 < math.dist(before[key]["coords"], before[carbon]["coords"]) < 1.6:
                raise ValueError("Oxygen is not carboxyl-bonded")
            if abs(after[key]["charge"] - after[other]["charge"]) > 1e-9:
                raise ValueError("Charges differ; oxygens are not interchangeable")
            mapping[key] = other[3]
            evidence.append({"chain": residue[0], "residue_id": residue[1], "residue_name": residue[2],
                             "meeko_atom_name": key[3], "source_atom_name_at_same_coordinate": other[3],
                             "unchanged_coordinates": after[key]["coords"],
                             "unchanged_charge": after[key]["charge"], "unchanged_atom_type": after[key]["atom_type"]})
    output = []
    for line in prepared.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            key = (line[21:22], line[22:27].strip(), line[17:20].strip(), line[12:16].strip())
            if key in mapping:
                line = line[:12] + f" {mapping[key]:<3}" + line[16:]
        output.append(line)
    derived.write_text("\n".join(output) + "\n")
    audit = audit_receptor(source, derived)
    return {"input_pdbqt": str(prepared), "derived_pdbqt": str(derived), "mapping": evidence,
            "protocol_change": "restore source atom names only for equivalent carboxylate O/OXT; no coordinate/type/charge change",
            "audit": audit}


def prepare_receptor_for_evaluation(original: Path, output_dir: Path) -> dict:
    """Strict preparation, with recorded conformer/name normalization only.

    The returned effective_pdb must also be used for PoseBusters. Consumers
    must check passed before reading pdbqt; genuine missing atoms stay failed.
    """
    original, output_dir = Path(original).resolve(), Path(output_dir).resolve()
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "source": str(original), "source_sha256": source_hash,
              "effective_pdb": str(original), "pdbqt": None, "audit": None}
    try:
        attempt = prepare_receptor(original, output_dir / "strict")
        result["strict_attempt"] = attempt
        effective = original
        if not attempt["passed"] and any(line.startswith(("ATOM  ", "HETATM")) and line[16].strip()
                                         for line in original.read_text().splitlines()):
            effective = output_dir / "selected_conformer.pdb"
            selection = select_receptor_altloc(original, effective)
            selection_path = output_dir / "altloc_selection.json"
            selection_path.write_text(json.dumps(selection, indent=2) + "\n")
            result["altloc_selection"] = selection
            result["effective_pdb"] = str(effective)
            attempt = prepare_receptor(effective, output_dir / "selected")
            result["selected_attempt"] = attempt
        if attempt["passed"]:
            result.update(passed=True, pdbqt=attempt["pdbqt"], audit=attempt["audit"])
        elif attempt.get("returncode") == 0:
            repaired = output_dir / "receptor_source_names.pdbqt"
            repair = restore_equivalent_terminal_oxygen_names(effective, Path(attempt["pdbqt"]), repaired)
            result["equivalent_name_repair"] = repair
            result.update(passed=True, pdbqt=str(repaired), audit=repair["audit"])
        else:
            result["error"] = attempt.get("error", "Strict Meeko preparation failed")
        result["source_unchanged"] = hashlib.sha256(original.read_bytes()).hexdigest() == source_hash
        result["passed"] = bool(result["passed"] and result["source_unchanged"])
    except Exception as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    (output_dir / "preparation_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
