# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for chembl29_metal_scan.py — the B1 blocker's executable evidence.

WHY THIS EXISTS (2026-10-21)
----------------------------
Blocker B1 ("external independent cohort") rested on "the local ChEMBL 29 dump
has zero metal structures", written into ``todo.json`` as prose from an ad-hoc
query with no script and no artifact. The script makes that claim repeatable;
these tests pin the two ways it could go wrong:

  * a sodium salt counting as "the dump has metals" — MEASURED: exactly two
    ``[Na]`` hits, both sodium salts of an azo dye with an impossible Na-C bond;
  * a target-metal hit not flipping the verdict.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import chembl29_metal_scan as scan  # noqa: E402


def _mk_db(path: Path, smiles: list[str]) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE compound_structures (molregno INTEGER, molfile TEXT, "
        "standard_inchi TEXT, standard_inchi_key TEXT, canonical_smiles TEXT)"
    )
    for i, smi in enumerate(smiles):
        con.execute(
            "INSERT INTO compound_structures VALUES (?, '', '', '', ?)", (i, smi)
        )
    con.commit()
    con.close()
    return path


def test_a_dump_without_metals_is_reported_as_such(tmp_path):
    db = _mk_db(tmp_path / "chembl.db", ["CCO", "c1ccccc1", "CC(=O)O"])
    report = scan.scan(db)
    assert report["total_structures"] == 3
    assert report["verdict"] == "NO_TARGET_METAL_STRUCTURES"
    assert report["n_target_metals_with_any_hit"] == 0


def test_a_sodium_salt_does_not_count_as_a_target_metal(tmp_path):
    """The two real [Na] hits are dye salts, not a metal-chemistry cohort."""
    db = _mk_db(tmp_path / "chembl.db", ["CCO", "Nc1ccccc1OCCC[Na]"])
    report = scan.scan(db)
    assert report["by_metal"]["Na"]["n"] == 1
    assert report["n_metals_with_any_hit"] == 1
    assert report["n_target_metals_with_any_hit"] == 0
    assert report["verdict"] == "NO_TARGET_METAL_STRUCTURES"


def test_one_target_metal_flips_the_verdict(tmp_path):
    db = _mk_db(tmp_path / "chembl.db", ["CCO", "c1ccccc1[Pt](Cl)Cl"])
    report = scan.scan(db)
    assert report["by_metal"]["Pt"]["n"] == 1
    assert report["verdict"] == "TARGET_METAL_STRUCTURES_PRESENT"


def test_the_query_text_is_recorded_with_every_count(tmp_path):
    """A count nobody can re-run is the defect this script exists to fix."""
    db = _mk_db(tmp_path / "chembl.db", ["CCO"])
    report = scan.scan(db)
    for metal, block in report["by_metal"].items():
        assert block["query"].startswith("SELECT COUNT(*)")
        assert f"[{metal}]" in block["query"]
    assert report["legacy_substring_probe"]["query"].count("%Pt%") == 1


def test_the_artifact_is_written_and_parses(tmp_path):
    db = _mk_db(tmp_path / "chembl.db", ["CCO"])
    out = tmp_path / "out.json"
    assert scan.main(["--db", str(db), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["metric"] == "chembl29_metal_scan"
    assert payload["status"] == "MEASURED"
    assert payload["db"]["size_bytes"] > 0
