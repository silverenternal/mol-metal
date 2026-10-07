# Copyright 2026 The Mol-Metal Authors.
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
"""Wf-R43 m8 — the PB dock path must score a real pose, not a re-embedded one.

metrics/by_metric/pb_pass_rate.json records the contradiction that motivated
this module: the PB *adapter* path reported 0 passes while the Vina dispatcher
reported 20 for the SAME docked pose.  The cause was that
``PoseBustersAdapter.validate_docked(smiles, receptor)`` re-embeds the
molecule from SMILES, producing a free conformer that was never placed in the
binding site, so every protein-aware check failed by construction.

These tests pin the two properties that fix it:

1. ``validate_docked`` tags its own result ``coordinates_preserved=False`` so
   a re-embedded chemistry run can never be read as a docked-pose verdict.
2. ``validate_docked_pose`` scores the real pose and tags it ``True``.

They also guard the mmp13_real fixture, whose previous body was an HTML 404
page that RDKit parsed as ``None`` without raising.
"""

from pathlib import Path

import pytest
from rdkit import Chem

from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEPTOR = REPO_ROOT / "molmetal" / "data" / "mmp13_real" / "830c.pdb"
LIGAND = REPO_ROOT / "molmetal" / "data" / "mmp13_real" / "830c_ligand.pdb"

posebusters = pytest.importorskip("posebusters")


@pytest.fixture(scope="module")
def adapter():
    return PoseBustersAdapter(mode="dock")


@pytest.fixture(scope="module")
def docked_pose():
    mol = Chem.MolFromPDBFile(str(LIGAND), removeHs=False, sanitize=False)
    assert mol is not None, "830c_ligand.pdb failed to parse"
    Chem.SanitizeMol(mol)
    return mol


def test_mmp13_ligand_fixture_is_a_real_pdb(docked_pose):
    """The fixture is coordinates, not an HTML error page."""
    assert LIGAND.stat().st_size > 2000
    assert "404 Not Found" not in LIGAND.read_text()
    assert docked_pose.GetNumAtoms() == 48
    assert docked_pose.GetNumConformers() == 1


def test_smiles_path_declares_it_re_embedded(adapter, docked_pose):
    """validate_docked must not look like a docked-pose measurement."""
    smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(docked_pose)))
    result = adapter.validate_docked(smiles, str(RECEPTOR))
    assert result.coordinates_preserved is False
    assert result.to_dict()["coordinates_preserved"] is False


def test_docked_pose_path_preserves_coordinates(adapter, docked_pose):
    """The real pose is scored and the provenance is recorded."""
    result = adapter.validate_docked_pose(docked_pose, str(RECEPTOR))
    assert result.coordinates_preserved is True
    assert result.mode == "dock"
    assert result.n_checks > 0
    # A pose that genuinely docked cannot be scored 0/0.
    assert result.n_checks >= 20


def test_re_embedded_fails_the_check_a_real_pose_passes(adapter, docked_pose):
    """The discriminating property, on the real 830C fixture.

    A conformer re-embedded from SMILES sits at the origin, nowhere near
    the 830C pocket, so ``protein-ligand_maximum_distance`` MUST fail.
    The genuine docked pose sits inside the pocket and MUST NOT fail it.
    That single check is the m8 contradiction in one assertion: 0 pass vs
    20 pass for the same pose.

    We deliberately do not assert an overall pass.  Scoring the raw
    complex against the holo receptor also trips the cofactor checks,
    because 830c.pdb still contains its own RS1 as an organic cofactor,
    and PDBBlock bond perception leaves
    ``non-aromatic_ring_non-flatness`` failing.  Both are fixture
    artifacts, not defects in the dock path.
    """
    smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(docked_pose)))
    reembedded = adapter.validate_docked(smiles, str(RECEPTOR))
    posed = adapter.validate_docked_pose(docked_pose, str(RECEPTOR))

    assert "protein-ligand_maximum_distance" in reembedded.failed_checks
    assert "protein-ligand_maximum_distance" not in posed.failed_checks


def test_ligand_free_receptor_passes_the_docked_pose(adapter, docked_pose, tmp_path):
    """Strip the co-ligand from the receptor and the pose really does pass.

    This is the control for the fixture artifacts above: with RS1 removed
    from the receptor there is no organic-cofactor self-clash, and only
    the PDBBlock bond-perception ring check remains.
    """
    lines = (REPO_ROOT / "molmetal" / "data" / "mmp13_real" / "830c.pdb").read_text(
        encoding="utf-8"
    ).splitlines()
    stripped = [
        ln
        for ln in lines
        if not (ln.startswith("HETATM") and ln[17:20].strip() == "RS1")
    ]
    assert stripped != lines, "expected RS1 HETATM records in the holo receptor"
    apo = tmp_path / "830c_no_ligand.pdb"
    apo.write_text("\n".join(stripped) + "\nEND\n", encoding="utf-8")

    result = adapter.validate_docked_pose(docked_pose, str(apo))
    assert result.coordinates_preserved is True
    assert "minimum_distance_to_organic_cofactors" not in result.failed_checks
    assert "volume_overlap_with_organic_cofactors" not in result.failed_checks
    assert result.n_checks >= 20


def test_invalid_pose_does_not_score_as_pass(adapter):
    """An empty molecule must fail loudly, never score 0-checks-as-pass."""
    result = adapter.validate_docked_pose(None, str(RECEPTOR))
    assert result.passed is False
    assert result.n_checks == 0
