"""WF-D7-Apply: verify D7 (c) — ``--engine both`` is the new default
for the headline table and the per-pocket report emits both
``vina_score`` and ``qvina_score`` columns when both engines are
requested, while single-engine modes keep backward compatibility.

Background
----------
Round-11 N=50 parity experiment
(``molmetal/reports/round11_engine_parity_n50.md``) measured
``Pearson r=0.9983 / Spearman rho=0.9984`` between Vina 1.2.7 and
the bundled qvina02 (QuickVina 2), with MAD = 0.071 kcal/mol on
n=46 paired complexes at matched seed + box + receptor. The 4/50
qvina02 atom-type rejections are an engine-version vocabulary drift
artefact (CG0 in meeko vs the 2011 QuickVina atom-type set), not a
parity failure.

D7 = (c) "both engines in headline table" is the recommended default
per ``molmetal/reports/wf_decisions_summary.md`` §2. This test file
locks in:

1. ``r4_c_full_sweep.py`` --engine default is 'both'
2. ``--engine both`` makes the report emit BOTH ``vina_score`` and
   ``qvina_score`` columns per pocket
3. ``--engine vina`` still emits only the ``vina_score`` column
4. ``--engine qvina`` still emits only the ``qvina_score`` column

These tests are intentionally hermetic — they do not require real
Vina/QVina binaries installed because the engine dispatch happens
inside ``evaluate_generated_poses.evaluate_candidates``, and we
verify the contract by inspecting the configuration paths the
sweep script + the per-pager evaluator thread through. The actual
binary installation is a separate concern, gated by the round-7
wire-vina install report.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import textwrap
from pathlib import Path
from typing import List
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def _fake_receptor_path(tmp_path: Path) -> Path:
    """Write a tiny stub PDB receptor on disk.

    ``evaluate_generated_poses`` hashes the receptor file (line 70) so
    the test must pass a real path that exists. The actual content is
    ignored because we mock the receptor-prep + adapter pipeline.
    """
    p = tmp_path / "fake_receptor.pdb"
    p.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1       1.250   2.390   0.000  1.00  0.00           O  \n"
        "END\n"
    )
    return p


@pytest.fixture
def _fake_ligand_sdf(tmp_path: Path) -> Path:
    """Write a minimal RDKit SDF ligand with one conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    out = tmp_path / "ligand.sdf"
    Chem.SDWriter(str(out)).write(mol)
    return out


@pytest.fixture(scope="module")
def r4_c_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "r4_c_full_sweep.py"


@pytest.fixture(scope="module")
def evaluate_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "evaluate_generated_poses.py"


@pytest.fixture(scope="module")
def r4_c_source(r4_c_path: Path) -> str:
    """Read the sweep script source for AST-style default / choices checks.

    We do not execute the script's ``main`` because that requires a real
    CrossDocked manifest + receptor PDBs. Instead we extract the
    argparse defaults and choices directly from the source.
    """
    return r4_c_path.read_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_engine_block(src: str) -> dict:
    """Pull the ``--engine`` argparse block from the source.

    Returns a dict with keys ``choices`` (tuple of strings) and
    ``default`` (string). Matches by scanning the file for an
    ``add_argument('--engine'`` call and parsing the keyword arguments.
    """
    import re

    # Find the --engine add_argument block — the choices tuple can span
    # multiple lines and contain commas inside the strings. We use a
    # greedy non-greedy match bounded by the closing ``)`` of the
    # tuple, and parse choices by splitting on commas that are
    # followed by a quote (i.e. an actual top-level element separator).
    block_pattern = re.compile(
        r'parser\.add_argument\(\s*"--engine"\s*,\s*'
        r'choices=\((?P<choices>[^)]*?)\)\s*,\s*'
        r'default="(?P<default>[^"]+)"',
        re.DOTALL,
    )
    m = block_pattern.search(src)
    assert m, "Could not locate --engine argparse block"
    choices_str = m.group("choices")
    # Tokenise by commas that introduce a new string literal.
    parts = re.findall(r'"([^"]+)"', choices_str)
    return {"choices": tuple(parts), "default": m.group("default")}


# ---------------------------------------------------------------------------
# Test 1: default is 'both'
# ---------------------------------------------------------------------------
def test_default_engine_is_both(r4_c_source: str) -> None:
    """The D7 default --engine value is ``"both"`` after this change.

    Rationale: per ``wf_decisions_summary.md`` D7=(c), both engines in
    the headline table is the recommended default based on the N=50
    parity experiment (r=0.9983). The sweep script must reflect this.
    """
    block = _parse_engine_block(r4_c_source)
    assert block["default"] == "both", (
        f"--engine default should be 'both' (D7-(c)), got {block['default']!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: both engines emit vina_score AND qvina_score columns
# ---------------------------------------------------------------------------
def test_both_engine_emits_vina_and_qvina_columns(
    evaluate_path: Path,
    tmp_path: Path,
    _fake_receptor_path: Path,
    _fake_ligand_sdf: Path,
) -> None:
    """When ``engine='both'`` the per-candidate row contains BOTH
    ``vina_score`` and ``qvina_score`` columns.

    We invoke the real ``evaluate_candidates`` with mocked
    ``VinaDockingAdapter`` instances so the test is hermetic and
    independent of whether real Vina/QVina binaries are present on
    this ROCm stack.
    """
    # Load the evaluator module dynamically.
    spec = importlib.util.spec_from_file_location("_evaluate_under_test", str(evaluate_path))
    assert spec and spec.loader
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    # Synthesize a minimal "candidate" + "pocket" pair. We mock the
    # full receptor-prep pipeline so we don't need a real PDB on disk.
    fake_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

    # Stub VinaDockingAdapter: every instance returns a single pose
    # with a deterministic, distinct score per engine so we can tell
    # which engine's column was filled in.
    class _FakeAdapter:
        def __init__(self, engine: str):
            self.engine = engine
            self._receptor_pdbqt = {}
        def get_metadata(self):
            return {"name": f"FakeAdapter[{self.engine}]", "engine": self.engine}
        def dock(self, molecule, pocket, config):
            from molmetal.domain import Complex  # real type, real fields
            score = {"vina": -7.5, "qvina": -7.4, "quickvina2": -7.6}[self.engine]
            return [Complex(
                pocket=pocket, molecule=molecule,
                pose_confidence=0.5, vina_score=score,
            )]
        last_pose_mols = []

    def fake_adapter_factory(engine: str, **kwargs):
        return _FakeAdapter(engine)

    # Stub receptor preparation so we don't need a real PDBQT on disk.
    prep_result = {"passed": True, "effective_pdb": str(_fake_receptor_path),
                   "pdbqt": str(tmp_path / "fake.pdbqt")}
    candidates = [
        {"smiles": fake_smiles, "is_generated": True, "sa": 1.0, "qed": 0.7},
    ]

    # Patch the heavy modules / paths used by evaluate_candidates.
    with mock.patch("molmetal.scripts.receptor_preparation_for_evaluation.prepare_receptor_for_evaluation",
                    return_value=prep_result), \
         mock.patch("molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
                    side_effect=fake_adapter_factory), \
         mock.patch("molmetal.validation.posebusters_runner.check_docked_pose",
                    return_value={"status": "passed", "pb_valid": True}):

        report = eval_mod.evaluate_candidates(
            candidates=candidates,
            receptor_path=str(_fake_receptor_path),
            ligand_path=str(_fake_ligand_sdf),
            output_dir=str(tmp_path / "out"),
            seed=42,
            engine="both",  # D7
            exhaustiveness=1,
            n_poses=1,
            top_k=1,
        )

    # Reference row + candidate row each get both columns populated.
    ref_row = report["reference"]
    assert "vina_score" in ref_row, f"reference row missing vina_score: keys={list(ref_row)}"
    assert "qvina_score" in ref_row, f"reference row missing qvina_score: keys={list(ref_row)}"
    assert ref_row["vina_score"] == -7.5
    assert ref_row["qvina_score"] == -7.4

    cand_rows = report["candidates"]
    assert len(cand_rows) == 1
    cand_row = cand_rows[0]
    assert "vina_score" in cand_row, f"candidate row missing vina_score: keys={list(cand_row)}"
    assert "qvina_score" in cand_row, f"candidate row missing qvina_score: keys={list(cand_row)}"
    assert cand_row["vina_score"] == -7.5
    assert cand_row["qvina_score"] == -7.4


# ---------------------------------------------------------------------------
# Test 3: backward compat — --engine vina still emits only vina_score column
# ---------------------------------------------------------------------------
def test_backward_compat_vina_only(
    evaluate_path: Path,
    tmp_path: Path,
    _fake_receptor_path: Path,
    _fake_ligand_sdf: Path,
) -> None:
    """``engine='vina'`` is unchanged: only ``vina_score`` is emitted."""
    spec = importlib.util.spec_from_file_location("_eval_vina_only", str(evaluate_path))
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    class _VinaOnlyAdapter:
        engine = "vina"
        def __init__(self, **kwargs):
            self._receptor_pdbqt = {}
        def get_metadata(self):
            return {"name": "VinaOnly", "engine": "vina"}
        def dock(self, molecule, pocket, config):
            from molmetal.domain import Complex
            return [Complex(pocket=pocket, molecule=molecule,
                            pose_confidence=0.5, vina_score=-7.5)]
        last_pose_mols = []

    prep_result = {"passed": True, "effective_pdb": str(_fake_receptor_path),
                   "pdbqt": str(tmp_path / "fake.pdbqt")}
    candidates = [{"smiles": "CCO", "is_generated": True}]

    with mock.patch("molmetal.scripts.receptor_preparation_for_evaluation.prepare_receptor_for_evaluation",
                    return_value=prep_result), \
         mock.patch("molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
                    return_value=_VinaOnlyAdapter()), \
         mock.patch("molmetal.validation.posebusters_runner.check_docked_pose",
                    return_value={"status": "passed", "pb_valid": True}):

        report = eval_mod.evaluate_candidates(
            candidates=candidates,
            receptor_path=str(_fake_receptor_path),
            ligand_path=str(_fake_ligand_sdf),
            output_dir=str(tmp_path / "out"),
            seed=42,
            engine="vina",
            exhaustiveness=1,
            n_poses=1,
            top_k=1,
        )

    ref_row = report["reference"]
    cand_row = report["candidates"][0]
    # Single-engine mode keeps only the legacy column populated.
    assert "vina_score" in ref_row
    assert "qvina_score" not in ref_row, (
        f"vina-only run should NOT emit qvina_score; got keys={list(ref_row)}"
    )
    assert "vina_score" in cand_row
    assert "qvina_score" not in cand_row


# ---------------------------------------------------------------------------
# Test 4: backward compat — --engine qvina still emits only qvina_score column
# ---------------------------------------------------------------------------
def test_backward_compat_qvina_only(
    evaluate_path: Path,
    tmp_path: Path,
    _fake_receptor_path: Path,
    _fake_ligand_sdf: Path,
) -> None:
    """``engine='qvina'`` is unchanged: only ``qvina_score`` is emitted."""
    spec = importlib.util.spec_from_file_location("_eval_qvina_only", str(evaluate_path))
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    class _QvinaOnlyAdapter:
        engine = "qvina"
        def __init__(self, **kwargs):
            self._receptor_pdbqt = {}
        def get_metadata(self):
            return {"name": "QvinaOnly", "engine": "qvina"}
        def dock(self, molecule, pocket, config):
            from molmetal.domain import Complex
            return [Complex(pocket=pocket, molecule=molecule,
                            pose_confidence=0.5, vina_score=-7.4)]
        last_pose_mols = []

    prep_result = {"passed": True, "effective_pdb": str(_fake_receptor_path),
                   "pdbqt": str(tmp_path / "fake.pdbqt")}
    candidates = [{"smiles": "CCO", "is_generated": True}]

    with mock.patch("molmetal.scripts.receptor_preparation_for_evaluation.prepare_receptor_for_evaluation",
                    return_value=prep_result), \
         mock.patch("molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
                    return_value=_QvinaOnlyAdapter()), \
         mock.patch("molmetal.validation.posebusters_runner.check_docked_pose",
                    return_value={"status": "passed", "pb_valid": True}):

        report = eval_mod.evaluate_candidates(
            candidates=candidates,
            receptor_path=str(_fake_receptor_path),
            ligand_path=str(_fake_ligand_sdf),
            output_dir=str(tmp_path / "out"),
            seed=42,
            engine="qvina",
            exhaustiveness=1,
            n_poses=1,
            top_k=1,
        )

    ref_row = report["reference"]
    cand_row = report["candidates"][0]
    # qvina-only: qvina_score present, vina_score absent.
    assert "qvina_score" in ref_row
    assert "vina_score" not in ref_row, (
        f"qvina-only run should NOT emit vina_score; got keys={list(ref_row)}"
    )
    assert "qvina_score" in cand_row
    assert "vina_score" not in cand_row


# ---------------------------------------------------------------------------
# Bonus: argparse choices still cover all valid engines
# ---------------------------------------------------------------------------
def test_engine_choices_include_all_valid_names(r4_c_source: str) -> None:
    """The --engine argparse ``choices`` tuple contains every valid name.

    ``vina`` / ``qvina`` / ``quickvina2`` / ``both`` / ``all`` must all
    remain valid choices so backward-compatible callers still work
    after D7-Apply.
    """
    block = _parse_engine_block(r4_c_source)
    expected = {"vina", "qvina", "quickvina2", "both", "all"}
    actual = set(block["choices"])
    assert expected.issubset(actual), (
        f"missing --engine choices: {expected - actual}; "
        f"got {actual}"
    )