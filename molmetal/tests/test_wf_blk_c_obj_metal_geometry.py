"""Tests for the BLK-C-OBJ precious-metal geometry instrument.

Two jobs:

1. PIN the parts that must not drift. The embedder definition and the geometry
   criterion are both places where a silent edit would change every number the
   instrument reports without any test failing.

2. Make each NEW behaviour fail loudly when it is removed. The red checks are
   reported in the wave notes; each one was confirmed red by reverting the
   change and re-running.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

import wf_blk_c_obj_metal_geometry as mg  # noqa: E402

METRIC = REPO / "metrics/by_metric/cfm_metal_geometry.json"


# --------------------------------------------------------------------------
# 1. The embedder definition is pinned, not re-derived.
# --------------------------------------------------------------------------
def test_embedder_reproduces_the_recorded_54_of_66():
    """The recorded protocol gives exactly 54/66 on the GENERATED cohort.

    This number is already pinned by
    ``test_wf_blkc_l5_metal_deletion_control.py::test_embeddable_reproduces_the_recorded_generated_number``.
    It is re-asserted here because this script depends on it: if the embedder
    drifts, every bond length, every angle and every geometry class in this
    instrument moves, and nothing else here would notice.
    """
    Chem, _ = mg._rdkit()
    import csv

    smis = []
    with open(mg.GENERATED_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("smiles"):
                smis.append(row["smiles"])

    ok = sum(1 for s in smis if mg.embeddable(Chem.MolFromSmiles(s)))
    assert (len(smis), ok) == (66, 54), (
        f"measured {ok}/{len(smis)} embeddable; the pinned protocol gives 54/66. "
        "The embedder definition has drifted."
    )


def test_embedder_requires_addhs():
    """RED CHECK: removing ``Chem.AddHs`` makes ETKDG fail on organometallics.

    AddHs is load-bearing. Without it this returns 0/66 -- a confident, entirely
    artefactual "none of our generated complexes can be embedded" verdict, which
    is the error class this line keeps committing.
    """
    Chem, AllChem = mg._rdkit()
    import csv

    smis = [
        r["smiles"]
        for r in csv.DictReader(open(mg.GENERATED_CSV))
        if r.get("smiles")
    ]

    def embed_without_hs(mol):
        if mol is None:
            return None
        if AllChem.EmbedMolecule(
            Chem.Mol(mol), useRandomCoords=True, maxAttempts=1, randomSeed=0xF00D
        ) != 0:
            return False
        return mol.GetNumConformers() > 0

    without = sum(1 for s in smis if embed_without_hs(Chem.MolFromSmiles(s)))
    with_hs = sum(1 for s in smis if mg.embeddable(Chem.MolFromSmiles(s)))

    assert with_hs == 54
    assert without < with_hs, (
        "the no-AddHs variant matched the AddHs variant; the premise of this "
        "test is that omitting explicit hydrogens is strictly worse"
    )


# --------------------------------------------------------------------------
# 2. The geometry criterion is pinned against exact ideal constructions.
# --------------------------------------------------------------------------
def test_planarity_ratio_on_ideal_geometries():
    """Square planar -> 0, regular tetrahedron -> 1, both exactly.

    Anchored here because these two numbers are what the ``--planarity-cut``
    threshold of 0.15 sits between. If the criterion is edited, this fails
    rather than the threshold quietly re-meaning itself.
    """
    square = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    tetra = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

    assert mg._planarity_ratio(square) == pytest.approx(0.0, abs=1e-9)
    assert mg._planarity_ratio(tetra) == pytest.approx(1.0, abs=1e-6)


def test_planarity_ratio_cannot_separate_octahedral_from_trigonal_prism():
    """RED CHECK for an over-claim the first draft of the docstring made.

    An ideal octahedron and an ideal trigonal prism BOTH give 1.0, because both
    are fully 3D with isotropic donor covariance. Any criterion based on this
    ratio alone cannot tell them apart, so the instrument reports CN=6 as
    ``octahedral_like`` (a trans-pair test), never as ``octahedral``.
    """
    oct6 = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    trig_prism = [
        [1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0],
        [1, 0, 1.732], [-0.5, 0.866, 1.732], [-0.5, -0.866, 1.732],
    ]
    assert mg._planarity_ratio(oct6) == pytest.approx(1.0, abs=1e-6)
    assert mg._planarity_ratio(trig_prism) == pytest.approx(1.0, abs=1e-3)


def test_geometry_class_never_claims_plain_octahedral():
    """No CN=6 record may be labelled ``octahedral`` -- only ``octahedral_like``."""
    for cn, ratio, trans in [(6, 1.0, 3), (6, 1.0, 0), (6, 0.2, 3)]:
        label = mg._geometry_class(cn, ratio, 0.15, trans)
        assert label != "octahedral"
        assert "octahedral" not in label or label == "octahedral_like"


def test_geometry_class_at_cn4_separates_the_two_ideals():
    assert mg._geometry_class(4, 0.0, 0.15, 2) == "square_planar"
    assert mg._geometry_class(4, 1.0, 0.15, 0) == "tetrahedral"


# --------------------------------------------------------------------------
# 3. The clash control.
# --------------------------------------------------------------------------
def test_plausibility_filter_rejects_sub_1p5_angstrom_contacts():
    """RED CHECK: the filter must reject, and must reject a 0.28 A contact.

    The first run of this instrument reported a 0.28 A metal-ligand bond in
    REAL_ETKDG and put it in the reference band. ETKDG with maxAttempts=1
    returns success on a structure with two atoms on top of each other, so
    "it embedded" is not "it is a molecule".
    """
    assert mg._plausible([2.2, 2.1, 2.3]) is True
    assert mg._plausible([2.2, 0.28]) is False
    assert mg._plausible([1.49]) is False
    assert mg._plausible([]) is False
    assert mg._plausible(None) is False


def test_plausibility_threshold_is_the_documented_constant():
    assert mg.MIN_PLAUSIBLE_ML_BOND_A == 1.5


# --------------------------------------------------------------------------
# 4. The cohort population is intersected with the coordinate file.
# --------------------------------------------------------------------------
def test_dft_population_is_intersected_before_sampling():
    """RED CHECK: the sample is drawn from molecules that HAVE coordinates.

    ``tmQM_X1.xyz.gz`` holds 34,970 frames while ``tmqm_parsed.csv`` holds
    108,541 rows: the coordinate file is a strict SUBSET. Sampling the csv and
    only then looking up coordinates silently shrinks the cohort -- the first
    version of this script requested 800 molecules and received 165, a 4x
    shortfall in the reference band that no exit code would have reported.

    This asserts on ``select_dft_population`` rather than on the parsing
    helpers, because it is the CALL SITE that has to do the intersecting. A
    test of the helper alone passed even when the call site was reverted.
    """
    population = mg.select_dft_population(limit=300, seed=20261012)
    codes = [r[0] for r in population]

    assert len(codes) == 300, f"asked for 300, selected {len(codes)}"
    assert len(set(codes)) == 300, "sample contains duplicates"

    available = mg._read_xyz_csd_codes(mg.TMQM_XYZ)
    missing = [c for c in codes if c not in available]
    assert not missing, (
        f"{len(missing)}/{len(codes)} selected molecules have no coordinates; "
        "the population was not intersected with the coordinate file"
    )


def test_read_xyz_csd_codes_is_a_subset_of_the_csv():
    """The premise behind the guard above: the two files genuinely differ."""
    from wf_blk_c_obj_metal_geometry import _read_metal_csd_codes

    xyz_codes = mg._read_xyz_csd_codes(mg.TMQM_XYZ)
    all_rows = _read_metal_csd_codes(mg.TMQM_CSV, mg.TRAIN_METALS, 0, 0)
    restricted = _read_metal_csd_codes(
        mg.TMQM_CSV, mg.TRAIN_METALS, 0, 0, restrict_to=xyz_codes
    )

    assert len(all_rows) > len(restricted), (
        "the coordinate file covers the entire csv; the intersect-and-sample "
        "guard is then unnecessary and this test's premise is stale"
    )
    assert {r[0] for r in restricted} <= xyz_codes


# --------------------------------------------------------------------------
# 5. The emitted metric.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not METRIC.exists(), reason="metric not generated yet")
def test_generated_cohort_contains_no_platinum():
    """MEASURED on disk: the GENERATED cohort has ZERO Pt complexes.

    Ru 16, Ir 18, Rh 15, Re 13, Os 4, Pt 0 (counted by metal bracket atom in
    generated_smiles.csv). Pt is the single most common metal in real tmQM
    (8,275 of 33,429 six-metal rows) and it is the cisplatin class that
    motivates this entire geometry axis. So the square-planar pharmacophore
    cannot be assessed on our output at all -- there is nothing to assess.

    This test pins that fact rather than the code, so that if a later retrain
    does produce Pt, the test fails and the finding is re-examined rather than
    silently becoming stale.
    """
    data = json.loads(METRIC.read_text())
    gen = data["cohorts"]["GENERATED"]
    assert gen["geometry_class_per_metal"].get("Pt", {}) == {}, (
        "the GENERATED cohort now contains Pt complexes; the 'cannot assess "
        "square-planar, Pt is absent' finding no longer holds and must be "
        "re-measured before it is cited again"
    )


@pytest.mark.skipif(not METRIC.exists(), reason="metric not generated yet")
def test_the_embedder_alone_erases_the_square_planar_pharmacophore():
    """RED CHECK for the single most consequential caveat in this instrument.

    REAL_DFT finds square-planar Pt(II) in its CN=4 population. REAL_ETKDG --
    the SAME real molecules, pushed through the embedder the GENERATED cohort
    must use -- finds essentially none. ETKDG with maxAttempts=1 is destroying
    the very geometry class this axis exists to measure.

    Consequence: any square-planar measurement taken on generated SMILES via
    ETKDG is a statement about ETKDG, not about the generator. This is why the
    primary comparison band is REAL_ETKDG and why the metric ships the fidelity
    floor rather than a bare generated-vs-real number.
    """
    data = json.loads(METRIC.read_text())
    dft = data["cohorts"]["REAL_DFT"]["cn4_geometry_class_counts"]
    etkdg = data["cohorts"]["REAL_ETKDG"]["cn4_geometry_class_counts"]

    assert dft.get("square_planar", 0) > 0, (
        "the DFT band no longer shows square-planar CN=4; the premise that "
        "ETKDG erases the pharmacophore needs re-checking"
    )
    assert etkdg.get("square_planar", 0) < dft.get("square_planar", 0), (
        "the embedder now preserves square-planar CN=4 as well as DFT does; "
        "the instrument's central caveat is no longer true and must be revisited"
    )


@pytest.mark.skipif(not METRIC.exists(), reason="metric not generated yet")
def test_metric_has_all_three_cohorts_and_an_embedder_floor():
    """The comparison must never be GENERATED-vs-REAL_DFT without a floor.

    The generated cohort has SMILES only. Comparing ETKDG geometry against DFT
    geometry attributes part of the difference to the embedder instead of to
    the generator, which is precisely the mistake the fidelity control exists
    to make visible.
    """
    data = json.loads(METRIC.read_text())
    assert set(data["cohorts"]) == {"GENERATED", "REAL_ETKDG", "REAL_DFT"}

    dft = data["cohorts"]["REAL_DFT"]
    assert dft["n_molecules"] >= 100, "real reference band too small to be a band"
    assert dft["geometry_implausible_rejected"] >= 0

    floor = data["embedder_fidelity_control"]
    assert "real_dft_minus_real_etkdg_embedder_floor" in floor["cn4_planarity_ratio"]


@pytest.mark.skipif(not METRIC.exists(), reason="metric not generated yet")
def test_metric_records_the_no_double_count_and_the_sample_size():
    """The GENERATED cohort is 66 molecules; the report must not imply more."""
    data = json.loads(METRIC.read_text())
    gen = data["cohorts"]["GENERATED"]
    assert gen["n_molecules"] == 66
    assert (
        gen["geometry_unmeasured"] + gen["n_geometries_scored"]
        + gen["geometry_implausible_rejected"]
    ) == 66, "every generated molecule must land in exactly one bucket"


@pytest.mark.skipif(not METRIC.exists(), reason="metric not generated yet")
def test_dft_cohort_was_sampled_from_the_coordinate_file():
    data = json.loads(METRIC.read_text())
    dft = data["cohorts"]["REAL_DFT"]
    assert dft["n_molecules"] == dft.get("n_requested"), (
        "the DFT cohort came back smaller than requested, which means the "
        "sample was drawn without intersecting the coordinate file first"
    )


# --------------------------------------------------------------------------
# 6. End-to-end, on a tiny sample so it stays fast.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not METRIC.exists(), reason="metric not generated yet")
def test_script_runs_end_to_end_on_a_small_sample(tmp_path):
    """Guards the rc, not the numbers.

    A scaffold can import and print --help and still crash at compute time; per
    the project's own lesson the assertion here is that the run produced a
    complete three-cohort payload, not that it exited 0.
    """
    out = tmp_path / "geom.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "molmetal/scripts/wf_blk_c_obj_metal_geometry.py"),
            "--real-limit", "60",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    data = json.loads(out.read_text())
    assert set(data["cohorts"]) == {"GENERATED", "REAL_ETKDG", "REAL_DFT"}
    assert data["cohorts"]["GENERATED"]["n_molecules"] == 66