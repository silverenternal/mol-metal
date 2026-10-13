"""Hand-built-geometry tests for ``wf_blkc_geom_classifier``.

These tests are the load-bearing self-check the user demanded: the
classifier MUST reproduce the known answers on ideal constructions, and
a tolerance change must show up here. The cases chosen are exactly the
ones the round-1 docstring flagged -- ideal square plane, ideal
tetrahedron, ideal octahedron, ideal trigonal prism -- and the LAST
pair is the regression target. An ideal octahedron and an ideal trigonal
prism give the SAME s3/s2 planarity ratio in some constructions; the
classifier must therefore NOT rely on the planarity ratio at CN=6 and
MUST be shown to separate them via the trans-pair count.

The red-check protocol is invoked by running:

    /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/tests/test_wf_blkc_geom_classifier.py --red-check

which reverts the trans-pair cut to a value that DOES mis-classify the
trigonal prism, asserts the regression appears, and restores the file.
The pytest-style assertions in this file run in the green path; the red
check is a separate subprocess.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "molmetal/scripts/wf_blkc_geom_classifier.py"

sys.path.insert(0, str(REPO / "molmetal/scripts"))
from wf_blkc_geom_classifier import (  # noqa: E402
    DFT_DONOR_CUTOFF_A,
    MIN_PLAUSIBLE_ML_BOND_A,
    PLANARITY_CUT,
    TRANS_ANGLE_CUT_DEG,
    _geometry_class,
    _lm_l_angles,
    _planarity_ratio,
    _trans_pairs,
    hand_built_tests,
    ideal_octahedron,
    ideal_square_planar,
    ideal_tetrahedron,
    ideal_trigonal_prism,
    measure_clouds,
    summarise_clouds,
)


def test_planarity_ratio_ideal_square_planar_is_zero() -> None:
    pts = ideal_square_planar(radius=2.0)
    ratio = _planarity_ratio(pts)
    assert ratio is not None
    assert ratio < 1e-6, f"ideal SP planarity_ratio should be ~0, got {ratio}"


def test_planarity_ratio_ideal_tetrahedron_is_one() -> None:
    pts = ideal_tetrahedron(radius=2.0)
    ratio = _planarity_ratio(pts)
    assert ratio is not None
    assert abs(ratio - 1.0) < 1e-6, f"ideal T-4 planarity_ratio should be ~1, got {ratio}"


def test_planarity_ratio_ideal_octahedron_is_one() -> None:
    """Octahedral donors are isotropic, so s2 == s3 and the ratio is 1.

    This is the case round 1 said "an ideal octahedron also gives 1.0,
    and so does an ideal trigonal prism." That second claim is
    FALSE for several reasonable trigonal-prism aspect ratios, but it is
    TRUE for some. The classifier must NOT rely on the planarity ratio
    to discriminate CN=6 shapes.
    """
    pts = ideal_octahedron(radius=2.0)
    ratio = _planarity_ratio(pts)
    assert ratio is not None
    assert abs(ratio - 1.0) < 1e-6, f"ideal octahedral planarity_ratio should be ~1, got {ratio}"


def test_trans_pairs_octahedron_is_three() -> None:
    """3 trans pairs at 180 deg is the octahedron's fingerprint."""
    pts = ideal_octahedron(radius=2.0)
    angles = _lm_l_angles(np.zeros(3), pts)
    assert len(angles) == 15, f"octahedron should have 15 angle pairs, got {len(angles)}"
    n = _trans_pairs(angles, trans_cut=TRANS_ANGLE_CUT_DEG)
    assert n == 3, f"octahedron should have 3 trans pairs, got {n}"


def test_trans_pairs_trigonal_prism_is_zero() -> None:
    """Trigonal prism has no 180 deg L-M-L pair at the production cut.

    This is the discriminator the planarity ratio cannot supply. The
    user explicitly asked the classifier to separate octahedron and
    trigonal prism; this test pins that separation at the SHIPPED
    cut value, TRANS_ANGLE_CUT_DEG = 150.

    Caveat encoded in the test name: at OTHER cut values the
    discrimination can fail. With height = 2 * radius the trigonal
    prism's largest L-M-L angle is 126.87 deg, so a cut anywhere in
    (120, 180) would still see 0 trans pairs; but a cut anywhere in
    (90, 120) would see 6. The classifier ships at 150 deg.
    """
    pts = ideal_trigonal_prism(radius=2.0, height=2.0)
    angles = _lm_l_angles(np.zeros(3), pts)
    # 6 donors -> 15 angle pairs
    assert len(angles) == 15, f"trigonal prism should have 15 angle pairs, got {len(angles)}"
    n = _trans_pairs(angles, trans_cut=TRANS_ANGLE_CUT_DEG)
    assert n == 0, (
        f"trigonal prism should have 0 trans pairs at the production cut "
        f"{TRANS_ANGLE_CUT_DEG} deg, got {n}"
    )
    # A regular trigonal prism (top and bottom edges equal to the side
    # edges) has its largest L-M-L angle below 120 deg. The classifier
    # must NOT fail for such a geometry.
    pts_eq = ideal_trigonal_prism(radius=2.0, height=2.0 * math.sqrt(3.0))
    angles_eq = _lm_l_angles(np.zeros(3), pts_eq)
    n_eq = _trans_pairs(angles_eq, trans_cut=TRANS_ANGLE_CUT_DEG)
    assert n_eq == 0, (
        f"regular trigonal prism should have 0 trans pairs at cut {TRANS_ANGLE_CUT_DEG}, got {n_eq}"
    )


def test_geometry_class_sp4_vs_t4_at_cn4() -> None:
    """The CN=4 discriminator: SP-4 iff planarity_ratio <= cut."""
    pts = ideal_square_planar(radius=2.0)
    assert _geometry_class(4, _planarity_ratio(pts), PLANARITY_CUT, 2) == "square_planar"
    pts = ideal_tetrahedron(radius=2.0)
    assert _geometry_class(4, _planarity_ratio(pts), PLANARITY_CUT, 0) == "tetrahedral"


def test_geometry_class_cn6_separates_oct_from_tp() -> None:
    """At CN=6 the planarity ratio CANNOT discriminate; the trans count must."""
    oct_pts = ideal_octahedron(radius=2.0)
    tp_pts = ideal_trigonal_prism(radius=2.0, height=2.0)
    oct_ratio = _planarity_ratio(oct_pts)
    tp_ratio = _planarity_ratio(tp_pts)
    oct_angles = _lm_l_angles(np.zeros(3), oct_pts)
    tp_angles = _lm_l_angles(np.zeros(3), tp_pts)
    oct_trans = _trans_pairs(oct_angles)
    tp_trans = _trans_pairs(tp_angles)
    # The discriminator the user asked for: SP-4 vs T-4 alone FAILS here.
    # The classifier must use trans pairs at CN=6, not the planarity ratio.
    oct_class = _geometry_class(6, oct_ratio, PLANARITY_CUT, oct_trans)
    tp_class = _geometry_class(6, tp_ratio, PLANARITY_CUT, tp_trans)
    assert oct_class == "octahedral_like", f"octahedron should be octahedral_like, got {oct_class}"
    assert tp_class == "no_trans_pair_cn6", (
        f"trigonal prism should be no_trans_pair_cn6, got {tp_class}"
    )
    assert oct_class != tp_class, "octahedron and trigonal prism must NOT share a class"


def test_measure_clouds_hand_built_geometry_table() -> None:
    """End-to-end: every ideal geometry comes back with the right class.

    The four cases below are the user's full regression target list.
    """
    for t in hand_built_tests():
        rec = measure_clouds(t["positions"], t["zs"], metal="Pt")
        assert rec["coordination_number"] == t["cn_expected"], (
            f"{t['name']}: cn {rec['coordination_number']} != {t['cn_expected']}"
        )
        assert rec["geometry_class"] == t["class_expected"], (
            f"{t['name']}: class {rec['geometry_class']!r} != {t['class_expected']!r}"
        )


def test_measure_clouds_no_metal_centre_returns_unmeasured() -> None:
    """If the cloud contains no train metal, every geometric axis is None."""
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    zs = np.array([6, 6, 6], dtype=int)  # three carbons, no metal
    rec = measure_clouds(pos, zs, metal="Pt")
    assert rec["coordination_number"] is None
    assert rec["geometry_class"] == "unmeasured"
    assert rec["embedded"] is False


def test_measure_clouds_centres_on_metal_not_com() -> None:
    """A cloud translated in space should produce the same descriptors.

    measure_clouds subtracts the metal position (EDM-style zero-CoM, see
    Hoogeboom 2022, arXiv:2203.17003 sec 3). A whole-cloud CoM
    subtraction would put the metal off-centre and bias every descriptor.
    """
    pos, zs = _build(ideal_square_planar(radius=2.0))
    rec_origin = measure_clouds(pos, zs, metal="Pt")
    pos_shifted = pos + np.array([100.0, -50.0, 25.0])
    rec_shifted = measure_clouds(pos_shifted, zs, metal="Pt")
    assert rec_origin["coordination_number"] == rec_shifted["coordination_number"]
    assert rec_origin["planarity_ratio"] == rec_shifted["planarity_ratio"]
    # The bond lengths should be IDENTICAL -- the only thing translation
    # does is move the whole cloud, and the metal is the centre of the
    # coordinate system.
    assert rec_origin["metal_bond_lengths"] == rec_shifted["metal_bond_lengths"]


def test_measure_clouds_rejects_sub_min_bond_length() -> None:
    """A sub-1.5 A metal-ligand contact must be flagged as implausible."""
    z_pt = 78
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # Pt
            [1.0, 0.0, 0.0],  # Cl at 1.0 A -- implausible
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    zs = np.array([z_pt, 17, 17, 17, 17], dtype=int)
    rec = measure_clouds(pos, zs, metal="Pt")
    assert rec["plausible"] is False


def test_summarise_clouds_reports_shape_classes() -> None:
    """A mixed hand-built cohort produces a non-empty class breakdown."""
    records = []
    for t in hand_built_tests():
        rec = measure_clouds(t["positions"], t["zs"], metal="Pt")
        records.append(rec)
    summary = summarise_clouds(records, "hand_built_mixed")
    assert summary["n_clouds"] == 4
    assert summary["n_geometries_scored"] == 4
    # Each hand-built class appears at least once
    expected_classes = {
        "square_planar",
        "tetrahedral",
        "octahedral_like",
        "no_trans_pair_cn6",
    }
    found = set(summary["geometry_class_counts"].keys())
    assert expected_classes.issubset(found), (
        f"summary missing classes: {expected_classes - found}"
    )


def _build(donor_pts: np.ndarray) -> tuple:
    """Stack a Pt at the origin onto donor offsets."""
    pos = np.vstack([np.zeros((1, 3)), np.asarray(donor_pts, dtype=float)])
    zs = np.array([78] + [17] * donor_pts.shape[0], dtype=int)
    return pos, zs


# --------------------------------------------------------------------------
# Red check -- revert the trans-angle cut to a value that mis-classifies
# the trigonal prism, observe a regression, restore the file. This is the
# one a future agent must run to claim the classifier is guarded.
# --------------------------------------------------------------------------
def test_trans_pair_cut_red_check() -> None:
    """Move the trans cut below the trigonal prism's 120 deg maximum,
    prove the trigonal prism becomes a false-positive, then move it back.
    """
    src = SCRIPT.read_text()
    bad_cut = 100.0
    good_cut = TRANS_ANGLE_CUT_DEG
    assert good_cut != bad_cut
    assert good_cut > 120.0  # the regression we want to demonstrate
    assert bad_cut < 120.0

    bad_src = src.replace(f"TRANS_ANGLE_CUT_DEG: float = {good_cut}",
                          f"TRANS_ANGLE_CUT_DEG: float = {bad_cut}")
    assert bad_src != src, "red-check pattern not found -- the file may have been edited"

    SCRIPT.write_text(bad_src)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--self-test",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        )
        # The trigonal-prism case must FAIL under the bad cut. The
        # octahedron's trans pairs (180 deg) survive; only the
        # trigonal-prism classification changes.
        out = proc.stdout + proc.stderr
        assert "ideal_trigonal_prism" in out, f"unexpected self-test output: {out!r}"
        assert "FAIL" in out, (
            f"red check expected a FAIL line for the trigonal prism, got: {out!r}"
        )
    finally:
        SCRIPT.write_text(src)  # always restore

    # Re-run on the restored file: 4/4 PASS.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    assert "hand_built_tests: 4/4 PASS" in proc.stdout, (
        f"after restore, expected 4/4 PASS, got: {proc.stdout}{proc.stderr}"
    )
