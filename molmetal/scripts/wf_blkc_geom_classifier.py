#!/usr/bin/env python
"""BLK-C-OBJ geometry pipeline that BYPASSES ETKDG.

Round 1 (docs/CFM_STATUS_2026_10_12.md Sec 9.4) measured that ETKDG erases
the precious-metal pharmacophore: paired, 91 of 99 real CN=4 molecules whose
DFT geometry is square-planar become TETRAHEDRAL after the embedder (cf.
``metrics/by_metric/cfm_metal_geometry_pairing_control.json``). Any
coordination-geometry number taken from generated SMILES therefore describes
RDKit, not the generator.

This module provides a classifier that takes raw (positions, atomic_numbers)
and computes, WITHOUT ETKDG and WITHOUT SMILES:

  1. coordination number
  2. M-L bond lengths
  3. trans-pair count (CN=6 discriminator that DOES separate octahedron from
     trigonal prism, which the planarity ratio cannot)
  4. SP-4 vs T-4 shape at CN=4 (planarity_ratio <= cut)
  5. octahedral_like vs no_trans_pair_cn6 at CN=6

The intent is to apply it twice -- once to REAL tmQM DFT coordinates, once
to the CFM-sampled clouds BEFORE any decode -- so the comparison is
DFT-vs-generated, like for like. Round 1's classifier
(``molmetal/scripts/wf_blk_c_obj_metal_geometry.py``) already has a
``measure_dft_frame`` path that does this on real tmQM; the missing piece
was a clean function that takes (positions, atomic_numbers) without an
RDKit Mol. This module provides that function, exposes it for reuse, and
pins its behaviour with hand-built-geometry unit tests.

SHAPE VOCABULARY
----------------
The CN=4 / CN=6 vocabulary follows MetalHawk (JCIM 2024, DOI
10.1021/acs.jcim.3c00873) which classifies CN and geometry from CSD +
MetalPDB at 96.51%; at CN=4 its vocabulary is T-4 vs SP-4, and at CN=6
its vocabulary distinguishes the octahedron from the trigonal prism by
the trans-pair presence. The geometric tolerances used here are
documented in-line at each helper (Carvajal/Novoa/Alvarez are the standard
source for the metal-ligand 2.6 A donor window used here for CN
enumeration; the SP/T-4 planarity cut is the midpoint between ideal SP
(0.0) and ideal T-4 (1.0) of the s3/s2 ratio, sitting at 0.15).

WHAT THIS MODULE CANNOT DO
--------------------------
- It cannot separate octahedral from trigonal-prismatic by the
  planarity ratio alone -- both give 1.0. It uses the trans-pair
  count instead, and a hand-built test pins that separation.
- It does not run training, does not touch the network, and does not
  claim a result about QED / SA / ADMET.
- It does not separate every d/f-block polyhedron (capped square prism,
  pentagonal bipyramid, etc.); they are reported as ``other`` and the
  count is recorded.

Usage
-----
    from molmetal.scripts.wf_blkc_geom_classifier import (
        measure_clouds, summarise_clouds, hand_built_tests,
    )

    rec = measure_clouds(positions, atomic_numbers, metal="Pt")
    summary = summarise_clouds([rec, rec, ...])

CLI
---
    /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/wf_blkc_geom_classifier.py --self-test
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Project's six train metals.  Anything else in the cloud is a DONOR, never
#: a second metal centre; the classifier is single-metal, like the round-1
#: path it extends.
TRAIN_METALS: Dict[str, int] = {
    "Pt": 78,
    "Ru": 44,
    "Ir": 77,
    "Rh": 45,
    "Os": 76,
    "Re": 75,
}

#: DFT donor window. Round-1 audit (cfm_metal_geometry.json protocol) found
#: that 2.6 A is wide enough to catch the long Ru/Ir/Os contacts real tmQM
#: reports, while rejecting non-bonded contacts. Used identically here.
DFT_DONOR_CUTOFF_A: float = 2.6

#: Trans-pair threshold. Midpoint of the widest gap between the SP-4
#: ideal (180 deg) and the T-4 ideal (109.47 deg), rounded to a round number.
#: An ideal octahedron has 3 pairs at 180 deg; an ideal trigonal prism has 0.
TRANS_ANGLE_CUT_DEG: float = 150.0

#: Shortest plausible metal-ligand bond. 1.5 A sits below every known
#: organometallic bond (Ir/Ru hydrides bottom out near 1.6-1.7 A) and is
#: therefore an unambiguous embedder-clash floor.
MIN_PLAUSIBLE_ML_BOND_A: float = 1.5

#: Planarity cut for SP-4 vs T-4 at CN=4. The midpoint of the gap between
#: the ideal SP (0.0) and ideal T-4 (1.0) of s3/s2, sitting at 0.15.
PLANARITY_CUT: float = 0.15


# --------------------------------------------------------------------------
# Pure geometry helpers -- no RDKit, no ETKDG, no SMILES
# --------------------------------------------------------------------------
def _planarity_ratio(points: np.ndarray) -> Optional[float]:
    """s3/s2 of donor positions about their centroid.

    Returns ``None`` if the cloud is degenerate (fewer than 4 donors, or
    s2 below numerical zero). Ideal square planar (4 points on a circle in
    z=0) gives 0.0; ideal tetrahedron (4 isotropic vertices) gives 1.0
    (because s2 == s3); an ideal octahedron ALSO gives 1.0; an ideal
    trigonal prism ALSO gives 1.0. The ratio separates "in-plane" from
    "3D" but cannot separate octahedron from trigonal prism.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 3:
        return None
    centred = pts - pts.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    if len(sv) < 3 or sv[1] <= 1e-8:
        return None
    return float(sv[2] / sv[1])


def _lm_l_angles(centre: np.ndarray, points: np.ndarray) -> List[float]:
    """Every L-M-L angle in degrees. Empty if either input is empty."""
    c = np.asarray(centre, dtype=float).reshape(-1)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return []
    angles: List[float] = []
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            v1 = pts[i] - c
            v2 = pts[j] - c
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cos = float(np.dot(v1, v2) / (n1 * n2))
            angles.append(float(math.degrees(math.acos(max(-1.0, min(1.0, cos))))))
    return angles


def _trans_pairs(angles: Sequence[float], trans_cut: float = TRANS_ANGLE_CUT_DEG) -> int:
    """Number of L-M-L pairs at or beyond ``trans_cut`` degrees.

    An ideal octahedron has 3 trans pairs (180 deg). An ideal trigonal
    prism has 0 (the maximum angle is 120 deg). This is the discriminator
    the planarity ratio cannot supply.
    """
    return int(sum(1 for a in angles if a >= trans_cut))


def _geometry_class(
    cn: int, ratio: Optional[float], cut: float, trans_pairs: Optional[int]
) -> str:
    """Shape classification. Honest about what the descriptors can resolve.

    CN=4 -> SP-4 iff planarity_ratio <= cut, else T-4.
    CN=6 -> octahedral_like iff >=1 trans pair, else no_trans_pair_cn6.
    Anything else -> "other". A missing ratio or trans pair -> "unmeasured".
    """
    if ratio is None:
        return "unmeasured"
    if cn == 4:
        return "square_planar" if ratio <= cut else "tetrahedral"
    if cn == 6:
        return "octahedral_like" if (trans_pairs or 0) >= 1 else "no_trans_pair_cn6"
    return "other"


# --------------------------------------------------------------------------
# Per-cloud classifier -- the function that takes positions+Zs and returns
# every axis without RDKit.
# --------------------------------------------------------------------------
def _find_metal_idx(atomic_numbers: np.ndarray, want: Optional[str] = None) -> Optional[int]:
    """Index of the target metal in ``atomic_numbers``.

    Defaults to the first train metal present. If ``want`` is given (e.g.
    "Pt") that element is preferred -- important for a generated cohort
    that may contain a stray non-target metal.
    """
    if want is not None and want in TRAIN_METALS:
        z_want = TRAIN_METALS[want]
        idxs = np.where(atomic_numbers == z_want)[0]
        if len(idxs) > 0:
            return int(idxs[0])
    for sym, z in TRAIN_METALS.items():
        idxs = np.where(atomic_numbers == z)[0]
        if len(idxs) > 0:
            return int(idxs[0])
    return None


def measure_clouds(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    metal: Optional[str] = None,
    donor_cutoff_A: float = DFT_DONOR_CUTOFF_A,
    planarity_cut: float = PLANARITY_CUT,
) -> Dict[str, object]:
    """One cloud -> every geometry axis. No RDKit, no SMILES, no ETKDG.

    Parameters
    ----------
    positions
        (N, 3) array of Cartesian coordinates, in Angstrom. Should be
        heavy-atom only; H is treated as a donor if present (callers
        normally strip H first, which is the trainer's behaviour).
    atomic_numbers
        (N,) integer atomic numbers.
    metal
        Optional element symbol (e.g. "Pt"). If given, that element is
        preferred when locating the centre; otherwise the first train
        metal found is used.
    donor_cutoff_A
        Distance from the metal within which an atom is treated as a
        bonded donor. Default 2.6 A (round-1 protocol).
    planarity_cut
        SP-4 / T-4 boundary. Default 0.15, the midpoint between the two
        ideal values of s3/s2.

    Returns
    -------
    dict
        All keys: ``coordination_number``, ``donor_elements``,
        ``metal``, ``metal_bond_lengths``, ``lm_l_angles``,
        ``planarity_ratio``, ``trans_pairs``, ``geometry_class``,
        ``plausible``, ``centred``. ``embedded`` is set True (the
        geometric axes ARE measured) or False (no metal centre found).
    """
    pos = np.asarray(positions, dtype=float)
    zs = np.asarray(atomic_numbers, dtype=int)
    out: Dict[str, object] = {
        "coordination_number": None,
        "donor_elements": None,
        "metal": None,
        "metal_bond_lengths": None,
        "lm_l_angles": None,
        "planarity_ratio": None,
        "trans_pairs": None,
        "geometry_class": "unmeasured",
        "embedded": False,
        "plausible": None,
        "centred": False,
    }
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != zs.shape[0]:
        return out
    mi = _find_metal_idx(zs, metal)
    if mi is None:
        return out
    centre = pos[mi]
    # EDM-style zero-CoM (Hoogeboom 2022, arXiv:2203.17003 sec 3). CSD XYZ
    # files sit at an arbitrary Cartesian frame; the trainer does the
    # subtraction, so the classifier does too. Failing to centre would
    # inject the position of the molecule as a whole into every descriptor.
    pos_centre = pos - centre
    centred = True
    out["centred"] = centred
    # Find bonded donors: every non-metal atom within donor_cutoff_A of the
    # metal. This is a strict distance rule -- no molecular graph, no
    # valence table, no SMARTS.
    diffs = pos_centre
    dists = np.linalg.norm(diffs, axis=1)
    donor_mask = (dists > 1e-6) & (dists <= float(donor_cutoff_A))
    donor_idx = np.where(donor_mask)[0]
    if donor_idx.size == 0:
        return out
    # Sort donors by distance so bond-length reporting is deterministic.
    donor_idx = donor_idx[np.argsort(dists[donor_idx])]
    out["coordination_number"] = int(donor_idx.size)
    out["metal"] = metal or _z_to_symbol(int(zs[mi]))
    out["donor_elements"] = sorted(_z_to_symbol(int(zs[i])) for i in donor_idx)
    out["metal_bond_lengths"] = [round(float(dists[i]), 4) for i in donor_idx]
    out["plausible"] = (
        min(out["metal_bond_lengths"]) >= MIN_PLAUSIBLE_ML_BOND_A
        if out["metal_bond_lengths"]
        else None
    )
    donor_pts = pos_centre[donor_idx]
    angles = _lm_l_angles(np.zeros(3), donor_pts)
    out["lm_l_angles"] = [round(a, 2) for a in angles]
    out["trans_pairs"] = _trans_pairs(angles)
    ratio = _planarity_ratio(donor_pts)
    out["planarity_ratio"] = round(ratio, 4) if ratio is not None else None
    out["geometry_class"] = _geometry_class(
        out["coordination_number"], ratio, planarity_cut, out["trans_pairs"]
    )
    out["embedded"] = True
    return out


def _z_to_symbol(z: int) -> str:
    """Z -> element symbol. Restricted to the project's vocabulary."""
    for sym, zz in TRAIN_METALS.items():
        if zz == z:
            return sym
    common = {
        1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
        15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
    }
    return common.get(z, f"Z{z}")


# --------------------------------------------------------------------------
# Summarisation -- cohort-level, like the round-1 JSON
# --------------------------------------------------------------------------
def _stats(values: Sequence[Optional[float]]) -> Dict[str, object]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "sd": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def _histogram(values: Sequence[Optional[float]], bins) -> Dict[str, int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    counts, edges = np.histogram(np.asarray(vals, dtype=float), bins=bins)
    return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": int(c) for i, c in enumerate(counts)}


def summarise_clouds(
    records: Sequence[Dict[str, object]], label: str
) -> Dict[str, object]:
    """Aggregate per-cloud records into a cohort summary."""
    unmeasured = sum(1 for r in records if not r.get("embedded"))
    implausible = sum(
        1 for r in records if r.get("embedded") and r.get("plausible") is False
    )
    geo = [r for r in records if r.get("embedded") and r.get("plausible") is not False]
    cn = [r["coordination_number"] for r in records]
    donors = Counter()
    for r in records:
        for d in r.get("donor_elements") or []:
            donors[d] += 1
    classes = Counter(r["geometry_class"] for r in geo)
    lengths = [v for r in geo for v in (r.get("metal_bond_lengths") or [])]
    ratios = [r["planarity_ratio"] for r in geo]
    per_metal_len: Dict[str, Dict[str, object]] = {}
    per_metal_geo: Dict[str, Dict[str, int]] = {}
    for m in TRAIN_METALS:
        sub = [r for r in geo if r.get("metal") == m]
        per_metal_len[m] = _stats(
            [v for r in sub for v in (r.get("metal_bond_lengths") or [])]
        )
        per_metal_geo[m] = dict(Counter(r["geometry_class"] for r in sub))
    cn4_geo = Counter(r["geometry_class"] for r in geo if r["coordination_number"] == 4)
    cn6_geo = Counter(r["geometry_class"] for r in geo if r["coordination_number"] == 6)
    cn4 = [r["planarity_ratio"] for r in geo if r["coordination_number"] == 4]
    trans = [r["trans_pairs"] for r in geo if r["coordination_number"] == 4]
    return {
        "label": label,
        "n_clouds": len(records),
        "n_with_metal_centre": sum(1 for r in records if r.get("metal")),
        "geometry_unmeasured": unmeasured,
        "geometry_implausible_rejected": implausible,
        "n_geometries_scored": len(geo),
        "coordination_number_mean": round(
            float(sum(c for c in cn if c is not None))
            / max(1, sum(1 for c in cn if c is not None)),
            4,
        ),
        "coordination_number_distribution": {
            str(k): v
            for k, v in sorted(Counter(c for c in cn if c is not None).items())
        },
        "donor_element_counts": dict(donors.most_common()),
        "donor_element_top5_fraction": {
            d: round(v / max(1, sum(donors.values())), 4)
            for d, v in donors.most_common(5)
        },
        "geometry_class_counts": dict(classes.most_common()),
        "cn4_geometry_class_counts": dict(cn4_geo.most_common()),
        "cn4_trans_pair_distribution": {
            str(k): v
            for k, v in sorted(Counter(t for t in trans if t is not None).items())
        },
        "cn6_geometry_class_counts": dict(cn6_geo.most_common()),
        "cn4_planarity_ratio": _stats(cn4),
        "cn4_planarity_ratio_histogram": _histogram(
            cn4, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        ),
        "all_planarity_ratio": _stats(ratios),
        "metal_bond_length_A": _stats(lengths),
        "metal_bond_length_per_metal_A": per_metal_len,
        "geometry_class_per_metal": per_metal_geo,
    }


# --------------------------------------------------------------------------
# Hand-built geometries -- the test surface the user demanded
# --------------------------------------------------------------------------
def ideal_square_planar(radius: float = 2.0) -> np.ndarray:
    """4 points on a circle in the z=0 plane. CENTRED at the metal."""
    return np.array(
        [
            [radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, -radius, 0.0],
        ],
        dtype=float,
    )


def ideal_tetrahedron(radius: float = 2.0) -> np.ndarray:
    """4 isotropic vertices. CENTRED at the metal."""
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    ) * (radius / math.sqrt(3.0))


def ideal_octahedron(radius: float = 2.0) -> np.ndarray:
    """6 donors on the +/- axes. CENTRED at the metal."""
    return np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=float,
    )


def ideal_trigonal_prism(radius: float = 2.0, height: float = 2.0) -> np.ndarray:
    """3 donors above, 3 donors below, rotated 60 deg.

    This is the case the s3/s2 ratio cannot separate from the octahedron.
    """
    top = []
    bot = []
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        top.append([radius * math.cos(ang), radius * math.sin(ang), height / 2.0])
        bot.append([radius * math.cos(ang), radius * math.sin(ang), -height / 2.0])
    return np.array(top + bot, dtype=float)


def hand_built_tests() -> List[Dict[str, object]]:
    """The full set of hand-built geometry checks.

    These return their per-geometry classification, which the unit-test
    file iterates and asserts. A regression in the helper functions shows
    up here before any cohort is measured.

    Layout convention: the metal sits at the ORIGIN; the donor positions
    returned by ``ideal_square_planar`` etc. are the donor offsets. The
    classifier subtracts the metal position, so the same coordinates work
    with the metal at the origin or anywhere else.
    """
    z_pt = TRAIN_METALS["Pt"]
    out: List[Dict[str, object]] = []

    def _build(donor_pts: np.ndarray) -> tuple:
        """Stack a Pt at the origin onto donor offsets; return (pos, zs)."""
        pos = np.vstack([np.zeros((1, 3)), np.asarray(donor_pts, dtype=float)])
        zs = np.array([z_pt] + [17] * donor_pts.shape[0], dtype=int)
        return pos, zs

    sp_pos, sp_zs = _build(ideal_square_planar())
    out.append(
        {
            "name": "ideal_square_planar",
            "cn_expected": 4,
            "class_expected": "square_planar",
            "positions": sp_pos,
            "zs": sp_zs,
        }
    )
    th_pos, th_zs = _build(ideal_tetrahedron())
    out.append(
        {
            "name": "ideal_tetrahedron",
            "cn_expected": 4,
            "class_expected": "tetrahedral",
            "positions": th_pos,
            "zs": th_zs,
        }
    )
    # CN=6 -- the case the user explicitly flagged
    oc_pos, oc_zs = _build(ideal_octahedron())
    out.append(
        {
            "name": "ideal_octahedron",
            "cn_expected": 6,
            "class_expected": "octahedral_like",
            "positions": oc_pos,
            "zs": oc_zs,
        }
    )
    tp_pos, tp_zs = _build(ideal_trigonal_prism())
    out.append(
        {
            "name": "ideal_trigonal_prism",
            "cn_expected": 6,
            "class_expected": "no_trans_pair_cn6",
            "positions": tp_pos,
            "zs": tp_zs,
        }
    )
    return out


# --------------------------------------------------------------------------
# CLI -- self-test prints a PASS/FAIL line, which is what `pytest -q` and
# the workflow harness both look for.
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run the hand-built geometry checks and print PASS/FAIL.",
    )
    args = ap.parse_args(argv)
    if not args.self_test:
        ap.print_help()
        return 0

    tests = hand_built_tests()
    n_pass = 0
    n_total = 0
    for t in tests:
        n_total += 1
        pos = t["positions"]
        # Centre the metal at the origin so measure_clouds's CoM shift is a
        # no-op; that isolates the geometry from positioning noise.
        pos_c = pos.copy()
        zs = t["zs"]
        rec = measure_clouds(pos_c, zs, metal="Pt")
        ok_cn = rec["coordination_number"] == t["cn_expected"]
        ok_class = rec["geometry_class"] == t["class_expected"]
        ok = ok_cn and ok_class
        n_pass += int(ok)
        print(
            f"[{ 'OK' if ok else 'FAIL' }] {t['name']}: "
            f"cn={rec['coordination_number']} (expect {t['cn_expected']}), "
            f"class={rec['geometry_class']!r} (expect {t['class_expected']!r}), "
            f"planarity_ratio={rec['planarity_ratio']}, trans_pairs={rec['trans_pairs']}"
        )
    print(f"hand_built_tests: {n_pass}/{n_total} PASS")
    if n_pass != n_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
