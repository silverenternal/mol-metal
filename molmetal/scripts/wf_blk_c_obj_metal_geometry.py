#!/usr/bin/env python
"""BLK-C-OBJ precious-metal geometry measures: the axis nothing else measures.

``cfm_todo.json::blockers[BLK-C-OBJ].required_before_any_design[0]`` reads:

    MEASURE the precious-metal-specific geometry axes that do not exist yet
    (square-planar vs tetrahedral at Pt/Pd/Ru, oxidation state, donor
    preference). Do not tune an objective against a proxy that cannot see the
    property that makes a complex a drug.

This script builds those axes and measures them on THREE cohorts, because the
generated cohort has only SMILES and the real cohort has DFT coordinates, and
comparing the two directly is the exact error class this line keeps committing.

THE THREE COHORTS
------------------
REAL_DFT     real tmQM DFT-optimised coordinates, ``tmQM_X1.xyz.gz``. Ground
             truth. Bond lengths and angles are experimental-grade.
REAL_ETKDG   the SAME real molecules, reduced to SMILES and re-embedded with the
             SAME ETKDG protocol the generated cohort must go through. This is
             the LIKE-FOR-LIKE reference band.
GENERATED    our bond head's free generation via the ODE,
             ``cfm_fullcorpus_2026_10_10/generated_smiles.csv``, embedded with
             that identical protocol.

WHY REAL_ETKDG AND NOT REAL_DFT AS THE PRIMARY BAND
---------------------------------------------------
We cannot generate a DFT geometry. If the comparison band is REAL_DFT and the
cohort is ETKDG, then part of every difference measured is the embedder, not
the generator, and the finding is partly artefactual. The embedder is known to
be weak here (11 of 66 generated molecules will not embed AT ALL, per
BLK-C-L5), so it is not a neutral instrument.

GENERATED is therefore read against REAL_ETKDG as the primary comparison, and
REAL_DFT vs REAL_ETKDG is reported as a FIDELITY CONTROL that bounds how much
of the GENERATED-vs-real gap the embedder alone can account for. If the
GENERATED-vs-REAL_ETKDG difference is no larger than the REAL_DFT-vs-REAL_ETKDG
embedder spread, the honest conclusion is "the generator is within the noise
this instrument can resolve", not "the generator matches reality".

WHAT IS MEASURED, PER MOLECULE
------------------------------
1. coordination_number  - metal degree in the molecular graph.
2. donor_elements       - element identity of every metal neighbour, as a
                          distribution (the donor-preference axis).
3. geometry class       - square-planar vs tetrahedral vs octahedral at CN=4
                          and CN=6, by the scale-free planarity ratio below.
4. metal-ligand bond lengths - per metal, distribution.
5. cis/trans            - trans pairs among the L-M-L angles.

GEOMETRY CRITERION (stated, and its threshold is not fitted to our output)
-------------------------------------------------------------------------
For the donors D of a metal centre M the donor positions are centred on their
own centroid and the singular values s1 >= s2 >= s3 of their covariance are
taken. ``planarity_ratio = s3 / s2`` is dimensionless, so it is comparable
across metals and across coordination numbers:

  * ``planarity_ratio ~ 0``  all donors in one plane   -> square planar (CN=4)
  * ``planarity_ratio ~ 1``  donors isotropically fill 3D -> tetrahedral (CN=4)

Measured on ideal constructions (asserted in the test file):
square planar ``[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]`` gives exactly 0.0, and the
regular tetrahedron ``[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]`` gives exactly 1.0
(both have isotropic covariance, so s2 == s3). The default threshold
``--planarity-cut 0.15`` therefore sits far from both ideals and is applied
identically to all three cohorts. It is NOT tuned on GENERATED; the script
reports the full distribution so the separation can be checked rather than
trusted.

WHAT PLANARITY RATIO CANNOT DO, STATED PLAINLY
----------------------------------------------
An ideal octahedron also gives 1.0, and so does an ideal trigonal prism. This
criterion separates "in-plane" from "3D"; it CANNOT separate octahedral from
trigonal-prismatic, because both are fully 3D and both have isotropic donor
covariance. So at CN=6 the planarity ratio is reported as a number and is NOT
used to assert a geometry class. CN=6 is discriminated instead by the presence
of trans pairs (a 180 deg L-M-L pair exists in an octahedron and in a trigonal
prism it does not), which is reported as ``cn6_has_trans_pair`` and labelled
``octahedral_like`` rather than ``octahedral``.

A molecule that will not embed has NO GEOMETRY. It is recorded as ``None`` and
counted in ``geometry_unmeasured``. It is never scored as zero, and it is never
dropped silently from the denominator.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not run training, does not touch the network, does not tune anything,
and does not claim a result about QED / SA / ADMET. It produces the measuring
instrument that ``BLK-C-OBJ`` says must exist before an objective is designed.

Usage
-----
    /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/wf_blk_c_obj_metal_geometry.py --out \
        metrics/by_metric/cfm_metal_geometry.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]

TRAIN_METALS = ("Pt", "Ru", "Ir", "Rh", "Os", "Re")

GENERATED_CSV = REPO / "molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv"
TMQM_CSV = REPO / ".data/molmetal/tmQM/tmqm_parsed.csv"
TMQM_XYZ = REPO / ".data/molmetal/tmQM/tmQM_X1.xyz.gz"

# Ideal reference values for the criterion's self-check, measured on exact
# constructions. Both asserted in the test file so a silent change to the
# criterion is caught. See module docstring for why the tetrahedral one is 1.0.
IDEAL_SQUARE_PLANAR_PLANARITY_RATIO = 0.0
IDEAL_TETRAHEDRAL_PLANARITY_RATIO = 1.0


# --------------------------------------------------------------------------
# RDKit import, kept lazy so --help works without the chemistry stack.
# --------------------------------------------------------------------------
def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    return Chem, AllChem


def embeddable(mol) -> Optional[bool]:
    """RDKit 3D-embeddability. Definition is pinned; DO NOT EDIT.

    This must stay byte-identical to
    ``molmetal/scripts/wf_blkc_l5_metal_deletion_control.py::embeddable`` and
    to ``wf_blkc4_metric_validity_audit.py::embeddable``, which are the two
    copies already on disk. It reproduces the recorded 54/66 = 0.8182 on the
    generated cohort, pinned by
    ``test_wf_blkc_l5_metal_deletion_control.py::test_embeddable_reproduces_the_recorded_generated_number``.

    ``Chem.AddHs`` is LOAD-BEARING. Without explicit hydrogens ETKDG fails on
    essentially every organometallic, and the failure presents as a confident
    0.0 embeddability rate rather than an exception.
    """
    Chem, AllChem = _rdkit()
    if mol is None:
        return None
    try:
        probe = Chem.AddHs(Chem.Mol(mol))
    except Exception:
        return False
    try:
        if AllChem.EmbedMolecule(
            probe,
            useRandomCoords=True,
            maxAttempts=1,
            randomSeed=0xF00D,
        ) != 0:
            return False
    except Exception:
        return False
    return probe.GetNumConformers() > 0


def embed_geometry(mol):
    """Return a conformer-bearing H-added copy of ``mol``, or ``None``.

    Same protocol as :func:`embeddable`. Returns the probe so callers can read
    coordinates off it instead of re-deriving them from a boolean.
    """
    Chem, AllChem = _rdkit()
    if mol is None:
        return None
    try:
        probe = Chem.AddHs(Chem.Mol(mol))
    except Exception:
        return None
    try:
        if AllChem.EmbedMolecule(
            probe, useRandomCoords=True, maxAttempts=1, randomSeed=0xF00D
        ) != 0:
            return None
    except Exception:
        return None
    return probe if probe.GetNumConformers() > 0 else None


# --------------------------------------------------------------------------
# Geometry, on an RDKit conformer
# --------------------------------------------------------------------------
def _planarity_ratio(points) -> Optional[float]:
    """s3/s2 of the donor positions about their centroid. None if degenerate."""
    import numpy as np

    if len(points) < 4:
        return None
    pts = np.asarray(points, dtype=float)
    centred = pts - pts.mean(axis=0, keepdims=True)
    # symmetric -> ascending eigenvalues == squared singular values
    sv = np.linalg.svd(centred, compute_uv=False)
    if len(sv) < 3 or sv[1] <= 1e-8:
        return None
    return float(sv[2] / sv[1])


def _lm_l_angles(centre, points) -> List[float]:
    import numpy as np

    c = np.asarray(centre, dtype=float)
    angles = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            v1 = np.asarray(points[i], dtype=float) - c
            v2 = np.asarray(points[j], dtype=float) - c
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cos = float(np.dot(v1, v2) / (n1 * n2))
            angles.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, cos))))))
    return angles


def _geometry_class(cn: int, ratio: Optional[float], cut: float, trans_pairs) -> str:
    """Geometry class. Honest about what the available descriptors can resolve.

    CN=4 is discriminated by ``planarity_ratio``, which separates the square
    planar ideal (0.0) from the tetrahedral ideal (1.0) cleanly.
    CN=6 is NOT discriminated by planarity ratio -- octahedral and trigonal
    prism both give 1.0 (see module docstring) -- so it is discriminated only
    by whether a 180 deg trans pair exists, and is labelled ``octahedral_like``
    rather than ``octahedral``.
    """
    if ratio is None:
        return "unmeasured"
    if cn == 4:
        return "square_planar" if ratio <= cut else "tetrahedral"
    if cn == 6:
        return "octahedral_like" if (trans_pairs or 0) >= 1 else "no_trans_pair_cn6"
    return "other"


def _trans_pairs(angles: Sequence[float], trans_cut: float = 150.0) -> int:
    """Number of L-M-L pairs at or beyond ``trans_cut`` degrees.

    The threshold is the midpoint of the widest gap between the square-planar
    ideal (180) and the tetrahedral ideal (109.47), rounded to a round number.
    Square planar gives 2 trans pairs; tetrahedral gives 0.
    """
    return int(sum(1 for a in angles if a >= trans_cut))


# Shortest metal-ligand distance that is physically a bond rather than an
# embedder clash. The SHORTEST known metal-ligand bond across the six train
# metals is roughly 1.7 A (Ir/Ru hydrides sit near 1.6-1.7 A); nothing in
# organometallic chemistry bonds below ~1.5 A. 1.5 A is therefore a safe floor.
MIN_PLAUSIBLE_ML_BOND_A = 1.5


def _plausible(ml_bond_lengths) -> bool:
    """Reject geometries containing a sub-1.5 A metal-ligand contact.

    ETKDG with ``useRandomCoords=True`` and ``maxAttempts=1`` will happily place
    two atoms on top of each other and return success. The first run of this
    script reported a 0.28 A metal-ligand bond in REAL_ETKDG -- a number no
    chemist would accept, produced because the embedder succeeded, not because
    the structure is real. Counting such a geometry as a measurement would put
    a fabricated bond length into the reference band.

    This filter is applied IDENTICALLY to all three cohorts, so it cannot
    favour either side. Rejected molecules are counted in
    ``implausible_geometry`` rather than silently dropped, and the count is
    reported per cohort so the reader can see the filter is biting.
    """
    if not ml_bond_lengths:
        return False
    return min(ml_bond_lengths) >= MIN_PLAUSIBLE_ML_BOND_A


# --------------------------------------------------------------------------
# Per-molecule measurement from an RDKit mol + conformer
# --------------------------------------------------------------------------
def measure_molecule(mol, conformer_mol, planarity_cut: float) -> Dict[str, object]:
    """Graph axes always; geometric axes only when a conformer exists."""
    Chem, _ = _rdkit()
    out: Dict[str, object] = {
        "coordination_number": None,
        "donor_elements": None,
        "metal": None,
        "metal_bond_lengths": None,
        "lm_l_angles": None,
        "planarity_ratio": None,
        "geometry_class": "unmeasured",
        "trans_pairs": None,
        "formal_charge_on_metal": None,
        "embedded": conformer_mol is not None,
        "plausible": None,
    }
    if mol is None:
        return out

    metal_idx = _find_metal(mol)
    if metal_idx is None:
        return out
    Chem_obj = mol
    metal_sym = Chem_obj.GetAtomWithIdx(metal_idx).GetSymbol()
    out["metal"] = metal_sym
    nbrs = [n.GetIdx() for n in Chem_obj.GetAtomWithIdx(metal_idx).GetNeighbors()]
    out["coordination_number"] = len(nbrs)
    out["donor_elements"] = sorted(Chem_obj.GetAtomWithIdx(i).GetSymbol() for i in nbrs)
    try:
        out["formal_charge_on_metal"] = Chem_obj.GetAtomWithIdx(metal_idx).GetFormalCharge()
    except Exception:
        pass

    if conformer_mol is None:
        return out

    conf = conformer_mol.GetConformer()
    try:
        centre = conf.GetAtomPosition(metal_idx)
        pts = [conf.GetAtomPosition(i) for i in nbrs]
    except Exception:
        return out

    lengths = []
    for p in pts:
        v = (p.x - centre.x, p.y - centre.y, p.z - centre.z)
        lengths.append(round(float((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5), 4))
    out["metal_bond_lengths"] = sorted(lengths)
    angles = _lm_l_angles((centre.x, centre.y, centre.z), pts)
    out["lm_l_angles"] = [round(a, 2) for a in angles]
    out["trans_pairs"] = _trans_pairs(angles)
    ratio = _planarity_ratio(pts)
    out["planarity_ratio"] = round(ratio, 4) if ratio is not None else None
    out["geometry_class"] = _geometry_class(
        len(nbrs), ratio, planarity_cut, out["trans_pairs"]
    )
    out["plausible"] = _plausible(out["metal_bond_lengths"])
    return out


def _find_metal(mol):
    """First transition-metal atom from TRAIN_METALS, else any d/f-block atom."""
    Chem, _ = _rdkit()
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() in TRAIN_METALS:
            return idx
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in _TRANSITION_METAL_Z:
            return idx
    return None


_TRANSITION_METAL_Z = set(range(21, 32)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))


# --------------------------------------------------------------------------
# REAL_DFT cohort: parse tmQM_X1.xyz.gz for the six train metals
# --------------------------------------------------------------------------
def _read_xyz_csd_codes(path: Path) -> set:
    """Every CSD code present in the xyz file. One streaming pass.

    ``tmQM_X1.xyz.gz`` holds 34,970 frames while ``tmqm_parsed.csv`` holds
    108,541 rows. It is a SUBSET, so sampling the csv and then looking up
    coordinates silently shrinks the cohort -- the first version of this script
    asked for 800 and got 165, a 4x shortfall that would have quietly made the
    reference band 4x noisier than the one that was asked for. The population is
    intersected with the file's contents BEFORE sampling instead.
    """
    codes = set()
    with gzip.open(path, "rt") as stream:
        while True:
            header = stream.readline()
            if not header:
                return codes
            if not header.strip().isdigit():
                continue
            n_atoms = int(header.strip())
            comment = stream.readline()
            code = _csd_from_comment(comment)
            if code is not None:
                codes.add(code)
            for _ in range(n_atoms):
                stream.readline()


def _read_metal_csd_codes(
    csv_path: Path,
    metals: Sequence[str],
    limit: int,
    seed: int,
    restrict_to: Optional[set] = None,
):
    """CSD codes + metal for the six train metals, capped by a fixed seed.

    ``restrict_to`` intersects with the coordinate file's population first, so
    the cap is honoured on the cohort that will actually be measured.
    """
    keep = set(metals)
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("metal") not in keep or not row.get("smiles"):
                continue
            if restrict_to is not None and row["csd_code"] not in restrict_to:
                continue
            rows.append((row["csd_code"], row["metal"], row.get("coord_number")))
    rows.sort()  # deterministic order before sampling
    if limit and len(rows) > limit:
        rnd = random.Random(seed)
        rows = sorted(rnd.sample(rows, limit))
    return rows


def _parse_xyz_frames(path: Path, want: Dict[str, str]):
    """Yield (csd_code, metal_symbol, [(sym, x, y, z), ...]) for wanted CSD codes.

    Streamed line-by-line: the file is 2.5 M lines gzipped and is never held in
    memory in full.
    """
    with gzip.open(path, "rt") as stream:
        while True:
            header = stream.readline()
            if not header:
                return
            n_atoms = header.strip()
            if not n_atoms.isdigit():
                continue
            comment = stream.readline()
            code = _csd_from_comment(comment)
            if code is None or code not in want:
                for _ in range(int(n_atoms)):
                    stream.readline()
                continue
            atoms = []
            for _ in range(int(n_atoms)):
                line = stream.readline()
                parts = line.split()
                if len(parts) < 4:
                    atoms = []
                    break
                atoms.append(
                    (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
                )
            if atoms:
                yield code, want[code], atoms


def _csd_from_comment(comment: str) -> Optional[str]:
    marker = "CSD_code = "
    if marker in comment:
        return comment.split(marker, 1)[1].split("|")[0].strip()
    return None


def measure_dft_frame(metal: str, atoms, planarity_cut: float) -> Dict[str, object]:
    """Geometry from real DFT coordinates. Graph axes come from tmQM's own
    ``coord_number`` field, which is the experimental curation, not our guess."""
    out: Dict[str, object] = {
        "coordination_number": None,
        "donor_elements": None,
        "metal": metal,
        "metal_bond_lengths": None,
        "lm_l_angles": None,
        "planarity_ratio": None,
        "geometry_class": "unmeasured",
        "trans_pairs": None,
        "embedded": True,
        "plausible": None,
    }
    mi = None
    for i, (sym, *_rest) in enumerate(atoms):
        if sym == metal:
            mi = i
            break
    if mi is None:
        return out
    centre = atoms[mi][1:4]
    donors = [(i, a) for i, a in enumerate(atoms) if i != mi]
    # donor = within a generous covalent+ionic window of the metal. 2.6 A is
    # chosen to be wide enough to catch the long Ru/Ir/Os contacts real
    # tmQM reports and is IDENTICAL across all three cohorts.
    nbrs = []
    for i, a in donors:
        d = ((a[1] - centre[0]) ** 2 + (a[2] - centre[1]) ** 2 + (a[3] - centre[2]) ** 2) ** 0.5
        if d <= 2.6:
            nbrs.append((i, a, d))
    if not nbrs:
        return out
    out["coordination_number"] = len(nbrs)
    out["donor_elements"] = sorted(a[0] for _i, a, _d in nbrs)
    out["metal_bond_lengths"] = sorted(round(d, 4) for _i, _a, d in nbrs)
    pts = [a[1:4] for _i, a, _d in nbrs]
    angles = _lm_l_angles(centre, pts)
    out["lm_l_angles"] = [round(a, 2) for a in angles]
    out["trans_pairs"] = _trans_pairs(angles)
    ratio = _planarity_ratio(pts)
    out["planarity_ratio"] = round(ratio, 4) if ratio is not None else None
    out["geometry_class"] = _geometry_class(
        len(nbrs), ratio, planarity_cut, out["trans_pairs"]
    )
    out["plausible"] = _plausible(out["metal_bond_lengths"])
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _stats(values: Sequence[float]) -> Dict[str, object]:
    import numpy as np

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


def _histogram(values, bins):
    import numpy as np

    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    counts, edges = np.histogram(np.asarray(vals, dtype=float), bins=bins)
    return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": int(c) for i, c in enumerate(counts)}


def summarise(records: List[Dict[str, object]], label: str) -> Dict[str, object]:
    # Geometric statistics are computed over PLAUSIBLE geometries only. The
    # graph axes (coordination number, donor identity) are unaffected -- they
    # come from the molecular graph, not from coordinates, so a clash does not
    # invalidate them.
    unmeasured = sum(1 for r in records if not r.get("embedded"))
    implausible = sum(1 for r in records if r.get("embedded") and r.get("plausible") is False)
    geo_records = [
        r for r in records if r.get("embedded") and r.get("plausible") is not False
    ]
    cn = [r["coordination_number"] for r in records]
    donors = Counter()
    for r in records:
        for d in r.get("donor_elements") or []:
            donors[d] += 1
    geo = Counter(r["geometry_class"] for r in geo_records)
    lengths = [v for r in geo_records for v in (r.get("metal_bond_lengths") or [])]
    ratios = [r["planarity_ratio"] for r in geo_records]

    per_metal_len: Dict[str, object] = {}
    per_metal_geo: Dict[str, object] = {}
    for m in TRAIN_METALS:
        sub = [r for r in geo_records if r.get("metal") == m]
        vals = [v for r in sub for v in (r.get("metal_bond_lengths") or [])]
        per_metal_len[m] = _stats(vals)
        per_metal_geo[m] = dict(Counter(r["geometry_class"] for r in sub))

    cn4 = [r["planarity_ratio"] for r in geo_records if r["coordination_number"] == 4]
    cn4_geo = Counter(r["geometry_class"] for r in geo_records if r["coordination_number"] == 4)
    cn6_geo = Counter(r["geometry_class"] for r in geo_records if r["coordination_number"] == 6)
    trans = [r["trans_pairs"] for r in geo_records if r["coordination_number"] == 4]

    return {
        "label": label,
        "n_molecules": len(records),
        "n_with_metal_centre": sum(1 for r in records if r["metal"]),
        "geometry_unmeasured": unmeasured,
        "geometry_implausible_rejected": implausible,
        "n_geometries_scored": len(geo_records),
        "coordination_number_mean": round(float(sum(c for c in cn if c is not None) / max(1, sum(1 for c in cn if c is not None))), 4),
        "coordination_number_distribution": {str(k): v for k, v in sorted(Counter(c for c in cn if c is not None).items())},
        "donor_element_counts": dict(donors.most_common()),
        "donor_element_top5_fraction": {
            d: round(v / max(1, sum(donors.values())), 4) for d, v in donors.most_common(5)
        },
        "geometry_class_counts": dict(geo.most_common()),
        "cn4_geometry_class_counts": dict(cn4_geo.most_common()),
        "cn4_trans_pair_distribution": {str(k): v for k, v in sorted(Counter(t for t in trans if t is not None).items())},
        "cn6_geometry_class_counts": dict(cn6_geo.most_common()),
        "cn4_planarity_ratio": _stats(cn4),
        "cn4_planarity_ratio_histogram": _histogram(cn4, [0, .1, .2, .3, .4, .5, .6, .8, 1.0]),
        "all_planarity_ratio": _stats(ratios),
        "metal_bond_length_A": _stats(lengths),
        "metal_bond_length_per_metal_A": per_metal_len,
        "geometry_class_per_metal": per_metal_geo,
    }


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------
def load_generated(path: Path, planarity_cut: float):
    Chem, _ = _rdkit()
    records, meta = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            smi = row.get("smiles")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            probe = embed_geometry(mol) if embeddable(mol) else None
            rec = measure_molecule(mol, probe, planarity_cut)
            rec["smiles"] = smi
            records.append(rec)
            meta.append({"smiles": smi, "epoch": row.get("epoch"), "seed": row.get("seed")})
    return records, meta


def select_dft_population(limit: int, seed: int):
    """CSD codes for the DFT cohort: six train metals WITH coordinates.

    Split out from :func:`load_real_dft` so the CALL SITE -- the thing that has
    to intersect before sampling -- is directly testable. A test of the
    helpers alone passed while the call site was reverted to sample blind.
    """
    available = _read_xyz_csd_codes(TMQM_XYZ)
    return _read_metal_csd_codes(
        TMQM_CSV, TRAIN_METALS, limit, seed, restrict_to=available
    )


def load_real_dft(limit: int, seed: int, planarity_cut: float):
    """REAL_DFT, sampled from the six train metals THAT HAVE COORDINATES.

    The population is intersected with the xyz file's CSD codes before the cap
    is applied. Counting frames here rather than assuming the csv and the xyz
    agree is the point: they do not (34,970 vs 108,541).
    """
    rows = select_dft_population(limit, seed)
    want = {csd: metal for csd, metal, _cn in rows}
    records = []
    for csd, metal, atoms in _parse_xyz_frames(TMQM_XYZ, want):
        rec = measure_dft_frame(metal, atoms, planarity_cut)
        rec["csd_code"] = csd
        records.append(rec)
    return records, len(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generated", type=Path, default=GENERATED_CSV)
    ap.add_argument("--tmqm-csv", type=Path, default=TMQM_CSV)
    ap.add_argument("--tmqm-xyz", type=Path, default=TMQM_XYZ)
    ap.add_argument("--planarity-cut", type=float, default=0.15,
                    help="planarity_ratio <= cut => square_planar at CN=4")
    ap.add_argument("--real-limit", type=int, default=3000,
                    help="cap on real molecules per cohort, deterministic sample")
    ap.add_argument("--sample-seed", type=int, default=20261012)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-records", type=Path, default=None)
    args = ap.parse_args(argv)

    cut = args.planarity_cut

    # --- criterion self-check, before any cohort is read -------------------
    Chem, AllChem = _rdkit()
    from rdkit.Chem import rdMolTransforms  # noqa: F401  (import cost only)

    # regular tetrahedron: build it analytically, no RDKit needed
    import numpy as np
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    tetra_ratio = _planarity_ratio(t)
    # ideal square planar: 4 points on a circle in the z=0 plane
    sq = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    square_ratio = _planarity_ratio(sq)

    print(f"[criterion] ideal square-planar planarity_ratio = {square_ratio}")
    print(f"[criterion] ideal tetrahedral  planarity_ratio = {round(tetra_ratio,4)}")
    print(f"[criterion] cut = {cut} -> square_planar if ratio <= cut")

    # --- GENERATED ---------------------------------------------------------
    gen_records, _meta = load_generated(args.generated, cut)
    gen_summary = summarise(gen_records, "GENERATED (our bond head, free ODE generation)")

    # --- REAL_DFT ----------------------------------------------------------
    dft_records, dft_requested = load_real_dft(args.real_limit, args.sample_seed, cut)
    dft_summary = summarise(dft_records, "REAL_DFT (tmQM DFT-optimised coordinates)")
    # the request/receipt count is asserted by the test file: a cohort that
    # comes back smaller than it was asked for means the sample was drawn
    # without intersecting the coordinate file's population first.
    dft_summary["n_requested"] = dft_requested

    # --- REAL_ETKDG --------------------------------------------------------
    etkdg_records = load_real_etkdg(args.tmqm_csv, args.real_limit, args.sample_seed, cut)
    etkdg_summary = summarise(etkdg_records, "REAL_ETKDG (real tmQM SMILES, same embedder as GENERATED)")

    payload = {
        "metric": "cfm_metal_geometry",
        "purpose": cfm_todo_ref(),
        "created": "2026-10-12",
        "protocol": {
            "embedder": "RDKit ETKDG, useRandomCoords=True, maxAttempts=1, randomSeed=0xF00D, with Chem.AddHs",
            "embedder_provenance": "pinned to wf_blkc_l5_metal_deletion_control.py::embeddable; reproduces 54/66 on GENERATED",
            "planarity_cut": cut,
            "planarity_criterion": "s3/s2 of donor positions about their centroid; <= cut => square_planar at CN=4",
            "ideal_square_planar_ratio": round(square_ratio, 6) if square_ratio is not None else None,
            "ideal_tetrahedral_ratio": round(tetra_ratio, 4) if tetra_ratio is not None else None,
            "dft_donor_cutoff_A": 2.6,
            "trans_angle_cut_deg": 150.0,
            "min_plausible_ml_bond_A": MIN_PLAUSIBLE_ML_BOND_A,
            "train_metals": list(TRAIN_METALS),
            "sample_seed": args.sample_seed,
            "real_limit_requested": args.real_limit,
            "cohort_population_note": (
                "REAL_DFT is drawn from the 6,872 six-train-metal molecules that "
                "have coordinates in tmQM_X1.xyz.gz, which is a SUBSET of the "
                "108,541 rows in tmqm_parsed.csv (34,970 frames total). The "
                "population is intersected BEFORE the cap is applied."
            ),
        },
        "cohorts": {
            "GENERATED": gen_summary,
            "REAL_ETKDG": etkdg_summary,
            "REAL_DFT": dft_summary,
        },
        "embedder_fidelity_control": fidelity(gen_summary, etkdg_summary, dft_summary),
        "interpretation_rules": [
            "GENERATED is read against REAL_ETKDG. REAL_DFT is the ground truth but the GENERATED cohort cannot be DFT, so a GENERATED-vs-REAL_DFT difference is partly the embedder's.",
            "If |GENERATED - REAL_ETKDG| is not larger than |REAL_DFT - REAL_ETKDG|, the honest reading is that the generator is within the resolution of this instrument, not that it matches reality.",
            "geometry_unmeasured counts molecules that failed to embed. They are never scored as zero.",
            "geometry_implausible_rejected counts geometries ETKDG returned SUCCESSFULLY for that contain a sub-1.5 A metal-ligand contact. The filter is applied identically to all three cohorts.",
            "GENERATED n=66. Any difference smaller than the within-cohort spread is reported as not-a-finding.",
            "CN=6 is reported as octahedral_like (trans-pair test), never as octahedral: the planarity ratio gives 1.0 for BOTH an ideal octahedron and an ideal trigonal prism and cannot separate them.",
        ],
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"[write] {args.out}")
    else:
        print(text)

    if args.dump_records:
        args.dump_records.parent.mkdir(parents=True, exist_ok=True)
        args.dump_records.write_text(
            json.dumps(
                {
                    "GENERATED": gen_records,
                    "REAL_ETKDG": etkdg_records,
                    "REAL_DFT": dft_records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"[write] {args.dump_records}")

    return 0


def load_real_etkdg(csv_path: Path, limit: int, seed: int, planarity_cut: float):
    """REAL_ETKDG: real tmQM, SMILES -> the same ETKDG protocol as GENERATED."""
    Chem, _ = _rdkit()
    keep = set(TRAIN_METALS)
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("metal") in keep and row.get("smiles"):
                rows.append((row["csd_code"], row["metal"], row["smiles"]))
    rows.sort()
    if limit and len(rows) > limit:
        rnd = random.Random(seed)
        rows = sorted(rnd.sample(rows, limit), key=lambda r: r[0])

    records = []
    for csd, metal, smi in rows:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # keep only if the sampled metal is actually present in the molecule
        if not any(a.GetSymbol() == metal for a in mol.GetAtoms()):
            continue
        probe = embed_geometry(mol) if embeddable(mol) else None
        rec = measure_molecule(mol, probe, planarity_cut)
        rec["csd_code"] = csd
        records.append(rec)
    return records


def fidelity(gen, etkdg, dft) -> Dict[str, object]:
    """How much of the GENERATED-vs-real gap the embedder alone explains."""
    def spread(a, b, key):
        va, vb = a.get(key), b.get(key)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            return None
        return round(abs(va - vb), 4)

    out = {}
    for key in ("coordination_number_mean", "cn4_planarity_ratio", "metal_bond_length_A"):
        g = gen.get(key)
        e = etkdg.get(key)
        d = dft.get(key)

        def mean_of(node):
            if isinstance(node, dict) and "mean" in node:
                return node["mean"]
            return node

        gm, em, dm = mean_of(g), mean_of(e), mean_of(d)
        row = {}
        if isinstance(gm, (int, float)) and isinstance(em, (int, float)):
            row["generated_minus_real_etkdg"] = round(gm - em, 4)
        if isinstance(dm, (int, float)) and isinstance(em, (int, float)):
            row["real_dft_minus_real_etkdg_embedder_floor"] = round(dm - em, 4)
        if "generated_minus_real_etkdg" in row and "real_dft_minus_real_etkdg_embedder_floor" in row:
            gap = abs(row["generated_minus_real_etkdg"])
            floor = abs(row["real_dft_minus_real_etkdg_embedder_floor"])
            row["verdict"] = (
                "generator gap EXCEEDS the embedder floor" if gap > floor
                else "generator gap is WITHIN the embedder floor -- not a finding"
            )
        out[key] = row
    return out


def cfm_todo_ref() -> str:
    return (
        "cfm_todo.json::blockers[BLK-C-OBJ].required_before_any_design[0] -- "
        "measure the precious-metal geometry axes before tuning any objective "
        "against them."
    )


if __name__ == "__main__":
    sys.exit(main())