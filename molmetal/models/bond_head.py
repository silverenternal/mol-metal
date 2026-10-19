"""Learned bond-order head + bond-aware decoder for de novo SBDD.

A1 (WF-1, round-10 decoder audit) — addresses the open question
"Can the decoder actually recover chemistry from a learned bond-order
classifier?"  This module sits downstream of the flow-matching
coordinate generator and consumes per-pair features to predict
discrete bond orders, then assembles an RDKit ``Mol`` from the
predicted bonds and atom cloud.

Why a learned head instead of rule-based?
-----------------------------------------
The original decoder wired through ``molmetal_lam.lam_chem`` relies on
SMARTS pattern matching (amide / triazole / pincer / etc.).  That
covers ~80 % of metal-organic chemistries but fails on rarer motifs
(dative Pd-N, mixed-valence Fe clusters, strained bicyclics).  A
learned per-pair MLP generalises to those motifs while still emitting
a 5-class discrete bond order we can hand to RDKit for valence
sanitisation.

Inputs per pair ``(i, j)``
--------------------------
``distance``         scalar (Å) — Euclidean distance between atom i/j.
``z_i``              int atomic number of atom i.
``z_j``              int atomic number of atom j.
``angle_to_metal``   scalar (rad) — deviation from ideal donor–metal
                     angle for the geometry registered in
                     ``molmetal_lam.priors.metal_geometry.DEFAULT_METAL_GEOMETRY``.
                     Pass 0.0 if no metal centre is in the molecule.
``is_dative_candidate`` bool — the pair has (donor, metal) Z roles and
                     distance < 2.5 Å; the head should bias towards the
                     dative class for such pairs.

Outputs
-------
Logits over 5 classes:

    0 → no bond
    1 → single (covalent, default)
    2 → double (covalent, e.g. C=O)
    3 → triple (covalent, e.g. C#C)
    4 → aromatic (covalent, alternating single/double in ring)

Dative bonds are *not* a separate class — they reuse ``single`` with a
post-hoc flag because RDKit's valence model treats ``:`` and ``-``
identically.  The head biases its single-bond logit for dative
candidates and the decoder records them in a sidecar ``dative_bonds``
list on the returned ``Mol`` so downstream consumers can render them
differently.

Training recipe (synthetic, tmQM-style)
----------------------------------------
No pretrained checkpoint is shipped — the head is trained on a small
synthetic dataset of tmQM-style bond patterns
(:func:`make_synthetic_training_set`).  Training is a 200-step
closed-form supervised pass via :func:`train_synthetic` that reaches
~95 % top-1 accuracy on the held-out validation set and is good
enough for unit testing the decoder wiring.  A full re-train on the
real tmQM-21k corpus would slot in at ``train_synthetic`` call-site.

Honest framing
--------------
The numbers reported in :file:`molmetal/reports/wf1_a1_bond_head.md`
were MEASURED on 2026-09-14 against the synthetic dataset (200 steps,
``torch<=2.14`` CPU seed 0).  PROJECTED figures (e.g. tmQM-21k test
accuracy ≥ 92 %) are not yet validated and are clearly marked in the
report as PROJECTED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the canonical metal -> geometry dispatch from priors/metal_geometry.py.
# This keeps the bond head's notion of "is dative candidate?" in sync with
# the geometric prior in the rest of the pipeline.
from molmetal_lam.priors.metal_geometry import DEFAULT_METAL_GEOMETRY

# Wf-A16 (2026-10-13) — c-flow pocket propagation. PocketCloud lives in
# molmetal.data.tmqm so the loader owns the PDB parsing.  The forward
# reference here is import-only when TYPE_CHECKING is on (dataclass
# fields are evaluated at __post_init__ time, not annotation time, so
# the runtime dependency is one-way: tmqm.py does not import
# bond_head.py).
from molmetal.data.tmqm import PocketCloud  # noqa: E402  (post-import group)


__all__ = [
    "BOND_NO_BOND",
    "BOND_SINGLE",
    "BOND_DOUBLE",
    "BOND_TRIPLE",
    "BOND_AROMATIC",
    "BOND_LABELS",
    "NUM_BOND_CLASSES",
    "MAX_ATOMIC_NUMBER",
    "PAIR_FEATURE_NAMES",
    "PAIR_FEATURE_DIM",
    "BondOrderHead",
    "AtomCloud",
    "PairFeature",
    "BondAwareDecoder",
    "make_synthetic_training_set",
    "train_synthetic",
    "build_bond_pattern_mask",
    "ConnectivityAwareDecoder",
]


# ---------------------------------------------------------------------------
# Bond class constants
# ---------------------------------------------------------------------------
BOND_NO_BOND: int = 0
BOND_SINGLE: int = 1
BOND_DOUBLE: int = 2
BOND_TRIPLE: int = 3
BOND_AROMATIC: int = 4

BOND_LABELS: Tuple[str, ...] = ("none", "single", "double", "triple", "aromatic")
NUM_BOND_CLASSES: int = 5

# Wf-A86 (2026-10-19) — nominal Wiberg bond orders per class, the scale
# AtomCloud.bond_orders_continuous arrives on (tmQM: single ~1.0, aromatic
# ~1.4-1.5, double ~2.0, triple ~3.0).  Used by the synthetic generator so
# the bo_target column means the same thing in training and at inference.
_WIBERG_BO_BY_ORDER: dict = {
    BOND_NO_BOND: 0.0,
    BOND_SINGLE: 1.0,
    BOND_DOUBLE: 2.0,
    BOND_TRIPLE: 3.0,
    BOND_AROMATIC: 1.5,
}

# Fraction of synthetic rows that carry the teacher-forced BO instead of the
# gap fallback.  The generation path (a CFM-sampled cloud) only ever has the
# gap, so this is the fraction of the training set that does NOT match what
# the head sees when it is actually used — keep it a minority.  MEASURED
# (seed 0, 200 epochs): p=0.5 -> gap-regime acc 0.944 / aggregate 0.974;
# p=0.25 -> see Wf-A86 catalog entry.  Changing it changes default_trained_head.
_BO_TEACHER_FORCE_P: float = 0.25

# Maximum atomic number we one-hot.  Covers all of period 1-6 plus lanthanides
# up to 71.  Higher atomic numbers (actinides etc.) are clipped — they
# rarely appear in drug-like SBDD ligands.
MAX_ATOMIC_NUMBER: int = 90


# ---------------------------------------------------------------------------
# Pair-feature layout — the single source of truth for the head's input width
# ---------------------------------------------------------------------------
# 2026-10-19: the width used to be written as a literal in four places (this
# module's in_dim default, the synthetic generator's rows, the live
# BondAwareDecoder._featurise stack, and two tests).  Wf-A11 widened the
# layout from 9 to 11 and the tests kept asserting 9, so test_bond_head
# failed against a head that was correct.  Declaring the layout once means
# adding or reordering a feature is a one-line change here; every producer
# builds its vector BY NAME, so a name that is declared but not populated
# raises instead of silently emitting a shorter row.
PAIR_FEATURE_NAMES: Tuple[str, ...] = (
    "distance",        # 0 — pair distance (Å)
    "bucket_i",        # 1 — coarse Z bucket of atom i
    "bucket_j",        # 2 — coarse Z bucket of atom j
    "angle_to_metal",  # 3 — deviation from the ideal dative angle (rad)
    "is_dative",       # 4 — dative-candidate flag
    "z_i",             # 5 — Z_i / MAX_ATOMIC_NUMBER
    "z_j",             # 6 — Z_j / MAX_ATOMIC_NUMBER
    "bucket_prod",     # 7 — bucket_i * 6 + bucket_j
    "gap",             # 8 — distance - single-bond anchor
    "d_to_metal",      # 9 — midpoint-to-metal distance (0 when no metal)
    "bo_target",       # 10 — teacher-forced Wiberg BO, else the gap fallback
)
PAIR_FEATURE_DIM: int = len(PAIR_FEATURE_NAMES)


def _pair_feature_vector(values: "dict[str, object]") -> "list[object]":
    """Order a named feature mapping into the canonical layout.

    Called by every producer of pair features (the synthetic generator and
    BondAwareDecoder._featurise).  Raises on a missing or unknown name
    rather than padding or truncating, so a layout change can never again
    surface as a silent (N, 9) vs in_dim=11 mismatch deep inside
    train_synthetic.
    """
    missing = [n for n in PAIR_FEATURE_NAMES if n not in values]
    extra = [k for k in values if k not in PAIR_FEATURE_NAMES]
    if missing or extra:
        raise RuntimeError(
            "pair-feature layout mismatch: "
            f"missing={missing} extra={extra} "
            f"(PAIR_FEATURE_NAMES={PAIR_FEATURE_NAMES})"
        )
    return [values[n] for n in PAIR_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Synthetic training set
# ---------------------------------------------------------------------------
# Each entry is (features..., label).  Features = (distance, Z_i, Z_j,
# angle_to_metal, is_dative_candidate).  We use tmQM-style bond-pattern
# fingerprints: covalent radii ~ sum of Bondi single-bond radii, with
# a multiplicative shrinkage for higher orders (double ≈ 0.87,
# triple ≈ 0.78, aromatic ≈ 0.92 — values from Pyykkö 2009 averaged
# across C/N/O).
_SINGLE_RADII: dict = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.00,
    26: 1.32,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    77: 1.41,
    78: 1.36,
}

# Hard-coded single-bond distances for N-N, N-O, C-N where the
# additive model underestimates.  tmQM empirical averages (Å).
_NN_SINGLE: float = 1.45
_NO_SINGLE: float = 1.40
_CN_SINGLE_AMIN: float = 1.47  # amine C-N
_CN_SINGLE_AROM: float = 1.34  # aromatic (aniline) C-N
_CO_SINGLE: float = 1.43  # ether / alcohol C-O


def _bond_distance(z_i: int, z_j: int, order: int) -> float:
    """Return a typical tmQM bond distance (Å) for the given order.

    MEASURED on Pyykkö (2009) tables for C/N/O averaged; PROJECTED for
    metals based on bondi + 0.15 Å offset to keep dative bonds tight.
    """
    # Empirical single-bond anchors for known chemistries (tmQM averages).
    pair_key = tuple(sorted((z_i, z_j)))
    if order == BOND_SINGLE:
        if pair_key == (7, 7):
            return _NN_SINGLE
        if pair_key in ((7, 8), (8, 8)):
            return _NO_SINGLE
        if pair_key == (6, 7):
            return _CN_SINGLE_AMIN
        if pair_key == (6, 8):
            return _CO_SINGLE
        if pair_key == (7, 1):
            return 1.01
        if pair_key == (8, 1):
            return 0.96
        if pair_key == (6, 1):
            return 1.09
    ri = _SINGLE_RADII.get(z_i, 1.20)
    rj = _SINGLE_RADII.get(z_j, 1.20)
    base = ri + rj
    if order == BOND_DOUBLE:
        return base * 0.87
    if order == BOND_TRIPLE:
        return base * 0.78
    if order == BOND_AROMATIC:
        return base * 0.92
    return base  # single / default


def _is_metal(z: int) -> bool:
    return z in DEFAULT_METAL_GEOMETRY


def _is_donor(z: int) -> bool:
    """Crude donor classifier — N, O, S, P are typical lone-pair donors."""
    return z in (7, 8, 15, 16)


def _bo_target_for_edges(
    cloud: "AtomCloud",
    edge_index: torch.Tensor,
    gap: torch.Tensor,
) -> torch.Tensor:
    """Merge a cloud's continuous Wiberg BOs into the gap fallback.

    Wf-A86 (2026-10-19).  The ``bo_target`` column was added by Wf-A11 as
    "the loader's float Wiberg BO if the pair is in the loader's per-pair
    list, else 0 (gap)", but the featuriser read it from ``PairFeature``,
    which has no such field, so the column was permanently the gap.  The
    data actually lives on :class:`AtomCloud` (``bonds`` +
    ``bond_orders_continuous``, both (E,) aligned with each other).

    Returns ``gap`` unchanged when the cloud carries no bond truth — the
    CFM generation path samples positions and Z only, so its input is
    bit-identical to before.  When the cloud DOES carry truth, entries for
    real bonds get the loader's Wiberg BO and every other candidate pair
    keeps the gap; that is teacher forcing and the caller should not
    report such a decode as a blind prediction.

    Raises on a length mismatch instead of falling back silently: a cloud
    whose two per-pair tensors disagree is a loader bug, and swallowing it
    is how this slot stayed dead for six days.
    """
    bo = getattr(cloud, "bond_orders_continuous", None)
    bonds = getattr(cloud, "bonds", None)
    if bo is None or bonds is None:
        return gap
    n_edges = int(edge_index.shape[1])
    if n_edges == 0:
        return gap
    n_bonds = int(bonds.shape[1])
    if int(bo.shape[0]) != n_bonds:
        raise ValueError(
            "cloud.bond_orders_continuous must align with cloud.bonds: "
            f"got {int(bo.shape[0])} BOs for {n_bonds} bonds"
        )
    bo = bo.to(dtype=torch.float32)
    lookup = {}
    for k in range(n_bonds):
        i = int(bonds[0, k].item())
        j = int(bonds[1, k].item())
        value = float(bo[k].item())
        lookup[(i, j)] = value
        lookup[(j, i)] = value
    out = gap.clone()
    for k in range(n_edges):
        i = int(edge_index[0, k].item())
        j = int(edge_index[1, k].item())
        value = lookup.get((i, j))
        if value is not None:
            out[k] = value
    return out


def _bo_truth_available(pf: "PairFeature", cloud: "AtomCloud") -> bool:
    """Whether the bo_target column CAN be teacher-forced for this decode.

    Conservative on purpose: True when an explicit
    ``PairFeature.bond_orders_continuous`` is present, or when the cloud
    carries both ``bonds`` and ``bond_orders_continuous``.  It does not
    check whether any candidate edge actually matched a truth bond —
    "truth was available" is the property a consumer needs in order to
    refuse to call the decode blind.
    """
    if getattr(pf, "bond_orders_continuous", None) is not None:
        return True
    return (
        getattr(cloud, "bond_orders_continuous", None) is not None
        and getattr(cloud, "bonds", None) is not None
    )


def _angle_to_metal(
    cloud: "AtomCloud",
    atom_i: int,
    atom_j: int,
    metal_idx: int,
    centroid: Optional[torch.Tensor],
) -> float:
    """Compute the angular deviation of bond ``(i, j)`` from the metal normal.

    Wf-A11 (2026-10-13) — replaces the hardcoded 0.0 that
    :meth:`BondAwareDecoder._infer_pair_features` and
    :meth:`ConnectivityAwareDecoder._build_pair_features_from_edges` had been
    emitting since the head shipped. The metal normal is taken as the
    (centroid − metal) direction; for a square-planar Pt centre that gives
    ~0 rad for any of the four co-planar donors. For a cloud without
    ``metal_atom_index`` set, returns 0.0 (the prior behaviour).

    Closes D5 (1-of-9 dead feature slot in shipped weights) and #12
    honest-negative (bond-length prior / angle_to_metal).
    """
    import math as _math

    if metal_idx < 0:
        return 0.0
    n_atoms = int(cloud.positions.shape[0])
    if metal_idx >= n_atoms or atom_i >= n_atoms or atom_j >= n_atoms:
        return 0.0
    p_metal = cloud.positions[metal_idx]
    diff = cloud.positions[atom_j] - p_metal
    dist = float(torch.norm(diff).item())
    if dist < 1e-9:
        return 0.0
    # Reference axis = mean of all donor vectors (atom - metal). When the
    # metal sits at the centroid (square-planar Pt with 4 coplanar donors
    # in xy-plane, Pt at origin) this vector is zero, so fall back to +z
    # which is "out of the molecular plane" for any reasonable SP-4 / Td
    # geometry. Returns a non-zero angle for all non-degenerate bonds.
    donor_mask = torch.ones(n_atoms, dtype=torch.bool)
    donor_mask[metal_idx] = False
    if int(donor_mask.sum().item()) == 0:
        ref = torch.tensor([0.0, 0.0, 1.0])
    else:
        donor_vecs = cloud.positions[donor_mask] - p_metal
        ref = donor_vecs.mean(dim=0)
        ref_norm = float(torch.norm(ref).item())
        if ref_norm < 1e-9:
            ref = torch.tensor([0.0, 0.0, 1.0])
    cos = float(torch.dot(diff, ref).item()) / (
        dist * float(torch.norm(ref).item())
    )
    cos = max(-1.0, min(1.0, cos))
    return float(_math.acos(cos))


def make_synthetic_training_set(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X, y)`` tensors of synthetic per-pair bond features.

    X: shape ``(N, PAIR_FEATURE_DIM)`` — the columns are
    :data:`PAIR_FEATURE_NAMES`, the same table
    :meth:`BondAwareDecoder._featurise` stacks at inference time.  Both
    producers build their vector by name, so the synthetic width cannot
    drift away from ``BondOrderHead.in_dim`` (the default is
    ``PAIR_FEATURE_DIM``) without raising here.
    y: shape ``(N,)`` — long labels over ``BOND_LABELS``.
    """
    rng = torch.Generator().manual_seed(seed)

    # Z-buckets — keep order stable across code paths.
    # 0:C, 1:N, 2:O, 3:S/P, 4:metal, 5:other
    def _bucket(z: int) -> int:
        if z == 6:
            return 0
        if z == 7:
            return 1
        if z == 8:
            return 2
        if z in (15, 16):
            return 3
        if _is_metal(z):
            return 4
        return 5

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    # Helper that adds a single (z_i, z_j, order) pattern with a little
    # Gaussian noise so the head has to generalise.
    def _add(z_i: int, z_j: int, order: int, n: int, dative: bool = False) -> None:
        for _ in range(n):
            d = _bond_distance(z_i, z_j, order)
            d = max(0.6, d + torch.randn((), generator=rng).item() * 0.04)
            angle_dev = 0.0
            if dative:
                angle_dev = abs(torch.randn((), generator=rng).item()) * 0.05
            # Distance gap to single-bond anchor (the head should learn:
            # gap~0 -> single; gap<0 -> double/triple/aromatic;
            # gap>0.4 -> no bond).  Synthetic rows have no metal position,
            # so d_to_metal is 0.
            anchor_single = _bond_distance(z_i, z_j, BOND_SINGLE)
            gap = float(d - anchor_single)
            # Wf-A86 — the live bo_target column is teacher-forced: a cloud
            # with bond truth carries the loader's Wiberg BO, a CFM-generated
            # cloud carries the gap.  Train on BOTH regimes (one coin flip
            # per row) so neither path sees an out-of-distribution column.
            if torch.rand((), generator=rng).item() < _BO_TEACHER_FORCE_P:
                bo_target = _WIBERG_BO_BY_ORDER[order]
            else:
                bo_target = gap
            X_rows.append(
                _pair_feature_vector(
                    {
                        "distance": d,
                        "bucket_i": float(_bucket(z_i)),
                        "bucket_j": float(_bucket(z_j)),
                        "angle_to_metal": float(angle_dev),
                        "is_dative": 1.0 if dative else 0.0,
                        "z_i": float(min(z_i, MAX_ATOMIC_NUMBER)) / MAX_ATOMIC_NUMBER,
                        "z_j": float(min(z_j, MAX_ATOMIC_NUMBER)) / MAX_ATOMIC_NUMBER,
                        "bucket_prod": float(_bucket(z_i) * 6 + _bucket(z_j)),
                        "gap": gap,
                        "d_to_metal": 0.0,
                        "bo_target": bo_target,
                    }
                )
            )
            y_rows.append(order)

    # ---- Covalent patterns (tmQM-flavoured) ----
    _add(6, 6, BOND_SINGLE, 80)
    _add(6, 6, BOND_AROMATIC, 80)
    _add(6, 6, BOND_DOUBLE, 40)
    _add(6, 6, BOND_TRIPLE, 30)
    _add(6, 7, BOND_SINGLE, 80)
    _add(6, 7, BOND_DOUBLE, 30)
    _add(6, 8, BOND_DOUBLE, 80)
    _add(8, 1, BOND_SINGLE, 40)
    _add(7, 1, BOND_SINGLE, 40)
    _add(6, 8, BOND_SINGLE, 40)
    _add(7, 7, BOND_DOUBLE, 40)
    _add(6, 7, BOND_TRIPLE, 30)
    _add(7, 6, BOND_TRIPLE, 30)

    # ---- No-bond "rejection" patterns (long distances) ----
    # Real non-bonded pairs are at >2.5 Å; the head must learn that.
    def _add_far(z_i: int, z_j: int, n: int) -> None:
        for _ in range(n):
            d = 2.8 + abs(torch.randn((), generator=rng).item()) * 0.5
            anchor_single = _bond_distance(z_i, z_j, BOND_SINGLE)
            gap = float(d - anchor_single)
            # Wf-A86 — same two-regime mix as _add; a no-bond pair's truth BO
            # is 0.0, which is a different (and informative) value from a
            # far-pair gap of +0.5..+1.5.
            if torch.rand((), generator=rng).item() < _BO_TEACHER_FORCE_P:
                bo_target = _WIBERG_BO_BY_ORDER[BOND_NO_BOND]
            else:
                bo_target = gap
            X_rows.append(
                _pair_feature_vector(
                    {
                        "distance": d,
                        "bucket_i": float(_bucket(z_i)),
                        "bucket_j": float(_bucket(z_j)),
                        "angle_to_metal": 0.0,
                        "is_dative": 0.0,
                        "z_i": float(min(z_i, MAX_ATOMIC_NUMBER)) / MAX_ATOMIC_NUMBER,
                        "z_j": float(min(z_j, MAX_ATOMIC_NUMBER)) / MAX_ATOMIC_NUMBER,
                        "bucket_prod": float(_bucket(z_i) * 6 + _bucket(z_j)),
                        "gap": gap,
                        "d_to_metal": 0.0,
                        "bo_target": bo_target,
                    }
                )
            )
            y_rows.append(BOND_NO_BOND)

    _add_far(6, 6, 80)
    _add_far(6, 7, 40)
    _add_far(6, 8, 40)
    _add_far(7, 7, 30)
    _add_far(8, 8, 20)
    # ---- Dative patterns (ligand donor -> metal) ----
    # These are physically single-bond (the head predicts single) but
    # tagged with is_dative_candidate=1 so the decoder can flag them.
    _add(7, 78, BOND_SINGLE, 60, dative=True)  # N -> Pt(II)
    _add(7, 29, BOND_SINGLE, 50, dative=True)  # N -> Cu(I)
    _add(8, 78, BOND_SINGLE, 40, dative=True)  # O -> Pt(II)
    _add(8, 29, BOND_SINGLE, 40, dative=True)  # O -> Cu(I)

    X = torch.tensor(X_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    return X, y


# ---------------------------------------------------------------------------
# Atom cloud + pair feature containers
# ---------------------------------------------------------------------------
@dataclass
class AtomCloud:
    """A point-cloud of atoms, optionally carrying its true bond graph.

    Parameters
    ----------
    positions : (N, 3) tensor
    atomic_numbers : (N,) long tensor
    bonds : (2, E) int64 tensor, optional
        Ground-truth bond pairs, ``(i, j)`` per column, 0-based into this
        cloud's atoms.  Supplied by corpora that know the answer (tmQM ships
        Wiberg bond orders in the same atom order as its geometries).
    bond_types : (E,) long tensor, optional
        One :data:`BOND_LABELS` class per column of ``bonds``.

    Why these are optional and last
    -------------------------------
    ``AtomCloud`` is constructed in 63 places across the repo, most of which
    have no bond information to offer.  Defaulting both fields keeps every one
    of those working unchanged, while letting a corpus that DOES know the
    answer pass it through instead of forcing every consumer to re-derive it
    from a heuristic.

    Added 2026-10-04.  Before this the CFM trainer, having no bond truth to
    consume, synthesised one with a rule that labelled a pair as bonded iff one
    endpoint was Pt (Z=78) — and it was hardcoded.  On a Ru complex, 100% of
    the resulting targets were ``BOND_NO_BOND`` (MEASURED: 1331/1331), so the
    bond head learned to predict "no bond" everywhere, every decode came back
    as disconnected atoms, and ``decode_ratio`` reported 1.000 because
    ``C.C.C.C...`` is a valid SMILES.

    Per-atom / per-cloud supervision (added 2026-10-13, Wf-A11)
    -----------------------------------------------------------
    The 19 fields below are the loader-output path for signals that were
    previously dropped between the metadata CSV and the cloud object. All
    default to ``None`` so the 63 existing construction sites stay byte-
    identical; ``__post_init__`` enforces shape contracts ONLY when a field
    is supplied (never forces non-None).

    Per-atom tensor slots
        natural_charges           (N,) float32   — from tmQM_X.q
        bo_per_atom               (N,) float32   — sum of per-pair Wiberg BOs
        coord_neighbors_per_atom  (N, 12) int64  — per-atom neighbour Z, 0-pad
    Per-pair tensor slots (require bonds to be set)
        bond_orders_continuous    (E,)  float32  — float Wiberg BO (NOT int class)
        is_dative_pair            (E,)  bool     — per-pair dative flag
    Per-cloud scalar tensor slots
        metal_atom_index          () int64       — -1 if none (see below)
        oxidation_state           () int64
        spin_state                () int64
        csd_publication_year      () int64
        dft_electronic_e          () float32
        dft_dispersion_e          () float32
        dipole_m                  () float32
        metal_q                   () float32
        hl_gap                    () float32
        homo_e                    () float32
        lumo_e                    () float32
        polarizability            () float32
    Per-cloud str slots (no .to() needed)
        stoichiometry             str
        smiles                    str

    ``metal_atom_index`` defaults to ``-1`` (a class constant ``NO_METAL_INDEX``)
    because the bond head / ``compute_angle_to_metal`` callers need a sentinel
    distinct from any real atom index. Adding it as a first-class field closes
    the A13 root cause where downstream code had to re-scan ``atomic_numbers``
    for the metal Z on every call.

    ``K_NEIGHBOURS = 12`` for ``coord_neighbors_per_atom``: tmQM's largest
    coordination number in the d-block pool is 7 (Heptacoordinate Ln/Ac), 12
    leaves headroom for any future mononuclear d-block complex.
    """

    K_NEIGHBOURS: ClassVar[int] = 12
    NO_METAL_INDEX: ClassVar[int] = -1

    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    bonds: Optional[torch.Tensor] = None
    bond_types: Optional[torch.Tensor] = None
    # ---- per-atom supervision (19 new fields, Wf-A11) ----
    natural_charges: Optional[torch.Tensor] = None
    bo_per_atom: Optional[torch.Tensor] = None
    coord_neighbors_per_atom: Optional[torch.Tensor] = None
    # ---- per-pair supervision ----
    bond_orders_continuous: Optional[torch.Tensor] = None
    is_dative_pair: Optional[torch.Tensor] = None
    # ---- per-cloud scalar tensors ----
    metal_atom_index: Optional[torch.Tensor] = None
    oxidation_state: Optional[torch.Tensor] = None
    spin_state: Optional[torch.Tensor] = None
    csd_publication_year: Optional[torch.Tensor] = None
    dft_electronic_e: Optional[torch.Tensor] = None
    dft_dispersion_e: Optional[torch.Tensor] = None
    dipole_m: Optional[torch.Tensor] = None
    metal_q: Optional[torch.Tensor] = None
    hl_gap: Optional[torch.Tensor] = None
    homo_e: Optional[torch.Tensor] = None
    lumo_e: Optional[torch.Tensor] = None
    polarizability: Optional[torch.Tensor] = None
    # ---- per-cloud str slots ----
    stoichiometry: Optional[str] = None
    smiles: Optional[str] = None
    # ---- Wf-A16 (2026-10-13) c-flow pocket slot ----
    # Optional PocketCloud (coords/atom_types/mask/pocket_id).  Default
    # None preserves the 63 existing construction sites — the loader
    # attaches a PocketCloud only when ``data.load_pocket: true`` and a
    # pocket path is supplied.  ``__post_init__`` only validates when
    # the field is non-None (same contract as the other optional
    # tensors above).
    pocket: Optional[PocketCloud] = None
    # ---- Wf-A27 (2026-10-13) coord_geom_class slot ----
    # Rule-based coordination-geometry class for the dominant metal
    # site in the cloud (Pt Z=78 @ CN=4 -> 'square_planar' etc.) from
    # :mod:`molmetal.data.coord_geometry`.  Loader-attached; str
    # rather than int keeps the JSON side-channel human-readable and
    # matches the :mod:`molmetal.eval.scoring.pocket2mol_score`
    # vocabulary.  Default ``None`` so the 63 construction sites keep
    # a byte-identical signature.  Wf-A28 may add a parallel
    # ``coord_geom_oh`` (one-hot) field — kept opt-in for now.
    coord_geom_class: Optional[str] = None

    def __post_init__(self) -> None:
        if self.positions.dim() != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")
        if self.atomic_numbers.dim() != 1:
            raise ValueError("atomic_numbers must be (N,)")
        if self.positions.shape[0] != self.atomic_numbers.shape[0]:
            raise ValueError("positions and atomic_numbers must agree on N")
        if self.bonds is not None:
            if self.bonds.dim() != 2 or self.bonds.shape[0] != 2:
                raise ValueError("bonds must be (2, E)")
            if self.bond_types is None:
                raise ValueError(
                    "bond_types must be supplied alongside bonds — a pair list "
                    "without its classes cannot be used as a training target"
                )
            if int(self.bond_types.shape[0]) != int(self.bonds.shape[1]):
                raise ValueError(
                    f"bond_types has {int(self.bond_types.shape[0])} entries "
                    f"but bonds has {int(self.bonds.shape[1])} columns"
                )
        # ---- Wf-A11 shape contracts (only enforced when field is set) ----
        n_atoms = int(self.positions.shape[0])
        if self.natural_charges is not None:
            if (
                self.natural_charges.dim() != 1
                or int(self.natural_charges.shape[0]) != n_atoms
            ):
                raise ValueError(
                    f"natural_charges must be (N,) with N={n_atoms}; got "
                    f"shape {tuple(self.natural_charges.shape)}"
                )
        if self.bo_per_atom is not None:
            if self.bo_per_atom.dim() != 1 or int(self.bo_per_atom.shape[0]) != n_atoms:
                raise ValueError(
                    f"bo_per_atom must be (N,) with N={n_atoms}; got shape "
                    f"{tuple(self.bo_per_atom.shape)}"
                )
        if self.coord_neighbors_per_atom is not None:
            if (
                self.coord_neighbors_per_atom.dim() != 2
                or int(self.coord_neighbors_per_atom.shape[0]) != n_atoms
                or int(self.coord_neighbors_per_atom.shape[1]) != self.K_NEIGHBOURS
            ):
                raise ValueError(
                    f"coord_neighbors_per_atom must be (N, {self.K_NEIGHBOURS}) "
                    f"with N={n_atoms}; got shape "
                    f"{tuple(self.coord_neighbors_per_atom.shape)}"
                )
        if self.bond_orders_continuous is not None:
            if self.bonds is None:
                raise ValueError(
                    "bond_orders_continuous requires bonds to be set"
                )
            e = int(self.bonds.shape[1])
            if (
                self.bond_orders_continuous.dim() != 1
                or int(self.bond_orders_continuous.shape[0]) != e
            ):
                raise ValueError(
                    f"bond_orders_continuous must be (E,) with E={e}; got shape "
                    f"{tuple(self.bond_orders_continuous.shape)}"
                )
        if self.is_dative_pair is not None:
            if self.bonds is None:
                raise ValueError(
                    "is_dative_pair requires bonds to be set"
                )
            e = int(self.bonds.shape[1])
            if (
                self.is_dative_pair.dim() != 1
                or int(self.is_dative_pair.shape[0]) != e
            ):
                raise ValueError(
                    f"is_dative_pair must be (E,) with E={e}; got shape "
                    f"{tuple(self.is_dative_pair.shape)}"
                )

    def to(self, device) -> "AtomCloud":
        """Return a copy with every tensor moved to *device*.

        2026-10-05 — added because the absence of this method is what let a
        device mismatch hide for a whole run.  ``YuelBondDecoderHead.forward``
        fed ``cloud.positions`` straight into a ``nn.Linear``; a cloud on CPU
        against weights on CUDA raises ``RuntimeError``, which
        ``YuelBondDecoder.featurise`` catches and reports as
        ``decode_succeeded=False``.  That is indistinguishable from the model
        having produced bad chemistry, so ``gen_validity`` read 0/128 on CUDA
        and 28/72 on CPU for the SAME checkpoint (results/cfm_tmqm_all6).

        Carries ``bonds`` / ``bond_types`` too, so a moved cloud keeps its
        training truth — a move that dropped the labels would silently turn a
        supervised example into an unlabelled one.

        Wf-A11 (2026-10-13): extended to carry the 17 new tensor fields.
        A move that dropped ANY of them would silently strip supervision
        (natural_charges / bo_per_atom / coord_neighbors / bond_orders_continuous
        / is_dative_pair / metal_atom_index / oxidation_state / spin_state /
        csd_publication_year / dft_* / dipole_m / metal_q / hl_gap / homo_e /
        lumo_e / polarizability), which is exactly the failure mode the
        original two-field carry exists to prevent. The two ``str`` fields
        (stoichiometry, smiles) are device-agnostic and need no move.
        """
        return AtomCloud(
            positions=self.positions.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            bonds=(self.bonds.to(device) if self.bonds is not None else None),
            bond_types=(
                self.bond_types.to(device)
                if self.bond_types is not None
                else None
            ),
            # ---- Wf-A11 per-atom tensors ----
            natural_charges=(
                self.natural_charges.to(device)
                if self.natural_charges is not None
                else None
            ),
            bo_per_atom=(
                self.bo_per_atom.to(device) if self.bo_per_atom is not None else None
            ),
            coord_neighbors_per_atom=(
                self.coord_neighbors_per_atom.to(device)
                if self.coord_neighbors_per_atom is not None
                else None
            ),
            # ---- per-pair tensors ----
            bond_orders_continuous=(
                self.bond_orders_continuous.to(device)
                if self.bond_orders_continuous is not None
                else None
            ),
            is_dative_pair=(
                self.is_dative_pair.to(device)
                if self.is_dative_pair is not None
                else None
            ),
            # ---- per-cloud scalar tensors ----
            metal_atom_index=(
                self.metal_atom_index.to(device)
                if self.metal_atom_index is not None
                else None
            ),
            oxidation_state=(
                self.oxidation_state.to(device)
                if self.oxidation_state is not None
                else None
            ),
            spin_state=(
                self.spin_state.to(device) if self.spin_state is not None else None
            ),
            csd_publication_year=(
                self.csd_publication_year.to(device)
                if self.csd_publication_year is not None
                else None
            ),
            dft_electronic_e=(
                self.dft_electronic_e.to(device)
                if self.dft_electronic_e is not None
                else None
            ),
            dft_dispersion_e=(
                self.dft_dispersion_e.to(device)
                if self.dft_dispersion_e is not None
                else None
            ),
            dipole_m=(
                self.dipole_m.to(device) if self.dipole_m is not None else None
            ),
            metal_q=(
                self.metal_q.to(device) if self.metal_q is not None else None
            ),
            hl_gap=(self.hl_gap.to(device) if self.hl_gap is not None else None),
            homo_e=(self.homo_e.to(device) if self.homo_e is not None else None),
            lumo_e=(self.lumo_e.to(device) if self.lumo_e is not None else None),
            polarizability=(
                self.polarizability.to(device)
                if self.polarizability is not None
                else None
            ),
            # ---- str fields: no .to() needed ----
            stoichiometry=self.stoichiometry,
            smiles=self.smiles,
            # ---- Wf-A16 pocket sidecar ----
            # PocketCloud is frozen dataclass with its own float32 tensors.
            # If the caller moved the cloud to GPU, the pocket tensors must
            # follow — otherwise the c-flow path silently runs a CPU pocket
            # against CUDA atoms and the per-layer injector produces NaN.
            pocket=(
                PocketCloud(
                    coords=self.pocket.coords.to(device),
                    atom_types=self.pocket.atom_types.to(device),
                    mask=self.pocket.mask.to(device),
                    pocket_id=self.pocket.pocket_id,
                )
                if self.pocket is not None
                else None
            ),
        )

    def cuda(self, *args) -> "AtomCloud":
        """``to('cuda')`` — present so callers reach for the obvious name.

        Without it, code written against the torch idiom fails with
        ``AttributeError`` at the call site rather than doing the obvious
        thing, which is how device handling ends up reinvented (and, before
        :meth:`to` existed, forgotten) at a dozen places.

        Only the optional device INDEX is accepted, matching ``Tensor.cuda``.
        Keyword arguments are refused rather than ignored — silently dropping
        a caller's ``dtype=`` or ``non_blocking=`` is the failure mode this
        method exists to prevent.
        """
        if len(args) > 1:
            raise TypeError(
                f"AtomCloud.cuda() takes at most 1 argument "
                f"({len(args)} given); convert dtype explicitly with .to()"
            )
        return self.to(torch.device("cuda", *args) if args else torch.device("cuda"))

    def cpu(self) -> "AtomCloud":
        """``to('cpu')`` — see :meth:`cuda`."""
        return self.to(torch.device("cpu"))


@dataclass
class PairFeature:
    """Per-pair features fed to :class:`BondOrderHead`.

    All tensors are aligned: ``edge_index`` is shape ``(2, E)`` and the
    other arrays are length ``E``.
    """

    edge_index: torch.Tensor  # (2, E) int64 — (src, dst) per row
    distance: torch.Tensor  # (E,) float32
    z_i: torch.Tensor  # (E,) long
    z_j: torch.Tensor  # (E,) long
    angle_to_metal: torch.Tensor  # (E,) float32 — deviation from ideal angle (rad)
    is_dative_candidate: torch.Tensor  # (E,) bool — see module docstring
    # Wf-A86 (2026-10-19) — teacher-forced continuous bond order, aligned
    # with edge_index.  This is the slot _featurise reads for the
    # ``bo_target`` column; before this field existed the featuriser did
    # getattr(pf, "bond_orders_continuous", None) on a dataclass that never
    # had it, so the column was permanently the gap fallback.  None means
    # "derive it": from the cloud's bond truth when the cloud carries it
    # (AtomCloud.bond_orders_continuous + bonds), else the gap.  Supplying
    # a tensor makes it authoritative — that is the explicit teacher-forcing
    # path, and a caller that uses it is feeding the answer to the head.
    bond_orders_continuous: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# BondOrderHead — small per-pair MLP
# ---------------------------------------------------------------------------
class BondOrderHead(nn.Module):
    """Per-pair MLP that maps pair features to bond-order logits.

    Architecture (~120 LOC including docstrings + comments):
        Linear(in_dim, 64) -> ReLU -> Dropout
        Linear(64, 64)     -> ReLU -> Dropout
        Linear(64, NUM_BOND_CLASSES)

    The 64-hidden size was chosen because the synthetic dataset has
    < 1000 pairs per training run; larger MLPs overfit and smaller ones
    underfit the triple / dative minority classes (MEASURED on the
    synthetic training script — see report).

    The head is intentionally SE(3)-unaware — it consumes pre-computed
    distances and angle deviations, which keeps it cheap and decoupled
    from the EGNN backbone.  An equivariant variant would be a future
    extension but is out of scope for A1.

    WF-2 A5 — joint-training extensions:

    * ``atom_vocab`` (tuple[int]) — frozen set of atomic numbers the
      downstream atom head can sample.  Defaults to the round-10
      12-element donor + Pt vocabulary used by
      :class:`LipmanFlowMatchingAdapter`.  When provided, the head
      pre-computes :attr:`bond_pattern_mask` — a dict
      ``{(Z_i, Z_j, order) -> bool}`` where ``False`` flags bond
      patterns producing atoms outside the vocab.
    * ``training_mode`` (str) — ``"joint"`` (default, since 2026-10-13
      Wf-A11) or ``"frozen"`` (legacy A1 behaviour).  In ``"joint"``
      mode the head is trained end-to-end with the CFM loss; the
      :class:`LipmanFlowMatchingAdapter` optimizer pulls the head
      parameters into its AdamW param list.  In ``"frozen"`` mode the
      head is left out of the optimizer (legacy behaviour).
    * ``bond_pattern_mask`` — pre-computed
      ``(num_atoms_vocab, num_atoms_vocab, NUM_BOND_CLASSES)`` bool
      tensor (with Z=0 entries = True so they remain unconstrained)
      used by :meth:`BondAwareDecoder.decode` to zero forbidden logits.
    """

    def __init__(
        self,
        in_dim: int = PAIR_FEATURE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        num_classes: int = NUM_BOND_CLASSES,
        atom_vocab: Optional[Tuple[int, ...]] = None,
        training_mode: str = "joint",
        # ---- Wf-A11 (2026-10-13) wiring options (all backward-compat) ----
        n_layers: int = 3,
        bond_pattern_mask_in_train: bool = False,
        continuous_bo: bool = False,
        cloud_ctx_dim: int = 0,
        # ---- Wf-A16 (2026-10-13) c-flow pocket bias projector ----
        # Default ON so the head registers ``pocket_bias_proj``; the
        # forward() path is still a no-op when ``pocket_features is None``
        # (the lazy default for legacy callers).  Set False to drop the
        # projector entirely (saves ~hidden_dim^2 params) — useful only
        # for ablation studies that isolate pocket context.
        use_per_layer_pocket_injection: bool = True,
        pocket_feature_dim: int = 128,
        # ---- Wf-A68 (2026-10-17) optional residual skip ----
        # Adds a Linear(hidden_dim, hidden_dim) projection from fc1's
        # output to the pre-`out` activation.  Default OFF preserves
        # bit-exact backward compat with Wf-A67 / Wf-A11 / Wf-A15 / Wf-A27
        # checkpoints (the projection shape is identical, the skip is a
        # no-op when False).
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        if training_mode not in ("frozen", "joint"):
            raise ValueError(
                f"training_mode must be 'frozen' or 'joint', got {training_mode!r}"
            )
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_classes = num_classes
        # Default vocab matches the round-10 LipmanFlowMatchingAdapter
        # atom_vocab (H, C, N, O, F, P, S, Cl, Se, Br, I, Pt).  Kept in
        # sync here so unit tests can build a BondOrderHead without
        # importing the adapter.
        if atom_vocab is None:
            atom_vocab = (1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78)
        self.atom_vocab: Tuple[int, ...] = tuple(int(z) for z in atom_vocab)
        self.training_mode: str = training_mode
        self.bond_pattern_mask_in_train: bool = bool(bond_pattern_mask_in_train)
        # Wf-A11: 3-layer MLP (Round-10 default).  Wf-A11 grows to n_layers
        # by stacking Linear(hidden_dim, hidden_dim) blocks (fc1 is the
        # input projection; fc2..fcN are the message layers).
        #
        # Wf-A68 (2026-10-17) — optional residual skip from fc1 to the
        # pre-out activation.  Default OFF preserves bit-exact backward
        # compat with Wf-A11..Wf-A15..Wf-A27 checkpoints (the projection
        # shapes are unchanged; the skip is a no-op sum when use_skip is
        # False).  When use_skip=True, an additional Linear(hidden_dim,
        # hidden_dim) residual projection is registered so the input
        # projection's output can be added to the final hidden activation
        # — a tiny ResNet-style upgrade that lifts the head's effective
        # depth without inflating param count (one extra (hidden_dim^2 +
        # hidden_dim) matrix).
        self.use_skip: bool = bool(use_skip)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        )
        if self.use_skip:
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.skip_proj = None  # type: ignore[assignment]
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        # Wf-A71 (2026-10-18) — flip the BOND_NO_BOND bias sign.
        #
        # Pre-Wf-A71: bias[BOND_NO_BOND] = -0.5.  See
        # YuelBondDecoderHead.__init__ in
        # molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py for the
        # full rationale (the legacy -0.5 was a sign error that made
        # the head default to bonding every pair at init — verified
        # on the Wf-A70 5-epoch smoke: bond_pr_auc=0.9934 but
        # bond_precision=0.0474).
        #
        # Post-Wf-A71: +2.0.  The bias is still a learnable parameter
        # (Wf-A11 kept it that way), so this only changes the INITIAL
        # value; a checkpoint trained on either sign will continue to
        # work because the bias is overwritten by the checkpoint dict
        # on load.  ``init_legacy_no_bond_bias()`` below keeps its
        # ``-0.5`` default for callers that explicitly want the
        # legacy value.
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[BOND_NO_BOND] = +2.0
        # Wf-A11 (M-A7): optional continuous bond-order regression head.
        # Disabled by default so existing checkpoints and consumers see the
        # exact same `(E, num_classes)` logits forward.  When enabled, the
        # forward() returns `(logits, cont_bo)` (a 2-tuple instead of a
        # tensor) and consumers must adapt — typically only useful in
        # diagnostic / ablation contexts.
        self.continuous_bo: bool = bool(continuous_bo)
        if self.continuous_bo:
            self.cont_bo_head = nn.Linear(hidden_dim, 1)
        else:
            self.cont_bo_head = None
        # Wf-A11 (M-A5): optional per-cloud context projector.  When
        # `cloud_ctx_dim > 0`, the forward signature accepts an extra
        # `cloud_ctx` arg carrying per-pair features derived from per-cloud
        # chemistry (oxidation state, spin state, DFT scalars, metal index).
        # When `cloud_ctx_dim == 0` (default), the projector is the identity
        # and the original `(E, in_dim)` contract is preserved.
        self.cloud_ctx_dim: int = int(cloud_ctx_dim)
        if self.cloud_ctx_dim > 0:
            self.cloud_ctx_proj = nn.Linear(self.cloud_ctx_dim, hidden_dim)
        else:
            self.cloud_ctx_proj = None
        # Wf-A16 (2026-10-13) c-flow pocket bias projector.
        #
        # ``pocket_features`` is a (B, H_pocket) per-pocket embedding
        # produced by the trainer-side PocketEncoder (or, when no encoder
        # is wired, the lazy mean-pool produced by Batch.collate_fn).
        # When ``use_per_layer_pocket_injection=True`` (default), the
        # head registers ``pocket_bias_proj`` that maps H_pocket ->
        # hidden_dim and broadcast-adds to ``h`` after ``fc1`` — the
        # same wire-in pattern used by ``cloud_ctx_proj`` (M-A5).
        # ``pocket_feature_dim`` defaults to 128 to match
        # ``EGNNVelocityField(hidden_dim=128)`` so the same PocketEncoder
        # output feeds both velocity and bond-head without a projection
        # mismatch.  When False the projector is None and forward() skips
        # the pocket branch — bit-exact backward-compat with Wf-A15.
        self.use_per_layer_pocket_injection: bool = bool(use_per_layer_pocket_injection)
        if self.use_per_layer_pocket_injection:
            self.pocket_bias_proj = nn.Linear(int(pocket_feature_dim), hidden_dim)
        else:
            self.pocket_bias_proj = None
        # ------------------------------------------------------------------
        # WF-2 A5 — bond-pattern mask synced with atom vocab.  Forbidden
        # bond patterns are those that would yield an atomic number outside
        # the vocab (e.g. Pt=C, since C is in the vocab but Pt-C aromatic
        # patterns aren't a real chemistry).  Stored as a fixed-size bool
        # tensor of shape (max_z, max_z, NUM_BOND_CLASSES) so the decoder
        # can index directly by (z_i, z_j, order).
        # ------------------------------------------------------------------
        # We use MAX_ATOMIC_NUMBER rows/cols as the index space — this
        # keeps the tensor small (~90*90*5 = 40k booleans) and lets the
        # decoder call into it without rebuilding the dict on every pass.
        # Entries outside MAX_ATOMIC_NUMBER are clamped at lookup time.
        max_z = int(MAX_ATOMIC_NUMBER)
        # Build the mask on CPU then register as a non-persistent buffer
        # so it follows ``.to(device)`` when the head moves to GPU.
        mask_tensor = build_bond_pattern_mask(
            atom_vocab=self.atom_vocab,
            max_z=max_z,
            num_classes=num_classes,
        )
        self.register_buffer(
            "bond_pattern_mask",
            mask_tensor,
            persistent=False,
        )

    def forward(
        self,
        pair_features: torch.Tensor,
        cloud_ctx: Optional[torch.Tensor] = None,
        z_i: Optional[torch.Tensor] = None,
        z_j: Optional[torch.Tensor] = None,
        # ---- Wf-A16 (2026-10-13) c-flow pocket features ----
        # (B, H_pocket) per-pocket embedding; broadcast to (E, H) and
        # added to ``h`` after ``fc1`` when ``pocket_bias_proj`` is set.
        # Default None = skip the branch (bit-exact Wf-A15 behaviour).
        pocket_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits over ``NUM_BOND_CLASSES``.

        Parameters
        ----------
        pair_features : (E, in_dim) tensor
        cloud_ctx : (E, cloud_ctx_dim) tensor, optional
            Wf-A11 (M-A5) — per-pair context derived from per-cloud
            chemistry (oxidation / spin / DFT / metal index).  Required
            only when ``cloud_ctx_dim > 0``; ignored when the projector
            is the identity default.
        z_i, z_j : (E,) long tensors, optional
            Wf-A11 (M-A8) — atomic numbers of each pair endpoint, used to
            index :attr:`bond_pattern_mask` and zero forbidden logits
            during training when :attr:`bond_pattern_mask_in_train` is
            True.  When both are ``None`` the original 2-input contract
            is preserved and no masking is applied.

        Returns
        -------
        logits : (E, num_classes) tensor.
            When :attr:`continuous_bo` is True, returns
            ``(logits, cont_bo)`` where ``cont_bo`` is a (E,) tensor.

        Notes
        -----
        Wf-A61 (2026-10-17) L8 — empty-batch (n=0) is well-behaved:
        ``fc1(empty)`` returns ``(0, hidden_dim)``, dropout is a no-op
        on a zero-row tensor, ``out(empty)`` returns ``(0, num_classes)``
        and downstream ``argmax`` yields an empty list.  The trainer
        boundary at
        ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:3478-3479``
        (``if all_logits.shape[0] != all_targets.shape[0]: continue``)
        re-aligns when the batch collapses.  Do NOT call this head
        with an empty ``feats_tensor`` outside the trainer path — the
        boundary guard is the only thing keeping downstream CE on
        an empty mask correct.
        """
        if (
            pair_features.dim() != 2 or pair_features.shape[1] != self.in_dim
        ):
            raise ValueError(
                f"expected (E, {self.in_dim}) pair_features, "
                f"got {tuple(pair_features.shape)}"
            )
        if cloud_ctx is not None and self.cloud_ctx_proj is not None:
            if cloud_ctx.shape[-1] != self.cloud_ctx_dim:
                raise ValueError(
                    f"cloud_ctx last dim must be {self.cloud_ctx_dim}, "
                    f"got {tuple(cloud_ctx.shape)}"
                )
            # Per-pair ctx features projected to hidden_dim and broadcast-added
            # to fc1's output so every message layer sees the cloud signal.
            ctx_emb = self.cloud_ctx_proj(cloud_ctx)
        else:
            ctx_emb = 0.0
        # Wf-A16 (2026-10-13) c-flow pocket bias — broadcast (B, H) ->
        # (E, H) via per-pair expansion.  When ``pocket_bias_proj`` is
        # None (legacy path), this branch is skipped and ``h`` is
        # byte-identical to the Wf-A15 forward.
        if (
            pocket_features is not None
            and self.pocket_bias_proj is not None
        ):
            # (B, H_pocket) -> (E, H_pocket) by zero-padding to (B,
            # max_P, H_pocket) when needed.  Here we receive the
            # already-pooled (B, H_pocket) embedding from the trainer;
            # expand to (E, H_pocket) by repeating per-pair index via
            # ``pair_idx_to_batch_idx`` if supplied via the cloud's
            # pair index.  When the caller passes a single shared
            # embedding (one-pocket cohort), we simply broadcast across
            # all pairs.
            pocket_emb = self.pocket_bias_proj(pocket_features)
            # Broadcast across the E pairs: expand (B, H) -> (E, H)
            # using a uniform per-pair assignment (all pairs in cloud
            # b share the same pocket).  When the caller supplies an
            # (E, H_pocket) tensor instead of (B, H_pocket), skip
            # the broadcast.
            if pocket_emb.dim() == 2:
                e_size = int(pair_features.shape[0])
                pocket_emb = pocket_emb.expand(e_size, -1)
        else:
            pocket_emb = 0.0
        h = F.relu(self.fc1(pair_features))
        if isinstance(ctx_emb, torch.Tensor):
            h = h + ctx_emb
        # Wf-A16 (2026-10-13) c-flow pocket bias broadcast-add.
        if isinstance(pocket_emb, torch.Tensor):
            h = h + pocket_emb
        h = self.dropout(h)
        # Wf-A68 (2026-10-17) — stash pre-MLP activation for the optional
        # residual skip.  When use_skip is False (default) this is a no-op
        # and the loop produces bit-exact-identical h to the Wf-A67 head.
        _fc1_h = h if self.use_skip else None
        for fc in self.fc_layers:
            h = F.relu(fc(h))
            h = self.dropout(h)
        # Wf-A68 — apply residual skip: project the pre-MLP activation
        # through a single Linear(hidden_dim, hidden_dim) and add to the
        # final hidden activation.  Equivalent to a 1-step ResNet at the
        # head; shape is (E, hidden_dim) preserved.  When use_skip is
        # False, _fc1_h is None and this branch is skipped.
        if _fc1_h is not None and self.skip_proj is not None:
            h = h + self.skip_proj(_fc1_h)
        logits = self.out(h)
        # Wf-A11 (M-A8): apply the pre-computed bond pattern mask at TRAINING
        # time so the CE loss never sees forbidden patterns.  Previously the
        # mask was decoder-only (BondAwareDecoder.decode → apply_bond_pattern_mask)
        # which let the head drift into emitting patterns producing atoms
        # outside the atom vocab.  Refused silently when ``z_i`` / ``z_j``
        # are None (decode-only or relaxed callers).
        #
        # Wf-A61 (2026-10-17) L9 — fail fast on a trainer that opted
        # into ``bond_pattern_mask_in_train=True`` but forgot to wire
        # ``z_i`` / ``z_j``.  The previous silent-skip made the
        # vocab-discipline mask a no-op without telling anyone, which
        # was a regression vs. the A5 intent.  Decode-only callers
        # (relaxed mask) still go through the no-op branch because
        # they don't set the flag.
        if self.bond_pattern_mask_in_train and (z_i is None or z_j is None):
            raise ValueError(
                "bond_pattern_mask_in_train=True requires z_i and z_j "
                "(vocab atomic numbers of each pair endpoint); got "
                f"z_i={'None' if z_i is None else 'Tensor'} "
                f"z_j={'None' if z_j is None else 'Tensor'}. "
                "Pass the vocab atom indices explicitly, or set "
                "bond_pattern_mask_in_train=False for decode-only callers."
            )
        if (
            self.bond_pattern_mask_in_train
            and z_i is not None
            and z_j is not None
            and hasattr(self, "bond_pattern_mask")
            and self.bond_pattern_mask is not None
        ):
            mask = self.bond_pattern_mask  # (max_z, max_z, num_classes)
            if mask.device != z_i.device:
                mask = mask.to(z_i.device)
            zi = z_i.clamp(min=0, max=mask.shape[0] - 1)
            zj = z_j.clamp(min=0, max=mask.shape[1] - 1)
            per_pair_mask = mask[zi, zj]  # (E, num_classes) bool
            # Forbidden logits -> -inf; allowed logits untouched.
            logits = logits.masked_fill(~per_pair_mask, float("-inf"))
        if self.continuous_bo and self.cont_bo_head is not None:
            cont_bo = self.cont_bo_head(h.detach()).squeeze(-1)
            return logits, cont_bo
        return logits

    # ------------------------------------------------------------------
    # WF-2 A5 — joint-training API surface
    # ------------------------------------------------------------------
    def set_training_mode(self, mode: str) -> None:
        """Set :attr:`training_mode` to ``"frozen"`` or ``"joint"``.

        Convenience helper used by the adapter at :meth:`setup` time so
        callers don't poke the attribute directly.  Validates the
        string and is a no-op when the value is unchanged.
        """
        if mode not in ("frozen", "joint"):
            raise ValueError(f"training_mode must be 'frozen' or 'joint', got {mode!r}")
        self.training_mode = mode

    def init_legacy_no_bond_bias(self, value: float = -0.5) -> None:
        """Wf-A11 — restore the Round-10 hard-coded BOND_NO_BOND bias.

        After loading a Round-10 checkpoint under ``strict=False`` the
        :attr:`out.bias` initialiser wins (it is now a learned
        ``nn.Parameter``), so callers wanting the legacy hard-coded
        ``-0.5`` on the no-bond logit can call this helper to re-paint
        it explicitly.  No-op when the bias is already at ``value``.
        """
        with torch.no_grad():
            cur = float(self.out.bias[BOND_NO_BOND].item())
            if abs(cur - value) > 1e-6:
                self.out.bias[BOND_NO_BOND] = value

    def apply_bond_pattern_mask(
        self,
        logits: torch.Tensor,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """Set logits for forbidden ``(Z_i, Z_j, order)`` patterns to ``-inf``.

        ``-inf`` (not zero) is critical — the decoder uses ``argmax``,
        and a forbidden-class logit of zero would beat a legitimate
        negative logit (e.g. SINGLE = -0.31), causing the decoder to
        emit a forbidden pattern.  Using ``-inf`` guarantees the
        argmax cannot land on a forbidden slot regardless of the head's
        confidence on the legal slots.

        Operates on a new tensor (does NOT mutate ``logits`` in place).
        ``logits`` is shape ``(E, NUM_BOND_CLASSES)``; ``z_i`` / ``z_j``
        are shape ``(E,)`` long atomic numbers.  Z values >
        :attr:`MAX_ATOMIC_NUMBER` are clamped so the mask lookup stays
        in bounds.

        The mask is the pre-computed :attr:`bond_pattern_mask`; entries
        ``False`` in the mask are forbidden patterns and their logits
        are set to ``-inf``.  Allowed entries are unchanged.
        """
        if logits.dim() != 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"expected (E, {self.num_classes}) logits, got {tuple(logits.shape)}"
            )
        if z_i.shape != z_j.shape or z_i.shape[0] != logits.shape[0]:
            raise ValueError(
                f"z_i/z_j shape mismatch with logits: z_i={tuple(z_i.shape)}, "
                f"z_j={tuple(z_j.shape)}, logits={tuple(logits.shape)}"
            )
        # Clamp Z into the pre-allocated index space.
        zi = z_i.clamp(min=0, max=MAX_ATOMIC_NUMBER - 1)
        zj = z_j.clamp(min=0, max=MAX_ATOMIC_NUMBER - 1)
        # Ensure mask lives on the same device as the indexing tensors
        # (registered buffer may be on CPU under some init paths).
        if self.bond_pattern_mask.device != zi.device:
            self.bond_pattern_mask = self.bond_pattern_mask.to(zi.device)
        mask_e = self.bond_pattern_mask[zi, zj]  # (E, num_classes) bool
        # Apply mask: True = keep, False = -inf.  Use masked_fill to
        # produce a fresh tensor (callers should NOT rely on in-place).
        out = logits.masked_fill(
            ~mask_e.to(logits.device),
            float("-inf"),
        )
        return out


# ---------------------------------------------------------------------------
# Bond-pattern mask builder (WF-2 A5)
# ---------------------------------------------------------------------------
# Heuristic "forbidden bond pattern" table.
# ---------------------------------------------------------------------------
# Conservative chemistry constraints synced with the round-10 atom vocab
# (H, C, N, O, F, P, S, Cl, Se, Br, I, Pt).  Each row is
# ``(z_set, allowed_orders)`` — every pair of atoms NOT both in
# ``z_set`` is forbidden for the listed orders.
#
# Example: ``(frozenset({7, 78}), {BOND_SINGLE})`` — N-Pt pairs may be
# SINGLE (the dative bond representation).  All other orders (DOUBLE,
# TRIPLE, AROMATIC) are forbidden because they would over-valence one
# or both atoms (Pt(II) is at most 4-coordinate single-bonded; N-Pt=C
# chemistry doesn't exist in this ligand family).
#
# Conservative default: any (Z_i, Z_j) not present in
# :data:`_ALLOWED_PATTERNS` allows ALL orders.  Forbidden patterns are
# added on top — false negatives just disable a pattern that may have
# been valid (overly conservative), false positives would block valid
# chemistry (regression).  We err on the side of conservative to keep
# the WF-2 A5 mask aligned with the synthetic tmQM dataset's support.
_ALLOWED_PATTERNS: Tuple[Tuple[frozenset, frozenset], ...] = (
    # Plain organic chemistry (H, C, N, O, F, P, S, Cl, Se, Br, I).
    # All single + aromatic bonds are valid; double / triple patterns
    # restricted per atom valence rules.
    (frozenset({6, 7}), frozenset({1, 2, 3, 4})),  # C-N single/double/triple/aromatic
    (frozenset({6, 8}), frozenset({1, 2, 4})),  # C-O single/double/aromatic (no C≡O)
    (frozenset({7, 7}), frozenset({1, 2, 4})),  # N-N single/double/aromatic
    (frozenset({7, 8}), frozenset({1, 2})),  # N-O single/double (NO nitro-like)
    (frozenset({8, 8}), frozenset({1, 2})),  # O-O single/double (peroxide)
    (frozenset({6, 6}), frozenset({1, 2, 3, 4})),  # C-C all orders
    (frozenset({6, 1}), frozenset({1})),  # C-H single only
    (frozenset({7, 1}), frozenset({1})),  # N-H single only
    (frozenset({8, 1}), frozenset({1})),  # O-H single only
    (frozenset({16, 1}), frozenset({1})),  # S-H single only
    (frozenset({34, 1}), frozenset({1})),  # Se-H single only
    # Halogens — only single bonds (no double/triple halogen chemistry
    # at this vocab scale).
    (frozenset({6, 9}), frozenset({1})),
    (frozenset({6, 17}), frozenset({1})),
    (frozenset({6, 35}), frozenset({1})),
    (frozenset({6, 53}), frozenset({1})),
    # P / S / Se in carbon contexts — generous single + aromatic.
    (frozenset({6, 15}), frozenset({1, 4})),
    (frozenset({6, 16}), frozenset({1, 2, 4})),
    (frozenset({6, 34}), frozenset({1, 2, 4})),
    # Pt (Z=78) dative chemistry — single only with donor N/O/S/Se.
    (frozenset({7, 78}), frozenset({1})),
    (frozenset({8, 78}), frozenset({1})),
    (frozenset({16, 78}), frozenset({1})),
    (frozenset({34, 78}), frozenset({1})),
)


def build_bond_pattern_mask(
    atom_vocab: Tuple[int, ...],
    max_z: int = MAX_ATOMIC_NUMBER,
    num_classes: int = NUM_BOND_CLASSES,
) -> torch.Tensor:
    """Build the (max_z, max_z, num_classes) bool mask.

    ``True`` = the (Z_i, Z_j, order) pattern is ALLOWED.  ``False`` =
    FORBIDDEN (the atom head cannot predict this pattern).

    Rules:
    - ``BOND_NO_BOND`` (class 0) is ALWAYS allowed for any pair — the
      "no bond" class is the absence of a bond, so it is consistent
      with any atomic-number combination (incl. out-of-vocab).
    - For Z_i or Z_j outside :data:`atom_vocab` (and != 0 padding):
      mask out all bond *order* classes 1..num_classes-1 (NO_BOND
      remains True).
    - For in-vocab pairs: restrict to the orders in
      :data:`_ALLOWED_PATTERNS`.  When the pair is not in the table,
      all orders are allowed.

    Symmetry: mask[z_i, z_j] == mask[z_j, z_i].
    """
    vocab_set = set(int(z) for z in atom_vocab)
    mask = torch.ones(max_z, max_z, num_classes, dtype=torch.bool)
    # Z=0 is the padding slot — never restrict it (no real atom).
    for z_i in range(max_z):
        for z_j in range(max_z):
            # NO_BOND (class 0) is always allowed.
            mask[z_i, z_j, 0] = True
            if z_i == 0 or z_j == 0:
                # Padding atoms — leave everything allowed.
                continue
            if z_i not in vocab_set or z_j not in vocab_set:
                # The atom head cannot sample this atom, so any
                # *bonded* pattern is forbidden.  Mask out all orders
                # except NO_BOND (already True above).
                for order in range(1, num_classes):
                    mask[z_i, z_j, order] = False
                continue
            pair = frozenset({z_i, z_j})
            allowed = None
            for z_set, orders in _ALLOWED_PATTERNS:
                if z_set == pair:
                    allowed = orders
                    break
            if allowed is None:
                continue
            for order in range(1, num_classes):
                if order not in allowed:
                    mask[z_i, z_j, order] = False
    return mask


# ---------------------------------------------------------------------------
# BondAwareDecoder
# ---------------------------------------------------------------------------
@dataclass
class DecodedMol:
    """Result container — never raises on invalid chemistry, just records it."""

    mol: object = None  # rdkit.Chem.rdchem.Mol or None
    smiles: str = ""  # canonical SMILES ("" if mol is None)
    bond_orders: List[Tuple[int, int, int]] = field(default_factory=list)
    """``[(src_idx, dst_idx, bond_class)]`` in input atom order."""
    dative_bonds: List[Tuple[int, int]] = field(default_factory=list)
    """``[(donor_idx, metal_idx)]`` pairs flagged as dative."""
    n_atoms: int = 0
    n_bonds: int = 0
    sanitized: bool = False
    error: Optional[str] = None
    repair: str = ""
    """Non-empty when a decode-side repair produced a valid but altered
    molecule (Wf-A84: `"kekulize_demoted_all"`).  Empty on a clean decode.
    Consumers must keep treating `sanitized` as the validity flag; this
    field exists so a repaired success is not counted as a clean one."""
    teacher_forced: bool = False
    """True when the ``bo_target`` input column was populated from bond
    truth (Wf-A86/Wf-A87): either an explicit
    :attr:`PairFeature.bond_orders_continuous` or the cloud's own
    ``bonds`` + ``bond_orders_continuous``.  Such a decode saw the answer
    for its true pairs and MUST NOT be reported as a blind prediction —
    the CFM generation path has no truth, so its decodes stay False.
    """


def _safe_z(z: int) -> int:
    """Clamp Z to the rdkit-supported range."""
    if z < 1:
        return 6  # treat weird input as carbon
    if z > 90:
        return 6
    return int(z)


class BondAwareDecoder:
    """Decode an :class:`AtomCloud` + per-pair bond logits to an ``rdkit.Chem.Mol``.

    Parameters
    ----------
    bond_head : :class:`BondOrderHead`
        Trained (or untrained — falls back to argmax-over-logits)
        classifier producing bond logits.
    bond_cutoff : float
        Pairs with predicted distance above this are rejected as
        non-bonded regardless of the head's prediction.  Defaults to
        2.4 Å which captures covalent bonds up to C-C aromatic
        (~1.39 Å) plus generous slack.
    min_distance : float
        Pairs closer than this are rejected as physically impossible
        (steric clash / overlapping nuclei).  Defaults to 1.0 Å.
    use_geometry_prior : bool
        If True, when ``is_dative_candidate`` is True AND the predicted
        bond order is single, the pair is recorded in
        :attr:`DecodedMol.dative_bonds` instead of the covalent-bond
        list.  Set False to ignore the prior and decode everything as
        covalent.
    apply_bond_pattern_mask : bool
        WF-2 A5 — when ``True`` (default), the per-pair logits are
        zeroed for ``(Z_i, Z_j, order)`` patterns outside the
        :attr:`BondOrderHead.atom_vocab` (see
        :func:`build_bond_pattern_mask`).  Defaults to ``True`` so the
        decoder never proposes an out-of-vocab bond pattern.  Disable
        for the A1-style decode path.
    """

    def __init__(
        self,
        bond_head: BondOrderHead,
        bond_cutoff: float = 2.4,
        min_distance: float = 1.0,
        use_geometry_prior: bool = True,
        apply_bond_pattern_mask: bool = True,
    ) -> None:
        self.bond_head = bond_head
        self.bond_cutoff = float(bond_cutoff)
        self.min_distance = float(min_distance)
        self.use_geometry_prior = bool(use_geometry_prior)
        self.apply_bond_pattern_mask = bool(apply_bond_pattern_mask)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode(
        self,
        cloud: AtomCloud,
        pair_features: Optional[PairFeature] = None,
    ) -> DecodedMol:
        """Run the decoder.  Returns a :class:`DecodedMol`.

        Parameters
        ----------
        cloud : :class:`AtomCloud`
        pair_features : :class:`PairFeature`, optional
            If omitted, the decoder builds a fully-connected pair list
            with ``angle_to_metal = 0`` and ``is_dative_candidate``
            inferred from :data:`DEFAULT_METAL_GEOMETRY`.  When
            provided, only pairs listed there are scored.
        """
        try:
            return self._decode_impl(cloud, pair_features)
        except Exception as exc:  # noqa: BLE001 — graceful fallback contract
            return DecodedMol(error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------
    def _decode_impl(
        self,
        cloud: AtomCloud,
        pair_features: Optional[PairFeature],
    ) -> DecodedMol:
        # Validate cloud
        if cloud.positions.shape[0] == 0:
            return DecodedMol(error="empty atom cloud")
        if cloud.atomic_numbers.min().item() < 1:
            return DecodedMol(error="invalid atomic number < 1")

        # Build pair features if not supplied
        if pair_features is None:
            pair_features = self._infer_pair_features(cloud)

        if pair_features.edge_index.shape[1] == 0:
            # No pairs: return a cloud with explicit hydrogens but no bonds
            return self._assemble_mol(cloud, [], [])

        # Score each pair
        feats_tensor = self._featurise(pair_features, cloud)
        self.bond_head.eval()
        with torch.no_grad():
            logits = self.bond_head(feats_tensor)
            # WF-2 A5 — apply the bond-pattern mask (synced with atom
            # vocab) BEFORE the argmax.  Forbidden (Z_i, Z_j, order)
            # patterns get zero logits so the decoder cannot propose
            # chemistry that the downstream atom-head sampling softmax
            # would refuse to draw.
            if self.apply_bond_pattern_mask and hasattr(
                self.bond_head, "apply_bond_pattern_mask"
            ):
                logits = self.bond_head.apply_bond_pattern_mask(
                    logits,
                    pair_features.z_i,
                    pair_features.z_j,
                )
            predicted = torch.argmax(logits, dim=-1).tolist()

        bond_orders: List[Tuple[int, int, int]] = []
        dative_bonds: List[Tuple[int, int]] = []
        src_list = pair_features.edge_index[0].tolist()
        dst_list = pair_features.edge_index[1].tolist()
        is_dative = pair_features.is_dative_candidate.tolist()
        for k, (i, j, order, dative_ok) in enumerate(
            zip(src_list, dst_list, predicted, is_dative)
        ):
            if order == BOND_NO_BOND:
                continue
            d = float(pair_features.distance[k].item())
            if d > self.bond_cutoff:
                continue
            if d < self.min_distance:
                # Physically impossible — steric clash.
                continue
            if self.use_geometry_prior and dative_ok and order == BOND_SINGLE:
                # Treat as dative bond (still single order to RDKit).
                # Choose the donor as the non-metal.
                zi = int(cloud.atomic_numbers[i].item())
                zj = int(cloud.atomic_numbers[j].item())
                if _is_metal(zi) and not _is_metal(zj):
                    donor, metal = int(j), int(i)
                elif _is_metal(zj) and not _is_metal(zi):
                    donor, metal = int(i), int(j)
                else:
                    donor, metal = int(i), int(j)
                dative_bonds.append((donor, metal))
                bond_orders.append((int(i), int(j), BOND_SINGLE))
            else:
                bond_orders.append((int(i), int(j), int(order)))

        return self._assemble_mol(
            cloud,
            bond_orders,
            dative_bonds,
            teacher_forced=_bo_truth_available(pair_features, cloud),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_pair_features(self, cloud: AtomCloud) -> PairFeature:
        n = cloud.positions.shape[0]
        edges: List[Tuple[int, int]] = []
        distances: List[float] = []
        z_i_list: List[int] = []
        z_j_list: List[int] = []
        angles: List[float] = []
        dative_flags: List[bool] = []

        # Wf-A11 (2026-10-13) — resolve the metal index ONCE per cloud
        # instead of hardcoding 0.0. Falls back to -1 (no metal) when the
        # loader did not supply metal_atom_index, in which case every angle
        # is reported as 0.0 as before (preserves A1 decoder behaviour for
        # non-metallic clouds). Closes D5 (1-of-9 dead feature slot) and
        # #12 honest-negative angle_to_metal.
        metal_idx_attr = getattr(cloud, "metal_atom_index", None)
        metal_idx = int(metal_idx_attr.item()) if metal_idx_attr is not None else -1
        centroid = cloud.positions.mean(dim=0) if n > 0 else None

        for i in range(n):
            for j in range(i + 1, n):
                pi = cloud.positions[i]
                pj = cloud.positions[j]
                d = float(torch.norm(pj - pi).item())
                if d > self.bond_cutoff:
                    continue
                zi = int(cloud.atomic_numbers[i].item())
                zj = int(cloud.atomic_numbers[j].item())
                is_dative = (
                    (_is_metal(zi) and _is_donor(zj))
                    or (_is_metal(zj) and _is_donor(zi))
                ) and d < 2.5
                edges.append((i, j))
                distances.append(d)
                z_i_list.append(zi)
                z_j_list.append(zj)
                # Wf-A11 — real angle deviation from the metal normal.
                angles.append(_angle_to_metal(cloud, i, j, metal_idx, centroid))
                dative_flags.append(is_dative)
        edge_index = torch.tensor(edges, dtype=torch.long).T  # (2, E)
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return PairFeature(
            edge_index=edge_index,
            distance=torch.tensor(distances, dtype=torch.float32),
            z_i=torch.tensor(z_i_list, dtype=torch.long),
            z_j=torch.tensor(z_j_list, dtype=torch.long),
            angle_to_metal=torch.tensor(angles, dtype=torch.float32),
            is_dative_candidate=torch.tensor(dative_flags, dtype=torch.bool),
        )

    def _featurise(self, pf: PairFeature, cloud: AtomCloud) -> torch.Tensor:
        """Project :class:`PairFeature` into the ``in_dim``-D vector the
        head expects."""

        def _bucket(z: int) -> int:
            if z == 6:
                return 0
            if z == 7:
                return 1
            if z == 8:
                return 2
            if z in (15, 16):
                return 3
            if _is_metal(z):
                return 4
            return 5

        bi = torch.tensor(
            [_bucket(int(z)) for z in pf.z_i.tolist()],
            dtype=torch.float32,
        )
        bj = torch.tensor(
            [_bucket(int(z)) for z in pf.z_j.tolist()],
            dtype=torch.float32,
        )
        d = pf.distance
        angle = pf.angle_to_metal
        dative = pf.is_dative_candidate.to(torch.float32)
        zi_raw = pf.z_i.to(torch.float32) / MAX_ATOMIC_NUMBER
        zj_raw = pf.z_j.to(torch.float32) / MAX_ATOMIC_NUMBER
        bucket_prod = bi * 6.0 + bj
        # Distance gap to single-bond anchor — the head uses this to
        # distinguish no-bond (gap > 0.4) from single (gap ~ 0) and
        # from double/triple/aromatic (gap < 0).
        anchor_single = torch.tensor(
            [
                _bond_distance(
                    int(pf.z_i[k].item()), int(pf.z_j[k].item()), BOND_SINGLE
                )
                for k in range(pf.z_i.shape[0])
            ],
            dtype=torch.float32,
        )
        gap = d - anchor_single
        # Wf-A11 (2026-10-13) — two new pair features replace the dead
        # angle_to_metal=0.0 slot (D5) and add supervision on the
        # Wiberg continuous BO the loader now carries (A1 root cause was
        # the float getting stripped at tmqm_clouds.py:248-249).
        # distance_to_metal = ||midpoint - metal_pos|| in Angstroms;
        # bond_order_supervision = the loader's float Wiberg BO if the
        # pair is in the loader's per-pair list, else 0 (gap).
        metal_idx_attr = getattr(cloud, "metal_atom_index", None)
        metal_idx = (
            int(metal_idx_attr.item())
            if metal_idx_attr is not None
            else AtomCloud.NO_METAL_INDEX
        )
        n_atoms = int(cloud.positions.shape[0])
        if metal_idx >= 0 and metal_idx < n_atoms:
            p_metal = cloud.positions[metal_idx]
            src = pf.edge_index[0].tolist() if int(pf.edge_index.shape[1]) > 0 else []
            dst = pf.edge_index[1].tolist() if int(pf.edge_index.shape[1]) > 0 else []
            if src:
                mids = 0.5 * (
                    cloud.positions[torch.tensor(src, dtype=torch.long)]
                    + cloud.positions[torch.tensor(dst, dtype=torch.long)]
                )
                d_to_metal = torch.norm(mids - p_metal.unsqueeze(0), dim=-1)
            else:
                d_to_metal = torch.zeros((0,), dtype=torch.float32)
        else:
            d_to_metal = torch.zeros_like(d)
        # Wf-A86 — three-level precedence, most specific first:
        #   1. an explicit PairFeature.bond_orders_continuous (teacher forcing
        #      requested by the caller; authoritative),
        #   2. the cloud's own bond truth, mapped onto these candidate edges,
        #   3. the gap fallback (the generation path: a CFM-sampled cloud
        #      carries positions and Z only).
        # Before this, level 1 read a field PairFeature did not have, so the
        # column was always level 3 — a dead slot since Wf-A11.
        bo_target = getattr(pf, "bond_orders_continuous", None)
        if bo_target is not None:
            bo_target = bo_target.to(dtype=torch.float32)
            if tuple(bo_target.shape) != tuple(d.shape):
                raise ValueError(
                    "PairFeature.bond_orders_continuous must align with the "
                    f"edge list: got {tuple(bo_target.shape)}, expected {tuple(d.shape)}"
                )
        else:
            bo_target = _bo_target_for_edges(cloud, pf.edge_index, gap)
        # Ordered by name against PAIR_FEATURE_NAMES, the same table the
        # synthetic generator uses.  Adding a feature is one entry there plus
        # one entry here; a forgotten one raises instead of shifting every
        # column silently.
        return torch.stack(
            _pair_feature_vector(
                {
                    "distance": d,
                    "bucket_i": bi,
                    "bucket_j": bj,
                    "angle_to_metal": angle,
                    "is_dative": dative,
                    "z_i": zi_raw,
                    "z_j": zj_raw,
                    "bucket_prod": bucket_prod,
                    "gap": gap,
                    "d_to_metal": d_to_metal,
                    "bo_target": bo_target,
                }
            ),
            dim=-1,
        )

    def _assemble_mol(
        self,
        cloud: AtomCloud,
        bond_orders: Sequence[Tuple[int, int, int]],
        dative_bonds: Sequence[Tuple[int, int]],
        teacher_forced: bool = False,
    ) -> DecodedMol:
        # Late import keeps the module importable without rdkit installed.
        from rdkit import Chem
        from rdkit.Chem import rdChemReactions  # noqa: F401  (kept for future use)

        rw = Chem.RWMol()
        for z in cloud.atomic_numbers.tolist():
            atom_z = _safe_z(int(z))
            rw.AddAtom(Chem.Atom(atom_z))

        bond_type_map = {
            BOND_SINGLE: Chem.BondType.SINGLE,
            BOND_DOUBLE: Chem.BondType.DOUBLE,
            BOND_TRIPLE: Chem.BondType.TRIPLE,
            BOND_AROMATIC: Chem.BondType.AROMATIC,
        }
        for i, j, order in bond_orders:
            bt = bond_type_map.get(int(order), Chem.BondType.SINGLE)
            try:
                rw.AddBond(int(i), int(j), bt)
            except Exception:
                # RDKit rejected the bond (e.g. valence overflow) — skip.
                continue

        try:
            mol = rw.GetMol()
        except Exception as exc:
            return DecodedMol(
                error=f"RWMol.GetMol failed: {exc}",
                bond_orders=list(bond_orders),
                dative_bonds=list(dative_bonds),
                n_atoms=cloud.positions.shape[0],
            )

        # Sanitise; if it fails, mark and return what we have.
        sanitized = False
        sanitize_error: Optional[str] = None
        try:
            Chem.SanitizeMol(mol)
            sanitized = True
        except Exception as exc:
            sanitized = False
            sanitize_error = f"{type(exc).__name__}: {exc}"

        # Compute canonical SMILES (even unsanitized, for diagnostics).
        smiles = ""
        try:
            if sanitized:
                smiles = Chem.MolToSmiles(mol)
            else:
                # Try a partial sanitisation — keep unpaired electrons off.
                smiles = Chem.MolToSmiles(mol, sanitize=False)
        except Exception:
            smiles = ""

        return DecodedMol(
            mol=mol,
            smiles=smiles,
            bond_orders=list(bond_orders),
            dative_bonds=list(dative_bonds),
            n_atoms=cloud.positions.shape[0],
            n_bonds=len(bond_orders),
            sanitized=sanitized,
            error=sanitize_error,
            teacher_forced=bool(teacher_forced),
        )


# ---------------------------------------------------------------------------
# WF-2 A6 — connectivity-prior decoder (DropEdge + Gumbel-top-k fallback)
# ---------------------------------------------------------------------------
class ConnectivityAwareDecoder:
    """Decoder variant that gates edge candidates through a connectivity prior.

    The :class:`BondAwareDecoder` in A1 produces per-pair bond *order*
    logits but assumes *all* candidate pairs (i < j, distance ≤
    ``bond_cutoff``) get scored.  When the upstream flow-matching
    velocity field produces unphysical atom clouds (e.g. during
    pocket-conditioned CFM with a frozen tmQM sidecar that does not
    generalise), the candidate list explodes to hundreds of pairs and
    the decoder has to throw most of them away.

    The A6 fallback is to insert a *connectivity prior* between
    candidate-pair generation and bond-order scoring:

    1. Build the candidate pair list (same as :class:`BondAwareDecoder`).
    2. Apply :class:`DropEdge` — randomly zero out edges with
       probability ``drop_p`` during training (regularisation).
    3. Apply :class:`GumbelConnectivity` — produce per-edge soft
       scores during training (Gumbel-softmax with temperature
       annealing) and hard top-k during inference.
    4. Run the surviving edges through :class:`BondOrderHead` for
       bond-order classification (same head, same mask, same decoder
       back-end).

    Honest framing
    --------------
    The connectivity prior was MEASURED to keep throughput within
    ~10 % of the legacy decoder on a 100-edge cloud (see
    :file:`molmetal/reports/wf2_a6_gumbel_fallback.md`).  End-to-end
    pose-buster validation is left for WF-3 (round-12 pilot).

    Parameters
    ----------
    bond_head : :class:`BondOrderHead`
        Pre-trained (or untrained — falls back to argmax) classifier.
    connectivity : :class:`molmetal.models.connectivity_gumbel.GumbelConnectivity`
        Connectivity prior.  Defaults to a fresh
        :class:`GumbelConnectivity` with ``expected_bonds_per_atom=3.0``.
    drop_edge : :class:`molmetal.models.connectivity_gumbel.DropEdge`
        Edge-dropout regulariser.  Defaults to ``DropEdge(p=0.1)``.
    bond_cutoff : float
        Forwarded to :meth:`BondAwareDecoder._infer_pair_features`.
    min_distance : float
        Forwarded to :meth:`BondAwareDecoder._infer_pair_features`.
    use_geometry_prior : bool
        Forwarded to :class:`BondAwareDecoder`.
    apply_bond_pattern_mask : bool
        Forwarded to :class:`BondAwareDecoder`.
    """

    def __init__(
        self,
        bond_head: BondOrderHead,
        connectivity: Optional["object"] = None,
        drop_edge: Optional["object"] = None,
        bond_cutoff: float = 2.4,
        min_distance: float = 1.0,
        use_geometry_prior: bool = True,
        apply_bond_pattern_mask: bool = True,
    ) -> None:
        # Late import — avoids a circular dependency when
        # connectivity_gumbel imports bond_head helpers (it doesn't
        # today, but this keeps the door open for shared featurisers).
        from molmetal.models.connectivity_gumbel import (
            DropEdge,
            GumbelConnectivity,
        )

        self.bond_head = bond_head
        if connectivity is None:
            connectivity = GumbelConnectivity(
                in_dim=bond_head.in_dim,
                expected_bonds_per_atom=3.0,
            )
        if drop_edge is None:
            drop_edge = DropEdge(p=0.1)
        self.connectivity = connectivity
        self.drop_edge = drop_edge
        self.bond_cutoff = float(bond_cutoff)
        self.min_distance = float(min_distance)
        self.use_geometry_prior = bool(use_geometry_prior)
        self.apply_bond_pattern_mask = bool(apply_bond_pattern_mask)
        # Compose the inner :class:`BondAwareDecoder` so we reuse the
        # bond-order scoring + RDKit assembly code path verbatim.  The
        # only divergence is that we feed it a *filtered* pair list
        # (DropEdge + Gumbel-top-k).
        self._inner = BondAwareDecoder(
            bond_head=bond_head,
            bond_cutoff=bond_cutoff,
            min_distance=min_distance,
            use_geometry_prior=use_geometry_prior,
            apply_bond_pattern_mask=apply_bond_pattern_mask,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode(self, cloud: AtomCloud) -> DecodedMol:
        """Run the full decode pipeline.

        Equivalent to ``BondAwareDecoder.decode`` but with a
        connectivity prior sitting between candidate generation and
        bond-order classification.

        Parameters
        ----------
        cloud : :class:`AtomCloud`
        """
        return self.decode_gumbel(cloud, training=self.connectivity.training)

    def decode_gumbel(
        self,
        cloud: AtomCloud,
        edge_index: Optional[torch.Tensor] = None,
        training: Optional[bool] = None,
    ) -> DecodedMol:
        """Decode with DropEdge + Gumbel-top-k connectivity prior.

        Parameters
        ----------
        cloud : :class:`AtomCloud`
        edge_index : (2, E) optional long tensor — explicit edge list to
            score.  If ``None`` we build the fully-connected candidate
            list inside :attr:`bond_cutoff`.
        training : bool optional — override the connectivity head's
            training mode for this call only.  ``None`` keeps the
            head's current mode (default).
        """
        # 1. Build candidate pair features (same as BondAwareDecoder).
        if edge_index is None:
            pair_features = self._inner._infer_pair_features(cloud)
        else:
            # Caller-supplied edge_index — compute distances + features
            # inline (smaller surface area than PairFeature constructor).
            pair_features = self._build_pair_features_from_edges(
                cloud,
                edge_index,
            )

        if pair_features.edge_index.shape[1] == 0:
            return self._inner._assemble_mol(cloud, [], [])

        # 2. Featurise per the bond-head layout.
        feats = self._inner._featurise(pair_features, cloud)

        # 3. DropEdge — pure tensor, eval() disables dropping.
        de_training = self.connectivity.training if training is None else bool(training)
        dropped_feats, keep_mask = self.drop_edge(feats, training=de_training)
        if int(keep_mask.sum().item()) == 0:
            # DropEdge zeroed everything — keep at least one edge so the
            # decoder can still emit *something*.
            keep_mask[0] = True
            dropped_feats = feats.clone()
            keep_mask = keep_mask.to(torch.bool)
        # Filter the pair list to the surviving edges.
        surviving_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        edge_index = pair_features.edge_index[:, surviving_idx]
        distance = pair_features.distance[surviving_idx]
        z_i = pair_features.z_i[surviving_idx]
        z_j = pair_features.z_j[surviving_idx]
        angle_to_metal = pair_features.angle_to_metal[surviving_idx]
        is_dative = pair_features.is_dative_candidate[surviving_idx]
        pf_dropped = PairFeature(
            edge_index=edge_index,
            distance=distance,
            z_i=z_i,
            z_j=z_j,
            angle_to_metal=angle_to_metal,
            is_dative_candidate=is_dative,
        )
        dropped_feats = feats[surviving_idx]

        # 4. Gumbel-top-k connectivity prior.
        n_atoms = cloud.positions.shape[0]
        if de_training:
            # Training — soft Gumbel-softmax weights; the inner decoder
            # accepts *all* surviving pairs (no hard mask).
            out = self.connectivity(dropped_feats, hard=False)
            self.connectivity.step_anneal()
            inner_pf = pf_dropped
        else:
            # Inference — hard top-k.
            mask, out = self.connectivity.inference_topk(
                dropped_feats,
                n_atoms=n_atoms,
            )
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            inner_pf = PairFeature(
                edge_index=pf_dropped.edge_index[:, idx],
                distance=pf_dropped.distance[idx],
                z_i=pf_dropped.z_i[idx],
                z_j=pf_dropped.z_j[idx],
                angle_to_metal=pf_dropped.angle_to_metal[idx],
                is_dative_candidate=pf_dropped.is_dative_candidate[idx],
            )

        # 5. Run the inner decoder on the filtered pair list.
        result = self._inner.decode(cloud, inner_pf)
        # Annotate the result so callers can inspect the connectivity
        # decision without re-running the head.
        result.__dict__["connectivity_logits"] = out.logits
        result.__dict__["connectivity_soft_weights"] = out.soft_weights
        result.__dict__["connectivity_temperature"] = out.temperature
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_pair_features_from_edges(
        self,
        cloud: AtomCloud,
        edge_index: torch.Tensor,
    ) -> PairFeature:
        """Compute per-edge features from a caller-supplied edge_index.

        Mirrors :meth:`BondAwareDecoder._infer_pair_features` but
        starts from a fixed edge list rather than a dense i<j scan.
        """
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must be (2, E), got {tuple(edge_index.shape)}"
            )
        e = edge_index.shape[1]
        if e == 0:
            return PairFeature(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                distance=torch.zeros((0,), dtype=torch.float32),
                z_i=torch.zeros((0,), dtype=torch.long),
                z_j=torch.zeros((0,), dtype=torch.long),
                angle_to_metal=torch.zeros((0,), dtype=torch.float32),
                is_dative_candidate=torch.zeros((0,), dtype=torch.bool),
            )
        distances: List[float] = []
        z_i_list: List[int] = []
        z_j_list: List[int] = []
        angles: List[float] = []
        dative_flags: List[bool] = []
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        # Wf-A11 — resolve metal index ONCE per cloud; falls back to -1
        # when the loader didn't supply metal_atom_index. Closes D5
        # (1-of-9 dead feature slot in shipped weights).
        metal_idx_attr = getattr(cloud, "metal_atom_index", None)
        metal_idx = int(metal_idx_attr.item()) if metal_idx_attr is not None else -1
        n_atoms = int(cloud.positions.shape[0])
        centroid = cloud.positions.mean(dim=0) if n_atoms > 0 else None
        for i, j in zip(src, dst):
            pi = cloud.positions[int(i)]
            pj = cloud.positions[int(j)]
            d = float(torch.norm(pj - pi).item())
            distances.append(d)
            zi = int(cloud.atomic_numbers[int(i)].item())
            zj = int(cloud.atomic_numbers[int(j)].item())
            z_i_list.append(zi)
            z_j_list.append(zj)
            # Wf-A11 — real angle deviation (was hardcoded 0.0).
            angles.append(_angle_to_metal(cloud, int(i), int(j), metal_idx, centroid))
            dative_flags.append(
                (_is_metal(zi) and _is_donor(zj) and d < 2.5)
                or (_is_metal(zj) and _is_donor(zi) and d < 2.5)
            )
        return PairFeature(
            edge_index=edge_index.long(),
            distance=torch.tensor(distances, dtype=torch.float32),
            z_i=torch.tensor(z_i_list, dtype=torch.long),
            z_j=torch.tensor(z_j_list, dtype=torch.long),
            angle_to_metal=torch.tensor(angles, dtype=torch.float32),
            is_dative_candidate=torch.tensor(dative_flags, dtype=torch.bool),
        )

    def extra_repr(self) -> str:
        return (
            f"bond_cutoff={self.bond_cutoff}, min_distance={self.min_distance}, "
            f"drop_p={self.drop_edge.p}, expected_bonds_per_atom="
            f"{self.connectivity.expected_bonds_per_atom}"
        )


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------
def train_synthetic(
    head: Optional[BondOrderHead] = None,
    n_epochs: int = 600,
    lr: float = 5e-3,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[BondOrderHead, dict]:
    """Train :class:`BondOrderHead` on the synthetic tmQM-style dataset.

    Returns the trained head plus a metrics dict with keys ``train_acc``,
    ``val_acc``, ``n_train``, ``n_val``, ``loss_final``.

    Honest framing: this is a closed-form supervised pass on synthetic
    data (MEASURED on CPU).  Real tmQM-21k numbers are PROJECTED and
    not validated here.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        if head is None:
            head = BondOrderHead()
        head.to(device)

        X, y = make_synthetic_training_set(seed=seed)
        X = X.to(device)
        y = y.to(device)

        # 80/20 split
        n = X.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_train = int(0.8 * n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Class weights — boost the rare triple / aromatic / no-bond classes
        # so the head doesn't collapse to "everything is single".
        class_counts = torch.bincount(y_train, minlength=NUM_BOND_CLASSES).to(
            torch.float32
        )
        class_counts = torch.clamp(class_counts, min=1.0)
        inv = (class_counts.sum() / class_counts) ** 0.5
        inv = inv / inv.mean()
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(n_epochs))
        head.train()
        loss_final = 0.0
        for epoch in range(n_epochs):
            opt.zero_grad()
            logits = head(X_train)
            loss = F.cross_entropy(logits, y_train, weight=inv)
            loss.backward()
            opt.step()
            sched.step()
            loss_final = float(loss.item())
            if verbose and (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    pred = torch.argmax(head(X_val), dim=-1)
                    acc = float((pred == y_val).float().mean().item())
                print(f"epoch {epoch + 1:3d}: loss={loss_final:.4f} val_acc={acc:.3f}")

        head.eval()
        with torch.no_grad():
            train_pred = torch.argmax(head(X_train), dim=-1)
            val_pred = torch.argmax(head(X_val), dim=-1)
        train_acc = float((train_pred == y_train).float().mean().item())
        val_acc = float((val_pred == y_val).float().mean().item())

        metrics = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "n_train": int(n_train),
            "n_val": int(n - n_train),
            "loss_final": loss_final,
            "n_epochs": int(n_epochs),
        }
    return head, metrics


# ---------------------------------------------------------------------------
# Convenience: build a default-trained head for downstream callers.
# ---------------------------------------------------------------------------
def default_trained_head(n_epochs: int = 200, seed: int = 0) -> BondOrderHead:
    """Return a BondOrderHead trained on the synthetic dataset.

    Used by tests and by callers that want a 'works out of the box'
    decoder without managing training themselves.
    """
    head, _metrics = train_synthetic(n_epochs=n_epochs, seed=seed)
    return head
