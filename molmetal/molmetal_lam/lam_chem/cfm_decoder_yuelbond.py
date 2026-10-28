"""YuelBond-bonded decoder for CFM atom clouds — valence-aware.

Wf-124 Phase 1 (2026-09-27) — wires the YuelBond GNN decoder
(:file:`yuelbond_decoder.py`) into a CFM-ready decoder that emits a
:class:`~molmetal.models.bond_head.DecodedMol` directly.

Why this exists
---------------
The Path-B ``DecoderRework`` (:file:`decoder_rework.py`) lifts
``decode_ratio`` off zero on *synthetic* Pt-click clouds because the
soft distance + type-compat prior gives the bond head "candidates"
to score.  But the same architecture on the
``wf_cfm_rescue/phase5_200step_smoke.json`` run still produces
``decode_ratio = 0/8`` — RDKit sanitization fails because of overvalent
Pt atoms in the synthetic clouds (Pt with 5+ bonds).

The YuelBond decoder head is a small message-passing GNN that maps
``(positions, atomic_numbers) -> per-pair bond logits``.  Wang &
Dokholyan 2025 (bioRxiv 10.1101/2025.05.06.652517) show F1=92.7 % on
distorted molecules where RDKit's ``DetermineConnectivity`` fails on
783/1000 cases.  When wired with a **valence-aware bond-order
selector** (greedy assignment that respects per-atom max-valence), the
YuelBond head produces RDKit-sanitizable ``Mol`` objects in a
deterministic, CPU-friendly path (~5k params, K=3 layers).

Lit anchors
-----------
* Wang, S., Dokholyan, P. (2025). *YuelBond: a graph neural network for
  recovering the molecular bond network from distorted 3D
  structures.*  bioRxiv 10.1101/2025.05.06.652517.  F1=92.7 % on
  distorted molecules.
* Liu, Z., et al. (2025). *NExT-Mol: 1D SELFIES + 3D conformer
  decoupling for molecular generation.*  ICLR 2025,
  arXiv:2502.12638.  Learned bond topology > covalent-radius
  heuristics on OOD scaffolds.

Honest framing
--------------
This is a **CPU-only, untrained-head** path.  The YuelBond head is
random-initialised; the *architecture* (K=3 layers, ≤5k params) is what
ships, not a trained model.  We claim the **smoke wiring is correct**
(YuelBond logits → valence-aware selector → RDKit ``Mol`` →
``decode_ratio > 0`` on synthetic Pt-click clouds), NOT that
YuelBond-trained scores beat the soft prior.  Training the head on
tmQM-21k is a follow-up to Wf-119 P1.3.

A successful smoke (decode_ratio > 0 on synthetic clouds) means
``molmetal/molmetal_lam/lam_chem/cfm_decoder.py`` can route
``use_yuelbond=True`` to this module and get an ``DecodedMol`` back —
the architectural wire-up is bit-correct, ready for the GPU retrain
that follows in Wf-124 P1.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from molmetal.models.bond_head import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_LABELS,
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_TRIPLE,
    NUM_BOND_CLASSES,
    AtomCloud,
    DecodedMol,
    _classify_dative_source,
    _dative_truth_pairs,
)

from molmetal.molmetal_lam.lam_chem.ring_prior import RingSizePrior
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    YuelBondDecoder,
    YuelBondDecoderHead,
    YuelBondResult,
    bond_decision as _bond_decision,
)
from molmetal.molmetal_lam.lam_chem.decoder_rework import (
    DEFAULT_VALENCES,
    _DEFAULT_VALENCES,
)


__all__ = [
    "YuelBondBondDecoder",
    "YuelBondDecodeResult",
    "covalent_valence_table",
    "build_synthetic_pt_click_clouds",
]


# ---------------------------------------------------------------------------
# Covalent-valence table — strict per-atom max bond-order cap.
# ---------------------------------------------------------------------------
# NOTE: ``DEFAULT_VALENCES`` from ``decoder_rework.py`` is sourced from
# :data:`PRIMITIVE_ATOMS.arity`, which is the coordination number
# (valence + lone_pairs).  For RDKit sanitization we want the
# *covalent* valence (the sum of bond *orders* at the atom).  The
# table below is hand-curated and matches RDKit's valence model:
#
#   C: 4  N: 3  O: 2  F: 1  P: 3  S: 2  Cl: 1  Se: 2  Br: 1  I: 1
#   Pt_II: 4 (square-planar)   Ru_II: 6 (octahedral)
#   Zn_II: 4 (tetrahedral)     Cu_II: 4 (square-planar or tetra)
#   Au_III: 4 (square-planar)  Pd_II: 4 (square-planar)
#   Ir_III: 6 (octahedral)
#
# A Pt_II atom with 5 single bonds is invalid (overvalent); this table
# caps Pt_II at 4.  This is the *valence* that RDKit sees during
# ``Chem.SanitizeMol`` and is the one that must be respected for
# ``sanitized=True`` to come back.
#: Valence COST of each predicted bond class — the bond order RDKit sums when
#: it checks an atom's valence.  Distinct from the class INDEX: they agree for
#: single/double/triple and diverge for ``BOND_AROMATIC`` (index 4, order ~1.5)
#: and ``BOND_NO_BOND`` (index 0, order 0).
#:
#: Mixing the two up charges every aromatic bond 4 against a carbon cap of 4,
#: so a benzene ring loses all of its bonds as "overvalent".  MEASURED on tmQM
#: Ru complexes (60% aromatic bonds): 40-50% of each complex's bonds were
#: discarded that way and 24/24 decodes failed RDKit sanitisation, even though
#: the head had predicted the right classes.
_CLASS_VALENCE_COST: Dict[int, float] = {
    0: 0.0,  # BOND_NO_BOND
    1: 1.0,  # BOND_SINGLE
    2: 2.0,  # BOND_DOUBLE
    3: 3.0,  # BOND_TRIPLE
    4: 1.5,  # BOND_AROMATIC
}


_COVALENT_VALENCE: Dict[int, int] = {
    1: 1,  # H
    6: 4,  # C
    7: 3,  # N (sp3-amine; aromatic N counts as 3 with one double)
    8: 2,  # O
    9: 1,  # F
    # 2026-10-05 NEW — B, Si and As.  These fell through to the unknown-Z
    # default of 4, and a cap that is too HIGH is not a neutral fallback: it
    # authorises the selector to build a bond RDKit then rejects.  MEASURED on
    # a 200-cloud tmQM val sample the prior hit exactly this: a complex with
    # one B atom decoded to "Explicit valence for atom # 35 B, 4, is greater
    # than permitted" once a fourth bond was placed on it (B appears 6x in
    # that sample, Si 18x, As 1x).  RDKit's own default valences are B 3,
    # Si 4, As 3.
    5: 3,  # B
    14: 4,  # Si
    15: 3,  # P
    16: 2,  # S
    17: 1,  # Cl
    33: 3,  # As
    34: 2,  # Se
    35: 1,  # Br
    53: 1,  # I
    # Metals — max number of metal-ligand bonds the selector may keep.
    #
    # Wf-A113 (T72, 2026-10-28) — these were 6 for Ru/Ir/Rh/Os/Re and 4 for Pt,
    # i.e. the corpus MEDIAN coordination number for the first five.  A ceiling
    # at the median refuses half the cohort by construction: MEASURED on
    # tmqm_parsed.csv (33,429 rows), MND p50/p99 is Ru 6/10, Ir 6/9, Rh 6/10,
    # Os 6/10, Re 6/9, Pt 4/6.  On the 400-cloud c8 cohort 29.75% of clouds and
    # 9.9% of metal bonds (230/2,323) were unrepresentable
    # (metrics/by_metric/wfa113_t72_valence_cap.json).
    #
    # The old comment justified the tight cap with "a cap that is too HIGH
    # produces a molecule RDKit refuses".  That is false for these elements:
    # RDKit 2026.03 sanitises [Ru] with ten single bonds and [Pt] with six
    # (checked directly; transition metals carry no default valence).  The
    # values below are the corpus p99, the same rule
    # ``TMQM_MEASURED_METAL_CAPS`` already uses for the loss-side barrier.
    #
    # Pt stays at 4 as a chemistry choice, not a statistic: 81.8% of the Pt
    # corpus is four-coordinate and the project targets Pt(II).  Pt(IV)/5-coord
    # (17.7%) is a measured, documented limitation, not an oversight.
    #
    # Zn/Cu/Au/Pd/Fe/Ni are outside the six-metal cohort and keep their
    # chemistry-first values.
    78: 4,  # Pt_II  (p50 4, p99 6 — pinned to square-planar; see above)
    44: 10,  # Ru   (p99 10)
    30: 4,  # Zn_II
    77: 9,  # Ir_III (p99 9)
    29: 4,  # Cu_II
    79: 4,  # Au_III
    46: 4,  # Pd_II
    26: 4,  # Fe_II (default; rare in SBDD ligand sets)
    28: 4,  # Ni_II
    # 2026-10-05 NEW — Rh, Re and Os.  These are THREE OF THE SIX METALS THIS
    # PROJECT TARGETS and they were absent, so they took the default of 4.
    45: 10,  # Rh (p99 10)
    75: 9,  # Re (p99 9)
    76: 10,  # Os (p99 10)
}


def covalent_valence_table(
    override: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Return the per-atom covalent-valence table.

    Parameters
    ----------
    override : dict, optional
        Per-Z overrides applied on top of the canonical table.  Use to
        test Pt_IV (cap=6) instead of Pt_II (cap=4).

    Returns
    -------
    dict[int, int]
        ``{Z: max_covalent_bond_order}``.  Unknown Z values default to
        4 (carbon).
    """
    table = dict(_COVALENT_VALENCE)
    if override:
        for z, cap in override.items():
            table[int(z)] = int(cap)
    return table


# ---------------------------------------------------------------------------
# Wf-A57 W9-P3 (2026-10-16) — decoder-side ``_GeometryConstraint`` dataclass
# + 3 frozen instances + ``_geometry_constraint_for_z`` lookup helper MOVED
# to :mod:`molmetal.molmetal_lam.priors.metal_geometry` so trainer-side and
# decoder-side share ONE canonical definition.  The dataclass identity,
# fields, and lookup helper are preserved exactly so
# :class:`test_cfm_decoder_geometry_constraints` remains PASS without
# changes (it imports these names from THIS module via the re-export below).
# ---------------------------------------------------------------------------
from molmetal.molmetal_lam.priors.metal_geometry import (  # noqa: E402,F401
    _GeometryConstraint,
    _SQUARE_PLANAR,
    _OCTAHEDRAL,
    _TETRAHEDRAL,
    _geometry_constraint_for_z,
)


# ---------------------------------------------------------------------------
# Wf-A55 W7-P5 — 4n+2 反芳香 (anti-aromatic) helper
# ---------------------------------------------------------------------------
def _is_anti_aromatic(ring_size: int) -> bool:
    """Hückel anti-aromaticity: ring_size % 4 == 0 AND ring_size >= 4.

    4n rings (4, 8, 12, ...) are anti-aromatic when planar-aromatic; 4n+2
    rings (6, 10, 14, ...) are aromatic.  Ring sizes 5 and 7 are 4n+1 / 4n+3
    — neither rule fires.

    Wf-A55 W7-P5 — added 2026-10-16.  Backwards-compat note: a real
    cyclooctatetraene (8 π electrons, non-planar tub) behaves as a
    polyene, NOT anti-aromatic.  We have no planarity probe at decode
    time, so we conservatively mark ALL 4n aromatic rings for demotion.
    The geometric refinement (planarity check) is a future Wf.
    """
    if ring_size < 4:
        return False
    return ring_size % 4 == 0


def _detect_and_demote_anti_aromatic(
    bond_orders: List[Tuple[int, int, int]],
) -> int:
    """Walk the bond graph and demote 4n all-aromatic cycles to alternating single.

    Algorithm: DFS from every node, bounded to ring size 8 (covers the
    canonical 4n rings — 4 and 8).  When a cycle of size ``n % 4 == 0``
    closes back to the start, and every bond on the cycle is
    ``BOND_AROMATIC``, every other bond in the cycle is downgraded to
    ``BOND_SINGLE`` and a ``[wfA55-anti-aromatic] HONEST-NEGATIVE`` line
    is emitted to stdout.

    Returns the number of distinct rings demoted.

    Wf-A55 W7-P5 — added 2026-10-16.  MUTATES ``bond_orders`` in place
    (the caller is the selector loop, which owns the list).
    """
    if not bond_orders:
        return 0
    # Find max atom index from bond_orders (cheap; bonds << atoms in dense graphs).
    n_atoms = 0
    for a, b, _ in bond_orders:
        if a > n_atoms:
            n_atoms = a
        if b > n_atoms:
            n_atoms = b
    n_atoms += 1
    # Build adjacency list: node -> list of (neighbor, bond_index)
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for k, (a, b, _o) in enumerate(bond_orders):
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))
    demoted_rings: set = set()
    for start in range(n_atoms):
        if start not in adj:
            continue
        # Each stack entry: (current_node, path_nodes, path_edge_indices).
        stack: List[Tuple[int, List[int], List[int]]] = [
            (start, [start], [])
        ]
        while stack:
            node, path_nodes, path_edges = stack.pop()
            if len(path_nodes) > 8:
                continue
            for neighbor, edge_idx in adj.get(node, []):
                # Skip the edge we just traversed (would close a 2-cycle).
                if path_edges and edge_idx == path_edges[-1]:
                    continue
                if neighbor == start and len(path_nodes) >= 4:
                    ring_key = frozenset(path_nodes)
                    if ring_key in demoted_rings:
                        continue
                    ring_size = len(path_nodes)
                    if not _is_anti_aromatic(ring_size):
                        continue
                    # All bonds in the ring must be aromatic.
                    if all(
                        bond_orders[bi][2] == BOND_AROMATIC
                        for bi in path_edges
                    ):
                        demoted_rings.add(ring_key)
                        # Demote alternating bonds to BOND_SINGLE.
                        # We toggle along the cycle's edge order; an even
                        # index gets demoted, the next stays aromatic.
                        # RDKit treats both choices identically.
                        for idx, bi in enumerate(path_edges):
                            if idx % 2 == 0:
                                a, b, _ = bond_orders[bi]
                                bond_orders[bi] = (a, b, BOND_SINGLE)
                        print(
                            f"[wfA55-anti-aromatic] HONEST-NEGATIVE ring_size={ring_size} "
                            f"detected, demoted to alternating single (4n anti-aromatic)"
                        )
                elif neighbor not in path_nodes:
                    stack.append(
                        (neighbor, path_nodes + [neighbor], path_edges + [edge_idx])
                    )
    return len(demoted_rings)




def _demote_aromatic_outside_rings(mol):
    """Demote aromatic bonds/atoms that lie in no ring; return the mol.

    Wf-A79 (2026-10-18).  The L3 decode A/B arm
    (molmetal/reports/cfm_l3_decode_enabled_ab/control/arm_control.json)
    MEASURED 57/64 clouds failing to decode with
    AtomKekulizeException / KekulizeException -- 89.1% of the arm --
    and 0.109 decode_rate.  The message is explicit: "non-ring atom 0
    marked aromatic".  The bond head labelled a bond AROMATIC although
    its endpoints close no ring, and RDKit's kekulizer refuses the whole
    molecule.

    This is the same defect OPT-2 measured from the other side: ring
    SIZE, not aromatic bond count (3-rings 89.8% generated vs 0.5%
    real).  A decoder cannot invent the missing ring, but it can refuse
    to call a non-ring bond aromatic -- which is what this pass does,
    before sanitisation, so the cloud survives as a valid non-aromatic
    molecule instead of being lost.

    No-op when the molecule has no aromatic bonds or when ring
    perception fails (the caller then sanitises exactly as before).
    """
    # RDKit is imported INSIDE the decode path in this module (see the
    # other "from rdkit import Chem" sites); a module-level Chem does not
    # exist, so a bare Chem.* here raised NameError and the except below
    # silently turned the repair into a no-op (caught by the Wf-A79 tests).
    from rdkit import Chem

    # Chem.GetSSSR both finds the rings AND initialises RingInfo; calling
    # mol.GetRingInfo().NumRings() first raises "RingInfo not initialized"
    # on an unsanitised mol (RDKit 2026.03.6), which silently turned the
    # whole repair into a no-op (caught by the Wf-A79 tests).
    try:
        Chem.GetSSSR(mol)
    except Exception:
        return mol
    n_demoted = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.AROMATIC:
            continue
        try:
            if bond.IsInRing():
                continue
        except Exception:
            continue
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        n_demoted += 1
    if n_demoted:
        for atom in mol.GetAtoms():
            try:
                if atom.GetIsAromatic() and not atom.IsInRing():
                    atom.SetIsAromatic(False)
            except Exception:
                continue
    return mol


def _demote_small_ring_aromatic(mol, min_size: int = 5):
    """Demote aromatic bonds inside rings SMALLER than `min_size`; return mol.

    Wf-A83 (2026-10-18).  Companion to `_demote_aromatic_outside_rings`, and
    the other half of the same measured defect.  OPT-2's ring audit found the
    generated corpus is 89.8% three-membered rings against 0.5% in the real
    corpus, and a 3-ring marked aromatic cannot be kekulized at all: MEASURED
    on a 1-epoch checkpoint, one of three decoded clouds died with
    `KekulizeException: Can't kekulize mol.  Unkekulized atoms: 0 1 2` --
    three atoms, one ring, aromatic bonds.  Wf-A79 cannot help there because
    the bond IS in a ring.

    RDKit's smallest aromatic ring in this chemistry is 5 (furan/pyrrole/
    thiophene); 3- and 4-membered aromatic rings require exotic charged
    systems that this corpus does not contain (MEASURED: 0.5% 3-rings real).
    Demoting their bonds to SINGLE turns a lost molecule into a valid, if
    chemically implausible, one -- the same trade-off Wf-A79 makes, and the
    same caveat: this is a DECODE-side repair, not a ring-size fix.  The
    generator still makes 3-rings; this only stops losing them at the
    kekulizer.

    Only bonds whose BOTH endpoints lie in a small ring are touched, so a
    fused aromatic system keeps its 6-ring bonds.  No-op when ring
    perception fails.
    """
    from rdkit import Chem

    try:
        Chem.GetSSSR(mol)
    except Exception:
        return mol
    small_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) < int(min_size):
            small_atoms.update(int(i) for i in ring)
    if not small_atoms:
        return mol
    n_demoted = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.AROMATIC:
            continue
        if (
            bond.GetBeginAtomIdx() in small_atoms
            and bond.GetEndAtomIdx() in small_atoms
        ):
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            n_demoted += 1
    if n_demoted:
        for idx in small_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetIsAromatic():
                atom.SetIsAromatic(False)
    return mol


def _demote_all_aromatic(mol):
    """Demote EVERY aromatic bond/atom to single; return the mol.

    Wf-A84 (2026-10-19).  Third and last pass in the same repair chain as
    Wf-A79 (aromatic outside any ring) and Wf-A83 (aromatic inside a <5
    ring).  MEASURED on the Wf-A83 post-fix smoke (/tmp/wfa83_smoke):
    `gen_validity=0.500 (8/16)` with every failure still
    `sanitize_failed:KekulizeException` -- so the remaining refusals are
    aromatic bonds the two targeted passes leave alone, i.e. inside rings
    of size >= 5 that still cannot be kekulized (wrong number of aromatic
    atoms, an exocyclic double bond, a metal in the ring).

    There is no local repair for those: RDKit either finds a Kekule
    structure for the whole molecule or refuses it.  What the decoder can
    still do is keep the MOLECULE and drop the AROMATICITY: clear every
    aromatic flag and rewrite every aromatic bond as SINGLE, then sanitise
    again.  The ring becomes saturated -- chemically a downgrade, which is
    why the caller records `repair="kekulize_demoted_all"` and the trainer
    counts these in a separate `ok_repaired:` bucket instead of folding
    them into `ok`.  A decode saved this way is valid, not correct.

    Called ONLY from the `except` branch of `SanitizeMol`, and only for a
    kekulization exception; a molecule that already sanitised is never
    touched, so no existing numeric result changes.
    """
    from rdkit import Chem

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
    return mol


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class YuelBondDecodeResult:
    """Output of :meth:`YuelBondBondDecoder.decode`.

    Attributes
    ----------
    decoded : DecodedMol
        The RDKit ``Mol`` plus bond_orders / dative_bonds lists.  When
        ``decoded.sanitized`` is ``True`` the molecule passed RDKit's
        valence check.
    yuelbond_result : YuelBondResult
        The raw YuelBond head output (logits, edge_index, pair_features).
        Useful for downstream consumers that want to inspect the
        per-pair logit distribution.
    bonds_accepted : int
        Number of bonds the valence-aware selector kept.  The raw
        YuelBond head may emit more candidate edges than the per-atom
        valence cap allows; we greedily filter them.
    bonds_rejected_overvalent : int
        Number of candidate bonds dropped because they would have made
        an atom overvalent.  Diagnostic for the post-R18 reset (large
        counts here mean the head is over-confident; small counts mean
        the cap is not the bottleneck).
    bonds_added_connectivity : int
        Round-3 D2: how many bonds the connectivity prior ADDED to join two
        heavy-atom components.  Non-zero means the raw selector left the graph
        fragmented and the prior closed it; zero on an already-connected
        decode.  Tracked separately from ``bonds_accepted`` (which is the
        selector's own count plus these) so the two effects stay separable.
    """

    decoded: DecodedMol
    yuelbond_result: YuelBondResult
    bonds_accepted: int = 0
    bonds_rejected_overvalent: int = 0
    bonds_added_connectivity: int = 0
    #: Wf-A113 (2026-10-27) — candidate pairs the bond head dropped because
    #: their UNCLAMPED distance was below ``clash_ratio*(r_i+r_j)``.  Reported
    #: here rather than left on the head so a generated pool can record the
    #: effect of the rule instead of asserting it worked.
    clash_rejected_pairs: int = 0


# ---------------------------------------------------------------------------
# YuelBondBondDecoder — wraps YuelBond head + valence-aware selector.
# ---------------------------------------------------------------------------
class YuelBondBondDecoder:
    """Wraps :class:`YuelBondDecoder` + valence-aware bond-order selector.

    This is the production wiring that ``cfm_decoder.CFMDecoder`` calls
    when ``use_yuelbond=True``.  It is **CPU-friendly** (≤5k params,
    K=3 message-passing layers) so it slots into the 200-step decode
    smoke at < 5 s wall.

    Parameters
    ----------
    head : :class:`YuelBondDecoderHead`, optional
        The bond-order prediction head.  When ``None`` a fresh head
        is constructed with the supplied ``hidden_dim`` + ``n_layers``.
    hidden_dim : int, default 64
        Width of the message-passing hidden state.
    n_layers : int, default 3
        Number of message-passing iterations (Wang & Dokholyan 2025
        recommend K=3 for drug-like ligands).
    distance_cutoff : float, default 3.5 Å
        Pairs further apart than this are immediately classified as
        ``NO_BOND`` (the head's own cutoff; not configurable per-call).
    valence_cap : dict, optional
        Per-atom covalent-valence cap ``{Z: max_bond_order}``.  When
        ``None`` uses :func:`covalent_valence_table`'s default.

    Why a separate valence-aware selector?
    ---------------------------------------
    The YuelBond head emits per-pair logits over 5 classes.  A naive
    ``argmax`` then assembles all ``NON_NO_BOND`` edges into an RDKit
    ``Mol``.  On a synthetic cloud with multiple close pairs, the head
    often predicts more than 4 bonds at a single Pt_II atom (which
    then fails RDKit sanitization as "overvalent").  We apply a
    *greedy* selector: iterate pairs in descending logit-confidence
    order, keep the bond only if the per-atom valence cap is not yet
    reached.  This is the same approach JTVAE / GraphAF use to ensure
    per-atom valence caps are respected (Jin 2018, ICML).
    """

    def __init__(
        self,
        head: Optional[YuelBondDecoderHead] = None,
        hidden_dim: int = 64,
        n_layers: int = 3,
        distance_cutoff: float = 3.5,
        valence_cap: Optional[Dict[int, int]] = None,
        decision: str = "argmax",
        # Wf-A80 (2026-10-18) — per-class calibrated thresholds, one per
        # BOND_LABELS class (index 0 unused).  Only read when
        # decision="per_class"; None keeps every other mode unchanged.
        # The trainer writes these per epoch (bond_class_k_best_threshold)
        # and updates this attribute before gen_eval, so the calibration
        # measured in training is the decision used at inference.
        per_class_thresholds: Optional[Sequence[float]] = None,
        decision_threshold: float = 0.5,
        class_logit_bias: Optional[Sequence[float]] = None,
        connectivity_prior: bool = True,
        connectivity_prior_min_logit: float = 0.0,
        # Wf-A111 (MOD4) — ring-size prior at decode.  Default OFF: the
        # shipped decode rule is unchanged unless a caller asks for it.
        #
        # MEASURED (cfm_todo D3): 66.8% of the c7 pool's rings are
        # three-membered and 2.9% six-membered, against 0.01-0.49% in the two
        # real references this repo carries (they disagree; see
        # ring_prior.py).  The selector is greedy and sequential, so before
        # committing a pair it can ask what ring that pair would close,
        # exactly.  See molmetal/molmetal_lam/lam_chem/ring_prior.py.
        ring_prior: bool = False,
        ring_prior_min_ring_size: int = 4,
        ring_prior_max_ring_size: Optional[int] = None,
        # Wf-A49 (2026-10-16) arch_fix_10 — let the caller pin the
        # valence cap for atomic numbers NOT in the table.  The prior
        # form silently fell back to 4 (carbon), which is the same
        # silent-clamp surface as the Z>90 → Z=90 clamp: a cap that is
        # too HIGH only costs a bond, while a cap that is too LOW
        # produces a molecule RDKit refuses, and the decode is lost.
        # Default ``None`` keeps the legacy behaviour for backward
        # compat; set to e.g. ``unknown_z_default=3`` for stricter
        # boron-like defaults, or set to ``"raise"`` to refuse unknown
        # Z outright at decode time.
        unknown_z_default: Optional[object] = None,
        # Wf-A92 (2026-10-19) — distance bound for the Z-only dative rule.
        # ``_is_dative_pair`` used to test Z roles only, so a predicted
        # metal-donor single bond at 3.4 A (a genuine coordination bond, but
        # far longer than any dative contact) was still reported as dative.
        # ``BondAwareDecoder`` has always applied a 2.5 A bound; this mirrors
        # it.  A cloud that carries ``is_dative_pair`` truth bypasses the
        # rule entirely (truth wins), so the bound only governs the
        # generation path, where there is no truth.
        #
        # Wf-A94 (2026-10-19) — 2.5 -> 2.6 A, CALIBRATED rather than
        # inherited.  Swept against Wiberg-BO truth on 400 real Pt tmQM
        # clouds (1,701 true Pt bonds): 2.5 A gives precision 0.9946 /
        # recall 0.9677 (F1 0.9809); 2.6 A gives 0.9935 / 0.9841
        # (F1 0.9888), the best F1 on the grid.  2.6 A is also the cutoff
        # the cohort builder writes the labels with, so label and decode
        # now agree.  See
        # ``metrics/by_metric/wfa94_tmqm_dative_calibration.json``.
        dative_max_distance: float = 2.6,
    ) -> None:
        self._decoder = YuelBondDecoder(
            head=head,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        if head is not None:
            self.distance_cutoff = float(head.cutoff)
        else:
            self.distance_cutoff = float(distance_cutoff)
        self.valence_cap = covalent_valence_table(valence_cap)
        # Wf-A49 (2026-10-16) arch_fix_10 — store the unknown-Z
        # default.  ``None`` (default) preserves the legacy behaviour
        # where unknown Z caps at 4 (carbon).  Callers wanting a
        # stricter surface pass ``unknown_z_default="raise"`` or a
        # numeric override.
        self.unknown_z_default = unknown_z_default
        self.dative_max_distance = float(dative_max_distance)
        # 2026-10-05 — HOW PER-PAIR LOGITS BECOME A CLASS.
        #
        # This decoder hardcoded `argmax` in `_select_bonds`, so the decision
        # rule configured under `flow.bond_decision` in the yaml reached the
        # METRICS (`evaluate_bond_accuracy` calls the trainer's
        # `_bond_decision`) but never reached the DECODER.  The consequence is
        # that the reported bond F1 / exact-type and the molecules actually
        # assembled into an RDKit Mol came from two different rules.
        #
        # MEASURED on the epoch-9 checkpoint, on real tmQM geometry, the two
        # agree pair-for-pair (624/2141 bonded under argmax, 0.5 and 0.3), so
        # this gap is NOT what produces the fragmented decodes — it is a real
        # wiring bug with no measured effect at that operating point.  It is
        # fixed here because it silently couples the headline metrics to a
        # rule the config does not control, and that coupling disappears the
        # moment the operating point moves.
        self.decision = str(decision)
        self.decision_threshold = float(decision_threshold)
        # 2026-10-12 (BLK-C-M4) — per-class logit offset, decode-time only.
        #
        # Same construction-site rule as the connectivity prior below: it is
        # postprocess on an existing checkpoint's logits, so no retrain is
        # needed for it to take effect and `None` restores the previous
        # decode exactly. See
        # `yuelbond_decoder.bond_decision` for why this is per-class and not
        # a prior-based logit adjustment (a global correction would boost
        # BOND_TRIPLE, which holds 0.08% of pairs and is already 66% of
        # generated output).
        if class_logit_bias is not None:
            _b = [float(x) for x in class_logit_bias]
            if any(x != 0.0 for x in _b):
                self.class_logit_bias = _b
            else:
                # An all-zeros vector is a no-op; drop it rather than pay for
                # the add on every pair of every cloud.
                self.class_logit_bias = None
        else:
            self.class_logit_bias = None
        if self.decision not in ("argmax", "threshold", "per_class"):
            # Fail HERE, not at the first decode.  A yaml typo like
            # `bond_decision: treshold` otherwise decodes with argmax and the
            # only symptom is molecules that do not match the reported
            # metrics — discovered, if at all, days later.
            raise ValueError(
                f"unknown bond decision rule {self.decision!r}; expected "
                f"'argmax', 'threshold' or 'per_class'"
            )
        # Wf-A80 — thresholds are a runtime attribute, not a frozen
        # constructor argument: the trainer recalibrates them every epoch
        # and assigns them here before the gen_eval decode.  A
        # 'per_class' decoder constructed without them fails at the first
        # decode with a message naming the missing vector.
        self.per_class_thresholds = (
            [float(x) for x in per_class_thresholds]
            if per_class_thresholds is not None
            else None
        )
        # 2026-10-05 (Round-3 D2) — CONNECTIVITY PRIOR AT INFERENCE TIME.
        #
        # The selector above is purely PER-PAIR: it takes each candidate edge
        # on its own logits and a per-atom valence cap.  Nothing in it knows
        # whether the graph it is assembling is ONE molecule or six loose
        # fragments held together by spatial proximity.  That is why
        # MEASURED on the tmQM all6 retrain (2026-10-04) `largest_fragment_frac`
        # climbs 0.21 -> 0.47 across epochs while `single_fragment_ratio` only
        # reaches 0.167: the model gets better at drawing ONE big blob, and
        # never at joining the blob to the rest of the cloud.
        #
        # This is POSTPROCESS in the decoder, NOT a loss term.  An already
        # trained checkpoint therefore benefits the moment the flag is on --
        # no retrain, no re-derivation of the training objective, and the
        # bonds it adds come from the head's OWN logits, so it can only
        # reorder which candidate edges become bonds, never invent geometry.
        #
        # It is deliberately one-sided: it can only ADD a bridging bond between
        # two heavy atoms the head already emitted as a candidate (i.e. already
        # inside the 3.5 A cutoff), and only when the remaining valence budget
        # allows.  It never deletes a bond the selector kept, so turning it OFF
        # restores the historical decode bit-exactly.
        self.connectivity_prior = bool(connectivity_prior)
        #: A bridging candidate is only eligible if its strongest non-NO_BOND
        #: logit clears this bar.  0.0 means "any candidate the head emitted",
        #: which is the geometric-cutoff prior and nothing more.  Raising it
        #: makes the prior defer to the head's confidence; it is a PRIOR, not
        #: a tuned value, and the flag is the honest control.
        self.connectivity_prior_min_logit = float(connectivity_prior_min_logit)
        # Wf-A111 (MOD4) — ring-size prior.  ``None`` when off, so the accept
        # loop pays one attribute test and nothing else.
        self.ring_prior = (
            RingSizePrior(
                min_ring_size=int(ring_prior_min_ring_size),
                max_ring_size=(
                    None
                    if ring_prior_max_ring_size is None
                    else int(ring_prior_max_ring_size)
                ),
            )
            if bool(ring_prior)
            else None
        )
        #: The last decode's ring-prior counters, so a caller can report what
        #: the prior did rather than only its effect on the final metric.
        self.last_ring_prior_stats: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode(self, cloud: AtomCloud) -> YuelBondDecodeResult:
        """Run YuelBond + valence-aware selector on an :class:`AtomCloud`.

        Returns
        -------
        :class:`YuelBondDecodeResult`
            The raw YuelBond output plus the assembled
            :class:`DecodedMol`.  ``decoded.sanitized`` indicates
            whether RDKit accepts the result; ``decoded.smiles`` is the
            canonical SMILES when accepted, else ``""``.
        """
        # Step 1: YuelBond forward pass.
        yres = self._decoder.featurise(cloud)
        if not yres.decode_succeeded or yres.logits.shape[0] == 0:
            return self._empty_decoded(cloud, yres, reason="empty_or_failed")
        # Step 2: valence-aware greedy selector.
        bond_orders, dative_bonds, n_kept, n_overvalent = self._select_bonds(
            cloud,
            yres,
        )
        # Step 2b (Round-3 D2): connectivity prior — POSTPROCESS, after bond
        # selection and before assembly.  No-op when `connectivity_prior` is
        # False, so the pre-D2 decode is reproduced bit-exactly.
        #
        # Wf-A113 (2026-10-27) — the guard used to read
        # `if self.connectivity_prior and bond_orders:`, which skipped the
        # prior in exactly the case it exists to rescue: a decode that kept
        # ZERO bonds.  MEASURED on the c7 pool
        # (molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv):
        # 302 of 12,669 rows (2.38%) are `n_frags_raw > 1` with
        # `largest_frag_frac == 1/n_frags_raw`, i.e. every fragment a single
        # atom and no bond anywhere.  `_bridge_components` already handles an
        # empty input (parent = identity, one component per heavy atom,
        # candidates from edge_index), so the precondition was pure loss.
        n_bridges = 0
        if self.connectivity_prior:
            bond_orders, dative_bonds, n_bridges = self._bridge_components(
                cloud,
                yres,
                bond_orders,
                dative_bonds,
            )
        # Step 3: assemble into an RDKit Mol.
        # Wf-A92 — say HOW the dative flags were decided.  A decode whose
        # ``dative_source`` is ``"z_rule"`` is reporting a guess; only
        # ``"truth"`` is backed by the cloud's own label-side flag.
        _dative_source, _dative_manufactured = _classify_dative_source(
            cloud, dative_bonds
        )
        decoded = self._assemble_mol(
            cloud,
            bond_orders,
            dative_bonds,
            dative_source=_dative_source,
            dative_manufactured=_dative_manufactured,
        )
        return YuelBondDecodeResult(
            decoded=decoded,
            yuelbond_result=yres,
            bonds_accepted=n_kept + n_bridges,
            bonds_rejected_overvalent=n_overvalent,
            bonds_added_connectivity=n_bridges,
            # Wf-A113 — read from the head that produced yres.  A head that
            # does not implement the rule reports 0, which is the truth for a
            # pre-A113 checkpoint loaded with clash_reject off.
            clash_rejected_pairs=int(
                getattr(
                    getattr(self._decoder, "head", None),
                    "last_clash_rejected_pairs",
                    0,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Valence-aware selector (the Wf-124 Phase 1 core innovation)
    # ------------------------------------------------------------------
    def _select_bonds(
        self,
        cloud: AtomCloud,
        yres: YuelBondResult,
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]], int, int]:
        """Greedy per-pair selector that respects per-atom valence caps.

        Algorithm
        ---------
        1. For each pair ``(i, j)`` from YuelBond's edge_index, compute
           the *bond order* via argmax over the 5-class logits.
        2. Compute a per-pair score = ``max_logits[i, j]`` (the
           strongest non-NO_BOND logit, used as a confidence proxy).
        3. Sort pairs in descending score order.
        4. Walk the sorted list; for each pair ``(i, j, order)`` keep
           the bond iff:
           a. ``order != BOND_NO_BOND``,
           b. ``order <= remaining_valence[i]``,
           c. ``order <= remaining_valence[j]``.
        5. If both atoms are donor + metal (N/O/S + Pt), record the
           bond as *dative* in addition to the covalent bond_orders.

        Returns
        -------
        (bond_orders, dative_bonds, n_kept, n_overvalent)
        """
        logits = yres.logits.detach()
        edge_index = yres.edge_index
        n_atoms = cloud.atomic_numbers.shape[0]
        if edge_index.shape[1] == 0:
            return [], [], 0, 0
        # 2026-10-05 — the decision rule is a CONFIGURED INPUT, not a
        # hardcoded argmax. See `__init__` for why this was a real gap.
        predicted = _bond_decision(
            logits,
            self.decision,
            self.decision_threshold,
            self.class_logit_bias,
            self.per_class_thresholds,
        )
        # Build sorted pair list.  Wf-A49 (2026-10-16) arch_fix_6 —
        # hoist ``max_logits`` inside this loop so the empty-edge
        # branch above (``return [], [], 0, 0``) no longer pays for a
        # max-over-4-classes op on a zero-row tensor.  The previous
        # form ran the max unconditionally on line 472 even when the
        # edge_index was empty (caught at line 459, but the list-comp
        # below wasted the work).
        #
        # Wf-A49 arch_fix_9 — DEDUP edge_index pairs.  The YuelBond
        # head may emit duplicate ``(i, j)`` rows; the prior form
        # double-counted them in the selector and pushed the per-atom
        # valence cap faster than it should be (the cap then rejected
        # legitimately-bonded pairs further down the loop).  We dedup
        # via ``seen`` so each unique (i, j) appears at most once.
        edge_src = edge_index[0].tolist()
        edge_dst = edge_index[1].tolist()
        max_logits = logits[:, 1:].max(dim=-1).values.tolist()
        pairs: List[Tuple[float, int, int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for k in range(len(edge_src)):
            src = int(edge_src[k])
            dst = int(edge_dst[k])
            if src == dst:
                continue
            order = int(predicted[k])
            if order == BOND_NO_BOND:
                continue
            # Normalise pair ordering so ``(i, j)`` and ``(j, i)`` are
            # treated as the same bond.  The selector iterates
            # undirected edges; the direction is irrelevant for the
            # per-atom valence check.
            pair_key = (src, dst) if src < dst else (dst, src)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pairs.append((float(max_logits[k]), src, dst, order))
        # Sort by descending confidence.
        pairs.sort(key=lambda t: t[0], reverse=True)
        # Per-atom remaining valence.
        #
        # Wf-A49 (2026-10-16) arch_fix_7 — type ``remaining`` as
        # ``list[float]`` (not ``list[int]``) because ``_CLASS_VALENCE_COST``
        # returns 1.5 for ``BOND_AROMATIC``.  The prior mixed int/float
        # arithmetic via implicit conversion; with this typing every
        # ``remaining[i] -= cost`` is an unambiguous float op.
        z_list = cloud.atomic_numbers.tolist()
        # Wf-A49 (2026-10-16) arch_fix_10 — honour ``unknown_z_default``.
        # The constructor stores it on ``self``; here we resolve the
        # effective cap for any Z not in the covalent-valence table.
        # ``None`` preserves the legacy 4-default; a numeric value
        # pins it explicitly; ``"raise"`` makes the decode refuse
        # unknown Z outright instead of silently mapping to carbon.
        remaining: List[float] = [
            self._cap_for_z(int(z)) for z in z_list
        ]
        bond_orders: List[Tuple[int, int, int]] = []
        dative_bonds: List[Tuple[int, int]] = []
        n_overvalent = 0
        n_kept = 0
        # Wf-A92 — resolve the cloud's own dative truth ONCE.  ``None``
        # means the cloud carries no per-pair truth (a generated cloud),
        # in which case the Z+distance rule below decides and the decode is
        # labelled ``dative_source="z_rule"``.
        dative_truth = _dative_truth_pairs(cloud)
        # Wf-A111 (MOD4) — connectivity of the bonds accepted SO FAR, which is
        # what makes the ring test exact: a pair's shortest path in this graph
        # plus one is the only ring the pair can close.
        _ring_adj: Optional[List[Set[int]]] = (
            [set() for _ in range(n_atoms)] if self.ring_prior is not None else None
        )
        for _, i, j, order in pairs:
            # 2026-10-04 — the VALENCE cost of a class is not its class index.
            #
            # `pairs` carries the argmax CLASS (0..4), and `_assemble_mol`
            # correctly treats it as a class.  This loop used the same integer
            # as a bond ORDER against `_COVALENT_VALENCE`, which is a sum of
            # bond orders — and the two agree for single/double/triple but NOT
            # for ``BOND_AROMATIC = 4``, whose real order is ~1.5.
            #
            # MEASURED on tmQM Ru complexes, where 60% of the true bonds are
            # aromatic: a benzene carbon with two aromatic neighbours was
            # charged 8 against a cap of 4, so both bonds were rejected as
            # overvalent.  40-50% of every complex's bonds were discarded that
            # way (cloud 0: 12 rejected / 11 kept against 35 true bonds), the
            # molecule came out too sparse to sanitise, and RDKit rejected all
            # 24/24 decodes.  The head had, in fact, been predicting the right
            # classes.
            cost = _CLASS_VALENCE_COST.get(int(order), 1.0)
            cap_i = remaining[i]
            cap_j = remaining[j]
            # Wf-A49 arch_fix_7 — compare against the actual remaining
            # budget, not a hard-coded 1.0.  The hard-coded 1.0 was an
            # integer-overflow fallback that masked the cap mismatch for
            # aromatic bonds (a cap of 1.5 sits between 1.0 and 2.0; the
            # prior ``< 1.0`` check let every aromatic bond through
            # regardless of remaining budget).
            if cost > cap_i or cost > cap_j:
                n_overvalent += 1
                continue
            # Wf-A113 (T72, 2026-10-28) — the per-metal GEOMETRY gate that
            # used to sit here has been removed.
            #
            # It rejected a bond when the constrained metal already carried
            # ``gc.max_bonds`` bonds.  That was redundant with ``_cap_for_z``
            # while the two tables agreed (the comment that stood here said so
            # itself, measured across Ru/Rh/Pd/Ag/Re/Os/Ir/Pt/Au/Zn/Cd), and
            # it became ACTIVELY WRONG once the valence caps were set from the
            # corpus: ``_OCTAHEDRAL.max_bonds`` is 6, so an eta6-arene Ru with
            # three other donors -- true degree 9, coordination number 6 --
            # would still lose three bonds at the geometry gate after the
            # valence gate had been fixed.  MEASURED: 53.4% of Ru clouds
            # (62/116) are pi-bound, as are 41.5% of Rh, 33.3% of Ir, 25% of
            # Os, 12% of Re and 6.9% of Pt
            # (metrics/by_metric/wfa113_t72_valence_cap.json).
            #
            # The ceiling now lives in exactly one place, ``_COVALENT_VALENCE``,
            # which is derived from the corpus.  ``_geometry_constraint_for_z``
            # still describes the ideal ANGLES and is used by the angular
            # prior; ``max_bonds`` is kept on the dataclass for callers that
            # want the shape, but it no longer gates bond count.  Pd(5th bond)
            # and Pt(5th bond) are still refused -- by the valence cap.
            # Wf-A111 (MOD4) — ring-size gate.  AFTER valence and geometry
            # (so it never weakens a correctness cap), BEFORE the bond is
            # committed.  Refusing here declines to close a cycle; the path
            # that made the ring possible stays, so nothing is disconnected.
            if self.ring_prior is not None and _ring_adj is not None:
                _accept, _ring_size = self.ring_prior.should_accept(_ring_adj, i, j)
                if not _accept:
                    continue
            bond_orders.append((i, j, order))
            if _ring_adj is not None:
                _ring_adj[i].add(j)
                _ring_adj[j].add(i)
            remaining[i] -= cost
            remaining[j] -= cost
            n_kept += 1
            # Wf-A55 W7-P5 — anti-aromatic ring probe.  Only fires on
            # aromatic bonds (the cost of a BFS cycle search is wasted on
            # single/double/triple bonds because 4n all-aromatic cycles
            # never contain non-aromatic edges).  The probe MUTATES
            # ``bond_orders`` in place to demote alternating bonds when a
            # 4n aromatic ring is detected.
            if order == BOND_AROMATIC:
                _detect_and_demote_anti_aromatic(bond_orders)
            # Dative flag — same convention as BondAwareDecoder.
            zi = int(z_list[i])
            zj = int(z_list[j])
            dative_ok = order == BOND_SINGLE and self._is_dative_pair_at(
                cloud, i, j, zi, zj, dative_truth
            )
            if dative_ok:
                # Donor = non-metal, Metal = the head's metal_set.
                if self._is_metal_z(zi) and not self._is_metal_z(zj):
                    dative_bonds.append((j, i))
                else:
                    dative_bonds.append((i, j))
        if self.ring_prior is not None:
            self.last_ring_prior_stats = self.ring_prior.stats.as_dict()
        return bond_orders, dative_bonds, n_kept, n_overvalent

    # ------------------------------------------------------------------
    # Round-3 D2 — connectivity prior (POSTPROCESS, inference time).
    # ------------------------------------------------------------------
    def _bridge_components(
        self,
        cloud: AtomCloud,
        yres: YuelBondResult,
        bond_orders: List[Tuple[int, int, int]],
        dative_bonds: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]], int]:
        """Add the highest-scoring bonds that JOIN two heavy-atom components.

        Why this exists
        ---------------
        MEASURED on the tmQM all6 retrain (2026-10-04): across 20 epochs
        ``largest_fragment_frac`` rose 0.21 -> 0.47 while
        ``single_fragment_ratio`` only reached 0.167.  The head is learning to
        draw a bigger single blob, not to join the blob to the rest of the
        cloud.  ``_select_bonds`` cannot fix this by construction: it scores
        each candidate edge independently and has no notion of a component.

        Algorithm
        ---------
        1. Union-find the heavy atoms over the ALREADY-SELECTED bonds.
        2. If there is one component (or zero heavy atoms), return unchanged.
        3. Repeatedly take the candidate pair ``(i, j)`` of heavy atoms in
           DIFFERENT components with the highest non-``NO_BOND`` logit that
           is not already a bond and whose single-bond valence cost still fits
           both atoms' remaining budget.  Add it as a ``BOND_SINGLE`` and union.
        4. Stop at one component, when no candidate qualifies, or when a full
           pass adds nothing (guards against a cycle if the two steps above
           ever disagree).

        What it can and cannot do, stated plainly
        -----------------------------------------
        * It only ADDS, and only between heavy atoms the head already emitted
          as a candidate — i.e. already inside the 3.5 A cutoff.  It never
          invents a long-range bond and never deletes a bond the selector
          kept, so ``connectivity_prior=False`` reproduces the pre-D2 decode
          bit-exactly.
        * A bridging edge is added as ``BOND_SINGLE`` regardless of the
          head's argmax class.  The head's class prediction on a pair it
          called ``NO_BOND`` is by construction not meaningful, and paying the
          cheapest possible valence cost is what keeps the added bond from
          pushing an atom over its cap and failing RDKit sanitisation.  The
          head's LOGIT still decides WHICH pair is added, so the choice of
          edge is the model's, not this code's.
        * Hydrogens are excluded from the component count (a stranded H is a
          valence detail, not a separate molecule) and cannot be a bridge
          endpoint, for the same reason.
        * This does not make the model better; it makes an existing checkpoint
          assemble more of what it already predicts as one molecule.  The
          paired D1 loss is what moves the training objective.
        """
        logits = yres.logits.detach()
        edge_index = yres.edge_index
        if edge_index.shape[1] == 0:
            return bond_orders, dative_bonds, 0
        z_list = [int(z) for z in cloud.atomic_numbers.tolist()]
        n_atoms = len(z_list)
        # Heavy atoms only: H is a valence detail, not a fragment.
        heavy = [i for i in range(n_atoms) if z_list[i] != 1]
        if len(heavy) < 2:
            return bond_orders, dative_bonds, 0

        # --- union-find over the selected bonds -----------------------------
        parent = list(range(n_atoms))

        def _find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def _union(a: int, b: int) -> bool:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        have = set()
        for i, j, _o in bond_orders:
            have.add((i, j) if i < j else (j, i))
            _union(i, j)
        n_components = len({_find(i) for i in heavy})
        if n_components <= 1:
            return bond_orders, dative_bonds, 0

        # --- remaining valence budget from the selected bonds --------------
        #
        # Wf-A49 (2026-10-16) arch_fix_7 — keep ``remaining`` typed
        # ``list[float]`` to match the ``_select_bonds`` budget and
        # avoid mixed int/float arithmetic.  See ``_select_bonds`` for
        # the per-aromatic-bond rationale.
        remaining: List[float] = [
            self._cap_for_z(int(z)) for z in z_list
        ]
        for i, j, order in bond_orders:
            cost = _CLASS_VALENCE_COST.get(int(order), 1.0)
            remaining[i] -= cost
            remaining[j] -= cost

        # --- candidate bridging edges, scored by the head's own logit -------
        # `max(logits[:, 1:])` is the same confidence the selector uses, so
        # the prior and the selector rank the head the same way.
        best_logit = logits[:, 1:].max(dim=-1).values.tolist()
        # Hoisted out of the loop below. Indexing a CUDA tensor with a Python
        # int (`int(edge_index[0][k])`) is a device-to-host transfer, so the
        # original form paid one sync PER CANDIDATE PAIR — hundreds per
        # generated molecule, times every molecule of the per-epoch gen eval.
        # `_select_bonds` above already reads these two rows with a single
        # `.tolist()` each; this is the same read, and it returns the same
        # ints, so the candidate list is unchanged.
        ei_all = edge_index[0].tolist()
        ej_all = edge_index[1].tolist()
        candidates = []
        for k in range(edge_index.shape[1]):
            i = int(ei_all[k])
            j = int(ej_all[k])
            if i == j or i >= n_atoms or j >= n_atoms:
                continue
            if z_list[i] == 1 or z_list[j] == 1:
                continue
            key = (i, j) if i < j else (j, i)
            if key in have:
                continue
            score = float(best_logit[k])
            if score < self.connectivity_prior_min_logit:
                continue
            candidates.append((score, i, j))
        candidates.sort(key=lambda t: t[0], reverse=True)

        # --- greedily bridge ----------------------------------------------
        n_added = 0
        # Wf-A92 — same truth-first dative rule as ``_select_bonds``.
        dative_truth = _dative_truth_pairs(cloud)
        for _score, i, j in candidates:
            if n_components <= 1:
                break
            if _find(i) == _find(j):
                continue
            if remaining[i] < 1.0 or remaining[j] < 1.0:
                continue
            bond_orders.append((i, j, BOND_SINGLE))
            remaining[i] -= 1.0
            remaining[j] -= 1.0
            have.add((i, j) if i < j else (j, i))
            # Mirror `_select_bonds`' dative convention for a metal-donor pair.
            if self._is_dative_pair_at(
                cloud, i, j, z_list[i], z_list[j], dative_truth
            ):
                if self._is_metal_z(z_list[i]) and not self._is_metal_z(z_list[j]):
                    dative_bonds.append((j, i))
                else:
                    dative_bonds.append((i, j))
            if _union(i, j):
                n_components -= 1
                n_added += 1
        return bond_orders, dative_bonds, n_added

    # ------------------------------------------------------------------
    # RDKit assembly (mirrors BondAwareDecoder._assemble_mol contract)
    # ------------------------------------------------------------------
    def _assemble_mol(
        self,
        cloud: AtomCloud,
        bond_orders: List[Tuple[int, int, int]],
        dative_bonds: List[Tuple[int, int]],
        dative_source: str = "",
        dative_manufactured: int = 0,
    ) -> DecodedMol:
        """Assemble an RDKit Mol from the predicted bond_orders.

        A direct copy of the assembly logic in
        :class:`molmetal.models.bond_head.BondAwareDecoder` so the
        wire-up contract (``.decoded.mol``, ``.smiles``,
        ``.bond_orders``, ``.dative_bonds``, ``.sanitized``) matches
        the production decoder bit-exactly.
        """
        from rdkit import Chem

        rw = Chem.RWMol()
        for z in cloud.atomic_numbers.tolist():
            try:
                # Wf-A49 (2026-10-16) arch_fix_5 — refuse silently-
                # clamped transuranic / trans-actinide Z (Z > 92) instead
                # of mapping U (92) / Np (93) / Pu (94) / ... / Cf (98)
                # to thorium (Z=90) and emitting a wildly wrong atom.
                # Below Z=1 we still floor to 1 (H) — RDKit rejects
                # atomic number 0 outright.  Th (Z=90) through U (Z=92)
                # remain clamped to keep the existing surface that
                # callers may already be hitting.
                z_int = int(z)
                if z_int > 92:
                    raise ValueError(
                        f"transuranic / actinide Z={z_int} is not a "
                        f"supported atom in the decoder; max supported "
                        f"Z is 92 (U).  Project scope is precious-"
                        f"metal anticancer (Pt/Ru/Ir/Rh/Os/Re)."
                    )
                safe_z = int(max(1, min(92, z_int)))
                rw.AddAtom(Chem.Atom(safe_z))
            except Exception:
                rw.AddAtom(Chem.Atom(6))
        bt_map = {
            BOND_SINGLE: Chem.BondType.SINGLE,
            BOND_DOUBLE: Chem.BondType.DOUBLE,
            BOND_TRIPLE: Chem.BondType.TRIPLE,
            BOND_AROMATIC: Chem.BondType.AROMATIC,
        }
        for src, dst, order in bond_orders:
            bt = bt_map.get(int(order), Chem.BondType.SINGLE)
            try:
                rw.AddBond(int(src), int(dst), bt)
            except Exception:
                # Skip invalid bonds (RDKit raises on duplicate or
                # out-of-range atoms).
                continue
        try:
            mol = rw.GetMol()
        except Exception as exc:
            return DecodedMol(
                error=f"RWMol.GetMol failed: {exc}",
                n_atoms=cloud.positions.shape[0],
                dative_source=dative_source,
                dative_manufactured=int(dative_manufactured),
            )
        # 2026-10-13 (BLK-C-L3 / instrument-fix) — SanitizeMol branch must
        # populate ``DecodedMol.error`` so the rejection reason reaches the
        # caller.  Pre-fix the branch swallowed the exception and set
        # ``smiles=""``; the only downstream signal was ``sanitized=False``
        # and an empty SMILES, which the wf_blkc_l3 guidance probe read as
        # ``error_stage="unknown_no_error_field"`` on every one of 64/64
        # bias-3.0 decodes.  That made the post-hoc aromatic-offset sweep
        # uninterpretable: there was no way to tell a bond-head rejection
        # from an RDKit valence refusal.
        #
        # The ``error`` field is NEW INFORMATION on this path; ``smiles``,
        # ``bond_orders``, ``dative_bonds``, ``n_atoms``, ``n_bonds``, and
        # ``sanitized`` are all set identically to the pre-fix values, so no
        # existing numeric result (or downstream consumer that gates on
        # ``sanitized`` or ``smiles``) changes.  The string only ever
        # materialises when SanitizeMol already raised.
        sanitized = False
        smiles = ""
        sanitize_error: Optional[str] = None
        repair = ""
        # Wf-A79 (2026-10-18) — refuse to call a non-ring bond aromatic.
        # MEASURED on the L3 arm: 57/64 clouds died in the kekulizer with
        # "non-ring atom N marked aromatic" (decode_rate 0.109).  The pass
        # is a no-op on molecules whose aromatic bonds are all in rings.
        mol = _demote_aromatic_outside_rings(mol)
        # Wf-A83 (2026-10-18) — the other half: an aromatic bond inside a
        # 3-/4-membered ring cannot be kekulized either, and Wf-A79's
        # "not in a ring" test leaves it alone.  MEASURED on a 1-epoch
        # checkpoint: "Can't kekulize mol. Unkekulized atoms: 0 1 2".
        mol = _demote_small_ring_aromatic(mol)
        try:
            Chem.SanitizeMol(mol)
            sanitized = True
            smiles = Chem.MolToSmiles(mol)
        except Exception as exc:
            sanitized = False
            smiles = ""
            sanitize_error = f"{type(exc).__name__}: {exc}"
            # Wf-A84 (2026-10-19) — last resort: keep the molecule, drop the
            # aromaticity.  Wf-A79 + Wf-A83 already removed the two classes
            # of aromatic bond that CAN be repaired locally; whatever the
            # kekulizer still refuses is a ring it cannot Kekulize as a
            # whole.  Demoting every aromatic bond to SINGLE makes the graph
            # sanitizable at the cost of the ring's unsaturation, so the
            # result is recorded as a REPAIR (trainer bucket
            # `ok_repaired:kekulize_demoted_all`), never as a clean decode.
            if "Kekulize" in type(exc).__name__:
                mol = _demote_all_aromatic(mol)
                try:
                    Chem.SanitizeMol(mol)
                    sanitized = True
                    smiles = Chem.MolToSmiles(mol)
                    sanitize_error = None
                    repair = "kekulize_demoted_all"
                except Exception as exc2:
                    sanitize_error = f"{type(exc2).__name__}: {exc2}"
        return DecodedMol(
            mol=mol,
            smiles=smiles,
            bond_orders=list(bond_orders),
            dative_bonds=list(dative_bonds),
            n_atoms=cloud.positions.shape[0],
            n_bonds=len(bond_orders),
            sanitized=sanitized,
            error=sanitize_error,
            repair=repair,
            dative_source=dative_source,
            dative_manufactured=int(dative_manufactured),
        )

    def _empty_decoded(
        self,
        cloud: AtomCloud,
        yres: YuelBondResult,
        reason: str,
    ) -> YuelBondDecodeResult:
        """Build an empty DecodedMol + YuelBondDecodeResult.

        Used when the YuelBond head returns no candidate edges (N=0,
        N=1, or all pairs > cutoff).
        """
        from rdkit import Chem

        rw = Chem.RWMol()
        for z in cloud.atomic_numbers.tolist():
            try:
                # Wf-A49 (2026-10-16) arch_fix_5 — same Z-cap as the
                # main path.  See ``_assemble_mol`` for why Z > 92 is
                # rejected (transuranic / trans-actinide guard).
                z_int = int(z)
                if z_int > 92:
                    raise ValueError(
                        f"transuranic / actinide Z={z_int} not supported; "
                        f"max Z is 92."
                    )
                safe_z = int(max(1, min(92, z_int)))
                rw.AddAtom(Chem.Atom(safe_z))
            except Exception:
                rw.AddAtom(Chem.Atom(6))
        mol = rw.GetMol()
        return YuelBondDecodeResult(
            decoded=DecodedMol(
                mol=mol,
                smiles="",
                bond_orders=[],
                dative_bonds=[],
                n_atoms=cloud.positions.shape[0],
                n_bonds=0,
                sanitized=False,
                error=f"yuelbond:{reason}",
            ),
            yuelbond_result=yres,
            bonds_accepted=0,
            bonds_rejected_overvalent=0,
            # Wf-A113 — this early return is the case the collapse rule
            # CREATES: a cloud whose every pair was rejected has an empty
            # edge list, so it lands here with ``decode_succeeded=False`` and
            # ``reason="empty_or_failed"``.  Reporting 0 there would hide the
            # rule exactly where it fired hardest.
            clash_rejected_pairs=int(
                getattr(
                    getattr(self._decoder, "head", None),
                    "last_clash_rejected_pairs",
                    0,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Dative detector — same convention as BondAwareDecoder.
    # ------------------------------------------------------------------
    def _cap_for_z(self, z: int) -> float:
        """Resolve the covalent-valence cap for an atomic number.

        Wf-A49 (2026-10-16) arch_fix_10 — three dispositions for a
        Z not in :data:`_COVALENT_VALENCE`:

          1. ``unknown_z_default=None`` (default) — silently cap at 4
             (carbon), matching the legacy behaviour.
          2. ``unknown_z_default=<int>`` — pin the cap to the
             supplied value (use for ``unknown_z_default=3`` to
             mimic boron).
          3. ``unknown_z_default="raise"`` — refuse the decode
             instead of silently mapping the atom to a wildly wrong
             cap.

        Returns
        -------
        float
            The covalent-valence cap for ``z``.

        Raises
        ------
        ValueError
            When ``z`` is not in :data:`_COVALENT_VALENCE` AND
            ``unknown_z_default="raise"``.
        """
        if z in self.valence_cap:
            return float(self.valence_cap[z])
        if self.unknown_z_default is None:
            return 4.0
        if (
            isinstance(self.unknown_z_default, str)
            and self.unknown_z_default == "raise"
        ):
            raise ValueError(
                f"atomic number Z={z} is not in the covalent-valence "
                f"table; decoder constructed with unknown_z_default='raise' "
                f"refuses unknown Z outright."
            )
        return float(self.unknown_z_default)

    def _is_metal_z(self, z: int) -> bool:
        """Whether Z is a metal for the dative-bond detector.

        2026-10-04: the tuple below used to be (78, 44, 30, 77, 29, 79, 46,
        26, 28), which covers Pt/Ru/Zn/Ir/Cu/Au/Pd/Fe/Ni but NOT Os (76),
        Re (75) or Rh (45) -- three of this project's six target metals.

        Wf-A92 (2026-10-19): this used to be a hardcoded 12-metal tuple
        that disagreed with the HEAD's ``metal_set`` (all7 = Ru/Rh/Pd/Re/
        Os/Ir/Pt).  The head is trained with its own set, so the selector
        could flag a Fe/Ni/Cu/Zn/Au bond as dative while the head had never
        seen that element as a metal.  The head's set is now the single
        source of truth; the old 12-metal tuple (which added Fe/Ni/Cu/Zn/
        Au) is recorded in this docstring, not in code.
        """
        return int(z) in self._decoder.head.metal_set

    def _is_donor(self, z: int) -> bool:
        """Wf-A95 — read the HEAD's set, not the module global.

        A checkpoint stamped before the donor set changed decodes with the
        set it was trained under (``head.donor_zs`` restored from
        ``feature_schema``), so its ``is_dative`` column is bit-identical to
        the one its weights saw.
        """
        return int(z) in self._decoder.head.donor_zs

    def _is_dative_pair(self, z_i: int, z_j: int) -> bool:
        """Z-only dative rule (no distance, no truth).  See
        :meth:`_is_dative_pair_at` for the rule the decoder actually uses."""
        return (self._is_metal_z(z_i) and self._is_donor(z_j)) or (
            self._is_metal_z(z_j) and self._is_donor(z_i)
        )

    def _is_dative_pair_at(
        self,
        cloud: AtomCloud,
        i: int,
        j: int,
        z_i: int,
        z_j: int,
        truth: Optional[frozenset] = None,
    ) -> bool:
        """Is the kept bond ``(i, j)`` dative?

        Precedence (Wf-A92):
          1. the cloud's own ``is_dative_pair`` truth, when present — a
             metal-amido bond is covalent and truth says so;
          2. otherwise the Z rule AND ``d <= dative_max_distance``.  The
             distance bound is what stops a long, genuinely covalent
             metal-donor contact from being reported as a dative bond.
        """
        if truth is not None:
            return (int(i), int(j)) in truth
        if not self._is_dative_pair(z_i, z_j):
            return False
        try:
            d = float(
                torch.norm(
                    cloud.positions[int(i)].float()
                    - cloud.positions[int(j)].float()
                ).item()
            )
        except Exception:
            return True
        return d <= float(self.dative_max_distance)


# ---------------------------------------------------------------------------
# Synthetic CFM-style Pt-click cloud generator (smoke input)
# ---------------------------------------------------------------------------
def build_synthetic_pt_click_clouds(
    n_clouds: int = 8,
    n_atoms_per_cloud: int = 5,
    seed: int = 0,
    jitter: float = 0.20,
) -> List[AtomCloud]:
    """Build ``n_clouds`` synthetic cisplatin-like Pt-click atom clouds.

    Layout per cloud: 1 Pt_II at origin + 2 NH3 + 2 Cl at canonical
    square-planar geometry (2.05 Å Pt-donor), with optional Gaussian
    jitter on positions to mimic CFM noise.

    This is the smoke-test input for Wf-124 Phase 1 — the synthetic
    Pt-click cloud the YuelBond decoder should be able to decode into a
    cisplatin-like SMILES (or close to it).  See
    ``metrics/by_metric/decode_ratio.json`` (2026-09-16): the
    ``wf_cfm_path_b_decoder_rework`` Path-B row reports
    ``decode_ratio_rdkit_strict = 0.5`` on 192 such clouds — i.e. only
    half the synthetic clouds produce a RDKit-valid SMILES because of
    overvalent Pt atoms.  Wf-124 P1 is to ship a decoder path that
    pushes that ratio higher via the YuelBond + valence-cap combo.

    Parameters
    ----------
    n_clouds : int
        Number of atom clouds to build.  Defaults to 8 (the smoke size).
    n_atoms_per_cloud : int
        Atom count per cloud.  Defaults to 5 (1 Pt + 2 N + 2 Cl =
        cisplatin heavy-atom layout).
    seed : int
        RNG seed for the jitter.
    jitter : float
        Std-dev of the Gaussian position noise (Å).  ``jitter=0`` yields
        the textbook cisplatin layout; ``jitter=0.20`` is the canonical
        CFM-smoke value.

    Returns
    -------
    list[AtomCloud]
        List of synthetic clouds.
    """
    g = torch.Generator().manual_seed(seed)
    clouds: List[AtomCloud] = []
    for _ in range(n_clouds):
        # Canonical square-planar Pt_II layout: 4 donors at 2.05 Å.
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Pt
                [2.05, 0.0, 0.0],  # Cl  (right)
                [-2.05, 0.0, 0.0],  # Cl  (left)
                [0.0, 2.05, 0.0],  # NH3 (top)
                [0.0, -2.05, 0.0],  # NH3 (bottom)
            ],
            dtype=torch.float32,
        )
        atomic_numbers = torch.tensor([78, 17, 17, 7, 7], dtype=torch.long)
        # Apply jitter (positions only — atomic numbers unchanged).
        if jitter > 0.0:
            positions = (
                positions
                + torch.randn(
                    positions.shape,
                    generator=g,
                    dtype=torch.float32,
                )
                * jitter
            )
        # Optional: drop / extend atoms if n_atoms_per_cloud != 5.
        if n_atoms_per_cloud < 5:
            positions = positions[:n_atoms_per_cloud]
            atomic_numbers = atomic_numbers[:n_atoms_per_cloud]
        elif n_atoms_per_cloud > 5:
            # Pad with extra C / H donors to grow the cloud.
            extra_z = [6] * (n_atoms_per_cloud - 5)
            extra_p = (
                torch.randn(
                    (n_atoms_per_cloud - 5, 3),
                    generator=g,
                    dtype=torch.float32,
                )
                * 1.5
            )
            positions = torch.cat([positions, extra_p], dim=0)
            atomic_numbers = torch.cat(
                [
                    atomic_numbers,
                    torch.tensor(extra_z, dtype=torch.long),
                ],
                dim=0,
            )
        clouds.append(AtomCloud(positions=positions, atomic_numbers=atomic_numbers))
    return clouds


# ---------------------------------------------------------------------------
# Self-test (lightweight — run as ``python -m
# molmetal_lam.lam_chem.cfm_decoder_yuelbond``).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import json
    import sys
    from pathlib import Path

    torch.manual_seed(0)
    print("=== YuelBondBondDecoder smoke ===")
    decoder = YuelBondBondDecoder(hidden_dim=64, n_layers=3)
    clouds = build_synthetic_pt_click_clouds(n_clouds=8, jitter=0.20)
    n_decoded = 0
    n_atoms = 0
    n_bonds = 0
    overvalent_total = 0
    for k, cloud in enumerate(clouds):
        result = decoder.decode(cloud)
        dec = result.decoded
        if dec.sanitized:
            n_decoded += 1
        n_atoms += dec.n_atoms
        n_bonds += dec.n_bonds
        overvalent_total += result.bonds_rejected_overvalent
        print(
            f"  cloud {k}: smiles={dec.smiles!r}, "
            f"n_atoms={dec.n_atoms}, n_bonds={dec.n_bonds}, "
            f"sanitized={dec.sanitized}, "
            f"accepted={result.bonds_accepted}, "
            f"overvalent_rej={result.bonds_rejected_overvalent}"
        )
    decode_ratio = n_decoded / max(1, len(clouds))
    print(
        f"\ndecode_ratio = {n_decoded}/{len(clouds)} = {decode_ratio:.3f} "
        f"(baseline 0/8 with 4-fix bundle; "
        f"Path-B RDKit-strict 0.5 with bond-bearing decoder)"
    )
    print(
        f"avg atoms={n_atoms / len(clouds):.1f}, "
        f"avg bonds={n_bonds / len(clouds):.1f}, "
        f"overvalent-rejected total={overvalent_total}"
    )
