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
from typing import Dict, List, Optional, Sequence, Tuple

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
)

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
    # Metals — coordination number (max sum of bond orders).
    78: 4,  # Pt_II  (Pt_IV = 6; we use the conservative Pt_II cap)
    44: 6,  # Ru_II  (octahedral)
    30: 4,  # Zn_II
    77: 6,  # Ir_III
    29: 4,  # Cu_II
    79: 4,  # Au_III
    46: 4,  # Pd_II
    26: 4,  # Fe_II (default; rare in SBDD ligand sets)
    28: 4,  # Ni_II
    # 2026-10-05 NEW — Rh, Re and Os.  These are THREE OF THE SIX METALS THIS
    # PROJECT TARGETS and they were absent, so they took the default of 4.
    # MEASURED on a 200-cloud tmQM val sample: Rh appears 32x, Re 2x, Os 2x.
    # An octahedral ceiling is the conservative choice — a cap that is too
    # LOW only costs a bond, while a cap that is too HIGH produces a molecule
    # RDKit refuses, and the decode is lost outright.
    45: 6,  # Rh (octahedral — the common Rh(III) count)
    75: 6,  # Re (octahedral)
    76: 6,  # Os (octahedral)
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
        decision_threshold: float = 0.5,
        class_logit_bias: Optional[Sequence[float]] = None,
        connectivity_prior: bool = True,
        connectivity_prior_min_logit: float = 0.0,
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
        if self.decision not in ("argmax", "threshold"):
            # Fail HERE, not at the first decode.  A yaml typo like
            # `bond_decision: treshold` otherwise decodes with argmax and the
            # only symptom is molecules that do not match the reported
            # metrics — discovered, if at all, days later.
            raise ValueError(
                f"unknown bond decision rule {self.decision!r}; expected "
                f"'argmax' or 'threshold'"
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
        n_bridges = 0
        if self.connectivity_prior and bond_orders:
            bond_orders, dative_bonds, n_bridges = self._bridge_components(
                cloud,
                yres,
                bond_orders,
                dative_bonds,
            )
        # Step 3: assemble into an RDKit Mol.
        decoded = self._assemble_mol(cloud, bond_orders, dative_bonds)
        return YuelBondDecodeResult(
            decoded=decoded,
            yuelbond_result=yres,
            bonds_accepted=n_kept + n_bridges,
            bonds_rejected_overvalent=n_overvalent,
            bonds_added_connectivity=n_bridges,
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
        )
        # Confidence score = max logit per pair (skip NO_BOND class for
        # a stronger "is this pair bonded" signal).
        # max_logit[i] = max(logits[i, 1:])  (excluding class 0)
        max_logits = logits[:, 1:].max(dim=-1).values.tolist()
        # Build sorted pair list.
        pairs: List[Tuple[float, int, int, int]] = []
        for k, (src, dst, order) in enumerate(
            zip(edge_index[0].tolist(), edge_index[1].tolist(), predicted),
        ):
            if int(src) == int(dst):
                continue
            if int(order) == BOND_NO_BOND:
                continue
            pairs.append((float(max_logits[k]), int(src), int(dst), int(order)))
        # Sort by descending confidence.
        pairs.sort(key=lambda t: t[0], reverse=True)
        # Per-atom remaining valence.
        z_list = cloud.atomic_numbers.tolist()
        remaining = [int(self.valence_cap.get(int(z), 4)) for z in z_list]
        bond_orders: List[Tuple[int, int, int]] = []
        dative_bonds: List[Tuple[int, int]] = []
        n_overvalent = 0
        n_kept = 0
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
            if cost > cap_i or cost > cap_j:
                n_overvalent += 1
                continue
            bond_orders.append((i, j, order))
            remaining[i] -= cost
            remaining[j] -= cost
            n_kept += 1
            # Dative flag — same convention as BondAwareDecoder.
            zi = int(z_list[i])
            zj = int(z_list[j])
            dative_ok = self._is_dative_pair(zi, zj) and order == BOND_SINGLE
            if dative_ok:
                # Donor = non-metal, Metal = the Pt/Au/Cu/Ru/Ir/Zn/Pd
                if self._is_metal(zi) and not self._is_metal(zj):
                    dative_bonds.append((j, i))
                elif self._is_metal(zj) and not self._is_metal(zi):
                    dative_bonds.append((i, j))
                else:
                    dative_bonds.append((i, j))
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
        remaining = [int(self.valence_cap.get(z, 4)) for z in z_list]
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
            if self._is_dative_pair(z_list[i], z_list[j]):
                if self._is_metal(z_list[i]) and not self._is_metal(z_list[j]):
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
                safe_z = int(max(1, min(90, int(z))))
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
        try:
            Chem.SanitizeMol(mol)
            sanitized = True
            smiles = Chem.MolToSmiles(mol)
        except Exception as exc:
            sanitized = False
            smiles = ""
            sanitize_error = f"{type(exc).__name__}: {exc}"
        return DecodedMol(
            mol=mol,
            smiles=smiles,
            bond_orders=list(bond_orders),
            dative_bonds=list(dative_bonds),
            n_atoms=cloud.positions.shape[0],
            n_bonds=len(bond_orders),
            sanitized=sanitized,
            error=sanitize_error,
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
                safe_z = int(max(1, min(90, int(z))))
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
        )

    # ------------------------------------------------------------------
    # Dative detector — same convention as BondAwareDecoder.
    # ------------------------------------------------------------------
    @staticmethod
    def _is_metal(z: int) -> bool:
        """Whether Z is a metal for the purposes of the dative-bond detector.

        2026-10-04: the tuple below used to be (78, 44, 30, 77, 29, 79, 46,
        26, 28), which covers Pt/Ru/Zn/Ir/Cu/Au/Pd/Fe/Ni but NOT Os (76),
        Re (75) or Rh (45) -- three of this project's six target metals.  A
        tmQM Os/Re/Rh complex would therefore have had its metal-to-donor
        bonds classified as ordinary covalent bonds: the vocabulary extension
        alone would not have been enough.

        Grouped rather than left as a bare tuple so the gap is visible:
          * the six project targets: Pt 78, Ru 44, Ir 77, Rh 45, Os 76, Re 75
          * first-row transition metals already in the list: Fe 26, Ni 28,
            Cu 29, Zn 30
          * other d8/d10 metals already in the list: Pd 46, Au 79
        """
        return z in (
            78, 44, 77, 45, 76, 75,  # the six project targets (Rh/Os/Re NEW)
            26, 28, 29, 30, 46, 79,  # pre-existing: Fe, Ni, Cu, Zn, Pd, Au
        )

    @staticmethod
    def _is_donor(z: int) -> bool:
        return z in (7, 8, 15, 16, 34)

    def _is_dative_pair(self, z_i: int, z_j: int) -> bool:
        return (self._is_metal(z_i) and self._is_donor(z_j)) or (
            self._is_metal(z_j) and self._is_donor(z_i)
        )


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
