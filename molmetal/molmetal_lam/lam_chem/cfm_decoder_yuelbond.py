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
Dokholyan (2026, J. Chem. Inf. Model. 66(2) 1003-1012) report F1=92.7 %
on the CDG set (Gaussian noise sigma = 0.2 A) — their **worst case**;
on standard 3-D structures they report F1 = 98.4 %.  The RDKit baseline
they benchmark is ``DetermineBonds``, which the paper describes as
"internally … leveraging the xyz2mol program"; on the same 1,000 CDGs
the paper attributes the 217-processed / 783-failed split to RDKit
itself (it prints that integer split, never a percentage).
``DetermineConnectivity`` is the connectivity-only step that
``DetermineBonds`` calls internally before assigning bond orders — the
two are not independent siblings.  (VERIFIED 2026-10-31 against the
RDKit 2026.03.6 source file
``Code/GraphMol/DetermineBonds/DetermineBonds.cpp``: L543 calls
``determineConnectivity(...)``, then L544 calls ``determineBondOrders(...)``.)
Called on its own, ``DetermineConnectivity`` assigns no bond orders at all
(MEASURED locally: every bond it emits is SINGLE); the paper compares
against ``DetermineBonds``, which additionally assigns bond orders.
When wired with a **valence-aware bond-order
selector** (greedy assignment that respects per-atom max-valence), the
YuelBond head produces RDKit-sanitizable ``Mol`` objects in a
deterministic, CPU-friendly path (~5k params, K=3 layers).

Lit anchors
-----------
* Wang, J.; Dokholyan, N. V. (2026). *Multimodal Bond Reconstruction
  toward Generative Molecular Design.*  J. Chem. Inf. Model. 66(2),
  1003-1012.  DOI 10.1021/acs.jcim.5c03052  (preprint: bioRxiv
  10.1101/2025.05.06.652517, 2025-05-11).  F1 = 92.7 % on the CDG set
  (worst case, sigma = 0.2 A); F1 = 98.4 % on standard 3-D.
  (Citation corrected 2026-10-31, Wf-A143 — this bullet previously
  printed the title "YuelBond: a graph neural network for recovering the molecular bond network from distorted 3D structures", the author initials "Wang, S., Dokholyan, P." and the attribution of the 783/1000 RDKit failure count to ``DetermineConnectivity``.  See
  `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.)
  Attribution corrected 2026-10-31 (Wf-A146) — this bullet previously
  read "The RDKit baseline they benchmark is ``DetermineBonds``; on the
  same 1,000 CDGs ``xyz2mol`` processed 217 (=> 783 failures)."  The
  paper attributes the 217/783 split to **RDKit** itself and prints that
  integer split, never a percentage; ``xyz2mol`` is the program
  ``DetermineBonds`` "internally … leverages".  Superseded wording
  preserved verbatim per AGENTS.md §5.5 rule #6; see
  `docs/HONEST_NEGATIVE_CATALOG.md` §6.259.
  Relation corrected 2026-10-31 (Wf-A148) — the ``DetermineConnectivity``
  paragraph above previously read:

      ``DetermineConnectivity`` is the
      connectivity-only sibling and assigns no bond orders at all (MEASURED
      locally: every bond it emits is SINGLE), so it is not the function the
      paper compares against.

  The "sibling" relation is wrong: ``DetermineBonds`` calls
  ``DetermineConnectivity`` internally (RDKit 2026.03.6,
  ``Code/GraphMol/DetermineBonds/DetermineBonds.cpp`` L543) and then assigns
  bond orders on top (L544), so ``DetermineConnectivity`` is the
  connectivity-only STEP INSIDE ``DetermineBonds``, not an independent
  sibling of it.  The "MEASURED locally: every bond it emits is SINGLE"
  observation stands; it is now scoped to ``DetermineConnectivity`` called on
  its own.  Superseded wording preserved verbatim per AGENTS.md §5.5 rule
  #6.  (Catalog entry pending — this wave does not own
  ``docs/HONEST_NEGATIVE_CATALOG.md``.)

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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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

#: Elements whose aromatic bonds are charged 1.0, not 1.5.
#:
#: Wf-A113 (T78, 2026-10-28) — 1.5 is the average of the Kekulé single/double
#: pair, which is right for CARBON: benzene C = 2 x 1.5 + 1 H = 4 = its cap.
#: It is wrong for a heteroatom, which carries the aromatic bond either as a
#: lone pair (pyrrole N, furan O, thiophene S) or as the Kekulé double-bond
#: partner without a second substituent (pyridine N).  Charging 1.5 there
#: over-bills every ring by 0.5 per bond, and the selector then drops a real
#: bond:
#:
#:     pyrrole N    2 x 1.5 + 1 H = 4.0  >  cap 3
#:     furan O      2 x 1.5       = 3.0  >  cap 2
#:     amide N      1.5 + 1 + 1   = 3.5  >  cap 3   (aryl-NH-C=O)
#:
#: MEASURED on the six-metal cohort (33,429 complexes, every bond >= 0.3,
#: hydrogens dropped the way ``load_tmqm_clouds`` drops them — the decoder
#: never sees an H, so a cap measured on an H-bearing graph measures a
#: different molecule): 61.7% of N atoms and 30.7% of O atoms exceeded their
#: cap on the CORPUS's own graph, i.e. the decoder could not rebuild the truth
#: it was trained on.  With this rule: N 3.6%, O 9.2%, S 34.7%, Se 26.3%.
#: Carbon is unchanged at 1.2%, which is the control: the rule must not touch
#: it (``metrics/by_metric/wfa113_t78_element_cost_fit.json``).  Those are
#: the numbers with the ligand still paying for its metal bond; T79 removes
#: that charge and the same table reads N 0.13%, O 0.07%, S 20.76%,
#: Se 1.10% (``metrics/by_metric/wfa113_t79_dative_ligand_budget.json``).
_AROMATIC_HETERO_ZS: frozenset = frozenset(
    {
        5,   # B
        7,   # N
        8,   # O
        15,  # P
        16,  # S
        33,  # As
        34,  # Se
    }
)


def _endpoint_valence_cost(
    order: int,
    z_self: int,
    z_other: int,
    is_metal: Callable[[int], bool],
    arom_bonds_so_far: int = 0,
) -> float:
    """Valence budget one bond consumes at ONE of its two endpoints.

    This is the single place the selector (``_select_bonds``) and the
    connectivity prior (``_bridge_components``) get their accounting from.
    They used to carry two hand-copied blocks, and every unit fix landed in
    one of them and not the other: T77 (a metal's budget is a bond COUNT, not
    a bond-order sum) was applied to both by hand, T78 (an aromatic bond at a
    heteroatom is Kekule-single) only to the selector.  Sharing the function
    is what stops that recurring.

    Three rules, in order:

    1. **Dative at the donor.**  A bond to a metal costs the ligand endpoint
       nothing.  The electron pair is the ligand's lone pair, so the donor's
       valence is unchanged.  RDKit implements exactly this in
       ``cleanupOrganometallics()``, which runs inside ``SanitizeMol`` on
       every mol this decoder produces: it rewrites a single bond between a
       metal and a hypervalent atom into ``BondType.DATIVE``, and a dative
       bond does not count towards the donor's explicit valence.  Verified on
       RDKit 2026.03.6 -- an ether O with 2 C and 1 Ru reports explicit
       valence 2, sanitises, and writes ``C[O](C)->[Ru]``; the same O with 3 C
       and 1 Ru raises ``AtomValenceException``.  Upstream describes the step
       as "the organometallic cleanup code (this converts single bonds
       between hypervalent atoms and metals to dative bonds)"
       (rdkit/rdkit#8312).  Charging the ligand made the selector STRICTER
       than the validator it feeds: it dropped a real bond RDKit would have
       kept, which opens a ring and turns a one-fragment decode into a
       multi-fragment one.  MEASURED on the six-metal cohort (33,429
       complexes, heavy atoms only): ligand atoms over their cap fall from
       N 3.64% to 0.13%, O 9.18% to 0.07%, Se 26.29% to 1.10%, S 34.72% to
       20.76%, C 1.23% to 0.24%
       (``metrics/by_metric/wfa113_t79_dative_ligand_budget.json``).

    2. **A metal's budget counts bonds.**  The metal caps in
       ``_COVALENT_VALENCE`` come from tmQM's ``coord_number`` (MND p99), a
       count, so a metal bond costs 1.0 whatever class the head gave it.
       Charging the class cost instead made a 10-bond ceiling unreachable in
       exactly the complexes that need it: 30.7% of Re's metal-incident bonds
       are not single, so the cost sum exceeded the bond count in 92.3% of Re
       complexes (``metrics/by_metric/wfa113_t77_metal_bond_cost_units.json``).

    3. **Aromatic at a heteroatom is Kekule-single.**  1.5 is the average of
       the Kekule single/double pair, which is right for carbon (benzene C =
       2 x 1.5 + 1 H = 4, its cap exactly) and wrong for N/O/S/Se/P/As/B,
       which carry the aromatic bond as a lone pair (pyrrole N, furan O,
       thiophene S) or as the Kekule partner with no second substituent
       (pyridine N).  Billing 1.5 there dropped a real ring bond
       (``metrics/by_metric/wfa113_t78_element_cost_fit.json``).

    4. **A carbon's THIRD aromatic bond is Kekule-single.**  The 1.5 average
       assumes exactly two aromatic bonds, where the Kekule assignment is one
       double and one single.  A ring-fusion carbon has three, and no Kekule
       structure can give it two doubles, so the sum is 4 (one double, two
       single) and not 4.5.  Charging 1.5 for all three put the fusion carbon
       at 4.5 against its cap of 4, so the selector dropped the third bond and
       broke the fusion.  MEASURED: 1,317 carbon atoms in the six-metal cohort
       are a fusion carbon of an ``aromatic/aromatic/aromatic`` pattern
       (``metrics/by_metric/wfa113_t80_residual_caps.json``).  ``arom_bonds_so_far``
       is the caller's count of aromatic bonds already charged at this atom;
       the total is order-independent (1.5 + 1.5 + 1.0 = 4 however the three
       are ordered).

    ``is_metal`` is passed in rather than read from module state so the
    selector's ``metal_set`` (which the trainer can override) is the one
    that applies.
    """
    if is_metal(z_other) and not is_metal(z_self):
        return 0.0
    if is_metal(z_self):
        return 1.0
    if int(order) == BOND_AROMATIC:
        if z_self in _AROMATIC_HETERO_ZS:
            return 1.0
        if arom_bonds_so_far >= 2:
            return 1.0
    return _CLASS_VALENCE_COST.get(int(order), 1.0)


#: Wf-A155 (2026-11-01) — RDKit's own default valence per element.
#:
#: The decode graph is HYDROGEN-FREE (the cloud carries no H), so an atom's
#: "target valence" in the heavy-atom graph is not the covalent-valence CAP
#: (which is a ceiling: S 6, P 5) but the valence RDKit will complete with
#: implicit hydrogens.  RDKit's periodic table is the authority for that, and
#: this wrapper is the ONE place it is read.
#:
#: MEASURED: read live from RDKit's periodic table, not from a table of our
#: own; 0 for an element RDKit has no default valence for (every metal).
_DEFAULT_VALENCE_CACHE: Dict[int, float] = {}


def _rdkit_default_valence(z: int) -> float:
    """RDKit's default valence for element ``z``; 0.0 when it has none.

    MEASURED: the value comes from ``Chem.GetPeriodicTable()`` at call
    time (cached per element).  0.0 is returned for an element RDKit does
    not know and for every transition metal — RDKit carries no default
    valence for a metal, which is exactly why a metal's coordination cap
    (see :data:`_COVALENT_VALENCE`) is its valence target instead.

    PROJECTED: nothing here is a claim about a decoded pool.
    """
    zi = int(z)
    cached = _DEFAULT_VALENCE_CACHE.get(zi)
    if cached is not None:
        return cached
    val = 0.0
    try:
        from rdkit import Chem

        val = float(Chem.GetPeriodicTable().GetDefaultValence(zi))
    except Exception:
        val = 0.0
    _DEFAULT_VALENCE_CACHE[zi] = val
    return val


def _replay_remaining_budget(
    bond_orders: Sequence[Tuple[int, int, int]],
    z_list: Sequence[int],
    caps: Sequence[float],
    is_metal: Callable[[int], bool],
) -> Tuple[List[float], List[int]]:
    """Replay the selector's own valence accounting over a bond list.

    Returns ``(remaining, arom_kept)`` — the per-atom valence budget left
    after every bond in ``bond_orders``, and the per-atom count of aromatic
    bonds charged so far (which ``_endpoint_valence_cost`` needs for its
    aromatic-heteroatom and fusion-carbon rules).

    Wf-A133 (2026-10-31) — this block used to be written out TWICE: once
    in ``_select_bonds`` (which charges as it goes) and once in
    ``_bridge_components`` (which replays what the selector already
    charged).  The forced completion stage needs the same replay, and a
    THIRD hand-copied version is exactly how the T77/T78 accounting bugs
    happened — one fix landing in one copy and not in the others.  One
    function, two callers, no third copy.

    The accounting is deliberately IDENTICAL to the selector's: same
    per-endpoint cost, same ``arom_kept`` increment order.  A stage that
    used a different accounting could authorise a bond the selector would
    have refused as overvalent, and RDKit would then reject the molecule.
    """
    remaining: List[float] = [float(c) for c in caps]
    arom_kept: List[int] = [0] * len(caps)
    for i, j, order in bond_orders:
        remaining[i] -= _endpoint_valence_cost(
            order, int(z_list[i]), int(z_list[j]), is_metal, arom_kept[i]
        )
        remaining[j] -= _endpoint_valence_cost(
            order, int(z_list[j]), int(z_list[i]), is_metal, arom_kept[j]
        )
        if int(order) == BOND_AROMATIC:
            arom_kept[i] += 1
            arom_kept[j] += 1
    return remaining, arom_kept


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
    15: 5,  # P (Wf-A113 T78 — RDKit permits 3 or 5; the corpus is phosphate,
    #          P with four single bonds, so 3 refused the 4th bond on 98.8% of
    #          P atoms in the six-metal cohort, and P is in 7.7% of complexes)
    16: 6,  # S (Wf-A113 T80 — RDKit's limit for S is 6, not 2.  A cap of 2 is
    #          the thioether/thiophene value and refuses every sulfoxide,
    #          sulfone and sulfonium in the corpus.  MEASURED on the six-metal
    #          cohort: S bond counts are 1:198, 2:7,120, 3:1,720, 4:2,090,
    #          5:2, 6:29, so 3,841 of 11,159 S atoms (34.4%) have more than
    #          two bonds.  After T79 removed the metal-bond charge, 20.76% of S
    #          atoms were still over cap, which is 5.31% of all complexes.)
    17: 1,  # Cl
    33: 5,  # As (Wf-A113 T80 — RDKit's limit is 5.  The corpus is As(V):
    #          bond counts 3:37, 4:364, 5:4, 6:2, so a cap of 3 refused 370 of
    #          409 As atoms (90.5%).  The two 6-bond atoms are beyond RDKit.)
    34: 6,  # Se (Wf-A113 T80 — RDKit's limit is 6, same group as S.  Corpus
    #          bond counts 1:10, 2:394, 3:137, 4:1, 5:2.)
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
    # Wf-A113 (T88, 2026-10-29) — Pt used to stay at 4 as a "chemistry choice,
    # not a statistic": 81.8% of the Pt corpus is four-coordinate and the
    # project targets Pt(II).  That exception was the last survivor of the
    # defect T72 removed for the other five metals, and it cost more than all
    # of them together.  MEASURED (tmqm_parsed.csv, 8,275 Pt rows): 1,465
    # complexes (17.7%) exceed 4 and 2,714 metal bonds are stranded, against
    # 0.2-0.9% and 10-105 bonds for Ru/Rh/Ir/Os/Re.  The artifact that
    # justified T72 recorded this and nobody read the column: Pt showed
    # ``frac_metal_bonds_unrepresentable = 0.0813`` in 18.6% of clouds while
    # every other metal was at 0
    # (``metrics/by_metric/wfa113_t72_valence_cap.json``).
    #
    # The "six-coordinate ceiling lets the decoder build Pt(IV) on a Pt(II)
    # prior" objection does not survive either half: (a) RDKit carries no
    # default valence for Pt, so a high cap cannot lose a decode (verified:
    # [Pt] with six single bonds sanitises); (b) the cap is a PERMISSION, not
    # a compulsion -- it only bounds what the selector may keep of what the
    # head already proposed, so it cannot add a bond; (c) Pt(IV) octahedral
    # prodrugs are a first-class anticancer class (satraplatin, the first oral
    # Pt drug to reach the clinic; ormaplatin, iproplatin; the 2025-2026
    # literature is largely Pt(IV) prodrug design), so excluding them excludes
    # target chemistry, not noise.  Pt now takes the same p99 rule as the
    # other five.
    #
    # Zn/Cu/Au/Pd/Fe/Ni are outside the six-metal cohort and keep their
    # chemistry-first values.
    78: 6,  # Pt  (p50 4, p99 6)
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


def _demote_small_ring_aromatic(mol, min_size: int = 4):
    """Demote aromatic bonds inside rings SMALLER than `min_size`; return mol.

    Wf-A83 (2026-10-18).  Companion to `_demote_aromatic_outside_rings`, and
    the other half of the same measured defect.  OPT-2's ring audit found the
    generated corpus is 89.8% three-membered rings against 0.5% in the real
    corpus, and a 3-ring marked aromatic cannot be kekulized at all: MEASURED
    on a 1-epoch checkpoint, one of three decoded clouds died with
    `KekulizeException: Can't kekulize mol.  Unkekulized atoms: 0 1 2` --
    three atoms, one ring, aromatic bonds.  Wf-A79 cannot help there because
    the bond IS in a ring.

    Wf-A113 (T87, 2026-10-29) — ``min_size`` was 5, which also demoted
    FOUR-membered rings.  MEASURED on RDKit 2026.03.6: a 4-membered
    all-aromatic ring kekulizes (``C1=CC=C1``), and so does a bicyclic
    4+4 system; only the 3-membered ring fails.  Demoting 4-rings therefore
    turned a molecule RDKit accepts into cyclobutane and threw away the
    unsaturation the head had predicted.  The threshold is now 4, so only
    rings RDKit actually refuses are touched.

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


def _ring_bond_indices(mol) -> List[List[int]]:
    """One list of bond indices per SSSR ring; empty when perception fails.

    Wf-A113 (T83).  ``Chem.GetSSSR`` both finds the rings and initialises
    ``RingInfo``; calling ``GetRingInfo().AtomRings()`` first raises on an
    unsanitised mol (RDKit 2026.03.6), the same trap Wf-A79 hit.
    """
    from rdkit import Chem

    try:
        Chem.GetSSSR(mol)
    except Exception:
        return []
    out: List[List[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        n = len(ring)
        idxs: List[int] = []
        for k in range(n):
            bond = mol.GetBondBetweenAtoms(int(ring[k]), int(ring[(k + 1) % n]))
            if bond is None:
                idxs = []
                break
            idxs.append(bond.GetIdx())
        if len(idxs) == n:
            out.append(idxs)
    return out


def _demote_bond_set(mol, bond_indices) -> int:
    """Rewrite a specific set of bonds to SINGLE; return how many changed.

    Atom aromatic flags are cleared only for atoms left with no aromatic bond,
    so demoting one ring of a fused system leaves the rest aromatic.
    """
    from rdkit import Chem

    changed = 0
    touched = set()
    for idx in bond_indices:
        bond = mol.GetBondWithIdx(int(idx))
        if bond.GetBondType() != Chem.BondType.AROMATIC:
            continue
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        touched.add(bond.GetBeginAtomIdx())
        touched.add(bond.GetEndAtomIdx())
        changed += 1
    for atom_idx in touched:
        atom = mol.GetAtomWithIdx(atom_idx)
        if not atom.GetIsAromatic():
            continue
        if any(b.GetBondType() == Chem.BondType.AROMATIC for b in atom.GetBonds()):
            continue
        atom.SetIsAromatic(False)
    return changed


def _resolve_partial_rings(mol) -> int:
    """Resolve every partial-aromatic ring as a whole; return how many.

    Wf-A113 (T83).  A ring whose bonds are *partly* in the Wiberg aromatic
    band cannot be kekulized: RDKit needs every aromatic atom to sit in a
    CLOSED conjugated system, and a path is not one.  The ring is the unit,
    so each one is decided whole — promoted to aromatic when at least half
    its bonds already are, demoted otherwise.

    Promotion is the right call for the dominant case.  A delocalised ring's
    C-C bonds sit near 1.4 and land in the aromatic band, while its C-N /
    C-O / C-S bonds sit near 1.2 and land in SINGLE, so pyridine arrives as
    4 aromatic + 2 single and furan as 4 aromatic + 1 single.  Demoting those
    turns pyridine into piperidine.  MEASURED against the corpus's own
    reference SMILES on 400 X1 complexes: demote-only reproduced the
    reference aromatic-ring count on 19/400, this rule on 66/400
    (metrics/by_metric/wfa113_t83_truth_roundtrip.json).
    """
    from rdkit import Chem

    n = 0
    for idxs in _ring_bond_indices(mol):
        types = [mol.GetBondWithIdx(i).GetBondType() for i in idxs]
        n_arom = sum(1 for t in types if t == Chem.BondType.AROMATIC)
        if not (0 < n_arom < len(idxs)):
            continue
        if n_arom * 2 >= len(idxs):
            _promote_bond_set(mol, idxs)
        else:
            _demote_bond_set(mol, idxs)
        n += 1
    return n


def _promote_bond_set(mol, bond_indices) -> int:
    """Rewrite a set of ring bonds to AROMATIC and flag their atoms."""
    from rdkit import Chem

    changed = 0
    for idx in bond_indices:
        bond = mol.GetBondWithIdx(int(idx))
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            continue
        bond.SetBondType(Chem.BondType.AROMATIC)
        bond.SetIsAromatic(True)
        bond.GetBeginAtom().SetIsAromatic(True)
        bond.GetEndAtom().SetIsAromatic(True)
        changed += 1
    return changed


def _aromatic_components(mol) -> List[List[int]]:
    """Connected components of the aromatic-bond subgraph, smallest first.

    Wf-A113 (T83).  A ring is not a safe unit of repair:
    ``Chem.GetSSSR`` returns a *smallest* set of smallest rings, so a ring
    fused into a larger system can be absent from it, and a residual aromatic
    bond then survives every demotion.  MEASURED on ERUFII / JOFFUJ / ZAZBUB
    (tmQM X1): 5-10 aromatic bonds still present after the SSSR-driven loop
    had demoted everything it could see, and the kekulizer still refused.
    Components are exhaustive by construction — demoting all of them is
    exactly ``_demote_all_aromatic`` — so the loop below always terminates in
    a sanitizable state if one exists.
    """
    from rdkit import Chem

    arom = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.AROMATIC
    ]
    if not arom:
        return []
    parent = {i: i for i in arom}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_atom: Dict[int, int] = {}
    for idx in arom:
        bond = mol.GetBondWithIdx(idx)
        for atom_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
            if atom_idx in by_atom:
                ra, rb = find(by_atom[atom_idx]), find(idx)
                if ra != rb:
                    parent[ra] = rb
            else:
                by_atom[atom_idx] = idx
    groups: Dict[int, List[int]] = {}
    for idx in arom:
        groups.setdefault(find(idx), []).append(idx)
    out = list(groups.values())
    out.sort(key=len)
    return out


def _RDKIT_AROMATIC():
    """``Chem.BondType.AROMATIC`` without importing RDKit at module scope."""
    from rdkit import Chem

    return Chem.BondType.AROMATIC


def _try_sanitize(mol) -> Optional[str]:
    """Sanitize ``mol`` in place; return ``None`` on success, else the error.

    Wf-A113 (T83).  ``SanitizeMol`` MUTATES the molecule and, when the
    kekulizer refuses, rewrites every aromatic bond to SINGLE before raising.
    A repair pass that inspects the molecule after a failed sanitize is
    therefore looking at a molecule RDKit has already stripped.  Callers must
    run each attempt on ``Chem.Mol(base)`` and use this helper rather than
    catching the exception themselves.
    """
    from rdkit import Chem

    try:
        Chem.SanitizeMol(mol)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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
class _ConnectivityCompletion:
    """What the FORCED connectivity completion did (Wf-A133, 2026-10-31).

    Attributes
    ----------
    bonds_added : int
        Bonds the completion ADDED to join two heavy-atom components.
    bonds_demoted : int
        Bond ORDERS the completion LOWERED to free valence budget for one
        of those bonds.  No edge is ever deleted: the published repair is
        "lower the bond order, never delete an edge" (arXiv:2609.08333,
        App. C.2 step 8).
    reasons : Dict[str, int]
        Why a candidate was REFUSED, as a count map.  These are the
        residual: the guarantee could not be met for those pairs.
    components_after : int
        Heavy-atom components left after the completion.
    stranded_atoms : int
        Heavy atoms left OUTSIDE the largest component — the atoms the
        decoder could not rescue.
    """

    bonds_added: int = 0
    bonds_demoted: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    components_after: int = 0
    stranded_atoms: int = 0


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
    #: Wf-A128 (2026-10-31) — bonds RDKit REFUSED while the molecule was
    #: assembled, i.e. the number of predicted bonds that never made it into
    #: the ``RWMol``.  A dropped bond is a SPLIT: the two atoms it would have
    #: joined end up in different fragments, and nothing in the decode said
    #: so.  Before this field the count did not exist anywhere — the
    #: ``except`` in ``_assemble_mol`` was a bare ``continue`` — while every
    #: other fragmentation quantity in the repo (``n_fragments``,
    #: ``largest_fragment_frac``, ``n_frags_raw``, the clash counters) IS
    #: measured in 6+ places.
    #:
    #: 0 means "every predicted bond was accepted", which is the only state
    #: in which the decode's fragment count can be read as the bond head's
    #: intent.  Non-zero means the decode is a FRAGMENTATION of the head's
    #: output, and the reader needs to know that before quoting the fragment
    #: count.
    bonds_dropped_rdkit: int = 0
    #: Wf-A128 — the exception TYPE behind each dropped bond, as a count map
    #: (e.g. ``{"RuntimeError": 2}``).  RDKit 2026.03.6 raises
    #: ``RuntimeError`` for a duplicate bond, an out-of-range atom index and
    #: a self-bond, and ``OverflowError`` for a negative atom index (measured
    #: on a constructed duplicate-bond case).  The map is here so a reader can
    #: tell "RDKit refused a duplicate" from "the decoder produced a
    #: nonsensical index" without re-deriving it.
    bonds_dropped_rdkit_reasons: Dict[str, int] = field(default_factory=dict)
    #: Wf-A133 (2026-10-31) — how many of the ``bonds_added_connectivity``
    #: bonds came from the FORCED completion stage (the one that runs after
    #: the logit-floored pass left the graph fragmented), as opposed to the
    #: logit-floored pass itself.  ``bonds_added_connectivity`` is the total;
    #: this is the split, so a reader can tell how much of the prior's work
    #: was the head's own confidence and how much was the geometry's.
    connectivity_forced_bonds: int = 0
    #: Wf-A133 — bond ORDERS the forced completion LOWERED to free valence
    #: budget for a bridge.  No edge was deleted, so this is a chemistry
    #: change (a double becomes a single) and not a graph change; it is
    #: counted because it is invisible in the bond COUNTS.
    connectivity_bonds_demoted: int = 0
    #: Wf-A133 — why the forced completion could not use a candidate, as a
    #: count map.  A non-empty map is the honest report of the residual:
    #: ``below_force_min_logit`` (the head refused the pair), ``budget_*``
    #: (no bond order could be lowered to free the budget), ``ring_prior``
    #: (the bridge would have closed a ring the caller refused).
    connectivity_forced_reasons: Dict[str, int] = field(default_factory=dict)
    #: Wf-A133 — heavy-atom components left after the forced completion.
    #: 1 means the guarantee held; > 1 means it did not.  0 means the
    #: completion did not run (the connectivity prior was off, or the cloud
    #: had fewer than two heavy atoms).
    connectivity_components_after: int = 0
    #: Wf-A133 — heavy atoms left OUTSIDE the largest component after the
    #: forced completion.  These are the atoms the decoder could NOT rescue:
    #: either no candidate pair existed inside the head's 3.5 A window
    #: (irreducible without changing the sampler or the cutoff) or every
    #: candidate was refused.
    connectivity_stranded_atoms: int = 0
    #: Wf-A155 (2026-11-01) — how many bonds the VALENCE-DEFICIT PROMOTION
    #: added on this decode.  Zero unless the caller turned the stage on
    #: (``valence_deficit_promotion=True``), which is the default, so a
    #: reader can tell "the promotion did nothing" from "the promotion did
    #: not run".  A bond added here RAISES an under-valent atom's valence;
    #: it never deletes an edge and never lowers a bond order, so it is
    #: additive on top of ``bonds_accepted``.
    #:
    #: PROJECTED: what this stage does to the pool's fragmentation rate is
    #: **PROJECTED, NOT MEASURED** — Wf-A155 is CPU-only and no re-decode
    #: was run.
    bonds_added_valence_deficit: int = 0


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
        Number of message-passing iterations.  K=3 is **this repo's** own
        choice — the published paper states no layer count, so it is not
        a figure taken from the paper.  (Corrected 2026-10-31, Wf-A143 —
        this line previously read "Wang & Dokholyan 2025 recommend K=3 for
        drug-like ligands"; see
        `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.)
    distance_cutoff : float, default 3.5 Å
        Pairs further apart than this are immediately classified as
        ``NO_BOND`` (the head's own cutoff; not configurable per-call).

        **Declared deviation from the paper (2026-10-31, Wf-A143).** The
        published YuelBond paper uses a **3 Å** candidate-edge cutoff;
        this repo uses **3.5 Å**.  Left unchanged on purpose — it is a
        decode-behaviour parameter and re-validating it needs a GPU run,
        which the current CPU-only mandate forbids.  See
        `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.
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
        # Wf-A133 (2026-10-31) — the FORCED completion of the connectivity
        # prior, and the floor it uses.
        #
        # `_bridge_components` bridges a cross-component pair only when the
        # head's strongest non-NO_BOND logit clears
        # `connectivity_prior_min_logit` (0.0) AND both endpoints still have
        # valence budget.  Those two gates are INDEPENDENT and each one on
        # its own leaves the molecule fragmented — MEASURED on CPU by the
        # Wf-A133 audit (/tmp/wfa133/a2-bridge-veto-audit.md, §2.2): a head
        # that calls the cross pair NO_BOND bridges 0; a head that calls it
        # SINGLE but whose endpoints are valence-saturated also bridges 0.
        # A pair the head ARGMaxes as NO_BOND has, by construction, all four
        # bond-class logits below the NO_BOND logit, so with a 0.0 floor the
        # prior can only ever resurrect a bond the head already proposed and
        # the selector dropped — it is a budget-recovery pass, not a prior
        # over connectivity.
        #
        # The forced stage runs AFTER that pass, on the molecules it left
        # fragmented, and it is the only part of the decoder that gives a
        # GUARANTEE: one heavy-atom component whenever the geometry permits
        # it.  See `_force_connectivity` for exactly what it may and may not
        # do — it draws its candidates from the head's OWN post-veto
        # `edge_index`, so it can never bridge a steric collapse and can
        # never invent a pair outside the 3.5 A window.
        #
        # `connectivity_force_min_logit` is the floor for the forced stage
        # and is SEPARATE from `connectivity_prior_min_logit` on purpose:
        # moving the latter would change the meaning of the existing knob
        # (51 configs pin it at 0.0).  -2.0 is the value the audit proposed
        # for the widened floor; it is a PRIOR, not a tuned value, and it is
        # a knob so the A/B can move it.
        connectivity_force: bool = True,
        connectivity_force_min_logit: float = -2.0,
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
        # Wf-A155 (2026-11-01) — the VALENCE-DEFICIT PROMOTION, default OFF.
        #
        # The rule: when an atom is UNDER-VALENT after the current bond
        # assignment, promote an eligible CANDIDATE bond so the atom's
        # valence rises toward its target.  It is the mirror image of the
        # forced-completion stage (`_force_connectivity`), which LOWERS a
        # bond order to free budget; this one RAISES an atom's valence by
        # ADDING a bond.
        #
        # WHY IT IS OPT-IN: it can promote a candidate the head called
        # NO_BOND, so it overrides the model's own refusal.  That is a
        # hypothesis about the fragmentation failure mode, and this wave is
        # CPU-only (no re-decode), so the default path must stay
        # bit-identical and the effect must be labelled PROJECTED.
        #
        # `valence_deficit_min_logit` is the same kind of deference knob as
        # `connectivity_force_min_logit`: a candidate whose strongest
        # bond-class logit is below this floor is left alone.  -2.0 is the
        # forced stage's value, reused so the two stages defer to the head
        # by the same amount; it is a PRIOR, not a tuned value.
        #
        # `valence_deficit_max_promotions_per_atom` bounds how many bonds a
        # single under-valent atom may gain (default 1).  One bond is enough
        # to attach a stranded atom; a metal with a larger coordination
        # deficit would want more, and that is a decision the caller has to
        # make explicitly rather than inherit.
        valence_deficit_promotion: bool = False,
        valence_deficit_min_logit: float = -2.0,
        valence_deficit_max_promotions_per_atom: int = 1,
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
        # Wf-A133 — the forced completion stage and its floor.  It runs only
        # when `connectivity_prior` is ON, so `connectivity_prior=False`
        # still reproduces the pre-D2 decode bit-exactly.
        self.connectivity_force = bool(connectivity_force)
        self.connectivity_force_min_logit = float(connectivity_force_min_logit)
        # Wf-A155 (2026-11-01) — the valence-deficit promotion and its two
        # knobs.  Default OFF: the decode is bit-identical to the pre-Wf-A155
        # path unless the caller asks for it.
        self.valence_deficit_promotion = bool(valence_deficit_promotion)
        self.valence_deficit_min_logit = float(valence_deficit_min_logit)
        self.valence_deficit_max_promotions_per_atom = max(
            0, int(valence_deficit_max_promotions_per_atom)
        )
        #: Wf-A155 — how many bonds the valence-deficit promotion added on
        #: the LAST decode.  Read by :meth:`decode` into
        #: :attr:`YuelBondDecodeResult.bonds_added_valence_deficit`.
        self.last_valence_deficit_promoted: int = 0
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
        #: Wf-A128 (2026-10-31) — the last decode's RDKit bond-drop count and
        #: the exception types behind it.  Set by :meth:`_assemble_mol` and
        #: read (once) by :meth:`decode` to populate
        #: :attr:`YuelBondDecodeResult.bonds_dropped_rdkit`.  Initialised here
        #: so a decode that never reaches the assembler reads 0, not an
        #: AttributeError.
        self._last_bonds_dropped: int = 0
        self._last_bonds_dropped_reasons: Dict[str, int] = {}

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
        # Step 2a (Wf-A155, 2026-11-01): the VALENCE-DEFICIT PROMOTION.
        #
        # It runs right after the selector and BEFORE the connectivity prior,
        # because it changes the valence accounting the prior then reads: an
        # atom that the promotion attached is no longer stranded, so the
        # prior sees one fewer fragment to bridge.  Running it first also
        # matches the direction of the rule — a STRANDED atom is exactly the
        # case the promotion exists to rescue, and the prior would otherwise
        # reach it by adding a bond the head declined.
        #
        # No-op when `valence_deficit_promotion` is False (the default), so
        # the pre-Wf-A155 decode is reproduced bit-exactly.
        n_valence_promotions = 0
        if self.valence_deficit_promotion:
            n_valence_promotions = self._promote_valence_deficit_bonds(
                cloud,
                yres,
                bond_orders,
                dative_bonds,
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
        # Step 2c (Wf-A133, 2026-10-31): the FORCED completion of the
        # connectivity prior.  It runs AFTER the logit-floored pass above,
        # and only on the molecules that pass left fragmented.
        #
        # WHY A SECOND STAGE AND NOT A WIDER FLOOR: the two gates of
        # `_bridge_components` (the head's logit floor and the per-atom
        # valence budget) are INDEPENDENT, so widening only the floor still
        # leaves every molecule whose endpoints are valence-saturated
        # fragmented.  MEASURED on CPU by the Wf-A133 audit (§2.2, cases B
        # and C): one gate opened, one gate closed, two fragmented molecules
        # out of two.  This stage is the only part of the decoder that
        # states a GUARANTEE, and it states it over the reachable set: it
        # never leaves a molecule fragmented while a candidate pair exists
        # inside the head's own 3.5 A window.
        #
        # WHY IT IS SAFE: it draws its candidates from the head's OWN
        # post-veto `edge_index`, so (i) it can never bridge a pair the
        # steric-collapse rule rejected, and (ii) it can never invent a
        # pair outside the head's 3.5 A candidate window.  It only ADDS
        # bonds, and the only bonds it ever alters are ones it lowers in
        # ORDER (never deletes) to free valence budget.
        forced = _ConnectivityCompletion()
        if self.connectivity_prior and self.connectivity_force:
            forced = self._force_connectivity(
                cloud,
                yres,
                bond_orders,
                dative_bonds,
            )
            n_bridges += forced.bonds_added
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
            bonds_accepted=n_kept + n_bridges + n_valence_promotions,
            bonds_rejected_overvalent=n_overvalent,
            bonds_added_connectivity=n_bridges,
            # Wf-A155 — the bonds the valence-deficit promotion added.  Zero
            # on the default path (the stage is OFF), so the rest of the
            # result is bit-identical to the pre-Wf-A155 decode.
            bonds_added_valence_deficit=int(n_valence_promotions),
            # Wf-A113 — read from the head that produced yres.  A head that
            # does not implement the rule reports 0, which is the truth for a
            # pre-A113 checkpoint loaded with clash_reject off.
            #
            # Wf-A128 (2026-10-31) — the count now comes from the RESULT
            # (``yres.clash_rejected_pairs``), which :meth:`featurise` read
            # off the head IMMEDIATELY after the forward that produced THIS
            # decode.  The old read went to the head's mutable
            # ``last_clash_rejected_pairs`` attribute, so a decode whose
            # forward returned early (or never wrote the attribute) reported
            # whatever the PREVIOUS decode had left there.  MEASURED before
            # this change: a 5-atom cloud with 1 rejection followed by a
            # 1-atom cloud reported ``clash_rejected_pairs == 1`` for the
            # 1-atom cloud, which had zero rejections.
            clash_rejected_pairs=int(getattr(yres, "clash_rejected_pairs", 0) or 0),
            # Wf-A128 — how many of the predicted bonds RDKit refused while
            # the molecule was assembled.  Non-zero means the decode is a
            # FRAGMENTATION of the head's output: the bonds the head voted
            # for and the selector accepted under the valence budget did not
            # all survive assembly.
            bonds_dropped_rdkit=int(self._last_bonds_dropped),
            bonds_dropped_rdkit_reasons=dict(self._last_bonds_dropped_reasons),
            # Wf-A133 — the split of the connectivity prior's work: the
            # logit-floored pass vs the forced completion, plus the honest
            # report of what the forced completion could not reach.
            connectivity_forced_bonds=int(forced.bonds_added),
            connectivity_bonds_demoted=int(forced.bonds_demoted),
            connectivity_forced_reasons=dict(forced.reasons),
            connectivity_components_after=int(forced.components_after),
            connectivity_stranded_atoms=int(forced.stranded_atoms),
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
        # Wf-A113 (T81) — aromatic bonds already accepted at each atom, so the
        # third one at a carbon is charged the Kekule-single 1.0 instead of the
        # two-bond average 1.5 (see ``_endpoint_valence_cost`` rule 4).
        arom_kept: List[int] = [0] * n_atoms
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
            # Wf-A113 (T77/T78/T79, 2026-10-28) — per-endpoint cost.  The three
            # rules (dative at the donor, a metal's budget counts bonds, an
            # aromatic bond at a heteroatom is Kekule-single) live in
            # ``_endpoint_valence_cost`` together with their measurements,
            # because this block and the one in ``_bridge_components`` used to
            # be hand-copied and each unit fix landed in only one of them.
            zi = int(z_list[i])
            zj = int(z_list[j])
            cost_i = _endpoint_valence_cost(
                order, zi, zj, self._is_metal_z, arom_kept[i]
            )
            cost_j = _endpoint_valence_cost(
                order, zj, zi, self._is_metal_z, arom_kept[j]
            )
            cap_i = remaining[i]
            cap_j = remaining[j]
            # Wf-A49 arch_fix_7 — compare against the actual remaining
            # budget, not a hard-coded 1.0.  The hard-coded 1.0 was an
            # integer-overflow fallback that masked the cap mismatch for
            # aromatic bonds (a cap of 1.5 sits between 1.0 and 2.0; the
            # prior ``< 1.0`` check let every aromatic bond through
            # regardless of remaining budget).
            if cost_i > cap_i or cost_j > cap_j:
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
            # is still re-exported from this module (the tests import it here)
            # and still describes the ideal shape, but nothing in this decoder
            # reads it: there is no angular prior in this file, and
            # ``max_bonds`` no longer gates bond count.  Pd(5th bond) and
            # Pt(5th bond) are still refused -- by the valence cap.
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
            remaining[i] -= cost_i
            remaining[j] -= cost_j
            if int(order) == BOND_AROMATIC:
                arom_kept[i] += 1
                arom_kept[j] += 1
            n_kept += 1
            # Wf-A113 (T86, 2026-10-29) — the Wf-A55 anti-aromatic demotion
            # that used to run here has been REMOVED.  It demoted every other
            # bond of a ``ring_size % 4 == 0`` all-aromatic ring to SINGLE,
            # on the premise that RDKit cannot kekulize a 4n ring.  MEASURED
            # on RDKit 2026.03.6: 4-, 8-, 10- and 12-membered all-aromatic
            # monocycles all kekulize (C1=CC=C1, C1=CC=CC=CC=C1, ...); only
            # ODD sizes fail, and the rule never touched those.  It also
            # mutated ``bond_orders`` in place, which is the list returned to
            # the caller, and did not credit ``remaining[]`` for the 0.5 it
            # gave back per demoted bond -- so a later real bond could be
            # refused.  The six-metal cohort contains 0 complexes with a 4n
            # all-aromatic ring in its true bond graph, so the rule only ever
            # fired on a head hallucination, where it made the reported bond
            # composition disagree with what the head predicted.
            # (metrics/by_metric/wfa113_t83_truth_roundtrip.json)
            # Dative flag — same convention as BondAwareDecoder.
            # (``zi`` / ``zj`` were resolved above, when the metal's bond was
            # charged at unit cost.)
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
        #
        # Wf-A113 (T81) — replay in the selector's own acceptance order so the
        # per-atom aromatic counter reproduces what the selector charged.
        #
        # Wf-A133 (2026-10-31) — the replay itself now lives in
        # ``_replay_remaining_budget``, shared with the forced completion
        # stage.  It used to be written out here a second time, and the
        # T77/T78 accounting bugs are what a second copy costs: the fix
        # landed in one copy and not the other.
        caps = [self._cap_for_z(int(z)) for z in z_list]
        remaining, _arom_kept = _replay_remaining_budget(
            bond_orders, z_list, caps, self._is_metal_z
        )

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
            # A bridge is always added as BOND_SINGLE, so the same
            # per-endpoint rule applies: the ligand side of a metal bond
            # needs no budget, the metal side needs one.
            _need_i = _endpoint_valence_cost(
                BOND_SINGLE, int(z_list[i]), int(z_list[j]), self._is_metal_z
            )
            _need_j = _endpoint_valence_cost(
                BOND_SINGLE, int(z_list[j]), int(z_list[i]), self._is_metal_z
            )
            if remaining[i] < _need_i or remaining[j] < _need_j:
                continue
            bond_orders.append((i, j, BOND_SINGLE))
            remaining[i] -= _need_i
            remaining[j] -= _need_j
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
    # Wf-A133 — the FORCED completion of the connectivity prior.
    # ------------------------------------------------------------------
    def _force_connectivity(
        self,
        cloud: AtomCloud,
        yres: YuelBondResult,
        bond_orders: List[Tuple[int, int, int]],
        dative_bonds: List[Tuple[int, int]],
    ) -> _ConnectivityCompletion:
        """Join the heavy-atom components the logit-floored prior left apart.

        Why this exists
        ---------------
        MEASURED on CPU by the Wf-A133 decode-path audit
        (``/tmp/wfa133/a2-bridge-veto-audit.md`` §2.2): two carbon fragments
        2.9 A apart, one cross-component candidate pair, and a random-init
        decoder.

        ==========  ==========================================  ==========
        head says   result                                     gate
        ==========  ==========================================  ==========
        SINGLE      1 bridge, 1 component                       —
        NO_BOND     0 bridges, 2 components                     logit floor
        SINGLE,     0 bridges, 2 components                     valence budget
        saturated
        ==========  ==========================================  ==========

        The two gates are INDEPENDENT, so widening one leaves the other
        standing, and a connectivity prior that can still hand back a
        fragmented molecule is a budget-recovery pass with a connectivity
        name.  This stage is the part of the decoder that states a
        guarantee, and the guarantee is deliberately narrow:

        **One heavy-atom component, whenever the geometry permits it.**

        The carve-out is the same one the training-side connectivity loss
        states in its own docstring
        (``assemble_connectivity.py``: "The head only emits candidate pairs
        within ``cutoff`` (3.5 A).  An atom with no candidate edge at all
        ... has an all-zero adjacency row, so its penalty is a constant and
        its gradient is exactly zero"): an atom with NO candidate pair
        inside the head's 3.5 A window cannot be reached by ANY decode-side
        change.  Those atoms are counted in
        :attr:`YuelBondDecodeResult.connectivity_stranded_atoms` rather
        than silently dropped.

        What it may do
        --------------
        * ADD a ``BOND_SINGLE`` between two heavy atoms in DIFFERENT
          components.  Nothing is ever deleted — no bond the selector kept
          disappears, and no edge is removed to make room.
        * LOWER the ORDER of one existing bond (double -> single, triple ->
          double) to free exactly 1.0 of valence budget at an endpoint
          that has none.  This is the published repair: "bond orders are
          lowered one level at a time ... No edge is deleted" (arXiv:
          2609.08333, App. C.2 step 8).  The number of lowered bonds is
          reported in :attr:`connectivity_bonds_demoted` because a lowered
          order is invisible in the bond counts.

        What it may NOT do
        ------------------
        * It may not bridge a pair the steric-collapse rule rejected.  Its
          candidate pool IS the head's post-veto ``edge_index``, so the
          clash veto stays a hard veto over every bond this decoder
          creates.  The veto is deliberately NOT reordered or scoped away:
          the audit MEASURED that it denies the bridge almost nothing (12 of
          16 fragmented clouds had their closest inter-component pair in
          the post-veto pool) and that it fires at 0.93-1.79 A against a
          MEDIAN closest inter-component distance of 2.75 A, i.e. it
          vetoes only pairs that are collapsed, which are exactly the pairs
          that must not become bonds.  Reordering it would buy nothing and
          would remove a real safety check.
        * It may not invent geometry.  Every candidate pair came from the
          head's own 3.5 A window.
        * It may not use a pair the head is CONFIDENT about.  The floor
          ``connectivity_force_min_logit`` (-2.0) keeps a head that ranks
          every bond class far below NO_BOND in charge of its own refusal,
          so the deference knob of the prior still works.
        * It may not close a ring.  Every forced bridge joins two DIFFERENT
          components, so the endpoints have no path between them and the
          bond closes nothing.  The ring-size prior is a gate on a bond
          INSIDE a growing component, which is the selector's case; this
          stage's bridges are structurally ring-free.

        Which pair is chosen
        --------------------
        The CLOSEST cross-component pair, with the head's logit breaking
        ties.  The head's logit is the prior's ranking because it is the
        model's ranking, but it is NOT used as the primary key here: the
        forced stage exists precisely for the pairs the head DECLINED to
        propose, and for those the head's opinion is not a connectivity
        signal (the Wf-A133 causal audit ranked the head's bond placement
        as a DOMINANT cause of the fragmentation, R2).  Physical proximity
        is the remaining evidence, and picking the closest pair also
        minimises the length of the bond that gets added.
        """
        stats = _ConnectivityCompletion()
        z_list = [int(z) for z in cloud.atomic_numbers.tolist()]
        n_atoms = len(z_list)
        # Heavy atoms only: H is a valence detail, not a fragment.
        heavy = [i for i in range(n_atoms) if z_list[i] != 1]
        if len(heavy) < 2:
            return stats

        # --- union-find over the bonds produced so far ---------------------
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

        def _census() -> Tuple[int, int]:
            """(components, atoms outside the largest component)."""
            groups: Dict[int, int] = {}
            for i in heavy:
                root = _find(i)
                groups[root] = groups.get(root, 0) + 1
            if not groups:
                return 0, 0
            largest = max(groups.values())
            return len(groups), len(heavy) - largest

        if n_components <= 1:
            stats.components_after, stats.stranded_atoms = _census()
            return stats

        edge_index = yres.edge_index
        if edge_index.shape[1] == 0:
            # The head scored nothing at all: there is no candidate pair to
            # bridge with, and the molecule stays as fragmented as it is.
            stats.components_after, stats.stranded_atoms = _census()
            return stats

        # --- candidate pairs, drawn from the head's OWN post-veto list ------
        logits = yres.logits.detach()
        best_logit = logits[:, 1:].max(dim=-1).values.tolist()
        ei_all = edge_index[0].tolist()
        ej_all = edge_index[1].tolist()
        positions = cloud.positions.detach()
        # The head's own confidence for every pair, so the demotion step can
        # pick the bond the head was LEAST sure about.
        pair_logit: Dict[Tuple[int, int], float] = {}
        candidates: List[Tuple[float, float, int, int]] = []
        n_below_floor = 0
        for k in range(edge_index.shape[1]):
            i = int(ei_all[k])
            j = int(ej_all[k])
            if i == j or i >= n_atoms or j >= n_atoms:
                continue
            if z_list[i] == 1 or z_list[j] == 1:
                continue
            key = (i, j) if i < j else (j, i)
            score = float(best_logit[k])
            pair_logit.setdefault(key, score)
            if key in have:
                continue
            if score < self.connectivity_force_min_logit:
                n_below_floor += 1
                continue
            try:
                dist = float(torch.norm(positions[j] - positions[i]))
            except Exception:
                dist = float("inf")
            # (distance asc, logit desc, i, j) -> the closest pair first,
            # the head's logit breaking ties.
            candidates.append((dist, -score, i, j))
        if n_below_floor:
            stats.reasons["below_force_min_logit"] = n_below_floor
        candidates.sort()

        # --- remaining valence budget ---------------------------------------
        caps = [self._cap_for_z(int(z)) for z in z_list]
        remaining, _arom_kept = _replay_remaining_budget(
            bond_orders, z_list, caps, self._is_metal_z
        )
        # Wf-A111 (MOD4) — the ring gate does NOT apply here, and cannot:
        # every forced bridge joins two DIFFERENT components, so the two
        # endpoints have no path between them and the bond closes no ring.
        # (The ring-size prior is a gate on a bond INSIDE a growing
        # component, which is the selector's case; the prior's bridges and
        # these forced bridges are structurally ring-free.)
        #
        # Wf-A92 — same truth-first dative rule as `_select_bonds`.
        dative_truth = _dative_truth_pairs(cloud)

        n_added = 0
        n_demoted = 0
        for _dist, _neg_score, i, j in candidates:
            if n_components <= 1:
                break
            if _find(i) == _find(j):
                continue
            # A bridge is always added as BOND_SINGLE, so the same
            # per-endpoint rule applies: the ligand side of a metal bond
            # needs no budget, the metal side needs one.
            need_i = _endpoint_valence_cost(
                BOND_SINGLE, z_list[i], z_list[j], self._is_metal_z
            )
            need_j = _endpoint_valence_cost(
                BOND_SINGLE, z_list[j], z_list[i], self._is_metal_z
            )
            if remaining[i] < need_i:
                if self._demote_one_order(
                    bond_orders, i, z_list, pair_logit
                ) is None:
                    stats.reasons["budget_endpoint_i"] = (
                        stats.reasons.get("budget_endpoint_i", 0) + 1
                    )
                    continue
                n_demoted += 1
                remaining, _arom_kept = _replay_remaining_budget(
                    bond_orders, z_list, caps, self._is_metal_z
                )
            if remaining[j] < need_j:
                if self._demote_one_order(
                    bond_orders, j, z_list, pair_logit
                ) is None:
                    stats.reasons["budget_endpoint_j"] = (
                        stats.reasons.get("budget_endpoint_j", 0) + 1
                    )
                    continue
                n_demoted += 1
                remaining, _arom_kept = _replay_remaining_budget(
                    bond_orders, z_list, caps, self._is_metal_z
                )
            bond_orders.append((i, j, BOND_SINGLE))
            remaining[i] -= need_i
            remaining[j] -= need_j
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
        stats.bonds_added = n_added
        stats.bonds_demoted = n_demoted
        stats.components_after, stats.stranded_atoms = _census()
        return stats

    # ------------------------------------------------------------------
    # Wf-A155 (2026-11-01) — the VALENCE-DEFICIT PROMOTION
    # ------------------------------------------------------------------
    def _valence_deficit(
        self,
        z: int,
        assigned: float,
        cap: float,
        n_bonds: int,
    ) -> float:
        """``target - assigned`` for one atom.  ``> 0`` means UNDER-VALENT.

        The target is NOT the covalent-valence cap, because the decode graph
        is HYDROGEN-FREE (the cloud carries no H) and RDKit completes a
        neutral atom's valence with implicit hydrogens:

        * **A metal** takes no implicit hydrogen, so its coordination cap IS
          its valence target: ``deficit = cap - assigned``.  This is the
          metal's coordination deficit (frontier option D, the metal's
          coordination sphere).
        * **Any other atom** that already carries at least one bond is
          complete as far as the heavy-atom graph is concerned: RDKit's
          default valence ``V`` minus the assigned valence is exactly the
          hydrogen count RDKit will add, so ``deficit = V - h - assigned`` is
          ``0`` whenever ``assigned <= V``.  Such an atom is never
          under-valent, and the rule must not touch it — a plain CH3 carbon
          is "1.0 under" its cap of 4, and that 1.0 is the implicit hydrogen.
        * **A stranded atom** (no bond at all) is the exception: it is
          attached to NOTHING, so the hydrogens RDKit would place on it are
          a fiction — RDKit would turn a lone carbon into methane, a second
          fragment.  Its hydrogen allowance is therefore 0 and its deficit is
          its whole default valence.

        MEASURED: ``_rdkit_default_valence`` reads RDKit's own periodic
        table; the rest is arithmetic over the decoder's own bond list.

        PROJECTED: whether closing these deficits reduces the fragmentation
        rate is UNMEASURED under the 0-GPU mandate of this wave.
        """
        if self._is_metal_z(z):
            # A metal takes no implicit hydrogen (RDKit carries no default
            # valence for one), so the coordination cap is its target.
            return float(cap) - float(assigned)
        v = _rdkit_default_valence(z)
        if n_bonds <= 0:
            h = 0.0
        else:
            h = max(0.0, v - float(assigned))
        return (v - h) - float(assigned)

    def _promote_valence_deficit_bonds(
        self,
        cloud: AtomCloud,
        yres: YuelBondResult,
        bond_orders: List[Tuple[int, int, int]],
        dative_bonds: List[Tuple[int, int]],
    ) -> int:
        """Promote candidate bonds so UNDER-VALENT atoms reach their target.

        The rule, in one sentence: **when an atom is under-valent after the
        current bond assignment, promote an eligible candidate bond to raise
        it.**

        It is the mirror image of the forced-completion stage
        (:meth:`_force_connectivity`), which LOWERS a bond order to free
        valence budget.  This one RAISES an atom's valence by ADDING a bond.
        Like that stage, it never DELETES an edge: it only appends.

        Eligibility of a candidate pair
        -------------------------------
        A pair is eligible when ALL of the following hold:

        1. it is in the head's OWN post-veto ``edge_index`` — so it is
           inside the head's own candidate window and survived the
           steric-collapse veto, and no geometry is invented;
        2. neither endpoint is hydrogen;
        3. the pair is not already a bond;
        4. the head's strongest bond-class logit for the pair clears
           ``valence_deficit_min_logit`` (the same deference knob, and the
           same value, as ``connectivity_force_min_logit``);
        5. both endpoints have the valence budget for one more SINGLE bond,
           as the DECODER's own model charges it.

        Which bond is promoted
        -----------------------
        The CLOSEST eligible candidate incident to the under-valent atom, with
        the head's logit breaking ties.  The same ordering
        :meth:`_force_connectivity` uses, and for the same reason: this stage
        exists precisely for the pairs the head DECLINED to propose, and for
        those the head's opinion is not a connectivity signal — physical
        proximity is the remaining evidence.

        Valence accounting
        ------------------
        Every charge here comes from :func:`_endpoint_valence_cost`, the
        decoder's own model.  That means **a metal bond does not cost the
        ligand any valence** (Wf-A113 T79): the donor's budget is untouched
        by its metal bond, exactly as the selector and the forced stage
        already charge it.  A promotion that would raise a ligand's valence
        by way of a metal bond therefore CANNOT close that ligand's
        deficit — the only exception is a STRANDED ligand, whose problem is
        that it is attached to nothing at all, and where attaching it to the
        metal is the whole point.

        MEASURED: the deficits, the eligibility list and the charges are
        computed from the decode's own inputs and are exercised on synthetic
        molecules by ``molmetal/tests/test_wfa155_decoder_bond_candidacy.py``.

        PROJECTED: the effect of this stage on the fragmentation rate is
        **PROJECTED, NOT MEASURED** — this wave is CPU-only and no re-decode
        was run.  The stage is OFF by default.
        """
        self.last_valence_deficit_promoted = 0
        z_list = [int(z) for z in cloud.atomic_numbers.tolist()]
        n_atoms = len(z_list)
        max_per_atom = self.valence_deficit_max_promotions_per_atom
        if n_atoms < 2 or max_per_atom <= 0:
            return 0
        edge_index = yres.edge_index
        if edge_index.shape[1] == 0:
            return 0

        caps = [self._cap_for_z(z) for z in z_list]
        remaining, _arom_kept = _replay_remaining_budget(
            bond_orders, z_list, caps, self._is_metal_z
        )
        have: Set[Tuple[int, int]] = set()
        n_bonds = [0] * n_atoms
        for i, j, _o in bond_orders:
            have.add((i, j) if i < j else (j, i))
            n_bonds[i] += 1
            n_bonds[j] += 1

        # --- candidate pool: the head's OWN post-veto edge list -------------
        logits = yres.logits.detach()
        best_logit = logits[:, 1:].max(dim=-1).values.tolist()
        ei = edge_index[0].tolist()
        ej = edge_index[1].tolist()
        positions = cloud.positions.detach()
        cands: List[Tuple[float, float, int, int]] = []
        for k in range(len(ei)):
            i = int(ei[k])
            j = int(ej[k])
            if i == j or i >= n_atoms or j >= n_atoms:
                continue
            if z_list[i] == 1 or z_list[j] == 1:
                continue
            key = (i, j) if i < j else (j, i)
            if key in have:
                continue
            score = float(best_logit[k])
            if score < self.valence_deficit_min_logit:
                continue
            try:
                dist = float(torch.norm(positions[j] - positions[i]))
            except Exception:
                dist = float("inf")
            # (distance asc, logit desc) -> the closest pair first.
            cands.append((dist, -score, i, j))
        cands.sort()
        if not cands:
            return 0

        dative_truth = _dative_truth_pairs(cloud)
        n_promoted = 0
        used: Set[int] = set()

        # --- the under-valent atoms, most deficient first -------------------
        deficits: List[Tuple[float, int]] = []
        for a in range(n_atoms):
            assigned = caps[a] - remaining[a]
            d = self._valence_deficit(z_list[a], assigned, caps[a], n_bonds[a])
            if d > 1e-9:
                deficits.append((d, a))
        deficits.sort(key=lambda t: (-t[0], t[1]))

        for _deficit, a in deficits:
            promoted_here = 0
            while promoted_here < max_per_atom:
                assigned = caps[a] - remaining[a]
                deficit = self._valence_deficit(
                    z_list[a], assigned, caps[a], n_bonds[a]
                )
                if deficit <= 1e-9:
                    break
                chosen: Optional[Tuple[int, int, int, float, float]] = None
                for idx, (_dist, _neg, i, j) in enumerate(cands):
                    if idx in used:
                        continue
                    if a != i and a != j:
                        continue
                    other = j if i == a else i
                    cost_a = _endpoint_valence_cost(
                        BOND_SINGLE, z_list[a], z_list[other], self._is_metal_z
                    )
                    # A metal bond costs the LIGAND nothing (Wf-A113 T79), so
                    # it cannot RAISE a non-stranded ligand's valence.  It is
                    # still allowed for a STRANDED atom, whose problem is
                    # that it is attached to nothing at all.
                    if cost_a <= 0.0 and n_bonds[a] > 0:
                        continue
                    cost_other = _endpoint_valence_cost(
                        BOND_SINGLE, z_list[other], z_list[a], self._is_metal_z
                    )
                    if remaining[a] < cost_a or remaining[other] < cost_other:
                        continue
                    chosen = (idx, i, j, cost_a, cost_other)
                    break
                if chosen is None:
                    break
                idx, i, j, cost_a, cost_other = chosen
                other = j if i == a else i
                used.add(idx)
                bond_orders.append((a, other, BOND_SINGLE))
                remaining[a] -= cost_a
                remaining[other] -= cost_other
                n_bonds[a] += 1
                n_bonds[other] += 1
                have.add((a, other) if a < other else (other, a))
                # Mirror `_select_bonds` / `_force_connectivity`'s dative
                # convention for a metal-donor pair.
                if self._is_dative_pair_at(
                    cloud, a, other, z_list[a], z_list[other], dative_truth
                ):
                    if self._is_metal_z(z_list[a]) and not self._is_metal_z(
                        z_list[other]
                    ):
                        dative_bonds.append((other, a))
                    else:
                        dative_bonds.append((a, other))
                promoted_here += 1
                n_promoted += 1
        self.last_valence_deficit_promoted = n_promoted
        return n_promoted

    def _demote_one_order(
        self,
        bond_orders: List[Tuple[int, int, int]],
        endpoint: int,
        z_list: Sequence[int],
        pair_logit: Dict[Tuple[int, int], float],
    ) -> Optional[int]:
        """Lower ONE bond order at ``endpoint`` to free 1.0 of valence budget.

        Returns the new bond order (``BOND_DOUBLE`` for a triple,
        ``BOND_SINGLE`` for a double) or ``None`` when no bond at that
        endpoint can be lowered.

        Which bonds are eligible, and why the others are not
        ------------------------------------------------------
        * ``BOND_DOUBLE`` and ``BOND_TRIPLE`` only.  Lowering an AROMATIC
          bond inside a ring leaves a ring RDKit cannot kekulize, and the
          repository's own fallback then strips the AROMATICITY of the WHOLE
          molecule (``_demote_all_aromatic``) — a far larger chemistry
          change than this stage is allowed to make, and one that would
          quietly rewrite every ring in the pool.
        * Both endpoints must be NON-metals.  A metal's budget is a BOND
          COUNT, so lowering the order of one of its bonds frees nothing
          (Wf-A113 T77); and a ligand's budget does not pay for its metal
          bond at all (Wf-A113 T79), so lowering that bond frees nothing
          either.
        * A bond whose endpoints are both saturated with SINGLE bonds has
          no order left to lower, and lowering one would mean DELETING an
          edge — which the published repair forbids and which this
          repository measures as a silent split (``bonds_dropped_rdkit``).
          Such an endpoint is reported as ``budget_endpoint_*`` instead.

        Among the eligible bonds, the one the head was LEAST sure about is
        chosen (the lowest bond-class logit), because lowering that one
        disturbs the head's own output the least.  Ties break on the list
        index, so the choice is deterministic.
        """
        best: Optional[Tuple[Tuple[float, int], int, int, int]] = None
        for idx, (i, j, order) in enumerate(bond_orders):
            if endpoint != i and endpoint != j:
                continue
            if int(order) not in (BOND_DOUBLE, BOND_TRIPLE):
                continue
            other = j if i == endpoint else i
            if self._is_metal_z(int(z_list[endpoint])) or self._is_metal_z(
                int(z_list[other])
            ):
                continue
            key = (i, j) if i < j else (j, i)
            rank = (float(pair_logit.get(key, 0.0)), int(idx))
            if best is None or rank < best[0]:
                best = (rank, int(idx), int(i), int(j))
        if best is None:
            return None
        _rank, idx, i, j = best
        new_order = BOND_DOUBLE if int(bond_orders[idx][2]) == BOND_TRIPLE else BOND_SINGLE
        bond_orders[idx] = (i, j, new_order)
        return int(new_order)

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

        # Wf-A128 — reset the drop counters at the TOP so that a decode which
        # returns early (before the bond loop) can never inherit the PREVIOUS
        # decode's count.
        self._last_bonds_dropped = 0
        self._last_bonds_dropped_reasons: Dict[str, int] = {}

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
        # Wf-A128 (2026-10-31) — COUNT the bonds RDKit refuses.
        #
        # A dropped bond is not a cosmetic event: it is the ONE place in the
        # decode path where the decode itself can split one molecule into
        # two.  The bond was predicted, the head voted for it, the selector
        # accepted it under the valence budget, and then the assembler threw
        # it away — leaving two fragments where the head meant one molecule.
        # Before this change the ``except`` was a bare ``continue``: no
        # counter, no log, no field, no trace anywhere.
        #
        # The catch is DELIBERATELY LEFT WIDE.  Measured on RDKit 2026.03.6
        # (see molmetal/tests/test_wfa128_decode_bond_drop_counter.py):
        # a duplicate bond, an out-of-range atom index and a self-bond all
        # raise ``RuntimeError``; a NEGATIVE atom index raises
        # ``OverflowError`` (Python's int -> unsigned conversion, not
        # RDKit).  Narrowing the catch to ``(RuntimeError, OverflowError)``
        # would therefore change behaviour: ``int(src)`` on a non-numeric
        # value raises ``ValueError``/``TypeError``, which the wide catch
        # swallows (the bond is dropped) and a narrow catch would PROPAGATE
        # (the whole decode raises).  A narrowed catch that starts
        # propagating a previously-swallowed error IS a behaviour change, so
        # the wide catch stays and the exception TYPE is recorded instead —
        # which is strictly more information than the old ``continue`` gave.
        n_bonds_dropped = 0
        dropped_reasons: Dict[str, int] = {}
        for src, dst, order in bond_orders:
            bt = bt_map.get(int(order), Chem.BondType.SINGLE)
            try:
                rw.AddBond(int(src), int(dst), bt)
            except Exception as exc:
                # Skip invalid bonds (RDKit raises on duplicate or
                # out-of-range atoms).
                n_bonds_dropped += 1
                _exc_name = type(exc).__name__
                dropped_reasons[_exc_name] = dropped_reasons.get(_exc_name, 0) + 1
                continue
        # Stashed on the decoder so ``decode()`` can put the count on the
        # result.  Reset at the TOP of the method (not here) so a decode that
        # returns early can never inherit the previous decode's count.
        self._last_bonds_dropped = n_bonds_dropped
        self._last_bonds_dropped_reasons = dropped_reasons
        try:
            mol = rw.GetMol()
        except Exception as exc:
            _dropped = DecodedMol(
                error=f"RWMol.GetMol failed: {exc}",
                n_atoms=cloud.positions.shape[0],
                dative_source=dative_source,
                dative_manufactured=int(dative_manufactured),
            )
            if n_bonds_dropped:
                _dropped.warnings.append(f"bonds_dropped_rdkit:{n_bonds_dropped}")
            return _dropped
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
        # Wf-A113 (T83) — snapshot the molecule BEFORE the first sanitize.
        # ``Chem.SanitizeMol`` mutates in place and, on a kekulize failure,
        # rewrites every aromatic bond to SINGLE while leaving the atom
        # aromatic flags set.  A repair chain that copies ``mol`` after the
        # exception therefore inherits a molecule with no aromatic bonds but
        # twelve aromatic atoms, finds nothing to demote, and reports the
        # failure as unfixable — which is how 63.7% of the cohort ended up in
        # the demote-everything bucket.
        base = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol)
            sanitized = True
            smiles = Chem.MolToSmiles(mol)
        except Exception as exc:
            sanitized = False
            smiles = ""
            sanitize_error = f"{type(exc).__name__}: {exc}"
            # Wf-A113 (T83, 2026-10-29) — the repair chain below used to run
            # on ``mol`` AFTER ``SanitizeMol`` had already raised, and that is
            # a no-op: RDKit's kekulizer rewrites every aromatic bond to
            # SINGLE in place BEFORE it throws, so ``_demote_all_aromatic``
            # found nothing to demote and the "repair" merely re-sanitised a
            # molecule RDKit had already stripped.  MEASURED on the corpus's
            # own bond classes: 63.7% of the six-metal cohort took that path,
            # i.e. every aromatic ring was destroyed on molecules the head had
            # reproduced exactly (metrics/by_metric/wfa113_t83_truth_roundtrip.json).
            #
            # Each attempt now runs on a FRESH COPY, so the pre-failure
            # aromatic state survives for the next attempt to inspect.
            if "Kekulize" in type(exc).__name__:
                # ``state`` is never sanitised: every attempt runs on a copy,
                # and the accepted copy is what the caller gets back.  This
                # keeps the accumulated demotions visible to the next attempt,
                # which a failed in-place ``SanitizeMol`` would erase.
                state = Chem.Mol(base)
                # Wf-A113 (T83) — repair 1: a ring that is only PARTIALLY
                # aromatic (some of its bonds in the Wiberg aromatic band,
                # some not) is not a closed conjugated system, so the
                # kekulizer cannot close it.  Resolve each such ring as a
                # whole: promote it when at least half its bonds are already
                # aromatic, demote it otherwise.
                #
                # Promotion is the right call for the dominant case.  A
                # delocalised ring's C-C bonds sit near 1.4 and land in the
                # aromatic band, while its C-N / C-O / C-S bonds sit near
                # 1.2 and land in SINGLE, so pyridine arrives as 4 aromatic
                # + 2 single and furan as 4 aromatic + 1 single.  Demoting
                # those turns pyridine into piperidine.  MEASURED against the
                # corpus's own reference SMILES on 400 X1 complexes: the
                # demote-only rule reproduced the reference aromatic-ring
                # count on 19/400, this rule on 66/400.
                n_partial = _resolve_partial_rings(state)
                if n_partial:
                    cand = Chem.Mol(state)
                    if _try_sanitize(cand) is None:
                        mol, sanitized, smiles, sanitize_error = (
                            cand, True, Chem.MolToSmiles(cand), None
                        )
                        repair = "kekulize_resolved_partial_rings"
                # Wf-A113 (T83) — repair 2: a FULL aromatic fragment RDKit
                # still refuses.  Demote them one at a time, smallest first,
                # and keep each demotion, so a benzene elsewhere in the
                # molecule survives and only the fragment that actually
                # blocks the kekulizer is lost.
                if not sanitized:
                    for comp in _aromatic_components(state):
                        _demote_bond_set(state, comp)
                        cand = Chem.Mol(state)
                        if _try_sanitize(cand) is None:
                            mol, sanitized, smiles, sanitize_error = (
                                cand, True, Chem.MolToSmiles(cand), None
                            )
                            repair = (
                                "kekulize_resolved_partial_rings"
                                if n_partial
                                else "kekulize_demoted_one_ring"
                            )
                            break
                # Wf-A84 (2026-10-19) — last resort: keep the molecule, drop
                # the aromaticity.  Everything local has been tried; a decode
                # saved this way is valid, not correct, which is why the
                # trainer counts it in `ok_repaired:kekulize_demoted_all`.
                if not sanitized:
                    cand = _demote_all_aromatic(Chem.Mol(state))
                    err = _try_sanitize(cand)
                    if err is None:
                        mol, sanitized, smiles, sanitize_error = (
                            cand, True, Chem.MolToSmiles(cand), None
                        )
                        repair = "kekulize_demoted_all"
                    else:
                        sanitize_error = err
        _decoded = DecodedMol(
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
        # Wf-A128 (2026-10-31) — the PER-MOLECULE decode record carries the
        # drop count as a ``key:value`` warning, the same convention the
        # trainer already uses for ``dative_source:<src>``.  The count also
        # rides on :attr:`YuelBondDecodeResult.bonds_dropped_rdkit`; this
        # copy is on the object a per-molecule consumer already holds.
        if n_bonds_dropped:
            _decoded.warnings.append(f"bonds_dropped_rdkit:{n_bonds_dropped}")
        return _decoded

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
            #
            # Wf-A128 (2026-10-31) — read from the RESULT, not from the
            # head's mutable attribute (see the note in ``decode``).
            clash_rejected_pairs=int(getattr(yres, "clash_rejected_pairs", 0) or 0),
            # Wf-A128 — no molecule was assembled on this path, so no bond
            # could have been dropped by RDKit.  Stated explicitly so a
            # reader does not have to know which branch produced the result.
            bonds_dropped_rdkit=0,
            bonds_dropped_rdkit_reasons={},
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
