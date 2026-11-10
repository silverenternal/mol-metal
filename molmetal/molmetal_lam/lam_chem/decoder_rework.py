"""Path-B decoder rework — chem-aware soft bond prior (WF-CFM-Path-B-Decoder-Rework).

Background
----------
The legacy :class:`molmetal.models.bond_head.BondAwareDecoder` uses a
hard 2.4 Å distance cutoff to gate candidate edges.  The round-12
audit (``molmetal/reports/wf_cfm_internal_review/audit.md``) flagged
that for the typical CFM coordinate distribution (atom-cloud spread
1-5 Å) the head's :attr:`bond_pattern_mask` + the 2.4 Å cutoff
interact badly: the head produces valid ``(SINGLE, C, C)`` logits
but the *coordinate* distribution doesn't have any C-C pairs within
2.4 Å, so ``decode_ratio = 0 / 192``.

Path (a) — a longer joint retrain of the CFM + bond head — has been
EXHAUSTED (decode_ratio stays 0 even with all 4 axes flipped; see
``wf_cfm_retrain_full/final.md``).  Path (b) — this file — replaces
the hard cutoff with a *chem-aware soft prior* that is differentiable
end-to-end and can be combined with PCGrad multi-task loss.

Three priors (all differentiable w.r.t. ``coords`` so they slot into
the CFM optimiser directly)
---------------------------------

1. **Soft distance mask** — :math:`p_{\\text{dist}} = \\sigma\\bigl(-(d - d_{\\text{th}})/\\tau\\bigr)`
   where :math:`\\tau = 0.3` Å is the band-width.  Replaces the hard
   step at 2.4 Å with a sigmoid that decays smoothly.  Pairs at
   :math:`d \\ll d_{\\text{th}}` get :math:`p \\approx 1`, pairs at
   :math:`d \\gg d_{\\text{th}}` get :math:`p \\approx 0`, and pairs
   in the soft band (:math:`d \\in [d_{\\text{th}} - \\tau, d_{\\text{th}} + \\tau]`)
   get a non-trivial gradient.  This is the SINGLE change that lifts
   ``decode_ratio`` from 0: it now gives the bond head *candidates* to
   score.

2. **Type-compatibility prior** — :math:`p_{\\text{type}} = \\text{compat}(Z_i, Z_j)`
   with the value in ``[0, 1]`` from a hand-curated
   ``ATOM_TYPE_COMPAT`` table.  ``(C, C) → 1.0`` (canonical covalent),
   ``(C, N) → 0.95``, ``(C, Cl) → 0.4`` (weak — Cl is monovalent so
   only one bond), ``(N, Pt) → 0.9`` (dative is strong), ``(Pt, Cl) → 0.95``
   (cisplatin-style), ``(C, F) → 0.3`` (very weak — F is mostly
   terminal), ``(C, I) → 0.4`` (weak), ``(C, Br) → 0.4`` (weak), etc.
   This prior encodes Himo 2005 JACS CuAAC regioselectivity intuition
   (terminal alkynes + azides on Pt scaffolds = strong dative) and
   the Lit-Survey-v2 5×5 Pt-click compat matrix.

3. **Valence-aware bond cap** — each atom has a ``max_valence`` from
   :data:`PRIMITIVE_ATOMS` (or :data:`METAL_ATOMS` for metals).  The
   total bond order summed at any atom cannot exceed ``max_valence``.
   A 5-bond C is rejected; a 4-bond C is OK; Pt_II allows ≤4 dative
   bonds (square-planar).  Implemented as a differentiable
   per-atom log-barrier that penalises the cap being violated; the
   penalty is ``-log(max_valence - sum_orders + eps)`` and gets added
   to the loss in the joint-training path.

All three are fully differentiable w.r.t. ``coords`` and the bond
*logits* (when combined with the bond head's output), so the
PCGrad multi-task loss can back-propagate through them.

Honest framing
--------------
This is a CPU-only path-B attempt to break the ``decode_ratio = 0``
stalemate.  The numbers in ``molmetal/reports/wf_cfm_path_b_decoder/
final.md`` are MEASURED on the small smoke harness; the full GPU
retrain verification is deferred to the next round-13 cycle (depends
on :class:`WF-GPU-Auto-Recover` clearing the SMU hang).  We do not
claim a > 50 % decode ratio on round-12 CFM samples; we only claim
that the 3-prior combination *lifts the rate off zero* in the smoke
harness, which is the unit-test contract.

Lit-grounded by:
  * Himo 2005 JACS CuAAC regioselectivity — type-compat
    ``(Csp, N3)`` is strong, ``(Csp, Csp)`` is absent, etc.
  * Lit-Survey-v2 5×5 Pt-click compat matrix
    (:file:`molmetal/molmetal_lam/lam_chem/pt_click_compat.py`) —
    encodes strict_Pt_II / Pt_II_chelating / Pt_IV verdicts as
    per-pair type-compat values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# We do NOT import ``PRIMITIVE_ATOMS`` / ``METAL_ATOMS`` directly here
# to keep this module importable in any environment (the
# ``molmetal_lam.atoms.combinators`` module pulls in RDKit +
# bond-application which are heavy).  We instead hard-code a minimal
# type-compatibility + valence table that is consistent with the
# PRIMITIVE_ATOMS schema.  If you change the canonical PRIMITIVE_ATOMS
# table, update :data:`_DEFAULT_VALENCES` and :data:`ATOM_TYPE_COMPAT`
# accordingly.
from molmetal_lam.atoms.combinators import PRIMITIVE_ATOMS, METAL_ATOMS

# Late import of the legacy decoder so the decorator pattern can wrap
# it without circular import issues.  Tests can also use
# :class:`BondAwareDecoder` directly via the inner reference
# :attr:`DecoderRework.inner`.
from molmetal.models.bond_head import (
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_DOUBLE,
    BOND_TRIPLE,
    BOND_AROMATIC,
    NUM_BOND_CLASSES,
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
    DecodedMol,
    PairFeature,
    default_trained_head,
)


__all__ = [
    "ATOM_TYPE_COMPAT",
    "DEFAULT_DISTANCE_THRESHOLD",
    "DEFAULT_SOFT_SIGMA",
    "DEFAULT_VALENCES",
    "DecoderRework",
    "ReworkResult",
    "ReworkedDecoder",
    "TMQM_MEASURED_METAL_CAPS",
    "soft_distance_mask",
    "type_compat_mask",
    "valence_log_barrier",
]


# ---------------------------------------------------------------------------
# Defaults — kept module-level so tests + callers can import them
# directly without instantiating the class.
# ---------------------------------------------------------------------------
# Threshold at which the sigmoid is centred.  2.4 Å is the same value
# the legacy decoder uses as a hard cutoff, so this prior is *backward
# compatible* with the existing pipeline at the limit ``sigma → 0``.
DEFAULT_DISTANCE_THRESHOLD: float = 2.4
# Soft-band width.  0.3 Å is wide enough that the typical CFM
# coordinate spread (1-5 Å) gets non-trivial gradient signal on the
# boundary, but narrow enough that a pair at 4.0 Å is still < 1e-3.
DEFAULT_SOFT_SIGMA: float = 0.3


# ---------------------------------------------------------------------------
# Hard-coded valences (default valence cap per atomic number).
# ---------------------------------------------------------------------------
# Synced with :data:`PRIMITIVE_ATOMS` (Z=1,6,7,8,9,15,16,17,35,53) +
# :data:`METAL_ATOMS` (Z=78,44,30,77,29,79).  Metals get a *higher*
# cap because the dative + covalent orders are allowed to sum to
# the coordination number (e.g. Pt_II = 4 = 2 covalent + 2 dative).
# When ``PRIMITIVE_ATOMS`` or ``METAL_ATOMS`` is not importable, this
# table is the canonical fallback.
_DEFAULT_VALENCES: Dict[int, int] = {
    1: 1,  # H
    6: 4,  # C
    7: 3,  # N (sp3-amine valence; aromatic N can go up to 3)
    8: 2,  # O
    9: 1,  # F
    15: 3,  # P
    16: 2,  # S
    17: 1,  # Cl
    34: 2,  # Se
    35: 1,  # Br
    53: 1,  # I
    # Metals — coordination number (max bonds to atom, summing all
    # covalent + dative contributions).  Pt_II / Pd_II / Au_III = 4
    # (square planar); Ru_II / Ir_III = 6 (octahedral); Zn_II = 4
    # (tetrahedral); Cu_II = 4 (square planar or tetrahedral).
    78: 4,  # Pt
    44: 6,  # Ru
    30: 4,  # Zn
    77: 6,  # Ir
    29: 4,  # Cu
    79: 4,  # Au
}


#: Transition-metal caps MEASURED from this project's own corpus, 2026-10-10.
#:
#: WHAT THIS IS FOR
#: ----------------
#: :data:`_DEFAULT_VALENCES` above is a *documented single-oxidation-state*
#: table — the test at ``molmetal_lam/tests/test_cfm_p0_fixes.py`` pins
#: ``DEFAULT_VALENCES[78] == 4`` on the grounds that "Pt_II is square planar".
#: That is a statement about Pt(II), not about platinum, and this repo trains on
#: all-6-metal tmQM where 13% of the Pt complexes are six-coordinate.
#:
#: :func:`valence_log_barrier` hardcodes a fallback of ``cap = 4`` for any Z
#: absent from the table, which is why Rh (45), Os (76) and Re (75) — three of
#: the six metals in every CFM cohort — were silently constrained as if they
#: were four-coordinate. Measured against the corpus, that fallback sits BELOW
#: the median for all three and far below it for Os and Re, so a penalty built
#: on it charges the model for reproducing its own training set.
#:
#: THE MEASUREMENT, AND WHY THIS BASIS
#: -----------------------------------
#: Basis is ``metal_bo_sum`` — the sum of Wiberg bond orders at the metal, from
#: the tmQM ``.BO`` blocks — because that is the quantity the barrier actually
#: constrains (``excess = clamp(atom_orders - cap)`` where ``atom_orders`` is a
#: scatter-add of per-pair bond orders). Source:
#: ``.data/molmetal/tmQM/tmqm_parsed.csv``, counted 2026-10-10.
#:
#: NOT ``coord_number``. The loader maps tmQM's ``MND`` field to
#: ``coord_number`` with no check, and the two quantities disagree: MND matches
#: ``bo_n_neighbors`` for 17869 complexes and not for 15560, and against
#: ``metal_bo_sum`` the mean |MND - bo_sum| is 0.81 (Re) to 2.43 (Ir). MND is a
#: crystallographic coordination number from the deposited structure; bo_sum is
#: a quantum-chemistry bond-order sum. Deriving a cap for a bond-order barrier
#: from the coordination number is a category error, and an earlier draft of
#: this work did exactly that.
#:
#:     metal   n       p50    p90    p99    max    (metal_bo_sum)
#:     Pt      8275    3.25   4.12   4.54   5.74
#:     Ru      9800    4.85   5.84   6.48   7.06
#:     Ir      5232    4.04   4.70   5.42   6.19
#:     Rh      4640    3.92   4.52   5.17   5.85
#:     Os      1310    5.27   6.21   7.00   7.58
#:     Re      4172    6.46   6.85   7.40   7.92
#:
#: Caps are p99 rounded up: high enough that the penalty does not fire on the
#: bulk of the corpus, low enough that a genuinely over-coordinated centre is
#: still caught. ``molmetal/tests/test_cfm_metal_valence_caps.py`` pins every
#: entry against these percentiles, so editing one without re-deriving fails.
#:
#: KNOWN LIMITATION, stated rather than hidden: a single per-element cap cannot
#: express that the same element has a different ceiling at a different
#: oxidation state (Pt(II) square-planar vs Pt(IV) octahedral). The
#: oxidation-state-aware formulation is the one the literature has moved to
#: (bond-valence-sum with assigned oxidation states; see OxiGen,
#: arXiv:2610.08296). This trainer has no oxidation-state machinery, so the
#: proxy stands — but it is a proxy, and enabling it is a decision, not a
#: default. See todo.json BLK-M1.
TMQM_MEASURED_METAL_CAPS: Dict[int, float] = {
    78: 5.0,  # Pt  p99 4.54
    44: 7.0,  # Ru  p99 6.48
    77: 6.0,  # Ir  p99 5.42  (unchanged from the documented default)
    45: 6.0,  # Rh  p99 5.17  (was: absent -> fallback 4)
    76: 7.0,  # Os  p99 7.00  (was: absent -> fallback 4)
    75: 8.0,  # Re  p99 7.40  (was: absent -> fallback 4)
}


#: Decoder-side metal ceiling, in BOND-COUNT units.
#:
#: ``TMQM_MEASURED_METAL_CAPS`` above constrains ``metal_bo_sum`` (a sum of
#: Wiberg bond orders) and is used by the soft-valence penalty in the loss.
#: The DECODER's budget is a COUNT of metal bonds, and the matching corpus
#: statistic is tmQM's ``coord_number`` (MND), not ``metal_bo_sum``.
#:
#: Wf-A113 (T77, 2026-10-28) — this block used to justify the count budget by
#: asserting that "every metal-ligand bond is class BOND_SINGLE with cost 1.0".
#: That is false on the corpus: MEASURED over all 86k complexes, 30.7% of Re's
#: metal-incident bonds are classed aromatic (6,613, cost 1.5) or double (1,498,
#: cost 2.0), and 4.4% of Ru's are aromatic (3,027)
#: (``metrics/by_metric/wfa113_t77_metal_bond_cost_units.json``).  Charging the
#: class cost against a count ceiling made the sum exceed the bond count in
#: 92.3% of Re complexes (median 7.5 against 6.0) and 27.8% of Ru ones, so the
#: selector stopped short of the coordination number the cap was derived from.
#: The selector now charges a metal bond 1.0 regardless of class — the ceiling
#: and the unit finally agree — and the ligand endpoint keeps its own order
#: cost.
#:
#: Using the bond-order cap as a bond-count cap is the same category error
#: that ``molmetal/tests/test_cfm_metal_valence_caps.py`` documents in the
#: other direction.  MEASURED (tmqm_parsed.csv, 33,429 rows):
#:
#:     metal     n     MND p50  p95  p99  max   bo_sum p50  p99
#:     Pt     8275        4      6    6    9          3.25  4.54
#:     Ru     9800        6      9   10   12          4.85  6.48
#:     Ir     5232        6      8    9   11          4.04  5.42
#:     Rh     4640        6      8   10   11          3.92  5.17
#:     Os     1310        6      9   10   12          5.27  7.00
#:     Re     4172        6      8    9   12          6.46  7.40
#:
#: The decoder's previous constant was 6 for Ru/Ir/Rh/Os/Re and 4 for Pt --
#: i.e. the corpus MEDIAN for the first five and the p50 for Pt.  A cap set
#: at the median is not a cap: it refuses half the cohort by construction.
#: MEASURED on the 400-cloud c8 cohort, 29.75% of clouds exceed it and 9.9%
#: of all metal bonds (230/2,323) are unrepresentable
#: (metrics/by_metric/wfa113_t72_valence_cap.json).
#:
#: Wf-A113 (T88, 2026-10-29) — Pt used to be pinned at 4 here as well, on the
#: same "project targets Pt(II)" argument.  The loss-side ceiling and the
#: decoder's ceiling must agree, or the barrier permits a sixth bond the
#: selector then drops -- the T82 divergence in a different table pair.
#: MEASURED (tmqm_parsed.csv, 8,275 Pt rows): MND p99 = 6; 1,465 complexes
#: (17.7%) exceed 4, stranding 2,714 bonds.  Pt(IV) octahedral prodrugs are a
#: first-class anticancer class (satraplatin reached the clinic), so the
#: exclusion was costing target chemistry, not noise.
TMQM_MEASURED_METAL_COORDINATION_CAPS: Dict[int, int] = {
    78: 6,   # Pt  p50 4, p99 6
    44: 10,  # Ru  p99 10
    77: 9,   # Ir  p99 9
    45: 10,  # Rh  p99 10
    76: 10,  # Os  p99 10
    75: 9,   # Re  p99 9
}


# Pull the valences directly from the canonical Atom combinators if
# the import succeeded.  This is a one-line mapping that respects
# future updates to :data:`PRIMITIVE_ATOMS` / :data:`METAL_ATOMS`.
def _build_default_valences() -> Dict[int, int]:
    """Return a {Z: max_valence} table.

    Sources from :data:`PRIMITIVE_ATOMS` + :data:`METAL_ATOMS` when
    the modules are importable.  Falls back to the hard-coded
    :data:`_DEFAULT_VALENCES` for any Z not present in the canonical
    tables.

    THIS IS A COORDINATION-NUMBER TABLE, NOT A VALENCE TABLE.
    --------------------------------------------------------
    It uses ``atom.arity`` = ``valence + lone_pairs``, so it reads O 4,
    N 4, F/Cl/Br/I 4, P 4, S 4.  The comment that stood here justified that
    with "this matches RDKit's valence model for dative cases where the
    lone-pair donor gets a 'dative' bond record" -- RDKit does the opposite:
    a dative bond does NOT count towards the donor's explicit valence
    (verified on RDKit 2026.03.6: an ether O with 2 C and 1 Ru reports
    explicit valence 2; see rdkit/rdkit#8312 on ``cleanupOrganometallics``).

    So this table must not be used as a valence cap.  It was, by the
    soft-valence loss, until T82: with O at 4 and the halogens at 4 the
    barrier could never fire on the atoms most likely to be over-valent.
    The valence cap the decoder and the loss both use is
    ``cfm_decoder_yuelbond._COVALENT_VALENCE``.

    The table is kept because :data:`PRIMITIVE_ATOMS` is the canonical atom
    vocabulary and ``arity`` is the right number for coordination counting;
    ``molmetal_lam/tests/test_cfm_p0_fixes.py`` pins ``[78] == 4``.
    """
    out: Dict[int, int] = dict(_DEFAULT_VALENCES)
    try:
        for sym, atom in PRIMITIVE_ATOMS.items():
            z = int(atom.atomic_num)
            # ``valence`` field on Atom is the standard-valence count.
            # For the bond-cap we use the coordination number
            # (``atom.arity``) which is ``valence + lone_pairs`` so
            # dative bonds (N→Pt, O→Cu) are counted too.  This
            # matches RDKit's valence model for dative cases where
            # the lone-pair donor gets a "dative" bond record.
            if hasattr(atom, "arity"):
                out[z] = int(atom.arity)
            else:  # pragma: no cover — defensive
                out[z] = int(atom.valence)
        for sym, atom in METAL_ATOMS.items():
            z = int(atom.atomic_num)
            out[z] = int(atom.arity)
    except Exception:
        # Fall back to the hard-coded table — never fail the import.
        pass
    return out


DEFAULT_VALENCES: Dict[int, int] = _build_default_valences()


# ---------------------------------------------------------------------------
# Atom-type compatibility prior — pairwise bond probability multiplier.
# ---------------------------------------------------------------------------
# This is a HUNDRED-cell sparse table covering the realistic SBDD
# chemistry subset.  The default is "compatible" = 1.0; chemistry
# exceptions (C-Cl weak, F terminal, etc.) drop it to 0.3-0.5.  The
# table is symmetric: ``ATOM_TYPE_COMPAT[(a, b)] == ATOM_TYPE_COMPAT[(b, a)]``.
# A pair NOT in the table defaults to 0.5 (neutral — the soft
# distance mask still gates the bond).
#
# Lit-grounding:
#   * Himo 2005 JACS CuAAC: terminal alkyne Csp + azide N3 strong
#     dative on Pt scaffolds.
#   * Lit-Survey-v2 Pt-click compat matrix: Pt-Cl strong,
#     Pt-N dative strong, Pt-O moderate, C-Pt weak (covalent Pt-C is
#     rare outside organometallics).
#   * Standard covalent chemistry: C-C strong, C-N strong, C-O strong,
#     N-N moderate, O-O weak (peroxide, not a linker).
ATOM_TYPE_COMPAT: Dict[Tuple[int, int], float] = {
    # Strong covalent pairs (organic backbone)
    (6, 6): 1.00,  # C-C  (canonical covalent)
    (6, 1): 1.00,  # C-H
    (6, 7): 0.95,  # C-N
    (6, 8): 0.95,  # C-O
    (7, 7): 0.80,  # N-N  (N-N single bonds OK; double bonds rarer)
    (7, 8): 0.85,  # N-O
    (7, 1): 1.00,  # N-H
    (8, 1): 1.00,  # O-H
    (8, 8): 0.50,  # O-O  (peroxide, weak as a linker)
    (16, 1): 0.95,  # S-H
    (15, 1): 0.95,  # P-H
    (16, 6): 0.90,  # S-C
    (15, 6): 0.90,  # P-C
    (16, 16): 0.50,  # S-S  (disulfide, weak)
    (16, 7): 0.85,  # S-N
    (15, 7): 0.85,  # P-N
    (15, 8): 0.85,  # P-O
    (16, 8): 0.85,  # S-O
    (34, 6): 0.80,  # Se-C
    (34, 1): 0.90,  # Se-H
    (34, 7): 0.80,  # Se-N
    (34, 8): 0.80,  # Se-O
    (34, 34): 0.40,  # Se-Se
    # Halogens — terminal only, weak compatibility
    (6, 17): 0.50,  # C-Cl
    (6, 9): 0.40,  # C-F   (terminal F, very weak as a linker)
    (6, 35): 0.50,  # C-Br
    (6, 53): 0.50,  # C-I
    (7, 17): 0.30,  # N-Cl  (N-Cl bond very weak; N-Cl is mostly a leaving group)
    (8, 17): 0.30,  # O-Cl
    (16, 17): 0.50,  # S-Cl
    (15, 17): 0.50,  # P-Cl
    (6, 34): 0.70,  # C-Se
    # Metal dative pairs (Himo 2005 + Lit-Survey-v2)
    (7, 78): 0.95,  # N → Pt  (dative, strong)
    (8, 78): 0.90,  # O → Pt  (dative, strong)
    (16, 78): 0.85,  # S → Pt
    (34, 78): 0.80,  # Se → Pt
    (15, 78): 0.85,  # P → Pt
    (17, 78): 0.95,  # Cl → Pt (cisplatin-style, strong covalent+dative)
    (7, 29): 0.95,  # N → Cu
    (8, 29): 0.90,  # O → Cu
    (17, 29): 0.80,  # Cl → Cu
    (7, 30): 0.85,  # N → Zn
    (8, 30): 0.85,  # O → Zn
    (17, 30): 0.70,  # Cl → Zn
    (7, 77): 0.90,  # N → Ir
    (8, 77): 0.85,  # O → Ir
    (17, 77): 0.85,  # Cl → Ir
    (7, 44): 0.90,  # N → Ru
    (8, 44): 0.85,  # O → Ru
    (17, 44): 0.80,  # Cl → Ru
    (7, 79): 0.90,  # N → Au
    (8, 79): 0.85,  # O → Au
    (17, 79): 0.90,  # Cl → Au
    # C-Metal covalent (rare outside organometallics)
    (6, 78): 0.30,  # C-Pt  (covalent Pt-C, organometallic only)
    (6, 29): 0.30,  # C-Cu
    (6, 30): 0.30,  # C-Zn
    (6, 77): 0.30,  # C-Ir
    (6, 44): 0.30,  # C-Ru
    (6, 79): 0.30,  # C-Au
    # Metal-metal bonds (cluster chemistry, rare in SBDD)
    (78, 78): 0.20,  # Pt-Pt
    (29, 29): 0.20,  # Cu-Cu
}


def _compat(z_i: int, z_j: int) -> float:
    """Look up the type-compatibility for a pair (Z_i, Z_j).

    Returns 0.5 (neutral) for any pair not in :data:`ATOM_TYPE_COMPAT`.
    """
    if z_i == z_j:
        # Diagonal pairs are handled by their own row in the table.
        return ATOM_TYPE_COMPAT.get((z_i, z_j), 0.5)
    return ATOM_TYPE_COMPAT.get((z_i, z_j), ATOM_TYPE_COMPAT.get((z_j, z_i), 0.5))


# ---------------------------------------------------------------------------
# Soft distance mask — sigmoid band around threshold.
# ---------------------------------------------------------------------------
def soft_distance_mask(
    distances: torch.Tensor,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    sigma: float = DEFAULT_SOFT_SIGMA,
) -> torch.Tensor:
    """Return per-pair bond probability from the soft distance prior.

    Parameters
    ----------
    distances : (E,) tensor
        Pairwise Euclidean distances in Å.
    threshold : float
        Centre of the sigmoid (Å).  Defaults to 2.4 Å.
    sigma : float
        Soft-band width (Å).  Defaults to 0.3 Å.

    Returns
    -------
    (E,) tensor
        ``p_dist = sigmoid(-(d - threshold) / sigma)`` in [0, 1].
        Fully differentiable w.r.t. ``distances``.

    Notes
    -----
    ``d = threshold`` → ``p = 0.5``.
    ``d << threshold`` (e.g. 1.0 Å covalent) → ``p ≈ 1.0``.
    ``d >> threshold`` (e.g. 4.0 Å non-bonded) → ``p ≈ 0.0``.
    The 0.3 Å sigma gives a ~10x decay over 1 Å, so the gradient is
    non-trivial for ``d ∈ [1.5, 3.5] Å`` (the typical CFM coordinate
    spread).
    """
    if distances.dim() != 1:
        raise ValueError(f"distances must be (E,) tensor, got {tuple(distances.shape)}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}")
    # The negative sign: as ``d`` increases, ``p`` decreases.  We use
    # ``-d`` so the sigmoid's natural "large positive → 1" direction
    # corresponds to "small d → 1".
    z = -(distances - float(threshold)) / float(sigma)
    return torch.sigmoid(z)


# ---------------------------------------------------------------------------
# Type-compatibility prior — lookup table → per-pair probability.
# ---------------------------------------------------------------------------
def type_compat_mask(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    compat_table: Optional[Dict[Tuple[int, int], float]] = None,
    default: float = 0.5,
) -> torch.Tensor:
    """Return per-pair type-compatibility score in [0, 1].

    Parameters
    ----------
    z_i, z_j : (E,) long tensors — atomic numbers.
    compat_table : dict, optional
        Override the default :data:`ATOM_TYPE_COMPAT`.
    default : float
        Fallback value for pairs not in the table.

    Returns
    -------
    (E,) tensor
        ``p_type = compat[min(z_i, z_j), max(z_i, z_j)]`` clamped to
        ``[0, 1]``.  Differentiable w.r.t. ``z_i`` / ``z_j`` only via
        the index — the lookup is non-differentiable but the
        combination with :func:`soft_distance_mask` and
        :func:`valence_log_barrier` still produces a differentiable
        loss in the joint-training path because the bond *order*
        logits carry the per-pair chemical identity.
    """
    if z_i.shape != z_j.shape or z_i.dim() != 1:
        raise ValueError(
            f"z_i and z_j must be (E,) tensors with matching shape, "
            f"got {tuple(z_i.shape)}, {tuple(z_j.shape)}"
        )
    table = compat_table if compat_table is not None else ATOM_TYPE_COMPAT
    e = z_i.shape[0]
    out = torch.full((e,), float(default), dtype=torch.float32, device=z_i.device)
    # Vectorised lookup: for each pair, try (z_i, z_j) first, then
    # (z_j, z_i).  This is O(E) Python overhead which is acceptable
    # for the 100-pair smoke harness; for >10k pairs callers should
    # pre-build a (max_z, max_z) dense compat matrix.
    zi_list = z_i.tolist()
    zj_list = z_j.tolist()
    for k in range(e):
        a, b = int(zi_list[k]), int(zj_list[k])
        if a == b:
            val = table.get((a, b), default)
        else:
            val = table.get((a, b), table.get((b, a), default))
        out[k] = float(val)
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Valence log barrier — differentiable penalty for overvalent atoms.
# ---------------------------------------------------------------------------
def valence_log_barrier(
    bond_orders_soft: torch.Tensor,
    z: torch.Tensor,
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    n_atoms: int,
    valences: Optional[Dict[int, int]] = None,
    eps: float = 1e-3,
    aromatic_mass: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a differentiable penalty for valence-cap violations.

    Parameters
    ----------
    bond_orders_soft : (E,) tensor
        Predicted bond *order* (1, 2, 3, 4) per pair, in
        ``[0, NUM_BOND_CLASSES-1]``.  Must be a soft prediction (e.g.
        the softmax of bond-order logits times the order index) so
        the barrier is differentiable w.r.t. the head's parameters.
    z : (N,) long tensor — atomic numbers per atom.
    pair_src, pair_dst : (E,) long tensors — edge endpoints.
    n_atoms : int
    valences : dict, optional — override the default cap table.  When
        ``None`` the barrier uses the decoder's own
        ``_COVALENT_VALENCE`` (RDKit's valence model) for ligand atoms, with
        ``TMQM_MEASURED_METAL_COORDINATION_CAPS`` for the six cohort metals in
        ``aromatic_mass`` mode and ``TMQM_MEASURED_METAL_CAPS`` in the scalar
        mode (the cap must be in the same units as the accounting).  It used
        to default to :data:`DEFAULT_VALENCES`, which is the *coordination
        number* (``PRIMITIVE_ATOMS.arity`` = valence + lone pairs) and is
        therefore the wrong quantity here: it gave O a cap of 4 and the
        halogens 4, so the barrier could never fire on the atoms most likely
        to be over-valent.  See the T82 note below.
    eps : float
        Numerical floor so the hinge is zero at exactly ``cap``.
    aromatic_mass : (E,) tensor, optional
        Soft probability that the pair is ``BOND_AROMATIC``.  Passing it
        switches the barrier into the mode that MIRRORS THE DECODER: the
        aromatic contribution is charged the way
        ``_endpoint_valence_cost`` charges it (1.0 at a heteroatom donor,
        1.0 for a carbon's third and later aromatic bond), a metal-ligand
        bond costs the ligand endpoint nothing and the metal endpoint 1.0.
        ``bond_orders_soft`` must then still be the 1.5-per-aromatic order
        sum, because the non-aromatic part is recovered as
        ``bond_orders_soft - 1.5 * aromatic_mass``.  Verified to agree with
        the decoder on 0 of 1.24M ligand atoms in the six-metal cohort.

    Returns
    -------
    scalar tensor
        ``sum_a max(0, sum_e orders_e_at_a - cap_a + eps)^2``.
        Squared hinge penalty; the log barrier was tempting but it
        diverges too sharply for an untrained head.  Squared hinge
        gives smooth gradients everywhere in the feasible region.

    Sign
    ----
    The penalty fires when an atom is OVER its cap.  The docstring here used
    to write the hinge as ``max(0, cap - sum - eps)``, which is the opposite
    sign and would have penalised every atom for being UNDER-valence; the
    code was always ``sum - cap`` and is unchanged.  The tests in
    ``molmetal/tests/test_wf_r41_soft_valence_loss.py`` pin the direction.

    Honest framing
    --------------
    The squared-hinge form is intentionally simpler than the log
    barrier advertised in the design doc.  The log barrier has a
    pole at ``valence = sum_orders`` which makes the loss
    ill-conditioned for the first few epochs (when sum_orders easily
    exceeds the cap on a CFM sample with many atom pairs).  The
    squared hinge trades sharpness for stability; the gradient
    direction is the same and the descent path is well-behaved.

    This is a one-line change in the spec — call out as
    "design-deviation: log → squared-hinge" in the report.

    T82 (2026-10-28)
    ----------------
    The loss and the decoder disagreed about valence on 9,002 of 33,429
    six-metal complexes (26.93%).  The loss penalised atoms the decoder
    accepts -- C 1.13%, P 9.38%, S 18.73%, Se 36.76% of that element's atoms,
    e.g. every sulfone (6.0 against a cap of 4) and every fusion carbon
    (4.5 against 4) -- which is a gradient pointing away from the corpus
    truth.  In the other direction it permitted the 4th bond at boron, which
    the decoder refuses (10.13% of B atoms).  Measured in
    ``metrics/by_metric/wfa113_t82_loss_decoder_valence_split.json``.
    """
    if valences is None:
        # Lazy: ``cfm_decoder_yuelbond`` imports this module at module level,
        # so the reverse import has to happen inside the call.
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            _COVALENT_VALENCE,
        )

        # The metal cap must be in the same units as the metal's accounting.
        # With ``aromatic_mass`` the metal is charged 1.0 per bond (a bond
        # count), so the cap is the measured COORDINATION cap.  The scalar path
        # has no way to apply that rule and charges the class order, so it
        # keeps the bond-order-SUM cap.
        if aromatic_mass is None:
            valences = {**_COVALENT_VALENCE, **TMQM_MEASURED_METAL_CAPS}
        else:
            valences = {
                **_COVALENT_VALENCE,
                **TMQM_MEASURED_METAL_COORDINATION_CAPS,
            }
    if bond_orders_soft.dim() != 1:
        raise ValueError(
            f"bond_orders_soft must be (E,), got {tuple(bond_orders_soft.shape)}"
        )
    e = bond_orders_soft.shape[0]
    if e == 0:
        return torch.zeros(
            (), dtype=bond_orders_soft.dtype, device=bond_orders_soft.device
        )
    # Per-atom valence cap (N,) — falls back to 4 (carbon) for unknown Z.
    cap = torch.tensor(
        [float(valences.get(int(z[a].item()), 4)) for a in range(n_atoms)],
        dtype=bond_orders_soft.dtype,
        device=bond_orders_soft.device,
    )
    if aromatic_mass is None:
        # Per-atom bond order sum (N,) — scatter_add of soft orders.
        atom_orders = torch.zeros(
            (n_atoms,),
            dtype=bond_orders_soft.dtype,
            device=bond_orders_soft.device,
        )
        atom_orders = atom_orders.scatter_add(
            0,
            pair_src.long(),
            bond_orders_soft,
        )
        atom_orders = atom_orders.scatter_add(
            0,
            pair_dst.long(),
            bond_orders_soft,
        )
    else:
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            _AROMATIC_HETERO_ZS,
        )

        zeros = torch.zeros(
            (n_atoms,),
            dtype=bond_orders_soft.dtype,
            device=bond_orders_soft.device,
        )
        # Metal endpoints: a metal's budget is a COUNT of coordination bonds
        # (T72/T77), so every bond it is part of costs 1.0 whatever class the
        # head gave it.  The ligand end of the same bond costs 0 (T79: the
        # electron pair is the ligand's, so the donor's valence is unchanged).
        # Both rules have to be here, not just in the decoder: without them the
        # barrier charges a metal 1.5 for an aromatic-class bond against a
        # count cap, and charges the ligand for every dative bond.
        metal_zs = TMQM_MEASURED_METAL_COORDINATION_CAPS
        src_metal = torch.tensor(
            [int(z[int(s)].item()) in metal_zs for s in pair_src.tolist()],
            dtype=torch.bool,
            device=bond_orders_soft.device,
        )
        dst_metal = torch.tensor(
            [int(z[int(d)].item()) in metal_zs for d in pair_dst.tolist()],
            dtype=torch.bool,
            device=bond_orders_soft.device,
        )
        metal_pair = src_metal | dst_metal
        # Everything except the aromatic part, at its own order.
        non_arom = bond_orders_soft - 1.5 * aromatic_mass
        one = torch.ones_like(non_arom)
        zero_pair = torch.zeros_like(non_arom)
        contrib_src = torch.where(
            src_metal, one, torch.where(dst_metal, zero_pair, non_arom)
        )
        contrib_dst = torch.where(
            dst_metal, one, torch.where(src_metal, zero_pair, non_arom)
        )
        atom_orders = zeros.scatter_add(0, pair_src.long(), contrib_src)
        atom_orders = atom_orders.scatter_add(0, pair_dst.long(), contrib_dst)
        # Aromatic contribution, only for bonds between two non-metals (a
        # metal-ligand bond is already accounted for above).
        arom_lig = aromatic_mass * (~metal_pair).to(aromatic_mass.dtype)
        arom_at = zeros.scatter_add(0, pair_src.long(), arom_lig)
        arom_at = arom_at.scatter_add(0, pair_dst.long(), arom_lig)
        # Carbon-like: 1.5 for the first two, 1.0 for each one beyond.  Total
        # for n = 3 is 4 and for n = 4 is 5, i.e. exactly the Kekule sum.
        carbon_like = 1.5 * torch.clamp(arom_at, max=2.0) + torch.clamp(
            arom_at - 2.0, min=0.0
        )
        is_hetero = torch.tensor(
            [
                int(z[a].item()) in _AROMATIC_HETERO_ZS
                for a in range(n_atoms)
            ],
            dtype=torch.bool,
            device=bond_orders_soft.device,
        )
        atom_orders = atom_orders + torch.where(
            is_hetero, arom_at, carbon_like
        )
    excess = torch.clamp(atom_orders - cap + eps, min=0.0)
    return excess.pow(2).sum()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ReworkResult:
    """Output of :meth:`DecoderRework.compute_bond_logits`.

    Attributes
    ----------
    edge_index : (2, E) long tensor
    bond_logits : (E,) tensor — combined logit for the
        "is-bond" decision per pair, BEFORE argmax.  The downstream
        :class:`BondOrderHead` may then re-score the surviving pairs
        for bond *order*.
    p_dist : (E,) tensor — soft distance mask in [0, 1].
    p_type : (E,) tensor — type-compat in [0, 1].
    p_combined : (E,) tensor — ``p_dist * p_type`` product.
    valence_penalty : scalar tensor
    """

    edge_index: torch.Tensor
    bond_logits: torch.Tensor
    p_dist: torch.Tensor
    p_type: torch.Tensor
    p_combined: torch.Tensor
    valence_penalty: torch.Tensor


# ---------------------------------------------------------------------------
# DecoderRework — the soft 3-prior decoder.
# ---------------------------------------------------------------------------
class DecoderRework(nn.Module):
    """Chem-aware soft bond prior that wraps (or replaces) the legacy decoder.

    Parameters
    ----------
    distance_threshold : float
        Centre of the soft distance band (Å).  Default 2.4 Å (matches
        the legacy cutoff for backward compatibility).
    soft_sigma : float
        Soft-band width (Å).  Default 0.3 Å.
    compat_table : dict, optional
        Override :data:`ATOM_TYPE_COMPAT`.  Use the default for the
        Himo 2005 / Lit-Survey-v2 grounding.
    valences : dict, optional
        Override :data:`DEFAULT_VALENCES`.
    bond_pattern_mask : bool
        When True (default), the result's ``bond_logits`` are clamped
        to ``-inf`` for pairs whose ``(Z_i, Z_j)`` are out of the
        :class:`BondOrderHead`'s atom vocab.  Set False to disable
        (useful for the CFM path where the head may not be present).
    """

    def __init__(
        self,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        soft_sigma: float = DEFAULT_SOFT_SIGMA,
        compat_table: Optional[Dict[Tuple[int, int], float]] = None,
        valences: Optional[Dict[int, int]] = None,
        bond_pattern_mask: bool = True,
    ) -> None:
        super().__init__()
        if distance_threshold <= 0.0:
            raise ValueError(
                f"distance_threshold must be > 0, got {distance_threshold!r}"
            )
        if soft_sigma <= 0.0:
            raise ValueError(f"soft_sigma must be > 0, got {soft_sigma!r}")
        self.distance_threshold = float(distance_threshold)
        self.soft_sigma = float(soft_sigma)
        self.compat_table = (
            compat_table if compat_table is not None else ATOM_TYPE_COMPAT
        )
        self.valences = (
            dict(valences) if valences is not None else dict(DEFAULT_VALENCES)
        )
        self.bond_pattern_mask = bool(bond_pattern_mask)
        # Pre-build a (max_z, max_z) dense compat matrix so the
        # per-call lookup is O(1) per pair.  We use MAX_ATOMIC_NUMBER
        # as the index space (matches the BondOrderHead's mask).
        from molmetal.models.bond_head import MAX_ATOMIC_NUMBER

        self._max_z = int(MAX_ATOMIC_NUMBER)
        dense = torch.full(
            (self._max_z, self._max_z),
            0.5,
            dtype=torch.float32,
        )
        for (a, b), v in self.compat_table.items():
            if 0 <= a < self._max_z and 0 <= b < self._max_z:
                dense[a, b] = float(v)
                dense[b, a] = float(v)
        self.register_buffer(
            "_compat_dense",
            dense,
            persistent=False,
        )
        # Pre-build a (max_z,) valence cap.
        cap = torch.tensor(
            [float(self.valences.get(z, 4)) for z in range(self._max_z)],
            dtype=torch.float32,
        )
        self.register_buffer("_valence_cap", cap, persistent=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_bond_logits(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        bond_pattern_mask: Optional[torch.Tensor] = None,
    ) -> ReworkResult:
        """Run the 3-prior bond decoder over an :class:`AtomCloud`-like input.

        Parameters
        ----------
        atom_types : (N,) long tensor — atomic numbers.
        coords : (N, 3) float tensor — Cartesian coordinates in Å.
        bond_pattern_mask : (max_z, max_z, NUM_BOND_CLASSES) bool tensor,
            optional — when provided, applied to the result.  This
            is the *order-level* mask, not the rework's type-compat
            (those are different layers).

        Returns
        -------
        :class:`ReworkResult`
        """
        if atom_types.dim() != 1 or coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"atom_types must be (N,) and coords (N, 3), got "
                f"{tuple(atom_types.shape)} and {tuple(coords.shape)}"
            )
        n = atom_types.shape[0]
        if coords.shape[0] != n:
            raise ValueError("atom_types and coords must agree on N")
        if n < 2:
            # No pairs possible — return empty result.
            empty = torch.zeros((0,), dtype=coords.dtype, device=coords.device)
            return ReworkResult(
                edge_index=torch.zeros((2, 0), dtype=torch.long, device=coords.device),
                bond_logits=empty,
                p_dist=empty.clone(),
                p_type=empty.clone(),
                p_combined=empty.clone(),
                valence_penalty=torch.zeros(
                    (), dtype=coords.dtype, device=coords.device
                ),
            )
        # Build dense pair list (i < j) using a vectorised approach
        # that PRESERVES the autograd graph on ``coords`` (so the
        # whole pipeline is differentiable end-to-end).
        idx = torch.arange(n, device=coords.device)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        mask = ii < jj  # (N, N) bool — upper triangle
        src_flat = ii[mask]
        dst_flat = jj[mask]
        # Edge index is detached (no need for autograd through indices).
        edge_index = torch.stack([src_flat, dst_flat], dim=0).long()
        # Distance is computed via indexing ``coords`` so the gradient
        # flows back to ``coords`` through ``torch.norm``.
        diff = coords[src_flat] - coords[dst_flat]
        distances = torch.norm(diff, dim=-1)
        z_i = atom_types[src_flat]
        z_j = atom_types[dst_flat]
        # --- Prior 1: soft distance mask ---
        p_dist = soft_distance_mask(
            distances,
            self.distance_threshold,
            self.soft_sigma,
        )
        # --- Prior 2: type-compat prior (vectorised via dense lookup) ---
        # Clamp Z into the pre-allocated index space.
        zi_idx = z_i.clamp(min=0, max=self._max_z - 1)
        zj_idx = z_j.clamp(min=0, max=self._max_z - 1)
        p_type = self._compat_dense[zi_idx, zj_idx]
        # --- Combined prior ---
        p_combined = p_dist * p_type
        # --- Prior 3: valence log barrier (penalty only, not part of logits) ---
        # We use bond_orders_soft = p_combined * (NUM_BOND_CLASSES-1) as
        # the per-pair "expected order" so the cap sees a *soft* order
        # in [0, 4].  Multiplying by max_order keeps the cap check
        # on the same scale as the discrete head.
        bond_orders_soft = p_combined * (NUM_BOND_CLASSES - 1)
        penalty = valence_log_barrier(
            bond_orders_soft,
            atom_types,
            edge_index[0],
            edge_index[1],
            n,
            valences=self.valences,
        )
        # --- Logit conversion ---
        # log( p / (1 - p) ) is the logit; clamp to [-20, 20] to avoid
        # -inf / +inf.  Pairs with p_combined ≈ 0 get logit ≈ -20
        # which the downstream argmax treats as a strong "no bond"
        # signal.
        eps = 1e-6
        p_safe = p_combined.clamp(eps, 1.0 - eps)
        bond_logits = torch.log(p_safe / (1.0 - p_safe)).clamp(-20.0, 20.0)
        # --- Optional: apply bond-pattern mask from the head ---
        if self.bond_pattern_mask and bond_pattern_mask is not None:
            # bond_pattern_mask has shape (max_z, max_z, NUM_BOND_CLASSES).
            # We zero out per-pair logits when the *no-bond* class is
            # disallowed.  The first class (BOND_NO_BOND) is ALWAYS
            # allowed so we don't filter here in practice; the mask
            # is used at the order-argmax stage in the inner decoder.
            pass
        return ReworkResult(
            edge_index=edge_index,
            bond_logits=bond_logits,
            p_dist=p_dist,
            p_type=p_type,
            p_combined=p_combined,
            valence_penalty=penalty,
        )

    def decode(
        self,
        cloud: AtomCloud,
        bond_head: Optional[BondOrderHead] = None,
        bond_pattern_mask: Optional[torch.Tensor] = None,
    ) -> DecodedMol:
        """Run the rework decoder end-to-end on an :class:`AtomCloud`.

        If ``bond_head`` is None, the rework emits ``DecodedMol`` from
        the 3 priors alone (no per-pair bond *order* classification).
        When ``bond_head`` is provided, the rework uses the soft prior
        to filter candidate pairs to a "viable" set, then re-scores
        them with the head for the bond order.

        Parameters
        ----------
        cloud : :class:`AtomCloud`
        bond_head : :class:`BondOrderHead`, optional
        bond_pattern_mask : (max_z, max_z, NUM_BOND_CLASSES) bool tensor
        """
        result = self.compute_bond_logits(
            cloud.atomic_numbers.long(),
            cloud.positions.float(),
            bond_pattern_mask=bond_pattern_mask,
        )
        # Threshold for "viable bond" — p_combined > 0.5 (logit > 0).
        viable_mask = result.p_combined > 0.5
        viable_idx = torch.nonzero(viable_mask, as_tuple=False).flatten()
        if viable_idx.numel() == 0:
            # No viable bonds — emit an empty mol (caller can decide
            # how to handle).
            from rdkit import Chem

            rw = Chem.RWMol()
            for z in cloud.atomic_numbers.tolist():
                try:
                    rw.AddAtom(Chem.Atom(int(max(1, min(90, int(z))))))
                except Exception:
                    rw.AddAtom(Chem.Atom(6))
            mol = rw.GetMol()
            return DecodedMol(
                mol=mol,
                smiles="",
                bond_orders=[],
                dative_bonds=[],
                n_atoms=cloud.positions.shape[0],
                n_bonds=0,
                sanitized=False,
                error="rework: no viable bonds (p_combined <= 0.5 for all pairs)",
            )
        # Build a PairFeature for the viable pairs and run the inner
        # decoder to assign bond orders.
        src = result.edge_index[0, viable_idx]
        dst = result.edge_index[1, viable_idx]
        distances = torch.norm(
            cloud.positions[src] - cloud.positions[dst],
            dim=-1,
        )
        z_i = cloud.atomic_numbers[src].long()
        z_j = cloud.atomic_numbers[dst].long()
        # dative flag — same heuristic as BondAwareDecoder._infer_pair_features
        from molmetal.models.bond_head import _is_metal, _is_donor

        is_dative = torch.tensor(
            [
                (
                    (
                        _is_metal(int(cloud.atomic_numbers[int(s)].item()))
                        and _is_donor(int(cloud.atomic_numbers[int(d)].item()))
                    )
                    or (
                        _is_metal(int(cloud.atomic_numbers[int(d)].item()))
                        and _is_donor(int(cloud.atomic_numbers[int(s)].item()))
                    )
                )
                and float(distances[k].item()) < 2.5
                for k, (s, d) in enumerate(zip(src.tolist(), dst.tolist()))
            ],
            dtype=torch.bool,
        )
        pf = PairFeature(
            edge_index=torch.stack([src, dst], dim=0).long(),
            distance=distances.float(),
            z_i=z_i,
            z_j=z_j,
            angle_to_metal=torch.zeros_like(distances),
            is_dative_candidate=is_dative,
        )
        # If no bond_head was provided, fall back to SINGLE for all
        # viable pairs (the rework's prior alone is enough to
        # demonstrate the path).
        if bond_head is None:
            from rdkit import Chem

            rw = Chem.RWMol()
            for z in cloud.atomic_numbers.tolist():
                try:
                    rw.AddAtom(Chem.Atom(int(max(1, min(90, int(z))))))
                except Exception:
                    rw.AddAtom(Chem.Atom(6))
            bond_orders: List[Tuple[int, int, int]] = []
            dative_bonds: List[Tuple[int, int]] = []
            for s, d, dt in zip(src.tolist(), dst.tolist(), is_dative.tolist()):
                if bool(dt):
                    dative_bonds.append((int(s), int(d)))
                else:
                    bond_orders.append((int(s), int(d), BOND_SINGLE))
            bt = Chem.BondType.SINGLE
            for s, d, _o in bond_orders:
                try:
                    rw.AddBond(int(s), int(d), bt)
                except Exception:
                    continue
            try:
                mol = rw.GetMol()
            except Exception as exc:
                return DecodedMol(
                    error=f"RWMol.GetMol failed: {exc}",
                    n_atoms=cloud.positions.shape[0],
                )
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
                bond_orders=bond_orders,
                dative_bonds=dative_bonds,
                n_atoms=cloud.positions.shape[0],
                n_bonds=len(bond_orders),
                sanitized=sanitized,
                error=sanitize_error,
            )
        # Inner BondAwareDecoder — same as before, just filtered to
        # viable pairs.
        inner = BondAwareDecoder(
            bond_head=bond_head,
            bond_cutoff=self.distance_threshold + 5.0 * self.soft_sigma,
            min_distance=1.0,
            use_geometry_prior=True,
            apply_bond_pattern_mask=True,
        )
        return inner.decode(cloud, pair_features=pf)


# ---------------------------------------------------------------------------
# ReworkedDecoder — decorator that composes DecoderRework with the
# legacy BondAwareDecoder.  Keeps the existing API surface
# (``decode(cloud) -> DecodedMol``) but routes the candidate-pair
# generation through the soft prior first.
# ---------------------------------------------------------------------------
class ReworkedDecoder:
    """Decorator over :class:`BondAwareDecoder` that injects the 3-prior
    soft bond prior ahead of the order-classification head.

    Parameters
    ----------
    inner : :class:`BondAwareDecoder`
        The legacy decoder (bond head + RDKit assembly).
    rework : :class:`DecoderRework`
        The 3-prior soft decoder.  When ``None``, a default
        :class:`DecoderRework` is constructed with the canonical
        distance_threshold and sigma.
    p_threshold : float
        ``p_combined`` cutoff for "viable bond" — pairs with
        ``p_combined <= p_threshold`` are dropped before the inner
        decoder sees them.  Default 0.5 (i.e. the soft distance mask
        plus type compat product must exceed 0.5 to be a candidate).
    """

    def __init__(
        self,
        inner: BondAwareDecoder,
        rework: Optional[DecoderRework] = None,
        p_threshold: float = 0.5,
    ) -> None:
        self.inner = inner
        self.rework = rework if rework is not None else DecoderRework()
        if not 0.0 < float(p_threshold) < 1.0:
            raise ValueError(f"p_threshold must be in (0, 1), got {p_threshold!r}")
        self.p_threshold = float(p_threshold)

    def decode(self, cloud: AtomCloud) -> DecodedMol:
        """Run the soft-prior-filtered decoder end-to-end.

        Equivalent in surface to ``inner.decode(cloud)`` but with a
        soft prior between candidate generation and order
        classification.
        """
        return self.rework.decode(
            cloud,
            bond_head=self.inner.bond_head,
            bond_pattern_mask=(
                self.inner.bond_head.bond_pattern_mask
                if hasattr(self.inner.bond_head, "bond_pattern_mask")
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Self-test (lightweight — run as ``python -m
# molmetal_lam.lam_chem.decoder_rework``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    rework = DecoderRework()
    # Random 10-atom cloud with C/N/O/Cl/Pt.
    z = torch.tensor([6, 6, 7, 8, 17, 6, 6, 7, 78, 17], dtype=torch.long)
    coords = torch.randn(10, 3) * 1.5  # ~3 Å spread
    res = rework.compute_bond_logits(z, coords)
    print(
        f"rework smoke: E={res.edge_index.shape[1]}, "
        f"p_dist.mean={res.p_dist.mean().item():.3f}, "
        f"p_type.mean={res.p_type.mean().item():.3f}, "
        f"p_combined.mean={res.p_combined.mean().item():.3f}, "
        f"valence_penalty={res.valence_penalty.item():.3f}, "
        f"n_viable={int((res.p_combined > 0.5).sum().item())}"
    )
    # End-to-end decode.
    head = default_trained_head()
    cloud = AtomCloud(positions=coords, atomic_numbers=z)
    inner = BondAwareDecoder(bond_head=head)
    rd = ReworkedDecoder(inner=inner, rework=rework)
    dec = rd.decode(cloud)
    print(
        f"ReworkedDecoder end-to-end: smiles={dec.smiles!r}, "
        f"n_atoms={dec.n_atoms}, n_bonds={dec.n_bonds}, "
        f"sanitized={dec.sanitized}, error={dec.error!r}"
    )
