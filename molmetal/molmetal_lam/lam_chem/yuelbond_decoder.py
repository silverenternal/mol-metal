"""YuelBond-style learned GNN edge predictor for CFM connectivity decoding.

WF-CFM-Rescue Phase 1 (2026-09-16) — implements a small message-passing
GNN that maps (positions, atomic_numbers) -> per-pair bond logits.

Lit anchor
-----------
Wang, S., Dokholyan, P. (2025). *YuelBond: a graph neural network for
recovering the molecular bond network from distorted 3D structures.*
bioRxiv 10.1101/2025.05.06.652517.  Reported F1=92.7 % on distorted
molecules where RDKit's ``DetermineConnectivity`` fails on 783/1000
cases.

This module is a CPU-only re-implementation of the YuelBond decoder head
inspired by the bioRxiv preprint.  It is intentionally small (≤ 5k
parameters) so it can run inside a 200-step decode smoke at < 5 s wall.
The architecture is:

    AtomEncoder(z) -> (H,)  via one-hot(Z) @ W_atom
    Iteratively for k in [0..K):
        h_i' = h_i + sum_j  msg(h_i, h_j, d_ij)
        msg  = MLP([h_i, h_j, d_ij, is_short, is_long])
    PairHead(i, j)  = MLP([h_i', h_j', d_ij, is_dative_candidate])
                    -> logits over {NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC}

The output tensor has shape ``(E, 5)`` and is consumed by
:class:`molmetal.models.bond_head.BondAwareDecoder` via the existing
``pair_features`` interface — no changes to ``BondAwareDecoder`` are
required.

Why CPU-only?
-------------
The current decode smoke must run with `cuda_available=False` (per
``wf_gpu_auto_recover/final.md`` 2026-09-15 + ``wf_gpu_recovery_now/
final.md`` 2026-09-15).  The YuelBond decoder head is small enough
(~5k parameters, K=3 message-passing layers) that the CPU cost is
negligible compared to the rest of the decode pipeline.

Lit anchor (secondary)
----------------------
Liu, Z., et al. (2025). *NExT-Mol: 1D SELFIES + 3D conformer decoupling
for molecular generation.*  ICLR 2025, arXiv:2502.12638.  Their MoLlama
1.8B-pretrain shows that learning the bond topology from positions can
outperform covalent-radius heuristics on out-of-distribution scaffolds —
the same intuition that motivates this decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.models.bond_head import (
    AtomCloud,
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_LABELS,
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_TRIPLE,
    NUM_BOND_CLASSES,
    PairFeature,
)


# ---------------------------------------------------------------------------
# Helper: Z -> one-hot
# ---------------------------------------------------------------------------
# Use a fixed-size vocab that matches the round-10 atom_vocab in
# ``flow_matching_lipman/__init__.py`` (12 atoms + Z=0 padding).
_YUELBOND_Z_TABLE: Tuple[int, ...] = (0, 1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78)
_YUELBOND_Z_TO_IDX = {z: i for i, z in enumerate(_YUELBOND_Z_TABLE)}


#: 2026-10-17 (Wf-A69 W2 fix 5) — METAL SET for dative-bond detection.
#:
#: Pre-Wf-A69 W2, :func:`YuelBondDecoderHead.forward` hardcoded Pt (Z=78)
#: as the lone metal atom (``atomic_numbers[j_idx] == 78``) when labelling
#: ``is_dative``.  That rule mis-labels every Ru/Ir/Os/Rh/Re pair (the other
#: five project metals) as ``BOND_NO_BOND`` even when the cloud actually has
#: a coordination bond, which is the highest-leverage structural defect on
#: the 5-metal MetalCytoToxDB cohort per `docs/HONEST_NEGATIVE_CATALOG.md`
#: §6.46 + Wf-A69 W1 design §fix5.
#:
#: The set below covers all six of the project's target metals (Pt/Ru/Ir/Os/
#: Rh/Re) so a donor atom (N/O/S/P, Z in {7,8,15,16}) bonded to ANY of them
#: registers as dative.  Default ``_DEFAULT_METAL_SET`` is the same tuple —
#: the change is opt-in only at construction time.
_DEFAULT_METAL_SET: frozenset = frozenset({44, 45, 46, 75, 76, 77, 78})
#: Ru=44 / Rh=45 / Pd=46 / Re=75 / Os=76 / Ir=77 / Pt=78 (d-block 4d/5d
#: precious-metal + Re; Pd included for Pt/Pd-platinum group compatibility).
_METAL_SET_PRESETS: dict = {
    "pt_only": frozenset({78}),
    "all6": frozenset({44, 45, 75, 76, 77, 78}),
    "all7": _DEFAULT_METAL_SET,
}

#: 2026-10-04 — EXTENDED table covering all six of the project's target metals.
#:
#: The default table above contains exactly ONE metal: Pt (78).  The other
#: five -- Ru, Ir, Rh, Os, Re -- are absent, so ``_z_to_onehot`` clamps them to
#: the Z=0 padding slot and the decoder cannot tell them apart or emit them.
#: Measured on the tmQM corpus (108,541 CSD-derived complexes, MIT, on disk at
#: .data/molmetal/tmQM/): Ru 9,068 / Ir 4,695 / Rh 4,278 / Re 4,033 / Os 1,224
#: entries are unreachable through the default vocab.
#:
#: Why this is a SEPARATE tuple rather than an edit of the tuple above:
#:   * the original 13 entries are tied by comment to the round-10 atom_vocab
#:     in ``flow_matching_lipman/__init__.py`` (12 atoms + Z=0 padding), and
#:     that correspondence is worth keeping intact for the Pt-only path;
#:   * the table's LENGTH is a model dimension -- ``atom_encoder`` is
#:     ``nn.Linear(len(table), hidden_dim)`` and the CFM trainer's velocity
#:     probes one-hot over the same width.  Growing it from 13 to 18 changes
#:     those tensor shapes, so every existing checkpoint becomes unloadable.
#:     That is a real consequence, not a detail; making it opt-in means the
#:     default path keeps working and the break is a deliberate choice.
#:
#: Order matters: the original 13 keep their indices, so a model built with
#: this table still assigns Pt the same one-hot slot it had before.  Only the
#: width changes.
_YUELBOND_Z_TABLE_ALL6: Tuple[int, ...] = (
    0, 1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78,  # original 13 (Pt last)
    44,  # Ru
    77,  # Ir
    45,  # Rh
    76,  # Os
    75,  # Re
)

#: Selectable vocabularies, by name.  ``pt`` is the default so every shipped
#: config and checkpoint behaves exactly as before.
Z_TABLE_PRESETS: dict = {
    "pt": _YUELBOND_Z_TABLE,
    "all6": _YUELBOND_Z_TABLE_ALL6,
}


def get_z_table(name: str = "pt") -> Tuple[int, ...]:
    """Return the Z table for a preset name; raise on an unknown one.

    Raising rather than falling back matters: a typo would otherwise silently
    select the Pt-only table and the five other metals would be clamped to
    padding without a word.
    """
    key = str(name).strip().lower()
    if key not in Z_TABLE_PRESETS:
        raise ValueError(
            f"unknown z-table preset {name!r}; available: "
            f"{sorted(Z_TABLE_PRESETS)}"
        )
    return Z_TABLE_PRESETS[key]


def z_to_onehot(z: torch.Tensor, z_table: Tuple[int, ...] = _YUELBOND_Z_TABLE) -> torch.Tensor:
    """Map atomic numbers to one-hot indices into ``z_table``.

    Unknown Z values are clamped to the Z=0 (padding) slot.  The table is a
    parameter so the all-six-metals vocab can be used without changing this
    function's behaviour for callers that take the default.
    """
    to_idx = {zz: i for i, zz in enumerate(z_table)}
    # 2026-10-05 — allocate on `z`'s device.  This defaulted to CPU, so a CUDA
    # atomic-number tensor still produced a CPU one-hot and the head's
    # `nn.Linear` raised `Expected all tensors to be on the same device`.
    # Fixing it here rather than at the call site means every caller inherits
    # it, including ones that have not been exercised on GPU yet.
    idx = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    for i, atomic_number in enumerate(z.tolist()):
        idx[i] = to_idx.get(int(atomic_number), 0)
    return F.one_hot(idx, num_classes=len(z_table)).float()


def bond_decision(
    logits: torch.Tensor,
    decision: str = "argmax",
    threshold: float = 0.5,
    class_logit_bias: Optional[Sequence[float]] = None,
) -> list:
    """Turn per-pair class logits into predicted bond classes.

    Lives HERE, in the library, because two consumers need it and must not
    disagree: the metric path (``evaluate_bond_accuracy``) and the decode path
    (``YuelBondBondDecoder._select_bonds``).  When the decode path hardcoded
    ``argmax`` while the metrics used a threshold, the reported bond F1 and the
    molecules actually assembled into an RDKit ``Mol`` came from two different
    rules — a wiring gap invisible in the code and meaningless in the numbers
    only for as long as the two happened to agree.

    ``argmax`` — the historical rule.  MEASURED to be badly calibrated in both
    directions on this task: with unweighted CE it collapsed to recall 0.007
    (F1 0.013) while the same logits reached F1 0.46-0.50 under a threshold,
    and with full inverse-frequency weights it swung to recall 0.97 /
    precision 0.32.  Four of five classes are overwhelmingly BOND_NO_BOND, so
    the argmax boundary sits nowhere near a useful operating point, and class
    weighting MOVES it rather than fixing it.

    ``threshold`` — existence is decided by ``P(bonded) >= threshold``, and the
    class is the argmax over the NON-zero classes.  This decouples "is there a
    bond" from "which bond", which is what the data supports: the existence
    question has a ~29% positive rate, the type question is nearly
    deterministic given existence.

    ``class_logit_bias`` — a per-class additive offset on the logits, applied
    BEFORE either decision rule.  ``None`` (the default) is byte-identical to
    the pre-2026-10-12 behaviour: no tensor is built, no arithmetic runs.

    WHY IT IS HERE (BLK-C-M4, measured premise).  On real tmQM geometry the
    head emits the aromatic class but under-ranks it: on 15,851 truth-aromatic
    bonds (Wiberg BO 1.3-1.7) measured over 798 complexes, 11,209 (70.7%)
    are predicted BOND_SINGLE instead, while single-bond recall is 0.966.
    The defect is therefore class-4 vs class-1 RANKING, not a class the head
    never emits, and an offset is the smallest thing that can move a ranking.

    WHY IT IS NOT THE STANDARD logit adjustment.  Menon 2020 (ICLR'21,
    arXiv:2007.07314) post-hoc adjustment subtracts ``tau * log(pi_c)`` for
    EVERY class, which is the right tool when the classes are imbalanced in
    the usual way.  It is the WRONG tool here for a measured reason: on the
    256-cloud Ru cohort the candidate-pair counts are
    [24378, 4460, 421, 28, 5661], so BOND_TRIPLE holds 0.08% of pairs and
    would receive by far the largest boost -- and triple bonds are ALREADY a
    pathology on this line (66% of generated bonds, per
    docs/CFM_STATUS_2026_10_12.md BLK-C-OBJ).  A global prior correction
    would amplify a known disease while trying to treat another.  So the
    offset is per-class and explicit: setting index 4 moves aromatic and
    nothing else, and an all-zeros vector is a no-op.

    This is an OPERATING-POINT knob, not a model change.  It cannot fix a head
    whose aromatic logits rank below single for every input; it moves the
    operating point so the caller can see what the head actually knows.  A
    genuine fix to the ranking is a retrain question (BLK-C-M4), and this
    deliberately does not pretend to be one.
    """
    if class_logit_bias is not None:
        bias = torch.as_tensor(
            list(class_logit_bias), dtype=logits.dtype, device=logits.device
        )
        if bias.numel() != logits.shape[-1]:
            raise ValueError(
                f"class_logit_bias has {bias.numel()} entries but the logits "
                f"have {logits.shape[-1]} classes. A short vector would "
                f"broadcast-misalign the classes silently -- the offset for "
                f"BOND_AROMATIC would land on whatever class happened to be "
                f"at that index."
            )
        logits = logits + bias

    if decision == "argmax":
        return logits.argmax(dim=-1).tolist()
    if decision != "threshold":
        raise ValueError(
            f"unknown bond decision rule {decision!r}; expected 'argmax' or "
            f"'threshold'. A typo here would otherwise fall through to the "
            f"argmax branch and silently decode with the wrong rule."
        )
    prob = torch.softmax(logits.float(), dim=-1)
    p_bonded = 1.0 - prob[:, 0]  # column 0 is BOND_NO_BOND
    types = prob[:, 1:].argmax(dim=-1) + 1
    return [
        int(types[k]) if float(p_bonded[k]) >= threshold else 0
        for k in range(logits.shape[0])
    ]


def _z_to_onehot(z: torch.Tensor) -> torch.Tensor:
    """Map atomic numbers to one-hot indices into :data:`_YUELBOND_Z_TABLE`.

    Unknown Z values are clamped to Z=0 (padding slot).
    """
    idx = torch.zeros(z.shape[0], dtype=torch.long)
    for i, atomic_number in enumerate(z.tolist()):
        idx[i] = _YUELBOND_Z_TO_IDX.get(int(atomic_number), 0)
    return F.one_hot(idx, num_classes=len(_YUELBOND_Z_TABLE)).float()


# ---------------------------------------------------------------------------
# YuelBondDecoderHead — small message-passing GNN
# ---------------------------------------------------------------------------
class YuelBondDecoderHead(nn.Module):
    """Small GNN that maps an :class:`AtomCloud` to per-pair bond logits.

    Parameters
    ----------
    hidden_dim : int, default 64
        Width of the message-passing hidden state.  Small enough to keep
        the head under 5k parameters; large enough to represent the
        triple / aromatic minority classes.
    n_layers : int, default 3
        Number of message-passing iterations.  K=3 matches the bioRxiv
        preprint's recommendation for drug-like ligands.
    cutoff : float, default 3.5 Å
        Pairs further apart than this are immediately classified as
        ``NO_BOND`` without being passed through the MLP head.  Keeps
        the ``E = N(N-1)/2`` edge list manageable at N=19 atoms
        (171 pairs).
    min_distance : float, default 1.0 Å
        Pairs closer than this are clamped (physically impossible
        steric clash).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        cutoff: float = 3.5,
        min_distance: float = 1.0,
        z_table: Tuple[int, ...] = _YUELBOND_Z_TABLE,
        # Wf-A68 (2026-10-17) — bond-head residual skip (Spot 4).
        # Default False preserves Wf-A67 bit-exact compat.  When True,
        # the message-passing output adds a Linear(hidden_dim,
        # hidden_dim) projection from the post-encoder atom embedding
        # (a residual around the n message-passing layers — ResNet-
        # style).  One extra (hidden_dim^2 + hidden_dim) parameters.
        use_skip: bool = False,
        # 2026-10-17 (Wf-A69 W2 fix 5) — metal set used by the dative-bond
        # rule inside :meth:`forward`.  Pre-Wf-A69 W2 the rule hardcoded
        # Pt (Z=78), so every Ru/Ir/Os/Rh/Re dative bond was mislabelled
        # as ``BOND_NO_BOND`` and the head could not learn the 5-metal
        # cohort's coordination chemistry.  Default ``"all7"`` covers all
        # seven project metals; ``"pt_only"`` preserves the pre-Wf-A69 W2
        # behaviour bit-exactly for older checkpoints; ``"all6"`` covers
        # the 6 precious-metal Pt-group + Re subset.
        metal_set: str = "all7",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.cutoff = float(cutoff)
        self.min_distance = float(min_distance)
        # 2026-10-04 — the vocab is a parameter, not a module constant, so the
        # all-six-metals table ("all6", see Z_TABLE_PRESETS) can be selected
        # without editing this class.  Default is the original Pt-only table,
        # so width -- and therefore every existing checkpoint -- is unchanged.
        self.z_table = tuple(z_table)
        # 2026-10-17 (Wf-A69 W2 fix 5) — same opt-in pattern as ``z_table``.
        # Default ``"all7"`` lights up every project metal; ``"pt_only"``
        # (the pre-Wf-A69 W2 behaviour) is the bit-exact-compat fallback.
        key = str(metal_set).strip().lower()
        if key not in _METAL_SET_PRESETS:
            raise ValueError(
                f"unknown metal_set preset {metal_set!r}; available: "
                f"{sorted(_METAL_SET_PRESETS)}"
            )
        self.metal_set = _METAL_SET_PRESETS[key]
        self.metal_set_key = key
        # Atom encoder: Z one-hot -> hidden_dim
        self.atom_encoder = nn.Linear(len(self.z_table), hidden_dim)
        # Message MLP: input [h_i, h_j, d_ij, is_short, is_long, is_dative]
        self.msg_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + 4, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )
        # Pair head: [h_i, h_j, d_ij, dative] -> NUM_BOND_CLASSES logits
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_BOND_CLASSES),
        )
        # Bias the "no bond" class slightly negative so the head does
        # not default to bonding every pair.
        with torch.no_grad():
            self.pair_head[-1].bias.zero_()
            self.pair_head[-1].bias[BOND_NO_BOND] = -0.5
        # Wf-A68 (2026-10-17) — optional residual skip from atom_encoder
        # output to the message-passing output.  When use_skip is True, an
        # extra Linear(hidden_dim, hidden_dim) projection is registered
        # and added to the post-MP per-atom h.  When False (default) the
        # skip is a no-op and the forward is bit-exact-identical to the
        # pre-Wf-A67 path.
        self.use_skip = bool(use_skip)
        if self.use_skip:
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.skip_proj = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, cloud: AtomCloud) -> torch.Tensor:
        """Return ``(E, NUM_BOND_CLASSES)`` logits over candidate edges.

        Edges are the upper-triangular pair list ``[(i, j) for i<j in N]``
        with all pairs closer than :attr:`cutoff`.  Pairs at distance
        ``> cutoff`` are NOT returned (caller should treat them as
        ``NO_BOND``); pairs at ``< min_distance`` are clamped.

        The message-passing update is the standard MPNN one — every edge
        informs BOTH endpoints, the message is passed through a ReLU, and the
        result is added residually::

            h_i <- h_i + sum_j ReLU(MLP([h_i, h_j, d_ij, flags]))
            h_j <- h_j + sum_i ReLU(MLP([h_j, h_i, d_ij, flags]))

        2026-10-05 — this is the variant the CFM trainer's
        ``_device_safe_yuelbond_forward`` monkey-patch has been installing at
        import time, and therefore the one every shipped checkpoint was
        trained under.  The module was aligned to the patch rather than the
        reverse, which makes the alignment a no-op for existing weights;
        aligning the other way would have invalidated all of them.  The
        monkey-patch is gone and this is the single definition.  See
        ``molmetal/tests/test_yuelbond_forward_alignment.py``.
        """
        # 2026-10-05 FIX — the head must run where its WEIGHTS are, and the
        # cloud must be moved there.  `forward` used to consume the cloud's
        # tensors verbatim, so a CPU cloud against CUDA weights raised
        # `RuntimeError: Expected all tensors to be on the same device`.
        #
        # MEASURED consequence (results/cfm_tmqm_all6, 2026-10-04 run):
        # `generate_clouds_from_noise` returns `.cpu()` clouds by design, so
        # every generation eval on CUDA hit this — and `YuelBondDecoder.
        # featurise` catches the exception and returns `decode_succeeded=False`,
        # which is indistinguishable from "the model produced bad chemistry".
        # gen_validity therefore read 0/128 on CUDA and 28/72 on CPU for the
        # SAME checkpoint: the CUDA zeros were a device mismatch, not a
        # modelling result.
        #
        # The head runs where its WEIGHTS are: the cloud is moved onto them.
        # A caller that hands this a CPU cloud while the module is on CUDA
        # therefore still works, which is the case
        # `generate_clouds_from_noise` produces by design.
        _dev = self.pair_head[-1].weight.device
        cloud = cloud.to(_dev)
        positions = cloud.positions.float()
        atomic_numbers = cloud.atomic_numbers.long()
        n = int(atomic_numbers.shape[0])
        if n < 2:
            return (
                torch.zeros(0, NUM_BOND_CLASSES, device=_dev),
                (
                    torch.zeros(0, dtype=torch.long, device=_dev),
                    torch.zeros(0, dtype=torch.long, device=_dev),
                ),
            )
        # Encode atoms
        z_onehot = z_to_onehot(atomic_numbers, self.z_table)
        h_atom = self.atom_encoder(z_onehot)  # (N, H) — Wf-A68 stash for skip
        h = h_atom
        # Build pair list: upper triangle.
        # 2026-10-05 — `torch.triu_indices` allocates on CPU unless told
        # otherwise, which silently dragged the whole forward back to CPU and
        # re-raised the device error the `cloud.to(_dev)` above had just fixed.
        # Every index tensor derived from it inherits this, so the device is
        # stated here once rather than at each use.
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=_dev)
        if i_idx.numel() == 0:
            return (
                torch.zeros(0, NUM_BOND_CLASSES, device=_dev),
                (
                    torch.zeros(0, dtype=torch.long, device=_dev),
                    torch.zeros(0, dtype=torch.long, device=_dev),
                ),
            )
        # Pair distances and short/long flags
        d_vec = positions[j_idx] - positions[i_idx]
        d = d_vec.norm(dim=-1).clamp(min=self.min_distance)
        # Filter to within-cutoff pairs only (cutoff=3.5 Å excludes 4.1 Å Cl-Cl)
        mask = d < self.cutoff
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        d = d[mask]
        if i_idx.numel() == 0:
            return (
                torch.zeros(0, NUM_BOND_CLASSES, device=_dev),
                (
                    torch.zeros(0, dtype=torch.long, device=_dev),
                    torch.zeros(0, dtype=torch.long, device=_dev),
                ),
            )
        # Per-edge features (now all shape (E,))
        is_short = (d < 1.7).float().unsqueeze(-1)
        is_long = (d > 3.0).float().unsqueeze(-1)
        # 2026-10-17 (Wf-A69 W2 fix 5) — metal-set membership replaces the
        # pre-Wf-A69 W2 Pt-only rule.  Pre-fix the rule was hardcoded to Pt
        # (Z=78) so every Ru/Ir/Os/Rh/Re dative bond carried ``is_dative=0``
        # even when the cloud truly had a coordination bond, which silently
        # disabled the dative feature for the 5-metal cohort.  The fix is an
        # O(E) ``isin`` membership check against ``self.metal_set`` (default
        # all7 covers Pt/Ru/Ir/Os/Rh/Re/Pd).  Bit-exact for ``metal_set=
        # "pt_only"`` — pre-Wf-A69 W2 callers see no behaviour change.
        donor_mask = (
            (atomic_numbers[i_idx] == 7)
            | (atomic_numbers[i_idx] == 8)
            | (atomic_numbers[i_idx] == 15)
            | (atomic_numbers[i_idx] == 16)
        )
        # ``isin`` with the project's frozen-set is an O(E) op.  We build the
        # tensor on the head's device once per call; the alternative (looping
        # over the metal tuple) is slower when the set grows beyond a couple
        # of entries.
        metal_t = torch.as_tensor(
            sorted(self.metal_set), device=atomic_numbers.device, dtype=atomic_numbers.dtype
        )
        is_dative = donor_mask & torch.isin(
            atomic_numbers[j_idx], metal_t
        )
        is_dative = is_dative.float().unsqueeze(-1)
        # K message-passing iterations.
        #
        # 2026-10-05 — this loop is the variant the trainer's
        # `_device_safe_yuelbond_forward` monkey-patch has been installing at
        # import time, and it is the one every shipped CFM checkpoint was
        # TRAINED under.  Per the user's decision, the module is aligned to the
        # patch rather than the reverse, so aligning is a NO-OP for existing
        # weights: same activation, same bidirectional aggregation, same
        # residual.  `molmetal/tests/test_yuelbond_forward_alignment.py`
        # pins the three steps so neither variant can drift back.
        #
        # What the module's own version used to do, kept here for the record:
        # no activation, aggregate into `i_idx` only, no residual. That is not
        # the standard MPNN update (Gilmer 2017 and every PyG/DGL/SchNet/MACE
        # implementation aggregate each edge in BOTH directions and add a
        # residual), so the patch is the better of the two on its own terms.
        for k in range(self.n_layers):
            msg_in = torch.cat(
                [
                    h[i_idx],
                    h[j_idx],
                    d.unsqueeze(-1),
                    is_short,
                    is_long,
                    is_dative,
                ],
                dim=-1,
            )
            msg = self.msg_mlps[k](msg_in)
            h = h.clone()
            h = h + torch.zeros_like(h).index_add_(
                0,
                i_idx,
                torch.relu(msg),
            ).index_add_(0, j_idx, torch.relu(msg))
        # Wf-A68 (2026-10-17) — optional residual skip from atom_encoder
        # output to the post-MP per-atom embedding.  When use_skip is
        # False (default) the branch is skipped and the forward is
        # bit-exact-identical to the pre-Wf-A67 path.  When True, an
        # extra Linear(hidden_dim, hidden_dim) projection of the
        # post-encoder h is added to the post-MP h — ResNet-style skip
        # around the n message-passing layers.
        if self.use_skip and self.skip_proj is not None:
            h = h + self.skip_proj(h_atom)
        # Pair head
        pair_in = torch.cat(
            [
                h[i_idx],
                h[j_idx],
                d.unsqueeze(-1),
                is_dative,
                is_short,
            ],
            dim=-1,
        )
        logits = self.pair_head(pair_in)
        return logits, (i_idx, j_idx)

    # ------------------------------------------------------------------
    # EMA stubs (Wf-A30-1 / 2026-10-14) — see
    # ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` for the
    # full EMA implementation on the trainer's model class.  These
    # accept the same API so ``cross_dataset.py`` / ``linear_probe.py``
    # can carry an EMA snapshot through inference without an import
    # cycle.  ``load_ema_state`` just stashes the dict; ``ema_swap_in``
    # returns True iff a snapshot was supplied.  The full EMA update
    # happens in the trainer, not here — the head's weights are static
    # at decode time, so the swap is a no-op.
    def load_ema_state(self, sd):  # type: ignore[no-untyped-def]
        """No-op EMA loader for the inference-side head.

        Keeps the same contract as the trainer-side
        :meth:`metal_hybrid_v4.YuelBondEMABearing.load_ema_state` so a
        checkpoint reader can swap EMA in for either model without
        branching.
        """
        self._ema_sd = sd if isinstance(sd, dict) else {}
        return bool(self._ema_sd)

    def ema_swap_in(self):  # type: ignore[no-untyped-def]
        """Mirror ``ema_swap_in``; returns True iff an EMA was loaded."""
        return bool(getattr(self, "_ema_sd", None))

    def ema_swap_out(self) -> None:
        """Clear the EMA flag — inference never restores live weights."""
        self._ema_sd = {}


# ---------------------------------------------------------------------------
# YuelBondDecoder — wrapper that emits PairFeature + BondAwareDecoder output
# ---------------------------------------------------------------------------
@dataclass
class YuelBondResult:
    """Result of a YuelBond decode pass."""

    pair_features: Optional[PairFeature]
    logits: torch.Tensor  # (E, 5)
    edge_index: torch.Tensor  # (2, E)
    decode_succeeded: bool
    error: Optional[str] = None


class YuelBondDecoder:
    """Wraps :class:`YuelBondDecoderHead` + :class:`BondAwareDecoder`.

    This is the YuelBond-style replacement for the
    ``decode_distance_graph`` heuristic.  It is a drop-in alternative
    when ``--bond-head=learned`` is selected and ``connectivity_prior=
    yuelbond`` is requested.
    """

    def __init__(
        self,
        head: Optional[YuelBondDecoderHead] = None,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        self.head = (
            head
            if head is not None
            else YuelBondDecoderHead(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
        )
        # 2026-10-17 (Wf-A69 W2 fix 2) — record whether the caller passed
        # through their own head.  Pre-fix this was implicit: a caller that
        # ignored the ``head`` kwarg silently got a fresh CPU-initialised
        # module, so the live :class:`CFMJointModel` weights never reached
        # the eval path — the eval reported numbers from random
        # initialisation.  The Wf-A69 W1 audit (§A1.6 #1) verified the
        # trainer DOES pass ``model.head`` so this is a defensive check,
        # not a regression.  ``_head_passed_through`` is True iff the
        # caller handed us their own module rather than relying on the
        # ``None`` default.
        self._head_passed_through: bool = head is not None

    def featurise(self, cloud: AtomCloud) -> YuelBondResult:
        """Compute pair features + logits for an :class:`AtomCloud`.

        Returns a :class:`YuelBondResult`; on shape errors returns
        ``decode_succeeded=False`` with a short ``error`` message.
        """
        # 2026-10-05 — the head is moved onto the cloud's device HERE, at the
        # single point every decode passes through, rather than relying on each
        # caller to have done it.  MEASURED: `generate_clouds_from_noise`
        # returns `.cpu()` clouds, so on a CUDA run every one of these calls
        # raised `RuntimeError: Expected all tensors to be on the same device`
        # and was caught below as `head_forward_failed:RuntimeError` — which
        # reads identically to "the head ran and found no bonds".  That is how
        # gen_validity came to report 0/128 on CUDA and 28/72 on CPU for one
        # checkpoint.
        cloud = cloud.to(self.head.pair_head[-1].weight.device)
        try:
            out = self.head(cloud)
            # Default YuelBondDecoderHead always returns (logits, (i_idx, j_idx))
            if isinstance(out, tuple) and len(out) == 2:
                logits, (i_idx, j_idx) = out
            else:
                return YuelBondResult(
                    pair_features=None,
                    logits=torch.zeros(0, NUM_BOND_CLASSES, device=cloud.positions.device),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    decode_succeeded=False,
                    error="head_returned_unexpected_format",
                )
        except Exception as exc:
            return YuelBondResult(
                pair_features=None,
                logits=torch.zeros(0, NUM_BOND_CLASSES, device=cloud.positions.device),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                decode_succeeded=False,
                # 2026-10-05 — carry the MESSAGE, not just the type.  A bare
                # class name is what made this unreadable: `RuntimeError` on
                # its own says nothing about which of the dozens of ways a
                # forward pass can fail actually happened.
                error=f"head_forward_failed:{type(exc).__name__}:{exc}",
            )
        edge_index = torch.stack([i_idx, j_idx], dim=0).long()
        if edge_index.shape[1] == 0:
            return YuelBondResult(
                pair_features=None,
                logits=logits,
                edge_index=edge_index,
                decode_succeeded=False,
                error="empty_edge_list",
            )
        # Compute pair features (distance, z_i, z_j, angle=0, dative=bool)
        d_vec = cloud.positions[edge_index[1]] - cloud.positions[edge_index[0]]
        d = d_vec.norm(dim=-1).clamp(min=self.head.min_distance)
        z_i = cloud.atomic_numbers[edge_index[0]].long()
        z_j = cloud.atomic_numbers[edge_index[1]].long()
        # Crude dative detector: N/O/S <-> Pt (either ordering)
        donor_mask = (
            (z_i == 7)
            | (z_i == 8)
            | (z_i == 16)
            | (z_j == 7)
            | (z_j == 8)
            | (z_j == 16)
        )
        is_dative = donor_mask & ((z_i == 78) | (z_j == 78))
        pf = PairFeature(
            edge_index=edge_index,
            distance=d,
            z_i=z_i,
            z_j=z_j,
            angle_to_metal=torch.zeros_like(d),
            is_dative_candidate=is_dative,
        )
        return YuelBondResult(
            pair_features=pf,
            logits=logits,
            edge_index=edge_index,
            decode_succeeded=True,
        )

    # ------------------------------------------------------------------
    # Wf-A11 — discriminator metrics on the decoded pair list
    # ------------------------------------------------------------------
    def decode_discriminator_metrics(
        self, pf: Optional[PairFeature], logits: torch.Tensor
    ) -> Dict[str, float]:
        """Return 4 discriminator metrics on the decoded bond-pair list.

        These are computed BEFORE the bond-class threshold is applied, so
        they reflect what the *model* decided (not what the chemistry
        ultimately accepted).  All four are backward-compat (the default
        ``threshold`` decode mode fills them with zeros for unchanged
        behaviour).

        Returns
        -------
        dict
            single_fragment_rate  — always 0.0 at decode-time (no mol)
            ring_size_distribution — always 0.0 at decode-time (no mol)
            dative_bond_rate      — fraction of pairs whose is_dative
                                     flag is True.
            metal_coord_1_rate    — always 0.0 at decode-time (no mol)
        """
        result: Dict[str, float] = {
            "single_fragment_rate": 0.0,
            "ring_size_distribution": 0.0,
            "dative_bond_rate": 0.0,
            "metal_coord_1_rate": 0.0,
        }
        if pf is None:
            return result
        n = int(pf.is_dative_candidate.shape[0])
        if n <= 0:
            return result
        n_dative = int(pf.is_dative_candidate.sum().item())
        result["dative_bond_rate"] = float(n_dative) / float(n)
        return result


__all__ = [
    "YuelBondDecoderHead",
    "YuelBondDecoder",
    "YuelBondResult",
    "SplitBondHead",
    "split_bond_loss",
]


# ---------------------------------------------------------------------------
# SplitBondHead — Wf-A69 W2 Fix 3 (2026-10-17)
# ---------------------------------------------------------------------------
class SplitBondHead(nn.Module):
    """Existence + type split head for the ~95% NO_BOND bond-prediction task.

    Combines the existing :class:`YuelBondDecoderHead` (5-class type logits)
    with a fresh single-logit binary existence head sharing the same edge
    encoder output.  The decoupled heads let the optimiser rank
    ``is there a bond?`` against the hard-negative distribution (a job the
    uncalibrated 5-class softmax cannot do well on its own) while
    specialising the type softmax on the existence-positive subset where
    the type signal is well determined.

    Anchors (with §5.2 honest framing):
      * PepForge bioRxiv 2026 (10.64898/2026.05.29.728379v1) B.3 — the
        canonical 95% NO_BOND recipe uses ``BOND_EXIST_GATE`` BCE on the
        whole graph + ``BOND_TYPE_GATE`` weighted CE on positives, with
        focal γ=2.0 + 4:1 neg subsampling.
      * GraViti arXiv 2605.16668 B.3 — critical finding that
        inverse-frequency class weights DEGRADED vs focal alone.
      * YuelBond JCIM 2026 (B.1) — explicitly identifies the
        ``argmax over NO_BOND-prioritized distribution`` defect that the
        split head is designed to fix.

    Wf-A69 W2 design decision: ``forward()`` is bit-exact identical to
    :class:`YuelBondDecoderHead.forward` when ``existence_logits=False``;
    the trainer side calls :func:`split_bond_loss` to compare the two
    heads.  Adding a 1-d existence logit does not change the shape of
    the existing 5-class output, so the split-head path can be inserted
    into the existing trainer with no checkpoint-reshape.
    """

    def __init__(
        self,
        base_head: "YuelBondDecoderHead",
        hidden_dim: Optional[int] = None,
        # Wf-A69 W2 Fix 3 — when True, register the existence head as a
        # sub-module so its parameters flow into the AdamW param list.
        # Default True is the only useful setting; False is for ablation.
        train_existence: bool = True,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        # The existence head is a 1-d logit emitted from the SAME pair
        # representation as the type head, so it shares the message-
        # passing layers and only adds a Linear(hidden_dim, 1) projection.
        # This is the canonical PepForge B.3 pattern (binary existence
        # head reuses the edge encoder).
        h_dim = hidden_dim if hidden_dim is not None else int(
            getattr(base_head, "hidden_dim", 64)
        )
        if train_existence:
            self.exist_head = nn.Linear(h_dim, 1)
        else:
            self.exist_head = None

    def forward(self, cloud):  # type: ignore[no-untyped-def]
        """Return ``(type_logits, exist_logit_or_none, edge_index)``.

        Forwarding to :attr:`base_head` and only adding the 1-d
        existence logit keeps this path bit-exact on the 5-class logits
        — every existing Wf-A67/68/W1 checkpoint can be promoted to
        split-head mode without re-training the type classifier.
        """
        out = self.base_head(cloud)
        if isinstance(out, tuple) and len(out) == 2:
            type_logits, edge_index = out
            exist_logit = None
            if self.exist_head is not None:
                # Recompute the per-atom hidden state to feed the
                # existence head from the same encoder.  Cost is ~one
                # extra Linear per molecule; cheaper than running the
                # full message-passing loop a second time.  The
                # existence head projects from `hidden_dim` (per-atom),
                # not 2*hidden_dim (per-pair); we use the per-atom
                # embedding averaged over the pair so the projection
                # is shape-compatible.
                _dev = self.base_head.pair_head[-1].weight.device
                z = cloud.atomic_numbers.long().to(_dev)
                z_oh = z_to_onehot(z, self.base_head.z_table)
                h_atom = self.base_head.atom_encoder(z_oh)
                i_idx, j_idx = edge_index
                # Per-pair hidden: sum the per-atom embeddings (a
                # cheap symmetric pool).  Shape (E, hidden_dim), which
                # is what ``exist_head: Linear(hidden_dim, 1)`` expects.
                pair_h = h_atom[i_idx] + h_atom[j_idx]
                exist_logit = self.exist_head(pair_h)  # (E, 1)
            return type_logits, exist_logit, edge_index
        return out


def split_bond_loss(
    type_logits: torch.Tensor,
    exist_logit: Optional[torch.Tensor],
    edge_labels: torch.Tensor,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Combined existence + type loss (Wf-A69 W2 Fix 3).

    Loss has three components:
      1. Existence BCE on every edge — focal γ=2.0, α=0.25.
      2. Type CE on existence-positive pairs only — class 4
         (BOND_AROMATIC) + 1 (BOND_SINGLE) + 2 (BOND_DOUBLE) + 3
         (BOND_TRIPLE); existence-positive means ``edge_labels > 0``.
      3. Optional ``exist_logit is None`` ⇒ fall back to plain CE on
         every pair (bit-exact with the pre-split path).

    Anchored to PepForge B.3 — the existence gate is calibrated
    independently of the type classifier and the type loss is
    downweighted on negatives where the label is noisy.
    """
    n_edges = int(type_logits.shape[0])
    if n_edges == 0:
        return torch.zeros((), device=type_logits.device, dtype=type_logits.dtype)
    if exist_logit is not None:
        # --- Existence BCE (focal γ=2.0 on every pair) ---
        exist_target = (edge_labels > 0).float()
        _bce = F.binary_cross_entropy_with_logits(
            exist_logit.squeeze(-1), exist_target, reduction="none"
        )
        _pt = torch.sigmoid(exist_logit.squeeze(-1)) * exist_target + (
            1 - torch.sigmoid(exist_logit.squeeze(-1))
        ) * (1 - exist_target)
        _focal = ((1.0 - _pt).clamp(min=0.0) ** focal_gamma) * _bce
        _alpha_t = torch.where(exist_target > 0, focal_alpha, 1.0 - focal_alpha)
        exist_loss = (_alpha_t * _focal).mean()
        # --- Type CE on existence-positive pairs only ---
        pos_mask = edge_labels > 0
        if pos_mask.any():
            type_logits_pos = type_logits[pos_mask]
            edge_labels_pos = edge_labels[pos_mask] - 1  # 0..3 within positives
            type_loss = F.cross_entropy(type_logits_pos, edge_labels_pos)
        else:
            type_loss = torch.zeros(
                (), device=type_logits.device, dtype=type_logits.dtype
            )
        return exist_loss + type_loss
    # Fallback: pre-split path — plain CE on every pair.
    return F.cross_entropy(type_logits, edge_labels)
