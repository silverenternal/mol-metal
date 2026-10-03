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
from typing import Dict, List, Optional, Tuple

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
_COVALENT_VALENCE: Dict[int, int] = {
    1: 1,  # H
    6: 4,  # C
    7: 3,  # N (sp3-amine; aromatic N counts as 3 with one double)
    8: 2,  # O
    9: 1,  # F
    15: 3,  # P
    16: 2,  # S
    17: 1,  # Cl
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
    """

    decoded: DecodedMol
    yuelbond_result: YuelBondResult
    bonds_accepted: int = 0
    bonds_rejected_overvalent: int = 0


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
        # Step 3: assemble into an RDKit Mol.
        decoded = self._assemble_mol(cloud, bond_orders, dative_bonds)
        return YuelBondDecodeResult(
            decoded=decoded,
            yuelbond_result=yres,
            bonds_accepted=n_kept,
            bonds_rejected_overvalent=n_overvalent,
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
        # Per-pair argmax — predict discrete bond order.
        predicted = torch.argmax(logits, dim=-1).tolist()
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
            cap_i = remaining[i]
            cap_j = remaining[j]
            if order > cap_i or order > cap_j:
                n_overvalent += 1
                continue
            bond_orders.append((i, j, order))
            remaining[i] -= order
            remaining[j] -= order
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
        sanitized = False
        smiles = ""
        try:
            Chem.SanitizeMol(mol)
            sanitized = True
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            sanitized = False
            smiles = ""
        return DecodedMol(
            mol=mol,
            smiles=smiles,
            bond_orders=list(bond_orders),
            dative_bonds=list(dative_bonds),
            n_atoms=cloud.positions.shape[0],
            n_bonds=len(bond_orders),
            sanitized=sanitized,
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
        return z in (78, 44, 30, 77, 29, 79, 46, 26, 28)

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
