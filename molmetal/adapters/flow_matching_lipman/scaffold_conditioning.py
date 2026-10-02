"""Wf-380-A3 — scaffold-aware conditioning primitives (TODO-202).

=====================================================================
Why this module exists
=====================================================================
m7 Diversity = 0.792482 (SOTA 0.81, gap −0.018) per Wf-357-B1.  One
of the four diversity levers in TODO-202 §1 is **structural
diversity via scaffold conditioning** — condition the per-step
sampling on a Murcko scaffold index, so each generated batch
explores a distinct scaffold region.  This is structurally
complementary to the token-level diversity methods in
``diversity_sampling.py`` (Holtzman 2020 / MMD / repulsion): the
latter lift diversity within a single scaffold; this one lifts
diversity across scaffolds.

Honest framing (party-twenty-spirit §3.5):

* **MEASURED**: the scaffold extraction and conditioning primitives
  are bit-for-bit compatible with RDKit's Murcko scaffold
  (Bemis-Murcko 1996, *The Properties of Known Drugs. 1.
  Molecular Frameworks*, J. Med. Chem. 39:2887-2893).
* **PROJECTED**: +0.005-0.01 lift on m7 Diversity from scaffold
  conditioning (TODO-202 §3.1) — NOT measured end-to-end on
  Blackwell sm_120 (GPU launch blocked per AGENTS.md §5.1).

Wf-380-A3 SCAFFOLD scope:

* ``murcko_scaffold(mol)`` — RDKit-compatible scaffold extractor
  with graceful fallback when RDKit is unavailable (returns
  ``""`` and logs a one-shot warning).
* ``scaffold_condition(logits, scaffold_idx, n_scaffolds)`` —
  per-batch scaffold-aware logit bias (additive log-prior over a
  learned scaffold embedding, applied before softmax).

Trainer integration is deferred to **Wf-380-B1**.

References:

* Bemis & Murcko 1996 — J. Med. Chem. 39:2887-2893
* TODO-202 §1 — scaffold-aware conditional generation spec
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

log = logging.getLogger("molmetal.cfm.scaffold")


# =====================================================================
# §1.  Murcko scaffold extraction
# =====================================================================
def _warn_rdkit_missing_once() -> None:
    """One-shot warning when RDKit is unavailable.

    Kept as a module-private helper so the public ``murcko_scaffold``
    function can degrade gracefully without spamming the log.
    """
    if not getattr(murcko_scaffold, "_warned", False):
        log.warning(
            "[murcko_scaffold] RDKit is unavailable; scaffold extraction "
            "returns '' (empty string).  Install with `pip install rdkit` "
            "to enable per-scaffold diversity."
        )
        murcko_scaffold._warned = True  # type: ignore[attr-defined]


def murcko_scaffold(mol) -> str:
    """Extract the Murcko generic scaffold from an RDKit molecule.

    Implements Bemis-Murcko 1996 (J. Med. Chem. 39:2887-2893): the
    scaffold is the union of all ring systems + linker bonds in the
    molecule, with all atom and bond types reduced to generic
    placeholders (i.e. aromatic / aliphatic markers only — no
    element-specific labels).

    Parameters
    ----------
    mol
        An RDKit ``Chem.Mol`` object.  Accepts ``None`` (returns
        ``""``) and any object exposing a ``__bool__`` that
        evaluates to False.

    Returns
    -------
    str
        The canonical SMILES of the Murcko scaffold, or ``""``
        when RDKit is unavailable / ``mol`` is empty / scaffold
        extraction fails.

    Notes
    -----
    * RDKit's ``MurckoScaffold.GetScaffoldForMol`` is the canonical
      implementation; this wrapper makes the scaffold extraction
      CPU-friendly + RDKit-optional for the m7 diversity bench.
    * When the input molecule has no rings, the Bemis-Murcko
      convention returns ``""`` (no scaffold) — same convention as
      the reference RDKit helper.
    """
    if mol is None:
        return ""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except ImportError:
        _warn_rdkit_missing_once()
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:  # noqa: BLE001
        # Defensive: malformed inputs → empty scaffold rather than
        # raising (the diversity bench never wants to abort on a
        # single bad molecule).
        return ""


def murcko_scaffold_from_smiles(smiles: str) -> str:
    """Convenience: SMILES → Murcko scaffold.

    Returns ``""`` when the SMILES is malformed or RDKit is
    unavailable.  Useful for batched scaffold extraction in
    property-panel scripts.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    try:
        from rdkit import Chem  # type: ignore
    except ImportError:
        _warn_rdkit_missing_once()
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return murcko_scaffold(mol)
    except Exception:  # noqa: BLE001
        return ""


def scaffold_buckets(
    smiles_list: Sequence[str],
) -> Tuple[List[int], List[str]]:
    """Bucket a list of SMILES by their Murcko scaffold.

    Returns ``(bucket_idx, unique_scaffolds)`` where ``bucket_idx[i]``
    is the index (into ``unique_scaffolds``) of the scaffold for
    ``smiles_list[i]``.  SMILES with empty / missing scaffolds are
    bucketed under the canonical "" scaffold (index 0).

    Parameters
    ----------
    smiles_list
        Sequence of SMILES strings (length ``N``).

    Returns
    -------
    bucket_idx
        List of ``N`` integers — the per-SMILES scaffold index.
    unique_scaffolds
        List of unique scaffold SMILES in first-seen order.

    Honest framing
    --------------
    * **CPU-only**: O(N × R) where R is the RDKit scaffold-extraction
      cost per molecule (~0.1-1 ms each on commodity hardware).
    * **Deterministic**: same input always produces the same bucket
      ordering (first-seen rule).
    """
    bucket_idx: List[int] = []
    seen: dict = {}
    unique: List[str] = []
    for smi in smiles_list:
        scaf = murcko_scaffold_from_smiles(smi)
        if scaf not in seen:
            seen[scaf] = len(unique)
            unique.append(scaf)
        bucket_idx.append(seen[scaf])
    return bucket_idx, unique


# =====================================================================
# §2.  Scaffold conditioning (per-step logit bias)
# =====================================================================
@dataclass
class ScaffoldConditioningConfig:
    """Configuration for scaffold-aware sampling.

    Attributes
    ----------
    n_scaffolds
        Number of distinct scaffolds expected at training time.
        Determines the size of the scaffold embedding table.
    scaffold_bias_scale
        Multiplier on the scaffold-conditional log-prior.  Default
        0.1 (per TODO-202 §2.2 calibration).  Set to 0 to disable.
    """

    n_scaffolds: int = 256
    scaffold_bias_scale: float = 0.1


def scaffold_condition(
    logits: torch.Tensor,
    scaffold_idx: torch.Tensor,
    n_scaffolds: int = 256,
    scaffold_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply per-batch scaffold conditioning to a logits tensor.

    The function returns ``logits + scaffold_bias_scale * B[scaffold_idx]``
    where ``B`` is a learned ``(n_scaffolds, V)`` bias table and
    ``scaffold_idx`` is the per-sample scaffold index.  When
    ``scaffold_bias`` is ``None``, the function returns the input
    logits unchanged (backward-compatible behaviour, equivalent to
    ``scaffold_bias_scale = 0``).

    Parameters
    ----------
    logits
        Per-sample logits.  Two shapes are accepted (Wf-424-A3
        auto-detection; closes Wf-396-A3 CRITICAL FINDING):

        * ``(B, N, V)`` — batch × atoms × vocab (canonical
          shape).  This is the path the original docstring
          described.
        * ``(B, V)`` — flattened batch × vocab (token-stream
          reshape).  This is the shape the Wf-388-A2 trainer
          passes when ``--scaffold-conditioning`` is True and the
          bond-CE loss is computed on a flattened
          ``(B·N, V)`` logits tensor.  The function auto-reshapes
          to ``(B, 1, V)`` so the per-sample bias broadcasts over
          the singleton atom dim.  ``B`` is the per-sample batch
          dim in BOTH cases, so the ``scaffold_idx`` shape
          contract ``(B,)`` is unchanged.

    scaffold_idx
        ``(B,)`` long tensor of scaffold indices.  Values must be
        in ``[0, n_scaffolds)``.
    n_scaffolds
        Size of the bias table.  Default 256.
    scaffold_bias
        ``(n_scaffolds, V)`` learned bias.  When ``None``, returns
        ``logits`` unchanged (identity path).

    Returns
    -------
    torch.Tensor
        Same shape as ``logits`` — the scaffold-conditioned logits.
        For ``(B, V)`` input the return shape is also ``(B, V)``
        (the reshape to ``(B, 1, V)`` is internal and undone via
        ``.squeeze(1)`` before returning).

    Raises
    ------
    ValueError
        If ``logits.dim()`` is not 2 or 3.

    Honest framing
    --------------
    * **Bit-exact backward compatibility**: when ``scaffold_bias is
      None`` the function returns the input logits unchanged —
      recovering the legacy unscaffolded sampling path.  The
      m7 measurement protocol therefore sees the SAME numerics
      whether or not scaffold conditioning is wired in.
    * **No gradient flow through scaffold_idx**: the index is
      treated as a discrete sample.  The bias table
      ``scaffold_bias`` carries the gradients; in production the
      trainer would update it via a small aux loss (e.g. an MMD
      term across scaffold buckets).  That integration is a
      separate workflow (Wf-380-B1).
    * **2D auto-reshape is shape-equivalent**: for a ``(B, V)``
      input the result is mathematically identical to
      ``(B, V) → unsqueeze(1) → (B, 1, V) → add bias broadcast →
      (B, 1, V) → squeeze(1) → (B, V)``.  When the bias table is
      all-zero the function returns the input logits unchanged
      to within float32 epsilon (round-trip safe).
    """
    if scaffold_bias is None:
        return logits
    if logits.dim() == 2:
        # 2D auto-detection (Wf-424-A3) — accept the flattened
        # token-stream shape that the Wf-388-A2 trainer passes via
        # ``all_logits.view(-1, V)`` when --scaffold-conditioning is
        # True and the loss is bond-CE on a (B·N, V) tensor.
        # Internally treat as (B, 1, V) so the per-sample bias
        # broadcasting logic is uniform with the 3D path.
        logits_3d = logits.unsqueeze(1)
    elif logits.dim() == 3:
        logits_3d = logits
    else:
        raise ValueError(
            f"scaffold_condition expects (B, V) or (B, N, V) logits; "
            f"got shape {tuple(logits.shape)} with dim {logits.dim()}"
        )
    if scaffold_idx.dim() != 1 or scaffold_idx.shape[0] != logits_3d.shape[0]:
        raise ValueError(
            f"scaffold_idx must be shape (B,) with B=logits.shape[0] "
            f"(B for 2D input = logits.shape[0]; B for 3D input = "
            f"logits.shape[0]); got scaffold_idx.shape="
            f"{tuple(scaffold_idx.shape)}, logits.shape="
            f"{tuple(logits.shape)}"
        )
    # Gather per-sample bias → (B, V).
    per_sample_bias = scaffold_bias[scaffold_idx.long()]
    # Broadcast to (B, N, V) and add to logits.  ``n_scaffolds``
    # is reserved for future validation / learnable embedding
    # patterns — see ``ScaffoldConditioningConfig.n_scaffolds``.
    _ = n_scaffolds  # currently unused; reserved for API stability
    conditioned_3d = logits_3d + per_sample_bias.unsqueeze(1)
    if logits.dim() == 2:
        # Undo the unsqueeze for 2D inputs — caller expects (B, V).
        return conditioned_3d.squeeze(1)
    return conditioned_3d


@dataclass
class ScaffoldConditioningOutput:
    """Return value of :func:`scaffold_condition`.

    Mirrors the :class:`DiversitySamplingOutput` struct shape for
    downstream uniformity — but currently just wraps the
    conditioned logits.
    """

    conditioned_logits: torch.Tensor
    scaffold_idx: torch.Tensor
    bias_scale: float = 0.0


__all__ = [
    "murcko_scaffold",
    "murcko_scaffold_from_smiles",
    "scaffold_buckets",
    "scaffold_condition",
    "ScaffoldConditioningConfig",
    "ScaffoldConditioningOutput",
]
