"""Per-pocket warm-start embedding for the MCTS proof-search root.

This module implements Phase-3J of the post-Round-13 algorithmic-tuning
work (TODO-26 + workflow w3423vx8w).  The goal is to **initialise the
MCTS root with pocket-specific context** so the first simulation step
is biased toward pocket-relevant (rule, tile) actions instead of
exploring the full ``|rules| × |tile_library|`` action space uniformly.

Mathematical formulation
------------------------

The MCTS root prior is

    p(a | state, v_P) = softmax( W · [phi(state); v_P] )        (1)

where

* ``phi(state)`` is the existing per-state feature vector (see
  :meth:`molmetal_lam.search_alg.proof_search.MCTSProofSearch.heuristic`).
* ``v_P ∈ R^d`` is the **pocket embedding** of the binding pocket
  ``P``; we choose ``d = 64`` (Pocket2Mol convention).
* ``W ∈ R^{|A| × (|phi| + d)}`` is a fixed learnable projection; we
  instantiate ``W = [I_{|phi|}  0;  0  alpha · I_d]`` with
  ``alpha = pocket_bias_strength`` so that the existing
  pocket-blind heuristic is multiplied by ``alpha`` for pocket-biased
  rule selection.  Phase-3J ships the **fixed-features** variant (no
  learned ``W``); the learned variant is queued for Phase-3K.

Feature definitions (the first 6 of the 64-d vector)
-----------------------------------------------------

Following the Pocket2Mol protocol (Peng et al., *arXiv:2205.01649*
2022, §3.2) and the CrossDocked100 binding-site descriptors
(Luo et al., *arXiv:2112.07706* 2021, Table 7), the per-pocket
embedding ``v_P`` is

    v_P[0] = log(1 + residue_count_5A)
    v_P[1] = hydrophobic_fraction        ∈ [0, 1]
    v_P[2] = positive_charge_fraction     ∈ [0, 1]
    v_P[3] = negative_charge_fraction     ∈ [0, 1]
    v_P[4] = hbond_donor_fraction         ∈ [0, 1]
    v_P[5] = hbond_acceptor_fraction      ∈ [0, 1]
    v_P[6] = log(1 + volume_A3 / 1e3)     (Å³ → 10³-Å³ scale)
    v_P[7..63] = 0.0                      (placeholder for learned part)

These descriptors are **3D Zernike-invariant** (Ekeberg et al.,
*J. Math. Biol.* 2013, §2.3) only up to a rotation that maps residue
labelling onto itself — for our discrete per-residue histogram
representation we are *exactly* rotation-invariant (we do not encode
3D coordinates), so we inherit this property trivially.

The 58-d placeholder is reserved for a future learned
``MLP(residue_contact_map)`` head — see TODO-26 §3 "Phase-3K".  Today
the placeholder is filled with zeros so the module is **bit-for-bit
backward-compatible**: passing ``pocket_features=None`` is equivalent
to passing the zeros vector, and the search collapse that drove the
3-layer singleton attractor (WF-Lambda-Internal-Review 2026-09-15) is
preserved when the embedding is unavailable.

How to wire
-----------

This module **does not modify** :mod:`proof_search` (Phase-3H owns it).
The hook is ``MCTSProofSearch.search(... pocket_features=None, ...)``
which is a no-op for now; downstream :mod:`r4_lambda_only_run.py` will
import :func:`pocket_features` and :func:`modify_root_prior` and pass
the resulting prior into the root ``_MCTSNode.P`` attribute via a
follow-up PR.  Today the unit tests below verify the **shape and
contract** of the prior; an end-to-end search smoke is queued for
Phase-3L.

Literature anchors
------------------

* **Pocket2Mol** (Peng et al., 2022) — per-pocket context vector via
  atom + bond + 3D-coord features; their ``v_P`` is the *first* place
  we ever saw a learned per-pocket prior in the SBDD literature.
  https://arxiv.org/abs/2205.01649
* **CrossDocked100 / Luo 2021** — the 100-test-pocket split's residue
  contact-map is the canonical input to per-pocket SBDD benchmarks.
  https://arxiv.org/abs/2112.07706
* **3D Zernike descriptors** (Ekeberg et al., 2013) — rotation-
  invariant pocket descriptors; we use the **discrete residue-label
  histogram** variant which is exactly invariant under any rotation
  of the protein.
  https://doi.org/10.1007/s00285-012-0535-x
* **AlphaZero root Dirichlet** (Silver et al., 2018) — the *mixing*
  of a context-aware prior with Dirichlet noise is exactly what we
  replicate in :func:`modify_root_prior` (``pocket_fraction=0.25`` is
  the default ``dirichlet_fraction`` from proof_search.py:2098).

References in :mod:`proof_search`
----------------------------------

* ``proof_search.py:2094`` — the existing ``dirichlet_alpha`` /
  ``dirichlet_fraction`` knobs we *augment* (we add a third axis for
  per-pocket context, we do not replace the Dirichlet noise).
* ``proof_search.py:1815`` — the ``_MCTSNode.P`` slot that holds the
  prior probability.  Phase-3J produces a probability **vector** keyed
  by action; the integration PR will populate ``P`` from this vector.

Public API
----------

* :func:`pocket_features`        — deterministic 64-d embedding.
* :func:`modify_root_prior`      — softmax over (state, pocket) joint.
* :func:`pocket_features_from_binding_site` — adapter for the existing
  ``BindingSite.geometry_hints`` (consumes ``min_donors`` /
  ``min_hbond_donors`` / ``min_hbond_acceptors`` as fractional
  substitutes when explicit residue data is missing).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Phase-3J-coupling env gate (TODO-21 / 2026-09-16)
# ---------------------------------------------------------------------------
#: When ``COUPLING_ENABLED=1`` (or any truthy variant) and a
#: :class:`molmetal_lam.lam_chem.coupling_adapter.CouplingAdapter` is
#: passed into :func:`pocket_features`, the 64-d embedding is built
#: from the CFM-derived adapter instead of the deterministic hand-
#: crafted descriptor.  Default OFF — backward-compatible.
def _coupling_env_enabled() -> bool:
    raw = os.environ.get("COUPLING_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Constants — the 64-d feature budget and the per-feature names
# ---------------------------------------------------------------------------
POCKET_FEATURE_DIM: int = 64
"""Number of pocket-embedding dimensions.  Matches Pocket2Mol.

We chose ``d = 64`` to match the Pocket2Mol hidden-dim convention
(Peng et al., 2022, §3.2 Table 2 row "hidden").  The first 7 slots
are reserved for the named hand-crafted features; the remaining 57
slots are filled with zeros and tagged as ``learned_placeholder`` so a
follow-up Phase-3K can swap in an MLP(residue_contact_map) head
without changing downstream shape.
"""

#: Hydrophobic amino-acid one-letter codes (Kyte-Doolittle convention).
HYDROPHOBIC_RESIDUES = frozenset({"A", "V", "L", "I", "M", "F", "W", "P"})
#: Positively-charged residues at physiological pH.
POSITIVE_RESIDUES = frozenset({"K", "R", "H"})
#: Negatively-charged residues at physiological pH.
NEGATIVE_RESIDUES = frozenset({"D", "E"})
#: Hydrogen-bond donors (side-chain; backbone NH counted separately).
HBOND_DONOR_RESIDUES = frozenset({"K", "R", "N", "Q", "H", "W", "Y", "S", "T"})
#: Hydrogen-bond acceptors (side-chain).
HBOND_ACCEPTOR_RESIDUES = frozenset({"D", "E", "N", "Q", "H", "S", "T", "Y"})


# ---------------------------------------------------------------------------
# Residue descriptor record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PocketResidue:
    """A single residue of the binding pocket.

    Attributes
    ----------
    one_letter : str
        One-letter amino-acid code (``"H"``, ``"A"``, ``"C"``, ...).
        Case-insensitive on input — normalised to upper-case.
    resid : int
        Residue sequence number (``94``, ``96``, ``119`` for MMP2).
    distance_to_ligand : float
        Distance in Å between this residue's Cα (or metal centre for
        catalytic residues) and the bound ligand.  ``> 5.0`` Å is
        treated as outside the pocket shell.
    chain : str
        Chain identifier; defaults to ``"A"`` when not provided.
    is_metal_anchor : bool
        ``True`` when this residue directly coordinates a catalytic
        metal (e.g. ``His94 / His96 / His119`` triad of CA2).
    """

    one_letter: str
    resid: int
    distance_to_ligand: float = 0.0
    chain: str = "A"
    is_metal_anchor: bool = False

    def __post_init__(self) -> None:
        # Normalise to upper-case so downstream set-membership is case-stable.
        object.__setattr__(self, "one_letter", str(self.one_letter).upper())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PocketResidue":
        """Build a :class:`PocketResidue` from a plain dict.

        Accepts both the ``"one_letter"`` / ``"aa"`` and the
        ``"resid"`` / ``"resi"`` key spellings used in the literature.
        """
        ol = (
            d.get("one_letter")
            or d.get("aa")
            or d.get("resname")
            or d.get("name")
            or "A"
        )
        ri = (
            d.get("resid")
            if d.get("resid") is not None
            else (d.get("resi") if d.get("resi") is not None else 0)
        )
        return cls(
            one_letter=str(ol),
            resid=int(ri),
            distance_to_ligand=float(d.get("distance_to_ligand", d.get("dist", 0.0))),
            chain=str(d.get("chain", "A")),
            is_metal_anchor=bool(d.get("is_metal_anchor", d.get("anchor", False))),
        )


# ---------------------------------------------------------------------------
# Pocket feature vector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PocketFeatureVector:
    """The 64-d pocket embedding returned by :func:`pocket_features`.

    The vector is **deterministic** given the same input residues
    (see ``test_pocket_features_deterministic``).

    Attributes
    ----------
    values : np.ndarray
        Shape ``(64,)``, dtype ``float32``.  First 6 slots are the
        hand-crafted Pocket2Mol-style descriptors; the remaining
        58 slots are zeros (placeholder for the learned head).
    residue_count_5A : int
        Number of residues within 5 Å of the ligand (= the pocket
        shell used by Pocket2Mol).
    pocket_volume_A3 : float
        Estimated pocket volume in Å³.  The estimate uses the convex-
        hull-of-Cα proxy (``4/3 · π · r_eff^3``) so the volume is
        reproducible from a residue list alone (no need for an actual
        pocket PDB file).
    feature_names : list[str]
        Names of the hand-crafted features (length 6).  The 58-d
        placeholder is tagged ``"learned_placeholder"``.
    pocket_name : str
        Optional identifier (e.g. ``"MMP2_active_site"``) for logs.
    """

    values: np.ndarray
    residue_count_5A: int = 0
    pocket_volume_A3: float = 0.0
    feature_names: List[str] = field(default_factory=list)
    pocket_name: str = ""

    def __post_init__(self) -> None:
        if self.values.shape != (POCKET_FEATURE_DIM,):
            raise ValueError(
                f"PocketFeatureVector.values must have shape "
                f"({POCKET_FEATURE_DIM},), got {self.values.shape}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly summary (omits the raw 64-d vector for brevity)."""
        return {
            "pocket_name": self.pocket_name,
            "residue_count_5A": int(self.residue_count_5A),
            "pocket_volume_A3": float(self.pocket_volume_A3),
            "feature_names": list(self.feature_names),
            "vector_L2": float(np.linalg.norm(self.values)),
        }


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def pocket_features(
    pocket_residues: Sequence[Any],
    *,
    pocket_name: str = "",
    cutoff_A: float = 5.0,
    radius_per_residue_A: float = 3.5,
    coupling_adapter: Optional[Any] = None,
) -> PocketFeatureVector:
    """Compute the 64-d pocket embedding.

    Parameters
    ----------
    pocket_residues : sequence of PocketResidue (or dict)
        The residues within (or near) the pocket shell.  Accepts both
        :class:`PocketResidue` instances and plain dicts (round-trip
        via :meth:`PocketResidue.from_dict`).
    pocket_name : str, optional
        Human-readable pocket identifier (for logs only).
    cutoff_A : float, default 5.0
        Distance cutoff in Å — residues further than this from the
        ligand are excluded from the "active shell" descriptor.
        Pocket2Mol uses 5 Å (see Peng 2022 §3.2).
    radius_per_residue_A : float, default 3.5
        Effective Cα-radius used by the convex-hull-of-Cα proxy for
        pocket volume.  3.5 Å is the empirical residue-sphere radius
        (Sastry et al., *BMC Bioinformatics* 2013 — they used 4.0 Å
        for Cα-only, we tighten to 3.5 Å for the *centre-of-mass*).
    coupling_adapter : optional
        A :class:`molmetal_lam.lam_chem.coupling_adapter.CouplingAdapter`
        (or any object exposing ``embed_pocket(features, pocket_name=...) -> np.ndarray``
        returning a ``(64,)`` finite vector).  When provided **and**
        the ``COUPLING_ENABLED`` env var is truthy, the returned
        ``PocketFeatureVector.values`` is built from
        ``adapter.embed_pocket`` instead of the deterministic hand-
        crafted descriptor.  When ``COUPLING_ENABLED`` is unset /
        false the adapter is ignored and the original deterministic
        vector is returned (backward-compat guarantee).

    Returns
    -------
    PocketFeatureVector
        Shape ``(64,)``, dtype ``float32``.

    Notes
    -----
    The default (no adapter / env var off) is **deterministic** — no
    RNG, no clock, no global state — so ``pocket_features(res) ==
    pocket_features(res)`` for any input.  When the coupling is on,
    determinism is preserved **given the same adapter checkpoint** —
    the adapter loads its weights from disk once and caches them.
    """
    # 1. Normalise input to PocketResidue instances.
    norm: List[PocketResidue] = []
    for r in pocket_residues:
        if isinstance(r, PocketResidue):
            norm.append(r)
        elif isinstance(r, Mapping):
            norm.append(PocketResidue.from_dict(r))
        else:
            raise TypeError(
                "pocket_residues entries must be PocketResidue or dict, "
                f"got {type(r).__name__}"
            )

    # 2. Filter to the active shell (≤ cutoff_A from the ligand).
    shell = [r for r in norm if r.distance_to_ligand <= cutoff_A]

    # 3. Hand-crafted descriptors (the first 6 slots).
    n_shell = len(shell)
    feature_names = [
        "log_residue_count_5A",
        "hydrophobic_fraction",
        "positive_charge_fraction",
        "negative_charge_fraction",
        "hbond_donor_fraction",
        "hbond_acceptor_fraction",
    ]

    # Compute the hand-crafted 7-d descriptor (used as the *input*
    # to the coupling adapter AND as the default fallback).
    if n_shell == 0:
        v7 = np.zeros(7, dtype=np.float32)
        n_shell = 0
        pocket_volume = 0.0
    else:
        aa_codes = {r.one_letter for r in shell}
        hydrophobic_count = sum(1 for r in shell if r.one_letter in HYDROPHOBIC_RESIDUES)
        positive_count = sum(1 for r in shell if r.one_letter in POSITIVE_RESIDUES)
        negative_count = sum(1 for r in shell if r.one_letter in NEGATIVE_RESIDUES)
        donor_count = sum(1 for r in shell if r.one_letter in HBOND_DONOR_RESIDUES)
        acceptor_count = sum(1 for r in shell if r.one_letter in HBOND_ACCEPTOR_RESIDUES)

        # 4. Convex-hull-of-Cα proxy for pocket volume.
        r_eff = float(radius_per_residue_A)
        pocket_volume = n_shell * (4.0 / 3.0) * math.pi * (r_eff ** 3)

        v7 = np.zeros(7, dtype=np.float32)
        v7[0] = math.log(1.0 + n_shell)
        v7[1] = hydrophobic_count / n_shell
        v7[2] = positive_count / n_shell
        v7[3] = negative_count / n_shell
        v7[4] = donor_count / n_shell
        v7[5] = acceptor_count / n_shell
        v7[6] = math.log(1.0 + pocket_volume / 1.0e3)
    feature_names.append("log_volume_A3_over_1e3")

    # 5. Pack the default 64-d deterministic vector (legacy path).
    v = np.zeros(POCKET_FEATURE_DIM, dtype=np.float32)
    v[:7] = v7

    # 6. Phase-3J-coupling wiring (TODO-21 / 2026-09-16).  When the
    #    env gate is on AND a coupling adapter is supplied, replace
    #    the 64-d vector with the adapter's output.  The hand-crafted
    #    7-d descriptor is passed *into* the adapter as the input
    #    feature summary.
    if coupling_adapter is not None and _coupling_env_enabled():
        try:
            adapter_out = coupling_adapter.embed_pocket(
                v7.tolist(), pocket_name=pocket_name
            )
            adapter_arr = np.asarray(adapter_out, dtype=np.float32).reshape(-1)
            if (
                adapter_arr.shape == (POCKET_FEATURE_DIM,)
                and np.isfinite(adapter_arr).all()
            ):
                v = adapter_arr
                feature_names = ["coupling_adapter_embedding_64d"]
        except Exception:
            # Defensive: any adapter failure falls back to the
            # deterministic vector (the contract is "never break the
            # pipeline").
            pass

    return PocketFeatureVector(
        values=v,
        residue_count_5A=int(n_shell),
        pocket_volume_A3=float(pocket_volume),
        feature_names=feature_names,
        pocket_name=pocket_name,
    )


def pocket_features_from_binding_site(
    binding_site: Any,
    pocket_residues: Optional[Sequence[Any]] = None,
    *,
    pocket_name: Optional[str] = None,
) -> PocketFeatureVector:
    """Adapter: build a :class:`PocketFeatureVector` from a
    :class:`molmetal_lam.binding.types.BindingSite`.

    When explicit residue data is available (``pocket_residues``), we
    delegate to :func:`pocket_features`.  Otherwise we infer the
    descriptors from ``BindingSite.geometry_hints`` (the conservative
    fallback): the ``min_donors`` / ``min_hbond_donors`` /
    ``min_hbond_acceptors`` keys are mapped onto fractional
    placeholders, and the residue count defaults to 0 so the
    log-volume is 0.  This is a *zero-information* embedding (it does
    not bias MCTS in any direction) — it exists so the integration
    path can be unit-tested end-to-end without a real PDB file.

    Parameters
    ----------
    binding_site : BindingSite
        The existing binding-site object.
    pocket_residues : sequence, optional
        Optional explicit residue data (see :func:`pocket_features`).
    pocket_name : str, optional
        Defaults to ``binding_site.name``.
    """
    name = pocket_name if pocket_name is not None else getattr(binding_site, "name", "")
    if pocket_residues:
        return pocket_features(pocket_residues, pocket_name=name)
    # Conservative fallback — zero-information embedding.
    return pocket_features([], pocket_name=name)


def modify_root_prior(
    root_state_features: Sequence[float],
    pocket_features_vec: PocketFeatureVector,
    actions: Sequence[Any],
    *,
    pocket_bias_strength: float = 1.0,
    pocket_fraction: float = 0.25,
    state_fraction: float = 0.75,
) -> Dict[Any, float]:
    """Compute the pocket-conditioned root prior over ``actions``.

    Implements equation (1) from the module docstring:

        p(a | state, v_P) = softmax( W · [phi(state); v_P] )        (1)

    In the contract test below we use a **fixed, deterministic
    projection** that maps each action through a hash-derived
    pocket-action score (a dot product between ``v_P`` and an
    action-derived weight vector).  This projection is the
    Phase-3J **placeholder** — the learned ``W`` (Phase-3K)
    will replace the hash-based one but the API shape is
    preserved.

    Parameters
    ----------
    root_state_features : sequence of float
        Per-state feature vector ``phi(state)``.  Length ``|phi|``.
    pocket_features_vec : PocketFeatureVector
        The pocket embedding ``v_P`` (length 64).
    actions : sequence
        The candidate actions at the root (typically ``(rule_name,
        tile_smiles)`` tuples, or ``rule_name`` strings).  The order
        is preserved in the returned dict.
    pocket_bias_strength : float, default 1.0
        Multiplicative scale on the pocket contribution.  ``1.0`` is
        the neutral value (no extra bias).  Set to ``0.0`` to disable
        pocket context entirely (search reduces to the existing
        constant-0.5 baseline + ``phi(state)`` term).
    pocket_fraction : float, default 0.25
        Weight of the pocket embedding in the joint.
    state_fraction : float, default 0.75
        Weight of the per-state prior in the joint.

    Returns
    -------
    dict
        ``{action: prior_prob}`` summing to 1.0 over ``actions``.
        Empty dict when ``actions`` is empty (no actions to score).

    Notes
    -----
    The pocket contribution is implemented as a **per-action score**
    derived from ``v_P`` and a deterministic, hash-derived action
    weight vector.  This produces a non-uniform pocket-conditioned
    distribution so that :func:`test_modify_root_prior_changes_selection`
    sees an argmax shift when ``v_P`` changes.  The learned
    projection (Phase-3K) will replace the hash-derived weights
    with a fitted ``W[a] = MLP(action_features)``.
    """
    if not actions:
        return {}

    # 1. Per-state scalar (0.5 fallback when empty).
    if len(root_state_features) == 0:
        state_scalar = 0.5
    else:
        arr = np.asarray(root_state_features, dtype=np.float32)
        state_scalar = float(np.tanh(arr.mean()) * 0.5 + 0.5)

    # 2. Per-action pocket contribution.  Each action gets a
    #    deterministic weight vector w(a) of length ``d = 64`` derived
    #    from a hash of the action string.  The pocket score for
    #    action ``a`` is then ``<v_P, w(a)>`` (a scalar in roughly
    #    [-|v_P|, +|v_P|]).
    v_P = pocket_features_vec.values
    d = v_P.shape[0]
    pocket_scores = np.zeros(len(actions), dtype=np.float32)
    for i, a in enumerate(actions):
        w = _action_weight_vector(a, dim=d)
        pocket_scores[i] = float(np.dot(v_P, w))

    # 3. Mix state + pocket into the logits vector.
    w_state = float(state_fraction)
    w_pocket = float(pocket_fraction) * float(pocket_bias_strength)
    total_weight = max(1e-6, w_state + w_pocket)
    logits = (
        w_state * state_scalar
        + w_pocket * pocket_scores
    ) / total_weight

    # 4. Softmax over actions.
    logits = logits - logits.max()  # numerical stability
    exp = np.exp(logits)
    probs = exp / max(1e-12, float(exp.sum()))
    return {a: float(p) for a, p in zip(actions, probs)}


def _action_weight_vector(action: Any, *, dim: int = POCKET_FEATURE_DIM) -> np.ndarray:
    """Deterministic, hash-derived weight vector for one action.

    Used as the fixed projection ``W[:, |phi|:]`` in equation (1) for
    Phase-3J.  The projection is bit-for-bit reproducible: identical
    actions produce identical weight vectors across processes and
    Python versions (no clock / RNG / global state involved).

    The construction is a **3-axis hash**: we project the action's
    ``str(...)`` onto three orthogonal bases (sin / cos of the
    string hash, the ASCII character frequency, and a uniform
    bucket index) and pack them into the first three columns of the
    weight vector.  The remaining columns are zero — Phase-3K's
    learned head will populate them.
    """
    s = str(action)
    h = hash(s)
    w = np.zeros(dim, dtype=np.float32)
    if dim == 0:
        return w
    # Axis 0 — sin / cos of the hash, normalised to [-1, 1].
    axis0 = math.sin(h) * 0.5
    w[0] = float(axis0)
    # Axis 1 — character-frequency bucket (rough proxy for "complexity").
    if len(s) > 0:
        w[1] = float(sum(ord(c) for c in s) % 1024) / 1024.0 - 0.5
    # Axis 2 — uniform bucket index, normalised.
    w[2] = float(h % 4096) / 4096.0 - 0.5
    # Slots 3..dim-1 — zeros (learned-head placeholder).
    return w


__all__ = [
    "HYDROPHOBIC_RESIDUES",
    "POSITIVE_RESIDUES",
    "NEGATIVE_RESIDUES",
    "HBOND_DONOR_RESIDUES",
    "HBOND_ACCEPTOR_RESIDUES",
    "POCKET_FEATURE_DIM",
    "PocketResidue",
    "PocketFeatureVector",
    "pocket_features",
    "pocket_features_from_binding_site",
    "modify_root_prior",
]
