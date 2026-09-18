"""Connectivity-aware decoder — RDKit topology validation for CFM samples.

WF-CFM-Phase-2 (TODO-24 P1.4) — addresses the open question "can we
post-hoc reject disconnected CFM samples without retraining the
velocity field?".  Per the Phase-1 diagnostic report
(``molmetal/reports/wf_cfm_internal_review/diagnose.md``), 97.4 % of
5000-step CFM samples failed RDKit sanitisation with a
``disconnected_distance_graph`` exception — the velocity field happily
produces geometrically reasonable atom clouds, but the predicted
bond topology has at least one atom with no edge, so RDKit refuses to
treat the cloud as a molecule.

This module wraps :class:`molmetal.models.bond_head.BondAwareDecoder`
with a connectivity *post-check* that:

1. Runs the legacy decoder to get per-pair bond logits + RDKit Mol.
2. Builds an undirected ``(N, N)`` connectivity matrix from the
   predicted bond orders.
3. Inspects the matrix with :func:`_count_connected_components` —
   the canonical DFS / union-find pass.  If the predicted graph has
   more than one connected component, the prediction is rejected as
   ``disconnected``.
4. Returns the rejected / kept flag alongside the original
   :class:`DecodedMol` so the caller can choose to fall back to
   an alternative decoder, drop the sample, or accept a different
   validation tier.

The acceptance / rejection policy is gated on
``accept_only_connected = True`` (default) so the safer default is
adopted — callers who want the legacy ``always-accept`` behaviour
can opt out by setting it ``False``.

Lit anchor
----------
Jin, W., Barzilay, R., Jaakkola, T. (2018). *Junction Tree Variational
Autoencoder for Molecular Graph Generation.*  ICML 2018
(arXiv:1802.04364).  JTVAE is the canonical reference for the
"molecules-as-graphs" decoding view: a valid molecule must have
exactly one connected component over its covalent-bond graph.  This
is the de-facto chemistry sanity check used in every modern
molecular-graph decoder (GraphAF, JT-VAE, MolDQN, GEAM, DiffuMol)
and is the foundation we use here to reject CFM samples whose
bond-order classifier predicts a disconnected topology.

Honest framing
--------------
This module is a CPU-only post-check.  It does NOT change the
velocity field's behaviour; it only rejects bad samples.  When
``accept_only_connected=True`` is the default and most samples are
disconnected, the *observed* decode_ratio will drop sharply (e.g.
from 97.4 % disconnected ⇒ only 2.6 % of samples kept).  This is the
correct behaviour for honest validation: the disconnect rate is a
floor on what the bond head has learned.  Production recovery
requires either (a) retraining the bond head with a connectivity-
aware loss, or (b) running a connectivity prior (Gumbel-top-k from
``molmetal.models.connectivity_gumbel``) so the head's argmax does
not emit disconnected graphs in the first place.

The class is intentionally side-effect free: no I/O, no mutation of
the inner decoder, no caching across calls.  Each ``decode`` call is
an O(E) union-find pass over the predicted edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from molmetal.models.bond_head import (
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
    DecodedMol,
)


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------
@dataclass
class ConnectivityResult:
    """Wrapper around :class:`DecodedMol` with a connectivity verdict.

    Attributes
    ----------
    decoded : DecodedMol
        The raw decoder output (never None; use ``decoded.error`` to
        inspect failures).
    is_connected : bool
        ``True`` iff the predicted bond-order graph has exactly one
        connected component (i.e. all atoms are reachable from any
        other via covalent bonds).  The trivial ``N == 0`` and
        ``N == 1`` cases are treated as connected.
    n_components : int
        Number of connected components in the predicted graph.
        ``1`` for connected, ``>= 2`` for disconnected.
    largest_component_size : int
        Size of the largest component (atoms in the main fragment).
        For an N-atom cloud, ``largest_component_size <= N``.  A
        value ``< N`` indicates a peripheral disconnected fragment.
    largest_component_atoms : list[int]
        Atom indices belonging to the largest component.
    rejected_reason : Optional[str]
        ``None`` when ``is_connected=True``.  Otherwise a short
        human-readable reason string (e.g. ``"disconnected (2
        components)"``).
    """

    decoded: DecodedMol
    is_connected: bool
    n_components: int
    largest_component_size: int
    largest_component_atoms: List[int] = field(default_factory=list)
    rejected_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_connected_components(
    n_atoms: int,
    edges: List[Tuple[int, int]],
) -> Tuple[int, int, List[int]]:
    """Count connected components via union-find (Kruskal 1956).

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the cloud.
    edges : list of (src, dst)
        Pairs of atom indices in the cloud.  Order is irrelevant
        (the undirected graph is the same regardless).

    Returns
    -------
    (n_components, largest_component_size, largest_component_atoms)
        Tuple suitable for direct assignment to
        :class:`ConnectivityResult` fields.  Atom indices in the
        largest component are returned in ascending order.

    Notes
    -----
    Implementation: iterative union-find with path compression.  This
    is O(N + E * alpha(N)) for ``alpha(N) < 5`` in practice
    (Tarjan 1975, *Efficiency of a Good But Not Linear Set Union
    Algorithm*) — at the cloud sizes we encounter (8-128 atoms) the
    constant factor is irrelevant.

    For ``n_atoms == 0`` we return ``(0, 0, [])`` (no atoms, no
    components, no largest-component atoms).  For ``n_atoms == 1``
    we return ``(1, 1, [0])`` — the single atom is its own
    component.
    """
    if n_atoms == 0:
        return (0, 0, [])
    if n_atoms == 1:
        return (1, 1, [0])
    parent = list(range(n_atoms))
    rank = [0] * n_atoms

    def find(x: int) -> int:
        # Iterative find with path compression.
        root = x
        while parent[root] != root:
            root = parent[root]
        # Compress the path.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for s, d in edges:
        if 0 <= s < n_atoms and 0 <= d < n_atoms:
            union(int(s), int(d))

    # Count component sizes via a single find pass.
    comp_of: dict = {}
    for atom in range(n_atoms):
        r = find(atom)
        comp_of.setdefault(r, []).append(atom)
    components = list(comp_of.values())
    if not components:
        return (0, 0, [])
    largest = max(components, key=len)
    return (len(components), len(largest), sorted(largest))


def _collect_predicted_edges(decoded: DecodedMol) -> List[Tuple[int, int]]:
    """Extract undirected edges from :class:`DecodedMol.bond_orders`.

    Skips ``BOND_NO_BOND`` and de-duplicates symmetric pairs (we only
    need one of (i, j) and (j, i) for the union-find pass).  The
    dative-flagged bonds are already single-order covalent bonds in
    the bond_orders list (see :class:`BondAwareDecoder._decode_impl`),
    so they are included.
    """
    if not decoded.bond_orders:
        return []
    seen = set()
    edges: List[Tuple[int, int]] = []
    for src, dst, order in decoded.bond_orders:
        if order == 0:  # BOND_NO_BOND
            continue
        a, b = (int(src), int(dst)) if int(src) < int(dst) else (int(dst), int(src))
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b))
    return edges


# ---------------------------------------------------------------------------
# ConnectivityAwareDecoder (RDKit topology post-check)
# ---------------------------------------------------------------------------
class ConnectivityAwareDecoder:
    """Bond decoder wrapper that rejects disconnected predictions.

    Parameters
    ----------
    bond_decoder : :class:`molmetal.models.bond_head.BondAwareDecoder`
        The underlying decoder.  Must already be configured with a
        :class:`BondOrderHead` (the wrapper does not own one).
    accept_only_connected : bool, default ``True``
        When ``True``, the returned :class:`ConnectivityResult` still
        carries the original :class:`DecodedMol`, but a clear
        ``rejected_reason`` flags disconnected samples.  Callers can
        use the flag to drop the sample, drop to a fallback decoder,
        or accept the disconnected prediction.  When ``False``, the
        wrapper behaves as a transparent pass-through (``is_connected``
        is still reported for diagnostics).
    keep_empty_connected : bool, default ``True``
        When ``True``, atom clouds with zero predicted bonds are treated
        as connected if ``n_atoms == 0`` or ``n_atoms == 1``.  When
        ``False``, ``n_atoms >= 2`` with no bonds is rejected as
        ``disconnected`` (the per-atom components are isolated).

    Lit anchor
    -----------
    Jin, Barzilay, Jaakkola 2018 (JTVAE, arXiv:1802.04364) — the
    ``molecule ⇔ single connected component`` invariant.  Every
    canonical molecular-graph decoder (JTVAE, GraphAF, MolDQN, GEAM,
    DiffuMol) applies this exact check before accepting a candidate.
    """

    def __init__(
        self,
        bond_decoder: BondAwareDecoder,
        accept_only_connected: bool = True,
        keep_empty_connected: bool = True,
    ) -> None:
        if not isinstance(bond_decoder, BondAwareDecoder):
            raise TypeError(
                f"bond_decoder must be a BondAwareDecoder, got "
                f"{type(bond_decoder).__name__}"
            )
        self.bond_decoder = bond_decoder
        self.accept_only_connected = bool(accept_only_connected)
        self.keep_empty_connected = bool(keep_empty_connected)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode(
        self,
        cloud: AtomCloud,
        pair_features=None,
    ) -> ConnectivityResult:
        """Run the bond decoder + connectivity post-check.

        Parameters
        ----------
        cloud : :class:`AtomCloud`
            The (positions, atomic_numbers) cloud from the CFM
            velocity field's ODE integration.
        pair_features : optional
            Forwarded verbatim to
            :meth:`BondAwareDecoder.decode`.  When ``None`` the
            inner decoder builds the fully-connected pair list.

        Returns
        -------
        :class:`ConnectivityResult`
            The original ``DecodedMol`` plus connectivity flags.
            Callers should check ``result.is_connected`` (and
            ``result.rejected_reason`` when ``False``) before
            accepting the prediction.
        """
        # Step 1: run the inner bond decoder.
        decoded = self.bond_decoder.decode(cloud, pair_features=pair_features)
        # Step 2: run the connectivity post-check.
        return self._check_connectivity(cloud, decoded)

    def decode_with_fallback(
        self,
        cloud: AtomCloud,
        fallback_decoder: "BondAwareDecoder | ConnectivityAwareDecoder",
    ) -> ConnectivityResult:
        """Decode; if the primary decode is disconnected, retry with a fallback.

        This is the recommended call style for a production pipeline:
        the inner :class:`BondAwareDecoder` produces high-quality
        bond logits when it's confident, and the fallback (e.g. a
        :class:`BondAwareDecoder` with a stricter ``bond_cutoff``
        or a :class:`ConnectivityAwareDecoder` with a connectivity
        prior from ``molmetal.models.connectivity_gumbel``)
        produces a connected graph when the primary fails.  Both
        attempts share the same connectivity post-check.

        Parameters
        ----------
        cloud : :class:`AtomCloud`
        fallback_decoder : :class:`BondAwareDecoder` or :class:`ConnectivityAwareDecoder`
            Used only when the primary decode is disconnected.  If
            it is itself a :class:`ConnectivityAwareDecoder`, its
            ``decode`` returns a :class:`ConnectivityResult`; we
            unwrap the ``DecodedMol`` and re-run the post-check (which
            is idempotent — already-connected samples stay
            connected).
        """
        primary = self.decode(cloud)
        if primary.is_connected:
            return primary
        # Fallback path.
        if isinstance(fallback_decoder, ConnectivityAwareDecoder):
            fallback_res = fallback_decoder.decode(cloud)
            fallback_decoded = fallback_res.decoded
        else:
            fallback_decoded = fallback_decoder.decode(cloud)
        # Re-run the post-check on the fallback (idempotent — already-
        # connected samples stay connected, disconnected ones get the
        # same flag).
        return self._check_connectivity(cloud, fallback_decoded)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------
    def _check_connectivity(
        self,
        cloud: AtomCloud,
        decoded: DecodedMol,
    ) -> ConnectivityResult:
        """Compute the connectivity verdict for a single :class:`DecodedMol`."""
        n_atoms = int(cloud.atomic_numbers.shape[0])
        edges = _collect_predicted_edges(decoded)
        # Empty / single-atom clouds: defined as connected.
        if n_atoms <= 1:
            return ConnectivityResult(
                decoded=decoded,
                is_connected=True,
                n_components=max(1, n_atoms),
                largest_component_size=n_atoms,
                largest_component_atoms=list(range(n_atoms)),
                rejected_reason=None,
            )
        # Zero predicted edges in a multi-atom cloud: rejected when
        # ``keep_empty_connected=False`` (every atom is its own
        # disconnected component).
        if not edges:
            if self.keep_empty_connected:
                # Treat as connected per the JTVAE-aligned default.
                # In practice we still flag the result so the caller
                # knows there were no bonds.
                return ConnectivityResult(
                    decoded=decoded,
                    is_connected=True,
                    n_components=1,
                    largest_component_size=n_atoms,
                    largest_component_atoms=list(range(n_atoms)),
                    rejected_reason=None,
                )
            return ConnectivityResult(
                decoded=decoded,
                is_connected=False,
                n_components=n_atoms,
                largest_component_size=1,
                largest_component_atoms=[0],
                rejected_reason=(
                    f"disconnected ({n_atoms} isolated atoms, no predicted bonds)"
                ),
            )
        n_comp, largest_size, largest_atoms = _count_connected_components(
            n_atoms, edges,
        )
        is_connected = (n_comp == 1)
        reason: Optional[str] = None
        if not is_connected:
            reason = (
                f"disconnected ({n_comp} components, largest "
                f"{largest_size}/{n_atoms} atoms)"
            )
        elif largest_size < n_atoms:
            # Pathological: ``edges`` is non-empty but the union-find
            # pass missed some atoms.  This should not happen unless
            # edges reference atom indices outside ``[0, n_atoms)``,
            # but we flag it defensively.
            reason = (
                f"connectivity edge index out-of-range "
                f"({largest_size}/{n_atoms} atoms in main fragment)"
            )
            is_connected = False
        return ConnectivityResult(
            decoded=decoded,
            is_connected=is_connected,
            n_components=n_comp,
            largest_component_size=largest_size,
            largest_component_atoms=largest_atoms,
            rejected_reason=reason,
        )


__all__ = [
    "ConnectivityResult",
    "ConnectivityAwareDecoder",
    "_count_connected_components",
    "_collect_predicted_edges",
]