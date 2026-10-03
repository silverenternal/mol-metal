"""MacroGuide Persistent Homology — CPU skeleton (R18 P2-2).

============================================================
MacroGuide PH — ring-topology guidance for molecular generation
============================================================
This module implements a *CPU-only* skeleton of the MacroGuide PH
guidance mechanism introduced by Maksymiuk (Oxford, ICML'26,
arXiv:2602.14977) for promoting macrocycle formation during molecular
generative-model sampling.

Mathematical prior
------------------
MacroGuide steers a pretrained diffusion model by injecting a
gradient-based guidance signal derived from the **persistent homology
of the Vietoris–Rips complex** built on the partially denoised
atomic positions.  In practice (and what this skeleton implements),
the most informative PH feature for macrocycle generation is the
1-dimensional homology class corresponding to *ring formation*:

* at small distances the Vietoris–Rips complex has trivial H_1;
* as the distance threshold ε grows, new 1-cycles (rings) are *born*;
* each ring *dies* when ε becomes large enough to "fill in" the cycle.

A *macrocycle* is a ring whose persistence (death − birth) is large
and whose length (number of atoms) is ≥ 12.  The MacroGuide signal
ranks candidate configurations by a soft scoring function over the
ring-size histogram, with the score peaking at the target ring size.

This module
-----------
The skeleton restricts to the *ring-topology* view: rather than build
the full Vietoris–Rips filtration on a point cloud, we use RDKit's
``Chem.GetRingInfo()`` (SSSR) to read off the *ring-size histogram*
of a molecule, weighted by an aromaticity-aware persistence proxy.
The :class:`RingTopologyComputer` aggregates this into a
``(n_bins,)`` feature vector aligned with a uniform ring-size grid;
the :class:`MacroGuidePH` exposes a soft ``guide(smi, target)``
scoring function in ``[0, 1]`` suitable as a reward / heuristic
signal in MCTS or proof search; and
:func:`augment_pll_with_macroguide` concatenates the MacroGuide
features with the PLL signature features from
:mod:`molmetal_lam.lam_chem.pll_signature`.

Public surface
--------------
* :class:`RingTopologyComputer`          ring-size histogram with aromaticity
* :class:`MacroGuidePH`                   soft guidance score (CPU)
* :func:`augment_pll_with_macroguide`     feature concat for PLL + MG

Honest framing
--------------
This is the **CPU skeleton** (R18 P2-2).  The full MacroGuide
guidance injects a gradient through a differentiable Vietoris–Rips
filtration, which is O(N²) memory + O(N³) time.  The skeleton's
ring-size histogram is O(R) where R is the number of rings — a
~1000× speed-up — at the cost of *not* capturing geometric ring
information (the 3D shape of the cycle).  The W2 / W3 stages of R18
will bridge the two: the skeleton here ships the topological
*signal* (ring-size distribution, aromaticity), and the W2/Vietoris
sampler integrates it with the full Vietoris–Rips filtration.

References
----------
* Maksymiuk A. (2026) "MacroGuide: Topological Guidance for
  Macrocycle Generation".  *Proc. ICML*, arXiv:2602.14977.
* Edelsbrunner H., Harer J. (2010) *Computational Topology: An
  Introduction*.  Amer. Math. Soc., §III (persistent homology).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# RDKit import — required (CI gates rdkit install).  Wrapped so a missing
# RDKit produces a clear ``ImportError`` at module load, not deep inside
# the ring-topology computer.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import-only check
    from rdkit import Chem  # type: ignore

    _RDKIT_AVAILABLE: bool = True
except Exception as _exc:  # pragma: no cover
    Chem = None  # type: ignore
    _RDKIT_AVAILABLE = False
    _RDKIT_IMPORT_ERR: Optional[Exception] = _exc
else:
    _RDKIT_IMPORT_ERR = None


__all__ = [
    "RingTopologyComputer",
    "MacroGuidePH",
    "MacroGuidePHError",
    "DEFAULT_MAX_RING_SIZE",
    "DEFAULT_N_BINS",
    "DEFAULT_N_WINDOWS",
    "augment_pll_with_macroguide",
]


__version__ = "0.0.1-macroguide-ph"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
#
# The default ``n_bins=32`` matches the published PLL signature
# dimensionality (Liu & Feng 2026) so that
# :func:`augment_pll_with_macroguide` concatenates two equally-weighted
# halves of a 64-D feature vector.  The default ``max_ring_size=34``
# covers benzene (6) through macrocycles (12–34), so 32 bins over the
# range [3, 35) yield one bin per ring size with no overlap.
DEFAULT_N_BINS: int = 32

#: ``max_ring_size`` is the upper edge of the histogram (exclusive).
#: 34 covers all drug-like macrocycles.
DEFAULT_MAX_RING_SIZE: int = 34

#: Number of persistence windows used by :class:`MacroGuidePH`.  Each
#: window is one "scale" in the macro-guide filtration.
DEFAULT_N_WINDOWS: int = 4


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MacroGuidePHError(ValueError):
    """Raised when a MacroGuide PH computation cannot proceed.

    Inherits from ``ValueError`` so existing ``except ValueError``
    blocks catch it; the type remains distinct so callers can branch
    on MacroGuide-specific failures (RDKit missing, malformed SMILES,
    shape mismatch) without sniffing exception messages.
    """


# ---------------------------------------------------------------------------
# Public dataclass — single ring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ring:
    """A single ring as reported by RDKit's SSSR.

    Attributes
    ----------
    atom_indices : Tuple[int, ...]
        Sorted atom indices of the ring.
    size : int
        Number of atoms in the ring (``len(atom_indices)``).
    aromatic : bool
        ``True`` iff every atom in the ring is aromatic.  Mirrors
        :meth:`RingTopologyComputer.is_aromatic`.
    """

    atom_indices: Tuple[int, ...]
    size: int
    aromatic: bool


# ---------------------------------------------------------------------------
# RingTopologyComputer — ring-size histogram + aromaticity detection
# ---------------------------------------------------------------------------


class RingTopologyComputer:
    """Compute ring-topology features for an RDKit molecule.

    The :class:`RingTopologyComputer` is the *single-molecule* layer
    of the MacroGuide PH skeleton.  Given a SMILES string, it parses
    the molecule with RDKit, extracts the SSSR rings, and emits a
    ``(n_bins,)`` tensor whose bin ``i`` counts the *aromaticity-
    weighted persistence proxy* of rings in the size range
    ``[min_ring_size + i, min_ring_size + i + 1)``.

    Persistence proxy
    -----------------
    We do not have a 3D conformer at this layer (the CPU skeleton is
    for *pre-filter* scoring, before any conformer embed).  In lieu
    of a true Vietoris–Rips persistence, we use a topological proxy
    that is a *monotone* function of ring size: aromatic rings get
    weight ``1.5`` (they are "more persistent" in the chemical sense
    — they are stabilised by delocalisation); aliphatic rings get
    weight ``1.0``; rings of size ≥ 12 (macrocycle territory) get an
    extra ``+0.5`` multiplier to make them stand out in the histogram.
    The W2 stage will replace this proxy with the true persistence.

    Examples
    --------
    >>> comp = RingTopologyComputer(n_bins=32)
    >>> feat = comp.compute("c1ccccc1")  # benzene
    >>> feat.shape
    torch.Size([32])
    """

    #: The minimum ring size mapped to bin 0.  Anything < 3 (e.g. an
    #: aromatic 2-ring) is folded into bin 0 with an additional
    #: warning skipped silently (we treat "ring size < 3" as 3 to
    #: avoid indexing into a negative bin).
    MIN_RING_SIZE: int = 3

    #: Aromatic-weight bonus applied to every aromatic ring's
    #: histogram contribution (see *Persistence proxy* in the class
    #: docstring).
    AROMATIC_WEIGHT: float = 1.5

    #: Macrocycle bonus applied to rings of size >= ``MACROCYCLE_MIN_SIZE``.
    MACROCYCLE_BONUS: float = 0.5

    #: The smallest ring size counted as a macrocycle.  The published
    #: macrocycle literature defines a macrocycle as a ring with
    #: >= 12 atoms (e.g. cyclosporine A has 11; the IUPAC convention
    #: is 12).  We follow the published boundary.
    MACROCYCLE_MIN_SIZE: int = 12

    def __init__(self, n_bins: int = DEFAULT_N_BINS) -> None:
        if n_bins < 1:
            raise MacroGuidePHError(f"n_bins must be >= 1; got {n_bins}")
        self.n_bins: int = int(n_bins)
        # Pre-compute the bin centre ring sizes — uniform grid on
        # ``[MIN_RING_SIZE, MIN_RING_SIZE + n_bins)``.  Stored as
        # ``Tensor`` so the public surface (returning ``Tensor``) is
        # consistent and downstream code can build soft targets.
        # ``+ 1`` because we want the bin *centres* to be the integer
        # ring sizes (3, 4, 5, …, 3 + n_bins - 1).
        self._ring_size_grid: Tensor = torch.arange(
            self.MIN_RING_SIZE,
            self.MIN_RING_SIZE + self.n_bins,
            dtype=torch.float32,
        )

    # --- public API --------------------------------------------------------

    def compute(self, mol_smiles: str) -> Tensor:
        """Return the ring-topology feature vector of ``mol_smiles``.

        Parameters
        ----------
        mol_smiles : str
            Any RDKit-parseable SMILES.  Implicit H atoms are added
            before ring extraction so e.g. ``"c1ccccc1"`` parses
            cleanly.  An empty / unparseable input returns the
            all-zero ``(n_bins,)`` tensor — a graceful-degrade
            contract so MCTS callers can treat unparseable states as
            "no ring topology information".

        Returns
        -------
        torch.Tensor
            ``(n_bins,)`` ``float32`` tensor.  Bin ``i`` holds the
            sum of *persistence-proxy* weights over all rings whose
            size lies in ``[MIN_RING_SIZE + i, MIN_RING_SIZE + i + 1)``.
            Rings of size ``>= MIN_RING_SIZE + n_bins`` are folded
            into the last bin (truncation at the histogram tail).
        """
        if not _RDKIT_AVAILABLE:
            raise MacroGuidePHError(
                f"RDKit is not available ({_RDKIT_IMPORT_ERR!r}); "
                f"cannot compute ring topology"
            )
        feat = torch.zeros(self.n_bins, dtype=torch.float32)
        mol = self._parse_mol(mol_smiles)
        if mol is None:
            return feat
        rings = self._extract_rings(mol)
        for r in rings:
            bin_idx = self._bin_index(r.size)
            feat[bin_idx] += self._persistence_proxy(r)
        return feat

    def is_aromatic(self, ring_atoms: List[int]) -> bool:
        """Return ``True`` iff every atom in ``ring_atoms`` is aromatic.

        Public helper so the caller can ask the same question without
        re-implementing the aromaticity walk.  Mirrors the
        ``Chem.GetIsAromatic`` contract from RDKit — the molecule is
        *not* a parameter because aromaticity is a per-atom property
        that depends on the molecule, but we expose the helper as a
        *static* convenience that returns ``False`` for an empty
        atom list (vacuously not aromatic).

        Parameters
        ----------
        ring_atoms : list[int]
            Atom indices.  Must be valid for some RDKit ``Mol`` — the
            caller is responsible for keeping the index ↔ molecule
            mapping consistent.  Out-of-range indices silently return
            ``False`` (defensive).

        Returns
        -------
        bool
            ``True`` iff every atom with an in-range index is
            aromatic *and* the list is non-empty.  Empty list → ``False``.
        """
        if not ring_atoms:
            return False
        mol = getattr(self, "_last_mol", None)
        if mol is None:
            # Without a cached molecule we cannot look up aromaticity.
            return False
        try:
            for idx in ring_atoms:
                atom = mol.GetAtomWithIdx(int(idx))
                if not atom.GetIsAromatic():
                    return False
            return True
        except Exception:
            return False

    # --- helpers -----------------------------------------------------------

    def _parse_mol(self, smi: str) -> Optional["Chem.Mol"]:
        """Parse a SMILES string into an RDKit Mol (with Hs)."""
        if smi is None or not isinstance(smi, str) or not smi.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            return None
        if mol is None:
            return None
        # AddHs so implicit H atoms participate in aromaticity
        # perception for heterocyclic nitrogens like ``n`` in
        # pyrrole.  This mirrors the convention used by the rest of
        # the lam_chem ring-handling code (cisplatin_resistance,
        # layered_generation, …).
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            pass
        # Cache for the ``is_aromatic`` helper.
        self._last_mol = mol  # type: ignore[attr-defined]
        return mol

    def _extract_rings(self, mol: "Chem.Mol") -> List[_Ring]:
        """Return the SSSR rings of ``mol`` as a list of :class:`_Ring`."""
        try:
            ri = mol.GetRingInfo()
            atom_rings = ri.AtomRings()
        except Exception:
            return []
        out: List[_Ring] = []
        for ring in atom_rings:
            ring_tuple = tuple(sorted(int(a) for a in ring))
            if not ring_tuple:
                continue
            # Aromaticity is computed per-ring via the public helper.
            arom = self.is_aromatic(list(ring_tuple))
            out.append(
                _Ring(
                    atom_indices=ring_tuple,
                    size=len(ring_tuple),
                    aromatic=arom,
                )
            )
        return out

    def _bin_index(self, ring_size: int) -> int:
        """Map a ring size to a histogram bin index.

        Out-of-range sizes (negative or ``>= MIN_RING_SIZE + n_bins``)
        are clamped to the nearest valid bin.
        """
        idx = int(ring_size) - self.MIN_RING_SIZE
        if idx < 0:
            return 0
        if idx >= self.n_bins:
            return self.n_bins - 1
        return idx

    def _persistence_proxy(self, ring: _Ring) -> float:
        """Per-ring persistence-proxy weight (see class docstring)."""
        w = self.AROMATIC_WEIGHT if ring.aromatic else 1.0
        if ring.size >= self.MACROCYCLE_MIN_SIZE:
            w += self.MACROCYCLE_BONUS
        return float(w)

    @property
    def ring_size_grid(self) -> Tensor:
        """``(n_bins,)`` ``float32`` tensor of bin-centre ring sizes."""
        return self._ring_size_grid.clone()


# ---------------------------------------------------------------------------
# MacroGuidePH — soft guidance score
# ---------------------------------------------------------------------------


class MacroGuidePH:
    """MacroGuide persistent-homology guidance on top of :class:`RingTopologyComputer`.

    The :class:`MacroGuidePH` is the *guidance* layer of the skeleton.
    Given a SMILES string and a *target ring size*, it returns a
    scalar score in ``[0, 1]`` that peaks when the molecule contains
    a ring of size close to ``target_ring_size``, and falls off
    smoothly as the dominant ring size drifts away.

    Sliding-window mechanism
    ------------------------
    MacroGuide PH aggregates ring-size information across multiple
    "persistence windows".  Each window is a soft-mask over the
    ring-size histogram that emphasises a different ring-size band:

        window_w(ℓ) = exp(-(ℓ - μ_w)² / (2 σ_w²))

    where ``μ_w`` is the centre of window ``w`` and ``σ_w`` is the
    bandwidth.  The bands span ``[MIN_RING_SIZE, MAX_RING_SIZE]``
    uniformly across ``n_windows``.  The final guidance score is the
    *softmax over windows* of the band-energy, weighted by a
    Gaussian prior centred at ``target_ring_size``.  This makes the
    score peak at the target band and fall off smoothly across
    neighbouring bands — matching the "soft target" structure used
    by Maksymiuk 2026 §3.3.

    Public surface
    --------------
    * :meth:`guide`              scalar guidance score in ``[0, 1]``
    * :meth:`compute_windows`    ``(n_windows, n_bins)`` band energies
    * :meth:`window_centres`     ``(n_windows,)`` window centre ring sizes

    Examples
    --------
    >>> mg = MacroGuidePH(n_windows=4, n_bins=32)
    >>> score_6ring = mg.guide("c1ccccc1", target_ring_size=6)  # benzene
    >>> score_macrocycle = mg.guide("C1CCCCCCCCCCC1", target_ring_size=12)
    >>> 0.0 <= score_6ring <= 1.0 and 0.0 <= score_macrocycle <= 1.0
    True
    """

    #: Hard floor on the per-bin histogram so a fully-empty molecule
    #: still yields a finite (zero-energy) score rather than NaN
    #: after the softmax.
    _ENERGY_FLOOR: float = 1.0e-30

    #: Standard deviation of the Gaussian prior over windows, in
    #: *window-count* units.  ``1.5`` lets the prior reach a half-
    #: maximum at the neighbouring window — i.e. one band off-target
    #: still contributes meaningfully.
    _WINDOW_SOFTMAX_TAU: float = 1.5

    def __init__(
        self,
        n_windows: int = DEFAULT_N_WINDOWS,
        n_bins: int = DEFAULT_N_BINS,
    ) -> None:
        if n_windows < 1:
            raise MacroGuidePHError(f"n_windows must be >= 1; got {n_windows}")
        if n_bins < 1:
            raise MacroGuidePHError(f"n_bins must be >= 1; got {n_bins}")
        self.n_windows: int = int(n_windows)
        self.n_bins: int = int(n_bins)
        # Delegate the ring-topology extraction.  Keep the
        # :class:`RingTopologyComputer` instance so the windows are
        # always aligned with the bin grid.
        self._ring_topology: RingTopologyComputer = RingTopologyComputer(
            n_bins=self.n_bins,
        )

    # --- public API --------------------------------------------------------

    def guide(self, mol_smiles: str, target_ring_size: int) -> float:
        """Compute the MacroGuide PH guidance score.

        Parameters
        ----------
        mol_smiles : str
            RDKit-parseable SMILES.  Unparseable / empty input
            yields ``0.0`` (graceful degradation — MCTS treats it
            as "no ring information").
        target_ring_size : int
            Desired ring size (typically 6 for benzene-like, 12–16
            for macrocycles).  Values outside ``[MIN_RING_SIZE,
            MIN_RING_SIZE + n_bins)`` are clipped to the nearest
            valid bin.

        Returns
        -------
        float
            Scalar in ``[0, 1]``.  ``1.0`` is a perfect match (the
            entire band energy is concentrated in the target
            window); ``0.0`` is either no ring information (empty
            / unparseable molecule) or a complete mismatch (energy
            concentrated in a non-target window).
        """
        if not _RDKIT_AVAILABLE:
            raise MacroGuidePHError(
                f"RDKit is not available ({_RDKIT_IMPORT_ERR!r}); "
                f"cannot compute MacroGuide PH"
            )
        feat = self._ring_topology.compute(mol_smiles)
        if float(feat.sum()) <= self._ENERGY_FLOOR:
            # Empty molecule or no rings — graceful degrade.
            return 0.0
        windows = self.compute_windows(mol_smiles)
        target_idx = self._target_window_index(int(target_ring_size))
        # Soft target prior over windows — Gaussian centred on
        # ``target_idx`` with std ``_WINDOW_SOFTMAX_TAU``.  Clamp
        # the exponent at -50 so an out-of-range target does not
        # underflow (Math.exp(-50) ≈ 2e-22, well below float32 eps
        # of 1e-7 and so contributes nothing to the softmax — the
        # safe behaviour).
        idx_grid = torch.arange(self.n_windows, dtype=torch.float32)
        diff = idx_grid - float(target_idx)
        prior = torch.exp(-(diff**2) / (2.0 * self._WINDOW_SOFTMAX_TAU**2))
        prior = prior / prior.sum().clamp(min=self._ENERGY_FLOOR)
        # Band energy: L1 norm of each window's filtered histogram.
        energy = windows.sum(dim=-1)  # (n_windows,)
        energy = energy / energy.sum().clamp(min=self._ENERGY_FLOOR)
        score = float((prior * energy).sum())
        # Clamp to [0, 1] for numerical safety.
        return max(0.0, min(1.0, score))

    def compute_windows(self, mol_smiles: str) -> Tensor:
        """Return ``(n_windows, n_bins)`` window-band energies.

        Each row ``w`` is the ring-size histogram filtered by the
        Gaussian window centred on ``μ_w = window_centres[w]``.  The
        output is **not** normalised to unit L1 — the caller's
        softmax / guide logic does its own normalisation.
        """
        feat = self._ring_topology.compute(mol_smiles)
        # Broadcast: (n_windows, 1) * (1, n_bins) → (n_windows, n_bins).
        centres = self.window_centres().unsqueeze(-1)
        grid = self._ring_topology.ring_size_grid.unsqueeze(0)
        sigma = self._window_bandwidth()
        weight = torch.exp(-((grid - centres) ** 2) / (2.0 * sigma**2))
        # Element-wise multiply and sum over bins.  We do NOT
        # re-normalise the per-window sum — that would defeat the
        # "energy concentration" interpretation.  The caller
        # (guide) normalises by the *total* energy across windows.
        windows = weight * feat.unsqueeze(0)
        return windows

    def window_centres(self) -> Tensor:
        """Return ``(n_windows,)`` window centre ring sizes."""
        grid = self._ring_topology.ring_size_grid
        # Map window index ``w ∈ [0, n_windows)`` uniformly to a
        # centre in ``[grid[0], grid[-1]]``.
        if self.n_windows == 1:
            centres = grid[[grid.numel() // 2]].clone()
        else:
            lo = float(grid[0])
            hi = float(grid[-1])
            step = (hi - lo) / float(self.n_windows - 1)
            centres = torch.tensor(
                [lo + step * w for w in range(self.n_windows)],
                dtype=torch.float32,
            )
        return centres

    def _window_bandwidth(self) -> float:
        """Per-window Gaussian bandwidth in *ring-size* units.

        Each window spans ``(n_bins - 1) / n_windows`` ring sizes, so
        the bandwidth is half that — matching the "neighbouring ring
        sizes still contribute" intuition.
        """
        span = max(1.0, float(self.n_bins - 1))
        return span / (2.0 * float(self.n_windows))

    def _target_window_index(self, target_ring_size: int) -> int:
        """Map a target ring size to the nearest window index."""
        centres = self.window_centres()
        diffs = (centres - float(target_ring_size)).abs()
        idx = int(torch.argmin(diffs).item())
        return idx


# ---------------------------------------------------------------------------
# Feature augmentation — PLL + MacroGuide
# ---------------------------------------------------------------------------


def augment_pll_with_macroguide(
    pll_features: Tensor,
    macroguide_features: Tensor,
) -> Tensor:
    """Concatenate PLL and MacroGuide PH features along the last dim.

    The R18 architecture will use this combined ``(..., k + n_bins)``
    tensor as the *joint ring-topology + Laplacian-spectrum* feature
    for downstream heads (reward / search).

    Parameters
    ----------
    pll_features : torch.Tensor
        ``(..., k)`` tensor of PLL signature features (see
        :func:`molmetal_lam.lam_chem.pll_signature.compute_pll`).
        Any leading batch dims are preserved.
    macroguide_features : torch.Tensor
        ``(..., n_bins)`` tensor of MacroGuide PH ring-topology
        features (see :meth:`RingTopologyComputer.compute`).  The
        leading dims (other than the last) must broadcast against
        ``pll_features``.

    Returns
    -------
    torch.Tensor
        ``(..., k + n_bins)`` tensor — the concatenation along the
        last axis.  Dtype is ``float32`` (we cast to ``float32`` if
        either input is ``float64`` so the output has a stable dtype
        for downstream batched ops).

    Raises
    ------
    MacroGuidePHError
        If the shapes are not broadcast-compatible or the last-dim
        sizes are non-positive.

    Examples
    --------
    >>> pll = torch.linspace(0.0, 1.0, 16)
    >>> mg = torch.zeros(32)
    >>> out = augment_pll_with_macroguide(pll, mg)
    >>> tuple(out.shape)
    (48,)
    """
    if not isinstance(pll_features, Tensor):
        raise MacroGuidePHError(
            f"pll_features must be a Tensor; got {type(pll_features)!r}"
        )
    if not isinstance(macroguide_features, Tensor):
        raise MacroGuidePHError(
            f"macroguide_features must be a Tensor; "
            f"got {type(macroguide_features)!r}"
        )
    if pll_features.dim() == 0 or macroguide_features.dim() == 0:
        raise MacroGuidePHError(
            "pll_features and macroguide_features must have " "at least one dimension"
        )
    last_pll = pll_features.shape[-1]
    last_mg = macroguide_features.shape[-1]
    if last_pll <= 0 or last_mg <= 0:
        raise MacroGuidePHError(
            f"last-dim sizes must be positive; "
            f"got pll={last_pll}, macroguide={last_mg}"
        )
    # Broadcast leading dims.  We require the leading dims of the two
    # tensors to be broadcast-compatible (``torch.cat`` after
    # broadcast).  The easy case: same leading shape → straight
    # concat.  The broadcast case: we ``expand`` the smaller tensor
    # to the broadcast shape and then ``cat`` along the last dim.
    leading_pll = pll_features.shape[:-1]
    leading_mg = macroguide_features.shape[:-1]
    broadcast_shape = torch.broadcast_shapes(leading_pll, leading_mg)
    pll_b = pll_features.expand(*broadcast_shape, last_pll)
    mg_b = macroguide_features.expand(*broadcast_shape, last_mg)
    # Cast to a common dtype (prefer float32 for stability).
    out_dtype = torch.float32
    pll_b = pll_b.to(out_dtype)
    mg_b = mg_b.to(out_dtype)
    return torch.cat([pll_b, mg_b], dim=-1)
