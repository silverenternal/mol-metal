"""PLL feature extraction adapter for the CFM trainer (Wf-382-A1 SCAFFOLD).

This module is the *adapter-layer* wrapper around the canonical PLL
implementation at :mod:`molmetal.molmetal_lam.lam_chem.pll_signature`
(Liu & Feng 2026, 684 LOC) so that downstream CFM trainers (R20-P3
ship target) can consume PLL features without a direct cross-module
dependency on ``molmetal_lam``.

The adapter is SCAFFOLD-only (Wf-382-A1).  It exposes three thin
functions that are bit-equivalent to the canonical implementation up
to a constant: the underlying math is unchanged.

Mathematical prior
------------------
For a 3D conformer of an RDKit ``Chem.Mol``, the *Persistent Local
Laplacian* (PLL) spectrum is the top-k eigenvalues of the *normalised*
graph Laplacian of a kNN graph on the heavy-atom point cloud, with
edges weighted by the heat kernel ``1 / (1 + d^2)``.  See
:func:`molmetal.molmetal_lam.lam_chem.pll_signature.compute_pll` for
the full reference implementation.

Public surface
--------------
* :func:`persistent_laplacian`   wrapper returning a frozen dataclass
* :func:`pll_eigenvalues`         descending eigenvalues array
* :func:`pll_to_feature_vector`   fixed-length feature vector (default top_k=10)
* :class:`PLLFeatureError`        raised on missing conformer / n_atoms < 2

References
----------
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein–Ligand Binding Affinities".  *J. Chem. Inf. Model.* 66 (3),
  1024-1037.  DOI: 10.1021/acs.jcim.6c02460.
* Belkin M. (2018) "Persistent Laplacian".  *Proc. ICML* 80:1752-1761.
* von Luxburg U. (2007) "A Tutorial on Spectral Clustering".
  *Stat. Comput.* 17:395-416.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Lazy canonical import — the underlying module may be slow to import
# (scipy + RDKit + lru_cache).  We defer until first call so that the
# adapter module itself is cheap to import.
try:
    from molmetal.molmetal_lam.lam_chem.pll_signature import (
        PLLSignature,
        PLLSignatureError,
        compute_pll,
    )

    _PLL_CANONICAL_AVAILABLE = True
except ImportError:  # pragma: no cover — best-effort lazy fallback
    _PLL_CANONICAL_AVAILABLE = False

    class PLLSignatureError(ValueError):
        """Raised when the canonical PLL module is unavailable."""

    @dataclass(frozen=True)
    class PLLSignature:
        """Stub PLLSignature used only when canonical module is unavailable."""

        eigenvalues: np.ndarray
        eigenvectors: np.ndarray
        n_atoms: int
        k_requested: int
        k_used: int
        smiles: str

    def compute_pll(mol, k: int = 16, k_nn: int = 8, smiles: Optional[str] = None):
        raise PLLSignatureError(
            "Canonical PLL module (molmetal.molmetal_lam.lam_chem.pll_signature) "
            "is not importable. Install scipy + RDKit or restore the canonical "
            "module to enable PLL feature extraction."
        )


# ---------------------------------------------------------------------------
# Adapter-local error type — re-export the canonical one when available.
# ---------------------------------------------------------------------------


class PLLFeatureError(ValueError):
    """Raised when a PLL feature cannot be extracted by the adapter.

    Inherits from ``ValueError`` so existing ``except ValueError`` blocks
    catch it; the type remains distinct so callers can branch on
    adapter-specific failures (missing 3D conformer, canonical module
    unavailable, mismatched top_k) without sniffing exception messages.
    """


# ---------------------------------------------------------------------------
# Adapter-local dataclass — frozen, JSON-friendly, downstream-trainer-ready
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLLFeatureAdapter:
    """Adapter view of a PLL signature, tailored for downstream trainers.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Length-``top_k`` vector of the *largest* PLL eigenvalues,
        sorted in **descending** order.  dtype ``float64``.  When the
        molecule has fewer atoms than ``top_k``, the tail is zero-padded
        (canonical behaviour for the Wf-359-A1 W₂ recipe).
    n_atoms : int
        Number of heavy atoms used to build the kNN graph.
    k_used : int
        Effective rank returned by the canonical ``compute_pll`` —
        ``min(k_requested, n_atoms - 1)``.
    smiles : str
        Canonical SMILES of the molecule (empty string if RDKit is
        missing).
    """

    eigenvalues: np.ndarray
    n_atoms: int
    k_used: int
    smiles: str

    def as_feature_dict(self) -> dict:
        """Return a JSON-friendly dict for downstream trainer consumption."""
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "n_atoms": int(self.n_atoms),
            "k_used": int(self.k_used),
            "smiles": self.smiles,
        }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def persistent_laplacian(
    mol,
    k: int = 16,
    k_nn: int = 8,
    smiles: Optional[str] = None,
) -> PLLFeatureAdapter:
    """Compute the adapter view of the PLL signature for ``mol``.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule with at least one 3D conformer (call
        ``AllChem.EmbedMolecule(mol)`` first).
    k : int
        Number of PLL eigenvalues requested from the canonical module.
        Default 16 matches Liu & Feng 2026 Table 2.
    k_nn : int
        Number of nearest neighbours per atom.  Default 8.
    smiles : str, optional
        Canonical SMILES used as cache key.  When ``None`` we derive
        it from ``mol`` via ``Chem.MolToSmiles``.

    Returns
    -------
    PLLFeatureAdapter
        Frozen dataclass with the canonical PLL signature projected
        onto the adapter view (eigenvalues + n_atoms + k_used + smiles).

    Raises
    ------
    PLLFeatureError
        * if RDKit or scipy is missing,
        * if ``mol`` has no conformer,
        * if ``n_atoms < 2``,
        * if ``k < 1``.
    """
    if not _PLL_CANONICAL_AVAILABLE:  # pragma: no cover — defensive
        raise PLLFeatureError(
            "Canonical PLL module is unavailable — cannot compute PLL features."
        )
    if k < 1:
        raise PLLFeatureError(f"k must be >= 1; got k={k}")

    try:
        sig = compute_pll(mol, k=k, k_nn=k_nn, smiles=smiles)
    except PLLSignatureError as exc:
        # Re-raise as the adapter-specific error type so callers can branch
        raise PLLFeatureError(str(exc)) from exc

    return PLLFeatureAdapter(
        eigenvalues=sig.eigenvalues.copy(),
        n_atoms=int(sig.n_atoms),
        k_used=int(sig.k_used),
        smiles=str(sig.smiles),
    )


def pll_eigenvalues(pll: PLLFeatureAdapter) -> np.ndarray:
    """Return the descending PLL eigenvalues for ``pll``.

    Parameters
    ----------
    pll : PLLFeatureAdapter
        Output of :func:`persistent_laplacian`.

    Returns
    -------
    np.ndarray
        Length-``pll.k_used`` descending eigenvalues array.  dtype
        ``float64``.  The returned reference is the *same* array as
        ``pll.eigenvalues`` — callers that mutate should ``.copy()``
        first.
    """
    if not isinstance(pll, PLLFeatureAdapter):
        raise PLLFeatureError(
            f"pll must be a PLLFeatureAdapter; got {type(pll).__name__}"
        )
    return pll.eigenvalues


def pll_to_feature_vector(
    eigenvalues: Sequence[float] | np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Project a PLL eigenvalue vector to a fixed-length feature vector.

    When ``len(eigenvalues) < top_k``, the tail is zero-padded (the
    Wf-359-A1 W₂ recipe contract — see
    ``molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json``).  When
    ``len(eigenvalues) > top_k``, the leading ``top_k`` entries are
    kept (the canonical descending sort means the *largest* eigenvalues
    carry the most discriminative signal per von Luxburg 2007 §4.3).

    Parameters
    ----------
    eigenvalues : Sequence[float] or np.ndarray
        Descending PLL eigenvalues (output of :func:`pll_eigenvalues`).
    top_k : int
        Target feature length.  Default 10 — chosen to match the
        Wf-382-A1 task brief.

    Returns
    -------
    np.ndarray
        Length-``top_k`` ``float64`` vector.
    """
    if top_k < 1:
        raise PLLFeatureError(f"top_k must be >= 1; got top_k={top_k}")

    arr = np.asarray(eigenvalues, dtype=np.float64)
    if arr.ndim != 1:
        raise PLLFeatureError(
            f"eigenvalues must be 1-D; got shape {arr.shape}"
        )

    if arr.size >= top_k:
        return arr[:top_k].copy()
    # Zero-pad the tail — the canonical descending sort means padding
    # the small-eigenvalue end is the least-discriminative choice.
    padded = np.zeros(top_k, dtype=np.float64)
    padded[: arr.size] = arr
    return padded


__all__: List[str] = [
    "PLLFeatureAdapter",
    "PLLFeatureError",
    "persistent_laplacian",
    "pll_eigenvalues",
    "pll_to_feature_vector",
]
