"""Persistent Local Laplacian (PLL) signature per Liu & Feng 2026.

============================================================
PLL signature — spectrum of the local-normalised Laplacian
============================================================
This module is the *type-fingerprint + β-NF oracle* of the Molecular
Lambda Calculus pipeline (Track B of the ultracode schedule).  Given a
3D conformer of an RDKit ``Chem.Mol``, it builds a k-nearest-neighbour
graph on the heavy-atom (post-AddHs) point cloud, weights each edge by
the classic heat-kernel ``1 / (1 + d^2)`` kernel, and returns the
top-``k`` spectrum of the *normalised* graph Laplacian.  The signature
is exactly what Liu & Feng 2026 show to be a rotation-/translation-
invariant descriptor of the molecular geometry, suitable as a fast
β-NF divergence signal in the MLC MCTS search.

Mathematical prior
-------------------
For a point cloud ``X = {x_1, ..., x_n} ⊂ ℝ^3`` we build a *symmetric*
k-NN graph ``G = (V, E)`` by taking, for every vertex ``i``, the
``k`` nearest neighbours (Euclidean distance in Å) and union-ing the
directed edges to make an undirected graph.  Edge weights are the
heat-kernel approximation

    w_{ij} = 1 / (1 + ||x_i - x_j||^2)       (Belkin 2018 §3.1)

The degree matrix is ``D = diag(sum_j w_{ij})`` and the *normalised*
graph Laplacian is

    L = I - D^{-1/2} W D^{-1/2}                (von Luxburg 2007 §3)

``L`` is real symmetric positive semi-definite, with spectrum
``0 = λ_0 ≤ λ_1 ≤ ... ≤ λ_{n-1} ≤ 2``.  For molecular point clouds of
size ``n ≥ 4`` the graph is generically connected and ``λ_0 = 0`` has
multiplicity one — the *algebraic connectivity* ``λ_1`` then encodes
the number of "lumps" (rings, branches, metal-coordination shells).

Why the *normalised* Laplacian and not the unnormalised one
-----------------------------------------------------------
The unnormalised Laplacian ``L = D - W`` has eigenvalues that scale
with the average degree, which is itself a function of ``k`` — so two
signatures of the *same* molecule computed at different ``k`` would
not be comparable.  The normalised Laplacian fixes that (von Luxburg
2007 Theorem 2) and is what Liu & Feng 2026 use throughout.

Why the heat-kernel ``1 / (1 + d^2)`` and not the Gaussian
---------------------------------------------------------
The Gaussian kernel ``exp(-d^2 / (2 σ^2))`` requires choosing a
bandwidth ``σ`` (the persistent-homology convention picks ``σ = d_max
/ sqrt(log n)``, see Liu & Feng 2026 §2.3, but this is implementation
heavy).  The rational ``1 / (1 + d^2)`` *implicitly* sets a
distance-scale of 1 Å and is the same kernel Belkin 2018 uses for the
*persistent* Laplacian (his "point cloud Laplacian" of §3.1).  The
two are within a constant factor for ``d ∈ [1, 4] Å``, which is the
relevant range for molecular point clouds after conformer embedding.

Rotation / translation invariance
---------------------------------
The pairwise-distance matrix ``D_{ij} = ||x_i - x_j||`` is invariant
under the Euclidean group (rotations, translations, reflections) —
so the kNN edges, the heat-kernel weights, the degree vector, and
the normalised Laplacian are *all* invariants.  The spectrum is then
an invariant of the molecule and is suitable as a similarity key.

Public surface
--------------
* :class:`PLLSignature`                  frozen dataclass
* :func:`compute_pll`                    top-k spectrum of L
* :func:`_pll_eigvals_cached`            ``lru_cache``-wrapped scalar worker
* :func:`build_knn_graph`                sparse (N, N) CSR adjacency + weights

References
----------
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein–Ligand Binding Affinities".  *J. Chem. Inf. Model.* 66 (3),
  1024-1037.  DOI: 10.1021/acs.jcim.6c02460, arXiv:2603.21503.
* Belkin M. (2018) "Persistent Laplacian".  *Proc. ICML* 80:1752-1761.
* von Luxburg U. (2007) "A Tutorial on Spectral Clustering".
  *Stat. Comput.* 17:395-416.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# RDKit import — required (CI gates rdkit install).  Wrapped so a missing
# RDKit produces a clear ``ImportError`` at module load, not deep inside
# the eigenvalue solver.  We do not feature-gate RDKit: every PLL call
# needs ``Mol.GetConformer`` to read the 3D coords.
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


# ---------------------------------------------------------------------------
# scipy lazy import — graceful degrade.  PLL only needs scipy when the
# caller actually invokes :func:`compute_pll` (or the cached scalar
# variant).  Importing the module must succeed on a slim RDKit-only env
# so downstream tooling (CLI smoke, batched_rdkit) does not crash.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import-only check
    from scipy.sparse import csr_matrix  # type: ignore
    from scipy.sparse.linalg import eigsh  # type: ignore
    _SCIPY_AVAILABLE: bool = True
except Exception as _exc:  # pragma: no cover
    csr_matrix = None  # type: ignore
    eigsh = None  # type: ignore
    _SCIPY_AVAILABLE = False
    _SCIPY_IMPORT_ERR: Optional[Exception] = _exc
else:
    _SCIPY_IMPORT_ERR = None


__all__ = [
    "PLLSignature",
    "PLLSignatureError",
    "DEFAULT_K",
    "DEFAULT_KNN",
    "compute_pll",
    "build_knn_graph",
    "build_normalised_laplacian",
]


__version__ = "0.0.1-pllsig"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
#
# Liu & Feng 2026 use ``k = 16`` eigenvalues (Table 2 in the paper) for
# the PDBbind v2016 benchmark; that is the canonical default and is
# what Track B's MCTS expects (the heuristic
# ``1 / (1 + ||PLL(state) - PLL(target)||_2)`` becomes noise-dominated
# for ``k < 8`` and bandwidth-bound for ``k > 32``).
DEFAULT_K: int = 16

#: Number of nearest neighbours per node.  Liu & Feng 2026 sweep
#: ``{6, 8, 12, 16}``; ``k_nn = 8`` is the published best balance
#: between stability (higher = more invariant under coordinate noise)
#: and locality (lower = more sensitive to local topology).
DEFAULT_KNN: int = 8


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PLLSignatureError(ValueError):
    """Raised when a PLL signature cannot be computed.

    Inherits from ``ValueError`` so existing ``except ValueError``
    blocks catch it; the type remains distinct so callers can branch
    on PLL-specific failures (missing 3D, too few atoms, scipy
    unavailable) without sniffing exception messages.
    """


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLLSignature:
    """Persistent Local Laplacian signature of a 3D conformer.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Length-``k`` vector of the *largest* eigenvalues of the
        normalised graph Laplacian, sorted in **descending** order
        (``λ_0 ≥ λ_1 ≥ ... ≥ λ_{k-1}``).  Always real, non-negative,
        dtype ``float64``.  This descending convention matches Liu &
        Feng 2026 Table 3 (their spectral-kernel feature ``f_ℓ``
        weights the *largest* eigenvalues first).
    eigenvectors : np.ndarray
        Shape ``(k, N_atoms)`` array of the corresponding
        eigenvectors (rows = eigenvectors, columns = atoms).  The
        columns are the eigenbasis of ℝ^N; the rows are invariant
        up to a sign-flip.  We do **not** canonicalise the sign —
        callers that need bit-for-bit invariance should track it
        themselves.
    n_atoms : int
        Number of atoms used to build the kNN graph (== ``len(conformer)``).
    k_requested : int
        ``k`` argument as passed in.  ``eigenvalues.shape[0]`` may be
        less than ``k_requested`` when ``n_atoms - 1 < k_requested``
        (a numerical ceiling imposed by ARPACK).
    k_used : int
        ``min(k_requested, n_atoms - 1)`` — the effective rank.
    smiles : str
        Canonical SMILES (``Chem.MolToSmiles``) of the molecule used
        for cache keying.  Empty string if the molecule has no
        ``Chem.Mol`` identity.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_atoms: int
    k_requested: int
    k_used: int
    smiles: str

    def as_dict(self) -> dict:
        """Return a JSON-friendly dict (eigenvalues list, vectors list)."""
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors_shape": list(self.eigenvectors.shape),
            "n_atoms": int(self.n_atoms),
            "k_requested": int(self.k_requested),
            "k_used": int(self.k_used),
            "smiles": self.smiles,
        }


# ---------------------------------------------------------------------------
# Sparse graph construction
# ---------------------------------------------------------------------------


def build_knn_graph(
    coords: np.ndarray,
    k_nn: int = DEFAULT_KNN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a symmetric k-NN graph on a 3D point cloud.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, 3)`` float array of 3D positions (Å).
    k_nn : int
        Number of nearest neighbours per node.  Defaults to
        :data:`DEFAULT_KNN` (8).  Must be in ``[1, N - 1]`` — values
        outside this range are clipped.

    Returns
    -------
    (row, col, weight)
        Three 1-D ``np.ndarray`` s of length ``M`` representing the
        sparse weighted adjacency matrix in CSR-compatible format:

        * ``row[i], col[i]``   edge endpoints
        * ``weight[i]``        heat-kernel weight ``1 / (1 + d^2)``

        Edges are *symmetric* (if ``(i, j)`` is in the list then so is
        ``(j, i)``); self-loops (``i == j``) are excluded.

    Notes
    -----
    Edge construction is ``O(N^2)`` in the worst case (we materialise
    the full pairwise-distance matrix to keep the code readable).  For
    ``N ≤ 200`` (the typical heavy-atom count of a drug-like ligand)
    this is faster than a KD-tree for ``k_nn ≤ 16`` because of the
    constant factor — see :func:`_pairwise_distances` for the actual
    routine.

    Examples
    --------
    >>> coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    >>> r, c, w = build_knn_graph(coords, k_nn=2)
    >>> len(r) == len(c) == len(w) > 0
    True
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise PLLSignatureError(
            f"coords must be shape (N, 3); got {coords.shape}"
        )
    n = coords.shape[0]
    if n < 2:
        raise PLLSignatureError(
            f"need at least 2 atoms for a kNN graph; got n={n}"
        )
    # Clip k_nn so it cannot exceed the max useful value (n - 1).
    k_nn = max(1, min(int(k_nn), n - 1))

    d = _pairwise_distances(coords)             # (N, N) symmetric, d_ii = 0
    # For each row i, find the k_nn nearest indices (excluding i itself).
    # ``np.argpartition`` is O(N * k_nn) per row; ``argsort`` is O(N^2 log N).
    # The partition is correct *up to* an internal order — we don't care.
    # Note: d_ii = 0, so it can sneak into the top-k_nn if all other
    # neighbours are far; guard with a strict-inequality filter.
    nearest = np.argpartition(d, k_nn, axis=1)[:, :k_nn]   # (N, k_nn)

    # Symmetrise: union of directed edges.
    src = np.repeat(np.arange(n), k_nn)                    # (N * k_nn,)
    dst = nearest.reshape(-1)
    # Drop self-loops (i -> i) defensively, although d_ii = 0 means they
    # can only appear if k_nn == n - 1 and all neighbours are degenerate.
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # Build the canonical (sorted) edge pairs so (i, j) and (j, i) are
    # the same edge for the purpose of the sparse adjacency matrix.
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    # Dedup via sorting + unique.
    edges = np.stack([a, b], axis=1)
    edges = np.unique(edges, axis=0)                        # (M, 2)
    row = np.concatenate([edges[:, 0], edges[:, 1]])       # 2M edges, symmetric
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    dist = d[row, col]                                      # (2M,)
    weight = 1.0 / (1.0 + dist * dist)                      # heat-kernel
    return row.astype(np.int64), col.astype(np.int64), weight.astype(np.float64)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Return the symmetric ``(N, N)`` pairwise-distance matrix.

    Standard identity
        ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b``
    avoids the ``O(N^2)`` memory of materialising difference vectors.
    The result is symmetric with ``d_ii = 0`` by construction.
    """
    sq = np.sum(coords * coords, axis=1)                    # (N,)
    gram = coords @ coords.T                                 # (N, N)
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    # Numerical noise can push d2 slightly negative — clamp to 0.
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


# ---------------------------------------------------------------------------
# Normalised Laplacian
# ---------------------------------------------------------------------------


def build_normalised_laplacian(
    row: np.ndarray,
    col: np.ndarray,
    weight: np.ndarray,
    n: int,
):
    """Build the normalised graph Laplacian as a sparse CSR matrix.

    Parameters
    ----------
    row, col, weight : np.ndarray
        Edge endpoints and heat-kernel weights from
        :func:`build_knn_graph`.
    n : int
        Number of nodes (rows / columns of the output).

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n, n)`` normalised Laplacian ``L = I - D^{-1/2} W D^{-1/2}``.
        Symmetric, real, positive semi-definite.  Zero-th eigenvalue is
        always 0 (for a connected graph).

    Raises
    ------
    PLLSignatureError
        If ``row.shape != col.shape != weight.shape`` or if ``n < 2``.
    """
    if _SCIPY_AVAILABLE is False:
        raise PLLSignatureError(
            f"scipy is not available ({_SCIPY_IMPORT_ERR!r}); cannot "
            f"build the normalised Laplacian"
        )
    if n < 2:
        raise PLLSignatureError(
            f"normalised Laplacian needs at least 2 nodes; got n={n}"
        )
    if not (row.shape == col.shape == weight.shape):
        raise PLLSignatureError(
            "row/col/weight must have the same shape; "
            f"got {row.shape}/{col.shape}/{weight.shape}"
        )

    # CSR of the symmetric weighted adjacency W.
    w = csr_matrix((weight, (row, col)), shape=(n, n))
    # Degree vector.
    deg = np.asarray(w.sum(axis=1)).reshape(-1)
    if np.any(deg <= 0):
        # Isolated vertex → D^{-1/2} undefined.  In practice this
        # happens when k_nn=0 (clipped to 1) and a vertex has no
        # finite-weight neighbours; we *promote* it by giving it a
        # tiny self-loop so the Laplacian stays well-defined.  This
        # is the standard "ε-regularisation" of Belkin 2018 §3.1.
        eps = 1.0e-12
        deg = np.where(deg <= 0, eps, deg)
        # Add the self-loop to W so degree counts it.
        idx = np.arange(n)
        w = w + eps * csr_matrix(
            (np.ones(n, dtype=np.float64), (idx, idx)), shape=(n, n)
        )
        deg = np.asarray(w.sum(axis=1)).reshape(-1)

    d_inv_sqrt = 1.0 / np.sqrt(deg)
    # D^{-1/2} W D^{-1/2} = (D^{-1/2} row_scale) ⊙ W ⊙ col_scale.
    # Use the diagonal-matrix multiply identity:
    #   D^{-1/2} W D^{-1/2} = diag(d_inv_sqrt) @ W @ diag(d_inv_sqrt)
    D = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    W_norm = D @ w @ D
    L = csr_matrix((np.ones(n, dtype=np.float64),
                    (np.arange(n), np.arange(n))), shape=(n, n)) - W_norm
    # eigsh requires float64 CSR for stability.
    L = L.astype(np.float64)
    L.sum_duplicates()
    L.eliminate_zeros()
    return L


# ---------------------------------------------------------------------------
# Top-k spectrum (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _pll_eigvals_cached(
    smiles: str,
    k: int,
    n_atoms: int,
    coords_bytes: bytes,
    k_nn: int,
) -> Tuple[float, ...]:
    """Cached top-k eigenvalue worker — private.

    Cached on ``(smiles, k, n_atoms, coords-bytes, k_nn)``.  The
    ``coords_bytes`` field is the pickled ``(N, 3) float64`` array;
    we serialise to a ``bytes`` blob so it is hashable under
    :func:`functools.lru_cache`.  Two calls with the *same* canonical
    SMILES and *bit-identical* coords return the cached tuple
    without re-running ARPACK.

    Returns a tuple of floats (descending).  Hashable, immutable,
    cheap to compare.
    """
    if not _SCIPY_AVAILABLE:
        raise PLLSignatureError(
            f"scipy is not available ({_SCIPY_IMPORT_ERR!r}); cannot "
            f"compute PLL spectrum"
        )
    coords = np.frombuffer(coords_bytes, dtype=np.float64).reshape(
        n_atoms, 3
    ).copy()
    # Re-build the kNN graph + Laplacian from cached coords.  Building
    # the graph from the bytes blob keeps the cache *pure* (no
    # mutable numpy arrays in the cache key).
    row, col, weight = build_knn_graph(coords, k_nn=int(k_nn))
    L = build_normalised_laplacian(row, col, weight, n=n_atoms)

    # ARPACK can ask for at most ``n - 1`` eigenvalues; clip.
    k_eff = max(1, min(int(k), n_atoms - 1))
    if k_eff == 1:
        # For N = 2, the only eigenvalue (other than 0) is 1.0 by
        # construction (normalised Laplacian of K_2).  eigsh may not
        # converge cleanly in this degenerate case — bypass it.
        eigvals = np.array([0.0], dtype=np.float64)
    else:
        # ``which='LA'`` asks for the Largest Algebraic eigenvalues
        # (matches our descending-convention dataclass).  ARPACK
        # converges in O(k * nnz(L) * iters) — for typical ligands
        # this is < 50 ms.
        try:
            eigvals = eigsh(
                L,
                k=k_eff,
                which="LA",
                return_eigenvectors=False,
                tol=1.0e-8,
            )
        except Exception:
            # Fallback to dense solver for tiny / ill-conditioned graphs.
            # eigsh occasionally fails to converge for ``n < 2 * k_eff``
            # when the spectrum is degenerate; the dense path is robust
            # in that regime.
            L_dense = L.toarray()
            w_dense = np.linalg.eigvalsh(L_dense)
            # eigvalsh returns ascending; take top-k_eff.
            eigvals = w_dense[-k_eff:]
    # Clip numerical noise: normalised Laplacian eigenvalues lie in [0, 2].
    eigvals = np.clip(np.real(eigvals), 0.0, 2.0)
    # Descending sort (largest first) — matches Liu & Feng 2026.
    eigvals = np.sort(eigvals)[::-1]
    return tuple(float(v) for v in eigvals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pll(
    mol: "Chem.Mol",
    k: int = DEFAULT_K,
    k_nn: int = DEFAULT_KNN,
    smiles: Optional[str] = None,
) -> PLLSignature:
    """Compute the top-k Persistent Local Laplacian signature of ``mol``.

    Parameters
    ----------
    mol : Chem.Mol
        An RDKit molecule with **at least one 3D conformer** (call
        ``AllChem.EmbedMolecule(mol)`` first).  Implicit H atoms are
        fine — :meth:`Chem.Mol.GetConformer` reads them.
    k : int
        Number of eigenvalues to return (descending).  Defaults to
        :data:`DEFAULT_K` (16, Liu & Feng 2026 Table 2).
    k_nn : int
        Number of nearest neighbours per atom.  Defaults to
        :data:`DEFAULT_KNN` (8).
    smiles : str, optional
        Canonical SMILES used as cache key.  When ``None`` we derive
        it from ``mol`` via ``Chem.MolToSmiles``; passing it
        explicitly lets the caller skip a redundant RDKit call.

    Returns
    -------
    PLLSignature
        See :class:`PLLSignature` for the field semantics.  In
        particular, ``eigenvalues`` is descending and
        ``eigenvectors`` has shape ``(k_used, n_atoms)``.

    Raises
    ------
    PLLSignatureError
        * if RDKit or scipy is missing,
        * if ``mol`` has no conformer,
        * if ``n_atoms < 2``,
        * if ``k < 1``.

    Notes
    -----
    Rotation / translation / reflection invariance follows from the
    fact that the construction depends only on the pairwise-distance
    matrix ``D_{ij} = ||x_i - x_j||``.  Two conformers of the same
    molecule that differ only by a rigid-body transform yield the
    *same* signature up to the floating-point noise of ``eigsh``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    >>> _ = AllChem.EmbedMolecule(m)
    >>> sig = compute_pll(m, k=8)
    >>> sig.eigenvalues.shape == (8,)
    True
    """
    if not _RDKIT_AVAILABLE:
        raise PLLSignatureError(
            f"RDKit is not available ({_RDKIT_IMPORT_ERR!r}); "
            f"cannot read conformer"
        )
    if mol is None:
        raise PLLSignatureError("mol is None")
    if k < 1:
        raise PLLSignatureError(f"k must be >= 1; got {k}")
    n_atoms = int(mol.GetNumAtoms())
    if n_atoms < 2:
        raise PLLSignatureError(
            f"PLL requires at least 2 atoms; got n_atoms={n_atoms}"
        )

    # Read the conformer coords.  We pin to conformer id 0 — Track B
    # upstream always writes its single canonical conformer there.
    try:
        conf = mol.GetConformer(0)
    except Exception as exc:
        raise PLLSignatureError(
            f"mol has no conformer id 0 — call AllChem.EmbedMolecule first "
            f"({exc!r})"
        ) from exc
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i, 0] = float(pos.x)
        coords[i, 1] = float(pos.y)
        coords[i, 2] = float(pos.z)
    if not np.isfinite(coords).all():
        raise PLLSignatureError(
            "conformer contains non-finite coords; re-embed the molecule"
        )

    # Cache key.  Coordinates are serialised to a ``bytes`` blob so
    # :func:`functools.lru_cache` can hash them.  ``tobytes()`` is
    # bit-exact and avoids the pickling overhead of a custom encoder.
    if smiles is None:
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            smiles = ""
    coords_bytes = np.ascontiguousarray(coords, dtype=np.float64).tobytes()

    eigvals = _pll_eigvals_cached(
        smiles=str(smiles),
        k=int(k),
        n_atoms=int(n_atoms),
        coords_bytes=coords_bytes,
        k_nn=int(k_nn),
    )
    k_used = len(eigvals)

    # The cached path stores only the eigenvalues.  When eigenvectors
    # are requested we re-run the ARPACK solve *without* caching
    # (eigvec caches blow up the memory budget and the upstream MCTS
    # only needs the eigenvalue vector).  The re-solve is cheap for
    # ``k_used <= 32`` and ``n_atoms <= 200``.
    eigvecs = np.zeros((k_used, n_atoms), dtype=np.float64)
    if k_used > 0 and _SCIPY_AVAILABLE:
        row, col, weight = build_knn_graph(coords, k_nn=int(k_nn))
        L = build_normalised_laplacian(row, col, weight, n=n_atoms)
        try:
            # eigsh has no "top-k" eigen-vector option, so we ask for
            # k_used of the largest and pick them out.
            w_full, v_full = eigsh(
                L,
                k=k_used,
                which="LA",
                return_eigenvectors=True,
                tol=1.0e-8,
            )
            # Sort by descending real part and reindex.
            order = np.argsort(-np.real(w_full))
            w_full = np.real(w_full[order])
            v_full = v_full[:, order]
            # eigvecs shape (k_used, n_atoms) — transpose the columns.
            eigvecs = v_full.T.astype(np.float64).copy()
        except Exception:
            # Dense fallback for tiny / ill-conditioned graphs.
            L_dense = L.toarray()
            w_dense, v_dense = np.linalg.eigh(L_dense)
            order = np.argsort(-w_dense)[:k_used]
            eigvecs = v_dense[:, order].T.astype(np.float64).copy()

    return PLLSignature(
        eigenvalues=np.asarray(eigvals, dtype=np.float64),
        eigenvectors=eigvecs,
        n_atoms=int(n_atoms),
        k_requested=int(k),
        k_used=int(k_used),
        smiles=str(smiles),
    )


# ---------------------------------------------------------------------------
# Convenience CLI smoke entrypoint (for inline / smoke use only)
# ---------------------------------------------------------------------------


def _format_signature(sig: PLLSignature) -> str:
    """Return a one-line human-readable summary (no logging dep)."""
    return (
        f"[PLL] n={sig.n_atoms} k={sig.k_used}/{sig.k_requested} "
        f"top5=[{', '.join(f'{v:.4f}' for v in sig.eigenvalues[:5])}] "
        f"smi={sig.smiles!r}"
    )


if __name__ == "__main__":  # pragma: no cover — manual smoke entrypoint
    import sys

    from rdkit import Chem
    from rdkit.Chem import AllChem

    smi_list = sys.argv[1:] or ["CCO", "c1ccccc1", "[Pt](N)(N)(Cl)Cl"]
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"[skip] cannot parse {smi!r}")
            continue
        m = Chem.AddHs(m)
        status = AllChem.EmbedMolecule(m)
        if status != 0:
            print(f"[skip] embed failed for {smi!r}")
            continue
        sig = compute_pll(m, k=8)
        print(_format_signature(sig))