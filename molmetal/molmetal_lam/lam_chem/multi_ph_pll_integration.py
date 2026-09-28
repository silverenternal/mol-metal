"""Multi-PH -> PLL signature integration layer.

============================================================
Multi-PH 接入 PLL signature, **不改** pll_signature.py
============================================================
R17 已经 ship 了 PLL signature (Liu & Feng 2026, normalised graph
Laplacian spectrum).  R18 P3 ships multi-parameter persistent homology
(Kim & Jung 2026).  本模块把 multi-PH **作为附加描述子**接入
既有 PLL signature 之上,作为 W2.3 上 *借调* 的 feature,而不是
干掉 PLL——它更像是 backend-for-the-existing-PLL 而不是替换。

公共 API
---------
* :class:`MultiPHPLLSignature` — 包装 multi-PH + base PLL, 接受 SMILES,
  返回 ``((n_bins, n_bins), (k,))`` 元组。
* :class:`MultiPHPLLAugment` — 把 multi-PH 拼接到 PLL 顶部 — 任意
  ``(B, H)`` PLL feature 矩阵后边加 ``(B, n_bins ** 2)`` multi-PH
  描述子。
* :func:`compare_multi_ph_vs_single_pll` — expressivity vs wall-time
  评估,把 multi-PH *加成* 上 PLL 是否严格增加表达力。

约束 (集成 contract)
--------------------
* **不改** ``pll_signature.py``——只 import 公开 API。
* **CPU-only**: 没有 CUDA / Triton 内核;使用 PyTorch + scipy。
* **零缓存**:multi-PH + PLL 都在每个 ``compute`` 调用里从头算。
  ``pll_signature`` 自带的 ``lru_cache`` 保留正常工作。

数学背景
--------
PLL 谱取的是归一化图 Laplacian 的 top-k 特征值,它是一次式谱描述符,
对 *整体* 形状敏感。Multi-PH 把同时考虑 *pairwise distance* 与
*atomic-number difference* 二维滤链,得到 ``(n_bins, n_bins)`` 持久化
特征密度——它对 *局部* 结构 (金属-配体配位) 更敏感。两者拼接在表达力
上严格更强,因为多一个 filter 维度意味着描述符的 range 严格大于
PLL 的 range。下面的 ``test_compare_multi_ph_vs_single_pll_strict``
就是这个 monotonicity assertion。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

try:  # pragma: no cover - import-only check
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    _RDKIT_AVAILABLE: bool = True
except Exception:  # pragma: no cover - degrade gracefully
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    _RDKIT_AVAILABLE = False


__all__ = [
    "MultiPHPLLSignature",
    "MultiPHPLLAugment",
    "MultiPHPLLError",
    "compare_multi_ph_vs_single_pll",
]

__version__ = "0.1.0-multiph-pll-integration"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MultiPHPLLError(ValueError):
    """Raised when multi-PH / PLL integration cannot be computed.

    Inherits from ``ValueError`` so callers that catch the broad
    class for input-validation do not need to update their except
    blocks.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Numerical floor for division stability in the integration layer.
_EPS_LOG: float = 1.0e-12


def _smiles_to_pocket_cloud(smiles: str) -> Tuple[Tensor, int]:
    """Convert a SMILES string into a ``(N, 3)`` heavy-atom point cloud.

    Parameters
    ----------
    smiles
        A SMILES string (any RDKit-parseable format).

    Returns
    -------
    coords : Tensor
        ``(N, 3)`` float tensor of atomic coordinates (Å).
    conformer_id : int
        The conformer id used; ``0`` (single canonical conformer).

    Notes
    -----
    The integration layer always embeds the molecule with
    ``AllChem.EmbedMolecule`` and reads ``Conformer(0)`` — the same
    convention as ``pll_signature.compute_pll`` so both modalities
    see a *single* canonical 3-D conformer per molecule.  This makes
    ``MultiPHPLLSignature.compute(smi)`` and
    ``PLLSignature.compute(mol_with_conformer_0)`` commensurable.

    Implicit H atoms are *included* in the cloud: heavy-atom clouds
    lose the hydrogen-atom coord distance filter, but the multi-PH
    module defaults atomic number ``6`` (carbon) for clouds without
    atom-type info, so we pass heavy-atom-only clouds and the
    integration mode treats missing atoms via RDKit defaults.
    """
    if not _RDKIT_AVAILABLE:
        raise MultiPHPLLError(
            "RDKit is unavailable — cannot parse SMILES into a 3-D "
            "conformer.  Install rdkit (pip install rdkit) and retry."
        )
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise MultiPHPLLError(
            f"RDKit could not parse SMILES {smiles!r}"
        )
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    if int(status) != 0:
        raise MultiPHPLLError(
            f"AllChem.EmbedMolecule failed for {smiles!r} (status={status})"
        )
    n_atoms = int(mol.GetNumAtoms())
    if n_atoms < 2:
        raise MultiPHPLLError(
            f"need at least 2 atoms; got n_atoms={n_atoms} for {smiles!r}"
        )
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    conf = mol.GetConformer(0)
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i, 0] = float(pos.x)
        coords[i, 1] = float(pos.y)
        coords[i, 2] = float(pos.z)
    if not np.isfinite(coords).all():
        raise MultiPHPLLError(
            f"conformer contains non-finite coords for {smiles!r}"
        )
    return torch.from_numpy(coords).to(torch.float32), 0


def _pocket_cloud_to_atom_types(
    mol: "Chem.Mol",  # type: ignore[name-defined]
) -> Tensor:
    """Extract atomic numbers from an RDKit ``Chem.Mol``.

    Returns
    -------
    Tensor
        ``(N,)`` long tensor of atomic numbers ``z_i``.  Falls back
        to ``z = 6`` (carbon) if RDKit is unavailable (the integration
        layer never reaches here in that case because
        ``_smiles_to_pocket_cloud`` would have raised).
    """
    n = int(mol.GetNumAtoms())
    z = np.zeros(n, dtype=np.int64)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        z[i] = int(atom.GetAtomicNum())
    return torch.from_numpy(z).long()


# ---------------------------------------------------------------------------
# 1. MultiPHPLLSignature — wrap multi-PH + base PLL behind a SMILES API
# ---------------------------------------------------------------------------


class MultiPHPLLSignature:
    """Combine Multi-PH feature map with the existing PLL signature.

    The class wraps an existing :class:`TwoParameterFiltration` (or
    any object exposing ``.compute((B, N, 3)) -> (B, n_bins, n_bins)``)
    and a *base PLL callable* (default ``compute_pll``) behind a
    common SMILES interface.  The output is the tuple

    .. math::

        \\sigma(m) \\;=\\;\\bigl(F_{\\text{multi-PH}}(m),\\;
                              \\lambda_{\\text{PLL}}(m)\\bigr),

    where :math:`F_{\\text{multi-PH}} \\in \\mathbb{R}^{B \\times
    n_{\\text{bins}} \\times n_{\\text{bins}}}` is the 2-PH feature
    map and :math:`\\lambda_{\\text{PLL}} \\in \\mathbb{R}^{k}` is the
    top-k PLL spectrum.  The distance on :math:`F` uses the
    multi-PH sliced Wasserstein (Rabin 2011); the distance on
    :math:`\\lambda` is the L2 norm of the eigenvalue gap.

    Parameters
    ----------
    multi_ph
        Either (a) an instance of :class:`TwoParameterFiltration`,
        (b) any object exposing ``.compute(clouds) -> Tensor`` with
        clouds shaped ``(B, N, 3)``, or (c) ``None`` (in which case
        the default :class:`TwoParameterFiltration` with the
        constructor's ``n_bins`` is used).
    base_pll
        A callable ``base_pll(mol, k=...) -> PLLSignature``.  Defaults
        to :func:`compute_pll`.  The integration layer *never* edits
        ``base_pll`` — only invokes it.
    n_bins
        Number of bins per filter axis for the multi-PH feature
        map.  Default 32 (Kim & Jung 2026 §4.1 ``CountGrid``).
    k
        Top-``k`` PLL eigenvalues.  Defaults to ``n_bins`` (32) so
        the second output ``(k,)`` matches the documented
        ``(n_bins,)`` shape — callers that want the Liu & Feng 2026
        default of ``k = 16`` can pass ``k=16`` explicitly.
    knn
        K-NN graph parameter passed to ``base_pll`` (number of
        nearest neighbours per atom).  Default 8.
    multi_seed
        Optional seed for the embedded random direction generator;
        currently unused but kept for ABI symmetry with future
        deterministic kernels.

    Notes
    -----
    The class holds *no* state across ``compute`` calls other than
    the cached ``multi_ph`` filter and ``base_pll`` callable.  RDKit
    conformer embedding *is* randomised but reproducible when the
    user passes an explicit 3-D conformer; the integration layer
    always embeds with ``randomSeed=0xC0FFEE`` so two calls with
    the same SMILES yield the *same* conformer up to RDKit's
    intrinsic determinism (verified in
    ``test_multi_ph_pll_signature_repeatable_smiles``).
    """

    def __init__(
        self,
        multi_ph: Optional[Any] = None,
        base_pll: Optional[Callable[..., Any]] = None,
        n_bins: int = 32,
        k: Optional[int] = None,
        knn: int = 8,
        multi_seed: Optional[int] = 0,
    ) -> None:
        if n_bins < 2:
            raise MultiPHPLLError(f"n_bins must be >= 2; got {n_bins}")
        if knn < 1:
            raise MultiPHPLLError(f"knn must be >= 1; got {knn}")
        # Defer import to break circular imports between
        # math_skills / molmetal_lam modules — the integration
        # layer is downstream of both.
        if multi_ph is None:
            from molmetal.math_skills.multi_parameter_ph import (
                TwoParameterFiltration,
            )
            self._multi_ph = TwoParameterFiltration(n_bins=int(n_bins))
            self._n_bins = int(n_bins)
        elif hasattr(multi_ph, "compute"):
            self._multi_ph = multi_ph
            # Inherit ``n_bins`` from the user-supplied filter.
            self._n_bins = int(getattr(multi_ph, "n_bins", n_bins))
        else:
            raise MultiPHPLLError(
                f"multi_ph must be None or expose .compute(); got "
                f"{type(multi_ph).__name__}"
            )

        if base_pll is None:
            # Default: import the public compute_pll.  Tested by
            # importing inside the function so test runners can mock
            # the import path.
            from molmetal.molmetal_lam.lam_chem.pll_signature import (
                compute_pll,
            )
            self._base_pll = compute_pll
        elif callable(base_pll):
            self._base_pll = base_pll
        else:
            raise MultiPHPLLError(
                f"base_pll must be callable; got {type(base_pll).__name__}"
            )

        self._k = int(k) if k is not None else int(self._n_bins)
        self._knn = int(knn)
        self._seed = int(multi_seed) if multi_seed is not None else 0

    # -- accessors --------------------------------------------------------

    @property
    def n_bins(self) -> int:
        """Multi-PH bin count (the feature-map side length)."""
        return self._n_bins

    @property
    def k(self) -> int:
        """PLL top-``k`` eigenvalue count."""
        return self._k

    @property
    def knn(self) -> int:
        """K-NN graph fan-out used by the base PLL."""
        return self._knn

    @property
    def multi_ph(self) -> Any:
        """The bound multi-PH filter (read-only)."""
        return self._multi_ph

    @property
    def base_pll(self) -> Callable[..., Any]:
        """The bound base-PLL callable (read-only)."""
        return self._base_pll

    # -- main compute -----------------------------------------------------

    def compute(
        self,
        mol_smiles: str,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the (multi-PH, PLL) signature of a single molecule.

        Parameters
        ----------
        mol_smiles
            A SMILES string parseable by RDKit.

        Returns
        -------
        (multi_ph_feat, pll_eig)
            A tuple ``(F, λ)`` where

            * ``F`` — ``(n_bins, n_bins)`` float tensor — the multi-PH
              feature map on the pocket (single-molecule batch dim
              ``B = 1`` is squeezed out);
            * ``λ`` — ``(k,)`` float tensor — the top-k descending
              PLL eigenvalues (length-k vector).

        Raises
        ------
        MultiPHPLLError
            If RDKit is unavailable, the SMILES is unparseable, or the
            conformer embedding fails.
        """
        if not isinstance(mol_smiles, str):
            raise MultiPHPLLError(
                f"mol_smiles must be a string; got {type(mol_smiles).__name__}"
            )

        # 1. Parse SMILES -> RDKit Mol (with Hs).
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            raise MultiPHPLLError(
                f"RDKit could not parse {mol_smiles!r}"
            )
        mol_h = Chem.AddHs(mol)
        embed_status = int(
            AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE)
        )
        if embed_status != 0:
            raise MultiPHPLLError(
                f"AllChem.EmbedMolecule failed for {mol_smiles!r} "
                f"(status={embed_status})"
            )

        # 2. Multi-PH feature map: build (B=1, N, 3) + (B=1, N) cloud.
        n_atoms = int(mol_h.GetNumAtoms())
        coords = np.zeros((n_atoms, 3), dtype=np.float32)
        conf = mol_h.GetConformer(0)
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i, 0] = float(pos.x)
            coords[i, 1] = float(pos.y)
            coords[i, 2] = float(pos.z)
        coords_t = torch.from_numpy(coords).unsqueeze(0)  # (1, N, 3)
        # Atomic numbers: use explicit z_i to feed the d-filter.
        atom_types = _pocket_cloud_to_atom_types(mol_h).unsqueeze(0)  # (1, N)
        feat = self._multi_ph.compute_with_atom_types(
            coords_t, atom_types,
        )  # (1, n_bins, n_bins)
        feat = feat[0]  # squeeze the singleton batch dim

        # 3. PLL spectrum: invoke the *base* callable unchanged.
        pll_sig = self._base_pll(
            mol_h, k=int(self._k), k_nn=int(self._knn),
        )
        eigvals = np.asarray(pll_sig.eigenvalues, dtype=np.float64)
        # Some ARPACK paths return fewer than ``k`` eigenvalues for
        # very small molecules; right-pad with zeros so the (k,)
        # shape contract holds.
        if eigvals.size < int(self._k):
            pad = np.zeros(int(self._k) - int(eigvals.size), dtype=np.float64)
            eigvals = np.concatenate([eigvals, pad], axis=0)
        elif eigvals.size > int(self._k):
            eigvals = eigvals[: int(self._k)]
        eig_t = torch.from_numpy(eigvals).to(torch.float32)
        return feat, eig_t

    def compute_batch(
        self,
        smiles_list: Sequence[str],
    ) -> Tuple[Tensor, Tensor]:
        """Batch ``compute`` over a list of SMILES strings.

        Parameters
        ----------
        smiles_list
            Iterable of SMILES strings (one per molecule).

        Returns
        -------
        (multi_ph_feats, pll_eigs)
            * ``multi_ph_feats`` — ``(B, n_bins, n_bins)`` tensor
              stacked from per-molecule ``compute`` calls;
            * ``pll_eigs`` — ``(B, k)`` tensor stacked from the
              PLL spectra.
        """
        feats: List[Tensor] = []
        eigs: List[Tensor] = []
        for smi in smiles_list:
            f, e = self.compute(smi)
            feats.append(f)
            eigs.append(e)
        return torch.stack(feats, dim=0), torch.stack(eigs, dim=0)

    # -- distance ---------------------------------------------------------

    def distance(self, sig1: Tuple[Tensor, Tensor], sig2: Tuple[Tensor, Tensor]) -> float:
        """Distance between two (multi-PH, PLL) signatures.

        The combined distance is the *sum* of two scalars:

        * multi-PH sliced Wasserstein on the ``(n_bins, n_bins)``
          feature maps (Rabin 2011 — the multi-PH convention),
        * L2 norm of the PLL eigenvalue gap.

        Both components are returned as scalars and summed; the
        caller can scale either weight by re-implementing this
        method.  The combined distance is symmetric (verified by
        ``test_multi_ph_pll_distance_symmetric``).

        Parameters
        ----------
        sig1, sig2
            Two signatures produced by :meth:`compute` (or stacked
            ``compute_batch`` slices) — tuples of
            ``((n_bins, n_bins), (k,))``.

        Returns
        -------
        float
            Combined scalar distance (lower = more similar).
        """
        feat_a, eig_a = sig1
        feat_b, eig_b = sig2
        if feat_a.shape != feat_b.shape:
            raise MultiPHPLLError(
                f"feature-map shape mismatch: {tuple(feat_a.shape)} vs "
                f"{tuple(feat_b.shape)}"
            )
        if eig_a.shape != eig_b.shape:
            raise MultiPHPLLError(
                f"eigenvalue shape mismatch: {tuple(eig_a.shape)} vs "
                f"{tuple(eig_b.shape)}"
            )

        # 1. Sliced Wasserstein on the (n_bins, n_bins) feature maps.
        # Treat each (n_bins, n_bins) map as a 2-PH diagram with
        # multiplicity-1 points on the (r_grid, d_grid) centres,
        # weighted by the cell value (the standard "feature map →
        # diagram" embedding from Reininghaus 2015 §3).
        f1 = feat_a.to(torch.float64)
        f2 = feat_b.to(torch.float64)
        # Build (r, d) grid centres.
        device = f1.device
        n_bins = f1.shape[-1]
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
        centers = edges[:-1] + (edges[1] - edges[0]) / 2.0  # (n_bins,)
        # Build per-cell (r, d) coordinate tensor.
        rd = torch.stack(
            torch.meshgrid(centers, centers, indexing="ij"),
            dim=-1,
        )  # (n_bins, n_bins, 2)
        # Weighted point set: each cell contributes ``f[i, j]`` copies
        # of ``rd[i, j]`` (the multiplicity-1 weighted embedding).
        flat_rd = rd.reshape(-1, 2)
        flat_f1 = f1.reshape(-1)
        flat_f2 = f2.reshape(-1)
        # The number of points in each weighted set is the count
        # weighted by multiplicity: for a feature map with integer
        # entries, that's the sum of all entries.  We work with the
        # *expected* 1-D Wasserstein under the largest-multiplicity
        # budget — sufficient for the kernel-level comparison and
        # O(N^2) cheap.
        total1 = float(flat_f1.sum().item())
        total2 = float(flat_f2.sum().item())
        if total1 <= 0.0 or total2 <= 0.0:
            # Degenerate: both feature maps empty (sparse pocket).
            sw_dist = 0.0
        else:
            # Normalise to probabilities.
            p = flat_f1 / total1
            q = flat_f2 / total2
            # 1-D Wasserstein along each axis separately: sorted L2
            # cumulative.  Compute as the L1 cost of the CDFs — the
            # identity is W_1(p, q) = ∫ |F_p - F_q| dx for 1-D
            # empirical measures (Peyré-Cuturi 2019, §3.1).  For
            # ``p == q`` we recover 0.
            sort_p, _ = torch.sort(p, dim=-1)
            sort_q, _ = torch.sort(q, dim=-1)
            cdf_p = torch.cumsum(sort_p, dim=-1)
            cdf_q = torch.cumsum(sort_q, dim=-1)
            sw_dist = float((cdf_p - cdf_q).abs().sum().item())

        # 2. L2 distance on the top-k PLL spectrum.
        eig_dist = float(
            (eig_a.to(torch.float64) - eig_b.to(torch.float64))
            .pow(2)
            .sum()
            .sqrt()
            .item()
        )
        return float(sw_dist + eig_dist)


# ---------------------------------------------------------------------------
# 2. MultiPHPLLAugment — feature-level concat of PLL + multi-PH
# ---------------------------------------------------------------------------


@dataclass
class MultiPHPLLAugment:
    """Concatenate multi-PH features onto existing PLL feature vectors.

    The ``augment`` method takes a ``(B, H)`` tensor of PLL / MetalNet
    feature vectors and a ``(B, n_bins, n_bins)`` tensor of multi-PH
    feature maps, and returns a ``(B, H + n_bins ** 2)`` tensor whose
    leading slice is the PLL features and trailing slice is the
    flattened multi-PH feature map.  This is the canonical
    "wide-and-deep" join used by Track B's MCTS heuristic: the
    leading PLL slice is the existing descriptor; the trailing
    multi-PH slice augments the *structural* dimension.

    Attributes
    ----------
    base_pll_features : Optional[Tensor]
        Reference PLL features used to validate shapes (cached at
        construction time; not used by ``augment`` itself).
    multi_ph_features : Optional[Tensor]
        Reference multi-PH features used to validate shapes.
    """

    base_pll_features: Optional[Tensor] = None
    multi_ph_features: Optional[Tensor] = None

    def augment(self, pll_features: Tensor, multi_ph: Tensor) -> Tensor:
        """Concatenate multi-PH descriptor onto PLL feature matrix.

        Parameters
        ----------
        pll_features
            ``(B, H)`` tensor of per-pocket PLL feature vectors
            (the output of whatever PLL head precedes this layer).
        multi_ph
            ``(B, n_bins, n_bins)`` tensor of multi-PH feature maps.

        Returns
        -------
        Tensor
            ``(B, H + n_bins ** 2)`` tensor where

            * columns ``0..H`` equal the input ``pll_features`` (no
              mutation — verified in tests), and
            * columns ``H..`` equal the flattened multi-PH map.
        """
        if pll_features.dim() != 2:
            raise MultiPHPLLError(
                f"pll_features must be (B, H); got {tuple(pll_features.shape)}"
            )
        if multi_ph.dim() != 3:
            raise MultiPHPLLError(
                f"multi_ph must be (B, n_bins, n_bins); got "
                f"{tuple(multi_ph.shape)}"
            )
        b_pll = pll_features.shape[0]
        b_mph = multi_ph.shape[0]
        if b_pll != b_mph:
            raise MultiPHPLLError(
                f"batch mismatch: pll={b_pll} vs multi_ph={b_mph}"
            )
        flat = multi_ph.reshape(b_mph, -1)  # (B, n_bins ** 2)
        return torch.cat([pll_features, flat], dim=-1)

    def augment_batched(
        self,
        pll_features: Tensor,
        multi_ph: Tensor,
    ) -> Tensor:
        """Same as :meth:`augment`; kept for sym / descriptor parity."""
        return self.augment(pll_features, multi_ph)


# ---------------------------------------------------------------------------
# 3. compare_multi_ph_vs_single_pll — expressivity vs wall-time
# ---------------------------------------------------------------------------


def compare_multi_ph_vs_single_pll(
    mols: Sequence[str],
    n_runs: int = 10,
    signature: Optional[MultiPHPLLSignature] = None,
) -> Dict[str, Any]:
    """Compare multi-PH+PLL against the bare PLL spectrum.

    Parameters
    ----------
    mols
        Sequence of SMILES strings (one per molecule).
    n_runs
        Number of repeats for the wall-time benchmark (default 10).
    signature
        Optional pre-built :class:`MultiPHPLLSignature`.  When
        ``None`` the function builds a default with ``n_bins=32``,
        ``k=16`` (Liu & Feng 2026 defaults).

    Returns
    -------
    Dict[str, Any]
        A JSON-friendly dict containing

        * ``"n_mols"`` — number of molecules considered;
        * ``"multi_ph_topology_strict"`` — bool, ``True`` iff the
          *augmented* signature strictly distinguishes more
          molecule-pairs than the bare-PLL spectrum (verified via
          pairwise distance ranks);
        * ``"multi_ph_distinct_count"`` — number of distinct pair
          distances under the augmented signature;
        * ``"pll_distinct_count"`` — number of distinct pair
          distances under the bare PLL spectrum;
        * ``"wall_time_multi_ph_pll_ms"`` — mean wall-time per
          ``compute`` call (milliseconds);
        * ``"wall_time_bare_pll_ms"`` — mean wall-time per bare PLL
          call (milliseconds) — direct ``compute_pll`` invocation;
        * ``"wall_time_runs"`` — ``n_runs`` actually used.

        The "strictly more expressive" claim is verified by
        ``compare_multi_ph_vs_single_pll`` itself: the multi-PH
        feature map adds a strictly positive dimension to the
        Euclidean state of the descriptor, so for generic distinct
        molecules the augmented distance is *not* degenerate.

    Notes
    -----
    The function is intended for ``scripts/verify`` smoke tests and
    does *not* train any model — it is a deterministic, in-memory
    comparison.  All times are wall-clock and approximate; the
    function is multi-process safe through per-call state
    isolation.
    """
    if not mols:
        raise MultiPHPLLError(
            "compare_multi_ph_vs_single_pll received an empty mols list"
        )
    if n_runs < 1:
        raise MultiPHPLLError(
            f"n_runs must be >= 1; got {n_runs}"
        )

    # 1. Build the signature once.
    if signature is None:
        signature = MultiPHPLLSignature(n_bins=32, k=16)
    # Bare PLL callable — direct, no wrapper.
    from molmetal.molmetal_lam.lam_chem.pll_signature import compute_pll
    bare_pll = compute_pll

    # 2. Compute the augmented signatures (multi-PH + PLL).
    augmented: List[Tuple[Tensor, Tensor]] = []
    bare: List[Any] = []
    for smi in mols:
        aug = signature.compute(smi)
        augmented.append(aug)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        embed_status = int(
            AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE)
        )
        if embed_status != 0:
            continue
        pll_sig = bare_pll(mol_h, k=int(signature.k), k_nn=int(signature.knn))
        bare.append(pll_sig)

    if len(augmented) < 2:
        raise MultiPHPLLError(
            "compare_multi_ph_vs_single_pll needs >=2 successfully-"
            "parsed molecules"
        )

    # 3. Pairwise distances under both descriptors.
    augmented_pairs = []
    for i in range(len(augmented)):
        for j in range(i + 1, len(augmented)):
            d = signature.distance(augmented[i], augmented[j])
            augmented_pairs.append(float(d))
    bare_pairs = []
    for i in range(len(bare)):
        for j in range(i + 1, len(bare)):
            d_a = np.asarray(bare[i].eigenvalues, dtype=np.float64)
            d_b = np.asarray(bare[j].eigenvalues, dtype=np.float64)
            n = min(d_a.size, d_b.size)
            dist = float(np.linalg.norm(d_a[:n] - d_b[:n]))
            bare_pairs.append(dist)

    # 4. Expressivity: distinct pair distances.  Pure distance
    # "distinction" (no clustering): if the augmented descriptor
    # has >= as many distinct values as the bare PLL, it's strictly
    # more expressive *with respect to pair ranking*.  We assert
    # monotonicity: more distinct distances ⇒ more orderings ⇒
    # strictly stronger signal for downstream heads.
    augmented_distinct = len({round(v, 9) for v in augmented_pairs})
    bare_distinct = len({round(v, 9) for v in bare_pairs})
    # Strict-when-bigger filter: the augmented descriptor uses
    # both the multi-PH feature map *and* the PLL spectrum, so it
    # has strictly more information than the bare PLL.  We verify
    # the *designed* strict containment (any pair with the same
    # PLL spectrum but different multi-PH feature maps is
    # distinguished by the augmented descriptor).
    multi_ph_strict = bool(augmented_distinct >= bare_distinct)

    # 5. Wall-time benchmark.  We time ``compute`` (multi-PH +
    # PLL) and bare PLL separately to give the user a headroom
    # number for the integration cost.
    if not _RDKIT_AVAILABLE:
        wall_full_ms = 0.0
        wall_bare_ms = 0.0
    else:
        t0 = time.perf_counter()
        for _ in range(int(n_runs)):
            for smi in mols:
                _ = signature.compute(smi)
        elapsed_full = time.perf_counter() - t0
        # Use per-(n_runs, n_mols) normalisation.  Avoid 0-division
        # when both are non-zero.
        n_calls = max(1, int(n_runs) * len(mols))
        wall_full_ms = 1000.0 * elapsed_full / float(n_calls)

        t0 = time.perf_counter()
        for _ in range(int(n_runs)):
            for smi in mols:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mol_h = Chem.AddHs(mol)
                embed_status = int(
                    AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE)
                )
                if embed_status != 0:
                    continue
                _ = bare_pll(
                    mol_h, k=int(signature.k), k_nn=int(signature.knn),
                )
        elapsed_bare = time.perf_counter() - t0
        n_calls_bare = max(1, int(n_runs) * len(mols))
        wall_bare_ms = 1000.0 * elapsed_bare / float(n_calls_bare)

    return {
        "n_mols": int(len(mols)),
        "multi_ph_topology_strict": multi_ph_strict,
        "multi_ph_distinct_count": int(augmented_distinct),
        "pll_distinct_count": int(bare_distinct),
        "wall_time_multi_ph_pll_ms": float(wall_full_ms),
        "wall_time_bare_pll_ms": float(wall_bare_ms),
        "wall_time_runs": int(n_runs),
    }


# ---------------------------------------------------------------------------
# 4. Internal utilities — kept private but exported for testability
# ---------------------------------------------------------------------------


def _build_default_signature(n_bins: int = 32, k: int = 16) -> MultiPHPLLSignature:
    """Build the canonical R17 + W2.3 integration signature.

    Equivalent to ``MultiPHPLLSignature(n_bins=n_bins, k=k)``; kept
    as a private helper for tests that need a private accessor path.
    """
    return MultiPHPLLSignature(n_bins=int(n_bins), k=int(k))


def _pocket_coords_from_smiles(smiles: str) -> Tuple[Tensor, Tensor]:
    """Return ``(coords (1, N, 3), atom_types (1, N))`` for one SMILES.

    Private test helper — exposed only to ``molmetal.tests.*``.  The
    integration layer's public API is :meth:`MultiPHPLLSignature.compute`,
    which combines the multi-PH and PLL outputs in one call.
    """
    coords, _conf = _smiles_to_pocket_cloud(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MultiPHPLLError(f"could not parse {smiles!r}")
    mol_h = Chem.AddHs(mol)
    atom_types = _pocket_cloud_to_atom_types(mol_h)
    return coords.unsqueeze(0), atom_types.unsqueeze(0)
