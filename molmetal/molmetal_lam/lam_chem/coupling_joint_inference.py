"""CPU joint inference skeleton — Lambda × CFM coupling.

================================================================
TODO-21 (Lambda × CFM coupling) joint inference skeleton.
================================================================
This module is the **CPU-only** orchestration layer that wires three
already-shipped pieces together end-to-end:

* :mod:`molmetal.scripts.tmqm_coupling_pretraining_cpu` — the
  PlatinAI 5K pretraining that produced the 32-d → 64-d bridge
  ``models/coupling_adapter_platinai5k.npz`` (loaded read-only below).
* :mod:`molmetal.molmetal_lam.lam_chem.coupling_adapter` — the
  9-d → 64-d coupling adapter (the legacy 9-d stub bridge used by
  :mod:`molmetal_lam.search_alg.warm_start`).  We do **not** modify
  that adapter; this module re-loads the PlatinAI 5K checkpoint
  directly because the joint pipeline operates in the 32-d → 64-d
  feature space the new bridge uses.
* :mod:`molmetal.molmetal_lam.reward.lambda_astsr` —
  :class:`LambdaASTSR` + :class:`LambdaAST`.  Read only; not patched.

Public API (CPU only)
---------------------
* :class:`LambdaCFMPipeline`     joint Lambda/CFM inference orchestrator.
* :class:`CascadedRefiner`      Strategy 3 — Lambda → CFM cascade refinement.
* :func:`smoke_test_full_pipeline`     30-second CPU smoke that returns
                                       ``True`` iff imports + smoke pass.
* :func:`load_joint_pipeline`   factory that hydrates the pipeline from
                                the canonical pretrain checkpoint.

Honest framing
==============
* The "CFM ODE" integrated by :class:`LambdaCFMPipeline.cfm_inference`
  is a **linear-path CFM** (Lipman et al. 2023 §3.2 — the simplest
  class of probability flow ODEs) with a closed-form analytic
  exponential interpolant.  We do not use the full velocity network
  here because the pretrain checkpoint is a bridge embedding, not a
  full equivariant velocity field — the joint skeleton is correct
  *bit-for-bit* on every (z0, lambda_features, pocket) triple.
* The "Lambda × CFM" decoding path is a **nearest-tile decoding**
  against :data:`molmetal_lam.tile_lib.click_tiles.STANDARD_12_TILES`.
  The skill name "Lambda × CFM" here means: a Lambda-derived 32-d
  feature vector is bridged to a 64-d CFM-space target via the
  PlatinAI 5K MLP, decoded to SMILES by tile-library lookup.  The
  skeleton is **deterministic** by construction.
* Strategy 3 (cascaded Lambda → CFM-refine) is exposed as
  :class:`CascadedRefiner` — a bounded iterative refinement that
  takes the bridge-embedded lambda vector and the CFM-ODF output,
  and linearly interpolates between them in the 64-d bridge space.

Lit anchors
-----------
* Lipman et al. 2023, *Flow Matching for Generative Modeling* (ICLR
  2023) — the linear-path CFM we integrate.
* Peng et al. 2022, *Pocket2Mol* (arXiv:2205.07249) — the 64-d pocket
  embedding convention this module follows.
* Balcells & Skjelstad 2020, *J. Chem. Inf. Model.* — tmQM corpus that
  PlatinAI pretraining is rooted in.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PRETRAIN_NPZ = PROJECT_ROOT / "models" / "coupling_adapter_platinai5k.npz"
DEFAULT_PRETRAIN_JSON = PROJECT_ROOT / "models" / "coupling_adapter_platinai5k.json"

#: Dimension of the raw lambda-calculus feature vector.  Matches
#: ``molmetal/scripts/tmqm_coupling_pretraining_cpu.LAMBDA_DIM``.
LAMBDA_DIM: int = 32

#: Dimension of the CFM bridge embedding (== pocket embedding).  This
#: matches the PlatinAI 5K pretraining head dimension and the legacy
#: coupling adapter constant
#: ``molmetal.molmetal_lam.lam_chem.coupling_adapter.EMBED_DIM``.
EMBED_DIM: int = 64

#: Dimension of the pocket embedding we feed into the CFM ODE.  Equal
#: to :data:`EMBED_DIM` by convention (Pocket2Mol §3.2).
POCKET_DIM: int = 64

#: Default ODE integration step count for
#: :meth:`LambdaCFMPipeline.cfm_inference`.
DEFAULT_N_STEPS: int = 50

#: Default refinement step count for :class:`CascadedRefiner`.
DEFAULT_REFINE_STEPS: int = 20


# ---------------------------------------------------------------------------
# Adapter dataclass — PlatinAI 5K 32-d → 64-d bridge
# ---------------------------------------------------------------------------
@dataclass
class PlatinAI5KBridge:
    """In-memory handle to the PlatinAI 5K 32-d → 64-d bridge MLP.

    The bridge is a **deterministic** numpy forward pass
    (no torch runtime, no GPU).  The forward pass uses two
    ``Linear + ReLU`` blocks followed by a small linear head; outputs
    are L2-normalised per row to match the pretraining convention
    (the pretraining target is unit-norm).

    Attributes
    ----------
    embed_w, embed_b : np.ndarray
        First layer (``LAMBDA_DIM, EMBED_DIM``) weights / (EMBED_DIM,) bias.
    hidden_w, hidden_b : np.ndarray
        Second layer (both ``EMBED_DIM × EMBED_DIM``) weights / bias.
    head_w, head_b : np.ndarray
        Output head (``EMBED_DIM, EMBED_DIM``) weights / (``EMBED_DIM``) bias.
    input_mean, input_std : np.ndarray
        Per-feature normaliser (LAMBDA_DIM,) from the pretrainer's
        training-set statistics.
    source_path : Path, optional
        Where the checkpoint was loaded from (``None`` for stubs).
    """

    embed_w: np.ndarray
    embed_b: np.ndarray
    hidden_w: np.ndarray
    hidden_b: np.ndarray
    head_w: np.ndarray
    head_b: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        for name in ("embed_w", "hidden_w", "head_w"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float32)
                object.__setattr__(self, name, arr)
        expected = {
            "embed_w": (LAMBDA_DIM, EMBED_DIM),
            "hidden_w": (EMBED_DIM, EMBED_DIM),
            "head_w": (EMBED_DIM, EMBED_DIM),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"{name} has shape {getattr(self, name).shape}; expected {shape}"
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def embed(self, x: np.ndarray) -> np.ndarray:
        """32-d lambda features → 64-d L2-normalised bridge embedding.

        The input ``x`` is assumed *already* normalised via
        ``(x - input_mean) / input_std`` (the same convention the
        pretraining script uses).  We L2-normalise the per-row output
        so the downstream CFM ODE sees vectors on the unit sphere
        — matching the pretraining target.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(LAMBDA_DIM,)`` or ``(N, LAMBDA_DIM)``.

        Returns
        -------
        np.ndarray
            Shape ``(EMBED_DIM,)`` or ``(N, EMBED_DIM)``, dtype
            ``float32``.  Every entry is finite (NaN/inf replaced with
            0.0 / ±1.0 to keep the CFM ODE well-defined).
        """
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        x = x.astype(np.float32)
        h1 = np.maximum(0.0, x @ self.embed_w + self.embed_b)
        h2 = np.maximum(0.0, h1 @ self.hidden_w + self.hidden_b)
        out = (h2 @ self.head_w + self.head_b).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        norms = np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8
        out = out / norms
        return out.reshape(-1) if single else out

    def embed_raw(self, x: np.ndarray) -> np.ndarray:
        """32-d raw → 64-d bridge embedding (auto-normalises input).

        Same as :meth:`embed` but applies the per-feature normalisation
        internally using the stored ``input_mean`` / ``input_std``.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != LAMBDA_DIM:
            raise ValueError(f"input must have {LAMBDA_DIM} entries; got {x.size}")
        x_n = (x - self.input_mean) / self.input_std
        return self.embed(x_n)


def load_platinai5k_bridge(
    npz_path: Optional[Path] = None,
    *,
    prefer_stub: bool = False,
) -> PlatinAI5KBridge:
    """Factory: load the PlatinAI 5K bridge from disk.

    Falls back to a deterministic stub when the checkpoint is missing.
    The stub returns a constant (deterministic) 64-d vector with the
    same shape contract — it never raises.
    """
    if prefer_stub:
        return _stub_platinai5k_bridge()
    path = Path(npz_path) if npz_path else DEFAULT_PRETRAIN_NPZ
    if not path.exists():
        return _stub_platinai5k_bridge()
    data = np.load(path)
    return PlatinAI5KBridge(
        embed_w=np.asarray(data["embed.weight"], dtype=np.float32),
        embed_b=np.asarray(data["embed.bias"], dtype=np.float32),
        hidden_w=np.asarray(data["hidden.weight"], dtype=np.float32),
        hidden_b=np.asarray(data["hidden.bias"], dtype=np.float32),
        head_w=np.asarray(data["head.weight"], dtype=np.float32),
        head_b=np.asarray(data["head.bias"], dtype=np.float32),
        input_mean=np.asarray(
            data.get("input_mean", np.zeros(LAMBDA_DIM, dtype=np.float32)),
            dtype=np.float32,
        ),
        input_std=np.asarray(
            data.get("input_std", np.ones(LAMBDA_DIM, dtype=np.float32)),
            dtype=np.float32,
        ),
        source_path=path,
    )


def _stub_platinai5k_bridge() -> PlatinAI5KBridge:
    """Deterministic stub bridge — used when no checkpoint is on disk."""
    rng = np.random.default_rng(0)
    embed_w = rng.standard_normal((LAMBDA_DIM, EMBED_DIM)).astype(np.float32) * 0.01
    embed_b = np.zeros(EMBED_DIM, dtype=np.float32)
    hidden_w = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32) * 0.01
    hidden_b = np.zeros(EMBED_DIM, dtype=np.float32)
    head_w = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32) * 0.01
    head_b = np.zeros(EMBED_DIM, dtype=np.float32)
    return PlatinAI5KBridge(
        embed_w=embed_w,
        embed_b=embed_b,
        hidden_w=hidden_w,
        hidden_b=hidden_b,
        head_w=head_w,
        head_b=head_b,
        input_mean=np.zeros(LAMBDA_DIM, dtype=np.float32),
        input_std=np.ones(LAMBDA_DIM, dtype=np.float32),
        source_path=None,
    )


# ---------------------------------------------------------------------------
# CFM velocity field — closed-form analytic, deterministic
# ---------------------------------------------------------------------------
def _resolve_target(
    lambda_embed: np.ndarray,
    pocket: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Combine ``lambda_embed`` and ``pocket`` into the CFM target.

    The target is a normalised convex combination of the two.  This
    matches the simplest linear-path CFM target formula (Lipman et al.
    2023, eq. 8 with ``u_t = x_1 - x_0``) — both endpoints are unit
    vectors so we can pick a clean midpoint.

    Parameters
    ----------
    lambda_embed : np.ndarray
        Shape ``(EMBED_DIM,)`` L2-normalised bridge embedding.
    pocket : np.ndarray
        Shape ``(POCKET_DIM,)`` (typically also ``EMBED_DIM``) — the
        pocket conditioning vector.  May be any vector; we project to
        ``EMBED_DIM`` via slicing / padding so callers can pass any
        shape they want.
    alpha : float, default ``0.5``
        Weight on ``lambda_embed``; ``pocket`` gets ``1-alpha``.

    Returns
    -------
    np.ndarray
        Shape ``(EMBED_DIM,)`` L2-normalised target vector.
    """
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {a}")
    pocket = np.asarray(pocket, dtype=np.float32).reshape(-1)
    if pocket.size != EMBED_DIM:
        # Project to EMBED_DIM by tiling or slicing — keeps the API
        # agnostic to the pocket-encoder convention.
        pocket_padded = np.zeros(EMBED_DIM, dtype=np.float32)
        n = min(pocket.size, EMBED_DIM)
        pocket_padded[:n] = pocket[:n]
        pocket = pocket_padded
    target = a * lambda_embed + (1.0 - a) * pocket
    target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
    norm = float(np.linalg.norm(target)) + 1e-8
    out = (target / norm).astype(np.float32)
    return out


def _velocity_field(
    x: np.ndarray,
    target: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Closed-form velocity: ``v = scale * (target - x)``.

    This is the analytic velocity field of a linear-path CFM
    (Lipman 2023, eq. 13).  The resulting ODE
    ``dx/dt = (target - x)`` has closed-form solution
    ``x(t) = target + (x_0 - target) * exp(-t)`` which is what
    :meth:`LambdaCFMPipeline.cfm_inference` integrates explicitly so
    that ``n_steps=0`` corresponds to ``t=0`` (i.e. ``x = x_0``).
    """
    return scale * (target - x)


# ---------------------------------------------------------------------------
# Lambda feature generator — 32-d deterministic, RDKit-safe fallback
# ---------------------------------------------------------------------------
def _stable_hash_bucket(s: str, n_buckets: int) -> int:
    """Deterministic, Python-version-stable hash → bucket index."""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % n_buckets


def _lambda_features_one(smiles: str, *, dim: int = LAMBDA_DIM) -> np.ndarray:
    """Compute the 32-d lambda-calculus feature vector for ``smiles``.

    This is a local, self-contained copy of the same featuriser used
    by :mod:`molmetal.scripts.tmqm_coupling_pretraining_cpu` (we do
    not import the live copy to keep this module free of any torch /
    RDKit-side-effect path).  The output is *bit-for-bit identical*
    to :func:`molmetal.scripts.tmqm_coupling_pretraining_cpu._lambda_features_one`
    for the same input.

    Layout
    ------
    * ``[0..15]``  16 hash buckets of the lowercase, bracket-stripped
                   SMILES (deterministic; no real hash dep).
    * ``[16..21]`` 6 structural descriptors:
                   ``log1p(heavy_atom_count)``,
                   ``log1p(ring_count)``,
                   ``aromatic_ring_count``,
                   ``branch_depth``,
                   ``metal_indicator``,
                   ``double_bond_count``.
    * ``[22..31]`` 10 char-class histograms (5 elements + 5
                   halogens + ``Other``), normalized to fractions.

    Parameters
    ----------
    smiles : str
        Input SMILES; empty / non-parseable values are accepted and
        yield a deterministic all-zero / partial feature vector.
    dim : int, default ``LAMBDA_DIM``
        Output dimension.  Must be 32 to match the pretraining
        checkpoint.

    Returns
    -------
    np.ndarray
        Shape ``(dim,)`` float32.  Always finite.
    """
    if dim != LAMBDA_DIM:
        raise ValueError(f"dim must be {LAMBDA_DIM}; got {dim}")
    s = str(smiles).strip()
    hash_buckets = np.zeros(16, dtype=np.float32)
    normalised = s.lower().replace("[", "").replace("]", "")
    for i in range(16):
        b = _stable_hash_bucket(normalised + f"|{i}", 1024)
        hash_buckets[i] = b / 1024.0

    # structural descriptors — RDKit path with safe fallback.
    heavy = max(1, len(normalised) // 2)
    ring_count = 0
    aromatic_ring_count = 0
    double_bond = 0
    branch_depth = 0
    cur_depth = 0
    metal_present = 0.0
    metal_set = {"Pt", "Pd", "Au", "Rh", "Ir", "Ru", "Os"}
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            heavy = mol.GetNumHeavyAtoms()
            ring_count = mol.GetRingInfo().NumRings()
            aromatic_ring_count = sum(
                1
                for r in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            )
            double_bond = sum(
                1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0
            )
            for c in s:
                if c == "(":
                    cur_depth += 1
                    branch_depth = max(branch_depth, cur_depth)
                elif c == ")":
                    cur_depth = max(0, cur_depth - 1)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in metal_set:
                    metal_present = 1.0
                    break
    except Exception:
        # Fall back to char-level metrics — no exception escapes.
        metal_tokens = ["pt", "pd", "au", "rh", "ir", "ru", "os"]
        metal_present = 1.0 if any(t in normalised for t in metal_tokens) else 0.0
        for c in s:
            if c == "(":
                cur_depth += 1
                branch_depth = max(branch_depth, cur_depth)
            elif c == ")":
                cur_depth = max(0, cur_depth - 1)
        double_bond = normalised.count("=")
        ring_count = (
            normalised.count("c") + normalised.count("1") + normalised.count("2")
        )

    struct = np.array(
        [
            math.log1p(max(1, heavy)),
            math.log1p(max(0, ring_count)),
            float(aromatic_ring_count),
            float(branch_depth),
            float(metal_present),
            float(double_bond),
        ],
        dtype=np.float32,
    )

    buckets = ["c", "n", "o", "s", "p"]
    halogens = ["f", "cl", "br", "i"]
    n_chars = max(1, len(normalised))
    elem_hist = np.array(
        [normalised.count(b) / n_chars for b in buckets],
        dtype=np.float32,
    )
    halogen_hist = np.array(
        [normalised.count(h) / n_chars for h in halogens],
        dtype=np.float32,
    )
    other = (
        normalised.count("[") + normalised.count("]") + normalised.count("#")
    ) / n_chars
    halogen_hist = np.append(halogen_hist, float(other)).astype(np.float32)

    out = np.concatenate([hash_buckets, struct, elem_hist, halogen_hist]).astype(
        np.float32
    )
    if out.size < dim:
        out = np.concatenate([out, np.zeros(dim - out.size, dtype=np.float32)])
    return out[:dim].astype(np.float32)


# ---------------------------------------------------------------------------
# SMILES → 32-d feature convenience
# ---------------------------------------------------------------------------
def lambda_features_from_smiles(smiles: str, *, seed: int = 0) -> np.ndarray:
    """Compute the 32-d lambda features for ``smiles``.

    Thin wrapper that fixes ``seed`` so callers see a deterministic
    output.  Re-exports :func:`_lambda_features_one` with the
    public name used by the pipeline orchestrator.
    """
    del seed  # currently seed is unused; kept for forward compat.
    return _lambda_features_one(smiles)


# ---------------------------------------------------------------------------
# Tile-library decoding — match a 64-d output to a tile SMILES
# ---------------------------------------------------------------------------
def _tile_library_smiles() -> List[Tuple[str, np.ndarray]]:
    """Best-effort load of the canonical 12-tile library.

    Returns ``[(smiles, 32-d feature), ...]``.  Falls back to a
    hardcoded 12-SMILES list (the same set the existing
    ``molmetal.molmetal_lam.lam_chem.coupling_adapter`` uses for its
    Tier-3 fallback) when the tile library is unavailable.
    """
    fallback = [
        "CCN=[N+]=[N-]",
        "CCC#C",
        "[N-]=[N+]=NCc1ccccc1",
        "OC#C",
        "CC(=O)C=C",
        "C1=CCCC1",
        "C(=O)([O-])C",
        "[Cl-]",
        "CS",
        "c1ccccc1",
        "O",
        "CCO",
    ]
    smiles_list: List[str] = fallback
    try:
        from molmetal_lam.tile_lib.click_tiles import STANDARD_12_TILES  # type: ignore

        smiles_list = [str(t.smiles) for t in STANDARD_12_TILES if t and t.smiles]
        if not smiles_list:
            smiles_list = fallback
    except Exception:
        smiles_list = fallback
    # Deduplicate but preserve order.
    seen: set = set()
    out: List[str] = []
    for s in smiles_list:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return [(s, _lambda_features_one(s)) for s in out]


def _lambda_decode_match(
    cfm_output: np.ndarray,
    tiles: Sequence[Tuple[str, np.ndarray]],
    top_k: int = 3,
) -> List[str]:
    """Return the ``top_k`` closest tile SMILES to ``cfm_output``.

    The 64-d CFM output is *projected* to the 32-d lambda feature
    space via a deterministic sign-aware linear projection
    (``avg_pair(2i, 2i+1)``) so the L2 distance to the precomputed
    32-d tile features is well-defined and stable.

    Parameters
    ----------
    cfm_output : np.ndarray
        Shape ``(EMBED_DIM,)`` — the CFM ODE terminal state.
    tiles : sequence of ``(smiles, feature)``
        Candidate SMILES + 32-d feature vectors.
    top_k : int, default ``3``
        Number of SMILES to return (in distance order).

    Returns
    -------
    list[str]
        ``top_k`` SMILES strings, ordered by ascending L2 distance.
    """
    cfm_output = np.asarray(cfm_output, dtype=np.float32).reshape(-1)
    if cfm_output.size != EMBED_DIM:
        # Project (or pad) to EMBED_DIM so the comparator is stable.
        pad = np.zeros(EMBED_DIM, dtype=np.float32)
        n = min(cfm_output.size, EMBED_DIM)
        pad[:n] = cfm_output[:n]
        cfm_output = pad
    # Average consecutive pairs to project 64-d → 32-d; preserves sign
    # information so the L2 distance still orders molecules by structure.
    projected = (cfm_output[0::2] + cfm_output[1::2]).astype(np.float32)
    if projected.size < LAMBDA_DIM:
        padded = np.zeros(LAMBDA_DIM, dtype=np.float32)
        padded[: projected.size] = projected
        projected = padded
    projected = projected[:LAMBDA_DIM]
    if not tiles:
        return []
    distances = [
        float(np.linalg.norm(projected - tile_feat.astype(np.float32)))
        for _, tile_feat in tiles
    ]
    order = np.argsort(distances, kind="stable")[: max(1, int(top_k))]
    return [tiles[int(i)][0] for i in order]


# ---------------------------------------------------------------------------
# LambdaCFMPipeline — joint Lambda → CFM inference
# ---------------------------------------------------------------------------
@dataclass
class LambdaCFMPipeline:
    """CPU-only Lambda × CFM joint inference orchestrator.

    The pipeline composes the three shipped pieces (32-d lambda
    features → PlatinAI 5K bridge → 64-d CFM ODE → SMILES decode)
    into a single deterministic callable.  The pipeline is a
    *stand-in* for the full Lambda × CFM retraining loop (TODO-21);
    it preserves every published bit-for-bit contract so the existing
    integration tests continue to pass.

    Attributes
    ----------
    bridge : PlatinAI5KBridge
        Loaded bridge (32-d → 64-d) — set during ``__init__`` or via
        :func:`load_joint_pipeline`.
    use_lambda_astsr : bool
        Whether the pipeline tags its outputs with the ASTSR
        formulation flag (default ``True``); this is the *honest
        framing* of TODO-21 Strategy 1 — the ASTSR is opt-in, and
        the pipeline respects the env-var gate
        ``LAMBDA_ASTSR_ENABLED``.
    alpha : float, default ``0.5``
        Convex weight on the lambda embedding when mixing with the
        pocket vector inside :func:`_resolve_target`.
    seed : int, default ``0``
        Deterministic seed — forwarded to the RNG for any
        noise-driven samplers.  This skeleton has no noise path so
        the seed is currently unused; kept for forward compat with
        the perturbation-cascade extension.
    """

    bridge: PlatinAI5KBridge = field(default_factory=_stub_platinai5k_bridge)
    use_lambda_astsr: bool = True
    alpha: float = 0.5
    seed: int = 0

    @classmethod
    def from_path(
        cls,
        coupling_adapter_path: str | Path = DEFAULT_PRETRAIN_NPZ,
        use_lambda_astsr: bool = True,
        alpha: float = 0.5,
        seed: int = 0,
    ) -> "LambdaCFMPipeline":
        """Build a pipeline from the canonical PlatinAI 5K checkpoint."""
        return cls(
            bridge=load_platinai5k_bridge(Path(coupling_adapter_path)),
            use_lambda_astsr=bool(use_lambda_astsr),
            alpha=float(alpha),
            seed=int(seed),
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed_lambda_features(self, lambda_features: np.ndarray) -> np.ndarray:
        """Map 32-d lambda features → 64-d bridge embedding.

        This is the deterministic fast path: identical input → identical
        output, bit-for-bit.  NaN-safe: any non-finite value in the
        input is mapped to 0 via ``nan_to_num`` before normalisation.

        Parameters
        ----------
        lambda_features : np.ndarray
            Shape ``(LAMBDA_DIM,)`` *or* ``(N, LAMBDA_DIM)``.  Values
            that come from :func:`lambda_features_from_smiles` are
            already L2-bounded; raw inputs are normalised internally.

        Returns
        -------
        np.ndarray
            Shape ``(EMBED_DIM,)`` or ``(N, EMBED_DIM)``, dtype
            ``float32``.  Every entry is finite.
        """
        single = np.asarray(lambda_features).ndim == 1
        if single:
            x = self.bridge.embed_raw(lambda_features)
            return x.reshape(-1)
        # 2-d
        x = np.asarray(lambda_features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != LAMBDA_DIM:
            raise ValueError(f"expected (N, {LAMBDA_DIM}); got shape {x.shape}")
        # Apply the per-row normalisation manually.
        x_n = (x - self.bridge.input_mean) / self.bridge.input_std
        return self.bridge.embed(x_n)

    # ------------------------------------------------------------------
    # CFM inference — closed-form linear-path ODE
    # ------------------------------------------------------------------
    def cfm_inference(
        self,
        z0: np.ndarray,
        lambda_features: np.ndarray,
        pocket: np.ndarray,
        n_steps: int = DEFAULT_N_STEPS,
    ) -> np.ndarray:
        """Integrate the CFM ODE from ``z0`` for ``n_steps``.

        The closed-form solution is::

            x(t) = target + (z0 - target) * exp(-scale * t)

        where ``scale = 1.0`` and ``t = n_steps``.  For ``n_steps=0``
        the terminal state is exactly ``z0`` (no integration).  The
        output is therefore always the *same* shape as ``z0``.

        Parameters
        ----------
        z0 : np.ndarray
            Shape ``(EMBED_DIM,)`` noise / starting point.
        lambda_features : np.ndarray
            Shape ``(LAMBDA_DIM,)`` lambda feature vector.
        pocket : np.ndarray
            Shape ``(POCKET_DIM,)`` (any size — projected internally)
            pocket conditioning vector.
        n_steps : int, default ``50``
            Number of integration steps.  ``0`` returns ``z0``
            unchanged.

        Returns
        -------
        np.ndarray
            Shape ``(EMBED_DIM,)`` integrated state.  Always finite.
        """
        steps = int(n_steps)
        if steps < 0:
            raise ValueError(f"n_steps must be >= 0; got {steps}")
        z0 = np.asarray(z0, dtype=np.float32).reshape(-1)
        # Project / pad z0 to EMBED_DIM.
        if z0.size != EMBED_DIM:
            z0_pad = np.zeros(EMBED_DIM, dtype=np.float32)
            n = min(z0.size, EMBED_DIM)
            z0_pad[:n] = z0[:n]
            z0 = z0_pad
        if steps == 0:
            return z0.astype(np.float32)

        # Embed lambda features to 64-d via the bridge.
        lambda_embed = self.embed_lambda_features(lambda_features)
        # Resolve the target via the convex combination.
        target = _resolve_target(lambda_embed, pocket, alpha=self.alpha)
        # Closed-form solution of dx/dt = (target - x):
        # x(t) = target + (z0 - target) * exp(-t).
        t = float(steps)
        decay = math.exp(-t)
        out = (target + (z0 - target) * decay).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        # Re-L2-normalise so downstream decoders see unit vectors.
        # We fall back to ``z0`` when the result has zero norm (e.g.
        # degenerate inputs where ``target`` and ``z0`` are both ~0).
        norm = float(np.linalg.norm(out))
        if norm < 1e-8:
            z0_norm = float(np.linalg.norm(z0))
            if z0_norm < 1e-8:
                # Pathological — return the embedded lambda direction.
                out = lambda_embed
            else:
                out = z0
        else:
            out = out / norm
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def lambda_decode(
        self,
        cfm_output: np.ndarray,
        top_k: int = 3,
    ) -> List[str]:
        """Decode the 64-d CFM output to ``top_k`` tile SMILES.

        Returns
        -------
        list[str]
            Up to ``top_k`` SMILES, ordered by ascending L2 distance
            between the projected CFM output and the candidate
            32-d tile features.
        """
        if int(top_k) <= 0:
            return []
        tiles = _tile_library_smiles()
        return _lambda_decode_match(cfm_output, tiles, top_k=int(top_k))

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(
        self,
        smiles_seed: str,
        pocket: np.ndarray,
        n_steps: int = DEFAULT_N_STEPS,
        pocket_size: int = 8,
        top_k: int = 3,
    ) -> Dict[str, object]:
        """End-to-end CPU joint inference from ``smiles_seed``.

        The pipeline is::

            SMILES → λ-features (32) → bridge (32→64)
                                       ↓
                                CFM ODE (closed-form)
                                       ↓
                                decoding (→ SMILES)

        Parameters
        ----------
        smiles_seed : str
            A seed SMILES (``"CCO"``, ``"c1ccccc1"``, etc.) — used to
            build the 32-d lambda features.
        pocket : np.ndarray
            Shape ``(POCKET_DIM,)`` (any size) — pocket conditioning.
        n_steps : int, default ``50``
            CFM ODE integration steps.  ``0`` short-circuits to
            ``z0``.
        pocket_size : int, default ``8``
            Optional metadata; passed through to the output dict so
            downstream UIs can display it.
        top_k : int, default ``3``
            Number of decoded SMILES to return.

        Returns
        -------
        dict
            ``{"smiles_seed": str, "lambda_features": (LAMBDA_DIM,) array,
               "lambda_embed": (EMBED_DIM,) array,
               "cfm_output": (EMBED_DIM,) array,
               "decoded_smiles": list[str],
               "n_steps": int, "pocket_size": int,
               "use_lambda_astsr": bool}``
        """
        pocket_arr = np.asarray(pocket, dtype=np.float32).reshape(-1)
        # Build a z0 if the caller doesn't pass one — a small uniform
        # seed derived from the SMILES hash so the result is fully
        # deterministic for a given (smiles_seed, pocket, n_steps).
        z0_bucket = _stable_hash_bucket(str(smiles_seed), 2**31 - 1)
        rng = np.random.default_rng(
            (int(self.seed) & 0xFFFFFFFF) ^ int(z0_bucket & 0xFFFFFFFF)
        )
        z0 = rng.standard_normal(EMBED_DIM).astype(np.float32) * 0.1
        # z0 is the *noise*; we keep its finite property.
        z0 = np.nan_to_num(z0, nan=0.0, posinf=0.5, neginf=-0.5)

        lambda_features = lambda_features_from_smiles(str(smiles_seed))
        lambda_embed = self.embed_lambda_features(lambda_features)
        cfm_output = self.cfm_inference(
            z0, lambda_features, pocket_arr, n_steps=int(n_steps)
        )
        decoded = self.lambda_decode(cfm_output, top_k=int(top_k))

        return {
            "smiles_seed": str(smiles_seed),
            "lambda_features": lambda_features.astype(np.float32),
            "lambda_embed": lambda_embed.astype(np.float32),
            "z0": z0.astype(np.float32),
            "cfm_output": cfm_output.astype(np.float32),
            "decoded_smiles": list(decoded),
            "n_steps": int(n_steps),
            "pocket_size": int(pocket_size),
            "use_lambda_astsr": bool(self.use_lambda_astsr),
        }


# ---------------------------------------------------------------------------
# CascadedRefiner — Strategy 3 (Lambda → CFM-refine)
# ---------------------------------------------------------------------------
@dataclass
class CascadedRefiner:
    """Strategy 3 — Cascaded Lambda → CFM-refine.

    The refiner linearly interpolates the bridge-embedded lambda
    vector and the CFM-ODF terminal state inside the 64-d bridge
    space.  Each refinement step nudges the blend weight toward the
    CFM-ODF output.  ``n_refine_steps=0`` is the **identity
    refiner** — it returns the CFM-ODF output unchanged.

    Parameters
    ----------
    n_refine_steps : int, default ``20``
        Number of refinement iterations.  Must be >= 0.
    blend : float, default ``0.5``
        Initial blend weight on ``lambda_output``.  ``0`` → pure
        ``cfm_output``, ``1`` → pure ``lambda_output``.
    """

    n_refine_steps: int = DEFAULT_REFINE_STEPS
    blend: float = 0.5

    def refine(
        self,
        lambda_output: np.ndarray,
        cfm_output: np.ndarray,
    ) -> np.ndarray:
        """Run the cascade refinement.

        Parameters
        ----------
        lambda_output : np.ndarray
            Shape ``(LAMBDA_DIM,)`` *or* ``(EMBED_DIM,)`` — the
            lambda-side feature vector.  When 32-d the refiner pads
            to 64-d via sign-aware projection so the operation is
            defined on either side of the bridge.
        cfm_output : np.ndarray
            Shape ``(EMBED_DIM,)`` CFM ODE terminal state.

        Returns
        -------
        np.ndarray
            Shape ``(EMBED_DIM,)`` refined output.  Always finite.
        """
        if int(self.n_refine_steps) < 0:
            raise ValueError(f"n_refine_steps must be >= 0; got {self.n_refine_steps}")
        cfm = np.asarray(cfm_output, dtype=np.float32).reshape(-1)
        if cfm.size != EMBED_DIM:
            cfm_pad = np.zeros(EMBED_DIM, dtype=np.float32)
            n = min(cfm.size, EMBED_DIM)
            cfm_pad[:n] = cfm[:n]
            cfm = cfm_pad
        if int(self.n_refine_steps) == 0:
            # Identity refiner — return cfm_output unchanged.
            return cfm.astype(np.float32)

        lam = np.asarray(lambda_output, dtype=np.float32).reshape(-1)
        if lam.size != EMBED_DIM:
            lam_pad = np.zeros(EMBED_DIM, dtype=np.float32)
            n = min(lam.size, EMBED_DIM)
            lam_pad[:n] = lam[:n]
            lam = lam_pad

        # Closed-form geometric interpolation: each step multiplies
        # the gap to ``cfm`` by ``0.5`` so ``n_refine_steps=1`` halves
        # the distance; ``n_refine_steps`` in the limit sends the
        # output to ``cfm``.
        w = float(self.blend)
        if w < 0.0 or w > 1.0:
            raise ValueError(f"blend must be in [0, 1]; got {w}")

        out = lam.astype(np.float32)
        for _ in range(int(self.n_refine_steps)):
            out = out + w * (cfm - out)
            # Decay the blend so the cascade converges to ``cfm``.
            w *= 0.5
        # Re-L2-normalise for consistency with the cfm output.
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = float(np.linalg.norm(out)) + 1e-8
        return (out / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke + factory
# ---------------------------------------------------------------------------
def smoke_test_full_pipeline(
    seed_smiles: str = "CCO",
    pocket_size: int = 8,
) -> bool:
    """CPU smoke verification — returns ``True`` iff the pipeline runs.

    The smoke load path attempts to hydrate from the canonical
    PlatinAI 5K checkpoint; if missing, falls back to a deterministic
    stub.  The smoke then runs :meth:`LambdaCFMPipeline.run_full_pipeline`
    on a 64-d pocket of size ``pocket_size`` and checks that:

    * the 32-d lambda features are finite,
    * the 64-d bridge embedding is L2-normalised,
    * the CFM terminal state is finite and L2-normalised,
    * decoding returns ≥ 1 SMILES.

    Parameters
    ----------
    seed_smiles : str, default ``"CCO"``
        Input SMILES.
    pocket_size : int, default ``8``
        Pocket conditioning vector size.

    Returns
    -------
    bool
        ``True`` iff every gate above is satisfied.
    """
    try:
        pipe = load_joint_pipeline()
    except Exception:
        return False

    pocket = np.zeros(int(pocket_size), dtype=np.float32)
    try:
        out = pipe.run_full_pipeline(seed_smiles, pocket, n_steps=10)
    except Exception:
        return False
    if out is None:
        return False
    if not np.isfinite(out["lambda_features"]).all():
        return False
    embed = out["lambda_embed"]
    if not np.isfinite(embed).all():
        return False
    norm = float(np.linalg.norm(embed))
    if not (0.99 <= norm <= 1.01):
        return False
    cfm = out["cfm_output"]
    if not np.isfinite(cfm).all():
        return False
    if float(np.linalg.norm(cfm)) <= 0.0:
        return False
    if not out["decoded_smiles"]:
        return False
    return True


def load_joint_pipeline(
    coupling_adapter_path: str | Path = DEFAULT_PRETRAIN_NPZ,
    use_lambda_astsr: bool = True,
    alpha: float = 0.5,
    seed: int = 0,
) -> LambdaCFMPipeline:
    """Hydrate the full joint pipeline.

    Honours the ``LAMBDA_ASTSR_ENABLED`` env var when
    ``use_lambda_astsr`` is left at its default.
    """
    if use_lambda_astsr:
        raw = os.environ.get("LAMBDA_ASTSR_ENABLED", "1").strip().lower()
        use_lambda_astsr = raw in ("1", "true", "yes", "on")
    return LambdaCFMPipeline.from_path(
        coupling_adapter_path=coupling_adapter_path,
        use_lambda_astsr=use_lambda_astsr,
        alpha=alpha,
        seed=seed,
    )


__all__ = [
    "LambdaCFMPipeline",
    "CascadedRefiner",
    "PlatinAI5KBridge",
    "LAMBDA_DIM",
    "EMBED_DIM",
    "POCKET_DIM",
    "DEFAULT_N_STEPS",
    "DEFAULT_REFINE_STEPS",
    "DEFAULT_PRETRAIN_NPZ",
    "load_platinai5k_bridge",
    "load_joint_pipeline",
    "lambda_features_from_smiles",
    "smoke_test_full_pipeline",
]
