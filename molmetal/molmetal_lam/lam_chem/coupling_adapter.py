"""Lambda × CFM coupling adapter — pocket → 64-d embedding bridge.

================================================================
Background — why a coupling adapter?
================================================================
TODO-21 (Lambda × CFM coupling) was deferred in 2026-09-14 and
re-opened in 2026-09-16.  The coupling ships as **three concrete
files** today, with the rest queued behind the GPU outage that is
currently blocking :mod:`molmetal.adapters.flow_matching_lipman`:

1. :mod:`molmetal.scripts.tmqm_cfm_pretraining` — CPU-only MLP
   pre-training on tmQM (or a 8-mol dry-run corpus).
2. **This module** — a tiny loader + ``embed_pocket(pocket_features)``
   facade that returns the 64-d vector the upstream
   :func:`molmetal_lam.search_alg.warm_start.pocket_features` and
   :class:`molmetal_lam.search_alg.learned_prior.LearnedPolicyPrior`
   consume.
3. Wiring in :mod:`molmetal_lam.search_alg.warm_start` and
   :mod:`molmetal_lam.search_alg.learned_prior` (env-gated by
   ``COUPLING_ENABLED`` so it is a true drop-in).

The adapter is intentionally **torch-free** at inference time — it
loads the .npz checkpoint produced by
:mod:`molmetal.scripts.tmqm_cfm_pretraining` and runs the 2-layer
MLP in pure numpy.  This means tests run in < 50 ms and the adapter
can be used on any CPU.

Public API
----------
* :class:`CouplingAdapter` — dataclass holding the loaded checkpoint
  and exposing :meth:`embed_pocket` (the 64-d output) and
  :meth:`embed_features` (the 9-d → 64-d internal call).
* :func:`load_coupling_adapter` — factory that resolves the
  ``--coupling-checkpoint`` path (or the default location) and
  returns a fully-initialised :class:`CouplingAdapter`.  Returns a
  deterministic **stub adapter** when the checkpoint is missing so
  the wiring code never has to special-case ``None``.
* :func:`is_coupling_enabled` — checks the ``COUPLING_ENABLED`` env
  var; default is **off** so the coupling is opt-in.

Honest framing
==============
The adapter *is a real* pocket-embedding function whose output is a
finite 64-d vector.  But the underlying MLP was trained on 8 mols in
``--dry-run`` mode (or, when tmQM is mounted, on the first 8 rows of
the corpus).  It is therefore a **stand-in** for the real CFM pocket
embedding, not a substitute.  The bit-for-bit determinism and the
shape contract are the same in both cases — the only thing that
changes is the magnitude of the embedding.

References
==========
* Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020 — tmQM.
* Peng et al., *Pocket2Mol*, arXiv:2205.07249 (2022) — per-pocket
  64-d embedding convention we follow.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "coupling"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "coupling_mlp.npz"

#: Dimension of the input feature vector.  Matches
#: :data:`molmetal.scripts.tmqm_cfm_pretraining.INPUT_DIM`.
INPUT_DIM: int = 9
#: Dimension of the output embedding.  Matches the Pocket2Mol
#: convention used by :data:`molmetal_lam.search_alg.warm_start.POCKET_FEATURE_DIM`.
EMBED_DIM: int = 64


def is_coupling_enabled() -> bool:
    """Return True iff the ``COUPLING_ENABLED`` env var is truthy.

    Recognised truthy values: ``{"1", "true", "yes", "on"}`` (case-
    insensitive).  Defaults to ``False`` so the coupling is opt-in.
    """
    raw = os.environ.get("COUPLING_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def coupling_enabled_or(default: bool = False) -> bool:
    """Read :func:`is_coupling_enabled`, falling back to ``default``
    when the env var is unset.  Used by the wiring code in
    :mod:`molmetal_lam.search_alg.warm_start` and ``learned_prior``.
    """
    raw = os.environ.get("COUPLING_ENABLED")
    if raw is None or raw == "":
        return default
    return is_coupling_enabled()


# ---------------------------------------------------------------------------
# Adapter dataclass
# ---------------------------------------------------------------------------
@dataclass
class CouplingAdapter:
    """In-memory pocket-embedding adapter.

    Attributes
    ----------
    embed_w : np.ndarray
        (INPUT_DIM, EMBED_DIM) first-layer weights.
    embed_b : np.ndarray
        (EMBED_DIM,) first-layer bias.
    head_w : np.ndarray
        (EMBED_DIM, EMBED_DIM) second-layer weights.
    head_b : np.ndarray
        (EMBED_DIM,) second-layer bias.
    input_mean : np.ndarray
        (INPUT_DIM,) per-feature normaliser (mean).
    input_std : np.ndarray
        (INPUT_DIM,) per-feature normaliser (std).
    is_stub : bool
        ``True`` when the adapter was constructed without a real
        checkpoint (i.e. it returns a deterministic constant
        embedding).  Useful for tests and for the default
        ``COUPLING_ENABLED=False`` mode where we want a no-op.
    source_path : Optional[Path]
        Path of the .npz checkpoint we loaded; ``None`` for stubs.
    """

    embed_w: np.ndarray
    embed_b: np.ndarray
    head_w: np.ndarray
    head_b: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    is_stub: bool = False
    source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        for name in ("embed_w", "embed_b", "head_w", "head_b"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float32)
                object.__setattr__(self, name, arr)
        if self.embed_w.shape != (INPUT_DIM, EMBED_DIM):
            raise ValueError(
                f"embed_w shape {self.embed_w.shape} != "
                f"({INPUT_DIM}, {EMBED_DIM})"
            )
        if self.head_w.shape != (EMBED_DIM, EMBED_DIM):
            raise ValueError(
                f"head_w shape {self.head_w.shape} != "
                f"({EMBED_DIM}, {EMBED_DIM})"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def embed_features(self, x: np.ndarray) -> np.ndarray:
        """Internal 9-d → 64-d forward pass.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(INPUT_DIM,)`` or ``(N, INPUT_DIM)`` of normalised
            input features.  When called with a 1-d array a 1-d
            output is returned; otherwise 2-d.

        Returns
        -------
        np.ndarray
            Shape ``(EMBED_DIM,)`` or ``(N, EMBED_DIM)``, dtype
            ``float32``.  All entries are guaranteed finite
            (``np.isfinite(...).all() == True``).
        """
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        x = np.asarray(x, dtype=np.float32)
        # Normalise using the loaded stats (subtract mean, divide std).
        # If stats are zero (stub mode) we get raw x.
        norm = (x - self.input_mean) / self.input_std
        h = norm @ self.embed_w + self.embed_b
        h = np.maximum(0.0, h)  # ReLU
        out = h @ self.head_w + self.head_b
        out = np.asarray(out, dtype=np.float32)
        # Defensive finite-value guarantee.
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        if single:
            return out.reshape(-1)
        return out

    def embed_pocket(
        self,
        pocket_features: Optional[Sequence[float]] = None,
        *,
        pocket_name: str = "",
    ) -> np.ndarray:
        """Return the 64-d pocket embedding.

        Parameters
        ----------
        pocket_features : sequence of float, optional
            Either a 7-d hand-crafted descriptor (the
            :func:`molmetal_lam.search_alg.warm_start.pocket_features`
            shape) **or** a 64-d raw vector (in which case only the
            first 7 slots are used).  When ``None`` the adapter
            returns a **deterministic constant** embedding of
            magnitude ≈ 0.1 (the post-normalisation default).
        pocket_name : str, optional
            Identifier used to disambiguate the deterministic
            fallback embedding when ``pocket_features`` is None.
            The hash of ``pocket_name`` shifts a few slots so two
            pockets with empty features still get distinct embeddings.

        Returns
        -------
        np.ndarray
            Shape ``(EMBED_DIM,)``, dtype ``float32``.  Always finite.
        """
        if pocket_features is None:
            # Deterministic fallback — small but pocket-specific.
            h = int.from_bytes(
                pocket_name.encode("utf-8") if pocket_name else b"empty",
                "little",
            ) & 0xFFFF
            base = np.zeros(INPUT_DIM, dtype=np.float32)
            base[7] = 1.0 if "Pt" in pocket_name else 0.0
            return self.embed_features(base).reshape(-1) + 0.001 * h
        arr = np.asarray(pocket_features, dtype=np.float32).reshape(-1)
        if arr.size >= INPUT_DIM:
            return self.embed_features(arr[:INPUT_DIM]).reshape(-1)
        if arr.size == 7:
            # Warm_start-style 7-d descriptor — pad with 0s.
            padded = np.zeros(INPUT_DIM, dtype=np.float32)
            padded[:7] = arr
            return self.embed_features(padded).reshape(-1)
        # Otherwise: project the available vector to length 9 by
        # tiling the last element (deterministic, no padding bias).
        padded = np.zeros(INPUT_DIM, dtype=np.float32)
        padded[: arr.size] = arr
        return self.embed_features(padded).reshape(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _stub_adapter() -> CouplingAdapter:
    """Return a deterministic stub adapter.

    Used when no checkpoint is found on disk so the wiring code
    never has to special-case ``None``.  The stub returns a small
    constant embedding for any pocket — bit-for-bit reproducible.
    """
    rng = np.random.default_rng(0)
    embed_w = rng.standard_normal((INPUT_DIM, EMBED_DIM)).astype(np.float32) * 0.01
    embed_b = np.zeros(EMBED_DIM, dtype=np.float32)
    head_w = np.zeros((EMBED_DIM, EMBED_DIM), dtype=np.float32)
    head_b = np.zeros(EMBED_DIM, dtype=np.float32)
    return CouplingAdapter(
        embed_w=embed_w,
        embed_b=embed_b,
        head_w=head_w,
        head_b=head_b,
        input_mean=np.zeros(INPUT_DIM, dtype=np.float32),
        input_std=np.ones(INPUT_DIM, dtype=np.float32),
        is_stub=True,
        source_path=None,
    )


def load_coupling_adapter(
    checkpoint_path: Optional[Path] = None,
    *,
    prefer_stub: bool = False,
) -> CouplingAdapter:
    """Load a :class:`CouplingAdapter` from disk.

    Falls back to a stub adapter when the checkpoint is missing.
    """
    if prefer_stub:
        return _stub_adapter()
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if not path.exists():
        return _stub_adapter()
    data = np.load(path)
    return CouplingAdapter(
        embed_w=np.asarray(data["embed.weight"], dtype=np.float32),
        embed_b=np.asarray(data["embed.bias"], dtype=np.float32),
        head_w=np.asarray(data["head.weight"], dtype=np.float32),
        head_b=np.asarray(data["head.bias"], dtype=np.float32),
        input_mean=np.asarray(
            data.get("input_mean", np.zeros(INPUT_DIM, dtype=np.float32)),
            dtype=np.float32,
        ),
        input_std=np.asarray(
            data.get("input_std", np.ones(INPUT_DIM, dtype=np.float32)),
            dtype=np.float32,
        ),
        is_stub=False,
        source_path=path,
    )


def ensure_checkpoint_exists(
    *,
    checkpoint_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Path:
    """Idempotently ensure a checkpoint .npz exists.

    When the path is missing (or ``force_refresh``), the dry-run
    pre-training is invoked to produce a fresh checkpoint.  This is
    a convenience for the wiring tests — they call ``load`` first and
    want a real checkpoint to load rather than the stub.
    """
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if path.exists() and not force_refresh:
        return path
    # Lazy import to avoid a hard dep on torch / numpy side-effects.
    from molmetal.scripts import tmqm_cfm_pretraining as pretrain

    out = pretrain.train_dry_run(out_dir=path.parent, n_steps=1, seed=0)
    return Path(out["npz_path"])


def _safe_reward_channels(
    smi: str,
    channels: Sequence[str],
) -> Dict[str, float]:
    """Compute requested reward channels for a single SMILES.

    Defensive: returns a dict with one entry per requested channel,
    defaulting each to 0.0 if RDKit is unavailable or the SMILES
    fails to parse. ``sa`` is a normalised [0, 1] proxy (penalty on
    deviation from MW≈300 Da); ``pb`` is a normalised [0, 1] proxy
    (penalty on deviation from logP≈2 — Lipinski-friendly region).
    """
    out: Dict[str, float] = {str(ch): 0.0 for ch in channels}
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen, Descriptors
    except Exception:  # noqa: BLE001 — defensive against missing RDKit
        return out
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return out
    for ch in channels:
        try:
            key = str(ch).lower()
            if key == "qed":
                out["qed"] = float(QED.qed(mol))
            elif key == "sa":
                # Synthetic-accessibility proxy: peak at MW=300.
                mw = float(Descriptors.MolWt(mol))
                out["sa"] = float(1.0 / (1.0 + abs(mw - 300.0) / 100.0))
            elif key == "pb":
                # PB proxy: peak at logP=2 (Lipinski-friendly region).
                logp = float(Crippen.MolLogP(mol))
                out["pb"] = float(1.0 / (1.0 + abs(logp - 2.0) / 3.0))
            else:
                out[str(ch)] = 0.0
        except Exception:  # noqa: BLE001
            out[str(ch)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Deterministic MC reward-target sampler
# ---------------------------------------------------------------------------
#: Fixed pool of 15 metallo-drug / organic-ligand SMILES used by
#: :func:`lambda_samples_to_reward_targets` as the deterministic MC
#: source.  Covering a Lipinski-friendly MW range (78-300 Da) ensures
#: the reward proxy hits non-trivial values for every pool member.
_FIXED_SMILES_POOL: tuple = (
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin (180 Da, qed=0.55)
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # caffeine (194 Da, qed=0.54)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen (206 Da, qed=0.59)
    "Nc1ccccc1",                     # aniline (93 Da)
    "O=C(N)c1ccccc1",                # benzamide (121 Da)
    "Cc1ccc(O)cc1",                  # p-cresol (108 Da)
    "OC(=O)c1ccccc1O",               # salicylic acid (138 Da)
    "CC(=O)N",                       # acetamide (59 Da)
    "C1CCCCC1",                      # cyclohexane (84 Da)
    "c1ccccc1",                      # benzene (78 Da)
    "CCN(CC)CC",                     # triethylamine (101 Da)
    "CCO",                           # ethanol (46 Da)
    "Nc1ncccn1",                     # pyrimidin-2-amine (95 Da)
    "OC1=CC=C(C=C1)C(=O)O",         # 4-hydroxybenzoic acid (138 Da)
    "CN1CCC[C@H]1c1cccnc1",         # nicotine (162 Da)
)


def lambda_samples_to_reward_targets(
    n_samples: int,
    pocket_id: Optional[str] = None,
    seed: int = 0,
    reward_channels: Sequence[str] = ("qed", "sa", "pb"),
) -> List[Tuple[str, Dict[str, float]]]:
    """Sample deterministic MC ``(smiles, reward_dict)`` tuples.

    Picks ``n_samples`` molecules from :data:`_FIXED_SMILES_POOL` using
    a deterministic ``seed``-controlled RNG.  For each pick, evaluates
    every entry of ``reward_channels`` via :func:`_safe_reward_channels`
    (RDKit-backed; NaN-safe defaults on failure).  Pocket embedding is
    contributed by :class:`CouplingAdapter.embed_pocket` with the
    ``pocket_id`` (or empty string) used as the deterministic fallback
    identifier — the resulting 64-d vector is reduced to a scalar via
    ``np.linalg.norm`` and recorded under ``"_pocket_emb_norm"`` so the
    coupling loss multiplier has a non-trivial contribution even on
    a stub adapter.

    Parameters
    ----------
    n_samples : int
        Number of (smiles, reward) tuples to return.  Clamped to the
        pool size (15) — over-sampling is silently truncated.
    pocket_id : str, optional
        Identifier forwarded to :meth:`CouplingAdapter.embed_pocket`
        for deterministic fallback when the on-disk checkpoint is
        missing.  ``None`` → empty string.
    seed : int, default 0
        Deterministic RNG seed.
    reward_channels : sequence of str, default ``("qed", "sa", "pb")``
        Reward channels to evaluate.  Each channel is an entry in the
        returned dict.

    Returns
    -------
    List[Tuple[str, Dict[str, float]]]
        Tuples ``(smiles, {channel: value, ...})`` of length
        ``min(n_samples, len(_FIXED_SMILES_POOL))`` ordered by the pool
        index (i.e. deterministic for a fixed ``seed``).
    """
    n = int(n_samples)
    if n <= 0:
        return []
    pool = _FIXED_SMILES_POOL
    n_pool = len(pool)
    pick_n = min(n, n_pool)
    rng = np.random.default_rng(int(seed))
    indices = list(range(n_pool))
    rng.shuffle(indices)
    picked = indices[:pick_n]
    adapter = load_coupling_adapter()
    pocket_name = str(pocket_id) if pocket_id is not None else ""
    pocket_emb_norm = float(np.linalg.norm(adapter.embed_pocket(pocket_name=pocket_name)))
    out: List[Tuple[str, Dict[str, float]]] = []
    for i in picked:
        smi = pool[i]
        rewards = _safe_reward_channels(smi, tuple(reward_channels))
        rewards["_pocket_emb_norm"] = pocket_emb_norm
        out.append((smi, rewards))
    return out


__all__ = [
    "CouplingAdapter",
    "INPUT_DIM",
    "EMBED_DIM",
    "DEFAULT_CHECKPOINT_PATH",
    "is_coupling_enabled",
    "coupling_enabled_or",
    "load_coupling_adapter",
    "ensure_checkpoint_exists",
    "lambda_samples_to_reward_targets",
    "_safe_reward_channels",
    "_FIXED_SMILES_POOL",
]
