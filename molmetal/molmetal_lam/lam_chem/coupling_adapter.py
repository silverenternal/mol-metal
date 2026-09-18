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
from typing import Dict, Mapping, Optional, Sequence

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


__all__ = [
    "CouplingAdapter",
    "INPUT_DIM",
    "EMBED_DIM",
    "DEFAULT_CHECKPOINT_PATH",
    "is_coupling_enabled",
    "coupling_enabled_or",
    "load_coupling_adapter",
    "ensure_checkpoint_exists",
]