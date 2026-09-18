"""Worker-side helpers for :meth:`RewardAggregator._batch_rewards`.

The :class:`RewardAggregator` channel callables (SA via ``sascorer``,
QED via ``Descriptors.qed``, Vina_proxy as a hand-coded linear
score) are RDKit-bound and CPU-only.  When the closed loop needs to
score hundreds of candidates per iteration the per-call C++ overhead
is the bottleneck.

This module provides two helpers that let :meth:`RewardAggregator.
_batch_rewards` dispatch the per-state scoring to a
:class:`multiprocessing.Pool`:

* :func:`aggregator_to_config` — serialise the aggregator's
  *stateless* per-channel formulas (channel name → enable + weights +
  channel parameters) into a plain dict that pickles cleanly across
  process boundaries.
* :func:`score_one` — reconstruct the per-channel formulas in the
  worker process from the config dict + the canonical SMILES, and
  return a single float.

Only RDKit-backed channels (SA, QED, Vina_proxy) are evaluated in the
worker.  Channels that capture non-picklable state (e.g. a fitted
docking oracle) are silently dropped — the parent process still applies
them in the serial fallback when ``n_workers == 1``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Safe RDKit imports — workers re-import them in their own process.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — RDKit optional
    from rdkit import Chem as _Chem  # type: ignore
    from rdkit.Chem import Descriptors as _Descriptors  # type: ignore
    try:
        from rdkit.Contrib.SA_Score import sascorer as _sascorer  # type: ignore
    except Exception:  # pragma: no cover
        _sascorer = None  # type: ignore
    _RDKIT_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore
    _Descriptors = None  # type: ignore
    _sascorer = None  # type: ignore
    _RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------
def aggregator_to_config(agg: Any) -> Dict[str, Any]:
    """Extract the picklable subset of a :class:`RewardAggregator`.

    The returned dict contains the channel-enable flags + weights +
    ``vina_invert`` + bonus fields.  Channel *callables* are *not*
    forwarded (they may capture non-picklable RDKit state); instead the
    worker reconstructs them from the canonical RDKit API.
    """
    return {
        "w_vina": float(getattr(agg, "w_vina", 0.0) or 0.0),
        "w_sa": float(getattr(agg, "w_sa", 0.0) or 0.0),
        "w_qed": float(getattr(agg, "w_qed", 0.0) or 0.0),
        "w_vina_proxy": float(getattr(agg, "w_vina_proxy", 0.0) or 0.0),
        "w_posebusters": float(getattr(agg, "w_posebusters", 0.0) or 0.0),
        "w_pic50": float(getattr(agg, "w_pic50", 0.0) or 0.0),
        "w_retro": float(getattr(agg, "w_retro", 0.0) or 0.0),
        "w_reinvent4": float(getattr(agg, "w_reinvent4", 0.0) or 0.0),
        "w_synth": float(getattr(agg, "w_synth", 0.0) or 0.0),
        "vina_invert": bool(getattr(agg, "vina_invert", True)),
        "bonus_typed": float(getattr(agg, "bonus_typed", 0.0) or 0.0),
        "bonus_binder": float(getattr(agg, "bonus_binder", 0.0) or 0.0),
        # Channel *presence* flags — the worker only re-evaluates RDKit-
        # backed channels (SA, QED, Vina_proxy).  All other channels
        # are silently disabled because they may capture non-picklable
        # state (a docking oracle, an sklearn wrapper, etc.).
        "has_r_sa": getattr(agg, "r_sa", None) is not None,
        "has_r_qed": getattr(agg, "r_qed", None) is not None,
        "has_r_vina_proxy": getattr(agg, "r_vina_proxy", None) is not None,
        "has_r_vina": getattr(agg, "r_vina", None) is not None,
    }


# ---------------------------------------------------------------------------
# Per-channel worker formulas
# ---------------------------------------------------------------------------
def _score_sa(smi: str) -> float:
    """1 - sascorer/10, clipped to [0, 1].  Returns 0.0 on failure."""
    if not _RDKIT_AVAILABLE or _Chem is None or _sascorer is None or not smi:
        return 0.0
    try:
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        raw = float(_sascorer.calculate_score(mol))
        return max(0.0, min(1.0, 1.0 - raw / 10.0))
    except Exception:
        return 0.0


def _score_qed(smi: str) -> float:
    """Descriptors.qed in [0, 1].  Returns 0.0 on failure."""
    if not _RDKIT_AVAILABLE or _Chem is None or _Descriptors is None or not smi:
        return 0.0
    try:
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        return max(0.0, min(1.0, float(_Descriptors.qed(mol))))
    except Exception:
        return 0.0


def _score_vina_proxy(smi: str) -> float:
    """Lightweight Vina proxy — atom/bond counts, no SMARTS needed.

    The full feature-linear proxy on the parent aggregator also mixes
    in the :class:`SymbolicPrior` (when attached).  In the worker we
    cannot forward the prior (it captures non-picklable state), so we
    fall back to the prior-free feature-linear formula.  This matches
    the ``prior=None`` path on the parent aggregator, so the worker
    and serial paths agree when no prior is fitted.

    Returns 0.5 (constant baseline) when RDKit is unavailable.
    """
    if not _RDKIT_AVAILABLE or _Chem is None or not smi:
        return 0.5
    try:
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.5
        n_atoms = float(mol.GetNumAtoms())
        n_bonds = float(mol.GetNumBonds())
        # No free_sites signal available without the closed-term — use
        # n_free_electrons as a stand-in (cheap and RDKit-native).
        try:
            n_free = float(sum(
                a.GetNumRadicalElectrons() for a in mol.GetAtoms()
            ))
        except Exception:
            n_free = 0.0
    except Exception:
        return 0.5
    linear_proxy = (
        0.04 * n_atoms
        + 0.05 * n_bonds
        - 0.03 * max(0.0, n_free - 4.0)
    )
    raw = 0.5 + 0.5 * (0.5 - 0.5) + linear_proxy
    return float(max(0.0, min(1.0, raw)))


def _score_vina(smi: str) -> float:
    """Vina channel proxy — without a fitted oracle we return 0.0.

    The parent aggregator's ``r_vina`` is typically a docking-oracle
    wrapper that is *not* picklable.  The worker always returns 0.0
    for this channel, which is the same value the parent would return
    if the oracle is unavailable.  Callers that have a fitted oracle
    should rely on the serial path (``n_workers=1``) to keep the
    real oracle live.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------
def score_one(
    cfg: Dict[str, Any],
    smiles: str,
    satisfies_typed: bool,
    binds_target: bool,
) -> float:
    """Compute the aggregated reward for a single state (worker-side).

    Parameters
    ----------
    cfg : dict
        Output of :func:`aggregator_to_config` — plain-dict config
        containing channel-enable flags and weights.
    smiles : str
        Canonical SMILES for the state.
    satisfies_typed, binds_target : bool
        Boolean flags added with the same bonuses as the scalar path.

    Returns
    -------
    float
        The aggregated reward in the parent's scale.
    """
    try:
        value = 0.0

        if cfg.get("has_r_vina"):
            v = _score_vina(smiles)
            if cfg.get("vina_invert"):
                v = -v
            value += cfg["w_vina"] * v

        if cfg.get("has_r_sa"):
            value += cfg["w_sa"] * _score_sa(smiles)

        if cfg.get("has_r_qed"):
            value += cfg["w_qed"] * _score_qed(smiles)

        if cfg.get("has_r_vina_proxy"):
            value += cfg["w_vina_proxy"] * _score_vina_proxy(smiles)

        if satisfies_typed:
            value += cfg["bonus_typed"]
        if binds_target:
            value += cfg["bonus_binder"]
        return float(value)
    except Exception:
        # Defensive — never let a worker crash; return 0.0 so the
        # caller still has a well-defined score.
        return 0.0


__all__ = [
    "aggregator_to_config",
    "score_one",
]