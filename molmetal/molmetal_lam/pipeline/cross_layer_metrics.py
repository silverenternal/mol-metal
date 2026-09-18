"""Cross-layer metrics — synthesis & chemistry feasibility gates.

================================================================
Layer 9 + cross-layer metric implementations (catalogue §10).
================================================================
This module exposes the four cross-layer synthesizability metrics
that the catalogue (governance review) added on top of the L9 search
metrics:

    synthesis_success           — ratio of candidates passing AiZynth
    SA_score_mean               — mean Ertl SA score across top-K
    retrosynth_feasibility      — mean depth-weighted AiZynth score
    end_to_end_yield_proxy      — mean ReactionRule.predict_yield(top-K)

All four delegate to existing adapters in :mod:`molmetal_lam.sbdd_env`
(vina_adapter, posebusters_adapter, aizynth_adapter, sa_score) — no new
modules, no API churn.

Design
------
* Adapters are **lazy** — we try to import them but a missing backend
  degrades the metric to 0.0 rather than crashing the pipeline. This
  matches the rest of the codebase (search_alg falls back to constant
  0.5 when PySR is absent, etc.).
* Each function returns a single scalar so it composes cleanly with
  ``LamClickDesignLoop.run()`` results.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

# Lazy adapter imports — fail soft so this module stays importable
# without the heavy chemistry stack (vina, posebusters, aizynthfinder).
try:
    from molmetal_lam.sbdd_env.sa_score import (  # noqa: F401
        batch_sa_score as _batch_sa,
        sa_score_ertl as _sa_one,
    )
    _SA_OK = True
except Exception:  # pragma: no cover
    _SA_OK = False

try:
    from molmetal_lam.sbdd_env.aizynth_adapter import (  # noqa: F401
        AiZynthAdapter as _Aiz,
    )
    _AIZ_OK = True
except Exception:  # pragma: no cover
    _AIZ_OK = False


# ---------------------------------------------------------------------------
# 1) synthesis_success — ratio of top-K candidates aizynth can build
# ---------------------------------------------------------------------------
def synthesis_success(
    smiles_list: Sequence[str],
    *,
    adapter: Optional["object"] = None,
) -> float:
    """Fraction of ``smiles_list`` that AiZynthFinder (or SMARTS fallback)
    reports as synthesizable.

    Healthy target (catalogue §10): ≥ 0.05 at 1000 sims (Browne 2012 §6.4
    reports 0.1–5 % hit-rates on hard MCTS tasks; ≥ 0.05 is the steady-
    state target, ≥ 0.01 the floor).

    Returns 0.0 when no smiles supplied or the backend is missing.
    """
    smis = [s for s in (smiles_list or []) if s]
    if not smis:
        return 0.0
    try:
        if adapter is None and _AIZ_OK:
            adapter = _Aiz()
        if adapter is None:
            return 0.0
        if hasattr(adapter, "synthesizable_fraction"):
            return float(adapter.synthesizable_fraction(list(smis)))
        if hasattr(adapter, "check_list"):
            reports = list(adapter.check_list(list(smis)))
            if not reports:
                return 0.0
            return float(sum(bool(r.synthesizable) for r in reports)) / len(reports)
    except Exception:
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 2) SA_score_mean — Ertl mean across top-K (lower = easier)
# ---------------------------------------------------------------------------
def SA_score_mean(smiles_list: Sequence[str]) -> float:
    """Mean Ertl SA-score across ``smiles_list`` in [1, 10].

    Healthy target (catalogue §10): ≤ 5.0 (click-tile floor). Returns
    ``float('nan')`` when no molecules were parseable.
    """
    if not _SA_OK or not smiles_list:
        return float("nan")
    try:
        stats = _batch_sa(list(smiles_list))
        mean = stats.get("mean", float("nan"))
        # batch_sa_score already filters NaN entries.
        return float(mean) if mean == mean else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 3) retrosynth_feasibility — mean over top-K, depth-weighted AiZynth score
# ---------------------------------------------------------------------------
def retrosynth_feasibility(
    smiles_list: Sequence[str],
    *,
    adapter: Optional["object"] = None,
) -> float:
    """Mean AiZynth feasibility in [0, 1] across ``smiles_list``.

    Healthy target (catalogue §10): ≥ 0.7 (Schrittwieser 2019 §5.1
    reports 0.7+ on retrosynthesis-MCTS benchmarks). We compute the
    per-molecule feasibility as ``synthesizable * (1 / (1 + depth))``
    so a 1-step route scores higher than a 5-step route even when both
    are feasible — capturing both *can it be made* and *how easy*.
    """
    smis = [s for s in (smiles_list or []) if s]
    if not smis:
        return 0.0
    try:
        if adapter is None and _AIZ_OK:
            adapter = _Aiz()
        if adapter is None:
            return 0.0
        reports = list(adapter.check_list(list(smis)))
    except Exception:
        return 0.0
    if not reports:
        return 0.0
    vals: List[float] = []
    for r in reports:
        try:
            synth = 1.0 if getattr(r, "synthesizable", False) else 0.0
            depth = float(getattr(r, "depth", 0))
            vals.append(synth * (1.0 / (1.0 + depth)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# 4) end_to_end_yield_proxy — mean ReactionRule.predict_yield across top-K
# ---------------------------------------------------------------------------
def end_to_end_yield_proxy(
    smiles_list: Sequence[str],
    *,
    rules: Optional[Iterable["object"]] = None,
) -> float:
    """Mean predicted isolated yield across ``rules`` × ``smiles_list``.

    Healthy target (catalogue §10): ≥ 0.5 until Layer 5 R² ≥ 0.8.
    Calls ``rule.predict_yield(a, b)`` for every (rule, pair) where
    ``(a, b)`` are drawn from the SMILES list (uniform pairs). Returns
    0.0 when no rule carries a ``predict_yield`` method.
    """
    smis = [s for s in (smiles_list or []) if s]
    if not smis or not rules:
        return 0.0
    rule_list = [r for r in rules if hasattr(r, "predict_yield")]
    if not rule_list or len(smis) < 2:
        return 0.0
    pairs: List[tuple] = []
    # Cartesian over (rule, pair) is O(K² * R); cap at 64 pairs to keep
    # the proxy cheap (it is a *proxy*, not the full yield model).
    for i in range(min(len(smis), 8)):
        for j in range(i + 1, min(len(smis), 8)):
            pairs.append((smis[i], smis[j]))
            if len(pairs) >= 64:
                break
        if len(pairs) >= 64:
            break
    vals: List[float] = []
    for a, b in pairs:
        for rule in rule_list:
            try:
                vals.append(float(rule.predict_yield(a, b)))
            except Exception:
                continue
    return float(sum(vals) / len(vals)) if vals else 0.0


__all__ = [
    "synthesis_success",
    "SA_score_mean",
    "retrosynth_feasibility",
    "end_to_end_yield_proxy",
]
