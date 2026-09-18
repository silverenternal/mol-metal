"""Wire-up helper: AiZynthFinder retrosynthesis as a SOTA scoring column.

================================================================
What this is
================================================================
This module is the *pipeline-side* glue that turns a list of Lambda
candidates into per-pocket retrosynthesis statistics, suitable for use
as a column in the r4_c_full_sweep report alongside DiffDock / FlowDock
columns.  It is intentionally decoupled from
:mod:`molmetal_lam.sbdd_env.aizynth_adapter` (which is the
:func:`RetrosynthesisChecker` Protocol implementation used by the search
itself) and from :mod:`molmetal_lam.sbdd_env.aizynth_isolated`
(the CPU-environment subprocess bridge).

The wire-up supports three execution modes selected by the caller:

* ``"smarts"`` (default, CPU-only) — RDKit reaction-SMARTS fallback via
  :func:`retrosynthesis.synthesis_success_rate`.  Always available;
  used as the SOTA-comparable column whenever a real AiZynthFinder
  policy / stock config is not on disk.

* ``"aizynthfinder"`` (in-process) — imports :mod:`aizynthfinder` in
  the worker process and constructs an
  :class:`molmetal_lam.sbdd_env.aizynth_adapter.AiZynthAdapter`.  Will
  only succeed when ``aizynthfinder`` is installed AND ``config.yml`` +
  pre-trained policy + stock files are available on disk.

* ``"aizynthfinder_isolated"`` (subprocess) — spawns the dedicated
  CPU-only AiZynth environment via
  :class:`molmetal_lam.sbdd_env.aizynth_isolated.IsolatedAiZynthChecker`
  through ``environments/aizynth/run.sh``.  Keeps the ROCm torch
  process isolated from the (non-ROCm) retrosynthesis process so
  aizynthfinder can run on whichever Python the isolated env ships.

Public dataclass
---------------
:class:`AiZynthScoreColumn` — per-pocket synthesis oracle column with
``status``, ``n_invoked``, ``n_scored``, ``synthesis_success_rate``
(in [0, 1]), ``n_synthesis_route`` (count of candidates with a
retrosynthesis route found), and ``per_smiles`` diagnostics.

Honest framing
--------------
* ``aizynthfinder`` is NOT installed in the default ROCm Python 3.12
  environment (hundreds of MB of policy weights and a stock file are
  needed for full MCTS).  When the adapter is asked to run real
  AiZynth but neither config nor weights are available, it returns
  ``status="unavailable"`` and the column is recorded as
  ``null``/``NaN`` — the same honest SOTA-comparable behaviour as
  :mod:`diffdock_sota_scoring` and :mod:`flowdock_sota_scoring`.
* The SMARTS fallback *is* a real measurement of synthesis feasibility
  against the five click reactions Lambda uses (CuAAC, SPAAC, SPC,
  DielsAlder, ThiolEne).  See :mod:`retrosynthesis` for the exact
  reverse-templates and their justification.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Default repo root for the cloned AiZynth checkout (not currently
# present in references/, but kept for forward-compatibility).
DEFAULT_REPO_ROOT = "molmetal/references/AiZynthFinder"
# Default isolated-environment entry point.
DEFAULT_AIZYNTH_RUN_SH = "environments/aizynth/run.sh"


@dataclass
class AiZynthScoreColumn:
    """Per-pocket SOTA scoring column from AiZynthFinder.

    Attributes
    ----------
    status : str
        One of ``{"unavailable", "smarts", "ok", "error", "no_candidates"}``.
    mode : str
        ``"smarts"``, ``"aizynthfinder"`` or ``"aizynthfinder_isolated"``.
    n_invoked : int
        Number of candidates we attempted to check.
    n_scored : int
        Number of candidates for which a finite boolean answer
        (``synthesizable``) was produced.
    n_synthesis_route : int
        Number of candidates for which the oracle reported
        ``synthesizable=True`` (i.e. at least one retrosynthesis route).
    synthesis_success_rate : float | None
        ``n_synthesis_route / n_scored`` in [0, 1]; ``None`` when
        ``n_scored == 0``.
    per_smiles : dict[str, dict]
        ``smiles -> {"synthesizable": bool, "depth": int,
        "engine": str, "route_smiles": list[str]}``.
    """

    status: str = "unavailable"
    mode: str = "smarts"
    n_invoked: int = 0
    n_scored: int = 0
    n_synthesis_route: int = 0
    synthesis_success_rate: Optional[float] = None
    per_smiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def aizynthfinder_available() -> bool:
    """True iff :mod:`aizynthfinder` is importable in the current Python."""
    try:
        import aizynthfinder  # noqa: F401
        return True
    except Exception:
        return False


def aizynth_config_available(config_path: Optional[str]) -> bool:
    """True iff ``config_path`` points to an existing file."""
    return bool(config_path) and Path(str(config_path)).is_file()


def aizynth_isolated_available(run_sh: str = DEFAULT_AIZYNTH_RUN_SH) -> bool:
    """True iff the dedicated AiZynth environment entry-point script exists.

    The isolated env ships its own pinned Python via ``uv run --isolated``,
    so the script's mere presence is a sufficient availability signal.
    """
    return Path(run_sh).is_file()


def _candidate_smiles(candidates: Sequence[dict]) -> List[str]:
    """Pull SMILES out of Lambda candidate dicts (only generated ones).

    Mirrors the pattern used by DiffDock/FlowDock wirings so the
    columns are comparable across scorers.
    """
    out: List[str] = []
    for cand in candidates:
        if cand.get("is_generated") is not True:
            continue
        smi = cand.get("smiles", "")
        if smi:
            out.append(smi)
    return out


def _finite_bool_rate(rows: Iterable[bool]) -> tuple[int, int]:
    n_total = 0
    n_true = 0
    for row in rows:
        if isinstance(row, bool):
            n_total += 1
            if row:
                n_true += 1
    return n_true, n_total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def score_candidates(
    candidates: Sequence[dict],
    *,
    mode: str = "smarts",
    config_path: Optional[str] = None,
    isolated_run_sh: str = DEFAULT_AIZYNTH_RUN_SH,
    timeout_sec: float = 600.0,
    max_iterations: int = 200,
    time_limit_s: int = 30,
    seed: int = 0,
) -> AiZynthScoreColumn:
    """Run retrosynthesis on each candidate SMILES.

    Parameters
    ----------
    candidates : sequence of dict
        Lambda candidate rows; only ``is_generated=True`` rows are
        scored.  Rows without a SMILES are skipped.
    mode : {"smarts", "aizynthfinder", "aizynthfinder_isolated"}
        Execution mode.  ``"smarts"`` is always available and uses
        RDKit reverse-templates from :mod:`retrosynthesis`.  The two
        real-AiZynth modes require external assets and degrade
        gracefully to ``status="unavailable"`` when those assets
        are missing.
    config_path : str | None
        AiZynthFinder ``config.yml`` (real mode only).  Recorded in
        ``notes`` for traceability.
    isolated_run_sh : str
        Path to ``environments/aizynth/run.sh``.  Only used in
        ``"aizynthfinder_isolated"`` mode.
    timeout_sec : float
        Wall-clock budget for the *entire batch* in isolated mode.
    max_iterations, time_limit_s, seed : int
        AiZynth search-loop parameters forwarded to the checker.
    """
    smiles_list = _candidate_smiles(candidates)
    column = AiZynthScoreColumn(mode=mode, n_invoked=len(smiles_list))
    if not smiles_list:
        column.status = "no_candidates"
        column.notes = "no generated candidates with a SMILES"
        return column
    if mode == "smarts":
        return _score_smarts(smiles_list, column)
    if mode == "aizynthfinder":
        return _score_aizynthfinder(smiles_list, column, config_path=config_path,
                                    max_iterations=max_iterations, time_limit_s=time_limit_s)
    if mode == "aizynthfinder_isolated":
        return _score_aizynthfinder_isolated(
            smiles_list, column,
            config_path=config_path, isolated_run_sh=isolated_run_sh,
            timeout_sec=timeout_sec, max_iterations=max_iterations,
            time_limit_s=time_limit_s, seed=seed,
        )
    column.status = "unavailable"
    column.notes = f"unknown synthesis-oracle mode: {mode!r}"
    return column


# ---------------------------------------------------------------------------
# SMARTS fallback (always available)
# ---------------------------------------------------------------------------
def _score_smarts(smiles_list: List[str], column: AiZynthScoreColumn) -> AiZynthScoreColumn:
    from molmetal_lam.sbdd_env.retrosynthesis import retrosynthesize
    per_smiles: Dict[str, Dict[str, Any]] = {}
    for smi in smiles_list:
        try:
            ok, depth, engine, leaves = _smarts_one(smi)
        except Exception as exc:  # pragma: no cover - defensive
            per_smiles[smi] = {"synthesizable": False, "depth": 0,
                                "engine": "smarts_error",
                                "route_smiles": [],
                                "error": f"{type(exc).__name__}: {exc}"}
            continue
        per_smiles[smi] = {"synthesizable": bool(ok), "depth": int(depth),
                           "engine": str(engine),
                           "route_smiles": list(leaves)}
    column.per_smiles = per_smiles
    n_route, n_scored = _finite_bool_rate(row["synthesizable"] for row in per_smiles.values())
    column.n_scored = n_scored
    column.n_synthesis_route = n_route
    column.synthesis_success_rate = (n_route / n_scored) if n_scored else None
    column.status = "smarts"
    column.notes = "RDKit reverse-SMARTS retrosynthesis (see retrosynthesis.py)"
    return column


def _smarts_one(smiles: str) -> tuple[bool, int, str, List[str]]:
    """Wrap :func:`retrosynthesize` to also report depth + engine + leaves.

    Uses :func:`retrosynthesize_with_report` which returns
    ``(synthesizable, list_of_rule_names)``.  We treat each fired
    rule as one retrosynthesis step (depth 1).
    """
    from molmetal_lam.sbdd_env.retrosynthesis import retrosynthesize_with_report
    ok, rule_names = retrosynthesize_with_report(smiles)
    n_rules = len(rule_names) if rule_names else 0
    return bool(ok), int(n_rules), "smarts_fallback", list(rule_names)


# ---------------------------------------------------------------------------
# Real AiZynthFinder (in-process)
# ---------------------------------------------------------------------------
def _score_aizynthfinder(
    smiles_list: List[str],
    column: AiZynthScoreColumn,
    *,
    config_path: Optional[str],
    max_iterations: int,
    time_limit_s: int,
) -> AiZynthScoreColumn:
    if not aizynthfinder_available():
        column.status = "unavailable"
        column.notes = "aizynthfinder not importable in this Python"
        return column
    if not aizynth_config_available(config_path):
        column.status = "unavailable"
        column.notes = (
            f"AiZynthFinder installed but config_path missing: "
            f"{config_path!r}; pass --synthesis-config"
        )
        return column
    try:
        from molmetal_lam.sbdd_env.aizynth_adapter import AiZynthAdapter
        adapter = AiZynthAdapter(config_path=config_path,
                                 max_iterations=max_iterations,
                                 time_limit_s=time_limit_s)
        reports = adapter.check_list(smiles_list)
        per_smiles: Dict[str, Dict[str, Any]] = {}
        for smi, report in zip(smiles_list, reports):
            per_smiles[smi] = {
                "synthesizable": bool(report.synthesizable),
                "depth": int(report.depth),
                "engine": str(report.engine),
                "route_smiles": list(report.route_smiles),
            }
        column.per_smiles = per_smiles
        n_route, n_scored = _finite_bool_rate(row["synthesizable"] for row in per_smiles.values())
        column.n_scored = n_scored
        column.n_synthesis_route = n_route
        column.synthesis_success_rate = (n_route / n_scored) if n_scored else None
        column.status = "ok" if n_scored else "unavailable"
        column.notes = f"in-process AiZynthFinder; config={config_path}"
    except Exception as exc:  # pragma: no cover - upstream
        column.status = "error"
        column.notes = f"{type(exc).__name__}: {exc}"
    return column


# ---------------------------------------------------------------------------
# Isolated AiZynthFinder (subprocess)
# ---------------------------------------------------------------------------
def _score_aizynthfinder_isolated(
    smiles_list: List[str],
    column: AiZynthScoreColumn,
    *,
    config_path: Optional[str],
    isolated_run_sh: str,
    timeout_sec: float,
    max_iterations: int,
    time_limit_s: int,
    seed: int,
) -> AiZynthScoreColumn:
    if not aizynth_config_available(config_path):
        column.status = "unavailable"
        column.notes = (
            f"isolated mode requires a valid config_path; got {config_path!r}"
        )
        return column
    if not aizynth_isolated_available(isolated_run_sh):
        column.status = "unavailable"
        column.notes = f"isolated AiZynth run.sh missing at {isolated_run_sh!r}"
        return column
    try:
        from molmetal_lam.sbdd_env.aizynth_isolated import IsolatedAiZynthChecker
        checker = IsolatedAiZynthChecker(
            config_path=str(config_path),
            max_iterations=max_iterations,
            time_limit_s=time_limit_s,
            seed=seed,
        )
        reports = checker.check_many(smiles_list)
        per_smiles: Dict[str, Dict[str, Any]] = {}
        for smi, report in zip(smiles_list, reports):
            per_smiles[smi] = {
                "synthesizable": bool(report.synthesizable),
                "depth": int(report.depth),
                "engine": str(report.engine),
                "route_smiles": list(report.route_smiles),
            }
        column.per_smiles = per_smiles
        n_route, n_scored = _finite_bool_rate(row["synthesizable"] for row in per_smiles.values())
        column.n_scored = n_scored
        column.n_synthesis_route = n_route
        column.synthesis_success_rate = (n_route / n_scored) if n_scored else None
        column.status = "ok" if n_scored else "unavailable"
        column.notes = f"isolated AiZynthFinder via {isolated_run_sh}; config={config_path}"
    except Exception as exc:  # pragma: no cover - subprocess
        column.status = "error"
        column.notes = f"{type(exc).__name__}: {exc}"
    return column


def rate_from_column(column: AiZynthScoreColumn) -> Optional[float]:
    """Sanitised access to ``synthesis_success_rate`` (NaN-safe)."""
    val = column.synthesis_success_rate
    if val is None:
        return None
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return float(val)


__all__ = [
    "AiZynthScoreColumn",
    "score_candidates",
    "aizynthfinder_available",
    "aizynth_config_available",
    "aizynth_isolated_available",
    "rate_from_column",
    "DEFAULT_REPO_ROOT",
    "DEFAULT_AIZYNTH_RUN_SH",
]