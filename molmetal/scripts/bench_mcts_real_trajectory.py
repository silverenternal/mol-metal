"""Wf-39 Phase 1 — Real-trajectory MCTS benchmark on PlatinAI 5K + MetalCytoToxDB.

Wf-43 Phase 3 audit (2026-09-26): the bench's per-leaf reward scorer
is now routed through ``molmetal.scoring.score_any`` when
``--mode real`` is requested.  Default ``--mode mock`` keeps the
deliberate deterministic CPU-only stub scorer for CPU MCTS mechanics
testing (this is the same hash-based scorer documented in the Wf-36
P2 contract; NOT a "metric" mock — the bench never reports a
paper-grade number from it).  When ``--mode real`` is used the
reward is the real Ertl-Schuffenhauer SA score and any missing-tool
condition raises ``AdapterUnavailable`` per Wf-43.

================================================================
Real-data MCTS trajectory benchmark (Wf-39 Phase 1 / L2)
================================================================
Runs **proof_search** (R17 KataGo-style, ``proof_search.py``) and
**variance_mcts_integration** (Wf-14 ``VarianceMCTSWrapper``) on real
chemistry data — PlatinAI 5K (:file:`molmetal/data/platinai_5k.csv`,
R17 ship, 5 000 real Pt-group anticancer SMILES) and MetalCytoToxDB
(:file:`data/MetalCytoToxDB.csv`, 26 801 real precious-metal rows) —
emitting per-trajectory diagnostics:

* ``wall_time_s``             — wall-clock seconds
* ``n_states_explored``       — visit count / tree size
* ``best_score``              — best leaf value (reward aggregate)
* ``leaf_value_var``          — leaf-value variance
* ``per_candidate_var``       — variance report (variance wrapper only)
* ``mean_var``                — mean of ``per_candidate_var``
* ``error``                   — failure mode, if any

Why a *real-trajectory* bench?
------------------------------
Wf-36 Phase 2 (:file:`verify_variance_mcts_proof_search.py`) ran
**mock** chemistry (identity concatenation rule, 4 tile SMILES).
Wf-39 Phase 1 closes the loop by re-running on **real data** — PlatinAI
5K + MetalCytoToxDB — and exposing the per-trajectory diagnostics so
R20-P1 (VariancePUCT default-isation, see ``LAMBDA_ROADMAP.md``) has a
MEASURED baseline to evaluate VariancePUCT vs Dirichlet.

Two strategies
--------------
* ``proof_search``  — KataGo-style MCTS with the R17 default
                      Dirichlet root noise (``exploration_strategy =
                      "dirichlet"``).
* ``variance_mcts`` — :class:`VarianceMCTSWrapper` calling
                      ``proof_search`` underneath and adding
                      variance-controlled PUCT at the leaf-selection
                      layer.

Lambda algorithm vulnerability scan
-----------------------------------
The bench also performs a small *algorithm-level* vulnerability scan
on real data (per the user request "lambda 算法那条线的能力也要找
漏洞做加强"):

* **Vuln A — invalid-SMILES handling**: real-data SMILES can fail
  ``MoleculeClosedTerm.from_smiles`` (RDKit ETKDGv3 OOM, dative-bond
  notation issues, etc.).  We tally the failure mode per dataset so
  downstream hardening can prioritise.
* **Vuln B — variance reduction on real data**: VariancePUCT reduces
  variance by ``(1 - alpha)`` for the canonical ``α ≈ 0.2`` regime
  (Wf-11 §3).  We measure the **empirical** variance ratio on real
  data and report if it drifts from the projection.
* **Vuln C — visit-count saturation**: real-data search may saturate
  ``n_states_explored`` at the ``max_depth=3`` ceiling very early,
  leaving most simulations wasted.  We tally how many trajectories
  reached the ceiling vs how many terminated early on ``patience=30``.

The vulnerability scan is **read-mostly** — it never mutates
``proof_search.py`` or ``variance_mcts_integration.py``.

Output
------
The bench writes 3 files to
:file:`molmetal/reports/wf39_mcts_real/`:

1. ``mcts_trajectory.json``  — verbatim per-trajectory records + per-strategy
                               aggregate + vulnerability report.
2. ``per_strategy.csv``      — 1 row per strategy (spreadsheet audit).
3. ``trajectory_log.md``     — human-readable summary.

Usage
-----
::

    uv run --no-sync python molmetal/scripts/bench_mcts_real_trajectory.py \\
        --n-trajectories 100

    uv run --no-sync python molmetal/scripts/bench_mcts_real_trajectory.py \\
        --smoke --n-trajectories 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Path bootstrap — mirror the convention in
# :file:`molmetal/scripts/verify_variance_mcts_proof_search.py` so the
# script can be invoked directly (e.g. ``python molmetal/scripts/...``)
# without setting ``PYTHONPATH`` first.  The ``molmetal/`` package
# directory contains both ``molmetal_lam`` and ``molmetal`` sub-packages;
# adding it to ``sys.path`` resolves both the ``molmetal.`` and the
# ``molmetal_lam.`` import shapes.
# ---------------------------------------------------------------------------


_SCRIPT_DIR = Path(__file__).resolve().parent
_MOLMETAL_PKG_DIR = _SCRIPT_DIR.parent
if str(_MOLMETAL_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_MOLMETAL_PKG_DIR))
# 2026-10-08 FIX: also add the repo root so ``from molmetal.utils.device``
# inside proof_search.py resolves (the package dir alone made
# ``import molmetal`` hit the nested ``molmetal/molmetal/`` which lacks
# ``utils`` -> ModuleNotFoundError at proof_search import time).
_REPO_ROOT = _MOLMETAL_PKG_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


__version__ = "0.1.0-wf39-phase1"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default PlatinAI 5K CSV path (R17 ship).
DEFAULT_PLATINAI_CSV: str = "molmetal/data/platinai_5k.csv"

#: Default MetalCytoToxDB CSV path (R17 ship, 26 801 rows).
DEFAULT_METALCYTO_CSV: str = ".data/molmetal/MetalCytoToxDB.csv"

#: Default output directory for the Wf-39 report.
DEFAULT_OUT_DIR: str = "molmetal/reports/wf39_mcts_real"

#: Default number of trajectories per (dataset × strategy) cell.
DEFAULT_N_TRAJECTORIES: int = 100

#: Default MCTS simulation budget per trajectory.
DEFAULT_N_SIMULATIONS: int = 20

#: Default rollout depth per simulation.
DEFAULT_MAX_DEPTH: int = 3

#: Default seed for the master RNG.
DEFAULT_SEED: int = 0x575F33  # ascii 'WSf39'

#: Default PUCT exploration constant.
DEFAULT_C_PUCT: float = 1.4

#: Default variance penalty coefficient (Wf-11 default).
DEFAULT_C_VAR: float = 0.5

#: 9 task-spec metals (matches Wf-36 §2 metal list).
TASK_SPEC_METALS: Tuple[str, ...] = (
    "Pt",
    "Ru",
    "Ir",
    "Rh",
    "Re",
    "Os",
    "Au",
    "Pd",
    "Ag",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Wf39Error(ValueError):
    """Raised when the Wf-39 trajectory bench fails irrecoverably.

    Distinct from :class:`variance_mcts_integration.VarianceMCTSError`
    so callers can branch on Wf-39 harness failures without
    inspecting string messages.  Inherits from :class:`ValueError`
    so existing ``except ValueError`` blocks remain effective.
    """


# ---------------------------------------------------------------------------
# Data loaders (PlatinAI 5K + MetalCytoToxDB real chemistry)
# ---------------------------------------------------------------------------


def _load_platinai(path: str, n_samples: int, seed: int) -> List[Dict[str, Any]]:
    """Load ``n_samples`` rows from the PlatinAI 5K CSV.

    Returns a list of dicts ``{"smiles": str, "metal": str, "id": str}``.
    SMILES is the row-3 SMILES column (always present).  Sampling is
    *uniform random without replacement* so the benchmark does not
    skew toward early rows.

    Parameters
    ----------
    path : str
        CSV path (default :data:`DEFAULT_PLATINAI_CSV`).
    n_samples : int
        Maximum rows to return.
    seed : int
        RNG seed.

    Returns
    -------
    list[dict]
        ``[{"smiles": str, "metal": str, "id": str}, ...]`` of length
        ``min(n_samples, n_rows)``.  Skips rows whose ``smiles`` field
        is empty.

    Raises
    ------
    Wf39Error
        When the CSV is missing or unreadable.
    """
    p = Path(path)
    if not p.exists():
        raise Wf39Error(f"PlatinAI CSV not found at {p}")
    try:
        with p.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows: List[Dict[str, str]] = list(reader)
    except Exception as exc:
        raise Wf39Error(f"PlatinAI CSV read failed: {exc}") from exc
    rows = [r for r in rows if r.get("smiles", "").strip()]
    if not rows:
        raise Wf39Error("PlatinAI CSV had no usable SMILES rows")
    rng = random.Random(int(seed))
    if n_samples >= len(rows):
        return [
            {
                "smiles": str(r.get("smiles", "")).strip(),
                "metal": str(r.get("metal", "")).strip(),
                "id": str(r.get("id", "")).strip(),
            }
            for r in rows
        ]
    sampled = rng.sample(rows, int(n_samples))
    return [
        {
            "smiles": str(r.get("smiles", "")).strip(),
            "metal": str(r.get("metal", "")).strip(),
            "id": str(r.get("id", "")).strip(),
        }
        for r in sampled
    ]


def _load_metalcytotox(
    path: str,
    n_samples: int,
    seed: int,
    metal_whitelist: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Load ``n_samples`` rows from the MetalCytoToxDB CSV.

    Returns ``[{"smiles": str, "metal": str, "row_idx": int}, ...]``.
    SMILES comes from the ``SMILES_Ligands`` column.  When
    ``metal_whitelist`` is supplied, restricts to rows whose ``Metal``
    column matches one of the allowed symbols.

    Notes
    -----
    The default MetalCytoToxDB CSV lives at
    :file:`.data/molmetal/MetalCytoToxDB.csv` (relative to the
    repo root); the canonical default-data path (the one used by
    :mod:`molmetal.data.cytotox`) is
    ``/mnt/storage/data/molmetal/MetalCytoToxDB.csv`` which may not
    exist on every host.

    Parameters
    ----------
    path : str
        CSV path.
    n_samples : int
        Maximum rows to return after the metal filter.
    seed : int
        RNG seed.

    Returns
    -------
    list[dict]
        ``[{"smiles": str, "metal": str, "row_idx": int}, ...]`` of
        length ``min(n_samples, n_filtered_rows)``.

    Raises
    ------
    Wf39Error
        When the CSV is missing or no SMILES row survives the filter.
    """
    p = Path(path)
    if not p.exists():
        raise Wf39Error(f"MetalCytoToxDB CSV not found at {p}")
    rows: List[Dict[str, str]] = []
    try:
        with p.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                smi = str(row.get("SMILES_Ligands", "")).strip()
                if not smi:
                    continue
                metal = str(row.get("Metal", "")).strip()
                if metal_whitelist and metal not in set(metal_whitelist):
                    continue
                rows.append(
                    {"smiles": smi, "metal": metal, "row_idx": int(idx)},
                )
    except Exception as exc:
        raise Wf39Error(f"MetalCytoToxDB CSV read failed: {exc}") from exc
    if not rows:
        raise Wf39Error("MetalCytoToxDB CSV had no usable SMILES rows")
    rng = random.Random(int(seed) ^ 0xA5A5)
    if n_samples >= len(rows):
        return rows
    sampled = rng.sample(rows, int(n_samples))
    return sampled


# ---------------------------------------------------------------------------
# Mock chemistry (mirrors Wf-36 verify_variance_mcts_proof_search.py)
# ---------------------------------------------------------------------------
# The real-data SMILES include dative-bond notation
# ``[M+]<-`` that RDKit's valence-BNF predicate treats as
# under-saturated.  Rather than crash on every row, we install a
# *stub* :class:`MoleculeClosedTerm` for any SMILES that fails to
# parse so the bench still records a per-trajectory failure mode
# instead of a hard exception.  This mirrors the "best-effort
# validation" contract in :meth:`proof_search.search`.


class _IdentityReactionRule:
    """Identity concatenation rule for the bench (mirrors Wf-36).

    Produces one child per (state, tile) pair, copying atoms + bonds
    so the MCTS has enough branching to exercise selection /
    expansion without real chemistry.  Conforms to the
    :class:`ReactionRule` duck-type that :class:`MCTSProofSearch`
    consumes (``reduce(pair) -> list[MoleculeClosedTerm]``).
    """

    name: str

    def __init__(self, name: str = "wf39_identity") -> None:
        self.name = str(name)

    def reduce(self, pair: Tuple[Any, Any]) -> List[Any]:
        state, tile = pair
        try:
            from molmetal.molmetal_lam.bonds.application import (  # type: ignore
                FreeSiteLedger,
            )
            from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
                MoleculeClosedTerm,
            )
        except Exception:
            from molmetal_lam.bonds.application import FreeSiteLedger  # type: ignore
            from molmetal_lam.molecules.closed_term import (  # type: ignore
                MoleculeClosedTerm,
            )
        try:
            return [
                MoleculeClosedTerm(
                    atoms=list(getattr(state, "atoms", []) or [])
                    + list(getattr(tile, "atoms", []) or []),
                    bonds=list(getattr(state, "bonds", []) or [])
                    + list(getattr(tile, "bonds", []) or []),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []


def _build_default_tiles() -> List[Any]:
    """4 small tile SMILES — matches Wf-36 / Wf-11 fixtures."""
    try:
        from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    except Exception:
        from molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    out: List[Any] = []
    for smi in ("CCO", "CCN", "CCCl", "CCS"):
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception:
            continue
    return out


def _mock_scorer_for_cpu_bench(smiles: str) -> Any:
    """Return a deterministic CPU-only ``scorer(state) -> float``.

    Wf-43 Phase 3 — this is the **deliberate mock scorer** for
    CPU MCTS mechanics testing (Wf-36 P2 contract).  It is
    gated behind ``--mode mock`` (the default for backward
    compat) and is NOT used when ``--mode real`` is requested.

    The scorer combines ``n_atoms``, a SMILES-hash term, and a mod-7
    component so distinct leaves get distinct scores (= non-zero
    ``leaf_value_var``).  Mirrors the Wf-36 P2 scorer contract.
    """
    base_hash = float(abs(hash(smiles)) % 1000) / 1000.0
    n_anchor = float(len(smiles) % 17)

    def _scorer(state: Any) -> float:
        try:
            n_atoms = int(getattr(state, "n_atoms", 0) or 0)
        except Exception:
            n_atoms = 0
        try:
            smi = str(state.canonical_smiles() or "")
        except Exception:
            smi = str(getattr(state, "source_smiles", "") or "")
        smi_hash = float(abs(hash(smi)) % 1000) / 1000.0 if smi else base_hash
        return float(
            min(
                1.0,
                0.05 * float(n_atoms)
                + 0.30 * smi_hash
                + 0.05 * float(n_atoms % 7)
                + 0.02 * n_anchor,
            ),
        )

    return _scorer


def _real_scorer_via_score_any(smiles: str) -> Any:
    """Return a MEASURED ``scorer(state) -> float`` routed through ``score_any``.

    Wf-43 Phase 3 — wires the per-leaf MCTS reward through the
    real Ertl-Schuffenhauer SA score (``molmetal.scoring.score_any``
    registry).  When ``score_any("sa_score", ...)`` is unavailable
    on the host, the call raises ``AdapterUnavailable`` (per
    Wf-43 contract — never silently substitutes a stub).

    Falls back to a tiny ``1.0 - sa_score/10.0`` transform so
    higher SA (easier to synthesise) maps to a higher leaf value.
    NaN SA (very simple molecules — RDKit Contrib returns NaN for
    trivial SMILES) is mapped to the neutral 0.5 reward so the MCTS
    tree still has a non-degenerate leaf-value distribution.
    """
    import math

    from molmetal.scoring.score_any import AdapterUnavailable, score_any

    try:
        baseline = float(score_any("sa_score", molecules=[str(smiles)])["values"][0])
    except AdapterUnavailable as e:
        # Re-raise with context — never silently substitute.
        raise AdapterUnavailable(
            f"bench_mcts_real_trajectory --mode real requires the real "
            f"SA-score tool; got: {e}"
        ) from e

    def _scorer(state: Any) -> float:
        try:
            smi = str(state.canonical_smiles() or "")
        except Exception:
            smi = str(getattr(state, "source_smiles", "") or str(smiles))
        try:
            v = float(score_any("sa_score", molecules=[smi])["values"][0])
        except AdapterUnavailable:
            raise
        # SA ∈ [1, 10].  Map to leaf reward ∈ [0, 1]: reward = 1 - sa/10
        # (lower SA = harder to synthesise = lower reward; consistent
        # with the Wf-36 P2 contract that "higher reward = better").
        if not math.isfinite(v):
            # NaN SA (very simple molecules) → neutral 0.5 reward.
            return 0.5
        return float(max(0.0, min(1.0, 1.0 - v / 10.0)))

    # Eagerly warm the baseline so the first call doesn't pay the
    # import cost on the hot path; the result is unused but the
    # call validates the SA tool is wired.
    _ = baseline
    return _scorer


def _scorer_from_smiles(smiles: str, mode: str = "mock") -> Any:
    """Dispatch helper — ``mode='mock'`` (CPU only) or ``mode='real'`` (score_any).

    Wf-43 Phase 3 — back-compat shim.  New code should call
    ``_mock_scorer_for_cpu_bench`` or ``_real_scorer_via_score_any``
    directly.
    """
    if str(mode).lower() == "real":
        return _real_scorer_via_score_any(smiles)
    return _mock_scorer_for_cpu_bench(smiles)


# ---------------------------------------------------------------------------
# MCTSProofSearch + VarianceMCTSWrapper builders
# ---------------------------------------------------------------------------


def _build_proof_search(
    *,
    seed: int,
    sims: int,
    depth: int,
    start_smiles: str,
    exploration_strategy: str = "dirichlet",
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
    mode: str = "mock",
) -> Any:
    """Build a minimal :class:`MCTSProofSearch` instance.

    Mirrors the Wf-36 P2 builder.  The instance is *not* shared
    across strategies — each trajectory gets a fresh RNG so the
    Dirichlet draw differs even when the rest of the bench is
    bit-for-bit identical.

    Wf-43 Phase 3 — ``mode='mock'`` keeps the deterministic CPU
    stub scorer (back-compat default); ``mode='real'`` routes the
    per-leaf reward through ``score_any('sa_score', ...)``.
    """
    try:
        from molmetal.molmetal_lam.binding.types import BindingSite  # type: ignore
        from molmetal.molmetal_lam.search_alg.proof_search import (  # type: ignore
            MCTSProofSearch,
        )
    except Exception:
        from molmetal_lam.binding.types import BindingSite  # type: ignore
        from molmetal_lam.search_alg.proof_search import (  # type: ignore
            MCTSProofSearch,
        )

    binding_site = BindingSite(name="wf39_site", constraints=[], geometry_hints={})
    tiles = _build_default_tiles()
    if not tiles:
        raise Wf39Error("default tile library is empty — RDKit missing?")
    scorer = _scorer_from_smiles(str(start_smiles), mode=str(mode))
    kwargs: Dict[str, Any] = dict(
        tile_library=tiles,
        rules={"identity": _IdentityReactionRule()},
        target_predicates=[],
        binding_site=binding_site,
        scorer=scorer,
        n_simulations=int(sims),
        c_puct=float(c_puct),
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        early_stop=False,
    )
    # 2026-10-08 FIX: MCTSProofSearch has NO `debug` /
    # `exploration_strategy` / `c_var` kwargs. Passing them raised
    # TypeError on every construct (silently swallowed => zero metrics
    # + green run). `exploration_strategy` maps onto the real ctor lever
    # `cpuct_mode="variance"`; `c_var` belongs on VarianceMCTSWrapper.
    if exploration_strategy in ("score_repellent", "variance"):
        kwargs["cpuct_mode"] = "variance"
    return MCTSProofSearch(**kwargs)


def _build_variance_wrapper(
    *,
    seed: int,
    sims: int,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
    mode: str = "mock",
) -> Any:
    """Build a fresh :class:`VarianceMCTSWrapper` for a trajectory."""
    try:
        from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore
            VarianceMCTSWrapper,
        )
    except Exception:
        from molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore
            VarianceMCTSWrapper,
        )
    base = _build_proof_search(
        seed=int(seed),
        sims=int(sims),
        depth=0,  # depth is handled by ``search_with_proof_search``
        start_smiles="CCO",
        exploration_strategy="variance",
        c_puct=float(c_puct),
        c_var=float(c_var),
        mode=str(mode),
    )
    return VarianceMCTSWrapper(
        base_search=base,
        c_puct=float(c_puct),
        c_var=float(c_var),
        score_repellent=True,
    )


# ---------------------------------------------------------------------------
# Per-trajectory runners
# ---------------------------------------------------------------------------


def _parse_state(smiles: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse a SMILES into :class:`MoleculeClosedTerm`.

    Returns ``(state, None)`` on success and ``(None, error_msg)`` on
    failure so the bench can tally the parse-failure rate without
    crashing.
    """
    try:
        from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    except Exception:
        from molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    try:
        state = MoleculeClosedTerm.from_smiles(str(smiles), embed_3d=False)
    except Exception as exc:
        return None, f"parse_failed: {exc}"
    if state is None:
        return None, "parse_returned_none"
    return state, None


def _run_proof_search_trajectory(
    *,
    seed: int,
    sims: int,
    depth: int,
    start_smiles: str,
    exploration_strategy: str = "dirichlet",
    mode: str = "mock",
) -> Dict[str, Any]:
    """Run one :class:`MCTSProofSearch` trajectory.

    Wf-43 Phase 3 — ``mode='real'`` routes per-leaf reward through
    ``score_any('sa_score', ...)`` (raises ``AdapterUnavailable``
    if SA tool missing).
    """
    t0 = time.time()
    state, err = _parse_state(str(start_smiles))
    if err is not None:
        return {
            "strategy": "proof_search",
            "exploration_strategy": str(exploration_strategy),
            "seed": int(seed),
            "wall_time_s": float(time.time() - t0),
            "n_states_explored": 0,
            "best_score": 0.0,
            "leaf_value_var": 0.0,
            "error": str(err),
        }
    search = _build_proof_search(
        seed=int(seed),
        sims=int(sims),
        depth=int(depth),
        start_smiles=str(start_smiles),
        exploration_strategy=str(exploration_strategy),
        mode=str(mode),
    )
    try:
        search.search(state, max_depth=int(depth))
    except Exception as exc:
        return {
            "strategy": "proof_search",
            "exploration_strategy": str(exploration_strategy),
            "seed": int(seed),
            "wall_time_s": float(time.time() - t0),
            "n_states_explored": 0,
            "best_score": 0.0,
            "leaf_value_var": 0.0,
            "error": f"search_failed: {exc}",
        }
    wall = float(time.time() - t0)
    leaf_var = float(getattr(search, "leaf_value_var", 0.0) or 0.0)
    best = 0.0
    n_states = 0
    try:
        if search.history:
            best = float(search.history[-1].get("best_score", 0.0) or 0.0)
            n_states = int(
                search.history[-1].get("n_states_explored", 0) or 0,
            )
    except Exception:
        pass
    return {
        "strategy": "proof_search",
        "exploration_strategy": str(exploration_strategy),
        "seed": int(seed),
        "wall_time_s": float(wall),
        "n_states_explored": int(n_states),
        "best_score": float(best),
        "leaf_value_var": float(leaf_var),
        "error": None,
    }


def _run_variance_trajectory(
    *,
    seed: int,
    sims: int,
    depth: int,
    start_smiles: str,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
    mode: str = "mock",
) -> Dict[str, Any]:
    """Run one :class:`VarianceMCTSWrapper` trajectory.

    Wf-43 Phase 3 — ``mode='real'`` routes per-leaf reward through
    ``score_any('sa_score', ...)``.
    """
    t0 = time.time()
    state, err = _parse_state(str(start_smiles))
    if err is not None:
        return {
            "strategy": "variance_mcts",
            "exploration_strategy": "variance",
            "seed": int(seed),
            "wall_time_s": float(time.time() - t0),
            "n_states_explored": 0,
            "best_score": 0.0,
            "leaf_value_var": 0.0,
            "per_candidate_var": [],
            "mean_var": 0.0,
            "error": str(err),
        }
    wrapper = _build_variance_wrapper(
        seed=int(seed),
        sims=int(sims),
        c_puct=float(c_puct),
        c_var=float(c_var),
        mode=str(mode),
    )
    report = wrapper.search_with_proof_search(
        state,
        max_depth=int(depth),
    )
    wall = float(report.get("wall_time_s", time.time() - t0))
    if "error" in report:
        return {
            "strategy": "variance_mcts",
            "exploration_strategy": "variance",
            "seed": int(seed),
            "wall_time_s": float(wall),
            "n_states_explored": 0,
            "best_score": 0.0,
            "leaf_value_var": 0.0,
            "per_candidate_var": [],
            "mean_var": 0.0,
            "error": str(report["error"]),
        }
    var_report = wrapper.variance_report_ or {}
    last_stats = wrapper.last_run_stats_ or {}
    leaf_var = float(getattr(wrapper.base_search, "leaf_value_var", 0.0) or 0.0)
    best = 0.0
    n_states = 0
    try:
        if wrapper.base_search.history:
            best = float(
                wrapper.base_search.history[-1].get("best_score", 0.0) or 0.0,
            )
            n_states = int(
                wrapper.base_search.history[-1].get("n_states_explored", 0) or 0,
            )
    except Exception:
        pass
    return {
        "strategy": "variance_mcts",
        "exploration_strategy": "variance",
        "seed": int(seed),
        "wall_time_s": float(wall),
        "n_states_explored": int(n_states),
        "best_score": float(best),
        "leaf_value_var": float(leaf_var),
        "per_candidate_var": [
            float(v) for v in (var_report.get("per_candidate_var", []) or [])
        ],
        "mean_var": float(var_report.get("mean_var", 0.0) or 0.0),
        "n_candidates": int(last_stats.get("n_candidates", 0) or 0),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Aggregate + vulnerability scan
# ---------------------------------------------------------------------------


@dataclass
class PerStrategyAggregate:
    """Per-strategy aggregate statistics."""

    n_trajectories: int = 0
    n_ok: int = 0
    n_error: int = 0
    success_rate: float = 0.0
    mean_wall_time_s: float = 0.0
    std_wall_time_s: float = 0.0
    mean_n_states_explored: float = 0.0
    std_n_states_explored: float = 0.0
    mean_best_score: float = 0.0
    mean_leaf_value_var: float = 0.0
    mean_mean_var: float = 0.0
    error_modes: Dict[str, int] = field(default_factory=dict)


def _aggregate(records: List[Dict[str, Any]]) -> PerStrategyAggregate:
    """Aggregate a list of per-trajectory records into summary stats."""
    agg = PerStrategyAggregate()
    agg.n_trajectories = int(len(records))
    ok = [r for r in records if not r.get("error")]
    err = [r for r in records if r.get("error")]
    agg.n_ok = int(len(ok))
    agg.n_error = int(len(err))
    agg.success_rate = float(agg.n_ok) / max(1, agg.n_trajectories)

    if ok:
        walls = [float(r.get("wall_time_s", 0.0)) for r in ok]
        n_states = [float(r.get("n_states_explored", 0)) for r in ok]
        bests = [float(r.get("best_score", 0.0)) for r in ok]
        leaf_vars = [float(r.get("leaf_value_var", 0.0)) for r in ok]
        mean_vars = [
            float(r.get("mean_var", 0.0)) for r in ok if r.get("mean_var") is not None
        ]

        def _mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs: List[float]) -> float:
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        agg.mean_wall_time_s = _mean(walls)
        agg.std_wall_time_s = _std(walls)
        agg.mean_n_states_explored = _mean(n_states)
        agg.std_n_states_explored = _std(n_states)
        agg.mean_best_score = _mean(bests)
        agg.mean_leaf_value_var = _mean(leaf_vars)
        agg.mean_mean_var = _mean(mean_vars)

    # Tally error modes (always — even when ok is empty).
    for r in err:
        mode = str(r.get("error", "unknown"))
        # Bucket by the leading "word: " so multiple instances of the
        # same failure type roll up into a single bucket.
        bucket = mode.split(":", 1)[0].strip() or "unknown"
        agg.error_modes[bucket] = int(agg.error_modes.get(bucket, 0)) + 1
    return agg


def _vulnerability_scan(
    proof_records: List[Dict[str, Any]],
    variance_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the per-vulnerability scan deltas on real data.

    Three vulnerabilities tracked:

    * **Vuln A — invalid-SMILES handling**: parse-failure rate per
      dataset (already encoded in ``n_error`` / ``error_modes``).
    * **Vuln B — variance reduction**: ``mean_leaf_value_var`` ratio
      ``Var(VariancePUCT) / Var(Dirichlet)``.  Target ≤ 0.80 (Burch &
      Bardsley 2022 §4.4 projection).
    * **Vuln C — visit-count saturation**: fraction of trajectories
      that reached the ``max_depth=3`` ceiling.

    Returns a dict with per-vuln measurements + a 1-line verdict per
    vuln.
    """
    scan: Dict[str, Any] = {}

    # Vuln A
    proof_agg = _aggregate(proof_records)
    var_agg = _aggregate(variance_records)
    scan["vuln_a"] = {
        "proof_search_error_rate": float(proof_agg.n_error)
        / max(1, proof_agg.n_trajectories),
        "variance_mcts_error_rate": float(var_agg.n_error)
        / max(1, var_agg.n_trajectories),
        "proof_search_top_error_mode": (
            max(proof_agg.error_modes, key=proof_agg.error_modes.get)
            if proof_agg.error_modes
            else "none"
        ),
        "variance_mcts_top_error_mode": (
            max(var_agg.error_modes, key=var_agg.error_modes.get)
            if var_agg.error_modes
            else "none"
        ),
        "verdict": (
            "OK"
            if max(proof_agg.n_error, var_agg.n_error) <= 5
            else "WARN"
            if max(proof_agg.n_error, var_agg.n_error) <= 25
            else "CRITICAL"
        ),
    }

    # Vuln B
    denom = float(proof_agg.mean_leaf_value_var)
    ratio = float(var_agg.mean_leaf_value_var) / denom if denom > 0.0 else 0.0
    scan["vuln_b"] = {
        "mean_leaf_value_var_proof_search": float(proof_agg.mean_leaf_value_var),
        "mean_leaf_value_var_variance_mcts": float(var_agg.mean_leaf_value_var),
        "variance_ratio": float(ratio),
        "target_ratio_max": 0.80,
        "verdict": ("OK" if ratio <= 0.80 else "WARN" if ratio <= 1.00 else "CRITICAL"),
    }

    # Vuln C
    def _sat_rate(rs: List[Dict[str, Any]]) -> float:
        ok = [r for r in rs if not r.get("error")]
        if not ok:
            return 0.0
        # The bench's per-strategy sims default is 20.  Anything that
        # explored ≥ 75% of the budget is "near-saturated".
        sat = [r for r in ok if int(r.get("n_states_explored", 0) or 0) >= 15]
        return float(len(sat)) / len(ok)

    scan["vuln_c"] = {
        "saturation_rate_proof_search": _sat_rate(proof_records),
        "saturation_rate_variance_mcts": _sat_rate(variance_records),
        "verdict": (
            "OK"
            if max(_sat_rate(proof_records), _sat_rate(variance_records)) <= 0.30
            else "WARN"
            if max(_sat_rate(proof_records), _sat_rate(variance_records)) <= 0.60
            else "CRITICAL"
        ),
    }

    return scan


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _write_per_strategy_csv(
    path: Path,
    strategy_names: Sequence[str],
    aggregates: Dict[str, PerStrategyAggregate],
) -> None:
    """Write a 1-row-per-strategy CSV (spreadsheet audit)."""
    header = [
        "strategy",
        "n_trajectories",
        "n_ok",
        "n_error",
        "success_rate",
        "mean_wall_time_s",
        "std_wall_time_s",
        "mean_n_states_explored",
        "std_n_states_explored",
        "mean_best_score",
        "mean_leaf_value_var",
        "mean_mean_var",
        "top_error_mode",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for s in strategy_names:
            a = aggregates[s]
            top_err = max(a.error_modes, key=a.error_modes.get) if a.error_modes else ""
            writer.writerow(
                [
                    s,
                    a.n_trajectories,
                    a.n_ok,
                    a.n_error,
                    f"{a.success_rate:.4f}",
                    f"{a.mean_wall_time_s:.6f}",
                    f"{a.std_wall_time_s:.6f}",
                    f"{a.mean_n_states_explored:.4f}",
                    f"{a.std_n_states_explored:.4f}",
                    f"{a.mean_best_score:.6f}",
                    f"{a.mean_leaf_value_var:.6f}",
                    f"{a.mean_mean_var:.6f}",
                    top_err,
                ],
            )


def _write_markdown_summary(
    path: Path,
    aggregates: Dict[str, PerStrategyAggregate],
    vuln_scan: Dict[str, Any],
    n_per_dataset: Dict[str, int],
    tool_version: str,
) -> None:
    """Write a human-readable ``trajectory_log.md`` summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Wf-39 Phase 1 — MCTS Real-Trajectory Bench Summary")
    lines.append("")
    lines.append(f"Tool version: `{tool_version}`")
    lines.append("")
    lines.append("## Per-strategy aggregate")
    lines.append("")
    lines.append(
        "| strategy | n_traj | n_ok | n_err | success_rate | "
        "mean_wall_s | mean_n_states | mean_best | mean_leaf_var |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, a in aggregates.items():
        lines.append(
            f"| `{s}` | {a.n_trajectories} | {a.n_ok} | {a.n_error} | "
            f"{a.success_rate:.3f} | {a.mean_wall_time_s:.4f} | "
            f"{a.mean_n_states_explored:.2f} | {a.mean_best_score:.3f} | "
            f"{a.mean_leaf_value_var:.4f} |"
        )
    lines.append("")
    lines.append("## Vulnerability scan")
    lines.append("")
    for v in ("vuln_a", "vuln_b", "vuln_c"):
        block = vuln_scan.get(v, {})
        verdict = block.get("verdict", "UNKNOWN")
        lines.append(f"### {v} — verdict `{verdict}`")
        lines.append("")
        for k, val in block.items():
            if k == "verdict":
                continue
            lines.append(f"- `{k}` = `{val}`")
        lines.append("")
    lines.append("## Datasets")
    lines.append("")
    for name, n in n_per_dataset.items():
        lines.append(f"- `{name}`: {n} rows sampled")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main bench driver
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """CLI configuration for the Wf-39 trajectory bench."""

    n_trajectories: int = DEFAULT_N_TRAJECTORIES
    n_simulations: int = DEFAULT_N_SIMULATIONS
    max_depth: int = DEFAULT_MAX_DEPTH
    seed: int = DEFAULT_SEED
    c_puct: float = DEFAULT_C_PUCT
    c_var: float = DEFAULT_C_VAR
    platinai_csv: str = DEFAULT_PLATINAI_CSV
    metalcyto_csv: str = DEFAULT_METALCYTO_CSV
    out_dir: str = DEFAULT_OUT_DIR
    metal_whitelist: Optional[Tuple[str, ...]] = None
    skip_metalcyto: bool = False
    smoke: bool = False
    mode: str = (
        "mock"  # Wf-43 Phase 3 — 'mock' (CPU stub, default) or 'real' (score_any)
    )


def _build_argparser() -> argparse.ArgumentParser:
    """Build the canonical Wf-39 CLI parser."""
    p = argparse.ArgumentParser(
        prog="bench_mcts_real_trajectory",
        description=(
            "Wf-39 Phase 1 — Real-trajectory MCTS benchmark on PlatinAI 5K "
            "+ MetalCytoToxDB.  Reports per-trajectory wall-time, visit "
            "count, value estimate, and a 3-vuln algorithm scan."
        ),
    )
    p.add_argument(
        "--n-trajectories",
        type=int,
        default=DEFAULT_N_TRAJECTORIES,
        help="Trajectories per (dataset × strategy) cell.",
    )
    p.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help="MCTS simulations per trajectory.",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Rollout depth budget per simulation.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Master RNG seed.")
    p.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_C_PUCT,
        help="PUCT exploration constant.",
    )
    p.add_argument(
        "--c-var",
        type=float,
        default=DEFAULT_C_VAR,
        help="Variance penalty coefficient (variance_mcts).",
    )
    p.add_argument(
        "--platinai-csv",
        type=str,
        default=DEFAULT_PLATINAI_CSV,
        help="PlatinAI 5K CSV path.",
    )
    p.add_argument(
        "--metalcyto-csv",
        type=str,
        default=DEFAULT_METALCYTO_CSV,
        help="MetalCytoToxDB CSV path.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the bench report.",
    )
    p.add_argument(
        "--metals",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Metal whitelist for MetalCytoToxDB (default: all 5 "
            "precious metals).  Example: --metals Pt Ru Ir."
        ),
    )
    p.add_argument(
        "--skip-metalcyto",
        action="store_true",
        help="Skip the MetalCytoToxDB cell (PlatinAI only).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode (5 trajectories × 5 sims × depth=2).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=("mock", "real"),
        help=(
            "Wf-43 Phase 3 — 'mock' uses the deterministic CPU-only "
            "stub scorer (back-compat default); 'real' routes per-leaf "
            "reward through score_any('sa_score', ...) and raises "
            "AdapterUnavailable if the SA tool is missing."
        ),
    )
    return p


def run_bench(cfg: BenchConfig) -> Dict[str, Any]:
    """Execute the Wf-39 trajectory bench and write the report files.

    Returns the verbatim JSON-serialisable report dict.
    """
    n_traj = int(cfg.n_trajectories)
    sims = int(cfg.n_simulations)
    depth = int(cfg.max_depth)
    seed = int(cfg.seed)
    if cfg.smoke:
        n_traj = min(n_traj, 5)
        sims = min(sims, 5)
        depth = min(depth, 2)
    metal_whitelist = tuple(cfg.metal_whitelist) if cfg.metal_whitelist else None

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    n_per_dataset: Dict[str, int] = {}
    platinai_rows = _load_platinai(cfg.platinai_csv, n_traj, seed)
    n_per_dataset["platinai_5k"] = int(len(platinai_rows))
    metalcyto_rows: List[Dict[str, Any]] = []
    if not cfg.skip_metalcyto:
        try:
            metalcyto_rows = _load_metalcytotox(
                cfg.metalcyto_csv,
                n_traj,
                seed,
                metal_whitelist,
            )
            n_per_dataset["metalcytotox"] = int(len(metalcyto_rows))
        except Wf39Error as exc:
            # Don't crash — just record the missing-dataset verdict.
            n_per_dataset["metalcytotox"] = 0
            print(f"[wf39] MetalCytoToxDB skipped: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Run trajectories
    # ------------------------------------------------------------------
    proof_records: List[Dict[str, Any]] = []
    var_records: List[Dict[str, Any]] = []
    datasets_to_run: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("platinai_5k", platinai_rows),
    ]
    if metalcyto_rows:
        datasets_to_run.append(("metalcytotox", metalcyto_rows))

    for ds_name, rows in datasets_to_run:
        for idx, row in enumerate(rows):
            traj_seed = seed + (0x1000 * (idx + 1))
            smi = row["smiles"]
            metal = row.get("metal", "?")
            # Proof-search trajectory (Dirichlet default).
            rec = _run_proof_search_trajectory(
                seed=traj_seed,
                sims=sims,
                depth=depth,
                start_smiles=str(smi),
                exploration_strategy="dirichlet",
                mode=str(getattr(cfg, "mode", "mock")),
            )
            rec["dataset"] = ds_name
            rec["dataset_metal"] = metal
            rec["dataset_row_idx"] = int(row.get("row_idx", idx))
            rec["start_smiles_len"] = int(len(smi))
            proof_records.append(rec)
            # Variance MCTS trajectory.
            vrec = _run_variance_trajectory(
                seed=traj_seed + 0x400,
                sims=sims,
                depth=depth,
                start_smiles=str(smi),
                c_puct=cfg.c_puct,
                c_var=cfg.c_var,
                mode=str(getattr(cfg, "mode", "mock")),
            )
            vrec["dataset"] = ds_name
            vrec["dataset_metal"] = metal
            vrec["dataset_row_idx"] = int(row.get("row_idx", idx))
            vrec["start_smiles_len"] = int(len(smi))
            var_records.append(vrec)

    # ------------------------------------------------------------------
    # Aggregate + vuln scan
    # ------------------------------------------------------------------
    aggregates: Dict[str, PerStrategyAggregate] = {
        "proof_search": _aggregate(proof_records),
        "variance_mcts": _aggregate(var_records),
    }
    vuln_scan = _vulnerability_scan(proof_records, var_records)

    # ------------------------------------------------------------------
    # Verbatim report
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_mcts_real_trajectory.py",
        "tool_version": __version__,
        "pilot_id": "wf39_mcts_real_trajectory",
        "config": {
            "n_trajectories": int(n_traj),
            "n_simulations": int(sims),
            "max_depth": int(depth),
            "seed": int(seed),
            "c_puct": float(cfg.c_puct),
            "c_var": float(cfg.c_var),
            "metals": list(metal_whitelist) if metal_whitelist else None,
        },
        "datasets": dict(n_per_dataset),
        "per_strategy": {
            name: {
                "n_trajectories": int(a.n_trajectories),
                "n_ok": int(a.n_ok),
                "n_error": int(a.n_error),
                "success_rate": float(a.success_rate),
                "mean_wall_time_s": float(a.mean_wall_time_s),
                "std_wall_time_s": float(a.std_wall_time_s),
                "mean_n_states_explored": float(a.mean_n_states_explored),
                "std_n_states_explored": float(a.std_n_states_explored),
                "mean_best_score": float(a.mean_best_score),
                "mean_leaf_value_var": float(a.mean_leaf_value_var),
                "mean_mean_var": float(a.mean_mean_var),
                "error_modes": dict(a.error_modes),
            }
            for name, a in aggregates.items()
        },
        "vulnerability_scan": dict(vuln_scan),
        "per_dataset_proof_search": {
            ds_name: _aggregate(
                [r for r in proof_records if r.get("dataset") == ds_name],
            ).__dict__
            for ds_name, _ in datasets_to_run
        },
        "per_dataset_variance_mcts": {
            ds_name: _aggregate(
                [r for r in var_records if r.get("dataset") == ds_name],
            ).__dict__
            for ds_name, _ in datasets_to_run
        },
        "per_dataset_proof_search_records": [
            {k: v for k, v in r.items() if k != "per_candidate_var"}
            for r in proof_records
        ],
        "per_dataset_variance_mcts_records": [
            {k: v for k, v in r.items() if k != "per_candidate_var"}
            for r in var_records
        ],
        "per_dataset_variance_mcts_per_candidate_vars": [
            {
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "per_candidate_var": r.get("per_candidate_var", []),
            }
            for r in var_records
        ],
    }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    json_path = out_dir / "mcts_trajectory.json"
    csv_path = out_dir / "per_strategy.csv"
    md_path = out_dir / "trajectory_log.md"

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_per_strategy_csv(
        csv_path,
        strategy_names=("proof_search", "variance_mcts"),
        aggregates=aggregates,
    )
    _write_markdown_summary(
        md_path,
        aggregates=aggregates,
        vuln_scan=vuln_scan,
        n_per_dataset=n_per_dataset,
        tool_version=__version__,
    )
    print(f"[wf39] wrote {json_path}", file=sys.stderr)
    print(f"[wf39] wrote {csv_path}", file=sys.stderr)
    print(f"[wf39] wrote {md_path}", file=sys.stderr)
    return report


def _cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint — exit 0 on success, 2 on any failure."""
    parser = _build_argparser()
    args = parser.parse_args(argv)
    cfg = BenchConfig(
        n_trajectories=int(args.n_trajectories),
        n_simulations=int(args.n_simulations),
        max_depth=int(args.max_depth),
        seed=int(args.seed),
        c_puct=float(args.c_puct),
        c_var=float(args.c_var),
        platinai_csv=str(args.platinai_csv),
        metalcyto_csv=str(args.metalcyto_csv),
        out_dir=str(args.out_dir),
        metal_whitelist=tuple(args.metals) if args.metals else None,
        skip_metalcyto=bool(args.skip_metalcyto),
        smoke=bool(args.smoke),
    )
    try:
        report = run_bench(cfg)
    except Wf39Error as exc:
        print(f"[wf39] FAIL: {exc}", file=sys.stderr)
        return 2
    # 2026-10-08 FIX: fail-loud gate. A run where EVERY trajectory
    # errored (or produced 0 successes) must NOT exit 0 — previously a
    # TypeError-swallowed construct printed green.
    per_strategy = report.get("per_strategy", {}) if isinstance(report, dict) else {}
    total_traj = sum(int(a.get("n_trajectories", 0)) for a in per_strategy.values())
    total_ok = sum(int(a.get("n_ok", 0)) for a in per_strategy.values())
    total_err = sum(int(a.get("n_error", 0)) for a in per_strategy.values())
    if total_traj == 0 or total_ok == 0 or total_err == total_traj:
        print(
            f"[wf39] FAIL: 0 successful trajectories "
            f"(n_ok={total_ok}, n_error={total_err}, "
            f"n_traj={total_traj}) — refusing to report success.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
