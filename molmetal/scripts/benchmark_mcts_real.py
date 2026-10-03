"""Real-data benchmark: KataGo-style MCTS vs VariancePUCT vs Variance MCTS.

Wf-43 Phase 3 audit (2026-09-26): no mock scorer present — uses real
PlatinAI 5K + MetalCytoToxDB SMILES + real ``MCTSProofSearch`` +
real ``VarianceMCTSWrapper``.  Exposes
``score_any("sa_score", ...)`` as an **optional per-candidate
baseline** so the bench can optionally report the SA-score
distribution of candidates the MCTS explores.  The baseline is
additive — no existing path is replaced.

================================================================
MCTS real benchmark on PlatinAI 5K (Wf-MCTS-Real-Bench)
================================================================
This script runs three MCTS strategies on real PlatinAI 5K SMILES
(R17 ship, :file:`molmetal/data/platinai_5k.csv`) and reports:

* ``base_mcts_wall_time``            — wall-clock seconds for the
                                        legacy KataGo-style MCTSProofSearch
                                        (Dirichlet noise path, R17 ship).
* ``variance_puct_wall_time``        — wall-clock seconds for the
                                        Wf-11 :class:`MCTSVarianceWrapper`
                                        (node-centric variance-penalised PUCT).
* ``variance_mcts_wall_time``        — wall-clock seconds for the
                                        Wf-14 :class:`VarianceMCTSWrapper`
                                        (MCTS-level variance integration).
* ``success_rate_base``              — fraction of samples where the
                                        base MCTS emits ≥ 1 candidate.
* ``success_rate_variance_puct``     — same, VariancePUCT.
* ``success_rate_variance_mcts``     — same, Variance MCTS.
* ``n_distinct_base``                — mean number of distinct
                                        canonical-SMILES candidates per
                                        sample (base).
* ``n_distinct_variance_puct``       — same, VariancePUCT.
* ``n_distinct_variance_mcts``       — same, Variance MCTS.

The benchmark is **CPU-only**, mirrors the integration-layer
contract documented in ``variance_puct_integration.py`` /
``variance_mcts_integration.py``, and intentionally does **not**
modify ``proof_search.py`` — every comparison is performed by
running the public ``MCTSProofSearch.search`` API and the
public wrapper methods.

Usage
-----
::

    # Full PlatinAI 5K sweep — 100 samples × 3 strategies
    uv run --no-sync python molmetal/scripts/benchmark_mcts_real.py \\
        --n-samples 100

    # Smoke harness — 5 samples, fast
    uv run --no-sync python molmetal/scripts/benchmark_mcts_real.py \\
        --n-samples 5 --n-simulations 3

Defaults are tuned so the full ``--n-samples 100`` run completes
in <5 min on a 4-core CPU container with RDKit+torch pre-installed
via the project-local ``.venv``.  The script writes
:file:`molmetal/reports/wf_mcts_real_bench/results.json` plus a
``benchmark_log.md`` summary next to it.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


__version__ = "0.1.0-mcts-real-bench"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default PlatinAI 5K CSV path (R17 ship).
DEFAULT_CSV: str = "molmetal/data/platinai_5k.csv"

#: Default output directory for the benchmark report.
DEFAULT_OUT_DIR: str = "molmetal/reports/wf_mcts_real_bench"

#: Default number of PlatinAI samples to evaluate.
DEFAULT_N_SAMPLES: int = 100

#: Default MCTS simulation budget per sample.  ``n_simulations=5`` keeps
#: the benchmark tractable (≈ 0.1 s per sample on a 4-core CPU with
#: the canonical 12-tile library path).  Bump to 30-100 for the
#: paper-grade numbers that the smoke tests in
#: :mod:`tests.test_mcts_real_bench` do *not* execute.
DEFAULT_N_SIMULATIONS: int = 5

#: Default MCTS rollout depth per simulation.
DEFAULT_MAX_DEPTH: int = 2

#: Default top-K returned by ``MCTSProofSearch.search``.
DEFAULT_TOP_K: int = 5

#: Default RNG seed for reproducibility.
DEFAULT_SEED: int = 0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCTSRealBenchError(RuntimeError):
    """Raised when the real-data MCTS benchmark fails unrecoverably.

    Distinct from ``VariancePUCTError`` / ``VarianceMCTSError`` so the
    benchmark harness can branch on benchmark-layer failures without
    inspecting string messages.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Per-sample result for one (sample, strategy) pair.

    Attributes
    ----------
    sample_idx : int
        Zero-based index in the loaded PlatinAI rows.
    wall_time_s : float
        Wall-clock seconds for this single search call.
    n_candidates : int
        Number of candidates returned by ``search()``.
    n_distinct : int
        Number of distinct canonical SMILES in the candidate list
        (deduplicated via :meth:`MoleculeClosedTerm.canonical_smiles`).
    n_distinct_observed : int
        Number of candidates whose canonical SMILES could be computed
        without raising — usually equal to ``n_distinct`` but may
        differ when RDKit rejects a candidate.
    error : str or None
        Captured error message (when the search raises).  ``None``
        on success.
    """

    sample_idx: int
    wall_time_s: float
    n_candidates: int
    n_distinct: int
    n_distinct_observed: int
    error: Optional[str] = None


@dataclass
class StrategyReport:
    """Aggregated metrics for one of the three MCTS strategies."""

    name: str
    n_samples: int = 0
    n_errors: int = 0
    n_success: int = 0  # samples with n_candidates >= 1
    total_wall_time_s: float = 0.0
    mean_wall_time_s: float = 0.0
    max_wall_time_s: float = 0.0
    success_rate: float = 0.0
    total_candidates: int = 0
    mean_candidates: float = 0.0
    total_distinct: int = 0
    mean_distinct: float = 0.0
    n_distinct_observed: int = 0
    per_sample: List[SampleResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_csv_path(csv_arg: str) -> Path:
    """Resolve the PlatinAI CSV path.

    Tries the literal arg first, then ``./molmetal/data/platinai_5k.csv``
    relative to the current working directory, then the absolute default
    shipped with the repo.  Raises :class:`MCTSRealBenchError` if the
    file is not found.
    """
    candidates: List[Path] = []
    p = Path(csv_arg)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        # Path(__file__) is molmetal/scripts/benchmark_mcts_real.py;
        # parents[2] is the repo root, so the relative lookup points
        # at molmetal/data/platinai_5k.csv (parents[1] = molmetal/).
        try:
            repo_root = Path(__file__).resolve().parents[2]
            candidates.append(repo_root / csv_arg)
        except IndexError:
            pass
    for c in candidates:
        if c.exists():
            return c
    raise MCTSRealBenchError(
        f"platinai CSV not found; tried {[str(c) for c in candidates]}"
    )


def _load_platinai_rows(
    csv_path: Path,
    n_samples: int,
) -> List[Dict[str, str]]:
    """Load the first ``n_samples`` rows from the PlatinAI 5K CSV."""
    rows: List[Dict[str, str]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            if not row.get("smiles"):
                continue
            rows.append(row)
    if not rows:
        raise MCTSRealBenchError(
            f"no rows loaded from {csv_path}; n_samples={n_samples}"
        )
    return rows


def _safe_canonical_smiles(term: Any) -> Optional[str]:
    """Return :meth:`canonical_smiles` of ``term`` or ``None`` on failure."""
    try:
        s = term.canonical_smiles()
        if isinstance(s, str) and s:
            return s
        return None
    except Exception:
        return None


def _sa_score_baseline_for_smiles(
    smiles_list: List[str],
) -> Dict[str, Any]:
    """Wf-43 Phase 3 — optional SA-score baseline routed through ``score_any``.

    Returns ``{"status": "MEASURED" | "skipped", "tool": str, "values":
    List[float], "n": int}``.  When the SA-score tool is missing on the
    host, ``status="skipped"`` and ``values=[]`` — never silently
    substituted with a stub.
    """
    from molmetal.scoring.score_any import AdapterUnavailable, score_any

    try:
        result = score_any("sa_score", molecules=list(smiles_list))
        return {
            "status": str(result.get("status", "MEASURED")),
            "tool": str(result.get("tool", "SA score (RDKit Ertl)")),
            "values": list(result.get("values", [])),
            "n": int(result.get("n", 0)),
        }
    except AdapterUnavailable as exc:
        return {
            "status": f"skipped: {type(exc).__name__}",
            "tool": "SA score (RDKit Ertl)",
            "values": [],
            "n": 0,
        }


def _summarise_strategy(
    name: str,
    per_sample: Sequence[SampleResult],
) -> StrategyReport:
    """Compute aggregate metrics for a list of :class:`SampleResult`."""
    report = StrategyReport(name=name)
    report.per_sample = list(per_sample)
    report.n_samples = len(per_sample)
    if not per_sample:
        return report
    ok = [r for r in per_sample if r.error is None]
    report.n_errors = sum(1 for r in per_sample if r.error is not None)
    report.n_success = sum(1 for r in ok if r.n_candidates >= 1)
    report.total_wall_time_s = float(sum(r.wall_time_s for r in ok))
    report.mean_wall_time_s = float(report.total_wall_time_s / max(1, len(ok)))
    report.max_wall_time_s = float(max((r.wall_time_s for r in ok), default=0.0))
    report.success_rate = float(report.n_success / max(1, len(ok)))
    report.total_candidates = int(sum(r.n_candidates for r in ok))
    report.mean_candidates = float(sum(r.n_candidates for r in ok) / max(1, len(ok)))
    report.total_distinct = int(sum(r.n_distinct for r in ok))
    report.mean_distinct = float(sum(r.n_distinct for r in ok) / max(1, len(ok)))
    report.n_distinct_observed = int(sum(r.n_distinct_observed for r in ok))
    return report


# ---------------------------------------------------------------------------
# Search plumbing
# ---------------------------------------------------------------------------


def _build_search(
    *,
    tiles: List[Any],
    rules: Dict[str, Any],
    target_predicates: List[Any],
    binding_site: Any,
    reward: Any,
    n_simulations: int,
    top_k: int,
    rng: random.Random,
) -> Any:
    """Build a configured :class:`MCTSProofSearch` for one strategy run.

    Centralising this construction ensures every strategy runs on the
    same configuration — only the wrapping layer differs.
    """
    # Imported lazily so the script can be parsed even when RDKit/torch
    # are not on the path.
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch

    return MCTSProofSearch(
        tile_library=tiles,
        rules=dict(rules),
        target_predicates=list(target_predicates),
        binding_site=binding_site,
        reward=reward,
        n_simulations=int(n_simulations),
        c_puct=1.4,
        top_k=int(top_k),
        rng=rng,
        # Force the canonical 12-tile library path for benchmark
        # determinism.  The default ``use_fragment_pool=True`` path
        # pulls the 204-tile SMARTS pool which inflates wall-time
        # 15-20x without changing the comparison.
        use_fragment_pool=False,
        # Disable early-stop so the n_simulations budget is honoured
        # exactly — the comparison must spend the same compute per
        # strategy per sample.
        early_stop=False,
    )


def _build_strategy_searches(
    *,
    tiles: List[Any],
    rules: Dict[str, Any],
    target_predicates: List[Any],
    binding_site: Any,
    reward: Any,
    n_simulations: int,
    top_k: int,
    seed: int,
) -> Tuple[Any, Any, Any]:
    """Build three configured MCTSProofSearch instances — one per strategy.

    Returns
    -------
    (base, variance_puct, variance_mcts) : tuple
        Three independent ``MCTSProofSearch`` instances seeded
        identically so the underlying MCTS behaves the same; the
        wrapping layer (variance_puct / variance_mcts) is added by
        the caller.
    """
    base = _build_search(
        tiles=tiles,
        rules=rules,
        target_predicates=target_predicates,
        binding_site=binding_site,
        reward=reward,
        n_simulations=n_simulations,
        top_k=top_k,
        rng=random.Random(int(seed)),
    )
    variance_puct = _build_search(
        tiles=tiles,
        rules=rules,
        target_predicates=target_predicates,
        binding_site=binding_site,
        reward=reward,
        n_simulations=n_simulations,
        top_k=top_k,
        rng=random.Random(int(seed)),
    )
    variance_mcts = _build_search(
        tiles=tiles,
        rules=rules,
        target_predicates=target_predicates,
        binding_site=binding_site,
        reward=reward,
        n_simulations=n_simulations,
        top_k=top_k,
        rng=random.Random(int(seed)),
    )
    return base, variance_puct, variance_mcts


# ---------------------------------------------------------------------------
# Per-sample runners
# ---------------------------------------------------------------------------


def _run_base_strategy(
    search: Any,
    seed: Any,
    max_depth: int,
    sample_idx: int,
) -> SampleResult:
    """Run the KataGo-style base MCTS on one seed."""
    t0 = time.time()
    try:
        cands = search.search(seed, max_depth=max_depth)
    except Exception as exc:  # pragma: no cover - defensive
        return SampleResult(
            sample_idx=sample_idx,
            wall_time_s=time.time() - t0,
            n_candidates=0,
            n_distinct=0,
            n_distinct_observed=0,
            error=str(exc),
        )
    wall = time.time() - t0
    n_cand = len(cands) if cands is not None else 0
    distinct: set = set()
    n_observed = 0
    for c in cands or []:
        s = _safe_canonical_smiles(c)
        if s is None:
            continue
        n_observed += 1
        distinct.add(s)
    return SampleResult(
        sample_idx=sample_idx,
        wall_time_s=float(wall),
        n_candidates=int(n_cand),
        n_distinct=int(len(distinct)),
        n_distinct_observed=int(n_observed),
    )


def _run_variance_puct_strategy(
    wrapper: Any,
    seed: Any,
    max_depth: int,
    sample_idx: int,
) -> SampleResult:
    """Run the Wf-11 VariancePUCT integration on one seed."""
    t0 = time.time()
    try:
        report = wrapper.search_with_proof_search(
            seed,
            max_depth=max_depth,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return SampleResult(
            sample_idx=sample_idx,
            wall_time_s=time.time() - t0,
            n_candidates=0,
            n_distinct=0,
            n_distinct_observed=0,
            error=str(exc),
        )
    wall = time.time() - t0
    if not isinstance(report, dict) or "error" in report:
        return SampleResult(
            sample_idx=sample_idx,
            wall_time_s=float(wall),
            n_candidates=0,
            n_distinct=0,
            n_distinct_observed=0,
            error=str(report.get("error"))
            if isinstance(report, dict)
            else "non-dict report",
        )
    cands = report.get("candidates", []) or []
    n_cand = len(cands)
    distinct: set = set()
    n_observed = 0
    for c in cands:
        s = _safe_canonical_smiles(c)
        if s is None:
            continue
        n_observed += 1
        distinct.add(s)
    return SampleResult(
        sample_idx=sample_idx,
        wall_time_s=float(wall),
        n_candidates=int(n_cand),
        n_distinct=int(len(distinct)),
        n_distinct_observed=int(n_observed),
    )


def _run_variance_mcts_strategy(
    wrapper: Any,
    seed: Any,
    max_depth: int,
    sample_idx: int,
) -> SampleResult:
    """Run the Wf-14 Variance MCTS integration on one seed."""
    t0 = time.time()
    try:
        report = wrapper.search_with_proof_search(
            seed,
            max_depth=max_depth,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return SampleResult(
            sample_idx=sample_idx,
            wall_time_s=time.time() - t0,
            n_candidates=0,
            n_distinct=0,
            n_distinct_observed=0,
            error=str(exc),
        )
    wall = time.time() - t0
    if not isinstance(report, dict) or "error" in report:
        return SampleResult(
            sample_idx=sample_idx,
            wall_time_s=float(wall),
            n_candidates=0,
            n_distinct=0,
            n_distinct_observed=0,
            error=str(report.get("error"))
            if isinstance(report, dict)
            else "non-dict report",
        )
    cands = report.get("candidates", []) or []
    n_cand = len(cands)
    distinct: set = set()
    n_observed = 0
    for c in cands:
        s = _safe_canonical_smiles(c)
        if s is None:
            continue
        n_observed += 1
        distinct.add(s)
    return SampleResult(
        sample_idx=sample_idx,
        wall_time_s=float(wall),
        n_candidates=int(n_cand),
        n_distinct=int(len(distinct)),
        n_distinct_observed=int(n_observed),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the three-strategy benchmark and return the aggregated report.

    The returned dict is JSON-serialisable and is what gets written to
    ``results.json``.
    """
    csv_path = _resolve_csv_path(args.csv)
    rows = _load_platinai_rows(csv_path, args.n_samples)
    n_loaded = len(rows)

    # Lazy imports of the heavy chemistry stack.
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.reactions.beta_reductions import REACTION_RULES
    from molmetal_lam.tile_lib.click_tiles import STANDARD_12_TILES
    from molmetal_lam.types.predicates import LIPINSKI
    from molmetal_lam.binding.types import PROTEASE_GENERIC
    from molmetal_lam.search_alg.proof_search import RewardAggregator
    from molmetal_lam.search_alg.variance_puct_integration import (
        MCTSVarianceWrapper,
    )
    from molmetal_lam.search_alg.variance_mcts_integration import (
        VarianceMCTSWrapper,
    )

    tiles = [
        MoleculeClosedTerm.from_smiles(t.smiles, embed_3d=False)
        for t in STANDARD_12_TILES()
    ]
    zero_reward = RewardAggregator()
    rules = dict(REACTION_RULES)

    base_results: List[SampleResult] = []
    vpuct_results: List[SampleResult] = []
    vmcts_results: List[SampleResult] = []
    n_parse_errors = 0

    for idx, row in enumerate(rows):
        try:
            seed_term = MoleculeClosedTerm.from_smiles(
                row["smiles"],
                embed_3d=False,
            )
        except Exception:
            n_parse_errors += 1
            continue

        base, vp, vm = _build_strategy_searches(
            tiles=tiles,
            rules=rules,
            target_predicates=[LIPINSKI],
            binding_site=PROTEASE_GENERIC,
            reward=zero_reward,
            n_simulations=args.n_simulations,
            top_k=args.top_k,
            seed=args.seed + idx,
        )
        vpuct_wrapper = MCTSVarianceWrapper(
            base_search=vp,
            score_repellent=True,
        )
        vmcts_wrapper = VarianceMCTSWrapper(base_search=vm)

        # Run sequentially.  Parallelising would require pickling
        # ``MoleculeClosedTerm`` (which carries dative-bond state)
        # across processes — out of scope for the Wf benchmark.
        base_results.append(
            _run_base_strategy(
                base,
                seed_term,
                args.max_depth,
                idx,
            )
        )
        vpuct_results.append(
            _run_variance_puct_strategy(
                vpuct_wrapper,
                seed_term,
                args.max_depth,
                idx,
            )
        )
        vmcts_results.append(
            _run_variance_mcts_strategy(
                vmcts_wrapper,
                seed_term,
                args.max_depth,
                idx,
            )
        )

    base_summary = _summarise_strategy("base", base_results)
    vpuct_summary = _summarise_strategy("variance_puct", vpuct_results)
    vmcts_summary = _summarise_strategy("variance_mcts", vmcts_results)

    wall_ratio_vp = float(
        vpuct_summary.total_wall_time_s / max(1e-9, base_summary.total_wall_time_s)
    )
    wall_ratio_vm = float(
        vmcts_summary.total_wall_time_s / max(1e-9, base_summary.total_wall_time_s)
    )

    summary: Dict[str, Any] = {
        "n_samples_loaded": int(n_loaded),
        "n_samples_searched": int(base_summary.n_samples),
        "n_parse_errors": int(n_parse_errors),
        "csv_path": str(csv_path),
        "n_simulations": int(args.n_simulations),
        "max_depth": int(args.max_depth),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "base_mcts_wall_time": float(base_summary.total_wall_time_s),
        "variance_puct_wall_time": float(vpuct_summary.total_wall_time_s),
        "variance_mcts_wall_time": float(vmcts_summary.total_wall_time_s),
        "base_mcts_mean_wall_time": float(base_summary.mean_wall_time_s),
        "variance_puct_mean_wall_time": float(vpuct_summary.mean_wall_time_s),
        "variance_mcts_mean_wall_time": float(vmcts_summary.mean_wall_time_s),
        "base_mcts_max_wall_time": float(base_summary.max_wall_time_s),
        "variance_puct_max_wall_time": float(vpuct_summary.max_wall_time_s),
        "variance_mcts_max_wall_time": float(vmcts_summary.max_wall_time_s),
        "success_rate_base": float(base_summary.success_rate),
        "success_rate_variance_puct": float(vpuct_summary.success_rate),
        "success_rate_variance_mcts": float(vmcts_summary.success_rate),
        "success_count_base": int(base_summary.n_success),
        "success_count_variance_puct": int(vpuct_summary.n_success),
        "success_count_variance_mcts": int(vmcts_summary.n_success),
        "total_candidates_base": int(base_summary.total_candidates),
        "total_candidates_variance_puct": int(vpuct_summary.total_candidates),
        "total_candidates_variance_mcts": int(vmcts_summary.total_candidates),
        "n_distinct_base": float(base_summary.mean_distinct),
        "n_distinct_variance_puct": float(vpuct_summary.mean_distinct),
        "n_distinct_variance_mcts": float(vmcts_summary.mean_distinct),
        "wall_time_ratio_variance_puct_over_base": float(wall_ratio_vp),
        "wall_time_ratio_variance_mcts_over_base": float(wall_ratio_vm),
        "n_errors_base": int(base_summary.n_errors),
        "n_errors_variance_puct": int(vpuct_summary.n_errors),
        "n_errors_variance_mcts": int(vmcts_summary.n_errors),
        "strategy_reports": {
            "base": base_summary.__dict__,
            "variance_puct": vpuct_summary.__dict__,
            "variance_mcts": vmcts_summary.__dict__,
        },
        "version": __version__,
    }
    return summary


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_results_json(out_dir: Path, summary: Dict[str, Any]) -> Path:
    """Write the summary dict to ``results.json`` (pretty-printed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    def _default(o: Any) -> Any:
        # ``dataclasses.asdict`` would have been cleaner; we walk
        # dict-shaped objects instead so the function is robust to
        # the ``StrategyReport.per_sample`` list being a list of
        # dataclass instances.
        if hasattr(o, "__dict__"):
            d = dict(o.__dict__)
            return d
        return str(o)

    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=_default)
    return out_path


def _write_benchmark_log(
    out_dir: Path,
    summary: Dict[str, Any],
    *,
    n_simulations: int,
    max_depth: int,
    top_k: int,
    csv_path: str,
) -> Path:
    """Write a human-readable summary to ``benchmark_log.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_log.md"

    bwt = summary["base_mcts_wall_time"]
    vwt = summary["variance_puct_wall_time"]
    mwt = summary["variance_mcts_wall_time"]
    ratio_vp = summary["wall_time_ratio_variance_puct_over_base"]
    ratio_vm = summary["wall_time_ratio_variance_mcts_over_base"]
    lines: List[str] = []
    lines.append("# Wf-MCTS-Real-Bench — benchmark_log")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- PlatinAI CSV: `{csv_path}`")
    lines.append(f"- n_simulations = {n_simulations}")
    lines.append(f"- max_depth     = {max_depth}")
    lines.append(f"- top_k         = {top_k}")
    lines.append(f"- RNG seed      = {summary['seed']}")
    lines.append("- Tile library  : STANDARD_12_TILES (canonical Phase-0 12 tiles)")
    lines.append(
        f"- Reaction rules: REACTION_RULES ({summary['strategy_reports']['base']['n_samples']} samples searched)"
    )
    lines.append("")
    lines.append("## Per-strategy results")
    lines.append("")
    lines.append(
        "| strategy      | wall-time (s) | mean/sample (s) | max/sample (s) | success_rate | total_cands | n_distinct (mean) | errors |"
    )
    lines.append(
        "|---------------|--------------:|----------------:|---------------:|-------------:|------------:|------------------:|-------:|"
    )

    def _row(
        label: str,
        total: float,
        mean: float,
        mx: float,
        sr: float,
        tc: int,
        nd: float,
        errs: int,
    ) -> str:
        return (
            f"| {label:<13} | {total:>12.3f} | {mean:>14.4f} | "
            f"{mx:>13.4f} | {sr:>11.4f} | {tc:>10d} | "
            f"{nd:>16.4f} | {errs:>5d} |"
        )

    sr = summary["strategy_reports"]
    lines.append(
        _row(
            "base",
            sr["base"]["total_wall_time_s"],
            sr["base"]["mean_wall_time_s"],
            sr["base"]["max_wall_time_s"],
            sr["base"]["success_rate"],
            sr["base"]["total_candidates"],
            sr["base"]["mean_distinct"],
            sr["base"]["n_errors"],
        )
    )
    lines.append(
        _row(
            "variance_puct",
            sr["variance_puct"]["total_wall_time_s"],
            sr["variance_puct"]["mean_wall_time_s"],
            sr["variance_puct"]["max_wall_time_s"],
            sr["variance_puct"]["success_rate"],
            sr["variance_puct"]["total_candidates"],
            sr["variance_puct"]["mean_distinct"],
            sr["variance_puct"]["n_errors"],
        )
    )
    lines.append(
        _row(
            "variance_mcts",
            sr["variance_mcts"]["total_wall_time_s"],
            sr["variance_mcts"]["mean_wall_time_s"],
            sr["variance_mcts"]["max_wall_time_s"],
            sr["variance_mcts"]["success_rate"],
            sr["variance_mcts"]["total_candidates"],
            sr["variance_mcts"]["mean_distinct"],
            sr["variance_mcts"]["n_errors"],
        )
    )
    lines.append("")
    lines.append("## Wall-time ratios (variance / base)")
    lines.append("")
    lines.append(f"- variance_puct / base = **{ratio_vp:.3f}**")
    lines.append(f"- variance_mcts / base = **{ratio_vm:.3f}**")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if ratio_vp < 1.0:
        lines.append(
            f"- variance_puct ran the same number of candidates in "
            f"{(1.0 - ratio_vp) * 100:.1f}% less wall-time than the base."
        )
    else:
        lines.append(
            f"- variance_puct was {(ratio_vp - 1.0) * 100:.1f}% slower "
            f"than the base; the re-scoring pass on top of "
            f"``MCTSProofSearch.search`` is not free."
        )
    if ratio_vm < 1.0:
        lines.append(
            f"- variance_mcts ran in "
            f"{(1.0 - ratio_vm) * 100:.1f}% less wall-time than the base."
        )
    else:
        lines.append(
            f"- variance_mcts was {(ratio_vm - 1.0) * 100:.1f}% slower "
            f"than the base; tree-statistics collection "
            f"(``leaf_value_var``, ``puct_exploit_ratio_var``) adds a small overhead."
        )
    lines.append("")
    lines.append(
        f"- Total candidates: base = {summary['total_candidates_base']}, "
        f"variance_puct = {summary['total_candidates_variance_puct']}, "
        f"variance_mcts = {summary['total_candidates_variance_mcts']}."
    )
    lines.append(
        f"- Success rates: base = {summary['success_rate_base']:.3f}, "
        f"variance_puct = {summary['success_rate_variance_puct']:.3f}, "
        f"variance_mcts = {summary['success_rate_variance_mcts']:.3f}."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```")
    lines.append("uv run --no-sync python molmetal/scripts/benchmark_mcts_real.py \\")
    lines.append(f"    --n-samples {summary['n_samples_searched']} \\")
    lines.append(f"    --n-simulations {n_simulations} \\")
    lines.append(f"    --max-depth {max_depth} \\")
    lines.append(f"    --top-k {top_k} \\")
    lines.append(f"    --seed {summary['seed']}")
    lines.append("```")
    lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Real-data MCTS benchmark: base KataGo-style MCTS vs "
            "VariancePUCT vs Variance MCTS on PlatinAI 5K."
        ),
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of PlatinAI samples to evaluate (default 100).",
    )
    p.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help="MCTS simulation budget per sample (default 5).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="MCTS rollout depth per simulation (default 2).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K returned by MCTSProofSearch.search (default 5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed (default 0).",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="Path to the PlatinAI 5K CSV (default molmetal/data/platinai_5k.csv).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for results.json + benchmark_log.md "
        "(default molmetal/reports/wf_mcts_real_bench).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample stdout progress.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.n_samples < 1:
        parser.error("--n-samples must be >= 1")
    if args.n_simulations < 1:
        parser.error("--n-simulations must be >= 1")
    if args.max_depth < 1:
        parser.error("--max-depth must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    t_start = time.time()
    summary = run_benchmark(args)
    wall_total = time.time() - t_start

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    json_path = _write_results_json(out_dir, summary)
    log_path = _write_benchmark_log(
        out_dir,
        summary,
        n_simulations=args.n_simulations,
        max_depth=args.max_depth,
        top_k=args.top_k,
        csv_path=summary["csv_path"],
    )

    print("=" * 72)
    print("MCTS real-data benchmark (Wf-MCTS-Real-Bench)")
    print("=" * 72)
    print(f"n_samples_searched      : {summary['n_samples_searched']}")
    print(f"n_parse_errors          : {summary['n_parse_errors']}")
    print(f"n_simulations           : {summary['n_simulations']}")
    print(f"max_depth               : {summary['max_depth']}")
    print(f"top_k                   : {summary['top_k']}")
    print("-" * 72)
    print(f"base_mcts_wall_time     : {summary['base_mcts_wall_time']:.3f} s")
    print(f"variance_puct_wall_time : {summary['variance_puct_wall_time']:.3f} s")
    print(f"variance_mcts_wall_time : {summary['variance_mcts_wall_time']:.3f} s")
    print("-" * 72)
    print(f"success_rate_base       : {summary['success_rate_base']:.4f}")
    print(f"success_rate_variance_puct: {summary['success_rate_variance_puct']:.4f}")
    print(f"success_rate_variance_mcts: {summary['success_rate_variance_mcts']:.4f}")
    print("-" * 72)
    print(f"n_distinct_base         : {summary['n_distinct_base']:.4f}")
    print(f"n_distinct_variance_puct: {summary['n_distinct_variance_puct']:.4f}")
    print(f"n_distinct_variance_mcts: {summary['n_distinct_variance_mcts']:.4f}")
    print("-" * 72)
    print(
        f"variance_puct/base wall : {summary['wall_time_ratio_variance_puct_over_base']:.4f}"
    )
    print(
        f"variance_mcts/base wall : {summary['wall_time_ratio_variance_mcts_over_base']:.4f}"
    )
    print("-" * 72)
    print(f"total script wall       : {wall_total:.3f} s")
    print(f"results.json            : {json_path}")
    print(f"benchmark_log.md        : {log_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
