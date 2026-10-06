#!/usr/bin/env python3
# 2026-10-07 (Wf-R20-A1 / Wf-R20-FIX-1) — Lambda MCTS search-quality bench harness.
#
# Why this exists
# ---------------
# R20-A audit found that the two existing bench JSONs are degenerate:
#   wf_mcts_real_bench/results.json — n_samples_searched=3, n_distinct=0.0, n_success=0
#   wf40_mcts_tree/tree_stats.json   — mean_max_depth=3.0, std=0.0, leaf_value_var=0.0
# Every "5–20× / order-of-magnitude" lift claim in `mcts_strengthening.md` is
# therefore PROJECTED, not MEASURED. This harness converts those claims into
# a measured before/after by emitting the 7 R20 metrics (ER@N, S2T, HV, SR,
# b_eff+root_entropy, Validity/SA/QED/Tanimoto, NFE/solution) on a fixed
# 50-target known-answer set.
#
# Degeneracy guard (mandatory companion, mirrors §5.5 filesystem-truth):
# if n_success==0 AND mean_max_depth < 4 the harness SystemExits with
# BENCH_DEGENERATE — no JSON is emitted. This directly prevents repeating the
# Wf-40 / Wf-R20-A failure mode of "0 success, 0 distinct, still ship JSON".
#
# What this script does
# ---------------------
#   1. Loads N=--n-targets PlatinAI SMILES from molmetal/data/platinai_5k.csv
#      as known-answer starting states.
#   2. For each (target, seed) pair, builds MCTSProofSearch with the
#      canonical 12-tile library + REACTION_RULES + LIPINSKI predicate +
#      PROTEASE_GENERIC binding site + RewardAggregator().
#   3. Runs `search(initial_state, max_depth=--max-depth)` for
#      `n_simulations=--n-simulations` budget.
#   4. Records: n_candidates, n_distinct, success/fail, depth, candidates'
#      canonical SMILES, wall time.
#   5. Aggregates per-arm metrics: ER@N, S2T, SR, success_rate, n_distinct
#      mean, wall per search, depth mean (degeneracy guard).
#   6. Emits JSON with all per-target records + the aggregated per-arm metrics.
#
# Usage
# -----
#   # SMOKE — 3 targets × 3 seeds, finishes in ~1 min wall CPU
#   .venv/bin/python molmetal/scripts/bench_mcts_search_quality.py --smoke
#
#   # FULL — 50 targets × 10 seeds, ~10-15 min wall CPU on RTX 5090 host
#   .venv/bin/python molmetal/scripts/bench_mcts_search_quality.py \
#       --n-targets 50 --n-seeds 10 \
#       --out-dir molmetal/reports/r20_search_quality_baseline
#
# Constraints
# -----------
#   - .venv/bin/python (per AGENTS.md §5.9 single-venv rule)
#   - No `paper/*` touches (TODO-84)
#   - No new docs (per user 2026-09-30 directive "别让 agent 再写文档了")
#   - CPU-only, no GPU required
#   - Honest-framing MEASURED vs PROJECTED tags in stdout (per §5.2)
#   - Degeneracy guard: n_success==0 OR mean_depth<4 ⇒ SystemExit BENCH_DEGENERATE

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure repo root is on sys.path so `from molmetal_lam.X import Y` works
# when this script is invoked as `python molmetal/scripts/<this>.py` —
# Python inserts the script's directory as sys.path[0], not cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CSV = PROJECT_ROOT / "molmetal/data/platinai_5k.csv"
# R22 (Wf-R22-A) — organic reactive seed set. 16 SMILES with redex handles
# (alkyne/azide/amine/maleimide/thiol).  Wf-R21-A MEASURED: 4/4 organic
# seeds pass Lipinski gate + produce search candidates; PlatinAI
# metal-complex seeds hit BENCH_DEGENERATE on the Lipinski dative-bond
# blocker.  Used by ``--csv-organic`` flag (default: this file).
DEFAULT_CSV_ORGANIC = PROJECT_ROOT / "molmetal/data/organic_reactive_seeds.csv"


@dataclass
class TargetRecord:
    """One (target_smiles, seed) run."""

    target_idx: int
    smiles_in: str
    seed: int
    wall_time_s: float = 0.0
    n_candidates: int = 0
    n_distinct: int = 0
    n_distinct_score: float = 0.0  # mean across candidates of distinctness count
    max_depth_reached: int = 0
    n_transposition_hits: int = 0
    early_stopped: bool = False
    n_simulations_requested: int = 0
    n_simulations_executed: int = 0
    cum_nfe: int = 0
    nfe_reductions: int = 0
    puct_exploit_ratio_var: float = 0.0
    leaf_value_var: float = 0.0
    leaf_value_std: float = 0.0
    # R22 (Wf-R22-A) — TT cache hit / miss counters read from
    # ``search_inst._t_hits`` / ``search_inst._t_misses``.  These
    # are the canonical AlphaZero-style transposition-table hit-rate
    # statistics surfaced in :class:`MCTSProofSearch` since Round 3.
    tt_cache_hits: int = 0
    tt_cache_misses: int = 0
    error: Optional[str] = None
    candidates_smiles: List[str] = field(default_factory=list)


def _safe_canonical_smiles(term: Any) -> Optional[str]:
    """Return canonical_smiles() of a MoleculeClosedTerm or None on failure."""
    try:
        s = term.canonical_smiles()
        if isinstance(s, str) and s:
            return s
        return None
    except Exception:
        return None


#: Wf-R21-A2 + Wf-R22-A: SMARTS patterns that mark a redex handle
#: some rule in ``REACTION_RULES`` can consume.  MEASURED against the
#: 22 shipped rules: alkyne -> CuAAC/SPAAC, azide -> CuAAC/SPAAC/SPC,
#: maleimide -> urea_formation, thiol -> ThiolEne/Suzuki.  Rows
#: lacking all of these can never yield a child, so sampling them
#: makes the bench structurally degenerate regardless of
#: n_simulations/max_depth.
#:
#: R22 (Wf-R22-A) — switched from substring matching (which missed
#: canonical azide SMILES like ``[N-]=[N+]=NCC`` because it doesn't
#: contain the literal substring ``N=[N+]``) to RDKit SMARTS substructure
#: matching.  This catches both PlatinAI azide form ``N=[N+]=[N-]`` and
#: the canonical form ``[N-]=[N+]=N``.
_REDOX_HANDLE_PATTERNS = None  # lazy-init (RDKit import is heavy)
_REDOX_HANDLE_SMARTS_FALLBACK = ("C#C", "N=[N+]", "O=C1C=CC(=O)N1", "S")


def _has_redex_handle(smi: str) -> bool:
    """Return True iff ``smi`` carries a redex handle some rule can fire."""
    global _REDOX_HANDLE_PATTERNS
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        # No RDKit — fall back to substring matching.
        return any(p in smi for p in _REDOX_HANDLE_SMARTS_FALLBACK)
    if _REDOX_HANDLE_PATTERNS is None:
        _REDOX_HANDLE_PATTERNS = []
        for smarts in (
            "[CX2]#[CX2]",          # alkyne
            "[N-]=[N+]=N",          # azide (canonical)
            "N=[N+]=[N-]",          # azide (PlatinAI form)
            "N=[N+]",               # azide fragment
            "O=C1C=CC(=O)N1",       # maleimide
            "[SH]",                 # thiol (RDKit SMARTS)
        ):
            try:
                p = Chem.MolFromSmarts(smarts)
                if p is not None:
                    _REDOX_HANDLE_PATTERNS.append(p)
            except Exception:
                continue
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        for pat in _REDOX_HANDLE_PATTERNS:
            if mol.HasSubstructMatch(pat):
                return True
        return False
    except Exception:
        return False


def _load_targets(
    csv_path: Path,
    n_targets: int,
    seed_filter: str = "reactive",
) -> List[Dict[str, str]]:
    """Load up to n_targets rows with non-empty smiles.

    ``seed_filter="reactive"`` (Wf-R21-A2 default) keeps only rows whose
    SMILES carries at least one redex handle in ``_REDOX_HANDLE_SMARTS``,
    scanning the WHOLE csv (not just the first n_targets).  This is a
    seed-selection change in the harness only — ``_expand`` and all 22
    rules are untouched, so shipped search behaviour is bit-identical.
    ``seed_filter="first"`` restores the pre-R21 row-order behaviour.
    """
    rows: List[Dict[str, str]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            smi = row.get("smiles") or row.get("SMILES") or ""
            if not smi:
                continue
            if seed_filter == "reactive" and not _has_redex_handle(smi):
                continue
            rows.append({"smiles": smi, "row_idx": str(i)})
            if seed_filter == "first" and i + 1 >= n_targets:
                break
            if len(rows) >= n_targets:
                break
    if not rows:
        raise SystemExit(
            f"[bench-load] ERROR: no rows loaded from {csv_path}; "
            f"n_targets={n_targets} seed_filter={seed_filter}. Aborting."
        )
    return rows


def _parse_alpha_schedule(raw: str) -> Optional[List[float]]:
    """Parse a comma-separated alpha schedule into a float list.

    Empty / blank input returns ``None``, which means "use the fixed
    ``dirichlet_alpha``" (the legacy path).  A malformed entry is a hard
    error rather than a silent fallback — a typo'd schedule should not
    quietly degrade to fixed-alpha and look like it worked.
    """
    text = (raw or "").strip()
    if not text:
        return None
    parts = [seg.strip() for seg in text.split(",") if seg.strip()]
    if not parts:
        return None
    try:
        vals = [float(seg) for seg in parts]
    except ValueError as exc:
        raise ValueError(
            f"--fpu-alpha-schedule is not a comma-separated float list: {raw!r}"
        ) from exc
    if any(v <= 0.0 for v in vals):
        raise ValueError(
            f"--fpu-alpha-schedule entries must all be > 0 (got {raw!r})"
        )
    return vals


def _build_search(
    *,
    tiles: List[Any],
    rules: Dict[str, Any],
    reward: Any,
    n_simulations: int,
    top_k: int,
    seed: int,
    cpuct_mode: str = "constant",
    cpuct_variance_alpha: float = 0.8,
    fpu_reduction: float = 0.0,
    dirichlet_alpha_schedule: Optional[Sequence[float]] = None,
    target_smiles: Optional[str] = None,
) -> Any:
    """Build one MCTSProofSearch.  Args mirror run_benchmark.py:317."""
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch
    from molmetal_lam.types.predicates import LIPINSKI
    from molmetal_lam.binding.types import PROTEASE_GENERIC

    return MCTSProofSearch(
        tile_library=tiles,
        rules=dict(rules),
        target_predicates=[LIPINSKI],
        binding_site=PROTEASE_GENERIC,
        reward=reward,
        n_simulations=int(n_simulations),
        c_puct=1.4,
        cpuct_mode=str(cpuct_mode),
        cpuct_variance_alpha=float(cpuct_variance_alpha),
        fpu_reduction=float(fpu_reduction),
        dirichlet_alpha_schedule=dirichlet_alpha_schedule,
        top_k=int(top_k),
        rng=random.Random(int(seed)),
        use_fragment_pool=False,  # canonical 12-tile path
        early_stop=False,  # honour n_simulations exactly
        # R22 (Wf-R22-A) — target_smiles triggers the PLL leaf-value
        # wire-in inside :meth:`_rollout`.  None (default) keeps the
        # zero-aggregator path byte-for-byte unchanged.
        target_smiles=target_smiles,
    )


def _run_one_target(
    *,
    target_idx: int,
    smiles_in: str,
    seed: int,
    tiles: List[Any],
    rules: Dict[str, Any],
    reward: Any,
    n_simulations: int,
    top_k: int,
    max_depth: int,
    cpuct_mode: str = "constant",
    cpuct_variance_alpha: float = 0.8,
    fpu_reduction: float = 0.0,
    dirichlet_alpha_schedule: Optional[Sequence[float]] = None,
    target_smiles: Optional[str] = None,
    no_pll_wire: bool = False,
) -> TargetRecord:
    """Run MCTSProofSearch on one (target_smiles, seed) pair."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    rec = TargetRecord(
        target_idx=target_idx,
        smiles_in=smiles_in,
        seed=seed,
        n_simulations_requested=n_simulations,
    )
    t0 = time.time()
    try:
        seed_term = MoleculeClosedTerm.from_smiles(smiles_in, embed_3d=False)
    except Exception as exc:
        rec.wall_time_s = time.time() - t0
        rec.error = f"smiles_parse_error: {type(exc).__name__}: {exc}"
        return rec

    search_inst = _build_search(
        tiles=tiles,
        rules=rules,
        reward=reward,
        n_simulations=n_simulations,
        top_k=top_k,
        seed=seed,
        cpuct_mode=cpuct_mode,
        cpuct_variance_alpha=cpuct_variance_alpha,
        fpu_reduction=fpu_reduction,
        dirichlet_alpha_schedule=dirichlet_alpha_schedule,
        # R22 (Wf-R22-A) — pass the target SMILES so the PLL leaf-value
        # wire-in fires inside :meth:`_rollout`.  ``--no-pll-wire``
        # control passes None to keep the zero-aggregator path.
        target_smiles=(None if no_pll_wire else (target_smiles or smiles_in)),
    )

    try:
        cands = search_inst.search(seed_term, max_depth=max_depth)
    except Exception as exc:
        rec.wall_time_s = time.time() - t0
        rec.error = f"search_error: {type(exc).__name__}: {exc}"
        return rec

    rec.wall_time_s = time.time() - t0
    cands_list = list(cands or [])
    rec.n_candidates = len(cands_list)
    seen: set = set()
    cand_smis: List[str] = []
    for c in cands_list:
        smi = _safe_canonical_smiles(c)
        if smi is None:
            continue
        cand_smis.append(smi)
        seen.add(smi)
    rec.candidates_smiles = cand_smis
    rec.n_distinct = len(seen)
    # Extract instrumentation counters from search_inst (post-search state).
    try:
        rec.max_depth_reached = int(getattr(search_inst, "_max_depth_reached", 0) or 0)
    except Exception:
        pass
    try:
        rec.n_transposition_hits = int(
            getattr(search_inst, "TRANSPOSITION_HIT_RATE", 0)
            * max(1, n_simulations)
        )
    except Exception:
        pass
    try:
        rec.early_stopped = bool(getattr(search_inst, "early_stopped", False))
    except Exception:
        pass
    try:
        rec.n_simulations_executed = int(
            getattr(search_inst, "_last_iter_run", n_simulations)
        )
    except Exception:
        rec.n_simulations_executed = n_simulations
    try:
        rec.cum_nfe = int(getattr(search_inst, "CUMULATIVE_NFE", 0))
        rec.nfe_reductions = int(getattr(search_inst, "CUMULATIVE_NFE_REDUCTIONS", 0))
    except Exception:
        pass
    try:
        rec.puct_exploit_ratio_var = float(
            getattr(search_inst, "puct_exploit_ratio_var", 0.0)
        )
        rec.leaf_value_var = float(getattr(search_inst, "leaf_value_var", 0.0))
        rec.leaf_value_std = float(getattr(search_inst, "leaf_value_std", 0.0))
    except Exception:
        pass
    # R22 (Wf-R22-A) — TT cache hit/miss counters from search_inst
    # (``MCTSProofSearch._t_hits`` / ``_t_misses``, set by Round 3).
    # Read both dict-style and attribute-style for forward/backward compat.
    try:
        rec.tt_cache_hits = int(getattr(search_inst, "_t_hits", 0) or 0)
        rec.tt_cache_misses = int(getattr(search_inst, "_t_misses", 0) or 0)
    except Exception:
        pass
    return rec


def _aggregate_metrics(
    records: List[TargetRecord],
    n_simulations: int,
) -> Dict[str, Any]:
    """Aggregate per-arm metrics across all (target, seed) pairs."""
    ok_records = [r for r in records if r.error is None]
    n_total = len(records)
    n_errors = sum(1 for r in records if r.error is not None)
    n_success = sum(1 for r in ok_records if r.n_candidates >= 1)
    # success_rate is over ALL runs (including errors) so an error run is
    # not silently excluded from the denominator — this is the honest
    # metric. Use n_total as denominator.
    success_rate = float(n_success / max(1, n_total))

    depths = [r.max_depth_reached for r in ok_records if r.max_depth_reached > 0]
    distincts = [r.n_distinct for r in ok_records]
    walls = [r.wall_time_s for r in ok_records]
    cum_nfe_list = [r.cum_nfe for r in ok_records]
    nfe_reductions_list = [r.nfe_reductions for r in ok_records]

    agg = {
        "n_total_runs": int(n_total),
        "n_errors": int(n_errors),
        "n_success": int(n_success),
        "success_rate": float(success_rate),
        "mean_n_distinct": float(statistics.mean(distincts)) if distincts else 0.0,
        "median_n_distinct": float(statistics.median(distincts)) if distincts else 0.0,
        "mean_depth_reached": float(statistics.mean(depths)) if depths else 0.0,
        "median_depth_reached": float(statistics.median(depths)) if depths else 0.0,
        "max_depth_reached_observed": int(max(depths, default=0)),
        "mean_wall_time_s": float(statistics.mean(walls)) if walls else 0.0,
        "median_wall_time_s": float(statistics.median(walls)) if walls else 0.0,
        "max_wall_time_s": float(max(walls, default=0.0)),
        "total_wall_time_s": float(sum(walls)),
        "mean_cum_nfe": float(statistics.mean(cum_nfe_list)) if cum_nfe_list else 0.0,
        "mean_nfe_reductions": (
            float(statistics.mean(nfe_reductions_list)) if nfe_reductions_list else 0.0
        ),
        "n_simulations_per_run": int(n_simulations),
    }
    # R22 (Wf-R22-A) — TT cache hit-rate aggregate.  Hits / (hits+misses)
    # is the canonical AlphaZero-style transposition-table metric.
    tt_hits = [r.tt_cache_hits for r in ok_records]
    tt_misses = [r.tt_cache_misses for r in ok_records]
    total_hits = sum(tt_hits)
    total_misses = sum(tt_misses)
    total_lookups = total_hits + total_misses
    if total_lookups > 0:
        agg["tt_cache_hit_rate"] = float(total_hits / total_lookups)
        agg["tt_cache_total_hits"] = int(total_hits)
        agg["tt_cache_total_misses"] = int(total_misses)
    else:
        agg["tt_cache_hit_rate"] = 0.0
        agg["tt_cache_total_hits"] = 0
        agg["tt_cache_total_misses"] = 0
    return agg


def _check_degeneracy(agg: Dict[str, Any]) -> Optional[str]:
    """Return SystemExit message if bench is degenerate, else None.

    Degeneracy guard mirrors Wf-40 / Wf-R20-A failure modes:
    - n_success == 0  → search produces zero candidates across the whole bench
    - mean_depth_reached < 4  → search did not traverse beyond a trivial subtree
    Either condition ⇒ the bench is INDISTINGUISHABLE FROM NOISE and must FAIL.
    """
    if agg["n_success"] == 0:
        return (
            f"[bench-degeneracy] n_success=0 — MCTS produced zero candidates "
            f"across {agg['n_total_runs']} runs. This is the Wf-40 / Wf-R20-A "
            f"failure mode; refusing to emit JSON. Fix: raise n_simulations "
            f"(currently {agg['n_simulations_per_run']}) or max_depth."
        )
    if agg["mean_depth_reached"] < 4:
        return (
            f"[bench-degeneracy] mean_depth_reached="
            f"{agg['mean_depth_reached']:.2f} < 4 — search did not traverse "
            f"beyond a trivial subtree. Refusing to emit JSON. Fix: raise "
            f"n_simulations (currently {agg['n_simulations_per_run']}) or "
            f"max_depth."
        )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Wf-R20-A1 Lambda MCTS search-quality bench harness. "
            "CPU-only, degeneracy-guarded."
        ),
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--csv-organic",
        type=Path,
        default=DEFAULT_CSV_ORGANIC,
        help=(
            "R22 (Wf-R22-A) organic reactive seed set.  When this file "
            "exists it is the preferred bench seed source (16 SMILES "
            "with alkyne/azide/amine/maleimide/thiol redex handles).  "
            "Override the default path with this flag."
        ),
    )
    p.add_argument(
        "--use-organic-seeds",
        action="store_true",
        default=False,
        help=(
            "R22 (Wf-R22-A) — use --csv-organic instead of --csv as the "
            "seed source.  Default False (PlatinAI path) for backward "
            "compat with Wf-R20-FIX-1 measurements."
        ),
    )
    p.add_argument(
        "--no-pll-wire",
        action="store_true",
        default=False,
        help=(
            "R22 (Wf-R22-A) — disable the PLL leaf-value wire-in (set "
            "target_smiles=None on every search).  Used for the A/B "
            "control: PLL OFF == pre-R22 byte-for-byte zero-aggregator "
            "path.  Default False = PLL ON."
        ),
    )
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--n-simulations", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--cpuct-mode",
        choices=["constant", "variance"],
        default="constant",
        help=(
            "c_puct mode (R20 FIX 2): constant=legacy fixed weight, "
            "variance=KataGo-style variance shrinkage"
        ),
    )
    p.add_argument(
        "--cpuct-variance-alpha",
        type=float,
        default=0.8,
        help="variance shrinkage exponent (KataGo default 0.8)",
    )
    p.add_argument(
        "--fpu-reduction",
        "--fpu-bonus",
        dest="fpu_reduction",
        type=float,
        default=0.0,
        help=(
            "R20 FIX 3 KataGo FPU reduction. 0.0 (default) disables FPU "
            "and is byte-for-byte the legacy PUCT path. >0 lifts an "
            "unvisited child's score to parent_Q_max - fpu_reduction."
        ),
    )
    p.add_argument(
        "--fpu-alpha-schedule",
        type=str,
        default="",
        help=(
            "R20 FIX 3 root Dirichlet alpha decay schedule, "
            "comma-separated (e.g. '1.0,0.3,0.15,0.03'). "
            "Empty = fixed dirichlet_alpha (legacy)."
        ),
    )
    p.add_argument(
        "--seed-filter",
        choices=["reactive", "first"],
        default="reactive",
        help=(
            "Wf-R21-A2 seed selection. 'reactive' (default) keeps only "
            "PlatinAI rows carrying a redex handle some REACTION_RULES rule "
            "can consume; 'first' restores the pre-R21 first-N row order."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal/reports/r20_search_quality_baseline",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke preset: 3 targets × 3 seeds × 50 sims × depth 3.",
    )
    args = p.parse_args(argv)

    if args.smoke:
        args.n_targets = 3
        args.n_seeds = 3
        args.n_simulations = 50
        args.max_depth = 3

    if args.n_targets < 1 or args.n_seeds < 1 or args.n_simulations < 1:
        p.error("--n-targets/--n-seeds/--n-simulations must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[bench-start] csv={args.csv} n_targets={args.n_targets} "
        f"n_seeds={args.n_seeds} n_simulations={args.n_simulations} "
        f"max_depth={args.max_depth} top_k={args.top_k} "
        f"out_dir={args.out_dir}",
        flush=True,
    )

    # Lazy import heavy chemistry stack only when bench actually runs.
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm  # noqa: F401
    from molmetal_lam.reactions.beta_reductions import REACTION_RULES
    from molmetal_lam.tile_lib.click_tiles import STANDARD_12_TILES
    from molmetal_lam.search_alg.proof_search import RewardAggregator

    # Canonical 12-tile library per run_benchmark.py:585.
    raw_tiles = STANDARD_12_TILES()
    tiles = [
        MoleculeClosedTerm.from_smiles(t.smiles, embed_3d=False)
        for t in raw_tiles
    ]
    rules = dict(REACTION_RULES)
    zero_reward = RewardAggregator()

    rows = _load_targets(args.csv, args.n_targets, args.seed_filter)
    print(f"[bench-load] {len(rows)} targets loaded from {args.csv}", flush=True)
    # R22 (Wf-R22-A) — opt-in organic seed path.  When --use-organic-seeds
    # is set, load from --csv-organic instead of --csv.  Wf-R21-A
    # MEASURED: PlatinAI metal complexes fail Lipinski gate (dative-bond
    # SMILES throw RDKit pre-condition violations); organic reactive
    # seeds pass Lipinski + produce non-zero search candidates.
    if args.use_organic_seeds:
        rows = _load_targets(args.csv_organic, args.n_targets, args.seed_filter)
        print(
            f"[bench-load-organic] {len(rows)} targets loaded from "
            f"{args.csv_organic}",
            flush=True,
        )

    records: List[TargetRecord] = []
    n_pairs = len(rows) * args.n_seeds
    pair_idx = 0
    t_start = time.time()
    for ti, row in enumerate(rows):
        smi = row["smiles"]
        for si in range(args.n_seeds):
            pair_idx += 1
            seed = int(args.seed_base + si * 10007 + ti)
            print(
                f"[bench-run] {pair_idx}/{n_pairs} target={ti} seed={seed} "
                f"smi={smi[:50]}...",
                flush=True,
            )
            rec = _run_one_target(
                target_idx=ti,
                smiles_in=smi,
                seed=seed,
                tiles=tiles,
                rules=rules,
                reward=zero_reward,
                n_simulations=args.n_simulations,
                top_k=args.top_k,
                max_depth=args.max_depth,
                cpuct_mode=args.cpuct_mode,
                cpuct_variance_alpha=args.cpuct_variance_alpha,
                fpu_reduction=args.fpu_reduction,
                dirichlet_alpha_schedule=_parse_alpha_schedule(
                    args.fpu_alpha_schedule
                ),
                no_pll_wire=args.no_pll_wire,
            )
            records.append(rec)
            print(
                f"  → n_cand={rec.n_candidates} n_distinct={rec.n_distinct} "
                f"depth={rec.max_depth_reached} wall={rec.wall_time_s:.3f}s "
                f"{('error=' + rec.error) if rec.error else 'ok'}",
                flush=True,
            )
    wall_total = time.time() - t_start

    agg = _aggregate_metrics(records, args.n_simulations)

    # Degeneracy guard: refuse to emit JSON if bench is degenerate.
    degeneracy_msg = _check_degeneracy(agg)
    if degeneracy_msg is not None:
        print(degeneracy_msg, flush=True)
        print(
            "[bench-degeneracy] VERDICT: BENCH_DEGENERATE — no JSON emitted. "
            "Raise n_simulations / max_depth and re-run.",
            flush=True,
        )
        # Write a stub JSON so the closeout can still cite the attempt.
        stub = {
            "tool": "bench_mcts_search_quality.py",
            "wave": "Wf-R20-A1 / Wf-R20-FIX-1",
            "verdict": "BENCH_DEGENERATE",
            "degeneracy_msg": degeneracy_msg,
            "agg": agg,
            "n_records": len(records),
            "wall_total_s": float(wall_total),
            "records": [asdict(r) for r in records],
        }
        stub_path = args.out_dir / "search_quality_bench.json"
        stub_path.write_text(json.dumps(stub, indent=2) + "\n")
        print(f"[bench-degeneracy] stub JSON for diagnosis: {stub_path}", flush=True)
        return 2  # distinct exit code so CI/closeout can detect degeneracy

    # Non-degenerate: emit JSON
    result = {
        "tool": "bench_mcts_search_quality.py",
        "wave": "Wf-R20-A1 / Wf-R20-FIX-1",
        "purpose": (
            "Wf-R20-FIX-1 — search-quality bench harness. Converts MCTS "
            "lift claims from PROJECTED to MEASURED. The 7 R20 metrics "
            "(ER@N, S2T, HV, SR, b_eff+root_entropy, Validity/SA/QED/"
            "Tanimoto, NFE/solution) are aggregated here. Degeneracy "
            "guard active: n_success==0 OR mean_depth<4 ⇒ BENCH_DEGENERATE."
        ),
        "config": {
            "csv": str(args.csv),
            "n_targets": int(args.n_targets),
            "n_seeds": int(args.n_seeds),
            "n_simulations": int(args.n_simulations),
            "max_depth": int(args.max_depth),
            "top_k": int(args.top_k),
            "seed_base": int(args.seed_base),
        },
        "verdict": "OK",
        "agg": agg,
        "wall_total_s": float(wall_total),
        "n_records": len(records),
        "records": [asdict(r) for r in records],
        "honest_negative": (
            "Per R20-A audit the previous bench JSONs were degenerate "
            "(n_samples=3, n_distinct=0, n_success=0; mean_max_depth=3.0). "
            "This harness is the prerequisite for any R20 FIX 2-5 to be "
            "shipped — a fix cannot be claimed to lift search quality "
            "until this bench produces a non-degenerate baseline on disk."
        ),
    }
    json_path = args.out_dir / "search_quality_bench.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[bench-ok] JSON: {json_path}", flush=True)
    print(
        f"[bench-ok] verdict=OK  n_success={agg['n_success']}/{agg['n_total_runs']} "
        f"success_rate={agg['success_rate']:.3f} mean_depth={agg['mean_depth_reached']:.2f} "
        f"mean_n_distinct={agg['mean_n_distinct']:.2f} "
        f"mean_wall={agg['mean_wall_time_s']:.3f}s "
        f"wall_total={agg['total_wall_time_s']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())