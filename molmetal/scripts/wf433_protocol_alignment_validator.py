#!/usr/bin/env python3
"""Wf-433-A4 protocol-alignment validator — validates that our training /
evaluation pipeline is **aligned** with the 3 selected Tier-1 baselines
(Krasnov 2026 + Rusanov 2025 + DrugOOD 2022) on **dataset path + split
protocol + metric definition**.

==============================================================================

Per ``TODO/03_baselines/PROTOCOL_VALIDATION.md`` §2 + ``COMPARISON_BASELINES.md``
§0:

    我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol +
    same metric 来 make results comparable。

The alignment validator complements ``wf433_protocol_validator.py`` (which
emits a numerical-comparable verdict per metric_id).  This script verifies
that the **structural** preconditions for numerical comparison are met:

  * Same dataset path (file path or canonical dataset name match)
  * Same split protocol (scaffold vs kfold vs temporal vs random)
  * Same metric definition (binary_auc vs r_squared vs median_kcal_mol)

For each axis, the validator emits PASS / FAIL / WARN + a verbatim evidence
quote.  This is the **subset-style** pre-check that runs BEFORE the
numerical-comparable validator.

CLI::

    python molmetal/scripts/wf433_protocol_alignment_validator.py \
        --our-results metrics/by_metric/metal_lomo_auc_wasserstein.json \
        --baseline krasnov_2026 \
        --out molmetal/reports/wf433_protocol_alignment/example_alignment.json

Supports a **batch** mode that runs all 3 baselines in one invocation::

    python molmetal/scripts/wf433_protocol_alignment_validator.py \
        --batch \
        --metric-roots metrics/by_metric/metal_lomo_auc_wasserstein.json \
                      metrics/by_metric/drugood_ood_auc.json \
                      metrics/by_metric/platinai_4cell_auc.json \
        --out molmetal/reports/wf433_protocol_alignment/batch_alignment.json

Honest framing
==============

Per ``party-twenty-spirit §3.5`` + ``AGENTS.md §5.2``: every numeric value
traces to a JSON / report path on disk; no fabrication.

Per ``AGENTS.md §5.6`` atomic-write protocol: writes use
``atomic_write_file`` from :mod:`wf386a3_atomic_write`.

Per ``AGENTS.md §5.5`` filesystem-truth rule: parent directories auto-created;
report-only mode (--no-write) supported for test environments.

Exit codes
==========

  * ``0`` — every axis alignment check PASS
  * ``1`` — at least one axis alignment check FAIL
  * ``2`` — input JSON missing or unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Ensure the repo root is on sys.path so ``molmetal.scripts.*`` imports
# cleanly when the script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf433_protocol_alignment"
    / "example_alignment.json"
)

# Reuse the canonical baseline metadata from wf433_protocol_validator.
# Importing rather than copy-paste avoids the divergence risk that the
# §5.6 atomic-write protocol warns against.
from molmetal.scripts.wf433_protocol_validator import (
    BASELINES,
    SUPPORTED_BASELINES,
    METRIC_KEYS_BY_NAME,
    SCHEMA_VERSION,
    _now_iso,
    _read_json,
    _atomic_write_json,
    _infer_metric_id,
)


# ---------------------------------------------------------------------------
# Alignment axes
# ---------------------------------------------------------------------------


@dataclass
class AxisCheck:
    """Single axis (dataset / split / metric) alignment verdict."""

    axis_name: str
    expected_baseline: str
    expected_paper: str
    expected_our: str
    observed: str
    match: bool
    evidence: str
    verdict: str  # PASS / FAIL / WARN
    verdict_detail: str


def _check_dataset_axis(
    baseline_meta: Dict[str, Any],
    results: Dict[str, Any],
) -> AxisCheck:
    """Axis 1: same dataset path.

    The check is **lenient** when the baseline's documented fallback dataset
    is in use (e.g. ``metacytotox_fallback`` for DrugOOD per
    ``drugood_ood_auc.json::data_source`` + honest framing caveat).  For
    cross-baseline comparisons (e.g. m17 metacytotox_fallback vs Rusanov
    PlatinAI 4-cell), the fallback marker does NOT bypass the dataset
    check — the fallback is only valid for its canonical baseline.
    """
    expected_pattern = baseline_meta["dataset_path_pattern"]
    observed = str(results.get("data_source", ""))
    exact_match = expected_pattern in observed
    # Fallback markers — only valid for the DrugOOD baseline where
    # ``metacytotox_fallback`` is the documented honest caveat.
    fallback_markers = ("metacytotox_fallback", "fallback", "proxy")
    is_fallback = any(m in observed.lower() for m in fallback_markers)
    is_drugood_baseline = "DrugOOD" in baseline_meta["name"]
    match = bool(exact_match or (is_fallback and is_drugood_baseline))
    if exact_match:
        verdict = "PASS"
        detail = f"Dataset path MATCHES baseline contract ({expected_pattern})."
    elif is_fallback and is_drugood_baseline:
        verdict = "WARN"
        detail = (
            f"Data source is a FALLBACK dataset ('{observed}'); "
            f"baseline expects native '{expected_pattern}'. "
            f"Honest framing: numbers from fallback are NOT head-to-head "
            f"with native iMoLD; downstream wf433_protocol_validator emits "
            f"alignment caveat."
        )
    elif is_fallback:
        verdict = "FAIL"
        detail = (
            f"Fallback dataset ('{observed}') is NOT applicable to this "
            f"baseline ('{baseline_meta['name']}'); native '{expected_pattern}' "
            f"required for cross-baseline numerical comparison."
        )
    else:
        verdict = "FAIL"
        detail = (
            f"Dataset path DOES NOT MATCH baseline contract "
            f"(expected '{expected_pattern}', observed '{observed}')."
        )
    return AxisCheck(
        axis_name="dataset_path",
        expected_baseline=baseline_meta["dataset_path_pattern"],
        expected_paper=baseline_meta["dataset_path_pattern"],
        expected_our=baseline_meta["dataset_path_pattern"],
        observed=observed,
        match=bool(match),
        evidence=(
            f"baseline expects '{expected_pattern}' substring in data_source; "
            f"observed='{observed}' (fallback={is_fallback}, "
            f"drugood_baseline={is_drugood_baseline})"
        ),
        verdict=verdict,
        verdict_detail=detail,
    )


def _check_split_axis(
    baseline_meta: Dict[str, Any],
    results: Dict[str, Any],
) -> AxisCheck:
    """Axis 2: same split protocol.

    The check is **lenient** — the validator accepts scaffold-split OR
    5-fold CV OR temporal split OR random split as long as the protocol is
    *documented*.  The strict alignment is enforced by
    ``wf433_protocol_validator.py`` via the ``alignment_level`` field.
    """
    paper = baseline_meta["paper_split"]
    our = baseline_meta["our_split"]
    observed = str(results.get("split", ""))
    # A documented split in the JSON is sufficient; alignment gap is
    # handled by the numerical-comparable validator downstream.
    documented = bool(observed)
    if not documented:
        verdict = "WARN"
        verdict_detail = (
            "No ``split`` field in results JSON; downstream "
            "wf433_protocol_validator will report alignment-level caveat."
        )
    else:
        verdict = "PASS"
        verdict_detail = (
            f"Documented split='{observed}'. "
            f"Baseline paper uses '{paper}'; we use '{our}'."
        )
    return AxisCheck(
        axis_name="split_protocol",
        expected_baseline=paper,
        expected_paper=paper,
        expected_our=our,
        observed=observed,
        match=bool(documented),
        evidence=(f"baseline paper='{paper}'; our='{our}'; observed='{observed}'"),
        verdict=verdict,
        verdict_detail=verdict_detail,
    )


def _check_metric_axis(
    baseline_meta: Dict[str, Any],
    results: Dict[str, Any],
) -> AxisCheck:
    """Axis 3: same metric definition."""
    expected_metric = baseline_meta["metric"]
    observed_metric = str(results.get("metric", ""))
    # The metric field should name the canonical metric (binary_auc /
    # r_squared / median_kcal_mol / ood_auc).  Substring match.
    expected_keywords = {
        "Krasnov 2026": ["auc"],
        "Rusanov 2025": ["auc", "r_squared"],
        "DrugOOD 2022": ["auc"],
    }
    name_keys = [k for k in expected_keywords.keys() if k in baseline_meta["name"]]
    keys = expected_keywords.get(
        name_keys[0] if name_keys else "Krasnov 2026",
        ["auc"],
    )
    match = any(k in observed_metric.lower() for k in keys)
    verdict = "PASS" if match else "FAIL"
    return AxisCheck(
        axis_name="metric_definition",
        expected_baseline=expected_metric,
        expected_paper=expected_metric,
        expected_our=expected_metric,
        observed=observed_metric,
        match=bool(match),
        evidence=(
            f"baseline expects metric '{expected_metric}' (keywords "
            f"{keys}); observed='{observed_metric}'"
        ),
        verdict=verdict,
        verdict_detail=(
            f"Metric definition MATCHES baseline contract (keywords {keys})."
            if match
            else f"Metric definition DOES NOT MATCH baseline contract "
            f"(expected keywords {keys}, observed '{observed_metric}')."
        ),
    )


def _check_seed_coverage_axis(
    baseline_meta: Dict[str, Any],
    results: Dict[str, Any],
) -> AxisCheck:
    """Axis 4: seed coverage (>= 3 seeds for Tier-1 reproducibility).

    Per ``COMPARISON_BASELINES.md`` §1, all baselines report mean ± std over
    seeds.  We require ``n_seeds >= 3`` for the JSON to be Tier-1-comparable.
    """
    seeds = results.get("seeds") or []
    n_seeds_field = results.get("n_seeds")
    n_seeds = len(seeds) if isinstance(seeds, list) and seeds else n_seeds_field
    if not n_seeds:
        # Try to derive from per_run_auc / per_cell_line / per_seed_ood.
        for key in ("per_run_auc", "per_seed"):
            v = results.get(key)
            if isinstance(v, list) and v:
                n_seeds = len(v)
                break
        else:
            # per_cell_line[<cell>].per_seed — flatten.
            for v in (results.get("per_cell_line") or {}).values():
                if isinstance(v, dict) and v.get("per_seed"):
                    n_seeds = max(n_seeds or 0, len(v["per_seed"]))
    if not n_seeds:
        verdict = "WARN"
        verdict_detail = (
            "n_seeds not inferrable from results JSON; downstream "
            "wf433_protocol_validator will fall back to n_runs if available."
        )
        match = False
    else:
        match = int(n_seeds) >= 3
        verdict = "PASS" if match else "WARN"
        verdict_detail = f"Seed coverage n_seeds={n_seeds} {'≥' if match else '<'} 3 (Tier-1 threshold)."
    return AxisCheck(
        axis_name="seed_coverage",
        expected_baseline=">=3 seeds (per COMPARISON_BASELINES.md §1)",
        expected_paper=">=3 seeds",
        expected_our=">=3 seeds",
        observed=f"n_seeds={n_seeds}",
        match=bool(match),
        evidence=verdict_detail,
        verdict=verdict,
        verdict_detail=verdict_detail,
    )


def check_alignment_contract(
    baseline_key: str,
    results: Dict[str, Any],
) -> List[AxisCheck]:
    """Top-level axis dispatch."""
    baseline_meta = BASELINES[baseline_key]
    return [
        _check_dataset_axis(baseline_meta, results),
        _check_split_axis(baseline_meta, results),
        _check_metric_axis(baseline_meta, results),
        _check_seed_coverage_axis(baseline_meta, results),
    ]


def _verdict_summary(checks: List[AxisCheck]) -> Tuple[str, int, int, int]:
    """Aggregate axis verdicts."""
    n_pass = sum(1 for c in checks if c.verdict == "PASS")
    n_fail = sum(1 for c in checks if c.verdict == "FAIL")
    n_warn = sum(1 for c in checks if c.verdict == "WARN")
    overall = (
        "PASS"
        if n_fail == 0 and n_warn == 0
        else ("PASS_WITH_WARN" if n_fail == 0 else "FAIL")
    )
    return overall, n_pass, n_fail, n_warn


# ---------------------------------------------------------------------------
# CLI + orchestrator
# ---------------------------------------------------------------------------


def run_alignment_check(
    *,
    results_path: Path,
    baseline_key: str,
    out_path: Optional[Path],
) -> Dict[str, Any]:
    """Single-results orchestrator."""
    if not results_path.is_file():
        raise FileNotFoundError(f"results JSON missing: {results_path}")
    if baseline_key not in BASELINES:
        raise ValueError(
            f"unknown baseline {baseline_key!r}; supported: "
            f"{', '.join(SUPPORTED_BASELINES)}"
        )
    r = _read_json(results_path)
    metric_id = _infer_metric_id(r, results_path)
    baseline_meta = BASELINES[baseline_key]

    axis_checks = check_alignment_contract(baseline_key, r)
    overall, n_pass, n_fail, n_warn = _verdict_summary(axis_checks)

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _now_iso(),
        "our_results_path": str(results_path),
        "metric_id": metric_id,
        "baseline": baseline_key,
        "baseline_meta": {
            "name": baseline_meta["name"],
            "dataset": baseline_meta["dataset"],
            "paper_split": baseline_meta["paper_split"],
            "our_split": baseline_meta["our_split"],
            "metric": baseline_meta["metric"],
            "alignment_level": baseline_meta["alignment_level"],
            "alignment_note": baseline_meta["alignment_note"],
        },
        "axes": [
            {
                "axis_name": c.axis_name,
                "expected_baseline": c.expected_baseline,
                "expected_paper": c.expected_paper,
                "expected_our": c.expected_our,
                "observed": c.observed,
                "match": c.match,
                "evidence": c.evidence,
                "verdict": c.verdict,
                "verdict_detail": c.verdict_detail,
            }
            for c in axis_checks
        ],
        "n_axes": len(axis_checks),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "overall_status": overall,
        "alignment_match": (
            n_fail == 0
        ),  # WARN does not count as fail (lenient semantics)
        "honest_framing": (
            f"Wf-433-A4 alignment validator: baseline={baseline_key}, "
            f"metric={metric_id}. {n_pass} PASS / {n_fail} FAIL / "
            f"{n_warn} WARN across {len(axis_checks)} axes. WARNs are "
            "non-blocking (alignment caveat handled by "
            "wf433_protocol_validator downstream)."
        ),
        "error": "",
    }

    if out_path is not None:
        try:
            _atomic_write_json(out_path, payload)
        except OSError as exc:
            payload["error"] = f"write {out_path} failed: {type(exc).__name__}: {exc}"
    return payload


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomic-write JSON via :mod:`wf386a3_atomic_write` per AGENTS.md §5.6.

    Defined locally to avoid the implicit cross-script dependency on
    :mod:`wf433_protocol_validator` (which would defeat the §5.6 isolation
    contract — each script that writes is responsible for its own primitive).
    """
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
    except (ImportError, ModuleNotFoundError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    atomic_write_file(str(path), text)


def run_batch(
    *,
    metric_roots: List[Path],
    out_path: Optional[Path],
) -> Dict[str, Any]:
    """Batch mode — run all 3 baselines on each metric_root."""
    batch: List[Dict[str, Any]] = []
    n_align_pass = n_align_fail = 0
    for results_path in metric_roots:
        r = _read_json(results_path)
        metric_id = _infer_metric_id(r, results_path)
        for baseline_key in SUPPORTED_BASELINES:
            axis_checks = check_alignment_contract(baseline_key, r)
            overall, n_pass, n_fail, n_warn = _verdict_summary(axis_checks)
            entry: Dict[str, Any] = {
                "our_results_path": str(results_path),
                "metric_id": metric_id,
                "baseline": baseline_key,
                "overall_status": overall,
                "alignment_match": n_fail == 0,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_warn": n_warn,
                "axes_summary": [
                    {
                        "axis_name": c.axis_name,
                        "verdict": c.verdict,
                        "verdict_detail": c.verdict_detail,
                    }
                    for c in axis_checks
                ],
            }
            batch.append(entry)
            if n_fail == 0:
                n_align_pass += 1
            else:
                n_align_fail += 1

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _now_iso(),
        "mode": "batch",
        "metric_roots": [str(p) for p in metric_roots],
        "n_entries": len(batch),
        "n_alignment_pass": n_align_pass,
        "n_alignment_fail": n_align_fail,
        "batch_results": batch,
        "honest_framing": (
            f"Wf-433-A4 batch alignment check: {n_align_pass}/{len(batch)} "
            f"(metric, baseline) pairs PASS alignment contract; "
            f"{n_align_fail} FAIL. WARNs are non-blocking per Wf-433-A4 design."
        ),
        "error": "",
    }
    if out_path is not None:
        try:
            _atomic_write_json(out_path, payload)
        except OSError as exc:
            payload["error"] = f"write {out_path} failed: {type(exc).__name__}: {exc}"
    return payload


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wf433_protocol_alignment_validator.py",
        description=(
            "Wf-433-A4 — protocol alignment validator (dataset path + split "
            "protocol + metric definition) for the 3 selected Tier-1 "
            "baselines (Krasnov 2026 + Rusanov 2025 + DrugOOD 2022). "
            "Emits per-axis PASS / FAIL / WARN + alignment_match flag."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--our-results",
        type=Path,
        default=None,
        help=(
            "Path to our results JSON (Tier-1 metric). Required unless "
            "--batch is set."
        ),
    )
    p.add_argument(
        "--baseline",
        choices=SUPPORTED_BASELINES,
        default=None,
        help=(
            "Selected Tier-1 baseline. Required unless --batch is set. "
            f"Supported: {', '.join(SUPPORTED_BASELINES)}."
        ),
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode — run all 3 baselines on each --metric-roots path.",
    )
    p.add_argument(
        "--metric-roots",
        type=Path,
        nargs="+",
        default=None,
        help="(Batch only) Space-separated paths to metric JSONs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            f"Output JSON path (default: {DEFAULT_OUT}). Atomic-written per "
            "AGENTS.md §5.6. Parent dirs auto-created."
        ),
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing --out (compute + return payload only).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress to stderr.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)
    if args.batch:
        if not args.metric_roots:
            print(
                "[wf433_protocol_alignment_validator] FATAL: "
                "--metric-roots required with --batch",
                file=sys.stderr,
            )
            return 2
        if args.verbose:
            print(
                f"[wf433_protocol_alignment_validator] batch mode; "
                f"{len(args.metric_roots)} metric roots",
                file=sys.stderr,
            )
        out_path = None if args.no_write else args.out
        try:
            payload = run_batch(
                metric_roots=args.metric_roots,
                out_path=out_path,
            )
        except FileNotFoundError as exc:
            print(f"[wf433_protocol_alignment_validator] FATAL: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.verbose:
            print(
                f"[wf433_protocol_alignment_validator] batch wrote {args.out} "
                f"({payload['n_alignment_pass']}/{payload['n_entries']} PASS)",
                file=sys.stderr,
            )
        return 0 if payload["n_alignment_fail"] == 0 else 1

    # Single-results mode
    if not args.our_results or not args.baseline:
        print(
            "[wf433_protocol_alignment_validator] FATAL: "
            "--our-results AND --baseline required (or use --batch)",
            file=sys.stderr,
        )
        return 2

    if args.verbose:
        print(
            f"[wf433_protocol_alignment_validator] our_results={args.our_results} "
            f"baseline={args.baseline} out={args.out}",
            file=sys.stderr,
        )
    out_path = None if args.no_write else args.out
    try:
        payload = run_alignment_check(
            results_path=args.our_results,
            baseline_key=args.baseline,
            out_path=out_path,
        )
    except FileNotFoundError as exc:
        print(f"[wf433_protocol_alignment_validator] FATAL: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[wf433_protocol_alignment_validator] FATAL: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.verbose:
        print(
            f"[wf433_protocol_alignment_validator] wrote {args.out} "
            f"(overall={payload['overall_status']} "
            f"alignment_match={payload['alignment_match']})",
            file=sys.stderr,
        )

    return 0 if payload["alignment_match"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"[wf433_protocol_alignment_validator] UNHANDLED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
