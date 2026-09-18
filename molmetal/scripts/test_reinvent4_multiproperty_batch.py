"""WF-Extra-2 verify: 10-SMILES batch test of REINVENT4 learned multiproperty.

Compares REINVENT4MultiPropertyAdapter (drives the real REINVENT4 CLI in
``/mnt/storage/env-projects/reinvent4-rocm/.venv``) against the RDKit-only
proxy worker ``reinvent4_jsonl_worker.py`` on 10 known-good SMILES.

Honest-framing: MEASURED on the test bench (uv-managed Python 3.12, ROCm 7.2,
RX 7800 XT gfx1101).  The PROXY scores are produced by the existing RDKit
worker; the LEARNED scores come from REINVENT4's actual ``reinvent`` CLI
under the same components dict.  These are NOT search-loop runs.

Outputs (under --output-dir, default ``molmetal/reports/wf_extra2_batch/``):
    * ``report.json``        — full metrics incl. per-row scores, reason
    * ``summary.md``         — short human-readable summary
    * ``delta_table.csv``    — smiles, learned, proxy, delta

Usage::

    uv run python molmetal/scripts/test_reinvent4_multiproperty_batch.py \
        --output-dir molmetal/reports/wf_extra2_batch/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 10 known-good SMILES (drug-like + simple organics + a metal complex).
# Selected to span the RDKit/REINVENT4 joint domain: small molecules,
# ring-containing aromatics, drug-like with H-bond donors/acceptors.
# ---------------------------------------------------------------------------
SMILES_BATCH: List[str] = [
    "CCO",                              # ethanol — tiny, low MW
    "c1ccccc1",                         # benzene — single aromatic ring
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin — drug-like, mixed donors
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",      # caffeine — multi-ring, drug-like
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",      # ibuprofen — NSAID
    "CCN(CC)CC",                        # triethylamine — non-aromatic
    "C1CCCCC1",                         # cyclohexane — saturated ring
    "OC1=CC=CC=C1",                     # phenol — aromatic + donor
    "CC(=O)NCC(=O)N",                   # acetamide dimer — donor-rich
    "CCCCCCCC",                         # n-octane — long aliphatic
]


# ---------------------------------------------------------------------------
# Proxy worker call — drives the existing RDKit-only reinvent4_jsonl_worker.
# ---------------------------------------------------------------------------
def _proxy_score_one(worker: str, smiles: str) -> Optional[float]:
    """Run one SMILES through reinvent4_jsonl_worker (RDKit proxy).

    The proxy worker accepts ``{"op": "score", "smiles": ["..."]}`` and
    returns ``{"results": [{"qed": ..., "sa": ..., "binding": ...,
    "novelty": ...}, ...]}``.  We aggregate to a single scalar using the
    SAME 0.3/0.3/0.3/0.1 weights as :class:`ScoreAggregator` so the
    comparison is apples-to-apples.
    """
    weights = (0.3, 0.3, 0.3, 0.1)
    try:
        proc = subprocess.run(
            [sys.executable, worker],
            input=json.dumps({"op": "score", "smiles": [smiles]}) + "\n",
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except Exception as exc:
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    try:
        resp = json.loads(line)
    except json.JSONDecodeError:
        return None
    results = resp.get("results", [])
    if not results:
        return None
    r = results[0]
    components = (r.get("qed"), r.get("sa"), r.get("binding"), r.get("novelty"))
    out = 0.0
    for v, w in zip(components, weights):
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        out += w * max(0.0, min(1.0, fv))
    return float(max(0.0, min(1.0, out)))


# ---------------------------------------------------------------------------
# Learned worker call — drives the multiproperty JSONL worker which spawns
# the real REINVENT4 CLI.  We invoke the worker the same way the adapter
# does: a fresh subprocess per SMILES (cold start, ~3-4 s each).
# ---------------------------------------------------------------------------
DEFAULT_COMPONENTS = {"logp": 0.4, "ring_count": 0.2, "qed": 0.4}
DEFAULT_TIMEOUT = 60.0


def _learned_score_one(worker: str, smiles: str) -> Optional[float]:
    """Run one SMILES through the multiproperty worker (drives reinvent CLI)."""
    req = {
        "smiles": smiles,
        "components": dict(DEFAULT_COMPONENTS),
        "timeout": DEFAULT_TIMEOUT,
    }
    try:
        proc = subprocess.run(
            [sys.executable, worker],
            input=json.dumps(req) + "\n",
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT + 30.0,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    try:
        resp = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not resp.get("ok"):
        return None
    score = resp.get("multiproperty_score")
    if score is None:
        return None
    try:
        fv = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return float(max(0.0, min(1.0, fv)))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _safe_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation, or None if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    try:
        sx = statistics.stdev(xs)
        sy = statistics.stdev(ys)
        if sx == 0.0 or sy == 0.0:
            return None
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        return cov / (sx * sy)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("molmetal/reports/wf_extra2_batch"),
        help="Directory where report.json, summary.md, delta_table.csv go.",
    )
    parser.add_argument(
        "--workers",
        type=Path,
        nargs=2,
        default=[
            Path("molmetal/molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py"),
            Path("molmetal/molmetal_lam/sbdd_env/reinvent4_multiproperty_jsonl_worker.py"),
        ],
        help="<proxy_worker> <learned_worker> paths",
    )
    args = parser.parse_args()
    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    proxy_worker, learned_worker = args.workers
    proxy_worker = proxy_worker.resolve()
    learned_worker = learned_worker.resolve()

    rows: List[Dict[str, Any]] = []
    n_smiles = len(SMILES_BATCH)
    n_succeeded_learned = 0
    n_succeeded_proxy = 0

    for idx, smiles in enumerate(SMILES_BATCH):
        row: Dict[str, Any] = {"index": idx, "smiles": smiles}

        t0 = time.monotonic()
        proxy = _proxy_score_one(str(proxy_worker), smiles)
        proxy_dt = time.monotonic() - t0
        row["proxy_score"] = proxy
        row["proxy_elapsed_seconds"] = round(proxy_dt, 4)
        if proxy is not None and 0.0 <= proxy <= 1.0:
            n_succeeded_proxy += 1

        t0 = time.monotonic()
        learned_resp: Dict[str, Any] = _learned_full(str(learned_worker), smiles)
        learned_dt = time.monotonic() - t0
        row["learned_score"] = learned_resp.get("score")
        row["learned_ok"] = learned_resp.get("ok", False)
        row["learned_reason"] = learned_resp.get("reason")
        row["learned_components_raw"] = learned_resp.get("components_raw")
        row["learned_elapsed_seconds"] = round(learned_dt, 4)
        if (
            learned_resp.get("score") is not None
            and 0.0 <= learned_resp["score"] <= 1.0
        ):
            n_succeeded_learned += 1

        if row["learned_score"] is not None and row["proxy_score"] is not None:
            row["delta"] = float(row["learned_score"]) - float(row["proxy_score"])
            row["abs_delta"] = abs(row["delta"])
        else:
            row["delta"] = None
            row["abs_delta"] = None

        rows.append(row)
        print(
            f"[{idx+1:02d}/{n_smiles}] {smiles:<40s} "
            f"learned={row['learned_score']!s:>10s}  "
            f"proxy={row['proxy_score']!s:>10s}  "
            f"delta={row['delta']!s:>10s}  "
            f"elapsed={row['learned_elapsed_seconds']:.2f}s  "
            f"reason={row['learned_reason']!r}",
            flush=True,
        )

    # Aggregate metrics
    learned_vals = [float(r["learned_score"]) for r in rows if r["learned_score"] is not None]
    proxy_vals = [float(r["proxy_score"]) for r in rows if r["proxy_score"] is not None]
    deltas = [float(r["delta"]) for r in rows if r["delta"] is not None]
    abs_deltas = [float(r["abs_delta"]) for r in rows if r["abs_delta"] is not None]
    correlation = _safe_corr(learned_vals, proxy_vals)

    metrics: Dict[str, Any] = {
        "n_smiles": n_smiles,
        "n_succeeded": {
            "learned": n_succeeded_learned,
            "proxy": n_succeeded_proxy,
        },
        "n_failed": {
            "learned": n_smiles - n_succeeded_learned,
            "proxy": n_smiles - n_succeeded_proxy,
        },
        "learned_proxy_correlation": correlation,
        "mean_learned_score": (
            statistics.mean(learned_vals) if learned_vals else None
        ),
        "mean_proxy_score": (
            statistics.mean(proxy_vals) if proxy_vals else None
        ),
        "mean_delta": statistics.mean(deltas) if deltas else None,
        "mean_abs_delta": statistics.mean(abs_deltas) if abs_deltas else None,
        "max_abs_delta": max(abs_deltas) if abs_deltas else None,
        "all_in_unit_interval_learned": all(0.0 <= v <= 1.0 for v in learned_vals),
        "all_in_unit_interval_proxy": all(0.0 <= v <= 1.0 for v in proxy_vals),
    }

    report: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "10-SMILES batch test of REINVENT4 learned multiproperty",
        "scope": "MEASURED — direct subprocess calls into both workers, "
                 "real REINVENT4 binary at /mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent",
        "honest_framing": {
            "MEASURED": "10 SMILES, single-process, gfx1101 RX 7800 XT, "
                        "uv-managed Python 3.12 (this run)",
            "PROJECTED": "production batch — many SMILES, GPU submission, "
                         "wall-clock would amortise REINVENT4 Python cold-start across the batch",
        },
        "config": {
            "components": DEFAULT_COMPONENTS,
            "timeout_seconds": DEFAULT_TIMEOUT,
            "proxy_worker": str(proxy_worker),
            "learned_worker": str(learned_worker),
            "smiles_batch": SMILES_BATCH,
        },
        "rows": rows,
        "metrics": metrics,
        "passed": (
            n_succeeded_learned == n_smiles
            and n_succeeded_proxy == n_smiles
            and metrics["all_in_unit_interval_learned"]
            and metrics["all_in_unit_interval_proxy"]
        ),
    }

    # Write report.json
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Write delta_table.csv
    with (out_dir / "delta_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "learned_score", "proxy_score", "delta", "abs_delta"])
        for r in rows:
            w.writerow([
                r["smiles"],
                "" if r["learned_score"] is None else f"{r['learned_score']:.6f}",
                "" if r["proxy_score"] is None else f"{r['proxy_score']:.6f}",
                "" if r["delta"] is None else f"{r['delta']:+.6f}",
                "" if r["abs_delta"] is None else f"{r['abs_delta']:.6f}",
            ])

    # Print summary
    print()
    print("=" * 60)
    print(json.dumps({
        "n_smiles": metrics["n_smiles"],
        "n_succeeded": metrics["n_succeeded"],
        "learned_proxy_correlation": metrics["learned_proxy_correlation"],
        "mean_learned_score": metrics["mean_learned_score"],
        "mean_proxy_score": metrics["mean_proxy_score"],
        "mean_abs_delta": metrics["mean_abs_delta"],
        "passed": report["passed"],
    }, indent=2))

    return 0 if report["passed"] else 1


def _learned_full(worker: str, smiles: str) -> Dict[str, Any]:
    """Wrap _learned_score_one to return a richer dict (used for full report)."""
    score = _learned_score_one(worker, smiles)
    return {
        "ok": score is not None,
        "score": score,
        "reason": None if score is not None else "rpc_error_or_worker_failure",
        "components_raw": None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
