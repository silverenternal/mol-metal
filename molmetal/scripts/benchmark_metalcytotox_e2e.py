"""R18 end-to-end pipeline on MetalCytoToxDB 5K real data — per-metal benchmark.

================================================================
Wf-E2E-2 dispatch (2026-09-26, follow-on to Wf-E2E-1 / orchestrator smoke).

Where :file:`r18_pipeline_orchestrator.py` runs the orchestrator on a
flat slice of MetalCytoToxDB SMILES (no per-metal breakdown), this
script runs the **same** :class:`R18Pipeline` once **per metal** and
records per-step × per-metal wall-time + memory + status counts.

Pipeline
--------
The five-step R18 pipeline is the canonical orchestrator:

  step1_closure_theorem   — MLC closure BFS (molmetal_lam.lam_chem.closure)
  step2_se3_sb_ode        — SE(3) SB + Lipman midpoint solver
  step3_multiph_features  — Multi-PH + PLL signature features
  step4_variance_mcts     — Variance-controlled PUCT selector
  step5_hilbert_reward    — Hilbert-ASTSR evolutionary reward

Each sample is run end-to-end through the five steps; the wall-time
of the entire chain is recorded as ``total_wall_time_s``, and each
step contributes its ``wall_time_s`` to the per-step totals.

Per-metal input
---------------
The MetalCytoToxDB CSV (``MetalCytoToxDB.csv``) has columns
``SMILES_Ligands, Counterion, ..., Metal, ...``.  We filter rows by
the ``Metal`` column (the SMILES itself stores only the ligand +
counterion), then cap each metal at ``--n-samples-per-metal`` rows.

Output
------
* ``molmetal/reports/wf_e2e_pipeline/metalcytotox_e2e.json`` — full
  per-metal × per-step report (~200 lines JSON).
* ``molmetal/reports/wf_e2e_pipeline/metalcytotox_e2e_summary.md`` —
  per-metal wall-time + per-step breakdown tables (~100 lines).

Honest framing
--------------
* **CPU-only.**  The orchestrator pins to the CPU backend; no CUDA /
  Triton / CFM retraining is exercised.  The Wf-E2E-1 closure theorem
  pre-computes the productive space once (lazy on first call).
* **Deterministic.**  Every step pins its random seed; running twice
  on the same SMILES yields the same step outputs (modulo the
  ``tracemalloc`` snapshot which is wall-time deterministic on CPU).
* **Real data.**  MetalCytoToxDB.csv is the **real** R17 ship CSV
  (26.8 k rows).  The benchmark never falls back to synthetic / toy
  inputs.

Usage::

    source env.sh && uv run --no-sync python \\
        molmetal/scripts/benchmark_metalcytotox_e2e.py --n-samples-per-metal 500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

# ---------------------------------------------------------------------------
# Paths — keep absolute so the script can be invoked from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default CSV — same path the orchestrator's ``run_metalcytotox_e2e``
# uses (resolved via the ``MOLMETAL_DATA_DIR`` env var, which
# ``env.sh`` already exports to ``$REPO_ROOT/.data/molmetal``).
DEFAULT_CSV = _REPO_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"
DEFAULT_METALS: List[str] = ["Ru", "Ir", "Rh", "Re", "Os"]
DEFAULT_N_SAMPLES_PER_METAL: int = 500

# Output paths
REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_e2e_pipeline"
OUT_JSON = REPORT_DIR / "metalcytotox_e2e.json"
OUT_MD = REPORT_DIR / "metalcytotox_e2e_summary.md"

# Step IDs in the order the orchestrator executes them
STEP_IDS: List[str] = [
    "step1_closure",
    "step2_se3",
    "step3_multiph",
    "step4_mcts",
    "step5_hilbert",
]
STEP_LABELS: Dict[str, str] = {
    "step1_closure": "MLC closure BFS",
    "step2_se3": "SE(3) SB + Lipman ODE",
    "step3_multiph": "Multi-PH + PLL",
    "step4_mcts": "Variance-controlled PUCT",
    "step5_hilbert": "Hilbert-ASTSR reward",
}


# ---------------------------------------------------------------------------
# CSV loader — filter by Metal column (case-sensitive), skip blank rows.
# ---------------------------------------------------------------------------
def _load_metal_samples(
    csv_path: Path,
    metal: str,
    n: int,
) -> List[str]:
    """Return the first ``n`` SMILES for ``metal`` from the CSV.

    The CSV uses the column name ``SMILES_Ligands`` (not ``smiles``);
    falls back to ``smiles`` for the multi-task CSV variant.
    Empty / whitespace ``SMILES_Ligands`` are skipped.
    """
    out: List[str] = []
    if not csv_path.exists():
        return out
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        smi_col = "SMILES_Ligands" if "SMILES_Ligands" in fields else "smiles"
        if "Metal" not in fields:
            return out
        for row in reader:
            if (row.get("Metal", "") or "").strip() != metal:
                continue
            smi = (row.get(smi_col, "") or "").strip()
            if smi:
                out.append(smi)
            if len(out) >= int(n):
                break
    return out


# ---------------------------------------------------------------------------
# Per-metal benchmark — runs the orchestrator on ``smiles`` and
# records per-step + per-sample metrics.
# ---------------------------------------------------------------------------
def _benchmark_metal(
    metal: str,
    smiles: List[str],
    pipeline: Any,
) -> Dict[str, Any]:
    """Run the full R18 pipeline on ``smiles`` for one metal.

    Returns a dict with the per-step wall-time totals, per-step memory
    deltas, per-step status counts, and aggregate metrics.
    """
    per_step_wall_s: Dict[str, float] = {step: 0.0 for step in STEP_IDS}
    per_step_mem_mb: Dict[str, float] = {step: 0.0 for step in STEP_IDS}
    per_step_status: Dict[str, Dict[str, int]] = {
        step: {"ok": 0, "skipped": 0, "error": 0} for step in STEP_IDS
    }
    per_sample_total_wall_s: List[float] = []
    n_success = 0

    t_metal_0 = time.perf_counter()
    for smi in smiles:
        res = pipeline.run_full_pipeline(smi)
        per_sample_total_wall_s.append(float(res.get("total_wall_s", 0.0)))
        if bool(res.get("success", False)):
            n_success += 1
        for step in STEP_IDS:
            step_dict = res.get(step, {}) or {}
            w = float(step_dict.get("wall_time_s", 0.0) or 0.0)
            m = float(step_dict.get("memory_mb", 0.0) or 0.0)
            st = str(step_dict.get("status", "error"))
            per_step_wall_s[step] += w
            per_step_mem_mb[step] += m
            per_step_status[step][st] = per_step_status[step].get(st, 0) + 1
    wall_s = time.perf_counter() - t_metal_0

    n = max(1, len(smiles))
    return {
        "metal": metal,
        "n_samples": int(len(smiles)),
        "n_success": int(n_success),
        "success_rate": float(n_success) / n,
        "wall_time_s": float(wall_s),
        "per_step_wall_time_s": {
            step: float(per_step_wall_s[step]) for step in STEP_IDS
        },
        "per_step_memory_mb": {step: float(per_step_mem_mb[step]) for step in STEP_IDS},
        "per_step_status_counts": per_step_status,
        "per_step_mean_wall_time_s": {
            step: float(per_step_wall_s[step]) / n for step in STEP_IDS
        },
        "per_sample_total_wall_s_mean": (float(sum(per_sample_total_wall_s)) / n),
        "per_sample_total_wall_s_median": (_median(per_sample_total_wall_s)),
    }


def _median(xs: Sequence[float]) -> float:
    """Return the median of ``xs`` (0.0 when ``xs`` is empty)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * (float(s[n // 2 - 1]) + float(s[n // 2]))


# ---------------------------------------------------------------------------
# JSON-safe writer — coerces numpy scalars and sets.
# ---------------------------------------------------------------------------
def _write_json(payload: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(o: Any) -> Any:
        try:
            import numpy as _np  # local import; cheap

            if isinstance(o, (_np.floating, _np.integer)):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)

    with dest.open("w") as fh:
        json.dump(payload, fh, indent=2, default=_coerce)


# ---------------------------------------------------------------------------
# Markdown summary writer
# ---------------------------------------------------------------------------
def _write_markdown(
    payload: Dict[str, Any],
    per_metal: Dict[str, Dict[str, Any]],
    dest: Path,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    metals: List[str] = payload["metals"]
    n_per = int(payload["n_samples_per_metal"])
    n_actual = payload["per_metal_n_samples"]
    lines: List[str] = []
    lines.append("# R18 end-to-end pipeline benchmark on MetalCytoToxDB")
    lines.append("")
    lines.append(f"**Date**: {payload['timestamp_utc']}  ")
    lines.append(f"**Dataset**: `{payload['dataset']}`  ")
    lines.append(f"**Pipeline**: 5-step R18 orchestrator (32 ship-modules chained)  ")
    lines.append(f"**N samples per metal** (target): {n_per}  ")
    lines.append(f"**Metals**: {', '.join(metals)}  ")
    lines.append("")

    lines.append("## Per-metal summary")
    lines.append("")
    lines.append(
        "| Metal | N samples | Wall-time (s) | Success rate | Mean sample wall (s) |"
    )
    lines.append(
        "|-------|-----------|---------------|---------------|---------------------|"
    )
    for m in metals:
        d = per_metal[m]
        lines.append(
            f"| {m} | {n_actual.get(m, 0)} | {d['wall_time_s']:.2f} "
            f"| {d['success_rate']:.2%} | {d['per_sample_total_wall_s_mean']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Total wall-time**: {payload['total_wall_time_s']:.2f} s "
        f"across all {len(metals)} metals "
        f"({sum(n_actual.values())} samples)."
    )
    lines.append("")
    lines.append(f"**Peak memory delta**: {payload['peak_memory_mb_total']:.2f} MB")
    lines.append("")

    lines.append("## Per-step × per-metal total wall-time (seconds)")
    lines.append("")
    header = "| Step | " + " | ".join(metals) + " | **Sum** |"
    sep = "|---|" + "|".join(["---"] * (len(metals) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for step in STEP_IDS:
        cells: List[str] = []
        s = 0.0
        for m in metals:
            v = float(payload["per_step_per_metal_wall_time_s"][step].get(m, 0.0))
            cells.append(f"{v:.4f}")
            s += v
        lines.append(
            f"| {step} ({STEP_LABELS[step]}) | "
            + " | ".join(cells)
            + f" | **{s:.4f}** |"
        )
    lines.append("")

    lines.append("## Per-step × per-metal mean wall-time per sample (seconds)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for step in STEP_IDS:
        cells: List[str] = []
        for m in metals:
            v = float(payload["per_step_per_metal_mean_wall_time_s"][step].get(m, 0.0))
            cells.append(f"{v:.6f}")
        lines.append(f"| {step} | " + " | ".join(cells) + " |  |")
    lines.append("")

    lines.append("## Per-step × per-metal status counts")
    lines.append("")
    for step in STEP_IDS:
        lines.append(f"### {step}")
        lines.append("")
        lines.append("| Metal | ok | skipped | error | total |")
        lines.append("|-------|----|---------|-------|-------|")
        for m in metals:
            st = payload["per_step_per_metal_status_counts"][step].get(m, {})
            ok = int(st.get("ok", 0))
            sk = int(st.get("skipped", 0))
            er = int(st.get("error", 0))
            lines.append(f"| {m} | {ok} | {sk} | {er} | {ok + sk + er} |")
        lines.append("")

    lines.append("## How to re-run")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "source env.sh && uv run --no-sync python "
        "molmetal/scripts/benchmark_metalcytotox_e2e.py "
        f"--n-samples-per-metal {n_per}"
    )
    lines.append("```")
    lines.append("")

    with dest.open("w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_metals(arg: str) -> List[str]:
    out: List[str] = []
    for token in arg.split(","):
        t = token.strip()
        if t and t not in out:
            out.append(t)
    return out or list(DEFAULT_METALS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "R18 end-to-end pipeline on MetalCytoToxDB 5K real data — "
            "per-metal × per-step benchmark."
        ),
    )
    parser.add_argument(
        "--n-samples-per-metal",
        type=int,
        default=DEFAULT_N_SAMPLES_PER_METAL,
        help="Per-metal sample cap (default: 500).",
    )
    parser.add_argument(
        "--metals",
        type=str,
        default=",".join(DEFAULT_METALS),
        help="Comma-separated metal list (default: Ru,Ir,Rh,Re,Os).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV),
        help="MetalCytoToxDB CSV path (default: .data/molmetal/MetalCytoToxDB.csv).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(OUT_JSON),
        help="Output JSON path (default: wf_e2e_pipeline/metalcytotox_e2e.json).",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=str(OUT_MD),
        help="Output Markdown summary path.",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.stderr.write(f"[bench] CSV not found: {csv_path}\n")
        return 1

    metals = _parse_metals(args.metals)
    n_per_metal = max(1, int(args.n_samples_per_metal))

    # Build the orchestrator once; the inner module initialisation
    # takes a few seconds and is shared across all metals.
    from molmetal.scripts.r18_pipeline_orchestrator import R18Pipeline

    print(f"[bench] Initialising R18Pipeline ...")
    t_init_0 = time.perf_counter()
    pipeline = R18Pipeline()
    t_init = time.perf_counter() - t_init_0
    print(f"[bench]   init done in {t_init:.2f}s")

    # Per-metal accumulators
    per_metal_n_samples: Dict[str, int] = {m: 0 for m in metals}
    per_metal_wall_s: Dict[str, float] = {m: 0.0 for m in metals}
    per_metal_success_rate: Dict[str, float] = {m: 0.0 for m in metals}
    per_step_per_metal_wall_s: Dict[str, Dict[str, float]] = {
        step: {m: 0.0 for m in metals} for step in STEP_IDS
    }
    per_step_per_metal_memory_mb: Dict[str, Dict[str, float]] = {
        step: {m: 0.0 for m in metals} for step in STEP_IDS
    }
    per_step_per_metal_status: Dict[str, Dict[str, Dict[str, int]]] = {
        step: {m: {"ok": 0, "skipped": 0, "error": 0} for m in metals}
        for step in STEP_IDS
    }
    per_step_per_metal_mean_wall_s: Dict[str, Dict[str, float]] = {
        step: {m: 0.0 for m in metals} for step in STEP_IDS
    }
    per_metal_results: Dict[str, Dict[str, Any]] = {}

    t_total_0 = time.perf_counter()
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]

    for metal in metals:
        print(f"[bench] === metal={metal} loading samples ...")
        smiles = _load_metal_samples(csv_path, metal, n_per_metal)
        per_metal_n_samples[metal] = len(smiles)
        if not smiles:
            sys.stderr.write(f"[bench] no SMILES for metal={metal} in {csv_path}\n")
            continue
        print(
            f"[bench]   loaded {len(smiles)} samples, "
            f"running 5-step R18 pipeline ..."
        )
        result = _benchmark_metal(metal, smiles, pipeline)
        per_metal_results[metal] = result
        per_metal_wall_s[metal] = float(result["wall_time_s"])
        per_metal_success_rate[metal] = float(result["success_rate"])
        for step in STEP_IDS:
            per_step_per_metal_wall_s[step][metal] = float(
                result["per_step_wall_time_s"][step]
            )
            per_step_per_metal_memory_mb[step][metal] = float(
                result["per_step_memory_mb"][step]
            )
            per_step_per_metal_status[step][metal] = {
                k: int(v) for k, v in result["per_step_status_counts"][step].items()
            }
            per_step_per_metal_mean_wall_s[step][metal] = float(
                result["per_step_mean_wall_time_s"][step]
            )
        print(
            f"[bench]   metal={metal} wall_s={result['wall_time_s']:.2f} "
            f"success={result['n_success']}/{result['n_samples']} "
            f"({result['success_rate']:.2%})"
        )

    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    total_wall_s = time.perf_counter() - t_total_0

    # Build the report payload.
    try:
        import torch  # local; cheap

        torch_ver = torch.__version__
    except Exception:
        torch_ver = "n/a"

    py_ver = ".".join(str(v) for v in sys.version_info[:3])

    payload: Dict[str, Any] = {
        "dataset": "metalcytotox",
        "csv_path": str(csv_path),
        "metals": list(metals),
        "n_samples_per_metal": int(n_per_metal),
        "n_metals": int(len(metals)),
        "n_steps": int(len(STEP_IDS)),
        "step_id_order": list(STEP_IDS),
        "step_labels": {s: STEP_LABELS[s] for s in STEP_IDS},
        "per_metal_n_samples": {m: int(v) for m, v in per_metal_n_samples.items()},
        "per_metal_wall_time_s": {m: float(v) for m, v in per_metal_wall_s.items()},
        "per_metal_success_rate": {
            m: float(v) for m, v in per_metal_success_rate.items()
        },
        "per_metal_per_sample_wall_time_mean_s": {
            m: float(
                per_metal_results.get(m, {}).get("per_sample_total_wall_s_mean", 0.0)
            )
            for m in metals
        },
        "per_metal_per_sample_wall_time_median_s": {
            m: float(
                per_metal_results.get(m, {}).get("per_sample_total_wall_s_median", 0.0)
            )
            for m in metals
        },
        "per_step_per_metal_wall_time_s": {
            step: {m: float(per_step_per_metal_wall_s[step][m]) for m in metals}
            for step in STEP_IDS
        },
        "per_step_per_metal_memory_mb": {
            step: {m: float(per_step_per_metal_memory_mb[step][m]) for m in metals}
            for step in STEP_IDS
        },
        "per_step_per_metal_status_counts": {
            step: {
                m: {k: int(v) for k, v in per_step_per_metal_status[step][m].items()}
                for m in metals
            }
            for step in STEP_IDS
        },
        "per_step_per_metal_mean_wall_time_s": {
            step: {m: float(per_step_per_metal_mean_wall_s[step][m]) for m in metals}
            for step in STEP_IDS
        },
        "total_wall_time_s": float(total_wall_s),
        "pipeline_init_time_s": float(t_init),
        "peak_memory_mb_total": float(peak_mem - base_mem) / (1024 * 1024),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": py_ver,
        "torch": torch_ver,
        "honest_framing": (
            "CPU-only. The orchestrator pins to the CPU backend; "
            "no CUDA / Triton / CFM retraining is exercised. "
            "Pipeline init (~few s) is amortised across all metals. "
            "Per-metal = per_metal_wall_time_s excludes pipeline init."
        ),
    }

    # Write JSON
    out_json = Path(args.out_json)
    _write_json(payload, out_json)

    # Write Markdown
    out_md = Path(args.out_md)
    _write_markdown(payload, per_metal_results, out_md)

    n_total_samples = sum(per_metal_n_samples.values())
    n_total_success = sum(
        per_metal_results.get(m, {}).get("n_success", 0) for m in metals
    )
    print(
        f"[bench] DONE: metals={len(metals)} "
        f"samples={n_total_samples} "
        f"success={n_total_success}/{n_total_samples} "
        f"total_wall_s={total_wall_s:.2f} "
        f"peak_mem_mb={payload['peak_memory_mb_total']:.1f}"
    )
    for m in metals:
        print(
            f"[bench]   {m}: n={per_metal_n_samples[m]} "
            f"wall_s={per_metal_wall_s[m]:.2f} "
            f"success={per_metal_success_rate[m]:.2%}"
        )
    print(f"[bench] wrote {out_json}")
    print(f"[bench] wrote {out_md}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
