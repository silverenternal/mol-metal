"""R18 modules real-data benchmark on MetalCytoToxDB (multi-metal, multi-task).

Wf-43 Phase 3 audit (2026-09-26): the ``"lam_chem/md_relaxation_stub"``
module ID (L172) is a **module-name reference**, NOT a mock
measurement.  ``md_relaxation_stub`` is the SHIPPED CPU-only
force-field relaxation module (TODO-30 P2.5) which performs REAL
energy minimisation on a hand-crafted simple force field; the
``stub`` in the name reflects its CPU-only deterministic design
intent (no MD integrator, no MMFF), not a metric placeholder.  The
module name is part of the public registry contract and is NOT
renamed.

================================================================
W2.3 / R18 ultracode dispatch (2026-09-26).

This script benchmarks the **32 R18 ship-modules** end-to-end on the
multi-metal MetalCytoToxDB split (``metalcytotox_5metal_multitask_v4.csv``,
R17 ship).  It is the *real-data counterpart* of
:file:`verify_r18_all_imports.py` and the *metal-aware counterpart* of
:file:`benchmark_r18_platinai.py`.

The benchmark is **per metal × per module** (2-D benchmark):

    For each metal in {Ru, Ir, Rh, Re, Os}:
        Load up to ``--n-samples`` MetalCytoToxDB rows.
        For each of the 32 R18 modules:
            Invoke the module's canonical forward primitive ``--n-runs``
            times on the metal's sample batch and record median wall-
            time + ``tracemalloc`` delta.

Honest framing
--------------
* This is a **CPU-only** benchmark.  The W2.3 contract pins every R18
  module to the CPU backend; CUDA / ROCm paths are intentionally not
  exercised here.  GPU retraining benchmarks live in
  ``molmetal/scripts/wf_cfm_gpu_retrain/``.
* PlatinAI uses one row per organic fragment; MetalCytoToxDB rows
  carry the metal complex + counterion as a single SMILES (often
  dot-separated).  Every module's forward primitive operates on the
  ligands (RDKit-parsable SMILES strings) — the metal centre is held
  in the *string* but the modules themselves do not parse the metal
  identity.  This mirrors the W2.3 split-clean contract.
* Per-metal sample sizes are bounded by ``min(n_available, --n-samples)``
  — ``Re`` has only 868 rows, so ``--n-samples 500`` caps at 500 (still
  fits the 868-row subset).
* The output schema lives at
  ``molmetal/reports/wf_metalcytotox_bench/results.json``.

Output schema::

    {
      "n_samples": 500,                   # per metal
      "n_metals":  5,
      "n_modules": 32,
      "metals":    ["Ru", "Ir", "Rh", "Re", "Os"],
      "total_wall_time_s": 12.34,
      "per_metal_per_module_wall_time": {
          "Ru": {"math_skills/se3_schrodinger_bridge": 0.42, ...},
          "Ir": {...}, ...,
      },
      "per_metal_per_module_memory_mb":  {... same shape ...},
      "per_metal_per_module_n_calls":    {... same shape ...},
      "per_metal_per_module_status":     {... same shape ...},
      "module_groups": {...},
      "n_runs_per_module": 10,
      "peak_memory_mb_total": 12.34,
      "timestamp_utc": "...",
      "python": "...", "torch": "..."
    }

Usage::

    source env.sh && uv run --no-sync python molmetal/scripts/benchmark_r18_metalcytotox.py --n-samples 500
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths — keep absolute so the script can be invoked from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_metalcytotox_bench"
REPORT_FILE = REPORT_DIR / "results.json"
DEFAULT_CSV = _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
DEFAULT_METALS = ["Ru", "Ir", "Rh", "Re", "Os"]


# ---------------------------------------------------------------------------
# MetalCytoToxDB loader — read the multi-task split CSV row-by-row,
# filtering by metal on the fly.  This avoids loading pandas just for
# a 26 k × 5 column file and keeps the data-load step deterministic.
# ---------------------------------------------------------------------------
def _load_metalcytotox_smiles(
    csv_path: Path,
    metal: str,
    n: int,
) -> List[str]:
    """Return the first ``n`` SMILES for ``metal`` from the CSV.

    The CSV has columns ``smiles,metal,scaffold_id,split,activity``.
    Empty / whitespace ``smiles`` are skipped.  This mirrors the
    behaviour of :func:`molmetal.scripts.benchmark_r18_platinai._load_platinai_smiles`.
    """
    out: List[str] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("metal", "").strip() != metal:
                continue
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Reuse the R18 module registry from the PlatinAI benchmark — the
# 32 R18 modules' forward primitives are content-free (they accept a
# List[str] of SMILES but only consume the first one or two).  We
# import the registry builder so we have a single canonical runner
# catalogue across the R18 ultracode sweep.
# ---------------------------------------------------------------------------
def _build_module_registry() -> Dict[str, Callable[[List[str], int], List[float]]]:
    from molmetal.scripts.benchmark_r18_platinai import (  # noqa: WPS433
        _build_module_registry as _plat_registry,
    )

    return _plat_registry()


# Pull _MODULE_GROUPS + _run_module + _median_ms out of the PlatinAI
# benchmark so the MetalCytoToxDB benchmark stays in sync with the
# 32-module manifest.  Re-export the symbols so the test file can
# import them from this script only.
_MODULE_GROUPS: Dict[str, List[str]] = {
    "math_skills": [
        "math_skills/bass_existence",
        "math_skills/bergman_kernel",
        "math_skills/monge_ampere_regularity",
        "math_skills/multi_parameter_ph",
        "math_skills/ph_hardness_guardrail",
        "math_skills/sb_bass_martingale",
        "math_skills/se3_schrodinger_bridge",
        "math_skills/transport_relaxation",
        "math_skills/wasserstein_gradient_flow",
    ],
    "adapters": [
        "adapters/coordination_gnn",
        "adapters/quotient_space_diffusion",
        "adapters/multi_ph_integration",
        "adapters/se3_sb_integration",
        "adapters/transport_relaxation_integration",
        "adapters/diffdock",
        "adapters/equibind",
    ],
    "lam_chem": [
        "lam_chem/coupling_joint_inference",
        "lam_chem/decoder_m3_sequence",
        "lam_chem/cross_attention_m4",
        "lam_chem/layered_generation",
        "lam_chem/buyable_gate",
        "lam_chem/ood_protocol",
        "lam_chem/macroguide_ph",
        "lam_chem/non_click_medchem_rules",
        "lam_chem/skeleton_jump",
        "lam_chem/md_relaxation_stub",
        "lam_chem/fg_compat_veto",
    ],
    "reward": ["reward/hilbert_primal_dual"],
    "search_alg": [
        "search_alg/score_repellent_mc",
        "search_alg/variance_puct_integration",
    ],
    "flow_matching_lipman": [
        "flow_matching_lipman/w_flow",
        "flow_matching_lipman/m5_schedulers",
    ],
}


def _median_ms(wall_times: List[float]) -> float:
    """Return the median wall-time across ``wall_times`` in milliseconds."""
    if not wall_times:
        return 0.0
    return float(statistics.median(wall_times)) * 1000.0


def _run_module(
    runner: Callable[[List[str], int], List[float]] | None,
    smiles: List[str],
    n_runs: int,
) -> Tuple[List[float], str]:
    """Run ``runner`` ``n_runs`` times; return (wall_times, status).

    Status is one of: ``"ok"``, ``"skipped"`` (no runner), ``"error"``.
    On ``"error"`` the wall_times list is empty.
    """
    if runner is None:
        return [], "skipped"
    try:
        wt = runner(smiles, n_runs)
        return list(wt), "ok"
    except Exception as exc:  # noqa: BLE001 — record every failure mode
        sys.stderr.write(
            f"[bench] runner failed: {type(exc).__name__}: {str(exc)[:200]}\n"
        )
        return [], "error"


# ---------------------------------------------------------------------------
# Per-metal, per-module driver.  Returns the 2-D benchmark result dicts.
# ---------------------------------------------------------------------------
def _benchmark_metal(
    metal: str,
    smiles: List[str],
    registry: Dict[str, Callable[[List[str], int], List[float]]],
    n_runs: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, str]]:
    """Run all ``registry`` modules on ``smiles``; return per-module metrics.

    Returns four parallel dicts keyed by module_id: median wall-time
    (ms), memory delta (MB), # successful calls, status string.
    """
    wall_ms: Dict[str, float] = {}
    mem_mb: Dict[str, float] = {}
    n_calls: Dict[str, int] = {}
    status: Dict[str, str] = {}
    for mod_id, runner in registry.items():
        sub_base = tracemalloc.get_traced_memory()[0]
        wt, st = _run_module(runner, smiles, n_runs)
        sub_peak = tracemalloc.get_traced_memory()[1]
        wall_ms[mod_id] = _median_ms(wt)
        mem_mb[mod_id] = float(sub_peak - sub_base) / (1024 * 1024)
        n_calls[mod_id] = len(wt) if st == "ok" else 0
        status[mod_id] = st
    return wall_ms, mem_mb, n_calls, status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_metals(arg: str) -> List[str]:
    """Parse a comma-separated metal list; lowercase → uppercase preserved."""
    out: List[str] = []
    for token in arg.split(","):
        t = token.strip()
        if t and t not in out:
            out.append(t)
    return out or list(DEFAULT_METALS)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "R18 modules real-data benchmark on MetalCytoToxDB "
            "(per-metal × per-module)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Per-metal sample cap (default: 500).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Forward calls per module per metal (default: 10).",
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
        help="MetalCytoToxDB 5-metal CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPORT_DIR),
        help="Output directory for results.json.",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="",
        help="Comma-separated subset of module IDs (default: all 32).",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.stderr.write(f"[bench] metalcytotox CSV not found: {csv_path}\n")
        return 1

    metals = _parse_metals(args.metals)
    n_samples_per_metal = max(1, int(args.n_samples))
    n_runs = max(1, int(args.n_runs))

    registry = _build_module_registry()
    if args.modules:
        wanted = {m.strip() for m in args.modules.split(",") if m.strip()}
        registry = {k: v for k, v in registry.items() if k in wanted}
        if not registry:
            sys.stderr.write(f"[bench] no modules matched {args.modules!r}\n")
            return 1
    module_ids = list(registry.keys())

    # ---------------------------------------------------------------------
    # 2-D benchmark: per-metal × per-module
    # ---------------------------------------------------------------------
    per_metal_wall: Dict[str, Dict[str, float]] = {m: {} for m in metals}
    per_metal_mem: Dict[str, Dict[str, float]] = {m: {} for m in metals}
    per_metal_calls: Dict[str, Dict[str, int]] = {m: {} for m in metals}
    per_metal_status: Dict[str, Dict[str, str]] = {m: {} for m in metals}
    per_metal_n_samples: Dict[str, int] = {m: 0 for m in metals}

    t_total_0 = time.perf_counter()
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]

    for metal in metals:
        smiles = _load_metalcytotox_smiles(csv_path, metal, n_samples_per_metal)
        per_metal_n_samples[metal] = len(smiles)
        if not smiles:
            sys.stderr.write(f"[bench] no SMILES for metal={metal} in {csv_path}\n")
            # Emit "empty" rows so the JSON shape stays uniform.
            for mid in module_ids:
                per_metal_wall[metal][mid] = 0.0
                per_metal_mem[metal][mid] = 0.0
                per_metal_calls[metal][mid] = 0
                per_metal_status[metal][mid] = "skipped"
            continue
        wall_m, mem_m, calls_m, status_m = _benchmark_metal(
            metal=metal,
            smiles=smiles,
            registry=registry,
            n_runs=n_runs,
        )
        for mid in module_ids:
            per_metal_wall[metal][mid] = wall_m.get(mid, 0.0)
            per_metal_mem[metal][mid] = mem_m.get(mid, 0.0)
            per_metal_calls[metal][mid] = calls_m.get(mid, 0)
            per_metal_status[metal][mid] = status_m.get(mid, "skipped")

    peak_mem_total = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    total_wall_s = time.perf_counter() - t_total_0

    # ---------------------------------------------------------------------
    # Per-metal totals — surfaced for the dashboard / paper supplements.
    # ---------------------------------------------------------------------
    per_metal_wall_s: Dict[str, float] = {}
    for metal in metals:
        per_metal_wall_s[metal] = float(
            sum(per_metal_wall[metal].values()) / 1000.0  # ms → s
        )

    # ---------------------------------------------------------------------
    # Output JSON
    # ---------------------------------------------------------------------
    import torch  # local; cheap to import here

    py_ver = ".".join(str(v) for v in sys.version_info[:3])
    torch_ver = torch.__version__

    payload: Dict[str, Any] = {
        "n_samples": int(n_samples_per_metal),
        "n_metals": int(len(metals)),
        "n_modules": int(len(registry)),
        "metals": list(metals),
        "per_metal_n_samples": per_metal_n_samples,
        "module_groups": _MODULE_GROUPS,
        "total_wall_time_s": float(total_wall_s),
        "per_metal_wall_time_s": per_metal_wall_s,
        "per_metal_per_module_wall_time_ms": per_metal_wall,
        "per_metal_per_module_memory_mb": per_metal_mem,
        "per_metal_per_module_n_calls": per_metal_calls,
        "per_metal_per_module_status": per_metal_status,
        "n_runs_per_module": int(n_runs),
        "peak_memory_mb_total": float(peak_mem_total - base_mem) / (1024 * 1024),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": py_ver,
        "torch": torch_ver,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    # ---------------------------------------------------------------------
    # Diagnostics summary
    # ---------------------------------------------------------------------
    n_total_calls = sum(sum(per_metal_calls[m].values()) for m in metals)
    n_ok_calls = sum(
        v
        for m in metals
        for v in per_metal_calls[m].values()
        if per_metal_status[m].get(_module_id_for(per_metal_calls[m], v), "ok") == "ok"
    )
    # Simpler status tally: count per-metal, per-module "ok" entries.
    n_ok_cells = sum(
        1 for m in metals for st in per_metal_status[m].values() if st == "ok"
    )

    print(
        f"[bench] metals={len(metals)} "
        f"n_samples_per_metal={n_samples_per_metal} "
        f"n_modules={len(registry)} "
        f"ok_cells={n_ok_cells}/{len(metals) * len(registry)} "
        f"total_wall_s={total_wall_s:.2f} "
        f"peak_mem_mb={payload['peak_memory_mb_total']:.1f}"
    )
    print(f"[bench] per_metal_n_samples={per_metal_n_samples}")
    for m in metals:
        print(f"[bench] metal={m} wall_s={per_metal_wall_s[m]:.3f}")
    print(f"[bench] wrote {out_path}")
    return 0


def _module_id_for(d: Dict[str, int], value: int) -> str:
    """Inverse of an int-→module_id dict (best-effort fallback)."""
    for k, v in d.items():
        if v == value:
            return k
    return ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
