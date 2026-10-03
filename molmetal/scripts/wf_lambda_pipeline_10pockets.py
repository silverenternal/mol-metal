"""TODO/32 B5 — Lambda pipeline smoke, 10-pocket expansion (Track-B KataGo).

================================================================
Goal
================================================================
Scale the B5 single-pocket smoke (workflow H, ``test_001``) to
``test_010..test_019`` — 10 pockets from the CrossDocked2020 subset.
This is the B5-expansion entry in
``TODO/pending/34_parallel_ultracode_dispatch.md`` §2.2 (Track B).

Per-pocket pipeline (reused from `wf_lambda_pipeline_smoke.py`):
    pocket (`test_010..test_019`)
        │
        ▼
    [1] PLL signature
        ▼
    [2] PLL heuristic
        ▼
    [3] LamMCTS search  (1000 sims, FPU=0.25, PLL prior)
        ▼
    [4] top-100 candidates + closed β-NF annotation
        ▼
    [5] CSV + RDKit validate + SA/QED stats

Output
------
* ``logs/lambda_pipeline_smoke_10pockets/<pocket_id>/candidates.csv``
* ``logs/lambda_pipeline_smoke_10pockets/<pocket_id>/summary.json``
* ``logs/lambda_pipeline_smoke_10pockets/aggregate.json`` — 10-pocket roll-up
* ``molmetal/reports/wf_lambda_pipeline_10pockets/final.md`` — per-pocket highlights

Wall-clock budget
-----------------
Each pocket ≈ 12 min CPU (1000 sims × 12 seeds × ~83 sims/seed on
``test_001`` was 693 s ≈ 11.5 min).  10 pockets ≈ 115 min total —
ran sequentially in this script via ``run_in_background``.

Pocket manifest
---------------
``molmetal/data/test_010_019_subset_manifest.csv`` (10 rows) — we
use the ``pocket_id`` column directly; the SDF ligand_path is
resolved by ``_resolve_pocket_smiles`` from the underlying
``crossdocked100_manifest.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Repo root on sys.path so ``molmetal_lam`` + ``scripts.wf_lambda_pipeline_smoke``
# both resolve without an install.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "molmetal" / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "molmetal" / "scripts"))

# Reuse the workflow H smoke primitives unchanged.
from wf_lambda_pipeline_smoke import (
    N_CANDIDATES,
    N_SIMULATIONS,
    MAX_DEPTH,
    TOP_K,
    save_artifacts,
    write_report_md,
    run_pipeline,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("wf_lambda_pipeline_10pockets")


# ---------------------------------------------------------------------------
# Constants / manifest
# ---------------------------------------------------------------------------

OUT_ROOT = _REPO_ROOT / "logs" / "lambda_pipeline_smoke_10pockets"
REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_lambda_pipeline_10pockets"
MANIFEST_CSV = _REPO_ROOT / "molmetal" / "data" / "test_010_019_subset_manifest.csv"


def _load_pocket_ids_from_manifest() -> List[str]:
    """Read ``pocket_id`` column from the subset manifest CSV.

    Falls back to ``test_010..test_019`` (10 IDs) when the file is
    missing or unreadable — keeps the smoke runnable on a fresh clone.
    """
    fallback = [f"test_{i:03d}" for i in range(10, 20)]
    if not MANIFEST_CSV.exists():
        log.warning("manifest %s missing — using fallback IDs %s",
                    MANIFEST_CSV, fallback)
        return fallback
    try:
        with MANIFEST_CSV.open() as f:
            reader = csv.DictReader(f)
            ids = [str(r["pocket_id"]).strip() for r in reader
                   if r.get("pocket_id")]
        if not ids:
            log.warning("manifest %s empty — using fallback", MANIFEST_CSV)
            return fallback
        log.info("loaded %d pocket IDs from manifest: %s", len(ids), ids)
        return ids
    except Exception as exc:
        log.warning("manifest read failed (%s) — using fallback", exc)
        return fallback


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(per_pocket: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up the 10 per-pocket stats dicts into a single summary."""
    n_total = len(per_pocket)
    if n_total == 0:
        return {"n_pockets": 0, "aggregate": {}}

    def _mean(key: str) -> Optional[float]:
        vals = [float(p[key]) for p in per_pocket
                if p.get(key) is not None
                and isinstance(p.get(key), (int, float))]
        if not vals:
            return None
        return float(np.mean(vals))

    def _sum(key: str) -> int:
        return int(sum(int(p.get(k) or 0) for p in per_pocket
                       for k in (key,) if isinstance(p.get(key), (int, float))
                       ))

    # Per-pocket rate fields (use scalar mean across pockets).
    rate_keys = [
        "n_emitted", "n_valid_rdkit", "n_unique_valid_smi",
        "sa_n", "qed_n", "n_simulations_total", "nfe",
        "nfe_reductions", "history_entries", "n_tiles",
    ]
    aggregate: Dict[str, Any] = {
        "n_pockets": int(n_total),
        "wallclock_s_sum": float(sum(
            float(p.get("wallclock_s", 0.0) or 0.0) for p in per_pocket
        )),
        "wallclock_s_mean": float(np.mean([
            float(p.get("wallclock_s", 0.0) or 0.0) for p in per_pocket
        ])),
        "n_unique_valid_smi_mean": _mean("n_unique_valid_smi"),
        "validity_rate_mean": (
            float(np.mean([
                (float(p.get("n_valid_rdkit", 0)) /
                 max(1.0, float(p.get("n_emitted", 1))))
                for p in per_pocket
            ]))
        ),
        "uniqueness_rate_mean": (
            float(np.mean([
                (float(p.get("n_unique_valid_smi", 0)) /
                 max(1.0, float(p.get("n_valid_rdkit", 1))))
                for p in per_pocket
            ]))
        ),
        "sa_mean": _mean("sa_mean"),
        "qed_mean": _mean("qed_mean"),
        "n_emitted_total": int(sum(
            int(p.get("n_emitted", 0) or 0) for p in per_pocket
        )),
        "n_valid_total": int(sum(
            int(p.get("n_valid_rdkit", 0) or 0) for p in per_pocket
        )),
        "n_unique_total": int(sum(
            int(p.get("n_unique_valid_smi", 0) or 0) for p in per_pocket
        )),
        "nfe_total": int(sum(
            int(p.get("nfe", 0) or 0) for p in per_pocket
        )),
        "nfe_reductions_total": int(sum(
            int(p.get("nfe_reductions", 0) or 0) for p in per_pocket
        )),
        "rules_used_union": sorted({
            str(r) for p in per_pocket
            for r in (p.get("rules_used") or [])
        }),
        "fpu_reduction": float(per_pocket[0].get("fpu_reduction", 0.25)),
        "pll_k": int(per_pocket[0].get("pll_k", 16)),
        "pll_knn": int(per_pocket[0].get("pll_knn", 8)),
        "pll_available_all": bool(all(
            bool(p.get("pll_available", False)) for p in per_pocket
        )),
        "n_simulations_per_pocket": int(per_pocket[0].get(
            "n_simulations_total", N_SIMULATIONS,
        )),
        "max_depth": int(per_pocket[0].get("max_depth", MAX_DEPTH)),
        "top_k": int(per_pocket[0].get("top_k", TOP_K)),
    }
    return {
        "n_pockets": int(n_total),
        "pocket_ids": [str(p.get("pocket_id")) for p in per_pocket],
        "per_pocket": per_pocket,
        "aggregate": aggregate,
    }


def _write_aggregate_json(per_pocket: List[Dict[str, Any]], out_path: Path) -> None:
    """Dump the aggregate roll-up to ``aggregate.json``."""
    payload = _aggregate(per_pocket)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("wrote aggregate -> %s", out_path)


def _write_final_md(per_pocket: List[Dict[str, Any]], md_path: Path) -> None:
    """Render the human-readable 10-pocket summary."""
    agg = _aggregate(per_pocket)["aggregate"]
    lines: List[str] = []
    add = lines.append
    add("# TODO/32 B5 — Lambda pipeline smoke, 10-pocket expansion")
    add("")
    add(f"- n_pockets: {agg['n_pockets']} ({per_pocket[0].get('pocket_id','?')}..{per_pocket[-1].get('pocket_id','?')})")
    add(f"- pipeline: PLL signature → PLL heuristic → LamMCTS "
        f"(KataGo FPU={agg['fpu_reduction']}) → top-{agg['top_k']} candidates "
        f"→ closed β-NF annotation")
    add(f"- n_simulations_per_pocket: {agg['n_simulations_per_pocket']} "
        f"max_depth: {agg['max_depth']}")
    add("")
    add("## Aggregate metrics")
    add("")
    add(f"- n_unique_valid_smi_mean: {agg['n_unique_valid_smi_mean']:.2f}")
    add(f"- validity_rate_mean: {agg['validity_rate_mean']:.4f}")
    add(f"- uniqueness_rate_mean: {agg['uniqueness_rate_mean']:.4f}")
    add(f"- sa_mean: {agg['sa_mean']:.4f}" if agg['sa_mean'] is not None
        else "- sa_mean: N/A")
    add(f"- qed_mean: {agg['qed_mean']:.4f}" if agg['qed_mean'] is not None
        else "- qed_mean: N/A")
    add(f"- n_emitted_total: {agg['n_emitted_total']} "
        f"n_valid_total: {agg['n_valid_total']} "
        f"n_unique_total: {agg['n_unique_total']}")
    add(f"- nfe_total: {agg['nfe_total']} "
        f"nfe_reductions_total: {agg['nfe_reductions_total']}")
    add(f"- wallclock_s_sum: {agg['wallclock_s_sum']:.1f} "
        f"(mean per pocket: {agg['wallclock_s_mean']:.1f})")
    add(f"- rules_used (union): {agg['rules_used_union']}")
    add(f"- pll_k: {agg['pll_k']} pll_knn: {agg['pll_knn']} "
        f"pll_available_all: {agg['pll_available_all']}")
    add("")
    add("## Per-pocket highlights")
    add("")
    add("| pocket | pocket_smiles | n_unique_valid | sa_mean | qed_mean | wallclock_s |")
    add("|---|---|---:|---:|---:|---:|")
    for p in per_pocket:
        smi = p.get("pocket_smiles_resolved") or "?"
        smi_short = (smi[:40] + "..") if len(smi) > 42 else smi
        sa_v = p.get("sa_mean")
        qed_v = p.get("qed_mean")
        sa_str = f"{sa_v:.3f}" if isinstance(sa_v, (int, float)) else "N/A"
        qed_str = f"{qed_v:.3f}" if isinstance(qed_v, (int, float)) else "N/A"
        add(f"| `{p.get('pocket_id')}` | `{smi_short}` | "
            f"{p.get('n_unique_valid_smi','?')} | "
            f"{sa_str} | {qed_str} | "
            f"{float(p.get('wallclock_s',0.0)):.1f} |")
    add("")
    add("PLL + KataGo-style FPU (Liu & Feng 2026, "
        "DOI: 10.1021/acs.jcim.6c02460).  Closed β-NF term via "
        "`_molecule_to_lambda_expr` (atoms=left-associated app).")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("wrote final report -> %s", md_path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pocket_ids = _load_pocket_ids_from_manifest()
    log.info("=== TODO/32 B5 — lambda pipeline smoke (10 pockets) ===")
    log.info("pockets: %s", pocket_ids)
    log.info("out_root: %s", OUT_ROOT)

    per_pocket_stats: List[Dict[str, Any]] = []
    t_global = time.time()
    for i, pocket_id in enumerate(pocket_ids):
        log.info("--- [%d/%d] pocket=%s ---", i + 1, len(pocket_ids), pocket_id)
        t_pocket = time.time()
        try:
            candidates, stats = run_pipeline(
                pocket_id=pocket_id,
                n_candidates=N_CANDIDATES,
                n_simulations=N_SIMULATIONS,
                max_depth=MAX_DEPTH,
                top_k=TOP_K,
                seed=i,  # different seed per pocket for diversity
                n_seeds=12,
            )
            out_dir = OUT_ROOT / pocket_id
            save_artifacts(candidates, stats, out_dir)
            stats["wallclock_s"] = float(time.time() - t_pocket)
            per_pocket_stats.append(stats)
            log.info("--- [%d/%d] pocket=%s done in %.1fs (n_unique_valid=%d) ---",
                     i + 1, len(pocket_ids), pocket_id,
                     stats["wallclock_s"],
                     stats.get("n_unique_valid_smi", 0))
        except Exception as exc:
            log.error("--- [%d/%d] pocket=%s FAILED: %s ---",
                      i + 1, len(pocket_ids), pocket_id, exc)
            per_pocket_stats.append({
                "pocket_id": pocket_id,
                "error": str(exc),
                "wallclock_s": float(time.time() - t_pocket),
            })

    # ---- aggregate + final report
    _write_aggregate_json(per_pocket_stats, OUT_ROOT / "aggregate.json")
    _write_final_md(per_pocket_stats, REPORT_DIR / "final.md")

    log.info("=== DONE; %d pockets, %.1fs wall ===",
             len(pocket_ids), time.time() - t_global)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
