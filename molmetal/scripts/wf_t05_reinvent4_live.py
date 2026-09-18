"""WF-T05 — TODO-05 REINVENT4 live multiproperty smoke via DIRECT-API.

Drives :class:`REINVENT4APIAdapter` (NOT subprocess) against the real
REINVENT4 source tree that lives in the isolated
``/mnt/storage/env-projects/reinvent4-rocm`` venv. The adapter builds a
:class:`reinvent.scoring.Scorer` in-process and calls
``scorer(smilies, valid_mask, duplicate_mask)`` directly — there is no
JSONL worker, no subprocess, no RPC.

This script MUST be launched with the reinvent4-rocm Python interpreter
so that ``import reinvent`` resolves. We bake that assumption into the
CLI's hard requirement: if the upstream REINVENT4 import is not
available the adapter degrades to all-zeros and we record the failure
mode (NOT a silent success).

Outputs (under --output-dir, default
``molmetal/reports/wf_t05_reinvent4_live/``):
    * ``report.json`` — per-SMILES score + latency + aggregate stats
    * ``summary.md``  — human-readable summary
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# 10 known-good calibration SMILES (verbatim from
# molmetal/scripts/test_reinvent4_multiproperty_batch.py SMILES_BATCH).
SMILES_BATCH: List[str] = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CCN(CC)CC",
    "C1CCCCC1",
    "OC1=CC=CC=C1",
    "CC(=O)NCC(=O)N",
    "CCCCCCCC",
]

# Extended 40 SMILES (50 total when concatenated) — a quick ChEMBL-ish
# drug-like spread. We derive 50 SMILES by adding a 40-SMILES list of
# common small organic fragments / drug-like molecules to the 10 above.
EXTRA_40: List[str] = [
    # Drug-like aromatics
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC1(C)SC2C(NC(=O)C2c3ccc(O)cc3)C(=O)N1C4CCCCC4",  # penicillin G skeleton
    "O=C(O)c1ccccc1O",  # salicylic acid
    "Cc1nnc2n1-c2c(=O)n(C)c(=O)n(C)c2",  # caffeine variant
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "CN(C)CCNc2cccc(c2)n1ncc2cc(ccc12)Cl",  # pyrimidine aniline
    "NC(=O)c1ccc(N)cc1",  # 4-aminobenzamide
    "O=C1NC(=O)C(=Cc2ccc(O)c(CO)c2)N1",  # uracil-derivative
    "CCOCCOc1ccc(CCN)cc1",  # phenoxyethylamine
    "OC1CCNCC1",  # 4-hydroxypiperidine
    "CN1CCNCC1",  # 1-methylpiperazine
    "CCN(CC)c1ccc(N)cc1",  # aniline derivative
    "Nc1ncnc2[nH]cnc12",  # adenine core
    "O=c1[nH]c(N)nc2[nH]cnc12",  # guanine core
    "OC(=O)c1ccc(N)cc1",  # PABA
    "CC(=O)Oc1ccc(CCN)cc1",  # phenacetin variant
    "NC1=NC(=O)N(Cc2ccccc2)C=C1",  # thymine core
    "CN1c2ccccc2C(=O)N(C)C1=O",  # caffeine analog
    "Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1",  # adenosine core
    "OCC(O)C(O)C(O)C(O)CO",  # mannitol
    # Aliphatic
    "CC(=O)O",  # acetic acid
    "CCCO",  # n-propanol
    "CCCCO",  # n-butanol
    "CCCCN",  # n-butylamine
    "CC(C)O",  # isopropanol
    "CC(C)C",  # isobutane
    "CCCCC(=O)O",  # valeric acid
    "CCC(=O)C",  # methyl ethyl ketone
    "CC(=O)CC(=O)C",  # acetylacetone
    "CCC(=O)N",  # propanamide
    "CC#N",  # acetonitrile
    "CC=O",  # acetaldehyde
    # Heterocycles
    "c1cc[nH]c1",  # pyrrole
    "c1ccoc1",  # furan
    "c1ccsc1",  # thiophene
    "c1cnc2ccccc12",  # quinoline
    "C1=CN=CN1",  # pyrimidine (tautomer)
    "c1ccc2[nH]ccc2c1",  # indole
    "O=C1CCCCN1",  # caprolactam
    "O=C1CCCN1",  # 2-pyrrolidone
    "O=C1CCCC1",  # cyclopentanone
    "O=C1CC=CC=C1",  # 2H-pyran-2-one
]


def _safe_corr(xs: List[float], ys: List[float]) -> Optional[float]:
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


def _score_smiles(adapter, smiles: str) -> Dict[str, Any]:
    """Score a single SMILES with full timing."""
    t0 = time.monotonic()
    out = adapter.score(smiles)
    dt = time.monotonic() - t0
    if not out:
        return {"smiles": smiles, "score": None, "elapsed_seconds": dt, "ok": False,
                "reason": "empty_result"}
    v = out[0]
    if v is None:
        return {"smiles": smiles, "score": None, "elapsed_seconds": dt, "ok": False,
                "reason": adapter.last_error or "null_score"}
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return {"smiles": smiles, "score": None, "elapsed_seconds": dt, "ok": False,
                "reason": "non_numeric"}
    if fv != fv:  # NaN
        return {"smiles": smiles, "score": None, "elapsed_seconds": dt, "ok": False,
                "reason": "nan"}
    return {"smiles": smiles, "score": fv, "elapsed_seconds": round(dt, 4), "ok": True,
            "reason": None}


def _score_batch(adapter, batch: List[str], label: str) -> Dict[str, Any]:
    """Score a batch and return aggregate stats."""
    print(f"\n[{label}] scoring {len(batch)} SMILES via direct API ...")
    rows = [_score_smiles(adapter, s) for s in batch]
    scores = [r["score"] for r in rows if r["ok"]]
    elapsed = [r["elapsed_seconds"] for r in rows]
    n_ok = len(scores)
    n_fail = len(rows) - n_ok
    metrics = {
        "label": label,
        "n_smiles": len(batch),
        "n_succeeded": n_ok,
        "n_failed": n_fail,
        "mean_score": statistics.mean(scores) if scores else None,
        "std_score": statistics.stdev(scores) if len(scores) > 1 else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "all_in_unit_interval": all(0.0 <= v <= 1.0 for v in scores),
        "mean_latency_seconds": statistics.mean(elapsed) if elapsed else None,
        "median_latency_seconds": statistics.median(elapsed) if elapsed else None,
        "total_wall_seconds": sum(elapsed),
        "first_invocation_seconds": elapsed[0] if elapsed else None,
        "steady_state_mean_seconds": (
            statistics.mean(elapsed[1:]) if len(elapsed) > 1 else None
        ),
    }
    return {"rows": rows, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("molmetal/reports/wf_t05_reinvent4_live"),
    )
    parser.add_argument(
        "--reinvent-root",
        type=Path,
        default=Path("/mnt/storage/env-projects/reinvent4-rocm/upstream"),
        help="Path to upstream REINVENT4 source tree (must contain reinvent/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/home/hugo/codes/try_triton_on_rocm/molmetal/references/REINVENT4/configs/stage1_scoring.toml"
        ),
        help="REINVENT4 scoring TOML config.",
    )
    parser.add_argument(
        "--smiles-list",
        choices=["ten", "fifty"],
        default="ten",
        help="Use the 10-SMILES calibration set or the full 50-SMILES batch.",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make repo importable.
    repo_root = Path(__file__).resolve().parents[2]
    for sub in (repo_root, repo_root / "molmetal"):
        sp = str(sub)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from molmetal_lam.sbdd_env import reinvent4_api_adapter as _api  # noqa: E402

    # Force the reinvent root to the reinvent4-rocm upstream tree.
    invent_root = str(args.reinvent_root.resolve())
    print(f"== WF-T05: TODO-05 REINVENT4 live multiproperty smoke (direct API) ==")
    print(f"invent_root: {invent_root}")
    print(f"scoring_config: {args.config}")

    # Probe importability BEFORE construction.
    importable = _api.is_reinvent_importable(invent_root)
    print(f"is_reinvent_importable (pre-add): {importable}")

    # Construct the adapter. This triggers Scorer init which may be slow
    # on first import.
    t0 = time.monotonic()
    adapter = _api.REINVENT4APIAdapter(
        scoring_config=str(args.config),
        reinvent_root=invent_root,
        device="cpu",
        timeout=120.0,
    )
    construct_dt = time.monotonic() - t0
    print(f"adapter construction: {construct_dt:.2f}s")
    print(f"adapter.available: {adapter.available}")
    print(f"adapter.last_error: {adapter.last_error}")
    if not adapter.available:
        # Honest NO-OP: don't pretend we ran a batch.
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "REINVENT4APIAdapter direct-API smoke (TODO-05 verify)",
            "scope": "FAILURE — adapter.unavailable; direct-API path cannot "
                     "construct Scorer in this environment",
            "honest_framing": {
                "importable_pre_add": importable,
                "importable_post_add": _api.is_reinvent_importable(invent_root),
                "construction_seconds": round(construct_dt, 4),
                "last_error": adapter.last_error,
            },
            "passed": False,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    # Phase 1: 10 SMILES calibration set.
    batch10 = SMILES_BATCH
    # Phase 2: 50 SMILES full batch.
    batch50 = SMILES_BATCH + EXTRA_40
    if args.smiles_list == "ten":
        results = [_score_batch(adapter, batch10, "ten_smiles_calibration")]
    else:
        results = [
            _score_batch(adapter, batch10, "ten_smiles_calibration"),
            _score_batch(adapter, batch50, "fifty_smiles_batch"),
        ]

    # Aggregate everything.
    all_scores: List[float] = []
    for r in results:
        all_scores.extend(s for s in (row["score"] for row in r["rows"]) if s is not None)
    overall_metrics = {
        "n_smiles_total": sum(r["metrics"]["n_smiles"] for r in results),
        "n_succeeded_total": sum(r["metrics"]["n_succeeded"] for r in results),
        "n_failed_total": sum(r["metrics"]["n_failed"] for r in results),
        "mean_score_overall": statistics.mean(all_scores) if all_scores else None,
        "std_score_overall": statistics.stdev(all_scores) if len(all_scores) > 1 else None,
        "all_in_unit_interval_overall": all(0.0 <= v <= 1.0 for v in all_scores),
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "REINVENT4APIAdapter direct-API smoke (TODO-05 verify)",
        "scope": "MEASURED — direct in-process Scorer() invocation, "
                 "no subprocess, no JSONL worker",
        "honest_framing": {
            "importable_pre_add": importable,
            "importable_post_add": _api.is_reinvent_importable(invent_root),
            "construction_seconds": round(construct_dt, 4),
            "adapter_available": adapter.available,
            "adapter_last_error": adapter.last_error,
            "adapter_last_components": adapter.last_components,
        },
        "config": {
            "reinvent_root": invent_root,
            "scoring_config": str(args.config),
            "device": "cpu",
            "timeout_seconds": 120.0,
            "smiles_lists": {
                "ten_smiles_calibration": SMILES_BATCH,
                "extra_40": EXTRA_40,
            },
        },
        "batches": results,
        "overall": overall_metrics,
        "passed": (
            all(r["metrics"]["all_in_unit_interval"] for r in results)
            and all(r["metrics"]["n_failed"] == 0 for r in results)
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + "=" * 60)
    print(json.dumps({
        "passed": report["passed"],
        "overall": overall_metrics,
        "batch_metrics": [r["metrics"] for r in results],
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())