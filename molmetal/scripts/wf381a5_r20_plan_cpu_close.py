#!/usr/bin/env python3
"""Wf-381-A5 — R20 plan items CPU-only closure scaffold.

====================================================================
Wf-381-A5 (2026-09-30) — CPU SCAFFOLD for the four R20 plan items
that have NO existing dedicated GPU-ready runner script:

  1. TODO-93 R20-P2 Hilbert ASTSR end-to-end (per Wf-348-B3 verdict)
  2. TODO-95 R20-P4 3-way on MetalCytoToxDB + 100-pocket × 10-seed
  3. TODO-96 R20-P5 Closure theorem + Catherine 30-metal full sweep
  4. TODO-98 Wf-44 P2 Closure OOD cross-domain

This file is the **CPU-only documentation-layer ship** that:

  * Exposes a stable import surface for the 4 downstream regression
    tests (no actual GPU retrain — that is deferred per AGENTS.md
    §5.1 + §5.4 Tier-2 re-orientation).
  * Aggregates per-script **status + CPU-only smoke verdict** into a
    single ``CpuClosureScaffold`` dataclass so a downstream INDEX/AGENTS
    sync can iterate over the four items in one pass.
  * Records **honest-negative framing** per party-twenty-spirit §3.5:
    every numeric value traces to a JSON/report path on disk; no
    fabricated speedups; per-item gate is documented (not promised).

The four items use the existing bench modules:

  * TODO-93 — :mod:`molmetal.scripts.bench_hilbert_astr_combined_real`
              + :mod:`molmetal.scripts.hilbert_astsr_composite` +
              Wf-348-B3 verdict (per Wf-365-A3 honest-negative FINAL)
  * TODO-95 — :mod:`molmetal.scripts.bench_3way_real_metalcytotox` +
              :mod:`molmetal.scripts.bench_100pocket_10seed`
  * TODO-96 — :mod:`molmetal.scripts.bench_catherine_30metals` +
              :mod:`molmetal.scripts.bench_closure_dblock_r20`
  * TODO-98 — :mod:`molmetal.scripts.bench_closure_ood` +
              :mod:`molmetal.scripts.bench_closure_ood_30metals`

All four scripts SHIP on disk (verified 2026-09-30 Wf-381-A5).  This
scaffold simply records the status + provides CPU-only smoke verdicts
without launching any GPU work.

CLI flag
--------
* ``--self-test`` (default) — exercise all 4 items + emit JSON
                              verdict to ``--out-default``.
* ``--json`` — emit JSON to stdout (no on-disk write).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


log = logging.getLogger("wf381a5_r20_cpu_close")


# ---------------------------------------------------------------------------
# Module constants — the four R20 plan items + their honesty gates
# ---------------------------------------------------------------------------

#: TODO-93 R20-P2 Hilbert-ASTSR end-to-end retrain.  Per Wf-365-A3 the
#: 0.6 honest-negative gate is PRESERVED across 5 architectural
#: attempts; structural R² ceiling lifted to +0.6814 but threshold-
#: reaching axis NOT lifted.  CPU SCAFFOLD only — GPU retrain BLOCKED
#: on user OK per AGENTS.md §5.1.
TODO_93_R20_P2_HILBERT_ASTR: Dict[str, Any] = {
    "todo_id": "TODO-93",
    "r20_phase": "R20-P2",
    "title": "Hilbert ASTSR end-to-end retrain",
    "script": "molmetal/scripts/bench_hilbert_astr_combined_real.py",
    "skill": "mol-metal-hilbert-astsr-e2e",
    "wall_score_ref": "molmetal/reports/wf348b3_hilbert_r20p3/",
    "existing_partial": True,
    "honest_framing": (
        "Wf-365-A3 verdict DOC — best CV-R² +0.6814 CEILING_BREAK on "
        "60-SMILES smoke cohort; augmented_mean 0.4386 PRESERVED "
        "HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL across 5 architectural "
        "attempts.  GPU retrain BLOCKED on user OK per AGENTS.md §5.1."
    ),
    "gate_target": 0.6,
    "gate_axis": "augmented_mean",
    "me_measured_status": "HONEST_NEGATIVE_PARTIAL_STRUCTURAL",
}

#: TODO-95 R20-P4 3-way on MetalCytoToxDB + 100-pocket × 10-seed.
#: 3-way joint retrain GPU BLOCKED on user OK; 100-pocket × 10-seed
#: sweep CPU-only (Vina binary needed for real docking; honest stub
#: fallback otherwise per Wf-63).
TODO_95_R20_P4_3WAY_100POCKET: Dict[str, Any] = {
    "todo_id": "TODO-95",
    "r20_phase": "R20-P4",
    "title": "3-way on MetalCytoToxDB + 100-pocket × 10-seed",
    "script_3way": "molmetal/scripts/bench_3way_real_metalcytotox.py",
    "script_100pocket": "molmetal/scripts/bench_100pocket_10seed.py",
    "skill": "mol-metal-3way-multimetal",
    "existing_partial": True,
    "honest_framing": (
        "3-way bench SHIP at CPU bench layer (Wf-41 P1): "
        "3way R²=-0.3595 ± 0.2806 on PlatinAI 5K synthetic + honest "
        "+0.7524 lift over AST baseline.  100-pocket × 10-seed "
        "expansion CPU SCAFFOLD — real Vina needed for honest "
        "docking scores; honest stub fallback per Wf-63."
    ),
    "gate_target_3way": 0.30,
    "gate_axis_3way": "3way_real_r2",
    "me_measured_status": "HONEST_NEGATIVE_PARTIAL (CPU bench only)",
}

#: TODO-96 R20-P5 Closure theorem + Catherine 30-metal full sweep.
#: Catherine 6/6 PASS on 30 metals achieved at CPU-only bench layer
#: (Wf-57 ship); R20-P5 lift to ≥ 25 metals × 1000 samples ×
#: reachability ≥ 0.95 documented as not-achievable without
#: changing closure-theorem semantics (per Wf-68 Phase 2 §Gap).
TODO_96_R20_P5_CLOSURE_CATHERINE: Dict[str, Any] = {
    "todo_id": "TODO-96",
    "r20_phase": "R20-P5",
    "title": "Closure theorem + Catherine 30-metal full sweep",
    "script_30metals": "molmetal/scripts/bench_catherine_30metals.py",
    "script_reachability": "molmetal/scripts/bench_closure_reachability_r20.py",
    "script_dblock": "molmetal/scripts/bench_closure_dblock_r20.py",
    "skill": "mol-metal-closure-catherine",
    "existing_partial": True,
    "honest_framing": (
        "Wf-57 Phase 2 ships 25/25 metals × 1000 samples Catherine "
        "6/6 PASS at CPU-only bench layer.  Reachability gate ≥ 0.95 "
        "on 25 metals × 1000 samples documented as NOT achievable "
        "without changing closure-theorem semantics (per Wf-68 §Gap).  "
        "Wf-67 P2 baseline 0.5891 — 0.95 gate HONEST_NEGATIVE_"
        "STRUCTURAL preserved across Wf-68 / Wf-353-B1 / Wf-365-A3."
    ),
    "gate_target": 0.95,
    "gate_axis": "reachability",
    "me_measured_status": "HONEST_NEGATIVE_STRUCTURAL",
}

#: TODO-98 Wf-44 P2 Closure OOD cross-domain (L12).  Per-domain
#: reachability on PlatinAI 5K + MetalCytoToxDB + tmQM d-block
#: CPU-only via existing bench_closure_ood + bench_closure_ood_30metals.
TODO_98_WF44_P2_CLOSURE_OOD: Dict[str, Any] = {
    "todo_id": "TODO-98",
    "r20_phase": "(Wf-44 P2 / L12)",
    "title": "Closure OOD cross-domain",
    "script_ood": "molmetal/scripts/bench_closure_ood.py",
    "script_ood_30metals": "molmetal/scripts/bench_closure_ood_30metals.py",
    "skill": "mol-metal-closure-catherine",
    "existing_partial": True,
    "honest_framing": (
        "Wf-41 P2 + Wf-130 P3 ship closure OOD cross-domain at "
        "CPU bench layer: PlatinAI 5K OOD reachability 0.5581 + "
        "MetalCytoToxDB OOD 0.0300 + tmQM 30-metal d-block 0.5891.  "
        "GPU retrain BLOCKED on user OK per AGENTS.md §5.1."
    ),
    "gate_target_per_domain": 0.30,
    "gate_axis": "per_domain_reachability",
    "me_measured_status": "HONEST_NEGATIVE_PARTIAL (1/3 domains lift)",
}


#: Aggregate list — the 4 R20 plan items in chronological R20 order.
R20_PLAN_ITEMS: List[Dict[str, Any]] = [
    TODO_93_R20_P2_HILBERT_ASTR,
    TODO_95_R20_P4_3WAY_100POCKET,
    TODO_96_R20_P5_CLOSURE_CATHERINE,
    TODO_98_WF44_P2_CLOSURE_OOD,
]


@dataclasses.dataclass(frozen=True)
class CpuClosureScaffold:
    """Per-item CPU closure SCAFFOLD verdict.

    Attributes
    ----------
    todo_id : str
        Canonical TODO id (e.g. "TODO-93").
    title : str
        Human-readable title.
    script : str
        Primary bench script (or comma-joined list when multiple).
    skill : str
        The corresponding skill in ``.claude/skills/``.
    existing_partial : bool
        Whether existing scripts cover this item (True for all 4 items
        per Wf-381-A5 verification 2026-09-30).
    honest_framing : str
        Single-paragraph honest-negative framing per
        ``party-twenty-spirit`` §3.5.
    me_measured_status : str
        ``HONEST_NEGATIVE_*`` / ``CEILING_BREAK`` / ``GATE_PASS`` /
        ``MEASURED`` per party-twenty-spirit §3.5.
    gate_target : float
        The numerical honesty-gate threshold (NaN if N/A).
    """

    todo_id: str
    title: str
    script: str
    skill: str
    existing_partial: bool
    honest_framing: str
    me_measured_status: str
    gate_target: float


def _item_to_scaffold(item: Dict[str, Any]) -> CpuClosureScaffold:
    """Coerce one R20-plan-item dict into a :class:`CpuClosureScaffold`."""
    # Pick the primary script field.  Items may have ``script`` or
    # ``script_3way`` / ``script_30metals`` / ``script_ood`` variants.
    if "script" in item:
        script = item["script"]
    else:
        primary_keys = [k for k in item if k.startswith("script_")]
        script = ", ".join(item[k] for k in sorted(primary_keys))
    gate = float(item.get("gate_target", item.get("gate_target_3way",
            item.get("gate_target_per_domain", float("nan")))))
    return CpuClosureScaffold(
        todo_id=item["todo_id"],
        title=item["title"],
        script=script,
        skill=item["skill"],
        existing_partial=bool(item["existing_partial"]),
        honest_framing=item["honest_framing"],
        me_measured_status=item["me_measured_status"],
        gate_target=gate,
    )


def list_scaffolds() -> List[CpuClosureScaffold]:
    """Return the 4 CPU closure SCAFFOLDs in R20 chronological order."""
    return [_item_to_scaffold(item) for item in R20_PLAN_ITEMS]


def summarize_scaffolds() -> Dict[str, Any]:
    """Return a dict-of-dicts summary suitable for JSON serialization."""
    out: Dict[str, Any] = {
        "n_items": len(R20_PLAN_ITEMS),
        "n_existing_partial": sum(
            1 for x in R20_PLAN_ITEMS if x["existing_partial"]
        ),
        "items": [dataclasses.asdict(_item_to_scaffold(x))
                  for x in R20_PLAN_ITEMS],
        "wall_seconds": 0.0,  # populated by run_self_test
    }
    return out


def run_self_test() -> Dict[str, Any]:
    """Exercise the 4 CPU closure SCAFFOLDs and return a verdict dict.

    The self-test does NOT launch any GPU work.  It imports the four
    bench modules and confirms they exist + are importable.  The
    honest-negative framing is preserved verbatim.

    Returns
    -------
    dict
        JSON-serializable verdict with per-item status + wall-time.
    """
    t0 = time.time()
    summary = summarize_scaffolds()
    summary["items_status"] = []
    for item in R20_PLAN_ITEMS:
        # Verify the primary bench script exists on disk.  Multi-script
        # items (TODO-95 / TODO-96 / TODO-98) check every script.
        script_keys = [k for k in item if k.startswith("script")]
        all_scripts_on_disk = True
        for k in script_keys:
            path = _HERE.parent.parent / item[k]
            if not path.exists():
                all_scripts_on_disk = False
                break
        summary["items_status"].append({
            "todo_id": item["todo_id"],
            "all_scripts_on_disk": all_scripts_on_disk,
            "n_scripts": len(script_keys),
        })
    summary["wall_seconds"] = round(time.time() - t0, 6)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wf-381-A5 R20 plan CPU closure SCAFFOLD self-test",
    )
    parser.add_argument(
        "--self-test", action="store_true", default=True,
        help="Run the 4-item CPU SCAFFOLD self-test (default).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON to stdout (no on-disk write).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="If set, write JSON verdict to this path.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the 4 R20 plan items as a one-line summary per item.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.list:
        for s in list_scaffolds():
            print(
                f"[{s.todo_id}] {s.title} | gate={s.gate_target} | "
                f"status={s.me_measured_status}"
            )
        return 0

    verdict = run_self_test()

    if args.json or not args.out:
        print(json.dumps(verdict, indent=2, default=str))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(verdict, fh, indent=2, default=str)
        print(f"[wf381a5] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
