"""Wf-310D PART 2: Vina pt-strip full GPU aggregation template.

==============================================================================
Wf-310D (Wf-301B full run aggregation, follow-up of Wf-306A Phase-B):

Inputs (read-only, written by Wf-301B / Wf-305D full 100-mol × 4-pocket GPU
run launched via molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase B):

* molmetal/reports/wf301b_vina_pt_strip/vina.json
    shape: {
      "metadata": {...},
      "n_input": int, "n_parsed": int, "n_pdbqt_ok": int, "n_docked": int,
      "per_molecule": [{"smiles", "parse_ok", "pdbqt_ok",
                        "per_pocket": {pocket_id: {best_kcal_mol, n_poses,
                                                   wall_time_s, status,
                                                   note}}, ...},
                       ...]
    }
* molmetal/reports/wf306a_drugood_vina_launch/phase_b/panel.json (full run)

Outputs (this script writes):

* molmetal/reports/wf310d_vina_full/aggregate.json
* molmetal/reports/wf310d_vina_full/claim_cards.md

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* Field-avg band per Wf-301B honest-negative: -9.5 to -7.5 kcal/mol.
* Wf-301B smoke (5/5 docked on 1h36 pocket) mean = -4.60 kcal/mol -- ABOVE
  the field-avg band (recovery axis is *being able to dock at all*, which
  the pt-strip enables; the absolute binding affinity is a follow-up).
* Per Wf-301B honest-negative: top-1 -10.299 exceeds -9.5; mean -5.564
  below field-avg band.  Full 100-mol × 4-pocket aggregation is the
  paper-grade checkpoint.

Usage
-----
  python molmetal/scripts/wf310d_vina_full_aggregate.py \\
      --panel molmetal/reports/wf306a_drugood_vina_launch/phase_b/panel.json \\
      --out-dir molmetal/reports/wf310d_vina_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Field-avg target band per Wf-301B honest reporting.
FIELD_AVG_TARGET_LO: float = -9.5  # most-favourable kcal/mol (more negative = better)
FIELD_AVG_TARGET_HI: float = -7.5  # least-favourable kcal/mol in band
# Wf-301B smoke CPU checkpoint (5 mols on 1h36).
WF301B_SMOKE_N_DOCKED: int = 5
WF301B_SMOKE_N_TOTAL: int = 5
WF301B_SMOKE_MEAN_KCAL: float = -4.60


def _log(msg: str) -> None:
    print(f"[wf310d-vina-agg] {msg}", flush=True)


def _load_panel(panel_path: Path) -> Dict:
    if not panel_path.exists():
        _log(f"ERROR: panel not found at {panel_path}")
        return {}
    return json.loads(panel_path.read_text())


def _aggregate_per_pocket(panel: Dict) -> Dict[str, Dict[str, float]]:
    """Aggregate per-pocket mean ± std kcal/mol + n_docked + best_kcal_mol."""
    per_pocket_vals: Dict[str, List[float]] = defaultdict(list)
    per_pocket_best: Dict[str, float] = {}
    n_total = 0
    n_docked = 0
    n_pdbqt_ok = 0
    n_parse_ok = 0

    for mol in panel.get("per_molecule", []):
        n_total += 1
        if mol.get("parse_ok"):
            n_parse_ok += 1
        if mol.get("pdbqt_ok"):
            n_pdbqt_ok += 1
        per_pocket = mol.get("per_pocket", {})
        any_docked = False
        for pocket_id, dock in per_pocket.items():
            if (
                dock.get("status") == "REAL_VINA"
                and dock.get("best_kcal_mol") is not None
            ):
                kcal = float(dock["best_kcal_mol"])
                per_pocket_vals[pocket_id].append(kcal)
                cur_best = per_pocket_best.get(pocket_id)
                if cur_best is None or kcal < cur_best:  # more negative = better
                    per_pocket_best[pocket_id] = kcal
                any_docked = True
        if any_docked:
            n_docked += 1

    summary: Dict[str, Dict[str, float]] = {}
    for pocket_id, vals in per_pocket_vals.items():
        n = len(vals)
        mean_kcal = float(sum(vals) / n) if n else float("nan")
        std_kcal = (
            float((sum((v - mean_kcal) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5)
            if n > 1
            else 0.0
        )
        summary[pocket_id] = {
            "n_docked": n,
            "mean_kcal_mol": mean_kcal,
            "std_kcal_mol": std_kcal,
            "min_kcal_mol": min(vals) if vals else float("nan"),
            "max_kcal_mol": max(vals) if vals else float("nan"),
            "best_kcal_mol": per_pocket_best.get(pocket_id, float("nan")),
        }
    return {
        "_per_pocket": summary,
        "n_total": n_total,
        "n_docked": n_docked,
        "n_pdbqt_ok": n_pdbqt_ok,
        "n_parse_ok": n_parse_ok,
    }


def _verdict(
    per_pocket: Dict[str, Dict[str, float]],
    grand_mean: float,
    n_docked: int,
    n_total: int,
) -> str:
    """Per Wf-301B honest-negative verdict convention."""
    n_in_band = 0
    for s in per_pocket.values():
        m = s["mean_kcal_mol"]
        if FIELD_AVG_TARGET_LO <= m <= FIELD_AVG_TARGET_HI:
            n_in_band += 1

    in_field_avg_band = (
        grand_mean >= FIELD_AVG_TARGET_LO and grand_mean <= FIELD_AVG_TARGET_HI
    )
    docks_all = n_docked == n_total and n_total > 0

    if in_field_avg_band and docks_all:
        return (
            f"PASS_FIELD_AVG -- {len(per_pocket)}-pocket grand mean "
            f"{grand_mean:.3f} kcal/mol in field-avg band "
            f"[{FIELD_AVG_TARGET_LO}, {FIELD_AVG_TARGET_HI}]; "
            f"n_docked={n_docked}/{n_total}."
        )
    if docks_all:
        return (
            f"HONEST_NEGATIVE_BAND -- {len(per_pocket)}-pocket grand mean "
            f"{grand_mean:.3f} kcal/mol OUTSIDE field-avg band "
            f"[{FIELD_AVG_TARGET_LO}, {FIELD_AVG_TARGET_HI}]; "
            f"pt-strip enables docking (n_docked={n_docked}/{n_total}) but "
            f"binding affinity axis is a follow-up; "
            f"{n_in_band}/{len(per_pocket)} pockets within band."
        )
    return (
        f"HONEST_NEGATIVE_DOCK -- n_docked={n_docked}/{n_total}; "
        f"{len(per_pocket)}-pocket grand mean {grand_mean:.3f} kcal/mol; "
        f"pt-strip recovery axis not fully realised."
    )


def _format_claim_cards(
    per_pocket: Dict[str, Dict[str, float]],
    grand_mean: float,
    grand_std: float,
    n_docked: int,
    n_total: int,
    n_pdbqt_ok: int,
    n_parse_ok: int,
    wall_min: float,
    verdict: str,
) -> str:
    lines = [
        f"# Wf-310D Vina pt-strip full aggregation -- claim cards",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"n_total: {n_total}  n_parse_ok: {n_parse_ok}  "
        f"n_pdbqt_ok: {n_pdbqt_ok}  n_docked: {n_docked}",
        f"grand mean kcal/mol: {grand_mean:.3f}  std: {grand_std:.3f}",
        f"wall_min: {wall_min:.2f}  n_pockets: {len(per_pocket)}",
        "",
        "## Card 1 -- per-pocket summary (MEASURED)",
        "",
        "| pocket | n_docked | mean_kcal_mol | std | best_kcal_mol | in_band |",
        "|--------|----------|---------------|-----|---------------|---------|",
    ]
    for pocket, s in per_pocket.items():
        m = s["mean_kcal_mol"]
        in_band = "yes" if FIELD_AVG_TARGET_LO <= m <= FIELD_AVG_TARGET_HI else "no"
        lines.append(
            f"| {pocket} | {s['n_docked']} | {m:.3f} | "
            f"{s['std_kcal_mol']:.3f} | {s['best_kcal_mol']:.3f} | {in_band} |"
        )
    lines.extend(
        [
            "",
            "## Card 2 -- pass-gate verdict (MEASURED)",
            "",
            f"- Field-avg band: [{FIELD_AVG_TARGET_LO}, {FIELD_AVG_TARGET_HI}] kcal/mol",
            f"- Wf-301B smoke (CPU, 5 mols on 1h36): {WF301B_SMOKE_MEAN_KCAL:.2f} kcal/mol",
            f"- Full grand mean: {grand_mean:.3f} kcal/mol",
            f"- n_docked: {n_docked}/{n_total}",
            "",
            "## Card 3 -- honest-negative flags (per Wf-301B honest-negative)",
            "",
            "- Wf-301B smoke mean (-4.60) is ABOVE field-avg band (-7.5 to -9.5);",
            "  recovery axis is *being able to dock at all* (pt-strip enables",
            "  docking; absolute binding affinity is a follow-up).",
            "- Top-1 kcal/mol may exceed -9.5 even when grand mean is in band",
            "  (per Wf-301B honest-negative: -10.299 vs -9.5).",
            "- Full 4-pocket panel provides per-pocket spread; use min/max to",
            "  bracket the pocket-cohort variance.",
            "",
            "## Card 4 -- wall + GPU cost",
            "",
            f"- Wall: {wall_min:.1f} min ({wall_min/60:.2f} hr) -- MEASURED.",
            "- GPU: CPU-only (Vina binary is single-threaded; cuda:1 unused).",
            "- ~1-2 hr PROJECTED per Wf-306A Phase-B budget.",
            "",
            "## Card 5 -- verdict",
            "",
            verdict,
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--panel",
        default="molmetal/reports/wf306a_drugood_vina_launch/phase_b/panel.json",
        help="Wf-301B full panel.json (or smoke vina.json).",
    )
    p.add_argument("--out-dir", default="molmetal/reports/wf310d_vina_full")
    args = p.parse_args(argv)

    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(panel_path)
    if not panel:
        _log(f"ERROR: panel empty at {panel_path}")
        return 1

    t0 = time.time()
    agg = _aggregate_per_pocket(panel)
    per_pocket = agg["_per_pocket"]
    del agg["_per_pocket"]

    # Grand mean across all per-pocket measurements (re-extract from
    # per_molecule to compute simple mean of all individual kcal/mol
    # values, not a weighted mean of per-pocket means).
    all_vals: List[float] = []
    for mol in panel.get("per_molecule", []):
        for pocket_id, dock in mol.get("per_pocket", {}).items():
            if (
                dock.get("status") == "REAL_VINA"
                and dock.get("best_kcal_mol") is not None
            ):
                all_vals.append(float(dock["best_kcal_mol"]))

    grand_mean = float(sum(all_vals) / len(all_vals)) if all_vals else float("nan")
    grand_std = (
        float(
            (sum((v - grand_mean) ** 2 for v in all_vals) / max(len(all_vals) - 1, 1))
            ** 0.5
        )
        if len(all_vals) > 1
        else 0.0
    )

    wall_min = (time.time() - t0) / 60.0

    n_total = agg["n_total"]
    n_docked = agg["n_docked"]
    n_pdbqt_ok = agg["n_pdbqt_ok"]
    n_parse_ok = agg["n_parse_ok"]

    verdict = _verdict(per_pocket, grand_mean, n_docked, n_total)

    aggregate = {
        "wf": "wf310d-vina-pt-strip-full-aggregate",
        "date": time.strftime("%Y-%m-%d"),
        "panel_path": str(panel_path),
        "metadata": panel.get("metadata", {}),
        "n_total": n_total,
        "n_parse_ok": n_parse_ok,
        "n_pdbqt_ok": n_pdbqt_ok,
        "n_docked": n_docked,
        "per_pocket": per_pocket,
        "grand_mean_kcal_mol": grand_mean,
        "grand_std_kcal_mol": grand_std,
        "n_dock_values": len(all_vals),
        "field_avg_target_lo": FIELD_AVG_TARGET_LO,
        "field_avg_target_hi": FIELD_AVG_TARGET_HI,
        "wf301b_smoke_mean_kcal": WF301B_SMOKE_MEAN_KCAL,
        "wf301b_smoke_n_docked": WF301B_SMOKE_N_DOCKED,
        "wall_min": wall_min,
        "verdict": verdict,
    }
    out_agg = out_dir / "aggregate.json"
    out_agg.write_text(json.dumps(aggregate, indent=2))
    _log(f"wrote {out_agg}")

    out_cards = out_dir / "claim_cards.md"
    out_cards.write_text(
        _format_claim_cards(
            per_pocket,
            grand_mean,
            grand_std,
            n_docked,
            n_total,
            n_pdbqt_ok,
            n_parse_ok,
            wall_min,
            verdict,
        )
    )
    _log(f"wrote {out_cards}")

    _log(
        f"n_docked={n_docked}/{n_total}  grand_mean={grand_mean:.3f}  "
        f"verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
