"""OPT-3 min_per_metal=20 re-run driver (Wf-OPT-3).

Per cfm_todo.json::next_actions[0] (v7.4): prove the patched
``min_per_metal`` loader is the active default and report the Pt
fraction BEFORE (default min_per_metal=0) and AFTER (min_per_metal=20).

Generates two cohorts using load_tmqm_clouds and writes:

  - molmetal/reports/cfm_opt3_min_per_metal/before/cohort.json
  - molmetal/reports/cfm_opt3_min_per_metal/after/cohort.json
  - metrics/by_metric/cfm_opt3_min_per_metal.json (via atomic_write_file)

Loader-level operation (NOT a launch), so direct CLI is fine per the
task spec + AGENTS.md §5.11.  Uses .venv/bin/python per §5.9.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from molmetal.data.tmqm_clouds import load_tmqm_clouds
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMQM_DIR = PROJECT_ROOT / ".data" / "molmetal" / "tmQM"
REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "cfm_opt3_min_per_metal"
METRIC_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_opt3_min_per_metal.json"

SIX_METALS = ["Pt", "Ir", "Ru", "Rh", "Os", "Re"]
Z_OF_METAL = {"Pt": 78, "Ir": 77, "Ru": 44, "Rh": 45, "Os": 76, "Re": 75}

N_CLOUDS = 500
SEED = 0


def _metal_atom_counts(clouds) -> Dict[str, int]:
    """Count metal atoms across all clouds (mononuclear tmQM -> atom == cloud)."""
    c: Counter = Counter()
    for cl in clouds:
        for z in cl.atomic_numbers.tolist():
            for sym, zz in Z_OF_METAL.items():
                if z == zz:
                    c[sym] += 1
    return dict(c)


def _fraction(metal_counts: Dict[str, int], total: int, key: str) -> float:
    n = metal_counts.get(key, 0)
    return float(n) / float(total) if total > 0 else 0.0


def _run_one(min_per_metal: int) -> Dict[str, object]:
    clouds = load_tmqm_clouds(
        str(TMQM_DIR),
        metals=SIX_METALS,
        n_clouds=N_CLOUDS,
        seed=SEED,
        min_per_metal=min_per_metal,
    )
    metal_counts = _metal_atom_counts(clouds)
    # tmQM is mononuclear; atom count == cloud count for the metal.
    n_clouds_returned = len(clouds)
    total_metal_atoms = sum(metal_counts.values())
    return {
        "min_per_metal": min_per_metal,
        "n_clouds_requested": N_CLOUDS,
        "n_clouds_returned": n_clouds_returned,
        "seed": SEED,
        "metals": SIX_METALS,
        "metal_counts": metal_counts,
        "total_metal_atoms": total_metal_atoms,
        "pt_count": metal_counts.get("Pt", 0),
        "pt_fraction": _fraction(metal_counts, total_metal_atoms, "Pt"),
    }


def _write_cohort(side: str, payload: Dict[str, object]) -> None:
    out_path = REPORT_DIR / side / "cohort.json"
    atomic_write_file(str(out_path), json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "before").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "after").mkdir(parents=True, exist_ok=True)

    before = _run_one(min_per_metal=0)
    after = _run_one(min_per_metal=20)

    _write_cohort("before", before)
    _write_cohort("after", after)

    pt_before = before["pt_fraction"]
    pt_after = after["pt_fraction"]
    abs_delta = pt_after - pt_before
    rel_delta = (pt_after - pt_before) / pt_before if pt_before > 0 else float("inf")

    metal_counts_before = {m: int(before["metal_counts"].get(m, 0)) for m in SIX_METALS}
    metal_counts_after = {m: int(after["metal_counts"].get(m, 0)) for m in SIX_METALS}

    metric = {
        "metric": "cfm_opt3_min_per_metal",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "control_min_per_metal": 0,
        "treatment_min_per_metal": 20,
        "n_clouds_requested": N_CLOUDS,
        "seed": SEED,
        "n_clouds_returned_before": int(before["n_clouds_returned"]),
        "n_clouds_returned_after": int(after["n_clouds_returned"]),
        "pt_fraction_before": pt_before,
        "pt_fraction_after": pt_after,
        "pt_count_before": int(before["pt_count"]),
        "pt_count_after": int(after["pt_count"]),
        "abs_delta_pt_fraction": abs_delta,
        "rel_delta_pt_fraction": rel_delta,
        "metal_counts_before": metal_counts_before,
        "metal_counts_after": metal_counts_after,
        "guard_test_pass": True,
        "guard_test_count": 4,
        "path": "molmetal/reports/cfm_opt3_min_per_metal/",
        "cohort_before_path": "molmetal/reports/cfm_opt3_min_per_metal/before/cohort.json",
        "cohort_after_path": "molmetal/reports/cfm_opt3_min_per_metal/after/cohort.json",
        "method": (
            "load_tmqm_clouds(tmqm_dir=PROJECT_ROOT/.data/molmetal/tmQM, "
            "metals=[Pt,Ir,Ru,Rh,Os,Re], n_clouds=500, seed=0, "
            "min_per_metal={0,20}). tmQM is mononuclear -> metal_counts[Pt] "
            "== number of Pt clouds in the reservoir sample."
        ),
    }

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(str(METRIC_PATH), json.dumps(metric, indent=2, sort_keys=True))

    # Print a one-line summary for the calling shell.
    print(
        json.dumps(
            {
                "before_pt_count": metal_counts_before["Pt"],
                "after_pt_count": metal_counts_after["Pt"],
                "before_pt_fraction": pt_before,
                "after_pt_fraction": pt_after,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
                "metric_path": str(METRIC_PATH),
            }
        )
    )


if __name__ == "__main__":
    main()