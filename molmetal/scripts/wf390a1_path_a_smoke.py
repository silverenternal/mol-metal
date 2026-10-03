#!/usr/bin/env python3
"""Wf-390-A1 — m2 R² Path A 5-metal subset smoke (CPU-only).

================================================================
Wf-390-A1 — CPU-only smoke for the 5-metal subset retrain path
(Path A in `TODO/pending/376_m2_r2_gpu_retrain_plan.md`).

This script verifies that the existing trainer
``molmetal/scripts/train_dmpnn_multitask_wasserstein.py`` accepts the
5-metal subset ``{Ru, Ir, Rh, Os, Re}`` natively — i.e., does NOT
violate the ``FIVE_METALS`` argparse CHOICES constraint at line 881.

Per Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL: the wrapper
``scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`` requests
11 metals (Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au) which would fail
argparse on first invocation. Path A works WITHIN the existing
trainer constraint by using only the 5 metals the trainer natively
supports.

This smoke is SCAFFOLD mode (CPU-only):
* Verifies FIVE_METALS subset parses cleanly (no SystemExit 2)
* Verifies the Wf-379-A2 11-metal set fails with SystemExit 2
* Emits per-cohort config JSON to ``molmetal/reports/wf390a1_path_a/``
* 0 GPU launches; 0 paper/* touches (TODO-84 honored)

Usage
-----
    # Smoke (default)
    PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/wf390a1_path_a_smoke.py

    # Custom output directory
    PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/wf390a1_path_a_smoke.py \\
        --out-dir /tmp/wf390a1_path_a_smoke

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* This is a SCAFFOLD smoke — verifies CLI + emits per-cohort config
  JSON. No GPU launches, no trainer invocation, no model state.
* Actual GPU retrain is USER_OK-gated per AGENTS.md §5.1 +
  Wf-355-B1 wrapper pattern.
* Per Wf-316A: the 12-d RDKit 2D signal floor is a structural
  ceiling; Path A may lift CEILING_BREAK on ``metalcytotox_combined``
  (n=2000) but is NOT guaranteed to lift the small n=200 cohorts
  past 0 R².
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants (per Wf-122 P1.3 + Wf-335-M2 + Wf-355-B1)
# ---------------------------------------------------------------------------
TRAINER_PATH = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)

# Path A 5-metal subset (matches trainer FIVE_METALS exactly)
PATH_A_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")

# Wf-379-A2 11-metal set (would FAIL argparse; preserved for negative test)
WF379A2_METALS: Tuple[str, ...] = (
    "Cr",
    "Mn",
    "Cu",
    "Zn",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Pt",
    "Ir",
    "Au",
)

# Path A per-cohort config (per Wf-380-A2 + Wf-388-A1 SCAFFOLD/APPLIED pattern)
PATH_A_COHORT_CONFIGS: List[Dict[str, object]] = [
    {
        "cohort_id": "metalcytotox_combined",
        "n_smiles": 2000,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
        ),
        "metals": list(PATH_A_METALS),
        "seeds": [0, 1, 2, 3, 4],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "mtl_baseline",
        "single_task": False,
        "baseline_r2_measured": 0.3196,
        "baseline_source": "Wf-335-M2 CEILING_BREAK",
        "projected_r2_low": 0.40,
        "projected_r2_high": 0.55,
        "wall_projected_hr": 4.0,
        "vram_projected_gb": 10.0,
    },
    {
        "cohort_id": "metalcytotox_5metal_multitask_v4",
        "n_smiles": 26800,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
        ),
        "metals": list(PATH_A_METALS),
        "seeds": [0, 1, 2, 3, 4],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "mtl_baseline",
        "single_task": False,
        "baseline_r2_measured": None,
        "baseline_source": "new axis — no prior R² MEASURED",
        "projected_r2_low": 0.30,
        "projected_r2_high": 0.50,
        "wall_projected_hr": 6.0,
        "vram_projected_gb": 12.0,
    },
    {
        "cohort_id": "per_metal_singlets_Ru",
        "n_smiles": 5000,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_Ru_only.csv"
        ),
        "metals": ["Ru"],
        "seeds": [0],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "single_task",
        "single_task": True,
        "baseline_r2_measured": None,
        "baseline_source": "new axis — no prior R² MEASURED",
        "projected_r2_low": 0.20,
        "projected_r2_high": 0.40,
        "wall_projected_hr": 0.4,
        "vram_projected_gb": 8.0,
    },
    {
        "cohort_id": "per_metal_singlets_Ir",
        "n_smiles": 5000,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_Ir_only.csv"
        ),
        "metals": ["Ir"],
        "seeds": [0],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "single_task",
        "single_task": True,
        "baseline_r2_measured": None,
        "baseline_source": "new axis — no prior R² MEASURED",
        "projected_r2_low": 0.20,
        "projected_r2_high": 0.40,
        "wall_projected_hr": 0.4,
        "vram_projected_gb": 8.0,
    },
    {
        "cohort_id": "per_metal_singlets_Rh",
        "n_smiles": 1200,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_Rh_only.csv"
        ),
        "metals": ["Rh"],
        "seeds": [0],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "single_task",
        "single_task": True,
        "baseline_r2_measured": None,
        "baseline_source": "new axis — no prior R² MEASURED",
        "projected_r2_low": -0.10,
        "projected_r2_high": 0.10,
        "wall_projected_hr": 0.4,
        "vram_projected_gb": 8.0,
    },
    {
        "cohort_id": "per_metal_singlets_Os",
        "n_smiles": 5000,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_Os_only.csv"
        ),
        "metals": ["Os"],
        "seeds": [0],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "single_task",
        "single_task": True,
        "baseline_r2_measured": None,
        "baseline_source": "new axis — no prior R² MEASURED",
        "projected_r2_low": 0.20,
        "projected_r2_high": 0.40,
        "wall_projected_hr": 0.4,
        "vram_projected_gb": 8.0,
    },
    {
        "cohort_id": "per_metal_singlets_Re",
        "n_smiles": 800,
        "csv_path": str(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_Re_only.csv"
        ),
        "metals": ["Re"],
        "seeds": [0],
        "epochs": 50,
        "hidden": 300,
        "depth": 3,
        "w2_weight": 0.05,
        "alpha": 0.5,
        "ablation_config": "single_task",
        "single_task": True,
        "baseline_r2_measured": None,
        "baseline_source": "new axis — no prior R² MEASURED",
        "projected_r2_low": -0.20,
        "projected_r2_high": 0.05,
        "wall_projected_hr": 0.4,
        "vram_projected_gb": 8.0,
    },
]


# ---------------------------------------------------------------------------
# Trainer integration smoke (CPU-only)
# ---------------------------------------------------------------------------
def import_trainer():
    """Import the trainer module + return (FIVE_METALS, parse_args)."""
    import importlib.util
    import sys as _sys

    module_name = "wf390a1_trainer_module"
    spec = importlib.util.spec_from_file_location(module_name, str(TRAINER_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer spec from {TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module — the trainer uses
    # @dataclass which introspects sys.modules[cls.__module__]; without
    # this registration dataclass fails with 'NoneType has no __dict__'.
    _sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.FIVE_METALS, mod.parse_args


def smoke_path_a_5metal() -> Dict[str, object]:
    """Verify FIVE_METALS subset parses cleanly (no CHOICES violation).

    Returns a dict with:
    * ``five_metals_tuple`` — the canonical FIVE_METALS from trainer
    * ``path_a_metals_parsed`` — argparse-validated list of Path A metals
    * ``path_a_metals_match`` — True iff trainer FIVE_METALS == PATH_A_METALS
    * ``wf379a2_11metal_systemexit_code`` — expected 2 (argparse choices
      violation per Wf-379-A2 §1.1)
    * ``smoke_passed`` — True iff all 3 checks PASS
    """
    FIVE_METALS, parse_args = import_trainer()

    # Check 1: FIVE_METALS tuple exactly matches PATH_A_METALS
    path_a_match = tuple(FIVE_METALS) == PATH_A_METALS

    # Check 2: parse_args accepts the 5-metal subset cleanly
    try:
        ns = parse_args(["--metals", *PATH_A_METALS])
        path_a_parsed = list(ns.metals)
        path_a_parse_ok = path_a_parsed == list(PATH_A_METALS)
    except SystemExit:
        path_a_parse_ok = False
        path_a_parsed = []

    # Check 3: Wf-379-A2 11-metal set is REJECTED with SystemExit 2.
    # Suppress argparse's noisy usage printout via stderr redirect (the
    # rejection is the EXPECTED outcome of this negative test, not a bug).
    import contextlib
    import io as _io

    with contextlib.redirect_stderr(_io.StringIO()):
        try:
            parse_args(["--metals", *WF379A2_METALS])
            wf379a2_rejected = False
            wf379a2_exit_code = None
        except SystemExit as e:
            wf379a2_rejected = True
            wf379a2_exit_code = e.code

    smoke_passed = path_a_match and path_a_parse_ok and wf379a2_rejected

    return {
        "five_metals_tuple": list(FIVE_METALS),
        "path_a_metals_parsed": path_a_parsed,
        "path_a_metals_match": path_a_match,
        "wf379a2_11metal_rejected": wf379a2_rejected,
        "wf379a2_11metal_systemexit_code": wf379a2_exit_code,
        "smoke_passed": smoke_passed,
    }


def build_path_a_aggregate(smoke_result: Dict[str, object]) -> Dict[str, object]:
    """Build the Path A aggregate JSON (per-cohort configs + smoke result).

    Output schema (per Wf-380-A2 + Wf-388-A1 SCAFFOLD/APPLIED pattern):

    {
      "wf": "Wf-390-A1",
      "task": "m2 R² Path A 5-metal subset smoke",
      "date": "2026-09-30",
      "user_ok_required": true,
      "user_ok_gate_pattern": "Wf-355-B1 §3",
      "smoke_result": {...},
      "cohort_configs": [...],
      "total_wall_projected_hr": float,
      "total_vram_projected_gb": float,
      "n_cohort_configs": int,
      "honest_framing": {...},
      "cross_references": [...],
    }
    """
    total_wall = sum(
        float(c.get("wall_projected_hr", 0.0)) for c in PATH_A_COHORT_CONFIGS
    )
    max_vram = max(
        float(c.get("vram_projected_gb", 0.0)) for c in PATH_A_COHORT_CONFIGS
    )

    return {
        "wf": "Wf-390-A1",
        "task": "m2 R² Path A 5-metal subset smoke (CPU-only docs+plan)",
        "date": "2026-09-30",
        "user_ok_required": True,
        "user_ok_gate_pattern": "Wf-355-B1 §3 (USER_OK=1 explicit env gate)",
        "smoke_result": smoke_result,
        "cohort_configs": PATH_A_COHORT_CONFIGS,
        "total_wall_projected_hr": total_wall,
        "total_vram_projected_gb": max_vram,
        "n_cohort_configs": len(PATH_A_COHORT_CONFIGS),
        "honest_framing": {
            "scope": "CPU-only docs + smoke; 0 GPU launches; 0 paper/* touches",
            "smoke_passes_argparse_check": bool(smoke_result.get("smoke_passed")),
            "wf379a2_negative_test_validates": bool(
                smoke_result.get("wf379a2_11metal_rejected")
            ),
            "path_a_match_trainer_five_metals": bool(
                smoke_result.get("path_a_metals_match")
            ),
            "lift_values_are_projected_not_measured": True,
            "n_cohort_only_has_projected_lift": True,
            "small_cohort_ceiling_preserved": (
                "Rh (n=1.2k) and Re (n=800) PROJECTED R² remains near 0 "
                "or negative per Wf-316A 12-d RDKit signal floor + "
                "Wf-367-B1 multi-stream overfit"
            ),
            "user_ok_required_for_actual_launch": True,
            "kill_then_adapt_directive_honored": (
                "SCAFFOLD smoke does NOT launch GPU. Per AGENTS.md §5.1 + "
                "Wf-355-B1 wrapper pattern, actual launch requires "
                "USER_OK=1 + VRAM ≥12 GB + pre-flight 89/2/3/94 + "
                "wave budget USER_OK waiver."
            ),
        },
        "cross_references": [
            "TODO/pending/376_m2_r2_gpu_retrain_plan.md (this plan, Wf-390-A1)",
            "TODO/pending/376a3_m2_r2_gpu_retrain_plan.md (Path A=larger cohorts, Wf-376-A3)",
            "molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json "
            "(Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL root cause)",
            "molmetal/scripts/train_dmpnn_multitask_wasserstein.py:141 FIVE_METALS "
            "+ line 881 argparse choices=FIVE_METALS",
            "TODO/pending/179_wf123_p3_w2_pic50_alignment.md (W₂ alignment STATUS_HOLD)",
            "TODO/completed/367b1_m2_r2_recovery.md (Wf-367-B1 multi-stream HONEST_NEGATIVE)",
            "docs/M2_R2_FINAL_VERDICT.md (Wf-369-A2 FINAL VERDICT 320 LOC)",
            "scripts/launch_wf124_p3_g5.sh + scripts/launch_wf124_p3_g6.sh "
            "(Wf-366 USER_OK-gated launchers)",
            "scripts/launch_wf131_p3_deferred_runs.sh (Wf-355-B1 USER_OK-gated LOMO)",
            "AGENTS.md §5.1 (Standing GPU authorization) + §5.2 (Honest framing mandate)",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Wf-390-A1 — m2 R² Path A 5-metal subset smoke (CPU-only). "
            "Verifies trainer CLI accepts the 5-metal subset that Path A "
            "retrain would use, emits per-cohort config JSON. No GPU launch."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "molmetal" / "reports" / "wf390a1_path_a_smoke"),
        help=(
            "Output directory for the aggregate.json. "
            "Default: molmetal/reports/wf390a1_path_a_smoke/"
        ),
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / "aggregate.json"

    # Run the smoke
    smoke_result = smoke_path_a_5metal()
    aggregate = build_path_a_aggregate(smoke_result)

    # Write aggregate JSON
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=False)

    # Print summary
    summary_lines = [
        "Wf-390-A1 — m2 R² Path A 5-metal subset smoke (CPU-only)",
        "=" * 64,
        f"trainer FIVE_METALS: {smoke_result['five_metals_tuple']}",
        f"Path A 5-metal match: {smoke_result['path_a_metals_match']}",
        f"Path A 5-metal argparse OK: {bool(smoke_result['path_a_metals_parsed'])}",
        (
            f"Wf-379-A2 11-metal SystemExit: "
            f"{smoke_result['wf379a2_11metal_systemexit_code']}"
        ),
        (
            f"Wf-379-A2 11-metal rejected (expected): "
            f"{smoke_result['wf379a2_11metal_rejected']}"
        ),
        f"SMOKE PASSED: {smoke_result['smoke_passed']}",
        f"n_cohort_configs: {aggregate['n_cohort_configs']}",
        f"total_wall_projected_hr: {aggregate['total_wall_projected_hr']:.2f}",
        f"max_vram_projected_gb: {aggregate['total_vram_projected_gb']:.1f}",
        f"aggregate_json: {aggregate_path}",
    ]
    print("\n".join(summary_lines))

    return 0 if smoke_result["smoke_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
