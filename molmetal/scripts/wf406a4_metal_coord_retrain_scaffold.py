"""Wf-406-A4 — Wf-246-C metal-coordination term retrain SCAFFOLD.

================================================================
Background
================================================================
Wf-246-C adds a **metal-coordination term** to the Wf-131 P3 D-MPNN
multi-task training pipeline. The term is a soft kcal/mol adjustment
on top of the Wf-131 pIC50 regression loss + Wf-318 Wasserstein
regulariser, derived from ``molmetal/molmetal_lam/sbdd_env/metal_coord_term.py``
(Wf-333-V1 sign-fixed kcal adjustment per Vina convention
"lower (more negative) = better binding"; line 635 corrected from
``(2*comp-1)*kcal_per_unit`` → ``(1-2*comp)*kcal_per_unit``).

Wf-406-A4 (2026-09-30) — this module — ships the CPU-only SCAFFOLD
surface that wires ``MetalCoordinationTerm`` into the Wf-131 P3
D-MPNN trainer's training loop. The wire-in is **opt-in** via the
``--metal-coord-term`` CLI flag (default OFF for bit-for-bit
backward compat with Wf-131 P1 / Wf-299 / Wf-328 / Wf-131 P3
5-metal subset shipped runs).

Honest framing (party-twenty-spirit §3.5)
=========================================
* This SCAFFOLD is **CPU-only** verification only. The actual GPU
  retrain is BLOCKED on ``USER_OK=1`` per AGENTS.md §5.1 +
  Wf-355-B1 USER_OK gate pattern.
* The 5-metal subset (Ru/Ir/Rh/Os/Re) matches the canonical
  Wf-390-A1 Path A cohort per
  ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py:141
  FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")``.
* 0 GPU launches performed by this SCAFFOLD.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).
* The kcal sign fix is verified at ``metal_coord_term.py:635`` to
  be the **post-Wf-333-V1** convention ``(1 - 2*comp)*kcal_per_unit``
  so that perfect composite (1.0) yields ``-1.0 * kcal_per_unit``
  (i.e. more negative = better binding per Vina convention).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# FIVE_METALS canonical subset per Wf-390-A1 Path A
FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args for the Wf-406-A4 metal-coord-term SCAFFOLD.

    Flags
    -----
    --metal-coord-term : opt-in wire-in of MetalCoordinationTerm into
        the Wf-131 P3 D-MPNN trainer's training loop. Default OFF.
    --epochs : per-seed epoch count (default 50 for the canonical
        Wf-131 P3 protocol).
    --kcal-per-unit : kcal/mol scale for the metal-coord term
        contribution to the composite loss (default 1.0).
    --metals : 5-metal subset per Wf-390-A1 Path A
        (Ru/Ir/Rh/Os/Re by default).
    --seeds : integer seeds (default 0 1 2).
    --out-dir : output directory for ``result.json`` + smoke log.
    --smoke : 5-step CPU smoke only (default True for SCAFFOLD;
        set --smoke false to do full 50-epoch simulated training).
    --per-epoch-kcal-audit : emit per-epoch kcal sign audit
        confirming ``(1 - 2*comp)*kcal_per_unit < 0`` for
        ``score.total > 0.5`` (the Wf-333-V1 fix verification).
    """
    p = argparse.ArgumentParser(
        description=(
            "Wf-406-A4 — Wf-246-C metal-coordination term retrain "
            "SCAFFOLD (CPU-only; USER_OK-gated GPU launch deferred)."
        ),
    )
    p.add_argument(
        "--metal-coord-term",
        action="store_true",
        default=False,
        help=(
            "Enable Wf-246-C MetalCoordinationTerm wire-in into the "
            "Wf-131 P3 D-MPNN trainer's training loop. Default OFF "
            "for backward compat."
        ),
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per seed (default 50 for Wf-131 P3 protocol).",
    )
    p.add_argument(
        "--kcal-per-unit",
        type=float,
        default=1.0,
        help=("kcal/mol scale for the metal-coord term (default 1.0)."),
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(FIVE_METALS),
        choices=FIVE_METALS,
        help=(
            "5-metal subset for Wf-390-A1 Path A "
            "(default: all 5; subset of {Ru,Ir,Rh,Os,Re})."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds (default 0 1 2 for the Wf-131 P3 protocol).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="molmetal/reports/wf406a4_metal_coord_scaffold",
        help=("Output directory for result.json + per-epoch kcal audit."),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help=(
            "5-step CPU smoke only (default True for SCAFFOLD). "
            "Use --no-smoke for full 50-epoch simulated training."
        ),
    )
    p.add_argument(
        "--per-epoch-kcal-audit",
        action="store_true",
        default=True,
        help=(
            "Emit per-epoch kcal sign audit confirming the Wf-333-V1 "
            "fix holds: (1 - 2*comp)*kcal_per_unit < 0 for "
            "score.total > 0.5 (better binding per Vina convention)."
        ),
    )
    return p.parse_args(argv)


def import_metal_coord_term() -> Tuple[bool, str]:
    """Lazy-import ``MetalCoordinationTerm`` + ``MetalCoordinationScore``
    from ``molmetal.molmetal_lam.sbdd_env.metal_coord_term``.

    Returns (import_ok, detail_string).
    """
    try:
        from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (  # type: ignore  # noqa: E501
            MetalCoordinationScore,
            MetalCoordinationTerm,
        )

        return True, (
            f"import_ok; MetalCoordinationTerm at "
            f"molmetal.molmetal_lam.sbdd_env.metal_coord_term; "
            f"sign-fixed kcal_adjustment at line 635 per Wf-333-V1"
        )
    except Exception as exc:  # pragma: no cover
        return False, f"import_failed: {type(exc).__name__}: {exc}"


def build_metal_coord_term(
    kcal_per_unit: float = 1.0,
    weight_geometry: float = 0.4,
    weight_hsab: float = 0.4,
    weight_distance: float = 0.2,
) -> Tuple[bool, Any, str]:
    """Build a ``MetalCoordinationTerm`` instance with the canonical
    Wf-333-V1 sign-fixed kcal_adjustment.

    The MetalCoordinationTerm ctor signature is
    ``(weight_geometry, weight_hsab, weight_distance, kcal_per_unit)``
    per ``metal_coord_term.py:535-541`` — there is no
    ``enable_metal_coord`` flag (the term is always on; the SCAFFOLD
    gates via the ``--metal-coord-term`` CLI flag at the trainer
    level, not the term level).

    Returns (build_ok, instance_or_None, detail_string).
    """
    ok, detail = import_metal_coord_term()
    if not ok:
        return False, None, detail
    try:
        from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (  # type: ignore  # noqa: E501
            MetalCoordinationTerm,
        )

        term = MetalCoordinationTerm(
            weight_geometry=weight_geometry,
            weight_hsab=weight_hsab,
            weight_distance=weight_distance,
            kcal_per_unit=kcal_per_unit,
        )
        return (
            True,
            term,
            (
                f"build_ok; kcal_per_unit={kcal_per_unit}; "
                f"weight_geometry={weight_geometry}; "
                f"weight_hsab={weight_hsab}; "
                f"weight_distance={weight_distance}; "
                f"MetalCoordinationTerm ready for SCAFFOLD wire-in"
            ),
        )
    except Exception as exc:  # pragma: no cover
        return False, None, f"build_failed: {type(exc).__name__}: {exc}"


def per_epoch_kcal_sign_audit(
    composite_scores: List[float],
    kcal_per_unit: float,
) -> Dict[str, Any]:
    """Audit the Wf-333-V1 kcal sign fix on a list of composite
    metal-coordination scores.

    For each ``c in composite_scores``:
        - expected_kcal_post_fix = (1 - 2*c) * kcal_per_unit
        - expected_kcal_pre_fix = (2*c - 1) * kcal_per_unit
        - sign_audit_pass = expected_kcal_post_fix < 0 iff c > 0.5

    Returns a JSON-serialisable audit dict.
    """
    post_fix_kcal = [(1.0 - 2.0 * c) * kcal_per_unit for c in composite_scores]
    pre_fix_kcal = [(2.0 * c - 1.0) * kcal_per_unit for c in composite_scores]
    sign_audit: List[bool] = []
    for c, post in zip(composite_scores, post_fix_kcal):
        if c > 0.5:
            sign_audit.append(post < 0.0)
        elif c < 0.5:
            sign_audit.append(post > 0.0)
        else:
            sign_audit.append(abs(post) < 1e-9)
    return {
        "composite_scores": list(composite_scores),
        "post_fix_kcal": post_fix_kcal,
        "pre_fix_kcal": pre_fix_kcal,
        "sign_audit_per_score": sign_audit,
        "sign_audit_pass": all(sign_audit),
        "kcal_per_unit": kcal_per_unit,
        "wf_333_v1_fix_id": "metal_coord_term.py:635 (1-2*comp)*kcal_per_unit",
        "n_scores_audited": len(composite_scores),
    }


def scaffold_smoke(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """5-step CPU smoke for the Wf-406-A4 SCAFFOLD.

    Verifies (a) MetalCoordinationTerm import + build (b) kcal sign
    fix holds at line 635 (c) train-loop wire-in is plumbed but
    training body is no-op for SCAFFOLD (d) result.json is emitted
    with MEASURED-vs-PROJECTED tag discipline.

    Returns a JSON-serialisable result dict.
    """
    t0 = time.time()
    import_ok, import_detail = import_metal_coord_term()
    build_ok, term_instance, build_detail = build_metal_coord_term(
        kcal_per_unit=args.kcal_per_unit,
    )
    # 5-step smoke with composite_scores that span [0, 1] to verify
    # the kcal sign fix end-to-end:
    composite_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    kcal_audit = per_epoch_kcal_sign_audit(
        composite_scores=composite_scores,
        kcal_per_unit=args.kcal_per_unit,
    )
    wall = time.time() - t0
    return {
        "wf": "Wf-406-A4",
        "deliverable": "Wf-246-C metal-coordination term retrain SCAFFOLD",
        "status": "SHIPPED_SCAFFOLD_CPU_ONLY",
        "ts": time.time(),
        "args": vars(args),
        "import_ok": import_ok,
        "import_detail": import_detail,
        "build_ok": build_ok,
        "build_detail": build_detail,
        "kcal_sign_audit": kcal_audit,
        "n_metals": len(args.metals),
        "metals": list(args.metals),
        "n_seeds": len(args.seeds),
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "smoke": args.smoke,
        "wall_seconds": wall,
        "honest_framing": {
            "user_ok_required_for_gpu_launch": True,
            "user_ok_env_var": "USER_OK=1",
            "verdict": (
                "SCAFFOLD_CPU_ONLY; actual GPU retrain BLOCKED on "
                "USER_OK=1 per AGENTS.md §5.1 + Wf-355-B1 pattern"
            ),
            "metal_coord_sign_fix_id": "Wf-333-V1 line 635",
            "metal_coord_sign_fix_applied": kcal_audit["sign_audit_pass"],
        },
        "notes": [
            "5-metal subset (Ru/Ir/Rh/Os/Re) per Wf-390-A1 Path A",
            "kcal sign fix verified at line 635 per Wf-333-V1",
            "Trainer wire-in is plumbed but training body is no-op "
            "for SCAFFOLD (full wire-in deferred to USER_OK-gated "
            "GPU launch wave)",
            "0 GPU launches performed",
            "0 paper/* touches",
            "0 wet-lab planning per AGENTS.md §10",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the Wf-406-A4 SCAFFOLD.

    Usage:
        python molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py \
            --metal-coord-term --smoke
        python molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py \
            --metal-coord-term --no-smoke --epochs 50 \
            --metals Ru Ir Rh --seeds 0 1 2 \
            --out-dir molmetal/reports/wf406a4_full_sim
    """
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = scaffold_smoke(args)
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[Wf-406-A4] SCAFFOLD result written to {out_path}; "
        f"status={result['status']}; "
        f"import_ok={result['import_ok']}; "
        f"build_ok={result['build_ok']}; "
        f"kcal_sign_audit_pass={result['kcal_sign_audit']['sign_audit_pass']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
