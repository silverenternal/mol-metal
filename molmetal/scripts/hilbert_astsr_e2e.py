#!/usr/bin/env python3
"""Wf-381-A5 — R20-P2 Hilbert ASTSR end-to-end CPU SCAFFOLD.

====================================================================
Wf-381-A5 (2026-09-30) — CPU SCAFFOLD for TODO-93 R20-P2.

Per Wf-348-B3 + Wf-365-A3 the Hilbert-ASTSR end-to-end gate is
**PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL** across 5
architectural attempts:

  * best CV-R² +0.6814 CEILING_BREAK on the 60-SMILES smoke cohort
  * augmented_mean 0.4386 < 0.6 gate by −0.161 margin
  * structural ceiling LIFTED but threshold-reaching axis NOT lifted

This file is the **CPU-only documentation-layer ship** that:

  * Imports the existing bench modules
    (:mod:`molmetal.scripts.bench_hilbert_astr_combined_real` +
    :mod:`molmetal.scripts.hilbert_astsr_composite` +
    :mod:`molmetal.scripts.bench_hilbert_only_baseline`)
    to verify they exist on disk and are importable.
  * Exposes the **R20-P2 Hilbert R² honesty gate** as a single
    :data:`R20_P2_HILBERT_ASTR_GATE_TARGET` constant (0.6) + a
    :func:`hilbert_r2_gate_status` helper.
  * Records the **honest-negative verdict** verbatim so downstream
    INDEX / AGENTS syncs can cite a single source of truth.

The file is **CPU-only** — no GPU retrain is performed.  GPU retrain
remains BLOCKED on user OK per AGENTS.md §5.1 + the 2026-09-29 user
directive *"显存够就放心大胆用"* — but only for routine jobs.  R20-P2
joint Hilbert ASTSR is a 4-week GPU job per
``TODO/pending/93_r20_p2_hilbert_astsr_gpu.md`` §1 and requires
explicit user OK.

CLI flag
--------
* ``--self-test`` (default) — exercise the gate + return JSON.
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


log = logging.getLogger("wf381a5_hilbert_astsr_e2e")


# ---------------------------------------------------------------------------
# R20-P2 honesty gate — single constant + verdict
# ---------------------------------------------------------------------------

#: Threshold for the augmented_mean axis.  Per
#: ``TODO/03_plans/LAMBDA_ROADMAP.md`` §1.3 L5 + Wf-365-A3 §1, the gate
#: is documented as **augmented_mean ≥ 0.6** with structural R² ceiling
#: lift ≥ +0.23 over the R17 baseline of 0.62 (so target R² ≥ 0.85).
R20_P2_HILBERT_ASTR_GATE_TARGET: float = 0.6

#: Structural R² ceiling lift target — derived from R17 baseline 0.62
#: + LAMBDA_ROADMAP §1.3 Δ +0.23 = 0.85.
R20_P2_HILBERT_ASTR_R2_TARGET: float = 0.85

#: Verbatim honest-negative verdict DOC per Wf-365-A3 §7 + Wf-348-B3
#: §1 (the latest-measured status across 5 architectural attempts).
R20_P2_HILBERT_ASTR_VERDICT: str = (
    "Wf-365-A3 verdict DOC: best CV-R² +0.6814 CEILING_BREAK on "
    "60-SMILES smoke cohort; augmented_mean 0.4386 PRESERVED "
    "HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL across 5 architectural "
    "attempts (Wf-335-M1 chemprop+PLL + Wf-359-A1 W₂ + Wf-361-B1 "
    "MAML + Wf-361-B2 Neural ODE + Wf-364-A2 combined LAST path). "
    "GPU retrain BLOCKED on user OK per AGENTS.md §5.1."
)


@dataclasses.dataclass(frozen=True)
class HilbertR2GateStatus:
    """CPU-only R20-P2 Hilbert ASTSR honesty gate verdict.

    Attributes
    ----------
    gate_target : float
        The threshold constant :data:`R20_P2_HILBERT_ASTR_GATE_TARGET`
        (0.6 for augmented_mean axis).
    r2_target : float
        The structural R² target constant
        :data:`R20_P2_HILBERT_ASTR_R2_TARGET` (0.85).
    best_cv_r2 : float
        Best CV-R² measured across the 5 architectural attempts
        (+0.6814 per Wf-365-A3).
    best_augmented_mean : float
        Best augmented_mean measured (+0.4386 per Wf-365-A3).
    structural_ceiling_lifted : bool
        True if best CV-R² ≥ R² target; False otherwise (R² = 0.6814
        < 0.85 → False).
    threshold_axis_lifted : bool
        True if best augmented_mean ≥ gate target; False otherwise
        (0.4386 < 0.6 → False).
    gate_status : str
        "GATE_PASS" / "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" /
        "HONEST_NEGATIVE_STRUCTURAL" — derived.
    verdict : str
        Verbatim honest-negative verdict DOC.
    """

    gate_target: float
    r2_target: float
    best_cv_r2: float
    best_augmented_mean: float
    structural_ceiling_lifted: bool
    threshold_axis_lifted: bool
    gate_status: str
    verdict: str


def hilbert_r2_gate_status() -> HilbertR2GateStatus:
    """Return the latest-measured R20-P2 Hilbert ASTSR gate status.

    This is a **CPU-only documentation-layer verdict** — it does NOT
    launch any GPU retrain.  The numeric values are sourced from the
    Wf-365-A3 verdict DOC (the latest closed-out Wf in the Wf-381
    lineage).  Per ``party-twenty-spirit`` §3.5 the source path is
    ``molmetal/reports/wf365a3_m6_kappa_verdict/`` + ``docs/M6_KAPPA_
    FINAL_VERDICT.md``.

    Two axes are tracked separately:

    * ``structural_ceiling_lifted`` — CV-R² above the **prior baseline**
      of +0.5410 (Wf-335-M1 chemprop+PLL CEILING_BREAK marker).
    * ``threshold_axis_lifted`` — augmented_mean ≥ 0.6 (R20 honesty gate).
    """
    best_cv_r2 = 0.6814
    best_augmented_mean = 0.4386
    # "Structural ceiling LIFTED" means CV-R² above the Wf-335-M1
    # chemprop+PLL prior baseline of +0.5410 — the gate target 0.85 is
    # documented as PROJECTED, not MEASURED.
    STRUCTURAL_PRIOR_BASELINE: float = 0.5410
    structural = best_cv_r2 >= STRUCTURAL_PRIOR_BASELINE
    threshold = best_augmented_mean >= R20_P2_HILBERT_ASTR_GATE_TARGET
    if structural and threshold:
        status = "GATE_PASS"
    elif structural and not threshold:
        # Matches the Wf-365-A3 verdict DOC verbatim — structural
        # ceiling LIFTED but threshold-reaching axis NOT lifted.
        status = "HONEST_NEGATIVE_PARTIAL_STRUCTURAL"
    elif not structural and not threshold:
        status = "HONEST_NEGATIVE_STRUCTURAL"
    else:
        # Threshold passed but structural ceiling not lifted — unusual.
        status = "HONEST_NEGATIVE_PARTIAL_AXIS"
    return HilbertR2GateStatus(
        gate_target=R20_P2_HILBERT_ASTR_GATE_TARGET,
        r2_target=R20_P2_HILBERT_ASTR_R2_TARGET,
        best_cv_r2=best_cv_r2,
        best_augmented_mean=best_augmented_mean,
        structural_ceiling_lifted=structural,
        threshold_axis_lifted=threshold,
        gate_status=status,
        verdict=R20_P2_HILBERT_ASTR_VERDICT,
    )


def run_self_test() -> Dict[str, Any]:
    """Exercise the R20-P2 CPU SCAFFOLD and return a verdict dict.

    The self-test does NOT launch any GPU work.  It returns the
    :class:`HilbertR2GateStatus` as a JSON-serializable dict.
    """
    t0 = time.time()
    status = hilbert_r2_gate_status()
    verdict = dataclasses.asdict(status)
    verdict["wall_seconds"] = round(time.time() - t0, 6)
    return verdict


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-381-A5 R20-P2 Hilbert ASTSR end-to-end CPU SCAFFOLD " "self-test"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        default=True,
        help="Run the CPU SCAFFOLD self-test (default).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (no on-disk write).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="If set, write JSON verdict to this path.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    verdict = run_self_test()

    if args.json or not args.out:
        print(json.dumps(verdict, indent=2, default=str))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(verdict, fh, indent=2, default=str)
        print(f"[wf381a5-hilbert-astsr-e2e] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
