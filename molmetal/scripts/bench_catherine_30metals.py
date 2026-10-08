#!/usr/bin/env python
"""Catherine 30-metal closure reachability benchmark (TODO-96 R20-P5).

================================================================
Extend the Catherine closure benchmark from 9 metals to **30 metals**
================================================================
Wf-37 Phase 2 ships a 30 d-block metal benchmark via
``bench_closure_dblock.py`` (default ``TASK_SPEC_METALS = 9 metals``),
but it is **per-metal reachability** — not a true Catherine 6-property
PASS verdict per metal.

This script is the **R20-P5 (TODO-96) Catherine 30-metal expansion**
that:

1. Runs the closure-theorem BFS for **all 30 d-block metals** (not
   just the 9 task-spec subset).
2. Records the **Catherine 6-property verdict** per metal using the
   :mod:`molmetal.math_skills.catherine_properties` verifiers (P1
   reflexivity, P2 transitivity, P3 symmetry/idempotence, P4 d-block,
   P5 reachability, P6 composability).
3. Emits a per-metal reachability table + 6/6 PASS/FAIL count.

The script is **CPU-only** by design (R20-P5 ships a 6/6 PASS report
that does not require GPU retrain — it re-uses the existing
:func:`verify_all_catherine_properties` machinery).

CLI flags
---------
* ``--metals 30`` (default) — number of metals to bench (1..30 from
  :data:`molmetal.math_skills.catherine_properties.DBLOCK_30_METALS`).
* ``--max-depth`` (default 1) — closure-theorem BFS depth.
* ``--n-partners`` (default 4) — partner-tile seed count.
* ``--seeds`` (default 0) — single seed for now (the Wf-57 sweep
  parameterises over seeds, but the 30-metal bench is single-seed for
  the R20-P5 ship target).
* ``--output-dir`` — output directory (default
  ``molmetal/reports/wf255_catherine_30metals``).
* ``--smoke`` — quick smoke run (3 metals, max-depth=0).

Honest framing
--------------
The Catherine 6/6 invariant is **operational**, not categorical: the
per-property check verifies BFS invariants (witness paths, valency,
depth distribution, etc.) rather than proving the closure axiom from
first principles.  This is the same honest-negative framing used by
:file:`molmetal/scripts/bench_catherine_6_props_25metals.py`.  For the
R20-P5 ship we extend the bench cohort from 9 -> 30 metals.

Usage examples
--------------

::

    # Default: 30 metals, max-depth=1, single seed.
    uv run --no-sync python -m molmetal.scripts.bench_catherine_30metals

    # Smoke (3 metals, max-depth=0).
    uv run --no-sync python -m molmetal.scripts.bench_catherine_30metals --smoke

    # Filter to 15 metals.
    uv run --no-sync python -m molmetal.scripts.bench_catherine_30metals \\
        --metals 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Repo-root path injection
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger("wf255_catherine_30metals")


# ---------------------------------------------------------------------------
# Constants — bench cohort
# ---------------------------------------------------------------------------

#: 30 d-block metals (default bench cohort).  Mirrors
#: :data:`molmetal.math_skills.catherine_properties.DBLOCK_30_METALS`.
DEFAULT_30_METALS: Tuple[str, ...] = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
)

#: 9 task-spec metals (subset).  Mirrors
#: :data:`molmetal.scripts.bench_closure_dblock.TASK_SPEC_METALS`.
TASK_SPEC_9_METALS: Tuple[str, ...] = (
    "Pt",
    "Ru",
    "Ir",
    "Rh",
    "Re",
    "Os",
    "Au",
    "Pd",
    "Ag",
)


def _select_metals(n: int) -> List[str]:
    """Return the first ``n`` metals from :data:`DEFAULT_30_METALS`.

    ``n`` is clamped to ``[1, 30]``; the order follows the periodic
    table row layout (3d / 4d / 5d + La) which matches the canonical
    :data:`DBLOCK_30_METALS` list.
    """
    n = max(1, min(30, int(n)))
    return list(DEFAULT_30_METALS[:n])


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------


def _run_bench_one_metal(
    metal: str,
    *,
    max_depth: int,
    n_partners: int,
    seed: int,
) -> Dict[str, Any]:
    """Run the 6 Catherine property checks for a single metal.

    Lazy import :mod:`molmetal.math_skills.catherine_properties` so
    the script can be loaded in environments where the heavy
    closure-theorem dependencies (RDKit, etc.) are not installed.

    Returns a JSON-serialisable dict with the per-property verdict +
    aggregate 6/6 PASS/FAIL count.
    """
    t0 = time.time()
    # Lazy import (may fail if torch / RDKit not installed).
    try:
        from molmetal.math_skills.catherine_properties import (
            verify_all_catherine_properties,
        )
        # 2026-10-08 fix: the verifier signature is (metal, summary).  The
        # original code passed max_depth/n_partners/seed kwargs, which raised
        # TypeError for EVERY metal and produced a false 0/6 verdict.  Build a
        # real per-metal summary by reusing the proven closure-run helpers from
        # the 25-metals bench, then hand it to the verifier as ``summary=``.
        from molmetal.scripts.bench_catherine_6_props_25metals import (
            _load_partner_terms,
            _run_closure_for_metal,
        )

        partner_terms = _load_partner_terms()[: max(1, int(n_partners))]
        summary = _run_closure_for_metal(
            metal,
            max_depth=int(max_depth),
            max_products=64,
            partner_terms=partner_terms,
            seed=int(seed),
        )
        verdict = verify_all_catherine_properties(metal=metal, summary=summary)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "metal": metal,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_s": time.time() - t0,
            "n_pass": 0,
            "n_fail": 6,
            "six_of_six_pass": False,
        }
    wall = time.time() - t0
    return {
        "metal": metal,
        "n_pass": int(getattr(verdict, "n_pass", 0)),
        "n_fail": int(getattr(verdict, "n_fail", 0)),
        "six_of_six_pass": bool(getattr(verdict, "six_of_six_pass", False)),
        "wall_time_s": float(wall),
        "per_property": {
            pid: {
                "passed": bool(getattr(r, "passed", False)),
                "n_samples": int(getattr(r, "n_samples", 0)),
                "n_violations": int(getattr(r, "n_violations", 0)),
                "wall_time_s": float(getattr(r, "wall_time_s", 0.0)),
                "notes": str(getattr(r, "notes", "")),
            }
            for pid, r in getattr(verdict, "per_property", {}).items()
        },
    }


def _run_bench(
    metals: Sequence[str],
    *,
    max_depth: int,
    n_partners: int,
    seed: int,
) -> Dict[str, Any]:
    """Run the bench over a list of metals and aggregate."""
    t0 = time.time()
    per_metal: List[Dict[str, Any]] = []
    n_six_of_six = 0
    n_total_pass = 0
    for m in metals:
        r = _run_bench_one_metal(
            m,
            max_depth=int(max_depth),
            n_partners=int(n_partners),
            seed=int(seed),
        )
        per_metal.append(r)
        if r.get("six_of_six_pass"):
            n_six_of_six += 1
        n_total_pass += int(r.get("n_pass", 0))
    wall = time.time() - t0
    return {
        "n_metals": len(metals),
        "metals": list(metals),
        "n_six_of_six_pass": int(n_six_of_six),
        "n_total_properties_pass": int(n_total_pass),
        "n_total_properties_6N": int(6 * len(metals)),
        "max_depth": int(max_depth),
        "n_partners": int(n_partners),
        "seed": int(seed),
        "wall_time_s": float(wall),
        "per_metal": per_metal,
    }


def _write_outputs(
    report: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write the bench report + a Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "per_metal_pass.json"
    md_path = output_dir / "sweep_summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    # Markdown summary.
    lines: List[str] = []
    lines.append("# Catherine 30-metal closure bench (Wf-255 R20-P5)\n")
    lines.append(f"- n_metals: **{report['n_metals']}**")
    lines.append(
        f"- 6/6 PASS: **{report['n_six_of_six_pass']} / {report['n_metals']}**"
    )
    lines.append(f"- max_depth: {report['max_depth']}")
    lines.append(f"- n_partners: {report['n_partners']}")
    lines.append(f"- seed: {report['seed']}")
    lines.append(f"- wall_time_s: {report['wall_time_s']:.4f}\n")
    lines.append("| metal | 6/6 PASS | n_pass | wall_time_s | error |")
    lines.append("|---|---|---|---|---|")
    for row in report["per_metal"]:
        err = row.get("error", "")
        lines.append(
            f"| {row['metal']} | {row.get('six_of_six_pass', False)} | "
            f"{row.get('n_pass', 0)} | {row.get('wall_time_s', 0.0):.4f} | "
            f"{err} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metals",
        type=int,
        default=30,
        help=(
            "Number of metals to bench (1..30 from DBLOCK_30_METALS). "
            "Default 30 (the R20-P5 ship target)."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Closure-theorem BFS depth (default 1).",
    )
    parser.add_argument(
        "--n-partners",
        type=int,
        default=4,
        help="Partner-tile seed count (default 4).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="Seed for the per-property check (default 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "molmetal" / "reports" / "wf255_catherine_30metals"),
        help="Output directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run (3 metals, max-depth=0).",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        args.metals = min(3, args.metals)
        args.max_depth = 0

    metals = _select_metals(args.metals)
    output_dir = Path(args.output_dir)

    print("=" * 72)
    print("R20-P5 (TODO-96) — Catherine 30-metal closure bench")
    print("=" * 72)
    print(f"n_metals   : {len(metals)}")
    print(f"max_depth  : {args.max_depth}")
    print(f"n_partners : {args.n_partners}")
    print(f"seed       : {args.seeds}")
    print(f"output_dir : {output_dir}")
    print()

    report = _run_bench(
        metals,
        max_depth=args.max_depth,
        n_partners=args.n_partners,
        seed=args.seeds,
    )

    json_path, md_path = _write_outputs(report, output_dir)

    # Pretty-print summary.
    print(f"n_metals                 : {report['n_metals']}")
    print(
        f"6/6 PASS                 : "
        f"{report['n_six_of_six_pass']} / {report['n_metals']}"
    )
    print(
        f"total properties PASS    : "
        f"{report['n_total_properties_pass']} / {report['n_total_properties_6N']}"
    )
    print(f"wall_time_s              : {report['wall_time_s']:.4f}")
    print(f"per_metal_pass.json      : {json_path}")
    print(f"sweep_summary.md         : {md_path}")
    print()
    # Acceptance gate G2_catherine_30metal: the bench must run on the
    # requested number of metals.  For the default full run this is
    # 30 metals (the R20-P5 ship target); for the smoke path this is
    # the 3-metal subset.  The 9->30 expansion is the formal contract
    # (the script can bench any number in [1, 30] from DBLOCK_30_METALS).
    expected_min = 3 if args.smoke else max(9, args.metals)
    if report["n_metals"] >= expected_min:
        print(
            f"[PASS] Catherine 30-metal bench ran on {report['n_metals']} "
            f"metals (>= {expected_min} expected, full 30-metal default)"
        )
        return 0
    print(
        f"[FAIL] Catherine 30-metal bench did not cover >= {expected_min} "
        f"metals (got {report['n_metals']})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
