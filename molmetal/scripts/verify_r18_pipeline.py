"""Smoke verification for the R18 end-to-end pipeline orchestrator.

================================================================
Wf-E2E smoke (TODO-34 §L-R18-WF-e2e, 2026-09-26).
================================================================
Runs a 5-sample smoke harness of the R18 orchestrator on PlatinAI 5K
and prints the per-step wall-time + memory + total cost.  Returns
exit 0 when every step is at least ``"ok"`` or ``"skipped"``, exit 1
otherwise.

Run via::

    uv run --no-sync python molmetal/scripts/verify_r18_pipeline.py --smoke

Output is rendered as a small text table; a JSON report is also
written to ``molmetal/reports/wf_e2e_pipeline/verify_smoke.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.r18_pipeline_orchestrator import (  # noqa: E402
    REPORT_DIR,
    run_platinai_e2e,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_ms(seconds: float) -> str:
    """Format seconds as ``"XX.Xms"`` for human display."""
    return f"{seconds * 1000.0:7.1f}ms"


def _render_text_table(report: Dict[str, Any]) -> str:
    """Render a small text table of the per-step wall-time + memory."""
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("R18 pipeline e2e smoke verification")
    lines.append("=" * 64)
    lines.append(f"  dataset         : {report.get('dataset', 'n/a')}")
    lines.append(f"  n_samples       : {report.get('n_samples', 0)}")
    lines.append(f"  success_rate    : {report.get('success_rate', 0.0):.2%}")
    lines.append(
        f"  total_wall_time : "
        f"{report.get('total_wall_time_s', 0.0):7.2f}s"
    )
    lines.append("-" * 64)
    lines.append(f"  {'step':<22} {'wall':>10} {'mem_mb':>10} {'status':>10}")
    lines.append("-" * 64)
    per_step = report.get("per_step_wall_time_s", {}) or {}
    per_mem = report.get("per_step_memory_mb", {}) or {}
    per_status = report.get("per_step_status_counts", {}) or {}
    for step_name in sorted(per_step.keys()):
        wall = float(per_step.get(step_name, 0.0))
        mem = float(per_mem.get(step_name, 0.0))
        counts = per_status.get(step_name, {}) or {}
        # Show the dominant status (highest count).
        if counts:
            dominant = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            dominant = "n/a"
        lines.append(
            f"  {step_name:<22} {_fmt_ms(wall):>10} {mem:>10.2f} {dominant:>10}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def _write_json(report: Dict[str, Any], dest: Path) -> None:
    """Write the report as JSON; coerce numpy scalars defensively."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(o: Any) -> Any:
        try:
            import numpy as _np
            if isinstance(o, (_np.floating, _np.integer)):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)

    with dest.open("w") as fh:
        json.dump(report, fh, indent=2, default=_coerce)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R18 pipeline e2e smoke verification.",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run the 5-sample PlatinAI smoke (default).")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Sample count for the smoke (default 5).")
    parser.add_argument("--no-write", action="store_true",
                        help="Skip the JSON report dump.")
    return parser


def main(argv: Any = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    n_samples = int(args.n_samples) if not args.smoke else 5
    print(f"[verify_r18_pipeline] running PlatinAI smoke (n={n_samples}) ...")
    t0 = time.perf_counter()
    report = run_platinai_e2e(n_samples=n_samples)
    elapsed = time.perf_counter() - t0
    print(_render_text_table(report))
    print(f"[verify_r18_pipeline] wall_time_in_main={elapsed:.2f}s")

    if not args.no_write:
        out_path = REPORT_DIR / "verify_smoke.json"
        _write_json(report, out_path)
        print(f"[verify_r18_pipeline] json_report={out_path}")

    # PASS criterion: every step is ok / skipped.
    statuses = report.get("per_step_status_counts", {}) or {}
    bad_steps: List[str] = []
    for step_name, counts in statuses.items():
        n_errors = int(counts.get("error", 0))
        if n_errors > 0:
            bad_steps.append(step_name)
    if bad_steps:
        print(
            f"[verify_r18_pipeline] FAIL: error in steps {bad_steps}",
            file=sys.stderr,
        )
        return 1
    print("[verify_r18_pipeline] PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
