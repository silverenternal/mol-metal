#!/usr/bin/env python3
"""Wf-29 Phase 2 — CPU-only smoke verification of ``wf29_phase2_summarize.py``.

Exercises the Wf-29 Phase 2 summarizer on a synthetic 1-recipe smoke
fixture (1 seed × 1 epoch) and asserts:

1.  The summarizer module imports cleanly.
2.  :func:`aggregate_smoke` returns a well-formed bench_summary dict.
3.  :func:`render_comparison_md` emits a markdown string that
    references every recorded recipe and contains both the headline
    table and the per-metal table.
4.  Missing recipes are recorded under ``recipes_skipped_missing``.
5.  The end-to-end CLI runs to completion and writes the documented
    outputs.

CPU-only, no GPU.  No mutation of Wf-29 trainer scripts.

Usage::

    uv run --no-sync python molmetal/scripts/verify_wf29_phase2_summarize.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The summarizer module (the Wf-29 Phase 2 deliverable).
import wf29_phase2_summarize as sm  # type: ignore[import-not-found]


def _check_imports() -> None:
    assert hasattr(sm, "aggregate_smoke")
    assert hasattr(sm, "render_comparison_md")
    assert hasattr(sm, "_extract_recipe_summary")
    assert hasattr(sm, "_per_metal_auc_from_per_seed")
    print("[1/5] module imports OK")


def _check_aggregate_smoke_on_stub(tmpdir: Path) -> dict:
    """Build a 1-recipe stub and verify aggregate_smoke parses it."""
    smoke_dir = tmpdir / "wf29_smoke_stub"
    recipe_dir = smoke_dir / "se3sb"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "recipe": "r18_w2.1_se3sb_5metal_multitask",
        "n_seeds": 1,
        "epochs": 1,
        "init": "se3sb",
        "ru_test_auc_per_seed": [0.55],
        "ru_mean_test_auc": 0.55,
        "wall_seconds": 30.0,
        "per_seed": [
            {
                "seed": 0,
                "best_val_mean_auc": 0.6,
                "test_auc_per_metal": {
                    "Ru": 0.55,
                    "Ir": 0.6,
                    "Rh": 0.65,
                    "Re": 0.7,
                    "Os": 0.75,
                },
            }
        ],
        "_provenance": "MEASURED",
    }
    (recipe_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Stub a missing recipe (no metrics.json) — should land in skipped.
    (smoke_dir / "transport").mkdir(exist_ok=True)

    bench, summaries = sm.aggregate_smoke(smoke_dir, ["se3sb", "transport", "multiph"])
    assert bench["recipe"] == "wf29_phase2_smoke_bench_summary"
    assert "se3sb" in bench["recipes_recorded"]
    assert "transport" in bench["recipes_skipped_missing"]
    assert "multiph" in bench["recipes_skipped_missing"]
    assert len(summaries) == 1
    s = summaries[0]
    assert s["ru_mean_test_auc"] == 0.55
    assert s["per_metal_mean_test_auc"]["Ru"] == 0.55
    assert s["per_metal_mean_test_auc"]["Os"] == 0.75
    assert s["provenance"] == "MEASURED"
    print("[2/5] aggregate_smoke parses 1-recipe stub (transport/multiph → skipped)")
    return bench


def _check_render_comparison_md(bench: dict) -> None:
    md = sm.render_comparison_md(bench)
    assert "Wf-29 Phase 2 — 4-Recipe Smoke Comparison" in md
    assert "| Recipe | Init | n_seeds | Ru AUC mean |" in md
    assert "| Recipe | Ru | Ir | Rh | Re | Os |" in md
    assert "Honest-negative provenance" in md
    assert "se3sb" in md
    assert "transport" in md  # skipped but still listed
    assert "multiph" in md
    print("[3/5] render_comparison_md emits headline + per-metal tables")


def _check_cli_end_to_end(tmpdir: Path) -> None:
    """Drive the CLI main() on a stub smoke dir."""
    smoke_dir = tmpdir / "wf29_smoke_cli"
    recipe_dir = smoke_dir / "se3sb"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "metrics.json").write_text(
        json.dumps(
            {
                "recipe": "r18_w2.1_se3sb_5metal_multitask",
                "n_seeds": 1,
                "epochs": 1,
                "init": "se3sb",
                "ru_test_auc_per_seed": [0.5],
                "ru_mean_test_auc": 0.5,
                "wall_seconds": 10.0,
                "per_seed": [],
                "_provenance": "MEASURED",
            },
            indent=2,
        )
    )
    summary_path = tmpdir / "wf29_summary.json"
    md_path = tmpdir / "comparison.md"
    rc = sm.main(
        [
            "--smoke-dir", str(smoke_dir),
            "--recipes", "se3sb",
            "--summary-path", str(summary_path),
            "--comparison-md", str(md_path),
        ]
    )
    assert rc == 0, f"main() returned {rc}"
    assert summary_path.exists(), "summary_path missing"
    assert md_path.exists(), "md_path missing"
    payload = json.loads(summary_path.read_text())
    assert payload["recipe"] == "wf29_phase2_smoke_bench_summary"
    assert payload["best_recipe"] == "se3sb"
    print(f"[4/5] main() end-to-end OK (rc=0, summary={payload['best_recipe']})")


def _check_honest_negative_in_md(tmpdir: Path) -> None:
    """Verify the honest-negative provenance section is present."""
    smoke_dir = tmpdir / "wf29_smoke_hn"
    summary_path = tmpdir / "hn_summary.json"
    md_path = tmpdir / "hn_comparison.md"
    bench, _ = sm.aggregate_smoke(smoke_dir, ["se3sb"])
    md = sm.render_comparison_md(bench)
    assert "no recipes recorded" in md
    assert "Honest-negative provenance" in md
    print("[5/5] empty smoke dir renders honest-negative row")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-tmp", action="store_true", help="Keep the temp dir for inspection.")
    return p.parse_args(argv)


def main() -> int:
    print("=== wf29_phase2_summarize smoke verify ===")
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _check_imports()
        bench = _check_aggregate_smoke_on_stub(tmpdir)
        _check_render_comparison_md(bench)
        _check_cli_end_to_end(tmpdir)
        _check_honest_negative_in_md(tmpdir)
        if args.keep_tmp:
            print(f"[verify_wf29_phase2_summarize] tmp kept at {tmpdir}")
    print("=== all checks done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
