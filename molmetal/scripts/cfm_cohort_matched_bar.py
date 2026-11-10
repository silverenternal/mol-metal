# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``metrics/by_metric/cfm_cohort_matched_bar.json``.

WHY THIS SCRIPT EXISTS
-----------------------
The bar artifact was, until Wf-A114, a JSON file with NO generator: nothing
in the repo could recompute it, so nothing could check it.  The numbers in
it were correct, but "correct" was a claim about a file, not a property of
the code.  This script closes that gap.

It does NOT re-implement the comparison.  It calls
``molmetal/scripts/cfm_cohort_matched_compare.py`` -- the ONE owner of the
end-to-end formula and of the six axes -- and then writes down what that
harness returned.  A number can only reach the artifact by passing the
harness's axes; the artifact cannot carry a number the harness refuses.

The three bars are NOT cohort medians.  Each one is a PER-RUN number,
against a NAMED run, under one reading, at one budget:

  primary   cfm_r9_fix_v2        final-epoch reading, cohort 26715/6714
  secondary cfm_fullcorpus_seed1 best-epoch reading,  cohort 26715/192
  stretch   cfm_fm_ab_2026_10_09 epoch 12,               cohort 256/192
            (+ cfm_pcgrad_off_2026_10_15) -- REPORT-ONLY: a different
            cohort AND a different budget, so it is never a pass/fail.

Each bar's value is traced back to disk through
``trace_bar_to_disk``, which is INTEGRITY-GATED (Wf-A114 verify:D D1): a
value that lives on a run whose recorded history is broken is reported as
REJECTED, never as MEASURED.  That gate is what stops the WITHDRAWN
``cfm_r11_eval256`` number (0.572265625) from being re-certified here.

CPU-ONLY.  Reads the per-epoch JSONs already on disk; trains nothing,
samples nothing, touches no CUDA.

Usage:
    .venv/bin/python molmetal/scripts/cfm_cohort_matched_bar.py
    .venv/bin/python molmetal/scripts/cfm_cohort_matched_bar.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_cohort_matched_compare import (  # noqa: E402
    DEFAULT_REPORTS_ROOT,
    compare_runs,
    e2e_generation,
    load_run,
    trace_bar_to_disk,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: The published artifact this script regenerates.
ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "cfm_cohort_matched_bar.json"

#: The census artifact, used ONLY as a cross-check (never a gate).
CENSUS = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"

#: The three bars.  ``epoch`` is the compared epoch INDEX; ``reading`` is the
#: reading the number is taken under.  ``pass_fail`` says whether the bar may
#: be used as a pass/fail threshold -- the stretch bar may not (different
#: cohort AND different budget).
BARS = (
    {
        "name": "primary",
        "run": "cfm_r9_fix_v2",
        "reading": "epoch",
        "epoch": 10,
        "pass_fail": True,
        "rule": (
            "a new run in this cohort must EXCEED this value under the SAME "
            "reading and a budget-matched epoch index, with a "
            "contiguous-from-zero history"
        ),
    },
    {
        "name": "secondary",
        "run": "cfm_fullcorpus_seed1",
        "reading": "best",
        "epoch": 13,
        "pass_fail": True,
        "rule": (
            "a new run in this cohort must EXCEED this value under the SAME "
            "reading, and only when BOTH runs' recorded epoch sets are "
            "identical.  The best reading is a MAX over a noisy series (the "
            "census labels it 'optimistically biased'), so it is a SECONDARY "
            "bar and never interchangeable with the primary one"
        ),
    },
    {
        "name": "stretch",
        "run": "cfm_fm_ab_2026_10_09",
        "also": "cfm_pcgrad_off_2026_10_15",
        "reading": "epoch",
        "epoch": 12,
        "pass_fail": False,
        "rule": (
            "a DIFFERENT cohort AND a DIFFERENT budget: report-only, NEVER a "
            "pass/fail.  It is what the 256-cohort runs reached at epoch 12; "
            "the 26715-cohort runs are out of phase in the same oscillation "
            "and reach it at a different index"
        ),
    },
)

#: The six comparisons the artifact records.  Each one is run through the
#: harness, so a REFUSAL in the artifact is a REFUSAL the harness produced.
VERDICTS = (
    {
        "run_a": "cfm_frag_assemble",
        "run_b": "cfm_frag_control",
        "reading": "epoch",
        "epoch": 29,
        "note": (
            "the assemble/connectivity loss term was already tried and made "
            "fragmentation WORSE on both views"
        ),
    },
    {
        "run_a": "cfm_frag_assemble",
        "run_b": "cfm_frag_control",
        "reading": "best",
        "epoch": None,
        "note": "the same pair under the 'best' reading (identical epoch sets)",
    },
    {
        "run_a": "cfm_r9_fix_v2",
        "run_b": "cfm_r8_cosine_40ep",
        "reading": "epoch",
        "epoch": 8,
        "note": "the PRIMARY bar's own run, against a same-cohort peer",
        "census_crosscheck": True,
    },
    {
        "run_a": "cfm_fullcorpus_seed1",
        "run_b": "cfm_fullcorpus_seed2",
        "reading": "epoch",
        "epoch": 14,
        "note": "the SECONDARY bar's own run, against a same-cohort peer",
    },
    {
        "run_a": "cfm_fm_ab_2026_10_09",
        "run_b": "cfm_pcgrad_off_2026_10_15",
        "reading": "epoch",
        "epoch": 12,
        "note": "the STRETCH pair: two 256-cohort runs at epoch 12",
    },
    {
        "run_a": "cfm_r11_eval256",
        "run_b": "cfm_fm_ab_2026_10_09",
        "reading": "epoch",
        "epoch": 19,
        "note": "A REFUSAL IS A SUCCESS: this pair does not match",
    },
)

#: The numbers this artifact refuses to publish, with the reason.
RETRACTED_NUMBERS = {
    "0.572276953125": (
        "cfm_r11_eval256 -- WITHDRAWN as a bar. MEASURED: its own "
        "per_epoch_metrics carries exactly ONE recorded epoch (index 19) out "
        "of a declared 20, the history is not contiguous from zero, and its "
        "train_loss is nan.  The full-precision JSON value of the same cell "
        "is 0.572265625; the 0.572276953125 form is the 4-decimal CSV "
        "rendering.  This artifact and the harness read the JSON."
    ),
}


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reading_text(reading: str) -> str:
    if reading == "best":
        return (
            "best (the max over the run's own series -- optimistically "
            "biased; never a bar by itself)"
        )
    return (
        "epoch (the run's own FINAL epoch; the run recorded "
        "{epochs} epochs, 0-{last})"
    )


def _bar_entry(
    spec: Dict[str, Any],
    reports_root: Path,
) -> Dict[str, Any]:
    """One bar, computed from the run's own row and traced to disk."""
    run = load_run(reports_root, spec["run"])
    epoch = int(spec["epoch"])
    row = run.row_at(epoch)
    value = e2e_generation(row)
    if value is None:
        raise SystemExit(
            f"{spec['run']} has no measured end-to-end rate at epoch {epoch}"
        )

    traced = trace_bar_to_disk(value, reports_root)
    entry: Dict[str, Any] = {
        "value": value,
        "label": "MEASURED" if (traced and traced["status"] == "TRACED") else "REJECTED",
        "source": (
            f"{run.source} -> per_epoch_metrics[epoch=={epoch}]"
        ),
        "traced_to": {
            "run": traced["run"] if traced else spec["run"],
            "epoch": traced["epoch"] if traced else epoch,
            "value": traced["value"] if traced else value,
            "integrity_ok": bool(traced and traced["status"] == "TRACED"),
        },
        "reading": _reading_text(spec["reading"]).format(
            epochs=len(run.epochs), last=run.last_epoch
        ),
        "cohort": {
            "n_train": run.n_train,
            "n_val": run.n_val,
        },
        "cohort_strict": f"{run.n_train}/{run.n_val}",
        "budget_epochs": epoch + 1,
        "budget_epochs_meaning": (
            "the number of training epochs the run had completed WHEN THE "
            "BAR VALUE WAS OBSERVED (epoch index + 1).  It is NOT the run's "
            "total recorded epoch count: the secondary bar's maximum sat at "
            "epoch 13 of a run that recorded 15 epochs, so its budget is 14, "
            "not 15"
        ),
        "run_epochs_recorded": len(run.epochs),
        "rule": spec["rule"],
        "pass_fail": spec["pass_fail"],
    }
    if spec.get("also"):
        other = load_run(reports_root, spec["also"])
        other_row = other.row_at(epoch)
        other_value = e2e_generation(other_row)
        entry["also_carried_by"] = {
            "run": other.name,
            "epoch": epoch,
            "value": other_value,
            "source": f"{other.source} -> per_epoch_metrics[epoch=={epoch}]",
        }
    if entry["label"] == "REJECTED":
        entry["rejection_reason"] = (
            traced.get("reason") if traced else "the value could not be traced"
        )
    return entry


def _verdict_entry(
    spec: Dict[str, Any],
    reports_root: Path,
    census_path: Optional[Path],
) -> Dict[str, Any]:
    """One comparison, run through the harness itself."""
    a = load_run(reports_root, spec["run_a"])
    b = load_run(reports_root, spec["run_b"])
    reading = spec["reading"]
    epoch = spec["epoch"]
    if reading == "best":
        result = compare_runs(a, b, reading="best")
        index = result.epoch_index
    else:
        result = compare_runs(a, b, reading="epoch", epoch_index=epoch)
        index = result.epoch_index

    out: Dict[str, Any] = {
        "verdict": "VERDICT" if result.ok else "REFUSED",
        "run_a": a.name,
        "run_b": b.name,
        "reading": reading,
        "epoch_index": index,
        "compared_budget_epochs": index + 1,
        "compared_epochs": {
            a.name: result.compared_epochs.get(a.name),
            b.name: result.compared_epochs.get(b.name),
        },
        "unmatched_axes": result.unmatched_axes,
        "compare_json": (
            "molmetal/reports/wfa114_cohort_matched_compare/"
            f"{a.name.replace('/', '__')}__vs__{b.name.replace('/', '__')}"
            + ("__reading_best" if reading == "best" else "")
            + "/compare.json"
        ),
        "note": spec["note"],
    }
    if result.ok:
        va = result.values.get(a.name)
        vb = result.values.get(b.name)
        out["e2e_a"] = va
        out["e2e_b"] = vb
        out["delta_b_minus_a"] = None if (va is None or vb is None) else vb - va
        if reading == "best":
            out["reading_note"] = (
                "under the 'best' reading each arm contributes its OWN maximum "
                "over the searched window; the two maxima need not sit at the "
                "same epoch -- the per-arm epoch is in `compared_epochs`"
            )
    else:
        out["e2e_a"] = None
        out["e2e_b"] = None
        out["delta_b_minus_a"] = None
        out["refusal_detail"] = {
            name: result.axes[name].detail for name in result.unmatched_axes
        }

    if spec.get("census_crosscheck") and census_path and census_path.is_file():
        from molmetal.scripts.cfm_cohort_matched_compare import census_crosscheck

        out["census_crosscheck"] = census_crosscheck(census_path, [a, b])
    return out


def build_artifact(
    reports_root: Path,
    census_path: Optional[Path],
) -> Dict[str, Any]:
    """The whole artifact, computed from the corpus on disk."""
    bars = {spec["name"]: _bar_entry(spec, reports_root) for spec in BARS}
    verdicts = [
        _verdict_entry(spec, reports_root, census_path) for spec in VERDICTS
    ]
    return {
        "metric": "cfm_cohort_matched_bar",
        "status": "MEASURED",
        "measured_at": "2026-10-30",
        "generated_by": (
            "molmetal/scripts/cfm_cohort_matched_bar.py -- this artifact has "
            "a GENERATOR; it is not hand-written"
        ),
        "generated_at": _iso_now(),
        "headline_key": "bars.primary.value",
        "unit": "ratio",
        "scope": (
            "CFM (Conditional Flow Matching) free-generation path -- the share "
            "of ALL generated samples that are one connected molecule"
        ),
        "purpose": (
            "The cohort- and budget-MATCHED bar for the CFM line's generation "
            "quality. A bar must be a PER-RUN, same-reading, same-cohort, "
            "same-budget comparison against a NAMED run -- never a cohort "
            "median."
        ),
        "source": (
            "molmetal/scripts/cfm_cohort_matched_compare.py over each run's "
            "own per_epoch_metrics.json (FULL float precision); nothing "
            "re-derived. The per_epoch_metrics.csv rendering rounds every "
            "rate to 4 decimals, so the harness never reads it."
        ),
        "formula": "(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio",
        "formula_meaning": (
            "gen_validity is an RDKit PARSE rate and gen_single_fragment_ratio "
            "is conditioned on the PARSED rows only; their product is the "
            "share of ALL generated samples that are one connected molecule"
        ),
        "harness": {
            "script": "molmetal/scripts/cfm_cohort_matched_compare.py",
            "test": "molmetal/tests/test_cfm_cohort_matched_compare.py",
            "generator": "molmetal/scripts/cfm_cohort_matched_bar.py",
            "invocation": (
                ".venv/bin/python molmetal/scripts/cfm_cohort_matched_compare.py "
                "--run-a <A> --run-b <B> [--reading epoch|best] [--epoch N] "
                "[--bar F] [--census PATH]"
            ),
            "exit_codes": {"0": "VERDICT", "1": "REFUSED", "2": "SKIP"},
            "axes": [
                "cohort",
                "epoch_budget",
                "not_measured",
                "reading",
                "history",
                "provenance",
            ],
            "pool_kind_owner": "molmetal/scripts/cfm_pool_kind.py (Wf-A114 cluster C)",
            "atomic_write": (
                "molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file "
                "(AGENTS.md 5.6)"
            ),
            "cpu_only": True,
        },
        "bars": bars,
        "verdicts": verdicts,
        "retracted_numbers": RETRACTED_NUMBERS,
        "caveat": (
            "one run is one run: no error bar, finite generation pool "
            "(gen_eval_n=32) => sampling noise of unknown sign; 2 seeds is "
            "still n=2"
        ),
        "why_not_a_cohort_median": (
            "a cohort median averages runs that did not train the same amount "
            "on the same data, and the two cohorts are OUT OF PHASE in the "
            "same oscillation -- the ordering between them flips at nearly "
            "every budget-matched epoch index. A bar is therefore always "
            "per-run, against a NAMED run, under one reading, at one budget."
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cfm_cohort_matched_bar.py",
        description=(
            "Regenerate metrics/by_metric/cfm_cohort_matched_bar.json from "
            "the run corpus on disk, through the cohort-matched comparison "
            "harness.  CPU-only; trains nothing."
        ),
    )
    parser.add_argument(
        "--reports-root",
        default=str(DEFAULT_REPORTS_ROOT),
        help="root of the run corpus (default: molmetal/reports)",
    )
    parser.add_argument(
        "--census",
        default=str(CENSUS),
        help="census artifact used as a CROSS-CHECK ONLY (never a gate)",
    )
    parser.add_argument(
        "--out",
        default=str(ARTIFACT),
        help="where to write the artifact",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the artifact and write nothing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    reports_root = Path(args.reports_root).expanduser().resolve()
    census_path = Path(args.census).expanduser() if args.census else None

    artifact = build_artifact(reports_root, census_path)

    text = json.dumps(artifact, indent=2, sort_keys=False) + "\n"
    if args.dry_run:
        print(text)
        return 0

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(str(out), text)
    print(f"wrote {out}")

    for name, bar in artifact["bars"].items():
        print(
            f"  {name:<10} {bar['value']!r}  {bar['label']}  "
            f"cohort {bar['cohort_strict']}  epoch {bar['budget_epochs'] - 1}"
        )
    n_pass = sum(1 for v in artifact["verdicts"] if v["verdict"] == "VERDICT")
    print(
        f"  verdicts: {n_pass} VERDICT, "
        f"{len(artifact['verdicts']) - n_pass} REFUSED"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
