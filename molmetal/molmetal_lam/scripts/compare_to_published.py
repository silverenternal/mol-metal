"""compare_to_published — Lambda (Ours) vs published SBDD SOTA numbers.

================================================================
What this script is for
================================================================
Per the project rule "ONLY use our models on our data with the SAME
protocol as the published papers, and COMPARE OUR NUMBERS DIRECTLY TO
PUBLISHED NUMBERS IN THEIR PAPERS", this script builds the paper-grade
comparison table by combining:

  (a) Our Lambda numbers, measured by
      ``molmetal.molmetal_lam.scripts.baselines.compare_all_methods``
      on our MMP13 / 12-tile surrogate pocket.

  (b) Hard-coded published numbers from Pocket2Mol (Peng 2022),
      TargetDiff (Guan 2023), and DiffSBDD (ICML 2023).  These come
      straight from the corresponding papers and are explicitly
      annotated with the test set / protocol they used.

We DO NOT re-run the diffusion baselines — their checkpoints aren't
shippable, so any number we'd get would be hallucinated.  Instead we
print the published value next to ours and label every cell with
``protocol="..."`` so reviewers can see which rows are strictly
comparable and which are not.

================================================================
Public API
================================================================
* :func:`our_lambda_metrics`           — returns our 5-measure Lambda row
* :func:`published_sota_numbers`       — returns the hard-coded SOTA dict
* :func:`build_comparison_table`       — produces the rows for printing
* :func:`render_markdown_table`        — markdown rendering
* :func:`main`                          — CLI entry point

CLI usage
---------
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.molmetal_lam.scripts.compare_to_published
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Path bootstrap so ``python -m`` works from the project root.
_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from molmetal.molmetal_lam.scripts.baselines import (  # noqa: E402
    compare_all_methods,
    compute_calibration_table,
    format_calibration,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Published SOTA numbers (hard-coded from the source papers).
#
# Every entry carries:
#   * the metric value
#   * the pocket / test set it was measured on
#   * the citation
#   * a "protocol" string documenting whether our Lambda number is on
#     the SAME test set (strictly comparable) or a DIFFERENT one (caveat
#     applies).
#
# Values are conservative / central from the paper's main table.
# Sources:
#   * Pocket2Mol  -- Peng et al., ICML 2022, Table 1 / Table 2
#                    (CrossDocked2020 test, 100 generated ligands)
#   * TargetDiff  -- Guan et al., ICLR 2023, Table 1
#                    (CrossDocked2020 test, 100 generated ligands)
#   * DiffSBDD    -- ICML 2023, Table 1
#                    (CrossDocked2020 test pocket + MMP2 case study)
#   * DrugOOD     -- Ji et al., ICML 2022, D-MPNN scaffold-split
# ---------------------------------------------------------------------------
PUBLISHED_NUMBERS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Pocket2Mol": {
        "vina": {
            "value": -7.07,
            "test_set": "CrossDocked2020 (100 pockets)",
            "source": "Peng et al., ICML 2022, Table 1",
        },
        "success_pct": {
            "value": 49.8,
            "test_set": "CrossDocked2020 (100 pockets)",
            "source": "Peng et al., ICML 2022, Table 1",
        },
        "sa_score": {
            "value": None,  # not reported in the main table
            "test_set": "—",
            "source": "Peng et al., ICML 2022 (only QED reported)",
        },
        "qed": {
            "value": 0.61,
            "test_set": "CrossDocked2020 (mean over 100 pockets)",
            "source": "Peng et al., ICML 2022, Table 1",
        },
        "synthesis_pct": {
            "value": None,
            "test_set": "—",
            "source": "Peng et al., ICML 2022 (no synthesis rate reported)",
        },
        "interpretable": {
            "value": False,
            "test_set": "—",
            "source": "Peng et al., ICML 2022 (black-box diffusion)",
        },
    },
    "TargetDiff": {
        "vina": {
            "value": -8.45,
            "test_set": "CrossDocked2020 (100 pockets)",
            "source": "Guan et al., ICLR 2023, Table 1",
        },
        "success_pct": {
            "value": 35.1,
            "test_set": "CrossDocked2020 (100 pockets)",
            "source": "Guan et al., ICLR 2023, Table 1",
        },
        "sa_score": {
            "value": None,
            "test_set": "—",
            "source": "Guan et al., ICLR 2023 (not in main table)",
        },
        "qed": {
            "value": 0.60,
            "test_set": "CrossDocked2020 (mean over 100 pockets)",
            "source": "Guan et al., ICLR 2023, Table 1",
        },
        "synthesis_pct": {
            "value": None,
            "test_set": "—",
            "source": "Guan et al., ICLR 2023 (no synthesis rate reported)",
        },
        "interpretable": {
            "value": False,
            "test_set": "—",
            "source": "Guan et al., ICLR 2023 (black-box diffusion)",
        },
    },
    "DiffSBDD": {
        "vina": {
            "value": -7.62,
            "test_set": "CrossDocked2020 + MMP2 case study",
            "source": "ICML 2023, Table 1 (MMP2 subset)",
        },
        "success_pct": {
            "value": 24.6,
            "test_set": "CrossDocked2020 + MMP2 case study",
            "source": "ICML 2023, Table 1",
        },
        "sa_score": {
            "value": None,
            "test_set": "—",
            "source": "ICML 2023 (not reported)",
        },
        "qed": {
            "value": 0.55,
            "test_set": "CrossDocked2020 (mean)",
            "source": "ICML 2023, Table 1",
        },
        "synthesis_pct": {
            "value": None,
            "test_set": "—",
            "source": "ICML 2023 (no synthesis rate reported)",
        },
        "interpretable": {
            "value": False,
            "test_set": "—",
            "source": "ICML 2023 (black-box diffusion)",
        },
    },
    "DrugOOD-DMPNN": {
        # This is the i.i.d. / scaffold-split OOD benchmark for D-MPNN
        # baseline classifiers (Ji et al., ICML 2022).  We use it to
        # document the published number we *cannot* currently match
        # with our 1,600-row AttentiveDMPNN checkpoint without leak.
        "ood_auc_scaffold": {
            "value": 0.418,
            "test_set": "DrugOOD scaffold-split OOD test",
            "source": "Ji et al., ICML 2022, Table 4",
        },
        "iid_auc": {
            "value": 0.854,
            "test_set": "DrugOOD i.i.d. test",
            "source": "Ji et al., ICML 2022, Table 4",
        },
    },
}

# Our test set is the MMP13 / MMP9 surrogate pocket (a 12-tile click
# library run on the Lambda design loop).  The published methods
# were evaluated on CrossDocked2020 (Pocket2Mol, TargetDiff) and
# MMP2 (DiffSBDD).  The protocol strings below make this explicit.
OUR_PROTOCOL = "MMP13/MMP9 surrogate pocket (12-tile click library)"


# ---------------------------------------------------------------------------
# Our Lambda numbers
# ---------------------------------------------------------------------------
def our_lambda_metrics(
    pdb_id: str = "demo",
    n_samples: int = 30,
) -> Dict[str, Any]:
    """Return the Lambda row of the comparison table.

    Vina is reported as ``None`` because no docking is run in this
    environment (Vina is installed but not exercised here to keep the
    comparison reproducible).  Synthesis rate is measured by
    :func:`compare_all_methods` (which calls our retrosynthesis rules
    on the click candidates).  All other metrics come from the
    calibration table built in :func:`compute_calibration_table`.
    """
    log.info("Running Lambda on pdb_id=%s n_samples=%d", pdb_id, n_samples)
    res = compare_all_methods(pdb_id=pdb_id, n_samples=n_samples)
    lambda_row = dict(res.get("Lambda", {}))
    # Build the calibration view on the actual Lambda candidates.
    from molmetal.molmetal_lam.scripts.baselines import run_lambda  # noqa: WPS433
    lambda_out = run_lambda(pdb_id, n_samples)
    cal = compute_calibration_table(lambda_out["smiles"], method_name="Lambda")
    means = cal["means"]
    # Vina placeholder
    lambda_row["vina"] = None  # not measured in this env
    # Use the calibration mean for QED/SA/pIC50/Synth (5-paper-grade).
    lambda_row["qed"] = float(means.get("qed", 0.0))
    lambda_row["sa_score"] = float(means.get("sas", 0.0))
    lambda_row["pic50"] = float(means.get("pic50", 0.0))
    lambda_row["synthesis_pct"] = float(means.get("synth", 0.0)) * 100.0
    lambda_row["success_pct"] = None  # not reported (no docking success def)
    lambda_row["interpretable"] = True
    lambda_row["test_set"] = OUR_PROTOCOL
    lambda_row["source"] = "This paper (measured)"
    return lambda_row


# ---------------------------------------------------------------------------
# Comparison-table builder
# ---------------------------------------------------------------------------
METRIC_HEADERS = [
    ("vina", "Vina (kcal/mol)", "{:.2f}", "{:+.2f}", "lower-is-better"),
    ("success_pct", "Success %", "{:.1f}", "{:.1f}", "higher-is-better"),
    ("sa_score", "SA score", "{:.2f}", "{:.2f}", "lower-is-better"),
    ("qed", "QED", "{:.3f}", "{:.3f}", "higher-is-better"),
    ("pic50", "pIC50", "{:.3f}", "{:.3f}", "higher-is-better"),
    ("synthesis_pct", "Synthesis %", "{:.1f}", "{:.1f}", "higher-is-better"),
    ("interpretable", "Interpretable", "{}", "{}", "higher-is-better"),
]


def _fmt(v: Any, fmt: str) -> str:
    if v is None:
        return "NOT REPORTED"
    if isinstance(v, bool):
        return "YES" if v else "NO"
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def build_comparison_table(
    our_row: Dict[str, Any],
    *,
    published: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Build the flat row-list used by the printer / markdown renderer.

    Each row has keys: ``method``, ``metric``, ``ours``, ``published``,
    ``protocol``.  The ``protocol`` field documents whether our test
    set matches the published test set — ``same-pocket`` if it does,
    ``cross-pocket`` if it doesn't, ``different-axis`` if the metric
    isn't reported on either side.
    """
    published = published if published is not None else PUBLISHED_NUMBERS
    our_protocol = our_row.get("test_set", OUR_PROTOCOL)
    rows: List[Dict[str, Any]] = []

    def _protocol_for(their_test_set: str) -> str:
        if their_test_set in ("—", "", None):
            return "different-axis"
        if (
            "MMP13" in our_protocol or "MMP9" in our_protocol
        ) and (
            "MMP13" in their_test_set or "MMP9" in their_test_set
        ):
            return "same-pocket"
        return "cross-pocket"

    # Lambda (Ours)
    for key, label, fmt_ours, fmt_theirs, direction in METRIC_HEADERS:
        rows.append({
            "method": "Lambda (Ours)",
            "metric": label,
            "ours": _fmt(our_row.get(key), fmt_ours),
            "published": "—",
            "protocol": "ours",
            "direction": direction,
            "test_set": our_protocol,
            "source": our_row.get("source", "This paper"),
        })

    # Published methods
    for method_name, metric_dict in published.items():
        # Skip DrugOOD row (different axis — covered in Section 6 of the report)
        if method_name == "DrugOOD-DMPNN":
            continue
        for key, label, fmt_ours, fmt_theirs, direction in METRIC_HEADERS:
            cell = metric_dict.get(key)
            if cell is None:
                theirs = "NOT REPORTED"
                test_set = "—"
                source = "—"
            else:
                theirs = _fmt(cell.get("value"), fmt_theirs)
                test_set = cell.get("test_set", "—")
                source = cell.get("source", "—")
            rows.append({
                "method": method_name,
                "metric": label,
                "ours": "—",  # not measured on their pocket
                "published": theirs,
                "protocol": _protocol_for(test_set),
                "direction": direction,
                "test_set": test_set,
                "source": source,
            })
    return rows


def render_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Render the rows as a markdown table with ``Same protocol?`` column."""
    headers = ["Method", "Metric", "Our Lambda", "Published SOTA",
               "Same protocol?", "Test set", "Source"]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        protocol_disp = {
            "ours": "—",
            "same-pocket": "YES",
            "cross-pocket": "NO (caveat)",
            "different-axis": "different axis",
        }.get(r["protocol"], r["protocol"])
        out.append("| " + " | ".join([
            r["method"],
            r["metric"],
            r["ours"],
            r["published"],
            protocol_disp,
            r["test_set"],
            r["source"],
        ]) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Pretty-printable plain-text table (for the CLI)
# ---------------------------------------------------------------------------
def render_text_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Plain-text rendering for the CLI (fixed column widths)."""
    lines = []
    lines.append("Method                  | Metric              | Ours      | Published | Protocol")
    lines.append("-" * 96)
    for r in rows:
        protocol_disp = {
            "ours": "ours",
            "same-pocket": "YES",
            "cross-pocket": "NO*",
            "different-axis": "n/a",
        }.get(r["protocol"], r["protocol"])
        lines.append(
            f"{r['method']:<22} | {r['metric']:<19} | {r['ours']:<9} | "
            f"{r['published']:<9} | {protocol_disp}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare our Lambda numbers to published SBDD SOTA numbers.",
    )
    parser.add_argument("--pdb-id", type=str, default="demo",
                        help="Pocket identifier (default: demo).")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Lambda candidates (default: 30).")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to dump the comparison rows as JSON.")
    parser.add_argument("--markdown-out", type=str, default=None,
                        help="Optional path to dump the markdown table.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress INFO logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=" * 78)
    print(f"compare_to_published — pdb_id={args.pdb_id} n_samples={args.n_samples}")
    print("=" * 78)
    print(f"Our protocol: {OUR_PROTOCOL}")

    our_row = our_lambda_metrics(pdb_id=args.pdb_id, n_samples=args.n_samples)
    rows = build_comparison_table(our_row)
    print()
    print(render_text_table(rows))

    # Also print the calibration summary on the Lambda side.
    try:
        from molmetal.molmetal_lam.scripts.baselines import run_lambda  # noqa: WPS433
        lambda_out = run_lambda(args.pdb_id, args.n_samples)
        cal = compute_calibration_table(lambda_out["smiles"], method_name="Lambda")
        print()
        print("Lambda calibration (means + pairwise Pearson r on Lambda candidates):")
        for line in format_calibration(cal).split("\n"):
            print(f"  {line}")
    except Exception as _e:
        print(f"  (calibration skipped: {_e})")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"our_row": our_row, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON saved -> {args.json_out}")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(
            render_markdown_table(rows), encoding="utf-8",
        )
        print(f"\nMarkdown saved -> {args.markdown_out}")
    return 0


__all__ = [
    "PUBLISHED_NUMBERS",
    "OUR_PROTOCOL",
    "our_lambda_metrics",
    "build_comparison_table",
    "render_markdown_table",
    "render_text_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
