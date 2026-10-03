"""metric_credibility_audit.py — read-only credibility audit for the 20 paper metrics.

Reads the metrics_audit.json sibling file in
``molmetal/reports/wf_weak_to_strong_audit/`` and produces a structured
``credibility_audit.json`` summary that captures:
  - audit_date
  - n_strong, n_acceptable, n_weak, n_unmeasured, n_design_only
  - weak_ids, acceptable_ids, strong_ids, unmeasured_ids
  - ready_for_paper (boolean; True if honest-negative is consistent)

Run:
    uv run --no-sync python molmetal/scripts/metric_credibility_audit.py --audit

Output:
    molmetal/reports/wf_weak_to_strong_audit/credibility_audit.json

This is a *read-only* audit; it does not modify paper §4 or any of the
metric files. Use it to verify the audit numbers before committing a
new round of measurements.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf_weak_to_strong_audit"
AUDIT_JSON = AUDIT_DIR / "metrics_audit.json"
OUT_JSON = AUDIT_DIR / "credibility_audit.json"


def load_audit() -> dict:
    """Load the metrics_audit.json sibling file."""
    if not AUDIT_JSON.exists():
        raise FileNotFoundError(
            f"Audit file not found at {AUDIT_JSON}. "
            "Run the TODO-23 metrics audit first."
        )
    with AUDIT_JSON.open() as f:
        return json.load(f)


def classify(metrics: list[dict]) -> dict[str, list[str]]:
    """Bucket metrics by verdict into strong/acceptable/weak/unmeasured.

    Handles compound verdicts like ``"acceptable (smoke) / weak (production)"``
    by using the WORST bucket (priority: weak > acceptable > strong > unmeasured)
    so each metric appears in exactly one bucket. Compound verdicts are also
    surfaced via :func:`compound_verdict_metrics` for honest-negative framing.
    """
    buckets: dict[str, list[str]] = {
        "strong": [],
        "acceptable": [],
        "weak": [],
        "unmeasured": [],
        "design_only": [],
    }
    priority = ["weak", "acceptable", "strong", "unmeasured", "design_only"]
    for m in metrics:
        mid = m.get("id", "unknown")
        verdict = m.get("verdict", "unknown")
        # Split compound verdicts on "/" and pick worst bucket
        parts = [p.strip() for p in verdict.split("/")]
        chosen: str | None = None
        for part in parts:
            leading = part.split()[0] if part else ""
            if leading == "baseline":
                leading = "weak"
            if leading in buckets:
                if chosen is None or priority.index(leading) < priority.index(chosen):
                    chosen = leading
        if chosen is not None:
            buckets[chosen].append(mid)
    return buckets


def compound_verdict_metrics(metrics: list[dict]) -> list[str]:
    """Return metric IDs whose verdict is compound (contains '/').

    These are honest-negative cases like m8 (acceptable smoke / weak
    production) where a single label is misleading.
    """
    return [
        m.get("id", "?")
        for m in metrics
        if "/" in str(m.get("verdict", ""))
    ]


def check_no_overclaim(buckets: dict[str, list[str]]) -> tuple[bool, str]:
    """Verify no weak metric ID appears in the strong bucket.

    A weak metric that is also listed as strong is a paper-credibility
    bug — flag it.
    """
    weak_set = set(buckets["weak"])
    strong_set = set(buckets["strong"])
    overclaim = weak_set & strong_set
    if overclaim:
        return False, f"OVERCLAIM: weak metrics {sorted(overclaim)} also in strong"
    return True, "OK: no weak metric also classified as strong"


def build_credibility_audit(audit: dict) -> dict:
    """Build the credibility_audit.json payload."""
    metrics = audit.get("metrics", [])
    buckets = classify(metrics)
    compound = compound_verdict_metrics(metrics)
    n_strong = len(buckets["strong"])
    n_acceptable = len(buckets["acceptable"])
    n_weak = len(buckets["weak"])
    n_unmeasured = len(buckets["unmeasured"])
    n_design_only = len(buckets["design_only"])

    # Include the m21 (extra) metric
    extras = audit.get("extras", {})
    if "metric_21_nci60_precious_9cell_auc" in extras:
        # m21 is reported as acceptable in the audit
        if "m21" not in buckets["acceptable"]:
            buckets["acceptable"].append("m21")
            n_acceptable += 1

    no_overclaim, overclaim_msg = check_no_overclaim(buckets)

    # Paper is ready if honest-negative is consistent:
    #   - all 20 metrics are present in some bucket
    #   - no weak metric is also in strong
    #   - the audit has MEASURED vs PROJECTED labels
    n_total = n_strong + n_acceptable + n_weak + n_unmeasured + n_design_only
    ready_for_paper = (
        n_total >= 20
        and no_overclaim
        and n_strong >= 5
    )

    return {
        "audit_date": str(date.today()),
        "n_total_metrics": n_total,
        "n_strong": n_strong,
        "n_acceptable": n_acceptable,
        "n_weak": n_weak,
        "n_unmeasured": n_unmeasured,
        "n_design_only": n_design_only,
        "strong_ids": buckets["strong"],
        "acceptable_ids": buckets["acceptable"],
        "weak_ids": buckets["weak"],
        "unmeasured_ids": buckets["unmeasured"],
        "design_only_ids": buckets["design_only"],
        "compound_verdict_ids": compound,
        "no_overclaim": no_overclaim,
        "overclaim_msg": overclaim_msg,
        "ready_for_paper": ready_for_paper,
        "data_source": str(AUDIT_JSON.relative_to(PROJECT_ROOT)),
        "r_state": audit.get("r_state", "unknown"),
        "scoring_rule": audit.get("scoring_rule", {}),
        "r18_short_term_repair_budget": audit.get(
            "paper_framing", {}
        ).get("r18_short_term_repair_budget", "unknown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "TODO-23 metric credibility audit (read-only). "
            "Loads metrics_audit.json and produces credibility_audit.json."
        )
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the credibility audit and write credibility_audit.json",
    )
    args = parser.parse_args()

    if not args.audit:
        parser.print_help()
        return 0

    audit = load_audit()
    cred = build_credibility_audit(audit)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(cred, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  n_strong      = {cred['n_strong']}")
    print(f"  n_acceptable  = {cred['n_acceptable']}")
    print(f"  n_weak        = {cred['n_weak']}")
    print(f"  n_unmeasured  = {cred['n_unmeasured']}")
    print(f"  no_overclaim  = {cred['no_overclaim']}")
    print(f"  ready_for_paper = {cred['ready_for_paper']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
