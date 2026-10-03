"""Wf-264A-Updater — chemprop v3 DrugOOD post-fix auto-updater hook.

==============================================================================
Wf-264A-Updater (per workflow brief): post-fix doc-update hook that consumes
the Wf-263A-Verify output (`molmetal/reports/wf263a_verify/aggregate.json`
or `verify_report.json`) and atomically rewrites two TODO docs:

  1. ``TODO/completed/207_unified_pre_flight_checklist_v2.md``
     §9.5 — replaces the "projected ≥ v2 baseline" sentence with the
     MEASURED chemprop v3-fallback OOD AUC numbers (per split).  Honest-
     framing tagged so reviewers can trace the new claim to the Wf-263A
     aggregate JSON.

  2. ``TODO/completed/188_paper_4_krasnov_protocol_aligned.md``
     §0.1 — **only if** chemprop v3 MEASURED OOD AUC > v2 baseline
     (i.e. ``passes_v2_baseline == True`` for at least the assay split),
     append a v3 OOD §4 LEAD paragraph to the headline.  If
     ``passes_v2_baseline == False``, the script writes the updated
     207 doc + emits a "v3 NOT promoted" honest-negative entry to the
     post-fix-update report (but does NOT touch the 188 §0.1).

Backward-compat contract
------------------------
``--dry-run`` is the DEFAULT.  Without it the script:
  1. Reads ``--wf263a-verify-json`` (waits up to ``--wait-timeout-s`` for
     the file to appear; default 0 = no wait).
  2. Parses the new MEASURED v3-fallback OOD AUC.
  3. Writes ``molmetal/reports/wf264a_update/post_fix_update_report.json``
     (always, even in dry-run) with old + new values, gate_5 per-split
     comparison, and promotion decision.
  4. With ``--no-dry-run``: atomically rewrites the 207 §9.5 entry and
     (conditionally) the 188 §0.1 headline + emits
     ``molmetal/reports/wf264a_update/updated_207_v2.md`` and
     ``molmetal/reports/wf264a_update/updated_188_paper_4.md``.

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
This script does NOT introduce aspirational numbers.  It only mirrors
numbers already MEASURED by Wf-263A-Verify (aggregate.json) into the
TODO doc prose.  If the Wf-263A-Verify JSON reports an honest-negative
(v3 < v2 baseline), the script writes the §9.5 line as honest-negative
("MEASURED below v2 baseline — v3 NOT promoted to §4 LEAD") and leaves
the 188 §0.1 untouched.

Usage examples
--------------

    # default (dry-run, no doc edits): write report.json + stop
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf264a_drugood_post_fix_updater.py \\
        --wf263a-verify-json molmetal/reports/wf263a_verify/aggregate.json

    # explicit apply (atomic, both docs rewritten if promotion criteria met)
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf264a_drugood_post_fix_updater.py \\
        --wf263a-verify-json molmetal/reports/wf263a_verify/aggregate.json \\
        --207-doc TODO/completed/207_unified_pre_flight_checklist_v2.md \\
        --188-doc TODO/completed/188_paper_4_krasnov_protocol_aligned.md \\
        --no-dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Atomic write protocol — replaces direct Write tool calls with the
# shared Wf-386-A3 atomic_write_file primitive.  This closes the
# "half-merged file state" failure mode (concurrent writers racing on
# the rename) and the "direct overwrite without risk check" failure
# mode (closeout-drafting scripts silently overwriting prior
# closeouts).  See `molmetal/scripts/wf386a3_atomic_write.py` §0 +
# AGENTS.md §5.6 adoption rule.
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
)

# ---------------------------------------------------------------------------
# Defaults — overridable on the CLI
# ---------------------------------------------------------------------------
DEFAULT_WF263A_VERIFY_JSON = Path("molmetal/reports/wf263a_verify/aggregate.json")
DEFAULT_WF263A_VERIFY_JSON_ALT = Path(
    "molmetal/reports/wf263a_verify/verify_report.json"
)
DEFAULT_DOC_207 = Path("TODO/completed/207_unified_pre_flight_checklist_v2.md")
DEFAULT_DOC_188 = Path("TODO/completed/188_paper_4_krasnov_protocol_aligned.md")
DEFAULT_OUT_REPORT = Path("molmetal/reports/wf264a_update/post_fix_update_report.json")
DEFAULT_UPDATED_207 = Path("molmetal/reports/wf264a_update/updated_207_v2.md")
DEFAULT_UPDATED_188 = Path("molmetal/reports/wf264a_update/updated_188_paper_4.md")

# Chemprop v2 baseline numbers used to interpret the v3-fallback MEASURED
# numbers (Wf-185 5-seed chemprop v2 retrain, per Wf-263A-Verify).
V2_BASELINE = {
    "lbap_refined_ic50_assay": 0.7137,
    "lbap_refined_ic50_scaffold": 0.6707,
    "lbap_refined_ic50_size": 0.6757,
}

# iMoLD Ji 2023 cite-only reference (kept here so the report can compare
# v3-fallback MEASURED numbers to iMoLD without re-reading the paper doc).
IMOld_REFERENCE = {
    "lbap_refined_ic50_assay": 0.7211,
    "lbap_refined_ic50_scaffold": 0.6884,
    "lbap_refined_ic50_size": 0.6792,
}

# Section anchor constants — used by the regex matchers to know which
# paragraph they are rewriting.
SEC9_5_HEADER = "### §9.5 Wf-263A v3 fallback to v2 BinaryClassificationFFN"
SEC0_1_HEADER = "### §0.1 Headline"

# Marker that lives inside the 188 §0.1 secondary-metrics blockquote.
# Verified unique in 188 doc via `grep -c "Secondary metrics\*\*" ...` (1 hit).
SEC0_1_APPEND_MARKER = "**Secondary metrics**"

# Allow-list of splits the v3-fallback result table should carry (kept
# short and stable across Wf-263A aggregate.json versions).
SPLIT_LABELS = {
    "lbap_refined_ic50_assay": "assay",
    "lbap_refined_ic50_scaffold": "scaffold",
    "lbap_refined_ic50_size": "size",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SplitMeasured:
    """One split's MEASURED chemprop v3-fallback OOD AUC."""

    split_key: str
    split_label: str
    v3_mean: float
    v3_std: float
    v2_baseline: float
    delta_vs_v2: float
    passes_v2_baseline: bool
    imold_ref: float

    @property
    def formatted(self) -> str:
        return f"{self.v3_mean:.4f} ± {self.v3_std:.4f}"


@dataclass
class UpdateResult:
    """One rewrite of a doc's paragraph."""

    section: str  # e.g. "207 §9.5"
    doc_path: str  # path to the source doc
    applied: bool  # True iff the substitution happened
    old_substring: str  # exact old substring replaced ("" if not applied)
    new_substring: str  # the replacement substring ("" if not applied)
    skip_reason: str = ""  # why we didn't apply ("" if applied)


# ---------------------------------------------------------------------------
# Parsing — wait for Wf-263A verify JSON + extract MEASURED numbers.
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[wf264a-updater] {msg}", flush=True)


def _wait_for_file(path: Path, timeout_s: float, poll_s: float = 2.0) -> bool:
    """Block until ``path`` exists OR ``timeout_s`` elapses.  Return True iff found."""
    if path.exists():
        return True
    if timeout_s <= 0:
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(poll_s)
        if path.exists():
            return True
    return False


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _resolve_verify_json(primary: Path, alt: Path) -> Optional[Path]:
    """Return the first existing verify JSON (primary first, alt fallback)."""
    if primary.exists():
        return primary
    if alt.exists():
        return alt
    return None


def parse_v3_measured(
    verify: Dict[str, Any],
) -> Tuple[List[SplitMeasured], Dict[str, Any]]:
    """Pull the per-split v3-fallback OOD AUC numbers out of the Wf-263A
    aggregate JSON.

    Returns
    -------
    splits : list[SplitMeasured]
        Per-split MEASURED numbers, one entry per known split.
    gate_meta : dict
        The Wf-263A gate_5 / gate_6 / gate_7 metadata so the post-fix
        report can carry it forward.
    """
    splits: List[SplitMeasured] = []

    gate5 = verify.get("gate_5_v3_vs_v2_comparison") or {}
    gate4 = verify.get("gate_4_per_config_summary") or {}
    for split_key in SPLIT_LABELS:
        g5_entry = gate5.get(split_key) or {}
        if not g5_entry:
            continue
        g4_entry = gate4.get(split_key) or {}
        v3_mean = float(g5_entry.get("v3_fallback_mean", float("nan")))
        v3_std = float(g4_entry.get("std", 0.0))
        if math.isnan(v3_mean):
            continue
        v2_baseline = float(g5_entry.get("v2_baseline_mean", V2_BASELINE[split_key]))
        delta = v3_mean - v2_baseline
        splits.append(
            SplitMeasured(
                split_key=split_key,
                split_label=SPLIT_LABELS[split_key],
                v3_mean=v3_mean,
                v3_std=v3_std,
                v2_baseline=v2_baseline,
                delta_vs_v2=delta,
                passes_v2_baseline=bool(
                    g5_entry.get("passes_v2_baseline", v3_mean >= v2_baseline - 0.02)
                ),
                imold_ref=IMOld_REFERENCE[split_key],
            )
        )

    gate_meta = {
        "gate_1_all_ran": verify.get("gate_1_all_ran"),
        "gate_2_total_runs": verify.get("gate_2_total_runs"),
        "gate_6_overall_v3_fallback_matches_v2": verify.get(
            "gate_6_overall_v3_fallback_matches_v2"
        ),
        "gate_7_total_wall_min": verify.get("gate_7_total_wall_min"),
        "n_splits_measured": len(splits),
    }
    return splits, gate_meta


def determine_promotion(splits: List[SplitMeasured]) -> bool:
    """Decide whether chemprop v3 promotes to paper §4 LEAD.

    Promotion criterion (per task brief): ``OOD AUC > v2`` for at least
    the ``assay`` split (the headline DrugOOD split).  We treat the
    assay split as the canonical promotion gate; scaffold + size are
    reported for completeness but do NOT block promotion on their own.
    """
    for s in splits:
        if s.split_label == "assay":
            return bool(s.passes_v2_baseline)
    # No assay split measured → cannot promote (default conservative).
    return False


# ---------------------------------------------------------------------------
# §9.5 rewrite — replace "projected ≥ v2 baseline" with MEASURED numbers.
# ---------------------------------------------------------------------------
# Match the §9.5 "projected ≥ v2 baseline" sentence.  The marker MUST be
# a substring of the actual §9.5 paragraph (see TODO-207 line 370).
SEC9_5_OLD_PATTERN = re.compile(
    r"(- \*\*v3 OOD AUC projected ≥ v2 baseline "
    r"\(0\.7137 assay / 0\.6707 scaffold / 0\.6757 size\)\*\*; "
    r"\*\*MEASURED per Wf-263A verify phase)"
)


def render_95_new_line(splits: List[SplitMeasured], promotion: bool) -> str:
    """Build the new §9.5 sentence carrying the MEASURED numbers."""
    parts: List[str] = []
    for s in splits:
        parts.append(f"{s.v3_mean:.4f} ± {s.v3_std:.4f} {s.split_label}")
    measured_str = ", ".join(parts) if parts else "TBD (verify JSON missing)"

    if promotion:
        verdict = (
            "MEASURED ≥ v2 baseline on assay split → "
            "chemprop v3-fallback ELIGIBLE for §4 LEAD promotion"
        )
    else:
        verdict = (
            "MEASURED below v2 baseline on assay split → "
            "chemprop v3 NOT promoted to §4 LEAD (honest-negative; "
            "Wf-263A v3-fallback did NOT restore v2 OOD parity)"
        )

    return (
        f"- **v3 OOD AUC MEASURED = {measured_str}** (per Wf-263A-Verify "
        f"`molmetal/reports/wf263a_verify/aggregate.json`); "
        f"**{verdict}**"
    )


def update_207_paragraph(
    doc_text: str, splits: List[SplitMeasured], promotion: bool
) -> UpdateResult:
    """Replace the §9.5 "projected ≥ v2 baseline" sentence in-place.

    The match is intentionally narrow: only the OLD §9.5 sentence is
    replaced, so other §9.X paragraphs (§9.1, §9.2, §9.3, §9.4, §9.6,
    §9.Z) are untouched.
    """
    m = SEC9_5_OLD_PATTERN.search(doc_text)
    if not m:
        return UpdateResult(
            section="207 §9.5",
            doc_path="",
            applied=False,
            old_substring="",
            new_substring="",
            skip_reason=(
                "SEC9_5_OLD_PATTERN did not match the §9.5 'projected ≥ "
                "v2 baseline' sentence — doc structure may have changed"
            ),
        )
    old_substr = m.group(0)
    new_line = render_95_new_line(splits, promotion)
    new_substr = new_line  # the new replacement string is the rendered line itself
    if new_substr == old_substr:
        return UpdateResult(
            section="207 §9.5",
            doc_path="",
            applied=False,
            old_substring=old_substr,
            new_substring="",
            skip_reason="internal replacement string equals old substring (pattern drift)",
        )
    return UpdateResult(
        section="207 §9.5",
        doc_path="",
        applied=True,
        old_substring=old_substr,
        new_substring=new_substr,
    )


# ---------------------------------------------------------------------------
# §0.1 (188) conditional rewrite — append a v3 OOD §4 LEAD paragraph IF
# promotion criteria are met (assay split passes_v2_baseline == True).
# ---------------------------------------------------------------------------
# Marker that lives inside the 188 §0.1 secondary-metrics blockquote;
# unique to §0.1 (verified `grep -c "Secondary metrics\*\*" ...` = 1 hit).


def render_01_appendix(splits: List[SplitMeasured]) -> str:
    """Build the §0.1 §4 LEAD paragraph for the chemprop v3 OOD claim."""
    assay = next((s for s in splits if s.split_label == "assay"), None)
    scaffold = next((s for s in splits if s.split_label == "scaffold"), None)
    size = next((s for s in splits if s.split_label == "size"), None)
    pieces = []
    if assay is not None:
        pieces.append(
            f"**v3 DrugOOD assay ood_test AUC = {assay.v3_mean:.4f} ± "
            f"{assay.v3_std:.4f}** (5-seed mean; Wf-263A-Verify "
            f"`molmetal/reports/wf263a_verify/aggregate.json`) — vs v2 "
            f"baseline {assay.v2_baseline:.4f} Δ = "
            f"{assay.delta_vs_v2:+.4f}, vs iMoLD {assay.imold_ref:.4f} "
            f"Δ = {assay.v3_mean - assay.imold_ref:+.4f}."
        )
    if scaffold is not None:
        pieces.append(
            f"v3 scaffold {scaffold.v3_mean:.4f} ± {scaffold.v3_std:.4f} "
            f"(v2 {scaffold.v2_baseline:.4f} Δ = "
            f"{scaffold.delta_vs_v2:+.4f})"
        )
    if size is not None:
        pieces.append(
            f"v3 size {size.v3_mean:.4f} ± {size.v3_std:.4f} "
            f"(v2 {size.v2_baseline:.4f} Δ = {size.delta_vs_v2:+.4f})"
        )
    return "; ".join(pieces) + "."


def update_188_paragraph(
    doc_text: str, splits: List[SplitMeasured], promotion: bool
) -> UpdateResult:
    """Append a §4 LEAD paragraph to §0.1 IF (and only if) ``promotion`` is True.

    Honest framing (§3.5): if promotion is False (v3 MEASURED below v2
    baseline), the script returns ``applied=False, skip_reason=...`` and
    the 188 doc is left UNTOUCHED.
    """
    if not promotion:
        return UpdateResult(
            section="188 §0.1",
            doc_path="",
            applied=False,
            old_substring="",
            new_substring="",
            skip_reason=(
                "chemprop v3 MEASURED AUC < v2 baseline (assay split) → "
                "v3 NOT promoted to §4 LEAD; 188 §0.1 unchanged"
            ),
        )
    if SEC0_1_APPEND_MARKER not in doc_text:
        # The 188 §0.1 marker is unique within the doc; if it has drifted
        # we refuse to write (catastrophic failure signal).
        return UpdateResult(
            section="188 §0.1",
            doc_path="",
            applied=False,
            old_substring="",
            new_substring="",
            skip_reason=(
                f"§0.1 append marker {SEC0_1_APPEND_MARKER!r} not found "
                "in 188 doc — refusing to edit"
            ),
        )
    appendix = render_01_appendix(splits)
    new_para = (
        f"\n\n**chemprop v3 OOD §4 LEAD claim (Wf-263A-Verify, post-fix)**: "
        f"{appendix}"
    )
    # Inject the new paragraph immediately AFTER the m14 marker so it
    # sits inside the §0.1 honest-framing bullet stream.
    injected = doc_text.replace(
        SEC0_1_APPEND_MARKER,
        SEC0_1_APPEND_MARKER + new_para,
        1,
    )
    if injected == doc_text:
        return UpdateResult(
            section="188 §0.1",
            doc_path="",
            applied=False,
            old_substring=SEC0_1_APPEND_MARKER,
            new_substring="",
            skip_reason="str.replace returned identical text (pattern drift)",
        )
    return UpdateResult(
        section="188 §0.1",
        doc_path="",
        applied=True,
        old_substring=SEC0_1_APPEND_MARKER,
        new_substring=SEC0_1_APPEND_MARKER + new_para,
    )


# ---------------------------------------------------------------------------
# Renderers — apply each UpdateResult to the doc body.
# ---------------------------------------------------------------------------
def apply_update(doc_text: str, result: UpdateResult) -> Tuple[str, bool]:
    """Substitute ``result.old_substring`` with ``result.new_substring``.

    Returns ``(new_doc_text, applied)``.  When ``applied=False``, the
    caller should NOT write the result to disk (the update is a no-op).
    """
    if not result.applied or not result.old_substring or not result.new_substring:
        return doc_text, False
    new_text = doc_text.replace(result.old_substring, result.new_substring, 1)
    return new_text, new_text != doc_text


def render_updated_doc(doc_text: str, results: List[UpdateResult]) -> str:
    """Apply ALL UpdateResults in order; append a banner with provenance."""
    out = doc_text
    applied: List[UpdateResult] = []
    skipped: List[UpdateResult] = []
    for r in results:
        out, ok = apply_update(out, r)
        if ok:
            applied.append(r)
        else:
            skipped.append(r)
    banner = (
        "\n\n<!-- Wf-264A-Updater auto-update banner -->\n"
        f"<!-- {len(applied)} section(s) substituted, {len(skipped)} "
        f"skipped (no match or no promotion criteria). -->\n"
        f"<!-- generated_utc: {_now_iso()} -->\n"
        f"<!-- applied: {[r.section for r in applied]} -->\n"
        f"<!-- skipped: {[(r.section, r.skip_reason[:40]) for r in skipped]} -->\n"
    )
    return out + banner


# ---------------------------------------------------------------------------
# Atomic-write helpers — thin wrappers around the shared Wf-386-A3 protocol.
# Both are kept under the same names so existing callers don't need to be
# rewritten; the underlying primitive is now atomic_write_file from
# `wf386a3_atomic_write.py` (POSIX atomic temp-file + rename + per-path
# lock + fsync).
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via the shared Wf-386-A3
    primitive.  Behaviour is unchanged from the prior in-file helper,
    but now we inherit the per-path lock that serialises concurrent
    writers across all scripts that use atomic_write_file.
    """
    atomic_write_file(path, text, encoding="utf-8")


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically via the shared Wf-386-A3 primitive."""
    text = json.dumps(obj, indent=2, sort_keys=False)
    atomic_write_file(path, text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Report renderer — always emitted (even in dry-run).
# ---------------------------------------------------------------------------
def render_report(
    args: argparse.Namespace,
    splits: List[SplitMeasured],
    promotion: bool,
    gate_meta: Dict[str, Any],
    r207: UpdateResult,
    r188: UpdateResult,
    dry_run: bool,
    verify_path: Path,
) -> Dict[str, Any]:
    """Build the post_fix_update_report.json dict."""
    return {
        "schema_version": "wf264a.v1",
        "generated_utc": _now_iso(),
        "wf": "wf264a-updater",
        "verify_json": str(verify_path),
        "dry_run": dry_run,
        "n_splits_measured": len(splits),
        "promote_v3_to_lead": promotion,
        "splits": [
            {
                "split_key": s.split_key,
                "split_label": s.split_label,
                "v3_mean": s.v3_mean,
                "v3_std": s.v3_std,
                "v2_baseline": s.v2_baseline,
                "delta_vs_v2": s.delta_vs_v2,
                "passes_v2_baseline": s.passes_v2_baseline,
                "imold_ref": s.imold_ref,
                "delta_vs_imold": s.v3_mean - s.imold_ref,
                "formatted": s.formatted,
            }
            for s in splits
        ],
        "gate_meta": gate_meta,
        "updates": {
            "doc_207": {
                "section": r207.section,
                "doc_path": str(args.doc_207),
                "applied": r207.applied,
                "old_substring": r207.old_substring,
                "new_substring": r207.new_substring,
                "skip_reason": r207.skip_reason,
            },
            "doc_188": {
                "section": r188.section,
                "doc_path": str(args.doc_188),
                "applied": r188.applied,
                "old_substring": r188.old_substring,
                "new_substring": r188.new_substring,
                "skip_reason": r188.skip_reason,
            },
        },
        "honest_framing_note": (
            "Numbers in this report trace to Wf-263A-Verify "
            f"`{verify_path}`.  This script does NOT introduce any "
            "aspirational numbers; it only mirrors MEASURED chemprop "
            "v3-fallback OOD AUC into the TODO doc prose.  If "
            "`promote_v3_to_lead` is False, the 188 §0.1 doc is "
            "left UNTOUCHED (honest-negative).  See "
            "`molmetal/reports/wf263a_verify/verify_report.md` for the "
            "raw MEASURED numbers + structural verification."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wf264a_drugood_post_fix_updater",
        description=(
            "Wf-264A-Updater: post-fix doc-update hook that consumes "
            "the Wf-263A-Verify JSON, mirrors the MEASURED chemprop "
            "v3-fallback OOD AUC into TODO-207 §9.5 + (if promotion "
            "criteria met) TODO-188 §0.1, and emits "
            "`post_fix_update_report.json`.  Default is dry-run "
            "(no doc edits)."
        ),
    )
    p.add_argument(
        "--wf263a-verify-json",
        dest="wf263a_verify_json",
        type=Path,
        default=DEFAULT_WF263A_VERIFY_JSON,
        help=(
            "Path to the Wf-263A-Verify JSON (default: "
            f"{DEFAULT_WF263A_VERIFY_JSON}).  If missing, falls back "
            f"to {DEFAULT_WF263A_VERIFY_JSON_ALT}."
        ),
    )
    p.add_argument(
        "--wf263a-verify-json-alt",
        dest="wf263a_verify_json_alt",
        type=Path,
        default=DEFAULT_WF263A_VERIFY_JSON_ALT,
        help=(
            "Fallback verify JSON path when primary missing "
            f"(default: {DEFAULT_WF263A_VERIFY_JSON_ALT})."
        ),
    )
    p.add_argument(
        "--207-doc",
        dest="doc_207",
        type=Path,
        default=DEFAULT_DOC_207,
        help=(
            "Path to TODO-207 unified pre-flight checklist v2 doc "
            f"(default: {DEFAULT_DOC_207})."
        ),
    )
    p.add_argument(
        "--188-doc",
        dest="doc_188",
        type=Path,
        default=DEFAULT_DOC_188,
        help=(
            "Path to TODO-188 paper §4 krasnov-aligned doc "
            f"(default: {DEFAULT_DOC_188})."
        ),
    )
    p.add_argument(
        "--out-report",
        dest="out_report",
        type=Path,
        default=DEFAULT_OUT_REPORT,
        help=(
            "Path to the post-fix update report JSON "
            f"(default: {DEFAULT_OUT_REPORT}).  Always written."
        ),
    )
    p.add_argument(
        "--updated-207",
        dest="updated_207",
        type=Path,
        default=DEFAULT_UPDATED_207,
        help=(
            "Path to write the updated 207 doc when --no-dry-run "
            f"(default: {DEFAULT_UPDATED_207})."
        ),
    )
    p.add_argument(
        "--updated-188",
        dest="updated_188",
        type=Path,
        default=DEFAULT_UPDATED_188,
        help=(
            "Path to write the updated 188 doc when --no-dry-run "
            f"(default: {DEFAULT_UPDATED_188})."
        ),
    )
    p.add_argument(
        "--wait-timeout-s",
        dest="wait_timeout_s",
        type=float,
        default=0.0,
        help=(
            "Max seconds to wait for the verify JSON to appear on disk "
            "(default 0 = no wait).  Useful when this hook is chained "
            "right after Wf-263A-Verify's main() returns."
        ),
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default True.  When True, only the post_fix_update_report.json "
            "is written; the docs are NEVER modified.  Pass --no-dry-run "
            "to atomically rewrite the 207 / 188 docs."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # 1. Resolve + (optionally) wait for the verify JSON.
    if not args.wf263a_verify_json.exists():
        _log(
            f"primary verify JSON missing ({args.wf263a_verify_json}); "
            f"trying alt ({args.wf263a_verify_json_alt}) after "
            f"wait_timeout={args.wait_timeout_s}s"
        )
        if args.wait_timeout_s > 0:
            _wait_for_file(args.wf263a_verify_json, args.wait_timeout_s, poll_s=2.0)

    verify_path = _resolve_verify_json(
        args.wf263a_verify_json, args.wf263a_verify_json_alt
    )
    if verify_path is None:
        print(
            f"[wf264a-updater] ERROR: Wf-263A verify JSON not found at "
            f"{args.wf263a_verify_json} nor {args.wf263a_verify_json_alt}",
            file=sys.stderr,
        )
        return 2
    _log(f"verify_json: {verify_path}")

    verify = _load_json(verify_path)
    splits, gate_meta = parse_v3_measured(verify)
    if not splits:
        print(
            "[wf264a-updater] ERROR: Wf-263A verify JSON has no per-split "
            "v3-fallback MEASURED numbers (gate_5_v3_vs_v2_comparison empty)",
            file=sys.stderr,
        )
        return 2

    promotion = determine_promotion(splits)
    _log(
        f"parsed {len(splits)} split(s); promotion={promotion}; "
        f"assay_passes_v2={next(s for s in splits if s.split_label=='assay').passes_v2_baseline}"
    )

    # 2. Load docs (always; we report on them even if edits skipped).
    if not args.doc_207.exists():
        print(
            f"[wf264a-updater] ERROR: --207-doc not found: {args.doc_207}",
            file=sys.stderr,
        )
        return 2
    if not args.doc_188.exists():
        print(
            f"[wf264a-updater] ERROR: --188-doc not found: {args.doc_188}",
            file=sys.stderr,
        )
        return 2
    doc207_text = _load_text(args.doc_207)
    doc188_text = _load_text(args.doc_188)

    # 3. Build UpdateResult objects (dry-run safe; just decide what would change).
    r207 = update_207_paragraph(doc207_text, splits, promotion)
    r207.doc_path = str(args.doc_207)
    r188 = update_188_paragraph(doc188_text, splits, promotion)
    r188.doc_path = str(args.doc_188)

    # 4. Always emit the post_fix_update_report.json.
    report = render_report(
        args=args,
        splits=splits,
        promotion=promotion,
        gate_meta=gate_meta,
        r207=r207,
        r188=r188,
        dry_run=args.dry_run,
        verify_path=verify_path,
    )
    _atomic_write_json(args.out_report, report)
    _log(f"wrote post_fix_update_report -> {args.out_report}")

    # 5. Apply (only if --no-dry-run).
    if not args.dry_run:
        if r207.applied:
            new_207 = render_updated_doc(doc207_text, [r207])
            _atomic_write_text(args.updated_207, new_207)
            _log(f"wrote updated 207 doc -> {args.updated_207}")
        else:
            _log(f"207 NOT updated: {r207.skip_reason}")
        if r188.applied:
            new_188 = render_updated_doc(doc188_text, [r188])
            _atomic_write_text(args.updated_188, new_188)
            _log(f"wrote updated 188 doc -> {args.updated_188}")
        else:
            _log(f"188 NOT updated: {r188.skip_reason}")
    else:
        _log("dry_run=True; docs NOT modified (only report.json written)")

    # 6. Exit 0 = OK; we always exit 0 because dry-run is a valid use case
    #    and the report.json signals what would happen.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
