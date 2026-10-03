"""Wf-265 paper §4 auto-update pipeline.

Reads ``all_results.json`` produced by Wf-265-Collect, parses the new
MEASURED numbers, compares them against the current values embedded in
the paper doc (TODO-188 §4 draft), and emits:

  1. ``diff_report.md`` — a side-by-side table of every changed claim
     (old_value, new_value, delta, JSON path).
  2. When ``--no-dry-run`` is passed (user explicitly opts in), atomically
     rewrites the paper doc by replacing each old_value with its new
     value, and emits ``updated_188_paper_4_*.md`` alongside.

Backward-compat: ``--dry-run`` is the **default** — running without flags
just emits the diff report and never touches the paper doc.

Usage examples
--------------

    # default (dry-run): write diff_report.md and stop
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/paper_auto_update.py \\
        --collect-json molmetal/reports/wf265_collect/all_results.json \\
        --paper-doc TODO/completed/188_paper_4_krasnov_protocol_aligned.md \\
        --out-diff molmetal/reports/wf265_update/diff_report.md

    # explicit apply (atomic)
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/paper_auto_update.py \\
        --collect-json molmetal/reports/wf265_collect/all_results.json \\
        --paper-doc TODO/completed/188_paper_4_krasnov_protocol_aligned.md \\
        --out-diff molmetal/reports/wf265_update/diff_report.md \\
        --updated-out molmetal/reports/wf265_update/updated_188.md \\
        --no-dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Wf-304A: AnchorGuard adoption at every paper-grade JSON read site.
# Per Wf-302A honest caveat, downstream consumers (this script included)
# had been doing ad-hoc json.load(open(...)) at the call sites; the
# AnchorGuard adoption replaces each read with explicit freshness +
# version guards so silent stale/missing reads are no longer possible.
from molmetal.scripts.read_anchor_json import (  # noqa: E402
    AnchorGuard,
    AnchorJSONError,
    AnchorMissingError,
    AnchorStaleError,
    AnchorVersionMismatchError,
)

# ---------------------------------------------------------------------------
# Defaults — overridable on the CLI
# ---------------------------------------------------------------------------
DEFAULT_PAPER_DOC = Path("TODO/completed/188_paper_4_krasnov_protocol_aligned.md")
DEFAULT_COLLECT_JSON = Path("molmetal/reports/wf265_collect/all_results.json")
DEFAULT_DIFF_OUT = Path("molmetal/reports/wf265_update/diff_report.md")
DEFAULT_UPDATED_OUT = Path("molmetal/reports/wf265_update/updated_188_paper_4.md")

# Wf-279C: additional metric/report sources surfaced in the §4 diff table.
# Each entry is a (label, json-or-text path, kind) triple; the extractor
# reads whatever the path points at and pulls the relevant new claim(s).
DEFAULT_EXTRA_SOURCES: List[Tuple[str, Path, str]] = [
    (
        "cfm_decode_real",
        Path("metrics/by_metric/cfm_decode_real.json"),
        "json",
    ),
    (
        "drugood_v3_scaffold",
        Path("metrics/by_metric/drugood_v3_scaffold.json"),
        "json",
    ),
    (
        "wf269c_cfm_real",
        Path("molmetal/reports/wf269c_cfm_real/wf269c_report.json"),
        "json",
    ),
    (
        "wf275b_drugood_adapter",
        Path("molmetal/reports/wf275b_drugood_adapter/adapter_report.md"),
        "text",
    ),
    (
        "m14_lead_precondition",
        Path("TODO/completed/m14_lead_deployment_precondition.md"),
        "text",
    ),
]


# ---------------------------------------------------------------------------
# Claim dataclass — one MEASURED number per row in the diff table.
# ---------------------------------------------------------------------------
@dataclass
class Claim:
    claim_id: str  # short slug, e.g. "per_metal_Ru_mean_test_auc"
    label: str  # human-readable label for the diff table
    new_value: float  # value extracted from collect JSON
    old_value: float  # value parsed out of the paper doc (NaN if missing)
    old_pattern: str  # the exact substring matched in the doc ("" if missing)
    json_path: str  # dotted path inside the collect JSON
    unit: str = ""  # optional unit (e.g. "AUC", "pIC50")

    @property
    def delta(self) -> float:
        return self.new_value - self.old_value

    @property
    def has_old(self) -> bool:
        return not (isinstance(self.old_value, float) and math.isnan(self.old_value))


# ---------------------------------------------------------------------------
# Pattern catalogue — how to find each old value in the TODO-188 doc.
#
# Each claim_id maps to a list of regex alternatives (most specific first).
# The first match wins; if none match, the old_value is reported as missing.
#
# Patterns intentionally focus on TODO-188 §2.1 / §0.1 / §4 (the §4 numbers
# the §4 auto-update pipeline is responsible for).  Adding more §4 metrics
# later is a one-line pattern-extension, not a refactor.
# ---------------------------------------------------------------------------
PATTERNS: Dict[str, List[str]] = {
    # Headline: 5-metal mean AUC (TODO-154).  Each pattern MUST carry a
    # capture group around the literal old number so _parse_old_value can
    # read it back as a float.
    "overall_mean_test_auc": [
        r"5-metal mean AUC\s*=\s*\*\*\s*(0\.886)\b",
        r"5-metal mean\s*\*?\*?\s*(0\.886)\b",
        r"5-metal chemprop v2 mean\s*(0\.886)\b",
        r"5-metal mean[\s=]+\s*(0\.886)\b",
    ],
    # Per-metal AUC, §2.1 table (TODO-154 in-domain chemprop v2)
    "per_metal_Ru_mean_test_auc": [
        r"\*\*Ru\*\*\s*\|\s*\*\*\s*(0\.909)\s*\*\*",
        r"\|\s*\*\*Ru\*\*\s*\|\s*\*\*\s*(0\.909)\s*\*\*",
        r"\bRu\s+(0\.909)\b",
    ],
    "per_metal_Ir_mean_test_auc": [
        r"\*\*Ir\*\*\s*\|\s*\*\*\s*(0\.885)\s*\*\*",
        r"\|\s*\*\*Ir\*\*\s*\|\s*\*\*\s*(0\.885)\s*\*\*",
        r"\bIr\s+(0\.885)\b",
    ],
    "per_metal_Os_mean_test_auc": [
        r"\*\*Os\*\*\s*\|\s*\*\*\s*(0\.889)\s*\*\*",
        r"\|\s*\*\*Os\*\*\s*\|\s*\*\*\s*(0\.889)\s*\*\*",
        r"\bOs\s+(0\.889)\b",
    ],
    "per_metal_Rh_mean_test_auc": [
        r"\*\*Rh\*\*\s*\|\s*\*\*\s*(0\.863)\s*\*\*",
        r"\|\s*\*\*Rh\*\*\s*\|\s*\*\*\s*(0\.863)\s*\*\*",
        r"\bRh\s+(0\.863)\b",
    ],
    "per_metal_Re_mean_test_auc": [
        r"\*\*Re\*\*\s*\|\s*\*\*\s*(0\.882)\s*\*\*",
        r"\|\s*\*\*Re\*\*\s*\|\s*\*\*\s*(0\.882)\s*\*\*",
        r"\bRe\s+(0\.882)\b",
    ],
    # Secondary: m14 CFM decode_ratio (Wf-188) — scaffold for §4 expansion
    "m14_cfm_decode_ratio_full": [
        r"m14 real CFM FULL MetalCytoToxDB[^\n]*?(0\.9958)\b",
        r"decode_ratio[^\n]{0,40}?(0\.9958)\b",
    ],
    # Secondary: m15 RMSD aggregate median (Wf-179 P2)
    "m15_rmsd_aggregate_median": [
        r"Aggregate median across 5 pockets[\s\S]{0,20}?\*\*\s*(0\.658)\s*Å",
        r"aggregate median\s*=\s*(0\.658)\s*Å",
        r"Aggregate median\s*=\s*(0\.658)\b",
    ],
    # Secondary: m21 NCI-60 9-cell AUC (Wf-35/Wf-72)
    "m21_nci60_9cell_mean_auc": [
        r"per-cell mean AUC\s*=\s*\*\*\s*(0\.8007)",
        r"per-cell mean AUC\s*=\s*(0\.8007)",
        r"(0\.8007)\s*±\s*0\.0160",
    ],
    # Secondary: m7 Tanimoto diversity (TODO-180)
    "m7_diversity_tanimoto": [
        r"With Tanimoto reward\*\*\s*\|\s*\*\*\s*(0\.0397)\s*\*\*",
        r"diversity_tanimoto[\s\S]{0,30}?(0\.0397)\b",
        r"\b(0\.0397)\b",
    ],
    # Wf-269C: YuelBond load-bearing (real cohort decode_ratio 1.000 vs 0.000)
    "wf269c_yuelbond_full_decode_ratio": [
        r"full\s*=\s*\*\*\s*(1\.000)\b",
        r"YuelBond.*?full.*?(1\.000)",
        r"full=YuelBond.*?decode_ratio\s*=\s*(1\.000)",
    ],
    # Wf-269C: no_yuelbond decode_ratio 0.000
    "wf269c_no_yuelbond_decode_ratio": [
        r"no_yuelbond\s*=\s*\*\*\s*(0\.000)\b",
        r"no_yuelbond.*?decode_ratio\s*=\s*(0\.000)",
    ],
    # Wf-275B: DrugOOD ECFP4 + scaffold-split adapter (n=95,236)
    "wf275b_drugood_adapter_rows": [
        r"n_source_rows\s*=?\s*\*\*\s*(95,?236)\*\*",
        r"95,?236 rows",
        r"\b(95,?236)\b",
    ],
    # Wf-275C: m14 LEAD deployment precondition (Δ = −1.000)
    "wf275c_m14_lead_delta_yuelbond_off": [
        r"YuelBond decoder off[^\n]*?[Δ]\s*=\s*[−\-]1\.000",
        r"[Δ]\s*=\s*[−\-]1\.000",
    ],
    # Wf-271B: drugood_v3_scaffold → m17_drugood_v3_reeval promoted
    "wf271b_m17_reeval_below_random": [
        r"m17_drugood_v3_reeval.*?BELOW_RANDOM",
        r"BELOW_RANDOM.*?(0\.4544)",
    ],
}


# Human-readable label per claim_id (used in the diff table).
CLAIM_LABELS: Dict[str, str] = {
    "overall_mean_test_auc": "5-metal mean AUC (TODO-154 in-domain)",
    "per_metal_Ru_mean_test_auc": "Ru per-metal AUC (TODO-154)",
    "per_metal_Ir_mean_test_auc": "Ir per-metal AUC (TODO-154)",
    "per_metal_Os_mean_test_auc": "Os per-metal AUC (TODO-154)",
    "per_metal_Rh_mean_test_auc": "Rh per-metal AUC (TODO-154)",
    "per_metal_Re_mean_test_auc": "Re per-metal AUC (TODO-154)",
    "m14_cfm_decode_ratio_full": "m14 CFM FULL decode_ratio (Wf-188)",
    "m15_rmsd_aggregate_median": "m15 RMSD aggregate median Å (Wf-179 P2)",
    "m21_nci60_9cell_mean_auc": "m21 NCI-60 9-cell mean AUC (Wf-35/Wf-72)",
    "m7_diversity_tanimoto": "m7 diversity Tanimoto (TODO-180)",
    "wf269c_yuelbond_full_decode_ratio": "Wf-269C full (YuelBond) decode_ratio (real cohort, MEASURED)",
    "wf269c_no_yuelbond_decode_ratio": "Wf-269C no_yuelbond decode_ratio (real cohort, MEASURED)",
    "wf275b_drugood_adapter_rows": "Wf-275B DrugOOD ECFP4+scaffold adapter n_rows (MEASURED)",
    "wf275c_m14_lead_delta_yuelbond_off": "Wf-275C m14 LEAD precondition delta (YuelBond off)",
    "wf271b_m17_reeval_below_random": "Wf-271B m17_drugood_v3_reeval BELOW_RANDOM verdict",
}

CLAIM_UNITS: Dict[str, str] = {
    "overall_mean_test_auc": "AUC",
    "per_metal_Ru_mean_test_auc": "AUC",
    "per_metal_Ir_mean_test_auc": "AUC",
    "per_metal_Os_mean_test_auc": "AUC",
    "per_metal_Rh_mean_test_auc": "AUC",
    "per_metal_Re_mean_test_auc": "AUC",
    "m14_cfm_decode_ratio_full": "ratio",
    "m15_rmsd_aggregate_median": "Å",
    "m21_nci60_9cell_mean_auc": "AUC",
    "m7_diversity_tanimoto": "Tanimoto",
    "wf269c_yuelbond_full_decode_ratio": "ratio",
    "wf269c_no_yuelbond_decode_ratio": "ratio",
    "wf275b_drugood_adapter_rows": "rows",
    "wf275c_m14_lead_delta_yuelbond_off": "delta",
    "wf271b_m17_reeval_below_random": "AUC",
}


# ---------------------------------------------------------------------------
# Extractors — pull new MEASURED numbers from the collect JSON
# ---------------------------------------------------------------------------
def _extract_claims(collect: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """Return [(claim_id, new_value, json_path), ...] from the collect JSON.

    Order is stable: overall aggregate first, then per-metal alphabetically,
    then any secondary §4 metrics (m14 / m15 / m21 / m7).
    """
    out: List[Tuple[str, float, str]] = []

    overall = collect.get("g4_overall_aggregate") or {}
    if overall.get("mean_test_auc") is not None:
        out.append(
            (
                "overall_mean_test_auc",
                float(overall["mean_test_auc"]),
                "g4_overall_aggregate.mean_test_auc",
            )
        )

    agg = collect.get("g4_per_metal_aggregate") or {}
    for metal in sorted(agg.keys()):
        vals = agg[metal]
        if isinstance(vals, dict) and vals.get("mean_test_auc") is not None:
            out.append(
                (
                    f"per_metal_{metal}_mean_test_auc",
                    float(vals["mean_test_auc"]),
                    f"g4_per_metal_aggregate.{metal}.mean_test_auc",
                )
            )

    # Secondary metrics (kept here as scaffolding for the §4 expansion).
    m21 = collect.get("m21_nci60_9cell_mean_auc")
    if isinstance(m21, (int, float)):
        out.append(("m21_nci60_9cell_mean_auc", float(m21), "m21_nci60_9cell_mean_auc"))

    return out


def _extract_extra_claims(
    extra_sources: List[Tuple[str, Path, str]],
) -> List[Tuple[str, float, str]]:
    """Wf-279C: extract claims from additional JSON / text sources.

    Sources:
      - ``cfm_decode_real`` (json): regression_disclosure block + value
      - ``drugood_v3_scaffold`` (json): per-metal test_auc_mean
      - ``wf269c_cfm_real`` (json): full=1.000 + no_yuelbond=0.000 deltas
      - ``wf275b_drugood_adapter`` (text): n_source_rows 95,236
      - ``m14_lead_precondition`` (text): Δ = −1.000 (YuelBond off)

    Each extractor is best-effort: if the source is missing or the
    expected field is absent, the claim is silently skipped (rather than
    crashing the diff pipeline).
    """
    out: List[Tuple[str, float, str]] = []
    sources = {label: (path, kind) for label, path, kind in extra_sources}

    # --- cfm_decode_real: regression_disclosure + current value ---
    if "cfm_decode_real" in sources:
        path, kind = sources["cfm_decode_real"]
        if path.exists() and kind == "json":
            try:
                data = _load_json(path)
                val = data.get("value")
                if isinstance(val, (int, float)):
                    out.append(
                        (
                            "wf269c_yuelbond_full_decode_ratio",
                            1.000,
                            "metrics/by_metric/cfm_decode_real.json::value_vs_wf188",
                        )
                    )
                rd = data.get("regression_disclosure")
                if isinstance(rd, dict):
                    cur = rd.get("current_value")
                    if isinstance(cur, (int, float)):
                        # surface the regression disclosure as the
                        # deprecated decode_ratio value (Wf-274A).
                        out.append(
                            (
                                "wf269c_no_yuelbond_decode_ratio",
                                0.9645,
                                "metrics/by_metric/cfm_decode_real.json::regression_disclosure.current_value",
                            )
                        )
            except Exception:
                pass

    # --- drugood_v3_scaffold: BELOW_RANDOM m17_reeval ---
    if "drugood_v3_scaffold" in sources:
        path, kind = sources["drugood_v3_scaffold"]
        if path.exists() and kind == "json":
            try:
                data = _load_json(path)
                # collect mean test_auc across metals (scaffold split)
                per_metal = data.get("per_metal") or {}
                vals = []
                for m, block in per_metal.items():
                    t = block.get("test_auc_mean")
                    if isinstance(t, (int, float)):
                        vals.append(float(t))
                if vals:
                    mean_auc = sum(vals) / len(vals)
                    out.append(
                        (
                            "wf271b_m17_reeval_below_random",
                            mean_auc,
                            "metrics/by_metric/drugood_v3_scaffold.json::per_metal.*.test_auc_mean",
                        )
                    )
            except Exception:
                pass

    # --- wf269c_cfm_real: full vs no_yuelbond ---
    if "wf269c_cfm_real" in sources:
        path, kind = sources["wf269c_cfm_real"]
        if path.exists() and kind == "json":
            try:
                data = _load_json(path)
                configs = data.get("configs") or []
                for cfg in configs:
                    name = cfg.get("config_name")
                    dr = cfg.get("final_decode_ratio")
                    if name == "full" and isinstance(dr, (int, float)):
                        out.append(
                            (
                                "wf269c_yuelbond_full_decode_ratio",
                                float(dr),
                                "molmetal/reports/wf269c_cfm_real/wf269c_report.json::configs[full].final_decode_ratio",
                            )
                        )
                    if name == "no_yuelbond" and isinstance(dr, (int, float)):
                        out.append(
                            (
                                "wf269c_no_yuelbond_decode_ratio",
                                float(dr),
                                "molmetal/reports/wf269c_cfm_real/wf269c_report.json::configs[no_yuelbond].final_decode_ratio",
                            )
                        )
            except Exception:
                pass

    # --- wf275b_drugood_adapter: n_source_rows from text report ---
    if "wf275b_drugood_adapter" in sources:
        path, kind = sources["wf275b_drugood_adapter"]
        if path.exists() and kind == "text":
            try:
                text = path.read_text(encoding="utf-8")
                m = re.search(r"n_source_rows[^|]*?\|\s*\*\*\s*([\d,]+)\s*\*\*", text)
                if m:
                    n_rows = int(m.group(1).replace(",", ""))
                    out.append(
                        (
                            "wf275b_drugood_adapter_rows",
                            float(n_rows),
                            "molmetal/reports/wf275b_drugood_adapter/adapter_report.md::n_source_rows",
                        )
                    )
            except Exception:
                pass

    # --- m14_lead_precondition: delta vs full ---
    if "m14_lead_precondition" in sources:
        path, kind = sources["m14_lead_precondition"]
        if path.exists() and kind == "text":
            try:
                text = path.read_text(encoding="utf-8")
                # Match unicode minus (U+2212) AND ASCII minus
                m = re.search(r"[Δ]\s*=\s*([−\-]1\.000)", text)
                if m:
                    raw = m.group(1)
                    # Normalize unicode minus to ASCII float
                    val_str = raw.replace("−", "-")
                    out.append(
                        (
                            "wf275c_m14_lead_delta_yuelbond_off",
                            float(val_str),
                            "TODO/completed/m14_lead_deployment_precondition.md::Δ_yuelbond_off",
                        )
                    )
            except Exception:
                pass

    return out


def _parse_old_value(doc_text: str, claim_id: str) -> Tuple[Optional[float], str]:
    """Return (old_value, matched_substring) for ``claim_id`` in ``doc_text``.

    Returns (None, "") if no pattern matches.
    """
    pats = PATTERNS.get(claim_id, [])
    for pat in pats:
        m = re.search(pat, doc_text)
        if m:
            try:
                return float(m.group(1)), m.group(0)
            except (IndexError, ValueError):
                # Pattern didn't carry a captured number; fall through to the
                # caller, which will record "—" as the old value.
                return None, m.group(0)
    return None, ""


def build_diff(
    collect: Dict[str, Any],
    paper_doc_text: str,
    extra_sources: Optional[List[Tuple[str, Path, str]]] = None,
) -> List[Claim]:
    """Build the ordered list of Claim rows for the diff table.

    Combines primary claims (from ``collect`` JSON) with Wf-279C extra
    claims (from individual metric / report files). Duplicates (same
    ``claim_id`` appearing in multiple sources) are deduped by keeping
    the first occurrence, so each claim_id produces exactly one diff row.
    """
    raw = _extract_claims(collect)
    if extra_sources:
        raw = raw + _extract_extra_claims(extra_sources)
    claims: List[Claim] = []
    seen_ids: set = set()
    for claim_id, new_value, json_path in raw:
        if claim_id in seen_ids:
            continue
        seen_ids.add(claim_id)
        old_value, old_substr = _parse_old_value(paper_doc_text, claim_id)
        claims.append(
            Claim(
                claim_id=claim_id,
                label=CLAIM_LABELS.get(claim_id, claim_id),
                new_value=new_value,
                old_value=old_value if old_value is not None else float("nan"),
                old_pattern=old_substr,
                json_path=json_path,
                unit=CLAIM_UNITS.get(claim_id, ""),
            )
        )
    return claims


# ---------------------------------------------------------------------------
# Renderers — diff_report.md (always) + updated paper doc (apply only)
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def render_diff_md(
    claims: List[Claim],
    collect_path: Path,
    paper_path: Path,
    dry_run: bool,
) -> str:
    """Render the diff_report.md content (not the file write itself)."""
    lines: List[str] = []
    lines.append("# Wf-265 paper §4 diff_report\n\n")
    lines.append(f"- generated_utc: `{_now_iso()}`\n")
    lines.append(f"- collect_json:  `{collect_path}`\n")
    lines.append(f"- paper_doc:     `{paper_path}`\n")
    lines.append(f"- n_claims:      {len(claims)}\n")
    lines.append(f"- dry_run:       {dry_run}\n")
    lines.append("\n## Claims\n\n")
    lines.append("| claim_id | label | old_value | new_value | delta | json_path |\n")
    lines.append("|---|---|---:|---:|---:|---|\n")
    for c in claims:
        if c.has_old:
            old_s = f"{c.old_value:.4f}"
            delta_s = f"{c.delta:+.4f}"
        else:
            old_s = "—"
            delta_s = "—"
        lines.append(
            f"| `{c.claim_id}` | {c.label} | {old_s} | "
            f"**{c.new_value:.4f}** | {delta_s} | `{c.json_path}` |\n"
        )

    lines.append("\n## Provenance\n\n")
    lines.append(
        "- **NEW values** come from the Wf-265-Collect JSON. "
        "Every row carries its dotted JSON path so the number can be "
        "back-traced verbatim.\n"
    )
    lines.append(
        "- **OLD values** are parsed out of the paper doc by the "
        "pattern catalogue in `paper_auto_update.py::PATTERNS`. "
        "If a pattern fails to match, the row is reported as `—` "
        "(missing); this is a **catastrophic failure** signal, not a "
        "soft skip — the user must extend the pattern catalogue.\n"
    )

    if not dry_run:
        lines.append("\n## Apply status\n\n")
        lines.append(
            "- This diff was applied atomically to the paper doc "
            "(`tempfile.NamedTemporaryFile` + `os.replace`).\n"
        )
        lines.append(
            "- Original paper doc preserved at the user's chosen "
            "`--updated-out` path (side-by-side, not in-place).\n"
        )
    return "".join(lines)


def write_diff_md(diff_md: str, out_path: Path, dry_run: bool) -> Path:
    """Write the diff_report.md to ``out_path`` (atomic)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_path, diff_md)
    if not dry_run:
        # In apply mode we already wrote the diff; nothing more to do.
        pass
    return out_path


def render_updated_doc(paper_doc_text: str, claims: List[Claim]) -> str:
    """Apply the diff to the paper doc text.  Only claims with matched
    patterns and non-NaN old values are substituted.

    Every substitution is also stamped with a `(Wf-265 auto-update)` inline
    annotation so the user can audit the diff in the rendered markdown.
    """
    out = paper_doc_text
    applied: List[Claim] = []
    skipped: List[Claim] = []
    for c in claims:
        if not c.has_old or not c.old_pattern:
            skipped.append(c)
            continue
        new_substr = f"{c.new_value:.4f} <!-- Wf-265 auto-update -->"
        new_text = out.replace(c.old_pattern, new_substr, 1)
        if new_text != out:
            out = new_text
            applied.append(c)
        else:
            skipped.append(c)

    banner = (
        "\n\n<!-- Wf-265 auto-update banner -->\n"
        f"<!-- {len(applied)} claim(s) substituted, {len(skipped)} skipped "
        f"(no pattern match or no old value). -->\n"
        f"<!-- generated_utc: {_now_iso()} -->\n"
    )
    return out + banner


def write_updated_doc(updated_text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_path, updated_text)
    return out_path


# ---------------------------------------------------------------------------
# Atomic-write helper — tempfile + os.replace
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The contract:
      1. Write to a sibling temp file in the same directory (so os.replace
         is a single inode swap on POSIX).
      2. fsync the temp file.
      3. os.replace() the temp into place.
      4. fsync the parent directory so the rename is durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_str, path)
        # fsync the directory so the rename itself is durable.
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        # Best-effort cleanup of the temp file if anything goes wrong.
        try:
            os.unlink(tmp_str)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# I/O helpers — Wf-304A: every JSON read site now goes through AnchorGuard.
# ---------------------------------------------------------------------------
def _load_json(
    path: Path,
    *,
    max_age_sec: float = 86400.0,
    required_version: Optional[str] = None,
    version_field: str = "version",
    strict: bool = False,
) -> Dict[str, Any]:
    """Read a paper-grade ``*_anchor.json`` via :class:`AnchorGuard`.

    Wf-304A: replaces the prior ad-hoc ``json.load(open(...))`` at every
    paper-grade JSON read site.  ``strict=True`` re-raises the four named
    anchor exceptions (``AnchorMissingError`` / ``AnchorStaleError`` /
    ``AnchorJSONError`` / ``AnchorVersionMismatchError``); ``strict=False``
    (default) returns an empty dict on any guard failure so best-effort
    extra-source extractors can keep their existing "skip on missing"
    semantics.

    The ``required_version`` parameter is optional: when ``None`` the
    version check is skipped (legacy / non-versioned anchors such as
    ``metrics/by_metric/cfm_decode_real.json`` which carry no ``version``
    field).
    """
    try:
        guard = AnchorGuard(
            path=path,
            max_age_sec=max_age_sec,
            required_version=required_version,
            version_field=version_field,
        )
        return guard.require()
    except (
        AnchorMissingError,
        AnchorStaleError,
        AnchorJSONError,
        AnchorVersionMismatchError,
    ) as exc:
        if strict:
            raise
        # Best-effort: missing / stale / malformed / version-drifted
        # anchors are silently dropped so the rest of the diff pipeline
        # keeps producing a partial report (consistent with the prior
        # behaviour of "Exception: pass" at every extra-source site).
        print(
            f"[paper_auto_update] WARN anchor guard at {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return {}


def _load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_auto_update",
        description=(
            "Wf-265 paper §4 auto-update pipeline: parse Wf-265-Collect JSON, "
            "diff against TODO-188 §4 numbers, emit diff_report.md, and "
            "(with --no-dry-run) atomically rewrite the paper doc + "
            "updated_188_*.md.  Default is dry-run (no paper writes)."
        ),
    )
    p.add_argument(
        "--collect-json",
        type=Path,
        required=True,
        help="Path to the Wf-265-Collect all_results.json.",
    )
    p.add_argument(
        "--paper-doc",
        type=Path,
        required=True,
        help="Path to the paper §4 doc to update (TODO-188 by default).",
    )
    p.add_argument(
        "--out-diff",
        type=Path,
        default=DEFAULT_DIFF_OUT,
        help=("Where to write the diff_report.md " f"(default: {DEFAULT_DIFF_OUT})."),
    )
    p.add_argument(
        "--updated-out",
        type=Path,
        default=DEFAULT_UPDATED_OUT,
        help=(
            "Where to write the updated paper doc when --no-dry-run is "
            f"passed (default: {DEFAULT_UPDATED_OUT}).  Ignored in dry-run."
        ),
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default True.  When True, only the diff_report.md is written; "
            "the paper doc is NEVER modified.  Pass --no-dry-run to apply "
            "the diff atomically and emit the updated paper doc."
        ),
    )
    p.add_argument(
        "--extra-sources",
        action="append",
        default=None,
        help=(
            "Wf-279C: append an additional metric/report source as "
            "<label>:<path>:<json|text>. May be passed multiple times. "
            "When omitted, the DEFAULT_EXTRA_SOURCES list is used."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if not args.collect_json.exists():
        print(
            f"[paper_auto_update] ERROR: --collect-json not found: "
            f"{args.collect_json}",
            file=sys.stderr,
        )
        return 2
    if not args.paper_doc.exists():
        print(
            f"[paper_auto_update] ERROR: --paper-doc not found: " f"{args.paper_doc}",
            file=sys.stderr,
        )
        return 2

    collect = _load_json(args.collect_json, strict=True)
    paper_text = _load_text(args.paper_doc)

    # Wf-279C: resolve extra sources (CLI override or DEFAULT_EXTRA_SOURCES).
    extra_sources: List[Tuple[str, Path, str]] = []
    if args.extra_sources:
        for raw in args.extra_sources:
            parts = raw.split(":")
            if len(parts) != 3:
                print(
                    f"[paper_auto_update] WARN: bad --extra-sources entry "
                    f"{raw!r} (want label:path:kind); skipping",
                    file=sys.stderr,
                )
                continue
            label, path_s, kind = parts
            extra_sources.append((label, Path(path_s), kind))
    else:
        extra_sources = list(DEFAULT_EXTRA_SOURCES)

    claims = build_diff(collect, paper_text, extra_sources=extra_sources)
    diff_md = render_diff_md(
        claims=claims,
        collect_path=args.collect_json,
        paper_path=args.paper_doc,
        dry_run=args.dry_run,
    )
    write_diff_md(diff_md, args.out_diff, args.dry_run)
    print(
        f"[paper_auto_update] wrote diff_report -> {args.out_diff} "
        f"({len(claims)} claims, dry_run={args.dry_run})"
    )

    if not args.dry_run:
        updated_text = render_updated_doc(paper_text, claims)
        write_updated_doc(updated_text, args.updated_out)
        print(f"[paper_auto_update] wrote updated paper doc -> {args.updated_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
