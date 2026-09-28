#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wf-87 Phase 1 — per-metric disk audit for 22 conceptual metrics.

For each of the 22 conceptual metrics enumerated in ``TODO/INDEX.md`` §17.1,
this script reads the on-disk evidence file under ``metrics/by_metric/``
(plus a small set of pointer files under ``TODO/completed/`` for the 5
metrics whose canonical reference is a markdown file rather than a JSON
metric blob) and emits:

  1. ``molmetal/reports/wf87_p1_disk_audit/per_metric_status.json`` — one
     row per conceptual metric with disk-state fields (path, exists,
     mtime, byte size, MEASURED-count vs PROJECTED-count, latest_value,
     latest_status, deferred_reason).
  2. ``molmetal/reports/wf87_p1_disk_audit/synthesis.md`` — a human-readable
     ledger cross-referenced with INDEX §17 + §17.5 (Wf-70 + Wf-72 close-out).

Honest framing (per ``party-twenty-spirit §3.5``):
  - "MEASURED" status is taken only from on-disk evidence.
  - "PROJECTED" status (where the file exists but the value lives only
    in a R22 plan template JSON) is clearly distinguished.
  - "DEFERRED_HOST_GAP" entries (file on disk with ``deferred_reason``)
    are reported as their own bucket — they are NOT counted as MEASURED.
  - No GPU retrain; no metric value is fabricated.

CPU-only; no paper work touched (TODO-84 gate honored).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"
TODO_COMPLETED = REPO_ROOT / "TODO" / "completed"
OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf87_p1_disk_audit"
OUT_JSON = OUT_DIR / "per_metric_status.json"
OUT_MD = OUT_DIR / "synthesis.md"

# ---------------------------------------------------------------------------
# 22 conceptual metrics per INDEX §17.1 baseline
# ---------------------------------------------------------------------------
# Each entry: (metric_id, title, category_hint, primary_source_relpath,
# fallback_source_relpath_or_None, category_bucket).
# - category_bucket ∈ {STRONG_REAL, WEAK_REAL, MOCK, DEFERRED, PROJECTED}.
# - For MOCK/PROJECTED/DEFERRED entries the JSON metric file may be absent or
#   intentionally DEFERRED; we report what is on disk honestly.

METRICS_22: List[Dict[str, str]] = [
    # STRONG_REAL_PAPER_READY (10 per INDEX §17.1)
    {"id": "m6",  "title": "QED",                                "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/qed.json",                          "fallback": "TODO/completed/87l_undefined_metrics_definition.md", "rationale": "Wf-61 + Wf-87 audit close — JSON on disk"},
    {"id": "m9",  "title": "Water solubility logS",              "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/aqueous_solubility_logS.json",     "fallback": "TODO/completed/87l_undefined_metrics_definition.md", "rationale": "Wf-61 NEW MEASURED"},
    {"id": "m11", "title": "n_distinct",                         "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/n_distinct.json",                "fallback": None, "rationale": "RDKit n_distinct=20"},
    {"id": "m13", "title": "RDKit validity",                     "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/rdkit_validity.json",               "fallback": "TODO/completed/87l_undefined_metrics_definition.md", "rationale": "RDKit parse 1.000 (Wf-87 audit close)"},
    {"id": "m14", "title": "CFM decode",                         "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/decode_ratio.json",              "fallback": None, "rationale": "CFM decode=1.000"},
    {"id": "m17", "title": "DrugOOD OOD AUC",                    "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/drugood_ood_auc.json",          "fallback": None, "rationale": "0.891 beats iMoLD 0.7211"},
    {"id": "m21", "title": "NCI-60 9-cell AUC",                  "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/nci60_precious_9cell_auc.json", "fallback": None, "rationale": "0.8007 ± 0.0160 Krasnov-approaching"},
    {"id": "m22", "title": "m18 Protein-Metal binding AUC",      "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/metric18_protein_metal_v2.json", "fallback": None, "rationale": "Wf-32 P2 SHIP — NEW m22"},
    {"id": "m23", "title": "PlatinAI 3-cell resistance AUC",     "category_bucket": "STRONG_REAL", "primary": "metrics/by_metric/platinai_3cell_resistance.json", "fallback": None, "rationale": "Wf-31 + Wf-61 SHIP — NEW m23"},
    {"id": "m24", "title": "cross-dataset macro R²",             "category_bucket": "STRONG_REAL", "primary": "molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json", "fallback": None, "rationale": "Wf-71 P3 Wf-49 round-2 SHIP — NEW m24"},
    # WEAK_REAL_NEEDS_MORE_DATA (11 per INDEX §17.1)
    {"id": "m1",  "title": "Ru AUC",                             "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/metalcytotox_ru_auc.json",       "fallback": None, "rationale": "0.7123 (1-seed); Krasnov gate 0.81 NOT MET"},
    {"id": "m2",  "title": "Pt 4-cell AUC",                      "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/platinai_4cell_auc.json",        "fallback": None, "rationale": "0.553; lift blocked on R20-P2/P3 GPU"},
    {"id": "m4",  "title": "Vina triple success",                "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/vina_triple_success.json",      "fallback": None, "rationale": "0.0; per-pocket Vina Phase-1B pending"},
    {"id": "m5",  "title": "SA mean",                            "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/sa_mean.json",                  "fallback": None, "rationale": "3.099 borderline"},
    {"id": "m7",  "title": "Diversity Tanimoto",                 "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/diversity_tanimoto.json",       "fallback": None, "rationale": "R17 0.1065 canonical; Wf-31 MEASURED +0.9036 but cohort differs (PlatinAI 5K vs generated)"},
    {"id": "m8",  "title": "PoseBusters pass-rate",              "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/pb_pass_rate.json",            "fallback": None, "rationale": "n_distinct=1 collapse; Wf-32 m16/m19/m8 lift-blocked"},
    {"id": "m10", "title": "Vina min score",                     "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/vina_min_score.json",           "fallback": None, "rationale": "-4.158 kcal/mol; per-pocket retrain pending"},
    {"id": "m15", "title": "RMSD",                               "category_bucket": "WEAK_REAL",   "primary": "TODO/completed/85_mock_audit.md",                  "fallback": None, "rationale": "0.428; ground truth missing"},
    {"id": "m16", "title": "Affinity Pearson r",                 "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/pearson_r_pic50.json",          "fallback": None, "rationale": "0.207 ridge; Wf-32 P4 SMOKE -0.040 on PlatinAI 4-cell"},
    {"id": "m19", "title": "Multi-metal LOMO",                   "category_bucket": "WEAK_REAL",   "primary": "metrics/by_metric/metal_lomo_auc.json",           "fallback": None, "rationale": "0.535; Rh 0.368 bottleneck"},
    {"id": "per_pocket_vina", "title": "per-pocket Vina Phase-1B","category_bucket": "WEAK_REAL",  "primary": "TODO/completed/82b_vina_dock_retrain.md",         "fallback": None, "rationale": "Vina stub m3 fix; Phase-1B 5-pocket retrain planned"},
    # MOCK_NEEDS_REMEASURE (4+ per INDEX §17.1)
    {"id": "m3",  "title": "Vina Dock stub",                     "category_bucket": "MOCK",        "primary": "TODO/completed/82b_vina_dock_retrain.md",         "fallback": None, "rationale": "stub -9.6574 CRITICAL; per-host Vina install + receptor fetch needed"},
    {"id": "external_scorers", "title": "4 external scorers (coord_geometry / phototherapy / ptiv / metalloprotein)", "category_bucket": "MOCK", "primary": "TODO/completed/85_mock_audit.md",        "fallback": None, "rationale": "MOCK_DELETED per ROUTING_POLICY"},
    # DEFERRED_HOST_GAP (3 per INDEX §17.1, kept on disk honestly)
    {"id": "diversity_subpocket", "title": "diversity_subpocket", "category_bucket": "DEFERRED",   "primary": "metrics/by_metric/diversity_subpocket.json",      "fallback": None, "rationale": "DESIGN_ONLY (RDKit not installed on host)"},
    {"id": "steric_clash", "title": "steric_clash (PB dock-mode)", "category_bucket": "DEFERRED",  "primary": "metrics/by_metric/steric_clash_pb_dock.json",     "fallback": None, "rationale": "DEFERRED_posebusters_unavailable"},
    {"id": "vina_test_001", "title": "vina_*_test_001 (×4 sub-variants)", "category_bucket": "DEFERRED", "primary": "metrics/by_metric/vina_score_test_001.json", "fallback": None, "rationale": "DEFERRED_receptors_missing"},
]


# ---------------------------------------------------------------------------
# Per-file disk audit primitives
# ---------------------------------------------------------------------------

_STATUS_RE = re.compile(r'"status"\s*:\s*"([A-Za-z_]+)"')
_VALUE_RE = re.compile(r'"value"\s*:\s*([0-9eE+\-.]+|null|true|false)')
_AUC_VALUE_RE = re.compile(r'"(value_overall_mean_auc|test_auc_mean|ood_auc_mean|id_auc_mean|auc_mean|cross_dataset_macro_r2_mean|value_1d|value_2d)"\s*:\s*([0-9eE+\-.]+)')
_DEFERRED_RE = re.compile(r'"deferred_reason"\s*:\s*"([^"]+)"')
_DATE_RE = re.compile(r'"date_utc"\s*:\s*"([^"]+)"')
_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([^"]+)"')
_MEASURED_BUCKET = {"MEASURED", "MEASURED_SMOKE", "MEASURED_FILE", "MEASURED_BELOW_GATE", "MEASURED_DEFERRED_GPU"}
_DEFERRED_STATUS = {"DEFERRED", "DESIGN_ONLY", "DESIGN", "DESIGN_ONLY_DEFERRED_HOST_GAP"}
_DEFERRED_PREFIX = ("DEFERRED_", "DESIGN_ONLY", "DESIGN")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_measured_status(tok: str) -> bool:
    return tok in _MEASURED_BUCKET or tok.startswith("MEASURED")


def _is_deferred_status(tok: str) -> bool:
    return tok in _DEFERRED_STATUS or any(tok.startswith(p) for p in _DEFERRED_PREFIX)


def _scan_status_counts(text: str) -> Tuple[int, int, int]:
    """Return (measured_count, projected_count, deferred_count) from text.

    "MEASURED*" tokens (e.g. MEASURED, MEASURED_SMOKE, MEASURED_BELOW_GATE)
    are counted as measured; PROJECTED as projected; DEFERRED_* / DESIGN_ONLY
    / DESIGN as deferred. We also catch top-level "value" + verdict /
    *_auc_mean single-blob metric files (e.g. metal_lomo_auc,
    metric18_protein_metal_v2, platinai_3cell_resistance) by treating any
    "verdict" / "test_auc_mean" / "value_overall_mean_auc" presence as a
    measured signal — BUT only when there is no DEFERRED token in the file,
    so that ``diversity_subpocket.json`` (status="DESIGN") stays DEFERRED.
    """
    measured = sum(1 for m in _STATUS_RE.finditer(text) if _is_measured_status(m.group(1)))
    projected = sum(1 for m in _STATUS_RE.finditer(text) if m.group(1) == "PROJECTED")
    deferred = sum(1 for m in _STATUS_RE.finditer(text) if _is_deferred_status(m.group(1)))

    # Heuristic: top-level "value" + "timestamp" JSON-blob metrics.
    # Only apply when there is no DEFERRED token (preserves honest framing).
    if measured == 0 and deferred == 0:
        if _AUC_VALUE_RE.search(text) or _VERDICT_RE.search(text):
            measured += 1
    return measured, projected, deferred


def _scan_latest_value(text: str) -> Optional[float]:
    """Return the most recent numeric ``"value":`` literal in the text."""
    matches: List[Tuple[int, float]] = []
    for m in _VALUE_RE.finditer(text):
        tok = m.group(1)
        if tok in ("null", "true", "false"):
            continue
        try:
            matches.append((m.start(), float(tok)))
        except ValueError:
            continue
    if not matches:
        return None
    return matches[-1][1]


def _scan_deferred_reason(text: str) -> Optional[str]:
    m = _DEFERRED_RE.search(text)
    return m.group(1) if m else None


def _scan_latest_date(text: str) -> Optional[str]:
    matches = list(_DATE_RE.finditer(text))
    return matches[-1].group(1) if matches else None


def audit_file(relpath: str) -> Dict[str, Any]:
    """Audit one on-disk file (JSON or MD) for MEASURED/PROJECTED/DEFERRED counts."""
    path = REPO_ROOT / relpath
    exists = path.exists()
    if not exists:
        return {
            "exists": False,
            "size_bytes": 0,
            "mtime_utc": None,
            "measured_count": 0,
            "projected_count": 0,
            "deferred_count": 0,
            "latest_value": None,
            "latest_status": None,
            "latest_date_utc": None,
            "deferred_reason": None,
        }
    stat = path.stat()
    text = _read_text(path)
    measured, projected, deferred = _scan_status_counts(text)
    latest_value = _scan_latest_value(text)
    deferred_reason = _scan_deferred_reason(text)
    latest_date = _scan_latest_date(text)

    # Pick the dominant status for the headline field:
    # if any deferred tokens → DEFERRED;
    # elif any measured tokens → MEASURED;
    # elif any projected tokens → PROJECTED;
    # elif value is null (e.g. vina_*_test_001 — host-gap deferred by
    #     structure, not by status field) → DEFERRED;
    # elif latest value is a numeric with timestamp → MEASURED;
    # else REVIEW_NEEDED.
    if deferred and not measured:
        latest_status = "DEFERRED" if deferred_reason else "DEFERRED_HOST_GAP"
    elif measured:
        latest_status = "MEASURED"
    elif projected:
        latest_status = "PROJECTED"
    elif '"value"' in text and '"null"' in text and re.search(r'"n_docked"\s*:\s*0', text):
        # vina_*_test_001 pattern: top-level value: {mean: null, n_docked: 0}
        latest_status = "DEFERRED_HOST_GAP"
    elif latest_value is not None and re.search(r'"timestamp"\s*:', text):
        # Single-blob JSON: has a top-level value + timestamp but no status field.
        latest_status = "MEASURED"
    else:
        latest_status = "REVIEW_NEEDED"

    mtime = _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc).isoformat()

    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": mtime,
        "measured_count": measured,
        "projected_count": projected,
        "deferred_count": deferred,
        "latest_value": latest_value,
        "latest_status": latest_status,
        "latest_date_utc": latest_date,
        "deferred_reason": deferred_reason,
    }


# ---------------------------------------------------------------------------
# Per-metric row builder
# ---------------------------------------------------------------------------

def build_per_metric_status() -> Dict[str, Any]:
    """Build the 22-row per-metric status table."""
    rows: List[Dict[str, Any]] = []
    for m in METRICS_22:
        primary_rel = m["primary"]
        fallback_rel = m.get("fallback")
        primary_audit = audit_file(primary_rel)
        fallback_audit = audit_file(fallback_rel) if fallback_rel else None

        # Aggregate measured/projected counts across primary + fallback.
        agg_measured = primary_audit["measured_count"]
        agg_projected = primary_audit["projected_count"]
        agg_deferred = primary_audit["deferred_count"]
        if fallback_audit:
            agg_measured += fallback_audit["measured_count"]
            agg_projected += fallback_audit["projected_count"]
            agg_deferred += fallback_audit["deferred_count"]

        # Determine the headline category on disk. If the file is absent or
        # only has DEFERRED/PROJECTED tokens, respect the conceptual bucket
        # so the cross-ref with INDEX §17 still lines up.
        if primary_audit["exists"] or (fallback_audit and fallback_audit["exists"]):
            on_disk_state = (
                primary_audit["latest_status"]
                if primary_audit["exists"]
                else fallback_audit["latest_status"]
            )
        else:
            on_disk_state = "MISSING"

        row = {
            "metric_id": m["id"],
            "title": m["title"],
            "category_bucket": m["category_bucket"],
            "rationale": m["rationale"],
            "primary_source": primary_rel,
            "primary_audit": primary_audit,
            "fallback_source": fallback_rel,
            "fallback_audit": fallback_audit,
            "on_disk_state": on_disk_state,
            "agg_measured_count": agg_measured,
            "agg_projected_count": agg_projected,
            "agg_deferred_count": agg_deferred,
        }
        rows.append(row)

    return {
        "workflow": "Wf-87 Phase 1 — per-metric disk audit",
        "date": _dt.date.today().isoformat(),
        "constraint": "CPU-only; no GPU retrain; paper-gate (TODO-84) honored; party-twenty-spirit §3.5 honest framing",
        "by_metric_dir_scanned": str(BYO_REL := BY_METRIC_DIR.relative_to(REPO_ROOT)),
        "files_in_by_metric_dir": len(list(BY_METRIC_DIR.glob("*.json"))),
        "conceptual_metric_count": len(METRICS_22),
        "per_metric_rows": rows,
        "category_counts": _category_counts(rows),
        "on_disk_state_counts": _on_disk_state_counts(rows),
    }


def _category_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r["category_bucket"]] = out.get(r["category_bucket"], 0) + 1
    return out


def _on_disk_state_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r["on_disk_state"]] = out.get(r["on_disk_state"], 0) + 1
    return out


# ---------------------------------------------------------------------------
# Markdown synthesis writer
# ---------------------------------------------------------------------------

_TABLE_HEADER = (
    "| # | Metric | Bucket | On-disk state | M | P | D | Latest value | mtime_utc | Source |\n"
    "|---:|---|---|---|---:|---:|---:|---:|---|---|\n"
)


def _row_to_md(idx: int, r: Dict[str, Any]) -> str:
    primary = r["primary_source"]
    fallback = r["fallback_source"] or "—"
    latest_value = r["primary_audit"].get("latest_value")
    latest_status = r["primary_audit"].get("latest_status") or "—"
    mtime = r["primary_audit"].get("mtime_utc") or "—"
    latest_str = (
        f"{latest_value:.4f}"
        if isinstance(latest_value, float)
        else (str(latest_value) if latest_value is not None else "—")
    )
    return (
        f"| {idx} | {r['title']} (`{r['metric_id']}`) | {r['category_bucket']} | "
        f"{latest_status} | {r['agg_measured_count']} | {r['agg_projected_count']} | "
        f"{r['agg_deferred_count']} | {latest_str} | `{mtime}` | `{primary}` (+`{fallback}`) |"
    )


def write_synthesis(per_metric: Dict[str, Any]) -> Path:
    rows = per_metric["per_metric_rows"]
    by_md = per_metric["files_in_by_metric_dir"]
    n_conc = per_metric["conceptual_metric_count"]
    cat_counts = per_metric["category_counts"]
    state_counts = per_metric["on_disk_state_counts"]

    md_lines: List[str] = []
    md_lines.append("# Wf-87 Phase 1 — Per-Metric Disk Audit")
    md_lines.append("")
    md_lines.append(f"**Date**: {per_metric['date']}  ")
    md_lines.append(f"**Workflow**: Wf-87 Phase 1 — per-metric disk audit  ")
    md_lines.append(f"**Constraint**: {per_metric['constraint']}  ")
    md_lines.append("")
    md_lines.append("## 1. Aggregate")
    md_lines.append("")
    md_lines.append(f"- `metrics/by_metric/` files scanned: **{by_md}** JSON files.")
    md_lines.append(f"- Conceptual metrics audited (per `TODO/INDEX.md` §17.1): **{n_conc}**.")
    md_lines.append("- Conceptual category distribution:")
    for cat, n in sorted(cat_counts.items()):
        md_lines.append(f"  - **{cat}**: {n}")
    md_lines.append("- On-disk state distribution (post-scan):")
    for st, n in sorted(state_counts.items()):
        md_lines.append(f"  - **{st}**: {n}")
    md_lines.append("")

    md_lines.append("## 2. Per-metric table (22 conceptual metrics)")
    md_lines.append("")
    md_lines.append(_TABLE_HEADER)
    for i, r in enumerate(rows, start=1):
        md_lines.append(_row_to_md(i, r))
    md_lines.append("")

    md_lines.append("## 3. Honest-framing notes")
    md_lines.append("")
    md_lines.append(
        "- All values above are taken directly from on-disk evidence — "
        "no fabricated MEASURED numbers and no GPU retrain executed."
    )
    md_lines.append(
        "- `MEASURED_BELOW_GATE` (e.g. `metalcytotox_ru_auc.json`) is "
        "treated as MEASURED on disk but flagged WEAK_REAL because the "
        "value is below the Krasnov-2026 gate (0.81)."
    )
    md_lines.append(
        "- `diversity_tanimoto.json` carries the R17 baseline (0.1065) as "
        "canonical for *generated* molecules; the Wf-31 entry "
        "(`+0.9036`) is on a *different cohort* (PlatinAI 5K random subset) "
        "and is not an apples-to-apples lift. See file-level `note`."
    )
    md_lines.append(
        "- The three `DEFERRED_HOST_GAP` rows (`diversity_subpocket`, "
        "`steric_clash_pb_dock`, `vina_*_test_001`) are kept on disk with "
        "an explicit `deferred_reason` per Wf-61 §1.4–1.6 close-out."
    )
    md_lines.append(
        "- The m22/m23/m24 STRONG_REAL rows reference `metric18_protein_"
        "metal_v2.json`, `platinai_3cell_resistance.json`, and the Wf-71 "
        "P3 cross-dataset summary JSON respectively."
    )
    md_lines.append("")
    md_lines.append("## 4. Cross-references")
    md_lines.append("")
    md_lines.append("- `TODO/INDEX.md` §17 (22 metrics × 5 categories table)")
    md_lines.append("- `TODO/INDEX.md` §17.5 (R22 plan rows TODO 100-107)")
    md_lines.append("- `TODO/completed/r20_status_ledger_2026_09_26.md` (Wf-70)")
    md_lines.append("- `molmetal/scripts/audit_5cat_extend.py` (Wf-72 Phase 3)")
    md_lines.append("- `molmetal/reports/wf72_5cat_extend/audit_22_to_25.json`")
    md_lines.append("")
    md_lines.append("## 5. Outputs")
    md_lines.append("")
    md_lines.append("- `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json`")
    md_lines.append("- `molmetal/reports/wf87_p1_disk_audit/synthesis.md` (this file)")
    md_lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    return OUT_MD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Wf-87 Phase 1 — per-metric disk audit.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output JSON path")
    parser.add_argument("--out-md", default=str(OUT_MD), help="Output markdown synthesis path")
    args = parser.parse_args()

    per_metric = build_per_metric_status()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(per_metric, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = write_synthesis(per_metric)

    n_conc = per_metric["conceptual_metric_count"]
    n_strong = per_metric["category_counts"].get("STRONG_REAL", 0)
    n_weak = per_metric["category_counts"].get("WEAK_REAL", 0)
    n_mock = per_metric["category_counts"].get("MOCK", 0)
    n_deferred = per_metric["category_counts"].get("DEFERRED", 0)
    n_files = per_metric["files_in_by_metric_dir"]

    print(f"[wf87-p1] scanned {n_files} files in metrics/by_metric/")
    print(f"[wf87-p1] {n_conc} conceptual metrics audited")
    print(f"[wf87-p1] category counts: STRONG={n_strong} WEAK={n_weak} MOCK={n_mock} DEFERRED={n_deferred}")
    print(f"[wf87-p1] JSON → {out_json}")
    print(f"[wf87-p1] MD   → {md_path}")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)