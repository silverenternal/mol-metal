# Wf-289D — Cf-265 dashboard refresh post-Wf-287A m14 anchor

**Date**: 2026-09-29
**Workflow**: Wf-289D (refresh on top of Wf-287E / Wf-280C / Wf-279C / Wf-268C-Dash baseline)
**Trigger**: Wf-287A registered `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` (m14 LEAD anchor = 0.9958, n=26,801, 5-metal Ru/Ir/Os/Rh/Re, real CFM cuda; SUPERSEDES prior 0.9645 PlatinAI pt_subset).

## §1 Pre-refresh audit

**Source**: `molmetal/scripts/paper_dashboard.py` (read at session start, 2026-09-29).

**Claim catalog (`CLAIM_CATALOG`) — Wf-287A m14 anchor pre-existing entries**:
- `claim_m14_cfm_decode_ratio` (lines 217-224): "m14 real CFM FULL MetalCytoToxDB decode_ratio = 0.9958 (n=26,801, Wf-188)" — `BADGE_MEASURED`, json_path `molmetal/reports/wf188_m14_full_metalcytotox/result.json`, ts `2026-09-28`.
- `claim_m14_decode_ratio_synthetic_gap` (lines 225-233): honest-negative "m14 CFM decode_ratio 0.9958 is −0.0042 BELOW synthetic upper bound 1.000" — json_path `TODO/completed/207_unified_pre_flight_checklist_v2.md §9.6.3`.

**Cohort (`_extract_cfm_m14_cohort`, lines 642-658)**: 6-row per-metal decode_ratio table — Ru 0.9948 / Ir 0.9974 / Os 1.000 / Rh 1.000 / Re 1.000 / TOTAL 0.9958 (n=26,801, n_decode_ok=26,687).

**Pre-refresh verdict**: Wf-287A m14 anchor IS already reflected in the dashboard catalog (claim card + cohort table + honest-negative). The m14_anchor.json metadata file (`molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json`) is the artefact-of-record that POINTS TO the dashboard entries; it does not need its own claim card because the underlying Wf-188 result.json path is already canonical.

## §2 Wf-265 paper_auto_update run (dry-run)

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/paper_auto_update.py \
    --collect-json molmetal/reports/wf265_collect/all_results.json \
    --paper-doc TODO/completed/188_paper_4_krasnov_protocol_aligned.md \
    --out-diff /tmp/wf289d_diff_report.md
```

**Output**: `[paper_auto_update] wrote diff_report -> /tmp/wf289d_diff_report.md (11 claims, dry_run=True)`

**Key m14-relevant rows** (from `/tmp/wf289d_diff_report.md`):
- `wf269c_yuelbond_full_decode_ratio` — 1.0000 (delta +0.0000, no change since prior paper doc already cites Wf-188 0.9958).
- `wf269c_no_yuelbond_decode_ratio` — 0.9645 (NEW, in metrics/by_metric/cfm_decode_real.json::regression_disclosure; NOT a paper doc delta — flag preserved via `claim_wf274a_regression_disclosure` honest-negative).
- `wf275c_m14_lead_delta_yuelbond_off` — −1.0000 (NEW precondition; preserved via `claim_wf275c_m14_lead_precondition` MEASURED card + `hn_no_yuelbond_off_deployment` honest-negative).

**Dry-run verdict**: 0 paper doc edits required (m14 anchor 0.9958 already canonical in TODO-188 §4.1 + dashboard cohort + claim card).

## §3 Wf-265 paper_dashboard run

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/paper_dashboard.py \
    --out-html /tmp/dashboard_v289d.html --verbose
```

**Output**: `[Wf-268C-Dash] built dashboard: 33 claims, 17 honest-negatives, 6 cohorts, html_len=76985; [Wf-265-Dashboard] wrote /tmp/dashboard_v289d.html (75.6 KB); build JSON at /tmp/dashboard_v289d.build.json.`

## §4 m14 anchor verification in dashboard

`grep -E "(Wf-188|0\.9958|wf188_m14)" /tmp/dashboard_v289d.html` returned **14 hits**, including:

- `<h3>CFM m14 real decode_ratio (Wf-188, n=26,801)</h3>` (cohort header).
- `<code>molmetal/reports/wf188_m14_full_metalcytotox/result.json + TODO-188 §4.1</code>` (cohort JSON path).
- `<td>0.9958</td>` (TOTAL row in cfm_m14 per-metal table).
- Honest-framing note: "synthetic upper bound = 1.000; real 0.9958 is −0.0042 below the synthetic ceiling — expected structural distance from real metal-coordination chemistry."
- `claim_m14_cfm_decode_ratio` claim card: "m14 real CFM FULL MetalCytoToxDB decode_ratio = 0.9958 (n=26,801, Wf-188)" with `value: 0.9958 (n=26,801)`, `json_path: molmetal/reports/wf188_m14_full_metalcytotox/result.json`, `badge: MEASURED`, `ts: 2026-09-28`.
- Inline source-JSON dump (verbatim dict of the catalog row) for click-to-expand.

`m14_anchor.json` itself is NOT directly referenced in the dashboard — it is a metadata pointer; the underlying Wf-188 result.json is the canonical JSON path. This is the correct architecture: the anchor file ROUTES readers to the right claim card, but does not duplicate it.

## §5 Counts

| Metric | Pre-refresh (Wf-280C baseline) | Post-refresh (Wf-289D) | Δ |
|---|---:|---:|---:|
| `n_claims` | 33 | 33 | 0 |
| `n_honest_negatives` | 17 | 17 | 0 |
| `n_cohort_sections` | 6 | 6 | 0 |
| `m14_anchor_included` (0.9958 in HTML) | YES (14 grep hits) | YES (14 grep hits) | unchanged |

**Verdict**: Wf-287A m14 anchor was already correctly reflected in the dashboard before Wf-289D; no catalog edits required. The dashboard catalog predates Wf-287A by virtue of Wf-279C's `claim_m14_cfm_decode_ratio` + `claim_wf275c_m14_lead_precondition` MEASURED cards and Wf-287A's `m14_anchor.json` is consistent with them. Wf-289D is therefore a **CONSISTENT-RUN** (no schema bump, no claim additions), not a feature-add refresh.

## §6 Honest framing (party-twenty-spirit §3.5)

- The 0.9958 anchor is **MEASURED_PASS** (per Wf-287A §1 verdict).
- The −0.0042 gap vs synthetic 1.000 is **expected** (per TODO-188 §3.5 + Wf-188 synthesis.md "Honest framing").
- The 0.9645 prior carry is **NOT** cited as LEAD; it is flagged as `silent_regression_methodology_change_not_disclosed` via `hn_no_cfm_decode_real_09645_as_lead` (§9.AI) and `claim_wf274a_regression_disclosure` (MEASURED honest-negative §5.2).
- The dashboard does not need a new m14_anchor.json claim card because the Wf-188 result.json claim already carries the same number with the same provenance. Adding a duplicate card would create a `BADGE_MEASURED` count inflation (33 → 34) without adding any new information.

## §7 Outputs

- **diff_report**: `/tmp/wf289d_diff_report.md` (11 claims, dry-run).
- **dashboard_html**: `/tmp/dashboard_v289d.html` (75.6 KB, 33 claims, 17 honest-negatives, 6 cohorts).
- **dashboard_build_json**: `/tmp/dashboard_v289d.build.json` (companion to HTML).
- **refresh_report**: `molmetal/reports/wf289d_dashboard_refresh/refresh_report.md` (this file).

**End of Wf-289D refresh_report. Verdict: CONSISTENT-RUN (m14 anchor pre-existing in dashboard); 0 schema edits; 0 catalog edits; dashboard_path=/tmp/dashboard_v289d.html.**