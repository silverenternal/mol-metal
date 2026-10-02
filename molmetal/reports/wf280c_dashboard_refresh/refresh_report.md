# Wf-280C: Dashboard Refresh Report

- **generated_utc:** 2026-09-29
- **wf_id:** Wf-280C (refresh on top of Wf-279C)
- **schema_version:** 280.v1
- **dashboard_path:** `/home/hugo/codes/mol-metal/molmetal/reports/wf280c_dashboard_refresh/dashboard.html`
- **build_json:** `/home/hugo/codes/mol-metal/molmetal/reports/wf280c_dashboard_refresh/dashboard.build.json`
- **diff_report:** `/home/hugo/codes/mol-metal/molmetal/reports/wf265_update/diff_report.md` (n_claims=11, dry_run=True)

## Counts

- **claims_total:** 32 (pre=32 / post=32; no new claim cards added by Wf-280C — all 5 task findings were already baked into the Wf-279C baseline)
- **honest_negatives_total:** 17
- **MEASURED:** 22
- **PROJECTED:** 4
- **honest-negative (claim cards):** 6
- **honest-negative (catalog entries):** 17

## Findings verified present in the dashboard

All 5 task findings are visible in the dashboard HTML (`grep` confirmed):

| Wf ID | finding | dashboard claim card ID | badge |
|---|---|---|---|
| Wf-274A | cfm_decode_real.json regression_disclosure (silent 0.9980 → 0.9645) | `claim_wf274a_regression_disclosure` | honest-negative |
| Wf-275B | DrugOOD ECFP4 + Bemis-Murcko scaffold-split adapter (n=95,236; 0 leak) | `claim_wf275b_drugood_adapter` | MEASURED |
| Wf-275C | m14 LEAD deployment precondition (YuelBond-on mandatory) | `claim_wf275c_m14_lead_precondition` | MEASURED |
| Wf-269C | YuelBond decoder LOAD-BEARING (Δ=−1.000; full=1.000 vs no_yuelbond=0.000) | `claim_wf269c_yuelbond_load_bearing` (section §4.5.4) | MEASURED |
| Wf-271B | drugood_v3_scaffold → m17_drugood_v3_reeval BELOW_RANDOM (0.4544) | `claim_wf271b_m17_drugood_v3_reeval` | honest-negative |

Plus two related Wf-279C honest-negative catalog entries:

| ID | item |
|---|---|
| `hn_no_cfm_decode_real_09645_as_lead` | DO NOT cite cfm_decode_real.json value=0.9645 as the m14 LEAD (cite Wf-188 = 0.9958 instead) |
| `hn_no_yuelbond_off_deployment` | DO NOT deploy m14 LEAD CFM with --no-yuelbond flag (Δ=−1.000; silent invalidation) |

## paper_auto_update.py ingest list

`DEFAULT_EXTRA_SOURCES` (5 entries, all loaded successfully by the auto-update):

1. `cfm_decode_real` → `metrics/by_metric/cfm_decode_real.json` (json)
2. `drugood_v3_scaffold` → `metrics/by_metric/drugood_v3_scaffold.json` (json)
3. `wf269c_cfm_real` → `molmetal/reports/wf269c_cfm_real/wf269c_report.json` (json)
4. `wf275b_drugood_adapter` → `molmetal/reports/wf275b_drugood_adapter/adapter_report.md` (text)
5. `m14_lead_precondition` → `TODO/completed/m14_lead_deployment_precondition.md` (text)

All 5 produced entries in `diff_report.md`. No new sources required for Wf-280C.

## Schema version bump

- 279.v1 → 280.v1
- `paper_dashboard.py` updated:
  - `SCHEMA_VERSION = "280.v1"` (line 76)
  - `wf_id = "Wf-280C (refresh on top of Wf-279C)"` (update_report)
  - `<meta name="generator" content="Wf-280C paper_dashboard.py schema=280.v1" />`

## Files modified / produced

- modified: `molmetal/scripts/paper_dashboard.py` (3 single-line edits: SCHEMA_VERSION, update_report wf_id, generator meta)
- produced: `molmetal/reports/wf280c_dashboard_refresh/dashboard.html` (73.6 KB)
- produced: `molmetal/reports/wf280c_dashboard_refresh/dashboard.build.json`
- refreshed: `molmetal/reports/wf265_update/diff_report.md` (n_claims=11, dry_run=True)
- produced: `molmetal/reports/wf280c_dashboard_refresh/refresh_report.md` (this file)

## Verification

- `paper_auto_update.py` exit code: 0
- `paper_dashboard.py` exit code: 0
- Dashboard HTML size: 73.6 KB (74,914 chars)
- All 5 Wf-IDs (Wf-274A, Wf-275B, Wf-275C, Wf-269C, Wf-271B) grep-confirmed in dashboard.html
- Schema 280.v1 + Wf-280C wf_id grep-confirmed in dashboard.html and dashboard.build.json