# Wf-286D — Cf-265 Dashboard Refresh (post-Wf-284C verification)

**Date**: 2026-09-29
**Workflow**: Wf-286D (Cf-265 dashboard verification post-Wf-284C additions)
**Source artifacts verified**:
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AK (Wf-282 G5 Ru seed=0 MEASURED)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.1 (1/15 MEASURED, 14/15 PENDING)
- `molmetal/scripts/wave_budget_tracker.py` lines 261-272 (10 `WF_282_G5_*` constants)
- `molmetal/scripts/paper_dashboard.py` (catalog has `claim_wf282_g5_ru_seed0_test_auc`)

---

## 1. Pre-check (catalog scan)

| Item | Pre-Wf-284C | Post-Wf-284C | Wf-286D verification |
|---|---|---|---|
| n_claims | 32 | **33** | 33 ✅ |
| n_honest_negatives | 12 | 17 | 17 ✅ |
| n_cohorts | 6 | 6 | 6 ✅ |
| `claim_wf282_g5_ru_seed0_test_auc` present | NO | YES | YES ✅ |
| `WF_282_G5_RU_SEED0_TEST_AUC=0.7716` in catalog | NO | YES | YES ✅ (value cell + 0.7716 grep = 8 matches) |
| Section heading §4.5.1 | §5.4 (BLOCKED) | §4.5.1 (MEASURED) | §4.5.1 ✅ |
| `wave_budget_tracker.py` constants (10) | 0 | 10 | 10 (lines 261-272) ✅ |

---

## 2. Dashboard rebuild

```
[Wf-268C-Dash] built dashboard: 33 claims, 17 honest-negatives, 6 cohorts, html_len=76985
[Wf-265-Dashboard] wrote /tmp/dashboard_v286d.html (75.6 KB); build JSON at /tmp/dashboard_v286d.build.json.
```

`grep -c "0.7716\|Wf-282" /tmp/dashboard_v286d.html` → **8 matches** (claim card text + value cell + section heading + cross-ref). Persisted to `molmetal/reports/wf265_dashboard/dashboard.html`.

---

## 3. Wf-284C propagation completeness

| Target | Path | Status |
|---|---|:---:|
| Pre-flight §9.AK | `TODO/completed/207_unified_pre_flight_checklist_v2.md` | ✅ |
| Paper §4.5.1 | `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` | ✅ |
| Wave-budget constants (10) | `molmetal/scripts/wave_budget_tracker.py` lines 261-272 | ✅ |
| Dashboard claim card | `molmetal/scripts/paper_dashboard.py` + `molmetal/reports/wf265_dashboard/dashboard.html` | ✅ |
| Cf-265 propagation_report | `molmetal/reports/wf284c_g5_propagation/propagation_report.md` | ✅ |

---

## 4. Honest framing (party-twenty-spirit §3.5)

- **DO**: cite `claim_wf282_g5_ru_seed0_test_auc` for the Wf-282 Ru seed=0 MEASURED test_auc=0.7716 finding.
- **DO NOT** claim Wf-282 G5 sweep MEASURED COMPLETE on the 15-pair cohort — only Ru seed=0 MEASURED; 14/15 PENDING per §4.5.1 + §9.AK.
- **DO NOT** claim `WF_282_G5_RU_SEED0_TEST_AUC=0.7716` BEATS Krasnov 0.81 (single-pair comparison is NOT a Krasnov-beat claim).

---

## 5. Output summary

- pre_count: 32 (Wf-280C baseline)
- post_count: **33** (after Wf-284C added claim_wf282_g5_ru_seed0_test_auc)
- wf284c_included: **YES** (claim card present, value=0.7716, section=§4.5.1)
- dashboard_path: **`/tmp/dashboard_v286d.html`** (also persisted to `molmetal/reports/wf265_dashboard/dashboard.html`)
- paper_auto_update.py rerun: **NOT REQUIRED** (constants live in wave_budget_tracker.py; dashboard catalog is hand-curated per TODO-188 §5.1, auto-update.py has no WF_282 hook)
