# Wf-284C — G5 Ru seed=0 test_auc=0.7716 Propagation Report

**Date**: 2026-09-29 00:35 CST
**Workflow**: Wf-284C (post-Wf-282 + Wf-283A G5 sweep recovery)
**Source MEASURED finding**: `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` line: `[Ru seed=0] DONE best_epoch=47 val_auc=0.8177 test_auc=0.7716 test_pearson=0.548 (2850.0s, featurise=0.5s)` + `molmetal/reports/wf283a_g5_recovery/recovery_report.md` §1.3-§1.5
**Status**: ✅ PROPAGATED to 4 catalogs + 1 dashboard

---

## 1. Propagation targets

| # | Catalog | Path | Action | Status |
|---:|---|---|---|:---:|
| 1 | Pre-flight checklist | `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AK | Added new entry "Wf-282 G5 Ru seed=0 MEASURED test_auc=0.7716" (replaces the §9.AB "BLOCKED at GPU-ordinal layer" status for the Ru seed=0 pair specifically; 14/15 PENDING preserved) | ✅ |
| 2 | Paper §4.5 | `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.1 | Updated from "BLOCKED at GPU-ordinal layer (0/15 pairs)" to "1/15 MEASURED (Ru seed=0 test_auc=0.7716), 14/15 PENDING" | ✅ |
| 3 | Wave budget tracker | `molmetal/scripts/wave_budget_tracker.py` (lines 248-269) | Added 9 new constants: `WF_282_G5_RU_SEED0_TEST_AUC=0.7716`, `WF_282_G5_RU_SEED0_VAL_AUC_PEAK=0.8177`, `WF_282_G5_RU_SEED0_BEST_EPOCH=47`, `WF_282_G5_RU_SEED0_TEST_PEARSON=0.5479`, `WF_282_G5_RU_SEED0_ELAPSED_SECONDS=2850.5`, `WF_282_G5_RU_SEED0_N_EPOCHS=50`, `WF_282_G5_WRAPPER_PID=609528`, `WF_282_G5_CKPT_TS`, `WF_282_G5_CKPT_PATH`, `WF_282_G5_SOURCE_LOG` | ✅ |
| 4 | Paper dashboard | `molmetal/scripts/paper_dashboard.py` (added claim `claim_wf282_g5_ru_seed0_test_auc` to CLAIMS list) + `molmetal/reports/wf265_dashboard/dashboard.html` | Added new MEASURED claim card (id=`claim_wf282_g5_ru_seed0_test_auc`, section=§4.5.1); re-ran `paper_dashboard.py --verbose --out-html /tmp/dashboard_v284c.html` → 33 claims (was 32), 17 honest-negatives, 6 cohorts, html_len=76985 bytes; copied to `molmetal/reports/wf265_dashboard/dashboard.html` (77,420 bytes) | ✅ |

---

## 2. Constants added

| Constant | Value | Source |
|---|---:|---|
| `WF_282_G5_RU_SEED0_TEST_AUC` | **0.7716** | `Ru/seed0/ru/per_seed_metrics.json::test_auc` (status=OK) |
| `WF_282_G5_RU_SEED0_VAL_AUC_PEAK` | 0.8177 | run.log best_epoch=47 val_auc=0.8177 |
| `WF_282_G5_RU_SEED0_BEST_EPOCH` | 47 | run.log DONE best_epoch=47 |
| `WF_282_G5_RU_SEED0_TEST_PEARSON` | 0.5479 | run.log test_pearson=0.548 |
| `WF_282_G5_RU_SEED0_ELAPSED_SECONDS` | 2850.5 | per_seed_metrics elapsed_seconds=2850.5 |
| `WF_282_G5_RU_SEED0_N_EPOCHS` | 50 | run.log [epoch 50/50] final line |
| `WF_282_G5_WRAPPER_PID` | 609528 | zsh wrapper launched 2026-09-28 23:41:04 CST (Wf-283A recovery) |
| `WF_282_G5_CKPT_TS` | 2026-09-29T00:28:00+08:00 | ckpt save timestamp |
| `WF_282_G5_CKPT_PATH` | `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ckpts/dmpnn_mtw_ru_seed0.pt` | 11.3 MB final-epoch ckpt |
| `WF_282_G5_SOURCE_LOG` | `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` | per Wf-283A recovery §1.3 + Wf-282 synthesis |

---

## 3. Dashboard verification

Dashboard regeneration log:
```
[Wf-268C-Dash] built dashboard: 33 claims, 17 honest-negatives, 6 cohorts, html_len=76985
[Wf-265-Dashboard] wrote /tmp/dashboard_v284c.html (75.6 KB); build JSON at /tmp/dashboard_v284c.build.json.
```

Verification (`grep -c "0.7716\|Wf-282" /tmp/dashboard_v284c.html`): **4 matches** (claim card text + value cell + section heading + cross-ref).

Persisted to `molmetal/reports/wf265_dashboard/dashboard.html` (77,420 bytes; +2,506 vs Wf-280C baseline per `molmetal/reports/wf280c_dashboard_refresh/`).

---

## 4. Honest framing (party-twenty-spirit §3.5)

- **DO claim**: "Wf-282 G5 Ru seed=0 MEASURED test_auc=0.7716 (first G5 50-ep test_auc from TODO-181 sweep on RTX 5090 GPU-1, cuda:0 fix post-cuda:1-bug, YuelBond + Wasserstein + closure-bounded default config card, val_auc peak 0.8177 @ epoch 47, test_pearson 0.5479, ckpt 11.3 MB saved at 00:28 CST 2026-09-29)."
- **DO claim**: "Wf-282 G5 Ru seed=0 test_auc 0.7716 vs TODO-180 G4 30-ep Ru mean 0.7527 → Δ=+0.0189 (expected 30→50-ep lift, within honest framing)."
- **DO NOT claim**:
  - "Wf-282 G5 5-metal × 3-seed × 50-ep sweep MEASURED COMPLETE" — only Ru seed=0 MEASURED; Ru seed=1 RUNNING; 13/15 PENDING per `molmetal/reports/wf282_synthesis/followup_summary.md` §HN-1
  - "Wf-282 G5 Ru seed=0 test_auc 0.7716 BEATS Krasnov 0.81" — Krasnov comparison is per §4.4 paper target, NOT G5 single-pair target
  - "Wf-280D's cut-at-epoch-47/no-ckpt claim was correct" — Wf-283A confirms it was WRONG; full 50-epoch run + 11.3 MB ckpt saved

---

## 5. Cross-references

- 207 §9.AK (this propagation entry)
- 188 §4.5.1 (updated from BLOCKED → 1/15 MEASURED)
- `molmetal/scripts/wave_budget_tracker.py` constants block (lines 248-269)
- `molmetal/scripts/paper_dashboard.py` CLAIMS list (claim_wf282_g5_ru_seed0_test_auc)
- `molmetal/reports/wf265_dashboard/dashboard.html` (persisted)
- `molmetal/reports/wf283a_g5_recovery/recovery_report.md` §1.3-§1.5 (authoritative MEASURED source)
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (raw log line)
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics.json` (parsed JSON)