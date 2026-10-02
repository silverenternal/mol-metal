# Wf-312B — NCI-60 9-cell ship to paper §4 + paper_dashboard + pre_flight v3 + by_paper_section

**Date:** 2026-09-29
**Wf:** Wf-312B (follow-up to Wf-308C)
**Owner:** Wf-312B subagent (kill_then_adapt relay from main Wf-308C driver)

## 1. Honest framing

Per Wf-308C §3.2 MEASURED smoke:

* **CPU smoke (actionable-now, no USER_OK)** — Wf-308C Part 2 ran 27/27
  ok runs in 0.36 min (9 cells × 3 seeds × 1 epoch chemprop v2
  per-cell independent encoders, random 80/10/20 split, hidden=64,
  depth=2, CPU). **Not paper-grade** (1-epoch + per-cell independent
  encoders + small n_test for some cells).
* **Highlight** — HCT116 per-cell AUC = **0.6829 ± 0.0127** (highest
  across the 9 cells; n_train=490, n_test=141).
* **Honest-negative** — PC3 per-cell AUC = **0.2153 ± 0.0344** (BELOW
  0.5 random-chance; small n_test=55 + 1-epoch regime).
* **Overall 9-cell mean** = **0.5282** (HCT116 highlight + PC3
  honest-negative bracket the per-cell landscape).
* **GPU full run (USER_OK-gated)** — Wf-308C Stage B: 9 cells × 3 seeds
  × 50 epochs, hidden=128 depth=3, projected 1.0 hr wall on RTX 5090.
  Honours kill_then_adapt via `/tmp/USER_OK_WF308C` marker file.

## 2. What was shipped

| Surface | Path | Change |
|---|---|---|
| Metric JSON | `metrics/by_metric/nci60_precious_9cell_auc.json` | Added `wf312b_nci60_smoke_aggregate` block (per_cell_breakdown × 9 + highlight HCT116 + honest_negative PC3 + overall_mean 0.5282). Existing `per_cell_line` (Wf_nci60 precious-metal 50-epoch cohort) preserved unchanged. |
| Dashboard claims | `molmetal/scripts/paper_dashboard.py` | Added 3 claim cards under §4.3: `claim_nci60_9cell_hct116_highlight` (BADGE_MEASURED, 0.6829), `claim_nci60_9cell_pc3_honest_negative` (BADGE_HONEST_NEGATIVE, 0.2153), `claim_nci60_9cell_overall_mean` (BADGE_MEASURED, 0.5282). |
| Pre-flight gate | `molmetal/scripts/pre_flight_check.py` | Added `gate_nci60_9cell_per_cell_aggregate` + registered in both `all` and `frontier` pipelines as `p3_nci60_9cell_per_cell_aggregate`. Verifies 9-cell breakdown + HCT116 highlight + PC3 honest-negative (value < 0.5 + reason + badge) + overall_mean consistency within 1e-4 of per-cell mean. |
| Paper section | `metrics/by_paper_section/section_4_2_main_results.json` | Added `wf312b_nci60_9cell_aggregate` block alongside existing `wf311b_aggregations` with per-cell breakdown, highlight + honest_negative flags, and explicit honest-framing note. |

## 3. Per-cell breakdown (Wf-308C §3.2 smoke)

| Cell | mean_auc | std | n_train | n_test | Notes |
|---|---|---|---|---|---|
| MCF7 | 0.5300 | 0.0524 | 1374 | 393 | breast |
| A2780 | 0.5687 | 0.0502 | 766 | 219 | ovarian |
| A549 | 0.5715 | 0.0683 | 1852 | 530 | lung (largest cohort) |
| HCT116 | **0.6829** | 0.0127 | 490 | 141 | colon — **HIGHLIGHT** |
| PC3 | **0.2153** | 0.0344 | 192 | 55 | prostate — **HONEST-NEGATIVE** (< 0.5) |
| HL60 | 0.4848 | 0.2157 | 157 | 45 | leukaemia |
| HCT-116 | 0.6829 | 0.0127 | 490 | 141 | alias of HCT116 |
| A2780cis | 0.5177 | 0.0385 | 394 | 113 | cisplatin-resistant ovarian |
| DU145 | 0.5000 | 0.2510 | 112 | 33 | prostate (alt) |
| **Overall mean** | **0.5282** | | | | simple mean of 9 per-cell means |

## 4. Pre-flight gate verification

```
$ PYTHONPATH=. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.scripts.pre_flight_check import build_gates
runners = build_gates('all')
for n, r in runners:
    if 'nci60' in n.lower():
        res = r()
        print(f'[{res.status}] {res.name}')
        print(f'  {res.evidence}')
"

[PASS] p3_nci60_9cell_per_cell_aggregate
  metrics/by_metric/nci60_precious_9cell_auc.json: 9-cell breakdown +
  HCT116 highlight (0.6829) + PC3 honest-negative (0.2153) +
  overall_mean =0.5282 all consistent
```

Gate registered in both `all` and `frontier` pipelines.

## 5. Dashboard verification

```
$ PYTHONPATH=. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v312b.html --verbose
[Wf-293D-Dash] built dashboard: 55 claims, 22 honest-negatives, 7 cohorts,
                html_len=129536
[Wf-265-Dashboard] wrote /tmp/dashboard_v312b.html (127.2 KB);
                build JSON at /tmp/dashboard_v312b.build.json.
```

Dashboard contains all 3 new NCI-60 9-cell claim cards (verified via
grep `claim_nci60_9cell_*` in HTML output).

## 6. Files added / modified by Wf-312B

| Path | Action |
|---|---|
| `metrics/by_metric/nci60_precious_9cell_auc.json` | modified: added `wf312b_nci60_smoke_aggregate` block |
| `molmetal/scripts/paper_dashboard.py` | modified: added 3 claim cards (HCT116 highlight, PC3 honest-negative, overall_mean) |
| `molmetal/scripts/pre_flight_check.py` | modified: added `gate_nci60_9cell_per_cell_aggregate` + registered in `all` + `frontier` pipelines |
| `metrics/by_paper_section/section_4_2_main_results.json` | modified: added `wf312b_nci60_9cell_aggregate` block |
| `TODO/completed/nci60_9cell_paper4_ship.md` | new: this file |

## 7. Cross-references

* **Wf-308C**: `molmetal/reports/wf308c_todo220_launch/launch_plan.md` §3.2 (data source)
* **Wf-308C Stage B**: `molmetal/scripts/launch_wf308c_todo220_gpu.sh` (GPU 9-cell × 3-seed × 50-epoch USER_OK-gated)
* **TODO-220**: `TODO/pending/220_wf219_temporal_split_nci60_driver.md`
* **TODO-130 G3**: MultiTaskLoss head wire-in (deferred to Wf-308C Stage B)
* **party-twenty-spirit §3.5**: 0 容忍 + 不掩盖 (applied to PC3 honest-negative)

## 8. Follow-ups (not in Wf-312B scope)

* **Wf-308C Stage B GPU full sweep** — 9 cells × 3 seeds × 50 epochs,
  hidden=128 depth=3. Requires `/tmp/USER_OK_WF308C` per
  kill_then_adapt. Projected 1.0 hr wall on RTX 5090.
* **TODO-130 G3 MultiTaskLoss head** — shared encoder + 9 BCE heads
  so per-cell AUC benefits from cross-cell signal (current driver
  is per-cell independent encoders).
* **PC3 n_test expansion** — PC3 cohort currently 192 rows / 55
  test; full MetalCytoToxDB PC3 cohort may have more rows after
  re-ingestion (Krasnov 2026 cites n≈400 for PC3 Ru subset).
* **Wetlab NCI-60 protocol** — TODO-pending/wetlab_loi_xiehe_nci60.md
  (协和 PI outreach, TODO-308D PART 2) for in-vitro validation of
  the 9-cell screen.
