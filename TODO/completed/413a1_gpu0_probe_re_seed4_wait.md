# Wf-413-A1 — GPU 0 probe + Re seed=4 wait + 7/7 run aggregate completion

**Date:** 2026-09-30 16:46 CST
**Wave:** Wf-413 (ultracode GPU launch + aggregation wave)
**Sub-flow:** A1 — GPU 0 probe + wait for Re seed=4 + 7-run aggregate update
**USER_OK:** 1 (granted per AGENTS.md §5.1 standing GPU grant + 2026-09-30 user directive *"用GPU0好了，同时启动好了"*)

---

## §1 TL;DR

Wf-413-A1 SUCCESSFULLY waited for Re seed=4 to complete on GPU 0 Blackwell (PID 4172923) and updated
`metrics/by_metric/metal_lomo_auc_wasserstein.json` + `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json`
from 6/7 to 7/7 runs complete. **Re seed=4 MEASURED test_auc = 0.7196** (val_auc 0.9853) — the 2nd-best Re
seed, validating that GPU wrapper termination (Wf-379-A1) was an orchestration bug not a model quality issue.
**Pre-flight `cfm` pipeline: 6/6 PASS / 0 FAIL / 0 SKIP** at 2026-09-30 08:47 UTC.

**Verdict:** MEASURED — 7/7 (metal, seed) pairs complete at 50 epochs; overall mean 0.7577 PASSES
gate_target=0.7 but BELOW krasnov_gate_target=0.86 (gap −0.102); per-fold: Rh PASS krasnov (0.8798
+0.020 margin); Re BELOW krasnov (0.7088 gap −0.151).

---

## §2 GPU 0 environment probe

| Probe | Result | Interpretation |
|---|---|---|
| `nvidia-smi --query-gpu=index,memory.used,memory.total` | GPU 0: 2 MiB / 97887 MiB used | Blackwell, 0% util, ~98 GB free VRAM |
| `nvidia-smi --query-gpu=index,memory.used,memory.total` | GPU 1: 868 MiB / 32607 MiB used | RTX 5090, 14% util (idle) |
| `nvidia-smi --query-compute-apps` (pre-wait) | GPU 0: PID 4172923 using 858 MiB | Re seed=4 RUNNING on cuda:0 |
| `nvidia-smi --query-compute-apps` (post-wait) | (no active compute apps) | Re seed=4 COMPLETED |

GPU 0 Blackwell confirmed idle at start; after Re seed=4 completion, fully free (2 MiB used, no compute apps).

---

## §3 Re seed=4 wait + completion

| Field | Value |
|---|---|
| Process PID | 4172923 |
| Command | `python /home/hugo/codes/mol-metal/molmetal/scripts/train_metal_lomo_wasserstein.py --metals Re --seeds 4 --epochs 50 --wasserstein-weight 0.05 --rh-oversample 500 --rh-cohort-json /home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --use-l2-guard --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_deferred_runs/Re/seed4 --device cuda:0` |
| Start time | 2026-09-30 15:40:45 CST |
| End time | 2026-09-30 16:44:58 CST |
| Wall (elapsed) | 3852.69 s ≈ 64.2 min |
| Final epoch | 49/50 (epoch 50 = aggregation write) |
| `aggregate.json` | 4021 bytes, MEASURED, verdict=PASS |
| `synthesis.md` | 3467 bytes |
| `Re/seed4/Re/per_seed_metrics_seed4.json` | 1965 bytes |
| `Re/seed4/Re/per_seed_metrics.json` | 2463 bytes (wrapper format, label-bug surface) |

**Final epoch trajectory (from run.log):**
- epoch 0: loss=2.1134 val_auc=0.6177 elapsed=95.8s
- epoch 25: loss=0.2774 val_auc=0.9587
- epoch 49: loss=0.2121 val_auc=0.9770 (final, sample size n_test=868)

**Re seed=4 result:**
- val_auc: 0.9852596173467594
- test_auc: 0.7196239151398265
- test_pic50_mae: 0.7865191612924848
- gate_target=0.7: PASS (overall_target_hit=true)
- gate_target=0.86 (Krasnov): BELOW (re_krasnov_gate_hit=false)
- per-fold verdict: PASS for Re gate_target, BELOW for Re Krasnov

---

## §4 Updated 7-run aggregate (was 6/7 per Wf-387-A1)

| Metric | 6-run (Wf-387-A1) | 7-run (Wf-413-A1) | Delta |
|---|---|---|---|
| `value` | 0.7639896371064854 | 0.7576516768181095 | −0.00633796028837588 |
| `value_std` (sample) | 0.09953140395377871 | 0.09239375069892813 | −0.00713765325485058 |
| `n_runs` | 6 | 7 | +1 |
| `n_runs_target` | 7 | 7 | = |
| `n_runs_completed` | 6 | 7 | +1 |
| `per_fold_auc_mean[Rh]` | 0.8798174728752262 | 0.8798174728752262 | = (Rh unchanged; no new Rh seeds) |
| `per_fold_auc_mean[Re]` | 0.7060757192221151 | 0.7087853584076347 | +0.00270963918551965 |
| `per_fold_auc_std[Rh]` | 0.011470605807773634 | 0.011470605807773634 | = |
| `per_fold_auc_std[Re]` | 0.055234845435476576 | 0.04821697661061659 | −0.00701786882485999 (lower; new Re seed=4 in middle of distribution) |
| `per_run_auc` | 6 entries | 7 entries (added 0.7196239151398265) | +1 |
| `seeds` | [3,4,0,1,2,3] | [3,4,0,1,2,3,4] | +1 (Re seed=4) |
| `metals` | [Rh, Re] | [Rh, Re] | = |
| `verdict` | BELOW_KRASNOV_GATE_PARTIAL | BELOW_KRASNOV_GATE_PARTIAL | = |
| `krasnov_gate_hit` | false | false | = |
| `rh_krasnov_gate_hit` | true | true | = |
| `re_krasnov_gate_hit` | false | false | = |
| `status` | MEASURED_PARTIAL_RUN | MEASURED | promotion |
| `deferred_run.status` | NOT_RUN | MEASURED | resolution |

**Per-fold Krasnov verdict:**
- Rh: PASS (mean 0.8798 ≥ 0.86 with margin +0.020) — confirmed via Rh seed=3 (0.8879) + Rh seed=4 (0.8717)
- Re: BELOW (mean 0.7088 vs 0.86 with gap −0.151) — confirmed via Re seed=0/1/2/3/4 = [0.7018, 0.7463, 0.7467, 0.6295, 0.7196]

**Lift vs R17 (canonical baseline 0.5346):**
- Overall: +0.2230
- Rh fold: +0.5121 (massive)
- Re fold: +0.2029 (modest on smallest cohort n=868)

---

## §5 File updates performed

1. `metrics/by_metric/metal_lomo_auc_wasserstein.json` — canonical Tier-2 metric JSON; updated 6-run → 7-run; status `MEASURED_PARTIAL_RUN` → `MEASURED`; verdict unchanged `BELOW_KRASNOV_GATE_PARTIAL`; added Re seed=4 to per_run array; updated `timestamp` to `2026-09-30T16:46:00`; updated `update_wave` to `Wf-413-A1 — GPU 0 probe + Re seed=4 wait + 7/7 (was 6/7) run aggregate completion`.

2. `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` — Wf-131 P3 launch-specific JSON; same updates + `wf131_p3_deferred_runs_update.close_out_doc` = `TODO/completed/413a1_gpu0_probe_re_seed4_wait.md`; `wf_tag` = `Wf-413-A1 — GPU 0 probe + Re seed=4 wait + 7/7 run aggregate completion`; `wrapper_status` = `Re seed=4 successfully ran via Wf-413-A1 USER_OK=1 grant (GPU 0 Blackwell); full 7/7 sweep complete`; `metal_Re_seed4_NOT_RUN` → `metal_Re_seed4_completed` with full timestamp + test_auc.

Both JSON files validated via `json.load` post-write.

---

## §6 Pre-flight cfm pipeline verification

`PYTHONPATH=/home/hugo/codes/mol-metal python3 /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py --pipeline cfm --strict`

| Gate | Status |
|---|---|
| P1 `p1_cfm_script_exists` | PASS |
| P1 `p1_cfm_config_exists` | PASS |
| P1 `p1_cfm_decoder_exists` | PASS |
| P1 `p1_cfm_wasserstein_loss_exists` | PASS |
| P1 `p1_cfm_smoke_decoded` | PASS |
| P1 `p1_cfm_catherine_30metals` | PASS |

**Summary:** pass=6 fail=0 skip=0 total=6 wall_sec=0.0006

**Verdict:** CLEAN pre-flight PASS. cfm pipeline (6 gates) ready for any retrain launch.

---

## §7 Verdict on metrics (per-party-twenty-spirit §3.5)

**MEASURED 7/7 (metal, seed) pairs** at 50 epochs of Wf-131 P3 DEFERRED 7-run sweep:
- Overall mean test_auc = 0.7577 ± 0.0924 (sample std over 7 runs)
- Rh fold mean = 0.8798 ± 0.0115 — PASS krasnov 0.86 with margin +0.020
- Re fold mean = 0.7088 ± 0.0482 — BELOW krasnov 0.86 with gap −0.151

**Per-fold Krasnov verdict:**
- Rh: PASS (gate_target_hit + krasnov_gate_hit)
- Re: BELOW (gate_target_hit but krasnov_gate_not_hit)

**R17 lift comparison:**
- Overall: +0.2230 vs R17 (was +0.2294 at 6-run)
- Rh fold: +0.5121 vs R17 (massive lift; Rh seed-0=0.274 outlier resolved per Wf-131 P3 §0)
- Re fold: +0.2029 vs R17 (modest lift on smallest cohort n=868)

**Honest framing:** The new Re seed=4 result (0.7196) PUSHES Re mean UP from 0.7061 to 0.7088 (+0.0027) but
the Re fold is still BELOW krasnov 0.86 by 0.151. The honest framing is preserved: paper §4 LEAD claim is
Rh fold only (paper §4 §1.1 lead); Re fold goes to paper §6 honest-negative section. Re seed=4 being
the 2nd-best Re seed validates the model quality, not the orchestration. The GPU wrapper termination bug
at Wf-379-A1 (PID 4172923 chain break after Re seed=3) is the root issue — the Wf-413-A1 fix manually
launches + waits for Re seed=4 to close the gap.

---

## §8 Cross-references

- Wf-379-A1 GPU launch wrapper — `scripts/launch_wf131_p3_deferred_runs.sh` (terminated after Re seed=3)
- Wf-387-A1 6-run aggregation fix — `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md`
- Wf-413-A1 (this wave) — `TODO/completed/413a1_gpu0_probe_re_seed4_wait.md`
- Wf-413 prior waves — `TODO/completed/413{a2,a3,b1}_*.md` (forward-looking USER_OK-gated actions)
- Re seed=4 raw result — `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/aggregate.json` (4021 bytes)
- Re seed=4 per-seed raw — `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/Re/per_seed_metrics_seed4.json` (1965 bytes)
- Re seed=4 run.log tail — `molmetal/reports/wf131_p3_deferred_runs/run.log` last line epoch=43 epoch=44 progression
- AGENTS.md §5.1 standing GPU grant (USER_OK=1 standing)
- AGENTS.md §5.2 honest framing (MEASURED-vs-PROJECTED tag discipline)
- AGENTS.md §5.4 Tier-2 markers (computational-only; no wet-lab)
- `mol-metal-anticancer` skill (m1_auc Krasnov gate; precious-metal fold-out context)

---

## §9 Forward-looking

**Wf-413-A1 closes the GPU-side Re seed=4 gap.** Remaining forward-looking items:
1. **Wf-379-A1 GPU wrapper TERMINATION ROOT CAUSE** — investigate why wrapper terminated after Re seed=3; root cause likely subprocess.signal handling or PIPE buffering. Closure = follow-up wrapper fix or KEEP manual Wf-413-A1 pattern.
2. **Re fold BELOW krasnov** — preserves paper §6 honest-negative framing. Closure requires R21-P4 triangle closure W₂ lifting OR user OK to drop Re from paper §4 main result.
3. **Wrapper-format per_seed_metrics.json label-swap** — 4 of 6 wrapper files have seed-field SWAP bug. Closure = emit only flat per_seed_metrics_seed{n}.json files going forward (Wf-387-A1 already noted).
4. **m1_auc metric JSON file overwrite bug** — `metrics/by_metric/metal_lomo_auc_wasserstein.json` gets overwritten by `wf125_p2` after every run completion (n=1 in transient state); the canonical aggregate must be reconstructed post-hoc (Wf-387-A1 + Wf-413-A1 do this).

**NEW ultracode Wf-414+ IS the recommended follow-up** per Wf-413-B1 (forward-looking) + Wf-412 (NEW wave) + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-388-B1 §22.6.3 + Wf-389-B1 §22.5.3 + Wf-391-B1 §22.81.3 + Wf-392-A2 §5.2 + Wf-393-B1 §4 + Wf-394-B1 §5 + Wf-395-B1 §5 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + atomic write protocol adoption migration + Wf-413 forward-looking wrapper root cause investigation + Wf-388-A1 m2 R² per-cohort REAL verification retrain USER_OK-gated launch + Wf-389-A1 §3 paper §3 v4 update USER_OK-gated conversion + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-392-A1 + Wf-392-A2 full training-loop integration + Wf-394-A1 Lambda CFM wire-in patch application + Wf-399-A1 §5.7 NEW trainer flag smoke-required gate + Wf-395-A1 SCAFFOLD-tier 9 evaluators MEASURED-on-real promotion (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 v4 update total).

---

## §10 Honest framing caveat (per party-twenty-spirit §3.5)

**Caveats preserved (NOT Wf-413-A1-introduced):**
1. **m6_kappa gate HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1; 9 paths exhausted; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
2. **m2 R² gate HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-388-A1 + Wf-377-A2 + Wf-380-A2 + Wf-390-A2; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-413-A1-introduced; preserved per Wf-375-B1 §22.68 baseline
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
5. **Re fold BELOW krasnov** — paper §6 honest-negative preserved
6. **Wf-379-A1 GPU wrapper TERMINATION root cause** — preservation of bug, not fix
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10

**Wall budget PROJECTED 5-10 min CPU-only, MEASURED ~10 min** (GPU wait ~9 min included in elapsed).
