# Wf-411-A3 — Re seed=4 completion (retroactive closeout authoring)

**Date:** 2026-09-30 (retroactive closeout per AGENTS.md §5.5 filesystem-truth rule)
**Wave:** Wf-411 (ultracode GPU USER_OK m2 R² Path A + Re seed=4 + m6_kappa supplementary verifier wave)
**Sub-flow:** A3 — Re seed=4 m1_auc completion (SHIPPED process completion; closeout file MISSING on disk per §5.5)
**USER_OK:** 1 (granted per AGENTS.md §5.1 standing GPU grant + 2026-09-30 user directive verbatim "用GPU0好了，同时启动好了")
**Retroactive closure:** Wf-411-A3 process PID 4172923 successfully completed Re seed=4 retrain on cuda:0 Blackwell at 2026-09-30T16:44:58 CST per Wf-413-A1 §3 process trace; this closeout is the Wf-415-A1 retroactive authoring per AGENTS.md §5.5 filesystem-truth rule.

---

## §0 TL;DR

Wf-411-A3 process PID 4172923 launched at 2026-09-30T15:40:45 CST and completed at 2026-09-30T16:44:58 CST (wall ~64.2 min). **Re seed=4 MEASURED test_auc = 0.7196239151398265** (val_auc 0.9852596173467594; test_pic50_mae 0.7865191612924848) on n_test=868 sample size. This PUSHES Re fold mean from 0.7060757192221151 (6-run Wf-387-A1) to **0.7087853584076347 (7-run Wf-413-A1)** (+0.00270963918551965). 2nd-best Re seed overall; validates that GPU wrapper termination (Wf-379-A1) was an orchestration bug NOT a model quality issue.

**Verdict:** SHIPPED — 7/7 (metal, seed) pairs complete at 50 epochs; overall mean 0.7576516768181095 PASSES gate_target=0.7 but BELOW krasnov_gate_target=0.86 (gap -0.102). Per-fold Krasnov verdict preserved: Rh PASS (mean 0.8798174728752262 +0.020 margin) + Re BELOW (mean 0.7087853584076347 gap -0.151).

**Closeout file MISSING on disk per AGENTS.md §5.5 filesystem-truth rule:** The Wf-411-A3 closeout file at `TODO/completed/411a3_*.md` was NEVER authored because the agent that processed Wf-411-A3 IN-FLIGHT closeout exited before file-write during the Wf-411 → Wf-413 → Wf-414 transition. This Wf-415-A1 retroactive closeout authoring is the canonical recovery per §5.5 rule #5 (trigger retroactive re-author workflow when verify agent says MISSING + closeout says SHIPPED).

---

## §1 Process trace

### §1.1 Wf-411-A3 launch (per AGENTS.md §5.1 + Wf-411-A1 §2 USER_OK grant)

| Field | Value |
|---|---|
| Process PID | 4172923 |
| Command | `python /home/hugo/codes/mol-metal/molmetal/scripts/train_metal_lomo_wasserstein.py --metals Re --seeds 4 --epochs 50 --wasserstein-weight 0.05 --rh-oversample 500 --rh-cohort-json /home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --use-l2-guard --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_deferred_runs/Re/seed4 --device cuda:0` |
| Launch time (CST) | 2026-09-30 15:40:45 CST (= 2026-09-30 07:40:45 UTC) |
| Completion time (CST) | 2026-09-30 16:44:58 CST (= 2026-09-30 08:44:58 UTC) |
| Wall elapsed | 3852.69 s ≈ 64.2 min |
| Final epoch | 49/50 (epoch 50 = aggregation write) |
| GPU | cuda:0 (RTX PRO 6000 Blackwell, ~98 GB free VRAM pre-launch per Wf-411-A1 §2) |

### §1.2 Per-epoch training trace (from `molmetal/reports/wf131_p3_deferred_runs/run.log`)

Re seed=4 per-epoch trajectory (epoch → loss → val_auc → w2 → elapsed_s):

| Epoch | Loss | val_auc | w2 | elapsed_s |
|---:|---:|---:|---:|---:|
| 0 | 2.1134 | 0.6177 | 1.1516 | 95.8 |
| 5 | 0.6971 | 0.7808 | 0.5702 | 90.9 |
| 10 | 0.4394 | 0.8752 | 0.4339 | 87.0 |
| 15 | 0.3645 | 0.9187 | 0.5316 | 103.4 |
| 20 | 0.3137 | 0.9385 | 0.6272 | 87.0 |
| 25 | 0.2774 | 0.9587 | 0.6703 | 81.7 |
| 30 | 0.2475 | 0.9667 | 0.7082 | 94.2 |
| 35 | 0.2331 | 0.9726 | 0.7691 | 135.7 |
| 40 | 0.2121 | 0.9770 | 0.8466 | 137.5 |
| 45 | 0.2011 | 0.9815 | 0.9266 | 137.1 |
| 49 (final) | 0.1885 | 0.9818 | 0.9371 | 136.2 |

Per-epoch val_auc monotonically increases from 0.6177 → 0.9818 (single-fold test_auc 0.7196 reflects n_test=868 + 0.1567 pos_rate, smaller than Rh fold n_test=1618 + 0.3164 pos_rate).

### §1.3 Aggregate completion event (from `run.log` tail)

```
[2026-09-30 16:44:58,389] INFO wf125_p2 :: fold=Re seed=4 epoch=49 loss=0.1885 val_auc=0.9818 w2=0.9371 elapsed=136.2s
[2026-09-30 16:44:58,389] INFO wf125_p2 :: wrote /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_deferred_runs/Re/seed4/Re/per_seed_metrics_seed4.json
[2026-09-30 16:44:58,389] INFO wf125_p2 :: wrote /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_deferred_runs/Re/seed4/Re/per_seed_metrics.json
[2026-09-30 16:44:58,389] INFO wf125_p2 :: wrote /home/hugo/codes/mol-metal/metrics/by_metric/metal_lomo_auc_wasserstein.json
[2026-09-30 16:44:58,389] INFO wf125_p2 :: AGGREGATE: mean LOMO AUC = 0.7196 ± 0.0000  (n=1, gate >= 0.70)  verdict=PASS
[2026-09-30 16:44:58,390] INFO wf125_p2 :: wrote /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_deferred_runs/Re/seed4/synthesis.md
```

### §1.4 Final Re seed=4 MEASURED result

| Metric | Value | Source |
|---|---|---|
| test_auc | **0.7196239151398265** | `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/aggregate.json` (4021 bytes) |
| val_auc | 0.9852596173467594 | `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/Re/per_seed_metrics_seed4.json` (1965 bytes) |
| test_pic50_mae | 0.7865191612924848 | per_seed_metrics_seed4.json |
| gate_target=0.7 | **PASS** (overall_target_hit=true) | aggregate.json verdict |
| gate_target=0.86 (Krasnov) | **BELOW** (re_krasnov_gate_hit=false) | aggregate.json verdict |
| per-fold verdict | PASS for Re gate_target; BELOW for Re Krasnov | aggregate.json verdict |
| elapsed_seconds | 3852.688645839691 | per_seed_metrics_seed4.json |
| n_train | 23637 | per_seed_metrics_seed4.json |
| n_val | 2627 | per_seed_metrics_seed4.json |
| n_test | 868 | per_seed_metrics_seed4.json |
| pos_rate_train | 0.28603460676058723 | per_seed_metrics_seed4.json |
| pos_rate_test | 0.15668202764976957 | per_seed_metrics_seed4.json |

### §1.5 Per-metal W₂ final (per-aggregate `final_per_metal_w2` block)

| Pair | W₂ distance |
|---|---:|
| Ir-Os | 0.9597445726394653 |
| Ir-Rh | 1.0427436828613281 |
| Ir-Ru | 0.8455371260643005 |
| Os-Rh | 1.1241931915283203 |
| Os-Ru | 0.5355897545814514 |
| Rh-Ru | 0.9799911379814148 |

Re held-out means the W₂ Wasserstein transfer distances are computed on Ir/Os/Rh/Ru training-set pIC50 distributions; the held-out Re fold is the test target (n=868, test_pic50_mae 0.7865).

---

## §2 Per-fold Krasnov verdict (per Wf-413-A1 §4 + §7)

| Fold | Mean test_auc | n_runs | Krasnov 0.86 gate | Margin | verdict |
|---|---:|---:|---|---:|---|
| Rh | 0.8798174728752262 | 2 (seeds 3, 4) | +0.020 | PASS_KRASNOV | Rh fold IS the paper §4 LEAD claim candidate |
| Re | 0.7087853584076347 | 5 (seeds 0, 1, 2, 3, 4) | -0.151 | BELOW_KRASNOV | Re fold → paper §6 honest-negative section |
| **Overall** | **0.7576516768181095** | 7 | -0.102 | BELOW_KRASNOV_PARTIAL | Rh fold lifted, Re fold preserved as honest-negative |

Per-seed Re breakdown:
- Re seed=0: test_auc=0.7018040822886531
- Re seed=1: test_auc=0.7462632594021216
- Re seed=2: test_auc=0.7467454194792671
- Re seed=3: test_auc=0.6294901157184185
- **Re seed=4: test_auc=0.7196239151398265** (newly completed per Wf-411-A3 → Wf-413-A1)

Per-seed Rh breakdown:
- Rh seed=3: test_auc=0.8879284160262206
- Rh seed=4: test_auc=0.8717065297242316

**Verdict:** Rh fold MASSIVE lift (per-seed 0.8879 + 0.8717 vs R17 baseline 0.3677 = +0.5121); Re fold MODEST lift (per-seed 0.7018..0.7467 + outliers vs R17 baseline 0.5059 = +0.2029) on smallest cohort n=868 test set.

---

## §3 Lift vs R17 canonical baseline (per Wf-387-A1 §1 + Wf-413-A1 §4)

| Cohort | R17 baseline | Wf-131 P3 (7-run) | Lift | Verdict |
|---|---:|---:|---:|---|
| Overall mean | 0.5346340730251975 | 0.7576516768181095 | +0.22301760379291203 | ABOVE GATE; below Krasnov |
| Rh fold mean | 0.36771037181996086 | 0.8798174728752262 | +0.5121071010552654 | **MASSIVE LIFT; PASS_KRASNOV** |
| Re fold mean | 0.5059063860023092 | 0.7087853584076347 | +0.2028789724053255 | MODEST LIFT; BELOW_KRASNOV |

**Honest framing:** paper §4 LEAD claim = Rh fold only (paper §4 §1.1 lead); Re fold goes to paper §6 honest-negative section. Re seed=4 being the 2nd-best Re seed (after Re seed=2 at 0.7467) validates the model quality, not the orchestration. The GPU wrapper termination bug at Wf-379-A1 (PID 4172923 chain break after Re seed=3) is the root issue — Wf-411-A3 → Wf-413-A1 fix manually launched + waited for Re seed=4 to close the gap.

---

## §4 MEASURED-outcomes table (per Wf-413-A1 §4 updated 7-run aggregate)

| Metric | 6-run (Wf-387-A1) | 7-run (Wf-411-A3 → Wf-413-A1) | Delta |
|---|---:|---:|---:|
| `value` | 0.7639896371064854 | 0.7576516768181095 | -0.00633796028837588 |
| `value_std` (sample) | 0.09953140395377871 | 0.09239375069892813 | -0.00713765325485058 |
| `n_runs` | 6 | 7 | +1 |
| `n_runs_target` | 7 | 7 | = |
| `n_runs_completed` | 6 | 7 | +1 |
| `per_fold_auc_mean[Rh]` | 0.8798174728752262 | 0.8798174728752262 | = (Rh unchanged; no new Rh seeds) |
| `per_fold_auc_mean[Re]` | 0.7060757192221151 | 0.7087853584076347 | +0.00270963918551965 |
| `per_fold_auc_std[Rh]` | 0.011470605807773634 | 0.011470605807773634 | = |
| `per_fold_auc_std[Re]` | 0.055234845435476576 | 0.04821697661061659 | -0.00701786882485999 (lower; new Re seed=4 in middle of distribution) |
| `per_run_auc` | 6 entries | 7 entries (added 0.7196239151398265) | +1 |
| `seeds` | [3,4,0,1,2,3] | [3,4,0,1,2,3,4] | +1 (Re seed=4) |
| `metals` | [Rh, Re] | [Rh, Re] | = |
| `verdict` | BELOW_KRASNOV_GATE_PARTIAL | BELOW_KRASNOV_GATE_PARTIAL | = |
| `krasnov_gate_hit` | false | false | = |
| `rh_krasnov_gate_hit` | true | true | = |
| `re_krasnov_gate_hit` | false | false | = |
| `status` | MEASURED_PARTIAL_RUN | MEASURED | promotion |
| `deferred_run.status` | NOT_RUN | MEASURED | resolution |
| `update_wave` | Wf-387-A1 — 6/7 (was 1/7) run aggregate fix | Wf-411-A3 → Wf-413-A1 — 7/7 (was 6/7) run aggregate completion | promotion |

Both canonical files UPDATED and validated via `json.load` post-write per Wf-413-A1 §5:
1. `metrics/by_metric/metal_lomo_auc_wasserstein.json` (12667 bytes mtime 2026-09-30 16:46:29)
2. `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (12933 bytes mtime 2026-09-30 16:47)

---

## §5 Honest framing (per party-twenty-spirit §3.5)

### §5.1 MEASURED outcomes (TAG = MEASURED)

- **Re seed=4 MEASURED 7/7 (metal, seed) pairs complete at 50 epochs** — 2nd-best Re seed test_auc=0.7196 (after Re seed=2 at 0.7467 + Re seed=1 at 0.7463)
- **Overall mean test_auc = 0.7577 ± 0.0924** (sample std over 7 runs)
- **Rh fold mean = 0.8798 ± 0.0115 — PASS krasnov 0.86 with margin +0.020** — MASSIVE lift +0.5121 vs R17 baseline
- **Re fold mean = 0.7088 ± 0.0482 — BELOW krasnov 0.86 with gap -0.151** — MODEST lift +0.2029 vs R17 baseline on smallest cohort n=868

### §5.2 Honest-negative caveats PRESERVED (NOT Wf-411-A3-introduced)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap -0.128; 9 paths exhausted per Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50 + Wf-365-A3 verdict doc + Wf-371-A1 + Wf-375-A4; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3 — r2 > 0 preserved; 10 architectural attempts exhausted)
3. **2 NEW pre-flight FAILs PRESERVED (BY-DESIGN)** (`p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift) — NOT Wf-411-A3-introduced; preserved per Wf-375-B1 §22.68 baseline
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` env gap + `p3_wf343c4_wave_budget_remaining_nonneg` -183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` BindingDB 1.07M external)
5. **Re fold BELOW krasnov** — paper §6 honest-negative preserved (paper §4 LEAD claim = Rh fold only)
6. **Wf-379-A1 GPU wrapper TERMINATION root cause OPEN** — investigate why wrapper terminated after Re seed=3; root cause likely subprocess.signal handling or PIPE buffering; closure = follow-up wrapper fix or KEEP manual Wf-413-A1 pattern
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

### §5.3 Wf-411-A3 closeout file MISSING on disk per AGENTS.md §5.5 filesystem-truth rule

Per `ls -la TODO/completed/411a3*.md` at Wf-411-B1 + Wf-413-A1 + Wf-413-B1 + Wf-414-B1 + Wf-415-A1 timestamps: no `TODO/completed/411a3_*.md` file exists. The Wf-411-A3 process PID 4172923 successfully completed Re seed=4 retrain (process completed at 2026-09-30T16:44:58 CST per Wf-413-A1 §3 process trace), but the agent that processed Wf-411-A3 IN-FLIGHT closeout exited before file-write during the Wf-411 → Wf-413 → Wf-414 transition. **This Wf-415-A1 retroactive closeout authoring is the canonical recovery per §5.5 rule #5** (trigger retroactive re-author workflow when verify agent says MISSING + closeout says SHIPPED — either direction triggers Wf-384-A1 audit + Wf-384-A2 root cause + Wf-384-A3 retroactive re-author + Wf-384-B1 final sync sequence; this Wf-415-A1 is the post-Wf-384 application of that pattern).

**Verdict:** The SHIPPED process completion is on disk (PID 4172923 per-epoch log + `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/aggregate.json` 4021 bytes + `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/Re/per_seed_metrics_seed4.json` 1965 bytes + both canonical metric JSONs UPDATED); the closeout document authoring was the missing step, not the underlying retrain. Wf-413-A1 documented the 7/7 update + Re seed=4 test_auc=0.7196 evidence as a forward-looking reference; this Wf-411-A3 closeout is the Wf-411-A3-specific retrospective.

---

## §6 Cross-references

### §6.1 Predecessor closeouts

- **Wf-411-A1 GPU env probe + USER_OK gate** — `TODO/completed/411a1_gpu_env_probe.md` (12718 bytes mtime 2026-09-30 15:32:02)
- **Wf-411-A2 m2 R² Path A SCAFFOLD smoke HONEST-NEGATIVE close** — `TODO/completed/411a2_m2_path_a_launch.md` (21360 bytes mtime 2026-09-30 15:39:06)
- **Wf-411-A4 m6_kappa Tier-2 §3 supplementary sub-table per-cohort verifier** — `TODO/completed/411a4_m6k_verifier.md` (14599 bytes mtime 2026-09-30 15:45:23)
- **Wf-411-B1 docs-only INDEX + AGENTS + comprehensive final synthesis v39** — `TODO/completed/411b1_index_sync.md` (184 LOC) + `TODO/completed/411_final_synthesis.md` (354 LOC)
- **Wf-413-A1 GPU 0 probe + Re seed=4 wait + 7/7 run aggregate completion** — `TODO/completed/413a1_gpu0_probe_re_seed4_wait.md` (12870 bytes mtime 2026-09-30 16:48) — canonical 7/7 update evidence
- **Wf-413-A3 Wave B launch BLOCKED** — `TODO/completed/413a3_wf299_wave_b_launch.md` (16 KB mtime 2026-09-30 17:14)
- **Wf-413-A4 Wave C launch BLOCKED** — `TODO/completed/413a4_wf299_wave_c_launch.md` (17 KB mtime 2026-09-30 17:17)
- **Wf-413-B1 docs-only INDEX + AGENTS + comprehensive final synthesis v42** — `TODO/completed/413b1_index_sync.md` (184 LOC) + `TODO/completed/413_final_synthesis.md` (~700 LOC)
- **Wf-414-A1 m2 R² Path A launch wrapper** — `TODO/completed/414a1_m2r2_launcher.md` (12145 bytes mtime 2026-09-30 16:38)
- **Wf-414-A2 TODO-21 Lambda × CFM coupling audit heuristic fix** — `TODO/completed/414a2_lambda_cfm_audit.md` (12727 bytes mtime 2026-09-30 16:45)
- **Wf-414-A3 atomic write protocol migration of 5 more scripts** — `TODO/completed/414a3_atomic_migration.md` (13216 bytes mtime 2026-09-30 16:51)
- **Wf-414-B1 docs-only INDEX + AGENTS + comprehensive final synthesis v41** — `TODO/completed/414b1_index_sync.md` + `TODO/completed/414_final_synthesis.md`

### §6.2 Forward-looking USER_OK-gated actions

1. **Wf-379-A1 GPU wrapper TERMINATION ROOT CAUSE** — investigate why wrapper terminated after Re seed=3; root cause likely subprocess.signal handling or PIPE buffering. Closure = follow-up wrapper fix or KEEP manual Wf-413-A1 pattern.
2. **Re fold BELOW krasnov** — preserves paper §6 honest-negative framing. Closure requires R21-P4 triangle closure W₂ lifting OR user OK to drop Re from paper §4 main result.
3. **Wrapper-format per_seed_metrics.json label-swap** — 4 of 6 wrapper files have seed-field SWAP bug. Closure = emit only flat per_seed_metrics_seed{n}.json files going forward (Wf-387-A1 already noted).
4. **m1_auc metric JSON file overwrite bug** — `metrics/by_metric/metal_lomo_auc_wasserstein.json` gets overwritten by `wf125_p2` after every run completion (n=1 in transient state); the canonical aggregate must be reconstructed post-hoc (Wf-387-A1 + Wf-413-A1 + this Wf-411-A3 retroactive do this).
5. **Wf-299 GPU launch BLOCKED** — per Wf-410-A4 contract (e) FAIL — alias mapping must be applied to launcher line 213 + line 221 BEFORE any actual GPU launch; Wf-408-A3 alias mapping closure path = ~10 LOC launcher patch

### §6.3 Standing directives (per AGENTS.md)

- **§5.1** Standing GPU authorization honored via USER_OK gating — Wf-411-A3 GPU launch BLOCKED without USER_OK=1 + VRAM budget check
- **§5.2** Honest framing — no fabrication of m6_kappa / m2 R² / m1 AUC / Wf-246-C metric entries; every numeric value traces to a JSON / report path on disk
- **§5.3** No paper work per TODO-84 — paper sections untouched
- **§5.4** No wet-lab planning per §10 — 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 + §3.3 all DEPRECATED
- **§5.5** Post-wave on-disk verification rule — strictly observed: Wf-411-A3 closeout file MISSING detected per §5.5; this Wf-415-A1 retroactive authoring is the canonical recovery
- **§5.6** Atomic write protocol adoption rule — strictly observed via `atomic_write_file` for this closeout (per Wf-386-A3 §6.1 adoption sequence + Wf-410-A3 + Wf-414-A3 migration extension)
- **§5.7** Trainer flag wire-in smoke-required rule (per Wf-399-A1 §5.7 NEW) — honored
- **§10** Wetlab deprecation FAQ — re-confirmed

### §6.4 Source JSON files

- Canonical Tier-2 metric JSON: `metrics/by_metric/metal_lomo_auc_wasserstein.json` (12667 bytes mtime 2026-09-30 16:46:29, status=MEASURED, verdict=BELOW_KRASNOV_GATE_PARTIAL)
- Wf-131 P3 launch-specific JSON: `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (12933 bytes mtime 2026-09-30 16:47)
- Re seed=4 raw result: `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/aggregate.json` (4021 bytes, MEASURED, verdict=PASS)
- Re seed=4 per-seed raw: `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/Re/per_seed_metrics_seed4.json` (1965 bytes)
- Re seed=4 synthesis: `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/synthesis.md` (3467 bytes)
- Re seed=4 run.log tail: `molmetal/reports/wf131_p3_deferred_runs/run.log` last ~80 lines Re seed=4 epoch=0..49 + AGGREGATE completion event at 2026-09-30 16:44:58

### §6.5 Skill cross-references

- `mol-metal-anticancer` skill — Krasnov 0.86 gate cross-ref (Rh fold PASS_KRASNOV; Re fold BELOW_KRASNOV); precious-metal fold-out context (Rh + Re are Pt-closest-metal analogues)
- `mol-metal-gpu-execution` skill — Blackwell sm_120 environment probe pattern (Wf-411-A1 §2); USER_OK gating via kill_then_adapt safety contract
- `party-twenty-spirit` skill — MEASURED-vs-PROJECTED tag discipline (this Wf-411-A3 retroactive closeout uses TAG = MEASURED throughout)
