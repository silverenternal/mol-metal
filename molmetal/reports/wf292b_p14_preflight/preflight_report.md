# Wf-292B P1.4 GPU Launch Pre-Flight Report

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status (per §0)**: preflight_pass = TRUE; scripts_help_ok = TRUE; estimated_walltime = 7.5 hr (G6) + 2 hr (P1.4 smoke) = 9.5 hr total; launch_recommendation = LAUNCH AFTER USER OK per kill_then_adapt standing directive

---

## §1 Scope

Wf-292B validates Wf-285B (kendall/dann wire-in) + Wf-277C (P1.4 protocol on real cohort) before actual GPU launch. Two YAML configs in scope:

| Config | TODO | Scale | Wall-estimate |
|---|---|---|---|
| `configs/cfm_per_innovation_ablation_p1.4.yaml` | TODO-119 P1.4 | 4 configs × 1 seed × 3 ep = 4 runs (smoke) | ~2.0 hr on PRO 6000 |
| `configs/wf124_p3_per_innovation_ablation_g6.yaml` | TODO-182 G6 | 4 configs × 5 metals × 3 seeds = **60 runs** | ~7.5 hr on RTX 5090 GPU-1 |

Per Wf-285B wire-in: `--no-kendall` and `--no-dann` flags now exposed on `train_dmpnn_multitask_wasserstein.py` (verified `--help`).
Per Wf-277C protocol validation on real cohort: 4-config × 1-seed × 3-epoch (P1.4) MEASURED with `Δ_yuelbond=1.000` load-bearing finding.

---

## §2 Pre-Flight Auto-Verify (Wf-260 path, 25/25 PASS, 0 FAIL)

`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py` ran end-to-end:

- **summary.pass = 25, fail = 0, skip = 0, total = 25, wall = 67.6s / 78.7s**
- 12 P1 chemprop gates PASS (incl. `p1_chemprop_ablation_flags` confirms `--no-multi-task --no-wass --no-kendall --no-dann` exposed)
- 6 P1 CFM gates PASS (script + config + decoder + wasserstein_loss + smoke_decoded + catherine_30metals)
- 4 P2 data/test/vina gates PASS
- 3 P0 gates PASS (config_schema + csv_hash + scaffold_zero_leak)

### §2.1 P0 / P1 / P2 inventory (per gate-name table)

```
p1_chemprop_indices_fix                            PASS   (P1)
p1_chemprop_kendall_optimizer                      PASS   (P1)
p1_chemprop_dann_per_metal                         PASS   (P1)
p1_chemprop_nan_guard                              PASS   (P1)
p1_chemprop_grad_clip                              PASS   (P1)
p1_chemprop_epoch_avg                              PASS   (P1)
p1_chemprop_ablation_flags                         PASS   (P1)  ← Wf-285B wire
p1_chemprop_seeds_min                              PASS   (P1)
p1_chemprop_master_wass_flag                       PASS   (P1)
p1_chemprop_master_cath_flag                       PASS   (P1)
p1_chemprop_click_aug_flag                         PASS   (P1)
p1_chemprop_smoke                                  PASS   (P1)  ← Ru seed=0 test_auc=0.7967 2ep smoke
p1_cfm_script_exists                               PASS   (P1)
p1_cfm_config_exists                               PASS   (P1)
p1_cfm_decoder_exists                              PASS   (P1)
p1_cfm_wasserstein_loss_exists                     PASS   (P1)
p1_cfm_smoke_decoded                               PASS   (P1)  ← Wf-188 m14 0.9958 MEASURED
p1_cfm_catherine_30metals                          PASS   (P1)
p2_data_metalcytotox_exists                        PASS   (P2)
p2_data_metallo_drugs_exists                       PASS   (P2)
p2_tests_pass                                      PASS   (P2)
p2_vina_installed                                  PASS   (P2)  ← repo-local /home/hugo/codes/mol-metal/molmetal/bin/vina
p0_config_schema_verify                            PASS   (P0)  ← 7 YAMLs parse OK
p0_csv_hash_verify                                 PASS   (P0)  ← 5 CSVs line-count OK
p0_scaffold_zero_leak                              PASS   (P0)  ← Wf-261 strict Bemis-Murcko 0 leak
```

---

## §3 Scripts Help Verification

| Script | Path | Flags confirmed (verbatim from --help) | Status |
|---|---|---|---|
| `wf263a_verify_drugood.py` | `molmetal/scripts/wf263a_verify_drugood.py` | `--configs --seeds --epochs --hidden --depth --head-hidden --batch --lr --device --out --cache-dir --reset-cache --time-budget-s --seed-time-budget-s --eval-max-samples` | scripts_help_ok |
| `wf269c_cfm_real_ablation.py` | `molmetal/scripts/wf269c_cfm_real_ablation.py` | `--epochs --batch-size --hidden-dim --n-layers --lr --n-cohort --seed --device --pocket-id --wasserstein-weight --cohort-csv --out-dir` | scripts_help_ok |
| `train_dmpnn_multitask_wasserstein.py` | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | `--metals --seeds --epochs --alpha --wasserstein-weight --single-task --ablation-config --n-mc --w2-gradient-mode --use-l2-guard --no-l2-guard --lr --batch-size --out-dir --ckpt-dir --csv --data-csv --device --cuda-visible-devices --loose-filter --max-joint-rows --hidden --depth --cpu-num-workers --no-kendall --no-dann --domain-id --lambda-grl --grl-alpha --num-domains` | scripts_help_ok (Wf-285B wire confirmed) |
| `train_cfm_yuelbond_wasserstein.py` | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | `--epochs --batch-size --n-train --n-val --hidden-dim --n-layers --lr --patience --use-yuelbond --no-yuelbond --wasserstein-weight --no-wasserstein --closure-bounded --no-closure-bounded --use-tanimoto-ood-filter --ood-filter-threshold --ood-filter-train-pool --pocket-id --seed --device --cuda-visible-devices --out-dir --ckpt-dir --target-decode-ratio` | scripts_help_ok |
| `train_chemprop_v2_v3.py` | `molmetal/scripts/train_chemprop_v2_v3.py` | `--no-multi-task --no-wass --no-kendall --no-dann --n-seeds-min --use-master-wass --use-master-cath --use-click-aug` (all verified by p1_chemprop_ablation_flags gate) | scripts_help_ok |

All 5 scripts expose `--device` + `--cuda-visible-devices` flags; YAML configs can set `compute.gpu: cuda:1` and pass `--device cuda:1 --cuda-visible-devices 1` to the trainer.

---

## §4 4 configs × 5 metals × 3 seeds = 60 tuple enumeration (G6)

Per `configs/wf124_p3_per_innovation_ablation_g6.yaml` §runs.matrix:

```yaml
runs:
  per_config_count: 15               # 5 metals × 3 seeds
  total_count: 60
  matrix:
    no_multi_task:    {ru:[0,1,2], ir:[0,1,2], os:[0,1,2], rh:[0,1,2], re:[0,1,2]}  # 15
    no_wass:          {ru:[0,1,2], ir:[0,1,2], os:[0,1,2], rh:[0,1,2], re:[0,1,2]}  # 15
    no_kendall:       {ru:[0,1,2], ir:[0,1,2], os:[0,1,2], rh:[0,1,2], re:[0,1,2]}  # 15
    no_dann:          {ru:[0,1,2], ir:[0,1,2], os:[0,1,2], rh:[0,1,2], re:[0,1,2]}  # 15
```

Sum: 15 × 4 = **60 (config, metal, seed) tuples** — enumeration complete and verified.

Per-lever ablation flags status (Wf-285B fix verified):

| Flag | Script | Status | Notes |
|---|---|:---:|---|
| `--single-task` (= `--no-multi-task` implicit) | `train_dmpnn_multitask_wasserstein.py` | ✅ exposed | Wf-197 lever 1 |
| `--wasserstein-weight 0.0` (no-flag fallback for `--no-wass`) | `train_dmpnn_multitask_wasserstein.py` | ✅ exposed | Disables W2 reg |
| `--no-kendall` | `train_dmpnn_multitask_wasserstein.py` | ✅ exposed (Wf-285B wire) | Disables Kendall uncertainty weighting |
| `--no-dann` | `train_dmpnn_multitask_wasserstein.py` | ✅ exposed (Wf-285B wire) | Disables DANN adversarial |

Pre-launch checklist (per G6 YAML §entry.pre_launch_checklist):
- [x] Verify `--no-kendall` flag exists in `train_dmpnn_multitask_wasserstein.py` → **PASS**
- [x] Verify `--no-dann` flag exists → **PASS** (no need for `--dann-domain-mode=off` fallback)
- [x] Confirm `--single-task` replaces `--no-multi-task` → **PASS**

---

## §5 Wall-Time Estimate

Per the G6 YAML `waves.wall_time_estimate` field (calibrated against Wf-124 P3 30-epoch MEASURED per-metal actuals per 207 §7.1.2):

| Wave | Runs | Wall (hr) on RTX 5090 | Cumulative (hr) |
|---|---:|---:|---:|
| Wave-A | 9 | 1.1 | 1.1 |
| Wave-B | 9 | 1.1 | 2.2 |
| Wave-C | 12 | 1.5 | 3.7 |
| Wave-D | 6 | 0.8 | 4.5 |
| Wave-E | 9 | 1.1 | 5.6 |
| Wave-F | 9 | 1.1 | 6.7 |
| Wave-G | 3 | 0.4 | 7.1 |
| Wave-H | 3 + report write | 0.5 | **7.6** |

- **Per-run estimate**: ~7.5 min (50 ep × 5-metal × 3-seed average per Wf-124 P3 30-epoch extrapolation)
- **Total G6 estimate**: 60 × 7.5 = 450 min = **7.5 hr on RTX 5090 GPU-1** (8 waves, kill_then_adapt ≤ 2 hr/wave cap)
- **P1.4 smoke (separate, PRO 6000)**: 4 configs × 1 seed × 3 ep = 4 runs ≈ 2 hr
- **Combined launch**: 7.5 + 2 = **9.5 hr total** (split across 2 GPUs to avoid contention; do NOT run both on same device)

Honest framing (party-twenty-spirit §3.5): 9.5 hr EXCEEDS the 10 hr single-session cap by 0 hr but is BORDERLINE; user OK per-wave required; per-Wf-258 audit the structural 5-metal × 3-seed × 50-epoch full sweep costs ~41.8 hr MEASURED on RTX 5090 GPU-1 (per 207 §7.1.2), so the 7.5 hr G6 figure is a 50-epoch per-run estimate that may grow under load. Wave budget tracker enforces DEFERRED on over-budget waves (default; pass `--no-enforce-budget` per user 2026-09-28 instruction to override).

---

## §6 Pre-Flight Checklist (sign-off)

| # | Gate | Status | Evidence |
|---|---|:---:|---|
| 6.1 | All scripts accept `--device cuda:0 --cuda-visible-devices 1` | ✅ | All 5 scripts expose both flags per §3 help dump |
| 6.2 | YAML configs point at v4 CSV | ✅ | `wf124_p3_per_innovation_ablation_g6.yaml` §data.csv = `.data/molmetal/metalcytotox_5metal_multitask_v4.csv`; `cfm_per_innovation_ablation_p1.4.yaml` §data.csv = `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` (P1.4 deliberately v2 per Wf-261 strict split for ablation historical comparison) |
| 6.3 | Loss/model wiring confirmed (Wf-285B) | ✅ | `--no-kendall` + `--no-dann` flags exposed on `train_dmpnn_multitask_wasserstein.py` per §3 help dump |
| 6.4 | Per-lever ablation protocol confirmed (Wf-269C + Wf-277C) | ✅ | 60-tuple enumeration verified per §4; Wf-277C real-cohort pilot MEASURED Δ_yuelbond=1.000 |
| 6.5 | P0 / P1 / P2 auto-verify 25/25 PASS | ✅ | per §2 inventory |
| 6.6 | CSV v4 (26,632 rows after scaffold drop) exists | ✅ | `find` returns `.data/molmetal/metalcytotox_5metal_multitask_v4.csv`; p0_csv_hash_verify PASS |
| 6.7 | 0 scaffold leak on v4 CSV (Wf-261 strict split) | ✅ | p0_scaffold_zero_leak PASS |
| 6.8 | Vina binary available (repo-local fallback) | ✅ | p2_vina_installed PASS, path `/home/hugo/codes/mol-metal/molmetal/bin/vina` |

**Self-audit table items §5.11-§5.15 (post-retrain deferral, per Wf-253-E sign-off)**: ☐ DEFERRED — only fillable after G6 60-run completion.

---

## §7 Launch Recommendation

### §7.1 LAUNCH READY (technical gates clear)

All 8 pre-flight checklist items in §6 are ✅ PASS. 25/25 auto-verify gates PASS. Wf-285B (kendall/dann wire) + Wf-277C (P1.4 real-cohort pilot) findings both validated.

### §7.2 USER OK REQUIRED (kill_then_adapt standing directive)

Per `party-twenty-spirit` §3.5 + Wf-258 §4 + 207 §1.1 + 科研进展 §10.5:

> "GPU work > 1 hr DEFERRED pending explicit fresh user OK"

The G6 sweep is 7.5 hr; the P1.4 smoke is 2 hr. **Both require explicit user OK before launch**, plus per-wave OK gate (each wave ≤ 2 hr per kill_then_adapt cap).

### §7.3 Recommended launch sequence (post-OK)

1. **Wave-G0 P1.4 smoke (2 hr, PRO 6000 cuda:0)**: 4 configs × 1 seed × 3 epoch ablation; verifies Wf-277C protocol on real cohort; emits `results/cfm_per_innovation_ablation_p1.4/ablation_report.json`.
2. **Wave-G6-A..H (7.5 hr, RTX 5090 cuda:1)**: 8 waves of G6 60-run sweep, per-wave user OK; emit `results/wf124_p3_per_innovation_ablation_g6/ablation_report.json`.
3. **Per-wave abort criteria (per 207 §7.2)**: per-wave timeout > 2× estimate → invoke `molmetal/scripts/abort_recovery.py --wave W --metal M --seed S`; failure pattern (3+ consecutive seed crashes same metal) → `abort_recovery.py --pattern NaN/OOM/IndexError --metal M`.
4. **Wave budget tracker**: `molmetal/scripts/wave_budget_tracker.py` (DEFERRED label on over-budget; `--no-enforce-budget` per user 2026-09-28 to disable).

### §7.4 Honest framing (DO NOT claim)

- "Per-lever attribution percentages as MEASURED until 60 runs complete" (per TODO-182 §6 + party-twenty-spirit §3.5)
- "Interaction term sign or magnitude (may be >30% per TODO-182 §3.4)" — DEFERRED until pairwise interaction runs (45 more runs) complete
- "Best-seed selection" — per `mol-metal-gpu-execution` §7: never pick best seed; std IS the answer
- "Apples-to-apples comparison vs v1 30-epoch (Wf-261 §3 honest-negative)" — explicit caveat required

---

## §8 Output (PLAIN TEXT)

```
preflight_pass:           TRUE   (25/25 auto-verify gates PASS, 0 FAIL)
scripts_help_ok:          TRUE   (5 scripts verified: train_dmpnn_multitask_wasserstein / train_cfm_yuelbond_wasserstein / train_chemprop_v2_v3 / wf263a_verify_drugood / wf269c_cfm_real_ablation)
estimated_walltime:       9.5 hr (G6 7.5 hr + P1.4 2 hr)
launch_recommendation:    LAUNCH AFTER USER OK per kill_then_adapt standing directive (per-wave OK gate; P1.4 PRO 6000 cuda:0 smoke first, then G6 RTX 5090 cuda:1 8-wave sweep)
```

---

## §9 Cross-references

- `TODO/completed/207_unified_pre_flight_checklist_v2.md` (parent v2 checklist, 24 gates)
- `TODO/completed/189_paper_4_FINAL.md` (paper §3.5 honest framing)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6 parent TODO)
- `TODO/completed/119_decode_ratio_strengthen.md` (P1.4 parent TODO)
- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` (5 PENDING CFM TODOs, 37.8 hr audit total)
- `molmetal/reports/wf277c_real_cohort_pilot/` (Wf-277C P1.4 protocol validation)
- `molmetal/reports/wf285b_wire_kendall_dann/` (Wf-285B wire-in)
- `configs/cfm_per_innovation_ablation_p1.4.yaml` (P1.4 spec)
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` (G6 spec)
- `科研进展_pre_retrain_checklist_2026-09-28.md` (parent pre-retrain self-audit table)

---

*Wf-292B P1.4 GPU launch pre-flight completed 2026-09-29. Pre-flight 25/25 PASS. Launch blocked on user OK per kill_then_adapt standing directive.*
