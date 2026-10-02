# Wf-295D: TODO-119 P1.4 + TODO-182 G6 Launch Orchestration Plan

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Parent**: Wf-295 flow-up from prior workflow insight
**Refs**: Wf-292B pre-flight (25/25 PASS), Wf-285B kendall/dann wire, Wf-293A/B yaml nits, 207 §7.2 abort criteria, kill_then_adapt standing directive

---

## §0 Status snapshot

| Gate | Status | Source |
|---|---|---|
| Pre-flight auto-verify | **25/25 PASS, 0 FAIL** | re-ran `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py` (PASS=25, FAIL=0) |
| G5 wrapper alive | **PID 1015907 RUNNING** (27:46 CPU, python) | `ps -p 1015907` |
| `--no-enforce-budget` wired in G6 yaml | **CONFIRMED** at line 252 (`configs/wf124_p3_per_innovation_ablation_g6.yaml`) | grep |
| `--no-enforce-budget` wired in G5 yaml | **CONFIRMED** at line 222 (`configs/wf124_p3_50epoch_full_sweep_g5.yaml`) | grep |
| User "kill then adapt" OK gate | **PENDING** (per §1.1 of 207 — user must grant before any GPU launch) | TODO/INDEX.md |

**launch_recommendation**: LAUNCH P1.4 SMOKE FIRST after user OK; G6 follows sequentially (not concurrent with P1.4 due to shared venv / 13 h budget).

---

## §1 Launch sequence (3 waves)

### Wave-A: P1.4 smoke (TODO-119) — 4 configs × 1 seed × 3 epochs

| Property | Value |
|---|---|
| Config | `configs/cfm_per_innovation_ablation_p1.4.yaml` |
| Scale | 4 ablation configs × 1 seed × 3 epochs = 4 runs |
| GPU | PRO 6000 cuda:0 (GPU-0) |
| Wall-est | **2.0 hr** (4 sub-waves × 0.5 hr each per yaml §WAVES) |
| Sub-waves | (a) baseline, (b) YuelBond only, (c) W2 only, (d) Catherine only |
| Output | `results/cfm_per_innovation_ablation_p1.4/{config_name}/` |
| Purpose | Per-lever attribution smoke (party-twenty-spirit §3.5 hard requirement) |

**Sub-wave decomposition** (per yaml):
- wave_a baseline: 0.5 hr on PRO 6000 cuda:0
- wave_b YuelBond only: 0.5 hr
- wave_c W2 only: 0.5 hr
- wave_d Catherine only: 0.5 hr
- Total: 2.0 hr (verified by `wall_time_estimate` per wave)

**Why P1.4 first**:
- Cheapest path to validate `Δ_yuelbond=1.000` finding on real cohort (Wf-277C pre-validated protocol)
- 13h G6 sweep needs warm CFM+chemprop stack; smoke catches infra regressions before 7.5 h G6 run
- Different GPU (cuda:0 PRO 6000) — no contention with G6 on cuda:1 RTX 5090

### Wave-B: G6 C12 no_wass smoke (TODO-182) — 1 config × 1 metal × 1 seed

| Property | Value |
|---|---|
| Config | `configs/wf124_p3_per_innovation_ablation_g6.yaml` |
| Scale | 1 ablation config (no_wass) × 1 metal (Ru) × 1 seed (0) = 1 run |
| GPU | RTX 5090 cuda:1 (GPU-1) |
| Wall-est | **10 min** (~30 min / 3 seeds ÷ 3 = ~10 min for 1 seed) |
| Output | `results/wf124_p3_per_innovation_ablation_g6/no_wass/Ru/per_seed_metrics_seed0.json` |
| Purpose | Cheapest G6 MEASURED wave — validate `--wasserstein-weight 0.0` flag path before 60-run sweep |

**Why C12 no_wass smoke first**:
- no_wass is config index 2 in yaml `ablation_configs` block; smoke validates W2-off path before 60 runs commit
- Ru is heaviest metal (28.9 hr / 3 seeds MEASURED per §7.1.2); if Ru seed=0 hits NaN/AUC regression, abort G6 sweep before burning 7.5 hr
- Different GPU than P1.4 — no contention

### Wave-C: G6 full sweep (TODO-182) — 4 configs × 5 metals × 3 seeds = 60 runs

| Property | Value |
|---|---|
| Config | `configs/wf124_p3_per_innovation_ablation_g6.yaml` |
| Scale | 60 runs total (verified-skip + actual = 30 + 30 per task spec) |
| GPU | RTX 5090 cuda:1 (GPU-1) |
| Wall-est | **7.5 hr** on RTX 5090 GPU-1 (per Wf-292B §1 MEASURED + §7.1.2 per-metal wall-times) |
| Wave structure | 8 waves × ~1 hr each (per yaml §WAVES: waves a..h) |
| Output | `results/wf124_p3_per_innovation_ablation_g6/{config}/{metal}/per_seed_metrics_seed{0,1,2}.json` |
| Purpose | Full per-lever attribution ablation for paper §3 algorithm-description requirement |

**Per-metal MEASURED wall (per §7.1.2, 3-seed aggregate on RTX 5090)**:
| Metal | 3-seed wall (hr) | Note |
|---|---:|---|
| Ru | 28.9 | heaviest cohort; ~463 s/seed × 3 |
| Ir | 7.7 | medium cohort |
| Os | 1.9 | collapse-guarded (§7.2.4) |
| Rh | 1.7 | LOMO bottleneck |
| Re | (projected ~2.0) | extrapolated from Rh |

**Wave decomposition** (per yaml §WAVES, 60 runs / 8 waves):
| Wave | Description | Runs | Wall (hr) |
|---|---|---:|---:|
| a | no_multi_task Ru/Ir/Os | 9 | 1.1 |
| b | no_multi_task Rh/Re + no_wass Ru | 9 | 1.1 |
| c | no_wass Ir/Os/Rh/Re | 12 | 1.5 |
| d | no_kendall Ru/Ir | 6 | 0.8 |
| e | no_kendall Os/Rh/Re | 9 | 1.1 |
| f | no_dann Ru/Ir/Os | 9 | 1.1 |
| g | no_dann Rh | 3 | 0.4 |
| h | no_dann Re + ablation_report write | 3 | 0.5 |
| **Total** | | **60** | **7.6 hr** |

---

## §2 Sequencing constraints

| Constraint | Rationale |
|---|---|
| P1.4 BEFORE G6 | CFM stack warm-up + 2 hr cheap smoke catches infra issues before 7.5 hr G6 run |
| G6 C12 smoke BEFORE G6 sweep | Validates `--wasserstein-weight 0.0` flag before 60-run commit |
| Sequential, not concurrent | Shared venv + 13 h budget + kill_then_adapt 2 hr/wave cap |
| G5 wrapper alive PID 1015907 | Optional monitor for G5 background sweep (do not interfere) |

---

## §3 Per-wave abort criteria (per 207 §7.2)

Applied to every wave. Detection + recovery script path cross-ref to `molmetal/scripts/abort_recovery.py`:

| # | Pattern | Detection | Recovery |
|---:|---|---|---|
| 7.2.1 | **Per-wave timeout**: wave exceeds 2.0× projected wall | ckpt write older than 2× projected wall-time | `molmetal/scripts/abort_recovery.py --wave W --metal M --seed S` |
| 7.2.2 | **Failure pattern**: 3+ consecutive seeds crash on same metal (NaN AUC, OOM, IndexError) | `grep "NaN\|OOM\|IndexError" molmetal/reports/wf*/log` | `molmetal/scripts/abort_recovery.py --pattern NaN --metal M` |
| 7.2.3 | **GPU contention regression**: util drops < 70% for 5+ min | `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 30 -c 10` | Stop other user's process (out of scope); fall back to GPU 0 |
| 7.2.4 | **Os collapse pattern**: Os AUC = 0.5000 (structural, per §1.2.1 + Wf-244-B) | per-seed JSON `test_auc < 0.51` | Run `--single-metal Os --no-multi-task` (workaround per §1.2.1) |
| 7.2.5 | **Decode_ratio collapse**: Wf-124 P2 went-to-zero | per_epoch_metrics.json min < 0.05 | `molmetal/scripts/abort_recovery.py --cfm-collapse --restore-cf124p2-ckpt` |
| 7.2.6 | **m7 Tanimoto drop below 0.04** (per Wf-181 m7 honest-negative baseline) | m7 JSON `tanimoto_diversity < 0.04` | DEFER m7 claim; report honest-gap in paper §6 |
| 7.2.7 | **CHECKPOINT_LOADED_FORWARD_DEFERRED**: smoke loads ckpt but does not retrain (per Wf-252 §F3) | forward-pass only, no `epoch_*` keys | Mark state as `CHECKPOINT_LOADED_FORWARD_DEFERRED` (honest-negative, NOT FAIL) |
| 7.2.8 | **Partial pilot honesty**: pilot reports n_completed < n_total (per Wf-252 F4 5/12 partial) | check `n_completed < n_total` | Never cite as `total/total`; report `n_completed/total partial` |

---

## §4 Per-wave OK gate (per kill_then_adapt)

A wave is `OK` only when ALL of the following are true:

| # | OK gate | Evidence path |
|---:|---|---|
| 4.1 | **All runs in wave produced per-seed JSON** | `ls -la results/{wave_dir}/{config}/{metal}/per_seed_metrics_seed{0,1,2}.json` |
| 4.2 | **No NaN / OOM / IndexError in logs** | `grep -E "NaN\|OOM\|IndexError" molmetal/reports/wf295d_*/log/*.log` returns empty |
| 4.3 | **AUC ≥ 0.51 (no collapse)** | `python -c "import json,glob; [json.load(open(f))['test_auc'] for f in glob.glob('.../per_seed_metrics_seed*.json') if json.load(open(f))['test_auc'] < 0.51]"` returns empty |
| 4.4 | **Wall time ≤ 2.0× projected** | `python -c "import json,glob; [json.load(open(f))['elapsed_sec'] for f in ...]"` — no run > 2× its projected |
| 4.5 | **GPU util ≥ 70%** during wave wall (not idle) | `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 30 -c 20` average |
| 4.6 | **Wave budget tracker PASS** | `molmetal/scripts/wave_budget_tracker.py --wave W` returns `OK` (not `DEFERRED`) |
| 4.7 | **Per-wave sub-checklist PASS** | `molmetal/scripts/per_wave_checklist.py --wave W` returns all gates PASS |
| 4.8 | **User OK required + granted** | `per_wave_user_ok_required: true` per yaml; if any sub-pattern triggers, abort |

**Per-wave escalation**:
- 1 run FAIL → mark partial, continue wave (per §7.2.8 honest-negative framing)
- 2 runs FAIL same pattern → halt wave, invoke §7.2.2 abort recovery
- 3 runs FAIL same pattern → ABORT entire sweep; report to user; do NOT auto-continue

---

## §5 Pre-launch checklist (P0 BLOCKER gates from 207 §1)

| # | Gate | Status | Source |
|---:|---|:---:|---|
| 1.1 | User "kill then adapt" OK gate lifted | **PENDING** | User must grant before Wave-A |
| 1.2 | GPU state live verify: cuda:0 + cuda:1 ≤ 5 GB used | **VERIFY at launch** | `nvidia-smi` |
| 1.3 | YAML schema + hash verify (all 7 active YAMLs) | **PASS** | pre_flight_check.py (25/25) |
| 1.4 | Data CSV hash + scaffold zero leak | **PASS** | pre_flight_check.py (25/25) |

---

## §6 Post-launch artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Wave-A P1.4 master log | `results/cfm_per_innovation_ablation_p1.4/master.log` | 4 sub-wave sequencing |
| Wave-A ablation report | `results/cfm_per_innovation_ablation_p1.4/ablation_report.json` | Per-lever attribution |
| Wave-B C12 smoke JSON | `results/wf124_p3_per_innovation_ablation_g6/no_wass/Ru/per_seed_metrics_seed0.json` | Validates `--wasserstein-weight 0.0` path |
| Wave-C G6 master log | `results/wf124_p3_per_innovation_ablation_g6/master.log` | 8-wave sequencing |
| Wave-C ablation report | `results/wf124_p3_per_innovation_ablation_g6/ablation_report.json` | Full per-lever attribution (60 runs) |
| Per-wave checklist JSON | `molmetal/reports/wf295d_launch_orchestration/wave_{a,b,c}_checklist.json` | Per-wave OK gate evidence |
| Per-wave budget tracker | `molmetal/reports/wf295d_launch_orchestration/wave_{a,b,c}_budget.json` | Wall-time actuals vs projected |

---

## §7 Honest framing (party-twenty-spirit §3.5)

- **G6 60-run MEASURED total** = 7.5 hr; **P1.4 MEASURED total** = 2.0 hr; **combined** = 9.5 hr
- **13 h user budget** with `--no-enforce-budget` opt-out (per §7.1.1) — budget is tracked but not enforced
- `target_test_auc: 0.86` in G6 yaml is **PROJECTED, NOT MEASURED** (per honest_framing block)
- Per-lever attribution is paper §3 algorithm-description hard requirement, not optimization
- Wave-A P1.4 findings (`Δ_yuelbond=1.000` etc.) are MEASURED, but interaction term magnitude is NOT (TODO-182 §3.4 hypothesis: may exceed 30%)
- Os AUC = 0.5000 collapse is STRUCTURAL and known (§1.2.1 / §7.2.4); workaround in place

---

## §8 References

- `molmetal/reports/wf292b_p14_preflight/preflight_report.md` — 25/25 PASS evidence
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — §1 P0 / §7.2 abort criteria / §7.3 recovery contract
- `configs/cfm_per_innovation_ablation_p1.4.yaml` — Wave-A config (4 ablation configs × 1 seed × 3 ep)
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` — Wave-B/C config (4 ablation configs × 5 metals × 3 seeds)
- `molmetal/scripts/abort_recovery.py` — §7.2 recovery script (Wf-262 SHIPPED)
- `molmetal/scripts/wave_budget_tracker.py` — §7.1 budget tracker with `--no-enforce-budget` (Wf-262 SHIPPED)
- `molmetal/scripts/per_wave_checklist.py` — per-wave OK gate validator (Wf-262 SHIPPED)
- `molmetal/scripts/pre_flight_check.py` — 25/25 PASS auto-verify (Wf-262 SHIPPED)

---

**END Wf-295D plan** — ready for user OK gate (207 §1.1) before Wave-A launch.