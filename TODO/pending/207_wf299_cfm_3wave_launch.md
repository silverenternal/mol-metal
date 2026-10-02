# 207 — Wf-299 CFM 3-Wave Launch (integrate all deep integration + 10 innovation points + GPU retrain)

**Document ID:** TODO-207 / Wf-299-LAUNCH
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §7 — Wf-299 CFM 3-wave is THE canonical CFM-with-innovation-points retrain

---

## §0 TL;DR

**问题**: Wf-299 CFM 3-wave 是 AGENTS.md §5.1 早就 dry-run READY 的 launcher，但被 `decode_ratio=0` blocker 卡住（per Wf-291D + Wf-299）。Wf-379 GPU retrain 用的是**同样的 D-MPNN/chemprop v3 架构**，没有用 Wf-340 + Wf-353 的 10 个 innovation point。

**Solution**: **整合**：
1. 完成 TODO-200 (Wf-340 + Wf-353 deep integration; 10 innovation points wire 到 trainer)
2. 完成 TODO-201 (m2 R² per-cohort features)
3. 完成 TODO-202 (m7 diversity improvements)
4. 解 `decode_ratio=0` blocker（per Wf-291D upstream fix）
5. Launch Wf-299 CFM 3-wave (per AGENTS.md §5.1, USER_OK=1 granted 2026-09-30)

**Wall**: ~9.5 hr GPU + ~10-15 days CPU (deep integration) + ~3-5 days CPU (TODO-201/202)

---

## §1 Background

### Wf-299 3-wave launcher

Per AGENTS.md §5.1:
- Wf-299 CFM 3-wave launcher: `scripts/launch_wf299_cfm_main.sh`
- ~9.5 hr sequential (3 waves)
- dry-run READY; launch wrapper ready
- **Blocked on**: `decode_ratio=0` upstream fix (per Wf-291D + Wf-299)

### What's needed before launch

1. **TODO-200** (Wf-340 + Wf-353 deep integration): 10 innovation points wired to `train_cfm_yuelbond_wasserstein.py` as CLI flags with default ON (post-validation)
2. **TODO-201** (m2 R² per-cohort features): per-cohort-specific feature config
3. **TODO-202** (m7 diversity improvements): top-p + top-k + MMD + repulsion
4. **TODO-203** (m3 Vina docking tuning): exhaustiveness + n_poses
5. **TODO-204** (m15 RMSD multi-template): MM-GBSA rescoring
6. **TODO-205** (m4 SA generation penalty): SA-aware sampling
7. **decode_ratio=0 blocker fix**: per Wf-291D §3 — YuelBond decoder issue

### What Wf-299 3-wave tests

Per Wf-299 brief:
- **Wave 1**: CFM smoke (4-metal small cohort, ~1 hr)
- **Wave 2**: CFM full (5-metal, ~5 hr)
- **Wave 3**: CFM ablation (3-innovation subset, ~3.5 hr)
- Total: ~9.5 hr sequential GPU

### Expected metric lift (post deep integration)

| Metric | Pre-Wf-299 | Post-Wf-299 (PROJECTED) |
|---|---:|---:|
| m1 Ru AUC | 0.7613 | ≥ 0.85 |
| m2 R² platinai_4cell | -0.1251 | ≥ 0 |
| m2 R² hilbert | -0.1268 | ≥ 0 |
| m2 R² metalcytotox_combined | +0.2227 | ≥ 0.30 |
| m7 Diversity | 0.792482 | ≥ 0.81 |
| m14 CFM decode | 1.000 | 1.000 (preserved) |
| m15 RMSD | 0.6578 Å | ≤ 0.55 Å |

---

## §2 Pre-launch checklist

### §2.1 TODO-200 (deep integration, 10-15 days)

- [ ] §2.1 EDM σ-preconditioning: `--edm-preconditioning` flag wired + 1 test PASS
- [ ] §2.2 OT-CFM: `--use-ot-cfm` flag wired + 1 test PASS
- [ ] §2.3 StableVM: full impl + `--stable-vm` flag + 1 test PASS
- [ ] §2.4 SO(3)-averaged FM: `so3_average.py` + `--so3-average` flag + 1 test PASS
- [ ] §2.5 Triton mini-batch OT: `triton_mini_batch_ot()` + `--triton-ot` flag + 1 test PASS
- [ ] §2.6 Shortcut distillation: full impl + `--shortcut-distill` flag + 1 test PASS
- [ ] §2.7 Consistency regularization: full impl + `--consistency-weight` flag + 1 test PASS
- [ ] §2.8 FlowMol unified: license-gated (USER_OK required) + `--flowmol-unified` flag
- [ ] §2.9 log-normal t-sampling: `--log-normal-t` flag + 1 test PASS
- [ ] §2.10 linear lr-warmup: `--lr-warmup-steps` flag + 1 test PASS

### §2.2 TODO-201 + TODO-202 (1-2 days each)

- [ ] m2 R² per-cohort config: 3 tests PASS
- [ ] m7 diversity sampling: 3 tests PASS

### §2.3 TODO-203 + TODO-204 + TODO-205 (1-2 days each)

- [ ] m3 Vina docking tuning: 1 test PASS
- [ ] m15 RMSD multi-template + MM-GBSA: 2 tests PASS
- [ ] m4 SA generation penalty: 1 test PASS

### §2.4 decode_ratio=0 blocker fix

- [ ] Per Wf-291D §3: YuelBond decoder issue → verify upstream fix in Wf-291D+
- [ ] Smoke test: `--smoke 50-step training` gives `decode_ratio > 0`

### §2.5 Pre-launch verification

- [ ] All 12 NEW tests PASS (10 from TODO-200 + 3 from TODO-201 + 3 from TODO-202 + 1 from TODO-203 + 2 from TODO-204 + 1 from TODO-205 = 20 tests total)
- [ ] Pre-flight v3 fresh-run 89/2/3/94 PRESERVED
- [ ] GPU launch wrapper `scripts/launch_wf299_cfm_main.sh` has USER_OK gate
- [ ] Smoke 50-step training with all flags ON: `--edm-preconditioning --use-ot-cfm --stable-vm --so3-average --triton-ot --shortcut-distill --consistency-weight 1.0 --log-normal-t --lr-warmup-steps 1000` gives no errors + decode_ratio > 0

---

## §3 Launch protocol

### §3.1 USER_OK gate

Per AGENTS.md §5.1 + 2026-09-30 user grant: USER_OK=1 granted. 

```bash
if [ "${USER_OK:-0}" != "1" ]; then
    echo "FATAL: USER_OK=1 required for LIVE mode per AGENTS.md §5.1"
    exit 64
fi
```

### §3.2 Wave 1 (CFM smoke)

```bash
cd /home/hugo/codes/mol-metal
USER_OK=1 CUDA_VISIBLE_DEVICES=0 \
  nohup bash scripts/launch_wf299_cfm_main.sh --wave 1 \
  --edm-preconditioning --use-ot-cfm --stable-vm --so3-average \
  --triton-ot --shortcut-distill --consistency-weight 1.0 \
  --log-normal-t --lr-warmup-steps 1000 \
  > molmetal/reports/wf299_wave1/wrapper.log 2>&1 &
```

### §3.3 Wave 2 (CFM full)

```bash
USER_OK=1 CUDA_VISIBLE_DEVICES=0 \
  nohup bash scripts/launch_wf299_cfm_main.sh --wave 2 \
  [same flags] \
  > molmetal/reports/wf299_wave2/wrapper.log 2>&1 &
```

### §3.4 Wave 3 (CFM ablation)

```bash
USER_OK=1 CUDA_VISIBLE_DEVICES=0 \
  nohup bash scripts/launch_wf299_cfm_main.sh --wave 3 \
  [3-innovation subset] \
  > molmetal/reports/wf299_wave3/wrapper.log 2>&1 &
```

### §3.5 Total wall: ~9.5 hr GPU sequential

---

## §4 Post-launch verification

### §4.1 Per-wave MEASURED

- Wave 1: smoke 4-metal small cohort — m1_auc, m2_r2, m7_diversity
- Wave 2: full 5-metal — all 22 metrics
- Wave 3: ablation — per-innovation lift identification

### §4.2 Aggregate metrics

- `metrics/by_metric/m1_auc.json`: update with Wf-299 wave 2 result
- `metrics/by_metric/m2_r2.json`: update with per-cohort results
- `metrics/by_metric/m7_diversity.json`: update with Wf-299 wave 2 result
- `metrics/by_metric/m15_rmsd.json`: update with Wf-299 wave 2 result

### §4.3 Honest framing

- Per Wf-335-SYN §3: PROJECTED 10-15× speedup; MEASURED 3-4×
- Wf-299 actual lift may be less than projected if SCAFFOLDs don't deep-integrate cleanly
- Per Wf-369-A2: m2 small cohorts may still R² < 0 even after Wf-299 (intrinsic ceiling)
- Per Wf-371-A1: m6_kappa intrinsic-bound, won't improve regardless

---

## §5 Cross-references

- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §7 — this is the high-priority pending TODO
- `TODO/pending/200_wf340_353_deep_integration_audit.md` — prerequisite
- `TODO/pending/201_m2_r2_per_cohort_feature_engineering.md` — prerequisite
- `TODO/pending/202_m7_diversity_sampling_improvements.md` — prerequisite
- `TODO/pending/203_m3_vina_docking_param_tuning.md` — prerequisite
- `TODO/pending/204_m15_rmsd_multitemplate_mm_gbsa.md` — prerequisite
- `TODO/pending/205_m4_sa_generation_penalty.md` — prerequisite
- `scripts/launch_wf299_cfm_main.sh` — target launcher
- AGENTS.md §5.1 — standing GPU authorization + USER_OK gate
- Wf-291D — decode_ratio=0 blocker fix

---

**END OF TODO-207** — 2026-09-30, 5 sections, full pre-launch checklist + launch protocol, ~9.5 hr GPU + ~10-15 days CPU