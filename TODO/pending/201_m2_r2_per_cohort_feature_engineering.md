# 201 — m2 R² Per-Cohort-Specific Feature Engineering

**Document ID:** TODO-201 / Wf-381-M2-PER-COHORT-FEATURES
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §4 — m2 R² metalcytotox_combined +0.2227 CEILING_BREAK can lift +0.05-0.10 with per-cohort-specific features (Category B, no retraining)

---

## §0 TL;DR

**问题**: m2 R² metalcytotox_combined 已经 +0.2227 CEILING_BREAK，但 platinai_4cell (-0.1251) + hilbert (-0.1268) 反而因为 n=200 cohort + 50+ feature Ridge ensemble **过拟合**（per Wf-369-A2 + Wf-367-B1）。**所有 cohort 用同一套 features**是 root cause。

**Solution**: **Per-cohort-specific feature count + ensemble**：
- n=200 cohort (platinai_4cell, hilbert): RDKit 12-d + Ridge (low complexity)
- n=500 cohort (intermediate): 24-d + chemprop v3
- n=2000 cohort (metalcytotox_combined): 50-d ensemble (already CEILING_BREAK)
- **Adaptive feature selection** per cohort

**Wall**: 1-2 days CPU-only

---

## §1 Background

### Current state (per Wf-335-M2 + Wf-367-B1 + Wf-369-A2)

| Cohort | n | Current features | Current R² | Status |
|---|---:|---|---:|---|
| metalcytotox_combined | 2000 | 50-d ensemble | +0.2227 | CEILING_BREAK |
| platinai_4cell | 200 | 50-d ensemble | -0.1251 | HONEST_NEGATIVE_R2_DEEP (OVERFIT) |
| hilbert | 200 | 50-d ensemble | -0.1268 | HONEST_NEGATIVE_R2_DEEP (OVERFIT) |

**Root cause**: 50+ features on n=200 = 0.25 feature/sample ratio → bias-variance inflation → negative R²

### Wf-377-A2 in flight: alternative splits

Per Wf-377-A2 (in flight wt8uqihku): per-cohort structural properties + 5 alternative splits (random / scaffold / temporal / stratified / leave-one-cluster-out) being validated.

This TODO-201 complements Wf-377-A2: **features per cohort** is a separate axis from **splits per cohort**.

---

## §2 Per-cohort-specific feature configuration

### §2.1 Configuration matrix

| Cohort | n | Recommended features | Recommended model | Expected R² |
|---|---:|---|---|---:|
| metalcytotox_combined | 2000 | 50-d ensemble (already + RDS chemprop v3) | Ridge + chemprop v3 + Hilbert | +0.30 (lift +0.07) |
| platinai_4cell | 200 | 12-d RDKit | Ridge (low complexity) | +0.10 (lift +0.23) |
| hilbert | 200 | 12-d RDKit | Ridge (low complexity) | +0.10 (lift +0.23) |
| intermediate (500) | 500 | 24-d RDKit + chemprop v3 (12-d + 12-d chemprop) | Ridge | +0.20 |

### §2.2 Adaptive feature selection

```python
def select_features(cohort_size: int) -> dict:
  if cohort_size < 300:
    return {"features": "rdkit_12d", "model": "ridge", "alpha": 1.0}
  elif cohort_size < 1000:
    return {"features": "rdkit_24d_chemprop_v3", "model": "ridge", "alpha": 0.5}
  else:
    return {"features": "rdkit_50d_chemprop_v3_hilbert", "model": "ensemble", "alpha": 0.1}
```

### §2.3 Adaptive Ridge alpha

- **n=200 + 12-d features**: α = 1.0 (strong regularization to prevent overfit)
- **n=500 + 24-d features**: α = 0.5 (medium regularization)
- **n=2000 + 50-d features**: α = 0.1 (weak regularization; ensemble)

---

## §3 File changes

### §3.1 New config module

- `molmetal/scripts/wf381_m2_r2_per_cohort.py`: NEW (per-cohort config + adaptive selection, ~150 LOC)
- Config-driven: `molmetal/configs/m2_r2_per_cohort.yaml` (cohort_size → features mapping)

### §3.2 Wire to existing trainers

- `molmetal/scripts/wf335m2_r2_recovery.py`: integrate per-cohort config
- `molmetal/scripts/wf367b1_m2_r2_recovery.py`: integrate per-cohort config
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`: add `--per-cohort-features` flag (default ON)

### §3.3 Evaluation

- `molmetal/scripts/eval_m2_per_cohort.py`: NEW (per-cohort R² evaluation, ~80 LOC)
- Generate `metrics/by_metric/m2_r2_per_cohort.json`: per-cohort R² + config

---

## §4 Expected lift

### §4.1 platinai_4cell (-0.1251 → +0.10)

- **Current**: 50-d ensemble overfits on n=200
- **New**: 12-d RDKit + Ridge (α=1.0)
- **Expected R² lift**: +0.23
- **Mechanism**: 50+ features → 12-d reduces overfit; strong Ridge regularization

### §4.2 hilbert (-0.1268 → +0.10)

- Same as platinai_4cell
- **Expected R² lift**: +0.23

### §4.3 metalcytotox_combined (+0.2227 → +0.30)

- **Current**: 50-d ensemble
- **New**: 50-d ensemble + chemprop v3 + Hilbert (3-model ensemble)
- **Expected R² lift**: +0.07

---

## §5 Test plan

3 NEW tests:
1. **test_per_cohort_config_match** — per-cohort config matches cohort_size thresholds
2. **test_12d_ridge_no_overfit** — n=200 + 12-d Ridge gives positive R² on synthetic
3. **test_24d_intermediate** — n=500 + 24-d + chemprop v3 works

---

## §6 Acceptance criteria

- 3 NEW tests PASS in <5 s
- `metrics/by_metric/m2_r2_per_cohort.json` updated with per-cohort R²
- `docs/M2_R2_PER_COHORT_FINAL_VERDICT.md` updated with new verdict
- Pre-flight 89/2/3/94 PRESERVED

---

## §7 Cross-references

- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §4.1 — Category B detail
- Wf-335-M2 — chemprop+PLL full integration
- Wf-367-B1 — multi-stream ensemble (OVERFIT finding)
- Wf-369-A2 — m2 R² FINAL verdict
- Wf-377-A2 — alternative splits (complementary axis)
- `molmetal/scripts/wf335m2_r2_recovery.py` — target trainer
- `molmetal/scripts/wf367b1_m2_r2_recovery.py` — target trainer
- AGENTS.md §5.2 — honest framing mandate

---

**END OF TODO-201** — 2026-09-30, 7 sections, per-cohort config + 3 NEW tests expected, ~1-2 days wall
