# TODO 182 — Wf-124 P3 Gap: Per-innovation ablation (Wf-119 P1.4 deferred)

**Date**: 2026-09-27 (Wf-124 P3 gap analysis per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 P1 — IN-FLIGHT (Wf-124 P3 ran all 3 innovations simultaneously; per-innovation ablation NOT done)
**Parent**: `molmetal/reports/wf124_p3_multitask/synthesis.md` §3 honest framing + `TODO/pending/119_decode_ratio_strengthen.md` §1.4
**Sibling**: `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4) + `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5)

---

## §1 Gap description

**Wf-124 P3 §3 honest framing** explicitly states:

> "we run all 3 innovations simultaneously. **Per-innovation ablation (W₂ only, closure only, YuelBond only) is a TODO-119 P1.4 follow-up to fully attribute the lift to each lever.**"

The Wf-124 P3 multi-task + W₂ recipe is run with all components enabled by default. The +0.049 absolute test_auc lift (0.7123 → 0.7613, Ru seed=0, 30 epochs) is therefore an **aggregate lift** — it does NOT attribute the lift to:

1. **Multi-task D-MPNN** (shared encoder + per-metal head) — TODO-120 P1.2 main lever
2. **Wasserstein pIC50 regularization** (Wf-123 P1 M1 module) — TODO-120 P1.2 secondary lever
3. **L2 guard** (grad-norm clip at 5.0) — stability lever
4. **Joint 5-metal cohort** (vs single-metal Ru-only) — data lever

Without the per-innovation ablation, paper §3 cannot honestly attribute the lift to each lever. Per `party-twenty-spirit` §3.5 honest framing, this is a hard requirement for §3 algorithm description — readers must know which lever did what.

---

## §2 Current MEASURED state

| Configuration | test_auc (Ru seed=0, 30 epochs) | W₂ final | Status |
|---|---:|---:|---|
| **Baseline** (single-task D-MPNN, no W₂, no L2 guard, Ru-only) | **0.7123** | n/a | MEASURED (Wf-120 P2 baseline) |
| **Multi-task only** (no W₂, no L2 guard, 5-metal) | — | n/a | ⏳ NOT MEASURED |
| **W₂ only** (single-task D-MPNN + W₂ + L2 guard, Ru-only) | — | — | ⏳ NOT MEASURED |
| **L2 guard only** (single-task D-MPNN + L2 guard, no W₂, Ru-only) | — | n/a | ⏳ NOT MEASURED |
| **Multi-task + W₂ + L2 guard** (Wf-124 P3 default) | **0.7613** | 0.3906 | ✅ MEASURED (Ru seed=0) |
| **Krasnov 2026 target** | 0.86 | — | PROJECTED |

**Gap**: 4 ablation configurations × 5 metals × 3 seeds = 60 runs NOT yet executed.

**Wall-time estimate**:
- Per (config, metal, seed): ~7.5 min on RTX 5090 (30 epochs)
- Total: 60 × 7.5 min ≈ 7.5 hr on RTX 5090

---

## §3 What's needed

### §3.1 Per-innovation ablation matrix

Run the 4 ablation configurations × 5 metals × 3 seeds = 60 runs:

1. **Multi-task only** (no W₂, no L2 guard, 5-metal joint cohort):
   - `--multi-task --no-wass --no-l2-guard`
2. **W₂ only** (single-task D-MPNN + W₂ + L2 guard, Ru-only):
   - `--no-multi-task --wass --l2-guard --single-metal {M}`
3. **L2 guard only** (single-task D-MPNN + L2 guard, no W₂, Ru-only):
   - `--no-multi-task --no-wass --l2-guard --single-metal {M}`
4. **Multi-task + W₂ + L2 guard** (Wf-124 P3 default):
   - `--multi-task --wass --l2-guard` (already run for 15 (metal, seed) pairs in TODO-180)

### §3.2 Attribution analysis

For each metal (Ru, Ir, Os, Rh, Re) and the aggregate:

```
Δ_auc_total = auc(multi-task + W₂ + L2 guard) - auc(baseline)
Δ_auc_multi-task = auc(multi-task only) - auc(baseline)
Δ_auc_W₂ = auc(W₂ only) - auc(baseline)
Δ_auc_L2_guard = auc(L2 guard only) - auc(baseline)
```

Verify additivity (or interactions):
```
Δ_auc_total ≈ Δ_auc_multi-task + Δ_auc_W₂ + Δ_auc_L2_guard (if independent)
Δ_auc_total ≠ Σ (if interactive — quantify the interaction)
```

### §3.3 Ablation report

Write `molmetal/reports/wf124_p3_ablation/ablation_report.json`:

```json
{
  "ablation_date": "2026-09-27",
  "n_configs": 4,
  "n_metals": 5,
  "n_seeds": 3,
  "n_total_runs": 60,
  "per_metal_attribution": {
    "Ru": {
      "baseline_auc": 0.7123,
      "multi_task_only_auc": ...,
      "wass_only_auc": ...,
      "l2_guard_only_auc": ...,
      "full_recipe_auc": 0.7613,
      "delta_total": 0.0490,
      "delta_multi_task": ...,
      "delta_wass": ...,
      "delta_l2_guard": ...,
      "interaction_term": ...
    },
    ...
  },
  "conclusion": "Multi-task accounts for X% of lift, W₂ accounts for Y%, L2 guard accounts for Z%, interaction accounts for W%"
}
```

### §3.4 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Wf-124 P3 default (multi-task + W₂ + L2 guard) Ru seed=0 test_auc=0.7613 (+0.049 vs baseline)
- **PROJECTED**: multi-task likely accounts for 60-80% of lift (joint 5-metal signal is the main lever), W₂ for 10-30%, L2 guard for ~5%
- **Interaction caveat**: the 3 levers may interact (e.g., multi-task provides a better starting point for W₂ to work with), so the simple additive decomposition may underestimate the interaction
- **Per-metal caveat**: smaller metals (Rh/Os/Re) may have different attribution than Ru
- **No fabrication**: do NOT cite per-lever attribution percentages as MEASURED until the 60 ablation runs complete

---

## §4 Acceptance gate

- [G6.1] 60 ablation runs complete (4 configs × 5 metals × 3 seeds)
- [G6.2] Per-metal attribution reported (multi-task %, W₂ %, L2 guard %, interaction %)
- [G6.3] Ablation report JSON written at `molmetal/reports/wf124_p3_ablation/ablation_report.json`
- [G6.4] Additivity check (or interaction quantification) computed
- [G6.5] Paper §3 algorithm description can cite the per-lever attribution
- [G6.6] Honest framing per `party-twenty-spirit` §3.5: per-lever percentages reported with std; no best-seed selection; no overclaim

---

## §5 ETA + dependencies

**ETA**: 2 - 3 days (7.5 hr wall-time + 1 day overhead for run + audit + ablation analysis + report)

**Dependencies**:
- RTX 5090 GPU-1 (CUDA_VISIBLE_DEVICES=1)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC — Wf-124 P3 ship; needs `--no-multi-task`, `--no-wass`, `--no-l2-guard`, `--single-metal` flags if not present)
- `molmetal/baselines/dmpnn_multitask.py` (MultiTaskLoss + DMPNNMultiTaskModel — Wf-124 P3 ship)
- TODO-180 G4 (14 PENDING (metal, seed) pairs complete first for the Wf-124 P3 default row)
- TODO-181 G5 (50-epoch full sweep — should this be 50-epoch or 30-epoch ablation? Honest choice: ablation should match the larger epoch count for fair comparison)
- `mol-metal-gpu-execution` §6 (5-seed × 50-epoch protocol — waiver for 3-seed per session time budget)

**New infrastructure**: trainer needs ablation flags if not present (`--no-multi-task`, `--no-wass`, `--no-l2-guard`, `--single-metal`).

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Wf-124 P3 default Ru seed=0 test_auc=0.7613 (+0.049 vs 0.7123 baseline); aggregate lift from 3 simultaneous innovations
- **PROJECTED**: per-lever attribution (multi-task 60-80%, W₂ 10-30%, L2 guard ~5%) — but this is a hypothesis, not measured; honest framing is "uncertain, needs measurement"
- **Interaction caveat**: the 3 levers may interact (positive or negative); simple additive decomposition may be misleading
- **Per-metal caveat**: Rh/Os/Re < 1.2k rows may have different attribution than Ru; the per-metal attribution matrix is essential for honest reporting
- **30-epoch vs 50-epoch choice** (per TODO-181 G5): ablation should match the larger epoch count for fair comparison; otherwise the ablation may understate the lift (some levers may need more epochs to fully kick in)
- **No fabrication**: do NOT cite per-lever attribution percentages as MEASURED until the 60 ablation runs complete; current paper §3 algorithm description must say "aggregate lift" not "per-lever attribution"

---

## §7 Cross-references

- `molmetal/reports/wf124_p3_multitask/synthesis.md` §3 (3-innovation simultaneous run cite)
- `molmetal/reports/wf124_p3_multitask/synthesis.md` §4 (Ru seed=0 default config MEASURED)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC — Wf-124 P3 ship; may need ablation flags)
- `molmetal/baselines/dmpnn_multitask.py` (MultiTaskLoss + DMPNNMultiTaskModel)
- `molmetal/molmetal_lam/reward/wasserstein_loss.py` (~530 LOC — M1 module)
- `TODO/pending/119_decode_ratio_strengthen.md` §1.4 (per-innovation ablation follow-up cite)
- `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 (parent strengthen TODO — m1 Ru AUC ≥ 0.86)
- `TODO/pending/126_innovation_strengthen_roadmap.md` §3 (M1+M2 master plan)
- `mol-metal-gpu-execution` §6 (5-seed × 50-epoch protocol)
- `mol-metal-gpu-execution` §7 (MEASURED vs PROJECTED tag discipline)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4 sibling)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5 sibling)

---

## §8 Follow-ups

- Once G6 closed: paper §3 algorithm description can cite per-lever attribution with measured percentages
- If attribution is dominated by multi-task (e.g., 80%+): W₂ may not be a useful lever for paper §4 LEAD claim; down-weight W₂ in future strengthening plans
- If W₂ contributes meaningfully (e.g., 20%+): W₂ is a real lever; expand to other strengthen TODOs (TODO-121, TODO-122, TODO-123, TODO-125)
- If interaction term is large (e.g., >30% of total lift): the levers are NOT independent; the simple "stack innovations" approach is misleading; need to model the interaction explicitly

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: The Wf-124 P3 default (multi-task + W₂ + L2 guard) is an aggregate recipe. Per-lever attribution is a hard requirement for honest paper §3 algorithm description — readers must know which lever did what. Until the 60 ablation runs complete, paper §3 must say "aggregate lift" not "per-lever attribution". The projected attribution (multi-task 60-80%, W₂ 10-30%, L2 guard ~5%) is a hypothesis, not measured.
