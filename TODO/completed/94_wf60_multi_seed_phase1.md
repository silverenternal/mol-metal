# Wf-60 Phase 1 — Multi-seed expansion for m1 Ru + m2 Pt 4-cell + m19 LOMO

**Date**: 2026-09-26
**Workflow**: Wf-60 Phase 1 (computed task dispatched by user).
**Status**: 🚀 **In-flight** — orchestrator + tests ship, 3 training runs in background on GPU-1.
**Parent**: `TODO/pending/82_metric_gap_filling_plan.md` §3.5 (TODO-82 honest framing).

---

## §0 TL;DR

| Deliverable | File | Status |
|---|---|---|
| `molmetal/scripts/bench_weak_real_multi_seed.py` | ~360 LOC wrapper | ✅ SHIP |
| `molmetal/tests/test_weak_real_multi_seed.py` | 20 tests, all pass | ✅ SHIP |
| m1 Ru 5-seed × 50-ep × strict-scaffold | `molmetal/reports/wf60_multi_seed/m1/train_5seed.log` | ⏳ in-flight |
| m2 Pt 4-cell 5-seed × 50-ep × strict-scaffold | `molmetal/reports/wf60_multi_seed/m2/train_5seed.log` | ⏳ in-flight |
| m19 LOMO 5-fold × 5-seed × 30-ep | `molmetal/reports/wf60_multi_seed/m19/train_5seed.log` | ⏳ in-flight |
| `molmetal/reports/wf60_multi_seed/synthesis.{json,md}` | aggregator | ⏳ pending run completion |

20/20 unit tests pass on CPU. 3 training runs are executing on GPU-1 (RTX 5090) and will land per-seed AUC + mean/std at the 5-seed-by-target gates (m1 ≥ 0.74, m2 ≥ 0.60, m19 ≥ 0.55).

---

## §1 Why this Wf exists

Wf-22 + Wf-31 + Wf-32 collided on GPU-1 (32 GB) — see Wf-31 Phase 4 / Wf-32 Phase 4 honest-negative + TODO-82. The on-disk Wf-22/Wf-31/Wf-32 MEASURED values are smoke-scale or DEFERRED. Wf-60 re-runs m1 + m2 + m19 on the **same** RTX 5090 at 5-seed × {50, 50, 30} epochs × strict scaffold split (Wf-59 strict protocol) and emits per-seed ledgers so §4 Table 2 cells can be backed by real numbers.

Key constraints honored:
- **GPU-1 only** — `CUDA_VISIBLE_DEVICES=1` (RTX 5090 32 GB)
- **No GPU-0 collision** — Blackwell sm_120 untouched (97 GB free)
- **CPU fallback acceptable** — three trainers are CPU-portable
- **`--strict-scaffold`** — every m1/m2 call passes `--scaffold-split` (m19 is LOMO-only)

---

## §2 What shipped

### §2.1 Orchestrator — `molmetal/scripts/bench_weak_real_multi_seed.py` (~360 LOC)

A pure orchestrator (zero torch / rdkit / sklearn imports). It:
1. Routes `--metrics {m1,m2,m19}` to the corresponding R17-close-out trainer via subprocess (CUDA env propagated).
2. Aggregates each trainer's `per_seed_metrics.json` into a top-level ledger.
3. Computes mean/std across seeds; compares to R17 baseline + Wf-60 gate targets.
4. Emits `<out-dir>/synthesis.{json,md}`.

The dry-run flag (`--dry-run`) prints the planned commands without invoking them — used for CI verification and reproducing the exact command stream from JSON.

### §2.2 Tests — `molmetal/tests/test_weak_real_multi_seed.py` (20 tests, all pass)

Covers:
- Recipes catalogue (3 metrics present, gate targets match spec).
- CLI parsing (defaults, custom seeds, dry-run, unknown metric rejection, default GPU-1).
- Aggregation (mean matches `statistics.fmean`, gate_pass semantics, failure marking, Δ vs baseline).
- Synthesis markdown emissions (PII of the run ledger + per-metric verdict + failed-metric handling).
- Subprocess invocation (mocked) and `dry-run` mode.
- Wrapper purity (no `torch` / `rdkit` imports leak).

Run all tests:

```
cd /home/hugo/codes/mol-metal && source env.sh && molmetal/.venvs/mol-metal-venv/bin/python -m pytest \\
    molmetal/tests/test_weak_real_multi_seed.py -v
```

Result: **20 passed in 0.24s**.

### §2.3 Three parallel training runs (in-flight on GPU-1)

| Metric | Trainer script | Seeds | Epochs | Process ID | GPU | Wall estimate |
|---|---|---:|---:|---:|---|---|
| m1 | `train_metal_cytotox.py --metal Ru --scaffold-split` | 0..4 | 50 | 3996317 | RTX 5090 | ~12 min |
| m2 | `train_platinai_4cell.py --scaffold-split` | 0..4 | 50 | 3995113 | RTX 5090 | ~12 min |
| m19 | `train_metal_lomo.py --metals Ru Ir Os Rh Re` | 0..4 | 30 | 3996358 | RTX 5090 | ~8 min |

All three use `MOLMETAL_DATA_DIR=/home/hugo/codes/mol-metal/.data/molmetal` so `MetalCytoToxDB.csv` resolves to the project-local dataset (legacy `/mnt/storage/data/molmetal` is unwritable on this host). `platinum_4cell.csv` resolves to `molmetal/data/platinai_4cell.csv` (project-local).

GPU-1 (RTX 5090, 32 GB) at 100 % utilization, ~2.4 GB VRAM used across the three jobs — multi-process PyTorch safely shares the device.

---

## §3 Per-metric recipe

```python
RECIPES = {
    "m1": MetricRecipe(
        gate_baseline=0.7123,    # R17 baseline (1-seed × 5-epoch)
        gate_target=0.74,        # Wf-60 pass criterion (5-seed mean)
        ...
    ),
    "m2": MetricRecipe(
        gate_baseline=0.5532,    # R17 baseline (3-seed × 50-epoch)
        gate_target=0.60,        # Wf-60 pass criterion
        ...
    ),
    "m19": MetricRecipe(
        gate_baseline=0.5346,    # R17 baseline (3-seed × 30-epoch)
        gate_target=0.55,        # Wf-60 pass criterion
        ...
    ),
}
```

`gate_pass = mean >= gate_target` (NaN-safe).

---

## §4 Wf-60 close-out vs spec

| Spec requirement | Status |
|---|---|
| `molmetal/scripts/bench_weak_real_multi_seed.py` (~300 LOC) | ✅ ~360 LOC |
| `molmetal/tests/test_weak_real_multi_seed.py` (~18 tests) | ✅ 20 tests, all pass |
| `molmetal/reports/wf60_multi_seed/{m1,m2,m19}/per_seed_metrics.json` | ⏳ in-flight (per trainer) |
| `molmetal/reports/wf60_multi_seed/synthesis.md` | ⏳ pending run completion (auto-emitted by orchestrator) |
| 5-seed mean m1 ≥ 0.74 | ⏳ pending m1 run completion |
| 5-seed mean m2 Pt 4-cell ≥ 0.60 | ⏳ pending m2 run completion |
| 5-seed mean m19 LOMO ≥ 0.55 | ⏳ pending m19 run completion |
| GPU-1 only (no GPU-0 collision) | ✅ `CUDA_VISIBLE_DEVICES=1` |
| CPU fallback acceptable | ✅ torch.is_available fallback in trainers |

---

## §5 Honest framing (party-twenty-spirit §3.5)

Per the meta-skill's no-fabrication rule:
- The wrappers ship test PASS *before* the training runs land; if any of the three trainers collides on GPU-1 the synthesis will report a `FAILED` entry per metric, **not** a synthetic mean.
- The `--strict-scaffold` flag is mandatory for m1 + m2 (the script enforces it via `extra_args`). m19 is LOMO-only and therefore is not scaffold-split (it leaves one metal out at a time).
- The recipe's `gate_baseline` values exactly match `metrics/by_metric/{metalcytotox_ru_auc,platinai_4cell_auc,metal_lomo_auc}.json` `value` fields (R17 canon). Δ-vs-baseline is computed from the 5-seed mean, not a single-seed point.
- No MEASURED number is faked. Once the three runs land, this doc's TL;DR table will be updated with `MEASURED` AUCs.

---

## §6 Cross-references

- Computed harness task: Wf-60 Phase 1 (multi-seed expansion)
- Parent plan: `TODO/pending/82_metric_gap_filling_plan.md` §3.5
- R17 baselines canon:
  - `metrics/by_metric/metalcytotox_ru_auc.json` (`value = 0.7123`)
  - `metrics/by_metric/platinai_4cell_auc.json` (`value_overall_mean_auc = 0.5532`)
  - `metrics/by_metric/metal_lomo_auc.json` (`value = 0.5346`)
- Wf-31 smoke outputs: `molmetal/reports/wf31_m1_summary.json` + `molmetal/reports/wf31_m1_m2_m7_lift_summary.md`
- Wf-32 smoke outputs: `molmetal/reports/wf32_m19_summary.json` + `molmetal/reports/wf32_m16_m19_m8_lift_summary.md`

---

## §7 Parallelization note (ultracode analog)

There is **no `ultracode` binary or module** in this environment — `ultracode` is the historical naming convention for heavyweight parallel workflows invoked through the `Task` tool (not available in this harness). The closest analog in the harness is `Bash run_in_background: true` × multiple parallel calls, used here to launch the three Wf-60 training runs simultaneously on RTX 5090.

If the user wants to dispatch multiple Wf-style tasks as parallel subagents, the project would need to be configured to use a subagent-orchestrating tool (e.g. the `Task` tool or an external scheduler). Within the current harness capabilities, the Wf-60 task (m1/m2/m19) plus the wrapper + tests ship as one structured parallel dispatch.
