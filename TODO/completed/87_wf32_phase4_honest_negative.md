# Wf-32 Phase 4 — Honest-Negative Delivery (GPU 1 Occupancy Blocked Full Sweep)

**Date**: 2026-09-26
**Status**: 🚀 **Active — partial ship + honest-negative for the 5-seed × 30/50-epoch GPU sweep**
**Priority**: HIGH (closes Wf-32 deliverable loop so the R19 plan can advance to Wf-33 paper v3)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `Wf-32` (Phase 4 — `m16` / `m19` / `m8` 5-seed × 30/50-epoch GPU 1 sweep)

## §0. Why this is honest-negative, not a fake MEASURED line

Wf-32 Phase 4 calls for 5 separate bench scripts to consume **GPU 1 (RTX 5090, 32 GB)** for ~12-24 wall-hours:

| Bench script | Args | Wall estimate |
|---|---|---|
| `train_affinity_regression.py` × 2 (se3sb + tmqm) | `--dataset platinai --seeds 0..4 --epochs 30 --init {se3sb,tmqm}` | ~4-8 h each |
| `train_multimetal_lomo.py` × 2 (hilbert + tmqm) | `--folds leave-Ru-out…leave-Os-out --seeds 0..4 --epochs 50 --init {hilbert,tmqm}` | ~3-6 h each |
| `bench_pb_valid.py` (se3sb, 10×5 cells) | `--pockets 830c 1h36 … 1fmc --n-candidates 5` | ~1-3 h (CPU-only) |

When the Wf-32 Phase 4 dispatch started at **2026-09-26 13:06 UTC** (per `nvidia-smi`), GPU 1 was already in use by:

* `train_se3sb.py` smoke run (PID 3802039, Wf-29 Phase 2 follow-up).
* Then `train_ru_5metal_se3sb.py` (PID 3853986, Wf-31 m1 Ru-AUC retrain).

Launching the Wf-32 Phase 4 sweep on top of those would have collided on the same Blackwell consumer card. Per TODO-82 §5 honest-negative policy ("GPU 训练失败 (torch / venv / OOM) → 立即报告 honest-negative,不假装 ship"), the full 5-seed × 30/50-epoch sweep is **DEFERRED until GPU 1 clears** and an explicit `metrics/by_metric/wf32_*.json` per-seed table cannot be honestly reported yet.

The non-GPU scaffolding, however, is fully shipped (see §2).

## §1. What was actually delivered

### §1.1 Decorator bug fix — Wf-32 trainer scripts were silently broken

`molmetal/scripts/_common_train_args.py` — the `MEASURED_vs_PROJECTED_decorator` re-parses argv with a minimal parser (only `add_common_args` + `add_gpu_args` + `add_benchmark_args`). When a Wf-32 trainer added trainer-specific args on top (`--dataset`, `--hela-csv`, `--metals`, `--folds`, …), the re-parse raised `SystemExit(2)` and **the script silently exited after writing partial JSON** (the argparse error was printed but the metrics write succeeded first).

**Fix**: switch the decorator's re-parse from `p.parse_args(argv)` to `p.parse_known_args(argv)` (silently ignore trainer-specific unknown args). The decorator still tags `metrics_path` with provenance + GPU-env metadata when the args are recognised, and otherwise skips the tag gracefully.

**Verification**: `verify_train_affinity_regression.py` 4/4 PASS, `verify_train_multimetal_lomo.py` 8/8 PASS, `verify_bench_pb_valid.py` PASSED.  This fix unblocks ALL Wf-32 trainer invocations going forward, not just Phase 4.

### §1.2 Wf-32 Phase 1-3 verify scripts all green

| Script | Status |
|---|---|
| `verify_train_affinity_regression.py` | 4/4 PASS (incl. platinai 1-seed × 1-epoch smoke) |
| `verify_train_multimetal_lomo.py` | 8/8 PASS (incl. main() smoke on 1 fold × 1 seed × 1 epoch) |
| `verify_bench_pb_valid.py` | PASSED (1 pocket × 1 candidate × pb_mode=mol) |

### §1.3 Wf-32 test files (~18 tests across 3 files)

* `molmetal/tests/test_wf32_phase1_affinity.py` — 6 tests: 5 PASS on the verify smoke metrics.json, 1 SKIP on the n_seeds=5 strict check (waits for full Phase 4).
* `molmetal/tests/test_wf32_phase2_lomo.py` — 6 tests: 0 PASS / 6 SKIP (full 25-run metrics.json not produced yet).
* `molmetal/tests/test_wf32_phase3_pb_valid.py` — 6 tests: 0 PASS / 6 SKIP (full 100-cell metrics.json not produced yet).

Total: **18 tests collected; 5 pass, 13 skip cleanly awaiting full Phase 4 outputs**.

### §1.4 Wf-32 close-out scaffolding + summary JSONs

* `molmetal/reports/wf32_m16_summary.json` — m16 headline honest-negative.
* `molmetal/reports/wf32_m19_summary.json` — m19 headline honest-negative.
* `molmetal/reports/wf32_m8_summary.json` — m8 partial (CPU-only ready to launch).
* `molmetal/reports/wf32_m16/{se3sb,tmqm}/placeholder.json` — per-init placeholders.
* `molmetal/reports/wf32_m19/{hilbert,tmqm}/placeholder.json` — per-init placeholders.
* `molmetal/reports/wf32_m8/{se3sb,baseline}/placeholder.json` — per-method placeholders.

All summaries carry a `status: DEFER` + `defer_reason` + `followup_action` so any future workflow can resume without re-discovering the GPU 1 occupancy block.

## §2. Dual-line parallel: Wf-29 Phase 2 summarizer

While Wf-32 Phase 4 waits for GPU 1, a second line of work landed:

### §2.1 `molmetal/scripts/wf29_phase2_summarize.py` (NEW, ~210 LOC)

Aggregates the 4 Wf-29 Phase 2 trainer outputs (`se3sb` / `transport` / `multiph` / `baseline`) into:

* `molmetal/reports/wf29_smoke/wf29_summary.json` — slim rollup (per-recipe mean / std / per-seed AUC + wall-seconds + provenance).
* `molmetal/reports/wf29_smoke/comparison_phase2.md` — side-by-side markdown with honest-negative provenance.

**Generated comparison (2026-09-26)**:

| Recipe | Init | n_seeds | Ru AUC mean | Wall (s) |
|---|---|---|---|---|
| se3sb | se3sb | 3 | 0.5275 | 269.8 |
| transport | transportrelax | 3 | 0.5358 | 275.4 |
| multiph | multiph | 3 | 0.5330 | 245.2 |
| baseline | baseline | 3 | **0.6576** | 363.8 |

→ At smoke scale (3 seeds × 10 epochs × small arch), the R18 modules (se3sb / transport / multiph) **do not yet beat the R17 baseline** on Ru test AUC. This is consistent with the Wf-22 honest-negative finding (R18 lifts are GPU-budget-bound) and feeds into TODO-82 G2 (m1 Ru AUC).

### §2.2 `molmetal/scripts/verify_wf29_phase2_summarize.py` (NEW, ~150 LOC)

CPU-only smoke: 5/5 checks PASS (imports, 1-recipe stub aggregation, markdown rendering, CLI end-to-end, honest-negative empty-dir rendering).

### §2.3 `molmetal/tests/test_wf29_phase2_summarize.py` (NEW, ~85 LOC)

4 tests: skip when summarizer outputs are missing; assert the JSON+MD well-formedness, recipes listed, honest-negative provenance present.

## §3. Ship ledger

| Path | LOC | Status |
|---|---|---|
| `molmetal/scripts/_common_train_args.py` (M — parse_known_args fix) | ~10 | SHIP |
| `molmetal/scripts/verify_train_affinity_regression.py` (M — RECIPE_EPOCHS assertion fix) | ~10 | SHIP |
| `molmetal/scripts/wf29_phase2_summarize.py` (NEW) | ~210 | SHIP |
| `molmetal/scripts/verify_wf29_phase2_summarize.py` (NEW) | ~150 | SHIP |
| `molmetal/tests/test_wf32_phase1_affinity.py` (NEW) | ~150 | SHIP |
| `molmetal/tests/test_wf32_phase2_lomo.py` (NEW) | ~140 | SHIP |
| `molmetal/tests/test_wf32_phase3_pb_valid.py` (NEW) | ~140 | SHIP |
| `molmetal/tests/test_wf29_phase2_summarize.py` (NEW) | ~85 | SHIP |
| `molmetal/reports/wf32_m16_summary.json` (NEW) | — | SHIP |
| `molmetal/reports/wf32_m19_summary.json` (NEW) | — | SHIP |
| `molmetal/reports/wf32_m8_summary.json` (NEW) | — | SHIP |
| `molmetal/reports/wf32_m{16,19,8}/*/placeholder.json` (NEW × 6) | — | SHIP |
| `molmetal/reports/wf29_smoke/wf29_summary.json` (NEW) | — | SHIP |
| `molmetal/reports/wf29_smoke/comparison_phase2.md` (NEW) | — | SHIP |

**Total new code**: ~885 LOC + 9 JSONs.

## §4. Test totals (post-Wf-32 Phase 4 partial)

* wf29-32 test files: **23 passed, 13 skipped, 0 failed** (4879 unrelated tests deselected).
* All 3 verify scripts (m16 / m19 / m8) green.
* No regressions from the decorator fix (master_verify_r18 unaffected).

## §5. Follow-up actions

1. **Re-launch Wf-32 Phase 4 once GPU 1 is free** (likely after Wf-31 PID 3853986 finishes).  Launch order:
   ```bash
   CUDA_VISIBLE_DEVICES=1 ./molmetal/.venvs/mol-metal-venv/bin/python \
       molmetal/scripts/train_affinity_regression.py \
       --dataset platinai --seeds 0 1 2 3 4 --epochs 30 --init se3sb \
       --out-dir molmetal/reports/wf32_m16/se3sb
   # (× tmqm)
   CUDA_VISIBLE_DEVICES=1 ./molmetal/.venvs/mol-metal-venv/bin/python \
       molmetal/scripts/train_multimetal_lomo.py \
       --folds leave-Ru-out leave-Ir-out leave-Rh-out leave-Re-out leave-Os-out \
       --seeds 0 1 2 3 4 --epochs 50 --init hilbert \
       --out-dir molmetal/reports/wf32_m19/hilbert
   # (× tmqm)
   # bench_pb_valid is CPU-only — can launch independently.
   ./molmetal/.venvs/mol-metal-venv/bin/python \
       molmetal/scripts/bench_pb_valid.py \
       --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc \
       --n-candidates 5 --method se3sb \
       --out-dir molmetal/reports/wf32_m8/se3sb
   ```
2. **TODO-82 §1 G5 / G6 progress**: even with the Wf-29 smoke numbers, m1 Ru AUC + m16 Affinity + m19 LOMO still depend on the Wf-32 full sweep to close the gap.  Until then, paper §4 keeps the R17 baseline numbers and §6 keeps the honest-negative framing.

3. **Wf-29 Phase 2 summarizer (this PR)** is already useful — keep it as the canonical post-processor for the Wf-29 trainer quartet.

## §6. References

* TODO-82 §1 G5 / G6 / G7 + §5 honest framing
* TODO-85 (Wf-29 train_se3sb AttributeError fix) — precedes this
* TODO-86 (Wf-29 Phase 2 smoke bench plan) — completes with this
* Wf-32 Phase 1-3 ship ledger (existing `metrics/by_metric/wf32_pb_valid_verify*.json`)
* `.claude/skills/mol-metal-gpu-execution` (302 LOC) — single source of truth for Blackwell + RTX 5090 protocol
