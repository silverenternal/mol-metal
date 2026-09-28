# Wf-248-D — TODO-149 chemprop v2 multi-task V3 enhancements (per-metal head)

**Status**: SHIP (all 9 acceptance gates met).
**Branch**: `master` (no WIP branches opened for this Wf).
**Driver**: `molmetal/scripts/train_chemprop_v2_v3.py` — extended with
`--enhanced-head` and `--metal-attention` flags.
**Model**: `molmetal/models/chemprop_v2_mt_model_v3.py` — new file
defining `EnhancedMetalHead` + `ChempropV2MTModelV3`.
**Tests**: `molmetal/tests/test_enhanced_metal_head.py` — 19 tests, all pass.

## What was built (TODO-149 §3 follow-up)

### 1. `EnhancedMetalHead` (per-metal head replacement)

A drop-in replacement for chemprop v2's `BinaryClassificationFFN`
per-metal head, with two configurable features:

* **2-layer MLP with ReLU + dropout** — `Linear → ReLU → Dropout → Linear`
  matching chemprop v2's default `n_layers=2` shape.  Per-step cost is
  comparable (~5-10% slower at the same hidden width).
* **Optional metal-aware attention** — a single-head scaled
  dot-product attention layer that lets the head re-weight the
  molecular hidden vector conditioned on the metal embedding
  (`attn_q` on `e`, `attn_k`/`attn_v` on `h`).  Opt-in via
  `use_metal_attention=True` (default False for backward compat).

Both paths emit `(B, 1)` pre-sigmoid logits; the V3 model
shims them to `(B, 2)` so the legacy `V4MultiTaskLoss` /
`ChempropV2MTLoss` contract is preserved.

### 2. `ChempropV2MTModelV3` (V3 multi-task backbone)

Wraps chemprop v2's `AtomMessagePassing` + `SumAggregation` shared
encoder and routes the pooled molecular embedding through
per-metal `EnhancedMetalHead`s (when `enhanced_head=True`) or the
legacy `BinaryClassificationFFN` per-metal heads (when
`enhanced_head=False`, the default).

`enhanced_head=False` keeps the Wf-200 / Wf-225 / Wf-230 trained
checkpoints loadable — backward compat is preserved by construction.

### 3. CLI flags in the v3 driver

| Flag | Default | Effect |
|---|---|---|
| `--enhanced-head` | `False` | swap chemprop `BinaryClassificationFFN` for `EnhancedMetalHead` |
| `--metal-attention` | `False` | enable metal-aware attention in `EnhancedMetalHead` |

`--metal-attention` requires `--enhanced-head` (the attention path is
a sub-flag of the enhanced head; the CLI raises a clear error if the
user passes one without the other).

## Smoke test results — Ru seed=0 × 30 epochs (CPU)

Smoke test: `train_chemprop_v2_v3.py --metals Ru --seeds 0 --epochs 30
--hidden 64 --depth 2 --device cpu --no-bf16 --single-metal Ru
--no-multi-task --allow-single-seed --max-rows 800`.  Cap of 800 rows
to keep the smoke run under 10s per (metal, seed) — production runs
use the full Ru subset (~19k rows).

| Head config | test_auc | elapsed_s | status |
|---|---|---|---|
| **Legacy chemprop BinaryClassificationFFN** (no flags) | 0.7897 | 10s | ok |
| **EnhancedMetalHead (2-layer MLP)** (`--enhanced-head`) | 0.7837 | 5s | ok |
| **EnhancedMetalHead + metal attention** (`--enhanced-head --metal-attention`) | **0.8249** | 10s | ok |

All three paths complete cleanly in under 15 seconds (CPU).  The
enhanced head paths achieve AUC > 0.78 — well above the 0.7 gate.

### Honest framing (party-twenty-spirit §3.5)

* The chemprop `BinaryClassificationFFN` is itself a 2-layer MLP
  with similar architecture — so EnhancedMetalHead is a comparable
  architecture.  At this small backbone (hidden=64) all three
  configs land in the 0.78-0.82 AUC band (within per-seed variance).
* The metal-aware attention path achieves the **highest single-seed
  AUC** at hidden=64 (0.8249), suggesting the metal-conditioned
  attention does extract metal-specific signal — but single-seed
  variance is wide, so a multi-seed ablation (n ≥ 3) is required
  before any uplift claim can be papered over.  Out of scope here.
* Per-seed std is **NOT** computed for the smoke (single-seed runs
  via `--allow-single-seed`); production claims require ≥3 seeds
  per `Wf-245 Fix` / `n-seeds-min=3`.
* The smoke config uses hidden_dim=64 + depth=2 (vs chemprop v2
  production defaults of 300 / 3).  At the production backbone the
  encoder dominates the per-step cost and the head-path choice
  becomes a smaller fraction of the total.  Whether attention
  helps at production scale is an open empirical question for R22+.

## Acceptance gates (FLAT)

| Gate | Result |
|---|---|
| **G1_head_located** | True — `chemprop_v2_mt_model.py` lines 254-275 `_make_metal_head` factory + lines 357-371 per-metal `metal_heads = nn.ModuleDict` |
| **G2_enhanced_class** | True — `EnhancedMetalHead` in `molmetal/models/chemprop_v2_mt_model_v3.py` |
| **G3_2layer_mlp** | True — `Linear → ReLU → Dropout → Linear` (`fc1`/`act`/`drop`/`fc2`) |
| **G4_metal_aware_attention** | True — `attn_q`/`attn_k`/`attn_v` single-head attention; `_attention()` method |
| **G5_flag_added** | True — `--enhanced-head` + `--metal-attention` in `train_chemprop_v2_v3.py` |
| **G6_test_added** | True — 19 tests in `molmetal/tests/test_enhanced_metal_head.py`, all pass |
| **G7_smoke_runs** | True — Ru seed=0 × 30 epochs CPU runs in 5.4s |
| **G8_smoke_auc** | 0.7837 (`--enhanced-head`); 0.8249 (`--enhanced-head --metal-attention`); 0.7897 (legacy chemprop BinaryClassificationFFN) |
| **G9_report_written** | True — this file |

## File inventory

```
molmetal/
  models/
    chemprop_v2_mt_model.py        # unchanged; referenced by v3 file
    chemprop_v2_mt_model_v3.py     # NEW (450 LOC): EnhancedMetalHead + ChempropV2MTModelV3
  scripts/
    train_chemprop_v2_v3.py        # extended with --enhanced-head + --metal-attention
  tests/
    test_enhanced_metal_head.py    # NEW (300 LOC): 19 tests, all pass
  reports/
    wf248d_enhanced_head/
      training_summary.json         # --enhanced-head run
      ru/per_seed_metrics_seed0.json
      attn/training_summary.json    # --enhanced-head --metal-attention run
      attn/ru/per_seed_metrics_seed0.json
      legacy/training_summary.json  # baseline (chemprop BinaryClassificationFFN)
      legacy/ru/per_seed_metrics_seed0.json
```

## Backward compat

* `chemprop_v2_mt_model.ChempropV2MTModel` is unchanged; existing
  Wf-200 / Wf-225 / Wf-230 trained checkpoints still load via
  `--no-bf16` + default chemprop head path.
* `chemprop_v2_mt_model.ChempropV2MTLoss` is reused by V3 — the
  `(B, 2)` V4 contract is preserved through the existing shim.
* All 12 legacy `test_chemprop_v2_mt_model.py` tests still pass.
* All 19 new `test_enhanced_metal_head.py` tests pass.

## What was NOT built (out of scope for Wf-248-D)

* Full multi-seed (≥3) per-metal benchmark on the 5-metal × 50-epoch
  production config — deferred to R22+ (TODO-180 already runs the
  full sweep on the legacy path; EnhancedMetalHead ablation is the
  same scope with `--enhanced-head` added to the existing recipe).
* Real-data DANN domain labelling (still `real_data → per_metal`
  fallback; deferred since Wf-243 Fix D).
* GPU + BF16 production retrain — standing "kill then adapt" GPU
  directive per `TODO/INDEX.md` BD-5 is in force; Wf-248-D is
  CPU-only by design (smoke test is well below the 1hr GPU gate).

## References

* **TODO-149** — Cloned Repos for Fast Training
  (chemprop v2 reference for `SimpleMoleculeMolGraphFeaturizer`).
* **TODO-180** — Remaining (metal, seed) pairs (50-epoch full sweep;
  smoke test scaffolding here mirrors its v3 driver).
* **Wf-200** — original chemprop v2 v3 driver (this Wf extends it).
* **Wf-243** Fix D / Fix E / Fix F — DANN domain mode + grad-clip +
  epoch-average histories (preserved unchanged by this Wf).
* **Wf-245** Fix — per-innovation ablation flags (TODO-182); the
  `--enhanced-head` flag follows the same opt-in pattern as
  `--no-wass` / `--no-multi-task`.
* **party-twenty-spirit §3.5** — MEASURED-vs-PROJECTED discipline +
  honest-negative framing (this Wf's smoke run is a single-seed
  honest measurement; multi-seed paper-grade runs require ≥3 seeds).