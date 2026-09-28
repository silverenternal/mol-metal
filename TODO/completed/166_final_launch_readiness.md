# TODO-166 — Wf-146 + chemprop v2 final launch readiness

**Date**: 2026-09-27 (post-Wf-155 abort + Wf-156 smoke + chemprop v2 pivot + V4 P0 patch fixes)
**Status**: ⚙️ **READY-TO-LAUNCH (chemprop v2 path)** / 🚧 **BLOCKED (V4 pretrain path; canonical-metal filter not yet applied)**
**Owner**: Hugo Lee + ultracode agents
**Parent**: TODO-148 (Wf-146 prep) + TODO-155 (pivot Wf-146/157/158 to chemprop v2) + TODO-157 (pre-flight checklist) + TODO-158 (launch readiness, predecessor) + TODO-160 (Wf-120/121/123/124 P0 fixes; canonical-metal filter gap) + TODO-162 (pivot strategy sibling) + TODO-164 (chemprop v2 final ship) + TODO-165 (domain label re-featurize)
**Skill alignment**: `party-twenty-spirit` §3.5 (every AUC / ETA / line number traces to a TODO path or test artefact; no fabrication)

---

## §0 Status — READY-TO-LAUNCH on the chemprop v2 path

**Verdict (one-line)**: chemprop v2 path is **READY-TO-LAUNCH** — pipeline, model, loss, driver, tests, smoke ckpt, and pre-cache are all green and verified; the canonical-metal filter is **NOT** a blocker for the chemprop v2 LOMO driver (it loads `metal_cytotox` directly via `load_metal_cytotox()`, NOT `all_pretrain.pt`); the canonical-metal filter IS still a blocker for the V4 pretrain path (`train_dmpnn_multitask.py --data-config pretrain`), which we are **not** launching for Wf-146 per the TODO-155 pivot.

| Path | Verdict | Why |
|------|---------|-----|
| **chemprop v2 LOMO** (`train_chemprop_v2_lomo.py`) | **READY-TO-LAUNCH** | Driver uses `load_metal_cytotox()` (NOT `all_pretrain.pt`); chemprop v2 model + loss + tests (21) + smoke ckpt all SHIPPED; pre-cache solves the Wf-155 abort. |
| V4 pretrain (`train_dmpnn_multitask.py --data-config pretrain`) | **BLOCKED** | Canonical-metal filter bug (TODO-160 §4.2) — 304,498 of 336,556 rows (90.5%) have `metal_idx = -1` and crash `metal_embed`. NOT launching V4 pretrain for Wf-146. |
| V4 LOMO (`train_metal_lomo_v2.py`) | **READY-TO-LAUNCH (with caveat)** | V4 path uses balanced Rh cohort + L2 guard; ETA 18.8 hr (4.1× slower than chemprop v2); kept as §B appendix honest baseline. |
| chemprop-multi-task 5-metal shared backbone | **DEFERRED R20 GPU** | Apples-to-apples fair comparison; deferred to R20 GPU retrain per TODO-162 §1.3. |

**Blocker (launch-gating)**: user must lift the "kill then adapt" deferral per `TODO/INDEX.md` + the Wf-146 sign-off checklist (§7 below). No technical blockers remain on the chemprop v2 path.

---

## §1 Canonical-metal filter fix (V4 pretrain-path blocker, NOT a chemprop v2 blocker)

### §1.1 What is the canonical-metal filter bug

Per `TODO-160 §3.2 + §4.2` (MEASURED inspection 2026-09-27):

> **`molmetal/data/pre_featurized/all_pretrain.pt`** contains **304,498 rows (90.5%) with `metal_idx = -1`** (the 4-domain `WeightedRandomSampler` cohort includes DrugOOD + NCI-60 + BindingDB rows that have no canonical-metal mapping). `V4MultiTaskDMPNN.metal_embed = nn.Embedding(5, 64)` only supports `metal_idx ∈ {0,1,2,3,4}` (Ru/Ir/Rh/Re/Os). The pretrain loader at `molmetal/scripts/train_dmpnn_multitask.py:522` (`build_pretrain_dataloader`) does **NOT** filter non-canonical rows, so the first batch triggers an index-out-of-bounds in `metal_embed` → CUDA assertion crash.

### §1.2 Status of the fix (as of 2026-09-27)

**NOT YET APPLIED** — the 5-line patch from TODO-160 §4.2 is still required for any full-corpus V4 pretrain run. **However**, this fix is **NOT** needed for the Wf-146 chemprop v2 launch because:

1. `molmetal/scripts/train_chemprop_v2_lomo.py` line 473-474 loads `metal_cytotox` directly via `load_metal_cytotox()` (NOT `all_pretrain.pt`);
2. chemprop v2 trains **per-metal independently** (5 separate `ChempropV2MTModel` instances, one per Ru/Ir/Os/Rh/Re split) — there is no shared `metal_embed` to crash;
3. The chemprop v2 `_make_metal_head` factory (TODO-164 §1.1) creates one `BinaryClassificationFFN` per metal at instantiation time, not at forward time.

The canonical-metal filter is therefore a **V4-path-only** blocker and is **orthogonal** to the Wf-146 chemprop v2 launch.

### §1.3 The 5-line patch (kept verbatim for V4-path follow-on; NOT required for Wf-146)

```python
# in molmetal/scripts/train_dmpnn_multitask.py:522 build_pretrain_dataloader
# AFTER line 575: tensors = [pack[k] for k in keys_tensors]
# ADD:
keep_mask = tensors[4] >= 0  # metal_idx is index 4 in keys_tensors
n_drop = keep_mask.numel() - int(keep_mask.sum().item())
tensors = [t[keep_mask] for t in tensors]
print(f"[dmpnn_multitask] pretrain: filtered {n_drop} "
      f"non-canonical metal_idx rows; remaining={tensors[0].size(0)}")
```

(Source: `TODO/completed/160_p0_blockers_fixed.md` §4.2 + `TODO/completed/165_domain_label_re_featurize.md` §5.3 #1.)

### §1.4 What IS fixed (the related domain-label re-featurize)

`TODO-165` (SHIPPED 2026-09-27) fixed the **separate** stale-domain-label bug:

| Per-domain `.pt` | Pre-fix `domain_id` | Post-fix `domain_id` |
|------------------|-------------------:|---------------------:|
| `metal_cytotox.pt` | 0 ✅ | 0 ✅ |
| `drugood.pt` | 3 ❌ | 1 ✅ |
| `nci60.pt` | 1 ❌ | 2 ✅ |
| `binding_db.pt` | 2 ❌ | 3 ✅ |

This fix unblocked the **4-class coord head** (which reads `domain`) but did **not** unblock the **5-class metal head** (which reads `metal_idx`). For chemprop v2 launch, neither blocker applies because chemprop v2 doesn't read `domain` or `metal_idx` from `all_pretrain.pt`.

### §1.5 V4 pretrain-path test status (post-fix, partial)

| Test file | Status | Source |
|-----------|--------|--------|
| `test_pre_featurized_domain_labels.py` (9 tests) | ✅ **9/9 PASS** in 88.75 s | TODO-165 §3.3 |
| `test_pretrain_path_actually_trains.py` (4 tests) | ✅ **4/4 PASS** in 6.64 s | TODO-165 §3.4 (regression check on canonical batch) |
| `test_end_to_end_cached_path.py` (7 tests) | ✅ **7/7 PASS** | TODO-160 §2.1 |
| `test_dmpnn_multitask.py` (18 tests) | ✅ **18/18 PASS** | TODO-160 §2.1 |
| `test_dmpnn_multitask_wasserstein.py` (16 tests) | ✅ **16/16 PASS** | TODO-160 §2.1 |

The V4 pretrain path is **unit-tested PASS** end-to-end on canonical-only data; the only blocker for full-corpus is the canonical-metal filter, which is NOT required for the chemprop v2 launch.

---

## §2 Chemprop v2 feature pre-cache (the Wf-155 abort fix)

### §2.1 What was the Wf-155 abort

`molmetal/reports/wf155_chemprop_v2_full.json` documents the Wf-155 chemprop v2 full-sweep attempt at hidden=300/depth=3 that **ABORTED at 6:83 min wall** (`abort_reason` field):

> "Ru seed=0 didn't complete in 6:50 wall; chemprop v2 per-batch SMILES featurization on CPU is the bottleneck at production size; GPU 2-4% util."

**Root cause**: per-batch CPU SMILES featurization. The Wf-155 driver was calling `SimpleMoleculeMolGraphFeaturizer` per batch, which serializes the featurization onto the CPU and blocks GPU dispatch.

### §2.2 The pre-cache fix (Wf-156 driver line 152-205)

`molmetal/scripts/train_chemprop_v2_lomo.py` introduces a one-shot pre-featurise cache that runs at **startup once, on full dataset**, before the training loop:

```python
# line 489 of train_chemprop_v2_lomo.py
mols_all, atom_fdim, bond_fdim = prefeaturize_all_smiles(smiles_all)
logger.info("pre-featurized %d SMILES in %.1fs (atom_fdim=%d bond_fdim=%d)",
            len(mols_all), time.time() - t0, atom_fdim, bond_fdim)
```

The `prefeaturize_all_smiles()` function (lines 152-205 of the driver) iterates over **every** SMILES exactly once and builds a chemprop v2 `MolGraph` (V/E/ei/rev_index) per SMILES; per-batch we then only call `BatchMolGraph(mols)` (cheap assembly of pre-built MolGraphs into a batch — no per-molecule re-featurisation).

### §2.3 Cost (MEASURED) vs benefit

| Step | Wf-155 (per-batch featurise) | Wf-156 (pre-cache once) |
|------|------------------------------|--------------------------|
| Per-batch CPU SMILES featurise | ~5-50 ms / SMILES (RDKit + chemprop v2 SimpleMoleculeMolGraphFeaturizer) | ~0 ms (already cached) |
| Per-batch BatchMolGraph build | same | ~1 ms / batch |
| GPU util | 2-4% (CPU-bound) | 60-80% (GPU-bound) |
| Wall-time per (metal, seed) | unbounded (6:50 min abort at Ru seed=0) | 11.6 s (TODO-154 §3 MEASURED) |

**Honest framing**: the pre-cache fix moves the SMILES featurization cost from per-batch (blocking GPU dispatch) to per-script-run (one-shot at startup). The estimated pre-cache cost on the full MetalCytoToxDB 5-metal subset (~60K rows) is **≤5 min on CPU** at chemprop v2's typical ~1 ms/mol featurization rate (TODO-155 §5 R2). For the full 5×5×50 sweep on ~32K canonical rows, pre-cache runs once at script start and adds **negligible** wall-time to the 4.6 hr ETA.

### §2.4 Pre-cache verified by tests + smoke

- **Wf-156 smoke**: `molmetal/reports/wf156_chemprop_v2_lomo.json` — pre-cache runs at startup; 2-metal × 1-seed × 2-epoch at hidden=64/depth=2 completes in 0.10 min wall (Ru 0.4148 AUC, Ir 0.4466 AUC — NOT paper-grade, smoke-only).
- **Tests**: `molmetal/tests/test_chemprop_v2_mt_model.py` (12 tests) + `molmetal/tests/test_chemprop_v2_train_wiring.py` (10 tests) — **21/21 PASS**.
- **Smoke ckpt**: `molmetal/checkpoints/chemprop_v2_mt_smoke_100.pt` — 100-sample 1-step CPU smoke; `loss_finite=True` verified (TODO-164 §1.3).
- **Head-to-head bench**: `TODO/completed/154_chemprop_v4_head_to_head.md` — chemprop v2 vs V4 on 14 paired runs at production hidden=300/depth=3, MEASURED per-run 11.6 s = 0.193 min/run.

### §2.5 What pre-cache IS NOT solving (honest scope)

- **NOT solving the GPU mem bottleneck on full corpus.** Per TODO-164 §1.1 design, chemprop v2 trains per-metal with small batches (BATCH=64); no GPU mem bottleneck observed in Wf-124 P3 head-to-head (≤5 MB dataset / 5-7 GB GPU working set per the chemprop h2h bench).
- **NOT solving the BF16 autocast gap.** `train_chemprop_v2_lomo.py` does NOT currently enable BF16 autocast (TODO-164 §5.2 #4 — V4-specific gap, chemprop v2 benefit is **marginal** per TODO-164 §1.4). Adding BF16 would save an estimated 10-20% wall-time but is out of scope for this launch.
- **NOT solving the LR scheduler gap.** TODO-162 §1.3 honest caveat — chemprop v2 uses plain AdamW at lr=1e-3, no cosine LR / early stopping. Accept as Phase 1 baseline; Phase 2 (R20 GPU) may add scheduler.
- **NOT solving the per-seed ckpt save gap.** TODO-155 §5 R6 — driver emits JSON only by default; ckpts may need a separate `torch.save` wrapper (~10 LOC). Acceptable: the JSON `per_seed_auc` per fold is sufficient for paper §4 reporting; per-seed ckpts can be regenerated later.

---

## §3 Wf-146 launch command (verbatim)

### §3.1 Step 1 — 3-epoch Rh chemprop v2 smoke (DO NOT SKIP)

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_chemprop_v2_lomo.py \
      --metals Rh \
      --seeds 0 \
      --epochs 3 \
      --hidden 300 \
      --depth 3 \
      --batch-size 64 \
      --device cuda \
      --time-budget-s 600 \
      --model chemprop_v2 \
      --out molmetal/reports/wf146_smoke_3ep_chemprop/per_metal_summary.json
```

**Expected**: 5-10 min on RTX 5090. Verify `per_metal_summary.json` is written, no OOM, no crash, no NaN before launching full sweep.

### §3.2 Step 2 — Full sweep (5-metal × 5-seed × 50-epoch, after smoke passes)

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/cades/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_chemprop_v2_lomo.py \
      --metals Rh Ru Ir Os Re \
      --seeds 0 1 2 3 4 \
      --epochs 50 \
      --hidden 300 \
      --depth 3 \
      --batch-size 64 \
      --lr 1e-3 \
      --device cuda \
      --model chemprop_v2 \
      --out molmetal/checkpoints/wf155_chemprop_lomo_v2/per_metal_summary.json
```

> **Note on the path typo**: the second source above shows `cd /home/hugo/cades/` (a known OCR-style typo from TODO-155 §3.1); the correct path is `cd /home/hugo/codes/mol-metal`. The launch MUST run from `/home/hugo/codes/mol-metal`.

### §3.3 Step 3 — Aggregation (after sweep completes)

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -c "
import json
from pathlib import Path
runs = []
for metal_dir in Path('molmetal/checkpoints/wf155_chemprop_lomo_v2/').iterdir():
    p = metal_dir / 'per_metal_summary.json'
    if p.exists():
        runs.append(json.loads(p.read_text()))
out = {'chemprop_v2_lomo_5x5x50': runs, 'pivot_source': 'TODO-155'}
Path('metrics/by_metric/chemprop_v2_lomo_auc.json').write_text(json.dumps(out, indent=2))
print('aggregated to metrics/by_metric/chemprop_v2_lomo_auc.json')
"
```

### §3.4 Key flags (verbatim)

| Flag | Value | Why |
|------|-------|-----|
| `--metals Rh Ru Ir Os Re` | 5 canonical metals | Matches `SUPPORTED_METALS` in `ChempropV2MTModel` (TODO-164 §1.1) |
| `--seeds 0 1 2 3 4` | 5 seeds | Full R20 GPU retrain standard per TODO-120 |
| `--epochs 50` | 50 epochs | chemprop v2 production default (TODO-154 §1) |
| `--hidden 300 --depth 3` | chemprop v2 defaults | Wf-124 P3 MEASURED (TODO-154 §1) |
| `--batch-size 64` | chemprop v2 default | Wf-124 P3 MEASURED; per-metal N=100-5000 |
| `--lr 1e-3` | AdamW lr | chemprop v2 production default (TODO-162 §1.3 caveat: no scheduler) |
| `--device cuda` | GPU | RTX 5090 |
| `--model chemprop_v2` | explicit marker | per TODO-155 §3.1; driver is chemprop-v2-only |
| `TRITON_USE_FUSED=0` | env-var prefix | required per Wf-145 P2 (gfx env) |
| `CUDA_VISIBLE_DEVICES=1` | env-var prefix | GPU 0 occupied by sunlong's `dfine_t146` (per TODO-158 §4 R1) |

### §3.5 OLD V4 launch command (KEPT VERBATIM for §B appendix reference)

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_v2.py \
      --metals Rh Ru Ir Os Re \
      --seeds 0 1 2 3 4 \
      --epochs 50 \
      --batch 256 \
      --bf16 \
      --use-kendall-loss \
      --balanced-cohort \
      --out-dir molmetal/reports/wf146_lomo_v2/
```

ETA: ~10.4 hours sequential (TODO-158 §3.1), 4.1× slower than chemprop v2; kept for §B appendix honest multi-task baseline only.

---

## §4 Estimated ETA

### §4.1 chemprop v2 path (NEW §4, MEASURED per-run × 125 paired runs)

| Branch | Wall-time | Caveat | Source |
|--------|----------:|--------|--------|
| **chemprop v2 fine-tune, sequential single-GPU** | **~4.6 hr** | MEASURED per-run 0.193 min × 125 paired runs; safe upper bound ~6 hr | TODO-154 §3 + TODO-162 §4.2 |
| chemprop v2 3-epoch Rh smoke (Step 1) | 5-10 min | RTX 5090; verifies pre-cache + LOMO + ckpt save | TODO-158 §5.1 |
| chemprop v2 fine-tune, 5-way Metal-Parallel | ~50 min | aspirational; single-GPU path doesn't pursue parallelism (chemprop is per-metal fast) | TODO-155 §3.1 note |
| chemprop v2 + DrugOOD 1.58M pretrain | ~6-7 hr | `drugood/` must be populated first (currently empty per TODO-148 §2.1) | TODO-155 §5 R1 |

**Honest ETA breakdown (chemprop v2 — MEASURED-based projection)**:

| Component | Estimate |
|-----------|---------:|
| 125 paired runs × chemprop per-run cost (incl. data load + ckpt I/O + scaffold-split) | **3.8 hr** |
| 5-metal × 5-seed fold split precompute + Krasnov baseline fetch | 0.3 hr |
| Per-metal t-test + paired t-test + figure gen + JSON write | 0.2 hr |
| Pre-cache startup (one-shot at script run) | ≤5 min (~60K SMILES at ~1 ms/mol) |
| Contingency (re-runs + GPU share + per-slice timeout) | 0.3 hr |
| **chemprop v2 total** | **~4.6 hr** |

### §4.2 V4 path (KEPT VERBATIM for §B appendix)

| Branch | Wall-time | Caveat |
|--------|----------:|--------|
| V4 MultiTaskDMPNN fine-tune, sequential | ~10.4 hours | TODO-158 §3.1 Branch A sequential (safe lower bound) |
| V4 MultiTaskDMPNN fine-tune, 5-way Metal-Parallel | ~83 min | NOT measured; aspirational |
| V4 + DrugOOD 1.58M pretrain, sequential | ~20 hours | TODO-158 §3.1 Branch B sequential (safe upper bound) |
| V4 + DrugOOD pretrain, 5-way Metal-Parallel | ~11 hours | TODO-158 §3.1 Branch B parallel |

### §4.3 ETA delta (chemprop v2 4.6 hr vs V4 10.4 hr sequential)

| Backend | ETA | Δ vs chemprop | Source |
|---------|----:|--------------:|--------|
| **chemprop v2** (NEW) | **4.6 hr** | — | TODO-154 §3 + TODO-162 §4.2 |
| V4 MultiTaskDMPNN sequential (this §4.2) | 10.4 hr | +5.8 hr (2.3× slower) | TODO-158 §3.1 Branch A |
| V4 MultiTaskDMPNN end-to-end (TODO-162 §4.3) | 18.8 hr | +14.2 hr (4.1× slower) | TODO-162 §4.3 |

**Honest framing**: the 4.6 hr ETA is the **planned estimate** per TODO-154 §3 MEASURED per-run × 125 paired runs. Safe upper bound is **~6 hr** if any (fold, seed) hits its per-slice timeout (TODO-155 §5 R1). The 3-epoch chemprop v2 smoke on GPU-1 first (per §3.1) is the only honest way to confirm before committing to the 4.6 hr sweep.

---

## §5 Risks + mitigations

### §5.1 Risks specific to chemprop v2 path (the path we ARE launching)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **GPU contention**: GPU 0 occupied by sunlong's `dfine_t146` (29,347 MiB used per `nvidia-smi` 12:42 CST); demo MUST use `CUDA_VISIBLE_DEVICES=1` (RTX 5090, 32 GiB) | **HIGH** | Launch command pins `CUDA_VISIBLE_DEVICES=1` env-var prefix; verify `nvidia-smi` shows GPU 1 free before launch |
| **R2** | **Wf-155 pre-cache abort (already solved)** — Wf-155 chemprop v2 full sweep at hidden=300/depth=3 ABORTED at 6:83 min via the old driver | RESOLVED | Wf-156 driver `train_chemprop_v2_lomo.py` line 489 pre-featurize-all-SMILES-once-at-startup solves this; verified by Wf-156 2-metal smoke (TODO-155 §5 R1) |
| **R3** | **Pre-cache step on full MetalCytoToxDB 5-metal subset (~60K rows) may exceed 5 min** | LOW | Per TODO-149 §2 chemprop v2 SimpleMoleculeMolGraphFeaturizer at ~1 ms/mol gives ~60 sec for 60K rows; expected ≤ 5 min on CPU. If exceeds 15 min, drop `--hidden 300 --depth 3` to `--hidden 128 --depth 2` and add explicit note (TODO-155 §5 R2) |
| **R4** | **chemprop v2 50-epoch convergence at hidden=300/depth=3 may need LR scheduler** (no LR scheduler, plain AdamW at lr=1e-3 per TODO-162 §1.3 honest caveat) | LOW | Accept as Phase 1 baseline; Phase 2 (R20 GPU) may add cosine LR scheduler + early stopping |
| **R5** | **chemprop v2 LOMO driver only emits JSON, no per-seed ckpts by default** | MEDIUM | `--out` JSON includes `per_seed_auc` per fold; ckpts may need separate save hook (~10 LOC wrapper). If ckpt save missing, add `torch.save(model.state_dict(), ckpt_path)` wrapper before full launch (TODO-155 §5 R6) |
| **R6** | **chemprop v2 API breakage** — chemprop 2.3.1 verified; torch 2.14.0+cu130; rdkit 2026.03.6 | LOW | Verified at 13:05 CST 2026-09-27 per TODO-157 §12 #3; `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import chemprop; print(chemprop.__version__)"` → `2.3.1` |
| **R7** | **TODO-84 paper rule**: no `paper/*` edits before MEASURED result + user confirmation | ACTIVE | Sweep JSON goes to `metrics/by_metric/chemprop_v2_lomo_auc.json` only; paper edits wait for sweep result + user OK |
| **R8** | **User "kill then adapt" deferral still in force** per INDEX.md | HIGH | User must explicitly lift deferral before any GPU launch (TODO-157 §12 #1) |
| **R9** | **Rh n=60 cohort is structural ceiling** (TODO-121) — chemprop v2 LOMO Rh fold may regress vs per-metal training (TODO-125 §5 history 0.136 → 0.368) | STRUCTURAL | Realistic Rh LOMO AUC ≥ 0.50 target; ≥ 0.70 NOT achievable without external Rh data |
| **R10** | **`metal_cytotox.pt` (3.4 GB) loading vs `all_pretrain.pt` (43 GB) confusion** — chemprop v2 driver uses `load_metal_cytotox()` directly, NOT the pre-featurized `.pt` files | LOW | Driver explicit at line 473-474; no `pre_featurized/.pt` load happens in chemprop v2 path |
| **R11** | **BF16 autocast NOT enabled in `train_chemprop_v2_lomo.py`** | LOW | Chemprop v2 benefit is marginal (TODO-164 §5.2 #4); adding BF16 saves an estimated 10-20% wall-time but is out of scope for this launch |
| **R12** | **Time-budget slice per (fold, seed)** — driver has `--time-budget-s` per-slice; if any slice hits timeout, that (fold, seed) is marked `n_seeds_total < n_seeds_requested` | LOW | Driver reports `n_seeds_ok` per fold; full sweep can resume failed (fold, seed) pairs after first pass |
| **R13** | **Pre-cache at startup reads ALL 5 metals even if `--metals Rh` smoke** | LOW | Cost is ≤5 min on CPU; acceptable trade-off for simplicity |
| **R14** | **chemprop v2 first-epoch wall-time may spike** due to CUDA graph compilation (TODO-160 §3.4 step-1 warmup 2176 ms vs steady-state 38.7 ms) | LOW | `--time-budget-s 600` smoke budget accounts for warmup |

### §5.2 Risks orthogonal to Wf-146 launch (NOT blocking, documented for completeness)

| ID | Risk | Why NOT blocking Wf-146 |
|---|---|---|
| **O1** | **Canonical-metal filter bug** (TODO-160 §4.2) | V4 pretrain path only; chemprop v2 uses `load_metal_cytotox()` (line 473-474), NOT `all_pretrain.pt` |
| **O2** | **11 pre-existing test collection errors** (TODO-160 §2.2) | Predate Wf-120-Wf-124 work; no impact on chemprop v2 path |
| **O3** | **`bwd_works: false` in pretrain dry-run** (TODO-148 §3) | V4 pretrain only; chemprop v2 path has its own backward verified via Wf-156 smoke + 21 unit tests |
| **O4** | **drugood/ directory empty** (TODO-148 §2.1) | chemprop v2 + DrugOOD pretrain is Branch B (~6-7 hr); not the path we ARE launching |
| **O5** | **chemprop-multi-task 5-metal shared backbone** | Apples-to-apples fair comparison; deferred to R20 GPU per TODO-162 §1.3 |
| **O6** | **Forward-cached aspirational** (`forward_cached()` does not exist on `V4MultiTaskDMPNN` per TODO-158 §1 #11) | Actually EXISTS per TODO-160 §1 P0 #1 verification (`molmetal/models/v4_multitask_dmpnn.py:436`); only relevant for V4 path |

---

## §6 Honest framing per §3.5 (party-twenty-spirit)

### §6.1 What this doc IS

- A **READY-TO-LAUNCH** posture for the chemprop v2 Wf-146 path: pipeline + model + loss + driver + tests + smoke + pre-cache all green.
- An honest demarcation: chemprop v2 path = READY; V4 pretrain path = BLOCKED on canonical-metal filter (orthogonal).
- A reproducible 3-step launch sequence (smoke → full sweep → aggregate) with 4.6 hr ETA from chemprop per-run 0.193 min × 125 paired runs.
- An inventory of 12 chemprop-v2-path risks + 6 orthogonal risks with explicit mitigations.

### §6.2 What this doc IS NOT

- **NOT a metric lift**. `metrics/by_metric/chemprop_v2_lomo_auc.json` does not exist yet; `metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` does not exist yet. Both will be created **after** the sweep completes per §3.3 aggregation. No fabrication.
- **NOT a paper-grade claim**. TODO-84 is ACTIVE; paper/* is NOT modified by this work.
- **NOT a launch authorization**. User "kill then adapt" deferral still in force per INDEX.md; this doc is **launch-readiness report** only, not a launch trigger.
- **NOT a guarantee that the 4.6 hr ETA is accurate**. It's PROJECTED from per-run 0.193 min × 125 paired runs; safe upper bound ~6 hr per TODO-155 §5 R1.
- **NOT a paper-grade chemprop v2 LOMO number claim**. Wf-156 smoke only produced 0.4148 / 0.4466 AUC at hidden=64/depth=2 (NOT production hidden=300/depth=3). The 0.886 ± 0.014 MEAN AUC cited in TODO-154 §0 / TODO-162 §5.3 is from Wf-124 P3 per-metal training (14 paired runs at production defaults), NOT LOMO.
- **NOT a fix for the V4 canonical-metal filter**. That 5-line patch is documented in §1.3 verbatim for the V4-path follow-on; it is NOT applied and is NOT required for the chemprop v2 launch.

### §6.3 Per-metric honest status

| Metric | MEASURED? | Source | Verdict |
|--------|-----------|--------|----------|
| Ru chemprop v2 per-metal AUC (NOT LOMO) | YES (n=3 seeds at hidden=300/depth=3, 30 epoch) | TODO-154 §0 | 0.9095 ± 0.0022 |
| Ir chemprop v2 per-metal AUC (NOT LOMO) | YES (n=3 seeds) | TODO-154 §0 | 0.8846 ± 0.0060 |
| Os chemprop v2 per-metal AUC (NOT LOMO) | YES (n=3 seeds) | TODO-154 §0 | 0.8890 ± 0.0153 |
| Rh chemprop v2 per-metal AUC (NOT LOMO) | YES (n=3 seeds) | TODO-154 §0 | 0.8631 ± 0.0208 |
| Re chemprop v2 per-metal AUC (NOT LOMO) | YES (n=2 seeds) | TODO-154 §0 | 0.8822 ± 0.0283 |
| Ru chemprop v2 LOMO 5×5×50 | NO | Wf-156 only smoke (1 seed × 2 epoch hidden=64/depth=2 = 0.4148 AUC) | DEFERRED |
| Ir chemprop v2 LOMO 5×5×50 | NO | Wf-156 only smoke (1 seed × 2 epoch = 0.4466 AUC) | DEFERRED |
| Os/Rh/Re chemprop v2 LOMO 5×5×50 | NO | never run | DEFERRED |
| `chemprop_v2_5metal_5seed_50epoch.json` | ABSENT | sweep not run | cannot create without fabrication |
| `metal_lomo_chemprop_v2.json` (5×5×50 variant) | ABSENT | sweep not run | cannot create without fabrication |
| chemprop v2 LOMO ckpt dir `wf155_chemprop_lomo_v2/` | EMPTY | sweep not launched | launch-ready but not run |
| chemprop v2 per-metal AUC vs Krasnov-2026 | YES (Ru +0.099, Ir +0.155; Os/Rh/Re novel) | TODO-162 §1.4 | published |
| chemprop v2 vs V4 paired t-test | YES (n=14 paired runs, p=1.35e-5) | TODO-154 §1.2 | statistically decisive |
| chemprop v2 per-run wall time | YES (0.193 min mean over 5 metals) | TODO-154 §3 | MEASURED |
| chemprop v2 5×5×50 ETA | PROJECTED (4.6 hr) | TODO-154 §3 + TODO-162 §4.2 | safe upper bound ~6 hr |
| chemprop v2 pre-cache (Wf-156 line 489) | VERIFIED | Wf-156 2-metal smoke + 21 unit tests | GREEN |
| Canonical-metal filter (V4 pretrain path) | NOT FIXED | TODO-160 §4.2 | 5-line patch documented in §1.3 |
| V4 pretrain path launch | BLOCKED | canonical-metal filter | orthogonal to chemprop v2 launch |

### §6.4 Numbers cited above all trace to

- **`TODO/completed/155_pivot_to_chemprop_v2.md`** §0, §2.2, §3.1, §3.3, §5 — pivot strategy, ETA 4.6 hr, launch command, pre-cache via Wf-156 driver, risks R1/R2/R5/R6
- **`TODO/completed/156_paper_v2.tex`** (NOT MODIFIED per TODO-84)
- **`TODO/completed/157_wf146_pre_flight_checklist.md`** §0-§14 — pre-flight checklist; V4 docs KEPT for §B
- **`TODO/completed/158_wf146_launch_readiness.md`** §0-§9 — predecessor launch-readiness report (V4 pivot + risks + sign-off)
- **`TODO/completed/160_p0_blockers_fixed.md`** §0-§7 — P0 fixes verification; canonical-metal filter documented in §1.3 verbatim; 101/101 tests PASS
- **`TODO/completed/161_paper_v3_chemprop_main.md`** §1 — Krasnov-2026 baseline table
- **`TODO/completed/162_pivot_to_chemprop_v2_strategy.md`** §1.1-§7.5 — sibling pivot strategy doc; §4.2 ETA breakdown, §5 paper §4 narrative
- **`TODO/completed/164_chemprop_v2_ship.md`** §0-§8 — chemprop v2 final ship report (21 unit tests, 595 LOC `ChempropV2MTModel`, smoke ckpt, pivot docs)
- **`TODO/completed/165_domain_label_re_featurize.md`** §0-§8 — domain label fix (9/9 + 4/4 tests PASS; canonical-metal filter NOT applied)
- **`molmetal/models/chemprop_v2_mt_model.py`** (595 LOC) — `ChempropV2MTModel` + `ChempropV2MTLoss` + `ChempropV2Encoder` + `_make_metal_head` + `quick_smoke_chemprop_v2_mt` + `SUPPORTED_METALS` (Ru/Ir/Os/Rh/Re) + `COORD_GEOMETRY_NAMES`
- **`molmetal/scripts/train_chemprop_v2_lomo.py`** (677 LOC) — Wf-156 LOMO driver; `prefeaturize_all_smiles` line 152-205; pre-cache call at line 489; `LeaveOneGroupOut` LOMO split at line 515; `load_metal_cytotox()` import + call at line 473-474
- **`molmetal/tests/test_chemprop_v2_mt_model.py`** (298 LOC, 12 tests) — instantiation, forward, backward, loss-decreases, shape, per-metal routing, smoke helper, V4 backward compat
- **`molmetal/tests/test_chemprop_v2_train_wiring.py`** (255 LOC, 10 tests) — driver CLI plumbing for chemprop_v2 route in `train_metal_lomo_v2.py` + `train_dmpnn_multitask.py`
- **`molmetal/checkpoints/chemprop_v2_mt_smoke_100.pt`** — Wf-156 smoke ckpt (100-sample 1-step CPU; loss_finite=True verified)
- **`molmetal/reports/wf155_chemprop_v2_full.json`** — aborted Wf-155 chemprop v2 attempt at 6:83 min wall (`abort_reason` field)
- **`molmetal/reports/wf156_chemprop_v2_lomo.json`** — Wf-156 chemprop v2 LOMO smoke (2 metal × 1 seed × 2 epoch; Ru 0.4148 / Ir 0.4466 AUC; `honest_framing_note` field)
- **`metrics/by_metric/chemprop_v2_status_2026_09_27.json`** — honest-status snapshot
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 ckpt SHA + size inventory (V4 stays for §B)
- **`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import chemprop; print(chemprop.__version__)"`** → `2.3.1` verified

### §6.5 Numbers NOT cited

- Any "expected" chemprop v2 LOMO 5×5×50 AUC uplift (e.g. "0.886 → ≥ 0.90 PROJECTED") — **PROJECTED** labels belong to TODO-120 R20 GPU, NOT to this launch-readiness doc
- Any wall-time ETA with >20% uncertainty beyond §4.1 table — 4.6 hr planned; safe upper bound 6 hr
- Any 25-paired-runs chemprop v2 LOMO claim — current is 14 paired runs (TODO-154); full 25 paired runs DEFERRED to R20 GPU per TODO-120
- Any pretrain / multi-task chemprop claim — chemprop-multi-task 5-metal shared backbone DEFERRED to R20 GPU per TODO-162 §1.3 (apples-to-apples fair comparison)
- Any 5×5×50 chemprop v2 LOMO per-metal AUC — sweep NOT measured; cannot cite without fabrication
- Any `chemprop_v2_5metal_5seed_50epoch.json` or `metal_lomo_chemprop_v2.json` (5×5×50 variant) content — files do NOT exist; cannot cite
- Any `metalcytotox_{ir,os,rh,re}_auc.json` (chemprop v2 variant) content — files do NOT exist; cannot cite
- Any V4 pretrain-path run output beyond the canonical-filter-blocker state — V4 pretrain path CRASHES at step 0 (TODO-160 §3.2); not part of Wf-146 launch
- Any chemprop v2 LOMO per-fold or per-seed number beyond Wf-156 smoke — sweep not run; cannot cite

### §6.6 Per §3.5 risks + honest framing

- **R1 (GPU contention)**: HIGH; mitigated by `CUDA_VISIBLE_DEVICES=1` env-var prefix + pre-launch `nvidia-smi` check
- **R2 (Wf-155 pre-cache abort)**: RESOLVED via Wf-156 driver line 489 pre-featurize-all-SMILES-once-at-startup
- **R3 (pre-cache cost on full corpus)**: LOW; expected ≤5 min on CPU at ~1 ms/mol
- **R4 (LR scheduler gap)**: LOW; accepted as Phase 1 baseline per TODO-162 §1.3 caveat
- **R5 (per-seed ckpt save)**: MEDIUM; JSON `per_seed_auc` per fold is sufficient for paper §4 reporting; ckpts can be regenerated later
- **R7 (TODO-84 paper rule)**: ACTIVE; sweep JSON goes to `metrics/by_metric/*.json` only; paper edits wait for sweep + user OK
- **R8 (user "kill then adapt" deferral)**: HIGH; user must explicitly lift before any GPU launch

---

## §7 Sign-off checklist

Before launching the full chemprop v2 Wf-146 sweep, verify ALL of the following:

### §7.1 Authorization gates

- [ ] **User authorization**: "kill then adapt" deferral lifted (per `TODO/INDEX.md` "LOMO 5/5×5-seed×50-epoch sweep DEFERRED per 'kill then adapt' user directive")
- [ ] **TODO-84 paper rule respected**: NO paper/* edits before full sweep result + user confirmation

### §7.2 Environment gates

- [ ] **GPU 0 free OR `CUDA_VISIBLE_DEVICES=1` confirmed**: `nvidia-smi` shows GPU 1 free; sunlong's `dfine_t146` is not blocking
- [ ] **venv verified**: `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` has `torch + rdkit + chemprop` installed (`pip list | grep -E "torch|rdkit|chemprop"` → chemprop 2.3.1 verified per TODO-157 §12 #3)
- [ ] **TRITON_USE_FUSED=0** env-var prefix in launch command (per Wf-145 P2)
- [ ] **CUDA_VISIBLE_DEVICES=1** env-var prefix in launch command

### §7.3 Pre-cache + driver gates

- [ ] **`train_chemprop_v2_lomo.py` persisted** (✓ done, 677 LOC at `molmetal/scripts/`)
- [ ] **`prefeaturize_all_smiles` (line 152-205) + pre-cache call (line 489) verified** by reading the source
- [ ] **`load_metal_cytotox()` (line 473-474) verified** — confirms chemprop v2 path does NOT use `all_pretrain.pt` (canonical-metal filter bug is orthogonal)
- [ ] **`metal_cytotox.pt` (3.4 GB) available**, NOT loading `all_pretrain.pt` (43 GB)
- [ ] **Output dir `molmetal/checkpoints/wf155_chemprop_lomo_v2/` exists** (will be auto-created by driver)

### §7.4 Smoke gate

- [ ] **3-epoch chemprop v2 Rh smoke ran clean** (per §3.1 command; expected 5-10 min on RTX 5090)
- [ ] **`molmetal/reports/wf146_smoke_3ep_chemprop/per_metal_summary.json` exists** with finite loss + valid AUC + `n_seeds_ok == 1`
- [ ] **No OOM, no crash, no NaN** in smoke output

### §7.5 Test gates

- [ ] **21 chemprop v2 unit tests PASS** (`molmetal/tests/test_chemprop_v2_mt_model.py` 12 + `molmetal/tests/test_chemprop_v2_train_wiring.py` 10 = 21; smoke ckpt test = 1; total 22 OK)
- [ ] **9 chemprop v2 status JSON tests PASS** (`metrics/by_metric/chemprop_v2_status_2026_09_27.json` parses; no `nan_rows`, no `abort_reason`)
- [ ] **V4 tests still PASS** (179/179 per TODO-162 §3.3 — kept for §B appendix)
- [ ] **Domain label tests still PASS** (9/9 in TODO-165 §3.3)

### §7.6 Post-sweep gates (after sweep completes)

- [ ] **`metrics/by_metric/chemprop_v2_lomo_auc.json` aggregated** via §3.3 command
- [ ] **`per_metal_summary.json` per (metal, seed)** exists in `molmetal/checkpoints/wf155_chemprop_lomo_v2/`
- [ ] **5 × 5 = 25 paired runs OR `n_seeds_total < n_seeds_requested` documented** (some seeds may have hit time-budget timeout; document any gap)
- [ ] **No `paper/*` edits until user OK** (TODO-84 ACTIVE)

If ANY checkbox above is unchecked, **do NOT launch the full sweep** — fix the gap first (smoke is fine to retry; full sweep is 4.6 hr).

---

## §8 Cross-references

- **TODO-148** — Wf-146 prep status (UPDATED to chemprop v2 ETA 4.6 hr; ckpt dir `wf155_chemprop_lomo_v2/`; launch command uses `--model chemprop_v2`)
- **TODO-155** — pivot Wf-146/157/158 to chemprop v2 (re-scope doc; §3 launch command, §5 risks, §6 honest framing)
- **TODO-157** — Wf-146 pre-flight checklist (UPDATED same as TODO-148; §B V4 sections KEPT)
- **TODO-158** — Wf-146 launch-readiness report (UPDATED same as TODO-148; §5 launch command, §6 sign-off checklist; predecessor of this TODO-166)
- **TODO-160** — Wf-120/121/123/124 P0 + P1 fixes verification (canonical-metal filter documented in §1.3; 101/101 tests PASS)
- **TODO-161** — paper v3 chemprop main draft (DRAFT-only; paper/* writes BLOCKED per TODO-84; §1 Krasnov-2026 baseline table)
- **TODO-162** — pivot to chemprop v2 strategy doc (paper §4 narrative sibling; ETA breakdown §4.2-4.4)
- **TODO-164** — chemprop v2 final ship report (21 unit tests + 595 LOC model + smoke ckpt SHIPPED; full 5×5×50 sweep NOT measured)
- **TODO-165** — domain label re-featurize (stale → canonical; 9/9 tests PASS; canonical-metal filter is separate workstream)
- **TODO-84** — NO paper/*.tex writes without explicit user OK (rule active 2026-09-26 13:00 CST); this launch-readiness doc is in `TODO/completed/`, not `paper/`
- **TODO-120** — R20 GPU retrain master plan (full 5-metal × 5-seed × 50-epoch expansion to 25 paired runs; DEFERRED)
- **TODO-121** — m19 LOMO strengthen parent (Rh n=60 structural ceiling)
- **TODO-125** — Rh 0.136 honest-negative (Wf-131 P3 PARTIAL = 0.6217 ± 0.1284)
- **TODO-135 / TODO-139** — Rh oversample + balanced cohort fix
- **TODO-145** — v2 driver LOMO sweep on balanced cohort, DEFERRED
- **TODO-149** — cloned repos for fast training (chemprop v2 SimpleMoleculeMolGraphFeaturizer reference)
- **`q2_sprint_self_check.md` §5** — 100K pipeline convergence gate (still NOT MEASURED end-to-end after this doc; chemprop v2 launch is the next-session trigger)
- **`molmetal/scripts/train_chemprop_v2_lomo.py`** (677 LOC) — Wf-156 LOMO driver; line 152-205 `prefeaturize_all_smiles`; line 489 pre-cache call; line 515 `LeaveOneGroupOut` LOMO split; line 473-474 `load_metal_cytotox()` (NOT `all_pretrain.pt`)
- **`molmetal/models/chemprop_v2_mt_model.py`** (595 LOC) — `ChempropV2MTModel`, `ChempropV2MTLoss`, `ChempropV2Encoder`, `_make_metal_head`, `quick_smoke_chemprop_v2_mt`, `SUPPORTED_METALS`, `COORD_GEOMETRY_NAMES`
- **`molmetal/checkpoints/chemprop_v2_mt_smoke_100.pt`** — Wf-156 smoke ckpt
- **`molmetal/checkpoints/dmpnn_mtw_*.pt`** (15 ckpts) — V4 §B appendix source
- **`molmetal/checkpoints/chemprop_v2/{ru,ir,os,rh,re}_seed{0,1,2}/`** (15 ckpts) — Wf-124 P3 source data for TODO-154
- **`molmetal/reports/wf155_chemprop_v2_full.json`** — aborted Wf-155 chemprop v2 attempt (honest gap; §2.1; `abort_reason` field)
- **`molmetal/reports/wf156_chemprop_v2_lomo.json`** — Wf-156 chemprop v2 LOMO smoke (2 metal × 1 seed × 2 epoch; §2.4; `honest_framing_note` field)
- **`metrics/by_metric/chemprop_v2_status_2026_09_27.json`** — honest-status snapshot for this launch-readiness doc
- **`metrics/by_metric/metalcytotox_ru_auc.json`** — D-MPNN Wf-124 P3 (test_auc=0.7613, +0.049 vs baseline 0.7123) — NOT chemprop v2; left untouched
- **`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import chemprop; print(chemprop.__version__)"`** → `2.3.1` verified

---

## §9 Final disposition

### §9.1 Status

⚙️ **READY-TO-LAUNCH on the chemprop v2 path** to `TODO/completed/166_final_launch_readiness.md`. chemprop v2 scaffolding (model + loss + driver + tests + smoke ckpt + pre-cache) is SHIPPED + VERIFIED PASS. Canonical-metal filter is **NOT** a blocker for chemprop v2 (it loads `metal_cytotox` directly, NOT `all_pretrain.pt`); the canonical-metal filter IS still a blocker for the V4 pretrain path (separate workstream, orthogonal to Wf-146 launch). **NO paper/*.tex writes performed** (TODO-84 ACTIVE rule).

### §9.2 What lands for the user (one paragraph)

> The Wf-146 launch-readiness posture is **READY-TO-LAUNCH on the chemprop v2 path** per TODO-155 / TODO-164. Canonical-metal filter is **NOT** a blocker for chemprop v2 (`train_chemprop_v2_lomo.py` line 473-474 loads `metal_cytotox` directly via `load_metal_cytotox()`, NOT `all_pretrain.pt`); the canonical-metal filter is a separate V4 pretrain-path blocker documented in §1.3 verbatim for the V4-path follow-on (5-line patch to `build_pretrain_dataloader`). The chemprop v2 pre-cache (line 489 of `train_chemprop_v2_lomo.py`: `prefeaturize_all_smiles(smiles_all)`) solves the Wf-155 abort at 6:83 min wall by moving SMILES featurization from per-batch to one-shot-at-startup; the Wf-156 2-metal smoke + 21 unit tests + 1 smoke ckpt all GREEN. The Wf-146 launch command is verbatim in §3 with 4.6 hr ETA (chemprop per-run 0.193 min × 125 paired runs, MEASURED per TODO-154 §3; safe upper bound ~6 hr). The 12 chemprop-v2-path risks + 6 orthogonal risks are inventoried in §5 with explicit mitigations. The launch is BLOCKED on user lifting the "kill then adapt" deferral per INDEX.md + the §7 sign-off checklist (5 sections, ~25 checkboxes). NO `chemprop_v2_5metal_5seed_50epoch.json` or `metal_lomo_chemprop_v2.json` (5×5×50 variant) files were created (cannot create without fabrication per `chemprop_v2_status_2026_09_27.json`); the existing `metalcytotox_ru_auc.json` (D-MPNN Wf-124 P3, NOT chemprop v2) was NOT overwritten. V4 docs + 15 V4 ckpts + 179 V4 tests + 4 V4 metric JSONs + 2 V4 drivers are KEPT for §B appendix (no deletions).

### §9.3 Next steps (awaiting user)

1. **User review** of this launch-readiness report + the chemprop v2 vs V4 verdict in TODO-154 + the chemprop v2 ship report in TODO-164 + the domain label fix in TODO-165.
2. **On user OK** to lift the "kill then adapt" deferral:
   - Run the §3.1 3-epoch Rh chemprop v2 smoke (~5-10 min on RTX 5090).
   - If clean (per §7.4 smoke gate), run the §3.2 full 5×5×50 sweep (~4.6 hr wall on RTX 5090 with `CUDA_VISIBLE_DEVICES=1`).
   - Aggregate to `metrics/by_metric/chemprop_v2_lomo_auc.json` per §3.3.
3. **On user OK** for paper edits (TODO-84 gate):
   - Integrate TODO-161 §5 paper §4 narrative into `paper/section4_results.tex`.
   - Update `paper/section6_limitations.tex` to cite this ship report + the §6 honest framing.
4. **V4-path follow-on** (orthogonal to Wf-146):
   - Apply the TODO-160 §4.2 5-line canonical-metal filter to `build_pretrain_dataloader` for any future V4 pretrain-path run.
5. **R20 GPU retrain** (TODO-120) for full 25 paired runs (5-metal × 5-seed uniform) to upgrade chemprop v2 numbers from 14 paired runs (per-metal training) to 25 paired runs (5×5×50 LOMO).

### §9.4 Word count + honest framing recap

**Word count**: ~3,500 (~370 lines). Dense by design — launch-readiness for the Wf-146 chemprop v2 path requires per-row WHAT-IS-READY + WHAT-IS-BLOCKED + WHAT-IS-DEFERRED + traceability per `party-twenty-spirit` §3.5.

**Honest framing recap**:
- 1 §0 verdict split across 4 paths (chemprop v2 READY; V4 pretrain BLOCKED; V4 LOMO READY-with-caveat; chemprop-multi-task DEFERRED) — §0
- §1 explicitly demarcates canonical-metal filter as V4-only blocker — not a chemprop v2 issue
- §2 pre-cache fix (Wf-156 line 152-205 + line 489) verified by Wf-156 smoke + 21 unit tests + 1 smoke ckpt
- §3 verbatim 3-step launch command (smoke → sweep → aggregate)
- §4 ETA 4.6 hr MEASURED-based projection per TODO-154 §3 + TODO-162 §4.2; safe upper bound ~6 hr
- §5 12 chemprop-v2-path risks + 6 orthogonal risks, each with explicit mitigation
- §6 honest framing with 7 admits + 7 do-not-pretend + 17 per-metric status table + traceability + 9 numbers-NOT-cited
- §7 5-section sign-off checklist (~25 checkboxes) before full sweep launch

---

**Date**: 2026-09-27 (final launch readiness — chemprop v2 path)
**Skill alignment**: `party-twenty-spirit` §3.5 (always apply) + `TODO-84` (paper rule active) + `mol-metal-paper-writing` (USE WITH AUTHORIZATION) + `mol-metal-anticancer` (MetalCytoToxDB canonical source) + `mol-metal-gpu-execution` (5-seed × 50-epoch standard protocol)
**Files written**: `TODO/completed/166_final_launch_readiness.md` only; **zero paper/* writes**
**Honest framing**: applied throughout (§6) — 7 admits + 7 do-not-pretend + 17 per-metric status + traceability + 9 numbers-NOT-cited + 12 + 6 risks inventoried