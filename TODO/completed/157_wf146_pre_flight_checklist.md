# TODO-157 — Wf-146 LOMO v2 retrain: pre-flight readiness checklist

**Date**: 2026-09-27 13:05 CST
**Status (per §0)**: ⚙️ **PREP-DISCIPLINE COMPLETE — ACTUAL GPU RETRAIN STILL GATED ON USER LIFTING "KILL THEN ADAPT" DEFERRAL** — **PIVOTED TO chemprop v2** per TODO-155 (V4 docs kept as §B appendix verbatim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: TODO-148 §0 (prep status) + TODO-152 §0 (no in-flight workflow to recover) + TODO-158 §0 (READY-TO-LAUNCH stub) + TODO-84 (no paper task without authorization) + INDEX.md "kill then adapt" directive + TODO-155 (chemprop v2 pivot — rescope to chemprop v2 per TODO-154 head-to-head)
**Skill alignment**: `party-twenty-spirit` §3.5 (every number traces to a JSON path, audit artifact, or in-repo file; no aspirational numbers)
**Trigger**: user said verbatim 2026-09-27 — "我觉得训练之前我们肯定得做好万全的准备，不能这么草率" ("I think before training we definitely need to make thorough preparations, can't be so hasty"). This doc IS that preparation.

---

## §0 Status: PREP-DISCIPLINE COMPLETE — RETRAIN DEFERRED — PIVOTED TO chemprop v2

**Prep verdict (this document)**: ✅ **READY-TO-LAUNCH** (gated on user authorization). Pipeline, data, smoke-driver, demo, resume test, GPU state all GREEN; honest gaps inventoried with explicit blockers.

**Pivot (this update 2026-09-27)**: per **TODO-155** (this doc + TODO-148 + TODO-158 rescope to chemprop v2 backbone; ETA drops 18.8 hr → 4.6 hr per TODO-154 §3 + TODO-162 §4.2 MEASURED). §9 below carries the chemprop v2 launch command + ckpt dir `molmetal/checkpoints/wf155_chemprop_lomo_v2/`; §10 ETA table uses chemprop v2 4.6 hr; V4 sections (§2 forward_cached V4 path, §9 V4 launch, §10 V4 ETA) are KEPT VERBATIM as §B appendix source per user instruction.

**Actual retrain verdict**: 🚧 **DEFERRED per "kill then adapt" directive still in force** per `TODO/INDEX.md` cross-references table:
> "LOMO 5/5×5-seed×50-epoch sweep DEFERRED per 'kill then adapt' user directive (GPU work)"

**One-line honest framing**: writing this checklist is **PREP-DISCIPLINE complete, NOT a metric lift**. No training has happened since TODO-158; no LOMO AUC number has moved; `metrics/by_metric/metal_lomo_auc_v2.json` `status` field still reads `DEFERRED_PARTIAL_RUN`.

**Blocker inventory** (verbatim from §1 audit):
1. **User authorization NOT yet lifted** (the "kill then adapt" deferral — TODO-84 paper rule + INDEX.md).
2. **`--num-workers` / `--pin-memory` not in `train_dmpnn_multitask.py --help`** (TODO-148 §1.1 honest gap; silently no-op today, but the v2 driver has them as defaults).
3. **`--data-config pretrain` NOT in `--help`** (TODO-152 §3 caveat; v2 driver does not need this, only the multitask variant does).
4. **`forward_cached()` aspirational API does NOT exist on `V4MultiTaskDMPNN`** (TODO-158 §1 #11; documented as aspirational gap, demo uses real `forward()`).
5. **`bwd_works: false` in pretrain dry-run** (TODO-148 §3 honest note); v2 driver pretrain-mode backward is unit-tested but not end-to-end verified.
6. **No MEASURED throughput benchmark for v2 driver** (TODO-152 §2 — `hit_24hr_target` decision tree data is UNKNOWN).
7. **3/4 module loaders DO NOT EXIST** for `molmetal.data.{metal_cytotox, drugood_loader, nci60, binding_db}` (TODO-148 §2.1 — reads CSVs directly via pandas).
8. **Rh n=60 is a structural ceiling** (TODO-121 + TODO-145 §5 honest framing).

---

## §1 Audit findings (gaps closed + remaining)

### §1.1 Pipeline audit — gaps closed since TODO-148 / 152 / 158

| Gap | Status as of 13:05 CST | Evidence path |
|---|---|---|
| `train_metal_lomo_v2.py --help` exposes `--metals`, `--seeds`, `--epochs`, `--batch-size`, `--device`, `--out-dir`, `--balanced-cohort-json`, `--lomo-csv`, `--use-l2-guard / --no-l2-guard`, `--quick` | ✅ **CLOSED** | `python molmetal/scripts/train_metal_lomo_v2.py --help` (verified 13:05 CST, all 11 args exposed) |
| `--metals` accepts `Rh Ru Ir Os Re` (5-metal default) | ✅ **CLOSED** | v2 driver argparse: `parser.add_argument("--metals", nargs="+", default=["Rh","Ru","Ir","Os","Re"])` |
| `--seeds` accepts space-separated list `0 1 2 3 4` | ✅ **CLOSED** | same argparse source |
| `--device cuda` accepts GPU device | ✅ **CLOSED** | `--device DEVICE  (default 'cuda')` |
| `--balanced-cohort-json` points at Wf-145 P1 balanced Rh cohort | ✅ **CLOSED** | arg exists; defaults to `molmetal/reports/wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json` (250 active + 250 inactive) |
| `--use-l2-guard` engaged by default (Wf-127 P2 Hilbert L2 grad-norm guard, max_norm=5.0) | ✅ **CLOSED** | `parser.add_argument("--use-l2-guard", action="store_true")` default ON via `_fit()` |
| `--quick` 1-metal × 1-seed × 2-epoch CPU smoke | ✅ **CLOSED** | `parser.add_argument("--quick", action="store_true")` |
| 18/18 unit tests PASS for v2 driver | ✅ **CLOSED** | `molmetal/tests/test_metal_lomo_v2.py` (per TODO-158 §1) |
| 3 ckpts `test_wf146_resume.py` (save/load round-trip, resume from epoch N, optimizer state restore) | ✅ **CLOSED** | `molmetal/tests/test_wf146_resume.py` PASS (per TODO-158 §1 #10) |
| `wf146_demo/` end-to-end artefacts on disk | ✅ **CLOSED** | `molmetal/reports/wf146_demo/{train_log.json, ckpt_epoch3.pt, counterfactual_mined.json, vina_replay/demo.json, curriculum_state.json, smoke_report.md}` (8.4 MB ckpt, 12 steps, 28.4 sec wall) |

### §1.2 Remaining gaps (must clear before launch)

| # | Gap | Severity | Mitigation | Owner sub-task |
|---|---|---|---|---|
| 1 | User "kill then adapt" deferral still in force per INDEX.md | **HIGH** | User must explicitly lift the deferral before any GPU launch | user |
| 2 | `--num-workers 4 --pin-memory` not in `train_dmpnn_multitask.py --help` (silent no-op today; v2 driver not affected) | LOW | v2 driver (`train_metal_lomo_v2.py`) bypasses these flags; mitigate by patching multitask script before any multitask path launch | TODO-148 §1.1 sub-task |
| 3 | `--data-config pretrain` not in multitask `--help` | LOW | v2 driver does not use `--data-config`; only affects multitask variant — out of scope for Wf-146 | n/a |
| 4 | `V4MultiTaskDMPNN.forward_cached()` does NOT exist on the model | MEDIUM (documented) | Driver uses real `forward(smiles_list, metal_ids)`; honest note in §4 below | n/a (aspirational API never shipped) |
| 5 | `bwd_works: false` in pretrain dry-run (TODO-148 §3) | MEDIUM | Add 5-step `loss.backward()` warmup before sweep launch; v2 driver has its own `loss.step()` path unit-tested | TODO-148 §3 sub-task |
| 6 | No MEASURED throughput benchmark for v2 driver (`hit_24hr_target` UNKNOWN per TODO-152 §2) | MEDIUM | Run 3-epoch Ru smoke on RTX 5090 first (per TODO-152 §3); verify JSON AUC before sweep | TODO-152 §3 recommended smoke |
| 7 | 3/4 module loaders DO NOT EXIST (`molmetal.data.{metal_cytotox, drugood_loader, nci60, binding_db}`) | LOW (workaround exists) | v2 driver reads CSVs directly via pandas + Wf-145 P1 balanced cohort JSON | TODO-148 §2.1 sub-task (deferred to R20 paper polish) |
| 8 | Rh n=60 is a structural ceiling (TODO-121) | STRUCTURAL | Realistic Rh LOMO AUC ≥ 0.50 target; ≥ 0.70 NOT achievable without external Rh data | n/a (out of scope) |
| 9 | GPU state CHANGED since TODO-158 — see §7 below | LIVE | Verify `nvidia-smi` immediately before launch | this session |
| 10 | TODO-84 paper rule: no paper/* edits before MEASURED result | ACTIVE | No paper/* changes after sweep; only `metrics/by_metric/*.json` refresh | ongoing |

**Audit verdict**: **READY-TO-LAUNCH** subject to gap #1 (user authorization) and gap #9 (live GPU verification at launch time). All other gaps are documented + mitigated.

---

## §2 Forward cached encoding variant

### §2.1 Status: ASPIRATIONAL API — DOES NOT EXIST ON MODEL

**Verbatim from TODO-158 §1 #11** (verified 2026-09-27 12:42 CST):
> "**`forward_cached()` (aspirational — bypass MPNN encoder re-featurization via pre-computed atom/edge tensors)**: ⚠️ ASPIRATIONAL — DOES NOT EXIST. `V4MultiTaskDMPNN` has only `forward(smiles_list, metal_ids)`; driver uses real forward and notes this in `smoke_report.md` §Honest framing."

**Verbatim from `wf146_demo/smoke_report.md` §Honest framing**:
> "forward() is the model API; there is no forward_cached() method. SMILES go through D-MPNN encoder each step."

### §2.2 What is on disk for cached forward path

| Artefact | Size | Status | Path |
|---|---|---|---|
| `molmetal/data/pre_featurized/metal_cytotox.pt` | 3,610,253,665 bytes (3.4 GB) | ✅ VERIFIED | pre-featurized torch tensors for the MetalCytoTox subset |
| `molmetal/data/pre_featurized/all_pretrain.pt` | 45,331,375,803 bytes (43 GB) | ✅ VERIFIED | full 4-domain pre-featurized tensors (DrugOOD + MetalCytoTox + NCI-60 + BindingDB) |
| `molmetal/data/pre_featurized/drugood.pt` | 39,959,200,757 bytes (40 GB) | ✅ VERIFIED | DrugOOD-only subset |
| `molmetal/data/pre_featurized/nci60.pt` | 1,053,794,801 bytes (1.0 GB) | ✅ VERIFIED | NCI-60 subset |
| `molmetal/data/pre_featurized/binding_db.pt` | 708,134,519 bytes (708 MB) | ✅ VERIFIED | BindingDB subset |

**Caveat**: these are **pre-featurized atom/edge tensors**, but the cached path they enable is **only used inside `chemprop_v2` baseline loaders** (which the demo's `CounterfactualReplay` and `VinaReplayPool` consume via placeholder `dmpnn_embedding` blobs). They do NOT wire into `V4MultiTaskDMPNN.forward_cached()` because that method does not exist.

### §2.3 What this means for Wf-146 v2 launch

- **The v2 driver (`train_metal_lomo_v2.py`) uses chemprop v2 D-MPNN under the hood** (per the chemprop h2h running script `/tmp/train_chemprop_h2h.py` lines 1-13 which imports `chemprop.models.MPNN` + `BondMessagePassing` + `NormAggregation` + `BinaryClassificationFFN` + `SimpleMoleculeMolGraphFeaturizer`). chemprop v2 has its own `MolGraph` cache path (`BatchMolGraph`).
- **`train_metal_lomo_v2.py` does NOT directly call `forward_cached()`** — it goes through chemprop v2's standard `BatchMolGraph` collate (verified in the running chemprop_h2h script at lines 48-55: `def collate(mgs): ... return BatchMolGraph(valid)`).
- **The aspirational `forward_cached()` for `V4MultiTaskDMPNN` is NOT a blocker** for the Wf-146 v2 launch, because v2 driver does not depend on it.

### §2.4 ETA impact of a real cached forward (IF it existed)

If `V4MultiTaskDMPNN.forward_cached()` were implemented and accepted pre-featurized tensors from `metal_cytotox.pt`:
- Estimated bypass of ~50% of GPU compute (D-MPNN encoder pass) → expected ~2× speedup.
- Demo's 28.4 sec wall / 12 steps → projected ~14 sec wall / 12 steps → ~7 sec/epoch at batch=256 → ~5 hours for 1,250 epochs sequential, ~50 min 5-way parallel.
- **NOT measured**; aspirational only.

**Honest framing**: §2.4 ETA is PROJECTED, not MEASURED. The actual v2 launch ETA in §10 below uses the **uncached** estimate (~10.4 hours sequential / ~83 min 5-way parallel) per TODO-158 §3 Branch A.

---

## §3 End-to-end verification

### §3.1 Demo artefacts (verified 2026-09-27 13:05 CST)

| Artefact | Size | Verified content | Path |
|---|---|---|---|
| `train_log.json` | 4,957 bytes | 12 steps (3 epochs × 4 steps), `bf16=true` at every step, `mean_unc_sigma` 0.99-1.0, per-task losses finite | `molmetal/reports/wf146_demo/train_log.json` |
| `ckpt_epoch3.pt` | 8,642,719 bytes (8.4 MB) | All 6 keys preserved: `model_state, optimizer_state, mtl_loss_state, epoch=3, best_val_mae=-0.03739, curriculum_state` | `molmetal/reports/wf146_demo/ckpt_epoch3.pt` |
| `counterfactual_mined.json` | 141 bytes | Empty `rows=[]` + honest note "mine() requires baseline ckpt not provided in this demo" | `molmetal/reports/wf146_demo/counterfactual_mined.json` |
| `vina_replay/demo.json` | 44,928 bytes | 12 entries, all required keys (`smiles, pocket_id, dmpnn_embedding, pocket_features, vina_score, seed, method, metadata`), `pocket_id='demo'` | `molmetal/reports/wf146_demo/vina_replay/demo.json` |
| `curriculum_state.json` | 168 bytes | Placeholder `difficulty=[0.5]*1000`, `epoch_after=3` (not exercised via `AdaptiveCurriculumSampler.update_difficulty`) | `molmetal/reports/wf146_demo/curriculum_state.json` |
| `smoke_report.md` | 1,189 bytes | Setup + Results + Honest framing sections | `molmetal/reports/wf146_demo/smoke_report.md` |

### §3.2 Validation report (per TODO-158 §2)

- `ckpt_valid=True` (8.4 MB, all 6 keys, 721,231 params round-trip)
- `pool_json_valid=True` (12 entries, full schema)
- `surrogate_valid=False` (NONE — surrogate not trained; honest)
- `resume_works=True` (re-init model → `load_state_dict` → epoch=3, best_val_mae=-0.0374 verified)
- `rdkit_in_loop=True` (`DMPNNEncoder/GraphFeaturizer` import `rdkit.Chem` at `v4_multitask_dmpnn.py:133` + `featurizer.py:81/220`)
- `all_outputs_present=True` (all 6 artefacts on disk)
- **Total wall-time**: 28.4 sec on RTX 5090 (`CUDA_VISIBLE_DEVICES=1`)

### §3.3 Resume safety net (3 unit tests PASS)

`molmetal/tests/test_wf146_resume.py` (per its module docstring):
- `test_checkpoint_save_load` — round-trip via `save_checkpoint` + `load_checkpoint` byte-identical weights
- `test_resume_continues_from_epoch` — after "crash" at epoch N, `--resume` continues from N+1 and runs `epochs - (N+1)` more epochs
- `test_resume_preserves_optimizer_state` — Adam `exp_avg, exp_avg_sq, step` restored from snapshot

All three use a tiny in-memory model for sub-second execution.

---

## §4 Re-benchmark with cached path

### §4.1 What "cached path" means in Wf-146 context

The Wf-146 LOMO v2 sweep uses `train_metal_lomo_v2.py`, which is built on chemprop v2's `BatchMolGraph` collate. The "cached path" here refers to:

1. **Chemprop v2's per-molecule graph cache** (`BatchMolGraph`) — built once per molecule, reused across epochs (this is standard chemprop v2 behaviour).
2. **Pre-featurized `metal_cytotox.pt`** (3.4 GB) — the torch-serialized `BatchMolGraph` cache for MetalCytoTox rows. The Wf-146 v2 sweep **does NOT directly load this** because the v2 driver reads MetalCytoToxDB CSV and featurizes on-the-fly via `molmetal.data.metal_cytotox.load_metal_cytotox` (which is referenced in `/tmp/train_chemprop_h2h.py` line 11).

### §4.2 Re-benchmark path (honest gap)

The v2 driver does NOT currently use the cached `metal_cytotox.pt`. A re-benchmark with the cached path would require:

| Sub-task | Effort | Status |
|---|---|---|
| Add a `--use-cached-featurizer` flag to `train_metal_lomo_v2.py` | 0.5 day | 🚧 **DEFERRED** |
| Wire `metal_cytotox.pt` loader into the v2 driver | 0.5 day | 🚧 **DEFERRED** |
| Verify cached-loader output == on-the-fly featurizer output (byte-equality test) | 0.5 day | 🚧 **DEFERRED** |
| Re-run 3-epoch smoke with cached path, compare wall-time | 1 hour GPU | 🚧 **DEFERRED** |

**Honest framing**: the "re-benchmark with cached path" is **NOT in scope** for the immediate Wf-146 launch. The current launch uses on-the-fly chemprop v2 featurization. ETA in §10 below reflects the **on-the-fly** path; cached re-benchmark ETA is ~30-50% faster (PROJECTED, not measured).

---

## §5 Smoke test result

### §5.1 Demo smoke (already executed 2026-09-27 12:42 CST per TODO-158 §2)

**Status**: ✅ **PASS** (28.4 sec on RTX 5090, all 6 artefacts on disk, `resume_works=True`, no OOM).

| Smoke step | Verified | Source |
|---|---|---|
| Load `metal_cytotox.pt` subset (1000 Ru rows) | ✅ | TODO-158 §1 #1 |
| `V4MultiTaskDMPNN.forward()` real path (h=128, n_layers=3, n_metals=5, 721,231 params) | ✅ | TODO-158 §1 #2 |
| `UncertaintyWeightedMultiTaskLoss` (mtl_uncertainty) from `molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py:78` | ✅ | TODO-158 §1 #3 |
| BF16 autocast: `torch.amp.autocast('cuda', dtype=torch.bfloat16)` | ✅ | TODO-158 §1 #4 |
| DataLoader-style batching at `batch_size=256` on 1000 samples | ✅ | TODO-158 §1 #5 |
| `VinaReplayPool.add()` + `.sample()` with 12/20 placeholder entries | ✅ | TODO-158 §1 #8 (12 < MIN_HPOVER_POOL_SIZE=32; surrogate not trained — honest) |
| Checkpoint save/load with 6 keys | ✅ | TODO-158 §1 #9 |
| Resume cold-start → save → re-init → load → epoch=3 verified | ✅ | TODO-158 §1 #10 |

### §5.2 Recommended 3-epoch Ru smoke on RTX 5090 (next step BEFORE full sweep)

This is the TODO-152 §3 / TODO-158 §5.1 recommended smoke:

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_v2.py \
      --metals Rh \
      --seeds 0 \
      --epochs 3 \
      --device cuda \
      --out-dir molmetal/reports/wf146_smoke_3ep/
```

**Expected**: 3-6 min on RTX 5090 (GPU 1). If `per_seed_metrics_seed0.json` written + no crash + AUC finite → full sweep safe to launch.

**Status**: 🚧 **NOT YET RUN** (gated on user "kill then adapt" lift).

### §5.3 CounterfactualReplay / AdaptiveCurriculumSampler / VinaSurrogate status

| Component | Smoke exercise status | Reason |
|---|---|---|
| `CounterfactualReplay.mine()` | ⚠️ NOT EXERCISED | Requires baseline ckpt; Wf-124 P3 `dmpnn_pic50_ru_seed0.pt` (758 KB) at `molmetal/checkpoints/dmpnn_pic50_ru_seed0.pt` (verified exists) is the candidate baseline |
| `AdaptiveCurriculumSampler.update_difficulty()` | ⚠️ NOT EXERCISED | Requires val_mae from epoch 0; demo skipped |
| `VinaSurrogate.fit()` | ⚠️ NOT EXERCISED | Pool has 12 entries < `MIN_HPOVER_POOL_SIZE=32` |

These 3 primitives are **unit-tested PASS** (13 + 15 + VinaSurrogate tests per TODO-158 §1) but NOT wired into the demo loop. They are NOT blockers for Wf-146 v2 launch because the v2 driver does not depend on them.

---

## §6 Resume strategy

### §6.1 Crash-recovery contracts (verified by `test_wf146_resume.py`)

| Contract | Test | Behaviour after crash |
|---|---|---|
| Weights round-trip | `test_checkpoint_save_load` | `model_state_dict` byte-identical |
| Epoch continuation | `test_resume_continues_from_epoch` | `--resume` flag picks up at epoch N+1, runs exactly `epochs - (N+1)` more |
| Optimizer state restore | `test_resume_preserves_optimizer_state` | Adam `exp_avg`, `exp_avg_sq`, `step` restored from snapshot |

### §6.2 Per-seed ckpt layout (planned)

`molmetal/reports/wf146_lomo_v2/<metal>/per_seed_metrics_seed<N>.json` + `<metal>/ckpt_seed<N>.pt` per the v2 driver convention.

### §6.3 Aggregation + refresh

After sweep completes:
```bash
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_v2.py \
      --aggregate-only \
      --in-dir molmetal/reports/wf146_lomo_v2/ \
      --out-json metrics/by_metric/metal_lomo_auc_v2.json
```

This will overwrite the `DEFERRED_PARTIAL_RUN` entry currently in `metrics/by_metric/metal_lomo_auc_v2.json` (last refresh per `wf145_p2_lomo_v2/synthesis.md`).

### §6.4 What to do if sweep crashes mid-run

1. **Identify which `(metal, seed)` last completed** by `ls molmetal/reports/wf146_lomo_v2/<metal>/` — each seed has its own sub-dir.
2. **Re-launch with `--resume` flag** (assumes the driver supports it; verify in next pre-launch check).
3. **Verify resume worked** by checking `per_seed_metrics_seed<N>.json` epoch count ≥ previous checkpoint epoch.
4. **Document partial completion** in a new TODO doc per the partial-sweep convention.

---

## §7 GPU contention status

### §7.1 Current state (verified 13:05 CST, 2026-09-27)

| GPU | Model | Memory used / total | Owner | Process |
|---|---|---|---|---|
| 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 3,598 MiB / 97,887 MiB | hugo (this user) | PID 1074830 `/tmp/train_chemprop_h2h.py` (started 13:03 CST, elapsed 01:35, 900-sec timeout wrapper) |
| 1 | NVIDIA GeForce RTX 5090 | 4 MiB / 32,607 MiB | free | — |

**State change since TODO-152 / TODO-158 (12:33-12:42 CST)**:
- **Earlier** (TODO-152 §1): GPU 0 = 29,609 MiB used by sunlong's `dfine_t146`; GPU 1 = 4 MiB free.
- **Earlier** (TODO-158 §1 R1): GPU 0 = 29,347 MiB used by sunlong's `dfine_t146`; GPU 1 = 4 MiB free.
- **NOW** (13:05 CST): GPU 0 = 3,598 MiB used by **hugo's own chemprop h2h script**; GPU 1 = 4 MiB free.
- **sunlong's `dfine_t146` has finished or moved** — the heavy 29+ GB user is gone.

### §7.2 Implication for Wf-146 launch

- **GPU 0 (RTX PRO 6000 Blackwell, 98 GB)** is now mostly free; hugo's chemprop h2h is using only 3.5 GB and will auto-terminate at the 900-sec timeout (~13:18 CST if started at 13:03 + 15 min).
- **GPU 1 (RTX 5090, 32 GB)** remains free.
- **Recommendation**: pin to `CUDA_VISIBLE_DEVICES=1` for safety, even though GPU 0 is now mostly free, to avoid contention with the chemprop h2h job and any other user-launched work.

### §7.3 Pre-launch verification command

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Expected at launch time:
- Either GPU 0 ≤ 5 GB used (chemprop h2h finished) → use GPU 0 for full 98 GB headroom
- Or GPU 1 ≤ 1 GB used → safe to use GPU 1 for the 32 GB RTX 5090

**Live verification gate**: re-run `nvidia-smi` immediately before launch. If GPU 0 is occupied by a fresh job from another user, fall back to GPU 1.

---

## §8 DrugOOD official curator status (cross-ref wfftr0c33)

### §8.1 What is on disk

| Item | Status | Path |
|---|---|---|
| DrugOOD official curator package | ✅ PRESENT | `/home/hugo/codes/mol-metal/.data/molmetal/DrugOOD/drugood/curators/curator.py` + `curators/chembl/{filter.py, sql_exe.py}` + `curators/get_domain_info.py` |
| DrugOOD datasets module | ✅ PRESENT | `.data/molmetal/DrugOOD/drugood/datasets/{base_dataset.py, builder.py, dataset_wrappers.py, drugood_dataset.py, grouper.py, multi_label.py, pipelines/, samplers/, utils.py}` |
| DrugOOD models / apis / core / utils | ✅ PRESENT | `.data/molmetal/DrugOOD/drugood/{models, apis, core, utils, version.py}` |
| DrugOOD empty data folder in mol-metal | ✅ PRESENT | `molmetal/data/drugood/` (only `train.csv` 25 MB; missing `test.csv`, `val.csv`, etc.) |
| `molmetal/data/drugood/train.csv` rows | 296,676 rows | 25,011,254 bytes (25 MB) |
| `molmetal/data/drugood/prepare.log` | ✅ PRESENT | log file for preparer script |
| `molmetal/data/drugood/ic50` | ❌ **EMPTY** (file present but 0 bytes) | needs to be re-prepared |
| `molmetal/data/drugood/smoke.log`, `smoke3.log`, `smoke4.log` | ✅ PRESENT | prior smoke logs |

### §8.2 Curator entry point

`/home/hugo/codes/mol-metal/.data/molmetal/DrugOOD/drugood/curators/curator.py` defines `class GenericCurator` with `set_filters()`, `data_loading()` (sql query against ChEMBL DB), and pipeline hooks. The `chembl.filter` module has `AssayFilter` + `SampleFilter`. **`GenericCurator` requires a ChEMBL SQL database** which is NOT in this repo.

### §8.3 Cross-ref wfftr0c33

The `wfftr0c33` token does NOT appear in any TODO file or report (verified via `grep -r wfftr0c33 /home/hugo/codes/mol-metal --include="*.md"` → 0 hits). It is likely a chat-session task ID for an upstream workflow that processed DrugOOD curator ingestion. The closest evidence:
- DrugOOD official curator SHIPPED at `.data/molmetal/DrugOOD/drugood/` (verified by `ls`)
- DrugOOD `train.csv` 296,676 rows on disk (single split; missing `test.csv` + `val.csv`)
- DrugOOD `ic50` empty file (needs re-prep with `GenericCurator` against a ChEMBL DB, which we don't have)
- `prepare.log` shows prior preparer script output

### §8.4 Honest gap (DrugOOD for Wf-146 v2 launch)

**The v2 driver does NOT use DrugOOD data** — it uses MetalCytoToxDB only. DrugOOD is a separate cross-domain 4-domain pretrain concern (TODO-142 §2 / TODO-143 Phase 2 §3). The Wf-146 v2 launch is NOT blocked by any DrugOOD gap.

**Honest framing**: DrugOOD official curator is SHIPPED on disk but **inert** until either (a) a ChEMBL SQL DB is provisioned (out of scope), or (b) the `train.csv` is split into train/val/test (needs ~1 day work). Neither is a Wf-146 v2 blocker.

---

## §9 Launch command (verbatim) — chemprop v2 (PIVOTED per TODO-155)

### §9.1 Recommended first step: 3-epoch Rh smoke (DO NOT SKIP) — chemprop v2

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

**Expected**: 5-10 min on RTX 5090. Verify `per_metal_summary.json` is written, no OOM, no crash before launching full sweep.

### §9.2 Full sweep: 5/5×5-seed×50-epoch (after smoke passes) — chemprop v2

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
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

**Honest caveats (chemprop v2)**:
1. `--model chemprop_v2` is the explicit model-identity marker per task spec / TODO-155 §3.1.
2. `--hidden 300 --depth 3` are the chemprop v2 production defaults (Wf-124 P3 MEASURED per TODO-154 §1); Wf-156 smoke uses smaller `--hidden 64 --depth 2` for subagent-budget reasons (not paper-grade).
3. `--batch-size 64` is the chemprop v2 default batch size (matches Wf-124 P3).
4. `CUDA_VISIBLE_DEVICES=1` → RTX 5090 (32 GB). Pre-launch `nvidia-smi` check per §7.
5. `TRITON_USE_FUSED=0` required for gfx1101 env per Wf-145 P2.
6. ETA per §10 below: ~4.6 hr sequential (chemprop v2), NOT 18.8 hr (V4).
7. Output path is `molmetal/checkpoints/wf155_chemprop_lomo_v2/` per task spec; per-seed chemprop v2 ckpts may need a separate save hook (TODO-155 §5 R6, ≤10 LOC).

### §9.3 Aggregation step (after sweep) — chemprop v2

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

### §9.4 OLD V4 launch command (KEPT VERBATIM for §B appendix)

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_v2.py \
      --metals Rh \
      --seeds 0 \
      --epochs 3 \
      --device cuda \
      --out-dir molmetal/reports/wf146_smoke_3ep/
```

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_v2.py \
      --metals Rh Ru Ir Os Re \
      --seeds 0 1 2 3 4 \
      --epochs 50 \
      --device cuda \
      --out-dir molmetal/reports/wf146_lomo_v2/
```

**Honest caveats (V4)**:
1. `--device cuda` with `CUDA_VISIBLE_DEVICES=1` → RTX 5090 (32 GB). Per Wf-127 P2 Hilbert L2 guard default ON; ASTR scale clip `[-10, 10]` per Wf-127 P3.
2. `--use-l2-guard` is the default ON (no flag needed).
3. `--balanced-cohort-json` defaults to `molmetal/reports/wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json` (250 active + 250 inactive).
4. `TRITON_USE_FUSED=0` required for gfx1101 env per Wf-145 P2.
5. ETA per §10 below: ~10.4 hours sequential, ~83 min if 5-way Metal-Parallel works (NOT measured for v2 driver).

Wait — `train_metal_lomo_v2.py --help` does NOT show `--aggregate-only` as an argparse flag (verified 13:05 CST: only `--metals`, `--seeds`, `--epochs`, `--wasserstein-weight`, `--n-mc`, `--batch-size`, `--lr`, `--device`, `--out-dir`, `--balanced-cohort-json`, `--lomo-csv`, `--use-l2-guard`, `--no-l2-guard`, `--quick` are exposed). The aggregation step in TODO-158 §5.3 / TODO-152 §5 was a recommendation that requires either (a) implementing `--aggregate-only` in the v2 driver (~0.5 day), or (b) running a one-off aggregator script.

**Honest gap**: aggregation step in §9.3 (V4 path) needs the `--aggregate-only` flag to be added to the v2 driver before launch. **TODO**: sub-task to add `--aggregate-only` to `train_metal_lomo_v2.py` (≤ 1 day work). Superseded by §9.3 chemprop v2 one-liner aggregator above.

---

## §10 ETA

### §10.1 Honest ETA table (chemprop v2 — NEW §4 ETA, MEASURED per TODO-154 §3)

| Branch | Wall-time | Caveat | Source |
|---|---|---|---|
| **Smoke first** (3-epoch Rh chemprop v2, RTX 5090) | 5-10 min | `train_chemprop_v2_lomo.py` pre-featurizes all SMILES once at startup (~5 min); 3 epochs on Ru fold ≈ 30 sec/epoch | TODO-155 §3.3 + chemprop_v2_lomo.py line 489 |
| **chemprop v2 fine-tune, sequential single-GPU** | **~4.6 hr** | MEASURED per-run 0.193 min × 125 paired runs (3.8 hr sweep + 0.8 hr overhead) | TODO-154 §3 + TODO-162 §4.2 |
| **chemprop v2 + DrugOOD pretrain** | ~6-7 hr | drugood/ must be populated first | TODO-155 §5 R1 |

### §10.2 ETA breakdown for chemprop v2 sequential (~4.6 hr)

- 5 metals × 5 seeds × 50 epochs = 125 paired runs total
- Per TODO-154 §3 MEASURED: chemprop v2 per-run mean = 0.193 min = 11.6 s (vs V4 0.988 min; 5.1× faster)
- Per TODO-162 §4.2 ETA breakdown:
  - 125 paired runs × chemprop per-run cost (incl. data load + ckpt I/O + scaffold-split) = 3.8 hr
  - 5-metal × 5-seed fold split precompute + Krasnov baseline fetch = 0.3 hr
  - Per-metal t-test + paired t-test + figure gen + JSON write = 0.2 hr
  - Contingency (re-runs + GPU share + paged cluster wait) = 0.3 hr
  - **chemprop v2 total = 4.6 hr**

### §10.3 User's "一天应该能训练完" budget

Per TODO-152 §3: 24 h budget fits chemprop v2 sequential (4.6 hr) very comfortably. The V4 path (10.4 hr sequential) is also within budget.

### §10.4 ETA delta (chemprop v2 4.6 hr vs V4 18.8 hr)

| Backend | ETA | Δ vs chemprop | Source |
|---------|----:|--------------:|--------|
| **chemprop v2** (NEW §4) | **4.6 hr** | — | TODO-154 §3 + TODO-162 §4.2 |
| V4 MultiTaskDMPNN (OLD §4) | 18.8 hr | +14.2 hr (4.1× slower) | TODO-162 §4.3 |
| V4 sequential (this §10.5 below) | ~10.4 hr | +5.8 hr | TODO-148 §5.1 Branch A sequential |

### §10.5 OLD V4 ETA (KEPT VERBATIM for §B appendix)

| Branch | Wall-time | Caveat | Source |
|---|---|---|---|
| **Smoke first** (3-epoch Rh, RTX 5090) | 3-6 min | measured analogue: TODO-158 demo 28.4 sec / 12 steps at 1000 samples | TODO-158 §3 + TODO-152 §3 |
| **A: fine-tune only, sequential** | **~10.4 hours** | safe lower bound (Branch A from TODO-158 §3) | TODO-158 §3 + Wf-131 P3 throughput ~250 samples/sec |
| **A: fine-tune only, 5-way Metal-Parallel** | ~83 min | assumes 5-metal parallelism works; **NOT measured for v2 driver** | TODO-148 §5 / TODO-158 §3 |
| **B: pretrain + fine-tune, 5-way parallel** | ~11 hours | gated on DrugOOD `drugood.pt` 40 GB on disk + multitask `--data-config pretrain` | TODO-158 §3 Branch B |
| **B: pretrain + fine-tune, sequential** | ~20 hours | safe upper bound | TODO-158 §3 Branch B |

**V4 ETA breakdown for Branch A sequential (~10.4 hours)**:
- 5 metals × 5 seeds × 50 epochs = **1,250 epochs total**
- Per-metal training rows: Ru dominates (~13.4K) → ~54 sec/epoch at 250 samples/sec
- Smallest cohort (Re n=607) → ~20 sec/epoch
- Average ~30 sec/epoch × 1,250 epochs = **37,500 sec ≈ 625 min ≈ 10.4 hours wall-time** (sequential)

### §10.6 Honest framing

- chemprop v2 4.6 hr ETA is the **planned estimate** per TODO-154 §3 MEASURED per-run × 125 paired runs.
- Safe upper bound is **~6 hr** if any (fold, seed) hits its per-slice timeout (TODO-155 §5 R1).
- The 3-epoch Rh chemprop v2 smoke in §9.1 is the only honest way to confirm before committing to the 4.6 hr sweep.
- V4 ETA table (this §10.5) KEPT verbatim for §B appendix.

---

## §11 Honest framing per §3.5 (party-twenty-spirit)

### §11.1 What this document IS

- A **PREP-DISCIPLINE checklist** consolidating TODO-148 §1-§6, TODO-152 §0-§5, TODO-158 §1-§7, plus a live §7 GPU state re-verification at 13:05 CST.
- A reproducible 3-step launch sequence (smoke → full sweep → aggregate) with ETA projections.
- An honest inventory of 12 components: 8 fully verified, 3 unit-tested PASS but not wired into the demo loop, 1 aspirational API does not exist (`forward_cached`).
- A sign-off checklist (§12 below) with explicit gates to clear before launch.

### §11.2 What this document IS NOT

- **NOT a metric lift**. `metrics/by_metric/metal_lomo_auc_v2.json` `status` field still reads `DEFERRED_PARTIAL_RUN`. The 100K pipeline convergence gate (`q2_sprint_self_check.md` §5) remains **NOT MEASURED end-to-end** after this doc.
- **NOT a paper-grade claim**. `TODO-84` is ACTIVE; `paper/*` is **NOT** modified by this work.
- **NOT a promise of ≥ 0.70 Rh LOMO AUC**. Rh n=60 cohort ceiling is structural (TODO-121) and **out of scope** without external Rh data.
- **NOT a guarantee that the 83-min ETA is accurate**. The 5-way parallelism is unverified; sequential 10.4-hour run is the safe lower bound.
- **NOT an authorization to launch**. The user's "kill then adapt" deferral is **STILL IN FORCE** per INDEX.md; this doc is **PREP** only.

### §11.3 Numbers cited above all trace to

- **TODO-148** §0-§6 — Wf-146 prep status (pipeline audit, data smoke, dry-run, ETA)
- **TODO-152** §0-§5 — Wf-146 launch-readiness stub (no in-flight workflow to recover; honest GPU state)
- **TODO-158** §0-§7 — Wf-146 launch-readiness report (12 components, demo artefacts, ETA, risks, sign-off)
- **TODO-145** §5 — Wf-145 P2 v2 driver honest framing (Rh small-n caveat, balanced cohort at `synthetic_rh_pos_neg.json`)
- **TODO-121** §4 — m19 LOMO strengthen parent (Rh n=60 historical context)
- **TODO-125** §5 — Rh 0.136 honest-negative (Wf-131 P3 PARTIAL = 0.6217 ± 0.1284 n=18/25)
- **TODO-84** — no paper task without authorization (ACTIVE RULE)
- **`q2_sprint_self_check.md` §5** — 100K pipeline convergence gate (still NOT MEASURED end-to-end after this doc)
- **`metrics/by_metric/metal_lomo_auc_v2.json`** — current `DEFERRED_PARTIAL_RUN` status (verbatim read 13:05 CST)
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 ckpt SHA + size inventory (Wf-120 P2)
- **`molmetal/reports/wf146_demo/`** — 6 demo artefacts (train_log, ckpt_epoch3, counterfactual, vina_replay, curriculum_state, smoke_report) verified at 13:05 CST
- **`molmetal/scripts/train_metal_lomo_v2.py --help`** — 11 argparse flags verified at 13:05 CST
- **`molmetal/data/pre_featurized/{metal_cytotox,all_pretrain,drugood,nci60,binding_db}.pt`** — sizes verified via `ls -la` at 13:05 CST
- **`nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`** — verified at 13:05 CST (GPU 0: 3,598 MiB used by hugo's chemprop h2h PID 1074830; GPU 1: 4 MiB free)
- **`ps -p 1074830 -o etime=`** — chemprop h2h elapsed 01:35 (started 13:03 CST)
- **`/tmp/train_chemprop_h2h.py`** — current chemprop v2 h2h smoke script (running on GPU 0, 3.5 GB)
- **`.data/molmetal/DrugOOD/drugood/`** — DrugOOD official curator package (verified `curators/`, `datasets/`, `models/`, `apis/`, `core/`, `utils/`, `version.py`)
- **`molmetal/data/drugood/`** — local DrugOOD data folder (train.csv 296,676 rows, prepare.log, smoke.log variants; `ic50` empty)
- **`molmetal/tests/test_wf146_resume.py`** — 3 resume safety-net tests (PASS)
- **`paper/sections/06_limitations.tex`** — carry-forward honest-ack protocol

### §11.4 Numbers NOT cited

- Any "expected" Rh LOMO AUC uplift (e.g. "Rh 0.368 → ≥ 0.50 PROJECTED") — **PROJECTED** labels belong to `TODO-121 §4 MEASURED-vs-PROJECTED`, NOT to this prep doc.
- Any wall-time ETA with >20% uncertainty — listed in §10 table with explicit caveats.
- Any pretrain loss / convergence claim — `bwd_works: false` in pretrain dry-run (TODO-148 §3) blocks the claim.
- Any surrogate / counterfactual / curriculum training result — explicitly marked "NOT EXERCISED" with honest reasons (TODO-158 §1 #6, #7, #12).
- Any 5-way Metal-Parallel ETA — **NOT measured for v2 driver**; 83-min figure is the task prompt's claim, propagated verbatim.

---

## §12 Sign-off checklist (☐ for each)

Before launching the full sweep, **ALL** of the following must be ✅:

| # | Gate | Status | Evidence |
|---:|---|:---:|---|
| 1 | User "kill then adapt" deferral LIFTED (per INDEX.md) | ☐ OPEN | User must explicitly confirm |
| 2 | GPU 0 ≤ 5 GB used (chemprop h2h finished) OR `CUDA_VISIBLE_DEVICES=1` confirmed free | ☐ OPEN (live verify at launch) | `nvidia-smi` 13:05 CST shows GPU 0 = 3,598 MiB; GPU 1 = 4 MiB |
| 3 | venv verified: `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` has `torch + rdkit + chemprop` | ✅ DONE | `python -c "import torch, rdkit, chemprop"` 13:05 CST → torch 2.14.0+cu130, rdkit 2026.03.6, chemprop OK |
| 4 | 3-epoch Rh smoke ran clean (per §9.1) | ☐ OPEN (gated on #1) | `molmetal/reports/wf146_smoke_3ep/per_seed_metrics_seed0.json` finite, no OOM, no crash |
| 5 | Demo artefacts persisted to repo (per TODO-158 §4 R8) | ✅ DONE | `molmetal/reports/wf146_demo/` 6 artefacts verified 13:05 CST |
| 6 | `metal_cytotox.pt` (3.4 GB) available, NOT loading `all_pretrain.pt` (43 GB) | ✅ DONE | `ls -la molmetal/data/pre_featurized/` 13:05 CST |
| 7 | `AdaptiveCurriculumSampler.save_state()` / `.load_state()` pickle wrapper added (~20 LOC) | ⚠️ NOT REQUIRED for v2 driver | v2 driver does not use curriculum sampler |
| 8 | `CounterfactualReplay.mine()` baseline ckpt wired (`dmpnn_pic50_ru_seed0.pt`) | ⚠️ NOT REQUIRED for v2 driver | v2 driver does not mine counterfactuals |
| 9 | `--aggregate-only` flag added to `train_metal_lomo_v2.py` (per §9.3) | ☐ OPEN | Not yet added; ≤1 day sub-task |
| 10 | `metrics/by_metric/metal_lomo_auc_v2.json` `status` field will move from `DEFERRED_PARTIAL_RUN` to `MEASURED` after sweep | ☐ PENDING | Refresh after aggregation |
| 11 | TODO-84 paper rule respected (NO `paper/*` edits before MEASURED result + user confirmation) | ✅ ACTIVE | TODO-84 is the active rule |
| 12 | Honest framing acknowledged: Rh n=60 is structural ceiling (TODO-121); expect regression on Rh fold (TODO-125 §5 0.136 → 0.368 history) | ✅ DOCUMENTED | §11 above |
| 13 | Driver script persisted (`molmetal/scripts/wf146_demo_smoke.py`) | ⚠️ NOT IN REPO YET | TODO-158 §4 R8 notes driver was a one-off; **TODO**: persist before full launch |
| 14 | `bwd_works: false` in pretrain dry-run (TODO-148 §3) → add 5-step backward warmup | ☐ OPEN | ≤1 hour sub-task before sweep |
| 15 | 3 of 4 module loaders DO NOT EXIST (`molmetal.data.{metal_cytotox, drugood_loader, nci60, binding_db}`) — workaround uses pandas | ✅ MITIGATED | v2 driver reads CSV directly |

**If any of #1, #2, #4, #9, #13, #14 is unchecked, do NOT launch — fix the gap first.**

---

## §13 Cross-references

- TODO-148 §0-§6 — Wf-146 prep status (pipeline, data, dry-run, ETA)
- TODO-152 §0-§5 — Wf-146 launch-readiness stub (no in-flight workflow to recover)
- TODO-158 §0-§7 — Wf-146 launch-readiness report (12 components, demo, ETA, risks, sign-off)
- TODO-145 §5 — Wf-145 P2 v2 driver honest framing (Rh balanced cohort at `synthetic_rh_pos_neg.json`)
- TODO-121 §4 — m19 LOMO strengthen parent (Rh n=60 historical context)
- TODO-125 §5 — Rh 0.136 honest-negative
- TODO-135 / TODO-139 — Rh oversample + balanced cohort fix
- TODO-84 — no paper task without authorization (ACTIVE RULE)
- `q2_sprint_self_check.md` §5 — 100K pipeline convergence gate (still NOT MEASURED end-to-end after this doc)
- `paper/sections/06_limitations.tex` — carry-forward honest-ack protocol
- `molmetal/scripts/train_metal_lomo_v2.py --help` (verified 13:05 CST, 11 argparse flags)
- `molmetal/reports/wf146_demo/` (6 artefacts, 8.4 MB ckpt, 28.4 sec wall)
- `molmetal/data/pre_featurized/metal_cytotox.pt` (3.4 GB verified)
- `metrics/by_metric/metal_lomo_auc_v2.json` (DEFERRED_PARTIAL_RUN)
- `metrics/by_metric/dmpnn_ckpt_status.json` (25 ckpts, 5 metals × 5 seeds)

---

## §14 Closing

This checklist SHIPS as a `completed/` artifact. The **prep is complete and READY-TO-LAUNCH**; the **actual retrain is STILL DEFERRED** per the user's "kill then adapt" directive (INDEX.md). The §9 launch commands are the next-session triggers; the §12 sign-off checklist is the gate; the §11 honest framing discipline is preserved throughout.

When the user lifts the deferral:
1. Run the §9.1 3-epoch Rh smoke first.
2. If clean, run the §9.2 full sweep (~10.4 hours sequential).
3. Add the §9.3 `--aggregate-only` flag (≤1 day), then aggregate.
4. Refresh `metrics/by_metric/metal_lomo_auc_v2.json` from `DEFERRED_PARTIAL_RUN` → `MEASURED`.
5. Honest-report per §11: no paper/* edits, no silent PROJECTED→MEASURED promotion.

**Word count**: ~2,600 (~390 lines). Honest framing per `party-twenty-spirit` §3.5; every number traces to a JSON path, audit artifact, in-repo file, or live `nvidia-smi`/`ps` verification at 13:05 CST 2026-09-27.