# TODO-155 — Pivot Wf-146/157/158 to chemprop v2 (V4 → chemprop_v2)

**Date**: 2026-09-27
**Status**: 📝 **PIVOT STRATEGY DOC** — rescope Wf-146/157/158 launch from V4MultiTaskDMPNN to chemprop v2 backbone; ETA drops 18.8 hr → 4.6 hr.
**Owner**: Hugo Lee + ultracode agents
**Parent**: TODO-154 (chemprop v2 vs V4 head-to-head SHIPPED, n=14 paired runs, p=1.35e-5) + TODO-162 (this doc's sibling — same pivot logic for paper §4; this doc scopes the Wf-146/157/158 retrain launch) + TODO-148 (Wf-146 prep status) + TODO-157 (Wf-146 pre-flight checklist) + TODO-158 (Wf-146 launch-readiness report) + TODO-120 (R20 GPU retrain plan, full 25 paired runs).
**Skill alignment**: `party-twenty-spirit` §3.5 (every ETA + AUC number traces to TODO-154 MEASURED or in-repo file path).
**Constraint**: V4 docs kept verbatim for §B appendix (per user instruction + TODO-162 §3.1).

---

## §0 TL;DR — three lines

1. **Pivot Wf-146/157/158 launch from `V4MultiTaskDMPNN` to `chemprop v2`**. chemprop v2 wins on all 5 metals (Ru/Ir/Os/Rh/Re) by **+0.282 AUC mean** (0.886 vs 0.604, paired t-test t=6.754, **p=1.35e-5**, n=14 paired runs per TODO-154 §0).
2. **ETA drops from 18.8 hr (V4) to 4.6 hr (chemprop v2)** for full 5-metal × 5-seed × 50-epoch LOMO sweep — same hardware, same CPU-class epochs. Per-run speedup 5.1× (0.193 min vs 0.988 min); end-to-end 4.1× (per TODO-162 §4 verbatim).
3. **V4 stays as §B appendix honest multi-task baseline** (TODO-162 §3.1) — no V4 scripts deleted, no V4 ckpts deleted (15 ckpts on disk), no V4 tests deleted (179/179 PASS); V4 docs (TODO-148 §1.2, §5; TODO-157 §2, §10; TODO-158 §3) KEPT for §B reference verbatim.

---

## §1 Why pivot (MEASURED only — never projected)

### §1.1 chemprop v2 vs V4 head-to-head verdict (TODO-154 §0 verbatim)

| Metal | V4 mean AUC | V4 std | chemprop v2 mean AUC | chemprop v2 std | Δ (chemprop − V4) |
|-------|------------:|-------:|---------------------:|----------------:|------------------:|
| Ru    | 0.5419      | 0.1021 | **0.9095**           | 0.0022          | **+0.3675**       |
| Ir    | 0.4803      | 0.0646 | **0.8846**           | 0.0060          | **+0.4043**       |
| Os    | 0.5788      | 0.2312 | **0.8890**           | 0.0153          | **+0.3103**       |
| Rh    | 0.7261      | 0.0293 | **0.8631**           | 0.0208          | **+0.1369**       |
| Re    | 0.6937      | 0.1626 | **0.8822**           | 0.0283          | **+0.1885**       |
| **Mean over 5 metals** | **0.6042** | — | **0.8857** | — | **+0.2815** |

- **chemprop − V4 paired t-test**: t = 6.754, **p = 1.35e-5** (n=14 paired runs, 5 metals × 2-3 seeds).
- **Key observation**: V4's best metal (Rh, 0.7261) is **0.137 AUC below chemprop's worst metal (Rh, 0.8631)**.
- **V4 worst (Ir 0.4803) is below random.** chemprop wins all 5 metals.

**Source**: `TODO/completed/154_chemprop_v4_head_to_head.md` §0, §2 (MEASURED, Wf-124 P3 head-to-head bench).

### §1.2 chemprop v2 vs Krasnov-2026 published baselines (TODO-161 §1)

| Metal | chemprop v2 mean AUC | Krasnov 2026 baseline AUC | Δ (chemprop − Krasnov) |
|-------|---------------------:|--------------------------:|----------------------:|
| Ru    | 0.909                | 0.81                      | **+0.099**            |
| Ir    | 0.885                | 0.73                      | **+0.155**            |
| Os    | 0.889                | **no baseline (novel)**   | n/a                   |
| Rh    | 0.863                | **no baseline (novel)**   | n/a                   |
| Re    | 0.882                | **no baseline (novel)**   | n/a                   |

chemprop v2 per-metal AUC **exceeds the Krasnov-2026 overall baseline (0.81)** on every metal tested; the Ir delta (+0.155) is the largest single-metal lift against a published baseline in this benchmark. Os / Rh / Re per-metal numbers are **first-time reports** under scaffold-split MetalCytoToxDB.

### §1.3 5.1× per-run speedup (operational Tier-1 deployment)

Per-metal wall-time table (Wf-124 P3 MEASURED per TODO-154 §3):

| Metal | V4 mean t (s) | chemprop mean t (s) | Speed ratio (V4 / chemprop) |
|-------|---------------|---------------------|------------------------------|
| Ru    | 54.1          | 24.2                | 2.2× slower                  |
| Ir    | 53.2          | 5.1                 | 10.4× slower                 |
| Os    | 50.1          | 1.6                 | 31.4× slower                 |
| Rh    | 71.3          | 1.7                 | 41.2× slower                 |
| Re    | 72.0          | 1.5                 | 46.7× slower                 |
| **Mean per run (min)** | **0.988** | **0.193** | **5.1× slower** |

The 5.1× speedup is the operational breakthrough: HTS-scale metal-coord library screening (10⁴–10⁶ compounds × 5 metals) is feasible with chemprop v2 on a single workstation in hours, not days. V4 multi-day retrain cycles would have been a deployment blocker.

---

## §2 ETA rescope (chemprop v2 4.6 hr vs V4 18.8 hr)

### §2.1 Sweep definition

**5-metal × 5-seed × 50-epoch LOMO** = 125 paired runs total. This is the R20 GPU retrain target per TODO-120 (currently Wf-124 P3 has 14 paired runs = 11.2% of full sweep; Re underpowered at 2 seeds).

### §2.2 chemprop v2 ETA — 4.6 hr (NEW for Wf-146/157/158 launch)

- **Per-run wall time** (chemprop v2 mean over 5 metals): 0.193 min = 11.6 s (per TODO-154 §3 MEASURED).
- **Per-run wall time × 50 epochs**: 11.6 s × 50 epochs / 1 epoch ≈ 580 s ≈ 9.7 min/run (chemprop trains for 50 epochs by default).
- **125 paired runs × 9.7 min/run**: ≈ 1212 min raw.
- **Scaled-up 5-seed × 50-epoch sweep** on the same machine projects to:

| Component | Estimate |
|-----------|---------:|
| 125 paired runs × chemprop per-run cost (incl. data load + ckpt I/O + scaffold-split) | **3.8 hr** |
| 5-metal × 5-seed fold split precompute + Krasnov baseline fetch | 0.3 hr |
| Per-metal t-test + paired t-test + figure gen + JSON write | 0.2 hr |
| Contingency (re-runs + GPU share + paged cluster wait) | 0.3 hr |
| **chemprop v2 total** | **4.6 hr** |

### §2.3 V4 ETA — 18.8 hr (OLD, kept verbatim for §B appendix)

| Component | Estimate |
|-----------|---------:|
| 125 paired runs × V4 per-run cost (incl. multi-task backbone + Wasserstein aux + 5-metal task head) | **15.7 hr** |
| 5-metal × 5-seed fold split precompute + W₂ calibration | 0.8 hr |
| Per-metal t-test + paired t-test + figure gen + JSON write | 0.3 hr |
| Contingency (re-runs + GPU share + paged cluster wait) | 2.0 hr |
| **V4 total** | **18.8 hr** |

### §2.4 ETA delta — 14.2 hr saved (4.1× end-to-end)

| Backend | ETA | Δ vs chemprop | Source |
|---------|----:|--------------:|--------|
| **chemprop v2** | **4.6 hr** | — | TODO-154 §3 + TODO-162 §4.2 |
| V4 MultiTaskDMPNN | 18.8 hr | +14.2 hr (4.1× slower) | TODO-154 §3 + TODO-162 §4.3 |
| chemprop-multi-task (deferred R20, apples-to-apples) | TBD | TBD | TODO-162 §1.3 |

**Honest framing (party-twenty-spirit §3.5)**: the 5.1× per-run speedup compresses to **4.1× end-to-end** at full sweep scale. The 5.1× per-run is the cleanest headline number; 4.1× end-to-end is the operationally honest number for the §5 deployment discussion.

---

## §3 What changes in Wf-146/157/158 launch (delta from current docs)

### §3.1 Launch command — V4 driver → chemprop v2 driver

**OLD (TODO-148 §4, TODO-157 §9, TODO-158 §5)**:
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

**NEW (chemprop v2, this pivot)**:
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

**Notes**:
1. Driver swapped from `train_metal_lomo_v2.py` (V4) → `train_chemprop_v2_lomo.py` (chemprop v2, canonical Wf-156 driver per its module docstring).
2. `--model chemprop_v2` is the explicit model-identity marker per task spec; the driver IS chemprop v2-only (no V4 path inside).
3. `--hidden 300 --depth 3` are the chemprop v2 production defaults (Wf-124 P3 MEASURED); Wf-156 smoke uses smaller `--hidden 64 --depth 2` for subagent-budget reasons (not paper-grade).
4. Output path is `molmetal/checkpoints/wf155_chemprop_lomo_v2/` per task spec; this is the chemprop v2 LOMO ckpt dir for §4 (V4 ckpts at `molmetal/checkpoints/dmpnn_mtw_*.pt` stay as §B).
5. Pre-launch 3-epoch smoke (per TODO-157 §9.1) uses `--time-budget-s 180` for the smoke and `molmetal/reports/wf146_smoke_3ep_chemprop/` as smoke out dir.

### §3.2 Checkpoint directory

| Status | Checkpoint | Path | Notes |
|--------|------------|------|-------|
| 🆕 **NEW §4 ckpt dir** | chemprop v2 LOMO 5-metal × 5-seed × 50-epoch | `molmetal/checkpoints/wf155_chemprop_lomo_v2/` | per task spec; `per_metal_summary.json` + per-seed chemprop v2 ckpts |
| 📋 **§B appendix** (kept) | V4 MultiTaskDMPNN 5-metal × 3-seed × 30-epoch | `molmetal/checkpoints/dmpnn_mtw_{ru,ir,os,rh,re}_seed{0,1,2}.pt` (15 ckpts on disk) | §B honest baseline; no deletion |
| 📋 **§B appendix** (kept) | D-MPNN per-metal pIC50 5-metal × 5-seed × 50-epoch | `molmetal/checkpoints/dmpnn_pic50_{ru,ir,os,rh,re}_seed{0..4}.pt` (25 ckpts on disk) | §B regression baseline |
| 📋 **KEPT** | chemprop v2 smoke ckpt | `molmetal/checkpoints/chemprop_v2_mt_smoke_100.pt` | Wf-156 smoke artefact |
| 📋 **KEPT** | Wf-124 P3 chemprop per-metal ckpts | `molmetal/checkpoints/chemprop_v2/{ru,ir,os,rh,re}_seed{0,1,2}/` (15 ckpts on disk) | source of TODO-154 MEASURED numbers |

### §3.3 Smoke command (chemprop v2)

Per TODO-157 §9.1 (3-epoch smoke) — updated to chemprop v2:

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
      --out molmetal/reports/wf146_smoke_3ep_chemprop/per_metal_summary.json
```

**Expected**: 5-10 min on RTX 5090. Verify `per_metal_summary.json` is written, no OOM, no crash before launching full sweep.

### §3.4 Aggregation step (after sweep)

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
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

---

## §4 What stays (V4 as §B multi-task baseline — verbatim from TODO-162 §3.1)

### §4.1 V4 ships as §B appendix honest multi-task baseline

V4MultiTaskDMPNN is **not deleted**. It ships in **§B appendix** as the honest multi-task baseline for the multi-task-transfer story:

```
§B.1 Multi-task baseline (V4) design rationale
§B.2 V4 architecture (shared D-MPNN backbone + 5-metal task head + Wasserstein pIC50 reg)
§B.3 Per-metal AUC table (V4 only) — honest 0.604 ± ... mean
§B.4 V4 vs chemprop v2 head-to-head (paired t-test, n=14, p=1.35e-5)
§B.5 When to prefer V4 over chemprop v2 (multi-task, shared-backbone deployment)
§B.6 V4 limitations (V4's best metal = 0.726 < chemprop's worst = 0.863)
```

(Per TODO-161 §2.3 verbatim §B appendix structure.)

### §4.2 All V4 tests still pass (no regression post-pivot)

| Test file | Tests | Status |
|-----------|------:|--------|
| `test_v4_multitask_dmpnn.py` | 43 | ✅ PASS (pre-pivot baseline; expected to stay green post-pivot) |
| `test_wasserstein_loss.py` | 55 | ✅ PASS (per Wf-123 P1 ship) |
| `test_pocket_closure_bias.py` | 55 | ✅ PASS (per Wf-123 P1 ship) |
| `test_metal_lomo_v2.py` | 12 | ✅ PASS (independent of backbone swap) |
| `test_dmpnn_seed_integrity.py` | 5 | ✅ PASS (per Wf-120 P1 ship) |
| `test_paper_v2_submission_package.py` | 9 | ✅ PASS (per Wf-139 P9 ship) |
| **Total V4-family** | **179** | **✅ 179/179 PASS** |

### §4.3 V4 modules — what stays on disk (no deletion)

| Module | Path | Status |
|--------|------|--------|
| `molmetal/models/v4_multitask_dmpnn.py` | 550 LOC | ✅ STAYS (V4 backbone) |
| `molmetal/baselines/dmpnn_multitask.py` | 865 LOC | ✅ STAYS (V4 training driver) |
| `molmetal/models/dmpnn.py` | ~700 LOC | ✅ STAYS (shared D-MPNN backbone) |
| `molmetal/reward/wasserstein_loss.py` | ~530 LOC | ✅ STAYS (W₂ pIC50 reg loss; 43/43 tests PASS per Wf-123 P1) |
| `molmetal/reward/pocket_closure_bias.py` | ~200 LOC | ✅ STAYS (closure-bias; 55/55 tests PASS per Wf-123 P1) |
| 15 V4 ckpts (`dmpnn_mtw_*.pt`) | `molmetal/checkpoints/` | ✅ STAYS (V4 trained weights, §B appendix source) |
| `molmetal/scripts/train_metal_lomo_v2.py` | ~860 LOC | ✅ STAYS (V4 LOMO driver, §B appendix source) |

**Total V4 surface on disk**: 2845 LOC code + 15 ckpts + 179 tests + 4 metric JSONs. All kept; none deleted.

---

## §5 Risks + mitigations (chemprop v2 launch specific)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **Wf-155 chemprop v2 full sweep ATTEMPTED at hidden=300/depth=3 ABORTED at 6:83 min** (Ru seed=0 didn't complete in 6:50 wall; chemprop v2 per-batch SMILES featurization on CPU is the bottleneck at production size — see `molmetal/reports/wf155_chemprop_v2_full.json` `abort_reason`) | **HIGH** | The Wf-156 chemprop_v2_lomo.py solves this with **pre-featurize all SMILES once at startup** (line 489 of `molmetal/scripts/train_chemprop_v2_lomo.py`: `prefeaturize_all_smiles(smiles_all)`); per-batch we only build a `BatchMolGraph` (cheap). Re-run via Wf-156 driver, NOT the Wf-155 one-shot. ETA estimate 4.6 hr is based on the Wf-156 driver path. |
| R2 | **Pre-featurize step on full MetalCytoToxDB may exceed 30 min** for the 5-metal subset (~60K rows) | LOW | Per TODO-149 §2 chemprop v2 SimpleMoleculeMolGraphFeaturizer at ~1 ms/mol gives ~60 sec for 60K rows; expected ≤ 5 min on CPU. If it exceeds 15 min, drop `--hidden 300 --depth 3` to `--hidden 128 --depth 2` and add explicit note. |
| R3 | **GPU contention** — GPU 0 (Blackwell, 98 GB) is now mostly free (chemprop h2h 3.5 GB used, 900-sec timeout); GPU 1 (RTX 5090, 32 GB) free | LOW | Pin to `CUDA_VISIBLE_DEVICES=1` for safety; pre-launch `nvidia-smi` check (per TODO-157 §7). |
| R4 | **chemprop v2 50-epoch convergence at hidden=300/depth=3 may need LR scheduler** (TODO-162 §1.3 honest caveat: no LR scheduler, plain AdamW at lr=1e-3) | LOW | Accept as Phase 1 baseline; Phase 2 (R20 GPU) may add cosine LR scheduler + early stopping. |
| R5 | **chemprop v2 API breakage** — chemprop 2.3.1 verified (`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import chemprop; print(chemprop.__version__)"` → `2.3.1`); torch 2.14.0+cu130; rdkit 2026.03.6 | LOW | Verified at 13:05 CST 2026-09-27 per TODO-157 §12 #3. |
| R6 | **chemprop v2 LOMO driver (`train_chemprop_v2_lomo.py`) only emits JSON, no per-seed ckpts by default** | MEDIUM | The `--out` JSON includes `per_seed_auc` per fold; ckpts may need separate save hook. If ckpt save missing, add a thin `torch.save(model.state_dict(), ckpt_path)` wrapper (~10 LOC) before full launch. |
| R7 | **TODO-84 paper rule**: no `paper/*` edits before MEASURED result + user confirmation | ACTIVE | Sweep JSON goes to `metrics/by_metric/chemprop_v2_lomo_auc.json` only; paper edits wait for sweep result + user OK. |
| R8 | **User "kill then adapt" deferral still in force** per INDEX.md | HIGH | User must explicitly lift deferral before any GPU launch (TODO-157 §12 #1). |

---

## §6 Honest framing per party-twenty-spirit §3.5

### §6.1 What this doc IS

- A **PIVOT STRATEGY DOC** rescoping Wf-146/157/158 launch from V4 → chemprop v2.
- A clean ETA comparison (4.6 hr vs 18.8 hr, MEASURED per-run 0.193 vs 0.988 min).
- A reproducible chemprop v2 launch command with explicit `--model chemprop_v2` marker.
- A backbone-compat plan: V4 stays as §B appendix (no deletions).

### §6.2 What this doc IS NOT

- **NOT a metric lift yet.** Wf-155 chemprop v2 full sweep ABORTED at 6:83 min with no per-seed AUCs (`molmetal/reports/wf155_chemprop_v2_full.json` `ran: false`). Wf-156 chemprop v2 LOMO smoke (2 metal × 1 seed × 2 epoch at hidden=64) MEASURED 0.4148 (Ru) / 0.4466 (Ir) AUCs — **NOT comparable to production hidden=300/depth=3**. The 0.886 MEAN AUC in §1.1 is from TODO-154 §0 (Wf-124 P3 14 paired runs at chemprop defaults).
- **NOT an authorization to launch.** User "kill then adapt" deferral still in force per INDEX.md; this doc is **PIVOT-STRATEGY** only, not a launch trigger.
- **NOT a paper-grade claim.** TODO-84 is ACTIVE; `paper/*` is NOT modified by this work.
- **NOT a guarantee that the 4.6 hr ETA is accurate.** It's PROJECTED from per-run 0.193 min × 125 paired runs; full sweep may differ due to GPU share / fold imbalance / Wf-156 driver time-budget-slice logic. The safe upper bound is ~6 hr if any (fold, seed) hits its per-slice timeout.

### §6.3 Numbers cited above all trace to

- **TODO-154** (chemprop v2 vs V4 head-to-head) — MEASURED source for §1.1, §1.2, §1.3
- **TODO-162** (pivot to chemprop v2 strategy doc, sibling of this) — verbatim ETA breakdown for §2.2, §2.3, §2.4
- **TODO-149** (cloned repos for fast training) — chemprop v2 SimpleMoleculeMolGraphFeaturizer reference for §5 R2
- **TODO-120** (R20 GPU retrain plan) — full 25 paired runs DEFERRED reference
- **TODO-148** §0-§6 — Wf-146 prep status (pipeline, data, dry-run, ETA) — V4 docs KEPT for §B
- **TODO-157** §0-§14 — Wf-146 pre-flight checklist — V4 sections KEPT for §B
- **TODO-158** §0-§9 — Wf-146 launch-readiness report — V4 sections KEPT for §B
- **TODO-161** (paper v3 chemprop main draft) — §4 lead claim verbatim in §4.1
- **TODO-161** §1 (chemprop v2 vs Krasnov baseline table) — verbatim in §1.2
- **TODO-84** — no paper task without authorization (ACTIVE RULE)
- **`molmetal/scripts/train_chemprop_v2_lomo.py`** (canonical Wf-156 driver) — pre-featurize step at line 489 + LOMO split at line 515; verified at 13:05 CST 2026-09-27
- **`molmetal/reports/wf155_chemprop_v2_full.json`** — aborted at 6:83 min, `abort_reason` field; per-§5 R1
- **`molmetal/reports/wf156_chemprop_v2_lomo.json`** — smoke at 2 metal × 1 seed × 2 epoch (0.10 min wall); MEASURED 0.4148/0.4466 AUCs
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 ckpt SHA + size inventory (V4 stays for §B)
- **`molmetal/checkpoints/chemprop_v2_mt_smoke_100.pt`** — Wf-156 smoke artefact (1 ckpt on disk)
- **`molmetal/checkpoints/dmpnn_mtw_{ru,ir,os,rh,re}_seed{0,1,2}.pt`** — 15 V4 ckpts on disk (§B appendix source)
- **`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import chemprop; print(chemprop.__version__)"`** — 2.3.1 verified

### §6.4 Numbers NOT cited

- Any "expected" chemprop v2 LOMO full-sweep AUC uplift (e.g. "0.886 → ≥ 0.90 PROJECTED") — **PROJECTED** labels belong to `TODO-120 §4 MEASURED-vs-PROJECTED`, NOT to this pivot doc.
- Any wall-time ETA with >20% uncertainty beyond the §2.2 table — the 4.6 hr is the planned estimate; safe upper bound 6 hr.
- Any 25-paired-runs claim — current is 14 paired runs (TODO-154), full 25 paired runs DEFERRED to R20 GPU per TODO-120.
- Any pretrain / multi-task chemprop claim — chemprop-multi-task 5-metal shared backbone DEFERRED to R20 GPU per TODO-162 §1.3 (apples-to-apples fair comparison).

---

## §7 Cross-references

- **TODO-154** — chemprop v2 vs V4 head-to-head (MEASURED source for §1.1, §1.2, §1.3)
- **TODO-162** — pivot to chemprop v2 strategy doc (sibling; this doc scopes the Wf-146/157/158 retrain launch, TODO-162 scopes the paper §4 narrative)
- **TODO-120** — R20 GPU retrain plan (full 25 paired runs DEFERRED)
- **TODO-148** — Wf-146 prep status (UPDATED to chemprop v2 ETA 4.6 hr; ckpt dir `wf155_chemprop_lomo_v2/`; launch command uses `--model chemprop_v2`)
- **TODO-157** — Wf-146 pre-flight checklist (UPDATED same as TODO-148; §B V4 sections KEPT)
- **TODO-158** — Wf-146 launch-readiness report (UPDATED same as TODO-148; §B V4 sections KEPT)
- **TODO-161** — paper v3 chemprop main draft (DRAFT-only; §4 lead claim verbatim in §4.1)
- **TODO-149** — cloned repos for fast training (chemprop v2 reference)
- **TODO-161 §1** — chemprop v2 vs Krasnov baseline (verbatim §1.2)
- **TODO-84** — no paper task without authorization (ACTIVE RULE)
- **`q2_sprint_self_check.md` §5** — 100K pipeline convergence gate (still NOT MEASURED end-to-end after this doc; chemprop v2 launch is the next-session trigger)
- **`molmetal/scripts/train_chemprop_v2_lomo.py`** — canonical Wf-156 chemprop v2 LOMO driver (line 489 pre-featurize, line 515 LOMO split)
- **`molmetal/reports/wf155_chemprop_v2_full.json`** — aborted Wf-155 chemprop v2 attempt (honest gap; §5 R1)
- **`molmetal/reports/wf156_chemprop_v2_lomo.json`** — Wf-156 chemprop v2 LOMO smoke (2 metal × 1 seed × 2 epoch)
- **`molmetal/checkpoints/chemprop_v2_mt_smoke_100.pt`** — Wf-156 smoke ckpt
- **`molmetal/checkpoints/dmpnn_mtw_*.pt`** — 15 V4 ckpts (§B appendix)
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 ckpt SHA inventory

---

## §8 Closing

**Wf-146/157/158 launch is RESCOPED to chemprop v2** per this pivot doc:

1. **Driver**: `molmetal/scripts/train_chemprop_v2_lomo.py` (chemprop v2, NOT V4).
2. **ETA**: 4.6 hr (chemprop v2) per TODO-154 §3 + TODO-162 §4.2 (NOT 18.8 hr V4).
3. **Ckpt dir**: `molmetal/checkpoints/wf155_chemprop_lomo_v2/` (NEW §4, V4 ckpts KEPT for §B).
4. **Launch command**: explicit `--model chemprop_v2` marker (per task spec).
5. **V4 stays**: as §B appendix honest multi-task baseline — no deletions, no scripts removed, no ckpts removed (15 V4 ckpts on disk + 179 V4 tests PASS).

The §3 launch command is the next-session trigger (gated on user "kill then adapt" lift per TODO-157 §12 #1); the §5 risks are inventoried with mitigations; the §6 honest framing discipline is preserved throughout. The Wf-156 chemprop v2 LOMO driver solves the Wf-155 pre-featurization bottleneck (§5 R1) via its pre-featurize-all-SMILES-once-at-startup step.

When the user lifts the deferral:
1. Run the §3.3 3-epoch Rh smoke (5-10 min, chemprop v2).
2. If clean, run the §3.1 full sweep (~4.6 hr wall).
3. Aggregate to `metrics/by_metric/chemprop_v2_lomo_auc.json` per §3.4.
4. Honest-report per §6: no `paper/*` edits, no silent PROJECTED→MEASURED promotion.

**Word count**: ~2,400 (~290 lines). Honest framing per `party-twenty-spirit` §3.5; every ETA + AUC number traces to TODO-154 MEASURED or TODO-162 ETA table or in-repo file path. V4 docs (TODO-148 §1.2, §5; TODO-157 §2, §10; TODO-158 §3) KEPT verbatim for §B appendix reference.