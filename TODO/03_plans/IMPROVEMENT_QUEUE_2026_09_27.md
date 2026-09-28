# IMPROVEMENT_QUEUE — improvement tasks + acceptance gates + serial execution

**Date**: 2026-09-27
**Owner**: Hugo Lee + ultracode agents (Wf-139 → Wf-142)
**Status**: 🟢 ACTIVE — most tasks already shipped; 4 remaining + Wf-139 P5 done this session
**Parent**: TODO-131 weak metric lift master plan + audit #1/#2/#3

---

## §0 枚举结果 (Enumerate Result — actual state)

**Total tasks in `pending/` originally counted**: 17 (numbered 119-138)
**Already SHIPPED (completion doc exists)**: 16/17
**Still PENDING (real work to do)**: 4/17 (TODO-134, 135, 136 §3, 138)

### Already SHIPPED (16):
| ID | Topic | Completion doc |
|---|---|---|
| 119 | Decode_ratio strengthen (YuelBond+Wasserstein) | `completed/119_closeout_v2_loi_sop.md` |
| 120 | D-MPNN seed integrity | `completed/120_dmpnn_seed_integrity.md` |
| 121 | m19 LOMO strengthen | `completed/121_m19_lomo_strengthen.md` |
| 122 | Hilbert L2 ASTSR guard | `completed/122_hilbert_l2_astsr_guard_closeout.md` |
| 123 | m15 RMSD strengthen | `completed/123_m15_rmsd_strengthen.md` |
| 124 | m5 SA synth strengthen | `completed/124_m5_sa_synth_strengthen.md` |
| 125 | Rh oversample wasserstein transfer | `completed/125_rh_oversample_wasserstein_transfer_closeout.md` |
| 126 | Innovation strengthen roadmap | `completed/126_innovation_strengthen_roadmap.md` |
| 127 | L5 Hilbert ASTSR rescue | `completed/127_l5_hilbert_astsr_rescue.md` |
| 128 | L10 3-way real pIC50 rescue | `completed/128_l10_3way_real_pic50_rescue.md` |
| 129 | L2 L4 L7 joint rescue | `completed/129_l2_l4_l7_joint_rescue.md` |
| 130 | Wf-130 P1 Catherine 30 | `completed/130_wf130_p1_catherine_30.md` |
| 131 | m1 m19 joint rescue | `completed/131_m1_m19_joint_rescue.md` |
| 132 | Wf-130 P3 OOD 30 metals | `completed/132_wf130_p3_ood_30metals.md` |
| 133 | L1 L6 L12 closure full | `completed/133_l1_l6_l12_closure_full.md` |
| 137 | L5 L7 L10 arch ceiling honest | `completed/137_l5_l7_l10_arch_ceiling_honest.md` |
| 139 | Wf-139 P5 (prior session) | `completed/139_wf139_p5_cfm_decode_real.md` |
| **142** | **Wf-139 P6 (this session!)** | `completed/142_wf139_p6_sa_closure_bias_generation.md` |

### Still PENDING (3 — actual remaining work):
| # | ID | Task | Priority | Acceptance Gate | Source |
|---:|---|---|---|---|---|
| 1 | **Wf-139 P5** (TODO-136 §2) | m14 real CFM decode benchmark | P1 | ≥100 samples AND pytest ≥12 PASS AND real > 0 (PARTIAL ok) | ✅ **DONE 2026-09-27** |
| 2 | **Wf-139 P6** (TODO-136 §1 Phase 1) | m5 SA closure-bias generation 5pocket × 100 | P1 | aggregate SA ≤ 2.8 (Q2 stretch) AND ≥10 tests PASS | ✅ **DONE 2026-09-27** |
| 3 | TODO-134 / TODO-136 §3 | m15 Docking RMSD redesign Phase 2 (Vina redocking) | P0 | per-pocket median best RMSD ≤ 2.5 Å (≥1 pocket ≤ 2.0 Å SOTA success) | `pending/134_*.md` |
| 4 | TODO-135 | m19 LOMO Rh oversample regression fix + 5-seed re-bench | P0 | Rh fold AUC ≥ 0.30 (recover from 0.136 regression) | `pending/135_*.md` |
| 5 | TODO-138 | Paper v2 submission package (Q2 JCIM) | P0 | paper builds clean + cover letter + figures + supp | `pending/138_*.md` |

---

## §1 验收门槛协议 (Acceptance Gate Protocol)

每个 task 完成前必须满足:
1. **Implementation Gate**: 脚本/模块存在 + 可 import + 可调用
2. **Test Gate**: pytest ≥12 tests PASS（per task §X verify）
3. **Metric Gate**: 数值 MEASURED（不假装）
4. **Honest-Framing Gate**: 不达目标时显式 ship honest-negative（per `party-twenty-spirit` §3.5）
5. **Registry Gate**: `metrics/by_metric/<metric>.json` REFRESHED
6. **Completion Doc Gate**: `TODO/completed/<WF_ID>_*.md` ship

如果任何 gate fail, **HALT pipeline** — 不进入下一项。

---

## §2 串行执行顺序 (Serial Execution Order)

### ✅ DONE this session
- **Wf-139 P5 (TODO-136 §2)**: real CFM decode_ratio = 0.9745 (PARTIAL, expected)
  - 19/19 tests PASS
  - 274 valid Pt samples
  - `metrics/by_metric/decode_ratio.json` refreshed
  - `TODO/completed/139_wf139_p5_cfm_decode_real.md` shipped
- **Wf-139 P6 (TODO-136 §1 Phase 1)**: closure-bias SA generation 5pocket × 100 × 3 seeds
  - **Aggregate SA = 2.463 (PASS ≤2.8 stretch)**
  - **Per-pocket**: 1h36=2.373 / 3eml=2.465 / 3ny8=2.660 / 4aaw=2.359 (best) / 4rlu=2.456
  - **5/5 pockets pass ≤3.0; 5/5 pass ≤2.8; 4/5 pass TargetDiff 2.65 ceiling**
  - 16/16 tests PASS; RDKit validity = 100%; Lipinski Ro5 = 100%
  - 231 emissions / 57 unique (matches Wf-140 P1 reachability ceiling)
  - `metrics/by_metric/sa_mean.json` refreshed (12 measurements)
  - `TODO/completed/142_wf139_p6_sa_closure_bias_generation.md` shipped

### 🟡 NEXT: 任务 #2 — TODO-134 / TODO-136 §3 (m15 Docking RMSD redesign Phase 2)
**Status**: pending, ready to start
**Inputs**:
- TODO-134 (master plan, 280 LOC)
- TODO-136 §3 (Phase 2 spec, 250 LOC)
- Wf-126 P1 5/5 pockets READY
- Wf-121 P3 `_dock_one_python_binding` function

**Output**:
1. `molmetal/scripts/bench_docked_pose_rmsd.py` (~280 LOC)
2. `molmetal/tests/test_docked_pose_rmsd.py` (≥12 tests)
3. `molmetal/reports/wf134_p2_redocking/{pocket}/per_seed_metrics.json` (5 dirs)
4. `molmetal/reports/wf134_p2_redocking/synthesis.md`
5. `metrics/by_metric/docked_pose_rmsd.json` (NEW)
6. `metrics/by_metric/rigid_fragment_rmsd.json` (PATCHED — HONEST_NEGATIVE_REDESIGNED)

**Verify**:
- 5 pocket × 3 seed × n_poses=5 = 75 docked poses
- Per-pocket median best RMSD ≤ 2.5 Å (SOTA paper 接受)
- ≥ 1 pocket ≤ 2.0 Å (SOTA success)

**Constraints**: CPU + GPU-0; honest framing

### ⏳ Then: 任务 #3 — TODO-135 (m19 LOMO Rh oversample regression fix)
**Status**: pending, depends on GPU availability
**Action**: 5-seed × 30-epoch re-bench with positive-label Rh oversample
**Verify**: Rh fold AUC ≥ 0.30 (recover from 0.136)

### ⏳ Then: 任务 #4 — TODO-138 (Paper v2 submission package)
**Status**: pending, depends on 134 + 135 + Wf-139 P5
**Action**: compile paper v2 + cover letter + figures + supp
**Verify**: paper builds clean (0 unresolved refs) + JCIM submission checklist complete

---

## §3 进度跟踪 (Progress Ledger)

| Wf | Status | Started | Gate Status | Notes |
|---|---|---|---|---|
| **Wf-139 P5** | ✅ **COMPLETE** | 2026-09-27 | ✅ PASS (PARTIAL) | real 0.9745 / synthetic 1.000 — gate met; 19/19 tests pass |
| **Wf-139 P6** | ✅ **COMPLETE** | 2026-09-27 | ✅ PASS (PASS) | aggregate SA = 2.463 (PASS ≤2.8); 16/16 tests; 5/5 pockets pass |
| Wf-134 P2 (TODO-134) | ⚪ QUEUED | — | — | m15 Docking RMSD redesign Phase 2 (Vina redocking) |
| Wf-135 (TODO-135) | ⚪ QUEUED | — | — | m19 Rh oversample regression fix |
| Wf-138 (TODO-138) | ⚪ QUEUED | — | — | Paper v2 + JCIM submission package |

---

## §4 Wf-139 P5 完成摘要

- **Real CFM decode_ratio**: 0.9745 (267/274) on PlatinAI 5K Pt subset
- **Synthetic baseline**: 1.000 (Wf-124 P2 ship)
- **Gap**: −0.0255 (~2.5pp; expected per honest framing)
- **Tests**: 19/19 PASS
- **Completion doc**: `TODO/completed/139_wf139_p5_cfm_decode_real.md`
- **Metric refresh**: `metrics/by_metric/decode_ratio.json` updated

## §4b Wf-139 P6 完成摘要

- **Aggregate SA mean (5pocket closure-bounded)**: 2.463 (simple) / 2.458 (weighted, n=231)
- **Unconstrained baseline**: 3.099 → Wf-139 P6 lift = **−0.636** (−20.5% relative)
- **Per-pocket**: 1h36=2.373 / 3eml=2.465 / 3ny8=2.660 / 4aaw=2.359 (best) / 4rlu=2.456
- **5/5 pockets pass target ≤3.0; 5/5 pass stretch ≤2.8; 4/5 pass TargetDiff 2.65 ceiling**
- **RDKit validity = 100%, Lipinski Ro5 = 100%, QED 0.50-0.68**
- **16/16 tests PASS**
- **Completion doc**: `TODO/completed/142_wf139_p6_sa_closure_bias_generation.md`
- **Metric refresh**: `metrics/by_metric/sa_mean.json` updated (12 measurements; `wf139_p6_status` field)

---

## §5 Honest Framing (per party-twenty-spirit §3.5)

- 不假装 ship 全部 — 显式 ship honest-negative 如果 fail
- 不假装 GPU retrain 收敛 — 真实记录数值
- 不假装 paper § lead — 真实 MEASURED 数字

---

## §6 Final Verdict

- **18 tasks originally counted** (added Wf-139 P6): 17 already shipped (from prior sessions), 1 done this session, 0 remaining "new" tasks
- **3 still pending**: TODO-134, TODO-135, TODO-138 (all P0; 1-3 weeks serial)
- **Realistic ETA**: 1-3 weeks serial (with GPU dependencies)
- **Fallback**: paper §4 + §6 ship current 8 真 SOTA + 8 honest-negative — Q2 ready

---

## §7 How to start the next task

When user authorizes Wf-134 P2 (m15 Docking RMSD redesign Phase 2):
```bash
cd /home/hugo/codes/mol-metal
source env.sh
# Read the spec
cat TODO/pending/134_m15_docking_rmsd_redesign.md
cat TODO/pending/136_m5_m14_m15_residual_lift.md  # §3
# Run the bench
CUDA_VISIBLE_DEVICES= uv run --no-sync python -u molmetal/scripts/bench_docked_pose_rmsd.py
```
