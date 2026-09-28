# Wf-22 (Blackwell GPU retrain) — synthesis report

**Date**: 2026-09-26
**Workflow slot**: Wf-22 in INDEX.md task ledger (corresponds to Wf-23 GPU ultracode queue, 4-week schedule)
**Parent plan**: `TODO/03_plans/R18_PLAN.md` §15.5 R19+ roadmap priority 1 (Block GPU training) + `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` §2 Stage 1.4
**Final report**: `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` (this run; live JSON)
**Reproduce guide**: `molmetal/reports/wf_blackwell_guide/R18_GPU_TRAINING_GUIDE.md`

---

## 1. TL;DR — Blackwell blocked 误判 推翻

**结论**: Blackwell GPU 0 实际可用 + GPU 1 完全空闲; "Blackwell 30-min CI budget" / "Wf-23 4-week 排队" 的保守 framing 是 **保守过头**, 不是真实物理约束。

| 资源 | 数值 | 状态 |
|---|---|---|
| **GPU 0 — RTX PRO 6000 Blackwell (sm_120)** | 95.6 GB total VRAM / **85.1 GB free** / 19% util | ✅ 实际可用 (≈ 98 GB rounded) |
| **GPU 1 — RTX 5090 (sm_120)** | 31.8 GB total VRAM / **31.4 GB free** / 0% util | ✅ 完全空闲 |
| **合计 free VRAM** | **+116.5 GB** | ✅ 远超任何 single-GPU retrain 需求 |
| **torch + CUDA** | `torch==2.14.0+cu130` + CUDA 13.0 + sm_120 | ✅ Blackwell-native |
| **kanzi contention** | PID 3515656, 9.96 GB on GPU 0 | ⚠️ 占 GPU 0 ≈ 10 GB, 仍剩 85 GB |

**推翻点**: `R18_PLAN.md` §10.4 / §15.5 把 Blackwell 标成 "30-min CI budget 排期 4 周" — 这是 **CI 排队策略** (避免与 kanzi_venv sweep 抢 GPU 0), 不是 GPU 物理容量约束。Wf-22 把这个 framing 修正为: **GPU 0 + GPU 1 = 116.5 GB free VRAM**, 远超过 R18 retrain 的全部需求, Wf-23 4-week 排队不是必要的物理上界, 而是 **kanzi 共存策略**。

---

## 2. Per-deliverable (5 deliverables, 1 SHIP + 4 R18 lift claims)

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| **D1** | GPU env check (Blackwell + RTX 5090 dual probe) | ✅ SHIP today | `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json`; `molmetal/scripts/check_blackwell_env.py` |
| **D2** | **R18 retrain M22 (SE(3) Schrödinger Bridge)** | ⏳ Phase 2 in queue | `molmetal/adapters/se3_sb_integration.py` adapter ready (Wf-1 T1 + Wf-2 W2.1 ship today); Blackwell retrain target m3 ≤ −10.5 (R17 baseline −8.5) |
| **D3** | **R18 retrain M23 (Transport Relaxation)** | ⏳ Phase 1 next week | `molmetal/adapters/transport_relaxation_integration.py` adapter ready (Wf-2 W2.2 ship today); Blackwell retrain target m14 = 1.000 (no regression) + cfm_loss < 25 (R17 baseline 252330) |
| **D4** | **R18 retrain M24 (Multi-PH)** | ⏳ Phase 3 week 6-7 | `molmetal/adapters/multi_ph_integration.py` adapter ready (Wf-1 T5 + Wf-2 W2.3 + Wf-15 Wf-17 ship today); Blackwell retrain target m7 ≥ 0.75 (R17 baseline 0.137) |
| **D5** | **5-seed × 5-metal × 4 conditions benchmark** | ⏳ Phase 4 joint retrain | Per `train_ru_multitask.py` Stage 1.4 recipe: 5 conditions = (i) random init, (ii) tmQM pretrain, (iii) multi-task, (iv) multi-task + tmQM pretrain (+ (v) joint SE(3) SB / Multi-PH); 5 metals = Ru/Ir/Rh/Re/Os; 5 seeds {0..4} × 100 epochs × batch=128 BF16; target m1 Ru AUC ≥ 0.86 (Krasnov 2026 reference 0.81) |
| **D6** | **Paper v2 update** | ⏳ Phase 4 week 8 | Replace PROJECTED → MEASURED in `paper/sections/03_method.tex` §3.4 SE(3) SB + §3.5 Transport Relax + §3.6 Multi-PH + `paper/sections/04_results.tex` Table 2 R18 row |

---

## 3. MEASURED vs PROJECTED

### 3.1 MEASURED (Wf-22 today, 2026-09-26)

| Metric | Value | Source |
|---|---|---|
| GPU 0 VRAM free | 85.1 GB / 95.6 GB total | `gpu_env_check.json` |
| GPU 1 VRAM free | 31.4 GB / 31.8 GB total | `gpu_env_check.json` |
| Combined free VRAM | 116.5 GB / 127.4 GB total | sum |
| GPU 0 compute capability | sm_120 (Blackwell-native) | torch probe |
| torch + CUDA | 2.14.0+cu130 / CUDA 13.0 | torch probe |
| kanzi contention | 9.96 GB on GPU 0, util 19% | nvidia-smi |
| Wf-22 ship size | 1 JSON + 1 guide (350 lines) | `wf_blackwell_guide/R18_GPU_TRAINING_GUIDE.md` |
| GPU retrain wall-time | 0 (CPU-only skeleton ship today; GPU queue 待 launch) | Wf-22 status |
| Ru AUC MEASURED lift | **none yet** — Phase 4 joint retrain 排本周 | honest framing |

### 3.2 PROJECTED (R18 Phase 1-4 GPU ultracode, Blackwell + RTX 5090)

> 4 conditions × 5 metals × 5 seeds = 100 (seed × metal × condition) cells; mean+std 报告

| Metric | R17 baseline (1-seed, plain D-MPNN) | R18 PROJECTED | Source |
|---|---|---|---|
| **m1 Ru AUC** | **0.7123** (1 seed × 5 epoch) | **≥ 0.86** (5 seeds × 100 epoch + multi-task + tmQM pretrain + M22 + M29) | `train_ru_multitask.py:7` + `R18_PLAN.md` §15.2 |
| m21 NCI-60 9-cell AUC | 0.80 | 0.85 (M24 + M22 + M29) | `R18_PLAN.md` §15.2 |
| m3 Vina Dock median | −8.5 | ≤ −10.5 (M22 + M27 + M29 + M30 + M37) | `R18_PLAN.md` §15.2 |
| m7 Diversity (Morgan ECFP4) | 0.65 | ≥ 0.75 (M24 + M28 + M35 + M36 + M39 + M44) | `R18_PLAN.md` §15.2 |
| m14 Decode ratio | 1.000 | 1.000 (no regression, **HARD GATE**) | `R18_PLAN.md` §15.2 |

**Lift 实际数值 (vs R17 baseline 0.712)**: PROJECTED **+0.15 absolute AUC** (target ≥ 0.86) — driven by 5 ablation conditions stacked (random init → tmQM pretrain → multi-task → M22 SE(3) SB → M24 multi-PH). Per-condition partial lifts (current best estimate from R18 §15.2 ablations):

| Condition | Partial Ru AUC | Δ vs R17 baseline 0.712 |
|---|---|---|
| Random init (multi-task) | ~0.74 (R17 3-seed pilot) | +0.03 |
| + tmQM pretrain (encoder init) | ~0.77 (MEASURED +0.046 lift per `53_phase_a_reality_check.md`) | +0.06 |
| + 5-metal multi-task × 100 epoch | ~0.81 (match Krasnov 2026) | +0.10 |
| + M22 SE(3) SB retrain | ~0.84 (geometric correctness lift) | +0.13 |
| **+ M24 multi-PH + Joint retrain (D5 final)** | **≥ 0.86 (target)** | **+0.15** |

---

## 4. 推翻 "Blackwell 30 min budget" 误判

**原始 framing** (`R18_PLAN.md` §15.5 + `wf_blackwell_guide/R18_GPU_TRAINING_GUIDE.md` §1.1):
> "30 min per ultracode workflow (CI ceiling); ≤ 3 h for full R18 Phase 2 retrain"

**误判来源**: 把 **CI workflow timeout** (30 min default for `.github/workflows/ci.yml` jobs) 与 **GPU 物理 capacity** 混淆。

**Wf-22 推翻证据** (`gpu_env_check.json` 2026-09-26T01:59:40+0800):

```
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  VRAM total 95.6 GB / free 85.1 GB / util 19% / temp 54°C / power 148 W
GPU 1: NVIDIA GeForce RTX 5090
  VRAM total 31.8 GB / free 31.4 GB / util 0% / temp 43°C / power 9 W
kanzi contention: 9.96 GB on GPU 0 (PID 3515656, etime 34:46)
torch probe: torch 2.14.0+cu130, CUDA 13.0, sm_120, device_count 2 — OK
```

| Framing 维度 | 旧 framing | Wf-22 修正 |
|---|---|---|
| GPU 物理 capacity | "30-min budget 排期 4 周" | **+116.5 GB combined free VRAM** — 物理上无 4 周上限 |
| kanzi 共存 | (未明确) | kanzi 占 GPU 0 ≈ 10 GB; **GPU 1 完全空闲**, R18 retrain 走 GPU 1 (RTX 5090) 可并行 |
| CI ceiling | 30 min/job hard timeout | 仍是真约束 (per `.github/workflows/ci.yml`), 但 **local Blackwell retrain 不受 CI ceiling 限制** |
| Wall-time budget | retrain ≤ 3 h | retrain 单 10k-step × 3 seeds ≈ **6 h Blackwell / 36 h RX 7800 XT** (per `R18_GPU_TRAINING_GUIDE.md` §4.3); **实际可走 2-GPU 并行把 wall-time 砍半** |

**Net result**: Wf-22 把 "Blackwell 4-week 排队" 修正为 **2-week 紧凑排期**, 通过 Blackwell (主) + RTX 5090 (并行 seed / fallback) 双卡负载分担。

---

## 5. R18 lift 实际数值 (vs R17 baseline 0.712)

### 5.1 R17 baseline 来源

| 引用 | 数值 | 来源 |
|---|---|---|
| `train_ru_multitask.py:7` docstring | "lift m1 Ru AUC from **0.712** (1 seed × 5 epochs)" | script header |
| `TODO/completed/53_phase_a_reality_check.md:67` | "Mol-Metal achieves Ru AUC **0.7123** on MetalCytoToxDB (Tier-1 baseline Krasnov **0.81**)" | R17 close-out |
| `TODO/03_plans/R18_PLAN.md:18` Table 1 m1 row | "R17 baseline **0.712**, R18 target **0.81**, BLOCK" | R18 plan |
| Krasnov 2026 reference | **0.81** | Tier-1 published baseline |

### 5.2 PROJECTED R18 lift (5 conditions stacked)

- **m1 Ru AUC**: 0.712 → ≥ 0.86 = **+0.15 absolute / +21% relative**
- **m21 NCI-60 9-cell**: 0.80 → 0.85 = **+0.05 absolute / +6.25% relative**
- **m3 Vina Dock median**: −8.5 → ≤ −10.5 = **−2.0 kcal/mol** (binding affinity)

### 5.3 MEASURED components today (R18 §15.2 Wf-22 ship)

- tmQM pretrain MEASURED lift: **+0.046** Ru AUC over random init (per `53_phase_a_reality_check.md:104`)
- 5-metal multi-task MEASURED lift: TBD (Wf-22 in flight)
- M22 SE(3) SB MEASURED lift: TBD (Phase 2 retrain)
- M24 multi-PH MEASURED lift: TBD (Phase 3 retrain)

---

## 6. 多 GPU 方案 (Blackwell 主力 + RTX 5090 对照/并行 seed)

### 6.1 排期 (Wf-22 修正后, 2-week 紧凑)

| Phase | 任务 | GPU 0 (Blackwell) | GPU 1 (RTX 5090) | Wall-time |
|---|---|---|---|---|
| **P0** | GPU env check + kanzi coexistence | (kanzi_venv sweep) | ✅ R18 D1 ship | 5 min |
| **P1** | Transport Relaxation (M23) retrain | ✅ primary | fallback | 1.5 h Blackwell / 9 h RX 7800 XT |
| **P2** | SE(3) Schrödinger Bridge (M22) retrain | ✅ primary | **并行 seed 1/3/5** | 6 h Blackwell (单卡) / 3 h (双卡并行) |
| **P3** | Multi-PH + Hilbert (M24 + M25) | ✅ primary | **对照 seed 2/4** | 2 h |
| **P4** | Joint M22+M23+M24 retrain (D5 final) | ✅ primary | ✅ parallel seeds | 4 h |
| **P5** | Paper v2 update + Table 2 MEASURED row | (CPU only) | (CPU only) | 1 h |

### 6.2 双卡负载分担规则

- **Blackwell (GPU 0) = primary**: 所有 R18 retrain 主力, BF16 autocast + sm_120 tensor cores
- **RTX 5090 (GPU 1) = secondary**:
  - **mode A (对照)**: 单 seed 对照 (e.g. seed=2 vs Blackwell seed=0); 验证 batch size + lr scaling 不破坏 reproducibility
  - **mode B (并行 seed)**: 同时跑 5 seeds 中 2-3 个; 把 wall-time 砍 50%
  - **mode C (fallback)**: Blackwell OOM / SMU hang (R16 risk) 时无缝接管

### 6.3 torch CUDA_VISIBLE_DEVICES 配置

```bash
# D1 env check
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# Blackwell primary (env.sh)
export CUDA_VISIBLE_DEVICES=0   # RTX PRO 6000

# RTX 5090 secondary (env_5090.sh, 新)
export CUDA_VISIBLE_DEVICES=1   # RTX 5090

# Dual-GPU joint (env_dual.sh, 新)
export CUDA_VISIBLE_DEVICES=0,1 # Blackwell + RTX 5090 (DataParallel 或 distributed)
```

---

## 7. Paper v2 update (MEASURED 替换 PROJECTED)

**Current state** (Wf-20 ship 2026-09-26, `paper/main.tex` v1.7): §4 R18 row **全部 PROJECTED**, §6 honest limitations 完整标注 R17 baseline failures。

**v2 update plan** (Phase 4 week 8, after Wf-22 retrain completes):

| Section | Current (v1.7) | v2 update |
|---|---|---|
| `paper/sections/03_method.tex` §3.4 SE(3) SB | "PROJECTED to lift m3 Vina Dock −8.5 → −10.5" | 替换为 "MEASURED −8.5 → −X.XX (Wf-22 D2 retrain)" |
| `paper/sections/03_method.tex` §3.5 Transport Relax | "PROJECTED KL monotone non-increasing" | 替换为 "MEASURED KL trace + anneal schedule" |
| `paper/sections/03_method.tex` §3.6 Multi-PH | "PROJECTED m7 0.137 → 0.75" | 替换为 "MEASURED m7 = X.XX" |
| `paper/sections/04_results.tex` Table 2 R18 row | "PROJECTED m1 ≥ 0.86, m3 ≤ −10.5" | 替换为 "MEASURED m1 = 0.XXX, m3 = −X.XX" (95% CI over 5 seeds × 5 metals × 4 conditions) |
| `paper/sections/04_results.tex` §4.1 R17 honest framing | "Ru AUC 0.7123 < Krasnov 0.81" | 保留 + 加 "Wf-22 retrain lifts to X.XXX" |
| `paper/refs.bib` | 162 keys | + 2 (Burch-Bardsley 2022 score-repellent MC, Kim-Jung 2026 multi-PH if not already) |

**No-claim discipline**: 保持 R18_PLAN §honest + MERGE_GATE §6; 即便 Wf-22 retrain m1 ≥ 0.86, 不 claim "beat Krasnov 0.81 by +0.05" (该 framing 已经在 v1.7 §6 中明确撤回); 仅作 "R18 SE(3) SB + multi-PH + tmQM pretrain 实证 lift" 报告。

---

## 8. 不修改承诺 (no `train_ru_multitask.py` modify)

**承诺**: Wf-22 retrain **不修改** `molmetal/scripts/train_ru_multitask.py` 现有代码; 仅 launch as-is (`python molmetal/scripts/train_ru_multitask.py --seeds 0,1,2,3,4 --epochs 100` on Blackwell GPU 0)。

**理由**:
1. R17 baseline (0.7123, 1 seed × 5 epoch) 是 `train_ru_multitask.py` **as-shipped** 跑出的; 改代码会破坏 bit-identical 对照
2. R18 lift 通过 **adapter 加法** (`transport_relaxation_integration.py` + `se3_sb_integration.py` + `multi_ph_integration.py`) 实现, 不动 `train_ru_multitask.py` 主流程
3. Wf-21 lambda 端 5 真活已经 ship 走 "0 R17-modify" 原则, Wf-22 保持一致

**Adapter 接入方式** (additive, no `train_ru_multitask.py` change):
- `train_ru_multitask.py` 跑 baseline (R17 复现 + Wf-22 5-seed × 4-conditions)
- R18 adapters 单独跑 (P1-P4 retrain), 输出 .ckpt 给下游
- Paper v2 引用 **两个独立 run 的 MEASURED 数字**, 不混合

---

## 9. Cross-reference — R19+ roadmap priority 1 已 ship

**`R18_PLAN.md` §15.5 R19+ roadmap priority order** (原文 line 493-494):
> 1. **Block GPU training** —— M22-M30 10 个机制真正上 Blackwell GPU retrain (Blackwell 30-min CI budget 排期 4 周)

**Wf-22 ship status**: **PARTIAL** —
- ✅ D1 (GPU env check) SHIP today
- ✅ Wf-22 (本报告) SHIP today, 推翻 4-week 排队 framing → 2-week 紧凑排期
- ⏳ D2/D3/D4 (M22/M23/M24 retrain) 排 Wf-23 2-week 队列
- ⏳ D5 (5-seed × 5-metal × 4 conditions benchmark) 排 Wf-23 Phase 4
- ⏳ D6 (paper v2 update) 排 Wf-23 Phase 4 week 8

**R18_PLAN §10.4** (out-of-scope 推 R19+) — Block GPU training + Phase A re-attempt + Wet-lab + paper v2 四 priority 顺序不变; Wf-22 把 priority 1 的物理约束 (4-week 排队) 推翻, **保留** 战略 priority 顺序 (Block GPU > Phase A > Wet-lab > paper v2)。

**Cross-refs**:
- `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` — Wf-22 D1 live JSON
- `molmetal/reports/wf_blackwell_guide/R18_GPU_TRAINING_GUIDE.md` — D6 reproduce guide
- `molmetal/scripts/check_blackwell_env.py` — D1 probe script (read-only, safe to re-run)
- `molmetal/scripts/train_ru_multitask.py` — D5 baseline (no modify per §8)
- `TODO/03_plans/R18_PLAN.md` §10.4 / §15.5 — R19+ priority 1 framing
- `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` §2 Stage 1.4 — D5 recipe source
- `TODO/completed/53_phase_a_reality_check.md` — tmQM pretrain MEASURED +0.046 lift source
- `TODO/completed/74_lambda_remaining.md` — Wf-21 "0 R17-modify" precedent
- `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` — Blackwell shim + 20-atom vocab fixes (Wf-22 retrain 依赖)

---

**End Wf-22 synthesis. Phase 1 Transport Relaxation retrain launch 排 next.**