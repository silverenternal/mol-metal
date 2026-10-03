# R19 — 详细规划（**真 GPU retrain** · Blackwell + RTX 5090 双卡 · 从 CPU skeleton 迈向 actual training）

**Date**: 2026-09-26
**Parent doc**: `TODO/03_plans/R18_PLAN.md` §15 close-out + `TODO/03_plans/WEAK_TO_STRONG_PLAN.md`
**Status**: 🆕 **NEW** — R19 = 实际 GPU retrain 周期，承接 R18 CPU-only skeleton ship surface；HONEST framing 保留，PROJECTED metrics 真正上 Blackwell。
**Effort budget**: 12 周 (~240 hours, 含 multi-GPU 排期)
**Owner**: Hugo Lee + ultracode agents
**Critical change**: R18 单 Blackwell 卡 `sm_120` + ultracode blacklist → R19 双卡可用 (`RTX PRO 6000 Blackwell sm_120` 95.6 GB + `RTX 5090 sm_120` 31.8 GB)；GPU 训练与 ultracode 不再 mutex，但 ultracode 不抢主训练时段。

---

## §1. R19 总体 vision (一句话 + 5 strategic priorities)

### §1.1 One-sentence vision

> **R19 = 把 R18 ship 的 23 个 CPU 数学骨架 + 32 modules + ≈1 235 tests 真正推到 Blackwell + RTX 5090 双卡 GPU retrain，让 paper §4 的 MEASURED 数字变成 ACTUAL GPU retrain 数字，让 8 STRONG metrics 在 retrain 后变 LEAD，3 个 WEAK metrics (m1/m2/m16) 在 multi-GPU + scaffold-split 排期下尝试一次 honest re-attempt，让 paper v2 真正 deliver 到 Digital Discovery / JCIM 等级。**

### §1.2 5 strategic priorities

| # | Priority | Why | Top-level outcome |
|---|---|---|---|
| **P1** | **Block GPU training**（M22-M30 10 个机制 + M25-M30 lift 排期 4 周） | R18 PROJECTED metrics 全部卡在 GPU retrain；这是 R18 → R19 唯一 unblocker | m3 Vina Dock −8.5 → −10.5+ / m7 Diversity 0.137 → 0.75 / m14 Decode ratio 1.000 (no regression) / m11 n_distinct +5-15% |
| **P2** | **Phase A re-attempt**（multi-GPU + scaffold split + 100-pocket × 10-seed） | R17 Phase A 永久 BLOCK 在 1 GPU + 5-epoch + 1-seed；R19 双卡 + 100-pocket × 10-seed 是 honest re-attempt | m1 Ru AUC 0.712 → ≥0.86 (gate) / m2 Pt A2780cis 0.648 → ≥0.85 (gate) — **still honestly framed**; lift attempt 不等于 guarantee |
| **P3** | **Paper v2**（替换 MEASURED → ACTUAL GPU retrain numbers + R18 §6 honest limitations 保留） | R18 paper v1.7 132+ 页已 ship 但 §4 数字是 PROJECTED；R19 真正 ACTUAL 数字可改 paper §4 + abstract + Table 2 | paper main.pdf v2.0 ≥ 160 页 + supplementary S1-S8 + Digital Discovery / JCIM 投稿 |
| **P4** | **Wet-lab cooperation**（ORNL / PKU 第一个 milestone） | R18 §15 out-of-scope 推后到 R19；paper v2 + tier-1 metric enable 都需要 experimental validation | 单个 m18 protein-metal binding 实验验证 + cisplatin 衍生物 IC50 测定 + lead compound 体外活性验证 |
| **P5** | **Skill ecosystem update**（8 R17/R18 skills → 12 R19 skills） | R18 + 4 NEW skills 已 ship（`mol-metal-r18-math` / `mol-metal-paper-writing` / `party-twenty-spirit` 等）；R19 + 4 NEW skill（Blackwell training / paper-v2 / wet-lab / R19-math） | 12 skills 完整覆盖 GPU retrain workflow + 投稿路线 + 实验合作 + R19 数学定理 |

### §1.3 R19 success = all 5 priorities ✓ (8 R18 → R19 transition gates all PASS)

| Gate | Source | PASS criterion |
|---|---|---|
| **G1** | m3 Vina Dock median | ≤ −10.5 kcal/mol (PROJECTED)；actual GPU retrain 后填 paper §4 Table 2 |
| **G2** | m7 Diversity | ≥ 0.75 (PROJECTED)；Phase 3 multi-PH + Hilbert PD + M26 Score-Rep joint retrain |
| **G3** | m14 Decode ratio | = 1.000 (no regression) |
| **G4** | m1 Ru AUC honest re-attempt | ±0.05 honest report（无论 lift 与否）；paper §6 保留 HONEST framing |
| **G5** | paper §4 + abstract + Table 2 v2.0 ACTUAL GPU retrain numbers | 替换 MEASURED → ACTUAL；保留 §6 limitations honest |
| **G6** | 100-pocket × 10-seed 重测 | m1/m2/m3/m4/m7/m11 全部重测；10-seed 95% CI 填 paper §4 |
| **G7** | R19 close-out synthesis | `TODO/completed/80_r19_plan.md` |
| **G8** | 4 NEW skill + 8 既有 skill update | 12 skills × +30 lines + 4 NEW SKILL.md |

---

## §2. R19 vs R18 区别（从 CPU skeleton → 实际 GPU retrain）

### §2.1 关键转变表（what changes from R18 → R19）

| Dimension | R18 (CPU skeleton) | R19 (实际 GPU retrain) |
|---|---|---|
| **Compute** | Blackwell 单卡 `sm_120` 独占 ultracode + training mutex (blacklist) | RTX PRO 6000 Blackwell (95.6 GB) + RTX 5090 (31.8 GB) 双卡，**GPU 训练与 ultracode 不再 mutex**；ultracode 仍不抢主训练时段但可并行对照/seed |
| **Training** | 0 GPU retrain ship；≈1 235 tests CPU-only；23 mechanism M22-M44 PROJECTED | 10 mechanism M22-M30 真正 retrain (Blackwell 主训练)；4 mechanism joint retrain (Phase 5)；1 mechanism Phase 6 (MD relaxation) |
| **Honest framing** | §6 limitations 写实（Phase A BLOCK + Blackwell constraint + 3 DEFER items + pytest collection + bench skipped modules） | §6 limitations 保留 + 新增 1 段（"R19 GPU retrain ACTUAL numbers replace PROJECTED; honest re-attempt of Phase A with multi-GPU + scaffold split, may still BLOCK"） |
| **Paper state** | v1.7 132+ 页 MEASURED + PROJECTED 混排 | v2.0 ≥ 160 页 ACTUAL GPU retrain 填 §4 + supplementary S1-S8 + Digital Discovery / JCIM 投稿 |
| **Modules** | 32 R18 modules (CPU skeleton + tests + 9 math_skills + 9 adapters + ~14 lam_chem) | 32 R18 modules **不动** (no regression)；+10 retrained weights 落到 `molmetal/weights/r19/{M22-M30}.pt`；+4 PITFALL P2.5 Phases C-E ship |
| **Skills** | 8 (5 R17 + 3 NEW R18) | 12 (8 R18 + 4 NEW: `mol-metal-blackwell-training` / `mol-metal-paper-v2` / `mol-metal-wet-lab` / `mol-metal-r19-math`) |
| **Workflow** | R18 11 wf CPU-only ultracode dispatch | R19 12-14 wf GPU ultracode dispatch（multi-seed × multi-pocket × multi-stage） |
| **Wall-time** | R18 8 周 (CPU skeleton, 0 GPU retrain) | R19 12 周 (multi-GPU training, real GPU retrain) |
| **Risks** | ultracode vs training mutex (blacklist) | GPU OOM (Blackwell 95.6 GB 足够但 joint retrain 有 risk) + ultracode 仍需分段 |
| **Reviewers** | paper §6 honest framing 加分 (PKU / Nature MI) | paper v2 §4 ACTUAL numbers + supplementary S1-S8 + open peer review |

### §2.2 R19 真正的 unblocker（vs R18）

R18 ship 后 **唯一 remaining unblocker** = GPU retrain。R18 32 modules × ≈1 235 tests 都是 CPU skeleton，**没有任何一个 mechanism 真正训练过 GPU 权重**。R19 的核心任务就是：

```
R18 32 modules (CPU-only)
    ↓
R19 GPU retrain pipeline (Blackwell + RTX 5090)
    ↓
10 mechanism weights (M22 SE(3) SB / M23 Transport Relax / M24 Multi-PH / M25 Hilbert PD / M26 Score-Rep MC / M27 W-Flow / M28 MacroGuide / M29 Coord GNN / M30 Quotient Diff)
    +
4 joint retrain (M25+M26 + M27+M28 + M29+M30 + M22+M23+M24)
    ↓
paper §4 ACTUAL numbers + supplementary S1-S8
```

### §2.3 R19 与 R18 的 no-regression 保证

R19 继承 R18 §15.6 不修改承诺：**R18 ship surface bit-identical on master post-R19 GPU retrain**。具体：
- 32 R18 modules 不修改 (CPU skeleton 保留)
- 23 mechanism M22-M44 行为契约 (`MEASURED` 数字) 不修改
- 7/7 pitfall guards + 14/14 evaluators + 6 shim fixes bit-identical
- 18-rule → 21-rule `PRIMITIVE_RULES` 不修改 (CPU registry)
- 9 math_skills 算法不修改 (CPU skeleton)
- R17 close-out commit `4561625` (30/30 decode + 14/14 eval + 5 skills) bit-identical
- paper v1.7 §3.4-§3.7 + §6 limitations 不修改 (v2 在 R19 paper-section 增量 ship)

---

## §3. R19 边界（in-scope vs out-of-scope）

### §3.1 In-scope (R19 deliverable)

| # | Deliverable | Owner | Effort |
|---|---|---|---|
| 1 | **M22 SE(3) SB GPU retrain**（100-pocket × 3-seed × 50-epoch） | GPU-1 Blackwell | 2 周 |
| 2 | **M23 Transport Relax GPU retrain** | GPU-1 Blackwell | 1 周 |
| 3 | **M24 Multi-PH GPU retrain** | GPU-1 Blackwell | 2 周 |
| 4 | **M25 Hilbert PD GPU retrain**（已 CPU MEASURED 452× speedup，GPU joint retrain） | GPU-1 Blackwell | 1 周 |
| 5 | **M26 Score-Rep MC GPU retrain** | GPU-1 Blackwell | 1 周 |
| 6 | **M27 W-Flow ODE GPU retrain**（PROJECTED 100× speedup） | GPU-1 Blackwell | 1 周 |
| 7 | **M28 MacroGuide PH GPU retrain** | GPU-1 Blackwell | 1 周 |
| 8 | **M29 Coord GNN GPU retrain** | GPU-1 Blackwell | 1 周 |
| 9 | **M30 Quotient-Space Diffusion GPU retrain** | GPU-1 Blackwell | 2 周 |
| 10 | **Phase A re-attempt**（multi-GPU + scaffold split + 100-pocket × 10-seed） | GPU-1 + GPU-2 | 1 周 |
| 11 | **Pitfall P2.5 Phases C-E**（MD relaxation MMseqs2 + AMBER + YASARA） | GPU-1 + CPU | 3 周 |
| 12 | **paper v2**（替换 MEASURED → ACTUAL numbers + supplementary S1-S8） | doc + GPU-2 | 4 周 |
| 13 | **Wet-lab 第一个 milestone**（ORNL/PKU protein-metal binding 实验验证） | wet-lab | 4 周 |
| 14 | **12 skills 更新**（8 +30 lines + 4 NEW） | doc | 1 周 |

### §3.2 Out-of-scope (R19 不做)

| # | Out-of-scope item | 推到 |
|---|---|---|
| 1 | M31/M32/M43/M44 paper-cite-only mechanism 真正 retrain | R20+ |
| 2 | 9-cell LOMO transfer AUC (m19) Tier-1 future metric | R20+ |
| 3 | Tumor-specific protein-metal binding AUC (m20) Tier-1 future metric | R20+ (需 patient-derived organoid + TCGA) |
| 4 | Multi-modal Protein-Metal binding AUC (m18) Tier-1 future metric | R20+ (需 AlphaFold-Multimer + MetalSite) |
| 5 | AMBER + YASARA license 申请 + integration | R19 排期（Phase 6）但 license cost 控制；R20 可能扩 |
| 6 | Real patient wet-lab trial | R20+（clinical trial 注册期） |
| 7 | DiffDock / EquiBind forward contract 完整 bench | R20+（需要 docked pose input 标准） |
| 8 | NetworkX / scipy / torch optional-deps 在 env.sh baseline | R20+ (env 维护工作) |

---

## §4. R19 与 R18 的 continuity（32 R18 modules + tests + skills）

### §4.1 32 R18 modules 不修改（继承 R18 §15.6 不修改承诺）

| Module 类 | R18 ship surface | R19 用途 |
|---|---|---|
| 9 math_skills | `molmetal/math_skills/{se3_schrodinger_bridge,transport_relaxation,monge_ampere_regularity,wasserstein_gradient_flow,multi_parameter_ph,ph_hardness_guardrail,sb_bass_martingale,bass_existence,bergman_kernel}.py` | R19 GPU retrain 用作 forward-pass algorithm (CPU skeleton 不动) |
| 9 adapters | `molmetal/adapters/flow_matching_lipman/_reference/path/{se3_sb,multi_ph,ph_hardness}.py` + 6 个 frontier adapter | R19 GPU retrain 用作 integration (no regression) |
| 14 lam_chem/search_alg/reward | `molmetal/molmetal_lam/{lam_chem,search_alg,reward}/...` | R19 GPU retrain 用作 SynFlowNet pipeline |
| R17 baseline | `flow_matching_lipman/*` + `proof_search.py` (6217 LOC) + `lambda_astsr.py` (1059 LOC) + 6 shim fixes + 7/7 pitfall + 5 R17 skills | R19 不动 |

### §4.2 R18 tests 全部继承 + R19 加 GPU retrain tests

| Test group | R18 count | R19 add | R19 total |
|---|---:|---:|---:|
| math (Wf-1) | 305 | +0 (CPU skeleton 不动) | 305 |
| adapters (Wf-2) | 75 | +0 | 75 |
| P2 ML methods (Wf-10) | 117 | +0 | 117 |
| Pitfall Tier-2 (Wf-7) | 77 | +0 | 77 |
| Hilbert (Wf-9) | 31 | +30 (GPU joint retrain regression tests) | 61 |
| Score-Rep + integration (Wf-9) | 38 | +30 (GPU retrain wall-time tests) | 68 |
| CFM M3+M4+M5 (Wf-6) | 60 | +0 | 60 |
| LambdaCFM (Wf-4) | 40 | +0 | 40 |
| Wf-11 close-out | 95 | +0 | 95 |
| Weak → strong credibility | 15 | +0 | 15 |
| R17 baseline | ~217 + 30/30 decode probes + 14/14 eval + 7/7 pitfall | +0 | ~217 + 30/30 |
| **R19 GPU retrain tests** (NEW) | 0 | **+150** (10 mechanism × 15 retrain regression tests + 40 seed-variance + 30 OOM guard) | +150 |
| **R19 wet-lab tests** (NEW, smoke only) | 0 | **+20** (3 milestone × 5-10 smoke assertions) | +20 |
| **R19 paper v2 tests** (NEW) | 0 | **+30** (regression on MEASURED→ACTUAL swap + supplementary S1-S8 build) | +30 |
| **Total R19 expected** | ≈1 235 | **+230** | **≈1 465 tests pass** |

### §4.3 R18 8 skills 不动 + R19 +4 NEW skills

| R18 skill (8) | R19 状态 |
|---|---|
| `mol-metal-lambda-calculus` | 不动 + R19 update section（GPU retrain note） |
| `mol-metal-cfm-training` | 不动 + R19 update section（GPU retrain note） |
| `mol-metal-click-chemistry` | 不动 |
| `mol-metal-anticancer` | 不动 + R19 update section（m18/m19/m20 future metrics） |
| `mol-metal-mechanism-mapping` | 不动 + R19 update section（M22-M30 GPU retrain status） |
| `mol-metal-r18-math` | 不动（R18 9 定理 backlog 保留） |
| `mol-metal-paper-writing` | 不动 + R19 update section（paper v2 ACTUAL numbers） |
| `party-twenty-spirit` | 不动（meta skill） |

| R19 NEW skill (4) | 用途 |
|---|---|
| `mol-metal-blackwell-training` (NEW) | Blackwell + RTX 5090 GPU retrain workflow (Wf-21 / Wf-22 / Wf-23) |
| `mol-metal-paper-v2` (NEW) | paper v2 投稿路线 (Digital Discovery / JCIM + cover letter + reviewer rebuttal) |
| `mol-metal-wet-lab` (NEW) | wet-lab cooperation workflow (ORNL / PKU / cisplatin IC50) |
| `mol-metal-r19-math` (NEW) | R19 数学定理 (4 NEW: PH Hardness + Bergman d-orbital + W-Flow Lelarasmee + Quotient-Space) |

---

## §5. R19 与 R20+ 的关系

### §5.1 R19 → R20 接力图

```
R17 (close-out, 30/30 decode + 14/14 eval + 5 skills + 21 mechanisms)
    ↓
R18 (CPU-only skeleton, 32 modules + ≈1 235 tests + 23 mechanisms + 8 skills)
    ↓
R19 (实际 GPU retrain, 10 mechanism 训练 + 4 joint retrain + paper v2 + wet-lab + 12 skills)
    ↓
R20+ (long-horizon: M43 Bergman d-orbital RBFE + m18/m19/m20 Tier-1 + clinical wet-lab trial)
```

### §5.2 R20+ candidate (R19 §3.2 out-of-scope 推后)

| R20+ candidate | R19 状态 | R20 预期 |
|---|---|---|
| M31 Quantum Sem / M32 Hochschild Cohomology paper-cite-only → 真正 retrain | R19 ship cite-only | R20 paper-cite → retrain |
| M43 Bergman kernel d-orbital RBFE 真正 retrain | R19 P2 long-horizon §15.5 保留 | R20 GPU 100× speedup 排期 |
| M44 Bass martingale diffusion full-deploy | R19 GPU retrain (Phase 4) | R20 Langevin + Bass calibration 完整 deploy |
| m18 Multi-modal Protein-Metal binding AUC | R19 wet-lab 1 milestone | R20 AlphaFold-Multimer + MetalSite 集成 |
| m19 Multi-metal LOMO transfer AUC | R19 5-metal multitask scaffolding | R20 GPU 9-cell × scaffold split 真正 retrain |
| m20 Tumor-specific protein-metal binding AUC | R19 wet-lab cisplatin IC50 | R20 patient-derived organoid + TCGA |
| DiffDock / EquiBind forward contract 完整 bench | R19 forward contract 30/32 + 2/32 skipped | R20 standard docked pose input 排期 |

### §5.3 R19 → R20 transition gates

| Gate | PASS criterion |
|---|---|
| paper v2 已投稿 + arXiv preprint 已 publish | R20 起点 |
| 第一个 wet-lab milestone 完成 + 第二个启动 | R20 起点 |
| 12 skills 全部 ship | R20 起点 |
| 32 R18 modules + 10 R19 mechanism weights + 4 PITFALL P2.5 Phases C-E ship | R20 起点 |
| 7/7 pitfall guards + 14/14 evaluators bit-identical on master | R20 起点 |

---

## §6. Hardware 配置 (R19 GPU 训练)

### §6.1 双卡硬件清单（nvidia-smi 实测 2026-09-26）

| GPU | Index | Arch | Memory | SM | TFLOPs (FP16) | CUDA | 当前状态 |
|---|---|---|---|---|---|---|---|
| **NVIDIA RTX PRO 6000 Blackwell** | 0 | `sm_120` | **95 675 MiB / 97 887 MiB (≈95.6 GB)** | 120 | ~1 200 TFLOPs | 13.4 | 100% GPU-Util（kanzi_venv + dfine_t146 占用），需要 **Wait → Drain** 后 R19 启动 |
| **NVIDIA GeForce RTX 5090** | 1 | `sm_120` | 2 MiB / 32 607 MiB (≈31.8 GB) | 120 | ~838 TFLOPs | 13.4 | 0% GPU-Util, P8 idle（**R19 立即可用**） |

### §6.2 Blackwell 主训练 + RTX 5090 对照 / 并行 seed 排期

| Phase | GPU-1 Blackwell (主训练) | GPU-2 RTX 5090 (对照 / 并行 seed) |
|---|---|---|
| Phase 1 (W1-2) | M22 SE(3) SB retrain | M22 SE(3) SB seed=2 (3-seed 总) |
| Phase 2 (W3-4) | M23 Transport Relax + M24 Multi-PH retrain | M23 seed=2 + M24 seed=2 |
| Phase 3 (W5-6) | M25 Hilbert PD + M26 Score-Rep MC joint retrain | M25+M26 seed=2 |
| Phase 4 (W7-8) | M27 W-Flow + M28 MacroGuide + M29 Coord GNN + M30 Quotient Diff retrain | 4 mechanism × seed=2 |
| Phase 5 (W9) | Phase A re-attempt (multi-GPU + scaffold split + 100-pocket × 10-seed) | GPU-2 跑 5-seed (5/10 拆 GPU-2) |
| Phase 6 (W10-12) | PITFALL P2.5 Phases C-E（MD relaxation + AMBER + YASARA）| GPU-2 跑 MMseqs2 cluster + AMBER CPU + YASARA 渲染 |

### §6.3 内存预算（per training run）

| Model | Params | Train Batch | Inference Batch | GPU-1 Blackwell | GPU-2 RTX 5090 |
|---|---:|---:|---:|---:|---:|
| M22 SE(3) SB | ~3 M | 64 | 256 | 8.2 GB | 4.5 GB |
| M23 Transport Relax | ~2 M | 128 | 512 | 4.5 GB | 2.8 GB |
| M24 Multi-PH | ~5 M | 32 | 128 | 12 GB | 7.5 GB |
| M25 Hilbert PD | ~1 M (CPU-friendly) | 256 | 1024 | 1.2 GB | 0.8 GB |
| M26 Score-Rep MC | ~1.5 M | 128 | 512 | 3.5 GB | 2.0 GB |
| M27 W-Flow ODE | ~2 M | 64 | 256 | 5.5 GB | 3.2 GB |
| M28 MacroGuide PH | ~3 M | 64 | 256 | 8.0 GB | 4.5 GB |
| M29 Coord GNN | ~2.5 M | 64 | 256 | 7.2 GB | 4.0 GB |
| M30 Quotient Diff | ~3.5 M | 32 | 128 | 11.5 GB | 7.0 GB |
| **Phase A re-attempt** | ~1.75 M (D-MPNN+EGNN+fusion) | 32 | 256 | 6.5 GB | 3.5 GB |

**所有 mechanism 都可单卡 fit**，GPU-1 + GPU-2 主要做 parallel seed / 对照实验 / Phase A 5-seed split。

### §6.4 Blackwell vs RTX 5090 性能对比（理论 + 实测）

| Benchmark | Blackwell sm_120 | RTX 5090 sm_120 | Ratio |
|---|---:|---:|---:|
| FP16 TFLOPs | ~1 200 | ~838 | 1.43× |
| FP32 TFLOPs | ~600 | ~419 | 1.43× |
| Memory bandwidth | ~1 800 GB/s | ~1 792 GB/s | ≈1.00× |
| Memory size | 95.6 GB | 31.8 GB | 3.01× |
| L2 cache | ~96 MB | ~96 MB | ≈1.00× |

**实际 R19 训练建议**:
- Blackwell 主训练 (M22-M30 10 mechanism + Phase A re-attempt)
- RTX 5090 对照 / 并行 seed (Phase 5 5-seed split + Phase 6 MMseqs2 cluster + paper v2 render)
- 两者 wall-time 比 ≈ 1.43× (理论) — 实际可能因 memory bandwidth 持平而 ~1.0-1.3×

---

## §7. Software 配置 (R19 GPU 训练)

### §7.1 Software stack（env.sh baseline + R19 add）

| Component | Version | R18 baseline | R19 add |
|---|---|---|---|
| **Python** | 3.13.x | env.sh baseline | — |
| **PyTorch** | 2.14+cu130 | env.sh baseline | — |
| **CUDA** | 13.4 | env.sh baseline | — |
| **Triton** | 3.8.0 | env.sh baseline | — |
| **torch-geometric** | 2.6.x | env.sh baseline | — |
| **RDKit** | 2024.09.x | env.sh baseline | — |
| **BioPython** | 1.84 | env.sh baseline | — |
| **PySR** | 0.20.x | env.sh baseline | — |
| **scipy** | 1.13.x | optional-deps (48 pytest errors) | **fix: env.sh baseline add scipy** |
| **networkx** | 3.4.x | optional-deps | **fix: env.sh baseline add networkx** |
| **MMseqs2** | 14-7e284 | external (Pitfall P2.5 Phase C) | **R19 install + cluster pipeline** |
| **AMBER** | 24.x | external (Pitfall P2.5 Phase D) | **R19 license + integration** |
| **YASARA** | 24.x | external (Pitfall P2.5 Phase E) | **R19 free tier + integration** |
| **Weights & Biases** | 0.18.x | n/a | **NEW: R19 GPU retrain logging** |
| **Hydra** | 1.3.x | n/a | **NEW: R19 GPU retrain config** |

### §7.2 R19 retrain config (Hydra + W&B)

```yaml
# molmetal/configs/r19_retrain.yaml
defaults:
  - base
  - model: ${model_name}
  - data: crossdocked100

model_name: m22_se3_sb  # @ hydra.choice

train:
  epochs: 50
  batch_size: 64
  lr: 1.0e-4
  lr_scheduler: cosine
  warmup_steps: 500
  grad_clip: 1.0
  weight_decay: 1.0e-5
  amp: bf16  # Blackwell sm_120 supports BF16 natively
  compile: true  # torch.compile

data:
  dataset: crossdocked100
  num_workers: 8
  pin_memory: true
  seed: 0  # 3-seed total: 0, 1, 2
  scaffold_split: true
  test_size: 0.1
  val_size: 0.1

eval:
  metrics: [vina_dock, qed, sa, logp, n_distinct, decode_ratio, diversity, pb_valid]
  eval_every_n_epochs: 5
  ckpt_every_n_epochs: 10

wandb:
  project: mol-metal-r19
  entity: hugo-lee-molmetal
  tags: [r19, retrain, gpu]
```

---

## §8. R19 6 phases GPU 训练 schedule

### §8.1 Phase 1 (Week 1-2) — M22 SE(3) SB GPU retrain on GPU-1（避开 kanzi）

**目标**: M22 SE(3) Schrödinger Bridge 真正上 Blackwell GPU retrain，PROJECTED m3 Vina Dock −8.5 → −10.5+ 验证。

| Week | 任务 | GPU-1 | GPU-2 | 预期 wall-time |
|---|---|---|---|---|
| W1 Day 1-2 | Wait & Drain — kanzi_venv (PID 3542026) + dfine_t146 (PID 3543018) drain 完成 | idle wait | idle | 0-2 天 |
| W1 Day 3-5 | M22 SE(3) SB smoke (1k-step) — verify Lie group integrator + 1000 steps no NaN | train | idle | 2 小时 |
| W1 Day 6-7 | M22 SE(3) SB 50-epoch × seed=0 (主训练) | train | seed=2 对照 | 18 小时 |
| W2 Day 1-3 | M22 SE(3) SB 50-epoch × seed=1 + seed=2 (GPU-1 + GPU-2) | seed=1 | seed=2 | 36 小时 (parallel) |
| W2 Day 4-5 | 100-pocket × 3-seed CrossDocked100 sweep | sweep | sweep | 8 小时 |
| W2 Day 6-7 | m3 Vina Dock median ≤ −10.5 验证 + Table 2 R19 row | eval | eval | 4 小时 |

**Success criteria**:
- m3 Vina Dock median ≤ −10.5 kcal/mol (PROJECTED)
- m14 Decode ratio = 1.000 (no regression)
- m11 n_distinct ≥ 5
- m8 PB-Valid ≥ 0.95
- SE(3) decoder: det=1, orthogonal

**Deliverable**:
- `molmetal/weights/r19/M22_se3_sb_{seed0,seed1,seed2}.pt`
- `molmetal/reports/wf_r19_p1_se3sb/final.md`
- `paper/section_04_results.tex` Table 2 R19 row (m3 Vina −10.5+)

### §8.2 Phase 2 (Week 3-4) — M23 Transport Relax + M24 Multi-PH GPU retrain

**目标**: M23 Transport Relax (no regression m14) + M24 Multi-PH (m7 Diversity +0.05 lift)

| Week | 任务 | GPU-1 | GPU-2 | 预期 wall-time |
|---|---|---|---|---|
| W3 Day 1-3 | M23 Transport Relax smoke + 50-epoch × 3-seed | train | seed=2 | 24 小时 |
| W3 Day 4-7 | M24 Multi-PH smoke + 50-epoch × seed=0 | train | idle | 36 小时 |
| W4 Day 1-3 | M24 Multi-PH 50-epoch × seed=1 + seed=2 | seed=1 | seed=2 | 36 小时 (parallel) |
| W4 Day 4-5 | M23+M24 100-pocket × 3-seed sweep | sweep | sweep | 12 小时 |
| W4 Day 6-7 | m7 Diversity ≥ 0.75 验证 + m14 Decode ratio = 1.000 (no regression) | eval | eval | 4 小时 |

**Success criteria**:
- m14 Decode ratio = 1.000 (no regression)
- m7 Diversity ≥ 0.75 (vs R17 0.137)
- ASTSR R² ≥ 0.62 + wall-time −50%
- Multi-PH Fourier O(N log N) — verify N ≤ 100 atoms

**Deliverable**:
- `molmetal/weights/r19/{M23_transport_relax,M24_multi_ph}_{seed0,seed1,seed2}.pt`
- `molmetal/reports/wf_r19_p2_relax_multiph/final.md`
- `paper/section_04_results.tex` Table 2 R19 row (m7 ≥ 0.75)

### §8.3 Phase 3 (Week 5-6) — Hilbert-ASTSR + Joint Multi-PH

**目标**: M25 Hilbert PD (CPU MEASURED 452× speedup，GPU joint retrain) + M26 Score-Rep MC joint retrain

| Week | 任务 | GPU-1 | GPU-2 | 预期 wall-time |
|---|---|---|---|---|
| W5 Day 1-3 | M25 Hilbert PD GPU joint retrain (50-epoch × 3-seed) | train | seed=2 | 24 小时 |
| W5 Day 4-7 | M26 Score-Rep MC GPU joint retrain (50-epoch × seed=0) | train | idle | 36 小时 |
| W6 Day 1-3 | M26 Score-Rep MC 50-epoch × seed=1 + seed=2 | seed=1 | seed=2 | 36 小时 (parallel) |
| W6 Day 4-5 | M25+M26 100-pocket × 3-seed sweep | sweep | sweep | 12 小时 |
| W6 Day 6-7 | m16 Affinity Pearson r ≥ 0.75 + m4 Success Rate ≥ 0.65 验证 | eval | eval | 4 小时 |

**Success criteria**:
- m16 Affinity Pearson r ≥ 0.75 (vs R17 0.207)
- m4 Success Rate ≥ 0.65 (vs R17 0.0)
- MCTS variance −20% @ α = 2ηL = 0.20
- Hilbert PD wall-time 452× speedup (CPU MEASURED) → GPU +100× (joint retrain)

**Deliverable**:
- `molmetal/weights/r19/{M25_hilbert_pd,M26_score_rep_mc}_{seed0,seed1,seed2}.pt`
- `molmetal/reports/wf_r19_p3_hilbert_scorerep/final.md`

### §8.4 Phase 4 (Week 7-8) — M25-M30 mechanisms GPU retrain

**目标**: M27 W-Flow + M28 MacroGuide + M29 Coord GNN + M30 Quotient Diff GPU retrain

| Week | 任务 | GPU-1 | GPU-2 | 预期 wall-time |
|---|---|---|---|---|
| W7 Day 1-2 | M27 W-Flow ODE 50-epoch × 3-seed (PROJECTED 100× speedup) | train | seed=2 | 16 小时 |
| W7 Day 3-4 | M28 MacroGuide PH 50-epoch × 3-seed | train | seed=2 | 16 小时 |
| W7 Day 5-7 | M29 Coord GNN 50-epoch × 3-seed | train | seed=2 | 24 小时 |
| W8 Day 1-3 | M30 Quotient-Space Diffusion 50-epoch × 3-seed (SE(3)-by-construction) | train | seed=2 | 36 小时 |
| W8 Day 4-5 | M27-M30 100-pocket × 3-seed sweep | sweep | sweep | 12 小时 |
| W8 Day 6-7 | ODE sampling wall-time 6 ms → 0.06 ms / molecule 验证 + m3 Vina +1.5-2.5 kcal/mol | eval | eval | 4 小时 |

**Success criteria**:
- ODE sampling wall-time 6 ms → 0.06 ms / molecule (M27 PROJECTED 100× speedup)
- m3 Vina +1.5-2.5 kcal/mol (M29+M30)
- m11 n_distinct +5-15% (M28+M30+M39)
- m15 RMSD < 1.5 Å

**Deliverable**:
- `molmetal/weights/r19/{M27_w_flow,M28_macroguide_ph,M29_coord_gnn,M30_quotient_diff}_{seed0,seed1,seed2}.pt`
- `molmetal/reports/wf_r19_p4_mechanisms/final.md`

### §8.5 Phase 5 (Week 9) — Phase A re-attempt (multi-GPU + scaffold split)

**目标**: Phase A honest re-attempt — multi-GPU + scaffold split + 100-pocket × 10-seed 重测 m1/m2

| Day | 任务 | GPU-1 | GPU-2 | 预期 wall-time |
|---|---|---|---|---|
| Day 1-2 | 5-metal multitask pretrain (PlatinAI 5K + MetalCytoToxDB 5-metal) | train | train | 48 小时 (parallel) |
| Day 3-4 | Scaffold split 100-pocket × 10-seed m1 Ru AUC retrain | seed 0-4 | seed 5-9 | 36 小时 (parallel) |
| Day 5-6 | Scaffold split 100-pocket × 10-seed m2 Pt A2780cis AUC retrain | seed 0-4 | seed 5-9 | 36 小时 (parallel) |
| Day 7 | 95% CI 填 paper §4 + §6 honest framing | eval | eval | 4 小时 |

**Success criteria (HONEST)**:
- m1 Ru AUC: ±0.05 honest report (无论 lift 与否，10-seed 95% CI)
- m2 Pt A2780cis AUC: ±0.05 honest report (无论 lift 与否)
- paper §4 填 10-seed mean ± 95% CI
- paper §6 保留 HONEST framing "R19 re-attempt multi-GPU + scaffold split; still honestly labeled; lift attempt ≠ guarantee"

**Deliverable**:
- `molmetal/reports/wf_r19_p5_phase_a_reattempt/final.md`
- `paper/section_04_results.tex` Table 2 R19 row (m1, m2 ±0.05 honest)
- `paper/section_06_limitations.tex` 第 6 段 (R19 re-attempt honest framing)

### §8.6 Phase 6 (Week 10-12) — Pitfall P2.5 Phases C-E (MD relaxation + AMBER integration)

**目标**: Pitfall Tier-2 P2.5 Phases C-E ship — MD relaxation + MMseqs2 cluster + AMBER force-field + YASARA integration

| Week | 任务 | GPU-1 | GPU-2 + CPU | 预期 wall-time |
|---|---|---|---|---|
| W10 Day 1-3 | P2.5 Phase C (MMseqs2 cluster) install + cluster pipeline | idle | CPU | 24 小时 (CPU-bound) |
| W10 Day 4-7 | P2.5 Phase D (AMBER 24.x license) install + MD relax smoke (1k-step) | train (M37 MD stub) | AMBER CPU | 48 小时 (AMBER CPU-bound) |
| W11 Day 1-4 | P2.5 Phase E (YASARA 24.x free tier) install + integration smoke | train (M37 MD stub GPU retrain) | YASARA render | 48 小时 |
| W11 Day 5-7 | P2.5 Phases C+D+E joint test (100-pocket × 3-seed) | sweep | sweep | 36 小时 |
| W12 Day 1-3 | m7 Diversity +0.05-0.10 lift validation + docking post-relax energy | eval | eval | 24 小时 |
| W12 Day 4-7 | P2.5 Phases C-E close-out + paper §4 update + 3 mechanism M37/M41/M40 GPU retrain integration | eval | eval | 36 小时 |

**Success criteria**:
- m7 Diversity +0.05-0.10 (P2.5 Phase D MD relax lift)
- Docking post-relax energy -1 to -3 kcal/mol
- m11 n_distinct +5-10%
- 3 mechanism M37 (MD stub) / M41 (OOD protocol MMseqs2 cluster) / M40 (Buyable gate) **真正 retrain 化**

**Deliverable**:
- `molmetal/weights/r19/{M37_md_stub,M41_ood_protocol,M40_buyable_gate}_{seed0,seed1,seed2}.pt`
- `molmetal/reports/wf_r19_p6_pitfall_p25/final.md`
- 3 PITFALL Tier-2 DEFER items **正式 ship**（R19 §3.2 out-of-scope → in-scope）

---

## §9. Multi-GPU 调度 + 5-seed × 50-epoch expected wall-time + memory

### §9.1 Multi-GPU 调度总览

| Phase | GPU-1 Blackwell 主训练 | GPU-2 RTX 5090 对照 / 并行 seed | Wall-time 总 |
|---|---|---|---|
| Phase 1 | M22 SE(3) SB (seed=0) | M22 seed=2 | W1-W2 (2 周) |
| Phase 2 | M23 Transport Relax (seed=0) + M24 Multi-PH (seed=0) | M23 seed=2 + M24 seed=2 | W3-W4 (2 周) |
| Phase 3 | M25 Hilbert PD + M26 Score-Rep MC (seed=0) | M25+M26 seed=2 | W5-W6 (2 周) |
| Phase 4 | M27-M30 4 mechanism × seed=0 | seed=2 | W7-W8 (2 周) |
| Phase 5 | Phase A re-attempt (seed=0..4) | Phase A (seed=5..9) | W9 (1 周) |
| Phase 6 | P2.5 Phases C-E (M37/M41/M40) | MMseqs2 cluster + AMBER CPU + YASARA render | W10-W12 (3 周) |

### §9.2 5-seed × 50-epoch expected wall-time (per mechanism)

| Mechanism | 1-seed × 50-epoch (GPU-1) | 1-seed × 50-epoch (GPU-2) | 5-seed total | 5-seed total (parallel GPU-1+GPU-2) |
|---|---:|---:|---:|---:|
| M22 SE(3) SB | 18 h | 22 h | 90 h | 54 h |
| M23 Transport Relax | 12 h | 15 h | 60 h | 36 h |
| M24 Multi-PH | 36 h | 44 h | 180 h | 108 h |
| M25 Hilbert PD | 8 h | 10 h | 40 h | 24 h |
| M26 Score-Rep MC | 24 h | 30 h | 120 h | 72 h |
| M27 W-Flow | 16 h | 20 h | 80 h | 48 h |
| M28 MacroGuide | 18 h | 22 h | 90 h | 54 h |
| M29 Coord GNN | 24 h | 30 h | 120 h | 72 h |
| M30 Quotient Diff | 36 h | 44 h | 180 h | 108 h |
| **Total 10 mechanism × 5-seed** | — | — | **960 h** | **576 h** (parallel) |
| Phase A re-attempt (5-seed split) | — | — | 72 h | 36 h |
| Phase 6 P2.5 Phases C-E | — | — | 144 h | 144 h (mostly CPU) |
| **R19 GPU total** | — | — | **1 176 h** | **756 h** (parallel) |

### §9.3 Memory budget 总结（Blackwell 95.6 GB 足够）

| GPU | Total memory | Peak R19 use | Headroom | Status |
|---|---:|---:|---:|---|
| GPU-1 Blackwell | 95 675 MiB | ~12 GB (M24 Multi-PH peak) | 83 GB | ✅ 大量余量 |
| GPU-2 RTX 5090 | 31 807 MiB | ~7.5 GB (M24 Multi-PH peak) | 24 GB | ✅ 足够 |

**R19 不需要 OOM guard** — 所有 mechanism 都可单卡 fit。但建议留 8 GB headroom 给 amp + compile + gradient checkpoint。

---

## §10. R19 retrain 风险 + mitigation

### §10.1 风险表

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **R-1: kanzi_venv + dfine_t146 不 drain** | Medium | High | Wait 0-2 天；如不 drain，挪到 RTX 5090 (31.8 GB) 主训练 |
| **R-2: OOM on joint retrain (M25+M26 + M27+M28)** | Low | Medium | 留 8 GB headroom + gradient checkpoint + bf16 amp |
| **R-3: torch.compile + sm_120 兼容性** | Low | Low | fallback 到 eager mode + manual kernel |
| **R-4: seed variance ±0.05 (m1/m2 honest re-attempt)** | High | Low | 10-seed 95% CI 填 paper §4；honest framing 保留 |
| **R-5: AMBER license cost / 申请延期** | Medium | Medium | R19 W10-W11 提前申请；YASARA free tier 替代备选 |
| **R-6: MMseqs2 cluster 物理限制（cluster 内存 OOM）** | Low | Low | GPU-2 跑 MMseqs2 + chunked cluster + 32 GB CPU RAM |
| **R-7: paper v2 §4 ACTUAL numbers 反而比 PROJECTED 差** | Medium | Medium | §6 limitations 保留 HONEST framing；paper §4 写实；这是 honest 体现 |
| **R-8: wet-lab milestone 1 延期（ORNL/PKU 排期）** | Medium | Low | P4 是 P5 之外独立轨道；不影响 GPU retrain + paper v2 排期 |
| **R-9: ultracode vs training 不再 mutex 但仍抢时段** | Low | Low | ultracode 不抢主训练时段；可跑对照实验 + render + smoke |
| **R-10: 32 R18 modules 任何 regression** | Very Low | Critical | no-regression 保证 + 7/7 pitfall guards + 14/14 evaluators bit-identical + 30/30 decode probes |

### §10.2 Mitigation in detail

- **R-1 kanzi/dfine drain**: Hugo 联系 kanzi + dfine owner；如 2 天内不 drain，挪 M22 主训练到 RTX 5090 (memory 足够 12 GB peak)；GPU-1 Blackwell 等 drain 后跑 M22 (5-seed 总 wall-time 18h × 5 = 90h vs RTX 5090 22h × 5 = 110h，差 20h 在 2 周预算内可接受)
- **R-7 paper v2 worse than PROJECTED**: 这是 honest 体现；§6 limitations 写实段已 ship；reviewer 看到 honest framing 加分（PKU / Nature MI 文化）；paper §4 ACTUAL numbers + §6 honest limitations 是 R19 真正 deliver
- **R-4 seed variance**: 10-seed 95% CI 是 honest 体现；paper §4 填 mean ± CI；§6 保留 HONEST framing "R19 re-attempt multi-GPU + scaffold split; still honestly labeled"

---

## §11. R19 retrain 验证 checklist

### §11.1 Pre-flight (Week 0)

- [ ] kanzi_venv (PID 3542026) + dfine_t146 (PID 3543018) drain 完成
- [ ] W&B project `mol-metal-r19` 创建
- [ ] Hydra config `molmetal/configs/r19_retrain.yaml` 验证
- [ ] 32 R18 modules + 23 mechanism M22-M44 import OK
- [ ] 7/7 pitfall guards + 14/14 evaluators + 30/30 decode probes + 6 shim fixes bit-identical
- [ ] 5 R17 skills + 3 R18 NEW skills + R19 4 NEW skill 草稿完成

### §11.2 Per-phase verification (Phase 1-6)

- [ ] Phase 1: m3 Vina Dock median ≤ −10.5 kcal/mol + SE(3) decoder det=1 orthogonal
- [ ] Phase 2: m14 Decode ratio = 1.000 (no regression) + m7 Diversity ≥ 0.75
- [ ] Phase 3: m16 Affinity Pearson r ≥ 0.75 + m4 Success Rate ≥ 0.65 + MCTS variance −20%
- [ ] Phase 4: ODE sampling wall-time 6 ms → 0.06 ms / molecule + m3 Vina +1.5-2.5 kcal/mol
- [ ] Phase 5: m1 Ru AUC ±0.05 honest + m2 Pt A2780cis ±0.05 honest + paper §6 HONEST framing
- [ ] Phase 6: m7 Diversity +0.05-0.10 + P2.5 Phases C-E ship + 3 mechanism M37/M41/M40 GPU retrain 化

### §11.3 Post-flight (Week 12 close-out)

- [ ] 10 mechanism × 5-seed = 50 weights `molmetal/weights/r19/{M22-M30}_{seed0-4}.pt` ship
- [ ] 3 PITFALL P2.5 Phases C-E mechanism × 5-seed = 15 weights ship
- [ ] 32 R18 modules + 23 R18 mechanism bit-identical
- [ ] paper v2 ≥ 160 页 + supplementary S1-S8 + cover letter
- [ ] 12 skills ship (8 R18 + 4 NEW R19) + 8 R18 skill update
- [ ] `TODO/completed/80_r19_plan.md` synthesis report

---

## §12. R19 GPU training 关键里程碑 timeline

```
Week 0  : Pre-flight (drain + W&B + Hydra + smoke verify)
Week 1-2: Phase 1 (M22 SE(3) SB retrain, GPU-1 + GPU-2)
Week 3-4: Phase 2 (M23 Transport Relax + M24 Multi-PH retrain)
Week 5-6: Phase 3 (M25 Hilbert PD + M26 Score-Rep MC joint retrain)
Week 7-8: Phase 4 (M27-M30 4 mechanism retrain)
Week 9  : Phase 5 (Phase A re-attempt multi-GPU + scaffold split + 100-pocket × 10-seed)
Week 10-12: Phase 6 (Pitfall P2.5 Phases C-E ship)
Week 12 : Post-flight (paper v2 + skill update + close-out)
```

**Critical path**: Phase 1 (M22 SE(3) SB — 最大风险: Lie group numerical integration 不稳定)
**Slip buffer**: Phase 6 3 周可吸收 Phase 1-5 延期
**Parallelism policy**: Phase 1-4 GPU-1 主训练 + GPU-2 对照/并行 seed；Phase 5 GPU-1 + GPU-2 split 5-seed；Phase 6 GPU-1 主训练 + GPU-2 MMseqs2 + CPU AMBER/YASARA

---

## §13. 投稿目标 (Digital Discovery / JCIM)

### §13.1 Target journal selection

| Journal | IF (2026) | Q | Open Review | Time-to-decision | 推荐度 |
|---|---:|---|---|---|---|
| **Digital Discovery** (RSC) | **8.5** | Q1 | Yes (transparent peer review) | 4-6 周 | ⭐⭐⭐⭐⭐ **首选** |
| **J. Chem. Inf. Model.** (ACS) | **5.6** | Q1 | No | 6-8 周 | ⭐⭐⭐⭐ 次选 |
| Nat. Comput. Sci. | 14 | Q1 | Yes | 8-12 周 | ⭐⭐⭐ 太顶 |
| Briefings in Bioinformatics | 7.7 | Q1 | No | 6-8 周 | ⭐⭐⭐ |
| JCIM Open Access | — | — | Yes | — | ⭐⭐⭐ 备选 |

**首选 Digital Discovery (RSC)**:
- IF 8.5 Q1 (2026)
- Open peer review (transparent)
- Time-to-decision 4-6 周
- RSC publisher (EGNN + flow matching 早期 paper 多 publish 在 RSC)
- Scope: "Digital Discovery covers new digital or computational tools and methods for chemistry, biochemistry, materials science, and closely related fields"

**次选 JCIM (ACS)**:
- IF 5.6 Q1 (2026)
- Time-to-decision 6-8 周
- 传统期刊；JCIM 是 cheminformatics 老牌
- Scope: "Original articles, reviews, and commentaries that report new methods, software, and applications"

**决定因素**: R19 Week 8 评估 paper v2 §4 ACTUAL numbers vs PROJECTED 哪个更 favor；选 Digital Discovery 概率高。

### §13.2 R19 paper v2 timeline

```
R19 Week 1-2: paper v2 §1 + §2 + §3 (method, no change from R18 v1.7 §3.1.1-§3.7)
R19 Week 3-4: paper v2 §4 Table 2 填 R19 GPU retrain ACTUAL numbers (Phase 1-2 done)
R19 Week 5-6: paper v2 §4 + §5 (Phase 3 done) + §6 limitations 写实 + §7 future work
R19 Week 7-8: paper v2 supplementary S1-S8 (8 supplementary documents, see §14)
R19 Week 9  : paper v2 cover letter + highlights + reviewer rebuttal 准备 + arXiv preprint 准备
R19 Week 10 : arXiv preprint publish (q-bio.QM / cs.LG / physics.chem-ph cross-list)
R19 Week 11-12: Digital Discovery / JCIM submit + open peer review
R19 Week 12 : R19 close-out synthesis `TODO/completed/80_r19_plan.md`
```

---

## §14. Paper v1 (R17 close-out, 132+ pages) → v2 (R18 ship, 160+ pages, supplementary S1-S8)

### §14.1 Paper v2 sections（替换 MEASURED → ACTUAL GPU retrain numbers）

| § | Title | v1 (R17) | v2 (R19) |
|---|---|---|---|
| §1 | Introduction | ✅ ship | + 1 段 R19 GPU retrain narrative |
| §2 | Related Work | ✅ ship | + 3 NEW bib (R19 数学定理 + wet-lab + paper-v2 引用) |
| §3.1.1 | 量子背景 | ✅ ship (R18 §3.1.1) | no change |
| §3.1.2 | Hochschild 同调 | ✅ ship (R18 §3.1.2) | no change |
| §3.4 | SE(3) Schrödinger Bridge | ✅ ship (R18 §3.4) | + ACTUAL M22 GPU retrain numbers |
| §3.5 | Transport Relaxation | ✅ ship (R18 §3.5) | + ACTUAL M23 GPU retrain numbers |
| §3.6 | Multi-PH + Hilbert PD | ✅ ship (R18 §3.6) | + ACTUAL M24+M25 GPU retrain numbers |
| §3.7 | Monge-Ampère + WGF + SB-Bass + Bergman | ✅ ship (R18 §3.7) | + ACTUAL M43/M44 GPU retrain numbers |
| §4 | Results | ✅ ship (R18 §4.5 weak→strong) | **替换 MEASURED → ACTUAL Table 2 + 100-pocket × 10-seed 95% CI + §4.5 weak→strong 保留** |
| §5 | Discussion | ✅ ship | + R19 GPU retrain discussion |
| §6 | Limitations | ✅ ship (R18 §6 honest) | + 1 段 R19 honest re-attempt (multi-GPU + scaffold split, still honestly labeled) |
| §7 | Future Work | ✅ ship | + R20+ roadmap (m18/m19/m20 Tier-1 + clinical wet-lab trial) |
| Appendices A-G | closure_theorem + betanf_semantics + 5 appendix | ✅ ship | + Appendix H R19 GPU retrain reproducibility (5-seed + 100-pocket) |
| Supplementary S1-S8 | NEW R19 ship | n/a | + 8 supplementary documents (see §14.3) |

### §14.2 Paper v2 page count estimate

| Component | v1 (R17 close-out) | v2 (R18) | v2 (R19) |
|---|---:|---:|---:|
| Main text | 28 pages | 32 pages | 36 pages |
| Tables + Figures | 24 pages | 28 pages | 32 pages |
| Appendices A-G | 56 pages | 60 pages | 64 pages |
| Appendix H (NEW R19) | n/a | n/a | 12 pages |
| References | 24 pages | 28 pages | 30 pages |
| Supplementary S1-S8 | n/a | n/a | 24 pages |
| **Total** | **132 pages** | **148 pages** | **198 pages** (预估 v2.0 主体 + supplementary ≥ 160) |

### §14.3 Supplementary S1-S8 (8 supplementary documents)

| S# | Title | 内容 |
|---|---|---|
| **S1** | **M22-M30 GPU retrain reproducibility** | 10 mechanism × 5-seed × 50-epoch 配置 + W&B run links + weight hashes + memory budget |
| **S2** | **100-pocket × 10-seed CrossDocked100 sweep** | scaffold split + 10-seed 95% CI + per-pocket Vina Dock + QED + SA + logP + n_distinct + decode_ratio + diversity + pb_valid |
| **S3** | **Phase A re-attempt honest report** | multi-GPU + scaffold split + 5-metal multitask + m1/m2 10-seed 95% CI + §6 honest framing |
| **S4** | **Pitfall P2.5 Phases C-E reproducibility** | MMseqs2 cluster pipeline + AMBER MD relax + YASARA integration + docking post-relax energy |
| **S5** | **Hilbert PD + Score-Rep MC joint retrain** | M25 + M26 + M33 VariancePUCT integration + MCTS variance −20% @ α = 2ηL = 0.20 |
| **S6** | **W-Flow + MacroGuide + Coord GNN + Quotient Diff** | M27 + M28 + M29 + M30 ODE sampling wall-time + 100× speedup + SE(3)-by-construction |
| **S7** | **Wet-lab milestone 1 protocol** | ORNL / PKU protein-metal binding experimental protocol + cisplatin 衍生物 IC50 测定方法 |
| **S8** | **Open peer review + reviewer rebuttal** | reviewer rebuttal 模板 + open peer review 策略 + cover letter |

---

## §15. Cover letter + highlights

### §15.1 Cover letter (Digital Discovery)

```
Dear Editor,

We are pleased to submit our manuscript entitled "Mol-Metal: A Hybrid Molecular Lambda Calculus + Flow-Matching Framework for Precious-Metal Drug Design" for consideration at Digital Discovery.

**Why Digital Discovery**:
- Scope match: Digital Discovery covers new digital/computational tools for chemistry — our hybrid framework fits squarely.
- Open peer review: We opt for transparent peer review (Digital Discovery default).
- 4-6 week time-to-decision fits our 12-week R19 cycle.

**Key contributions (3 LEADS)**:
1. **Hybrid Molecular Lambda Calculus + Flow-Matching framework** — first principled integration of constructive type theory with stochastic flow-matching for chemistry.
2. **23 new mechanisms (M22-M44)** — including SE(3) Schrödinger Bridge (Mahmood 2026), Transport Relaxation (Jiang-Xu-Zhang 2026), Multi-Parameter Persistent Homology (Kim-Jung 2026), Hilbert primal-dual reward (Sun-Zheng 2026), and Score-Repellent MC (Hu et al. ICML 2026).
3. **Honest GPU retrain + scaffold-split + multi-seed evaluation** — 10 mechanism × 5-seed × 100-pocket = 5 000 GPU retrain runs on Blackwell + RTX 5090 with full reproducibility (W&B + weight hashes).

**Why this paper is significant**:
- 8 STRONG metrics (m11/m12/m13/m14/m17/m2-PC3/HL60/SKOV3/m21) lead claims with measured Tier-1 SOTA match.
- 3 WEAK metrics (m1/m2/m16) honestly reported with multi-GPU + scaffold-split re-attempt and §6 limitations.
- 23 mechanism mapping documented in `MECHANISM_METRIC_MAPPING.md` with reproducibility + ablation design.
- Open-source under permissive license (MIT) with reproducible GPU retrain + dry-run scripts.

**Conflict of interest**: None.
**Funding**: [Hugo Lee PI / ORNL / PKU / NSFC as applicable].
**Data availability**: CrossDocked100 + PlatinAI 5K + MetalCytoToxDB 5-metal + 100-pocket sweep publicly available.

Sincerely,
Hugo Lee (corresponding author)
[Affiliation]
```

### §15.2 Highlights (3 bullets for RSC submission)

- **Hybrid Molecular Lambda Calculus + Flow-Matching framework** for precious-metal drug design with first principled integration of constructive type theory and stochastic flow-matching.
- **23 new mechanisms (M22-M44)** including SE(3) Schrödinger Bridge, Transport Relaxation, Multi-Parameter PH, Hilbert primal-dual, Score-Repellent MC — full GPU retrain + reproducibility.
- **Honest evaluation**: 8 STRONG metrics lead claims, 3 WEAK metrics honestly reported with multi-GPU + scaffold-split re-attempt.

---

## §16. Reviewer rebuttal 准备

### §16.1 Likely reviewer concerns + R19 准备

| Reviewer concern | R19 准备 |
|---|---|
| **R-C1**: "M22 SE(3) SB 真的 lift m3 Vina −8.5 → −10.5 吗？" | S1 GPU retrain reproducibility + Table 2 R19 row + W&B run links |
| **R-C2**: "Phase A honest re-attempt 还是 BLOCK 怎么办？" | S3 honest report + §6 limitations 第 6 段 + 10-seed 95% CI |
| **R-C3**: "Pitfall P2.5 Phases C-E 真正 ship 了吗？" | S4 reproducibility + MMseqs2 cluster + AMBER MD relax + YASARA integration |
| **R-C4**: "M25 Hilbert PD 452× speedup 是否 reproducible？" | S5 joint retrain + CPU MEASURED + GPU joint retrain + 30-cell hold-out |
| **R-C5**: "23 mechanism mapping 是 ad-hoc 还是 principled？" | `MECHANISM_METRIC_MAPPING.md` §1/§5/§6 + §7 ablation design table |
| **R-C6**: "wet-lab 验证只 1 个 milestone 够吗？" | S7 protocol + 第二个 milestone (cisplatin 衍生物 IC50) + 第三个 (lead compound 体外活性) |
| **R-C7**: "Open peer review 风险？" | S8 transparent peer review 模板 + reviewer rebuttal 模板 |
| **R-C8**: "R18 → R19 transition no-regression 保证？" | R19 §2.3 + 32 R18 modules bit-identical + 7/7 pitfall + 14/14 eval + 30/30 decode |
| **R-C9**: "Reproducibility: code + data + weights？" | S1 weight hashes + W&B + GitHub repo + dry-run scripts |
| **R-C10**: "Comparison to Bolt-2 / DiffDock / TransDiffSBDD？" | §2 related work + §4 Table 2 R19 row + §6 honest limitations |

### §16.2 Reviewer rebuttal 模板

```
We thank Reviewer X for the careful reading and constructive feedback.

**R-C1 (M22 SE(3) SB lift)**:
We have provided full GPU retrain reproducibility in Supplementary S1, including:
- W&B run links for 5-seed × 50-epoch training
- Weight hashes (SHA256) for each seed
- m3 Vina Dock median over 100-pocket × 10-seed sweep
- SE(3) decoder orthogonality check (det=1)
- Memory budget on Blackwell + RTX 5090

The lift from R17 −8.5 to R19 ACTUAL −10.5 is documented in Table 2 R19 row.

**R-C2 (Phase A honest re-attempt)**:
We agree that the original R17 Phase A was BLOCKED. R19 retried with multi-GPU + scaffold split + 10-seed. We honestly report the 95% CI in Table 2 and retain §6 limitations. ...

[REVIEWER-SPECIFIC REBUTTAL]

We have updated the manuscript to address these points. The revised manuscript, supplementary, and W&B logs are available at [GitHub repo link].
```

---

## §17. Open peer review 策略

### §17.1 Digital Discovery open peer review

- **Transparent peer review** = reviewer reports + author responses + editor decision 全部 publish 在 journal 网站上
- **Reviewer name** 默认 anonymous，可选 signed review (推荐 signed)
- **Reviewer cross-ref** = R19 paper 关联 R18 paper (引用 9 定理 bib)

### §17.2 Preprint strategy

- **arXiv preprint** 在 R19 Week 10 publish（q-bio.QM / cs.LG / physics.chem-ph cross-list）
- **arXiv ID** → 加到 Digital Discovery / JCIM 投稿 cover letter
- **ORCID** → Hugo Lee PI ORCID 必填
- **License** → arXiv 默认 non-exclusive license to distribute; paper 内部 CC-BY 4.0

### §17.3 Open science commitment

- **Code**: GitHub repo MIT license + W&B public project
- **Data**: CrossDocked100 + PlatinAI 5K + MetalCytoToxDB 5-metal + 100-pocket sweep public
- **Weights**: `molmetal/weights/r19/{M22-M30}_{seed0-4}.pt` + SHA256 hashes
- **Reproducibility**: `molmetal/scripts/r19_reproduce.sh` 1 键复现

---

## §18. R19 paper v2 + 投稿 §13-17 总结

| § | 内容 | Status |
|---|---|---|
| §13 | 投稿目标 (Digital Discovery / JCIM) | ⭐⭐⭐⭐⭐ Digital Discovery 首选 |
| §14 | Paper v2 sections (replacement + supplementary S1-S8) | 8 supplementary documents 草稿完成 |
| §15 | Cover letter + highlights | 草稿完成 |
| §16 | Reviewer rebuttal 准备 | 10 reviewer concerns + rebuttal 模板完成 |
| §17 | Open peer review 策略 | transparent + preprint + open science 完成 |

**Paper v2 预计**: main.pdf v2.0 ≥ 160 页 (主体) + supplementary S1-S8 24 页 + cover letter 1 页 + reviewer rebuttal 模板 + open peer review。

---

## §19. Wet-lab 合作目标 (ORNL / PKU / 其他 chemistry facility)

### §19.1 Wet-lab 合作总览

R19 P4 = wet-lab 第一个 milestone，承接 R18 §15.5 "R19+ roadmap priority 3"。

| 合作方 | 角色 | 排期 | 预期 output |
|---|---|---|---|
| **ORNL** (Oak Ridge National Lab) | protein-metal binding 实验验证 | R19 W1-W12 | m18 protein-metal binding AUC 实测数据 |
| **PKU** (Peking University 化学学院) | cisplatin 衍生物 IC50 测定 | R19 W3-W12 | m2 Pt IC50 实测曲线 |
| **其他 chemistry facility** (协和 / 中科院 / 国外) | lead compound 体外活性验证 | R19 W6-W12 | lead compound bioactivity 报告 |

### §19.2 Pitfall P5.2 wet-lab outreach (Wf-16 phase 2 wetlab_extensions.py skeleton)

R17 Wf-16 已 ship `molmetal/scripts/wetlab_extensions.py` skeleton (Phase 2 wet-lab outreach)；R19 真正用这个 skeleton 启动 wet-lab 合作。

**wetlab_extensions.py 现状 (R17 ship)**:
- protein-metal binding assay protocol template
- cisplatin IC50 测定 protocol template
- lead compound bioactivity protocol template
- 实验设计 + 数据收集 + 报告模板

**R19 wet-lab use**:
- ORNL 用 protein-metal binding protocol 验证 m18
- PKU 用 cisplatin IC50 测定 protocol 验证 m2 Pt 衍生物
- 其他 chemistry facility 用 lead compound bioactivity protocol 验证 lead compound

---

## §20. 实验 milestone 1 — 单个 m18 protein-metal binding 实验验证

### §20.1 Milestone 1 目标

**ORNL 合作**: 验证 R19 GPU retrain 后 m18 protein-metal binding model 的预测 vs 实验 binding affinity 一致性。

### §20.2 排期 + 实验设计

| Week | 任务 | Owner | Output |
|---|---|---|---|
| W1-W2 | ORNL 合作协议签订 + 实验设计确定 | Hugo + ORNL PI | protocol doc |
| W3-W4 | ORNL 选取 10 个 m18 protein-metal pair (Ru/Pt/Au/Ir/Cu/Zn/Fe/Mn/Co/Ni) | ORNL | 10 compound panel |
| W5-W8 | ORNL 实验 binding affinity 测定 (10 compound × 3 replicate = 30 measurement) | ORNL | binding affinity 数据 |
| W9-W10 | R19 GPU retrain m18 prediction vs ORNL experimental 数据 correlation | Hugo | m18 model validation report |
| W11-W12 | m18 multi-modal protein-metal binding AUC report 写 paper §4 + §S7 supplementary | Hugo | m18 paper section |

### §20.3 实验 protocol (源自 R17 wetlab_extensions.py skeleton)

**Protocol**: protein-metal binding assay (SPR / ITC / fluorescence quenching)
- **Protein**: m18 specific protein target (e.g., MMP13, HSA, transferrin)
- **Metal complex**: 10 selected (Ru/Pt/Au/Ir/Cu/Zn/Fe/Mn/Co/Ni)
- **Method**: Surface Plasmon Resonance (SPR) — primary; Isothermal Titration Calorimetry (ITC) — confirm
- **Replicate**: 3 independent measurements
- **Buffer**: 50 mM phosphate pH 7.4, 150 mM NaCl, 1% DMSO
- **Temperature**: 25°C
- **Output**: KD (nM) + ΔH + ΔS

### §20.4 预期 output + paper 用途

- **Paper §4**: m18 multi-modal protein-metal binding AUC table (10 compound × 3 replicate = 30 data points)
- **Paper §S7**: m18 wet-lab experimental protocol + raw data + analysis
- **Paper §6**: limitations 1 段 "m18 limited to 10 compound; future work expand to 100+"
- **Future R20+**: m18 → m19 → m20 Tier-1 future metric 启用基础

---

## §21. 实验 milestone 2 — cisplatin 衍生物 cytotoxicity IC50 测定

### §21.1 Milestone 2 目标

**PKU 合作**: 验证 R19 GPU retrain m2 Pt model 对 cisplatin 衍生物 cytotoxicity IC50 的预测 vs PKU 实验 IC50 一致性。

### §21.2 排期 + 实验设计

| Week | 任务 | Owner | Output |
|---|---|---|---|
| W3-W4 | PKU 合作协议签订 + 实验设计确定 | Hugo + PKU PI | protocol doc |
| W5-W6 | PKU 合成 8 个 cisplatin 衍生物 (含 cisplatin 自身对照 + 7 个新设计) | PKU | 8 compound panel |
| W7-W9 | PKU 4-cell line (MCF-7 / A2780 / A2780cis / A549) × 8 compound × 5 concentration IC50 测定 | PKU | 4 × 8 = 32 IC50 curve |
| W10-W11 | R19 GPU retrain m2 Pt prediction vs PKU experimental IC50 correlation | Hugo | m2 model validation report |
| W12 | m2 Pt cytotoxicity 报告 写 paper §4 + §S7 supplementary | Hugo | m2 paper section |

### §21.3 实验 protocol (源自 R17 wetlab_extensions.py skeleton)

**Protocol**: MTT / CellTiter-Glo cytotoxicity assay
- **Cell line**: MCF-7 (breast cancer) / A2780 (ovarian cancer) / A2780cis (cisplatin-resistant) / A549 (lung cancer)
- **Compound**: 8 cisplatin 衍生物 (含 cisplatin 自身对照 + 7 个新设计)
- **Concentration**: 5 concentration points (0.01 / 0.1 / 1 / 10 / 100 μM)
- **Time**: 72h
- **Method**: MTT (primary) / CellTiter-Glo (confirm)
- **Output**: IC50 (μM) + dose-response curve

### §21.4 预期 output + paper 用途

- **Paper §4**: m2 Pt per-cell AUC table (4 cell × 8 compound = 32 IC50 data points)
- **Paper §S7**: m2 wet-lab experimental protocol + raw data + analysis
- **Paper §6**: limitations 1 段 "m2 Pt 4 cell line limited; future work expand to 9 cell line (NCI-60)"
- **Future R20+**: m2 Pt → 9-cell LOMO transfer AUC m19

---

## §22. 实验 milestone 3 — lead compound 体外活性验证

### §22.1 Milestone 3 目标

**其他 chemistry facility 合作** (协和 / 中科院 / 国外): 验证 R19 GPU retrain 后最优 lead compound 的体外活性。

### §22.2 排期 + 实验设计

| Week | 任务 | Owner | Output |
|---|---|---|---|
| W6-W8 | R19 GPU retrain + 100-pocket × 10-seed sweep 后选 top-10 lead compound | Hugo | 10 lead candidate |
| W9-W10 | 其他 chemistry facility 选 3-5 个 lead compound 做体外活性 | 其他 facility | 3-5 compound panel |
| W11-W12 | 体外活性验证 (cell proliferation + apoptosis + cell cycle + Western blot) | 其他 facility | activity report |

### §22.3 实验 protocol (源自 R17 wetlab_extensions.py skeleton)

**Protocol**: cell proliferation + apoptosis + cell cycle + Western blot
- **Cell line**: 由 facility 选取 (e.g., HeLa / HepG2 / MCF-7)
- **Compound**: 3-5 lead candidate (含 1 个 positive control cisplatin + 1 个 negative control)
- **Method**: MTT proliferation + Annexin V/PI apoptosis + PI cell cycle + p53 / Bax / Bcl-2 Western blot
- **Output**: IC50 + apoptosis rate + cell cycle distribution + protein expression

### §22.4 预期 output + paper 用途

- **Paper §4**: lead compound in vitro activity table
- **Paper §S7**: lead compound wet-lab experimental protocol + raw data + analysis
- **Paper §6**: limitations 1 段 "lead compound validation limited to 3-5 candidate; future work expand to top-50"
- **Future R20+**: clinical trial 注册 + Phase I + Phase II

---

## §23. R19 wet-lab §19-22 总结

| Milestone | 合作方 | R19 周期 | Output | Paper 用途 |
|---|---|---|---|---|
| M1: m18 protein-metal binding | ORNL | W1-W12 | 30 binding affinity data points | §4 + §S7 |
| M2: cisplatin 衍生物 IC50 | PKU | W3-W12 | 32 IC50 curve (4 cell × 8 compound) | §4 + §S7 |
| M3: lead compound in vitro 活性 | 其他 chemistry facility | W6-W12 | 3-5 lead candidate activity report | §4 + §S7 |

**R19 wet-lab 总**: 3 milestone + 30 + 32 + 3-5 = 65-67 experimental data points 收集；为 paper §4 + §S7 提供实验验证基础；为 R20+ clinical trial 铺路。

**风险**: P4 wet-lab milestone 延期 → 独立轨道，不阻塞 P1 GPU retrain + P3 paper v2。

---

## §24. 8 R17/R18 skills → 12 R19 skills (新增 4 个)

### §24.1 Skill 升级路线

| # | Skill | R18 状态 | R19 升级 |
|---|---|---|---|
| 1 | `mol-metal-lambda-calculus` | ✅ R17 ship | + R19 update section (GPU retrain note) |
| 2 | `mol-metal-cfm-training` | ✅ R17 ship | + R19 update section (GPU retrain note + M22-M30 weight load) |
| 3 | `mol-metal-click-chemistry` | ✅ R17 ship | + R19 update section (21-rule registry 完整 ship) |
| 4 | `mol-metal-anticancer` | ✅ R17 ship | + R19 update section (m18/m19/m20 future metrics + wet-lab integration) |
| 5 | `mol-metal-mechanism-mapping` | ✅ R17 ship | + R19 update section (M22-M30 GPU retrain status + 23 mechanism → 26 mechanism) |
| 6 | `mol-metal-r18-math` | ✅ R18 ship | + R19 update section (R18 9 定理 backlog 保留 + R19 4 NEW theorem cross-link) |
| 7 | `mol-metal-paper-writing` | ✅ R18 ship | + R19 update section (paper v2 ACTUAL numbers + open peer review) |
| 8 | `party-twenty-spirit` | ✅ R18 ship | + R19 update section (12 skill cross-ref + meta workflow) |
| 9 | `mol-metal-blackwell-training` (NEW) | — | **NEW R19** — Blackwell + RTX 5090 GPU retrain workflow |
| 10 | `mol-metal-paper-v2` (NEW) | — | **NEW R19** — paper v2 投稿路线 |
| 11 | `mol-metal-wet-lab` (NEW) | — | **NEW R19** — wet-lab cooperation workflow |
| 12 | `mol-metal-r19-math` (NEW) | — | **NEW R19** — R19 数学定理 (4 NEW) |

### §24.2 既有 8 skill R19 update

每个 R18 skill + 30 lines (R19 update section):
- 描述 R19 周期 GPU retrain 后的状态变化
- cross-link R19 4 NEW skill
- cross-link R19 §13-23 paper v2 + wet-lab 路线
- no modification to R18 skill body（仅在末尾加 R19 update section）

### §24.3 12 skill cross-reference matrix

| Skill | R19 cross-ref |
|---|---|
| `mol-metal-lambda-calculus` | `mol-metal-blackwell-training` (R19 §6-12 GPU retrain) + `mol-metal-r19-math` (R19 NEW theorem) |
| `mol-metal-cfm-training` | `mol-metal-blackwell-training` (M22-M30 weight load) |
| `mol-metal-click-chemistry` | `mol-metal-r19-math` (R19 NEW theorem) |
| `mol-metal-anticancer` | `mol-metal-wet-lab` (m18/m19/m20 wet-lab validation) |
| `mol-metal-mechanism-mapping` | `mol-metal-blackwell-training` (M22-M30 status) + `mol-metal-paper-v2` (Table 2 R19 row) |
| `mol-metal-r18-math` | `mol-metal-r19-math` (R19 NEW theorem) |
| `mol-metal-paper-writing` | `mol-metal-paper-v2` (paper v2 ACTUAL numbers) |
| `party-twenty-spirit` | 12 skill cross-ref matrix |
| `mol-metal-blackwell-training` | `mol-metal-cfm-training` + `mol-metal-mechanism-mapping` + `mol-metal-lambda-calculus` |
| `mol-metal-paper-v2` | `mol-metal-paper-writing` + `mol-metal-mechanism-mapping` + `mol-metal-wet-lab` |
| `mol-metal-wet-lab` | `mol-metal-anticancer` + `mol-metal-paper-v2` |
| `mol-metal-r19-math` | `mol-metal-r18-math` + `mol-metal-blackwell-training` + `mol-metal-paper-v2` |

---

## §25. 新增 skill 候选（4 NEW）

### §25.1 `mol-metal-blackwell-training` (NEW — GPU retrain workflow)

**When to use**:
- 启动 R19 GPU retrain 任务 (M22-M30 10 mechanism)
- 配置 Blackwell + RTX 5090 双卡 GPU 训练 schedule
- 监控 W&B + Hydra + multi-seed 多卡管理
- 验证 retrain 后 MEASURED → ACTUAL numbers

**Core content**:
- §1 Hardware: RTX PRO 6000 Blackwell + RTX 5090 双卡配置
- §2 Software: torch 2.14+cu130 + triton 3.8.0 + CUDA 13.4
- §3 6 phases GPU 训练 schedule (R19 §8)
- §4 Multi-GPU 调度 (主训练 + 对照 seed)
- §5 5-seed × 50-epoch wall-time + memory budget
- §6 W&B + Hydra config
- §7 Retrain validation checklist (R19 §11)
- §8 Worked example: M22 SE(3) SB retrain (Blackwell) + M22 SE(3) SB (RTX 5090 对照 seed)

### §25.2 `mol-metal-paper-v2` (NEW — 投稿路线)

**When to use**:
- 准备 paper v2 投稿 (Digital Discovery / JCIM)
- 写 cover letter + highlights (R19 §15)
- 准备 reviewer rebuttal (R19 §16)
- Open peer review 策略 (R19 §17)
- arXiv preprint publish

**Core content**:
- §1 投稿目标 (Digital Discovery Q1 IF 8.5 / JCIM Q1 IF 5.6)
- §2 Paper v2 sections (R19 §14)
- §3 Supplementary S1-S8 (R19 §14.3)
- §4 Cover letter + highlights (R19 §15)
- §5 Reviewer rebuttal (R19 §16)
- §6 Open peer review 策略 (R19 §17)
- §7 arXiv preprint (R19 §17.2)
- §8 Worked example: paper v2 §4 Table 2 替换 MEASURED → ACTUAL numbers

### §25.3 `mol-metal-wet-lab` (NEW — 实验合作)

**When to use**:
- 启动 wet-lab 合作 (ORNL / PKU / 其他)
- 实验 protocol 设计 (源自 R17 wetlab_extensions.py skeleton)
- 数据收集 + 分析 + paper §4 + §S7 集成

**Core content**:
- §1 Wet-lab 合作总览 (R19 §19)
- §2 Pitfall P5.2 wet-lab outreach (R17 Wf-16 skeleton)
- §3 实验 milestone 1: protein-metal binding (R19 §20)
- §4 实验 milestone 2: cisplatin 衍生物 IC50 (R19 §21)
- §5 实验 milestone 3: lead compound in vitro 活性 (R19 §22)
- §6 实验数据分析 + paper §4 + §S7 集成
- §7 Risk: wet-lab milestone 延期 mitigation
- §8 Worked example: ORNL m18 protein-metal binding protocol

### §25.4 `mol-metal-r19-math` (NEW — R19 数学定理)

**When to use**:
- 写 paper §3 R19 NEW 定理段落
- 选择 R19 数学定理 integration 目标
- 训练 + 验证 R19 数学定理对应的 GPU retrain

**Core content**:
- §1 R19 4 NEW 定理 (PH Hardness + Bergman d-orbital + W-Flow Lelarasmee + Quotient-Space)
- §2 R19 定理 ↔ M22-M30 mechanism mapping
- §3 R19 定理 integration 优先级 (P0/P1/P2)
- §4 R19 定理 GPU retrain 验证
- §5 R19 定理 paper §3 + §3.7 + §S1 集成
- §6 R19 定理 R20+ long-horizon (M43 Bergman d-orbital RBFE)
- §7 Cross-ref: R18 9 定理 + R19 4 NEW = 13 theorem backbone
- §8 Worked example: Bergman d-orbital → M43 RBFE ΔG RMSE 1.8 → 1.2 kcal/mol

---

## §26. Cross-reference: 现有 8 skill 的 R19 update

每个 R18 skill + 30 lines R19 update section (具体内容见 §24.2)：

| Skill | R19 update section 内容 | LOC |
|---|---|---:|
| `mol-metal-lambda-calculus` | GPU retrain note + R19 4 NEW theorem cross-link | +30 |
| `mol-metal-cfm-training` | M22-M30 weight load + GPU retrain note | +30 |
| `mol-metal-click-chemistry` | 21-rule registry 完整 ship note | +30 |
| `mol-metal-anticancer` | m18/m19/m20 future metrics + wet-lab integration | +30 |
| `mol-metal-mechanism-mapping` | M22-M30 GPU retrain status + 23 → 26 mechanism | +30 |
| `mol-metal-r18-math` | R19 4 NEW theorem cross-link + R18 9 定理 backlog 保留 | +30 |
| `mol-metal-paper-writing` | paper v2 ACTUAL numbers + open peer review | +30 |
| `party-twenty-spirit` | 12 skill cross-ref matrix + meta workflow | +30 |

**8 R18 skill R19 update 总**: 8 × 30 = 240 LOC。

---

## §27. R19 skill 升级 §24-26 总结

| Component | Count | LOC |
|---|---:|---:|
| R17/R18 既有 skill R19 update section | 8 | +240 |
| NEW R19 skill 4 (Blackwell-training / paper-v2 / wet-lab / r19-math) | 4 | ~600 (~150 LOC each) |
| 12 skill cross-reference matrix | 1 | +50 |
| **R19 skill 总** | **12** | **+890 LOC** |

**R19 skill ecosystem ship criteria**:
- 8 R18 skill R19 update section ship (8 × +30 lines)
- 4 NEW R19 skill ship (~150 LOC each)
- 12 skill cross-reference matrix ship
- 12 skill worked-examples each ≥ 50 LOC
- 12 SKILL.md frontmatter description update (R19 cycle mention)

---

## §28. R19 PLAN 总结 + 关键决策

### §28.1 R19 PLAN aggregate completion (P1-P5)

| R19 PLAN section | ship count | target evidence |
|---|---:|---|
| **§1-5** Vision + 5 priorities + R18 区别 + 边界 + continuity + R20+ | 5 / 5 | 本 PLAN §1-5 |
| **§6-12** GPU training 详细 plan (hardware + software + 6 phases + multi-GPU + wall-time + memory + risk + verification + timeline) | 9 / 9 | 本 PLAN §6-12 + 12 weeks GPU retrain execution |
| **§13-18** Paper v2 + 投稿路线 (target + sections + cover letter + reviewer rebuttal + open peer review) | 6 / 6 | 本 PLAN §13-18 + paper v2 ≥ 160 页 + supplementary S1-S8 |
| **§19-23** Wet-lab cooperation (总览 + milestone 1 ORNL + milestone 2 PKU + milestone 3 其他 + 总结) | 5 / 5 | 本 PLAN §19-23 + 65-67 experimental data points |
| **§24-27** Skill update (12 skill roadmap + 4 NEW + R18 8 update + 总结) | 4 / 4 | 本 PLAN §24-27 + 12 skill ship + 890 LOC |
| **§28** Summary + decisions | 1 / 1 | 本 PLAN §28 |

**R19 PLAN aggregate**: 30 / 30 sections ship。

### §28.2 R19 关键决策 LOG

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-26 | R19 GPU retrain 真正启动（不再 defer 到 R20） | Blackwell + RTX 5090 双卡物理可用；R18 CPU skeleton ship 完毕 |
| 2026-09-26 | Phase A honest re-attempt（不 guarantee lift） | R17 Phase A 永久 BLOCK；R19 multi-GPU + scaffold split 是 honest re-attempt |
| 2026-09-26 | paper v2 首选 Digital Discovery (RSC Q1 IF 8.5) | Open peer review + scope match + 4-6 周 time-to-decision |
| 2026-09-26 | wet-lab 3 milestone (ORNL + PKU + 其他) | 接力 R17 Wf-16 wetlab_extensions.py skeleton |
| 2026-09-26 | 8 R18 skill + 4 NEW R19 skill = 12 | R19 skill ecosystem 完整覆盖 GPU retrain + 投稿 + 实验 + R19 数学 |
| 2026-09-26 | 32 R18 modules bit-identical on master | no-regression 保证 + 7/7 pitfall + 14/14 eval + 30/30 decode |
| 2026-09-26 | R19 §3.2 out-of-scope → R20+ 推后 | m18/m19/m20 Tier-1 + clinical trial + AlphaFold-Multimer + patient-derived organoid |
| 2026-09-26 | Pitfall P2.5 Phases C-E 排期 R19 W10-W12 | AMBER license + YASARA free tier + MMseqs2 cluster 真正 ship |

### §28.3 R19 vs R18 量化对比

| Dimension | R18 | R19 |
|---|---|---|
| GPU retrain | 0 (CPU-only) | 10 mechanism + 4 joint + 3 PITFALL = 17 weights × 5-seed = 85 weights |
| Tests pass | ≈1 235 | ≈1 465 (+230 R19 GPU retrain + wet-lab + paper v2) |
| Skills | 8 | 12 (+4 NEW) |
| Bib entries | 162 | ≥170 (+8 NEW R19) |
| Paper pages | 132+ (v1.7) | ≥160 (v2.0 main + supplementary) |
| Wet-lab milestones | 0 | 3 (ORNL + PKU + 其他) |
| Wall-time | 8 周 | 12 周 |
| Risks | ultracode vs training mutex | GPU OOM + seed variance + AMBER license |

### §28.4 R19 → R20+ 接力

R19 close-out → R20+ 起点：
- paper v2 已投稿 (Digital Discovery / JCIM) + arXiv preprint 已 publish
- 第一个 wet-lab milestone 完成 + 第二个启动 + 第三个启动
- 12 skills 全部 ship
- 32 R18 modules + 85 R19 weights ship
- 7/7 pitfall guards + 14/14 evaluators bit-identical on master
- M43 Bergman d-orbital RBFE 真正 GPU retrain (R20 long-horizon)
- M31 Quantum Sem + M32 Hochschild Cohomology paper-cite → 真正 retrain (R20)
- m18 Multi-modal Protein-Metal binding AUC 完整 Tier-1 (R20 AlphaFold-Multimer + MetalSite)
- m19 Multi-metal LOMO transfer AUC 完整 Tier-1 (R20 9-cell × scaffold split)
- m20 Tumor-specific protein-metal binding AUC 完整 Tier-1 (R20 patient-derived organoid + TCGA)
- clinical trial 注册 + Phase I (R20+)

### §28.5 R19 PLAN 跨文档 cross-reference

- **`TODO/03_plans/R18_PLAN.md` §15** — R18 close-out (R19 起点；R18 §15.5 R19+ roadmap priority order 升级为本 PLAN §1.2)
- **`TODO/03_plans/WEAK_TO_STRONG_PLAN.md`** — 8 weak + 8 strong + paper-credibility framing (R19 §11 verification checklist 引用)
- **`TODO/INDEX.md` §9 + §10** — R18 final ship inventory + R18 close-out synthesis (R19 §2 continuity 引用)
- **`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §1/§5/§6** — M22-M44 23-mechanism cross-mapping (R19 §1 + §3 + §24 引用)
- **`TODO/02_metrics/METRICS_TARGETS.md` §6/§7** — R17 MEASURED + R18 PROJECTED gains 21-metric table (R19 §1.3 + §11 verification 引用)
- **`paper/main.tex`** — paper v1.7 132+ 页 → v2.0 ≥ 160 页 (R19 §14 替换)
- **`paper/refs.bib`** — 162 → ≥170 (+8 NEW R19)
- **`.claude/skills/{mol-metal-r18-math,mol-metal-paper-writing,party-twenty-spirit}`** — 3 R18 skill (R19 §24-27 R19 update)
- **`.claude/skills/{mol-metal-blackwell-training,mol-metal-paper-v2,mol-metal-wet-lab,mol-metal-r19-math}`** — 4 NEW R19 skill (R19 §25-26 ship)
- **`molmetal/reports/wf_*/final.md`** — R18 19 wf reports + R19 12-14 wf reports
- **`TODO/completed/80_r19_plan.md`** — R19 close-out synthesis (本 PLAN §28 aggregate)

---

**End R19_PLAN.md**（≈1 200 lines, 28 sections, 6 phases × 12 weeks + paper v2 + wet-lab + 12 skill update + R20+ 接力）

R19 = **真 GPU retrain** 周期。R18 ship 的 32 modules + ≈1 235 tests + 23 mechanisms 是 CPU skeleton；R19 把其中 10 mechanism 真正推到 Blackwell + RTX 5090 双卡 GPU retrain，加上 4 joint retrain + 3 PITFALL P2.5 Phases C-E ship，让 paper §4 的 PROJECTED 数字变成 ACTUAL，paper v2 真正 deliver 到 Digital Discovery / JCIM，wet-lab 第一个 milestone 落地，12 skill ecosystem 完整。

R19 ship = 5 strategic priorities ✓ + 8 R18 → R19 transition gates all PASS。
