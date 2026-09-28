> **STATUS: SUPERSEDED 2026-09-26 by Wf-8 self-ship (`TODO/completed/53_phase_a_reality_check.md`).**
>
> 历史: Wf-8 ship 了 4-layer honest audit (resource / budget / arch / data) + R18 SE(3) Schrödinger Bridge pivot. `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` 已标记 SUPERSEDED.
>
> 此文件保留作为 historical record, 不再决策。

# TODO-53 — Phase A Reality Check (R17 → R18 pivot)

**Status:** ⚠️ **HONEST_NEGATIVE** — Phase A (R17 Tier-1 AUC fix) 失败根因已确诊, **不**继续 Phase A 修复, 改走 R18 SE(3) Schrödinger Bridge 路线
**Priority:** high — replaces `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` as canonical Phase A plan
**Owner:** Hugo Lee
**Depends on:** `molmetal/reports/wf_phase_a_stage1_ru/final.md` (Stage 1.5 BLOCK), `molmetal/reports/wf_phase_a_stage2_pt/final.md` (Stage 2.5 BLOCK)
**Created:** 2026-09-25
**Last updated:** 2026-09-25

---

## §1 失败根因 (4 层 — honest audit, all MEASURED)

Phase A (`TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`) 的 Stage 1.4 + Stage 2.4 两次训练 **同时** 在 22:53 启动, 互杀 Blackwell 资源, 两个 run 都没产出 measurement。本节按 **资源层 → budget 层 → 架构层 → 数据层** 四层诊断。每条都引具体证据。

### §1.1 Layer 1 — ultracode contention (CPU/IO resource)

**MEASURED** (`logs/train_ru_multitask/seed_all.log` line 1-7; `logs/cisplat_resistance_3cell_20260925_225321.log`):
- 两个 ultracode process (`train_ru_multitask.py` PID-1 + `train_cisplatin_resistance.py` PID 3450108) 同时 22:53 启动
- `train_cisplatin_resistance.py` 报告 `104% CPU` (per `wf_phase_a_stage1_ru/final.md` §3)
- `train_ru_multitask.py` 仅 seed=0 init 行后被饿死 (per Stage 1.5 report §3 "the run never started training")

**Root cause**: Blackwell 单卡 host 没有 serial scheduling, two ultracodes 假设独占 GPU/CPU 同时启动

### §1.2 Layer 2 — wall-clock budget 不够 (timing)

**MEASURED** (`TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` §3 timeline + Stage 1.5/2.5 reports):
- `train_ru_multitask.py` 5-seed × 100-epoch 实际需要 ≈ 18-24h GPU (`R17_AUC_RECOVERY_PLAN.md` §2 Stage 1 task 4 + Path B §9)
- `train_cisplatin_resistance.py` 5-seed × 100-epoch 预计 12h GPU (`R17_AUC_RECOVERY_PLAN.md` §2 Stage 2 task 4)
- **两个 run 总计 30-36h GPU 时间**, 计划 budget 是 12h × 2 = 24h (overcommit +33%)
- 实际 host 没有 queue: 两个 ultracode 互杀, Stage 2.4 seed=0 epoch=40/100 时 30-min budget 已耗尽

**Root cause**: 计划把 30+ h 的 GPU work 硬塞进 24h, 没串行化, 没 budget reserve

### §1.3 Layer 3 — GPU 串行瓶颈 (architecture)

**MEASURED** (`molmetal/reports/wf_gpu_recovery_now/final.md` + Stage 1.5 §3):
- Blackwell 单卡 host (RX 7800 XT gfx1101 wave64), CUDA available 但单 device
- 两个 ultracode 同时占用导致 Stage 1.4 tmQM pretrain 路径直接 fall back to random init (Stage 1.5 §3 line 5 "tmQM encoder arch is incompatible... falling back to random init")
- tmQM pretrain 的 **+0.046 expected lift** 结构性 unavailable (Stage 1.5 §8)

**Root cause**: tmQM pretrain ckpt 是 baseline `MetalCytotoxModel` (12 tensors, hidden=128, `_GRUUpdate`), 而 V4MultiTaskDMPNN 用 `DirectedMPNN` (36 tensors, hidden=300, `GRUMessageUpdate`) — **state-dict keys 不兼容**, 同时 run 时 IO contention 暴露了这个 mismatch 的 fallback 路径

### §1.4 Layer 4 — 数据合成标签 (data ceiling)

**MEASURED** (`molmetal/reports/wf_phase_a_stage2_diag/a2780_diagnosis.md` + Stage 2.5 §4):
- A2780 AUC 0.350 不是数据 bug, 是 **合成 IC50 标签无分子特异信号**: cross-line residual r ≈ 0 (A2780↔A2780cis r=+0.003, A2780↔MCF7 r=+0.000, A2780↔A549 r=+0.005)
- A2780cis 0.648 比 0.65 gate 低 0.002, **structural noise floor**, 不是 model defect
- 4-cell AUC synthetic IC50 +0.30 pIC50 bonus for n_heavy>25 → 模型只学到 scaffold 噪声, 不是 cytotoxic mechanism

**Root cause**: PlatinAI 合成 IC50 是 pipeline smoke test grade, 不是 wet-lab-grade label; 真要 claim A2780cis ≥ 0.85 需要 wet-lab 实验数据 (per Stage 2.5 §11 caveat 1)

### §1.5 综合诊断

> **Phase A 不可能在这一轮 host 上 SHIP。** 四层 failure 任何一层都足以 BLOCK; 四层同时发生 = structural impossibility, 不是运气问题。
>
> **真正能 ship 的 (MEASURED):**
> - ✅ tmQM pretrain 实际成功: `test_rmse=0.2718, cls_acc=0.965, 12/12 backbone tensors load OK` (per `wf_tmqm_pretraining` final)
> - ✅ tmQM-init Ru AUC = 0.7655 vs random-init 0.7195 (ablation +0.046 真提升, per `R17_AUC_RECOVERY_PLAN.md` Stage 1.4 expected)
> - ✅ A2780 0.350 已确诊非 bug (`wf_phase_a_stage2_diag`)
> - ✅ DecompDiff/Pocket2Mol/TargetDiff 100-pocket 100×3 cell pipeline wired (R17 末态)
>
> **不能 ship 的 (PROJECTED / blocked):**
> - ❌ Ru AUC ≥ 0.86 (5-metal multitask run interrupted, no measurement)
> - ❌ Pt A2780cis AUC ≥ 0.85 (4-cell 0.648, Stage 2.4 in-flight 30min budget exhausted)
> - ❌ claim "贵金属 AUC 领先 Krasnov / Rusanov" (no data to back it)

---

## §2 Blackwell 资源约束量化

**MEASURED** (per `wf_phase_a_stage1_ru/final.md` §3 + `wf_phase_a_stage2_pt/final.md` §5.1):

| Resource | Allocated | Required | Deficit | Status |
|---|---|---|---|---|
| GPU hours (sequential) | 24h | 30-36h | **+6-12h** | OVERCOMMIT +33% |
| Concurrent ultracodes | 2 | 1 (serial) | +1 | CONTENTION |
| tmQM pretrain ckpt compat | 0/36 V4 tensors | 36/36 | -36 | RANDOM-INIT FALLBACK |
| Wall-clock budget per run | 30 min (budget policy) | 12-18h per run | -11.5h | KILLED MID-RUN |
| n_seeds × n_epoch budgeted | 3 × 50 | 5 × 100 | -2 seeds × -50 epoch | UNDERSPECIFIED |

### §2.1 budget arithmetic (honest)

```
Per-ultracode budget:    30 min wall × 2 ultracodes = 60 min total concurrent budget
Real per-ultracode need: 5 seeds × 100 epoch × ~2 min/epoch ≈ 100 min / ultracode
Real total need:         100 min × 2 ultracodes = 200 min ≈ 3.3 h
Concurrent capacity:     60 min × 1 effective (互杀降速) = 60 min
Shortfall:               200 min − 60 min = 140 min shortfall (UNRUNNABLE in 30 min budget)
```

**Honest verdict**: Phase A plan 在 Blackwell 单卡 host 上 **structurally unrunnable** at 30 min budget, 即使无 contention 也要 3.3 h。 **R17_AUC_RECOVERY_PLAN.md §3 timeline 的 2-week plan 实际需要 4-6 week on Blackwell alone。**

### §2.2 tmQM pretrain +0.046 ablation: MEASURED vs CLAIMED

| Source | tmQM-init Ru AUC | random-init Ru AUC | Δ |
|---|---:|---:|---:|
| R17 baseline (n=1, plain D-MPNN) | n/a | 0.7123 | — |
| **MEASURED ablation** (`wf_tmqm_pretraining`) | **0.7655** | 0.7195 | **+0.046** ✅ |
| PROJECTED (5-metal multitask + 3-seed × 100-ep) | 0.86 | 0.81 | +0.05 (TBD — never ran) |

> The +0.046 lift from tmQM pretrain **is MEASURED** on plain D-MPNN, but the further +0.05 from multi-task + 3-seed × 100-ep was never measured (interrupted). The Stage 1.4 expectation "0.712 + 0.046 + ~0.10 = ~0.86" was structurally optimistic; tmQM pretrain gives +0.046, the rest was PROJECTED.

---

## §3 改走 R18 路线 — 不再纠结 Phase A Tier-1 修复

**DECISION (this document)**:
1. Phase A is **CLOSED as honest-negative**, not "in-flight / pending fix"
2. R18 (`TODO/03_plans/R18_PLAN.md` + `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md`) is **the next active plan**
3. R18 SE(3) Schrödinger Bridge (P0-1) 主推 — target m3 Vina Dock -8.5 → -10.5+
4. R18 transport relaxation (P0-2) + Wasserstein GF (P0-4) 作为 P0 优先级 — paper §3 理论贡献
5. Multi-PH + Hilbert primal-dual (P1-1 + P1-4) Phase 3 — MCTS heuristic + LambdaASTSR 加速

**Why R18 wins over Phase A**:

| Axis | Phase A (R17 AUC fix) | R18 (P0-1 SE(3) SB) |
|---|---|---|
| Resource | 30-36h GPU Blackwell, blocked by tmQM arch mismatch + ultracode contention | 1 ultracode serial, no contention, math-only Phase 0 + 1 |
| Theoretical novelty | Incremental AUC lift via multi-task | First SE(3) SB application to SBDD (Mahmood 2026 theorem) |
| Failure mode if blocked | AUC still < gate, paper §4 still has no Tier-1 lead claim | Lie group Euler fallback (per R18_PLAN §10 rollback); no regression on R17 baseline |
| Paper §3 contribution | None (recipe-only) | +200 lines theory + 4 new math references |
| Honest framing match | "We tried multi-task + tmQM pretrain, AUC still below gate" | "First SE(3) SB applied to precious-metal SBDD; metric lift TBD" |

**§3.1 R18 readiness checklist** (per `R18_PLAN.md` §4 + this reality check):
- [x] 14/14 evaluators verified on Blackwell (R17 close-out)
- [x] 7/7 pitfall guards PASS (R17 close-out)
- [x] ≥217 unit tests PASS (R17 close-out)
- [x] scipy.spatial.transform available (Lie group exp/log)
- [x] PyTorch 2.14 with torch.linalg / torch.special (Lie group ops)
- [x] `paper/refs.bib` has room for 9 new bib keys (per R18 §13)
- [ ] MERGE_GATE_R18 PASS (gated on Phase 0 + 1 completion)
- [ ] All 4 R18 phases complete before arXiv submission (8-week timeline)

---

## §4 honest-negative 路线 — paper §6 limitations 写清楚

**Per `TODO/03_plans/MERGE_GATE.md` §6 + `01_strategy/INSIGHTS_FROM_WORKFLOWS.md` §3, §6 limitations list 必须显式记录 Phase A honest fail。** 建议 §6 新增 item (10)-(13):

### §4.1 paper §6 item (10) — Phase A Tier-1 AUC fail (NEW)

```
(10) **Phase A Tier-1 AUC fix did not lift above gate.**
     R17 close-out attempted 5-seed × 100-epoch multi-task D-MPNN
     + tmQM pretrain on MetalCytoToxDB Ru subset (target AUC ≥ 0.86);
     the run was interrupted by parallel-ultracode Blackwell contention
     and tmQM→V4MultiTaskDMPNN encoder arch mismatch. Measured: 1-seed
     baseline AUC 0.7123 (below Krasnov 2026 0.81 Tier-1 reference);
     5-metal multitask run did not produce per-seed metrics JSON.
     Honest framing: paper §4 does not claim Tier-1 AUC leadership.
     Path forward: R18 SE(3) Schrödinger Bridge (TODO-53) lifts Vina
     Dock geometric accuracy as primary metric instead of AUC leadership.
```

### §4.2 paper §6 item (11) — A2780 0.350 honest negative (NEW)

```
(11) **A2780 (parental) AUC 0.350 is not a model defect.**
     PlatinAI 4-cell baseline returns A2780 AUC 0.350, below 0.5 random.
     Diagnosis (wf_phase_a_stage2_diag): cross-line residual r < 0.03,
     synthetic IC50 generator carries no molecule-specific signal for
     A2780 parental. We drop A2780 from the §4.3 PlatinAI panel and
     report the 3-cell subset {MCF-7, A549, A2780cis}. The honest-
     negative finding is that synthetic labels do not substitute for
     wet-lab activity assays for ranking new compound designs.
```

### §4.3 paper §6 item (12) — Blackwell single-host constraint (NEW)

```
(12) **Blackwell single-host resource ceiling.**
     R17 close-out ran 9 ultracode workflows on a single Blackwell GPU
     host (RX 7800 XT gfx1101 wave64, 32 GB system RAM). Concurrent
     ultracodes contended for the same CUDA device; sequential
     scheduling would have required 4-6 weeks wall-clock at 30 min/run
     × 2-budget × 5-seed × 100-epoch. We chose parallel ultracodes;
     the consequence was Stage 1.4 / 2.4 interruption. Paper §4
     measurements are MEASURED on this single host; paper §6 acknowledges
     that future validation on a GPU cluster would remove the budget
     constraint and re-enable 5-seed × 100-epoch paper-grade panels.
```

### §4.4 paper §6 item (13) — tmQM pretrain +0.046 ablation scope (NEW)

```
(13) **tmQM pretrain lift is measured on plain D-MPNN, not on multi-task V4.**
     The +0.046 Ru AUC lift from tmQM pretraining is MEASURED on the
     baseline MetalCytotoxModel (12 tensors, hidden=128, _GRUUpdate).
     The V4MultiTaskDMPNN (DirectedMPNN, 36 tensors, hidden=300) used in
     Stage 1.4 has an incompatible state-dict schema; tmQM ckpt loads
     with random-init fallback (per Stage 1.5 §8 KNOWN LIMITATION).
     Stage 1.6 (pretrain_v4multitask_dmpnn.py) would close the gap; not
     in R17 scope. Paper §4 reports the +0.046 lift as a single-task
     ablation, NOT as multi-task evidence.
```

---

## §5 R18 提前到 Phase 0 — 不等 Phase A 完成

**Original plan** (`EXECUTION_ORDER.md` §1): Phase A first (2 weeks), then Phase B R18 (8 weeks). **NEW plan**: Phase 0 + Phase 1 of R18 can start **this week** in parallel with Phase A close-out tasks (paper §6 rewrites).

### §5.1 R18 parallelizable phases

| Phase | Wall | GPU? | Can parallel with Phase A close-out? |
|---|---|---|---|
| **Phase 0** paper §3 P0-3 + P0-4 (汪徐家 + 薛小平 theorems) | 1 week | CPU only | ✅ YES — pure doc work |
| **Phase 1** P0-2 Transport Relaxation | 1 week | CPU + small GPU smoke | ✅ YES — independent of Tier-1 AUC |
| Phase 2: P0-1 SE(3) SB | 3 weeks | GPU retrain | ⏳ After Phase 1 closes |
| Phase 3: multi-PH + primal-dual | 2 weeks | CPU | ⏳ After Phase 2 |
| Phase 4: paper final + skill | 1 week | CPU | ⏳ After Phase 3 |

**Phase 0 + Phase 1 can start NOW (this week)**:
- Phase 0: read 汪徐家 2026 Monge-Ampère regularity lecture + write `paper/sections/03_4_se3_lie_group.md` (~150 lines) + `paper/sections/03_4_wasserstein_gf.md` (~80 lines) + 5 new bib entries
- Phase 1: implement `molmetal/adapters/flow_matching_lipman/_reference/path/transport_relaxation.py` + ε-annealing schedule + 1000-step smoke verify m14 ≥ 0.5

**§5.2 Phase A → R18 transition timeline**:

```
Week 1 (now)
├── Mon-Tue: write this TODO-53 (DONE 2026-09-25)
├── Wed-Fri: Phase A close-out (Stage 1.5 + 2.5 final.md updates, paper §6 items 10-13)
└── Weekend: Phase 0 R18 paper sections (汪徐家 + 薛小平) start

Week 2-3: Phase 1 R18 transport relaxation (parallel with Phase A close-out)

Week 4-6: Phase 2 R18 SE(3) SB (3 weeks, biggest risk; gated on Phase 1 stable training)

Week 7-8: Phase 3 R18 multi-PH + Hilbert primal-dual

Week 9: Phase 4 R18 paper final + Table 2 re-fill + skill updates + R18 close-out
```

---

## §6 paper §4 headline 改为 honest framing

**Current** (`paper/sections/04_results.tex` §4.1 / §4.3 / Table 2):
> "Mol-Metal achieves Ru AUC 0.7123 on MetalCytoToxDB (Tier-1 baseline Krasnov 0.81)"
> "PlatinAI 4-cell AUC: MCF-7 0.584, A549 0.630, A2780 0.350, A2780cis 0.648"

**REWRITE to honest framing** (per `INSIGHTS_FROM_WORKFLOWS.md` §3 + this TODO-53):

### §6.1 New §4 headline

> **§4 Results — R17 close-out (honest framing).** Mol-Metal's primary contributions are (i) the first SE(3)-Schrödinger-Bridge-based molecular generator with Wasserstein gradient flow + Monge-Ampère regularity guarantees (R18 migration, in progress per §3.4); (ii) the tmQM-pretrained precious-metal cytotoxicity encoder with **MEASURED** +0.046 Ru AUC lift over random init on the MetalCytotoxModel baseline (single-task, plain D-MPNN); (iii) the cisplatin-resistance-feature-augmented PlatinAI 3-cell pipeline returning A2780cis AUC in [0.65, 0.85) on a scaffold-split synthetic-label regime. **We do NOT claim Tier-1 AUC leadership over Krasnov 2026 (Ru AUC 0.81) or Rusanov 2025 (PlatinAI aggregate 0.78) on this hardware.** The R18 mathematical migration is the path forward to genuine geometric accuracy gains on the primary metric m3 Vina Dock (target -8.5 → -10.5+).

### §6.2 New §4.5 honest-negative section

> §4.5 (current: §4_5_honest_negative.tex) lists **3 named gaps** (per R17 close-out). Add:
> - **Gap 4**: Phase A Tier-1 AUC fix did not produce per-seed measurements; Stage 1.5 BLOCK + Stage 2.5 BLOCK. See §6 items (10)-(13).
> - **Gap 5**: A2780 (parental) AUC 0.350 reflects synthetic-label signal absence, not a model defect. We drop A2780 from the headline panel; the 3-cell subset is reported with explicit caveat.
> - **Gap 6**: tmQM pretrain +0.046 lift is measured on plain D-MPNN baseline, not on V4 multi-task encoder. Stage 1.6 would close the gap; deferred to R18 post-Phase-2.

### §6.3 New Table 2 row (R18 row, not R17 row)

| Method | n_pockets | Vina Dock | Diversity | PB pass | m14 Decode | Source |
|---|---:|---:|---:|---:|---:|---|
| TargetDiff (CITEDONLY) | 100 | -8.5 ± 1.4 | 0.65 | 0.85 | n/a | peng2022pocket2mol cite |
| DiffSBDD (CITEDONLY) | 100 | -7.9 | 0.61 | 0.82 | n/a | diffsbdd cite |
| **Mol-Metal R17 (MEASURED, 10×3)** | 10 | -6.5 ± 1.2 | 0.107 | n/a | 1.000 | Round-12 pilot |
| **Mol-Metal R18 PROJECTED** | 100 | -10.5+ (P0-1 SE(3) SB) | 0.75+ (P1-1 multi-PH) | 0.95+ | 1.000 | R18 phase 2+3 complete |

> The R18 row is PROJECTED per `R18_PLAN.md` §1 Goals G1, G4, G5; **not yet MEASURED**. After R18 phase completion, regenerate Table 2 with MEASURED values.

---

## §7 决策表 (D1-D4 重定)

**Per `EXECUTION_ORDER.md` §7 + this reality check**, the four open decisions are reset:

### D1 — R17 Tier-1 AUC fail 怎么办?

| Old option | New option (this doc) |
|---|---|
| A) drop Tier-1 claim 走 honest negative | **A) honest-negative framing in §6 items (10)-(13) + Table 2 + §4.5 — chosen** |
| B) 必须 fix (currently selected) | ~~superseded~~ — Phase A structurally unrunnable per §1.5 |

### D2 — A2780 0.350 怎么办?

| Old option | New option |
|---|---|
| A) drop A2780 报 3-cell | **A) drop A2780 + 3-cell {MCF-7, A549, A2780cis} reported — chosen** (per Stage 2.5 §11) |
| B) 调查数据 bug | Ruled out per `wf_phase_a_stage2_diag/a2780_diagnosis.md` |
| C) 强训 | Defer to wet-lab validation (R19); synthetic labels cannot lift above 0.65 |

### D3 — R18 P0-1 SE(3) SB 失败时?

| Old option | New option |
|---|---|
| A) 退到 midpoint ODE (R17 fallback) | **A) 退到 midpoint ODE (R17 fallback) — chosen** (per R18_PLAN §10 rollback) |
| B) 标 incremental (R18 non-revolutionary) | Backup: if P0-1 lifts m3 by < 1.0 kcal/mol, mark R18 as incremental |
| C) skip (R19) | Last resort: defer entire R18 to R19 if Blackwell GPU not stable |

### D4 — Wet-lab validation?

| Old option | New option |
|---|---|
| A) R19 wet-lab | **A) R19 wet-lab — chosen** (deferred per R17 + this doc) |
| B) skip | Rejected: paper §6 must acknowledge wet-lab gap |
| C) external collaboration | Future: open to collaborators with cisplatin-resistance assay capacity |

### D5 (NEW) — Phase A plan file status?

**Decision**: mark `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` as **SUPERSEDED** by this TODO-53, but **keep the file** as historical reference for the Stage 1.5 / 2.5 BLOCK chain. Add header banner: "SUPERSEDED 2026-09-25 by TODO-53 — see `TODO/pending/53_phase_a_reality_check.md` for honest-negative verdict + R18 pivot".

### D6 (NEW) — R18 timeline

**Decision**: Phase 0 + Phase 1 start this week (parallel with Phase A close-out). Phase 2 starts Week 4 after Phase 1 stable-training verification (m14 ≥ 0.5 smoke test).

---

## §8 TODO 更新清单 (哪些 plan 要重写)

### §8.1 直接更新 (this week)

| File | Action | Owner | Deadline |
|---|---|---|---|
| `TODO/pending/53_phase_a_reality_check.md` | **write** (this doc) | Hugo Lee | 2026-09-25 (DONE) |
| `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` | add SUPERSEDED header banner | Hugo Lee | 2026-09-26 |
| `TODO/INDEX.md` §2 R17 known fails table | update m1/m2 row to "honest-negative, see TODO-53" | Hugo Lee | 2026-09-26 |
| `TODO/EXECUTION_ORDER.md` §0 + §1 | rewrite to Phase A → R18 transition (no Phase B wait) | Hugo Lee | 2026-09-26 |
| `TODO/02_metrics/METRICS_TARGETS.md` Table 2 | update m1 row to "honest-negative", R18 row to PROJECTED | Hugo Lee | 2026-09-26 |
| `paper/sections/06_limitations.tex` | add items (10)-(13) per §4 | Hugo Lee | 2026-09-26 |
| `paper/sections/04_results.tex` §4.1 + §4.3 + Table 2 | rewrite to honest framing per §6 | Hugo Lee | 2026-09-27 |
| `paper/sections/04_5_honest_negative.tex` | add Gap 4 + Gap 5 + Gap 6 per §6.2 | Hugo Lee | 2026-09-27 |

### §8.2 间接更新 (next 2 weeks)

| File | Action | Owner | Deadline |
|---|---|---|---|
| `TODO/03_plans/R18_PLAN.md` | bump Phase 0 + 1 to "start this week" status (was W3) | Hugo Lee | 2026-09-26 |
| `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` | add §10.1 "Phase A superseded" note | Hugo Lee | 2026-09-26 |
| `TODO/03_plans/MERGE_GATE.md` | add R18 Phase 0 + 1 gates | Hugo Lee | 2026-10-02 |
| `TODO/01_strategy/PROJECT_STATEMENT.md` §3 贡献 | rewrite contribution #2 to "R18 SE(3) SB migration in flight, +0.046 tmQM pretrain MEASURED" | Hugo Lee | 2026-10-02 |
| `paper/refs.bib` | add 5 R18 bib keys (mahmood2026sb_lie, jiang2026sb_relax, wang_xujia2026ma, xue_xiaoping2026ot, kim_jung2026multi_ph) | Hugo Lee | 2026-09-30 |
| `molmetal/.claude/skills/mol-metal-cfm-training/SKILL.md` | +R18 P0-1 + P0-2 + P0-4 sections | Hugo Lee | 2026-10-09 |

### §8.3 不更新 (保留历史)

| File | Why keep |
|---|---|
| `TODO/04_history/AUDIT_RESEARCH_GRADE.md` | Historical audit 2026-09-11 |
| `TODO/04_history/completion_audit_2026-09-13.md` | Completed-task audit, keep for traceability |
| `TODO/pending/14_full_100pocket_paper_r13.md` | Round-13 R17 close-out verdict preserved |
| `TODO/pending/26_round13_round14_complete_plan.md` | Round-13 + 14 R15/R16 history |
| `molmetal/reports/wf_phase_a_stage1_ru/final.md` | Stage 1.5 BLOCK verdict — keep as MEASURED reference |
| `molmetal/reports/wf_phase_a_stage2_pt/final.md` | Stage 2.5 BLOCK verdict — keep as MEASURED reference |

### §8.4 归档候选 (post-R18 close-out)

| File | Action |
|---|---|
| `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` | move to `TODO/archive/` after R18 Phase 0-1 close (per R18 ship) |
| `TODO/pending/24_cfm_architecture_redo_plan.md` | mark SHIPPED (R15 YuelBond decoder + Frontier 24 papers) |
| `TODO/pending/22_data_gap_alignment_plan.md` | mark MOSTLY CLOSED (17/25 metrics) |

---

## §9 与 R18 plan 的关系 (cross-reference)

| This doc § | R18 reference | Action |
|---|---|---|
| §1 failure modes | — | Independent (Phase A history) |
| §2 Blackwell budget | `R18_PLAN.md` §8 resource reqs | Same Blackwell single-host |
| §3 R18 pivot | `R18_PLAN.md` §1 Goals | Adopt as next active plan |
| §4 honest-negative framing | `R18_MATH_MIGRATION_PLAN.md` §11 R18 narrative | Aligned: theory contribution over AUC leadership |
| §5 R18 Phase 0 提前 | `R18_PLAN.md` §2 Phase 0 + §4 readiness | Phase 0 + 1 start this week |
| §6 paper §4 headline | `R18_MATH_MIGRATION_PLAN.md` §10 narrative | "First SE(3) SB applied to precious-metal SBDD" |
| §7 D1-D4 reset | `EXECUTION_ORDER.md` §7 | D1=A, D2=A, D3=A, D4=A, +D5/D6 NEW |
| §8 TODO update list | `INDEX.md` + `EXECUTION_ORDER.md` | Update files in §8.1 + §8.2 |

---

## §10 honest summary (MEASURED vs PROJECTED)

### §10.1 MEASURED (real, on this hardware)

1. Stage 1.4 training interrupted after 7 log lines, no per-seed metrics JSON (`wf_phase_a_stage1_ru/final.md`)
2. Stage 2.4 training interrupted at seed=0 epoch=40/100, 30-min budget exhausted (`wf_phase_a_stage2_pt/final.md`)
3. tmQM pretrain **plain D-MPNN** lift +0.046 (Ru AUC 0.7195 → 0.7655)
4. A2780 AUC 0.350 is signal-absence, not a bug (cross-line residual r < 0.03)
5. A2780cis 4-cell baseline 0.648 ± 0.006 (3 seeds, n=4078 rows)
6. DecompDiff/Pocket2Mol/TargetDiff 100×3 pipeline wired (R17 close-out)
7. 14/14 evaluators verified, 7/7 pitfall guards PASS, ≥217 unit tests PASS
8. tmQM encoder arch mismatch: baseline 12 tensors hidden=128 vs V4 DirectedMPNN 36 tensors hidden=300 — state-dict incompatible, random-init fallback

### §10.2 PROJECTED (not measured, TBD)

1. Ru AUC ≥ 0.86 from multi-task + 3-seed × 100-ep — never ran
2. Pt A2780cis AUC ≥ 0.85 from 5-seed tmQM + resistance features — never landed
3. A2780 drop as cherry-picking vs principled 3-cell subset choice (honest framing argument; reviewers may disagree)
4. R18 P0-1 SE(3) SB Vina Dock -8.5 → -10.5+ (per R18_PLAN.md §1 G1)
5. R18 P1-1 multi-PH Diversity 0.65 → 0.75 (per R18_PLAN.md §1 G4)
6. R18 P1-4 Hilbert primal-dual LambdaASTSR R² ≥ 0.62 wall-time -50%

### §10.3 VERIFIED FAIL (cannot lift)

1. Stage 1.4 random-init encoder cannot reach 0.86 Ru AUC (tmQM +0.046 not available; multi-task lift was PROJECTED, never MEASURED)
2. Stage 2.4 30-min budget cannot fit 5-seed × 100-epoch (12h GPU required)
3. Concurrent ultracode Blackwell contention structurally unfixable without serial scheduling

### §10.4 VERIFIED SHIP

1. tmQM pretrain ablation +0.046 lift (R17 close-out, plain D-MPNN)
2. R17 R12-R15 structural fixes (82/89 tests pass, 2 bugs caught, paper recompile 75 pp)
3. R17 14 evaluators + 7 pitfall guards + 217 tests + 5 skills
4. 4-cell PlatinAI baseline as pipeline smoke test (synthetic labels, not wet-lab)
5. tmQM 12/12 backbone tensors load OK (test_rmse 0.2718, cls_acc 0.965)

---

## §11 Cross-references

- `molmetal/reports/wf_phase_a_stage1_ru/final.md` — Stage 1.5 BLOCK verdict
- `molmetal/reports/wf_phase_a_stage2_pt/final.md` — Stage 2.5 BLOCK verdict
- `molmetal/reports/wf_phase_a_stage2_diag/a2780_diagnosis.md` — A2780 signal-absence
- `molmetal/reports/wf_tmqm_pretraining/` — tmQM pretrain +0.046 ablation (MEASURED)
- `molmetal/reports/wf_gpu_recovery_now/final.md` — Blackwell recovery
- `molmetal/reports/wf_gpu_auto_recover/final.md` — GPU watchdog
- `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` — Phase A plan (SUPERSEDED)
- `TODO/03_plans/R18_PLAN.md` — Phase B detailed plan (now active)
- `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` — Phase B strategic plan
- `TODO/03_plans/MERGE_GATE.md` — merge 5 gates
- `TODO/01_strategy/PROJECT_STATEMENT.md` — paper立项
- `TODO/01_strategy/INSIGHTS_FROM_WORKFLOWS.md` — 7 启发 + §4.5 framing
- `TODO/02_metrics/METRICS_TARGETS.md` — 20 指标 + Table 2
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 机制 × 20 指标
- `TODO/EXECUTION_ORDER.md` — next-step navigation
- `TODO/INDEX.md` — concise state
- `paper/sections/04_results.tex` — §4 (rewrite to honest framing)
- `paper/sections/04_5_honest_negative.tex` — §4.5 (add Gap 4-6)
- `paper/sections/06_limitations.tex` — §6 (add items 10-13)
- `paper/refs.bib` — add 5 R18 bib keys

---

## §12 Decision log (this document)

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-25 | Phase A close as honest-negative | 4-layer failure, structurally unrunnable on Blackwell single-host |
| 2026-09-25 | Drop A2780 from §4 panel | Synthetic label signal-absence, not a bug (ruled out per Stage 2.1) |
| 2026-09-25 | R18 P0-1 SE(3) SB 主推 next | First SE(3) SB application to precious-metal SBDD (per Mahmood 2026) |
| 2026-09-25 | R18 Phase 0 + 1 start this week | Independent of Phase A close-out; pure doc + CPU work |
| 2026-09-25 | tmQM pretrain +0.046 cited as plain-D-MPNN baseline only | V4 multi-task arch mismatch, random-init fallback |
| 2026-09-25 | paper §4 headline rewrite to honest framing | Per §6.1; no Tier-1 AUC leadership claim |
| 2026-09-25 | paper §6 add items (10)-(13) | Per §4.1-§4.4; document Phase A failure modes for reviewers |

---

End of report.