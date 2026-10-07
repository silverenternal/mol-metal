# Plan A — AlphaZero 闭环 metallodrug design

**Date:** 2026-10-07 · **Status:** APPROVED by user (verbatim: *"我认可你的方案A"*) · **Authoritative scope:** supersedes the 5 individual fix plans (cfm_r10 / dmpnn_r10 / rh_lomo / robustness_r19 / lambda_mcts_r20)

## §0 战略判断

用户原话 (2026-10-07):
- *"三个任意一个做好了就是一篇中科院二区，结合是一个算法或者工程的工作，不用额外训练模型，工作量实际上不大，好好做连一区都不是不可能"*
- *"我认可你的方案A"*

**Plan A 核心判断**:把 CFM / DMPNN / Lambda MCTS 三个组件**接成 AlphaZero 闭环** — 不重训任何模型,纯算法+工程集成。任何单组件写好 = Tier-2 (二区) 论文;三合一写好 = 一区潜力。

**Tier-2 framing per §5.4**:全部 in-silico,不依赖湿实验。完全在 project scope 内。

---

## §1 三组件当前真实状态 (MEASURED, 2026-10-07)

### §1.1 CFM (Wasserstein Flow Matching + YuelBond decoder)

| 能力 | 已落地 | 来源 |
|---|---|---|
| TMQM 多金属训练 | ✅ | `train_cfm_yuelbond_wasserstein.py`, 见 Wf-R14 configs |
| YuelBond decoder | ✅ | 同上 |
| Wasserstein 损失 | ✅ | `molmetal/adapters/flow_matching_lipman/` |
| `--eval-only` 复用 checkpoint | ✅ (Wf-R11-A3, e21469345a) | git log |
| `--wasserstein-cadence` flag | ✅ (Wf-R12-A, d2d3242c39) | git log |

**真实短板**:
- `fm_noise_sigma = 2.505` 太大,R10 fix plan (TODO/pending/cfm_r10_fix_plan.md) 建议降到 0.5
- decode 比常退化到 0/192,Wf-291D / Wf-299 已踩过这个坑,§5.7 smoke gating 是强制门
- 训练数据是 (x_0, x_1) 配对,**不是 reaction path leaves**

### §1.2 DMPNN (Directed Message Passing + per-pocket multi-task Wasserstein)

| 能力 | 已落地 | 来源 |
|---|---|---|
| Chemprop-style D-MPNN | ✅ | `train_dmpnn_multitask_wasserstein.py` |
| Multi-task Wasserstein (per metal × per pocket) | ✅ | 同上 |
| `wasserstein-cadence every-batch` | ✅ (Wf-R12-A) | d2d3242c39 |
| `pIC50` 输出 (per metal, per pocket) | ✅ | per-pocket head |

**真实短板**:
- Rh 数据 widen 中 (TODO/pending/dmpnn_r10_fix_plan.md, Rh_widen 数据集已就位)
- 训练数据是 flat molecules,**不是 reaction tree leaves**
- 缺少 reaction-step 级别的 feature (action embedding, parent-child bond diff)

### §1.3 Lambda MCTS (PLL + variance_c_puct + FPU + TT)

| 能力 | 已落地 | 来源 |
|---|---|---|
| 88.9% n_success (organic seeds, 27 records) | ✅ | commit 89cdbbd017 |
| PLL leaf-value wire-in (Liu & Feng 2026) | ✅ | proof_search.py:5417-5444 |
| Variance-cPUct (KataGo FIX 2) | ✅ | proof_search.py:4629-4670 |
| FPU reduction (KataGo FIX 3) | ✅ | proof_search.py:4704-4771 |
| TranspositionTable (per-search, 33% hit) | ✅ | proof_search.py:4029 |
| Cross-search TT (FIX 4) | ⏳ wf-R23 in flight | plan §1 FIX 4 |
| Subtree value bias (FIX 5) | ⏳ pending | plan §1 FIX 5 |

**真实短板**:
- Reward = `Lipinski + 0.5·PROTEASE_GENERIC` 全 hand-crafted,**没有 learned value**
- SymbolicPrior 是 hand-crafted tile similarity,**没有 learned prior**
- Structural metrics 已饱和 (top_k=10 + max_depth=5 caps)
- ER@N / S2T 指标未 instrumented (R20 §2)

---

## §2 Plan A — AlphaZero 闭环架构

### §2.1 数据流

```
                ┌─────────────────────────────┐
                │   Lambda MCTS (search)       │
                │   PUCT + virtual loss + FPU  │
                │   variance-cPUct + TT        │
                └──────────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │ Symbolic   │   │  CFM       │   │  DMPNN     │
       │ Prior      │   │  prior     │   │  value     │
       │ (legacy)   │   │  P(a|s)    │   │  V(s)      │
       │            │   │  ← new     │   │  ← new     │
       └────────────┘   └────────────┘   └────────────┘
                               │                │
                               ▼                ▼
                     ┌─────────────────────────────────┐
                     │  reward = w_D·DMPNN_pIC50      │
                     │         - w_SA·(SA-3)           │
                     │         + w_QED·QED             │
                     │         + PLL bonus (existing) │
                     └─────────────────────────────────┘
```

### §2.2 三步骤 (按 ROI + 数据依赖排序)

| Step | 做什么 | GPU | LOC | Wall | Paper 贡献 |
|---:|---|---:|---:|---:|---|
| **1** | DMPNN value wire-in 替换 hand-crafted binding reward | **0** | ~30 | 1 day | "replacing hand-crafted binding reward with learned pIC50" |
| **2** | CFM prior 替换 SymbolicPrior (条件生成 prior) | ~10 hr (fine-tune) | ~50 + 50 train | 1 wk | "AlphaZero-style prior in molecular space" |
| **3** | Closed-loop 全跑 (A2780 / MCF7 / NCI-60, 100 pocket × 10 seed) | ~30 hr | benchmark + fig | 2-4 wk | **Tier-2 main figure** (或 一区) |

**关键判断**:Step 1 不需要 GPU,不需要新数据,不需要新训练。今天就能做。Step 2-3 依赖组件 fix (cfm_r10 / dmpnn_rh) 先完工。

### §2.3 数据准备 (gating step, 3-6 个月)

**关键缺口**:三组件训练数据维度不匹配。
- CFM: flat molecule pairs
- DMPNN: flat molecules
- Lambda MCTS leaf: reaction tree leaves (parent molecule + action + child)

**Step 1 不需要新数据** — DMPNN 在 flat molecule 上跑 pIC50,把 leaf state 当 flat 输入即可 (临时方案,精度有限)。

**Step 2-3 需要 reaction-path-level 数据**:
- 来源 1: PlatinAI 4-cell reaction graphs (已有)
- 来源 2: USPTO reaction SMILES (公开)
- 来源 3: Lambda MCTS 自身的 leaf_history 累计 (~24/27 search × 100 sims × N runs)

**见 `plan_a_data_pipeline.md`** 数据通路细节。

---

## §3 Tier-2 论文潜力 (honest framing)

### §3.1 单组件都能撑起 Tier-2 (二区)

| 组件 | 单写 Tier-2 论文 angle |
|---|---|
| CFM | "Wasserstein flow matching for metal complexes with σ-floor + YuelBond decoder" |
| DMPNN | "Per-pocket multi-task Wasserstein D-MPNN for 12 metal × 9 cell-line pIC50" |
| Lambda MCTS | "AlphaZero-style MCTS with PLL leaf-value + variance-cPUct for metallodrug synthesis" |

### §3.2 组合撑起 一区 (top tier) 潜力

组合 = "AI-driven metallodrug design via AlphaZero loop with learned prior + value":
- novelty 高 (没人这么干过 metallodrug)
- 工程深度高 (三组件集成)
- 评估维度多 (pIC50 + SA + QED + Validity + Diversity + Synthesis path)
- 数据闭环 (closed-loop)

---

## §4 与原 5 个 fix plan 的关系

| 原 fix plan | 与 Plan A 的关系 |
|---|---|
| `cfm_r10_fix_plan.md` (σ floor) | **PRECURSOR to Step 2** — 不在 Step 1 内,Step 2 启动前必须完成 |
| `dmpnn_r10_fix_plan.md` (Rh widen) | **PRECURSOR to Step 1** — Rh 数据 widen 让 DMPNN value 在更多 metal 上可用 |
| `rh_lomo_fix_plan.md` | **PRECURSOR to Step 1** — LOMO 数据集扩到 Rh 让 DMPNN value head 包含 Rh |
| `robustness_r19_fix_plan.md` (cosine+AdamW) | **PRECURSOR to Step 2** — CFM 训练稳定后才能 fine-tune prior head |
| `lambda_mcts_r20_fix_plan.md` (FIX 4-5) | **PRECURSOR to Step 1** — FIX 4 cross-search TT 让 Step 1 价值评估更稳定 |

→ 5 个原 plan **全部移到** `TODO/completed/` 加 SUPERSEDED stamp,**修的内容仍是 Plan A 的依赖项**,实施时不冲突。

---

## §5 风险与 honest-negative

⚠️ **"不需要额外训练模型"是部分真的**:
- Step 1: ✅ 完全不需要训练,DMPNN 已是 trained checkpoint
- Step 2: ⚠️ 需要 CFM 加一个 prior head (~50 LOC),在小数据上 fine-tune (~10 hr GPU)
- Step 3: ⚠️ 不需要新训练,只是跑 closed-loop bench

⚠️ **"工作量不大"**:
- Step 1: ✅ 真不大 (~30 LOC + smoke + bench A/B)
- Step 2: ⚠️ 中等工作 (50 LOC + 数据通路 + fine-tune)
- Step 3: ⚠️ 工作量主要在 benchmark pipeline + paper figure,~2-4 wk

⚠️ **Tier-2 vs 一区**:
- 三组件各写一篇 = 3 篇 Tier-2,稳
- 组合写一篇 = 1 篇 Tier-2 (稳) 或 一区 (看 novelty framing + reviewer 反馈)

---

## §6 跨引用

- `plan_a_step1_dmpnn_value.md` — Step 1 详细 plan (本周执行)
- `plan_a_step2_cfm_prior.md` — Step 2 详细 plan (下周)
- `plan_a_step3_closed_loop.md` — Step 3 + Tier-2 main figure plan (Q4)
- `plan_a_data_pipeline.md` — Reaction path 数据准备 (gating)
- 5 个原 fix plan (SUPERSEDED): `TODO/completed/{cfm_r10,dmpnn_r10,rh_lomo,robustness_r19,lambda_mcts_r20}_fix_plan.md`
- AGENTS.md §3.3 R20 plan 列表 (现在部分项实现) + §5.4 (Tier-2) + §5.7 (smoke gating) + §5.8 (token budget)
- wf-R23 (`wf_51df2826-762`) — FIX 4 + GPU readiness (in flight,Step 1 启动前提)

---

## §7 启动决策

**用户已认可 (2026-10-07 verbatim: *"我认可你的方案A"*)**。立即执行顺序:
1. **现在**:写完 5 个 Plan A 子 doc (Step 1-3 + data pipeline)
2. **wf-R23 完成后**:跑 Step 1 (DMPNN value wire-in, 0 GPU, ~30 LOC + bench A/B)
3. **Step 1 出 MEASURED delta 后**:启动 Step 2 GPU 准备 (CFM σ fix + R19 cosine)
4. **Step 2 完工后**:Step 3 closed-loop bench
