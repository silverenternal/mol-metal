# Plan A · Step 2 — CFM prior 替换 SymbolicPrior

**Date:** 2026-10-07 · **GPU:** ~10 hr (CFM prior head fine-tune) · **LOC:** ~100 + fine-tune · **Wall:** 1 wk · **Status:** DESIGN, gated on Step 1 + cfm_r10 fix completion

## §0 为什么这是第二步 (不是第一步)

Plan A overview §2.2 把 Step 2 排在 Step 1 之后,因为:
- **依赖 cfm_r10 σ floor fix** 完成 (fm_noise_sigma 2.505 → 0.5,否则 prior 输出是噪声)
- **依赖 R19 cosine+AdamW** 完成 (否则 fine-tune 不稳定)
- **Step 1 先验证 V(s) 通道** — 知道 DMPNN value 接入是否真的 lift metric 后,Step 2 才有信心接入 prior

**关键差异**:Step 1 是 V(s) 通道(learned value 替换 hand-crafted value),Step 2 是 P(a|s) 通道(learned prior 替换 hand-crafted prior)。前者先验证 learned signal 在 MCTS 里 work,后者才进入"prior + value 双 learned"。

## §1 现状 (MEASURED, 2026-10-07)

当前 SymbolicPrior 在 `proof_search.py` 里 (待 grep 精确位置):
```python
class SymbolicPrior:
    """Hand-crafted tile similarity prior."""
    def score(self, state: MoleculeClosedTerm, action: Action) -> float:
        # 基于子结构匹配 / Tanimoto similarity / SMARTS 反应性
        # 完全 hand-crafted,无 learned 参数
```

**实测**:bench `n_distinct` 在 top_k=10 cap 处饱和 → 当前 prior 已经把搜索带到"找 top-10 distinct"的水平。

**短板**:prior 给的 ranking 不知道哪个 reaction 真能在化学上 lead to high-pIC50 product。完全依赖 Lipinski + binding + PLL 在 rollout 阶段打补丁。

## §2 Step 2 目标 — CFM 学出 P(a|s)

### §2.1 设计

CFM 在 reaction-tree 上 fine-tune 一个 prior head:

```
                ┌──────────────────────────────────┐
                │  CFM (existing checkpoint)        │
                │  flow matching x_0 → x_1         │
                └──────────────┬───────────────────┘
                               │ freeze most params
                               ▼
                ┌──────────────────────────────────┐
                │  NEW prior head (~50 LOC)         │
                │  input: state latent + action id  │
                │  output: log P(action | state)    │
                └──────────────────────────────────┘
```

**训练数据**:reaction path leaves 累积
- 来源 1: PlatinAI 4-cell reaction graphs (已有,待提取)
- 来源 2: USPTO reaction SMILES (公开,大)
- 来源 3: Lambda MCTS 自身的 leaf_history (Step 1 跑完后开始累计)

**Fine-tune 范围**:
- CFM 主干 **freeze** (保 Wasserstein-trained features)
- 只训 prior head (~10K params)
- 数据量 ~10K reaction steps → 1-2 epoch → ~2 hr GPU on RTX 5090

### §2.2 文件清单

| 文件 | 操作 | LOC |
|---|---|---|
| `molmetal/molmetal_lam/search_alg/cfm_prior.py` | NEW SCAFFOLD | ~120 |
| `molmetal/scripts/train_cfm_prior_head.py` | NEW training entry | ~150 |
| `configs/cfm_prior_head.yaml` | NEW config | ~30 |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | EDIT SymbolicPrior 接入点 (默认仍 SymbolicPrior) | ~30 |
| `molmetal/scripts/bench_mcts_search_quality.py` | ADD `--cfm-prior` flag | ~20 |
| `molmetal/tests/test_cfm_prior.py` | NEW | ~60 |
| `molmetal/tests/test_step2_cfm_wire.py` | NEW smoke | ~40 |

**总计 ~450 LOC, 7 文件**

### §2.3 Bench A/B 设计

**A** (control): `--no-cfm-prior` (SymbolicPrior)
**B** (Step 2): `--cfm-prior` (CFM trained prior head)

**MEASURED delta 关注**:
- `prior_entropy`: 应该 ↑ (learned prior 比 hand-crafted 更 contrast)
- `puct_exploit_ratio_var`: 应该 ↑ (prior + value 双 learned)
- `sims_to_threshold` (S2T, 需新 instrument):应该 ↓ (prior 更好 → 收敛更快)
- `n_success` / `mean_n_distinct`: 已 saturated,关注 best_score / mean_score
- `wall_time_s`: 应该 ↑ (~10-20% overhead,prior head forward)

## §3 实施顺序 (gated on Step 1 + cfm_r10 fix)

1. **Step 1 完工 + MEASURED delta 出表** (per plan_a_step1 §5)
2. **cfm_r10 σ floor fix 完工** (commit expected per cfm_r10_fix_plan.md)
3. **R19 cosine+AdamW 完工** (commit expected per robustness_r19_fix_plan.md)
4. **(1 day)** 写 CFM prior head SCAFFOLD + train script
5. **(1 day)** 数据 prep — 累计 Lambda MCTS leaf_history + PlatinAI 4-cell reaction graphs 提取
6. **(2 hr GPU)** 跑 fine-tune 1-2 epoch on RTX 5090
7. **(15 min)** smoke gating per §5.7
8. **(20 min CPU)** 跑 bench A/B → MEASURED delta 出表
9. **(10 min)** Commit + chat closeout (no doc files)

**Total: ~3 days wall + 2 hr GPU**

## §4 Step 2 完成判据

✅ All of:
1. CFM prior head fine-tune 完,checkpoint on disk
2. `--cfm-prior` smoke PASS (50-step + decode_ratio 替代物)
3. Bench A/B JSON on disk (`molmetal/reports/r24_step2_cfm_prior/`)
4. `prior_entropy` ↑ (learned prior 比 hand-crafted 更 contrast)
5. MEASURED delta 表更新到 `plan_a_step2_cfm_prior.md` §5
6. Commit landed (per §5.5)

## §5 MEASURED delta 表 (待 Step 2 跑完填)

| 指标 | baseline (SymbolicPrior) | Step 2 (CFM prior) | Δ |
|---|---:|---:|---:|
| prior_entropy | TBD | TBD | — |
| puct_exploit_ratio_var | TBD (Step 1) | TBD | — |
| sims_to_threshold (S2T) | TBD (new) | TBD | — |
| best_score | TBD (Step 1) | TBD | — |
| n_success | 24/27 | TBD | — |
| mean_n_distinct | 8.67 | TBD | — |
| mean_wall_time_s | 22.8 | TBD | — |
| GPU wall (fine-tune) | n/a | TBD hr | — |

## §6 风险 + 缓解

⚠️ **prior head 数据不足**:
- Lambda MCTS leaf_history 起步小 (~24 × 100 × 5 seeds = 12K samples)
- 缓解:用 USPTO 公开数据 pretrain (~1M reactions),再 fine-tune on leaf_history

⚠️ **fine-tune 过拟合**:
- 缓解:hold-out 10% PlatinAI 4-cell reaction graphs 验 OOD

⚠️ **prior + value 双 learned 可能震荡**:
- PUCT 公式 `Q + c·P·sqrt(N)/(1+n)` 里 P 和 Q 都在变,可能不稳定
- 缓解:prior 权重从 0.1 开始 warm-up,Step 2 启动时先验证

⚠️ **cfm_r10 / R19 fix 未完工 → Step 2 不能启动**:
- 这是 hard gate
- 缓解:Step 1 跑完后,我先把 cfm_r10 σ fix 派给 agent,~1 day 完成

## §7 跨引用

- `plan_a_overview.md` §2.2
- `plan_a_step1_dmpnn_value.md` (Step 1 必须先完工,验 V(s) 通道)
- `plan_a_data_pipeline.md` (反应数据准备细节)
- TODO/pending/cfm_r10_fix_plan.md → completed/ (PRECURSOR,fix 仍需做)
- TODO/pending/robustness_r19_fix_plan.md → completed/ (PRECURSOR,fix 仍需做)
- AGENTS.md §5.7 (smoke gating)
