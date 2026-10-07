# Plan A · 数据通路 — reaction path leaves 数据准备

**Date:** 2026-10-07 · **GPU:** 0 · **LOC:** ~150 (data extraction scripts) · **Wall:** 3-6 months · **Status:** DESIGN, gating Step 2-3 quality

## §0 为什么数据通路是单独的 doc

Plan A overview §2.3 提到的关键缺口 — 三组件训练数据维度不匹配 — 不在 Step 1 范围,但 Step 2-3 的精度上限由它决定。

**核心问题**:
- CFM 训练数据 = flat molecule pairs (x_0, x_1)
- DMPNN 训练数据 = flat molecules with pIC50
- Lambda MCTS leaf = reaction tree leaf (parent + action + child + path_so_far)

→ **要 fine-tune CFM 当 prior,必须有 (parent_state, action, child_state, pIC50_target) 数据集。**

## §1 数据源 (3 个)

### §1.1 PlatinAI 4-cell reaction graphs

**现状**:`molmetal/data/platinai_4cell.csv` 已有,但 reaction graph 提取未做完 (Wf-49 Phase 3 broken,n_samples=0)。

**需要的修复**:
- 重新 extract reaction SMILES from PlatinAI 4-cell (mcf7 / a549 / a2780 / a2780cis)
- 每个 reaction = (substrate, conditions, product)
- pIC50 target per cell-line
- 预计 ~4 × ~500 = 2000 reactions

**当前负责**:wf-R49 / wf-49 后续 (per TODO/INDEX.md §17.3)

### §1.2 USPTO reaction SMILES (公开)

**来源**:USPTO 50K / USPTO-full,公开数据集 (https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873)

**提取流程**:
1. 下载 USPTO 50K (~50K reactions)
2. SMILES → RDKit → 标准化
3. 过滤:organic subset only (重原子 ≤ 30,不含金属)
4. 输出:CSV (substrate_smi, product_smi, reaction_class, conditions)

**预计**:
- 数据量:~30K organic reactions (过滤后)
- 时间:1 day CPU (extract + clean)
- 存储:~50 MB CSV

### §1.3 Lambda MCTS 自身的 leaf_history 累计

**来源**:跑 Lambda MCTS bench 时,每个 leaf 都自动 record (parent_state, action, child_state, reward)。

**Step 1 启动后开始累计**:
- Per bench run: 27 records × ~100 leaves/record = ~2700 leaves
- 100 runs (Step 1+2+3 累计) = ~270K leaves
- **这是最大的 in-house 数据源**

**Schema** (already in `proof_search.py:_leaf_value_history`):
```python
@dataclass
class LeafRecord:
    parent_smiles: str
    action_id: int
    child_smiles: str
    reward: float           # bonus_typed + bonus_binder + dmpnn_value
    pIC50_target: float     # NEW: per-run target
    metal: str              # NEW: per-run metal context
    pocket: str             # NEW: per-run pocket context
    depth: int
    timestamp: float
```

**改动**:~30 LOC 在 `_leaf_history.append()` 加 4 个 NEW 字段,默认 None → byte-identical to baseline。

## §2 数据通路 pipeline

```
PlatinAI 4-cell  ─────► ┐
USPTO 50K        ─────► ├──►  unified CSV  ────►  train_step2_prior
Lambda MCTS leaf ─────► ┘     (~270K rows)
                                  │
                                  ▼
                          hold-out split
                                  │
                                  ▼
                          test_platinai_4cell
                          (~10K rows, OOD)
```

## §3 文件清单

| 文件 | 操作 | LOC |
|---|---|---|
| `molmetal/data/extract_platinai_reactions.py` | NEW (extract reaction graphs from PlatinAI 4-cell) | ~80 |
| `molmetal/data/extract_uspto_organic.py` | NEW (download + clean USPTO 50K) | ~100 |
| `molmetal/data/merge_reaction_dataset.py` | NEW (merge 3 sources into unified schema) | ~60 |
| `molmetal/scripts/record_leaf_history.py` | EDIT `_leaf_history.append()` 加 4 NEW fields | ~30 |
| `molmetal/tests/test_data_pipeline.py` | NEW | ~50 |

**总计 ~320 LOC, 5 文件**

## §4 实施顺序 (可在 Step 1 启动**并行**进行)

1. **(parallel)** `extract_uspto_organic.py` — 1 day CPU
2. **(parallel)** `extract_platinai_reactions.py` — 1 day CPU (依赖 Wf-49 PlatinAI 4-cell 修复)
3. **(parallel)** `record_leaf_history.py` — 30 min,Step 1 启动时一起做
4. **(sequential)** `merge_reaction_dataset.py` — 等 1+2+3 完成后
5. **(gating)** Step 2 启动前必须完成 merge,有 unified CSV 才能 fine-tune CFM prior

## §5 数据集 quality gate

- `n_unique_reactions >= 10K` (Step 2 fine-tune 起步)
- `pIC50_target_distribution` 覆盖 [5, 9] (paper benchmark 范围)
- `metal_distribution`: Pt ≥ 30%, Ru ≥ 20%, Rh ≥ 10%, Au ≥ 10%, Ir ≥ 10%
- `hold-out OOD gap` < 0.5 pIC50 (不能 fine-tune 后 OOD 全崩)

如果 gate 不过:
- Step 2 fine-tune data → fallback 到 USPTO only (30K organic reactions)
- Step 3 benchmark → fallback 到 PlatinAI 4-cell only (no Rh)

## §6 风险 + 缓解

⚠️ **PlatinAI 4-cell 提取仍 broken (Wf-49 Phase 3)**
   - 缓解:Step 1 期间,先派 1 个 agent 修 PlatinAI 4-cell extract (~1 day CPU)

⚠️ **USPTO 50K 下载可能网络问题**
   - 缓解:有 mirror 备份 (https://github.com/connorcoley/rexgen_direct 等)

⚠️ **Lambda MCTS leaf_history schema 改动需要兼容旧 JSON**
   - 缓解:4 NEW fields 全 default None → 旧 JSON 读不报错

⚠️ **270K leaf 数据可能 imbalance** (某些 seed 出现频次高)
   - 缓解:`merge_reaction_dataset.py` 里 per-source 上限 cap

## §7 跨引用

- `plan_a_overview.md` §2.3 (数据缺口 overview)
- `plan_a_step2_cfm_prior.md` §3 (Step 2 启动前必须完成 merge)
- `plan_a_step3_closed_loop.md` §3 (Step 3 用 9 cell-line 数据)
- TODO/pending/dmpnn_r10_fix_plan.md → completed/ (PRECURSOR,Rh 数据让 metal distribution 平衡)
- TODO/pending/rh_lomo_fix_plan.md → completed/ (PRECURSOR,Rh LOMO 数据)
- Wf-49 后续 (PlatinAI 4-cell 修复)
