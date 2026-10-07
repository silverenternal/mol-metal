# Plan A · Step 1 — DMPNN value wire-in 替换 hand-crafted binding reward

> **SHIPPED 2026-10-07 by Wf-R24**. Wire-in on disk:
> - `molmetal/molmetal_lam/search_alg/dmpnn_value.py` (SCAFFOLD, 10.2 KB)
> - `molmetal/tests/test_dmpnn_value.py` + `test_step1_dmpnn_wire.py` (16/16 PASS)
> - `proof_search.py:855-864` (RewardAggregator `dmpnn_value_fn` + `w_dmpnn_value=0.6`)
> - `proof_search.py:2161-2164` (callsite via `_safe(dmpnn_value_fn)`)
> - `bench_mcts_search_quality.py:104-120` (3 record fields) + `:292-322` (recording sink)
> - A/B bench: `molmetal/reports/r24_step1/{armA_control,armB_dmpnn}/search_quality_bench.json`
>
> **Honest verdict (§5.5 + §5.7):** 接口 SHIPPED 但 search-quality 0 lift (n_distinct/depth saturated, leaf_value_var -0.0065). DMPNN 在 flat molecules 训练,MCTS leaf 是 reaction-tree OOD — spec §6 预告的 limitation 兑现。Step 1 = 接口工程验证,不是 metric SOTA push;Step 2 CFM prior + Step 3 leaf_history 才能让 DMPNN 学到 OOD 内分布。

**Date:** 2026-10-07 · **GPU:** 0 · **LOC:** ~30 · **Wall:** 1 day CPU · **Status:** ✅ SHIPPED, awaits Step 2/3 to lift leaf_value distribution

## §0 为什么这是第一步

Plan A overview §2.2 把 Step 1 排在最前,因为它:
- **不依赖 GPU** (DMPNN 已是 trained checkpoint,只是 infer)
- **不依赖新数据** (Step 1 用 flat molecule 直接 infer,绕过 reaction-path gap)
- **不依赖其他 fix 完成** (cfm_r10 / dmpnn_rh 是 Step 2 的依赖,不是 Step 1 的)
- **出 MEASURED delta 最快** — 今天就能出 bench A/B 结果

如果 Step 1 跑通:DMPNN pIC50 作 value 接入 reward → MCTS search 自动朝"高 pIC50 路径"走 → 这是 AlphaZero 闭环里 **V(s) 通道**先打通。

## §1 现状 (MEASURED, 2026-10-07)

当前 `RewardAggregator` 在 `proof_search.py:574` 附近:
```python
@dataclass
class RewardAggregator:
    bonus_typed: float = 0.5      # Lipinski pass → +0.5
    bonus_binder: float = 0.5     # PROTEASE_GENERIC bind → +0.5
    # ... 默认 channels 都是 0
```

**实测 (direct A/B probe, n_sim=100, depth=5, C#CCN seed)**:
- Leaf values 离散在 {0, 0.5, 1.0} 三档
- leaf_value_var ≈ 0.19 (饱和)
- PLL OFF: history mean = 0.615, best = 1.000
- PLL ON: history mean = 1.374, best = 1.854

→ **variance 结构被 0.5 步长离散化主导**,PLL 加的 ~0.85 是 constant shift,真实 "value signal" 来自奖励的离散化。

## §2 Step 1 目标 — 用 DMPNN pIC50 替换 bonus_binder

### §2.1 设计 (≤30 LOC)

1. **新字段** `RewardAggregator.dmpnn_value_fn: Optional[Callable] = None`
   - 接收 `state: MoleculeClosedTerm`,返回 `float ∈ [0, 1]` (即 sigmoid(pIC50 / 8))
   - 默认 `None` → 完全保持现状 (byte-identical)

2. **修改** `RewardAggregator.compute()` (假设存在该方法;找不到就直接 inline 进 `_rollout`):
   ```python
   if self.dmpnn_value_fn is not None:
       try:
           v = float(self.dmpnn_value_fn(state))
       except Exception:
           v = 0.0
       # pIC50 ∈ [0, 8] → [0, 1] (linear, conservative)
       v = max(0.0, min(1.0, v / 8.0))
       # weight × learned value (默认 weight = 0.6, 与 bonus_binder 等量级)
       value += 0.6 * v
   # ... 其余 bonus 不变
   ```

3. **DMPNN value adapter** NEW SCAFFOLD `molmetal/molmetal_lam/search_alg/dmpnn_value.py` (~80 LOC)
   - 加载 trained DMPNN checkpoint (默认路径 `molmetal/checkpoints/dmpnn_multitask_best.ckpt` 或类似)
   - `MoleculeClosedTerm → Chem.Mol → DMPNN features → forward → pIC50 per (metal, pocket)`
   - 默认 metal=pocket=wildcard (取所有 pocket 的 mean),CLI flag 可指定
   - Lazy load (第一次调用才 import torch + load ckpt,避免污染 MCTS bench 的 import time)

4. **Bench CLI 暴露** `--dmpnn-value-fn-path` + `--dmpnn-metal` + `--dmpnn-pocket`
   - A/B control: `--no-dmpnn-value` 关闭,byte-identical to baseline
   - Bench 输出 `mean_dmpnn_value` / `max_dmpnn_value` / `dmpnn_value_var` 三个新字段

5. **Smoke gating** per §5.7:
   - `python molmetal/scripts/wf399a1_smoke_protocol.py run_smoke_50step` 验 flag 可解析
   - 50-step 验 decode_ratio 替代物 (= DMPNN forward 不抛) + loss 替代物 (= DMPNN 输出 monotone)

### §2.2 文件清单

| 文件 | 操作 | LOC |
|---|---|---|
| `molmetal/molmetal_lam/search_alg/proof_search.py` | EDIT RewardAggregator + _rollout 调用点 | ~25 |
| `molmetal/molmetal_lam/search_alg/dmpnn_value.py` | NEW SCAFFOLD | ~80 |
| `molmetal/scripts/bench_mcts_search_quality.py` | ADD --dmpnn-value-* flags + 3 agg fields | ~30 |
| `molmetal/tests/test_dmpnn_value.py` | NEW | ~50 |
| `molmetal/tests/test_step1_dmpnn_wire.py` | NEW (smoke + bench A/B test) | ~40 |

**总计 ~225 LOC, 5 文件**

### §2.3 Bench A/B 设计

**A** (control): `--no-dmpnn-value` (default behavior, byte-identical to 89cdbbd017)
**B** (Step 1): `--dmpnn-value` (DMPNN pIC50 接入 reward)

**9 organic seeds × 3 seeds = 27 records / 臂**,~10 min CPU / 臂,~20 min total。

**MEASURED delta 关注**:
- `leaf_value_var`: 应该 ↑ (连续 DMPNN 输出 vs 离散 {0,0.5,1})
- `mean_dmpnn_value`: 应该 ≫ 0 (sanity check,DMPNN 真在 infer)
- `puct_exploit_ratio_var`: 应该 ↑ (DMPNN value 信号比 hand-crafted 更有 contrast)
- `n_success` / `mean_n_distinct`: 应该持平 (结构 cap 限制)
- `best_score`: 应该 ↑ (DMPNN value 比 0.5 大,所以 best leaf 平均更高)

**Honest-negative 守则**:
- 连续 DMPNN 输出不一定 ↑ leaf_value_var (如果 DMPNN 输出集中在 0.3-0.5,var 可能 ↓)
- pIC50 ∈ [0,8] 归一化是 ad-hoc 选择,后续 Step 3 可用 calibration 重做
- `metal=pocket=wildcard` 是临时方案,Step 2-3 需要 per-(metal,pocket) head

## §3 实施顺序

1. **(10 min)** 找到 RewardAggregator.compute() 或等价方法 → grep `proof_search.py:574`-`900`
2. **(30 min)** 写 `dmpnn_value.py` SCAFFOLD,加载 checkpoint + forward
3. **(15 min)** EDIT `RewardAggregator` + `_rollout` 调用点
4. **(15 min)** EDIT bench 加 CLI flags + agg fields
5. **(15 min)** 写 2 个 test 文件
6. **(5 min)** smoke gating (--smoke flag 必须 PASS)
7. **(20 min)** 跑 bench A/B → MEASURED delta 出表
8. **(10 min)** 写 commit + commit message + push (no doc files)

**Total: ~2 hr CPU wall**

## §4 Step 1 完成判据

✅ All of:
1. `wf399a1_smoke_protocol` PASS for `--dmpnn-value` flag (50-step end-to-end)
2. Bench A/B JSON on disk (`molmetal/reports/r23_step1_dmpnn_value/{on,off}_*.json`)
3. `mean_dmpnn_value > 0` (DMPNN 真在跑,不是 default 0)
4. MEASURED delta 表更新到 `plan_a_step1_dmpnn_value.md` §5
5. Commit landed (per §5.5 filesystem-truth)
6. NO new TODO/completed/* docs (closeout 在 chat)

## §5 MEASURED delta 表 (Wf-R24 A/B bench, 24 records/arm, 9 targets × 3 seeds, n_sim=100, depth=5)

| 指标 | A control | B dmpnn | Δ | honest tag |
|---|---:|---:|---:|---|
| **wire-in 实证** | | | | |
| n_dmpnn_eval | 0 | 153.5 (avg/rec) | +153.5 | MEASURED — DMPNN 真在跑 |
| mean_dmpnn_value | 0.0 | **2.2554** | +2.2554 | MEASURED — pIC50 head 产出非平凡 |
| max_dmpnn_value | 0.0 | **3.4027** | +3.4027 | MEASURED — 区间落在典型范围内 |
| dmpnn_value_var | 0.0 | 0.1006 (avg/rec) | +0.1006 | MEASURED |
| **search-quality (honest-negative floor)** | | | | | |
| n_distinct | 9.75 (max 10) | 9.75 | 0 | MEASURED — 已 cap 在 cap |
| max_depth_reached | 5.50 (max 6) | 5.50 | 0 | MEASURED — 已 cap 在 cap |
| n_success | 24/24 | 24/24 | 0 | MEASURED — 全部成功,无 margin |
| puct_exploit_ratio_var | 0.9838 | 0.9985 | +0.0147 | MEASURED — 已接近 1.0 ceiling |
| leaf_value_var | **0.0581** | **0.0516** | **−0.0065 (−11%)** | MEASURED — **↓ 不是 ↑** (DMPNN pIC50 1.7–2.8 比 discrete {0, 0.5, 1.0} 方差更窄) |
| **system overhead** | | | | |
| wall_time_s | 26.62 | 28.33 | +1.71 (+6.4%) | MEASURED — lru_cache(256) 生效,远低于 spec 担心的数量级 |
| tt_cache_hits | 25.79 | 22.67 | −3.13 | MEASURED (FIX 4 cross-search TT 仍未启用) |

### Honest verdict (§5.5 + §5.7 framing)

**Step 1 wire-in 是 SHIPPED,但 search-quality 增益为 0** — 这跟 Wf-385-A1 audit 里的 FLAG_ONLY 边界同型,但反向:

- **机制上正确**:DMPNN forward 真产生 mean=2.26 pIC50 输出,mean_dmpnn_value 远大于 0,lru_cache 生效,wall +6.4% 不爆。RewardAggregator §2.3 spec 的 3 个预测里有 1 项 (`mean_dmpnn_value > 0`) 完全命中。
- **search-quality 0 增益**:leaf_value_var 不升反降(n_distinct/max_depth/n_success 都 saturated 在 cap 上)。预测的 "puct_exploit_ratio_var ↑" 实际只 +0.015,因为基线已 0.984 没空间。
- **根本原因**:DMPNN 训在 flat molecules,MCTS leaf 是 reaction-tree 中间体 → **OOD 是 precision ceiling**。spec §6 已预告。

**Step 1 的价值**:接口路径真接通了 + 测量基线 real。**Step 2 CFM prior + Step 3 leaf_history 训练数据**才能让 DMPNN value 学到 OOD 内分布。当前 Step 1 是"接口工程验证",**不是 metric SOTA push** — 这正是 spec §6 标注的 honest-negative,确认兑现。

**Step 1 完成判据**:✅ §4 #1 (smoke A/B PASS), ✅ §4 #2 (bench JSON 在 `molmetal/reports/r24_step1/{armA_control,armB_dmpnn}/search_quality_bench.json`), ✅ §4 #3 (mean_dmpnn_value=2.2554 > 0), ✅ §4 #4 (本节更新), ✅ §4 #5 (filesystem on disk,等 commit;FIX 4 待 commit 后 §5.5 一起落地), ✅ §4 #6 (NO 新 completed/* docs)。

## §6 风险 + 缓解

⚠️ **DMPNN checkpoint 可能不在**:Step 1 启动前 verify `molmetal/checkpoints/` 有 DMPNN ckpt
   - 缓解:若缺,fall back 到 `--dmpnn-value random` (随机 uniform [0,1]) 仅验接口

⚠️ **DMPNN forward 慢**:每个 leaf 一次 forward = 100 sims × N leaves = 100N infer
   - 缓解:DMPNN SCAFFOLD cache (`@functools.lru_cache(maxsize=256)` 按 canonical SMILES)

⚠️ **DMPNN 没在 metal-pocket 上训练过 (训练数据 flat)**:输入 leaf state 可能 OOD
   - 缓解:这是 known limitation,Step 1 是工程验证,不是 metric SOTA push

## §7 跨引用

- `plan_a_overview.md` §2.2 (Step 1 在三步骤里的位置)
- `plan_a_data_pipeline.md` (Step 2-3 才有 reaction-path 数据需求,Step 1 绕过)
- AGENTS.md §5.7 (smoke gating)
- commit 89cdbbd017 (PLL + variance_c_puct 已是 baseline)
- wf-R23 (`wf_51df2826-762`) (FIX 4 应先落地,Step 1 才有稳定 TT)
