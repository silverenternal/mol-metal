# STR-1 — 五 seed OOF 与 seed 平均

**日期**：2026-10-19
**状态**：DONE / STABLE（gate 通过）
**脚本**：`molmetal/scripts/msrs_seed_ensemble.py`
**测试**：`molmetal/tests/test_msrs_seed_ensemble.py`（9 个，全过）
**产物**：`metrics/by_metric/msrs_seed_ensemble.json` + `results/msrs_seed_ensemble/oof_seed{0..4}.npz`
**输入**：`msrs_generalization.py --seed 0..4`（四协议 × 5 seed = 20 次树模型，30m54s）

## 为什么做

MET-3（Δ）、MET-4（校准）、MET-5（决策）全部消费同一个单 seed OOF
（`results/msrs_generalization/oof.npz`，seed 20261018）。MET-1 实测 blend 的
seed SD 是 0.0102，所以「Δ 符号 0.6193」「regret 差 −0.0034」这些第二位数
可能只是 seed 抽签。STR-1 在谈「Δ 不可学」之前先把这一项排掉。

## 做法

- 同一份 21,404 行、同一组 fold（`scheme_folds`），只换 LightGBM 的
  `random_state`：seed 0/1/2/3/4。
- 每个 seed 的 OOF 单独存盘；聚合时先校验 `y` 与四协议 `fold` 逐行相同，
  再对预测取 `nanmean`。
- 在 seed 平均预测上重算 Δ（4 个阈值 + DOI 聚类 bootstrap）、conformal
  （global / metal Mondrian / difficulty）、决策曲线（全覆盖 + q=0.5）。
- 同时报每个 seed 的 r2 / mae / sign@0.2091 / MAE(Δ)，给出 mean / sd /
  min / max，用来判断「seed 噪声是否主导」。
- gate 写死在看到结果之前：`max(sign seed SD) > 0.05` → `SEED_NOISE_DOMINATED`。

## MEASURED

### 主指标：seed 平均 vs 已发布单 seed

| 协议 | R²（5-seed 均值预测） | R² seed mean ± sd | ΔR² vs 单 seed | Δ 符号（均值预测） | Δ 符号 seed mean ± sd | Δ 符号 vs 单 seed | MAE(Δ) | MAE(Δ) seed sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ligand | **0.5596** | 0.5555 ± 0.0026 | **+0.0040** | **0.6101** | 0.6110 ± 0.0179 | −0.0046 | 0.4999 | 0.0014 |
| doi | **0.3380** | 0.3322 ± 0.0029 | **+0.0059** | **0.5826** | 0.5670 ± 0.0332 | **−0.0367** | 0.5115 | 0.0055 |
| scaffold | **0.4530** | 0.4475 ± 0.0028 | **+0.0027** | **0.6330** | 0.6027 ± 0.0387 | −0.0092 | 0.5032 | 0.0072 |
| year | **0.3780** | 0.3728 ± 0.0066 | **+0.0041** | **0.4082** | 0.4245 ± 0.0303 | −0.0204 | 0.5281 | 0.0091 |

单 seed 参考（已发布）：ligand R² 0.5556 / 符号 0.6147；doi 0.3321 / 0.6193；
scaffold 0.4503 / 0.6422；year 0.3739 / 0.4286。

### 校准与决策（seed 平均预测）

| 协议 | coverage | 中位宽 | Winkler | PIT KS | regret（全覆盖） | regret（q=0.5） | p_holm |
|---|---:|---:|---:|---:|---:|---:|---:|
| ligand | 0.9003 | 1.6459 | 2.2335 | 0.0011 | 0.1193 | 0.1247 | 1.0 |
| doi | 0.9001 | 2.0040 | 2.7273 | 0.0011 | 0.1279 | 0.1072 | 1.0 |
| scaffold | 0.9000 | 1.8297 | 2.4596 | 0.0013 | 0.1126 | 0.1115 | 1.0 |
| year | 0.9003 | 2.0350 | 2.7395 | 0.0022 | 0.1613 | 0.1577 | 1.0 |

单 seed 参考：doi 中位宽 2.0052、全覆盖 regret 0.1168；ligand 1.6515 / 0.1191。

### 判定

`max_sign_seed_sd = 0.0387`（scaffold）< 0.05 → **STABLE**。

## 诚实读数

1. **gate 过了，但「STABLE」不等于「单 seed 数字可信」。** DOI 的 Δ 符号
   单 seed 是 **0.6193**，五个 seed 的实际范围是 0.5183–0.5963——单 seed
   那个值比**所有五个 seed 都高**，比 seed 均值 0.5670 高 1.6 SD。seed 平均
   预测给出 0.5826。所以 0.6193 是一次偏高的抽样，不是可复现的中心值。
   主表应改用 0.5826（或 0.567 ± 0.033），不能继续引用 0.6193。
2. **seed 平均在四个协议上一致抬高 R²，幅度 +0.003～+0.006。** 这不是噪声
   翻转，而是方差缩减：四个协议同向、且都不小于单 seed。R² 的 seed SD 只有
   0.0026–0.0066，远小于 STR-2 要检测的 +0.01–0.03，所以 5 个 seed 足够
   分辨融合效应，不需要再加 seed。
3. **Δ 的 MAE 基本不动。** ligand/doi/scaffold 变化 ≤0.003；year 从 0.5484
   降到 0.5281（−0.0203），是四个协议里唯一实质改善的——year 只有 6,499 行
   被评估、110 个配对，seed 噪声本来就最大（SD 0.0091）。
4. **校准完全不是 seed 问题。** DOI 中位宽 2.0040 vs 单 seed 2.0052，coverage
   0.9001 vs 0.9004。MET-4 的「conformal 宽 ≈ 高斯宽、difficulty 无益」结论
   不受 seed 影响。
5. **决策层的 null 反而更硬了。** DOI 全覆盖 regret 从 0.1168 变成 0.1279
   （+0.0111），q=0.5 从 0.1067 变成 0.1072，p_holm 仍是 1.0。也就是说
   单 seed 的 0.1168 已经是偏乐观的一次抽样；MET-5 的「决策层没有增益」
   在 seed 平均下没有变弱。
6. **year 的符号 0.4082 低于 0.5。** 110 个配对、5 seed SD 0.0303——这个协议
   的配对样本量根本不足以支撑符号主张，任何 year 符号数字都只能当描述。
7. **SD 本身也不精确。** 5 个 seed 估出的 SD 相对误差约 `1/√(2(n−1)) = 35%`。
   scaffold 的 0.0387 可能是 0.025，也可能是 0.052——gate 的「通过」在边界
   附近，不该被读成「seed 噪声很小」，只能读成「没有证据说它主导」。
8. **这次没有翻正任何负结果。** 四个协议的 Δ 符号仍然全部 <0.65（最高
   scaffold 0.6330），DOI R² 仍然 0.3380 < 0.36，regret 差仍然 p=1.0。
   STR-1 的作用是把这些数字的误差棒钉住，不是改善它们。

## 对主表的影响

- DOI 行改为：R² **0.3380**（5-seed 均值预测；单 seed 0.3321），Δ 符号
  **0.5826**（单 seed 0.6193，seed 均值 0.567 ± 0.033）。
- 所有「第二位数」的表格必须注明是 5-seed 均值，不能再写 seed 20261018。
- 决策 regret 用 0.1279（全覆盖）/ 0.1072（q=0.5），MET-5 的 null 结论不变。
- 校准数字可以继续用单 seed（差异在小数第三位），但表里统一换成 STR-1 的
  数值更省事。

## 复现

```bash
for s in 0 1 2 3 4; do
  .venv/bin/python molmetal/scripts/msrs_generalization.py --seed $s \
    --out /tmp/msrs_gen/seed$s.json \
    --oof-out results/msrs_seed_ensemble/oof_seed$s.npz
done
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_seed_ensemble.py \
  --dir results/msrs_seed_ensemble
OMP_NUM_THREADS=1 .venv/bin/python -m pytest molmetal/tests/test_msrs_seed_ensemble.py -q
```
