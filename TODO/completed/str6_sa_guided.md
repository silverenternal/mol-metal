# STR-6 — SA 引导解码（不可执行 / KILL）

**日期**：2026-10-20
**状态**：DONE / **KILL**（计划中的 A/B 无法执行；改用产量审计判定）
**脚本**：`molmetal/scripts/cfm_sa_yield_audit.py`
**测试**：`molmetal/tests/test_cfm_sa_yield_audit.py`（6 个，全过）
**产物**：`metrics/by_metric/cfm_sa_guided.json`

**更正（2026-10-31，Wf-A120）**：本文件原先把三个生成池合成一条 **0.67%** 的
SA≤4.5 产量线。Wf-A120 给 `metrics/by_metric/cfm_sa_guided.json` 补上了 scale 标签
与分层：`combined.scale_label = "mixed_cohort_scale"`、`combined.comparable = false`。
**这条 0.67% 是混尺度（mixed-cohort-scale）率，跨层不可比**——它的分母把
full_corpus（n_train = 26,715，1 池）与 small（n_train = 256，2 池）两个训练队列
混在一起。原文一字未删，但下文已补上分层读数：full_corpus 层 0 / 66 = **0.0%**，
Wilson 95% CI [0.0%, 5.5%]；small 层 16 / 2,330 = **0.69%**，Wilson 95% CI
[0.42%, 1.11%]。判定基准也已改为分层口径：`verdict_detail.rate_sa_le_gate = 0.0`
（full_corpus 层，n = 66，`verdict_detail.cohort_tier = "full_corpus"`），**KILL
结论因此更硬，不是更软**。本次更正不改动任何历史数值，只补标签与分层。

## 计划与现实的差距（先记这一条）

计划写的是「3 seed × 200 样本，开/关 SA penalty 对照」。执行前核实，**这个实验
在三个独立的点上无法成立**：

1. **`--sa-penalty` 是空转的。** `sa_penalty.py` 的 docstring 自己写明：sascorer
   是片段频次查表，对参数没有梯度；trainer 的代理分被 `torch.tensor(float(...))`
   包住（`requires_grad=False`）。**开与关的参数更新逐位相同。** 拿它做 A/B 只会
   得到「两组完全一样」。
2. **没有可微的 SA 代理可以用于解码引导。** 2026-10-04 那次审计把假的代理删掉了。
   没有代理，「SA 引导解码」就没有可优化的目标。
3. **每个 run 只有一个 checkpoint（`..._seed0.pt`）。** 3 seed 需要先补训两个 seed。

## 试图补测 600 样本，失败（MEASURED）

用 panel of record 的 checkpoint 生成 600 个样本，失败：

```
RuntimeError: Error(s) in loading state_dict for CFMJointModel:
  Unexpected key(s) in state_dict: "vel_field.time_embed_proj.weight", ...
```

原因：该 checkpoint 的权重里有 `time_embed_proj`（形状 128×32）与
`pocket_per_layer_injector`，但它的 `args` 里**没有记录** `t_embedding_dim` /
`pocket_inject_per_layer`（读出来是 `None`）。也就是说，**这个 checkpoint 无法
从它自己记录的配置重建模型**。从权重形状反推开关能跑通，但那等于猜架构；猜错
会静默加载一份错权重，产出的分子看起来正常但来源不明。**没有做这个猜测。**

`cfm_pcgrad_on/off_2026_10_14/15` 的 checkpoint 也已损坏（`Invalid magic number`）。

## 实际测到的（MEASURED）— 无引导池的 SA 产量

| 池 | n | SA 均值 | ≤4.0 | ≤4.5 | ≤5.0 | ≤5.5 |
|---|---:|---:|---:|---:|---:|---:|
| panel of record（192 次尝试 → 66） | 66 | 6.4423 | 0 | **0** | 2 | — |
| pcgrad_on（30 epoch 转储） | 1,155 | 7.5478 | — | **9** | 31 | 72 |
| pcgrad_off（30 epoch 转储） | 1,175 | 7.4741 | — | **7** | 26 | 83 |
| **合计** | **2,396** | — | — | **16** | — | — |

**（2026-10-31 更正，上表「合计」行是一个混尺度行）**：它的分母 = full_corpus 层
66 个分子（panel of record，n_train = 26,715）+ small 层 2,330 个分子
（pcgrad_on / pcgrad_off，n_train = 256）。**跨层不可比**，分层读数见下表。

**SA ≤ 4.5 的比例：16 / 2,396 = 0.67%，Wilson 95% CI [0.41%, 1.08%]。**
（`combined`：`scale_label = "mixed_cohort_scale"`、`comparable = false`、
`scope = "generation_only"`。**这是一条混尺度率，跨层不可比**，它不是 full-corpus
生成器的率——分母 = full_corpus 层 66 + small 层 2,330。见下表分层读数。）

**分层读数（MEASURED，2026-10-31 更正；`cfm_sa_guided.json:combined_by_tier`）**

| 层（tier） | 池 | n_train | n | SA ≤ 4.5 | 比例 | Wilson 95% CI |
|---|---|---:|---:|---:|---:|---|
| full_corpus | panel_of_record | 26,715 | 66 | 0 | **0.0%** | [0.0%, 5.5%] |
| small | pcgrad_on + pcgrad_off | 256 | 2,330 | 16 | **0.69%** | [0.42%, 1.11%] |
| 合计（混尺度） | 3 池 | 26,715 + 256 × 2 | 2,396 | 16 | 0.67% | [0.41%, 1.08%] |

两个 reconstruction 池**不并入任何生成率**：`recon_fullcorpus_b64` 40 / 12 = 30.0%
[18.07%, 45.43%]、`recon_objective_terms` 43 / 16 = 37.21% [24.38%, 52.14%]。重建池
继承它起手的真实化学，它的率是**解码器保真度**
（`combined_by_kind.reconstruction.verdict_scope = "decoder_fidelity_on_real_geometries"`），
不是生成器的碎裂率；见 `combined.pools_excluded`。

**跨尺度对照（`metrics/by_metric/c7_scale_matched_pools.json`，Wf-A120 的第二个权威）**：
同一个 SA 门槛（4.5）在更大的一组 full_corpus 自由生成池上仍是同一量级的低率——
`by_tier_by_kind.generation.full_corpus`：3 池、14,846 个分子、60 个过线、
**0.40%**，Wilson 95% CI [0.31%, 0.52%]；`by_tier_by_kind.generation.small`：
2 池、2,330 个分子、16 个过线、**0.69%**，Wilson 95% CI [0.42%, 1.11%]。
`verdict.reading` 的原话是「scaling to the full corpus is not better on this gate」
——**放大到 full_corpus 并不能让 SA≤4.5 的产量变好**，两层都远低于 5% 判定线。

按 0.67% 算，计划里的 3 × 200 = 600 个样本期望只有 **约 4 个**分子过线。
（分层看：small 层 0.69% 同样是约 4 个；**full_corpus 层 0.0% 则是 0 个**——
这条期望值只对 small 队列成立。）

## 门内子集的另一半（MEASURED）

门槛是「SA ≤4.5 **且** 平均 novelty ≥0.90」。把这 16 个过线分子拿出来算
（Morgan r=2 / 1024 bit，对训练集 18,137 行）：

- 平均 novelty **0.8757**，中位 0.8889，**低于 0.90 门槛**；
- n=16，**统计上不可用**（`underpowered`）。

所以就算按 SA 手挑 0.67%，novelty 也补不上。**两条门槛同时不过。**
（2026-10-31 更正：这 16 个过线分子**全部来自 small 层**（pcgrad_on / pcgrad_off），
full_corpus 层一个都没有（`combined_by_tier.full_corpus.n_sa_le_gate = 0`，
`combined_by_tier.small.n_sa_le_gate = 16`）——所以门内子集的 novelty 读数
（0.8757，n = 16）也是 small 队列的读数。）

## 判定

**KILL**：

- 产量 0.67% < 5% 的判定线；
  （2026-10-31 更正：判定基准已按层重述为 full_corpus 层——
  `verdict_detail.rate_sa_le_gate = 0.0`、`verdict_detail.n_scored = 66`、
  `verdict_detail.cohort_tier = "full_corpus"`。`verdict_detail.verdict_basis` 原文片段：
  "the pool of record's cohort tier (full_corpus, panel_of_record, n=66)"
  ——原文完整句尾为 "-- a single training cohort"。**0.0% < 5%**，KILL 更硬；混尺度的
  0.67% 不作判定基准，见 `verdict_detail.mixed_scale_note`。）
- 门内子集 novelty 0.8757 < 0.90；
- 引导机制不存在（空转的 flag + 无可微代理 + 单 checkpoint）。

**「SA ≤4.5」不能作为生成器的性质来写。** 它只对一个 0.67% 的手挑子集成立，
而且那个子集的 novelty 还没过线。论文里如果写「SA 4.5」，必须同时写产量。

## 诚实读数

1. **这一轮没有花 3 小时 GPU，因为实验设计本身不成立。** 不是省事，是三处
   独立的阻断（见上）；继续跑只会得到「开=关」的 A/B 和一份来源可疑的权重。
2. **2,396 个分子的证据比 600 个强。** 600 个样本对 0.67% 的率给出的是
   ±0.4pp 的区间；2,396 个给出 [0.41%, 1.08%]。所以即使 600 样本跑成了，
   结论也一样。
   **（尺度限定：2,396 是个混尺度集合——full_corpus 层 66 个分子 + small 层
   2,330 个分子——所以「证据更强」说的是区间更窄，不是率本身可比。跨层的率必须
   分层读：full_corpus 层 0.0% [0.0%, 5.5%]、small 层 0.69% [0.42%, 1.11%]。
   结论（两层都远低于 5% 判定线）不变。）**
3. **pool of record 是最严的一个：0/66，最小值 4.751。** 它连一个 4.5 附近的
   分子都没有；大池里那 16 个过线分子来自 30 个 epoch 的转储（含早期 epoch），
   不是收敛模型的干净样本。所以 0.67% 是**上限侧**的估计，不是下限。
   **（2026-10-31 更正：pool of record 属于 full_corpus 层（n_train = 26,715），
   并且是该层的唯一池——所以 full_corpus 层 = 0 / 66 = 0.0%，Wilson 95% CI
   [0.0%, 5.5%]；这比混尺度的 0.67% 更严，pool of record 就是判定基准。）**
4. **SA 均值 6.44–7.55，参考真实 tmQM 是 5.03。** 生成物比真实语料**更难合成**
   +1.4 到 +2.5。这是架构/数据层面的问题，不是采样层能修的。
5. **QED 也在同一方向。** panel of record 的 QED 0.0944 vs 真实 0.2685。SA 与
   QED 同时差，说明不是单一指标的偏差。
6. **对接仍然 BLOCKED**（Vina 贵金属，catalog §6.52d）。STR-6 没有改变这一点，
   也没有假装能算。
7. **生成线到此定稿。** 没有可用引导机制，产量与 novelty 双不过线；后续若要
   改善，需要重训（新 checkpoint + 可微代理），不属于本轮范围。

## 对主表的影响

- 生成线结论固定为：**SA 6.4423（panel of record）、QED 0.0944、novelty 0.8517、
  对接 BLOCKED、SA≤4.5 产量 0.67%**。不能再写「SA 引导可到 4.5」。
  （2026-10-31 更正：0.67% 是混尺度率 `scale_label = "mixed_cohort_scale"`，跨层不可比；
  分层为 full_corpus 层 **0.0%**（0/66）与 small 层 **0.69%**（16/2,330）。）
- STR-7 的主张台账里，生成相关的每一条都必须带产量或 BLOCKED 标签。

## 复现

```bash
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/cfm_sa_yield_audit.py
```
