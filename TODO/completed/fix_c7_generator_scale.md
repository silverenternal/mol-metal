# FIX-C7 — 生成线的 SA 数字既测错了模型，也测错了管线

**日期**：2026-10-20（v1.1 修正同日）
**状态**：DONE / **NOT SUPPORTED（归因第 3 次改写）**
**脚本**：`molmetal/scripts/c7_scale_matched_pools.py`（复用 `cfm_sa_yield_audit.pool_report`）
**测试**：`test_c7_scale_matched_pools.py`（16 个，全过）
**产物**：`metrics/by_metric/c7_scale_matched_pools.json`（version 1.1）
**成本**：CPU，0 GPU，秒级

> ⚠️ **CORRECTED 2026-10-21 — 本文的核心事实已被后续 run 推翻。**
> 「全语料自由生成池不存在」在 2026-10-20 是真的，2026-10-21 起不再成立：
> `cfm_c7_fullgen_2026_10_20` 跑完 **18/18 个 epoch**（`exit=0`，wall 11,232 s），
> 盘上有 12,669 行真生成池。它的 SA≤4.5 = **0.28% [0.21,0.39]**（36/12,668），
> 仍低于 n_train=256 池的 0.69% —— 结论（NOT SUPPORTED）不变，**理由的第 4 次改写**。
> 本文以下内容保留为历史记录，**不要**再引用「不存在」这一句。
> 权威版本：`docs/HONEST_NEGATIVE_CATALOG.md` §6.150b/§6.150c、`docs/CLAIM_LEDGER.md` C7 行。

> ⚠️ **RE-QUOTED 2026-10-31 (Wf-A118 移数 + Wf-A120 补标签；本次修补 Wf-A123)。**
> 本文出现的三个 SA 产量数**分母各不相同**，引用时必须带规模标签：
> * **0.28% [0.21,0.39]（36/12,668）= 单个池** `cfm_c7_fullgen_2026_10_20`
>   （n_train=26,715，18/18 epoch）的率。Wf-A118 把跨规模的数搬到 tier 口径之后，
>   **这个池的数没有变**（`metrics/by_metric/c7_scale_matched_pools.json:/pools[0]/
>   sa_counts_at_or_below/4.5` = 36、`/pools[0]/n_scored` = 12,668、
>   `/pools[0]/joint_gate/n` = 10,190；CI = 该比率的 Wilson 95%，与本产物的
>   `wilson_ci95` 同口径）——但「全语料」三个字现在只能指**这一个池**。
> * **0.4% [0.31,0.52]（60/14,846）= 全语料生成层**（三池：`cfm_c7_fullgen_2026_10_20`
>   12,668 + `cfm_c8_fullgen_2026_10_24` 2,112 + `cfm_fullcorpus_2026_10_10` 66）的率
>   （`c7_scale_matched_pools.json:/by_pipeline/generation/by_tier/full_corpus/rate_sa_le_gate`
>   = 0.004、`/wilson_ci95` = [0.0031, 0.0052]、`/n_scored` = 14,846、
>   `/n_sa_le_gate` = 60）。**本文下面写 0.28% 的地方若被读成「全语料生成层」，
>   就是读错了分母。**
> * **16/2,396 = 0.67% = 混合规模集合**，不是任何一层的率
>   （`metrics/by_metric/cfm_sa_guided.json:/combined/scale_label` = `mixed_cohort_scale`、
>   `/combined/comparable` = false：panel_of_record 的 66 行来自 n_train=26,715，
>   pcgrad_on/off 的 2,330 行来自 n_train=256）。Wf-A120 补上了标签与分层，
>   **可引用的是分层后的两个数**：全语料层 **0/66 = 0.0% [0.0%, 5.5%]**
>   （`/combined_by_tier/full_corpus`）、small 层 **16/2,330 = 0.69% [0.42%, 1.11%]**
>   （`/combined_by_tier/small`）。**本文原措辞「16/2,396 = 0.67%」按惯例保留为历史记录。**
> * 顺带校正头部：本文「产物」行写的 `c7_scale_matched_pools.json` **version 1.1**
>   已过时——盘上同一路径现在是 **version 1.6**（generated_utc 2026-10-30T18:45:28Z，
>   新增 `by_tier_by_kind` 管线分层视图）。本文其余正文按历史原样保留。
>
> **结论（NOT SUPPORTED）与所有负向陈述一律不变**——变的只是分母标签：换成全语料
> 生成层口径也是 0.4% < n_train=256 池的 0.69%，方向不变。

## 问题

C7 的可写数字是「SA 6.4423（真实 tmQM 5.0269）、QED 0.0944（真实 0.2685）、
SA≤4.5 产量 16/2,396 = 0.67%」。这个数字被读成「生成器做不出可合成的金属配合物」。

两轮归因：

| 轮次 | 归因 | 错在哪 |
|---|---|---|
| v0 | 架构不行 | 没查训练规模 |
| v1.0 | 训练规模（256 样本 vs 26,715） | **把两种管线当成一个变量** |
| **v1.1** | **盘上不存在全语料的自由生成池** | — |

### v1.1 的发现

v1.0 把 `cfm_fullcorpus_b64_2026_10_11`（40 个）与 `cfm_objective_terms_2026_10_11`
（43 个）当成「全语料生成池」，得出 SA 5.032 ≈ 真实 5.0269 的结论。

**这两个 CSV 不是生成物。** 它们的表头是
`epoch,seed,smiles,sanitized,n_fragments,largest_fragment_frac,metal_z`——
正是 `molmetal/scripts/wf_cfm_decode_dump_smiles.py` 的输出格式。该脚本的
docstring 自己写着（**2026-10-30 已被 Wf-A114D 修正**——这句话**只对 2026-10-12
之前写出的文件成立**）：

> **!! "THESE ARE NOT GENERATED MOLECULES" WAS TRUE FOR THE FILES WRITTEN BEFORE**
> This script loads REAL tmQM geometries via `load_tmqm_clouds` and runs only the
> bond head: featurise -> `_select_bonds` -> `_bridge_components` -> `_assemble_mol`.
> It never calls the sampler.

也就是说：这些 SMILES 是**真实 tmQM 分子的键头重建**。几何是给定的，所以 SA 必然
继承真实语料的 SA——**「与真实语料持平」正是这个假象的指纹，不是质量的证据。**

真正的生成路径（`evaluate_generation_validity` → `generate_clouds_from_noise`）
写出的表头是 `epoch,smiles,n_frags_raw,largest_frag_frac`。

> ⚠️ **CORRECTED 2026-10-30 (Wf-A114D + Wf-A114E) — 表头本身不再决定管线。**
> **表头相同 ≠ 管线相同。** `wf_cfm_decode_dump_smiles.py` 在 2026-10-12 之后
> 改成了 ODE 路径（`evaluate_generation_validity` → `generate_clouds_from_noise`），
> **写出的表头一模一样**。区分二者必须看**内容**：
> `sanitized == 1` 且 `metal_z` 非空 ⇒ 重建（era A）；`sanitized == 0` 且
> `metal_z` 为空 ⇒ 生成（era B）。唯一权威规则：
> `molmetal/scripts/cfm_pool_kind.py`。
> **本文下面那张表把 `cfm_fullcorpus_2026_10_10` 记成 `reconstruction`，是错的：
> 它是 era B 的自由生成池。** 它的 `n_train` 也不是表里写的 `64`——`64` 来自
> 一次 1-epoch CPU smoke 覆盖了 `per_epoch_metrics.json`（Wf-A114E 的 census
> 人口冲突缺陷）；真实 cohort 是 **26,715**（epoch 10 的 checkpoint，
> `metrics/by_metric/cfm_free_generation_panel_seed0.json` → `/checkpoint`
> 早已写明 "epoch 10, n_train 26715"）。

## MEASURED — 按管线 × 规模分层

```
[c7] cfm_fullcorpus_2026_10_10        generation     n=  66 n_train=26715 sa=6.4423 gate=0  gen_validity=0.3438
[c7] cfm_fullcorpus_b64_2026_10_11    reconstruction n=  40 n_train=26715 sa=5.0320 gate=12 gen_validity=0.0859
[c7] cfm_objective_terms_2026_10_11   reconstruction n=  43 n_train=26715 sa=5.0334 gate=16 gen_validity=0.0742
[c7] cfm_ode_link_probe               generation     n=  12 n_train=2048  sa=5.6503 gate=1  gen_validity=0.0130
[c7] cfm_pcgrad_off_2026_10_15        generation     n=1175 n_train=256   sa=7.4741 gate=7  gen_validity=0.0833
[c7] cfm_pcgrad_on_2026_10_14         generation     n=1155 n_train=256   sa=7.5478 gate=9  gen_validity=0.2917
```

> **⚠️ RE-QUOTED 2026-10-30 (Wf-A114D + Wf-A114E).** 第一行原来记的是
> `reconstruction n_train=64 gen_validity=0.9141`。三个字段全错：
> * `reconstruction` → **`generation`**（era B 的 ODE 自由生成池；表头相同不代表
>   管线相同，见上）。
> * `n_train=64` → **`26715`**（`64` 来自覆盖了 `per_epoch_metrics.json` 的
>   1-epoch CPU smoke；真实 cohort 由 `cfm_free_generation_panel_seed0.json`
>   的 `/checkpoint` 独立佐证 = "epoch 10, n_train 26715"）。
> * `gen_validity=0.9141` → **`0.3438`**（= 66/192，来自
>   `cfm_free_generation_panel_seed0.json` → `/generation/gen_validity`；
>   `0.9141` 是被污染的 smoke 行）。
> 其余 5 行的 pipeline / n_train / gen_validity **不变**（已逐行核对
> `metrics/by_metric/cfm_fragmentation_audit.json`）——污染只落在 4 个
> class-A run 上，其中只有这 1 个出现在本表。

| 管线 | 分子 | 覆盖的规模层 | **缺的层** |
|---|---:|---|---|
| `generation`（真生成） | 2,408 | small(2,330)、mid(12)、full_corpus(**66**) | full_corpus 的**量**（只有 66 个） |
| `reconstruction`（键头重建） | 83 | full_corpus(83) | small、mid |

**全语料自由生成池：存在，但极小。** 盘上**唯一**一个 `n_train=26715` 的自由生成池
是 `cfm_fullcorpus_2026_10_10` 的 **66** 行（epoch 10，ODE 从噪声采样），
SA≤4.5 = **0/66**。其余真生成池都来自 `n_train=256` 或更小的 run。
> ⚠️ **CORRECTED 2026-10-21**：这一句只在 2026-10-20 成立。此后全语料 run 跑到 18/18，
> 12,669 行真生成池已存在。见本文顶部横幅。

v1.0 的「全语料 SA≤4.5 = 33.7% [24.5,44.4]」全部来自重建池，**不能作为生成器性质引用**。

### 唯一可跨规模比较的数字

`gen_validity` 由训练器同一段代码在每个规模上计算，因此可比：

| 规模 | run | gen_validity | n |
|---|---|---:|---:|
| 全语料 26,715 | `cfm_fullcorpus_2026_10_10`（**第 13 epoch**，修正后） | **0.5964** | 768 |
| 全语料 26,715 | `cfm_fullcorpus_b64` | **0.0859** | 768 |
| 全语料 26,715 | `cfm_objective_terms` | **0.0742** | 768 |
| 256 | `cfm_pcgrad_on` | 0.2917 | 48 |
| 256 | `cfm_pcgrad_off` | 0.0833 | 48 |
| 2,048（2 epoch） | `cfm_ode_link_probe` | 0.0130 | — |

**扩大训练规模没有提升生成有效率**：全语料 0.074–0.086，256 样本 0.083–0.292。
这是「训练规模不是解释」的直接证据，且与 SA 无关、与管线无关。

> **⚠️ CORRECTED 2026-10-30 (Wf-A114E).** 上表原来只有 5 行，缺
> `cfm_fullcorpus_2026_10_10`——因为 census 把它记成 1 epoch 的 CPU smoke
> （`gen_validity_end = 0.9140625`），其真实 14-epoch 序列被
> `per_epoch_metrics.json` 的覆盖抹掉了。**从 `events.jsonl` 的 ground truth
> 读回**（`molmetal/reports/cfm_fullcorpus_2026_10_10/events.jsonl`，第 0 个
> population）：epoch 0 = 0.08203125、epoch 10 = 0.3255208333333333、
> **epoch 13 = 0.5963541666666666**（分母均为 768）。
> **所以「全语料 0.074–0.086」这句话本身就是幸存者偏差**：它只引了两个
> 全语料 run，而全语料人口在最后一个 epoch 上的 `gen_validity` 实际横跨
> **0.048 – 0.75**（`metrics/by_metric/cfm_run_census.json` → `runs[*]`，
> 14 个 `n_train=26715` 的 run）。**「训练规模不是解释」这个结论本身仍然成立**
> （同规模内方差 0.048–0.75，远大于跨规模差），但**原来那两句支撑它的数字不能
> 再引用**。

## 判定

**C7 仍是 NOT SUPPORTED。理由：可判定该主张的 artifact 不存在。**

- 要说「生成器能产出 SA≤4.5 且新颖的金属配合物」，需要**全语料的自由生成池**；
  盘上一个都没有。
- 现有真生成证据全部来自 `n_train=256`（2,330 个分子，SA≤4.5 = 16/2,330 = 0.69%
  [0.42,1.11]）。
- 全语料唯一可比的 `gen_validity` 没有随规模改善。

> **⚠️ RE-QUOTED 2026-10-30 (Wf-A114D + Wf-A114E).** 「artifact 不存在」这个
> **理由**本身是错的（全语料自由生成池**存在**，见顶部横幅），**结论 NOT SUPPORTED
> 不变，但理由升级为更强的一条**：
> * 全语料自由生成池 = `cfm_fullcorpus_2026_10_10`（**n_train 26715**，epoch 10，
>   ODE 从噪声采样），**66** 行；
> * 其 SA≤4.5 = **0/66 = 0.00%**（**独立复算**：RDKit `sascorer`，SA 均值
>   **6.4423**，最小 SA **4.7510** > 4.5 门槛）——**不是「没测」，是「测了，零命中」**；
> * 其诚实的端到端单分子率 = **64/192 = 0.3333**（192 次尝试 → 66 个可解析
>   SMILES → 全部 sanitize 通过 → 64 个单分子），判 **FAIL**；
> * 更大的 12,669 行全语料生成池（`cfm_c7_fullgen_2026_10_20`）SA≤4.5 =
>   **0.28% [0.21,0.39]**，仍低于 256 样本池的 0.69%。
> **所以 C7 的诚实理由是「两个全语料生成池都低于 n_train=256 的池」，而不是
> 「证据不存在」。**

## 诚实读数

1. **v1.0 的「SA 半边可达」必须撤回。** 那句话建立在重建池上。真实语料 SA 5.0269
   被重建继承，是构造使然。
2. **v1.0 的「novelty 随规模反向（0.8677→0.6977）」也一并失效**：0.6977 是重建物对
   训练集的 novelty，重建物本来就来自训练集，这个数只会更低，不能与生成物比。
3. **`cfm_sa_yield_audit` 的 0.67% 本身没错**，它测的是真生成；错的只是把它归给架构。
4. **`cfm_r9_fix_v2`（type-F1 0.947、n_train=26,715）没有 `checkpoints/`**，所以连
   重建都没法补做。要判定必须重训。
5. **`--sa-penalty` 仍是零梯度、无可微代理。** 旧结论不变。

## 对台账的影响

> ⚠️ **SUPERSEDED 2026-10-21.** 下面这段是当时的台账措辞，已不适用：全语料真生成池存在，
> 台账 C7 行现在是「SA≤4.5 = 0.28%，低于 256 样本的 0.69%」。见 `docs/CLAIM_LEDGER.md`。

C7 行改为：

> **NOT SUPPORTED（无全语料自由生成池；2,330 个真生成分子 SA≤4.5 = 0.69%）**。
> 盘上全语料池全部是键头重建（`wf_cfm_decode_dump_smiles.py`，给定真实几何），
> 其 SA 5.032 是构造产物。真生成池最大只到 `n_train=256`：SA 7.47–7.55、
> SA≤4.5 **0.69% [0.42,1.11]**。跨规模唯一可比的 `gen_validity` 全语料
> 0.074–0.086 vs 256 样本 0.083–0.292，**没有随规模改善**。

## 复现

```bash
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/c7_scale_matched_pools.py
```

## 下一步（已获用户 OK，2026-10-20）

在全语料（26,715）上重训一个 CFM run，**保留 checkpoint** 且开启
`save_generated_smiles`，产出 ≥1,000 个**真生成**分子，再重跑本审计。
这是唯一能填上 `generation/full_corpus` 空格、把 C7 从「artifact 不存在」变成
可点估计的路径。

- 配置：`configs/c7_fullgen_2026_10_20.yaml`（`cfm_fullcorpus_b64_2026_10_11.yaml`
  的逐字节副本，只改 epochs 16 / gen_eval_seeds 6 个 / save_generated_smiles /
  checkpoint_interval 1 / device cuda:0 / 输出路径）
- 冒烟：`configs/c7_fullgen_smoke_2026_10_20.yaml`，1 epoch × 512 clouds，
  MEASURED 301 s，写出 `epoch,smiles,n_frags_raw,largest_frag_frac` 32 行 —— 确认走的是
  生产生成路径，不是键头重建
- pre-flight：114 pass / 0 fail / 10 skip

## 顺带发现（不在 C7 范围，未在 run 中修复）

`--consistency-term-weight` 默认 **0.05** 且 `enabled=True`，但
`consistency_loss_fn` 内部的 `sample_edm_sigma` 默认 `device=cpu`，与 cuda 上的
`noise` 相乘抛 `RuntimeError`，被 `ConsistencyTerm.compute` 的 `except` 捕获后
返回 `torch.zeros(())`。三行复现：

```python
from molmetal.adapters.flow_matching_lipman.consistency_regularization_wf380 import consistency_loss_fn
x = torch.randn(4, 3, device="cuda:0")
consistency_loss_fn(lambda z: z * 0.5, x, weight=0.05)
# RuntimeError: Expected all tensors to be on the same device ... cuda:0 and cpu
```

**这一项自该权重上线起每一步都是精确 0。** 没有在本次 run 中修复，因为修了会改变
目标函数、使本 run 与 `cfm_fullcorpus_b64` 不可比；而这一项在 b64 里同样是死的，
所以两者仍然可比。已记为 `FIX-C8`（OPEN）。

