# FIX-C7 — 生成线的 SA 数字既测错了模型，也测错了管线

**日期**：2026-10-20（v1.1 修正同日）
**状态**：DONE / **NOT SUPPORTED（归因第 3 次改写）**
**脚本**：`molmetal/scripts/c7_scale_matched_pools.py`（复用 `cfm_sa_yield_audit.pool_report`）
**测试**：`test_c7_scale_matched_pools.py`（16 个，全过）
**产物**：`metrics/by_metric/c7_scale_matched_pools.json`（version 1.1）
**成本**：CPU，0 GPU，秒级

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
docstring 自己写着：

> **!! THESE ARE NOT GENERATED MOLECULES. MEASURED 2026-10-11. !!**
> This script loads REAL tmQM geometries via `load_tmqm_clouds` and runs only the
> bond head: featurise -> `_select_bonds` -> `_bridge_components` -> `_assemble_mol`.
> It never calls the sampler.

也就是说：这些 SMILES 是**真实 tmQM 分子的键头重建**。几何是给定的，所以 SA 必然
继承真实语料的 SA——**「与真实语料持平」正是这个假象的指纹，不是质量的证据。**

真正的生成路径（`evaluate_generation_validity` → `generate_clouds_from_noise`）
写出的表头是 `epoch,smiles,n_frags_raw,largest_frag_frac`。

## MEASURED — 按管线 × 规模分层

```
[c7] cfm_fullcorpus_2026_10_10        reconstruction n=  66 n_train=64    sa=6.4423 gate=0  gen_validity=0.9141
[c7] cfm_fullcorpus_b64_2026_10_11    reconstruction n=  40 n_train=26715 sa=5.0320 gate=12 gen_validity=0.0859
[c7] cfm_objective_terms_2026_10_11   reconstruction n=  43 n_train=26715 sa=5.0334 gate=16 gen_validity=0.0742
[c7] cfm_ode_link_probe               generation     n=  12 n_train=2048  sa=5.6503 gate=1  gen_validity=0.0130
[c7] cfm_pcgrad_off_2026_10_15        generation     n=1175 n_train=256   sa=7.4741 gate=7  gen_validity=0.0833
[c7] cfm_pcgrad_on_2026_10_14         generation     n=1155 n_train=256   sa=7.5478 gate=9  gen_validity=0.2917
```

| 管线 | 分子 | 覆盖的规模层 | **缺的层** |
|---|---:|---|---|
| `generation`（真生成） | 2,342 | small(2,330)、mid(12) | **full_corpus** |
| `reconstruction`（键头重建） | 149 | full_corpus(83)、small(66) | mid |

**全语料自由生成池：不存在。** 盘上最大的真生成池来自 `n_train=256` 的 run。

v1.0 的「全语料 SA≤4.5 = 33.7% [24.5,44.4]」全部来自重建池，**不能作为生成器性质引用**。

### 唯一可跨规模比较的数字

`gen_validity` 由训练器同一段代码在每个规模上计算，因此可比：

| 规模 | run | gen_validity | n |
|---|---|---:|---:|
| 全语料 26,715 | `cfm_fullcorpus_b64` | **0.0859** | 768 |
| 全语料 26,715 | `cfm_objective_terms` | **0.0742** | 768 |
| 256 | `cfm_pcgrad_on` | 0.2917 | 48 |
| 256 | `cfm_pcgrad_off` | 0.0833 | 48 |
| 2,048（2 epoch） | `cfm_ode_link_probe` | 0.0130 | — |

**扩大训练规模没有提升生成有效率**：全语料 0.074–0.086，256 样本 0.083–0.292。
这是「训练规模不是解释」的直接证据，且与 SA 无关、与管线无关。

## 判定

**C7 仍是 NOT SUPPORTED。理由：可判定该主张的 artifact 不存在。**

- 要说「生成器能产出 SA≤4.5 且新颖的金属配合物」，需要**全语料的自由生成池**；
  盘上一个都没有。
- 现有真生成证据全部来自 `n_train=256`（2,330 个分子，SA≤4.5 = 16/2,330 = 0.69%
  [0.42,1.11]）。
- 全语料唯一可比的 `gen_validity` 没有随规模改善。

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
