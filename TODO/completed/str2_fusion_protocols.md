# STR-2 — 四协议融合臂（GNN + blend + stack）

**日期**：2026-10-19
**状态**：DONE / **PARTIAL**（效应成立，绝对门槛差 0.0003）
**脚本**：`molmetal/scripts/msrs_dmpnn_protocols.py`（GNN OOF）+ `molmetal/scripts/msrs_fusion_protocols.py`（融合）
**测试**：`molmetal/tests/test_msrs_fusion_protocols.py`（12 个，全过）
**产物**：`metrics/by_metric/msrs_fusion_protocols.json`、`results/msrs_dmpnn_protocols/`（5 seed × 4 协议 OOF）
**输入**：STR-1 的树 OOF（5 seed）+ 本次 GNN OOF（5 seed，GPU0 并发，1h38m）

## 为什么做

MET-2 测了树在四个协议下的崩塌（DOI 0.3321），但融合臂的 +0.0273 只在
ligand 划分上验证过。`msrs_generalization.design.arms_not_measured = 3` 一直
是「跨论文融合是否成立」的空白。STR-2 把这块补上：GNN 与树在**同一组 fold**
上产出 OOF，再比较 blend / stack。

## 做法

- GNN：D-MPNN `metal_state` 臂，超参与 FIX-M1 完全相同（hidden 128 / depth 3 /
  60 epoch / bs 64），只换 fold。内层 early-stopping 也按协议分组留出
  （DOI 协议留 DOI、year 协议留年份），避免用配体分组泄漏论文。
- 树：直接复用 STR-1 的 5 个 `oof_seed*.npz`。
- 两边的 seed 数都是 5；预测先按 seed 取均值，再评分。
- 臂：`tree` / `gnn` / `blend_equal`（无参数，主臂）/ `blend_foldwise`
  （w 在其余 fold 上拟合）/ `stack_foldwise`（a,b,c 同样拟合）。
- ΔR² 用**配对 DOI 聚类 bootstrap**（2000 次，同一批 DOI 同时重采样两个模型）。
- gate 写死在 `todo.json`：DOI stack R² ≥0.36 且配对 CI 下界 >0。

## MEASURED — seed 平均（5 树 + 5 GNN）

| 协议 | tree | gnn | blend_equal | stack | Δ stack−tree | 配对 95% CI | Δ equal−tree |
|---|---:|---:|---:|---:|---:|---|---:|
| ligand | 0.5596 | 0.5670 | 0.5929 | **0.5997** | **+0.0401** | [+0.0256, +0.0522] | +0.0333 |
| doi | 0.3380 | 0.3089 | **0.3606** | 0.3597 | **+0.0217** | [+0.0044, +0.0377] | +0.0226 |
| scaffold | 0.4530 | 0.4402 | 0.4795 | **0.4839** | **+0.0310** | [+0.0202, +0.0416] | +0.0265 |
| year | 0.3780 | 0.3554 | 0.3999 | **0.3968** | +0.0189 | [−0.0092, +0.0431] | +0.0219 |

残差相关：ligand 0.8648 / doi 0.8902 / scaffold 0.8814 / year 0.8952。

## MEASURED — 单模型（同一 seed 的树与 GNN，5 seed 均值 ± sd）

| 协议 | tree | gnn | stack | Δ stack−tree |
|---|---:|---:|---:|---:|
| ligand | 0.5555 ± 0.0026 | 0.5032 ± 0.0248 | 0.5828 ± 0.0102 | +0.0273 |
| doi | 0.3322 ± 0.0029 | 0.2442 ± 0.0235 | 0.3485 ± 0.0046 | +0.0162 |
| scaffold | 0.4475 ± 0.0028 | 0.3705 ± 0.0215 | 0.4693 ± 0.0068 | +0.0218 |
| year | 0.3728 ± 0.0066 | 0.2986 ± 0.0158 | 0.3862 ± 0.0082 | +0.0134 |

ligand 的 0.5828 ± 0.0102 与 MET-1 完全一致——这是两条独立管线的交叉校验。

## 判定

**PARTIAL。**

- DOI `stack_foldwise` R² = **0.3597**，门槛 0.36，**差 0.0003**；
  配对 ΔR² CI 下界 **+0.0044 > 0**（效应成立）。
- `blend_equal`（无参数主臂）R² = **0.3606**，过 0.36；其 CI [+0.0045, +0.0407]。
- 按写死的规则，gate 臂是 stack，所以结论记为 **miss**，不把 0.3597 四舍五入成 0.36。

## 诚实读数

1. **效应是真的，绝对水平差一点。** 四个协议的 ΔR² 全部为正，三个 CI 排除 0；
   DOI 的 +0.0217 [+0.0044, +0.0377] 是项目里第一个「跨论文显著为正」的模型增益。
   但 gate 写的是绝对 R²，0.3597 就是不达标，不能因为「差一点」改判。
2. **融合增益有一半来自 GNN 的 seed 平均，不是来自融合本身。** DOI 上单模型
   GNN 是 0.2442，5 模型平均后 0.3089（+0.065）；树只从 0.3322 到 0.3380
   （+0.006）。所以单模型 stack 是 0.3485 ± 0.0046，5 模型 stack 是 0.3597。
   论文若写 0.3597，必须同时说明这是 **5 个 GNN + 5 个树**的集成；写 0.3485
   才是单模型可复现的数字。
3. **融合把 DOI 的 Δ 符号弄差了。** 树 0.5826 → stack 0.5413、equal 0.5413；
   MAE(Δ) 只从 0.5115 降到 0.5092。R² 与排序在这条线上**方向相反**。决策层
   消费的是排序，所以融合不能用来支撑「金属推荐」那条故事。
4. **两个模型犯的错很像。** 残差相关 0.86–0.90，融合只能吃到很小一块去相关
   分量。这也解释了为什么 +0.02 就是上限附近：即使 GNN 完全独立，R² 的提升
   也受树本身的误差地板限制。
5. **year 的 CI 含 0。** 只有 6,499 行被评估、625 个聚类，+0.0189 不成立。
   year 协议的配对样本量不支持任何增益主张。
6. **ligand 的 0.5997 是项目最高值，但它是插值。** 同一批论文的配体同时出现在
   训练和测试里；这个数字不能当跨论文性能。跨论文是 0.3597。
7. **stack 与 equal 的差距在噪声内。** DOI 0.3597 vs 0.3606，ligand 0.5997 vs
   0.5929。拟合权重没有带来稳定收益，无参数的 equal 更值得作为主臂。
8. **这次没有改任何 gate。** 0.3597 < 0.36 就是 miss；equal 的 0.3606 是同一
   实验的另一个臂，报出来是为了让读者看到边界，不是用来替换 gate 臂。

## 对主表的影响

- 跨论文主行可以写：DOI R² **0.3597**（5+5 集成 stack），相对树的
  **+0.0217 [+0.0044, +0.0377]**；同时给单模型 0.3485 ± 0.0046。
- 不能写「融合把 DOI 抬过 0.36」——stack 没到；只能写「融合显著改善，
  但绝对水平仍在 0.36 以下」。
- Δ 符号行必须注明融合后是 0.5413（树 0.5826），不能只报 R²。
- 决策/推荐线继续用树的 0.5826，不要用融合模型。

## 复现

```bash
# GNN OOF：5 个 seed 并发（GPU0），约 1h40m
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_dmpnn_protocols.py \
    --config configs/msrs_dmpnn_protocols_gpu0.yaml --seed $s &
done; wait

OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_fusion_protocols.py
OMP_NUM_THREADS=1 .venv/bin/python -m pytest molmetal/tests/test_msrs_fusion_protocols.py -q
```
