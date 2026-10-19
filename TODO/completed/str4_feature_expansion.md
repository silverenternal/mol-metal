# STR-4 — 特征扩展（ECFP6 / MACCS+RDKit / 实验条件）

**日期**：2026-10-19
**状态**：DONE / **PASS**（DOI R² 0.3767，Δ 符号未过）
**脚本**：`molmetal/scripts/msrs_feature_expansion.py`（单 seed 筛选 + 5 seed）
、`molmetal/scripts/msrs_feature_expansion_ensemble.py`（聚合 + 融合）
**测试**：`molmetal/tests/test_msrs_feature_expansion.py`（3 个）+ `test_msrs_feature_expansion_ensemble.py`（6 个），9/9 全过
**产物**：`metrics/by_metric/msrs_feature_expansion.json`（单 seed 三臂筛选）、
`msrs_feature_expansion_5seed.json`（5 seed）、`msrs_feature_fusion.json`（与 GNN 融合）、
`msrs_feature_expansion_shuffle.json`（置换控制，5 seed）、
`results/msrs_feature_expansion/oof_seed{0..4}.npz`

## 为什么做

当前 2,074 维 = ECFP4(2048) + 金属/细胞类别码 + 金属态/配位块。DOI 留出下
新论文的实验条件没有被显式建模，ECFP4 也可能截断大片段信息。三组臂，每组
只加一个块，保证效应可归因。

## 做法

- **ecfp6**：ECFP6(4096, radius 3) 替换 ECFP4(2048)。
- **maccs_rdkit**：ECFP4 + MACCS(167) + 20 个 RDKit 描述符（MolWt / LogP /
  TPSA / HBD / HBA / RotB / RingCount / HeavyAtomCount / FractionCSP3 /
  NumAromatic/Aliphatic/SaturatedRings / NumHeteroatoms / Max,MinPartialCharge /
  BalabanJ / BertzCT / Chi0 / Chi1 / Kappa1）。
- **assay**：ECFP4 + `log1p(time_h)` + `is_cancer_line`。
  **注意：语料没有 `method` 列**，所以这一块无法包含实验方法；这是记录，不是静默补齐。
- 同一组 fold（`scheme_folds`），同一组类别码，比较对象是**同 seed 的 ECFP4 树**。
- 配对 DOI 聚类 bootstrap（2,000 次）。
- **置换控制**：把新增块整块按行打乱（保留宽度、密度、边缘分布，摧毁与配体的
  对应关系），5 seed。如果增益还在，说明只是「多了 187 列」；如果消失，说明
  增益来自内容。

## MEASURED — 单 seed 筛选（seed 0，Δ 相对同 seed 树）

| 协议 | 树 | ecfp6 | maccs_rdkit | assay |
|---|---:|---:|---:|---:|
| ligand | 0.5554 | 0.5402 | 0.5563 | **0.5670** |
| doi | 0.3332 | 0.3321 | **0.3706** | 0.3410 |
| scaffold | 0.4484 | 0.4397 | **0.4638** | 0.4587 |
| year | 0.3703 | 0.3757 | **0.3860** | 0.3783 |

判定：`ecfp6` KILL（DOI −0.0012）、`assay` PARTIAL（+0.0078，CI 含 0）、
`maccs_rdkit` PASS（DOI 0.3706、Δ +0.0374 [+0.0205, +0.0539]）。

## MEASURED — 5 seed（seed 平均预测；seed mean ± sd）

| 协议 | 树 | maccs R²（均值） | maccs seed mean ± sd | Δ vs 树 [95% CI] | Δ 符号 |
|---|---:|---:|---:|---|---:|
| ligand | 0.5596 | 0.5593 | 0.5540 ± 0.0024 | −0.0003 [−0.0483, +0.0278] | 0.5917 |
| doi | 0.3380 | **0.3767** | 0.3697 ± 0.0022 | **+0.0387 [+0.0241, +0.0535]** | 0.5826 |
| scaffold | 0.4530 | **0.4733** | 0.4664 ± 0.0059 | **+0.0204 [+0.0077, +0.0337]** | 0.5826 |
| year | 0.3780 | 0.3946 | 0.3884 ± 0.0058 | +0.0166 [−0.0050, +0.0385] | 0.3673 |

### 置换控制（DOI，5 seed）

打乱后的块：0.3238 / 0.3286 / 0.3287 / 0.3270 / 0.3294 → **0.3275 ± 0.0022**，
低于原树 0.3332。**增益来自描述符内容，不是「多 187 列」的宽度/正则化效应。**

### 与 GNN 融合（5 树 + 5 GNN，stack_foldwise）

| 协议 | 融合 R² | Δ vs 树 [95% CI] | Δ 符号 |
|---|---:|---|---:|
| ligand | 0.5935 | +0.0339 [−0.0054, +0.0593] | 0.6101 |
| doi | **0.3811** | **+0.0431 [+0.0260, +0.0604]** | 0.5505 |
| scaffold | 0.4903 | +0.0373 [+0.0223, +0.0526] | 0.6147 |
| year | 0.4011 | +0.0232 [−0.0059, +0.0515] | 0.3878 |

## 判定

**PASS（R² 门槛）**：DOI R² **0.3767** ≥ 0.36，配对 ΔR² CI 下界 **+0.0241 > 0**。
融合后 **0.3811**。

**但 Δ 符号没过**：0.5826（maccs）/ 0.5505（融合），门槛 0.65，CI 下界也远低于
0.55。DP-2 因此是「R² 是、符号否」。

## 诚实读数

1. **这是项目里第一个真正过 0.36 的配置。** 树单模 0.3767（5 seed 均值预测，
   seed SD 0.0022），融合 0.3811。增益 +0.0387 是 DOI R² seed SD 的 17 倍。
2. **增益来自描述符内容，不是维度。** 置换控制把同样的 187 列打乱后掉到
   0.3275（比原树还低），说明模型确实在读 MACCS/描述符与配体的对应关系。
3. **R² 提升与排序能力无关。** Δ 符号 0.5826，与 ECFP4 树的 0.5826 完全相同；
   融合后反而降到 0.5505。全局描述符改善的是「水平」，不是「谁更好」。
   这直接决定了推荐线不能用这个模型。
4. **ECFP6 是负结果。** DOI 0.3321，比 ECFP4 还低。更大的指纹没有带来更多
   跨论文信息，反而稀释了信号（4,096 位里更多是论文特异子结构）。
5. **实验条件几乎没用。** assay 臂 DOI +0.0078，CI 含 0；`time_h` 与
   `is_cancer_line` 在 DOI 留出下没有可迁移的信号。
6. **ligand 上没有任何增益（−0.0003）。** 说明这 187 列不是「更强的指纹」，
   而是「跨论文更稳的指纹」——插值场景下 ECFP4 已经够用。
7. **`method` 列不存在。** assay 臂只含 `time_h` 与 `is_cancer_line`；论文里
   必须这样写，不能说「实验方法已控制」。
8. **MACCS 与 20 个描述符的归因还没拆。** 本轮只测了合并块；下一步若要写
   「描述符而不是 MACCS 起作用」，需要再跑两个 5-seed DOI 臂（约 10 分钟）。
   现在只能说「这个 187 维块」有效。
9. **year 的 CI 仍含 0**（+0.0166 [−0.0050, +0.0385]），6,499 行 / 625 聚类
   不足以支撑 year 的增益主张。

## 对主表的影响

- 跨论文主行改为：DOI R² **0.3767**（maccs 树，5 seed）、**0.3811**（+GNN 融合），
  相对 ECFP4 树 **+0.0387 / +0.0431**，CI 均排除 0。
- 必须同时给置换控制 0.3275，证明增益不是维度效应。
- Δ 符号行仍写 0.5826 / 0.5505，并注明未达 0.65——R² 与排序的分离是本论文的
  核心负结果之一。
- ligand 行不变（0.5593），不能把 maccs 说成「全面更强」。

## 复现

```bash
# 单 seed 三臂筛选（约 21 分钟）
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_feature_expansion.py --seed 0

# 5 seed × maccs（5 进程并发，约 6 分钟）
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_feature_expansion.py \
    --seed $s --arms maccs_rdkit \
    --oof-out results/msrs_feature_expansion/oof_seed$s.npz &
done; wait

# 聚合 + 融合
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_feature_expansion_ensemble.py

# 置换控制（DOI，5 seed；每个 seed 落一个 JSON，再聚合成 *_shuffle.json）
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_feature_expansion.py \
    --seed $s --protocols doi --arms maccs_rdkit --shuffle-extra \
    --out metrics/by_metric/msrs_feature_expansion_shuffle_seed$s.json &
done; wait
```
