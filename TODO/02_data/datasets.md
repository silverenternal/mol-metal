# 数据集规格

## 主要训练数据:MetalCytoToxDB

**完整下载**:https://zenodo.org/records/17106822 (v latest),还有 15853577、20153879
**Paper**:10.1021/acs.jmedchem.5c02755 (Krasnov 2026)
**大小**:6.4 MB CSV
**已 web 验证可达**(2026-09-10 实测)

### 字段规格(24 列)

| 字段 | 含义 | 用于 |
|---|---|---|
| `SMILES_Ligands` | 配体 SMILES(不含金属) | 输入 |
| `Counterion` | 反离子 SMILES | 输入 (Krasnov 没用 → 我们的探索点) |
| `Abbreviation_in_the_article` | 论文缩写 | 调试用 |
| `IC50_Dark(M*10^-6)` | 暗 IC50(μM) | label(原始) |
| `IC50_Dark_standard_error` | SE | 可选置信加权 |
| `IC50_Dark_value` | 数值 IC50(ML 用) | **主 label** |
| `IC50_Light_value` | 光照下 IC50(PDT) | 第二 label(可选) |
| `Excitation_Wavelength(nm)` | PDT 波长 | PDT 子任务 |
| `Irradiation_Time(min)` | PDT 时间 | PDT 子任务 |
| `Irradiation_Power(W/m²)` | PDT 功率 | PDT 子任务 |
| **`Cell_line`** | A549, HeLa 等 | **分组/条件** |
| **`Time(h)`** | 暴露时长 | **任务分组** |
| `DOI` | 来源 | provenance |
| **`Year`** | 文献年份 | **时间切分** |
| `IC50_Cisplatin_value` | 顺铂参考 | baseline/对照 |
| **`Charge_complex`** | 复合物总电荷 | 输入(可条件化) |
| **`Metal`** | 中心原子(Ru/Ir/...) | **任务过滤** |
| `Oxidation_state` | 形式氧化态 | 输入 |
| `Atomic_number` | 金属原子序数 | 输入 |
| `Valence_e` | 价电子数 | 输入 |

### 关键子集划分

| 子集 | 复合物数 | IC50 数 | 预期用途 |
|---|---|---|---|
| Ru only | ~3000 | ~10000 | 主模型 |
| Ir only | ~1500 | ~5000 | 第二基准 |
| Pt only (与 MB Finder 合并) | 3700 + 17000 | 20000+ | 时间切分 benchmark |
| Ru+Ir 多金属 | ~4500 | ~15000 | 跨金属迁移 |

### 必做预处理

1. **过滤 Time ≥ 24h**(短期数据噪声大)
2. **过滤 IC50_Dark_value**:剔除 SE>IC50、缺失值、`< 0.01 μM`(体外数据不可靠)
3. **分类标签**:`active` (IC50 < 10 μM) vs `inactive` (>10 μM)— Krasnov 用的就是这个 cutoff
4. **数值标签**:`pIC50 = -log10(IC50 μM)` — D-MPNN 回归用

### 反离子建模(我们的差异化)

Krasnov 说"counterion 不考虑"。**我们想试试**:
- 用 RDKit 把 Counterion 配上去(`Chem.AllChem.CombineMols` 类似)
- 看 AUC 是否涨

## 第二数据:MB Finder / PlatinAI 数据

**URL**:https://www.mb-finder.org
**Paper**:ChemRxiv 2026, DOI 10.26434/chemrxiv-2025-pp32k
**格式**:在线查询 + CSV 下载
**统计**:17,732 IC50 / 3,725 Pt 复合物

**评估协议**:
- 训练:pre-2024 论文
- 测试:2024+ 论文
- **指标**:hit rate (% 一致率) — PlatinAI 72%, 我们的目标 75%+

## 第三数据:tmQM(电子结构辅助)

**URL**:https://github.com/uiocompcat/tmQM ✅ (注意:原文档写错,正确组织是 UiO Computational Catalysis,不是 Ulissi Group)
**大小**:100,703 个 TMC,7 个 DFT 性质(HOMO/LUMO/电荷/...)
**用于**:作为特征(不是 label)— 把 DFT 计算的电子描述符喂进模型

### 怎么用 tmQM

不直接是 cytotoxicity,但可以:
1. **预训练**:在 tmQM 上预训练 EGNN 学电子结构 → fine-tune 到 MetalCytoToxDB
2. **特征增强**:对每个 metal complex,查 tmQM 找相似复合物的 DFT 描述符,作为额外特征

预训练路线(2026 ElemeNet 论文支持):在 tmQM 上 train on HOMO/LUMO → 转移到小数据 IC50。
**风险**:tmQM 没有 Pt 几何(主要是 Ru/Ir)。ElemeNet 也是先 train on 通用然后 fine-tune。

## 第四数据:NCI-60 (作为 baseline 用)

**来源**:https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Growth+Inhibition+Data
**大小**:~130k 化合物 × 60 细胞系
**包含**顺铂、卡铂、奥沙利铂等 — 有这些 Pt 复合物的 cytotoxicity

**用法**:验证我们的模型在普通(非贵金属)抗癌药上也不弱于 D-MPNN,作为通用性证据。

## 数据拼装计划

```python
# Pseudo
metal_cytotox_df = pd.read_csv("MetalCytoToxDB.csv")
# 1. Filter: Ru only
ru_df = metal_cytotox_df[metal_cytotox_df["Metal"] == "Ru"]
# 2. Active/inactive
ru_df["active"] = (ru_df["IC50_Dark_value"] < 10.0).astype(int)
# 3. Optional: 3D conformer via RDKit ETKDGv3
ru_df["mol"] = ru_df["SMILES_Ligands"].apply(RDKit_embed_3d)
# 4. Featurize
dataset = MetalCytoDataset(ru_df, split_strategy="random")  # or "temporal"
```

## 数据集下载清单 (实际状态 — 2026-09-11)

| 数据 | 大小 | URL | 状态 |
|---|---|---|---|
| MetalCytoToxDB.csv | 4.7 MB | zenodo.org/records/17106822 | ✅ `/mnt/storage/data/molmetal/MetalCytoToxDB.csv` (26801 rows) |
| MB Finder Pt 数据 | 6.1 MB | chemrxiv.org/doi/suppl/10.26434/chemrxiv-2025-pp32k | ✅ 3 个 xlsx (226k SMILES + 214k predictions) |
| tmQM | 415 MB | github.com/uiocompcat/tmQM | ✅ `/mnt/storage/data/molmetal/tmQM/` |
| NCI-60 pGI50 | 396 MB | wiki.nci.nih.gov + Zenodo 镜像 | ✅ `/mnt/storage/data/molmetal/NCI60_GI50/GI50.csv` (官方,396 MB 解压) |
| NCI-60 PharmacoSet | 327 MB (gz) | zenodo.org/records/5570629 | ✅ `/mnt/storage/data/molmetal/NCI60_pharmacoset.rds` (ORCESTRA 处理后,2.3 GB 解压) |
| SPINDR | 375 MB (zip) / 11.4 GB | zenodo.org/records/15257565 | ✅ `/mnt/storage/data/molmetal/SPINDR/smol_data/` (FLOWR 训练用) |
| CrossDocked2020 (CascadeDiff 处理) | 1.6 GB | zenodo.org/records/20703074 | ✅ `/mnt/storage/data/molmetal/CrossDocked2020_cascadediff.zip` |

## 关键风险

- **MB Finder 在线查询限速** — 可能需要批量爬取
- **tmQM 太大** — 考虑只取 Pt/Ru/Ir 子集(~30k 复合物)
- **SMILES_Ligands 不含金属中心** — 我们要在 SMILES 里把金属加上 (`[Pt]`) 才能跑 RDKit 3D

## 接下来该跑的命令

```bash
# 1. 下载 MetalCytoToxDB
curl -L https://zenodo.org/records/17106822/files/MetalCytoToxDB.csv \
    -o data/MetalCytoToxDB.csv

# 2. 探查
source .venv/bin/activate
python -c "
import pandas as pd
df = pd.read_csv('data/MetalCytoToxDB.csv')
print(df.columns.tolist())
print(df['Metal'].value_counts())
print(df.shape)
"
```