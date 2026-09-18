# 靶点选择

## 三大类目标(对比)

### A. 通用 benchmark (CrossDocked / SPINDR)

- **范围**:100+ 任意蛋白口袋
- **优势**:论文有直接对比基线(FLOWR, TargetDiff 论文)
- **缺点**:广而不深,没突出我们贵金属特化的优势
- **建议**:起步用,作为 baseline 验证

### B. 单个热门靶点 (PARP1, BRAF, EGFR)

- **PARP1** (聚腺苷二磷酸核糖聚合酶 1)
  - **临床药物**:olaparib(铂以外的金属 chelator 类似机制)
  - **PDB ID**:常用 5DS3, 7KK4
  - **口袋特征**:深口袋,NAD+ 结合位点
  - **与金属关系**:有报道的金属-PARP1 抑制剂(少见)

- **BRAF** (V600E 突变,黑色素瘤)
  - **临床药物**:vemurafenib, dabrafenib
  - **PDB ID**:1UWH, 3OG7, 4R5Y
  - **口袋特征**:中等深,ATP 竞争

- **EGFR** (肺癌)
  - **临床药物**:osimertinib(第三代 TKI)
  - **PDB ID**:1M17, 2ITY, 4HJO
  - **口袋**:深

**优势**:能发有深度的 case study
**缺点**:单一靶点,泛化性弱

### C. Undruggable targets (KRAS, MYC, p53)

- **KRAS G12C**
  - **临床药物**:sotorasib (AMG 510)
  - **PDB ID**:6T5V
  - **挑战**:浅口袋,无 deep cavity(2018 前被认为 undruggable)
  - **论文热点**:很多 SBDD 专门打这个靶
  - **与金属**:有 KRAS-Pt 偶联物研究(用于蛋白 pull-down)

- **MYC**
  - **挑战**:转录因子,无经典口袋
  - **现状**:被认为 "really undruggable"

- **p53**
  - **临床药物**:eprenetapopt (PRIMA-1MET, 重新激活 mutant p53)
  - **PDB ID**:复杂,需要野生型 / mutant 对比
  - **与金属**:一些 Pt 复合物可以重新激活 mutant p53

**优势**:顶级难度,能出好论文
**缺点**:成功概率低,可能要花很多时间

## 我们的贵金属切入点

**金属在 SBDD 中的角色**:
- 经典 Pt 药物(cisplatin)与 DNA 交联
- 但有**金属-蛋白**相互作用的研究:
  - **金属伴侣蛋白**(metallochaperones):Atox1, Ctr1 — 负责 Pt/Cu 转运
  - **金属酶**:carbonic anhydrase, matrix metalloproteinases (MMPs) — Zn 依赖
  - **金属结合蛋白**:Zn-finger, Fe-S cluster, heme
- **金属-Scaffold 蛋白-配体 三元复合物**预测 = SOTA 没解决的开放问题

## 推荐起点

**Phase 0 (1 周)**: 通用 benchmark
- 跑 CrossDocked100 / SPINDR subset
- 复现 FLOWR / TargetDiff 在 5-10 个 pocket 上的 Vina score
- 验证 adapter 正确性

**Phase 1 (2 周)**: 单个热门靶点
- **推荐:PARP1** (有公开结构、benchmark 完整、mechanism 清晰)
- 或 **BRAF V600E** (有晶体结构 + 临床药)

**Phase 2 (2-3 周)**: 挑战题
- **KRAS G12C** (sotorasib 设计)
- 我们的金属 drug 特化: **Zn-dependent MMPs** (matrix metalloproteinases)
  - MMP 是 Zn 依赖的,刚好契合"贵金属 drug"主题
  - 有公开 inhibitor(抑制剂),可作为 baseline

## 关键蛋白-MMP 是我们的最自然切入点

**为什么 MMPs**:
1. **金属依赖** (Zn²⁺ / Ca²⁺) — 与贵金属 anticancer 天然契合
2. **公开抑制剂多**(batimastat, marimastat, prinomastat 等)
3. **结构清晰** — PDB 大量 co-crystal
4. **现有 ML 关注少** — niche,有差异化空间
5. **Pocket 适合** — shallow/medium depth,既不太难也不太易

**MMP2 / MMP9** (gelatinases, 与肿瘤转移相关):
- PDB: 1QIB, 1GKC, 4H2D 等
- 临床药物:海洋药物如 tetraxetan

**MMP13** (collagenase-3, 骨关节炎+肿瘤):
- PDB: 1XUC, 1ZTQ, 3KRY

## 我们推荐的起点: MMP + generic

**Phase 0**: CrossDocked100(通用 baseline)
**Phase 1**: MMP2 + MMP9(我们的 niche)
**Phase 2**: BRAF 或 KRAS(更难靶点)
**Phase 3**: 真实 metal-protein(把 metal 当成 ligand 的一部分)
**Phase 4**: 综合评估 + 论文

## 待你回答

| 问题 | 选项 |
|---|---|
| 1. 第一靶点 | (a) 通用 CrossDocked (b) PARP1 (c) BRAF (d) KRAS (e) **MMP2/9** (推荐) |
| 2. 我们的金属 drug 是 | (a) 仅 Pt(II) (b) Pt + Ru + Au (c) **更广:含 Zn(II) MMPs** |
| 3. 起点数据来源 | (a) CrossDocked (b) PDBbind (c) **SPINDR** (d) DrugDesignAI-Benchmark |

## 立即可做(不等答案)

1. Clone FLOWR, DiffDock, TargetDiff 到 `molmetal/references/`
2. 写 `molmetal/domain/{pocket,molecule,complex}.py`
3. 写 `molmetal/ports/{generator,docking,predictor,scorer,design_loop}.py`
4. 写 mock adapter + e2e test
5. 跑 FLOWR 在 CrossDocked sample 上,看 dock+score 链路通

## 资源下载(开工时用)

```bash
# SPINDR (FLOWR 自带)
git clone https://github.com/jule-c/flowr.git molmetal/references/FLOWR
# (注意:原文档写错,insitro/FLOWR 不存在,正确作者是 Julian Cremer 等)
# 另:还建议 clone FLOWR.root (jule-c/flowr_root) — 2026 新版带 affinity prediction
# 数据会自动下载

# DiffDock
git clone https://github.com/gcorso/DiffDock.git molmetal/references/DiffDock
# Checkpoint: https://github.com/gcorso/DiffDock#checkpoints

# TargetDiff
git clone https://github.com/guanjq/targetdiff.git molmetal/references/TargetDiff
# Checkpoint: https://drive.google.com/drive/folders/1wWNy37ZoAlwB0ZSkH7NXOLXgZQsZVDMQ

# CrossDocked100 (sample)
# 来自 TargetDiff repo 的 data/ 目录
```

## 蛋白结构准备

工具:
- **PDBe API** (https://www.ebi.ac.uk/pdbe/) — 下载 PDB
- **MDAnalysis** — 处理 MD trajectory
- **OpenMM / PyMOL** — 预处理口袋
- **RDKit** — 加氢 / 标准化

```python
# 从 PDB ID 加载口袋
from molmetal.domain.pocket import Pocket
pocket = Pocket.from_pdb_file("1QIB.pdb", ligand_center=..., radius=6.0)
```