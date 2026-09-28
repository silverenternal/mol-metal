# TODO-142: Cross-domain training recommendation (4 sources × 7.5M molecules)

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P0** — 当前只用了 0.17% available data (13K / 7.5M molecules),paper §4 lead claim 不完整
**Parent**: User-pasted recommendation (跨域分子属性预测的完整训练建议,2026-09-27) + TODO-141 §2.1 audit
**Target journal**: **JCIM IF 5.6** (Q2 中科院二区)
**Skill alignment**: `party-twenty-spirit` §3.5 + `mol-metal-mechanism-mapping`

---

## §0 Why this TODO exists

### User-pasted 建议核心判断

> "工程侧的四个机制(归一化、不确定性加权、域采样、域对抗)是必须做的,不做 7.5M 训练会在某个维度上悄悄崩塌;层论和拓扑工具是理论增强层,提供结构化的域对齐和可解释的泛化监控。"

### Mol-Metal 当前 audit(诚实)

| 数据域 | 当前训练用量 | Available | 覆盖率 |
|---|---:|---:|---:|
| MetalCytoToxDB (5 metal) | **5,207** | 5,207 | 100% |
| DrugOOD native | **0** | **1,580,000** | 0% |
| NCI-60 9-cell subset | **7,824** | 7,824 | 100% |
| NCI-60 GI50 raw | **0** | **4,850,000** | 0% |
| BindingDB IC50 | **0** | **1,072,000** | 0% |
| PlatinAI 4-cell | ~5,000 | 5,000 | 100% |
| CrossDocked (SBDD targets) | **0 trained** | 183K PDB | 0% |

**总训练用量**:~13K unique molecules,占 available 7.5M 的 **0.17%**!

**结论**:这是 paper §4 lead claim 不完整的根本原因 — 不是模型不 work,是数据没用够。

---

## §1 Mol-Metal 4 个数据域对齐

按 user-pasted 建议的 4 个域概念,映射到 Mol-Metal:

| Mol-Metal 数据域 | 化学类型 | 标签类型 | 用途 |
|---|---|---|---|
| **D1: MetalCytoToxDB** | 5 precious metal complexes | pIC50 / active flag | Layer 3 主数据 |
| **D2: DrugOOD** | General drug-like + IC50 | pIC50 / active flag | Layer 2 anticancer OOD |
| **D3: NCI-60 GI50** | General drug + cell line | GI50 / activity | Layer 2 wet-lab validation |
| **D4: BindingDB** | General drug + protein | IC50/Kd/Ki | Layer 2 binding affinity |

**Mol-Metal 任务** = 9 cell line × 5 metal = **45 task heads**(NCI-60 9-cell multi-task D-MPNN) + DrugOOD OOD 3 splits + CFM decoder

---

## §2 工程侧 — 必须做的 4 个机制(Phase 1,1 周)

### §2.1 机制 1:BatchNorm → GroupNorm/LayerNorm(半天)

**Why Mol-Metal needs this**:
- D-MPNN 当前用 BatchNorm-like layers
- 4 个数据域的化学分布差异极大(D1=metal-coordinated, D2=general drug, D3=cell-specific, D4=protein binding)
- BatchNorm running stats 会被域组成污染 — 80% DrugOOD batch 时 NCI-60 样本被错误归一化

**Audit + Replace**:
1. `molmetal/molmetal_lam/sbdd_env/dmpnn.py` (or equivalent) — find all BatchNorm layers
2. Replace with `nn.GroupNorm(num_groups=32, num_channels=...)` 或 `nn.LayerNorm`
3. Verify all multi-task heads still produce consistent shape
4. Re-test on 100K subsample before scaling

**Verify gate**:
- ✅ No `BatchNorm1d` / `BatchNorm2d` in D-MPNN backbone
- ✅ GroupNorm groups chosen based on channel count (32 groups typical)
- ✅ Forward pass shape不变 + 多任务头输出维度不变

### §2.2 机制 2:Kendall 不确定性加权(1 天)

**Why Mol-Metal needs this**:
- 45 task heads(9 cell × 5 metal)手工调权重不现实
- 当前 NCI-60 9-cell 用 hardcoded task weights — 一旦加更多 metal/cell,会失衡

**Implementation**(PyTorch):
```python
class MultiTaskUncertaintyLoss(nn.Module):
    """Kendall et al. CVPR 2018 同方差不确定性加权"""
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses_per_task):
        total = 0
        for i, L in enumerate(losses_per_task):
            precision = torch.exp(-self.log_vars[i])
            total += precision * L + self.log_vars[i]
        return total
```

**Wire into**:
- `molmetal/molmetal_lam/sbdd_env/multi_task.py`(or equivalent)
- 45 task heads(9 cls + 9 reg × 5 metal)+ DrugOOD 3 splits = 48 task total

**Verify gate**:
- ✅ log_vars initialized 0, training drives them non-zero
- ✅ At convergence, learned weights should be reasonable(不能全 0,不能全 ∞)
- ✅ Multi-task eval metric aggregate matches pre-change baseline ±0.01

### §2.3 机制 3:Domain-aware 采样权重(半天)

**Why Mol-Metal needs this**:
- 4 域大小悬殊(D3 4.85M vs D1 5K)
- 直接 concat 后随机采样 → 模型学到"哪个数据集大就偏向哪个"的捷径

**Formula**:
```
w_d ∝ (N_target / N_d)^α,  α ∈ [0.5, 1.0]
```

**Implementation**:
```python
from torch.utils.data import WeightedRandomSampler
# per-domain weights, normalized
weights = []
for d in domains:
    for idx in d.indices:
        weights.append((N_target / len(d)) ** alpha)
sampler = WeightedRandomSampler(weights, num_samples=total_per_epoch,
                                 replacement=True)
```

**Critical detail** (per user-pasted advice):
- 分布式训练 **必须用** `WeightedRandomSampler`,**不能**用 `random.choices()`
- 后者会丢 state,导致 worker 间分布不一致 → 梯度发散

**Verify gate**:
- ✅ Per-domain batch composition matches weights within ±5%
- ✅ Multi-epoch training stable(没有 NaN 或梯度爆炸)

### §2.4 机制 4:DANN 域对抗(2 天)

**Why Mol-Metal needs this**:
- D1 "toxicity" ≠ D3 "GI50" — 标签语义不同
- 多域联合训练会学"域身份 → 标签"的伪相关
- DANN 让主干表示"域分类器分不出来",同时"任务头预测准"

**Architecture**:
```
backbone → [task heads × 45] (主损失)
         ↘ [domain classifier × 4] (对抗,GRL 反向传播)
```

**Implementation**:
- 主干后加 4-class 域分类头(softmax over D1/D2/D3/D4)
- 梯度反转层(`-λ` 系数,λ 从 0.1 开始 grid search)
- Loss: `L_total = L_task - λ * L_domain`(GRL 让 L_domain 反向优化)

**Reference** (per user-pasted advice):
- **TwistDAN** 已在 640K 分子上验证 DANN + 半监督学习组合

**Verify gate**:
- ✅ 域分类器在 train 上 AUC 接近 0.5(对抗成功,domain-agnostic)
- ✅ 主任务 multi-task AUC 不下降(对抗没有 destroy task signal)
- ✅ OOD 跨域 AUC ≥ baseline(实际 gain)

---

## §3 理论侧 — Phase 2-3 升级(2-4 周)

按 user-pasted 建议分层实施:

### §3.1 Sheaf theory(层论)— 结构化域协调

**Sheaf-FRL(2026)**:每个域潜在表示 → 层截面 → 层拉普拉斯"粘合惩罚"

**Mol-Metal 映射**:
- 4 域 → 层节点
- 层拉普拉斯正则项 = 域间表示一致性
- 替换机制 3 的 per-domain WeightedRandomSampler(更结构性)

**Sheaf-FMTL(TMLR 2025)**:层拉普拉斯任务关系建模
- 45 task heads → 层节点
- 自动学习任务间关系拓扑(谁 share info、share 多少)
- 替换 Kendall 加权(Sheaf-FMTL 学关系,Kendall 学权重)

**SGPC(AAAI 2026)**:Sheaf + PAC-Bayes
- 提供认证置信区间,回应"对数据分布不敏感"

**Implementation caveat**:
- 限制映射初始化敏感 → 用 **Procrustes 闭式更新** + 局部梯度交替
- 不要完全替换架构,在现有 D-MPNN 上加辅助损失

### §3.2 Topological deep learning(TDL)

**TopoAD(2025)**:中间层激活图欧拉特征 + 能量分数 → OOD 检测
**pTU(2025)**:预测拓扑不确定性,显著性检验

**Mol-Metal 映射**:
- 每个 epoch 后,对每个域验证集计算拓扑漂移指标
- 如果某域漂移 ↑ → 增大该域采样权重(动态反馈闭环)

### §3.3 Kan extensions(范畴论)

**Learning Transfers via Kan Extensions(2026)**:
- 任务变化定义为函子 J: 𝒜 → ℬ
- 左 Kan 扩展 Lan_J F 显式刻画"哪些结构应该被保留"
- **迁移差异**:Comp_J(F, G) = sup_b d_V(Lan_J F(b), G(b))

**Mol-Metal 映射**:
- LODO 评估时不仅报告性能,还计算迁移差异
- 定位"哪个域的哪些拓扑结构在迁移中丢失"

---

## §4 落地路线图(3 阶段)

### Phase 1:最小可行版本(1 周,4 工程机制)

| 优先级 | 动作 | 工作量 | 资源 |
|:---:|---|---|---|
| 1 | BatchNorm → GroupNorm audit + replace | 0.5 天 | CPU |
| 2 | Kendall 不确定性加权 module | 1 天 | CPU |
| 3 | Per-domain WeightedRandomSampler | 0.5 天 | CPU |
| 4 | 100K subsample pipeline 验证 | 1 天 | GPU-0 |
| 5 | 3-stage curriculum + 分层 batch | 2 天 | GPU-0 |

**评估协议**:
- **LODO**(Leave-One-Domain-Out):4 域各跑一遍留一验证
- **Per-domain breakdown**:所有主指标按域拆开报告
- **分布距离监控**:每 epoch 每个域计算 MMD 或 Wasserstein 距离

### Phase 2:域对抗 + 拓扑监控(1-2 周)

| 动作 | 工作量 | 资源 |
|---|---|---|
| DANN 域对抗(λ 从 0.1 搜索) | 2 天 | GPU-0 |
| TopoAD/pTU 域漂移监控 | 2 天 | CPU |
| Kan 扩展迁移差异审计 | 2 天 | CPU |

### Phase 3:层论增强(2-3 周,可选升级)

| 动作 | 工作量 | 资源 |
|---|---|---|
| Sheaf-FRL 层拉普拉斯辅助损失 | 3 天 | GPU-0 |
| Sheaf-FMTL 任务关系建模 | 5 天 | GPU-0 |
| SGPC PAC-Bayes 泛化边界 | 3 天 | GPU-0 |

---

## §5 自查表(self-check rigor)

| Check | Standard | Verified |
|---|---|:---:|
| **BatchNorm 替换** | `grep -r BatchNorm` = 0 hits in D-MPNN | ? |
| **Kendall 权重非退化** | log_vars 标准差 > 0.1 after 50 epoch | ? |
| **Per-domain 采样平衡** | domain batch ratio within ±5% of weights | ? |
| **DANN 对抗成功** | domain classifier AUC < 0.6 on val | ? |
| **LODO 4 域** | D1/D2/D3/D4 各跑一遍 + honest average | ? |
| **Per-domain breakdown** | 所有主指标按域拆开 ship in §4 | ? |
| **Sheaf-FRL 收敛** | 限制映射 Procrustes update 稳定 | ? |
| **Sheaf-FMTL 关系学习** | 学到的 45 任务关系图可视化 | ? |
| **TopoAD 漂移监控** | 监控指标 ship + 自动告警阈值 | ? |
| **Kan 迁移差异审计** | LODO 报告含 Comp_J(F, G) 值 | ? |

---

## §6 Key decision points(per user-pasted 建议)

### 关于归一化层
GroupNorm 已证明可替代 BatchNorm。先做这个,**成本最低、收益最确定**。

### 关于 Kendall vs Sheaf-FMTL
- Kendall:学权重(1 天实现,稳定)
- Sheaf-FMTL:学关系拓扑(5 天+,更优雅但需调试)
- **建议先 Kendall 保底**,跑通后再考虑 Sheaf-FMTL 升级

### 关于 DANN vs 层论
- DANN:对抗消除域信息(激进,可能丢域特信息)
- 层论:学习域间结构关系(温和,保留局部结构)
- **建议先 DANN**,如果发现域特信息丢失严重,再切换层论

### 关于数据扩展
- **先做 100K subsample 验证 pipeline 收敛**
- 确认 OK 后再扩展到 7.5M
- mini 版不收敛 → 7.5M 版一定崩
- 省下的 GPU 时间够十次超参搜索

---

## §7 Mol-Metal 当前 vs 建议的状态对比

| 维度 | 当前 | 建议 Phase 1 | 建议 Phase 3 |
|---|---|---|---|
| 训练数据 | 13K (0.17%) | 100K (1.3%) | **7.5M (100%)** |
| 归一化 | BatchNorm | GroupNorm | GroupNorm |
| 任务加权 | hardcoded | Kendall | Sheaf-FMTL |
| 域采样 | uniform random | WeightedRandomSampler | Sheaf-FRL |
| 域对抗 | none | DANN | DANN + Kan |
| 漂移监控 | none | TopoAD/pTU | TopoAD/pTU |
| 泛化保证 | empirical | empirical | PAC-Bayes |

---

## §8 Resource + Risk

**资源**:
- ✅ GPU-0 RTX PRO 6000 Blackwell 96GB(0% util,空闲)
- ⚠️ GPU-1 RTX 5090 32GB(被 Wf-131 P3 占用中,~2.5 小时)
- ✅ 32 CPU + 94 GB RAM + 1.8 TB disk
- ✅ pocket/external_baselines/12 Tier-1 baselines 已 clone

**风险**:
- **LODO 4 域扩展可能 OOM**:7.5M rows × Morgan FP(2048) = 60 GB RAM(刚好够)
- **DANN λ 搜索**:需要 5-10 次 grid run,每跑 ~3 小时 = 15-30 小时 GPU
- **Sheaf-FRL 收敛**:限制映射初始化敏感,需要 Procrustes 调优
- **Sheaf-FMTL 5 天+调试**:可能拖到 2 周,可能 fail

**Likelihood**:
- Phase 1 完成率:**85%**(4 个工程机制都成熟,只缺时间)
- Phase 2 完成率:**70%**(DANN 已成熟,TopoAD/pTU 需要额外实现)
- Phase 3 完成率:**50%**(Sheaf-FMTL / SGPC 都是 2025-2026 新,实现风险高)

---

## §9 Honest framing per party-twenty-spirit §3.5

**承认**:
1. **当前 13K / 7.5M = 0.17%** — 训练数据严重不足,paper §4 lead 不完整
2. **DrugOOD 0.891/0.697/0.748 是 metacytotox_fallback**,不是真 DrugOOD head-to-head
3. **LOMO 7/125 = 5.6%** — Os/Rh/Re 完全没训
4. **4 个工程机制当前全没实现** — 这是新的工程债
5. **Phase 2-3 工具(Sheaf-FRL / Sheaf-FMTL / TopoAD)都是 2025-2026 新方法**,实现风险高

**不假装**:
1. 不假装 4 个机制是 optional — 都是 **必须做**
2. 不假装 7.5M 是 optional — 不做就没有 paper §4 honest lead
3. 不假装 Sheaf-FRL/FMTL 是 incremental — 是 high-risk experimental
4. 不假装 4 域 DANN 自动成功 — 需要 λ grid search

**paper §6 必须 ship**:
- "Current models trained on 13K molecules (0.17% of available 7.5M); Phase 1 engineering mechanisms (GroupNorm + Kendall + WeightedRandomSampler + DANN) implemented in Wf-XXX for full data utilization; Phase 2-3 theoretical enhancements (Sheaf-FRL/FMTL) out of scope for this paper."

---

## §10 Cross-reference

- **TODO-140**:Tier-1 baseline alignment (Phase 1 alignment)
- **TODO-141**:Q2 JCIM completion sprint(本 TODO 升级其 Phase 2-3)
- **Wf-131 P3**:LOMO 5/5 metal × 5 seed × 50 epoch (7/125 done)
- **Wf-145**:m19 Rh fix (fail,需要 Rh n=60 + balanced cohort)
- **Wf-126 / Wf-140 / Wf-142 / Wf-144**:CrossDocked 5 pocket Vina work
- **metrics/by_metric/nci60_precious_9cell_auc.json**:NCI-60 9-cell 当前数字
- **metrics/by_metric/drugood_ood_auc.json**:DrugOOD fallback + native ship 数字
- **TODO-138**:paper submission package 5-day plan
- **TODO-84**:paper rule active — Phase 1+2 docs 待 user 授权

---

## §11 Final verdict

**建议优先级**(per user-pasted + Mol-Metal audit):

| Phase | 动作 | 工作量 | ROI |
|---|---|---|---|
| **A** | Phase 1 完整 1 周(4 工程机制 + 100K subsample) | 1 周 | **高** |
| **B** | + LOMO 5/5 metal × 5 seed × 50 epoch 完整 | +3-5 小时 | **高** |
| **C** | + Phase 2 DANN + TopoAD | +1-2 周 | 中 |
| **D** | + Phase 3 Sheaf-FRL/FMTL | +2-3 周 | 低(高风险) |

**Likelihood of acceptance**:
- 现在 ship(无 Phase 1):**70-80%**
- Phase 1 + B 后 ship:**80-85%**
- Phase 1 + B + C 后 ship:**85-90%**
- Phase 1 + B + C + D 后 ship:**85-90%**(Phase 3 风险大,可能 delay)

**下一步**:用户决定:
- **A**:只启动 Phase 1 工程机制(BatchNorm + Kendall + WeightedRandomSampler + DANN)~1 周
- **B**:Phase 1 + LOMO 完整 5/5 metal(我推荐)— 1 周 + 5 小时 GPU
- **C**:A + Phase 2(2-3 周)— 完整 Mol-Metal v2
- **D**:A + B + C + D — 6 周冲刺

我推荐 **B**:Phase 1 + LOMO 完整,这是 paper §4 lead claim 真正能 ship 的最小改进。

---

## §12 User-pasted 建议原文(完整保留作为 reference)

```text
跨域分子属性预测的完整训练建议

综合之前的工程讨论和理论调研,以下是一份从工程实践到数学前沿的完整建议。
核心判断是:工程侧的四个机制(归一化、不确定性加权、域采样、域对抗)是必须
做的,不做 7.5M 训练会在某个维度上悄悄崩塌;层论和拓扑工具是理论增强层,
提供结构化的域对齐和可解释的泛化监控。

[完整 4 部分内容在原始 pasted message 中保留作为 reference;
包括:第一部分工程侧 4 机制,第二部分理论侧 3 类数学工具(Sheaf-FRL/Sheaf-FMTL/
SGPC + TopoAD/pTU + Kan extensions),第三部分 3 阶段落地路线图,第四部分关键
决策点。完整中文版保存在 git history / conversation log。]
```

---

## §13 Related skill alignment

- `party-twenty-spirit` §3.5:0-tolerance honest-negative framing → §9
- `mol-metal-mechanism-mapping`:3 层 stack 与 4 数据域映射 → §1
- `mol-metal-anticancer`:anticancer domain context → §1
- `mol-metal-lambda-calculus`:Lambda + CFM + MCTS architecture 与新工具集成 → §2-3
- `mol-metal-evaluation-rigor`:LODO 协议 + per-domain breakdown → §5