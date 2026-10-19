# CLAIM LEDGER — 主张 → 证据 → 判定

**日期**：2026-10-20（STR-7）
**规则**：每条主张必须指向一个磁盘上的 artifact 指针；没有指针的写 `UNMEASURED`，
被外部依赖挡住的写 `BLOCKED`。**这张表是正文的上限**——正文不能比它走得远。
**数值来源**：`metrics/by_metric/*.json`；表格由
`molmetal/scripts/paper_metric_tables.py --strict` 机械生成（12 张表 / 0 MISSING）。

| ID | 主张 | 判定 | 关键数值 | 证据 |
|---|---|---|---|---|
| C1 | 金属感知表示（配合物 + 态 + 配位）显著优于配体单独 | **SUPPORTED** | Δ R² +0.1002（p=2.3e-20）；metal_blind 0.4873 vs full 0.5556 | `msrs_complex_features.json:/splits/fragment_grouped/complex/delta_r2_vs_main`；`msrs_metal_ablation.json:/protocols/ligand/arms/metal_blind/r2` |
| C2 | 跨论文（整篇 DOI 留出）绝对强度预测达 R² 0.36 | **SUPPORTED** | 0.3767（MACCS 树）/ 0.3811（融合） | `msrs_feature_expansion_5seed.json:/protocols/doi/r2_seed_averaged`；`msrs_feature_fusion.json:/protocols/doi/fused_r2` |
| C3 | 模型能判断同一队列里"哪个金属更好" | **NOT SUPPORTED** | 0.5826 / 0.5505 / 0.5183，门槛 0.65 | `msrs_seed_ensemble.json:/protocols/doi/ensemble/delta/0.2091/sign_accuracy`；`msrs_delta_direct.json:/gated/sign_accuracy` |
| C4 | 树 + GNN 融合提升跨论文 R² | **SUPPORTED（仅 R²）** | DOI +0.0217 [+0.0044,+0.0377]；融合后 Δ 符号 0.5826→0.5505 | `msrs_fusion_protocols.json:/protocols/doi/arms/stack_foldwise/vs_tree` |
| C5 | 不确定性区间校准 | **SUPPORTED（有边界）** | DOI 覆盖 0.9004、中位宽 2.0052；同队列多金属仅 0.8096 | `msrs_calibration.json:/protocols/doi/variants/global/coverage` |
| C6 | 跨金属迁移达到可用水平 | **NOT SUPPORTED**（LOMO）；within-metal 另计 | LOMO 0.5575±0.0949 < 0.7 门槛；within-metal Ru 0.8271 / Ir 0.8430 | `metal_lomo_auc_wasserstein.json:/value`；`metal_activity_transfer_seed_stability.json:/within_metal_stability` |
| C7 | 生成物可合成且新颖的金属药物 | **NOT SUPPORTED** | SA 6.4423、QED 0.0944、novelty 0.8517；SA≤4.5 产量 0.67% | `cfm_generation_metrics.json:/checks/sa/value`；`cfm_sa_guided.json:/combined/rate_sa_le_gate` |

---

## C1 — 金属感知表示承重

**能写**：「把配合物、金属氧化态与配位环境显式编码，比只用配体指纹提高 R² 0.1002
（片段分组，p=2.3e-20）；去掉金属身份码掉 0.0683，去掉态掉 0.0272，去掉配位掉
0.0130。」

**不能写**：不能写「架构创新」——增益来自**输入里有金属**，不是来自新的网络结构。

## C2 — 跨论文绝对预测

**能写**：「整篇论文留出下 R² 0.3767（5 seed，MACCS+RDKit 树），与 GNN 融合后
0.3811；相对 ECFP4 树 0.3380 的配对增益 +0.0387 [+0.0241,+0.0535]，置换控制
0.3275 证明增益来自描述符内容而非维度。」

**不能写**：「配体分组的 0.5593 是跨论文性能」——跨论文就是 0.3767。也不能写
「0.3767 代表可部署精度」——DOI 是论文代理，见 C2 的 `feature_headroom_audit`：
DOI offset 值 +0.1109 R²，而金属化学在 DOI 之后只有 −0.0003。

## C3 — 排序（哪个金属更好）

**能写**：「三条路线都未达预注册门槛 0.65：水平相减 0.5826 [0.4670,0.6805]、
融合 0.5505、Δ 直接回归 0.5183 [0.3902,0.6539]（无门槛 0.4988 ≈ 随机）。
R² 与排序在本语料上分离。」

**不能写**：「模型可以推荐金属」——任何推荐线的主张都属于 C3，而 C3 是 NOT
SUPPORTED。也不能写「直接回归显著更差」：区间与水平模型重叠。

## C4 — 融合

**能写**：「融合在 DOI 上 +0.0217（CI 排除 0），ligand +0.0401、scaffold +0.0310；
year 的 CI 含 0。**融合不改善排序**（0.5826→0.5505）。」

**不能写**：「融合是全面更强的模型」——它在排序上更差，且 ligand 之外的增益有一半
来自 GNN 的 seed 平均而非融合本身。

## C5 — 校准

**能写**：「DOI 留出、α=0.10 的 split conformal 覆盖 0.9004，中位宽 2.0052 log10；
点校准斜率 0.9256。」

**不能写**：「区间在任何子群都可行动」——同一队列多金属（851 行）覆盖只有 0.8096，
单金属 0.9050。这个子群必须单独报。

## C6 — 跨金属迁移

**能写**：「within-metal 分类在 5 seed 上稳定超过 Krasnov：Ru 0.8271（0.8128–0.8363）
vs 0.81，Ir 0.8430（0.8295–0.8597）vs 0.73；最差 seed 仍过线。」

**不能写**：「跨金属迁移可用」——LOMO 0.5575±0.0949，低于 0.7 的预注册门槛，
artifact 的 verdict 就是 `BELOW_TARGET_BASELINE_REAL_ONLY`。within-metal 是另一个
任务，不能拿它替代迁移主张。

## C7 — 生成

**能写**：「生成物 SA 6.4423（真实 tmQM 参考 5.0269）、QED 0.0944（真实 0.2685）、
novelty 0.8517、金属合规 1.0；SA≤4.5 的产量是 16/2,396 = 0.67% [0.41%,1.08%]。」

**不能写**：「SA 引导解码可把 SA 降到 4.5」——`--sa-penalty` 零梯度、无可微代理，
机制不存在；「SA≤4.5」只对 0.67% 手挑子集成立，且那 16 个的 novelty 0.8757 还没
过 0.90。对接仍 BLOCKED。

---

## UNMEASURED（完全没有数字）

| 主张 | 为什么没有 | 需要什么 |
|---|---|---|
| 外部数据集上的绝对精度 | PlatinAI 只有 Pt、MSRS 无 Pt、NCI-60 无 SMILES，三键交集 0 | 一个新的、含多贵金属且带 SMILES 的外部集 |
| 细胞系特征带来的增益 | 0 个 CCLE/GDSC 文件；46/711 细胞系覆盖 80% 数据 | 细胞系分子特征表 |
| 湿实验验证 | 硬性不做（§5.4） | — |

## BLOCKED（有依赖挡着）

| 主张 | 阻断原因 | 证据 |
|---|---|---|
| 贵金属配合物的对接打分 | Vina 无贵金属参数 | catalog §6.52d |
| SA 引导解码 | 无梯度 + 无可微代理 + 单 checkpoint | catalog §6.140 |
| 3-seed 生成扫描 | 每 run 只有一个 `seed0` checkpoint | catalog §6.140 |

---

## 与表格的对应

| 台账 | 表 |
|---|---|
| C1 | T4_mechanism |
| C2 | T2_generalization（单 seed ECFP4）、T11_generalization_strong（5 seed 三臂） |
| C3 | T12_ranking |
| C4 | T11_generalization_strong、T2_generalization |
| C5 | T3_calibration |
| C6 | T10_external、`metal_activity_transfer_seed_stability.json` |
| C7 | T7_generation |

**重生成命令**：

```bash
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/paper_metric_tables.py --strict
```

退出码 0 = 0 MISSING。`manifest.json` 记录每个源文件的 sha256；重跑 artifact 后
必须重跑本脚本，否则表会与产物脱节。
