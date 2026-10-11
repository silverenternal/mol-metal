# 投稿计划 — Paper A（预测线）

Decided 2026-10-11. Scope + shared-asset manifest: [`PAPER_SCOPE.md`](PAPER_SCOPE.md).
Aligned-baseline table: [`ALIGNED_COMPARISON.md`](ALIGNED_COMPARISON.md).
本文件只讲**投稿动作与顺序**，不重复 scope 论证。

**Nothing here has touched `paper/`** (AGENTS.md §5.3, TODO-84). `paper/main.tex` currently
holds the OLD single-paper narrative (λ-calculus + CFM), which is Paper B's material. Paper A is
a **new manuscript**; see §4.

---

## 1. 目标期刊

| | Paper A（预测线） |
|---|---|
| **首选** | **J. Chem. Inf. Model.** (ACS, IF ≈ 5.6, 中科院二区) |
| **备选 1** | *Digital Discovery* (RSC) |
| **备选 2** | *Journal of Medicinal Chemistry* — 门槛更高，需要额外药化论证 |

**为什么 JCIM 而不是备选 1：** Paper A 的核心是**评价协议**（跨金属迁移这个轴在文献里空着），
不是分子性质预测精度。JCIM 的读者群正是做 QSAR/描述符/基准的人，会关心"你的协议为什么和他们
的不一样"。Digital Discovery 偏生成与自动化合成，会追问生成分支——那是 Paper B 的战场，
会审到我们 `gen_validity` 0.23–0.38 的诚实负面，等于把 Paper B 的弱点提前暴露在 Paper A 的
审稿人面前。

**已知风险：** JCIM 审稿人熟悉 MetalCytoToxDB，很可能问"为什么你的 0.7717 比 Krasnov 的 0.810
低"。这不是可以回避的问题，见 §3 的应对。

---

## 2. 时间线

| 阶段 | 内容 | 前置 | 阻塞 |
|---|---|---|---|
| **S0** | 关闭 §3 的 4 条口径矛盾 | — | 3/4 已完成（`c1f3b972c4`），**第 4 条等 bench** |
| **S1** | 起草 Paper A 正文（**新文件**，不动 `paper/main.tex`） | S0 | — |
| **S2** | 图表生成（6 张，见 §5） | S1 | — |
| **S3** | 补实验：仅当 S1 写完发现数字缺口 | S1 | GPU 排队 |
| **S4** | 内审一轮（对抗式，重点找 overclaim） | S2 | — |
| **S5** | 投 JCIM | S4 | — |
| **S6** | A 投出后，才把 Paper B 拆成第二仓库 | S5 | — |

**arXiv 预印**：现有 `paper/SUBMISSION_CHECKLIST.md` 的 W49=2026-12-09 窗口是**旧论文**的，
与 Paper A 无关。若走 arXiv，建议**等 JCIM 投稿后**再挂（S5 之后），避免审稿人看到未经同行评议的
预印本并影响判断。

---

## 3. 投稿前必须关闭的四条口径矛盾

| # | 矛盾 | 状态 | 处置 |
|---|---|---|---|
| 1 | `within_metal_cv` Ru 0.8363 与 Krasnov 0.81 队列不可比 | ✅ `c1f3b972c4` | 加 `comparability_note`；撤回理由换成两个盘上理由 |
| 2 | tie 规则文档写反（源码 `sum*2>=n` = ACTIVE） | ✅ `c1f3b972c4` | 改 2 处文档；历史产物逐字保留（§5.5） |
| 3 | `cfm_decode_real` 报 aggregate 但两条臂 0 载入 | ✅ `c1f3b972c4` | 加 `coverage_note` 限定 MEASURED_PASS 许可什么说法 |
| 4 | **bond 数字跨口径并置** | ✅ 2026-10-11 | 见下 |

### 第 4 条的处置

`PAPER_SCOPE.md:14` 写的 "`bond_type_f1` 0.9194 vs OpenBabel 0.505" **同时差三个变量**：

| | 0.9194（训练器 gate） | 0.5051（bench 的 OpenBabel） |
|---|---|---|
| 聚合 | pooled（全部 pair 汇总算一个 F1，`train_cfm...py:5010`） | per-complex（逐复合物算 F1 再平均，`bench...py:752`） |
| 模型 | n_train 26715 | n_train 256，epoch 3 |
| 口径 | 训练器 val split，192 复合物 | bench 900 复合物采样 |

代码注释（`train_cfm...py:4996`）早已写明两者"not interchangeable"，字段名的 `_pooled` 后缀
就是这个警告。

**同口径的 CFM 行已在盘上**（`tmqm_bond_perception.json::controlled_comparison_2026_10_04.table.cfm`）：
`macro_f1_type 0.2801` vs OpenBabel `0.5051`——**我们输**，且该文件自己写着 "the CFM does NOT
beat OpenBabel on either axis at this training scale"。

**正在跑的判定实验**（唯一能定案的一次）：

```bash
PYTHONPATH=/home/hugo/codes/mol-metal .venv/bin/python \
  molmetal/scripts/bench_tmqm_bond_perception.py \
  --baselines cfm,openbabel,rdkit_stripped \
  --cfm-ckpt molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \
  --cfm-decision threshold --cfm-decision-threshold 0.5 \
  --heavy-atom-scope --n-samples 150 \
  --out-dir molmetal/reports/tmqm_bond_perception_fullcorpus
```

**判定实验已跑完**（`molmetal/reports/tmqm_bond_perception_fullcorpus/`）。同一个 checkpoint
（epoch 10，n_train 26715）在 bench 口径下得分：

| | CFM（生产规模） | OpenBabel | 结论 |
|---|---|---|---|
| bond EXISTENCE F1 | **0.9961** | 0.9771 | **赢** |
| exact-type rate | **0.0978** | 0.0300 | **赢 3.3×** |
| macro-F1(type) | 0.4723 | **0.5051** | **输，落后 0.0328** |
| success_rate | 1.000 | 1.000 | 持平 |

处置：
- **0.9194 撤回**，不进论文。它是 pooled 口径在 192 复合物 val split 上的数，同一 checkpoint
  在同协议下是 0.4723——差近一倍。留在内部记录里作为口径差异的证据。
- **Table 3 用上面的四行**，含我们输的那一行。
- 规模效应是真的：同协议 n_train 256 → 26715，F1(exist) 0.6118 → 0.9961、type 0.2801 → 0.4723、
  exact-type 0.0011 → 0.0978。旧结论"at this training scale 打不过 OpenBabel"在 256 上为真，
  现在只在一半的轴上为真。

**注意这条主要属于 Paper B。** Paper A 表格里只作为 tmQM 语料/成键真值的支撑证据出现一次，
且必须带上 macro-F1(type) 输的那一行——只报赢的两项就是选择性呈现。

---

## 4. Paper A 的文件落点

**新目录，不复用 `paper/`。** `paper/main.tex` 是旧单篇叙事（λ-calculus + CFM，398 行，
§1–§7 + 4 张 fig），那是 Paper B 的材料，且 TODO-84 禁止我改它。

```
paper_a/                      # 新建，待用户 OK 后才动手
├── main.tex
├── sections/01_intro.tex     # 重写，不迁移旧的
├── sections/02_related.tex
├── sections/03_method.tex    # 队列定义 / 去重 / LOMO 协议 / tmQM 预训练
├── sections/04_results.tex
├── sections/05_ablation.tex  # tie 敏感性 + 去重影响 + 预训练 A/B
├── sections/06_limitations.tex
└── refs.bib
```

`paper/` 保持原样，等 Paper B 定稿时整体接手。

---

## 5. 图表清单（每行一个盘上来源）

| # | 类型 | 内容 | source JSON |
|---|---|---|---|
| Fig 1 | 热图 | 跨金属迁移矩阵，Os/Re 跨金属落在 0.5 以下 | `metal_activity_transfer_seed_stability.json` |
| Table 1 | 表 | within vs cross 并排，5 seed / 3 seed | 同上 |
| Fig 2 | 斜率图 | tmQM 预训练 A/B，5 seed 配对 | `tmqm_pretrain_init_ab_ru.json` |
| Table 2 | 表 | 与 Krasnov 同协议对比（**含 0.7717 vs 0.810 的负面**） | `metalcytotox_ru_auc.json::paper_protocol_measurement` |
| Fig 3 | 条形 | 去重影响（Ru 66% 重复行）+ tie 敏感性区间 | `..._seed_stability` / `..._tie_sensitivity` |
| Table 3 | 表 | 成键感知四行（900 复合物同口径）—— **已定案** | `tmqm_bond_perception.json::production_scale_comparison_2026_10_11` |

Fig 3 不是附录材料：**Ru 的 0.8363 到底算不算赢 Krasnov 0.81，取决于 tie 规则**。这必须画出来，
让审稿人看到结论的脆弱边界在哪，而不是只看到我们选的那一侧。

---

## 6. 审稿人会问的三个问题（预先备答）

**Q1. "为什么你的 0.7717 比 Krasnov 的 0.810 低？模型不行？"**
A：这是**跨队列估计**，不是复现。对方队列 23 细胞系 / 4793 复合物 / 15799 行，DOI-split 10CV；
我们用同一 protocol 但队列范围是 591 细胞系。差距 0.0383，10 折中 3 折达标，不同模型类
（D-MPNN vs LightGBM）。已在正文 §6 报告，不埋。

**Q2. "跨金属坍塌是不是去重造成的假象？"**
A：within 和 cross 用**同一队列同一 scorer**，去重在两侧同时生效；去重把 Ru 的 3668 行压到
1255 唯一配体（66% 重复），若不去重两侧都会虚高，差值的方向不受影响。Fig 3 给出两个口径。

**Q3. "Os/Re 跨金属低于随机，是 bug 还是真实现象？"**
A：3 个 seed 下 Os 0.5195 / Re 0.4429，Re 的 5 折 within 0.9480 与其 cross 0.4429 形成极端反差，
说明跨金属预测携带的是**金属特异的**信号而非通用毒性信号。源：`..._seed_stability.json`。

---

## 7. 与 Paper B 的耦合（顺序不可逆）

Paper A **显式引用** Paper B 的 encoder（`models/tmqm_pretrained/tmqm_pretrained_encoder.pt`
被 Claim 2 和两个论文共用）。共享预训练权重必须**由我们自己先说**，否则审稿人在 B 的
supplementary 里发现同一个 checkpoint，会认为存在未披露的重复发表。

因此：**A 投出后**才把 B 拆出去。A 投出前拆仓 = 隐藏共享来源。

---

## 8. 阻塞项与诚实说明

| 项 | 状态 |
|---|---|
| 第 4 条矛盾 | ✅ 已关闭并定案（0.9194 撤回，Table 3 用 900 复合物同口径四行） |
| `paper/sections/04_method.tex` §4.4 仍写旧的 +0.046 | **需用户 OK** 才改（§5.3）。5-seed × 30ep 实测是 **+0.0181** |
| CFM 3-seed 结果 | ⏳ seed1/2 跑中；与 Paper A 无关，属 Paper B |
| Paper A 补实验 | 目前**无缺口**。三张表六张图全部有盘上来源 |
| **S0 全部四条口径矛盾** | ✅ **已关闭。可以起草正文。** |

**没有为了投稿而新跑实验的需要** —— 这是查证优先带来的结果：所有数字都已在盘上，
问题一直是口径而不是数据。