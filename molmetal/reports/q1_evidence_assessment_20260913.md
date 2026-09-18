# 一区论文实验与证据评估（2026-09-13）

结论：当前为有真实数据、真实GPU执行和可追溯输出的研究原型；尚未形成支持一区方法论文主要结论的完整证据链。不存在统一的Q1指标门槛，JCR Q1和中科院一区也不是同一个分类。本评估不核定期刊当年分区/影响因子，也不以代码覆盖率推算投稿完成百分比。

## 已测结果及证据边界

| 任务 | 已测结果 | 不能推出的结论 |
|---|---|---|
| GPU物理评价 | 原始前10 test pairs×3 seeds，70候选实例、4个不同结构；63真实docking/PB通过，7因原受体缺原子未测，coverage63/70 | 70个独立新分子、多样性充足、完整100-pocket协议 |
| GPU分数 | 有限63观测均值−5.1365 kcal/mol；9个redocked-reference/SA/QED联合成功，低于−8的0个 | 与其他引擎/预算或文献均值直接比较得到SOTA优劣 |
| 参考配体 | 27可redock任务中9次参考pose PB失败；reference-PB-passing子集有8次联合成功 | 忽略参考质量后把所有9次成功当强亲和力证据 |
| 实际闭环反馈 | 最新test001/seed42中3次真实GPU docking、7次缓存命中，实测能量进入MCTS | 1口袋4sim说明闭环有效提升；还需on/off同预算对照 |
| 模型OT | 修复图间串扰后，QM9 train64/val32、每组30更新、3seed，paired loss delta−0.069303±0.021093；90/90 GPU耦合收敛 | 新分子validity、3D稳定性、亲和力提升或大规模泛化 |
| 线性prior | development32真实反应产物，冻结3seed off/on，真实PUCT调用但reward delta全0 | prior有增益；该模型是PySR符号回归 |
| Learned环境 | REINVENT官方RNN NLL真实AMD；AiZynth官方legacy权重/模板/库存真实GPU策略与路线 | 模型NLL是药效；3个smoke靶标路线成功率可代表生成分布 |
| 历史tmQM | 21,615样本，random holdout2161，Wiberg BO MAE.176/R².9239；CN97% exact，简单图degree基线92% | CN高R²证明量子化学发现；预训练必然改善下游生成或活性 |
| 历史Ru pIC50 | random test200，Pearson.407、Spearman.372、RMSE.724；细胞系/时点混合 | 可精确预测新骨架抗癌活性，或与不相同任务的文献指标比较 |

## 最影响投稿的数据缺口

1. **生成分布过窄且未证明口袋适应。** 4种结构在10口袋间反复评价，不能支撑丰富的de novo化学空间。应记录每口袋unique/scaffold覆盖、结构重复和失败分母，执行相同初始化/反应库/预算的真实闭环on/off。扩大到100口袋前先查反应、binding gate、深度、top-k各阶段怎样限制分布。
2. **主要创新的因果证据缺失。** prior小试零提升；OT只有loss；CFG/metal几何与真正分子质量/亲和力消融尚待完成。模型与Lambda分别验证，所有处理共享数据/seed/训练预算，不能每种配置换训练seed、只展示成功样本或把未实际调用的prior当生效。
3. **基线不公平或不足。** 补同口袋重测的简单随机反应/无prior搜索/等预算搜索与可运行公开模型；同引擎、受体、box、pose/PB协议。文献聚合均值只做背景，不做one-sample t-test。统计以口袋为单位，保留同口袋多seed依赖。
4. **数据拆分与novelty未闭合。** 补训练集最近邻、scaffold novelty、完整配体/重复测量分组，检查盐型、金属中心和同文献重复泄漏；测试集不能用来拟合prior或选择阈值。已暂时跑通的split检查不等于所有下游数据都无泄漏。
5. **金属抗癌主张明显超出现有证据。** 当前蛋白docking/SA/QED/PB不能证明Pt/Ru/Ir配位稳定性、DNA共价结合、GSH竞争、水解/氧化还原或细胞选择性。需要真实相关金属结构/任务和独立参考。普通Vina缺少经验证的Pt参数时不能制造Pt亲和力；Fe-heme口袋不能当Pt方平面先验的证据。
6. **预测器需要重新校准与外推评估。** Ru活性混合assay的随机切分结果弱，应分细胞系/暴露时长、合理合并重复，报告scaffold/temporal/ligand-dedup测试、简单基线和不确定性。不能以其他数据集的r或分类AUC作为合格线。
7. **规模和效率证据不足。** 原N10为4sim/depth1/低docking预算。 full100×3、预先声明预算与覆盖、端到端wall time/峰值显存尚需完成；本AMD驱动OpenCL event时间戳无效，不能直接宣称GPU加速倍数。

## 推荐推进顺序

先收敛当前runtime和回归，再完成多样性诊断及N10真实闭环匹配消融；随后扩大完整100×3并补基线、训练novelty和口袋级CI。模型线同步执行固定checkpoint的真实CFG评价和独立金属几何验证。历史tmQM结果可以成为有价值章节，但必须以严格外推split和下游迁移消融支撑贡献。

计算方法论文可以在无湿实验时讨论受控计算任务表现；如果主结论是发现有效金属抗癌候选，当前缺少的活性/选择性/机制证据不能由继续堆砌docking指标代替。当前目标继续保留金属任务，不因收窄论文措辞就把其待办判为完成。

## 主要证据文件

- `r4_click_gpu_test10_seed3_analysis_v2.{json,md}`（恢复PDBQT只在恢复时hash的限制见源JSON）
- `r4_measured_reward_test001_seed42.json`
- `model_import_batch_sampling_repairs_20260913.md`
- `linear_prior_development_ablation_20260913/report.md`
- `reinvent_prior_adapter_amd/report.json`
- `aizynth_real_backend_20260913/rocm_learned_smoke.json`
- `f2_tmqm_pretraining.md`、`h2_pic50_predictor_calibration.md`（历史结果，需依照上述边界解释）

## 后续实测更新

真实GPU奖励控制已完成2口袋×3seed，零权重和0.4权重均实际调用oracle。
两组的真实调用次数逐组相等，14个候选实例全部物理评价完成；生成集合
逐组相同，top1实测能量平均变化0。这是保留的负面消融，进一步说明
只接入真实打分还不够，需定位候选空间/搜索预算限制；不能写成闭环增益。
证据：`measured_reward_control_20260913_v2/comparison.{json,md}`。

## 活性标签审计更新

`pic50_assay_audit_20260913/report.json`重放旧Ru数据筛选顺序，发现旧2000条
样本中356条原始dark IC50含比较界限（17.8%），不应作为精确回归值；Ru全量
19,135条正数记录中3,864条有此问题。4,833个canonical结构中3,984个跨细胞系，
395个跨暴露时长，1,142个结构的“首条/末条”pIC50相差超过0.5。因此，原
Pearson0.407不仅是模型容量问题，也有目标定义/观测删选问题；不能把跨assay
差异都称为实验重复噪声。旧checkpoint保留，新训练必须另立条件化/删失值协议。
已导出15,091条无明确删失符号且细胞系/时长已知记录，仍不是可直接随机切分
的训练集。最大单一条件队列HeLa48h有695个不同结构，A54948h有634个。

## 最新核验：模型、novelty与条件化活性

- **模型线仍未完成有效分子生成。** 等变速度场v2修复后，32个真实训练配体、每seed 2000更新，2测试口袋×3seed×2CFG×8样本，共96次请求均得到有限坐标，但0个成功解码（92个断图、4个含训练词表外元素），因此没有下游docking/CFG增益证据。当前无learned bond head，口袋仅全局不变量pool，缺少配体相对受体的方向消息。这是建模缺口，不能靠扩大评估样本或删除坏原子处理。证据：`r10_cfg_real_crossdocked_v2_train32_2000/report.json`。
- **训练新颖性已完成，但多样性缺口仍在。** 完整100000训练SDF去重为8765结构；现有4个生成结构均无训练graph/scaffold重合，最近Tanimoto为0.323–0.400，真实GPU结果与RDKit全量对照一致。新颖性只针对该训练库；4个结构不能支撑广泛化学空间结论。官方口袋split存在训练/测试配体结构重合，应另报结构不重合子组，不能仅凭此判定官方任务泄漏。证据：`training_novelty_20260913/report.md`。
- **活性已有更可信的简单基线。** HeLa/48h/dark无明确删失队列，702种formulation、383个scaffold，3组scaffold切分GPU ridge测试RMSE为0.597/0.612/0.415，Pearson为0.477/0.606/0.634，各组优于对应训练均值预测器。与旧混合assay指标不能直接比较；尚缺神经模型同split重训、文献/时间外推、其他细胞系和不确定性。证据：`pic50_conditioned_baseline_20260913/report.md`。
- **金属几何只有局部正证据。** 8个真实tmQM Pt CN4结构×3seed配对控制，donor-vector RMSE平均降低约0.098 Å、对DFT参考的角度MAE降低约6.08°；属于固定图几何控制，不等于新金属分子生成、Pt亲和力或抗癌活性。证据：`r10_tmqm_geometry_control/report.json`。
- ThiolEne连键与搜索终止/树统计正在分别完成修复和受控复测；旧物理实验和零增益消融继续保留，不把新代码修复写成已测性能提升。活性Attentive DMPNN的batch padding依赖已修复，相关8项测试通过，历史checkpoint仍需重新训练/校准。

当前投稿优先级：先让模型产生可验证分子、让Lambda在匹配预算下展示实际多样性与优化收益；再完成100口袋×3seed、同协议公开/简单基线及口袋级置信区间；金属抗癌主张另补适用的物理与独立活性证据。主要瓶颈已转向科学有效性，而非GPU环境是否可用。
