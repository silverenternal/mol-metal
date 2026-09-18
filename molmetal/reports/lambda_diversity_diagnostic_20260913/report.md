# Lambda 结构多样性诊断（2026-09-13，修复前冻结基线）

原 N10 的 70 个候选实例来自 4 个不同结构：本次按原始 seeds 42/0/1234、standard12、CuAAC、4 simulations、depth1、search top10 完整复现了每种 seed 的 3/3/1 个返回结构，跨 seed 并集恰为 4。相同配置与 seed 的无口袋 reward 搜索不会随 pocket ID 改变结构。扩大 pocket 数量不会自动扩大生成分布。

## 协议与证据边界

- `diagnose_lambda_diversity.py`：3 个原始基线 + 24 个完整因子组合（2 library × 2 rule sets × 3 depth × 2 branching caps），网格 seed42，4sim/top10/patience50。24 个完成、3 个初始化失败；失败保留。每进程90秒超时，27case实际进程总时间110.45秒，无超时。RDKit CPU；无 docking、学习 prior、synthesis gate；关闭可自动发现的 leaf oracle，所有 case nfe_oracle=0。
- runner 在进程内使用观察子类；Python profiler 捕获原 search 返回前的 candidates/leaves，输出和 gates 不变。每种新结构采用去 atom map、去显式 H 的 canonical isomeric SMILES；不代表训练 novelty。
- `root_enumeration` 是搜索完成后独立穷举所选 root × 全部实际 rule/tile 的序列漏斗，明确区分 nonmatch 与 exception。`search_observations` 包含 expansion/rollout 的重叠调用；不能把其调用次数当独立分子分母。
- typed=原 LIPINSKI；binding=原 PROTEASE_GENERIC 的属性/供受体约束，未调整阈值，也不等于真实 pocket 亲和力。所有报告保留逐结构拒绝理由。
- 实际 final top-k 前后的数量直接来自原函数局部变量。原物理评价另有 top3，本次 N10 每 seed 最多3个，所以两个 top-k 都未造成其4种结构瓶颈。
- `summary.json` 存配置、运行前源码 SHA256，并确认所列源在运行期间不变；`proof_search_before.py` 保留原搜索源码。首次 pilot 的观察子类初始化错误保留在独立 `_pilot` 目录，不计入27case；修正后 `_pilot_v2` 三case完成。

## 计数

下表网格三种深度（1/2/3）的结构数一致，表中合并显示。all5 改变 seed rule 抽样与所选 root，因此不是固定 root 的单因子因果估计。

|配置|实际tile/尝试|根新结构|typed通过|随后binding通过|top10返回|截断|
|---|---:|---:|---:|---:|---:|---:|
|standard_12/CuAAC/cap12/seed42|12/12|3|3|3|3|0|
|standard_12/CuAAC/cap12/seed0|12/12|3|3|3|3|0|
|standard_12/CuAAC/cap12/seed1234|12/12|3|3|1|1|0|
|standard_12/CuAAC/cap60/seed42|12/12|3|3|3|3|0|
|standard_12/CuAAC/cap1020/seed42|12/12|3|3|3|3|0|
|standard_12/all_5/cap60/seed42|12/60|7|7|2|2|0|
|standard_12/all_5/cap1020/seed42|12/60|7|7|2|2|0|
|extended_204/CuAAC/cap60/seed42|60/60|10|10|4|4|0|
|extended_204/CuAAC/cap1020/seed42|220/220|54|54|15|10|5|
|extended_204/all_5/cap60/seed42|12/60|初始化失败|—|—|0|—|
|extended_204/all_5/cap1020/seed42|204/1020|10|10|1|1|0|

## 主要瓶颈与反证

1. 原 standard12/CuAAC 每个 root 仅3种反应产物。12尝试分为3 feasible、7 nonmatch、2 RDKit implicit-valence异常；safe_reduce合并异常为空，独立穷举保留异常。seed42/0两门控全过；seed1234的3种中2种缺H-bond donor被binding拒绝。原N10没有top-k截断。
2. `extended_204` 实际返回220tiles。CuAAC/cap60保留60，cap1020保留220；all5/cap60只保留prefix12，使所有3depth初始化失败（51 witness校验均无成功）。all5/cap1020保留204，最后16未进入扩展池。库名和branching标签不可当实际规模。
3. 本次大库CuAAC给出54种新结构、15种binding通过，search top10丢5种；大库all5只选到Suzuki root，18 feasible pairs给出10个不同结构，9个未满足原binding药效团要求，仅1个返回。增加规则数量本身不保证增加可接受结构。
4. 初始看到最终树均深度1，曾推断βNF终止是本次网格主因；后续 `terminal_reactivity/` 直接检查实际leaf state反证了该推断：3种standard CuAAC、54种extended CuAAC、10种extended all5 leaf均βNF=False。CuAAC两库的所有leaf在原有有序反应方向下都没有后续可行partner；大库CuAAC depth3对同一无反应leaf重复穷举4次，合计1108 reductions。此处实际问题包含单向反应空间和无反应缓存缺失，不应把βNF当本次网格已证主因。
5. all5有4个post-search继续反应见证，但均来自ThiolEne且产物为断开的两片段；这不能作为有效共价生长或真实多样性改善。已交规则owner修复，并保留原产物/partner。只有部分叶被4sim访问，不能因post-search witness存在便断言原budget内会走到该叶。
6. 另行构造的真实反例证明βNF判终止本身有bug：βNF=True的二羧酸 `O=C(O)CC(=O)O` 可与 `CN` 连续AmideCoupling为 `CNC(=O)CC(=O)NC`，而原无Dirichlet search零次reduction。此反例独立于上述网格主因。合成反向边和diamond也分别复现TT循环/parent覆写。新增5项回归修前全失败，日志 `search_regression_before_v2.log`。

## 后续修复与验证顺序

- 保留所有 typed/binding 门控，先修明确RDKit/ThiolEne化学正确性问题及搜索终止/空缓存/TT树结构，再冻结新版本重跑同网格。新版本结果不得覆写此基线。
- 以实际反应handle分层采样扩展池，避免prefix截断删除整类反应；为多步增长规划可继续反应的多功能tile，明确双分子反应方向与去重。不要仅提高depth/cap标签。
- 先在development数据预先声明多root/seed、预算和top-k的消融方案；新增反应库/初始化会同时改变root分布，需固定root比较或明确作为整套配置比较。
- 能够产生足够有效不同结构后，才扩展冻结的真实pocket reward on/off和100pocket评价；不以此廉价门控结果宣称docking、novelty、药效或SOTA提升。
