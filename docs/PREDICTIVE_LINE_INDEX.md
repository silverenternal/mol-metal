# PREDICTIVE LINE — 单一索引（single source of truth）

**建立**：2026-10-21（用户指令「把预测线的文档整理一下，不要让项目那么乱」）
**作用**：预测线（MSRS 跨金属 pIC50 / Krasnov / chemprop / D-MPNN）每个数字只有一个权威出处。
**规则**：本文档里的每个数字都由 `molmetal/tests/test_predictive_line_index.py` 对着 artifact 校验；
改了 artifact 不改这里 → 测试红。**引用前先看 §3 撤回日志**——预测线最容易犯的错是引用一个已经撤回的数。
**不覆盖**：生成线见 `cfm_todo.json`；论文主张台账见 `docs/CLAIM_LEDGER.md`。

---

## §1 一句话结论

- **跨论文绝对强度预测过关**：整篇 DOI 留出 R² **0.3767**（5 seed，MACCS+RDKit 树）/ **0.3811**（+GNN 融合），门槛 0.36。
- **"换哪个金属更好"点估计不可学、balanced 口径可学**：门内 218 对的多数类率 **0.6009**，三条点估计路线（相减 0.5826 / 融合 0.5505 / 直接回归 0.5183）**全部低于恒定预测**；但同一批折改 listwise 目标后 balanced **0.5985 [0.5262, 0.6663]**，首次显著优于 0.5（raw 仍只有 0.5417）。**这是"方向学到了、精度没到"，不是"可用了"。** 详见 §2.7。
- **跨金属迁移不到可用水平**：可入闸的 Ru/Ir LOMO **0.5974** < 0.6；within-metal 稳定超过 Krasnov（Ru 0.8271 vs 0.81，Ir 0.8430 vs 0.73）。
- **加性约束不是 metal_out 的瓶颈**（UP-1，2026-10-21）：放开 ligand×metal 交互三个臂全部**更差**（free −0.097 / pair −0.113 / product −0.164 R²），最好的 free_onehot 只 +0.0204 且 CI 含 0。**方差分解里 0.712 的交互份额，在留出金属上不可外推。**
- **校准的失败层是 cohort size，不是金属**（UP-6）：`cohort_metal` Mondrian 把最差层 0.8096 → **0.8956**，中位宽只涨 0.65%（2.0052 → 2.0183）。

按 `todo.json::north_star.decision_rule`（门槛在看到结果前写死）：4 条里过 2 条 → **边界论文**。
UP 轮之后仍是 2/4：UP-6 是**已过项的内部加固**，UP-1 是**新增诚实负结果**，UP-2b 是**部分翻案但未过门**。

---

## §2 权威数字

### 2.1 四协议 × 三臂（5 seed）

| 协议 | ECFP4 树 | MACCS+RDKit 树 | +GNN 融合 |
|---|---:|---:|---:|
| ligand（**内插，不可当泛化报**） | 0.5596 | 0.5593 | 0.5935 |
| **doi（判据协议）** | 0.3380 | **0.3767** | **0.3811** |
| scaffold | 0.4530 | 0.4733 | 0.4903 |
| year | 0.3780 | 0.3946 | 0.4011 |

DOI 相对 ECFP4 的配对增益 **+0.0431 [+0.0260, +0.0604]**（DOI 聚类配对 bootstrap，2000 次）。
表：`metrics/paper_tables/T11_generalization_strong.md`

### 2.2 金属感知表示（C1）

| 臂 | R² | Δ vs full | p |
|---|---:|---:|---:|
| full（配合物+态+配位+金属码） | 0.5556 | — | — |
| metal_blind（去掉金属相关一切） | 0.4873 | **0.0683** | 0.0099 |
| no_state | 0.5284 | 0.0272 | <0.001 |
| no_coord | 0.5426 | 0.0130 | 0.029 |
| no_metal_code（只去身份码） | 0.5492 | 0.0064 | 0.005 |

全配合物表征 vs 配体单片段：**+0.1002 R²（p=2.3e-20）**。
**不能写"架构创新"**——增益来自输入里有金属。

### 2.3 排序（C3，DOI 留出，|Δ|≥0.2091，218 对）

| 路线 | raw sign | balanced | 95% CI (balanced) |
|---|---:|---:|---|
| 水平相减 | 0.5826 | 0.5272 | [0.4359, 0.6198] |
| 融合 | 0.5505 | — | — |
| Δ 直接回归 | 0.5183 | 0.5162 | [0.3939, 0.6430] |
| **恒定预测（多数类）** | **0.6009** | 0.5 | — |

门内只有 3/10 个金属对：Os–Ru 153、Ir–Rh 64、Rh–Ru 1。**五金属排序在本队列上不可识别。**

### 2.4 校准（C5）

DOI 留出、α=0.10 split conformal：覆盖 **0.9004**，中位宽 **2.0052** log10。
同队列多金属子群（n=851）覆盖只有 0.8096 —— 必须单独报。

### 2.5 跨金属迁移（C6）

| | within-metal（5 seed 均值） | Krasnov 2026 | LOMO |
|---|---:|---:|---:|
| Ru（n_unique 1,311） | **0.8271** | 0.81 | 可入闸 |
| Ir（n_unique 538） | **0.8430** | 0.73 | 可入闸 |
| Rh / Os / Re（51 / 36 / 74） | 0.6974 / 0.4960 / 0.9480 | — | 低于 `MIN_UNIQUE_FOR_GATE=200`，**不得入闸** |

可解析子集 LOMO：**0.5974**（seed 集成 0.5947），目标 0.6，两个 CI 都含 0.6。
**不能写"差 0.04"** —— 那是把三个不可解析的折等权平均进去。

**LOMO 有五个 artifact，只有一个能引：**

| artifact | value | 是什么 | 能不能引 |
|---|---:|---|---|
| `metal_lomo_auc.json` | 0.5346 ± 0.1096 | `train_metal_lomo.py`，5 折 × 3 seed 原始 m19 | 历史值，已被后续覆盖 |
| `metal_lomo_auc_v2.json` | `null`（`DEFERRED_PARTIAL_RUN`） | `train_metal_lomo_v2.py`，平衡 Rh 队列重测，**没跑完** | **不可引** |
| `metal_lomo_auc_wasserstein.json` | 0.5575 ± 0.0949 | `train_metal_lomo_wasserstein.py`；`real_only_fold_value` = 0.3812 | 只能引可解析子集口径 |
| `lomo_resolvability_audit.json` | **0.5974** | 把 Rh/Os/Re 三个低于 `MIN_UNIQUE_FOR_GATE=200` 的折剔除后重聚合 | **这是可引的那个** |
| `lomo_seed_ensemble.json` | 0.5947 | 同一可解析子集，但按「先平均分数再算 AUC」（部署口径） | 可引，作为 0.5974 的稳健性 |

**已知冲突（未修，需要裁定）——不是「pin 过时」，是两个不同的估计量：**

| 出处 | 数 | 估计量 |
|---|---:|---|
| `configs/locks/protocol_lock.yaml::m19_lomo` | — | 锁定协议写的是 `target_metric: mean test AUC over 5 metals`，`n_folds: 5` |
| `molmetal/scripts/registry.py::LINE_METRICS` → `metal_lomo_auc_wasserstein` | **0.5575** | 5 个金属**等权**平均（含 Rh 51 / Os 36 / Re 74 三个低于 `MIN_UNIQUE_FOR_GATE=200` 的折） |
| `docs/CLAIM_LEDGER.md`（STR-7 / FIX-C6）→ `lomo_resolvability_audit` | **0.5974** | 只对可入闸的 Ru(1,311) / Ir(538) 平均 |

所以 `registry.py cite` 说引 0.5575、台账说引 0.5974，两边都符合各自的定义。**以本索引和台账为准**
（0.5974），因为把三个测量地板上的折等权平均进去会低估。但要真正消除冲突，需要改
`protocol_lock.yaml` 的 `target_metric` 定义，而不是只改一个 pin——那会改变锁定协议，
本次没有动，留作待裁定项。

### 2.6 机器校验块

测试从下面这个代码块读取，逐条比对 artifact。**不要手改数值，改 artifact 后同步这里。**

```json predictive-line-index
[
  {"id": "G.ligand.ecfp4", "value": 0.5596, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.ligand.ensemble.r2"},
  {"id": "G.doi.ecfp4", "value": 0.3380, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.doi.ensemble.r2"},
  {"id": "G.doi.maccs", "value": 0.3767, "artifact": "metrics/by_metric/msrs_feature_expansion_5seed.json", "path": "protocols.doi.r2_seed_averaged"},
  {"id": "G.doi.fused", "value": 0.3811, "artifact": "metrics/by_metric/msrs_feature_fusion.json", "path": "protocols.doi.fused_r2"},
  {"id": "G.doi.fusion_delta", "value": 0.0431, "artifact": "metrics/by_metric/msrs_feature_fusion.json", "path": "protocols.doi.vs_tree.delta_r2"},
  {"id": "G.doi.fusion_ci_low", "value": 0.0260, "artifact": "metrics/by_metric/msrs_feature_fusion.json", "path": "protocols.doi.vs_tree.ci_low"},
  {"id": "G.scaffold.ecfp4", "value": 0.4530, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.scaffold.ensemble.r2"},
  {"id": "G.year.ecfp4", "value": 0.3780, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.year.ensemble.r2"},
  {"id": "C1.complex_delta", "value": 0.1002, "artifact": "metrics/by_metric/msrs_complex_features.json", "path": "splits.fragment_grouped.complex.delta_r2_vs_main"},
  {"id": "C1.full", "value": 0.5556, "artifact": "metrics/by_metric/msrs_metal_ablation.json", "path": "protocols.ligand.arms.full.r2"},
  {"id": "C1.metal_blind", "value": 0.4873, "artifact": "metrics/by_metric/msrs_metal_ablation.json", "path": "protocols.ligand.arms.metal_blind.r2"},
  {"id": "C1.metal_blind_delta", "value": 0.0683, "artifact": "metrics/by_metric/msrs_metal_ablation.json", "path": "protocols.ligand.arms.metal_blind.delta_r2_vs_full"},
  {"id": "C3.majority_class", "value": 0.6009, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "baselines.majority_class_rate"},
  {"id": "C3.diff_raw", "value": 0.5826, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.level_differencing.raw_sign"},
  {"id": "C3.diff_balanced", "value": 0.5272, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.level_differencing.balanced"},
  {"id": "C3.direct_raw", "value": 0.5183, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.direct_regression.raw_sign"},
  {"id": "C3.direct_balanced", "value": 0.5162, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.direct_regression.balanced"},
  {"id": "C5.coverage", "value": 0.9004, "artifact": "metrics/by_metric/msrs_calibration.json", "path": "protocols.doi.variants.global.coverage"},
  {"id": "C5.median_width", "value": 2.0052, "artifact": "metrics/by_metric/msrs_calibration.json", "path": "protocols.doi.variants.global.median_width"},
  {"id": "C6.lomo_resolvable", "value": 0.5974, "artifact": "metrics/by_metric/lomo_resolvability_audit.json", "path": "aggregates.resolvable_only.value"},
  {"id": "C6.lomo_ensemble", "value": 0.5947, "artifact": "metrics/by_metric/lomo_seed_ensemble.json", "path": "aggregates.resolvable_only.auc_of_mean_score"},
  {"id": "C6.lomo_raw_v1", "value": 0.5346, "artifact": "metrics/by_metric/metal_lomo_auc.json", "path": "value"},
  {"id": "C6.lomo_wasserstein", "value": 0.5575, "artifact": "metrics/by_metric/metal_lomo_auc_wasserstein.json", "path": "value"},
  {"id": "C6.lomo_wasserstein_real_only", "value": 0.3812, "artifact": "metrics/by_metric/metal_lomo_auc_wasserstein.json", "path": "real_only_fold_value"},
  {"id": "C6.ru_within", "value": 0.8271, "artifact": "metrics/by_metric/metal_activity_transfer_seed_stability_v2.json", "path": "within_metal_stability.Ru.mean", "tol": 0.0002},
  {"id": "C6.ir_within", "value": 0.8430, "artifact": "metrics/by_metric/metal_activity_transfer_seed_stability_v2.json", "path": "within_metal_stability.Ir.mean", "tol": 0.0002},
  {"id": "C7.sa_le_4p5_full_corpus", "value": 0.0028, "artifact": "metrics/by_metric/c7_scale_matched_pools.json", "path": "by_pipeline.generation.by_tier.full_corpus.rate_sa_le_gate"},
  {"id": "C8.dominant_share", "value": 0.7052, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "dominant_term.mean_share"},
  {"id": "C8.offset_share", "value": 0.1002, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_offset_share"},
  {"id": "C8.scale_share", "value": 0.2060, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_scale_share"},
  {"id": "C8.shape_share", "value": 0.6938, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_shape_share"},
  {"id": "C8.ceiling_r2", "value": 0.1524, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_ceiling_r2"},
  {"id": "C8.rho_metal_out", "value": 0.2960, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.pooled.centred.pooled_spearman"},
  {"id": "C8.rho_ligand_disjoint", "value": 0.1575, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out_ligand_disjoint.pooled.centred.pooled_spearman"},
  {"id": "C8.metals_above_gate", "value": 2, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.n_metals_above_threshold"},
  {"id": "C8.perfect_level_gain", "value": 0.0784, "artifact": "metrics/by_metric/msrs_metal_descriptor_level.json", "path": "counterfactual_summary.mean_gain_over_cells"},
  {"id": "C8.metals_still_negative_after_level", "value": 4, "artifact": "metrics/by_metric/msrs_metal_descriptor_level.json", "path": "counterfactual_summary.max_metals_still_negative"},
  {"id": "C10.k50_metals_nonnegative", "value": 0, "artifact": "metrics/by_metric/msrs_kshot_calibration.json", "path": "endpoints.potency.splits.metal_out_ligand_disjoint.verdict_at_gate_k.n_metals_r2_nonnegative"},
  {"id": "C9.ir_doi_overlap", "value": 0.9955, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.doi_overlap_share"},
  {"id": "C9.ir_new_dois", "value": 1, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.n_new_dois"},
  {"id": "C9.ir_identical_rows", "value": 0.8943, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.row_agreement.share_within_0p01"},
  {"id": "C9.ir_median_abs_diff", "value": 0.0, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.row_agreement.median_abs_diff_log10"},
  {"id": "C9.unmatchable_rows", "value": 0.2194, "artifact": "metrics/by_metric/msrs_cell_feature_probe.json", "path": "identity_mismatch.unmatchable_row_share"},
  {"id": "UP1.reference_r2", "value": -0.0719, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.reference_weighted_r2"},
  {"id": "UP1.free_desc_delta", "value": -0.0971, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_desc.delta_r2_weighted"},
  {"id": "UP1.pair_desc_delta", "value": -0.1128, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.pair_desc.delta_r2_weighted"},
  {"id": "UP1.product_desc_delta", "value": -0.1637, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.product_desc.delta_r2_weighted"},
  {"id": "UP1.free_onehot_delta", "value": 0.0204, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_onehot.delta_r2_weighted"},
  {"id": "UP1.free_onehot_ci_low", "value": -0.0154, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_onehot.bootstrap.ci95.0"},
  {"id": "UP1.free_onehot_ligand_disjoint", "value": -0.0202, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_onehot.delta_r2_ligand_disjoint"},
  {"id": "UP2a.ligand_cell_n_pairs", "value": 360, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.overall.n_pairs"},
  {"id": "UP2a.ligand_cell_raw", "value": 0.5333, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.overall.raw_sign_accuracy"},
  {"id": "UP2a.ligand_cell_balanced", "value": 0.4705, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.overall.balanced_accuracy"},
  {"id": "UP2a.ligand_cell_ceiling_raw", "value": 0.6222, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.additive_ceiling.raw_sign_accuracy"},
  {"id": "UP2b.ligand_cell_raw", "value": 0.5417, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.overall.raw_sign_accuracy"},
  {"id": "UP2b.ligand_cell_balanced", "value": 0.5985, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.overall.balanced_accuracy"},
  {"id": "UP2b.ligand_cell_balanced_ci_low", "value": 0.5262, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.bootstrap.balanced_ci95.0"},
  {"id": "UP2b.ligand_cell_balanced_ci_high", "value": 0.6663, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.bootstrap.balanced_ci95.1"},
  {"id": "UP2b.strict_balanced", "value": 0.5055, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.strict.overall.balanced_accuracy"},
  {"id": "UP6.reference_worst_stratum", "value": 0.8096, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "design.reference.worst_stratum"},
  {"id": "UP6.cohort_metal_coverage", "value": 0.8996, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.cohort_metal.coverage"},
  {"id": "UP6.cohort_metal_worst_stratum", "value": 0.8956, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.cohort_metal.worst_stratum.coverage"},
  {"id": "UP6.cohort_metal_width", "value": 2.0183, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.cohort_metal.median_width"},
  {"id": "UP6.width_change", "value": 0.0065, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "gate.width_change_vs_reference"},
  {"id": "UP6.metal_worst_stratum", "value": 0.8331, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.metal.worst_stratum.coverage"},
  {"id": "UP3.explicit_metal_share", "value": 0.0, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.explicit_metal_share"},
  {"id": "UP3.multi_fragment_share", "value": 1.0, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.multi_fragment_share"},
  {"id": "UP3.organic_embed", "value": 1.0, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.organic_embed_success"},
  {"id": "UP3.complete_embed", "value": 0.065, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.complete_embed_success"},
  {"id": "UP3.legal_embed", "value": 0.25, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.legal_embed_success"},
  {"id": "UP3.needs_carbanion", "value": 0.335, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.needs_aromatic_carbanion"},
  {"id": "UP3.needs_aromatic_n", "value": 0.84, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.needs_aromatic_n"},
  {"id": "UP3b.hetero_r2", "value": 0.2504, "artifact": "metrics/by_metric/msrs_dmpnn_donor_ab.json", "path": "results.hetero.r2"},
  {"id": "UP3b.carbanion_r2", "value": 0.2454, "artifact": "metrics/by_metric/msrs_dmpnn_donor_ab.json", "path": "results.hetero_carbanion.r2"},
  {"id": "UP3b.delta_r2", "value": -0.005, "artifact": "metrics/by_metric/msrs_dmpnn_donor_ab.json", "path": "delta_r2"},
  {"id": "UP4.potency_metal_out_raw", "value": -0.0734, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.weighted_r2_raw"},
  {"id": "UP4.potency_metal_out_erac", "value": -0.0744, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.weighted_r2_erac"},
  {"id": "UP4.potency_ligdisj_raw", "value": -0.15, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out_ligand_disjoint.weighted_r2_raw"},
  {"id": "UP4.potency_ligdisj_erac", "value": -0.1739, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out_ligand_disjoint.weighted_r2_erac"},
  {"id": "UP4.cis_metal_out_raw", "value": -0.0793, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.cisplatin_ratio|metal_out.weighted_r2_raw"},
  {"id": "UP4.cis_metal_out_erac", "value": -0.0904, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.cisplatin_ratio|metal_out.weighted_r2_erac"},
  {"id": "UP4.pooling_none", "value": -0.0806, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.pooling_spectrum.none"},
  {"id": "UP4.pooling_one_hot", "value": -0.0708, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.pooling_spectrum.one_hot"},
  {"id": "UP4.sign_p", "value": 0.5034, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "sign_test.p_two_sided"}
]
```

### 2.8 UP-3 探针：3D 配位图在这份数据上做不出来（2026-10-21）

`msrs_complex3d_probe.json`，200 个金属分层抽样的 (配体, 金属) 对：

| 量 | 值 | 含义 |
|---|---:|---|
| `explicit_metal_share` | **0.000** | 配体 SMILES 里**根本没有金属原子** |
| `multi_fragment_share` | **1.000** | 每个"配合物"都是多点断开串（中位 3 段，平均 2.23 个带电荷辅片段） |
| `organic_embed_success` | 1.000 | 有机片段能嵌入 —— 但片段之间不相连，相对几何完全不受约束 |
| `legal_embed_success` | 0.250 | 只连 RDKit 能表示的给体（非芳香杂原子 + 卤素）时的成功率 |
| `complete_embed_success` | **0.065** | 连上真正会配位的给体（含芳香 N 0.84 / Cp、芳基碳负离子 0.335）后 |
| `complete_fail_graph_rejected` | **0.935** | 失败全部发生在 sanitize（kekulize）阶段，不是嵌入阶段 |

**判据用的是 complete 臂**：一个把 Cp / 吡啶给体丢掉的构造，描述的不是表里那个物种。
⇒ `ORGANIC_ONLY_FALLBACK`：退回「2D 图 + 显式金属节点 + 配位键」，而这条退路**已经存在**
（`molmetal/data/metal_graph.py:303-310`），就是 `msrs_dmpnn_surface.json` 的 `metal_state` 臂。

### 2.7 UP 强化轮（2026-10-21，四臂全部预注册门槛）

三个 artifact 的门槛都写在 `todo.json::gates_frozen_at = 2026-10-21T15:10+08:00`，**在看到任何结果之前**。

**UP-1 交互约束消融**（`msrs_interaction_ab.json`，6 臂 × 5 金属 × 2 split × 2 endpoint，单 seed 筛选 + 5 seed 确认）：

| 臂 | 约束 | ΔR² vs additive_desc（potency/metal_out） | 95% CI |
|---|---|---:|---|
| additive_desc（参照，= 生产模型） | `[[lig],[met],[cell]]` | — | — |
| free_desc | 无 | **−0.0971** | [−0.1438, −0.0682] |
| pair_desc | `[[lig,met],[cell]]` | **−0.1128** | [−0.1674, −0.0814] |
| product_desc | 无（显式乘积特征） | **−0.1637** | [−0.3715, −0.0530] |
| free_onehot | 无 | **+0.0204** | [−0.0154, +0.0440] |

参照臂 R² = **−0.0719**，与 `msrs_lomo_surface.json` 的 Ru descriptors −0.0777 同源，**说明参照臂复现了生产模型**，Δ 是可信的。
判据（best ≥ +0.05 且 CI 下界 > 0 且 ligand-disjoint 同号）**三项全不满足** ⇒ `CONSTRAINT_IS_NOT_THE_LEVER`。
**关键机制**：留出金属上，`additive_onehot`（−0.0708）与 `additive_desc`（−0.0719）几乎相同 —— 金属描述符块在 metal_out 上贡献 **0.0011 R²**。放开交互只会把训练金属之间的差异当信号外推，方差涨得比偏差掉得快。

**UP-2a 排序重测**（`msrs_ranking_expanded.json`，不重训，只重算已有 5-seed OOF）：

| 配对键 | n 对 | raw | balanced | 加性上限 raw | balanced |
|---|---:|---:|---:|---:|---:|
| strict（ligand, cell, doi, time_h） | 218 | 0.5826 | 0.5272 | 0.6055 | 0.5 |
| ligand_cell（ligand, cell） | **360** | 0.5333 | **0.4705** | 0.6222 | 0.5 |

放宽配对键把 n 从 218 提到 360，**结论更差**：balanced 0.4705 低于 0.5。**加性上限的 balanced 恒为 0.5**，所以任何"balanced > 0.5"都是非加性信号 —— 这条对照基线是判据的核心。

**UP-2b listwise 排序**（`msrs_listwise_ranker.json`，LightGBM `lambdarank`，group = (ligand, cell)，DOI 分组 5 折，同一批对）：

| 配对键 | n 对 | raw | balanced | 95% CI (balanced) |
|---|---:|---:|---:|---|
| strict | 218 | 0.4312 | 0.5055 | — |
| ligand_cell | 360 | 0.5417 | **0.5985** | **[0.5262, 0.6663]** |

**这是本线第一次有 balanced 显著高于 0.5 的排序结果**（pointwise 同批对只有 0.4705，+0.128）。
但预注册三条要求同时满足：`balanced ≥ 0.58` ✓、`CI 下界 > 0.52` ✓、`raw ≥ 0.63` ✗（0.5417）⇒ verdict 仍为 **FAIL**。
方向自检 `corr(score, y) = −0.156`，分数没反号。strict 键下 balanced 只有 0.5055，**说明增益依赖放宽后的配对定义**，这是它的脆弱点。

**UP-6 条件覆盖**（`msrs_calibration_shift.json`，复用 `msrs_generalization` 的 OOF，不重训）：

| 变体 | 覆盖 | 中位宽 | 最差层 | 最差层覆盖 |
|---|---:|---:|---|---:|
| global（参照） | 0.9004 | 2.0052 | cohort_size=2 (n=851) | 0.8096 |
| metal | 0.9002 | 2.0231 | cohort_size=2 | 0.8331 |
| difficulty | 0.8789 | 1.9195 | cohort_size=2 | 0.8073 |
| cohort | 0.8994 | 1.9847 | Os (n=900) | 0.8844 |
| **cohort_metal** | 0.8996 | 2.0183 | Os (n=900) | **0.8956** |

**metal-Mondrian 只把 0.8096 抬到 0.8331 —— 失败层不是金属。** 按 cohort size 分箱后最差层才真的被抬起来。
`cohort_metal` 宽度只涨 **0.65%**，过门槛的第二分支（宽度增幅 <5% 且最差层 ≥0.88）⇒ **PASS**。
**宽度本身改不了**：90% 区间宽 2.0052 ⇒ 残差 SD ≈ 0.61，与 R²=0.38 一致；校准只能改条件覆盖，不能凭空缩窄。

---

## §3 撤回日志（引用前必看）

| 日期 | 撤回的数 | 换成 | 为什么 |
|---|---|---|---|
| 2026-10-19 | **0.5828 ± 0.0102**（ligand-grouped R²，曾被叫作 "canonical"） | DOI 留出 **0.3380** | ligand 划分里 **89.43%** 的测试行与训练集共享同一篇 DOI，学的是论文内实验偏移。同一套特征/超参/模型，换整篇留出即掉 0.20 |
| 2026-10-19 | **Δ 符号 0.6193** | **0.5826**（5 seed） | 单 seed 高抽样：5 个 seed 落在 0.5183–0.5963，0.6193 比全部五个都高 |
| 2026-10-20 | 「0.5826 接近 0.65 门槛」 | 多数类率 **0.6009** | 门内 218 对只有两个金属对，正号占 60.1%。0.5826 **低于"永远答正号"**，不是"接近门槛" |
| 2026-10-20 | 生成 **SA≤4.5 = 33.7%** | **0.28%** | 33.7% 是键头重建池（给定真实 tmQM 几何只跑键头），从不调用采样器 |
| 2026-10-19 | 「LOMO 0.5575，差 0.04」 | 可解析子集 **0.5974** | 0.5575 把 Rh(51)/Os(36)/Re(74) 三个低于 `MIN_UNIQUE_FOR_GATE=200` 的折等权平均进去 |
| 2026-10-21 | 「换哪个金属更好**不可学**」（无条件全称） | 「**点估计不可学（raw 低于多数类率），balanced 口径可学但未达可用**」 | UP-2b 在同一批折上改 listwise 目标后 balanced 0.5985 [0.5262, 0.6663]，CI 不含 0.5。**但没有撤回 0.5826/0.5333 那三行**：它们仍是 raw 口径的正确读数。说"不可学"而不说口径，是这次的错误 |
| 2026-10-21 | 「交互份额 0.712 ⇒ 放开加性约束能救 metal_out」 | **约束不是杠杆**（free −0.097 / pair −0.113 / product −0.164） | 方差分解测的是**观测到的**交互，metal_out 要求把它**外推到没见过的金属**；这两件事不是同一个量。以后引用 0.712 必须同时说它只描述训练金属内部 |

**这五条里前两条是同一个根因**：单 seed + ligand 划分。凡是看到预测线数字没有写协议、没有写 seed 数的，先假定它属于这一类。

---

## §4 脚本地图

每个 artifact 的 `provenance.script` 是权威。下表按用途分组。

### 4.1 主模型（in-house D-MPNN）

| 脚本 | 用途 | 引用数 |
|---|---|---:|
| `molmetal/baselines/dmpnn.py` | **主模型实现本体**（不是脚本） | — |
| `train_dmpnn_multitask_wasserstein.py` | 5 金属多任务 + Wasserstein pIC50 正则；论文 §3.2 折划分 | 434 |
| `train_dmpnn_multitask.py` | 多任务 D-MPNN 基础 CLI（回归头 + 分类头） | 483 |
| `train_dmpnn_pic50_real.py` | 每金属 pIC50 回归基线（alpha=1.0） | 12 |
| `pre_featurize_dmpnn.py` | 特征预计算缓存 | 37 |

### 4.2 LOMO 家族（**5 个训练器，容易混淆**）

| 脚本 | 是什么 | 产出 | 引用数 |
|---|---|---|---:|
| `train_metal_lomo.py` | **原始 m19 LOMO**（5 折 × 3 seed，gate 0.70） | `metal_lomo_auc.json` | 157 |
| `train_metal_lomo_v2.py` | Wf-145 平衡 Rh 队列重测 | `metal_lomo_auc_v2.json` | 47 |
| `train_metal_lomo_wasserstein.py` | Wf-125 W₂ 迁移正则（**W₂ 默认关闭**，Wf-E1 证明旧实现 detach 后零梯度） | `metal_lomo_auc_wasserstein.json` | 91 |
| `train_multimetal_lomo.py` | Wf-32 **另一个架构**（SE(3)-SB backbone）的 LOMO 包装器 | — | 25 |
| `train_chemprop_lomo_paper_cohort.py` | chemprop v2 **仅作参考对照**（PA3），不是主模型 | `metal_lomo_chemprop_v2.json` | 44 |

**要点**：`metal_lomo_auc_wasserstein.json` 的 headline 是 **half-synthetic**，引用必须带 `real_only_fold_value`。

### 4.3 MSRS 泛化 / 排序 / 校准

| artifact | 脚本 |
|---|---|
| `msrs_seed_ensemble.json` | `msrs_seed_ensemble.py` |
| `msrs_generalization.json` | `msrs_generalization.py` |
| `msrs_feature_expansion_5seed.json` + `msrs_feature_fusion.json` | `msrs_feature_expansion_ensemble.py` |
| `msrs_delta_metric_audit.json` | `msrs_delta_metric_audit.py` |
| `msrs_delta_direct.json` | `msrs_delta_direct.py` |
| `msrs_calibration.json` | `msrs_calibration.py` |
| `msrs_complex_features.json` | `msrs_complex_features.py` |
| `msrs_metal_ablation.json` | `msrs_metal_ablation.py` |
| `msrs_rank_within_cohort.json` | `msrs_rank_within_cohort.py` |

### 4.4 活性迁移（Krasnov 对照）

| artifact | 脚本 |
|---|---|
| `metal_activity_transfer.json` / `_baselines` / `_scaffold` / `_permutation_null` / `_significance` | `metal_activity_transfer_probe.py` |
| `metal_activity_transfer_seed_stability_v2.json` | `metal_activity_transfer_seed_stability.py` |
| `lomo_resolvability_audit.json` | `lomo_resolvability_audit.py` |
| `lomo_seed_ensemble.json` | `lomo_seed_ensemble.py` |

### 4.5 零引用脚本（2026-10-21 实测，git grep 全仓库）

`pa16_nci60_funnel.py`、`train_dmpnn_pic50_ablation.py` —— 仓库内无任何引用。
**没有删除**（可能是手工入口），但任何新工作不应再依赖它们。

---

## §5 数据阻断（2026-10-21 EXT 阶段后：B1 已从"缺数据"升级为"领域事实"）

| ID | 阻断 | 实测 | 需要什么 |
|---|---|---|---|
| B1 | 外部独立 cohort | **不存在。** IrCytoToxDB（Scientific Data 2024，2,694 条 / 222 篇 / 127 系，CC-BY-4.0）与 MSRS 的 DOI 重叠 **221/222 = 99.55%**，仅 1 个新 DOI；653 个匹配 (DOI,细胞系) 格里 **89.4%** 数值相同到 0.01 log10，中位差 **0.0**，r=0.9611。其余来源：ChEMBL 29 本地库 0 个目标金属、PlatinAI 只有 Pt、NCI-60 无 SMILES、DepMap Cloudflare Turnstile、GDSC HTTP 410、Cellosaurus 超时、GEO GSE36133 只有 CEL 原始文件、RuCytoToxDB 无 Zenodo 存款（921 篇 ⊂ MSRS 的 1,427 个 Ru DOI） | 人工整理 2023–2025 文献（EXT-5 的两周预算，**未做，未编造**） |
| B2 | 细胞系生物学特征 | **永久关闭。** 0 个注释文件；21.9% 的行来自 CCLE/DepMap 里按构造不存在的系（HELA / HEK293 / A2780 / A2780CISR / A549CISR）；top-46 覆盖 80.22% | 一份包含耐药衍生系与非 CCLE 系的注释，或 MSRS 自建注释——都不是下载 |

**不要用 Pt 的 PlatinAI 冒充外部验证。也不要用 IrCytoToxDB 冒充外部验证**——
它是同一份 curation 的 Ir 切片（见 C9）。

### 5.1 EXT 阶段一句话

跨金属外推的失败位置已经确定：**shape（排序）占平方误差 0.653–0.757，
offset（水平）只占 0.045–0.100**。所以"校准一下就能用"在算术上不成立
（完美水平只值 +0.078 R²，4/5 金属仍为负）；k=50 校准时 R²≥0 的金属是 **0/5**，
且小 k 校准在 55–95% 的重复里比不校准更差。表观迁移主要来自配体重叠：
potency 的合并 ρ 从 0.296（metal_out）掉到 0.158（ligand_disjoint）。
详见 `TODO/completed/ext_phase_results.md`。

---

## §6 怎么重跑

```bash
# 表格（12 张 / 0 MISSING；manifest 记 sha256）
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/paper_metric_tables.py --strict

# 引用等级（CITE / NUM / NO）
.venv/bin/python molmetal/scripts/registry.py cite

# 本索引自校验
.venv/bin/python -m pytest molmetal/tests/test_predictive_line_index.py -q
```

重跑任何 artifact 之后，**必须先重跑 `paper_metric_tables.py --strict`**，否则表和产物脱节。
