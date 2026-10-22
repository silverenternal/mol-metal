# PREDICTIVE_LINE_MAIN_TABLE — 按估计目标分层的数字主表

由 `molmetal/scripts/msrs_main_table.py` 生成（2026-10-22T20:58:33+00:00）。
**每个数字都从 artifact 读出，不手抄**；与 `docs/PREDICTIVE_LINE_INDEX.md` 的机器校验块交叉核对。

## §1 怎么读

| ESTIMAND | 含义 |
|---|---|
| `pooled` | 混合行总体上的一个数（含没有任何模型能定位的行，所以可以是负数而模型仍在工作） |
| `centered` | 每个金属先减测试均值后再评分 —— oracle-offset 上界，不是可部署分数 |
| `per_pair` | 金属对内 / 组内的统计量 |
| `conditional` | 关于预测某个区域（覆盖、宽度、最差层）的陈述，不是均值 |
| `edge_level` | 金属图某一条边的性质 |

| IDENTIFIABLE | 含义 |
|---|---|
| `yes` | 目标可由模型看得见的东西决定 |
| `partial` | 只在「有配对证据」的子群上被识别 |
| `no` | 依赖留出金属自己携带的水平，任何留出协议都拿不到 |

## §2 可识别（可以按原样引用）

| 指标 | 值 | ESTIMAND | artifact:path | 为什么是这个标签 |
|---|---:|---|---|---|
| `G.doi.fused` | 0.3811 | `pooled` | `msrs_feature_fusion.json:protocols.doi.fused_r2` | same split, GNN fused in |
| `G.doi.maccs` | 0.3767 | `pooled` | `msrs_feature_expansion_5seed.json:protocols.doi.r2_seed_averaged` | whole-paper hold-out; the headline the paper quotes |
| `UP19.pooled_identifiable_r2_abs` | 0.3299 | `pooled` | `msrs_identifiability.json:metal_out.pooled.descriptors.identifiable_unit_rows.r2_abs` | same model restricted to rows that carry paired evidence: 0.3299 |
| `UP21.base_r2` | 0.3670 | `pooled` | `msrs_paper_meta_offset.json:aggregate.base.r2_mean` | DOI hold-out with no offset correction |
| `UP21.ridge_design_r2` | 0.3557 | `pooled` | `msrs_paper_meta_offset.json:aggregate.ridge_design.r2_mean` | metadata-only offset model: 0.3557, worse than doing nothing |
| `UP26.r2_n0` | 0.3674 | `pooled` | `msrs_paper_calibration.json:aggregate.0.eb_random.r2_mean` | zero calibration rows: reproduces 0.3674, the number every earlier round reported |
| `UP26.r2_n10` | 0.6839 | `pooled` | `msrs_paper_calibration.json:aggregate.10.eb_random.r2_mean` | 10 measured rows from each test paper; the deployable number, not the zero-calibration one |
| `UP28.dmpnn_mean` | 0.2760 | `pooled` | `msrs_dmpnn_doi.json:results.dmpnn.r2_mean_over_seeds` | what a single training run delivers; seed ensembling is worth +0.0522 on the graph, +0.0070 on the tree |
| `UP28.dmpnn_seed_avg` | 0.3282 | `pooled` | `msrs_dmpnn_doi.json:results.dmpnn.r2_seed_averaged` | DOI hold-out on 21,404 rows: 0.3282 vs the 0.3767 tree, so the graph representation loses 0.0485 |
| `UP28.fp_reproduce` | 0.3767 | `pooled` | `msrs_dmpnn_doi.json:results.fp_tree.r2_seed_averaged` | the protocol check inside the same run: 0.3767, bit-identical to the quoted headline |
| `UP30.size_multiple` | 94.5300 | `pooled` | `msrs_mmp_delta.json:gate.checks.size.multiple` | 61,539 clean pairs vs 651 exact-ligand pairs: the pairing definition was not the constraint |
| `UP33.cqr_reproduce` | 1.9871 | `pooled` | `msrs_conformal_calibrated.json:aggregate.cqr_uncal.median_width_mean` | the UP-27 CQR arm reproduces 1.9871 bit for bit inside this run |
| `UP33.mondrian_width` | 1.4390 | `pooled` | `msrs_conformal_calibrated.json:aggregate.mondrian.median_width_mean` | the group-coverage repair costs nothing: 1.4390, narrower than the global arm |
| `UP33.split_cal_coverage` | 0.9007 | `pooled` | `msrs_conformal_calibrated.json:aggregate.primary.coverage_mean` | marginal 90% coverage holds after the offset is applied inside the score |
| `UP33.split_cal_width` | 1.4478 | `pooled` | `msrs_conformal_calibrated.json:aggregate.primary.median_width_mean` | calibrated split-conformal at 0.9007 coverage: 1.4478 against 2.0073 uncalibrated on the same rows |
| `UP34.baseline` | -0.0734 | `pooled` | `msrs_metal_calibration.json:aggregate.baseline.weighted_r2` | leave-one-metal-out with no correction reproduces -0.0734 |
| `UP34.k10_primary` | -0.1250 | `pooled` | `msrs_metal_calibration.json:aggregate.k_curve.10.primary` | the pre-registered two-level EB arm at k=10 is -0.1250, worse than no correction |
| `UP34.k10_raw` | -0.0651 | `pooled` | `msrs_metal_calibration.json:aggregate.k_curve.10.metal_only_raw` | the unshrunk mean of the same 10 rows recovers 0.10 of the gap: the rows carry little at this k |
| `UP34.k50_raw` | -0.0171 | `pooled` | `msrs_metal_calibration.json:aggregate.k_curve.50.metal_only_raw` | at k=50 the same estimator recovers 0.68, so the information is there but needs five times the rows |
| `UP35.k0_auc` | 0.5975 | `pooled` | `msrs_lomo_kcal.json:per_k.0.mean_auc` | k=0 reproduces the published zero-shot LOMO AUC 0.5974 within the frozen 0.01 tolerance |
| `UP35.k10_auc` | 0.5897 | `pooled` | `msrs_lomo_kcal.json:per_k.10.mean_auc` | ten measured rows in training drop the AUC to 0.5897, below the 0.62 kill line |
| `UP35.k10_ru` | 0.5861 | `pooled` | `msrs_lomo_kcal.json:per_k.10.per_fold.Ru.auc_mean` | the Ru fold falls 0.0056 and the Ir fold 0.0100 at k=10: both metals move the same way |
| `UP35.k50_auc` | 0.6028 | `pooled` | `msrs_lomo_kcal.json:per_k.50.mean_auc` | fifty rows buy +0.0053 over zero-shot, 10% of the distance to 0.65: not a sample-size gap |
| `UP35.shift_auc_diff` | 0.0000 | `pooled` | `msrs_lomo_kcal.json:shift_arm.max_abs_auc_diff` | a constant logit shift moves AUC by 5.75e-07, confirming the shift arm is a theorem check |
| `UP12.additive_within` | 0.2128 | `centered` | `msrs_two_stage.json:within_metal.aggregate.additive.weighted_r2` | ligand-grouped within-metal; the reference the interaction arms are measured against |
| `UP12.free_within` | 0.2519 | `centered` | `msrs_two_stage.json:within_metal.aggregate.free.weighted_r2` | free interaction, same protocol: +0.0391 over additive |
| `UP19.pooled_identifiable_r2_centred` | 0.3806 | `centered` | `msrs_identifiability.json:metal_out.pooled.descriptors.identifiable_unit_rows.r2_centred` | oracle-offset upper bound 0.3806; not a deployable score |
| `UP23.n_pairs_kept` | 315 | `per_pair` | `msrs_ranking_hierarchical.json:design.n_pairs_kept` | 315 of 651 pairs survive the deadzone (|delta| >= the metal pair's noise SD) |
| `C5.coverage` | 0.9004 | `conditional` | `msrs_calibration.json:protocols.doi.variants.global.coverage` | marginal coverage under the DOI protocol |
| `C5.median_width` | 2.0052 | `conditional` | `msrs_calibration.json:protocols.doi.variants.global.median_width` | width at that coverage; the pair must always be quoted together |
| `UP14.doi_pair_in_train_share` | 0.0057 | `conditional` | `msrs_support_stratified.json:protocols.doi.unit_train_evidence.has_pair_in_train.share_of_rows` | 0.57% of DOI-hold-out test rows have paired evidence in training |
| `UP19.identifiable_share` | 0.0547 | `conditional` | `msrs_identifiability.json:decomposition_summary.identifiable_row_share` | 5.47% of metal-out test rows have a paired partner in training |
| `UP20.cqr_coverage` | 0.9000 | `conditional` | `msrs_cqr.json:aggregate.cqr.coverage_mean` | marginal coverage at the nominal 0.90 |
| `UP20.cqr_width` | 1.9871 | `conditional` | `msrs_cqr.json:aggregate.cqr.median_width_mean` | width at that coverage; 1.9871, i.e. the width target is not met |
| `UP27.weighted_cqr_width` | 1.9849 | `conditional` | `msrs_conformal_shift.json:aggregate.weighted_cqr.median_width_mean` | weighted conformal width at 0.90 coverage: 1.9849 vs 1.9871 unweighted, so the density ratio buys 0.0022 |
| `UP33.worst_metal` | 0.8088 | `conditional` | `msrs_conformal_calibrated.json:aggregate.primary.worst_metal_coverage` | Os is the failing group at 0.8088 against a 0.87 gate; the metal-Mondrian variant reaches 0.8888 |
| `UP6.cohort_metal_worst_stratum` | 0.8956 | `conditional` | `msrs_calibration_shift.json:variants.cohort_metal.worst_stratum.coverage` | worst stratum after Mondrian; a repair of the tail, not of the width |
| `UP10.potency.irrh_separability` | 0.1150 | `edge_level` | `msrs_support_audit.json:endpoints.potency.pairs.4.separability` | Ir-Rh: a precise zero, i.e. it constrains u_Ir ~ u_Rh rather than reporting an effect |
| `UP10.potency.ruos_separability` | 0.5920 | `edge_level` | `msrs_support_audit.json:endpoints.potency.pairs.2.separability` | Ru-Os: the only edge whose effect exceeds its own assay noise |
| `UP18.irrh_t_cluster` | 0.9580 | `edge_level` | `msrs_metal_graph.json:edges_by_pair.Ir-Rh.t_cluster` | 0.958: the mean is indistinguishable from zero, which is itself the constraint |
| `UP18.ruos_t_cluster` | 3.3640 | `edge_level` | `msrs_metal_graph.json:edges_by_pair.Ru-Os.t_cluster` | ligand-clustered; 3.364 with a 95% CI [0.0618, 0.2164] on the mean |
| `UP18.ruroh_precision_share_cluster` | 0.0054 | `edge_level` | `msrs_metal_graph.json:edges_by_pair.Ru-Rh.precision_share_cluster` | largest effect, worst precision: the two edge rankings are different sets |

## §3 部分可识别（必须同时引用条件）

| 指标 | 值 | ESTIMAND | artifact:path | 为什么是这个标签 |
|---|---:|---|---|---|
| `C1.metal_blind_delta` | 0.0683 | `pooled` | `msrs_metal_ablation.json:protocols.ligand.arms.metal_blind.delta_r2_vs_full` | ligand split; the sign of the metal-aware gain is identified, the level is not |
| `C6.lomo_resolvable` | 0.5974 | `pooled` | `lomo_resolvability_audit.json:aggregates.resolvable_only.value` | Ru+Ir folds only; Rh/Os/Re are below MIN_UNIQUE_FOR_GATE |
| `UP17.ligand_metal_forward` | 0.8015 | `pooled` | `msrs_shape_decomposition.json:independent.ligand_x_metal.share_forward` | order-dependent: 0.8015 structure-first vs 0.0 batch-first |
| `UP17.paper_metal_batch_first` | 0.5484 | `pooled` | `msrs_shape_decomposition.json:independent.paper_x_metal.share_batch_first` | the same block under the other ordering: 0.5484 vs 0.0 |
| `UP22.re_np3_r2` | 0.1745 | `pooled` | `msrs_paired_kshot.json:aggregate.Re.ridge|3.mean_r2` | Re has 0 paired rows, so this number cannot come from pairing by construction |
| `UP22.rh_np0_r2` | 0.0709 | `pooled` | `msrs_paired_kshot.json:aggregate.Rh.ridge|0.mean_r2` | row-level sampling with zero paired points: the gain is not a pairing effect |
| `UP24.bce_auc` | 0.5449 | `pooled` | `msrs_lomo_pairwise.json:aggregate.bce.auc_overall_resolvable` | linear ECFP4 baseline on Ru+Ir; not the D-MPNN 0.5974 |
| `UP24.pairwise_edge_auc` | 0.5529 | `pooled` | `msrs_lomo_pairwise.json:aggregate.pairwise_edge.auc_overall_resolvable` | +0.0080 over the matched BCE arm: the loss is not the binding constraint |
| `UP31.bce_auc` | 0.5924 | `pooled` | `msrs_lomo_pairwise_dmpnn.json:gate.checks.reproduce.value` | the control arm; reproduces 0.5974 within +/-0.01, which is what makes the comparison admissible |
| `UP31.pairwise_auc` | 0.5928 | `pooled` | `msrs_lomo_pairwise_dmpnn.json:gate.checks.primary.value` | Ru+Ir only; pairwise 0.5928 vs BCE 0.5924, a redistribution across folds rather than a gain |
| `UP8.zero_shot_os` | 0.3479 | `pooled` | `msrs_meta_kshot.json:results.potency.splits.metal_out.per_metal.Os.zero_shot_r2` | Os is the one metal whose level transfers; not a general statement |
| `C3.diff_raw` | 0.5826 | `per_pair` | `msrs_delta_metric_audit.json:routes.level_differencing.raw_sign` | same strict pairs; raw sign accuracy, not balanced |
| `C3.majority_class` | 0.6009 | `per_pair` | `msrs_delta_metric_audit.json:baselines.majority_class_rate` | strict 218-pair set only; a 0.58 raw score is below this |
| `UP16.irrh_balanced` | 0.5600 | `per_pair` | `msrs_ranking_support.json:strata.listwise_ligand_cell.by_metal_pair.Ir-Rh.balanced_accuracy` | Wilson CI [0.4799, 0.6190] includes 0.5 |
| `UP16.osru_balanced` | 0.5807 | `per_pair` | `msrs_ranking_support.json:strata.listwise_ligand_cell.by_metal_pair.Os-Ru.balanced_accuracy` | Wilson CI [0.4990, 0.6548] includes 0.5 |
| `UP23.lambdarank_deadzone_balanced` | 0.5612 | `per_pair` | `msrs_ranking_hierarchical.json:arms.lambdarank_deadzone.all_pairs.overall.balanced_accuracy` | tie-aware labels: 0.5612, worse than leaving the noisy pairs alone |
| `UP23.lambdarank_none_balanced` | 0.5985 | `per_pair` | `msrs_ranking_hierarchical.json:arms.lambdarank_none.all_pairs.overall.balanced_accuracy` | reproduces UP-2b's 0.5985 on all 360 pairs |
| `UP29.davidson_balanced` | 0.5312 | `per_pair` | `msrs_davidson_ties.json:results.davidson_none.balanced_accuracy` | 315 retained pairs only; ties as a third outcome reproduce Bradley-Terry to -0.0005 |
| `UP2b.ligand_cell_balanced` | 0.5985 | `per_pair` | `msrs_listwise_ranker.json:pairs.ligand_cell.overall.balanced_accuracy` | 360 cross-paper pairs; no single metal pair clears 0.5 on its own (UP-16) |
| `UP27.aci_cqr_coverage` | 0.9106 | `conditional` | `msrs_conformal_shift.json:aggregate.aci_cqr_g0p01.coverage_mean` | ACI recovers coverage online (0.9106) but pays width (2.0572); a rolling protocol, not a static interval |
| `UP18.ruroh_t_cluster` | 2.5770 | `edge_level` | `msrs_metal_graph.json:edges_by_pair.Ru-Rh.t_cluster` | 15 units, sd 1.4003; t=2.577 survives but the interval spans a factor of five |

## §4 不可识别（不得作为能力陈述引用）

| 指标 | 值 | ESTIMAND | artifact:path | 为什么是这个标签 |
|---|---:|---|---|---|
| `G.ligand.ecfp4` | 0.5596 | `pooled` | `msrs_seed_ensemble.json:protocols.ligand.ensemble.r2` | ligand-grouped split: 89.4% of test rows share a DOI with training (retracted 2026-10-19) |
| `UP14.metal_out_weighted_r2` | -0.0734 | `pooled` | `msrs_support_stratified.json:protocols.metal_out.weighted_r2_per_metal` | the held-out metal's level is unidentifiable; 94.5% of these rows have no paired partner |
| `UP19.pooled_all_r2_abs` | -0.0640 | `pooled` | `msrs_identifiability.json:metal_out.pooled.descriptors.all_rows.r2_abs` | dominated by the 94.5% of rows whose metal level cannot be recovered |
| `UP21.lopo_ridge_r2` | -0.0259 | `pooled` | `msrs_paper_meta_offset.json:per_seed.20261022.offset_diagnostics_fold0.lopo_ridge_r2` | leave-one-paper-out R2 of the offset model: -0.0259, i.e. worse than the mean |
| `UP30.r2_ligand_only` | -0.1457 | `pooled` | `msrs_mmp_delta.json:gate.checks.quality.pooled_r2_ligand_only` | held-out-metal MMP delta -0.1457, below the -0.0163 constant baseline; the widened set carries no signal |
| `UP33.oracle` | 1.3717 | `pooled` | `msrs_conformal_calibrated.json:aggregate.split_oracle.median_width_mean` | per-paper oracle at 1.3717 with realized coverage 0.9569: a floor, not a deployable interval |
| `UP34.oracle` | 0.0092 | `pooled` | `msrs_metal_calibration.json:aggregate.oracle.weighted_r2` | the ceiling for any per-metal constant offset is 0.0092; the whole recoverable gap is 0.0826 |
| `UP11.complete_case_ligands` | 0 | `edge_level` | `msrs_pair_delta.json:rank_structure.complete_case_ligands` | zero ligands span 3 metals, so the rank-1 interaction model has no identifiable row |

## §5 一致性检查

| 断言 | 结果 |
|---|---|
| 每行都有 ESTIMAND + IDENTIFIABLE + artifact 路径 | PASS |
| 主表里每个 artifact 都在 registry 中 | PASS |
| 与 PREDICTIVE_LINE_INDEX 机器校验块一致（71 个 id 交叉核对） | PASS |

