# CFM 线弱项清单（2026-10-13 盘点）

> 全部数字均来自盘上 metric 文件，每条都附来源路径。
> 不含原因分析、不含修复建议——只列弱项。机制层面的根因和处置见 `cfm_todo.json::round7_review_2026_10_13` / `cfm_todo.json::round9_finding_2026_10_13` / `docs/CFM_SIX_ROUND_CONCLUSION_2026_10_13.md`。

---

## A. 生成质量

### A.1 QED 极低（生成的分子 40% 不如真实语料）

- 生成均值 0.0944（n=66）
- real_tmqm 均值 0.2348（n=31150）
- 缺口 -0.14034，**生成仅达 real 的 40.23%**
- 来源：`metrics/by_metric/cfm_objective_inventory.json::targets.QED.measured_on_disk`
- 中位数 0.0738 vs real 0.2092
- 缺口归因（counterfactual ADS 描述符替换）：

  | 描述符 | dQED | 占缺口比例 |
  |---|---:|---:|
  | ROTB | +0.0577 | 最大 |
  | ALERTS | +0.0561 | 第二大 |
  | ALOGP | +0.0211 | 第三 |
  | AROM | **-0.0215** | 反向——加芳香会变差 |

### A.2 SA 比真实语料难 1.46 SA 单位

- 生成均值 6.44，real_tmqm 4.98
- 缺口 +1.464 SA 单位（**更难合成**）
- 缺口归因（71.6% / 校正后 69.3% 来自 stereocentre 罚项）：

  | 罚项 | SA 单位 | 占缺口 |
  |---|---:|---:|
  | 立体中心（log10(n+1)）| -1.048 | **71.6%** |
  | 桥头 | 0.016 | 几乎无 |
  | 螺环 | -0.059 | 几乎无 |
  | 大小 | 0.012 | 几乎无 |
  | 片段稀有度 | +0.315 | **反向**——生成更稀有 |
  | 特征综合 | +1.075 | 占主体 |

  来源：`metrics/by_metric/cfm_sa_gap_decomposition.json`
  校正：`metrics/by_metric/cfm_sa_stereocentre_control.json::corrections_2026_10_12`（prefix-4000 → full corpus 31150）

### A.3 立体中心 19.8 倍过生成

- 生成：6.258 potential stereocentres/mol（97% 分子有 ≥1）
- real_tmqm：0.316 potential stereocentres/mol（83.2% 分子 0 个）
- **比例 19.83x**
- 即便排除金属邻接原子，仍是 25.86x（生成 4.67 vs real 0.18）
- 来源：`metrics/by_metric/cfm_sa_stereocentre_control.json::corrections_2026_10_12.remeasured_same_protocol_this_pass::FULL_31150`

### A.4 分子结构特征（生成 vs real）

- ROTB 16.86/mol vs 5.18（**3.26x**）
- 三键 bond 比例 0.667 vs 0.265
- 芳香键比例 **0.0012 vs 0.4546**（0.27% of real）
- 至少一个芳香键的分子比例 0.0152 vs 0.8999
- logP 8.09 vs 4.15
- 重原子 37.15 vs 26.18（**大 42%**）
- 来源：`metrics/by_metric/cfm_objective_inventory.json::QED.underlying_chemistry`

### A.5 Decode yield

- 主 cohort baseline 0.344（66/192）
- Phase 3 L3 A/B control arm 0.109（7/64，**比 baseline 差 3.16x**）
- 来源：`metrics/by_metric/cfm_free_generation_panel_seed0.json::generation.gen_validity` + `molmetal/reports/cfm_l3_decode_enabled_ab/control/arm_control.json::decode.decode_rate`

### A.6 解码后 50% 是碎片

- 50.4% 是 ≥2 碎片
- 23% 的分子金属配位 = 0（**金属彻底无键**）
- 所有 metal-coord-0 案例同时是 multifragment（同一类失败）
- 来源：`metrics/by_metric/cfm_decode_fragmentation.json::headline`

### A.7 ADMET 全部由有机描述符决定，0/8 读金属

- 8 个 ADMET proxy 全是 logP/MW/TPSA/RotB/HBD/6 SMARTS alerts 的纯函数
- 0 of 8 读金属 → ADMET 不可能表达任何金属特定事实
- ames 0.0 vs 0.0349 "更好"实属假象（acyclic ligand 不触发 SMARTS alerts）
- logPapp -3.04 vs -3.68 "更好"实属 clipping artefact（中位数卡在 -3.0）
- logS -8.01 vs -6.70（差 -1.31 log 单位，是唯一真失败；全 inherit 自 logP 8.09 vs 4.15）
- 来源：`metrics/by_metric/cfm_objective_inventory.json::ADMET` + `cfm_real_tmqm_sixmetal_admet.json`

---

## B. 几何（sampled from noise）— **没有任何现有 metric 能看到**

24 clouds smoke scale 测量（`cfm_generated_geometry_defect.json`）：

| 量 | 生成 | real | 含义 |
|---|---:|---:|---|
| 最小原子对距离 | **0.570 Å** | 1.172 Å | **键长尺度接触**（不是 non-bonded） |
| metal-ligand 最近 | **1.234 Å** | 1.953 Å | **ligand 落在金属 covalent radius 内** |
| median pair distance | 4.71 Å | 5.00 Å | 大体尺寸对 |
| radius of gyration | 3.57 Å | 3.81 Å | 大体尺寸对 |
| hard clash pairs 总数 | **143 / 24 clouds** | **0 / 24 clouds** | 24/24 generated 有 clash |
| n_clouds_with_any_hard_clash | 24/24 | 0/24 | 100% clash rate |

- hard clash 定义：距离 < 0.7 × (rcov_i + rcov_j)，Cordero 2008 共价半径
- 来源：`metrics/by_metric/cfm_generated_geometry_defect.json::generated_vs_real`
- L1（速度场 regression）没看到这个，因为它只看 real 几何
- RDKit 也没看到（EmbedMolecule 丢弃坐标）

---

## C. Loss 函数 — 8/12 gradient-dead

`cfm_loss_term_liveness.json` 实测 shipped `configs/cfm.yaml`（arm_B 真实 cohort）：

| Term | 值 | 状态 |
|---|---:|---|
| **fm_term** | **12.458 LIVE** | 主导项 |
| **bond_ce** | **2.157 LIVE** | 唯一接触 decoded chemistry 的项，但是 per-edge |
| wasserstein*w2 | 0.082 LIVE | 几何正则化（real clouds） |
| cb_loss | 0.0004 LIVE | 组成平衡（real clouds） |
| diversity_term | 0 DEAD | |
| shortcut_distill_loss | 0 DEAD | |
| consistency_term | 0 DEAD | |
| so3_term | 0 DEAD | |
| edm_term | 0 DEAD | |
| assemble_term | 0 DEAD（实测 even at weight=1.0：~5e-09 gradient）| |
| soft_valence_term | 0 DEAD | |
| entropy_term | 0 DEAD（实测 weight=0.01 时 0.11-0.19% 不可观测）| |

**没有 molecule-level term 看到过 decoded molecule**：4 个候选（batch_soft_valence / batch_assemble / QED / _bridge_components）都读 bond head 在 **real** training cloud 上的 logits，不在 generated 上。
- 来源：`metrics/by_metric/cfm_objective_inventory.json::what_the_loss_can_reach` + `cfm_loss_term_liveness.json`

---

## D. Bond head

### D.1 macro-F1(type) 落后 OpenBabel

- 3-seed：0.4784 ± 0.0055（gap -0.0267，**4.85x std**）
- OpenBabel 同口径：0.5051
- 来源：`metrics/by_metric/cfm_bond_type_f1_3seed.json::seeds` + `metrics/by_metric/tmqm_bond_perception_3seed.json`
- Bond EXISTENCE F1 0.9961（不是这个的瓶颈）

### D.2 70.7% 芳香→单键 collapse

完整 bond-set 扫描（`cfm_m4_aromatic_lo_band_sweep.json`，900 complexes，round-7 caveat 已 honor）：

| AROMATIC_LO | n_truth_arom | n_predicted_arom | recall | precision_full | F1_full | 塌缩比 |
|---|---:|---:|---:|---:|---:|---:|
| 1.30 | 17106 | **5539** | 0.2872 | 0.8868 | 0.4338 | 0.7128 |
| 1.45 | 3195 | **5539** | 0.4473 | 0.2580 | 0.3272 | 0.5527 |
| 1.50 | 931 | **5539** | 0.7315 | 0.1229 | 0.2105 | 0.2685 |

**5539 aromatic predictions 跨 3 个 band 设置完全不变**——F1 跟着 precision 反向走（precision 0.887 → 0.258 → 0.123），head 的 overcall 是 invariant 的。

### D.3 阈值扫描也无法校准

9 个 bond_decision 阈值（0.05-0.90）扫描结果（`cfm_m4_threshold_sweep.json`）：

- n_predicted_aromatic spread：5539-5540（**仍不变**）
- F1 spread：0.43381-0.43383（**第 4 位小数才动**）
- max F1 = 0.4338（threshold=0.10）
- `calibrates_5539_count = false`
- 结论：阈值 alone 不能关 F1 gap

### D.4 t-degradation 曲线（CLIFF_THEN_PLATEAU）

32 clouds 的 t sweep（`cfm_arch1_t_degradation.json`）：

| t | 1.0 | 0.9 | 0.75 | 0.5 | 0.25 | 0.1 | 0.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| macro-F1 | 0.611 | 0.233 | 0.206 | 0.195 | 0.200 | 0.194 | 0.200 |
| exist-F1 | 0.882 | 0.293 | 0.279 | 0.263 | 0.271 | 0.208 | 0.228 |

- 0.611 → 0.233 一次悬崖（drop 0.418）
- t<1 之后平台在 [0.193, 0.233] 区间，**不依赖 t**
- ARCH-1 strict-monotonic 预测**部分被推翻**

### D.5 Patience=3 default 让 headline 偏低

- 3 seed bond_type_f1：seed0 0.9194 / seed1 0.9297 / seed2 0.9703
- seed0/seed1 在 ep14/15 早停，**最后 3 epoch 还在升**（0.9031/0.8922/0.9144）
- seed2 跑到 ep28 → 0.9703
- **0.0269 std 不是 noise floor，是 convergence-time spread**
- 0.9194 "headline" 是 lower bound，不是 config 真值
- 来源：`metrics/by_metric/cfm_bond_type_f1_3seed.json::root_cause_found_2026_10_12`
- 修复：`configs/cfm_fullcorpus_seed*.yaml` 显式设 `train.early_stop_patience: 10`（**未做**）

---

## E. ADMET — 0/8 proxy 读金属

8 个 ADMET proxy（`molmetal/molmetal_lam/sbdd_env/metrics_v2.py`）：

| Proxy | 输入 | 读金属？ |
|---|---|---|
| ames_mutagen | 6 SMARTS alerts | 否 |
| cell_permeability_logPapp | logP, TPSA, HBD | 否 |
| aqueous_solubility_logS | logP, MW, RotB, aromfrac | 否 |
| gi50_proxy | 有机描述符 | 否 |
| herg_cardio_risk | 有机描述符 | 否 |
| hepatotox_index | 有机描述符 | 否 |
| plasma_protein_binding | 有机描述符 | 否 |
| herg_cardio_risk | 有机描述符 | 否 |

- **0 of 8 读金属** → 加 ADMET loss 改的是 logP / RotB / aromaticity（QED 同轴）
- 唯一真差 logS -8.01 vs -6.70（-1.31 log 单位），全 inherit 自 logP 8.09 vs 4.15
- 来源：`metrics/by_metric/cfm_objective_inventory.json::ADMET.gap_in_its_own_units.the_structural_fact` + `cfm_real_tmqm_sixmetal_admet.json`

---

## F. Sampling artefact — 0/149 generated molecules 含 Pt

- 3 cohort × 149 molecules，**ZERO Pt**
- real_tmqm Pt 比例 25.2%
- 根因：`load_tmqm_clouds(n_clouds=192, seed=0, split='val')` 抽 192/960 val cloud，**window 里恰好 0 Pt**（reservoir luck）
- 复现：n_clouds=960 → Pt 64；n_clouds=2000 → Pt 491
- patch `min_per_metal: int = 0`（默认 0 = 字节级与原行为一致）**SHIPPED + 4/4 red-test PASS**（`molmetal/data/tmqm_clouds.py:263,300,424,429,539,549,554,557-565,588,597-606,700,705`）
- **但 paper Table 1/2 用的 cohort 还没 re-run**，所有 anticancer claim 不可信
- 来源：`metrics/by_metric/cfm_objective_inventory.json::precious_metal_anticancer_specific.the_zero_platinum_finding`

---

## G. Headline 与现实的矛盾（已撤回/修正的旧主张）

| 旧主张 | 撤回理由 | 来源 |
|---|---|---|
| coordination number 4.87 vs 4.91 = "at parity" | 4.91 是 failing generated subset 的 mean degree，不是 real tmQM；real 5.95，gap -1.07（不是 parity） | `cfm_objective_inventory.json::D5` |
| 18.8% non-embeddable (BLK-C-L5) | 删金属后 repair rate：生成 0.917 vs real 0.95，差 -0.033（**base rate, not model defect**） | `cfm_l5_metal_deletion_control.json` |
| Reference band n=636 是 "real" | **是 decoder reconstruction**（model compared against itself，CIRCULAR） | `cfm_reference_band_audit.json` 标记 7 处 CIRCULAR |
| 22.8x pad_groups speedup | **WITHDRAWN**（92 GiB OOM 死掉的实现）；runnable 4.78x，McNemar p=0.0769 SUB-FLOOR，63/192 clouds flip | `cfm_pad_groups_mcnemar.json` |
| 66% generated bonds are triple | **misread**（cfm_objective_inventory 的 fraction-of-molecules 误读为 fraction-of-bonds） | `cfm_objective_inventory.json::target.ADMET.gap` |
| soft_valence 32537x scale | **REFUTED**（per-atom division 在 c3ee284b64 已经 ship，同步修复） | `cfm_loss_term_liveness.json::STATUS_2026_10_12` |
| 31x stereocentre / 89% zero-centre | **REWORDED** 到 19.8x / 83.2%（prefix-4000 不是 random sample） | `cfm_sa_stereocentre_control.json::corrections_2026_10_12` |
| LigandDiff 引用为 "JCIM 2024" | **CITATION CORRECTION**：J. Chem. Theory Comput. 2024, 20(10), 4377-4384, DOI 10.1021/acs.jctc.4c00232, PMID 38743854 | `cfm_objective_inventory.json::claim_defects_found_while_inventorying` |
| soft_valence 0/2146 valence-invalid (BLK-C11) | **WITHDRAWN**（cfm_property_panel 调 nonexistent RDKit API 吞掉 AttributeError；fix 后 0.8495 正常） | `cfm_loss_term_liveness.json::findings.BLK_C11_root_cause_is_visible` |
| Pretrain +0.046 → transferable | **REFUTED**（换 Bemis-Murcko scaffold split 后 paired delta -0.0094，95% CI [-0.0280, +0.0116], t(4)=-0.839, gate FAIL） | `metrics/by_metric/tmqm_pretrain_init_ab_ru.json` |

---

## H. Stale defaults 压低 headline

- `DEFAULT_EARLY_STOP_PATIENCE=3`（不是 10）—— `train_cfm_yuelbond_wasserstein.py:720`
- bond_type_f1 走 `--early-stop-patience`，不走 `--patience`（默认 10）控制 decode_ratio 的另一条路径
- 0.9194 "headline" 是 lower bound，真值未测
- 0.0269 std 是 convergence-time spread
- 0.9398 不能 claim "beats OpenBabel 0.5051"（不同 cohort 不同 aggregation：192 val vs 900 bench）
- 修复：`configs/cfm_fullcorpus_seed*.yaml` 显式 set `train.early_stop_patience: 10`（**未做**）
- 来源：`metrics/by_metric/cfm_bond_type_f1_3seed.json::root_cause_found_2026_10_12`

---

## I. 其他弱项

### I.1 sascorer fork 不可用

- `molmetal/eval/scoring/sascorer.py` 缺 `fpscores.pkl.gz`（`.gitignore` 路径，磁盘不存在）
- 4 处公式分歧：fpSize=2048 折叠 vs sparse 32-bit ids；缺 unmatched-bit 罚；nf 除数错；多算 score4 项
- 8/8 `test_sascorer_fork_divergence.py` PASS 证明 fork 不可用
- 训练时 `sa_penalty.py:81-125` Path 1 走 RDKit-Contrib（OK）
- 解码时 `stereo_postprocess.py:106-108` 缺 fallback（panel helper 风险）
- 3 个模块（`pocket2mol_score` / `softmol_score` / `targetdiff_score`）import 失败
- 来源：`metrics/by_metric/cfm_sa_stereocentre_control.json::why_this_control` + `molmetal/tests/test_sascorer_fork_divergence.py`

### I.2 assemble_term 即使 weight=1.0 也 underflow 到 ~5e-09

- `assemble_fragment_loss` 用 `prod_{k=1..12}(P^k)[i,i]`（product 公式）
- 实测：connected chain 4.62e-09 / metal-isolated 6.17e-09 / weak bonds 3.98e-08
- 对 bond_ce gradient O(1) 是 **~8 个数量级太小**，训练不动
- `formulation='escape_time'`（算术平均）能修，gradient 3.5e-04 到 2.6e-03（**~7e4x 大**）
- 但 flipping default 破 10/21 tests（含 `test_fully_connected_graph_has_near_zero_gradient`，"connected graph 零 gradient" 是 DELIBERATE property）
- 两个 formulation 真有矛盾，选择是 judgement call，**非 bug fix**
- 来源：`metrics/by_metric/cfm_decode_fragmentation.json::why_this_is_the_real_blocker`

### I.3 stereo_postprocess 失活 vs m14 decode ratio 1.000 错觉

- stereo_postprocess 删立体中心 4 个干预里 2 个 vacuous（set_tag / remove_annotation 不影响 includeUnassigned=True 的计数）
- saturation / remove_atom 是 score-level postprocess（删 6.26 atoms/mol，最差留 12-atom fragment），不是 chemistry improvement
- 1.000 chemistry_validity_pass_rate 是 **RDKit parse check**（每原子独立），不是 quality metric
- 真实 discriminator 是 single_fragment_rate / SA / QED **相对** real-corpus reference 的偏离方向
- generation 0.9697 vs real 0.8365（**生成比 real 更连**，因为 head 学了"填配位"）
- 来源：`metrics/by_metric/cfm_free_generation_panel_seed0.json::the_discriminator_correction_2026_10_12`

---

## 数字速查表（按严重度排序）

| 严重度 | 弱项 | 数字 | 关键 metric |
|---|---|---|---|
| **CATASTROPHIC** | 24/24 generated cloud 有 hard clash，metal-ligand 1.234 Å | 落在 metal covalent radius 内 | `cfm_generated_geometry_defect.json` |
| **CRITICAL** | Decode yield 0.109 on Phase 3 (vs baseline 0.344) | 3.16x drop | `cfm_l3_decode_enabled_ab.json` |
| **CRITICAL** | 0/149 generated molecules 含 Pt | real 25.2% | `cfm_objective_inventory.json::precious_metal_anticancer_specific.the_zero_platinum_finding` |
| **CRITICAL** | 8/12 loss terms gradient-dead in shipped config | 没有 molecule-level term 看到过 decoded molecule | `cfm_loss_term_liveness.json` |
| **CRITICAL** | 50.4% decoded 是 multifragment + 23% metal-detached | 一类 failure | `cfm_decode_fragmentation.json` |
| **CRITICAL** | QED 0.0944 vs real 0.2348 | 仅 40% of real | `cfm_objective_inventory.json` |
| **CRITICAL** | SA 6.44 vs real 4.98（+1.46 units harder）| 71.6% 来自 stereocentre 罚项 | `cfm_sa_gap_decomposition.json` |
| **HIGH** | 70.7% 芳香→单键 collapse（F1 0.43 invariant）| 5539 aromatic preds 跨 3 band 不变 | `cfm_m4_aromatic_lo_band_sweep.json` |
| **HIGH** | macro-F1(type) 0.4784 vs OpenBabel 0.5051 | -0.0267 (4.85x std) | `cfm_bond_type_f1_3seed.json` |
| **HIGH** | 19.8x potential stereocentres 过生成 | 6.26 vs 0.32/mol | `cfm_sa_stereocentre_control.json` |
| **HIGH** | 0/8 ADMET proxy 读金属 | ADMET 不可为独立 target | `cfm_objective_inventory.json::ADMET` |
| **HIGH** | sascorer fork 不可用（fpscores.pkl.gz missing + 4 公式分歧）| 8/8 test PASS 证不可用 | `test_sascorer_fork_divergence.py` |
| **HIGH** | assemble_term gradient ~5e-09（~8 数量级太小）| product 公式 underflow | `cfm_decode_fragmentation.json` |
| **HIGH** | coord number 4.87 vs real 5.95（gap -1.07）| 不是 parity | `cfm_objective_inventory.json::D5` |
| **HIGH** | patience=3 default 让 headline 偏低 | 0.9194 是 lower bound | `cfm_bond_type_f1_3seed.json` |
| **MEDIUM** | t-degradation cliff-then-plateau | 0.611 → 0.194 → 0.200 | `cfm_arch1_t_degradation.json` |
| **MEDIUM** | 18.8% non-embeddable 是 RDKit base rate, not model | repair rate 0.917 vs 0.95 | `cfm_l5_metal_deletion_control.json` |
| **MEDIUM** | n=636 "real" reference band 是 decoder reconstruction | 7 处 CIRCULAR | `cfm_reference_band_audit.json` |
| **MEDIUM** | 22.8x pad_groups speedup WITHDRAWN | 实际 4.78x，63/192 flip | `cfm_pad_groups_mcnemar.json` |
| **MEDIUM** | stereo_postprocess 2/4 干预 vacuous | set_tag / remove_annotation 不影响计数 | `cfm_opt1_stereo_suppression.json` |
| **MEDIUM** | L5 18.8% 假阳性（被 metal-deletion control 推翻）| repair rate 接近 real | `cfm_l5_metal_deletion_control.json` |
| **MEDIUM** | 1.000 chemistry_validity 是 RDKit parse check, not quality | 配 single_fragment 才是真 discriminator | `cfm_free_generation_panel_seed0.json` |
| **LOW** | tmQM pretrain +0.046 transfer FAIL on scaffold split | paired delta -0.0094 | `tmqm_pretrain_init_ab_ru.json` |
| **LOW** | soft_valence 32537x REFUTED | c3ee284b64 已修 | `cfm_loss_term_liveness.json` |
| **LOW** | 31x stereocentre REWORDED to 19.8x | prefix 不是 random sample | `cfm_sa_stereocentre_control.json` |
| **LOW** | 0/2146 valence-invalid BLK-C11 WITHDRAWN | API 不存在 + AttributeError 吞掉 | `cfm_loss_term_liveness.json` |
| **LOW** | 66% of generated bonds are triple 是 misread | fraction-of-molecules vs fraction-of-bonds 混淆 | `cfm_objective_inventory.json` |
| **LOW** | LigandDiff "JCIM 2024" 错 | 实为 JCTC 2024 | DOI 10.1021/acs.jctc.4c00232 |

---

## 数据点完整来源索引

49 个 `metrics/by_metric/cfm_*.json` 中实际引用的：
- `cfm_objective_inventory.json`（43.8 KB，QED/SA/ADMET/precious-metal/Loss 全栈）
- `cfm_sa_stereocentre_control.json`（3.7 KB，19.8x stereocentre）
- `cfm_sa_gap_decomposition.json`（2.4 KB，71.6% gap 归因）
- `cfm_bond_type_f1_3seed.json`（4.0 KB，0.4784 + patience=3）
- `cfm_free_generation_panel_seed0.json`（4.4 KB，66/192 panel）
- `cfm_loss_term_liveness.json`（11.4 KB，8/12 dead）
- `cfm_generated_geometry_defect.json`（5.0 KB，24/24 hard clash）
- `cfm_decode_fragmentation.json`（3.4 KB，50.4% multifragment + 23% metal-detached）
- `cfm_l5_metal_deletion_control.json`（2.8 KB，repair 0.917 vs 0.95）
- `cfm_m4_aromatic_lo_band_sweep.json`（15.2 KB，3-band sweep）
- `cfm_m4_threshold_sweep.json`（11.2 KB，9-threshold sweep）
- `cfm_arch1_t_degradation.json`（11.2 KB，t 曲线 CLIFF_THEN_PLATEAU）
- `cfm_opt1_stereo_suppression.json`（2.2 KB，stereocentre gap 1.27 SA）
- `cfm_opt3_min_per_metal.json`（1.2 KB，min_per_metal patch 验证）
- `cfm_pad_groups_mcnemar.json`（2.0 KB，4.78x + 63/192 flip）
- `cfm_reference_band_audit.json`（33.9 KB，7 处 CIRCULAR）
- `cfm_real_tmqm_sixmetal_baseline.json`（2.6 KB，n=31150 real band）
- `cfm_real_tmqm_sixmetal_admet.json`（0.6 KB，n=31152 real ADMET）
- `cfm_opt2_ring_content.json`（772 KB，3-ring 89.8% vs real 0.5%）
- `cfm_opt2_ring_dative_correction.json`（4.8 KB，dative→single 校正后 7.3x）
- `cfm_l3_guidance_ab.json`（4.4 KB，L3 guidance A/B）
- `cfm_l3_guidance_sweep.json`（19.9 KB，4-arm decode 0.344/0.359/0.531/0.828）
- `cfm_blkc_m3_noise_floor.json`（7.3 KB，McNemar noise floor）
- `cfm_blk_c_obj_decision.json`（14.2 KB，per-target decision）
- `cfm_run_census.json`（12.9 KB，run census）

非 CFM 但相关：
- `metrics/by_metric/tmqm_bond_perception_3seed.json`（OpenBabel 0.5051 同口径）
- `metrics/by_metric/tmqm_pretrain_init_ab_ru.json`（pretrain transfer FAIL）
- `molmetal/tests/test_sascorer_fork_divergence.py`（8/8 fork 不可用）
- `molmetal/tests/test_tmqm_min_per_metal.py`（4/4 min_per_metal patch）
- `molmetal/reports/cfm_blkc_m4_subband_diagnostic_2026_10_13/diagnostic.json`（900 complex headline）
- `molmetal/reports/cfm_blkc_m4_band_sweep_1.30/1.45/1.50_2026_10_13/diagnostic.json`
- `molmetal/reports/cfm_l3_decode_enabled_ab/control/arm_control.json`
- `molmetal/data/tmqm_clouds.py:263,300,424,429,539,549,554,557-565,588,597-606,700,705`
