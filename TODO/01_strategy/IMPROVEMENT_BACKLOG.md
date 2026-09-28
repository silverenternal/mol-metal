# IMPROVEMENT_BACKLOG — 小规模改进 backlog (multi-track, small-scale)

**Date**: 2026-09-25 (R17 close-out)
**Purpose**: 多做改进；不依赖大训练；快速 ship
**Constraint**: 实验规模 ≤ 单 GPU 几小时 / CPU ≤ 30 min
**Audience**: Hugo Lee + ultracode agents
**Status**: Active — back-logged items 立即可启动

---

## 0. 现状基线 (2026-09-25)

| 维度 | 当前 | 我们要改进的地方 |
|---|---|---|
| 文档 | README 200 / INDEX 150 / EXECUTION_ORDER 210 | 已精简 ✅ |
| Skills | 5 个 SHIPPED | 文档 + examples 待补 |
| Tests | 1604+ passed, 7 skipped, 16 pre-existing fail | 把 16 pre-existing fail 修掉 |
| 14 evaluators | 14/14 PASS | ✓ 不需改 |
| 7 pitfall guards | 7/7 PASS | ✓ 不需改 |
| 9 ultracode workflows | 9/10 完成 | ✓ R17 drain |
| 5 math ref 必引 | 部分已加 (Deflex + PLL + Krasnov + Rusanov) | 还有 9 篇要加 |
| paper sections | §3-§7 写好 | 还要 §3.4 SE(3) Schrödinger Bridge (R18) |
| TODO dir 子目录 | 还没建 | 可整理 |

---

## 1. 改进总览 (28 items, 7 类)

| 类别 | 数量 | 总 effort | 优先级 |
|---|---|---|---|
| A. 文档 / 路径 / 命名 | 5 items | 1 day | 🔴 HIGH |
| B. Code quality / error messages | 4 items | 1.5 day | 🟡 MED |
| C. 测试覆盖 + 修 16 pre-existing fail | 4 items | 2 day | 🟡 MED |
| D. Diagnostic + 可观测性 | 3 items | 1 day | 🟡 MED |
| E. 数学引用补全 (9 篇) | 9 items | 1 day | 🟡 MED |
| F. Skill examples + accuracy | 3 items | 0.5 day | 🟢 LOW |
| **总计** | **28 items** | **~7 day** | — |

每个 item ≤ 4h effort；适合 ultracode 单 agent 一次性跑完。

---

## 2. 详细 items

### A. 文档 / 路径 / 命名 (5 items)

| ID | Item | Effort | 优先级 |
|---|---|---|---|
| **A1** | 把 R17 之前的 legacy docs (`pending/26_round13_round14_complete_plan.md`, `AUDIT_RESEARCH_GRADE.md`, etc.) 标 "LEGACY — pre-R17, see INDEX.md" | 30 min | 🔴 |
| **A2** | 创建 `TODO/01_strategy/`, `02_metrics/`, `03_plans/`, `04_history/` 子目录（按 EXECUTION_ORDER §3） | 1h | 🔴 |
| **A3** | 移动根目录 docs 到子目录（需要更新所有 cross-refs） | 2h | 🔴 |
| **A4** | 给每个 R17 完成文档 (`completed/35-41`) 加一行 "supersedes TODO-XX" | 30 min | 🟡 |
| **A5** | 把 `B5_10POCKETS_INPROGRESS.md` 归档（已 obsolete）→ `archive/` | 15 min | 🟢 |

### B. Code quality / error messages (4 items)

| ID | Item | Effort | 优先级 |
|---|---|---|---|
| **B1** | `molmetal/scripts/metallo_drug_smoke_retrain.py`: 加 `--smoke` flag（200-step fast 验证） | 1h | 🟡 |
| **B2** | `molmetal/adapters/flow_matching_lipman/__init__.py`: 改进 error message — 当 `atom_vocab` OOV 时打印哪些原子是 OOV（而不只是断言失败） | 2h | 🟡 |
| **B3** | `molmetal/molmetal_lam/search_alg/proof_search.py`: 加 `--debug-mcts` flag 输出每步 UCB / PUCT 计算 | 1h | 🟢 |
| **B4** | `molmetal/scripts/r17_gpu_retrain.py`: 加 `--dry-run` 验证 CLI + GPU 检测但不起训练 | 30 min | 🟢 |

### C. 测试覆盖 + 修 16 pre-existing fail (4 items)

| ID | Item | Effort | 优先级 |
|---|---|---|---|
| **C1** | 修 `test_atom_training_contract.py::test_atom_targets_are_supervised_but_not_supplied_as_features`（vocab mask F3 改了 random-init CE 假设） | 2h | 🟡 |
| **C2** | 修 5 个 test_rocm / test_lipman / test_pocket_conditioned 测试（pre-existing fail） | 4h | 🟡 |
| **C3** | 给 `molmetal/reports/wf_*` 加 conftest.py fixture 让 pytest 自动 skip `molmetal/reports/wf_*/final.md` 这种 markdown 文件 | 30 min | 🟢 |
| **C4** | 加 `molmetal/tests/test_r17_legacy_compat.py` — 验证 R17 修复后 legacy 行为不变 | 2h | 🟢 |

### D. Diagnostic + 可观测性 (3 items)

| ID | Item | Effort | 优先级 |
|---|---|---|---|
| **D1** | `molmetal/scripts/diagnose_metric.py` — 给定 metric name，自动跑对应 unit test + 打印 failure mode + suggested fix (从 `MECHANISM_METRIC_MAPPING.md` §4 自动抽取) | 3h | 🟡 |
| **D2** | `molmetal/scripts/wf_pitfall_guard/check_all.py` 加 JSON output（让 CI 能 parse） | 1h | 🟢 |
| **D3** | `paper/figures/fig*.py` 加 README — 解释每个 figure 怎么生成 + 怎么更新 | 1h | 🟢 |

### E. 数学引用补全 (9 papers from METRICS_TARGETS §8)

| ID | Paper | bibtex key | Effort | Priority |
|---|---|---|---|---|
| **E1** | TransDiffSBDD 2025 | `transdiffsbdd2025` | 15 min | 🔴 (paper §2 必引) |
| **E2** | DecompDiff 2024 | `guan2024decompdiff` | 15 min | 🔴 |
| **E3** | MolCRAFT 2024 | `molcraft2024` | 15 min | 🟡 |
| **E4** | TargetDiff 2023 | `guan2023targetdiff` | 15 min | 🔴 |
| **E5** | DiffSBDD 2024 | `schneuing2024diffsbdd` | 15 min | 🔴 |
| **E6** | Pocket2Mol 2022 | `peng2022pocket2mol` | 15 min | 🟡 |
| **E7** | AlphaDrug 2024 | `alphadrug2024` | 15 min | 🟡 |
| **E8** | MolChord 2025 | `molchord2025` | 15 min | 🟡 |
| **E9** | Boltz-2 2025 | `boltz2_2025` | 15 min | 🟢 |

（Krasnov, Rusanov, Deflex, PLL 已加；这 9 篇补全必引清单）

### F. Skill examples + accuracy (3 items)

| ID | Item | Effort | 优先级 |
|---|---|---|---|
| **F1** | `mol-metal-mechanism-mapping` SKILL.md 加 3 个使用 example（脚本输出 → 推荐机制检查 → pytest 命令） | 2h | 🟡 |
| **F2** | 5 个 skill 各加 1 个 "common mistake" 段落（user 容易踩的坑） | 3h | 🟢 |
| **F3** | 把 R17 实际跑通的全 pipeline 写成一个 `molmetal/scripts/wf_full_pipeline.sh`（end-to-end demo） | 2h | 🟢 |

---

## 3. 不在 backlog 的（明确不做）

- ❌ R17 Tier-1 AUC fix（3-seed × 100-epoch training）— 这是 Phase A in `R17_AUC_RECOVERY_PLAN.md`，**不在这**（实验规模超出 small-scale）
- ❌ R18 SE(3) Schrödinger Bridge 主推（3 weeks）— 在 `R18_PLAN.md`，**不在这**
- ❌ 100-pocket × 3-seed paper-grade sweep（GPU 6h）— 在 `pending/14_full_100pocket_paper_r13.md`
- ❌ Wet-lab validation — 在 `R18_PLAN.md` Phase C
- ❌ NCI-60 full 60-cell（现在是 9-cell subset）— 推到 R19

---

## 4. 推荐的 ultracode 编排 (3 个并行)

### ultracode A: 文档整理 + 数学引用 (1.5 天)
- A1, A2, A3, A4, A5 (文档整理) — agent A
- E1-E9 (9 篇数学引用补全) — agent B (parallel)

### ultracode B: Code quality + 测试 fix (2 天)
- B1, B2, B3, B4 (code quality) — agent A
- C1, C2, C3, C4 (test fix) — agent B (parallel)

### ultracode C: Diagnostic + Skill examples (1.5 天)
- D1, D2, D3 (diagnostic) — agent A
- F1, F2, F3 (skill examples) — agent B (parallel)

**总计**: 3 ultracode × ~2 day each = ~6 wall-day (with parallelism)
**实际**: ~3-4 day wall if 3 ultracode parallel

---

## 5. Success Criteria

每个 ultracode 完成后：
- [ ] 所有列出的 items 都 shipped
- [ ] `molmetal/scripts/eval_all_evaluators.py` 14/14 PASS（无 regression）
- [ ] `molmetal/scripts/wf_pitfall_guard/check_all.py` 7/7 PASS（无 regression）
- [ ] 至少 +5 unit tests PASS（测 C1, C2, D1 等的 regression）
- [ ] `TODO/completed/4X_*.md` 写清楚每 item 怎么 ship 的
- [ ] `paper/refs.bib` 增 9 entries (E1-E9) 后 `pdflatex 4 passes` 0 unresolved
- [ ] `.claude/skills/` 5 个 skill 都有 R17 production-state + improvement items updates

---

## 6. 不要做

- ❌ 任何 100-pocket × 3-seed 重跑（GPU 6h）
- ❌ 任何 5-metal multi-task 训练（12h GPU）
- ❌ 任何 SE(3) Schrödinger Bridge 实现（3 weeks）
- ❌ 任何 wet-lab
- ❌ 任何 24-paper Frontier refs.bib 集成（已被 M-R17 完成）
- ❌ 任何 CI guard 重写（已被 CI+docs 完成）
- ❌ 任何 NCI-60 扩展到 60 cell line

这些在 `R17_AUC_RECOVERY_PLAN.md` + `R18_PLAN.md` 里都有专门规划，**不在 small-scale improvement backlog**。

---

## 7. 与 EXECUTION_ORDER.md 的关系

`EXECUTION_ORDER.md` 优先级排序：
1. **Phase A** (R17 Tier-1 AUC fix, 2 周) — **大实验**，不在本 backlog
2. **Phase B** (R18 数学升级, 8 周) — **大实验**，不在本 backlog
3. **本 backlog** (28 items, 7 天) — **小改进，可与 Phase A/B 并行**

**建议**: **小改进与 Phase A/B 并行跑**。Phase A 用 GPU 时（晚上），ultracode 跑改进。

---

## 8. 下一步

1. ✅ 写本规划 `TODO/01_strategy/IMPROVEMENT_BACKLOG.md`（本文）
2. ⏳ 启动 ultracode A: 文档整理 + 9 篇数学引用（agent A + agent B parallel, 1.5 天）
3. ⏳ 启动 ultracode B: code quality + test fix（agent A + agent B, 2 天）
4. ⏳ 启动 ultracode C: diagnostic + skill examples（agent A + agent B, 1.5 天）
5. ⏳ 等 3 ultracode 完成 → `TODO/completed/4X_*.md` 写最终 report
6. ⏳ 如果 Phase A 在跑，与本 backlog 并行 ship

---

## 9. 用户决策点

| Decision | 选项 |
|---|---|
| **D1**: 优先级 — A 先做还是 B/C 先做? | A) 文档 + 数学引用 (paper 必引) / B) code quality / C) diagnostic + skill |
| **D2**: 多少个 ultracode 并行? | A) 3 (推荐, 快) / B) 6 (更快) / C) 1 (sequential, 慢) |
| **D3**: 是否等 Phase A 完成再启动? | A) 是 / **B) 否, 与 Phase A 并行** (推荐) |
