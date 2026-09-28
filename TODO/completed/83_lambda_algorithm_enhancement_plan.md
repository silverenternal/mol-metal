# TODO-83 — Lambda 算法能力加强 (post-Wf-35 refactor)

**Date**: 2026-09-26
**Status**: 🚀 **Active** (post-Wf-29~35 ship)
**Priority**: **HIGH** — Lambda 算法端是 R18 close-out 核心,但还有 8 个明显漏洞未 ship
**Owner**: Hugo Lee + ultracode agents
**Parent**: `WEAK_TO_STRONG_PLAN.md` §1.2 + `R19_PLAN.md` §8
**Skill alignment**: mol-metal-lambda-calculus + mol-metal-r18-math + mol-metal-mechanism-mapping + party-twenty-spirit

---

## §0. Why this plan exists

Wf-1~35 全部 ship,但 **Lambda 算法端 (R17 + R18 真活) 有 8 个明显能力漏洞** 没补:

| Gap | 现有能力 | 缺什么 | 影响 |
|---|---|---|---|
| **L1** | `lambda_combinators.py` R17 ship | **缺**: Closure theorem 端到端 verifier + 性能 bench | m-mcts (closure) 实际跑得通没验证 |
| **L2** | `proof_search.py` R17 ship | **缺**: Variance MCTS 真正接入 proof_search (Wf-14 ship variance_mcts 但没整合到 proof_search) | proof_search 仍然用旧 Dirichlet PUCT |
| **L3** | `score_repellent_mc.py` Wf-9 + `variance_puct_integration.py` Wf-11 + `variance_mcts_integration.py` Wf-14 | **缺**: 三者联合使用 (ScoreRepellent → VariancePUCT → MCTS) | 实际 MCTS trajectory 没测 |
| **L4** | `mcts_replay_buffer.py` Wf-21 ship | **缺**: Replay → 实际训练 (现在只是 buffer,没有 learner/trainer) | replay buffer 是死的 |
| **L5** | `hilbert_astsr_integration.py` Wf-14 + `hilbert_primal_dual.py` Wf-9 | **缺**: Hilbert-ASTSR end-to-end training script (用真实 IC50 data) | Wf-25 跑出 R² -0.443 (numerical instability) |
| **L6** | `closure.py` R17 ship + `closure_real_benchmark.py` Wf-21 | **缺**: Closure theorem 在 d-block metal complex 上的 benchmark (实际 30-d-block) | 现在只测试 cisplatin 类似的简单 case |
| **L7** | `pll_signature.py` R17 + `multi_ph_pll_integration.py` Wf-14 | **缺**: multi-PH + PLL 实际端到端 feature extraction (Wf-25 R² instability) | Wf-25 R² -0.443 是 numerical bug, 没真正测 lift |
| **L8** | `coupling_adapter.py` R17 + `coupling_joint_inference.py` Wf-4 + `hilbert_astsr_integration.py` Wf-14 | **缺**: ASTSR ↔ Hilbert ↔ MCTS 三角闭环 | 现有只是单向耦合 |

---

## §1. Cross-skill strategy

利用现有 skills:

| Gap | Primary skill | Secondary skill |
|---|---|---|
| L1 Closure verifier | mol-metal-lambda-calculus | mol-metal-mechanism-mapping |
| L2 MCTS variance | mol-metal-lambda-calculus | mol-metal-r18-math (M26 Score-Rep) |
| L3 Joint MCTS | mol-metal-lambda-calculus | mol-metal-r18-math |
| L4 Replay learner | mol-metal-lambda-calculus | mol-metal-cfm-training |
| L5 Hilbert ASTSR | mol-metal-lambda-calculus | mol-metal-r18-math (P1-4) |
| L6 Closure d-block | mol-metal-lambda-calculus | mol-metal-click-chemistry |
| L7 Multi-PH + PLL | mol-metal-lambda-calculus | mol-metal-r18-math (T5) |
| L8 Triangle closure | mol-metal-lambda-calculus | mol-metal-cfm-training |

---

## §2. 4 双线 phases (4 wf × parallel lines)

### Line 1 (核心算法加强): Wf-36 + Wf-37

**Wf-36: Lambda algorithm core strengthening** (L1-L4)
- Phase A: Closure theorem end-to-end verifier + benchmark (L1)
- Phase B: Variance MCTS 真正接入 proof_search (L2)
- Phase C: ScoreRepellent + VariancePUCT + MCTS 三者联合 (L3)
- Phase D: MCTS replay learner (AlphaGo-style) (L4)

**Wf-37: Lambda ↔ ASTSR ↔ Hilbert triangle** (L5-L8)
- Phase A: Hilbert-ASTSR end-to-end training with real IC50 (L5)
- Phase B: Closure theorem d-block metal complex benchmark (L6)
- Phase C: Multi-PH + PLL 实际端到端 feature extraction (L7)
- Phase D: ASTSR ↔ Hilbert ↔ MCTS 三角闭环 (L8)

### Line 2 (配套文档): Wf-38
**Wf-38: Lambda algorithm documentation** (CPU-only docs)
- Lambda algorithm roadmap (R20-R25)
- Lambda 端 benchmark suite 完整化
- 8 skill cross-link (与 mol-metal-lambda-calculus 等)

---

## §3. Wf-36 详细 plan (4 phases)

### Phase 1: Closure theorem end-to-end verifier (L1)

| Task | 输出 |
|---|---|
| `molmetal/scripts/verify_closure_theorem.py` (NEW, ~150 LOC) | 跑 closure theorem across 100+ 分子 (含 d-block 30 metal) |
| `molmetal/tests/test_closure_theorem_verifier.py` (NEW, ~15 tests) | closure reachability 100% pass |
| `molmetal/reports/wf36_closure_theorem/closure_bench.json` (NEW) | per-metal closure rate |
| `molmetal/scripts/verify_closure_end_to_end.py` (NEW, ~80 LOC) | smoke |

### Phase 2: Variance MCTS 真正接入 proof_search (L2)

| Task | 输出 |
|---|---|
| `molmetal/scripts/verify_variance_mcts_integration.py` (NEW, ~150 LOC) | 跑 100 trajectories 对比 VariancePUCT vs Dirichlet |
| `molmetal/tests/test_variance_mcts_proof_search.py` (NEW, ~15 tests) | variance_mcts ↔ proof_search 联合 API |
| 修改 `proof_search.py` 加 variance_mcts option (1-line 改动: arg `--exploration=dirichlet / score_repellent`) |
| `molmetal/reports/wf36_variance_mcts/variance_bench.json` (NEW) |

### Phase 3: ScoreRepellent + VariancePUCT + MCTS 三者联合 (L3)

| Task | 输出 |
|---|---|
| `molmetal/scripts/joint_score_repellent_variance_puct_mcts.py` (NEW, ~200 LOC) | 完整 3-stage pipeline |
| `molmetal/tests/test_joint_3stage_mcts.py` (NEW, ~15 tests) | 三者 API 联合 OK |
| `molmetal/reports/wf36_3stage_mcts/3stage_bench.json` (NEW) |

### Phase 4: MCTS replay learner (L4)

| Task | 输出 |
|---|---|
| `molmetal/scripts/mcts_replay_learner.py` (NEW, ~250 LOC) | 真正 train value/policy network on replay buffer |
| `molmetal/tests/test_mcts_replay_learner.py` (NEW, ~15 tests) | 训练 loss 收敛 |
| `molmetal/reports/wf36_replay_learner/learner_bench.json` (NEW) |

**Verify**: 4 个 bench JSON + 60+ tests pass

---

## §4. Wf-37 详细 plan (4 phases)

### Phase 1: Hilbert-ASTSR end-to-end training with real IC50 (L5)

| Task | 输出 |
|---|---|
| `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (NEW, ~280 LOC) | 真实 IC50 + Hilbert ASTSR |
| `molmetal/tests/test_train_hilbert_astsr_end_to_end.py` (NEW, ~15 tests) | 数值稳定性 (Wf-25 R² instability fix) |
| `molmetal/reports/wf37_hilbert_astsr/r2_bench.json` (NEW) | R² MEASURED vs evolutionary |

### Phase 2: Closure theorem d-block metal complex benchmark (L6)

| Task | 输出 |
|---|---|
| `molmetal/scripts/bench_closure_dblock.py` (NEW, ~200 LOC) | 30 d-block metal × closure rate |
| `molmetal/tests/test_closure_dblock.py` (NEW, ~15 tests) | per-metal closure rate |
| `molmetal/reports/wf37_closure_dblock/per_metal_bench.json` (NEW) | Pt/Ru/Ir/Rh/Re/Os/Au/Pd/Ag 等 per-metal |

### Phase 3: Multi-PH + PLL 实际端到端 feature extraction (L7)

| Task | 输出 |
|---|---|
| `molmetal/scripts/extract_multiph_pll_end_to_end.py` (NEW, ~250 LOC) | fix Wf-25 R² numerical bug + 真实 extraction |
| `molmetal/tests/test_multiph_pll_extraction.py` (NEW, ~15 tests) | 数值稳定 + 5-seed run OK |
| `molmetal/reports/wf37_multiph_pll/extraction_bench.json` (NEW) | real features + R² |

### Phase 4: ASTSR ↔ Hilbert ↔ MCTS 三角闭环 (L8)

| Task | 输出 |
|---|---|
| `molmetal/scripts/triangle_closure.py` (NEW, ~300 LOC) | ASTSR ↔ Hilbert ↔ MCTS 三方闭环 |
| `molmetal/tests/test_triangle_closure.py` (NEW, ~15 tests) | 三方 API 联合 OK |
| `molmetal/reports/wf37_triangle/triangle_bench.json` (NEW) |

**Verify**: 4 个 bench JSON + 60+ tests pass

---

## §5. Wf-38 详细 plan (4 phases, CPU-only docs)

### Phase 1: Lambda algorithm roadmap

| Task | 输出 |
|---|---|
| `TODO/03_plans/LAMBDA_ROADMAP.md` (NEW, ~250 lines) | R20-R25 Lambda algorithm 6 phases |

### Phase 2: Lambda benchmark suite 完整化

| Task | 输出 |
|---|---|
| `molmetal/scripts/lambda_bench_suite.py` (NEW, ~250 LOC) | 整合 Wf-36/37 全部 bench |
| `molmetal/tests/test_lambda_bench_suite.py` (NEW, ~15 tests) |

### Phase 3: 8 skill cross-link

| Task | 输出 |
|---|---|
| 8 SKILL.md 加 R20 Lambda reference 段 | 每 skill +30 lines |

### Phase 4: Synthesis

| Task | 输出 |
|---|---|
| `TODO/completed/83_lambda_enhancement.md` (NEW, ~250 lines) | final synthesis |

---

## §6. Execution timeline

| Workflow | Phase | Wall-time | 依赖 |
|---|---|---|---|
| **Wf-36** | L1-L4 | 2 days | Wf-25/14 ships |
| **Wf-37** | L5-L8 | 2 days | Wf-31 ships |
| **Wf-38** | Docs | 1 day | (独立) |

**3 wf 双线并行**: ~3 days wall-time (节省 ~2 days vs 串行)

---

## §7. Honest framing

按 party-twenty-spirit §3.5:
- **Wf-25 R² numerical instability** (-0.443) 仍未解决 — Wf-37 L5 尝试 fix,但若仍失败,诚实报告 honest-negative
- **multi-PH + PLL integration** L7 — 实际可能仍 R² 数值问题,不假装 ship
- **variance_mcts ↔ proof_search** L2 — 实际接可能性能不如 Dirichlet, 报告 honest
- **replay learner** L4 — AlphaGo-style trainer 可能没 GPU 跑不动, CPU-only 验证 + paper §3 引用
- **closure d-block** L6 — 30 metal × closure rate 可能某些 metal (e.g., Tc, Pm) 不稳定, 报告 honest

---

## §8. Cross-reference

- `WEAK_TO_STRONG_PLAN.md` §1.2 8 WEAK metrics
- `R19_PLAN.md` §8-12 R19 Lambda algorithm plan (R19 6 phases)
- `R18_PLAN.md` §15 R18 close-out (32 modules ship, 但 Lambda 端 8 gap 仍 open)
- `TODO/pending/82_metric_gap_filling_plan.md` (7 gap Wf-29~34 跑)
- `TODO/pending/83_lambda_algorithm_enhancement.md` (this plan, 8 gap Wf-36~38 跑)
- `.claude/skills/mol-metal-lambda-calculus/SKILL.md` — 主 skill for this plan

---

## §9. Final verdict

**8 gap 是 Lambda 端的核心算法能力缺**, 不是 docs 缺。

If Wf-36 + Wf-37 + Wf-38 都 ship:
- Lambda 端 8 个漏洞补完
- ASTSR ↔ Hilbert ↔ MCTS 三角闭环 ship
- Lambda algorithm roadmap (R20-R25) ready
- 与 R19_PLAN §8-12 配合: R19 真正能跑 R18 modules + R20 计划 Lambda 算法继续推进

If 部分 gap 没 ship, those become R20-R25 plan items.

**Date**: 2026-09-26 (created)
