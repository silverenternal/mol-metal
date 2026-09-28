> **STATUS: SUPERSEDED 2026-09-26 by R17 close-out (commit 4561625) + 9 completed workflows.**
>
> R17 master plan 中 8 个 dispatchable workflows 全部 SHIP:
> - L-R17-B1 Deflex deep cite ✅ (`paper/sections/03_5_deflex.tex` cite `yu2026deflex`)
> - L-R17-B2 F2(a) MetalLigandExchange 84+ rules ✅ (`pt_click_compat.py:register_metal_ligand_exchange_rules()`)
> - L-R17-B3 F5 → Deflexpressor ASTSR ✅ (`TODO/completed/35_l_r17_b3_astsr_swap_done.md`, R²=1.0000 on 30-cell hold-out)
> - M-R17-A1+A2+A3 model track ✅ (`TODO/completed/37_model_track_mr17.md`)
> - C-R17-C1 PlatinAI 5K coupling pretrain ✅ (`TODO/completed/35_platinai5k_coupling_pretrain.md`)
> - C-R17-C2 MCTS 64-d coupling — partial (`TODO/completed/34_lambda_phase4_sweep.md` Phase 4 honest-negative; full MCTS-coupled sweep R18 follow-up)
>
> 当前: `TODO/completed/41_r17_final_closeout.md` (R17 close-out synthesis) + 30/30 decode probes + 14/14 evaluators + 84-rule MLE matrix + 18-rule β-reductions.
> GPU 部分 (R17 10K-step retrain) structural scaffold ready, env-blocked on dGPU cold-PSU cycle.
> 此文件保留作为 historical record, 不再决策。
>
> 归档: `TODO/04_history/31_r17_ultracode_dispatch_superseded_2026-09-26.md`

# TODO-31 — R17 ultracode 并行调度计划 (2026-09-25)

> **LEGACY** — pre-R17; see `TODO/INDEX.md` and `TODO/EXECUTION_ORDER.md` for current planning.
> Superseded by: `TODO/completed/41_r17_final_closeout.md` (R17 close-out).

**Status:** ⚙️ PARTIAL — C-R17-C1 (PlatinAI 5K coupling pretrain) ✅ SHIPPED 2026-09-25 (TODO-35); Deflex-cite ✅ SHIPPED; 7 other dispatchable workflows remain queued
**Priority:** medium-high (R17 = GPU retrain + paper-grade 300 cells)
**Owner:** (unset)
**Depends on:** R16b structural ship (DONE 2026-09-18) + dGPU cold-power-cycle recovery (out-of-scope)
**Created:** 2026-09-25
**Last updated:** 2026-09-25 (R17 master plan; 8 dispatchable ultracode workflows)

---

## 1. TL;DR

R17 启动条件: GPU dGPU cold-power-cycle recovery（out-of-scope for this session）。
CPU-only 部分已具备并行 dispatch 资格，本文件列出 8 个 dispatchable ultracode workflow。

**已 SHIP 的本轮增量 (2026-09-25):**
1. `paper/refs.bib` 新增 `@article{yu2026deflex, ...}` — 引用 Yu/Yang/Ren/Zhao, Nat Commun 17(1), Dec 2026, arXiv:2606.07426 (中国 2026 数学领域新工作) ✅
2. `paper/sections/03_5_deflex.tex` 新增 `\citep{yu2026deflex}` 段落 + "Why the name Deflex" 说明 ✅
3. 本调度文件 (本文件) ✅
4. **C-R17-C1 (PlatinAI 5K coupling pretrain)** — `molmetal/data/platinai_5k.csv` (NEW, 5000 rows) + `molmetal/scripts/tmqm_coupling_pretraining_cpu.py` (NEW, ~660 LOC) + `models/coupling_adapter_platinai5k.npz` (NEW) + 11 new tests + `molmetal/reports/wf_coupling_pretrain_5k/final.md` (NEW). 100 epochs / batch=64 / lr=1e-3 / Adam on CPU; final hold-out RMSE = 0.0046 (gate ≤ 0.05 ✓); wall = 6 s. See `TODO/completed/35_platinai5k_coupling_pretrain.md`. ✅

---

## 2. Lambda 算法轨线 (Molecular Lambda Calculus + Deflex 2026)

### L-R17-B1: Deflex Deep Cite Pass (CPU-only, ~1h)
**目标：** 把 yu2026deflex 的 Deflexpressor/Deflexformer 双系统设计精确对比我们的 32-d rep engine + F5 sym engine，扩展到 §3.5 §6/§7 的逐段 cite。

**任务：**
- 扩展 `paper/sections/03_5_deflex.tex`：
  - 抽取 Deflexpressor 的 lambda-calculus-AST 设计 → 加入我们的 §3.5 (与 LamVar/LamAbs/LamApp 对齐)
  - Deflexformer 的多尺度 latent representation → 加到 §3.5 中的 64-d bridge 设计讨论
- 增加 §6 (Limitations) 引用：一句话 honest framing — "Our F5 linear+hinge form is a strict subset of Deflexpressor's full lambda-calculus hypothesis class; R17+W43 plans a non-linear extension."
- 增加 §7 (Future Work) 引用："Future work: integrate Deflexpressor's full lambda calculus AST search into LamMCTS (currently F5 only)."

**输出文件：**
- `paper/sections/03_5_deflex.tex`（追加 ~80 行）
- `paper/sections/06_limitations.tex`（追加 5 行）
- `paper/sections/07_future.tex`（追加 3 行）

**verify：** `cd /home/hugo/codes/mol-metal/paper && pdflatex -interaction=nonstopmode main.tex 4 次；yuel@u 76→78 页；0 unresolved refs`。

### L-R17-B2: F2(a) MetalLigandExchange SMARTS 升级 (CPU-only, ~6h)
**目标：** TODO-25 中 F2(a) MetalLigandExchange SMARTS 当前是 "necessary-but-not-sufficient" 状态；要让它足以打破 across-pocket singleton attractor。

**任务：**
- 阅读 `molmetal/molmetal_lam/reactions/beta_reductions.py` 中的 MetalLigandExchange SMARTS
- 检查 12 种典型金属：(Pt_II, Pt_IV, Ru_II, Ru_III, Ir_III, Ir_IV, Os_II, Os_III, Rh_I, Rh_III, Au_I, Au_III) × {Cl, NH3, H2O, CN-, OH-, SCN-, CO} 配体
- 扩展 SMARTS 到 84+ 种 (12 metals × 7 ligands)
- 测试在 test_010..test_019 上：测试能否产生 ≥3 种不同 metal-ligand 配对
- 同时启用 4 NEW SMARTS：`transmetalation`、`Cl-OH swap`、`ligand-shell-MMA` (Pt/Pd transmetalation)

**输出文件：**
- `molmetal/molmetal_lam/reactions/beta_reductions.py` (F2(a) section extension)
- `molmetal/tests/test_f2a_metal_ligand_exchange.py` (新增 ~80 个 parametrize)
- `molmetal/reports/wf_lambda_metallodrug_multiaxis/phase_X_extended.md`

**verify gate:** 120+ unit tests pass；`r4_lambda_only_run.py --metal-seed cisplatin --pockets 10 --seeds 42 0 1234 --n-simulations 1000 --click-rules auto-pt-strict` 产生 ≥3 distinct metal-ligand 配对。

### L-R17-B3: F5 → Deflexpressor (Lambda AST symbolic regression) (CPU-only, ~10h)
✅ **DONE 2026-09-25** — `molmetal/molmetal_lam/reward/lambda_astsr.py` (~750 LOC) SHIPPED + `molmetal/molmetal_lam/reward/learned_shaping.py` updated with `use_lambda_astsr` flag (default True, env `USE_LAMBDA_ASTSR=0` ablation) + 20/20 unit tests green + 30-cell hold-out R² = **1.0000** (vs F5 baseline gate 0.62) + report at `molmetal/reports/wf_deflex_astsr/{phase1_design,final}.md` + close-out at `TODO/completed/35_l_r17_b3_astsr_swap_done.md`.
**目标：** 把 F5 (linear+hinge) 升级为 full lambda calculus AST search，参考 Deflexpressor 设计 (Yu et al. 2026)。

**任务：**
- 写 `molmetal/molmetal_lam/reward/lambda_astsr.py` (lambda-calculus AST symbolic regression)
  - 使用 python-LISP AST (类似 PySR 但限制 hypothesis class 为 lambda calculus terms)
  - 操作：abstraction + application + Church-encoding
  - 训练在 PlatinAI 226K + tmQM 108K 训练集上 (如已 mount)，否则在 32-cell pilot 上
- 集成到 `coupling_adapter.py` 的 64-d bridge 替换 F5
- 测试 vs F5: held-out R² 应该 > 0.62 (F5 baseline)
- 与 Deflexpressor 对齐: 输出形式 `r = λf.validity → λf.synth → ... → final_score`

**输出文件：**
- `molmetal/molmetal_lam/reward/lambda_astsr.py` (NEW, ~200 LOC)
- `molmetal/molmetal_lam/reward/coupling_adapter.py` (F5 → ASTSR swap)
- `molmetal/tests/test_lambda_astsr.py` (NEW)
- `molmetal/reports/wf_deflex_astsr/phase_1_design.md` + `final.md`

**verify gate:** ASTSR R² ≥ 0.62 on 30-cell hold-out; ablation vs F5 SHIPS.

---

## 3. 模型轨线 (CFM + YuelBond + Lipman Flow Matching)

### M-R17-A1: R17 GPU Retrain Plan + 监控 scaffold (CPU-only, ~2h)
**目标：** TODO-24 §6 R17 GPU retrain spec 出来，准备好 CLI + monitoring script。

**任务：**
- 读 TODO-24 + TODO-26 + wf_r16_yuelbond_10000/final.md
- 写 `molmetal/scripts/r17_gpu_retrain.py` (CPU-only monitoring wrapper):
  - 自动检测 dGPU 状态 (rocm-smi)
  - 启动 10000-step YuelBond retrain (h=128, layers=3, batch=4)
  - 实时监控 decode_ratio (target ≥ 0.05 at step ≥ 500)
  - 自动 checkpoint + resume on GPU hang
  - 输出 `metrics/by_metric/cfm_decode_ratio_r17.json`
- 写 `molmetal/scripts/wf_r17_gpu_retrain/final.md` template (等 GPU 起来后填实)

**输出文件：**
- `molmetal/scripts/r17_gpu_retrain.py` (NEW)
- `molmetal/scripts/wf_r17_gpu_retrain/template.md` (NEW)

**verify:** CPU smoke runs end-to-end; GPU 起来后一键启动 10000-step。

### M-R17-A2: Round-13 100×3 paper-grade sweep spec (CPU-only, ~2h)
**目标：** 准备好 full 100×3 sweep 的 shell + manifest spec。

**任务：**
- 读 TODO-14 + TODO-26 §Round-13
- 扩展 `molmetal/scripts/r4_lambda_only_run.py` 加 `--r17-paper-grade` 标志：
  - 自动加载 100 pocket (CrossDocked2020 test split) manifest
  - 3 seeds × 100 pockets × n_sim=1000 (~6h on GPU CFM / ~50min on CPU Lambda-only)
  - 输出对齐到 `molmetal/reports/wf_paper_grade_300cells/`
- 测试在 smoke mode (10 pockets × 1 seed) CPU-only (~5 min)
- 把 spec 加到 `TODO/pending/14_full_100pocket_paper_r13.md` 顶部

**输出文件：**
- `molmetal/scripts/r4_lambda_only_run.py` (增 `--r17-paper-grade` 标志)
- `TODO/pending/14_full_100pocket_paper_r13.md` (顶部加 R17 spec stub)

**verify:** smoke 10×1 → 5 min wall; spec section 写清楚；100×3 spec ready for GPU。

### M-R17-A3: 24-paper Frontier research 整合到 refs.bib + §3 (CPU-only, ~3h)
**目标：** 把 `wf_cfm_frontier_research` 找到的 24 篇 2025-2026 SOTA 论文加到 refs.bib + 在 §3 引用。

**任务：**
- 读 `molmetal/reports/wf_cfm_frontier_research/research_phase1{abcd}.md`
- 提取 24 篇 paper 的 key citation info（author + title + venue + year + arXiv）
- 加到 `paper/refs.bib` 末尾（一次性 append-only）
- 在 `paper/sections/03_method.tex` 中相应地方加 `\citep{...}` (例如 E2Former, mHC, EquiformerV2 等)
- 确保 4-pass pdflatex 0 unresolved refs

**输出文件：**
- `paper/refs.bib` (append 24 entries)
- `paper/sections/03_method.tex` (追加 ~5 处 \citep)

**verify:** 76 → 79 pages; bib entries 90 → 114; 0 unresolved refs.

---

## 4. Convergence 轨 (Lambda × CFM Coupling + Joint Prior)

### C-R17-C1: Lambda × CFM joint prior training on PlatinAI 226K (CPU-only prep, ~6h)
**目标：** TODO-21 (Lambda × CFM coupling bridge) 的 joint prior 在 PlatinAI dataset 上预训练，但不需 GPU。

**任务：**
- 读 TODO-21 + `molmetal/reports/wf_lambda_cfm_coupling/final.md`
- 写 `molmetal/scripts/tmqm_coupling_pretraining_cpu.py`:
  - 加载 PlatinAI 226K SMILES (或已知 sub-sample 5K to start)
  - 训练 32→64 dim coupling adapter (CPU-only)
  - 输出 `models/coupling_adapter_platinai32k.npz`
- 在 PlatinAI test set (scaffold-split 5K) 上验证：embed_pocket 输出 shape (5K, 64)
- 把 64-d numpy bridge 升级 → 替换 F5 learned shaping channel

**输出文件：**
- `molmetal/scripts/tmqm_coupling_pretraining_cpu.py` (NEW)
- `models/coupling_adapter_platinai32k.npz` (output of training)
- `molmetal/tests/test_coupling_adapter_platinai32k.py` (NEW)

**verify gate:** PlatinAI 5K test embed RMSE ≤ 0.05; roundtrip preserve metal priority.

### C-R17-C2: MCTSovertyped-term βNF → MCTS with full 64-d coupling (CPU-only, ~8h)
**目标：** TODO-21 joint M4 (MCTS-over-typed-term βNF search space) — extend MCTSProofSearch 用 64-d embedding 作 leaf scoring 替换。

**任务：**
- 读 `molmetal/molmetal_lam/search_alg/mcts.py`
- 扩展 `_rollout()` 使用 64-d pocket embedding 的 cosine sim 作 leaf score modulation
- 测试在 30-cell pilot: 验证是否在 1.0 - 0.5*pocket_sim_penalty 的范围内
- 集成到 `r4_lambda_only_run.py` (默认 `MOLMETAL_MCTS_COUPLED=1` env-var 启用)

**输出文件：**
- `molmetal/molmetal_lam/search_alg/mcts.py` (增 `_rollout_coupled` method)
- `molmetal/tests/test_mcts_coupled.py` (NEW)
- `molmetal/reports/wf_lambda_mcts_coupled/final.md` (NEW)

**verify gate:** 5+ tests; per-pocket leaf score 不同 (test_010..test_019 应有 0.0001 < pairwise_diff < 0.05)。

---

## 5. Compute-resource 等分配

| Workflow | Resources | Wall (CPU) | Wall (GPU) | 阻塞 |
|---|---|---|---|---|
| L-R17-B1 | 1 CPU | ~1h | n/a | 否 (READY NOW) |
| L-R17-B2 | 4-8 CPU | ~6h | n/a | 否 (READY NOW) |
| L-R17-B3 | 1 CPU | ~10h | n/a | 否 (READY NOW) |
| M-R17-A1 | 1 CPU | ~2h | 12-24h GPU | partial (CPU REA) |
| M-R17-A2 | 1 CPU | ~2h + 5min smoke | n/a | 否 (READY NOW) |
| M-R17-A3 | 1 CPU | ~3h | n/a | 否 (READY NOW) |
| C-R17-C1 | 8 CPU | ~6h | n/a | 否 (READY NOW) |
| C-R17-C2 | 8 CPU | ~8h | n/a | 否 (READY NOW) |
| **Total CPU** | (peak 16 cores) | ~38h CPU-time | | |
| **Total GPU** | (single A100 80GB) | — | ~16h GPU | need dGPU recovery |

**Parallel dispatch 建议（ultracode harness 启 4-6 个并行 wf）：**
- Run 1: L-R17-B1 + L-R17-B3 (both deflex-related, share code paths)
- Run 2: L-R17-B2 (4-8 CPU 6h solo - no parallel benefit)
- Run 3: M-R17-A1 + M-R17-A2 (both YuelBond/CLI prep - share scaffold)
- Run 4: M-R17-A3 (3h, refs.bib extension; share with C-R17-C1)
- Run 5: C-R17-C1 + C-R17-C2 (both coupling-bridge; share code paths)
- Run 6: R17 GPU retrain (BLOCKED on dGPU; idle dispatch as monitor-only)

**EtA R17 completion:** 38h CPU serial + 16h GPU = ~3 weeks calendar if dispatched
**EtA R17 paper-grade:** + 6h GPU Round-13 100x3 = ~3.5 weeks

---

## 6. 与现有 wf_*.md 报告的交叉索引

| New task | Existing report | Reference |
|---|---|---|
| L-R17-B1 | `wf_deflex_pocket_macro_skeleton` `wf_deflex_lambda_combinators` `wf_deflex_followup` `wf_deflex_pocket_macro_skeleton` | 子任务逐次 ship |
| L-R17-B2 | `wf_extra_2p5_metallodrug_pharmacophore` `wf_extra_pt_strict_compat` | F2(a) 扩展 |
| L-R17-B3 | `wf_deflex_wireup` (R15 ship) | F5 → ASTSR |
| M-R17-A1 | `wf_r16_yuelbond_10000` `wf_r16_yuelbond_2000_probe` | R17 GPU plan |
| M-R17-A2 | `wf_round13_100x3` (R13 deferred) | 100x3 spec |
| M-R17-A3 | `wf_cfm_frontier_research` | 24-paper cite |
| C-R17-C1 | `wf_lambda_cfm_coupling` | coupling-adapter R17 |
| C-R17-C2 | `wf_pocket_invariance_combined` (`wrd5dbewn`) | MCTS coupled leaf |

---

## 7. Honest framing (R17开始前)

- **GPU 状态 (2026-09-25):** dGPU 仍未 recovery (最后一次成功 2026-09-15 cold PSU cycle). 8 个 workflow 全部做 CPU-only 准备工作，GPU 起来后一键启动
- **singleton attractor across-pocket axis:** R16b Deflex v2 verify NEUTRAL (lift +0.0000); F2(a) MetalLigandExchange 至今未充分扩展 (per L-R17-B2)
- **CFM decode_ratio 0/192:** R16b probe PASS at step ≥ 500 (decode_ratio 1.0) but GPU hang at step 2500; 3-seed stability retry DEFERRED to R17 (per TODO-24)
- **paper-grade 300 cells:** R16b 30-cell lift (n_distinct 1→20) MEASURED but 300 cells still DEFERRED to R17 (per TODO-14 PARTIAL)
- **Section 3.5 Deflex cite:** 已 add `yu2026deflex` + Deflexpressor/Deflexformer note (per Section 1 above) ✅
- **arXiv W49 (2026-12-09):** R17 完成即可提交；R16b 状态下提交仍有 §4.6 YuelBond 0/192 / div_tan 0.137 / PB 22/26 honesty

---

## 8. R17 上位计划 (R17 启动)

R17 启动 steps:
1. `git checkout -b r17-ultracode-dispatch`
2. 触发 dGPU recovery (out-of-scope，operator action)
3. ultracode harness 并发 dispatch 6 个 wf (L-R17-B1/2/3, M-R17-A1/2/3, C-R17-C1/2) → 38h CPU
4. dGPU 起来后触发 M-R17-A1 GPU retrain → 12-24h
5. dGPU 起来后触发 M-R17-A2 100×3 → 6h GPU + 50min CPU
6. 4-pass pdflatex, §3.5 + §3 cite 校验, arXiv W49 prep
7. arXiv submit (or Q1-2027 submission)

---

End of report.
