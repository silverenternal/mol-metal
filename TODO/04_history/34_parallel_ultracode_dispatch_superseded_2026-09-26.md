> **STATUS: SUPERSEDED 2026-09-26 by R17 close-out (commit 4561625) + 6 parallel tracks ship.**
>
> 6 条并行 ultracode 轨道 全部 SHIP:
> - UC-A Track A Flow Matching + EGNN ✅ (`TODO/completed/37_model_track_mr17.md` + `TODO/completed/01_lipman_fm_adapter.md`)
> - UC-B Track B Lambda + PLL 2026 ✅ (`TODO/completed/32_trackB_mlc_pll_wireup_done.md` + `TODO/completed/32_late_phase.md` + `TODO/completed/32_trackB_uc_b1_metal_atoms_30.md`)
> - UC-L R17 Lambda (L-R17-B1 Deflex cite / B2 84-rule MLE / B3 ASTSR) ✅ (`TODO/completed/35_l_r17_b3_astsr_swap_done.md` + 84-rule matrix in `pt_click_compat.py`)
> - UC-M R17 CFM (M-R17-A1 CLI / A2 100×3 spec / A3 24-paper cite) ✅ (`TODO/completed/37_model_track_mr17.md`)
> - UC-C R17 Coupling (C-R17-C1 PlatinAI 5K pretrain / C2 MCTS 64-d) ✅ (`TODO/completed/35_platinai5k_coupling_pretrain.md` + `TODO/completed/34_lambda_phase4_sweep.md` honest-negative)
> - UC-F R17 plumbing fixes (P1-P7) ✅ (`TODO/completed/33_r17_fixes_blackwell_loader_paths.md` + `TODO/completed/34_p7_zenodo_wrapper_done.md`)
>
> 当前: `TODO/completed/41_r17_final_closeout.md` (R17 close-out synthesis) + 30/30 decode probes + 14/14 evaluators + 20 metrics + 21 mechanisms + 5 skills activated.
> UC-F2 GPU 部分 (P1 YuelBond decode_ratio=0) 已 SHIP CPU-side shim fix;GPU-side dGPU 仍 env-blocked.
> 此文件保留作为 historical record, 不再决策。
>
> 归档: `TODO/04_history/34_parallel_ultracode_dispatch_superseded_2026-09-26.md`

# TODO-34 — Parallel Ultracode Dispatch (R17 + Track A/B + R17 fixes)

> **LEGACY** — pre-R17; see `TODO/INDEX.md` and `TODO/EXECUTION_ORDER.md` for current planning.
> Superseded by: `TODO/completed/41_r17_final_closeout.md` + `TODO/completed/34_p7_zenodo_wrapper_done.md` + `TODO/completed/34_lambda_phase4_sweep.md` (R17 close-out).

> **创建日期**: 2026-09-25
> **状态**: ⚙️ 待 dispatch — 6 个并行 ultracode 轨道已识别,均可 CPU-only 启动
> **关联**: `TODO/pending/31_r17_ultracode_dispatch.md` (R17 master) +
>           `TODO/pending/31_ultracode_trackA_flow_matching.md` (Track A) +
>           `TODO/pending/32_ultracode_trackB_lambda_calculus.md` (Track B, 部分完成) +
>           `TODO/pending/33_r17_fixes_blackwell_loader_paths.md` (R17 fixes)

## 0. TL;DR — 并行调度矩阵

下表是 **6 条独立的并行 ultracode 轨道**,每条可单独 dispatch 给一个 ultracode session。
**总 CPU 耗时 ~38h** (串行) → **~10-12h wall** (6 路并行),**GPU 阻塞项目 idle 等 GPU 起来**。

| 轨道 | 关联 plan | 工作流 (ultracode ID) | CPU wall | GPU 需求 | 阻塞? |
|---|---|---|---|---|---|
| **Track A** | `31_ultracode_trackA_flow_matching.md` | UC-A1..A4 (Flow Matching + EGNN wire-up) | ~7d | 仅 smoke 需 | 否 |
| **Track B-剩余** | `32_ultracode_trackB_lambda_calculus.md` | UC-B1..B3 (30-metal table + reward coupling + 5 rules library) | ~5d | 无 | 否 (env-blocked RDKit?) |
| **R17 Lambda** | `31_r17_ultracode_dispatch.md` §2 | UC-L1: L-R17-B1 (Deflex cite 1h) + UC-L2: L-R17-B2 (F2(a) 6h) + UC-L3: L-R17-B3 (ASTSR 10h) | ~17h | 无 | 否 |
| **R17 CFM** | `31_r17_ultracode_dispatch.md` §3 | UC-M1: M-R17-A1+A2+A3 (CLI prep + 24-paper cite) | ~7h | 部分 GPU | 否 |
| **R17 Coupling** | `31_r17_ultracode_dispatch.md` §4 | UC-C1: C-R17-C1 (PlatinAI pretrain 6h) + UC-C2: C-R17-C2 (MCTS 64-d 8h) | ~14h | 无 | 否 |
| **R17 fixes** | `33_r17_fixes_blackwell_loader_paths.md` | UC-F1: P2+P3+P4+P7 (plumbing fix 4-5h); UC-F2: P1 F1+F2+F4 (CFM BF16+LR+clamp, GPU-only) | ~5h | UC-F2 需 GPU | UC-F2 BLOCKED on dGPU |

**EtA R17 CPU 全部 ship**: 1-2 周 (calendar)
**EtA R17 GPU 部分** (UC-F2 + UC-M1 GPU retrain): 待 dGPU 起来后 ~12-24h GPU

---

## 1. 调度策略 (ultracode harness 怎么启)

### 1.1 并行启发式

每条轨道**内部**串行执行其子任务 (避免 race condition),**轨道间**完全独立。
- Track A 涉及 4 个文件 + `references/flow_matching` clone,**独立**。
- Track B 写 `lam_chem/`、`reward/` 子目录,**与 Track A 不共享文件**。
- R17 Lambda 写 `paper/sections/*.tex` + `molmetal/molmetal_lam/reactions/beta_reductions.py`,**不与 Track A 冲突**。
- R17 CFM 写 `molmetal/scripts/r17_*` + `TODO/pending/14_*.md`,**CLI spec 不冲突**。
- R17 Coupling 写 `molmetal/scripts/tmqm_coupling_*` + `molmetal/molmetal_lam/search_alg/mcts.py`,**可能与 Track B 的 PLL reward 共享 reward 目录**。
- R17 fixes (UC-F1) 改 `_reference_loader.py` + `pyproject.toml` + `env.sh`,**不与 Track A 冲突**。

### 1.2 资源预算

| 轨道 | CPU cores | 内存 | 备注 |
|---|---|---|---|
| Track A | 4 | 8GB | clone + tests + smoke |
| Track B 剩余 | 2 | 4GB | 只读 RDKit 5-cell 验证 |
| R17 Lambda (UC-L2) | 4-8 | 8GB | F2(a) SMARTS 84 rule synthesis |
| R17 CFM (UC-M1) | 2 | 4GB | bib + pdflatex |
| R17 Coupling (UC-C1) | 8 | 16GB | PlatinAI 226K embed,CPU-only |
| R17 Coupling (UC-C2) | 4 | 8GB | MCTS 30-cell pilot |
| R17 fixes (UC-F1) | 1 | 2GB | 单文件改 + 路径候选 |

**峰值**: ~25 cores / 48GB — 在 64-core / 128GB 主机上完全 fit。

### 1.3 Dispatch 命令模板

```bash
# 用 ultracode CLI 派发每条轨道 (假设 CLI 已存在;若不存在,用 6 个 tmux session 替代)
ultracode dispatch --track A --plan TODO/pending/31_ultracode_trackA_flow_matching.md
ultracode dispatch --track B-remaining --plan TODO/pending/32_ultracode_trackB_lambda_calculus.md
ultracode dispatch --track R17-lambda --plan TODO/pending/31_r17_ultracode_dispatch.md --only L-R17-B1,L-R17-B2,L-R17-B3
ultracode dispatch --track R17-cfm --plan TODO/pending/31_r17_ultracode_dispatch.md --only M-R17-A1,M-R17-A2,M-R17-A3
ultracode dispatch --track R17-coupling --plan TODO/pending/31_r17_ultracode_dispatch.md --only C-R17-C1,C-R17-C2
ultracode dispatch --track R17-fixes --plan TODO/pending/33_r17_fixes_blackwell_loader_paths.md --only P2,P3,P4,P7
```

---

## 2. 6 条轨道的详细范围

### 2.1 UC-A — Flow Matching + EGNN wire-up (`31_ultracode_trackA_flow_matching.md`)

**任务清单** (按 plan.md Day 1-7 顺序):

- **UC-A1 (Day 1, ~1h)**:
  - `git clone https://github.com/facebookresearch/flow_matching.git molmetal/references/flow_matching`
  - `git clone https://github.com/jule-c/flowr.git molmetal/references/FLOWR`
  - 核对 Lipman adapter 落点 `molmetal/adapters/flow_matching_lipman/`
  - **依赖**: 无
  - **产物**: 2 个 clone + adapter 落点确认

- **UC-A2 (Day 2-3, ~5h)**:
  - 包装 `flow_matching.loss.ConditionalFlowMatchingLoss` → `MolCFMConfig`
  - OT path scheduler `ConditionalOTScheduler` → `molmetal/adapters/flow_matching_lipman/path_schedulers.py`
  - Unit test: CrossDocked pocket-1 loss 收敛 100 step
  - **依赖**: UC-A1
  - **产物**: `path_schedulers.py` + 1 unit test

- **UC-A3 (Day 4-5, ~8h)**:
  - EGNN velocity field (`molmetal/models/egnn.py` → velocity_net wrapper)
  - 输入 (coords, atom_features, edge_index),输出 (v_coords, v_features)
  - SE(3) 等变性 test (100 随机旋转/平移)
  - **依赖**: UC-A1,UC-A2
  - **产物**: velocity field wrapper + SE(3) test

- **UC-A4 (Day 6-7, ~12h)**:
  - `MoleculeGenerator` Protocol 实现 → `FlowMatchingAdapter`
  - CrossDocked pocket-1 生成 100 分子,QuickVina 对接,记录 best IC50 / SA / QED
  - 与现有 `LipmanFM` 对比 (产率 / 3D RMSD / SE(3) 不变性)
  - **依赖**: UC-A1,A2,A3
  - **产物**: `FlowMatchingAdapter` + smoke json + 对比表

**DoD**:
1. `python -m molmetal.pipelines.fm_smoke --pocket 1 --n-samples 100` 全程跑通,耗时 < 30 min
2. 输出 `metrics/fm_smoke_pocket1.json`
3. 3 个 unit test 通过
4. 在 `TODO/completed/` 下新建 `31_trackA_fm_egnn_wireup_done.md`

---

### 2.2 UC-B-剩余 — Track B 剩余 (Lambda + PLL 2026)

**已完成 (per `32_trackB_mlc_pll_wireup_done.md`)**:
- ✅ PLL engine (`pll_signature.py`)
- ✅ β-reduction reducer (`reducer.py`)
- ✅ BioLM-Score adapter (`reward/biolm_score_adapter.py`)

**剩余** (按 plan Day 1-7 顺序):

- **UC-B1 (Day 1-2, ~3h)**:
  - 30-metal `METAL_ATOMS` 表扩展 (现仅 6: Pt_II/Ru_II/Zn_II/Ir_III/Cu_II/Au_III,需补 Sc/Ti/V/Cr/Mn/Fe/Co/Ni/Cu/Zn/Mo/Ru/Rh/Pd/Ag/Cd/W/Re/Os/Ir/Pt/Au/Hg 共 30)
  - 导出 `METAL_SYMBOLS` 公开 API
  - 单元测试:30 个金属的 arity = valence + lone pairs 与 RDKit 一致
  - **依赖**: 无 (PLL 已 ship)
  - **产物**: `atoms/combinators.py` METAL_ATOMS 扩展 + tests

- **UC-B2 (Day 2-3, ~3h)**:
  - Reward coupling (env-blocked 解决):
    - `molmetal/molmetal_lam/reward/lambda_reward_dataset.py` — PLL signature → reward
    - `molmetal/molmetal_lam/reward/coupling_adapter.py` — PLL → bond head
  - 测试在 5 个金属 + 7 个配体上验证 reward shape = (n_ligands, 64)
  - **依赖**: UC-B1
  - **产物**: 2 reward modules + tests

- **UC-B3 (Day 3-4, ~2h)**:
  - 5 NEW click rules library (per `wf_lambda_cfm_bridge_smoke.md`):
    - `SuzukiCoupling` (extend existing or new)
    - `AmideCoupling` (already stubbed?)
    - `ReductiveAmination`
    - `Esterification`
    - `BocDeprotection`
  - **依赖**: 无
  - **产物**: 5 new `ReactionRule` classes + tests

**DoD**:
1. METAL_ATOMS 30 entries + 4 unit tests green
2. Reward coupling shape (5 metals × 7 ligands, 64) verified
3. 5 click rules SMILES I/O test 10/10
4. 在 `TODO/completed/` 下新建 `33_trackB_remainder_done.md` 总结

---

### 2.3 UC-L — R17 Lambda 轨道 (3 sub-workflows)

#### UC-L1: L-R17-B1 Deflex Deep Cite (~1h, 1 CPU)

- 扩展 `paper/sections/03_5_deflex.tex`:
  - 抽取 Deflexpressor 的 lambda-calculus-AST 设计 → 与 LamVar/LamAbs/LamApp 对齐
  - Deflexformer 的多尺度 latent representation → 64-d bridge 设计讨论
- 增加 `paper/sections/06_limitations.tex` 一段 honest framing
- 增加 `paper/sections/07_future.tex` 一段 future work 引用
- **依赖**: 无 (yu2026deflex 已 add 进 refs.bib)
- **产物**: 3 个 tex 文件扩展 (~90 行) + 4-pass pdflatex 0 unresolved refs

#### UC-L2: L-R17-B2 F2(a) MetalLigandExchange SMARTS 升级 (~6h, 4-8 CPU)

- 12 金属 × 7 配体 = 84 base exchanges,加 4 NEW SMARTS (transmetalation / Cl-OH swap / ligand-shell-MMA / Pt/Pd transmetalation)
- `molmetal/molmetal_lam/reactions/beta_reductions.py` 加 `register_metal_ligand_exchange_rules(metal_set, ligand_set) -> dict`
- `molmetal/tests/test_f2a_metal_ligand_exchange.py` 新增 ~80 parametrize
- **依赖**: 无
- **产物**: 84+ rules + 80 tests + report
- **Verify**: `uv run python -c "from molmetal.molmetal_lam.reactions.beta_reductions import register_metal_ligand_exchange_rules; r = register_metal_ligand_exchange_rules(); print(len(r))"` → ≥ 84

#### UC-L3: L-R17-B3 F5 → Deflexpressor ASTSR (~10h, 1 CPU)

- 写 `molmetal/molmetal_lam/reward/lambda_astsr.py` (~200 LOC)
  - python-LISP AST symbolic regression
  - 操作: abstraction + application + Church-encoding
  - 训练在 PlatinAI 226K + tmQM 108K (如 mount),否则 32-cell pilot
- 集成到 `coupling_adapter.py` 替换 F5
- **依赖**: 无 (但建议 L-R17-B2 先 ship,以免 F2(a) SMARTS 干扰 ASTSR)
- **产物**: `lambda_astsr.py` + tests + `wf_deflex_astsr/{phase_1,final}.md`
- **Verify**: ASTSR R² ≥ 0.62 on 30-cell hold-out

**UC-L 顺序建议**:
- UC-L1 + UC-L2 并行 (互不干扰)
- UC-L3 在 UC-L2 之后 (因为 F2(a) 84 rules 是 ASTSR 训练数据的"骨架候选")

---

### 2.4 UC-M — R17 CFM 轨道 (3 sub-workflows)

#### UC-M1: M-R17-A1 R17 GPU Retrain Plan + 监控 scaffold (~2h, 1 CPU)

- 读 TODO-24 + TODO-26 + `wf_r16_yuelbond_10000/final.md`
- 写 `molmetal/scripts/r17_gpu_retrain.py`:
  - 自动检测 dGPU 状态 (`rocm-smi`)
  - 启动 10000-step YuelBond retrain (h=128, l=3, batch=4)
  - 实时监控 decode_ratio (target ≥ 0.05 at step ≥ 500)
  - 自动 checkpoint + resume on GPU hang
  - 输出 `metrics/by_metric/cfm_decode_ratio_r17.json`
- 写 `molmetal/scripts/wf_r17_gpu_retrain/template.md`
- **依赖**: 无
- **产物**: `r17_gpu_retrain.py` + template

#### UC-M2: M-R17-A2 Round-13 100×3 paper-grade spec (~2h, 1 CPU)

- 扩展 `molmetal/scripts/r4_lambda_only_run.py` 加 `--r17-paper-grade` 标志:
  - 自动加载 100 pocket CrossDocked2020 test split manifest
  - 3 seeds × 100 pockets × n_sim=1000 (~6h GPU / ~50min CPU Lambda-only)
  - 输出对齐到 `molmetal/reports/wf_paper_grade_300cells/`
- 测试 smoke mode (10 pockets × 1 seed) CPU-only (~5 min)
- 把 spec 加到 `TODO/pending/14_full_100pocket_paper_r13.md` 顶部
- **依赖**: 无
- **产物**: `--r17-paper-grade` flag + spec section + smoke test

#### UC-M3: M-R17-A3 24-paper Frontier cite (~3h, 1 CPU)

- 读 `molmetal/reports/wf_cfm_frontier_research/research_phase1{abcd}.md`
- 提取 24 篇 paper 的 citation info (author + title + venue + year + arXiv)
- 加到 `paper/refs.bib` 末尾 (一次性 append-only)
- 在 `paper/sections/03_method.tex` 相应地方加 `\citep{...}` (E2Former, mHC, EquiformerV2 等)
- **依赖**: 无
- **产物**: refs.bib +24 entries + §3 5 处 `\citep` + 76→79 页

**UC-M 顺序建议**:
- UC-M1 + UC-M2 + UC-M3 完全并行 (互不干扰)

---

### 2.5 UC-C — R17 Coupling 轨道 (2 sub-workflows)

#### UC-C1: C-R17-C1 Lambda × CFM joint prior training (~6h, 8 CPU)

- 写 `molmetal/scripts/tmqm_coupling_pretraining_cpu.py`:
  - 加载 PlatinAI 226K SMILES (或已知 sub-sample 5K)
  - 训练 32→64 dim coupling adapter (CPU-only)
  - 输出 `models/coupling_adapter_platinai32k.npz`
- PlatinAI test set (scaffold-split 5K) 验证: embed_pocket shape (5K, 64)
- 64-d numpy bridge 升级 → 替换 F5 learned shaping channel
- **依赖**: 无 (PlatinAI 已知加载方式见 `wf_platinai_load/`)
- **产物**: `tmqm_coupling_pretraining_cpu.py` + `models/coupling_adapter_platinai32k.npz` + tests

#### UC-C2: C-R17-C2 MCTS 64-d coupling (~8h, 8 CPU)

- 扩展 `molmetal/molmetal_lam/search_alg/mcts.py` `_rollout()` 用 64-d pocket embedding cosine sim 作 leaf score modulation
- 测试 30-cell pilot: leaf score 在 `1.0 - 0.5*pocket_sim_penalty` 范围
- 集成到 `r4_lambda_only_run.py` (默认 `MOLMETAL_MCTS_COUPLED=1` env-var 启用)
- **依赖**: UC-C1 (需要 64-d numpy bridge)
- **产物**: `mcts.py:_rollout_coupled` + `test_mcts_coupled.py` + `wf_lambda_mcts_coupled/final.md`

**UC-C 顺序建议**:
- UC-C1 先 → UC-C2 后 (耦合依赖)

---

### 2.6 UC-F — R17 plumbing fixes 轨道 (5 sub-fixes)

#### UC-F1 (CPU-only, ~4-5h): P2 + P3 + P4 + P7

| Fix | 任务 | 时间 | 依赖 |
|---|---|---|---|
| **P2** | `flow_matching` loader auto-detect + shim re-apply | 1-2h | 无 |
| **P3** | SyntheMol + BioLM-Score path candidates | 30min | 无 |
| **P4** | `pyproject.toml` 加 `[tool.uv]` managed=false + env.sh tighten | 30min | 无 |
| **P7** | `molmetal/scripts/zenodo_fetch.py` wrapper script | 1-2h | 无 |

**依赖**: 全部独立可并行,但建议串行避免文件冲突
**产物**: 4 files + `mol-metal/.venv` 不再出现 + Zenodo wrapper

#### UC-F2 (GPU-only, BLOCKED): P1 YuelBond decode_ratio=0 fix

| Fix | 任务 | 时间 | 依赖 |
|---|---|---|---|
| **P1 F1** | `MOLMETAL_CFM_AMP_DTYPE=float32` (1 LOC) | 30min | dGPU 起来 |
| **P1 F2** | LR warmup `scheduler.warmup_steps=200` | 1h | F1 |
| **P1 F4** | clamp logits before softmax | 15min | F2 |
| **P1 F3** | PCGrad on conflicting gradients (R16 计划内) | 3h | F4 |

**依赖**: dGPU cold-power-cycle recovery (out-of-scope)
**EtA**: GPU 起来后 ~5h
**产物**: 4 fixes + 3-seed smoke at step 1000/2000 decode_ratio > 0

---

## 3. Verify gates (per track)

| Track | Verify |
|---|---|
| UC-A | `python -m molmetal.pipelines.fm_smoke --pocket 1 --n-samples 100` < 30 min;3 unit tests pass |
| UC-B 剩余 | `pytest molmetal/tests/test_combinators.py molmetal/tests/test_lambda_reward_coupling.py` 30+ pass |
| UC-L | 4-pass pdflatex 0 unresolved refs;F2(a) print(len(r)) ≥ 84;ASTSR R² ≥ 0.62 |
| UC-M | CPU smoke < 30 min;refs.bib 90 → 114;76 → 79 pages |
| UC-C | PlatinAI 5K test embed RMSE ≤ 0.05;30-cell leaf pairwise_diff 0.0001 < x < 0.05 |
| UC-F | 43+ pytest pass (P2);`mol-metal/.venv` 不再现;GPU 起来后 3-seed decode_ratio > 0 |

---

## 4. Dispatch 后端选择

| 方案 | 适用场景 |
|---|---|
| **ultracode CLI** (推荐) | 若 `ultracode dispatch --track X` 命令已 ship;否则 |
| **6 tmux sessions** | 每条轨道一个 session,各自 Claude agent session |
| **6 git worktrees** + 子进程 | 每条轨道一个 worktree,各自运行 scripts |
| **单 ultracode 串行** | 资源不够并行,fallback |

**当前可用** (per `ListAgents`): 1 个 peer session `wave-127-roi-todo-refactor`,但其任务无关。
**建议**: 在 host 上启 6 个 Claude Code CLI 会话,每条轨道一个工作目录(worktree 隔离)。

---

## 5. 与现有 R17 master 调度的差异

`TODO/pending/31_r17_ultracode_dispatch.md` 是 R17 master plan (定义 8 sub-workflows L-R17-B1..B3 + M-R17-A1..A3 + C-R17-C1..C2),本文 `34_parallel_ultracode_dispatch.md` 是**调度层 plan**:
- 把 8 个 sub-workflows 合并成 6 条并行轨道 (Track A/B/C-L/M/C/F)
- 加入 Track A 和 Track B-剩余 (原 R17 master 没列出)
- 加入 P2/P3/P4/P7 plumbing fix (原 R17 master 31_r17_ultracode_dispatch.md §5 only 列了 R17 fixes 但没说派发)
- 给每条轨道 CPU/GPU/内存资源预算
- 给每条轨道 DoD + verify gate

---

## 6. Definition of Done (本 dispatch 文件)

1. ✅ 6 条并行轨道识别清楚 (UC-A / UC-B 剩余 / UC-L / UC-M / UC-C / UC-F)
2. ✅ 每条轨道 CPU/GPU/时间预算清楚
3. ✅ 每条轨道 DoD + verify gate 清楚
4. ✅ Dispatch 后端策略明确 (ultracode CLI / tmux / worktree)
5. ✅ 与现有 R17 master plan 的差异说明清楚

---

## 7. References

- `TODO/pending/31_r17_ultracode_dispatch.md` — R17 master plan (8 sub-workflows)
- `TODO/pending/31_ultracode_trackA_flow_matching.md` — Track A (Flow Matching)
- `TODO/pending/32_ultracode_trackB_lambda_calculus.md` — Track B (Lambda + PLL)
- `TODO/pending/33_r17_fixes_blackwell_loader_paths.md` — R17 Blackwell fixes (P1-P7)
- `TODO/pending/14_full_100pocket_paper_r13.md` — Round-13 100×3 (target of UC-M2 spec)
- `TODO/completed/32_trackB_mlc_pll_wireup_done.md` — Track B 已完成部分

---

End of report.
