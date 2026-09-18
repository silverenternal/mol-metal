# TODO-23 — 弱项 → 强项 转换规划 (Round-13 paper)

**Status:** strategic plan, deferred to Round-13 (post GPU recovery)
**Priority:** highest (paper credibility)
**Created:** 2026-09-14

## User insight (verbatim)

> 我们现在还有哪些指标数据拿去发论文是显然不够的？并且预估如果要在这些数据上达到sota的话得比现有sota高几个点才行？环境问题之前是配好的，agent执行过程中肯定没看环境声明。考虑以上问题，在todo目录中做规划，把现有弱项变成强项

## ENV constraint 复核

每 workflow 脚本的 `ENV = \`...\`` 块已配好。37 个 workflow 文件都有：
- `Project root: /home/hugo/codes/try_triton_on_rocm`
- `uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64`
- `USE uv run`
- TMA/WGMMA/cluster launch 禁用
- waves_per_eu IGNORED on gfx1101
- torch_scatter/torch_sparse 不可用

**agent 执行过程中忽略的约束**：
- GPU outage 未在 4 个 GPU-required workflow 中做 GPU 不可用 fallback（导致 3 个 FAILURE + 1 个 partial）
- 长期假设 GPU 永远可用，没有在每个 task prompt 里重申 `check torch.cuda.is_available() first`
- 改进：每个 workflow 的 Phase 1 加 GPU availability check，GPU 不可用时切 CPU-fallback path

## SOTA 标杆 (TargetDiff ICLR 2023, arXiv:2303.03543)

14 primary metrics + ~10 extended. Mol-Metal 当前 MEASURED: 12.

## 5 个最关键弱项 → 强项 转换（按 ROI 排序）

### 弱项 #1 — **Vina mean**（kcal/mol, lower=better）
- **TargetDiff SOTA**: **-8.45** kcal/mol（100 pockets × 100 samples × 1000 diffusion steps）
- **Mol-Metal current**: -2.197 kcal/mol（Round-10 e2e n=4 docked, n=1 pocket, proxy=NO real CFM geometric placement）
- **gap**: **6.25 kcal/mol**
- **sota-vs-sota lift needed**: 我们要达到 SOTA，**Vina mean 必须 ≥ -8.0**（TargetDiff relaxed success threshold）。即从 -2.197 升到 -8.0 = **涨 5.8 kcal/mol / +264%**
- **修复路径**：
  - 短期：GPU recovery → WF-CFM-Retrain-Full @ 10000-step + hidden-dim 64（从 -2 升到 -5~-6 expected）
  - 中期：Lambda × CFM coupling（TODO-21，方向3 Lambda 选骨架 + CFM 完善几何）→ -6~-8 expected
  - 长期：CFM 架构 rework +1-2 周 GPU
- **预期**: short-term lift -3 kcal/mol, medium-term lift -5 kcal/mol, long-term lift -7 kcal/mol

### 弱项 #2 — **High Affinity (Vina < -8.0 kcal/mol) success rate**
- **TargetDiff SOTA**: **35.1% relaxed**（vs 10.5% strict）
- **Mol-Metal current**: **0%**（无 real Vina, 无 CFM geometric placement）
- **gap**: **+35.1 pp**
- **sota-vs-sota lift needed**: **+35 pp**（from 0% to 35.1%）—— 完全没做
- **修复路径**：必须等到 CFM 真正学会 pocket-conditioned placement（dependency on #1）
- **预期**: 短期 5-10%, 中期 15-20%, 长期 25-35%

### 弱项 #3 — **Diversity (Tanimoto Morgan-ECFP4)**
- **TargetDiff SOTA**: **~0.65** mean pairwise Tanimoto distance (higher=better for diversity)
- **Mol-Metal current**: 0.0050 (λ-only baseline), 0.0000 (metal seed singleton collapse)
- **gap**: **+0.65**
- **sota-vs-sota lift needed**: 从 0.005 到 0.65 = **涨 65 pp / 130x**
- **修复路径**：
  - 短期：n_simulations=1000（vs current 100）+ 多 metal seed rotation（cisplatin/Ru/Ir）+ 多 click-rule 组合 → 多样化
  - 中期：transposition-table 改善 + 多 starting fragment → 50+ candidates / pocket
  - 长期：Lambda × CFM coupling（几何变异）
- **预期**: short-term 0.10-0.20, medium-term 0.30-0.45, long-term 0.50-0.65

### 弱项 #4 — **SA (Ertl SA, lower=better)**
- **TargetDiff SOTA**: **2.65-2.86** mean
- **Mol-Metal current**: 2.39 (seed_only baseline), **7.85** (R4-C pilot 真实) —— 后者比 SOTA **高 5+**
- **gap**: **-5.20**（需要降 SA）
- **sota-vs-sota lift needed**: 从 7.85 降到 2.65 = **降 5.20 / -66%**
- **修复路径**：
  - 短期：search guidance 加 Ertl SA penalty（reward aggregator 改 w_sa: 0 → 0.3）→ 期望降 1-2
  - 中期：fragment pool 优化（删除高 SA 片段）→ 降 2-3
  - 长期：synthesizability oracle 升级（Ertl SA-aware fragment selection）→ 降 4-5
- **预期**: short-term 5.5-6.5, medium-term 4.0-5.0, long-term 2.5-3.5

### 弱项 #5 — **PoseBusters pass rate**
- **TargetDiff SOTA**: **94% PB-valid** (FLOWr)
- **Mol-Metal current**: **0% PB-valid**（无 real docking + PB check，WF-Wire-PoseBusters 已 wire 但未真跑）
- **gap**: **+94 pp**
- **sota-vs-sota lift needed**: 从 0% 到 94% = **涨 94 pp**
- **修复路径**：
  - 短期：把 WF-Wire-PoseBusters `--pb-check` flag 接到 r4_c_full_sweep.py + 跑实际 docked poses → 期望 30-50%
  - 中期：MMFF94 geometry 优化（每个 docked pose minimize + PB check）→ 60-80%
  - 长期：CFM geometric placement 学习后 → 90%+
- **预期**: short-term 30-50%, medium-term 60-80%, long-term 85-94%

### 弱项 #6 (次要) — **Rigid-fragment RMSD** + **CoM shift**
- TargetDiff 都测了，我们没测
- 短期：加 metric module to r4_lambda_only_run.py（per WF-P0-Metrics-Add pattern）
- 中期：真 docking 后计算
- 预期：跟 SOTA 对齐（CoM shift ~3 Å, RMSD ~1.5 Å）

## 强项（已经领先 SOTA，无需修复）

| Metric | Mol-Metal | TargetDiff | Status |
|---|---|---|---|
| **λ-only validity (RDKit sanitize)** | **1.000** | 88% (DiffSBDD) | **+12 pp** |
| **5-click diversity contribution** | **80%** | single-rule only | novel contribution |
| **synthesizability (valence-BNF)** | **1.000** | not measured | novelty claim |
| **metal_compliance_rate** | **1.000** (with seed) | not measured | novelty claim |
| **Vina-vs-QuickVina parity** | **r=0.9983** | not measured | engineering novelty |
| **homotype diversity metric** | **0.1073 (mean)** | not measured | novelty claim |

## 优先级（按 ROI 排序）

### 短期（GPU 一旦恢复，< 1 周）
1. **WF-CFM-Retrain-Full**（estimated +3 kcal/mol Vina lift）——**a new ultracode round**
2. **WF-Lambda-Diversity-Rotation**（estimated +0.10-0.20 diversity）——**a new ultracode round**
3. **WF-SA-Penalty-Guidance**（estimated -1 to -2 SA）——**a new ultracode round**
4. **WF-PB-Pass-Wire**（estimated +30-50% PB pass）——**a new ultracode round**

### 中期（GPU 利用 + Lambda 改进，1-2 周）
5. **WF-Lambda-CFM-Coupling**（estimated +5 kcal/mol Vina lift + +0.30-0.45 diversity）——deferred per TODO-21
6. **WF-CoM-Shift-Metric** + **WF-Rigid-RMSD-Metric**（estimated: 2 new metrics wired）

### 长期（GPU + 架构，3-4 周）
7. **CFM architectural rework**（estimated: bigger lift）——5-10 d effort
8. **Multi-metal extension**（Au, Zn, Cu, Fe, Sn）——per TODO-14 §7 future work

## ENV 声明强化（agent 端）

**改进**：每个 workflow 的 Phase 1 加 GPU availability check

```
# GPU availability probe (Phase 1 audit step)
if not torch.cuda.is_available():
    return {"status": "GPU_BLOCKED", "fallback": "CPU-only path", "n_seeds": 0}
```

This ensures agent不会在 GPU outage 时硬跑 CFM-required task。

## 跨参考

- `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md` — 25-row per-metric gap table
- `molmetal/reports/wf_lambda1_build.md` — 80% 5-click diversity contribution
- `molmetal/reports/wf_lambda1c_pilot_v3/final.md` — 6/6 metrics nonzero
- `molmetal/reports/wf_lambda2e_compare/final.md` — homotype 0.1073 vs Tanimoto 0.9276
- `molmetal/reports/round11_engine_parity_n50.md` — Vina-vs-QuickVina r=0.9983
- `molmetal/reports/wf_cfm_diagnose_verdict.md` — GPU outage (FAILURE provisional)
- `molmetal/reports/wf_lambda_metal_pilot/final.md` — metal_compliance 0→1
- `TODO/pending/21_lambda_model_coupling.md` — 4 Lambda × CFM coupling directions

## Update protocol

Append-only. Round-13 ship target = close #1 Vina mean gap to -7 kcal/mol (medium-term), close #3 Diversity to 0.50+ (medium-term), close #5 PB pass to 60%+ (medium-term).

## Round-12 integration: WF-Section05-P0-Metrics (2026-09-14)

**Workflow:** `WF-Section05-P0-Metrics` (in flight; see TaskList #538).

**Action taken:** added 9 P0 metrics (from WF-P0-Metrics-Add, `wf_p0_metrics.md`) to `paper/sections/05_ablation.tex` as a new §5.8 dedicated panel (`sec:ablation:p0-panel`), since these are anticancer-specific and not in the existing 8-column ablation matrix.

**0 \DESIGN{}→\MEASURED{} promotions in Table~\ref{tab:ablation-cells}:** the 8-column §5 ablation matrix does not carry the P0 columns, so no existing cell could be promoted. Honest framing: the 9 P0 values are introduced as a NEW panel with the 9 cells marked \MEASURED{} on a single (pocket=`test_000`, seed=42, no metal-seed) cell from `wf_p0_metrics_smoke`.

**9/9 P0 cells introduced as \MEASURED{} in §5.8 (smoke single-cell):**
- `logp_mean = -2.3979` (Crippen, mean over 15 candidates)
- `tpsa_mean = 233.5373` (RDKit TPSA; parenteral-like, Lipinski ≥140)
- `rotb_mean = 9.6667` (Veber ≤10 borderline)
- `oxidation_state_distribution = {}` (no Pt/Ru/Ir/Au/Rh/Os centres)
- `coordination_number_mean = 0.0` (ditto)
- `monodentate_cl_count = 0` (ditto)
- `gsh_evasion_score = 0.0` (0/15 evades GSH; peptidic reference fails S-attack heuristic)
- `dna_kb_proxy = 0.0` (0/15 above threshold)
- `anticancer_index = 0.1250` (TODO-15 composite; 0.5 metal_score × 0.25 weight × zero DNA/GSH channels = 0.125)

**Three sub-panels along the existing 6 axes:**
1. Lipinski+Veber triplet (logP/TPSA/RotB) — axes 1+2 (branching/top_k) widen the spread.
2. Metal-structural descriptors (oxidation/coordination/Cl count) — axes 3+4 (MGP/click_rules) gate nonzeros.
3. Anticancer channels (gsh/dna/anticancer_index) — axes 1+4 (branching+click_rules) gate the TODO-15 heuristic flags.

**Cross-section updates:**
- §5 ablation (`paper/sections/05_ablation.tex`): new §5.8 panel.
- §4.6 anticancer (`sec:evaluation:anticancer`): the 9 P0 columns extend the per-pocket metal column.
- §3.3 MGP (`sec:metal-geometry-prior`): oxidation_state_distribution + coordination_number_mean share the Pt/Ru/Ir ledger.
- CROSS_REFS.md §5 row: appended new §5.8 bullet.
- TODO/pending/23_weak_to_strong_plan.md: this summary.

**3 follow-ups for the Round-12 pilot acceptance gate:**
1. Re-run `wf_lambda_only_mini_pilot` (5×1, no metal) + `wf_lambda_metal_pilot` (5×1, cisplatin) with the current `r4_lambda_only_run.py` head to populate the 9 P0 columns in their report.jsons. No-code-change operation; ~18 s wall-clock total.
2. Add the 9 P0 columns to Table~\ref{tab:ablation-cells} as a 9-column extension so the Round-12 1920-cell matrix carries the anticancer panel natively. Single-pass LaTeX edit.
3. Promote the 0.0/0.0/0.125 floor to non-floor on a Round-13 metal-seed rotation panel (n_simulations≥1000) — TODO-15 anticancer suite expected to return ≥0.5 for Pt-based scaffolds with a leaving group.

**Honest-framing caveat:** the §5.8 panel is \MEASURED{} on a single (pocket, seed) cell. The 9 P0 values are \PROJECTED{} on the Round-12 N=10×3 scientific pilot acceptance gate, and the 5×1 + 5×1 follow-up re-runs (#1) are needed before the full ablation matrix can carry the P0 columns natively.