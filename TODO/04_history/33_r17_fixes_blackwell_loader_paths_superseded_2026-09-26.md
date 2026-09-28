> **STATUS: SUPERSEDED 2026-09-26 by R17 close-out (commit 4561625) + Blackwell fixes done.**
>
> 7 个 blocking 问题 全部 SHIP:
> - P1 YuelBond decode_ratio=0 (Bug A: shim ModelWrapper unwrap + Bug B: OOV atoms 12→20 element vocab) ✅
> - P2 flow_matching loader auto-detect + shim re-apply ✅
> - P3 SyntheMol + BioLM-Score path fallback candidates ✅
> - P4 stale .venv (`pyproject.toml [tool.uv] managed=false` + env.sh tighten) ✅
> - P5 symlink 兼容 (via P2 fix) ✅
> - P6 `final.json` `delta_first_to_last` 键 ✅
> - P7 Zenodo DNS-blocked fallback wrapper ✅ (`TODO/completed/34_p7_zenodo_wrapper_done.md`)
>
> 当前: `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` (P1+P2 shim fixes) + `molmetal/reports/wf_todo33_p2_shim_fix/final.md` + 3-seed verification `logs/yuelbond_p1_shim_fix/seed_{0,1,2}/` (30/30 = 1.000 decode_ratio).
> atom vocab 12 → 20 elements 覆盖 454/7500 OOV atoms.
> 此文件保留作为 historical record, 不再决策。
>
> 归档: `TODO/04_history/33_r17_fixes_blackwell_loader_paths_superseded_2026-09-26.md`

# TODO-33 — R17 修复：Blackwell decode_ratio=0 + 加载器/路径回滚

> **LEGACY** — pre-R17; see `TODO/INDEX.md` and `TODO/EXECUTION_ORDER.md` for current planning.
> Superseded by: `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` (R17 close-out — TODO-33 P1+P2 shim fixes shipped).

> 创建日期: 2026-09-25
> 状态: 待派发（CPU-only 部分可立即 dispatch）
> 关联目录: `molmetal/adapters/flow_matching_lipman/`, `TODO/pending/24_cfm_architecture_redo_plan.md`
> 前置: TODO-31（R17 调度）已 SHIP；TODO-32（Lambda+PLL）Phase 0 已 SHIP

## 0. TL;DR

R17 第一轮 ultracode + 设备迁移（NVIDIA RTX PRO 6000 Blackwell sm_120）发现了 7 个 blocking 问题：
- **P1 — YuelBond decode_ratio=0.000** on Blackwell（vs R16 ROCm 的 1.000）。`atom_loss` 从 step 1 起就是 `Inf`；`cfm_loss` 17.7→252330 发散。
- **P2 — flow_matching 加载器路径回滚**。`_reference_loader.py` 和 `_reference/__init__.py` 被外部 process 还原成原版（只接受 `ref / "flow_matching"` 不接受 clone root 本身），导致 shim 失效。
- **P3 — SyntheMol / BioLM-Score 路径回滚**。adapter 中 `_SYNTHEMOL_PATH` 和 `_DEFAULT_BIOLM_REPO_ROOT` 还原成单一路径（`try_triton_on_rocm/...`），丢失了 R17 兼容路径。
- **P4 — `mol-metal/.venv` stale venv 反复出现**。每次 `uv run` 不带 `UV_PROJECT_ENVIRONMENT` 都会在 `mol-metal/` 下新建 venv，覆盖 13MB site-packages 空间。
- **P5 — `mol-metal/molmetal/references/flow_matching` 仍是符号链接**。修复 P2 后需要把它转成真目录或保持 symlink 同时 loader 兼容。
- **P6 — `final.json` summary 缺 `delta_first_to_last` 键**（已修复——见 §6）。
- **P7 — Zenodo DNS-blocked 的脆弱 fallback**（`--resolve zenodo.org:443:159.100.250.205`）。当前能用但不优雅。

## 1. P1 — YuelBond decode_ratio=0 on Blackwell

### 症状

`logs/yuelbond_multiseed/seed_0.log`（seed=0，2000 steps，h=128/l=3/batch=8/BF16/Blackwell）：
```
step    1/2000  loss=inf  cfm=17.7399  atom=inf  bond=inf
step   50/2000  loss=inf  cfm=11.8844  atom=inf  bond=inf
step  150/2000  loss=inf  cfm=8.0158   atom=inf  bond=inf    <-- cfm min
step  500/2000  loss=inf  cfm=4589.6   atom=inf  bond=inf
 >> decode_smoke step=500: decode_ratio=0.000  n_decoded=0/16  mean_atoms=0.0
step 1000/2000  loss=inf  cfm=789.9    atom=inf  bond=inf
 >> decode_smoke step=1000: decode_ratio=0.000  n_decoded=0/16
step 2000/2000  loss=inf  cfm=252330.1  atom=inf  bond=inf
done in 104.3s (2000 param steps, 0.05s/step)
```

### 已知对比（R16 AMD ROCm RX 7800 XT，per `molmetal/reports/wf_r16_yuelbond_2000_probe/final.md`）
- 2000-probe decode_ratio=1.000 at step 500/1000/1500/2000
- 0 disconnected, 0/192 floor lifted to 192/192
- cfm_loss min=6.65, first=7.37 (无 Inf)

### Root cause hypotheses（按可能性排序）

1. **BF16 numerical instability on Blackwell** — atom_logits overflow → atom_loss=Inf from step 1。R16 报告记录了 R10 同样的 "atom_loss Inf spikes"，但 AMD ROCm 上 BF16 行为与 NVIDIA 不同。**最可能。**
2. **Adam LR 太大** — `lr=1e-4` 在 Blackwell + BF16 下可能引起 early divergence。R16 没用 LR warmup。
3. **CFG scale / atom_vocab_mask 行为差异** — Blackwell BF16 GEMM 内核对负 logits 的处理可能与 ROCm 不同。
4. **PyTorch 2.14 vs 2.14+rocm7.2 内核差异** — autocast 路径可能不同。

### 修复方案（4 个候选，按预期效果排序）

| Fix | Effort | Expected Δ decode_ratio | 风险 |
|---|---|---|---|
| **F1: 切换 AMP dtype 到 FP32** (`MOLMETAL_CFM_AMP_DTYPE=float32`) | 1 LOC | +0.0-0.5 | 显存占用 2x（Blackwell 96GB 不怕） |
| **F2: 加 LR warmup** (`scheduler.warmup_steps=200`) | 5 LOC + test | +0.2-0.6 | 训练变慢 ~10% |
| **F3: PCGrad on conflicting gradients** (TODO-24 P1.3) | 80 LOC | +0.3-0.8 (R16 已规划) | 复杂、需要新 grad hook |
| **F4: clamp logits before softmax** | 2 LOC | +0.1-0.3 | 可能限制模型容量 |

### 推荐顺序
1. F1 立即试（30 min）
2. F2 + F1 (1 h)
3. F4 (15 min)
4. F3（PCGrad，最后做，因为是 R16 plan 内）

## 2. P2 — flow_matching 加载器路径回滚

### 症状

`molmetal/adapters/flow_matching_lipman/_reference_loader.py` 的 `configure_reference()` 当前只接受 `ref / "flow_matching"` 路径（parent dir），不接受 clone root 本身。但 `LipmanFlowMatchingAdapter.__init__` 的默认参数 `ref_repo_path="molmetal/references/flow_matching"` 是 clone root，导致 `root = ref / "flow_matching"` 指向 `molmetal/references/flow_matching/flow_matching`（不存在）。

`molmetal/adapters/flow_matching_lipman/_reference/__init__.py` 当前只设置 `__path__ = [str(configure_reference())]`，丢失了本地 shim 路径。

### 修复方案

**F1**: 在 `configure_reference()` 中加 auto-detect：
```python
if (ref / "__init__.py").is_file() and ref.name == "flow_matching":
    root = ref
else:
    root = ref / "flow_matching"
```

**F2**: 在 `_reference/__init__.py` 的 `__path__` 中加入本地 shim 目录，并 pre-load upstream clone 为 `_reference.upstream_flow_matching`。

**F3**: 重写 4 个 shim 文件（已写好但因 P2 回滚失效）：
- `_reference/path/__init__.py` — `AffineProbPath` facade + `_PathSample` dataclass
- `_reference/path/scheduler/__init__.py` — `CondOTScheduler` facade
- `_reference/solver/__init__.py` — `ODESolver` facade
- `_reference/utils/__init__.py` — `ModelWrapper` facade

修复后 3 个 shim 把 mol-metal 自家 fork 的扁平 API（`loss.py`, `sampler.py`, `interpolation.py`, `sampler_config.py`）重新映射成 Lipman 2023 老 API（`path.AffineProbPath`, `path/scheduler.CondOTScheduler`, `solver.ODESolver`, `utils.ModelWrapper`）。

### Effort: 1-2 h（已写好，重做即可）

## 3. P3 — SyntheMol / BioLM-Score 路径回滚

### 症状

- `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` 的 `_SYNTHEMOL_PATH` 还原成单一路径：`/home/hugo/codes/try_triton_on_rocm/molmetal/references/SyntheMol/synthemol`（旧 AMD 工作区路径）。
- `molmetal/molmetal_lam/reward/biolm_score_adapter.py` 的 `_DEFAULT_BIOLM_REPO_ROOT` 还原成相对路径 `"molmetal/references/BioLM-Score"`。

### 修复方案

**F1**: SyntheMol adapter 加 fallback candidates：
```python
_SYNTHEMOL_PATH_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "references" / "SyntheMol" / "synthemol",
    Path("/home/hugo/codes/mol-metal/references/SyntheMol/synthemol"),  # canonical after R17
    Path("/home/hugo/codes/try_triton_on_rocm/molmetal/references/SyntheMol/synthemol"),  # legacy
)
```

**F2**: BioLM-Score adapter 类似，加绝对路径候选。

### Effort: 30 min

## 4. P4 — stale `.venv` 反复出现

### 症状

`uv run --no-sync` 在没设 `UV_PROJECT_ENVIRONMENT` 时，会在 `mol-metal/` 下新建 venv，覆盖约 13 MB space。root cause：`uv run` 默认查找 `pyproject.toml` 然后建 venv，不会自动用外部 env。

### 修复方案

**F1**: 更新 `env.sh` 加 strict 模式：在脚本末尾 `unset VIRTUAL_ENV` 防止继承，强制通过 `UV_PROJECT_ENVIRONMENT` 走共享 env。已部分实现。

**F2**: 在 `pyproject.toml` 加 `[tool.uv]` 配置，让 `uv` 默认识别共享 env：
```toml
[tool.uv]
managed = false  # 不强制 manage，避免建 venv
```

**F3**: 添加 pre-commit hook 检测 `mol-metal/.venv` 存在时报警。

### Effort: 30 min

## 5. P5 — `mol-metal/molmetal/references/flow_matching` 符号链接

### 症状

`mol-metal/molmetal/references/flow_matching` 是符号链接 → `/home/hugo/codes/mol-metal/flow_matching/`。`.resolve()` 后变成后者，导致 `ref / "flow_matching"` 路径错误。

### 修复方案

**F1**: 删除符号链接，把 `flow_matching/` 实际目录移到 `mol-metal/molmetal/references/flow_matching/`。但这会复制 ~50KB 源码 + 100+ submodules — 不必要。

**F2**（推荐）: 保留符号链接，P2 中 loader 的 auto-detect 已兼容（clone root vs parent）。

### Effort: 0（已通过 P2 修复解决）

## 6. P6 — `final.json` summary 缺 `delta_first_to_last` 键

### 症状

`metallo_drug_smoke_retrain.py` 的 summary formatter 用 `cfm_summary["delta_first_to_last"]` 但 `loss_curves_summary` 的 `cfm` 字典（当 loss 是 Inf 时）只有 `n/first/last/min/max`，没有 `delta_first_to_last`。

### 修复方案

**F1**（已修复）: 在 `_summarize` 函数中 always compute `delta_first_to_last` = `last - first`（即使 first/last 是 Inf）。

### Effort: 5 min（已修）

## 7. P7 — Zenodo DNS-blocked 的脆弱 fallback

### 症状

`zenodo.org` 在本机 DNS 解析到 `0.0.0.0`。当前用 `curl --resolve zenodo.org:443:159.100.250.205` 强制走 Cloudflare anycast。但：
- IP 可能变化
- 依赖 Cloudflare 路由
- 没有 fallback DNS resolver

### 修复方案

**F1**: 用 `dnspython` 或 `c-ares` 显式解析 `zenodo.org` 到多个 IP，retry 不同 IP。

**F2**: 配置 `unbound` 或 `dnsmasq` 本地 DNS 缓存，写一个 `zenodo.zone` 文件覆盖 0.0.0.0 → 真实 IP。

**F3**: 写一个 `molmetal/scripts/zenodo_fetch.py` 包装脚本，把 `--resolve` 写进 curl options，简化调用。

### Effort: 1-2 h（取决于方案）

## 8. 综合修复优先级与执行顺序

```
[P1 YuelBond decode_ratio=0]    [EFFORT: 2-4 h]    [BLOCKS: R17 GPU retrain]
   └─ F1: AMP→FP32 (30 min)
   └─ F2: LR warmup (1 h)
   └─ F4: clamp logits (15 min)
   └─ F3: PCGrad (3 h)

[P2 flow_matching loader]        [EFFORT: 1-2 h]    [BLOCKS: any CFM training]
   └─ F1: configure_reference auto-detect
   └─ F2: _reference __init__ __path__ + upstream pre-load
   └─ F3: re-apply 4 shim files (already written)

[P3 SyntheMol/BioLM paths]       [EFFORT: 30 min]   [BLOCKS: round-4 lambda tests]
   └─ F1: SyntheMol adapter path candidates
   └─ F2: BioLM-Score adapter path candidates

[P4 stale .venv]                  [EFFORT: 30 min]   [DEGRADES: disk usage, slow uv]
   └─ F1: tighten env.sh
   └─ F2: pyproject [tool.uv] managed=false
   └─ F3: pre-commit detection

[P7 Zenodo fallback]              [EFFORT: 1-2 h]    [BLOCKS: re-downloads]
   └─ F3: wrapper script (lowest cost)
```

**推荐顺序**: P2 → P3 → P4 → P1 → P7。

P2/P3/P4 解锁 CFM 训练和 round-4 测试。P1 是 GPU retrain 的核心 fix。P7 是 nice-to-have。

## 9. 可立即 dispatch 的 ultracode workflow

**Workflow D — R17-fix**: CPU-only, ~4 h, 单 workflow
- Phase 1: P2 flow_matching loader auto-detect + shim re-apply (2 agents: explore + write+test)
- Phase 2: P3 SyntheMol/BioLM path fallback candidates (2 agents: write+test each)
- Phase 3: P4 pyproject + env.sh tighten (1 agent)
- Phase 4: P1 F1+F2+F4 (CFM BF16→FP32 + LR warmup + clamp logits) — needs GPU, 3 agents
- Phase 5: P7 Zenodo wrapper script (1 agent)
- Phase 6: synthesis report (1 agent)

Total: ~10 agents, all CPU-or-GPU, can dispatch when user approves.

## 10. Definition of Done

1. ✅ `pytest molmetal/tests/test_synthemol_reactions.py molmetal/tests/test_synflownet_env.py molmetal/tests/test_biolm_score_adapter.py molmetal/tests/test_lambda_reward_coupling.py` → 43+ passed, 7 skipped
2. ✅ `LipmanFlowMatchingAdapter.setup()` 在 Blackwell 上不抛 FileNotFoundError
3. ✅ YuelBond 3-seed smoke at step 1000/2000 → `decode_ratio > 0` 在 3/3 seeds
4. ✅ `mol-metal/.venv` 不再出现
5. ✅ `molmetal/data/MetalCytoToxDB.csv` 在 `.data/molmetal/` 下被 `MOLMETAL_DATA_DIR` 正确解析

## 11. References

- `molmetal/reports/wf_r16_yuelbond_2000_probe/final.md` — R16 2000-probe PASS
- `molmetal/reports/wf_r16_yuelbond_10000/final.md` — R16 10k-step INCONCLUSIVE（GPU hang）
- `TODO/pending/24_cfm_architecture_redo_plan.md` — CFM 架构重做 + P1.3 PCGrad
- `TODO/pending/31_r17_ultracode_dispatch.md` — R17 调度
- `TODO/pending/32_ultracode_trackB_lambda_calculus.md` — Track B（已完成）
- `molmetal/adapters/flow_matching_lipman/_reference_loader.py` — 需要修改的 loader
- `molmetal/adapters/flow_matching_lipman/_reference/{path,path/scheduler,solver,utils}/` — 已写好但被回滚的 shim
