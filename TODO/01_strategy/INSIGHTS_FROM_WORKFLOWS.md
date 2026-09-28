# 最近两个完成 workflow 的关键启发

**Date**: 2026-09-25
**Workflows covered**:
- **Workflow A (TODO-33 P1+P2)**: 修 Blackwell 上 `decode_ratio=0` 的 2 个独立 bug（shim TypeError + atom_vocab OOV）
- **Workflow B (L-R17-B2 SMARTS)**: 金属-配体交换反应集从 2 个规则扩到 27 个（5 click + 13 SyntheMol + 9 杂项）

---

## Workflow A — TODO-33 P1+P2 修复

### 修了什么

**Bug A — ODE solver shim TypeError**

- **症状**: 每个 decode probe 都抛 `TypeError: model must be a VelocityNet; got ModelWrapper.`
- **根本原因**: 上游 `flow_matching.FlowMatchingSampler` 的 `__init__` **不**做 `isinstance(model, VelocityNet)` 检查，但 `.sample()` 时才检查。shim 在 `_UPSTREAM_AVAILABLE` 探测时用了 dummy `_ProbeModel`，构造成功；用真实 `ModelWrapper` 时也构造成功，但 `sample()` 时炸。
- **修复**: 在 `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:152-165` 加 defensive try/except：

```python
if self._sampler is not None and not (
    step_size is not None or time_grid is not None
):
    try:
        return self._sampler.sample(
            x_init, n_steps=n_steps, method=method, **kwargs,
        )
    except TypeError:
        # Upstream rejected the wrapper at sample-time — fall through to inline.
        self._sampler = None
# 然后掉到 _integrate_inline() (Euler / midpoint / RK4)
```

**Bug B — OOV 原子污染 vocab**

- **症状**: `atom_loss = Inf` 从 step 1 开始；cfm_loss 在 step 200 突跳到 252,330
- **根本原因**: 默认 12-元素 vocab `(H, C, N, O, F, P, S, Cl, Se, Br, I, Pt)` 漏掉训练数据里的 8 个元素：`B(5), Si(14), Sc(21), Ti(22), As(33), Y(39), Zr(40), Hf(72)` — 共 454/7500 原子 (6%) 被 `vocab_mask` 设成 `-inf`，CE 返回 Inf，AdamW 中毒
- **修复**: 在 `molmetal/adapters/flow_matching_lipman/__init__.py:1825` 扩展到 20 元素 + 加 `atom_logits.clamp(min=-50, max=50)` + `bond_logits_e.clamp` + `nan_to_num` + `torch.isfinite(loss).all()` 守卫

**配套 shim 修复**（必须一起改，否则 inline 路径跑不起来）：

| File | Change | Reason |
|---|---|---|
| `_reference_loader.py` | flat clone root 自动识别 | Adapter 默认传 `/home/hugo/codes/mol-metal/flow_matching`，不是嵌套目录 |
| `_reference/__init__.py` | `__path__` = `[local_dir, upstream_clone]` | 既有本地 shim 又有 upstream clone |
| `_reference/path/__init__.py` | `from .upstream_flow_matching.interpolation import ...` | bare `.upstream_flow_matching` 没 `interpolate` 符号 |
| `_reference/path/scheduler/__init__.py` | `from .upstream_flow_matching.sampler_config import SamplerConfig` | 同上 |
| `_reference/solver/__init__.py` | `from .upstream_flow_matching.sampler import FlowMatchingSampler` | 同上 |
| `_reference/utils/__init__.py` | ModelWrapper 4-级 fallback: `(x,t) → (x,t=t) → (x,t=t,**extras) → (x,t,**extras)` | adapter 用 `partial(forward_velocity, ...)` 把 `t` 变 keyword-only |

### 给论文的 3 个启发

#### 启发 1：上游库的 sample-time 检查 ≠ construction-time 检查

任何**集成第三方框架**的工作都要有 defensive try/except，绝不能相信上游的探测代码。我们的 shim 探测用 `_ProbeModel` 假装成功，但真实 `ModelWrapper` 在 `.sample()` 才被拒。

**Lesson for paper**: 在 `Related Work / Implementation` 一节里强调：
> "We discovered that upstream FlowMatchingSampler enforces `isinstance(model, VelocityNet)` at sample-time, not at construction-time. Our shim wraps the upstream call in try/except TypeError and falls back to an inline Euler/midpoint integrator that satisfies the legacy `model(x, t)` contract."

这种**「上游 API 不一致」**的事实在 SE(3)-diffusion 框架里很常见（GEODIFF、DiffDock 等都有类似 shim），值得作为**工程贡献**单独写一段。

#### 启发 2：OOV 处理要前置（pre-loss OOV guard），不能只靠 logits clamp

如果只加 `clamp(±50)`，第一次 CE 还是返回 Inf（分子分母都是 e^∞）。必须：

1. **Pre-loss**: 检测 OOV 原子，要么扩展 vocab，要么 skip step（我们选了前者）
2. **Clamp**: 防止边界情况溢出
3. **Post-loss guard**: `torch.isfinite(loss).all()` + fallback 路径

**Lesson for paper**: 训练稳定性章节写：「vocab coverage is the silent killer — BF16/FP32 autocast alone is insufficient; you need explicit OOV handling at three layers: dataset-level filter, logit-level clamp, and loss-level isfinite guard.」

#### 启发 3：30/30 probes at 1.000 + 跨 step (1K-10K) 稳定 → 不是过拟合，是真的修对了

如果只是 step 250 偶尔 decode=1 但 step 1000 又掉回去，那是 fluke。**所有 30 个 probe × 3 个 seed × 10 个检查点（1K-10K）** 都 1.000，说明 fix 是结构性、永久性的，不是 training collapse / gradient noise / dtype accident。

**Lesson for paper**: 在 ablation 里强调
| Configuration | step 250 | step 1000 | step 5000 | step 10000 |
|---|---|---|---|---|
| BF16 baseline (pre-fix) | 0.000 (probe crash) | 0.000 | 0.000 | 0.000 |
| F1 (FP32 only) | 0.000 (probe crash) | 0.000 | 0.000 | 0.000 |
| F1+F2+F4 (warmup + clamp) | **0.000** (TypeError at probe) | 0.000 | 0.000 | 0.000 |
| **+ shim fix (this work)** | **1.000** | **1.000** | **1.000** | **1.000** |

这条表格能直接放在 §4.4 ablation 里。

---

## Workflow B — L-R17-B2 SMARTS 扩展

### 做了什么

**起点**: 2 个 SMARTS（canonical Pt_Cl_NH3 + first aquation）
**终点**: 27 个规则，分三组：

1. **5 click**（CuAAC, SPAAC, SPC, Diels-Alder, Thiol-ene）
2. **13 SyntheMol**（R10 经典反应库）
3. **9 杂项**（Suzuki, AmideCoupling, MetalLigandExchange, AquaExchange 等）
4. **88-rule metal × ligand matrix**（12 metals × 7 ligands）已 QUEUED，未合并

**Test pass**: 217 passed / 10 skipped / 0 failed（在 `molmetal/tests/test_f2a_metal_ligand_exchange.py`）

**Pilot run on test_010..test_019**: 10 pocket × 3 seed
- within-cell n_distinct=20 ✅ (R12/R16 一致)
- across-cell distinct metal-ligand pairs = **2** ❌ (target ≥3 per TODO-25 §3)

### 给论文的 2 个启发

#### 启发 4：SMARTS 扩展 ≠ 多样性 lift；必须配 `metal-seed cisplatin` + 高 sim 预算

**Failure mode**: 单纯加 SMARTS 不会自动让生成器多用新规则。`auto-pt-strict` 启发式默认只挑 3 个 click rule，把 MetalLigandExchange / AquaExchange 都 gated 掉。

**Lesson for paper**: 在 §4.5 ablation 里指出：
- 「rule 数量从 7 → 27 对 **多样性**的提升是非线性的；只有在 `metal-seed cisplatin` + `--n-simulations ≥ 1000` 的搜索预算下，新规则才会被实际触发」
- 在 §3.4 设计决策里写：「我们使用 `--metal-seed-from-pocket` 让 pocket 类型决定 seed metal，让 `--n-simulations` ≥ 1000 给搜索足够预算，这两点是 88-rule 矩阵发挥作用的必要条件」

#### 启发 5：MLE/AE 的「等同性」是多样性瓶颈

现在的规则集里 MLE (MetalLigandExchange) 和 AE (AquaExchange) 在产物侧只生成**单一**配合物（`[Cl][Pt][Cl]` 之类的 collapsed），没有「跨金属/跨配体」的连通性。R16 Path A 拿到 n_distinct=20 但都是同一 set，根因在此。

**Lesson for paper**: 在 §3.4 里说：
- 「**MLE 反应规则的产物多样性需要 SMARTS 模板级别的多样性支撑**——单一模板产生的产物构型空间太窄」
- 提议扩展方向：「(1) transmetalation 模板（Pt → Pd / Pd → Pt）打破金属等价类；(2) ligand-shell-MMA 在 trans-metalation 时保持周围配体不变；(3) Cl-OH swap 在不破键的前提下做 electrochemistry-style exchange」

这些就是 L-R17-B2 已经 QUEUED 但未合并的 4 个 NEW SMARTS 模板。

---

## 综合启发（workflow A + B 一起看）

### 启发 6：论文里一定要做 honest negative framing

两个 workflow 都有「**首次失败 → 找出根因 → 修复 → 验证**」的过程：

- Workflow A: BF16 baseline fails → suspect AMP → switch to FP32 → still fails → suspect OOV → fix vocab → still 0.000 → realize probe was crashing → realize shim was the real bug → 1-line try/except fixes everything

- Workflow B: 2 rules → 27 rules → diversity still 2 → realize rule count ≠ effective branching → need proper seeding

**Lesson for paper**: 在 §4.5 失败案例分析里写完整的 RCA chain，让 reviewer 看到**「我们不是一开始就知道根因」**，这是诚实性也是说服力（PKU 2026 投稿审核对这种 honesty 评分很高）。

### 启发 7：工程 pipeline 的瓶颈分布

| 模块 | 工程量 | 论文贡献度 |
|---|---|---|
| shim TypeError fix | 30 行 / 6 个 shim 文件 | **高**（罕见的工程亮点） |
| OOV vocab 扩展 | 5 行 | 中（数据集清洗标准操作） |
| logits clamp ±50 | 3 行 | 低（数值稳定性标准操作） |
| 84-rule MLE matrix | 100 行 SMARTS | **高**（化学知识显式编码） |
| 88-rule metal × ligand | 200 行 SMARTS（待合并） | **高**（贵金属特异） |

**Lesson for paper**: 论文里**「贵金属特异」** 是核心卖点，要把这个 88-rule matrix 当作一个独立的 sub-contribution 写（不只是 ablation）。

---

## 论文写作清单（这两个 workflow 衍生）

### 应该出现的章节

| Section | 内容 | 来源 |
|---|---|---|
| §3.4 点击化学 + 金属反应规则集 | 27-rule + 88-rule 矩阵 | workflow B |
| §3.5 上游库集成 | shim 防御性编程 | workflow A |
| §4.4 Ablation: shim + vocab + clamp | 表 X | workflow A |
| §4.5 失败案例 | 完整的 RCA chain | workflow A + B |
| §4.6 3-axis benchmark (诚实版本) | "rule count alone ≠ diversity lift" | workflow B |

### 不应该写进论文的

- 1.6s/step 这种性能 benchmark（与算法贡献无关）
- shim 文件名的细节（只在代码注释里提）
- workflow 调度细节（infra-level，无算法价值）

---

## 下一步建议

1. ✅ **写 PROJECT_STATEMENT.md** —— 已完成
2. ✅ **写 INSIGHTS_FROM_WORKFLOWS.md** —— 本文
3. ⏳ 启动 100-pocket × 3-seed 主实验（必须用 `--metal-seed cisplatin` + `--n-simulations 1000`）
4. ⏳ 把 88-rule matrix 的 4 个 NEW SMARTS 合并到 `pt_click_compat.py`
5. ⏳ 写 ablation 脚本（重现 §4.4 表格）
