# Lambda Track 双线并行合并门槛 / 自查表

**Date**: 2026-09-25
**Status**: Active — 当前两条 ultracode 并行运行中（task IDs `wpxpczzgp` + `whc1k10la`）
**Purpose**: 设定明确的"两条线什么时候可以合并 / 合并前必须自查什么"门槛

---

## 1. 当前两条线（ultracode 任务清单）

| Line | Task ID | 关注点 | Phase 数 |
|---|---|---|---|
| **B-line** (规则 + 修复) | `wpxpczzgp` | 84-rule MLE matrix + 4 NEW SMARTS + Deflexpressor + flag fix + 10×3 verify | 6 |
| **C-line** (论文 + 数据) | `whc1k10la` | Deflex cite pass + coupling pretrain + pipeline 10-pocket expand + coupling ablation | 5 |

---

## 2. 为什么需要 merge gate？

如果不设门槛直接合并，会发生：

1. **接口不匹配**: B-line 改的 `pt_click_compat.py`（84-rule matrix） vs C-line 的 `coupling_adapter.py`（LambdaASTSR）如果接口签名不一致，import 会爆
2. **数据流断**: B-line 用 `r4_lambda_only_run.py --metal-seed cisplatin` 路径，C-line 用 `r4_lambda_only_run.py --r17-paper-grade`，参数语法不统一会 fail
3. **回归风险**: 合并时可能让之前 30/30 probes = 1.000 的成果掉回去
4. **算力浪费**: 合并后才发现某个 track 的 output 不可用 → 必须返工

---

## 3. 合并的 5 个门槛（必须 ALL PASS）

### Gate 1 — 独立完工（per-track completion）

每条线自己先把全部 phase 跑完、所有 phase 都有 pass verdict。

**自查命令**:
```bash
# B-line: check all 6 phases PASS in journal
cat /home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_d0decfa9-7a0/journal.jsonl | \
  python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line.strip())
    if d.get('type') == 'result':
        print(f\"  {d.get('label','?')}: {'PASS' if d.get('result') else 'FAIL'}\")
"

# C-line: same for wf_a52c87ce-f55
cat /home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_a52c87ce-f55/journal.jsonl | \
  python3 -c "..."
```

**Gate 1 PASS 条件**: 两条线的所有 phase 都返回 result（非空、非 error）。

---

### Gate 2 — 接口契约一致（interface contract）

合并前必须验证 4 个 skill 关注的接口都还兼容。

#### 2a. CFM 接口（mol-metal-cfm-training skill）

```bash
# 1. LipmanFlowMatchingAdapter 仍可 import + setup
cd /home/hugo/codes/mol-metal
source env.sh
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
a = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2)
a.setup(device='cuda')
print('CFM import + setup: PASS')
print('atom_vocab:', a.atom_vocab)
"

# 2. decode smoke 仍能跑
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/metallo_drug_smoke_retrain.py \
  --steps 100 --batch-size 4 --seed 0 \
  --hidden-dim 64 --n-layers 2 \
  --lr-warmup-steps 20 \
  --decode-smoke-every 50 \
  --out-dir /tmp/merge_gate_cfm_smoke 2>&1 | grep "decode_smoke\|VERDICT"
```

**期望**: decode_ratio > 0.5（合并后不能掉到 0）

#### 2b. Lambda 接口（mol-metal-lambda-calculus skill）

```bash
# 1. atoms/combinators.py 仍可 import
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.molmetal_lam.atoms.combinators import PRIMITIVE_ATOMS, METAL_SYMBOLS
print(f'PRIMITIVE_ATOMS: {len(PRIMITIVE_ATOMS)} entries')
print(f'METAL_SYMBOLS: {len(METAL_SYMBOLS)} metals')
assert len(PRIMITIVE_ATOMS) >= 42, f'atom regression: {len(PRIMITIVE_ATOMS)} < 42'
assert len(METAL_SYMBOLS) >= 30, f'metal regression: {len(METAL_SYMBOLS)} < 30'
print('atoms/combinators: PASS')
"

# 2. pt_click_compat 注册规则数（合并前 B-line 应 ≥ 101）
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.molmetal_lam.lam_chem.pt_click_compat import list_registered_rules
rules = list_registered_rules()
print(f'Click rules: {len(rules)}')
for r in rules[:5]: print(f'  - {r}')
"

# 3. MCTSProofSearch with PLL heuristic
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.molmetal_lam.search_alg.proof_search import MCTSProofSearch
m = MCTSProofSearch(use_pll_heuristic=True, fpu_reduction=0.25)
print('MCTS+PLL: PASS')
"
```

#### 2c. Click chemistry 接口（mol-metal-click-chemistry skill）

```bash
# 84-rule matrix 调用路径仍有效
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.molmetal_lam.lam_chem.pt_click_compat import register_metal_ligand_exchange_rules
rules = register_metal_ligand_exchange_rules()
print(f'MLE rules registered: {len(rules)}')
assert len(rules) >= 84, f'MLE regression: {len(rules)} < 84'
print('Click chemistry: PASS')
"
```

#### 2d. Anticancer 接口（mol-metal-anticancer skill）

```bash
# MetalCytoToxDB + PlatinAI 加载仍正常
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.data.cytotox import load_metal_cytotox
from molmetal.molmetal_lam.lam_chem.platinai_dataset import load_platinai_subset
cyto = load_metal_cytotox()
plat = load_platinai_subset(n=100)
print(f'MetalCytoToxDB rows: {len(cyto)}')
print(f'PlatinAI subset: {len(plat)}')
"
```

**Gate 2 PASS 条件**: 4 个 skill 接口全部 import 通过 + 没有 regression。

---

### Gate 3 — 测试套件全绿（test suite）

```bash
cd /home/hugo/codes/mol-metal
source env.sh
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_f2a_metal_ligand_exchange.py \
  molmetal/tests/test_synthemol_reactions.py \
  molmetal/tests/test_lambda_combinators_expansion.py \
  molmetal/tests/test_pll_signature.py \
  molmetal/tests/test_reducer.py \
  molmetal/tests/test_lambda_astsr.py \
  molmetal/tests/test_coupling_adapter_platinai5k.py \
  molmetal/tests/test_lambda_reward_coupling.py \
  molmetal/tests/test_biolm_score_adapter.py \
  molmetal/tests/test_synflownet_env.py \
  -q --tb=short
```

**期望**: pass 数 ≥ 217 (R17 baseline 数字) + B-line 新加的 tests + C-line 新加的 tests（如果任何 phase 拉了 baseline 之下 = FAIL）。

**Gate 3 PASS 条件**: 测试套件 ≥ baseline。

---

### Gate 4 — 端到端 smoke（joint integration）

合并两条线的产物，跑一个 100-pocket × 1-seed smoke test，验证 metric 不回归。

```bash
cd /home/hugo/codes/mol-metal
source env.sh

# 1. 用 B-line 的 84-rule + LambdaASTSR + flag fix
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/r4_lambda_only_run.py \
  --metal-seed cisplatin \
  --pockets 10 --seeds 0 \
  --n-simulations 1000 \
  --click-rules auto-pt-strict \
  --output-dir merge_gate_smoke \
  --r17-paper-grade \
  --use-lambda-astsr 2>&1 | tee logs/merge_gate_smoke.log
```

**期望指标**:
- valid ≥ 0.95
- n_distinct ≥ 3 metal-ligand pairs（per TODO-25 §3 gate）
- SA < 3.5, QED > 0.55
- 不出现 `TypeError`、`ImportError`、`AttributeError`

**Gate 4 PASS 条件**: 10-pocket smoke 在所有 metric 上 ≥ R16 baseline。

---

### Gate 5 — 论文 + 文档同步

合并前必须更新文档，让以后的人能理解全貌。

**自查清单**:

- [ ] `paper/refs.bib` 含 `yu2026deflex`、`krasnov2026metalcytotox`、`liu2026pll`、`balcells2020tmqm` 等 9+ 篇必引论文
- [ ] `paper/sections/03_5_deflex.tex` 已扩展（C-line 完成的 Deflex Deep Cite）
- [ ] `paper/sections/04_method.tex` 已包含 84-rule matrix + LambdaASTSR
- [ ] `paper/sections/06_limitations.tex` 含 honest framing（failing case + F5→ASTSR 的 rationale）
- [ ] `molmetal/scripts/metallo_drug_smoke_retrain.py` 的 docstring 更新到 R17 production
- [ ] `.claude/skills/mol-metal-lambda-calculus/SKILL.md` 含 84-rule matrix + KataGo MCTS + LambdaASTSR
- [ ] `.claude/skills/mol-metal-click-chemistry/SKILL.md` 含 84-rule matrix + 4 NEW SMARTS
- [ ] `TODO/INDEX.md` 顶部时间戳更新到合并日期
- [ ] `TODO/01_strategy/PROJECT_STATEMENT.md` 已写完且 review 过
- [ ] `TODO/01_strategy/INSIGHTS_FROM_WORKFLOWS.md` 已写完且 review 过

**Gate 5 PASS 条件**: 上述清单逐项 checked。

---

## 4. 自查表（合并前必填）

```
=== MERGE_GATE.md 自查表 (合并日期: __________) ===

[ ] Gate 1 — 独立完工
    [ ] B-line (wpxpczzgp) 6 phases 全 PASS
    [ ] C-line (whc1k10la) 5 phases 全 PASS

[ ] Gate 2 — 接口契约一致
    [ ] CFM (LipmanFlowMatchingAdapter) import + setup PASS
    [ ] CFM decode_smoke 在 100 steps 后 ratio > 0.5
    [ ] atoms/combinators.py 仍 ≥ 42 entries + METAL_SYMBOLS ≥ 30
    [ ] pt_click_compat MLE rules ≥ 84
    [ ] MCTSProofSearch with PLL heuristic PASS
    [ ] MetalCytoToxDB + PlatinAI 加载正常

[ ] Gate 3 — 测试套件
    [ ] pytest 全套 ≥ 217 passed / 0 failed
    [ ] 任何新加 test 都进 CI

[ ] Gate 4 — 端到端 smoke
    [ ] r4_lambda_only_run.py 10-pocket × 1-seed smoke 完成
    [ ] n_distinct metal-ligand pairs ≥ 3
    [ ] valid ≥ 0.95, SA < 3.5, QED > 0.55
    [ ] 无 TypeError/ImportError/AttributeError

[ ] Gate 5 — 论文 + 文档同步
    [ ] paper/refs.bib 9+ 必引论文
    [ ] paper/sections/03_5_deflex.tex 已扩展
    [ ] paper/sections/04_method.tex 含 84-rule + ASTSR
    [ ] paper/sections/06_limitations.tex 含 honest framing
    [ ] 4 个 skill 都更新到 R17
    [ ] TODO/INDEX.md 时间戳更新

合并决策:
    [ ] 5 个 gate 全 PASS → 可合并
    [ ] 任何 gate FAIL → 不合并，先修该 gate

签字: ____________________
```

---

## 5. 合并失败时怎么办？

### Gate 1 FAIL: 某 phase 返回 error
- 不合并。先看 journal.jsonl 找 root cause
- 用 workflow resume 修复或重跑该 phase

### Gate 2 FAIL: 某个 import 报错或 ratio 掉到 0
- **高优先级**: B-line 可能改了 `pt_click_compat.py`，与 `coupling_adapter.py` 不兼容
- 回退：git checkout HEAD~ 找出 regression commit
- 重跑 merge smoke 确认

### Gate 3 FAIL: 某 test 失败
- 看 test 名字判断是哪条线破坏
- 如果是 B-line 破坏：回退 B-line 最近 commit，重跑 Gate 2 + 3
- 如果是 C-line 破坏：同上

### Gate 4 FAIL: smoke metric 回归
- 看哪个 metric 退步最多
- 常见原因：(a) LambdaASTSR 预测有 bug；(b) 84-rule matrix 引入了无效 SMARTS
- 解决：先 binary-search 关掉 LambdaASTSR 看是否回到 baseline；再 binary-search 关 84-rule matrix

### Gate 5 FAIL: 文档缺失
- 不阻塞合并，但合并后 1 周内必须补齐
- 用 TodoWrite 记录

---

## 6. 与 4 个 skill 的对应关系

| Skill | 在 merge gate 中的角色 | 必查项 |
|---|---|---|
| **mol-metal-cfm-training** | Gate 2a + Gate 4（end-to-end 的 metric 来源） | decode_ratio ≥ 0.5, atom_vocab 不 regression |
| **mol-metal-lambda-calculus** | Gate 2b（PLL MCTS、atoms table、PRIMITIVE_ATOMS） | 42-entry atom table、PLL 谱、Deflexpressor |
| **mol-metal-click-chemistry** | Gate 2c + Gate 3（84-rule MLE matrix） | rules ≥ 84, ≥ 4 NEW SMARTS |
| **mol-metal-anticancer** | Gate 2d（数据集接口） | MetalCytoToxDB + PlatinAI 不 regression |

---

## 7. 合并之后

合并后立刻做：

1. **跑 100-pocket × 3-seed 主实验**: per workflow I spec, 用 `--metal-seed cisplatin --n-simulations 1000`，目标 n_distinct ≥ 3, SA < 3.5, QED > 0.55
2. **3-axis SOTA benchmark**: vs TargetDiff / DiffSBDD / Pocket2Mol
3. **写 paper §4 Results**（占 5 页）
4. **DFT/extended validation** on 5-10 top candidates

---

## 8. 时间线参考

| 日期 | 事件 |
|---|---|
| 2026-09-18 | R16b ship (decode_ratio=1.000 first time) |
| 2026-09-25 | TODO-33 P1+P2 fix shipped, R17 10K retrain PASS, 2 ultracode 启动 |
| 2026-09-25 EOD | 预计 2 ultracode 完成（约 2-3h 后） |
| 2026-09-26 (T+1) | 跑 merge gate 全套 |
| 2026-09-26 PM | 100-pocket × 3-seed 主实验启动 |
| 2026-09-27 (T+2) | 主实验 + ablation 完成 |
| 2026-09-30 (T+5) | paper draft 1.0 内部 review |
| 2026-10-15 (T+3 周) | paper submission to CAS Zone 2 |
