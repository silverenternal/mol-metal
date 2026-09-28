# TODO 82 — Wf-36 Phase 1: Closure theorem end-to-end verifier (L1)

**Date**: 2026-09-26 (Wf-36 Phase 1 ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIP (L1 gap closed — closure theorem end-to-end benchmark operational)
**Parent**: `TODO/pending/83_lambda_algorithm_enhancement.md` §0 (L1-L8 lambda algorithm gaps)
**Sibling inputs**:
- `molmetal/molmetal_lam/lam_chem/closure.py` (R17 ship, NOT modified)
- `molmetal/molmetal_lam/atoms/combinators.py` (42-entry `PRIMITIVE_ATOMS`, R17 ship)
- `molmetal/scripts/wf_lambda4_verify.py` (WF-Lambda-4 ship, 3 closure scenarios)
- `molmetal/scripts/closure_real_benchmark.py` (Wf-17 P4.2 ship, PlatinAI 5K benchmark)
- `molmetal/data/tmqm_dblock_subset.csv` (Wf-28 ship, 5K × 30 d-block metals)
- `molmetal/data/platinai_5k.csv` (R17 ship, 5K real Pt-group anticancer)

---

## §1 TL;DR

Wf-36 Phase 1 (L1) **closure theorem end-to-end verifier** 全部 ship — 对 9 task-spec metal (Pt/Ru/Ir/Rh/Re/Os/Au/Pd/Ag) × 100 SMILES = 836 real d-block complexes 跑 closure theorem, 产出:

| Deliverable | 状态 | 文件 | 规模 |
|---|---|---|---|
| **Verbatim bench JSON** | ✅ SHIP | `molmetal/reports/wf36_closure_theorem/closure_bench.json` | 836 samples × 16 fields per sample |
| **Per-metal CSV** | ✅ SHIP | `molmetal/reports/wf36_closure_theorem/per_metal.csv` | 9 rows × 12 fields (spreadsheet audit) |
| **Human summary MD** | ✅ SHIP | `molmetal/reports/wf36_closure_theorem/closure_bench.md` | run metadata + aggregate + per-metal table |
| **Verbatim verifier** | ✅ SHIP | `molmetal/scripts/verify_closure_theorem.py` | 841 lines, full argparse CLI |
| **Smoke script** | ✅ SHIP | `molmetal/scripts/smoke_closure.py` | 138 lines, 1-molecule fast smoke |
| **Test suite** | ✅ SHIP | `molmetal/tests/test_closure_theorem_verifier.py` | 33 tests (20 contract + 13 bonus invariants) |

**MEASURED key numbers (tmQM d-block 836 samples × 9 metals)**:

| Metric | Value |
|---|---|
| Total samples run | 836 |
| closure_ok count | 427 |
| closure_reachability_rate (overall) | **0.5108** |
| mean_wall_time per sample | **0.084 s** |
| peak_memory_mb | **0.73 MB** |
| total_wall_time | **72.6 s** |
| mean_n_products per sample | **9.19** |
| mean_beta_depth seen | **0.999** |
| n_error | 0 (100% parse success) |

**Per-metal reachability (sorted ascending)**:

| Metal | N | closure_ok | rate | mean_wall |
|---|---|---|---|---|
| Ir | 100 | 38 | 0.38 | 0.074s |
| Os | 55 | 22 | 0.40 | 0.052s |
| Au | 100 | 44 | 0.44 | 0.053s |
| Pd | 100 | 45 | 0.45 | 0.057s |
| Pt | 100 | 47 | 0.47 | 0.088s |
| Re | 100 | 54 | 0.54 | 0.081s |
| Rh | 100 | 55 | 0.55 | 0.106s |
| Ag | 81 | 50 | 0.62 | 0.080s |
| Ru | 100 | 72 | **0.72** | 0.148s |

**Verdict**: Closure theorem 端到端可达 — 但真实数据 closure_reachability_rate 只 0.51 (整体)。L1 gap closed, L2-L8 (variance MCTS / joint 3-stage / replay learner / Hilbert-ASTSR / d-block bench / multi-PH+PLL / triangle closure) 仍在 TODO-83 待办。

---

## §2 Lambda algorithm vulnerabilities exposed by this benchmark

按 user 要求 "lambda 算法那条线的能力也要找漏洞做加强", Wf-36 Phase 1 实跑暴露 3 个明显 algorithm-level gaps:

### §2.1 Gap A — closure_reachability_rate 只有 0.51 (一半不通过)

**Symptom**: 836 real d-block complexes 跑 closure theorem, 只有 427 (51%) 通过。

**Root cause** (诚实归因):
- tmQM data 用 dative-bond `[M+]<-` notation, RDKit valence-BNF predicate 把 MLC ledger 当作 under-saturated (over-conservative)。
- 启动 term 已经 fail valence-BNF → closure_theorem 直接 return False, 不再展开 BFS。
- Ir (38%) / Os (40%) / Au (44%) 率最低 — 都因 dative 配位 chemistry。

**Fix 方向** (R20+): 让 valence-BNF predicate 对 d-block `[M+]<-` 配位使用 metal-specific relaxed valence, 而不是 main-group carbon-style saturated check。

### §2.2 Gap B — partner terms 7 条 only

**Symptom**: Bi-molecular click rules 需要 partner term 才能 fire, 但只 hard-code 7 条 (propargylamine / azide / propyne / thiol / propene / benzoic acid / amine)。

**Root cause**: 现有 partner 列表没有 d-block-specific tiles (e.g. Pt_II-chloro-azide / Ru-NH3-H2O)。`Pt_II + NH3 + Cl` 没有 CuAAC partner tile 时只能展开 self-loop (无 product)。

**Fix 方向**: 增加 14 条 metal-coordination partner tiles (W-flow 写进 `--metals` arg)。

### §2.3 Gap C — depth=1 only

**Symptom**: Default `--max-depth=1`, 跑 closure 只能 expand 1 click application. 实际 multi-step synthesis (e.g. CuAAC → AmideCoupling → SPAAC) 没法验证。

**Root cause**: Bi-molecular click rules 在 closure BFS 中呈 O(B · N · |R|) 增长, depth=2 已经 `n_products=10` (cisplatin) / `38` (oxaliplatin). depth=3 预计 10² × N.

**Fix 方向**: Wf-37 (L6) 跑 closure d-block 30 metal, 把 depth=2 / depth=3 一起跑通。

---

## §3 Per-deliverable

### §3.1 `molmetal/scripts/verify_closure_theorem.py`

**Functional scope**: 端到端 verifier, argparse 6 flags:
- `--dataset {tmqm,platinai}` — 默认 tmqm (30-metal coverage)
- `--n-samples N` — 每个 metal N 条 SMILES (默认 100)
- `--metals Pt Ru ...` — 显式列表, 或 `all_dblock` (30 metals)
- `--max-depth N` — BFS depth budget (默认 1)
- `--max-products N` — ProductiveSpace 安全 cap (默认 64)
- `--smoke` — smoke mode

**Output 3 files**:
- `closure_bench.json` — full per-sample + per-metal aggregate JSON
- `per_metal.csv` — 1 row per metal (spreadsheet audit)
- `closure_bench.md` — human-readable summary

**Schema (closure_bench.json)**:
```
{
  "schema_version": 1,
  "tool": "verify_closure_theorem.py",
  "pilot_id": "wf36_closure_theorem",
  "metals": [...9 task-spec metals...],
  "n_samples_per_metal": 100,
  "n_samples_total": 836,
  "max_depth": 1,
  "click_rule_names": ["CuAAC","SPAAC","ThiolEne","Suzuki","AmideCoupling"],
  "per_metal": { "Pt": { "aggregate": {...}, "records": [...] }, ... },
  "per_sample": [...836 records...],
  "aggregate": {...overall...}
}
```

### §3.2 `molmetal/scripts/smoke_closure.py`

**Functional scope**: 1-molecule fast smoke check
- 默认 SMILES = cisplatin `[NH3][Pt]([NH3])([Cl])([Cl])`
- 跑 closure_theorem at depth 1 + 7 partner terms
- 输出 closure_ok + n_visited + n_products + wall_time_s

**Smoke result**: cisplatin canonical = `[NH2][Pt]([NH2])([Cl])[Cl]`, closure_ok=True, n_visited=18, n_products=10, wall=0.057s.

### §3.3 `molmetal/tests/test_closure_theorem_verifier.py`

**20 contract tests + 13 bonus invariants = 33 tests, all PASS in 5.21s**:

| # | Test | Verifies |
|---|---|---|
| 1 | test_cisplatin_closure | cisplatin passes closure_theorem |
| 2 | test_oxaliplatin_closure | oxaliplatin passes closure_theorem |
| 3 | test_carboplatin_closure | carboplatin analogue passes closure_theorem |
| 4 | test_Pt_II_closure | Pt_II square-planar class |
| 5 | test_Pt_IV_closure | Pt_IV octahedral class |
| 6 | test_Ru_II_III_closure | Ru_II + Ru_III octahedral class |
| 7 | test_module_imports | script imports cleanly |
| 8 | test_task_spec_metals_present | 9 task-spec metals defined |
| 9 | test_partner_smile_list_size | partner list has ≥ 6 tiles |
| 10 | test_default_dataset_is_tmqm | default dataset = tmqm |
| 11 | test_default_n_samples_per_metal_is_100 | default --n-samples = 100 |
| 12 | test_per_metal_closure_rate_in_unit | closure rate ∈ [0, 1] |
| 13 | test_per_metal_records_have_all_keys | every record has all keys |
| 14 | test_closure_ok_is_bool_per_record | closure_ok is bool |
| 15 | test_wall_time_non_negative | wall_time_s ≥ 0 |
| 16 | test_memory_non_negative | memory_mb ≥ 0 |
| 17 | test_max_depth_seen_bounded | max_depth_seen ≤ max_depth |
| 18 | test_per_metal_aggregate_keys | per-metal aggregate has keys |
| 19 | test_closure_bench_json_written | JSON written + schema OK |
| 20 | test_per_metal_csv_written | per_metal.csv has header |
| Bonus 1-13 | load helpers / is_pt_coordinated / utcnow_iso / benchmark_one / smoke budget / platinai subset / aggregate empty / csv paths | invariants |

---

## §4 Honest framing

按 party-twenty-spirit §3.5:

- **R² / AUC 等 metric 不 ship**: L1 closure theorem verifier 只 ship bench schema + measured wall-time + closure-ok rate。不是 metric ship。
- **MEASURED vs PROJECTED**: 836 samples 全部 MEASURED (CPU-only 真实跑)。无 PROJECTED。
- **Real data, no synthetic**: tmQM CSV + PlatinAI CSV 都是 R17 / Wf-28 真实 ship data。无 mock fallback。
- **No closure.py modification**: 按 spec 约束, `closure.py` (R17 ship) 0-byte 修改。
- **5 click rules only**: Public-facing click rules (CuAAC/SPAAC/ThiolEne/Suzuki/AmideCoupling), 不含 `MetalLigandExchange` / `AquaExchange`. Wf-37 (L6) 才接 metal-coord rules.

---

## §5 Cross-reference

- `TODO/pending/83_lambda_algorithm_enhancement.md` — L1 closed; L2-L8 open
- `molmetal/molmetal_lam/lam_chem/closure.py` — R17 ship, untouched
- `molmetal/molmetal_lam/atoms/combinators.py` — R17 UC-B1, 30 d-block metals
- `molmetal/scripts/wf_lambda4_verify.py` — WF-Lambda-4 ship, 3 cisplatin scenarios
- `molmetal/scripts/closure_real_benchmark.py` — Wf-17 P4.2 ship, PlatinAI 5K benchmark
- `molmetal/data/tmqm_dblock_subset.csv` — Wf-28 ship, 5K × 30 d-block
- `molmetal/data/platinai_5k.csv` — R17 ship, 5K real Pt-group anticancer
- `.claude/skills/mol-metal-lambda-calculus/` — main skill for closure

## §6 Next (Wf-37 / L2-L8)

L1 closed. Next workflows:
- **Wf-36 Phase 2 (L2)** — Variance MCTS 真正接入 proof_search
- **Wf-36 Phase 3 (L3)** — ScoreRepellent + VariancePUCT + MCTS 三者联合
- **Wf-36 Phase 4 (L4)** — MCTS replay learner (AlphaGo-style)
- **Wf-37 Phase 1 (L5)** — Hilbert-ASTSR end-to-end training
- **Wf-37 Phase 2 (L6)** — closure d-block 30 metal × depth 2/3
- **Wf-37 Phase 3 (L7)** — multi-PH + PLL real extraction
- **Wf-37 Phase 4 (L8)** — ASTSR ↔ Hilbert ↔ MCTS triangle closure

**Date**: 2026-09-26 (Wf-36 Phase 1 ship)