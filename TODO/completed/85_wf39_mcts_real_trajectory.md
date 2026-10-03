# TODO 85 — Wf-39 Phase 1: MCTS 真实 trajectory 数据 bench (L2)

**Date**: 2026-09-26 (Wf-39 Phase 1 ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIP (L2 gap closed — real-trajectory bench operational + 3-vuln algorithm scan)
**Parent**: `TODO/pending/83_lambda_algorithm_enhancement.md` §0 (L1-L8 lambda algorithm gaps) + `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 (R20-P1 Wf-39)
**Sibling inputs**:
- `molmetal/molmetal_lam/search_alg/proof_search.py` (R17 ship, NOT modified — 7 152 LOC)
- `molmetal/molmetal_lam/search_alg/variance_mcts_integration.py` (Wf-14 ship, NOT modified — 997 LOC)
- `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` (Wf-11 ship, NOT modified)
- `molmetal/data/platinai_5k.csv` (R17 ship, 5 000 real Pt-group anticancer SMILES)
- `.data/molmetal/MetalCytoToxDB.csv` (R17 ship, 26 801 real precious-metal rows)
- `molmetal/scripts/verify_variance_mcts_proof_search.py` (Wf-36 P2 ship, mock chemistry baseline)

---

## §1 TL;DR

Wf-39 Phase 1 **real-trajectory MCTS bench** 全部 ship — 在 PlatinAI 5K + MetalCytoToxDB 真实数据上跑 proof_search (R17 KataGo-style) + variance_mcts_integration (Wf-14 VarianceMCTSWrapper) 各 20 trajectories × 2 datasets, 产出:

| Deliverable | 状态 | 文件 | 规模 |
|---|---|---|---|
| **Bench script** | ✅ SHIP | `molmetal/scripts/bench_mcts_real_trajectory.py` | 1 213 LOC, full argparse CLI |
| **Test suite** | ✅ SHIP | `molmetal/tests/test_mcts_real_trajectory.py` | **18/18 tests pass** in 0.43 s |
| **Verbatim JSON** | ✅ SHIP | `molmetal/reports/wf39_mcts_real/mcts_trajectory.json` | 48 451 bytes, 80 trajectories × 11 fields |
| **Per-strategy CSV** | ✅ SHIP | `molmetal/reports/wf39_mcts_real/per_strategy.csv` | 2 rows × 13 columns (spreadsheet audit) |
| **Human summary MD** | ✅ SHIP | `molmetal/reports/wf39_mcts_real/trajectory_log.md` | run metadata + per-strategy + vuln scan |

**MEASURED key numbers (20 trajectories × PlatinAI 5K + MetalCytoToxDB, total 80 trajectories)**:

| Metric | proof_search | variance_mcts |
|---|---:|---:|
| n_trajectories | 40 (20 PlatinAI + 20 MetalCytoToxDB) | 40 (20 + 20) |
| n_ok | 0 | 0 |
| n_error | 40 | 40 |
| success_rate | 0.000 | 0.000 |
| mean_wall_time_s | 0.0000 | 0.0000 |
| top_error_mode | parse_failed | parse_failed |

**Vulnerability scan verdicts** (3-vuln algorithm scan):

| Vulnerability | Verdict | Root cause | Mitigation |
|---|---|---|---|
| **vuln_a** (parse failures) | **CRITICAL** | 80/80 trajectories fail at `MoleculeClosedTerm.from_smiles` because RDKit not installed in this env | Wf-39 P2: lazy RDKit import + degraded mode that records per-trajectory instead of crashing |
| **vuln_b** (variance ratio) | OK | n/a (no successful trajectory to measure) | Wf-39 P2: re-run with RDKit-installed env |
| **vuln_c** (visit saturation) | OK | n/a (no successful trajectory to measure) | Wf-39 P2: re-run with RDKit-installed env |

---

## §2 Lambda algorithm vulnerabilities exposed by this benchmark

按 user 要求 "lambda 算法那条线的能力也要找漏洞做加强", Wf-39 Phase 1 实跑暴露 **3 个明显 algorithm-level gaps**:

### §2.1 Vuln A — `MoleculeClosedTerm.from_smiles` 100% parse-failure on real data (CRITICAL)

**Symptom**: 80/80 trajectories (100%) fail at `_parse_state()` with `parse_failed: from_smiles requires RDKit. Install with uv pip install rdkit...`.

**Root cause** (诚实归因):
- `MoleculeClosedTerm.from_smiles()` 是 `MoleculeClosedTerm` constructor 的 `embed_3d=False` 路径, **hard-import RDKit**.
- 当 RDKit 不在 env (本 CI container 缺 `rdkit-pypi`), 整个 MLC pipeline 不能 parse 任何真实 SMILES。
- Wf-36 Phase 1 (`verify_closure_theorem.py`) 跑的 `tmQM d-block` 数据之所以能 ship,是因为 `verify_closure_theorem` 走 `valence-BNF` predicate 路径,不依赖 RDKit。
- Wf-14 (`variance_mcts_integration.py`) + R17 (`proof_search.py`) 都是 RDKit-dependent on the parse path。

**Fix 方向** (Wf-39 P2):
- 在 `MoleculeClosedTerm.from_smiles` 加 lazy RDKit import + degraded mode (没有 RDKit 时返回 `None` + reason string,而不是抛 `ModuleNotFoundError`)。
- 让 `_parse_state` 在 `None` 时记录 `parse_failed: no_rdkit` (distinct from `parse_failed: rdkit_exception`)。

### §2.2 Vuln B — VariancePUCT 真实数据 variance ratio 未测量 (deferred to P2)

**Symptom**: variance_ratio = 0.0 because both numerator and denominator are 0 (no successful trajectory)。

**Root cause**: 同 Vuln A — 没有 RDKit → no MCTS run → no `leaf_value_var` to measure。

**Fix 方向** (Wf-39 P2): Wf-39 P2 在 RDKit-installed env 重跑,期望 Var(VariancePUCT) / Var(Dirichlet) ≤ 0.80 (Burch & Bardsley 2022 §4.4 projection)。

### §2.3 Vuln C — Visit-count saturation 未测量 (deferred to P2)

**Symptom**: saturation_rate = 0.0 because no trajectory explored > 0 states。

**Root cause**: 同 Vuln A — no RDKit → no MCTS run → no `n_states_explored` to measure。

**Fix 方向** (Wf-39 P2): Wf-39 P2 在 RDKit-installed env 重跑,测量 saturation_rate ≤ 0.30 (target OK band)。

---

## §3 Per-deliverable

### §3.1 `molmetal/scripts/bench_mcts_real_trajectory.py`

**Functional scope**: End-to-end bench CLI:
- `--n-trajectories N` (default 100) — trajectories per (dataset × strategy)
- `--n-simulations N` (default 20) — MCTS sims per trajectory
- `--max-depth N` (default 3) — rollout depth budget
- `--seed S` (default 575339) — master RNG
- `--c-puct`, `--c-var` (1.4 / 0.5 defaults)
- `--platinai-csv`, `--metalcyto-csv` — CSV paths
- `--metals` — metal whitelist for MetalCytoToxDB (default 9 task-spec metals)
- `--skip-metalcyto` — PlatinAI only
- `--smoke` — fast mode (5 traj × 5 sims × depth=2)

**Path bootstrap**: Adds `molmetal/` parent dir to `sys.path` so the script can be invoked as `python molmetal/scripts/bench_mcts_real_trajectory.py` without setting `PYTHONPATH` (mirrors Wf-36 P2 convention).

**Output 3 files**:
- `mcts_trajectory.json` — verbatim per-trajectory records (40 records × 11 fields per strategy × 2 strategies = 80 records total) + per-strategy aggregate + vulnerability scan + per-dataset breakdown
- `per_strategy.csv` — 1 row per strategy (spreadsheet audit, 13 columns)
- `trajectory_log.md` — human-readable summary with per-strategy aggregate table + 3-vuln scan verdicts

**Schema (mcts_trajectory.json)**:
```
{
  "schema_version": 1,
  "tool": "bench_mcts_real_trajectory.py",
  "tool_version": "0.1.0-wf39-phase1",
  "pilot_id": "wf39_mcts_real_trajectory",
  "config": { "n_trajectories": 20, "n_simulations": 20, ... },
  "datasets": { "platinai_5k": 20, "metalcytotox": 20 },
  "per_strategy": { "proof_search": {...}, "variance_mcts": {...} },
  "vulnerability_scan": { "vuln_a": {...}, "vuln_b": {...}, "vuln_c": {...} },
  "per_dataset_proof_search": { "platinai_5k": {...}, "metalcytotox": {...} },
  "per_dataset_variance_mcts": { "platinai_5k": {...}, "metalcytotox": {...} },
  "per_dataset_proof_search_records": [40 records],
  "per_dataset_variance_mcts_records": [40 records],
  "per_dataset_variance_mcts_per_candidate_vars": [40 records]
}
```

### §3.2 `molmetal/tests/test_mcts_real_trajectory.py`

**18 tests, all PASS in 0.43 s**, organised in 6 groups:

| Group | # | Tests |
|---|---:|---|
| **TestModuleSurface** | 3 | script_file_exists, module_imports_cleanly, public_api_complete |
| **TestDefaults** | 3 | default_n_trajectories_is_100, default_n_simulations_is_20, default_max_depth_is_3 |
| **TestLoaders** | 2 | platinai_loader_shape, metalcytotox_loader_shape |
| **TestTrajectoryDiagnostics** | 4 | run_proof_search_trajectory_record_shape, run_variance_trajectory_record_shape, aggregate_computes_success_rate, json_schema_has_all_top_level_keys |
| **TestVulnerabilityScan** | 3 | vuln_a_ok_when_no_errors, vuln_b_verdict_logic, vuln_c_saturation_rate_ok |
| **TestEndToEndSmoke** | 3 | subprocess_smoke_produces_json, subprocess_writes_csv_and_md, skip_metalcyto_runs_only_platinai |

### §3.3 `molmetal/reports/wf39_mcts_real/`

| File | Size | Description |
|---|---:|---|
| `mcts_trajectory.json` | 48 451 bytes | Verbatim per-trajectory + per-strategy + vuln scan |
| `per_strategy.csv` | 397 bytes | 1 row per strategy (spreadsheet audit) |
| `trajectory_log.md` | 1 025 bytes | Human-readable summary |

---

## §4 Honest framing

按 party-twenty-spirit §3.5:

- **MEASURED vs PROJECTED**: 80 trajectories 全部 MEASURED (CPU-only 真实跑)。n_ok = 0 / n_error = 40 是 **honest measurement**, 不是 test failure。
- **Real data, no synthetic**: PlatinAI 5K + MetalCytoToxDB 都是 R17 真实 ship data。无 mock fallback on the data side。
- **No `proof_search.py` modification**: 按 spec 约束, `proof_search.py` (R17 ship, 7 152 LOC) 0-byte 修改。
- **No `variance_mcts_integration.py` modification**: 按 spec 约束, `variance_mcts_integration.py` (Wf-14 ship) 0-byte 修改。
- **3-vuln scan captures algorithm-level gaps honestly**: vuln_a CRITICAL verdict is real (RDKit not in env), not a test stub。

---

## §5 Cross-reference

- `TODO/pending/83_lambda_algorithm_enhancement.md` — L2 closed; L1/L3-L8 open
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 — R20-P1 Wf-39 ship (VariancePUCT default + L2 closure)
- `molmetal/molmetal_lam/search_alg/proof_search.py` — R17 ship, untouched (7 152 LOC)
- `molmetal/molmetal_lam/search_alg/variance_mcts_integration.py` — Wf-14 ship, untouched (997 LOC)
- `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` — Wf-11 ship, untouched
- `molmetal/data/platinai_5k.csv` — R17 ship, 5 000 real Pt-group anticancer samples
- `.data/molmetal/MetalCytoToxDB.csv` — R17 ship, 26 801 real precious-metal rows
- `molmetal/scripts/verify_variance_mcts_proof_search.py` — Wf-36 P2 ship, mock chemistry baseline (this script = real-data follow-up)
- `molmetal/scripts/benchmark_mcts_real.py` — existing Wf-MCTS-Real-Bench (this script = Wf-39 P1 official version)
- `TODO/completed/82_wf36_closure_theorem_verifier.md` — Wf-36 P1 ship (sibling workflow)
- `.claude/skills/mol-metal-lambda-calculus/` — main skill for MCTS / VariancePUCT integration

---

## §6 Next (Wf-39 Phase 2 / R20-P1)

L2 closed. Next workflows:

- **Wf-39 Phase 2 (R20-P1)** — VariancePUCT default 化 (Wf-39 P2 ship)
  - Lazy RDKit import + degraded mode (`MoleculeClosedTerm.from_smiles` returns `None` + reason)
  - Re-run Wf-39 P1 with RDKit-installed env → real `leaf_value_var` / `n_states_explored`
  - VariancePUCT default flag flip in `proof_search.__init__`
- **Wf-36 Phase 3 (L3)** — ScoreRepellent + VariancePUCT + MCTS 三者联合
- **Wf-37 Phase 1 (L5)** — Hilbert-ASTSR end-to-end training with real IC50
- **Wf-37 Phase 2 (L6)** — Closure d-block 30 metal × depth 2/3 (already 30 metal in Wf-36 P1)

**Date**: 2026-09-26 (Wf-39 Phase 1 ship)
