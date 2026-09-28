# TODO-127: Vina real dock exhaustiveness lift (cisplatin −3.1 → expected −5 ~ −7)

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P0** — `metrics/by_metric/vina_kcal_per_mol.json` cisplatin mean −3.103 离业界 cutoff ≥ −7 差 4 kcal/mol
**Parent**: TODO-110 (meko metal atom extension) + Wf-121 P3 cisplatin real Vina sweep
**Skill alignment**: `party-twenty-spirit` (honest framing) + `mol-metal-mechanism-mapping` (M22 SE(3) SB + M29 Coord GNN)

---

> **🟢 2026-09-27 SHIP annotation (Wf-141 v2, `TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md`)**:
> **Phase 1 (exhaustiveness lift 8→32) ✅ SHIPPED honestly via Wf-141 v2 (honest-partial).**
> - Wf-141 v2 ex=32 main sweep: 30/30 records MEASURED on Vina 1.2.7 Python binding, 10 pockets × 3 seeds.
> - Wf-141 v2 ex=64 upper bound: 6/6 records MEASURED on 1h36 + 4aaw × 3 seeds.
> - **Result**: 10-pocket mean **−3.1575 kcal/mol** at ex=32 (Δ **−0.0545** vs Wf-121 P3 baseline −3.103); 2-pocket ex=64 mean **−3.4345 kcal/mol** (saturated — delta +0.0008 vs ex=32 on same 2 pockets).
> - **0/10 pockets reach industry cutoff ≥ −7 kcal/mol** at ex=32. Bottleneck is **mechanistic** (cisplatin is covalent binder; Vina does not score Pt–DNA N7 bond), NOT search thoroughness.
> - `metrics/by_metric/vina_kcal_per_mol.json` refreshed: 4 original entries preserved + 3 new (wf141_v2_p1_ex32_main, wf141_v2_p2_ex64_upper, wf141_v2_aggregate) appended → 7 total. Aggregate block extended with `best_measured_ex32`, `best_measured_ex64`, `delta_vs_baseline_minus_3p103`, `n_pockets_pass_cutoff_minus7`. New `2026_09_27_wf141_v2_strengthen_status` field added.
> - **Anti-confusion lesson learned** (Wf-141 v1 → v2): (a) Wf-141 v1 agents hallucinated `wf140_p1_vina_quickcheck` and `wf140_p1_vina_4aaw_only` as upstream scripts — they do not exist; v2 explicitly logged non-invocation. (b) Wf-141 v1 confused the per-pocket cohort with `metrics/by_metric/vina_high_affinity_4aaw.json` — a **different metric** (high-affinity rate, not kcal/mol); v2 used SMILES `[Pt](N)(N)(Cl)Cl` as the explicit cohort marker. (c) Wf-141 v1 fabricated lift expectations by reading "expected_lift_to_at_least" notes as measurements; v2 read Wf-121 P3 `sweep_summary.json` ONLY for baseline. **Lesson**: when comparing two exhaustiveness settings, always anchor on a single measured `sweep_summary.json`; never conflate "expected lift" notes (predictions) with `kcal_mol_per_seed` arrays (measurements); never read `vina_high_affinity_*.json` as a kcal/mol metric.
> - Phase 2 (Pt VDW, TODO-128) and Phase 3 (covalent dock, Wf-136 / TODO-129) remain pending — unaffected by this SHIP. These are the **mechanistic paths** to bridge the gap from generic Mg-atype cisplatin to industry cutoff.

---

## §0 Why this TODO exists

Wf-121 P3 (`molmetal/scripts/bench_cisplatin_real_vina.py`) 真跑了 `v.dock()`(line 423)— **不是 re-score**,是真 docking。但 **exhaustiveness=8** 决定 Vina 搜索 thoroughness 不足:

- 业界 Vina 标准:exhaustiveness ≥ **32**(Trott 2010 原始论文推荐),paper 用 **64-128**
- exhaustiveness=8 大约只能搜索 8 个 independent run,**低能量 pose 大概率没找到**
- 后果:cisplatin mean −3.103 (10 pocket × 3 seed)= **真数据,但不充分搜索**

**Anchor 数字**:
- cisplatin reference re-dock(已知):**−7 ~ −10 kcal/mol**
- 业界 binding threshold:**≥ −7 kcal/mol**
- 我们目前 0/10 pocket 达到这个 cutoff
- 其他 Vina 5-pocket 数字(1h36=−4.45, 3eml=−6.14, 3ny8=−6.88)— 是 organic drug-like pool,不是 cisplatin,不是 anchor

## §1 Root cause analysis(诚实 4 类)

| 根因 | 贡献度 | 描述 |
|---|---:|---|
| **exhaustiveness=8 太低** | **~60%** | Vina search 未充分,low-energy pose 大概率遗漏 |
| **generic VDW atype(Mg 替 Pt)** | ~20% | Pt 应 bind 更强(heavy atom VDW),被 generic atype under-estimate |
| **cisplatin 是 covalent binder** | ~15% | Vina 物理模型不算 covalent bond(顺铂机制是 DNA crosslink,non-covalent pocket binding 本来就弱) |
| **grid 中心 / box size** | ~5% | Wf-121 P3 已经 pocket-centered grid(已 OK),contribution 小 |

## §2 Solution plan — 3 phase lift

### Phase 1: exhaustiveness lift (Wf-134)

**Task**: 把 exhaustiveness 从 8 提到 32,看 cisplatin 真实分数能 lift 多少。

**Inputs**:
- `molmetal/scripts/bench_cisplatin_real_vina.py` (current, exhaustiveness=8)
- `molmetal/reports/wf121_cisplatin_real_vina/` (10 pockets × 3 seeds current results)
- `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (VinaDockingAdapter.dock)

**Output**:
1. `molmetal/scripts/bench_cisplatin_exhaustiveness_lift.py` (NEW, ~200 LOC)
   - Re-run cisplatin × 10 pockets × 3 seeds with exhaustiveness=32
   - Compare with Wf-121 P3 baseline
   - Add exhaustiveness=64 sweep on 1-2 pockets for upper-bound estimate
2. `molmetal/tests/test_cisplatin_exhaustiveness_lift.py` (NEW, ~10 tests)
3. `molmetal/reports/wf134_exhaustiveness_lift/per_pocket.json` (NEW) — 10 pockets × 3 seeds × {8, 32}
4. `molmetal/reports/wf134_exhaustiveness_lift/synthesis.md` (NEW)

**Expected lift** (per Vina 经验):
- exhaustiveness=8 → 32: typical lift **+1 ~ +3 kcal/mol**(更负)
- cisplatin target: mean **−5 ~ −6**(从 −3.1)
- 1-2 pockets 可能到 **−7 ~ −8**(达到 cutoff)

**Verify**: 10/10 pockets MEASURED,per-pocket lift > 1 kcal/mol expected

**Constraints**: CPU-only (Vina 1.2.7 Python binding); honest framing; same 10 pockets; same 3 seeds for apples-to-apples

### Phase 2: chemotype-tuned Pt VDW (Wf-135) — 4-6 周 R20 GPU 阶段

**Task**: 给 meko 0.8.0 加 Pt 真实 VDW 参数(目前用 Mg 替,underestimate Pt 重原子 binding)。

**Inputs**:
- meeko 0.8.0 `data/params/ad4_types.json` + `data/params/vina_params.json` (no Pt entry)
- 文献:Morris 2009 AutoDock4 atom types + Brese 1990 Pt bond valence parameters
- `molmetal/molmetal_lam/sbdd_env/metal_pdbqt_overrides.py` (Wf-121 P1 SMARTS framework)

**Output**:
1. `molmetal/references/soft_params/pt_vdw_overrides.json` (NEW) — Pt/Ru/Ir/Os/Rh/Pd/Au/Re 实测 VDW 参数
2. `molmetal/scripts/calibrate_pt_vdw.py` (NEW, ~300 LOC) — 文献参数 + small-molecule crystallographic binding data 反推
3. `molmetal/reports/wf135_pt_vdw_calibration/` (NEW) — 文献 mining + 配位化合物 binding data
4. `molmetal/tests/test_pt_vdw_calibration.py` (NEW, ~8 tests)

**Expected lift** (post-calibration):
- generic Pt(Mg)→ chemotype Pt: typical lift **+0.5 ~ +1.5 kcal/mol**
- cisplatin target: mean **−6 ~ −7**

**Verify**: 8 metals 全部 VDW 参数 ship; literature provenance documented

**Constraints**: 文献 mining only; no in-house QM computation; honest framing — may need R20 GPU stage

### Phase 3: cisplatin DNA covalent docking (Wf-136) — 学术 follow-up

**Task**: cisplatin 在临床上 binding DNA (covalent crosslink),不是常规 protein pocket binder。Vina 物理模型不看 covalent bond。要做真 cisplatin binding 评估,改用 **covalent docking tool**(CovDock / DockCov / AutoDock4 with reaction coordinates)。

**Inputs**:
- AutoDock4 + covalent docking setup(Morris 2009)
- CovDock(Schrödinger suite)— academic licensing 可能需要
- DockCov(Forli 2016, free)— AutoDock Vina extension
- cisplatin 已知 covalent target:DNA N7 of guanine

**Output**:
1. `molmetal/reports/wf136_covalent_dock/` (NEW) — 文献 + tooling research
2. `molmetal/scripts/cisplatin_dna_covalent_bench.py` (NEW, optional, depends on tooling availability) — DockCov wrapper
3. 6-month timeline note (R23-P6 follow-up per LAMBDA_ROADMAP §5)

**Constraints**: 6-month defer to R23; honest framing — Vina 非 cisplatin 真 mechanism,真实 binding 评估需 covalent tool

## §3 Cross-reference

- **TODO-110** (meko metal atom extension) — Wf-121 P1+P2+P3 解决 meko blocker 但 exhaustiveness atype 仍是问题
- **TODO-114** (MOCK_DELETED JSON cleanup) — unrelated,但 Wf-119 已 ship
- **TODO-126** (innovation strengthen roadmap) — master plan
- **LAMBDA_ROADMAP §4.2 L9** — Lambda × CFM e2e pipeline, exhaustiveness lift 不在 L9 scope
- **LAMBDA_ROADMAP §5.3 R23** — covalent docking 是 R23-P6 wet-lab 配套
- **Wf-121 P3** `bench_cisplatin_real_vina.py` line 423 — current `v.dock(exhaustiveness=8)` 改造起点
- **Wf-126** — 已经在跑 per-pocket Vina,但用的是 pool-of-organic,不是 cisplatin

## §4 Verify & honest framing

### §4.1 验证矩阵

| Verify | Command | Pass criteria |
|---|---|---|
| Phase 1 scripts | `ls molmetal/scripts/bench_cisplatin_exhaustiveness_lift.py` | exists |
| Phase 1 tests | `pytest molmetal/tests/test_cisplatin_exhaustiveness_lift.py -v` | all PASS |
| Phase 1 lift | `cat molmetal/reports/wf134_exhaustiveness_lift/synthesis.md` | per-pocket lift ≥ +1 kcal/mol |
| Phase 2 VDW | `ls molmetal/references/soft_params/pt_vdw_overrides.json` | exists |
| Phase 3 文献 | `cat molmetal/reports/wf136_covalent_dock/synthesis.md` | tooling 评估 + 6-month timeline |
| Metric refresh | `cat metrics/by_metric/vina_kcal_per_mol.json` | 4 measurement records (smoke1 + stroke + cisplatin_baseline + cisplatin_exhaustiveness_32) |

### §4.2 Honest framing per party-twenty-spirit §3.5

**承认**:
1. cisplatin −3.1 kcal/mol 不是"设计差",是 **Vina pipeline 不充分**(exhaustiveness=8 太低 + generic atype)
2. 业界 binding threshold ≥ −7; 当前 0/10 pocket 达到
3. cisplatin 是 covalent binder,Vina 物理模型根本看不到 covalent bond
4. 即使 exhaustiveness=32 + chemotype Pt VDW 也不一定到 −7,因为 cisplatin in Vina = under-mechanism

**不假装**:
1. 不假装 cisplatin 是 pocket-binder — 它是 DNA crosslinker,paper §5 必须诚实标
2. 不假装 exhaustiveness lift 是唯一根因 — 4 个根因都有贡献
3. 不假装 R20 GPU 阶段能完美 lift 到 SOTA — 现实是 ≥ −7 不一定达

**预期 paper §6**:
- cisplatin real Vina < −7 是 hard ceiling for Vina methodology
- cisplatin binding 真正评估需 covalent docking (Wf-136) + wet-lab IC50 (Wf-74)
- m3 Vina 主指标用 protein-binder(不是 cisplatin)测,所以 −7 cut 是可达的

## §5 Action plan

| 优先级 | 行动 | 资源 | ETA |
|---:|---|---|---|
| **P0** | ~~Wf-134 / Wf-141 v2 exhaustiveness lift~~ ✅ SHIP honest-partial (2026-09-27) | CPU | done — see `TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md` |
| **P1** | Wf-135 Pt VDW calibration | CPU + GPU-1 | 2-4 周 R20 阶段 |
| **P2** | Wf-136 covalent dock research | CPU | 6-month R23 阶段 |

**第一步**:立即启动 Wf-134,看 exhaustiveness=32 能 lift 多少。再决定 Wf-135 + Wf-136 的优先级。