# TODO-129: Cisplatin covalent docking follow-up (DNA N7 crosslink)

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P2** — 6-month R23 defer,非立即 action,但 paper §5 必须诚实标
**Parent**: TODO-127 (exhaustiveness) + TODO-128 (chemotype VDW) + cisplatin honest framing discussion 2026-09-27
**Skill alignment**: `party-twenty-spirit` (honest framing) + `mol-metal-anticancer` (cisplatin mechanism) + `mol-metal-wet-lab` (R23-P6)

---

## §0 Why this TODO exists

**根本问题**:cisplatin 在临床上 binding DNA 通过 covalent N7-guanine crosslink(经典 platinum chemotherapeutic mechanism)。**Vina 是 non-covalent 物理模型**,**看不到 covalent bond**。

后果:
- Vina cisplatin kcal/mol 永远 under-predict 临床真实 binding
- paper §3-§4 任何 cisplatin Vina 数字都不代表真 binding affinity
- 即使 Wf-134 (exhaustiveness=32) + Wf-128 (chemotype VDW) lift 到 −7,也**不代表 cisplatin 是好的 pocket binder**,因为 covalent mechanism 不在 Vina 模型内

**Anchor**:
- cisplatin IC50 in vitro(HeLa, A549, MCF-7):**1-10 μM**(细胞水平 真实 binding)
- cisplatin Vina kcal/mol: −3 ~ −7(**Vina 看不到 covalent**,数字无意义 for 真 binding)
- cisplatin DNA adduct:monoadduct → diadduct → crosslink(时间依赖),Vina 给的是 monoadduct 的 non-covalent estimation

## §1 Root cause 详解

### 1.1 临床 mechanism

cisplatin 体内机制(经典,Wang 2005 Nature Rev Cancer):
1. Cisplatin → aquation (loss of Cl⁻ in low-Cl⁻ cytoplasm) → [Pt(NH₃)₂(H₂O)(Cl)]⁺ or [Pt(NH₃)₂(H₂O)₂]²⁺
2. Aquated cisplatin → covalent binding to **N7 of guanine**(DNA major groove)
3. Monoadduct → diadduct(1,2-intrastrand crosslink, ~90%)→ cytotoxicity via DNA damage response
4. **Non-covalent pocket binding 不是 cisplatin 主机制**

### 1.2 Vina limitation

Vina score function:
```
score = gauss_1 + gauss_2 + repulsion + hydrophobic + hydrogen_bonding
```

**没有 reaction coordinate**,**没有 bond formation energy**,**没有 covalent docking capability**。

要 cisplatin 真 binding 评估,需要:
- CovDock(Schrödinger)— academic licensing 难
- DockCov(Forli 2016)— AutoDock Vina extension,相对可达
- GOLD covalent(CCDC,商业)— high-quality but licensing
- AutoDock4 + custom reaction coordinate — academic,可行

## §2 Solution plan — 3 phase 研究 + tooling + bench

### Phase 1: 文献 + tooling survey (2-3 天)

**Task**: 文献 mining covalent docking tools,评估可行性。

**Inputs**:
- Forli 2016 DockCov paper(J Comput Aided Mol Des 30:271)
- Scholz 2011 CovDock(J Chem Inf Model 51:2007)
-近期 review:Rashid 2015 Expert Opin Drug Discov 10:117

**Output**:
1. `molmetal/reports/wf136_covalent_dock/tooling_survey.md` (NEW, ≤ 400 lines)
   - 4-6 个 covalent docking tools 对比
   - License / cost / availability
   - Accuracy on standard test sets
   - 适配 cisplatin + Pt-group metals 的可行性
2. `molmetal/reports/wf136_covalent_dock/recommendation.md` (NEW, ≤ 200 lines)
   - 推荐工具(DockCov 最有可能 academic-friendly)
   - R23-P6 timeline 排期
3. `molmetal/reports/wf136_covalent_dock/tooling_table.json` (NEW)

**Verify**: tooling_survey.md exists; ≥ 3 tools compared; recommendation 明确

**Constraints**: 文献 + license research only,no actual docking

### Phase 2: cisplatin + DNA covalent benchmark (R23-P6, 6 月 defer)

**Task**: 用 DockCov 或选定工具,对 cisplatin-DNA N7 covalent binding 做 benchmark。

**Inputs**:
- Phase 1 推荐工具
- DNA target structure(PDB 1A2E,1AU5,B-form DNA duplex)
- Drug N7 guanine adduct reaction coordinate

**Output**:
1. `molmetal/scripts/cisplatin_dna_covalent_bench.py` (NEW, ~300 LOC)
   - DockCov wrapper for cisplatin-DNA
2. `molmetal/reports/wf136_p2_dna_bench/` (NEW) — IC50-style metric for covalent
3. `molmetal/tests/test_cisplatin_dna_covalent_bench.py` (NEW, ~12 tests)

**Verify**: pytest PASS (smoke only); real DNA-cisplatin IC50 MEASURED on DNA target

**Constraints**: R23-P6 6-month defer;诚实 framing — 非 paper v2 LEAD claim,只 paper §5 future work

### Phase 3: paper §5 wet-lab integration

**Task**: 与 wet-lab IC50 assay 整合(R23-P6 阶段)。

**Inputs**:
- TODO-97 (R20-P6 wet-lab IC50 plan) + TODO-116 (Wf-74 v2 close-out)
- Phase 2 covalent dock results

**Output**:
- paper §5 wet-lab integration section
- cisplatin + 5 metal cohort(HeLa / A549 / MCF-7 / A2780 / A2780cis) IC50
- 与 in-silico covalent dock 对照

**Verify**: paper §5 written + wet-lab IC50 MEASURED

**Constraints**: R23-P6,需要 ORNL / PKU / 协和 wet-lab 合作;$5M funding per LAMBDA_ROADMAP §5.2

## §3 Cross-reference

- **TODO-110** (meko metal atom extension) — Vina-only 解决,不能处理 covalent
- **TODO-127** (Vina exhaustiveness lift) — non-covalent Vina lift,与 covalent dock orthogonal
- **TODO-128** (Pt chemotype VDW) — non-covalent Vina lift,同样 orthogonal
- **TODO-97** (R20-P6 wet-lab IC50) — cisplatin 真 IC50 测量必须靠 wet-lab
- **LAMBDA_ROADMAP §5.3 R23-P6** — 6-month timeline 含 covalent dock research
- **Wf-74 v2 close-out** `TODO/completed/97c_wf74_reclassification.md` — wet-lab outreach 6 docs ship
- **party-twenty-spirit §3.5** — paper §5 必须诚实标"cisplatin Vina ≠ 真 binding"

## §4 Honest framing

### §4.1 承认

1. cisplatin Vina kcal/mol 永远 under-predict 真 binding,因为 Vina 物理模型根本看不到 covalent bond
2. 即使 Wf-134 + Wf-128 组合 lift 到 −7,也**不证明 cisplatin 是好的 pocket binder** — 它本来就是 covalent binder
3. cisplatin 真 binding 评估:**wet-lab IC50** (TODO-97 + R23-P6 wet-lab) **或** covalent docking (Phase 2)
4. paper §3 + §4 用 cisplatin Vina 数字 → 误导读者,**必须 paper §6 honest framing 标注**

### §4.2 不假装

1. 不假装 cisplatin Vina 数字代表 pocket binding 强度 — 那是 Vina 不能看的机制
2. 不假装 m3 Vina 主指标用 cisplatin — **paper §4 main 用 protein-binder pool**(organic drug-like,non-covalent),cisplatin 仅作 §5 wet-lab 验证
3. 不假装 R20 阶段可以做 covalent docking — Phase 2 是 R23-P6,6-month defer
4. 不假装 wet-lab IC50 不重要 — 它是 cisplatin 真 binding 评估的唯一金标准

### §4.3 Paper §6 + §5 必须写

- §6 limitations:cisplatin Vina ≠ 真 binding (covalent mechanism)
- §5 wet-lab:cisplatin + 5 metal cohort IC50 是 R23-P6
- §6 timeline:covalent docking (DockCov) 研究 是 R23 阶段

## §5 Action plan

| 优先级 | 行动 | 资源 | ETA |
|---:|---|---|---|
| **P2** | Phase 1 tooling survey | CPU | 2-3 天 |
| **P2** | Phase 2 cisplatin-DNA covalent dock | GPU-1 | R23-P6(6 月后) |
| **P2** | Phase 3 paper §5 + wet-lab | wet-lab 合作 | R23-P6 |
| **Total** | | | **6-12 月 R23 阶段** |

**第一步**:Phase 1 tooling survey,**立即可启动** — 文献 + license research 2-3 天出报告。
不依赖 R20 GPU 阶段,可与 TODO-127/128 并行。

**依赖关系**:
- TODO-127 (Wf-134 exhaustiveness) — **independent** — 立即跑
- TODO-128 (Wf-128 chemotype VDW) — **independent** — 立即跑
- TODO-129 (Phase 1 tooling) — **independent** — 立即跑
- TODO-129 (Phase 2 DNA bench) — depends on R23-P6 funding + wet-lab partner

## §6 Priority ranking

按 ROI 排序:

1. **TODO-127 / Wf-134** (exhaustiveness=32) — **P0,1-2 天,expected lift +1~+3 kcal/mol**
2. **TODO-128 / Wf-128** (chemotype VDW) — **P1,1-2 周,expected lift +0.5~+1.5 kcal/mol**
3. **TODO-129 Phase 1** (tooling survey) — **P2,2-3 天,no bench impact,paper §5 / §6 prep**
4. **TODO-129 Phase 2** (DNA covalent dock) — **R23-P6,6 月后**

**建议启动顺序**:
- Day 1-2: 启动 Wf-134 (exhaustiveness lift)
- Day 1-3: 启动 Wf-136 Phase 1 (covalent dock tooling survey)
- Week 1-2: 启动 Wf-128 (chemotype VDW literature mining + meko atype)
- Month 6+: R23-P6 启动 Wf-136 Phase 2 (cisplatin-DNA covalent bench)
