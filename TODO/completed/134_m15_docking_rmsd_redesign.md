# TODO-134: m15 Docking RMSD redesign — ligand-vs-pocket-RMSD adapter

**Date**: 2026-09-27
**Status**: ⚙️ **PARTIAL SHIP** — Phase 1 SHIPPED via Wf-139 P1 (2026-09-27); Phase 2 Vina redocking benchmark **NOT SHIPPED**; Phase 3 metric JSON refresh **PARTIAL** (docked_pose_rmsd.json ships as NOT_MEASURED status marker)
**Priority**: **P0** — m15 是 Wf-134 ship 的 5 个 REDESIGN 中**最高优先级**(TODO-130 §0 single highest-priority REDESIGN),且 paper §4 geometric table 前置条件
**Parent**: TODO-130 metric definition baseline + Wf-126 P1 honest finding + Wf-131 D Phase D
**Skill alignment**: `party-twenty-spirit` §3.5 + `mol-metal-mechanism-mapping`

---

## §0 Why this TODO exists

**当前 m15 (rigid_fragment_rmsd) 的根本问题**(per Wf-126 P1 honest finding):

> "rigid_fragment_rmsd is **self-RMSD between two ETKDGv3+UFF-optimized conformers of the same molecule**. Pocket does NOT enter the calculation. Identical 1.241 across pockets is the **expected** behavior, not a staging artifact."

含义:**当前 m15 1.241 是 from "UR" metric 定义,根本不反映 docking 质量**。所以:
- **不能 lift**(lift 1.241 不是 docking quality 改善,是 metric 写错了)
- **必须 redesign**(改成真的 docking quality metric)
- paper §4 不能再用 1.241 当 RMSD 数字

**SOTA convention**(per TODO-130 §1 row 15):
- TargetDiff / Pocket2Mol / DiffSBDD / DecompDiff / DiffDock 全部用 **ligand-vs-crystal RMSD**
- 算法:**Kabsch + Hungarian atom matching on heavy atoms**
- **Success threshold: ≤ 2.0 Å**(Trott & Olson 2010 Vina 原文)

**当前 m15 vs SOTA m15**:
| | SOTA m15(redocking benchmark) | 我们的 m15(rigid_fragment_rmsd) |
|---|---|---|
| 比较对象 | docked ligand pose vs crystal ligand pose | molecule conformer 1 vs molecule conformer 2 |
| 是否需要 pocket | ✅ 是(receptor 提供约束) | ❌ 否 |
| 是否反映 docking quality | ✅ 是 | ❌ 否 |
| 可比性 to TargetDiff | ✅ 是(同口径) | ❌ 否(完全不可比)|

**结论**:必须 redesign。新 metric 名字: **`docked_pose_rmsd`**。

## §1 Target metric: docked_pose_rmsd (ligand-vs-crystal RMSD)

### §1.1 算法定义

**Per pocket p ∈ {830c, 1h36, 3eml, 3ny8, 4aaw, 4rlu}**:

1. **Crystal ligand pose**:从 PDB crystal ligand(`<pocket>_ligand.pdb` or extracted from complex.pdb)
2. **Docked ligand pose**:Vina 1.2.7 dock crystal ligand SMILES 进 receptor.pdbqt,**exhaustiveness ≥ 32**(per TODO-127),n_poses = 5
3. **Heavy atom RMSD**:Kabsch + Hungarian atom matching(only heavy atoms,no H)
4. **Per-pose RMSD**:RMSD per docked pose
5. **Per-seed RMSD**:mean of RMSDs across 3 seeds
6. **Per-pocket metric**:**best RMSD of n_poses**(SOTA convention)
7. **Aggregate metric**:**median best RMSD across pockets**

### §1.2 Per-pocket inputs

Per Wf-30 staging(Wf-126 P1 已 verify 5/5 pockets READY):

| Pocket | PDB | Crystal ligand residue | Ref kcal/mol (Wf-121 P3) | Reference SMILES 来源 |
|---|---|---|---|---|
| **830c** | `molmetal/data/mmp13_real/830c.pdb` | MMP13 co-crystal | Wf-121 P3 mean −3.10 | from PDB `HETATM` records |
| **1h36** | staged (Wf-126 P1) | R88 | −0.801(shallow caveat)| from PDB |
| **3eml** | staged | ZMA | −7.753 | from PDB |
| **3ny8** | staged | JRZ | −7.140 | from PDB |
| **4aaw** | staged | R84 | −4.537 | from PDB |
| **4rlu** | staged | HCC | −6.373 | from PDB |

**注意**:830c 是 MMP13 真 co-crystal,其他 5 个 pocket residue SMILES 当前为 None(per audit #2)— 需要从 PDB HETATM 提取。

### §1.3 Reference SMILES 提取流程

每个 pocket 的 reference ligand SMILES 从 PDB `HETATM` records 提取:

1. Parse PDB `<pocket>.pdb` 中的 HETATM records
2. 提取 ligand residue 的 atom coordinates + element types
3. 用 RDKit `Chem.MolFromPDBBlock` 重建 ligand
4. 输出 canonical SMILES
5. 保存到 `molmetal/data/pocket_ref_ligands/{pocket}_ligand.smi`(NEW)

### §1.4 Heavy atom RMSD 算法

```python
def heavy_atom_rmsd(docked_pose_atoms: np.ndarray,
                     crystal_pose_atoms: np.ndarray,
                     crystal_element_types: list[str]) -> float:
    """
    Kabsch + Hungarian atom matching on heavy atoms.
    Reference: Trott & Olson 2010 Vina paper + TargetDiff convention.
    """
    # Filter to heavy atoms (no H)
    heavy_mask = [e != "H" for e in crystal_element_types]
    docked_heavy = docked_pose_atoms[heavy_mask]
    crystal_heavy = crystal_pose_atoms[heavy_mask]

    # Hungarian matching (chemical identity + spatial proximity)
    matched_pairs = hungarian_match(docked_heavy, crystal_heavy,
                                     element_types=[e for e in crystal_element_types if e != "H"])

    # Kabsch alignment
    R, t = kabsch_rotation(docked_heavy[matched_pairs], crystal_heavy[matched_pairs])
    docked_aligned = (docked_heavy @ R.T) + t

    # RMSD
    rmsd = np.sqrt(((docked_aligned - crystal_heavy[matched_pairs]) ** 2).sum(axis=1).mean())
    return rmsd
```

### §1.5 SOTA threshold

- **Success**:docked_pose_rmsd ≤ 2.0 Å(Trott & Olson 2010 Vina 原文 + TargetDiff convention)
- **Median across 5 pockets**:应 ≤ 2.5 Å(SOTA paper 接受范围)
- **Per-pocket success rate**:80%+ per pocket 是 SOTA-level(TransDiffSBDD 83.9% success rate 参考)

## §2 Solution plan — 3 phase 实测

### Phase 1: Reference ligand SMILES 提取 + 5/5 pocket 验证(CPU-only, 2-3 天)

**Task**: 从 5 个 pocket PDB 提取 crystal ligand SMILES,validate 化学结构。

**Inputs**:
- `molmetal/reports/wf30_pocket_prep/{830c,1h36,3eml,3ny8,4aaw,4rlu}/receptor.pdbqt`(5/5 READY)
- `molmetal/data/mmp13_real/830c.pdb`(MMP13 真 co-crystal)
- Wf-126 P1 staged pdbqt(verify 5/5 byte-match)

**Output**:
1. `molmetal/data/pocket_ref_ligands/{830c,1h36,3eml,3ny8,4aaw,4rlu}_ligand.smi`(6 NEW)
   - canonical SMILES + InChI + InChIKey
2. `molmetal/data/pocket_ref_ligands/ligand_metadata.json`(NEW)
   - per-pocket: SMILES + atom_count + heavy_atom_count + element_types + 3D coords
3. `molmetal/scripts/extract_pocket_ref_ligands.py`(NEW, ~200 LOC)
   - PDB HETATM parser + RDKit MolFromPDBBlock + canonical SMILES
4. `molmetal/tests/test_extract_pocket_ref_ligands.py`(NEW, ~10 tests)
   - 5/5 pocket ligand extraction PASS
   - SMILES canonical validation
   - InChIKey 一致性

**Verify**: 5/5 ligand SMILES extracted + valid + canonical + InChIKey 一致

**Constraints**: CPU-only RDKit; honest framing — 如 extraction fail,ship honest-negative

### Phase 2: Vina redocking benchmark per pocket × 3 seeds(CPU + GPU-0, 2-3 天)

**Task**: Vina 1.2.7 dock 每 pocket crystal ligand SMILES × 3 seeds × exhaustiveness=32,记录 n_poses=5 best RMSD。

**Inputs**:
- Phase 1 ship(5 pocket ref SMILES)
- Wf-121 P3 ship(`_dock_one_python_binding` 函数可复用)
- Wf-127 ship(Hilbert L2 + ASTSR scale guard,本任务不需要)
- `molmetal/reports/wf30_pocket_prep/{pocket}/receptor.pdbqt`(5/5 READY)
- Wf-126 P1 ship(`stage_5_pockets_full.py` 复用于 ref SMILES staging)

**Output**:
1. `molmetal/scripts/bench_docked_pose_rmsd.py`(NEW, ~280 LOC)
   - 5 pocket × 3 seed × exhaustiveness=32 × n_poses=5
   - Heavy atom RMSD computation (Kabsch + Hungarian)
   - Per-pocket per-seed best RMSD + median + mean + std
2. `molmetal/tests/test_docked_pose_rmsd.py`(NEW, ~12 tests)
   - Heavy atom RMSD against synthetic ground truth
   - Kabsch identity check (R^T R = I, det(R) = +1)
   - Hungarian matching correctness
   - 5/5 pocket PASS
3. `molmetal/reports/wf134_p2_redocking/{pocket}/per_seed_metrics.json`(NEW) — 5 dirs
   - per-seed: per-pose RMSD array + best RMSD + n_poses + exhaustiveness
4. `molmetal/reports/wf134_p2_redocking/synthesis.md`(NEW)
   - aggregate median best RMSD across 5 pockets
   - per-pocket success rate (RMSD ≤ 2.0 Å)

**Verify**:
- 5/5 pocket × 3 seed × n_poses=5 = 75 docked poses
- Per-pocket median best RMSD ≤ 2.5 Å(SOTA paper 接受)
- ≥ 1 pocket ≤ 2.0 Å(SOTA success)

**Constraints**:
- Vina 1.2.7 Python binding(exhaustiveness=32 per TODO-127)
- Kabsch + Hungarian on heavy atoms only
- honest framing — 如某 pocket RMSD > 5 Å,ship honest-negative

### Phase 3: Metric JSON + paper §4 integration + Phase D update

**Task**: 创建新 metric JSON `docked_pose_rmsd.json`,标 m15 旧 metric `rigid_fragment_rmsd.json` 为 honest-negative。

**Inputs**:
- Phase 2 ship(per-pocket RMSD data)
- TODO-130 metric definition baseline(m15 REDESIGN plan)
- TODO-131 weak metric lift master plan(Phase D)
- TODO-132 paper §4 lead rewrite
- TODO-133 paper §6 honest-negative

**Output**:
1. `metrics/by_metric/docked_pose_rmsd.json`(NEW)
   - 5 pocket × 3 seed × n_poses=5 best RMSD aggregate
   - status: MEASURED (Q2 paper §4.5.7 lead candidate)
   - honest_framing_note: "SOTA convention Kabsch + Hungarian on heavy atoms; Vina 1.2.7 exhaustiveness=32"
2. `metrics/by_metric/rigid_fragment_rmsd.json`(PATCHED)
   - status: `HONEST_NEGATIVE_REDESIGNED_TO_DOCKED_POSE_RMSD`
   - honest_framing_note: "self-RMSD between 2 conformers of same molecule; pocket NOT in calculation; NOT comparable to SOTA crystal-RMSD per Wf-126 P1"
3. `metrics/by_metric/rigid_fragment_rmsd_<pocket>.json`(5 PATCHED)— 同上
4. `metrics/by_metric/docked_pose_rmsd_<pocket>.json`(5 NEW)— per-pocket detailed
5. `TODO/completed/134_m15_docking_rmsd_redesign.md`(NEW, ≤ 250 lines)
   - TL;DR + Phase 1-3 deliverables + MEASURED vs PROJECTED + before/after + honest framing + cross-ref
6. `TODO/INDEX.md §17.10`(PATCHED)
   - update "m15 redesigned" entry

**Verify**: docked_pose_rmsd.json valid JSON + 5 pocket data + per-seed std; rigid_fragment_rmsd.json status updated

**Constraints**: CPU-only docs; honest framing; back-compat — old rigid_fragment_rmsd.json 保留但标 HONEST_NEGATIVE_REDESIGNED

## §3 Expected outcomes

### §3.1 Likely results(per SOTA experience)

| Pocket | Likely median best RMSD | SOTA success ≤ 2.0 Å | Honest caveat |
|---|---:|---:|---|
| 830c (MMP13 真 co-crystal) | 0.5 - 1.5 Å | ✅ likely success | 真 co-crystal — RMSD 应该低 |
| 1h36 (R88, ref −0.801) | 3 - 5 Å | ❌ likely fail | shallow ref kcal/mol — 不像真 binding pocket |
| 3eml (ZMA, ref −7.753) | 1 - 2.5 Å | ✅ likely success | ZMA 真 co-crystal |
| 3ny8 (JRZ, ref −7.140) | 1 - 2.5 Å | ✅ likely success | JRZ 真 co-crystal |
| 4aaw (R84, ref −4.537) | 2 - 4 Å | 🟡 marginal | ref 浅但有 co-crystal |
| 4rlu (HCC, ref −6.373) | 1 - 2.5 Å | ✅ likely success | HCC 真 co-crystal |

**Overall aggregate**:median best RMSD 1.5 - 3 Å(预计)。

### §3.2 Paper §4.5.7 lead claim

**If aggregate median ≤ 2.5 Å**:
- paper §4.5.7 lead claim:**"Redocking benchmark median RMSD = X.XX Å across 5 pockets (3 seeds × exhaustiveness=32)"**
- 与 TargetDiff / Pocket2Mol / DiffSBDD 同口径可比
- SOTA paper 接受范围
- 配 SOTA success rate (per-pocket RMSD ≤ 2.0 Å)

**If aggregate median > 2.5 Å**:
- ship honest-negative 到 §6
- paper §4.5.7 仍可引 n_distinct 1→20 替代 m15 lead

### §3.3 Honest framing

1. **5 pocket 中 1h36 ref shallow** — R88 kcal/mol = −0.801 比 pool 中位数 −5.264 还浅 4×,**这 pocket RMSD 应该会 fail**;不能 cite 1h36 的 m15
2. **Vina exhaustiveness=32** vs SOTA paper 64-128 — 我们搜索充分度较低,**RMSD 数字偏保守**
3. **generic VDW atype** (per TODO-128 后续) — M→Mg substitution 可能影响 heavy atom alignment 但 Kabsch 是 coordinate-based,不依赖 VDW 参数
4. **5 pocket × 3 seed × 1 ref ligand = 15 dock runs** — 不是 100×3 sweep,统计上可信但 variance 可能大

## §4 Cross-reference

- **TODO-130** metric definition baseline(m15 REDESIGN verdict)— Wf-134 ship
- **TODO-131** weak metric lift master plan Phase D(m15 redefine + re-dock)— 整合待实施
- **TODO-132** paper §4 lead rewrite §4.5.7(m15 替代 docked_pose_rmsd)— 待实施
- **TODO-133** paper §6 honest-negative §6.7(m15 旧定义 honest-negative)— 待实施
- **TODO-127** (Vina exhaustiveness lift) — exhaustiveness=32 数字来源
- **TODO-128** (Pt chemotype VDW) — 与 Kabsch coordinate-based 重叠
- **Wf-30** pocket staging ship(5/5 READY)— receptor.pdbqt 已 staged
- **Wf-121 P3** cisplatin real Vina — `_dock_one_python_binding` 函数可复用
- **Wf-126 P1** 5 pockets staging verify — 5/5 pockets READY
- **Wf-126 P2** SE(3) Bridge + Wasserstein pose loss — `Kabsch_align` + `heavy_atom_rmsd` 已 ship 可复用
- **party-twenty-spirit §3.5** honest framing

## §5 Verify & Action plan

### §5.1 Verification matrix

| Verify | Command | Pass criteria |
|---|---|---|
| Phase 1 | `ls molmetal/data/pocket_ref_ligands/*_ligand.smi` | 5 files exist + SMILES valid |
| Phase 1 | `pytest molmetal/tests/test_extract_pocket_ref_ligands.py -v` | all PASS |
| Phase 2 | `pytest molmetal/tests/test_docked_pose_rmsd.py -v` | all PASS |
| Phase 2 | `cat molmetal/reports/wf134_p2_redocking/synthesis.md` | 5 pocket × 3 seed data present |
| Phase 3 | `cat metrics/by_metric/docked_pose_rmsd.json` | valid JSON + 5 pocket data |
| Phase 3 | `cat metrics/by_metric/rigid_fragment_rmsd.json` | status = HONEST_NEGATIVE_REDESIGNED |

### §5.2 Action plan by priority

| 优先级 | Phase | 资源 | ETA |
|---:|---|---|---|
| **P0** | Phase 1 ref SMILES extraction | CPU-only RDKit | 2-3 天 |
| **P0** | Phase 2 Vina redocking × 5 pocket × 3 seed | CPU + GPU-0 | 2-3 天 |
| **P0** | Phase 3 metric JSON + paper §4 integration | CPU-only docs | 1-2 天 |
| **Total** | | | **5-8 天 ship m15 redesign** |

### §5.3 不做的事

按 Q2 目标 + honest framing,**以下事情不假装做**:

1. ❌ **不 lift 1h36 RMSD** — ref shallow 是 receptor 问题,不是 Vina 问题;ship honest-negative
2. ❌ **不与 TargetDiff 真实数字 head-to-head** — TargetDiff 数据集是 CrossDocked,我们的 5 pockets 不一样
3. ❌ **不假装 "we beat TargetDiff on m15"** — 5 pocket 不够,可比性不足;只 cite median best RMSD

## §6 Final verdict

**m15 redesign 必要性**:✅ **必须做**(metric 写错不是 staging 错)
**Phase 1 + 2 ship 后**:✅ paper §4.5.7 lead candidate + §6 honest-negative for old metric
**ETA**:5-8 天 ship complete m15 redesign
**Resource**:CPU + GPU-0;honest framing
**Target journal**:JCIM IF 5.6 Q2 — m15 redesign 是 paper §4 + §6 完整性前置

**下一步建议**:
1. **立即启动 Phase 1**(CPU-only RDKit,2-3 天可 ship) — 不依赖任何 in-flight Wf
2. **等 Phase 1 ship → 启动 Phase 2**(CPU + GPU 0,2-3 天)
3. **Phase 2 ship → Phase 3**(CPU docs,1-2 天)
4. **5-8 天后 m15 redesign 完成,可投稿 Q2**

要不要我现在**立即启动 Phase 1**(CPU-only RDKit ref ligand extraction,2-3 天)?