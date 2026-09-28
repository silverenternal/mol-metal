# TODO-136 — m5 SA + m14 decode + m15 RMSD residual lift close-out (PARTIAL 2/3)

**Date**: 2026-09-27
**Status**: ⚙️ **PARTIAL — 2/3 closed** (m5 SA + m14 real CFM SHIPPED; m15 redesign remains PENDING per TODO-134)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Wf-124 ship (decode 1.000) + Wf-126 ship (5/5 pockets + RMSD self-RMSD honest) + TODO-131 Phase C/D
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing) + `mol-metal-mechanism-mapping`

---

## §0 TL;DR

| Metric | TODO-136 status (2026-09-27) | Ship evidence | Close verdict |
|---|---|---|---|
| **m5 SA mean** | ✅ **SHIPPED** | `metrics/by_metric/sa_mean.json` 12-measurement entry `wf139_p6_sa_closure_bias_generation` (5/5 pockets PASS target ≤3.0 AND stretch ≤2.8; 4/5 PASS TargetDiff 2.65 ceiling) | CLOSED |
| **m14 decode_ratio real CFM** | ✅ **SHIPPED** | `metrics/by_metric/decode_ratio.json` entries `wf139_p5_cfm_decode_real` (PlatinAI 0.9745) + `wf148_m14_real_cfm` (NCI-60+DrugOOD 0.9980; report at `molmetal/reports/wf139_p5_cfm_decode_real/` with Wf-148 follow-up GPU-0 Blackwell) | CLOSED |
| **m15 docked_pose_rmsd redesign** | ❌ **NOT SHIPPED** | TODO-134 still in `TODO/pending/134_m15_docking_rmsd_redesign.md`; `metrics/by_metric/docked_pose_rmsd.json` does NOT exist; `metrics/by_metric/rigid_fragment_rmsd.json` is still using self-RMSD definition (NOT ligand-vs-pocket) | **OPEN — TODO-134 P0 follow-up required** |

**Aggregate verdict**: ⚙️ **PARTIAL — 2 of 3 residual gaps closed.**

The task brief from the workflow harness stated "m15 redesign Phase 2 SHIPPED via Wf-141 ex32_lift". **This is factually INCORRECT based on filesystem evidence.** `Wf-141 ex32_lift` (`TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md`) is the **Vina exhaustiveness lift for cisplatin kcal/mol (m3 metric)** — NOT the m15 docking RMSD redesign. TODO-134 remains pending and the `docked_pose_rmsd.json` metric file does not exist. Per `party-twenty-spirit` §3.5, this close-out explicitly calls out the gap rather than falsely confirming m15.

---

## §1 m5 SA mean lift — ✅ SHIPPED via Wf-139 P6 + Wf-140 P1

### §1.1 Ship evidence

**Wf-140 P1** (the 5-pocket extension that closed the original gap):
- `molmetal/reports/wf140_p1_sa_5pocket/{1h36,3eml,3ny8,4aaw,4rlu}/` — 5 pocket dirs with per_seed_metrics.json + cohort_smiles.jsonl + synthesis.md + (for 3 new) sa_metrics.json + per_compound_sa.csv + sa_run_summary.json
- `molmetal/reports/wf140_p1_sa_5pocket/sa_qed_lipinski.json` — cross-pocket cross-tab
- `metrics/by_metric/sa_mean.json` — REFRESHED (11 measurements, 5-pocket aggregate added, `2026_09_27_wf140_strengthen_status` field)
- `TODO/completed/140_m5_sa_5pocket_extension.md` — close-out doc

**Wf-139 P6** (the 3-seed re-validation + QED/Lipinski/AiZynth cross-tab):
- `molmetal/reports/wf139_p6_sa_closure_bias/{sa_metrics.json, per_compound_sa.csv, cohort_smiles.jsonl, sa_run_summary.json, synthesis.md}`
- `metrics/by_metric/sa_mean.json` — REFRESHED (12 measurements, latest entry `wf139_p6_sa_closure_bias_generation`)
- `TODO/completed/142_wf139_p6_sa_closure_bias_generation.md` — close-out doc

### §1.2 MEASURED aggregate

**5-pocket aggregate (closure-bounded, n=215 total emissions across Wf-140 P1)**:
- Weighted mean = **2.479** (vs unconstrained baseline 3.099, delta −0.620)
- Simple mean of 5 per-pocket means = **2.449**
- Std of 5 per-pocket means = 0.138

**Per-pocket (all PASS target ≤3.0 AND stretch ≤2.8)**:
- 1h36 = 2.373 (n=51, catherine_pass_rate=1.0)
- 3eml = 2.465 (n=54, catherine_pass_rate=1.0)
- 3ny8 = 2.660 (n=54, catherine_pass_rate=1.0)
- **4aaw = 2.288 (n=14) — additionally PASSES TargetDiff 2.65 ceiling (gap +0.362)**
- 4rlu = 2.456 (n=42, catherine_pass_rate=0.875)

**Wf-139 P6 re-run** (3 seeds × closure-BFS, 231 emissions, 57 unique SMILES):
- Aggregate SA mean = **2.463** (simple) / 2.458 (weighted, n=231) — PASS ≤2.8
- 5/5 pockets PASS target ≤3.0; 5/5 pass stretch ≤2.8; 4/5 pass TargetDiff 2.65 ceiling
- RDKit validity = 100%, Lipinski Ro5 pass = 100%, QED 0.50-0.68, AiZynth synth 0.29-0.77

### §1.3 Single surgical change

Added 2 click-handle sulfonamide seeds (4-azido-sulfanilamide + 4-ethynyl-sulfanilamide) to `molmetal/molmetal_lam/reward/pocket_closure_bias.py:_pocket_seed_smiles['4aaw']`. Lifted 4aaw closure-BFS reachable space from 5 → 14 unique catherine-pass candidates. Other 4 pockets UNCHANGED.

### §1.4 Honest framing per `party-twenty-spirit` §3.5

**承认**:
1. 5/5 PASS is NOT a universal SA fix — pocket-specific closure-bounded cohort re-definition
2. 3ny8 still sits +0.010 above TargetDiff 2.65 — kinase hinge pyrimidine pharmacophore ceiling
3. n=14-42 per pocket is statistically thin — adequate for threshold PASS, not for sub-0.1 mean-diff
4. SA score definition (Ertl 2008) UNCHANGED — lift is purely cohort re-definition

**不假装**:
1. We do NOT claim a sub-TargetDiff SA across all pockets — 3ny8 misses by +0.010
2. We do NOT claim the closure-bounded lift generalises to the open unconstrained cohort
3. We do NOT claim this is a methodology paper — it is a single-cohort re-definition per pocket

**Verdict**: **m5 SA MEASURED at 5/5 pockets PASS** — Q2 paper §4.5.6 partial lead candidate, with §6 honest caveat preserved.

---

## §2 m14 decode_ratio real CFM — ✅ SHIPPED via Wf-139 P5 (+ Wf-148 follow-up)

### §2.1 Ship evidence (Wf-139 P5 — PlatinAI CPU)

**Date**: 2026-09-27
**Owner**: Hugo Lee + ultracode agents
**Close-out**: `TODO/completed/139_wf139_p5_cfm_decode_real.md`
**Report**: `molmetal/reports/wf139_p5_cfm_decode_real/{per_seed_metrics.json, synthesis.md}`
**Metric JSON**: `metrics/by_metric/decode_ratio.json` (entry `wf139_p5_cfm_decode_real`)

### §2.2 MEASURED results (Wf-139 P5)

| Statistic | Value |
|---|---:|
| Synthetic baseline | 1.000 |
| Real (PlatinAI 5K Pt subset, 3-seed) | **0.9745** (267/274) |
| Gap | −0.0255 absolute (~2.5pp) |
| Verdict | PARTIAL — expected per honest framing |

Per-seed:
- seed 0: 82/85 = 0.9647
- seed 1: 92/94 = 0.9787
- seed 2: 93/95 = 0.9789
- mean: 0.9745, std across seeds: 0.0082 (consistent; gate PASS)

### §2.3 Wf-148 follow-up (NCI-60 + DrugOOD, GPU-0 Blackwell)

**Date**: 2026-09-27 (supersedes PlatinAI measurement per `metrics/by_metric/decode_ratio.json` aggregate block)
**Data**: NCI-60 9-cell (n=7823) + DrugOOD native IC50 subsample (n=1000), subsampled to 2000×3=6000 (5991 successfully embedded)
**Device**: cuda:0 (Blackwell, 96GB)
**Report**: `molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md` (labelled Wf-148), log at `molmetal/reports/wf148_m14_real_cfm.log`

| Statistic | Value |
|---|---:|
| Synthetic baseline | 1.000 |
| Real (NCI-60 + DrugOOD, 3-seed GPU-0) | **0.9980** (5979/5991) |
| Gap | −0.0020 absolute (~0.2pp) |
| Verdict | PASS — essentially at synthetic parity |

### §2.4 Honest framing per `party-twenty-spirit` §3.5

**承认**:
1. Real-CFM (0.9745 PlatinAI / 0.9980 NCI-60+DrugOOD) < synthetic (1.000) by 0.20–2.55pp
2. Synthetic Pt-click clouds are idealised (clean 5-atom cisplatin motifs with known topology); real PlatinAI/NCI-60/DrugOOD ligands have variable atom counts, mixed metals, and noisy 3D embeddings
3. SOTA papers (TargetDiff, DecompDiff, etc.) all report the same real-vs-synthetic gap

**不假装**:
1. We do NOT claim real >= synthetic — we MEASURE and report honestly
2. We do NOT claim Wf-139 P5 closes the m14 lead alone — it is one architectural data point
3. We do NOT claim NCI-60+DrugOOD is head-to-head with TargetDiff's CrossDocked — different cohort, different decoder architecture

**Paper §6 honest gap** (per TODO-133):
- m14 real CFM decode_ratio = 0.9980 (NCI-60+DrugOOD) / 0.9745 (PlatinAI) vs synthetic 1.000 = 0.20pp / 2.55pp gap
- Synthetic = upper bound; real is the canonical claim; both are strong (close to ceiling)

**Verdict**: **m14 decode MEASURED real CFM** — Q2 paper §4.5.7 lead candidate (real CFM).

---

## §3 m15 docked_pose_rmsd redesign — ❌ NOT SHIPPED (TODO-134 still PENDING)

### §3.1 Status as of 2026-09-27

**Filesystem evidence**:
- `TODO/pending/134_m15_docking_rmsd_redesign.md` — **STILL PENDING** (P0 status)
- `metrics/by_metric/docked_pose_rmsd.json` — **DOES NOT EXIST**
- `metrics/by_metric/rigid_fragment_rmsd.json` — **STILL USING self-RMSD definition** (mean 0.428 Å for 10 organic drug-like SMILES; per-pocket cisplatin values 6.1117 Å — same uniform value across pockets per Wf-126 P1 honest finding)
- No `wf134_p2_redocking/` report directory exists
- No `bench_docked_pose_rmsd.py` script exists

### §3.2 Why the workflow harness brief was incorrect

The task brief from the workflow harness stated:
> "m15 redesign Phase 2 SHIPPED via Wf-141 ex32_lift"

This is **factually incorrect**. `Wf-141 ex32_lift` (`TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md`) is the **Vina exhaustiveness lift for cisplatin kcal/mol** — a Phase 1 follow-up of TODO-127 (cisplatin real Vina kcal/mol). It addresses the **m3 Vina metric** (kcal/mol of cisplatin docking), not the **m15 RMSD metric** (ligand-vs-pocket RMSD after Kabsch + Hungarian heavy-atom alignment).

Per Wf-141 v2 P1 results: 10-pocket cisplatin mean ex=32 = **−3.1575 kcal/mol** (Δ −0.0545 vs ex=8 baseline −3.103). This is a **search-thoroughness lift** for cisplatin kcal/mol scoring — it does NOT measure docked pose quality vs crystal pose. The 0/10 pockets reaching industry cutoff ≥ −7 kcal/mol is explicitly reported in `TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md` §3.1.

### §3.3 What TODO-134 actually requires to ship

Per `TODO/pending/134_m15_docking_rmsd_redesign.md` §2 (3-phase plan, 5-8 day ETA):

**Phase 1**: Reference ligand SMILES extraction (CPU-only RDKit, 2-3 days)
- Output: `molmetal/data/pocket_ref_ligands/{830c,1h36,3eml,3ny8,4aaw,4rlu}_ligand.smi` (6 NEW)
- Script: `molmetal/scripts/extract_pocket_ref_ligands.py` (~200 LOC)
- Tests: `molmetal/tests/test_extract_pocket_ref_ligands.py` (~10 tests)

**Phase 2**: Vina redocking benchmark per pocket × 3 seeds (CPU + GPU-0, 2-3 days)
- Output: `molmetal/scripts/bench_docked_pose_rmsd.py` (~280 LOC), `molmetal/reports/wf134_p2_redocking/{pocket}/per_seed_metrics.json` (5 dirs)
- Verify: 5/5 pocket × 3 seed × n_poses=5 = 75 docked poses; per-pocket median best RMSD ≤ 2.5 Å; ≥ 1 pocket ≤ 2.0 Å

**Phase 3**: Metric JSON + paper §4 integration + Phase D update (CPU-only docs, 1-2 days)
- Output: `metrics/by_metric/docked_pose_rmsd.json` (NEW), `metrics/by_metric/rigid_fragment_rmsd.json` (PATCHED to `HONEST_NEGATIVE_REDESIGNED_TO_DOCKED_POSE_RMSD`)
- Verify: docked_pose_rmsd.json valid JSON + 5 pocket data; rigid_fragment_rmsd.json status updated
- Close-out: `TODO/completed/134_m15_docking_rmsd_redesign.md`

**None of these 3 phases have shipped**. Per the P0 status in TODO-134, m15 redesign is the **highest-priority single REDESIGN** for paper §4 geometric table (per `TODO/completed/130_metric_definition_baseline.md` §2.1).

### §3.4 Honest framing per `party-twenty-spirit` §3.5

**承认**:
1. m15 redesign is **NOT shipped** as of 2026-09-27 — TODO-134 P0 pending
2. The current `rigid_fragment_rmsd` metric is self-RMSD between 2 conformers of the same molecule — NOT pocket-aware
3. Paper §4.5.7 geometric table lead candidate (target RMSD ≤ 2.5 Å) is **NOT available** until TODO-134 ships

**不假装**:
1. We do NOT claim m15 redesign shipped via Wf-141 (it did not — Wf-141 is the m3 Vina exhaustiveness lift)
2. We do NOT claim `rigid_fragment_rmsd` is comparable to TargetDiff/Pocket2Mol/DiffSBDD crystal-RMSD
3. We do NOT close TODO-136 as fully shipped

**Paper §6 honest gap** (per TODO-133):
- m15 RMSD redesign is OPEN — paper §4.5.7 geometric table lead candidate deferred until TODO-134 ships
- Old m15 self-RMSD disclosure (MANDATORY): every paper mention of "rigid_fragment_rmsd" or "RMSD" in §4/§6 MUST include the Wf-126 P1 honest-framing footnote that the metric is self-RMSD between two conformers of the same molecule, NOT pocket-aware

**Verdict**: **m15 redesign OPEN — TODO-134 P0 follow-up required** (5-8 day ETA per TODO-134 §5.2).

---

## §4 Cross-reference

- **Wf-124** decode_ratio 1.000 synthetic ship — TODO-136 §2 接力 real CFM (SHIPPED via Wf-139 P5 + Wf-148)
- **Wf-126** 5/5 pockets READY + RMSD self-RMSD honest — TODO-136 §3 接力 redesign (NOT YET SHIPPED)
- **Wf-130** Catherine 30/30 + L6 0.5589 honest-partial — TODO-136 §1 closure-bounded generation (SHIPPED via Wf-140 P1 + Wf-139 P6)
- **Wf-139 P5** m14 real CFM decode (PlatinAI 0.9745) — `TODO/completed/139_wf139_p5_cfm_decode_real.md`
- **Wf-139 P6** m5 SA closure-bias (5/5 PASS) — `TODO/completed/142_wf139_p6_sa_closure_bias_generation.md`
- **Wf-140 P1** m5 SA 5-pocket extension — `TODO/completed/140_m5_sa_5pocket_extension.md`
- **Wf-141 v2** m3 Vina exhaustiveness 8→32 (NOT m15 redesign) — `TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md`
- **Wf-148** m14 NCI-60+DrugOOD follow-up (real 0.9980) — report dir `molmetal/reports/wf139_p5_cfm_decode_real/` (Wf-148 used the same dir)
- **TODO-131** Phase C decode_ratio YuelBond — DONE; Phase 2 real CFM (SHIPPED via Wf-139 P5 + Wf-148)
- **TODO-131** Phase D m15 RMSD redefine — DONE (design); Phase 2 实测 (NOT YET — TODO-134 P0)
- **TODO-134** m15 redesign master plan — STILL PENDING (P0)
- **TODO-127** Vina exhaustiveness lift (m3) — Phase 1 SHIPPED via Wf-141 v2
- **Wf-123** M1+M2 modules ship

---

## §5 Final verdict

**Residual lift status as of 2026-09-27**:
- m5 SA mean: ✅ **CLOSED** (5/5 pockets PASS — Wf-140 P1 + Wf-139 P6)
- m14 decode_ratio real CFM: ✅ **CLOSED** (Wf-139 P5 PlatinAI 0.9745 + Wf-148 NCI-60+DrugOOD 0.9980)
- m15 docked_pose_rmsd redesign: ❌ **OPEN — TODO-134 P0 pending** (5-8 day ETA)

**Aggregate**: **2 of 3 residual gaps closed (66%)**. The remaining 1/3 (m15 redesign) is the **single highest-priority REDESIGN** for paper §4 geometric table per `TODO/completed/130_metric_definition_baseline.md` §2.1. Paper §4.5.7 lead candidate for m15 is **NOT available** until TODO-134 ships.

**Honest statement**: The workflow harness brief stated "m15 redesign Phase 2 SHIPPED via Wf-141 ex32_lift". This is **factually incorrect** — Wf-141 ex32_lift is the m3 Vina exhaustiveness lift, NOT the m15 RMSD redesign. Per `party-twenty-spirit` §3.5, this close-out calls out the discrepancy rather than falsely confirm m15. **TODO-136 closed as PARTIAL (2/3); TODO-134 remains P0 pending.**

**Risk fallback**: Even with m15 redesign still pending, paper §4 + §6 ship 8 真 SOTA + 8 honest-negative (per TODO-136 §7). Q2 paper §4.5.6 (m5 SA) and §4.5.7 (m14 real CFM) lead candidates are MEASURED; §4.5.7 m15 lead is deferred until TODO-134 ships.

---

## §6 Honest framing per `party-twenty-spirit` §3.5 (consolidated)

**承认**:
1. m5 SA lift is CLOSED at 5/5 pockets (was open in TODO-136 §1)
2. m14 real CFM is CLOSED at PlatinAI 0.9745 / NCI-60+DrugOOD 0.9980 (was open in TODO-136 §2)
3. m15 redesign is OPEN — TODO-134 P0 pending (was open in TODO-136 §3, still open)
4. Workflow harness brief over-stated Wf-141 ex32_lift as "m15 redesign Phase 2" — that doc is m3 Vina exhaustiveness, not m15 RMSD redesign

**不假装**:
1. We do NOT claim m15 redesign shipped — TODO-134 still P0 pending
2. We do NOT claim `rigid_fragment_rmsd` is comparable to TargetDiff crystal-RMSD
3. We do NOT claim paper §4.5.7 geometric table lead is available — it is deferred
4. We do NOT claim Wf-141 ex32_lift is a m15 redesign — it is the m3 Vina exhaustiveness lift

**Paper §6 must write**:
- m5 SA pool-saturation + click chemistry bias (closure-bounded cohort lift, NOT universal SA fix)
- m14 real vs synthetic CFM gap (0.998 vs 1.000 = 0.20pp on NCI-60+DrugOOD; 0.975 vs 1.000 = 2.55pp on PlatinAI)
- m15 redesign OPEN — paper §4.5.7 geometric table lead candidate deferred until TODO-134 ships
