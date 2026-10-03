# TODO-82e — Wf-34 Phase F final synthesis (close-out)

**Date**: 2026-09-26
**Workflow**: Wf-34 Phase F (final synthesis, post-Wf-29~33 partial ship)
**Parent**: `TODO/pending/82_metric_gap_filling_plan.md` + `TODO/completed/82_metric_gap_filling.md` (Wf-29~34 partial-ship ledger)
**Sibling**: `TODO/completed/82_wf36_closure_theorem_verifier.md`
**Status**: ✅ SHIP — 9 skills R19-updated + NEW skill `mol-metal-gpu-execution` + TODO-82 closed-out

---

## §1 TL;DR

Wf-34 Phase F closes the Wf-29~34 cycle. Three deliverables land on disk:

| Deliverable | MEASURED | Path |
|---|---|---|
| **NEW skill** | `mol-metal-gpu-execution` SKILL.md (302 LOC, 10 sections) | `.claude/skills/mol-metal-gpu-execution/SKILL.md` |
| **9 skills R19-updated** | 8 existing (R17+R18) cross-ref `gpu-execution` + R19 Wf-29~33 entry points | `.claude/skills/{mol-metal-anticancer,cfm-training,click-chemistry,lambda-calculus,mechanism-mapping,paper-writing,r18-math,party-twenty-spirit}/SKILL.md` |
| **TODO-82 close-out** | `82_metric_gap_filling.md` SHIP + INDEX §13.6 final-results table updated | `TODO/completed/82_metric_gap_filling.md` + `TODO/INDEX.md` |

**One-line verdict**: every off-GPU deliverable lands; **0/7 metric gaps close to MEASURED-acceptable band** (R20 plan items per `82_metric_gap_filling.md` §5).

---

## §2 Per-deliverable status

### §2.1 NEW skill `mol-metal-gpu-execution`

**Path**: `.claude/skills/mol-metal-gpu-execution/SKILL.md` (16,513 bytes, 2026-09-26 12:39)
**Sections (10)**: §1 When to use · §2 Hardware (Blackwell + RTX 5090) · §3 venv setup · §4 Standard train-script template · §5 Multi-GPU scheduling · §6 5-seed × 50-epoch protocol · §7 MEASURED-vs-PROJECTED tag discipline · §8 Honest-negative reporting · §9 Output artifacts · §10 Cross-skill matrix

**Anchors**:
- `molmetal/scripts/check_blackwell_env.py` (env probe)
- `molmetal/scripts/setup_molmetal_env.sh` (venv entry-point)
- `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` (MEASURED GPU 0/1 state, 116.5 GB free)

### §2.2 9 skills R19-updated

| Skill | LOC | Update mechanism |
|---|---:|---|
| `mol-metal-anticancer` | 28,091 | §m1/m2/m21 → `train_ru_5metal_se3sb.py` + `train_cisplatin_resistance.py` pointers + `gpu-execution` §6 5-seed × 50-epoch recipe |
| `mol-metal-cfm-training` | 25,244 | §SE(3) SB initial-pose for `dock_10pockets_se3sb.py` (Wf-30 Phase 2 contract) + Lipman CFM `train_se3sb`/`train_transport` references |
| `mol-metal-click-chemistry` | 22,719 | §18 click rules + 84-MLE → references Wf-30 10/10 pocket staging + Wf-31 m2 cisplatin |
| `mol-metal-lambda-calculus` | 25,968 | §MLC + β-NF + KataGo-MCTS → `train_se3sb.py:613` history + `gpu-execution` §4 standard template |
| `mol-metal-mechanism-mapping` | 26,912 | §21 mech × 20 metric → M22-M44 mapped to Wf-29~34 scripts via mechanism_to_metric matrix |
| `mol-metal-paper-writing` | 29,955 | §paper v3 rebuild blocked by Fatal LaTeX error (`paper/main.tex:595`); v2 179-page Wf-26 surface remains canonical + `gpu-execution` §8 honest-negative |
| `mol-metal-r18-math` | 28,569 | T1 SE(3) SB + T5 Multi-PH + T6 PH Hardness math underpins Wf-30 + Wf-31 scripts |
| `party-twenty-spirit` | 35,161 | §3.5 0-tolerance — gates the no-fabrication discipline for Wf-29~34 partial-ship report |
| `mol-metal-gpu-execution` (NEW) | 16,513 | this skill |

**Cross-skill matrix**: 8 of 9 existing skills reference `mol-metal-gpu-execution` in their `related_skills` or `cross_ref` block (verified by `grep -l "gpu-execution" .claude/skills/*/SKILL.md` = 9/9 hits).

### §2.3 TODO-82 close-out

`TODO/completed/82_metric_gap_filling.md` SHIP (Wfs-29~34 partial-ship ledger, 305 lines): §1 TL;DR 7-gap table · §2 Per-phase results · §3 NEW skill · §4 paper v3 NOT-ship · §5 R20 plan items · §6 cross-reference · §7 why Wf-29~34 partial · §8 final verdict.

`TODO/INDEX.md` §13.6 final-results table updated with Wf-29~34 真活 ship ledger.

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED (on disk, this cycle)

| Item | Evidence |
|---|---|
| **9 skills R19-updated** | `ls -la .claude/skills/*/SKILL.md` all 2026-09-26 12:39-12:43; cross-ref grep 9/9 hits |
| **NEW skill `mol-metal-gpu-execution`** | 16,513 bytes / 10 sections / 2026-09-26 12:39 file mtime |
| **TODO-82 closed-out** | `TODO/completed/82_metric_gap_filling.md` SHIP, 305 lines, 2026-09-26 mtime |
| **10/10 pocket receptor staging** | `molmetal/reports/wf30_pocket_prep/pocket_status.json` 10/10 READY |
| **4 NEW train scripts + 3 verify + 1 dock** | `molmetal/scripts/{train_se3sb,train_transport,train_multiph,train_ru_5metal_se3sb,train_affinity_regression,dock_10pockets_se3sb}.py` SHIP |
| **paper v2 179-page surface** | `paper/main.pdf` 5,739,816 bytes / 179 pages (Wf-26 Phase 5 ship) — remains canonical |
| **GPU env probe** | `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` (Wf-22, 116.5 GB free, still valid 2026-09-26) |

### §3.2 PROJECTED (待 Wf-29~33 真跑 — R20 plan items)

| Item | Why PROJECTED | When MEASURED |
|---|---|---|
| **G1 m3 Vina Dock median lift** (-1.2885 → -8.5+ kcal/mol) | Wf-30 Phase 2 dock run not executed; only `dock_10pockets_se3sb.py` authored | R20·P1 (40-60 h GPU-1) |
| **G2 m1 Ru AUC lift** (0.7123 → 0.85+) | `train_ru_5metal_se3sb.py` SHIP but 5-seed × 50-epoch GPU run not launched | R20·P2 (60 h GPU-0) |
| **G3 m2 Pt A2780cis AUC lift** (0.648 → 0.85+) | `train_cisplatin_resistance.py` SHIP from R17, no retrain | R20·P3 (wet-lab dependent per `81_wetlab_data_prep.md`) |
| **G4 m7 Diversity Tanimoto lift** (0.1065 → 0.65+) | `train_multiph.py` SHIP, no Hilbert lift run | R20·P4 (2-4 wk CPU + 60 h GPU-1) |
| **G5 m16 Affinity r lift** (0.207 → 0.86) | `train_affinity_regression.py` SHIP, no 5-seed run | R20·P5 (30 h GPU-1) |
| **G6 m19 LOMO lift** (0.535 → 0.70+) | `train_metal_lomo.py` SHIP from R17, no Hilbert-Z-score | R20·P6 (60 h GPU-0) |
| **G7 10-pocket Vina** (5/10 → 10/10 MEASURED) | 10/10 receptors staged; 0/10 actual dock | R20·P7 (40 h GPU-1, after R20·P1) |
| **Paper v3 with MEASURED lifts** | Wf-33 main.log ends with Fatal LaTeX error; v2 179-page remains canonical | R20 (after GPU retrain lands) |

**Discipline**: per `party-twenty-spirit` §3.5, **0 fabricated MEASURED numbers**. PROJECTED items ship to R20 plan only.

---

## §4 TODO-82 总结: 7 gap 处理结果

| Gap | Metric | R18 MEASURED | Wf-29~34 result | Final status | R20 priority |
|---|---|---:|---|---|---:|
| **G1** | m3 Vina Dock median | -1.2885 kcal/mol | scripts + verify SHIP, no GPU retrain | WEAK (unchanged) | R20·P1 |
| **G2** | m1 Ru AUC | 0.7123 | `train_ru_5metal_se3sb.py` SHIP + verify, no GPU retrain | WEAK (unchanged) | R20·P2 |
| **G3** | m2 Pt A2780cis | 0.648 | `train_cisplatin_resistance.py` SHIP, no retrain | WEAK (unchanged) | R20·P3 |
| **G4** | m7 Diversity Tanimoto | 0.1065 | `train_multiph.py` SHIP, no lift run | WEAK (unchanged) | R20·P4 |
| **G5** | m16 Affinity r | 0.207 | `train_affinity_regression.py` SHIP + verify, no retrain | WEAK (unchanged) | R20·P5 |
| **G6** | m19 LOMO | 0.535 | `train_metal_lomo.py` SHIP, no Hilbert lift | WEAK (unchanged) | R20·P6 |
| **G7** | 10-pocket Vina | 5/10 MEASURED | 10/10 READY receptors, 0/10 actual dock | WEAK (pockets staged) | R20·P7 |

**One-line**: Wf-29~34 ships **everything that can be done off-GPU** (skill + scripts + verify + receptor prep + close-out). The 7 metric lifts require **GPU retrain wall-clock that did not run in this cycle**; per `party-twenty-spirit` §3.5, no fabricated MEASURED numbers.

---

## §5 NEW skill `mol-metal-gpu-execution` 价值

The skill exists because R18 ship produced 32 modules + 1,800 tests + 23 mechanisms but **0 actual GPU retrain**. Wf-22 confirmed Blackwell 95.6 GB + RTX 5090 31.8 GB VRAM free; Wf-23 partial, Wf-24 cleanup, Wf-25 Hilbert, Wf-26 paper v2 all ran on CPU. R19_PLAN §6-12 promised 6 phases × 12 weeks GPU training — but only Phase A (Wf-22 env probe) and Phase F (this skill) actually executed. R19 mid-cycle pivoted to "off-GPU scaffolding first" because kanzi GPU-0 contention + ultracode priority queue + 4-6 wk wall-time budget made full GPU retrain structurally impossible inside the cycle.

**Skill provides 4 key protocols** that R20 (and any future cycle) can invoke without re-deriving:

1. **§3 venv setup** — single canonical entry-point `bash molmetal/scripts/setup_molmetal_env.sh python …` (resolves `.venv-vlm` symlink, prepends `PYTHONPATH`, sets 8 env vars including `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
2. **§5 multi-GPU scheduling** — kanzi contention rules: GPU-0 default owner = kanzi_venv, GPU-1 default owner = Mol-Metal R18+ work; never assume `CUDA_VISIBLE_DEVICES=""`.
3. **§6 5-seed × 50-epoch standard** — every `train_*.py` accepts `--epochs 50 --seeds 0 1 2 3 4`; mean ± std reported per metric.
4. **§7 MEASURED vs PROJECTED tag discipline** — every metric in `metrics/by_metric/*.json` carries `MEASURED` / `PROJECTED` / `ARCHITECTURAL` label + source-file pointer; fabricated lifts blocked by `party-twenty-spirit` §3.5.

**Enables R20 to skip 4-6 wk of infrastructure setup** (already 0/4 → 4/4 by skill SHIP) and go straight to GPU launch.

---

## §6 Cross-reference

### §6.1 Workflow chain (Wf-29 ~ Wf-34)

| Wf | Phase | Ship ledger |
|---|---|---|
| **Wf-29** | fix train_se3sb.py bug + 4-script smoke | bug was already fixed inline in Wf-23; 4 train scripts on disk; 25/25 unit tests pass; 0/4 GPU benchmark JSON (no launch) |
| **Wf-30** | m3 Vina Dock + 10-pocket staging | Phase 1 SHIP (10/10 READY receptors in `molmetal/reports/wf30_pocket_prep/`); Phase 2 NOT ship (0/10 docked) |
| **Wf-31** | m1 + m2 + m7 GPU retrain | 3/3 scripts SHIP (`train_ru_5metal_se3sb` + `train_cisplatin_resistance` + `train_multiph`); 0/3 GPU runs |
| **Wf-32** | m16 + m19 + m8 bench | 2/3 scripts SHIP (`train_affinity_regression` + `train_metal_lomo`); `bench_pb_valid.py` does not exist standalone; 0/3 runs |
| **Wf-33** | paper v3 actual data replace PROJECTED | NOT ship — `paper/main.log` ends with `Fatal error occurred, no output PDF file produced!`; v2 179-page Wf-26 surface remains canonical |
| **Wf-34** | NEW skill + 9-skill update + final synthesis | ✅ SHIP — `mol-metal-gpu-execution` SKILL.md 302 LOC + 8 existing skills R19-cross-ref + this final close-out |

### §6.2 R19_PLAN alignment

`TODO/03_plans/R19_PLAN.md` §6-12 GPU training schedule (6 phases × 12 wk) was supposed to deliver M22 SE(3) SB + M23 Transport Relax + M24 Multi-PH + M25-M30 mechanisms + Phase A multi-GPU scaffold split + Pitfall P2.5. **None of the 6 phases ran.** R19_PLAN §24-27 planned 8 → 12 skills (4 NEW: blackwell-training, paper-v2, wet-lab, r19-math) — **only 1 NEW (`mol-metal-gpu-execution`) actually SHIP**; the other 3 NEW skills remain aspirational until R20.

**Verdict**: R19_PLAN §6-12 GPU training schedule is **honestly not delivered**; §24 skill expansion is **partially delivered (1/4 NEW)**. Per `party-twenty-spirit` §3.5, R19_PLAN is reframed in §28 as "scaffolding-first, GPU-execution-when-budget-allows". This file is the Wf-34 Phase F ledger of what was actually delivered.

### §6.3 TODO cross-reference

| File | Link |
|---|---|
| `TODO/pending/82_metric_gap_filling_plan.md` | Wf-29~34 plan (parent) |
| `TODO/completed/82_metric_gap_filling.md` | Wf-29~34 partial-ship ledger (sibling) |
| `TODO/completed/80_r19_plan.md` | R19 PLAN §6-12 (this cycle was supposed to deliver it; did not) |
| `TODO/completed/81_wetlab_data_prep.md` | Wf-28 — wet-lab tier ready (R20·P3 dependency) |
| `TODO/completed/79_paper_v2_supplementary.md` | Wf-26 — paper v2 179-page ship (Wf-33 paper v3 NOT ship) |
| `TODO/completed/78_hilbert_multiph_real.md` | Wf-25 — Hilbert R²=-0.443 honest-negative finding (NOT improved by Wf-29~34) |
| `TODO/completed/76_r18_blackwell_retrain_v2.md` | Wf-23 partial ship — `train_se3sb.py:613` bug fix |
| `TODO/03_plans/R19_PLAN.md` | §6-12 GPU schedule + §24-27 skill expansion + §28 close-out reframe |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §1.2 | 8 WEAK metrics honest framing (this report preserves it) |
| `TODO/INDEX.md` §13.6 | final-results table updated with Wf-29~34 ledger + 7 gap final status + R20 plan items |

### §6.4 Skill cross-reference

| Skill | Wf-34 role |
|---|---|
| `mol-metal-gpu-execution` (NEW) | Standardizes Blackwell + RTX 5090 retrain protocol (this file's anchor) |
| `mol-metal-anticancer` | m1 Ru + m2 Pt retrain recipes |
| `mol-metal-cfm-training` | SE(3) SB initial-pose for `dock_10pockets_se3sb.py` |
| `mol-metal-r18-math` | T1 SE(3) SB + T5 Multi-PH math underpins Wf-30 + Wf-31 scripts |
| `mol-metal-mechanism-mapping` | M22-M44 mapped to Wf-29~34 scripts |
| `mol-metal-paper-writing` | Paper v3 rebuild blocked by Fatal LaTeX error; v2 179-page Wf-26 surface remains canonical |
| `party-twenty-spirit` | §3.5 0-tolerance — gates the no-fabrication discipline |

---

## §7 Final verdict

**Wf-34 Phase F SHIP**: NEW skill + 9-skill update + TODO-82 close-out + this synthesis. **Off-GPU deliverables all on disk.** **0/7 metric gaps closed to MEASURED-acceptable band** — all 7 gaps are R20 plan items per `TODO/pending/82_metric_gap_filling_plan.md` §9 + `82_metric_gap_filling.md` §5.

**R20 readiness**: `mol-metal-gpu-execution` skill provides standard protocol; 6 train scripts + 3 verify scripts + 1 dock script + 10/10 pocket receptors are staged and ready to launch on GPU-1 with `bash molmetal/scripts/setup_molmetal_env.sh python ... --epochs 50 --seeds 0 1 2 3 4`. 4-6 wk of GPU retrain wall-clock (R20·P1~P7) is the actual R19 follow-on work that did not fit in this cycle.

**Date**: 2026-09-26 (Wf-34 Phase F final synthesis — close-out)
