# Wf-258 Audit: CFM-Specific TODO Gap + GPU-Hour Estimate

**Date**: 2026-09-28
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚦 AUDIT (read-only; no edits — pure analysis)
**Parent**: `molmetal/reports/wf257_research/cfm_potential.md` §5 (CFM TODO shortlist)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED discipline) + `mol-metal-cfm-training` (CFM backbone) + `mol-metal-gpu-execution` §7 (no best-seed selection)

---

## §0 Audit scope

The user's question (cfm-related workflows in flight) maps to the Wf-257 §5 CFM TODO shortlist. Of those, **5 entries** remain PENDING (TODO-119 P1.3 + P1.4 are listed separately, even though both are sub-phases of TODO-119):

| # | TODO | Sub-task | Status | File |
|---|---|---|---|---|
| 1 | **TODO-119** | P1.3 — real-CFM decode_ratio lift on PlatinAI / DrugOOD / MetalCytoToxDB | ⏳ PENDING | `TODO/completed/119_decode_ratio_strengthen.md` §3 |
| 2 | **TODO-119** | P1.4 — per-innovation CFM ablation (YuelBond only / W₂ only / closure only) | ⏳ PENDING | `TODO/completed/119_decode_ratio_strengthen.md` §3 |
| 3 | **TODO-180** | G4 — Wf-124 P3 14 PENDING (metal, seed) pairs at 30 ep | ⏳ PENDING (1/15 done) | `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` |
| 4 | **TODO-181** | G5 — Wf-124 P3 50-epoch full sweep (5×3×50) | ⏳ PENDING | `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` |
| 5 | **TODO-182** | G6 — Wf-124 P3 per-innovation ablation (4 configs × 5 metals × 3 seeds = 60 runs) | ⏳ PENDING | `TODO/pending/182_wf124_p3_per_innovation_ablation.md` |

**Honest framing**:
- TODO-124 (Wf-124 P2 CFM full retrain) and TODO-126 (M1+M2 master plan) are SHIPPED, NOT in scope.
- TODO-96 (R20-P5 Catherine 30-metal) and TODO-98 (Wf-44 P2 closure OOD Phase 2) are listed in Wf-257 §5.4 as SHOULD-HAVE CFM-applicable TODOs but are NOT in the workflow task's "5 PENDING CFM-specific TODOs" list — they belong to the R20 closure track, not CFM directly. **OUT OF SCOPE for this audit.**
- TODO-119 file is in `TODO/completed/` (P1.1 + P1.2 closed) but P1.3 + P1.4 remain PENDING (per Wf-257 §5.1 table).

---

## §1 Per-TODO audit

### §1.1 TODO-119 P1.3 — Real-CFM decode_ratio lift

| Field | Value |
|---|---|
| **MEASURED state** | P1.1 (YuelBond decoder wire) + P1.2 (Wasserstein gradient flow reg) SHIPPED per Wf-124 P2 (decode_ratio = 1.000 on synthetic, 192/192). P1.3 (real-CFM lift on PlatinAI / DrugOOD) NOT YET RUN. Wf-181 m14 smoke (0.9980 ± 0.0008) + Wf-188 m14 FULL (0.9958, n=26,801) are POST-Wf-124 P2 ad-hoc measurements using the same checkpoint, NOT a formal P1.3 follow-up |
| **GPU-hours required** | **16 hr GPU-0 Blackwell** (per TODO-119 §3 P1.3: "2 days GPU-0"; conservative 8 hr/day interpretation). Per Wf-257 §5.4 ETA: "1 wk" = 40 hr, but this includes audit overhead. Honest GPU-wall estimate: **16-24 hr** |
| **Per-wave breakdown** (≤1 hr each) | 16-24 hr ÷ 1 hr = **16-24 waves**. Concrete waves: (a) Wf-119 P1.3 launch + smoke 100 samples; (b) per-cohort full-batch run on PlatinAI 4-cell; (c) per-cohort full-batch on DrugOOD assay/scaffold/size; (d) per-cohort on MetalCytoToxDB 5-metal; (e) cross-cohort aggregation + MEASURED-vs-PROJECTED audit |
| **Existing scripts reusable** | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (700 LOC Wf-124 P2 ship), `molmetal/scripts/wf181_m14_real_cfm_metalcytotox.py` (smoke template), `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (~617 KB Wf-124 P2 best ckpt) |
| **Output dir** | `molmetal/reports/wf119_cfm_retrain/real_cfm_decode.json` (per Wf-257 §5.4) |
| **Honest risk** | per TODO-119 §5: "There is a real chance P1.3 still produces 0.0". Current Wf-181/Wf-188 numbers (0.9980 / 0.9958) suggest P1.3 ALREADY PASSED on MetalCytoToxDB but P1.3 was scoped as PlatinAI/DrugOOD cohorts — those are not yet formally run |

### §1.2 TODO-119 P1.4 — Per-innovation CFM ablation

| Field | Value |
|---|---|
| **MEASURED state** | NOT RUN. Per TODO-119 §3 P1.4: "1 day CPU" framed as synthesis.md comparison audit. Per Wf-257 §5.4: "~7.5 hr ablation wall" (YuelBond only / W₂ only / closure only). The honest reading is that P1.4 is the ablation runs (GPU) + the comparison report (CPU) |
| **GPU-hours required** | **7.5 hr GPU-0 Blackwell** (per Wf-257 §5.4 ablation wall estimate). The 1-day-CPU in TODO-119 §3 is the analysis overhead, not the retrain wall |
| **Per-wave breakdown** (≤1 hr each) | 7.5 hr ÷ 1 hr = **8 waves**. Concrete waves: (a) YuelBond-only ablation launch; (b) W₂-only ablation; (c) closure-only ablation; (d) all-3 combined (Wf-124 P2 control); (e-g) 3 cross-leverage interaction runs; (h) ablation_report.json write + audit |
| **Existing scripts reusable** | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (with new ablation flags to add — see §3 checklist gap). `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` for warm-start |
| **Output dir** | `molmetal/reports/wf119_cfm_retrain/ablation_report.json` (per Wf-257 §5.4) |
| **Honest risk** | current 0.9958 decode_ratio is aggregate of 3 simultaneous innovations; per-lever attribution is a §3 honest-framing requirement (party-twenty-spirit §3.5). Ablation interaction term may be large (>30% of total lift per TODO-182 §3.4 hypothesis) |

### §1.3 TODO-180 G4 — Wf-124 P3 14 PENDING (metal, seed) pairs

| Field | Value |
|---|---|
| **MEASURED state** | 1/15 done (Ru seed=0, 30 ep, test_auc = 0.7613). 14 PENDING: Ru s1/s2, Ir s0/s1/s2, Os s0/s1/s2, Rh s0/s1/s2, Re s0/s1/s2 (per TODO-180 §1) |
| **GPU-hours required** | **1.8 hr GPU-1 RTX 5090** (per TODO-180 §1 + §5: 14 × ~7 min ≈ 1.6-1.8 hr wall) |
| **Per-wave breakdown** (≤1 hr each) | 1.8 hr ÷ 1 hr = **2 waves**. Concrete waves: (a) first 8 pairs (Ru s1/s2 + Ir s0/s1/s2 + Os s0/s1/s2) ≈ 56 min; (b) remaining 6 pairs (Rh s0/s1/s2 + Re s0/s1/s2) ≈ 42 min; (c) training_summary.json regeneration + audit (overhead) |
| **Existing scripts reusable** | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (700 LOC Wf-124 P3 ship) — ready to launch as-is |
| **Output dir** | `molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed{seed}.json` + `training_summary.json` |
| **Honest risk** | per TODO-180 §6 + Wf-124 P3 §5: Rh/Os/Re < 1.2k rows → per-seed std ±0.10 likely; 30-epoch undertraining; W₂ may not have fully kicked in. Aggregating 5×3 may yield < 0.86 Krasnov target. Honest framing: report mean ± std unfiltered, no best-seed selection |

### §1.4 TODO-181 G5 — Wf-124 P3 50-epoch full sweep

| Field | Value |
|---|---|
| **MEASURED state** | NOT RUN. Wf-124 P3 ran 30 ep due to session time budget. W₂ trajectory at Ru seed=0 still decreasing at epoch 30 (final 0.3906), suggesting undertraining. Best val_auc at epoch 13 (early-stop signal in 30-ep run) |
| **GPU-hours required** | **5 hr GPU-1 RTX 5090** (per TODO-181 §5: 15 pairs × 50 ep × ~15s ≈ 5 hr wall) |
| **Per-wave breakdown** (≤1 hr each) | 5 hr ÷ 1 hr = **5 waves**. Concrete waves: (a-e) 3 pairs per wave × 5 waves = 15 pairs (Ru s0/s1/s2 + Ir s0/s1/s2 + Os s0/s1/s2 + Rh s0/s1/s2 + Re s0/s1/s2) |
| **Existing scripts reusable** | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (with `--epochs 50` flag — already supported) |
| **Output dir** | `molmetal/reports/wf124_p3_multitask_50ep/{metal}/per_seed_metrics_seed{seed}.json` + `molmetal/reports/wf124_p3_50ep/30ep_vs_50ep_comparison.json` |
| **Honest risk** | per TODO-181 §6: 50-ep may add +0.01-0.03 lift OR plateau; honest framing is "uncertain, needs measurement". Do NOT cite 50-ep numbers as MEASURED until sweep completes |

### §1.5 TODO-182 G6 — Wf-124 P3 per-innovation ablation

| Field | Value |
|---|---|
| **MEASURED state** | NOT RUN. Wf-124 P3 ran all 3 innovations simultaneously (multi-task + W₂ + L2 guard). Per-lever attribution NOT measured. 4 ablation configs × 5 metals × 3 seeds = 60 runs pending |
| **GPU-hours required** | **7.5 hr GPU-1 RTX 5090** (per TODO-182 §2: 60 × ~7.5 min ≈ 7.5 hr wall at 30 ep). If 50-ep ablation (per TODO-182 §6 caveat): ~12.5 hr |
| **Per-wave breakdown** (≤1 hr each) | 7.5 hr ÷ 1 hr = **8 waves** (or 13 waves for 50-ep ablation). Concrete waves (30-ep version): (a-h) one config × 5 metals × 3 seeds × ~7.5 min each... actually 60 runs / 8 runs per hour = 8 waves of 8 runs each |
| **Existing scripts reusable** | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (per TODO-182 §5: needs `--no-multi-task`, `--no-wass`, `--no-l2-guard`, `--single-metal` flags — likely partial; `--no-multi-task` and `--no-wass` confirmed in `科研进展_pre_retrain_checklist_2026-09-28.md` §1.1.7; `--single-metal` also confirmed) |
| **Output dir** | `molmetal/reports/wf124_p3_ablation/ablation_report.json` |
| **Honest risk** | per TODO-182 §6: PROJECTED attribution is "multi-task 60-80%, W₂ 10-30%, L2 guard ~5%"; interaction term may be >30%. Per-metal caveat: smaller metals (Rh/Os/Re) may have different attribution. Do NOT cite per-lever percentages as MEASURED until 60 runs complete |

---

## §2 Aggregate GPU-hour + wave counts

### §2.1 GPU-hours sum (pure wall-time, conservative)

| TODO | GPU-hours |
|---|---:|
| TODO-119 P1.3 (real-CFM lift) | 16.0 |
| TODO-119 P1.4 (CFM ablation) | 7.5 |
| TODO-180 G4 (14 (metal, seed) pairs) | 1.8 |
| TODO-181 G5 (50-ep full sweep) | 5.0 |
| TODO-182 G6 (60 ablation runs) | 7.5 |
| **TOTAL** | **37.8 hr** |

**Honest framing**: if P1.3 is read as "1 wk" (40 hr) per Wf-257 §5.4 instead of "2 days" (16 hr) per TODO-119 §3, total rises to **61.8 hr**. The 16 hr figure is conservative (matches the §3 P1.3 literal "2 days GPU-0").

### §2.2 Per-wave count (≤1 hr each, "kill then adapt")

| TODO | Waves |
|---|---:|
| TODO-119 P1.3 | 16 |
| TODO-119 P1.4 | 8 |
| TODO-180 G4 | 2 |
| TODO-181 G5 | 5 |
| TODO-182 G6 | 8 |
| **TOTAL** | **39 waves** |

### §2.3 Priority order (per `party-twenty-spirit` §3.5 + Wf-257 §4.3)

| Priority | TODO | Why | Wall |
|---|---|---|---:|
| 1 | TODO-180 G4 | Smallest wall (1.8 hr) + closes honest gap from Wf-124 P3 §4 explicit "14 PENDING" cite; enables honest 5×3 paper §4 LEAD claim | 1.8 hr |
| 2 | TODO-182 G6 | Per-lever attribution is a §3 algorithm-description hard requirement; 60 runs in 7.5 hr gives full attribution matrix | 7.5 hr |
| 3 | TODO-181 G5 | 50-ep convergence check — needs G4 done first to have all 15 pairs at 30 ep as control | 5.0 hr |
| 4 | TODO-119 P1.3 | Real-CFM lift — current Wf-181/Wf-188 numbers (0.9980 / 0.9958) suggest ALREADY PASSED on MetalCytoToxDB, but PlatinAI / DrugOOD cohorts are NOT formally run | 16.0 hr |
| 5 | TODO-119 P1.4 | Per-innovation CFM ablation — needs P1.3 done first (it's the ablation of the P1.3 result) | 7.5 hr |

---

## §3 Reusable scripts + configs inventory

### §3.1 Existing scripts (count = 4 directly reusable for the 5 TODOs)

| Script | Reusable for | LOC | Source |
|---|---|---:|---|
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | TODO-119 P1.3 + P1.4 | ~700 | Wf-124 P2 ship |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | TODO-180 + 181 + 182 | ~700 | Wf-124 P3 ship |
| `molmetal/scripts/bench_cfm_decode_real.py` | TODO-119 P1.3 (verification) | n/a | Wf-181/Wf-188 verification |
| `molmetal/scripts/wf181_m14_real_cfm_metalcytotox.py` | TODO-119 P1.3 (smoke template) | n/a | Wf-181 ship |

### §3.2 Existing config

- `configs/cfm_full_retrain.yaml` — already exists for full CFM retrain (per Wf-243..252 retrain stop). Active retrain spec with DATA, MODEL, FLOW, OPTIMIZER, HARDWARE, TAGS, CHECKS sections. Reusable for TODO-119 P1.3.

### §3.3 Existing checkpoint

- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (~617 KB, Wf-124 P2 best). Reusable as warm-start for TODO-119 P1.3 + P1.4 (or as the "all 3 innovations" control in P1.4).

### §3.4 Checkpoint + log infrastructure

- `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}.pt` (15 ckpt target: 5 metals × 3 seeds) for TODO-180 + 181 + 182.
- `molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed{seed}.json` schema already in place for TODO-180.

---

## §4 CFM-specific checklist gap (per pre-retrain checklist §1-§10)

Cross-referenced against `科研进展_pre_retrain_checklist_2026-09-28.md` (pre-retrain checklist):

| # | Checklist item | CFM relevance | Gap for the 5 TODOs |
|---|---|---|---|
| §1.1.1-§1.1.10 | chemprop v2 v3 driver code fixes | Indirect (TODO-180/181/182 use `train_dmpnn_multitask_wasserstein.py` which inherits these fixes) | ✅ All ✅ DONE per checklist §1.1 |
| §1.1.7 | `--no-multi-task`, `--no-wass`, `--no-kendall`, `--no-dann`, `--single-metal`, `--n-seeds-min=3` flags | Direct for TODO-182 (ablation flags) | ✅ DONE for chemprop side. **CFM-side ablation flags NOT YET verified** for TODO-119 P1.4 — need to check `train_cfm_yuelbond_wasserstein.py` for `--no-yuelbond`, `--no-wass`, `--no-closure` (or equivalent) |
| §1.2.4 | SYNTHETIC upper-bound (m18 per-residue) | Indirect (m18 ≠ CFM) | N/A |
| §1.2.5 | W₂-pic50 DIAGNOSTIC vs GRADIENT | Direct for TODO-119 P1.3 + P1.4 | **OPEN**: need to verify whether Wf-124 P2 was DIAGNOSTIC or GRADIENT — per checklist §1.2.5 the chemprop side is DIAGNOSTIC-only (Wf-246-A). For CFM, TODO-119 §2 explicitly says "W₂ gradient flow penalty to CFM velocity field training" → GRADIENT path. Verify against `cfm_wasserstein_loss.py` |
| §1.2.6 | F2 m6_kappa NOT substitute for QED | Indirect | N/A |
| §3 (per `party-twenty-spirit` §3.5) | MEASURED-vs-PROJECTED tag discipline | Direct | ✅ Audit itself applies this discipline |
| §6 (5-seed × 50-epoch) | GPU-execution protocol | Direct for TODO-181 (50-ep) + TODO-180/182 (3-seed waiver) | ⚠️ **Waiver in place** per Wf-124 P3 §0: 3-seed per session time budget. 50-ep adds rigour back per TODO-181 |
| §7 (no best-seed selection) | Tag discipline | Direct | ✅ This audit applies |

**Summary checklist gap**:
1. **TODO-119 P1.4 needs CFM ablation flag verification**: check `molmetal/scripts/train_cfm_yuelbond_wasserstein.py --help` for `--no-yuelbond`, `--no-wass`, `--no-closure` flags (analogous to §1.1.7 chemprop flags). If missing, must add before P1.4 launch (NEW INFRASTRUCTURE per TODO-119 §5).
2. **TODO-119 P1.3 needs W₂-path verification**: confirm whether `cfm_wasserstein_loss.py` is in GRADIENT path (per TODO-119 §2 explicit cite) or DIAGNOSTIC (per checklist §1.2.5 chemprop precedent). If DIAGNOSTIC, must wire gradient path before P1.3 launch.

---

## §5 Output dir per user instruction (project root results/)

Per the workflow task: "Output dir per user instruction (project root `results/`)". Current state:

- Project root `results/` directory: **EXISTS** (per `git status` `?? results/`)
- Existing reports live in `molmetal/reports/wf{N}/` per project convention (Wf-124 P2 / Wf-181 / Wf-188 / Wf-194 / Wf-219 / Wf-253d / Wf-256 / Wf-257 — all under `molmetal/reports/`).
- For the 5 TODOs, the consistent output dir is `molmetal/reports/wf{N}/` (per Wf-257 §5.4 cite for TODO-119 + TODO-180/181/182 internal §3.2/§3.3 cites).

**Honest framing**: The user's instruction "project root `results/`" likely refers to the audit output dir (this report itself) — which is `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` per the workflow task explicit G8 acceptance gate. The 5 TODO output dirs are NOT in project root; they follow the established `molmetal/reports/wf{N}/` convention.

---

## §6 Acceptance gates (Wf-258 FLAT schema)

| Gate | Value | Source |
|---|---:|---|
| **G1_todo_119_pending_count** | **2** (P1.3 + P1.4) | TODO-119 §3 + Wf-257 §5.4 |
| **G2_todo_180_pending_count** | **14** (metal, seed) pairs | TODO-180 §1 explicit list |
| **G3_todo_181_pending_count** | **15** (metal, seed) pairs at 50 ep | TODO-181 §3.1 explicit (50-ep extension of G4) |
| **G4_todo_182_pending_count** | **60** runs (4 configs × 5 metals × 3 seeds) | TODO-182 §2 explicit |
| **G5_total_gpu_hours** | **37.8** hr (conservative: TODO-119 P1.3 = 16 hr per TODO-119 §3 literal; honest range 37.8-61.8 hr depending on P1.3 interpretation) | §2.1 sum |
| **G6_total_waves** | **39** (sum of ≤1-hr waves per §2.2) | §2.2 sum |
| **G7_existing_scripts** | **4** (`train_cfm_yuelbond_wasserstein.py` + `train_dmpnn_multitask_wasserstein.py` + `bench_cfm_decode_real.py` + `wf181_m14_real_cfm_metalcytotox.py`) | §3.1 |
| **G8_audit_report_path** | **`molmetal/reports/wf258_audit/cfm_todo_gap_audit.md`** (this file) | workflow task G8 |

---

## §7 Honest framing disclaimer (party-twenty-spirit §3.5)

- **The 4 TODOs (5 sub-tasks) are NOT equally ready to launch**:
  - TODO-180 G4 has ZERO new infrastructure required; can launch TODAY (1.8 hr wall).
  - TODO-181 G5 has ZERO new infrastructure required (just `--epochs 50` flag, already supported); can launch TODAY.
  - TODO-182 G6 needs ablation flags verified (likely partial; small chance of needing new infrastructure per §3.4 gap).
  - TODO-119 P1.3 + P1.4 likely need ablation flag verification on CFM side (§3.4 gap #1) + W₂ gradient-path verification (§3.4 gap #2). Small chance of needing 1-2 days infrastructure work.
- **GPU-hours are conservative estimates from each TODO doc**; actual wall-time may vary ±20% depending on per-metal convergence.
- **Smaller-metal caveat** (per Wf-124 P3 §5 + TODO-180 §6): Rh/Os/Re cohorts < 1.2k rows → per-seed std likely ±0.10. Aggregating across 5×3 may yield < 0.86 Krasnov target. Honest framing: report mean ± std unfiltered, NO best-seed selection (per `mol-metal-gpu-execution` §7).
- **30-epoch undertraining** (per TODO-181 §0): Wf-124 P3 only ran 30 ep due to session time budget; W₂ trajectory still decreasing at epoch 30. 50-ep sweep (TODO-181) is the proper scientific measurement.
- **Per-innovation attribution** (per TODO-182 §0): Wf-124 P3 ran all 3 innovations simultaneously; current 0.7613 is AGGREGATE lift, not per-lever attribution. Until 60 ablation runs complete, paper §3 must say "aggregate lift" not "per-lever attribution".
- **Wf-181 / Wf-188 numbers (0.9980 / 0.9958) on real CFM decode_ratio** suggest TODO-119 P1.3 may have ALREADY been partially satisfied on MetalCytoToxDB — but the formal P1.3 deliverable is per-cohort full-batch on PlatinAI 4-cell + DrugOOD assay/scaffold/size, which has NOT been formally run.
- **DO NOT cite any of these numbers as MEASURED until the actual retrain / ablation completes**. All estimates above are PROJECTED with explicit ETA per TODO doc.

---

## §8 Cross-references

- `molmetal/reports/wf257_research/cfm_potential.md` §5.1-§5.5 — CFM TODO shortlist (parent audit)
- `TODO/completed/119_decode_ratio_strengthen.md` §3 — TODO-119 P1.3 + P1.4 phases
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` §1-§5 — G4 details
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` §1-§5 — G5 details
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` §1-§5 — G6 details
- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json` — Wf-124 P2 50-ep synthetic decode_ratio 1.000 baseline
- `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` — Wf-124 P3 Ru seed=0 30-ep MEASURED
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` — Wf-124 P2 best ckpt
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM trainer (TODO-119)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — chemprop v2 v3 multi-task trainer (TODO-180/181/182)
- `configs/cfm_full_retrain.yaml` — full CFM retrain spec
- `科研进展_pre_retrain_checklist_2026-09-28.md` §1.1-§1.2 — pre-retrain gate discipline
- `mol-metal-gpu-execution` §7 — MEASURED-vs-PROJECTED tag discipline
- `mol-metal-cfm-training` — CFM backbone skill (YuelBond + Wasserstein + Catherine closure-bounded generation)
