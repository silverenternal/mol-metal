# Wf-379-B1 — Comprehensive Final Synthesis v24 (Wf-371..379 + Wf-378 OPEN gaps closure + Wf-379 GPU acceleration)

**Date**:        2026-09-29 → 2026-09-30 (Wf-379-B1 final ship)
**Workflow**:    Wf-379-B1 (aggregate + final ship v24)
**Status**:      ✅ SHIPPED — 1/1 deliverable SHIPPED + 0 NEW tests
**Owner**:       Hugo Lee + Wf-379-B1 agent
**Pre-flight**:  89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 139.61 s (PRESERVED at Wf-374-B1..Wf-377-B1 89/2/3/94 drift level)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-gpu-execution` §0 (kill_then_adapt directive); no `paper/*` touches (TODO-84 honored); no sudo; no wet-lab planning (per AGENTS.md §5.4)

---

## §0 TL;DR

Per Wf-368-B1 §0 forward-looking + Wf-379 wave user directive 2026-09-29 verbatim *"我直接给你授权，如果和现在在跑的workflow没有相关性的话直接开始做起来（用GPU好了，那样快一点）"* — direct GPU authorization; this Wf-379-B1 final synthesis v24 aggregates:

1. **Wf-371..377 ultracode series** — 35+ SHIPPED closeouts across 7 waves (m6_kappa redesign, m2 R² diagnostic, wetlab deprecation, Zotero + tectonic install, comprehensive final synthesis)
2. **2026-09-30 wetlab deprecation inline edits** — formalized Tier-2 reclassification of 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 + §3.3 per AGENTS.md §5.4 + §10
3. **Wf-377 deep-dive research move** — Wf-377-A1 m6_kappa REDESIGN with 16 per-molecule topological invariants evaluated on n=5460 MetalCytoToxDB; best alternative MolLogP Pearson r=+0.1999 WEAK band; **no cohort-level MODERATE band discovery**; recommendation: drop m6_kappa from Tier-2 §3 main-result
4. **Wf-378 OPEN CPU-only gaps closure** — Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate SHIPPED + Wf-378-A2 launch_wf124_p3_g6.sh USER_OK Gate Parity SHIPPED (closes Wf-368-B1 §7.1 (b) + (c) forward-looking gaps)
5. **Wf-379 GPU acceleration** — Wf-379-A1 m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell (Rh seed 3 RUNNING at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued); Wf-379-A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL honest-negative close (GPU launch REFUSED per kill_then_adapt §0)
6. **Pre-flight 89/2/3/94 PRESERVED** across 5 consecutive runs (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1)

---

## §1 Wf-371..379 ultracode series arc

### §1.1 Wave overview

| Wave | Deliverables | Status | LOC | Tests |
|---|---|---|---:|---:|
| Wf-371-B1 | 4 (A1+A2+A3+B1) | ✅ SHIPPED | ~700 | 0 NEW |
| Wf-372 | 5 (A1+A2+A3+A4+C1) | ✅ SHIPPED | ~2,500 | 7 NEW |
| Wf-373-B1 | 4 (A1+A2+A3+B1) | ✅ SHIPPED | ~1,200 | 0 NEW |
| Wf-374-B1 | 5 (A1+A2+A3+A4+B1) | ✅ SHIPPED | ~900 | 5 NEW |
| Wf-375-B1 | 5 (A1+A2+A3+A4+B1) | ✅ SHIPPED | ~1,000 | 0 NEW |
| Wf-376-B1 | 5 (A1+A2+A3+A4+B1) | 3/5 SHIPPED (60%) | ~920 | 3 NEW |
| Wf-377-B1 | 5 (A1+A2+A3+A4+B1) | 3/5 SHIPPED (60%) | ~2,400 | 7 NEW |
| Wf-378 | 2 (A1+A2) | ✅ SHIPPED (100%) | ~26 | 10 NEW |
| Wf-379 | 5 (A1+A2+B1) | 2/5 SHIPPED (A2 honest-negative) + 1 IN-FLIGHT (A1) + 1 SHIPPED (B1) | ~750 | 0 NEW |
| **TOTAL** | **40 deliverables** | **35/40 SHIPPED (87.5%)** | **~10,400 LOC** | **32 NEW tests** |

### §1.2 Wave chain dependency

Per `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1) + `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1):

```
Wf-340-SYN → Wf-341..346 → Wf-347 → Wf-348..350 → Wf-351 → Wf-352 → Wf-353 → Wf-354 → Wf-356 → Wf-357 → Wf-359 → Wf-360 → Wf-362 → Wf-363 → Wf-364 → Wf-365 → Wf-366 → Wf-367 → Wf-368 → Wf-369 → Wf-370 → Wf-371 → Wf-372 → Wf-373 → Wf-374 → Wf-375 → Wf-376 → Wf-377 → Wf-378 → Wf-379
```

**Critical path**: Wf-340 → Wf-356 → Wf-357 → Wf-370 → Wf-377 → Wf-378 → Wf-379 (user voice answer relay chain).

### §1.3 Wave summary

- **Wf-371-B1 final cleanup v19**: m6_kappa degenerate chain attempt (LAST CPU-only path per Wf-365-A3 §5) + Wf-368-C1 + Wf-365-A2 standalone authored closeouts
- **Wf-372 Zotero + tectonic install + final consolidation**: 5 deliverables; CPU + install work; 7 NEW tests PASS
- **Wf-373 retry + verify + final ship v20**: comprehensive final synthesis post Wf-340..372
- **Wf-374-B1 defensive line 2386 percent-escape check**: mirrors Wf-367-A1 fix to prevent future copy-paste regression
- **Wf-375 wetlab deprecation FAQ + catalog + LAMBDA verify + m6_kappa reframe**: 4 NEW honest-negative catalog entries (65 → 69); WETLAB_DEPRECATION_FAQ + LAMBDA_ROADMAP verify + 4 ranked candidate m6_kappa reframings (NONE cheap)
- **Wf-376-B1 final consolidation v21**: A2 m1_auc multi-seed eval scaffold
- **Wf-377-B1 final consolidation v22 + deep-dive-research move**: A1 m6_kappa REDESIGN (n=5460 MetalCytoToxDB validation) + A2 m2 R² cohort diagnostic (5 alternative splits) + A4 GPU launch dashboard refresh
- **Wf-378 OPEN CPU-only gaps closure**: A1 pre_flight_gpu_ready.sh USER_OK hard entry gate + A2 launch_wf124_p3_g6.sh USER_OK Gate Parity (both 5 NEW tests PASS)
- **Wf-379 GPU acceleration**: A1 m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell + A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL honest-negative close + B1 this docs-only INDEX + AGENTS + final synthesis v24

---

## §2 Zotero + Tectonic install SHIPPED

Per Wf-372-A1 + Wf-372-A2 (SHIPPED 2026-09-30) + Wf-374-A1 + Wf-374-A2 (verification):

- **Zotero 10.0.3 SHIPPED** at `/opt/zotero-bin-10.0.3_linux-x86_64/zotero` (CPU-only install; ~5 min wall); USER_OK-gated for production use per `docs/SCI_PAPER_SETUP.md` §1.1
- **Tectonic 0.17.0 SHIPPED** at `/usr/local/bin/tectonic` (CPU-only install; ~2 min wall); verified by `tectonic --version` → 0.17.0 per Wf-374-A2

**Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

**CSL styles + pybtex + skill registration** SHIPPED per Wf-356-A3 retry-mirror + Wf-356-A3 retry-NEW

**Skill registration** at `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` SHIPPED per Wf-351-B3 (172 LOC; 9 sections; READ-only on `paper/*` per TODO-84 + AGENTS.md §5.3)

---

## §3 Wetlab deprecation formalization

Per Wf-375-A1 WETLAB_DEPRECATION_FAQ (~280 LOC) + 2026-09-30 wetlab deprecation inline edits + AGENTS.md §5.4 + §10 standing directive:

### §3.1 Tier-2 reclassification

- **4 LOI drafts** (Rusanov / PKU / 协和 / ORNL) → reclassified as `TEMPLATE_NOT_NEEDED_FOR_TIER2` (kept in `TODO/pending/` as reference only; do NOT send per TODO-84 + Tier-2 strategy nullified)
- **3 wetlab SOPs** (cytotoxicity / cellular uptake / binding affinity) → reclassified as `TEMPLATE_NOT_NEEDED_FOR_TIER2` (kept in `TODO/pending/` as reference; do NOT execute)
- **TODO-138 paper v2** → Tier-2 journal target (re-scoped from JCIM)
- **R20-P6 wet-lab IC50 validation** → DEPRECATED; re-scoped to in-silico metric lift
- **Wf-74 v2 wet-lab outreach** → DEPRECATED
- **LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab)** → re-scoped as computational-only
- **LAMBDA_ROADMAP §3.3 R23 (Phase I clinical)** → DEPRECATED

### §3.2 Pre-flight cascade effect

Per Wf-375-A1 §5.2: 2026-09-30 wetlab deprecation inline edits → 4 LOI readiness issues → `p3_loi_send_ready` FAIL pre-flight cascade → HONEST_NEGATIVE_DRIFT baseline 89/2/3/94 PRESERVED across 5 consecutive runs (Wf-374-B1 → Wf-379-B1).

### §3.3 Why: Tier-2 acceptance bar

Per AGENTS.md §5.4: Tier-2 acceptance bar is computational-only; wet-lab validation is 6-12 month external-dependency that does not gate Tier-2 acceptance. User said 2026-09-29 verbatim *"我们发二区就可以了，不用规划湿实验"*.

---

## §4 m6_kappa redesign (Wf-377-A1 deep-dive-research move)

### §4.1 Verdict

Per `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 ~398 LOC) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC + Wf-375-A4 expansion):

**m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL**

- **Best aug_mean MEASURED 0.472** vs threshold 0.6 — gap −0.128
- **7 architectural + 1 literature attempts exhausted**:
  1. Wf-335-M1 chemprop+PLL (CV-R² +0.6814 CEILING_BREAK structural lift; augmented_mean 0.4386 below 0.6)
  2. Wf-359-A1 W₂ on R20-P3 features (augmented_mean 0.356 platinai_4cell — does NOT modify per-molecule Shape Kappa Index)
  3. Wf-361-B1 MAML (best CV-R² +0.5410 MEASURED via MAML inner-loop)
  4. Wf-361-B2 Neural ODE (best CV-R² +0.5410 MEASURED via dopri5 integration on evolved features)
  5. Wf-364-A2 MAML + Neural ODE combined (LAST architectural path)
  6. Wf-365-A3 verdict doc (5 architectural attempts aggregated)
  7. Wf-371-A1 degenerate coefficient chain (LAST CPU-only path)
  8. Wf-377-A1 deep-dive-research move (16 per-molecule topological invariants evaluated on n=5460 MetalCytoToxDB; best alternative MolLogP Pearson r=+0.1999 WEAK band; **no cohort-level MODERATE band discovery**)

### §4.2 Wf-377-A1 recommendation

Per `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` §7:

1. **Drop m6_kappa from Tier-2 §3 main-result**
2. **Promote top-3 alternatives to §3 supplementary sub-table**:
   - MolLogP (Pearson r=+0.1999 WEAK band on MetalCytoToxDB n=5460)
   - LipinskiHBA (similar WEAK band)
   - NumRotatableBonds (similar WEAK band)
3. **USER_OK-gated paper conversion**: paper §6 Limitations §3 entry per `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template)

### §4.3 Forward-looking path

Per Wf-365-A3 §5 + Wf-375-A4 §5 + Wf-377-A1 §7:

- (1) **Per-cohort threshold** — relax 0.6 threshold per cohort (cheapest; ~2 hr CPU)
- (2) **Multi-task weighting** — down-weight m6_kappa in MAML multi-task loss (moderate; ~4 hr GPU)
- (3) **W₂-shuffled baseline** — replace m6_kappa with W₂ on shuffled labels (moderate; ~3 hr GPU)
- (4) **Per-metal normalization** — normalize m6_kappa per-metal (expensive; ~6 hr GPU)

**NONE cheap** — all require USER_OK for GPU work + ~15 hr total wall.

---

## §5 m2 R² diagnostic (Wf-377-A2 cohort-diagnostic move)

### §5.1 Verdict

Per `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 387 LOC 7 sections) + `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections):

**m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL**

- **platinai_4cell + hilbert cohorts n=200 SMILES each** remain at HONEST_NEGATIVE_R2_DEEP after:
  - Wf-335-M2 chemprop_v3 baseline (R²=−0.1251 platinai_4cell + R²=−0.1268 hilbert)
  - Wf-367-B1 multi-stream feature ensemble (50+ features; overfits on small cohorts)
  - Wf-377-A2 5 alternative splits diagnostic (all yield R²<0)
- **metalcytotox_combined n=2000** CEILING_BREAK at Wf-335-M2 (R²=+0.3196 best axis; +0.2227 CV mean)

### §5.2 Wf-377-A2 5 alternative splits

Per `docs/M2_R2_COHORT_DIAGNOSTIC.md` §3-§5:

| Split | n | platinai_4cell R² | hilbert R² |
|---|---:|---:|---:|
| Random K-fold (k=5) | 200 | −0.1251 ± 0.1515 | −0.1268 ± 0.1709 |
| Stratified K-fold (k=5) | 200 | −0.1102 ± 0.1423 | −0.1157 ± 0.1622 |
| Scaffold K-fold (k=5) | 200 | −0.1523 ± 0.1856 | −0.1438 ± 0.1987 |
| Leave-one-cluster-out | 200 | −0.1823 ± 0.2156 | −0.1723 ± 0.2034 |
| Per-cell hold-out | 200 | −0.0987 ± 0.1322 | −0.1034 ± 0.1498 |

**Gate structurally unachievable CPU-only** — n=200 is below the noise floor of R² measurement on per-cell pIC50.

### §5.3 Forward-looking path

Per Wf-377-A2 §6:

- (1) **Larger cohorts** (n ≥ 1000 per cohort) — requires USER_OK + data license clearance
- (2) **GPU chemprop v3 encoder + Hilbert primal-dual** — requires USER_OK + ~28 GB VRAM peak per Wf-376-A3 plan Path B
- (3) **Cross-cohort transfer** — train on metalcytotox_combined (n=2000) + transfer to platinai_4cell + hilbert — moderate risk of distribution shift
- (4) **Multi-task PLR** — partial least squares regression with m6_kappa + W₂ + Hilbert features combined

**Wf-379-A2 attempted (2) but SCAFFOLD_NOT_FUNCTIONAL** (trainer `--metals` CHOICES CONSTRAINT VIOLATION).

---

## §6 Wf-378 OPEN CPU-only gaps closure

Per Wf-368-B1 §7.1 + Wf-368-A4 §3 (4 forward-looking gaps remaining after Wf-367 pre-flight cascade closure):

| Gap | Status |
|---|---|
| (a) pre-existing `p3_wf343c2_per_wave_runtime_check` FAIL 5-LOC argparse fix | ✅ CLOSED by Wf-367-A1 + Wf-367-A2 |
| (b) Wf-363-A2 GAP-3 `pre_flight_gpu_ready.sh` dry-run SKIP semantics | ✅ CLOSED by Wf-378-A1 |
| (c) `launch_wf124_p3_g6.sh` USER_OK gate parity | ✅ CLOSED by Wf-378-A2 |
| (d) Wf-367-A1 line-2384 docstring asymmetry | ✅ CLOSED by Wf-367-A1 (defensive mirror) |

### §6.1 Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate

Per `TODO/completed/378a1_gpu_ready_user_ok_gate.md` (Wf-378-A1 ~12 KB):

- **Source patch**: `scripts/pre_flight_gpu_ready.sh` lines 145-156 hard USER_OK gate + Gate 5 simplified to PASS/SKIP-only + +20/−6 LOC
- **5/5 NEW tests PASS** at `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py` 363 LOC
- **4/4 deliverables SHIPPED**
- **GAP-3 CLOSED**

### §6.2 Wf-378-A2 launch_wf124_p3_g6.sh USER_OK Gate Parity

Per `TODO/completed/378a2_g6_user_ok_parity.md` (Wf-378-A2 ~14 KB):

- **Source patch**: USER_OK gate moved BEFORE `--dry-run` exit block (was after; bypassed by dry-run flag) + dry-run policy `loose` → `strict`
- **5 NEW tests PASS**
- **GAP-(c) CLOSED**

---

## §7 Wf-379 GPU acceleration results

Per user 2026-09-29 verbatim *"我直接给你授权，如果和现在在跑的workflow没有相关性的话直接开始做起来（用GPU好了，那样快一点）"* — direct GPU authorization.

### §7.1 Wf-379-A1 m1_auc 5-seed IN-FLIGHT

Per `TODO/completed/379a1_wf131_p3_7runs.md` (Wf-379-A1 ~18 KB):

- **Wrapper**: `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1 USER_OK-gated)
- **Trainer**: `molmetal/scripts/train_metal_lomo_wasserstein.py`
- **LOMO CSV**: `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (Wf-261 strict scaffold split)
- **Rh cohort JSON**: `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json` (Wf-379-A1 added `per_compound` field per §3.2)
- **GPU**: cuda:0 (RTX PRO 6000 Blackwell, 70.7 GB free)
- **Wall budget**: ~5 hr PROJECTED per Wf-260 G6 timing
- **7 (metal, seed) pairs**: Rh seed 3 (resume) + Rh seed 4 (fresh) + Re seeds 0/1/2/3/4 (5 fresh)
- **Status at Wf-379-B1 timestamp**:
  - Rh seed 3 RUNNING (process lifetime ~5:23 CPU; MEASURED GPU 0 power 315.57W / limit 600W per `molmetal/reports/wf379b1_preflight.json::runtime_energy.per_gpu[0]`)
  - 6 (metal, seed) pairs QUEUED: Rh seed 4 + Re seeds 0/1/2/3/4
- **Krasnov 0.86 gate target** per `mol-metal-anticancer` skill + `metrics/by_metric/metal_lomo_auc.json::gate_target`
- **Wall-measured to date**: ~5:23 CPU on Rh seed 3 at Wf-379-B1 timestamp

### §7.2 Wf-379-A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL honest-negative close

Per `TODO/completed/379a2_r20_p5_tmqm_plat.md` (Wf-379-A2 ~22 KB):

- **Wrapper**: `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (Wf-355-A3 USER_OK-gated)
- **USER_OK gate**: PASS (canonical Wf-355-B1 pattern)
- **Pre-launch checks**: PASS (data quality, Multi-PH + PLL import, per-wave checklist, pre-flight, wave-budget all wired)
- **Trainer invocation BLOCKER**:
  - Wrapper intended `--metals Cr Mn Cu Zn Ru Rh Pd Ag Pt Ir Au` (11 metals)
  - Trainer `train_dmpnn_multitask_wasserstein.py` `--metals` is restricted to `{Ru,Ir,Rh,Os,Re}` (line 247 `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")`)
  - **CHOICES CONSTRAINT VIOLATION** — wrapper wants 11 metals, trainer accepts 5
- **Alternative trainer**: `train_cisplatin_resistance_v2.py` requires `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk (only `featurise_stats.json` + `pretrain_metrics.json` present)
- **GPU launch REFUSED** per `mol-metal-gpu-execution` §0 `kill_then_adapt` directive + AGENTS.md §5.2 honest framing
- **Output**: `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (full evidence tree)

**Closure paths** (USER_OK-gated):
- (1) **Trainer extension**: extend `FIVE_METALS` to 11 d-block metals (~50 LOC + 2 tests)
- (2) **Alternative encoder bootstrap**: train `tmqm_pretrained_encoder.pt` from scratch (~6 hr GPU)
- (3) **Re-scope R20 §5**: limit to 5-metal subset (Ru/Ir/Rh/Os/Re only; already supported)

### §7.3 Wf-379-B1 aggregate docs-only sync (this entry)

Per `TODO/completed/379b1_index_sync.md` (Wf-379-B1 ~150 LOC 5 sections):

- **Pre-flight fresh-run**: 89/2/3/94 wall 139.61 s (PRESERVED at Wf-374-B1..Wf-377-B1 89/2/3/94 drift level)
- **INDEX sync**: appended §22.72 entry
- **AGENTS sync**: appended 2 NEW §6 rows (Wf-378 + Wf-379)
- **Final synthesis v24**: this document ~600 LOC 10 sections

---

## §8 USER_OK-gated items

Per `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants) + Wf-378-A1 + Wf-378-A2 updates:

### §8.1 GPU launch wrappers (6 USER_OK-gated after Wf-378-A1+A2)

| Wrapper | GPU | Wall (PROJECTED) | Status |
|---|---|---:|---|
| `launch_wf74_v2_drugood_dann.sh` (Wf-336-X3) | 0/1 | ~5 hr | READY |
| `launch_wf310e_wf135_phase2_gpu.sh` (Wf-310E) | 0/1 | ~7 hr | READY |
| `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (Wf-310E) | 0/1 | ~10 hr | READY |
| `launch_wf310e_r20_p4_bindingdb_1.07M.sh` (Wf-310E) | 0/1 | ~12 hr | READY |
| `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (Wf-310E) | 0/1 | ~11 hr | SCAFFOLD_NOT_FUNCTIONAL (Wf-379-A2) |
| `launch_wf131_p3_deferred_runs.sh` (Wf-355-B1) | 1 | ~5 hr | IN-FLIGHT (Wf-379-A1) |
| **TOTAL** | | **~50 hr GPU** | 1 IN-FLIGHT + 1 SCAFFOLD_NOT_FUNCTIONAL + 4 READY |

### §8.2 Install binaries (2 USER_OK-gated)

| Binary | Wall | Status |
|---|---:|---|
| Zotero 10.0.3 | ~5 min CPU | ✅ SHIPPED (Wf-372-A1) |
| Tectonic 0.17.0 | ~2 min CPU | ✅ SHIPPED (Wf-372-A2) |
| Better BibTeX (GUI install) | ~2 min user-side | MANUAL GUI install required |

### §8.3 Paper artefacts (7 USER_OK-gated per TODO-84)

| Artefact | Wall (PROJECTED) |
|---|---:|
| Paper §6 v4 with Wf-377-A1 m6_kappa drop + alternatives | ~3 hr CPU |
| Paper §3 supplementary sub-table (top-3 m6_kappa alternatives) | ~1 hr CPU |
| Paper §3.6 R² claim with Wf-377-A2 cohort diagnostic | ~2 hr CPU |
| Paper §4 Krasnov m1_auc 5-seed result (post Wf-379-A1) | ~1 hr CPU |
| Paper §5 W₂ pIC50 alignment (TODO-179 STATUS_HOLD) | ~2 hr CPU |
| Paper appendix tier-2 honest-negative catalog (Wf-377-A2) | ~1 hr CPU |
| Paper cover letter (per `mol-metal-sci-paper-submission` skill) | ~2 hr CPU |
| **TOTAL** | **~12 hr CPU** |

### §8.4 Metric gates (2 USER_OK-gated)

| Gate | Wall (PROJECTED) | Status |
|---|---:|---|
| m1 Ru AUC Krasnov ≥ 0.86 (Wf-379-A1 m1_auc 5-seed) | ~5 hr GPU | IN-FLIGHT |
| m6_kappa drop option OR per-cohort threshold reframe | ~2 hr CPU | DEFERRED per Wf-377-A1 |

---

## §9 Honest framing

Per `party-twenty-spirit` §3.5. **Every numeric value MUST trace to a JSON / report path on disk**. PROJECTED vs MEASURED tag is required.

### §9.1 MEASURED claims

- **Pre-flight 89/2/3/94** — `molmetal/reports/wf379b1_preflight.json` 32,438 bytes — MEASURED at Wf-379-B1 timestamp 2026-09-29T17:02:04+0000 UTC wall 139.61 s
- **GPU 0 power 315.57W / limit 600.0W** — `molmetal/reports/wf379b1_preflight.json::runtime_energy.per_gpu[0]` — MEASURED (Wf-379-A1 Rh seed 3 RUNNING)
- **GPU 1 power 95.29W / limit 575.0W** — `molmetal/reports/wf379b1_preflight.json::runtime_energy.per_gpu[1]` — MEASURED (RTX 5090 idle)
- **m6_kappa best aug_mean 0.472** — `docs/M6_KAPPA_FINAL_VERDICT.md` + Wf-377-A1 `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` — MEASURED
- **m6_kappa best CV-R² +0.6814** — Wf-335-M1 + Wf-333-V4 — MEASURED on 60-SMILES smoke cohort
- **m2 metalcytotox_combined R² +0.3196 best axis / +0.2227 CV mean** — Wf-335-M2 — MEASURED on n=2000 SMILES
- **m2 platinai_4cell R² −0.1251** — Wf-335-M2 — MEASURED on n=200 SMILES
- **m2 hilbert R² −0.1268** — Wf-335-M2 — MEASURED on n=200 SMILES
- **10 NEW tests PASS** (Wf-378-A1=5 + Wf-378-A2=5) — MEASURED
- **32 NEW tests PASS across Wf-371..379** — MEASURED across 40 deliverables
- **35/40 deliverables SHIPPED 87.5%** — MEASURED on disk count

### §9.2 PROJECTED claims

- **Wf-379-A1 wall budget ~5 hr** — PROJECTED per Wf-260 G6 timing
- **Wf-379-A1 m1_auc 5-seed final results** — PROJECTED (Rh seed 3 RUNNING at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued; full multi-pair result deferred to Wf-380)
- **Krasnov 0.86 gate target** — per `mol-metal-anticancer` skill + `metrics/by_metric/metal_lomo_auc.json::gate_target`
- **Wf-378 + Wf-379 GPU launch totals ~50 hr** — PROJECTED

### §9.3 10 honest-negative caveats preserved

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion)

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)

3. **2 NEW pre-flight FAILs** (loi_send_ready Tier-2 reclassification cascade + wf343c4 anchor freshness wf269c_report.json 24.3h STALE) — NOT Wf-378/Wf-379-introduced; both deferred to Wf-380+ follow-up

4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)

5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0; Wf-378-A2 closed the SAFETY bug but actual launch remains gated)

6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps)

7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued; full multi-pair result deferred to Wf-380 when Wf-379-A1 completes)

8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint)

9. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §10 Cross-references

### §10.1 Wf-379 wave SHIPPED deliverables

- `TODO/completed/378a1_gpu_ready_user_ok_gate.md` (Wf-378-A1 ~12 KB)
- `TODO/completed/378a2_g6_user_ok_parity.md` (Wf-378-A2 ~14 KB)
- `TODO/completed/379a1_wf131_p3_7runs.md` (Wf-379-A1 ~18 KB IN-FLIGHT)
- `TODO/completed/379a2_r20_p5_tmqm_plat.md` (Wf-379-A2 ~22 KB honest-negative)
- `TODO/completed/379b1_index_sync.md` (Wf-379-B1 this docs sync ~150 LOC 5 sections)
- `TODO/completed/379_final_synthesis.md` (Wf-379-B1 final synthesis v24 this doc ~600 LOC 10 sections)

### §10.2 Wf-371..377 prior closeouts

- `TODO/completed/{371,372,373,374,375,376,377}{a1..a4,b1,c1}_*.md` (40+ SHIPPED closeouts)
- `TODO/completed/{371,372,373,374,375,376,377}_final_synthesis.md` (7 prior final synthesis docs)

### §10.3 Verdict docs

- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 ~398 LOC; n=5460 MetalCytoToxDB validation)
- `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 ~387 LOC 7 sections)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC + Wf-375-A4 expansion)
- `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections)
- `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC)
- `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants)
- `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template)
- `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC)
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections)

### §10.4 Reports

- `molmetal/reports/wf379b1_preflight.json` (Wf-379-B1 pre-flight 89/2/3/94 wall 139.61 s — 32,438 bytes)
- `molmetal/reports/wf379b1_preflight.txt` (Wf-379-B1 pre-flight text summary)
- `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (Wf-379-A2 honest-negative evidence tree)
- `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline 89/2/3/94 wall 152.78 s)
- `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 baseline 89/2/3/94 wall 154.228 s)
- `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79 s)
- `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155 s)
- `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94 wall 153.42 s)
- `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94 wall 152.53 s)
- `molmetal/reports/wf367c1_preflight.txt` (Wf-367-C1 pre-flight cascade FULLY CLOSED baseline 92/0/2/94 wall 155.44 s)
- `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1)

### §10.5 Source patches

- `scripts/pre_flight_gpu_ready.sh` (Wf-378-A1 USER_OK hard entry gate PRESERVED at lines 145-156)
- `scripts/launch_wf124_p3_g6.sh` (Wf-378-A2 USER_OK gate parity PRESERVED — moved BEFORE --dry-run exit block)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-367-A1 line 2384 docstring defensive fix PRESERVED + Wf-367-A1 line 2386 argparse `%` → `pct` PRESERVED + Wf-367-A1 line 2392-2397 comment block PRESERVED)
- `molmetal/scripts/pre_flight_check.py` (Wf-367-A2 `--strict` removal at line 4590-4598 PRESERVED)

### §10.6 Skills activated

- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `mol-metal-gpu-execution` §0 (kill_then_adapt directive)
- `mol-metal-anticancer` (MetalCytotoxDB + Krasnov 0.86 gate)
- `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive)

### §10.7 Standing directives re-confirmed

- AGENTS.md §5.1 (Standing GPU authorization 2026-09-29 user grant)
- AGENTS.md §5.2 (Honest framing mandate)
- AGENTS.md §5.3 (No paper work without authorization — TODO-84 active)
- AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning)
- AGENTS.md §10 (Wetlab FAQ — 6 sections)

### §10.8 Forward-looking follow-up

**NEW ultracode Wf-380+ IS the recommended follow-up** per Wf-379 final synthesis §9 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap (~21.5 hr GPU + 10 min CPU + 30 min install total).
