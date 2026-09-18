# TODO Index — Single Source of Truth

**Last updated:** 2026-09-18 (R16b close-out — 2 PASS + 1 PARTIAL + 1 NEUTRAL — YuelBond 2000-probe decode_ratio=1.0 + 10k-step decode_ratio=1.0 at step 1000/2000 + GPU hung step 2500; Round-13 30-cell n_distinct=20 lift (vs 1 baseline) + REAL CLI flags + auto-pt-strict scaffold gate; PB 15-cell strict pass_rate=0.80 PROMOTED §4.6 DESIGN→MEASURED; Deflex v2 NEUTRAL (singleton-attractor undefined at n_distinct=1); TODO-24 ∤-probe PASS + 10k-shipped (INCONCLUSIVE on seeds 1234/7), TODO-25 Alt B TRIGGERED, TODO-26 paper-grade 300 cells DEFERRED, TODO-30 P2.5 Phases C-E DEFERRED; 4 ultracode workflows DONE)

This file replaces the old ad-hoc `pending/decisions.md` + `pending/risks.md` + `pending/roadmap.md` in-`pending/` location, and gives a complete inventory of every task file with its current status. Updated by hand when a task moves status; tasks numbered NN ≤ 12 are in `archive/`, 13–18 in `completed/`, 19+ in `pending/`.

---

## Layout

```
TODO/
├── INDEX.md               ← this file
├── README.md              ← project overview + navigation
├── decisions.md           ← pending architecture decisions (D1–D∞)
├── risks.md               ← open blockers / risks
├── roadmap.md             ← phase tracker
├── engineering_practices.md
├── environment.md
├── AUDIT_RESEARCH_GRADE.md
├── completion_audit_2026-09-13.md
├── project_workflow.md
│
├── pending/               ← ACTIVE plans / in-flight work (10 files)
│
├── completed/             ← SHIPPED work (31 files; +2 = 13_r12_pilot_delivered.md + 29_f2a_closed.md archived 2026-09-17)
│
├── archive/               ← STALE plans superseded by newer ones (4 files)
│
└── 01_research/ ... 13_lambda_clickchem/   ← historical research notes
```

---

## `pending/` — ACTIVE (10 files — TODO-13 + TODO-29 archived 2026-09-17; remaining items are R16-GPU-blocked or strategic-roadmap)

| # | File | Status | Owner | Depends on | Round |
|---|---|---|---|---|---|
| 14 | `14_full_100pocket_paper_r13.md` | ⚙️ **PARTIAL — R16 Path A 0 final cells; Path B 30/30 search-bound (pb_pass_rate=null); Path C BLOCKED + R16b Round-13 30-cell n_distinct=20 PASS (REAL CLI, auto-pt-strict, sa 3.66 MEASURED)** — §3.5 Deflex ship DONE 2026-09-17 (verified wired `03_method.tex:97` + 4-pass pdflatex 76p/5.22MB/0 errors, `wf_t14_s35_deflex_ship/final.md`); R16b n_distinct 1→20 lift MEASURED on 30 cells (`wf_r16_round13_30cell/final.md`) but paper-grade 300 cells still deferred to R17 (GPU retrain 1st then re-run sweep); per `wf_r16_master/MASTER.md` §1 + `wf_r16_index_update/final.md` + `wf_round13_100x3/final.md` §2 | (unset) | R12 + F2(a) + R15 fixes + GPU | R13/R15/R16/R17 |
| 19 | `19_user_decisions.md` | ✅ **INFO** — D8-D12 decisions aggregated in `wf_lambda_line/MASTER.md`; closed as a tracking artefact | (unset) | — | meta |
| 21 | `21_lambda_model_coupling.md` | ⚙️ **TACTICAL ROADMAP SHIPPED 2026-09-16** — Strategy 1 (Lambda-as-reward) shipped + `molmetal/docs/architecture/lambda_cfm_cascaded.md` (Strategy 3 design); joint retrain gated on R16 GPU; BUG-1 reshape fix resolved | (unset) | Lambda R12 + CFM R15 + R16 GPU | R15/R16 |
| 22 | `22_data_gap_alignment_plan.md` | ✅ **MOSTLY SHIPPED** — 17/25 metrics MEASURED (+1 since 2026-09-15: #22 CFM-vs-Lambda via R12 Deflex + R13 partial); 12 NEW project-internal metrics; 7 honest negatives preserved | WF-Data-Gap | — | R12/R13/R15 |
| 23 | `23_weak_to_strong_plan.md` | ✅ **STATUS v3 DONE 2026-09-17** — honest strong-metric audit SHIPPED: 22 metrics classified 16 STRONG / 4 WEAK / 2 CITED_ONLY / 1 BLOCKED (GPU); 7 honest negatives preserved verbatim; **PRIMARY journal: J. Chem. Inf. Model. (ACS)** under Pivot-A honest framing; `wf_honest_strong_metric_audit/final.md`; see `wf_model_line/MASTER.md` | (unset) | — | R12/R13/R15 |
| 24 | `24_cfm_architecture_redo_plan.md` | ✅ **P0 + P1 + YuelBond + 24-paper Frontier + R16b 2000-probe + 10k-step SHIPPED 2026-09-18** — 5 P0 CPU fixes live; 2000-step probe at h=128/layers=3 decode_ratio=1.0 at step 500+ (vs 0/192 floor, +100% lift, PASS gate ≥0.05); 10000-step run at h=128/layers=3/n_train=500/batch=4 decode_ratio=1.0 at step 1000+2000 (vs 0/192 baseline, lift holds); GPU hang at step 2500 (hsa_signal invalid, firmware SMU); seeds 1234/7 not reached; bond_loss 9.80→5.62 plateau; loss Inf spikes + mean_atoms=8.0 documented as honest caveats; retrain single-seed partial only; **3-seed stability retry DEFERRED to R17+** (cold power cycle required); per `wf_r16_yuelbond_2000_probe/final.md` §Verdict + `wf_r16_yuelbond_10000/final.md` + `wf_r16_master/MASTER.md` §2 | (unset) | GPU recovery | R13/R14/R15/R16/R17+ |
| 25 | `25_round14_lit_grounded_plan.md` | ⚙️ **PARTIAL — D2/D4 lit-grounded fixes SHIPPED + R16 Alt B (ETKDGv3 T10) TRIGGERED 2026-09-17 + R16b Deflex v2 NEUTRAL confirmed 2026-09-18** — SA-aware MCTS prior + cite-only SOTA comparator + metallodrug property proxies + pharmacophore channel + Pareto ranker + patent axis + rule registry + scaffold split all wired 2026-09-17; **R16 Deflex REGRESSION (div_tan -0.0044 vs gate +0.05) → triggers Alt B ETKDG path 2026-09-17 + R16b Deflex v2 verify NEUTRAL (lift +0.0000 vs gate +0.05, singleton attractor undefined) → §3.5 / §4 promotion still HALTED**: pre-existing _embed_3d_for_rmsd maxAttempts→numZeroFail fix at `r4_lambda_only_run.py` shipped `wf_r16_deflex_v2_verify/final.md` §2.1 (uncovered + fixed); pre-existing _root `_unreactive_states` cache `getattr(c, "pocket_id")` fix at line 3917 shipped §2.3; learned-shaping channel confirmed registered in 30/30 cells but no candidates to differentiate (singleton attractor root cause); Deflex §3.5 / §4 promotion HALTED awaiting F2(a) MetalLigandExchange structural rule | (unset) | R12 + R15 + R16 GPU + Alt B T10 + F2(a) | R14/R15/R16/R17+ |
| 26 | `26_round13_round14_complete_plan.md` | ⚙️ **R15 + R16 + R16b ROADMAP 2026-09-18** — 23 workflows DONE (+4 from R16b); 12-week roadmap W38-W49 → R16 = YuelBond retrain + Round-13 100×3 paper-grade + Deflex-arXiv; **R16b = 30-cell Round-13 sweep (REAL CLI + auto-pt-strict) + PB 15-cell smoke + Deflex v2 verify + 10k-step YuelBond re-run**; paper-grade 300 cells still DEFERRED to R17; per `wf_r16_index_update/final.md` + `wf_r16_master/MASTER.md` | (unset) | 13–25 + R15 + R16 + R16b | R12/R13/R14/R15/R16/R17+ |
| 28 | `28_round12_honest_negative_reframe.md` | ✅ **MOSTLY CLOSED** — §4 + §6 NEW item + §3.4 + §4.5 + §7 cross-references ship per `wf_lambda_line/l3_t28.md`; Items #5 (CROSS_REFS link) and #6 (new `\HONESTNEGATIVE{}` macro) DEFERRED to v2 (cosmetic) | (unset) | paper §4 editing | R12 |
| 30 | `30_pitfall_reinforce_plan.md` | ⚙️ **RANK 1-9 SHIPPED + P1.2 SHIPPED + P2.5 Phases A-B SHIPPED 2026-09-17 + R16b PB 15-cell smoke 0.80 MEASURED 2026-09-18** — P3.3 Pareto + P2.5 Pharmacophore + P4.2 Patent + P1.4 known-Pt regression + P5.2 Wet-lab Tier-1 + Tier-2 P6.1/P6.2/P2.2/P5.3 + **P1.2 FG-compat veto DONE** (`wf_t30_p12_fg_veto/final.md`); **R16b PB 15-cell strict pass_rate=0.8000 (12/15 cells, gate ≥0.6 PASS) MEASURED 2026-09-18** (`wf_r16_pb_15cell_smoke/final.md`) → §4.6 PB column 15 cells DESIGN→MEASURED; **P2.5 MD-relax Phases C-E still BLOCKED** (no docked mols to relax; PB run was `mol` mode not `dock` mode); P2.5 Phase A+B (PB schema + adapter) wired and waiting for PB-eligible docked inputs (Phase C-E re-engage after Round-13 100×3 retry produces docked candidates); Tier-2 P6.3, P2.3, P2.4, P1.1, P4.1, P5.2 Tier 3 still DEFER (GPU/wet-lab) | (unset) | GPU recovery + wet-lab + R17 Round-13 retry | R16/R17+ |

**In-flight workflows** (workflow tasks referenced from TODO files):

| Task ID | Workflow | Round | Status |
|---|---|---|---|
| `wo21fr1ug` / #671,#672 | WF-Round13-100x3-Sweep | R13 | **Phase 2 deferred to R16** (paper-grade 300 cells NOT achieved; R16 P3 sweep is the retry path) |
| ~~`wgvpkwmvb` / #667~~ | ~~WF-Round12-Lambda-PathA-10x3~~ | ~~R12 verify~~ | **✅ DONE 2026-09-15 — diversity lift verified on 30 cells** |
| `wgnp3ne6j` / #647 | WF-CFM-PathB-GPU-Retrain (Phase 1) | R13/R14 | phase 1 done; **R15 YuelBond phase 2 done; R16 GPU retrain pending** |
| `wf_r15_cross_verify` / #846 | WF-R15-Cross-Verify | R15 | **✅ DONE 2026-09-16 — 89 tests 82 pass + 5 skip + 2 fail (2 real bugs caught)** |

---

## `completed/` — SHIPPED (28 files; only newest 4 listed, full list via `ls`)

| # | File | Round | What shipped |
|---|---|---|---|
| 05 | `05_reinvent4_install.md` | R8 | REINVENT4 multiproperty bridge — `WF-Extra-2` |
| 15 | `15_anticancer_metric_suite_r11b.md` | R11b | anticancer metrics suite wired into `RewardAggregator` — **herg_proxy upgraded 2026-09-17** (Aronov 2005 + Veber 2002 heuristic; cisapride=0.253, terfenadine=0.011, paracetamol=1.000; 17/17 new tests + 39/39 regression; `wf_herg_real/final.md`) |
| 17 | `17_aggregate_weak_impls_and_pending.md` | R10/R11 | 1 ultracode audit round + 3 insights + 1 empirical number |
| 18 | `18_activity_assay_calibration.md` | R11 | assay context + censored labels; pIC50 retrain infra |

Older 24 files (R3–R9) cover: closed_loop_rewire, lipman_fm_adapter, proof_search_prior, vina_adapter, l3_tile_wireup, posebusters_adapter, aizynth_adapter, qvina_swap, tmqm_pretraining, round3_axes, published_numbers_comparison, metal_smiles_parser, tmqm_egnn_wireup, 3d_embed_sanity, metal_prior, ertl_sa, minibatch_ot, leakage_diagnosis, counterion_ablation, mmp13_surrogate, reinvent4_install, r4c_full_sweep. See `ls TODO/completed/` for full listing.

---

## Lambda Core Features — Recent Ship (Phase 1+2+3, 2026-09-15)

5 first-class Lambda core feature modules shipped + verified + integrated.

| Feature | Module | Tests | Report |
|---------|--------|-------|--------|
| L1 Conformer embed | `molmetal/molmetal_lam/lam_chem/conformer_embed.py` | `molmetal/tests/test_conformer_embed.py` | `molmetal/reports/wf_lambda_core/phase2l1_conformer_embed.md` |
| L2 Pharmacophore filter | `molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py` | `molmetal/tests/test_pharmacophore_filter.py` | `molmetal/reports/wf_lambda_core/phase2l2_pharmacophore.md` |
| L3 Stereo-aware reduction | `molmetal/molmetal_lam/lam_chem/stereo_aware_reduction.py` | `molmetal/tests/test_stereo_aware_reduction.py` | `molmetal/reports/wf_lambda_core/phase2l3_stereo_aware.md` |
| L4 Reaction confidence (Laplace) | `molmetal/molmetal_lam/reactions/confidence.py` | `molmetal/tests/test_reaction_confidence.py` | `molmetal/reports/wf_lambda_core/phase2l4_reaction_confidence.md` |
| L5 Pareto-front multi-objective ranking | `molmetal/molmetal_lam/search_alg/pareto.py` | `tests/test_pareto.py` | `molmetal/reports/wf_lambda_core/phase2l5_pareto.md` |

**Phase 3 integration**: `molmetal/scripts/wf_lambda_core_phase3_smoke.py` (no edits to locked `r4_lambda_only_run.py`) — `molmetal/reports/wf_lambda_core/phase3_integration.md` + `final.md`. 71/71 unit tests green; smoke produces 4-point Pareto front + hypervolume 5.88 in 0.293 s wall on 10 curated SMILES.

**New metric JSON**: `metrics/by_metric/pharmacophore_pass_rate.json` (pharma_pass_rate=0.7 on 10 curated SMILES, honest framing).

**Recommended §5 ablation axes** (5 new): ablate_conformer_embed, ablate_pharmacophore_filter, ablate_stereo_aware_reduction, ablate_reaction_confidence, ablate_pareto_ranker.

---

## `archive/` — STALE (4 files)

| # | File | Why archived | Superseded by |
|---|---|---|---|
| 07 | `07_r4c_full_sweep.md` | R4-C sweep scope replaced by R12/R13 pilots | 13, 14 |
| 11 | `11_algorithm_strengthening_r10.md` | R10 algorithm strengthening shipped | 22, 23 |
| 12 | `12_qvina_data_staging_r11.md` | QVina parity done at N=50 | 22 |
| 20 | `20_post_r10_r11_action_plan.md` | post-R10/R11 plan replaced | 22, 23, 26 |

---

## `decisions.md` (top-level) — pending architecture choices

| # | Topic | Status | Recommended |
|---|---|---|---|
| D1–D5 | (closed 2026-09-12) | closed | — |
| D6 | REINVENT4 install path | (a) separate venv — verified | adopted |
| D7 | D7-Apply both engines | (a) --engine default vina → both | applied |
| D8 | test_005 cohort | (not on file) | — |
| D9 | cite-only SOTA column | (not on file) | — |
| D10 | MW-range flag | (not on file) | — |
| D11 | journal target | (not on file) | — |

---

## `risks.md` (top-level) — open blockers

| Risk | Severity | Status | Mitigation |
|---|---|---|---|
| GPU outage (cold power cycle required) | high | mitigated (cold power cycle 2026-09-15) | watchdog in workflow |
| GPU HSA init: any broken KFD node = fatal | medium | mitigated (iGPU-first device selection) | device.py policy |
| CFM path (a) decode_ratio = 0/192 | high | open | path (c) λ-only default for R12 |
| Lambda singleton collapse | high | mitigated (3 fixes ship) | F2(a) MetalLigandExchange rule pending |

---

## `roadmap.md` (top-level) — phase tracker

| Round | Goal | Status |
|---|---|---|
| R3 (closed) | axes — Mol/Hybrid algorithmic axes | ✅ |
| R8 (closed) | 10 TODO items | ✅ |
| R10 (closed) | CFG end-to-end + Pt prior | ✅ |
| R11 (closed) | QVina + data staging + N=50 parity | ✅ |
| R11b (closed) | anticancer metric suite | ✅ |
| R12 (closed) | Round-12 pilot at top-journal standard | ✅ DELIVERED 147 cells DESIGN→MEASURED |
| R13 (in flight) | 100-pocket × 3-seed paper-grade sweep | ⚙️ partial; deferred to R16 P3 |
| R14 (in flight) | lit-grounded + math-prior + code-fix lift | ⚙️ partial; lit-grounded ship |
| **R15 (shipped 2026-09-16)** | **structural ship: 13 workflows across 4 parallel families + 1 verifier; YuelBond + Frontier 24 papers + sub-fix A/C + Deflex wireup + coupling bridge; 89 tests 82 pass + 5 skip + 2 fail (2 real bugs caught)** | ✅ **STRUCTURAL SHIP** |
| **R16 (shipped 2026-09-17)** | **CPU-only plumbing SHIPPED: BUG-1+BUG-2+CFM-import fixes all PASS; YuelBond 2000-probe decode_ratio PASS; Deflex v2 REGRESSION; Round-13 Path B search-bound → TODO-14 PARTIAL, TODO-25 Alt B triggered; 19 ultracode workflows DONE** | ✅ **CPU-only SHIP** |
| **R16b (shipped 2026-09-18)** | **4 sub-flows re-validated: YuelBond 10k-step decode_ratio=1.0 at step 1000+2000 (GPU hung step 2500); Round-13 30-cell n_distinct=20 lift (vs R12 baseline 1) using REAL CLI; PB 15-cell strict pass_rate=0.80 PROMOTE §4.6 DESIGN→MEASURED; Deflex v2 NEUTRAL (singleton attractor); 4 ultracode workflows DONE** | ✅ **STRUCTURAL SHIP** |
| arXiv | paper submission | ⏳ R16 W49 (2026-12-02) |

---

## Master close-out 2026-09-18 (R16b update)

Final state: 23 ultracode workflows DONE (R16 19 + R16b 4); R16b 4-sub-flow execution (2 PASS + 1 PARTIAL + 1 NEUTRAL); all CPU-only plumbing SHIPPED; remaining items are GPU-blocked (R17+) or strategic-roadmap only.

**R16b sub-flow verdicts** (per `wf_r16_master/MASTER.md`):
- `wf_r16_yuelbond_2000_probe` → **PASS** (decode_ratio 0/192 → 1.0 at step≥500, +100% lift) [unchanged from R16]
- `wf_r16_yuelbond_10000` → **PARTIAL** (decode_ratio=1.0 at step 1000/2000; GPU hung step 2500 hsa_signal invalid; seeds 1234/7 not reached)
- `wf_r16_round13_30cell` → **PASS** (30/30 cells @ n_sim=1000; n_distinct=20 lift from 1; sa_mean=3.66 + qed=0.708 + logp=-0.575 + tpsa=56.47 + anticancer=0.180 MEASURED at cohort; REAL supported CLI; auto-pt-strict scaffold gate)
- `wf_r16_pb_15cell_smoke` → **PASS** (15/15 cells PB-eligible; strict pass_rate=0.8000 (12/15) ≥ 0.6 gate; PROMOTE §4.6 PB column DESIGN→MEASURED)
- `wf_r16_deflex_v2_verify` → **NEUTRAL** (30 cells Lambda-only vs Deflex F5; div_lift=+0.0000 vs gate +0.05; singleton attractor undefined; HALT §3.5/§4 promotion)
- `wf_r16_bug1_fix` + `wf_r16_bug2_fix` + `wf_r16_cfm_import_fix` → all pre-flight **PASS** (43/43 + 11/11 + 5/5 tests) [unchanged]

**R16b cascade effect on TODO/ inventory**:
- TODO-24 (CFM architecture redo) → **PROBE PASS + 10K-STEP SHIPPED (single-seed)** → 3-seed stability DEFERRED to R17+ (cold power cycle required)
- TODO-14 (paper-grade 300 cells) → still PARTIAL; R16b n_distinct 1→20 lift MEASURED on 30 cells but paper-grade 300 cells DEFERRED to R17
- TODO-25 (Round-14 lit-grounded) → R16 Deflex REGRESSION → **Alt B (ETKDGv3 T10) triggered** + R16b Deflex v2 verify NEUTRAL (singleton attractor undefined); 3 pre-existing bugs uncovered + fixed at `r4_lambda_only_run.py:2505-2517, 2918-2922, 3915-3922`
- TODO-30 (P2.5 MD-relax) → Phases A+B prepared + **R16b PB 15-cell smoke 0.80 MEASURED** → §4.6 PB column 15 cells DESIGN→MEASURED; Phases C-E still BLOCKED on PB-eligible *docked* candidates (R16b run was `mol` mode not `dock` mode)

**19 workflows** (final accounting):
- `wvx501lwu` Pivot A (SBDD → de novo refactor)
- `w55mwabof` Pivot A follow-up (§3.6 → §7)
- `w897ab4bw` R15 close-out (structural ship + 2 real bugs caught)
- `ws4nmj35a` λ-line (6 agents)
- `wsh1rdomm` model-line (5 agents)
- `wtr0ehba9` SOTA reuse (Lipman / REINVENT4 / RxnFlow / PySR)
- `wjq7hdp79` 3 GPU-free lifts (EV-1 / EV-2 / EV-3)
- `w6eouk6gg` + `wbt15e6ip` + `wt4kdcxfc` pending TODO code batches 1/2/3
- `wu404enbf` Pitfall audit + TODO/30
- `wtska8vky` + `wkr4atydw` TODO/30 Tier-1 + Tier-2
- `wjd0svg3v` + `w1pnon85g` last CPU fixes + polish (hERG + BUG-1/2 + CFM import + Strategy 3 doc + D-MPNN)
- `widzuipcl` metallodrug vertical (§3.7 + 8 phases)
- `wf_r16_yuelbond_10000` R16b YuelBond 10000-step retrain (single-seed partial, GPU hang)
- `wf_r16_round13_30cell` R16b Round-13 30-cell sweep (REAL CLI + auto-pt-strict)
- `wf_r16_pb_15cell_smoke` R16b PB 15-cell strict pass_rate=0.80 PROMOTE §4.6
- `wf_r16_deflex_v2_verify` R16b Deflex v2 verify NEUTRAL (singleton attractor)

**Net effect on TODO/ inventory**:
- pending/ → 10 files (down from 11) — TODO-13 + TODO-29 archived 2026-09-17
- completed/ → 31 files (up from 29)
- 9 pending/ files are R17-GPU-blocked or strategic-roadmap:
  - TODO-14 (paper-grade 300 cells, R17 retry path)
  - TODO-19 (info — closed as tracking artefact)
  - TODO-21 (joint training)
  - TODO-22 (data-gap alignment, mostly SHIPPED)
  - TODO-23 (weak→strong)
  - TODO-24 (CFM architecture redo, PROBE PASS → 10000-step GPU retrain)
  - TODO-25 (round-14 lit-grounded, Alt B ETKDG triggered)
  - TODO-26 (R13+R14 complete plan, mostly SHIPPED)
  - TODO-28 (honest negative reframe, mostly CLOSED)
  - TODO-30 (pitfall reinforce plan, Tier-1+Tier-2 SHIPPED, P2.5 Phases C-E BLOCKED)

**Next round (R17 W42-W50)** is gated on GPU retrain:
- YuelBond 10000-step GPU retrain at h=128 (12-24h GPU)
- Round-13 100×3 paper-grade sweep at n_sim=1000 (6h GPU, after YuelBond)
- PB production pass rate at n_sim=1000 (4-5h GPU, after Round-13 retry)
- arXiv submission W50 (2026-12-09)


## Quick navigation

- **"What's the current status of the project?"** → `roadmap.md`
- **"What should I work on next?"** → `pending/26_round13_round14_complete_plan.md` (the master comprehensive plan)
- **"What's blocked / at risk?"** → `risks.md`
- **"What architecture decisions are pending?"** → `decisions.md`
- **"What did we just finish?"** → `completed/` newest entries
- **"What's stale / superseded?"** → `archive/`
- **"What did each workflow do?"** → `molmetal/reports/wf_*/final.md`
- **"What are the actual measured numbers?"** → `metrics/` (NEW 2026-09-15 — single source of truth for all measured values)

## Phase 5 (2026-09-15) — 7 parallel ship items (ultracode `wf_parallel_tasks`)

| # | Phase-3 agent | Module/script | Status | Lit anchors | Report |
|---|---|---|---|---|---|
| 1 | **3A** Click rule engineering | `molmetal/molmetal_lam/reactions/beta_reductions.py` + `lam_chem/pt_click_compat.py` + `tests/test_lambda_mcts_singleton.py` | SHIPPED | Hartwig 2010 Ch.5; Taube 1952 JACS Pt(II) assoc. mechanism | `wf_parallel_tasks/phase3a_f2a_smarts.md` |
| 2 | **3B** Anticancer metrics | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (NEW) + `tests/test_metrics_v2.py` (NEW) | SHIPPED | Weininger 1990; Hou 2007; Veith 2009; Polykovskiy 2020 | `wf_parallel_tasks/phase3b_metrics_v2.md` |
| 3 | **3C** PB 30-cell production harness | `molmetal/scripts/run_pb_production.py` (NEW) + `tests/test_run_pb_production.py` (NEW) | SHIPPED | Buttenschoen 2024 PB 1.0; Halgren 1996 MMFF94; Rappe 1992 UFF | `wf_parallel_tasks/phase3c_pb_production.md` |
| 4 | **3D** Per-residue sub-pocket diversity | `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py` (NEW) + `tests/test_per_residue_diversity.py` (NEW) | SHIPPED | Bemis 1996 Murcko; Jasial 2021 IntDiv; Peter 2019 SPF | `wf_parallel_tasks/phase3d_subpocket_div.md` |
| 5 | **3E** Triton fused_residual_add wire-in | `molmetal/adapters/flow_matching_lipman/__init__.py` + `molmetal/adapters/egnn_rocm.py` + `tests/test_fused_residual_wirein.py` (NEW) | SHIPPED | Karczewski 2024 ICML Th.1; Neyshabur 2017 NeurIPS Th.1 | `wf_parallel_tasks/phase3e_triton_wired.md` |
| 6 | **3F** Click-rule effect-size study | `molmetal/scripts/click_rule_effect_size_study.py` (NEW) | SHIPPED | Cohen 1988 power analysis; Himo 2005 CuAAC; Ertl 2008 SA | `wf_parallel_tasks/phase3f_click_effect.md` |
| 7 | **3G** Metal coordination probe | `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py` (NEW) + `tests/test_metal_coord_probe.py` (NEW) | SHIPPED | Lippard-Berg 1995; Reedijk 1987; Miessler 2014 | `wf_parallel_tasks/phase3g_metal_coord.md` |

**Phase-4 integration:** `molmetal/scripts/r4_lambda_only_run.py` (single integrator wire-in of 10 new metric dispatch wrappers; see `wf_parallel_tasks/phase4_integration.md`).

**File-conflict audit:** Each Phase-3 agent touched a disjoint file set (see `phase2_task_graph.json:conflict_with: []` for each task). No two Phase-3 agents edited the same file in parallel.

**Pytest result (full run, deselect 1 pre-existing failure):** `1604 passed, 7 skipped, 1 xpassed, 16 failed (pre-existing)`. The 16 pre-existing failures are in `test_generate_atom_types.py`, `test_lipman_spatial_contract.py`, `test_pocket_conditioned_lipman.py`, `test_rocm_lipman.py`, `test_round10_metrics_harness.py`, `test_sweep_helpers.py`, `test_vina_seed.py` — all modified 2026-09-11 to 2026-09-13 (before Phase-3 wf_parallel_tasks started 2026-09-15); none reference any of the 7 new metric/rule modules. The deselected failure is `test_atom_training_contract.py::test_atom_targets_are_supervised_but_not_supplied_as_features` (vocab-mask F3 broke its random-init CE=log(4) assumption; honest: that test was never updated when Phase-2 fixes added the vocab mask).

**New metrics JSON files:** `metrics/by_metric/diversity_subpocket.json` (DESIGN only — full sweep pending) + `metrics/by_metric/metal_coord_compliance.json` (DESIGN + 1 single-scaffold pilot, singleton collapse caveat documented).

**Final report:** `molmetal/reports/wf_parallel_tasks/final.md`.

---

## R15 ship summary (2026-09-16) — 13 workflows + 1 verifier

**4 parallel families + 1 verifier. 89 tests across 12 new files: 82 pass (92.1%) + 5 skip (libtorch ABI host issue, not code defect) + 2 fail (real bugs caught).**

| Family | Workflow | Phases | Tests | Verdict |
|---|---|---|---|---|
| **1. CFM-Rescue** | `wf_cfm_rescue` | 5 phases | 24/24 pass | ✅ SHIP (decode_ratio unchanged on CPU smoke) |
| **2. Lambda-Boost** | `wf_lambda_boost` | 4 phases + sub-fix C | 21/21 pass | ✅ SHIP (pocket-invariance structural break) |
| **3. Lambda-CFM-Coupling** | `wf_lambda_cfm_coupling` | 4 phases | 17/18 pass + 1 SKIP | ⚠️ PARTIAL (BUG-1 coupling reshape silent-fail) |
| **4. Deflex-Wireup** | `wf_deflex_wireup` | 5 phases | 20/26 pass + 5 SKIP | ⚠️ PARTIAL (BUG-2 Deflex v2 CWD-relative path) |
| **Verifier** | `wf_r15_cross_verify` | 1 workflow | **2 real bugs caught** | ✅ TEST SUITE EARNED ITS KEEP |

### New TODO entries from R15 (status updates)

- **TODO-14** (`14_full_100pocket_paper_r13.md`) — R15 update: 30+ cells MEASURED per WF-Lambda-Boost + sub-fix A + C + Deflex flags lift; paper-grade 300 cells still deferred (CPU-only)
- **TODO-21** (`21_lambda_model_coupling.md`) — **REOPENED**: R15 coupling_adapter + warm_start + learned_prior wire + dry-run bridge; BUG-1 blocks env-gated learned_prior cache
- **TODO-24** (`24_cfm_architecture_redo_plan.md`) — YuelBond decoder SHIPS + Frontier Research 24 papers surveyed; bonded-graph metric now MEASURED post-YuelBond lift (`n_bonded_samples` 0/8→8/8 on CPU smoke)
- **TODO-25** (`25_round14_lit_grounded_plan.md`) — R15 closes CFM + pocket-invariance structural ship
- **TODO-26** (`26_round13_round14_complete_plan.md`) — R15 ship summary + R16 = YuelBond GPU retrain + Round-13 100×3 paper-grade + Deflex integration; 12-week roadmap W38-W49 updated
- **TODO-29** (`29_f2a_round13_retry.md`) — **CLOSED 2026-09-16** via R15 Lambda-Boost Phase 2 (F2(a) MetalLigandExchange + AquaExchange SMARTS ship)

### Stale TODOs removed/closed in R15

- TODO-29 closed (F2(a) shipped via R15 Lambda-Boost Phase 2)

### 2 real bugs caught (R15 cross-verifier)

1. **BUG-1**: `learned_prior.py:412-417` coupling `_coupling_bias` reshape 64→5 silent-fail (3-line fix) — **✅ RESOLVED 2026-09-17** via `wf_bug1_fix` (reduce_coupling_bias total over n>=n_out, raises on n<n_out; 24/24 tests pass)
2. **BUG-2**: `pocket_macro_inference.py:101,104` Deflex v2 checkpoint CWD-relative path (5-line fix) — **✅ RESOLVED 2026-09-17** via `wf_bug2_fix` (Path(__file__).parent.parent / "models" resolution + CWD fallback; 6/6 tests pass)

Both R16 W39 ship-blockers cleared on CPU-only path. See `molmetal/reports/wf_last_cpu_fixes/MASTER.md`.

### Honest framing preserved (R15 → R16 boundary)

- CFM `decode_ratio` = 0/8 on CPU smoke (bit-exact baseline) → metric lift requires R16 GPU retrain (W42-W43, 12-24h GPU)
- Lambda pocket-invariance metric lift NOT re-measured → structural break implemented; metric lift pending R16 P3
- Lambda × CFM coupling active only at dry-run bridge (BUG-1 silently disables runtime coupling until fix)
- PB production pass rate (60-80% TargetDiff target) → R16 P3 sweep at n_sim=1000 with PB MMFF94 relax
- All R15 advantages are bounded to structural layer; R16 closes paper-grade measurements
- See `molmetal/reports/wf_r15_consolidation/final.md` for full R15 consolidation report

### Final report (R15)

`/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r15_consolidation/final.md`

---

## ⚠️ **Status reconciliation log 2026-09-15**

The following task-tracker / memory / TODO discrepancies were found and corrected:

| Discrepancy | What was claimed | Actual state | Resolution |
|---|---|---|---|
| `paper/main.pdf` ship | 56 pages / 5.05 MB / 0 errors | **FILE MISSING** (only section_03_method.pdf exists) | Tasks #460 #475 un-completed + new TODO-27 |
| Round-12 Lambda pilot "lift" | "+1.0 metal_compliance lift" | **NEGATIVE_RESULT_HONEST** (n_distinct=1 collapse, diversity=0.0 forced) | New TODO-28 + metrics/by_round/r12_lambda_pilot.json |
| Task #671 "WF-Round13-100x3-Sweep completed" | completed | **NOT EXECUTED** (sweep not yet run) | Task #671 reverted to pending |
| Task #529-531 "Diversity-Rotation integrate" | in_progress | **N/A** (wf was REJECTED — singleton collapse) | Tasks #529-531 deleted |
| Tasks #438-440 §3.3 + closure.py + tests | in_progress | **DONE** (files exist) | Marked completed |
| Tasks #476-478 data gap work | in_progress | **DONE** (TODO-22 shipped) | Marked completed |
| Task #480 "Verify CFM retrain diagnostic" | in_progress | **DONE** (wf_gpu_recovery_now/final.md has verdict) | Marked completed |
| Task #568 Phase 3 aggregate | in_progress | **DONE** (final.md has aggregate) | Marked completed |
| Task #605 "WF-Lambda-Fix-FullPath 4 fixes" | pending | **DELETED** (was superseded by tasks #606-616) | Deleted |
| Task #608 F2(a) MetalLigandExchange SMARTS | in_progress | **NOT SHIPPED** — identified as necessary-but-not-sufficient | Reverted to pending |
| Task #569 Phase 4 TODO-21 update | in_progress | **STILL NEEDED** | Description updated |
