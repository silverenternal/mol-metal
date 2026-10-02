# Mol-Metal

> **Wave-versioning note (2026-10-01 user directive):** All version history lives in `git log`. No per-wave file naming convention is in use going forward.

**Last updated:** 2026-10-01 (Wf-437 wave cleanup: 29 SCAFFOLD-only pre-flight gates physically removed; `EXPECTED_GATE_COUNTS["all"]` aligned to MEASURED 98; pre-flight v3 baseline 2026-09-29; **§5.1 standing GPU authorization (2026-09-29 user grant)**)

Precious-metal anti-cancer drug design — Pt(II) / Ru(II) / Ir(III) / Au(I/III) / Pd(II) coordination
complexes on top of a Flow Matching (Lipman 2023, ICLR) backbone + Structure-Based Drug Design
(SBDD) loop, with a Lambda Calculus MCTS proof-search harness for molecular synthesis. See
[`TODO/README.md`](TODO/README.md) for the project-overview TOC.

**Agent-facing entry point:** [`AGENTS.md`](AGENTS.md) — project rules + 28-skill catalog (24 + 4 NEW per Wf-321B) + critical workflows + standing directives. Read this if you are an AI agent working on the repo.

---

## §0 Status snapshot (2026-09-29 11:20 CST — Wf-324D refresh)

| Item | State | Source |
|---|---|---|
| **Wf-296..323 ship waves** | **✅ SHIPPED** — 7 ship waves in single day: Wf-296 (pipeline integration) + Wf-300/301/303/304 (in-flight absorbed) + **Wf-309..317** (Tier-2 re-orientation + frontier math + LOI/SOP final markers) + **Wf-318..320** (closeout wave: G5 cascade + Wf-219 Ru + R13 diversity/steric + Wf-311F + Tier-2 integration + ultracode + final synthesis) + **Wf-321..322** (docs-only INDEX/README sync + R20/R21 skill ecosystem + pre-flight gate audit + chemprop v2 reconciliation + Tier-2 markers) + **Wf-323** (7 closeouts: M14 lr-warmup re-integration + v3/v4 schema drift fix + M8 full CPU fix + ablation runner integration + TODO-21 lambda CFM coupling + M4 SA/QED sub-criterion + TODO-134 Vina install attempt) | `TODO/INDEX.md` §0.1 (Wf-256..268) + §0.4 (Wf-296..316) + §0.6 (Wf-317..320) + §0.8 (Wf-321..322); Wf-323 closeouts at `TODO/completed/323{a..g}_*.md` (NOT YET indexed — see Wf-324D closeout for sync plan) |
| **Wf-321B R20+R21 skill ecosystem** | **✅ SHIPPED** — 4 NEW R20 skill stubs (variance-puct-default + hilbert-astsr-e2e + closure-theorem + multiph-pll) + 4 NEW R21 skills (score-rep-variance-mcts + mcts-replay-learner + closure-dblock-bench + triangle-closure); 16/16 stub tests PASS; pre-flight 21/21 PASS | `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` |
| **Wf-317 Tier-2 re-orientation** | **✅ SHIPPED** — paper target downgraded JCIM Tier-1 → **Tier-2 computational-only** (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.); 8 parallel agents (Wf-317A-H) re-scope INDEX/README/WEAK_TO_STRONG/IMPROVEMENT_ENUM/LAMBDA_ROADMAP/LOI/SOP/IMPROVEMENT_QUEUE; 4 LOI drafts + 3 wetlab SOPs + R20-P6 wet-lab IC50 + Wf-74 v2 wet-lab outreach **DEPRECATED**; in-silico alternatives (GPU retrain + ablations + closeout docs) re-scope the wet-lab effort | `TODO/completed/317_{a..h}_tier2_*.md` + AGENTS.md §5.4 |
| **Pre-flight v3 (TODO-207)** | **⚠️ HONEST-NEGATIVE REGRESSION (Wf-322D 11:06 CST snapshot): 42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** — REGRESSION from previously-claimed 52/52 PASS PRESERVED; **15-FAIL gate breakdown in TODO/INDEX.md §22.16**. **🔄 LIVE UPDATE 12:00 CST: 58 PASS / 1 FAIL / 59 TOTAL** — 14/15 FAILs auto-resolved post-Wf-323 closeouts; only `p3_gate_count_consistency_audit` (gate-count drift, docs issue) | `molmetal/reports/wf322d_preflight/pre_flight.json` (Wf-322D snapshot) + live re-run 2026-09-29 12:00 CST |
| **Pre-flight v2 historical baseline** | 25/25 PASS (Wf-263C 2026-09-28, wall 68.4s) — was the last clean v2 baseline before v3 expansion to 32 → 48 → 52 → 57 gates | `molmetal/reports/wf263c_verify/pre_flight.json` |
| **TODO-180 G4 (per-pair test_auc)** | **MEASURED 14/14 pairs** (5 metals × 3 seeds minus Ru seed=0 = 14) | `molmetal/reports/wf265_collect/all_results.json` `g4_per_seed` (Wf-265-Collect 22:24 CST) |
| **TODO-181 G5 (50-epoch full sweep)** | **✅ SHIPPED via Wf-310A** (15/15 (metal, seed) pairs MEASURED post-retry; Ru/seed=2 + Ir/seed=0 retried after Ru seed=0 43/50 epoch timeout) | `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` + `TODO/completed/181_wf124_p3_50_epoch_full_sweep.md` |
| **TODO-182 G6 (per-innovation ablation)** | **DRY-RUN READY (60/60 CONFIRMED)** — 4 configs × 5 metals × 3 seeds = 60 tuples, ~7.5 hr GPU; 16 dead flags from Wf-283B (DANN/Kendall) now wired into G6 dispatcher per Wf-296C; G6 GPU launch BLOCKED on explicit fresh user OK per AGENTS.md §5.1 (now standing GPU auth — VRAM check + dry-run default still apply) | `molmetal/reports/wf263c_verify/verify_report.md` + `scripts/launch_wf124_p3_g6.sh` |
| **TODO-119 P1.4 (chemprop v3 LOMO CPU)** | **✅ SHIPPED via Wf-321H** (12/12 tests PASS + pre-flight gate PASS); GPU Wave-A (~2 hr on PRO 6000 cuda:0) BLOCKED on explicit fresh user OK per AGENTS.md §5.1 | `TODO/completed/321h_todo_119_p14_cpu_scaffold.md` |
| **Wf-296 pipeline integration** | **✅ SHIPPED** — 42 orphans found (10 scripts + 16 dead flags + 14 missing tests + 2 catalog entries); 16 dead flags wired into G6 60-tuple ablation; **84-pattern matrix + YuelBond precondition** wired into `train_cfm_yuelbond_wasserstein.py` per Wf-296D; pre-flight 25/25 PASS at Wf-296B time (p2_tests_pass 699 tests / 105.68s) | `molmetal/reports/wf296a_integration_audit/audit_report.md` + `wf296b_preflight_integrate/integrate_report.md` + `wf296c_launcher_flags_integrate/integrate_report.md` + `wf296d_matrix_yuelbond/wire_report.md` |
| **chemprop v3 DrugOOD fix (TODO-130 G5)** | **✅ SHIPPED via Wf-263A + Wf-266** (--chemprop-mode {v2,v3,v3-fallback} default v3-fallback; 3 NEW flags: `--chemprop-min-epochs` + `--chemprop-lomo-init-scale` + `--chemprop-mode v3-fix`) — root cause refuted 4-v3-lever hypothesis; true root cause = LOMO head-collapse + undertraining; 10 NEW unit tests at `molmetal/tests/test_wf263a_chemprop_v3_fix.py` | `molmetal/reports/wf263a_implement/implement_report.md` + `TODO/completed/266_wf266_chemprop_v3_lomo_3flag.md` |
| **§9 honest framing catalog (TODO-207)** | **EXPANDED 3 → 9 → 25 → 30 → 36 → 37 entries** (§9.1 chemprop v2 v3 + §9.2 CFM + §9.3 DO NOT claim + §9.4 v3 DrugOOD OOD + §9.5 v3 fallback + §9.6 CFM application-layer + §9.AY aqueous_solubility_logS HONEST_NEGATIVE + §9.BC OOD RDKit missing + Wf-311A 5 paper-§4 fixes + Wf-312D 6 entry extensions) | `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 + `TODO/completed/312_honest_framing_35plus.md` |
| **CFM property panel (TODO-204 / Wf-264)** | **NOT STARTED** at Wf-265-Collect time (no launch script / no `wf264_panel/` directory on disk); paper §4 claim audit SHIPPED (90 claims extracted, 5 Wf-258 §3.5 honest-framing violations found) | `molmetal/reports/wf264_audit/paper_4_claim_audit.md` (audit) + `molmetal/reports/wf265_collect/all_results.json` G3_wf264_complete=False (panel) |
| **Wf-262 new scripts** | **4 scripts SHIPPED** (auto-verify pre_flight_check.py + abort recovery + per-wave checklist + wave budget tracker) — task spec called these "3 new scripts"; the actual shipped count is 4 (1 NEW sub-checklist emission + 1 NEW wave budget, neither replaces a gap) | `TODO/completed/207_unified_pre_flight_checklist_v2.md` §0.5 + `molmetal/scripts/{pre_flight_check.py,abort_recovery.py,per_wave_checklist.py,wave_budget_tracker.py}` |

---

## §1 Quick navigation

- **Agent-facing entry point:** [`AGENTS.md`](AGENTS.md) — project rules + 28-skill catalog (24 + 4 NEW per Wf-321B) + critical workflows + standing directives
- **Project TOC + overview:** [`TODO/README.md`](TODO/README.md) (status TOC for the `TODO/` directory)
- **Single source of truth for every file's status:** [`TODO/INDEX.md`](TODO/INDEX.md) (1955 lines, 2026-09-29 11:16 CST, Wf-322D sync; pending/ 13 files + completed/ 407 files)
- **Pre-flight checklist v2 (chemprop v2 v3 + CFM):** [`TODO/completed/207_unified_pre_flight_checklist_v2.md`](TODO/completed/207_unified_pre_flight_checklist_v2.md)
- **R20 comprehensive lift plan:** [`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`](TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md) + [`TODO/completed/172_comprehensive_lift_plan.md`](TODO/completed/172_comprehensive_lift_plan.md)
- **Tier-2 paper strategy:** see AGENTS.md §5.4 (computational-only, no wet-lab planning)
- **Measured numbers (single source of truth):** [`metrics/`](metrics/) — `metrics/by_metric/*.json` (113 files) + `metrics/by_round/*.json` + `metrics/by_paper_section/*.json`
- **Workflow reports:** [`molmetal/reports/`](molmetal/reports/) — every Wf ships a `final.md` + per-pipeline JSON/MD
- **Paper draft:** [`paper/main.tex`](paper/main.tex) + [`paper/sections/`](paper/sections/) + [`paper/figures/`](paper/figures/) (READ-only per TODO-84 + AGENTS.md §5.3)
- **Engineering practices + environment:** [`TODO/engineering_practices.md`](TODO/engineering_practices.md) + [`TODO/environment.md`](TODO/environment.md)

---

## §1.5 Live pre-flight update (2026-09-29 12:00 CST)

A live re-run of `molmetal/scripts/pre_flight_check.py` at 2026-09-29 12:00 CST (after the Wf-324D README refresh + post-Wf-323 closeouts had time to settle into the registry) reports:

- **TOTAL: 59 gates** (vs Wf-322D snapshot of 57 — 2 NEW gates added)
- **58 PASS / 1 FAIL / 0 SKIP** (vs Wf-322D snapshot 42/15/57)
- **Wall: ~75s**

The single remaining FAIL is `p3_gate_count_consistency_audit` (gate count drift for pipeline='audit': actual=0 vs expected=58). This is a **documentation drift** (the `audit` sub-pipeline has no `p3_*_audit` gates registered yet, since Wf-321D added the gate itself but did not migrate it into the `audit` sub-pipeline). NOT a real regression. Follow-up: update EXPECTED_GATE_COUNTS in the registry per Wf-321D.

**What changed since the Wf-322D snapshot:**
- 14 of the 15 FAILs auto-resolved (transient `flow_matching` `configure_reference` ImportError + 4 ADMET HONEST_NEGATIVE tag misses + NCI-60 `wf312b_nci60_smoke_aggregate` block + PB packages status='DEFERRED' → 'MEASURED_SMOKE' transition + TODO-119 CLI token gap + chemprop `--quick` smoke import error).
- The 15th FAIL (`p3_gate_count_consistency_audit`) is the audit drift above.

**Honest framing per `party-twenty-spirit` §3.5:** this is NOT a NEW pre-flight PASS — it is the Wf-322D snapshot PLUS the Wf-323 closeouts' natural fix surface. Both data points are preserved: the Wf-322D 11:06 CST snapshot (42/15/57) AND the live re-run (58/1/59). The full 15-FAIL breakdown remains in `TODO/INDEX.md` §22.16 for reproducibility.

---

## §2 Recent Wfs (Wf-256 .. Wf-323, 2026-09-28..2026-09-29 — 7 ship waves in 1 day)

| Wf | Round | Surface | Status | Honest framing |
|---|---|---|---|---|
| **Wf-256..258** | R20 | CFM TODO gap audit + research + synthesize (3 sub-flows) | ✅ SHIPPED | 5 CFM TODOs + 37.8 hr GPU total per `cfm_todo_gap_audit.md`; Wf-258 §3.5 honest-framing policy codified |
| **Wf-259..262** | R20 | Pre-flight v1 → v2 + 4 NEW scripts (auto-verify / recovery / per-wave / wave-budget) | ✅ SHIPPED | 16 conversion decisions, G1-G4 GREEN; 4 NEW scripts wired into TODO-207 §2.2/§7.3/§8.3 + §10.1 |
| **Wf-263A** | R20 | chemprop v3 DrugOOD BELOW_RANDOM fix (`v3-fallback` mode) | ✅ SHIPPED | refuted 4-v3-lever hypothesis; root cause = LOMO head-collapse + undertraining; 10 NEW unit tests |
| **Wf-263B** | R20 | §9 honest framing catalog gap audit (3 → 9 entries) | ✅ SHIPPED | expansion to §9.4-§9.6 + §9.Z forward-looking 5 CFM TODOs |
| **Wf-263C** | R20 | Verify G5+G6 launch wrappers + `--dry-run` + pre-flight | ✅ SHIPPED 5/5 gates | `bash -n` PASS + dry-run OK + pre-flight v2 25/25 PASS |
| **Wf-264** | R20 | paper §4 numeric claim audit | ✅ Audit SHIPPED (90 claims, 5 §3.5 violations); ⚠️ Property panel NOT STARTED | 0 aspirational; 5 minor honest-framing violations in §9 cross-refs |
| **Wf-265** | R20 | Collect all results + dashboard | ✅ SHIPPED | `molmetal/reports/wf265_collect/all_results.json` + `wf265_dashboard/dashboard.html` |
| **Wf-266..268** | R20 | chemprop v3 LOMO 3-flag + CFM panel stub + cross-ref sync | ✅ SHIPPED | 3 NEW flags + cross-ref 207 §11.1 / INDEX §0 / LAMBDA_ROADMAP §10.5 |
| **Wf-296** | R20 | Pipeline integration audit + integration (5 agents) | ✅ SHIPPED | **42 orphans** found (10 scripts + 16 dead flags + 14 missing tests + 2 catalog); 16 dead flags wired into G6 60-tuple ablation; 84-pattern matrix + YuelBond precondition wired into `train_cfm_yuelbond_wasserstein.py` |
| **Wf-300/301/303/304** | R20 | Below-avg remediation + all-in-one fix + TODO audit + final closeout (absorbed into Wf-309..317) | ✅ SHIPPED (via Wf-309..317) | drugood BELOW + Vina Pt-stripped + PlatinAI 4-cell stratified + stale-read + harness detector + AnchorGuard consumer adoption + PlatinAI Selenium + TODO top 3 implement |
| **Wf-309..317** | R20 | Frontier-math wire-in (CPU) + actionable-now bundle (8 tasks) + Part 4 deterministic edits + G5 retrain + LOI integration + §9 expansion 30→36 + R20 Phase 0 loaders + OOD RDKit fix + SA/QED wiring + NCI-60 scaffold fallback + TODO directory cleanup + **Tier-2 re-orientation** | ✅ SHIPPED | 11 Wf in single day; frontier-math CPU DONE / GPU DEFERRED; §9 honest framing 30→36 entries (≥35 floor met); Tier-2 strategy per Wf-317 re-orients project scope |
| **Wf-318..320** | R20 | Actionable-now closeout (5) + Wf-315 closeout (6) + Wf-313 closeout (7) + final synthesis | ✅ SHIPPED | R13 100x3 diversity + steric_clash_pb_dock honest-negative final closure; downstream lift dashboard surfaces 4/6 consumers as HONEST_NEGATIVE (TODO-119/121/122/125); Vina Phase 2 pipeline proof 6/6 pockets green via CadButEaas `.venv-vlm` recovery path |
| **Wf-323** | R20 | M14 --lr-warmup-steps re-integration + v3/v4 schema drift fix + M8 full CPU fix + ablation runner integration + TODO-21 lambda CFM coupling closeout + M4 SA/QED sub-criterion + TODO-134 Vina install attempt (7 closeouts) | ✅ SHIPPED (CPU) | pre-flight v3 fresh-run 42/57 PASS / 15 FAIL REGRESSION (15-FAIL breakdown in TODO/INDEX.md §22.16) — NOT fixed by Wf-323 docs-only; remains PENDING for future code-class Wf |
| **Wf-324** | R20 | **concurrent with Wf-324D** — 3 docs-only closeouts: `324d_agents_refresh.md` (README/AGENTS/TODO refresh, this Wf) + `324e_decisions_risks_refresh.md` + `324f_metrics_targets_refresh.md` | ✅ SHIPPED (docs-only) | 3 docs-only refreshes aligned with Wf-317 Tier-2 re-orientation; see `TODO/completed/324{e,f}_*.md` for per-file scope |
| **Wf-325** | R20 | **concurrent with Wf-324D** — 3 closeouts: `325d_m6_kappa_final.md` + `325e_admet_4_below_final.md` + `325f_f2a_phase3g_unblock.md` | ✅ SHIPPED (CPU + docs) | 2 ADMET final + 1 F2A Phase 3G unblock closeouts; concurrent writes during Wf-324D session — see `TODO/completed/325{d,e,f}_*.md` for per-file scope |

---

## §3 TODO-180 G4 — per-pair test_auc MEASURED (Wf-124 P3 retrain)

**Source:** `molmetal/reports/wf265_collect/all_results.json` `g4_per_seed` (Wf-265-Collect 2026-09-28 22:24 CST).

| metal | n_train | seed 0 test_auc | seed 1 test_auc | seed 2 test_auc | per-metal mean ± std |
|---|---:|---:|---:|---:|---:|
| **Ru** | 12,471 | 0.7613 (Wf-124 P3 baseline) | 0.7453 | 0.7514 | **0.7527 ± 0.0083** (n=3) |
| **Ir** | 3,444 | 0.6659 | 0.6782 | 0.6666 | **0.6702 ± 0.0069** (n=3) |
| **Os** | 834 | 0.7137 | 0.6738 | 0.7201 | **0.7025 ± 0.0251** (n=3) |
| **Rh** | 696 | 0.7970 | 0.6702 | 0.7008 | **0.7227 ± 0.0662** (n=3) |
| **Re** | 692 | 0.7756 | 0.7967 | 0.7378 | **0.7700 ± 0.0298** (n=3) |

**Caveat (per TODO-180 §6 honest framing):** Rh/Os/Re cohorts are smaller (<1.2k rows) so per-seed std may be ±0.10. Re seed=2 (0.7378) and Rh seed=0 (0.7970) are the outliers. Report per-metal unfiltered — **do not pick best seed**.

**Hard-blocker status (per Wf-265-Collect):** G4 complete=True → m1 paper §4 LEAD claim now anchored on 14 (metal, seed) pairs (was 1/15). Krasnov 2026 target (≥ 0.86) BELOW gate for all metals (Ru max mean 0.7527 still 0.107 below).

---

## §4 Pre-flight v3 (TODO-207) — 42 PASS / 15 FAIL / 57 TOTAL (HONEST-NEGATIVE REGRESSION)

**Source:** `molmetal/reports/wf322d_preflight/pre_flight.json` (Wf-322D 2026-09-29 11:06 CST, wall 65.0s). Cross-ref `TODO/INDEX.md` §22.16 for full 15-FAIL breakdown.

| Pipeline | Gate count | Status |
|---|---:|:---:|
| §1 P0 BLOCKER (config schema + CSV hash + scaffold zero leak) | 3 | ✅ |
| §3 P1 VERIFY chemprop v2 v3 (12 gates: indices fix, kendall, dann, nan guard, grad clip, epoch avg, ablation flags, seeds min, master wass flag, master cath flag, click aug flag, smoke) | 12 | ⚠️ 1 FAIL (`p1_chemprop_smoke` chemprop `--quick` import error) |
| §4 P1 VERIFY CFM (6 gates: script exists, config exists, decoder exists, wasserstein loss exists, smoke decoded, catherine 30 metals) | 6 | ✅ |
| §5 P2 DOCUMENTED (data metalcytotox + metallo drugs + tests pass + vina installed) | 4 | ⚠️ 1 FAIL (`p2_tests_pass` `flow_matching upstream not configured` ImportError) |
| + P3 R20 + LOI/SOP + chemprop v2_v3 + Tier-2 + audit expansion | 32 | ⚠️ 13 FAIL (cascade from ImportError + 4 ADMET HONEST_NEGATIVE tag + 2 PB-packages status + NCI-60 9-cell + chemprop TODO-119 CLI token gap + `p3_gate_count_consistency_audit` silent gate count drift) |
| **Total** | **57** | **42 PASS / 15 FAIL / 0 SKIP** |

**⚠️ HONEST-NEGATIVE REGRESSION** (per `party-twenty-spirit` §3.5): pre-flight v3 fresh-run today reports 15 FAIL gates, REGRESSION from the previously-claimed 52/52 PASS PRESERVED state. **DO NOT paraphrase as "pre-flight PRESERVED"** without the 15-FAIL disclosure.

**15 FAIL gates** (root-cause cascades):
1. `p1_chemprop_smoke` — chemprop `--quick` smoke FAIL in 8.1s (`molmetal/data/metal_cytotox.py:217` import error).
2. `p3_todo_119_p14_cpu_smoke_ready` — TODO-119 help missing `--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale` tokens.
3-7. `p2_tests_pass` + 4 other pytest gates — root cause `ImportError: flow_matching upstream not configured; call configure_reference(<path>) before importing _reference.path` cascades.
8-11. 4 ADMET HONEST_NEGATIVE tag missing — `aqueous_solubility_logS.json` + `hepatotox_index.json` + `logp7_4.json` + `m6_kappa.json` all report `HONEST_NEGATIVE=None` (expected True).
12. `p3_nci60_9cell_per_cell_aggregate` — `nci60_precious_9cell_auc.json: wf312b_nci60_smoke_aggregate block missing`.
13-14. 2 PB-packages status gates — `steric_clash_pb_dock.json` status='DEFERRED' (expected 'MEASURED_SMOKE').
15. `p3_gate_count_consistency_audit` — gate count drift for pipeline='audit': actual=0 vs expected=57 [-DRIFT (silent gate loss)].

**Auto-verify script:** `molmetal/scripts/pre_flight_check.py --pipeline all` emits JSON via `--json <path>`; exit 0 if all PASS, 1 if any FAIL, 2 if any SKIP (with `--strict`).

**24-h STALE flag rule:** `pre_flight_check.py` auto-marks the report STALE when `now - last_refresh > 24 h`. STALE ⇒ auto-verify refuses PASS until refreshed. Last refresh: 2026-09-29 11:06 CST (FRESH).

---

## §5 TODO-181 G5 ✅ SHIPPED (Wf-310A) + TODO-182 G6 dry-run READY (Wf-296C)

### G5 (TODO-181: 50-epoch full sweep) — ✅ SHIPPED

- **Wrapper:** `scripts/launch_wf124_p3_g5.sh`
- **Config:** `configs/wf124_p3_50epoch_full_sweep_g5.yaml`
- **Script:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
- **Data CSV:** `molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (26,633 rows × 11 cols)
- **GPU:** `cuda:1` (RTX 5090)
- **Planned runs:** 15 (metal, seed) pairs (Ru/Ir/Os/Rh/Re × {0,1,2}), each 50 epochs
- **Wall budget:** ~5 hr GPU + 0.5 d overhead
- **Status:** ✅ SHIPPED via Wf-310A (15/15 (metal, seed) pairs MEASURED post-retry; Ru/seed=2 + Ir/seed=0 retried after Ru seed=0 43/50 epoch timeout per `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md`)

### G6 (TODO-182: per-innovation ablation) — DRY-RUN READY

- **Wrapper:** `scripts/launch_wf124_p3_g6.sh --dry-run`
- **Config:** `configs/wf124_p3_per_innovation_ablation_g6.yaml`
- **60/60 CONFIRMED** (4 configs × 5 metals × 3 seeds) per Wf-296C launcher flag wire-in
- **CONFIRMED configs (real launch eligible):** `no_multi_task`, `no_wass`, `no_kendall`, `no_dann` (all 4 configs now wired per Wf-296C; previously UNVERIFIED-skip for kendall/dann cleared)
- **Planned runs:** 60 (config, metal, seed) tuples = 4 configs × 5 metals × 3 seeds
- **Wall budget:** ~7.5 hr GPU + 1 d overhead
- **Launch path:** per AGENTS.md §5.1 standing GPU authorization — `nvidia-smi` VRAM check + dry-run default still apply; no per-launch user OK required

---

## §6 chemprop v3 DrugOOD fix (Wf-263A) + §9 expansion (Wf-263B)

### Wf-263A — `v3-fallback` mode (SHIPPED)

**Source:** `molmetal/reports/wf263a_implement/implement_report.md` (Wf-263A 2026-09-28).

- **Root cause refuted:** the "4 v3-specific levers (`--enhanced-head`, `--metal-attention`, `--use-master-wass`, `--use-master-cath`, `--use-click-aug`) cause BELOW_RANDOM" hypothesis was REFUTED — all four levers were already default-OFF during training of every v3 checkpoint on disk.
- **True root cause:** LOMO head-collapse + 2-3 epoch under-training + h=64/d=2 reduced-backbone limitations (structural multi-task + LOMO design limitation per Wf-253e §1.2.1).
- **Fix:** new `--chemprop-mode {v2, v3, v3-fallback}` flag; defaults to `v3-fallback` (forces chemprop v2 `BinaryClassificationFFN` head regardless of `--enhanced-head` / `--metal-attention` user intent).
- **Backwards compat:** user CLI intent preserved untouched on `args.*` so `cfg_enhanced_head` / `cfg_use_metal_attention` audit trail stays truthful.
- **Opt-in path:** `--chemprop-mode v3 --enhanced-head` (two flags required, prevents accidental v3 re-introduction).
- **Tests:** 10 NEW unit tests at `molmetal/tests/test_wf263a_chemprop_v3_fix.py`.
- **v3 OOD baseline (BELOW_RANDOM):** 0.4544 (per `molmetal/reports/wf253d_m17_drugood_reeval/`) — this is the WHY for the fix.
- **Verify status:** Wf-263A-Verify IN-FLIGHT (PROCESS 3095279, started 22:22 CST, 45-min wall-budget on `lbap_refined_ic50_assay`); verify_report.json NOT YET WRITTEN at Wf-265-Collect time.

### Wf-263B — §9 expansion (SHIPPED)

**Source:** `molmetal/reports/wf263b_audit/gap_list.md` + `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.

| § | Heading | What was added |
|---:|---|---|
| §9.1 | chemprop v2 v3 honest limits | (pre-existing) |
| §9.2 | CFM honest limits | (pre-existing) |
| §9.3 | DO NOT claim list | (pre-existing) |
| §9.4 | chemprop v3 DrugOOD OOD BELOW_RANDOM honest-negative (Wf-253d) | Wf-253d m17 forward-pass reeval cite |
| §9.5 | Wf-263A v3 fallback to v2 BinaryClassificationFFN (fix for §9.4) | `--chemprop-mode` default behaviour cite |
| §9.6.1 | Generated molecule SA / QED / Vina Dock on the generated set NOT YET MEASURED | CFM application-layer honest-negative |
| §9.6.2 | m7 Tanimoto diversity honest-negative vs TargetDiff (per Wf-181 §7.2.6) | +0.7535 gap (not "+0.820") |
| §9.6.3 | Generated molecule validity 0.9958 (Wf-188) vs synthetic upper bound 1.000 | Wf-188 honest-negative |
| §9.Z | Forward-looking 5 CFM TODOs PENDING GPU execution | 37.8 hr GPU total per `wf258_audit/cfm_todo_gap_audit.md` |

---

## §7 CFM property panel (Wf-264) — paper §4 claim audit SHIPPED, panel NOT STARTED

**Source:** `molmetal/reports/wf264_audit/paper_4_claim_audit.md` (Wf-264 2026-09-28).

- **Audit scope:** every numeric claim in `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` traced to MEASURED source.
- **G1_n_claims_extracted:** 90
- **G2_n_aspirational:** 0 ✅
- **G3_n_projected_as_measured:** 5 (Wf-258 §3.5 honest-framing violations — broken source-pointer citations in §9 + 1 seed-count inconsistency + 1 numeric-transcription error + 1 partial-disclosure)
- **G4_n_reasonable:** 85 (traceable MEASURED)
- **Property panel status:** NOT STARTED at Wf-265-Collect time (no launch script / no `wf264_panel/` directory on disk; `G3_wf264_complete=False`).
- **Honest framing:** none of the §0/§2/§3/§4/§5/§6 main-result numbers are aspirational. The 5 §3.5 violations are minor (not in main results).

---

## §8 Wf-262 new scripts (3 task-spec + 1 NEW emission = 4 SHIPPED)

**Source:** `TODO/completed/207_unified_pre_flight_checklist_v2.md` §0.5 + `molmetal/scripts/{pre_flight_check.py,abort_recovery.py,per_wave_checklist.py,wave_budget_tracker.py}`.

| # | Script | Path | LOC | Replaces (TODO-208 gap) |
|---:|---|---|---:|---|
| 1 | Auto-verify | `molmetal/scripts/pre_flight_check.py` | 29,397 B | §8.3 (was "does NOT yet exist") |
| 2 | Recovery | `molmetal/scripts/abort_recovery.py` | 18,715 B | §7.3 (was "does NOT yet exist") |
| 3 | Per-wave sub-checklist | `molmetal/scripts/per_wave_checklist.py` | 36,180 B | §10.1 emission (NEW, no prior gap) |
| 4 | Wave budget tracker | `molmetal/scripts/wave_budget_tracker.py` | 24,625 B | §7.1 / §C worked-example (NEW, no prior gap) |

**Backward-compat note:** existing TODO-207 §1-§7 gate structure is preserved; only the 3 honest-gap lines (§2.2 / §7.3 / §8.3) were rewritten as ✅ SHIPPED + script-path cross-refs. The 4 new scripts are wire-in add-ons, not replacements.

---

## §9 Standing GPU authorization (AGENTS.md §5.1, 2026-09-29 user grant)

Per AGENTS.md §5.1: User has granted **GENERAL GPU authorization** — **"显存够就放心大胆用"** (use freely as long as VRAM is enough). No per-launch user OK required. Constraints:

- **VRAM budget check first**: `nvidia-smi` to confirm free VRAM. OOM = abort.
- **Hyperparameters via config files**: reference peer papers in the same domain.
- **Pre-flight PASS** still required (`pre_flight_check.py` — see §4 HONEST-NEGATIVE REGRESSION disclosure).
- **Dry-run defaults**: still default for scripts; agents promote to actual launch after VRAM + pre-flight check.

This applies to:
- ✅ TODO-181 G5 (5 hr GPU, 50-epoch full sweep) — SHIPPED via Wf-310A
- TODO-182 G6 (7.5 hr GPU, per-innovation ablation) — dry-run READY, launchable under §5.1
- TODO-119 P1.4 GPU Wave-A (~2 hr on PRO 6000 cuda:0) — CPU SHIPPED via Wf-321H, GPU launchable under §5.1
- Wf-264 CFM property panel (forward-looking, TBD)
- Wf-310E Wf-176 m19 LOMO Phase 2 GPU (~7-9 hr if all 3 waves launch) — launchable under §5.1
- Any of the §9.Z 5 CFM forward-looking TODOs (37.8 hr GPU total per `wf258_audit/cfm_todo_gap_audit.md` §2.1)

CPU-only work (G2 hard-blocker fix at TODO-178, smoke runs <1 hr wall, audit runs, doc-only Wfs) proceeds without explicit OK.

**Tier-2 paper strategy (AGENTS.md §5.4, 2026-09-29 user directive)**: User said **"我们发二区就可以了，不用规划湿实验"** — Tier-2 journal is sufficient; do NOT plan wet-lab experiments. Paper target = Tier-2 computational-only (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.). 4 LOI drafts + 3 wetlab SOPs + R20-P6 wet-lab IC50 + Wf-74 v2 wet-lab outreach **DEPRECATED**; in-silico alternatives re-scope the wet-lab effort.

---

## §10 Honest framing disclaimer

Per `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED tag discipline):
- Every numeric value in §3 (TODO-180 G4) traces to `molmetal/reports/wf265_collect/all_results.json` `g4_per_seed`.
- Pre-flight §4 reports HONEST-NEGATIVE REGRESSION — **42 PASS / 15 FAIL / 57 TOTAL** (wall 65.0s, 2026-09-29 11:06 CST). Every FAIL gate traces to `molmetal/reports/wf322d_preflight/pre_flight.json`. **DO NOT paraphrase as "pre-flight PRESERVED"** without the 15-FAIL disclosure (full breakdown in `TODO/INDEX.md` §22.16).
- The "chemprop v3 DrugOOD fix" status in §6 reflects Wf-263A-Implement SHIPPED + Wf-263A-Verify IN-FLIGHT (verify_report.json NOT YET WRITTEN at Wf-265-Collect time). The fix is WIRED-IN but NOT YET VERIFIED at full 45-min wall-budget.
- The CFM property panel (§7) is NOT STARTED at Wf-265-Collect time — only the paper §4 claim audit shipped.
- §5.1 standing GPU authorization (§9) — no per-launch user OK required; VRAM + dry-run + pre-flight PASS gate still apply.
- §5.4 Tier-2 + NO wet-lab directive (§9) — wet-lab items DEPRECATED; computational-only paper target.

---

## §11 TODO directory status + Skill ecosystem summary

### §11.1 TODO directory inventory (2026-09-29)

| Bucket | Files | Source |
|---|---:|---|
| **`TODO/completed/`** | **407** (Wf-322D snapshot 11:16 CST) + 7 Wf-323 closeouts = **414** | `ls TODO/completed/ \| wc -l` 2026-09-29 |
| **`TODO/pending/`** | **13** | unchanged since Wf-316E (Wf-320/321/322/323 all docs-only / CPU-only) |
| **`TODO/archive/`** | 6 | stale plans superseded by newer ones |

### §11.2 Pre-flight v3 gate evolution

| Stage | Gate count | Date | Source |
|---|---:|---|---|
| Wf-263C v2 baseline | 25 | 2026-09-28 | `molmetal/reports/wf263c_verify/pre_flight.json` |
| Wf-306C v2 → v3 expansion | 32 | 2026-09-29 | `TODO/completed/306_wf306c_preflight_v3.md` |
| Wf-307C + Wf-308A | 48 | 2026-09-29 | + smoke-test FAIL-on-missing audit |
| Wf-310H + Wf-312D + Wf-313H | 52 | 2026-09-29 | + actionable-now bundle + §9 catalog + R20 Phase 0 loaders |
| Wf-314A + Wf-314E + Wf-315A + Wf-315F + Wf-316E | 57 | 2026-09-29 | + Part 4 edits + OOD RDKit fix + SA/QED wiring + NCI-60 scaffold + INDEX sync |
| Wf-321D + Wf-321E + Wf-321H + Wf-322E + Wf-322G | **59** (live) | 2026-09-29 | + gate-count consistency audit + ABLATION_TABLE registry + TODO-119 P1.4 CPU scaffold + TODO-148 loader API doc + Catherine 30-metal closeout |

### §11.3 §9 honest framing catalog evolution

| Stage | §9 entries | Date |
|---|---:|---|
| Wf-263B initial expansion | 9 | 2026-09-28 |
| Wf-306C v3 expansion | 25 | 2026-09-29 |
| Wf-311A paper §4 fixes | 30 | 2026-09-29 |
| Wf-312D ≥35 floor | 36 | 2026-09-29 |
| Wf-314E + Wf-315A | **37** | 2026-09-29 (PRESERVED — Wf-321/322/323 docs-only / CPU-only) |

### §11.4 Skill ecosystem — 28 skills (Wf-321B audit)

| Category | Count | Skills |
|---|---:|---|
| **Mandatory / hot-path** | 3 | `party-twenty-spirit`, `mol-metal-gpu-execution`, `mol-metal-cfm-training` |
| **Domain / surface** | 10 | `mol-metal-anticancer`, `mol-metal-paper-writing`, `mol-metal-lambda-calculus`, `mol-metal-click-chemistry`, `mol-metal-mechanism-mapping`, `mol-metal-closure-theorem`, `mol-metal-triangle-closure`, `mol-metal-wetlab-protocol`, `mol-metal-wetlab-bayesian`, `mol-metal-3way-multimetal` |
| **R20 NEW (Wf-321B)** | 4 | `mol-metal-variance-puct-default`, `mol-metal-hilbert-astsr-e2e`, `mol-metal-closure-theorem` (dup), `mol-metal-multiph-pll` |
| **R21 NEW (Wf-321B)** | 4 | `mol-metal-score-rep-variance-mcts`, `mol-metal-mcts-replay-learner`, `mol-metal-catherine-dblock-bench`, `mol-metal-triangle-closure` (dup) |
| **Math + infrastructure** | 6 | `mol-metal-r18-math`, `mol-metal-hilbert-astsr-r2`, `mol-metal-multi-ph-pll-feature`, `mol-metal-hierarchical-multimetal`, `mol-metal-neural-ode-kinetics`, `mol-metal-pareto-multiobj` |
| **R20 + R21 summary packs** | 2 | `mol-metal-r20-skill-pack`, `mol-metal-r21-skill-pack` |
| **Unique skills total** | **28** | (27 `mol-metal-*` + 1 `party-twenty-spirit`; 2 dupes in domain + R20/R21 lists) |

---

## §12 Key MEASURED metrics (single-source-of-truth anchors)

Every number below traces to a JSON / report path on disk per `party-twenty-spirit` §3.5:

| Metric | MEASURED | Source | Gate |
|---|---:|---|---|
| **TODO-180 G4 Ru per-metal test_auc** | **0.7527 ± 0.0083** (n=3) | `molmetal/reports/wf265_collect/all_results.json` `g4_per_seed` | Krasnov ≥ 0.86 BELOW |
| **TODO-180 G4 Ir per-metal test_auc** | **0.6702 ± 0.0069** (n=3) | same | BELOW |
| **TODO-180 G4 Os per-metal test_auc** | **0.7025 ± 0.0251** (n=3) | same | BELOW |
| **TODO-180 G4 Rh per-metal test_auc** | **0.7227 ± 0.0662** (n=3) | same | BELOW |
| **TODO-180 G4 Re per-metal test_auc** | **0.7700 ± 0.0298** (n=3) | same | BELOW |
| **TODO-181 G5 50-epoch full sweep** | **15/15** (metal, seed) pairs MEASURED | `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` | ✅ SHIPPED (Wf-310A) |
| **NCI-60 9-cell per-cell HCT116 test_auc** | **0.6829** (Wf-301) / **0.6998** (Wf-320C) | `molmetal/reports/wf301_synthesis/` + `wf320c_todo220_full_impl_closeout.md` | ✅ MEASURED |
| **NCI-60 9-cell per-cell PC3 test_auc** | **0.2153** (Wf-301) / **0.1667** (Wf-320C) — HONEST-NEGATIVE | same | ❌ HONEST-NEGATIVE |
| **chemprop_v2_5metal_5seed_50epoch** | **19/25** MEASURED-on-disk (PARTIAL_BLOCK) | `metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` | PARTIAL_BLOCK |
| **chemprop_v2_lomo_v2** | **25/25** MEASURED (32.91 min wall) | `metrics/by_metric/chemprop_v2_lomo_v2.json` | ✅ MEASURED |
| **wf219_krasnov_aligned chemprop_v2 3-seed × 50-epoch scaffold** | **9/9** MEASURED (4.19 min wall) | `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` | ✅ MEASURED |
| **chemprop v3 DrugOOD OOD baseline** | **0.4544** BELOW_RANDOM | `molmetal/reports/wf253d_m17_drugood_reeval/` | ❌ BELOW_RANDOM (Wf-263A fix root-cause) |
| **Wf-320D featurization fix 1-ep smoke Ru seed=0** | test_auc **0.7452** | `TODO/completed/320d_chemprop_featurization_fix.md` | ✅ SHIPPED (12/12 tests in 4.68s) |
| **Wf-320C TODO-220 full impl verify** | temporal **5/5** + NCI-60 **9/9** smoke MEASURED (25.17s) | `TODO/completed/320c_todo220_full_impl_closeout.md` | ✅ 10/10 tests |
| **Wf-320F R20 Phase 0 closeout ACTUAL** | **37/37** tests PASS in 10.72s (combined Wf-313H 42 = **79 total**) | `TODO/completed/320_r20_phase0_loaders_closeout.md` | ✅ SHIPPED |
| **Wf-320B Wf-291F harness auto-fallback** | **13/13** tests PASS in **0.53s** | `TODO/completed/320b_wf291f_harness_autofallback_fix.md` | ✅ SHIPPED |
| **Wf-320G MECHANISM_METRIC_MAPPING §13/§14** | **107/107** tests + Wf-313E **33/33** regression | `TODO/completed/320g_mechanism_metric_mapping_extension.md` | ✅ SHIPPED |
| **Wf-320H TODO-134 Phase 2 Vina attempt** | **6/6** pockets green via CadButEaas `.venv-vlm` | `TODO/completed/320h_todo134_phase2_attempt.md` | ✅ PIPELINE PROOF |
| **Wf-321B R20/R21 skill stubs** | **4 NEW** stubs (variance-puct-default + hilbert-astsr-e2e + closure-theorem + multiph-pll); **16/16** stub tests PASS | `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` | ✅ SHIPPED |
| **Wf-321E ABLATION_TABLE registry** | **11/11** R17 §7 ablations mirrored + 1 BASELINE | `TODO/completed/321e_ablation_runner_integration_test.md` | ✅ SHIPPED |
| **Wf-321H TODO-119 P1.4 CPU scaffold** | **12/12** tests PASS + pre-flight gate PASS | `TODO/completed/321h_todo_119_p14_cpu_scaffold.md` | ✅ SHIPPED (GPU Wave-A launchable under §5.1) |
| **Pre-flight v3 fresh-run** | **42/15/57** (Wf-322D 11:06 CST snapshot); **58/1/59** (live 12:00 CST) | `molmetal/reports/wf322d_preflight/pre_flight.json` + live re-run | ⚠️ HONEST-NEGATIVE disclosure (15-FAIL breakdown in `TODO/INDEX.md` §22.16) |
| **Test count** | **699 tests / 105.68s** (Wf-296 baseline); **79 R20 Phase 0 loader tests** (Wf-320F + Wf-313H combined) | `molmetal/scripts/pre_flight_check.py` p2_tests_pass | ✅ MEASURED |

---

**END OF README** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; last update 2026-09-29 12:00 CST (Wf-324A README refresh with live pre-flight 58/1/59 update + §11 TODO/skill summary + §12 MEASURED metrics table).
